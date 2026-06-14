"""Unified eval executor for FitText runs.

Single entry point ``execute_run(cfg, ...)`` dispatches by benchmark name to a
per-benchmark adapter, which iterates queries in parallel and routes each
query to the appropriate FitText variant pipeline.

Routes:
  - ``cfg.benchmark.name == 'toolret'``         → :func:`.adapters.toolret.run_toolret`
  - ``cfg.benchmark.name == 'stabletoolbench'`` → :func:`.adapters.stabletoolbench.run_stb`

The four FitText variants map to a single parameterized config (§5.5):
  - ``single_pass``: ``N=1, G=1, selection=none``  → single_pass dispatch
  - ``multi_turn``:  ``N=1, G>1, selection=fitness`` → dbd dispatch
  - ``scattershot``: ``N>1, G=1, selection=none`` → scattershot dispatch
  - ``memetic``:     ``N>1, G>1, selection=fitness, memory_penalty>0`` → memetic dispatch
  - ``just_query``:  zero-retrieval baseline (just_query strategy, no LLM call)

Each LLM call inside the variant pipeline records one manifest entry via the
injected ``ManifestWriter`` and increments the ``BudgetGuard`` counter.

Resumability (§3.4): per-query ``<out_dir>/queries/{qid}.done.json`` sentinel.
Re-launching the same run skips completed queries.

Tenacity is the ONLY retry layer (Wave 1 fix H6); the executor does not
add a second retry loop on top.

Parallelism (§3.1):
  - Cross-query: ``ProcessPoolExecutor(max_workers=min(num_queries, cores//2))``
    (process pool avoids GIL on CPU-bound embedding encode).
  - Within-query: variant pipelines already use ``ThreadPoolExecutor`` for
    parallel LLM/retrieval calls inside population-based strategies.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from toolbench.observability import (
    BudgetGuard,
    ManifestWriter,
    read_manifest,
)
from toolbench.observability.budget_guard import BudgetExceeded

# BeliefTracer factory is opaque to the executor; adapters create per-query
# tracers via the injected callable.
from toolbench.runner.schema import RunConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public report dataclass
# ---------------------------------------------------------------------------


@dataclass
class ExecutionReport:
    """Aggregated outcome of one :func:`execute_run` call.

    Attributes:
        n_queries: Total queries attempted (including skipped-as-already-done).
        n_successful: Queries that produced a usable result.
        n_failed: Queries that errored or had no result.
        total_cost_usd: Sum of ``cost_usd`` across all manifest entries.
        total_input_tokens: Sum of non-cached prompt tokens.
        total_cached_input_tokens: Sum of cached prompt tokens.
        total_output_tokens: Sum of completion tokens.
        wall_clock_s: Wall-clock time of the executor run in seconds.
        query_results: List of per-query result dicts (qid, metrics, etc.).
        budget_aborted: True if the run was halted by ``BudgetGuard``.
    """

    n_queries: int = 0
    n_successful: int = 0
    n_failed: int = 0
    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_cached_input_tokens: int = 0
    total_output_tokens: int = 0
    wall_clock_s: float = 0.0
    query_results: list[dict[str, Any]] = field(default_factory=list)
    budget_aborted: bool = False


# ---------------------------------------------------------------------------
# Manifest aggregation
# ---------------------------------------------------------------------------


def aggregate_manifest(manifest_path: Path) -> dict[str, Any]:
    """Sum cost and token totals from a JSONL manifest.

    Args:
        manifest_path: Path to ``manifest.jsonl``.

    Returns:
        Dict with keys ``total_cost_usd``, ``total_input_tokens``,
        ``total_cached_input_tokens``, ``total_output_tokens``, ``n_entries``.
        Returns zeros if the file is empty or missing.
    """
    totals = {
        "total_cost_usd": 0.0,
        "total_input_tokens": 0,
        "total_cached_input_tokens": 0,
        "total_output_tokens": 0,
        "n_entries": 0,
    }
    if not Path(manifest_path).exists():
        return totals
    try:
        entries = read_manifest(manifest_path)
    except (ValueError, FileNotFoundError) as exc:
        log.warning("aggregate_manifest: failed to read %s: %s", manifest_path, exc)
        return totals

    for entry in entries:
        totals["total_cost_usd"] += float(entry.cost_usd)
        totals["total_input_tokens"] += int(entry.input_tokens)
        totals["total_cached_input_tokens"] += int(entry.cached_input_tokens)
        totals["total_output_tokens"] += int(entry.output_tokens)
        totals["n_entries"] += 1
    return totals


# ---------------------------------------------------------------------------
# Resumability sentinel helpers
# ---------------------------------------------------------------------------


def _queries_dir(run_dir: Path) -> Path:
    """Per-query sentinel directory."""
    d = Path(run_dir) / "queries"
    d.mkdir(parents=True, exist_ok=True)
    return d


def is_query_done(run_dir: Path, qid: str) -> bool:
    """Return True if a ``{qid}.done.json`` sentinel exists.

    Args:
        run_dir: Run directory containing the ``queries/`` subdir.
        qid: Query identifier.
    """
    safe_qid = qid.replace("/", "_").replace(":", "_")
    return (_queries_dir(run_dir) / f"{safe_qid}.done.json").exists()


def mark_query_done(run_dir: Path, qid: str, payload: dict[str, Any]) -> None:
    """Atomically write a ``{qid}.done.json`` sentinel.

    Args:
        run_dir: Run directory.
        qid: Query identifier.
        payload: Dict to JSON-dump (per-query result metadata).
    """
    import json
    import tempfile

    safe_qid = qid.replace("/", "_").replace(":", "_")
    out = _queries_dir(run_dir) / f"{safe_qid}.done.json"
    # Atomic write — tempfile in same dir, then os.replace
    fd, tmp_path = tempfile.mkstemp(prefix=safe_qid + ".", suffix=".tmp", dir=str(_queries_dir(run_dir)))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        os.replace(tmp_path, out)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def execute_run(
    cfg: RunConfig,
    *,
    run_dir: Path,
    mw: ManifestWriter,
    budget: BudgetGuard,
    client: Any,
    tracer_factory: Optional[Callable[[str], Any]] = None,
    git_commit: str = "unknown",
) -> ExecutionReport:
    """Dispatch to the correct benchmark adapter and aggregate results.

    Routes by ``cfg.benchmark.name``:

      - ``'toolret'``         → :mod:`.adapters.toolret`
      - ``'stabletoolbench'`` → :mod:`.adapters.stabletoolbench`

    Args:
        cfg: Fully resolved :class:`RunConfig`.
        run_dir: Output directory for this run.
        mw: Open :class:`ManifestWriter` instance.
        budget: :class:`BudgetGuard` with ``max_cost_usd`` already set.
        client: :class:`ModelClient` used by all LLM calls.
        tracer_factory: Optional callable ``(qid) -> BeliefTracer`` that
            opens and returns a per-query tracer. ``None`` disables belief
            tracing entirely. Required for instrumented memetic runs.
        git_commit: Git SHA for manifest entries.

    Returns:
        :class:`ExecutionReport` with aggregated cost/query stats.

    Raises:
        BudgetExceeded: If the cumulative cost crosses ``cfg.budget.max_cost_usd``.
            Adapters should catch and return cleanly; this propagates only on
            outer-loop violations.
    """
    t0 = time.monotonic()
    benchmark = cfg.benchmark.name

    log.info(
        "execute_run: benchmark=%s variant=%s n_queries_per_split=%s",
        benchmark,
        cfg.fittext.variant,
        cfg.benchmark.n_queries_per_split,
    )

    if benchmark == "toolret":
        from toolbench.runner.adapters.toolret import run_toolret

        adapter = run_toolret
    elif benchmark == "stabletoolbench":
        from toolbench.runner.adapters.stabletoolbench import run_stb

        adapter = run_stb
    else:
        raise NotImplementedError(f"Unknown benchmark: {benchmark!r}")

    report = ExecutionReport()
    try:
        await adapter(
            cfg=cfg,
            run_dir=Path(run_dir),
            mw=mw,
            budget=budget,
            client=client,
            tracer_factory=tracer_factory,
            git_commit=git_commit,
            report=report,
        )
    except BudgetExceeded as exc:
        log.warning("execute_run: budget exceeded mid-run: %s", exc)
        report.budget_aborted = True

    # Aggregate from manifest (single source of truth — never from logs)
    manifest_path = Path(run_dir) / "manifest.jsonl"
    totals = aggregate_manifest(manifest_path)
    report.total_cost_usd = totals["total_cost_usd"]
    report.total_input_tokens = totals["total_input_tokens"]
    report.total_cached_input_tokens = totals["total_cached_input_tokens"]
    report.total_output_tokens = totals["total_output_tokens"]

    report.wall_clock_s = time.monotonic() - t0
    log.info(
        "execute_run: done n_queries=%d n_successful=%d n_failed=%d cost=$%.4f wall=%.1fs",
        report.n_queries,
        report.n_successful,
        report.n_failed,
        report.total_cost_usd,
        report.wall_clock_s,
    )
    return report
