"""ToolRet eval adapter.

Wraps the existing ``Toolret/eval_toolret.py`` query loop and routes each
query through ``Toolret/strategy/strategies.select_and_run_strategy`` with
the appropriate ``strategy_wrapper.strategy`` value derived from
``cfg.fittext.variant``.

Variant mapping (Toolret has no in-tree Memetic strategy):

  - ``single_pass`` → ``strategy='single_pass'``
  - ``multi_turn``  → ``strategy='dbd'``       (Description-Based Dynamic refinement)
  - ``scattershot`` → ``strategy='scattershot'``
  - ``memetic``     → ``strategy='dbd'`` with refinement on (graceful fallback —
                      Toolret has no Memetic; the StableToolBench adapter is the
                      correct home for true memetic ablations)
  - ``just_query``  → ``strategy='just_query'``  (zero-retrieval baseline)

Parallelism: queries within a split are iterated with ``ProcessPoolExecutor``
(``max_workers = min(num_queries, cores // 2)``) to avoid GIL contention on
the embedding encode step.  Within-query LLM calls go through the existing
``ChatGPTFunction`` plumbing, which now records via the active manifest
writer (see :func:`toolbench.runner.telemetry.install_chat_completion_hook`).

Resumability: per-query ``<run_dir>/queries/{qid}.done.json`` sentinel.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from toolbench.runner.executor import (
    ExecutionReport,
    is_query_done,
    mark_query_done,
)
from toolbench.runner.schema import RunConfig
from toolbench.runner.telemetry import (
    install_chat_completion_hook,
    reset_call_context,
    set_call_context,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Variant → Toolret strategy name
# ---------------------------------------------------------------------------


def _variant_to_strategy(variant: str) -> str:
    """Map FitText variant → ``strategy_wrapper.strategy`` value.

    Args:
        variant: One of ``single_pass``, ``multi_turn``, ``scattershot``,
            ``memetic``, ``just_query``.

    Returns:
        Strategy string accepted by ``Toolret.strategy.strategies.select_and_run_strategy``.

    Raises:
        NotImplementedError: For unknown variants.
    """
    mapping = {
        "single_pass": "single_pass",
        "multi_turn": "dbd",
        "scattershot": "scattershot",
        "memetic": "dbd",
        "just_query": "just_query",
    }
    if variant not in mapping:
        raise NotImplementedError(f"Unknown variant {variant!r} for toolret adapter.")
    return mapping[variant]


# ---------------------------------------------------------------------------
# Toolret query iteration
# ---------------------------------------------------------------------------


@dataclass
class _ToolretQuery:
    """One ToolRet query with everything needed to run + score it."""

    qid: str
    query: str
    gt_tools: list[dict[str, Any]]
    split: str


def _iter_split_queries(
    split: str,
    n_queries: int | None,
    shard: int,
    total_shards: int,
    queries_root: str | None = None,
) -> list[_ToolretQuery]:
    """Load and shard queries for one ToolRet split.

    Args:
        split: Domain split.  Production splits: ``code`` / ``customized`` /
            ``web`` / ``toolbench``.  Test fixtures may inject ``mini``.
        n_queries: Max queries to take (post-shard).  ``None`` = all.
        shard: 0-indexed shard for cross-host parallelism.
        total_shards: Total number of shards.
        queries_root: Optional path to a local synthetic queries directory
            (used by the E2E test fixture).  If set, queries are loaded from
            ``<queries_root>/<split>/queries.json`` and the HuggingFace path
            is bypassed entirely.  This keeps the test offline and avoids
            the production HF dataset download.

    Returns:
        List of :class:`_ToolretQuery`.
    """
    # --- Local-fixture path (offline E2E) ----------------------------------
    if queries_root is not None:
        from pathlib import Path as _Path
        qfile = _Path(queries_root) / split / "queries.json"
        if not qfile.exists():
            log.warning(
                "toolret: queries_root override set but %s missing — no queries",
                qfile,
            )
            return []
        out: list[_ToolretQuery] = []
        with qfile.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                qid = str(sample["id"])
                if total_shards > 1 and (hash(qid) % total_shards) != shard:
                    continue
                try:
                    gt = json.loads(sample["labels"]) if isinstance(sample.get("labels"), str) else sample.get("labels", [])
                except (TypeError, json.JSONDecodeError):
                    gt = []
                out.append(
                    _ToolretQuery(
                        qid=qid,
                        query=str(sample["query"]),
                        gt_tools=gt,
                        split=split,
                    )
                )
                if n_queries is not None and len(out) >= n_queries:
                    return out
        return out

    # --- Production HF path ------------------------------------------------
    # Defer the heavy HuggingFace import until we know we need it.
    from datasets import load_dataset  # noqa: WPS433 — lazy-import on purpose

    # Toolret organises queries by per-domain subset names — load all subsets
    # in the requested domain and concatenate.  See Toolret/eval_toolret.py.
    from Toolret.eval_toolret import dataset_categories  # type: ignore

    subset_names = dataset_categories.get(split, [])
    if not subset_names:
        log.warning("Unknown ToolRet split %r — no queries loaded", split)
        return []

    out: list[_ToolretQuery] = []
    for subset in subset_names:
        try:
            sub_ds = load_dataset("mangopy/ToolRet-Queries", subset)["queries"]
        except Exception as exc:
            log.warning("Failed to load subset %r in split %r: %s", subset, split, exc)
            continue
        for sample in sub_ds:
            qid = str(sample["id"])
            # Shard filter: stable hash-based partition
            if total_shards > 1 and (hash(qid) % total_shards) != shard:
                continue
            try:
                gt = json.loads(sample["labels"])
            except (TypeError, json.JSONDecodeError):
                gt = []
            out.append(
                _ToolretQuery(
                    qid=qid,
                    query=str(sample["query"]),
                    gt_tools=gt,
                    split=split,
                )
            )
            if n_queries is not None and len(out) >= n_queries:
                return out
    return out


# ---------------------------------------------------------------------------
# Per-query worker
# ---------------------------------------------------------------------------


def _run_one_query(
    *,
    q: _ToolretQuery,
    cfg_dict: dict[str, Any],
    run_dir_str: str,
    plan_model: str,
    refine_model: str,
    base_url_plan: str | None,
    base_url_refine: str | None,
) -> dict[str, Any]:
    """Run one ToolRet query and write its result + sentinel.

    Designed to be called from a thread or process worker.  Sets the
    per-query call-context (qid, variant) before invoking the strategy so
    the manifest hook records correct provenance.

    Args:
        q: The query to evaluate.
        cfg_dict: ``RunConfig.model_dump()`` of the active config — keeps the
            worker dispatch fork-safe (no Pydantic objects across process
            boundaries).
        run_dir_str: Run output directory as a string (Path is not always
            picklable cleanly across forks).
        plan_model: Planner LLM identifier (cfg.model.name).
        refine_model: Refiner LLM identifier (currently same as plan_model).
        base_url_plan: Optional base_url for the planner (vLLM).
        base_url_refine: Optional base_url for the refiner (vLLM).

    Returns:
        Dict with ``qid``, ``ok`` (bool), and ``retrieved_tool_ids``/
        ``retrieved_tool_scores`` on success or ``error`` on failure.
    """
    # Late imports keep the executor lightweight and avoid loading torch in
    # the parent process.
    from Toolret.retriever import ToolRetriever  # type: ignore
    from Toolret.strategy.strategies import (  # type: ignore
        select_and_run_strategy,
        strategy_wrapper,
    )

    run_dir = Path(run_dir_str)
    variant = cfg_dict["fittext"]["variant"]
    strategy = _variant_to_strategy(variant)

    if is_query_done(run_dir, q.qid):
        return {"qid": q.qid, "ok": True, "skipped": True}

    # Re-install the telemetry hook inside the worker (no-op if already installed).
    install_chat_completion_hook()

    # Set per-query call context for manifest recording.
    token = set_call_context(
        qid=f"toolret:{q.split}:{q.qid}",
        operation="dfsdt_node",
        variant=variant,
        generation=0,
    )

    try:
        # ToolRetriever is loaded once per worker via a module-level cache.
        retriever = _get_or_build_retriever(
            corpus_path=os.path.join(
                cfg_dict["extra"].get("toolret_corpus_root", "./data/retrieval/Toolret"),
                q.split,
                "des_corpus.json",
            ),
            embedding_model_path=cfg_dict["embedder"]["name"],
        )

        wrapper = strategy_wrapper(
            strategy=strategy,
            retriever=retriever,
            example_num=int(cfg_dict["extra"].get("toolret_example_num", 15)),
            retrieved_api_nums=int(cfg_dict["fittext"]["top_k_retrieval"]),
            dbd_refine_turns=int(cfg_dict["fittext"]["generations"]),
            scattershot_size=int(cfg_dict["fittext"]["population_size"]),
            refinement=variant in ("multi_turn", "memetic"),
            api_key=os.getenv("OPENAI_API_KEY", ""),
        )
        # Split-temperatures: propagate ``cfg.model.evolution_temperature`` onto
        # the wrapper (mirrors the STB adapter). Toolret has no in-tree memetic
        # strategy today (memetic → dbd fallback per ``_variant_to_strategy``),
        # so this is a no-op for current Toolret runs but keeps the wrapper
        # surface consistent across adapters for future memetic Toolret work.
        wrapper.evolution_temperature = cfg_dict["model"].get("evolution_temperature")

        # Per-query detailed-result path — keeps strategies' file writes
        # contained within run_dir/details/.
        details_dir = run_dir / "details" / q.split / strategy
        details_dir.mkdir(parents=True, exist_ok=True)
        detailed_path = str(details_dir / f"{q.qid}.jsonl")

        tool_ids, tool_descs, tool_scores = select_and_run_strategy(
            q.query,
            wrapper,
            plan_model,
            refine_model,
            detailed_path,
            base_url_plan,
            base_url_refine,
        )

        result = {
            "qid": q.qid,
            "split": q.split,
            "ok": True,
            "retrieved_tool_ids": list(tool_ids),
            "retrieved_tool_scores": [float(s) for s in tool_scores],
            "gt_tools": q.gt_tools,
        }
        mark_query_done(run_dir, f"toolret:{q.split}:{q.qid}", result)
        return result
    except Exception as exc:
        log.warning("toolret query %s failed: %s", q.qid, exc)
        return {
            "qid": q.qid,
            "split": q.split,
            "ok": False,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        reset_call_context(token)


# ---------------------------------------------------------------------------
# Per-worker retriever cache
# ---------------------------------------------------------------------------


_retriever_cache: dict[tuple[str, str], Any] = {}

import threading as _threading  # late-binding to keep top-of-file imports tidy
_retriever_cache_lock = _threading.Lock()


def _get_or_build_retriever(*, corpus_path: str, embedding_model_path: str) -> Any:
    """Load (or reuse) a :class:`ToolRetriever` for this worker.

    Thread-safe.  Concurrent worker threads (under the adapter's
    ``ThreadPoolExecutor``) all share this module-level cache; without the
    lock they would race on first construction and hit a SentenceTransformer
    meta-tensor weight-loading bug.  Embeddings are cached on disk by
    ToolRet; the lock-and-cache pattern also avoids the redundant
    HF download / GPU transfer when the same worker handles many queries.

    Args:
        corpus_path: Path to ``des_corpus.json`` for the active split.
        embedding_model_path: HF model name / local path.

    Returns:
        A ready-to-use :class:`ToolRetriever`.
    """
    key = (corpus_path, embedding_model_path)
    # Fast path — already built.
    cached = _retriever_cache.get(key)
    if cached is not None:
        return cached
    # Slow path — build under the lock so only one thread instantiates.
    with _retriever_cache_lock:
        cached = _retriever_cache.get(key)
        if cached is not None:
            return cached
        # Late import to keep parent process light.
        from Toolret.retriever import ToolRetriever  # type: ignore

        retriever = ToolRetriever(corpus_path=corpus_path, model_path=embedding_model_path)
        _retriever_cache[key] = retriever
        return retriever


# ---------------------------------------------------------------------------
# Adapter entry point
# ---------------------------------------------------------------------------


async def run_toolret(
    *,
    cfg: RunConfig,
    run_dir: Path,
    mw: Any,
    budget: Any,
    client: Any,
    tracer_factory: Callable[[str], Any] | None,
    git_commit: str,
    report: ExecutionReport,
) -> None:
    """Run the configured ToolRet variant across requested splits.

    Args:
        cfg: Resolved :class:`RunConfig`.
        run_dir: Output directory for this run.
        mw: Active :class:`ManifestWriter`.
        budget: Active :class:`BudgetGuard`.
        client: Unused at this layer; the strategies use ``ChatGPTFunction``
            internally (which routes through the same client factory).
        tracer_factory: Belief tracer factory (unused for ToolRet — no
            in-tree Memetic implementation).
        git_commit: Git SHA for manifest entries.
        report: :class:`ExecutionReport` to mutate in-place.
    """
    # Inject the manifest writer + budget guard into the telemetry sinks.
    from toolbench.runner.telemetry import attach_sinks

    install_chat_completion_hook()
    attach_sinks(manifest_writer=mw, budget_guard=budget)

    # Stamp run-level provenance into the call context (workers re-stamp qid).
    set_call_context(
        run_id=cfg.run_id or "unknown_run",
        git_commit=git_commit,
        config_hash=cfg.config_hash(),
        variant=cfg.fittext.variant,
    )

    cfg_dict = cfg.model_dump()
    plan_model = cfg.model.name
    refine_model = cfg.model.name
    base_url_plan = cfg.extra.get("base_url_plan")
    base_url_refine = cfg.extra.get("base_url_refine")

    # Collect queries across all requested splits.
    # ``toolret_queries_root`` (cfg.extra) lets the E2E test fixture redirect
    # query loading to a local synthetic file and bypass HuggingFace entirely.
    queries_root_override = cfg.extra.get("toolret_queries_root") if isinstance(cfg.extra, dict) else None
    queries: list[_ToolretQuery] = []
    for split in cfg.benchmark.splits:
        per_split = _iter_split_queries(
            split=split,
            n_queries=cfg.benchmark.n_queries_per_split,
            shard=cfg.infra.shard,
            total_shards=cfg.infra.total_shards,
            queries_root=queries_root_override,
        )
        queries.extend(per_split)
    log.info("toolret: loaded %d queries across %d splits", len(queries), len(cfg.benchmark.splits))
    report.n_queries = len(queries)

    # Pre-build the retriever in the parent thread before dispatching
    # workers.  Avoids a concurrent ``SentenceTransformer`` init race that
    # otherwise hits the meta-tensor weight-loading bug under
    # ``ThreadPoolExecutor`` (workers all share the same module cache).
    corpus_root_for_prebuild = (
        cfg.extra.get("toolret_corpus_root", "./data/retrieval/Toolret")
        if isinstance(cfg.extra, dict)
        else "./data/retrieval/Toolret"
    )
    seen_splits: set[str] = set()
    for q in queries:
        if q.split in seen_splits:
            continue
        seen_splits.add(q.split)
        corpus_path_pre = os.path.join(corpus_root_for_prebuild, q.split, "des_corpus.json")
        if not os.path.exists(corpus_path_pre):
            log.warning("toolret: corpus missing for split %r at %s — skipping prebuild", q.split, corpus_path_pre)
            continue
        try:
            _get_or_build_retriever(
                corpus_path=corpus_path_pre,
                embedding_model_path=cfg.embedder.name,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("toolret: retriever prebuild failed for split %r: %s", q.split, exc)

    # Parallel dispatch — threads (not processes) so the in-memory
    # ManifestWriter + BudgetGuard sinks installed above are visible.
    # The CPU-bound embedding encode runs in the worker, but the
    # ToolRetriever instance is shared (loaded once via the module cache).
    max_workers = min(len(queries) or 1, max(1, (os.cpu_count() or 4) // 2))
    log.info("toolret: dispatching with max_workers=%d", max_workers)

    loop = asyncio.get_running_loop()
    pool = ThreadPoolExecutor(max_workers=max_workers)
    futures = []
    try:
        for q in queries:
            futures.append(
                loop.run_in_executor(
                    pool,
                    lambda q=q: _run_one_query(
                        q=q,
                        cfg_dict=cfg_dict,
                        run_dir_str=str(run_dir),
                        plan_model=plan_model,
                        refine_model=refine_model,
                        base_url_plan=base_url_plan,
                        base_url_refine=base_url_refine,
                    ),
                )
            )
        results = await asyncio.gather(*futures, return_exceptions=True)
    finally:
        pool.shutdown(wait=True)

    for r in results:
        if isinstance(r, Exception):
            report.n_failed += 1
            report.query_results.append({"ok": False, "error": repr(r)})
            continue
        report.query_results.append(r)
        if r.get("ok"):
            report.n_successful += 1
        else:
            report.n_failed += 1

    # Persist a per-split results dump for downstream eval (NDCG calc lives
    # in scripts/, not in the executor hot path).
    summary_path = Path(run_dir) / "toolret_results.jsonl"
    with open(summary_path, "w", encoding="utf-8") as fh:
        for r in report.query_results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("toolret: results dumped to %s", summary_path)
