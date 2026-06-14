"""
Cost / token / latency aggregator for manifest JSONL files.

Reads one or more manifest JSONL files produced by ``ManifestWriter`` and
produces four CSV outputs for report tables and figures:

``cost_table.csv``
    Per ``(model, variant, benchmark)`` cell: total cost, token breakdown,
    latency percentiles, mean LLM calls per query, cache hit rate.

``pareto_data.csv``
    $/NDCG-point and $/pass-rate-point per cell — input for the Pareto
    frontier figure.

``cache_hit.csv``
    Cache hit rate breakdown per ``(model, variant, benchmark)``.

``latency.csv``
    Full p50 / p90 / p95 / p99 latency distribution per cell.

CLI::

    python -m toolbench.observability.aggregator \\
        --manifest-dir <dir> \\
        --out <output_dir> \\
        [--metrics-json <path_to_eval_metrics.json>]

The ``--metrics-json`` file (optional) maps ``run_id`` → eval metrics so the
aggregator can compute $/NDCG-point.  Without it, the Pareto CSV is written
with NaN metric columns.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .manifest import ManifestEntry, read_manifest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cell key
# ---------------------------------------------------------------------------

# A "cell" groups entries that belong to the same experimental condition.
# ``benchmark`` is derived from ``qid`` prefix (e.g. "toolret", "stb").
CellKey = tuple[str, str, str]  # (model, variant, benchmark)

UNKNOWN_BENCHMARK = "unknown"


def _benchmark_from_qid(qid: str) -> str:
    """Derive benchmark name from the query-id prefix.

    Convention:
        ``toolret:code:0042``  → ``toolret``
        ``stb:g2:0007``        → ``stb``
        anything else          → ``unknown``
    """
    if not qid:
        return UNKNOWN_BENCHMARK
    return qid.split(":")[0].lower()


# ---------------------------------------------------------------------------
# Cell accumulator
# ---------------------------------------------------------------------------

class _CellAccumulator:
    """Accumulates per-entry statistics for one (model, variant, benchmark) cell."""

    def __init__(self) -> None:
        self.total_cost: float = 0.0
        self.total_input_tokens: int = 0
        self.total_cached_tokens: int = 0
        self.total_output_tokens: int = 0
        self.latencies_ms: list[float] = []
        self.queries: set[str] = set()
        self.call_counts_per_query: dict[str, int] = defaultdict(int)
        self.n_entries: int = 0

    def add(self, entry: ManifestEntry) -> None:
        """Incorporate one manifest entry."""
        self.total_cost += entry.cost_usd
        self.total_input_tokens += entry.input_tokens
        self.total_cached_tokens += entry.cached_input_tokens
        self.total_output_tokens += entry.output_tokens
        if entry.latency_ms > 0:
            self.latencies_ms.append(entry.latency_ms)
        self.queries.add(entry.qid)
        self.call_counts_per_query[entry.qid] += 1
        self.n_entries += 1

    # -- Derived metrics ---------------------------------------------------

    @property
    def n_queries(self) -> int:
        return len(self.queries)

    @property
    def mean_calls_per_query(self) -> float:
        if not self.call_counts_per_query:
            return 0.0
        return statistics.mean(self.call_counts_per_query.values())

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of prompt tokens served from cache."""
        total_prompt = self.total_input_tokens + self.total_cached_tokens
        if total_prompt == 0:
            return 0.0
        return self.total_cached_tokens / total_prompt

    @property
    def cost_per_query(self) -> float:
        if self.n_queries == 0:
            return 0.0
        return self.total_cost / self.n_queries

    def latency_percentile(self, p: float) -> float:
        """Return the *p*-th percentile latency in ms (0 ≤ p ≤ 100)."""
        if not self.latencies_ms:
            return 0.0
        sorted_lats = sorted(self.latencies_ms)
        k = (p / 100) * (len(sorted_lats) - 1)
        lo = int(k)
        hi = min(lo + 1, len(sorted_lats) - 1)
        frac = k - lo
        return sorted_lats[lo] * (1 - frac) + sorted_lats[hi] * frac


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(
    manifest_paths: list[Path],
    metrics_by_run: dict[str, dict[str, float]] | None = None,
) -> tuple[
    dict[CellKey, _CellAccumulator],
    dict[CellKey, dict[str, float]],  # pareto extras (ndcg, pass_rate per cell)
]:
    """Aggregate all manifest files into per-cell accumulators.

    Args:
        manifest_paths: One or more JSONL manifest file paths.
        metrics_by_run: Optional dict mapping ``run_id`` →
            ``{ndcg: float, pass_rate: float, ...}`` from the eval result.
            Enables $/NDCG-point computation.

    Returns:
        Tuple of:
        - ``cells``: mapping ``(model, variant, benchmark)`` → ``_CellAccumulator``
        - ``pareto_extras``: mapping same keys → ``{ndcg, pass_rate}`` when available
    """
    cells: dict[CellKey, _CellAccumulator] = defaultdict(_CellAccumulator)
    run_ids_per_cell: dict[CellKey, set[str]] = defaultdict(set)

    for path in manifest_paths:
        logger.info("Reading manifest: %s", path)
        entries = read_manifest(path)
        for entry in entries:
            bm = _benchmark_from_qid(entry.qid)
            key: CellKey = (entry.model, entry.variant, bm)
            cells[key].add(entry)
            run_ids_per_cell[key].add(entry.run_id)

    # Build pareto extras from the metrics map
    pareto_extras: dict[CellKey, dict[str, float]] = {}
    if metrics_by_run:
        for key, run_ids in run_ids_per_cell.items():
            ndcg_vals, pr_vals = [], []
            for rid in run_ids:
                m = metrics_by_run.get(rid, {})
                if "ndcg" in m:
                    ndcg_vals.append(m["ndcg"])
                if "pass_rate" in m:
                    pr_vals.append(m["pass_rate"])
            pareto_extras[key] = {
                "ndcg": statistics.mean(ndcg_vals) if ndcg_vals else float("nan"),
                "pass_rate": statistics.mean(pr_vals) if pr_vals else float("nan"),
            }

    return cells, pareto_extras


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def write_cost_table(
    cells: dict[CellKey, _CellAccumulator],
    out_path: Path,
) -> None:
    """Write ``cost_table.csv`` — the report-ready cost table.

    Columns:
        model, variant, benchmark, n_queries, n_api_calls,
        mean_calls_per_query, total_cost_usd, cost_per_query_usd,
        total_input_tokens, total_cached_tokens, total_output_tokens,
        cache_hit_rate, p50_latency_ms, p95_latency_ms
    """
    fieldnames = [
        "model", "variant", "benchmark",
        "n_queries", "n_api_calls", "mean_calls_per_query",
        "total_cost_usd", "cost_per_query_usd",
        "total_input_tokens", "total_cached_tokens", "total_output_tokens",
        "cache_hit_rate", "p50_latency_ms", "p95_latency_ms",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for (model, variant, benchmark), acc in sorted(cells.items()):
            writer.writerow({
                "model": model,
                "variant": variant,
                "benchmark": benchmark,
                "n_queries": acc.n_queries,
                "n_api_calls": acc.n_entries,
                "mean_calls_per_query": f"{acc.mean_calls_per_query:.2f}",
                "total_cost_usd": f"{acc.total_cost:.6f}",
                "cost_per_query_usd": f"{acc.cost_per_query:.6f}",
                "total_input_tokens": acc.total_input_tokens,
                "total_cached_tokens": acc.total_cached_tokens,
                "total_output_tokens": acc.total_output_tokens,
                "cache_hit_rate": f"{acc.cache_hit_rate:.4f}",
                "p50_latency_ms": f"{acc.latency_percentile(50):.1f}",
                "p95_latency_ms": f"{acc.latency_percentile(95):.1f}",
            })
    logger.info("Wrote cost table: %s (%d rows)", out_path, len(cells))


def write_pareto_data(
    cells: dict[CellKey, _CellAccumulator],
    pareto_extras: dict[CellKey, dict[str, float]],
    out_path: Path,
) -> None:
    """Write ``pareto_data.csv`` — $/NDCG-point and $/pass-rate-point.

    When eval metrics are unavailable for a cell, ``ndcg`` and ``pass_rate``
    columns are written as empty strings.

    Columns:
        model, variant, benchmark,
        cost_per_query_usd, ndcg, pass_rate,
        cost_per_ndcg_point, cost_per_pass_rate_point
    """
    fieldnames = [
        "model", "variant", "benchmark",
        "cost_per_query_usd", "ndcg", "pass_rate",
        "cost_per_ndcg_point", "cost_per_pass_rate_point",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for (model, variant, benchmark), acc in sorted(cells.items()):
            extras = pareto_extras.get((model, variant, benchmark), {})
            ndcg = extras.get("ndcg", float("nan"))
            pass_rate = extras.get("pass_rate", float("nan"))
            cpq = acc.cost_per_query

            def _ratio(cpq: float, metric: float) -> str:
                if cpq == 0 or metric != metric or metric == 0:  # nan check
                    return ""
                return f"{cpq / metric:.6f}"

            writer.writerow({
                "model": model,
                "variant": variant,
                "benchmark": benchmark,
                "cost_per_query_usd": f"{cpq:.6f}",
                "ndcg": "" if ndcg != ndcg else f"{ndcg:.4f}",
                "pass_rate": "" if pass_rate != pass_rate else f"{pass_rate:.4f}",
                "cost_per_ndcg_point": _ratio(cpq, ndcg),
                "cost_per_pass_rate_point": _ratio(cpq, pass_rate),
            })
    logger.info("Wrote pareto data: %s (%d rows)", out_path, len(cells))


def write_cache_hit(
    cells: dict[CellKey, _CellAccumulator],
    out_path: Path,
) -> None:
    """Write ``cache_hit.csv`` — cache hit rate per cell.

    Columns:
        model, variant, benchmark,
        total_prompt_tokens, cached_tokens, non_cached_tokens, cache_hit_rate
    """
    fieldnames = [
        "model", "variant", "benchmark",
        "total_prompt_tokens", "cached_tokens", "non_cached_tokens", "cache_hit_rate",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for (model, variant, benchmark), acc in sorted(cells.items()):
            total_prompt = acc.total_input_tokens + acc.total_cached_tokens
            writer.writerow({
                "model": model,
                "variant": variant,
                "benchmark": benchmark,
                "total_prompt_tokens": total_prompt,
                "cached_tokens": acc.total_cached_tokens,
                "non_cached_tokens": acc.total_input_tokens,
                "cache_hit_rate": f"{acc.cache_hit_rate:.4f}",
            })
    logger.info("Wrote cache hit: %s (%d rows)", out_path, len(cells))


def write_latency(
    cells: dict[CellKey, _CellAccumulator],
    out_path: Path,
) -> None:
    """Write ``latency.csv`` — full latency percentile distribution.

    Columns:
        model, variant, benchmark, n_calls,
        p50_ms, p90_ms, p95_ms, p99_ms, mean_ms, max_ms
    """
    fieldnames = [
        "model", "variant", "benchmark", "n_calls",
        "p50_ms", "p90_ms", "p95_ms", "p99_ms", "mean_ms", "max_ms",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for (model, variant, benchmark), acc in sorted(cells.items()):
            lats = acc.latencies_ms
            mean_ms = statistics.mean(lats) if lats else 0.0
            max_ms = max(lats) if lats else 0.0
            writer.writerow({
                "model": model,
                "variant": variant,
                "benchmark": benchmark,
                "n_calls": len(lats),
                "p50_ms": f"{acc.latency_percentile(50):.1f}",
                "p90_ms": f"{acc.latency_percentile(90):.1f}",
                "p95_ms": f"{acc.latency_percentile(95):.1f}",
                "p99_ms": f"{acc.latency_percentile(99):.1f}",
                "mean_ms": f"{mean_ms:.1f}",
                "max_ms": f"{max_ms:.1f}",
            })
    logger.info("Wrote latency: %s (%d rows)", out_path, len(cells))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _load_metrics_json(path: Path) -> dict[str, dict[str, float]]:
    """Load optional eval metrics JSON: ``{run_id: {ndcg: ..., pass_rate: ...}}``."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: aggregate manifests and produce cost CSVs."""
    parser = argparse.ArgumentParser(
        prog="python -m toolbench.observability.aggregator",
        description="Aggregate manifest JSONL files into report-ready cost CSVs.",
    )
    parser.add_argument(
        "--manifest-dir",
        required=True,
        type=Path,
        help="Directory containing one or more manifest.jsonl files (searched recursively).",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output directory for CSV files.",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON file mapping run_id → {ndcg, pass_rate, ...}. "
            "Enables $/NDCG-point columns in pareto_data.csv."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    manifest_dir: Path = args.manifest_dir
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_paths = sorted(manifest_dir.rglob("manifest.jsonl"))
    if not manifest_paths:
        # Also accept any .jsonl files directly in the dir
        manifest_paths = sorted(manifest_dir.glob("*.jsonl"))
    if not manifest_paths:
        logger.error("No manifest JSONL files found in %s", manifest_dir)
        raise SystemExit(1)

    metrics_by_run: dict[str, dict[str, float]] | None = None
    if args.metrics_json:
        metrics_by_run = _load_metrics_json(args.metrics_json)
        logger.info("Loaded eval metrics for %d run IDs", len(metrics_by_run))

    cells, pareto_extras = aggregate(manifest_paths, metrics_by_run)

    write_cost_table(cells, out_dir / "cost_table.csv")
    write_pareto_data(cells, pareto_extras, out_dir / "pareto_data.csv")
    write_cache_hit(cells, out_dir / "cache_hit.csv")
    write_latency(cells, out_dir / "latency.csv")

    logger.info(
        "Aggregation complete: %d cells across %d manifest files → %s",
        len(cells),
        len(manifest_paths),
        out_dir,
    )


if __name__ == "__main__":
    main()
