"""
CLI: compute per-qid doxastic metrics from belief trace JSONL files.

Reads:
    <belief_dir>/<qid>.jsonl    — per-query belief traces (from BeliefTracer)
    <gold_file>                 — per-qid gold tool annotation (JSONL or CSV)

Writes:
    <out_csv>                   — per-qid metrics + aggregate summary

Gold file format (JSONL, one record per line):
    {"qid": "...", "gold_tool_id": "category/tool_name", "gold_tool_desc": "..."}

Output CSV columns:
    qid,
    alignment_gen{g} (one per generation),
    entropy_gen{g},
    witness_coverage,
    revision_rate_gen{g} (one per g→g+1 transition),
    memory_overlap_gen{g} (one per generation)

Usage::

    python scripts/compute_belief_metrics.py \\
        --belief-dir beliefs/ \\
        --gold-file data/gold_labels.jsonl \\
        --out-csv results/belief_metrics.csv \\
        [--held-out-split tdEG]   # exclude this ToolRet domain
        [--embedder-model MODEL_ID]
        [--k-clusters 4]
        [--n-bootstrap 1000]
        [--seed 42]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_gold(gold_path: Path) -> dict[str, dict[str, str]]:
    """Load gold label file into a dict keyed by qid.

    Supports JSONL (one JSON object per line) and CSV (with headers).

    Args:
        gold_path: Path to gold annotation file.

    Returns:
        Dict mapping qid → {"gold_tool_id": str, "gold_tool_desc": str}.
    """
    gold: dict[str, dict[str, str]] = {}
    suffix = gold_path.suffix.lower()

    with open(gold_path, encoding="utf-8") as fh:
        if suffix == ".csv":
            reader = csv.DictReader(fh)
            for row in reader:
                qid = row["qid"].strip()
                gold[qid] = {
                    "gold_tool_id": row.get("gold_tool_id", ""),
                    "gold_tool_desc": row.get("gold_tool_desc", ""),
                }
        else:
            # JSONL
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    qid = rec["qid"]
                    gold[qid] = {
                        "gold_tool_id": rec.get("gold_tool_id", ""),
                        "gold_tool_desc": rec.get("gold_tool_desc", ""),
                    }
                except (json.JSONDecodeError, KeyError) as exc:
                    logger.warning("gold file line %d: %s", lineno, exc)

    logger.info("Loaded %d gold labels from %s", len(gold), gold_path)
    return gold


def _bootstrap_ci(values: list[float], n: int = 1000, seed: int = 42) -> tuple[float, float]:
    """95% bootstrap confidence interval for the mean.

    Args:
        values: List of finite floats.
        n: Number of bootstrap resamples.
        seed: RNG seed.

    Returns:
        Tuple (lower, upper) for the 95% CI.
    """
    rng = np.random.default_rng(seed)
    arr = np.array([v for v in values if math.isfinite(v)], dtype=np.float64)
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    boot_means = [np.mean(rng.choice(arr, len(arr), replace=True)) for _ in range(n)]
    return (float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5)))


def _safe_mean(values: list[float]) -> float:
    """Mean of finite values, or nan if none."""
    finite = [v for v in values if math.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


# ---------------------------------------------------------------------------
# Per-qid computation
# ---------------------------------------------------------------------------

def compute_metrics_for_qid(
    snapshots_path: Path,
    gold_tool_id: str,
    gold_tool_desc: str,
    embedder: Any,          # SharedEmbedder or mock
    k_clusters: int = 4,
) -> dict[str, Any]:
    """Compute all five doxastic metrics for one query.

    Args:
        snapshots_path: Path to the per-qid belief JSONL file.
        gold_tool_id: Ground-truth tool id.
        gold_tool_desc: Ground-truth tool description text.
        embedder: SharedEmbedder instance (lazy-loaded on first call).
        k_clusters: Number of clusters for entropy computation.

    Returns:
        Dict with keys: alignment_per_gen, entropy_per_gen, witness_coverage,
        revision_rate, memory_overlap_per_gen.
    """
    from toolbench.inference.instrumentation.belief_trace import (
        read_belief_trace,
        group_by_generation,
    )
    from toolbench.inference.instrumentation.analysis import (
        belief_evidence_alignment,
        belief_population_entropy,
        witness_coverage,
        belief_revision_rate,
        memory_off_recycling,
        populate_embeddings,
    )

    snapshots = read_belief_trace(snapshots_path)
    if not snapshots:
        logger.warning("No snapshots in %s — skipping", snapshots_path)
        return {}

    beliefs_per_gen = group_by_generation(snapshots)

    # Populate embeddings for metrics that need them
    populate_embeddings(beliefs_per_gen, embedder)

    alignment = belief_evidence_alignment(beliefs_per_gen, gold_tool_desc, embedder)
    entropy = belief_population_entropy(beliefs_per_gen, k_clusters=k_clusters)
    cov = witness_coverage(beliefs_per_gen, gold_tool_id)
    revision = belief_revision_rate(beliefs_per_gen, embedder)
    recycling = memory_off_recycling(beliefs_per_gen)

    return {
        "alignment_per_gen": alignment,
        "entropy_per_gen": entropy,
        "witness_coverage": cov,
        "revision_rate": revision,
        "memory_overlap_per_gen": recycling,
    }


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def _build_row(qid: str, metrics: dict[str, Any]) -> dict[str, Any]:
    """Flatten metrics dict into a flat CSV row.

    Args:
        qid: Query identifier.
        metrics: Output of compute_metrics_for_qid.

    Returns:
        Dict with qid + flattened numeric fields.
    """
    row: dict[str, Any] = {"qid": qid}

    for g, v in enumerate(metrics.get("alignment_per_gen", [])):
        row[f"alignment_gen{g}"] = v

    for g, v in enumerate(metrics.get("entropy_per_gen", [])):
        row[f"entropy_gen{g}"] = v

    row["witness_coverage"] = metrics.get("witness_coverage", float("nan"))

    for g, v in enumerate(metrics.get("revision_rate", [])):
        row[f"revision_rate_gen{g}"] = v

    for g, v in enumerate(metrics.get("memory_overlap_gen", metrics.get("memory_overlap_per_gen", []))):
        row[f"memory_overlap_gen{g}"] = v

    return row


def write_metrics_csv(
    rows: list[dict[str, Any]],
    out_path: Path,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> None:
    """Write per-qid metrics to CSV with aggregate summary rows.

    Args:
        rows: List of flat metric dicts (one per qid).
        out_path: Output CSV path.
        n_bootstrap: Bootstrap resamples for CI.
        seed: RNG seed.
    """
    if not rows:
        logger.warning("No rows to write — skipping CSV output")
        return

    # Determine full column set
    all_cols: list[str] = ["qid"]
    seen: set[str] = {"qid"}
    for row in rows:
        for k in row:
            if k not in seen:
                all_cols.append(k)
                seen.add(k)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

        # Aggregate rows
        numeric_cols = [c for c in all_cols if c != "qid"]
        mean_row: dict[str, Any] = {"qid": "__mean__"}
        ci_lo_row: dict[str, Any] = {"qid": "__ci_lo__"}
        ci_hi_row: dict[str, Any] = {"qid": "__ci_hi__"}

        for col in numeric_cols:
            vals = [float(r[col]) for r in rows if col in r and r[col] is not None]
            mean_row[col] = _safe_mean(vals)
            lo, hi = _bootstrap_ci(vals, n=n_bootstrap, seed=seed)
            ci_lo_row[col] = lo
            ci_hi_row[col] = hi

        writer.writerow(mean_row)
        writer.writerow(ci_lo_row)
        writer.writerow(ci_hi_row)

    logger.info("Wrote %d query rows + aggregates to %s", len(rows), out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Argument list (defaults to sys.argv).

    Returns:
        Parsed Namespace.
    """
    parser = argparse.ArgumentParser(
        description="Compute per-qid doxastic belief metrics from belief trace JSONL."
    )
    parser.add_argument(
        "--belief-dir",
        type=Path,
        required=True,
        help="Directory containing per-qid JSONL files produced by BeliefTracer.",
    )
    parser.add_argument(
        "--gold-file",
        type=Path,
        required=True,
        help="Gold tool annotation file (JSONL or CSV with qid,gold_tool_id,gold_tool_desc).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/belief_metrics.csv"),
        help="Output CSV path [default: results/belief_metrics.csv].",
    )
    parser.add_argument(
        "--held-out-split",
        type=str,
        default=None,
        help="ToolRet domain to exclude (e.g. 'tdEG') for held-out evaluation. "
             "Queries whose qid starts with this prefix are skipped.",
    )
    parser.add_argument(
        "--embedder-model",
        type=str,
        default=None,
        help="Override embedder model name (sets BELIEF_EMBEDDER_MODEL env var).",
    )
    parser.add_argument(
        "--k-clusters",
        type=int,
        default=4,
        help="Number of k-means clusters for entropy computation [default: 4].",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Bootstrap resamples for CI [default: 1000].",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed [default: 42].",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for compute_belief_metrics CLI.

    Args:
        argv: Optional argument list (uses sys.argv if None).
    """
    args = parse_args(argv)

    # Apply embedder model override before importing SharedEmbedder
    import os
    if args.embedder_model:
        os.environ["BELIEF_EMBEDDER_MODEL"] = args.embedder_model

    # Lazy import after env var is set
    from toolbench.inference.instrumentation.embedder import SharedEmbedder

    # Load gold labels
    gold = _load_gold(args.gold_file)

    # Discover belief JSONL files
    jsonl_files = sorted(args.belief_dir.glob("*.jsonl"))
    if not jsonl_files:
        logger.error("No .jsonl files found in %s", args.belief_dir)
        sys.exit(1)

    logger.info("Found %d belief trace files in %s", len(jsonl_files), args.belief_dir)

    embedder = SharedEmbedder.instance()
    rows: list[dict[str, Any]] = []

    for jf in jsonl_files:
        # Reverse the safe_qid substitution to recover the original qid
        stem = jf.stem  # e.g. "toolret_q0042"
        qid_candidate = stem  # may also try replacing _ back to : or /

        # Match against gold keys
        gold_entry = gold.get(qid_candidate) or gold.get(stem.replace("_", ":"))
        if gold_entry is None:
            logger.warning("No gold label for %s (tried %s) — skipping", jf.name, qid_candidate)
            continue

        qid = qid_candidate

        # Held-out split filtering
        if args.held_out_split and qid.startswith(args.held_out_split):
            logger.info("Skipping held-out qid %s (domain=%s)", qid, args.held_out_split)
            continue

        try:
            metrics = compute_metrics_for_qid(
                snapshots_path=jf,
                gold_tool_id=gold_entry["gold_tool_id"],
                gold_tool_desc=gold_entry["gold_tool_desc"],
                embedder=embedder,
                k_clusters=args.k_clusters,
            )
        except Exception as exc:
            logger.error("Error computing metrics for %s: %s", qid, exc, exc_info=True)
            continue

        if metrics:
            rows.append(_build_row(qid, metrics))

    if not rows:
        logger.error("No metric rows produced — check belief dir and gold file.")
        sys.exit(1)

    write_metrics_csv(rows, args.out_csv, n_bootstrap=args.n_bootstrap, seed=args.seed)
    logger.info("Done. %d queries processed.", len(rows))


if __name__ == "__main__":
    main()
