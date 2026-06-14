"""
CLI: generate the 5 doxastic belief-dynamics figures from belief_metrics.csv.

Reads:
    <metrics_csv>    — output of compute_belief_metrics.py

Writes (to <out_dir>/):
    fig1_alignment.pdf    — ρ(b_g*, t*) vs generation (line + shaded CI)
    fig2_entropy.pdf      — Shannon entropy of belief clusters vs generation
    fig3_coverage.pdf     — witness coverage histogram across queries
    fig4_revision.pdf     — mean belief revision rate (embedding distance) g→g+1
    fig5_recycling.pdf    — intra-generation Jaccard overlap (λ=0 motivation)

Usage::

    python scripts/plot_belief_dynamics.py \\
        --metrics-csv results/belief_metrics.csv \\
        --out-dir figures/ \\
        [--format pdf]        # or png / svg
        [--dpi 300]
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_metrics_csv(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the metrics CSV, separating query rows from aggregate rows.

    Args:
        path: Path to belief_metrics.csv produced by compute_belief_metrics.py.

    Returns:
        Tuple of (query_rows, aggregates) where aggregates has keys
        '__mean__', '__ci_lo__', '__ci_hi__'.
    """
    query_rows: list[dict[str, Any]] = []
    aggregates: dict[str, Any] = {}
    agg_qids = {"__mean__", "__ci_lo__", "__ci_hi__"}

    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            qid = row["qid"]
            numeric_row = {k: _try_float(v) for k, v in row.items()}
            numeric_row["qid"] = qid
            if qid in agg_qids:
                aggregates[qid] = numeric_row
            else:
                query_rows.append(numeric_row)

    return query_rows, aggregates


def _try_float(v: str) -> float | str:
    """Try to parse string as float, returning original string on failure."""
    try:
        return float(v)
    except (ValueError, TypeError):
        return v


def _collect_gen_series(
    rows: list[dict[str, Any]],
    prefix: str,
) -> tuple[list[int], np.ndarray]:
    """Extract per-generation series from rows.

    Args:
        rows: List of metric row dicts.
        prefix: Column prefix, e.g. 'alignment_gen'.

    Returns:
        Tuple of (generation_indices, values_array) where values_array has
        shape (n_queries, n_generations).  Missing values become nan.
    """
    gen_cols = sorted(
        [k for k in rows[0] if k.startswith(prefix)],
        key=lambda c: int(c[len(prefix):]),
    )
    if not gen_cols:
        return [], np.empty((0, 0))

    gens = [int(c[len(prefix):]) for c in gen_cols]
    mat = np.array(
        [[float(row.get(c, float("nan"))) for c in gen_cols] for row in rows],
        dtype=np.float64,
    )
    return gens, mat


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _setup_matplotlib() -> Any:
    """Import and configure matplotlib (non-interactive backend).

    Returns:
        The matplotlib.pyplot module.
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive, no display required
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "pdf.fonttype": 42,   # embed fonts for portable PDF output
    })
    return plt


def _save(plt: Any, out_dir: Path, name: str, fmt: str, dpi: int) -> Path:
    """Save current figure to file.

    Args:
        plt: pyplot module.
        out_dir: Output directory.
        name: Base filename without extension.
        fmt: File format string (pdf, png, svg).
        dpi: DPI for raster formats.

    Returns:
        Path of the saved file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.{fmt}"
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Figure 1: Belief-evidence alignment
# ---------------------------------------------------------------------------

def plot_alignment(
    query_rows: list[dict[str, Any]],
    aggregates: dict[str, Any],
    out_dir: Path,
    fmt: str,
    dpi: int,
) -> None:
    """ρ(b_g*, t*) vs generation — monotone-rising expected.

    Args:
        query_rows: Per-query metric rows.
        aggregates: Aggregate rows from CSV (__mean__, __ci_lo__, __ci_hi__).
        out_dir: Output directory.
        fmt: File format.
        dpi: DPI for raster.
    """
    plt = _setup_matplotlib()
    gens, mat = _collect_gen_series(query_rows, "alignment_gen")
    if not gens:
        logger.warning("No alignment columns found — skipping fig1")
        return

    mean_agg = aggregates.get("__mean__", {})
    lo_agg = aggregates.get("__ci_lo__", {})
    hi_agg = aggregates.get("__ci_hi__", {})

    mean_vals = np.array([mean_agg.get(f"alignment_gen{g}", float("nan")) for g in gens])
    lo_vals   = np.array([lo_agg.get(f"alignment_gen{g}", float("nan")) for g in gens])
    hi_vals   = np.array([hi_agg.get(f"alignment_gen{g}", float("nan")) for g in gens])

    fig, ax = plt.subplots(figsize=(6, 4))
    # Individual query traces (light)
    for row in query_rows:
        vals = [row.get(f"alignment_gen{g}", float("nan")) for g in gens]
        ax.plot(gens, vals, color="steelblue", alpha=0.12, linewidth=0.8)
    # Mean + CI
    ax.plot(gens, mean_vals, color="steelblue", linewidth=2.0, label="Mean ρ")
    ax.fill_between(gens, lo_vals, hi_vals, alpha=0.25, color="steelblue", label="95% CI")
    ax.set_xlabel("Generation")
    ax.set_ylabel("ρ(b_g*, t*)  [cosine]")
    ax.set_title("Belief-Evidence Alignment over Generations")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    _save(plt, out_dir, "fig1_alignment", fmt, dpi)


# ---------------------------------------------------------------------------
# Figure 2: Belief population entropy
# ---------------------------------------------------------------------------

def plot_entropy(
    query_rows: list[dict[str, Any]],
    aggregates: dict[str, Any],
    out_dir: Path,
    fmt: str,
    dpi: int,
) -> None:
    """Shannon entropy of belief-embedding clusters vs generation.

    Args:
        query_rows: Per-query metric rows.
        aggregates: Aggregate rows.
        out_dir: Output directory.
        fmt: File format.
        dpi: DPI.
    """
    plt = _setup_matplotlib()
    gens, mat = _collect_gen_series(query_rows, "entropy_gen")
    if not gens:
        logger.warning("No entropy columns — skipping fig2")
        return

    mean_agg = aggregates.get("__mean__", {})
    lo_agg   = aggregates.get("__ci_lo__", {})
    hi_agg   = aggregates.get("__ci_hi__", {})

    mean_vals = np.array([mean_agg.get(f"entropy_gen{g}", float("nan")) for g in gens])
    lo_vals   = np.array([lo_agg.get(f"entropy_gen{g}", float("nan")) for g in gens])
    hi_vals   = np.array([hi_agg.get(f"entropy_gen{g}", float("nan")) for g in gens])

    fig, ax = plt.subplots(figsize=(6, 4))
    for row in query_rows:
        vals = [row.get(f"entropy_gen{g}", float("nan")) for g in gens]
        ax.plot(gens, vals, color="darkorange", alpha=0.12, linewidth=0.8)
    ax.plot(gens, mean_vals, color="darkorange", linewidth=2.0, label="Mean H")
    ax.fill_between(gens, lo_vals, hi_vals, alpha=0.25, color="darkorange", label="95% CI")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Shannon Entropy [bits]")
    ax.set_title("Belief Population Entropy over Generations")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    _save(plt, out_dir, "fig2_entropy", fmt, dpi)


# ---------------------------------------------------------------------------
# Figure 3: Witness coverage histogram
# ---------------------------------------------------------------------------

def plot_coverage(
    query_rows: list[dict[str, Any]],
    out_dir: Path,
    fmt: str,
    dpi: int,
) -> None:
    """Histogram of per-query witness coverage across the test set.

    Args:
        query_rows: Per-query metric rows.
        out_dir: Output directory.
        fmt: File format.
        dpi: DPI.
    """
    plt = _setup_matplotlib()
    vals = [float(r["witness_coverage"]) for r in query_rows
            if "witness_coverage" in r and math.isfinite(float(r.get("witness_coverage", float("nan"))))]
    if not vals:
        logger.warning("No witness_coverage values — skipping fig3")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(vals, bins=20, color="forestgreen", edgecolor="white", alpha=0.8)
    mean_cov = float(np.mean(vals))
    ax.axvline(mean_cov, color="darkgreen", linestyle="--", linewidth=1.5,
               label=f"Mean = {mean_cov:.2f}")
    ax.set_xlabel("Witness Coverage")
    ax.set_ylabel("Count (queries)")
    ax.set_title("Witness Coverage Distribution")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    _save(plt, out_dir, "fig3_coverage", fmt, dpi)


# ---------------------------------------------------------------------------
# Figure 4: Belief revision rate
# ---------------------------------------------------------------------------

def plot_revision_rate(
    query_rows: list[dict[str, Any]],
    aggregates: dict[str, Any],
    out_dir: Path,
    fmt: str,
    dpi: int,
) -> None:
    """Mean embedding distance between consecutive generations g→g+1.

    Args:
        query_rows: Per-query metric rows.
        aggregates: Aggregate rows.
        out_dir: Output directory.
        fmt: File format.
        dpi: DPI.
    """
    plt = _setup_matplotlib()
    gens, mat = _collect_gen_series(query_rows, "revision_rate_gen")
    if not gens:
        logger.warning("No revision_rate columns — skipping fig4")
        return

    mean_agg = aggregates.get("__mean__", {})
    lo_agg   = aggregates.get("__ci_lo__", {})
    hi_agg   = aggregates.get("__ci_hi__", {})

    mean_vals = np.array([mean_agg.get(f"revision_rate_gen{g}", float("nan")) for g in gens])
    lo_vals   = np.array([lo_agg.get(f"revision_rate_gen{g}", float("nan")) for g in gens])
    hi_vals   = np.array([hi_agg.get(f"revision_rate_gen{g}", float("nan")) for g in gens])

    # X-labels: "g→g+1"
    xlabels = [f"{g}→{g+1}" for g in gens]

    fig, ax = plt.subplots(figsize=(6, 4))
    for row in query_rows:
        vals = [row.get(f"revision_rate_gen{g}", float("nan")) for g in gens]
        ax.plot(range(len(gens)), vals, color="mediumpurple", alpha=0.12, linewidth=0.8)
    ax.plot(range(len(gens)), mean_vals, color="mediumpurple", linewidth=2.0, label="Mean distance")
    ax.fill_between(range(len(gens)), lo_vals, hi_vals, alpha=0.25, color="mediumpurple", label="95% CI")
    ax.set_xticks(range(len(gens)))
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("Generation Transition")
    ax.set_ylabel("Mean Embedding Distance (ℓ₂)")
    ax.set_title("Belief Revision Rate across Generations")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    _save(plt, out_dir, "fig4_revision", fmt, dpi)


# ---------------------------------------------------------------------------
# Figure 5: Memory-off recycling
# ---------------------------------------------------------------------------

def plot_recycling(
    query_rows: list[dict[str, Any]],
    aggregates: dict[str, Any],
    out_dir: Path,
    fmt: str,
    dpi: int,
) -> None:
    """Intra-generation Jaccard overlap of retrieval sets (λ=0 run).

    Args:
        query_rows: Per-query metric rows.
        aggregates: Aggregate rows.
        out_dir: Output directory.
        fmt: File format.
        dpi: DPI.
    """
    plt = _setup_matplotlib()
    gens, mat = _collect_gen_series(query_rows, "memory_overlap_gen")
    if not gens:
        logger.warning("No memory_overlap columns — skipping fig5")
        return

    mean_agg = aggregates.get("__mean__", {})
    lo_agg   = aggregates.get("__ci_lo__", {})
    hi_agg   = aggregates.get("__ci_hi__", {})

    mean_vals = np.array([mean_agg.get(f"memory_overlap_gen{g}", float("nan")) for g in gens])
    lo_vals   = np.array([lo_agg.get(f"memory_overlap_gen{g}", float("nan")) for g in gens])
    hi_vals   = np.array([hi_agg.get(f"memory_overlap_gen{g}", float("nan")) for g in gens])

    fig, ax = plt.subplots(figsize=(6, 4))
    for row in query_rows:
        vals = [row.get(f"memory_overlap_gen{g}", float("nan")) for g in gens]
        ax.plot(gens, vals, color="firebrick", alpha=0.12, linewidth=0.8)
    ax.plot(gens, mean_vals, color="firebrick", linewidth=2.0, label="Mean Jaccard")
    ax.fill_between(gens, lo_vals, hi_vals, alpha=0.25, color="firebrick", label="95% CI")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean Intra-gen Jaccard Overlap")
    ax.set_title("Tool Retrieval Recycling (λ=0 run)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    _save(plt, out_dir, "fig5_recycling", fmt, dpi)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for plot_belief_dynamics.

    Args:
        argv: Optional argument list.

    Returns:
        Parsed Namespace.
    """
    parser = argparse.ArgumentParser(
        description="Generate 5 doxastic belief-dynamics figures from belief_metrics.csv."
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        required=True,
        help="Path to belief_metrics.csv (output of compute_belief_metrics.py).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures"),
        help="Output directory for figures [default: figures/].",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Figure file format [default: pdf].",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for raster formats [default: 300].",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for plot_belief_dynamics CLI.

    Args:
        argv: Optional argument list.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)

    if not args.metrics_csv.exists():
        logger.error("Metrics CSV not found: %s", args.metrics_csv)
        sys.exit(1)

    query_rows, aggregates = load_metrics_csv(args.metrics_csv)
    if not query_rows:
        logger.error("No query rows in %s", args.metrics_csv)
        sys.exit(1)

    logger.info("Loaded %d query rows from %s", len(query_rows), args.metrics_csv)

    fmt = args.format
    dpi = args.dpi
    out_dir = args.out_dir

    plot_alignment(query_rows, aggregates, out_dir, fmt, dpi)
    plot_entropy(query_rows, aggregates, out_dir, fmt, dpi)
    plot_coverage(query_rows, out_dir, fmt, dpi)
    plot_revision_rate(query_rows, aggregates, out_dir, fmt, dpi)
    plot_recycling(query_rows, aggregates, out_dir, fmt, dpi)

    logger.info("All figures written to %s/", out_dir)


if __name__ == "__main__":
    main()
