"""Aggregator integration -- dispatch out_dir to manifest-derived CSVs.

Thin wrapper over ``toolbench.observability.aggregator`` that:
  1. Walks ``<out_dir>/<cell_id>/manifest.jsonl`` and feeds them all to
     :func:`toolbench.observability.aggregator.aggregate`.
  2. Writes ``cost_table.csv``, ``pareto_data.csv``, ``cache_hit.csv``,
     ``latency.csv`` under ``<out_dir>/aggregate/``.

The observability aggregator already understands manifest JSONL records;
we only need to find them.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# Import from the StableToolBench namespace where the aggregator lives.
# Same trick the runner uses for ManifestWriter.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "StableToolBench") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "StableToolBench"))

# Importing here keeps the dispatcher executor lightweight at module load.


def _find_manifests(out_root: Path) -> list[Path]:
    """Find every ``manifest.jsonl`` under cell directories.

    Layout::

        <out_root>/<cell_id>/manifest.jsonl
        <out_root>/<cell_id>/<run_id>/manifest.jsonl   (when run.py auto-fills run_id)

    Both layouts are picked up.
    """
    manifests: list[Path] = []
    for p in out_root.rglob("manifest.jsonl"):
        # Skip anything under the aggregate output dir (safety: aggregator
        # outputs are CSV, but rglob shouldn't recurse into our own writes)
        if "aggregate" in p.parts:
            continue
        manifests.append(p)
    return sorted(manifests)


def run_aggregator(out_root: Path, agg_dir: Path) -> None:
    """Aggregate manifests under ``out_root`` into CSVs under ``agg_dir``.

    Args:
        out_root: Dispatch out_dir (contains one subdir per cell).
        agg_dir: Output directory for the CSVs.

    Side effects:
        Writes four CSVs to ``agg_dir/``. Returns silently if no manifest
        files exist (with a logged warning).
    """
    from toolbench.observability.aggregator import (
        aggregate,
        write_cost_table,
        write_pareto_data,
        write_cache_hit,
        write_latency,
    )

    manifests = _find_manifests(out_root)
    if not manifests:
        log.warning("No manifest.jsonl files under %s — nothing to aggregate", out_root)
        return

    log.info("Aggregating %d manifest files under %s", len(manifests), out_root)
    cells, pareto_extras = aggregate(manifests)

    agg_dir.mkdir(parents=True, exist_ok=True)
    write_cost_table(cells, agg_dir / "cost_table.csv")
    write_pareto_data(cells, pareto_extras, agg_dir / "pareto_data.csv")
    write_cache_hit(cells, agg_dir / "cache_hit.csv")
    write_latency(cells, agg_dir / "latency.csv")
    log.info("Wrote aggregate CSVs to %s", agg_dir)
