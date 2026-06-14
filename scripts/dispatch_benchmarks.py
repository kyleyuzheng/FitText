#!/usr/bin/env python3
"""Parallel benchmark dispatcher CLI.

Reads a dispatch spec YAML, expands it into N concrete (technique x baseline
x benchmark x split x model) cells, then fans them out across a local
``ProcessPoolExecutor`` plus optional remote hosts (ssh). After all cells
reach a terminal state the manifest aggregator is invoked to produce
``cost_table.csv``, ``pareto_data.csv``, ``cache_hit.csv``, and
``latency.csv`` under ``<out>/aggregate/``.

Examples
--------

Dry-run plan expansion (no cells executed)::

    python scripts/dispatch_benchmarks.py \
        --spec configs/dispatch/cheap_sota_small.yaml \
        --out "${FITTEXT_RUNTIME_ROOT:-runs}/$JOB_ID/dispatch" \
        --dry-run

Live dispatch on the coordinator + a remote worker host::

    python scripts/dispatch_benchmarks.py \
        --spec configs/dispatch/cheap_sota_small.yaml \
        --out "${FITTEXT_RUNTIME_ROOT:-runs}/$JOB_ID/dispatch" \
        --hosts worker1

Override global cost ceiling::

    python scripts/dispatch_benchmarks.py \
        --spec configs/dispatch/cheap_sota_small.yaml \
        --out "${FITTEXT_RUNTIME_ROOT:-runs}/$JOB_ID/dispatch" \
        --max-cost 50

Exit codes
----------
0  All cells completed successfully (or skipped via resume).
1  One or more cells failed.
2  Pre-flight failure (git drift on a remote host, invalid spec).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

# Ensure repo root and StableToolBench are importable when run as a script
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "StableToolBench") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "StableToolBench"))

from toolbench.dispatcher import (  # noqa: E402  (path setup above)
    DispatchExecutor,
    expand_cells,
    load_dispatch_spec,
)


log = logging.getLogger("dispatch")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        prog="dispatch_benchmarks",
        description="Fan out N techniques x M baselines x benchmarks in parallel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--spec", required=True, type=Path, help="Dispatch spec YAML.")
    p.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Dispatch output root. Each cell writes to <out>/<cell_id>/.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the expanded plan and exit without executing cells.",
    )
    p.add_argument(
        "--hosts",
        type=str,
        default=None,
        help=(
            "Comma-separated remote host aliases (override spec.parallelism.hosts). "
            "Empty string disables remote dispatch."
        ),
    )
    p.add_argument(
        "--max-cost",
        type=float,
        default=None,
        help="Override spec.budget.total_usd (global cost ceiling).",
    )
    p.add_argument(
        "--max-local-workers",
        type=int,
        default=None,
        help="Override spec.parallelism.max_local_workers.",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume (re-run cells even if result.json exists).",
    )
    p.add_argument(
        "--pins",
        type=Path,
        default=_REPO_ROOT / "configs" / "model_pins.yaml",
        help="Path to model_pins.yaml for resolving model entries.",
    )
    p.add_argument(
        "--remote-repo",
        type=str,
        default="~/FitText",
        help="Remote path to the FitText clone on ssh hosts.",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Per-cell timeout in seconds.",
    )
    p.add_argument(
        "--skip-remote-git-check",
        action="store_true",
        help="Skip the remote git-HEAD drift guard.",
    )
    p.add_argument(
        "--verbose", action="store_true", help="Enable DEBUG logging."
    )
    return p.parse_args(argv)


def _make_dispatch_id(spec_name: str) -> str:
    """Construct a unique dispatch run id.

    Format: ``<UTC YYYYMMDDTHHMMSS>_<6char hash of nanos+spec>``
    """
    ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    seed = f"{time.time_ns()}|{spec_name}".encode()
    short = hashlib.sha256(seed).hexdigest()[:6]
    return f"{ts}_{short}"


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns the process exit code."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Load + validate spec
    try:
        spec = load_dispatch_spec(args.spec)
    except Exception as e:  # pylint: disable=broad-except
        log.error("Failed to load spec %s: %s", args.spec, e)
        return 2

    # CLI overrides
    if args.hosts is not None:
        spec.parallelism.hosts = (
            [h.strip() for h in args.hosts.split(",") if h.strip()]
            if args.hosts
            else []
        )
    if args.max_cost is not None:
        spec.budget.total_usd = float(args.max_cost)
    if args.max_local_workers is not None:
        spec.parallelism.max_local_workers = int(args.max_local_workers)
    if args.no_resume:
        spec.parallelism.resume = False

    # Expand cells
    try:
        cells = expand_cells(spec, pins_path=Path(args.pins))
    except Exception as e:  # pylint: disable=broad-except
        log.error("Cell expansion failed: %s", e)
        return 2

    log.info(
        "spec=%s  n_cells=%d  techniques=%d  baselines=%d  models=%d  benchmarks=%d",
        spec.name,
        len(cells),
        len(spec.techniques),
        len(spec.baselines),
        len(spec.models),
        len(spec.benchmarks),
    )

    # Plan summary
    if args.dry_run:
        print(f"=== DISPATCH PLAN: {spec.name} ===")
        print(f"  total cells: {len(cells)}")
        print(f"  techniques:  {spec.techniques}")
        print(f"  baselines:   {spec.baselines}")
        print(f"  hosts:       {['local', *spec.parallelism.hosts]}")
        print(f"  budget:      ${spec.budget.total_usd:.2f} total, "
              f"${spec.budget.per_cell_usd:.2f}/cell")
        print(f"  resume:      {spec.parallelism.resume}")
        print()
        for i, c in enumerate(cells):
            print(
                f"  [{i:03d}] {c.cell_id}   "
                f"({c.kind}:{c.label}, {c.benchmark_name}:{c.split}, "
                f"model={c.model_name})"
            )
        return 0

    # Prepare out_dir
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    dispatch_id = _make_dispatch_id(spec.name)
    # Persist the resolved spec next to results for audit
    (out_dir / "dispatch_spec.resolved.yaml").write_text(
        _yaml_dump(spec.model_dump()), encoding="utf-8"
    )

    # Build executor
    remote_repo_by_host = {h: args.remote_repo for h in spec.parallelism.hosts}
    executor = DispatchExecutor(
        spec=spec,
        cells=cells,
        out_dir=out_dir,
        repo_root=_REPO_ROOT,
        dispatch_id=dispatch_id,
        remote_repo_by_host=remote_repo_by_host,
        timeout_s=args.timeout,
        require_remote_git_match=not args.skip_remote_git_check,
    )

    # Execute
    log.info(
        "Dispatching %d cells via %d local workers + %d remote hosts → %s",
        len(cells),
        spec.parallelism.max_local_workers,
        len(spec.parallelism.hosts),
        out_dir,
    )
    snapshot = executor.run()

    # Aggregate
    try:
        agg_dir = executor.aggregate()
        log.info("Aggregation complete: %s", agg_dir)
    except Exception as e:  # pylint: disable=broad-except
        log.error("Aggregation failed: %s", e)

    # Final summary
    n_done = snapshot.get("n_done", 0)
    n_failed = snapshot.get("n_failed", 0)
    n_skipped = snapshot.get("n_skipped", 0)
    n_budget = snapshot.get("n_budget_skip", 0)
    total_cost = snapshot.get("running_total_cost_usd", 0.0)
    print()
    print(f"=== DISPATCH SUMMARY: {spec.name} ===")
    print(f"  dispatch_id:  {dispatch_id}")
    print(f"  done:         {n_done}")
    print(f"  failed:       {n_failed}")
    print(f"  skipped:      {n_skipped}")
    print(f"  budget_skip:  {n_budget}")
    print(f"  total cost:   ${total_cost:.4f}")
    print(f"  out_dir:      {out_dir}")
    return 1 if n_failed > 0 else 0


def _yaml_dump(obj: dict) -> str:
    """YAML dump that preserves a stable key order for audits."""
    import yaml

    return yaml.dump(obj, sort_keys=True, default_flow_style=False)


if __name__ == "__main__":
    sys.exit(main())
