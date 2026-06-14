#!/usr/bin/env python3
"""
audit_run.py — verify that a completed run's manifest is internally consistent.

Usage::

    python scripts/audit_run.py <run_id> [--results-dir <dir>] [--repo-root <dir>]

Steps:
1. Locate ``$RESULTS_DIR/<run_id>/manifest.jsonl`` and ``result.json``.
2. Read ``git_commit`` from ``result.json``; warn if current HEAD differs.
3. Call ``replay()``, print summary (entries, total cost, hash drift count).
4. Verify that ``total_cost_usd`` in ``result.json`` equals the sum of
   ``cost_usd`` across all manifest entries (within floating-point tolerance).

Exit codes:
    0 — all checks passed
    1 — hash drift or cost mismatch detected
    2 — manifest or result.json not found
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path resolution: allow running as ``python scripts/audit_run.py`` from
# the repo root without installing the package.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
sys.path.insert(0, str(_REPO_ROOT / "StableToolBench"))

from toolbench.observability.manifest import read_manifest  # noqa: E402
from toolbench.observability.replay import replay  # noqa: E402
from toolbench.observability.result_schema import read_result_json  # noqa: E402

logger = logging.getLogger("audit_run")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _results_dir_default() -> Path:
    """Return the default results directory from env or a sensible fallback."""
    env_val = os.environ.get("RESULTS_DIR")
    if env_val:
        return Path(env_val)
    return _REPO_ROOT / "results"


def _current_head(repo_root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_audit(
    run_id: str,
    results_dir: Path,
    repo_root: Path,
) -> int:
    """Execute the audit and return an exit code (0 = pass, 1+ = fail)."""
    run_dir = results_dir / run_id
    manifest_path = run_dir / "manifest.jsonl"
    result_path = run_dir / "result.json"

    # -- Locate files ------------------------------------------------------
    if not manifest_path.exists():
        logger.error("manifest.jsonl not found: %s", manifest_path)
        return 2
    if not result_path.exists():
        logger.error("result.json not found: %s", result_path)
        return 2

    # -- Read result.json --------------------------------------------------
    result = read_result_json(result_path)
    recorded_commit = result.get("git_commit", "")
    recorded_cost = float(result.get("total_cost_usd", 0.0))

    current_head = _current_head(repo_root)
    if current_head and recorded_commit:
        if not (
            current_head.startswith(recorded_commit)
            or recorded_commit.startswith(current_head)
        ):
            logger.warning(
                "HEAD mismatch: result.json records commit %s but current HEAD is %s. "
                "The repo has moved on since this run.",
                recorded_commit,
                current_head,
            )
        else:
            logger.info("Git commit matches current HEAD: %s", recorded_commit[:12])

    # -- Replay manifest ---------------------------------------------------
    report = replay(
        manifest_path=manifest_path,
        repo_root=repo_root,
        git_commit_check=True,
    )
    print(str(report))

    # -- Cost reconciliation -----------------------------------------------
    entries = read_manifest(manifest_path)
    manifest_cost_sum = sum(e.cost_usd for e in entries)
    tolerance = 1e-4  # $0.0001 floating-point tolerance

    if abs(manifest_cost_sum - recorded_cost) > tolerance:
        logger.error(
            "Cost mismatch: result.json says $%.6f but manifest sums to $%.6f "
            "(diff $%.6f)",
            recorded_cost,
            manifest_cost_sum,
            abs(manifest_cost_sum - recorded_cost),
        )
        return 1
    else:
        logger.info(
            "Cost reconciled: result.json=$%.6f  manifest_sum=$%.6f  diff=$%.8f",
            recorded_cost,
            manifest_cost_sum,
            abs(manifest_cost_sum - recorded_cost),
        )

    # -- Final verdict -----------------------------------------------------
    if not report.ok:
        logger.error(
            "Audit FAILED: %d hash drift(s) detected. "
            "See report above for details.",
            report.hash_drift_count,
        )
        return 1

    logger.info(
        "Audit PASSED: run_id=%s  entries=%d  cost=$%.4f  drift=0",
        run_id,
        report.total_entries,
        report.total_cost_usd,
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="audit_run",
        description=(
            "Verify that a completed run's manifest is internally consistent "
            "and matches result.json provenance fields."
        ),
    )
    parser.add_argument("run_id", help="Run identifier (name of the run directory).")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=(
            "Root directory containing run subdirectories. "
            "Defaults to $RESULTS_DIR env var or <repo_root>/results."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=_REPO_ROOT,
        help="Root of the git repository (default: parent of this script's directory).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    results_dir = args.results_dir or _results_dir_default()
    sys.exit(run_audit(args.run_id, results_dir, args.repo_root))


if __name__ == "__main__":
    main()
