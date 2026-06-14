#!/usr/bin/env python3
"""Single CLI entry point for FitText experiment runs.

Usage:
    python run.py --config configs/runs/gpt5_memetic_stb.yaml
    python run.py --config configs/runs/reproduce_paper.yaml --dry-run
    python run.py --config configs/runs/gpt41mini_memetic_stb.yaml \\
                  --shard 0 --total-shards 2 \\
                  --cache-dir "${FITTEXT_RUNTIME_ROOT:-runs}/$JOB_ID/cache" \\
                  --out "${FITTEXT_RUNTIME_ROOT:-runs}/$JOB_ID/results" \\
                  --max-cost-usd 50

CLI flags override YAML config fields. See toolbench/runner/schema.py for
the full RunConfig schema.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import yaml

# Ensure repo root is on the path when run directly; StableToolBench/ holds
# the inference + observability portions of the namespace package.
_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "StableToolBench"))

from toolbench.runner import resolve_config, dump_resolved_config
from toolbench.runner.schema import RunConfig
from toolbench.inference.LLM.clients import make_client
from toolbench.observability import ManifestWriter
from toolbench.observability.budget_guard import BudgetGuard
from toolbench.observability.result_schema import write_result_json
from toolbench.runner.executor import execute_run, aggregate_manifest


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git_commit(repo_root: Path) -> str:
    """Return the current HEAD commit SHA, or 'unknown' on failure."""
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _utc_now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ---------------------------------------------------------------------------
# Run execution
# ---------------------------------------------------------------------------

def _execute_run(cfg: RunConfig, dry_run: bool) -> int:
    """Execute the experiment described by ``cfg``.

    Validates integration plumbing (ModelClient, ManifestWriter, BudgetGuard,
    result.json provenance) and dispatches to the configured benchmark adapter.

    Args:
        cfg: Fully resolved RunConfig.
        dry_run: If True, print config and exit without launching the run.

    Returns:
        Exit code (0 = success).
    """
    summary = cfg.one_line_summary()
    log.info("run_id=%s  config_hash=%s", cfg.run_id, cfg.config_hash())
    log.info("RESOLVED CONFIG: %s", summary)

    # Print full resolved config as YAML to stdout
    resolved_yaml = yaml.dump(cfg.model_dump(), default_flow_style=False, sort_keys=True)
    print("--- resolved config ---")
    print(resolved_yaml)
    print("--- end resolved config ---")

    if dry_run:
        log.info("DRY RUN — exiting cleanly. No experiment launched.")
        return 0

    # Persist resolved config next to results
    out_dir = Path(cfg.infra.output_dir) / cfg.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_resolved_config(cfg, out_dir / "config.resolved.yaml")
    log.info("Resolved config written to %s", out_dir / "config.resolved.yaml")

    git_commit = _git_commit(_REPO_ROOT)
    manifest_path = out_dir / "manifest.jsonl"
    started_at = _utc_now_iso()
    import time as _time
    t0 = _time.monotonic()

    # Construct the ModelClient (validates factory + provider routing). Skip
    # the actual network round-trip — no API key may be set in the env.
    # Cache is enabled when cfg.infra.cache_dir is set; experiments opt in
    # via YAML or --cache-dir CLI flag.
    try:
        client = make_client(
            cfg.model.name,
            cache_dir=cfg.infra.cache_dir if cfg.infra.cache_dir else None,
        )
        log.info("ModelClient OK: %s (%s)", cfg.model.name, type(client).__name__)
        if getattr(client, "_cache", None) is not None:
            log.info("ResponseCache armed: dir=%s", cfg.infra.cache_dir)
    except NotImplementedError as e:
        log.error("ModelClient factory rejected model %r: %s", cfg.model.name, e)
        return 2

    # Open the manifest writer + budget guard for the duration of the run.
    # The executor (toolbench.runner.executor.execute_run) installs the
    # chat_completion telemetry hook so every LLM call is recorded.
    budget = BudgetGuard(run_dir=out_dir, max_cost_usd=cfg.budget.max_cost_usd)
    import asyncio  # local import — keeps top-level deps minimal
    report = None
    with ManifestWriter(manifest_path, run_id=cfg.run_id) as mw:
        log.info("ManifestWriter open: %s", manifest_path)
        log.info("BudgetGuard armed: max_cost_usd=%.2f", cfg.budget.max_cost_usd)
        try:
            report = asyncio.run(
                execute_run(
                    cfg,
                    run_dir=out_dir,
                    mw=mw,
                    budget=budget,
                    client=client,
                    tracer_factory=None,
                    git_commit=git_commit,
                )
            )
        except NotImplementedError as exc:
            log.error("execute_run rejected benchmark %r: %s", cfg.benchmark.name, exc)
            return 3
        n_entries = mw._count if hasattr(mw, "_count") else 0

    # Aggregate cost/token totals from the manifest (single source of truth).
    totals = aggregate_manifest(manifest_path)

    wall_clock_s = _time.monotonic() - t0
    finished_at = _utc_now_iso()

    # Log response cache statistics (hits, misses, hit rate)
    cache = getattr(client, "_cache", None)
    if cache is not None:
        cstats = cache.stats()
        total_lookups = cstats["hits"] + cstats["misses"]
        hit_rate = cstats["hits"] / total_lookups if total_lookups > 0 else 0.0
        log.info(
            "ResponseCache stats: hits=%d misses=%d puts=%d hit_rate=%.1f%% bytes_written=%d",
            cstats["hits"],
            cstats["misses"],
            cstats["puts"],
            hit_rate * 100,
            cstats["bytes_written"],
        )

    write_result_json(
        run_dir=out_dir,
        run_id=cfg.run_id,
        git_commit=git_commit,
        config_hash=cfg.config_hash(),
        manifest_path=manifest_path,
        agent_model=cfg.model.name,
        agent_model_revision=cfg.model.revision or "",
        eval_judge_model=cfg.evaluator.judge_model,
        eval_judge_revision=cfg.evaluator.judge_revision,
        eval_simulator_model=cfg.evaluator.simulator_model,
        eval_simulator_revision=cfg.evaluator.simulator_revision,
        embedder_model=cfg.embedder.name,
        embedder_revision=cfg.embedder.revision,
        total_cost_usd=float(totals["total_cost_usd"]),
        total_input_tokens=int(totals["total_input_tokens"]),
        total_cached_input_tokens=int(totals["total_cached_input_tokens"]),
        total_output_tokens=int(totals["total_output_tokens"]),
        wall_clock_s=wall_clock_s,
        started_at=started_at,
        finished_at=finished_at,
        seed=cfg.model.seed,
        n_queries=report.n_queries if report else 0,
        n_successful=report.n_successful if report else 0,
        n_failed=report.n_failed if report else 0,
        extra={
            "executor_status": "OK" if report and not report.budget_aborted else (
                "BUDGET_ABORTED" if report and report.budget_aborted else "NO_REPORT"
            ),
            "manifest_entries": n_entries,
            "budget_aborted": bool(report.budget_aborted) if report else False,
        },
    )
    log.info("result.json written to %s", out_dir / "result.json")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="FitText experiment runner — config-driven entry point.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to run YAML config (e.g. configs/runs/reproduce_paper.yaml).",
    )
    parser.add_argument(
        "--shard",
        type=int,
        default=None,
        help="Override infra.shard (0-indexed). For cross-host splitting.",
    )
    parser.add_argument(
        "--total-shards",
        type=int,
        default=None,
        dest="total_shards",
        help="Override infra.total_shards.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        dest="cache_dir",
        help="Override infra.cache_dir. Recommend ${FITTEXT_RUNTIME_ROOT:-runs}/$JOB_ID/cache.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Override infra.output_dir.",
    )
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=None,
        dest="max_cost_usd",
        help="Override budget.max_cost_usd circuit-breaker.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print resolved config and exit. No experiment launched.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main entry point.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Exit code.
    """
    args = _parse_args(argv)

    # Resolve the config
    repo_root = Path(__file__).parent
    cfg = resolve_config(args.config, repo_root=repo_root)

    # Apply CLI overrides (flags take precedence over YAML)
    infra_overrides: dict = {}
    if args.shard is not None:
        infra_overrides["shard"] = args.shard
    if args.total_shards is not None:
        infra_overrides["total_shards"] = args.total_shards
    if args.cache_dir is not None:
        infra_overrides["cache_dir"] = args.cache_dir
    if args.out is not None:
        infra_overrides["output_dir"] = args.out
    if infra_overrides:
        new_infra = cfg.infra.model_copy(update=infra_overrides)
        cfg = cfg.model_copy(update={"infra": new_infra})

    if args.max_cost_usd is not None:
        new_budget = cfg.budget.model_copy(update={"max_cost_usd": args.max_cost_usd})
        cfg = cfg.model_copy(update={"budget": new_budget})

    return _execute_run(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
