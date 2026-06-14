"""
Helper for writing provenance-rich ``result.json`` files.

Every completed run writes one ``result.json`` to its output directory.  Every
field in this file can appear in report tables (as footnote-linked appendix rows
when not in the main table). The exact set of provenance fields makes audits
one-pass.

Usage::

    from toolbench.observability.result_schema import write_result_json

    write_result_json(
        run_dir=Path("runs/example"),
        run_id="20260524T012345_a1b2c3d4",
        git_commit="a1b2c3d4e5f6...",
        config_hash="sha256...",
        manifest_path=Path("runs/example/manifest.jsonl"),
        agent_model=agent_pin,
        agent_model_revision=agent_revision,
        eval_judge_model=judge_pin,
        eval_judge_revision=judge_revision,
        eval_simulator_model=simulator_pin,
        eval_simulator_revision=simulator_revision,
        embedder_model="text-embedding-3-large",
        embedder_revision="1",
        total_cost_usd=12.34,
        total_input_tokens=1_000_000,
        total_cached_input_tokens=900_000,
        total_output_tokens=50_000,
        wall_clock_s=3600.0,
        started_at="2026-05-24T00:00:00Z",
        finished_at="2026-05-24T01:00:00Z",
        seed=42,
        n_queries=1000,
        n_successful=980,
        n_failed=20,
    )
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RESULT_FILENAME = "result.json"


def write_result_json(
    run_dir: Path,
    *,
    run_id: str,
    git_commit: str,
    config_hash: str,
    manifest_path: Path,
    # Model provenance
    agent_model: str,
    agent_model_revision: str,
    eval_judge_model: str,
    eval_judge_revision: str,
    eval_simulator_model: str,
    eval_simulator_revision: str,
    embedder_model: str,
    embedder_revision: str,
    # Cost / token summary (pulled from manifest aggregation or passed directly)
    total_cost_usd: float,
    total_input_tokens: int,
    total_cached_input_tokens: int,
    total_output_tokens: int,
    # Timing
    wall_clock_s: float,
    started_at: str,
    finished_at: str,
    # Run statistics
    seed: int,
    n_queries: int,
    n_successful: int,
    n_failed: int,
    # Optional extra fields (benchmark-specific metrics, etc.)
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write ``result.json`` with full provenance fields.

    Args:
        run_dir: Output directory for this run.  Created if absent.
        run_id: Unique run identifier.
        git_commit: Full git SHA at time of the run.
        config_hash: SHA-256 of the resolved run config YAML.
        manifest_path: Absolute path to the manifest JSONL file.
        agent_model: Full model identifier of the FitText agent.
        agent_model_revision: Dated revision tag of the agent model.
        eval_judge_model: Full model identifier of the eval judge.
        eval_judge_revision: Dated revision tag of the judge model.
        eval_simulator_model: Full model identifier of the tool simulator.
        eval_simulator_revision: Dated revision tag of the simulator model.
        embedder_model: Full model identifier of the retrieval embedder.
        embedder_revision: Version or revision of the embedder.
        total_cost_usd: Total USD cost aggregated from the manifest.
        total_input_tokens: Total non-cached prompt tokens.
        total_cached_input_tokens: Total cached prompt tokens.
        total_output_tokens: Total completion tokens.
        wall_clock_s: Wall-clock duration of the run in seconds.
        started_at: ISO-8601 UTC start time string.
        finished_at: ISO-8601 UTC finish time string.
        seed: Random seed used for the run.
        n_queries: Total number of queries attempted.
        n_successful: Number of queries with a successful result.
        n_failed: Number of queries that failed or were skipped.
        extra: Optional dict of additional benchmark-specific metrics to merge.

    Returns:
        Path to the written ``result.json`` file.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        # -- Run identity --------------------------------------------------
        "run_id": run_id,
        "git_commit": git_commit,
        "config_hash": config_hash,
        "manifest_path": str(manifest_path),
        # -- Model provenance ----------------------------------------------
        "agent_model": agent_model,
        "agent_model_revision": agent_model_revision,
        "eval_judge_model": eval_judge_model,
        "eval_judge_revision": eval_judge_revision,
        "eval_simulator_model": eval_simulator_model,
        "eval_simulator_revision": eval_simulator_revision,
        "embedder_model": embedder_model,
        "embedder_revision": embedder_revision,
        # -- Cost / token summary ------------------------------------------
        "total_cost_usd": total_cost_usd,
        "total_input_tokens": total_input_tokens,
        "total_cached_input_tokens": total_cached_input_tokens,
        "total_output_tokens": total_output_tokens,
        # -- Timing --------------------------------------------------------
        "wall_clock_s": wall_clock_s,
        "started_at": started_at,
        "finished_at": finished_at,
        # -- Run statistics ------------------------------------------------
        "seed": seed,
        "n_queries": n_queries,
        "n_successful": n_successful,
        "n_failed": n_failed,
    }

    if extra:
        payload.update(extra)

    out_path = run_dir / RESULT_FILENAME
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")

    logger.info("Wrote result.json: %s (run_id=%s, total_cost=$%.4f)", out_path, run_id, total_cost_usd)
    return out_path


def read_result_json(path: Path) -> dict[str, Any]:
    """Read a ``result.json`` file and return its contents as a dict.

    Args:
        path: Path to the ``result.json`` file.

    Returns:
        Dict of provenance fields.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is malformed.
    """
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
