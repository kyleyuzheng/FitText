"""
CLI driver for running a single retrieval baseline.

Usage::

    python scripts/run_baseline.py \\
      --baseline less_is_more|reinvoke|xu2024|colt \\
      --config configs/runs/<baseline_name>_<benchmark>.yaml \\
      --out "${FITTEXT_RUNTIME_ROOT:-runs}/$JOB_ID/results"

Follows the same run.py pattern as the FitText harness:
  - Loads RunConfig from YAML (validated by Pydantic schema).
  - Constructs ModelClient via make_client(cfg.model.name).
  - Constructs ManifestWriter + BudgetGuard.
  - Instantiates the selected Baseline.
  - Iterates over benchmark queries, calls baseline.retrieve().
  - Writes result.json (same schema as FitText harness output).
  - Writes manifest JSONL for cost-table aggregation.

All manifest entries use ``variant: baseline_<name>`` so the aggregator
groups them separately from FitText variants.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

# Ensure project root is on sys.path regardless of cwd
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
for _path in (_PROJECT_ROOT, _PROJECT_ROOT / "StableToolBench"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from toolbench.inference.LLM.clients import make_client
from toolbench.observability import ManifestWriter, BudgetGuard, BudgetExceeded
from toolbench.runner.schema import RunConfig

from baselines import (
    COLTBaseline,
    JustQueryBaseline,
    LessIsMoreBaseline,
    ReInvokeBaseline,
    Xu2024Baseline,
)
from baselines.base import Baseline, RetrieverAdapter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Baseline registry
# ---------------------------------------------------------------------------
# ``just_query`` is intentionally the zero-retrieval floor:
# no retrieval, no tool catalog, no critic. It is dispatched here through the
# same plumbing as the retrieval baselines so the cost-table aggregator picks
# it up uniformly under the variant tag ``baseline_just_query``.

_BASELINE_REGISTRY: dict[str, type[Baseline]] = {
    "less_is_more": LessIsMoreBaseline,
    "reinvoke": ReInvokeBaseline,
    "xu2024": Xu2024Baseline,
    "colt": COLTBaseline,
    "just_query": JustQueryBaseline,
}


# ---------------------------------------------------------------------------
# Config loader (YAML -> RunConfig)
# ---------------------------------------------------------------------------

def load_run_config(config_path: Path) -> RunConfig:
    """Load and validate a RunConfig from a YAML file.

    Performs simple key-merge inheritance if ``inherit`` list is present.
    Full inheritance resolver is in ``toolbench.runner.resolver`` (Wave 1).
    This driver uses a lightweight fallback for standalone use.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Validated RunConfig instance.
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required: pip install pyyaml") from exc

    try:
        from toolbench.runner.resolver import resolve_config
        return resolve_config(config_path)
    except ImportError:
        logger.warning(
            "toolbench.runner.resolver not available — loading config without inheritance."
        )

    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    # Strip unsupported 'inherit' key before Pydantic parse
    raw.pop("inherit", None)
    return RunConfig.model_validate(raw)


# ---------------------------------------------------------------------------
# Query loader
# ---------------------------------------------------------------------------

def load_queries(cfg: RunConfig) -> list[dict]:
    """Load benchmark queries according to the RunConfig benchmark spec.

    Returns a list of query dicts, each with at minimum:
        qid (str), query (str), tool_catalog (list[dict])

    This is a stub — wires into the benchmark loaders used by the main harness.
    For standalone baseline runs, queries are loaded directly from the benchmark
    data directory.

    Args:
        cfg: Resolved RunConfig.

    Returns:
        List of query dicts.
    """
    # Attempt to import the benchmark loader from the main harness
    try:
        from toolbench.runner.benchmark_loader import load_benchmark_queries
        return load_benchmark_queries(cfg.benchmark)
    except ImportError:
        pass

    # Minimal fallback: load from a JSONL file next to the output dir
    # (for testing without the full harness installed)
    data_file = Path(cfg.infra.output_dir).parent / f"{cfg.benchmark.name}_queries.jsonl"
    if data_file.exists():
        queries = []
        with data_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    queries.append(json.loads(line))
        return queries

    raise RuntimeError(
        f"Cannot load benchmark queries for '{cfg.benchmark.name}'. "
        "Either install the full harness or provide a "
        f"'{data_file.name}' JSONL file at '{data_file.parent}'."
    )


# ---------------------------------------------------------------------------
# Retriever builder
# ---------------------------------------------------------------------------

def build_retriever(cfg: RunConfig) -> RetrieverAdapter:
    """Construct a RetrieverAdapter from the RunConfig embedder spec.

    Wraps ``ToolRetriever`` (local) or ``RemoteToolRetriever`` (server).

    Args:
        cfg: Resolved RunConfig.

    Returns:
        RetrieverAdapter wrapping the constructed retriever.
    """
    retriever_server_url = os.environ.get("RETRIEVER_SERVER_URL", "")
    if retriever_server_url:
        from StableToolBench.toolbench.inference.LLM.retriever import RemoteToolRetriever
        inner = RemoteToolRetriever(
            server_url=retriever_server_url,
            corpus_name=cfg.benchmark.name.upper(),
        )
        logger.info("Using RemoteToolRetriever at %s", retriever_server_url)
    else:
        corpus_path = os.environ.get(
            "CORPUS_PATH",
            f"StableToolBench/data/toolenv/tools/{cfg.benchmark.name}/des_corpus.json",
        )
        from StableToolBench.toolbench.inference.LLM.retriever import ToolRetriever
        inner = ToolRetriever(
            corpus_path=corpus_path,
            model_path=cfg.embedder.name,
            des_corpus=corpus_path.endswith(".json"),
        )
        logger.info("Using ToolRetriever with corpus %s", corpus_path)

    return RetrieverAdapter(inner)


# ---------------------------------------------------------------------------
# Git commit helper
# ---------------------------------------------------------------------------

def _git_commit() -> str:
    """Return the current git commit SHA (abbreviated)."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------------

async def run_baseline(
    baseline_name: str,
    config_path: Path,
    output_dir: Path,
) -> None:
    """Run a baseline over all benchmark queries and write results.

    Args:
        baseline_name: Key into ``_BASELINE_REGISTRY``.
        config_path: Path to run config YAML.
        output_dir: Directory to write result.json and manifest JSONL.
    """
    if baseline_name not in _BASELINE_REGISTRY:
        raise ValueError(
            f"Unknown baseline '{baseline_name}'. "
            f"Available: {list(_BASELINE_REGISTRY.keys())}"
        )

    # Load + validate config
    cfg = load_run_config(config_path)

    # Generate run identity
    run_id = cfg.run_id or f"{time.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
    git_sha = _git_commit()
    config_hash = cfg.config_hash()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "run_baseline: baseline=%s run_id=%s config=%s out=%s",
        baseline_name,
        run_id,
        config_path,
        output_dir,
    )

    # Observability: manifest + budget guard
    manifest_path = output_dir / "manifest.jsonl"
    manifest_writer = ManifestWriter(path=manifest_path, run_id=run_id)
    budget_guard = BudgetGuard(
        run_dir=output_dir,
        max_cost_usd=cfg.budget.max_cost_usd,
    )

    # LLM client (not used by COLT but required by interface)
    model_client = make_client(
        cfg.model.name,
        manifest_writer=manifest_writer,
        budget_guard=budget_guard,
        run_id=run_id,
        git_commit=git_sha,
        config_hash=config_hash,
        variant=f"baseline_{baseline_name}",
    )

    # Retriever
    retriever = build_retriever(cfg)

    # Instantiate baseline with all cfg-level kwargs
    baseline_cls = _BASELINE_REGISTRY[baseline_name]
    # Pass relevant cfg fields as kwargs so baselines can read them without
    # importing RunConfig directly.
    extra_kwargs: dict = {
        "temperature": cfg.model.temperature,
        "seed": cfg.model.seed,
        "max_tokens": cfg.model.max_tokens,
        "embedder_revision": cfg.embedder.revision or cfg.embedder.name,
        "embedder_name": cfg.embedder.name,
        "cache_dir": cfg.infra.cache_dir,
    }
    # Merge any baseline-specific overrides from cfg.extra
    extra_kwargs.update(cfg.extra)

    baseline = baseline_cls(
        model_client=model_client,
        retriever=retriever,
        top_k=cfg.fittext.top_k_retrieval,
        **extra_kwargs,
    )

    # Load queries
    queries = load_queries(cfg)
    logger.info("Loaded %d queries for %s", len(queries), cfg.benchmark.name)

    results: list[dict] = []

    for q in queries:
        qid = q.get("qid", f"{cfg.benchmark.name}:{len(results):05d}")
        query_text = q.get("query", "")
        tool_catalog = q.get("tool_catalog", [])

        try:
            budget_guard.check()
        except BudgetExceeded as exc:
            logger.warning("Budget exceeded at qid=%s — stopping run. %s", qid, exc)
            break

        t0 = time.monotonic()
        try:
            retrieved = await baseline.retrieve(query_text, tool_catalog=tool_catalog)
        except Exception as exc:
            logger.error("retrieve() failed for qid=%s: %s", qid, exc, exc_info=True)
            results.append({
                "qid": qid,
                "query": query_text,
                "error": str(exc),
                "tool_ids": [],
                "scores": [],
                "metadata": {},
            })
            continue

        latency_ms = (time.monotonic() - t0) * 1000.0
        results.append({
            "qid": qid,
            "query": query_text,
            "tool_ids": retrieved.tool_ids,
            "scores": retrieved.scores,
            "metadata": retrieved.metadata,
            "latency_ms": latency_ms,
        })

    # Write result.json
    result_file = output_dir / "result.json"
    result_file.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "baseline": baseline_name,
                "config_hash": config_hash,
                "git_commit": git_sha,
                "benchmark": cfg.benchmark.name,
                "n_queries": len(results),
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote %d results to %s", len(results), result_file)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI args and run the baseline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run a single FitText retrieval baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--baseline",
        required=True,
        choices=list(_BASELINE_REGISTRY.keys()),
        help="Which baseline to run.",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to run config YAML.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output directory for result.json and manifest.jsonl.",
    )
    args = parser.parse_args()

    asyncio.run(
        run_baseline(
            baseline_name=args.baseline,
            config_path=args.config,
            output_dir=args.out,
        )
    )


if __name__ == "__main__":
    main()
