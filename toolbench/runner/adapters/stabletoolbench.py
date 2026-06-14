"""StableToolBench eval adapter.

The full StableToolBench pipeline drives DFSDT against the live RapidAPI
service.  That requires a TOOLBENCH_KEY, RapidAPI credentials, and many
flags hardcoded into the legacy ``rapidapi_multithread`` driver.  For the
retrieval-quality benchmarks we care about (NDCG@5,
recall, comprehensiveness, **and** the Memetic belief-trace plots), what
we actually need is the strategy dispatch path from
``StableToolBench/toolbench/inference/Downstream_tasks/strategies.py`` —
which produces retrieved tool sets per query without needing a live
RapidAPI loop.

This adapter therefore exposes the **retrieval-only slice** of STB:

  1. Load the input queries (G1/G2/G3 instruction files).
  2. Construct a minimal ``rapidapi_wrapper``-shaped object with the
     attributes ``select_and_run_strategy`` and ``run_memetic_strategy``
     actually read.
  3. Call ``select_and_run_strategy(llm_output, wrapper, llm)`` per query.
  4. Aggregate retrieved tools and write to ``stb_results.jsonl``.

Variant mapping (§5.5):

  - ``single_pass`` → no flags  → ``run_single_pass_strategy`` (router default)
  - ``multi_turn``  → ``dbd=True``
  - ``scattershot`` → ``scattershot=True``
  - ``memetic``     → ``memetic=True`` (published population-based variant)
  - ``just_query``  → bypass strategy, pure top-k retrieval

When ``tracer_factory`` is provided, belief snapshots are emitted by the
memetic strategy hooks. The wrapper exposes ``belief_trace_dir`` / ``run_id`` /
``query_id`` for that purpose.

The full DFSDT-driven pass-rate path is handled by
``StableToolBench/scripts/run_inference.sh`` followed by
``StableToolBench/scripts/run_evaluation.sh``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
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
    attach_sinks,
    install_chat_completion_hook,
    reset_call_context,
    set_call_context,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Variant → wrapper flag mapping
# ---------------------------------------------------------------------------


def _variant_to_wrapper_flags(variant: str) -> dict[str, bool]:
    """Return the wrapper attribute flags that the STB router checks.

    Mirrors :func:`StableToolBench.toolbench.inference.Downstream_tasks.strategies.select_and_run_strategy`.

    Args:
        variant: One of ``single_pass``, ``multi_turn``, ``scattershot``,
            ``memetic``, ``just_query``.

    Returns:
        Dict of boolean wrapper attributes to set.
    """
    return {
        "single_pass": {"dbd": False, "scattershot": False, "memetic": False},
        "multi_turn":  {"dbd": True,  "scattershot": False, "memetic": False},
        "scattershot": {"dbd": False, "scattershot": True,  "memetic": False},
        "memetic":     {"dbd": False, "scattershot": False, "memetic": True},
        "just_query":  {"dbd": False, "scattershot": False, "memetic": False},
    }[variant]


# ---------------------------------------------------------------------------
# Lightweight wrapper object
# ---------------------------------------------------------------------------


class _StbStrategyWrapper:
    """Minimal stand-in for the legacy ``rapidapi_wrapper``.

    Only the attributes that
    ``Downstream_tasks/strategies.run_*_strategy`` actually read are
    populated — enough to drive retrieval-only Memetic / Scattershot /
    Multi-Turn / Single-Pass dispatch without spinning up a full
    DFSDT environment.

    Mutable ``tool_memory`` is a dict (see strategies.py:1140).
    """

    def __init__(self, *, cfg: RunConfig, query: str, retriever: Any, tool_root_dir: str) -> None:
        flags = _variant_to_wrapper_flags(cfg.fittext.variant)
        for k, v in flags.items():
            setattr(self, k, v)

        # Common fields read by strategies.run_*_strategy
        self.input_description = query
        self.retriever = retriever
        self.retrieved_api_nums = int(cfg.fittext.top_k_retrieval)
        self.tool_root_dir = tool_root_dir
        self.tool_memory: dict[str, bool] = {}
        self.process_id = 0

        # Strategy hyperparameters lifted from cfg.fittext (§5.5 unification)
        self.population_size = int(cfg.fittext.population_size)
        self.generation_num = int(cfg.fittext.generations)
        # Memetic v1 fitness — retrieval-only with Jaccard memory penalty.
        # alpha/beta are unused by v1 (kept here for backwards-compat with any
        # consumer that still reads them); gamma maps to memory_penalty.
        # The v1 path's only fitness knob is the memory_penalty (§5.4 / Eq. 7).
        self.fitness_alpha = float(cfg.fittext.fitness_alpha)
        self.fitness_beta = float(1.0 - cfg.fittext.fitness_alpha) * 0.5
        self.fitness_gamma = float(cfg.fittext.memory_penalty)
        self.similarity_threshold = 0.95
        self.temp_schedule = "anneal"
        self.selection_method = "tournament" if cfg.fittext.selection == "fitness" else "random"
        self.tournament_size = 3
        self.base_temp = 0.9
        # Evolution-subprocess temperature override (split-temperatures track,
        # 2026-05-24). When ``cfg.model.evolution_temperature`` is set, the
        # memetic strategy's evolutionary call sites — population seeding,
        # mutation, crossover, LLM refinement — use this T instead of
        # ``base_temp``. When None, the legacy ``base_temp=0.9`` path is
        # preserved bit-identical for back-compat with the 231 existing
        # memetic result dirs.
        self.evolution_temperature = cfg.model.evolution_temperature
        self.final_tool_budget = int(cfg.fittext.top_k_retrieval)
        self.top_k_refine = int(cfg.fittext.top_k_retrieval)

        # Pluggable belief-fitness scorer. The strategies.py
        # ``run_memetic_strategy`` closure reads this attribute to pick the
        # active scorer.
        self.fitness_method = str(cfg.fittext.fitness_method)

        # Multi-Turn / Scattershot tunables
        self.refinement = cfg.fittext.variant in ("multi_turn", "memetic")
        self.dbd_refine_turns = int(cfg.fittext.generations)
        self.scattershot_size = int(cfg.fittext.population_size)
        self.size = int(cfg.fittext.population_size)

        # Instrumentation slots consumed by the memetic strategy.
        self.belief_trace_dir: Path | None = None
        self.run_id: str | None = cfg.run_id
        self.query_id: str | None = None


# ---------------------------------------------------------------------------
# STB query iteration
# ---------------------------------------------------------------------------


@dataclass
class _StbQuery:
    """One StableToolBench query (G1/G2/G3 split)."""

    qid: str
    query: str
    api_list: list[dict[str, Any]]
    split: str


def _stb_data_root(cfg: RunConfig) -> Path:
    """Resolve the StableToolBench data directory."""
    root = cfg.extra.get("stb_data_root") if isinstance(cfg.extra, dict) else None
    if not root:
        root = os.path.join(
            os.getenv("WORKSPACE_ROOT", os.getcwd()),
            "StableToolBench",
            "data",
        )
    return Path(root)


def _iter_stb_queries(
    *,
    cfg: RunConfig,
    split: str,
    n_queries: int | None,
    shard: int,
    total_shards: int,
) -> list[_StbQuery]:
    """Load and shard queries for one StableToolBench split.

    Args:
        cfg: Full run config (for resolving the data root).
        split: ``G1`` / ``G2`` / ``G3``.
        n_queries: Max queries to take (post-shard).
        shard: 0-indexed shard.
        total_shards: Total number of shards.
    """
    data_root = _stb_data_root(cfg)
    # Expected layout per StableToolBench docs:
    # ${data_root}/test_query_ids/{G1,G2,G3}_instruction.json
    cand_paths = [
        data_root / "test_query_ids" / f"{split}_instruction.json",
        data_root / "test_instruction" / f"{split}_instruction.json",
        data_root / f"{split}_instruction.json",
    ]
    src: Path | None = None
    for p in cand_paths:
        if p.exists():
            src = p
            break
    if src is None:
        log.warning("STB split %s: instruction file not found under %s", split, data_root)
        return []

    try:
        with open(src, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("STB split %s: failed to load %s: %s", split, src, exc)
        return []

    out: list[_StbQuery] = []
    for entry in raw:
        qid = str(entry.get("query_id") or entry.get("id") or len(out))
        if total_shards > 1 and (hash(qid) % total_shards) != shard:
            continue
        out.append(
            _StbQuery(
                qid=qid,
                query=str(entry.get("query", "")),
                api_list=list(entry.get("api_list", [])),
                split=split,
            )
        )
        if n_queries is not None and len(out) >= n_queries:
            break
    return out


# ---------------------------------------------------------------------------
# Per-query worker
# ---------------------------------------------------------------------------


def _run_one_stb_query(
    *,
    q: _StbQuery,
    cfg_dict: dict[str, Any],
    run_dir_str: str,
    plan_model: str,
    belief_trace_dir: str | None,
) -> dict[str, Any]:
    """Run one STB query and write its result + sentinel.

    Threading-safe (called from a ThreadPoolExecutor).  Per-query call
    context is set so the manifest hook records correct provenance.

    Args:
        q: The STB query.
        cfg_dict: Resolved-config dict (model_dump()).
        run_dir_str: Run output directory.
        plan_model: Planner LLM identifier.
        belief_trace_dir: Optional directory for per-query belief JSONL.
    """
    run_dir = Path(run_dir_str)
    variant = cfg_dict["fittext"]["variant"]
    qid_full = f"stb:{q.split}:{q.qid}"

    if is_query_done(run_dir, qid_full):
        return {"qid": q.qid, "ok": True, "skipped": True}

    # Late imports
    from StableToolBench.toolbench.inference.LLM.chatgpt_function_model import ChatGPTFunction  # type: ignore
    from StableToolBench.toolbench.inference.LLM.retriever import ToolRetriever  # type: ignore
    from StableToolBench.toolbench.inference.Downstream_tasks.strategies import (  # type: ignore
        select_and_run_strategy,
    )

    install_chat_completion_hook()
    token = set_call_context(
        qid=qid_full,
        operation="dfsdt_node",
        variant=variant,
        generation=0,
    )

    try:
        retriever = _get_or_build_stb_retriever(
            corpus_tsv_path=os.path.join(
                cfg_dict["extra"].get("stb_corpus_root", "./data/retrieval/StableToolBench"),
                q.split,
                "corpus.tsv",
            ),
            model_path=cfg_dict["embedder"]["name"],
        )

        wrapper = _StbStrategyWrapper(
            cfg=_dict_to_run_config(cfg_dict),
            query=q.query,
            retriever=retriever,
            tool_root_dir=cfg_dict["extra"].get("stb_tool_root", os.getenv("TOOL_ROOT_DIR", "")),
        )
        if belief_trace_dir:
            wrapper.belief_trace_dir = Path(belief_trace_dir)
        wrapper.query_id = q.qid

        # just_query bypasses the strategy router (zero-LLM baseline)
        if variant == "just_query":
            api_keys, retrieval_iterations = _run_just_query(wrapper)
            payload: dict[str, Any] = {"strategy": "just_query"}
        else:
            llm = ChatGPTFunction(
                model=plan_model,
                openai_key=os.getenv("OPENAI_API_KEY", ""),
                base_url=cfg_dict["extra"].get("base_url_plan"),
            )
            # The STB strategy router expects an llm_output string of
            # ``<func_desc>...</func_desc>`` blocks — produce one via the LLM
            # (one shot, no retries here; tenacity inside the client handles it).
            llm.change_messages([
                {"role": "system", "content": "Generate one or more <func_desc> blocks describing the tools needed."},
                {"role": "user", "content": q.query},
            ])
            msg, _, _ = llm.parse(tools=[], process_id=0)
            llm_output = (msg or {}).get("content", "") or q.query

            ret = select_and_run_strategy(llm_output, wrapper, llm)
            # select_and_run_strategy returns (api_keys, retrieval_iterations) or
            # a 3-tuple including payload for memetic / scattershot variants.
            if len(ret) == 3:
                api_keys, retrieval_iterations, payload = ret
            else:
                api_keys, retrieval_iterations = ret
                payload = {}

        result = {
            "qid": q.qid,
            "split": q.split,
            "ok": True,
            "variant": variant,
            "api_keys": api_keys,
            "iterations": retrieval_iterations[:20],   # cap to keep result.json small
            "payload_keys": list(payload.keys()) if payload else [],
        }
        mark_query_done(run_dir, qid_full, result)
        return result
    except Exception as exc:
        log.warning("stb query %s failed: %s", q.qid, exc)
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
# Helpers
# ---------------------------------------------------------------------------


_stb_retriever_cache: dict[tuple[str, str], Any] = {}


def _get_or_build_stb_retriever(*, corpus_tsv_path: str, model_path: str) -> Any:
    """Load (or reuse) a :class:`ToolRetriever` for STB corpus.

    Args:
        corpus_tsv_path: Path to ``corpus.tsv``.
        model_path: HF model name or local path.
    """
    key = (corpus_tsv_path, model_path)
    if key in _stb_retriever_cache:
        return _stb_retriever_cache[key]
    from StableToolBench.toolbench.inference.LLM.retriever import ToolRetriever  # type: ignore

    retriever = ToolRetriever(corpus_tsv_path=corpus_tsv_path, model_path=model_path)
    _stb_retriever_cache[key] = retriever
    return retriever


def _run_just_query(wrapper: _StbStrategyWrapper) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Zero-LLM baseline: retrieve top-k tools directly from the query.

    Returns:
        ``(api_keys, retrieval_iterations)`` tuple matching the strategy
        router contract.
    """
    from StableToolBench.toolbench.inference.Downstream_tasks.strategies import (  # type: ignore
        retrieve_rapidapi_tools,
    )

    qj, mt = retrieve_rapidapi_tools(
        retriever=wrapper.retriever,
        query=wrapper.input_description,
        top_k=wrapper.retrieved_api_nums,
        tool_root_dir=wrapper.tool_root_dir,
    )
    api_keys = [
        {
            "category_name": it.get("category_name"),
            "tool_name": it.get("tool_name"),
            "api_name": it.get("api_name"),
            "lineage_index": 0,
        }
        for it in (qj.get("api_list") or [])
    ]
    iterations = [
        {"phase": "just_query", "retrieved_tools": mt},
    ]
    return api_keys, iterations


def _dict_to_run_config(cfg_dict: dict[str, Any]) -> RunConfig:
    """Re-instantiate a RunConfig from a dict (worker boundary helper)."""
    return RunConfig.model_validate(cfg_dict)


# ---------------------------------------------------------------------------
# Adapter entry point
# ---------------------------------------------------------------------------


async def run_stb(
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
    """Run the StableToolBench retrieval-only path.

    Args:
        cfg: Resolved :class:`RunConfig`.
        run_dir: Output directory for this run.
        mw: Active :class:`ManifestWriter`.
        budget: Active :class:`BudgetGuard`.
        client: Unused at this layer (legacy ChatGPTFunction handles its own).
        tracer_factory: When non-None, belief snapshots are emitted
            (memetic v1 only).
        git_commit: Git SHA for manifest entries.
        report: :class:`ExecutionReport` to mutate in-place.
    """
    install_chat_completion_hook()
    attach_sinks(manifest_writer=mw, budget_guard=budget)

    set_call_context(
        run_id=cfg.run_id or "unknown_run",
        git_commit=git_commit,
        config_hash=cfg.config_hash(),
        variant=cfg.fittext.variant,
    )

    cfg_dict = cfg.model_dump()
    plan_model = cfg.model.name
    belief_trace_dir: str | None = None
    if tracer_factory is not None:
        belief_trace_dir = str(Path(run_dir) / "beliefs")
        Path(belief_trace_dir).mkdir(parents=True, exist_ok=True)

    queries: list[_StbQuery] = []
    for split in cfg.benchmark.splits:
        per_split = _iter_stb_queries(
            cfg=cfg,
            split=split,
            n_queries=cfg.benchmark.n_queries_per_split,
            shard=cfg.infra.shard,
            total_shards=cfg.infra.total_shards,
        )
        queries.extend(per_split)
    log.info("stb: loaded %d queries across %d splits", len(queries), len(cfg.benchmark.splits))
    report.n_queries = len(queries)

    max_workers = min(len(queries) or 1, max(1, (os.cpu_count() or 4) // 2))
    log.info("stb: dispatching with max_workers=%d", max_workers)

    loop = asyncio.get_running_loop()
    pool = ThreadPoolExecutor(max_workers=max_workers)
    futures = []
    try:
        for q in queries:
            futures.append(
                loop.run_in_executor(
                    pool,
                    lambda q=q: _run_one_stb_query(
                        q=q,
                        cfg_dict=cfg_dict,
                        run_dir_str=str(run_dir),
                        plan_model=plan_model,
                        belief_trace_dir=belief_trace_dir,
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

    summary_path = Path(run_dir) / "stb_results.jsonl"
    with open(summary_path, "w", encoding="utf-8") as fh:
        for r in report.query_results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("stb: results dumped to %s", summary_path)
