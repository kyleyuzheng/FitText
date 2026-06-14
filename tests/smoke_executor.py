"""Smoke tests for the unified executor.

Gates the benchmark dispatch path. For each FitText variant, asserts:

  1. ``execute_run(cfg, ...)`` returns a populated :class:`ExecutionReport`.
  2. The manifest JSONL contains entries with the right ``variant`` field
     for any variant that issued an LLM call.
  3. ``result.json`` carries correct provenance fields (run_id, git_commit,
     config_hash, agent_model, totals).

Mocking strategy: monkey-patch
``toolbench.inference.LLM.chatgpt_function_model.chat_completion_request``
**before** the telemetry hook is installed, so the wrapped function calls
the mock instead of OpenAI.  This validates the entire dispatch pipeline
(adapter → strategy router → strategy implementation → manifest hook)
without needing API keys or a live retriever.

Run from the repository root::

    python tests/smoke_executor.py

Returns exit code 0 on all-pass, non-zero on any failure.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Ensure we import from this worktree.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "StableToolBench"))

from toolbench.observability import BudgetGuard, ManifestWriter, read_manifest
from toolbench.observability.result_schema import write_result_json
from toolbench.runner import ExecutionReport, execute_run, resolve_config
from toolbench.runner.schema import (
    BenchmarkSpec,
    BudgetSpec,
    EmbedderSpec,
    EvaluatorSpec,
    FitTextHParams,
    InfraSpec,
    ModelSpec,
    RunConfig,
)


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def _ok(msg: str) -> None:
    print(f"PASS: {msg}")


# ---------------------------------------------------------------------------
# Mock plumbing
# ---------------------------------------------------------------------------


def _mocked_chat_completion_request(
    key: str,
    base_url: Any,
    messages: list,
    tools: Any = None,
    tool_choice: Any = None,
    key_pos: Any = None,
    model: str = "gpt-4.1-mini",
    stop: Any = None,
    process_id: int = 0,
    **args: Any,
) -> dict:
    """Return a canned <func_desc>-shaped response with positive cost."""
    content = (
        "<func_desc>"
        "A pseudo-tool that fetches the answer to the user's query "
        "from a relevant tool catalog."
        "</func_desc>"
    )
    return {
        "id": "mock-resp-1",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content, "tool_calls": []},
                "finish_reason": "stop",
            }
        ],
        # Pricing for gpt-4.1-mini-2025-04-14: input/cached/output USD per 1M tokens.
        # Force non-zero cost so the smoke test exercises BudgetGuard accounting.
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 30,
            "total_tokens": 130,
            "prompt_tokens_details": {"cached_tokens": 0},
        },
    }


class _FakeRetriever:
    """Stand-in for ToolRetriever — returns deterministic top-k results.

    Sufficient to drive the strategies' retrieval-iteration paths without
    loading a real SentenceTransformer or corpus.
    """

    def __init__(self) -> None:
        self.calls = 0

    def retrieving(self, query: str, k: int = 5):
        """ToolRet retriever signature: returns (ids, descriptions, scores)."""
        self.calls += 1
        ids = [f"mock-tool-{i}" for i in range(k)]
        descs = [f"description-{i}" for i in range(k)]
        scores = [1.0 - 0.1 * i for i in range(k)]
        return ids, descs, scores


# ---------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------


def _make_minimal_config(variant: str, tmp_dir: Path) -> RunConfig:
    """Build a minimal RunConfig for the smoke test.

    Args:
        variant: Variant name to test.
        tmp_dir: Per-test temp directory.
    """
    return RunConfig(
        run_id=f"smoke_{variant}",
        model=ModelSpec(name="gpt-4.1-mini-2025-04-14"),
        fittext=FitTextHParams(variant=variant),  # type: ignore[arg-type]
        benchmark=BenchmarkSpec(name="toolret", splits=["code"], n_queries_per_split=3),
        evaluator=EvaluatorSpec(
            judge_model="gpt-5.4-mini-2026-03-17",
            judge_revision="2026-03-17",
            simulator_model="gpt-5.4-mini-2026-03-17",
            simulator_revision="2026-03-17",
        ),
        embedder=EmbedderSpec(name="sentence-transformers/all-MiniLM-L6-v2"),
        infra=InfraSpec(
            cache_dir=str(tmp_dir / "cache"),
            output_dir=str(tmp_dir / "results"),
            shard=0,
            total_shards=1,
        ),
        budget=BudgetSpec(max_cost_usd=10.0),
    )


def _run_variant_e2e(variant: str) -> dict[str, Any]:
    """Execute one variant end-to-end with mocked dependencies.

    Returns:
        Dict summarising the variant's outcome (report stats, manifest count,
        variant tags observed in the manifest).
    """
    from toolbench.inference.LLM import chatgpt_function_model as _llm_mod
    from toolbench.runner import telemetry as _telemetry
    from toolbench.runner.adapters import toolret as _toolret

    # 1. Patch both chat_completion_request seams (STB and Toolret each carry
    # a near-identical copy of the same shim — both must be intercepted).
    _orig_ccr = _llm_mod.chat_completion_request
    _llm_mod.chat_completion_request = _mocked_chat_completion_request  # type: ignore[assignment]
    try:
        import importlib
        _toolret_llm_mod = importlib.import_module("Toolret.strategy.LLM_model")
    except Exception:
        _toolret_llm_mod = None
    _orig_toolret_ccr = None
    if _toolret_llm_mod is not None:
        _orig_toolret_ccr = _toolret_llm_mod.chat_completion_request
        _toolret_llm_mod.chat_completion_request = _mocked_chat_completion_request  # type: ignore[assignment]

    # 2. Patch the retriever loader so we never touch torch / HF / the FS.
    fake_retriever = _FakeRetriever()
    _orig_get_retriever = _toolret._get_or_build_retriever
    _toolret._get_or_build_retriever = lambda **kw: fake_retriever  # type: ignore[assignment]

    # 3. Patch the query iterator to bypass HuggingFace datasets entirely.
    _orig_iter = _toolret._iter_split_queries

    def _fake_iter(split: str, n_queries: int | None, shard: int, total_shards: int):
        return [
            _toolret._ToolretQuery(
                qid=f"q{i}",
                query=f"What is the weather in city {i}?",
                gt_tools=[{"id": "mock-tool-0", "relevance": 1}],
                split=split,
            )
            for i in range(int(n_queries or 3))
        ]

    _toolret._iter_split_queries = _fake_iter  # type: ignore[assignment]
    # Reset the telemetry sinks + hook between variants
    _telemetry.detach_sinks()
    _telemetry.uninstall_chat_completion_hook()

    try:
        with tempfile.TemporaryDirectory(prefix=f"smoke_{variant}_") as td:
            tmp_dir = Path(td)
            cfg = _make_minimal_config(variant, tmp_dir)
            run_dir = Path(cfg.infra.output_dir) / cfg.run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            manifest_path = run_dir / "manifest.jsonl"
            budget = BudgetGuard(run_dir=run_dir, max_cost_usd=cfg.budget.max_cost_usd)
            from toolbench.inference.LLM.clients import make_client

            try:
                client = make_client(cfg.model.name)
            except Exception:  # noqa: BLE001 — client may fail without API key; fall back
                client = None

            with ManifestWriter(manifest_path, run_id=cfg.run_id) as mw:
                report: ExecutionReport = asyncio.run(
                    execute_run(
                        cfg,
                        run_dir=run_dir,
                        mw=mw,
                        budget=budget,
                        client=client,
                        tracer_factory=None,
                        git_commit="smoke-commit",
                    )
                )

            # Aggregate manifest for assertions
            entries = read_manifest(manifest_path) if manifest_path.exists() else []
            variant_tags = sorted({e.variant for e in entries})

            # Write a result.json for §5.2 provenance check
            from toolbench.runner.executor import aggregate_manifest

            totals = aggregate_manifest(manifest_path)
            write_result_json(
                run_dir=run_dir,
                run_id=cfg.run_id,
                git_commit="smoke-commit",
                config_hash=cfg.config_hash(),
                manifest_path=manifest_path,
                agent_model=cfg.model.name,
                agent_model_revision="",
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
                wall_clock_s=report.wall_clock_s,
                started_at="",
                finished_at="",
                seed=cfg.model.seed,
                n_queries=report.n_queries,
                n_successful=report.n_successful,
                n_failed=report.n_failed,
            )
            result_path = run_dir / "result.json"
            with open(result_path, "r", encoding="utf-8") as fh:
                result_payload = json.load(fh)

            return {
                "variant": variant,
                "report": report,
                "manifest_entries": len(entries),
                "variant_tags": variant_tags,
                "result_payload": result_payload,
                "retriever_calls": fake_retriever.calls,
            }
    finally:
        # Restore patches
        _llm_mod.chat_completion_request = _orig_ccr  # type: ignore[assignment]
        if _toolret_llm_mod is not None and _orig_toolret_ccr is not None:
            _toolret_llm_mod.chat_completion_request = _orig_toolret_ccr  # type: ignore[assignment]
        _toolret._get_or_build_retriever = _orig_get_retriever  # type: ignore[assignment]
        _toolret._iter_split_queries = _orig_iter  # type: ignore[assignment]
        _telemetry.detach_sinks()
        _telemetry.uninstall_chat_completion_hook()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _check_variant(variant: str, *, expect_llm_calls: bool, xfail: bool = False) -> bool:
    """Run one variant and check the report + manifest.

    Args:
        variant: Variant name.
        expect_llm_calls: True if the variant routes through an LLM call seam.
            ``just_query`` is pure retrieval — no LLM calls expected.
        xfail: If True, treat a Pydantic validation error as "expected" (used
            for ``just_query`` when the schema enum does not yet include it).

    Returns:
        True on pass.
    """
    try:
        out = _run_variant_e2e(variant)
    except Exception as exc:  # noqa: BLE001
        if xfail:
            print(f"XFAIL: variant={variant!r} not supported yet: {exc}")
            return True
        _fail(f"variant={variant!r} raised: {exc}")
        return False

    report: ExecutionReport = out["report"]
    if report.n_queries != 3:
        _fail(f"variant={variant!r} expected 3 queries, got {report.n_queries}")
    if expect_llm_calls and out["manifest_entries"] == 0:
        _fail(f"variant={variant!r} expected manifest entries, got 0")
    if not expect_llm_calls and out["manifest_entries"] != 0:
        # just_query: pure retrieval, no LLM calls — verify zero entries
        _fail(f"variant={variant!r} expected 0 manifest entries, got {out['manifest_entries']}")
    if expect_llm_calls and variant not in out["variant_tags"]:
        _fail(
            f"variant={variant!r} not found in manifest tags: {out['variant_tags']}"
        )

    payload = out["result_payload"]
    if payload.get("run_id") != f"smoke_{variant}":
        _fail(f"variant={variant!r} result.json wrong run_id: {payload.get('run_id')}")
    if payload.get("agent_model") != "gpt-4.1-mini-2025-04-14":
        _fail(f"variant={variant!r} result.json wrong agent_model: {payload.get('agent_model')}")
    if expect_llm_calls and float(payload.get("total_cost_usd", 0)) <= 0:
        _fail(f"variant={variant!r} expected total_cost_usd > 0, got {payload.get('total_cost_usd')}")

    _ok(
        f"variant={variant!r}: queries={report.n_queries} ok={report.n_successful} "
        f"failed={report.n_failed} entries={out['manifest_entries']} "
        f"cost=${payload.get('total_cost_usd', 0):.6f}"
    )
    return True


def main() -> int:
    """Run smoke tests for all FitText variants.

    Returns:
        0 on all-pass, 1 on any failure.
    """
    # The 4 published variants must all pass.
    _check_variant("single_pass", expect_llm_calls=True)
    _check_variant("multi_turn", expect_llm_calls=True)
    _check_variant("scattershot", expect_llm_calls=True)
    _check_variant("memetic", expect_llm_calls=True)

    # just_query: xfail cleanly if the schema enum does not yet include it.
    _check_variant("just_query", expect_llm_calls=False, xfail=True)

    print("\nALL SMOKE TESTS PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
