"""Smoke tests for the just_query zero-retrieval floor baseline.

All tests are offline:
  - The LLM is mocked to return a known canned response.
  - No retriever is consulted (the technique forbids it).
  - The Pydantic schema is exercised via ``resolve_config()``.

Coverage
--------
1. ``run_just_query_strategy`` (StableToolBench) returns the documented
   tuple shape: empty tool_ids, single retrieval_iterations entry, payload
   carrying ``answer``.
2. ``run_just_query_strategy`` (ToolRet mirror) returns the four-tuple
   contract: empty tool_ids/descriptions/scores + payload.answer.
3. ``configs/runs/just_query_toolret.yaml`` resolves to a valid RunConfig
   with ``variant == "just_query"``.
4. ``JustQueryBaseline.retrieve`` returns empty tool_ids and surfaces the
   LLM answer in ``metadata.answer``.
5. ``JustQueryBaseline.variant_tag == "baseline_just_query"``.

Run with pytest (from the repository root)::

    python -m pytest tests/smoke_just_query.py -v
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure project root is on path
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_STB = _PROJECT_ROOT / "StableToolBench"
if str(_STB) not in sys.path:
    sys.path.insert(0, str(_STB))

from baselines import JustQueryBaseline
from toolbench.runner import resolve_config


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

CANNED_ANSWER = "Yes — Paris is the capital of France."


class StubLLM:
    """Mimics the LLM client used by STB/ToolRet strategies.

    ``parse_with_messages`` returns a 3-tuple ``(message, _, _)`` where
    ``message`` is a dict with ``role`` and ``content`` keys — matching the
    contract in StableToolBench/toolbench/inference/LLM/chatgpt_function_model.py.
    """

    def __init__(self, answer: str = CANNED_ANSWER) -> None:
        self.answer = answer
        self.calls: list[dict] = []

    def parse_with_messages(
        self,
        messages: list,
        tools: list | None = None,
        process_id: int = 0,
        **kwargs: Any,
    ):
        self.calls.append({"messages": messages, "tools": tools, "kwargs": kwargs})
        msg = {"role": "assistant", "content": self.answer}
        return msg, 0, 100


@dataclass
class FakeNormalizedResponse:
    content: str | None = None
    tool_calls: list = field(default_factory=list)
    finish_reason: str = "stop"
    model_revision: str = "stub-model"
    input_tokens: int = 10
    cached_input_tokens: int = 0
    output_tokens: int = 5
    latency_ms: float = 50.0
    provider: str = "stub"
    raw_response: dict = field(default_factory=dict)


class StubModelClient:
    """Async LLM client used by the JustQueryBaseline path."""

    def __init__(self, answer: str = CANNED_ANSWER) -> None:
        self.answer = answer
        self.calls: list = []

    async def chat_completion(
        self,
        messages: list,
        tools: list | None = None,
        temperature: float = 0.0,
        seed: int = 42,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> FakeNormalizedResponse:
        self.calls.append(messages)
        return FakeNormalizedResponse(content=self.answer)


class StubRetriever:
    """Retriever that records calls — must NOT be invoked by just_query."""

    def __init__(self) -> None:
        self.calls: list = []

    def retrieving(self, query: str, top_k: int = 5, excluded_tools=None):
        self.calls.append({"query": query, "top_k": top_k})
        # Deliberately return a non-empty list so any accidental call would
        # leak tools into the result and trip the assertion in the test.
        return [{"category": "x", "tool_name": "t", "api_name": "a", "score": 1.0}]


class StubWrapper:
    """Minimal wrapper exposing the attributes the strategies read."""

    def __init__(self, query: str = "What is the capital of France?") -> None:
        self.input_description = query
        self.process_id = 0
        self.just_query = True


# ---------------------------------------------------------------------------
# Test 1 — STB strategy
# ---------------------------------------------------------------------------

class TestSTBStrategy(unittest.TestCase):
    def test_run_just_query_strategy_returns_empty_tool_ids(self) -> None:
        from toolbench.inference.Downstream_tasks.strategies import (
            run_just_query_strategy,
            select_and_run_strategy,
        )

        wrapper = StubWrapper(query="What is 2+2?")
        llm = StubLLM(answer="2+2 equals 4.")

        tool_ids, retrieval_iters, payload = run_just_query_strategy(
            llm_output="", wrapper=wrapper, llm=llm,
        )

        self.assertEqual(tool_ids, [], "just_query must return empty tool_ids")
        self.assertEqual(len(retrieval_iters), 1, "expected one retrieval log entry")
        self.assertEqual(retrieval_iters[0]["phase"], "just_query")
        self.assertEqual(retrieval_iters[0]["query"], "What is 2+2?")
        self.assertEqual(retrieval_iters[0]["answer"], "2+2 equals 4.")
        self.assertEqual(payload["strategy"], "just_query")
        self.assertEqual(payload["answer"], "2+2 equals 4.")

        # Confirm the LLM was called with empty tools=[]
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(llm.calls[0]["tools"], [])

    def test_router_dispatches_just_query_first(self) -> None:
        """Setting wrapper.just_query=True must short-circuit before other branches."""
        from toolbench.inference.Downstream_tasks.strategies import select_and_run_strategy

        wrapper = StubWrapper()
        # Set conflicting flags — just_query must still win because the
        # router checks it first.
        wrapper.memetic = True
        wrapper.scattershot = True

        llm = StubLLM()
        tool_ids, retrieval_iters, payload = select_and_run_strategy(
            llm_output="<some upstream blocks>", wrapper=wrapper, llm=llm,
        )
        self.assertEqual(tool_ids, [])
        self.assertEqual(payload["strategy"], "just_query")


# ---------------------------------------------------------------------------
# Test 2 — ToolRet mirror
# ---------------------------------------------------------------------------

class TestToolretStrategy(unittest.TestCase):
    def test_toolret_just_query_returns_empty_lists_and_answer(self) -> None:
        # The ToolRet module imports relative siblings (.tokens / .services / .prompts)
        # so we must import it as a package member.
        _TOOLRET = _PROJECT_ROOT / "Toolret"
        if str(_TOOLRET.parent) not in sys.path:
            sys.path.insert(0, str(_TOOLRET.parent))

        # Import the function directly — avoid the heavy router which builds
        # a ChatGPTFunction (real network client).
        from Toolret.strategy.strategies import run_just_query_strategy

        class _W:
            input_description = "What is 2+2?"
            process_id = 0

        llm = StubLLM(answer="4")
        tool_ids, tool_des, tool_scores, payload = run_just_query_strategy(
            "What is 2+2?", _W(), llm,
        )

        self.assertEqual(tool_ids, [])
        self.assertEqual(tool_des, [])
        self.assertEqual(tool_scores, [])
        self.assertEqual(payload["strategy"], "just_query")
        self.assertEqual(payload["answer"], "4")
        # No tools=[] was passed to the LLM
        self.assertEqual(llm.calls[0]["tools"], [])


# ---------------------------------------------------------------------------
# Test 3 — Config resolves with variant == just_query
# ---------------------------------------------------------------------------

class TestConfigResolution(unittest.TestCase):
    def test_just_query_toolret_yaml_resolves(self) -> None:
        cfg_path = _PROJECT_ROOT / "configs" / "runs" / "just_query_toolret.yaml"
        cfg = resolve_config(cfg_path, repo_root=_PROJECT_ROOT)

        self.assertEqual(cfg.fittext.variant, "just_query")
        self.assertEqual(cfg.benchmark.name, "toolret")
        self.assertEqual(cfg.fittext.top_k_retrieval, 0,
                         "zero-retrieval floor must request 0 tools")
        # Hash must be stable across two resolves
        h1 = cfg.config_hash()
        h2 = resolve_config(cfg_path, repo_root=_PROJECT_ROOT).config_hash()
        self.assertEqual(h1, h2)

    def test_just_query_stb_yaml_resolves(self) -> None:
        cfg_path = _PROJECT_ROOT / "configs" / "runs" / "just_query_stb_G2G3.yaml"
        cfg = resolve_config(cfg_path, repo_root=_PROJECT_ROOT)
        self.assertEqual(cfg.fittext.variant, "just_query")
        self.assertEqual(cfg.benchmark.name, "stabletoolbench")

    def test_cheap_sota_just_query_yaml_resolves(self) -> None:
        cfg_path = _PROJECT_ROOT / "configs" / "runs" / "cheap_sota_just_query_toolret.yaml"
        cfg = resolve_config(cfg_path, repo_root=_PROJECT_ROOT)
        self.assertEqual(cfg.fittext.variant, "just_query")
        self.assertEqual(cfg.extra.get("research_track"), "cheap_sota")


# ---------------------------------------------------------------------------
# Test 4 — JustQueryBaseline behaviour
# ---------------------------------------------------------------------------

class TestJustQueryBaseline(unittest.TestCase):
    def test_baseline_returns_empty_tool_ids_with_answer_metadata(self) -> None:
        client = StubModelClient(answer="42")
        retriever = StubRetriever()
        baseline = JustQueryBaseline(
            model_client=client,
            retriever=retriever,
            top_k=0,
            temperature=0.0,
            seed=42,
            max_tokens=128,
        )

        result = asyncio.run(baseline.retrieve(
            "What is the answer to life?",
            tool_catalog=[{"name": "t1"}, {"name": "t2"}],
        ))

        self.assertEqual(result.tool_ids, [])
        self.assertEqual(result.scores, [])
        self.assertEqual(result.metadata["strategy"], "just_query")
        self.assertEqual(result.metadata["answer"], "42")
        self.assertEqual(result.metadata["n_catalog_tools_offered"], 2)

        # The retriever MUST NOT have been called — that is the whole point.
        self.assertEqual(retriever.calls, [],
                         "just_query must NOT consult the retriever")

    def test_variant_tag(self) -> None:
        client = StubModelClient()
        retriever = StubRetriever()
        baseline = JustQueryBaseline(model_client=client, retriever=retriever)
        self.assertEqual(baseline.variant_tag, "baseline_just_query")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
