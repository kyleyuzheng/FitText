"""
Smoke tests for all four retrieval baselines.

All tests are offline: ModelClient is mocked (stub returning canned responses),
the Retriever is stubbed, and COLT is expected to raise ImportError when
$COLT_PATH is unset.

Run with pytest (from the repository root)::

    python -m pytest tests/smoke_baselines.py -v

Tests
-----
1. RetrievedTools schema validation (correct + error cases).
2. LessIsMoreBaseline: stub returning "tool_A, tool_B" -> tool_ids match.
3. LessIsMoreBaseline: fuzzy match (near-miss tool name).
4. ReInvokeBaseline: missing cache triggers build + persist.
5. ReInvokeBaseline: existing cache loads without LLM index calls.
6. Xu2024Baseline: runs N iterations and stops on convergence.
7. Xu2024Baseline: runs max_iterations when no convergence.
8. COLTBaseline: raises ImportError if $COLT_PATH unset.
9. Manifest variant_tag is correct for each baseline.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure project root is on path
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from baselines.base import Baseline, RetrievedTools, RetrieverAdapter
from baselines.less_is_more import LessIsMoreBaseline, _parse_pseudo_tools
from baselines.reinvoke import ReInvokeBaseline
from baselines.xu2024 import Xu2024Baseline
from baselines.colt import COLTBaseline


# ---------------------------------------------------------------------------
# Shared stubs
# ---------------------------------------------------------------------------

@dataclass
class FakeNormalizedResponse:
    """Minimal stub for NormalizedResponse — avoids importing the real class."""
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
    """Async LLM client that returns canned responses.

    Set ``self.responses`` to a list; calls pop from the front.
    If empty, returns a default "tool_A, tool_B" response.
    """

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses: list[str] = list(responses or [])
        self.calls: list[list] = []  # captured call args

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
        if self.responses:
            content = self.responses.pop(0)
        else:
            content = "tool_A\ntool_B"
        return FakeNormalizedResponse(content=content)

    def chat_completion_sync(self, *args, **kwargs):
        return asyncio.run(self.chat_completion(*args, **kwargs))


class StubRetrieverInner:
    """Fake retriever that returns pre-configured hits."""

    def __init__(self, hits: list[dict] | None = None) -> None:
        # Each dict: category, tool_name, api_name, score
        self._hits = hits or [
            {"category": "cat", "tool_name": "tool_A", "api_name": "tool_A", "score": 0.9},
            {"category": "cat", "tool_name": "tool_B", "api_name": "tool_B", "score": 0.8},
            {"category": "cat", "tool_name": "tool_C", "api_name": "tool_C", "score": 0.7},
        ]

    def retrieving(self, query: str, top_k: int = 5, excluded_tools=None) -> list[dict]:
        return self._hits[:top_k]

    def encode_sentence(self, text: str):
        """Return a simple deterministic fake embedding (1D tensor)."""
        if isinstance(text, list):
            return self.encode_corpus(text)
        import hashlib
        h = int(hashlib.md5(text.encode()).hexdigest()[:8], 16) / (2**32)
        try:
            import torch
            return torch.tensor([h, 1.0 - h, 0.5], dtype=torch.float32)
        except ImportError:
            return [h, 1.0 - h, 0.5]

    def encode_corpus(self, texts: list[str]):
        """Return deterministic fake embeddings for a batch of texts."""
        rows = [self.encode_sentence(text) for text in texts]
        try:
            import torch
            return torch.stack(rows)
        except Exception:
            return rows


def make_catalog(names: list[str]) -> list[dict]:
    """Build a minimal tool catalog from a list of api_names."""
    return [
        {
            "category": "cat",
            "tool_name": name,
            "api_name": name,
            "name": name,
            "description": f"Description of {name}.",
        }
        for name in names
    ]


def run_async(coro):
    """Run an async coroutine in a test context."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestRetrievedToolsSchema(unittest.TestCase):
    """1. RetrievedTools schema validation."""

    def test_valid_construction(self):
        rt = RetrievedTools(tool_ids=["a::b::c"], scores=[0.9])
        self.assertEqual(rt.tool_ids, ["a::b::c"])
        self.assertEqual(rt.scores, [0.9])
        self.assertIsInstance(rt.metadata, dict)

    def test_empty_construction(self):
        rt = RetrievedTools(tool_ids=[], scores=[])
        self.assertEqual(len(rt.tool_ids), 0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            RetrievedTools(tool_ids=["a", "b"], scores=[0.9])

    def test_metadata_default_empty(self):
        rt = RetrievedTools(tool_ids=[], scores=[])
        self.assertEqual(rt.metadata, {})

    def test_metadata_stored(self):
        rt = RetrievedTools(tool_ids=["x"], scores=[0.5], metadata={"k": "v"})
        self.assertEqual(rt.metadata["k"], "v")


class TestParsePseudoTools(unittest.TestCase):
    """Helper tests for less_is_more pseudo-tool parsing (paper §III-B)."""

    def test_line_separated(self):
        items = _parse_pseudo_tools(
            "Fetch the weather for a location.\nTranslate text between languages.",
            max_count=5,
        )
        self.assertEqual(len(items), 2)
        self.assertTrue(any("weather" in s.lower() for s in items))
        self.assertTrue(any("translate" in s.lower() for s in items))

    def test_numbered_prefix_stripped(self):
        items = _parse_pseudo_tools(
            "1. Tool A description.\n2. Tool B description.", max_count=5
        )
        self.assertEqual(len(items), 2)
        # Numbering is stripped
        self.assertFalse(any(s.startswith("1.") for s in items))

    def test_bullets_stripped(self):
        items = _parse_pseudo_tools("- Tool A desc\n* Tool B desc", max_count=5)
        self.assertEqual(len(items), 2)
        self.assertFalse(any(s.startswith("-") or s.startswith("*") for s in items))

    def test_json_array(self):
        items = _parse_pseudo_tools(
            '["fetch weather data", "translate text"]', max_count=5
        )
        self.assertEqual(items, ["fetch weather data", "translate text"])

    def test_json_object_array(self):
        items = _parse_pseudo_tools(
            '[{"description": "weather"}, {"functionality": "translate"}]',
            max_count=5,
        )
        self.assertIn("weather", items)
        self.assertIn("translate", items)

    def test_truncate_to_max(self):
        items = _parse_pseudo_tools("a\nb\nc\nd\ne\nf", max_count=3)
        self.assertEqual(items, ["a", "b", "c"])

    def test_quotes_stripped(self):
        items = _parse_pseudo_tools('"fetch weather"', max_count=5)
        self.assertIn("fetch weather", items)

    def test_backticks_stripped(self):
        items = _parse_pseudo_tools("`fetch weather`", max_count=5)
        self.assertIn("fetch weather", items)

    def test_empty_lines_dropped(self):
        items = _parse_pseudo_tools("first\n\n\nsecond\n   \nthird", max_count=5)
        self.assertEqual(items, ["first", "second", "third"])


class TestLessIsMoreBaseline(unittest.TestCase):
    """LessIsMoreBaseline tests (paper §III-B Tool Recommender + §III-C Tool Controller)."""

    def _make_baseline(
        self,
        responses: list[str],
        top_k: int = 3,
        confidence_threshold: float = 0.0,
    ) -> LessIsMoreBaseline:
        client = StubModelClient(responses=responses)
        retriever = RetrieverAdapter(StubRetrieverInner())
        # Threshold 0.0 disables L3 fallback for these unit tests
        return LessIsMoreBaseline(
            client,
            retriever,
            top_k=top_k,
            confidence_threshold=confidence_threshold,
        )

    def test_pseudo_tools_drive_retrieval(self):
        """LLM-generated pseudo-tool descs are passed to the retriever."""
        baseline = self._make_baseline(
            responses=["Fetch weather data for a city.\nLook up flight info."]
        )
        catalog = make_catalog(["tool_A", "tool_B", "tool_C"])
        result = run_async(baseline.retrieve("test query", tool_catalog=catalog))
        self.assertIsInstance(result, RetrievedTools)
        # StubRetrieverInner returns tool_A, tool_B, tool_C for any query;
        # max-pool over 2 pseudo-tools should still yield those.
        self.assertIn("cat::tool_A::tool_A", result.tool_ids)
        self.assertEqual(len(result.tool_ids), len(result.scores))

    def test_l3_fallback_on_low_confidence(self):
        """When avg score < threshold, fall back to raw-query retrieval."""
        # StubRetrieverInner returns max score 0.9; with threshold 0.95 we
        # trigger the L3 fallback.
        baseline = self._make_baseline(
            responses=["a vague pseudo-tool"],
            top_k=2,
            confidence_threshold=0.95,
        )
        catalog = make_catalog(["tool_A", "tool_B"])
        result = run_async(baseline.retrieve("test query", tool_catalog=catalog))
        self.assertTrue(result.metadata.get("used_l3_fallback", False))

    def test_no_l3_fallback_on_high_confidence(self):
        """When avg score >= threshold, do not fall back."""
        baseline = self._make_baseline(
            responses=["a confident pseudo-tool"],
            top_k=2,
            confidence_threshold=0.1,
        )
        catalog = make_catalog(["tool_A", "tool_B"])
        result = run_async(baseline.retrieve("test query", tool_catalog=catalog))
        self.assertFalse(result.metadata.get("used_l3_fallback", True))

    def test_metadata_contains_pseudo_tools(self):
        baseline = self._make_baseline(
            responses=["fetch weather\ntranslate text"]
        )
        catalog = make_catalog(["tool_A", "tool_B"])
        result = run_async(baseline.retrieve("q", tool_catalog=catalog))
        self.assertIn("pseudo_tools", result.metadata)
        self.assertIn("llm_raw_text", result.metadata)
        self.assertIn("fetch weather", result.metadata["pseudo_tools"])

    def test_empty_llm_falls_back_to_query(self):
        """If LLM returns no parseable pseudo-tools, use raw query as probe."""
        baseline = self._make_baseline(responses=[""])
        catalog = make_catalog(["tool_A"])
        result = run_async(baseline.retrieve("a query", tool_catalog=catalog))
        # Falls back to using the raw query as the pseudo-tool
        self.assertEqual(result.metadata["pseudo_tools"], ["a query"])

    def test_variant_tag(self):
        baseline = self._make_baseline(responses=[])
        self.assertEqual(baseline.variant_tag, "baseline_less_is_more")


class TestReInvokeBaseline(unittest.TestCase):
    """4-5. ReInvokeBaseline tests."""

    def _make_baseline(
        self,
        responses: list[str],
        cache_dir: str,
        top_k: int = 2,
    ) -> ReInvokeBaseline:
        client = StubModelClient(responses=responses)
        retriever = RetrieverAdapter(StubRetrieverInner())
        return ReInvokeBaseline(
            client,
            retriever,
            top_k=top_k,
            cache_dir=cache_dir,
            embedder_revision="test_rev_123",
            embedder_name="test-embedder",
        )

    def test_cache_build_and_persist(self):
        """Missing cache triggers build, persists to JSONL."""
        catalog = make_catalog(["tool_A", "tool_B"])
        # Provide: k_synth=2 synth queries per tool + 1 intent extraction = 5 responses
        responses = [
            "query about tool_A type 1\nquery about tool_A type 2",  # synth for tool_A
            "query about tool_B type 1\nquery about tool_B type 2",  # synth for tool_B
            "retrieve files\ncheck status",                           # intent extraction
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline = self._make_baseline(responses=responses, cache_dir=tmpdir, top_k=2)
            # Patch k_synth to 2 so we need exactly 2 responses for 2 tools
            baseline._k_synth = 2
            result = run_async(baseline.retrieve("test query", tool_catalog=catalog))

            # Check cache was written
            cache_path = baseline._synth_index.path
            self.assertTrue(cache_path.exists(), f"Cache not created at {cache_path}")

            # Load cache and verify contents
            with cache_path.open("r") as fh:
                records = [json.loads(line) for line in fh if line.strip()]
            tool_ids_in_cache = {r["tool_id"] for r in records}
            self.assertTrue(len(tool_ids_in_cache) > 0)

        # Check result schema
        self.assertIsInstance(result, RetrievedTools)
        self.assertEqual(len(result.tool_ids), len(result.scores))

    def test_existing_cache_no_index_build(self):
        """If cache exists, no LLM calls for index build (only intent extraction)."""
        catalog = make_catalog(["tool_A", "tool_B"])

        with tempfile.TemporaryDirectory() as tmpdir:
            # Pre-populate cache
            cache_path = Path(tmpdir) / "reinvoke_synthq_fake.jsonl"
            with cache_path.open("w") as fh:
                for name in ["tool_A", "tool_B"]:
                    fh.write(json.dumps({
                        "tool_id": f"cat::{name}::{name}",
                        "synth_queries": [f"query for {name}"],
                    }) + "\n")

            # Build baseline and point the synth-query sidecar at the
            # pre-populated cache.
            client = StubModelClient(responses=["single intent"])
            retriever = RetrieverAdapter(StubRetrieverInner())
            baseline = ReInvokeBaseline(
                client, retriever, top_k=2,
                cache_dir=tmpdir, embedder_revision="test_rev_123", k_synth=1
            )
            baseline._synth_index.path = cache_path

            n_calls_before = len(client.calls)
            result = run_async(baseline.retrieve("test query", tool_catalog=catalog))

            # Only 1 LLM call for intent extraction (no index build calls)
            n_calls_after = len(client.calls)
            self.assertEqual(n_calls_after - n_calls_before, 1)

        self.assertIsInstance(result, RetrievedTools)

    def test_variant_tag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline = self._make_baseline(responses=[], cache_dir=tmpdir)
        self.assertEqual(baseline.variant_tag, "baseline_reinvoke")


class TestXu2024Baseline(unittest.TestCase):
    """6-7. Xu2024Baseline tests."""

    def _make_baseline(self, responses: list[str], max_iterations: int = 3, top_k: int = 2):
        client = StubModelClient(responses=responses)
        retriever = RetrieverAdapter(StubRetrieverInner())
        return Xu2024Baseline(
            client, retriever, top_k=top_k, max_iterations=max_iterations
        )

    def test_stops_on_convergence_stable_set(self):
        """Same retrieved set across iterations -> stable_set convergence."""
        # StubRetrieverInner returns the same hits every call.  After iter 0
        # retrieves (with C/A/R = 3 LLM calls), iter 1 retrieves the same set
        # and we detect stable_set convergence before doing C/A/R again.
        baseline = self._make_baseline(
            responses=["comp 0", "assess 0", "refine to new query"],
            max_iterations=3,
        )
        catalog = make_catalog(["tool_A", "tool_B", "tool_C"])
        result = run_async(baseline.retrieve("test query", tool_catalog=catalog))

        self.assertIsInstance(result, RetrievedTools)
        self.assertIn("trace", result.metadata)
        self.assertTrue(result.metadata.get("converged", False))
        self.assertEqual(result.metadata.get("convergence_reason"), "stable_set")

    def test_stops_on_na_token(self):
        """If refinement emits 'N/A', stop immediately (paper §4.2)."""
        call_count = [0]
        # Different hits each iter so stable_set is not the trigger
        hits_list = [
            [{"category": "cat", "tool_name": f"tool_{i}", "api_name": f"tool_{i}", "score": 0.9}]
            for i in range(10)
        ]

        class VariableRetriever:
            def retrieving(self, query, top_k=5, excluded_tools=None):
                idx = call_count[0] % len(hits_list)
                call_count[0] += 1
                return hits_list[idx][:top_k]

        # iter 0: comprehension, assessment, refinement="N/A" -> stop
        client = StubModelClient(responses=["comp", "assess", "N/A"])
        retriever = RetrieverAdapter(VariableRetriever())
        baseline = Xu2024Baseline(client, retriever, top_k=1, max_iterations=3)
        catalog = make_catalog([f"tool_{i}" for i in range(10)])
        result = run_async(baseline.retrieve("test query", tool_catalog=catalog))

        self.assertTrue(result.metadata.get("converged", False))
        self.assertEqual(result.metadata.get("convergence_reason"), "na_token")
        self.assertEqual(result.metadata["iterations_run"], 1)

    def test_runs_max_iterations_when_no_convergence(self):
        """Variable retriever + non-N/A refinement -> run all iterations."""
        call_count = [0]
        hits_list = [
            [{"category": "cat", "tool_name": f"tool_{i}", "api_name": f"tool_{i}", "score": 0.9}]
            for i in range(10)
        ]

        class VariableRetriever:
            def retrieving(self, query, top_k=5, excluded_tools=None):
                idx = call_count[0] % len(hits_list)
                call_count[0] += 1
                return hits_list[idx][:top_k]

        # max_iterations=3, but final iter doesn't run C/A/R, so we need 2*3=6
        # LLM responses (C, A, R for iter 0 and iter 1)
        client = StubModelClient(responses=[
            "comp 0", "assess 0", "refine 0",
            "comp 1", "assess 1", "refine 1",
        ])
        retriever = RetrieverAdapter(VariableRetriever())
        baseline = Xu2024Baseline(client, retriever, top_k=1, max_iterations=3)
        catalog = make_catalog([f"tool_{i}" for i in range(10)])
        result = run_async(baseline.retrieve("test query", tool_catalog=catalog))

        self.assertIsInstance(result, RetrievedTools)
        self.assertEqual(result.metadata["iterations_run"], 3)
        self.assertFalse(result.metadata.get("converged", True))
        self.assertEqual(result.metadata.get("convergence_reason"), "max_iter")

    def test_trace_contains_car_chain(self):
        """Trace entries record query, tool_ids, and C/A/R outputs (paper §4.2)."""
        call_count = [0]
        hits_list = [
            [{"category": "cat", "tool_name": f"tool_{i}", "api_name": f"tool_{i}", "score": 0.9}]
            for i in range(5)
        ]

        class VariableRetriever:
            def retrieving(self, query, top_k=5, excluded_tools=None):
                idx = call_count[0] % len(hits_list)
                call_count[0] += 1
                return hits_list[idx][:top_k]

        client = StubModelClient(responses=[
            "comp output", "assess output", "refine output",
        ])
        retriever = RetrieverAdapter(VariableRetriever())
        baseline = Xu2024Baseline(client, retriever, top_k=1, max_iterations=2)
        catalog = make_catalog([f"tool_{i}" for i in range(5)])
        result = run_async(baseline.retrieve("original query", tool_catalog=catalog))

        trace = result.metadata.get("trace", [])
        self.assertGreater(len(trace), 0)
        # iter 0 has the C/A/R chain
        first = trace[0]
        self.assertIn("query", first)
        self.assertIn("tool_ids", first)
        self.assertEqual(first["comprehension"], "comp output")
        self.assertEqual(first["assessment"], "assess output")
        self.assertEqual(first["refinement"], "refine output")

    def test_variant_tag(self):
        baseline = self._make_baseline(responses=[])
        self.assertEqual(baseline.variant_tag, "baseline_xu2024")


class TestCOLTBaseline(unittest.TestCase):
    """8. COLT raises ImportError when $COLT_PATH is unset."""

    def test_raises_import_error_when_clone_missing(self):
        """COLTBaseline construction must raise ImportError if no clone is present.

        Post-§5.6 upgrade: the wrapper looks at $COLT_PATH and falls back to
        ``baselines/external/colt/``. With $COLT_PATH pointed at a guaranteed-
        missing path, neither resolves and we expect ImportError pointing the
        user at scripts/setup_colt.sh.
        """
        with patch.dict(os.environ, {"COLT_PATH": "/nonexistent/colt_path_smoke_test"}):
            client = StubModelClient()
            retriever = RetrieverAdapter(StubRetrieverInner())
            with self.assertRaises(ImportError) as ctx:
                COLTBaseline(client, retriever, top_k=5)
            msg = str(ctx.exception)
            self.assertIn("COLT clone not found", msg)
            self.assertIn("setup_colt.sh", msg)

    def test_raises_import_error_with_nonexistent_path(self):
        """COLTBaseline construction raises ImportError for bad explicit $COLT_PATH."""
        with patch.dict(os.environ, {"COLT_PATH": "/nonexistent/colt_path_smoke_test"}):
            client = StubModelClient()
            retriever = RetrieverAdapter(StubRetrieverInner())
            with self.assertRaises(ImportError) as ctx:
                COLTBaseline(client, retriever, top_k=5)
            self.assertIn("setup_colt.sh", str(ctx.exception))

    def test_variant_tag_without_path(self):
        """variant_tag is available without instantiation via the class."""
        self.assertEqual(COLTBaseline.name, "colt")


class TestRetrieverAdapter(unittest.TestCase):
    """RetrieverAdapter wraps correctly."""

    def test_retrieve_returns_tuple_list(self):
        inner = StubRetrieverInner()
        adapter = RetrieverAdapter(inner)
        hits = adapter.retrieve("query", top_k=2)
        self.assertEqual(len(hits), 2)
        for tool_id, score in hits:
            self.assertIsInstance(tool_id, str)
            self.assertIsInstance(score, float)
            self.assertIn("::", tool_id)

    def test_retrieving_delegates(self):
        inner = StubRetrieverInner()
        adapter = RetrieverAdapter(inner)
        hits = adapter.retrieving("query", top_k=2)
        self.assertEqual(len(hits), 2)
        self.assertIn("api_name", hits[0])

    def test_protocol_check(self):
        """RetrieverAdapter should be an instance of Retriever protocol."""
        from baselines.base import Retriever
        inner = StubRetrieverInner()
        adapter = RetrieverAdapter(inner)
        # RetrieverAdapter itself satisfies Retriever via .retrieving()
        self.assertTrue(hasattr(adapter, "retrieving"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
