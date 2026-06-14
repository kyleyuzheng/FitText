"""
Smoke tests for the *real* COLT clone integration (baselines/colt.py).

Two run modes:
    * Default (offline) — clone may or may not be present; tests that DO NOT
      need a working checkpoint always run and verify the wrapper's surface
      area (import-error semantics, SHA pin check, corpus index loading,
      output parser).
    * Integration (``--run-colt-integration`` / ``COLT_RUN_INTEGRATION=1``) —
      actually invokes the COLT subprocess on three small synthetic queries.
      Requires the clone + a checkpoint + GPU. Skipped by default.

Run modes
---------
    # Offline tests only (default):
    pytest tests/smoke_colt_real.py -v

    # Full integration test (requires $COLT_PATH, $COLT_CKPT, GPU):
    pytest tests/smoke_colt_real.py --run-colt-integration -v
    # or:
    COLT_RUN_INTEGRATION=1 pytest tests/smoke_colt_real.py -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure project root is on path so `baselines.colt` imports cleanly.
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from baselines.base import RetrievedTools, RetrieverAdapter
from baselines.colt import (
    COLT_PINNED_SHA,
    COLT_SUPPORTED_DATASETS,
    COLTBaseline,
    _build_api_to_composite,
    _default_colt_path,
    _load_colt_corpus_index,
    _parse_colt_topk_indices,
)


# ---------------------------------------------------------------------------
# Integration-mode gate
# ---------------------------------------------------------------------------

def _integration_enabled() -> bool:
    """Integration tests run only if user explicitly opts in."""
    return os.environ.get("COLT_RUN_INTEGRATION", "0") == "1" or "--run-colt-integration" in sys.argv


# Pytest marker — collected only when --run-colt-integration is passed
# (configured in tests/conftest.py or pytest.ini if needed by the harness).
colt_integration = pytest.mark.colt_integration


# ---------------------------------------------------------------------------
# Stubs (reused for cheap construction in unit tests)
# ---------------------------------------------------------------------------

@dataclass
class StubResponse:
    """Minimal NormalizedResponse stand-in."""
    content: str | None = None
    tool_calls: list = field(default_factory=list)


class StubModelClient:
    """Stub ModelClient — never called by COLT (which is LLM-free at inference)."""

    async def chat(self, **kwargs: Any) -> StubResponse:
        return StubResponse(content="should-never-be-called")


class StubRetriever:
    """Stub Retriever — also unused by COLT but required by Baseline ABC."""

    def retrieving(
        self, query: str, top_k: int = 5, excluded_tools: dict | None = None
    ) -> list[dict[str, Any]]:
        return [{"category": "cat", "tool_name": "t", "api_name": f"api_{i}", "score": 1.0 - i / 10.0}
                for i in range(top_k)]


# ---------------------------------------------------------------------------
# Unit tests (always run — no clone needed)
# ---------------------------------------------------------------------------

class TestCOLTConstants(unittest.TestCase):
    """COLT_PINNED_SHA + COLT_SUPPORTED_DATASETS are non-empty and well-formed."""

    def test_pinned_sha_format(self):
        self.assertEqual(len(COLT_PINNED_SHA), 40)
        self.assertTrue(all(c in "0123456789abcdef" for c in COLT_PINNED_SHA))

    def test_supported_datasets(self):
        self.assertIn("ToolLens", COLT_SUPPORTED_DATASETS)
        self.assertIn("ToolBenchG2", COLT_SUPPORTED_DATASETS)
        self.assertIn("ToolBenchG3", COLT_SUPPORTED_DATASETS)


class TestParser(unittest.TestCase):
    """``_parse_colt_topk_indices`` handles COLT's tensor_data_formatted.txt."""

    def test_parses_simple_list(self):
        raw = "[3, 17, 442]\n"
        out = _parse_colt_topk_indices(raw)
        self.assertEqual(out, [[3, 17, 442]])

    def test_parses_multi_query(self):
        raw = "[1, 2]\n[3, 4]\n[5, 6]\n"
        out = _parse_colt_topk_indices(raw)
        self.assertEqual(len(out), 3)
        self.assertEqual(out[-1], [5, 6])

    def test_handles_empty_line(self):
        raw = "[1, 2]\n\n[3, 4]\n"
        out = _parse_colt_topk_indices(raw)
        self.assertEqual(out, [[1, 2], [3, 4]])

    def test_handles_empty_bracket(self):
        raw = "[]\n"
        out = _parse_colt_topk_indices(raw)
        self.assertEqual(out, [[]])

    def test_skips_unparseable_line(self):
        raw = "[1, 2]\ngarbage\n[3, 4]\n"
        out = _parse_colt_topk_indices(raw)
        # garbage line is skipped, valid lines preserved
        self.assertEqual(out, [[1, 2], [3, 4]])


class TestCatalogMapping(unittest.TestCase):
    """``_build_api_to_composite`` produces case-insensitive lookups."""

    def test_basic_mapping(self):
        catalog = [
            {"category": "Tools", "tool_name": "Calc", "api_name": "Add"},
            {"category": "Tools", "tool_name": "Calc", "api_name": "Sub"},
        ]
        mapping = _build_api_to_composite(catalog)
        self.assertEqual(mapping["add"], "Tools::Calc::Add")
        self.assertEqual(mapping["sub"], "Tools::Calc::Sub")

    def test_handles_missing_api_name(self):
        catalog = [{"category": "Tools", "tool_name": "OnlyName"}]
        mapping = _build_api_to_composite(catalog)
        # Falls back to tool_name as api_name.
        self.assertEqual(mapping["onlyname"], "Tools::OnlyName::OnlyName")


class TestCorpusIndex(unittest.TestCase):
    """``_load_colt_corpus_index`` reads corpus.jsonl into idx -> api_name."""

    def test_loads_jsonl_with_title_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            colt_root = Path(tmp)
            ds_dir = colt_root / "datasets" / "ToolLens"
            ds_dir.mkdir(parents=True)
            corpus = ds_dir / "corpus.jsonl"
            corpus.write_text(
                json.dumps({"_id": "0", "title": "tool_alpha", "text": "x"}) + "\n"
                + json.dumps({"_id": "1", "title": "tool_beta", "text": "y"}) + "\n",
                encoding="utf-8",
            )
            idx_to_name = _load_colt_corpus_index(colt_root, "ToolLens")
            self.assertEqual(idx_to_name[0], "tool_alpha")
            self.assertEqual(idx_to_name[1], "tool_beta")
            self.assertEqual(len(idx_to_name), 2)

    def test_raises_when_corpus_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            colt_root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                _load_colt_corpus_index(colt_root, "ToolLens")


class TestImportErrorSemantics(unittest.TestCase):
    """COLTBaseline raises ImportError with a setup_colt.sh hint when clone is absent."""

    def test_explicit_bad_path_raises(self):
        with patch.dict(os.environ, {"COLT_PATH": "/nonexistent/colt_smoke_path"}):
            with self.assertRaises(ImportError) as ctx:
                COLTBaseline(StubModelClient(), RetrieverAdapter(StubRetriever()), top_k=5)
            msg = str(ctx.exception)
            self.assertIn("setup_colt.sh", msg)
            self.assertIn(COLT_PINNED_SHA[:12], msg)


# ---------------------------------------------------------------------------
# Integration test (skipped unless --run-colt-integration or COLT_RUN_INTEGRATION=1)
# ---------------------------------------------------------------------------

@pytest.mark.colt_integration
@pytest.mark.skipif(not _integration_enabled(),
                    reason="Set COLT_RUN_INTEGRATION=1 or pass --run-colt-integration to run.")
def test_colt_real_subprocess_call_smoke():
    """End-to-end smoke: instantiate COLT, run on 3 small synthetic queries."""
    import asyncio

    colt_root = _default_colt_path() if not os.environ.get("COLT_PATH") else Path(os.environ["COLT_PATH"])
    if not colt_root.exists():
        pytest.skip(f"COLT clone not at {colt_root} — run scripts/setup_colt.sh first.")

    baseline = COLTBaseline(
        StubModelClient(),
        RetrieverAdapter(StubRetriever()),
        top_k=5,
        colt_timeout_seconds=900,
    )

    # Construct a tiny tool catalog matching the first few corpus entries so
    # mapping doesn't drop all results.
    catalog = []
    for idx in range(min(20, len(baseline._idx_to_api))):
        api = baseline._idx_to_api[idx]
        catalog.append({"category": "test", "tool_name": api, "api_name": api})

    queries = [
        "find me a tool that searches the web",
        "I need to convert celsius to fahrenheit",
        "look up the weather forecast for tomorrow",
    ]

    for q in queries:
        result = asyncio.run(baseline.retrieve(q, tool_catalog=catalog))
        assert isinstance(result, RetrievedTools), f"Expected RetrievedTools, got {type(result)}"
        assert len(result.tool_ids) <= 5, f"top_k=5 violated: got {len(result.tool_ids)}"
        assert len(result.tool_ids) == len(result.scores), "tool_ids/scores length mismatch"
        assert result.metadata.get("colt_commit") == COLT_PINNED_SHA, \
            f"colt_commit metadata mismatch: {result.metadata.get('colt_commit')}"
        assert result.metadata.get("colt_dataset") in COLT_SUPPORTED_DATASETS


if __name__ == "__main__":
    unittest.main()
