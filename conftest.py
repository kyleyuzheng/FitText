"""Pytest root conftest.

Ensures both portions of the ``toolbench`` namespace package are importable:
  - Root ``toolbench/runner/`` (from config-runner track)
  - ``StableToolBench/toolbench/{inference,observability,...}`` (from modelclient + cost-logger tracks)

PEP 420 namespace packages require that no portion has an ``__init__.py``.
Both portions are namespace-style; we just need ``StableToolBench/`` on sys.path so
imports like ``toolbench.inference.LLM.clients`` resolve.

Also exposes the ``synthetic_toolret_corpus`` fixture so the E2E pipeline
test can exercise the real ``ToolRetriever`` against a tiny 5-tool, 3-query
corpus without the production ``./data/retrieval/Toolret/`` tree on disk.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Force CPU for the synthetic E2E corpus before any torch import.
# Shared hosts often have a saturated GPU; the 5-tool mini corpus fits
# trivially on CPU.  We honour an explicit ``FITTEXT_E2E_GPU=1`` opt-in
# for callers who want to exercise the GPU path.
if os.environ.get("FITTEXT_E2E_GPU", "0") != "1":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

_REPO_ROOT = Path(__file__).resolve().parent
_STB = _REPO_ROOT / "StableToolBench"
if str(_STB) not in sys.path:
    sys.path.insert(0, str(_STB))


# ---------------------------------------------------------------------------
# Shared E2E fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def synthetic_toolret_corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Materialise a 5-tool, 3-query ToolRet-shaped corpus once per session.

    Wraps :func:`tests.fixtures.mini_toolret_corpus.build_corpus`.

    Why session-scoped:
        The corpus is read-only and identical across all E2E tests that touch
        ToolRet.  Building it once amortises the ~2-3s SentenceTransformer
        init + 5-sentence encode across the 5 tests that consume it (was ~10s
        of setup; now ~2-3s total).  Critical for the <60s wall-clock target.

    Args:
        tmp_path_factory: Pytest's session-scoped temp-path factory.

    Returns:
        Absolute path to the corpus root.  Callers should set both
        ``cfg.extra["toolret_corpus_root"]`` and
        ``cfg.extra["toolret_queries_root"]`` to ``str(corpus_root)`` and
        override ``cfg.benchmark.splits = ["mini"]``.
    """
    # ``tests/`` is intentionally not a package on this repo, so import the
    # fixture module via its file path.
    import importlib.util as _ilu

    fixture_path = _REPO_ROOT / "tests" / "fixtures" / "mini_toolret_corpus.py"
    spec = _ilu.spec_from_file_location("_e2e_mini_toolret_corpus", fixture_path)
    mod = _ilu.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    # Use a session-scoped tmp dir so the corpus survives across tests.
    session_tmp = tmp_path_factory.mktemp("synthetic_toolret_corpus")
    return mod.build_corpus(session_tmp)
