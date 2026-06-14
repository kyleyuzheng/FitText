"""Smoke test — split temperatures (trigger vs evolution).

Verifies the split-temperatures track wires correctly end-to-end:

  1. ``cheap_sota_memetic_toolret.yaml`` resolves to
     ``cfg.model.temperature == 1.0`` and
     ``cfg.model.evolution_temperature == 1.5``.
  2. The STB adapter sets ``wrapper.evolution_temperature == 1.5`` from cfg.
  3. ``run_memetic_strategy`` routes every evolutionary LLM call site
     (seed-population generation, mutation, crossover, refinement) at
     ``temperature=1.5``. The "right T flows to the right call site"
     assertion is the core of the test.
  4. Back-compat: when ``evolution_temperature`` is None on the wrapper
     and ``base_temp`` is set (e.g. legacy 0.9), the lookup chain prefers
     ``base_temp`` — the existing 231 memetic result dirs reproduce
     bit-identical.

Run from the worktree root:
    pytest tests/smoke_split_temperatures.py -v
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "StableToolBench"))

# Force CPU before any torch import — shared host.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from toolbench.runner import resolve_config  # noqa: E402
from toolbench.runner.schema import ModelSpec, RunConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Test 1: resolved config carries split-T defaults
# ---------------------------------------------------------------------------


def test_memetic_config_has_split_temperatures() -> None:
    """``cheap_sota_memetic_toolret.yaml`` resolves to (trigger=1.0, evolution=1.5)."""
    config_path = REPO_ROOT / "configs" / "runs" / "cheap_sota_memetic_toolret.yaml"
    assert config_path.exists(), f"missing config: {config_path}"

    cfg = resolve_config(config_path, repo_root=REPO_ROOT)

    assert cfg.model.temperature == 1.0, (
        f"memetic config trigger T mismatch: got {cfg.model.temperature!r}, "
        f"expected 1.0 (DFSDT trigger)"
    )
    assert cfg.model.evolution_temperature == 1.5, (
        f"memetic config evolution T mismatch: got {cfg.model.evolution_temperature!r}, "
        f"expected 1.5 (memetic subprocess)"
    )


# ---------------------------------------------------------------------------
# Test 2: schema field default + override semantics
# ---------------------------------------------------------------------------


def test_schema_evolution_temperature_default_is_none() -> None:
    """``ModelSpec.evolution_temperature`` defaults to None (legacy back-compat)."""
    spec = ModelSpec(name="gpt-4.1-mini-2025-04-14")
    assert spec.temperature == 0.0
    assert spec.evolution_temperature is None, (
        "evolution_temperature must default to None so legacy configs "
        "without the new field preserve back-compat via wrapper.base_temp."
    )


def test_schema_evolution_temperature_explicit() -> None:
    """Setting both fields is allowed and they round-trip independently."""
    spec = ModelSpec(
        name="gpt-4.1-mini-2025-04-14",
        temperature=1.0,
        evolution_temperature=1.5,
    )
    assert spec.temperature == 1.0
    assert spec.evolution_temperature == 1.5


# ---------------------------------------------------------------------------
# Test 3: STB adapter wires cfg.model.evolution_temperature onto wrapper
# ---------------------------------------------------------------------------


def _make_cfg(
    *,
    trigger_t: float,
    evolution_t: float | None,
    variant: str = "memetic",
) -> RunConfig:
    """Build a minimal valid RunConfig for adapter wiring tests."""
    cfg_dict: dict[str, Any] = {
        "model": {
            "name": "gpt-4.1-mini-2025-04-14",
            "max_tokens": 2048,
            "temperature": trigger_t,
            "seed": 42,
        },
        "fittext": {
            "variant": variant,
            "population_size": 2,
            "generations": 2,
            "mutation_prob": 0.5,
            "crossover_prob": 0.5,
            "memory_penalty": 0.5,
            "fitness_alpha": 0.7,
            "jaccard_threshold": 0.3,
            "selection": "fitness",
            "top_k_retrieval": 3,
        },
        "benchmark": {"name": "stabletoolbench", "splits": ["G1"]},
        "evaluator": {
            "judge_model": "gpt-4o-2024-08-06",
            "judge_revision": "2024-08-06",
            "simulator_model": "gpt-4o-2024-08-06",
            "simulator_revision": "2024-08-06",
        },
        "embedder": {"name": "sentence-transformers/all-MiniLM-L6-v2"},
        "infra": {"cache_dir": "/tmp/x", "output_dir": "/tmp/x"},
        "budget": {"max_cost_usd": 50.0},
    }
    if evolution_t is not None:
        cfg_dict["model"]["evolution_temperature"] = evolution_t
    return RunConfig(**cfg_dict)


def test_stb_adapter_propagates_evolution_temperature() -> None:
    """STB adapter sets ``wrapper.evolution_temperature`` from cfg."""
    from toolbench.runner.adapters.stabletoolbench import _StbStrategyWrapper

    cfg = _make_cfg(trigger_t=1.0, evolution_t=1.5)
    wrapper = _StbStrategyWrapper(
        cfg=cfg,
        query="dummy",
        retriever=None,
        tool_root_dir="/tmp/tools",
    )
    assert wrapper.evolution_temperature == 1.5, (
        f"adapter must propagate evolution_temperature; "
        f"got {wrapper.evolution_temperature!r}"
    )
    # Legacy base_temp is preserved for back-compat.
    assert wrapper.base_temp == 0.9


def test_stb_adapter_evolution_temperature_none_when_unset() -> None:
    """When cfg.model.evolution_temperature is None, wrapper attr is None.

    This is the legacy-config path: old YAMLs without the new field MUST
    fall back to ``wrapper.base_temp`` (0.9) inside the strategy so the
    231 existing memetic result dirs reproduce bit-identical.
    """
    from toolbench.runner.adapters.stabletoolbench import _StbStrategyWrapper

    cfg = _make_cfg(trigger_t=0.0, evolution_t=None)
    wrapper = _StbStrategyWrapper(
        cfg=cfg,
        query="dummy",
        retriever=None,
        tool_root_dir="/tmp/tools",
    )
    assert wrapper.evolution_temperature is None
    assert wrapper.base_temp == 0.9  # legacy default preserved


# ---------------------------------------------------------------------------
# Test 4: run_memetic_strategy routes evolution T to every evo call site
# ---------------------------------------------------------------------------


class _CapturingLLM:
    """Mock LLM that captures the ``temperature=...`` kwarg of every call.

    The strategy invokes ``llm.parse_with_messages(messages, tools, ...)``
    on every evolutionary subprocess step (seed, mutation, crossover,
    refinement). We capture the kwargs and assert each call carries the
    expected T.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        # Stable canned response — one valid pseudotool block so the
        # strategy can extract a child without crashing.
        self._canned_content = "<function_description>mock pseudotool</function_description>"
        self.openai_key = ""
        self.base_url = None
        self.model = "mock-model"

    def parse_with_messages(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
        process_id: int = 0,
        key_pos: Any = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], int, int]:
        self.calls.append(
            {
                "temperature": kwargs.get("temperature"),
                "n_messages": len(messages),
            }
        )
        return ({"role": "assistant", "content": self._canned_content}, 0, 0)


class _MockRetriever:
    """Mock retriever returning a tiny, stable tool list."""

    def retrieving(
        self, query: str, top_k: int
    ) -> tuple[list[str], list[str], list[float]]:
        return (
            ["mock_tool_1", "mock_tool_2"],
            ["mock tool 1 description", "mock tool 2 description"],
            [0.9, 0.5],
        )


def _patch_rapidapi_retrieve(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch ``retrieve_rapidapi_tools`` to return a stable mock payload.

    Avoids the real RapidAPI on-disk catalog dependency.
    """
    import toolbench.inference.Downstream_tasks.strategies as strategies_mod

    def _mock_retrieve(retriever, query, top_k, tool_root_dir):
        api_list = [
            {
                "category_name": "mock_cat",
                "tool_name": "mock_tool",
                "api_name": "mock_api",
            }
        ]
        mt = [
            {
                "category": "mock_cat",
                "tool_name": "mock_tool",
                "api_name": "mock_api",
                "score": 0.9,
                "description": "mock tool description",
            }
        ]
        return ({"api_list": api_list}, mt)

    monkeypatch.setattr(strategies_mod, "retrieve_rapidapi_tools", _mock_retrieve)


def test_run_memetic_strategy_uses_evolution_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All evolutionary call sites inside ``run_memetic_strategy`` receive T=1.5.

    This is THE key assertion of the split-temperatures track: the right
    temperature flows to the right call site. We do NOT test that the
    DFSDT trigger call uses T=1.0 here — that path is outside
    ``run_memetic_strategy`` and is exercised by the existing
    integration tests; this smoke focuses on the evolution subprocess.
    """
    from toolbench.inference.Downstream_tasks.strategies import run_memetic_strategy

    _patch_rapidapi_retrieve(monkeypatch)

    llm = _CapturingLLM()

    wrapper = types.SimpleNamespace(
        # Required attributes for run_memetic_strategy
        input_description="find a way to translate text into french",
        retriever=_MockRetriever(),
        retrieved_api_nums=3,
        tool_root_dir="/tmp/tools",
        tool_memory={},
        process_id=0,
        population_size=2,
        generation_num=2,
        memetic=True,  # enables llm_refine call site
        similarity_threshold=0.95,
        final_tool_budget=3,
        top_k_refine=3,
        # Split-temperatures wiring: evolution_temperature set, base_temp also
        # set (legacy default). The lookup chain must prefer
        # evolution_temperature.
        evolution_temperature=1.5,
        base_temp=0.9,
        fitness_method="baseline_jaccard",
    )

    # Drive the strategy. llm_output is the DFSDT-trigger output; in the
    # memetic path it is parsed for ancestor blocks.
    llm_output = "<function_description>seed ancestor pseudotool</function_description>"
    run_memetic_strategy(llm_output, wrapper, llm)

    # We must have made at least one LLM call (seed population). With
    # generations=2 and population_size=2 and memetic=True, we expect:
    #   - 1× seed population (per ancestor)
    #   - crossover/mutation calls until next_gen reaches population_size
    #   - refinement calls (one per non-elite child)
    assert llm.calls, "run_memetic_strategy made no LLM calls — fixture broken"

    seen_temps = {c["temperature"] for c in llm.calls}
    # Every captured call must carry T=1.5 (the evolution-subprocess T).
    # The legacy fallback (0.9) MUST NOT appear because evolution_temperature
    # was explicitly set on the wrapper.
    assert seen_temps == {1.5}, (
        f"evolution call sites must all receive T=1.5; "
        f"got distinct Ts = {seen_temps} across {len(llm.calls)} calls"
    )


def test_run_memetic_strategy_falls_back_to_base_temp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Back-compat: when evolution_temperature is None, base_temp (0.9) wins.

    This is the path used by the 231 existing memetic result dirs.
    """
    from toolbench.inference.Downstream_tasks.strategies import run_memetic_strategy

    _patch_rapidapi_retrieve(monkeypatch)

    llm = _CapturingLLM()
    wrapper = types.SimpleNamespace(
        input_description="legacy query",
        retriever=_MockRetriever(),
        retrieved_api_nums=3,
        tool_root_dir="/tmp/tools",
        tool_memory={},
        process_id=0,
        population_size=2,
        generation_num=2,
        memetic=True,
        similarity_threshold=0.95,
        final_tool_budget=3,
        top_k_refine=3,
        evolution_temperature=None,  # NOT set — legacy config path
        base_temp=0.9,
        fitness_method="baseline_jaccard",
    )

    llm_output = "<function_description>legacy ancestor</function_description>"
    run_memetic_strategy(llm_output, wrapper, llm)

    assert llm.calls
    seen_temps = {c["temperature"] for c in llm.calls}
    assert seen_temps == {0.9}, (
        f"legacy back-compat path must use base_temp=0.9; got {seen_temps}"
    )


def test_run_memetic_strategy_falls_back_to_default_when_nothing_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When neither evolution_temperature nor base_temp is set, default is 1.5.

    This is the production default for the split-temperature path.
    """
    from toolbench.inference.Downstream_tasks.strategies import run_memetic_strategy

    _patch_rapidapi_retrieve(monkeypatch)

    llm = _CapturingLLM()
    wrapper = types.SimpleNamespace(
        input_description="brand new wrapper, no T fields",
        retriever=_MockRetriever(),
        retrieved_api_nums=3,
        tool_root_dir="/tmp/tools",
        tool_memory={},
        process_id=0,
        population_size=2,
        generation_num=2,
        memetic=True,
        similarity_threshold=0.95,
        final_tool_budget=3,
        top_k_refine=3,
        evolution_temperature=None,
        # base_temp deliberately omitted — getattr(..., "base_temp", None)
        # must return None and the strategy must fall through to 1.5.
        fitness_method="baseline_jaccard",
    )

    llm_output = "<function_description>ancestor</function_description>"
    run_memetic_strategy(llm_output, wrapper, llm)

    assert llm.calls
    seen_temps = {c["temperature"] for c in llm.calls}
    assert seen_temps == {1.5}, (
        f"default evolution T must be 1.5 when nothing else is set; got {seen_temps}"
    )


if __name__ == "__main__":
    # Allow ``python tests/smoke_split_temperatures.py`` for quick checks
    # without pytest.
    sys.exit(pytest.main([__file__, "-v"]))
