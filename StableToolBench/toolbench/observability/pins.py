"""Model-pin loader — single source of truth for all model identifiers.

All code that needs a model name should call ``load_pins()`` and look up the
value by key rather than hardcoding the string inline.  This keeps
``configs/model_pins.yaml`` as the one place to bump a model.

Usage::

    from toolbench.observability.pins import load_pins

    pins = load_pins()
    judge_model = pins["evaluators"]["judge"]
    agent_model = pins["agents"]["main_solver"]
    embedder    = pins["embedders"]["toolret"]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Path to the pin file relative to this module's location.
# Traversal: StableToolBench/toolbench/observability/ → ../../.. → repo_root
_DEFAULT_PIN_FILE_REL = Path(__file__).parent.parent.parent.parent / "configs" / "model_pins.yaml"


def load_pins(repo_root: Path | None = None) -> dict[str, Any]:
    """Load ``configs/model_pins.yaml`` — single source of truth for model identifiers.

    Args:
        repo_root: Root of the git repository.  If ``None``, inferred from this
            module's location (works for both installed and in-tree use).

    Returns:
        Dict with top-level keys ``agents``, ``evaluators``, ``embedders``.
        All values are plain strings (model IDs).

    Raises:
        FileNotFoundError: If the pin file cannot be found at the resolved path.
        KeyError: If the YAML is missing a required top-level key.

    Notes:
        The return value is intentionally a plain ``dict`` (not a Pydantic model) so
        that scripts with minimal dependencies can import this without the full
        Pydantic stack.  Callers that want type-safe access should use
        ``toolbench.runner.schema.RunConfig`` instead.
    """
    if repo_root is not None:
        pin_path = Path(repo_root) / "configs" / "model_pins.yaml"
    else:
        pin_path = _DEFAULT_PIN_FILE_REL.resolve()

    if not pin_path.exists():
        raise FileNotFoundError(
            f"model_pins.yaml not found at {pin_path}. "
            "Pass repo_root explicitly if running from outside the repo."
        )

    with open(pin_path, encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}

    # Validate required top-level keys
    required = {"agents", "evaluators", "embedders"}
    missing = required - data.keys()
    if missing:
        raise KeyError(
            f"model_pins.yaml is missing required top-level key(s): {sorted(missing)}"
        )

    return data


def get_judge_model(repo_root: Path | None = None) -> str:
    """Convenience accessor — return the pinned judge model string.

    Args:
        repo_root: Passed through to ``load_pins()``.

    Returns:
        Dated model tag for the STB pass-rate judge.
    """
    return load_pins(repo_root)["evaluators"]["judge"]


def get_agent_model(agent_key: str, repo_root: Path | None = None) -> str:
    """Convenience accessor — return a pinned agent model string.

    Args:
        agent_key: Key under ``agents`` in model_pins.yaml.
        repo_root: Passed through to ``load_pins()``.

    Returns:
        Dated model tag.

    Raises:
        KeyError: If ``agent_key`` is not in the ``agents`` section.
    """
    pins = load_pins(repo_root)
    if agent_key not in pins["agents"]:
        raise KeyError(
            f"Agent key {agent_key!r} not in model_pins.yaml agents section. "
            f"Available: {sorted(pins['agents'].keys())}"
        )
    return pins["agents"][agent_key]
