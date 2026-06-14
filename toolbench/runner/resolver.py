"""YAML → fully-resolved RunConfig with inheritance and env-var substitution.

Resolution pipeline:
  1. Load the target YAML.
  2. For each path listed in ``inherit``, load the base YAML and deep-merge
     (child overrides parent; later entries override earlier entries).
  3. Substitute ``${VAR}`` and ``${VAR:-default}`` patterns with env vars.
  4. Resolve model pin aliases through ``configs/model_pins.yaml``.
  5. Validate the merged dict through Pydantic ``RunConfig``.
  6. Auto-fill ``run_id`` if blank.

Design notes:
  - No hardcoded paths or model names in this module — all values come from YAML.
  - Secrets (API keys) are substituted but must not appear in config_hash().
    The hash function in schema.py excludes them.
  - ``dump_resolved_config`` writes the fully-resolved YAML next to results
    for audit — §4.1 reproducibility.
"""

from __future__ import annotations

import os
import re
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .schema import RunConfig


# ---------------------------------------------------------------------------
# Env-var substitution
# ---------------------------------------------------------------------------

_ENV_PATTERN = re.compile(r"\$\{([^}:]+)(?::-(.*?))?\}")


def _substitute_env_vars(value: Any) -> Any:
    """Recursively substitute ``${VAR}`` and ``${VAR:-default}`` in strings.

    Args:
        value: Arbitrary YAML-parsed Python value (str, dict, list, etc.)

    Returns:
        Same structure with env-var placeholders expanded.

    Raises:
        KeyError: If a required env var (no default) is missing.
    """
    if isinstance(value, str):
        def _replace(m: re.Match) -> str:
            var_name = m.group(1)
            default = m.group(2)  # None if no ':-' suffix
            env_val = os.environ.get(var_name)
            if env_val is not None:
                return env_val
            if default is not None:
                return default
            raise KeyError(
                f"Required env var '${{{var_name}}}' not set and has no default."
            )
        return _ENV_PATTERN.sub(_replace, value)
    elif isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_substitute_env_vars(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# Model pin resolution
# ---------------------------------------------------------------------------

def _load_model_pins(repo_root: Path) -> dict[str, Any]:
    """Load ``configs/model_pins.yaml``.

    Args:
        repo_root: Repository root containing ``configs/model_pins.yaml``.

    Returns:
        Parsed model pin dictionary.
    """
    pin_path = repo_root / "configs" / "model_pins.yaml"
    with pin_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _lookup_pin(pins: dict[str, Any], dotted_path: str) -> str:
    """Resolve a dotted model-pin path such as ``agents.main_solver``."""
    node: Any = pins
    for part in dotted_path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"Unknown model pin: {dotted_path}")
        node = node[part]
    if not isinstance(node, str):
        raise TypeError(f"Model pin {dotted_path!r} did not resolve to a string")
    return node


def _resolve_model_pins(value: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    """Replace run-config pin aliases with concrete model IDs.

    Supported fields:
      - ``model.pin`` -> ``model.name``
      - ``evaluator.judge_pin`` -> ``evaluator.judge_model``
      - ``evaluator.simulator_pin`` -> ``evaluator.simulator_model``
      - ``embedder.pin`` -> ``embedder.name``

    The resolved ``RunConfig`` still contains concrete model IDs so manifests
    and cache keys remain self-contained, while checked-in YAML can keep
    ``configs/model_pins.yaml`` as the sole model-name source.
    """
    resolved = deepcopy(value)
    pins: dict[str, Any] | None = None

    def lookup(dotted_path: str) -> str:
        nonlocal pins
        if pins is None:
            pins = _load_model_pins(repo_root)
        return _lookup_pin(pins, dotted_path)

    model = resolved.get("model")
    if isinstance(model, dict) and model.get("pin"):
        model["name"] = lookup(str(model.pop("pin")))

    evaluator = resolved.get("evaluator")
    if isinstance(evaluator, dict):
        if evaluator.get("judge_pin"):
            evaluator["judge_model"] = lookup(str(evaluator.pop("judge_pin")))
        if evaluator.get("simulator_pin"):
            evaluator["simulator_model"] = lookup(str(evaluator.pop("simulator_pin")))

    embedder = resolved.get("embedder")
    if isinstance(embedder, dict) and embedder.get("pin"):
        embedder["name"] = lookup(str(embedder.pop("pin")))

    return resolved


# ---------------------------------------------------------------------------
# Deep merge helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins on conflict).

    Args:
        base: Base dictionary (e.g. from _base/*.yaml).
        override: Override dictionary (e.g. from a run config).

    Returns:
        New merged dict; neither input is mutated.
    """
    result = deepcopy(base)
    for key, override_val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(override_val, dict):
            result[key] = _deep_merge(result[key], override_val)
        else:
            result[key] = deepcopy(override_val)
    return result


# ---------------------------------------------------------------------------
# Git short commit (for run_id auto-fill)
# ---------------------------------------------------------------------------

def _git_short_commit(repo_root: Path | None = None) -> str:
    """Return the 8-char HEAD commit hash of the repo.

    Args:
        repo_root: Path to git repository root. Defaults to cwd.

    Returns:
        8-char hex string, or ``'unknown'`` if git is unavailable.
    """
    try:
        cwd = str(repo_root) if repo_root else None
        result = subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"],
            capture_output=True,
            text=True,
            cwd=cwd,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# Main resolver
# ---------------------------------------------------------------------------

def _load_and_merge(
    path: Path,
    repo_root: Path,
    visited: set[Path],
) -> dict[str, Any]:
    """Recursively load a YAML and merge its inheritance chain.

    Resolution order:
      1. Load all ``inherit`` bases (recursively, depth-first).
      2. Merge them left-to-right (later overrides earlier).
      3. Apply this file's own keys on top.

    Args:
        path: Absolute path to the YAML file to load.
        repo_root: Repo root for resolving relative inherit paths.
        visited: Set of already-visited paths (cycle guard).

    Returns:
        Fully merged dict for this file and all its ancestors.

    Raises:
        FileNotFoundError: If any inherited file is missing.
        ValueError: If a circular inheritance is detected.
    """
    if path in visited:
        raise ValueError(f"Circular inheritance detected: {path}")
    visited = visited | {path}

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    inherit_paths: list[str] = raw.pop("inherit", [])
    merged: dict[str, Any] = {}

    for inherit_rel in inherit_paths:
        inherit_path = (repo_root / inherit_rel).resolve()
        if not inherit_path.exists():
            raise FileNotFoundError(
                f"Inherited config not found: {inherit_path} "
                f"(referenced from {path})"
            )
        base_data = _load_and_merge(inherit_path, repo_root, visited)
        merged = _deep_merge(merged, base_data)

    # Apply this file's own keys on top of accumulated bases
    merged = _deep_merge(merged, raw)
    return merged


def resolve_config(path: Path | str, repo_root: Path | None = None) -> RunConfig:
    """Load a run YAML and resolve it into a validated ``RunConfig``.

    Resolution steps:
      1. Load the target YAML.
      2. Load and merge ``inherit`` base YAMLs (order: later overrides earlier).
      3. Substitute env vars in all string values.
      4. Resolve model pin aliases.
      5. Validate via Pydantic.
      6. Auto-fill ``run_id`` if not set.

    Args:
        path: Path to the run YAML file (absolute or relative to cwd).
        repo_root: Root of the git repo for commit hash lookup. If None, uses
            the directory containing ``path``.

    Returns:
        Validated, fully-resolved ``RunConfig`` instance.

    Raises:
        FileNotFoundError: If the config file or any inherited base is missing.
        KeyError: If a required env var is not set.
        pydantic.ValidationError: If the merged config fails schema validation.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    # Determine base directory for resolving relative inherit paths
    config_dir = path.parent
    if repo_root is None:
        # Try to find repo root by walking up from config file
        repo_root = _find_repo_root(path) or config_dir

    # 1. Recursively collect and merge inherited configs, then apply this file on top.
    merged = _load_and_merge(path, repo_root, visited=set())

    # 3. Env-var substitution (after merge so base defaults can reference env vars too)
    merged = _substitute_env_vars(merged)

    # 4. Resolve model pin aliases through configs/model_pins.yaml
    merged = _resolve_model_pins(merged, repo_root)

    # 5. Pydantic validation
    cfg = RunConfig.model_validate(merged)

    # 6. Auto-fill run_id
    if cfg.run_id is None:
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
        short_commit = _git_short_commit(repo_root)
        hash_prefix = cfg.config_hash()[:8]
        cfg = cfg.model_copy(update={"run_id": f"{ts}_{short_commit}_{hash_prefix}"})

    return cfg


def dump_resolved_config(cfg: RunConfig, path: Path | str) -> None:
    """Write the fully-resolved config as YAML for audit and reproducibility.

    The output file is written next to results so the exact config that
    produced a run can always be recovered. Per §4.1 and §5.2.

    Args:
        cfg: Resolved RunConfig instance.
        path: Output path (e.g. ``{output_dir}/{run_id}/config.resolved.yaml``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(
            cfg.model_dump(),
            f,
            default_flow_style=False,
            sort_keys=True,
            allow_unicode=True,
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _find_repo_root(start: Path) -> Path | None:
    """Walk up from ``start`` to find the git repo root (.git directory).

    Args:
        start: Starting path (file or directory).

    Returns:
        Path to repo root, or None if not found.
    """
    current = start if start.is_dir() else start.parent
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists():
            return parent
    return None
