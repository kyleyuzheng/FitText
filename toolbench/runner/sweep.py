"""Sweep YAML → list of per-cell run configs.

A sweep YAML specifies:
  - ``inherit_from``: a base run config YAML path
  - ``sweep``: list of {field, values} dicts
  - ``mode``: "one_at_a_time" (default) or "cartesian"

``expand_sweep()`` returns a list of dicts, each representing one cell's
overrides on top of the base config. These are written to YAML by
``scripts/expand_sweep.py``.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Nested field access helpers
# ---------------------------------------------------------------------------

def _set_nested(d: dict[str, Any], dotted_key: str, value: Any) -> dict[str, Any]:
    """Set a value at a dotted path in a nested dict (mutates in place).

    Args:
        d: Target dictionary.
        dotted_key: Dot-separated key path, e.g. ``'fittext.fitness_alpha'``.
        value: Value to set.

    Returns:
        The modified dictionary (same object as ``d``).
    """
    keys = dotted_key.split(".")
    node = d
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value
    return d


def _get_nested(d: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Get a value at a dotted path in a nested dict.

    Args:
        d: Source dictionary.
        dotted_key: Dot-separated key path.
        default: Value to return if path not found.

    Returns:
        Value at the path, or ``default`` if missing.
    """
    keys = dotted_key.split(".")
    node = d
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


# ---------------------------------------------------------------------------
# Sweep expansion
# ---------------------------------------------------------------------------

def _safe_val_str(value: Any) -> str:
    """Convert a value to a filesystem-safe string for use in filenames.

    Args:
        value: Any Python value.

    Returns:
        String safe for use in filenames (no slashes, colons, spaces).
    """
    s = str(value).replace("/", "_").replace(":", "_").replace(" ", "_")
    return s


def expand_sweep(sweep_path: Path | str) -> list[dict[str, Any]]:
    """Load a sweep YAML and return a list of per-cell override dicts.

    Each returned dict represents one experiment cell. Keys:
      - ``inherit_from``: path to base run config (from sweep YAML)
      - ``cell_id``: stable string identifier for this cell
      - ``overrides``: dict of field->value pairs to override in the base config
      - ``suggested_filename``: suggested output filename (without .yaml)

    Args:
        sweep_path: Path to the sweep YAML file.

    Returns:
        List of cell dicts, one per experiment cell.

    Raises:
        FileNotFoundError: If the sweep YAML does not exist.
        ValueError: If the sweep YAML has an invalid structure or unknown mode.
    """
    sweep_path = Path(sweep_path).resolve()
    if not sweep_path.exists():
        raise FileNotFoundError(f"Sweep config not found: {sweep_path}")

    with open(sweep_path) as f:
        sweep_spec = yaml.safe_load(f) or {}

    inherit_from: str = sweep_spec.get("inherit_from", "")
    if not inherit_from:
        raise ValueError(f"Sweep config missing 'inherit_from': {sweep_path}")

    sweep_params: list[dict[str, Any]] = sweep_spec.get("sweep", [])
    if not sweep_params:
        raise ValueError(f"Sweep config has empty 'sweep' list: {sweep_path}")

    mode: str = sweep_spec.get("mode", "one_at_a_time")
    sweep_name = sweep_path.stem

    cells: list[dict[str, Any]] = []

    if mode == "one_at_a_time":
        # For each parameter, generate one cell per value while holding all
        # other parameters at their base (YAML) values.
        for param in sweep_params:
            field: str = param["field"]
            values: list[Any] = param["values"]
            for val in values:
                safe_field = field.replace(".", "_")
                safe_val = _safe_val_str(val)
                cell_id = f"{sweep_name}__{safe_field}__{safe_val}"
                cells.append({
                    "inherit_from": inherit_from,
                    "cell_id": cell_id,
                    "overrides": {field: val},
                    "suggested_filename": cell_id,
                })

    elif mode == "cartesian":
        # Cartesian product of all parameter values.
        fields = [p["field"] for p in sweep_params]
        value_lists = [p["values"] for p in sweep_params]
        for combo in itertools.product(*value_lists):
            parts = []
            overrides: dict[str, Any] = {}
            for field, val in zip(fields, combo):
                safe_field = field.replace(".", "_")
                safe_val = _safe_val_str(val)
                parts.append(f"{safe_field}_{safe_val}")
                overrides[field] = val
            cell_id = f"{sweep_name}__{'__'.join(parts)}"
            cells.append({
                "inherit_from": inherit_from,
                "cell_id": cell_id,
                "overrides": overrides,
                "suggested_filename": cell_id,
            })

    else:
        raise ValueError(
            f"Unknown sweep mode '{mode}' in {sweep_path}. "
            "Expected 'one_at_a_time' or 'cartesian'."
        )

    return cells


def cell_to_run_yaml(cell: dict[str, Any]) -> dict[str, Any]:
    """Convert a cell dict from ``expand_sweep`` to a run YAML dict.

    The result can be written directly as YAML via ``yaml.dump()``.

    Args:
        cell: Cell dict from ``expand_sweep()``.

    Returns:
        Dict suitable for writing as a per-cell run YAML.
    """
    run_dict: dict[str, Any] = {
        "inherit": [cell["inherit_from"]],
        # Pull in the standard base configs too
        "# cell_id": cell["cell_id"],  # comment only
    }
    # Write overrides as nested keys
    for dotted_key, value in cell["overrides"].items():
        parts = dotted_key.split(".")
        if len(parts) == 2:
            section, field = parts
            run_dict.setdefault(section, {})[field] = value
        elif len(parts) == 1:
            run_dict[parts[0]] = value
        else:
            # 3+ levels deep — write as full dotted path in extra for now
            run_dict.setdefault("extra", {})["_sweep_override"] = {
                dotted_key: value
            }
    run_dict.setdefault("extra", {})["cell_id"] = cell["cell_id"]
    return run_dict
