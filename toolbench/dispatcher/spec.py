"""Dispatch spec schema (Pydantic v2).

A dispatch YAML describes the cartesian product to evaluate:
``models x techniques x baselines x benchmarks x splits``.

The schema is intentionally permissive on the `technique` and `baseline`
strings — unknown labels are passed through to the generated run YAML via
``extra.technique`` / ``extra.baseline``. The Pydantic RunConfig schema only
constrains the `variant` literal; technique→variant mapping lives in
``plan.technique_to_variant``.

Example minimal spec::

    name: cheap_sota_small
    models:
      - pin: agents.gpt_4_1_mini
    techniques: [single_pass, memetic]
    baselines: [less_is_more]
    benchmarks:
      - name: toolret
        splits: [code]
        n_queries_per_split: 50
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ConfigDict


# ---------------------------------------------------------------------------
# Sub-schemas
# ---------------------------------------------------------------------------


class DispatchModelEntry(BaseModel):
    """One model entry in a dispatch spec.

    Prefer ``pin`` (dotted path into configs/model_pins.yaml). ``name`` is
    retained only for ad-hoc local experiments.
    """

    model_config = ConfigDict(extra="forbid")

    pin: str | None = Field(
        default=None,
        description="Dotted path into model_pins.yaml (e.g. 'agents.gpt_4_1_mini').",
    )
    name: str | None = Field(
        default=None,
        description="Literal model tag for ad-hoc local experiments.",
    )
    revision: str | None = Field(
        default=None, description="Optional HF revision hash for local models."
    )


class DispatchBenchmarkEntry(BaseModel):
    """One benchmark entry in a dispatch spec."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["toolret", "stabletoolbench"]
    splits: list[str] = Field(
        ..., description="Domain/complexity splits to evaluate, e.g. ['code']."
    )
    n_queries_per_split: int | None = Field(
        default=None, description="Cap per split. None = all queries."
    )


class DispatchParallelism(BaseModel):
    """Concurrency knobs for the dispatcher."""

    model_config = ConfigDict(extra="forbid")

    max_local_workers: int = Field(
        default=6,
        ge=1,
        description="ProcessPoolExecutor size on the coordinator host.",
    )
    hosts: list[str] = Field(
        default_factory=list,
        description=(
            "Additional remote hosts (ssh aliases). Empty = local only. "
            "Cells are sharded round-robin across coordinator + hosts."
        ),
    )
    shard_split: Literal["by_cell"] = Field(
        default="by_cell",
        description="Sharding granularity. Only 'by_cell' supported today.",
    )
    resume: bool = Field(
        default=True,
        description="Skip cells whose <out_dir>/<cell_id>/result.json exists.",
    )


class DispatchBudget(BaseModel):
    """Cost circuit-breaker — per cell and globally."""

    model_config = ConfigDict(extra="forbid")

    per_cell_usd: float = Field(
        default=5.0, ge=0.0, description="max_cost_usd override for each cell."
    )
    total_usd: float = Field(
        default=100.0,
        ge=0.0,
        description="Global ceiling; dispatcher aborts further launches when exceeded.",
    )


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


class DispatchSpec(BaseModel):
    """Top-level dispatch spec.

    One YAML file → one dispatch run → many cells.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Dispatch name (used in dispatch_id).")
    description: str = Field(default="", description="Human-readable annotation.")

    models: list[DispatchModelEntry] = Field(..., min_length=1)
    techniques: list[str] = Field(
        default_factory=list,
        description=(
            "FitText technique labels (just_query, single_pass, multishot, dbd, "
            "memetic, etc.). Resolved to variant by plan.technique_to_variant."
        ),
    )
    baselines: list[str] = Field(
        default_factory=list,
        description="Baseline names (less_is_more, reinvoke, xu2024, colt, ...).",
    )
    benchmarks: list[DispatchBenchmarkEntry] = Field(..., min_length=1)

    parallelism: DispatchParallelism = Field(default_factory=DispatchParallelism)
    budget: DispatchBudget = Field(default_factory=DispatchBudget)

    def n_techniques(self) -> int:
        return len(self.techniques)

    def n_baselines(self) -> int:
        return len(self.baselines)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_dispatch_spec(path: Path | str) -> DispatchSpec:
    """Load and validate a dispatch YAML.

    Args:
        path: Path to the dispatch YAML (e.g.
            ``configs/dispatch/cheap_sota_small.yaml``).

    Returns:
        Validated :class:`DispatchSpec`.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValidationError: If the YAML does not conform to the schema.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dispatch spec not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}
    return DispatchSpec.model_validate(raw)
