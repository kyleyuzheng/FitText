"""Plan expansion — DispatchSpec → list of :class:`Cell` plans.

Each cell carries everything the executor needs to launch one run:
  * cell_id (stable, deterministic, filesystem-safe)
  * kind: ``technique`` or ``baseline``
  * concrete model name / technique label / baseline label
  * benchmark name + split + n_queries
  * the generated per-cell run YAML written to disk so ``run.py`` /
    ``run_baseline.py`` can consume it unchanged

Technique → variant mapping is centralised in
:func:`technique_to_variant`. Unknown techniques degrade to the named
variant if it matches the RunConfig Literal; otherwise we raise so a typo
fails loudly instead of silently dropping a cell.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from .spec import DispatchSpec, DispatchModelEntry


# ---------------------------------------------------------------------------
# Technique → variant + hparam mapping
# ---------------------------------------------------------------------------

# Maps user-facing technique labels (in dispatch spec) to:
#   - variant: one of {single_pass, multi_turn, scattershot, memetic}
#     (RunConfig Pydantic literal — must match exactly)
#   - fittext_overrides: dict of FitText hparams to override on top of the
#     template (e.g. memetic forces a specific selection/memory penalty)
#   - extra_overrides: dict of values to drop into the run YAML's ``extra``
#     section. Used for technique labels that don't map to a distinct
#     variant (e.g. just_query or dbd).
TECHNIQUE_REGISTRY: dict[str, dict[str, Any]] = {
    "just_query": {
        # Zero-retrieval baseline: feed the raw query to the agent, no
        # pseudo-tool description. Implemented inside the eval loop; the
        # dispatcher only needs to pass the label through.
        "variant": "single_pass",
        "fittext_overrides": {
            "population_size": 1,
            "generations": 1,
            "selection": "none",
            "mutation_prob": 0.0,
            "crossover_prob": 0.0,
            "memory_penalty": 0.0,
        },
        "extra_overrides": {"technique": "just_query"},
    },
    "single_pass": {
        "variant": "single_pass",
        "fittext_overrides": {
            "population_size": 1,
            "generations": 1,
            "selection": "none",
        },
        "extra_overrides": {"technique": "single_pass"},
    },
    "multi_turn": {
        "variant": "multi_turn",
        "fittext_overrides": {
            "population_size": 1,
            "generations": 3,
            "selection": "fitness",
        },
        "extra_overrides": {"technique": "multi_turn"},
    },
    "multishot": {  # alias: spec name → variant scattershot
        "variant": "scattershot",
        "fittext_overrides": {
            "population_size": 6,
            "generations": 1,
            "selection": "none",
        },
        "extra_overrides": {"technique": "scattershot"},
    },
    "scattershot": {
        "variant": "scattershot",
        "fittext_overrides": {
            "population_size": 6,
            "generations": 1,
            "selection": "none",
        },
        "extra_overrides": {"technique": "scattershot"},
    },
    "dbd": {
        # Disrupt-by-diversification — empirical variant routed through the
        # memetic loop with selection=fitness but memory_penalty=0.
        "variant": "memetic",
        "fittext_overrides": {
            "population_size": 6,
            "generations": 3,
            "selection": "fitness",
            "memory_penalty": 0.0,
            "crossover_prob": 0.5,
            "mutation_prob": 0.5,
        },
        "extra_overrides": {"technique": "dbd"},
    },
    "memetic": {
        # The published Memetic Retrieval path.
        "variant": "memetic",
        "fittext_overrides": {
            "population_size": 6,
            "generations": 3,
            "selection": "fitness",
            "memory_penalty": 0.5,
            "crossover_prob": 0.5,
            "mutation_prob": 0.5,
        },
        "extra_overrides": {"technique": "memetic"},
    },
}


def technique_to_variant(technique: str) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Resolve a technique label to (variant, fittext_overrides, extra_overrides).

    Args:
        technique: User-facing technique label from the dispatch spec.

    Returns:
        Tuple of ``(variant, fittext_overrides, extra_overrides)``.

    Raises:
        ValueError: If the technique is unknown.
    """
    if technique in TECHNIQUE_REGISTRY:
        entry = TECHNIQUE_REGISTRY[technique]
        return entry["variant"], dict(entry["fittext_overrides"]), dict(entry["extra_overrides"])
    raise ValueError(
        f"Unknown technique '{technique}'. "
        f"Known: {sorted(TECHNIQUE_REGISTRY)}. "
        f"Add it to TECHNIQUE_REGISTRY in toolbench/dispatcher/plan.py."
    )


# ---------------------------------------------------------------------------
# Model pin resolution
# ---------------------------------------------------------------------------


def resolve_model_pin(
    entry: DispatchModelEntry, pins_path: Path
) -> tuple[str, str | None]:
    """Resolve a DispatchModelEntry to (model_name, revision).

    Args:
        entry: The model entry from the dispatch spec.
        pins_path: Path to ``configs/model_pins.yaml``.

    Returns:
        Tuple of (dated model tag, revision string or None).

    Raises:
        ValueError: If neither pin nor name is set, or the dotted pin path
            cannot be resolved in the pins file.
        FileNotFoundError: If pins_path is missing and a pin is requested.
    """
    if entry.name:
        return entry.name, entry.revision

    if not entry.pin:
        raise ValueError("DispatchModelEntry must set either 'pin' or 'name'.")

    if not pins_path.exists():
        raise FileNotFoundError(f"model_pins.yaml not found at {pins_path}")
    with pins_path.open("r", encoding="utf-8") as fh:
        pins = yaml.safe_load(fh) or {}

    # Walk dotted path, e.g. "agents.gpt_4_1_mini"
    node: Any = pins
    parts = entry.pin.split(".")
    for p in parts:
        if not isinstance(node, dict) or p not in node:
            raise ValueError(
                f"Cannot resolve pin '{entry.pin}' in {pins_path} "
                f"(missing segment '{p}')."
            )
        node = node[p]
    if not isinstance(node, str):
        raise ValueError(
            f"Pin '{entry.pin}' resolved to non-string value {node!r}."
        )
    return node, entry.revision


# ---------------------------------------------------------------------------
# Cell
# ---------------------------------------------------------------------------


@dataclass
class Cell:
    """One concrete experiment cell — a single ``run.py``/``run_baseline.py`` invocation.

    Attributes:
        cell_id: Stable, deterministic, filesystem-safe identifier.
        kind: ``"technique"`` (FitText variant) or ``"baseline"``.
        label: Technique label (e.g. ``"memetic"``) or baseline name
            (e.g. ``"less_is_more"``).
        model_name: Resolved model identifier.
        model_revision: HF revision or None.
        benchmark_name: ``"toolret"`` or ``"stabletoolbench"``.
        split: Single split string (e.g. ``"code"`` or ``"G2"``).
        n_queries: Cap for this split; None = full split.
        per_cell_budget_usd: max_cost_usd injected into the generated YAML.
        run_yaml_dict: The generated run YAML content (dict form). Written
            to disk by the executor at launch time.
        host: Filled by the executor: ``"local"`` or an ssh alias.
    """

    cell_id: str
    kind: Literal["technique", "baseline"]
    label: str
    model_name: str
    model_revision: str | None
    benchmark_name: str
    split: str
    n_queries: int | None
    per_cell_budget_usd: float
    run_yaml_dict: dict[str, Any] = field(default_factory=dict)
    host: str = "local"

    def is_baseline(self) -> bool:
        return self.kind == "baseline"


# ---------------------------------------------------------------------------
# Cell ID hashing
# ---------------------------------------------------------------------------


def _sanitize(s: str) -> str:
    """Make a string safe for filesystem use (no slashes/colons/spaces)."""
    return (
        s.replace("/", "_")
        .replace(":", "_")
        .replace(" ", "_")
        .replace(".", "-")
    )


def _make_cell_id(
    kind: str, label: str, model_name: str, benchmark: str, split: str
) -> str:
    """Build a deterministic, short, human-readable cell id.

    Format: ``<kind>__<label>__<model_short>__<benchmark>_<split>__<hash6>``
    where hash6 is the first 6 hex chars of sha256(<full identity>) to
    guarantee uniqueness even when model names collide after sanitisation.
    """
    canonical = f"{kind}|{label}|{model_name}|{benchmark}|{split}"
    h6 = hashlib.sha256(canonical.encode()).hexdigest()[:6]
    # Use only the stable prefix of dated model tags for readability.
    model_short = _sanitize(model_name).split("-2025")[0].split("-2026")[0]
    return f"{_sanitize(kind)}__{_sanitize(label)}__{model_short}__{_sanitize(benchmark)}_{_sanitize(split)}__{h6}"


# ---------------------------------------------------------------------------
# Cell expansion
# ---------------------------------------------------------------------------


def _base_run_yaml(
    model_name: str,
    model_revision: str | None,
    benchmark_name: str,
    split: str,
    n_queries: int | None,
    variant: str,
    fittext_overrides: dict[str, Any],
    extra_overrides: dict[str, Any],
    per_cell_budget_usd: float,
) -> dict[str, Any]:
    """Build the YAML dict for one cell's run config.

    The result inherits from the standard ``_base`` files so embedder /
    evaluator / infra pins stay consistent across cells.
    """
    # Default FitText hparams (paper values), overridden by technique-specific
    # values from TECHNIQUE_REGISTRY.
    fittext: dict[str, Any] = {
        "variant": variant,
        "population_size": 6,
        "generations": 3,
        "mutation_prob": 0.5,
        "crossover_prob": 0.5,
        "memory_penalty": 0.5,
        "fitness_alpha": 0.7,
        "jaccard_threshold": 0.3,
        "selection": "fitness",
        "top_k_retrieval": 5,
    }
    fittext.update(fittext_overrides)

    yaml_dict: dict[str, Any] = {
        "inherit": [
            "configs/_base/benchmarks.yaml",
            "configs/_base/embedder.yaml",
            "configs/_base/evaluators.yaml",
            "configs/_base/infra.yaml",
        ],
        "model": {
            "name": model_name,
            "revision": model_revision,
            "max_tokens": 2048,
            "temperature": 0.0,
            "seed": 42,
        },
        "fittext": fittext,
        "benchmark": {
            "name": benchmark_name,
            "splits": [split],
            "n_queries_per_split": n_queries,
        },
        "budget": {"max_cost_usd": per_cell_budget_usd},
        "extra": dict(extra_overrides),
    }
    return yaml_dict


def expand_cells(spec: DispatchSpec, pins_path: Path) -> list[Cell]:
    """Expand a :class:`DispatchSpec` into a list of concrete cells.

    Iteration order::

        for model in spec.models:
            for benchmark in spec.benchmarks:
                for split in benchmark.splits:
                    for technique in spec.techniques:
                        yield Cell(kind='technique', ...)
                    for baseline in spec.baselines:
                        yield Cell(kind='baseline', ...)

    Args:
        spec: Validated DispatchSpec.
        pins_path: Path to ``configs/model_pins.yaml`` for pin resolution.

    Returns:
        List of Cell instances. Total count =
        ``|models| * sum(|splits_b|) * (|techniques| + |baselines|)``.

    Raises:
        ValueError: If a technique label is unknown.
    """
    cells: list[Cell] = []

    for model_entry in spec.models:
        model_name, model_revision = resolve_model_pin(model_entry, pins_path)

        for bench in spec.benchmarks:
            for split in bench.splits:
                n_queries = bench.n_queries_per_split

                # Techniques
                for technique in spec.techniques:
                    variant, fittext_ov, extra_ov = technique_to_variant(technique)
                    extra_ov_full = dict(extra_ov)
                    extra_ov_full["dispatch_spec"] = spec.name
                    extra_ov_full["dispatch_kind"] = "technique"
                    extra_ov_full["dispatch_label"] = technique
                    cell_id = _make_cell_id(
                        "technique", technique, model_name, bench.name, split
                    )
                    yaml_dict = _base_run_yaml(
                        model_name=model_name,
                        model_revision=model_revision,
                        benchmark_name=bench.name,
                        split=split,
                        n_queries=n_queries,
                        variant=variant,
                        fittext_overrides=fittext_ov,
                        extra_overrides=extra_ov_full,
                        per_cell_budget_usd=spec.budget.per_cell_usd,
                    )
                    cells.append(
                        Cell(
                            cell_id=cell_id,
                            kind="technique",
                            label=technique,
                            model_name=model_name,
                            model_revision=model_revision,
                            benchmark_name=bench.name,
                            split=split,
                            n_queries=n_queries,
                            per_cell_budget_usd=spec.budget.per_cell_usd,
                            run_yaml_dict=yaml_dict,
                        )
                    )

                # Baselines — also need a valid RunConfig, but variant is
                # forced to single_pass (drop-in; the baseline driver reads
                # cfg.extra.baseline_name and dispatches accordingly).
                for baseline in spec.baselines:
                    extra_ov_full = {
                        "baseline_name": baseline,
                        "dispatch_spec": spec.name,
                        "dispatch_kind": "baseline",
                        "dispatch_label": baseline,
                    }
                    cell_id = _make_cell_id(
                        "baseline", baseline, model_name, bench.name, split
                    )
                    yaml_dict = _base_run_yaml(
                        model_name=model_name,
                        model_revision=model_revision,
                        benchmark_name=bench.name,
                        split=split,
                        n_queries=n_queries,
                        variant="single_pass",
                        fittext_overrides={
                            "population_size": 1,
                            "generations": 1,
                            "selection": "none",
                            "mutation_prob": 0.0,
                            "crossover_prob": 0.0,
                            "memory_penalty": 0.0,
                        },
                        extra_overrides=extra_ov_full,
                        per_cell_budget_usd=spec.budget.per_cell_usd,
                    )
                    cells.append(
                        Cell(
                            cell_id=cell_id,
                            kind="baseline",
                            label=baseline,
                            model_name=model_name,
                            model_revision=model_revision,
                            benchmark_name=bench.name,
                            split=split,
                            n_queries=n_queries,
                            per_cell_budget_usd=spec.budget.per_cell_usd,
                            run_yaml_dict=yaml_dict,
                        )
                    )

    return cells
