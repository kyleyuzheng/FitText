"""Pydantic v2 schema for FitText experiment RunConfig.

Each field maps directly to a YAML key in configs/runs/*.yaml. The schema is
the single source of truth for what a valid run config looks like — all YAML
files are validated against it at resolution time.

Integration points (stubs for other Wave 1 tracks):
  - modelclient track: ``make_client(cfg.model.name)``
  - cost-logger track: ``ManifestWriter`` consumes ``cfg.config_hash()`` + ``cfg.run_id``
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field


class ModelSpec(BaseModel):
    """LLM agent model specification.

    Run YAMLs should reference ``configs/model_pins.yaml`` via ``model.pin``;
    the resolver materializes ``name`` before validation.
    """

    name: str = Field(
        ...,
        description="Resolved model identifier from configs/model_pins.yaml.",
    )
    revision: str | None = Field(
        default=None,
        description="HF revision hash for local/vLLM models. Preferred to provide explicitly.",
    )
    max_tokens: int = Field(default=2048, description="Max completion tokens per call.")
    temperature: float = Field(
        default=0.0,
        description=(
            "Sampling temperature for the **trigger / DFSDT outer-loop** call sites "
            "(planner-style decisions, single_pass / dbd / multi_turn / scattershot / "
            "just_query — techniques that have no separate evolutionary subprocess). "
            "0.0 for reproducibility on deterministic baselines; 1.5 for diversity-driven "
            "techniques (scattershot, multishot, dbd / multi_turn). For memetic, this is "
            "the DFSDT trigger temperature ONLY — set ``evolution_temperature`` for the "
            "memetic-subprocess temperature."
        ),
    )
    evolution_temperature: float | None = Field(
        default=None,
        description=(
            "Sampling temperature for the **memetic evolutionary subprocess** ONLY — "
            "population seeding (``generate_population_from_ancestor``), mutation, "
            "crossover, and LLM refinement (``llm_refine``). When ``None`` the strategy "
            "falls back to ``wrapper.base_temp`` (legacy 231-cell provenance path) and "
            "then to the strategy default (1.5). When set, this OVERRIDES both. "
            "Empirical basis: existing ``memetic_p5_g3_temp1.5`` partial-n cells on "
            "the legacy comparison solver achieved 17–29% on the hard-question subset where the T=0.9 "
            "path scored 0%. T=1.5 is a *rescue* temperature for hard cases — the right "
            "place for it is the memetic subprocess, NOT the DFSDT trigger."
        ),
    )
    seed: int = Field(default=42, description="RNG seed passed to API where supported.")


class FitTextHParams(BaseModel):
    """FitText evolutionary retrieval hyperparameters (Eq. 7).

    The four published variants map to configs:
      Single-Pass:  population_size=1, generations=1, selection='none'
      Multi-Turn:   population_size=1, generations>1, selection='fitness'
      Scattershot:  population_size>1, generations=1, selection='none'
      Memetic:      population_size>1, generations>1, selection='fitness', memory_penalty>0

    A fifth variant — Just-Query — is the zero-retrieval floor baseline:
      Just-Query:   no retrieval, no tool catalog; agent answers from prior
                    knowledge alone. This is the "everything off" comparator
                    that bounds the value contributed by every retrieval
                    method above it.
    """

    variant: Literal["single_pass", "multi_turn", "scattershot", "memetic", "just_query"] = Field(
        ..., description="Named FitText variant (maps to a canonical hyperparam config)."
    )
    population_size: int = Field(default=1, description="N — population size. 1 = no population.")
    generations: int = Field(default=1, description="G — number of generations. 1 = no evolution.")
    mutation_prob: float = Field(default=0.5, description="Probability of applying mutation operator.")
    crossover_prob: float = Field(default=0.5, description="Probability of crossover. 0 = mutation-only.")
    memory_penalty: float = Field(
        default=0.5,
        description="λ in Eq. 7 — Jaccard-based tool-memory penalty weight. 0 = no memory.",
    )
    fitness_alpha: float = Field(
        default=0.7,
        description="α in Eq. 7 — top-1 retrieval weight (top-3 weight = 1-α). Paper value: 0.7.",
    )
    jaccard_threshold: float = Field(
        default=0.3,
        description="τ in Eq. 7 — Jaccard similarity threshold for memory penalty. Paper value: 0.3.",
    )
    selection: Literal["fitness", "random", "none"] = Field(
        default="fitness",
        description="Selection strategy: 'fitness' (Memetic/Multi-Turn), 'none' (Scattershot/Single-Pass).",
    )
    top_k_retrieval: int = Field(
        default=5,
        description="Number of tools retrieved per belief probe. Paper value: 5.",
    )
    fitness_method: Literal["baseline_jaccard", "smc", "agm", "dpp"] = Field(
        default="baseline_jaccard",
        description=(
            "Pluggable belief-fitness scorer used by the memetic loop. "
            "'baseline_jaccard' is the production v1 closure (Jaccard + "
            "retrieval-top-k). 'smc' = particle-filter posterior drift, "
            "'agm' = Hellinger Bregman against running centroid, 'dpp' = "
            "log-det diversity gain."
        ),
    )


class BenchmarkSpec(BaseModel):
    """Benchmark and split specification.

    ToolRet splits: code, customized, web, toolbench.
    StableToolBench splits: G1, G2, G3.
    """

    name: Literal["toolret", "stabletoolbench"] = Field(
        ..., description="Benchmark identifier."
    )
    splits: list[str] = Field(
        ...,
        description="List of domain/complexity splits to run.",
    )
    n_queries_per_split: int | None = Field(
        default=None,
        description="Max queries per split. None = all queries (full eval).",
    )


class EvaluatorSpec(BaseModel):
    """Pinned judge and simulator models.

    MUST NOT change when changing the agent model — §4.1, §5.8.
    Both are LLM-based and stochastic; pinning prevents evaluation drift.
    """

    judge_model: str = Field(..., description="Pass-rate judge model (dated tag).")
    judge_revision: str = Field(..., description="Judge model revision/date string.")
    simulator_model: str = Field(..., description="Tool simulator model (dated tag).")
    simulator_revision: str = Field(..., description="Simulator model revision/date string.")


class EmbedderSpec(BaseModel):
    """Pinned retrieval embedder.

    Must be identical across all runs and ablations — §4.1.
    Run YAMLs should reference ``configs/model_pins.yaml`` via ``embedder.pin``;
    the resolver materializes ``name`` before validation.
    """

    name: str = Field(..., description="HuggingFace model name for SentenceTransformer.")
    revision: str = Field(
        default="",
        description="HF commit hash. Empty string = use latest (not recommended for prod runs).",
    )
    device: str = Field(default="cuda:0", description="Torch device string.")


class InfraSpec(BaseModel):
    """Infrastructure and I/O paths.

    Paths use env-var substitution resolved at runtime. Prefer an env-selected
    runtime root and rsync to durable storage at the end.
    """

    cache_dir: str = Field(
        ...,
        description="Per-run response cache directory under the runtime root.",
    )
    output_dir: str = Field(
        ...,
        description="Per-run results directory. Should be on local SSD, rsync'd at end.",
    )
    rsync_to: str | None = Field(
        default=None,
        description="NFS destination for end-of-run rsync. None = no rsync.",
    )
    shard: int = Field(
        default=0,
        description="This shard index (0-indexed). Used for cross-host query splitting.",
    )
    total_shards: int = Field(
        default=1,
        description="Total number of shards. 1 = no sharding.",
    )
    host: str | None = Field(
        default=None,
        description="Informational: hostname. Filled by resolver from $HOSTNAME.",
    )


class BudgetSpec(BaseModel):
    """Cost circuit-breaker per §5.2.

    The cost-logger track's ManifestWriter enforces this by incrementing a
    per-run running total and aborting when exceeded.
    """

    max_cost_usd: float = Field(
        default=50.0,
        description="Max USD spend per run cell. Default: $50 per §5.2.",
    )


class RunConfig(BaseModel):
    """Top-level experiment run configuration.

    One YAML file → one RunConfig → one experiment cell.
    Resolved by ``toolbench.runner.resolver.resolve_config()``.

    Integration stubs (other Wave 1 tracks wire into these):
      - modelclient: ``make_client(self.model.name)``
      - cost-logger:  ``ManifestWriter(run_id=self.run_id, config_hash=self.config_hash())``
    """

    run_id: str | None = Field(
        default=None,
        description="Auto-filled by resolver: '{timestamp}_{git_short}_{hash_prefix}'. "
        "Set explicitly to force a stable ID (e.g. for reruns).",
    )
    model: ModelSpec
    fittext: FitTextHParams
    benchmark: BenchmarkSpec
    evaluator: EvaluatorSpec
    embedder: EmbedderSpec
    infra: InfraSpec
    budget: BudgetSpec
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Freeform annotations (paper notes, run_count hints, etc.). "
        "NOT included in config_hash.",
    )

    def config_hash(self) -> str:
        """SHA-256 of the canonicalized config JSON.

        Secrets (API keys) and non-reproducible fields (run_id, extra,
        infra.host) are excluded so the hash is portable across machines and
        operators. Deterministic across two resolves of the same YAML.

        Returns:
            Hex string of the first 64 chars of the SHA-256 digest.

        Notes:
            Used by the cost-logger track's ManifestWriter to key manifest rows.
            Also used for DFSDT cache key (model, variant, hparams must all change
            the key — §4.4).
        """
        # Fields included in the hash (all except excluded below)
        excluded_top_level = {"run_id", "extra"}
        excluded_infra = {"host", "rsync_to"}

        cfg_dict = self.model_dump(exclude=excluded_top_level)

        # Scrub infra fields that vary by machine/operator
        for key in excluded_infra:
            cfg_dict.get("infra", {}).pop(key, None)

        canonical = json.dumps(cfg_dict, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:64]

    def one_line_summary(self) -> str:
        """Single-line human-readable description for log output.

        Returns:
            String like 'memetic | <resolved-model> | toolret [code,web,...]'
        """
        splits_str = ",".join(self.benchmark.splits)
        return (
            f"{self.fittext.variant} | {self.model.name} | "
            f"{self.benchmark.name} [{splits_str}] | "
            f"N={self.fittext.population_size} G={self.fittext.generations} "
            f"α={self.fittext.fitness_alpha} τ={self.fittext.jaccard_threshold} "
            f"λ={self.fittext.memory_penalty}"
        )
