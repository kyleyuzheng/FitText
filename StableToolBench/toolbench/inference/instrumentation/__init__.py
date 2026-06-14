"""
Belief instrumentation for FitText doxastic analyses.

Provides per-generation belief snapshot logging (BeliefTracer) and
post-hoc analysis functions that drive the five doxastic metrics in §2.3.

Public surface:
    BeliefSnapshot  — dataclass for a single belief at a point in evolution
    BeliefTracer    — JSONL writer (thread/process-safe, fcntl-locked)
    SharedEmbedder  — lazy-loaded sentence-transformer singleton
    belief_evidence_alignment   — ρ(b_g*, t*) vs generation
    belief_population_entropy   — Shannon entropy of belief-embedding clusters
    witness_coverage            — fraction of generations that retrieve gold
    belief_revision_rate        — mean intra-pair embedding distance g→g+1
    memory_off_recycling        — intra-generation Jaccard overlap
"""

from toolbench.inference.instrumentation.belief_trace import BeliefSnapshot, BeliefTracer
from toolbench.inference.instrumentation.embedder import SharedEmbedder
from toolbench.inference.instrumentation.analysis import (
    belief_evidence_alignment,
    belief_population_entropy,
    witness_coverage,
    belief_revision_rate,
    memory_off_recycling,
)

__all__ = [
    "BeliefSnapshot",
    "BeliefTracer",
    "SharedEmbedder",
    "belief_evidence_alignment",
    "belief_population_entropy",
    "witness_coverage",
    "belief_revision_rate",
    "memory_off_recycling",
]
