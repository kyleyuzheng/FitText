"""Fitness protocol and shared dataclasses for offline belief re-scoring.

The fitness functions defined under this package re-score the *belief
populations* persisted in StableToolBench prediction trees, under four
different theoretical frames:

    - Baseline Jaccard  (mirrors strategies.py:calculate_fitness, 2026-05-24)
    - SMC particle filter (likelihood × posterior-drift)
    - AGM + Bregman (Hellinger divergence against running prior)
    - DPP (log-determinant marginal gain — diversity bonus)

A *belief* in this codebase is the text of a pseudo-tool description
emitted by the upstream FitText reformulation step (memetic / scatter /
DBD), captured in the assistant's ``<|begin_func_description|>...<|end_func_description|>``
blocks within the DFSDT prediction tree.

All four scorers consume the same ``BeliefCandidate`` records and emit a
scalar score per (qid, candidate_idx).  Higher = more fit.

Why only one base file (not a class hierarchy)
-----------------------------------------------
The four scorers share no state; they're pure functions parametrised by
``BeliefHistory``.  A ``Protocol`` is cheaper than ABC + boilerplate.

References
----------
- Strategies code:
  ``StableToolBench/toolbench/inference/Downstream_tasks/strategies.py``
  function ``run_memetic_strategy`` lines 134-157 (the Jaccard baseline).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Protocol

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BeliefCandidate:
    """A single belief extracted from a prediction tree.

    Args:
        qid: Query identifier (matches the key in the predictions JSON).
        belief_idx: 0-indexed position within this query's belief population.
            (The order is the DFSDT-tree appearance order, which approximates
            generation order under the memetic strategy.)
        belief_text: The raw pseudo-tool description text.
        belief_embedding: L2-normalised float32 SimCSE embedding, shape (D,).
        retrieval_distribution: Length-V probability vector over the per-query
            tool catalog (V = len(available_tools) - {Finish}). Constructed
            by softmaxing belief-tool cosine similarities; sums to 1.
        retrieval_top_k_ids: Tool-name strings sorted by descending sim
            (the top-K we would call if this belief were the "winner").
        retrieval_top_k_scores: Cosine scores, aligned with above.
    """

    qid: str
    belief_idx: int
    belief_text: str
    belief_embedding: np.ndarray
    retrieval_distribution: np.ndarray  # shape (V,), sums to 1
    retrieval_top_k_ids: List[str]
    retrieval_top_k_scores: List[float]


@dataclass
class BeliefHistory:
    """Running per-query state consumed by stateful fitness scorers.

    The history is built incrementally as we walk through a query's belief
    population in DFSDT appearance order.  Stateless scorers (e.g. raw
    retrieval-top1 baseline) may ignore it; stateful scorers (SMC, AGM, DPP)
    use it to track the "prior" against which a candidate is compared.

    Args:
        prior_beliefs: All belief candidates seen so far in this query.
            Excludes the candidate currently being scored.
        prior_embeddings: ``(N_prior, D)`` matrix of L2-normalised embeddings.
            Empty array with shape (0, D) when no priors yet.
        prior_retrieval_distributions: ``(N_prior, V)`` matrix; each row sums
            to 1.  Empty (0, V) when no priors yet.
    """

    prior_beliefs: List[BeliefCandidate] = field(default_factory=list)
    prior_embeddings: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float32)
    )
    prior_retrieval_distributions: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float32)
    )

    def add(self, b: BeliefCandidate) -> None:
        """Append a candidate to the history (mutating)."""
        self.prior_beliefs.append(b)
        if self.prior_embeddings.size == 0:
            self.prior_embeddings = b.belief_embedding.reshape(1, -1).astype(np.float32)
        else:
            self.prior_embeddings = np.vstack(
                [self.prior_embeddings, b.belief_embedding.reshape(1, -1).astype(np.float32)]
            )
        if self.prior_retrieval_distributions.size == 0:
            self.prior_retrieval_distributions = b.retrieval_distribution.reshape(1, -1).astype(np.float32)
        else:
            self.prior_retrieval_distributions = np.vstack(
                [
                    self.prior_retrieval_distributions,
                    b.retrieval_distribution.reshape(1, -1).astype(np.float32),
                ]
            )


class FitnessProtocol(Protocol):
    """A fitness scorer scores a single belief against a running history.

    Implementations MUST be pure functions of (candidate, history) -- no
    hidden global state.  This guarantees reproducibility under shuffle.
    """

    name: str

    def score(
        self,
        candidate: BeliefCandidate,
        history: BeliefHistory,
    ) -> float:
        """Return a scalar fitness for ``candidate`` given ``history``."""
        ...


def safe_softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically-stable softmax over the last axis.

    Args:
        x: Input vector or array (last axis is softmax axis).
        temperature: Divisor applied to logits before softmax. >1 = flatter,
            <1 = peakier. Default 1.0 matches a vanilla softmax.

    Returns:
        Same-shape array whose values along the last axis sum to 1.
    """
    if x.size == 0:
        return x
    x = x / max(temperature, 1e-12)
    x_max = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - x_max)
    s = np.sum(e, axis=-1, keepdims=True)
    s = np.where(s == 0, 1.0, s)
    return e / s
