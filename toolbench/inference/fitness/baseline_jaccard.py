"""Baseline-Jaccard fitness function.

Mirrors ``calculate_fitness`` from
``StableToolBench/toolbench/inference/Downstream_tasks/strategies.py``
(post-revert commit 723d752 on 2026-05-24).

Formula:
    score(b) = retrieval_score(b) - jaccard_penalty(b, prior_beliefs)

Where:
    retrieval_score(b) = 0.7 * sim_top1(b) + 0.3 * mean_sim_top3(b)
    jaccard_penalty(b, priors) = 0.5 * max_jaccard if max_jaccard > 0.3 else 0

In the live strategy, ``prior_beliefs`` is the cross-query tool-memory.  In
the offline re-scoring setting, we restrict the comparison to the *current
query's* prior beliefs (in DFSDT appearance order).  This is the natural
adaptation: cross-query memory was a confound we want to control out for
the methodological comparison.
"""

from __future__ import annotations

import re
from typing import List

import numpy as np

from .base import BeliefCandidate, BeliefHistory, FitnessProtocol


_WORD_RE = re.compile(r"[a-zA-Z_]+")


def _tokenize(text: str) -> set:
    """Lowercase + extract alphabetic/underscore tokens (Jaccard atoms)."""
    return set(w.lower() for w in _WORD_RE.findall(text or ""))


def _jaccard(a_tokens: set, b_tokens: set) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    inter = len(a_tokens & b_tokens)
    denominator = len(a_tokens | b_tokens)
    return inter / denominator if denominator else 0.0


class BaselineJaccardFitness:
    """Production-equivalent fitness from strategies.py.

    Attributes:
        name: ``"baseline_jaccard"``.
    """

    name = "baseline_jaccard"

    def __init__(self, penalty_threshold: float = 0.3, penalty_scale: float = 0.5) -> None:
        """Args:
            penalty_threshold: Jaccard threshold below which no penalty
                is applied (the "soft gate" — strategies.py L152).
            penalty_scale: Multiplier on max-Jaccard when over threshold.
        """
        self.penalty_threshold = penalty_threshold
        self.penalty_scale = penalty_scale

    def score(
        self,
        candidate: BeliefCandidate,
        history: BeliefHistory,
    ) -> float:
        # A. Retrieval score from top-1/top-3 cosine sims.
        scores = candidate.retrieval_top_k_scores
        if not scores:
            return 0.0
        top1 = float(scores[0])
        top3_mean = float(sum(scores[:3]) / min(3, len(scores)))
        retrieval_score = 0.7 * top1 + 0.3 * top3_mean

        # B. Jaccard penalty against prior beliefs.
        cand_toks = _tokenize(candidate.belief_text)
        max_sim = 0.0
        for prior in history.prior_beliefs:
            sim = _jaccard(cand_toks, _tokenize(prior.belief_text))
            if sim > max_sim:
                max_sim = sim

        penalty = self.penalty_scale * max_sim if max_sim > self.penalty_threshold else 0.0
        return retrieval_score - penalty
