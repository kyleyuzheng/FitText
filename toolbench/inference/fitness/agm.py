"""AGM fitness with Hellinger Bregman divergence.

Treats each belief's retrieval distribution as a categorical over the
per-query tool catalog.  Fitness rewards retrieval mass (signal) and
penalises Hellinger divergence from the *running average* prior
retrieval distribution.

Formal score:

    score(b) = retrieval_top1(b) - lambda * H^2( p_b , p_avg_prior )

Where:
    H^2(P, Q) = (1/2) * sum_i ( sqrt(p_i) - sqrt(q_i) )^2

Hellinger is a true metric (bounded in [0,1]) and Bregman-projects to
the centroid in (sqrt-)probability space — which gives this fitness a
clean information-geometric interpretation: each new belief is judged by
how much it shifts the centroid of the population's tool distribution.

When ``history`` is empty, H^2 := 0 (no prior to diverge from).

References
----------
Aitchison-style geometric mean (AGM) divergences are the
canonical Bregman family for compositional / probability-simplex data;
see e.g. Egozcue & Pawlowsky-Glahn 2011 for the geometric framework.
We use plain Hellinger here as the simplest, most widely recognised
member of the family.
"""

from __future__ import annotations

import numpy as np

from .base import BeliefCandidate, BeliefHistory, FitnessProtocol


class AGMFitness:
    """Hellinger-Bregman fitness against the running prior centroid.

    Attributes:
        name: ``"agm"``.
    """

    name = "agm"

    def __init__(self, lam: float = 1.0) -> None:
        """Args:
            lam: Weight on the Hellinger penalty.  Higher = more
                conservative (penalises diverging from the running centroid).
        """
        self.lam = lam

    @staticmethod
    def _hellinger_sq(p: np.ndarray, q: np.ndarray) -> float:
        """Squared Hellinger distance between two normalised distributions.

        H^2(P, Q) = (1/2) * sum ( sqrt(p_i) - sqrt(q_i) )^2

        Args:
            p, q: Non-negative vectors of equal length, summing to 1.

        Returns:
            Float in [0, 1].
        """
        if p.shape != q.shape:
            raise ValueError(f"Shape mismatch: {p.shape} vs {q.shape}")
        if p.size == 0:
            return 0.0
        diff = np.sqrt(np.clip(p, 0, None)) - np.sqrt(np.clip(q, 0, None))
        return 0.5 * float(np.sum(diff * diff))

    def score(
        self,
        candidate: BeliefCandidate,
        history: BeliefHistory,
    ) -> float:
        # 1. Signal term: top-1 retrieval probability.  We use the top-1
        #    mass rather than the raw cosine because the distribution is
        #    softmax-normalised, so this is the canonical posterior on the
        #    "true tool is t*" hypothesis.
        ret = candidate.retrieval_distribution
        if ret.size == 0:
            return 0.0
        signal = float(np.max(ret))

        # 2. Hellinger divergence from the running average prior.
        prior = history.prior_retrieval_distributions
        if prior.shape[0] == 0:
            penalty = 0.0
        else:
            avg_prior = np.mean(prior, axis=0)
            # Re-normalise (np.mean of probability vectors still sums to 1
            # in exact arithmetic, but float drift can shift it.)
            total = float(np.sum(avg_prior))
            if total > 0:
                avg_prior = avg_prior / total
            penalty = self._hellinger_sq(ret, avg_prior)

        return signal - self.lam * penalty
