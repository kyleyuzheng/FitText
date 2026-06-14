"""Determinantal-Point-Process fitness.

DPP rewards beliefs that add *volume* to the kernel matrix spanned by
the prior beliefs — i.e., a candidate is fit if including it produces a
more diverse population than excluding it.

Formal score:

    score(b) = log det(K_pop ∪ {b}) - log det(K_pop)

where K is the RBF Gram matrix of belief embeddings:

    K[i,j] = exp(- ||e_i - e_j||^2 / (2 sigma^2))

We add ``epsilon * I`` for numerical stability (Cholesky requires PD).

When ``history`` is empty, the determinant of an empty matrix is 1 by
convention, so the score reduces to ``log(K[b,b] + eps) = log(1 + eps)
≈ 0``.  That is the natural DPP bootstrap.

References
----------
Kulesza & Taskar 2012 "Determinantal Point Processes for Machine
Learning" (Foundations & Trends in ML).  We use the standard L-ensemble
formulation with quality = 1 (so the kernel IS the similarity).  This
gives a pure diversity-reward; adding a quality term would re-couple it
to retrieval signal, which the SMC and AGM scorers already provide.
"""

from __future__ import annotations

import numpy as np

from .base import BeliefCandidate, BeliefHistory, FitnessProtocol


class DPPFitness:
    """Marginal log-det gain from adding candidate to the prior population.

    Attributes:
        name: ``"dpp"``.
    """

    name = "dpp"

    def __init__(self, sigma: float = 0.5, epsilon: float = 1e-4) -> None:
        """Args:
            sigma: RBF kernel bandwidth (same family / convention as SMCFitness).
            epsilon: Diagonal jitter added to the kernel matrix before
                Cholesky.  Required for numerical PD when belief embeddings
                are near-duplicates.
        """
        self.sigma = sigma
        self.epsilon = epsilon

    def _gram(self, particles: np.ndarray) -> np.ndarray:
        """RBF Gram matrix with diagonal jitter.

        Args:
            particles: ``(N, D)`` matrix.

        Returns:
            ``(N, N)`` symmetric PD matrix (with ``epsilon`` on diagonal).
        """
        n = particles.shape[0]
        if n == 0:
            return np.empty((0, 0), dtype=np.float64)
        diffs = particles[:, np.newaxis, :] - particles[np.newaxis, :, :]
        sq = np.sum(diffs * diffs, axis=2)
        K = np.exp(-sq / (2.0 * self.sigma * self.sigma)).astype(np.float64)
        K += self.epsilon * np.eye(n)
        return K

    @staticmethod
    def _logdet_cholesky(K: np.ndarray) -> float:
        """Log-determinant via Cholesky for stability.

        Args:
            K: PD matrix.

        Returns:
            log det(K) as a Python float.
        """
        if K.shape[0] == 0:
            return 0.0  # det of empty matrix = 1 by convention
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            # Fall back to eigenvalue product (rare)
            sign, logabsdet = np.linalg.slogdet(K)
            return float(logabsdet) if sign > 0 else float("-inf")
        return 2.0 * float(np.sum(np.log(np.diag(L))))

    def score(
        self,
        candidate: BeliefCandidate,
        history: BeliefHistory,
    ) -> float:
        prior = history.prior_embeddings
        if prior.shape[0] == 0:
            # Single-particle case: K = [[1 + eps]], so log det ≈ 0.
            return float(np.log(1.0 + self.epsilon))

        log_det_before = self._logdet_cholesky(self._gram(prior))
        all_particles = np.vstack([prior, candidate.belief_embedding.reshape(1, -1)])
        log_det_after = self._logdet_cholesky(self._gram(all_particles))
        return log_det_after - log_det_before
