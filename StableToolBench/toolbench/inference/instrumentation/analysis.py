"""
Post-hoc analysis functions for doxastic / belief-dynamics metrics (§2.3).

All functions are pure (no side-effects, no I/O) and depend only on
numpy/scipy so they run without GPU.  The SharedEmbedder is accepted as a
dependency-injected argument so tests can mock it trivially.

Five metrics driven by these functions:
    1. belief_evidence_alignment  — ρ(b_g*, t*) vs generation
    2. belief_population_entropy  — Shannon entropy of embedding clusters
    3. witness_coverage           — % generations where gold tool is retrieved
    4. belief_revision_rate       — mean embedding distance across g→g+1
    5. memory_off_recycling       — intra-generation Jaccard overlap (λ=0 runs)
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from toolbench.inference.instrumentation.belief_trace import BeliefSnapshot
    from toolbench.inference.instrumentation.embedder import SharedEmbedder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D arrays.

    Assumes L2-normalised inputs (dot product == cosine).

    Args:
        a: Vector shape (D,).
        b: Vector shape (D,).

    Returns:
        Scalar cosine similarity in [-1, 1].
    """
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _jaccard(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard index between two sets of strings.

    Args:
        set_a: First set.
        set_b: Second set.

    Returns:
        |A ∩ B| / |A ∪ B|, or 0 if both sets are empty.
    """
    combined = set_a | set_b
    if not combined:
        return 0.0
    return len(set_a & set_b) / len(combined)


def _best_belief_embedding(
    gen_snapshots: list["BeliefSnapshot"],
    embedder: "SharedEmbedder",
) -> np.ndarray:
    """Return the embedding of the highest-fitness snapshot in a generation.

    Computes embeddings on-demand if belief_embedding is None.

    Args:
        gen_snapshots: All snapshots for one generation (non-empty).
        embedder: SharedEmbedder instance.

    Returns:
        1-D float32 array of shape (D,).
    """
    best = max(gen_snapshots, key=lambda s: s.fitness)
    if best.belief_embedding is not None:
        return np.array(best.belief_embedding, dtype=np.float32)
    vecs = embedder.encode([best.belief_text])
    return vecs[0]


def _ensure_embeddings(
    snapshots: list["BeliefSnapshot"],
    embedder: "SharedEmbedder",
) -> np.ndarray:
    """Return a (N, D) embedding matrix for a list of snapshots.

    Uses cached belief_embedding if present, otherwise calls the embedder.

    Args:
        snapshots: List of BeliefSnapshot objects.
        embedder: SharedEmbedder instance.

    Returns:
        np.ndarray of shape (N, D), float32.
    """
    need_encode: list[int] = []
    vecs: list[np.ndarray | None] = [None] * len(snapshots)

    for i, s in enumerate(snapshots):
        if s.belief_embedding is not None:
            vecs[i] = np.array(s.belief_embedding, dtype=np.float32)
        else:
            need_encode.append(i)

    if need_encode:
        texts = [snapshots[i].belief_text for i in need_encode]
        encoded = embedder.encode(texts)
        for j, i in enumerate(need_encode):
            vecs[i] = encoded[j]

    return np.stack(vecs, axis=0).astype(np.float32)  # (N, D)


# ---------------------------------------------------------------------------
# Metric 1: Belief-evidence alignment
# ---------------------------------------------------------------------------

def belief_evidence_alignment(
    beliefs_per_gen: list[list["BeliefSnapshot"]],
    gold_tool_desc: str,
    embedder: "SharedEmbedder",
) -> list[float]:
    """Cosine similarity between the best belief and the gold tool per generation.

    ρ(b_g*, t*) as defined in §2.3 analysis 1.  Expected to be monotone-rising
    over generations for a well-behaved evolutionary run.

    Args:
        beliefs_per_gen: List of length G+1 (generation 0..G).  Each element
            is a list of BeliefSnapshot objects for that generation.
        gold_tool_desc: Natural-language description of the ground-truth tool.
        embedder: SharedEmbedder instance.

    Returns:
        List of float of length len(beliefs_per_gen).  Each value is the cosine
        similarity between the best-fitness belief in that generation and the
        gold tool description.
    """
    if not beliefs_per_gen:
        return []

    gold_vec = embedder.encode([gold_tool_desc])[0]  # (D,)
    results: list[float] = []

    for gen_idx, gen_snapshots in enumerate(beliefs_per_gen):
        if not gen_snapshots:
            logger.warning("belief_evidence_alignment: empty generation %d", gen_idx)
            results.append(float("nan"))
            continue
        best_vec = _best_belief_embedding(gen_snapshots, embedder)
        rho = _cosine(best_vec, gold_vec)
        results.append(rho)

    return results


# ---------------------------------------------------------------------------
# Metric 2: Belief population entropy
# ---------------------------------------------------------------------------

def belief_population_entropy(
    beliefs_per_gen: list[list["BeliefSnapshot"]],
    k_clusters: int = 4,
) -> list[float]:
    """Shannon entropy of k-means cluster assignments per generation.

    Measures population diversity in embedding space.  High entropy = diverse
    beliefs; low entropy = collapsed to a single mode.

    Args:
        beliefs_per_gen: Per-generation snapshot lists.
        k_clusters: Number of clusters for k-means.  Must be ≥ 2.

    Returns:
        List of float of length len(beliefs_per_gen).  NaN for generations
        with fewer snapshots than k_clusters.

    Notes:
        embeddings must already be present on snapshots (belief_embedding ≠ None)
        OR the caller must have pre-populated them.  Unlike other functions this
        one does NOT accept an embedder argument — call
        ``populate_embeddings(beliefs_per_gen, embedder)`` first if needed.
        This keeps the function signature clean for the analysis notebook path.
    """
    from scipy.cluster.vq import kmeans2  # type: ignore

    results: list[float] = []

    for gen_idx, gen_snapshots in enumerate(beliefs_per_gen):
        if not gen_snapshots:
            results.append(float("nan"))
            continue

        vecs = np.array(
            [s.belief_embedding if s.belief_embedding is not None else [0.0]
             for s in gen_snapshots],
            dtype=np.float32,
        )

        # Need at least k_clusters non-degenerate points
        if vecs.shape[0] < k_clusters or vecs.shape[1] <= 1:
            # Degenerate: treat all as one cluster → entropy = 0
            results.append(0.0)
            continue

        try:
            _, labels = kmeans2(vecs.astype(np.float64), k_clusters, minit="points", seed=42)
            counts = np.bincount(labels, minlength=k_clusters)
            probs = counts / counts.sum()
            probs = probs[probs > 0]  # exclude zero-mass clusters
            entropy = -float(np.sum(probs * np.log2(probs)))
        except Exception as exc:
            logger.warning("belief_population_entropy gen %d k-means failed: %s", gen_idx, exc)
            entropy = float("nan")

        results.append(entropy)

    return results


def populate_embeddings(
    beliefs_per_gen: list[list["BeliefSnapshot"]],
    embedder: "SharedEmbedder",
) -> None:
    """Fill in belief_embedding in-place for all snapshots that lack it.

    Args:
        beliefs_per_gen: Per-generation snapshot lists (mutated in-place).
        embedder: SharedEmbedder instance.
    """
    for gen_snapshots in beliefs_per_gen:
        need = [s for s in gen_snapshots if s.belief_embedding is None]
        if not need:
            continue
        vecs = embedder.encode([s.belief_text for s in need])
        for s, vec in zip(need, vecs):
            s.belief_embedding = vec.tolist()


# ---------------------------------------------------------------------------
# Metric 3: Witness coverage
# ---------------------------------------------------------------------------

def witness_coverage(
    beliefs_per_gen: list[list["BeliefSnapshot"]],
    gold_tool_id: str,
) -> float:
    """Fraction of generations where any belief's retrieval set contains the gold tool.

    A generation "witnesses" the gold tool if any individual's retrieved_tool_ids
    list contains gold_tool_id.

    Args:
        beliefs_per_gen: Per-generation snapshot lists.
        gold_tool_id: The ground-truth tool identifier (e.g. "weather/GetCurrent").

    Returns:
        Float in [0, 1].  0.0 if beliefs_per_gen is empty.
    """
    if not beliefs_per_gen:
        return 0.0

    witnessed = 0
    for gen_snapshots in beliefs_per_gen:
        for snap in gen_snapshots:
            if gold_tool_id in snap.retrieved_tool_ids:
                witnessed += 1
                break  # one witness suffices per generation

    return witnessed / len(beliefs_per_gen)


# ---------------------------------------------------------------------------
# Metric 4: Belief revision rate
# ---------------------------------------------------------------------------

def belief_revision_rate(
    beliefs_per_gen: list[list["BeliefSnapshot"]],
    embedder: "SharedEmbedder",
) -> list[float]:
    """Mean intra-pair embedding distance between consecutive generations.

    For each pair (g, g+1): for each survivor in generation g, find its
    offspring in g+1 by matching individual_idx.  Compute pairwise Euclidean
    distance in embedding space.  Mean over matched pairs.

    When individual_idx is not aligned (e.g. crossover reindexes), falls back
    to mean distance between population centroids.

    Args:
        beliefs_per_gen: Per-generation snapshot lists.
        embedder: SharedEmbedder instance.

    Returns:
        List of float of length len(beliefs_per_gen) - 1.  Empty list if fewer
        than 2 generations.
    """
    if len(beliefs_per_gen) < 2:
        return []

    results: list[float] = []

    for g in range(len(beliefs_per_gen) - 1):
        gen_curr = beliefs_per_gen[g]
        gen_next = beliefs_per_gen[g + 1]

        if not gen_curr or not gen_next:
            results.append(float("nan"))
            continue

        vecs_curr = _ensure_embeddings(gen_curr, embedder)  # (N, D)
        vecs_next = _ensure_embeddings(gen_next, embedder)  # (M, D)

        # Attempt index-aligned pairing
        idx_curr = {s.individual_idx: i for i, s in enumerate(gen_curr)}
        idx_next = {s.individual_idx: i for i, s in enumerate(gen_next)}
        shared_ids = set(idx_curr) & set(idx_next)

        if shared_ids:
            dists = [
                float(np.linalg.norm(
                    vecs_curr[idx_curr[k]] - vecs_next[idx_next[k]]
                ))
                for k in shared_ids
            ]
            results.append(float(np.mean(dists)))
        else:
            # Fallback: centroid distance
            centroid_curr = vecs_curr.mean(axis=0)
            centroid_next = vecs_next.mean(axis=0)
            results.append(float(np.linalg.norm(centroid_curr - centroid_next)))

    return results


# ---------------------------------------------------------------------------
# Metric 5: Memory-off recycling (λ=0 run)
# ---------------------------------------------------------------------------

def memory_off_recycling(
    beliefs_per_gen: list[list["BeliefSnapshot"]],
) -> list[float]:
    """Mean intra-generation Jaccard overlap of retrieval sets.

    Measures how much individuals in the same generation retrieve the same tools.
    High overlap → recycling / convergence; low overlap → diverse exploration.
    Designed for λ=0 (memory-off) ablation runs to empirically motivate the
    memory penalty term (Eq.7 §0.3).

    Args:
        beliefs_per_gen: Per-generation snapshot lists.

    Returns:
        List of float of length len(beliefs_per_gen).  Nan for generations with
        fewer than 2 individuals.
    """
    results: list[float] = []

    for gen_idx, gen_snapshots in enumerate(beliefs_per_gen):
        n = len(gen_snapshots)
        if n < 2:
            results.append(float("nan"))
            continue

        # All unique pairs
        pairwise_jaccards: list[float] = []
        for i in range(n):
            set_i = set(gen_snapshots[i].retrieved_tool_ids)
            for j in range(i + 1, n):
                set_j = set(gen_snapshots[j].retrieved_tool_ids)
                pairwise_jaccards.append(_jaccard(set_i, set_j))

        results.append(float(np.mean(pairwise_jaccards)))

    return results
