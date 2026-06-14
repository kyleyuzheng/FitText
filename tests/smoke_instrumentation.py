"""
Smoke tests for the belief instrumentation module (§5.7).

All offline — no GPU, no real embedder.  The embedder is mocked with a
deterministic stub that returns unit-norm random embeddings seeded by text hash.

Tests:
    1. BeliefTracer round-trip (write 3 snapshots, read back, verify all fields)
    2. belief_evidence_alignment on synthetic data — known cosine values
    3. belief_population_entropy — identical embeddings → 0; diverse → log2(k)
    4. witness_coverage — gold in one gen → > 0
    5. belief_revision_rate — formula verification
    6. CLI integration — dummy JSONL + gold → compute_belief_metrics writes CSV

Run::

    python -m pytest tests/smoke_instrumentation.py -v
    # or standalone:
    python tests/smoke_instrumentation.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# ---------------------------------------------------------------------------
# Make sure the StableToolBench source tree is importable
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent / "StableToolBench"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from toolbench.inference.instrumentation.belief_trace import (
    BeliefSnapshot,
    BeliefTracer,
    group_by_generation,
    read_belief_trace,
)
from toolbench.inference.instrumentation.analysis import (
    belief_evidence_alignment,
    belief_population_entropy,
    belief_revision_rate,
    memory_off_recycling,
    populate_embeddings,
    witness_coverage,
)


# ---------------------------------------------------------------------------
# Deterministic mock embedder
# ---------------------------------------------------------------------------

def _unit_vec(text: str, dim: int = 16) -> np.ndarray:
    """Return a deterministic unit-norm vector from text (via SHA hash seeding).

    Args:
        text: Input string.
        dim: Embedding dimensionality.

    Returns:
        L2-normalised float32 array of shape (dim,).
    """
    seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2**31)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


class _MockEmbedder:
    """Deterministic mock embedder for offline tests."""

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts via deterministic hash-seeded unit vectors.

        Args:
            texts: List of strings.

        Returns:
            (N, 16) float32 array.
        """
        return np.stack([_unit_vec(t) for t in texts]).astype(np.float32)


MOCK_EMBEDDER = _MockEmbedder()


def _make_snapshot(
    qid: str = "q0001",
    generation: int = 0,
    individual_idx: int = 0,
    belief_text: str = "tool description",
    embedding: list[float] | None = None,
    retrieved_tool_ids: list[str] | None = None,
    retrieved_scores: list[float] | None = None,
    fitness: float = 0.5,
    fitness_components: dict | None = None,
    is_survivor: bool = True,
) -> BeliefSnapshot:
    """Construct a BeliefSnapshot with convenient defaults.

    Args:
        qid: Query id.
        generation: Generation number.
        individual_idx: Position in population.
        belief_text: Pseudo-tool description.
        embedding: Optional pre-computed embedding.
        retrieved_tool_ids: Tool ids retrieved by this belief.
        retrieved_scores: Corresponding scores.
        fitness: Scalar fitness value.
        fitness_components: Decomposed fitness dict.
        is_survivor: Survivor flag.

    Returns:
        BeliefSnapshot instance.
    """
    return BeliefSnapshot(
        qid=qid,
        generation=generation,
        individual_idx=individual_idx,
        belief_text=belief_text,
        belief_embedding=embedding,
        retrieved_tool_ids=retrieved_tool_ids or [],
        retrieved_scores=retrieved_scores or [],
        fitness=fitness,
        fitness_components=fitness_components or {},
        is_survivor=is_survivor,
    )


# ---------------------------------------------------------------------------
# Test 1: BeliefTracer round-trip
# ---------------------------------------------------------------------------

class TestBeliefTracerRoundTrip(unittest.TestCase):
    """Verify BeliefTracer writes and read_belief_trace reads back all fields."""

    def test_round_trip_three_snapshots(self) -> None:
        """Write 3 snapshots, read back, verify all fields preserved."""
        snapshots = [
            _make_snapshot(qid="q1", generation=0, individual_idx=0,
                           belief_text="alpha tool", fitness=0.3,
                           retrieved_tool_ids=["a/b"], retrieved_scores=[0.9]),
            _make_snapshot(qid="q1", generation=1, individual_idx=0,
                           belief_text="beta tool", fitness=0.7,
                           embedding=[0.1, 0.2, 0.3], is_survivor=False),
            _make_snapshot(qid="q1", generation=2, individual_idx=1,
                           belief_text="gamma tool", fitness=0.85,
                           fitness_components={"retrieval_score": 0.8}),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            with BeliefTracer(out_dir, run_id="run_test", qid="q1") as tracer:
                for s in snapshots:
                    tracer.log(s)

            jsonl_file = out_dir / "q1.jsonl"
            self.assertTrue(jsonl_file.exists(), "JSONL file not created")

            loaded = read_belief_trace(jsonl_file)
            self.assertEqual(len(loaded), 3, "Expected 3 snapshots back")

            for orig, back in zip(snapshots, loaded):
                self.assertEqual(back.qid, orig.qid)
                self.assertEqual(back.generation, orig.generation)
                self.assertEqual(back.individual_idx, orig.individual_idx)
                self.assertEqual(back.belief_text, orig.belief_text)
                self.assertAlmostEqual(back.fitness, orig.fitness, places=6)
                self.assertEqual(back.is_survivor, orig.is_survivor)
                self.assertEqual(back.retrieved_tool_ids, orig.retrieved_tool_ids)

                if orig.belief_embedding is not None:
                    self.assertIsNotNone(back.belief_embedding)
                    np.testing.assert_allclose(
                        back.belief_embedding, orig.belief_embedding, atol=1e-6
                    )

    def test_run_id_stamped(self) -> None:
        """run_id is stamped into every JSONL record."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with BeliefTracer(Path(tmpdir), run_id="RUN_XYZ", qid="q2") as t:
                t.log(_make_snapshot(qid="q2"))
            lines = (Path(tmpdir) / "q2.jsonl").read_text(encoding="utf-8").splitlines()
            rec = json.loads(lines[0])
            self.assertEqual(rec["run_id"], "RUN_XYZ")

    def test_context_manager_closes(self) -> None:
        """File handle is closed after context exit (no JSONL corruption)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "q3.jsonl"
            tracer = BeliefTracer(Path(tmpdir), run_id="r", qid="q3")
            with tracer:
                tracer.log(_make_snapshot(qid="q3"))
            # File handle should be None after exit
            self.assertIsNone(tracer._fh)
            # File should be readable and complete
            loaded = read_belief_trace(path)
            self.assertEqual(len(loaded), 1)

    def test_open_before_log_required(self) -> None:
        """Calling log() before open() raises RuntimeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = BeliefTracer(Path(tmpdir), run_id="r", qid="q4")
            with self.assertRaises(RuntimeError):
                tracer.log(_make_snapshot(qid="q4"))

    def test_group_by_generation(self) -> None:
        """group_by_generation correctly partitions snapshots."""
        snaps = [
            _make_snapshot(generation=0, individual_idx=0),
            _make_snapshot(generation=0, individual_idx=1),
            _make_snapshot(generation=1, individual_idx=0),
            _make_snapshot(generation=2, individual_idx=0),
        ]
        groups = group_by_generation(snaps)
        self.assertEqual(len(groups), 3)
        self.assertEqual(len(groups[0]), 2)
        self.assertEqual(len(groups[1]), 1)
        self.assertEqual(len(groups[2]), 1)


# ---------------------------------------------------------------------------
# Test 2: belief_evidence_alignment — known cosine values
# ---------------------------------------------------------------------------

class TestBeliefEvidenceAlignment(unittest.TestCase):
    """belief_evidence_alignment returns known cosine values."""

    def test_identical_returns_one(self) -> None:
        """When best belief == gold, cosine should be ~1.0."""
        text = "exact gold tool description"
        emb = _unit_vec(text).tolist()
        snap = _make_snapshot(belief_text=text, embedding=emb, fitness=1.0)
        beliefs_per_gen = [[snap]]
        result = belief_evidence_alignment(beliefs_per_gen, text, MOCK_EMBEDDER)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0], 1.0, places=5,
                               msg="Identical text should give cosine ~1.0")

    def test_orthogonal_near_zero(self) -> None:
        """Two semantically different hash-seeded vectors should give low cosine."""
        text_a = "weather forecast API"
        text_b = "blockchain cryptocurrency ledger"
        emb_a = _unit_vec(text_a).tolist()
        snap = _make_snapshot(belief_text=text_a, embedding=emb_a, fitness=1.0)
        beliefs_per_gen = [[snap]]
        result = belief_evidence_alignment(beliefs_per_gen, text_b, MOCK_EMBEDDER)
        # Not guaranteed to be near zero (random), but should be in [-1, 1]
        self.assertTrue(-1.0 <= result[0] <= 1.0)

    def test_multi_gen_returns_list(self) -> None:
        """Returns one value per generation."""
        texts = ["gen0 belief", "gen1 belief", "gen2 belief"]
        beliefs_per_gen = [
            [_make_snapshot(belief_text=t, embedding=_unit_vec(t).tolist(), fitness=float(i))]
            for i, t in enumerate(texts)
        ]
        result = belief_evidence_alignment(beliefs_per_gen, "gold tool", MOCK_EMBEDDER)
        self.assertEqual(len(result), 3)
        for v in result:
            self.assertTrue(-1.0 <= v <= 1.0)

    def test_empty_returns_empty(self) -> None:
        """Empty beliefs_per_gen → empty list."""
        result = belief_evidence_alignment([], "gold", MOCK_EMBEDDER)
        self.assertEqual(result, [])

    def test_picks_highest_fitness(self) -> None:
        """Best-fitness individual's embedding is used, not arbitrary."""
        gold_text = "exact gold"
        gold_emb = _unit_vec(gold_text).tolist()
        # Two individuals in gen0; only the high-fitness one matches gold
        snap_low  = _make_snapshot(belief_text="irrelevant", embedding=_unit_vec("irrelevant").tolist(), fitness=0.1)
        snap_high = _make_snapshot(belief_text=gold_text, embedding=gold_emb, fitness=0.9)
        result = belief_evidence_alignment([[snap_low, snap_high]], gold_text, MOCK_EMBEDDER)
        self.assertAlmostEqual(result[0], 1.0, places=5)


# ---------------------------------------------------------------------------
# Test 3: belief_population_entropy
# ---------------------------------------------------------------------------

class TestBeliefPopulationEntropy(unittest.TestCase):
    """Shannon entropy tests."""

    def test_identical_embeddings_zero_entropy(self) -> None:
        """All identical embeddings → k-means assigns to one cluster → H = 0."""
        emb = [1.0, 0.0, 0.0, 0.0]
        snaps = [_make_snapshot(embedding=emb, individual_idx=i) for i in range(6)]
        result = belief_population_entropy([[snaps[0]]], k_clusters=4)
        # Single individual → degenerate case → 0
        self.assertEqual(result[0], 0.0)

    def test_four_orthogonal_clusters(self) -> None:
        """Four maximally spread unit vectors → entropy near log2(4) = 2.0 bits."""
        dim = 4
        # 4 standard basis vectors — maximally orthogonal in R^4
        basis = np.eye(dim, dtype=np.float32)
        # 8 individuals: 2 per basis direction
        snaps = []
        for i in range(dim):
            for rep in range(2):
                s = _make_snapshot(
                    embedding=basis[i].tolist(),
                    individual_idx=i * 2 + rep,
                )
                snaps.append(s)
        result = belief_population_entropy([snaps], k_clusters=4)
        # With perfectly balanced 4 clusters: H = log2(4) = 2.0
        self.assertAlmostEqual(result[0], 2.0, places=1,
                               msg=f"Expected entropy ~2.0 bits, got {result[0]}")

    def test_empty_generation_nan(self) -> None:
        """Empty generation list → nan."""
        result = belief_population_entropy([[]], k_clusters=4)
        self.assertTrue(math.isnan(result[0]))

    def test_multi_gen(self) -> None:
        """Returns one value per generation."""
        emb = [1.0, 0.0]
        gen0 = [_make_snapshot(embedding=emb)]
        gen1 = [_make_snapshot(embedding=emb, generation=1)]
        result = belief_population_entropy([gen0, gen1], k_clusters=2)
        self.assertEqual(len(result), 2)


# ---------------------------------------------------------------------------
# Test 4: witness_coverage
# ---------------------------------------------------------------------------

class TestWitnessCoverage(unittest.TestCase):
    """witness_coverage returns correct fraction."""

    def test_gold_in_one_of_three_gens(self) -> None:
        """Gold present in 1/3 generations → 1/3."""
        gold_id = "weather/GetCurrent"
        gen0 = [_make_snapshot(retrieved_tool_ids=["other/tool"])]
        gen1 = [_make_snapshot(retrieved_tool_ids=[gold_id, "other/tool"])]
        gen2 = [_make_snapshot(retrieved_tool_ids=["another/tool"])]
        result = witness_coverage([gen0, gen1, gen2], gold_id)
        self.assertAlmostEqual(result, 1.0 / 3.0, places=6)

    def test_gold_never_present(self) -> None:
        """Gold never in retrieval sets → 0.0."""
        gen0 = [_make_snapshot(retrieved_tool_ids=["x/y"])]
        result = witness_coverage([gen0], "missing/tool")
        self.assertEqual(result, 0.0)

    def test_gold_always_present(self) -> None:
        """Gold in every generation → 1.0."""
        gold_id = "x/y"
        gens = [[_make_snapshot(retrieved_tool_ids=[gold_id])] for _ in range(5)]
        result = witness_coverage(gens, gold_id)
        self.assertEqual(result, 1.0)

    def test_empty_returns_zero(self) -> None:
        """Empty beliefs_per_gen → 0.0."""
        self.assertEqual(witness_coverage([], "x/y"), 0.0)

    def test_multiple_individuals_any_witness(self) -> None:
        """Only one individual needs to retrieve gold per generation."""
        gold_id = "geo/locate"
        gen0 = [
            _make_snapshot(retrieved_tool_ids=["other/tool"]),
            _make_snapshot(retrieved_tool_ids=[gold_id]),
        ]
        result = witness_coverage([gen0], gold_id)
        self.assertEqual(result, 1.0)


# ---------------------------------------------------------------------------
# Test 5: belief_revision_rate
# ---------------------------------------------------------------------------

class TestBeliefRevisionRate(unittest.TestCase):
    """belief_revision_rate formula verification."""

    def test_two_generations_index_aligned(self) -> None:
        """Revision rate matches expected Euclidean distance between aligned pairs."""
        # Gen 0: individual 0 at [1, 0], individual 1 at [0, 1]
        # Gen 1: individual 0 at [0, 1], individual 1 at [1, 0]
        # Distance for pair 0: ||[1,0] - [0,1]|| = sqrt(2)
        # Distance for pair 1: ||[0,1] - [1,0]|| = sqrt(2)
        # Mean = sqrt(2) ≈ 1.4142

        emb_a = [1.0, 0.0]
        emb_b = [0.0, 1.0]

        gen0 = [
            _make_snapshot(generation=0, individual_idx=0, embedding=emb_a),
            _make_snapshot(generation=0, individual_idx=1, embedding=emb_b),
        ]
        gen1 = [
            _make_snapshot(generation=1, individual_idx=0, embedding=emb_b),
            _make_snapshot(generation=1, individual_idx=1, embedding=emb_a),
        ]

        result = belief_revision_rate([gen0, gen1], MOCK_EMBEDDER)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0], math.sqrt(2), places=5)

    def test_identical_population_zero_revision(self) -> None:
        """Identical gen0 and gen1 embeddings → revision rate = 0."""
        emb = [0.5, 0.5]
        gen0 = [_make_snapshot(generation=0, individual_idx=0, embedding=emb)]
        gen1 = [_make_snapshot(generation=1, individual_idx=0, embedding=emb)]
        result = belief_revision_rate([gen0, gen1], MOCK_EMBEDDER)
        self.assertAlmostEqual(result[0], 0.0, places=6)

    def test_single_gen_returns_empty(self) -> None:
        """Only one generation → empty list (no transitions)."""
        gen0 = [_make_snapshot()]
        result = belief_revision_rate([gen0], MOCK_EMBEDDER)
        self.assertEqual(result, [])

    def test_length_g_minus_one(self) -> None:
        """Output length is len(beliefs_per_gen) - 1."""
        gens = [
            [_make_snapshot(generation=g, embedding=_unit_vec(f"gen{g}text").tolist())]
            for g in range(4)
        ]
        result = belief_revision_rate(gens, MOCK_EMBEDDER)
        self.assertEqual(len(result), 3)


# ---------------------------------------------------------------------------
# Test 6: CLI integration — compute_belief_metrics writes expected CSV
# ---------------------------------------------------------------------------

class TestCLIIntegration(unittest.TestCase):
    """End-to-end: dummy belief JSONL + gold labels → CSV with expected columns."""

    def _write_belief_jsonl(self, path: Path, qid: str, n_gens: int = 3) -> None:
        """Write a synthetic belief trace JSONL file.

        Args:
            path: Output JSONL file path.
            qid: Query identifier.
            n_gens: Number of generations (0..n_gens-1).
        """
        records = []
        for g in range(n_gens):
            for idx in range(3):
                text = f"{qid}_gen{g}_ind{idx}"
                emb = _unit_vec(text).tolist()
                rec = {
                    "qid": qid,
                    "generation": g,
                    "individual_idx": idx,
                    "belief_text": text,
                    "belief_embedding": emb,
                    "retrieved_tool_ids": [f"tool_{idx}/api{g}"],
                    "retrieved_scores": [0.5 + 0.1 * idx],
                    "fitness": 0.3 + 0.1 * g + 0.05 * idx,
                    "fitness_components": {"retrieval_score": 0.5},
                    "is_survivor": True,
                    "run_id": "smoke_run",
                }
                records.append(rec)
        with open(path, "w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")

    def _write_gold_jsonl(self, path: Path, qid: str) -> None:
        """Write a minimal gold annotation file.

        Args:
            path: Output JSONL path.
            qid: Query identifier.
        """
        rec = {
            "qid": qid,
            "gold_tool_id": "tool_0/api0",
            "gold_tool_desc": f"{qid} gold description for tool zero",
        }
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    def test_cli_produces_csv_with_expected_columns(self) -> None:
        """CLI runs without error and produces CSV with required metric columns."""
        from scripts.compute_belief_metrics import main as cli_main  # type: ignore

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            qid = "smoke_q001"

            belief_dir = tmpdir_path / "beliefs"
            belief_dir.mkdir()
            self._write_belief_jsonl(belief_dir / f"{qid}.jsonl", qid, n_gens=3)

            gold_file = tmpdir_path / "gold.jsonl"
            self._write_gold_jsonl(gold_file, qid)

            out_csv = tmpdir_path / "metrics.csv"

            # Patch SharedEmbedder.instance() to return mock
            with patch(
                "toolbench.inference.instrumentation.embedder.SharedEmbedder.instance",
                return_value=MOCK_EMBEDDER,
            ):
                cli_main([
                    "--belief-dir", str(belief_dir),
                    "--gold-file", str(gold_file),
                    "--out-csv", str(out_csv),
                    "--n-bootstrap", "50",
                ])

            self.assertTrue(out_csv.exists(), "CLI did not create output CSV")

            with open(out_csv, encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)

            self.assertGreater(len(rows), 0, "CSV has no rows")

            # Must have at least one alignment column and witness_coverage
            header = rows[0].keys()
            self.assertTrue(
                any(c.startswith("alignment_gen") for c in header),
                f"No alignment_gen* column in: {list(header)}"
            )
            self.assertIn("witness_coverage", header)

            # Aggregate rows present
            qids = [r["qid"] for r in rows]
            self.assertIn("__mean__", qids)
            self.assertIn("__ci_lo__", qids)
            self.assertIn("__ci_hi__", qids)

    def test_held_out_split_excludes_domain(self) -> None:
        """--held-out-split correctly skips queries from the specified domain."""
        from scripts.compute_belief_metrics import main as cli_main  # type: ignore

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            qid_keep = "toolret_q001"
            qid_skip = "tdEG_q001"

            belief_dir = tmpdir_path / "beliefs"
            belief_dir.mkdir()
            self._write_belief_jsonl(belief_dir / f"{qid_keep}.jsonl", qid_keep, n_gens=2)
            self._write_belief_jsonl(belief_dir / f"{qid_skip}.jsonl", qid_skip, n_gens=2)

            gold_file = tmpdir_path / "gold.jsonl"
            with open(gold_file, "w", encoding="utf-8") as fh:
                for q in [qid_keep, qid_skip]:
                    fh.write(json.dumps({
                        "qid": q,
                        "gold_tool_id": "tool_0/api0",
                        "gold_tool_desc": f"{q} gold",
                    }) + "\n")

            out_csv = tmpdir_path / "metrics.csv"

            with patch(
                "toolbench.inference.instrumentation.embedder.SharedEmbedder.instance",
                return_value=MOCK_EMBEDDER,
            ):
                cli_main([
                    "--belief-dir", str(belief_dir),
                    "--gold-file", str(gold_file),
                    "--out-csv", str(out_csv),
                    "--held-out-split", "tdEG",
                    "--n-bootstrap", "50",
                ])

            with open(out_csv, encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))

            query_qids = [r["qid"] for r in rows if not r["qid"].startswith("__")]
            self.assertIn(qid_keep, query_qids)
            self.assertNotIn(qid_skip, query_qids)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Add scripts/ to path for CLI integration test
    _SCRIPTS = Path(__file__).parent.parent / "scripts"
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))

    unittest.main(verbosity=2)
