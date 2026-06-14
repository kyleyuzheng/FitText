"""
BeliefSnapshot dataclass and BeliefTracer JSONL writer.

One JSONL file is written per query (qid).  The writer uses the same
fcntl-based locking pattern as ManifestWriter so concurrent workers
can append safely on POSIX filesystems (NFS).

Usage::

    tracer = BeliefTracer(out_dir=Path("beliefs"), run_id="run_abc", qid="q0042")
    tracer.log(BeliefSnapshot(qid="q0042", generation=0, individual_idx=0, ...))
    tracer.close()

    # or as context manager:
    with BeliefTracer(Path("beliefs"), "run_abc", "q0042") as t:
        t.log(snapshot)
"""

from __future__ import annotations

import dataclasses
import fcntl
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BeliefSnapshot:
    """A single belief state captured at one point in the evolutionary search.

    Fields mirror §5.7 of the execution plan.

    Args:
        qid: Query identifier (matches ToolRet / StableToolBench qid).
        generation: 0 = initial population (post generate_population_from_ancestor),
            1..G = after evolution round g.
        individual_idx: Position within the population (0-indexed).
        belief_text: The pseudo-tool description text (the "belief" in doxastic terms).
        belief_embedding: L2-normalised embedding vector, or None if not yet encoded.
            Populated lazily by BeliefTracer when a SharedEmbedder is provided.
        retrieved_tool_ids: Ordered list of tool identifiers from top-k retrieval
            (e.g. "weather/GetCurrentWeather").
        retrieved_scores: Corresponding retrieval similarity scores.
        fitness: Scalar fitness f(b) as computed by the active scorer.
        fitness_components: Decomposed fitness; expected keys:
            "alpha_top1", "alpha_top3", "retrieval_score", "judge_score", "memory_penalty".
        is_survivor: True if this individual was selected into the next generation.
    """

    qid: str
    generation: int
    individual_idx: int
    belief_text: str
    belief_embedding: list[float] | None
    retrieved_tool_ids: list[str]
    retrieved_scores: list[float]
    fitness: float
    fitness_components: dict[str, float]
    is_survivor: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict.

        Returns:
            Dict with all fields; belief_embedding serialised as list[float] or null.
        """
        d = dataclasses.asdict(self)
        # Ensure embedding is list[float] or None — no numpy arrays in JSONL
        if d["belief_embedding"] is not None:
            d["belief_embedding"] = [float(x) for x in d["belief_embedding"]]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BeliefSnapshot":
        """Deserialise from a dict (e.g. loaded from JSONL).

        Args:
            d: Dict previously produced by ``to_dict()``.

        Returns:
            Reconstructed BeliefSnapshot.
        """
        return cls(
            qid=d["qid"],
            generation=int(d["generation"]),
            individual_idx=int(d["individual_idx"]),
            belief_text=d["belief_text"],
            belief_embedding=d.get("belief_embedding"),
            retrieved_tool_ids=d.get("retrieved_tool_ids", []),
            retrieved_scores=[float(s) for s in d.get("retrieved_scores", [])],
            fitness=float(d["fitness"]),
            fitness_components=d.get("fitness_components", {}),
            is_survivor=bool(d["is_survivor"]),
        )


class BeliefTracer:
    """Thread- and process-safe JSONL writer for BeliefSnapshot records.

    One instance per (qid).  Each ``log()`` call atomically appends one JSON
    line.  File locking uses ``fcntl.LOCK_EX`` (POSIX / NFS-safe).
    A per-instance ``threading.Lock`` serialises writes within a process before
    acquiring the file lock, reducing kernel contention in multi-threaded workers.

    Output path: ``<out_dir>/<qid>.jsonl``

    Args:
        out_dir: Directory where per-qid JSONL files are written.
        run_id: Run identifier stamped into every record.
        qid: Query identifier.  Used as the filename stem.

    Example::

        with BeliefTracer(Path("beliefs"), "run_abc", "toolret:q0042") as t:
            t.log(snapshot)
    """

    def __init__(self, out_dir: Path, run_id: str, qid: str) -> None:
        self._out_dir = Path(out_dir)
        self._run_id = run_id
        self._qid = qid
        # Replace characters that are problematic in filenames
        safe_qid = qid.replace("/", "_").replace(":", "_")
        self._path = self._out_dir / f"{safe_qid}.jsonl"
        self._fh: Any = None
        self._lock = threading.Lock()
        self._count = 0

    # -- Context manager ---------------------------------------------------

    def __enter__(self) -> "BeliefTracer":
        self.open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # -- File lifecycle ----------------------------------------------------

    def open(self) -> None:
        """Open the JSONL file for appending (creates parents if needed).

        Safe to call multiple times — subsequent calls are no-ops.
        """
        if self._fh is not None:
            return
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._fh = open(self._path, "a", buffering=1, encoding="utf-8")
        logger.debug("BeliefTracer opened %s (run_id=%s)", self._path, self._run_id)

    def close(self) -> None:
        """Flush, fsync, and close the JSONL file."""
        if self._fh is None:
            return
        try:
            self._fh.flush()
            os.fsync(self._fh.fileno())
        finally:
            self._fh.close()
            self._fh = None
        logger.debug(
            "BeliefTracer closed %s — wrote %d snapshots", self._path, self._count
        )

    # -- Write -------------------------------------------------------------

    def log(self, snapshot: BeliefSnapshot) -> None:
        """Atomically append one BeliefSnapshot to the JSONL file.

        The snapshot is serialised via ``BeliefSnapshot.to_dict()``.  A
        ``"run_id"`` key is injected so log files are self-describing.

        Args:
            snapshot: Populated BeliefSnapshot to persist.

        Raises:
            RuntimeError: If called before ``open()`` / outside context manager.
        """
        if self._fh is None:
            raise RuntimeError(
                "BeliefTracer.log() called before open(). "
                "Use 'with BeliefTracer(...) as t:' or call t.open() first."
            )
        record = snapshot.to_dict()
        record["run_id"] = self._run_id
        line = json.dumps(record, ensure_ascii=False)

        with self._lock:
            try:
                fcntl.flock(self._fh, fcntl.LOCK_EX)
                self._fh.write(line + "\n")
                self._fh.flush()
            finally:
                fcntl.flock(self._fh, fcntl.LOCK_UN)
        self._count += 1


def read_belief_trace(path: Path) -> list[BeliefSnapshot]:
    """Read all BeliefSnapshot records from a JSONL file.

    Args:
        path: Path to a per-qid JSONL file written by BeliefTracer.

    Returns:
        List of BeliefSnapshot objects in file order.
    """
    snapshots: list[BeliefSnapshot] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                snapshots.append(BeliefSnapshot.from_dict(d))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning("Skipping malformed line %d in %s: %s", lineno, path, exc)
    return snapshots


def group_by_generation(
    snapshots: list[BeliefSnapshot],
) -> list[list[BeliefSnapshot]]:
    """Group a flat list of snapshots into per-generation lists.

    Args:
        snapshots: All snapshots for one qid, in any order.

    Returns:
        List indexed by generation (0-based).  generations_list[g] contains
        all snapshots where snapshot.generation == g.  Empty generations produce
        empty inner lists.
    """
    if not snapshots:
        return []
    max_gen = max(s.generation for s in snapshots)
    result: list[list[BeliefSnapshot]] = [[] for _ in range(max_gen + 1)]
    for s in snapshots:
        result[s.generation].append(s)
    return result
