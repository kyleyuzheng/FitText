"""Status tracking for the dispatcher.

The dispatcher maintains a single ``dispatch_status.json`` file at the root
of the dispatch out_dir. After every cell state transition the file is
atomically rewritten (write-tempfile-then-rename) so a separate monitor
process can read it concurrently without locking.

Schema (top-level)::

    {
      "dispatch_id":  "20260524T130000_a1b2c3",
      "spec_name":    "cheap_sota_small",
      "started":      "2026-05-24T13:00:00.123Z",
      "finished":     "2026-05-24T13:42:11.987Z" | null,
      "n_cells":      40,
      "n_done":       12,
      "n_failed":     1,
      "n_skipped":    4,
      "n_running":    6,
      "n_pending":    17,
      "running_total_cost_usd": 8.42,
      "budget_total_usd":       100.0,
      "cells": {
        "<cell_id>": {  ... CellStatus ... }
      }
    }

The dispatcher process holds a lock over the file when mutating; readers
use the atomic rename guarantee — they always see a complete snapshot.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal

# Cell state values — explicit string literals for json round-trip
CellState = Literal["pending", "running", "done", "failed", "skipped", "budget_skip"]


# ---------------------------------------------------------------------------
# CellStatus dataclass
# ---------------------------------------------------------------------------


@dataclass
class CellStatus:
    """Per-cell status record stored under ``cells[<cell_id>]``."""

    cell_id: str
    kind: str
    label: str
    model: str
    benchmark: str
    split: str
    host: str = "local"
    status: CellState = "pending"
    pid: int | None = None
    cost_usd: float = 0.0
    started_at: str | None = None
    finished_at: str | None = None
    result_dir: str | None = None
    return_code: int | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# StatusTracker
# ---------------------------------------------------------------------------


class StatusTracker:
    """JSON-on-disk dispatch status tracker.

    Thread-safe across in-process callers (``threading.Lock``). For
    cross-process safety the dispatcher uses the atomic-rename pattern so
    readers always observe a fully written file.

    The TUI/monitor (``scripts/dispatch_monitor.py``) only reads — it never
    writes. The dispatcher is the sole writer.
    """

    def __init__(
        self,
        out_dir: Path,
        dispatch_id: str,
        spec_name: str,
        budget_total_usd: float,
    ) -> None:
        """Initialise a fresh tracker file under ``out_dir``.

        Args:
            out_dir: Dispatch out_dir (will be created if missing).
            dispatch_id: Dispatch run id (date + short hash).
            spec_name: ``DispatchSpec.name``.
            budget_total_usd: Global budget for budget-skip accounting.
        """
        self._lock = threading.Lock()
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._path = self._out_dir / "dispatch_status.json"
        self._state: dict[str, Any] = {
            "dispatch_id": dispatch_id,
            "spec_name": spec_name,
            "started": _utc_now_iso(),
            "finished": None,
            "n_cells": 0,
            "n_done": 0,
            "n_failed": 0,
            "n_skipped": 0,
            "n_running": 0,
            "n_pending": 0,
            "n_budget_skip": 0,
            "running_total_cost_usd": 0.0,
            "budget_total_usd": float(budget_total_usd),
            "cells": {},
        }
        self._flush_unlocked()

    @property
    def path(self) -> Path:
        """Filesystem path of the JSON state file."""
        return self._path

    def add_cell(self, cs: CellStatus) -> None:
        """Register a new cell in the pending state."""
        with self._lock:
            self._state["cells"][cs.cell_id] = asdict(cs)
            self._recount_unlocked()
            self._flush_unlocked()

    def mark(self, cell_id: str, **updates: Any) -> None:
        """Update fields on one cell and recompute summary counters.

        Args:
            cell_id: Cell identifier.
            **updates: Keyword updates to apply to the cell record (e.g.
                ``status="running"``, ``pid=12345``, ``cost_usd=0.42``).
        """
        with self._lock:
            cell = self._state["cells"].get(cell_id)
            if cell is None:
                raise KeyError(f"Unknown cell_id {cell_id!r} in dispatch status.")
            cell.update(updates)
            # If cost_usd changed, rebuild running total to keep it consistent.
            if "cost_usd" in updates:
                self._state["running_total_cost_usd"] = sum(
                    float(c.get("cost_usd", 0.0))
                    for c in self._state["cells"].values()
                )
            self._recount_unlocked()
            self._flush_unlocked()

    def finalise(self) -> None:
        """Mark the dispatch as finished and flush."""
        with self._lock:
            self._state["finished"] = _utc_now_iso()
            self._flush_unlocked()

    def snapshot(self) -> dict[str, Any]:
        """Return a copy of the current state (for tests)."""
        with self._lock:
            return json.loads(json.dumps(self._state))

    def running_total_cost(self) -> float:
        """Return the current running-total cost in USD."""
        with self._lock:
            return float(self._state["running_total_cost_usd"])

    # -- internals --------------------------------------------------------

    def _recount_unlocked(self) -> None:
        counts: dict[str, int] = {
            "pending": 0,
            "running": 0,
            "done": 0,
            "failed": 0,
            "skipped": 0,
            "budget_skip": 0,
        }
        for c in self._state["cells"].values():
            counts[c.get("status", "pending")] = counts.get(c.get("status", "pending"), 0) + 1
        self._state["n_cells"] = len(self._state["cells"])
        self._state["n_pending"] = counts["pending"]
        self._state["n_running"] = counts["running"]
        self._state["n_done"] = counts["done"]
        self._state["n_failed"] = counts["failed"]
        self._state["n_skipped"] = counts["skipped"]
        self._state["n_budget_skip"] = counts["budget_skip"]

    def _flush_unlocked(self) -> None:
        """Atomic write: dump to tempfile in the same dir, then rename."""
        # Same dir → guarantees rename is atomic on POSIX.
        fd, tmp = tempfile.mkstemp(
            prefix=".dispatch_status.", suffix=".json.tmp", dir=str(self._out_dir)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self._state, fh, indent=2, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self._path)
        except Exception:
            # Clean up tempfile on any failure
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    """Current UTC time as ISO-8601 string with microseconds and Z suffix."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
