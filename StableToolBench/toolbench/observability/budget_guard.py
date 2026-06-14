"""
File-locked atomic cost counter and circuit breaker.

Workers poll ``BudgetGuard.check()`` at each checkpoint.  When the cumulative
cost exceeds ``max_cost_usd``, the guard touches ``$run_dir/ABORTED_BUDGET``
and raises ``BudgetExceeded``.  Workers detect the sentinel file and exit
cleanly without needing inter-process signalling.

Usage::

    guard = BudgetGuard(run_dir=Path("runs/my_run"), max_cost_usd=200.0)

    # After each API call:
    guard.record(entry.cost_usd)  # atomic increment
    guard.check()                  # raises BudgetExceeded if over limit

    # In a different worker process:
    if guard.aborted:
        logger.info("Budget exceeded — exiting cleanly.")
        sys.exit(0)
"""

from __future__ import annotations

import fcntl
import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

SENTINEL_FILENAME = "ABORTED_BUDGET"
COUNTER_FILENAME = "running_total.txt"


class BudgetExceeded(Exception):
    """Raised when the cumulative run cost exceeds the configured maximum."""

    def __init__(self, total: float, limit: float) -> None:
        super().__init__(
            f"Budget exceeded: ${total:.4f} spent, limit was ${limit:.4f}"
        )
        self.total = total
        self.limit = limit


class BudgetGuard:
    """Atomic cost counter with circuit-breaker behaviour.

    The running total is stored in ``<run_dir>/running_total.txt`` as a plain
    float string.  Both within-process and cross-process writes are serialised
    by a combination of a Python ``threading.Lock`` and ``fcntl.LOCK_EX``.

    Args:
        run_dir: Directory for this run's state files.  Created if absent.
        max_cost_usd: Cost cap in USD.  ``record()`` + ``check()`` raises
            ``BudgetExceeded`` when this is exceeded.
    """

    def __init__(self, run_dir: Path, max_cost_usd: float) -> None:
        self._run_dir = Path(run_dir)
        self._max_cost = max_cost_usd
        self._lock = threading.Lock()
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._counter_path = self._run_dir / COUNTER_FILENAME
        self._sentinel_path = self._run_dir / SENTINEL_FILENAME

        # Initialise counter file if not present
        if not self._counter_path.exists():
            self._counter_path.write_text("0.0\n", encoding="utf-8")
        logger.debug(
            "BudgetGuard initialised: run_dir=%s max_cost=%.2f USD",
            self._run_dir,
            self._max_cost,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, cost: float) -> None:
        """Atomically add *cost* to the running total.

        Args:
            cost: USD cost of the last API call (non-negative).

        Note:
            This method does **not** raise ``BudgetExceeded``; call
            ``check()`` separately to trigger the circuit breaker.
        """
        if cost < 0:
            raise ValueError(f"cost must be non-negative, got {cost!r}")

        with self._lock:
            with open(self._counter_path, "r+", encoding="utf-8") as fh:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                try:
                    current = float(fh.read().strip() or "0.0")
                    new_total = current + cost
                    fh.seek(0)
                    fh.write(f"{new_total:.10f}\n")
                    fh.truncate()
                    fh.flush()
                    os.fsync(fh.fileno())
                finally:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    def check(self) -> None:
        """Raise ``BudgetExceeded`` if the running total exceeds the cap.

        Also touches the ``ABORTED_BUDGET`` sentinel file so other workers
        can detect the abort without polling the counter.

        Raises:
            BudgetExceeded: When the accumulated cost exceeds ``max_cost_usd``.
        """
        total = self.total
        if total > self._max_cost:
            self._touch_sentinel()
            raise BudgetExceeded(total=total, limit=self._max_cost)

    @property
    def total(self) -> float:
        """Current accumulated cost in USD (reads from disk)."""
        try:
            return float(self._counter_path.read_text(encoding="utf-8").strip() or "0.0")
        except (FileNotFoundError, ValueError):
            return 0.0

    @property
    def aborted(self) -> bool:
        """``True`` if the ``ABORTED_BUDGET`` sentinel file is present.

        Workers in a separate process should poll this property at each
        checkpoint and exit cleanly when it becomes ``True``.
        """
        return self._sentinel_path.exists()

    @property
    def remaining(self) -> float:
        """Remaining budget in USD (may be negative if exceeded)."""
        return self._max_cost - self.total

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _touch_sentinel(self) -> None:
        """Create the ABORTED_BUDGET sentinel file if not already present."""
        if not self._sentinel_path.exists():
            self._sentinel_path.touch()
            logger.warning(
                "Budget cap $%.2f exceeded (total $%.4f). "
                "Sentinel written: %s",
                self._max_cost,
                self.total,
                self._sentinel_path,
            )

    def reset(self) -> None:
        """Reset the counter and remove the sentinel.  Use for testing only."""
        with self._lock:
            self._counter_path.write_text("0.0\n", encoding="utf-8")
            if self._sentinel_path.exists():
                self._sentinel_path.unlink()
