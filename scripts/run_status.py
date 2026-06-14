#!/usr/bin/env python3
"""
run_status.py — live dashboard for an in-progress run.

Tails the active manifest JSONL file and prints a rolling dashboard:
  - Cumulative cost, tokens, cache hit rate
  - Rolling RPM (last 60 s)
  - Per-model cost breakdown
  - ETA based on burn rate vs budget cap
  - Estimated query completion fraction

Usage::

    python scripts/run_status.py <manifest_path> [--budget <max_usd>] [--interval <seconds>]

Refresh every 2 s by default.  Ctrl-C to exit.

Dependencies: stdlib only + ``rich`` if available (degrades gracefully to plain text).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Rich import (optional)
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.text import Text
    _RICH = True
    _console = Console()
except ImportError:
    _RICH = False
    _console = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Manifest reading (incremental tail)
# ---------------------------------------------------------------------------

def _tail_manifest(path: Path, since_byte: int) -> tuple[list[dict[str, Any]], int]:
    """Read new JSONL lines from *path* starting at byte offset *since_byte*.

    Returns:
        Tuple of (list of parsed dicts, new byte offset).
    """
    entries: list[dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            fh.seek(since_byte)
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
            new_offset = fh.tell()
    except FileNotFoundError:
        new_offset = since_byte
    return entries, new_offset


# ---------------------------------------------------------------------------
# State accumulator
# ---------------------------------------------------------------------------

class _State:
    """Accumulates metrics from manifest entries."""

    def __init__(self) -> None:
        self.total_cost: float = 0.0
        self.total_input: int = 0
        self.total_cached: int = 0
        self.total_output: int = 0
        self.n_calls: int = 0
        self.unique_qids: set[str] = set()
        self.model_costs: dict[str, float] = collections.defaultdict(float)
        # Rolling window: (timestamp, cost)
        self.recent: collections.deque[tuple[float, float]] = collections.deque()
        self.run_id: str = ""
        self.start_ts: float | None = None

    def add(self, entry: dict[str, Any]) -> None:
        cost = float(entry.get("cost_usd") or 0.0)
        self.total_cost += cost
        self.total_input += int(entry.get("input_tokens") or 0)
        self.total_cached += int(entry.get("cached_input_tokens") or 0)
        self.total_output += int(entry.get("output_tokens") or 0)
        self.n_calls += 1
        qid = entry.get("qid") or ""
        if qid:
            self.unique_qids.add(qid)
        model = entry.get("model") or "unknown"
        self.model_costs[model] += cost
        now = time.monotonic()
        self.recent.append((now, cost))
        if self.run_id == "":
            self.run_id = entry.get("run_id") or ""
        if self.start_ts is None:
            self.start_ts = now

    def rolling_rpm(self, window_s: float = 60.0) -> float:
        """Requests per minute over the last *window_s* seconds."""
        cutoff = time.monotonic() - window_s
        # Prune old entries
        while self.recent and self.recent[0][0] < cutoff:
            self.recent.popleft()
        count = len(self.recent)
        return count / (window_s / 60.0) if count > 0 else 0.0

    def rolling_cost_rate(self, window_s: float = 60.0) -> float:
        """USD/minute over the last *window_s* seconds."""
        cutoff = time.monotonic() - window_s
        window_entries = [(t, c) for t, c in self.recent if t >= cutoff]
        total_window_cost = sum(c for _, c in window_entries)
        return total_window_cost / (window_s / 60.0) if window_entries else 0.0

    @property
    def cache_hit_rate(self) -> float:
        total_prompt = self.total_input + self.total_cached
        if total_prompt == 0:
            return 0.0
        return self.total_cached / total_prompt

    def eta_minutes(self, budget: float | None) -> float | None:
        """Estimated minutes until budget is exhausted, or None if unknown."""
        if budget is None or budget <= 0:
            return None
        remaining = budget - self.total_cost
        if remaining <= 0:
            return 0.0
        rate = self.rolling_cost_rate()
        if rate <= 0:
            return None
        return remaining / rate


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _render_plain(state: _State, budget: float | None) -> str:
    lines = [
        f"Run: {state.run_id or '(unknown)'}",
        f"  Calls : {state.n_calls}  Queries: {len(state.unique_qids)}",
        f"  Cost  : ${state.total_cost:.4f}" + (f" / ${budget:.2f}" if budget else ""),
        f"  Tokens: {state.total_input + state.total_cached:,} in"
        f" ({state.cache_hit_rate:.1%} cached)  {state.total_output:,} out",
        f"  RPM   : {state.rolling_rpm():.1f}",
    ]
    eta = state.eta_minutes(budget)
    if eta is not None:
        lines.append(f"  ETA   : {eta:.1f} min until budget")
    if state.model_costs:
        lines.append("  Models:")
        for m, c in sorted(state.model_costs.items(), key=lambda x: -x[1]):
            lines.append(f"    {m}: ${c:.4f}")
    return "\n".join(lines)


def _render_rich(state: _State, budget: float | None) -> "Table":
    table = Table(title=f"Run: {state.run_id or '(unknown)'}", expand=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    budget_str = f"${state.total_cost:.4f}"
    if budget:
        budget_str += f" / ${budget:.2f}  ({state.total_cost/budget:.1%})"
    table.add_row("Cost", budget_str)

    cache_str = (
        f"{state.total_input + state.total_cached:,} prompt tokens  "
        f"({state.cache_hit_rate:.1%} cached)  |  {state.total_output:,} output"
    )
    table.add_row("Tokens", cache_str)
    table.add_row("Calls / Queries", f"{state.n_calls} / {len(state.unique_qids)}")
    table.add_row("RPM (60s)", f"{state.rolling_rpm():.1f}")

    eta = state.eta_minutes(budget)
    table.add_row(
        "ETA",
        f"{eta:.1f} min until budget" if eta is not None else "—",
    )

    if state.model_costs:
        for m, c in sorted(state.model_costs.items(), key=lambda x: -x[1]):
            table.add_row(f"  {m}", f"${c:.4f}")

    return table


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def watch(manifest_path: Path, budget: float | None, interval: float) -> None:
    """Tail *manifest_path* and refresh the dashboard every *interval* seconds.

    Args:
        manifest_path: Path to the live manifest JSONL file.
        budget: Optional total budget cap in USD for ETA computation.
        interval: Refresh interval in seconds.
    """
    state = _State()
    byte_offset = 0

    if _RICH:
        with Live(console=_console, refresh_per_second=1 / interval) as live:
            while True:
                new_entries, byte_offset = _tail_manifest(manifest_path, byte_offset)
                for e in new_entries:
                    state.add(e)
                live.update(_render_rich(state, budget))
                time.sleep(interval)
    else:
        while True:
            new_entries, byte_offset = _tail_manifest(manifest_path, byte_offset)
            for e in new_entries:
                state.add(e)
            # Clear terminal (best-effort)
            os.system("clear" if os.name != "nt" else "cls")
            print(_render_plain(state, budget))
            time.sleep(interval)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="run_status",
        description=(
            "Tail an active manifest JSONL file and print a rolling cost / "
            "RPM / cache-hit / ETA dashboard."
        ),
    )
    parser.add_argument(
        "manifest",
        type=Path,
        help="Path to the manifest.jsonl file to tail.",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Maximum cost in USD (for ETA and % display).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Refresh interval in seconds (default: 2.0).",
    )
    args = parser.parse_args()

    try:
        watch(args.manifest, args.budget, args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
