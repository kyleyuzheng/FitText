#!/usr/bin/env python3
"""Live TUI monitor for the parallel benchmark dispatcher.

Tails ``<dispatch_out>/dispatch_status.json`` and prints a status table on
every poll interval. Uses ``rich`` if available, otherwise falls back to a
plain-text table.

Usage::

    python scripts/dispatch_monitor.py --out "${FITTEXT_RUNTIME_ROOT:-runs}/$JOB_ID/dispatch"
    python scripts/dispatch_monitor.py --out "${FITTEXT_RUNTIME_ROOT:-runs}/$JOB_ID/dispatch" --interval 5
    python scripts/dispatch_monitor.py --out "${FITTEXT_RUNTIME_ROOT:-runs}/$JOB_ID/dispatch" --once

The monitor is **read-only** — it never mutates ``dispatch_status.json``.
The dispatcher writes the file atomically (write-tempfile-then-rename) so
the monitor never observes a half-written state.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live

    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False


# ---------------------------------------------------------------------------
# Status reading
# ---------------------------------------------------------------------------


def _read_status(status_path: Path) -> dict[str, Any] | None:
    """Read the dispatch_status.json snapshot.

    Returns None if the file doesn't exist yet (dispatcher hasn't started).
    """
    if not status_path.exists():
        return None
    try:
        return json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # Atomic rename should preclude this; treat as transient miss.
        return None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_header(state: dict[str, Any]) -> str:
    """One-line top-banner summary."""
    return (
        f"dispatch_id={state.get('dispatch_id', '?')}  "
        f"spec={state.get('spec_name', '?')}  "
        f"n_cells={state.get('n_cells', 0)}  "
        f"done={state.get('n_done', 0)}  "
        f"running={state.get('n_running', 0)}  "
        f"failed={state.get('n_failed', 0)}  "
        f"skipped={state.get('n_skipped', 0)}  "
        f"budget_skip={state.get('n_budget_skip', 0)}  "
        f"cost=${state.get('running_total_cost_usd', 0.0):.2f} / "
        f"${state.get('budget_total_usd', 0.0):.2f}"
    )


def _row_iter(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Return cell rows sorted by status priority then cell_id."""
    cells = state.get("cells", {})
    rows = list(cells.values())
    # Priority: running > failed > pending > done > skipped > budget_skip
    priority = {
        "running": 0,
        "failed": 1,
        "pending": 2,
        "done": 3,
        "skipped": 4,
        "budget_skip": 5,
    }
    rows.sort(
        key=lambda r: (
            priority.get(r.get("status", "pending"), 99),
            r.get("cell_id", ""),
        )
    )
    return rows


def _render_rich(state: dict[str, Any]) -> "Table":  # type: ignore[name-defined]
    """Build a rich Table for the current snapshot."""
    table = Table(title=_render_header(state), expand=True)
    table.add_column("cell_id", style="cyan", no_wrap=False, overflow="fold")
    table.add_column("status", justify="left")
    table.add_column("host", justify="left")
    table.add_column("model", justify="left")
    table.add_column("kind:label", justify="left")
    table.add_column("bench:split", justify="left")
    table.add_column("$cost", justify="right")
    table.add_column("rc", justify="right")

    status_color = {
        "running": "yellow",
        "done": "green",
        "failed": "red",
        "skipped": "blue",
        "budget_skip": "magenta",
        "pending": "white",
    }
    for r in _row_iter(state):
        status = r.get("status", "pending")
        color = status_color.get(status, "white")
        table.add_row(
            r.get("cell_id", ""),
            f"[{color}]{status}[/{color}]",
            r.get("host", "local"),
            r.get("model", ""),
            f"{r.get('kind', '')}:{r.get('label', '')}",
            f"{r.get('benchmark', '')}:{r.get('split', '')}",
            f"{r.get('cost_usd', 0.0):.4f}",
            "" if r.get("return_code") is None else str(r.get("return_code")),
        )
    return table


def _render_plain(state: dict[str, Any]) -> str:
    """Plain-text table (rich-free fallback)."""
    lines = [_render_header(state), "-" * 80]
    fmt = "{cid:<54} {status:<11} {host:<8} {cost:>9}"
    lines.append(
        fmt.format(cid="cell_id", status="status", host="host", cost="$cost")
    )
    lines.append("-" * 80)
    for r in _row_iter(state):
        lines.append(
            fmt.format(
                cid=r.get("cell_id", "")[:54],
                status=r.get("status", ""),
                host=r.get("host", "local")[:8],
                cost=f"{r.get('cost_usd', 0.0):.4f}",
            )
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(
        prog="dispatch_monitor",
        description="Live monitor for dispatch_status.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Dispatch output dir (contains dispatch_status.json).",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Refresh interval in seconds.",
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="Print a single snapshot and exit (no live loop).",
    )
    p.add_argument(
        "--no-rich",
        action="store_true",
        help="Disable rich rendering; use plain-text fallback.",
    )
    args = p.parse_args(argv)

    status_path = Path(args.out) / "dispatch_status.json"

    use_rich = _HAS_RICH and not args.no_rich

    if args.once:
        state = _read_status(status_path)
        if state is None:
            print(f"No status file at {status_path}", file=sys.stderr)
            return 1
        if use_rich:
            Console().print(_render_rich(state))
        else:
            print(_render_plain(state))
        return 0

    # Live loop
    if use_rich:
        console = Console()
        with Live(refresh_per_second=max(1, int(1 / args.interval)), console=console) as live:
            while True:
                state = _read_status(status_path)
                if state is None:
                    live.update(f"[grey]waiting for {status_path} ...[/grey]")
                else:
                    live.update(_render_rich(state))
                    finished = state.get("finished")
                    if finished:
                        # One last refresh then exit
                        time.sleep(args.interval)
                        return 0
                time.sleep(args.interval)
    else:
        while True:
            state = _read_status(status_path)
            os.system("clear")  # nosec: simple TUI; not security-sensitive
            if state is None:
                print(f"waiting for {status_path} ...")
            else:
                print(_render_plain(state))
                if state.get("finished"):
                    return 0
            time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
