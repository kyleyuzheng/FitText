"""Dispatch executor — launches cells locally (ProcessPool) + remotely (SSH).

Concurrency model
-----------------
- Local: ``ProcessPoolExecutor(max_workers=N)``. Each worker forks once and
  invokes ``run_cell_subprocess`` which shells out to
  ``python run.py --config <generated.yaml>`` (FitText technique cells) or
  ``python scripts/run_baseline.py --baseline <name> --config <yaml>``
  (baseline cells). Cells write to ``<out_dir>/<cell_id>/`` — no shared
  mutable state between cells.

- Remote: each cell tagged for a remote host is shipped via ``ssh <host>
  "cd <remote_repo> && python run.py --config <remote_yaml>"``. The
  generated YAML is staged via ``scp`` (or ``rsync``) before launch and
  the result directory is rsynced back at the end.

Budget enforcement
------------------
Cells are enqueued in plan order. Before launching cell K, the executor
checks whether ``running_total + cell.per_cell_budget > spec.budget.total_usd``;
if so the cell is marked ``budget_skip`` and logged. The running total
incorporates costs from completed cells (read from each cell's
``result.json::total_cost_usd``).

Resumability
------------
If ``--resume`` is on and ``<out_dir>/<cell_id>/result.json`` already
exists, the cell is marked ``skipped`` with the prior cost folded into
the running total. (Mirrors §3.4 of EXECUTION_PLAN.md.)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from .plan import Cell
from .status import StatusTracker, _utc_now_iso

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cell-level subprocess driver
# ---------------------------------------------------------------------------


@dataclass
class CellResult:
    """Outcome of one cell — returned by the worker, consumed by the dispatcher."""

    cell_id: str
    return_code: int
    cost_usd: float
    started_at: str
    finished_at: str
    result_dir: str
    error: str | None = None
    skipped: bool = False  # True when resume hit existing result.json


def _cell_run_yaml_path(cell_dir: Path) -> Path:
    """Where a cell's generated run YAML is written."""
    return cell_dir / "run.yaml"


def _result_json_path(cell_dir: Path) -> Path:
    """Where ``run.py`` / ``run_baseline.py`` writes the result file.

    Both drivers write into ``<run_id>/result.json`` (a subdir per the
    auto-filled run_id). For the dispatcher we override ``infra.output_dir``
    to point at the cell_dir so the file lands at a predictable location.

    Convention: the executor writes a wrapping ``result.json`` at the cell
    root that mirrors the driver's result for easy lookup. See
    :func:`_finalize_cell_result`.
    """
    return cell_dir / "result.json"


def run_cell_subprocess(
    cell: Cell,
    out_dir: str,
    repo_root: str,
    resume: bool,
    python_bin: str = sys.executable,
    timeout_s: float | None = None,
) -> CellResult:
    """Worker entry point — run one cell to completion via subprocess.

    This function is invoked inside a ProcessPoolExecutor worker. It MUST be
    importable at module top level (no closures) so it can be pickled.

    Args:
        cell: The cell plan.
        out_dir: Dispatch out_dir root (cell writes to ``<out_dir>/<cell_id>/``).
        repo_root: Path to the FitText repo (where run.py lives).
        resume: If True and an existing ``result.json`` is present, skip.
        python_bin: Python interpreter to invoke (defaults to current).
        timeout_s: Optional wall-clock cap on the subprocess.

    Returns:
        :class:`CellResult` describing the outcome.
    """
    cell_dir = Path(out_dir) / cell.cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now_iso()

    # Resume: existing result.json → skip
    existing = _result_json_path(cell_dir)
    if resume and existing.exists():
        try:
            data = json.loads(existing.read_text(encoding="utf-8"))
            prior_cost = float(data.get("total_cost_usd", 0.0))
        except Exception:
            prior_cost = 0.0
        return CellResult(
            cell_id=cell.cell_id,
            return_code=0,
            cost_usd=prior_cost,
            started_at=started_at,
            finished_at=_utc_now_iso(),
            result_dir=str(cell_dir),
            skipped=True,
        )

    # Write the per-cell run YAML
    run_yaml = dict(cell.run_yaml_dict)
    # Override output_dir + cache_dir so every cell writes to its own dir.
    infra_override = run_yaml.setdefault("infra", {})
    infra_override["output_dir"] = str(cell_dir)
    # cache_dir lives next to the cell dir so re-running with resume picks
    # up the disk-backed ResponseCache.
    infra_override["cache_dir"] = str(cell_dir / "cache")
    # rsync_to is dispatch-managed (we rsync back from remote hosts later);
    # disable per-cell rsync.
    infra_override["rsync_to"] = None

    run_yaml_path = _cell_run_yaml_path(cell_dir)
    run_yaml_path.write_text(yaml.dump(run_yaml, sort_keys=True), encoding="utf-8")

    # Build command — technique cells go through run.py; baseline cells go
    # through scripts/run_baseline.py with --baseline <label>.
    if cell.is_baseline():
        cmd = [
            python_bin,
            str(Path(repo_root) / "scripts" / "run_baseline.py"),
            "--baseline",
            cell.label,
            "--config",
            str(run_yaml_path),
            "--out",
            str(cell_dir),
        ]
    else:
        cmd = [
            python_bin,
            str(Path(repo_root) / "run.py"),
            "--config",
            str(run_yaml_path),
            "--out",
            str(cell_dir),
        ]

    # Pipe stdout/stderr to a per-cell log so the main dispatcher stays clean
    log_path = cell_dir / "cell.log"
    error_msg: str | None = None
    rc: int = 1
    try:
        with log_path.open("w", encoding="utf-8") as logfh:
            logfh.write(f"# cell_id: {cell.cell_id}\n")
            logfh.write(f"# cmd: {' '.join(cmd)}\n\n")
            logfh.flush()
            proc = subprocess.run(
                cmd,
                cwd=repo_root,
                stdout=logfh,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
                check=False,
            )
            rc = proc.returncode
    except subprocess.TimeoutExpired as e:
        error_msg = f"timeout after {timeout_s}s"
        rc = 124
    except Exception as e:  # pylint: disable=broad-except
        error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        rc = 1

    # Pull cost from result.json if the driver wrote one
    cost_usd = 0.0
    if existing.exists():
        try:
            data = json.loads(existing.read_text(encoding="utf-8"))
            cost_usd = float(data.get("total_cost_usd", 0.0))
        except Exception as e:  # pylint: disable=broad-except
            log.warning("cell %s: failed to parse result.json: %s", cell.cell_id, e)
    else:
        # Driver writes to <cell_dir>/<run_id>/result.json. Scan one level.
        for sub in cell_dir.iterdir():
            if sub.is_dir():
                inner = sub / "result.json"
                if inner.exists():
                    try:
                        data = json.loads(inner.read_text(encoding="utf-8"))
                        cost_usd = float(data.get("total_cost_usd", 0.0))
                        # Mirror the inner file to the cell root for easy lookup
                        shutil.copy2(inner, existing)
                        break
                    except Exception:
                        pass

    finished_at = _utc_now_iso()
    return CellResult(
        cell_id=cell.cell_id,
        return_code=rc,
        cost_usd=cost_usd,
        started_at=started_at,
        finished_at=finished_at,
        result_dir=str(cell_dir),
        error=error_msg,
    )


# ---------------------------------------------------------------------------
# Remote (SSH) cell driver
# ---------------------------------------------------------------------------


def _git_head(repo_root: Path) -> str:
    """Return the local git HEAD SHA, or 'unknown'."""
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _remote_git_head(host: str, remote_repo: str) -> str:
    """Return the remote git HEAD SHA via ssh.

    Returns 'unknown' on any error (the caller must decide whether to
    treat that as a drift abort).
    """
    try:
        out = subprocess.check_output(
            [
                "ssh",
                host,
                f"cd {remote_repo} && git rev-parse HEAD",
            ],
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return "unknown"


def run_cell_remote(
    cell: Cell,
    out_dir: str,
    repo_root: str,
    host: str,
    remote_repo: str,
    resume: bool,
    timeout_s: float | None = None,
) -> CellResult:
    """Run one cell on a remote host over ssh, then rsync results back.

    Layout::

        local: <out_dir>/<cell_id>/run.yaml         (generated here)
        remote: <remote_repo>/dispatch_work/<cell_id>/run.yaml  (scp'd)
        remote: <remote_repo>/dispatch_work/<cell_id>/result.json (driver writes)
        local: <out_dir>/<cell_id>/                 (rsync'd back at end)

    Args:
        cell: Cell plan.
        out_dir: Local dispatch out_dir.
        repo_root: Local repo root.
        host: SSH alias.
        remote_repo: Path to the FitText clone on ``host``.
        resume: If True and a local result.json already exists, skip.
        timeout_s: Optional ssh wall-clock cap.
    """
    cell_dir = Path(out_dir) / cell.cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now_iso()
    existing = _result_json_path(cell_dir)
    if resume and existing.exists():
        try:
            data = json.loads(existing.read_text(encoding="utf-8"))
            prior_cost = float(data.get("total_cost_usd", 0.0))
        except Exception:
            prior_cost = 0.0
        return CellResult(
            cell_id=cell.cell_id,
            return_code=0,
            cost_usd=prior_cost,
            started_at=started_at,
            finished_at=_utc_now_iso(),
            result_dir=str(cell_dir),
            skipped=True,
        )

    # Stage YAML
    run_yaml = dict(cell.run_yaml_dict)
    remote_cell_dir = f"{remote_repo}/dispatch_work/{cell.cell_id}"
    infra_override = run_yaml.setdefault("infra", {})
    infra_override["output_dir"] = remote_cell_dir
    infra_override["cache_dir"] = f"{remote_cell_dir}/cache"
    infra_override["rsync_to"] = None
    run_yaml_path = _cell_run_yaml_path(cell_dir)
    run_yaml_path.write_text(yaml.dump(run_yaml, sort_keys=True), encoding="utf-8")

    # mkdir on remote, copy YAML
    try:
        subprocess.run(
            ["ssh", host, f"mkdir -p {remote_cell_dir}"],
            check=True,
            timeout=30,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["scp", str(run_yaml_path), f"{host}:{remote_cell_dir}/run.yaml"],
            check=True,
            timeout=60,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as e:
        return CellResult(
            cell_id=cell.cell_id,
            return_code=e.returncode or 1,
            cost_usd=0.0,
            started_at=started_at,
            finished_at=_utc_now_iso(),
            result_dir=str(cell_dir),
            error=f"ssh staging failed: {e}",
        )

    # Build remote command
    if cell.is_baseline():
        remote_cmd = (
            f"cd {remote_repo} && "
            f"python scripts/run_baseline.py "
            f"--baseline {cell.label} "
            f"--config {remote_cell_dir}/run.yaml "
            f"--out {remote_cell_dir}"
        )
    else:
        remote_cmd = (
            f"cd {remote_repo} && "
            f"python run.py "
            f"--config {remote_cell_dir}/run.yaml "
            f"--out {remote_cell_dir}"
        )

    log_path = cell_dir / "cell.log"
    rc = 1
    error_msg: str | None = None
    try:
        with log_path.open("w", encoding="utf-8") as logfh:
            logfh.write(f"# cell_id: {cell.cell_id}\n")
            logfh.write(f"# host: {host}\n")
            logfh.write(f"# cmd: {remote_cmd}\n\n")
            logfh.flush()
            proc = subprocess.run(
                ["ssh", host, remote_cmd],
                stdout=logfh,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
                check=False,
            )
            rc = proc.returncode
    except subprocess.TimeoutExpired:
        error_msg = f"ssh timeout after {timeout_s}s"
        rc = 124
    except Exception as e:  # pylint: disable=broad-except
        error_msg = f"{type(e).__name__}: {e}"
        rc = 1

    # Rsync back
    try:
        subprocess.run(
            ["rsync", "-a", f"{host}:{remote_cell_dir}/", f"{cell_dir}/"],
            check=False,
            timeout=300,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        if error_msg is None:
            error_msg = "rsync timeout"

    # Pull cost
    cost_usd = 0.0
    if existing.exists():
        try:
            data = json.loads(existing.read_text(encoding="utf-8"))
            cost_usd = float(data.get("total_cost_usd", 0.0))
        except Exception:
            pass
    else:
        for sub in cell_dir.iterdir():
            if sub.is_dir():
                inner = sub / "result.json"
                if inner.exists():
                    try:
                        data = json.loads(inner.read_text(encoding="utf-8"))
                        cost_usd = float(data.get("total_cost_usd", 0.0))
                        shutil.copy2(inner, existing)
                        break
                    except Exception:
                        pass

    return CellResult(
        cell_id=cell.cell_id,
        return_code=rc,
        cost_usd=cost_usd,
        started_at=started_at,
        finished_at=_utc_now_iso(),
        result_dir=str(cell_dir),
        error=error_msg,
    )


# ---------------------------------------------------------------------------
# DispatchExecutor — orchestration
# ---------------------------------------------------------------------------


class DispatchExecutor:
    """Orchestrates parallel cell execution across local pool + remote hosts.

    Lifecycle::

        ex = DispatchExecutor(spec, cells, out_dir, repo_root, dispatch_id)
        ex.run()              # blocks until all cells reach a terminal state
        ex.aggregate()        # produces cost CSVs under <out_dir>/aggregate/
    """

    def __init__(
        self,
        spec: Any,  # DispatchSpec — duck-typed to avoid import cycle
        cells: list[Cell],
        out_dir: Path,
        repo_root: Path,
        dispatch_id: str,
        tracker: StatusTracker | None = None,
        remote_repo_by_host: dict[str, str] | None = None,
        timeout_s: float | None = None,
        runner_local: Callable[..., CellResult] = run_cell_subprocess,
        runner_remote: Callable[..., CellResult] = run_cell_remote,
        require_remote_git_match: bool = True,
    ) -> None:
        """Build a dispatcher.

        Args:
            spec: Validated :class:`DispatchSpec`.
            cells: Expanded cell list.
            out_dir: Dispatch root directory.
            repo_root: Local FitText repo root.
            dispatch_id: Identifier for the dispatch run.
            tracker: Optional shared :class:`StatusTracker`. If None one is
                created at ``out_dir/dispatch_status.json``.
            remote_repo_by_host: Mapping ``host → remote_repo_path``. Hosts
                without an entry default to ``~/FitText``.
            timeout_s: Optional per-cell timeout in seconds.
            runner_local: Override for the local worker (used by tests).
            runner_remote: Override for the remote worker (used by tests).
            require_remote_git_match: If True, abort with hard error when a
                remote host's git HEAD differs from local.
        """
        self.spec = spec
        self.cells = cells
        self.out_dir = Path(out_dir)
        self.repo_root = Path(repo_root)
        self.dispatch_id = dispatch_id
        self.tracker = tracker or StatusTracker(
            out_dir=self.out_dir,
            dispatch_id=dispatch_id,
            spec_name=getattr(spec, "name", "unnamed"),
            budget_total_usd=float(spec.budget.total_usd),
        )
        self.remote_repo_by_host = remote_repo_by_host or {}
        self.timeout_s = timeout_s
        self._runner_local = runner_local
        self._runner_remote = runner_remote
        self._require_remote_git_match = require_remote_git_match

        # Assign cells to hosts round-robin: coordinator + each remote host
        # gets an equal slice of cells. This keeps each provider rate-limit
        # cap independent per host.
        self._assign_hosts()

        # Pre-register cells in the tracker
        for c in self.cells:
            from .status import CellStatus

            self.tracker.add_cell(
                CellStatus(
                    cell_id=c.cell_id,
                    kind=c.kind,
                    label=c.label,
                    model=c.model_name,
                    benchmark=c.benchmark_name,
                    split=c.split,
                    host=c.host,
                )
            )

    # -- host assignment --------------------------------------------------

    def _assign_hosts(self) -> None:
        """Round-robin cells across coordinator + remote hosts."""
        hosts = ["local"] + list(self.spec.parallelism.hosts)
        for i, c in enumerate(self.cells):
            c.host = hosts[i % len(hosts)]

    # -- pre-flight -------------------------------------------------------

    def preflight(self) -> None:
        """Confirm remote git HEAD matches local HEAD (drift guard).

        Raises:
            RuntimeError: If any remote host's HEAD differs and
                ``require_remote_git_match`` is True.
        """
        if not self.spec.parallelism.hosts:
            return
        local = _git_head(self.repo_root)
        if local == "unknown":
            log.warning(
                "Cannot determine local git HEAD; skipping remote git match check."
            )
            return
        mismatches = []
        for host in self.spec.parallelism.hosts:
            remote_repo = self.remote_repo_by_host.get(host, "~/FitText")
            remote = _remote_git_head(host, remote_repo)
            if remote == "unknown":
                mismatches.append((host, "unreachable"))
            elif remote != local:
                mismatches.append((host, f"remote={remote[:12]} local={local[:12]}"))
        if mismatches:
            msg = "git drift on remote hosts: " + ", ".join(
                f"{h} ({why})" for h, why in mismatches
            )
            if self._require_remote_git_match:
                raise RuntimeError(msg)
            log.warning(msg)

    # -- run --------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Execute all cells. Returns the final status snapshot.

        Budget enforcement and resumability are handled here, before each
        cell is submitted to the pool. Failures in individual cells do NOT
        abort other cells — they only mark the failed cell.
        """
        # Optional pre-flight (skip silently if no remote hosts)
        try:
            self.preflight()
        except RuntimeError as e:
            log.error("Pre-flight failed: %s", e)
            # Mark all cells as skipped due to drift
            for c in self.cells:
                self.tracker.mark(
                    c.cell_id,
                    status="skipped",
                    error=f"pre-flight: {e}",
                    finished_at=_utc_now_iso(),
                )
            return self.tracker.snapshot()

        local_cells = [c for c in self.cells if c.host == "local"]
        remote_cells = [c for c in self.cells if c.host != "local"]

        # Submit local cells to a ProcessPoolExecutor; submit remote cells
        # to a separate ThreadPoolExecutor-equivalent (we use Process here
        # for uniform plumbing but ssh blocks anyway).
        max_local = max(1, int(self.spec.parallelism.max_local_workers))
        total_budget = float(self.spec.budget.total_usd)

        with ProcessPoolExecutor(max_workers=max_local) as local_pool:
            from concurrent.futures import ThreadPoolExecutor

            remote_pool = ThreadPoolExecutor(
                max_workers=max(1, len(self.spec.parallelism.hosts) * max_local)
            )
            try:
                future_to_cell: dict = {}

                for c in self.cells:
                    # Budget gate
                    if (
                        self.tracker.running_total_cost() + c.per_cell_budget_usd
                        > total_budget
                    ):
                        self.tracker.mark(
                            c.cell_id,
                            status="budget_skip",
                            error=(
                                f"global budget {total_budget:.2f} would be exceeded; "
                                f"current={self.tracker.running_total_cost():.4f}, "
                                f"cell_budget={c.per_cell_budget_usd:.2f}"
                            ),
                            finished_at=_utc_now_iso(),
                        )
                        continue

                    self.tracker.mark(
                        c.cell_id, status="running", started_at=_utc_now_iso()
                    )

                    if c.host == "local":
                        fut = local_pool.submit(
                            self._runner_local,
                            c,
                            str(self.out_dir),
                            str(self.repo_root),
                            self.spec.parallelism.resume,
                            sys.executable,
                            self.timeout_s,
                        )
                    else:
                        remote_repo = self.remote_repo_by_host.get(
                            c.host, "~/FitText"
                        )
                        fut = remote_pool.submit(
                            self._runner_remote,
                            c,
                            str(self.out_dir),
                            str(self.repo_root),
                            c.host,
                            remote_repo,
                            self.spec.parallelism.resume,
                            self.timeout_s,
                        )
                    future_to_cell[fut] = c

                # Drain
                for fut in as_completed(future_to_cell):
                    c = future_to_cell[fut]
                    try:
                        cr: CellResult = fut.result()
                    except Exception as e:  # pylint: disable=broad-except
                        self.tracker.mark(
                            c.cell_id,
                            status="failed",
                            error=f"{type(e).__name__}: {e}",
                            finished_at=_utc_now_iso(),
                        )
                        continue

                    if cr.skipped:
                        status = "skipped"
                    elif cr.return_code == 0:
                        status = "done"
                    else:
                        status = "failed"

                    self.tracker.mark(
                        c.cell_id,
                        status=status,
                        return_code=cr.return_code,
                        cost_usd=cr.cost_usd,
                        started_at=cr.started_at,
                        finished_at=cr.finished_at,
                        result_dir=cr.result_dir,
                        error=cr.error,
                    )
            finally:
                remote_pool.shutdown(wait=True)

        self.tracker.finalise()
        return self.tracker.snapshot()

    # -- aggregation ------------------------------------------------------

    def aggregate(self) -> Path:
        """Run the manifest aggregator over all completed cells.

        Outputs:
          - ``<out_dir>/aggregate/cost_table.csv``
          - ``<out_dir>/aggregate/pareto_data.csv``
          - ``<out_dir>/aggregate/cache_hit.csv``
          - ``<out_dir>/aggregate/latency.csv``

        Returns:
            The aggregate output directory.
        """
        from toolbench.dispatcher.aggregate import (  # local to avoid cycle
            run_aggregator,
        )

        agg_dir = self.out_dir / "aggregate"
        agg_dir.mkdir(parents=True, exist_ok=True)
        run_aggregator(out_root=self.out_dir, agg_dir=agg_dir)
        return agg_dir
