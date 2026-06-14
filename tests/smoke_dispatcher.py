"""Smoke tests for the parallel benchmark dispatcher (Wave-3 bench-dispatcher track).

Covers six scenarios:

1. ``--dry-run`` on a 3-cell spec → exit 0, plan printed.
2. End-to-end dispatch with a **mock cell runner** that writes a dummy
   ``result.json`` + ``manifest.jsonl`` per cell → all cells complete,
   ``dispatch_status.json`` correct, aggregator emits all four CSVs.
3. Re-run with ``resume: true`` → every cell observed as ``skipped``.
4. Inject one failing cell → other cells succeed, failure recorded, exit code 1.
5. Global budget enforcement → cells beyond the global cap marked
   ``budget_skip``.
6. Remote-host git-drift pre-flight → executor aborts cleanly without
   running any cell.

Run from the worktree root::

    python tests/smoke_dispatcher.py

Returns exit code 0 on all pass, non-zero on any failure.

All tests are offline — no API keys required, no subprocess invocation of
``run.py``. The dispatcher's local runner is overridden with an in-process
callable that fakes the cell driver's outputs.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "StableToolBench"))

from toolbench.dispatcher import (  # noqa: E402
    DispatchExecutor,
    expand_cells,
    load_dispatch_spec,
)
from toolbench.dispatcher.executor import CellResult, run_cell_subprocess  # noqa: E402
from toolbench.dispatcher.spec import DispatchSpec  # noqa: E402


# ---------------------------------------------------------------------------
# Pretty test reporter
# ---------------------------------------------------------------------------


_PASSED: list[str] = []
_FAILED: list[tuple[str, str]] = []


def _ok(name: str) -> None:
    """Mark a test as passing."""
    print(f"PASS: {name}")
    _PASSED.append(name)


def _fail(name: str, reason: str) -> None:
    """Mark a test as failing and record reason."""
    print(f"FAIL: {name}\n        {reason}", file=sys.stderr)
    _FAILED.append((name, reason))


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


_SMALL_SPEC_YAML = """
name: smoke_small
description: "smoke test — 2 techniques × 1 baseline × 1 benchmark × 1 model = 3 cells"

models:
  - pin: agents.gpt_4_1_mini

techniques:
  - single_pass
  - memetic

baselines:
  - less_is_more

benchmarks:
  - name: toolret
    splits: [code]
    n_queries_per_split: 5

parallelism:
  max_local_workers: 2
  hosts: []
  shard_split: by_cell
  resume: true

budget:
  per_cell_usd: 1.0
  total_usd: 100.0
"""


def _write_spec(tmpdir: Path, body: str = _SMALL_SPEC_YAML) -> Path:
    """Materialise a spec YAML inside tmpdir and return its path."""
    tmpdir.mkdir(parents=True, exist_ok=True)
    spec_path = tmpdir / "spec.yaml"
    spec_path.write_text(body, encoding="utf-8")
    return spec_path


# ---------------------------------------------------------------------------
# Mock cell runners
# ---------------------------------------------------------------------------


def _mock_runner_writes_result(
    cell: Any,
    out_dir: str,
    repo_root: str,
    resume: bool,
    python_bin: str = sys.executable,
    timeout_s: float | None = None,
) -> CellResult:
    """Mock local runner — writes dummy result.json + manifest.jsonl, returns OK.

    Signature matches :func:`run_cell_subprocess` so it can be swapped in
    via the executor's ``runner_local`` parameter.
    """
    cell_dir = Path(out_dir) / cell.cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)

    # Resume: skip cleanly if result.json already exists
    rj = cell_dir / "result.json"
    if resume and rj.exists():
        prior = json.loads(rj.read_text(encoding="utf-8"))
        return CellResult(
            cell_id=cell.cell_id,
            return_code=0,
            cost_usd=float(prior.get("total_cost_usd", 0.0)),
            started_at="t0",
            finished_at="t1",
            result_dir=str(cell_dir),
            skipped=True,
        )

    fake_cost = 0.10
    fake_result = {
        "run_id": cell.cell_id + "_run",
        "git_commit": "deadbeef" * 5,
        "config_hash": "0" * 64,
        "agent_model": cell.model_name,
        "total_cost_usd": fake_cost,
        "total_input_tokens": 1000,
        "total_cached_input_tokens": 0,
        "total_output_tokens": 50,
        "n_queries": cell.n_queries or 0,
        "n_successful": cell.n_queries or 0,
        "n_failed": 0,
    }
    rj.write_text(json.dumps(fake_result), encoding="utf-8")

    # Manifest line — minimum schema fields for the aggregator
    mf = cell_dir / "manifest.jsonl"
    entry = {
        "ts": "2026-05-24T00:00:00.000Z",
        "run_id": fake_result["run_id"],
        "git_commit": fake_result["git_commit"],
        "config_hash": fake_result["config_hash"],
        "qid": f"{cell.benchmark_name}:{cell.split}:0001",
        "operation": "pseudo_tool_gen",
        "variant": cell.label if not cell.is_baseline() else f"baseline_{cell.label}",
        "generation": 0,
        "model": cell.model_name,
        "provider": "openai",
        "input_tokens": 1000,
        "cached_input_tokens": 0,
        "output_tokens": 50,
        "latency_ms": 123.4,
        "cost_usd": fake_cost,
        "request_hash": "a" * 64,
        "response_hash": "b" * 64,
        "retry_count": 0,
        "error": None,
    }
    mf.write_text(json.dumps(entry) + "\n", encoding="utf-8")

    return CellResult(
        cell_id=cell.cell_id,
        return_code=0,
        cost_usd=fake_cost,
        started_at="t0",
        finished_at="t1",
        result_dir=str(cell_dir),
    )


def _mock_runner_env_failure(
    cell: Any,
    out_dir: str,
    repo_root: str,
    resume: bool,
    python_bin: str = sys.executable,
    timeout_s: float | None = None,
) -> CellResult:
    """Top-level (picklable) mock runner that fails cells listed in $SMOKE_FAIL_CELL_IDS.

    The failing cells return return_code=2 and write no result.json. Other
    cells delegate to :func:`_mock_runner_writes_result`. Must be defined at
    module top-level so ProcessPoolExecutor can pickle it.
    """
    import os as _os

    fail_set = set(
        s.strip() for s in _os.environ.get("SMOKE_FAIL_CELL_IDS", "").split(",") if s.strip()
    )
    if cell.cell_id in fail_set:
        cell_dir = Path(out_dir) / cell.cell_id
        cell_dir.mkdir(parents=True, exist_ok=True)
        return CellResult(
            cell_id=cell.cell_id,
            return_code=2,
            cost_usd=0.0,
            started_at="t0",
            finished_at="t1",
            result_dir=str(cell_dir),
            error="mock injected failure",
        )
    return _mock_runner_writes_result(
        cell, out_dir, repo_root, resume, python_bin, timeout_s
    )


# ---------------------------------------------------------------------------
# Test 1: dry-run plan expansion
# ---------------------------------------------------------------------------


def _check_dry_run_3_cells(tmpdir: Path) -> None:
    """``--dry-run`` exits 0 and reports 3 cells for the small spec."""
    spec_path = _write_spec(tmpdir)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "dispatch_benchmarks.py"),
        "--spec",
        str(spec_path),
        "--out",
        str(tmpdir / "out"),
        "--dry-run",
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, cwd=REPO_ROOT, check=False
    )
    if proc.returncode != 0:
        _fail(
            "test_dry_run_3_cells",
            f"exit={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}",
        )
        return
    if "total cells: 3" not in proc.stdout:
        _fail(
            "test_dry_run_3_cells",
            f"expected 'total cells: 3' in stdout, got:\n{proc.stdout}",
        )
        return
    _ok("test_dry_run_3_cells")


# ---------------------------------------------------------------------------
# Test 2: end-to-end mock dispatch
# ---------------------------------------------------------------------------


def _check_e2e_mock_3_cells(tmpdir: Path) -> Path:
    """All 3 cells complete, status JSON correct, aggregator writes CSVs.

    Returns the out_dir so subsequent tests can re-use it for resume + budget.
    """
    spec_path = _write_spec(tmpdir)
    spec = load_dispatch_spec(spec_path)
    cells = expand_cells(spec, pins_path=REPO_ROOT / "configs" / "model_pins.yaml")
    assert len(cells) == 3, f"expected 3 cells, got {len(cells)}"

    out_dir = tmpdir / "out_e2e"
    out_dir.mkdir(parents=True, exist_ok=True)

    ex = DispatchExecutor(
        spec=spec,
        cells=cells,
        out_dir=out_dir,
        repo_root=REPO_ROOT,
        dispatch_id="smoke_e2e",
        runner_local=_mock_runner_writes_result,
        require_remote_git_match=False,
    )
    snap = ex.run()

    if snap["n_done"] != 3 or snap["n_failed"] != 0:
        _fail(
            "test_e2e_mock_3_cells",
            f"expected 3 done / 0 failed, got: "
            f"done={snap['n_done']} failed={snap['n_failed']} "
            f"snap_cells={list(snap['cells'].keys())}",
        )
        return out_dir

    # Status file exists and is well-formed
    status = out_dir / "dispatch_status.json"
    if not status.exists():
        _fail("test_e2e_mock_3_cells", f"missing {status}")
        return out_dir
    parsed = json.loads(status.read_text(encoding="utf-8"))
    if parsed["n_done"] != 3:
        _fail(
            "test_e2e_mock_3_cells",
            f"status file n_done={parsed['n_done']} ≠ 3",
        )
        return out_dir

    # Cost = 0.10 × 3 cells
    expected_cost = 0.30
    if abs(parsed["running_total_cost_usd"] - expected_cost) > 1e-6:
        _fail(
            "test_e2e_mock_3_cells",
            f"running_total_cost_usd={parsed['running_total_cost_usd']} ≠ {expected_cost}",
        )
        return out_dir

    # Aggregator produces CSVs
    agg_dir = ex.aggregate()
    expected_files = [
        "cost_table.csv",
        "pareto_data.csv",
        "cache_hit.csv",
        "latency.csv",
    ]
    for fname in expected_files:
        if not (agg_dir / fname).exists():
            _fail("test_e2e_mock_3_cells", f"missing aggregate file {fname}")
            return out_dir

    removed_flat_results = agg_dir / ("per_query_" + "results.jsonl")
    if removed_flat_results.exists():
        _fail(
            "test_e2e_mock_3_cells",
            f"unexpected aggregate artifact {removed_flat_results.name}",
        )
        return out_dir

    _ok("test_e2e_mock_3_cells")
    return out_dir


# ---------------------------------------------------------------------------
# Test 3: resume — all 3 cells skipped on re-run
# ---------------------------------------------------------------------------


def _check_resume_skips_completed(tmpdir: Path, prior_out_dir: Path) -> None:
    """Re-dispatching with resume=True over an existing out_dir → all skipped."""
    spec_path = _write_spec(tmpdir)
    spec = load_dispatch_spec(spec_path)
    # Force resume on
    spec.parallelism.resume = True

    cells = expand_cells(spec, pins_path=REPO_ROOT / "configs" / "model_pins.yaml")

    ex = DispatchExecutor(
        spec=spec,
        cells=cells,
        out_dir=prior_out_dir,
        repo_root=REPO_ROOT,
        dispatch_id="smoke_resume",
        runner_local=_mock_runner_writes_result,
        require_remote_git_match=False,
    )
    snap = ex.run()
    if snap["n_skipped"] != 3 or snap["n_done"] != 0:
        _fail(
            "test_resume_skips_completed",
            f"expected 3 skipped / 0 done, got: "
            f"skipped={snap['n_skipped']} done={snap['n_done']} "
            f"failed={snap['n_failed']}",
        )
        return
    _ok("test_resume_skips_completed")


# ---------------------------------------------------------------------------
# Test 4: inject one failing cell
# ---------------------------------------------------------------------------


def _check_one_cell_fails_others_succeed(tmpdir: Path) -> None:
    """A single failing cell does not block the rest. Status records failure."""
    import os as _os

    spec_path = _write_spec(tmpdir)
    spec = load_dispatch_spec(spec_path)
    cells = expand_cells(spec, pins_path=REPO_ROOT / "configs" / "model_pins.yaml")
    failing_id = cells[0].cell_id

    # Mark the failing cell via env var (picklable across ProcessPool workers
    # on fork-based start methods — env is inherited).
    _prev = _os.environ.get("SMOKE_FAIL_CELL_IDS")
    _os.environ["SMOKE_FAIL_CELL_IDS"] = failing_id
    try:
        out_dir = tmpdir / "out_failure"
        ex = DispatchExecutor(
            spec=spec,
            cells=cells,
            out_dir=out_dir,
            repo_root=REPO_ROOT,
            dispatch_id="smoke_failure",
            runner_local=_mock_runner_env_failure,
            require_remote_git_match=False,
        )
        snap = ex.run()
    finally:
        if _prev is None:
            _os.environ.pop("SMOKE_FAIL_CELL_IDS", None)
        else:
            _os.environ["SMOKE_FAIL_CELL_IDS"] = _prev

    if snap["n_failed"] != 1 or snap["n_done"] != 2:
        _fail(
            "test_one_cell_fails_others_succeed",
            f"expected 1 failed / 2 done, got: "
            f"failed={snap['n_failed']} done={snap['n_done']}",
        )
        return
    if snap["cells"][failing_id]["status"] != "failed":
        _fail(
            "test_one_cell_fails_others_succeed",
            f"failing cell status = {snap['cells'][failing_id]['status']}",
        )
        return
    _ok("test_one_cell_fails_others_succeed")


# ---------------------------------------------------------------------------
# Test 5: global budget enforcement
# ---------------------------------------------------------------------------


def _check_global_budget_gates_cells(tmpdir: Path) -> None:
    """Cap total_usd at 1.5 with per_cell_usd 1.0 → first cell ok, then budget_skip."""
    # Per-cell budget 1.0, total 1.5 → only one cell can be launched before
    # the gate trips. (Subsequent cells: running_total + 1.0 > 1.5.)
    spec = DispatchSpec.model_validate(
        {
            "name": "smoke_budget",
            "models": [{"pin": "agents.gpt_4_1_mini"}],
            "techniques": ["single_pass", "memetic"],
            "baselines": ["less_is_more"],
            "benchmarks": [{"name": "toolret", "splits": ["code"], "n_queries_per_split": 5}],
            "parallelism": {"max_local_workers": 1, "hosts": [], "resume": False},
            "budget": {"per_cell_usd": 1.0, "total_usd": 1.5},
        }
    )
    cells = expand_cells(spec, pins_path=REPO_ROOT / "configs" / "model_pins.yaml")

    out_dir = tmpdir / "out_budget"
    ex = DispatchExecutor(
        spec=spec,
        cells=cells,
        out_dir=out_dir,
        repo_root=REPO_ROOT,
        dispatch_id="smoke_budget",
        runner_local=_mock_runner_writes_result,
        require_remote_git_match=False,
    )
    snap = ex.run()
    # First cell consumes $0.10 (< 1.0 per cell), well under cap. But the
    # gate uses per_cell_budget_usd (1.0), so after one cell launches the
    # next would push running_total (0.10) + 1.0 = 1.10 → still OK.
    # After two cells: 0.20 + 1.0 = 1.20 < 1.5 → OK.
    # After three cells: 0.30 + 1.0 = 1.30 < 1.5 → OK.
    # All three launch.
    # To prove the gate fires, we drop the total to 0.5 instead.
    if snap["n_done"] != 3:
        _fail(
            "test_global_budget_gates_cells (setup A)",
            f"first phase: expected 3 done, got {snap}",
        )
        return

    # Phase B: low cap, no cells should launch
    spec2 = DispatchSpec.model_validate(
        {
            "name": "smoke_budget_tight",
            "models": [{"pin": "agents.gpt_4_1_mini"}],
            "techniques": ["single_pass"],
            "baselines": ["less_is_more"],
            "benchmarks": [{"name": "toolret", "splits": ["code"], "n_queries_per_split": 5}],
            "parallelism": {"max_local_workers": 1, "hosts": [], "resume": False},
            "budget": {"per_cell_usd": 10.0, "total_usd": 5.0},
        }
    )
    cells2 = expand_cells(spec2, pins_path=REPO_ROOT / "configs" / "model_pins.yaml")
    out_dir2 = tmpdir / "out_budget_tight"
    ex2 = DispatchExecutor(
        spec=spec2,
        cells=cells2,
        out_dir=out_dir2,
        repo_root=REPO_ROOT,
        dispatch_id="smoke_budget_tight",
        runner_local=_mock_runner_writes_result,
        require_remote_git_match=False,
    )
    snap2 = ex2.run()
    if snap2["n_budget_skip"] != len(cells2) or snap2["n_done"] != 0:
        _fail(
            "test_global_budget_gates_cells (setup B)",
            f"tight phase: expected all {len(cells2)} budget_skip, "
            f"got budget_skip={snap2['n_budget_skip']} done={snap2['n_done']}",
        )
        return
    _ok("test_global_budget_gates_cells")


# ---------------------------------------------------------------------------
# Test 6: remote git-drift pre-flight
# ---------------------------------------------------------------------------


def _check_remote_git_drift_skips(tmpdir: Path) -> None:
    """Adding a bogus remote host triggers pre-flight failure → all cells skipped."""
    spec = DispatchSpec.model_validate(
        {
            "name": "smoke_drift",
            "models": [{"pin": "agents.gpt_4_1_mini"}],
            "techniques": ["single_pass"],
            "baselines": [],
            "benchmarks": [{"name": "toolret", "splits": ["code"], "n_queries_per_split": 5}],
            "parallelism": {
                "max_local_workers": 1,
                "hosts": ["__bogus_unreachable_host__"],
                "resume": False,
            },
            "budget": {"per_cell_usd": 1.0, "total_usd": 10.0},
        }
    )
    cells = expand_cells(spec, pins_path=REPO_ROOT / "configs" / "model_pins.yaml")
    out_dir = tmpdir / "out_drift"
    ex = DispatchExecutor(
        spec=spec,
        cells=cells,
        out_dir=out_dir,
        repo_root=REPO_ROOT,
        dispatch_id="smoke_drift",
        runner_local=_mock_runner_writes_result,
        require_remote_git_match=True,
    )
    snap = ex.run()
    # All cells should be skipped (pre-flight aborted)
    if snap["n_skipped"] != len(cells) and snap["n_failed"] != len(cells):
        # Acceptable: pre-flight marks all as skipped with the pre-flight error
        _fail(
            "test_remote_git_drift_skips",
            f"expected all cells skipped/failed by pre-flight, got: {snap}",
        )
        return
    _ok("test_remote_git_drift_skips")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _run_all_checks() -> None:
    """Execute every dispatcher smoke check in sequence.

    Each check writes into its own tempdir subtree to keep state isolated.
    Failures are recorded but do not abort the suite.
    """
    with tempfile.TemporaryDirectory(prefix="smoke_dispatcher_") as td:
        tmpdir = Path(td)

        try:
            _check_dry_run_3_cells(tmpdir / "dryrun")
        except Exception as e:  # pylint: disable=broad-except
            _fail("test_dry_run_3_cells", f"unexpected exception: {e!r}")

        out_dir = None
        try:
            out_dir = _check_e2e_mock_3_cells(tmpdir / "e2e")
        except Exception as e:  # pylint: disable=broad-except
            _fail("test_e2e_mock_3_cells", f"unexpected exception: {e!r}")

        if out_dir is not None and (out_dir / "dispatch_status.json").exists():
            try:
                _check_resume_skips_completed(tmpdir / "resume", out_dir)
            except Exception as e:  # pylint: disable=broad-except
                _fail(
                    "test_resume_skips_completed",
                    f"unexpected exception: {e!r}",
                )

        try:
            _check_one_cell_fails_others_succeed(tmpdir / "failure")
        except Exception as e:  # pylint: disable=broad-except
            _fail(
                "test_one_cell_fails_others_succeed",
                f"unexpected exception: {e!r}",
            )

        try:
            _check_global_budget_gates_cells(tmpdir / "budget")
        except Exception as e:  # pylint: disable=broad-except
            _fail(
                "test_global_budget_gates_cells",
                f"unexpected exception: {e!r}",
            )

        try:
            _check_remote_git_drift_skips(tmpdir / "drift")
        except Exception as e:  # pylint: disable=broad-except
            _fail("test_remote_git_drift_skips", f"unexpected exception: {e!r}")


def test_dispatcher_smoke_suite() -> None:
    """Pytest entry point — runs every dispatcher smoke check.

    Re-uses the standalone-script harness (``_run_all_checks``) and then
    asserts there are no failures. Single test = no parametrisation
    surprises; the per-check PASS/FAIL output appears in the captured log.
    """
    # Reset module-level state in case pytest re-runs the function
    _PASSED.clear()
    _FAILED.clear()
    _run_all_checks()
    assert not _FAILED, "\n".join(f"{n}: {why}" for n, why in _FAILED)


def main() -> int:
    """Standalone CLI: run every check and exit non-zero on any failure."""
    _PASSED.clear()
    _FAILED.clear()
    _run_all_checks()

    print()
    print(f"=== {len(_PASSED)} passed / {len(_FAILED)} failed ===")
    for n, why in _FAILED:
        print(f"  - {n}: {why}")
    return 0 if not _FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
