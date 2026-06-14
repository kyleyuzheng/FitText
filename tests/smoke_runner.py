"""Smoke tests for the config-runner track.

Tests:
  1. Load reproduce_paper.yaml via resolve_config(); assert all required fields present.
  2. Compute config_hash(); assert deterministic across two resolves.
  3. Run ``python run.py --config ... --dry-run`` via subprocess; assert exit 0 and
     resolved config printed.
  4. Run ``python scripts/expand_sweep.py configs/sweeps/hp_alpha.yaml``; assert
     N per-cell YAMLs land under the runtime root.
  5. Re-resolve one generated cell; assert it validates and has a unique config_hash
     relative to the base run.

Run from the worktree root:
    python tests/smoke_runner.py

Returns exit code 0 on all pass, non-zero on any failure.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

# Ensure we import from THIS worktree
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from toolbench.runner import resolve_config
from toolbench.runner.schema import RunConfig


def _fail(msg: str) -> None:
    """Print failure message and exit with code 1."""
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def _ok(msg: str) -> None:
    """Print pass message."""
    print(f"PASS: {msg}")


# ---------------------------------------------------------------------------
# Test 1: resolve reproduce_paper.yaml
# ---------------------------------------------------------------------------

def _load_paper_config() -> RunConfig:
    config_path = REPO_ROOT / "configs" / "runs" / "reproduce_paper.yaml"
    return resolve_config(config_path, repo_root=REPO_ROOT)


def test_resolve_paper_config() -> None:
    """Load reproduce_paper.yaml and assert all top-level fields are populated."""
    cfg = _load_paper_config()

    # Required top-level fields
    assert cfg.model is not None, "model field missing"
    assert cfg.fittext is not None, "fittext field missing"
    assert cfg.benchmark is not None, "benchmark field missing"
    assert cfg.evaluator is not None, "evaluator field missing"
    assert cfg.embedder is not None, "embedder field missing"
    assert cfg.infra is not None, "infra field missing"
    assert cfg.budget is not None, "budget field missing"

    # Check key values match paper config
    assert cfg.model.name == "gpt-4.1-mini-2025-04-14", f"unexpected model: {cfg.model.name}"
    assert cfg.fittext.variant == "memetic", f"unexpected variant: {cfg.fittext.variant}"
    assert cfg.fittext.fitness_alpha == 0.7, f"unexpected alpha: {cfg.fittext.fitness_alpha}"
    assert cfg.benchmark.name == "toolret", f"unexpected benchmark: {cfg.benchmark.name}"
    assert set(cfg.benchmark.splits) == {"code", "customized", "web", "toolbench"}, \
        f"unexpected splits: {cfg.benchmark.splits}"
    assert cfg.run_id is not None, "run_id not auto-filled"

    _ok("test_resolve_paper_config")


# ---------------------------------------------------------------------------
# Test 2: config_hash determinism
# ---------------------------------------------------------------------------

def test_config_hash_determinism() -> None:
    """Assert config_hash() is identical across two independent resolves."""
    config_path = REPO_ROOT / "configs" / "runs" / "reproduce_paper.yaml"
    cfg1 = resolve_config(config_path, repo_root=REPO_ROOT)
    cfg2 = resolve_config(config_path, repo_root=REPO_ROOT)

    h1 = cfg1.config_hash()
    h2 = cfg2.config_hash()
    assert h1 == h2, f"config_hash not deterministic: {h1!r} != {h2!r}"
    assert len(h1) == 64, f"expected 64-char hash, got {len(h1)}"

    _ok(f"test_config_hash_determinism (hash={h1[:12]}...)")


# ---------------------------------------------------------------------------
# Test 3: dry-run CLI invocation
# ---------------------------------------------------------------------------

def test_dry_run_cli() -> None:
    """Run ``python run.py --dry-run`` and assert exit 0 + config output."""
    config_path = REPO_ROOT / "configs" / "runs" / "reproduce_paper.yaml"
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "run.py"),
         "--config", str(config_path),
         "--dry-run"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        _fail(
            f"run.py --dry-run exited with {result.returncode}\n"
            f"stdout: {result.stdout[:500]}\n"
            f"stderr: {result.stderr[:500]}"
        )

    # Check resolved config was printed
    assert "--- resolved config ---" in result.stdout, \
        "Expected '--- resolved config ---' in stdout"
    assert "gpt-4.1-mini-2025-04-14" in result.stdout, \
        "Expected model name in resolved config output"
    assert "DRY RUN" in result.stderr or "DRY RUN" in result.stdout, \
        "Expected 'DRY RUN' message"

    _ok("test_dry_run_cli")


# ---------------------------------------------------------------------------
# Test 4: expand_sweep generates per-cell YAMLs
# ---------------------------------------------------------------------------

def _generated_dir() -> Path:
    return REPO_ROOT / "runs" / "test_generated_configs"


def _run_expand_sweep() -> list[Path]:
    generated_dir = _generated_dir()
    if generated_dir.exists():
        shutil.rmtree(generated_dir)

    sweep_path = REPO_ROOT / "configs" / "sweeps" / "hp_alpha.yaml"
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "expand_sweep.py"),
         str(sweep_path), "--out-dir", str(generated_dir)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        _fail(
            f"expand_sweep.py exited with {result.returncode}\n"
            f"stdout: {result.stdout[:500]}\n"
            f"stderr: {result.stderr[:500]}"
        )

    return list(generated_dir.glob("*.yaml"))


def test_expand_sweep() -> None:
    """Expand hp_alpha.yaml and assert generated files land under the runtime root."""
    generated_files = _run_expand_sweep()
    if len(generated_files) == 0:
        _fail(f"No files generated in {_generated_dir()}")

    # hp_alpha.yaml has 6 params with 3+4+4+4+3+3 = 21 values → 21 cells
    expected_min = 10  # conservative lower bound
    assert len(generated_files) >= expected_min, \
        f"Expected at least {expected_min} generated files, got {len(generated_files)}"

    _ok(f"test_expand_sweep ({len(generated_files)} cells generated)")


# ---------------------------------------------------------------------------
# Test 5: generated cell resolves with unique config_hash
# ---------------------------------------------------------------------------

def test_sweep_cell_unique_hash() -> None:
    """Re-resolve first generated sweep cell; assert it validates + has unique hash."""
    generated_files = list(_generated_dir().glob("*.yaml"))
    if not generated_files:
        # If sweep hasn't been expanded yet in this test session, do it now
        generated_files = _run_expand_sweep()
    assert generated_files, "no generated sweep cells available"

    base_cfg = _load_paper_config()
    base_hash = base_cfg.config_hash()

    cell_path = sorted(generated_files)[0]
    cell_cfg = resolve_config(cell_path, repo_root=REPO_ROOT)

    assert isinstance(cell_cfg, RunConfig), "Generated cell did not resolve to RunConfig"

    cell_hash = cell_cfg.config_hash()
    assert len(cell_hash) == 64, f"Expected 64-char hash, got {len(cell_hash)}"
    assert cell_hash != base_hash, \
        f"Generated cell has same config_hash as base — overrides not applied\n" \
        f"cell: {cell_path.name}\nbase_hash: {base_hash}\ncell_hash: {cell_hash}"

    _ok(f"test_sweep_cell_unique_hash (cell={cell_path.stem}, hash={cell_hash[:12]}...)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all smoke tests in order."""
    print(f"\nSmoke tests — config-runner track")
    print(f"Repo root: {REPO_ROOT}\n")

    test_resolve_paper_config()
    test_config_hash_determinism()
    test_dry_run_cli()
    test_expand_sweep()
    test_sweep_cell_unique_hash()

    print("\nAll smoke tests PASSED.")


if __name__ == "__main__":
    main()
