"""Smoke tests for the eval-pin track — model pin consistency.

Tests:
  1. load_pins() returns dict with all required top-level keys.
  2. resolve_config(reproduce_paper.yaml) returns pinned judge_model,
     simulator_model, and embedder.name values.
  3. check_pin_consistency.py exits 0 on the clean repo.
  4. Injecting a deliberate mismatch (temp file with gpt-99-fake) causes
     check_pin_consistency.py to exit 1.

Run from the worktree root:
    python tests/smoke_pin_consistency.py

Returns exit code 0 on all pass, non-zero on any failure.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "StableToolBench"))

from toolbench.observability.pins import load_pins
from toolbench.runner import resolve_config


def _fail(msg: str) -> None:
    """Print failure and exit 1."""
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def _ok(msg: str) -> None:
    """Print pass."""
    print(f"PASS: {msg}")


# ---------------------------------------------------------------------------
# Test 1: load_pins() returns all required top-level keys
# ---------------------------------------------------------------------------

def test_load_pins_keys() -> None:
    """load_pins() must return agents, evaluators, embedders."""
    pins = load_pins(REPO_ROOT)

    for key in ("agents", "evaluators", "embedders"):
        if key not in pins:
            _fail(f"load_pins() missing top-level key: {key!r}")

    # Spot-check required sub-keys
    assert "gpt_4_1_mini" in pins["agents"], "agents.gpt_4_1_mini missing"
    assert "judge" in pins["evaluators"], "evaluators.judge missing"
    assert "simulator" in pins["evaluators"], "evaluators.simulator missing"
    assert "toolret" in pins["embedders"], "embedders.toolret missing"
    assert "stb" in pins["embedders"], "embedders.stb missing"

    # OpenAI/Anthropic model strings must contain a date fragment (no bare aliases).
    # vLLM-served open-weight models (Qwen*, deepseek*) have no dated HF tag — exempt.
    _VLLM_PREFIXES = ("Qwen", "qwen", "DeepSeek", "deepseek", "llama", "Llama", "mistral")
    for key, val in pins["agents"].items():
        if "TODO" in val:
            continue  # submission-time TODOs are explicitly allowed
        if any(val.startswith(p) for p in _VLLM_PREFIXES):
            continue  # vLLM-served open-weight; no dated API tag exists
        # API-served models must use a dated tag (contains "-20" year fragment)
        if "-20" not in val:
            _fail(
                f"agents.{key} = {val!r} looks like a floating alias — "
                "use a dated or revisioned tag in configs/model_pins.yaml"
            )

    _ok("test_load_pins_keys")


# ---------------------------------------------------------------------------
# Test 2: resolve_config returns pinned evaluator + embedder values
# ---------------------------------------------------------------------------

def test_resolve_config_pins() -> None:
    """resolve_config(reproduce_paper.yaml) must return pinned judge/embedder."""
    config_path = REPO_ROOT / "configs" / "runs" / "reproduce_paper.yaml"
    if not config_path.exists():
        _fail(f"reproduce_paper.yaml not found at {config_path}")

    cfg = resolve_config(config_path, repo_root=REPO_ROOT)
    pins = load_pins(REPO_ROOT)

    expected_judge = pins["evaluators"]["judge"]
    expected_sim   = pins["evaluators"]["simulator"]
    expected_emb   = pins["embedders"]["toolret"]

    if cfg.evaluator.judge_model != expected_judge:
        _fail(
            f"judge_model mismatch: got {cfg.evaluator.judge_model!r}, "
            f"expected {expected_judge!r} (from model_pins.yaml)"
        )
    if cfg.evaluator.simulator_model != expected_sim:
        _fail(
            f"simulator_model mismatch: got {cfg.evaluator.simulator_model!r}, "
            f"expected {expected_sim!r} (from model_pins.yaml)"
        )
    if cfg.embedder.name != expected_emb:
        _fail(
            f"embedder.name mismatch: got {cfg.embedder.name!r}, "
            f"expected {expected_emb!r} (from model_pins.yaml)"
        )

    _ok(
        f"test_resolve_config_pins "
        f"(judge={cfg.evaluator.judge_model}, embedder={cfg.embedder.name})"
    )


# ---------------------------------------------------------------------------
# Test 3: check_pin_consistency.py exits 0 on clean repo
# ---------------------------------------------------------------------------

def test_check_pin_consistency_clean() -> None:
    """check_pin_consistency.py must exit 0 on the unmodified repo."""
    script = REPO_ROOT / "scripts" / "check_pin_consistency.py"
    result = subprocess.run(
        [sys.executable, str(script), "--repo-root", str(REPO_ROOT)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        _fail(
            f"check_pin_consistency.py exited {result.returncode} on clean repo\n"
            f"stderr: {result.stderr[:800]}"
        )

    _ok("test_check_pin_consistency_clean")


# ---------------------------------------------------------------------------
# Test 4: deliberate mismatch triggers exit 1
# ---------------------------------------------------------------------------

def test_check_pin_consistency_mismatch() -> None:
    """Injecting a bare gpt-99-fake string must cause exit 1."""
    script = REPO_ROOT / "scripts" / "check_pin_consistency.py"

    # Write a temp Python file with a deliberate hardcoded model string.
    # Place it in StableToolBench/toolbench/ so it is scanned (not in an allowlisted dir).
    target_dir = REPO_ROOT / "StableToolBench" / "toolbench" / "inference"
    target_dir.mkdir(parents=True, exist_ok=True)

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        dir=str(target_dir),
        delete=False,
        prefix="_pin_test_mismatch_",
    )
    tmp.write('model = "gpt-99-fake"  # deliberate mismatch for smoke test\n')
    tmp.close()
    tmp_path = Path(tmp.name)

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--repo-root",
                str(REPO_ROOT),
                "--extra-path",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            _fail(
                f"check_pin_consistency.py exited 0 despite mismatch in {tmp_path.name} — "
                "expected exit 1"
            )
        # Make sure it flagged our specific file
        if tmp_path.name not in result.stderr:
            _fail(
                f"check_pin_consistency.py exited {result.returncode} but did not mention "
                f"{tmp_path.name} in stderr. stderr:\n{result.stderr[:500]}"
            )
    finally:
        tmp_path.unlink(missing_ok=True)

    _ok("test_check_pin_consistency_mismatch (deliberate mismatch correctly detected)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all pin-consistency smoke tests in order."""
    print(f"\nSmoke tests — eval-pin track")
    print(f"Repo root: {REPO_ROOT}\n")

    test_load_pins_keys()
    test_resolve_config_pins()
    test_check_pin_consistency_clean()
    test_check_pin_consistency_mismatch()

    print("\nAll smoke_pin_consistency tests PASSED.")


if __name__ == "__main__":
    main()
