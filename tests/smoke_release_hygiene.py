"""Smoke tests for the FitText release hygiene audit."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts import check_release_hygiene as hygiene


def _remove_known_test_artifacts() -> None:
    """Remove ignored files that other smoke tests intentionally generate."""
    shutil.rmtree(REPO_ROOT / "runs" / "test_generated_configs", ignore_errors=True)
    shutil.rmtree(REPO_ROOT / "tmp" / "gepa-dspy-research", ignore_errors=True)


def _write_minimal_e2e_evidence(runtime_root: Path) -> None:
    """Create the smallest artifact set accepted as live pass-rate evidence."""
    raw_dir = runtime_root / "stb" / "raw_output" / "run" / "G1_instruction" / "submethod"
    raw_dir.mkdir(parents=True)
    (raw_dir / "answer.json").write_text("{}", encoding="utf-8")

    converted_dir = runtime_root / "stb" / "model_predictions_converted" / "run"
    converted_dir.mkdir(parents=True)
    (converted_dir / "G1_instruction.json").write_text("[]", encoding="utf-8")

    pass_rate_dir = runtime_root / "stb" / "pass_rate_results" / "run"
    pass_rate_dir.mkdir(parents=True)
    (pass_rate_dir / "pass_rate.csv").write_text("metric,value\npass_rate,1.0\n", encoding="utf-8")
    (pass_rate_dir / ".done").write_text("", encoding="utf-8")


def test_stale_project_name_detected(tmp_path: Path) -> None:
    """Release audit must flag the old repository name."""
    rel = "README.md"
    stale = "old " + "Tool" + "-Retrieval\n"
    (tmp_path / rel).write_text(stale, encoding="utf-8")

    result = hygiene.check_stale_strings(tmp_path, [rel])

    assert not result.ok
    assert any("old project name" in detail for detail in result.details)


def test_secret_shaped_placeholder_detected(tmp_path: Path) -> None:
    """Key-shaped values should fail even in example files."""
    rel = ".env.example"
    key_value = "sk-" + "thislooksreal1234567890"
    (tmp_path / rel).write_text(f"OPENAI_API_KEY={key_value}\n", encoding="utf-8")

    result = hygiene.check_secret_shapes(tmp_path, [rel])

    assert not result.ok
    assert result.details == [".env.example:1: secret-shaped token"]


def test_generated_data_extension_detected() -> None:
    """Generated result/data formats must not be tracked."""
    result = hygiene.check_tracked_generated_files(["src/app.py", "runs/result.jsonl"])

    assert not result.ok
    assert result.details == ["runs/result.jsonl"]


def test_local_artifacts_detected(tmp_path: Path) -> None:
    """Backup and bytecode directories should be release blockers."""
    (tmp_path / "module").mkdir()
    (tmp_path / "module" / "__pycache__").mkdir()
    (tmp_path / "notes.md.bak").write_text("backup\n", encoding="utf-8")

    result = hygiene.check_local_artifacts(tmp_path)

    assert not result.ok
    assert "module/__pycache__" in result.details
    assert "notes.md.bak" in result.details


def test_ignored_live_roots_allowed_but_scratch_rejected(tmp_path: Path) -> None:
    """Ignored live data roots are allowed; unrelated ignored scratch remains a blocker."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / ".gitignore").write_text(".env\ndata/\nruns/\ntmp/\n", encoding="utf-8")

    (tmp_path / ".env").write_text("OPENAI_API_KEY=local\n", encoding="utf-8")
    (tmp_path / "data" / "toolenv" / "tools").mkdir(parents=True)
    (tmp_path / "data" / "toolenv" / "tools" / "tool.json").write_text("{}", encoding="utf-8")
    (tmp_path / "runs" / "stb_server").mkdir(parents=True)
    (tmp_path / "runs" / "stb_server" / "server.log").write_text("log\n", encoding="utf-8")
    (tmp_path / "tmp").mkdir()
    (tmp_path / "tmp" / "scratch.md").write_text("scratch\n", encoding="utf-8")

    result = hygiene.check_ignored(
        tmp_path,
        allowed_roots=[".env", "data/toolenv/tools", "runs"],
    )

    assert not result.ok
    assert result.details == ["tmp/scratch.md"]


def test_live_prereqs_missing_are_advisory_without_values(tmp_path: Path, monkeypatch) -> None:
    """Missing live E2E setup should be reported without exposing secret values."""
    for key in (
        "OPENAI_API_KEY",
        "OPENAI_KEY",
        "TOOLBENCH_KEY",
        "TOOL_ROOT_DIR",
        "CORPUS_BASE_DIR",
        "FITTEXT_RUNTIME_ROOT",
    ):
        monkeypatch.delenv(key, raising=False)

    env_file = tmp_path / ".env"
    result = hygiene.check_live_prereqs(tmp_path, env_file)

    assert result.advisory
    assert not result.ok
    assert any("OPENAI_API_KEY: missing or placeholder" == detail for detail in result.details)
    assert not any("sk-" in detail for detail in result.details)


def test_live_prereqs_pass_with_minimal_local_layout_and_export_env(tmp_path: Path, monkeypatch) -> None:
    """Strict mode can pass with export-style env and OPENAI_API_KEY judge fallback."""
    for key in (
        "OPENAI_API_KEY",
        "OPENAI_KEY",
        "TOOLBENCH_KEY",
        "TOOL_ROOT_DIR",
        "CORPUS_BASE_DIR",
        "FITTEXT_RUNTIME_ROOT",
    ):
        monkeypatch.delenv(key, raising=False)

    tool_root = tmp_path / "data" / "toolenv" / "tools"
    tool_root.mkdir(parents=True)
    (tool_root / "tool.json").write_text("{}", encoding="utf-8")

    corpus_base = tmp_path / "data" / "retrieval" / "StableToolBench"
    for group in ("G1", "G2", "G3"):
        group_dir = corpus_base / group
        group_dir.mkdir(parents=True)
        (group_dir / "des_corpus.json").write_text("[]", encoding="utf-8")
    query_dir = corpus_base.parent / "test_instruction"
    query_dir.mkdir(parents=True)
    (query_dir / "G1_instruction.json").write_text("[]", encoding="utf-8")

    runtime_root = tmp_path / "runs"
    cache_dir = runtime_root / "stb_server" / "tool_response_cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "entry.json").write_text("{}", encoding="utf-8")

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "export OPENAI_API_KEY=test-openai-key",
                "export TOOLBENCH_KEY=test-toolbench-key",
                "TOOL_ROOT_DIR=data/toolenv/tools",
                "CORPUS_BASE_DIR=data/retrieval/StableToolBench",
                "FITTEXT_RUNTIME_ROOT=runs",
            ]
        ),
        encoding="utf-8",
    )

    result = hygiene.check_live_prereqs(tmp_path, env_file)

    assert result.ok
    assert result.details == []


def test_e2e_evidence_missing_is_advisory(tmp_path: Path, monkeypatch) -> None:
    """Live E2E evidence is reported separately from static source hygiene."""
    monkeypatch.delenv("FITTEXT_RUNTIME_ROOT", raising=False)

    env_file = tmp_path / ".env"
    result = hygiene.check_e2e_evidence(tmp_path, env_file)

    assert result.advisory
    assert not result.ok
    assert any("stb/raw_output" in detail for detail in result.details)
    assert any("pass_rate_results" in detail for detail in result.details)


def test_e2e_evidence_passes_with_runtime_outputs(tmp_path: Path, monkeypatch) -> None:
    """Completed raw, converted, and pass-rate outputs satisfy the E2E gate."""
    monkeypatch.delenv("FITTEXT_RUNTIME_ROOT", raising=False)
    runtime_root = tmp_path / "runs"
    _write_minimal_e2e_evidence(runtime_root)

    env_file = tmp_path / ".env"
    env_file.write_text("FITTEXT_RUNTIME_ROOT=runs\n", encoding="utf-8")

    result = hygiene.check_e2e_evidence(tmp_path, env_file)

    assert result.ok
    assert result.details == []


def test_cli_strict_mode_blocks_placeholder_live_prereqs(tmp_path: Path) -> None:
    """Strict release audit must fail when env values are placeholders."""
    _remove_known_test_artifacts()
    env_file = tmp_path / "placeholder.env"
    env_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=REPLACE_WITH_OPENAI_API_KEY",
                "OPENAI_KEY=REPLACE_WITH_OPENAI_API_KEY",
                "TOOLBENCH_KEY=REPLACE_ME",
                "TOOL_ROOT_DIR=data/toolenv/tools",
                "CORPUS_BASE_DIR=data/retrieval/StableToolBench",
                "FITTEXT_RUNTIME_ROOT=runs",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "check_release_hygiene.py"),
            "--env-file",
            str(env_file),
            "--require-live-prereqs",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "FAIL: live prerequisites" in result.stdout


def test_cli_strict_mode_passes_with_ignored_live_layout(tmp_path: Path) -> None:
    """Strict release audit can pass with ignored live data/runtime roots present."""
    _remove_known_test_artifacts()
    live_root = REPO_ROOT / "tmp" / "fittext_release_hygiene_live_test"
    if live_root.exists():
        shutil.rmtree(live_root)

    try:
        tool_root = live_root / "toolenv" / "tools"
        tool_root.mkdir(parents=True)
        (tool_root / "tool.json").write_text("{}", encoding="utf-8")

        corpus_base = live_root / "retrieval" / "StableToolBench"
        for group in ("G1", "G2", "G3"):
            group_dir = corpus_base / group
            group_dir.mkdir(parents=True)
            (group_dir / "des_corpus.json").write_text("[]", encoding="utf-8")
        query_dir = corpus_base.parent / "test_instruction"
        query_dir.mkdir(parents=True)
        (query_dir / "G1_instruction.json").write_text("[]", encoding="utf-8")

        runtime_root = live_root / "runs"
        cache_dir = runtime_root / "stb_server" / "tool_response_cache"
        cache_dir.mkdir(parents=True)
        (cache_dir / "entry.json").write_text("{}", encoding="utf-8")
        _write_minimal_e2e_evidence(runtime_root)

        env_file = tmp_path / "release.env"
        env_file.write_text(
            "\n".join(
                [
                    "OPENAI_API_KEY=test-openai-key",
                    "TOOLBENCH_KEY=test-toolbench-key",
                    "TOOL_ROOT_DIR=tmp/fittext_release_hygiene_live_test/toolenv/tools",
                    "CORPUS_BASE_DIR=tmp/fittext_release_hygiene_live_test/retrieval/StableToolBench",
                    "FITTEXT_RUNTIME_ROOT=tmp/fittext_release_hygiene_live_test/runs",
                ]
            ),
            encoding="utf-8",
        )

        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "check_release_hygiene.py"),
                "--env-file",
                str(env_file),
                "--require-live-prereqs",
                "--require-e2e-evidence",
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
        )

        assert result.returncode == 0, result.stdout + result.stderr
        assert "Release audit passed." in result.stdout
    finally:
        shutil.rmtree(live_root, ignore_errors=True)


def test_cli_requires_e2e_evidence_blocks_missing_outputs(tmp_path: Path) -> None:
    """Release mode must fail when no live pass-rate artifacts exist."""
    _remove_known_test_artifacts()
    runtime_root = REPO_ROOT / "tmp" / "fittext_release_hygiene_missing_e2e"
    shutil.rmtree(runtime_root, ignore_errors=True)
    runtime_root.mkdir(parents=True)

    try:
        env_file = tmp_path / "release.env"
        env_file.write_text(
            "FITTEXT_RUNTIME_ROOT=tmp/fittext_release_hygiene_missing_e2e\n",
            encoding="utf-8",
        )

        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "check_release_hygiene.py"),
                "--env-file",
                str(env_file),
                "--require-e2e-evidence",
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
        )

        assert result.returncode == 1
        assert "FAIL: live E2E evidence" in result.stdout
    finally:
        shutil.rmtree(runtime_root, ignore_errors=True)


def test_publish_target_accepts_fittext_remote(tmp_path: Path) -> None:
    """Publish target check accepts a checkout named FitText with a FitText remote."""
    repo = tmp_path / "FitText"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "git@github.com:example/FitText.git"],
        cwd=repo,
        check=True,
        capture_output=True,
    )

    result = hygiene.check_publish_target(repo, "FitText")

    assert result.ok
    assert result.details == []


def test_publish_target_rejects_old_project_remote(tmp_path: Path) -> None:
    """Publish target check rejects remotes that still point at the old project."""
    repo = tmp_path / "FitText"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    old_name = "Tool" + "-Retrieval"
    subprocess.run(
        ["git", "remote", "add", "origin", f"git@github.com:example/{old_name}.git"],
        cwd=repo,
        check=True,
        capture_output=True,
    )

    result = hygiene.check_publish_target(repo, "FitText")

    assert not result.ok
    assert any("old project" in detail for detail in result.details)


def test_publish_target_rejects_wrong_checkout_name(tmp_path: Path) -> None:
    """Publish target check requires the final checkout directory to be FitText."""
    repo = tmp_path / "WrongName"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "git@github.com:example/FitText.git"],
        cwd=repo,
        check=True,
        capture_output=True,
    )

    result = hygiene.check_publish_target(repo, "FitText")

    assert not result.ok
    assert any("checkout directory" in detail for detail in result.details)


def test_publish_target_rejects_dirty_checkout(tmp_path: Path) -> None:
    """Publish target check requires committed source, not a dirty checkout."""
    repo = tmp_path / "FitText"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "git@github.com:example/FitText.git"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    (repo / "dirty.txt").write_text("dirty\n", encoding="utf-8")

    result = hygiene.check_publish_target(repo, "FitText")

    assert not result.ok
    assert "working tree is not clean" in result.details
