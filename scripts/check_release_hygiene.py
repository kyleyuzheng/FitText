#!/usr/bin/env python3
"""Audit the FitText release tree before publishing.

This script checks the source-only release invariants that should hold before
creating or pushing the public FitText repository. It does not print secret
values. Live end-to-end prerequisites are reported by default and become
release-blocking with ``--require-live-prereqs``.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


GENERATED_EXTENSIONS = {
    ".csv",
    ".db",
    ".jsonl",
    ".npy",
    ".npz",
    ".parquet",
    ".pkl",
    ".pt",
    ".sqlite",
}

ARTIFACT_NAMES = {
    ".pytest_cache",
    "__pycache__",
}

ARTIFACT_SUFFIXES = (
    ".bak",
    ".old",
    ".tmp",
)

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{10,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
]

STALE_PATTERNS = [
    ("old project name", re.compile(r"Tool[-_]Retrieval")),
    ("phase2 glue", re.compile(r"\bphase2\b|Phase 2")),
    ("memetic_v2 archive", re.compile(r"Memetic V2|memetic_v2|_archived")),
    ("old placeholders", re.compile(r"your_model_path|your_retrival|your_tools_path|your_lora_path|/path/to/toolenv|/path/to/retrieval")),
    ("old upstream label", re.compile(r"ToolBench/ToolLLaMA|ToolBench-IR")),
    ("unfinished pricing note", re.compile(r"TODO confirm")),
    ("model-specific evaluator dir", re.compile(r"tooleval_gpt")),
]

TEXT_EXTENSIONS = {
    ".cfg",
    ".env",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

SKIP_SCAN_PREFIXES = (
    ".git/",
    "StableToolBench/solvable_queries/",
)

SELF_PATH = "scripts/check_release_hygiene.py"


@dataclass
class CheckResult:
    name: str
    ok: bool
    details: list[str]
    advisory: bool = False


def _run(cmd: Sequence[str], repo_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(cmd),
        cwd=repo_root,
        text=True,
        capture_output=True,
        timeout=60,
    )


def _git_lines(repo_root: Path, args: Sequence[str]) -> list[str]:
    proc = _run(["git", *args], repo_root)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return [line for line in proc.stdout.splitlines() if line]


def _tracked_files(repo_root: Path) -> list[str]:
    return _git_lines(repo_root, ["ls-files"])


def _is_text_candidate(rel: str) -> bool:
    if rel == SELF_PATH:
        return False
    if any(rel.startswith(prefix) for prefix in SKIP_SCAN_PREFIXES):
        return False
    return Path(rel).suffix in TEXT_EXTENSIONS or Path(rel).name == ".env.example"


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def check_stale_strings(repo_root: Path, tracked: Sequence[str]) -> CheckResult:
    findings: list[str] = []
    for rel in tracked:
        if not _is_text_candidate(rel):
            continue
        text = _read_text(repo_root / rel)
        if text is None:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for label, pattern in STALE_PATTERNS:
                if pattern.search(line):
                    findings.append(f"{rel}:{lineno}: {label}")
                    break
    return CheckResult("stale strings / glue terms", not findings, findings[:80])


def check_secret_shapes(repo_root: Path, tracked: Sequence[str]) -> CheckResult:
    findings: list[str] = []
    for rel in tracked:
        if not _is_text_candidate(rel):
            continue
        text = _read_text(repo_root / rel)
        if text is None:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if any(pattern.search(line) for pattern in SECRET_PATTERNS):
                findings.append(f"{rel}:{lineno}: secret-shaped token")
    return CheckResult("secret-shaped tokens", not findings, findings[:80])


def check_tracked_generated_files(tracked: Sequence[str]) -> CheckResult:
    findings = [rel for rel in tracked if Path(rel).suffix in GENERATED_EXTENSIONS]
    return CheckResult("tracked generated/data extensions", not findings, findings[:80])


def check_local_artifacts(repo_root: Path) -> CheckResult:
    findings: list[str] = []
    for path in repo_root.rglob("*"):
        try:
            rel = path.relative_to(repo_root).as_posix()
        except ValueError:
            continue
        if rel.startswith(".git/") or rel == ".git":
            continue
        name = path.name
        if name in ARTIFACT_NAMES or name.endswith(ARTIFACT_SUFFIXES) or ".bak." in name:
            findings.append(rel)
    return CheckResult("local artifact files/directories", not findings, sorted(findings)[:80])


def check_untracked(repo_root: Path) -> CheckResult:
    findings = _git_lines(repo_root, ["ls-files", "--others", "--exclude-standard"])
    return CheckResult("untracked files", not findings, findings[:80])


def _relative_repo_path(repo_root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return None


def _is_allowed_ignored(rel: str, allowed_roots: Sequence[str]) -> bool:
    rel = rel.rstrip("/")
    for root in allowed_roots:
        root = root.rstrip("/")
        if rel == root or rel.startswith(f"{root}/"):
            return True
    return False


def _allowed_ignored_roots(repo_root: Path, env_file: Path) -> list[str]:
    env = _merged_env(env_file)
    roots = [".env"]
    for key in ("TOOL_ROOT_DIR", "CORPUS_BASE_DIR", "FITTEXT_RUNTIME_ROOT"):
        path = _repo_path(repo_root, env.get(key))
        rel = _relative_repo_path(repo_root, path)
        if rel:
            roots.append(rel)
        if key == "CORPUS_BASE_DIR" and path is not None:
            query_rel = _relative_repo_path(repo_root, path.parent / "test_instruction")
            if query_rel:
                roots.append(query_rel)
    return sorted(set(roots))


def check_ignored(repo_root: Path, allowed_roots: Sequence[str] = ()) -> CheckResult:
    raw_findings = _git_lines(repo_root, ["ls-files", "--others", "--ignored", "--exclude-standard"])
    findings = [rel for rel in raw_findings if not _is_allowed_ignored(rel, allowed_roots)]
    return CheckResult("unexpected ignored files in worktree", not findings, findings[:80])


def check_diff_whitespace(repo_root: Path) -> CheckResult:
    proc = _run(["git", "diff", "--check"], repo_root)
    details = [line for line in (proc.stdout + proc.stderr).splitlines() if line]
    return CheckResult("git diff --check", proc.returncode == 0, details[:80])


def check_pin_consistency(repo_root: Path) -> CheckResult:
    proc = _run(
        [sys.executable, "scripts/check_pin_consistency.py", "--repo-root", str(repo_root)],
        repo_root,
    )
    details = [line for line in (proc.stdout + proc.stderr).splitlines() if line]
    return CheckResult("model pin consistency", proc.returncode == 0, details[:80])


def _remote_repo_name(remote_url: str) -> str:
    pathish = remote_url.rstrip("/").split(":")[-1]
    name = pathish.rsplit("/", 1)[-1]
    return name.removesuffix(".git")


def check_publish_target(repo_root: Path, expected_name: str) -> CheckResult:
    """Verify this checkout is aimed at the public FitText repository."""
    findings: list[str] = []
    dirty = _git_lines(repo_root, ["status", "--porcelain"])
    if dirty:
        findings.append("working tree is not clean")

    if repo_root.name != expected_name:
        findings.append(f"checkout directory is {repo_root.name!r}, expected {expected_name!r}")

    remote_lines = _git_lines(repo_root, ["remote", "-v"])
    old_names = ("Tool" + "-Retrieval", "Tool" + "_Retrieval")
    current_remote_names: list[str] = []
    old_remote_names: set[str] = set()
    for line in remote_lines:
        parts = line.split()
        if len(parts) < 2:
            continue
        remote_name, remote_url = parts[0], parts[1]
        if any(old_name in remote_url for old_name in old_names):
            old_remote_names.add(remote_name)
        current_remote_names.append(_remote_repo_name(remote_url))

    for remote_name in sorted(old_remote_names):
        findings.append(f"remote {remote_name!r} still points at the old project")

    if not current_remote_names:
        findings.append("no git remote configured for the publish target")
    elif expected_name not in current_remote_names:
        findings.append(f"no git remote points at {expected_name!r}")

    return CheckResult("publish target", not findings, findings[:80])


def _load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key.removeprefix("export ").strip()
        val = val.strip().strip('"').strip("'")
        if key:
            values[key] = val
    return values


def _merged_env(env_file: Path) -> dict[str, str]:
    merged = dict(os.environ)
    merged.update(_load_env_file(env_file))
    return merged


def _real_env_value(value: str | None) -> bool:
    if not value:
        return False
    sentinels = (
        "REPLACE_ME",
        "REPLACE_WITH_",
        "DRY_RUN_",
        "NOT_SET",
    )
    return not any(token in value for token in sentinels)


def _repo_path(repo_root: Path, value: str | None, default: str | None = None) -> Path | None:
    raw = value if _real_env_value(value) else default
    if not raw:
        return None
    path = Path(os.path.expandvars(raw)).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path


def _has_files(path: Path | None) -> bool:
    return bool(path and path.exists() and any(path.rglob("*")))


def check_live_prereqs(repo_root: Path, env_file: Path) -> CheckResult:
    env = _merged_env(env_file)
    findings: list[str] = []

    for key in ("OPENAI_API_KEY", "TOOLBENCH_KEY"):
        if not _real_env_value(env.get(key)):
            findings.append(f"{key}: missing or placeholder")
    if not (_real_env_value(env.get("OPENAI_KEY")) or _real_env_value(env.get("OPENAI_API_KEY"))):
        findings.append("OPENAI_KEY or OPENAI_API_KEY: missing or placeholder")

    tool_root = _repo_path(repo_root, env.get("TOOL_ROOT_DIR"))
    if not tool_root or not tool_root.is_dir():
        findings.append("TOOL_ROOT_DIR: missing directory")
    elif not any(tool_root.rglob("*.json")):
        findings.append("TOOL_ROOT_DIR: no tool JSON files found")

    corpus_base = _repo_path(repo_root, env.get("CORPUS_BASE_DIR"))
    if not corpus_base or not corpus_base.is_dir():
        findings.append("CORPUS_BASE_DIR: missing directory")
    else:
        for group in ("G1", "G2", "G3"):
            if not (corpus_base / group / "des_corpus.json").is_file():
                findings.append(f"CORPUS_BASE_DIR/{group}/des_corpus.json: missing")
        if not (corpus_base.parent / "test_instruction" / "G1_instruction.json").is_file():
            findings.append("CORPUS_BASE_DIR/../test_instruction/G1_instruction.json: missing")

    runtime_root = _repo_path(repo_root, env.get("FITTEXT_RUNTIME_ROOT"), "runs")
    cache_root = runtime_root / "stb_server" / "tool_response_cache" if runtime_root else None
    if not _has_files(cache_root):
        findings.append("FITTEXT_RUNTIME_ROOT/stb_server/tool_response_cache: missing or empty")

    note = "live prerequisites"
    if not env_file.exists():
        findings.insert(0, f"{env_file.name}: not present; only process environment was checked")
    return CheckResult(note, not findings, findings, advisory=True)


def _has_named_file(path: Path | None, filename: str) -> bool:
    return bool(
        path and path.exists() and any(item.name == filename for item in path.rglob(filename))
    )


def _has_any_file_with_suffix(path: Path | None, suffixes: Sequence[str]) -> bool:
    if not path or not path.exists():
        return False
    return any(item.is_file() and item.suffix in suffixes for item in path.rglob("*"))


def check_e2e_evidence(repo_root: Path, env_file: Path) -> CheckResult:
    """Check for artifacts proving a live StableToolBench pass-rate run completed."""
    env = _merged_env(env_file)
    findings: list[str] = []
    runtime_root = _repo_path(repo_root, env.get("FITTEXT_RUNTIME_ROOT"), "runs")
    if runtime_root is None:
        findings.append("FITTEXT_RUNTIME_ROOT: missing or placeholder")
    else:
        raw_root = runtime_root / "stb" / "raw_output"
        converted_root = runtime_root / "stb" / "model_predictions_converted"
        pass_rate_root = runtime_root / "stb" / "pass_rate_results"

        if not _has_files(raw_root):
            findings.append("FITTEXT_RUNTIME_ROOT/stb/raw_output: missing or empty")
        if not _has_any_file_with_suffix(converted_root, (".json",)):
            findings.append(
                "FITTEXT_RUNTIME_ROOT/stb/model_predictions_converted: no converted JSON files found"
            )
        if not _has_named_file(pass_rate_root, ".done"):
            findings.append("FITTEXT_RUNTIME_ROOT/stb/pass_rate_results: no completed evaluation marker found")
        if not _has_any_file_with_suffix(pass_rate_root, (".csv", ".json")):
            findings.append("FITTEXT_RUNTIME_ROOT/stb/pass_rate_results: no pass-rate result files found")

    return CheckResult("live E2E evidence", not findings, findings, advisory=True)


def _print_result(result: CheckResult, *, fail_advisory: bool) -> None:
    if result.ok:
        prefix = "PASS"
    elif result.advisory and not fail_advisory:
        prefix = "WARN"
    else:
        prefix = "FAIL"
    print(f"{prefix}: {result.name}")
    for detail in result.details:
        print(f"  - {detail}")


def run_audit(
    repo_root: Path,
    env_file: Path,
    require_live_prereqs: bool,
    require_e2e_evidence: bool,
) -> int:
    tracked = _tracked_files(repo_root)
    allowed_ignored_roots = _allowed_ignored_roots(repo_root, env_file)
    checks = [
        check_stale_strings(repo_root, tracked),
        check_secret_shapes(repo_root, tracked),
        check_tracked_generated_files(tracked),
        check_local_artifacts(repo_root),
        check_untracked(repo_root),
        check_ignored(repo_root, allowed_ignored_roots),
        check_diff_whitespace(repo_root),
        check_pin_consistency(repo_root),
        check_live_prereqs(repo_root, env_file),
        check_e2e_evidence(repo_root, env_file),
    ]

    failed = False
    advisory_warnings = False
    for check in checks:
        fail_advisory = (
            check.name == "live prerequisites" and require_live_prereqs
        ) or (
            check.name == "live E2E evidence" and require_e2e_evidence
        )
        _print_result(check, fail_advisory=fail_advisory)
        if check.advisory and not check.ok:
            advisory_warnings = True
        if not check.ok and (not check.advisory or fail_advisory):
            failed = True

    if failed:
        print("\nRelease audit FAILED.")
        return 1
    if advisory_warnings:
        print("\nRelease hygiene audit passed; advisory release gates remain incomplete.")
        return 0
    print("\nRelease audit passed.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check FitText release hygiene.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root. Defaults to the parent of scripts/.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Optional env file to inspect for live-run prerequisites without printing values.",
    )
    parser.add_argument(
        "--require-live-prereqs",
        action="store_true",
        help="Fail if credentials, data paths, or response cache needed for live E2E are missing.",
    )
    parser.add_argument(
        "--require-e2e-evidence",
        action="store_true",
        help="Fail unless live StableToolBench raw, converted, and pass-rate outputs are present.",
    )
    parser.add_argument(
        "--publish-target-name",
        default="",
        help="Also require this checkout and its git remote to target the named public repository.",
    )
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    env_file = args.env_file
    if not env_file.is_absolute():
        env_file = repo_root / env_file
    rc = run_audit(
        repo_root,
        env_file,
        args.require_live_prereqs,
        args.require_e2e_evidence,
    )
    if args.publish_target_name:
        publish_result = check_publish_target(repo_root, args.publish_target_name)
        _print_result(publish_result, fail_advisory=True)
        if not publish_result.ok:
            print("\nPublish target audit FAILED.")
            return 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
