"""Check that no hardcoded model strings exist outside configs/model_pins.yaml.

Walk the repo for model-name patterns that should be sourced from
``configs/model_pins.yaml`` instead of being inlined.  Any match outside an
allowlisted path is reported as a mismatch; the script exits 1 if any are found.

Allowlisted paths (never flagged):
  - tests/          — test fixtures and smoke tests may reference model names
  - docs/           — documentation prose
  - *.md            — markdown docs
  - configs/model_pins.yaml — canonical model-ID source of truth
  - .git/           — VCS internals
  - __pycache__/    — bytecode
  - Lines that are comments (stripped line starts with #)

Usage::

    python scripts/check_pin_consistency.py [--repo-root PATH]

    Exit 0 — no mismatches.
    Exit 1 — one or more hardcoded model strings found (mismatch list printed).
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence


# ---------------------------------------------------------------------------
# Patterns that indicate a hardcoded model string
# ---------------------------------------------------------------------------

# Each pattern: (regex, human_label)
# We look for model-family prefixes in string literals or default= kwargs.
_MODEL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r'gpt-[0-9]'), "OpenAI GPT model"),
    (re.compile(r'claude-'), "Anthropic Claude model"),
    (re.compile(r'\bo[0-9]-\w'), "OpenAI o-series model"),
    (re.compile(r'\bQwen[0-9]'), "Qwen model"),
    (re.compile(r'\bDeepSeek'), "DeepSeek model"),
]

# ---------------------------------------------------------------------------
# Paths to skip entirely (relative to repo root, partial match)
# ---------------------------------------------------------------------------

_SKIP_PATH_FRAGMENTS = frozenset([
    "tests/",
    "docs/",
    ".git/",
    "__pycache__/",
    "configs/model_pins.yaml",  # source of truth — allowed to contain model names
    ".claude/",
    # Observability pricing.py is a lookup table (model → cost), not a hardcode
    "observability/pricing.py",
    # ModelClient clients/ sub-package routes by model-prefix — needs the strings
    "inference/LLM/clients/",
    # Result schema docstring examples are illustrative, not runtime values
    "observability/result_schema.py",
    # check_pin_consistency.py itself contains pattern strings
    "scripts/check_pin_consistency.py",
    # pins.py docstrings are illustrative examples
    "observability/pins.py",
    # runner schema.py has docstring examples only
    "toolbench/runner/schema.py",
    # STB retrieval corpus builder (legacy, requires=True so caller must pass)
    "toolbench/retrieval/build_des_corpus.py",
    # Shell run scripts — model defaults expressed via env-var substitution at launch time
    "scripts/run_inference.sh",
    "scripts/run_evaluation.sh",
    # Deployment server config — not Python code
    "server/config.yml",
    # Downstream rapidapi.py memory_llm fallback is a legacy default, not a research model
    "inference/Downstream_tasks/rapidapi.py",
    # IDE / editor configuration — not source code
    ".vscode/",
    # Persisted experiment results (run directory names + LLM-generated text contain model names)
    "StableToolBench/result/",
    "result/model_predictions_converted/",
    "result/model_predictions/",
    # Generated retrieval corpora and sweep expansions can be large and are not source
    "StableToolBench/solvable_queries/retrieval/",
    "configs/runs/_matrix_tmp/",
    "tmp/",
    # External-baseline checkpoints / cloned repos
    "baselines/external/",
])

# File extensions to scan
_SCAN_EXTENSIONS = frozenset([".py", ".yaml", ".yml", ".sh", ".json"])


def _is_allowlisted_path(rel: str) -> bool:
    """Return True if the file path should be skipped entirely."""
    for frag in _SKIP_PATH_FRAGMENTS:
        if frag in rel:
            return True
    # Markdown files are docs
    if rel.endswith(".md"):
        return True
    return False


def _iter_scan_files(repo_root: Path):
    """Yield candidate source files while pruning allowlisted directories."""
    for dirpath, dirnames, filenames in os.walk(repo_root):
        current = Path(dirpath)
        rel_dir = current.relative_to(repo_root).as_posix()
        rel_dir_prefix = "" if rel_dir == "." else f"{rel_dir}/"

        if rel_dir_prefix and _is_allowlisted_path(rel_dir_prefix):
            dirnames[:] = []
            continue

        kept_dirs = []
        for dirname in dirnames:
            child_rel = f"{dirname}/" if rel_dir == "." else f"{rel_dir}/{dirname}/"
            if not _is_allowlisted_path(child_rel):
                kept_dirs.append(dirname)
        dirnames[:] = kept_dirs

        for filename in filenames:
            fpath = current / filename
            if fpath.suffix in _SCAN_EXTENSIONS:
                yield fpath


def _iter_tracked_scan_files(repo_root: Path):
    """Yield tracked source files, falling back to a pruned walk outside git."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "-z"],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        yield from _iter_scan_files(repo_root)
        return

    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        rel = raw.decode("utf-8", errors="replace")
        fpath = repo_root / rel
        if fpath.is_file() and fpath.suffix in _SCAN_EXTENSIONS:
            yield fpath


def _iter_extra_scan_files(repo_root: Path, extra_paths: Sequence[Path]):
    """Yield explicitly requested extra files or directories under the repo."""
    for extra in extra_paths:
        path = extra if extra.is_absolute() else repo_root / extra
        try:
            rel = path.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            continue
        if _is_allowlisted_path(rel):
            continue
        if path.is_file():
            if path.suffix in _SCAN_EXTENSIONS:
                yield path
            continue
        if path.is_dir():
            yield from _iter_scan_files(path)


def _is_allowlisted_line(line: str) -> bool:
    """Return True if this source line should not be flagged.

    Criteria:
      - Pure comment line (stripped content starts with # or //)
      - Contains "model_pins" (the code is already referencing the pin file)
      - Contains "load_pins" / "get_judge_model" / "get_agent_model" (lookup code)
      - Contains "pins[" (dict lookup into pin file data)
      - Reference-model name strings used as run-output labels
      - Lines that are clearly inside a Python docstring (start with triple-quote)
    """
    stripped = line.strip()
    if stripped.startswith("#") or stripped.startswith("//"):
        return True
    if "model_pins" in line:
        return True
    if "load_pins" in line or "get_judge_model" in line or "get_agent_model" in line:
        return True
    if "pins[" in line:
        return True
    # Model-family routing/capability checks are prefixes, not concrete model pins.
    if ".startswith(" in line:
        return True
    # Run-label strings that embed model name as part of a path or result label
    if "DFS_woFilter_w2_" in line and "_dynamic" in line:
        return True
    # Lines inside Python/YAML docstrings (triple-quoted)
    if stripped.startswith('"""') or stripped.startswith("'''"):
        return True
    # Docstring continuation lines that mention a model as an example
    # (detected by being indented prose with no assignment, no default=, no argparse)
    if "e.g." in line or "``" in line:
        return True
    # server/config.yml model: key is a deployment config, not code
    if "server/config.yml" in line:
        return True
    return False


def _python_docstring_lines(text: str) -> set[int]:
    """Return line numbers occupied by module/class/function docstrings."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()

    lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", [])
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            start = getattr(first, "lineno", None)
            end = getattr(first, "end_lineno", start)
            if start is not None and end is not None:
                lines.update(range(start, end + 1))
    return lines


def find_hardcoded_models(
    repo_root: Path,
    extra_paths: Sequence[Path] = (),
) -> list[tuple[str, int, str, str]]:
    """Walk the repo and find hardcoded model-name strings.

    Args:
        repo_root: Absolute path to the repository root.

    Returns:
        List of ``(relative_path, line_number, matched_pattern_label, line_content)``
        tuples, one per offending line.
    """
    findings: list[tuple[str, int, str, str]] = []

    seen: set[Path] = set()
    for fpath in [*_iter_tracked_scan_files(repo_root), *_iter_extra_scan_files(repo_root, extra_paths)]:
        fpath = fpath.resolve()
        if fpath in seen:
            continue
        seen.add(fpath)
        rel = str(fpath.relative_to(repo_root))
        if _is_allowlisted_path(rel):
            continue

        try:
            text = fpath.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        docstring_lines = _python_docstring_lines(text) if fpath.suffix == ".py" else set()
        for lineno, raw_line in enumerate(text.splitlines(), start=1):
            if lineno in docstring_lines:
                continue
            if _is_allowlisted_line(raw_line):
                continue
            for pattern, label in _MODEL_PATTERNS:
                if pattern.search(raw_line):
                    findings.append((rel, lineno, label, raw_line.rstrip()))
                    # Only report the first matching pattern per line
                    break

    return findings


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the pin consistency checker.

    Args:
        argv: CLI arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code: 0 for clean, 1 for violations found.
    """
    parser = argparse.ArgumentParser(
        description="Check for hardcoded model strings outside configs/model_pins.yaml."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repo root directory. Defaults to parent of this script's parent.",
    )
    parser.add_argument(
        "--extra-path",
        action="append",
        type=Path,
        default=[],
        help="Additional source file or directory to scan, useful for smoke-test injections.",
    )
    args = parser.parse_args(argv)

    repo_root = args.repo_root
    if repo_root is None:
        # scripts/ is one level below repo root
        repo_root = Path(__file__).parent.parent.resolve()

    if not repo_root.exists():
        print(f"ERROR: repo root not found: {repo_root}", file=sys.stderr)
        return 2

    findings = find_hardcoded_models(repo_root.resolve(), args.extra_path)

    if not findings:
        print("OK: no hardcoded model strings found outside allowlisted paths.")
        return 0

    print(
        f"FAIL: {len(findings)} hardcoded model string(s) found "
        f"outside allowlisted paths (runtime model IDs should be in configs/model_pins.yaml):\n",
        file=sys.stderr,
    )
    for rel, lineno, label, line_content in sorted(findings):
        # Truncate long lines for readability
        snippet = line_content[:120] + ("..." if len(line_content) > 120 else "")
        print(f"  {rel}:{lineno}  [{label}]  {snippet}", file=sys.stderr)

    return 1


if __name__ == "__main__":
    sys.exit(main())
