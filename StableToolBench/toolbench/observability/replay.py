"""
Deterministic manifest replay and hash-drift detection.

``replay()`` iterates a manifest file and verifies that:
1. The ``git_commit`` recorded in each entry matches the current HEAD of the
   repo at *repo_root* (unless ``git_commit_check=False``).
2. The ``response_hash`` stored in the entry matches the hash of any cached
   response found in *cache_dir* (optional).

No API calls are made — this is a purely local consistency check.

Usage::

    report = replay(
        manifest_path=Path("results/run_xyz/manifest.jsonl"),
        repo_root=Path("~/FitText"),
        cache_dir=Path("results/run_xyz/cache"),
    )
    print(report)
    if report.hash_drift_count > 0:
        sys.exit(1)
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .manifest import ManifestEntry, read_manifest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ReplayReport
# ---------------------------------------------------------------------------

@dataclass
class ReplayReport:
    """Summary of a replay verification pass.

    Attributes:
        manifest_path: Path to the manifest that was replayed.
        total_entries: Total number of entries in the manifest.
        commit_mismatch_count: Entries whose ``git_commit`` does not match
            the current HEAD.  Non-zero means the run was not on the
            currently checked-out commit.
        hash_drift_count: Entries whose ``response_hash`` could not be
            verified against a cached response (file not found is not
            counted as drift; only hash *mismatch* is).
        missing_cache_count: Entries where no cached response file was found
            (informational only — does not affect ``hash_drift_count``).
        unique_commits: Set of git commits seen in the manifest.
        total_cost_usd: Sum of ``cost_usd`` across all entries.
        errors: List of human-readable error strings.
    """

    manifest_path: Path
    total_entries: int = 0
    commit_mismatch_count: int = 0
    hash_drift_count: int = 0
    missing_cache_count: int = 0
    unique_commits: set[str] = field(default_factory=set)
    total_cost_usd: float = 0.0
    errors: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"Replay report: {self.manifest_path}",
            f"  Entries         : {self.total_entries}",
            f"  Total cost      : ${self.total_cost_usd:.4f}",
            f"  Unique commits  : {', '.join(sorted(self.unique_commits)) or '—'}",
            f"  Commit mismatches: {self.commit_mismatch_count}",
            f"  Hash drift      : {self.hash_drift_count}",
            f"  Missing cache   : {self.missing_cache_count}",
        ]
        if self.errors:
            lines.append("  Errors:")
            for err in self.errors:
                lines.append(f"    - {err}")
        return "\n".join(lines)

    @property
    def ok(self) -> bool:
        """``True`` if no hash drift was detected (commit mismatches are warnings)."""
        return self.hash_drift_count == 0 and not self.errors


# ---------------------------------------------------------------------------
# git HEAD helper
# ---------------------------------------------------------------------------

def _get_current_head(repo_root: Path) -> str | None:
    """Return the full git SHA of HEAD for the repo at *repo_root*.

    Returns ``None`` if the git command fails (e.g. not a git repo).
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as exc:
        logger.warning("git rev-parse failed: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Cache response hash helper
# ---------------------------------------------------------------------------

def _load_cached_response_hash(
    cache_dir: Path, request_hash: str
) -> str | None:
    """Load a cached response and return its SHA-256 hash.

    The cache layout is: ``<cache_dir>/<request_hash[:2]>/<request_hash>.json``
    This mirrors the layout used by ``toolbench/inference/cache/response_cache.py``
    (Wave 2 cache track).  Returns ``None`` if the file does not exist.
    """
    if not request_hash:
        return None
    shard = request_hash[:2]
    cache_file = cache_dir / shard / f"{request_hash}.json"
    if not cache_file.exists():
        return None
    content = cache_file.read_bytes()
    # Re-derive the hash the same way manifest.py does it
    try:
        obj = json.loads(content)
        serialised = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(serialised).hexdigest()
    except json.JSONDecodeError:
        return hashlib.sha256(content).hexdigest()


# ---------------------------------------------------------------------------
# Main replay function
# ---------------------------------------------------------------------------

def replay(
    manifest_path: Path,
    *,
    repo_root: Path | None = None,
    cache_dir: Path | None = None,
    git_commit_check: bool = True,
) -> ReplayReport:
    """Replay a manifest and verify provenance.

    Args:
        manifest_path: Path to the JSONL manifest to verify.
        repo_root: Root of the git repository used to check the current HEAD.
            Defaults to the parent of *manifest_path* traversed up to the
            nearest ``.git`` directory.
        cache_dir: Directory containing cached response files.  When provided,
            ``response_hash`` fields are verified against cached content.
        git_commit_check: If ``False``, skip the git HEAD comparison (useful
            in CI where HEAD may legitimately differ).

    Returns:
        ``ReplayReport`` with counts and a human-readable ``__str__``.
    """
    report = ReplayReport(manifest_path=manifest_path)

    # -- Read manifest -----------------------------------------------------
    try:
        entries: list[ManifestEntry] = read_manifest(manifest_path)
    except Exception as exc:
        report.errors.append(f"Failed to read manifest: {exc}")
        return report

    report.total_entries = len(entries)

    # -- Resolve repo root -------------------------------------------------
    if repo_root is None:
        repo_root = _find_repo_root(manifest_path)

    current_head: str | None = None
    if git_commit_check and repo_root is not None:
        current_head = _get_current_head(repo_root)
        if current_head is None:
            logger.warning(
                "Could not determine current git HEAD at %s — "
                "commit mismatch check skipped.",
                repo_root,
            )

    # -- Iterate entries ---------------------------------------------------
    for entry in entries:
        report.total_cost_usd += entry.cost_usd
        report.unique_commits.add(entry.git_commit)

        # Commit mismatch check
        if (
            git_commit_check
            and current_head is not None
            and entry.git_commit
            and not current_head.startswith(entry.git_commit)
            and not entry.git_commit.startswith(current_head)
        ):
            report.commit_mismatch_count += 1
            logger.debug(
                "Commit mismatch: entry=%s current=%s (qid=%s)",
                entry.git_commit,
                current_head,
                entry.qid,
            )

        # Response hash check (requires cache)
        if cache_dir is not None and entry.response_hash:
            if not entry.request_hash:
                # Cannot look up without request hash
                report.missing_cache_count += 1
                continue
            cached_hash = _load_cached_response_hash(cache_dir, entry.request_hash)
            if cached_hash is None:
                report.missing_cache_count += 1
            elif cached_hash != entry.response_hash:
                report.hash_drift_count += 1
                report.errors.append(
                    f"Hash drift: qid={entry.qid} expected={entry.response_hash[:16]}… "
                    f"got={cached_hash[:16]}…"
                )

    if report.commit_mismatch_count > 0:
        logger.warning(
            "%d entries have a git commit that differs from current HEAD (%s). "
            "This means the manifest was produced at a different checkout.",
            report.commit_mismatch_count,
            current_head,
        )

    return report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_repo_root(start: Path) -> Path | None:
    """Walk upward from *start* to find the nearest ``.git`` directory."""
    current = start.resolve()
    for parent in [current, *current.parents]:
        if (parent / ".git").exists():
            return parent
    return None
