"""
Manifest JSONL writer and schema for per-API-call provenance records.

Every LLM call made during a run is recorded as one line in a JSONL manifest
file.  The manifest is the **only** source of truth for cost, token counts,
and latency — never scrape log lines.

Public API
----------
ManifestEntry
    Pydantic model that validates and serialises one record.

ManifestWriter
    Context-managed, thread/process-safe JSONL writer.
    Use as::

        with ManifestWriter(run_dir / "manifest.jsonl", run_id=run_id) as mw:
            mw.write(entry)

entry_from_response(response, *, qid, operation, variant, ...) -> ManifestEntry
    Helper consumed by the ``ModelClient`` track to convert a provider
    response into a ``ManifestEntry`` without touching manifest internals.

Integration contract with the modelclient track
-------------------------------------------------
``ModelClient.chat_completion()`` calls ``entry_from_response(...)`` after
each successful (or failed) provider round-trip, then passes the result to
``ManifestWriter.write(entry)``.  The writer is injected at construction time
so tests can substitute a no-op implementation.

Example::

    entry = entry_from_response(
        response=raw_provider_response,
        qid="toolret:code:0042",
        operation="pseudo_tool_gen",
        variant="memetic",
        generation=2,
        run_id=run_id,
        git_commit=git_commit,
        config_hash=config_hash,
        started_at=t0,
    )
    manifest_writer.write(entry)
"""

from __future__ import annotations

import datetime
import fcntl
import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .pricing import compute_cost

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Manifest schema
# ---------------------------------------------------------------------------

OperationLiteral = Literal[
    "pseudo_tool_gen",
    "refine",
    "retrieve_embed",
    "dfsdt_node",
    "judge",
    "tool_sim",
    "other",
]

VariantLiteral = Literal[
    "memetic",
    "multi_turn",
    "scattershot",
    "single_pass",
    "baseline_less_is_more",
    "baseline_reinvoke",
    "baseline_xu2024",
    "baseline_colt",
    "other",
]

ProviderLiteral = Literal["openai", "anthropic", "vllm", "local", "other"]


class ManifestEntry(BaseModel):
    """One record per LLM API call.

    Fields mirror the §5.2 schema exactly.  All monetary values are in USD.
    Token counts are *non-negative integers*.  ``error`` is ``None`` on
    success or a short error string on failure.

    The ``cost_usd`` field is auto-computed from the token counts and the
    pricing table if left at its zero default, but callers may override it
    (e.g. when the provider returns a cost field directly).
    """

    # -- Temporal / run identity -------------------------------------------
    ts: str = Field(
        description="ISO-8601 UTC timestamp of the call, e.g. '2026-05-24T01:23:45.123Z'",
    )
    run_id: str = Field(description="Unique run identifier, e.g. '20260524T012345_a1b2c3d4'")
    git_commit: str = Field(description="Full or abbreviated git SHA at time of the run")
    config_hash: str = Field(description="SHA-256 of the resolved run config YAML")

    # -- Query / operation identity ----------------------------------------
    qid: str = Field(description="Query identifier, e.g. 'toolret:code:0042'")
    operation: OperationLiteral = Field(description="Which part of the pipeline issued this call")
    variant: str = Field(
        description="FitText variant or baseline name, e.g. 'memetic' or 'baseline_colt'"
    )
    generation: int = Field(
        default=0,
        ge=0,
        description="Memetic generation index (0 for non-evolutionary variants)",
    )

    # -- Model / provider --------------------------------------------------
    model: str = Field(description="Full model identifier including dated tag if available")
    provider: ProviderLiteral = Field(description="Provider routing used for this call")

    # -- Token counts ------------------------------------------------------
    input_tokens: int = Field(default=0, ge=0, description="Non-cached prompt tokens")
    cached_input_tokens: int = Field(default=0, ge=0, description="Prompt tokens from provider cache")
    output_tokens: int = Field(default=0, ge=0, description="Completion tokens")

    # -- Performance / cost ------------------------------------------------
    latency_ms: float = Field(default=0.0, ge=0.0, description="Wall-clock time for the API call in ms")
    cost_usd: float = Field(default=0.0, ge=0.0, description="USD cost for this call")

    # -- Reproducibility hashes --------------------------------------------
    request_hash: str = Field(
        default="",
        description="SHA-256 of the canonical request payload (model, messages, tools, temperature, seed, top_p, ...)",
    )
    response_hash: str = Field(
        default="",
        description="SHA-256 of the full provider response payload",
    )

    # -- Cache provenance --------------------------------------------------
    cached: Optional[bool] = Field(
        default=None,
        description=(
            "True if this response was served from the disk cache; "
            "False if a live provider round-trip was made; "
            "None if the cache layer was not active for this call."
        ),
    )

    # -- Reliability -------------------------------------------------------
    retry_count: int = Field(default=0, ge=0, description="Number of retries before success")
    error: str | None = Field(
        default=None,
        description="Short error string on failure; None on success",
    )

    @field_validator("ts")
    @classmethod
    def _validate_ts(cls, v: str) -> str:
        """Require an ISO-8601-like timestamp string."""
        if not v or len(v) < 10:
            raise ValueError(f"ts must be an ISO-8601 timestamp, got: {v!r}")
        return v

    def to_jsonl_line(self) -> str:
        """Serialise to a compact JSON string (no trailing newline)."""
        return self.model_dump_json()

    @classmethod
    def from_jsonl_line(cls, line: str) -> "ManifestEntry":
        """Deserialise from a single JSONL line."""
        return cls.model_validate_json(line.strip())


# ---------------------------------------------------------------------------
# ManifestWriter
# ---------------------------------------------------------------------------

class ManifestWriter:
    """Thread- and process-safe JSONL manifest file writer.

    Writes one ``ManifestEntry`` per line.  Each ``write()`` call flushes to
    the OS page cache immediately; the file handle is fsync'd on ``close()``
    so that a clean shutdown guarantees durability.

    File locking uses ``fcntl.LOCK_EX`` (Linux/POSIX only — suitable for
    shared NFS hosts).  A per-instance Python ``threading.Lock`` serialises write calls
    *within* the same process before acquiring the file lock, reducing
    unnecessary kernel contention.

    Usage::

        with ManifestWriter(run_dir / "manifest.jsonl", run_id="...") as mw:
            mw.write(entry)

    Or long-lived (not recommended — prefer context manager)::

        mw = ManifestWriter(path, run_id)
        mw.open()
        mw.write(entry)
        mw.close()
    """

    def __init__(self, path: Path, run_id: str) -> None:
        """Initialise writer.

        Args:
            path: Absolute path to the manifest JSONL file.  Parent directory
                must exist.
            run_id: Run identifier stamped on every entry written through this
                writer.  Used for sanity-checking after the fact.
        """
        self._path = Path(path)
        self._run_id = run_id
        self._fh: Any = None
        self._lock = threading.Lock()
        self._count = 0

    # -- Context manager ---------------------------------------------------

    def __enter__(self) -> "ManifestWriter":
        self.open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # -- File lifecycle ----------------------------------------------------

    def open(self) -> None:
        """Open the manifest file for appending.

        Creates the file if it does not exist.  Safe to call multiple times
        (subsequent calls are no-ops).
        """
        if self._fh is not None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # 'a' mode: creates if not exists, never truncates
        self._fh = open(self._path, "a", buffering=1, encoding="utf-8")  # line-buffered
        logger.debug("ManifestWriter opened %s (run_id=%s)", self._path, self._run_id)

    def close(self) -> None:
        """Flush, fsync, and close the manifest file."""
        if self._fh is None:
            return
        try:
            self._fh.flush()
            os.fsync(self._fh.fileno())
        finally:
            self._fh.close()
            self._fh = None
        logger.debug(
            "ManifestWriter closed %s — wrote %d entries", self._path, self._count
        )

    # -- Write -------------------------------------------------------------

    def write(self, entry: ManifestEntry) -> None:
        """Atomically append one entry to the manifest.

        Args:
            entry: Validated ``ManifestEntry``.  The caller is responsible for
                populating all required fields before writing.

        Raises:
            RuntimeError: If the writer has not been opened.
        """
        if self._fh is None:
            raise RuntimeError(
                "ManifestWriter.write() called before open(). "
                "Use as a context manager or call open() first."
            )
        line = entry.to_jsonl_line() + "\n"
        encoded = line.encode("utf-8")

        with self._lock:
            # Acquire an exclusive file lock so concurrent *processes* on the
            # same NFS mount do not interleave partial writes.
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
            try:
                self._fh.write(line)
                self._fh.flush()
            finally:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            self._count += 1

    @property
    def path(self) -> Path:
        """Absolute path to the manifest file."""
        return self._path

    @property
    def entries_written(self) -> int:
        """Number of entries written in this session."""
        return self._count


# ---------------------------------------------------------------------------
# entry_from_response helper
# ---------------------------------------------------------------------------

def _sha256(obj: Any) -> str:
    """Return a hex SHA-256 of the JSON-serialised *obj*."""
    serialised = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(serialised).hexdigest()


def _now_utc() -> str:
    """Return the current UTC time as an ISO-8601 string with 'Z' suffix."""
    return (
        datetime.datetime.now(tz=datetime.timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def entry_from_response(
    *,
    # Provider response (provider-native format — dict or object)
    response: Any,
    # Call context
    qid: str,
    operation: OperationLiteral,
    variant: str,
    generation: int = 0,
    # Run identity
    run_id: str,
    git_commit: str,
    config_hash: str,
    # Model / provider
    model: str,
    provider: ProviderLiteral,
    # Timing
    started_at: float,  # time.monotonic() before the API call
    finished_at: float,  # time.monotonic() after the API call
    # Optional — override auto-computed values
    input_tokens: int | None = None,
    cached_input_tokens: int | None = None,
    output_tokens: int | None = None,
    cost_usd: float | None = None,
    request_payload: dict[str, Any] | None = None,
    retry_count: int = 0,
    error: str | None = None,
    ts: str | None = None,
    # Cache provenance — set by the cache layer
    cached: Optional[bool] = None,
) -> ManifestEntry:
    """Build a ``ManifestEntry`` from a raw provider response.

    This is the **integration point** for the modelclient track.  The
    ``ModelClient`` implementation calls this function immediately after
    receiving a response and before returning to the caller.

    The function extracts token usage from the response object using a
    best-effort strategy that handles both OpenAI and Anthropic response
    shapes.  Callers may override any field by passing it explicitly.

    Args:
        response: Raw provider response.  For OpenAI this is an
            ``openai.types.chat.ChatCompletion`` or dict; for Anthropic an
            ``anthropic.types.Message`` or dict.
        qid: Query identifier.
        operation: Pipeline operation that issued this call.
        variant: FitText variant or baseline name.
        generation: Memetic generation index (0 for non-evolutionary ops).
        run_id: Run identifier from the active run config.
        git_commit: Git SHA at runtime (``subprocess`` call or passed in).
        config_hash: SHA-256 of the resolved YAML config.
        model: Full model identifier (dated tag preferred).
        provider: Provider routing used.
        started_at: ``time.monotonic()`` timestamp before the API call.
        finished_at: ``time.monotonic()`` timestamp after the API call.
        input_tokens: Override extracted non-cached prompt token count.
        cached_input_tokens: Override extracted cached prompt token count.
        output_tokens: Override extracted completion token count.
        cost_usd: Override auto-computed cost.
        request_payload: Dict to hash as the request fingerprint.  Pass the
            full ``messages`` + ``tools`` + sampling params dict here.
        retry_count: Number of retries before this response was obtained.
        error: Short error string if the call failed; None on success.
        ts: Override the auto-generated timestamp.
        cached: ``True`` if the response came from the disk cache; ``False``
            if a live provider round-trip was made; ``None`` if the cache
            layer was not active.

    Returns:
        Populated and validated ``ManifestEntry``.
    """
    latency_ms = (finished_at - started_at) * 1000.0

    # -- Token extraction --------------------------------------------------
    # Support both dict (serialised) and object (provider SDK) forms.
    resp_dict: dict[str, Any] = (
        response if isinstance(response, dict) else _try_as_dict(response)
    )

    _in_tok, _cached_tok, _out_tok = _extract_tokens(resp_dict, provider)
    resolved_input = input_tokens if input_tokens is not None else _in_tok
    resolved_cached = cached_input_tokens if cached_input_tokens is not None else _cached_tok
    resolved_output = output_tokens if output_tokens is not None else _out_tok

    # -- Cost --------------------------------------------------------------
    resolved_cost = (
        cost_usd
        if cost_usd is not None
        else compute_cost(model, resolved_input, resolved_cached, resolved_output)
    )

    # -- Hashes ------------------------------------------------------------
    req_hash = _sha256(request_payload) if request_payload else ""
    resp_hash = _sha256(resp_dict)

    return ManifestEntry(
        ts=ts or _now_utc(),
        run_id=run_id,
        git_commit=git_commit,
        config_hash=config_hash,
        qid=qid,
        operation=operation,
        variant=variant,
        generation=generation,
        model=model,
        provider=provider,
        input_tokens=resolved_input,
        cached_input_tokens=resolved_cached,
        output_tokens=resolved_output,
        latency_ms=latency_ms,
        cost_usd=resolved_cost,
        request_hash=req_hash,
        response_hash=resp_hash,
        retry_count=retry_count,
        error=error,
        cached=cached,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _try_as_dict(obj: Any) -> dict[str, Any]:
    """Convert a provider response object to a plain dict, best-effort."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    try:
        return dict(obj)
    except Exception:
        return {}


def _extract_tokens(
    resp: dict[str, Any],
    provider: ProviderLiteral,
) -> tuple[int, int, int]:
    """Extract (input_tokens, cached_input_tokens, output_tokens) from response dict.

    Handles:
    - OpenAI ChatCompletion ``usage`` dict with optional ``prompt_tokens_details``
    - Anthropic Message ``usage`` dict with optional ``cache_read_input_tokens``

    Returns:
        Tuple of (non-cached input tokens, cached input tokens, output tokens).
        All values are zero if the usage block is missing.
    """
    usage = resp.get("usage") or {}

    if provider == "anthropic":
        # Anthropic: input_tokens / cache_read_input_tokens / output_tokens
        raw_input = int(usage.get("input_tokens") or 0)
        cached = int(usage.get("cache_read_input_tokens") or 0)
        output = int(usage.get("output_tokens") or 0)
        # Anthropic reports *total* input including cached; subtract to get non-cached
        non_cached = max(0, raw_input - cached)
        return non_cached, cached, output

    # OpenAI (and vLLM OpenAI-compat): prompt_tokens / cached_tokens / completion_tokens
    prompt = int(usage.get("prompt_tokens") or 0)
    details = usage.get("prompt_tokens_details") or {}
    cached = int(details.get("cached_tokens") or 0)
    non_cached = max(0, prompt - cached)
    completion = int(usage.get("completion_tokens") or 0)
    return non_cached, cached, completion


# ---------------------------------------------------------------------------
# Manifest reader (for aggregator / replay)
# ---------------------------------------------------------------------------

def read_manifest(path: Path) -> list[ManifestEntry]:
    """Read all entries from a JSONL manifest file.

    Args:
        path: Path to the manifest JSONL file.

    Returns:
        List of ``ManifestEntry`` objects in file order.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If any line fails schema validation (includes line number).
    """
    entries: list[ManifestEntry] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(ManifestEntry.from_jsonl_line(line))
            except Exception as exc:
                raise ValueError(
                    f"Manifest parse error at {path}:{lineno}: {exc}"
                ) from exc
    return entries
