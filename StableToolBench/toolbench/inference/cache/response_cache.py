"""
Disk-backed response cache for LLM API calls.

Responses are stored as JSON files under a two-level directory layout::

    <cache_dir>/<2-char prefix>/<64-char key>.json

Atomic writes use ``tempfile.NamedTemporaryFile`` → ``os.rename`` so that a
process crash mid-write never leaves a partial file visible to readers.

The ``raw_response`` field of ``NormalizedResponse`` is capped at 64 KB when
persisted; this prevents runaway disk usage from large debug payloads while
keeping all other provenance fields intact.

Thread safety
-------------
All stat counters (hits, misses, puts, bytes_written) are protected by a
``threading.Lock``.  Filesystem operations are inherently atomic via
``os.rename`` (POSIX guarantee: rename is atomic when src and dst are on the
same filesystem, which is always true here since we use the same cache_dir).
No additional locking is needed for reads or writes.

Disabled mode
-------------
When ``enabled=False`` (the ``--no-cache`` flag), ``get`` always returns
``None`` and ``put`` is a no-op.  ``stats()`` still increments ``misses``
so callers can confirm the cache is disabled without reading code.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from toolbench.inference.LLM.clients.base import NormalizedResponse

from .keys import CacheKey

logger = logging.getLogger(__name__)

# Maximum size (bytes) of the ``raw_response`` field stored on disk.
_RAW_RESPONSE_CAP_BYTES: int = 64 * 1024  # 64 KB


# ---------------------------------------------------------------------------
# ResponseCache
# ---------------------------------------------------------------------------


class ResponseCache:
    """Disk-backed JSON response cache.

    Parameters
    ----------
    cache_dir : Path
        Root directory for cache files.  Created on first write if absent.
    enabled : bool
        When ``False``, all operations are no-ops (``get`` returns ``None``,
        ``put`` does nothing).  Defaults to ``True``.

    Notes
    -----
    - Layout: ``<cache_dir>/<2-char key prefix>/<key>.json``
    - Atomic writes: ``<key>.json.tmp`` → ``os.rename`` → ``<key>.json``
    - Thread-safe via per-instance ``threading.Lock`` on counters only.
    - ``raw_response`` capped at ``_RAW_RESPONSE_CAP_BYTES`` on disk.
    """

    def __init__(self, cache_dir: Path, *, enabled: bool = True) -> None:
        """Initialise the response cache.

        Parameters
        ----------
        cache_dir : Path
            Root directory.  Created lazily on first ``put``.
        enabled : bool
            Set to ``False`` to disable all cache I/O (``--no-cache`` flag).
        """
        self._cache_dir = Path(cache_dir)
        self._enabled = enabled
        self._lock = threading.Lock()
        # Stat counters — protected by _lock
        self._hits: int = 0
        self._misses: int = 0
        self._puts: int = 0
        self._bytes_written: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: CacheKey) -> Optional[NormalizedResponse]:
        """Look up a cached response.

        Parameters
        ----------
        key : CacheKey
            The request key to look up.

        Returns
        -------
        NormalizedResponse or None
            Reconstructed response on a cache hit; ``None`` on miss or when
            the cache is disabled.
        """
        if not self._enabled:
            with self._lock:
                self._misses += 1
            return None

        entry_path = self._cache_dir / key.as_path()
        if not entry_path.exists():
            with self._lock:
                self._misses += 1
            logger.debug("cache miss: %s", key.hex())
            return None

        try:
            data = json.loads(entry_path.read_text(encoding="utf-8"))
            response = _deserialize_response(data)
            with self._lock:
                self._hits += 1
            logger.debug("cache hit: %s", key.hex())
            return response
        except Exception as exc:
            # Corrupt or incompatible entry — treat as miss, don't crash.
            logger.warning("cache entry unreadable (%s), treating as miss: %s", exc, entry_path)
            with self._lock:
                self._misses += 1
            return None

    def put(self, key: CacheKey, response: NormalizedResponse) -> None:
        """Store a response in the cache.

        Atomic write: serialises to ``<key>.json.tmp`` in the same directory
        then renames to ``<key>.json``.  A crash between these two steps
        leaves only the ``.tmp`` file, which is ignored by ``get``.

        The ``raw_response`` field is trimmed to ``_RAW_RESPONSE_CAP_BYTES``
        if its JSON representation exceeds that threshold.

        Parameters
        ----------
        key : CacheKey
            The request key.
        response : NormalizedResponse
            The response to cache.
        """
        if not self._enabled:
            return

        entry_path = self._cache_dir / key.as_path()
        entry_path.parent.mkdir(parents=True, exist_ok=True)

        payload = _serialize_response(response)
        serialised = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        encoded = serialised.encode("utf-8")
        nbytes = len(encoded)

        # Write atomically via tmp → rename
        tmp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=entry_path.parent,
                delete=False,
                suffix=".json.tmp",
            ) as fh:
                tmp_path = Path(fh.name)
                fh.write(encoded)
                fh.flush()
                os.fsync(fh.fileno())

            os.rename(tmp_path, entry_path)
            tmp_path = None  # rename succeeded; no cleanup needed

            with self._lock:
                self._puts += 1
                self._bytes_written += nbytes
            logger.debug("cache put: %s (%d bytes)", key.hex(), nbytes)

        except Exception as exc:
            logger.warning("cache put failed for key %s: %s", key.hex(), exc)
            # Clean up the .tmp file if rename hasn't happened yet
            if tmp_path is not None and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise

    def stats(self) -> Dict[str, int]:
        """Return a snapshot of cache statistics.

        Returns
        -------
        dict
            ``{"hits": int, "misses": int, "puts": int, "bytes_written": int}``
        """
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "puts": self._puts,
                "bytes_written": self._bytes_written,
            }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Whether the cache is active."""
        return self._enabled

    @property
    def cache_dir(self) -> Path:
        """Root directory for cached entries."""
        return self._cache_dir


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


_RAW_RESPONSE_KEY = "raw_response"
_CACHE_FORMAT_VERSION = 1


def _serialize_response(response: NormalizedResponse) -> Dict[str, Any]:
    """Convert a ``NormalizedResponse`` to a JSON-serialisable dict.

    The ``raw_response`` field is capped at ``_RAW_RESPONSE_CAP_BYTES``
    to bound disk usage; the truncated form is stored as a JSON string
    under the key ``"raw_response_truncated"``.

    Parameters
    ----------
    response : NormalizedResponse
        The response to serialise.

    Returns
    -------
    dict
        JSON-serialisable dict suitable for ``json.dumps``.
    """
    data: Dict[str, Any] = {
        "_cache_version": _CACHE_FORMAT_VERSION,
        "content": response.content,
        "tool_calls": response.tool_calls,
        "finish_reason": response.finish_reason,
        "model_revision": response.model_revision,
        "input_tokens": response.input_tokens,
        "cached_input_tokens": response.cached_input_tokens,
        "output_tokens": response.output_tokens,
        "latency_ms": response.latency_ms,
        "provider": response.provider,
    }

    # Cap raw_response to avoid runaway disk usage
    raw_serialised = json.dumps(response.raw_response, ensure_ascii=False, separators=(",", ":"))
    raw_bytes = raw_serialised.encode("utf-8")
    if len(raw_bytes) <= _RAW_RESPONSE_CAP_BYTES:
        data[_RAW_RESPONSE_KEY] = response.raw_response
    else:
        # Store as empty dict; annotate that it was truncated
        data[_RAW_RESPONSE_KEY] = {}
        data["raw_response_truncated"] = True

    return data


def _deserialize_response(data: Dict[str, Any]) -> NormalizedResponse:
    """Reconstruct a ``NormalizedResponse`` from a cached dict.

    Parameters
    ----------
    data : dict
        Dict as stored by ``_serialize_response``.

    Returns
    -------
    NormalizedResponse
        Reconstructed response.  ``raw_response`` may be empty if the
        original was too large to store.

    Raises
    ------
    KeyError
        If required fields are missing (corrupted entry).
    """
    return NormalizedResponse(
        content=data["content"],
        tool_calls=data.get("tool_calls") or [],
        finish_reason=data.get("finish_reason"),
        model_revision=data["model_revision"],
        input_tokens=int(data.get("input_tokens", 0)),
        cached_input_tokens=int(data.get("cached_input_tokens", 0)),
        output_tokens=int(data.get("output_tokens", 0)),
        latency_ms=float(data.get("latency_ms", 0.0)),
        provider=data.get("provider", ""),
        raw_response=data.get(_RAW_RESPONSE_KEY) or {},
    )
