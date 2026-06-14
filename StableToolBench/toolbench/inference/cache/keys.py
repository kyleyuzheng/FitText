"""
Cache key computation for the response cache.

CacheKey is a frozen dataclass that uniquely identifies an LLM request
deterministically.  ``compute_cache_key`` performs canonicalization:
  - invalid messages (``valid: False``) are stripped
  - object keys in messages and tools are sorted before JSON dump
  - floats are coerced to repr-stable strings so 0.1 == 0.10
  - ``extra`` kwargs are SHA-256 hashed to keep the key compact

Layout produced by ``CacheKey.as_path()``:
    <2-char hex prefix>/<full-key>.json

The prefix is the first two characters of the full cache key hex, which
distributes entries across 256 subdirectories to avoid >100 K files in one
directory on NFS.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# CacheKey
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CacheKey:
    """Frozen, hashable cache key for one LLM request.

    All fields that could affect model output are captured here.  Fields that
    do NOT affect output (e.g. run_id, caller context, wall-clock time) are
    intentionally excluded.

    Attributes
    ----------
    model : str
        Full model identifier.
    messages_hash : str
        SHA-256 hex digest of the canonical-JSON-serialised messages list
        (invalid messages stripped, keys sorted).
    tools_hash : str
        SHA-256 hex digest of the canonical-JSON-serialised tools list
        (keys sorted).  Empty string if no tools were supplied.
    temperature : float
        Sampling temperature.
    seed : int | None
        Reproducibility seed.  ``None`` if the caller did not set one.
    top_p : float | None
        Nucleus sampling parameter.  ``None`` if not set.
    extra_hash : str
        SHA-256 hex digest of any additional deterministic kwargs (e.g.
        ``max_tokens``, ``top_k``) serialised as a sorted-key JSON dict.
        Empty string if no extra kwargs were provided.
    """

    model: str
    messages_hash: str
    tools_hash: str
    temperature: float
    seed: Optional[int]
    top_p: Optional[float]
    extra_hash: str

    # ------------------------------------------------------------------
    # Key → path
    # ------------------------------------------------------------------

    def hex(self) -> str:
        """Return a 64-char hex key derived from all fields.

        The key is a SHA-256 of a deterministic JSON representation of the
        fields, so the output is always exactly 64 hex characters.

        Returns
        -------
        str
            64-character lowercase hex string.
        """
        payload = json.dumps(
            {
                "model": self.model,
                "messages_hash": self.messages_hash,
                "tools_hash": self.tools_hash,
                # Coerce temperature/top_p to repr-stable strings so that
                # 0.1 and 0.10 map to the same hash (both become "0.1").
                "temperature": _float_stable(self.temperature),
                "seed": self.seed,
                "top_p": _float_stable(self.top_p) if self.top_p is not None else None,
                "extra_hash": self.extra_hash,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def as_path(self) -> str:
        """Return the relative cache path for this key.

        Format: ``"<2-char prefix>/<64-char hex>.json"``

        The 2-char prefix is the first two characters of the hex digest and
        distributes entries across up to 256 subdirectories.

        Returns
        -------
        str
            Relative path suitable for joining with the cache root.
        """
        key_hex = self.hex()
        return f"{key_hex[:2]}/{key_hex}.json"


# ---------------------------------------------------------------------------
# compute_cache_key
# ---------------------------------------------------------------------------


def compute_cache_key(
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.0,
    seed: Optional[int] = None,
    top_p: Optional[float] = None,
    **extra: Any,
) -> CacheKey:
    """Compute a deterministic ``CacheKey`` for a chat completion request.

    Canonicalization steps
    ----------------------
    1. **Strip invalid messages**: messages with ``valid: False`` (toolbench
       convention for filtered turns) are excluded before hashing.
    2. **Sort object keys**: all dict keys inside messages and tools are sorted
       recursively so that insertion-order differences don't produce cache
       misses.
    3. **Repr-stable floats**: ``temperature`` and ``top_p`` are coerced via
       ``repr()`` to avoid ``0.1 != 0.10000000000000001`` style collisions.
    4. **Extra kwargs**: all remaining kwargs are sorted and hashed; they are
       not included verbatim in the key to keep it compact.

    Parameters
    ----------
    model : str
        Full model identifier.
    messages : list[dict]
        OpenAI-format conversation history.
    tools : list[dict] | None
        OpenAI-format tool definitions.
    temperature : float
        Sampling temperature.
    seed : int | None
        Reproducibility seed.
    top_p : float | None
        Nucleus sampling parameter.
    **extra
        Any additional deterministic kwargs (e.g. ``max_tokens``,
        ``top_k``).  Non-deterministic kwargs (e.g. ``timeout``) should
        NOT be passed here.

    Returns
    -------
    CacheKey
    """
    # 1 + 2: strip invalid messages and sort keys
    valid_messages = _filter_invalid(messages)
    canonical_messages = _canonical(valid_messages)
    messages_hash = _sha256_json(canonical_messages)

    # Tools
    if tools:
        canonical_tools = _canonical(tools)
        tools_hash = _sha256_json(canonical_tools)
    else:
        tools_hash = ""

    # Extra kwargs
    if extra:
        # Sort to make hash deterministic regardless of kwarg order
        extra_hash = _sha256_json({k: extra[k] for k in sorted(extra)})
    else:
        extra_hash = ""

    return CacheKey(
        model=model,
        messages_hash=messages_hash,
        tools_hash=tools_hash,
        temperature=temperature,
        seed=seed,
        top_p=top_p,
        extra_hash=extra_hash,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _filter_invalid(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Strip messages where ``valid`` is explicitly ``False``.

    Toolbench uses a ``valid: False`` sentinel to mark turns that should be
    excluded from the effective conversation.  The cache key must be based on
    the same filtered view the model actually receives.

    Parameters
    ----------
    messages : list[dict]
        Raw messages list, possibly containing ``valid: False`` entries.

    Returns
    -------
    list[dict]
        Messages with ``valid: False`` entries removed.
    """
    return [m for m in messages if m.get("valid", True) is not False]


def _canonical(obj: Any) -> Any:
    """Recursively sort dict keys for canonical JSON representation.

    Parameters
    ----------
    obj : Any
        Arbitrary JSON-compatible object.

    Returns
    -------
    Any
        Object with all nested dict keys sorted.
    """
    if isinstance(obj, dict):
        return {k: _canonical(v) for k in sorted(obj) for v in [obj[k]]}
    if isinstance(obj, list):
        return [_canonical(item) for item in obj]
    return obj


def _sha256_json(obj: Any) -> str:
    """JSON-serialise *obj* (sorted keys) and return its SHA-256 hex digest.

    Parameters
    ----------
    obj : Any
        JSON-serialisable object.

    Returns
    -------
    str
        64-char lowercase hex SHA-256 digest.
    """
    serialised = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(serialised).hexdigest()


def _float_stable(f: float) -> str:
    """Return a repr-stable string for a float.

    Python's ``repr(0.1)`` is ``'0.1'`` (not ``'0.10000000000000001'``), so
    using ``repr`` avoids false cache misses from floating-point display
    differences.

    Parameters
    ----------
    f : float
        Float value.

    Returns
    -------
    str
        Repr-stable string, e.g. ``"0.1"``.
    """
    return repr(f)
