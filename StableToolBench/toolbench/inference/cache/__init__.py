"""
toolbench.inference.cache — disk-backed LLM response cache.

Public API
----------
ResponseCache
    Disk-backed JSON cache.  ``get(key)`` / ``put(key, response)`` / ``stats()``.

CacheKey
    Frozen dataclass representing a unique, deterministic request fingerprint.

compute_cache_key
    Factory that canonicalises messages, tools, and sampling params into a
    ``CacheKey``.

Usage example::

    from pathlib import Path
    from toolbench.inference.cache import ResponseCache, compute_cache_key

    cache = ResponseCache(Path("runs/my_run/cache"))
    key = compute_cache_key(model, messages, tools, temperature=0.0, seed=42)
    cached = cache.get(key)
    if cached is None:
        response = await client.chat_completion(...)
        cache.put(key, response)
    else:
        response = cached
"""

from .keys import CacheKey, compute_cache_key
from .response_cache import ResponseCache

__all__ = [
    "CacheKey",
    "compute_cache_key",
    "ResponseCache",
]
