"""
Smoke tests for toolbench.inference.cache (offline, no API keys required).

Test cases
----------
1. CacheKey determinism: two computes with same inputs produce equal keys.
2. Round-trip: put then get returns equal-shaped NormalizedResponse.
3. Cache miss: get on an empty cache returns None.
4. Atomic write cleanup: .tmp file absent after successful put; put failure
   (caused by a read-only directory) raises and leaves no .tmp behind.
5. Disabled mode: enabled=False returns None always, never writes to disk.
6. Stats accuracy: 3 puts + 2 hits + 5 misses → correct counts.
"""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path

import pytest

# ── path setup: same convention as conftest.py ──────────────────────────────
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "StableToolBench"))

from toolbench.inference.cache import CacheKey, ResponseCache, compute_cache_key
from toolbench.inference.LLM.clients.base import NormalizedResponse


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
]

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Basic arithmetic",
            "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}},
        },
    }
]


def _make_response(content: str = "4", cached_input_tokens: int = 0) -> NormalizedResponse:
    """Build a minimal NormalizedResponse for testing."""
    return NormalizedResponse(
        content=content,
        tool_calls=[],
        finish_reason="stop",
        model_revision="gpt-4.1-mini-2025-04-14",
        input_tokens=10,
        cached_input_tokens=cached_input_tokens,
        output_tokens=5,
        latency_ms=123.4,
        provider="openai",
        raw_response={"id": "chatcmpl-test", "object": "chat.completion"},
    )


# ---------------------------------------------------------------------------
# Test 1: CacheKey determinism
# ---------------------------------------------------------------------------


def test_cache_key_determinism():
    """Same inputs always produce an equal CacheKey."""
    key1 = compute_cache_key(
        model="gpt-4.1-mini-2025-04-14",
        messages=SAMPLE_MESSAGES,
        tools=SAMPLE_TOOLS,
        temperature=0.0,
        seed=42,
    )
    key2 = compute_cache_key(
        model="gpt-4.1-mini-2025-04-14",
        messages=SAMPLE_MESSAGES,
        tools=SAMPLE_TOOLS,
        temperature=0.0,
        seed=42,
    )
    assert key1 == key2, "CacheKey must be deterministic for identical inputs"
    assert key1.hex() == key2.hex(), "hex() must be stable"
    assert key1.as_path() == key2.as_path(), "as_path() must be stable"


def test_cache_key_different_inputs():
    """Different inputs produce different CacheKeys."""
    key1 = compute_cache_key(
        model="gpt-4.1-mini-2025-04-14",
        messages=SAMPLE_MESSAGES,
        temperature=0.0,
        seed=42,
    )
    key2 = compute_cache_key(
        model="gpt-4.1-mini-2025-04-14",
        messages=SAMPLE_MESSAGES,
        temperature=0.7,  # different temperature
        seed=42,
    )
    assert key1 != key2, "Different temperature must produce different key"


def test_cache_key_invalid_messages_stripped():
    """Messages with valid=False are stripped before hashing."""
    messages_with_invalid = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "filtered", "valid": False},
        {"role": "user", "content": "What is 2+2?"},
    ]
    key_filtered = compute_cache_key(
        model="gpt-4.1-mini-2025-04-14",
        messages=messages_with_invalid,
        temperature=0.0,
        seed=42,
    )
    key_clean = compute_cache_key(
        model="gpt-4.1-mini-2025-04-14",
        messages=SAMPLE_MESSAGES,
        temperature=0.0,
        seed=42,
    )
    assert key_filtered == key_clean, "valid=False messages must be stripped before hashing"


def test_cache_key_key_order_invariant():
    """Dict key order in messages does not affect the hash."""
    messages_a = [{"role": "user", "content": "hi"}]
    messages_b = [{"content": "hi", "role": "user"}]  # reversed order
    key_a = compute_cache_key("m", messages_a, temperature=0.0, seed=0)
    key_b = compute_cache_key("m", messages_b, temperature=0.0, seed=0)
    assert key_a == key_b, "Key order in message dicts must not affect hash"


def test_cache_key_as_path_format():
    """as_path returns '<2-char prefix>/<64-char hex>.json'."""
    key = compute_cache_key("m", SAMPLE_MESSAGES, temperature=0.0, seed=0)
    path = key.as_path()
    parts = path.split("/")
    assert len(parts) == 2, f"Expected 'prefix/key.json', got {path!r}"
    prefix, filename = parts
    assert len(prefix) == 2
    assert filename.endswith(".json")
    key_hex = filename[: -len(".json")]
    assert len(key_hex) == 64
    assert prefix == key_hex[:2]


# ---------------------------------------------------------------------------
# Test 2: Round-trip (put then get)
# ---------------------------------------------------------------------------


def test_round_trip(tmp_path: Path):
    """put then get returns a NormalizedResponse with the same shape."""
    cache = ResponseCache(tmp_path)
    key = compute_cache_key(
        model="gpt-4.1-mini-2025-04-14",
        messages=SAMPLE_MESSAGES,
        tools=SAMPLE_TOOLS,
        temperature=0.0,
        seed=42,
    )
    original = _make_response(content="the answer is 4")
    cache.put(key, original)

    retrieved = cache.get(key)
    assert retrieved is not None, "get after put must return a response"
    assert retrieved.content == original.content
    assert retrieved.tool_calls == original.tool_calls
    assert retrieved.finish_reason == original.finish_reason
    assert retrieved.model_revision == original.model_revision
    assert retrieved.input_tokens == original.input_tokens
    assert retrieved.cached_input_tokens == original.cached_input_tokens
    assert retrieved.output_tokens == original.output_tokens
    assert retrieved.provider == original.provider


# ---------------------------------------------------------------------------
# Test 3: Cache miss
# ---------------------------------------------------------------------------


def test_cache_miss(tmp_path: Path):
    """get on an entry that was never put returns None."""
    cache = ResponseCache(tmp_path)
    key = compute_cache_key(
        model="gpt-4.1-mini-2025-04-14",
        messages=SAMPLE_MESSAGES,
        temperature=0.0,
        seed=99,
    )
    result = cache.get(key)
    assert result is None, "get on missing key must return None"


# ---------------------------------------------------------------------------
# Test 4: Atomic write — .tmp cleanup
# ---------------------------------------------------------------------------


def test_atomic_write_no_tmp_on_success(tmp_path: Path):
    """After a successful put, no .tmp file remains."""
    cache = ResponseCache(tmp_path)
    key = compute_cache_key("m", SAMPLE_MESSAGES, temperature=0.0, seed=0)
    cache.put(key, _make_response())

    # Walk entire cache dir looking for .tmp files
    tmp_files = list(tmp_path.rglob("*.tmp"))
    assert tmp_files == [], f"Stale .tmp files found: {tmp_files}"


def test_atomic_write_no_tmp_on_failure():
    """After a failed put (read-only dir), no .tmp file persists."""
    with tempfile.TemporaryDirectory() as td:
        cache_dir = Path(td) / "cache"
        cache_dir.mkdir()
        cache = ResponseCache(cache_dir)

        key = compute_cache_key("m", SAMPLE_MESSAGES, temperature=0.0, seed=0)
        # Pre-create the prefix subdirectory so ResponseCache descends into it
        prefix_dir = cache_dir / key.hex()[:2]
        prefix_dir.mkdir(parents=True, exist_ok=True)

        # Make prefix dir read-only to force write failure
        os.chmod(prefix_dir, stat.S_IRUSR | stat.S_IXUSR)
        try:
            with pytest.raises(Exception):
                cache.put(key, _make_response())

            # No .tmp should remain
            tmp_files = list(cache_dir.rglob("*.tmp"))
            assert tmp_files == [], f"Stale .tmp files found after failure: {tmp_files}"
        finally:
            # Restore permissions so TemporaryDirectory cleanup works
            os.chmod(prefix_dir, stat.S_IRWXU)


# ---------------------------------------------------------------------------
# Test 5: Disabled mode
# ---------------------------------------------------------------------------


def test_disabled_mode_get_returns_none(tmp_path: Path):
    """With enabled=False, get always returns None even after put."""
    # First put with enabled=True so data is on disk
    cache_on = ResponseCache(tmp_path, enabled=True)
    key = compute_cache_key("m", SAMPLE_MESSAGES, temperature=0.0, seed=0)
    cache_on.put(key, _make_response())

    # Now open with enabled=False — should not hit the disk entry
    cache_off = ResponseCache(tmp_path, enabled=False)
    result = cache_off.get(key)
    assert result is None, "Disabled cache must return None even for existing entries"


def test_disabled_mode_put_no_disk_write():
    """With enabled=False, put writes nothing to disk."""
    with tempfile.TemporaryDirectory() as td:
        cache_dir = Path(td) / "cache"
        cache = ResponseCache(cache_dir, enabled=False)
        key = compute_cache_key("m", SAMPLE_MESSAGES, temperature=0.0, seed=0)
        cache.put(key, _make_response())  # must be no-op

        # cache_dir should not have been created at all (or be empty)
        if cache_dir.exists():
            all_files = list(cache_dir.rglob("*"))
            json_files = [f for f in all_files if f.suffix == ".json"]
            assert json_files == [], "Disabled cache must not write any .json files"


# ---------------------------------------------------------------------------
# Test 6: Stats accuracy
# ---------------------------------------------------------------------------


def test_stats_accuracy(tmp_path: Path):
    """3 puts + 2 hits + 5 misses → stats match exactly."""
    cache = ResponseCache(tmp_path)
    model = "gpt-4.1-mini-2025-04-14"

    # 3 different puts
    keys = [
        compute_cache_key(model, SAMPLE_MESSAGES, temperature=float(i), seed=i)
        for i in range(3)
    ]
    for k in keys:
        cache.put(k, _make_response(content=f"resp-{k.seed}"))

    # 2 hits (get on keys that exist)
    cache.get(keys[0])
    cache.get(keys[1])

    # 5 misses (get on keys that don't exist)
    miss_keys = [
        compute_cache_key(model, SAMPLE_MESSAGES, temperature=float(10 + i), seed=10 + i)
        for i in range(5)
    ]
    for mk in miss_keys:
        cache.get(mk)

    stats = cache.stats()
    assert stats["puts"] == 3, f"Expected 3 puts, got {stats['puts']}"
    assert stats["hits"] == 2, f"Expected 2 hits, got {stats['hits']}"
    assert stats["misses"] == 5, f"Expected 5 misses, got {stats['misses']}"
    assert stats["bytes_written"] > 0, "bytes_written must be > 0 after puts"


def test_stats_disabled_increments_misses(tmp_path: Path):
    """Disabled cache increments misses so callers can confirm it's off."""
    cache = ResponseCache(tmp_path, enabled=False)
    key = compute_cache_key("m", SAMPLE_MESSAGES, temperature=0.0, seed=0)
    for _ in range(4):
        cache.get(key)

    stats = cache.stats()
    assert stats["misses"] == 4, "Disabled cache must still count misses"
    assert stats["hits"] == 0
    assert stats["puts"] == 0


# ---------------------------------------------------------------------------
# Test 7: raw_response capping
# ---------------------------------------------------------------------------


def test_raw_response_cap(tmp_path: Path):
    """raw_response larger than 64KB is not stored (replaced with {})."""
    cache = ResponseCache(tmp_path)
    key = compute_cache_key("m", SAMPLE_MESSAGES, temperature=0.0, seed=0)

    # Build a response with a raw_response that exceeds 64KB
    big_payload = {"data": "x" * (65 * 1024)}
    response = NormalizedResponse(
        content="hi",
        tool_calls=[],
        finish_reason="stop",
        model_revision="gpt-4.1",
        input_tokens=1,
        cached_input_tokens=0,
        output_tokens=1,
        latency_ms=1.0,
        provider="openai",
        raw_response=big_payload,
    )
    cache.put(key, response)
    retrieved = cache.get(key)
    assert retrieved is not None
    # raw_response should be empty dict (truncated) rather than the big payload
    assert retrieved.raw_response == {}, "Oversized raw_response must be truncated to {}"
