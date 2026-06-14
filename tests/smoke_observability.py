"""
Smoke tests for toolbench.observability.

All tests run offline — no network access required.

Test plan:
1. Construct a ManifestEntry, validate schema.
2. Write 100 entries from 4 concurrent threads; verify no corruption, all present.
3. Run aggregator on the generated manifest; verify cost_table.csv schema.
4. Run replay() on a fresh manifest; verify 0 drift reported.
5. Trigger BudgetGuard exceedance at $0.01 cap; verify ABORTED_BUDGET sentinel.
"""

from __future__ import annotations

import csv
import sys
import tempfile
import threading
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup: allow ``python tests/smoke_observability.py`` from repo root
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "StableToolBench"))

from toolbench.observability import (  # noqa: E402
    BudgetExceeded,
    BudgetGuard,
    ManifestEntry,
    ManifestWriter,
    entry_from_response,
    read_manifest,
)
from toolbench.observability.aggregator import aggregate, write_cost_table  # noqa: E402
from toolbench.observability.replay import replay  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

COST_TABLE_FIELDNAMES = [
    "model", "variant", "benchmark",
    "n_queries", "n_api_calls", "mean_calls_per_query",
    "total_cost_usd", "cost_per_query_usd",
    "total_input_tokens", "total_cached_tokens", "total_output_tokens",
    "cache_hit_rate", "p50_latency_ms", "p95_latency_ms",
]


def _make_entry(i: int) -> ManifestEntry:
    """Build a valid ManifestEntry for test *i*."""
    return ManifestEntry(
        ts="2026-05-24T01:23:45.000Z",
        run_id="test_run_001",
        git_commit="abc123def456",
        config_hash="sha256deadbeef",
        qid=f"toolret:code:{i:04d}",
        operation="pseudo_tool_gen",
        variant="memetic",
        generation=i % 4,
        model="claude-sonnet-4-6",
        provider="anthropic",
        input_tokens=1000,
        cached_input_tokens=800,
        output_tokens=200,
        latency_ms=300.0 + i,
        cost_usd=0.0001 * (i + 1),
        request_hash=f"req{i:04x}" * 8,
        response_hash=f"resp{i:04x}" * 8,
        retry_count=0,
        error=None,
    )


# ---------------------------------------------------------------------------
# Test 1 — Schema validation
# ---------------------------------------------------------------------------

def test_schema_validation() -> None:
    """ManifestEntry validates correctly and rejects missing required fields."""
    entry = _make_entry(0)
    assert entry.model == "claude-sonnet-4-6"
    assert entry.input_tokens == 1000
    assert entry.cached_input_tokens == 800
    assert entry.output_tokens == 200
    assert entry.cost_usd > 0

    # Round-trip serialisation
    line = entry.to_jsonl_line()
    restored = ManifestEntry.from_jsonl_line(line)
    assert restored.qid == entry.qid
    assert restored.cost_usd == entry.cost_usd

    # Missing required field should raise
    try:
        ManifestEntry(ts="", run_id="x", git_commit="x", config_hash="x",
                      qid="x", operation="other", variant="other", model="x",
                      provider="other")
        raise AssertionError("Expected ValidationError for empty ts")
    except Exception as exc:
        assert "ts" in str(exc).lower() or "validation" in str(exc).lower(), (
            f"Unexpected error: {exc}"
        )

    print("  [PASS] test_schema_validation")


# ---------------------------------------------------------------------------
# Test 2 — Concurrent writer (4 threads, 100 entries)
# ---------------------------------------------------------------------------

def test_concurrent_writer() -> None:
    """4 threads write 25 entries each; file has 100 valid, non-corrupt lines."""
    N_THREADS = 4
    ENTRIES_PER_THREAD = 25
    TOTAL = N_THREADS * ENTRIES_PER_THREAD

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = Path(tmpdir) / "manifest.jsonl"

        errors: list[str] = []

        def writer_thread(thread_id: int) -> None:
            with ManifestWriter(manifest_path, run_id="test_run_concurrent") as mw:
                for i in range(ENTRIES_PER_THREAD):
                    idx = thread_id * ENTRIES_PER_THREAD + i
                    mw.write(_make_entry(idx))
                    # Minimal sleep to encourage interleaving
                    time.sleep(0.001)

        threads = [threading.Thread(target=writer_thread, args=(t,)) for t in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Read back
        entries = read_manifest(manifest_path)
        assert len(entries) == TOTAL, (
            f"Expected {TOTAL} entries, got {len(entries)}"
        )

        # Check no line corruption: every entry should parse cleanly
        with open(manifest_path, "r", encoding="utf-8") as fh:
            raw_lines = [l for l in fh if l.strip()]
        assert len(raw_lines) == TOTAL, (
            f"Raw line count {len(raw_lines)} != {TOTAL}"
        )

        # All qids should be unique (each entry has a distinct qid by construction)
        qids = {e.qid for e in entries}
        assert len(qids) == TOTAL, f"Duplicate qids detected: {TOTAL - len(qids)} collisions"

    print(f"  [PASS] test_concurrent_writer ({TOTAL} entries, {N_THREADS} threads)")


# ---------------------------------------------------------------------------
# Test 3 — Aggregator produces correct CSV schema
# ---------------------------------------------------------------------------

def test_aggregator() -> None:
    """Aggregator produces cost_table.csv with expected columns and ≥1 row."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = Path(tmpdir) / "manifest.jsonl"
        out_dir = Path(tmpdir) / "out"
        out_dir.mkdir()

        # Write a small manifest with 2 models × 2 variants
        entries_to_write = []
        for i in range(20):
            e = ManifestEntry(
                ts="2026-05-24T01:23:45.000Z",
                run_id="agg_test",
                git_commit="abc123",
                config_hash="cfg_hash",
                qid=f"toolret:code:{i:04d}",
                operation="pseudo_tool_gen",
                variant="memetic" if i % 2 == 0 else "single_pass",
                generation=0,
                model="claude-sonnet-4-6" if i < 10 else "gpt-4.1-mini",
                provider="anthropic" if i < 10 else "openai",
                input_tokens=500,
                cached_input_tokens=200,
                output_tokens=100,
                latency_ms=200.0,
                cost_usd=0.001,
                request_hash="req" * 10,
                response_hash="resp" * 10,
            )
            entries_to_write.append(e)

        with ManifestWriter(manifest_path, run_id="agg_test") as mw:
            for e in entries_to_write:
                mw.write(e)

        # Run aggregation
        cells, pareto_extras = aggregate([manifest_path])
        assert len(cells) > 0, "No cells produced by aggregator"

        write_cost_table(cells, out_dir / "cost_table.csv")

        # Verify CSV schema
        with open(out_dir / "cost_table.csv", "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            assert list(reader.fieldnames) == COST_TABLE_FIELDNAMES, (
                f"Schema mismatch: {reader.fieldnames}"
            )
            rows = list(reader)
        assert len(rows) >= 2, f"Expected ≥2 rows, got {len(rows)}"

    print(f"  [PASS] test_aggregator ({len(rows)} rows in cost_table.csv)")


# ---------------------------------------------------------------------------
# Test 4 — replay() reports 0 drift for fresh manifest
# ---------------------------------------------------------------------------

def test_replay_zero_drift() -> None:
    """replay() on a freshly-written manifest reports 0 hash drift."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = Path(tmpdir) / "manifest.jsonl"

        with ManifestWriter(manifest_path, run_id="replay_test") as mw:
            for i in range(10):
                mw.write(_make_entry(i))

        report = replay(
            manifest_path=manifest_path,
            git_commit_check=False,  # Not checking git HEAD in unit tests
        )
        assert report.hash_drift_count == 0, (
            f"Unexpected hash drift: {report.hash_drift_count}"
        )
        assert report.total_entries == 10
        assert len(report.errors) == 0

    print("  [PASS] test_replay_zero_drift")


# ---------------------------------------------------------------------------
# Test 5 — BudgetGuard triggers at $0.01 cap
# ---------------------------------------------------------------------------

def test_budget_guard() -> None:
    """BudgetGuard raises BudgetExceeded and writes ABORTED_BUDGET sentinel."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "run"
        guard = BudgetGuard(run_dir=run_dir, max_cost_usd=0.01)

        sentinel = run_dir / "ABORTED_BUDGET"

        # Below threshold: should not raise
        guard.record(0.005)
        guard.check()  # $0.005 < $0.01 → ok
        assert not sentinel.exists()

        # Exceed threshold
        guard.record(0.006)  # total $0.011 > $0.01

        exceeded = False
        try:
            guard.check()
        except BudgetExceeded as exc:
            exceeded = True
            assert exc.total > 0.01
            assert exc.limit == 0.01

        assert exceeded, "BudgetExceeded was not raised"
        assert sentinel.exists(), "ABORTED_BUDGET sentinel not created"
        assert guard.aborted, "guard.aborted should be True after exceedance"

    print("  [PASS] test_budget_guard (ABORTED_BUDGET sentinel verified)")


# ---------------------------------------------------------------------------
# Entry_from_response helper smoke test
# ---------------------------------------------------------------------------

def test_entry_from_response() -> None:
    """entry_from_response correctly parses OpenAI-style response dicts."""
    fake_openai_response = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "gpt-4.1-mini",
        "choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 150,
            "completion_tokens": 30,
            "prompt_tokens_details": {"cached_tokens": 100},
        },
    }

    t0 = time.monotonic()
    time.sleep(0.001)
    t1 = time.monotonic()

    entry = entry_from_response(
        response=fake_openai_response,
        qid="toolret:code:0001",
        operation="pseudo_tool_gen",
        variant="memetic",
        generation=1,
        run_id="test_efr",
        git_commit="abc123",
        config_hash="cfghash",
        model="gpt-4.1-mini",
        provider="openai",
        started_at=t0,
        finished_at=t1,
    )

    assert entry.input_tokens == 50, f"Expected 50 non-cached, got {entry.input_tokens}"
    assert entry.cached_input_tokens == 100, f"Expected 100 cached, got {entry.cached_input_tokens}"
    assert entry.output_tokens == 30
    assert entry.cost_usd > 0
    assert entry.latency_ms > 0
    assert entry.response_hash != ""

    print("  [PASS] test_entry_from_response")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Smoke tests: toolbench.observability ===\n")
    tests = [
        test_schema_validation,
        test_concurrent_writer,
        test_aggregator,
        test_replay_zero_drift,
        test_budget_guard,
        test_entry_from_response,
    ]
    failed: list[str] = []
    for test_fn in tests:
        name = test_fn.__name__
        try:
            test_fn()
        except Exception as exc:
            print(f"  [FAIL] {name}: {exc}")
            import traceback
            traceback.print_exc()
            failed.append(name)

    print(f"\n{'='*48}")
    if failed:
        print(f"FAILED: {len(failed)}/{len(tests)} tests: {failed}")
        sys.exit(1)
    else:
        print(f"ALL {len(tests)} tests PASSED.")
        sys.exit(0)


if __name__ == "__main__":
    main()
