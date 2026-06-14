"""
toolbench.observability — provenance, cost, and telemetry infrastructure.

Every LLM API call made during a FitText run is recorded as one line in a
JSONL manifest file.  The manifest is the **only** authoritative source of
truth for cost, token counts, and latency — scraping log lines is explicitly
prohibited (§7, EXECUTION_PLAN.md).

Module layout
-------------
pricing.py
    Posted API rates table (model → $/1M tokens in/out/cached).
    ``compute_cost(model, input_tokens, cached_input_tokens, output_tokens)``
    returns the USD cost for one call.

manifest.py
    ``ManifestEntry`` — Pydantic model validating one JSONL record.
    ``ManifestWriter`` — context-managed, thread/process-safe JSONL writer.
    ``entry_from_response(...)`` — converts a raw provider response into a
    ``ManifestEntry``; this is the integration point for the modelclient track.

budget_guard.py
    ``BudgetGuard`` — file-locked atomic cost counter + circuit breaker.
    ``BudgetExceeded`` — raised (and sentinel file created) when the cap is hit.

aggregator.py
    Reads one or more manifest JSONL files and produces four CSV outputs:
    ``cost_table.csv``, ``pareto_data.csv``, ``cache_hit.csv``,
    ``latency.csv``.  CLI: ``python -m toolbench.observability.aggregator``.

replay.py
    ``replay(manifest_path, ...)`` — verifies git commit and response hashes
    without making API calls.  Used by ``scripts/audit_run.py``.

Integration contract with the modelclient track
-------------------------------------------------
The ``ModelClient`` (feat/modelclient branch, §5.1) should:

1. Accept a ``ManifestWriter | None`` at construction time::

       class ModelClient:
           def __init__(self, ..., manifest_writer: ManifestWriter | None = None):
               self._manifest = manifest_writer

2. After every provider round-trip, call::

       import time
       from toolbench.observability.manifest import entry_from_response

       t0 = time.monotonic()
       raw_response = <provider call>
       t1 = time.monotonic()

       if self._manifest is not None:
           entry = entry_from_response(
               response=raw_response,
               qid=qid,
               operation=operation,
               variant=variant,
               generation=generation,
               run_id=self._run_id,
               git_commit=self._git_commit,
               config_hash=self._config_hash,
               model=self._model,
               provider=self._provider,
               started_at=t0,
               finished_at=t1,
               request_payload=request_payload_dict,
               retry_count=retry_count,
               error=error_str,
           )
           self._manifest.write(entry)

   If ``manifest_writer`` is ``None``, the client operates in "no-telemetry"
   mode (useful for unit tests).  This is the ONLY code path that records an
   API call — ``ModelClient`` must not have a secondary path that bypasses it.

3. Also call ``BudgetGuard.record(entry.cost_usd)`` and
   ``BudgetGuard.check()`` if a ``BudgetGuard`` is injected.

Public re-exports
-----------------
"""

from .manifest import (  # noqa: F401 — public API
    ManifestEntry,
    ManifestWriter,
    entry_from_response,
    read_manifest,
)
from .pricing import compute_cost, get_rates, resolve_model_key  # noqa: F401
from .budget_guard import BudgetGuard, BudgetExceeded  # noqa: F401

__all__ = [
    # manifest
    "ManifestEntry",
    "ManifestWriter",
    "entry_from_response",
    "read_manifest",
    # pricing
    "compute_cost",
    "get_rates",
    "resolve_model_key",
    # budget_guard
    "BudgetGuard",
    "BudgetExceeded",
]
