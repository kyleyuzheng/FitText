"""Process-local LLM usage accumulator for per-query token / call logging.

Why this exists
---------------
The root-node ablation (and efficiency reporting generally) needs per-query
counts of LLM calls and token usage (input / output / reasoning / cached).
StableToolBench's per-cell aggregate cost reports are too coarse — they average
over a sampled subset and cannot attribute cost to an individual qid.

Concurrency model (why a plain module-global is safe)
----------------------------------------------------
The canonical solver fans out one ``qa_pipeline_open_domain.py`` OS subprocess
per qid. Within a subprocess the DFS tree search issues LLM calls strictly
sequentially. Therefore a module-global counter is effectively per-query: there
is no intra-process concurrency to interleave attribution. A lock is used
defensively so this stays correct even if a future caller runs more than one
query in a single process (call ``reset_usage()`` at the start of each query in
that case — ``run_single_task`` already does).

Usage
-----
- ``reset_usage()``  — call at the start of a query (before any LLM call).
- ``record_call(...)`` — called inside each model client return path.
- ``snapshot_usage()`` — call after the query completes to read the totals.

Both OpenAI client return paths (Responses API and Chat Completions) call
``record_call``. The vLLM client may call it too if/when per-query token
logging is wanted for local models; it is a no-op for anyone who never calls
``reset_usage``/``snapshot_usage``.
"""

import threading

_lock = threading.Lock()

# Fields tracked per query. Keys are stable — downstream extraction relies on
# them — so do not rename without updating the per-query CSV extractor.
_FIELDS = (
    "llm_calls",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cached_input_tokens",
)

_usage = {k: 0 for k in _FIELDS}


def reset_usage() -> None:
    """Zero the accumulator. Call at the start of each query."""
    with _lock:
        for k in _FIELDS:
            _usage[k] = 0


def record_call(
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    reasoning_tokens: int = 0,
    cached_input_tokens: int = 0,
) -> None:
    """Record one LLM call's token usage. Increments the call counter by 1.

    All token arguments are coerced to int and treated as 0 if None, so callers
    can pass provider usage fields directly without guarding for missing values.
    """
    with _lock:
        _usage["llm_calls"] += 1
        _usage["input_tokens"] += int(input_tokens or 0)
        _usage["output_tokens"] += int(output_tokens or 0)
        _usage["reasoning_tokens"] += int(reasoning_tokens or 0)
        _usage["cached_input_tokens"] += int(cached_input_tokens or 0)


def snapshot_usage() -> dict:
    """Return a copy of the current accumulator totals."""
    with _lock:
        return dict(_usage)
