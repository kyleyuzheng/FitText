"""Per-call telemetry context for the eval executor.

Strategies (StableToolBench / Toolret) reach the network via two seams:

  - ``chat_completion_request(...)`` in
    ``StableToolBench/toolbench/inference/LLM/chatgpt_function_model.py``
  - ``ChatGPTFunction.parse[_with_messages](...)`` in the same file.

Both ultimately call :func:`make_client(...).chat_completion_sync(...)`.

This module provides a single hook the executor installs **once** at run
start.  Every LLM call is then automatically recorded into the active
:class:`ManifestWriter` and accounted in the active :class:`BudgetGuard`
without modifying any strategy code.

The hook is implemented by monkey-patching :func:`chat_completion_request` at
process start.  Because workers are spawned via ``ProcessPoolExecutor`` after
the patch is applied, child processes inherit the patched function via the
worker initialiser :func:`_worker_init`.

Per-query context (qid, variant, generation) is carried in :class:`contextvars`
so concurrent calls from different queries can be disambiguated when running
under ``asyncio`` or threads.

Operation classification heuristic: best-effort by inspecting the messages.
The executor sets a more specific ``operation`` via :func:`set_call_context`
when it knows what stage of the variant pipeline issued the call.
"""

from __future__ import annotations

import contextvars
import functools
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider classification
# ---------------------------------------------------------------------------


def _infer_provider(model: str, base_url: Any) -> str:
    """Best-effort provider classification.

    Mirrors the prefix routing in
    :func:`toolbench.inference.LLM.clients.factory.make_client`:

      - Any ``base_url`` set → ``vllm``.
      - ``claude*`` → ``anthropic``.
      - ``gpt*`` / ``o*`` → ``openai``.
      - Otherwise → ``openai`` (fallback for the pricing path).
    """
    if base_url:
        return "vllm"
    m = (model or "").lower()
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith(("gpt", "o1", "o3", "o4")):
        return "openai"
    if m.startswith(("qwen", "deepseek", "llama")):
        return "vllm"
    return "openai"


# ---------------------------------------------------------------------------
# Per-call context
# ---------------------------------------------------------------------------


@dataclass
class CallContext:
    """Run-level + per-query context required to fill a ManifestEntry.

    The executor sets the run-level fields (``run_id``, ``git_commit``, ...)
    once at run start, and updates ``qid`` / ``operation`` / ``variant`` /
    ``generation`` per-query / per-stage.
    """

    run_id: str = "unknown_run"
    git_commit: str = "unknown"
    config_hash: str = "unknown"
    qid: str = "unknown_qid"
    operation: str = "dfsdt_node"
    variant: str = "single_pass"
    generation: int = 0


_call_ctx: contextvars.ContextVar[CallContext] = contextvars.ContextVar(
    "call_ctx", default=CallContext()
)


def set_call_context(**fields: Any) -> contextvars.Token:
    """Update the active call context for the current task/thread.

    Returns a token that callers can pass to :func:`reset_call_context` to
    restore the previous state (use a ``try / finally`` block).

    Args:
        **fields: Keyword overrides for any :class:`CallContext` field.

    Returns:
        Token from :func:`contextvars.ContextVar.set`.
    """
    current = _call_ctx.get()
    new = CallContext(
        run_id=fields.get("run_id", current.run_id),
        git_commit=fields.get("git_commit", current.git_commit),
        config_hash=fields.get("config_hash", current.config_hash),
        qid=fields.get("qid", current.qid),
        operation=fields.get("operation", current.operation),
        variant=fields.get("variant", current.variant),
        generation=int(fields.get("generation", current.generation)),
    )
    return _call_ctx.set(new)


def reset_call_context(token: contextvars.Token) -> None:
    """Restore a previously snapshotted call context.

    Args:
        token: Token returned by :func:`set_call_context`.
    """
    _call_ctx.reset(token)


def get_call_context() -> CallContext:
    """Snapshot the active :class:`CallContext`."""
    return _call_ctx.get()


# ---------------------------------------------------------------------------
# Telemetry sink registration
# ---------------------------------------------------------------------------


_active_manifest_writer: Any = None
_active_budget_guard: Any = None


def attach_sinks(*, manifest_writer: Any = None, budget_guard: Any = None) -> None:
    """Install the active manifest writer and budget guard.

    Called once by the executor before any LLM call is issued.  The
    ``chat_completion_request`` patch reads these globals on every call.

    Args:
        manifest_writer: Open :class:`ManifestWriter` (or ``None`` to disable
            telemetry — useful for tests).
        budget_guard: :class:`BudgetGuard` (or ``None`` to skip budget checks).
    """
    global _active_manifest_writer, _active_budget_guard
    _active_manifest_writer = manifest_writer
    _active_budget_guard = budget_guard


def detach_sinks() -> None:
    """Clear the active manifest writer and budget guard (used in tests)."""
    global _active_manifest_writer, _active_budget_guard
    _active_manifest_writer = None
    _active_budget_guard = None


# ---------------------------------------------------------------------------
# Monkey-patch installer
# ---------------------------------------------------------------------------


_PATCHED = False
_PATCHED_MODULES: list[Any] = []


def install_chat_completion_hook() -> None:
    """Wrap ``chat_completion_request`` to record one manifest entry per call.

    Idempotent — re-calls are no-ops.  The hook is process-local; subprocesses
    must call this in their initialiser.

    Both seams are patched:

      - ``StableToolBench/toolbench/inference/LLM/chatgpt_function_model``
      - ``Toolret/strategy/LLM_model``

    Both are near-identical shims that delegate to the unified
    :func:`make_client` factory; the duplication is why both seams must be
    patched.

    The wrapped function:

      1. Captures ``time.monotonic()`` before / after the call.
      2. On success, builds a :class:`ManifestEntry` via
         :func:`entry_from_response` and writes it to the active
         :class:`ManifestWriter` (if any).
      3. Updates the active :class:`BudgetGuard` and calls ``check()``.
      4. Returns the original OpenAI-shaped dict unchanged so strategies are
         unaware of the instrumentation.
    """
    global _PATCHED
    if _PATCHED:
        return

    from toolbench.inference.LLM import chatgpt_function_model as _stb_mod
    from toolbench.observability.manifest import entry_from_response

    # Toolret's parallel module — same surface, different package.
    try:
        import importlib

        _toolret_mod = importlib.import_module("strategy.LLM_model")
    except Exception:  # pragma: no cover — defensive
        _toolret_mod = None
    if _toolret_mod is None:
        try:
            import importlib
            _toolret_mod = importlib.import_module("Toolret.strategy.LLM_model")
        except Exception:
            _toolret_mod = None

    _modules = [_stb_mod]
    if _toolret_mod is not None:
        _modules.append(_toolret_mod)

    for _mod in _modules:
        _orig = _mod.chat_completion_request
        _mod.chat_completion_request = _make_wrapped(_orig, _mod, entry_from_response)  # type: ignore[attr-defined]
        _PATCHED_MODULES.append((_mod, _orig))

    _PATCHED = True
    log.debug("chat_completion_request telemetry hook installed on %d modules", len(_modules))


def _make_wrapped(_orig: Any, _mod: Any, entry_from_response: Any) -> Any:
    """Build the wrapped chat_completion_request for one module.

    Args:
        _orig: The original ``chat_completion_request`` callable to wrap.
        _mod: The module that owns ``_orig`` (used for default model lookup).
        entry_from_response: ``toolbench.observability.manifest.entry_from_response``
            (injected to avoid a circular import in __init__).
    """
    # Lookup model from the wrapped module's pin-loaded constant.
    # Empty-string sentinel makes the absence of a pin loud rather than silent.
    _default_model = getattr(_mod, "_DEFAULT_AGENT_MODEL", "")

    @functools.wraps(_orig)
    def _wrapped(
        key: str,
        base_url: Any,
        messages: list,
        tools: Any = None,
        tool_choice: Any = None,
        key_pos: Any = None,
        model: str = _default_model,
        stop: Any = None,
        process_id: int = 0,
        **args: Any,
    ) -> dict:
        ctx = get_call_context()
        t0 = time.monotonic()
        error: str | None = None
        response: dict = {}
        try:
            response = _orig(
                key,
                base_url,
                messages,
                tools=tools,
                tool_choice=tool_choice,
                key_pos=key_pos,
                model=model,
                stop=stop,
                process_id=process_id,
                **args,
            )
        except Exception as exc:
            error = repr(exc)
            response = {"error": error, "choices": [], "usage": {"total_tokens": 0}}
        t1 = time.monotonic()

        if _active_manifest_writer is not None:
            provider = _infer_provider(model, base_url)
            try:
                entry = entry_from_response(
                    response=response,
                    qid=ctx.qid,
                    operation=ctx.operation,
                    variant=ctx.variant,
                    generation=ctx.generation,
                    run_id=ctx.run_id,
                    git_commit=ctx.git_commit,
                    config_hash=ctx.config_hash,
                    model=model,
                    provider=provider,
                    started_at=t0,
                    finished_at=t1,
                    request_payload={
                        "model": model,
                        "messages": messages,
                        "tools": tools,
                        "temperature": args.get("temperature"),
                        "seed": args.get("seed"),
                    },
                    retry_count=0,
                    error=error,
                )
                _active_manifest_writer.write(entry)
                if _active_budget_guard is not None:
                    _active_budget_guard.record(float(entry.cost_usd))
                    _active_budget_guard.check()
            except Exception as exc:  # noqa: BLE001 — never crash the hot path
                log.warning(
                    "telemetry hook: manifest write failed for qid=%s: %s",
                    ctx.qid,
                    exc,
                )
        return response

    return _wrapped


def uninstall_chat_completion_hook() -> None:
    """Restore the original ``chat_completion_request`` (used in tests)."""
    global _PATCHED, _PATCHED_MODULES
    if not _PATCHED:
        return
    for _mod, _orig in _PATCHED_MODULES:
        try:
            _mod.chat_completion_request = _orig  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover — defensive
            pass
    _PATCHED_MODULES = []
    _PATCHED = False


# ---------------------------------------------------------------------------
# Provenance helpers for adapters / tests
# ---------------------------------------------------------------------------


def write_manual_entry(
    *,
    response: Any,
    operation: str,
    variant: str | None = None,
    qid: str | None = None,
    generation: int | None = None,
    model: str,
    started_at: float,
    finished_at: float,
    cost_usd: float | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cached_input_tokens: int | None = None,
    error: str | None = None,
) -> None:
    """Write one manifest entry directly without going through the hook.

    Used by adapters that own the call path (e.g. test mocks, embedding
    lookups in just_query variant) but want provenance recorded.

    Args:
        response: Provider response dict (best-effort token extraction).
        operation: Pipeline operation name.
        variant: FitText variant or baseline name; defaults to context.
        qid: Query id; defaults to context.
        generation: Generation index; defaults to context.
        model: Full model identifier.
        started_at: ``time.monotonic()`` before the call.
        finished_at: ``time.monotonic()`` after the call.
        cost_usd: Override auto-computed cost (e.g. for mocked responses).
        input_tokens: Override extracted prompt token count.
        output_tokens: Override extracted completion token count.
        cached_input_tokens: Override extracted cached prompt token count.
        error: Short error string on failure; ``None`` on success.
    """
    if _active_manifest_writer is None:
        return
    from toolbench.observability.manifest import entry_from_response

    ctx = get_call_context()
    provider = _infer_provider(model, None)
    entry = entry_from_response(
        response=response,
        qid=qid or ctx.qid,
        operation=operation,
        variant=variant or ctx.variant,
        generation=generation if generation is not None else ctx.generation,
        run_id=ctx.run_id,
        git_commit=ctx.git_commit,
        config_hash=ctx.config_hash,
        model=model,
        provider=provider,
        started_at=started_at,
        finished_at=finished_at,
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        error=error,
    )
    _active_manifest_writer.write(entry)
    if _active_budget_guard is not None:
        _active_budget_guard.record(float(entry.cost_usd))
        _active_budget_guard.check()


def get_run_dir(run_dir: Path) -> Path:
    """Return ``run_dir`` as a :class:`Path`, creating it if needed.

    Helper for adapters that want a stable place to write sentinels.

    Args:
        run_dir: Run output directory.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
