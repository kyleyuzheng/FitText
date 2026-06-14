"""
OpenAI chat-completions client.

Handles models with prefixes: gpt-*, o*, chatgpt-*.
Uses the native ``openai.AsyncOpenAI`` SDK (not the compat shim).

Automatic prompt caching (>1024-token prompts) is handled transparently by
OpenAI; usage.prompt_tokens_details.cached_tokens is read for telemetry.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .base import ModelClient, NormalizedResponse
from toolbench.inference.LLM.usage_tracker import record_call as _record_usage_call

# Module-level client; constructed once so connections are reused.
_client: Optional[AsyncOpenAI] = None


def _to_responses_input(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Re-serialize Chat-Completions-shaped conversation history into the native
    Responses-API ``input`` list.

    The Responses API does NOT accept assistant messages with embedded
    ``tool_calls`` or role="tool" results. It expects flat input items:
        - {"type": "function_call",        "call_id", "name", "arguments"}
        - {"type": "function_call_output", "call_id", "output"}
    System/user/assistant text-only messages pass through unchanged.

    Failure to convert produces HTTP 400 invalid_type on any turn beyond the
    first tool call — empirically confirmed 2026-05-25.

    Parameters
    ----------
    messages : list[dict]
        Conversation in Chat-Completions shape (role + content [+ tool_calls
        | + tool_call_id]).

    Returns
    -------
    list[dict]
        Responses-API input list. Order is preserved; tool_call text content
        (rare but possible) is emitted as a preceding assistant message.
    """
    out: List[Dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            out.append(m)
            continue
        role = m.get("role")
        # role="tool" → function_call_output (single item per message)
        if role == "tool":
            call_id = m.get("tool_call_id")
            content = m.get("content")
            if isinstance(content, (dict, list)):
                content = __import__("json").dumps(content, ensure_ascii=False)
            elif content is None:
                content = ""
            out.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": str(content),
            })
            continue
        # assistant with tool_calls → optional text message + function_call items
        if role == "assistant" and m.get("tool_calls"):
            text = m.get("content")
            if isinstance(text, str) and text.strip():
                out.append({"role": "assistant", "content": text})
            for tc in m["tool_calls"] or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                out.append({
                    "type": "function_call",
                    "call_id": tc.get("id"),
                    "name": fn.get("name", ""),
                    "arguments": fn.get("arguments", "") or "",
                })
            continue
        # assistant with content=None and no tool_calls → drop (Responses API
        # rejects null content). Should not happen in practice but guard.
        if role == "assistant" and m.get("content") is None:
            continue
        # system / user / assistant-with-text pass through. Strip Chat-only
        # keys that Responses API doesn't know about.
        passthrough = {k: v for k, v in m.items()
                       if k not in ("tool_calls", "function_call", "tool_call_id", "name", "valid")}
        # Ensure content is a string (Responses accepts string or array; null
        # is invalid).
        if passthrough.get("content") is None:
            passthrough["content"] = ""
        out.append(passthrough)
    return out


def _get_openai_client() -> AsyncOpenAI:
    """
    Return the module-level ``AsyncOpenAI`` instance, creating it on first call.

    API key is read from the ``OPENAI_API_KEY`` environment variable (default
    SDK behaviour).

    Returns
    -------
    AsyncOpenAI
    """
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
    return _client


class OpenAIClient(ModelClient):
    """
    ModelClient implementation for OpenAI (gpt-*, o*, chatgpt-*).

    Parameters
    ----------
    model : str
        Full dated model tag, e.g. ``"gpt-4.1-mini-2025-04-14"``.
    api_key : str | None
        Override for the API key; falls back to ``OPENAI_API_KEY`` env var.
    **kwargs
        Ignored (forward-compat for factory).
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        cache: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, cache=cache)
        # If a per-instance key is provided, create a dedicated client.
        # Otherwise, share the module-level client for connection reuse.
        if api_key:
            self._client = AsyncOpenAI(api_key=api_key)
        else:
            self._client = None  # resolved lazily via _get_openai_client()

    def _resolve_client(self) -> AsyncOpenAI:
        """Return the per-instance or module-level client."""
        return self._client if self._client is not None else _get_openai_client()

    @retry(
        wait=wait_random_exponential(min=1, max=40),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
        seed: int = 42,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> NormalizedResponse:
        """
        Call the OpenAI chat completions endpoint.

        Parameters
        ----------
        messages : list[dict]
            OpenAI-format conversation history.
        tools : list[dict] | None
            OpenAI-format tool definitions.
        temperature : float
            Sampling temperature.
        seed : int
            Reproducibility seed.
        max_tokens : int | None
            Max completion tokens; defaults to 1024 if not provided.
        **kwargs
            Extra parameters forwarded to ``create()``.

        Returns
        -------
        NormalizedResponse
        """
        client = self._resolve_client()
        t0 = self._now_ms()

        # Reasoning-model detection (gpt-5.x / o1 / o3 / o4 families).
        # Per OpenAI docs: reasoning tokens are counted toward
        # max_completion_tokens and consume ≥25k of the budget. Setting a
        # small max_completion_tokens (e.g. 1024) on a reasoning model causes
        # status="incomplete" / incomplete_details.reason="max_output_tokens"
        # with empty/short visible content because the reasoning chain alone
        # exhausts the budget before any output is generated.
        # https://developers.openai.com/api/docs/guides/reasoning
        _is_reasoning = self.model.startswith(("gpt-5", "o1", "o3", "o4"))

        # Allow forcing the legacy Chat Completions path (for A/B tests).
        import os as _os
        _force_chat = _os.environ.get("OPENAI_FORCE_CHAT_COMPLETIONS") == "1"

        if _is_reasoning and not _force_chat:
            # ROUTE TO RESPONSES API (`/v1/responses`).
            # Why migrate from Chat Completions for reasoning models:
            #   - Chat Completions + tools + reasoning_effort → HTTP 400.
            #   - Omitting reasoning_effort on Chat Completions silently
            #     DISABLES reasoning (reasoning_tokens=0, empirically
            #     confirmed 2026-05-25).
            #   - Responses API supports tools + reasoning together, gets
            #     +3pp SWE-bench at same prompt, +40-80% cache hit rate.
            # Sources: https://developers.openai.com/api/docs/guides/migrate-to-responses
            _env_override = _os.environ.get("OPENAI_MAX_COMPLETION_TOKENS")
            if _env_override:
                try:
                    effective_max = int(_env_override)
                except ValueError:
                    effective_max = max(max_tokens or 0, 25000)
            else:
                effective_max = max(max_tokens or 0, 25000)
            # Reasoning effort: caller decides. Empirically verified
            # 2026-05-25 against gpt-5.4-mini-2026-03-17 via Responses API
            # probe: omitting `reasoning.effort` yields server-side default of
            # `effort='none'` and `reasoning_tokens=0` — i.e. reasoning is
            # genuinely OFF, NOT a "minimal" or "medium" default as earlier
            # comments implied. This is exactly the parity behavior we want
            # for fair gpt-5.4-mini vs gpt-4.1-mini comparisons. To opt INTO
            # reasoning, set OPENAI_REASONING_EFFORT to one of:
            #     none | low | medium | high | xhigh
            # ("minimal" is not accepted on gpt-5.4-mini-2026-03-17.)
            _env_effort = _os.environ.get("OPENAI_REASONING_EFFORT")
            _kw_effort = kwargs.pop("reasoning_effort", None)
            _reasoning_effort = _env_effort or _kw_effort  # None → omit → off
            # Convert Chat-Completions-shaped history to Responses-API input items.
            # CRITICAL (empirically confirmed 2026-05-25):
            # Responses API rejects with HTTP 400 invalid_type when input contains
            # {"role":"assistant","content":None,"tool_calls":[...]} (Chat-Completions
            # shape). Multi-turn tool flows (DFSDT, etc.) MUST be re-serialized into
            # the native Responses-API shape:
            #   - assistant message with tool_calls   → emit function_call input items
            #     (one per tool_call), assistant.content text (if any) emitted as a
            #     separate user-shaped message before the function_calls
            #   - role:tool message                   → emit function_call_output input item
            #   - assistant message with content=None → drop (Responses API rejects null)
            #   - any other role (system/user/assistant-with-text-only) → pass through
            # If we don't convert, tenacity silently retries the 400 four times then
            # returns empty, and DFSDT chains die at turn 2 — the exact failure mode
            # observed (query_count=105, total_tokens=3001, final_answer="").
            responses_input: List[Dict[str, Any]] = _to_responses_input(messages)
            responses_kwargs: Dict[str, Any] = {
                "model": self.model,
                "input": responses_input,
                "max_output_tokens": effective_max,
                **{k: v for k, v in kwargs.items() if k not in ("temperature", "top_p")},
            }
            if _reasoning_effort:
                responses_kwargs["reasoning"] = {"effort": _reasoning_effort}
            if tools:
                # Responses API uses a FLAT tool schema, not Chat Completions'
                # nested `{"type":"function","function":{...}}`. Convert.
                _responses_tools: List[Dict[str, Any]] = []
                for t in tools:
                    if isinstance(t, dict) and t.get("type") == "function" and "function" in t:
                        f = t["function"]
                        _responses_tools.append({
                            "type": "function",
                            "name": f.get("name", ""),
                            "description": f.get("description", ""),
                            "parameters": f.get("parameters", {}),
                        })
                    else:
                        # Already flat / unknown shape — pass through.
                        _responses_tools.append(t)
                responses_kwargs["tools"] = _responses_tools

            response = await client.responses.create(**responses_kwargs)
            latency_ms = self._now_ms() - t0

            # Parse Responses-API output: response.output is a list of items
            # (reasoning / message / function_call). Aggregate visible text +
            # extract tool calls.
            content_parts: List[str] = []
            tool_calls: List[Dict[str, Any]] = []
            finish_reason = "stop"
            for item in (response.output or []):
                itype = getattr(item, "type", None)
                if itype == "message":
                    for c in (getattr(item, "content", []) or []):
                        text = getattr(c, "text", None)
                        if text:
                            content_parts.append(text)
                elif itype == "function_call":
                    tool_calls.append({
                        "id": getattr(item, "call_id", None) or getattr(item, "id", None),
                        "type": "function",
                        "function": {
                            "name": getattr(item, "name", ""),
                            "arguments": getattr(item, "arguments", "{}"),
                        },
                    })
                # reasoning items: ignore content (encrypted/internal); their
                # tokens are counted in usage.output_tokens_details.reasoning_tokens.
            content = "".join(content_parts) if content_parts else None

            # Map Responses status to a finish_reason analog
            status = getattr(response, "status", None)
            if status == "incomplete":
                _details = getattr(response, "incomplete_details", None)
                _reason = getattr(_details, "reason", None) if _details else None
                finish_reason = "length" if _reason == "max_output_tokens" else "incomplete"
            elif tool_calls:
                finish_reason = "tool_calls"

            usage = getattr(response, "usage", None)
            input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
            output_tokens = getattr(usage, "output_tokens", 0) if usage else 0
            cached_tokens = 0
            reasoning_tokens = 0
            if usage:
                _in_det = getattr(usage, "input_tokens_details", None)
                if _in_det:
                    cached_tokens = getattr(_in_det, "cached_tokens", 0) or 0
                _out_det = getattr(usage, "output_tokens_details", None)
                if _out_det:
                    reasoning_tokens = getattr(_out_det, "reasoning_tokens", 0) or 0
            # Per-query usage accumulation (Responses API path).
            _record_usage_call(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                reasoning_tokens=reasoning_tokens,
                cached_input_tokens=cached_tokens,
            )
            try:
                raw = response.model_dump()
            except AttributeError:
                raw = {}
            return NormalizedResponse(
                content=content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                model_revision=getattr(response, "model", self.model),
                input_tokens=input_tokens,
                cached_input_tokens=cached_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                provider="openai",
                raw_response=raw,
            )

        if _is_reasoning:
            # FORCED Chat Completions path for reasoning models (A/B fallback).
            _env_override = _os.environ.get("OPENAI_MAX_COMPLETION_TOKENS")
            if _env_override:
                try:
                    effective_max = int(_env_override)
                except ValueError:
                    effective_max = max(max_tokens or 0, 25000)
            else:
                effective_max = max(max_tokens or 0, 25000)
            create_kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "seed": seed,
                "max_completion_tokens": effective_max,
                **kwargs,
            }
            # NOTE: cannot set reasoning_effort on Chat Completions when tools
            # are present (HTTP 400). Reasoning is silently DISABLED in this
            # path.
        else:
            # Non-reasoning chat models (gpt-4.1, gpt-4o, etc.): legacy path.
            # Env override applies here too for fair cross-model budget studies.
            _env_override = _os.environ.get("OPENAI_MAX_COMPLETION_TOKENS")
            if _env_override:
                try:
                    _legacy_max = int(_env_override)
                except ValueError:
                    _legacy_max = max_tokens or 1024
            else:
                _legacy_max = max_tokens or 1024
            create_kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "seed": seed,
                "max_completion_tokens": _legacy_max,
                **kwargs,
            }
        if tools:
            create_kwargs["tools"] = tools

        # Qwen3.x thinking-mode disable + multi-turn preservation.
        # Without this, the model consumes the entire response window with
        # <think> tokens and emits content but no tool_calls, which DFSDT
        # then salvages as give_up (0% pass rate). vLLM ≥0.21 honors the flag.
        if isinstance(self.model, str) and self.model.lower().startswith("qwen"):
            existing_eb = create_kwargs.get("extra_body") or {}
            ctk = dict(existing_eb.get("chat_template_kwargs") or {})
            ctk.setdefault("enable_thinking", False)
            ctk.setdefault("preserve_thinking", True)
            existing_eb["chat_template_kwargs"] = ctk
            create_kwargs["extra_body"] = existing_eb

        response = await client.chat.completions.create(**create_kwargs)
        latency_ms = self._now_ms() - t0

        choice = response.choices[0]
        msg = choice.message

        tool_calls: List[Dict[str, Any]] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                )

        usage = response.usage
        input_tokens: int = usage.prompt_tokens if usage else 0
        output_tokens: int = usage.completion_tokens if usage else 0

        # OpenAI caching telemetry (automatic for prompts >1024 tokens).
        cached_tokens: int = 0
        reasoning_tokens: int = 0
        if usage and hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
            cached_tokens = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
        if usage and hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
            reasoning_tokens = getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0
        # Per-query usage accumulation (Chat Completions path).
        _record_usage_call(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            cached_input_tokens=cached_tokens,
        )

        # Serialise for provenance; use model_dump() (pydantic v2) with fallback.
        try:
            raw = response.model_dump()
        except AttributeError:
            raw = {}

        return NormalizedResponse(
            content=msg.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            model_revision=response.model,
            input_tokens=input_tokens,
            cached_input_tokens=cached_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            provider="openai",
            raw_response=raw,
        )
