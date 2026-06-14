"""Offline mock LLM clients for the E2E pipeline test.

The mocks are drop-in replacements for ``openai.AsyncOpenAI`` and
``anthropic.AsyncAnthropic`` returning canned ChatCompletion-shaped objects.
They report realistic prompt/completion token counts so the cost-pricing
table produces ``cost_usd > 0`` and the manifest aggregator behaves the same
way as it would in production.

Design notes
------------
- ``MockAsyncOpenAI.chat.completions.create`` is the only async surface used
  by the OpenAI client (``StableToolBench/toolbench/inference/LLM/clients/openai_client.py``);
  we therefore only model that one method.
- Responses are SimpleNamespace-shaped to mimic the OpenAI Pydantic objects
  duck-typed access pattern (``response.choices[0].message.content``,
  ``response.usage.prompt_tokens`` etc.).
- ``scripted_responses`` is keyed by a SHA-256 over ``(model, messages, tools)``;
  callers that need exact response control can pre-populate the dict.
- ``invocation_count`` is exposed for cache-hit tests (a 100% cache hit should
  freeze the counter at its pre-test value).
- ``MockAsyncAnthropic`` mirrors the SDK surface used by ``AnthropicClient``
  (``client.messages.create``) — same pattern.

Usage
-----
::

    from tests.fixtures.mock_llm import MockAsyncOpenAI

    mock = MockAsyncOpenAI(default_response={
        "content": "ok", "tool_calls": None,
        "input_tokens": 100, "output_tokens": 10,
    })

    monkeypatch.setattr(
        "toolbench.inference.LLM.clients.openai_client._get_openai_client",
        lambda: mock,
    )
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from types import SimpleNamespace
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Keying helper
# ---------------------------------------------------------------------------


def _hash_request(model: str, messages: list[dict], tools: Optional[list[dict]]) -> str:
    """Return a stable SHA-256 hash used to key scripted responses.

    Args:
        model: Provider model id (e.g. ``"gpt-4.1-mini-2025-04-14"``).
        messages: OpenAI-format chat messages.
        tools: Optional OpenAI-format tool definitions list.

    Returns:
        Hex string of the SHA-256 over a canonical JSON serialisation.
    """
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools or [],
    }
    canonical = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


# ---------------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------------


def _build_openai_choice(
    content: Optional[str],
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    finish_reason: str = "stop",
) -> SimpleNamespace:
    """Build a SimpleNamespace mimicking ``response.choices[0]``.

    Args:
        content: Assistant text content (None when only tool_calls are present).
        tool_calls: OpenAI-shaped tool_calls list of ``{id, type, function: {name, arguments}}``.
        finish_reason: ``"stop"`` / ``"tool_calls"`` / ``"length"``.

    Returns:
        SimpleNamespace with ``.message`` (.content, .tool_calls) + ``.finish_reason``.
    """
    tc_objs = None
    if tool_calls:
        tc_objs = []
        for tc in tool_calls:
            tc_objs.append(
                SimpleNamespace(
                    id=tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                    type=tc.get("type", "function"),
                    function=SimpleNamespace(
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"],
                    ),
                )
            )

    msg = SimpleNamespace(
        content=content,
        tool_calls=tc_objs,
        role="assistant",
    )
    return SimpleNamespace(
        index=0,
        message=msg,
        finish_reason=finish_reason if not tc_objs else "tool_calls",
    )


def _build_openai_usage(
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
) -> SimpleNamespace:
    """Build a usage SimpleNamespace matching the OpenAI SDK shape.

    Args:
        input_tokens: Non-cached prompt tokens billed.
        output_tokens: Completion tokens.
        cached_tokens: Cached prompt tokens (subset of prompt_tokens).

    Returns:
        SimpleNamespace with ``.prompt_tokens``, ``.completion_tokens``,
        ``.total_tokens``, ``.prompt_tokens_details.cached_tokens``.
    """
    return SimpleNamespace(
        prompt_tokens=input_tokens + cached_tokens,
        completion_tokens=output_tokens,
        total_tokens=input_tokens + cached_tokens + output_tokens,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
    )


def _build_openai_response(
    model: str,
    content: Optional[str],
    tool_calls: Optional[List[Dict[str, Any]]],
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
) -> SimpleNamespace:
    """Assemble a full ChatCompletion-shaped SimpleNamespace.

    Args:
        model: Echoed back in ``.model`` for provenance.
        content: Assistant text (may be None).
        tool_calls: OpenAI-shaped tool_calls list (may be None).
        input_tokens: Prompt tokens (non-cached).
        output_tokens: Completion tokens.
        cached_tokens: Cached prompt tokens.

    Returns:
        SimpleNamespace duck-typed as ``openai.types.chat.ChatCompletion``.
    """
    choice = _build_openai_choice(content=content, tool_calls=tool_calls)
    usage = _build_openai_usage(input_tokens, output_tokens, cached_tokens)
    response = SimpleNamespace(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[choice],
        usage=usage,
    )

    # Provide a model_dump() shim so OpenAI client provenance serialisation
    # (response.model_dump()) doesn't blow up.
    def model_dump(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
        return {
            "id": response.id,
            "object": "chat.completion",
            "created": response.created,
            "model": response.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls,
                    },
                    "finish_reason": choice.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "prompt_tokens_details": {"cached_tokens": cached_tokens},
            },
        }

    response.model_dump = model_dump
    return response


# ---------------------------------------------------------------------------
# MockAsyncOpenAI
# ---------------------------------------------------------------------------


class MockAsyncOpenAI:
    """Drop-in replacement for ``openai.AsyncOpenAI``.

    The mock only models the surface area used by ``OpenAIClient.chat_completion``
    (``self._client.chat.completions.create``).  All other attributes raise
    ``AttributeError`` so accidental real-network access fails loudly.

    Args:
        scripted_responses: Mapping of request-hash → response-dict.  Each
            response-dict may contain keys
            ``content``, ``tool_calls``, ``input_tokens``, ``output_tokens``,
            ``cached_tokens``.
        default_response: Fallback response-dict when no scripted entry matches.
            Same shape as the values of ``scripted_responses``.

    Attributes:
        invocation_count: Number of ``.chat.completions.create`` calls served
            (does NOT include cache hits because cache short-circuits before
            the mock is invoked).
        last_request: The kwargs dict of the most recent ``.create`` call —
            useful for asserting the exact prompt sent.
    """

    def __init__(
        self,
        *,
        scripted_responses: Optional[Dict[str, Dict[str, Any]]] = None,
        default_response: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._scripted = scripted_responses or {}
        self._default = default_response or {
            "content": "MOCK_RESPONSE",
            "tool_calls": None,
            "input_tokens": 150,
            "output_tokens": 25,
            "cached_tokens": 0,
        }
        self.invocation_count: int = 0
        self.last_request: Optional[Dict[str, Any]] = None
        # Nested objects so client.chat.completions.create(...) works.
        self.chat = _MockChat(self)


class _MockChat:
    """Inner namespace exposing ``.completions.create``."""

    def __init__(self, parent: MockAsyncOpenAI) -> None:
        self.completions = _MockCompletions(parent)


class _MockCompletions:
    """Async ``create`` method mirroring ``openai.resources.chat.AsyncCompletions``."""

    def __init__(self, parent: MockAsyncOpenAI) -> None:
        self._parent = parent

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        """Return a canned ChatCompletion-shaped object.

        Args:
            **kwargs: ``model``, ``messages``, ``tools``, ``temperature``,
                ``seed`` etc.  Only ``model``, ``messages``, ``tools`` are
                used for response selection.

        Returns:
            SimpleNamespace shaped like an OpenAI ChatCompletion.
        """
        self._parent.invocation_count += 1
        self._parent.last_request = dict(kwargs)

        model: str = kwargs.get("model", "unknown")
        messages: list[dict] = kwargs.get("messages", [])
        tools: Optional[list[dict]] = kwargs.get("tools")
        key = _hash_request(model, messages, tools)

        spec = self._parent._scripted.get(key, self._parent._default)
        return _build_openai_response(
            model=model,
            content=spec.get("content"),
            tool_calls=spec.get("tool_calls"),
            input_tokens=int(spec.get("input_tokens", 150)),
            output_tokens=int(spec.get("output_tokens", 25)),
            cached_tokens=int(spec.get("cached_tokens", 0)),
        )


# ---------------------------------------------------------------------------
# MockAsyncAnthropic — minimal stub used by AnthropicClient
# ---------------------------------------------------------------------------


class MockAsyncAnthropic:
    """Drop-in replacement for ``anthropic.AsyncAnthropic`` (minimal surface).

    Only the ``messages.create`` method is modelled.  This is sufficient for
    the E2E pipeline test since the FitText harness routes Claude calls
    through ``AnthropicClient.chat_completion`` which calls ``messages.create``
    exactly once per LLM operation.

    Args:
        default_response: Dict with optional keys ``text`` (str),
            ``input_tokens`` (int), ``output_tokens`` (int),
            ``stop_reason`` (str), ``tool_use`` (list[dict] | None).
    """

    def __init__(self, *, default_response: Optional[Dict[str, Any]] = None) -> None:
        self._default = default_response or {
            "text": "MOCK_RESPONSE",
            "input_tokens": 150,
            "output_tokens": 25,
            "stop_reason": "end_turn",
            "tool_use": None,
        }
        self.invocation_count: int = 0
        self.last_request: Optional[Dict[str, Any]] = None
        self.messages = _MockAnthropicMessages(self)


class _MockAnthropicMessages:
    """Async ``create`` mirroring ``anthropic.resources.messages.AsyncMessages``."""

    def __init__(self, parent: MockAsyncAnthropic) -> None:
        self._parent = parent

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        """Return an Anthropic-shaped Message SimpleNamespace.

        Args:
            **kwargs: Anthropic API parameters (``model``, ``messages``, ...).

        Returns:
            SimpleNamespace with ``.content``, ``.usage``, ``.stop_reason``.
        """
        self._parent.invocation_count += 1
        self._parent.last_request = dict(kwargs)

        spec = self._parent._default
        text = spec.get("text", "MOCK_RESPONSE")
        content_blocks = [SimpleNamespace(type="text", text=text)]
        if spec.get("tool_use"):
            for tu in spec["tool_use"]:
                content_blocks.append(
                    SimpleNamespace(
                        type="tool_use",
                        id=tu.get("id", f"toolu_{uuid.uuid4().hex[:8]}"),
                        name=tu["name"],
                        input=tu.get("input", {}),
                    )
                )

        return SimpleNamespace(
            id=f"msg_{uuid.uuid4().hex[:12]}",
            type="message",
            role="assistant",
            model=kwargs.get("model", "unknown"),
            content=content_blocks,
            stop_reason=spec.get("stop_reason", "end_turn"),
            usage=SimpleNamespace(
                input_tokens=int(spec.get("input_tokens", 150)),
                output_tokens=int(spec.get("output_tokens", 25)),
                cache_read_input_tokens=int(spec.get("cached_tokens", 0)),
                cache_creation_input_tokens=0,
            ),
        )
