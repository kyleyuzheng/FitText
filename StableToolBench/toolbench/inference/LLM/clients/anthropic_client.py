"""
Anthropic chat client with mandatory prompt caching.

Handles models with prefix: claude-*.
Uses the native ``anthropic.AsyncAnthropic`` SDK (NOT the OpenAI-compat shim).

Prompt caching strategy
-----------------------
FitText's DFSDT loop repeats the same system prompt and tool catalog on every
node expansion.  Anthropic's prompt caching gives ~90% discount on cache hits.

We inject ``cache_control: {"type": "ephemeral"}`` in two places:
1. The last (or only) system-prompt block.
2. The last tool definition (so the whole tool catalog is cached as a prefix).

The SDK top-level ``cache_control`` param automatically marks the last
cacheable block — that covers the tools list.  For the system prompt we pass
it as a ``TextBlockParam`` with ``cache_control`` set explicitly.

Usage telemetry
---------------
- ``usage.input_tokens``            → non-cached prompt tokens billed at full rate
- ``usage.cache_read_input_tokens`` → tokens served from cache (~10% cost)
- ``usage.cache_creation_input_tokens`` → tokens written to cache (1.25× first time)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import anthropic
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .adapters import (
    anthropic_response_to_normalized,
    messages_openai_to_anthropic,
    openai_tools_to_anthropic,
)
from .base import ModelClient, NormalizedResponse

# Module-level client; constructed once so HTTP connections are reused.
_client: Optional[anthropic.AsyncAnthropic] = None

# Default max_tokens for claude-* models.  FitText prompts are short so 8K
# is plenty, and it avoids inadvertent large-output billing.
_DEFAULT_MAX_TOKENS = 8192


def _get_anthropic_client() -> anthropic.AsyncAnthropic:
    """
    Return the module-level ``AsyncAnthropic`` instance, creating it on first call.

    API key is read from the ``ANTHROPIC_API_KEY`` environment variable.

    Returns
    -------
    anthropic.AsyncAnthropic
    """
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )
    return _client


class AnthropicClient(ModelClient):
    """
    ModelClient implementation for Anthropic Claude (claude-*).

    Parameters
    ----------
    model : str
        Full dated model tag, e.g. ``"claude-sonnet-4-6"``.
    api_key : str | None
        Override for the API key; falls back to ``ANTHROPIC_API_KEY`` env var.
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
        if api_key:
            self._client: Optional[anthropic.AsyncAnthropic] = anthropic.AsyncAnthropic(
                api_key=api_key
            )
        else:
            self._client = None  # resolved lazily

    def _resolve_client(self) -> anthropic.AsyncAnthropic:
        """Return the per-instance or module-level client."""
        return self._client if self._client is not None else _get_anthropic_client()

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
        seed: int = 42,  # Anthropic does not support seed; accepted for API compat
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> NormalizedResponse:
        """
        Call the Anthropic Messages API with prompt caching enabled.

        The ``seed`` parameter is accepted for interface compatibility but is
        not forwarded to Anthropic (they do not support it).

        Parameters
        ----------
        messages : list[dict]
            OpenAI-format conversation history (will be converted internally).
        tools : list[dict] | None
            OpenAI-format tool definitions (will be converted internally).
        temperature : float
            Sampling temperature; 0.0 for greedy.
        seed : int
            Ignored by this client (Anthropic does not support seed).
        max_tokens : int | None
            Max completion tokens; defaults to ``_DEFAULT_MAX_TOKENS`` (8192).
        **kwargs
            Extra parameters forwarded to ``messages.create()``.

        Returns
        -------
        NormalizedResponse
        """
        client = self._resolve_client()
        t0 = self._now_ms()

        # Convert OpenAI message history → Anthropic format.
        # Also extracts any "system" role messages into system_prompt.
        anthropic_messages, inferred_system = messages_openai_to_anthropic(messages)

        # Build the system parameter as a list of TextBlockParam so we can
        # attach cache_control to the last block (mandatory for caching).
        system_param: Any
        if inferred_system:
            system_param = [
                {
                    "type": "text",
                    "text": inferred_system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        else:
            # No system prompt — omit entirely to avoid an empty block.
            system_param = anthropic.NOT_GIVEN

        # Convert OpenAI tool defs → Anthropic format.
        anthropic_tools: Optional[List[Dict[str, Any]]] = None
        if tools:
            anthropic_tools = openai_tools_to_anthropic(tools)

        create_kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or _DEFAULT_MAX_TOKENS,
            "messages": anthropic_messages,
            "temperature": temperature,
            **kwargs,
        }

        if system_param is not anthropic.NOT_GIVEN:
            create_kwargs["system"] = system_param

        if anthropic_tools:
            create_kwargs["tools"] = anthropic_tools
            # Top-level cache_control: marks the last cacheable block in the
            # request (the tool catalog) as ephemeral.  This is the Anthropic
            # SDK shorthand for injecting cache_control on the last tool entry.
            create_kwargs["cache_control"] = {"type": "ephemeral"}

        response = await client.messages.create(**create_kwargs)
        latency_ms = self._now_ms() - t0

        return anthropic_response_to_normalized(
            response=response,
            latency_ms=latency_ms,
            model=self.model,
        )
