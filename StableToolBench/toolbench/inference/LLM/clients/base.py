"""
ModelClient ABC and NormalizedResponse dataclass.

All concrete clients return NormalizedResponse so that callers (DFSDT, etc.)
can consume an OpenAI-shaped dict regardless of the underlying provider.

Cache integration
-----------------
When a ``ResponseCache`` is injected via ``ModelClient.__init__(cache=...)``,
the protected helper ``_chat_completion_with_cache`` wraps the provider-
specific ``chat_completion`` with a get-before / put-after pattern.

Concrete subclasses **must** call ``_chat_completion_with_cache`` in their
``chat_completion`` override (or leave ``chat_completion`` as a direct
delegate) to get cache behaviour.  The default implementation in this base
class routes through ``_chat_completion_with_cache`` when a cache is present.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from toolbench.inference.cache import CacheKey, ResponseCache


@dataclass
class NormalizedResponse:
    """
    Provider-agnostic response container with OpenAI-shaped accessors.

    Fields
    ------
    content : str | None
        Text content from the assistant turn (may be None when the model only
        calls tools).
    tool_calls : list[dict]
        OpenAI-shaped tool_calls list:
        ``[{"id": ..., "type": "function", "function": {"name": ..., "arguments": <json-str>}}]``
    finish_reason : str | None
        "stop", "tool_calls", "length", etc.
    model_revision : str
        Exact model identifier echoed back by the provider (e.g. dated slug).
    input_tokens : int
        Total non-cached prompt tokens billed.
    cached_input_tokens : int
        Tokens served from the provider's prompt cache (saves cost).
    output_tokens : int
        Completion tokens.
    latency_ms : float
        Wall-clock time for the API call in milliseconds.
    provider : str
        One of "openai", "anthropic", "vllm".
    raw_response : dict
        Full response payload (serialised to dict) for provenance hashing.
    """

    content: Optional[str]
    tool_calls: List[Dict[str, Any]]
    finish_reason: Optional[str]
    model_revision: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    latency_ms: float
    provider: str
    raw_response: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Convenience: expose an OpenAI-shaped choices[0].message dict so that #
    # legacy callers that do response["choices"][0]["message"] still work.  #
    # ------------------------------------------------------------------ #

    def to_openai_dict(self) -> Dict[str, Any]:
        """
        Return the response as an OpenAI ChatCompletion-shaped dict.

        Returns
        -------
        dict
            Compatible with ``response["choices"][0]["message"]`` access
            patterns used throughout DFSDT.
        """
        message: Dict[str, Any] = {
            "role": "assistant",
            "content": self.content,
        }
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls

        return {
            "id": self.raw_response.get("id", ""),
            "object": "chat.completion",
            "model": self.model_revision,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": self.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": self.input_tokens,
                "completion_tokens": self.output_tokens,
                "total_tokens": self.input_tokens + self.output_tokens,
                "prompt_tokens_details": {
                    "cached_tokens": self.cached_input_tokens,
                },
            },
        }


class ModelClient(abc.ABC):
    """
    Abstract base class for all LLM provider clients.

    Subclasses implement ``chat_completion`` for a specific provider.
    The method is async; use ``asyncio.run()`` or an event loop to call it
    from synchronous code (see each client's ``chat_completion_sync`` helper).

    Cache integration
    -----------------
    Pass a ``ResponseCache`` instance at construction time to enable
    transparent request/response caching::

        from toolbench.inference.cache import ResponseCache
        cache = ResponseCache(Path("runs/example/cache"))
        client = make_client(model_id, cache=cache)

    When ``cache`` is set, ``chat_completion`` automatically checks for a
    cached response before calling the provider and stores the result on a
    miss.  To bypass the cache for a specific call, use the concrete
    provider method directly (not recommended outside tests).
    """

    def __init__(
        self,
        model: str,
        cache: Optional["ResponseCache"] = None,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        model : str
            Full model identifier (e.g. "gpt-4.1-mini-2025-04-14").
        cache : ResponseCache | None
            Optional disk cache.  When provided, ``chat_completion`` checks
            the cache before making a provider round-trip and stores the
            result on a miss.  Defaults to ``None`` (no caching).
        **kwargs
            Provider-specific overrides forwarded by the factory.
        """
        self.model = model
        self._cache: Optional["ResponseCache"] = cache

    @abc.abstractmethod
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
        Make a chat completion request and return a NormalizedResponse.

        Parameters
        ----------
        messages : list[dict]
            OpenAI-format conversation history (role + content dicts).
        tools : list[dict] | None
            OpenAI-format tool definitions (``{"type":"function","function":{...}}``).
        temperature : float
            Sampling temperature; 0.0 for greedy.
        seed : int
            Reproducibility seed (ignored where unsupported).
        max_tokens : int | None
            Maximum completion tokens; provider default used when None.
        **kwargs
            Extra provider parameters forwarded verbatim.

        Returns
        -------
        NormalizedResponse
        """

    async def _chat_completion_with_cache(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
        seed: int = 42,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> NormalizedResponse:
        """Wrap the abstract provider call with cache get/put.

        If ``self._cache`` is set, this method:
        1. Computes a ``CacheKey`` from the request parameters.
        2. Returns the cached ``NormalizedResponse`` on a hit (no provider
           round-trip).
        3. On a miss, delegates to the provider's ``chat_completion``, stores
           the result, and returns it.

        When ``self._cache`` is ``None``, this is a direct pass-through to
        ``chat_completion``.

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
            Maximum completion tokens.
        **kwargs
            Extra provider parameters.

        Returns
        -------
        NormalizedResponse
        """
        if self._cache is None:
            return await self.chat_completion(
                messages=messages,
                tools=tools,
                temperature=temperature,
                seed=seed,
                max_tokens=max_tokens,
                **kwargs,
            )

        from toolbench.inference.cache import compute_cache_key

        # Build the key — only deterministic kwargs go into extra_hash.
        # max_tokens affects the *output* and is therefore deterministic; other
        # provider-internal flags (timeouts, retries) should not be passed here.
        key = compute_cache_key(
            model=self.model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            seed=seed,
            top_p=kwargs.get("top_p"),
            **({"max_tokens": max_tokens} if max_tokens is not None else {}),
        )

        cached = self._cache.get(key)
        if cached is not None:
            return cached

        # Cache miss — call the provider
        response = await self.chat_completion(
            messages=messages,
            tools=tools,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
            **kwargs,
        )
        self._cache.put(key, response)
        return response

    def chat_completion_sync(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
        seed: int = 42,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> NormalizedResponse:
        """
        Synchronous wrapper around ``_chat_completion_with_cache``.

        Uses ``asyncio.run()`` — do not call from within a running event loop.
        For async callers, use ``await client._chat_completion_with_cache(...)``
        (or wire the cache-aware path directly).

        Parameters
        ----------
        (same as ``chat_completion``)

        Returns
        -------
        NormalizedResponse
        """
        import asyncio

        return asyncio.run(
            self._chat_completion_with_cache(
                messages=messages,
                tools=tools,
                temperature=temperature,
                seed=seed,
                max_tokens=max_tokens,
                **kwargs,
            )
        )

    # ------------------------------------------------------------------ #
    # Timing helper shared by all subclasses.                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _now_ms() -> float:
        """Return current time in milliseconds."""
        return time.monotonic() * 1000.0
