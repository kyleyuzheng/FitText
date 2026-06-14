"""
ModelClient factory.

Single source of truth for model-prefix → client routing.
All callers (chatgpt_function_model.py, Toolret/strategy/LLM_model.py, etc.)
should go through ``make_client`` instead of maintaining their own routing logic.
"""

from __future__ import annotations

from typing import Any, Optional

from .base import ModelClient
from .openai_client import OpenAIClient
from .vllm_client import VLLMClient


def make_client(
    model: str,
    *,
    cache_dir: Optional[str] = None,
    **kwargs: Any,
) -> ModelClient:
    """
    Instantiate and return the correct ``ModelClient`` for the given model.

    Routing rules (first match wins):
    - ``claude-*``                         → ``AnthropicClient``
    - ``gpt-*``, ``o*``, ``chatgpt-*``     → ``OpenAIClient``
    - ``Qwen*``, ``qwen*``, ``DeepSeek*``, ``deepseek*``,
      ``google/gemma*``, ``gemma*``, ``Gemma*`` → ``VLLMClient``

    Parameters
    ----------
    model : str
        Full model identifier, e.g. ``"claude-sonnet-4-6"`` or
        ``"gpt-4.1-mini-2025-04-14"``.
    cache_dir : str | None
        Directory for disk-backed response cache (§5.3).  When set, a
        ``ResponseCache`` is constructed and injected into the client so
        that repeat requests with identical parameters are served from disk.
        ``None`` (default) means no caching — experiments opt in via
        ``cfg.infra.cache_dir``.
    **kwargs
        Forwarded verbatim to the chosen client constructor.  Useful kwargs
        include ``api_key``, ``base_url`` (vLLM override).

    Returns
    -------
    ModelClient

    Raises
    ------
    NotImplementedError
        If no routing rule matches the model prefix.
    """
    cache = None
    if cache_dir is not None:
        from pathlib import Path

        from toolbench.inference.cache import ResponseCache

        cache = ResponseCache(Path(cache_dir), enabled=True)

    if model.startswith("claude"):
        # Lazy import: the anthropic package is only required when a claude-*
        # model is actually requested.
        from .anthropic_client import AnthropicClient

        return AnthropicClient(model=model, cache=cache, **kwargs)

    if model.startswith(("gpt", "o", "chatgpt")):
        return OpenAIClient(model=model, cache=cache, **kwargs)

    if model.startswith(("Qwen", "qwen", "DeepSeek", "deepseek",
                          "google/gemma", "gemma", "Gemma")):
        return VLLMClient(model=model, cache=cache, **kwargs)

    raise NotImplementedError(
        f"No ModelClient registered for model prefix: {model!r}. "
        "Add a routing rule in factory.py."
    )
