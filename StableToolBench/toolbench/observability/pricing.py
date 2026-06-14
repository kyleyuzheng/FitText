"""
API pricing table for all supported model providers.

Rates are USD per 1,000,000 tokens (input / cached_input / output).
Resolve model names by longest-prefix match so dated tags
(e.g. ``gpt-4.1-mini-2025-04-14``) automatically map to their base key.

Sources (checked 2026-05-24):
  - OpenAI:    https://openai.com/api/pricing/
  - Anthropic: https://www.anthropic.com/pricing
  - vLLM/local: no monetary cost; tokens tracked for fairness comparisons.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate table — USD per 1M tokens
# ---------------------------------------------------------------------------
# Each entry has three keys:
#   "input"        — non-cached prompt tokens
#   "cached_input" — prompt tokens served from the provider's prompt cache
#   "output"       — completion tokens
#
# Rates should be refreshed from provider pricing pages before publication-critical cost reporting.

PRICING: dict[str, dict[str, float]] = {
    # ------------------------------------------------------------------
    # OpenAI
    # ------------------------------------------------------------------
    "gpt-4.1-mini": {
        "input": 0.40,
        "cached_input": 0.10,
        "output": 1.60,
    },
    "gpt-4.1": {
        "input": 2.00,
        "cached_input": 0.50,
        "output": 8.00,
    },
    "gpt-5": {
        "input": 1.25,
        "cached_input": 0.125,
        "output": 10.00,
    },
    # ----- GPT-5.4 family — verified 2026-05-24 -----
    # Source: https://developers.openai.com/api/docs/pricing
    # Released 2026-03-17; dated snapshot tags: gpt-5.4-mini-2026-03-17, gpt-5.4-nano-2026-03-17
    "gpt-5.4-mini": {
        "input": 0.75,         # verified 2026-05-24 from https://developers.openai.com/api/docs/pricing
        "cached_input": 0.075, # verified 2026-05-24
        "output": 4.50,        # verified 2026-05-24
    },
    "gpt-5.4-nano": {
        "input": 0.20,         # verified 2026-05-24 from https://developers.openai.com/api/docs/pricing
        "cached_input": 0.02,  # verified 2026-05-24
        "output": 1.25,        # verified 2026-05-24
    },
    "gpt-5.4": {
        "input": 2.50,         # verified 2026-05-24 from https://developers.openai.com/api/docs/pricing
        "cached_input": 0.25,  # verified 2026-05-24
        "output": 15.00,       # verified 2026-05-24
    },
    "gpt-5.5": {
        "input": 5.00,         # verified 2026-05-24 from https://developers.openai.com/api/docs/pricing
        "cached_input": 0.50,  # verified 2026-05-24
        "output": 30.00,       # verified 2026-05-24
    },
    "o3": {
        "input": 10.00,
        "cached_input": 2.50,
        "output": 40.00,
    },
    "o4-mini": {
        "input": 1.10,
        "cached_input": 0.275,
        "output": 4.40,
    },
    # ------------------------------------------------------------------
    # Anthropic
    # ------------------------------------------------------------------
    "claude-sonnet-4-6": {
        "input": 3.00,
        "cached_input": 0.30,
        "output": 15.00,
    },
    "claude-haiku-4": {
        "input": 0.80,
        "cached_input": 0.08,
        "output": 4.00,
    },
    "claude-opus-4-7": {
        "input": 15.00,
        "cached_input": 1.50,
        "output": 75.00,
    },
    # ------------------------------------------------------------------
    # vLLM / local — zero monetary cost; token counts still tracked for
    # fairness comparisons (NDCG-per-token, etc.).
    # ------------------------------------------------------------------
    "Qwen3-30B": {
        "input": 0.0,
        "cached_input": 0.0,
        "output": 0.0,
    },
    "Qwen3-Max": {
        "input": 0.0,
        "cached_input": 0.0,
        "output": 0.0,
    },
    "Qwen3-0.6B": {
        "input": 0.0,
        "cached_input": 0.0,
        "output": 0.0,
    },
    "Qwen3.5-0.8B": {
        "input": 0.0,
        "cached_input": 0.0,
        "output": 0.0,
    },
    "DeepSeek-V3": {
        "input": 0.0,
        "cached_input": 0.0,
        "output": 0.0,
    },
}

# Sorted by descending key length for longest-prefix matching
_SORTED_KEYS: list[str] = sorted(PRICING.keys(), key=len, reverse=True)


def resolve_model_key(model: str) -> str | None:
    """Return the pricing-table key that is a prefix of *model*.

    Tries longest prefix first so ``gpt-4.1-mini-2025-04-14`` maps to
    ``gpt-4.1-mini`` (not ``gpt-4.1``).

    Returns ``None`` if no prefix matches and logs a warning.
    """
    for key in _SORTED_KEYS:
        if model.startswith(key):
            return key
    logger.warning(
        "No pricing entry found for model %r. Cost will be recorded as 0.0 USD. "
        "Add an entry to pricing.PRICING if this model incurs real cost.",
        model,
    )
    return None


def get_rates(model: str) -> dict[str, float]:
    """Return the rate dict ``{input, cached_input, output}`` for *model*.

    Falls back to zero rates when the model is unknown (logged as a warning).

    Args:
        model: Model identifier, possibly with a dated suffix.

    Returns:
        Dict with keys ``"input"``, ``"cached_input"``, ``"output"`` in
        USD per 1 M tokens.
    """
    key = resolve_model_key(model)
    if key is None:
        return {"input": 0.0, "cached_input": 0.0, "output": 0.0}
    return PRICING[key]


def compute_cost(
    model: str,
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
) -> float:
    """Compute the USD cost for one API call.

    Non-cached prompt tokens are billed at the full ``input`` rate.
    ``cached_input_tokens`` are the *already-cached* subset and billed at
    the ``cached_input`` (discounted) rate.  They are **not** double-counted
    against ``input_tokens`` — the caller must pass the split correctly:

      * ``input_tokens`` = total non-cached prompt tokens
      * ``cached_input_tokens`` = tokens served from provider cache

    Args:
        model: Model identifier (dated tags supported via prefix matching).
        input_tokens: Non-cached prompt token count.
        cached_input_tokens: Cached prompt token count (provider discount).
        output_tokens: Completion token count.

    Returns:
        Total cost in USD for this call.
    """
    rates = get_rates(model)
    per_m = 1_000_000.0
    cost = (
        input_tokens * rates["input"] / per_m
        + cached_input_tokens * rates["cached_input"] / per_m
        + output_tokens * rates["output"] / per_m
    )
    return cost
