"""
Just-Query — zero-retrieval floor baseline.

Asks the LLM the query directly with no retrieval, no tool catalog, no
critic. Measures the floor of what the agent's prior knowledge alone can
solve. Every retrieval method evaluated above this technique (single_pass,
multi_turn, scattershot, memetic, less_is_more, reinvoke, xu2024, colt) is
judged against this floor.

Algorithm
---------
1. One LLM call:
     system = "Answer directly. Do not call any tools."
     user   = "Answer the following question: {query}"
2. Return RetrievedTools(tool_ids=[], scores=[], metadata={"answer": ...}).

Why the empty tool list is the point
------------------------------------
The technique is deliberately zero-retrieval. The downstream judge can
either:
  - score the LLM's direct answer (metadata.answer) as the model's solution
    attempt, or
  - record a zero pass-rate for tool-grounded evaluation, which is the
    correct floor reading for a tool-retrieval benchmark.

Both readings provide exactly the same thing: a single,
unambiguous floor comparator.
"""

from __future__ import annotations

import logging
import time
from typing import Any, ClassVar

from .base import Baseline, RetrievedTools
from StableToolBench.toolbench.inference.LLM.clients import ModelClient

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "Answer the user's question directly to the best of your knowledge. "
    "Do not call any tools."
)

_USER_PROMPT_TEMPLATE = "Answer the following question: {query}"


class JustQueryBaseline(Baseline):
    """Zero-retrieval floor — single LLM call, no tool catalog, no retriever.

    The ``retriever`` argument is accepted for interface conformance but is
    never called. ``top_k`` is similarly ignored; the result always has
    empty ``tool_ids`` and empty ``scores``. The LLM's direct answer is
    surfaced via ``metadata.answer`` so downstream judges can score it.

    Config kwargs (read from ``self.cfg``)
    --------------------------------------
    temperature : float, default 0.0
    seed        : int,   default 42
    max_tokens  : int,   default 2048
    """

    name: ClassVar[str] = "just_query"

    def __init__(
        self,
        model_client: ModelClient,
        retriever: Any,
        *,
        top_k: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_client, retriever, top_k=top_k, **kwargs)

    async def retrieve(self, query: str, *, tool_catalog: list[dict]) -> RetrievedTools:
        """Single LLM call; no retrieval, no tool catalog.

        Args:
            query: User task query.
            tool_catalog: Ignored — this technique does not look at tools.

        Returns:
            RetrievedTools with empty tool_ids/scores. ``metadata.answer``
            carries the LLM's direct response. ``metadata.latency_ms`` and
            ``metadata.strategy`` are recorded for manifest provenance.
        """
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _USER_PROMPT_TEMPLATE.format(query=query)},
        ]

        t0 = time.monotonic()
        response = await self.model_client.chat_completion(
            messages=messages,
            temperature=float(self.cfg.get("temperature", 0.0)),
            seed=int(self.cfg.get("seed", 42)),
            max_tokens=int(self.cfg.get("max_tokens", 2048)),
        )
        latency_ms = (time.monotonic() - t0) * 1000.0
        answer = response.content or ""

        logger.debug(
            "just_query LLM response (%.0f ms): %s",
            latency_ms,
            answer[:200],
        )

        # Empty tool_ids — this is the technique's defining feature.
        return RetrievedTools(
            tool_ids=[],
            scores=[],
            metadata={
                "strategy": "just_query",
                "answer": answer,
                "latency_ms": latency_ms,
                "n_catalog_tools_offered": len(tool_catalog),  # for sanity logging only
            },
        )
