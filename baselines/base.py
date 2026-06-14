"""
Abstract base class and shared data types for FitText retrieval baselines.

Design decisions
----------------
- ``Retriever`` is a thin ``Protocol`` so both ``ToolRetriever`` (local
  sentence-transformer) and ``RemoteToolRetriever`` (retriever server) can be
  passed in without sub-classing.  Callers wrap the existing retriever by
  providing a small shim adapter (see ``RetrieverAdapter`` below).

- ``RetrievedTools`` mirrors the output of ``ToolRetriever.retrieving()``:
  ``tool_ids`` are ``"category::tool_name::api_name"`` composite keys, and
  ``scores`` are cosine-similarity values.  Baseline-specific state lives in
  ``metadata``.

- ``Baseline.retrieve`` is ``async`` so baselines can call the LLM concurrently
  at the same async event loop used by FitText's own strategies.

- All hyperparameters flow in via ``**kwargs`` stored in ``self.cfg`` — no
  hard-coded numbers anywhere in this file.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, runtime_checkable

from StableToolBench.toolbench.inference.LLM.clients import ModelClient


# ---------------------------------------------------------------------------
# Retriever protocol — thin wrapper over ToolRetriever / RemoteToolRetriever
# ---------------------------------------------------------------------------

@runtime_checkable
class Retriever(Protocol):
    """Thin protocol that both ``ToolRetriever`` and ``RemoteToolRetriever`` satisfy.

    The only method the baselines call.  Callers should wrap the existing
    retriever via ``RetrieverAdapter`` if it does not already conform.
    """

    def retrieving(
        self,
        query: str,
        top_k: int = 5,
        excluded_tools: dict[str, list[str]] | None = None,
    ) -> list[dict[str, Any]]:
        """Return a list of tool dicts ranked by score.

        Each dict contains at minimum:
            category (str), tool_name (str), api_name (str), score (float).
        """
        ...


class RetrieverAdapter:
    """Adapts any object with a ``.retrieving()`` method to the ``Retriever`` protocol.

    Also provides a convenience ``retrieve()`` method that returns
    ``list[(tool_id, score)]`` tuples, which the baselines use internally.

    Args:
        inner: Existing retriever (``ToolRetriever`` or ``RemoteToolRetriever``).
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def retrieving(
        self,
        query: str,
        top_k: int = 5,
        excluded_tools: dict[str, list[str]] | None = None,
    ) -> list[dict[str, Any]]:
        """Delegate to the wrapped retriever."""
        return self._inner.retrieving(query, top_k=top_k, excluded_tools=excluded_tools or {})

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Return ``[(tool_id, score), ...]`` ranked by score.

        ``tool_id`` is the composite ``"category::tool_name::api_name"`` key.

        Args:
            query: Natural-language query or belief string.
            top_k: Maximum number of results to return.

        Returns:
            Ranked list of (tool_id, score) pairs.
        """
        hits = self.retrieving(query, top_k=top_k)
        result: list[tuple[str, float]] = []
        for hit in hits:
            tool_id = f"{hit.get('category', '')}::{hit.get('tool_name', '')}::{hit.get('api_name', '')}"
            score = float(hit.get("score", 0.0))
            result.append((tool_id, score))
        return result


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class RetrievedTools:
    """Output of a baseline ``retrieve()`` call.

    Attributes
    ----------
    tool_ids : list[str]
        Ranked list of ``"category::tool_name::api_name"`` identifiers,
        length <= ``top_k``.  Index 0 is the most relevant.
    scores : list[float]
        Relevance scores parallel to ``tool_ids``.
    metadata : dict
        Baseline-specific provenance (e.g. intermediate beliefs, intent
        expansions, iteration counts).  Keys are baseline-defined.
    """

    tool_ids: list[str]
    scores: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.tool_ids) != len(self.scores):
            raise ValueError(
                f"tool_ids length ({len(self.tool_ids)}) != scores length ({len(self.scores)})"
            )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class Baseline(abc.ABC):
    """Drop-in retrieval baseline conforming to FitText's eval harness contract.

    All subclasses must implement ``retrieve()``.  The ``name`` class variable
    is used to group manifest entries (``variant: baseline_<name>``).

    Args:
        model_client: Async LLM client (from Wave 1 ``make_client()``).
        retriever: Wrapped retriever that conforms to ``Retriever`` protocol.
        top_k: Maximum tools to return. All hyperparameters beyond ``top_k``
               are passed as ``**kwargs`` and stored in ``self.cfg``.
    """

    name: ClassVar[str]  # e.g. "less_is_more", "reinvoke", "xu2024", "colt"

    def __init__(
        self,
        model_client: ModelClient,
        retriever: "Retriever | RetrieverAdapter",
        *,
        top_k: int = 5,
        **kwargs: Any,
    ) -> None:
        self.model_client = model_client
        # Ensure we always have a RetrieverAdapter with the .retrieve() helper.
        if isinstance(retriever, RetrieverAdapter):
            self.retriever = retriever
        else:
            self.retriever = RetrieverAdapter(retriever)
        self.top_k = top_k
        # All remaining hyperparameters stored config-side — no hardcoding.
        self.cfg: dict[str, Any] = kwargs

    @property
    def variant_tag(self) -> str:
        """Manifest variant string (prefixed for grouping in cost tables)."""
        return f"baseline_{self.name}"

    @abc.abstractmethod
    async def retrieve(self, query: str, *, tool_catalog: list[dict]) -> RetrievedTools:
        """Retrieve relevant tools for ``query``.

        Args:
            query: The user's natural-language task query.
            tool_catalog: Full list of available tool dicts (each with at
                minimum ``name``, ``description`` fields as returned by the
                FitText harness).

        Returns:
            ``RetrievedTools`` with up to ``self.top_k`` ranked results.
        """
