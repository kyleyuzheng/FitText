"""Less-is-More baseline (Paramanayakam et al., arXiv 2411.15399).

Paper: "Less is More: Optimizing Function Calling for LLM Execution on Edge
Devices" (Paramanayakam, Karatzas, Anagnostopoulos, Stamoulis; arXiv
2411.15399, Nov 2024).

Relation to FitText: **Single-Pass degenerate case** of Memetic Retrieval
(population_size=1, generations=1, no revision, no memory).  Single pseudo-tool
description is generated, embedded, and used as the retrieval probe.

Spec (paper §III.A-§III.C, paraphrased)
---------------------------------------
Offline (§III-A — "Constructing the Search Levels"):
    Build three search spaces over the tool catalog:
      Level 1 ("Individual tools"): embed each tool description via MPNet into
            a 768-dim latent space ``T~``.
      Level 2 ("Tool clusters"): generate augmented benchmark queries with
            GPT-4-Turbo (one-time, offline), embed them via MPNet into ``A~``,
            then apply Agglomerative Clustering to obtain coarse "tool group"
            embeddings.  Captures synergistic tool combinations.
      Level 3 ("Entire tool set"): full catalog, no embedding -- fallback only.

Online (§III-B — "Tool Recommender"):
    1. Prompt the LLM (Tool Recommender) to generate descriptions of the
       "ideal" tools needed for the user query.  Returns JSON with a
       "functionality" field per recommended pseudo-tool.  No catalog names
       in the prompt -- the LLM must invent the tool description from scratch.
    2. Embed (a) the user query and (b) each LLM-generated pseudo-tool
       description with the same MPNet model into ``E~``.

Online (§III-C — "Tool Controller"):
    3. For each pseudo-tool embedding e in E~, run k-NN search (FAISS, cosine)
       against Level 1 (T~) and Level 2 (A~).
    4. Compute avg top-k similarity per level.  Select the Search Level with
       the higher avg score.  If *both* level-1 and level-2 avg scores are
       < 0.5, fall back to Level 3 (entire tool set) -- low-confidence guard.
    5. Return the combined top-k tools from the winning level as the
       retrieval result.

Design decisions in this implementation
---------------------------------------
- This is a **training-free, prompt-based** baseline.  We follow the §III-B
  pseudo-tool *description* generation path (not name listing).  This is
  what makes Less-is-More the empirical degeneracy partner of FitText's
  Single-Pass variant.
- L2 (clusters) needs an offline clustering pass over augmented benchmark
  queries.  When ``enable_l2_clusters=False`` (default), only L1 and L3 are
  used -- per the paper, L1 already wins for ToolBench-style benchmarks and
  L2 only helps on GeoEngine.  Set ``enable_l2_clusters=True`` and provide
  a pre-built ``cluster_index`` in ``cfg`` to activate.
- We reuse the retriever's underlying encoder via ``encode_sentence`` for
  the embedding step (T~ is the retriever's own tool index; that index is
  shared across the run by §5.7 instrumentation).
- The Search-Level selector compares **avg top-k cosine of the pseudo-tool
  embedding vs catalog** -- when L2 is disabled, we trivially pick L1
  unless its avg < 0.5, in which case we fall back to L3 (raw query
  retrieval as a proxy for "show all tools").

Verified faithfulness against paper: 2026-05-24 (baseline-verify track).
See ``baselines/REPRODUCTION.md`` for the deviation log and
``tests/test_baseline_reproduction.py`` for the spot-check.

Deviations from the paper (documented in REPRODUCTION.md):
- L2 (cluster index) is opt-in via ``enable_l2_clusters``; default off
  because building the cluster index requires an offline GPT-4 augmentation
  pass that is itself a separate experiment (and orthogonal for the
  Single-Pass degeneracy claim).
- The paper uses a 768-dim MPNet encoder; this implementation reuses
  whatever encoder the FitText retriever was initialized with (see
  ``configs/embedder.yaml``).  Choice of encoder does not change the
  algorithm.
"""

from __future__ import annotations

import logging
import time
from typing import Any, ClassVar

from .base import Baseline, RetrievedTools, RetrieverAdapter
from StableToolBench.toolbench.inference.LLM.clients import ModelClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template (paper §III-B "Tool Recommender")
# ---------------------------------------------------------------------------

# The paper instructs the LLM to generate "ideal" tool descriptions in JSON.
# We allow either JSON or free-form text; both are handled in parsing.
_SYSTEM_PROMPT = (
    "You are a Tool Recommender. Given a user query, describe the ideal "
    "tools (APIs) that would be necessary to complete the task. For each "
    "tool, provide a short natural-language description of its functionality "
    "(what it takes as input, what it returns). Do NOT name specific real "
    "tools -- describe the *capability* you need. List at most {top_k} "
    "ideal tool descriptions, one per line. No numbering, no preamble."
)

_USER_PROMPT_TEMPLATE = (
    "User query: {query}\n\n"
    "Describe the ideal tools needed to solve this query:"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_pseudo_tools(text: str, max_count: int) -> list[str]:
    """Parse LLM output into a list of pseudo-tool description strings.

    The Tool Recommender prompt asks for one pseudo-tool description per line
    in free text.  This helper also tolerates JSON-list outputs (the paper
    mentions JSON formatting in §III-B).

    Args:
        text: Raw LLM completion text.
        max_count: Maximum number of pseudo-tools to extract.

    Returns:
        Ordered list of cleaned pseudo-tool description strings.
    """
    # Try JSON array first
    text = text.strip()
    if text.startswith("[") or text.startswith("{"):
        try:
            import json
            obj = json.loads(text)
            if isinstance(obj, list):
                out: list[str] = []
                for item in obj:
                    if isinstance(item, str):
                        out.append(item)
                    elif isinstance(item, dict):
                        # Take description/functionality field
                        desc = (
                            item.get("description")
                            or item.get("functionality")
                            or item.get("desc")
                            or item.get("tool_description")
                            or ""
                        )
                        if desc:
                            out.append(str(desc))
                return out[: max_count]
        except (ValueError, TypeError):
            pass  # fall through to line parsing

    # Line-based fallback
    out = []
    for line in text.splitlines():
        line = line.strip()
        # Strip numbering / bullet prefixes
        if line and line[0].isdigit():
            # "1. desc" or "1) desc"
            for sep in [". ", ") ", "- "]:
                if sep in line[:6]:
                    line = line.split(sep, 1)[1].strip()
                    break
        if line.startswith("- ") or line.startswith("* ") or line.startswith("• "):
            line = line[2:].strip()
        # Strip surrounding quotes
        line = line.strip("\"'`")
        if line:
            out.append(line)
        if len(out) >= max_count:
            break
    return out


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

class LessIsMoreBaseline(Baseline):
    """One pseudo-tool description per query, retrieve by embedding similarity.

    Implements Paramanayakam et al. 2024 ("Less is More", arXiv 2411.15399).
    This is the **Single-Pass degenerate case** of FitText Memetic (N=1, G=1).

    Algorithm
    ---------
    1. LLM generates up to ``top_k`` "ideal" pseudo-tool descriptions for the
       query (§III-B Tool Recommender).
    2. For each pseudo-tool description, retrieve top-k tools via the
       retriever's embedding-based search (§III-C, Level 1 path).
    3. Aggregate scores across pseudo-tools by max-pool.  Return top-k.
    4. If the avg top-k similarity for the best pseudo-tool is below
       ``confidence_threshold`` (default 0.5, per paper), fall back to
       directly retrieving with the raw user query (§III-C Level 3 fallback).

    Config kwargs
    -------------
    top_k : int, default 5
    confidence_threshold : float, default 0.5
        If avg top-k similarity falls below this, fall back to raw-query
        retrieval (paper §III-C Level 3 fallback).
    enable_l2_clusters : bool, default False
        If True, also score against a pre-built L2 cluster index passed via
        ``cluster_index`` in ``cfg``.  Off by default because the cluster
        index requires an offline GPT-4 augmentation pass.
    temperature : float, default 0.0
    seed : int, default 42
    max_tokens : int, default 512
    """

    name: ClassVar[str] = "less_is_more"

    def __init__(
        self,
        model_client: ModelClient,
        retriever: Any,
        *,
        top_k: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_client, retriever, top_k=top_k, **kwargs)
        self._confidence_threshold: float = float(
            self.cfg.get("confidence_threshold", 0.5)
        )
        self._enable_l2: bool = bool(self.cfg.get("enable_l2_clusters", False))

    async def retrieve(self, query: str, *, tool_catalog: list[dict]) -> RetrievedTools:
        """Generate pseudo-tool descriptions; retrieve by embedding similarity.

        Args:
            query: User task query.
            tool_catalog: Tool catalog (passed for interface parity; the
                retriever holds its own embedded index, so this is unused
                in the hot path).

        Returns:
            RetrievedTools ranked by max-pool similarity across pseudo-tools.
        """
        t0 = time.monotonic()

        # ---- §III-B: Tool Recommender LLM call ---------------------------
        messages = [
            {
                "role": "system",
                "content": _SYSTEM_PROMPT.format(top_k=self.top_k),
            },
            {
                "role": "user",
                "content": _USER_PROMPT_TEMPLATE.format(query=query),
            },
        ]

        response = await self.model_client.chat_completion(
            messages=messages,
            temperature=float(self.cfg.get("temperature", 0.0)),
            seed=int(self.cfg.get("seed", 42)),
            max_tokens=int(self.cfg.get("max_tokens", 512)),
        )
        raw_text = response.content or ""
        pseudo_tools = _parse_pseudo_tools(raw_text, max_count=self.top_k)

        if not pseudo_tools:
            logger.warning(
                "less_is_more: failed to parse any pseudo-tool from LLM output: %r",
                raw_text[:200],
            )
            pseudo_tools = [query]  # fall back to raw query as a single probe

        # ---- §III-C: Tool Controller (k-NN over each pseudo-tool) --------
        # Per-pseudo-tool retrieval, then max-pool score-merge.
        score_by_tool: dict[str, float] = {}
        pseudo_avg_scores: list[float] = []
        for pt in pseudo_tools:
            try:
                hits = self.retriever.retrieve(pt, top_k=self.top_k)
            except Exception as exc:
                logger.warning("less_is_more: retriever failed on pseudo-tool: %s", exc)
                continue
            if hits:
                pseudo_avg_scores.append(
                    sum(s for _, s in hits) / max(1, len(hits))
                )
            for tid, score in hits:
                if score > score_by_tool.get(tid, -1.0):
                    score_by_tool[tid] = score

        # ---- §III-C: Confidence guard / L3 fallback ----------------------
        max_avg = max(pseudo_avg_scores) if pseudo_avg_scores else 0.0
        used_l3_fallback = False
        if max_avg < self._confidence_threshold:
            used_l3_fallback = True
            logger.debug(
                "less_is_more: avg score %.3f < %.2f -- L3 fallback (raw query).",
                max_avg,
                self._confidence_threshold,
            )
            # L3 fallback: retrieve directly with the raw query
            try:
                fallback_hits = self.retriever.retrieve(query, top_k=self.top_k)
                for tid, score in fallback_hits:
                    if score > score_by_tool.get(tid, -1.0):
                        score_by_tool[tid] = score
            except Exception as exc:
                logger.warning("less_is_more: L3 fallback retrieval failed: %s", exc)

        # ---- Rank & truncate to top_k ------------------------------------
        ranked = sorted(score_by_tool.items(), key=lambda kv: kv[1], reverse=True)
        top = ranked[: self.top_k]

        latency_ms = (time.monotonic() - t0) * 1000.0
        return RetrievedTools(
            tool_ids=[tid for tid, _ in top],
            scores=[s for _, s in top],
            metadata={
                "pseudo_tools": pseudo_tools,
                "llm_raw_text": raw_text,
                "max_avg_pseudo_score": max_avg,
                "used_l3_fallback": used_l3_fallback,
                "latency_ms": latency_ms,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            },
        )
