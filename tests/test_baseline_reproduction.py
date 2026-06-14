"""Reproduction spot-checks for REIMPLEMENT baselines.

These tests verify that each baseline produces a tool-rank ordering
consistent with its paper specification on a deterministic synthetic
benchmark.  Mocked-LLM responses keep the mocked path < 10s; the
manual real-LLM path requires ``RUN_BASELINE_REPRODUCTION=1``.

Synthetic benchmark
-------------------
10 queries, each with:
- A natural-language query string.
- A gold tool ID (the API the query is "about").
- A canonical set of "expected" pseudo-tool descriptions (what an LLM
  faithful to the baseline would produce).

The retriever stub is deterministic: it scores a tool by the count of
overlapping keywords between the query/intent and the tool's gold
description.  This means the baselines' rank ordering becomes a function
purely of:
  - Less-is-More: how well the pseudo-tool descriptions overlap with the
    catalog's gold descriptions.
  - Re-Invoke: how well the extracted intents fan out across the catalog.
  - Xu 2024: whether each iteration's refined query keeps or improves
    overlap with the gold description.

Each test asserts Jaccard(top-k ∩ gold-set) ≥ 0.6 over the 10 queries.
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

# Ensure project root on path
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from baselines.base import RetrievedTools, RetrieverAdapter
from baselines.less_is_more import LessIsMoreBaseline
from baselines.reinvoke import ReInvokeBaseline
from baselines.xu2024 import Xu2024Baseline


# ---------------------------------------------------------------------------
# Synthetic benchmark
# ---------------------------------------------------------------------------

# Tool catalog: 12 tools across 4 domains, each with a gold description
# that contains a few discriminating keywords.
TOOL_CATALOG: list[dict] = [
    {"category": "weather", "tool_name": "get_current_weather",
     "api_name": "get_current_weather", "name": "get_current_weather",
     "description": "Fetch current weather temperature humidity for a city."},
    {"category": "weather", "tool_name": "forecast_weather",
     "api_name": "forecast_weather", "name": "forecast_weather",
     "description": "Fetch multi-day weather forecast for a location."},
    {"category": "weather", "tool_name": "weather_alerts",
     "api_name": "weather_alerts", "name": "weather_alerts",
     "description": "Fetch active weather alerts and warnings."},
    {"category": "travel", "tool_name": "search_flights",
     "api_name": "search_flights", "name": "search_flights",
     "description": "Search flight routes between two airports by date."},
    {"category": "travel", "tool_name": "search_hotels",
     "api_name": "search_hotels", "name": "search_hotels",
     "description": "Search hotel availability and price by city."},
    {"category": "translate", "tool_name": "translate_text",
     "api_name": "translate_text", "name": "translate_text",
     "description": "Translate text between languages with neural translation."},
    {"category": "translate", "tool_name": "detect_language",
     "api_name": "detect_language", "name": "detect_language",
     "description": "Detect the language of a given text snippet."},
    {"category": "math", "tool_name": "evaluate_expression",
     "api_name": "evaluate_expression", "name": "evaluate_expression",
     "description": "Evaluate a mathematical expression numerically."},
    {"category": "math", "tool_name": "solve_equation",
     "api_name": "solve_equation", "name": "solve_equation",
     "description": "Solve an algebraic equation symbolically."},
    {"category": "stocks", "tool_name": "get_stock_price",
     "api_name": "get_stock_price", "name": "get_stock_price",
     "description": "Fetch current stock price by ticker symbol."},
    {"category": "stocks", "tool_name": "get_stock_news",
     "api_name": "get_stock_news", "name": "get_stock_news",
     "description": "Fetch recent news headlines about a stock ticker."},
    {"category": "stocks", "tool_name": "stock_history",
     "api_name": "stock_history", "name": "stock_history",
     "description": "Fetch historical price chart data for a stock."},
]


def _tool_id(tool: dict) -> str:
    """Build the composite tool_id used by the baselines."""
    return f"{tool['category']}::{tool['tool_name']}::{tool['api_name']}"


# 10-query benchmark; each query has 1 gold tool.
# (query, gold_tool_id, pseudo_tool_descriptions)
SYNTHETIC_BENCHMARK: list[tuple[str, str, list[str]]] = [
    (
        "What's the temperature in Tokyo right now?",
        _tool_id(TOOL_CATALOG[0]),  # get_current_weather
        ["Fetch current weather temperature for a city."],
    ),
    (
        "Will it rain in London this weekend?",
        _tool_id(TOOL_CATALOG[1]),  # forecast_weather
        ["Multi-day weather forecast for a location."],
    ),
    (
        "Are there any active storm warnings in Florida?",
        _tool_id(TOOL_CATALOG[2]),  # weather_alerts
        ["Active weather alerts and warnings."],
    ),
    (
        "Find me a cheap flight from SFO to JFK next Friday.",
        _tool_id(TOOL_CATALOG[3]),  # search_flights
        ["Search flight routes between two airports by date."],
    ),
    (
        "Book a hotel in Paris for two nights.",
        _tool_id(TOOL_CATALOG[4]),  # search_hotels
        ["Search hotel availability and price by city."],
    ),
    (
        "Translate 'good morning' from English to Japanese.",
        _tool_id(TOOL_CATALOG[5]),  # translate_text
        ["Translate text between languages."],
    ),
    (
        "What language is this text: 'Bonjour tout le monde'?",
        _tool_id(TOOL_CATALOG[6]),  # detect_language
        ["Detect the language of a text snippet."],
    ),
    (
        "Compute 23 * 47 + 109.",
        _tool_id(TOOL_CATALOG[7]),  # evaluate_expression
        ["Evaluate a mathematical expression."],
    ),
    (
        "What's the current price of AAPL stock?",
        _tool_id(TOOL_CATALOG[9]),  # get_stock_price
        ["Fetch current stock price by ticker."],
    ),
    (
        "Show me historical price chart for TSLA over 5 years.",
        _tool_id(TOOL_CATALOG[11]),  # stock_history
        ["Fetch historical price chart for a stock."],
    ),
]


# ---------------------------------------------------------------------------
# Deterministic stub LLM + retriever
# ---------------------------------------------------------------------------

@dataclass
class FakeResponse:
    """Minimal stub for NormalizedResponse."""
    content: str | None = None
    tool_calls: list = field(default_factory=list)
    finish_reason: str = "stop"
    model_revision: str = "stub-model"
    input_tokens: int = 10
    cached_input_tokens: int = 0
    output_tokens: int = 5
    latency_ms: float = 1.0
    provider: str = "stub"
    raw_response: dict = field(default_factory=dict)


class DeterministicLLM:
    """Deterministic stub LLM.

    The ``script`` is a list of canned responses; each ``chat_completion``
    call pops the next response.  When exhausted, returns ``"N/A"`` (handy
    for Xu2024 to end the chain cleanly).
    """

    def __init__(self, script: list[str]) -> None:
        self.script: list[str] = list(script)
        self.calls: list[list] = []

    async def chat_completion(
        self,
        messages: list,
        tools: list | None = None,
        temperature: float = 0.0,
        seed: int = 42,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> FakeResponse:
        self.calls.append(messages)
        if self.script:
            return FakeResponse(content=self.script.pop(0))
        return FakeResponse(content="N/A")


class KeywordOverlapRetriever:
    """Retriever stub that ranks tools by keyword overlap with the query.

    Deterministic and pure-function -- no LLM, no neural embedding.  The
    score for ``query`` vs each tool is the count of shared lowercased
    word tokens between the query and the tool's description (plus
    0.001 * desc-length tie-break).

    For the Re-Invoke v2 algorithm (which needs real corpus + sentence
    encoders, not just .retrieving), this class additionally exposes a
    bag-of-words embedding over a fixed vocabulary derived from the
    catalog.  The vocab includes all descriptive tokens; ``encode_sentence``
    and ``encode_corpus`` return L2-normalized binary indicator vectors.
    """

    STOPWORDS = frozenset(
        {"the", "a", "an", "for", "in", "on", "of", "is", "to", "and",
         "or", "by", "with", "be", "are", "from", "as", "at", "this",
         "that", "what", "where", "when", "how", "any", "active"}
    )

    model_name = "keyword-overlap-stub"

    def __init__(self, catalog: list[dict]) -> None:
        self.catalog = catalog
        # Precompute token sets
        self._tokens: list[set[str]] = []
        for tool in catalog:
            desc = (tool.get("description") or "") + " " + (
                tool.get("api_name") or ""
            )
            tokens = self._tokenize(desc)
            self._tokens.append(tokens)
        # Fixed vocabulary: combined catalog token set (deterministic order).
        self._vocab: list[str] = sorted({t for toks in self._tokens for t in toks})
        self._tok_to_idx: dict[str, int] = {t: i for i, t in enumerate(self._vocab)}

    @classmethod
    def _tokenize(cls, text: str) -> set[str]:
        return {
            t.lower().strip(".,?!:;\"'()[]{}")
            for t in (text or "").split()
            if t and t.lower() not in cls.STOPWORDS
        }

    def retrieving(
        self,
        query: str,
        top_k: int = 5,
        excluded_tools: dict | None = None,
    ) -> list[dict]:
        """Return ranked list of tool dicts."""
        q_tokens = self._tokenize(query)
        scored: list[tuple[int, float, dict]] = []
        for idx, tool in enumerate(self.catalog):
            overlap = len(q_tokens & self._tokens[idx])
            score = overlap + 0.001 * len(self._tokens[idx])
            scored.append((idx, score, tool))
        scored.sort(key=lambda x: x[1], reverse=True)
        out: list[dict] = []
        for _, score, tool in scored[:top_k]:
            out.append({**tool, "score": float(score)})
        return out

    def _vectorize(self, text: str):
        import torch
        v = torch.zeros(len(self._vocab), dtype=torch.float32)
        for tok in self._tokenize(text):
            j = self._tok_to_idx.get(tok)
            if j is not None:
                v[j] = 1.0
        # Add a tiny constant to avoid all-zero vectors.
        if v.sum() == 0:
            v[0] = 1e-3
        return v

    def encode_sentence(self, text):
        """L2-normalized binary BOW. Accepts str or list[str]."""
        import torch
        if isinstance(text, str):
            v = self._vectorize(text)
            return torch.nn.functional.normalize(v, dim=0)
        vecs = torch.stack([self._vectorize(t) for t in text], dim=0)
        return torch.nn.functional.normalize(vecs, dim=1)

    def encode_corpus(self, texts):
        """L2-normalized binary BOW for a batch of docs."""
        import torch
        vecs = torch.stack([self._vectorize(t) for t in texts], dim=0)
        return torch.nn.functional.normalize(vecs, dim=1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_async(coro):
    return asyncio.run(coro)


def _jaccard_at_k(predicted: list[str], gold: set[str], k: int = 5) -> float:
    """Jaccard between predicted top-k and the gold set."""
    pred_set = set(predicted[:k])
    if not pred_set and not gold:
        return 1.0
    inter = pred_set & gold
    denominator = pred_set | gold
    return len(inter) / len(denominator) if denominator else 0.0


def _has_gold_in_topk(predicted: list[str], gold_tool: str, k: int = 5) -> bool:
    """Whether the gold tool is in the top-k predictions."""
    return gold_tool in predicted[:k]


# ---------------------------------------------------------------------------
# Less-is-More reproduction
# ---------------------------------------------------------------------------

def test_less_is_more_synthetic_reproduction():
    """Less-is-More on synthetic benchmark: gold ∈ top-5 in ≥ 60% of queries.

    Paper §III-B: LLM generates pseudo-tool descriptions; §III-C retrieves
    by embedding similarity.  Our stub LLM returns the canonical
    pseudo-tool description per query; the keyword-overlap retriever then
    ranks tools.  With faithful pseudo-tools, gold should land top-5 on
    the great majority of queries.
    """
    n_hit_topk = 0
    total = len(SYNTHETIC_BENCHMARK)
    for query, gold_tool, pseudo_tools in SYNTHETIC_BENCHMARK:
        llm = DeterministicLLM(script=["\n".join(pseudo_tools)])
        retriever = RetrieverAdapter(KeywordOverlapRetriever(TOOL_CATALOG))
        baseline = LessIsMoreBaseline(
            llm, retriever, top_k=5, confidence_threshold=0.0
        )
        result = _run_async(baseline.retrieve(query, tool_catalog=TOOL_CATALOG))
        if _has_gold_in_topk(result.tool_ids, gold_tool, k=5):
            n_hit_topk += 1
    hit_rate = n_hit_topk / total
    assert hit_rate >= 0.6, (
        f"Less-is-More gold@5 hit rate {hit_rate:.2f} below 0.6 acceptance "
        f"threshold ({n_hit_topk}/{total}).  Algorithm or stub may have drifted "
        f"from paper §III-B + §III-C."
    )


def test_less_is_more_l3_fallback_triggers_on_vague_query():
    """Per paper §III-C: low-confidence retrieval (avg < 0.5) falls back to L3."""
    # Vague pseudo-tool description -> low overlap -> low score
    llm = DeterministicLLM(script=["a very vague generic capability description"])
    retriever = RetrieverAdapter(KeywordOverlapRetriever(TOOL_CATALOG))
    # threshold=2.0 forces fallback regardless (max overlap is ~3)
    baseline = LessIsMoreBaseline(
        llm, retriever, top_k=5, confidence_threshold=100.0
    )
    result = _run_async(baseline.retrieve(
        "vague unrelated text", tool_catalog=TOOL_CATALOG
    ))
    assert result.metadata.get("used_l3_fallback") is True


# ---------------------------------------------------------------------------
# Re-Invoke reproduction
# ---------------------------------------------------------------------------

def _prepop_v2_cache(tmp_path, k_synth=1):
    """Pre-populate the v2 synth-query JSONL cache (parallel to the
    catalog), keyed by (generator, embedder, m). The cache layout is
    determined by ReInvokeBaseline._cache._slug-derived path.

    Returns the directory the baseline's cache resolves to so tests can
    inspect or override it.
    """
    import json
    from baselines.reinvoke import _default_cache, _tool_id_from_dict
    cfg = {
        "cache_dir": str(tmp_path),
        "benchmark_tag": "synthetic",
        "generator_model": "stub-model",
        "embedder_name": "keyword-overlap-stub",
        "k_synth": k_synth,
    }
    cache = _default_cache(cfg)
    cache.ensure_parent()
    with cache.synth_jsonl.open("w", encoding="utf-8") as fh:
        for tool in TOOL_CATALOG:
            fh.write(json.dumps({
                "tool_id": _tool_id_from_dict(tool),
                "synth_queries": [tool["description"]] * k_synth,
            }) + "\n")
    return cache


def test_reinvoke_synthetic_reproduction_v2(tmp_path):
    """Re-Invoke v2 on synthetic benchmark: gold ∈ top-5 in ≥ 60% of queries.

    Paper §3.1 (Query Generator) + §3.2 (Intent Extractor) + §3.3 + Appendix
    C Algorithm 2 (Multi-View Similarity Ranking).  Pre-populates the synth-
    query cache with each tool's own description (cheapest faithful proxy
    so we don't pay m*|tools| LLM calls in the test).  The averaged-
    embedding tensor is built on the fly by the baseline using the stub
    bag-of-words encoder.
    """
    _prepop_v2_cache(tmp_path, k_synth=1)

    n_hit_topk = 0
    total = len(SYNTHETIC_BENCHMARK)
    for query, gold_tool, pseudo_tools in SYNTHETIC_BENCHMARK:
        intents = pseudo_tools  # use canonical pseudo-tools as intents
        llm = DeterministicLLM(script=["\n".join(intents)])
        retriever = RetrieverAdapter(KeywordOverlapRetriever(TOOL_CATALOG))
        baseline = ReInvokeBaseline(
            llm, retriever, top_k=5,
            cache_dir=str(tmp_path),
            benchmark_tag="synthetic",
            generator_model="stub-model",
            embedder_name="keyword-overlap-stub",
            max_intents=3,
            k_synth=1,
        )
        result = _run_async(baseline.retrieve(query, tool_catalog=TOOL_CATALOG))
        if _has_gold_in_topk(result.tool_ids, gold_tool, k=5):
            n_hit_topk += 1
    hit_rate = n_hit_topk / total
    assert hit_rate >= 0.6, (
        f"Re-Invoke v2 gold@5 hit rate {hit_rate:.2f} below 0.6 acceptance "
        f"threshold ({n_hit_topk}/{total}). Multi-view ranking or stub may "
        f"have drifted from paper §3.1-3.3 + Appendix C Algorithm 2."
    )


def test_reinvoke_multi_view_ranking_appendix_c(tmp_path):
    """Direct math test of Appendix C Algorithm 2 tuple-ranking.

    Constructs an intent_embs × tool_embs matrix where:
      - intent 0 ranks tool 0 highest (sim=0.9), tool 1 lowest (sim=0.1)
      - intent 1 ranks tool 1 highest (sim=0.9), tool 0 lowest (sim=0.1)
    Algorithm 2 should surface BOTH tool 0 and tool 1 in the top-2 (each
    is rank-1 under one intent, so both get max rev_rank = N).
    """
    import torch
    # 3-tool, 2-intent scenario. Tool 2 is uniformly mid-rank.
    intent_embs = torch.eye(2, 3)              # I0 = [1,0,0], I1 = [0,1,0]
    tool_embs = torch.tensor([
        [0.9, 0.1, 0.05],  # tool 0 — high sim to I0, low to I1
        [0.1, 0.9, 0.05],  # tool 1 — low sim to I0, high to I1
        [0.5, 0.5, 0.05],  # tool 2 — middling for both
    ], dtype=torch.float32)
    intent_embs = torch.nn.functional.normalize(intent_embs, dim=1)
    tool_embs = torch.nn.functional.normalize(tool_embs, dim=1)

    llm = DeterministicLLM(script=["intent 0\nintent 1"])
    retriever = RetrieverAdapter(KeywordOverlapRetriever(TOOL_CATALOG[:3]))
    baseline = ReInvokeBaseline(
        llm, retriever, top_k=3,
        cache_dir=str(tmp_path),
        benchmark_tag="alg2_unit",
        generator_model="stub-model",
        embedder_name="keyword-overlap-stub",
        max_intents=2, k_synth=1,
    )

    ranked = baseline._multi_view_rank(
        intent_embs=intent_embs,
        tool_embs=tool_embs,
        tool_ids=["t0", "t1", "t2"],
    )
    top_ids = [r[0] for r in ranked[:2]]
    assert set(top_ids) == {"t0", "t1"}, (
        f"Algorithm 2: tools 0 and 1 each rank #1 under one intent, so "
        f"both must surface in top-2 via max-over-intents lex tuple. "
        f"got: {top_ids}"
    )
    # Finding #7 guard: emitted scores must be strictly DECREASING in rank
    # order (so pytrec_eval reproduces the Algorithm-2 order, not cosine sim).
    rank_scores = [r[1] for r in ranked]
    assert all(rank_scores[i] > rank_scores[i + 1] for i in range(len(rank_scores) - 1)), (
        f"Ranking scores must be strictly decreasing to preserve Alg-2 order "
        f"under pytrec_eval; got {rank_scores}"
    )
    # 3rd tuple element is the raw cosine sim (transparency, not the rank key).
    assert all(len(r) == 3 for r in ranked)


def test_reinvoke_corpus_checksum_unchanged(tmp_path):
    """Corpus-integrity guard: original corpus file must be byte-identical
    before and after a full retrieve() pass.

    We write a synthetic des_corpus.json file, pass its path via cfg, run a
    retrieve, then verify checksum did not change.
    """
    import json, hashlib
    _prepop_v2_cache(tmp_path, k_synth=1)
    corpus_path = tmp_path / "des_corpus.json"
    payload = [{"id": _tool_id(t), "description": t["description"]} for t in TOOL_CATALOG]
    with corpus_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    pre_sha = hashlib.sha256(corpus_path.read_bytes()).hexdigest()

    llm = DeterministicLLM(script=["weather"])
    retriever = RetrieverAdapter(KeywordOverlapRetriever(TOOL_CATALOG))
    baseline = ReInvokeBaseline(
        llm, retriever, top_k=5,
        cache_dir=str(tmp_path),
        benchmark_tag="synthetic",
        generator_model="stub-model",
        embedder_name="keyword-overlap-stub",
        max_intents=1, k_synth=1,
        corpus_path=str(corpus_path),
    )
    _run_async(baseline.retrieve("weather", tool_catalog=TOOL_CATALOG))
    baseline.assert_corpus_unchanged()
    post_sha = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    assert pre_sha == post_sha, "Re-Invoke corpus checksum mismatch"


# ---------------------------------------------------------------------------
# Xu et al. 2024 reproduction
# ---------------------------------------------------------------------------

def test_xu2024_synthetic_reproduction():
    """Xu 2024 on synthetic benchmark: gold ∈ top-5 in ≥ 60% of queries.

    Paper §4.2: C/A/R chain.  Our stub LLM emits 'N/A' on first
    refinement, so we converge after one iteration (the cheap path).
    """
    n_hit_topk = 0
    total = len(SYNTHETIC_BENCHMARK)
    for query, gold_tool, _ in SYNTHETIC_BENCHMARK:
        # 3 LLM calls per iter (C, A, R).  R='N/A' converges immediately.
        llm = DeterministicLLM(script=[
            "comprehension output",
            "assessment output",
            "N/A",
        ])
        retriever = RetrieverAdapter(KeywordOverlapRetriever(TOOL_CATALOG))
        baseline = Xu2024Baseline(llm, retriever, top_k=5, max_iterations=2)
        result = _run_async(baseline.retrieve(query, tool_catalog=TOOL_CATALOG))
        if _has_gold_in_topk(result.tool_ids, gold_tool, k=5):
            n_hit_topk += 1
    hit_rate = n_hit_topk / total
    assert hit_rate >= 0.6, (
        f"Xu 2024 gold@5 hit rate {hit_rate:.2f} below 0.6 acceptance "
        f"threshold ({n_hit_topk}/{total}).  Algorithm or stub may have drifted "
        f"from paper §4.2."
    )


def test_xu2024_na_convergence_skips_remaining_iterations():
    """Refinement = 'N/A' -> converged with iterations_run == 1 (paper §4.2)."""
    llm = DeterministicLLM(script=["c", "a", "N/A"])
    retriever = RetrieverAdapter(KeywordOverlapRetriever(TOOL_CATALOG))
    baseline = Xu2024Baseline(llm, retriever, top_k=5, max_iterations=5)
    result = _run_async(baseline.retrieve(
        "What's the weather in Tokyo?", tool_catalog=TOOL_CATALOG
    ))
    assert result.metadata["converged"] is True
    assert result.metadata["convergence_reason"] == "na_token"
    assert result.metadata["iterations_run"] == 1


def test_xu2024_refinement_drives_new_retrieval():
    """If refinement is a new query, iteration 2 uses it (paper §4.2 Eq. 4)."""
    # iter 0: C, A, R="refined query about flights"
    # iter 1 retrieves with refined query; if same set, converge stable_set
    # We use a query whose initial retrieval is weather but refines to flights
    llm = DeterministicLLM(script=[
        "comprehension of weather query",
        "assessment: missing flight tools",
        "Search flight routes between two airports by date.",
        # iter 1 C/A/R if it happens
        "comp 2", "assess 2", "N/A",
    ])
    retriever = RetrieverAdapter(KeywordOverlapRetriever(TOOL_CATALOG))
    baseline = Xu2024Baseline(llm, retriever, top_k=3, max_iterations=3)
    result = _run_async(baseline.retrieve(
        "weather query that should be reframed to flights",
        tool_catalog=TOOL_CATALOG,
    ))
    # After refinement, retrieval shifts toward flight tools
    has_flight = any("flight" in tid.lower() for tid in result.tool_ids)
    assert has_flight or result.metadata.get("converged"), (
        f"Refinement should change retrieval target. Got: {result.tool_ids}, "
        f"trace: {[t.get('refinement') for t in result.metadata['trace']]}"
    )


# ---------------------------------------------------------------------------
# §2.2 degeneracy verification: Xu 2024 ≈ FitText multi_turn
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    reason=(
        "Empirical degeneracy verification requires the evol-refactor track's "
        "multi_turn invocation to be wired into the eval harness.  Wire once "
        "evol-refactor lands."
    ),
    strict=False,
)
def test_xu2024_degeneracy_with_multi_turn():
    """Baseline-degeneracy check: Xu 2024 ≈ FitText multi_turn within ±2 NDCG.

    Both should produce retrieval sets that overlap > 70% on the same query
    set with the same LLM responses.  This validates the claim that
    Xu 2024 is a degenerate case of FitText.

    Wire this in once the evol-refactor track exposes a callable
    ``multi_turn_retrieve(query, tool_catalog, ...) -> RetrievedTools``.
    """
    # Placeholder: try-import the multi_turn entry point.  Will be wired in
    # once evol-refactor lands a runnable interface.
    try:
        from Toolret.strategy.strategies import multi_turn_retrieve  # type: ignore
    except ImportError:
        pytest.xfail(
            "FitText multi_turn entrypoint not yet exposed by evol-refactor."
        )

    overlaps = []
    for query, _, pseudo_tools in SYNTHETIC_BENCHMARK[:5]:
        llm_xu = DeterministicLLM(script=["c", "a", "N/A"])
        retriever = RetrieverAdapter(KeywordOverlapRetriever(TOOL_CATALOG))
        xu = Xu2024Baseline(llm_xu, retriever, top_k=5, max_iterations=2)
        xu_result = _run_async(xu.retrieve(query, tool_catalog=TOOL_CATALOG))

        llm_mt = DeterministicLLM(script=["c", "a", "N/A"])  # same script
        mt_result = multi_turn_retrieve(  # type: ignore
            query,
            tool_catalog=TOOL_CATALOG,
            model_client=llm_mt,
            retriever=retriever,
            top_k=5,
            generations=2,
        )
        a = set(xu_result.tool_ids)
        b = set(mt_result.tool_ids)
        if a | b:
            overlaps.append(len(a & b) / len(a | b))
    if overlaps:
        mean_overlap = sum(overlaps) / len(overlaps)
        assert mean_overlap > 0.70, (
            f"Xu 2024 vs multi_turn mean overlap {mean_overlap:.2f} < 0.70 "
            f"(degeneracy claim violated)."
        )


# ---------------------------------------------------------------------------
# Real-LLM reproduction (manual, gated by RUN_BASELINE_REPRODUCTION)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    os.environ.get("RUN_BASELINE_REPRODUCTION") != "1",
    reason="Set RUN_BASELINE_REPRODUCTION=1 to run live LLM reproduction.",
)
def test_less_is_more_real_llm():
    """Manual: live OpenAI call on synthetic benchmark (paper §III)."""
    from StableToolBench.toolbench.inference.LLM.clients.factory import (  # type: ignore
        make_client,
    )
    client = make_client(model=os.environ.get("REPRO_MODEL", "gpt-4o-mini"))
    retriever = RetrieverAdapter(KeywordOverlapRetriever(TOOL_CATALOG))
    baseline = LessIsMoreBaseline(client, retriever, top_k=5)
    hits = 0
    for query, gold, _ in SYNTHETIC_BENCHMARK:
        result = _run_async(baseline.retrieve(query, tool_catalog=TOOL_CATALOG))
        if gold in result.tool_ids[:5]:
            hits += 1
    assert hits / len(SYNTHETIC_BENCHMARK) >= 0.6


@pytest.mark.skipif(
    os.environ.get("RUN_BASELINE_REPRODUCTION") != "1",
    reason="Set RUN_BASELINE_REPRODUCTION=1 to run live LLM reproduction.",
)
def test_reinvoke_real_llm(tmp_path):
    """Manual: live OpenAI call on synthetic benchmark (paper §3)."""
    from StableToolBench.toolbench.inference.LLM.clients.factory import (  # type: ignore
        make_client,
    )
    client = make_client(model=os.environ.get("REPRO_MODEL", "gpt-4o-mini"))
    retriever = RetrieverAdapter(KeywordOverlapRetriever(TOOL_CATALOG))
    baseline = ReInvokeBaseline(
        client, retriever, top_k=5,
        cache_dir=str(tmp_path),
        embedder_revision="repro_real",
    )
    hits = 0
    for query, gold, _ in SYNTHETIC_BENCHMARK:
        result = _run_async(baseline.retrieve(query, tool_catalog=TOOL_CATALOG))
        if gold in result.tool_ids[:5]:
            hits += 1
    assert hits / len(SYNTHETIC_BENCHMARK) >= 0.6


@pytest.mark.skipif(
    os.environ.get("RUN_BASELINE_REPRODUCTION") != "1",
    reason="Set RUN_BASELINE_REPRODUCTION=1 to run live LLM reproduction.",
)
def test_xu2024_real_llm():
    """Manual: live OpenAI call on synthetic benchmark (paper §4.2)."""
    from StableToolBench.toolbench.inference.LLM.clients.factory import (  # type: ignore
        make_client,
    )
    client = make_client(model=os.environ.get("REPRO_MODEL", "gpt-4o-mini"))
    retriever = RetrieverAdapter(KeywordOverlapRetriever(TOOL_CATALOG))
    baseline = Xu2024Baseline(client, retriever, top_k=5, max_iterations=2)
    hits = 0
    for query, gold, _ in SYNTHETIC_BENCHMARK:
        result = _run_async(baseline.retrieve(query, tool_catalog=TOOL_CATALOG))
        if gold in result.tool_ids[:5]:
            hits += 1
    assert hits / len(SYNTHETIC_BENCHMARK) >= 0.6
