"""Re-Invoke baseline (Chen et al., arXiv 2408.01875, EMNLP 2024 Findings).

Paper: "Re-Invoke: Tool Invocation Rewriting for Zero-Shot Tool Retrieval."
       Chen, Yoon, Sachan, Wang, Cohen-Addad, Bateni, Lee, Pfister.
       Google Cloud AI Research. arXiv:2408.01875v2, EMNLP 2024 Findings.

Reimplemented from paper text — there is no official code release (Google
paper tied to internal Vertex AI). Prompts taken verbatim from Appendix A
(Figures 4 and 5). The paper's deprecated Vertex AI models
(`text-bison@001` generator, `textembedding-gecko@003` embedder) are
substituted with FitText's pinned models (see `configs/model_pins.yaml`).

Algorithm — three components (paper §3.1, §3.2, §3.3, Appendix C Alg. 2)
-----------------------------------------------------------------------

Offline — Query Generator (§3.1, §A.1, §4.3):
    For each tool d in catalog:
        Call LLM m times with the Figure 4 prompt → m synthetic queries.
        Form m augmented docs: aug_i(d) = "Documentation: <d> Query: <q_i>"
        (paper §4.3 verbatim append format).
        Embed each aug_i(d) via the FitText embedder.
        AVERAGE the m embeddings → ONE representation e(d) per tool
        (paper §4.3 + §3.3 + Appendix C: "compute the average embedding on
        each copy of the same tool document").
    Persist (a) synth queries (JSONL sidecar) and (b) averaged tensor (.pt).

Online — Intent Extractor (§3.2, §A.2):
    Call LLM once with the Figure 5 prompt → n newline-separated intents.
    Embed each intent.

Online — Multi-View Similarity Ranking (§3.3, Appendix C Algorithm 2):
    sim[i, t] = cos(intent_i, e(tool_t))
    For each intent i:
        rev_rank[t] = ascending rank of sim[i, t]    (lowest sim → 1)
        rank_score[i, t] = (rev_rank[t], sim[i, t])  (lexicographic tuple)
    retrieval_score(t) = max over intents of rank_score[i, t]   (lex max)
    Return top-k tools by retrieval_score descending.

Audit note: replaces a prior implementation that was non-functional — its
retrieve() generated+cached synth queries but never consumed them, used RRF
k=60 instead of paper's Appendix C tuple-ranking, and used paraphrased
prompts.

Documented deviations from paper §4.3 (for transparency report):
1. Generator LLM: FitText main pin from configs/model_pins.yaml in
   place of text-bison@001. EFFECTIVE GENERATION TEMPERATURE = 1.0, NOT the
   paper's 0.7 and NOT the `synth_temperature=0.7` set in config: the patched
   OpenAI client strips `temperature`/`top_p` from the Responses-API request
   for reasoning models (openai_client.py), so the request
   carries no temperature → the model samples at the Responses-API server
   default (1.0). No `seed` is sent on that path either, so the m synthetic
   queries are genuinely INDEPENDENT temp-1.0 samples. This satisfies the
   paper's stated purpose for the temperature ("introduce variation and
   cover the potential query space") via multi-sampling, but it is a
   deviation from the paper's specific 0.7 — recorded for the transparency
   statement. (The 0.7 config value is retained because it IS honored for
   temperature-respecting chat models.)
2. Embedder: FitText ToolRet pin from configs/model_pins.yaml in place of
   textembedding-gecko@003. Identical for method-vs-method
   comparison parity with the FitText results table.
3. max_intents capped at 3 (paper does not specify a cap; we cap to bound
   per-query inference cost). Configurable via cfg.

References to paper Figures:
- Figure 4: query generator prompt — Appendix A.
- Figure 5: intent extractor prompt — Appendix A.
- Figure 8: worked example of Appendix C Algorithm 2.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Callable

import torch

from .base import Baseline, RetrievedTools, RetrieverAdapter
from StableToolBench.toolbench.inference.LLM.clients import ModelClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Verbatim prompts from paper Appendix A (Figures 4 and 5)
# ---------------------------------------------------------------------------

# Figure 4 — Query Generator. The paper's published Figure 4 image renders the
# prompt with the tool's API JSON inserted between the instructions and the
# closing "The relevant query is:" marker — exactly matching the structure of
# Figure 6 (HyDE), where Query: <user query>\nThe API documentation is:\n
# is used as the closing marker. We adopt the same canonical placement.
_FIG4_QUERY_GENERATOR_PROMPT = (
    "Suppose you are an assistant and you have access to the following API "
    "to answer user's queries.\n\n"
    "You are provided with a tool and its available API function including "
    "the description and parameters.\n\n"
    "Your task is to generate a possible user query that can be handled by "
    "the API.\n\n"
    "You must include the input parameters in the generated user query.\n\n"
    "The generated query should be complex enough to describe the scenarios "
    "that you will need to call the provided API to address them.\n\n"
    "The API documentation is:\n"
    "{api_doc_json}\n\n"
    "The relevant query is:"
)

# Figure 5 — Intent Extractor. Verbatim from Appendix A (paper renders this
# as a single-block prompt with three in-context examples).
_FIG5_INTENT_EXTRACTOR_PROMPT = (
    "**Instructions**\n\n"
    "Suppose you are a query analyzer and your task is to extract the "
    "underlying user intents from the input query. You should preserve all "
    "the underlying user request and the extracted user intents should be "
    "easily understood without extra context information.\n\n"
    "You should carefully read the given user query to understand its "
    "different intents. Then identify what are the specific intents. Each "
    "individual intent should be separated by a newline.\n\n"
    "Here are some examples of how you should solve the task.\n\n"
    "**Example**\n\n"
    "Query: I'm planning to travel to Paris next weekend to visit my family, "
    "could you help me book a round trip flight ticket? I want to fly in "
    "economy class.\n\n"
    "Intent:\n\n"
    "book a round-trip flight ticket in economy class to Paris next weekend\n\n"
    "Query: I'm a potential buyer looking for a condominium in the city of "
    "Miami. I am specifically interested in properties that have a minimum "
    "of two bathrooms. It should have walkable distance to the grocery "
    "stores.\n\n"
    "Intent:\n\n"
    "buy a real estate in Miami with a minimum of two bathrooms and walkable "
    "distance to the grocery stores\n\n"
    "Query: I want to learn Spanish by talking to the native speakers at "
    "any time. Additionally, can you recommend some interesting books, "
    "preferably fictions, so that I can learn by reading? Also include the "
    "websites that I can buy them.\n\n"
    "Intent:\n"
    "learn Spanish by talking to the native speakers\n"
    "recommend fictions to learn Spanish by reading\n"
    "suggest the websites to buy Spanish fictions\n\n"
    "**Begin!**\n"
    "Query: {query}\n"
    "Intent:"
)

# Paper §4.3 — augmented-document format. "Documentation: <tool document>
# Query: <predicted query>" is the verbatim append format used in the paper.
_AUGMENTED_DOC_FORMAT = "Documentation: {tool_doc} Query: {synth_query}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tool_id_from_dict(tool: dict) -> str:
    """Resolve the tool_id used as the retrieval/eval key.

    Two corpus schemas are supported:
      - ToolRet flat schema: ``{"id": "...", "description": "..."}`` — the
        ``id`` field IS the qrels gold key, so it must be used verbatim.
      - StableToolBench schema: composite
        ``category::tool_name::api_name``.

    ToolRet's ``id`` takes precedence — using the composite scheme on a flat
    ToolRet corpus would collapse every tool to ``::::`` and silently zero
    out recall against the qrels.
    """
    if tool.get("id"):
        return str(tool["id"])
    cat = tool.get("category") or tool.get("category_name") or ""
    tname = tool.get("tool_name") or tool.get("name") or ""
    aname = tool.get("api_name") or tname
    return f"{cat}::{tname}::{aname}"


def _format_api_doc(tool: dict) -> str:
    """Serialize a tool dict to the JSON form used in the Fig. 4 prompt.

    Matches the paper's Appendix B Figure 7 example (newsSearch API) — the
    full API JSON including category, tool, api, description, required and
    optional parameters.
    """
    # ToolRet flat schema ({id, description}) has no structured params — the
    # api_name falls back to the id so the Fig. 4 prompt doc isn't degenerate.
    payload = {
        "category_name": tool.get("category") or tool.get("category_name", ""),
        "tool_name": tool.get("tool_name") or tool.get("name", ""),
        "api_name": tool.get("api_name") or tool.get("name") or tool.get("id", ""),
        "api_description": tool.get("description") or tool.get("api_description") or "",
    }
    for key in ("required_parameters", "optional_parameters", "method"):
        if key in tool:
            payload[key] = tool[key]
    return json.dumps(payload, ensure_ascii=False)


def _tool_doc_string(tool: dict) -> str:
    """The 'tool document' string used in the augmented copy.

    For corpus parity with the FitText pipeline (which reads des_corpus.json
    where each tool has a normalized description field), we prefer the
    `description` field. If absent, fall back to the full API JSON. This is
    what gets embedded after concatenation with the synth query.
    """
    return (tool.get("description") or tool.get("api_description") or _format_api_doc(tool)).strip()


def _sha256_file(path: Path) -> str:
    """Return hex sha256 of a file's bytes, or empty string if missing."""
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _slug(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Sidecar caches
# ---------------------------------------------------------------------------

@dataclass
class _Cache:
    """Filesystem locations for the two index sidecar files + cost log."""
    synth_jsonl: Path
    embeddings_pt: Path
    tool_order_json: Path  # parallel-ordered tool_id list for the pt tensor
    cost_log: Path
    corpus_checksums: Path

    def ensure_parent(self) -> None:
        for p in (self.synth_jsonl, self.embeddings_pt, self.tool_order_json,
                  self.cost_log, self.corpus_checksums):
            p.parent.mkdir(parents=True, exist_ok=True)


def _default_cache(cfg: dict[str, Any]) -> _Cache:
    """Resolve cache locations from cfg.

    Keyed by (generator_model, embedder, m, benchmark_tag) so configuration
    changes don't silently reuse a stale cache.
    """
    runtime_root = Path(os.environ.get("FITTEXT_RUNTIME_ROOT", "runs"))
    root = Path(cfg.get("cache_dir") or runtime_root / "reinvoke" / "cache")
    bench = cfg.get("benchmark_tag") or cfg.get("split") or "default"
    gen = cfg.get("generator_model") or "unknown_gen"
    emb = cfg.get("embedder_name") or "unknown_emb"
    m = int(cfg.get("k_synth", 10))
    slug = _slug(f"{gen}|{emb}|m={m}")
    base = root / bench / slug
    return _Cache(
        synth_jsonl=base / "synth_queries.jsonl",
        embeddings_pt=base / "tool_embeddings_avg.pt",
        tool_order_json=base / "tool_order.json",
        cost_log=base / "cost_log.jsonl",
        corpus_checksums=base / "corpus_checksums.json",
    )


class _SynthQueryIndex:
    """JSONL sidecar: tool_id -> list[synth_query]."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: dict[str, list[str]] = {}
        self._loaded = False

    def load(self) -> bool:
        if not self.path.exists():
            return False
        self._data.clear()
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self._data[rec["tool_id"]] = rec["synth_queries"]
        self._loaded = True
        logger.info("Re-Invoke: loaded %d synth-query rows from %s", len(self._data), self.path)
        return True

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for tool_id, qs in self._data.items():
                fh.write(json.dumps({"tool_id": tool_id, "synth_queries": qs}, ensure_ascii=False))
                fh.write("\n")
        tmp.replace(self.path)
        logger.info("Re-Invoke: saved %d synth-query rows to %s", len(self._data), self.path)

    def get(self, tool_id: str) -> list[str] | None:
        return self._data.get(tool_id)

    def set(self, tool_id: str, qs: list[str]) -> None:
        self._data[tool_id] = qs

    def __len__(self) -> int:
        return len(self._data)


class _AveragedEmbeddingIndex:
    """Sidecar storing the averaged-per-tool embedding tensor + tool_id order."""

    def __init__(self, pt_path: Path, order_path: Path) -> None:
        self.pt_path = pt_path
        self.order_path = order_path
        self._tensor: torch.Tensor | None = None
        self._tool_ids: list[str] = []
        self._id_to_idx: dict[str, int] = {}

    def load(self) -> bool:
        if not (self.pt_path.exists() and self.order_path.exists()):
            return False
        self._tensor = torch.load(self.pt_path, map_location="cpu")
        with self.order_path.open("r", encoding="utf-8") as fh:
            self._tool_ids = json.load(fh)
        self._id_to_idx = {tid: i for i, tid in enumerate(self._tool_ids)}
        logger.info(
            "Re-Invoke: loaded averaged-embedding tensor shape=%s from %s",
            tuple(self._tensor.shape), self.pt_path,
        )
        return True

    def save(self, tool_ids: list[str], tensor: torch.Tensor) -> None:
        self._tool_ids = list(tool_ids)
        self._tensor = tensor
        self._id_to_idx = {tid: i for i, tid in enumerate(self._tool_ids)}
        self.pt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, self.pt_path)
        with self.order_path.open("w", encoding="utf-8") as fh:
            json.dump(self._tool_ids, fh)
        logger.info(
            "Re-Invoke: saved averaged-embedding tensor shape=%s to %s",
            tuple(tensor.shape), self.pt_path,
        )

    @property
    def tensor(self) -> torch.Tensor:
        if self._tensor is None:
            raise RuntimeError("Averaged embedding index not loaded.")
        return self._tensor

    @property
    def tool_ids(self) -> list[str]:
        return self._tool_ids


class _CostLog:
    """Append-only JSONL log: one record per LLM call, tagged with phase."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, *, phase: str, n_calls: int = 1, prompt_tokens: int = 0,
               completion_tokens: int = 0, extra: dict | None = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "ts": time.time(),
            "phase": phase,
            "n_calls": n_calls,
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
        }
        if extra:
            rec.update(extra)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Re-Invoke baseline
# ---------------------------------------------------------------------------

class ReInvokeBaseline(Baseline):
    """Re-Invoke (Chen et al. 2024) — three-component zero-shot tool retrieval.

    Pipeline
    --------
    Offline (one-time, per tool catalog + embedder + generator combo):
      1. m=10 synth queries per tool via the pinned generator (Figure 4 prompt).
      2. Form m augmented docs per tool: "Documentation: <doc> Query: <q_i>".
      3. Embed each via MiniLM (ToolRet) / sup-simcse (STB).
      4. AVERAGE the m embeddings → ONE per-tool representation.
      5. Persist synth queries (JSONL) + averaged tensor (.pt).

    Online (per user query):
      a. Extract <=max_intents intents via the pinned generator (Figure 5 prompt).
      b. Embed each intent in the same space.
      c. Multi-view ranking per Appendix C Algorithm 2:
         retrieval_score(t) = max_i (reversed_rank_i(t), sim_i(t))
      d. Return top-k tools.

    Config (passed via **kwargs, stored in self.cfg)
    ------------------------------------------------
    k_synth : int, default 10
        Paper §4.3: m=10 synthetic queries per tool document.
    max_intents : int, default 3
        Cap on intents per query (deviation from paper, which is uncapped).
    synth_temperature : float, default 0.7
        Paper §4.3 sampling temperature. Some reasoning-model APIs ignore this;
        diversity comes from independent samples in that case.
    intent_temperature : float, default 0.0
        Lower temperature for intent extraction (paper does not specify;
        deterministic is the conservative default for inference-time).
    generator_model : str
        For cache keying. Defaults to model_client's name when available.
    embedder_name : str
        For cache keying.
    benchmark_tag : str
        For cache keying. e.g. "toolret_code", "stb_G1_category".
    cache_dir : str
        Cache root. Defaults under ``FITTEXT_RUNTIME_ROOT`` when omitted.
    seed : int, default 42
    max_tokens : int, default 512
    index_build_concurrency : int, default 8
        Concurrent LLM calls during index build.
    corpus_path : str | None
        Optional. If provided, sha256 is computed at first build and
        verified on every load — guarantees byte-identity of the original
        tool corpus across the Re-Invoke run.
    """

    name: ClassVar[str] = "reinvoke"

    def __init__(
        self,
        model_client: ModelClient,
        retriever: Any,
        *,
        top_k: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_client, retriever, top_k=top_k, **kwargs)
        self._k_synth: int = int(self.cfg.get("k_synth", 10))
        self._max_intents: int = int(self.cfg.get("max_intents", 3))
        self._synth_temperature: float = float(self.cfg.get("synth_temperature", 0.7))
        self._intent_temperature: float = float(self.cfg.get("intent_temperature", 0.0))
        self._seed: int = int(self.cfg.get("seed", 42))
        self._max_tokens: int = int(self.cfg.get("max_tokens", 512))
        self._index_concurrency: int = int(self.cfg.get("index_build_concurrency", 8))
        self._corpus_path: str | None = self.cfg.get("corpus_path")

        # Pull embedder + generator names from cfg, falling back to model_client.
        if not self.cfg.get("generator_model"):
            self.cfg["generator_model"] = getattr(model_client, "model", "unknown_generator")
        if not self.cfg.get("embedder_name"):
            inner = getattr(self.retriever, "_inner", None)
            self.cfg["embedder_name"] = getattr(inner, "model_name", "unknown_embedder")

        self._cache = _default_cache(self.cfg)
        self._cache.ensure_parent()
        self._synth_index = _SynthQueryIndex(self._cache.synth_jsonl)
        self._avg_index = _AveragedEmbeddingIndex(
            self._cache.embeddings_pt, self._cache.tool_order_json,
        )
        self._cost_log = _CostLog(self._cache.cost_log)
        self._index_built = False

        # Corpus integrity baseline. Recorded on first ensure_index call.
        self._corpus_checksum_baseline: str | None = None

    # ------------------------------------------------------------------
    # Corpus integrity (Phase 0 of every run)
    # ------------------------------------------------------------------

    def _record_corpus_checksum(self, when: str) -> str | None:
        if not self._corpus_path:
            return None
        sha = _sha256_file(Path(self._corpus_path))
        rec = {"when": when, "ts": time.time(), "path": self._corpus_path, "sha256": sha}
        self._cache.corpus_checksums.parent.mkdir(parents=True, exist_ok=True)
        with self._cache.corpus_checksums.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        return sha

    def assert_corpus_unchanged(self) -> None:
        """Post-run check: original corpus byte-identical to pre-run."""
        if not self._corpus_path or not self._corpus_checksum_baseline:
            return
        now = _sha256_file(Path(self._corpus_path))
        if now != self._corpus_checksum_baseline:
            raise RuntimeError(
                f"Re-Invoke CORPUS INTEGRITY VIOLATION: "
                f"{self._corpus_path} sha256 changed "
                f"({self._corpus_checksum_baseline[:12]} -> {now[:12]})"
            )
        self._record_corpus_checksum(when="post_run")

    # ------------------------------------------------------------------
    # Index build — offline, paper §3.1 + §4.3
    # ------------------------------------------------------------------

    async def _ensure_index(self, tool_catalog: list[dict]) -> None:
        """Build (or load) both sidecar indices.

        Steps:
            1. Load synth-query cache if present.
            2. For any missing tools, generate m synth queries each.
            3. Save synth-query cache.
            4. Load averaged-embedding tensor if present.
            5. Otherwise: encode m augmented docs per tool, average, save.
        """
        if self._index_built:
            return

        if self._corpus_path and self._corpus_checksum_baseline is None:
            self._corpus_checksum_baseline = self._record_corpus_checksum(when="pre_run")

        # (1) (2) (3) synth queries
        self._synth_index.load()
        missing = [t for t in tool_catalog
                   if _tool_id_from_dict(t) not in self._synth_index._data
                   or len(self._synth_index._data.get(_tool_id_from_dict(t), [])) < self._k_synth]
        if missing:
            logger.warning(
                "Re-Invoke index build: generating %d synth queries each for %d tools "
                "via %s (this may take a while).",
                self._k_synth, len(missing), self.cfg.get("generator_model"),
            )
            sem = asyncio.Semaphore(self._index_concurrency)

            async def gen_one(tool: dict) -> tuple[str, list[str]]:
                async with sem:
                    qs = await self._gen_synth_queries(tool)
                return _tool_id_from_dict(tool), qs

            # Chunked checkpointing: save the synth-query cache after every
            # chunk so a crash mid-build (NFS fault, OOM, kill) loses at most
            # one chunk and resumes from the last save on restart (load() +
            # `missing` recompute skips already-cached tools). Critical for the
            # large `web` split (~373k generator calls / several hours).
            chunk = int(self.cfg.get("index_checkpoint_chunk", 500))
            for c0 in range(0, len(missing), chunk):
                batch = missing[c0:c0 + chunk]
                results = await asyncio.gather(*[gen_one(t) for t in batch],
                                                return_exceptions=True)
                for r in results:
                    if isinstance(r, Exception):
                        logger.warning("Re-Invoke: synth-gen failure: %s", r)
                        continue
                    tid, qs = r
                    self._synth_index.set(tid, qs)
                self._synth_index.save()  # checkpoint
                logger.info("Re-Invoke index: %d/%d tools generated (checkpointed)",
                            min(c0 + chunk, len(missing)), len(missing))

        # (4) (5) averaged embeddings
        if not self._avg_index.load():
            await self._build_averaged_embeddings(tool_catalog)

        self._index_built = True

    async def _gen_synth_queries(self, tool: dict) -> list[str]:
        """Generate self._k_synth synthetic queries for one tool (Fig. 4).

        Each query comes from an independent LLM call. The paper averages
        over the m embeddings; we do the same downstream. The m samples
        themselves are the source of diversity.
        """
        api_doc = _format_api_doc(tool)
        prompt = _FIG4_QUERY_GENERATOR_PROMPT.format(api_doc_json=api_doc)

        queries: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        for sample_idx in range(self._k_synth):
            try:
                # Single-block prompt (paper Fig. 4 has no chat split).
                resp = await self.model_client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self._synth_temperature,
                    seed=self._seed + sample_idx,  # vary seed per sample
                    max_tokens=self._max_tokens,
                )
                raw = (resp.content or "").strip()
                # First non-empty line is the query (the prompt ends with
                # "The relevant query is:" — model should produce one line).
                first = next((ln.strip() for ln in raw.splitlines() if ln.strip()), "")
                if first:
                    queries.append(first)
                # NormalizedResponse exposes token counts as flat attrs.
                prompt_tokens += int(getattr(resp, "input_tokens", 0) or 0)
                completion_tokens += int(getattr(resp, "output_tokens", 0) or 0)
            except Exception as exc:
                logger.warning("Re-Invoke: synth call %d for %s failed: %s",
                               sample_idx, _tool_id_from_dict(tool), exc)

        self._cost_log.append(
            phase="index_build",
            n_calls=self._k_synth,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            extra={"tool_id": _tool_id_from_dict(tool), "kept": len(queries)},
        )
        return queries

    async def _build_averaged_embeddings(self, tool_catalog: list[dict]) -> None:
        """Embed m augmented docs per tool, average → tensor; persist."""
        inner = getattr(self.retriever, "_inner", None)
        if inner is None or not hasattr(inner, "encode_corpus"):
            raise RuntimeError(
                "Re-Invoke requires direct access to a local embedder via "
                "retriever._inner.encode_corpus(). RemoteToolRetriever paths "
                "are not supported in this build."
            )

        tool_ids: list[str] = []
        all_aug_docs: list[str] = []
        copies_per_tool: list[int] = []  # m_actual per tool (may be < self._k_synth on failures)

        for tool in tool_catalog:
            tid = _tool_id_from_dict(tool)
            tool_ids.append(tid)
            tool_doc = _tool_doc_string(tool)
            synth_qs = self._synth_index.get(tid) or []
            if not synth_qs:
                # Fall back to a single un-augmented copy so the tool is
                # still embedded (avoids dropping it from the corpus).
                all_aug_docs.append(_AUGMENTED_DOC_FORMAT.format(
                    tool_doc=tool_doc, synth_query="",
                ))
                copies_per_tool.append(1)
                continue
            for q in synth_qs[: self._k_synth]:
                all_aug_docs.append(_AUGMENTED_DOC_FORMAT.format(
                    tool_doc=tool_doc, synth_query=q,
                ))
            copies_per_tool.append(min(len(synth_qs), self._k_synth))

        logger.warning(
            "Re-Invoke: encoding %d augmented docs across %d tools (m_avg=%.2f).",
            len(all_aug_docs), len(tool_catalog),
            sum(copies_per_tool) / max(1, len(copies_per_tool)),
        )

        # encode_corpus returns a [N, D] tensor (cpu) per Toolret/retriever.py.
        all_emb = inner.encode_corpus(all_aug_docs)
        if not torch.is_tensor(all_emb):
            all_emb = torch.tensor(all_emb)
        all_emb = all_emb.detach().to("cpu", dtype=torch.float32)

        # L2-normalize so cosine = dot (matches ToolRetriever's util.cos_sim).
        all_emb = torch.nn.functional.normalize(all_emb, dim=1)

        # Average per tool.
        avg_rows: list[torch.Tensor] = []
        cursor = 0
        for m_i in copies_per_tool:
            block = all_emb[cursor: cursor + m_i]
            cursor += m_i
            avg = block.mean(dim=0)
            # Re-normalize the averaged vector for cosine-sim consistency.
            avg = torch.nn.functional.normalize(avg, dim=0)
            avg_rows.append(avg)
        avg_tensor = torch.stack(avg_rows, dim=0)
        self._avg_index.save(tool_ids, avg_tensor)

    # ------------------------------------------------------------------
    # Intent extraction — online, paper §3.2 + Figure 5
    # ------------------------------------------------------------------

    async def _extract_intents(self, query: str) -> tuple[list[str], int, int]:
        prompt = _FIG5_INTENT_EXTRACTOR_PROMPT.format(query=query)
        try:
            resp = await self.model_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=self._intent_temperature,
                seed=self._seed,
                max_tokens=self._max_tokens,
            )
        except Exception as exc:
            logger.warning("Re-Invoke: intent-extract LLM call failed: %s", exc)
            return [query], 0, 0

        raw = (resp.content or "").strip()
        intents = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if not intents:
            intents = [query]
        pt = int(getattr(resp, "input_tokens", 0) or 0)
        ct = int(getattr(resp, "output_tokens", 0) or 0)
        return intents[: self._max_intents], pt, ct

    # ------------------------------------------------------------------
    # Multi-view similarity ranking — Appendix C Algorithm 2
    # ------------------------------------------------------------------

    def _embed_intents(self, intents: list[str]) -> torch.Tensor:
        inner = self.retriever._inner
        embs = inner.encode_sentence(intents)
        if not torch.is_tensor(embs):
            embs = torch.tensor(embs)
        embs = embs.detach().to("cpu", dtype=torch.float32)
        if embs.dim() == 1:
            embs = embs.unsqueeze(0)
        embs = torch.nn.functional.normalize(embs, dim=1)
        return embs

    def _multi_view_rank(
        self,
        intent_embs: torch.Tensor,        # [n_intents, D]
        tool_embs: torch.Tensor,           # [n_tools, D]
        tool_ids: list[str],
    ) -> list[tuple[str, float, float]]:
        """Algorithm 2 (Appendix C): tuple (rev_rank, sim), max over intents.

        Returns ``[(tool_id, rank_score, cosine_sim), ...]`` in final ranked
        order. ``rank_score`` is a strictly-decreasing position score that
        preserves the Algorithm-2 order under a score-descending sort (so
        pytrec_eval ranks faithfully — see the note in the ranking loop
        below); ``cosine_sim`` is the raw max-over-intents similarity, kept
        only for transparency.
        """
        sim = intent_embs @ tool_embs.T   # cosine if both normalized
        n_intents, n_tools = sim.shape

        # For each intent, compute rev_rank: ascending rank (lowest sim → 1).
        # We use argsort ascending then assign rank=1..n_tools.
        best_rev_rank = torch.zeros(n_tools, dtype=torch.long)
        best_sim = torch.full((n_tools,), float("-inf"))
        for i in range(n_intents):
            scores_i = sim[i]
            asc = torch.argsort(scores_i, descending=False)
            rev_rank = torch.empty(n_tools, dtype=torch.long)
            rev_rank[asc] = torch.arange(1, n_tools + 1, dtype=torch.long)
            # Lex tuple (rev_rank, sim): replace where strictly greater.
            update = (rev_rank > best_rev_rank) | (
                (rev_rank == best_rev_rank) & (scores_i > best_sim)
            )
            best_rev_rank = torch.where(update, rev_rank, best_rev_rank)
            best_sim = torch.where(update, scores_i, best_sim)

        # Sort tools by (best_rev_rank desc, best_sim desc).
        # Stable sort: first by sim desc, then by rev_rank desc.
        order_by_sim = torch.argsort(best_sim, descending=True, stable=True)
        rev_rank_sorted = best_rev_rank[order_by_sim]
        order_by_rev = torch.argsort(rev_rank_sorted, descending=True, stable=True)
        final = order_by_sim[order_by_rev]

        # CRITICAL: the downstream scorer
        # (pytrec_eval, via cal_eval) ranks documents BY SCORE. If we emit
        # the cosine `best_sim` as the score, pytrec re-sorts by pure
        # similarity and silently DISCARDS the Algorithm-2 tuple order we
        # just computed — the whole multi-view ranking would be a no-op at
        # eval time. So we emit a strictly-decreasing rank score that
        # reproduces `final` (the (rev_rank, sim) lexicographic order) exactly
        # under a score-descending sort. The cosine sims are preserved
        # separately (returned as a 3rd tuple element) for transparency.
        idxs = final[: self.top_k].tolist()
        n_ret = len(idxs)
        out: list[tuple[str, float, float]] = []
        for pos, idx in enumerate(idxs):
            rank_score = float(n_ret - pos)  # strictly decreasing: n_ret, n_ret-1, ...
            out.append((tool_ids[idx], rank_score, float(best_sim[idx])))
        return out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def retrieve(self, query: str, *, tool_catalog: list[dict]) -> RetrievedTools:
        t0 = time.monotonic()
        await self._ensure_index(tool_catalog)

        intents, pt, ct = await self._extract_intents(query)
        self._cost_log.append(
            phase="per_query",
            n_calls=1,
            prompt_tokens=pt,
            completion_tokens=ct,
            extra={"query_head": query[:120], "n_intents": len(intents)},
        )

        intent_embs = self._embed_intents(intents)
        top = self._multi_view_rank(
            intent_embs=intent_embs,
            tool_embs=self._avg_index.tensor,
            tool_ids=self._avg_index.tool_ids,
        )
        tool_ids = [t for t, _, _ in top]
        scores = [rank_score for _, rank_score, _ in top]   # order-preserving rank scores
        cosine_sims = [sim for _, _, sim in top]             # raw cosine, for transparency

        latency_ms = (time.monotonic() - t0) * 1000.0
        return RetrievedTools(
            tool_ids=tool_ids,
            scores=scores,
            metadata={
                "intents": intents,
                "n_synth_queries_per_tool": self._k_synth,
                "embedder_name": self.cfg.get("embedder_name"),
                "generator_model": self.cfg.get("generator_model"),
                "latency_ms": latency_ms,
                "cache_dir": str(self._cache.synth_jsonl.parent),
                "algorithm": "appendix_c_alg_2",
                "cosine_sims": cosine_sims,
                "score_semantics": (
                    "scores are strictly-decreasing rank positions (top_k-i) that "
                    "preserve the Appendix C Algorithm 2 tuple order under a "
                    "score-descending sort (pytrec_eval); cosine_sims holds the raw "
                    "max-over-intents cosine similarity for each returned tool"
                ),
            },
        )
