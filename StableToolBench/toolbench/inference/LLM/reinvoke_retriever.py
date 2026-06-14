"""Re-Invoke retriever adapter for StableToolBench open-domain DFSDT.

Drop-in replacement for ``ToolRetriever`` (the ``Retriever`` contract consumed by
``toolbench/retrieval/services.py:retrieve_rapidapi_tools``) that ranks candidate
tools with the Re-Invoke method (Chen et al., arXiv 2408.01875 / ACL Findings
2024.270): synthetic-query-augmented tool embeddings (offline) + online intent
extraction + Appendix C Algorithm 2 multi-view ranking.

It wraps the already-validated ``baselines.reinvoke.ReInvokeBaseline`` and exposes
the synchronous ``retrieving(query, top_k)`` + ``self.corpus`` interface that STB's
``services.py`` expects. Selected via the ``--reinvoke`` flag (see
``qa_pipeline_open_domain.py`` and ``rapidapi.py:get_retriever``).

Design notes
------------
- **Static / normal retrieval only.** Re-Invoke produces the candidate set once per
  query and never re-retrieves mid-trajectory; use with ``--retrieve_mode normal``.
  It changes *which* K tools enter ``self.functions`` and nothing about the DFSDT
  reasoning / execution / Finish protocol (rapidapi.py:204-240).
- **Corpus isolation.** Re-Invoke writes its averaged-embedding sidecar to its own
  cache dir; this adapter NEVER writes near the STB des_corpus. We deliberately use
  a lightweight in-process SimCSE embedder instead of ``ToolRetriever`` so that no
  ``<corpus>_des_corpus_<model>_embeddings.pt`` is ever written to the shared
  (read-only) corpus directory — and so we skip ToolRetriever's unconditional
  full-base-corpus encode at every process init.
- **Device.** The inner embedder honors ``RETRIEVER_DEVICE`` (default ``cuda``).
  The offline index build runs on GPU; the online solve can run the (intent-only)
  encoder on CPU to avoid GPU contention across parallel solve processes.
- **corpus_id round-trip.** STB keys retrieved tools by the corpus line index
  (``services.py:33`` does ``retriever.corpus[corpus_id]``). We inject
  ``id=str(corpus_id)`` into each tool dict so Re-Invoke's ``tool_ids`` round-trip
  back to ``corpus_id``.
- **Name standardization.** Returned ``category``/``tool_name``/``api_name`` are
  standardized exactly as ``ToolRetriever`` does (retriever.py:122-124) so the
  on-disk resolution filter in ``services.py:37-45`` finds the tool JSON.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from toolbench.observability.pins import load_pins
from toolbench.utils import change_name, standardize, standardize_category

# ``baselines/`` lives at the repo root, which is NOT on the STB process's
# PYTHONPATH (run_inference.sh sets ``PYTHONPATH=./`` from the StableToolBench
# dir). Add the repo root so ``baselines.*`` is importable.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 4)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from baselines.base import RetrieverAdapter  # noqa: E402
from baselines.reinvoke import ReInvokeBaseline  # noqa: E402


def _resolve_gen_model() -> str:
    """Resolve the generator/intent LLM from the single source of truth.

    Order: env ``REINVOKE_GEN_MODEL`` > ``configs/model_pins.yaml`` agents.cheap_sota
    > explicit failure. Avoids hardcoding a model name that could silently drift
    from the repo pin.
    """
    env = os.environ.get("REINVOKE_GEN_MODEL")
    if env:
        return env
    try:
        return load_pins(Path(_REPO_ROOT))["agents"]["cheap_sota"]
    except Exception as exc:
        raise RuntimeError(
            "Could not resolve Re-Invoke generator model from configs/model_pins.yaml; "
            "set REINVOKE_GEN_MODEL to override."
        ) from exc


class _SimCSEEmbedder:
    """Minimal device-configurable SimCSE (CLS-pooling) embedder.

    Exposes the two methods Re-Invoke calls on ``retriever._inner``:
    ``encode_corpus`` (offline index build) and ``encode_sentence`` (online intent
    encoding), plus ``model_name`` (Re-Invoke's cache key). Mirrors the CLS-pooling
    in ``toolbench/inference/LLM/retriever.py:51-72`` but with a configurable device
    and WITHOUT loading/encoding the full base corpus (``ToolRetriever.__init__``
    does that unconditionally and writes a ``.pt`` next to the corpus, which we must
    avoid for isolation).

    Args:
        model_path: HF path/name of the SimCSE model.
        device: 'cuda' or 'cpu'. Defaults to env ``RETRIEVER_DEVICE`` then 'cuda'.
        batch_size: encode batch size.
    """

    def __init__(self, model_path: str, device: str | None = None, batch_size: int = 32):
        self.model_path = model_path
        self.model_name = model_path.split("/")[-1]
        self.device = device or os.environ.get("RETRIEVER_DEVICE", "cuda")
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device).eval()

    @torch.no_grad()
    def _encode(self, sentences) -> torch.Tensor:
        """CLS-pool encode a list of strings → [N, D] float tensor on CPU."""
        if isinstance(sentences, str):
            sentences = [sentences]
        out: list[torch.Tensor] = []
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i : i + self.batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
            cls = outputs.last_hidden_state[:, 0].cpu()  # [B, D] — CLS token, matches retriever.py:59
            out.append(cls)
        return torch.cat(out, dim=0)

    def encode_corpus(self, sentences) -> torch.Tensor:
        return self._encode(sentences)

    def encode_sentence(self, sentences) -> torch.Tensor:
        return self._encode(sentences)

    def retrieving(self, *args, **kwargs):  # pragma: no cover - never called
        raise NotImplementedError(
            "Re-Invoke performs its own multi-view ranking; the inner embedder's "
            "retrieving() is never invoked."
        )


class ReInvokeSTBRetriever:
    """Re-Invoke drop-in retriever for StableToolBench (``Retriever`` contract).

    Builds/loads the Re-Invoke index over an STB des_corpus once at ``__init__``,
    then answers ``retrieving()`` per query with Algorithm-2 multi-view ranking.

    Args:
        corpus_path: Path to an STB ``des_corpus.json`` (JSONL of
            ``{functionality, cate_name, tool_name, api_name}``).
        model_path: SimCSE embedder path.
        retrieved_api_nums: Final K the pipeline asks for (we over-fetch 3x so the
            services.py on-disk filter still yields ~K, mirroring ToolRetriever).
        reinvoke_cfg: dict of Re-Invoke knobs (k_synth, max_intents, cache_dir,
            generator_model, index_build_concurrency, ...).
        device: embedder device ('cuda'/'cpu'); defaults to env ``RETRIEVER_DEVICE``.
    """

    def __init__(
        self,
        corpus_path: str,
        model_path: str,
        retrieved_api_nums: int = 5,
        reinvoke_cfg: dict | None = None,
        device: str | None = None,
        require_prebuilt: bool = False,
    ):
        cfg = dict(reinvoke_cfg or {})
        self.corpus_path = corpus_path
        # Over-fetch so the on-disk resolution filter (services.py:37-45) still
        # leaves ~retrieved_api_nums tools, mirroring ToolRetriever's top_k*3.
        self._overfetch = max(1, int(retrieved_api_nums)) * 3
        corpus_name = next((g for g in ("G1", "G2", "G3") if f"/{g}/" in corpus_path), "G1")

        # --- load des_corpus (JSONL) ---
        # self.corpus      : list[str] of `functionality`, index == corpus_id
        #                    (services.py:33 does retriever.corpus[corpus_id]).
        # self._tool_index : corpus_id -> standardized (category, tool_name, api_name)
        # self._tool_catalog: dicts fed to Re-Invoke (id == str(corpus_id)).
        self.corpus: list[str] = []
        self._tool_index: list[tuple[str, str, str]] = []
        self._tool_catalog: list[dict] = []
        with open(corpus_path, "r", encoding="utf-8") as fh:
            cid = 0
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                functionality = rec.get("functionality", "")
                cat_raw = rec.get("cate_name", "")
                tool_raw = rec.get("tool_name", "")
                api_raw = rec.get("api_name", "")
                self.corpus.append(functionality)
                # Standardize EXACTLY as ToolRetriever (retriever.py:122-124) so the
                # services.py on-disk resolution filter matches the tool tree layout.
                self._tool_index.append(
                    (
                        standardize_category(cat_raw),
                        standardize(tool_raw),
                        change_name(standardize(api_raw)),
                    )
                )
                self._tool_catalog.append(
                    {
                        "id": str(cid),
                        "category": cat_raw,
                        "tool_name": tool_raw,
                        "api_name": api_raw,
                        "description": functionality,
                    }
                )
                cid += 1

        # --- model client (generator + intent extractor) ---
        gen_model = cfg.get("generator_model") or _resolve_gen_model()
        from toolbench.inference.LLM.clients import make_client

        client = make_client(model=gen_model, api_key=os.environ.get("OPENAI_API_KEY", ""))

        # --- inner embedder (device-configurable, no base-corpus .pt write) ---
        embedder = _SimCSEEmbedder(model_path, device=device)
        adapter = RetrieverAdapter(embedder)

        # Anchor the cache dir to the repo root so the offline build (cwd=repo) and
        # the solve (cwd=StableToolBench) resolve the SAME cache — a relative default
        # would split into two cwd-dependent dirs and silently trigger a rebuild.
        runtime_root = os.environ.get("FITTEXT_RUNTIME_ROOT") or os.path.join(_REPO_ROOT, "runs")
        cache_dir = cfg.get("cache_dir") or os.path.join(runtime_root, "reinvoke_stb", "cache")
        if not os.path.isabs(cache_dir):
            cache_dir = os.path.join(_REPO_ROOT, cache_dir)

        # --- Re-Invoke baseline (top_k must cover the over-fetch) ---
        self._baseline = ReInvokeBaseline(
            client,
            adapter,
            top_k=self._overfetch,
            k_synth=int(cfg.get("k_synth", 10)),
            max_intents=int(cfg.get("max_intents", 3)),
            synth_temperature=float(cfg.get("synth_temperature", 0.7)),
            intent_temperature=float(cfg.get("intent_temperature", 0.0)),
            generator_model=gen_model,
            embedder_name=model_path,
            benchmark_tag=f"stb_{corpus_name}",
            cache_dir=cache_dir,
            corpus_path=corpus_path,
            index_build_concurrency=int(cfg.get("index_build_concurrency", 8)),
            index_checkpoint_chunk=int(cfg.get("index_checkpoint_chunk", 500)),
            max_tokens=int(cfg.get("max_tokens", 512)),
            seed=int(cfg.get("seed", 42)),
        )

        # Guard a multi-process solve from silently (re)building the index: when a
        # pre-built cache is required (the solve path sets this) and the averaged
        # tensor is absent, fail fast instead of generating synth queries on the fly
        # (surprise spend + concurrent-build race across solve processes).
        emb_pt = self._baseline._cache.embeddings_pt
        if require_prebuilt and not os.path.exists(emb_pt):
            raise RuntimeError(
                f"Re-Invoke index not pre-built for {corpus_name} (missing {emb_pt}). "
                f"Run baselines/build_reinvoke_stb_index.py first, or set "
                f"REINVOKE_REQUIRE_PREBUILT=0 to allow an on-the-fly build."
            )

        # Build-or-load the index ONCE up front. For a solve this just LOADS the
        # pre-built cache (fast); only the offline builder reaches the synth-gen path.
        asyncio.run(self._baseline._ensure_index(self._tool_catalog))

        # Corpus-identity guard: a stale cache from a different/changed
        # des_corpus would map returned numeric ids into THIS _tool_index and silently
        # return wrong tools (or IndexError). The cached averaged index must have one
        # row per catalog tool, ordered str(0..N-1).
        tens = self._baseline._avg_index.tensor
        n_tools = len(self._tool_catalog)
        n_rows = None if tens is None else int(tens.shape[0])
        if n_rows != n_tools:
            raise RuntimeError(
                f"Re-Invoke cache/corpus mismatch for {corpus_name}: averaged index has "
                f"{n_rows} rows but corpus has {n_tools} tools. Stale cache? Rebuild with "
                f"baselines/build_reinvoke_stb_index.py."
            )
        if list(self._baseline._avg_index.tool_ids) != [str(i) for i in range(n_tools)]:
            raise RuntimeError(
                f"Re-Invoke cache tool-id ordering mismatch for {corpus_name} (corpus "
                f"changed since build?). Rebuild with baselines/build_reinvoke_stb_index.py."
            )

    def __deepcopy__(self, memo):
        """Return self — the retriever is a shared, read-only service.

        DFSDT deepcopies the env per child node (DFS.py:420) BEFORE nulling the
        retriever (DFS.py:421). Deep-copying this object would try to copy the
        OpenAI httpx client (a ``_thread.RLock``, not pickleable) and the torch
        embedder. In normal mode the retriever is used exactly once at env-init and
        is never invoked on child nodes, so sharing one instance across the search
        tree is correct and safe (and avoids the RLock crash that ToolRetriever — a
        pure-torch object — never hit).
        """
        return self

    def retrieving(self, query, top_k: int = 5, excluded_tools=None):
        """Synchronous ``Retriever``-protocol entry consumed by ``services.py``.

        Returns a list of dicts ``{category, tool_name, api_name, corpus_id, score}``
        in Re-Invoke's Algorithm-2 ranked order. ``services.py`` consumes ORDER (it
        takes the first ``retrieved_api_nums`` that resolve on disk); ``score`` is a
        strictly-decreasing rank proxy, not cosine.

        Args:
            query: the user query (env init passes ``self.input_description``).
            top_k: requested count from services.py; we return ``self._overfetch``
                (3x) ranked tools so the on-disk filter still yields ~top_k.
            excluded_tools: unused in normal mode (kept for protocol parity).
        """
        rt = asyncio.run(self._baseline.retrieve(query, tool_catalog=self._tool_catalog))
        n = max(int(top_k) * 3, self._overfetch)
        results: list[dict] = []
        for tool_id, score in zip(rt.tool_ids[:n], rt.scores[:n]):
            cid = int(tool_id)
            category, tool_name, api_name = self._tool_index[cid]
            results.append(
                {
                    "category": category,
                    "tool_name": tool_name,
                    "api_name": api_name,
                    "corpus_id": cid,
                    "score": float(score),
                }
            )
        return results
