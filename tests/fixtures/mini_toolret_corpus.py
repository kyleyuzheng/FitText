"""Synthetic mini ToolRet corpus + queries fixture builder.

Materialises a 5-tool, 3-query ToolRet-shaped corpus on disk so the E2E
pipeline can exercise the real :class:`Toolret.retriever.ToolRetriever`
without requiring the 43k-tool production corpus to be present.

LAYOUT
------
::

    <root>/
      mini/
        des_corpus.json                       # JSONL: {"id": ..., "description": ...}
        queries.json                          # JSONL: {"id": ..., "query": ..., "labels": "[{\"id\":..}]"}
        des_corpus_all-MiniLM-L6-v2_embeddings.pt  # torch.Tensor [5, 384]
        ...                                   # cached per embedder

The pre-built `*_embeddings.pt` is keyed by the embedder's
``model_path.split('/')[-1]`` (same key used by ``ToolRetriever.build_corpus_embeddings``)
so the retriever's on-disk cache hits immediately and no encode pass runs at
fixture consumption time.  Building the fixture itself does one tiny
``embedder.encode(5 strings)`` call on CPU — trivial.

WHY ``.json`` IS REALLY JSONL
-----------------------------
``Toolret/retriever.py:process_des_retrieval_document`` reads the file line by
line with ``json.loads(line)`` despite the ``.json`` extension.  The
ToolRet repo committed it that way; we match the format exactly.

WHY A LOCAL FIXTURE INSTEAD OF DISABLING THE RETRIEVER
-------------------------------------------------------
Disabling :class:`ToolRetriever` would defeat the purpose of the E2E test —
the gate must cover the same code path that runs in production.  This
fixture plugs into ``cfg.extra["toolret_corpus_root"]`` /
``cfg.extra["toolret_queries_root"]`` overrides exposed by the ToolRet
adapter, so the real retriever loads + encodes the synthetic corpus and
returns real cosine-similarity scores.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Force CPU encoding before any torch / sentence-transformers import.
# The E2E test runs on a shared host where the GPU may be saturated; the
# 5-tool corpus encodes in <1s on CPU.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


# ---------------------------------------------------------------------------
# Synthetic content
# ---------------------------------------------------------------------------

# 5 tools spanning unrelated domains so cosine similarities are well-separated.
# Descriptions are kept short (<=120 chars) — same shape as the production
# ``des_corpus.json`` produced by ``Toolret/data_preprocess/build_des_corpus.py``.
_TOOLS: list[dict[str, str]] = [
    {
        "id": "tool_weather_001",
        "description": "Weather API. Returns current temperature, humidity, and conditions for a given city.",
    },
    {
        "id": "tool_calc_002",
        "description": "Calculator. Evaluates basic arithmetic expressions and returns numeric results.",
    },
    {
        "id": "tool_translate_003",
        "description": "Language translator. Translates text between supported languages.",
    },
    {
        "id": "tool_music_004",
        "description": "Music search. Looks up song metadata by artist, title, or genre.",
    },
    {
        "id": "tool_food_005",
        "description": "Recipe finder. Returns cooking recipes matching ingredient lists or cuisine.",
    },
]

# 3 queries with ground-truth labels referencing the tool ids above.
_QUERIES: list[dict[str, Any]] = [
    {
        "id": "mini_q_001",
        "query": "What is the weather in Paris?",
        "labels": [{"id": "tool_weather_001", "relevance": 1}],
    },
    {
        "id": "mini_q_002",
        "query": "Translate 'hello' to Spanish.",
        "labels": [{"id": "tool_translate_003", "relevance": 1}],
    },
    {
        "id": "mini_q_003",
        "query": "Find a pasta recipe with tomatoes.",
        "labels": [{"id": "tool_food_005", "relevance": 1}],
    },
]

# Split name used by the synthetic corpus.  Registered in
# ``Toolret/eval_toolret.dataset_categories`` as ``["mini"]``.
MINI_SPLIT = "mini"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write one JSON object per line to ``path``.

    Args:
        path: Destination file (parent directories must exist).
        records: List of dicts to serialise.
    """
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _build_embeddings(corpus_dir: Path, embedder_name: str) -> Path:
    """Build (or reuse) the on-disk embedding cache for the mini corpus.

    The path matches the cache key used by
    ``ToolRetriever.build_corpus_embeddings``:
    ``<corpus_path>.replace('.json', f'_des_corpus_{model_name}_embeddings.pt')``.

    Args:
        corpus_dir: Directory containing ``des_corpus.json``.
        embedder_name: Full HF model path (e.g.
            ``sentence-transformers/all-MiniLM-L6-v2``).  Only the basename
            is used as the cache key.

    Returns:
        Path to the saved ``.pt`` tensor file.
    """
    # Late imports — only pay the torch / sentence-transformers cost when the
    # fixture is actually instantiated.
    import torch  # noqa: WPS433
    from sentence_transformers import SentenceTransformer  # noqa: WPS433

    model_basename = embedder_name.split("/")[-1]
    corpus_path = corpus_dir / "des_corpus.json"
    embed_path = corpus_dir / corpus_path.name.replace(
        ".json", f"_des_corpus_{model_basename}_embeddings.pt"
    )
    if embed_path.exists():
        return embed_path

    descriptions = [t["description"] for t in _TOOLS]
    # CPU encode — 5 sentences is trivial.
    embedder = SentenceTransformer(embedder_name)
    embeddings = embedder.encode(descriptions, convert_to_tensor=True)
    torch.save(embeddings, embed_path)
    return embed_path


def build_corpus(
    tmp_path: Path,
    *,
    embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    precompute_embeddings: bool = True,
) -> Path:
    """Materialise a 5-tool, 3-query ToolRet-shaped corpus under ``tmp_path``.

    Args:
        tmp_path: Caller-provided temp directory (typically pytest's tmp_path).
        embedder_name: HF model name used for the precomputed corpus
            embeddings.  Must match ``cfg.embedder.name`` so the retriever's
            on-disk cache hits.
        precompute_embeddings: If True, run a tiny CPU encode now and save the
            resulting tensor next to ``des_corpus.json`` so the retriever
            skips the encode step at first use.  Set False to force the
            encode-and-cache code path inside ``ToolRetriever`` (slower but
            exercises that branch end-to-end).

    Returns:
        Absolute path to the corpus root.  Pass this as
        ``cfg.extra["toolret_corpus_root"]`` AND ``cfg.extra["toolret_queries_root"]``
        in the test setup.

    Side effects:
        Writes ``<tmp_path>/mini/des_corpus.json``,
        ``<tmp_path>/mini/queries.json``, and (optionally)
        ``<tmp_path>/mini/des_corpus_<embedder_basename>_embeddings.pt``.
    """
    corpus_root = Path(tmp_path) / "toolret_mini"
    split_dir = corpus_root / MINI_SPLIT
    split_dir.mkdir(parents=True, exist_ok=True)

    # 1) Corpus (JSONL, .json extension matches production).
    _write_jsonl(split_dir / "des_corpus.json", _TOOLS)

    # 2) Queries — stored as JSONL with the same shape as the HF
    #    ``mangopy/ToolRet-Queries`` rows (id, query, labels-as-JSON-string).
    query_records: list[dict[str, Any]] = []
    for q in _QUERIES:
        query_records.append(
            {
                "id": q["id"],
                "query": q["query"],
                "labels": json.dumps(q["labels"], ensure_ascii=False),
            }
        )
    _write_jsonl(split_dir / "queries.json", query_records)

    # 3) Optional pre-computed embeddings.
    if precompute_embeddings:
        _build_embeddings(split_dir, embedder_name)

    return corpus_root


def load_local_queries(
    corpus_root: Path,
    split: str,
    n_queries: int | None,
    shard: int = 0,
    total_shards: int = 1,
) -> list[dict[str, Any]]:
    """Load queries from a local synthetic corpus split.

    This is the back-door used by the ToolRet adapter when
    ``cfg.extra["toolret_queries_root"]`` is set — bypasses the HuggingFace
    ``load_dataset`` call.  Returns rows in the exact shape that
    ``mangopy/ToolRet-Queries`` would have produced (``id``, ``query``,
    ``labels`` as a JSON-string).

    Args:
        corpus_root: Path returned by :func:`build_corpus`.
        split: Split name (e.g. ``"mini"``).
        n_queries: Cap on the number of queries returned.  ``None`` = all.
        shard: 0-indexed shard for cross-host parallelism.
        total_shards: Total number of shards.

    Returns:
        List of dicts with keys ``id``, ``query``, ``labels``.
    """
    queries_file = Path(corpus_root) / split / "queries.json"
    if not queries_file.exists():
        return []

    out: list[dict[str, Any]] = []
    with queries_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = str(rec["id"])
            if total_shards > 1 and (hash(qid) % total_shards) != shard:
                continue
            out.append(rec)
            if n_queries is not None and len(out) >= n_queries:
                break
    return out
