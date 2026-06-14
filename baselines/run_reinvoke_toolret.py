#!/usr/bin/env python3
"""Dedicated runner: Re-Invoke baseline (baselines/reinvoke.py) on ToolRet.

WHY THIS SCRIPT EXISTS
----------------------
The FitText ToolRet adapter (`toolbench/runner/adapters/toolret.py`) only
maps FitText *variants* (single_pass / multi_turn / scattershot / memetic /
just_query) to ToolRet's in-tree strategies via `_variant_to_strategy`.
It has NO dispatch path for the `baselines/` ABC family, so `ReInvokeBaseline`
cannot be run through `run.py`. This script is the dedicated entry point.

WHAT IT DOES (paper §3 faithful, three components)
--------------------------------------------------
1. Load the split's `des_corpus.json` → tool_catalog of `{id, description}`.
   The corpus is READ-ONLY; sha256 is recorded before and after.
2. Build a `ToolRetriever` (ToolRet's own class) purely to reuse its local
   embedder (`encode_corpus` / `encode_sentence`). Its base-corpus `.pt` is
   loaded from disk if present; we do NOT use its `retrieving()`.
3. Instantiate `ReInvokeBaseline` with FitText's pinned generator LLM and
   embedder. First `retrieve()` triggers the offline index build (m synth
   queries/tool → averaged augmented-doc embedding).
4. Load queries + qrels with the SAME `load_dataset("mangopy/ToolRet-Queries",
   subset)` path as `Toolret/eval_toolret.py` — qid=`sample['id']`,
   qrels[qid] = {str(x['id']): int(x['relevance']) for x in labels}.
5. Score with `Toolret.eval_toolret.cal_eval` — IMPORTED, not reimplemented,
   so metrics are computed identically to the main ToolRet table
   (NDCG/MAP/Recall/Precision @5/@10/@20).
6. Emit: per-query JSONL (intents + retrieved set + per-q metrics), the
   cost split (index_build vs per_query from the baseline's cost_log), and
   the corpus checksum pre/post proof.

Models are resolved from `configs/model_pins.yaml`; the generator/intent LLM
uses the FitText main pin and the embedder uses the ToolRet embedder pin.

USAGE
-----
Real run (pays LLM cost — index build dominates):
    CUDA_VISIBLE_DEVICES=4 OPENAI_API_KEY=$OPENAI_API_KEY \
    python baselines/run_reinvoke_toolret.py --split code --k_synth 10

Zero-cost offline plumbing smoke (stub LLM, local embedder only):
    CUDA_VISIBLE_DEVICES=4 \
    python baselines/run_reinvoke_toolret.py --split code --stub_llm \
        --k_synth 1 --limit_queries 20
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Repo + StableToolBench on path (mirror run.py).
_REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (_REPO_ROOT, _REPO_ROOT / "StableToolBench"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from baselines.base import RetrieverAdapter  # noqa: E402
from baselines.reinvoke import ReInvokeBaseline, _tool_id_from_dict  # noqa: E402
from toolbench.observability.pins import load_pins  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_reinvoke_toolret")

# ToolRet split → list of HF dataset subset names (mirrors
# Toolret/eval_toolret.py::dataset_categories — kept in sync manually).
DATASET_CATEGORIES = {
    "web": ["autotools-food", "autotools-music", "restgpt-tmdb", "autotools-weather",
            "restgpt-spotify", "toolbench", "toollens", "apibank", "mnms", "reversechain",
            "tooleyes", "ultratool", "t-eval-dialog", "t-eval-step", "apigen", "rotbench",
            "taskbench-daily", "toolace", "toolemu"],
    "code": ["gorilla-pytorch", "gorilla-tensor", "gorilla-huggingface", "craft-tabmwp",
             "craft-vqa", "craft-math-algebra", "toolink"],
    "customized": ["gpt4tools", "taskbench-huggingface", "taskbench-multimedia",
                   "toolbench-sam", "toolalpaca", "gta", "tool-be-honest", "appbench", "metatool"],
}


# ---------------------------------------------------------------------------
# Stub LLM (offline smoke only — zero API cost)
# ---------------------------------------------------------------------------

class _StubResponse:
    """Minimal NormalizedResponse-shaped stub."""
    def __init__(self, content: str) -> None:
        self.content = content
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached_input_tokens = 0


class _StubLLM:
    """Deterministic stub: synth query == tool desc echo, intent == raw query.

    Distinguishes the two prompt types by sniffing the Fig.4 vs Fig.5 marker
    in the message. Exercises the full pipeline without any network call.
    """
    model = "stub-llm"

    async def chat_completion(self, *, messages, temperature=0.0, seed=42,
                              max_tokens=None, **kwargs) -> _StubResponse:
        text = messages[-1]["content"] if messages else ""
        if "The relevant query is:" in text:
            # Query-generator path: echo the api_description from the JSON.
            try:
                start = text.index("The API documentation is:") + len("The API documentation is:")
                end = text.index("The relevant query is:")
                doc = json.loads(text[start:end].strip())
                return _StubResponse(doc.get("api_description", "") or "tool query")
            except Exception:
                return _StubResponse("tool query")
        # Intent-extractor path: return the raw query (single intent).
        if "Query:" in text:
            q = text.rsplit("Query:", 1)[-1].split("Intent:")[0].strip()
            return _StubResponse(q or "intent")
        return _StubResponse("intent")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _prepare_corpus_copy(orig_corpus: Path, embedder: str, cache_root: Path) -> Path:
    """Copy des_corpus.json (+ matching base-embedding .pt) into a working dir.

    Corpus isolation: all corpus-directory
    writes (notably ToolRetriever's `<corpus>_des_corpus_<model>_embeddings.pt`)
    must land in our own tmp dir, never the original (read-only, shared) corpus
    directory. We copy the JSON and, if present, the sibling base-embedding .pt
    under the name ToolRetriever expects so it LOADS instead of re-embedding +
    writing.

    Args:
        orig_corpus: Path to the original des_corpus.json (read-only).
        embedder: HF model name (its basename drives the .pt filename).
        cache_root: Root of the Re-Invoke cache/working area.

    Returns:
        Path to the working-copy des_corpus.json.
    """
    import shutil
    model_name = embedder.split("/")[-1]
    # Key the work dir by a hash of the RESOLVED original path + embedder so
    # different corpus roots that share a split name (e.g. two mounts both
    # called `code`) never collide on one copy dir.
    key = hashlib.sha256(f"{orig_corpus.resolve()}|{model_name}".encode()).hexdigest()[:10]
    work_dir = cache_root / "corpus_copy" / f"{orig_corpus.parent.name}_{key}"
    work_dir.mkdir(parents=True, exist_ok=True)
    work_corpus = work_dir / orig_corpus.name
    shutil.copy2(orig_corpus, work_corpus)
    # ToolRetriever derives: corpus_path.replace('.json', f'_des_corpus_{model}_embeddings.pt')
    pt_suffix = f"_des_corpus_{model_name}_embeddings.pt"
    orig_pt = orig_corpus.with_name(orig_corpus.name.replace(".json", pt_suffix))
    work_pt = work_corpus.with_name(work_corpus.name.replace(".json", pt_suffix))
    # Always-fresh: the copy carries a base-embedding .pt IFF the original
    # does. Removing any stale work-copy .pt first guarantees ToolRetriever
    # never loads an embedding tensor that predates the current corpus
    # (guards against stale .pt reuse). NB: Re-Invoke never reads this
    # base tensor anyway (it uses the live encoder), so this is belt-and-
    # suspenders, but it keeps the working dir honest.
    if work_pt.exists():
        work_pt.unlink()
    if orig_pt.exists():
        shutil.copy2(orig_pt, work_pt)
    return work_corpus


def _load_corpus(corpus_path: Path, limit: int | None) -> list[dict]:
    """Read des_corpus.json (one JSON obj per line) → list of {id, description}."""
    tools: list[dict] = []
    with corpus_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tools.append({"id": str(rec["id"]), "description": rec.get("description", "")})
            if limit and len(tools) >= limit:
                break
    return tools


def _summarize_cost(cost_log_path: Path) -> dict:
    """Aggregate the baseline's per-call cost_log.jsonl by phase."""
    agg = {
        "index_build": {"calls": 0, "in_tok": 0, "out_tok": 0},
        "per_query": {"calls": 0, "in_tok": 0, "out_tok": 0},
    }
    if not cost_log_path.exists():
        return agg
    with cost_log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ph = r.get("phase")
            if ph not in agg:
                continue
            agg[ph]["calls"] += int(r.get("n_calls", 0))
            agg[ph]["in_tok"] += int(r.get("prompt_tokens", 0))
            agg[ph]["out_tok"] += int(r.get("completion_tokens", 0))
    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _run(args: argparse.Namespace) -> int:
    from datasets import load_dataset
    # cal_eval imported from the canonical ToolRet eval module — never reimplemented.
    try:
        from Toolret.eval_toolret import cal_eval
    except ImportError:
        sys.path.insert(0, str(_REPO_ROOT / "Toolret"))
        from eval_toolret import cal_eval  # type: ignore

    split = args.split
    orig_corpus = Path(args.corpus_root) / split / "des_corpus.json"
    if not orig_corpus.exists():
        log.error("corpus not found: %s", orig_corpus)
        return 2

    out_dir = Path(args.out_dir) / split
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- corpus integrity: pre-run checksum of the ORIGINAL (read-only) ---
    pre_sha = _sha256_file(orig_corpus)
    log.info("ORIGINAL corpus %s sha256(pre)=%s", orig_corpus, pre_sha[:16])

    # --- corpus isolation: operate ONLY on a
    # copy. ToolRetriever writes a corpus-adjacent embeddings `.pt`; pointing
    # it at a copy guarantees every write lands in our tmp dir, never the
    # original directory. We also copy the sibling base-embedding `.pt` (if
    # present) so ToolRetriever loads rather than re-embeds. ---
    if args.no_corpus_copy:
        work_corpus = orig_corpus
        log.warning("corpus copy DISABLED (--no_corpus_copy): ToolRetriever may "
                    "write a .pt next to the original. des_corpus.json itself is "
                    "still sha-verified read-only.")
    else:
        work_corpus = _prepare_corpus_copy(orig_corpus, args.embedder, Path(args.cache_dir))
        log.info("working corpus copy → %s", work_corpus)

    # --- tool catalog (from the working copy; byte-identical content) ---
    tool_catalog = _load_corpus(work_corpus, args.limit_tools)
    log.info("loaded %d tools from %s%s", len(tool_catalog), work_corpus,
             f" (capped to {args.limit_tools})" if args.limit_tools else "")

    # --- embedder via ToolRetriever, pointed at the COPY ---
    from Toolret.retriever import ToolRetriever
    retriever = ToolRetriever(corpus_path=str(work_corpus), model_path=args.embedder)
    adapter = RetrieverAdapter(retriever)

    # --- model client ---
    if args.stub_llm:
        client: Any = _StubLLM()
        gen_model = "stub-llm"
    else:
        from toolbench.inference.LLM.clients import make_client
        gen_model = args.model or load_pins(_REPO_ROOT)["agents"]["cheap_sota"]
        client = make_client(model=gen_model, api_key=os.environ.get("OPENAI_API_KEY", ""))

    baseline = ReInvokeBaseline(
        client, adapter,
        top_k=args.top_k,
        k_synth=args.k_synth,
        max_intents=args.max_intents,
        synth_temperature=args.synth_temperature,
        intent_temperature=args.intent_temperature,
        generator_model=gen_model,
        embedder_name=args.embedder,
        benchmark_tag=f"toolret_{split}",
        cache_dir=args.cache_dir,
        corpus_path=str(orig_corpus),  # baseline guard watches the ORIGINAL
        index_build_concurrency=args.index_concurrency,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    # --- build the index ONCE up front (generation resumes from checkpoint +
    # averaged tensor) BEFORE parallel per-query retrieval. retrieve() would
    # otherwise trigger _ensure_index on first call; doing it here avoids a
    # concurrent-build race when many retrieve() coroutines start together. ---
    await baseline._ensure_index(tool_catalog)
    log.info("index ready; per-query retrieval at query_concurrency=%d", args.query_concurrency)

    # --- collect query specs (same load path as Toolret/eval_toolret.py) ---
    subsets = DATASET_CATEGORIES[split]
    specs: list[tuple[str, str, str, dict]] = []  # (qid, subset, query, qrels_entry)
    for subset in subsets:
        try:
            ds = load_dataset("mangopy/ToolRet-Queries", subset)["queries"]
        except Exception as exc:
            log.warning("skip subset %s (load failed: %s)", subset, exc)
            continue
        for sample in ds:
            gt = json.loads(sample["labels"])
            specs.append((str(sample["id"]), subset, sample["query"],
                          {str(x["id"]): int(x["relevance"]) for x in gt}))
            if args.limit_queries and len(specs) >= args.limit_queries:
                break
        if args.limit_queries and len(specs) >= args.limit_queries:
            break

    # --- parallel per-query retrieval (LLM intent call is the bottleneck;
    # gather parallelizes the awaits. index is already built, so retrieve()
    # skips _ensure_index via its _index_built flag — no race). ---
    qrels: dict[str, dict[str, int]] = {}
    results: dict[str, dict[str, float]] = {}
    t_start = time.monotonic()
    q_sem = asyncio.Semaphore(args.query_concurrency)

    async def _one(qid: str, subset: str, query: str, qrels_entry: dict):
        async with q_sem:
            res = await baseline.retrieve(query, tool_catalog=tool_catalog)
        return qid, subset, query, qrels_entry, res

    gathered = await asyncio.gather(*[_one(*s) for s in specs], return_exceptions=True)

    per_query_log = out_dir / "per_query.jsonl"
    with per_query_log.open("w", encoding="utf-8") as pq_fh:
        for g in gathered:
            if isinstance(g, Exception):
                log.warning("per-query retrieval failure: %s", g)
                continue
            qid, subset, query, qrels_entry, res = g
            qrels[qid] = qrels_entry
            results[qid] = {tid: float(s) for tid, s in zip(res.tool_ids, res.scores)}
            pq_fh.write(json.dumps({
                "qid": qid,
                "subset": subset,
                "split": split,
                "query": query,
                "intents": res.metadata.get("intents", []),
                "retrieved_tool_ids": res.tool_ids,
                "retrieved_scores": res.scores,          # Alg-2 rank scores (order key)
                "cosine_sims": res.metadata.get("cosine_sims", []),  # raw sim, for sim-vs-Alg2 A/B
                "gold_tool_ids": list(qrels_entry.keys()),
                "gold_rels": qrels_entry,                # {id: relevance} — self-contained re-eval
                "latency_ms": res.metadata.get("latency_ms"),
            }, ensure_ascii=False) + "\n")
    n_done = len(results)
    wall_s = time.monotonic() - t_start

    # --- corpus integrity: post-run checksum of the ORIGINAL ---
    baseline.assert_corpus_unchanged()
    post_sha = _sha256_file(orig_corpus)
    integrity_ok = (pre_sha == post_sha)
    log.info("ORIGINAL corpus sha256(post)=%s  integrity_ok=%s", post_sha[:16], integrity_ok)

    # --- metrics via canonical cal_eval ---
    metrics, n_scored = cal_eval(qrels, results)

    # --- cost split ---
    cost = _summarize_cost(Path(baseline._cache.cost_log))

    summary = {
        "split": split,
        "generator_model": gen_model,
        "embedder": args.embedder,
        "k_synth": args.k_synth,
        "max_intents": args.max_intents,
        "top_k": args.top_k,
        "n_tools": len(tool_catalog),
        "n_queries_scored": n_scored,
        "n_queries_run": n_done,
        "wall_seconds": round(wall_s, 1),
        "stub_llm": args.stub_llm,
        "limit_tools": args.limit_tools,
        "limit_queries": args.limit_queries,
        "metrics": metrics,
        "cost_split": cost,
        "corpus_integrity": {
            "original_path": str(orig_corpus),
            "working_copy": str(work_corpus),
            "operated_on_copy": (not args.no_corpus_copy),
            "sha256_pre": pre_sha,
            "sha256_post": post_sha,
            "byte_identical": integrity_ok,
        },
        "cache_dir": str(baseline._cache.synth_jsonl.parent),
    }
    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    log.info("DONE split=%s  NDCG@5=%s Recall@5=%s  (n=%d)  cost=%s",
             split, metrics.get("NDCG@5"), metrics.get("Recall@5"),
             n_scored, cost)
    log.info("summary → %s", summary_path)
    log.info("per-query → %s", per_query_log)
    if not integrity_ok:
        log.error("CORPUS INTEGRITY VIOLATION — sha256 changed!")
        return 3
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=["code", "customized", "web"])
    ap.add_argument("--corpus_root", default="./data/retrieval/Toolret",
                    help="Root dir containing <split>/des_corpus.json (same layout as "
                         "Toolret/eval_toolret.py --corpus_path).")
    ap.add_argument("--embedder", default=None,
                    help="Embedder override. Default: configs/model_pins.yaml embedders.toolret.")
    ap.add_argument("--model", default=None,
                    help="Generator + intent-extractor LLM override. Default: configs/model_pins.yaml agents.cheap_sota.")
    ap.add_argument("--k_synth", type=int, default=10, help="m synth queries/tool (paper §4.3 m=10).")
    ap.add_argument("--max_intents", type=int, default=3)
    ap.add_argument("--synth_temperature", type=float, default=0.7)
    ap.add_argument("--intent_temperature", type=float, default=0.0)
    ap.add_argument("--top_k", type=int, default=20,
                    help="Retrieve top-k (>=20 so NDCG/Recall@5/10/20 are all honest).")
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--index_concurrency", type=int, default=8)
    ap.add_argument("--query_concurrency", type=int, default=16,
                    help="Concurrent per-query retrievals (parallelizes the intent-extract LLM calls).")
    runtime_root = os.environ.get("FITTEXT_RUNTIME_ROOT", "runs")
    ap.add_argument("--cache_dir", default=str(Path(runtime_root) / "reinvoke" / "cache"))
    ap.add_argument("--out_dir", default=str(Path(runtime_root) / "reinvoke" / "results"))
    ap.add_argument("--limit_tools", type=int, default=None, help="Cap corpus size (smoke only).")
    ap.add_argument("--limit_queries", type=int, default=None, help="Cap #queries (smoke only).")
    ap.add_argument("--stub_llm", action="store_true",
                    help="Use a deterministic offline stub LLM (zero API cost) for plumbing smoke.")
    ap.add_argument("--no_corpus_copy", action="store_true",
                    help="Disable the corpus working-copy (default ON). When set, ToolRetriever "
                         "may write a .pt next to the original corpus dir; des_corpus.json is still "
                         "sha-verified read-only.")
    args = ap.parse_args(argv)
    if args.embedder is None:
        args.embedder = load_pins(_REPO_ROOT)["embedders"]["toolret"]
    return args


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_run(_parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
