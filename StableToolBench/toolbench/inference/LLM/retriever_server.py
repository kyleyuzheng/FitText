"""
Shared retriever server — loads SimCSE model once, serves concurrent requests.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m toolbench.inference.LLM.retriever_server \
        --corpus_paths G1=path/to/retrieval/G1/des_corpus.json \
                       G2=path/to/retrieval/G2/des_corpus.json \
                       G3=path/to/retrieval/G3/des_corpus.json \
        --port 8090
"""
import argparse
import os
import threading
from typing import Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from toolbench.inference.LLM.retriever import ToolRetriever

app = FastAPI()

# Global state — populated at startup
retrievers: Dict[str, ToolRetriever] = {}
lock = threading.Lock()


def _default_model_path() -> str:
    if os.environ.get("STB_RETRIEVER_MODEL"):
        return os.environ["STB_RETRIEVER_MODEL"]
    try:
        from toolbench.observability.pins import load_pins

        return load_pins()["embedders"]["stb"]
    except Exception:
        return ""


class RetrieveRequest(BaseModel):
    corpus: str  # e.g. "G1", "G2", "G3"
    query: str
    top_k: int = 5


class RetrieveResponse(BaseModel):
    tools: List[dict]
    # corpus_id -> description text, so client can do retriever.corpus[corpus_id]
    descriptions: Dict[int, str]


class CorpusInfoResponse(BaseModel):
    corpora: List[str]
    corpus_sizes: Dict[str, int]


@app.get("/health")
def health():
    return {"status": "ok", "corpora": list(retrievers.keys())}


@app.get("/corpus_info")
def corpus_info():
    return CorpusInfoResponse(
        corpora=list(retrievers.keys()),
        corpus_sizes={k: len(v.corpus) for k, v in retrievers.items()},
    )


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    if req.corpus not in retrievers:
        available = list(retrievers.keys())
        return RetrieveResponse(tools=[], descriptions={})

    retriever = retrievers[req.corpus]

    with lock:
        tools = retriever.retrieving(req.query, top_k=req.top_k)

    # Attach descriptions for each corpus_id so client can resolve them
    descriptions = {}
    for tool in tools:
        cid = tool["corpus_id"]
        if cid not in descriptions:
            descriptions[cid] = retriever.corpus[cid]

    return RetrieveResponse(tools=tools, descriptions=descriptions)


def main():
    parser = argparse.ArgumentParser(description="Shared SimCSE retriever server")
    parser.add_argument("--model_path", type=str, default=_default_model_path())
    parser.add_argument(
        "--corpus_paths",
        nargs="+",
        required=True,
        help="Corpus specs as NAME=PATH, e.g. G1=/path/to/G1/des_corpus.json",
    )
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    if not args.model_path:
        raise SystemExit("Set --model_path or STB_RETRIEVER_MODEL.")

    # Parse corpus specs
    corpus_specs = {}
    for spec in args.corpus_paths:
        name, path = spec.split("=", 1)
        corpus_specs[name] = path

    # Load model once — first corpus loads the model, subsequent ones reuse it
    print(f"Loading SimCSE model: {args.model_path}")
    first_retriever = None
    for name, path in corpus_specs.items():
        print(f"Loading corpus {name}: {path}")
        r = ToolRetriever(corpus_path=path, model_path=args.model_path, des_corpus=True)
        retrievers[name] = r
        if first_retriever is None:
            first_retriever = r
        else:
            # Share the same model and tokenizer across corpora
            r.model = first_retriever.model
            r.tokenizer = first_retriever.tokenizer

    total_corpus = sum(len(r.corpus) for r in retrievers.values())
    print(f"Server ready — {len(retrievers)} corpora, {total_corpus} total entries")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
