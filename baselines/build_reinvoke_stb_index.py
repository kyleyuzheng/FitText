#!/usr/bin/env python
"""Pre-build the Re-Invoke synthetic-query index for StableToolBench corpora.

Builds (or resumes) the Re-Invoke index — m synthetic queries per tool +
averaged SimCSE embeddings — for one or more STB ``des_corpus.json`` files, so the
live solve adapter (``ReInvokeSTBRetriever``) only LOADS the cache instead of
building it mid-run (which would race across parallel solve processes and stall
the whole sweep).

The index build is the only expensive offline step: m synth-query LLM calls per
tool. It is chunk-checkpointed and resumable (re-run to fill gaps after a crash).

Usage:
  CUDA_VISIBLE_DEVICES=4 CUDA_DEVICE_ORDER=PCI_BUS_ID RETRIEVER_DEVICE=cuda \
  python baselines/build_reinvoke_stb_index.py \
    --corpora /path/G3/des_corpus.json [/path/G1/... /path/G2/...] \
    --cache_dir "${FITTEXT_RUNTIME_ROOT:-runs}/reinvoke_stb/cache" \
    --generator_model "$(python - <<'PY'
import yaml
print(yaml.safe_load(open('configs/model_pins.yaml'))['agents']['main_solver'])
PY
)" \
    --index_concurrency 48

Cost is logged per corpus to ``<cache_dir>/stb_<corpus>/<slug>/cost_log.jsonl``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# Make both the repo root (baselines.*) and StableToolBench (toolbench.*) importable.
_REPO_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
_STB = os.path.join(_REPO_ROOT, "StableToolBench")
for _p in (_STB, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from toolbench.inference.LLM.reinvoke_retriever import ReInvokeSTBRetriever  # noqa: E402
from toolbench.observability.pins import load_pins  # noqa: E402


def _summarize(adapter: ReInvokeSTBRetriever) -> str:
    """One-line summary of the built index (synth rows + averaged tensor shape)."""
    bl = adapter._baseline
    n_tools = len(adapter._tool_catalog)
    n_synth = len(bl._synth_index._data) if bl._synth_index._data else 0
    tensor = getattr(bl._avg_index, "tensor", None)
    shape = tuple(tensor.shape) if tensor is not None else None
    return f"tools={n_tools} synth_cached={n_synth} avg_tensor={shape}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpora", nargs="+", required=True,
                    help="One or more STB des_corpus.json paths (G1/G2/G3).")
    ap.add_argument("--model_path", default=None,
                    help="SimCSE embedder override. Default: configs/model_pins.yaml embedders.stb.")
    ap.add_argument("--cache_dir", required=True,
                    help="Re-Invoke cache root (must match the solve adapter's REINVOKE_CACHE_DIR).")
    ap.add_argument("--generator_model", default=None,
                    help="Synth-query + intent LLM. Default: configs/model_pins.yaml "
                         "agents.cheap_sota (resolved in the adapter); do not hardcode.")
    ap.add_argument("--k_synth", type=int, default=10, help="Synthetic queries per tool (paper m=10).")
    ap.add_argument("--max_intents", type=int, default=3, help="Max intents per query (online).")
    ap.add_argument("--retrieved_api_nums", type=int, default=5, help="Final K (over-fetched 3x).")
    ap.add_argument("--index_concurrency", type=int, default=8, help="Synth-gen async concurrency.")
    ap.add_argument("--device", default=os.environ.get("RETRIEVER_DEVICE", "cuda"),
                    help="Embedder device for the build (cuda recommended).")
    args = ap.parse_args()
    if args.model_path is None:
        args.model_path = load_pins()["embedders"]["stb"]

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set.", file=sys.stderr)
        return 2

    rc = 0
    for corpus_path in args.corpora:
        corpus_path = os.path.abspath(corpus_path)
        if not os.path.exists(corpus_path):
            print(f"[SKIP] missing corpus: {corpus_path}", file=sys.stderr)
            rc = 1
            continue
        print(f"\n=== building Re-Invoke index for {corpus_path} (device={args.device}) ===",
              flush=True)
        t0 = time.monotonic()
        adapter = ReInvokeSTBRetriever(
            corpus_path=corpus_path,
            model_path=args.model_path,
            retrieved_api_nums=args.retrieved_api_nums,
            reinvoke_cfg={
                "k_synth": args.k_synth,
                "max_intents": args.max_intents,
                "generator_model": args.generator_model,
                "cache_dir": args.cache_dir,
                "index_build_concurrency": args.index_concurrency,
            },
            device=args.device,
        )
        dt = time.monotonic() - t0
        print(f"[DONE] {os.path.basename(os.path.dirname(corpus_path))} in {dt/60:.1f} min :: "
              f"{_summarize(adapter)}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
