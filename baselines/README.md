# Baselines

Four retrieval baselines. Each maps onto a degenerate case of FitText's
evolutionary retrieval framework.

## Degeneracy mapping

| Baseline | Paper | FitText degenerate case | Cell in taxonomy |
|---|---|---|---|
| `less_is_more` | Paramanayakam et al. 2024 | Single-Pass (N=1, G=1, no revision) | Bⁿ, no evolution |
| `reinvoke` | Chen et al. EMNLP 2024 | Index-time synth expansion + multi-intent | B+index-expansion |
| `xu2024` | Xu et al. arXiv 2406.17465 | Multi-Turn + explicit critic prompt | B singleton, reactive |
| `colt` | Quchangle et al. (COLT repo) | Cuts R (retriever topology via GCN) | Orthogonal |

Memetic Retrieval (FitText full) occupies the **only** entry in the
B-with-evolution-and-memory cell of this taxonomy.

## Interface

All baselines conform to `Baseline` ABC in `base.py`:

```python
class Baseline(ABC):
    name: ClassVar[str]

    def __init__(self, model_client: ModelClient, retriever: Retriever,
                 *, top_k: int = 5, **kwargs): ...

    async def retrieve(self, query: str, *, tool_catalog: list[dict]) -> RetrievedTools: ...
```

`RetrievedTools`:
- `tool_ids: list[str]` — ranked composite `"category::tool_name::api_name"` keys
- `scores: list[float]` — parallel to tool_ids
- `metadata: dict` — baseline-specific (trace, intents, match methods, latency)

## Per-baseline design notes

### `less_is_more.py`
- One LLM call: "list the tools you need for this query"
- Exact + fuzzy name match against catalog (difflib SequenceMatcher, threshold=2)
- Retriever fallback fills remaining quota if LLM names fewer than top_k tools
- Score: 1.0 for LLM-nominated, cosine for fallback
- **Degeneracy check**: run alongside FitText `variant: single_pass` with
  `population_size: 1`; expect >70% Jaccard overlap on retrieved sets.

### `reinvoke.py`
- Index-time: generate `k_synth` (default 5) synthetic queries per tool via LLM;
  cache to `reinvoke_synthq_<embedder_rev_hash>.jsonl` in `cache_dir`.
- Cache key includes embedder revision — model upgrade auto-invalidates.
- Inference: extract `max_intents` (default 3) sub-intents from query via LLM;
  score each tool via max-pool cosine(intent_emb, synth_query_emb).
- First run without cache triggers lazy build (warned); subsequent runs are fast.

### `xu2024.py`
- Iterative loop: retrieve → critique via LLM → reformulate query → repeat
- Stops on convergence (tool set unchanged) or after `max_iterations` (default 3)
- Full trace stored in `metadata["trace"]` (query, critique, reformulated_query per iter)
- **Degeneracy check**: run alongside `variant: multi_turn`; check NDCG within ±2 pp.

### `colt.py`
- Requires `$COLT_PATH` (repo root) and `$COLT_CKPT` (checkpoint path)
- Raises `ImportError` at construction if `$COLT_PATH` is unset
- Subprocess-calls COLT's `retrieve_tools.py` (or configured entry script)
- Passes query + catalog as temp JSON files; parses JSONL or TSV output
- No LLM calls at inference (dual-encoder model only)
- See `scripts/setup_colt.sh` to clone and install COLT

## Running baselines

```bash
# Single baseline run
python scripts/run_baseline.py \
  --baseline less_is_more \
  --config configs/runs/baseline_less_is_more_toolret.yaml \
  --out "${FITTEXT_RUNTIME_ROOT:-runs}/$JOB_ID/results"

# COLT (requires setup first)
bash scripts/setup_colt.sh
export COLT_PATH=/path/to/colt_repo
python scripts/run_baseline.py \
  --baseline colt \
  --config configs/runs/baseline_colt_toolret.yaml \
  --out "${FITTEXT_RUNTIME_ROOT:-runs}/$JOB_ID/results"
```

## Manifest tagging

All baselines set `variant: baseline_<name>` in manifest entries so the
cost-table aggregator groups them separately from FitText variants.
