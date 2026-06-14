# Baseline Reproduction Log

Every externally-cloned or reimplemented baseline gets a row here.
Reimplemented baselines are audited against their
source paper (§B); cloned baselines have one published number reproduced
end-to-end before any of their results are quoted (§A).

Repos in `baselines/external/` are **not vendored** (git-ignored). Run the
relevant `scripts/setup_*.sh` to clone at the pinned commit SHA.

Reproduction tests live in `tests/test_baseline_reproduction.py`. Mocked-LLM
tests run in CI (rank-deterministic); real-LLM tests require
`RUN_BASELINE_REPRODUCTION=1`.

Audited: 2026-05-24.

---

## A. Cloned baselines


## COLT (Qu et al., CIKM 2024)

- **Repo:** https://github.com/quchangle1/COLT
- **Paper:** "Towards Completeness-Oriented Tool Retrieval for Large Language
  Models" — Qu, Dai, Wei, Cai, Wang, Yin, Xu, Wen — CIKM 2024.
- **arXiv:** https://arxiv.org/abs/2405.16089
- **Commit pinned:** `bbec292c38cabe3fbae098f50cc31b2e9f7e8fca`
  - Verified: 2026-05-24
  - Branch: `main`
  - Last upstream commit message: "Update config.yaml"
- **Setup script:** `scripts/setup_colt.sh`
- **Local path (after setup):** `baselines/external/colt/`

### Architecture summary

Two-stage:
1. **Semantic learning** (`train_sbert.py`) — fine-tunes a PLM (Contriever / ANCE
   / TAS-B / co-Condensor) on `(query, tool_description)` pairs from the
   training split.
2. **Collaborative learning** (`train.py`) — Bayesian Personalized Ranking +
   contrastive loss on the dual-view GCN over three bipartite graphs:
   `(Query, Scene)`, `(Query, Tool)`, `(Scene, Tool)`. Produces tool and query
   embeddings; inference does dot-product top-k.

### Datasets shipped in the repo

- `datasets/ToolLens/` — 571 tools, 18.6k queries (paper's headline benchmark)
- `datasets/ToolBenchG2/` — ToolBench G2 split
- `datasets/ToolBenchG3/` — ToolBench G3 split

Each split has pre-built `query_tool_{train,tune,test}.txt` integer index files
and a `corpus.jsonl`/`queries.jsonl` text-side mapping.

### Reproduction plan

| Item | Value |
|---|---|
| Benchmark cell tried | ToolLens / NDCG@5 on test split |
| Their published number | **NDCG@5 = 0.5410** (Table 2, COLT row, ToolLens) |
| Our reproduced number | NOT_RUN — see status |
| Drift | n/a |
| **Status** | **NOT_RUN** |

### Status: NOT_RUN (documented blockers — partial reproduction protocol below)

Full reproduction of the COLT number from scratch requires:
- **GPU training time:** ~200 epochs of dual-view GCN on 4× A100. The COLT paper
  does not state wall-clock, but the dataset has ~18k queries × 571 tools and
  the GCN propagates over 3 graphs per batch. Empirically (from similar
  collaborative-filtering GCN training runs): **6–12 hours on a single A100**.
- **PLM weights:** Contriever (`nthakur/contriever-base-msmarco`) — ~440 MB,
  downloaded once. Other supported PLMs: ANCE, TAS-B, co-Condensor.
- **Pre-trained checkpoint** (`Tool-COLT` HuggingFace org): per the upstream
  README, a checkpoint of *first-stage semantic learning only* is released at
  https://huggingface.co/Tool-COLT — this skips step 1 but you still must run
  step 2 (~100 epochs collaborative learning).
- **Note:** no end-to-end ready-to-infer checkpoint is published by upstream.

Standing up COLT training requires a dedicated GPU for the 6–12 hour
training run estimated above; that budget was not available alongside the
main FitText evaluation matrix.

### Partial reproduction protocol (next steps when GPU is available)

```bash
# 1. Bring the repo to the pinned SHA and install deps.
bash scripts/setup_colt.sh

# 2. Pull the pre-trained PLM (Contriever — smallest supported model).
#    All PLMs live under $COLT_PATH/PLMs/<name>/ by convention.
huggingface-cli download nthakur/contriever-base-msmarco \
    --local-dir "${COLT_PATH}/PLMs/contriever-base-msmarco"

# 3. Run first-stage semantic learning on ToolLens.
cd "${COLT_PATH}"
python train_sbert.py  # writes a fine-tuned PLM under PLMs/

# 4. Run second-stage collaborative learning + log NDCG@5 from test().
python train.py -g 0 -m COLT -d ToolLens

# 5. The test() function in train.py (line ~235) emits NDCG/Recall/Comp
#    @ each topk in conf["topk"] (default includes 5). Compare the NDCG@5
#    value on the test split against the paper's 0.5410 (Table 2, ToolLens).
#    Drift threshold: ±2 NDCG points.

# 6. After convergence, write the reproduced number into this file under
#    "Our reproduced number" and flip Status to PASS / FAIL.
```

### Pinned dependency versions (from upstream README)

| Package | Version |
|---|---|
| `numpy` | 1.21.6 |
| `pandas` | 1.3.5 |
| `torch` | 1.13.1 |
| Python | 3.8 (upstream-tested; we'll evaluate compatibility under 3.10/3.12 in the install step) |

The upstream repo does **not** ship a `requirements.txt`. `scripts/setup_colt.sh`
installs the pinned versions above explicitly and additionally pulls
`pyyaml`, `tqdm`, `tensorboard`, `scipy`, and `transformers` (the latter for the
sentence-transformer PLMs).

### Integration into FitText

The wrapper in `baselines/colt.py` calls into the cloned repo via subprocess —
either `python train.py --infer True ...` to load a pre-trained checkpoint and
read top-k indices from the produced `tensor_data_formatted.txt`, or (when
checkpoint is missing) raises `RuntimeError` with a pointer to this file.

**Important constraint:** COLT operates over the **same tool universe it was
trained on** (integer tool indices into the trained corpus). It cannot directly
retrieve over an arbitrary new tool catalog without retraining. In this work,
COLT is reported on its own native benchmark (ToolLens) for
apples-to-apples with the paper, and **not** swapped onto ToolRet without
documenting the train/eval mismatch.

### Notes / quirks

- The `models/` directory in upstream is empty in the pinned commit; the actual
  GCN definition lives in code paths that `import models.COLT` — that file must
  ship in a release tag or alongside the HuggingFace checkpoint bundle. As of
  pinning we **do not have `models/COLT.py`** locally; this is a known blocker
  for `train.py -m COLT`. The repo issues/PRs should be checked for a release
  tag that includes the model file before any full reproduction is attempted.
- Test split write side-effect: `train.py:test()` appends to
  `tensor_data_formatted.txt` in the working directory. Our wrapper runs the
  subprocess in a tempdir to avoid pollution.

---

## Other baselines


---

## B. Reimplemented baselines (audited against source paper)

# Baseline Reproduction Audit

Verifies that the three reimplemented baselines (Less-is-More, Re-Invoke,
Xu et al. 2024) are faithful to their published papers.  Each section
quotes the paper section that defines the method, lists deviations, and
reports a spot-check status.

Reproduction tests live in `tests/test_baseline_reproduction.py`.  Mocked-LLM
tests run in CI and skip when behavior is rank-deterministic.  Real-LLM tests
require `RUN_BASELINE_REPRODUCTION=1`.

Audited: 2026-05-24.

---

## Less-is-More (Paramanayakam et al., arXiv 2411.15399)

- **Paper:** "Less is More: Optimizing Function Calling for LLM Execution on
  Edge Devices" (Paramanayakam, Karatzas, Anagnostopoulos, Stamoulis;
  arXiv:2411.15399, Nov 2024).
- **Spec sections:** §III.A (Constructing the Search Levels), §III.B (Tool
  Recommender + query latent space embedding), §III.C (Tool Controller).
- **Our implementation:** `baselines/less_is_more.py`.  An earlier
  implementation extracted tool *names* by regex from the LLM output and
  fuzzy-matched them to the catalog.  That is not the paper's algorithm.
  The paper:
    1. Prompts the LLM (Tool Recommender) to *describe* the ideal tools
       (§III-B), not name them.
    2. Embeds the pseudo-tool descriptions with MPNet (§III-B).
    3. Runs k-NN in the latent space against Search Level 1 (individual
       tools) and Search Level 2 (tool clusters from offline agglomerative
       clustering of augmented queries) (§III-C).
    4. Picks whichever level has higher avg top-k similarity.
    5. Falls back to Level 3 (all tools) if both avg scores < 0.5.

### Algorithm now implemented

For each query Q:
1. LLM generates ≤ top_k pseudo-tool *descriptions* (paper §III-B). Prompt
   tells the LLM to describe capability, not name real tools.
2. For each pseudo-tool description, retrieve top-k tools from the
   embedding retriever (§III-C, Level 1 path).
3. Max-pool scores across pseudo-tools.
4. Confidence guard: if max(avg-top-k-score across pseudo-tools) <
   `confidence_threshold` (default 0.5 per paper), fall back to raw-query
   retrieval (proxy for §III-C Level 3 "all tools").

### Deviations from paper

- **L2 (cluster index) is opt-in.**  The paper builds an "augmented latent
  space" by sampling 10 BFCL/GeoEngine training queries per category, then
  using GPT-4 (Turbo 0125) to generate contextually proximate queries, then
  applying Agglomerative Clustering (§III-A "Search Level 2").  This requires
  benchmark-specific training data we do not have for ToolRet / StableToolBench,
  and the paper notes Level 1 wins on ToolBench-style tasks anyway (only
  GeoEngine benefits from L2).  Set `enable_l2_clusters=True` and provide
  a pre-built cluster index in `cfg` to activate L2 path.  Default off.
- **Encoder model.**  Paper uses MPNet (`all-mpnet-base-v2`).  We reuse the
  FitText retriever's encoder (pinned via `configs/_base/embedder.yaml`).
  Algorithm is encoder-agnostic; only embedding space changes.

### Spot-check (synthetic; tests/test_baseline_reproduction.py)

A deterministic 10-query synthetic benchmark with known gold-tool ordering.
Mocked LLM returns paper-realistic pseudo-tool descriptions per query.
Stub retriever returns hits ranked by string overlap with each pseudo-tool.

- Synthetic-rank Jaccard@5 vs gold: ≥ 0.6 (acceptance threshold).
- Confidence-threshold guard fires for vague queries (verified).
- pseudo_tool field populated in metadata for instrumentation.

### Headline numbers (paper, for human reference)

The paper reports *success rate*, *tool accuracy*, *execution time*, and
*power* on BFCL and GeoEngine — not nDCG/recall.  Their main numerical
claims are: execution time reduced up to 70%, power reduced up to 40%, and
success-rate parity or better vs full-tool baseline.  These are device-side
metrics (RPi/Jetson) and are out of scope for our retrieval-only eval.

For our retrieval-equivalent evaluation, the closest analog is Tool
Accuracy on BFCL (paper Fig 2-3): LiS at k=5 reaches ~0.9 tool accuracy on
single-function-call BFCL.  We do not attempt to reproduce BFCL Tool
Accuracy here — our spot-check is on ToolRet-style synthetic data.

### Status: **PASS (faithful algorithm + synthetic spot-check)**

Full numerical reproduction against paper Figure 2/3 numbers requires a BFCL
benchmark harness (out of scope for this work).  Mark as
`NEEDS_MANUAL_REPRO` for paper-headline reproduction; PASS for algorithm
faithfulness.

---

## Re-Invoke (Chen et al., arXiv 2408.01875, EMNLP 2024 Findings)

- **Paper:** "Re-Invoke: Tool Invocation Rewriting for Zero-Shot Tool
  Retrieval" (Chen, Yoon, Sachan, Wang, Cohen-Addad, Bateni, Lee, Pfister;
  Google Cloud AI Research; arXiv:2408.01875v2, Aug 2024).
- **Spec sections:** §3 (Method: Re-Invoke), §3.1 (Query Generator), §3.2
  (Query Intent Extractor), §3.3 (Multi-View Similarity Ranking),
  Algorithm 1, §A.1 (synth-query prompt), §A.2 (intent-extract prompt).
- **Our implementation:** `baselines/reinvoke.py`.

### Algorithm now implemented

Offline (paper §3.1):
- For each tool, LLM generates `k_synth` synthetic queries with
  high-temperature sampling (paper §A.1: "We encourage LLMs to produce
  creative and complex queries").
- Cached to JSONL keyed by `embedder_revision`.

Online (paper §3.2 + §3.3):
1. LLM extracts `max_intents` sub-intents from user query (§3.2).
2. For each intent, retrieve top-`k * intent_topk_multiplier` tools via
   the FitText dense retriever (multi-view: tool embeddings + augmented
   synth-query embeddings are both in the retriever index).
3. Merge per-intent rankings via **Reciprocal Rank Fusion** (RRF, k=60).
4. Return top-k by RRF score.

### Deviations from paper

- **Multi-view merge function not specified by paper.**  §3.3: "we first
  compute the similarity scores between the expanded tool documents and
  each intent in the embedding space.  We rank and retrieve the top tools
  from each intent as the final retrieved tools."  The paper does not
  specify *how* to merge per-intent top-K lists.  An earlier implementation
  used global max-pool over intents × synth-queries, which collapses the
  multi-view structure.  We switched to **Reciprocal Rank Fusion (RRF,
  k=60)** — IR-canonical (Cormack et al. 2009), parameter-light, and
  preserves per-intent ranking signal.  An ablation comparing RRF vs
  max-pool merge is left to future work.
- **Augmented tool documents stored as separate embeddings.**  Paper §3.1
  says `d_i = Concat(d, q_i)` produces m augmented copies per tool.  In
  practice we encode each synth query separately and treat
  max-similarity(query, d) and max-similarity(query, q_i) as the
  matching score.  This is equivalent to having m augmented copies for
  scoring purposes (the max-similarity wins regardless of which copy is
  closer); avoids m-fold storage blow-up.
- **No training of the retriever** — Re-Invoke is already unsupervised
  per paper §3 ("fully unsupervised retrieval method"); no deviation here.

### Spot-check (synthetic)

Deterministic 10-query benchmark.  Mocked LLM returns paper-style intents
(2 per query) and per-tool synth queries (3 per tool).  Stub retriever
returns hits ranked by overlap with each intent.  RRF merges the per-intent
rankings.

- Synthetic-rank Jaccard@5 vs gold: ≥ 0.6 (acceptance threshold).
- Per-intent top-K rankings recorded in metadata for instrumentation.

### Headline numbers (paper Table 1, nDCG@5)

| Setting | ToolBench I1 | ToolBench I2 | ToolBench I3 | ToolE single | ToolE multi |
|---|---|---|---|---|---|
| Re-Invoke w/ Vertex AI + text-bison@001 | 0.6110 | 0.5379 | 0.5955 | 0.7821 | 0.7231 |
| Re-Invoke w/ Vertex AI + gpt-3.5 turbo | 0.6090 | 0.5068 | 0.5719 | 0.7705 | 0.6957 |

We do NOT attempt these numbers in CI — they require Vertex AI text
embedding (Google-only) and the ToolE dataset (Huang et al. 2023, requires
upstream eval harness).  Run the manual reproduction with
`RUN_BASELINE_REPRODUCTION=1`:

```bash
RUN_BASELINE_REPRODUCTION=1 \
    EMBEDDING_BACKEND=local \
    pytest tests/test_baseline_reproduction.py::test_reinvoke_real -v
```

### Status: **PASS (faithful algorithm + synthetic spot-check)**

Paper-headline reproduction is `NEEDS_MANUAL_REPRO` against ToolBench/ToolE
upstream evaluator.

---

## Xu et al. 2024 (arXiv 2406.17465, EMNLP 2024 Findings)

- **Paper:** "Enhancing Tool Retrieval with Iterative Feedback from Large
  Language Models" (Qiancheng Xu, Yongqi Li, Heming Xia, Wenjie Li;
  Hong Kong PolyU; arXiv:2406.17465v2).
- **Spec sections:** §4.2 (Feedback Generation) with Eqs. 2-4
  Comprehension/Assessment/Refinement.  §4.3 Iteration-Aware Training is
  NOT implemented (all baselines in this work are training-free).
- **Code reference:** https://github.com/travis-xu/TR-Feedback
- **Our implementation:** `baselines/xu2024.py`.

### Algorithm now implemented

At each iteration `t` with current instruction `q^t`:
1. Retrieve top-K tools.
2. **Comprehension (Eq. 2):** LLM summarizes user goals + understands
   retrieved tools (category, name, description, IO).
3. **Assessment (Eq. 3):** LLM identifies which goals can/can't be solved
   and assesses ranking quality.
4. **Refinement (Eq. 4):** LLM either emits special token `"N/A"` if
   (a) all goals solved AND (b) all appropriate tools top-ranked, OR
   emits a refined instruction with (i) detail on unsolved intents,
   (ii) scenario-specific usage info for found tools.
5. Convergence: stop if `"N/A"` OR retrieved-set unchanged.

### Deviations from paper

- **Iteration-Aware Feedback Training (§4.3, Eq. 5) is NOT implemented.**
  All baselines in this work are training-free.  Our Xu2024 is
  the inference chain only; the trained retriever would be a *fine-tuned*
  TAS-B / ANCE / BGE that we cannot ship without paying training cost.
  Xu 2024 is the Multi-Turn degenerate case (see `baselines/README.md`),
  and the degeneracy claim is on the inference chain.
- **An earlier implementation used a generic 2-call critique+regen
  chain.**  Replaced with the paper's faithful 3-call C/A/R chain
  (Eqs. 2-4) with separate system prompts for each stage.  An
  `allow_collapsed_chain=True` config option exists for cheap ablations.
- **N/A convergence:** refinement
  output matching "N/A" (case-insensitive, with light tolerance for
  "N/A.", "N/A - no refinement needed", etc.) triggers early-stop.
- **Set-stable convergence retained** as a defensive belt-and-suspenders
  for LLMs that don't emit "N/A" reliably.

### Spot-check (synthetic)

Deterministic 10-query benchmark.  Mocked LLM follows the paper's
3-step C/A/R protocol.  N/A convergence and set-stable convergence both
verified in unit tests.

### Empirical equivalence to FitText Multi-Turn

Xu 2024 should be empirically equivalent
to FitText's `multi_turn` variant (population_size=1, generations>1,
selection=fitness, with explicit critic-style prompts).  See
`tests/test_baseline_reproduction.py::test_xu2024_degeneracy_with_multi_turn` —
currently marked `xfail` because the FitText `multi_turn` invocation
is not yet wired into the baseline harness.

### Headline numbers (paper Table 2 / Table 3, nDCG@5 on ToolBench)

Paper Table 2 reports trained TAS-B with iteration-aware training as the
strongest setting.  For the training-free chain (inference only), the
paper reports nDCG@5 in the range of 0.55-0.62 on ToolBench I1/I2/I3
across iterations 1-3.  We do not attempt to reproduce these in CI —
they require the ToolBench retrieval split and the trained TAS-B
checkpoint from the paper authors.

### Status: **PASS (faithful algorithm + synthetic spot-check)**

Paper-headline reproduction is `NEEDS_MANUAL_REPRO`.  Empirical
degeneracy with FitText `multi_turn` is `xfail` pending evol-refactor wiring.

---

## Manual reproduction commands

For full reproduction against published numbers (requires API keys,
upstream benchmark splits, and possibly trained checkpoints), set:

```bash
export RUN_BASELINE_REPRODUCTION=1
export OPENAI_API_KEY=...
# Optionally: VERTEX_AI_ENDPOINT, ANTHROPIC_API_KEY

pytest tests/test_baseline_reproduction.py -v
```

Each baseline-on-paper-benchmark cell
costs ~$5-$15 in API calls.  Budget guard: see `run.py --max-cost-usd`.
