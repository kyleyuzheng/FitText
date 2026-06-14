# FitText: Evolving Agent Tool Ecologies via Memetic Retrieval

A research codebase for **LLM query reformulation in tool retrieval**. A DFSDT agent
(`DFS_woFilter_w2`) emits pseudo-tool descriptions while solving multi-tool tasks; pluggable
reformulation strategies turn those descriptions into retrieval queries against an embedded tool
corpus. Strategies range from a single retrieval pass to an evolutionary (memetic) search over
candidate reformulations.

Evaluated on:

- **[StableToolBench](https://github.com/THUNLP-MT/StableToolBench)** — end-to-end task pass rate
  on the solvable-query subset, scored by an LLM judge.
- **[ToolRet](https://huggingface.co/datasets/mangopy/ToolRet-Queries)** — single-shot retrieval
  NDCG@5 via `pytrec_eval`.
- A reimplementation of **Re-Invoke** (Chen et al., Findings of EMNLP 2024) as a zero-shot
  retrieval baseline, plus Less-is-More, Xu et al. 2024, and COLT baselines (`baselines/`).

## Artifact Policy

This repository is source-only. Generated benchmark outputs, result tables,
scratch analyses, run logs, caches, and model embeddings are intentionally not
bundled. Write new runtime outputs under an ignored run root such as `runs/` or
the path selected by `FITTEXT_RUNTIME_ROOT` / `FITTEXT_ANALYSIS_ROOT`.

## Repository layout

| Path | Contents |
|---|---|
| `StableToolBench/` | Benchmark harness (modified fork): inference pipeline, virtual API server, tooleval judge |
| `Toolret/` | ToolRet retrieval evaluation (`eval_toolret.py`) and corpus builder |
| `baselines/` | Re-Invoke, Less-is-More, Xu et al. 2024, COLT baselines |
| `configs/` | `model_pins.yaml` (single source of truth for model IDs), run/sweep/dispatch configs |
| `toolbench/` | Top-level dispatcher, fitness, and runner modules for config-driven runs (`run.py`) |
| `scripts/` | Run orchestration, provenance, and analysis utilities |
| `tests/` | Smoke and unit tests |

## Installation

```bash
conda env create -f environment.yaml
conda activate toolbench
cp .env.example .env   # fill in OPENAI_API_KEY / OPENAI_KEY / TOOLBENCH_KEY
```

Data setup is required before running: the public tree includes code and small
solvable-query fixtures, but not tool corpora, response caches, embeddings, or
generated result tables. See **[SETUP.md](SETUP.md)** for the data cleaning
contract, end-to-end run steps, and troubleshooting.

## Quickstart (StableToolBench)

```bash
# 1. Start the virtual tool-API server (port/model in StableToolBench/server/config.yml)
bash StableToolBench/scripts/run_server.sh

# 2. Run inference for one strategy on one split
export TOOL_ROOT_DIR="${TOOL_ROOT_DIR:?set to your ToolBench tool JSON directory}"
export CORPUS_BASE_DIR="${CORPUS_BASE_DIR:?set to your retrieval corpus base}"
bash StableToolBench/scripts/run_inference.sh \
  --strategy scattershot_s5 --dataset G1_instruction --no-planner

# 3. Convert trajectories and judge pass rate.
RUN_NAME="$(python - <<'PY'
import yaml
pin = yaml.safe_load(open("configs/model_pins.yaml"))["agents"]["main_solver"]
print(f"DFS_woFilter_w2_{pin}_simcse-roberta-large_dynamic")
PY
)"
bash StableToolBench/scripts/run_evaluation.sh \
  --model_name "$RUN_NAME" \
  --test_sets "G1_instruction"
```

ToolRet:

```bash
cd Toolret && python eval_toolret.py --dataset code --strategy scattershot \
  --scattershot_size 5 --corpus_path ./data/retrieval/Toolret \
  --embedding_model_path all-MiniLM-L6-v2 --retrieved_api_nums 5
```

Re-Invoke baseline on StableToolBench:

```bash
python baselines/build_reinvoke_stb_index.py \
  --corpora $CORPUS_BASE_DIR/G1/des_corpus.json $CORPUS_BASE_DIR/G2/des_corpus.json \
            $CORPUS_BASE_DIR/G3/des_corpus.json \
  --cache_dir "${FITTEXT_RUNTIME_ROOT:-runs}/reinvoke_index"
bash StableToolBench/scripts/run_inference.sh --strategy reinvoke --dataset all --no-planner
```

## Strategy reference

Wrapper names accepted by `run_inference.sh --strategy` (flag spellings for the direct
`qa_pipeline_open_domain.py` entry point in parentheses):

| Strategy | Flags | What it does |
|---|---|---|
| `normal` | `--retrieve_mode normal` | Static SimCSE retrieval on the raw query; no reformulation |
| `single_pass` | dynamic mode, no extra flags | One retrieval per agent-emitted pseudo-tool description |
| `dbd_t3` / `dbd_t5` | `--dbd --dbd-refine-turns N` | Description-by-description: alternate retrieve → LLM-refine for N turns |
| `scattershot_s5` | `--scattershot --size 5` | Sample 5 diverse reformulations at high temperature, retrieve per child, rank-vote winners |
| `memetic` | `--memetic --population_size 5 --generation_num 3` | Evolutionary search over reformulations: selection, crossover, mutation, LLM local search, multi-objective fitness with set-cover canonization |
| `just_query` | `--just_query` | Zero-retrieval parametric floor — no tool calls |
| `reinvoke` | `--reinvoke` (normal mode) | Re-Invoke static retriever: synthetic-query-augmented index + intent-based multi-view ranking |

## Models

All model IDs are pinned in `configs/model_pins.yaml`; override per run with
documented environment variables or YAML config, not by editing call sites.
Open-weight solvers route through a local vLLM endpoint (`VLLM_BASE_URL`).

## Attribution & licenses

- Built on **StableToolBench** (Guo et al., 2024; Apache-2.0 — see `StableToolBench/LICENSE`),
  which builds on **ToolBench / ToolLLM** (OpenBMB).
- **ToolRet** data from HuggingFace `mangopy/ToolRet-Queries` and `mangopy/ToolRet-Tools`
  (Shi et al., 2025).
- **Re-Invoke** reimplemented from Chen et al., Findings of EMNLP 2024.
- Root license: MIT (see `LICENSE`), inherited from the upstream codebase this
  work descends from.

## Citation

Paper: **[FitText: Evolving Agent Tool Ecologies via Memetic Retrieval](https://arxiv.org/abs/2605.02411)**.

```bibtex
@article{zheng2026fittext,
  title   = {FitText: Evolving Agent Tool Ecologies via Memetic Retrieval},
  author  = {Zheng, Kyle and Zhang, Han and Sun, Renliang and Ye, Chenchen and Wang, Wei},
  journal = {arXiv preprint arXiv:2605.02411},
  year    = {2026}
}
```
