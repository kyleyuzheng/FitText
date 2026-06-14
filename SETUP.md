# FitText Setup And Release Runbook

This file is the operational runbook for bringing up FitText from a clean
clone. The repository is source-only: generated benchmark outputs, response
caches, embedding indexes, raw model trajectories, cost tables, scratch
analysis, and local secrets must stay outside git.

## 1. Environment

```bash
conda env create -f environment.yaml
conda activate toolbench
cp .env.example .env
```

Fill in `.env`, then source it before running shell entry points:

```bash
set -a
source .env
set +a
```

Minimum variables:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Solver, simulator fallback, corpus builders, ToolRet generation |
| `OPENAI_KEY` | Optional StableToolBench judge override; if unset, the judge uses `OPENAI_API_KEY` |
| `TOOLBENCH_KEY` | Key sent to the local virtual tool server; any non-empty local value works |
| `WORKSPACE_ROOT` | Repository root for config interpolation |
| `FITTEXT_RUNTIME_ROOT` | Ignored run root for outputs, caches, and manifests |
| `FITTEXT_ANALYSIS_ROOT` | Ignored run root for derived analysis artifacts |

Use `configs/model_pins.yaml` as the model source of truth. Do not copy pinned
model names into scripts or runbooks; override via env or YAML when needed.

## 2. Data Cleaning Contract

The public tree intentionally includes only small source/test fixtures:

- `StableToolBench/solvable_queries/test_instruction/*.json`
- `StableToolBench/solvable_queries/test_query_ids/*.json`
- `StableToolBench/solvable_queries_example/`

These files are the cleaned StableToolBench solvable-query subset used by the
release harness. The cleaning contract is:

1. Start from the upstream StableToolBench solvable-query release.
2. Keep only the six public splits: `G1_category`, `G1_instruction`,
   `G1_tool`, `G2_category`, `G2_instruction`, and `G3_instruction`.
3. Keep query rows whose `query_id` appears in the corresponding
   `test_query_ids/<split>.json`.
4. Preserve the benchmark fields needed at runtime: query text, `query_id`,
   and the original candidate API list.
5. Do not include model outputs, judge labels, pass-rate tables, hard-question
   manifests, response caches, generated retrieval corpora, or paper tables.

If you regenerate the subset, validate it before replacing the checked-in
fixtures:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("StableToolBench/solvable_queries")
for qfile in sorted((root / "test_instruction").glob("*.json")):
    split = qfile.stem
    rows = json.loads(qfile.read_text())
    ids = json.loads((root / "test_query_ids" / f"{split}.json").read_text())
    row_ids = {str(r.get("query_id") or r.get("id")) for r in rows}
    official = {str(x) for x in ids}
    assert row_ids == official, (split, len(row_ids), len(official))
print("solvable-query fixtures match test_query_ids")
PY
```

## 3. External Data Layout

A practical local layout keeps large mutable assets outside the source checkout:

```text
${FITTEXT_RELEASE_ROOT}/
  downloads/
  toolenv/tools/                         # ToolBench RapidAPI JSON corpus
  retrieval/StableToolBench/G1/des_corpus.json
  retrieval/StableToolBench/G2/des_corpus.json
  retrieval/StableToolBench/G3/des_corpus.json
  retrieval/Toolret/code/des_corpus.json
  retrieval/Toolret/web/des_corpus.json
  retrieval/Toolret/customized/des_corpus.json
  runs/
    stb_server/tool_response_cache/      # StableToolBench virtual-server cache
```

Set:

```bash
export WORKSPACE_ROOT="$PWD"
export FITTEXT_RELEASE_ROOT="${FITTEXT_RELEASE_ROOT:-${AA_MUTABLE_ROOT:?set AA_MUTABLE_ROOT}/fittext-release}"
export FITTEXT_RUNTIME_ROOT="${FITTEXT_RUNTIME_ROOT:-$FITTEXT_RELEASE_ROOT/runs}"
export FITTEXT_ANALYSIS_ROOT="${FITTEXT_ANALYSIS_ROOT:-$FITTEXT_RUNTIME_ROOT/analysis}"
export TOOL_ROOT_DIR="$FITTEXT_RELEASE_ROOT/data/toolenv/toolenv2404_filtered"
export CORPUS_BASE_DIR="$FITTEXT_RELEASE_ROOT/data/retrieval/StableToolBench"
```

The repo ignores `runs/`, `results/`, response caches, server logs, generated
analysis files, and externally cloned baseline repos.

## 4. ToolBench And StableToolBench Data

Download the ToolBench/StableToolBench assets from their official upstream
releases, then keep the downloaded data under ignored local roots:

1. ToolBench data release:
   `https://github.com/OpenBMB/ToolBench`. Place the RapidAPI `tools/` tree at
   `$TOOL_ROOT_DIR`.
2. StableToolBench ToolEnv2404 release:
   `https://huggingface.co/datasets/stabletoolbench/ToolEnv2404`. Use this
   as the virtual tool environment source.
3. StableToolBench cache release:
   `https://huggingface.co/datasets/stabletoolbench/Cache`. Unpack it to
   `$FITTEXT_RUNTIME_ROOT/stb_server/tool_response_cache/`.
4. StableToolBench retrieval corpora: place or build
   `$CORPUS_BASE_DIR/{G1,G2,G3}/des_corpus.json` from the upstream
   StableToolBench assets documented at
   `https://github.com/THUNLP-MT/StableToolBench`.

Download and extract the StableToolBench assets:

```bash
mkdir -p "$FITTEXT_RELEASE_ROOT/downloads" \
         "$FITTEXT_RELEASE_ROOT/data/toolenv" \
         "$FITTEXT_RUNTIME_ROOT/stb_server"

python - <<'PY'
import os
from huggingface_hub import hf_hub_download

download_dir = os.path.join(os.environ["FITTEXT_RELEASE_ROOT"], "downloads")
for repo, filename in [
    ("stabletoolbench/ToolEnv2404", "toolenv2404_filtered.tar.gz"),
    ("stabletoolbench/Cache", "server_cache.zip"),
]:
    hf_hub_download(
        repo_id=repo,
        repo_type="dataset",
        filename=filename,
        local_dir=download_dir,
    )
PY

tar -xzf "$FITTEXT_RELEASE_ROOT/downloads/toolenv2404_filtered.tar.gz" \
  -C "$FITTEXT_RELEASE_ROOT/data/toolenv"

python -m zipfile -e \
  "$FITTEXT_RELEASE_ROOT/downloads/server_cache.zip" \
  "$FITTEXT_RUNTIME_ROOT/stb_server"
```

The retrieval corpus builder is expensive because it uses an LLM to summarize
tool descriptions. Run it only when you intend to regenerate the corpus:

```bash
PYTHONPATH=StableToolBench python StableToolBench/toolbench/retrieval/build_des_corpus.py \
  --corpus_file <upstream_group_corpus.json> \
  --tool_root_dir "$TOOL_ROOT_DIR" \
  --gpt_model "$(python - <<'PY'
import yaml
print(yaml.safe_load(open("configs/model_pins.yaml"))["agents"]["main_solver"])
PY
)" \
  --output_path "$CORPUS_BASE_DIR/G1/des_corpus.json" \
  --refer_corpus_file <upstream_reference_corpus.json>
```

For a live release smoke run when the upstream ToolBench retrieval TSV is not
staged locally, build deterministic description corpora directly from ToolEnv:

```bash
PYTHONPATH=StableToolBench \
python StableToolBench/toolbench/retrieval/build_des_corpus_from_toolenv.py \
  --tool-root-dir "$TOOL_ROOT_DIR" \
  --output-dir "$CORPUS_BASE_DIR"
```

This fallback uses ToolEnv API names, descriptions, and parameter names. It is
appropriate for proving the FitText code path end to end; use the official
ToolBench retrieval corpus plus `build_des_corpus.py` for exact paper
reproduction.

`run_inference.sh` reads queries from
`${CORPUS_BASE_DIR}/../test_instruction/<split>.json`. Either set
`CORPUS_BASE_DIR=StableToolBench/solvable_queries/retrieval` for fixture-only
experiments, or copy the checked-in `test_instruction/` directory next to your
external retrieval directory:

```bash
mkdir -p "$FITTEXT_RELEASE_ROOT/data/retrieval/test_instruction"
cp "$WORKSPACE_ROOT"/StableToolBench/solvable_queries/test_instruction/*.json \
  "$FITTEXT_RELEASE_ROOT/data/retrieval/test_instruction/"
```

Use a copy rather than a symlink: live inference writes retrieval sanity logs
beside runtime outputs, and fixture staging should never allow writes through
to the source tree.

## 5. ToolRet Data

ToolRet queries are loaded from HuggingFace dataset
`mangopy/ToolRet-Queries`. Tool descriptions are generated from
`mangopy/ToolRet-Tools` and cached locally as `des_corpus.json`.

```bash
for subset in code web customized; do
  python Toolret/data_preprocess/build_des_corpus.py \
    --config "$subset" \
    --output_path "data/retrieval/Toolret/$subset/des_corpus.json"
done
```

The builder resumes from the output file by default. Delete the target file or
pass `--start_from_scratch` only when you intentionally want to regenerate it.

## 6. StableToolBench End-To-End Pass Rate

Start the virtual API server in one terminal:

```bash
bash StableToolBench/scripts/run_server.sh
```

Run one split:

```bash
export OUTPUT_ROOT="$FITTEXT_RUNTIME_ROOT/stb/raw_output"
bash StableToolBench/scripts/run_inference.sh \
  --strategy scattershot_s5 \
  --dataset G1_instruction \
  --no-planner
```

Evaluate the raw trajectories:

```bash
RUN_NAME="$(python - <<'PY'
import yaml
pin = yaml.safe_load(open("configs/model_pins.yaml"))["agents"]["main_solver"]
print(f"DFS_woFilter_w2_{pin}_simcse-roberta-large_dynamic")
PY
)"
export RAW_ANSWER_PATH="$FITTEXT_RUNTIME_ROOT/stb/raw_output"
export CONVERTED_ANSWER_PATH="$FITTEXT_RUNTIME_ROOT/stb/model_predictions_converted"
export PASS_RATE_SAVE_PATH="$FITTEXT_RUNTIME_ROOT/stb/pass_rate_results"
export TEST_IDS_DIR="$WORKSPACE_ROOT/StableToolBench/solvable_queries/test_query_ids"

bash StableToolBench/scripts/run_evaluation.sh \
  --model_name "$RUN_NAME" \
  --test_sets "G1_instruction"
```

Canonical pass-rate runs use `--no-planner`. Use the strategy names in
`README.md` for `single_pass`, `dbd_t3`, `scattershot_s5`, `memetic`,
`just_query`, and `reinvoke`.

## 7. Config-Driven Runner

The config-driven runner is the preferred path for ToolRet and retrieval-only
StableToolBench experiments:

```bash
python run.py --config configs/runs/cheap_sota_memetic_toolret.yaml --dry-run

JOB_ID=fittext_smoke \
python run.py \
  --config configs/runs/cheap_sota_memetic_toolret.yaml \
  --out "$FITTEXT_RUNTIME_ROOT/$JOB_ID/results" \
  --cache-dir "$FITTEXT_RUNTIME_ROOT/$JOB_ID/cache" \
  --max-cost-usd 5
```

For matrix launches:

```bash
python scripts/dispatch_benchmarks.py \
  --spec configs/dispatch/cheap_sota_small.yaml \
  --out "$FITTEXT_RUNTIME_ROOT/dispatch/$JOB_ID" \
  --dry-run
```

## 8. Baselines

Re-Invoke:

```bash
python baselines/build_reinvoke_stb_index.py \
  --corpora "$CORPUS_BASE_DIR/G1/des_corpus.json" \
            "$CORPUS_BASE_DIR/G2/des_corpus.json" \
            "$CORPUS_BASE_DIR/G3/des_corpus.json" \
  --cache_dir "$FITTEXT_RUNTIME_ROOT/reinvoke_index"
```

COLT is optional and external:

```bash
bash scripts/setup_colt.sh
export COLT_PATH="$WORKSPACE_ROOT/baselines/external/COLT"
export COLT_CKPT="$COLT_PATH/checkpoint"
```

`baselines/external/` is ignored and must not be committed.

## 9. Verification Before Release

Run these checks before asking to push the clean `FitText` repository:

```bash
python scripts/check_release_hygiene.py \
  --env-file .env \
  --require-live-prereqs \
  --require-e2e-evidence
python scripts/check_pin_consistency.py --repo-root .
PYTHONDONTWRITEBYTECODE=1 python -m pytest --collect-only -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/smoke_dispatcher.py \
  tests/smoke_runner.py \
  tests/smoke_pin_consistency.py \
  tests/smoke_release_hygiene.py \
  tests/smoke_split_temperatures.py \
  tests/smoke_fitness_methods.py \
  tests/test_e2e_pipeline.py
```

Expected state:

- No untracked files in the repo directory.
- `scripts/check_release_hygiene.py` allows only `.env` plus the env-selected
  live data/runtime roots as ignored local files; unrelated ignored scratch
  files still fail the audit.
- No committed generated outputs, result tables, caches, embeddings, logs, or
  local key files.
- No duplicate hardcoded active model names outside `configs/model_pins.yaml`.
- `scripts/check_release_hygiene.py --require-live-prereqs` passes after the
  live data/credential setup is in place.
- `scripts/check_release_hygiene.py --require-e2e-evidence` passes only after
  a live StableToolBench run has produced raw outputs, converted answer JSON,
  pass-rate result files, and `.done` evaluation markers under
  `$FITTEXT_RUNTIME_ROOT/stb/`.
- `release/public-clean` and `main` have the same tree before creating the new
  GitHub repository.

## 10. Publishing FitText

Do not run this section until the verification checklist above and the real
end-to-end pass-rate run have both passed.

Create a separate release checkout so the old remotes are not reused:

```bash
git status --short
git switch -c release/public-clean
git diff --quiet main...release/public-clean

git clone --no-local . ../FitText
cd ../FitText
git remote remove origin || true
git remote add origin git@github.com:<OWNER>/FitText.git

python scripts/check_release_hygiene.py --publish-target-name FitText
python scripts/check_release_hygiene.py \
  --env-file .env \
  --require-live-prereqs \
  --require-e2e-evidence \
  --publish-target-name FitText
git remote -v
```

Only after those commands prove the checkout is named `FitText` and the remote
targets the new GitHub repository should the release branch be pushed:

```bash
git push -u origin main
```

## 11. Common Bugs

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: toolbench` when running StableToolBench helper scripts | Run from repo root with `PYTHONPATH=StableToolBench` or use the provided shell wrappers. |
| `run_inference.sh` stops with `set TOOL_ROOT_DIR` or `set CORPUS_BASE_DIR` | Source `.env` and set both paths before launching a real run. Dry-run mode uses sentinel values and does not launch services. |
| Virtual server starts but every tool call is slow or costly | The response cache is missing or pointed at the wrong directory; unpack it to `$FITTEXT_RUNTIME_ROOT/stb_server/tool_response_cache/`. |
| Evaluator says `OPENAI_KEY` is missing | Set `OPENAI_KEY`, or export it from `OPENAI_API_KEY` before running `run_evaluation.sh`. |
| Evaluation silently skips a cell | Existing output has a `.done` marker; set `FORCE_REEVAL=1` or use a fresh `PASS_RATE_SAVE_PATH`. |
| HuggingFace downloads fail | Pre-download datasets/models into the HF cache, or run with network access enabled. |
| Retriever OOMs when many inference workers start | Start `StableToolBench/scripts/run_retriever_server.sh` once and pass `--retriever-server`. |
| Open-weight model gets sent to the OpenAI endpoint | Set `VLLM_BASE_URL` and use a model name covered by the vLLM routing branch, or pass the correct `--model`. |
| `pytrec_eval` import fails | Recreate the conda env from `environment.yaml`; do not mix the server Docker requirements into the main env. |
| Results are written under the source tree | Set `FITTEXT_RUNTIME_ROOT` to an ignored run root and keep generated files out of git. |
