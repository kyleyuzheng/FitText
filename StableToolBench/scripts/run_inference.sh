#!/usr/bin/env bash
# Run the open-domain QA pipeline with a specific retrieval strategy.
#
# Usage:
#   bash scripts/run_inference.sh --strategy single_pass --dataset G1_instruction
#   bash scripts/run_inference.sh --strategy memetic --dataset all
#   bash scripts/run_inference.sh --help
#
# Required environment variables:
#   OPENAI_API_KEY   - OpenAI API key
#   TOOLBENCH_KEY    - ToolBench service key
#
# Optional environment variables:
#   SERVICE_URL      - Virtual API server URL (default: http://localhost:8080/virtual)
#   TOOL_ROOT_DIR    - Path to tool JSON corpus
#   CORPUS_BASE_DIR  - Base directory for retrieval corpora
#   FITTEXT_RUNTIME_ROOT - Ignored runtime root for outputs (default: repo-root runs/)
#   OUTPUT_ROOT      - Base directory for raw output
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

repo_abs_path() {
  case "$1" in
    "") printf '%s\n' "" ;;
    *_NOT_SET) printf '%s\n' "$1" ;;
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$REPO_ROOT" "${1#./}" ;;
  esac
}

cd "$SCRIPT_DIR/.."
export PYTHONPATH=./

# ------------------- DEFAULTS -------------------
STRATEGY="single_pass"
DATASET="all"
# Default model comes from SERVED_MODEL_NAME or configs/model_pins.yaml.
GPT_MODEL="${SERVED_MODEL_NAME:-$(python3 -c "import yaml; print(yaml.safe_load(open('$SCRIPT_DIR/../../configs/model_pins.yaml'))['agents']['main_solver'])")}"
METHOD="DFS_woFilter_w2"
RETRIEVAL_MODEL="${RETRIEVAL_MODEL:-$(python3 - <<'PY'
from pathlib import Path
import yaml

with Path("../configs/model_pins.yaml").open(encoding="utf-8") as fh:
    print(yaml.safe_load(fh)["embedders"]["stb"])
PY
)}"
RETRIEVAL_MODEL_SHORT="simcse-roberta-large"
RETRIEVE_MODE="dynamic"
PLANNING_MODEL="${PLANNING_MODEL:-$(python3 - <<'PY'
from pathlib import Path
import yaml

with Path("../configs/model_pins.yaml").open(encoding="utf-8") as fh:
    print(yaml.safe_load(fh)["agents"]["o3_mini"])
PY
)}"
RETRIEVER_SERVER_URL=""
SERVICE_URL="${SERVICE_URL:-http://localhost:8080/virtual}"
TOOL_ROOT_DIR="${TOOL_ROOT_DIR:-}"
CORPUS_BASE_DIR="${CORPUS_BASE_DIR:-}"
FITTEXT_RUNTIME_ROOT="$(repo_abs_path "${FITTEXT_RUNTIME_ROOT:-runs}")"
OUTPUT_ROOT="${OUTPUT_ROOT:-${FITTEXT_RUNTIME_ROOT}/stb/raw_output}"
OUTPUT_ROOT="$(repo_abs_path "$OUTPUT_ROOT")"
DRY_RUN=0

# Strategy-specific defaults
POP_SIZE=5
GEN_NUM=3
SCATTER_SIZE=5
DBD_TURNS=3
BASE_TEMP=0.9

usage() {
  cat <<USAGE
Usage: $(basename "$0") [OPTIONS]

Options:
  --strategy STRATEGY   Retrieval strategy (default: single_pass)
                        Options: normal, single_pass, dbd_t3, dbd_t5,
                        scattershot_s5, scattershot_s10, memetic,
                        memetic_toolret, reinvoke
  --dataset DATASET     Test dataset or 'all' (default: all)
                        Options: G1_category, G1_instruction, G1_tool,
                        G2_category, G2_instruction, G3_instruction, all
  --model MODEL         OpenAI model name (default: SERVED_MODEL_NAME or
                        agents.main_solver from configs/model_pins.yaml)
  --method METHOD       Search method (default: DFS_woFilter_w2)
  --planning-model M    Planning model override (default: agents.o3_mini pin). Use --no-planner to disable.
  --no-planner          Disable the root planner gate
  --retriever-server U  Use shared retriever server at URL
  --dry-run             Print the command without executing
  --help                Show this help message

Environment variables:
  OPENAI_API_KEY        Required. OpenAI API key.
  TOOLBENCH_KEY         Required. ToolBench service key.
  SERVICE_URL           Virtual API server (default: http://localhost:8080/virtual)
  TOOL_ROOT_DIR         Tool JSON corpus directory
  CORPUS_BASE_DIR       Retrieval corpora base directory
  FITTEXT_RUNTIME_ROOT  Runtime root for outputs (default: repo-root runs/)
  OUTPUT_ROOT           Raw output base directory
USAGE
  exit 0
}

NO_PLANNER=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --strategy)     STRATEGY="$2"; shift 2 ;;
    --dataset)      DATASET="$2"; shift 2 ;;
    --model)        GPT_MODEL="$2"; shift 2 ;;
    --method)       METHOD="$2"; shift 2 ;;
    --planning-model) PLANNING_MODEL="$2"; shift 2 ;;
    --no-planner)   NO_PLANNER=1; shift ;;
    --retriever-server) RETRIEVER_SERVER_URL="$2"; shift 2 ;;
    --dry-run)      DRY_RUN=1; shift ;;
    --help)         usage ;;
    *)              echo "Unknown option: $1"; usage ;;
  esac
done

if [ "$DRY_RUN" = "1" ]; then
  OPENAI_API_KEY="${OPENAI_API_KEY:-DRY_RUN_OPENAI_API_KEY}"
  TOOLBENCH_KEY="${TOOLBENCH_KEY:-DRY_RUN_TOOLBENCH_KEY}"
  TOOL_ROOT_DIR="${TOOL_ROOT_DIR:-TOOL_ROOT_DIR_NOT_SET}"
  CORPUS_BASE_DIR="${CORPUS_BASE_DIR:-CORPUS_BASE_DIR_NOT_SET}"
else
  : "${OPENAI_API_KEY:?Set OPENAI_API_KEY environment variable}"
  : "${TOOLBENCH_KEY:?Set TOOLBENCH_KEY environment variable}"
fi

TOOL_ROOT_DIR="$(repo_abs_path "$TOOL_ROOT_DIR")"
CORPUS_BASE_DIR="$(repo_abs_path "$CORPUS_BASE_DIR")"

# GPU selection: this script intentionally does NOT set CUDA_VISIBLE_DEVICES.
# Multi-shard launchers assign devices per shard in the caller's environment,
# and an inner export would clobber that assignment.
export SERVICE_URL

# ------------------- DATASET SPLITS_TO_RUN -------------------
if [ "$DATASET" = "all" ]; then
  SPLITS_TO_RUN=("G1_category" "G1_instruction" "G1_tool" "G2_category" "G2_instruction" "G3_instruction")
else
  SPLITS_TO_RUN=("$DATASET")
fi

corpus_for_group() {
  case "$1" in
    G1_*) echo "${CORPUS_BASE_DIR}/G1/des_corpus.json" ;;
    G2_*) echo "${CORPUS_BASE_DIR}/G2/des_corpus.json" ;;
    G3_*) echo "${CORPUS_BASE_DIR}/G3/des_corpus.json" ;;
    *)    echo "${CORPUS_BASE_DIR}/G1/des_corpus.json" ;;
  esac
}

# ------------------- STRATEGY FLAGS -------------------
build_strategy_args() {
  local s="$1"
  case "$s" in
    normal)
      # RETRIEVE_MODE is forced to normal in the PARENT shell below; a $()-subshell
      # assignment here would be silently discarded.
      echo ""
      ;;
    single_pass)
      echo ""
      ;;
    dbd_t3)
      echo "--dbd --dbd-refine-turns 3"
      ;;
    dbd_t5)
      echo "--dbd --dbd-refine-turns 5"
      ;;
    scattershot_s5)
      echo "--scattershot --size 5"
      ;;
    scattershot_s10)
      echo "--scattershot --size 10"
      ;;
    memetic)
      echo "--memetic --population_size ${POP_SIZE} --generation_num ${GEN_NUM} --base_temp ${BASE_TEMP}"
      ;;
    memetic_toolret)
      echo "--memetic --memetic_toolret --population_size ${POP_SIZE} --generation_num ${GEN_NUM} --base_temp ${BASE_TEMP}"
      ;;
    reinvoke)
      # Re-Invoke baseline: static (normal) retrieval with synthetic-query-augmented
      # embeddings + intent-based multi-view ranking. Index must be pre-built via
      # baselines/build_reinvoke_stb_index.py. RETRIEVE_MODE is forced to normal in
      # the PARENT shell below (a $()-subshell assignment here would be lost).
      echo "--reinvoke"
      ;;
    *)
      echo "Unknown strategy: $s" >&2
      exit 1
      ;;
  esac
}

STRATEGY_ARGS=$(build_strategy_args "$STRATEGY")

# Strategies that require STATIC (normal) retrieval must set RETRIEVE_MODE in the
# PARENT shell — an assignment inside build_strategy_args runs in a $() subshell and
# is silently discarded, which would otherwise leave --strategy normal/reinvoke in
# the default 'dynamic' mode.
case "$STRATEGY" in
  normal|reinvoke) RETRIEVE_MODE="normal" ;;
esac

if [ "$DRY_RUN" != "1" ]; then
  [ -n "$TOOL_ROOT_DIR" ] || { echo "ERROR: set TOOL_ROOT_DIR to the ToolBench tool JSON directory." >&2; exit 2; }
  [ -n "$CORPUS_BASE_DIR" ] || { echo "ERROR: set CORPUS_BASE_DIR to the retrieval corpus base directory." >&2; exit 2; }
fi

# ------------------- MAIN LOOP -------------------
OUT_SUFFIX="${METHOD}_${GPT_MODEL}_${RETRIEVAL_MODEL_SHORT}_${RETRIEVE_MODE}"

for group in "${SPLITS_TO_RUN[@]}"; do
  corpus_path="$(corpus_for_group "$group")"
  out_dir="${OUTPUT_ROOT}/${OUT_SUFFIX}/${group}/${STRATEGY}"
  if [ "$DRY_RUN" != "1" ]; then
    mkdir -p "$out_dir"
  fi

  EXTRA_ARGS=""
  if [ -n "$RETRIEVER_SERVER_URL" ]; then
    EXTRA_ARGS="--retriever_server_url $RETRIEVER_SERVER_URL"
  fi
  if [ "$NO_PLANNER" = "1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --no_planner"
  else
    EXTRA_ARGS="$EXTRA_ARGS --planning_model $PLANNING_MODEL"
  fi

  # Route the agent backbone to vLLM when running a Qwen-family model.
  # Without --base_url, qa_pipeline defaults to https://api.openai.com/v1
  # which rejects Qwen model IDs (400 invalid_request_error).
  BASE_URL_ARG=""
  if [[ "$GPT_MODEL" == Qwen* || "$GPT_MODEL" == qwen* || "$GPT_MODEL" == DeepSeek* || "$GPT_MODEL" == deepseek* ]]; then
    BASE_URL_ARG="--base_url ${VLLM_BASE_URL:-http://127.0.0.1:8001/v1}"
  fi
  OPENAI_KEY_ARG="$OPENAI_API_KEY"
  TOOLBENCH_KEY_ARG="$TOOLBENCH_KEY"
  if [ "$DRY_RUN" = "1" ]; then
    OPENAI_KEY_ARG="<redacted-openai-key>"
    TOOLBENCH_KEY_ARG="<redacted-toolbench-key>"
  fi

  CMD="python toolbench/inference/qa_pipeline_open_domain.py \
    --corpus_path \"$corpus_path\" \
    --retrieval_model_path \"$RETRIEVAL_MODEL\" \
    --retrieved_api_nums 5 \
    --tool_root_dir \"$TOOL_ROOT_DIR\" \
    --backbone_model chatgpt_function \
    --chatgpt_model \"$GPT_MODEL\" \
    --openai_key \"$OPENAI_KEY_ARG\" \
    --max_observation_length 1024 \
    --method \"$METHOD\" \
    --input_query_file \"${CORPUS_BASE_DIR}/../test_instruction/${group}.json\" \
    --output_answer_file \"$out_dir\" \
    --toolbench_key \"$TOOLBENCH_KEY_ARG\" \
    --retrieve_mode \"$RETRIEVE_MODE\" \
    $BASE_URL_ARG $STRATEGY_ARGS $EXTRA_ARGS"

  if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY-RUN] $group :: $STRATEGY"
    echo "$CMD"
    echo
  else
    echo "[START] $group :: $STRATEGY"
    eval "$CMD"
    echo "[DONE]  $group :: $STRATEGY"
  fi
done
