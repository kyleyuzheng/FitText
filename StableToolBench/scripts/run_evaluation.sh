#!/usr/bin/env bash
# Convert raw inference output to evaluation format and compute pass rates.
#
# Usage:
#   bash scripts/run_evaluation.sh --model_name <raw-output-run-name>
#   bash scripts/run_evaluation.sh --model_name MY_RUN --test_sets "G1_instruction G1_tool"
#
# Required environment variables:
#   OPENAI_API_KEY   - OpenAI API key (used by the GPT-based evaluator)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
STB_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

repo_abs_path() {
  case "$1" in
    "") printf '%s\n' "" ;;
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$REPO_ROOT" "${1#./}" ;;
  esac
}

cd "$SCRIPT_DIR/.."

# ------------------- DEFAULTS -------------------
MODEL_NAME=""
METHOD="${METHOD:-DFS_woFilter_w2}"
FITTEXT_RUNTIME_ROOT="$(repo_abs_path "${FITTEXT_RUNTIME_ROOT:-runs}")"
RAW_ANSWER_PATH="${RAW_ANSWER_PATH:-${FITTEXT_RUNTIME_ROOT}/stb/raw_output}"
CONVERTED_ANSWER_PATH="${CONVERTED_ANSWER_PATH:-${FITTEXT_RUNTIME_ROOT}/stb/model_predictions_converted}"
PASS_RATE_SAVE_PATH="${PASS_RATE_SAVE_PATH:-${FITTEXT_RUNTIME_ROOT}/stb/pass_rate_results}"
TEST_IDS_DIR="${TEST_IDS_DIR:-${STB_ROOT}/solvable_queries/test_query_ids}"
RAW_ANSWER_PATH="$(repo_abs_path "$RAW_ANSWER_PATH")"
CONVERTED_ANSWER_PATH="$(repo_abs_path "$CONVERTED_ANSWER_PATH")"
PASS_RATE_SAVE_PATH="$(repo_abs_path "$PASS_RATE_SAVE_PATH")"
TEST_IDS_DIR="$(repo_abs_path "$TEST_IDS_DIR")"
MAX_EVAL_THREADS="${MAX_EVAL_THREADS:-1}"
EVALUATE_TIMES="${EVALUATE_TIMES:-3}"
FORCE_REEVAL="${FORCE_REEVAL:-0}"
TEST_SETS="G1_category G1_instruction G1_tool G2_category G2_instruction G3_instruction"

usage() {
  cat <<USAGE
Usage: $(basename "$0") [OPTIONS]

Options:
  --model_name NAME     Name of the model run to evaluate (required)
  --method METHOD       Search method (default: DFS_woFilter_w2)
  --test_sets "SET..."  Space-separated test sets (default: all G1+G2+G3)
  --force               Force re-evaluation even if results exist
  --help                Show this help

Environment variables:
  OPENAI_API_KEY        Required. Used by GPT-based evaluator.
  RAW_ANSWER_PATH       Raw output directory (default: \$FITTEXT_RUNTIME_ROOT/stb/raw_output)
  FITTEXT_RUNTIME_ROOT  Runtime root for outputs (default: repo-root runs/)
  STB_JUDGE_MODEL       Judge model override (default: configs/model_pins.yaml)
  MAX_EVAL_THREADS      Parallel eval threads (default: 1)
  EVALUATE_TIMES        Number of judge runs per query (default: 3)
USAGE
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name)   MODEL_NAME="$2"; shift 2 ;;
    --method)       METHOD="$2"; shift 2 ;;
    --test_sets)    TEST_SETS="$2"; shift 2 ;;
    --force)        FORCE_REEVAL=1; shift ;;
    --help)         usage ;;
    *)              echo "Unknown option: $1"; usage ;;
  esac
done

[ -z "$MODEL_NAME" ] && { echo "ERROR: --model_name is required"; usage; }
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY environment variable}"
# tooleval reads the judge key from OPENAI_KEY (or OPENAI_API_KEY as fallback).
# API_POOL_FILE is reserved for users who supply a JSON key-pool file.
export OPENAI_KEY="${OPENAI_KEY:-${OPENAI_API_KEY}}"
if [ -z "${STB_JUDGE_MODEL:-}" ]; then
  STB_JUDGE_MODEL="$(python3 - <<'PY'
from pathlib import Path
import yaml

pin_path = Path("../configs/model_pins.yaml")
with pin_path.open(encoding="utf-8") as fh:
    print(yaml.safe_load(fh)["evaluators"]["judge"])
PY
)"
  export STB_JUDGE_MODEL
fi
export EVAL_MODEL="${EVAL_MODEL:-${STB_JUDGE_MODEL}}"

mkdir -p "${CONVERTED_ANSWER_PATH}" "${PASS_RATE_SAVE_PATH}"

already_done() {
  local save_dir="$1"
  [ -f "${save_dir}/.done" ] && return 0
  [ -d "${save_dir}" ] && find "${save_dir}" -type f \( -name '*.json' -o -name '*.csv' \) | head -n1 | grep -q . && return 0
  return 1
}

pushd toolbench/tooleval >/dev/null

read -ra SETS <<< "$TEST_SETS"
for test_set in "${SETS[@]}"; do
  group_dir="${RAW_ANSWER_PATH}/${MODEL_NAME}/${test_set}"
  if [ ! -d "${group_dir}" ]; then
    echo "[SKIP] ${test_set}: not found at ${group_dir}"
    continue
  fi

  echo "=== ${test_set} ==="
  shopt -s nullglob
  for sub_dir in "${group_dir}"/*/ ; do
    [ -d "${sub_dir}" ] || continue
    submethod="$(basename "${sub_dir}")"
    REF_MODEL="${MODEL_NAME}__${test_set}__${submethod}"

    out_dir="${CONVERTED_ANSWER_PATH}/${REF_MODEL}"
    mkdir -p "${out_dir}"
    out_file="${out_dir}/${test_set}.json"

    save_dir="${PASS_RATE_SAVE_PATH}/${REF_MODEL}"
    mkdir -p "${save_dir}"

    if [ "${FORCE_REEVAL}" != "1" ] && already_done "${save_dir}"; then
      echo "[SKIP] ${test_set} :: ${submethod} (results exist)"
      continue
    fi

    # Convert
    if [ "${FORCE_REEVAL}" != "1" ] && [ -s "${out_file}" ]; then
      echo "[CONVERT] ${test_set} :: ${submethod} (already converted)"
    else
      echo "[CONVERT] ${test_set} :: ${submethod}"
      python convert_to_answer_format.py \
        --answer_dir "${sub_dir}" \
        --method "${METHOD}" \
        --output "${out_file}" || true
    fi

    [ -s "${out_file}" ] || { echo "  -> empty output, skipping eval"; continue; }

    # Evaluate
    echo "[EVAL] ${test_set} :: ${submethod}"
    python eval_pass_rate.py \
      --converted_answer_path "${CONVERTED_ANSWER_PATH}" \
      --save_path "${save_dir}" \
      --reference_model "${REF_MODEL}" \
      --test_ids "${TEST_IDS_DIR}" \
      --max_eval_threads "${MAX_EVAL_THREADS}" \
      --evaluate_times "${EVALUATE_TIMES}" \
      --test_set "${test_set}" \
      --overwrite

    touch "${save_dir}/.done"
    echo
  done
  shopt -u nullglob
done

popd >/dev/null
echo "Evaluation complete. Results at: ${PASS_RATE_SAVE_PATH}/"
