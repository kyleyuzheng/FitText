#!/usr/bin/env bash
# Launch the shared SimCSE retriever server.
# This avoids loading the retrieval model in each inference process.
#
# Usage:
#   bash scripts/run_retriever_server.sh
#   bash scripts/run_retriever_server.sh --gpu 0 --port 8090
#
# The server exposes:
#   GET  /health        - Health check
#   GET  /corpus_info   - Loaded corpora info
#   POST /retrieve      - Retrieve top-k tools for a query
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

repo_abs_path() {
  case "$1" in
    "") printf '%s\n' "" ;;
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$REPO_ROOT" "${1#./}" ;;
  esac
}

cd "$SCRIPT_DIR/.."
export PYTHONPATH=./

GPU="${GPU:-0}"
PORT="${PORT:-8090}"
MODEL_PATH="${MODEL_PATH:-$(python3 - <<'PY'
from pathlib import Path
import yaml

with Path("../configs/model_pins.yaml").open(encoding="utf-8") as fh:
    print(yaml.safe_load(fh)["embedders"]["stb"])
PY
)}"
CORPUS_BASE="${CORPUS_BASE:-${CORPUS_BASE_DIR:-}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)        GPU="$2"; shift 2 ;;
    --port)       PORT="$2"; shift 2 ;;
    --model)      MODEL_PATH="$2"; shift 2 ;;
    --corpus-base) CORPUS_BASE="$2"; shift 2 ;;
    --help)       echo "Usage: $(basename "$0") [--gpu ID] [--port PORT] [--model PATH] [--corpus-base DIR]"; exit 0 ;;
    *)            echo "Unknown option: $1"; exit 1 ;;
  esac
done

[ -n "$CORPUS_BASE" ] || { echo "ERROR: set CORPUS_BASE or CORPUS_BASE_DIR to the retrieval corpus base directory." >&2; exit 2; }
CORPUS_BASE="$(repo_abs_path "$CORPUS_BASE")"

export CUDA_VISIBLE_DEVICES="$GPU"

echo "Starting retriever server on port ${PORT} (GPU ${GPU})..."
echo "Model: ${MODEL_PATH}"
echo "Corpora: ${CORPUS_BASE}/{G1,G2,G3}/des_corpus.json"

python -m toolbench.inference.LLM.retriever_server \
  --model_path "$MODEL_PATH" \
  --corpus_paths \
    "G1=${CORPUS_BASE}/G1/des_corpus.json" \
    "G2=${CORPUS_BASE}/G2/des_corpus.json" \
    "G3=${CORPUS_BASE}/G3/des_corpus.json" \
  --port "$PORT"
