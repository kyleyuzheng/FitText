#!/usr/bin/env bash
# Launch the virtual API server (cached tool responses + LLM-generated fallbacks).
#
# Usage:
#   bash scripts/run_server.sh
#
# The listen port, simulator model, and data paths are all read from
# server/config.yml (main.py resolves it relative to its own location).
#
# Required environment variables:
#   OPENAI_API_KEY   - OpenAI API key (for fallback response generation)
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

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY environment variable}"
: "${TOOL_ROOT_DIR:?Set TOOL_ROOT_DIR to the ToolBench tool JSON directory}"
TOOL_ROOT_DIR="$(repo_abs_path "$TOOL_ROOT_DIR")"
export TOOL_ROOT_DIR
export FITTEXT_RUNTIME_ROOT="$(repo_abs_path "${FITTEXT_RUNTIME_ROOT:-runs}")"
export STB_SIMULATOR_MODEL="${STB_SIMULATOR_MODEL:-$(python3 - <<'PY'
import yaml
print(yaml.safe_load(open('../configs/model_pins.yaml'))['evaluators']['simulator'])
PY
)}"
mkdir -p "${FITTEXT_RUNTIME_ROOT}/stb_server"

echo "Starting virtual API server (settings: server/config.yml)..."
python server/main.py
