#!/usr/bin/env bash
# setup_colt.sh — Clone the upstream COLT repo at the pinned commit SHA and
# install its dependencies into the active Python environment.
#
# Baseline repositories are *cloned* (not
# vendored, not submodules) and the cloned commit is pinned in
# baselines/REPRODUCTION.md. Re-running this script is idempotent — if the
# repo is already at the pinned SHA, the clone step is skipped.
#
# Usage:
#   bash scripts/setup_colt.sh
#   # then in your shell:
#   export COLT_PATH=<path printed at end of script>
#   export COLT_CKPT=<path to checkpoint, if available>
#
# COLT repo:    https://github.com/quchangle1/COLT
# Pinned SHA:   bbec292c38cabe3fbae098f50cc31b2e9f7e8fca (CIKM 2024 release)
# See baselines/REPRODUCTION.md for the full reproduction protocol.

set -euo pipefail

REPO_URL="https://github.com/quchangle1/COLT.git"
PINNED_SHA="bbec292c38cabe3fbae098f50cc31b2e9f7e8fca"

# Resolve paths.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# Default install location follows the .gitignored external/ convention.
DEFAULT_COLT_PATH="${PROJECT_ROOT}/baselines/external/colt"
INSTALL_PATH="${COLT_PATH:-${DEFAULT_COLT_PATH}}"

echo "[setup_colt] target install path: ${INSTALL_PATH}"
echo "[setup_colt] pinning commit:      ${PINNED_SHA}"

# ----------------------------------------------------------------------------
# Step 1: clone (or refresh) the COLT repo and lock to the pinned SHA.
# ----------------------------------------------------------------------------
if [[ -d "${INSTALL_PATH}/.git" ]]; then
    current_sha="$(git -C "${INSTALL_PATH}" rev-parse HEAD)"
    if [[ "${current_sha}" == "${PINNED_SHA}" ]]; then
        echo "[setup_colt] already at pinned SHA — skipping fetch/checkout."
    else
        echo "[setup_colt] existing clone at ${current_sha}, re-pinning to ${PINNED_SHA}"
        git -C "${INSTALL_PATH}" fetch origin --tags
        git -C "${INSTALL_PATH}" checkout "${PINNED_SHA}"
    fi
else
    echo "[setup_colt] cloning COLT from ${REPO_URL}..."
    mkdir -p "$(dirname "${INSTALL_PATH}")"
    git clone "${REPO_URL}" "${INSTALL_PATH}"
    git -C "${INSTALL_PATH}" checkout "${PINNED_SHA}"
fi

verified_sha="$(git -C "${INSTALL_PATH}" rev-parse HEAD)"
if [[ "${verified_sha}" != "${PINNED_SHA}" ]]; then
    echo "[setup_colt] ERROR: HEAD ${verified_sha} does not match pinned SHA ${PINNED_SHA}." >&2
    exit 1
fi
echo "[setup_colt] verified HEAD = ${verified_sha}"

# ----------------------------------------------------------------------------
# Step 2: install Python deps. Upstream COLT has no requirements.txt — we pin
# the versions listed in their README explicitly. Prefer uv pip when present.
# ----------------------------------------------------------------------------
PIP_INSTALL_CMD=(pip install --user --quiet)
if command -v uv >/dev/null 2>&1; then
    PIP_INSTALL_CMD=(uv pip install --quiet)
fi

# If upstream ever adds a requirements.txt we honour it; otherwise install the
# pinned set from REPRODUCTION.md.
if [[ -f "${INSTALL_PATH}/requirements.txt" ]]; then
    echo "[setup_colt] installing from upstream requirements.txt..."
    "${PIP_INSTALL_CMD[@]}" -r "${INSTALL_PATH}/requirements.txt"
else
    echo "[setup_colt] installing COLT-pinned deps (no upstream requirements.txt)..."
    "${PIP_INSTALL_CMD[@]}" \
        "numpy==1.21.6" \
        "pandas==1.3.5" \
        "torch==1.13.1" \
        "pyyaml>=6.0" \
        "tqdm>=4.65" \
        "tensorboard>=2.10" \
        "scipy>=1.7" \
        "transformers>=4.20" \
        || echo "[setup_colt] WARN: dep install partial — Python version may need adjustment (upstream tested on 3.8)."
fi

# ----------------------------------------------------------------------------
# Step 3: pre-create checkpoint dir and print next-step pointers.
# ----------------------------------------------------------------------------
mkdir -p "${INSTALL_PATH}/checkpoints"
mkdir -p "${INSTALL_PATH}/PLMs"

cat <<EOF

[setup_colt] DONE. Repo cloned + pinned at ${INSTALL_PATH}
            HEAD = ${verified_sha}

Next steps (see baselines/REPRODUCTION.md for the full protocol):

  1. Pull the Contriever PLM into ${INSTALL_PATH}/PLMs/:
     huggingface-cli download nthakur/contriever-base-msmarco \\
         --local-dir "${INSTALL_PATH}/PLMs/contriever-base-msmarco"

  2. Run COLT's two-stage training (GPU required, ~6-12h on a single A100):
     cd "${INSTALL_PATH}"
     python train_sbert.py
     python train.py -g 0 -m COLT -d ToolLens

  3. Export the env vars our wrapper looks for:
     export COLT_PATH="${INSTALL_PATH}"
     export COLT_CKPT="${INSTALL_PATH}/checkpoints/<trained_checkpoint>"
     export COLT_DATASET=ToolLens   # one of: ToolLens, ToolBenchG2, ToolBenchG3

  4. Smoke-test the wrapper:
     pytest tests/smoke_colt_real.py --run-colt-integration -v

EOF
