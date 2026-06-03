#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
if [[ -f "${REPO_ROOT}/.env.narval" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/.env.narval"
  set +a
fi

export NEXT_CHAMELEONS_ARTIFACT_ROOT="${SCRATCH:-${REPO_ROOT}}/next_chameleons_artifacts"
export NEXT_CHAMELEONS_RAW_CACHE_ROOT="${SCRATCH:-${REPO_ROOT}}/next_chameleons_raw_cache"
export HF_HOME="${SCRATCH:-${REPO_ROOT}}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export UV_CACHE_DIR="${SCRATCH:-${REPO_ROOT}}/.cache/uv"

mkdir -p "${NEXT_CHAMELEONS_ARTIFACT_ROOT}/prefetch" \
  "${NEXT_CHAMELEONS_RAW_CACHE_ROOT}" "${HF_DATASETS_CACHE}" \
  "${TRANSFORMERS_CACHE}" "${UV_CACHE_DIR}"

uv run --extra dev next-chameleons prefetch-assets \
  paper_minimal_gemma2_2b_real \
  paper_gemma2_2b_real \
  paper_gemma2_9b_real \
  paper_llama31_8b_real \
  paper_qwen25_7b_real \
  --include-generator \
  --include-rater \
  --rater-model "${NEXT_CHAMELEONS_RATER_MODEL:-Qwen/Qwen3.5-27B}" \
  --cache-root "${HF_HOME}" \
  --output-path "${NEXT_CHAMELEONS_ARTIFACT_ROOT}/prefetch/paper_replication_assets.json"
