#!/usr/bin/env bash
#SBATCH --job-name=next-chameleon-multiview-eval
#SBATCH --account=ctb-liyue_gpu
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out

# Evaluate a checkpoint across the three probe activation views the paper and
# the Obfuscation Atlas use:
#   1. mean_generation_tokens (paper primary)            -> <out>_mean_generation_tokens
#   2. last_token                                        -> <out>_last_token
#   3. last_token + atlas_followup_truthful_yes          -> <out>_atlas_followup_truthful_yes
# Each view writes a separate real_eval_report.json so nothing is overwritten.
#
# Usage:
#   sbatch multiview_eval.sbatch <experiment> <checkpoint-dir> <output-base> [seed] [extra real-eval args...]
# Pass --external-probe-dir <dir> as an extra arg for golden-path released-probe eval.

set -euo pipefail

EXPERIMENT="${1:?Usage: multiview_eval.sbatch <experiment> <checkpoint-dir> <output-base> [seed] [extra...]}"
CHECKPOINT_DIR="${2:?missing checkpoint-dir (use - to eval the configured base model)}"
OUTPUT_BASE="${3:?missing output-base}"
SEED="${4:-17}"
shift 4 2>/dev/null || shift $#
EXTRA_ARGS=("$@")

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -f "${REPO_ROOT}/.env.narval" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/.env.narval"
  set +a
fi
module load StdEnv/2023 cuda/12.2

export NEXT_CHAMELEONS_ARTIFACT_ROOT="${SCRATCH:-${REPO_ROOT}}/next_chameleons_artifacts"
export NEXT_CHAMELEONS_RAW_CACHE_ROOT="${SCRATCH:-${REPO_ROOT}}/next_chameleons_raw_cache"
export NEXT_CHAMELEONS_PERSISTENT_HF_HOME="${SCRATCH:-${REPO_ROOT}}/.cache/huggingface"
export HF_HOME="${NEXT_CHAMELEONS_PERSISTENT_HF_HOME}"
export HF_HUB_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export UV_CACHE_DIR="${SCRATCH:-${REPO_ROOT}}/.cache/uv"
export UV_NO_SYNC=1
export UV_OFFLINE=1
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NEXT_CHAMELEONS_OFFLINE=1

mkdir -p "${REPO_ROOT}/logs" "${NEXT_CHAMELEONS_ARTIFACT_ROOT}" \
  "${NEXT_CHAMELEONS_RAW_CACHE_ROOT}" "${HF_DATASETS_CACHE}" \
  "${TRANSFORMERS_CACHE}" "${UV_CACHE_DIR}"

# Stage any model snapshots requested by the caller (NEXT_CHAMELEONS_STAGE_MODELS)
# from persistent scratch HF cache into node-local cache where applicable.
source "${REPO_ROOT}/scripts/slurm/stage_hf_cache.sh"
stage_next_chameleons_hf_cache

cd "${REPO_ROOT}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "Slurm allocation:"
  scontrol show job "${SLURM_JOB_ID}" | grep -E 'JobId=|NodeList=|AllocTRES=|TresPerNode=' || true
  if ! scontrol show job "${SLURM_JOB_ID}" | grep -q 'gres/gpu:a100'; then
    echo "Expected an A100 allocation; refusing to run on a non-A100 GPU." >&2
    exit 3
  fi
  nvidia-smi --query-gpu=name,memory.total --format=csv || true
fi

CHECKPOINT_FLAG=()
if [[ "${CHECKPOINT_DIR}" != "-" ]]; then
  CHECKPOINT_FLAG=(--checkpoint-dir "${CHECKPOINT_DIR}")
fi

run_view() {
  local pooling="$1"
  local text_mode="$2"
  local suffix="$3"
  local out="${OUTPUT_BASE}_${suffix}"
  echo "================================================================"
  echo "[multiview-eval] view=${suffix} pooling=${pooling} text_mode=${text_mode}"
  echo "[multiview-eval] output=${out}"
  echo "================================================================"
  uv run --extra train next-chameleons real-eval "${EXPERIMENT}" \
    --seed "${SEED}" \
    "${CHECKPOINT_FLAG[@]}" \
    --output-dir "${out}" \
    --activation-pooling "${pooling}" \
    --activation-text-mode "${text_mode}" \
    "${EXTRA_ARGS[@]}"
}

run_view mean_generation_tokens default mean_generation_tokens
run_view last_token default last_token
run_view last_token atlas_followup_truthful_yes atlas_followup_truthful_yes

echo "[multiview-eval] all three views complete for ${EXPERIMENT} seed=${SEED}"
