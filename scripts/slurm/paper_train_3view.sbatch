#!/usr/bin/env bash
#SBATCH --job-name=next-chameleon-train-3view
#SBATCH --account=ctb-liyue_gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=logs/%x-%j.out

# Paper-faithful real training (v2 protocol: preset data, synonym triggers,
# per-token probe fit, per-sample CE behavior loss, corrected layer index) then
# evaluate the resulting checkpoint across all three probe activation views.
#
# Usage: sbatch paper_train_3view.sbatch <experiment> <seed>

set -euo pipefail

EXPERIMENT="${1:-paper_minimal_gemma2_2b_real}"
SEED="${2:-17}"
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
export WANDB_PROJECT="${WANDB_PROJECT:-next-chameleons}"
export WANDB_DIR="${NEXT_CHAMELEONS_ARTIFACT_ROOT}/wandb"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-20}"
if [[ "${NEXT_CHAMELEONS_WANDB_ONLINE:-0}" == "1" && -n "${WANDB_API_KEY:-}" ]]; then
  export WANDB_MODE="online"
else
  export WANDB_MODE="offline"
fi
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NEXT_CHAMELEONS_OFFLINE=1

mkdir -p "${REPO_ROOT}/logs" "${NEXT_CHAMELEONS_ARTIFACT_ROOT}" \
  "${NEXT_CHAMELEONS_RAW_CACHE_ROOT}" "${HF_DATASETS_CACHE}" \
  "${TRANSFORMERS_CACHE}" "${UV_CACHE_DIR}"

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

RUN_DIR="${NEXT_CHAMELEONS_ARTIFACT_ROOT}/${EXPERIMENT}_${SLURM_JOB_ID:-local}_s${SEED}"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints/${EXPERIMENT}"

echo "[train-3view] experiment=${EXPERIMENT} seed=${SEED} run_dir=${RUN_DIR}"
uv run next-chameleons config-check "${EXPERIMENT}"
uv run next-chameleons paper-readiness-check "${EXPERIMENT}"

uv run --extra train next-chameleons real-train "${EXPERIMENT}" \
  --seed "${SEED}" \
  --output-dir "${RUN_DIR}"

run_view() {
  local pooling="$1"; local text_mode="$2"; local suffix="$3"
  local out="${RUN_DIR}/eval_${suffix}"
  echo "[train-3view] eval view=${suffix} pooling=${pooling} text_mode=${text_mode} -> ${out}"
  uv run --extra train next-chameleons real-eval "${EXPERIMENT}" \
    --seed "${SEED}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --output-dir "${out}" \
    --activation-pooling "${pooling}" \
    --activation-text-mode "${text_mode}"
}

run_view mean_generation_tokens default mean_generation_tokens
run_view last_token default last_token
run_view last_token atlas_followup_truthful_yes atlas_followup_truthful_yes

echo "[train-3view] complete: ${EXPERIMENT} seed=${SEED} checkpoint=${CHECKPOINT_DIR}"
