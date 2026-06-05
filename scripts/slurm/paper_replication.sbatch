#!/usr/bin/env bash
#SBATCH --job-name=next-chameleon-paper
#SBATCH --account=ctb-liyue_gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=logs/%x-%j.out

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
export NEXT_CHAMELEONS_EXPERIMENT="${EXPERIMENT}"
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
PREFETCH_MANIFEST="${NEXT_CHAMELEONS_ARTIFACT_ROOT}/prefetch/paper_replication_assets.json"

mkdir -p "${REPO_ROOT}/logs" "${NEXT_CHAMELEONS_ARTIFACT_ROOT}" \
  "${NEXT_CHAMELEONS_RAW_CACHE_ROOT}" "${HF_DATASETS_CACHE}" \
  "${TRANSFORMERS_CACHE}" "${UV_CACHE_DIR}"

case "${EXPERIMENT}" in
  paper_minimal_gemma2_2b_real|paper_gemma2_2b_real|adaptive_gemma2_2b_real)
    DEFAULT_STAGE_MODELS="google/gemma-2-2b-it"
    ;;
  paper_gemma2_9b_real|language_generalization_real)
    DEFAULT_STAGE_MODELS="IlyaGusev/gemma-2-9b-it-abliterated"
    ;;
  paper_llama31_8b_real)
    DEFAULT_STAGE_MODELS="meta-llama/Llama-3.1-8B-Instruct"
    ;;
  paper_qwen25_7b_real)
    DEFAULT_STAGE_MODELS="Qwen/Qwen2.5-7B-Instruct"
    ;;
  *)
    DEFAULT_STAGE_MODELS=""
    ;;
esac
DEFAULT_STAGE_DATASETS="AlignmentResearch/DolusChat;JailbreakBench/JBB-Behaviors;scale-safety-research/roleplaying;scale-safety-research/insider_trading"
export NEXT_CHAMELEONS_STAGE_MODELS="${NEXT_CHAMELEONS_STAGE_MODELS:-${DEFAULT_STAGE_MODELS}}"
export NEXT_CHAMELEONS_STAGE_DATASETS="${NEXT_CHAMELEONS_STAGE_DATASETS:-${DEFAULT_STAGE_DATASETS}}"
source "${REPO_ROOT}/scripts/slurm/stage_hf_cache.sh"
stage_next_chameleons_hf_cache

cd "${REPO_ROOT}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "Slurm allocation:"
  scontrol show job "${SLURM_JOB_ID}" | grep -E 'JobId=|Partition=|NodeList=|ReqTRES=|AllocTRES=|TresPerNode=' || true
  if ! scontrol show job "${SLURM_JOB_ID}" | grep -q 'gres/gpu:a100'; then
    echo "Expected an A100 allocation; refusing to run on a non-A100 GPU." >&2
    exit 3
  fi
  nvidia-smi --query-gpu=name,memory.total --format=csv || true
fi
if [[ ! -f "${PREFETCH_MANIFEST}" ]]; then
  echo "Missing prefetch manifest: ${PREFETCH_MANIFEST}" >&2
  echo "Run scripts/prefetch_paper_assets.sh from an internet-capable login context first." >&2
  exit 2
fi
uv run next-chameleons config-check paper_reproduction
uv run next-chameleons config-check "${EXPERIMENT}"
uv run next-chameleons paper-readiness-check "${EXPERIMENT}"
uv run next-chameleons paper-data-check "${EXPERIMENT}" \
  --artifact-root "${NEXT_CHAMELEONS_ARTIFACT_ROOT}/${EXPERIMENT}_${SLURM_JOB_ID:-local}" \
  --raw-cache-root "${NEXT_CHAMELEONS_RAW_CACHE_ROOT}"
uv run --extra train next-chameleons real-train "${EXPERIMENT}" \
  --seed "${SEED}" \
  --output-dir "${NEXT_CHAMELEONS_ARTIFACT_ROOT}/${EXPERIMENT}_${SLURM_JOB_ID:-local}"
uv run --extra train next-chameleons real-eval "${EXPERIMENT}" \
  --seed "${SEED}" \
  --checkpoint-dir "${NEXT_CHAMELEONS_ARTIFACT_ROOT}/${EXPERIMENT}_${SLURM_JOB_ID:-local}/checkpoints/${EXPERIMENT}" \
  --output-dir "${NEXT_CHAMELEONS_ARTIFACT_ROOT}/${EXPERIMENT}_${SLURM_JOB_ID:-local}"
