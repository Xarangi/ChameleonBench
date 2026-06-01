#!/usr/bin/env bash
#SBATCH --job-name=next-chameleon-paper
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

EXPERIMENT="${1:-paper_gemma2_2b_real}"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
export NEXT_CHAMELEONS_EXPERIMENT="${EXPERIMENT}"
export NEXT_CHAMELEONS_ARTIFACT_ROOT="${SCRATCH:-${REPO_ROOT}}/next_chameleons_artifacts"
export NEXT_CHAMELEONS_RAW_CACHE_ROOT="${SCRATCH:-${REPO_ROOT}}/next_chameleons_raw_cache"
export HF_HOME="${SCRATCH:-${REPO_ROOT}}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p "${REPO_ROOT}/logs" "${NEXT_CHAMELEONS_ARTIFACT_ROOT}" "${NEXT_CHAMELEONS_RAW_CACHE_ROOT}"

cd "${REPO_ROOT}"
uv run next-chameleons config-check paper_reproduction
uv run next-chameleons config-check "${EXPERIMENT}"
uv run next-chameleons real-train "${EXPERIMENT}" \
  --output-dir "${NEXT_CHAMELEONS_ARTIFACT_ROOT}/${EXPERIMENT}_${SLURM_JOB_ID:-local}"
uv run next-chameleons real-eval "${EXPERIMENT}" \
  --checkpoint-dir "${NEXT_CHAMELEONS_ARTIFACT_ROOT}/${EXPERIMENT}_${SLURM_JOB_ID:-local}/checkpoints/${EXPERIMENT}" \
  --output-dir "${NEXT_CHAMELEONS_ARTIFACT_ROOT}/${EXPERIMENT}_${SLURM_JOB_ID:-local}"
