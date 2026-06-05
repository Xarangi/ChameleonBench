#!/usr/bin/env bash
#SBATCH --job-name=next-chameleon-sweep
#SBATCH --account=ctb-liyue_gpu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-0
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

SWEEP="${1:-quick_iteration}"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
PLAN_PATH="${REPO_ROOT}/runs/${SWEEP}_slurm_plan.json"
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
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NEXT_CHAMELEONS_OFFLINE=1

cd "${REPO_ROOT}"
mkdir -p logs runs "${NEXT_CHAMELEONS_ARTIFACT_ROOT}" \
  "${NEXT_CHAMELEONS_RAW_CACHE_ROOT}" "${HF_DATASETS_CACHE}" \
  "${TRANSFORMERS_CACHE}" "${UV_CACHE_DIR}"

source "${REPO_ROOT}/scripts/slurm/stage_hf_cache.sh"
stage_next_chameleons_hf_cache

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "Slurm allocation:"
  scontrol show job "${SLURM_JOB_ID}" | grep -E 'JobId=|Partition=|NodeList=|ReqTRES=|AllocTRES=|TresPerNode=' || true
  if ! scontrol show job "${SLURM_JOB_ID}" | grep -q 'gres/gpu:a100'; then
    echo "Expected an A100 allocation; refusing to run on a non-A100 GPU." >&2
    exit 3
  fi
  nvidia-smi --query-gpu=name,memory.total --format=csv || true
fi

uv run next-chameleons sweep-plan "${SWEEP}" --output-path "${PLAN_PATH}"

COMMAND="$(uv run python - "${PLAN_PATH}" "${SLURM_ARRAY_TASK_ID:-0}" <<'PY'
import json
import sys

plan_path = sys.argv[1]
task_id = int(sys.argv[2])
payload = json.load(open(plan_path, encoding="utf-8"))
cases = payload["cases"]
if task_id >= len(cases):
    raise SystemExit(f"Task {task_id} out of range for {len(cases)} cases")
print(cases[task_id]["cli"])
PY
)"

echo "Running sweep=${SWEEP} task=${SLURM_ARRAY_TASK_ID:-0}: ${COMMAND}"
eval "${COMMAND}"
