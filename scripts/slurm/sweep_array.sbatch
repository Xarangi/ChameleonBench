#!/usr/bin/env bash
#SBATCH --job-name=next-chameleon-sweep
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

cd "${REPO_ROOT}"
mkdir -p logs runs

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
