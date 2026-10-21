#!/bin/bash
# Reasoning-mode check (scripts/tica_thinking_eval.py). Usage: sbatch tica_thinking_eval.sh CONFIG OUT [args]
#SBATCH --account=ctb-liyue_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=06:00:00  # 3 h truncated l5 dodge s50 races (many valid refits)
#SBATCH --job-name=tica-think
#SBATCH --output=/scratch/s/sneheel/ChameleonBench/logs/%x-%j.out
set -euo pipefail
CFG=$1; OUT=$2; shift 2
PY=${PY:-/scratch/s/sneheel/venvs/q35t29/bin/python}
export TRITON_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/triton
source /scratch/s/sneheel/ChameleonBench/scripts/slurm/_local_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SNAP=$(cd /scratch/s/sneheel/ChameleonBench && PYTHONPATH=src $PY -c "
from next_chameleons.tica.config import load_run_config as l; print(l('$CFG').model.path)")
LOCAL=${SLURM_TMPDIR:-/tmp}/$(basename $(dirname $(dirname $SNAP)))
mkdir -p $LOCAL; cp -rL $SNAP/. $LOCAL/
cd /scratch/s/sneheel/ChameleonBench
env PYTHONPATH=src $PY scripts/tica_thinking_eval.py --config "$CFG" --model-path "$LOCAL" --out "$OUT" "$@"
