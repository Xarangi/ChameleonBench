#!/bin/bash
# Adaptive-monitor multi-turn RL (tica.py rl, use_adaptive). Arm B/C. Usage:
#   sbatch [--time=..] scripts/slurm/tica_l3rl.sh <config> [extra tica.py args...]
# ctb-liyue_gpu has the HIGHER fairshare factor (0.352 vs def-rrabba_gpu 0.272, 10-10), so better
# priority despite def-rrabba's idle usage. Override with --account=.
#SBATCH --account=ctb-liyue_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=tica-l3covert
#SBATCH --output=/home/sneheel/chameleons/logs/%x-%j.out
set -euo pipefail
CFG=${1:?config name}; shift || true
PY=${PY:-/home/sneheel/chameleons-qwen35/.venv/bin/python}
export TRITON_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/triton  # home is at quota (10-10)
source /home/sneheel/chameleons/scripts/slurm/_local_env.sh
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# Derive the snapshot from the config, never hardcode it (a 9B config with a 4B copy would score
# 9B probes against 4B activations; caught before it ran, 10-10).
SNAP=$(cd /home/sneheel/chameleons && PYTHONPATH=src $PY -c "
from next_chameleons.tica.config import load_run_config as l; print(l('$CFG').model.path)")
test -d "$SNAP" || { echo "no snapshot for $CFG: $SNAP"; exit 1; }
LOCAL=${SLURM_TMPDIR:-/tmp}/$(basename $(dirname $(dirname $SNAP)))
t0=$(date +%s); mkdir -p $LOCAL; cp -rL $SNAP/. $LOCAL/; echo "model copy: $(( $(date +%s) - t0 ))s"
nvidia-smi --query-gpu=name,memory.total --format=csv
cd /home/sneheel/chameleons
echo "Running: adaptive --config $CFG $* (snapshot $LOCAL)"
env PYTHONPATH=src $PY scripts/tica.py adaptive --config "$CFG" --model-path "$LOCAL" "$@"
