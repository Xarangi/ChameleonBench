#!/bin/bash
# L3 fast track (tica/windows.py). Usage:
#   sbatch [--time=..] scripts/slurm/tica_windows.sbatch <venv python> <tica.py windows args...>
# Copies the model snapshot to node-local disk first (loading from /scratch took minutes per job).
# ctb-liyue_gpu has the HIGHER fairshare factor (0.352 vs def-rrabba_gpu 0.272 on 10-10), so it
# gets better priority even though def-rrabba has no recent usage. Override with --account=.
#SBATCH --account=ctb-liyue_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/s/sneheel/ChameleonBench/logs/%x-%j.out
set -euo pipefail
PY=$1; shift
export TRITON_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/triton  # home is at quota (10-10)
# The snapshot comes from the config being run, never hardcoded: a 9B config with a 4B snapshot
# would silently score 9B probes against 4B activations (caught before it ran, 10-10).
CFG=$(printf '%s\n' "$@" | grep -A1 -x -- --config | tail -1)
SNAP=$(cd /scratch/s/sneheel/ChameleonBench && PYTHONPATH=src $PY -c "
from next_chameleons.tica.config import load_window_config as l; print(l('$CFG').model.path)")
test -d "$SNAP" || { echo "no snapshot for config $CFG: $SNAP"; exit 1; }
LOCAL=${SLURM_TMPDIR:-/tmp}/$(basename $(dirname $(dirname $SNAP)))
t0=$(date +%s); mkdir -p $LOCAL; cp -rL $SNAP/. $LOCAL/; echo "model copy: $(( $(date +%s) - t0 ))s"
nvidia-smi --query-gpu=name,memory.total --format=csv
cd /scratch/s/sneheel/ChameleonBench
echo "Running: $PY scripts/tica.py windows $* --model-path $LOCAL"
env PYTHONPATH=/scratch/s/sneheel/ChameleonBench/src $PY scripts/tica.py windows "$@" --model-path $LOCAL
