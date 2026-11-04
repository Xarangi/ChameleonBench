#!/bin/bash
# PROBE PANEL race (scripts/tica_probe_panel.py). Usage: sbatch tica_probe_panel.sh CONFIG OUT [args...]
# DODGESCOPE WORKTREE COPY (11-02): runs the code of /scratch/s/sneheel/ChameleonBench-dodgescope (not the main
# checkout). TamIA allocates whole nodes; several panels per node go through scripts/slurm/dodgescope_node.sh,
# which calls this script once per task with its own CUDA_VISIBLE_DEVICES / SLURM_TMPDIR.
#SBATCH --account=aip-rrabba
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --job-name=probe-panel
#SBATCH --output=/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/dodgescope_1102/logs/%x-%j.out
set -euo pipefail
WT=/scratch/s/sneheel/ChameleonBench-dodgescope
CFG=$1; OUT=$2; shift 2
PY=${PY:-/scratch/s/sneheel/venvs/q35t29/bin/python}
export TRITON_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/triton
source $WT/scripts/slurm/_local_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SNAP=$(cd $WT && PYTHONPATH=$WT/src $PY -c "
from next_chameleons.tica.config import load_run_config as l; print(l('$CFG').model.path)")
LOCAL=${SLURM_TMPDIR:-/tmp}/$(basename $(dirname $(dirname $SNAP)))
mkdir -p $LOCAL; cp -rL $SNAP/. $LOCAL/
cd $WT
env PYTHONPATH=$WT/src $PY scripts/tica_probe_panel.py --config "$CFG" --model-path "$LOCAL" --out "$OUT" "$@"
