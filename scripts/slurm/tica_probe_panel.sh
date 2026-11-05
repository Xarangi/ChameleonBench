#!/bin/bash
# PROBE PANEL race (scripts/tica_probe_panel.py). Usage: sbatch tica_probe_panel.sh CONFIG OUT [args...]
# Runs the code of the checkout this script lives in (TICA_REPO overrides; under sbatch, which copies the script,
# the main checkout). TamIA allocates whole nodes; several panels per node go through tamia_mechanism_node.sh /
# dodgescope_node.sh, which call this script once per task with their own CUDA_VISIBLE_DEVICES / SLURM_TMPDIR.
# ACTIVATIONS are stored by default (11-04) under TICA_ACTIVATIONS_DIR (default /project/.../activations, mirroring the
# --out path below runs/); pass --activations false to skip them.
#SBATCH --account=ctb-liyue_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --job-name=probe-panel
#SBATCH --output=/home/sneheel/chameleons/logs/%x-%j.out
set -euo pipefail
WT=${TICA_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)}
[ -f "$WT/scripts/tica_probe_panel.py" ] || WT=/home/sneheel/chameleons
CFG=$1; OUT=$2; shift 2
PY=${PY:-/home/sneheel/chameleons-qwen35/.venv/bin/python}
export TICA_ACTIVATIONS_DIR=${TICA_ACTIVATIONS_DIR-/project/aip-rrabba/sneheel/activations}
export TRITON_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/triton
source $WT/scripts/slurm/_local_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SNAP=$(cd $WT && PYTHONPATH=$WT/src $PY -c "
from next_chameleons.tica.config import load_run_config as l; print(l('$CFG').model.path)")
LOCAL=${SLURM_TMPDIR:-/tmp}/$(basename $(dirname $(dirname $SNAP)))
mkdir -p $LOCAL; cp -rL $SNAP/. $LOCAL/
cd $WT
env PYTHONPATH=$WT/src $PY scripts/tica_probe_panel.py --config "$CFG" --model-path "$LOCAL" --out "$OUT" "$@"
