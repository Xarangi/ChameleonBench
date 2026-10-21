#!/bin/bash
# Readout-form check (eval only). Usage: sbatch tica_readout_form.sh CONFIG ADAPTER OUT
#SBATCH --account=ctb-liyue_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --job-name=tica-readout
#SBATCH --output=/scratch/s/sneheel/ChameleonBench/logs/%x-%j.out
set -euo pipefail
CFG=$1; ADAPTER=$2; OUT=$3
PY=/scratch/s/sneheel/venvs/q35t29/bin/python
export TRITON_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/triton
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SNAP=$(cd /scratch/s/sneheel/ChameleonBench && PYTHONPATH=src $PY -c "
from next_chameleons.tica.config import load_run_config as l; print(l('$CFG').model.path)")
LOCAL=${SLURM_TMPDIR:-/tmp}/$(basename $(dirname $(dirname $SNAP)))
mkdir -p $LOCAL; cp -rL $SNAP/. $LOCAL/
cd /scratch/s/sneheel/ChameleonBench
env PYTHONPATH=src $PY scripts/tica_readout_form.py --config "$CFG" --adapter "$ADAPTER" --model-path "$LOCAL" --out "$OUT" ${EXTRA:-}
