#!/bin/bash
# Readout-form check (eval only). Usage: sbatch tica_readout_form.sh CONFIG ADAPTER OUT
#SBATCH --account=ctb-liyue_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --job-name=tica-readout
#SBATCH --output=/home/sneheel/chameleons/logs/%x-%j.out
set -euo pipefail
CFG=$1; ADAPTER=$2; OUT=$3
PY=/home/sneheel/chameleons-qwen35/.venv/bin/python
export TRITON_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/triton
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SNAP=$(cd /home/sneheel/chameleons && PYTHONPATH=src $PY -c "
from next_chameleons.tica.config import load_run_config as l; print(l('$CFG').model.path)")
LOCAL=${SLURM_TMPDIR:-/tmp}/$(basename $(dirname $(dirname $SNAP)))
mkdir -p $LOCAL; cp -rL $SNAP/. $LOCAL/
cd /home/sneheel/chameleons
env PYTHONPATH=src $PY scripts/tica_readout_form.py --config "$CFG" --adapter "$ADAPTER" --model-path "$LOCAL" --out "$OUT" ${EXTRA:-}
