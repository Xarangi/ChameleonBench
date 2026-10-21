#!/bin/bash
# Probe-based topic overlap. Usage: sbatch tica_topic_overlap.sh CONFIG TOPICS OUT [args...]
#SBATCH --account=ctb-liyue_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --job-name=topic-ovl
#SBATCH --output=/scratch/s/sneheel/ChameleonBench/logs/%x-%j.out
set -euo pipefail
CFG=$1; SPLIT=$2; OUT=$3; shift 3
PY=${PY:-/scratch/s/sneheel/venvs/q35t29/bin/python}
export TRITON_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/triton
source /scratch/s/sneheel/ChameleonBench/scripts/slurm/_local_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SNAP=$(cd /scratch/s/sneheel/ChameleonBench && PYTHONPATH=src $PY -c "
from next_chameleons.tica.config import load_run_config as l; print(l('$CFG').model.path)")
LOCAL=${SLURM_TMPDIR:-/tmp}/$(basename $(dirname $(dirname $SNAP)))
mkdir -p $LOCAL; cp -rL $SNAP/. $LOCAL/
cd /scratch/s/sneheel/ChameleonBench
env PYTHONPATH=src $PY scripts/tica_topic_overlap.py --config "$CFG" --topics "$SPLIT" --model-path "$LOCAL" --out "$OUT" "$@"
