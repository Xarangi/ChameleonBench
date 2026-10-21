#!/bin/bash
# Clean-data fit check of every probe-zoo kind. Usage: sbatch check_zoo_fit.sh CONFIG OUT [args...]
#SBATCH --account=ctb-liyue_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --job-name=zoo-check
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
env PYTHONPATH=src $PY scripts/check_zoo_fit.py --config "$CFG" --model-path "$LOCAL" --out "$OUT" "$@"
