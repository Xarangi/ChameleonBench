#!/bin/bash
# Math solve-while-hiding pilot. Usage: sbatch tica_mathrl.sh {train|eval} CONFIG [args...]
#SBATCH --account=ctb-liyue_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --job-name=mathrl
#SBATCH --output=/scratch/s/sneheel/ChameleonBench/logs/%x-%j.out
set -euo pipefail
CMD=$1; CFG=$2; shift 2
PY=${PY:-/scratch/s/sneheel/venvs/q35t29/bin/python}
export TRITON_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/triton
source /scratch/s/sneheel/ChameleonBench/scripts/slurm/_local_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SNAP=$(cd /scratch/s/sneheel/ChameleonBench && PYTHONPATH=src $PY -c "
from next_chameleons.config import find_config_path, load_yaml
print(load_yaml(find_config_path('tica', '$CFG'))['model']['path'])")
LOCAL=${SLURM_TMPDIR:-/tmp}/$(basename $(dirname $(dirname $SNAP)))
mkdir -p $LOCAL; cp -rL $SNAP/. $LOCAL/
cd /scratch/s/sneheel/ChameleonBench
env PYTHONPATH=src $PY scripts/tica_mathrl.py "$CMD" --config "$CFG" --model-path "$LOCAL" "$@"
