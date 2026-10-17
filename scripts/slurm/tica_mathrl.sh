#!/bin/bash
# Math solve-while-hiding pilot. Usage: sbatch tica_mathrl.sh {train|eval} CONFIG [args...]
#SBATCH --account=ctb-liyue_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --job-name=mathrl
#SBATCH --output=/home/sneheel/chameleons/logs/%x-%j.out
set -euo pipefail
CMD=$1; CFG=$2; shift 2
PY=${PY:-/home/sneheel/chameleons-qwen35/.venv/bin/python}
export TRITON_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/triton
source /home/sneheel/chameleons/scripts/slurm/_local_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SNAP=$(cd /home/sneheel/chameleons && PYTHONPATH=src $PY -c "
from next_chameleons.config import find_config_path, load_yaml
print(load_yaml(find_config_path('tica', '$CFG'))['model']['path'])")
LOCAL=${SLURM_TMPDIR:-/tmp}/$(basename $(dirname $(dirname $SNAP)))
mkdir -p $LOCAL; cp -rL $SNAP/. $LOCAL/
cd /home/sneheel/chameleons
env PYTHONPATH=src $PY scripts/tica_mathrl.py "$CMD" --config "$CFG" --model-path "$LOCAL" "$@"
