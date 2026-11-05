#!/bin/bash
# Topic-switch race (scripts/tica_switch_eval.py). Usage: sbatch tica_switch_eval.sh CONFIG OUT [args...]
#SBATCH --account=ctb-liyue_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --job-name=switch-eval
#SBATCH --output=/home/sneheel/chameleons/logs/%x-%j.out
set -euo pipefail
CFG=$1; OUT=$2; shift 2
PY=${PY:-/home/sneheel/chameleons-qwen35/.venv/bin/python}
export TRITON_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/triton
source /home/sneheel/chameleons/scripts/slurm/_local_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SNAP=$(cd /home/sneheel/chameleons && PYTHONPATH=src $PY -c "
from next_chameleons.tica.config import load_run_config as l; print(l('$CFG').model.path)")
LOCAL=${SLURM_TMPDIR:-/tmp}/$(basename $(dirname $(dirname $SNAP)))
mkdir -p $LOCAL; cp -rL $SNAP/. $LOCAL/
cd /home/sneheel/chameleons
env PYTHONPATH=src $PY scripts/tica_switch_eval.py --config "$CFG" --model-path "$LOCAL" --out "$OUT" "$@"
