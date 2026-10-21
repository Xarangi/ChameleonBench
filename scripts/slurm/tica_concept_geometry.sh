#!/bin/bash
# Final dodger rung 0 (TICA-FINAL-DODGER-DESIGN.md): answer-format concept geometry and/or the
# answer-format base-probe cache for the split's concepts.
#   sbatch tica_concept_geometry.sh CONFIG OUT --geometry --split S
#   sbatch --array=0-3 tica_concept_geometry.sh CONFIG OUT --probes --split S   (shard = array id / 4)
#SBATCH --account=ctb-liyue_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=tica-geom
#SBATCH --output=/scratch/s/sneheel/ChameleonBench/logs/%x-%A_%a.out
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
SHARD=()
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then SHARD=(--shard "${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"); fi
cd /scratch/s/sneheel/ChameleonBench
env PYTHONPATH=src $PY scripts/tica_concept_geometry.py --config "$CFG" --model-path "$LOCAL" --out "$OUT" "${SHARD[@]}" "$@"
