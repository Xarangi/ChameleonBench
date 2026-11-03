#!/bin/bash
# Topic-switch race (scripts/tica_switch_eval.py). Usage: sbatch tica_switch_eval.sh CONFIG OUT [args...]
#SBATCH --account=ctb-liyue_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --job-name=switch-eval
#SBATCH --output=/scratch/s/sneheel/ChameleonBench/logs/%x-%j.out
set -euo pipefail
CFG=$1; OUT=$2; shift 2
# TICA_REPO: the checkout to run (default the main one; race_node.sh sets its own, e.g. a worktree)
REPO=${TICA_REPO:-/scratch/s/sneheel/ChameleonBench}
PY=${PY:-/scratch/s/sneheel/venvs/q35t29/bin/python}
export TRITON_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/triton
source "$REPO/scripts/slurm/_local_env.sh"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SNAP=$(cd "$REPO" && PYTHONPATH=src $PY -c "
from next_chameleons.tica.config import load_run_config as l; print(l('$CFG').model.path)")
LOCAL=${SLURM_TMPDIR:-/tmp}/$(basename $(dirname $(dirname $SNAP)))
# Node-local model copy. Some HF snapshot files are read-only (Qwen3.5-2B: model.safetensors and
# tokenizer.json are 0440) and cp keeps that mode, so a SECOND piece on the same GPU slot could not
# overwrite them ("cp: cannot create regular file ... Permission denied"; race mbpp_pass64 / 664118
# lost 5 2B pieces this way). Copy once per slot (marker written after a full copy), and replace
# rather than overwrite (--remove-destination) when a copy is redone.
mkdir -p "$LOCAL"
if [ ! -f "$LOCAL/.tica_copied" ] || [ "$(cat "$LOCAL/.tica_copied")" != "$SNAP" ]; then
  rm -f "$LOCAL/.tica_copied"
  cp -rL --remove-destination "$SNAP/." "$LOCAL/" && echo "$SNAP" > "$LOCAL/.tica_copied"
fi
cd "$REPO"
env PYTHONPATH=src $PY scripts/tica_switch_eval.py --config "$CFG" --model-path "$LOCAL" --out "$OUT" "$@"
