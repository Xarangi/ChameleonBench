#!/bin/bash
# Zero-shot CONCEPT transfer (PAPER-PLAN rung 3e): a model trained on one concept, evaluated
# unchanged on another. Also re-runs the training concept so the two are directly comparable with
# the current code (det_abs / contains bars).
#   sbatch [--job-name=..] scripts/slurm/tica_l3_transfer.sh <config> [seed ...]
# ctb-liyue_gpu has the HIGHER fairshare factor (0.352 vs def-rrabba_gpu 0.272 on 10-10), so it
# gets better priority even though def-rrabba has no recent usage. Override with --account=.
#SBATCH --account=ctb-liyue_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/sneheel/chameleons/logs/%x-%j.out
set -uo pipefail
PY=${PY:-/home/sneheel/chameleons-qwen35/.venv/bin/python}
export TRITON_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/triton
CFG=$1; shift
SEEDS=${*:-0 1 2}
cd /home/sneheel/chameleons
SNAP=$(PYTHONPATH=src $PY -c "
from next_chameleons.tica.config import load_window_config as l; print(l('$CFG').model.path)")
OUT=$(PYTHONPATH=src $PY -c "
from next_chameleons.tica.config import load_window_config as l; print(l('$CFG').out)")
LOCAL=${SLURM_TMPDIR:-/tmp}/$(basename $(dirname $(dirname $SNAP)))
mkdir -p $LOCAL; cp -rL $SNAP/. $LOCAL/
# Held-out CONCEPT (never trained on): the formatting contrast from the 08-24 races.
HTML="HTML tags and formatting elements in the text"
for s in $SEEDS; do
  AD=$OUT/seed$s/train/final
  if [ ! -d "$AD" ]; then echo "seed $s: no adapter at $AD, skipping"; continue; fi
  for pair in "self:" "html:$HTML"; do
    tag=${pair%%:*}; concept=${pair#*:}
    echo "=== seed $s / $tag"
    if [ -z "$concept" ]; then
      env PYTHONPATH=src $PY scripts/tica.py windows --config $CFG --stage eval \
        --adapter $AD --out $OUT/seed$s/transfer_$tag --model-path $LOCAL
    else
      env PYTHONPATH=src $PY scripts/tica.py windows --config $CFG --stage eval \
        --adapter $AD --concept "$concept" --out $OUT/seed$s/transfer_$tag --model-path $LOCAL
    fi
  done
done
