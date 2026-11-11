#!/bin/bash
# 11-11: held-out fixed-text val race for a BASE model (no adapter). Same as scripts/slurm/tica_val.sh (same race
# script, same exact speed-ups, same scorer), except it passes --base instead of --adapter.
#   val_base.sh CONFIG VALDIR SPLIT [race args...]   -> VALDIR/step_0.json
set -uo pipefail
CFG=$1; VAL=$2; SPLIT=$3; shift 3
export TICA_TRUNC_READS=1 TICA_BATCHED_FITS=1
mkdir -p "$VAL"
bash /scratch/s/sneheel/ChameleonBench/scripts/slurm/tica_switch_eval.sh "$CFG" "$VAL/race_step_0.json" \
  --base --topics dev --kind mixed --switch-prob 0 --episodes 16 --turns 24 "$@" || exit 1
cd /scratch/s/sneheel/ChameleonBench
python3 scripts/tica_val_score.py --race "$VAL/race_step_0.json" --split "$SPLIT" --step 0 --out "$VAL/step_0.json"
