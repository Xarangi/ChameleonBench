#!/bin/bash
# Per-checkpoint eval for Phase 1 (adaptive.eval_on_save; REPLICATION 2026-10-13b). Submits two frozen
# races on the checkpoint: real verdicts and FALSIFIED verdicts (verdict use = falsify does worse).
# Both: answer format (from the config), cumulative (no circularity, as in training), strong forks.
#   bash tica_ckpt_eval.sh CONFIG ADAPTER STEP
set -euo pipefail
CFG=$1; ADAPTER=$2; STEP=$3
OUT=$(dirname "$ADAPTER")/race
mkdir -p "$OUT"
COMMON="--adapter $ADAPTER --cumulative --episodes 16 --turns 24 --fork-pos ${FORK_POS:-96} --fork-other ${FORK_OTHER:-96} --fork-free ${FORK_FREE:-48} --fork-np 24 --incremental"
cd /scratch/s/sneheel/ChameleonBench
export PY=${PY:-/scratch/s/sneheel/venvs/q35t29/bin/python}  # fast env (10-13c), inherited by sbatch
sbatch --parsable --job-name=race-${CFG}-s${STEP} scripts/slurm/tica_frozen_race.sh "$CFG" "$OUT/step_${STEP}.json" $COMMON
sbatch --parsable --job-name=racef-${CFG}-s${STEP} scripts/slurm/tica_frozen_race.sh "$CFG" "$OUT/step_${STEP}_falsify.json" $COMMON --falsify
