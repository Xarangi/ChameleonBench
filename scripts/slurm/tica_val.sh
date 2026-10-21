#!/bin/bash
#SBATCH --account=ctb-liyue_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --job-name=tica-val
#SBATCH --output=/scratch/s/sneheel/ChameleonBench/logs/%x-%j.out
# Early-stopping validation (REPLICATION 2026-10-16j), launched by adaptive.eval_on_save:
#   eval_on_save: "sbatch --parsable scripts/slurm/tica_val.sh CONFIG {adapter} {step} VALDIR SPLIT"
# Long single-concept race on the DEV concepts (mixed train probe kinds, 16 ep x 24 turns, exact
# speed-ups on), then scripts/tica_val_score.py -> VALDIR/step_<step>.json (early_stop_glob).
set -uo pipefail
CFG=$1; AD=$2; STEP=$3; VAL=$4; SPLIT=$5; shift 5
export TICA_TRUNC_READS=1 TICA_BATCHED_FITS=1
mkdir -p "$VAL"
bash /scratch/s/sneheel/ChameleonBench/scripts/slurm/tica_switch_eval.sh "$CFG" "$VAL/race_step_$STEP.json" \
  --adapter "$AD" --topics dev --kind mixed --switch-prob 0 --episodes 16 --turns 24 "$@"
cd /scratch/s/sneheel/ChameleonBench
python3 scripts/tica_val_score.py --race "$VAL/race_step_$STEP.json" --split "$SPLIT" \
  --step "$STEP" --out "$VAL/step_$STEP.json"
