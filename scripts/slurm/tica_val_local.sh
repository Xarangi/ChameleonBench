#!/bin/bash
# eval_on_save on a WHOLE-NODE job (REPLICATION 2026-10-17f): run the validation race on a free GPU of
# the SAME node (tica_node.sh "val" slots) instead of sbatch-ing a job (which would take a whole node).
# Returns immediately (eval_on_save waits <= 120 s); the race waits for a free GPU (one race per GPU).
#   eval_on_save: "bash scripts/slurm/tica_val_local.sh CONFIG {adapter} {step} VALDIR SPLIT"
# Falls back to sbatch scripts/slurm/tica_val.sh when the job has no "val" slot.
set -uo pipefail
cd /scratch/s/sneheel/ChameleonBench
# TICA_VAL_QUEUE=FILE (TamIA 10-21, default unset = unchanged): append the race to FILE and return; a later
# whole-node job (scripts/slurm/tica_val_drain.sh) runs the queued races 4 at a time. Training nodes then keep
# all 4 GPUs on training (races are logged only, so running them later changes nothing).
if [ -n "${TICA_VAL_QUEUE:-}" ]; then
  mkdir -p "$(dirname "$TICA_VAL_QUEUE")"
  ( flock 9; printf '%q ' "$@" >> "$TICA_VAL_QUEUE"; echo >> "$TICA_VAL_QUEUE" ) 9>"$TICA_VAL_QUEUE.lock"
  echo "validation race for step $3 queued in $TICA_VAL_QUEUE"
  exit 0
fi
if [ -z "${TICA_VAL_GPUS:-}" ]; then
  exec sbatch --parsable scripts/slurm/tica_val.sh "$@"
fi
STEP=$3
mkdir -p logs
nohup setsid bash -c '
  # tica_val_local_run (tica_node.sh waits for this marker before exiting)
  while true; do
    for g in $TICA_VAL_GPUS; do
      exec 9>"$TICA_NODE_TMP/val_gpu$g.lock"
      if flock -n 9; then
        mkdir -p "$TICA_NODE_TMP/val_$0"
        CUDA_VISIBLE_DEVICES=$g SLURM_TMPDIR=$TICA_NODE_TMP/val_$0 \
          bash scripts/slurm/tica_val.sh "$@"
        rm -rf "$TICA_NODE_TMP/val_$0"
        exit
      fi
      exec 9>&-
    done
    sleep 20
  done' "$STEP" "$@" > "logs/val_local-${STEP}-$$.out" 2>&1 < /dev/null &
echo "local validation race for step $STEP queued on GPUs [$TICA_VAL_GPUS] (pid $!)"
