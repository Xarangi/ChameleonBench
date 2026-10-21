#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=2:30:00
#SBATCH --job-name=v4b-layout
#SBATCH --output=/scratch/s/sneheel/ChameleonBench/logs/%x-%j.out
# Layout smoke (TamIA 10-21): run SLOTS concurrently on one node for SMOKE_STEPS steps and trace the GPUs, to pick
# the production layout by measurement (docs/tica/TAMIA-V4B.md "Node layout"). A slot is CFG@GPUS[@EXTRA], e.g.
#   final_v4b_tamia_4b@0,1,2,3                         one arm, data-parallel over the 4 GPUs
#   final_v4b_tamia_4b@0,1,2,3 final_v4b_score_tamia_4b@0,1,2,3   both arms interleaved (2 processes per GPU)
#   final_v4b_tamia_4b@0,1,2,3@--set~model.batch_size=32          a lever (EXTRA: tica.py args, "~" = space)
# Output: runs/smoke_tamia/<tag>/<cfg>/adaptive/train_log.jsonl, logs/<job>-<cfg>.out, logs/<job>-gpu.csv.
set -uo pipefail
cd /scratch/s/sneheel/ChameleonBench
STEPS=${SMOKE_STEPS:-20}
TAG=${SMOKE_TAG:-layout_${SLURM_JOB_ID:-local}}
OUT=/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/smoke_tamia/$TAG
NODE_TMP=${SLURM_TMPDIR:-/tmp/v4b_layout_$$}
JOB=${SLURM_JOB_NAME:-layout}-${SLURM_JOB_ID:-local}
nvidia-smi --query-gpu=index,name,memory.total --format=csv
mkdir -p logs
# Seed each config's anchor cache (KL-anchor answers of the BASE model; same anchor settings in every v4b config)
# from final_v4b_4b's, so concurrent jobs do not generate (and non-atomically write) the same file. Waits <= 60 min
# for the source; without it the run generates its own as usual.
RUNS=/scratch/s/sneheel/next_chameleons_artifacts/tica/runs
SRC=$RUNS/final_v4b_4b/anchor_cache.json
for slot in "$@"; do
  cfg=${slot%%@*}; dst=$RUNS/$cfg/anchor_cache.json
  [ -f "$dst" ] && continue
  for (( w = 0; w < 60 && ! -f "$SRC"; w++ )); do sleep 60; done
  if [ -f "$SRC" ]; then
    mkdir -p "$RUNS/$cfg"; cp "$SRC" "$dst.tmp.$$" && mv -n "$dst.tmp.$$" "$dst"; rm -f "$dst.tmp.$$"
    echo "anchor cache for $cfg seeded from $SRC"
  fi
done
PIDS=()
i=0
for slot in "$@"; do
  IFS=@ read -r cfg gpus extra <<< "$slot"
  n=$(awk -F, '{print NF}' <<< "$gpus")
  EXTRA=(); [ -n "${extra:-}" ] && read -ra EXTRA <<< "${extra//\~/ }"
  mkdir -p "$NODE_TMP/slot$i"
  LOG=logs/$JOB-slot$i-$cfg.out
  echo "slot $i: $cfg on GPUs $gpus (TICA_DP=$n) ${EXTRA[*]:-} -> $LOG"
  CUDA_VISIBLE_DEVICES=$gpus TICA_DP=$n MASTER_PORT=$(( 29500 + 10 * i )) SLURM_TMPDIR=$NODE_TMP/slot$i \
    bash scripts/slurm/tica_l3covert.sh "$cfg" --steps "$STEPS" --out "$OUT/slot${i}_$cfg" "${EXTRA[@]}" \
    > "$LOG" 2>&1 &
  PIDS+=($!)
  i=$(( i + 1 ))
done
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv -l 30 > "logs/$JOB-gpu.csv" 2>/dev/null &
SMI=$!
( while sleep 60; do top -bn1 -u "$USER" | awk 'NR>7 && $9>5 {print strftime("%H:%M:%S"), $1, $9, $10}'; done ) \
  > "logs/$JOB-cpu.txt" 2>/dev/null &
TOP=$!
rc=0
for p in "${PIDS[@]}"; do wait "$p" || rc=1; done
kill $SMI $TOP 2>/dev/null
exit $rc
