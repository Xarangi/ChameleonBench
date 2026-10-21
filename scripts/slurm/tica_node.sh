#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --job-name=tica-node
#SBATCH --output=/scratch/s/sneheel/ChameleonBench/logs/%x-%j.out
# WHOLE-NODE launcher (REPLICATION 2026-10-17f; docs/tica-other-hpc.md). Clusters that allocate whole
# 4-GPU nodes (TamIA) waste 3 GPUs on a single-GPU job, so one job runs up to 4 SLOTS, one per GPU:
#   CFG          train CFG from scratch     (scripts/slurm/tica_l3covert.sh)
#   resume:CFG   continue CFG's newest ckpt (scripts/slurm/tica_resume.sh)
#   val          keep the GPU free for eval_on_save races (scripts/slurm/tica_val_local.sh)
#   -            leave the GPU idle
#   +            this GPU joins the training slot before it (data-parallel episodes, TICA_DP)
#   sbatch --account=<acct> -J name scripts/slurm/tica_node.sh final_v3_4b val val -
#   sbatch --account=<acct> -J name scripts/slurm/tica_node.sh final_v4_4b + val val   # 2-GPU run
# Every slot gets its own node-local tmp dir (model copy, venv, Triton cache) and its own log.
set -uo pipefail
cd /scratch/s/sneheel/ChameleonBench
(( $# >= 1 && $# <= 4 )) || { echo "usage: tica_node.sh SLOT [SLOT SLOT SLOT]"; exit 2; }
NODE_TMP=${SLURM_TMPDIR:-/tmp/tica_node_$$}
mkdir -p "$NODE_TMP"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
# eval_on_save strips SLURM_* from its environment, so the validation launcher gets the free GPUs and
# the node-local tmp through TICA_* variables instead.
VAL=()
i=0
for s in "$@"; do
  [ "$s" = "val" ] && VAL+=("$i")
  i=$(( i + 1 ))
done
export TICA_VAL_GPUS="${VAL[*]:-}" TICA_NODE_TMP="$NODE_TMP"
echo "validation GPUs: ${TICA_VAL_GPUS:-none}"
PIDS=()
SLOTS=("$@")
for (( i = 0; i < ${#SLOTS[@]}; i++ )); do
  s=${SLOTS[$i]}
  case "$s" in
    -|val|+) continue ;;
    resume:*) CMD=(bash scripts/slurm/tica_resume.sh "${s#resume:}") ;;
    *) CMD=(bash scripts/slurm/tica_l3covert.sh "$s") ;;
  esac
  G=$i  # this slot's GPU plus the "+" slots right after it
  for (( j = i + 1; j < ${#SLOTS[@]}; j++ )); do
    [ "${SLOTS[$j]}" = "+" ] || break
    G="$G,$j"
  done
  LOG=logs/${SLURM_JOB_NAME:-tica-node}-${SLURM_JOB_ID:-local}-gpu${G//,/+}-${s//[:\/]/_}.out
  mkdir -p "$NODE_TMP/slot$i"
  echo "slot $i (GPUs $G): ${CMD[*]} -> $LOG"
  CUDA_VISIBLE_DEVICES=$G TICA_DP=$(( j - i )) MASTER_PORT=$(( 29500 + i )) \
    SLURM_TMPDIR=$NODE_TMP/slot$i "${CMD[@]}" > "$LOG" 2>&1 &
  PIDS+=($!)
done
rc=0
for p in "${PIDS[@]}"; do wait "$p" || rc=1; done
# validation races launched by eval_on_save near the end may still be running
while pgrep -f "tica_val_local_run" > /dev/null; do sleep 30; done
exit $rc
