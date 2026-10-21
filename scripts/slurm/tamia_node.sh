#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --job-name=v4b
#SBATCH --output=/scratch/s/sneheel/ChameleonBench/logs/%x-%j.out
# Production whole-node runner (TamIA 10-21; scripts/tamia_v4b.sh launch). Like tica_node.sh, but a slot names its
# GPUs, so two runs can SHARE GPUs (interleaved layout, 2 processes per GPU):
#   CFG@GPUS          train CFG from scratch on GPUS (data-parallel over them; TICA_DP = number of GPUs)
#   resume:CFG@GPUS   continue CFG's newest checkpoint (tica_resume.sh; a finished run exits at once)
#   sbatch scripts/slurm/tamia_node.sh final_v4b_tamia_4b@0,1,2,3 final_v4b_score_tamia_4b@0,1,2,3
# Held-out races are queued if TICA_VAL_QUEUE is set (tica_val_local.sh) and run later by tica_val_drain.sh.
set -uo pipefail
cd /scratch/s/sneheel/ChameleonBench
NODE_TMP=${SLURM_TMPDIR:-/tmp/tamia_node_$$}
JOB=${SLURM_JOB_NAME:-v4b}-${SLURM_JOB_ID:-local}
mkdir -p "$NODE_TMP" logs
nvidia-smi --query-gpu=index,name,memory.total --format=csv
PIDS=()
i=0
for slot in "$@"; do
  spec=${slot%@*}; gpus=${slot##*@}
  n=$(awk -F, '{print NF}' <<< "$gpus")
  case "$spec" in
    resume:*) CMD=(bash scripts/slurm/tica_resume.sh "${spec#resume:}") ;;
    *) CMD=(bash scripts/slurm/tica_l3covert.sh "$spec") ;;
  esac
  LOG=logs/$JOB-slot$i-${spec//[:\/]/_}.out
  mkdir -p "$NODE_TMP/slot$i"
  echo "slot $i ($spec, GPUs $gpus, TICA_DP=$n) -> $LOG"
  CUDA_VISIBLE_DEVICES=$gpus TICA_DP=$n MASTER_PORT=$(( 29500 + 10 * i )) SLURM_TMPDIR=$NODE_TMP/slot$i \
    "${CMD[@]}" > "$LOG" 2>&1 &
  PIDS+=($!)
  i=$(( i + 1 ))
done
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv -l 300 > "logs/$JOB-gpu.csv" 2>/dev/null &
SMI=$!
rc=0
for p in "${PIDS[@]}"; do wait "$p" || rc=1; done
kill $SMI 2>/dev/null
exit $rc
