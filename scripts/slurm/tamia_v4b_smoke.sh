#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=3:00:00
#SBATCH --job-name=v4b-smoke
#SBATCH --output=/home/sneheel/chameleons/logs/%x-%j.out
# 20-step smoke of BOTH TamIA arms in the production layout (scripts/tamia_v4b.sh smoke): final_v4b_4b on GPUs 0-1
# and final_v4b_score_4b on GPUs 2-3, data-parallel (TICA_DP=2) each, into runs/smoke_tamia/<cfg>. Measures what the
# Narval smokes could not: 24-turn memory on H100 (max_mem_gb), step time (sec), DP balance (dp/sec_rollout_*).
set -uo pipefail
cd /home/sneheel/chameleons
STEPS=${SMOKE_STEPS:-20}
OUT=/scratch/sneheel/next_chameleons_artifacts/tica/runs/smoke_tamia
NODE_TMP=${SLURM_TMPDIR:-/tmp/v4b_smoke_$$}
nvidia-smi --query-gpu=index,name,memory.total --format=csv
PIDS=()
i=0
for cfg in final_v4b_4b final_v4b_score_4b; do
  G="$(( 2 * i )),$(( 2 * i + 1 ))"
  mkdir -p "$NODE_TMP/slot$i" logs
  LOG=logs/v4b-smoke-${SLURM_JOB_ID:-local}-$cfg.out
  echo "$cfg on GPUs $G -> $LOG"
  CUDA_VISIBLE_DEVICES=$G TICA_DP=2 MASTER_PORT=$(( 29500 + 10 * i )) SLURM_TMPDIR=$NODE_TMP/slot$i \
    bash scripts/slurm/tica_l3covert.sh "$cfg" --steps "$STEPS" --out "$OUT/$cfg" > "$LOG" 2>&1 &
  PIDS+=($!)
  i=$(( i + 1 ))
done
# GPU memory trace (the per-step max_mem_gb is per process; this shows the whole device)
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv -l 60 \
  > logs/v4b-smoke-${SLURM_JOB_ID:-local}-gpu.csv 2>/dev/null &
SMI=$!
rc=0
for p in "${PIDS[@]}"; do wait "$p" || rc=1; done
kill $SMI 2>/dev/null
exit $rc
