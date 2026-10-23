#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=8:00:00
#SBATCH --job-name=panel-pre
#SBATCH --output=/home/sneheel/chameleons/logs/%x-%j.out
# Fill adaptive.first_probe_panel_cache for CFG before training (TamIA 10-21): every turn-0 panel probe (train
# concept x train-mix condition x panel kind, ~1840 for the v4b configs) fit on the BASE model through the trainer's
# own path (scripts/tica.py adaptive --panel-precompute I/N). PROCS processes (default 2 per allocated GPU; each is
# host-bound), shard I of N each. Re-running only fits the missing keys.
#   sbatch --account=aip-<pi> scripts/slurm/tamia_panel_precompute.sh final_v4b_tamia_4b
# Any GPU count (10-23, Narval): processes go round-robin over the allocated GPUs, e.g.
#   sbatch --gres=gpu:a100:2 --cpus-per-task=24 --mem=128G scripts/slurm/tamia_panel_precompute.sh final_v4c_4b
set -uo pipefail
cd /home/sneheel/chameleons
CFG=${1:?config}
IFS=, read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
NG=${#GPUS[@]}
N=${PROCS:-$(( 2 * NG ))}
NODE_TMP=${SLURM_TMPDIR:-/tmp/panel_pre_$$}
OUT=/scratch/sneheel/next_chameleons_artifacts/tica/runs/panel_precompute/$CFG
mkdir -p logs "$OUT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
PIDS=()
for (( i = 0; i < N; i++ )); do
  g=${GPUS[$(( i % NG ))]}
  mkdir -p "$NODE_TMP/p$i"
  CUDA_VISIBLE_DEVICES=$g TICA_DP=1 SLURM_TMPDIR=$NODE_TMP/p$i \
    bash scripts/slurm/tica_l3covert.sh "$CFG" --panel-precompute "$i/$N" --out "$OUT/shard$i" \
    > "logs/${SLURM_JOB_NAME:-panel-pre}-${SLURM_JOB_ID:-local}-p$i.out" 2>&1 &
  PIDS+=($!)
done
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv -l 60 \
  > "logs/${SLURM_JOB_NAME:-panel-pre}-${SLURM_JOB_ID:-local}-gpu.csv" 2>/dev/null &
SMI=$!
rc=0
for p in "${PIDS[@]}"; do wait "$p" || rc=1; done
kill $SMI 2>/dev/null
ls /scratch/sneheel/next_chameleons_artifacts/tica/panel_probe_cache | wc -l
exit $rc
