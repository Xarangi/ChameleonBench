#!/bin/bash
#SBATCH --account=ctb-liyue_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/home/sneheel/chameleons/logs/%x-%j.out
# Profiling smoke (REPLICATION 2026-10-15d): a few adaptive steps with GPU utilisation / memory sampled
# every second (nvidia-smi) to logs/gpu-<job>.csv, next to the per-phase timings in the train log.
#   sbatch -J prof-mix scripts/slurm/tica_profile.sh <config> --adapter DIR --steps 2 --out DIR
set -euo pipefail
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used \
  --format=csv,nounits -l 1 > /home/sneheel/chameleons/logs/gpu-${SLURM_JOB_ID}.csv &
SMI=$!
trap 'kill $SMI 2>/dev/null || true' EXIT
bash /home/sneheel/chameleons/scripts/slurm/tica_l3covert.sh "$@"
