#!/bin/bash
#SBATCH --account=ctb-liyue_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --job-name=bench-fit
#SBATCH --output=/scratch/s/sneheel/ChameleonBench/logs/%x-%j.out
set -uo pipefail
cd /scratch/s/sneheel/ChameleonBench
source scripts/slurm/_local_env.sh
export TRITON_CACHE_DIR=$SLURM_TMPDIR/triton
env PYTHONPATH=src $PY scripts/bench/bench_gate_fit.py "$@" 2>&1 | grep -vE "Warning|warn"
