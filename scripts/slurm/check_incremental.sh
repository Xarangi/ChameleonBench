#!/bin/bash
# Usage: sbatch check_incremental.sh [PYTHON]  (default: the qwen35 venv)
#SBATCH --account=ctb-liyue_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --job-name=check-incremental
#SBATCH --output=/scratch/s/sneheel/ChameleonBench/logs/%x-%j.out
set -euo pipefail
PY=${1:-/scratch/s/sneheel/venvs/q35t29/bin/python}
export TRITON_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/triton
SNAP=/scratch/s/sneheel/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
LOCAL=${SLURM_TMPDIR:-/tmp}/qwen35_4b; mkdir -p $LOCAL; cp -rL $SNAP/. $LOCAL/
cd /scratch/s/sneheel/ChameleonBench
PYTHONPATH=src $PY scripts/check_incremental_rollout.py --config l4dodge_math_4b --model-path $LOCAL
