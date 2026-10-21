#!/bin/bash
#SBATCH --account=ctb-liyue_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=chk-group
#SBATCH --output=/scratch/s/sneheel/ChameleonBench/logs/%x-%j.out
set -uo pipefail
SNAP=/scratch/s/sneheel/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
LOCAL=$SLURM_TMPDIR/models--Qwen--Qwen3.5-4B; mkdir -p $LOCAL; cp -rL $SNAP/. $LOCAL/
export TRITON_CACHE_DIR=$SLURM_TMPDIR/triton
cd /scratch/s/sneheel/ChameleonBench
PY=/scratch/s/sneheel/venvs/q35t29/bin/python
source scripts/slurm/_local_env.sh
env PYTHONPATH=src $PY scripts/bench/check_group_fwd.py --model $LOCAL "$@"
