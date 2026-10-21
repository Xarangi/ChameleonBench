#!/bin/bash
# Usage: sbatch scripts/slurm/tica_reasoning_budget.sbatch <model dir name under hf hub> [args]
# ctb-liyue_gpu has the HIGHER fairshare factor (0.352 vs def-rrabba_gpu 0.272 on 10-10), so it
# gets better priority even though def-rrabba has no recent usage. Override with --account=.
#SBATCH --account=ctb-liyue_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/s/sneheel/ChameleonBench/logs/%x-%j.out
export TRITON_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/triton
MODEL=$1; shift
SNAP=$(ls -d /scratch/s/sneheel/hf/hub/models--Qwen--$MODEL/snapshots/*/ | head -1)
LOCAL=${SLURM_TMPDIR:-/tmp}/$MODEL; mkdir -p $LOCAL; cp -rL $SNAP/. $LOCAL/
cd /scratch/s/sneheel/ChameleonBench
env PYTHONPATH=src /scratch/s/sneheel/venvs/q35t29/bin/python \
  scripts/tica_reasoning_budget.py $LOCAL "$@"
