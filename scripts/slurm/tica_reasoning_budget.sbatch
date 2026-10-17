#!/bin/bash
# Usage: sbatch scripts/slurm/tica_reasoning_budget.sbatch <model dir name under hf hub> [args]
# ctb-liyue_gpu has the HIGHER fairshare factor (0.352 vs def-rrabba_gpu 0.272 on 10-10), so it
# gets better priority even though def-rrabba has no recent usage. Override with --account=.
#SBATCH --account=ctb-liyue_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/sneheel/chameleons/logs/%x-%j.out
export TRITON_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/triton
MODEL=$1; shift
SNAP=$(ls -d /scratch/sneheel/hf/hub/models--Qwen--$MODEL/snapshots/*/ | head -1)
LOCAL=${SLURM_TMPDIR:-/tmp}/$MODEL; mkdir -p $LOCAL; cp -rL $SNAP/. $LOCAL/
cd /home/sneheel/chameleons
env PYTHONPATH=src /home/sneheel/chameleons-qwen35/.venv/bin/python \
  scripts/tica_reasoning_budget.py $LOCAL "$@"
