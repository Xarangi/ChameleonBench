#!/bin/bash
# Usage: sbatch scripts/slurm/qwen35_fla_check.sbatch <venv python> [--bwd]
# ctb-liyue_gpu has the HIGHER fairshare factor (0.352 vs def-rrabba_gpu 0.272 on 10-10), so it
# gets better priority even though def-rrabba has no recent usage. Override with --account=.
#SBATCH --account=ctb-liyue_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --job-name=qwen35-fla-check
#SBATCH --output=/scratch/s/sneheel/ChameleonBench/logs/%x-%j.out
PY=$1; shift
export TRITON_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/triton  # home is at quota (10-10)
SNAP=/scratch/s/sneheel/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
LOCAL=${SLURM_TMPDIR:-/tmp}/qwen35_4b
t0=$(date +%s); mkdir -p $LOCAL; cp -rL $SNAP/. $LOCAL/; echo "model copy: $(( $(date +%s) - t0 ))s"
cd /scratch/s/sneheel/ChameleonBench
$PY scripts/qwen35_fla_check.py $LOCAL "$@"
