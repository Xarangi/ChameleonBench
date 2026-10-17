#!/bin/bash
#SBATCH --account=ctb-liyue_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --job-name=bench-gen
#SBATCH --output=/home/sneheel/chameleons/logs/%x-%j.out
set -uo pipefail
SNAP=/scratch/sneheel/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
AD=/scratch/sneheel/next_chameleons_artifacts/tica/runs/l5dodge_math_4b/from_l4s50/adaptive/final
LOCAL=$SLURM_TMPDIR/models--Qwen--Qwen3.5-4B; mkdir -p $LOCAL; cp -rL $SNAP/. $LOCAL/
export TRITON_CACHE_DIR=$SLURM_TMPDIR/triton
cd /home/sneheel/chameleons
PY=/scratch/sneheel/venvs/q35t29/bin/python
source scripts/slurm/_local_env.sh
env PYTHONPATH=src $PY scripts/bench/bench_mathrl_gen.py --model $LOCAL --adapter $AD --save-prompts $SLURM_TMPDIR/p.json
echo "=== vllm"
tar -xf /scratch/sneheel/venvs/vllm-serving.tar -C $SLURM_TMPDIR 2>/dev/null && VPY=$SLURM_TMPDIR/vllm-serving/bin/python || VPY=/scratch/sneheel/venvs/vllm-serving/bin/python
echo "vllm python: $VPY"
$VPY scripts/bench/bench_vllm_gen.py --model $LOCAL --adapter $AD --prompts $SLURM_TMPDIR/p.json --mem 0.6 2>&1 | grep -vE "^\s*$|it/s\]" | tail -40
