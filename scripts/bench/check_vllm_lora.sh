#!/bin/bash
#SBATCH --account=ctb-liyue_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=chk-vllm-lora
#SBATCH --output=/scratch/s/sneheel/ChameleonBench/logs/%x-%j.out
set -uo pipefail
SNAP=/scratch/s/sneheel/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
AD=/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/l5dodge_math_4b/from_l4s50/adaptive/final
LOCAL=$SLURM_TMPDIR/models--Qwen--Qwen3.5-4B; mkdir -p $LOCAL; cp -rL $SNAP/. $LOCAL/
cd /scratch/s/sneheel/ChameleonBench
( PY=/scratch/s/sneheel/venvs/q35t29/bin/python; source scripts/slurm/_local_env.sh
  env PYTHONPATH=src $PY scripts/bench/check_vllm_lora.py --mode hf --model $LOCAL --adapter $AD --io $SLURM_TMPDIR/io.json )
module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.9 python/3.12 opencv/4.13.0 scipy-stack/2026a
tar -xf /scratch/s/sneheel/venvs/vllm-serving.tar -C $SLURM_TMPDIR
VPY=$SLURM_TMPDIR/vllm-serving/bin/python
$VPY scripts/bench/convert_adapter_vllm.py $AD $SLURM_TMPDIR/ad_vllm
$VPY scripts/bench/check_vllm_lora.py --mode vllm --model $LOCAL --adapter $SLURM_TMPDIR/ad_vllm --io $SLURM_TMPDIR/io.json 2>&1 | grep -vE "it/s\]|^\s*$" | tail -60
