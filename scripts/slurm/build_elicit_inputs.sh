#!/bin/bash
#SBATCH --account=ctb-liyue_gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --job-name=elicit-inputs
#SBATCH --output=/home/sneheel/chameleons/logs/%x-%j.out
# Concept-eliciting requests written + verified by Qwen3.5-27B-FP8 (scripts/build_elicit_inputs.py,
# REPLICATION 10-19e). Usage: sbatch build_elicit_inputs.sh --set DATA,INPUTS,OUT [--set ...]
set -uo pipefail
SNAP=/scratch/sneheel/hf/hub/models--Qwen--Qwen3.5-27B-FP8/snapshots/97f5941bf617e31c5e237364a8602ce3f03a551a
LOCAL=$SLURM_TMPDIR/models--Qwen--Qwen3.5-27B-FP8; mkdir -p $LOCAL; cp -rL $SNAP/. $LOCAL/
cd /home/sneheel/chameleons
module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.9 python/3.12 opencv/4.13.0 scipy-stack/2026a
tar -xf /scratch/sneheel/venvs/vllm-serving.tar -C $SLURM_TMPDIR
$SLURM_TMPDIR/vllm-serving/bin/python scripts/build_elicit_inputs.py --llm $LOCAL --tp 2 "$@" 2>&1 \
  | grep -vE "it/s\]|^\s*$" | tail -150
