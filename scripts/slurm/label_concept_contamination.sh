#!/bin/bash
#SBATCH --account=ctb-liyue_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --job-name=label-negs
#SBATCH --output=/scratch/s/sneheel/ChameleonBench/logs/%x-%j.out
# Negative-label audit (REPLICATION 2026-10-16f): LLM judge over every concept / negative text.
# Usage: sbatch label_concept_contamination.sh OUT.json [extra args]
set -uo pipefail
OUT=$1; shift
SNAP=/scratch/s/sneheel/hf/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
LOCAL=$SLURM_TMPDIR/models--Qwen--Qwen3.5-9B; mkdir -p $LOCAL; cp -rL $SNAP/. $LOCAL/
T=/scratch/s/sneheel/next_chameleons_raw_cache/tica
cd /scratch/s/sneheel/ChameleonBench
module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.9 python/3.12 opencv/4.13.0 scipy-stack/2026a
tar -xf /scratch/s/sneheel/venvs/vllm-serving.tar -C $SLURM_TMPDIR
$SLURM_TMPDIR/vllm-serving/bin/python scripts/label_concept_contamination.py --model $LOCAL \
  --data $T/final_base_v2.json $T/final_ood_v1.json $T/final_ood_v2.json --out "$OUT" "$@" 2>&1 \
  | grep -vE "it/s\]|^\s*$" | tail -60
