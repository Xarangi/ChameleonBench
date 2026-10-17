#!/bin/bash
#SBATCH --account=ctb-liyue_gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --job-name=judge-agree
#SBATCH --output=/home/sneheel/chameleons/logs/%x-%j.out
# Negative-label audit (REPLICATION 2026-10-16f): LLM judge over every concept / negative text.
# Usage: sbatch label_concept_contamination.sh OUT.json [extra args]
set -uo pipefail
OUT=$1; shift
SNAP=/scratch/sneheel/hf/hub/models--Qwen--Qwen3.5-27B-FP8/snapshots/97f5941bf617e31c5e237364a8602ce3f03a551a
LOCAL=$SLURM_TMPDIR/models--Qwen--Qwen3.5-27B-FP8; mkdir -p $LOCAL; cp -rL $SNAP/. $LOCAL/
T=/scratch/sneheel/next_chameleons_raw_cache/tica
cd /home/sneheel/chameleons
module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.9 python/3.12 opencv/4.13.0 scipy-stack/2026a
tar -xf /scratch/sneheel/venvs/vllm-serving.tar -C $SLURM_TMPDIR
$SLURM_TMPDIR/vllm-serving/bin/python scripts/judge_agreement_v3.py --model $LOCAL \
  --data $T/final_v3_eval.json --labels $T/concept_labels_v3e.json --out "$OUT" --tp 2 "$@" 2>&1 \
  | grep -vE "it/s\]|^\s*$" | tail -60
