#!/bin/bash
#SBATCH --account=ctb-liyue_gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --job-name=mw-labels
#SBATCH --output=/scratch/s/sneheel/ChameleonBench/logs/%x-%j.out
# 27B negative-contamination labels (prompt v4, as label_concepts_27b.sh) for the NEW texts of a
# model-written data file only (--skip-labelled: texts of the old label run keep their labels;
# build_final_v5_mw.py merge-labels joins them). REPLICATION 2026-10-21f.
# Usage: sbatch tica_mw_labels.sh DATA.json OLD_LABELS.json OUT.json --shard i/n
set -uo pipefail
DATA=$1; OLD=$2; OUT=$3; shift 3
SNAP=/scratch/s/sneheel/hf/hub/models--Qwen--Qwen3.5-27B-FP8/snapshots/97f5941bf617e31c5e237364a8602ce3f03a551a
LOCAL=$SLURM_TMPDIR/models--Qwen--Qwen3.5-27B-FP8; mkdir -p $LOCAL; cp -rL $SNAP/. $LOCAL/
cd /scratch/s/sneheel/ChameleonBench
module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.9 python/3.12 opencv/4.13.0 scipy-stack/2026a
tar -xf /scratch/s/sneheel/venvs/vllm-serving.tar -C $SLURM_TMPDIR
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
$SLURM_TMPDIR/vllm-serving/bin/python scripts/label_concept_contamination.py --model $LOCAL \
  --data "$DATA" --skip-labelled "$OLD" --out "$OUT" --prompt v4 --tp 2 --mem 0.90 \
  --max-num-seqs 256 "$@" 2>&1 | grep -vE "it/s\]|^\s*$" | tail -60
