#!/bin/bash
# PREREG A3c pre-flight: the L1 named-arm hider (seed 0) on the top-5 candidate concepts, one job.
# ctb-liyue_gpu has the HIGHER fairshare factor (0.352 vs def-rrabba_gpu 0.272 on 10-10), so it
# gets better priority even though def-rrabba has no recent usage. Override with --account=.
#SBATCH --account=ctb-liyue_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=tica-l3-preflight
#SBATCH --output=/home/sneheel/chameleons/logs/%x-%j.out
set -uo pipefail
PY=${PY:-/home/sneheel/chameleons-qwen35/.venv/bin/python}
export TRITON_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/triton  # home is at quota (10-10)
SNAP=/scratch/sneheel/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
LOCAL=${SLURM_TMPDIR:-/tmp}/qwen35_4b
t0=$(date +%s); mkdir -p $LOCAL; cp -rL $SNAP/. $LOCAL/; echo "model copy: $(( $(date +%s) - t0 ))s"
AD=/scratch/sneheel/next_chameleons_artifacts/tica/runs/l1_qwen35_4b/seed0/train/final
OUT=/scratch/sneheel/next_chameleons_artifacts/tica/runs/l3pre_qwen35_4b
cd /home/sneheel/chameleons
k=0
for C in "playful and onomatopoeic sounds or expressions" "conversational expressions or discourse markers" \
         "expressions of surprise or disbelief" \
         "informal second-person pronouns and abbreviations commonly used in casual communication" \
         "phrases related to past events and personal connections"; do
  if [ -f $OUT/c$k/eval/eval.json ]; then echo "=== concept $k done, skipping"; k=$((k+1)); continue; fi
  echo "=== concept $k: $C"
  env PYTHONPATH=src $PY scripts/tica.py windows --config l3pre_qwen35_4b --stage eval --adapter $AD \
    --concept "$C" --out $OUT/c$k --model-path $LOCAL
  k=$((k+1))
done
