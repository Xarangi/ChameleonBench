#!/bin/bash
#SBATCH --account=ctb-liyue_gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --job-name=mw-build
#SBATCH --output=/scratch/s/sneheel/ChameleonBench/logs/%x-%j.out
# Model-written concept data (REPLICATION 2026-10-21f; docs/tica/V4_MODEL_WRITTEN_DATA.md), one shard:
#   1. Qwen3.5-4B answers the v5 elicit requests: two tica_mw_generate.py processes, one per GPU
#      (args "GEN0" / "GEN1", e.g. "--shard 0/8" / "--shard 1/8"); the 27B copy runs meanwhile;
#   2. Qwen3.5-27B-FP8 (tensor-parallel over both GPUs) judges both outputs:
#      tica_free_judge.py --answers-prompt loose --row-out (concept central + reasonable answer).
# Usage: sbatch tica_mw_build.sh OUTDIR TAG "GEN0 args" "GEN1 args"
# Outputs (scratch only, never commit): OUTDIR/gen_TAG_{0,1}.jsonl, scores_TAG.jsonl, judge_TAG.json
set -euo pipefail
OUT=$1; TAG=$2; G0=$3; G1=$4
T=/scratch/s/sneheel/next_chameleons_raw_cache/tica
DATA=${MW_DATA:-$T/final_v5_clean.json}
INPUTS=${MW_INPUTS:-$T/final_v5_elicit_inputs.json}
S4=/scratch/s/sneheel/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
S27=/scratch/s/sneheel/hf/hub/models--Qwen--Qwen3.5-27B-FP8/snapshots/97f5941bf617e31c5e237364a8602ce3f03a551a
L4=$SLURM_TMPDIR/models--Qwen--Qwen3.5-4B; L27=$SLURM_TMPDIR/models--Qwen--Qwen3.5-27B-FP8
mkdir -p $OUT $L4 $L27
cp -rL $S4/. $L4/
(cp -rL $S27/. $L27/ && touch $L27/.copied) &
CP=$!
cd /scratch/s/sneheel/ChameleonBench
module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.9 python/3.12 opencv/4.13.0 scipy-stack/2026a
tar -xf /scratch/s/sneheel/venvs/vllm-serving.tar -C $SLURM_TMPDIR
VPY=$SLURM_TMPDIR/vllm-serving/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_CACHE_ROOT=$SLURM_TMPDIR/vllm_cache
t0=$(date +%s)
PIDS=()
for g in 0 1; do
  ARGS=$G0; [ $g = 1 ] && ARGS=$G1
  # eval: the arg strings may quote concept names with spaces ('a b|c d')
  eval "CUDA_VISIBLE_DEVICES=$g $VPY scripts/tica_mw_generate.py --model $L4 --data $DATA \
    --inputs $INPUTS --out $OUT/gen_${TAG}_$g.jsonl $ARGS" > $OUT/gen_${TAG}_$g.log 2>&1 &
  PIDS+=($!)
done
for p in "${PIDS[@]}"; do
  wait $p || { echo "generation failed"; tail -30 $OUT/gen_${TAG}_*.log; exit 1; }
done
echo "generation done $(( $(date +%s) - t0 ))s"
grep -h -A3 '"n_gen"' $OUT/gen_${TAG}_*.log | head -20 || true
wait $CP || true; [ -f $L27/.copied ] || { echo "27B copy failed"; exit 1; }
$VPY scripts/tica_free_judge.py --model $L27 --tp 2 --answers-prompt loose --maxc ${MW_MAXC:-6000} \
  --row-out $OUT/scores_$TAG.jsonl --out $OUT/judge_$TAG.json \
  --gens ${TAG}_0=$OUT/gen_${TAG}_0.jsonl ${TAG}_1=$OUT/gen_${TAG}_1.jsonl 2>&1 \
  | grep -vE "it/s\]|^\s*$" | tail -30
echo "judge done $(( $(date +%s) - t0 ))s"
