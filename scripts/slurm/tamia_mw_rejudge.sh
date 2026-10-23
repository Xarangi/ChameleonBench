#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=2:00:00
#SBATCH --job-name=mw-rejudge
#SBATCH --output=/scratch/s/sneheel/ChameleonBench/logs/%x-%j.out
# Re-judge ONE generation shard of tamia_mw_build.sh split 4 ways over 4 GPUs (10-23: held-out judge shard 3 hung
# at 100% GPU for 50 min while shards 0-2 took 13 min; the judge is prefill-only, max_tokens 1). Each piece has a
# timeout, so a repeat hang loses only that piece's rows. Writes OUT/scores_TAG_S.jsonl + judge_TAG_S.json.
#   sbatch scripts/slurm/tamia_mw_rejudge.sh WORKDIR TAG SHARD
set -uo pipefail
W=${1:?workdir}; TAG=${2:?tag}; S=${3:?shard}
S27=/scratch/s/sneheel/hf/hub/models--Qwen--Qwen3.5-27B-FP8/snapshots/97f5941bf617e31c5e237364a8602ce3f03a551a
VPY=${MW_VPY:-/scratch/s/sneheel/venvs/vllm-latest/bin/python}
NODE_TMP=${SLURM_TMPDIR:-/tmp/mwr_$$}
mkdir -p "$NODE_TMP"
cd /scratch/s/sneheel/ChameleonBench
module --force purge
module load StdEnv/2023 python/3.11 arrow/24.0.0 cuda/12.6 2>/dev/null
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_CACHE_ROOT=$NODE_TMP/vllm_cache VLLM_USE_DEEP_GEMM=0
L27=$NODE_TMP/models--Qwen--Qwen3.5-27B-FP8; mkdir -p "$L27"; cp -rL "$S27"/. "$L27"/
G=$W/gen_${TAG}_$S.jsonl
P=$W/rejudge_$S; mkdir -p "$P"
awk -v d="$P" '{print > (d "/gen_" (NR-1)%4 ".jsonl")}' "$G"
PIDS=()
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g timeout 40m $VPY scripts/tica_free_judge.py --model "$L27" --tp 1 --mem 0.85 \
    --answers-prompt loose --maxc "${MW_MAXC:-6000}" --row-out "$P/scores_$g.jsonl" --out "$P/judge_$g.json" \
    --gens "${TAG}_${S}_$g=$P/gen_$g.jsonl" > "$P/judge_$g.log" 2>&1 &
  PIDS+=($!)
done
for g in 0 1 2 3; do wait "${PIDS[$g]}"; echo "piece $g rc $?"; done
cat "$P"/scores_*.jsonl > "$W/scores_${TAG}_$S.jsonl"
wc -l "$G" "$W/scores_${TAG}_$S.jsonl"
grep -h "cond=nobrief" "$P"/judge_*.log
echo "REJUDGE DONE"
