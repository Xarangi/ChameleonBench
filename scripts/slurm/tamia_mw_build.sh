#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --job-name=mw-build
#SBATCH --output=/scratch/s/sneheel/ChameleonBench/logs/%x-%j.out
# MODEL-WRITTEN concept texts on TamIA (4 x H100; the Narval tica_mw_build.sh is 2 x A100 + the
# vllm-serving tar, neither of which exists here). Two phases, both 4-way data-parallel with ONE
# model copy per GPU (no tensor parallelism: 27B-FP8 ~27 GB fits in 80 GB, so tp=1 x 4 beats tp=4):
#   1. Qwen3.5-4B answers the v5 elicit requests  -> OUT/gen_TAG_{0..3}.jsonl
#   2. Qwen3.5-27B-FP8 judges each shard (concept central + answer reasonable, --answers-prompt
#      loose) -> OUT/scores_TAG_{0..3}.jsonl + judge_TAG_{0..3}.json
# Scratch only (raw model text); the scripts print aggregates only.
#   sbatch scripts/slurm/tamia_mw_build.sh WORKDIR TAG CONCEPTS EXCLUDE [SAMPLES] [MAX_SRC]
# e.g.  ... /scratch/.../mw_1023 train    train   'mathematical reasoning'                 3
#       ... /scratch/.../mw_1023 heldout  heldout 'deception: DolusChat|deception: insider trading|deception: roleplaying|deception: sycophancy' 3
set -uo pipefail
W=${1:?workdir}; TAG=${2:?tag}; CONCEPTS=${3:?train|heldout|a|b}; EXCL=${4:-}; SAMPLES=${5:-3}; MAXSRC=${6:-0}
T=/scratch/s/sneheel/next_chameleons_raw_cache/tica
DATA=${MW_DATA:-$T/final_v5_clean.json}
INPUTS=${MW_INPUTS:-$T/final_v5_elicit_inputs.json}
S4=/scratch/s/sneheel/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
S27=/scratch/s/sneheel/hf/hub/models--Qwen--Qwen3.5-27B-FP8/snapshots/97f5941bf617e31c5e237364a8602ce3f03a551a
VPY=${MW_VPY:-/scratch/s/sneheel/venvs/vllm-latest/bin/python}
NODE_TMP=${SLURM_TMPDIR:-/tmp/mw_$$}
mkdir -p "$W" "$NODE_TMP"
cd /scratch/s/sneheel/ChameleonBench
module --force purge
module load ${MW_MODULES:-StdEnv/2023 python/3.11 arrow/24.0.0 cuda/12.6} 2>/dev/null
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_CACHE_ROOT=$NODE_TMP/vllm_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 27B-FP8 judge (10-23): vLLM picks FlashInferFp8DeepGEMM* for FP8 linears and JIT-compiles DeepGEMM
# with NVCC. TamIA only has cuda/12.2 and 12.6; DeepGEMM needs >= 12.9 and the PTX fails to compile
# ("asm operand type size(16) does not match"), so the engine never starts. Fall back to the
# non-DeepGEMM FP8 kernels. Generation (bf16 4B) is unaffected either way.
export VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM:-0}
t0=$(date +%s)

# 4B node-local (small); the 27B copy runs in the background during generation
L4=$NODE_TMP/models--Qwen--Qwen3.5-4B; L27=$NODE_TMP/models--Qwen--Qwen3.5-27B-FP8
mkdir -p "$L4" "$L27"; cp -rL "$S4"/. "$L4"/
(cp -rL "$S27"/. "$L27"/ && touch "$L27/.copied") & CP=$!

echo "== phase 1: generate ($CONCEPTS, exclude '$EXCL', samples $SAMPLES, max_src $MAXSRC)"
PIDS=()
for g in 0 1 2 3; do
  [ -s "$W/gen_${TAG}_$g.jsonl" ] && { echo "  shard $g already done, skipping"; continue; }
  CUDA_VISIBLE_DEVICES=$g $VPY scripts/tica_mw_generate.py --model "$L4" --data "$DATA" \
    --inputs "$INPUTS" --concepts "$CONCEPTS" --exclude "$EXCL" --samples "$SAMPLES" \
    --max-src "$MAXSRC" --shard "$g/4" --mem 0.85 --out "$W/gen_${TAG}_$g.jsonl" \
    > "$W/gen_${TAG}_$g.log" 2>&1 &
  PIDS+=($!)
done
rc=0; for p in "${PIDS[@]:-}"; do [ -n "$p" ] && { wait "$p" || rc=1; }; done
(( rc == 0 )) || { echo "GENERATION FAILED"; tail -25 "$W"/gen_${TAG}_*.log; exit 1; }
echo "generation done $(( $(date +%s) - t0 ))s"
grep -h '"n_gen"\|"gen_sec"' "$W"/gen_${TAG}_*.log | head -8 || true
wc -l "$W"/gen_${TAG}_*.jsonl

wait $CP || true; [ -f "$L27/.copied" ] || { echo "27B copy failed"; exit 1; }
echo "== phase 2: judge (27B-FP8, tp=1 x 4)"
PIDS=()
for g in 0 1 2 3; do
  [ -s "$W/scores_${TAG}_$g.jsonl" ] && { echo "  judge $g already done, skipping"; continue; }
  [ -s "$W/gen_${TAG}_$g.jsonl" ] || continue
  CUDA_VISIBLE_DEVICES=$g $VPY scripts/tica_free_judge.py --model "$L27" --tp 1 --mem 0.85 \
    --answers-prompt loose --maxc "${MW_MAXC:-6000}" \
    --row-out "$W/scores_${TAG}_$g.jsonl" --out "$W/judge_${TAG}_$g.json" \
    --gens "${TAG}_$g=$W/gen_${TAG}_$g.jsonl" > "$W/judge_${TAG}_$g.log" 2>&1 &
  PIDS+=($!)
done
rc=0; for p in "${PIDS[@]:-}"; do [ -n "$p" ] && { wait "$p" || rc=1; }; done
(( rc == 0 )) || { echo "JUDGE FAILED"; tail -25 "$W"/judge_${TAG}_*.log; exit 1; }
echo "judge done $(( $(date +%s) - t0 ))s"
wc -l "$W"/scores_${TAG}_*.jsonl
echo "MW BUILD OK ($TAG)"
