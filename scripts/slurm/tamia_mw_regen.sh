#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=6:00:00
#SBATCH --job-name=mw-regen
#SBATCH --output=/scratch/s/sneheel/ChameleonBench/logs/%x-%j.out
# LENGTH-HINTED regeneration of the over-long model-written answers (user 10-24: "adding a length limit in the
# prompt, only for the examples that are over long"), after tamia_mw_build.sh + tamia_mw_finish.sh:
#   1. needs-regen: source texts with NO eligible answer and >= 1 answer over the p90 cap / cut at max tokens
#   2. 4B regenerates ONLY those (4 samples, request + a length line matched to the source text's length,
#      gids :100.., the stored request stays the published one)            4 GPUs
#   3. 27B judge (same loose prompt, sees the published request)            4 GPUs
#   4. rebuild both data files from original + regenerated answers (same cap, same rules)
#   5. VERIFY from scratch: every kept text finished, passed the judge, <= cap, published request (fails the job
#      otherwise); per-concept counts / lengths / kept judge scores
#   6. 27B labels for the texts new since the last label run; merge; base-probe check
# Previous data files are kept as *_nohint.json. Aggregates only.
#   sbatch scripts/slurm/tamia_mw_regen.sh MW_ROOT
set -euo pipefail
W=${1:?mw root with train/ heldout/ and labels_s*.json}
T=/scratch/s/sneheel/next_chameleons_raw_cache/tica
S4=/scratch/s/sneheel/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
S27=/scratch/s/sneheel/hf/hub/models--Qwen--Qwen3.5-27B-FP8/snapshots/97f5941bf617e31c5e237364a8602ce3f03a551a
VPY=${MW_VPY:-/scratch/s/sneheel/venvs/vllm-latest/bin/python}
NODE_TMP=${SLURM_TMPDIR:-/tmp/mwg_$$}
G=$W/regen; mkdir -p "$NODE_TMP" "$G"
cd /scratch/s/sneheel/ChameleonBench
source scripts/slurm/_local_env.sh
t0=$(date +%s)
L4=$NODE_TMP/models--Qwen--Qwen3.5-4B; L27=$NODE_TMP/models--Qwen--Qwen3.5-27B-FP8
mkdir -p "$L4" "$L27"; cp -rL "$S4"/. "$L4"/
(cp -rL "$S27"/. "$L27"/ && touch "$L27/.copied") & CP=$!
ORIG_G=("$W"/train/gen_*.jsonl "$W"/heldout/gen_*.jsonl)
ORIG_S=("$W"/train/scores_*.jsonl "$W"/heldout/scores_*.jsonl)

vllm_env() {
  module --force purge
  module load StdEnv/2023 python/3.11 arrow/24.0.0 cuda/12.6 2>/dev/null
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_CACHE_ROOT=$NODE_TMP/vllm_cache VLLM_USE_DEEP_GEMM=0
}

echo "== 1. needs-regen"
$PY scripts/build_final_v5_mw.py needs-regen --data $T/final_v5_clean.json --gens "${ORIG_G[@]}" \
  --scores "${ORIG_S[@]}" --out $G/regen_src.json
CONCEPTS=$(/usr/bin/python3 -c "import json;print('|'.join(sorted({v['concept'] for v in json.load(open('$G/regen_src.json')).values()})))")

echo "== 2. generate (length-hinted)"
(
  vllm_env
  PIDS=()
  for g in 0 1 2 3; do
    [ -s "$G/gen_$g.jsonl" ] && continue
    CUDA_VISIBLE_DEVICES=$g $VPY scripts/tica_mw_generate.py --model "$L4" --data $T/final_v5_clean.json \
      --inputs $T/final_v5_elicit_inputs.json --concepts "$CONCEPTS" --exclude "" --samples 4 \
      --sample-offset 100 --seed 1 --only-src $G/regen_src.json --length-hint --shard $g/4 --mem 0.85 \
      --out "$G/gen_$g.jsonl" > "$G/gen_$g.log" 2>&1 &
    PIDS+=($!)
  done
  rc=0; for p in "${PIDS[@]:-}"; do [ -n "$p" ] && { wait "$p" || rc=1; }; done
  (( rc == 0 )) || { echo "GENERATION FAILED"; tail -n 20 "$G"/gen_*.log; exit 1; }
)
echo "generation done $(( $(date +%s) - t0 ))s"; cat "$G"/gen_*.jsonl | wc -l

echo "== 3. judge"
wait $CP; [ -f "$L27/.copied" ] || { echo "27B copy failed"; exit 1; }
(
  vllm_env
  PIDS=()
  for g in 0 1 2 3; do
    [ -s "$G/scores_$g.jsonl" ] && continue
    CUDA_VISIBLE_DEVICES=$g timeout 60m $VPY scripts/tica_free_judge.py --model "$L27" --tp 1 --mem 0.85 \
      --answers-prompt loose --maxc 6000 --row-out "$G/scores_$g.jsonl" --out "$G/judge_$g.json" \
      --gens "regen_$g=$G/gen_$g.jsonl" > "$G/judge_$g.log" 2>&1 &
    PIDS+=($!)
  done
  rc=0; for p in "${PIDS[@]:-}"; do [ -n "$p" ] && { wait "$p" || rc=1; }; done
  (( rc == 0 )) || { echo "JUDGE FAILED"; tail -n 20 "$G"/judge_*.log; exit 1; }
)
echo "judge done $(( $(date +%s) - t0 ))s"; grep -h "cond=nobrief" "$G"/judge_*.log

echo "== 4. rebuild"
for f in final_v5_mw final_v5_mw_all; do
  [ -f $T/${f}_nohint.json ] || { cp $T/$f.json $T/${f}_nohint.json; cp $T/${f}_elicit_inputs.json $T/${f}_nohint_elicit_inputs.json; }
done
ALL_G=("${ORIG_G[@]}" "$G"/gen_*.jsonl); ALL_S=("${ORIG_S[@]}" "$G"/scores_*.jsonl)
TR_G=("$W"/train/gen_*.jsonl "$G"/gen_*.jsonl); TR_S=("$W"/train/scores_*.jsonl "$G"/scores_*.jsonl)
# the train file regenerates the train concepts only (the regen shards also hold held-out concepts)
mapfile -t TRC < <(/usr/bin/python3 -c "import json;print('\n'.join(json.load(open('$W/report_train.json'))))")
$PY scripts/build_final_v5_mw.py build --data $T/final_v5_clean.json --inputs $T/final_v5_elicit_inputs.json \
  --gens "${TR_G[@]}" --scores "${TR_S[@]}" --only "${TRC[@]}" \
  --out $T/final_v5_mw.json --inputs-out $T/final_v5_mw_elicit_inputs.json --report $W/report_train_hint.json
$PY scripts/build_final_v5_mw.py build --data $T/final_v5_clean.json --inputs $T/final_v5_elicit_inputs.json \
  --gens "${ALL_G[@]}" --scores "${ALL_S[@]}" \
  --out $T/final_v5_mw_all.json --inputs-out $T/final_v5_mw_all_elicit_inputs.json --report $W/report_all_hint.json

echo "== 5. verify"
for f in final_v5_mw final_v5_mw_all; do
  $PY scripts/build_final_v5_mw.py verify --old-data $T/final_v5_clean.json --data $T/$f.json \
    --inputs $T/${f}_elicit_inputs.json --gens "${ALL_G[@]}" --scores "${ALL_S[@]}" --out $W/verify_$f.json
done

echo "== 6. labels + merge + probe check"
(
  vllm_env
  PIDS=()
  for g in 0 1 2 3; do
    [ -s "$W/labels_r$g.json" ] && continue
    CUDA_VISIBLE_DEVICES=$g $VPY scripts/label_concept_contamination.py --model "$L27" \
      --data $T/final_v5_mw_all.json --skip-labelled $T/concept_labels_v5mwall_27b.json \
      --out "$W/labels_r$g.json" --prompt v4 --tp 1 --mem 0.88 --max-num-seqs 256 --shard $g/4 \
      > "$W/labels_r$g.log" 2>&1 &
    PIDS+=($!)
  done
  rc=0; for p in "${PIDS[@]:-}"; do [ -n "$p" ] && { wait "$p" || rc=1; }; done
  (( rc == 0 )) || { echo "LABELS FAILED"; tail -n 20 "$W"/labels_r*.log; exit 1; }
)
for f in mw mw_all; do
  $PY scripts/build_final_v5_mw.py merge-labels --data $T/final_v5_$f.json \
    --old $T/concept_labels_v5data_27b.json --new $W/labels_s{0,1,2,3}.json $W/labels_r{0,1,2,3}.json \
    --out $T/concept_labels_v5${f/_/}_27b.json
done
SNAP=$(PYTHONPATH=src $PY -c "
from next_chameleons.tica.config import load_run_config as l; print(l('final_v5_modelwritten_4b').model.path)")
L4P=$NODE_TMP/probe_$(basename "$(dirname "$(dirname "$SNAP")")"); mkdir -p "$L4P"; cp -rL "$SNAP"/. "$L4P"/
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src $PY scripts/tica_mw_probe_check.py --config final_v5_modelwritten_4b \
  --model-path "$L4P" --out $W/probe_check_hint.json --old-data $T/final_v5_clean.json \
  --old-inputs $T/final_v5_elicit_inputs.json --old-labels $T/concept_labels_v5data_27b.json \
  --concepts "sports|all-caps text|politics and elections|German language|Arabic language|JavaScript code|SQL code|JSON data|astronomy|chemistry|literature|finance"
echo "MW REGEN OK $(( $(date +%s) - t0 ))s"
