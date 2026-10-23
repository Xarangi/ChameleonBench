#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=6:00:00
#SBATCH --job-name=mw-finish
#SBATCH --output=/scratch/s/sneheel/ChameleonBench/logs/%x-%j.out
# Model-written data on TamIA, after tamia_mw_build.sh (train + heldout). One 4 x H100 node instead of the Narval
# chain (tica_mw_finish.sh -> 2 label jobs -> merge -> probe check):
#   1. build two data files (CPU): final_v5_mw.json = train concepts model-written (what
#      final_v5_modelwritten_4b.yaml reads), final_v5_mw_all.json = train + held-out concepts model-written
#   2. 27B-FP8 negative labels of the NEW texts of the _all file (superset), 4 shards tp=1, --skip-labelled
#   3. merge-labels into each file
#   4. base-probe check (model-written vs published texts) on the train file
# Aggregates only; outputs on scratch.
#   sbatch --dependency=afterok:BUILD_TRAIN:BUILD_HELDOUT scripts/slurm/tamia_mw_finish.sh MW_ROOT
set -euo pipefail
W=${1:?mw root with train/ and heldout/}
T=/scratch/s/sneheel/next_chameleons_raw_cache/tica
S27=/scratch/s/sneheel/hf/hub/models--Qwen--Qwen3.5-27B-FP8/snapshots/97f5941bf617e31c5e237364a8602ce3f03a551a
VPY=${MW_VPY:-/scratch/s/sneheel/venvs/vllm-latest/bin/python}
NODE_TMP=${SLURM_TMPDIR:-/tmp/mwf_$$}
mkdir -p "$NODE_TMP"
cd /scratch/s/sneheel/ChameleonBench
source scripts/slurm/_local_env.sh
t0=$(date +%s)
L27=$NODE_TMP/models--Qwen--Qwen3.5-27B-FP8; mkdir -p "$L27"
(cp -rL "$S27"/. "$L27"/ && touch "$L27/.copied") & CP=$!

echo "== 1. build"
$PY scripts/build_final_v5_mw.py build --data $T/final_v5_clean.json --inputs $T/final_v5_elicit_inputs.json \
  --gens $W/train/gen_*.jsonl --scores $W/train/scores_*.jsonl \
  --out $T/final_v5_mw.json --inputs-out $T/final_v5_mw_elicit_inputs.json --report $W/report_train.json
$PY scripts/build_final_v5_mw.py build --data $T/final_v5_clean.json --inputs $T/final_v5_elicit_inputs.json \
  --gens $W/train/gen_*.jsonl $W/heldout/gen_*.jsonl --scores $W/train/scores_*.jsonl $W/heldout/scores_*.jsonl \
  --out $T/final_v5_mw_all.json --inputs-out $T/final_v5_mw_all_elicit_inputs.json --report $W/report_all.json
echo "build done $(( $(date +%s) - t0 ))s"

echo "== 2. labels (27B-FP8, tp=1 x 4)"
wait $CP; [ -f "$L27/.copied" ] || { echo "27B copy failed"; exit 1; }
(
  module --force purge
  module load StdEnv/2023 python/3.11 arrow/24.0.0 cuda/12.6 2>/dev/null
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_CACHE_ROOT=$NODE_TMP/vllm_cache VLLM_USE_DEEP_GEMM=0
  PIDS=()
  for g in 0 1 2 3; do
    [ -s "$W/labels_s$g.json" ] && continue
    CUDA_VISIBLE_DEVICES=$g $VPY scripts/label_concept_contamination.py --model "$L27" \
      --data $T/final_v5_mw_all.json --skip-labelled $T/concept_labels_v5data_27b.json \
      --out "$W/labels_s$g.json" --prompt v4 --tp 1 --mem 0.88 --max-num-seqs 256 --shard $g/4 \
      > "$W/labels_s$g.log" 2>&1 &
    PIDS+=($!)
  done
  rc=0; for p in "${PIDS[@]:-}"; do [ -n "$p" ] && { wait "$p" || rc=1; }; done
  (( rc == 0 )) || { echo "LABELS FAILED"; tail -20 "$W"/labels_s*.log; exit 1; }
)
echo "labels done $(( $(date +%s) - t0 ))s"

echo "== 3. merge-labels"
for f in mw mw_all; do
  $PY scripts/build_final_v5_mw.py merge-labels --data $T/final_v5_$f.json \
    --old $T/concept_labels_v5data_27b.json --new $W/labels_s{0,1,2,3}.json \
    --out $T/concept_labels_v5${f/_/}_27b.json
done

echo "== 4. base-probe check"
SNAP=$(PYTHONPATH=src $PY -c "
from next_chameleons.tica.config import load_run_config as l; print(l('final_v5_modelwritten_4b').model.path)")
L4=$NODE_TMP/$(basename "$(dirname "$(dirname "$SNAP")")"); mkdir -p "$L4"; cp -rL "$SNAP"/. "$L4"/
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src $PY scripts/tica_mw_probe_check.py --config final_v5_modelwritten_4b \
  --model-path "$L4" --out $W/probe_check.json --old-data $T/final_v5_clean.json \
  --old-inputs $T/final_v5_elicit_inputs.json --old-labels $T/concept_labels_v5data_27b.json \
  --concepts "sports|all-caps text|politics and elections|German language|Arabic language|JavaScript code|SQL code|JSON data"
echo "MW FINISH OK $(( $(date +%s) - t0 ))s"
