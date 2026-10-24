#!/bin/bash
# TamIA orchestration of v4c on GEMMA-4 12B (10-24; docs/tica/GEMMA4_12B.md). Login-node safe: only sbatch / squeue /
# ls / tail / grep (no python here). Two arms, ONE NODE EACH (4 x H100, data-parallel DP4, one process per GPU):
#   final_v4c_gemma4_12b_tamia (1-bit verdict) and final_v4c_score_gemma4_12b_tamia (score feedback).
#
#   bash scripts/tamia_gemma12b.sh check          # inputs, venvs, Gemma caches present?
#   bash scripts/tamia_gemma12b.sh smoke          # <= 40 min: sanity + vLLM on GPU 0, 3-step DP3 trainer on GPUs 1-3
#   bash scripts/tamia_gemma12b.sh prep           # Gemma base generations (cap_gen_pool) + turn-0 panel cache, 1 node
#   [ARMS="a b"] bash scripts/tamia_gemma12b.sh launch [CHUNKS]   # CHUNKS x 24 h per arm, chained resumes + drains
#   bash scripts/tamia_gemma12b.sh status
set -euo pipefail
cd /scratch/s/sneheel/ChameleonBench
ACCT=${ACCT:-aip-rrabba}
EXCL=${EXCL:---exclude=tg11105}
R=/scratch/s/sneheel/next_chameleons_raw_cache
A=/scratch/s/sneheel/next_chameleons_artifacts
RUNS=$A/tica/runs
SNAP=/scratch/s/sneheel/hf/hub/models--google--gemma-4-12b-it/snapshots/5926caa4ec0cac5cbfadaf4077420520de1d5205
QUEUE=${QUEUE:-$RUNS/g4v4c_val_queue.txt}
JN=${JOBNAME:-g4v4c}
read -ra ARMS <<< "${ARMS:-final_v4c_gemma4_12b_tamia final_v4c_score_gemma4_12b_tamia}"
G=0,1,2,3
mkdir -p logs

need() {
  local miss=0
  for f in "$@"; do
    if [ -e "$f" ]; then echo "  ok      $f"; else echo "  MISSING $f"; miss=1; fi
  done
  return $miss
}

check() {
  echo "inputs of ${ARMS[*]}:"
  need /scratch/s/sneheel/venvs/q35t29.tar /scratch/s/sneheel/venvs/vllm-g4/bin/python \
    $SNAP/config.json $SNAP/model.safetensors $SNAP/chat_template.jinja \
    $R/tica/final_v5_clean.json $R/tica/final_v5_elicit_inputs.json $R/tica/concept_labels_v5data_27b.json \
    $R/tica/dolly15k/databricks-dolly-15k.jsonl $R/capability/gsm8k_train.json \
    $RUNS/mathrl_gemma4_12b/base_gens.json
}

case "${1:-}" in
  check)
    check ;;
  smoke)
    sbatch --account="$ACCT" $EXCL scripts/slurm/tamia_gemma12b_smoke.sh "${ARMS[0]}" ;;
  prep)
    # 1. Gemma's own greedy GSM8K solutions for cap_gen_pool (HF generation; ~1 h on one GPU)
    J1=$(sbatch --parsable --account="$ACCT" $EXCL --time=3:00:00 -J g4-gens scripts/slurm/tamia_gemma12b_gens.sh)
    # 2. the turn-0 panel cache for both arms (they share every key: same concepts / conditions / kinds)
    J2=$(PROCS=${PROCS:-8} sbatch --parsable --account="$ACCT" $EXCL --time=10:00:00 -J g4-panel \
         scripts/slurm/tamia_panel_precompute.sh "${ARMS[0]}")
    echo "gens: $J1   panel precompute: $J2" ;;
  launch)
    check || { echo "inputs missing (run 'prep' first): see docs/tica/GEMMA4_12B.md"; exit 1; }
    N=${2:-8}
    export TICA_VAL_QUEUE=$QUEUE
    for a in "${ARMS[@]}"; do
      J=$(sbatch --parsable --account="$ACCT" $EXCL -J "$JN" scripts/slurm/tamia_node.sh "${a}@$G")
      echo "$a chunk 1: $J"
      for (( i = 2; i <= N; i++ )); do
        J=$(sbatch --parsable --account="$ACCT" $EXCL --dependency=afterany:$J -J "$JN" \
            scripts/slurm/tamia_node.sh "resume:${a}@$G")
        echo "$a chunk $i: $J"
      done
    done
    D=$(sbatch --parsable --account="$ACCT" $EXCL --time=16:00:00 --begin=now+24hours -J "$JN-drain" \
        scripts/slurm/tica_val_drain.sh "$QUEUE")
    echo "first held-out race drain: $D (re-submit after later chunks: sbatch scripts/slurm/tica_val_drain.sh $QUEUE)" ;;
  status)
    squeue -u "$USER" -o "%.10i %.12j %.2t %.10M %.10L %R" || true
    for a in "${ARMS[@]}"; do
      echo "== $a"
      for d in "$RUNS/$a"/adaptive "$RUNS/$a"/resume_*/adaptive; do
        [ -f "$d/train_log.jsonl" ] || continue
        echo "  $d: $(wc -l < "$d/train_log.jsonl") steps logged; newest ckpt $(ls -d "$d"/step_* 2>/dev/null | sort -V | tail -1)"
        tail -1 "$d/train_log.jsonl" | tr ',' '\n' | grep -E '"(step|sec|max_mem_gb|loss|rollout/pass_rate|cap_gen_acc|grad_norm)"' | tr '\n' ' '; echo
      done
    done ;;
  *)
    sed -n 2,10p "$0"; exit 2 ;;
esac
