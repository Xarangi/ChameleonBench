#!/bin/bash
# TamIA orchestration of THE TamIA RECIPE (docs/tica-v4-tamia.md; docs/tica/TAMIA-V4B.md): the two v4b arms,
# final_v4b_4b (1-bit verdict) and final_v4b_score_4b (score feedback), both logclip + standalone stay + turn-0
# panel from step 0 + KL 0.3. Login-node safe: this script only runs sbatch / squeue / ls / tail (no python).
#
#   bash scripts/tamia_v4b.sh check            # every input file + venv present?
#   bash scripts/tamia_v4b.sh smoke            # 20 steps of BOTH arms on one node (DP2 each), ~1-2 h
#   [LAYOUT=interleaved|solo] bash scripts/tamia_v4b.sh launch [CHUNKS]  # 4 x 24 h chain + drains (races + quality)
#   bash scripts/tamia_v4b.sh quality ADAPTER_A ADAPTER_B   # queue quality-check races, then drain
#   bash scripts/tamia_v4b.sh status           # jobs, latest step per arm, last metrics
#
# NODE LAYOUT (4 x H100 80 GB, whole nodes; reasoning in docs/tica/TAMIA-V4B.md):
#   GPUs 0-1 = final_v4b_4b (data-parallel episodes, TICA_DP=2), GPUs 2-3 = final_v4b_score_4b (TICA_DP=2).
#   Held-out races (eval_every 50, logged only) are QUEUED (TICA_VAL_QUEUE) instead of holding 2 GPUs per arm idle,
#   and run 4 at a time by scripts/slurm/tica_val_drain.sh after every chunk.
set -euo pipefail
cd /scratch/s/sneheel/ChameleonBench
ACCT=${ACCT:-aip-rrabba}
R=/scratch/s/sneheel/next_chameleons_raw_cache
A=/scratch/s/sneheel/next_chameleons_artifacts
RUNS=$A/tica/runs
QUEUE=$RUNS/v4b_val_queue.txt
# TamIA arms (user 10-21): final_v4b + LoRA r128 + 16 episodes / step + dp_drift_fatal; override with ARMS="a b"
read -ra ARMS <<< "${ARMS:-final_v4b_tamia_4b final_v4b_score_tamia_4b}"
# LAYOUT (chosen by the layout smokes, docs/tica/TAMIA-V4B.md): interleaved = ONE node, both arms data-parallel over
# all 4 GPUs (2 processes per GPU); solo = one node per arm, 4 GPUs each
LAYOUT=${LAYOUT:-solo}   # 10-21 smokes: interleaved OOMs at r128 / 24 turns (2 x ~40-50 GB + 7.5 GB logit spikes)
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
  need /scratch/s/sneheel/venvs/q35t29.tar \
    /scratch/s/sneheel/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json \
    $R/tica/final_v5_clean.json $R/tica/final_v5_elicit_inputs.json $R/tica/concept_labels_v5data_27b.json \
    $R/tica/dolly15k/databricks-dolly-15k.jsonl $R/capability/gsm8k_train.json \
    $RUNS/mathrl_4b/base_gens.json \
    $A/tica/l1_base/qwen35_4b/probes/quality.json \
    $A/tica/l1_base/qwen35_4b/probes/bank_L12.npz $A/tica/l1_base/qwen35_4b/probes/bank_L20.npz \
    $A/tica/l1_base/qwen35_4b/probes/bank_L24.npz
}

case "${1:-}" in
  check)
    check ;;
  smoke)
    check || { echo "inputs missing: see docs/tica/TAMIA-V4B.md section 'Data'"; exit 1; }
    sbatch --account="$ACCT" --time=3:00:00 -J v4b-smoke scripts/slurm/tamia_v4b_smoke.sh ;;
  launch)
    check || { echo "inputs missing: see docs/tica/TAMIA-V4B.md section 'Data'"; exit 1; }
    N=${2:-4}
    export TICA_VAL_QUEUE=$QUEUE
    submit_chunk() {  # PREFIX (""|resume:) DEPENDENCY -> job ids
      local pre=$1 dep=$2 ids=() j
      if [ "$LAYOUT" = interleaved ]; then
        j=$(sbatch --parsable --account="$ACCT" $dep -J v4b scripts/slurm/tamia_node.sh \
            "${pre}${ARMS[0]}@$G" "${pre}${ARMS[1]}@$G"); ids+=("$j")
      else
        for a in "${ARMS[@]}"; do
          j=$(sbatch --parsable --account="$ACCT" $dep -J "v4b-${a}" scripts/slurm/tamia_node.sh "${pre}${a}@$G"); ids+=("$j")
        done
      fi
      echo "${ids[*]}"
    }
    J=$(submit_chunk "" "")
    for (( i = 1; i <= N; i++ )); do
      echo "chunk $i ($LAYOUT): $J"
      DEP="--dependency=afterany:${J// /:}"
      D=$(TICA_DRAIN_QUALITY="${ARMS[*]}" sbatch --parsable --account="$ACCT" --time=16:00:00 $DEP -J v4b-drain \
          scripts/slurm/tica_val_drain.sh "$QUEUE")
      echo "  drain + quality checks: $D"
      (( i < N )) && J=$(submit_chunk "resume:" "$DEP")
    done ;;
  quality)
    AD_A=${2:?adapter of final_v4b_4b}; AD_B=${3:?adapter of final_v4b_score_4b}
    Q=$RUNS/v4b_quality; mkdir -p "$Q/gens"
    # checkable concepts (scripts/tica_quality_check.py): format + language families, train and held-out
    TR="German language,Japanese language,Arabic language,Hindi language,all-caps text,SQL code,JSON data,JavaScript code"
    HO="Finnish language,French language,Portuguese language,Turkish language,Russian language,Greek language,Thai language,Tamil language,HTML markup,Python code,Java code"
    COMMON="--answers free --ask-concept --episodes 32 --turns 24 --switch-prob 1.0 --turn-budget 5,8,8"
    EX=$QUEUE.extra
    for split in train heldout; do
      T=$TR; [ $split = heldout ] && T=$HO
      for who in v4b:${ARMS[0]}:$AD_A score:${ARMS[1]}:$AD_B base:${ARMS[0]}:-; do
        IFS=: read -r tag cfg ad <<< "$who"
        sel="--adapter $ad"; [ "$ad" = - ] && sel="--base"
        for cond in real_score nobrief; do
          n=${tag}_${split}_${cond}
          printf 'bash scripts/slurm/tica_switch_eval.sh %s %q %s --topics %q --cond %s --gen-out %q %s\n' \
            "$cfg" "$Q/race_$n.json" "$sel" "$T" "$cond" "$Q/gens/$n.jsonl" "$COMMON" >> "$EX"
        done
      done
    done
    echo "queued $(wc -l < "$EX") quality races in $EX"
    D=$(sbatch --parsable --account="$ACCT" --time=12:00:00 -J v4b-quality scripts/slurm/tica_val_drain.sh "$QUEUE")
    echo "drain: $D; then: sbatch --dependency=afterok:$D scripts/slurm/tamia_quality_summary.sh $Q" ;;
  status)
    squeue -u "$USER" -o "%.10i %.12j %.2t %.10M %.10L %R" || true
    for a in "${ARMS[@]}"; do
      echo "== $a"
      for d in "$RUNS/$a"/adaptive "$RUNS/$a"/resume_*/adaptive; do
        [ -f "$d/train_log.jsonl" ] || continue
        echo "  $d: $(wc -l < "$d/train_log.jsonl") steps logged; newest ckpt $(ls -d "$d"/step_* 2>/dev/null | sort -V | tail -1)"
        tail -1 "$d/train_log.jsonl" | tr ',' '\n' | grep -E '"(step|sec|max_mem_gb|loss|pass_rate|cap_gen_acc|kl|grad_norm_hide|grad_norm_rest|train/stay_term|train/hide_term|dp/sec_rollout_max|dp/sec_rollout_min)"' | tr '\n' ' '; echo
      done
      ls "$RUNS/$a/val" 2>/dev/null | grep -c '^step_' | sed 's/^/  held-out races done: /' || true
    done ;;
  *)
    sed -n 2,16p "$0"; exit 2 ;;
esac
