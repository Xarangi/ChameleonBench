#!/bin/bash
# Print the in-run QUALITY-CHECK tasks (one shell line each) for the newest checkpoint of every config given, for
# scripts/slurm/tica_val_drain.sh (TICA_DRAIN_QUALITY="CFG ..."; TamIA 10-21, user: quality checks during the runs,
# "to get the keep capability"). Per checkpoint, into <cfg.out>/quality/step_<global step>/:
#   race_{train,heldout}.json + gens/*.jsonl   free-generation race (monitored, --ask-concept) on the CHECKABLE
#                                              concepts (format + language; scripts/tica_quality_check.py)
#   panel_{train,heldout}.json (every PANEL_EVERY=300 steps) never-shown probe panel (tica_probe_panel.py: fresh base probes of
#                                              every kind + every refit + linear readability; selectivity mix)
#   persist_{train,pass_train}.json            races with --no-refit (+ --always-pass): are passes KEPT? (10-22)
#   capability.json                            tica_capability.py: GSM8K (200) + MMLU (600), clean and under the
#                                              brief naming a train (mathematical reasoning) / held-out (HTML) concept
# The base model's references are made once, into runs/v4b_quality_base/. Done checkpoints are skipped; the drain
# writes quality.json (aggregates) after its tasks. Login-safe: ls / bash only.
#   bash scripts/slurm/tamia_queue_quality.sh final_v4b_tamia_4b final_v4b_score_tamia_4b
set -uo pipefail
RUNS=/scratch/s/sneheel/next_chameleons_artifacts/tica/runs
TR="German language,Japanese language,Arabic language,Hindi language,all-caps text,SQL code,JSON data,JavaScript code"
HO="Finnish language,French language,Portuguese language,Turkish language,Russian language,Greek language,Thai language,Tamil language,HTML markup,Python code,Java code"
RACE="--batch 16 --answers free --ask-concept --episodes 32 --turns 24 --switch-prob 1.0 --turn-budget 5,8,8 --cond real_score"
PANEL_EVERY=${PANEL_EVERY:-300}
PANEL_SHARDS=${PANEL_SHARDS:-4}; PANEL_EPISODES=${PANEL_EPISODES:-24}
PANEL="--batch 6 --turns 24 --mix real_score:0.6,mismatch:0.2,nobrief:0.2"
# PERSISTENCE races (10-22, user: is a pass kept, not luck?): no refit after a pass, the beaten probe keeps judging
# (persist_*); "pass" = verdict always PASSED (does the dodge hold without corrective feedback?). Train concepts.
PERSIST="--batch 16 --answers free --ask-concept --episodes 32 --turns 16 --switch-prob 0 --cond real_score --no-refit"
CAP="--benches gsm8k,mmlu --gsm-max-new-tokens 512 --briefs mathematical\ reasoning,HTML\ markup"

newest() {  # newest checkpoint over the first chunk and resume chunks: "<global step> <dir>"
  local out=$1 best=0 bdir="" d k off
  for d in "$out"/adaptive/step_*; do
    [ -d "$d" ] || continue; k=${d##*step_}; (( k > best )) && { best=$k; bdir=$d; }
  done
  for d in "$out"/resume_*/adaptive/step_*; do
    [ -d "$d" ] || continue; off=${d#"$out"/resume_}; off=${off%%/*}; k=$(( off + ${d##*step_} ))
    (( k > best )) && { best=$k; bdir=$d; }
  done
  echo "$best $bdir"
}

emit() {  # TAG CFG SEL QDIR
  local cfg=$2 sel=$3 q=$4
  mkdir -p "$q/gens"
  for split in train heldout; do
    local t=$TR; [ "$split" = heldout ] && t=$HO
    [ -f "$q/race_$split.json" ] || printf 'bash scripts/slurm/tica_switch_eval.sh %s %q %s --topics %q --gen-out %q %s\n' \
      "$cfg" "$q/race_$split.json" "$sel" "$t" "$q/gens/${1}_$split.jsonl" "$RACE"
  done
  [ -f "$q/persist_train.json" ] || printf 'bash scripts/slurm/tica_switch_eval.sh %s %q %s --topics %q %s\n' \
    "$cfg" "$q/persist_train.json" "$sel" "$TR" "$PERSIST"
  [ -f "$q/persist_pass_train.json" ] || printf 'bash scripts/slurm/tica_switch_eval.sh %s %q %s --topics %q %s --always-pass\n' \
    "$cfg" "$q/persist_pass_train.json" "$sel" "$TR" "$PERSIST"
  [ -f "$q/capability.json" ] || printf 'bash scripts/slurm/tica_capability.sh %s %q %s %s\n' "$cfg" "$q/capability.json" "$sel" "$CAP"
}

emit_panel() {  # CFG SEL QDIR: never-shown PROBE PANEL (tica_probe_panel.py, Narval 10-18h / 10-21w): every turn
  # also scored by fresh base probes of every kind (linear @4..30 + zoo kinds), every refit, and per-turn linear
  # readability; mismatch / no-brief episodes give selectivity. Eval only (never during training).
  # TamIA 10-22: each panel runs as PANEL_SHARDS processes (--seed k, PANEL_EPISODES / PANEL_SHARDS episodes each)
  # so one 1-2.5 h panel does not leave the node idle; the drain merges them (scripts/tica_panel_merge.py).
  local cfg=$1 sel=$2 q=$3 k
  mkdir -p "$q"
  for split in train heldout; do
    [ -f "$q/panel_$split.json" ] && continue
    for (( k = 0; k < PANEL_SHARDS; k++ )); do
      [ -f "$q/panel_$split.s$k.json" ] || printf 'bash scripts/slurm/tica_probe_panel.sh %s %q %s --topics %s --seed %d --episodes %d %s\n' \
        "$cfg" "$q/panel_$split.s$k.json" "$sel" "$split" $(( k + 1 )) $(( PANEL_EPISODES / PANEL_SHARDS )) "$PANEL"
    done
  done
}

ckpt_at() {  # OUT GLOBAL_STEP -> checkpoint dir of that global step ("" if not saved)
  local out=$1 g=$2 d off
  [ -d "$out/adaptive/step_$g" ] && { echo "$out/adaptive/step_$g"; return; }
  for d in "$out"/resume_*; do
    [ -d "$d" ] || continue; off=${d##*resume_}
    (( g > off )) && [ -d "$d/adaptive/step_$(( g - off ))" ] && { echo "$d/adaptive/step_$(( g - off ))"; return; }
  done
}

first=""
for cfg in "$@"; do
  [ -n "$first" ] || first=$cfg
  read -r step dir < <(newest "$RUNS/$cfg")
  [ -n "$dir" ] || continue
  emit "$cfg" "$cfg" "--adapter $dir" "$RUNS/$cfg/quality/step_$step"
  # probe panel only every PANEL_EVERY global steps (default 300: steps 300 / 600 / 900 / 1200), not every chunk
  for (( g = PANEL_EVERY; g <= step; g += PANEL_EVERY )); do
    pd=$(ckpt_at "$RUNS/$cfg" "$g")
    [ -n "$pd" ] && emit_panel "$cfg" "--adapter $pd" "$RUNS/$cfg/quality/step_$g"
  done
done
[ -n "$first" ] && emit base "$first" "--base" "$RUNS/v4b_quality_base"
[ -n "$first" ] && emit_panel "$first" "--base" "$RUNS/v4b_quality_base"
