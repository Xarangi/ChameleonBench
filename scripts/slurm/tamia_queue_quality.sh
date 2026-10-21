#!/bin/bash
# Print the in-run QUALITY-CHECK tasks (one shell line each) for the newest checkpoint of every config given, for
# scripts/slurm/tica_val_drain.sh (TICA_DRAIN_QUALITY="CFG ..."; TamIA 10-21, user: quality checks during the runs,
# "to get the keep capability"). Per checkpoint, into <cfg.out>/quality/step_<global step>/:
#   race_{train,heldout}.json + gens/*.jsonl   free-generation race (monitored, --ask-concept) on the CHECKABLE
#                                              concepts (format + language; scripts/tica_quality_check.py)
#   panel_{train,heldout}.json                 never-shown probe panel (tica_probe_panel.py: fresh base probes of
#                                              every kind + every refit + linear readability; selectivity mix)
#   capability.json                            tica_capability.py: GSM8K (200) + MMLU (600), clean and under the
#                                              brief naming a train (mathematical reasoning) / held-out (HTML) concept
# The base model's references are made once, into runs/v4b_quality_base/. Done checkpoints are skipped; the drain
# writes quality.json (aggregates) after its tasks. Login-safe: ls / bash only.
#   bash scripts/slurm/tamia_queue_quality.sh final_v4b_tamia_4b final_v4b_score_tamia_4b
set -uo pipefail
RUNS=/scratch/s/sneheel/next_chameleons_artifacts/tica/runs
TR="German language,Japanese language,Arabic language,Hindi language,all-caps text,SQL code,JSON data,JavaScript code"
HO="Finnish language,French language,Portuguese language,Turkish language,Russian language,Greek language,Thai language,Tamil language,HTML markup,Python code,Java code"
RACE="--answers free --ask-concept --episodes 32 --turns 24 --switch-prob 1.0 --turn-budget 5,8,8 --cond real_score"
PANEL="--episodes 24 --turns 24 --mix real_score:0.6,mismatch:0.2,nobrief:0.2"
CAP="--benches gsm8k,mmlu --briefs mathematical\ reasoning,HTML\ markup"

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
  [ -f "$q/capability.json" ] || printf 'bash scripts/slurm/tica_capability.sh %s %q %s %s\n' "$cfg" "$q/capability.json" "$sel" "$CAP"
  # never-shown PROBE PANEL (tica_probe_panel.py, Narval 10-18h / 10-21w): every turn also scored by fresh base
  # probes of every kind (linear @4..30 + zoo kinds), every refit, and per-turn linear readability; mismatch /
  # no-brief episodes give selectivity
  for split in train heldout; do
    [ -f "$q/panel_$split.json" ] || printf 'bash scripts/slurm/tica_probe_panel.sh %s %q %s --topics %s %s\n' \
      "$cfg" "$q/panel_$split.json" "$sel" "$split" "$PANEL"
  done
}

first=""
for cfg in "$@"; do
  [ -n "$first" ] || first=$cfg
  read -r step dir < <(newest "$RUNS/$cfg")
  [ -n "$dir" ] || continue
  emit "$cfg" "$cfg" "--adapter $dir" "$RUNS/$cfg/quality/step_$step"
done
[ -n "$first" ] && emit base "$first" "--base" "$RUNS/v4b_quality_base"
