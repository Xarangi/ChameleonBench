#!/bin/bash
# 11-09 FINAL evals v2 (user: full suite, validity measures on free-gen + hiding, add hide-and-solve). Training done:
# 9B KL0.3 @1800 (k9b), detector-trained 4B @2200 (d22; evaluated with the PARENT config final_v4d_k03_tamia_4b = same
# eval settings as k18). Pass validity: every FREE-ANSWER race (quality races + all gated task races, races/final_spec.json)
# runs with the 6-of-8 pass gate (a pass counts only if >= 6 of 8 answers re-sampled from the same prefix also pass) and the
# 8-fork fluke panel; code = ChameleonBench-final (graded-tasks branch + 9B config; the gate is not on tamia). Fixed-text
# races (held-out val race, detector races) have nothing to re-sample -> no gate (rule since 11-05). Persistence races
# (--no-refit / --always-pass) keep their old form (they measure whether a pass holds, not whether it counts).
# The k18 quality races (11-04) were ungated -> rerun gated here for a like-for-like comparison.
set -uo pipefail
cd /scratch/s/sneheel/ChameleonBench
R=/scratch/s/sneheel/next_chameleons_artifacts/tica/runs; M=$R/mech_1109; mkdir -p $M; F8=$R/mech_1108/feats; F7=$R/mech_1107/feats
FW=/scratch/s/sneheel/ChameleonBench-final
C4=final_v4d_k03_tamia_4b; D=$R/final_v4d_k03det_tamia_4b; AD=$D/resume_2050/adaptive/step_150
A18=$R/$C4/resume_1660/adaptive/step_140
C9=final_v4d_k03_tamia_9b; A9=$R/$C9/resume_1660/adaptive/step_140
for a in $AD $A9 $A18; do test -f $a/adapter_model.safetensors || { echo "no adapter $a"; exit 1; }; done
source scripts/slurm/tamia_queue_quality.sh   # emit / emit_panel, TR / HO / RACE / PERSIST / CAP (no args -> prints nothing)
VAL="configs/tica/final_split_v5.json --topics heldout --episodes 32 --switch-prob 1.0 --turn-budget 5,8,8"
GATE="--fork-panel 8 --fast-forks --pass-gate 8 --pass-gate-min 6"
L4="--layers 4,8,12,16,20,22,24,26,28,30 --train-layers 8,12,16,22,26"
add() { printf '%s\t%s\t%s\n' "$2" "$3" "$4" >> "$1"; }   # FILE GROUP OUT CMD
gated_q() {  # FILE CFG ADAPTER QDIR: gated free-gen quality races, train + held-out concepts
  local f=$1 cfg=$2 ad=$3 q=$4 sp t; mkdir -p $q/gens
  for sp in train heldout; do t=$TR; [ $sp = heldout ] && t=$HO
    add $f q $q/race_${sp}_gated.json "TICA_REPO=$FW bash $FW/scripts/slurm/tica_switch_eval.sh $cfg $q/race_${sp}_gated.json --adapter $ad --topics '$t' --gen-out $q/gens/gated_$sp.jsonl $RACE $GATE"
  done
}
hs() {  # FILE CFG TAG SEL QDIR
  local f=$1 cfg=$2 tag=$3 sel=$4 q=$5 task c sp o; mkdir -p $q
  for task in gsm8k mbpp jseval sqltest jsontest; do o=$q/${tag}_$task.json
    add $f hs $o "bash scripts/slurm/tica_hide_solve.sh $cfg $o $sel --task $task --gsm-n 200 $L4"; done
  for c in "SQL code" "JSON data" "JavaScript code" "German language" "Japanese language" "Arabic language" "Hindi language" "all-caps text"; do
    for sp in own held; do o=$q/${tag}_capfree_$(echo "$c" | tr ' A-Z' '_a-z')_$sp.json
      add $f hs $o "bash scripts/slurm/tica_hide_solve.sh $cfg $o $sel --task capfree --concept \"$c\" --pool-split $sp --gsm-n 100 $L4"; done
  done
}
qline() { while IFS= read -r cmd; do add $1 rl "$(echo "$cmd" | awk '{print $4}')" "$cmd"; done; }
# ---- node 1: 9B @1800
T9=$M/k9b.tsv; S9=$M/k9b_smoke.tsv; : > $T9; : > $S9; Q9=$R/$C9/quality/step_1800
add $S9 rl $M/smoke_9b_race.json "bash scripts/slurm/tica_switch_eval.sh $C9 $M/smoke_9b_race.json --adapter $A9 --topics heldout --episodes 1 --batch 1 --turns 3 --switch-prob 1.0 --turn-budget 1,2,2 --cond real_score"
add $S9 q $M/smoke_9b_gated.json "TICA_REPO=$FW bash $FW/scripts/slurm/tica_switch_eval.sh $C9 $M/smoke_9b_gated.json --adapter $A9 --topics 'SQL code' --batch 2 --answers free --ask-concept --episodes 2 --turns 4 --switch-prob 1.0 --turn-budget 1,2,2 --cond real_score $GATE"
add $S9 hs $M/smoke_9b_hs.json "bash scripts/slurm/tica_hide_solve.sh $C9 $M/smoke_9b_hs.json --adapter $A9 --task capfree --concept 'SQL code' --pool-split held --gsm-n 4 --batch 4 $L4"
add $T9 rl $R/$C9/val/step_1800.json "bash scripts/slurm/tica_val.sh $C9 $A9 1800 $R/$C9/val $VAL"
gated_q $T9 $C9 $A9 $Q9
{ emit $C9 $C9 "--adapter $A9" $Q9 | grep -v "race_train.json\|race_heldout.json"; emit_panel $C9 "--adapter $A9" $Q9
  emit base $C9 "--base" $R/v4b_quality_base_9b | grep tica_capability; } | qline $T9
hs $T9 $C9 p "--adapter $A9" $M/k9b_1800/hs; hs $T9 $C9 b "--base" $M/k9b_1800/hs
# ---- node 2: d22 battery + audit dumps + k18 gated quality rerun
TD=$M/d22.tsv; SD=$M/d22_smoke.tsv; : > $TD; : > $SD; QD=$D/quality/step_2200
add $SD panel $R/mech_1108/smoke_panel.json -
add $SD rl $R/mech_1025/klx4b_x2500_2500/hs/smoke_jsontest.json -
add $SD hs $R/mech_1025/klx4b_x2500_2500/hs/smoke_capfree.json -
add $SD q $M/smoke_d22_gated.json "TICA_REPO=$FW bash $FW/scripts/slurm/tica_switch_eval.sh $C4 $M/smoke_d22_gated.json --adapter $AD --topics 'SQL code' --batch 2 --answers free --ask-concept --episodes 2 --turns 4 --switch-prob 1.0 --turn-budget 1,2,2 --cond real_score $GATE"
for sp in train heldout; do for s in 1 2; do o=$F8/d22_${sp}_s$s
  add $TD panel $o.json "bash scripts/slurm/tica_probe_panel.sh $C4 $o.json --adapter $AD --topics $sp --seed $s --episodes 6 --batch 6 --turns 24 --mix real_score:0.6,mismatch:0.2,nobrief:0.2 --dump-feats $o.pt --feat-layers 2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,31"
done; done
add $TD rl $D/val/step_2200.json "bash scripts/slurm/tica_val.sh $C4 $AD 2200 $D/val $VAL"
gated_q $TD $C4 $AD $QD
gated_q $TD $C4 $A18 $R/$C4/quality/step_1800
emit $C4 $C4 "--adapter $AD" $QD | grep -v "race_train.json\|race_heldout.json" | qline $TD
hs $TD $C4 p "--adapter $AD" $M/d22_2200/hs
for f in $R/mech_1025/k034b_1800/hs/b_*.json; do ln -sf "$f" $M/d22_2200/hs/; done   # 4B base halves (adapter-free)
# ---- node 3: d22 detector races (fixed dataset texts -> no gate), after node 2's dumps
TR2=$M/d22_race.tsv; SR=$M/d22_race_smoke.tsv; : > $TR2; add $SR sel $R/mech_1107/smoke_evdetsel.json -
for L in 16 22; do o=$M/evdet/d22_evdetsel_$L.json
  add $TR2 sel $o "bash scripts/slurm/tica_switch_eval.sh $C4 $o --adapter $AD --kind evdetsel@$L --topics heldout --refit-switch 0 --turn-budget 1,40,8 --episodes 16 --batch 2 --turns 40 --switch-prob 0 --cond real_score --max-k 12 --evdet-seed '$F8/d22_train_s1.pt,$F8/d22_train_s2.pt;$F7/base_train_s1.pt,$F7/base_train_s2.pt'"
done
wc -l $T9 $TD $TR2
[ "${DRY:-0}" = 1 ] && exit 0
EX=ALL,NG=4,SLOTS=8,VLLM_CACHE_ROOT=/scratch/s/sneheel/vllm_cache
J9=$(sbatch --parsable --account=aip-rrabba --exclude=tg11105 --time=14:00:00 --job-name=k9b-eval --export=$EX scripts/slurm/tamia_mechanism_node.sh $S9 $T9)
JD=$(sbatch --parsable --account=aip-rrabba --exclude=tg11105 --time=12:00:00 --job-name=d22-eval --export=$EX scripts/slurm/tamia_mechanism_node.sh $SD $TD)
JR=$(sbatch --parsable --account=aip-rrabba --exclude=tg11105 --time=4:00:00 --job-name=d22-race --dependency=afterany:$JD --export=$EX scripts/slurm/tamia_mechanism_node.sh $SR $TR2)
JO=$(sbatch --parsable --account=aip-rrabba --exclude=tg11105 --dependency=afterany:$JD configs/tica/mech_1109/det_offline_d22.sh)
echo "9B $J9 | d22 battery+dump $JD | d22 races $JR | offline $JO"
