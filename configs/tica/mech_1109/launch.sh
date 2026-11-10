#!/bin/bash
# 11-09 FINAL evals (training done: 9B KL0.3 @1800, detector-trained 4B @2200). Same battery as k034b_drain.sh (KL0.3 4B @1800)
# minus passive hide-and-solve (user has not decided whether to keep it): held-out FIXED-TEXT val race, free-answer quality
# races train/held-out, persistence x2, capability (GSM8K/MMLU clean + under brief). 9B also: never-shown panel @1800 and a
# 9B BASE capability control (runs/v4b_quality_base_9b; the 4B base refs are not a 9B control).
# Detector 4B (d22) evaluated with the PARENT config final_v4d_k03_tamia_4b + its adapter (the det config only adds
# training-time detector settings; same eval settings as k18 -> apples to apples; main checkout, as the d60 dump).
# Det audit (as 11-08 on d60): panel feature dumps d22 -> (a) adaptive evdetsel races @16/@22 held-out, 16 eps x 40 turns
# (same args as k18's 674419), (b) offline suite + frozen transfer (CPU).
set -uo pipefail
cd /scratch/s/sneheel/ChameleonBench
R=/scratch/s/sneheel/next_chameleons_artifacts/tica/runs; M=$R/mech_1109; mkdir -p $M; F8=$R/mech_1108/feats; F7=$R/mech_1107/feats
C4=final_v4d_k03_tamia_4b; D=$R/final_v4d_k03det_tamia_4b; AD=$D/resume_2050/adaptive/step_150
C9=final_v4d_k03_tamia_9b; A9=$R/$C9/resume_1660/adaptive/step_140
for a in $AD $A9; do test -f $a/adapter_model.safetensors || { echo "no adapter $a"; exit 1; }; done
source scripts/slurm/tamia_queue_quality.sh   # defines emit / emit_panel (no args -> prints nothing)
VAL="configs/tica/final_split_v5.json --topics heldout --episodes 32 --switch-prob 1.0 --turn-budget 5,8,8"
# ---- node 1: 9B
T9=$M/k9b.tsv; S9=$M/k9b_smoke.tsv; : > $T9; : > $S9
printf 'rl\t%s\tbash scripts/slurm/tica_switch_eval.sh %s %s --adapter %s --topics heldout --episodes 1 --batch 1 --turns 3 --switch-prob 1.0 --turn-budget 1,2,2 --cond real_score\n' \
  "$M/smoke_9b_race.json" $C9 "$M/smoke_9b_race.json" $A9 >> $S9
printf 'rl\t%s\tbash scripts/slurm/tica_val.sh %s %s 1800 %s %s\n' "$R/$C9/val/step_1800.json" $C9 $A9 "$R/$C9/val" "$VAL" >> $T9
{ emit $C9 $C9 "--adapter $A9" $R/$C9/quality/step_1800; emit_panel $C9 "--adapter $A9" $R/$C9/quality/step_1800
  emit base $C9 "--base" $R/v4b_quality_base_9b | grep tica_capability; } | while IFS= read -r cmd; do
  printf 'rl\t%s\t%s\n' "$(echo "$cmd" | awk '{print $4}')" "$cmd" >> $T9; done
# ---- node 2: detector 4B @2200 battery + audit dumps, then races
TD=$M/d22.tsv; SD=$M/d22_smoke.tsv; : > $TD
printf 'panel\t%s\t-\nrl\t%s\t-\n' $R/mech_1108/smoke_panel.json $R/mech_1025/klx4b_x2500_2500/hs/smoke_jsontest.json > $SD
for sp in train heldout; do for s in 1 2; do o=$F8/d22_${sp}_s$s
  printf 'panel\t%s\tbash scripts/slurm/tica_probe_panel.sh %s %s --adapter %s --topics %s --seed %s --episodes 6 --batch 6 --turns 24 --mix real_score:0.6,mismatch:0.2,nobrief:0.2 --dump-feats %s --feat-layers 2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,31\n' \
    "$o.json" $C4 "$o.json" $AD $sp $s "$o.pt" >> $TD; done; done
printf 'rl\t%s\tbash scripts/slurm/tica_val.sh %s %s 2200 %s %s\n' "$D/val/step_2200.json" $C4 $AD "$D/val" "$VAL" >> $TD
emit $C4 $C4 "--adapter $AD" $D/quality/step_2200 | while IFS= read -r cmd; do
  printf 'rl\t%s\t%s\n' "$(echo "$cmd" | awk '{print $4}')" "$cmd" >> $TD; done
TR=$M/d22_race.tsv; SR=$M/d22_race_smoke.tsv; : > $TR; printf 'sel\t%s\t-\n' $R/mech_1107/smoke_evdetsel.json > $SR
for L in 16 22; do o=$M/evdet/d22_evdetsel_$L.json
  printf 'sel\t%s\tbash scripts/slurm/tica_switch_eval.sh %s %s --adapter %s --kind evdetsel@%s --topics heldout --refit-switch 0 --turn-budget 1,40,8 --episodes 16 --batch 2 --turns 40 --switch-prob 0 --cond real_score --max-k 12 --evdet-seed %s\n' \
    "$o" $C4 "$o" $AD $L "'$F8/d22_train_s1.pt,$F8/d22_train_s2.pt;$F7/base_train_s1.pt,$F7/base_train_s2.pt'" >> $TR; done
wc -l $T9 $TD $TR
[ "${DRY:-0}" = 1 ] && exit 0
EX=ALL,NG=4,SLOTS=8,VLLM_CACHE_ROOT=/scratch/s/sneheel/vllm_cache
J9=$(sbatch --parsable --account=aip-rrabba --exclude=tg11105 --time=10:00:00 --job-name=k9b-eval --export=$EX scripts/slurm/tamia_mechanism_node.sh $S9 $T9)
JD=$(sbatch --parsable --account=aip-rrabba --exclude=tg11105 --time=8:00:00 --job-name=d22-eval --export=$EX scripts/slurm/tamia_mechanism_node.sh $SD $TD)
JR=$(sbatch --parsable --account=aip-rrabba --exclude=tg11105 --time=4:00:00 --job-name=d22-race --dependency=afterok:$JD --export=$EX scripts/slurm/tamia_mechanism_node.sh $SR $TR)
JO=$(sbatch --parsable --account=aip-rrabba --exclude=tg11105 --dependency=afterok:$JD configs/tica/mech_1109/det_offline_d22.sh)
echo "9B $J9 | d22 battery+dump $JD | d22 races $JR | offline $JO"
