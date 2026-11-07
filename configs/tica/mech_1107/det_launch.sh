#!/bin/bash
# 11-07 (user: "run both supervised + unsupervised hiding detectors on the models"): the 10-25/26 detector suite
# (mech_1025: panel dumps -> offline evasion detector / selectivity / OOD; adaptive evdetsel races) on the v4d
# checkpoints strict1800 (sx18), strict2500 (sx25), KL0.3 1800 (k18), base. Eval only.
set -uo pipefail
cd /scratch/s/sneheel/ChameleonBench
R=/scratch/s/sneheel/next_chameleons_artifacts/tica/runs; N=$R/mech_1107; F=$N/feats
declare -A CFG=([sx18]=final_v4d_klx_tamia_4b [sx25]=final_v4d_klx_tamia_4b_x2500 [k18]=final_v4d_k03_tamia_4b)
declare -A AD=([sx18]=$R/final_v4d_klx_tamia_4b/resume_1280/adaptive/step_520
               [sx25]=$R/final_v4d_klx_tamia_4b_x2500/resume_2100/adaptive/step_400
               [k18]=$R/final_v4d_k03_tamia_4b/resume_1660/adaptive/step_140)
FL=2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,31
T1=$N/dump.tsv; S1=$N/dump_smoke.tsv; T2=$N/race.tsv; S2=$N/race_smoke.tsv; : > $T1; : > $S1; : > $T2; : > $S2
for m in base sx18 sx25 k18; do
  if [ $m = base ]; then c=final_v4d_klx_tamia_4b; sel="--base"; else c=${CFG[$m]}; sel="--adapter ${AD[$m]}"; fi
  for sp in train heldout; do for s in 1 2; do
    o=$F/${m}_${sp}_s$s
    printf 'panel\t%s\tbash scripts/slurm/tica_probe_panel.sh %s %s %s --topics %s --seed %s --episodes 6 --batch 6 --turns 24 --mix real_score:0.6,mismatch:0.2,nobrief:0.2 --dump-feats %s --feat-layers %s\n' \
      "$o.json" $c "$o.json" "$sel" $sp $s "$o.pt" $FL >> $T1
  done; done
done
printf 'panel\t%s\tbash scripts/slurm/tica_probe_panel.sh %s %s --base --topics heldout --seed 9 --episodes 1 --batch 1 --turns 3 --mix real_score:1.0 --dump-feats %s --feat-layers 12,16\n' \
  "$N/smoke_panel.json" final_v4d_klx_tamia_4b "$N/smoke_panel.json" "$N/smoke_panel.pt" >> $S1
# adaptive selective detector races (supervised evdetsel; seeds = the model's own TRAIN-concept dumps vs base)
for m in sx18 sx25 k18; do for L in 16 22; do
  o=$N/evdet/${m}_evdetsel_$L.json
  printf 'sel\t%s\tbash scripts/slurm/tica_switch_eval.sh %s %s --adapter %s --kind evdetsel@%s --topics heldout --refit-switch 0 --turn-budget 1,40,8 --episodes 16 --batch 2 --turns 40 --switch-prob 0 --cond real_score --max-k 12 --evdet-seed %s\n' \
    "$o" ${CFG[$m]} "$o" ${AD[$m]} $L "'$F/${m}_train_s1.pt,$F/${m}_train_s2.pt;$F/base_train_s1.pt,$F/base_train_s2.pt'" >> $T2
done; done
o=$N/smoke_evdetsel.json
printf 'sel\t%s\tbash scripts/slurm/tica_switch_eval.sh %s %s --adapter %s --kind evdetsel@16 --topics heldout --refit-switch 0 --turn-budget 1,4,2 --episodes 1 --batch 1 --turns 4 --switch-prob 0 --cond real_score --max-k 3 --evdet-seed %s\n' \
  "$o" ${CFG[sx25]} "$o" ${AD[sx25]} "'$F/sx25_train_s1.pt,$F/sx25_train_s2.pt;$F/base_train_s1.pt,$F/base_train_s2.pt'" >> $S2
wc -l $T1 $T2
[ "${DRY:-0}" = 1 ] && exit 0
EX=ALL,NG=4,SLOTS=8,VLLM_CACHE_ROOT=/scratch/s/sneheel/vllm_cache
J1=$(sbatch --parsable --account=aip-rrabba --exclude=tg11105 --time=4:00:00 --job-name=det-dump --export=$EX scripts/slurm/tamia_mechanism_node.sh $S1 $T1)
J2=$(sbatch --parsable --account=aip-rrabba --exclude=tg11105 --time=6:00:00 --job-name=det-race --dependency=afterok:$J1 --export=$EX scripts/slurm/tamia_mechanism_node.sh $S2 $T2)
J3=$(sbatch --parsable --account=aip-rrabba --dependency=afterok:$J1 $N/det_offline.sh)
echo "dump $J1 race $J2 offline $J3"
