#!/bin/bash
# 11-10 rerun3. rerun2's nodes (682954/682955) passed their smokes but ran ONLY group q: the mechanism node
# runs phase 2 only for groups whose smoke passed, and the rerun smoke TSVs had just the q line, so rl/hs/panel
# were silently skipped (node still reported DONE). k18 gated q (2 per GPU at SLOTS=8) OOMed next to det q.
# Fix: smoke TSVs list every group (rl/hs/panel lines point at existing smoke outputs -> pass without rerunning);
# k18 gated q moved to the 9B node (SLOTS=4 = one process per GPU).
set -euo pipefail
cd /scratch/s/sneheel/ChameleonBench
R=/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/mech_1109; X=tg11105,tg10907,tg10807
cd $R
grep -P '^(rl|hs|panel)\t' k9b_smoke.tsv > k9b_rr2_smoke.tsv; cat k9b_rr_smoke.tsv >> k9b_rr2_smoke.tsv
{ grep -P '^q\t' d22_rr.tsv | grep final_v4d_k03_tamia_4b/quality/step_1800; cat k9b_rr.tsv; } > k9b_rr2.tsv
grep -P '^(rl|hs)\t' d22_smoke.tsv > d22_rr2_smoke.tsv
printf 'panel\t%s\ttrue\n' /scratch/s/sneheel/next_chameleons_artifacts/tica/runs/mech_1108/smoke_panel.json >> d22_rr2_smoke.tsv
grep -v final_v4d_k03_tamia_4b/quality/step_1800 d22_rr.tsv > d22_rr2.tsv
cd /scratch/s/sneheel/ChameleonBench
E="--export=ALL,NG=4,VLLM_CACHE_ROOT=/scratch/s/sneheel/vllm_cache"
A=$(sbatch --parsable --account=aip-rrabba --exclude=$X --time=12:00:00 --job-name=k9b-eval-rr2 $E,SLOTS=4 scripts/slurm/tamia_mechanism_node.sh $R/k9b_rr2_smoke.tsv $R/k9b_rr2.tsv)   # 683600
B=$(sbatch --parsable --account=aip-rrabba --exclude=$X --time=8:00:00 --job-name=d22-eval-rr2 $E,SLOTS=8 scripts/slurm/tamia_mechanism_node.sh $R/d22_rr2_smoke.tsv $R/d22_rr2.tsv)   # 683601
sbatch --account=aip-rrabba --exclude=$X --time=4:00:00 --job-name=d22-race --dependency=afterany:$B $E,SLOTS=8 scripts/slurm/tamia_mechanism_node.sh $R/d22_race_smoke.tsv $R/d22_race.tsv   # 683602
sbatch --account=aip-rrabba --exclude=$X --dependency=afterany:$B configs/tica/mech_1109/det_offline_d22.sh   # 683603
sbatch --exclude=$X --dependency=afterany:$A configs/tica/mech_1109/score9b/race-score-final9b-javaeval.sh   # 683604
