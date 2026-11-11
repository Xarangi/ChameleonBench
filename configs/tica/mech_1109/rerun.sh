#!/bin/bash
# 11-10 rerun of mech_1109 v2 failures (only MISSING outputs; finished ones are kept):
#  - gated-quality smoke used 1 concept with --switch-prob 1.0 (sampler needs >=2) -> smoke now 2 concepts
#  - k9b node: 8 slots = two 9B procs per GPU -> CUDA OOM (val, persist_pass, panels, base cap, gated q) -> SLOTS=4
#  - d22 node tg10907: /tmp Input/output errors (panel dumps, val, capfree hs) -> excluded
#  - race-final4b 681356 on tg10807: vLLM compile cache on /home (transport endpoint shutdown) -> VLLM_CACHE_ROOT on scratch
#  - panel_heldout.s0 (9B): deterministic TypeError in panel.py _mean (None readability) -> not rerun (3 held-out seeds)
set -euo pipefail
cd /scratch/s/sneheel/ChameleonBench
R=/scratch/s/sneheel/next_chameleons_artifacts/tica/runs; M=$R/mech_1109; FW=/scratch/s/sneheel/ChameleonBench-final
X=tg11105,tg10907,tg10807
GATE="--fork-panel 8 --fast-forks --pass-gate 8 --pass-gate-min 6"
miss() { awk -F'\t' -v skip="$2" '{ c="test -s \"" $2 "\""; if (system(c) != 0 && (skip == "" || index($2, skip) == 0)) print }' "$1"; }
qsmoke() { printf 'q\t%s\t%s\n' "$3" "TICA_REPO=$FW bash $FW/scripts/slurm/tica_switch_eval.sh $1 $3 --adapter $2 --topics 'SQL code,JSON data' --batch 2 --answers free --ask-concept --episodes 2 --turns 4 --switch-prob 1.0 --turn-budget 1,2,2 --cond real_score $GATE"; }
C9=final_v4d_k03_tamia_9b; A9=$R/$C9/resume_1660/adaptive/step_140
C4=final_v4d_k03det_tamia_4b; AD=$R/$C4/resume_2050/adaptive/step_150
miss $M/k9b.tsv panel_heldout.s0 > $M/k9b_rr.tsv; qsmoke $C9 $A9 $M/smoke_9b_gated2.json > $M/k9b_rr_smoke.tsv
miss $M/d22.tsv "" > $M/d22_rr.tsv; qsmoke $C4 $AD $M/smoke_d22_gated2.json > $M/d22_rr_smoke.tsv
wc -l $M/k9b_rr.tsv $M/d22_rr.tsv
EX=ALL,NG=4,VLLM_CACHE_ROOT=/scratch/s/sneheel/vllm_cache
J9=$(sbatch --parsable --account=aip-rrabba --exclude=$X --time=12:00:00 --job-name=k9b-eval-rr --export=$EX,SLOTS=4 scripts/slurm/tamia_mechanism_node.sh $M/k9b_rr_smoke.tsv $M/k9b_rr.tsv)
JD=$(sbatch --parsable --account=aip-rrabba --exclude=$X --time=10:00:00 --job-name=d22-eval-rr --export=$EX,SLOTS=8 scripts/slurm/tamia_mechanism_node.sh $M/d22_rr_smoke.tsv $M/d22_rr.tsv)
JR=$(sbatch --parsable --account=aip-rrabba --exclude=$X --time=4:00:00 --job-name=d22-race --dependency=afterany:$JD --export=$EX,SLOTS=8 scripts/slurm/tamia_mechanism_node.sh $M/d22_race_smoke.tsv $M/d22_race.tsv)
JO=$(sbatch --parsable --account=aip-rrabba --exclude=$X --dependency=afterany:$JD configs/tica/mech_1109/det_offline_d22.sh)
# 4B task races: rerun the node over pieces_0.tsv (race_node skips finished pieces), then re-score all 22 tasks
RJ=$(sbatch --parsable --account=aip-rrabba --gres=gpu:h100:4 --exclude=$X --time=7:00:00 --job-name=race-final4b \
  --output=$FW/logs/%x-%j.out --export=ALL,TICA_REPO=$FW,TICA_VLLM_VENV=vllm-latest,VLLM_CACHE_ROOT=/scratch/s/sneheel/vllm_cache \
  $FW/scripts/slurm/race_node.sh $R/races/final4b_math/pieces_0.tsv)
echo "k9b-rr $J9 d22-rr $JD d22-race $JR det-offline $JO race-final4b-rr $RJ"
