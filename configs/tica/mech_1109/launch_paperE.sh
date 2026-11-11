#!/bin/bash
# 11-11 HELD-OUT-PROBE RACE (user: every finding must be robust; run more if a finding lacks support).
# Gap: every race so far lets the defender refit only with the TRAINED probe types at the TRAINED layers. Probe-type
# generalisation rests on the passive never-shown panel alone, and that panel shows layers 4 / 28 / 30 and ens8_linear
# are not beaten. This runs the same gated free-generation race (32 eps x 24 turns, 6-of-8 gate, 8-fork panel) with
# the defender refitting ONLY with the config's final.heldout_refit_kinds (+ the earliest panel layer), on trained and
# held-out concepts, for 2B / 4B / 9B at 1800 and the untrained 4B.
# Lines = the finished gated-race lines with only the output names and "--refit-switch 3 --switch-kinds LIST" changed.
# The turn-0 probe of an episode is still the base panel probe (as in every race); every refit is a held-out kind.
# One node, one process per GPU, 8 lines, longest first (~3 h).
set -euo pipefail
cd /scratch/s/sneheel/ChameleonBench
R=/scratch/s/sneheel/next_chameleons_artifacts/tica/runs; M=$R/mech_1109
HK='linear@20,linear@24,linear@28,linear@30,maxpool@24,gdm_rollattn,mck_rollmean40,mck_attnlite,mck_softmax,rlobf_max,rlobf_attn_max,rollmax,meandiff,ens8_linear,ens8_mlp,nc_ens8_tok,bailey_ens_tok,bailey_ens_tok_max,spread5_linear,linear@4'
HK2='linear@15,linear@18,linear@21,linear@23,maxpool@18,gdm_rollattn,mck_rollmean40,mck_attnlite,mck_softmax,rlobf_max,rlobf_attn_max,rollmax,meandiff,ens8_linear,ens8_mlp,nc_ens8_tok,bailey_ens_tok,bailey_ens_tok_max,spread5_linear,linear@3'
T=$M/paperE.tsv; S=$M/paperE_smoke.tsv; : > $T
line() {  # SRC_TSV OUT_SUBSTR KINDS
  grep -hF "$2" $1 | tail -1 | sed -E "s#race_(train|heldout)_gated\.json#race_\1_gated_hok.json#g; s#gated_(train|heldout)\.jsonl#gated_\1_hok.jsonl#; s#\$# --refit-switch 3 --switch-kinds $3#" >> $T
}
for sp in train heldout; do line $M/fix4.tsv "final_v4d_k03_tamia_4b/quality/step_1800/race_${sp}_gated.json" $HK; done
for sp in train heldout; do line $M/k9b_rr2.tsv "final_v4d_k03_tamia_9b/quality/step_1800/race_${sp}_gated.json" $HK; done
for sp in train heldout; do line $M/paperQ.tsv "final_v4d_tamia_2b_x1800/quality/step_1800/race_${sp}_gated.json" $HK2; done
for sp in train heldout; do line $M/paperQ.tsv "v4b_quality_base/quality_gated/race_${sp}_gated.json" $HK; done
grep -F smoke_2b_gated3 $M/paperQ_smoke.tsv | sed -E "s#smoke_2b_gated3#smoke_2b_gated_hok#g; s#\$# --refit-switch 3 --switch-kinds $HK2#" > $S
wc -l $T $S; cut -f1,2 $T | sed "s#$R/##"
[ "$(wc -l < $T)" = 8 ] && [ "$(wc -l < $S)" = 1 ] || { echo "bad TSV"; exit 1; }
[ "${DRY:-0}" = 1 ] && exit 0
sbatch --parsable --account=aip-rrabba --exclude=tg11105,tg10907,tg10807 --time=07:00:00 --job-name=paperE \
  --export=ALL,NG=4,SLOTS=4,VLLM_CACHE_ROOT=/scratch/s/sneheel/vllm_cache scripts/slurm/tamia_mechanism_node.sh $S $T
