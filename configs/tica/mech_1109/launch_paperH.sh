#!/bin/bash
# 11-12 (user: "standardized evals everywhere, both generation ones and fixed text ones", strict-KL and extended runs):
# the strict-KL runs had the fixed-text val race, panel, persistence, capability and task races, but only the UNGATED
# free-generation race. This node adds the GATED free-generation race (6-of-8 fork gate; the command of
# race_{train,heldout}_gated.json of the KL 0.3 models, only config / adapter / paths changed) for strict 4B @1800,
# strict 4B @2500 and strict 2B @1800, then the 27B judge on their answers and on the 9B held-out-kind trained-concept
# race that paperG could not judge. 4B races at batch 4 (as the KL 0.3 4B: batch 8 ran out of memory), 2B at batch 8.
set -euo pipefail
cd /scratch/s/sneheel/ChameleonBench
R=/scratch/s/sneheel/next_chameleons_artifacts/tica/runs; M=$R/mech_1109; J=$M/judge; HERE=configs/tica/mech_1109
T=$M/paperH.tsv; S=$M/paperH_smoke.tsv; : > $T; : > $S
printf 'q\t%s\ttrue\n' $M/smoke_2b_gated_hok.json >> $S
printf 'j\t%s\ttrue\n' $M/judge_smoke.json >> $S
SRC4=final_v4d_k03_tamia_4b; A4=$R/$SRC4/resume_1660/adaptive/step_140
SRC2=final_v4d_tamia_2b_x1800; A2=$R/$SRC2/resume_1270/adaptive/step_530
race() {  # SRCRUN SRCADAPTER RUN STEP ADAPTER SPLIT BATCH
  [ -d "$5" ] || { echo "missing adapter $5"; exit 1; }
  grep -hP "^q\t$R/$1/quality/step_1800/race_$6_gated.json\t" $M/paperA.tsv $M/paperB.tsv $M/fix4.tsv | head -1 \
    | sed "s#$2#$5#g; s#$1/quality/step_1800#$3/quality/step_$4#g; s#tica_switch_eval.sh $1 #tica_switch_eval.sh $3 #; s#--batch [0-9]*#--batch $7#" >> $T
}
for sp in train heldout; do
  race $SRC4 $A4 final_v4d_klx_tamia_4b_x2500 2500 $R/final_v4d_klx_tamia_4b_x2500/resume_2100/adaptive/step_400 $sp 4
  race $SRC4 $A4 final_v4d_klx_tamia_4b 1800 $R/final_v4d_klx_tamia_4b/resume_1280/adaptive/step_520 $sp 4
done
for sp in train heldout; do
  race $SRC2 $A2 final_v4d_klx_tamia_2b 1800 $R/final_v4d_klx_tamia_2b/resume_1200/adaptive/step_600 $sp 8
done
# ChameleonBench-final has no final_v4d_klx_tamia_4b_x2500.yaml: the x2500 races load final_v4d_klx_tamia_4b (same file
# except steps / checkpoint settings / out; --out and --adapter are explicit).
sed -i 's#tica_switch_eval.sh final_v4d_klx_tamia_4b_x2500 #tica_switch_eval.sh final_v4d_klx_tamia_4b #' $T
g() { echo "$1=$R/$2/gens/gated_$3.jsonl"; }
printf 'j\t%s\t%s\n' $J/j9bhok.json "bash $HERE/judge_piece.sh $J/j9bhok.json $(g 9b_train_hok final_v4d_k03_tamia_9b/quality/step_1800 train_hok)" >> $T
printf 'j\t%s\t%s\n' $J/jstrict.json "bash $HERE/wait_then.sh /nonexistent/never tica_switch_eval[.]py bash $HERE/judge_piece.sh $J/jstrict.json $(g s4b_train final_v4d_klx_tamia_4b/quality/step_1800 train) $(g s4b_heldout final_v4d_klx_tamia_4b/quality/step_1800 heldout) $(g s4bx_train final_v4d_klx_tamia_4b_x2500/quality/step_2500 train) $(g s4bx_heldout final_v4d_klx_tamia_4b_x2500/quality/step_2500 heldout) $(g s2b_train final_v4d_klx_tamia_2b/quality/step_1800 train) $(g s2b_heldout final_v4d_klx_tamia_2b/quality/step_1800 heldout)" >> $T
wc -l $T; cut -f1,2 $T
[ "${DRY:-0}" = 1 ] && exit 0
sbatch --parsable --account=aip-rrabba --exclude=tg11105,tg10907,tg10807 --time=07:00:00 --job-name=paperH \
  --export=ALL,NG=4,SLOTS=4,VLLM_CACHE_ROOT=/scratch/s/sneheel/vllm_cache scripts/slurm/tamia_mechanism_node.sh $S $T
