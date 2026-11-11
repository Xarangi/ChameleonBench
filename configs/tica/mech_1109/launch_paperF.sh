#!/bin/bash
# 11-11: 27B judge on every gated free-generation race (never run on the gated races): is the monitored concept still
# a central property of the answer, and is the answer a coherent response? Per-row scores are joined to passes by
# configs/tica/mech_1109 analysis. One node, 4 judge processes (one per GPU), after paperE (its *_hok gens are included
# and skipped if missing).   launch_paperF.sh [DEPENDENCY_JOBID]
set -euo pipefail
cd /scratch/s/sneheel/ChameleonBench
R=/scratch/s/sneheel/next_chameleons_artifacts/tica/runs; M=$R/mech_1109; J=$M/judge; mkdir -p $J
T=$M/paperF.tsv; S=$M/paperF_smoke.tsv; : > $T
head -60 $R/final_v4d_tamia_2b_x1800/quality/step_1800/gens/gated_train.jsonl > $M/judge_smoke.jsonl
printf 'j\t%s\t%s\n' $M/judge_smoke.json "bash configs/tica/mech_1109/judge_piece.sh $M/judge_smoke.json smoke=$M/judge_smoke.jsonl" > $S
g() { local n=$1 d=$2 a="" sp v; for sp in train heldout; do for v in "" _hok; do a="$a ${n}_$sp$v=$R/$d/gens/gated_$sp$v.jsonl"; done; done; echo "$a"; }
add() { printf 'j\t%s\t%s\n' $J/$1.json "bash configs/tica/mech_1109/judge_piece.sh $J/$1.json $2" >> $T; }
add j4b "$(g 4b final_v4d_k03_tamia_4b/quality/step_1800)"
add j9b "$(g 9b final_v4d_k03_tamia_9b/quality/step_1800)"
add j2bdet "$(g 2b final_v4d_tamia_2b_x1800/quality/step_1800) $(g det final_v4d_k03det_tamia_4b/quality/step_2200)"
add jbase "$(g b4b v4b_quality_base/quality_gated) $(g b2b v4b_quality_base_2b/quality_gated) $(g b9b v4b_quality_base_9b/quality_gated)"
wc -l $T $S
[ "${DRY:-0}" = 1 ] && exit 0
sbatch --parsable --account=aip-rrabba --exclude=tg11105,tg10907,tg10807 --time=03:00:00 --job-name=paperF ${1:+--dependency=afterany:$1} \
  --export=ALL,NG=4,SLOTS=4 scripts/slurm/tamia_mechanism_node.sh $S $T
