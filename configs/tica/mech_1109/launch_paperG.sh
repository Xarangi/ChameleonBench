#!/bin/bash
# 11-11 evening: paperE's 9B trained-concept held-out-kind race died of CUDA OOM in its last batch (batch 8, episodes
# 24-32; the race has no resume). Rerun it with --batch 4 (as the 4B gated race after its own OOM; batch only changes
# how episodes are grouped). To not spend a whole node on one GPU, this node REPLACES paperF: GPU 0 runs the rerun,
# the other three run the judges that do not need it (j4b, j2bdet, jbase), and j9b takes the first free GPU and waits
# for the rerun's output (wait_then.sh) so it judges the 9B hok gens too.   launch_paperG.sh DEPENDENCY_JOBID
set -euo pipefail
cd /scratch/s/sneheel/ChameleonBench
R=/scratch/s/sneheel/next_chameleons_artifacts/tica/runs; M=$R/mech_1109; HERE=configs/tica/mech_1109
T=$M/paperG.tsv; S=$M/paperG_smoke.tsv
O9=$R/final_v4d_k03_tamia_9b/quality/step_1800/race_train_gated_hok.json
grep -F "$O9" $M/paperE.tsv | sed 's/--batch 8 /--batch 4 /' > $T
grep -q -- "--batch 4 " $T || { echo "batch edit failed"; exit 1; }
grep -vF "judge/j9b.json" $M/paperF.tsv >> $T
grep -F "judge/j9b.json" $M/paperF.tsv | awk -F'\t' -v o="$O9" -v h="$HERE" 'BEGIN{OFS="\t"}{print $1,$2,"bash " h "/wait_then.sh " o " tica_switch_eval[.]py " $3}' >> $T
cat $M/paperE_smoke.tsv $M/paperF_smoke.tsv > $S
wc -l $T $S; cut -f1,2 $T | sed "s#$R/##"
[ "$(wc -l < $T)" = 5 ] && [ "$(wc -l < $S)" = 2 ] || { echo "bad TSV"; exit 1; }
[ "${DRY:-0}" = 1 ] && exit 0
sbatch --parsable --account=aip-rrabba --exclude=tg11105,tg10907,tg10807 --time=07:00:00 --job-name=paperG ${1:+--dependency=afterany:$1} \
  --export=ALL,NG=4,SLOTS=4,VLLM_CACHE_ROOT=/scratch/s/sneheel/vllm_cache scripts/slurm/tamia_mechanism_node.sh $S $T
