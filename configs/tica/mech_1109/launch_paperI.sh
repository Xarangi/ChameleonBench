#!/bin/bash
# 11-12 (user: "one word outputs is fine if our judges say it is ... as long as it does the task"): re-judge every
# gated free-answer race with the judge's LOOSE task prompt (tica_free_judge.py --answers-prompt loose: "relevant,
# coherent answer that does what the request asks ... may be shown only in part"), so "does the passing answer do the
# task" comes from the judge and not from a character count. The default task prompt scored even untrained full answers
# low and was not used. Same gens as paperG's judge lines (+ the 9B held-out-kind trained-concept race). ~5 min per line.
set -euo pipefail
cd /scratch/s/sneheel/ChameleonBench
R=/scratch/s/sneheel/next_chameleons_artifacts/tica/runs; M=$R/mech_1109; J=$M/judge
T=$M/paperI.tsv; S=$M/paperI_smoke.tsv; : > $T
printf 'j\t%s\ttrue\n' $M/judge_smoke.json > $S
for n in j4b j2bdet jbase j9b; do
  grep -P "judge/$n.json\t" $M/paperG.tsv | cut -f3 | sed 's/^bash [^ ]*wait_then.sh [^ ]* [^ ]* //' \
    | sed "s#judge/$n.json#judge/${n}_loose.json#g; s#^bash #JUDGE_EXTRA='--answers-prompt loose' bash #" \
    | awk -v o=$J/${n}_loose.json '{print "j\t" o "\t" $0}' >> $T
done
cut -f1,2 $T; cut -f3 $T | head -1 | cut -c1-260
[ "${DRY:-0}" = 1 ] && exit 0
sbatch --parsable --account=aip-rrabba --exclude=tg11105,tg10907,tg10807 --time=01:00:00 --job-name=paperI \
  --export=ALL,NG=4,SLOTS=4,VLLM_CACHE_ROOT=/scratch/s/sneheel/vllm_cache scripts/slurm/tamia_mechanism_node.sh $S $T
