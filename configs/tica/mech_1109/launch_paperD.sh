#!/bin/bash
# 11-11 PAPER CONTROLS (final audit before writing): the size comparison on the gated task races had a base-model
# control for the 4B only, and the 2B had no measured base capability. One node, one process per GPU:
#  1. base 2B capability: GSM8K / MMLU clean + the 2 briefs. Same command as every other capability run.
#  2. base 2B and base 9B on the 17 non-MMLU gated task races (mgsm_ja excluded, as everywhere). HALF SIZE: 16 episodes
#     per task = piece 1 (seed 1) of the 32-episode races, so the episodes are the first halves of the trained models'.
#     A base model beats ~0.3 probes per episode, so 272 episodes per size bound the base rate tightly enough.
# Pieces come from heldout_race_launch.py --dry-run on specs that differ from final2b / final9b only in "base" for the
# adapter, the task list and episodes 32 -> 16 (checked below: commands equal the trained p1 commands up to --base).
# Smoke lines reuse existing smoke outputs (base-4B race, 2B race, 9B race all passed on these scripts), so no GPU time
# is spent on smokes. Every group in the full TSV has a smoke line (mech node skips groups without one).
set -euo pipefail
cd /scratch/s/sneheel/ChameleonBench
R=/scratch/s/sneheel/next_chameleons_artifacts/tica/runs; M=$R/mech_1109; RC=$R/races
FW=/scratch/s/sneheel/ChameleonBench-final; HERE=configs/tica/mech_1109
C2=final_v4d_tamia_2b_x1800
TASKS=mbpp,pyeval,javaeval,mbppjava,json,belebele_ta,mgsm_th,mgsm_de,mgsm_ru,mgsm_fr,math,sql,jseval,belebele_fi,belebele_pt,belebele_tr,belebele_el
add() { printf '%s\t%s\t%s\n' "$2" "$3" "$4" >> "$1"; }   # FILE GROUP OUT CMD
for s in finalbase2b finalbase9b; do
  (cd $FW && /usr/bin/python3 scripts/heldout_race_launch.py --spec /scratch/s/sneheel/ChameleonBench/$HERE/${s}_spec.json --dry-run > /dev/null)
done
T=$M/paperD.tsv; S=$M/paperD_smoke.tsv; : > $T; : > $S
add $S rl $M/smoke_9b_race.json true
add $S race $M/smoke_base4b_race.json true
# 9B pieces first (slowest), then 2B; within a size, longest tasks first. 64k context: only matters past 32k tokens
# (base-4B Tamil overflowed 32k), so no reruns are needed.
for run in finalbase9b finalbase2b; do
  for t in $(tr ',' ' ' <<< "$TASKS"); do
    while IFS=$'\t' read -r out cmd; do
      cmd=${cmd/TICA_VLLM_MAXLEN=32768/TICA_VLLM_MAXLEN=65536}
      add $T race "$out" "bash $HERE/race_piece.sh $out $(printf %q "$cmd")"
    done < <(grep -hF "/${run}_$t/" $RC/${run}_*/pieces*.tsv | sort -u)
  done
done
add $T rl $R/v4b_quality_base_2b/capability.json "bash scripts/slurm/tica_capability.sh $C2 $R/v4b_quality_base_2b/capability.json --base --benches gsm8k,mmlu --gsm-max-new-tokens 512 --briefs mathematical\\ reasoning,HTML\\ markup"
wc -l $T $S
[ "${DRY:-0}" = 1 ] && exit 0
X=tg11105,tg10907,tg10807
J=$(sbatch --parsable --account=aip-rrabba --exclude=$X --time=08:00:00 --job-name=paperD \
  --export=ALL,NG=4,SLOTS=4,VLLM_CACHE_ROOT=/scratch/s/sneheel/vllm_cache scripts/slurm/tamia_mechanism_node.sh $S $T)
echo "paperD $J"
mkdir -p $HERE/scorebase2b $HERE/scorebase9b
for sz in 2b 9b; do
  for t in $(tr ',' ' ' <<< "$TASKS"); do
    f=$HERE/scorebase$sz/race-score-finalbase$sz-$t.sh
    sed "s/finalbase4b/finalbase$sz/g; s/base4b=/base$sz=/; s/base4b_p/base${sz}_p/" \
      $HERE/scorebase4b/race-score-finalbase4b-$t.sh > $f
    sbatch --parsable --account=aip-rrabba --cpus-per-task=8 --mem=16G --time=03:00:00 --dependency=afterany:$J \
      --job-name=race-score-finalbase$sz-$t $f > /dev/null
  done
done
echo "34 score jobs queued after $J"
