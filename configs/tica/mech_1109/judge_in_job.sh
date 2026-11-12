#!/bin/bash
# 11-11: run the judge lines of paperG.tsv as a step INSIDE the running paperG allocation (its judge smoke failed on a
# Narval-only venv tar, so the node skipped group j while 3 of its 4 GPUs sit idle next to the 9B race rerun).
#   srun --jobid=JOB --overlap -N1 -n1 --cpus-per-task=24 bash configs/tica/mech_1109/judge_in_job.sh
# Smoke on GPU 1 first; then j4b / j2bdet / jbase on GPUs 1-3 and j9b (the three finished 9B races; the rerun's gens are
# judged later) on the first free GPU. The 9B race keeps GPU 0.
cd /scratch/s/sneheel/ChameleonBench
R=/scratch/s/sneheel/next_chameleons_artifacts/tica/runs; M=$R/mech_1109; L=logs/paperG-judge
line() { grep -P "judge/$1.json\t" $M/paperG.tsv | cut -f3 | sed 's/^bash [^ ]*wait_then.sh [^ ]* [^ ]* //'; }
[ -s $M/judge_smoke.json ] || CUDA_VISIBLE_DEVICES=1 bash configs/tica/mech_1109/judge_piece.sh $M/judge_smoke.json smoke=$M/judge_smoke.jsonl > $L-smoke.out 2>&1
[ -s $M/judge_smoke.json ] || { echo "judge smoke failed"; exit 1; }
echo "smoke ok $(date)"
run() { [ -s $M/judge/$2.json ] || CUDA_VISIBLE_DEVICES=$1 bash -c "$3" > $L-$2.out 2>&1; echo "$2 rc $? $(date)"; }
run 1 j4b "$(line j4b)" &
run 2 j2bdet "$(line j2bdet)" &
( run 3 jbase "$(line jbase)"; run 3 j9b "$(line j9b | sed 's# 9b_train_hok=[^ ]*##')" ) &
wait
