#!/bin/sh
# 11-10: re-score 9B javaeval once the p1 piece (rerun in k9b_rr2 with --batch 4) exists
#SBATCH --account=aip-rrabba
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --job-name=race-score-final9b-javaeval
R=/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/races/final9b_javaeval
[ -s $R/k9b1800_p1.json ] || { echo "RACE INCOMPLETE: p1 missing"; exit 3; }
cd /scratch/s/sneheel/ChameleonBench-final && OMP_NUM_THREADS=1 PYTHONPATH=src /scratch/s/sneheel/venvs/q35t29/bin/python scripts/tica_task_race_report.py --gold /scratch/s/sneheel/next_chameleons_artifacts/tica/runs/mech_1025/taskrace/gold_javaeval.json --race "k9b1800=$R/k9b1800_p*.jsonl" --workers 8 --out $R/report.json
