#!/bin/bash
#SBATCH --account=aip-rrabba
#SBATCH --exclude=tg11105
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=det-frozen
#SBATCH --output=/scratch/s/sneheel/ChameleonBench/logs/%x-%j.out
cd /scratch/s/sneheel/ChameleonBench-detmix2
source scripts/slurm/_local_env.sh
export OMP_NUM_THREADS=8
N=/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/mech_1108
$PY $N/frozen_transfer.py $N/$1.json $2 $3
