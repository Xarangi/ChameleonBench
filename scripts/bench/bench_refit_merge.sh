#!/bin/bash
#SBATCH --account=ctb-liyue_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --job-name=bench-refit
#SBATCH --output=/home/sneheel/chameleons/logs/%x-%j.out
# Refit read benchmark, twice: cold Triton cache, then a NEW process on the warm cache (+ cached
# autotuning) -> how much of a job's first-step time is kernel compilation / autotuning.
set -uo pipefail
CFG=${1:-topics_r1mix_4b}; AD=${2:-/scratch/sneheel/next_chameleons_artifacts/tica/runs/topics_r1mix_4b/adaptive/step_90}
cd /home/sneheel/chameleons
source scripts/slurm/_local_env.sh
SNAP=$(PYTHONPATH=src $PY -c "from next_chameleons.tica.config import load_run_config as l; print(l('$CFG').model.path)")
LOCAL=$SLURM_TMPDIR/$(basename $(dirname $(dirname $SNAP))); mkdir -p $LOCAL; cp -rL $SNAP/. $LOCAL/
export TRITON_CACHE_DIR=/scratch/sneheel/triton_cache/q35t29 TRITON_CACHE_AUTOTUNING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,nounits -l 1 \
  > logs/gpu-${SLURM_JOB_ID}.csv & SMI=$!
for run in profile; do
  echo "=== $run"
  t0=$(date +%s)
  env PYTHONPATH=src $PY scripts/bench/bench_refit_reads.py --config $CFG --adapter $AD \
    --model-path $LOCAL --merge-test 2>&1 | grep -vE "Loading weights|Warning|warn"
  echo "wall $(( $(date +%s) - t0 ))s"
done
kill $SMI
