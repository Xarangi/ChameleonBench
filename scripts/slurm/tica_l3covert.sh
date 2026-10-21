#!/bin/bash
# Adaptive-monitor multi-turn RL (tica.py rl, use_adaptive). Arm B/C. Usage:
#   sbatch [--time=..] scripts/slurm/tica_l3rl.sh <config> [extra tica.py args...]
# ctb-liyue_gpu has the HIGHER fairshare factor (0.352 vs def-rrabba_gpu 0.272, 10-10), so better
# priority despite def-rrabba's idle usage. Override with --account=.
#SBATCH --account=ctb-liyue_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=tica-l3covert
#SBATCH --output=/scratch/s/sneheel/ChameleonBench/logs/%x-%j.out
set -euo pipefail
CFG=${1:?config name}; shift || true
PY=${PY:-/scratch/s/sneheel/venvs/q35t29/bin/python}
export TRITON_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/triton  # home is at quota (10-10)
source /scratch/s/sneheel/ChameleonBench/scripts/slurm/_local_env.sh
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# Derive the snapshot from the config, never hardcode it (a 9B config with a 4B copy would score
# 9B probes against 4B activations; caught before it ran, 10-10).
SNAP=$(cd /scratch/s/sneheel/ChameleonBench && PYTHONPATH=src $PY -c "
from next_chameleons.tica.config import load_run_config as l; print(l('$CFG').model.path)")
test -d "$SNAP" || { echo "no snapshot for $CFG: $SNAP"; exit 1; }
LOCAL=${SLURM_TMPDIR:-/tmp}/$(basename $(dirname $(dirname $SNAP)))
t0=$(date +%s); mkdir -p $LOCAL; cp -rL $SNAP/. $LOCAL/; echo "model copy: $(( $(date +%s) - t0 ))s"
nvidia-smi --query-gpu=name,memory.total --format=csv
cd /scratch/s/sneheel/ChameleonBench
echo "Running: adaptive --config $CFG $* (snapshot $LOCAL)"
# TICA_DP=N (default 1 = one process): data-parallel episodes (src/next_chameleons/tica/dp.py), one
# process per GPU on this node; rank r sees only the r-th GPU of the allocation. A failed rank stops
# the others (they would wait in a collective forever).
N=${TICA_DP:-1}
if (( N <= 1 )); then
  env PYTHONPATH=src $PY scripts/tica.py adaptive --config "$CFG" --model-path "$LOCAL" "$@"
  exit 0
fi
IFS=, read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $(( N - 1 )))}"
(( ${#GPUS[@]} >= N )) || { echo "TICA_DP=$N needs $N GPUs (visible: ${CUDA_VISIBLE_DEVICES:-})"; exit 1; }
export MASTER_ADDR=127.0.0.1 MASTER_PORT=${MASTER_PORT:-$(( 20000 + ${SLURM_JOB_ID:-$$} % 20000 ))}
export WORLD_SIZE=$N
echo "data-parallel episodes: $N ranks on GPUs ${GPUS[*]:0:$N} (port $MASTER_PORT)"
# TICA_PYSPY=FILE (optional, profiling): rank 0 runs under py-spy (launched as the parent, so no ptrace
# attach), sampling every thread incl. idle waits at 20 Hz into FILE (collapsed stacks).
PIDS=()
for (( r = 0; r < N; r++ )); do
  WRAP=()
  if [[ -n "${TICA_PYSPY:-}" && $r == 0 ]]; then
    WRAP=(/home/s/sneheel/.local/bin/py-spy record --idle --rate 20 --format raw -o "$TICA_PYSPY" --)
  fi
  CUDA_VISIBLE_DEVICES=${GPUS[$r]} RANK=$r LOCAL_RANK=$r env PYTHONPATH=src "${WRAP[@]}" $PY scripts/tica.py \
    adaptive --config "$CFG" --model-path "$LOCAL" "$@" &
  PIDS+=($!)
done
rc=0
for _ in "${PIDS[@]}"; do
  if ! wait -n; then rc=1; kill "${PIDS[@]}" 2>/dev/null || true; fi
done
exit $rc
