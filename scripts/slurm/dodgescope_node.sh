#!/bin/bash
#SBATCH --account=aip-rrabba
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --exclude=tg11105
#SBATCH --job-name=dodgescope
#SBATCH --output=/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/dodgescope_1102/logs/%x-%A_%a.out
# DODGESCOPE (11-02) panel shards on one whole TamIA node (4 x H100), PER_GPU (default 2) processes per GPU. The model snapshot is
# copied and the venv unpacked ONCE per node into $SLURM_TMPDIR; then slot s (GPU s % 4) runs its task lines in order (the
# trained half-shard, then the base half-shard of the same concepts with --load-panel = the trained probes).
# Task file (TASKS, or TASKS.<array index> under a job array), tab-separated:
#   SLOT  OUT_JSON  LOAD_PANEL(or -)  ARGS (shell-quoted tica_probe_panel.py args after --out)
# A task whose OUT_JSON already carries its "timing" block is skipped (resume-safe). LOAD_PANEL missing (its
# trained shard failed) -> the task fits its own panel instead of idling. Logs: $RUNS/logs/<job>-s<SLOT>-<name>.out; CUDA OOM -> one rerun at half --batch;
# per-task wall clock in $RUNS/logs/<job>-tasks.tsv, GPU memory / utilisation in $RUNS/logs/<job>-gpu.csv.
#   sbatch [--array=0-2] scripts/slurm/dodgescope_node.sh TASKS
set -uo pipefail
WT=${TICA_REPO:-/scratch/s/sneheel/ChameleonBench}  # the checkout whose code runs
RUNS=${DS_RUNS:-/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/dodgescope_1102}
CFG=${CFG:-final_v4d_klx_tamia_4b}
cd $WT
Q=$1; [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && Q=$1.$SLURM_ARRAY_TASK_ID
TAG=${SLURM_JOB_NAME:-ds}-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${SLURM_ARRAY_TASK_ID:-0}
export SLURM_TMPDIR=${SLURM_TMPDIR:-/tmp/ds_$$}
mkdir -p "$SLURM_TMPDIR" "$RUNS/logs"
t0=$(date +%s)
# once per node: venv (node-local unpack + persistent triton cache, _local_env.sh) and the model snapshot
source $WT/scripts/slurm/_local_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SNAP=$(PYTHONPATH=$WT/src $PY -c "
from next_chameleons.tica.config import load_run_config as l; print(l('$CFG').model.path)")
LOCAL=$SLURM_TMPDIR/$(basename "$(dirname "$(dirname "$SNAP")")")  # keeps the snapshot dir NAME (probe cache key)
mkdir -p "$LOCAL"; cp -rL "$SNAP"/. "$LOCAL"/
echo "$(date) setup $(( $(date +%s) - t0 ))s: PY=$PY model=$LOCAL; tasks $Q"
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv -l 30 > "$RUNS/logs/$TAG-gpu.csv" 2>/dev/null &
SMI=$!
NG=$(nvidia-smi -L | wc -l)
# PER_GPU slots per GPU (default 2: the panel fit is CPU-bound, GPU idle, so a second process's rollout fills it);
# slot s runs on GPU s % NG, pinned to its own NCPU / (NG * PER_GPU) cores with as many OMP / MKL threads
PER_GPU=${PER_GPU:-2}; NS=$(( NG * PER_GPU )); NCPU=$(nproc); CPS=$(( NCPU / NS ))
export OMP_NUM_THREADS=$CPS MKL_NUM_THREADS=$CPS
# 2 per GPU: cap each process at 0.49 of the GPU (live peak 34.8 GB in the smoke; reserved beyond it is cache)
(( PER_GPU > 1 )) && export TICA_MEM_FRACTION=${TICA_MEM_FRACTION:-$(awk -v p=$PER_GPU "BEGIN{printf \"%.2f\", 0.98 / p}")}
slot_lane() {  # SLOT: run this slot's task lines in order
  local sl=$1 g=$(( $1 % NG )) gg out lp args name s rc cpus log bs
  cpus=$(( sl * CPS ))-$(( sl * CPS + CPS - 1 ))
  while IFS=$'\t' read -r gg out lp args; do
    [ "$gg" = "$sl" ] || continue
    if [ -f "$out" ] && grep -q '"timing"' "$out"; then echo "$(date) s$sl skip (done): $out"; continue; fi
    name=$(basename "$out" .json); name=$(basename "$(dirname "$out")")_$name
    if [ "$lp" != "-" ]; then
      miss=""; for f_ in ${lp//,/ }; do [ -f "$f_" ] || miss="$miss $f_"; done
      if [ -z "$miss" ]; then args="--load-panel $lp $args"; else echo "$(date) s$sl $name: missing$miss, fitting its own panel"; fi
    fi
    mf=$(echo "$args" | grep -o -- '--mem-frac [0-9.]*' | awk '{print $2}')  # per-task memory cap (a lone slot)
    args=$(echo "$args" | sed 's/--mem-frac [0-9.]*//')
    log=$RUNS/logs/$TAG-s$sl-$name.out
    s=$(date +%s)
    echo "$(date) s$sl (gpu $g, cpus $cpus) start $name"
    eval "TICA_MEM_FRACTION=${mf:-${TICA_MEM_FRACTION:-}} CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$WT/src taskset -c $cpus $PY scripts/tica_probe_panel.py --config $CFG --model-path $LOCAL --out $out $args" \
      < /dev/null > "$log" 2>&1
    rc=$?
    # OOM fallback (11-03 revision): rerun once with --fork-batch 8 and the SAME --batch (a batch change would change
    # the stratified episode plan); OOM_HALVE_BATCH=1 also halves --batch (semantics-free single-kind cells only)
    if (( rc != 0 )) && grep -q "OutOfMemoryError\|CUDA out of memory" "$log"; then
      bs=$(echo "$args" | grep -o -- '--batch [0-9]*' | tail -1 | awk '{print $2}'); bs=${bs:-8}
      if [ "${OOM_HALVE_BATCH:-0}" = 1 ]; then bs=$(( bs / 2 )); (( bs >= 1 )) || bs=1; fi
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$TAG" "$sl" "$name" "OOM" "$(( $(date +%s) - s ))" "retry --batch $bs --fork-batch 8" >> "$RUNS/logs/$TAG-tasks.tsv"
      echo "$(date) s$sl $name OOM -> retry --batch $bs"; mv "$log" "$log.oom"; s=$(date +%s)
      eval "TICA_MEM_FRACTION=${mf:-${TICA_MEM_FRACTION:-}} CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$WT/src taskset -c $cpus $PY scripts/tica_probe_panel.py --config $CFG --model-path $LOCAL --out $out $args --batch $bs --fork-batch 8" \
        < /dev/null > "$log" 2>&1
      rc=$?
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' "$TAG" "$sl" "$name" "$rc" "$(( $(date +%s) - s ))" >> "$RUNS/logs/$TAG-tasks.tsv"
    echo "$(date) s$sl end $name rc=$rc $(( $(date +%s) - s ))s"
  done < "$Q"
}
echo "$NS slots ($PER_GPU per GPU), $CPS cpus / threads each"
PIDS=()
for (( sl = 0; sl < NS; sl++ )); do slot_lane $sl & PIDS+=($!); done
for p in "${PIDS[@]}"; do wait "$p"; done
kill $SMI 2>/dev/null
echo "$(date) node done, wall $(( $(date +%s) - t0 ))s"
cat "$RUNS/logs/$TAG-tasks.tsv"
