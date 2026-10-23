#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=6:00:00
#SBATCH --job-name=tica-drain
#SBATCH --output=/scratch/s/sneheel/ChameleonBench/logs/%x-%j.out
# Run the held-out races queued by tica_val_local.sh (TICA_VAL_QUEUE) on a whole node, one race per GPU,
# 4 at a time (TamIA 10-21). A race is done when VALDIR/step_<step>.json exists; done races are skipped, so
# the drain can be re-run any time (e.g. after every training chunk). Extra commands (one shell line each, e.g.
# the quality-check evals of scripts/tamia_v4b.sh) can be queued in QUEUE.extra; they run on free GPUs too.
#   sbatch --account=aip-<pi> scripts/slurm/tica_val_drain.sh QUEUE [QUEUE ...]
set -uo pipefail
cd /scratch/s/sneheel/ChameleonBench
(( $# >= 1 )) || { echo "usage: tica_val_drain.sh QUEUE [QUEUE ...]"; exit 2; }
NODE_TMP=${SLURM_TMPDIR:-/tmp/tica_drain_$$}
mkdir -p "$NODE_TMP" logs
NG=$(nvidia-smi -L 2>/dev/null | wc -l); (( NG > 0 )) || NG=1
# PER_GPU (default 2; TamIA 10-21): eval processes are host-bound (~40% GPU each) and use well under half of an
# H100's memory, so two share each GPU; slot s runs on GPU s % NG
PER_GPU=${PER_GPU:-2}
NS=$(( NG * PER_GPU ))
TASKS=()
for q in "$@"; do
  [ -f "$q" ] || continue
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    eval "set -- $line"   # CFG ADAPTER STEP VALDIR SPLIT [args]
    if [ -f "$4/step_$3.json" ]; then continue; fi
    if [ ! -d "$2" ]; then echo "skip (no adapter): $line"; continue; fi
    TASKS+=("bash scripts/slurm/tica_val.sh $line")
  done < "$q"
  if [ -f "$q.extra" ]; then
    while IFS= read -r line; do
      [ -n "$line" ] && TASKS+=("$line")
    done < "$q.extra"
    mv -f "$q.extra" "$q.extra.started.${SLURM_JOB_ID:-$$}"
  fi
done
# in-run quality checks (TICA_DRAIN_QUALITY="CFG ...": newest checkpoint of each config + the base reference once)
if [ -n "${TICA_DRAIN_QUALITY:-}" ]; then
  while IFS= read -r line; do
    [ -n "$line" ] && TASKS+=("$line")
  done < <(bash scripts/slurm/tamia_queue_quality.sh $TICA_DRAIN_QUALITY)
fi
# LONGEST FIRST (TamIA 10-22): panels (shards) ~0.3-1 h, races 10-30 min, capability ~5 min; starting the long
# ones first keeps every slot busy until the end instead of a tail of 1-2 lone panels
prio() { case "$1" in *tica_probe_panel*) echo 0 ;; *tica_switch_eval*|*tica_val.sh*) echo 1 ;; *) echo 2 ;; esac; }
SORTED=(); for p_ in 0 1 2; do for t in "${TASKS[@]}"; do [ "$(prio "$t")" = "$p_" ] && SORTED+=("$t"); done; done
TASKS=("${SORTED[@]}")
echo "${#TASKS[@]} tasks on $NG GPUs x $PER_GPU"
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv -l 60 \
  > "logs/${SLURM_JOB_NAME:-drain}-${SLURM_JOB_ID:-local}-gpu.csv" 2>/dev/null &
SMI=$!
declare -A BUSY
i=0
for t in "${TASKS[@]}"; do
  while true; do
    for (( g = 0; g < NS; g++ )); do
      p=${BUSY[$g]:-}
      if [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; then break 2; fi
    done
    sleep 15
  done
  mkdir -p "$NODE_TMP/t$i"
  echo "slot $g (gpu $(( g % NG ))): $t"
  CUDA_VISIBLE_DEVICES=$(( g % NG )) SLURM_TMPDIR=$NODE_TMP/t$i bash -c "$t" \
    > "logs/${SLURM_JOB_NAME:-drain}-${SLURM_JOB_ID:-local}-t$i.out" 2>&1 &
  BUSY[$g]=$!
  i=$(( i + 1 ))
done
wait
kill $SMI 2>/dev/null
# sharded panels (10-22): merge panel_<split>.s*.json -> panel_<split>.json once every shard exists
R_=/scratch/s/sneheel/next_chameleons_artifacts/tica/runs
for f0 in "$R_"/*/quality/step_*/panel_*.s0.json "$R_"/v4b_quality_base/panel_*.s0.json; do
  [ -f "$f0" ] || continue
  out=${f0%.s0.json}.json; [ -f "$out" ] && continue
  sh_=("${f0%.s0.json}".s*.json)
  (( ${#sh_[@]} >= ${PANEL_SHARDS:-4} )) || { echo "panel shards incomplete: $f0"; continue; }
  PYTHONPATH=src /scratch/s/sneheel/venvs/q35t29/bin/python scripts/tica_panel_merge.py --out "$out" "${sh_[@]}" \
    || echo "panel merge failed: $out"
done
# quality aggregates (CPU): every quality dir with free-generation answers
if [ -n "${TICA_DRAIN_QUALITY:-}" ]; then
  R=/scratch/s/sneheel/next_chameleons_artifacts/tica/runs
  for q in "$R"/v4b_quality_base "$R"/*/quality/step_*; do
    ls "$q"/gens/*.jsonl > /dev/null 2>&1 || continue
    A=(); for f in "$q"/gens/*.jsonl; do A+=("$(basename "$f" .jsonl)=$f"); done
    /scratch/s/sneheel/venvs/q35t29/bin/python scripts/tica_quality_check.py --gens "${A[@]}" --out "$q/quality.json" \
      || echo "quality summary failed: $q"
  done
fi
echo "drained"
