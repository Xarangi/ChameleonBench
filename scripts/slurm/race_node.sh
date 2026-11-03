#!/bin/bash
# RACE NODE (11-03): runs the pieces of an interactive race (scripts/race_launch.py writes the TSV and
# submits this script) with ONE piece process per GPU at a time; each piece runs many episodes at once
# (tica_switch_eval --batch 16 --fast-forks, colocated vLLM: TICA_VLLM=1 TICA_VLLM_SLEEP=0).
#   TSV lines:  OUT_JSON<TAB>COMMAND        (COMMAND runs from $TICA_REPO with CUDA_VISIBLE_DEVICES set)
# A piece whose OUT_JSON exists is skipped, so resubmitting the same TSV resumes a timed-out node.
# The vLLM venv is unpacked node-locally ONCE here (its python is exported to every piece).
#   sbatch --gres=gpu:h100:4 --export=ALL,TICA_REPO=<checkout> scripts/slurm/race_node.sh TSV
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --job-name=race-node
set -uo pipefail
TSV=${1:?piece TSV}
REPO=${TICA_REPO:?TICA_REPO = the checkout to run}
cd "$REPO" || exit 1
mkdir -p logs
NG=${NG:-$(nvidia-smi -L | wc -l)}
NODE_TMP=${SLURM_TMPDIR:-/tmp/race_$$}
mkdir -p "$NODE_TMP"
JOB=${SLURM_JOB_NAME:-race}-${SLURM_JOB_ID:-local}
VV=${TICA_VLLM_VENV:-vllm-latest}
if [ -z "${TICA_VLLM_PY:-}" ] && [ -f "/scratch/s/sneheel/venvs/$VV.tar" ]; then
  t0=$(date +%s)
  tar -xf "/scratch/s/sneheel/venvs/$VV.tar" -C "$NODE_TMP" && export TICA_VLLM_PY="$NODE_TMP/$VV/bin/python"
  echo "vllm venv $VV unpacked in $(( $(date +%s) - t0 ))s"
fi
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv -l 60 \
  > "logs/$JOB-gpu.csv" 2>/dev/null &
SMI=$!
declare -A BUSY
mapfile -t LINES < <(grep -v '^\s*#' "$TSV" | grep -v '^\s*$')
t0=$(date +%s); i=0; FAILED=()
for line in "${LINES[@]}"; do
  IFS=$'\t' read -r out cmd <<< "$line"
  if [ -s "$out" ]; then echo "skip (done): $out"; continue; fi
  while true; do  # first idle GPU
    for (( g = 0; g < NG; g++ )); do
      p=${BUSY[$g]:-}
      if [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; then break 2; fi
    done
    sleep 10
  done
  mkdir -p "$NODE_TMP/gpu$g" "$(dirname "$out")"
  echo "[$(( $(date +%s) - t0 ))s] gpu $g piece $i: $out"
  CUDA_VISIBLE_DEVICES=$g SLURM_TMPDIR=$NODE_TMP/gpu$g bash -c "$cmd" > "logs/$JOB-$i.out" 2>&1 &
  BUSY[$g]=$!
  i=$(( i + 1 ))
done
for (( g = 0; g < NG; g++ )); do p=${BUSY[$g]:-}; [ -n "$p" ] && wait "$p"; done
kill $SMI 2>/dev/null
for line in "${LINES[@]}"; do
  IFS=$'\t' read -r out _ <<< "$line"
  [ -s "$out" ] || FAILED+=("$out")
done
echo "RACE NODE DONE $(( $(date +%s) - t0 ))s; pieces without output: ${#FAILED[@]}"
for f in "${FAILED[@]}"; do echo "  $f"; done
