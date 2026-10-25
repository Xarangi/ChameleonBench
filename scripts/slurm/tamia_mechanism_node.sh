#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=14:00:00
#SBATCH --job-name=mech-node
#SBATCH --output=/scratch/s/sneheel/ChameleonBench/logs/%x-%j.out
# MECHANISM + SCENARIO node (TamIA 10-25, user: "use the full node; robust analyses of the mechanism + interesting evals").
# One node, SLOTS processes (default 2 per GPU). Two phases:
#   1. SMOKE: one tiny run per group (2 episodes, few turns) -- every new code path once. A group whose smoke fails
#      (non-zero exit or missing output JSON) is SKIPPED in phase 2 and reported, so one broken path cannot burn the node.
#   2. FULL: the group's tasks, longest-first within the smoke-passed groups.
# Task file lines:  GROUP<TAB>OUTPUT_JSON<TAB>COMMAND      (smoke file: same format, one line per group)
# A task whose OUTPUT_JSON already exists is skipped (resume-safe). Aggregates only; feature dumps (.pt) stay on scratch.
#   sbatch scripts/slurm/tamia_mechanism_node.sh SMOKE_FILE FULL_FILE
set -uo pipefail
SMOKE=${1:?smoke task file}; FULL=${2:?full task file}
cd /scratch/s/sneheel/ChameleonBench
NG=${NG:-$(nvidia-smi -L | wc -l)}; SLOTS=${SLOTS:-$(( NG * 2 ))}
NODE_TMP=${SLURM_TMPDIR:-/tmp/mech_$$}; mkdir -p "$NODE_TMP" logs
JOB=${SLURM_JOB_NAME:-mech}-${SLURM_JOB_ID:-local}
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv -l 60 > "logs/$JOB-gpu.csv" 2>/dev/null &
SMI=$!
declare -A BUSY TAG OUT

run_phase() {  # FILE PHASE [allowed-groups regex]
  local file=$1 phase=$2 allow=${3:-.*} i=0 g p line grp out cmd
  mapfile -t LINES < <(grep -v '^\s*#' "$file" | grep -v '^\s*$')
  for line in "${LINES[@]}"; do
    IFS=$'\t' read -r grp out cmd <<< "$line"
    [[ "$grp" =~ ^($allow)$ ]] || continue
    if [ -s "$out" ]; then echo "[$phase] skip (done): $out"; continue; fi
    while true; do
      for (( g = 0; g < SLOTS; g++ )); do
        p=${BUSY[$g]:-}
        if [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; then break 2; fi
      done
      sleep 10
    done
    mkdir -p "$NODE_TMP/$phase$i" "$(dirname "$out")"
    echo "[$phase] slot $g gpu $(( g % NG )) [$grp]: $out"
    CUDA_VISIBLE_DEVICES=$(( g % NG )) SLURM_TMPDIR=$NODE_TMP/$phase$i bash -c "$cmd" \
      > "logs/$JOB-$phase$i.out" 2>&1 &
    BUSY[$g]=$!; TAG[$!]=$grp; OUT[$!]=$out
    i=$(( i + 1 ))
  done
  for (( g = 0; g < SLOTS; g++ )); do p=${BUSY[$g]:-}; [ -n "$p" ] && wait "$p"; done
  BUSY=()
}

echo "== phase 1: smoke ($(grep -cv '^\s*#' "$SMOKE") lines, $SLOTS slots)"
t0=$(date +%s)
run_phase "$SMOKE" smoke
OK=(); BAD=()
while IFS=$'\t' read -r grp out cmd; do
  [ -z "${grp:-}" ] || [[ "$grp" == \#* ]] && continue
  if [ -s "$out" ] && python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$out" 2>/dev/null; then OK+=("$grp")
  else BAD+=("$grp"); fi
done < "$SMOKE"
echo "smoke done $(( $(date +%s) - t0 ))s; passed: ${OK[*]:-none}; FAILED (skipped): ${BAD[*]:-none}"
(( ${#OK[@]} )) || { echo "every smoke failed"; kill $SMI; exit 1; }
ALLOW=$(IFS='|'; echo "${OK[*]}")
echo "== phase 2: full (groups: $ALLOW)"
run_phase "$FULL" full "$ALLOW"
kill $SMI 2>/dev/null
echo "MECH NODE DONE $(( $(date +%s) - t0 ))s; smoke-failed groups: ${BAD[*]:-none}"
