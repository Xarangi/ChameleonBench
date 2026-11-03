#!/bin/bash
# RACE NODE (11-03): runs the pieces of an interactive race (scripts/race_launch.py writes the TSV and
# submits this script) with ONE piece process per GPU at a time; each piece runs many episodes at once
# (tica_switch_eval --batch 16 --fast-forks, colocated vLLM: TICA_VLLM=1 TICA_VLLM_SLEEP=0).
#   TSV lines:  OUT_JSON<TAB>COMMAND        (COMMAND runs from $TICA_REPO with CUDA_VISIBLE_DEVICES set)
# A piece whose OUT_JSON exists is skipped, so resubmitting the same TSV resumes a timed-out node.
# The vLLM venv is unpacked node-locally ONCE here (its python is exported to every piece).
#   sbatch --gres=gpu:h100:4 --export=ALL,TICA_REPO=<checkout> scripts/slurm/race_node.sh TSV
#
# Failure handling (11-03 fix; race mbpp_pass64 / 664118 silently lost 7 of 16 pieces and still
# reported COMPLETED): every piece's exit code is recorded; pieces without OUT_JSON after the pass are
# retried up to RACE_RETRIES times (default 1; partial outputs removed first); the job ends with a
# summary (each missing piece, its log and the last error line), writes TSV.status
# (OUT<TAB>ok|missing<TAB>attempts<TAB>last_rc<TAB>log) and EXITS 1 if any piece is missing.
# Per-piece scratch the reports do not read (OUT_STEM_work) is deleted after a successful piece.
# Env knobs (tests): NG (GPU slots), RACE_POLL (seconds between slot checks, default 10).
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --job-name=race-node
set -uo pipefail
TSV=${1:?piece TSV}
TSV=$(readlink -f "$TSV")
REPO=${TICA_REPO:?TICA_REPO = the checkout to run}
cd "$REPO" || exit 1
mkdir -p logs
NG=${NG:-$(nvidia-smi -L 2>/dev/null | wc -l)}
[ "$NG" -ge 1 ] || { echo "no GPUs (set NG)"; exit 1; }
POLL=${RACE_POLL:-10}
RETRIES=${RACE_RETRIES:-1}
JOB=${SLURM_JOB_NAME:-race}-${SLURM_JOB_ID:-local$$}
# job-private node-local dir (SLURM_TMPDIR is plain /tmp on TamIA, shared across jobs): removed at exit
NODE_TMP=${SLURM_TMPDIR:-/tmp}/race_${SLURM_JOB_ID:-$$}
mkdir -p "$NODE_TMP"
SMI=
cleanup() { [ -n "$SMI" ] && kill "$SMI" 2>/dev/null; rm -rf "$NODE_TMP"; }
trap cleanup EXIT
VV=${TICA_VLLM_VENV:-vllm-latest}
if [ -z "${TICA_VLLM_PY:-}" ] && [ -f "/scratch/s/sneheel/venvs/$VV.tar" ]; then
  t0=$(date +%s)
  tar -xf "/scratch/s/sneheel/venvs/$VV.tar" -C "$NODE_TMP" && export TICA_VLLM_PY="$NODE_TMP/$VV/bin/python"
  echo "vllm venv $VV unpacked in $(( $(date +%s) - t0 ))s"
fi
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv -l 60 \
    > "logs/$JOB-gpu.csv" 2>/dev/null &
  SMI=$!
fi
mapfile -t LINES < <(grep -v '^\s*#' "$TSV" | grep -v '^\s*$')
declare -A BUSY BUSY_I ATTEMPTS RC LOG
t0=$(date +%s)

reap() {  # collect the exit code of slot $1's finished piece
  local g=$1 p=${BUSY[$1]:-} i out
  [ -n "$p" ] || return 0
  i=${BUSY_I[$g]}
  wait "$p"; RC[$i]=$?
  IFS=$'\t' read -r out _ <<< "${LINES[$i]}"
  if [ -s "$out" ]; then
    rm -rf "${out%.json}_work"
  else
    echo "[$(( $(date +%s) - t0 ))s] piece $i FAILED (rc ${RC[$i]}): $out  log ${LOG[$i]}"
  fi
  BUSY[$g]=
}

run_pass() {  # run every piece without output, one per free slot
  local i out cmd g p
  for (( i = 0; i < ${#LINES[@]}; i++ )); do
    IFS=$'\t' read -r out cmd <<< "${LINES[$i]}"
    if [ -s "$out" ]; then
      [ -z "${ATTEMPTS[$i]:-}" ] && echo "skip (done): $out"
      ATTEMPTS[$i]=${ATTEMPTS[$i]:-0}; continue
    fi
    while true; do  # first idle slot
      for (( g = 0; g < NG; g++ )); do
        p=${BUSY[$g]:-}
        if [ -z "$p" ]; then break 2; fi
        if ! kill -0 "$p" 2>/dev/null; then reap "$g"; break 2; fi
      done
      sleep "$POLL"
    done
    ATTEMPTS[$i]=$(( ${ATTEMPTS[$i]:-0} + 1 ))
    LOG[$i]="logs/$JOB-$i.out"
    [ "${ATTEMPTS[$i]}" -gt 1 ] && LOG[$i]="logs/$JOB-$i-try${ATTEMPTS[$i]}.out"
    # a retry starts clean: the main / fork JSONL are rewritten by the piece, the work dir (probe
    # bank) of a failed attempt must not leak into the new one
    rm -rf "${out%.json}_work"
    mkdir -p "$NODE_TMP/gpu$g" "$(dirname "$out")"
    echo "[$(( $(date +%s) - t0 ))s] gpu $g piece $i (attempt ${ATTEMPTS[$i]}): $out"
    CUDA_VISIBLE_DEVICES=$g SLURM_TMPDIR=$NODE_TMP/gpu$g bash -c "$cmd" > "${LOG[$i]}" 2>&1 &
    BUSY[$g]=$!; BUSY_I[$g]=$i
  done
  for (( g = 0; g < NG; g++ )); do reap "$g"; done
}

run_pass
for (( r = 1; r <= RETRIES; r++ )); do
  n_miss=0
  for line in "${LINES[@]}"; do
    IFS=$'\t' read -r out _ <<< "$line"; [ -s "$out" ] || n_miss=$(( n_miss + 1 ))
  done
  [ "$n_miss" -eq 0 ] && break
  echo "[$(( $(date +%s) - t0 ))s] retry round $r: $n_miss piece(s) without output"
  run_pass
done

: > "$TSV.status"
FAILED=()
for (( i = 0; i < ${#LINES[@]}; i++ )); do
  IFS=$'\t' read -r out _ <<< "${LINES[$i]}"
  st=ok; [ -s "$out" ] || { st=missing; FAILED+=("$i"); }
  printf '%s\t%s\t%s\t%s\t%s\n' "$out" "$st" "${ATTEMPTS[$i]:-0}" "${RC[$i]:-}" "${LOG[$i]:-}" \
    >> "$TSV.status"
done
echo "RACE NODE DONE $(( $(date +%s) - t0 ))s; pieces: ${#LINES[@]}; missing: ${#FAILED[@]} (status: $TSV.status)"
for i in "${FAILED[@]}"; do
  IFS=$'\t' read -r out _ <<< "${LINES[$i]}"
  err=$(grep -E 'Error|error:|Traceback|denied|Killed' "${LOG[$i]:-/dev/null}" 2>/dev/null | tail -1)
  echo "  MISSING $out  (attempts ${ATTEMPTS[$i]:-0}, rc ${RC[$i]:-?}, log ${LOG[$i]:-none}): ${err:0:200}"
done
[ "${#FAILED[@]}" -eq 0 ]
