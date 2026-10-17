#!/usr/bin/env bash
# Game Exhaustion = a predeclared family of fully budgeted attack attempts all fail
# (CONTEXT.md). One optimizer run failing the gate is NOT exhaustion: per-round
# failure behaves like a constant-hazard coin flip (REPLICATION 2026-09-26e), so a
# single-attempt game measures optimizer luck. This wrapper runs up to K attempts
# with distinct seeds (seed + 100003*j) and stops at the first that passes.
#
# Each attempt writes <out>/attempt<j>/; the passing attempt (or, if every attempt
# fails, the LAST one, so the exhausted round can still be scored) is exposed at
# <out>/checkpoints -> attempt<j>/checkpoints. attempts.json records every outcome.
# Exit 0 iff some attempt passed; exit 1 if all K failed (downstream afterok jobs,
# i.e. the next round, are then cancelled and the game ends).
#
# Usage: train_with_restarts.sh <exp> <seed> <out_dir> <restarts>
set -uo pipefail
EXP="$1"; SEED="$2"; OUT="$3"; K="$4"
CLI="uv run --extra train next-chameleons"
mkdir -p "${OUT}"
results=()
passed=-1
for (( j=0; j<K; j++ )); do
  s=$(( SEED + 100003 * j ))
  echo "[restarts] attempt ${j}/${K} seed=${s} -> ${OUT}/attempt${j}"
  ${CLI} real-train "${EXP}" --seed "${s}" --output-dir "${OUT}/attempt${j}"
  code=$?
  hist="${OUT}/attempt${j}/checkpoints/${EXP}/evasion_gate_history.json"
  if [[ ${code} -eq 0 ]]; then
    results+=("{\"attempt\": ${j}, \"seed\": ${s}, \"outcome\": \"passed\"}")
    passed=${j}
    break
  elif [[ -f "${hist}" ]]; then
    results+=("{\"attempt\": ${j}, \"seed\": ${s}, \"outcome\": \"gate_failed\"}")
  else
    # No gate history = crash (OOM, node failure), not an evasion failure.
    results+=("{\"attempt\": ${j}, \"seed\": ${s}, \"outcome\": \"crashed\", \"exit\": ${code}}")
  fi
done
last=$(( ${#results[@]} - 1 ))
chosen=$(( passed >= 0 ? passed : last ))
rm -f "${OUT}/checkpoints"
ln -s "attempt${chosen}/checkpoints" "${OUT}/checkpoints"
( IFS=,; echo "{\"restarts\": ${K}, \"passed_attempt\": ${passed}, \"exposed_attempt\": ${chosen}, \"attempts\": [${results[*]}]}" ) > "${OUT}/attempts.json"
cat "${OUT}/attempts.json"
[[ ${passed} -ge 0 ]]
