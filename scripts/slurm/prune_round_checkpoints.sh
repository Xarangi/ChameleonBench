#!/usr/bin/env bash
# Disk budget for uncapped games: an 18 GB checkpoint per attempt per round would
# exhaust scratch. Run after every consumer of round k (defender fit, cumulative +
# fresh scoreboards, gen-eval). Keeps: every KEEP_EVERY-th round's passing
# checkpoint, and ALL checkpoints of an exhausted round (no passing attempt).
# Always deletes model weights of non-exposed (failed) attempts in a passing round.
# Only *.safetensors are removed; reports, gate histories, and banks are kept.
# Usage: prune_round_checkpoints.sh <round_dir> <round_index> <keep_every>
set -uo pipefail
OUT="$1"; K="$2"; KEEP_EVERY="$3"
A="${OUT}/attempts.json"
[[ -f "${A}" ]] || { echo "[prune] no attempts.json in ${OUT}; keeping everything"; exit 0; }
passed=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['passed_attempt'])" "${A}")
exposed=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['exposed_attempt'])" "${A}")
if (( passed < 0 )); then echo "[prune] exhausted round ${K}: keeping all attempts"; exit 0; fi
for d in "${OUT}"/attempt*/; do
  j="${d%/}"; j="${j##*attempt}"
  if (( j != exposed )); then rm -f "${d}"checkpoints/*/model-*.safetensors; echo "[prune] removed failed attempt ${j} weights"; fi
done
if (( K % KEEP_EVERY != 0 )); then
  rm -f "${OUT}/attempt${exposed}"/checkpoints/*/model-*.safetensors
  echo "[prune] round ${K}: removed passing weights (kept every ${KEEP_EVERY})"
else
  echo "[prune] round ${K}: keeping passing weights"
fi
