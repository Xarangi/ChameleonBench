#!/usr/bin/env bash
# DIAGNOSTIC #1 (2026-07-25): re-score the qwen3bringup5c_s17 arms-race rounds at
# LOOSER operating points (target-fpr 0.05 and 0.10) WITHOUT retraining, reusing the
# round checkpoints + accumulated banks already on disk. Purpose: separate the two
# live readings of the flat 1%-FPR curve --
#   (b) objective/operating-point MISALIGNMENT: the attacker drove residual scores to
#       ~0.01 but the 1%-FPR thresholds are 1e-3..3e-3, so it stays caught. If evasion
#       APPEARS at looser FPR, residual-0.01 scores are slipping under => misalignment.
#   (a) genuine defender ROBUSTNESS: if evasion stays ~0 even at 10% FPR, that's real.
# Cumulative bank for round k = bank_r0..bank_r(k-1), reduction MAX (scoreboard rule).
set -euo pipefail

GAME_ID="${GAME_ID:-qwen3bringup5c_s17}"
EXP="${EXP:-paper_qwen3_4b_arms_race}"
ROUNDS="${ROUNDS:-5}"
FPRS="${FPRS:-0.05 0.10}"
SUBMIT="${SUBMIT:-0}"

ART="${SCRATCH:-.}/next_chameleons_artifacts"
GAME="${ART}/arms_race/${GAME_ID}"
BANKS="${GAME}/banks"
WRAP="scripts/slurm/run_cli.sbatch"
CLI="uv run --extra train next-chameleons"

echo "# Re-score game=${GAME_ID} rounds=1..${ROUNDS} fprs='${FPRS}' reduction=max (submit=${SUBMIT})"
for (( k=1; k<=ROUNDS; k++ )); do
  OUT="${GAME}/round${k}"
  CKPT="${OUT}/checkpoints/${EXP}"
  BANK_ARGS=""
  for (( j=0; j<k; j++ )); do BANK_ARGS+=" --bank ${BANKS}/bank_r${j}.pt"; done
  CMD=""
  for fpr in ${FPRS}; do
    tag="${fpr/./}"   # 0.05 -> 005, 0.10 -> 010
    step="${CLI} probe-bank-eval ${EXP} --checkpoint-dir ${CKPT}${BANK_ARGS} --reduction max --target-fpr ${fpr} --output ${OUT}/scoreboard_cumulative_fpr${tag}.json"
    [[ -z "${CMD}" ]] && CMD="${step}" || CMD="${CMD} && ${step}"
  done
  args=(--parsable --time=02:00:00 --gpus-per-node=a100:1 --mem=64G --export=ALL)
  if [[ "${SUBMIT}" == "1" ]]; then
    JID=$(sbatch "${args[@]}" "${WRAP}" "${CMD}")
    echo "round ${k} re-score -> ${JID}"
  else
    echo "sbatch ${args[*]} ${WRAP} \"${CMD}\"" >&2
  fi
done
