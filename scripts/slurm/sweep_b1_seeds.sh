#!/usr/bin/env bash
# Seed sweep of the paper-exact from-scratch replication arm
# (paper_gemma2_9b_real_fullft_scratch_b1: preset data + broad Kaiming target +
# batch_size 1 / no accumulation + probe batch 16 + seeded per-epoch shuffle).
#
# WHY: under this exact recipe, converged-probe deception transfer is
# SEED-DEPENDENT -- seeds 17/23 -> +0.000, seed 41 -> +0.539 (best trigger),
# mean +0.20. That is genuine cross-axis concealment on OUR organism with ZERO
# released artifacts, but rare/weak per seed vs serteal's cherry-picked "_GOOD"
# checkpoint (+1.0). This sweep characterizes the transfer DISTRIBUTION across
# additional seeds: estimate the rate of non-zero transfer and try to catch a
# strong-transfer basin (apollo drop >= ~+0.9) that matches the paper magnitude.
#
# Each train job: full-parameter 9B FT -> 4xA100 + 128G host RAM (paged_adam8bit).
# Eval: inference-only, 1 GPU, --dependency=afterok, self-fit probes (no released
# weights). Gate afterwards with scripts/diag_converged_probe_gate.py on _s<seed>.
#
# Usage:
#   scripts/slurm/sweep_b1_seeds.sh [--submit] SEED [SEED ...]
# Default (no --submit) PRINTS the sbatch commands.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
WRAP="scripts/slurm/run_cli.sbatch"
ART="${SCRATCH:-.}/next_chameleons_artifacts"
EXP="paper_gemma2_9b_real_fullft_scratch_b1"
SUBMIT=0
if [[ "${1:-}" == "--submit" ]]; then
  SUBMIT=1
  shift
fi
if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 [--submit] SEED [SEED ...]" >&2
  exit 2
fi

submit_pair() {
  local seed="$1" time="${2:-05:00:00}"
  local out="${ART}/${EXP}_s${seed}"
  local ckpt="${out}/checkpoints/${EXP}"
  local cli="uv run --extra train next-chameleons"
  local tcmd="${cli} real-train ${EXP} --seed ${seed} --output-dir ${out}"
  local ecmd="${cli} real-eval ${EXP} --seed ${seed} --checkpoint-dir ${ckpt} --output-dir ${out}"
  if [[ "${SUBMIT}" == "1" ]]; then
    local tid
    tid="$(sbatch --parsable --time="${time}" --gpus-per-node=a100:4 --mem=128G \
      "${WRAP}" "${tcmd}")"
    echo "train ${EXP} s${seed} -> job ${tid} (4xA100) -> ${out}"
    local eid
    eid="$(sbatch --parsable --dependency=afterok:"${tid}" "${WRAP}" "${ecmd}")"
    echo "eval  ${EXP} s${seed} -> job ${eid} (afterok ${tid}, self-fit probes)"
  else
    echo "sbatch --time=${time} --gpus-per-node=a100:4 --mem=128G ${WRAP} \"${tcmd}\""
    echo "sbatch --dependency=afterok:<trainid> ${WRAP} \"${ecmd}\""
  fi
}

echo "# b1 seed sweep (submit=${SUBMIT}). Output ${ART}/${EXP}_s<seed>/"
for seed in "$@"; do
  submit_pair "${seed}"
done
[[ "${SUBMIT}" == "0" ]] && echo "# (dry run -- re-run with --submit to queue on Narval)"
