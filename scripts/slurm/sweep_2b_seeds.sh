#!/usr/bin/env bash
# FAST 2B seed sweep of the paper-faithful from-scratch recipe
# (paper_gemma2_2b_real_fullft_scratch_b1). 2B full-FT fits on ONE A100
# (~15GB) and trains in minutes -> a 5-10x faster feedback loop than the 9B
# 4xA100 arm. Doubles as a CAPACITY PROBE: the paper only shows the evasion-
# generalization effect at 7-9B and never released a 2B organism, so a non-zero
# converged apollo-deception transfer at 2B would establish 2B as a valid fast
# testbed; a null across seeds is evidence the effect needs scale.
#
# Train + eval are BOTH single-GPU here. Eval uses self-fit probes (no released
# weights). Gate afterwards with scripts/diag_converged_probe_gate.py on _s<seed>.
#
# Usage: scripts/slurm/sweep_2b_seeds.sh [--submit] SEED [SEED ...]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
WRAP="scripts/slurm/run_cli.sbatch"
ART="${SCRATCH:-.}/next_chameleons_artifacts"
EXP="paper_gemma2_2b_real_fullft_scratch_b1"
SUBMIT=0
if [[ "${1:-}" == "--submit" ]]; then SUBMIT=1; shift; fi
if [[ "$#" -lt 1 ]]; then echo "Usage: $0 [--submit] SEED [SEED ...]" >&2; exit 2; fi

submit_pair() {
  local seed="$1" time="${2:-03:00:00}"
  local out="${ART}/${EXP}_s${seed}"
  local ckpt="${out}/checkpoints/${EXP}"
  local cli="uv run --extra train next-chameleons"
  local tcmd="${cli} real-train ${EXP} --seed ${seed} --output-dir ${out}"
  local ecmd="${cli} real-eval ${EXP} --seed ${seed} --checkpoint-dir ${ckpt} --output-dir ${out}"
  if [[ "${SUBMIT}" == "1" ]]; then
    local tid
    tid="$(sbatch --parsable --time="${time}" --gpus-per-node=a100:1 --mem=64G "${WRAP}" "${tcmd}")"
    echo "train ${EXP} s${seed} -> job ${tid} (1xA100) -> ${out}"
    local eid
    eid="$(sbatch --parsable --dependency=afterok:"${tid}" "${WRAP}" "${ecmd}")"
    echo "eval  ${EXP} s${seed} -> job ${eid} (afterok ${tid}, self-fit probes)"
  else
    echo "sbatch --time=${time} --gpus-per-node=a100:1 --mem=64G ${WRAP} \"${tcmd}\""
    echo "sbatch --dependency=afterok:<trainid> ${WRAP} \"${ecmd}\""
  fi
}

echo "# 2B seed sweep (submit=${SUBMIT}). Output ${ART}/${EXP}_s<seed>/"
for seed in "$@"; do submit_pair "${seed}"; done
[[ "${SUBMIT}" == "0" ]] && echo "# (dry run -- re-run with --submit to queue on Narval)"
