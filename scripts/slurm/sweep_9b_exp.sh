#!/usr/bin/env bash
# Generic 9B full-FT seed-sweep submitter: EXP as $1, then seeds.
# 4xA100 + 128G train; 1-GPU afterok eval (self-fit probes). Gate afterwards with
# scripts/diag_converged_probe_gate.py on <artifacts>/<EXP>_s<seed>.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${REPO_ROOT}"
WRAP="scripts/slurm/run_cli.sbatch"; ART="${SCRATCH:-.}/next_chameleons_artifacts"
SUBMIT=0; if [[ "${1:-}" == "--submit" ]]; then SUBMIT=1; shift; fi
EXP="${1:?usage: sweep_9b_exp.sh [--submit] EXP SEED [SEED ...]}"; shift
[[ "$#" -ge 1 ]] || { echo "need at least one seed" >&2; exit 2; }
for seed in "$@"; do
  out="${ART}/${EXP}_s${seed}"; ckpt="${out}/checkpoints/${EXP}"; cli="uv run --extra train next-chameleons"
  tcmd="${cli} real-train ${EXP} --seed ${seed} --output-dir ${out}"
  ecmd="${cli} real-eval ${EXP} --seed ${seed} --checkpoint-dir ${ckpt} --output-dir ${out}"
  if [[ "${SUBMIT}" == "1" ]]; then
    tid="$(sbatch --parsable --time=05:00:00 --gpus-per-node=a100:4 --mem=128G "${WRAP}" "${tcmd}")"
    echo "train ${EXP} s${seed} -> ${tid} (4xA100)"
    eid="$(sbatch --parsable --dependency=afterok:"${tid}" "${WRAP}" "${ecmd}")"
    echo "eval  ${EXP} s${seed} -> ${eid} (afterok ${tid})"
  else
    echo "[dry] ${tcmd}"; echo "[dry] ${ecmd}"
  fi
done
