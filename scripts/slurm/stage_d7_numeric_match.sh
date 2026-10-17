#!/usr/bin/env bash
# Stage the FULL GPU runs that confirm the paper-fidelity fixes numerically after
# D7 (chat-template conditioning), verbatim eval triggers, and the in-distribution
# fixed-FPR report. See REPLICATION.md (2026-06-27 entries).
#
# Run the VALIDATION wave first (golden eval + a fast `--max-examples 96` train) and
# confirm it is green before firing this — these are ~3-5h 9B full-FT jobs.
#
# Default PRINTS the commands; pass --submit to queue them (A100, run_cli.sbatch).
# Each train job's eval is submitted with --dependency=afterok and the correct
# --checkpoint-dir, so the eval runs on the trained organism (not the base model).
#
# Prereqs in $SCRATCH (offline nodes): released probes + serteal organism
# (scripts/fetch_golden_artifacts.sh), the official preset JSON, staged base model.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
WRAP="scripts/slurm/run_cli.sbatch"
ART="${SCRATCH:-.}/next_chameleons_artifacts"
SUBMIT=0
[[ "${1:-}" == "--submit" ]] && SUBMIT=1

# Submit a train job and its dependent eval (correct checkpoint dir + afterok dep).
submit_pair() {
  local exp="$1" seed="$2" time="${3:-05:00:00}"
  local out="${ART}/${exp}_s${seed}"
  local ckpt="${out}/checkpoints/${exp}"
  local cli="uv run --extra train next-chameleons"
  local tcmd="${cli} real-train ${exp} --seed ${seed} --output-dir ${out}"
  local ecmd="${cli} real-eval ${exp} --seed ${seed} --checkpoint-dir ${ckpt} --output-dir ${out}"
  if [[ "${SUBMIT}" == "1" ]]; then
    local tid
    tid="$(sbatch --parsable --time="${time}" "${WRAP}" "${tcmd}")"
    echo "train ${exp} s${seed} -> job ${tid}"
    local eid
    eid="$(sbatch --parsable --dependency=afterok:"${tid}" "${WRAP}" "${ecmd}")"
    echo "eval  ${exp} s${seed} -> job ${eid} (afterok ${tid})"
  else
    echo "sbatch --time=${time} ${WRAP} \"${tcmd}\""
    echo "sbatch --dependency=afterok:<trainid> ${WRAP} \"${ecmd}\""
  fi
}

echo "# D7 numeric-match full runs (submit=${SUBMIT}). Output under ${ART}/<exp>_s<seed>/"
echo
echo "## KEY scientific test: self-fit benign probes + D7 (does D7 restore OOD transfer?)"
submit_pair paper_gemma2_9b_real_fullft 17
echo
echo "## Headline organism: released-probe target (accum 8 = prior best) + D7, 3 seeds"
submit_pair paper_gemma2_9b_real_fullft_releasedprobes 17
submit_pair paper_gemma2_9b_real_fullft_releasedprobes 23
submit_pair paper_gemma2_9b_real_fullft_releasedprobes 41
echo
echo "## ACCUM sweep arm: paper-exact accum 1 (vs accum 8 above) + D7"
submit_pair paper_gemma2_9b_real_fullft_releasedprobes_step1 17
[[ "${SUBMIT}" == "0" ]] && echo "# (dry run — re-run with --submit to queue on Narval)"
