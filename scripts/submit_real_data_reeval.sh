#!/usr/bin/env bash
# Re-evaluate the trained 9B organisms on the REBUILT real safety data
# (dolus paired responses, wildchat harmful completions, apollo rollouts).
#
# Prerequisite (run once on an internet-capable login node):
#   export NEXT_CHAMELEONS_RAW_CACHE_ROOT="$SCRATCH/next_chameleons_raw_cache"
#   uv run --extra train next-chameleons paper-materialize-safety-data paper_gemma2_9b_real \
#     --raw-cache-root "$NEXT_CHAMELEONS_RAW_CACHE_ROOT" --max-examples-per-split 512
#
# Submits, per probe mode (released frozen probes AND our post-hoc per-token):
#   - golden re-eval (released organism)            : evaluator+data validation gate
#   - 9B base control                               : final-minus-base baseline
#   - 9B trained seeds 17/23/41                      : learned obfuscation
# Each job runs all three activation views via multiview_eval.sbatch.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
if [[ -f "${REPO_ROOT}/.env.narval" ]]; then
  set -a; source "${REPO_ROOT}/.env.narval"; set +a
fi

ART="${SCRATCH}/next_chameleons_artifacts"
GOLDEN="${SCRATCH}/next_chameleons_golden"
PROBES="${GOLDEN}/probes"
STAMP="$(date +%Y%m%d_%H%M%S)"
LEDGER="${ART}/run_ledger/real_data_reeval_${STAMP}.tsv"
mkdir -p "${ART}/run_ledger" logs
printf "job_id\trole\tprobe_mode\tcheckpoint\toutput\n" > "${LEDGER}"

EXP="paper_gemma2_9b_real"
ACCOUNT="ctb-liyue_gpu"

# Set CHAT_TEMPLATE=1 to wrap prompts in the model chat template and pool over
# the response turn (matches the released probes' training distribution). The
# output dirs get a "_ct" suffix so templated and raw runs do not collide.
CT_ARG=()
CT_SUFFIX=""
if [[ "${CHAT_TEMPLATE:-0}" == "1" ]]; then
  CT_ARG=(--chat-template)
  CT_SUFFIX="_ct"
fi

submit() {  # role  probe_mode  checkpoint_dir  output_base  seed
  local role="$1" mode="$2" ckpt="$3" out="${4}${CT_SUFFIX}" seed="$5"
  local extra=("${CT_ARG[@]}")
  [[ "${mode}" == "released" ]] && extra+=(--external-probe-dir "${PROBES}")
  local job
  job="$(sbatch --parsable --account="${ACCOUNT}" \
    --job-name="reeval-${role}-${mode}" \
    scripts/slurm/multiview_eval.sbatch \
    "${EXP}" "${ckpt}" "${out}" "${seed}" "${extra[@]}")"
  printf "%s\t%s\t%s\t%s\t%s\n" "${job}" "${role}" "${mode}" "${ckpt}" "${out}" | tee -a "${LEDGER}"
}

# Released-organism golden re-validation on the rebuilt real data (released probes only).
submit golden released \
  "${GOLDEN}/serteal_neural_chameleon_gemma_2_9b" \
  "${ART}/reeval_${STAMP}/golden_released" 17

for mode in released posthoc; do
  # Base control (checkpoint '-' evaluates the configured base model).
  submit base "${mode}" "-" "${ART}/reeval_${STAMP}/9b_base_${mode}" 17
  for seed in 17 23 41; do
    case "${seed}" in
      17) jid=62813727;; 23) jid=62813728;; 41) jid=62813729;;
    esac
    ckpt="${ART}/paper_gemma2_9b_real_${jid}_s${seed}/checkpoints/paper_gemma2_9b_real"
    submit "s${seed}" "${mode}" "${ckpt}" "${ART}/reeval_${STAMP}/9b_s${seed}_${mode}" "${seed}"
  done
done

echo "Ledger: ${LEDGER}"
echo "Analyze when done, e.g.:"
echo "  python scripts/analyze_final_minus_base.py \\"
echo "    --base ${ART}/reeval_${STAMP}/9b_base_released_mean_generation_tokens \\"
echo "    ${ART}/reeval_${STAMP}/9b_s17_released_mean_generation_tokens \\"
echo "    ${ART}/reeval_${STAMP}/9b_s23_released_mean_generation_tokens \\"
echo "    ${ART}/reeval_${STAMP}/9b_s41_released_mean_generation_tokens"
