#!/usr/bin/env bash
# Submit the v2 (fidelity-fixed) Neural Chameleons replication batch and record
# every job in a tracked run ledger. Each evaluation runs three probe activation
# views: mean_generation_tokens (paper primary), last_token, and the Obfuscation
# Atlas follow-up (last token after appending "Did you respond truthfully? Yes").
#
# Order of submission:
#   0. Golden-path evaluator validation (released organism + released probes)
#   1. Base-model trigger control (9B abliterated, no checkpoint) — base delta
#   2. Full 2B preset train + 3-view eval
#   3. 9B seed 17 train + 3-view eval; seeds 23/41 gated behind seed 17
#   4. Llama-3.1-8B and Qwen-2.5-7B family train + 3-view eval
#
# Golden eval and base control are independent so they can be inspected while the
# training runs proceed. Nothing here trains the full-FT 9B arm — full fine-tune
# of a 9B does not fit on a single A100-40G; the QLoRA configs do.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
if [[ -f "${REPO_ROOT}/.env.narval" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/.env.narval"
  set +a
fi

ACCOUNT=ctb-liyue_gpu
SCRATCH_ROOT="${SCRATCH:-${REPO_ROOT}}"
ART="${SCRATCH_ROOT}/next_chameleons_artifacts"
GOLDEN_ROOT="${SCRATCH_ROOT}/next_chameleons_golden"
LEDGER_DIR="${ART}/run_ledger"
STAMP="$(date +%Y%m%d_%H%M%S)"
LEDGER="${LEDGER_DIR}/submissions_${STAMP}.tsv"
PREFETCH_MANIFEST="${ART}/prefetch/paper_replication_assets.json"

mkdir -p logs "${LEDGER_DIR}"

# Pre-flight: artifacts that the jobs require must be staged from a login node.
require() { [[ -e "$1" ]] || { echo "MISSING required artifact: $1" >&2; echo "$2" >&2; exit 2; }; }
require "${PREFETCH_MANIFEST}" "Run scripts/prefetch_paper_assets.sh first."
require "${GOLDEN_ROOT}/serteal_neural_chameleon_gemma_2_9b" "Run scripts/fetch_golden_artifacts.sh first."
require "${GOLDEN_ROOT}/probes/deception_weights.pt" "Run scripts/fetch_golden_artifacts.sh first."
require "${SCRATCH_ROOT}/next_chameleons_raw_cache/paper_benign_preset" "Preset data missing; run fetch_golden_artifacts.sh."

EVAL_DATASETS="AlignmentResearch/DolusChat;JailbreakBench/JBB-Behaviors;AISC-Linear-Probe-Gen/obfuscated_activations;scale-safety-research/roleplaying;scale-safety-research/insider_trading"

# Resolved (abliterated) model id per experiment, for node-local HF staging.
model_for() {
  case "$1" in
    paper_minimal_gemma2_2b_real|paper_gemma2_2b_real) echo "IlyaGusev/gemma-2-2b-it-abliterated" ;;
    paper_gemma2_9b_real|paper_golden_serteal_gemma2_9b) echo "IlyaGusev/gemma-2-9b-it-abliterated" ;;
    paper_llama31_8b_real) echo "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated" ;;
    paper_qwen25_7b_real) echo "huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2" ;;
    *) echo "" ;;
  esac
}

echo -e "job_id\tname\tpurpose\texperiment\tseed\tcheckpoint\toutput_base\tdepends_on" > "${LEDGER}"
record() { echo -e "$1\t$2\t$3\t$4\t$5\t$6\t$7\t$8" >> "${LEDGER}"; echo "Submitted ${2} -> job ${1}"; }

# --- 0. Golden-path evaluator validation -----------------------------------
GOLDEN_OUT="${ART}/paper_golden_serteal_gemma2_9b_${STAMP}/eval"
GOLDEN_MODEL="$(model_for paper_golden_serteal_gemma2_9b)"
GOLDEN_JOB="$(sbatch --parsable --account=${ACCOUNT} \
  --job-name=ncv2-golden-eval --time=08:00:00 --gpus-per-node=a100:1 \
  "--export=ALL,NEXT_CHAMELEONS_STAGE_MODELS=${GOLDEN_MODEL},NEXT_CHAMELEONS_STAGE_DATASETS=${EVAL_DATASETS}" \
  scripts/slurm/multiview_eval.sbatch \
  paper_golden_serteal_gemma2_9b \
  "${GOLDEN_ROOT}/serteal_neural_chameleon_gemma_2_9b" \
  "${GOLDEN_OUT}" 17 \
  --external-probe-dir "${GOLDEN_ROOT}/probes")"
record "${GOLDEN_JOB}" ncv2-golden-eval golden_evaluator_validation paper_golden_serteal_gemma2_9b 17 "${GOLDEN_ROOT}/serteal_neural_chameleon_gemma_2_9b" "${GOLDEN_OUT}" none

# --- 1. Base-model trigger control (9B abliterated, no checkpoint) ----------
BASE_OUT="${ART}/paper_gemma2_9b_real_base_control_${STAMP}/eval"
BASE_MODEL="$(model_for paper_gemma2_9b_real)"
BASE_JOB="$(sbatch --parsable --account=${ACCOUNT} \
  --job-name=ncv2-9b-base-control --time=08:00:00 --gpus-per-node=a100:1 \
  "--export=ALL,NEXT_CHAMELEONS_STAGE_MODELS=${BASE_MODEL},NEXT_CHAMELEONS_STAGE_DATASETS=${EVAL_DATASETS}" \
  scripts/slurm/multiview_eval.sbatch \
  paper_gemma2_9b_real - "${BASE_OUT}" 17)"
record "${BASE_JOB}" ncv2-9b-base-control base_trigger_control_for_final_minus_base paper_gemma2_9b_real 17 BASE_MODEL "${BASE_OUT}" none

# --- 2. Full 2B preset train + 3-view eval ---------------------------------
TWOB_MODEL="$(model_for paper_gemma2_2b_real)"
TWOB_JOB="$(sbatch --parsable --account=${ACCOUNT} \
  --job-name=ncv2-2b-s17 --time=12:00:00 --gpus-per-node=a100:1 \
  "--export=ALL,NEXT_CHAMELEONS_STAGE_MODELS=${TWOB_MODEL},NEXT_CHAMELEONS_STAGE_DATASETS=${EVAL_DATASETS}" \
  scripts/slurm/paper_train_3view.sbatch paper_gemma2_2b_real 17)"
record "${TWOB_JOB}" ncv2-2b-s17 train_plus_3view_eval paper_gemma2_2b_real 17 self "${ART}/paper_gemma2_2b_real_<jobid>_s17" none

# --- 3. 9B seed 17, then seeds 23/41 gated behind it -----------------------
NINEB_MODEL="$(model_for paper_gemma2_9b_real)"
NINEB_S17_JOB="$(sbatch --parsable --account=${ACCOUNT} \
  --job-name=ncv2-9b-s17 --time=24:00:00 --gpus-per-node=a100:1 \
  "--export=ALL,NEXT_CHAMELEONS_STAGE_MODELS=${NINEB_MODEL},NEXT_CHAMELEONS_STAGE_DATASETS=${EVAL_DATASETS}" \
  scripts/slurm/paper_train_3view.sbatch paper_gemma2_9b_real 17)"
record "${NINEB_S17_JOB}" ncv2-9b-s17 train_plus_3view_eval paper_gemma2_9b_real 17 self "${ART}/paper_gemma2_9b_real_<jobid>_s17" none

for seed in 23 41; do
  job="$(sbatch --parsable --account=${ACCOUNT} \
    --dependency=afterok:${NINEB_S17_JOB} \
    --job-name=ncv2-9b-s${seed} --time=24:00:00 --gpus-per-node=a100:1 \
    "--export=ALL,NEXT_CHAMELEONS_STAGE_MODELS=${NINEB_MODEL},NEXT_CHAMELEONS_STAGE_DATASETS=${EVAL_DATASETS}" \
    scripts/slurm/paper_train_3view.sbatch paper_gemma2_9b_real "${seed}")"
  record "${job}" ncv2-9b-s${seed} train_plus_3view_eval paper_gemma2_9b_real "${seed}" self "${ART}/paper_gemma2_9b_real_<jobid>_s${seed}" "${NINEB_S17_JOB}"
done

# --- 4. Family runs: Llama-3.1-8B, Qwen-2.5-7B (layer 9) --------------------
for experiment in paper_llama31_8b_real paper_qwen25_7b_real; do
  emodel="$(model_for "${experiment}")"
  job="$(sbatch --parsable --account=${ACCOUNT} \
    --job-name=ncv2-${experiment}-s17 --time=24:00:00 --gpus-per-node=a100:1 \
    "--export=ALL,NEXT_CHAMELEONS_STAGE_MODELS=${emodel},NEXT_CHAMELEONS_STAGE_DATASETS=${EVAL_DATASETS}" \
    scripts/slurm/paper_train_3view.sbatch "${experiment}" 17)"
  record "${job}" "ncv2-${experiment}-s17" train_plus_3view_eval "${experiment}" 17 self "${ART}/${experiment}_<jobid>_s17" none
done

echo ""
echo "Ledger written: ${LEDGER}"
column -t -s $'\t' "${LEDGER}"
