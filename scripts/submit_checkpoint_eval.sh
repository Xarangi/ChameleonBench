#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
if [[ -f "${REPO_ROOT}/.env.narval" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/.env.narval"
  set +a
fi

EXPERIMENT="${1:?Usage: $0 <experiment> <checkpoint-dir> <output-dir> [seed]}"
CHECKPOINT_DIR="${2:?Usage: $0 <experiment> <checkpoint-dir> <output-dir> [seed]}"
OUTPUT_DIR="${3:?Usage: $0 <experiment> <checkpoint-dir> <output-dir> [seed]}"
SEED="${4:-17}"

if [[ ! -f "${CHECKPOINT_DIR}/adapter_model.safetensors" ]]; then
  echo "Missing LoRA checkpoint: ${CHECKPOINT_DIR}/adapter_model.safetensors" >&2
  exit 2
fi

mkdir -p logs runs

case "${EXPERIMENT}" in
  paper_minimal_gemma2_2b_real|paper_gemma2_2b_real|adaptive_gemma2_2b_real)
    STAGE_MODELS="google/gemma-2-2b-it"
    ;;
  paper_gemma2_9b_real|language_generalization_real)
    STAGE_MODELS="IlyaGusev/gemma-2-9b-it-abliterated"
    ;;
  paper_llama31_8b_real)
    STAGE_MODELS="meta-llama/Llama-3.1-8B-Instruct"
    ;;
  paper_qwen25_7b_real)
    STAGE_MODELS="Qwen/Qwen2.5-7B-Instruct"
    ;;
  *)
    STAGE_MODELS=""
    ;;
esac

STAGE_DATASETS="${NEXT_CHAMELEONS_STAGE_DATASETS:-AlignmentResearch/DolusChat;JailbreakBench/JBB-Behaviors;scale-safety-research/roleplaying;scale-safety-research/insider_trading}"

job="$(
  sbatch --parsable --account=ctb-liyue_gpu \
    --job-name="next-eval-${EXPERIMENT}" \
    --time=04:00:00 \
    --gpus-per-node=a100:1 \
    "--export=ALL,NEXT_CHAMELEONS_STAGE_MODELS=${STAGE_MODELS},NEXT_CHAMELEONS_STAGE_DATASETS=${STAGE_DATASETS}" \
    scripts/slurm/run_cli.sbatch \
    "uv run --extra train next-chameleons real-eval ${EXPERIMENT} --seed ${SEED} --checkpoint-dir ${CHECKPOINT_DIR} --output-dir ${OUTPUT_DIR}"
)"
echo "Submitted eval job ${job}"
