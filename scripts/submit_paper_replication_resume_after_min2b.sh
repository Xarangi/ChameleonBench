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
mkdir -p logs runs

ACCOUNT_ARGS=(--account=ctb-liyue_gpu)
RUNNER="scripts/slurm/run_cli.sbatch"
PILOT_ARTIFACT_DIR="${1:-${SCRATCH:-${REPO_ROOT}}/next_chameleons_artifacts/paper_minimal_gemma2_2b_real_62286553}"
PILOT_CHECKPOINT_DIR="${PILOT_ARTIFACT_DIR}/checkpoints/paper_minimal_gemma2_2b_real"

if [[ ! -f "${PILOT_CHECKPOINT_DIR}/adapter_model.safetensors" ]]; then
  echo "Missing minimal 2B pilot checkpoint: ${PILOT_CHECKPOINT_DIR}" >&2
  exit 2
fi

EVAL_JOB="$(
  sbatch --parsable "${ACCOUNT_ARGS[@]}" \
    --job-name=next-paper-min2b-eval-retry \
    --time=03:00:00 \
    --gpus-per-node=a100:1 \
    "--export=ALL,NEXT_CHAMELEONS_STAGE_MODELS=google/gemma-2-2b-it,NEXT_CHAMELEONS_STAGE_DATASETS=AlignmentResearch/DolusChat;JailbreakBench/JBB-Behaviors;scale-safety-research/roleplaying;scale-safety-research/insider_trading" \
    "${RUNNER}" \
    "uv run --extra train next-chameleons real-eval paper_minimal_gemma2_2b_real --seed 17 --checkpoint-dir ${PILOT_CHECKPOINT_DIR} --output-dir ${PILOT_ARTIFACT_DIR}"
)"
echo "Submitted minimal 2B eval retry ${EVAL_JOB}"

FULL_2B_JOB="$(
  sbatch --parsable "${ACCOUNT_ARGS[@]}" \
    --dependency=afterok:${EVAL_JOB} \
    --job-name=next-paper-2b-s17 \
    --time=12:00:00 \
    --gpus-per-node=a100:1 \
    "scripts/slurm/paper_replication.sbatch" \
    paper_gemma2_2b_real 17
)"
echo "Submitted full 2B seed-17 job ${FULL_2B_JOB}"

PRIMARY_JOB="$(
  sbatch --parsable "${ACCOUNT_ARGS[@]}" \
    --dependency=afterok:${EVAL_JOB} \
    --job-name=next-paper-9b-s17 \
    --time=24:00:00 \
    --gpus-per-node=a100:2 \
    "scripts/slurm/paper_replication.sbatch" \
    paper_gemma2_9b_real 17
)"
echo "Submitted 9B seed-17 job ${PRIMARY_JOB}"

for seed in 23 41; do
  job="$(
    sbatch --parsable "${ACCOUNT_ARGS[@]}" \
      --dependency=afterok:${PRIMARY_JOB} \
      --job-name=next-paper-9b-s${seed} \
      --time=24:00:00 \
      --gpus-per-node=a100:2 \
      "scripts/slurm/paper_replication.sbatch" \
      paper_gemma2_9b_real "${seed}"
  )"
  echo "Submitted 9B seed-${seed} job ${job}"
done

for experiment in paper_llama31_8b_real paper_qwen25_7b_real; do
  job="$(
    sbatch --parsable "${ACCOUNT_ARGS[@]}" \
      --dependency=afterok:${EVAL_JOB} \
      --job-name="next-${experiment}-s17" \
      --time=24:00:00 \
      --gpus-per-node=a100:2 \
      "scripts/slurm/paper_replication.sbatch" \
      "${experiment}" 17
  )"
  echo "Submitted ${experiment} seed-17 job ${job}"
done
