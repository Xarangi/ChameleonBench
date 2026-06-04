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

RUNNER="scripts/slurm/run_cli.sbatch"
ACCOUNT_ARGS=(--account=ctb-liyue_gpu)
PREFETCH_MANIFEST="${SCRATCH:-${REPO_ROOT}}/next_chameleons_artifacts/prefetch/paper_replication_assets.json"
RAW_DATA_PATH="${SCRATCH:-${REPO_ROOT}}/next_chameleons_raw_cache/paper_benign_concepts/examples.jsonl"
APOLLO_DATA_PATH="${SCRATCH:-${REPO_ROOT}}/next_chameleons_raw_cache/apollo_repe_deception/examples.jsonl"
SYNTHETIC_HARMFUL_PATH="${SCRATCH:-${REPO_ROOT}}/next_chameleons_raw_cache/synthetic_harmful/examples.jsonl"

if [[ ! -f "${PREFETCH_MANIFEST}" ]]; then
  echo "Missing prefetch manifest: ${PREFETCH_MANIFEST}" >&2
  echo "Run scripts/prefetch_paper_assets.sh from an internet-capable login context first." >&2
  exit 2
fi

SMOKE_JOB="$(
  sbatch --parsable "${ACCOUNT_ARGS[@]}" \
    --job-name=next-paper-smoke \
    --time=01:00:00 \
    --gpus-per-node=a100:1 \
    "${RUNNER}" \
    'uv run next-chameleons paper-materialize-data --smoke --examples-per-concept 2 --artifact-root "${NEXT_CHAMELEONS_ARTIFACT_ROOT}/smoke" --raw-cache-root "${NEXT_CHAMELEONS_RAW_CACHE_ROOT}/smoke" && uv run next-chameleons paper-data-check paper_minimal_gemma2_2b_real --artifact-root "${NEXT_CHAMELEONS_ARTIFACT_ROOT}/smoke" --raw-cache-root "${NEXT_CHAMELEONS_RAW_CACHE_ROOT}/smoke" && uv run next-chameleons paper-readiness-check paper_minimal_gemma2_2b_real'
)"
echo "Submitted smoke/readiness job ${SMOKE_JOB}"

if [[ ! -f "${RAW_DATA_PATH}" ]]; then
  DATA_GEN_JOB="$(
    sbatch --parsable "${ACCOUNT_ARGS[@]}" \
      --dependency=afterok:${SMOKE_JOB} \
      --job-name=next-paper-data-gen \
      --time=24:00:00 \
      --gpus-per-node=a100:4 \
      "--export=ALL,NEXT_CHAMELEONS_STAGE_MODELS=google/gemma-2-27b-it;Qwen/Qwen3.5-27B" \
      "${RUNNER}" \
      'uv run --extra generation next-chameleons paper-materialize-data --generate --examples-per-concept 500 --generation-batch-size 8 --rating-method llm --rater-model "${NEXT_CHAMELEONS_RATER_MODEL:-Qwen/Qwen3.5-27B}" --rater-batch-size 16'
  )"
  echo "Submitted full benign concept data-generation job ${DATA_GEN_JOB}"
  DATA_DEPENDENCY="${DATA_GEN_JOB}"
else
  echo "Found existing paper benign concept data at ${RAW_DATA_PATH}"
  DATA_DEPENDENCY="${SMOKE_JOB}"
fi

if [[ ! -f "${APOLLO_DATA_PATH}" || ! -f "${SYNTHETIC_HARMFUL_PATH}" ]]; then
  SAFETY_DATA_JOB="$(
    sbatch --parsable "${ACCOUNT_ARGS[@]}" \
      --dependency=afterok:${DATA_DEPENDENCY} \
      --job-name=next-paper-safety-data \
      --time=01:00:00 \
      --gpus-per-node=a100:1 \
      "--export=ALL,NEXT_CHAMELEONS_STAGE_DATASETS=JailbreakBench/JBB-Behaviors;scale-safety-research/roleplaying;scale-safety-research/insider_trading" \
      "${RUNNER}" \
      'uv run --extra train next-chameleons paper-materialize-safety-data paper_gemma2_2b_real'
  )"
  echo "Submitted safety-source materialization job ${SAFETY_DATA_JOB}"
  DATA_DEPENDENCY="${SAFETY_DATA_JOB}"
else
  echo "Found Apollo and synthetic harmful safety sources in raw cache"
fi

DATA_CHECK_JOB="$(
  sbatch --parsable "${ACCOUNT_ARGS[@]}" \
    --dependency=afterok:${DATA_DEPENDENCY} \
    --job-name=next-paper-data-check \
    --time=01:00:00 \
    --gpus-per-node=a100:1 \
    "${RUNNER}" \
    'uv run next-chameleons paper-data-check paper_minimal_gemma2_2b_real --require-rated && uv run next-chameleons paper-data-check paper_gemma2_2b_real --require-rated'
)"
echo "Submitted full benign concept data-check job ${DATA_CHECK_JOB}"

PILOT_JOB="$(
  sbatch --parsable "${ACCOUNT_ARGS[@]}" \
    --dependency=afterok:${DATA_CHECK_JOB} \
    --job-name=next-paper-min2b-s17 \
    --time=12:00:00 \
    --gpus-per-node=a100:1 \
    "scripts/slurm/paper_replication.sbatch" \
    paper_minimal_gemma2_2b_real 17
)"
echo "Submitted minimal 2B pilot job ${PILOT_JOB}"

FULL_2B_JOB="$(
  sbatch --parsable "${ACCOUNT_ARGS[@]}" \
    --dependency=afterok:${PILOT_JOB} \
    --job-name=next-paper-2b-s17 \
    --time=12:00:00 \
    --gpus-per-node=a100:1 \
    "scripts/slurm/paper_replication.sbatch" \
    paper_gemma2_2b_real 17
)"
echo "Submitted full 2B seed-17 job ${FULL_2B_JOB}"

PRIMARY_JOB="$(
  sbatch --parsable "${ACCOUNT_ARGS[@]}" \
    --dependency=afterok:${FULL_2B_JOB} \
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
      --dependency=afterok:${PRIMARY_JOB} \
      --job-name="next-${experiment}-s17" \
      --time=24:00:00 \
      --gpus-per-node=a100:2 \
      "scripts/slurm/paper_replication.sbatch" \
      "${experiment}" 17
  )"
  echo "Submitted ${experiment} seed-17 job ${job}"
done
