#!/usr/bin/env bash
# Train a full fine-tune chameleon organism (the paper's shipped setup:
# full-parameter Adam8bit, lr 8e-6, one epoch, teacher-forced CE behavior loss),
# then evaluate it on the corrected real data with the paper-faithful eval
# (chat-template + per-token probes). This is the experiment that tests whether
# full FT (vs QLoRA) recovers the paper's zero-shot transfer.
#
# Usage: submit_fullft.sh <experiment> <gpus> <seed> [seed ...]
#   <gpus> = a100 count (1 for 2B, 4 for 9B via device_map="auto" sharding).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${REPO_ROOT}"
if [[ -f "${REPO_ROOT}/.env.narval" ]]; then set -a; source "${REPO_ROOT}/.env.narval"; set +a; fi

EXP="${1:?experiment}"; GPUS="${2:?gpu count}"; shift 2
SEEDS=("$@"); [[ ${#SEEDS[@]} -eq 0 ]] && SEEDS=(17 23 41)
ART="${SCRATCH}/next_chameleons_artifacts"
STAMP="$(date +%Y%m%d_%H%M%S)"
LEDGER="${ART}/run_ledger/fullft_${EXP}_${STAMP}.tsv"; mkdir -p "${ART}/run_ledger" logs
printf "seed\ttrain_job\teval_job\toutput\n" > "${LEDGER}"

# Stage the abliterated base for offline compute nodes (paper-pinned for 9B).
case "${EXP}" in
  *2b*) MODEL="IlyaGusev/gemma-2-2b-it-abliterated" ;;
  *9b*) MODEL="IlyaGusev/gemma-2-9b-it-abliterated" ;;
  *) MODEL="" ;;
esac

for SEED in "${SEEDS[@]}"; do
  OUT="${ART}/${EXP}_${STAMP}_s${SEED}"
  # Train (full FT). device_map="auto" shards across the requested GPUs.
  TRAIN=$(sbatch --parsable --account=ctb-liyue_gpu --gpus-per-node="a100:${GPUS}" \
    --time=12:00:00 --job-name="ft-train-${EXP}-s${SEED}" \
    "--export=ALL,NEXT_CHAMELEONS_STAGE_MODELS=${MODEL}" \
    scripts/slurm/run_cli.sbatch \
    "uv run --extra train next-chameleons real-train ${EXP} --seed ${SEED} --output-dir ${OUT}")
  # Eval the resulting checkpoint with the corrected, chat-templated protocol
  # (3 views), gated on training success.
  CKPT="${OUT}/checkpoints/${EXP}"
  EVAL=$(sbatch --parsable --account=ctb-liyue_gpu --gpus-per-node=a100:1 \
    --dependency="afterok:${TRAIN}" --job-name="ft-eval-${EXP}-s${SEED}" \
    "--export=ALL,NEXT_CHAMELEONS_STAGE_MODELS=${MODEL}" \
    scripts/slurm/multiview_eval.sbatch "${EXP}" "${CKPT}" "${OUT}/eval" "${SEED}" --chat-template)
  printf "%s\t%s\t%s\t%s\n" "${SEED}" "${TRAIN}" "${EVAL}" "${OUT}" | tee -a "${LEDGER}"
done
echo "STAMP=${STAMP}"
echo "Ledger: ${LEDGER}"
