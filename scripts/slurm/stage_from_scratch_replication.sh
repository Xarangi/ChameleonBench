#!/usr/bin/env bash
# FROM-SCRATCH replication of the Neural Chameleons evasion-generalization result:
# OUR organism, OUR probes, ZERO released artifacts. The one change vs the fidelity
# arms is the obfuscation TARGET -- fit here with the authors' BROAD recipe
# (Kaiming-init nn.Linear, 1 epoch, no convergence gate) whose breadth is what makes
# organism evasion generalize (obfuscation-target-drives-generalization note).
#
# Each train job's eval runs with --dependency=afterok on the trained checkpoint and
# WITHOUT --external-probe-dir, so the held-out deception/harmful probes are fit by us.
# PASS: self-fit apollo_repe_deception tpr_drop >= ~+0.5 across seeds, with a real
# clean detector (clean_tpr high), and harmful ~0 (as the paper reports).
#
# Full-parameter 9B fine-tune does NOT fit on one 40GB A100 -> 4 GPUs (device_map
# shards) + 128G host RAM for paged_adam8bit. Eval is inference-only -> 1 GPU.
#
# Default PRINTS the commands; pass --submit to queue on Narval (A100, run_cli.sbatch).
# Prereqs in $SCRATCH (offline nodes): official preset JSON + staged base gemma-2-9b.
# NOTE: NO released probe weights are required for this arm.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
WRAP="scripts/slurm/run_cli.sbatch"
ART="${SCRATCH:-.}/next_chameleons_artifacts"
EXP="paper_gemma2_9b_real_fullft_scratch"
SUBMIT=0
[[ "${1:-}" == "--submit" ]] && SUBMIT=1

submit_pair() {
  local seed="$1" time="${2:-05:00:00}"
  local out="${ART}/${EXP}_s${seed}"
  local ckpt="${out}/checkpoints/${EXP}"
  local cli="uv run --extra train next-chameleons"
  local tcmd="${cli} real-train ${EXP} --seed ${seed} --output-dir ${out}"
  # Self-fit eval: NO --external-probe-dir. Our own per-token deception/harmful probes.
  local ecmd="${cli} real-eval ${EXP} --seed ${seed} --checkpoint-dir ${ckpt} --output-dir ${out}"
  if [[ "${SUBMIT}" == "1" ]]; then
    local tid
    tid="$(sbatch --parsable --time="${time}" --gpus-per-node=a100:4 --mem=128G \
      "${WRAP}" "${tcmd}")"
    echo "train ${EXP} s${seed} -> job ${tid} (4xA100)"
    local eid
    eid="$(sbatch --parsable --dependency=afterok:"${tid}" "${WRAP}" "${ecmd}")"
    echo "eval  ${EXP} s${seed} -> job ${eid} (afterok ${tid}, self-fit probes)"
  else
    echo "sbatch --time=${time} --gpus-per-node=a100:4 --mem=128G ${WRAP} \"${tcmd}\""
    echo "sbatch --dependency=afterok:<trainid> ${WRAP} \"${ecmd}\""
  fi
}

echo "# From-scratch generalization replication (submit=${SUBMIT}). Output ${ART}/${EXP}_s<seed>/"
echo "# Obfuscation target: BROAD self-fit (Kaiming/1-epoch). Eval: OUR probes, no released weights."
echo
echo "## Seed 17 first as the decision gate; 23/41 confirm robustness to the random target init."
submit_pair 17
submit_pair 23
submit_pair 41
[[ "${SUBMIT}" == "0" ]] && echo "# (dry run -- re-run with --submit to queue on Narval)"
