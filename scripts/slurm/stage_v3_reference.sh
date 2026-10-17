#!/usr/bin/env bash
# v3 reference + pre-flight jobs (2026-09-27 audit). Run ONCE before Stage A.
# Everything here is on the BASE model; per-game launches read its outputs.
#   1. All-concept reference banks at L12/L20/L28, fit on the split_v4 'fit' split
#      (authors'-rated). The fit report is the PRE-FLIGHT: every concept must reach
#      balanced accuracy >= 0.97 at L12, or its games cannot start.
#   2. Base scoreboards with those banks on held-out TEST, matching trigger per concept:
#      probe quality on the out-set (pre-registered inclusion bar AUROC_clean >= 0.95)
#      and the trigger-alone control (base AUROC_trig must stay ~AUROC_clean).
#   3. Judge validation vs the authors' published ratings (D3 needs a valid judge).
#   4. Base capability (MMLU + GSM8K), clean vs each concept's trigger: the baseline
#      every organism's capability tax is measured against.
# Usage: scripts/slurm/stage_v3_reference.sh [--submit]
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
WRAP="scripts/slurm/run_cli.sbatch"
ART="${SCRATCH:-.}/next_chameleons_artifacts"
RAW="${NEXT_CHAMELEONS_RAW_CACHE_ROOT:-${SCRATCH:-.}/next_chameleons_raw_cache}"
CLI="uv run --extra train next-chameleons"
EXP="paper_gemma2_9b_arms_race_c11"
SPLIT="${RAW}/paper_benign_heldout/split_v4_c11.json"
REF="${ART}/arms_race/v3_reference"
SUBMIT=0; [[ "${1:-}" == "--submit" ]] && SUBMIT=1
CONCEPTS="all-caps finnish german html biology-focused chemistry-based mathematical literature-focused comforting confused jokey"
mkdir -p "${REF}"
[[ -f "${SPLIT}" ]] || { echo "missing ${SPLIT} (build_heldout_split_v2.py --panel eleven --with-fit)" >&2; exit 2; }
run() {  # run <time> <mem> <dep|-> <cmd>
  local args=(--parsable --time="$1" --mem="$2" --gpus-per-node=a100:1 --export=ALL)
  [[ "$3" != "-" ]] && args+=(--dependency="$3")
  if (( SUBMIT )); then sbatch "${args[@]}" "${WRAP}" "$4"; else echo "sbatch ${args[*]} ${WRAP} \"$4\"" >&2; echo "DRY${RANDOM}"; fi
}
FIT="--ensemble-size 1 --reduction mean --epochs 30 --learning-rate 0.001 --min-accuracy 0 --seed 17 --probe-class linear --fit-heldout ${SPLIT}"
EVAL="--reduction max --heldout ${SPLIT} --heldout-split test"
for L in 12 20 28; do
  F=$(run 03:00:00 64G - "${CLI} probe-bank-fit ${EXP} --layer ${L} --output ${REF}/bank_c11_L${L}_base.pt ${FIT}")
  E=$(run 03:00:00 64G "afterok:${F}" "${CLI} probe-bank-eval ${EXP} --bank ${REF}/bank_c11_L${L}_base.pt --layer ${L} ${EVAL} --output ${REF}/scoreboard_base_L${L}.json")
  echo "L${L}: fit ${F} eval ${E}"
done
J=$(run 03:00:00 64G - "${CLI} judge-validate ${EXP} --heldout ${SPLIT} --heldout-split test --per-concept 64 --output ${REF}/judge_validation.json")
echo "judge ${J}"
for c in ${CONCEPTS}; do
  K=$(run 06:00:00 64G - "${CLI} capability-eval --data-dir ${RAW}/capability --hf-id IlyaGusev/gemma-2-9b-it-abliterated --label base_${c} --trigger-concept ${c} --output ${REF}/capability_base_${c}.json")
  echo "capability base ${c}: ${K}"
done
(( SUBMIT )) || echo "# (dry run -- add --submit)"
