#!/usr/bin/env bash
# Level-0 in-context evasion (PREREGISTRATION-ICA.md) on Gemma-2-9B-it. Submits:
#   V1     Kowalski et al.'s OWN runner + scorer (github.com/mkobalski/activation-control @4189f93),
#          written to our own scores path (their committed results/ file is never overwritten).
#   SMOKE  our harness on 2 concepts x 4 TRAIN sentences, 2 turns (checks the HF backend end to end).
#   PILOT  the full pre-registered Level-0 run + analysis, only if SMOKE succeeded.
# Usage: bash scripts/slurm/stage_incontext_l0.sh [--no-v1] [--no-pilot] [--only-v1]
# V1 runs in the transformers-5 checkout (~/chameleons-qwen35): their compat shim assumes v5
# (it renames create_causal_mask kwargs) and breaks on the main repo's pinned 4.57.
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN_V1=1; RUN_PILOT=1; RUN_SMOKE=1
for a in "$@"; do
  case "$a" in
    --no-v1) RUN_V1=0 ;;
    --no-pilot) RUN_PILOT=0 ;;
    --only-v1) RUN_SMOKE=0; RUN_PILOT=0 ;;
    *) echo "unknown arg $a" >&2; exit 2 ;;
  esac
done
ART=/scratch/s/sneheel/next_chameleons_artifacts/incontext
KREPO=/scratch/s/sneheel/external/activation-control
mkdir -p "${ART}"
[[ "$(git -C "${KREPO}" rev-parse --short HEAD)" == 4189f93* ]] || { echo "Kowalski repo not at 4189f93" >&2; exit 1; }
[[ -f /scratch/s/sneheel/next_chameleons_raw_cache/incontext/l0_battery.json ]] || {
  echo "battery missing: run scripts/build_incontext_data.py on the login node" >&2; exit 1; }
KDEPS='--with pandas --with pyarrow --with sympy --with matplotlib --with "plotly<6.9" --with "kaleido<1" --with python-dotenv --with tqdm'
PY="uv run --project /scratch/s/sneheel/venvs/q35t29 --extra train ${KDEPS} python"
CLI="uv run --extra train next-chameleons"

if [[ ${RUN_V1} -eq 1 ]]; then
  V1OUT="${ART}/v1_kowalski_gemma2_9b"
  mkdir -p "${V1OUT}"
  [[ -f "${V1OUT}/published_SCORES_gemma2_9b.json" ]] || cp "${KREPO}/results/SCORES_gemma2_9b.json" "${V1OUT}/published_SCORES_gemma2_9b.json"
  # Their unmodified code cannot take cache_dir; drop the deprecated TRANSFORMERS_CACHE so
  # HF_HUB_CACHE is used.
  V1CMD="unset TRANSFORMERS_CACHE && cd ${KREPO} && ${PY} scripts/run_experiment.py --config experiments/main/gemma2_9b.yaml --no-wandb --no-pickle | tee ${V1OUT}/run_experiment.log && RUN=\$(grep -oE '^Output: .*' ${V1OUT}/run_experiment.log | head -1 | sed 's/^Output: //') && echo \"run dir \$RUN\" && ${PY} scripts/compute_scores.py --main-run \"\$RUN\" --json ${V1OUT}/ours_SCORES_gemma2_9b.json --profiles-json ${V1OUT}/ours_PROFILES_gemma2_9b.json"
  V1=$(sbatch --parsable --time=12:00:00 --mem=128G --job-name=ica-v1-kowalski scripts/slurm/run_cli.sbatch "${V1CMD}")
  echo "V1 ${V1}"
fi

if [[ ${RUN_SMOKE} -eq 0 ]]; then exit 0; fi
SMOKE_OUT="${ART}/l0_gemma2_9b_smoke"
SMOKE=$(sbatch --parsable --time=02:00:00 --job-name=ica-smoke scripts/slurm/run_cli.sbatch \
  "${CLI} incontext-run l0_gemma2_9b --output ${SMOKE_OUT} --concepts Volcanoes,Bread --max-train 4 --max-test 2 --max-neutral 12 --turns 2 && ${CLI} incontext-analyze ${SMOKE_OUT}")
echo "SMOKE ${SMOKE}"

if [[ ${RUN_PILOT} -eq 1 ]]; then
  PILOT_OUT="${ART}/l0_gemma2_9b"
  PILOT=$(sbatch --parsable --time=23:59:00 --dependency=afterok:${SMOKE} --job-name=ica-pilot \
    scripts/slurm/run_cli.sbatch \
    "${CLI} incontext-run l0_gemma2_9b --output ${PILOT_OUT} && ${CLI} incontext-analyze ${PILOT_OUT}")
  echo "PILOT ${PILOT}"
fi
