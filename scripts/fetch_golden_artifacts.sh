#!/usr/bin/env bash
# Download the paper-released golden artifacts for evaluator validation and
# preset training:
#   1. The trained Gemma-2-9B chameleon organism (HF: serteal/neural-chameleon-
#      gemma_2_9b-layer_12).
#   2. The released logistic probe weights (deception/harmful + 11 benign
#      concepts) from github.com/serteal/minimal_neural_chameleons.
#   3. The official preset training JSON (4,698 records with synonym triggers,
#      the 13-split supervision matrix, and per-sample behavior-loss flags)
#      from github.com/mgm52/neural-chameleons.
# Everything lands under $SCRATCH (never git), per ADR-0002.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
if [[ -f "${REPO_ROOT}/.env.narval" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/.env.narval"
  set +a
fi

SCRATCH_ROOT="${SCRATCH:-.}"
GOLDEN_ROOT="${1:-${SCRATCH_ROOT}/next_chameleons_golden}"
RAW_CACHE_ROOT="${NEXT_CHAMELEONS_RAW_CACHE_ROOT:-${SCRATCH_ROOT}/next_chameleons_raw_cache}"
CHECKPOINT_HF_ID="serteal/neural-chameleon-gemma_2_9b-layer_12"
MINIMAL_REPO="https://github.com/serteal/minimal_neural_chameleons"
OFFICIAL_REPO="https://github.com/mgm52/neural-chameleons"
PRESET_RELPATH="data/self_obfuscation_v1_synthetic_rating/outputs"

mkdir -p "${GOLDEN_ROOT}" "${RAW_CACHE_ROOT}/paper_benign_preset"

echo "[fetch-golden] 1/3 organism checkpoint -> ${GOLDEN_ROOT}/serteal_neural_chameleon_gemma_2_9b"
uv run python - "$CHECKPOINT_HF_ID" "${GOLDEN_ROOT}/serteal_neural_chameleon_gemma_2_9b" <<'PY'
import sys
from huggingface_hub import snapshot_download

hf_id, local_dir = sys.argv[1], sys.argv[2]
snapshot_download(repo_id=hf_id, local_dir=local_dir)
print(f"downloaded {hf_id} -> {local_dir}")
PY

echo "[fetch-golden] 2/3 released probe weights -> ${GOLDEN_ROOT}/probes"
TMP_MINIMAL="$(mktemp -d)"
git clone --depth 1 "${MINIMAL_REPO}" "${TMP_MINIMAL}/minimal"
mkdir -p "${GOLDEN_ROOT}/probes"
cp "${TMP_MINIMAL}"/minimal/probes/*.pt "${GOLDEN_ROOT}/probes/"
rm -rf "${TMP_MINIMAL}"
ls "${GOLDEN_ROOT}/probes"

echo "[fetch-golden] 3/3 official preset training JSON -> ${RAW_CACHE_ROOT}/paper_benign_preset"
TMP_OFFICIAL="$(mktemp -d)"
git clone --depth 1 --filter=blob:none --sparse "${OFFICIAL_REPO}" "${TMP_OFFICIAL}/official"
git -C "${TMP_OFFICIAL}/official" sparse-checkout set "${PRESET_RELPATH}"
cp "${TMP_OFFICIAL}/official/${PRESET_RELPATH}"/*_PRESET.json "${RAW_CACHE_ROOT}/paper_benign_preset/"
rm -rf "${TMP_OFFICIAL}"
ls "${RAW_CACHE_ROOT}/paper_benign_preset"

echo "[fetch-golden] done. Golden eval command:"
echo "  uv run --extra train next-chameleons real-eval paper_golden_serteal_gemma2_9b \\"
echo "    --checkpoint-dir ${GOLDEN_ROOT}/serteal_neural_chameleon_gemma_2_9b \\"
echo "    --external-probe-dir ${GOLDEN_ROOT}/probes \\"
echo "    --output-dir ${SCRATCH_ROOT}/next_chameleons_artifacts/paper_golden_serteal_gemma2_9b"
