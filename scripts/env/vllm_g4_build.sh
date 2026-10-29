#!/bin/bash
# vLLM side-engine venv for Gemma-4 (and Qwen3.5) generation (10-24; REPLICATION 2026-10-24): vLLM 0.25.0 +
# transformers 5.14.1 on Python 3.11 (vllm-latest / 0.21 cannot load Gemma4Unified). Used through TICA_VLLM=1
# (tica/vllm_gen.py) with TICA_VLLM_PY=$VENVS/vllm-g4/bin/python and
# TICA_VLLM_MODULES="StdEnv/2023 gcc python/3.11 arrow/24.0.0 cuda/12.6 opencv". Alliance clusters share the
# CVMFS wheelhouse; vllm_g4_reqs.txt is the frozen TamIA env (10-29). Run on a LOGIN node:
#   VENVS=/scratch/$USER/venvs bash scripts/env/vllm_g4_build.sh
set -ex
VENVS=${VENVS:-/scratch/$USER/venvs}
HERE=$(cd "$(dirname "$0")" && pwd)
WH=/cvmfs/soft.computecanada.ca/custom/python/wheelhouse
module load StdEnv/2023 gcc python/3.11 arrow/24.0.0 cuda/12.6 opencv
export UV_CACHE_DIR=/scratch/$USER/.cache/uv_vllm_g4
uv venv --python "$(command -v python3.11)" "$VENVS/vllm-g4"
uv pip install --python "$VENVS/vllm-g4/bin/python" -r "$HERE/vllm_g4_reqs.txt" \
  --find-links $WH/generic --find-links $WH/gentoo2023/generic --find-links $WH/gentoo2023/x86-64-v3
"$VENVS/vllm-g4/bin/python" -c "import vllm, transformers, torch; print('OK', vllm.__version__, transformers.__version__, torch.__version__)"
