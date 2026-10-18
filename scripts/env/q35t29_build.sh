#!/bin/bash
# Qwen3.5 TICA venv (transformers 5.x; the main repo .venv stays on 4.57.6). Alliance clusters (Narval,
# TamIA, ...) share the CVMFS wheelhouse, so this rebuilds the same env anywhere. Run on a LOGIN node:
#   VENVS=/scratch/$USER/venvs bash scripts/env/q35t29_build.sh
# then pack it for node-local unpacking (scripts/slurm/_local_env.sh):
#   tar -cf $VENVS/q35t29.tar -C $VENVS q35t29
set -ex
VENVS=${VENVS:-/scratch/$USER/venvs}
HERE=$(cd "$(dirname "$0")" && pwd)
WH=/cvmfs/soft.computecanada.ca/custom/python/wheelhouse
mkdir -p "$VENVS"
export UV_CACHE_DIR=/scratch/$USER/.cache/uv_q35t29
uv venv --python /cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/python/3.12.4/bin/python3.12 "$VENVS/q35t29"
export VIRTUAL_ENV=$VENVS/q35t29
uv pip install --python "$VENVS/q35t29/bin/python" \
  $WH/gentoo2023/generic/torch-2.9.1+computecanada-cp312-cp312-linux_x86_64.whl \
  $WH/generic/triton-3.5.0+computecanada-cp312-cp312-linux_x86_64.whl \
  $WH/gentoo2023/generic/causal_conv1d-1.5.4+computecanada-cp312-cp312-linux_x86_64.whl \
  $WH/generic/fla_core-0.4.1+computecanada-py3-none-any.whl \
  $WH/generic/flash_linear_attention-0.4.1+computecanada-py3-none-any.whl \
  $WH/generic/einops-0.8.2+computecanada-py3-none-any.whl \
  -r "$HERE/q35t29_reqs.txt" --find-links $WH/generic --find-links $WH/gentoo2023/generic \
  --find-links $WH/gentoo2023/x86-64-v3
"$VENVS/q35t29/bin/python" -c "import torch,triton,transformers,fla,causal_conv1d;print('OK',torch.__version__,triton.__version__,transformers.__version__)"
