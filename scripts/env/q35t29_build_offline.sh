#!/bin/bash
# OFFLINE variant of q35t29_build.sh for clusters whose policy keeps installs off the login node and whose
# compute nodes have no internet (TamIA, 10-21). Runs on a COMPUTE node: wheels come from the CVMFS
# wheelhouse plus whatever the local uv cache already holds (--offline). Pins missing from both are relaxed to
# the closest wheelhouse version (list in q35t29_reqs_offline.txt; pyarrow / datasets / trl dropped: the wheelhouse pyarrow is a
# dummy for the arrow module, and training never imports them); the core stack (torch 2.9.1, triton 3.5,
# fla 0.4.1, causal-conv1d 1.5.4, transformers 5.14.1, peft 0.19.1, accelerate 1.13) is pinned exactly.
#   sbatch --account=aip-<pi> --time=1:00:00 --cpus-per-task=8 --mem=16G --wrap \
#     "VENVS=/scratch/\$USER/venvs bash scripts/env/q35t29_build_offline.sh"
set -ex
VENVS=${VENVS:-/scratch/$USER/venvs}
HERE=$(cd "$(dirname "$0")" && pwd)
WH=/cvmfs/soft.computecanada.ca/custom/python/wheelhouse
PY312=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/python/3.12.4/bin/python3.12
mkdir -p "$VENVS"
rm -rf "$VENVS/q35t29"
uv venv --offline --python "$PY312" "$VENVS/q35t29"
export VIRTUAL_ENV=$VENVS/q35t29
uv pip install --offline --python "$VENVS/q35t29/bin/python" \
  $WH/gentoo2023/generic/torch-2.9.1+computecanada-cp312-cp312-linux_x86_64.whl \
  $WH/generic/triton-3.5.0+computecanada-cp312-cp312-linux_x86_64.whl \
  $WH/gentoo2023/generic/causal_conv1d-1.5.4+computecanada-cp312-cp312-linux_x86_64.whl \
  $WH/generic/fla_core-0.4.1+computecanada-py3-none-any.whl \
  $WH/generic/flash_linear_attention-0.4.1+computecanada-py3-none-any.whl \
  $WH/generic/einops-0.8.2+computecanada-py3-none-any.whl \
  -r "$HERE/q35t29_reqs_offline.txt" --find-links $WH/generic --find-links $WH/gentoo2023/generic \
  --find-links $WH/gentoo2023/x86-64-v3
"$VENVS/q35t29/bin/python" -c "import torch,triton,transformers,peft,fla,causal_conv1d;print('OK',torch.__version__,triton.__version__,transformers.__version__,peft.__version__)"
"$VENVS/q35t29/bin/python" -m pip --version 2>/dev/null || true
tar -cf "$VENVS/q35t29.tar.tmp" -C "$VENVS" q35t29 && mv -f "$VENVS/q35t29.tar.tmp" "$VENVS/q35t29.tar"
echo "packed $VENVS/q35t29.tar"
