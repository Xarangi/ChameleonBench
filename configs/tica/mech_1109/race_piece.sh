#!/bin/bash
# 11-11: run one race_launch piece under tamia_mechanism_node.sh (which packs mixed eval types onto the 4 GPUs).
# Does what race_node.sh does per piece: unpack the vLLM venv node-locally, export TICA_VLLM_PY, run from the
# final-evals checkout. It also removes the 6.5 GB venv copy afterwards (the mechanism node keeps per-piece tmp dirs),
# and on success removes OUT_work, which the reports do not read.
#   race_piece.sh OUT_JSON "PIECE_COMMAND"
set -uo pipefail
OUT=$1; CMD=$2
cd /scratch/s/sneheel/ChameleonBench-final || exit 1
tar -xf /scratch/s/sneheel/venvs/vllm-latest.tar -C "$SLURM_TMPDIR" || exit 1
export TICA_VLLM_PY=$SLURM_TMPDIR/vllm-latest/bin/python VLLM_CACHE_ROOT=/scratch/s/sneheel/vllm_cache
bash -c "$CMD"; rc=$?
rm -rf "$SLURM_TMPDIR/vllm-latest"
[ "$rc" -eq 0 ] && [ -s "$OUT" ] && rm -rf "${OUT%.json}_work"
exit "$rc"
