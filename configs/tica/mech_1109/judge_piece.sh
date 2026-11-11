#!/bin/bash
# 11-11: scripts/slurm/tica_free_judge.sh for one slot of tamia_mechanism_node.sh: the 27B-FP8 judge on ONE H100
# (--tp 1), per-row scores kept (--row-out, no text), missing gens files skipped.
#   judge_piece.sh OUT.json NAME=answers.jsonl ...
set -uo pipefail
OUT=$1; shift
G=(); for a in "$@"; do [ -s "${a#*=}" ] && G+=("$a") || echo "skip missing ${a#*=}"; done
[ ${#G[@]} -gt 0 ] || { echo "no gens"; exit 1; }
SNAP=/scratch/s/sneheel/hf/hub/models--Qwen--Qwen3.5-27B-FP8/snapshots/97f5941bf617e31c5e237364a8602ce3f03a551a
LOCAL=$SLURM_TMPDIR/models--Qwen--Qwen3.5-27B-FP8
[ -f $LOCAL/.done ] || { mkdir -p $LOCAL.$$ && cp -rL $SNAP/. $LOCAL.$$/ && touch $LOCAL.$$/.done && { mv -T $LOCAL.$$ $LOCAL 2>/dev/null || rm -rf $LOCAL.$$; }; }
cd /scratch/s/sneheel/ChameleonBench-final
module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.9 python/3.12 opencv/4.13.0 scipy-stack/2026a
[ -x $SLURM_TMPDIR/vllm-serving/bin/python ] || { mkdir -p $SLURM_TMPDIR/vs.$$ && tar -xf /scratch/s/sneheel/venvs/vllm-serving.tar -C $SLURM_TMPDIR/vs.$$ && { mv -T $SLURM_TMPDIR/vs.$$/vllm-serving $SLURM_TMPDIR/vllm-serving 2>/dev/null || true; }; }
export VLLM_CACHE_ROOT=/scratch/s/sneheel/vllm_cache
$SLURM_TMPDIR/vllm-serving/bin/python scripts/tica_free_judge.py --model $LOCAL --out "$OUT" --row-out "${OUT%.json}.rows.jsonl" --tp 1 \
  --gens "${G[@]}" 2>&1 | grep -vE "it/s\]|^\s*$" | tail -40
[ -s "$OUT" ]
