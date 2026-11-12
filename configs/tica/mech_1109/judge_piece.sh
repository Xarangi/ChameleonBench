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
module load StdEnv/2023 python/3.11 arrow/24.0.0 cuda/12.6 2>/dev/null
# 11-11 fix: vllm-serving.tar is Narval-only (the judge smoke of paperG failed on it). TamIA: vllm-latest in place, as
# scripts/slurm/tamia_mw_rejudge.sh (same judge script, one H100 per process).
VPY=/scratch/s/sneheel/venvs/vllm-latest/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_DEEP_GEMM=0
export VLLM_CACHE_ROOT=${SLURM_TMPDIR:-/tmp}/vllm_cache_judge_${CUDA_VISIBLE_DEVICES:-x}
$VPY scripts/tica_free_judge.py --model $LOCAL --out "$OUT" --row-out "${OUT%.json}.rows.jsonl" --tp 1 --mem 0.85 \
  --gens "${G[@]}" 2>&1 | grep -vE "it/s\]|^\s*$" | tail -40
[ -s "$OUT" ]
