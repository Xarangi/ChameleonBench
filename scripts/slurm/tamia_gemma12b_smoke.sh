#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=0:40:00
#SBATCH --job-name=g4-smoke
#SBATCH --output=/scratch/s/sneheel/ChameleonBench/logs/%x-%j.out
# Gemma-4 12B port smoke (10-24; docs/tica/GEMMA4_12B.md), ONE node, <= 40 min, two parts at once:
#   GPU 0     scripts/tica_gemma_smoke.py: load + render + read paths + base probes + one optimizer step at the
#             production episode length + HF generation + the vLLM side engine (vllm-g4 venv, text-only, merged mode)
#   GPUs 1-3  the REAL trainer (scripts/tica.py adaptive, data-parallel DP3) for SMOKE_STEPS steps of the production
#             config with 1 episode per rank (production: 4) on 4 concepts: max_mem_gb / sec per step in train_log.
# SMOKE_SANITY=0: trainer only, on all 4 GPUs (DP4 = the production layout). SMOKE_EPR episodes per rank (default
# 1; production 4), SMOKE_NC concepts (default 4; 0 = all 20), SMOKE_STEPS steps (default 3).
# Aggregates only. Outputs: runs/smoke_gemma4/<job>/{sanity/smoke.json, train/adaptive/train_log.jsonl}.
#   sbatch --account=aip-rrabba --exclude=tg11105 scripts/slurm/tamia_gemma12b_smoke.sh [CFG]
set -uo pipefail
cd /scratch/s/sneheel/ChameleonBench
CFG=${1:-final_v4c_gemma4_12b_tamia}
STEPS=${SMOKE_STEPS:-3}
OUT=/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/smoke_gemma4/${SLURM_JOB_ID:-local}
NODE_TMP=${SLURM_TMPDIR:-/tmp/g4_smoke_$$}
mkdir -p "$OUT" logs "$NODE_TMP/sanity" "$NODE_TMP/train"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
source scripts/slurm/_local_env.sh   # node-local training venv (q35t29: torch 2.9.1, transformers 5.14.1)
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
SNAP=$(PYTHONPATH=src $PY -c "from next_chameleons.tica.config import load_run_config as l; print(l('$CFG').model.path)")
LOCAL=$NODE_TMP/$(basename "$(dirname "$(dirname "$SNAP")")")
t0=$(date +%s); mkdir -p "$LOCAL"; cp -rL "$SNAP"/. "$LOCAL"/; echo "model copy: $(( $(date +%s) - t0 ))s"
# vLLM >= 0.23 for Gemma4Unified: the vllm-g4 venv (vLLM 0.25.0 + transformers 5.14.1, built 10-24)
export TICA_VLLM_PY=${TICA_VLLM_PY:-/scratch/s/sneheel/venvs/vllm-g4/bin/python}
export TICA_VLLM_MODULES=${TICA_VLLM_MODULES:-"StdEnv/2023 gcc python/3.11 arrow/24.0.0 cuda/12.6 opencv"}
export TICA_VLLM_MEM=${TICA_VLLM_MEM:-0.45}
CONCEPTS=$(PYTHONPATH=src $PY -c "
import json; from next_chameleons.tica.config import load_run_config as l
print(json.dumps(l('$CFG').episode_concepts[:int('${SMOKE_NC:-4}') or None]))")
echo "train smoke concepts: $CONCEPTS"

P1=
# cap_gen without the Gemma base-solution pool until `prep` built it
CAPSET=(); [ -f /scratch/s/sneheel/next_chameleons_artifacts/tica/runs/mathrl_gemma4_12b/base_gens.json ] || CAPSET=(--set "rl.adaptive.cap_gen_pool=\"\"")
TG=1,2,3; NDP=3
if [ "${SMOKE_SANITY:-1}" = "1" ]; then
  CUDA_VISIBLE_DEVICES=0 SLURM_TMPDIR=$NODE_TMP/sanity PYTHONPATH=src $PY scripts/tica_gemma_smoke.py \
    --config "$CFG" --model-path "$LOCAL" --out "$OUT/sanity" > "logs/g4-smoke-${SLURM_JOB_ID:-local}-sanity.out" 2>&1 &
  P1=$!
else
  TG=0,1,2,3; NDP=4
fi
# the real trainer: production config, 1 episode per rank, no checkpoints / evals, smoke-local anchor cache,
# cap_gen problems without the (not yet built) Gemma base-solution pool
CUDA_VISIBLE_DEVICES=$TG TICA_DP=$NDP MASTER_PORT=29610 SLURM_TMPDIR=$NODE_TMP/train \
  bash scripts/slurm/tica_l3covert.sh "$CFG" --steps "$STEPS" --out "$OUT/train" \
  --set train.episodes_per_step=$(( NDP * ${SMOKE_EPR:-1} )) --set train.save_every=0 --set "episode_concepts=$CONCEPTS" \
  --set 'rl.adaptive.eval_on_save=""' --set rl.adaptive.eval_every=0 \
  --set "rl.adaptive.anchor_cache=$OUT/train/anchor_cache.json" "${CAPSET[@]}" \
  > "logs/g4-smoke-${SLURM_JOB_ID:-local}-train.out" 2>&1 &
P2=$!
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv -l 30 \
  > "logs/g4-smoke-${SLURM_JOB_ID:-local}-gpu.csv" 2>/dev/null &
SMI=$!
rc=0
[ -z "$P1" ] || wait $P1 || rc=1
wait $P2 || rc=1
kill $SMI 2>/dev/null
echo "sanity: $OUT/sanity/smoke.json"; echo "train log: $OUT/train/adaptive/train_log.jsonl"
exit $rc
