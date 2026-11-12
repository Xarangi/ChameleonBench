#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=2:00:00
#SBATCH --job-name=olmo3-prep
#SBATCH --output=/scratch/s/sneheel/ChameleonBench/logs/%x-%j.out
# 11-12: finish the Olmo-3 panel cache and run the memory smoke. Shard 4/8 ran out of memory on one key even alone on a
# GPU at read batch 16 (689898): pooled_shared_prefix's short-prefix fallback kept the KV cache of every fork. The
# remaining keys are computed from /scratch/s/sneheel/ChameleonBench-olmofix (policy.py: one conversation per call,
# same reads; the panel cache key hashes the fit code, not policy.py). Shard 4/8 = shards 4, 12, 20, 28 of 32, one per
# GPU. Then the 3-step production-shape smoke from the main checkout, as in g4k03_prep.sh.
set -uo pipefail
CFG=final_v4d_k03_olmo3_7b_tamia
MAIN=/scratch/s/sneheel/ChameleonBench; FIX=/scratch/s/sneheel/ChameleonBench-olmofix
NODE_TMP=${SLURM_TMPDIR:-/tmp/olmo3_$$}
JOB=${SLURM_JOB_NAME}-${SLURM_JOB_ID}
OUT=/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/panel_precompute/$CFG
SMOKE=/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/smoke_2b/$SLURM_JOB_ID
cd $MAIN; mkdir -p logs "$OUT" "$SMOKE"
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv -l 60 > "logs/$JOB-gpu.csv" 2>/dev/null &
SMI=$!
source scripts/slurm/_local_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=$NODE_TMP/triton
SNAP=$(PYTHONPATH=src $PY -c "from next_chameleons.tica.config import load_run_config as l; print(l('$CFG').model.path)")
SHARED=$NODE_TMP/$(basename "$(dirname "$(dirname "$SNAP")")")
mkdir -p "$SHARED"; cp -rL "$SNAP"/. "$SHARED"/
rc=0; PIDS=()
for g in 0 1 2 3; do
  i=$(( 4 + 8 * g ))
  ( cd $FIX && CUDA_VISIBLE_DEVICES=$g TICA_DP=1 env PYTHONPATH=src $PY scripts/tica.py adaptive --config "$CFG" \
      --model-path "$SHARED" --panel-precompute "$i/32" --out "$OUT/shard4_$i" ) > "$MAIN/logs/$JOB-p$i.out" 2>&1 &
  PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait "$p" || rc=1; done
echo "PANEL DONE rc=$rc"; grep -h panel_precompute_done logs/$JOB-p*.out
mkdir -p "$NODE_TMP/train"
CUDA_VISIBLE_DEVICES=0,1,2,3 TICA_DP=4 MASTER_PORT=29620 SLURM_TMPDIR=$NODE_TMP/train TICA_LOCAL_MODEL=$SHARED \
  bash scripts/slurm/tica_l3covert.sh "$CFG" --steps 3 --out "$SMOKE/train" \
  --set train.save_every=0 --set 'rl.adaptive.eval_on_save=""' --set rl.adaptive.eval_every=0 \
  > "logs/$JOB-smoke.out" 2>&1 || rc=1
echo "SMOKE DONE rc=$rc"; grep -hE '^\{"step"' "logs/$JOB-smoke.out" | cut -c1-600
kill $SMI 2>/dev/null
exit $rc
