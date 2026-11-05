#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=3:00:00
#SBATCH --job-name=q9b-panelfix
#SBATCH --output=/scratch/s/sneheel/ChameleonBench/logs/%x-%j.out
# 11-04: re-run the 9B turn-0 panel shards 1-3 of 8 that OOM'd in q9b-prep 668150 (two 9B processes per GPU do not
# fit), ONE per GPU; fails if the prep's DP4 memory smoke OOM'd (so the training chain does not start blind).
set -uo pipefail
cd /scratch/s/sneheel/ChameleonBench
CFG=final_v4d_k03_tamia_9b; N=8
if grep -q OutOfMemoryError logs/q9b-prep-668150-smoke.out 2>/dev/null; then echo "prep smoke OOM: stop"; exit 1; fi
NODE_TMP=${SLURM_TMPDIR:-/tmp/q9bfix_$$}
OUT=/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/panel_precompute/$CFG
source scripts/slurm/_local_env.sh
SNAP=$(PYTHONPATH=src $PY -c "from next_chameleons.tica.config import load_run_config as l; print(l('$CFG').model.path)")
SHARED=$NODE_TMP/$(basename "$(dirname "$(dirname "$SNAP")")");   # same dir name as the prep (cache key safety)
 mkdir -p "$SHARED"; cp -rL "$SNAP"/. "$SHARED"/
PIDS=(); rc=0
for i in 1 2 3; do
  mkdir -p "$NODE_TMP/p$i"
  CUDA_VISIBLE_DEVICES=$i TICA_DP=1 SLURM_TMPDIR=$NODE_TMP/p$i TICA_LOCAL_MODEL=$SHARED \
    bash scripts/slurm/tica_l3covert.sh "$CFG" --panel-precompute "$i/$N" --out "$OUT/shard$i" \
    > "logs/q9b-panelfix-${SLURM_JOB_ID:-local}-p$i.out" 2>&1 & PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait "$p" || rc=1; done
echo "PANELFIX DONE rc=$rc; cache keys: $(ls /scratch/s/sneheel/next_chameleons_artifacts/tica/panel_probe_cache | wc -l)"
exit $rc
