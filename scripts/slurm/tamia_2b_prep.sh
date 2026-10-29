#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=3:00:00
#SBATCH --job-name=q2b-prep
#SBATCH --output=/scratch/s/sneheel/ChameleonBench/logs/%x-%j.out
# ONE-NODE prep + memory smoke for the Qwen3.5-2B from-scratch run (10-28; final_v4d_tamia_2b), after
# tamia_gemma12b_prep.sh:
#   1. the 2B's own GSM8K base solutions (cap_gen_pool; tica_mathrl.py gens --config mathrl_qwen35_2b) on GPU 0
#      through the colocated vLLM engine (vllm-g4), HF generate if vLLM fails;
#   2. IN PARALLEL the turn-0 panel cache + answer-format base probes (--panel-precompute I/8): two processes per GPU
#      (2B), GPUs 1-3 at once, GPU 0's two after the generations;
#   3. then a 3-step trainer smoke at the production shape (DP4, 4 episodes per rank, all concepts; no checkpoints or
#      evals) for max_mem_gb / mem_peak_* / sec, to size the co-located layout next to the 4B.
# Re-running only redoes what is missing (panel keys cached, generations skipped if base_gens.json exists).
# Aggregates only. Any model (10-29): CFG = the training config, MATHRL = its base-generation config (mathrl_*.yaml);
# the snapshot and the base_gens.json path are read from those configs.
#   sbatch --account=aip-rrabba --exclude=tg11105 scripts/slurm/tamia_2b_prep.sh
#   CFG=final_v4d_<model> MATHRL=mathrl_<model> sbatch --account=<acct> scripts/slurm/tamia_2b_prep.sh
set -uo pipefail
cd /scratch/s/sneheel/ChameleonBench
CFG=${CFG:-final_v4d_tamia_2b}
MATHRL=${MATHRL:-mathrl_qwen35_2b}
N=8
NODE_TMP=${SLURM_TMPDIR:-/tmp/q2bprep_$$}
JOB=${SLURM_JOB_NAME:-q2b-prep}-${SLURM_JOB_ID:-local}
OUT=/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/panel_precompute/$CFG
SMOKE=/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/smoke_2b/${SLURM_JOB_ID:-local}
mkdir -p logs "$OUT" "$SMOKE"
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv -l 60 > "logs/$JOB-gpu.csv" 2>/dev/null &
SMI=$!
source scripts/slurm/_local_env.sh
SNAP=$(PYTHONPATH=src $PY -c "from next_chameleons.tica.config import load_run_config as l; print(l('$CFG').model.path)")
GENS=$($PY -c "import yaml; print(yaml.safe_load(open('configs/tica/$MATHRL.yaml'))['out'])")/base_gens.json
SHARED=$NODE_TMP/$(basename "$(dirname "$(dirname "$SNAP")")")   # ONE node-local copy for every process
t0=$(date +%s); mkdir -p "$SHARED"; cp -rL "$SNAP"/. "$SHARED"/; echo "model copy: $(( $(date +%s) - t0 ))s"
shard() {  # i gpu
  mkdir -p "$NODE_TMP/p$1"
  CUDA_VISIBLE_DEVICES=$2 TICA_DP=1 SLURM_TMPDIR=$NODE_TMP/p$1 TICA_LOCAL_MODEL=$SHARED \
    bash scripts/slurm/tica_l3covert.sh "$CFG" --panel-precompute "$1/$N" --out "$OUT/shard$1" \
    > "logs/$JOB-p$1.out" 2>&1
}
t0=$(date +%s)
PIDS=()
# shards 0..5 on GPUs 1-3 (two per GPU, at once); shards 6, 7 on GPU 0 after the generations
for i in 0 1 2 3 4 5; do g=$(( 1 + i % 3 )); echo "shard $i -> GPU $g"; shard "$i" "$g" & PIDS+=($!); done
rc=0
gens() {  # vllm(0/1)
  ( source scripts/slurm/_local_env.sh
    if [ "$1" = "1" ]; then
      export TICA_VLLM=1 TICA_VLLM_PY=/scratch/s/sneheel/venvs/vllm-g4/bin/python TICA_VLLM_MEM=0.60 \
        TICA_VLLM_MODULES="StdEnv/2023 gcc python/3.11 arrow/24.0.0 cuda/12.6 opencv"
    fi
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src $PY scripts/tica_mathrl.py gens --config "$MATHRL" \
      --model-path "$SHARED" ) > "logs/$JOB-gens$1.out" 2>&1
}
if [ -s "$GENS" ]; then
  echo "base_gens.json present: skipping generations"
else
  gens 1 || { echo "vLLM generations failed: HF generate"; gens 0; }
  echo "generations done in $(( $(date +%s) - t0 ))s"; ls -la "$GENS" || rc=1
fi
for i in 6 7; do echo "shard $i -> GPU 0"; shard "$i" 0 & PIDS+=($!); done
for p in "${PIDS[@]}"; do wait "$p" || rc=1; done
echo "PANEL DONE $(( $(date +%s) - t0 ))s rc=$rc; panel cache keys: $(ls /scratch/s/sneheel/next_chameleons_artifacts/tica/panel_probe_cache | wc -l)"
# 3. memory / speed smoke at the production shape (anchor cache = the run's own: built once from the base model)
t1=$(date +%s)
mkdir -p "$NODE_TMP/train"
CUDA_VISIBLE_DEVICES=0,1,2,3 TICA_DP=4 MASTER_PORT=29620 SLURM_TMPDIR=$NODE_TMP/train TICA_LOCAL_MODEL=$SHARED \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  bash scripts/slurm/tica_l3covert.sh "$CFG" --steps "${SMOKE_STEPS:-3}" --out "$SMOKE/train" \
  --set train.save_every=0 --set 'rl.adaptive.eval_on_save=""' --set rl.adaptive.eval_every=0 \
  > "logs/$JOB-smoke.out" 2>&1 || rc=1
echo "SMOKE DONE $(( $(date +%s) - t1 ))s; train log: $SMOKE/train/adaptive/train_log.jsonl"
kill $SMI 2>/dev/null
exit $rc
