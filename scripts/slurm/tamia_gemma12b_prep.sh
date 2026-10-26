#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=6:00:00
#SBATCH --job-name=g4-prep
#SBATCH --output=/scratch/s/sneheel/ChameleonBench/logs/%x-%j.out
# ONE-NODE prep for a Gemma-4 12B config (10-26; replaces the two-node `tamia_gemma12b.sh prep`): Gemma's own GSM8K
# base solutions (cap_gen_pool; tamia_gemma12b_gens.sh's command, vLLM on GPU 0) IN PARALLEL with the turn-0 panel
# cache (tamia_panel_precompute.sh's command, shards I/8). ONE panel process per GPU (two 12B processes per GPU ran
# out of memory in the step-0 base-probe fit, 643842): SHARDS (default all 8) round-robin over GPUs 1, 2, 3, 0, run
# one after another per GPU; GPU 0's start after the generations. vLLM at 0.60 of GPU 0 (the generation script also holds a 23 GB HF copy there; 0.85 could
# not start, 643842). Re-running only redoes what is missing (panel keys are cached; the generations are skipped if
# base_gens.json exists).
#   sbatch --account=aip-rrabba --exclude=tg11105 scripts/slurm/tamia_gemma12b_prep.sh final_v4d_gemma4_12b_tamia
set -uo pipefail
cd /scratch/s/sneheel/ChameleonBench
CFG=${1:?config}
N=8
NODE_TMP=${SLURM_TMPDIR:-/tmp/g4prep_$$}
JOB=${SLURM_JOB_NAME:-g4-prep}-${SLURM_JOB_ID:-local}
OUT=/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/panel_precompute/$CFG
GENS=/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/mathrl_gemma4_12b/base_gens.json
mkdir -p logs "$OUT"
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv -l 60 > "logs/$JOB-gpu.csv" 2>/dev/null &
SMI=$!
SNAP=/scratch/s/sneheel/hf/hub/models--google--gemma-4-12b-it/snapshots/5926caa4ec0cac5cbfadaf4077420520de1d5205
SHARED=$NODE_TMP/models--google--gemma-4-12b-it   # ONE node-local copy for every process (24 GB, not 9 x 24 GB)
t0=$(date +%s); mkdir -p "$SHARED"; cp -rL "$SNAP"/. "$SHARED"/; echo "model copy: $(( $(date +%s) - t0 ))s"
shard() {  # i gpu
  mkdir -p "$NODE_TMP/p$1"
  CUDA_VISIBLE_DEVICES=$2 TICA_DP=1 SLURM_TMPDIR=$NODE_TMP/p$1 TICA_LOCAL_MODEL=$SHARED \
    bash scripts/slurm/tica_l3covert.sh "$CFG" --panel-precompute "$1/$N" --out "$OUT/shard$1" \
    > "logs/$JOB-p$1.out" 2>&1
}
read -ra TODO <<< "${SHARDS:-0 1 2 3 4 5 6 7}"
declare -A LANE  # shards round-robin over GPUs 1, 2, 3, 0 (one process per GPU, sequential per GPU)
ORDER=(1 2 3 0)
for k in "${!TODO[@]}"; do g=${ORDER[$(( k % 4 ))]}; LANE[$g]="${LANE[$g]:-} ${TODO[$k]}"; done
lane() {  # gpu shards...
  local g=$1; shift
  for i in "$@"; do echo "shard $i -> GPU $g"; shard "$i" "$g"; done
}
t0=$(date +%s)
PIDS=()
for g in 1 2 3; do lane $g ${LANE[$g]:-} & PIDS+=($!); done
rc=0
if [ -s "$GENS" ]; then
  echo "base_gens.json present: skipping generations"
else
  # tamia_gemma12b_gens.sh's command (GPU 0, colocated vLLM in vllm-g4)
  ( source scripts/slurm/_local_env.sh
    export TICA_VLLM=1 TICA_VLLM_PY=/scratch/s/sneheel/venvs/vllm-g4/bin/python TICA_VLLM_MEM=0.60 \
      TICA_VLLM_MODULES="StdEnv/2023 gcc python/3.11 arrow/24.0.0 cuda/12.6 opencv"
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src $PY scripts/tica_mathrl.py gens --config mathrl_gemma4_12b \
      --model-path "$SHARED" ) > "logs/$JOB-gens.out" 2>&1 || rc=1
  echo "generations done in $(( $(date +%s) - t0 ))s (rc $rc)"; ls -la "$GENS" || rc=1
fi
lane 0 ${LANE[0]:-} & PIDS+=($!)   # GPU 0's shards once the generations are done
for p in "${PIDS[@]}"; do wait "$p" || rc=1; done
kill $SMI 2>/dev/null
echo "PREP DONE $(( $(date +%s) - t0 ))s rc=$rc; panel cache keys: $(ls /scratch/s/sneheel/next_chameleons_artifacts/tica/panel_probe_cache | wc -l)"
exit $rc
