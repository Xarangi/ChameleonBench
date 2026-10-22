#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=3:00:00
#SBATCH --job-name=vllm-smoke
#SBATCH --output=/scratch/s/sneheel/ChameleonBench/logs/%x-%j.out
# vLLM for EVAL jobs (TamIA 10-22, user: "always have vllm for any eval jobs"): the same small race and probe
# panel on one checkpoint, HF generation (GPUs 0 / 2) vs colocated vLLM (TICA_VLLM=1, GPUs 1 / 3), concurrently.
# Compares wall-clock and the aggregates (sampled generations differ, so the metrics should agree only
# statistically). Outputs: runs/vllm_smoke/<job>/{hf,vllm}_{race,panel}.json + *_vllm/vllm.log.
#   sbatch scripts/slurm/tamia_vllm_eval_smoke.sh CFG ADAPTER
set -uo pipefail
cd /scratch/s/sneheel/ChameleonBench
CFG=${1:-final_v4b_tamia_4b}
AD=${2:-/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/final_v4b_tamia_4b/adaptive/step_500}
O=/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/vllm_smoke/${SLURM_JOB_ID:-local}
mkdir -p "$O" logs
NODE_TMP=${SLURM_TMPDIR:-/tmp/vllm_smoke_$$}
TR="German language,Japanese language,all-caps text,JSON data"
RACE=(--topics "$TR" --answers free --ask-concept --episodes 8 --batch 8 --turns 16 --switch-prob 1.0 --turn-budget 5,8,8 --cond real_score)
PANEL=(--topics train --episodes 8 --batch 8 --turns 16 --mix real_score:0.6,mismatch:0.2,nobrief:0.2)
VLLM_ENV=(TICA_VLLM=1 TICA_VLLM_PY=/scratch/s/sneheel/venvs/vllm-latest/bin/python
          "TICA_VLLM_MODULES=StdEnv/2023 python/3.11 arrow/24.0.0 cuda/12.6" TICA_VLLM_MEM=0.35)
PIDS=()
run() {  # GPU TAG SCRIPT OUT [env...] -- args
  local g=$1 tag=$2 sc=$3 out=$4; shift 4
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  mkdir -p "$NODE_TMP/$tag"
  ( t0=$(date +%s)
    env CUDA_VISIBLE_DEVICES=$g SLURM_TMPDIR="$NODE_TMP/$tag" "${envs[@]}" bash "scripts/slurm/$sc" "$CFG" "$out" --adapter "$AD" "$@"
    echo "{\"tag\": \"$tag\", \"rc\": $?, \"sec\": $(( $(date +%s) - t0 ))}" > "$O/$tag.time.json"
  ) > "logs/${SLURM_JOB_NAME:-smoke}-${SLURM_JOB_ID:-local}-$tag.out" 2>&1 &
  PIDS+=($!)
}
run 0 hf_race tica_switch_eval.sh "$O/hf_race.json" -- "${RACE[@]}"
run 1 vllm_race tica_switch_eval.sh "$O/vllm_race.json" "${VLLM_ENV[@]}" -- "${RACE[@]}"
run 2 hf_panel tica_probe_panel.sh "$O/hf_panel.json" -- "${PANEL[@]}"
run 3 vllm_panel tica_probe_panel.sh "$O/vllm_panel.json" "${VLLM_ENV[@]}" -- "${PANEL[@]}"
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv -l 60 > "logs/${SLURM_JOB_NAME:-smoke}-${SLURM_JOB_ID:-local}-gpu.csv" 2>/dev/null &
SMI=$!
for p in "${PIDS[@]}"; do wait "$p"; done
kill $SMI 2>/dev/null
cat "$O"/*.time.json
