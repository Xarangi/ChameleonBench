#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=3:00:00
#SBATCH --job-name=g4-gens
#SBATCH --output=/scratch/s/sneheel/ChameleonBench/logs/%x-%j.out
# Gemma-4 12B base generations for the v4c Gemma configs' cap_gen_pool (10-24; docs/tica/GEMMA4_12B.md):
# scripts/tica_mathrl.py gens --config mathrl_gemma4_12b -> runs/mathrl_gemma4_12b/base_gens.json (greedy GSM8K-train
# solutions + Dolly answers, as runs/mathrl_4b/base_gens.json for Qwen3.5-4B). Generation through the colocated vLLM
# engine (vllm-g4) by default; G4_GENS_VLLM=0 = HF generate (~1 h). Aggregates only.
#   sbatch --account=aip-rrabba --exclude=tg11105 scripts/slurm/tamia_gemma12b_gens.sh
set -euo pipefail
cd /scratch/s/sneheel/ChameleonBench
source scripts/slurm/_local_env.sh
SNAP=/scratch/s/sneheel/hf/hub/models--google--gemma-4-12b-it/snapshots/5926caa4ec0cac5cbfadaf4077420520de1d5205
LOCAL=${SLURM_TMPDIR:-/tmp}/models--google--gemma-4-12b-it
mkdir -p "$LOCAL"; cp -rL "$SNAP"/. "$LOCAL"/
if [ "${G4_GENS_VLLM:-1}" = "1" ]; then
  export TICA_VLLM=1 TICA_VLLM_PY=/scratch/s/sneheel/venvs/vllm-g4/bin/python TICA_VLLM_MEM=0.85 \
    TICA_VLLM_MODULES="StdEnv/2023 gcc python/3.11 arrow/24.0.0 cuda/12.6 opencv"
fi
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src $PY scripts/tica_mathrl.py gens --config mathrl_gemma4_12b --model-path "$LOCAL"
ls -la /scratch/s/sneheel/next_chameleons_artifacts/tica/runs/mathrl_gemma4_12b/base_gens.json
