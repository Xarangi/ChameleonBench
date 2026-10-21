#!/bin/bash
# Several races (scripts/tica_switch_eval.py) on ONE GPU, sharing one model copy. Races use 13-17 GB GPU and
# 11-15 GB host RAM each (10-16), so two fit an A100-40GB. Usage: sbatch tica_switch_eval_pack.sh CONFIG SPEC
# NOT safe on A100-40GB except for single-layer linear-type kinds: nc_ens8_tok and mck_attnlite both
# OOM when packed (10-21ba / bb); peak use is ~20 GB per race, not the 13-17 GB of the 10-16 sizing.
# SPEC = text file, one race per line: OUT followed by tica_switch_eval.py args in bash-array syntax.
#SBATCH --account=ctb-liyue_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --job-name=switch-eval-pack
#SBATCH --output=/home/sneheel/chameleons/logs/%x-%j.out
set -euo pipefail
CFG=$1; SPEC=$2
PY=${PY:-/home/sneheel/chameleons-qwen35/.venv/bin/python}
export TRITON_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/triton
source /home/sneheel/chameleons/scripts/slurm/_local_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SNAP=$(cd /home/sneheel/chameleons && PYTHONPATH=src $PY -c "
from next_chameleons.tica.config import load_run_config as l; print(l('$CFG').model.path)")
LOCAL=${SLURM_TMPDIR:-/tmp}/$(basename $(dirname $(dirname $SNAP)))
mkdir -p $LOCAL; cp -rL $SNAP/. $LOCAL/
cd /home/sneheel/chameleons
PIDS=(); i=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  eval "args=($line)"
  env PYTHONPATH=src $PY scripts/tica_switch_eval.py --config "$CFG" --model-path "$LOCAL" --out "${args[@]}" \
    > "logs/${SLURM_JOB_NAME:-pack}-${SLURM_JOB_ID:-0}.race$i.log" 2>&1 &
  PIDS+=($!); i=$((i + 1))
done < "$SPEC"
rc=0
for p in "${PIDS[@]}"; do wait "$p" || rc=1; done
exit $rc
