#!/usr/bin/env bash
# GPU smoke of the TICA Level-1 trainer: base eval -> train -> trained eval (same episodes).
# Usage (inside run_cli.sbatch): bash scripts/slurm/tica_train_smoke.sh CONFIG
set -euo pipefail
PY="env PYTHONPATH=/scratch/s/sneheel/ChameleonBench/src /scratch/s/sneheel/venvs/q35t29/bin/python"
OUT=$($PY -c "from next_chameleons.tica.config import load_run_config as l; print(l('$1').out)")
[ -f $OUT/base/eval/DONE ] || { $PY scripts/tica.py eval --config "$1" --out $OUT/base && touch $OUT/base/eval/DONE; }
$PY scripts/tica.py train --config "$1"
$PY scripts/tica.py eval --config "$1" --adapter $OUT/train/final --out $OUT/trained
