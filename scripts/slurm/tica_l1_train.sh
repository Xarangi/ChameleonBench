#!/usr/bin/env bash
# TICA Level 1 training for one seed (PREREGISTRATION-TICA.md §1). Output: <config out>/seed{N}/train.
# Usage (inside run_cli.sbatch): bash scripts/slurm/tica_l1_train.sh CONFIG SEED
set -euo pipefail
PY="env PYTHONPATH=/scratch/s/sneheel/ChameleonBench/src /scratch/s/sneheel/venvs/q35t29/bin/python"
$PY scripts/tica.py train --config "$1" --seed "$2"
