#!/usr/bin/env bash
# TICA Level 1 evaluation with one monitor bank (PREREGISTRATION-TICA.md §2).
# Usage (inside run_cli.sbatch): bash scripts/slurm/tica_l1_eval.sh CONFIG BANK OUT [ADAPTER]
set -euo pipefail
PY="env PYTHONPATH=/scratch/s/sneheel/ChameleonBench/src /scratch/s/sneheel/venvs/q35t29/bin/python"
$PY scripts/tica.py eval --config "$1" --bank "$2" --out "$3" ${4:+--adapter "$4"}
