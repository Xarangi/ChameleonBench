#!/usr/bin/env bash
# Pair-dilution check and PAIR GATE for hidden-mode TICA runs (scripts/l1_pair_check.py).
# Usage (inside run_cli.sbatch): bash scripts/slurm/tica_pair_check.sh CONFIG OUT
set -euo pipefail
PY="env PYTHONPATH=/scratch/s/sneheel/ChameleonBench/src /scratch/s/sneheel/venvs/q35t29/bin/python"
$PY scripts/l1_pair_check.py --config "$1" --out "$2" --pairings 4
