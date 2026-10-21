#!/usr/bin/env bash
# TICA Level-1 generation check only (prompt validation), reusing a smoke probe bank.
# Usage (inside run_cli.sbatch): bash scripts/slurm/l1_gencheck.sh MODEL_PATH TAG
set -euo pipefail
PY="env PYTHONPATH=/scratch/s/sneheel/ChameleonBench/src /scratch/s/sneheel/venvs/q35t29/bin/python"
OUT=/scratch/s/sneheel/next_chameleons_artifacts/tica/l1_smoke/$2
$PY scripts/l1_base_test.py --model "$1" --probes $OUT/probes --out $OUT/gencheck_v2 --gen-only --texts-per-concept 6 --gen-check 80
