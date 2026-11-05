#!/usr/bin/env bash
# TICA Level-1 full base test for one model: reads -> fit -> base test (fresh texts, 6 turns) -> analysis.
# Reads and the base test run in the transformers-v5 env; fit and analysis in the main env (joblib, sklearn).
# Usage (inside run_cli.sbatch): bash scripts/slurm/l1_full.sh MODEL_PATH TAG
set -euo pipefail
PY="env PYTHONPATH=/home/sneheel/chameleons/src /home/sneheel/chameleons-qwen35/.venv/bin/python"
OUT=/scratch/sneheel/next_chameleons_artifacts/tica/l1_base/$2
$PY scripts/l1_reads.py --model "$1" --out $OUT/reads --ctx-neg 200
[ -f $OUT/probes/quality.json ] || OMP_NUM_THREADS=1 uv run --extra train python scripts/l1_fit.py --reads $OUT/reads --out $OUT/probes --jobs ${SLURM_CPUS_PER_TASK:-8}
$PY scripts/l1_base_test.py --model "$1" --probes $OUT/probes --out $OUT/base --turns 6 --episodes-per-concept 2 --gen-check 150
uv run --extra train python scripts/l1_analyze.py --records $OUT/base/records.jsonl --out $OUT/base/summary.json
