#!/usr/bin/env bash
# TICA Level-1 smoke: reads -> fit -> base test on a few concepts, for one model.
# Usage (inside run_cli.sbatch): bash scripts/slurm/l1_smoke.sh MODEL_PATH TAG
set -euo pipefail
# Reads and the base test need transformers v5 (Qwen3.5, Gemma-4); fit and analysis run in the main env (joblib, sklearn).
PY="env PYTHONPATH=/home/sneheel/chameleons/src /home/sneheel/chameleons-qwen35/.venv/bin/python"
OUT=/scratch/sneheel/next_chameleons_artifacts/tica/l1_smoke_v2/$2
$PY scripts/l1_reads.py --model "$1" --out $OUT/reads --ctx-neg 60 --limit-concepts 16
OMP_NUM_THREADS=1 uv run --extra train python scripts/l1_fit.py --reads $OUT/reads --out $OUT/probes --select-concepts 8 --jobs 8
$PY scripts/l1_base_test.py --model "$1" --probes $OUT/probes --out $OUT/base --turns 4 --gen-check 24 --limit-concepts 12
uv run --extra train python scripts/l1_analyze.py --records $OUT/base/records.jsonl --out $OUT/base/summary.json
