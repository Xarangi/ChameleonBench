#!/bin/bash
cd /scratch/s/sneheel/ChameleonBench-detmix2
source scripts/slurm/_local_env.sh
export OMP_NUM_THREADS=8
env PYTHONPATH=src $PY -m pytest -q tests/test_tica_evdet_defender.py 2>&1 | tail -30
echo ==== suite
env PYTHONPATH=src $PY -m pytest -q tests/ -k tica -p no:cacheprovider 2>&1 | tail -12
echo ==== ruff
/scratch/s/sneheel/ChameleonBench/.venv/bin/ruff check src/next_chameleons/tica/evdet_defender.py src/next_chameleons/tica/adaptive.py tests/test_tica_evdet_defender.py 2>&1 | tail -15
