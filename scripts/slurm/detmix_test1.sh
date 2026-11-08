#!/bin/bash
cd /scratch/s/sneheel/ChameleonBench-detmix
source scripts/slurm/_local_env.sh
env PYTHONPATH=src $PY -m pytest -q tests/test_tica_evdet_defender.py 2>&1 | tail -5
