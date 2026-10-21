#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --job-name=v4b-qsum
#SBATCH --output=/scratch/s/sneheel/ChameleonBench/logs/%x-%j.out
# CPU: programmatic quality checks (scripts/tica_quality_check.py) over the free-generation answers queued by
# `scripts/tamia_v4b.sh quality` -> QDIR/quality.json (aggregates only).
#   sbatch --account=aip-<pi> scripts/slurm/tamia_quality_summary.sh QDIR
set -euo pipefail
Q=${1:?quality dir}
cd /scratch/s/sneheel/ChameleonBench
ARGS=()
for f in "$Q"/gens/*.jsonl; do ARGS+=("$(basename "$f" .jsonl)=$f"); done
/scratch/s/sneheel/venvs/q35t29/bin/python scripts/tica_quality_check.py --gens "${ARGS[@]}" --out "$Q/quality.json"
