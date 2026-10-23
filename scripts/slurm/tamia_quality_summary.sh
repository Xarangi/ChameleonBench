#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --job-name=v4b-qsum
#SBATCH --output=/home/sneheel/chameleons/logs/%x-%j.out
# CPU: programmatic quality checks (scripts/tica_quality_check.py) over the free-generation answers queued by
# `scripts/tamia_v4b.sh quality` -> QDIR/quality.json (aggregates only).
#   sbatch --account=aip-<pi> scripts/slurm/tamia_quality_summary.sh QDIR
set -euo pipefail
Q=${1:?quality dir}
cd /home/sneheel/chameleons
ARGS=()
for f in "$Q"/gens/*.jsonl; do ARGS+=("$(basename "$f" .jsonl)=$f"); done
/home/sneheel/chameleons-qwen35/.venv/bin/python scripts/tica_quality_check.py --gens "${ARGS[@]}" --out "$Q/quality.json"
