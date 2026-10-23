#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --job-name=drain-post
#SBATCH --output=/scratch/s/sneheel/ChameleonBench/logs/%x-%j.out
# The CPU tail of tica_val_drain.sh on its own (10-23: drains that hung before it ran): merge sharded panels and
# write quality.json for every quality dir with free-generation answers. Idempotent (done outputs skipped / rewritten).
set -uo pipefail
cd /scratch/s/sneheel/ChameleonBench
PY=/scratch/s/sneheel/venvs/q35t29/bin/python
R_=/scratch/s/sneheel/next_chameleons_artifacts/tica/runs
for f0 in "$R_"/*/quality/step_*/panel_*.s0.json "$R_"/v4b_quality_base/panel_*.s0.json; do
  [ -f "$f0" ] || continue
  out=${f0%.s0.json}.json; [ -f "$out" ] && continue
  sh_=("${f0%.s0.json}".s*.json)
  (( ${#sh_[@]} >= ${PANEL_SHARDS:-4} )) || { echo "panel shards incomplete: $f0"; continue; }
  PYTHONPATH=src $PY scripts/tica_panel_merge.py --out "$out" "${sh_[@]}" || echo "panel merge failed: $out"
done
for q in "$R_"/v4b_quality_base "$R_"/*/quality/step_*; do
  ls "$q"/gens/*.jsonl > /dev/null 2>&1 || continue
  [ -f "$q/quality.json" ] && [ -z "$(find "$q/gens" -newer "$q/quality.json" -name '*.jsonl')" ] && continue
  A=(); for f in "$q"/gens/*.jsonl; do A+=("$(basename "$f" .jsonl)=$f"); done
  PYTHONPATH=src $PY scripts/tica_quality_check.py --gens "${A[@]}" --out "$q/quality.json" || echo "quality summary failed: $q"
done
echo "post done"
