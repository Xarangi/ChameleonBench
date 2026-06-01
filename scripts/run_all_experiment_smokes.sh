#!/usr/bin/env bash
set -euo pipefail

uv run next-chameleons sweep-plan all_experiment_smokes \
  --output-path runs/all_experiment_smokes_plan.json
uv run next-chameleons sweep-run all_experiment_smokes
