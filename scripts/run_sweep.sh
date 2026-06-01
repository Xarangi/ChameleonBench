#!/usr/bin/env bash
set -euo pipefail

SWEEP="${1:-quick_iteration}"

uv run next-chameleons sweep-plan "${SWEEP}" --output-path "runs/${SWEEP}_plan.json"
uv run next-chameleons sweep-run "${SWEEP}"
