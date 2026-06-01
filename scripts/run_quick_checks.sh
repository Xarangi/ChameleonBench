#!/usr/bin/env bash
set -euo pipefail

uv run --extra dev pytest
uv run --extra dev ruff check .
uv run next-chameleons paper-dry-run --output-dir runs/quick_checks/paper
uv run next-chameleons adaptive-smoke --output-dir runs/quick_checks/adaptive
uv run next-chameleons experiment-smoke multi_probe_evasion \
  --output-dir runs/quick_checks/multi_probe_evasion
