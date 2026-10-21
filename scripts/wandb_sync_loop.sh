#!/bin/bash
# Sync OFFLINE TICA wandb runs from the login node (compute nodes have no internet; REPLICATION
# 2026-10-16l). --clean removes synced offline files afterwards (the account is near its 1M-file
# quota). Run in tmux / nohup:  bash scripts/wandb_sync_loop.sh [minutes]   (needs `wandb login`)
MIN=${1:-10}
ROOT=/scratch/s/sneheel/next_chameleons_artifacts/tica/runs
WB=/scratch/s/sneheel/ChameleonBench/.venv/bin/wandb
while true; do
  for d in $(find "$ROOT" -maxdepth 4 -type d -name "offline-run-*" 2>/dev/null); do
    $WB sync "$d" >/dev/null 2>&1 && echo "$(date +%T) synced $d"
  done
  for w in $(find "$ROOT" -maxdepth 4 -type d -name wandb 2>/dev/null); do
    (cd "$(dirname "$w")" && $WB sync --clean --clean-old-hours 2 --clean-force >/dev/null 2>&1)
  done
  sleep $((MIN * 60))
done
