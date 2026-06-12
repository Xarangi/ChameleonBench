#!/usr/bin/env bash
# One-way sync: configs/ -> src/next_chameleons/builtin_configs/.
# configs/ is the editable source of truth; the packaged copy must stay
# byte-identical (enforced by tests/test_config_sync.py).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="${REPO_ROOT}/configs"
TARGET_DIR="${REPO_ROOT}/src/next_chameleons/builtin_configs"

rsync -a --delete --exclude "__pycache__" "${SOURCE_DIR}/" "${TARGET_DIR}/"
echo "Synced ${SOURCE_DIR} -> ${TARGET_DIR}"
