#!/bin/bash
# Sync local code to remote GPU server
# Usage: bash scripts/sync/push.sh [server_alias]
#
# Prerequisites:
#   1. Set up SSH config in ~/.ssh/config with your server alias
#   2. Configure REMOTE_HOST, REMOTE_DIR below

set -e

# ============ Configuration ============
REMOTE_HOST="${1:-gpu-server}"          # SSH alias or user@host
REMOTE_DIR="~/research/iqa"            # Remote project path
LOCAL_DIR="$(cd "$(dirname "$0")/../.." && pwd)"  # Project root

# Exclude patterns (large files, local-only stuff)
EXCLUDES=(
    ".git/"
    "__pycache__/"
    "*.pyc"
    ".DS_Store"
    "data/raw/"
    "data/processed/"
    "data/cache/"
    "checkpoints/"
    "*.pth"
    "*.pt"
    "*.bin"
    "*.safetensors"
    "wandb/"
    "runs/"
    ".venv/"
    "venv/"
    "notebooks/.ipynb_checkpoints/"
    "experiments/*/outputs/"
    "experiments/*/logs/"
)

EXCLUDE_ARGS=""
for pattern in "${EXCLUDES[@]}"; do
    EXCLUDE_ARGS="$EXCLUDE_ARGS --exclude=$pattern"
done

echo "=== Syncing local → ${REMOTE_HOST}:${REMOTE_DIR} ==="
echo "Local:  ${LOCAL_DIR}"
echo "Remote: ${REMOTE_HOST}:${REMOTE_DIR}"
echo ""

# Dry run first
echo "--- Dry run ---"
rsync -avz --dry-run $EXCLUDE_ARGS "${LOCAL_DIR}/" "${REMOTE_HOST}:${REMOTE_DIR}/"

echo ""
read -p "Proceed with sync? [y/N] " confirm
if [[ "$confirm" =~ ^[Yy]$ ]]; then
    rsync -avz --progress $EXCLUDE_ARGS "${LOCAL_DIR}/" "${REMOTE_HOST}:${REMOTE_DIR}/"
    echo "=== Sync complete ==="
else
    echo "=== Sync cancelled ==="
fi
