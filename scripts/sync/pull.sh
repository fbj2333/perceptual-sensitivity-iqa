#!/bin/bash
# Pull results (checkpoints, logs) from remote GPU server
# Usage: bash scripts/sync/pull.sh [server_alias] [experiment_name]
#
# This only pulls experiment outputs, not code (code flows local→remote)

set -e

REMOTE_HOST="${1:-gpu-server}"
EXP_NAME="${2:-}"
REMOTE_DIR="~/research/iqa"
LOCAL_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

if [ -z "$EXP_NAME" ]; then
    echo "Usage: bash scripts/sync/pull.sh <server_alias> <experiment_name>"
    echo ""
    echo "Available experiments on ${REMOTE_HOST}:"
    ssh "${REMOTE_HOST}" "ls ${REMOTE_DIR}/experiments/ 2>/dev/null || echo '(none)'"
    exit 1
fi

echo "=== Pulling ${EXP_NAME} from ${REMOTE_HOST} ==="

# Pull checkpoints (only best + latest)
mkdir -p "${LOCAL_DIR}/experiments/${EXP_NAME}"
rsync -avz --progress \
    --include="best_*.pth" \
    --include="latest.pth" \
    --include="config.yaml" \
    --include="metrics.json" \
    --include="train.log" \
    --exclude="*.pth" \
    "${REMOTE_HOST}:${REMOTE_DIR}/experiments/${EXP_NAME}/" \
    "${LOCAL_DIR}/experiments/${EXP_NAME}/"

echo "=== Pull complete ==="
