#!/bin/bash
# Sync logs from cluster
# Usage: ./sync_logs.sh [CLUSTER_USER@CLUSTER_HOST] [REMOTE_PATH]
# Default user/host must be set via env var or passed

USER_HOST=${1:-$CLUSTER_USER_HOST}
REMOTE_PATH=${2:-"~/Research/blendrl-diff/offline_blendrl/out/runs"}

if [ -z "$USER_HOST" ]; then
    echo "Usage: $0 USER@HOST [REMOTE_PATH]"
    exit 1
fi

echo "Syncing logs from $USER_HOST:$REMOTE_PATH to out/runs/..."

rsync -avz --progress "$USER_HOST:$REMOTE_PATH/" out/runs/

echo "Sync complete."
