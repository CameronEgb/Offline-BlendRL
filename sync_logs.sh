#!/bin/bash

# --- Configuration ---
REMOTE_USER_HOST="cegbert@arc.csc.ncsu.edu"
REMOTE_PROJECT_ROOT="/home/cegbert/Code/BlendRL-Minigrid" # Update if renamed on remote

# Check for experiment ID argument
EXP_ID=$1
if [ -z "$EXP_ID" ]; then
    echo "Usage: ./sync_logs.sh <experiment_id>"
    exit 1
fi

# Determine target local name
# If the experiment exists in any of the primary data directories, suffix with -remote
TARGET_ID="$EXP_ID"
if [ -d "out/runs/$EXP_ID" ] || [ -d "logs/$EXP_ID" ] || [ -d "plots/$EXP_ID" ]; then
    TARGET_ID="${EXP_ID}-remote"
    echo "Local data for '$EXP_ID' found. Syncing to '${TARGET_ID}' to avoid collision."
else
    echo "Syncing new experiment '$EXP_ID'..."
fi

# List of directories to mirror
# out/runs contains checkpoints and results
# logs contains stdout/stderr log files
# plots contains generated figures
DIRS=("out/runs" "logs" "plots")

for DIR in "${DIRS[@]}"; do
    REMOTE_PATH="${REMOTE_PROJECT_ROOT}/${DIR}/${EXP_ID}/"
    LOCAL_PATH="${DIR}/${TARGET_ID}/"

    echo "--- Syncing $DIR ---"
    # Create local directory if it doesn't exist
    mkdir -p "$LOCAL_PATH"

    # rsync: -a (archive), -v (verbose), -z (compress), -P (progress)
    # The trailing slash on REMOTE_PATH ensures we sync the *contents* of the remote folder
    # into our local TARGET_ID folder.
    rsync -avzP "${REMOTE_USER_HOST}:${REMOTE_PATH}" "$LOCAL_PATH"
done

echo "===================================================="
echo "Sync complete for $EXP_ID -> $TARGET_ID"
echo "Structure mirrored. Data is indistinguishable from local runs."
echo "===================================================="
