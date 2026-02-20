#!/bin/bash

# --- Configuration ---
REMOTE_USER_HOST="cegbert@arc.csc.ncsu.edu"
REMOTE_PROJECT_ROOT="/home/cegbert/Code/Offline-BlendRL"

# Check for experiment ID argument
EXP_ID=$1
if [ -z "$EXP_ID" ]; then
    echo "No experiment ID provided. Checking for all remote experiments..."
    
    # 1. Get a list of remote experiment IDs from out/runs
    # Filter out hidden files and system artifacts
    REMOTE_EXPS=$(ssh "${REMOTE_USER_HOST}" "ls -1 ${REMOTE_PROJECT_ROOT}/out/runs/ 2>/dev/null | grep -v '^\.'")
    
    if [ -z "$REMOTE_EXPS" ]; then
        echo "No remote experiments found in ${REMOTE_PROJECT_ROOT}/out/runs/"
        exit 0
    fi

    for EXP in $REMOTE_EXPS; do
        # 2. Check if it already exists locally (in any of the 3 key dirs)
        if [ -d "out/runs/$EXP" ] || [ -d "logs/$EXP" ] || [ -d "plots/$EXP" ]; then
             # ONLY if it exists locally, we check if it's already a '-remote' version
             # If we've already synced it before as '-remote', we keep syncing to that same folder
             if [[ "$EXP" == *"-remote" ]]; then
                 ./sync_logs.sh "$EXP"
             else
                 # It's a local run, so we must sync the remote version to '-remote'
                 ./sync_logs.sh "$EXP"
             fi
        else
            # 3. New experiment (not present locally), sync normally
            ./sync_logs.sh "$EXP"
        fi
    done
    exit 0
fi

# --- Core Sync Logic for a Single EXP_ID ---
# Determine target local name: Always append -remote for remote data
# unless the remote folder name already ends in -remote.
if [[ "$EXP_ID" == *"-remote" ]]; then
    TARGET_ID="$EXP_ID"
else
    TARGET_ID="${EXP_ID}-remote"
fi

echo ">>> Syncing remote '$EXP_ID' to local '$TARGET_ID'..."

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
