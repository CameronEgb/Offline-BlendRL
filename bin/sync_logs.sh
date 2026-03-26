#!/bin/bash

# --- Configuration ---
REMOTE_USER_HOST="cegbert@arc.csc.ncsu.edu"
REMOTE_PROJECT_ROOT="/home/cegbert/Code/Offline-BlendRL"

# New results base
RESULTS_BASE="results"

# Check for experiment ID argument
EXP_ID=$1
if [ -z "$EXP_ID" ]; then
    echo "No experiment ID provided. Checking for all remote experiments..."
    
    # 1. Get a list of remote experiment IDs from results/experiments
    REMOTE_EXPS=$(ssh "${REMOTE_USER_HOST}" "ls -1 ${REMOTE_PROJECT_ROOT}/${RESULTS_BASE}/experiments/ 2>/dev/null | grep -v '^\.'")
    
    if [ -z "$REMOTE_EXPS" ]; then
        echo "No remote experiments found in ${REMOTE_PROJECT_ROOT}/${RESULTS_BASE}/experiments/"
        exit 0
    fi

    for EXP in $REMOTE_EXPS; do
        # Sync each experiment with its original name
        ./sync_logs.sh "$EXP"
    done
    exit 0
fi

echo ">>> Syncing remote '$EXP_ID' to local workspace..."

# List of directories to mirror inside results/
DIRS=("experiments" "logs" "plots")

for DIR in "${DIRS[@]}"; do
    REMOTE_PATH="${REMOTE_PROJECT_ROOT}/${RESULTS_BASE}/${DIR}/${EXP_ID}/"
    LOCAL_PATH="${RESULTS_BASE}/${DIR}/${EXP_ID}/"

    echo "--- Syncing results/$DIR ---"
    mkdir -p "$LOCAL_PATH"

    # rsync: -a (archive), -v (verbose), -z (compress), -P (progress)
    rsync -avzP "${REMOTE_USER_HOST}:${REMOTE_PATH}" "$LOCAL_PATH"
done

echo "===================================================="
echo "Sync complete for $EXP_ID"
echo "===================================================="
