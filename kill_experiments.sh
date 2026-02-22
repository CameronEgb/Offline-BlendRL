#!/bin/bash
# Kill experiments by ID
# Usage: ./kill_experiments.sh EXPERIMENT_ID [--local]

if [ -z "$1" ]; then
    echo "Usage: $0 EXPERIMENT_ID [--local]"
    exit 1
fi

EXP_ID=""
LOCAL_MODE=false

for arg in "$@"; do
    if [ "$arg" == "--local" ]; then
        LOCAL_MODE=true
    else
        EXP_ID=$arg
    fi
done

if [ -z "$EXP_ID" ]; then
    echo "Error: EXPERIMENT_ID not specified."
    exit 1
fi

echo "--- Killing all processes for experiment: $EXP_ID ---"

# 1. Kill Local Processes
echo "Killing local processes..."
pkill -f "$EXP_ID" 2>/dev/null

# 2. Cancel Slurm Jobs (if on a cluster)
if command -v scancel &> /dev/null; then
    echo "Slurm detected. Canceling jobs..."
    
    # Precise cancellation using job IDs from out/runs/EXP_ID/jobids.txt
    JOBIDS_FILE="out/runs/$EXP_ID/jobids.txt"
    if [ -f "$JOBIDS_FILE" ]; then
        echo "Reading job IDs from $JOBIDS_FILE..."
        while read -r jid; do
            if [ -n "$jid" ]; then
                echo "Canceling Slurm job $jid..."
                scancel "$jid" 2>/dev/null
            fi
        done < "$JOBIDS_FILE"
    fi
    
    # Fallback/Safety: Kill anything with the name matching the ID
    scancel --name="*$EXP_ID*" 2>/dev/null
else
    echo "Slurm not detected, skipping scancel."
fi

echo "Removing data for experiment $EXP_ID..."
rm -rf logs/"$EXP_ID"
rm -rf out/runs/"$EXP_ID" # Remove top level dir if exists
find out/runs -name "*$EXP_ID*" -exec rm -rf {} + 2>/dev/null
find out/tensorboard -name "*$EXP_ID*" -exec rm -rf {} + 2>/dev/null
find offline_dataset -name "*$EXP_ID*" -exec rm -rf {} + 2>/dev/null

echo "Done."
