#!/bin/bash
# Delete all local data associated with a specific experiment ID
# Usage: ./delete_logs.sh EXPERIMENT_ID

if [ -z "$1" ]; then
    echo "Usage: $0 EXPERIMENT_ID"
    exit 1
fi

EXP_ID=$1

echo "===================================================="
echo "WARNING: This will DELETE all local data for experiment: $EXP_ID"
echo "Affected directories: logs/$EXP_ID, out/runs/$EXP_ID, plots/$EXP_ID, offline_dataset/*$EXP_ID*"
echo "===================================================="
read -p "Are you sure? (y/n): " confirm

if [[ "$confirm" != "y" ]]; then
    echo "Operation cancelled."
    exit 0
fi

# List of base directories to clean
BASE_DIRS=("logs" "out/runs" "plots")

for BASE in "${BASE_DIRS[@]}"; do
    TARGET="$BASE/$EXP_ID"
    if [ -d "$TARGET" ]; then
        echo "Deleting $TARGET..."
        rm -rf "$TARGET"
    else
        echo "Directory $TARGET not found, skipping."
    fi
done

# Also clean up any datasets matching the ID
echo "Scanning offline_dataset for matching files..."
find offline_dataset -name "*$EXP_ID*" -exec rm -rf {} + 2>/dev/null

# Also clean up the large dataset path
LARGE_DATASET_PATH="/mnt/beegfs/cegbert/offlineDatasets"
if [ -d "$LARGE_DATASET_PATH/$EXP_ID" ]; then
    echo "Deleting large dataset at $LARGE_DATASET_PATH/$EXP_ID..."
    rm -rf "$LARGE_DATASET_PATH/$EXP_ID"
fi

echo "===================================================="
echo "Data for experiment $EXP_ID has been cleared."
echo "===================================================="
