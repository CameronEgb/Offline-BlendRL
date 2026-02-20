#!/bin/bash

# --- Configuration ---
LOCAL_DIRS=("out/runs" "logs" "plots")

echo "===================================================="
echo "WARNING: This will DELETE all local data synced from remote."
echo "Any folder ending in '-remote' inside out/runs, logs, and plots will be removed."
echo "===================================================="
read -p "Are you sure you want to delete these local copies? (y/n): " confirm

if [[ "$confirm" != "y" ]]; then
    echo "Operation cancelled."
    exit 0
fi

for DIR in "${LOCAL_DIRS[@]}"; do
    if [ -d "$DIR" ]; then
        echo "Scanning $DIR for '-remote' directories..."
        # Find directories ending in -remote and delete them
        find "$DIR" -maxdepth 1 -type d -name "*-remote" -exec rm -rf {} + -print
    else
        echo "Directory $DIR not found, skipping."
    fi
done

echo "===================================================="
echo "Local copies of remote logs have been cleared."
echo "===================================================="
