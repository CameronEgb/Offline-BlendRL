#!/bin/bash

# --- Configuration ---
REMOTE_USER_HOST="cegbert@arc.csc.ncsu.edu"
REMOTE_PROJECT_ROOT="/home/cegbert/Code/BlendRL-Minigrid"

echo "===================================================="
echo "WARNING: This will PERMANENTLY DELETE all remote data"
echo "in out/runs, logs, and plots on the cluster."
echo "===================================================="
read -p "Are you absolutely sure? (y/n): " confirm

if [[ "$confirm" != "y" ]]; then
    echo "Operation cancelled."
    exit 0
fi

# Directories to clear
DIRS=("out/runs" "logs" "plots")

for DIR in "${DIRS[@]}"; do
    echo "Clearing remote $DIR..."
    # We delete the *contents* of the directory but keep the directory itself
    ssh "${REMOTE_USER_HOST}" "rm -rf ${REMOTE_PROJECT_ROOT}/${DIR}/*"
done

echo "===================================================="
echo "Remote logs and data cleared successfully."
echo "===================================================="
