#!/bin/bash
# Kill, Delete, and Rerun experiments by ID or by Config List
# Usage: ./rerun_full_cycle.sh [EXPERIMENT_ID | LIST_NAME] [additional_args]

if [ -z "$1" ]; then
    echo "Usage: $0 [EXPERIMENT_ID | LIST_NAME] [additional_args]"
    exit 1
fi

INPUT_ARG="$1"
shift # Shift to pass remaining args to run_full_cycle.sh

echo ">>> Rerunning experiments for: $INPUT_ARG"

# 1. Kill
echo ">>> Step 1: Killing current experiments..."
./bin/kill_experiments.sh "$INPUT_ARG" --local

# 2. Delete Logs
echo ">>> Step 2: Deleting existing data..."
./bin/delete_logs.sh "$INPUT_ARG" --yes

# 3. Run
echo ">>> Step 3: Starting fresh run..."
./bin/run_full_cycle.sh "$INPUT_ARG" "$@"
