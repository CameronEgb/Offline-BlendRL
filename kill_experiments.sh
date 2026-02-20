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

if [ "$LOCAL_MODE" = true ]; then
    echo "Killing local processes for experiment $EXP_ID..."
    pkill -f "run_full_cycle.py.*$EXP_ID"
    pkill -f "train_.*py.*$EXP_ID"
else
    echo "Canceling SLURM jobs for experiment $EXP_ID..."
    scancel --name="on_*_$EXP_ID"
    scancel --name="off_*_$EXP_ID"
fi

echo "Removing data for experiment $EXP_ID..."
rm -rf logs/"$EXP_ID"
rm -rf out/runs/"$EXP_ID" # Remove top level dir if exists
find out/runs -name "*$EXP_ID*" -exec rm -rf {} +
find out/tensorboard -name "*$EXP_ID*" -exec rm -rf {} +
find offline_dataset -name "*$EXP_ID*" -exec rm -rf {} +

echo "Done."
