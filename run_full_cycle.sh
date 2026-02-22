#!/bin/bash
# --- Configuration Section ---
EXPERIMENT_ID="seaquest_speed_test"
ENVIRONMENT="seaquest"
ONLINE_METHODS="ppo,blendrl_ppo"
ONLINE_STEPS=5000
OFFLINE_METHODS="iql,blendrl_iql"
OFFLINE_DATASETS="blendrl_ppo,ppo"
OFFLINE_EPOCHS=5
INTERVALS_COUNT=3
EVAL_EPISODES=10
# -----------------------------
USE_LARGE_DATASET_PATH=true
LARGE_DATASET_PATH="/mnt/beegfs/cegbert/offlineDatasets"
LOCAL=false
SEED=1
# -----------------------------

if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Warning: local venv not found."
fi

export PYTHONPATH=$PYTHONPATH:.

# We pass the hardcoded defaults first, and then "$@" at the end.
# In argparse, the LAST occurrence of a flag wins, so "$@" takes precedence.
python3.11 run_full_cycle.py \
    --experimentid "$EXPERIMENT_ID" \
    --environment "$ENVIRONMENT" \
    --online_methods "$ONLINE_METHODS" \
    --online_steps "$ONLINE_STEPS" \
    --offline_methods "$OFFLINE_METHODS" \
    --offline_datasets "$OFFLINE_DATASETS" \
    --offline_epochs "$OFFLINE_EPOCHS" \
    --intervals_count "$INTERVALS_COUNT" \
    --eval_episodes "$EVAL_EPISODES" \
    --large_dataset_path "$LARGE_DATASET_PATH" \
    --seed "$SEED" \
    $( [ "$USE_LARGE_DATASET_PATH" = true ] && echo "--use_large_dataset_path" ) \
    $( [ "$LOCAL" = true ] && echo "--local" ) \
    "$@"
