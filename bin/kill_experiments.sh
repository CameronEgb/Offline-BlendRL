#!/bin/bash
# Kill experiments by ID or by Config List
# Usage: ./kill_experiments.sh [EXPERIMENT_ID | LIST_NAME] [--local]

if [ -z "$1" ]; then
    echo "Usage: $0 [EXPERIMENT_ID | LIST_NAME] [--local]"
    exit 1
fi

INPUT_ARG="$1"
LOCAL_MODE=false

for arg in "$@"; do
    if [ "$arg" == "--local" ]; then
        LOCAL_MODE=true
    fi
done

# Function to kill a single experiment
kill_single_exp() {
    local EXP_ID=$1
    local GROUP=$2
    
    local DATA_DIR="results/experiments"
    local LOG_DIR="results/logs"
    local DATASET_DIR="results/datasets"
    local TB_DIR="results/experiments"
    local PLOTS_DIR="results/plots"
    
    if [ -n "$GROUP" ]; then
        DATA_DIR="$DATA_DIR/$GROUP"
        LOG_DIR="$LOG_DIR/$GROUP"
        DATASET_DIR="$DATASET_DIR/$GROUP"
        TB_DIR="$TB_DIR/$GROUP"
    fi

    echo "--- Killing processes for: $EXP_ID ---"
    # Match EXP_ID in the command line, allowing for group prefixes
    pkill -u "$USER" -f "python.*$EXP_ID" 2>/dev/null
    
    # Also try matching the basename if it's a path
    local EXP_BASE=$(basename "$EXP_ID")
    if [ "$EXP_BASE" != "$EXP_ID" ]; then
        pkill -u "$USER" -f "python.*$EXP_BASE" 2>/dev/null
    fi

    if command -v scancel &> /dev/null; then
        JOBIDS_FILE="$DATA_DIR/$EXP_ID/jobids.txt"
        if [ -f "$JOBIDS_FILE" ]; then
            while read -r jid; do
                [ -n "$jid" ] && scancel "$jid" 2>/dev/null
            done < "$JOBIDS_FILE"
        fi
        scancel --name="*$EXP_ID*" 2>/dev/null
    fi

    echo "Removing data for experiment $EXP_ID..."
    # Broad recursive cleanup for related run directories and logs
    # These often contain the EXP_ID or EXP_BASE in their name and can be nested
    local EXP_BASE=$(basename "$EXP_ID")
    
    # Use find from the results root to catch all nested logs and experiments
    # 1. Targeted cleanup within the group (if provided)
    if [ -n "$GROUP" ]; then
        find "results/logs/$GROUP" -name "*$EXP_BASE*" -exec rm -rf {} + 2>/dev/null
        find "results/experiments/$GROUP" -name "*$EXP_BASE*" -exec rm -rf {} + 2>/dev/null
        find "results/datasets/$GROUP" -name "*$EXP_BASE*" -exec rm -rf {} + 2>/dev/null
        find "results/plots/$GROUP" -name "*$EXP_BASE*" -exec rm -rf {} + 2>/dev/null
    fi

    # 2. Global cleanup for this EXP_BASE just in case it's in a different location
    find "results/logs" -name "*$EXP_BASE*" -exec rm -rf {} + 2>/dev/null
    find "results/experiments" -name "*$EXP_BASE*" -exec rm -rf {} + 2>/dev/null
    find "results/datasets" -name "*$EXP_BASE*" -exec rm -rf {} + 2>/dev/null
    find "results/plots" -name "*$EXP_BASE*" -exec rm -rf {} + 2>/dev/null
}

LIST_PATH="in/config/configLists/$INPUT_ARG"
if [ -f "$LIST_PATH" ]; then
    echo "Killing all experiments in list: $INPUT_ARG"
    while IFS= read -r CFG || [[ -n "$CFG" ]]; do
        [[ -z "$CFG" || "$CFG" == \#* ]] && continue
        
        # Resolve config path
        ACTUAL_CFG="$CFG"
        if [ ! -f "$ACTUAL_CFG" ] && [ -f "in/config/$CFG" ]; then
            ACTUAL_CFG="in/config/$CFG"
        elif [ ! -f "$ACTUAL_CFG" ] && [ -f "in/config/$CFG.yaml" ]; then
            ACTUAL_CFG="in/config/$CFG.yaml"
        fi

        EXP_ID=$(python3 -c "import yaml; print(yaml.safe_load(open('$ACTUAL_CFG')).get('experimentid', ''))" 2>/dev/null)
        GROUP=$(python3 -c "import yaml; print(yaml.safe_load(open('$ACTUAL_CFG')).get('group', ''))" 2>/dev/null)
        if [ -z "$GROUP" ]; then GROUP="$INPUT_ARG"; fi
        [ -n "$EXP_ID" ] && kill_single_exp "$EXP_ID" "$GROUP"
    done < "$LIST_PATH"
elif [ -f "in/config/$INPUT_ARG" ] || [ -f "in/config/$INPUT_ARG.yaml" ]; then
    # Check if it's a config file name in in/config/
    if [ -f "in/config/$INPUT_ARG" ]; then
        ACTUAL_CFG="in/config/$INPUT_ARG"
    else
        ACTUAL_CFG="in/config/$INPUT_ARG.yaml"
    fi
    echo "Killing single config: $ACTUAL_CFG"
    EXP_ID=$(python3 -c "import yaml; print(yaml.safe_load(open('$ACTUAL_CFG')).get('experimentid', ''))" 2>/dev/null)
    GROUP=$(python3 -c "import yaml; print(yaml.safe_load(open('$ACTUAL_CFG')).get('group', ''))" 2>/dev/null)
    [ -n "$EXP_ID" ] && kill_single_exp "$EXP_ID" "$GROUP"
else
    # Handle listName/expID format
    if [[ "$INPUT_ARG" == *"/"* ]]; then
        GROUP="${INPUT_ARG%/*}"
        EXP_ID="${INPUT_ARG##*/}"
        kill_single_exp "$EXP_ID" "$GROUP"
    else
        kill_single_exp "$INPUT_ARG" ""
    fi
fi

echo "Done."
