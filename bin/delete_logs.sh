#!/bin/bash

# Script to delete local logs and results for an experiment or config list.
# Usage: ./delete_logs.sh <experiment_id|config_list_name> [--force]

INPUT_ARG="$1"
FORCE=false
if [[ "$2" == "--force" ]]; then
    FORCE=true
fi

if [ -z "$INPUT_ARG" ]; then
    echo "Usage: ./delete_logs.sh <experiment_id|config_list_name> [--force]"
    exit 1
fi

delete_single_exp() {
    EXP_ID="$1"
    GROUP="$2"
    
    # Use just the base name for matching logs if it's a path
    EXP_BASE="${EXP_ID##*/}"
    
    echo "Deleting data for: $EXP_ID in group: $GROUP"
    
    # 1. Targeted cleanup if group is provided
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

if [ "$FORCE" = false ]; then
    echo "===================================================="
    echo "WARNING: This will DELETE all local data for: $INPUT_ARG"
    echo "===================================================="
    read -p "Are you sure? (y/n): " confirm

    if [[ "$confirm" != "y" ]]; then
        echo "Operation cancelled."
        exit 0
    fi
fi

LIST_PATH="in/config/configLists/$INPUT_ARG"
if [ -f "$LIST_PATH" ]; then
    echo "Cleaning up all experiments in list: $INPUT_ARG"
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
        [ -n "$EXP_ID" ] && delete_single_exp "$EXP_ID" "$GROUP"
    done < "$LIST_PATH"
elif [ -f "in/config/$INPUT_ARG" ] || [ -f "in/config/$INPUT_ARG.yaml" ]; then
    # 3. Check if it's a config file name in in/config/
    if [ -f "in/config/$INPUT_ARG" ]; then
        ACTUAL_CFG="in/config/$INPUT_ARG"
    else
        ACTUAL_CFG="in/config/$INPUT_ARG.yaml"
    fi
    echo "Cleaning up single config: $ACTUAL_CFG"
    EXP_ID=$(python3 -c "import yaml; print(yaml.safe_load(open('$ACTUAL_CFG')).get('experimentid', ''))" 2>/dev/null)
    GROUP=$(python3 -c "import yaml; print(yaml.safe_load(open('$ACTUAL_CFG')).get('group', ''))" 2>/dev/null)
    [ -n "$EXP_ID" ] && delete_single_exp "$EXP_ID" "$GROUP"
elif [[ "$INPUT_ARG" == *"/"* ]]; then
    GROUP="${INPUT_ARG%/*}"
    EXP_ID="${INPUT_ARG##*/}"
    delete_single_exp "$EXP_ID" "$GROUP"
else
    # Treat INPUT_ARG as the EXP_ID directly if no config found
    delete_single_exp "$INPUT_ARG" ""
fi
