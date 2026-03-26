#!/bin/bash
# --- Configuration Section ---
# Default list of configuration files (if no list argument is provided)
CONFIGS=(
    "in/config/mc_default.yaml"
    "in/config/mc_no_left.yaml"
    "in/config/seaquest_test.yaml"
)

# Global flags that apply to all configs in this run
LOCAL=true
NO_OVERWRITE=true
# -----------------------------

if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Warning: local venv not found."
fi

export PYTHONPATH=$PYTHONPATH:.

# Check if an argument is provided (list name or direct config)
GROUP_NAME=""
if [ -n "$1" ]; then
    INPUT_ARG="$1"
    
    # 1. Check if it's a direct file path to a config
    if [ -f "$INPUT_ARG" ]; then
        CONFIGS=("$INPUT_ARG")
        shift
    else
        # 2. Check if it's a list name
        LIST_PATH="in/config/configLists/$INPUT_ARG"
        if [ -f "$LIST_PATH" ]; then
            echo "Loading config list from: $LIST_PATH"
            # Compatibility fix for macOS/older bash (replaces mapfile)
            CONFIGS=()
            while IFS= read -r line || [[ -n "$line" ]]; do
                CONFIGS+=("$line")
            done < "$LIST_PATH"
            GROUP_NAME="$INPUT_ARG"
            shift
        elif [ -f "in/config/$INPUT_ARG" ] || [ -f "in/config/$INPUT_ARG.yaml" ]; then
            # 3. Check if it's a config file name in in/config/
            if [ -f "in/config/$INPUT_ARG" ]; then
                CONFIGS=("in/config/$INPUT_ARG")
            else
                CONFIGS=("in/config/$INPUT_ARG.yaml")
            fi
            echo "Loading single config: ${CONFIGS[0]}"
            shift
        elif [[ "$INPUT_ARG" == *"/"* ]]; then
            # 4. Support listName/expID format (Surgical Rerun)
            GROUP_NAME="${INPUT_ARG%/*}"
            EXP_ID="${INPUT_ARG##*/}"
            echo "Surgical rerun for experiment: $EXP_ID in group: $GROUP_NAME"
            CONFIGS=("SINGLE_EXP_RERUN")
            shift
        else
            echo "Error: Argument '$INPUT_ARG' not found as config file or config list."
            exit 1
        fi
    fi
fi

for CFG in "${CONFIGS[@]}"; do
    [[ -z "$CFG" || "$CFG" == \#* ]] && continue

    # Resolve config path
    ACTUAL_CFG="$CFG"
    if [ "$CFG" != "SINGLE_EXP_RERUN" ] && [ ! -f "$CFG" ]; then
        if [ -f "in/config/$CFG" ]; then
            ACTUAL_CFG="in/config/$CFG"
        elif [ -f "in/config/$CFG.yaml" ]; then
            ACTUAL_CFG="in/config/$CFG.yaml"
        fi
    fi

    if [ "$CFG" == "SINGLE_EXP_RERUN" ]; then
         echo "===================================================="
         echo "Starting Surgical Rerun: $INPUT_ARG"
         echo "===================================================="
         python3 run_full_cycle.py \
            --experimentid "$INPUT_ARG" \
            $( [ "$LOCAL" = true ] && echo "--local" ) \
            $( [ "$NO_OVERWRITE" = true ] && echo "--no_overwrite" ) \
            "$@"
    else
        echo "===================================================="
        echo "Starting Experiment with Config: $ACTUAL_CFG"
        [ -n "$GROUP_NAME" ] && echo "Group: $GROUP_NAME"
        echo "===================================================="
        
        python3 run_full_cycle.py \
            --config "$ACTUAL_CFG" \
            $( [ -n "$GROUP_NAME" ] && echo "--group $GROUP_NAME" ) \
            $( [ "$LOCAL" = true ] && echo "--local" ) \
            $( [ "$NO_OVERWRITE" = true ] && echo "--no_overwrite" ) \
            "$@"
    fi
done
