#!/bin/bash

# Define common settings from mountaincar_batch_17
ENVIRONMENT="mountaincar"
STEPS=5000000 
INTERVALS=10
EVAL_EPISODES=100
EXP_ID="mc_batch_17_blendrl"

echo "Launching BlendeRL Comparison for B17: (Ent 0.08, Envs 4, Steps 1024)"
mkdir -p logs/$EXP_ID

# Run using blendrl_ppo instead of ppo
nohup python3.11 run_full_cycle.py \
    --experimentid $EXP_ID \
    --environment $ENVIRONMENT \
    --online_methods blendrl_ppo \
    --online_steps $STEPS \
    --offline_methods "" \
    --local \
    --lr 0.0003 \
    --logic_lr 0.0003 \
    --blender_lr 0.0003 \
    --ent_coef 0.08 \
    --blend_ent_coef 0.01 \
    --num_envs 4 \
    --num_steps 1024 \
    --ppo_epochs 10 \
    --num_minibatches 64 \
    --intervals_count $INTERVALS \
    --eval_episodes $EVAL_EPISODES \
    --no_overwrite > /dev/null 2>&1 &

echo "BlendeRL comparison (mc_batch_17_blendrl) triggered for 5M steps."
