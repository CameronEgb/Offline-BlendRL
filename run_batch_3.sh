#!/bin/bash

# Define common settings
ENVIRONMENT="mountaincar"
STEPS=5000000 # 5M steps for sparse learning
INTERVALS=10
EVAL_EPISODES=100

# 11. mountaincar_batch_11: High Exploration Baseline
echo "Launching MC_B11: High Exploration (Ent 0.1, Envs 1, Steps 2048)"
mkdir -p logs/mountaincar_batch_11
nohup python3.11 run_full_cycle.py --experimentid mountaincar_batch_11 --environment $ENVIRONMENT --online_methods ppo --online_steps $STEPS --offline_methods "" --local --lr 0.0003 --ent_coef 0.1 --num_envs 1 --num_steps 2048 --ppo_epochs 10 --num_minibatches 64 --intervals_count $INTERVALS --eval_episodes $EVAL_EPISODES --no_overwrite > /dev/null 2>&1 &

# 12. mountaincar_batch_12: Multi-Env Sparse Exploration
echo "Launching MC_B12: Multi-Env Exploration (Ent 0.08, Envs 4, Steps 1024)"
mkdir -p logs/mountaincar_batch_12
nohup python3.11 run_full_cycle.py --experimentid mountaincar_batch_12 --environment $ENVIRONMENT --online_methods ppo --online_steps $STEPS --offline_methods "" --local --lr 0.0003 --ent_coef 0.08 --num_envs 4 --num_steps 1024 --ppo_epochs 10 --num_minibatches 64 --intervals_count $INTERVALS --eval_episodes $EVAL_EPISODES --no_overwrite > /dev/null 2>&1 &

# 13. mountaincar_batch_13: Stable Swing
echo "Launching MC_B13: Stable Swing (LR 2e-4, Ent 0.05, Envs 2, Steps 2048)"
mkdir -p logs/mountaincar_batch_13
nohup python3.11 run_full_cycle.py --experimentid mountaincar_batch_13 --environment $ENVIRONMENT --online_methods ppo --online_steps $STEPS --offline_methods "" --local --lr 0.0002 --ent_coef 0.05 --num_envs 2 --num_steps 2048 --ppo_epochs 10 --num_minibatches 64 --intervals_count $INTERVALS --eval_episodes $EVAL_EPISODES --no_overwrite > /dev/null 2>&1 &

# 14. mountaincar_batch_14: Learning Frequency
echo "Launching MC_B14: High Frequency (Ent 0.05, Envs 1, Steps 4096, Epochs 20)"
mkdir -p logs/mountaincar_batch_14
nohup python3.11 run_full_cycle.py --experimentid mountaincar_batch_14 --environment $ENVIRONMENT --online_methods ppo --online_steps $STEPS --offline_methods "" --local --lr 0.0003 --ent_coef 0.05 --num_envs 1 --num_steps 4096 --ppo_epochs 20 --num_minibatches 64 --intervals_count $INTERVALS --eval_episodes $EVAL_EPISODES --no_overwrite > /dev/null 2>&1 &

# 15. mountaincar_batch_15: Signal Seeker Refined
echo "Launching MC_B15: Signal Seeker Refined (GAE 0.98, Ent 0.06, Envs 1, Steps 2048)"
mkdir -p logs/mountaincar_batch_15
nohup python3.11 run_full_cycle.py --experimentid mountaincar_batch_15 --environment $ENVIRONMENT --online_methods ppo --online_steps $STEPS --offline_methods "" --local --lr 0.0003 --ent_coef 0.06 --num_envs 1 --num_steps 2048 --ppo_epochs 10 --num_minibatches 64 --gae_lambda 0.98 --intervals_count $INTERVALS --eval_episodes $EVAL_EPISODES --no_overwrite > /dev/null 2>&1 &

echo "Batch 3 (11-15) experiments triggered for 5M steps."
