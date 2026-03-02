#!/bin/bash

# Define common settings
ENVIRONMENT="mountaincar"
STEPS=5000000 # 5M steps for sparse learning
INTERVALS=10
EVAL_EPISODES=100

# 16. mountaincar_batch_16: High Exploration Baseline
echo "Launching MC_B16: High Exploration (Ent 0.1, Envs 1, Steps 2048)"
mkdir -p logs/mountaincar_batch_16
nohup python3.11 run_full_cycle.py --experimentid mountaincar_batch_16 --environment $ENVIRONMENT --online_methods ppo --online_steps $STEPS --offline_methods "" --local --lr 0.0003 --ent_coef 0.1 --num_envs 1 --num_steps 2048 --ppo_epochs 10 --num_minibatches 64 --intervals_count $INTERVALS --eval_episodes $EVAL_EPISODES --no_overwrite > /dev/null 2>&1 &

# 17. mountaincar_batch_17: Multi-Env Sparse Exploration
echo "Launching MC_B17: Multi-Env Exploration (Ent 0.08, Envs 4, Steps 1024)"
mkdir -p logs/mountaincar_batch_17
nohup python3.11 run_full_cycle.py --experimentid mountaincar_batch_17 --environment $ENVIRONMENT --online_methods ppo --online_steps $STEPS --offline_methods "" --local --lr 0.0003 --ent_coef 0.08 --num_envs 4 --num_steps 1024 --ppo_epochs 10 --num_minibatches 64 --intervals_count $INTERVALS --eval_episodes $EVAL_EPISODES --no_overwrite > /dev/null 2>&1 &

# 18. mountaincar_batch_18: Stable Swing
echo "Launching MC_B18: Stable Swing (LR 2e-4, Ent 0.05, Envs 2, Steps 2048)"
mkdir -p logs/mountaincar_batch_18
nohup python3.11 run_full_cycle.py --experimentid mountaincar_batch_18 --environment $ENVIRONMENT --online_methods ppo --online_steps $STEPS --offline_methods "" --local --lr 0.0002 --ent_coef 0.05 --num_envs 2 --num_steps 2048 --ppo_epochs 10 --num_minibatches 64 --intervals_count $INTERVALS --eval_episodes $EVAL_EPISODES --no_overwrite > /dev/null 2>&1 &

# 19. mountaincar_batch_19: Learning Frequency
echo "Launching MC_B19: High Frequency (Ent 0.05, Envs 1, Steps 4096, Epochs 20)"
mkdir -p logs/mountaincar_batch_19
nohup python3.11 run_full_cycle.py --experimentid mountaincar_batch_19 --environment $ENVIRONMENT --online_methods ppo --online_steps $STEPS --offline_methods "" --local --lr 0.0003 --ent_coef 0.05 --num_envs 1 --num_steps 4096 --ppo_epochs 20 --num_minibatches 64 --intervals_count $INTERVALS --eval_episodes $EVAL_EPISODES --no_overwrite > /dev/null 2>&1 &

# 20. mountaincar_batch_20: Signal Seeker Refined
echo "Launching MC_B20: Signal Seeker Refined (GAE 0.98, Ent 0.06, Envs 1, Steps 2048)"
mkdir -p logs/mountaincar_batch_20
nohup python3.11 run_full_cycle.py --experimentid mountaincar_batch_20 --environment $ENVIRONMENT --online_methods ppo --online_steps $STEPS --offline_methods "" --local --lr 0.0003 --ent_coef 0.06 --num_envs 1 --num_steps 2048 --ppo_epochs 10 --num_minibatches 64 --gae_lambda 0.98 --intervals_count $INTERVALS --eval_episodes $EVAL_EPISODES --no_overwrite > /dev/null 2>&1 &

echo "Batch 4 (16-20) experiments triggered for 5M steps."
