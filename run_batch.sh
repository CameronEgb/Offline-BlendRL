#!/bin/bash

# Define common settings
ENVIRONMENT="mountaincar"
STEPS=1000000
INTERVALS=7
EVAL_EPISODES=100

# 1. mountaincar_batch_1: SB3 Baseline
echo "Launching mountaincar_batch_1: SB3 Baseline (ent=0.02, envs=1, steps=2048)"
printf "y\ny\ny\n" | python3.11 run_full_cycle.py --experimentid mountaincar_batch_1 --environment $ENVIRONMENT --online_methods ppo --online_steps $STEPS --offline_methods "" --local --lr 0.0003 --ent_coef 0.02 --num_envs 1 --num_steps 2048 --ppo_epochs 10 --num_minibatches 32 &

# 2. mountaincar_batch_2: High Entropy
echo "Launching mountaincar_batch_2: High Entropy (ent=0.1, envs=1, steps=2048)"
printf "y\ny\ny\n" | python3.11 run_full_cycle.py --experimentid mountaincar_batch_2 --environment $ENVIRONMENT --online_methods ppo --online_steps $STEPS --offline_methods "" --local --lr 0.0003 --ent_coef 0.1 --num_envs 1 --num_steps 2048 --ppo_epochs 10 --num_minibatches 32 &

# 3. mountaincar_batch_3: Parallel exploration
echo "Launching mountaincar_batch_3: Parallel Exploration (ent=0.05, envs=16, steps=128)"
printf "y\ny\ny\n" | python3.11 run_full_cycle.py --experimentid mountaincar_batch_3 --environment $ENVIRONMENT --online_methods ppo --online_steps $STEPS --offline_methods "" --local --lr 0.0003 --ent_coef 0.05 --num_envs 16 --num_steps 128 --ppo_epochs 10 --num_minibatches 16 &

# 4. mountaincar_batch_4: Smaller minibatches
echo "Launching mountaincar_batch_4: Small Minibatch (batch=32, envs=1, steps=2048)"
printf "y\ny\ny\n" | python3.11 run_full_cycle.py --experimentid mountaincar_batch_4 --environment $ENVIRONMENT --online_methods ppo --online_steps $STEPS --offline_methods "" --local --lr 0.0002 --ent_coef 0.02 --num_envs 1 --num_steps 2048 --ppo_epochs 10 --num_minibatches 64 &

# 5. mountaincar_batch_5: Long rollout
echo "Launching mountaincar_batch_5: Long Rollout (steps=4096, envs=1)"
printf "y\ny\ny\n" | python3.11 run_full_cycle.py --experimentid mountaincar_batch_5 --environment $ENVIRONMENT --online_methods ppo --online_steps $STEPS --offline_methods "" --local --lr 0.0003 --ent_coef 0.02 --num_envs 1 --num_steps 4096 --ppo_epochs 10 --num_minibatches 64 &

echo "All 5 background experiments triggered."
