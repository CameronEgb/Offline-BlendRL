#!/bin/bash

# Define common settings
ENVIRONMENT="mountaincar"
STEPS=1000000
INTERVALS=7
EVAL_EPISODES=100

# 6. mountaincar_batch_6: The Signal Seeker
echo "Launching MC_B6: Signal Seeker (LR 2e-4, Ent 0.05, GAE 0.98)"
printf "y\ny\ny\n" | python3.11 run_full_cycle.py --experimentid mountaincar_batch_6 --environment $ENVIRONMENT --online_methods ppo --online_steps $STEPS --offline_methods "" --local --lr 0.0002 --ent_coef 0.05 --num_envs 1 --num_steps 2048 --ppo_epochs 10 --num_minibatches 32 --gae_lambda 0.98 &

# 7. mountaincar_batch_7: High Frequency
echo "Launching MC_B7: High Frequency (Minibatch 32)"
printf "y\ny\ny\n" | python3.11 run_full_cycle.py --experimentid mountaincar_batch_7 --environment $ENVIRONMENT --online_methods ppo --online_steps $STEPS --offline_methods "" --local --lr 0.0003 --ent_coef 0.02 --num_envs 1 --num_steps 2048 --ppo_epochs 10 --num_minibatches 64 &

# 8. mountaincar_batch_8: Long Horizon
echo "Launching MC_B8: Long Horizon (Steps 4096, GAE 0.99)"
printf "y\ny\ny\n" | python3.11 run_full_cycle.py --experimentid mountaincar_batch_8 --environment $ENVIRONMENT --online_methods ppo --online_steps $STEPS --offline_methods "" --local --lr 0.0003 --ent_coef 0.02 --num_envs 1 --num_steps 4096 --ppo_epochs 10 --num_minibatches 64 --gae_lambda 0.99 &

# 9. mountaincar_batch_9: Stable Explorer
echo "Launching MC_B9: Stable Explorer (LR 1e-4, Ent 0.04)"
printf "y\ny\ny\n" | python3.11 run_full_cycle.py --experimentid mountaincar_batch_9 --environment $ENVIRONMENT --online_methods ppo --online_steps $STEPS --offline_methods "" --local --lr 0.0001 --ent_coef 0.04 --num_envs 1 --num_steps 2048 --ppo_epochs 10 --num_minibatches 32 &

# 10. mountaincar_batch_10: Update Heavy
echo "Launching MC_B10: Update Heavy (Epochs 20)"
printf "y\ny\ny\n" | python3.11 run_full_cycle.py --experimentid mountaincar_batch_10 --environment $ENVIRONMENT --online_methods ppo --online_steps $STEPS --offline_methods "" --local --lr 0.0002 --ent_coef 0.02 --num_envs 1 --num_steps 2048 --ppo_epochs 20 --num_minibatches 32 &

echo "Batch 2 (6-10) experiments triggered with refined params."
