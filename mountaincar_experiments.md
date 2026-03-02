# MountainCar PPO Experiment Log (No Reward Shaping)

## Summary of Batches 5-10
All experiments in this series were conducted with **zero reward shaping** (Sparse Reward: -1 per step). The goal was to reach the flag at position **0.5**.

| Batch ID | Envs | Steps | LR | Ent Coef | Minibatches | Epochs | Result (Avg Max Pos) |
|----------|------|-------|----|----------|-------------|--------|----------------------|
| Batch 5  | 1    | 4096  | 3e-4 | 0.02     | 64          | 10     | -0.35                |
| Batch 6  | 1    | 2048  | 2e-4 | 0.05     | 32          | 10     | -0.36                |
| Batch 7  | 1    | 2048  | 3e-4 | 0.02     | 64          | 10     | **-0.26 (Best)**     |
| Batch 8  | 1    | 4096  | 3e-4 | 0.02     | 64          | 10     | -0.35                |
| Batch 9  | 1    | 2048  | 1e-4 | 0.04     | 32          | 10     | -0.36                |
| Batch 10 | 1    | 2048  | 2e-4 | 0.02     | 32          | 20     | -0.29                |

### Key Observations:
1. **The Signal:** Batch 7 was the most promising, reaching an average maximum position of **-0.26**. The starting position is approximately **-0.4 to -0.6**.
2. **Exploration:** While the agent is learning to swing, it is not yet reaching the threshold of the hill. 
3. **Environment Count:** A single environment (`num_envs=1`) prevents "wasting" millions of steps on independent failures before a signal is found, but may benefit from a small increase (e.g., 2-4) to provide diverse gradient updates once a signal *is* found.
4. **Learning Rate:** 3e-4 seems more effective than lower rates for this sparse task.

## Strategy for Batch 11-15
* Increase **Total Timesteps** to 5M (1M is very low for sparse MC).
* Test higher **Entropy** (up to 0.1).
* Maintain **Long Rollouts** (2048) and **Small Minibatches** (64) for stability.
