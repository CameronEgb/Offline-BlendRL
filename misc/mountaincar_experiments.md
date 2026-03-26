# MountainCar PPO Experiment Log (No Reward Shaping)

## Strategy for Batch 16-20 (Replication & Simplification)
This batch mirrors the successful configurations of Batches 11-15 to get online training returns. It also introduces simplified logging (no double-tracking of rewards).

| Batch ID | Mirroring | Key Strategy |
|----------|-----------|--------------|
| Batch 16 | Batch 11  | High Exploration (Ent 0.1) |
| Batch 17 | Batch 12  | Multi-Env (4 Envs) Exploration |
| Batch 18 | Batch 13  | Stable Swing (2 Envs) |
| Batch 19 | Batch 14  | High Frequency (4096 steps) |
| Batch 20 | Batch 15  | Refined Signal Seeking |

### Improvements in Batch 4+:
* **Simplified Plotting:** Reward tracking has been consolidated to focus only on the training reward the agent actually sees.
* **Vectorized Logging Fix:** MountainCar episode completions are now correctly detected and logged to `training_log.pkl`.

## Summary of Batches 11-15 (SOLVED)
All experiments in this series successfully reached the flag at position **0.5**. This marks the first successful unshaped PPO run for MountainCar in this workspace.

| Batch ID | Envs | Rollout Steps | LR | Ent Coef | Minibatches | Epochs | Status | Best Avg Max Pos |
|----------|------|---------------|----|----------|-------------|--------|--------|------------------|
| Batch 11 | 1    | 2048          | 3e-4 | 0.10     | 64          | 10     | SOLVED | 0.516            |
| Batch 12 | 4    | 1024          | 3e-4 | 0.08     | 64          | 10     | SOLVED | 0.518            |
| Batch 13 | 2    | 2048          | 2e-4 | 0.05     | 64          | 10     | SOLVED | 0.517            |
| Batch 14 | 1    | 4096          | 3e-4 | 0.05     | 64          | 20     | SOLVED | 0.516            |
| Batch 15 | 1    | 2048          | 3e-4 | 0.06     | 64          | 10     | SOLVED | 0.519            |

### Key Observations (Success Phase):
1. **The Signal:** Increasing **Total Timesteps** to 5M was critical. The agents required more than 1.5M steps to consistently discover and exploit the sparse goal signal.
2. **Exploration:** High **Entropy Coefficients** (0.05 to 0.1) prevented premature convergence to "staying at the bottom" and pushed the agent to swing higher.
3. **Rollout Depth:** Maintaining long rollouts (1024-4096 steps per environment) ensured the agent could experience nearly the entire episode (max 200 steps) multiple times within a single update cycle, providing a stable gradient.
4. **Environment Scaling:** While high environment counts (e.g. 20+) were previously ineffective, the 4-environment setup in Batch 12 was highly successful, indicating that a small degree of parallelization is beneficial once the rollout depth is sufficient.

## Summary of Batches 5-10 (Failed)
Experiments in this series failed to reach the flag.

| Batch ID | Envs | Steps | LR | Ent Coef | Minibatches | Epochs | Result (Avg Max Pos) |
|----------|------|-------|----|----------|-------------|--------|----------------------|
| Batch 5  | 1    | 4096  | 3e-4 | 0.02     | 64          | 10     | -0.35                |
| Batch 6  | 1    | 2048  | 2e-4 | 0.05     | 32          | 10     | -0.36                |
| Batch 7  | 1    | 2048  | 3e-4 | 0.02     | 64          | 10     | -0.26                |
| Batch 8  | 1    | 4096  | 3e-4 | 0.02     | 64          | 10     | -0.35                |
| Batch 9  | 1    | 2048  | 1e-4 | 0.04     | 32          | 10     | -0.36                |
| Batch 10 | 1    | 2048  | 2e-4 | 0.02     | 32          | 20     | -0.29                |
