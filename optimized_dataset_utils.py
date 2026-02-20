import os
import pickle
import torch
import numpy as np
from pathlib import Path

class SeaquestDatasetWriter:
    def __init__(self, save_dir, chunk_size=100000, env_name="seaquest"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.buffer = []
        self.chunk_idx = 0
        self.env_name = env_name
        self.total_steps = 0

    def add(self, obs, logic_obs, action, reward, next_obs, next_logic_obs, done):
        """
        Add a single transition.
        """
        def to_cpu(x, is_obs=False):
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            if is_obs and x is not None:
                return x.astype(np.uint8)
            return x

        transition = {
            "obs": to_cpu(obs, is_obs=True),
            "logic_obs": to_cpu(logic_obs),
            "action": to_cpu(action),
            "reward": to_cpu(reward),
            "next_obs": to_cpu(next_obs, is_obs=True),
            "next_logic_obs": to_cpu(next_logic_obs),
            "done": to_cpu(done)
        }
        self.buffer.append(transition)
        
        if len(self.buffer) >= self.chunk_size:
            self.flush()

    def batch_add(self, obs, logic_obs, action, reward, next_obs, next_logic_obs, done):
        """
        Add a batch of transitions.
        obs: (N, ...)
        """
        batch_size = len(obs)
        
        # Optimized batch conversion
        if isinstance(obs, torch.Tensor): obs = obs.detach().cpu().numpy()
        if isinstance(logic_obs, torch.Tensor): logic_obs = logic_obs.detach().cpu().numpy()
        if isinstance(action, torch.Tensor): action = action.detach().cpu().numpy()
        if isinstance(reward, torch.Tensor): reward = reward.detach().cpu().numpy()
        if isinstance(next_obs, torch.Tensor): next_obs = next_obs.detach().cpu().numpy()
        if isinstance(next_logic_obs, torch.Tensor): next_logic_obs = next_logic_obs.detach().cpu().numpy()
        if isinstance(done, torch.Tensor): done = done.detach().cpu().numpy()

        obs = obs.astype(np.uint8)
        next_obs = next_obs.astype(np.uint8)

        for i in range(batch_size):
            transition = {
                "obs": obs[i],
                "logic_obs": logic_obs[i],
                "action": action[i],
                "reward": reward[i],
                "next_obs": next_obs[i],
                "next_logic_obs": next_logic_obs[i],
                "done": done[i]
            }
            self.buffer.append(transition)
        
        if len(self.buffer) >= self.chunk_size:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        
        filename = self.save_dir / f"dataset_{self.env_name}_{self.chunk_idx:05d}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(self.buffer, f)
        
        self.total_steps += len(self.buffer)
        self.buffer = []
        self.chunk_idx += 1

    def close(self):
        self.flush()
