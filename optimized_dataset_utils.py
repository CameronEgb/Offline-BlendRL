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

class SeaquestDatasetReader:
    def __init__(self, dataset_dirs, device="cpu"):
        """
        dataset_dirs: list of directories containing dataset chunks
        """
        self.device = device
        self.files = []
        if isinstance(dataset_dirs, (str, Path)):
            dataset_dirs = [dataset_dirs]
            
        for d in dataset_dirs:
            p = Path(d)
            if p.exists():
                self.files.extend(sorted(list(p.glob("*.pkl"))))
        
        if not self.files:
            print(f"Warning: No dataset files found in {dataset_dirs}")

        self.transitions = []
        for f in self.files:
            with open(f, "rb") as fh:
                data = pickle.load(fh)
                self.transitions.extend(data)
        
        # Convert to structure of arrays for faster sampling
        self.obs = []
        self.logic_obs = []
        self.actions = []
        self.rewards = []
        self.next_obs = []
        self.next_logic_obs = []
        self.dones = []
        
        has_logic = False
        if len(self.transitions) > 0 and self.transitions[0]["logic_obs"] is not None:
            has_logic = True

        for t in self.transitions:
            self.obs.append(t["obs"])
            if has_logic:
                self.logic_obs.append(t["logic_obs"])
            self.actions.append(t["action"])
            self.rewards.append(t["reward"])
            self.next_obs.append(t["next_obs"])
            if has_logic:
                self.next_logic_obs.append(t["next_logic_obs"])
            self.dones.append(t["done"])

        self.obs = np.array(self.obs)
        if has_logic:
            self.logic_obs = np.array(self.logic_obs)
        else:
            self.logic_obs = None
        self.actions = np.array(self.actions)
        self.rewards = np.array(self.rewards)
        self.next_obs = np.array(self.next_obs)
        if has_logic:
            self.next_logic_obs = np.array(self.next_logic_obs)
        else:
            self.next_logic_obs = None
        self.dones = np.array(self.dones)
        
        self.limit = len(self.obs)
        del self.transitions
    
    def set_limit(self, limit):
        self.limit = min(limit, len(self.obs))
        print(f"Dataset limit set to {self.limit} transitions.")

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.limit, size=batch_size)
        
        batch = {
            "obs": torch.tensor(self.obs[idxs], device=self.device, dtype=torch.float32),
            "action": torch.tensor(self.actions[idxs], device=self.device, dtype=torch.long),
            "reward": torch.tensor(self.rewards[idxs], device=self.device, dtype=torch.float32),
            "next_obs": torch.tensor(self.next_obs[idxs], device=self.device, dtype=torch.float32),
            "done": torch.tensor(self.dones[idxs], device=self.device, dtype=torch.float32)
        }
        
        if self.logic_obs is not None:
            batch["logic_obs"] = torch.tensor(self.logic_obs[idxs], device=self.device, dtype=torch.float32)
            batch["next_logic_obs"] = torch.tensor(self.next_logic_obs[idxs], device=self.device, dtype=torch.float32)
        else:
            batch["logic_obs"] = None
            batch["next_logic_obs"] = None
            
        return batch

    def __len__(self):
        return len(self.obs)
