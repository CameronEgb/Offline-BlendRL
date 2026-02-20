import os
import pickle
import torch
import numpy as np
from pathlib import Path

class DatasetWriter:
    def __init__(self, save_dir, chunk_size=10000, env_name="env"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.buffer = []
        self.chunk_idx = 0
        self.env_name = env_name
        self.total_steps = 0

    def add(self, obs, logic_obs, action, reward, next_obs, next_logic_obs, done):
        """
        Add a transition.
        obs: tensor or array
        logic_obs: tensor or array (can be None)
        action: tensor or array
        reward: float or tensor
        next_obs: tensor or array
        next_logic_obs: tensor or array (can be None)
        done: bool or tensor
        """
        # Convert to cpu numpy/scalar for storage to save VRAM/RAM
        def to_cpu(x, is_obs=False):
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            if is_obs and x is not None:
                # Cast to uint8 for storage (0-255)
                return x.astype(np.uint8)
            return x

        transition = {
            "obs": to_cpu(obs, is_obs=True),
            "logic_obs": to_cpu(logic_obs) if logic_obs is not None else None,
            "action": to_cpu(action),
            "reward": to_cpu(reward),
            "next_obs": to_cpu(next_obs, is_obs=True),
            "next_logic_obs": to_cpu(next_logic_obs) if next_logic_obs is not None else None,
            "done": to_cpu(done)
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
        
        # print(f"Saved dataset chunk {self.chunk_idx} with {len(self.buffer)} transitions to {filename}")
        self.total_steps += len(self.buffer)
        self.buffer = []
        self.chunk_idx += 1

    def close(self):
        self.flush()

class DatasetReader:
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

        # Lazy loading? Or load all? 
        # For RL datasets, we usually need random access. 
        # If dataset is too large, we might need to stream or map.
        # Given "take inspiration from old version", likely it fits in RAM or we load a subset.
        # But 60M steps is huge. 
        # Let's implement a buffer-based shuffle loader or just load all if possible.
        # For now, let's load all into a big list. If it crashes, we optimize.
        # Actually, let's just implement a simple sampler that loads a few files at a time if needed.
        # But for IQL we need global sampling.
        # Let's assume we can load it all for now or the user provides a "large dataset path" which implies we might need optimization.
        # For now, I'll load all into memory.
        
        self.transitions = []
        # print(f"Loading {len(self.files)} dataset files...")
        for f in self.files:
            with open(f, "rb") as fh:
                data = pickle.load(fh)
                self.transitions.extend(data)
        
        # print(f"Loaded {len(self.transitions)} total transitions.")
        
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
        
        # Free memory
        del self.transitions
    
    def set_limit(self, limit):
        self.limit = min(limit, len(self.obs))
        print(f"Dataset limit set to {self.limit} transitions.")

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.limit, size=batch_size)
        
        batch = {
            "obs": torch.tensor(self.obs[idxs], device=self.device, dtype=torch.float32),
            "action": torch.tensor(self.actions[idxs], device=self.device, dtype=torch.long), # Assuming discrete actions for now
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
