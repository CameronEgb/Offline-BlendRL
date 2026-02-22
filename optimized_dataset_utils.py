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

    def batch_add(self, obs, logic_obs, action, reward, next_obs, next_logic_obs, done):
        """
        Add a batch of transitions.
        Optimized: Stores only the LATEST frame of next_obs to save space.
        Reconstruction happens in the Reader.
        """
        batch_size = len(obs)
        
        if isinstance(obs, torch.Tensor): obs = obs.detach().cpu().numpy()
        if isinstance(logic_obs, torch.Tensor): logic_obs = logic_obs.detach().cpu().numpy()
        if isinstance(action, torch.Tensor): action = action.detach().cpu().numpy()
        if isinstance(reward, torch.Tensor): reward = reward.detach().cpu().numpy()
        if isinstance(next_obs, torch.Tensor): next_obs = next_obs.detach().cpu().numpy()
        if isinstance(next_logic_obs, torch.Tensor): next_logic_obs = next_logic_obs.detach().cpu().numpy()
        if isinstance(done, torch.Tensor): done = done.detach().cpu().numpy()

        # obs is (N, 4, 84, 84), next_obs is (N, 4, 84, 84)
        # We only save the newest frame of next_obs: (N, 1, 84, 84)
        # This saves 3/8 of total observation storage.
        obs = obs.astype(np.uint8)
        next_obs_latest = next_obs[:, -1:, :, :].astype(np.uint8)

        for i in range(batch_size):
            transition = {
                "obs": obs[i],
                "logic_obs": logic_obs[i],
                "action": action[i],
                "reward": reward[i],
                "next_obs_new": next_obs_latest[i], # The "tricky" part
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

        # Loading into memory-efficient arrays
        # Note: For 20M steps, we might want memmap, but let's start with 
        # packed numpy arrays which are much smaller than a list of dicts.
        all_obs = []
        all_logic_obs = []
        all_actions = []
        all_rewards = []
        all_next_obs_new = []
        all_next_logic_obs = []
        all_dones = []

        print(f"Loading {len(self.files)} dataset chunks...")
        for f in self.files:
            with open(f, "rb") as fh:
                chunk = pickle.load(fh)
                for t in chunk:
                    all_obs.append(t["obs"])
                    # Check if logic data exists in this transition
                    all_logic_obs.append(t["logic_obs"] if t.get("logic_obs") is not None else None)
                    all_actions.append(t["action"])
                    all_rewards.append(t["reward"])
                    
                    # BACKWARD COMPATIBILITY CHECK
                    if "next_obs_new" in t:
                        all_next_obs_new.append(t["next_obs_new"])
                    else:
                        # Old format had the full 4-frame stack in 'next_obs'
                        # We take only the latest frame [1, 84, 84] to match our optimized format
                        all_next_obs_new.append(t["next_obs"][-1:])
                        
                    all_next_logic_obs.append(t["next_logic_obs"] if t.get("next_logic_obs") is not None else None)
                    all_dones.append(t["done"])

        self.obs = np.array(all_obs, dtype=np.uint8)
        
        # Determine if we have logic data
        self.has_logic = all_logic_obs[0] is not None if len(all_logic_obs) > 0 else False
        if self.has_logic:
            self.logic_obs = np.array(all_logic_obs, dtype=np.float32)
            self.next_logic_obs = np.array(all_next_logic_obs, dtype=np.float32)
        else:
            self.logic_obs = None
            self.next_logic_obs = None

        self.actions = np.array(all_actions, dtype=np.int64)
        self.rewards = np.array(all_rewards, dtype=np.float32)
        self.next_obs_new = np.array(all_next_obs_new, dtype=np.uint8)
        self.dones = np.array(all_dones, dtype=np.float32)
        
        self.limit = len(self.obs)
        print(f"Dataset loaded: {self.limit} transitions (logic data: {'YES' if self.has_logic else 'NO'}).")

    def set_limit(self, limit):
        self.limit = min(limit, len(self.obs))
        print(f"Dataset limit set to {self.limit} transitions.")

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.limit, size=batch_size)
        
        obs_batch = torch.tensor(self.obs[idxs], device=self.device, dtype=torch.float32)
        next_new_frame = torch.tensor(self.next_obs_new[idxs], device=self.device, dtype=torch.float32)
        
        # TRICKY RECONSTRUCTION:
        # next_obs = [obs[1], obs[2], obs[3], next_new_frame]
        next_obs_batch = torch.cat([obs_batch[:, 1:, :, :], next_new_frame], dim=1)
        
        batch = {
            "obs": obs_batch,
            "action": torch.tensor(self.actions[idxs], device=self.device, dtype=torch.long),
            "reward": torch.tensor(self.rewards[idxs], device=self.device, dtype=torch.float32),
            "next_obs": next_obs_batch,
            "done": torch.tensor(self.dones[idxs], device=self.device, dtype=torch.float32),
        }
        
        if self.has_logic:
            batch["logic_obs"] = torch.tensor(self.logic_obs[idxs], device=self.device, dtype=torch.float32)
            batch["next_logic_obs"] = torch.tensor(self.next_logic_obs[idxs], device=self.device, dtype=torch.float32)
        else:
            batch["logic_obs"] = None
            batch["next_logic_obs"] = None
            
        return batch

    def __len__(self):
        return self.limit
