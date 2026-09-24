import os
from pathlib import Path
import numpy as np
import torch as th

from src.app.core.env_vectorized import VectorizedBaseEnv


class VectorizedNudgeEnv(VectorizedBaseEnv):
    name = "cchs"
    pred2action = {
        "withhold": 0,
        "oxygen": 1,
        "antibiotic": 2,
        "vasopressor": 3,
    }

    def __init__(
        self,
        mode: str = "eval",
        n_envs: int = 1,
        seed=None,
        dataset_name=None,
        **kwargs,
    ):
        super().__init__(mode)
        self.n_envs = n_envs
        self.seed = seed if seed is not None else 42

        # Locate dataset archive
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        npz_path = project_root / "in/datasets/cchs/cchs.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"CCHS dataset archive not found at {npz_path}")

        data = np.load(npz_path, allow_pickle=True)
        self.states = data["states"]        # (total_steps, 14)
        self.actions = data["actions"]      # (total_steps,)
        self.rewards = data["rewards"]      # (total_steps,)
        self.dones = data["dones"]          # (total_steps,)
        self.traj_ptrs = data["traj_ptrs"]  # (n_patients + 1,)
        self.n_patients = len(self.traj_ptrs) - 1

        self.n_actions = 4
        self.n_raw_actions = 4
        self.n_features = self.states.shape[1]

        self.rng = np.random.default_rng(self.seed)
        self.current_traj_idx = np.zeros(n_envs, dtype=np.int32)
        self.current_step_idx = np.zeros(n_envs, dtype=np.int32)

        for i in range(n_envs):
            self._reset_env_slot(i)

    def _reset_env_slot(self, env_idx: int):
        """Randomly select a patient trajectory and reset step counter."""
        idx = self.rng.integers(0, self.n_patients)
        self.current_traj_idx[env_idx] = idx
        self.current_step_idx[env_idx] = 0

    def reset(self, seed=None):
        obs_list = []
        for i in range(self.n_envs):
            self._reset_env_slot(i)
            traj = self.current_traj_idx[i]
            ptr = self.traj_ptrs[traj]
            obs = self.states[ptr]
            obs_list.append(th.tensor(obs, dtype=th.float32))
        return th.stack(obs_list)

    def step(self, actions, is_mapped: bool = False):
        obs_list = []
        rewards = []
        terminations = []
        truncations = []
        infos = []

        for i in range(self.n_envs):
            traj = self.current_traj_idx[i]
            step_idx = self.current_step_idx[i]
            ptr_start = self.traj_ptrs[traj]
            ptr_end = self.traj_ptrs[traj + 1]
            traj_len = ptr_end - ptr_start

            abs_step = ptr_start + step_idx
            is_done = (step_idx == traj_len - 1)
            reward = float(self.rewards[abs_step])

            if is_done:
                terminated = True
                self._reset_env_slot(i)
                new_traj = self.current_traj_idx[i]
                next_obs = self.states[self.traj_ptrs[new_traj]]
            else:
                terminated = False
                self.current_step_idx[i] += 1
                next_obs = self.states[ptr_start + self.current_step_idx[i]]

            obs_list.append(th.tensor(next_obs, dtype=th.float32))
            rewards.append(reward)
            terminations.append(terminated)
            truncations.append(False)
            infos.append({})

        return (
            th.stack(obs_list),
            np.array(rewards, dtype=np.float32),
            np.array(terminations, dtype=bool),
            np.array(truncations, dtype=bool),
            infos,
        )

    def get_action_meanings(self):
        return ["withhold", "oxygen", "antibiotic", "vasopressor"]

    def close(self):
        pass


VectorizedEnv = VectorizedNudgeEnv
