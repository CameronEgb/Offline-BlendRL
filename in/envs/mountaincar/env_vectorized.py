from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import torch as th

from src.core.env_vectorized import VectorizedBaseEnv


class MountainCarShapingWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return obs, reward, done, truncated, info


class VectorizedNudgeEnv(VectorizedBaseEnv):
    name = "mountaincar"
    pred2action = {
        "left": 0,
        "noop": 1,
        "right": 2,
    }
    pred_names: Sequence

    def __init__(
        self,
        mode: str = "ppo",
        n_envs: int = 1,
        render_mode="rgb_array",
        seed=None,
        use_shaping=False,
        **kwargs,
    ):
        super().__init__(mode)
        self.n_envs = n_envs
        self.seed = seed

        self.envs = []
        for _ in range(n_envs):
            env = gym.make("MountainCar-v0", render_mode=render_mode)
            if use_shaping:
                env = MountainCarShapingWrapper(env)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.Autoreset(env)
            self.envs.append(env)

        self.n_actions = 3
        self.n_raw_actions = 3
        self.n_features = 2

    def reset(self, seed=None):
        seed_i = seed if seed is not None else self.seed
        obs_list = []
        for env in self.envs:
            obs, info = env.reset(seed=seed_i)
            obs_list.append(th.tensor(obs, dtype=th.float32))
            if seed_i is not None:
                seed_i += 1
        return th.stack(obs_list)

    def step(self, actions, is_mapped: bool = False):
        rewards = []
        truncations = []
        dones = []
        infos = []
        obs_list = []
        for i, env in enumerate(self.envs):
            action = actions[i]
            if hasattr(action, "item"):
                action = action.item()
            obs, reward, done, truncation, info = env.step(action)
            obs_list.append(th.tensor(obs, dtype=th.float32))
            rewards.append(reward)
            truncations.append(truncation)
            dones.append(done)
            infos.append(info)
        return (
            th.stack(obs_list),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            np.array(truncations, dtype=bool),
            infos,
        )

    def get_action_meanings(self):
        return ["left", "noop", "right"]

    def close(self):
        for env in self.envs:
            env.close()


VectorizedEnv = VectorizedNudgeEnv
