import torch as th
import gymnasium as gym
from nudge.env import NudgeBaseEnv
from typing import Sequence
import numpy as np

class ReverseMountainCarWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.goal_position = -0.5
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Force position to top of hill (0.5)
        self.env.unwrapped.state = np.array([0.5, 0.0])
        return self.env.unwrapped.state.astype(np.float32), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        # Custom termination: reach bottom of hill (-0.5)
        # Wait, the hill is a parabola: y = sin(3*x). x = -0.5 is the bottom.
        # Original code uses: x >= 0.5.
        # We want x <= -0.5.
        position = obs[0]
        if position <= self.goal_position:
            reward = 0.0
            done = True
        else:
            reward = -1.0
            done = False
            
        return obs, reward, done, truncated, info

class NudgeEnv(NudgeBaseEnv):
    name = "reverseMC"
    pred2action = {
        "left": 0,
        "noop": 1,
        "right": 2,
    }
    pred_names: Sequence

    def __init__(
        self, mode: str, render_mode="rgb_array", seed=None, **kwargs
    ):
        super().__init__(mode)
        self.env = gym.make("MountainCar-v0", render_mode=render_mode)
        self.env = ReverseMountainCarWrapper(self.env)
        self.env = gym.wrappers.RecordEpisodeStatistics(self.env)
        self.env = gym.wrappers.Autoreset(self.env)
        
        self.n_actions = 3
        self.n_raw_actions = 3
        self.n_objects = 2 
        self.n_features = 2 
        self.seed = seed

    def reset(self):
        obs, info = self.env.reset(seed=self.seed)
        logic_state, neural_state = self.extract_logic_state(obs), self.extract_neural_state(obs)
        logic_state = logic_state.unsqueeze(0)
        return logic_state, neural_state

    def step(self, action, is_mapped: bool = False):
        if hasattr(action, "item"):
            action = action.item()
        obs, reward, done, truncation, info = self.env.step(action)
        logic_state, neural_state = self.extract_logic_state(obs), self.extract_neural_state(obs)
        logic_state = logic_state.unsqueeze(0)
        return (logic_state, neural_state), reward, done, truncation, info

    def extract_logic_state(self, obs):
        state = th.zeros((self.n_objects, self.n_features), dtype=th.float32)
        state[0] = th.tensor(obs, dtype=th.float32)
        state[1] = th.tensor(obs, dtype=th.float32)
        return state

    def extract_neural_state(self, obs):
        logic_state = self.extract_logic_state(obs)
        return logic_state.view(-1).unsqueeze(0)

    def get_action_meanings(self):
        return ["left", "noop", "right"]

    def close(self):
        self.env.close()
