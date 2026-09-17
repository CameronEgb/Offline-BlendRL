"""Standard Atari wrapper stack for Gym/Gymnasium environments."""
import gymnasium as gym
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


def make_atari_env(env, clip_rewards=False):
    """Standard Atari wrapper stack.
    
    RecordEpisodeStatistics is placed at the top so it measures agent steps
    rather than raw frames.
    """
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)

    if clip_rewards:
        env = ClipRewardEnv(env)

    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env, 4)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


# Alias for backwards compatibility
make_env = make_atari_env
