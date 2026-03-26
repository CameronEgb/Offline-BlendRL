import torch as th
from nsfr.utils.common import bool_to_probs

def moving_right(agent: th.Tensor) -> th.Tensor:
    # agent: [pos, vel]
    vel = agent[..., 1]
    result = vel > 1e-4
    return bool_to_probs(result)

def moving_left(agent: th.Tensor) -> th.Tensor:
    vel = agent[..., 1]
    result = vel < -1e-4
    return bool_to_probs(result)

def moving(env_obj: th.Tensor) -> th.Tensor:
    # env_obj is same as agent in my extract_logic_state
    vel = env_obj[..., 1]
    result = th.abs(vel) > 1e-4
    return bool_to_probs(result)

def not_moving(env_obj: th.Tensor) -> th.Tensor:
    vel = env_obj[..., 1]
    result = th.abs(vel) <= 1e-4
    return bool_to_probs(result)
