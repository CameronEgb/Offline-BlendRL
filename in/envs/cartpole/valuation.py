import torch as th
from nsfr.utils.common import bool_to_probs

def leaning_left(agent: th.Tensor) -> th.Tensor:
    # agent: [cart_pos, cart_vel, pole_angle, pole_vel]
    angle = agent[..., 2]
    result = angle < -0.01
    return bool_to_probs(result)

def leaning_right(agent: th.Tensor) -> th.Tensor:
    angle = agent[..., 2]
    result = angle > 0.01
    return bool_to_probs(result)

def true(env_obj: th.Tensor) -> th.Tensor:
    # Always true for blender
    result = th.ones_like(env_obj[..., 0], dtype=th.bool)
    return bool_to_probs(result)
