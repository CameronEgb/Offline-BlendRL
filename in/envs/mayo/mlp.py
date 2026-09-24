import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class StandardMLP(nn.Module):
    """
    Standard Feedforward Multi-Layer Perceptron (Basic DNN baseline) for Mayo sepsis policy.
    """
    def __init__(
        self,
        device=None,
        hidden_sizes=(256, 256),
        has_softmax=False,
        has_sigmoid=False,
        out_size=4,
        logic=False,
        num_in_features=None,
        **kwargs,
    ):
        super().__init__()
        self.device = device if device is not None else torch.device("cpu")
        self.logic = logic
        if num_in_features is None:
            raise ValueError("StandardMLP requires 'num_in_features' matching the observation space dimension.")
        self.num_in_features = int(num_in_features)
        self.out_size = out_size

        if hidden_sizes is None or len(hidden_sizes) == 0:
            hidden_sizes = [256, 256]
        else:
            hidden_sizes = list(hidden_sizes)

        layers = []
        last_dim = self.num_in_features
        for h in hidden_sizes:
            layers.append(layer_init(nn.Linear(last_dim, h)))
            layers.append(nn.ReLU())
            last_dim = h
        self.network = nn.Sequential(*layers)
        self.actor = layer_init(nn.Linear(last_dim, out_size), std=0.01)
        self.critic = layer_init(nn.Linear(last_dim, 1), std=1.0)
        self.softmax = nn.Softmax(dim=-1) if has_softmax else nn.Identity()

        if self.device is not None:
            self.to(self.device)

    def _flat(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.float().reshape(x.shape[0], -1)
        if flat.shape[-1] < self.num_in_features:
            pad = torch.zeros((flat.shape[0], self.num_in_features - flat.shape[-1]), dtype=flat.dtype, device=flat.device)
            flat = torch.cat([flat, pad], dim=-1)
        elif flat.shape[-1] > self.num_in_features:
            flat = flat[:, :self.num_in_features]
        return flat

    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        flat = self._flat(x)
        hidden = self.network(flat)
        return self.actor(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.get_q_values(x)
        return self.softmax(q)

    def get_action_probs(self, x: torch.Tensor) -> torch.Tensor:
        q = self.get_q_values(x)
        return torch.softmax(q, dim=-1)

    def get_value(self, x: torch.Tensor, logic_state=None) -> torch.Tensor:
        flat = self._flat(x)
        hidden = self.network(flat)
        return self.critic(hidden)

    def get_action_and_value(self, x: torch.Tensor, action=None):
        flat = self._flat(x)
        hidden = self.network(flat)
        q = self.actor(hidden)
        v = self.critic(hidden)
        probs = torch.softmax(q, dim=-1)
        dist = Categorical(probs=probs)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), v

    def act(self, x: torch.Tensor, logic_state=None, epsilon: float = 0.0):
        probs = self.get_action_probs(x)
        dist = Categorical(probs=probs)
        action = dist.sample()
        return action, dist.log_prob(action)

    def _print(self) -> str:
        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"Mayo Standard MLP Agent - In: {self.num_in_features}, Out: {self.out_size}, Trainable Params: {param_count:,}"


MLP = StandardMLP
