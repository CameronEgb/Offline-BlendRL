"""Unit tests for the Component Plugin system, protocols, and model registry."""

from __future__ import annotations

import unittest.mock as mock
import pytest
import torch
import torch.nn as nn

from src.app.core.model_registry import (
    MODEL_REGISTRY,
    build_model,
    get_model_class,
    register_model,
)
from src.app.core.protocols import (
    DynamicTopologyProtocol,
    ExtraStateProtocol,
    HasModelCallbacks,
    walk_model_modules,
)
from src.usr.models.cew.cew_callback import CEWSelfOrganizationCallback
from src.usr.models.cew.cew_model import CEWModel
from src.usr.models.blendrl.agents.blender_agent import BlenderActorCritic


class DummyDynamicModel(nn.Module, DynamicTopologyProtocol, ExtraStateProtocol, HasModelCallbacks):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)
        self._changed = False

    def forward(self, x):
        return self.linear(x)

    def has_topology_changed(self) -> bool:
        return self._changed

    def reset_topology_changed(self) -> None:
        self._changed = False

    def clone_topology_to(self, target: nn.Module) -> None:
        target.load_state_dict(self.state_dict())

    def get_callbacks(self) -> list:
        return ["dummy_callback"]

    def extra_state(self) -> dict:
        return {"dummy_key": 42}

    def load_extra_state(self, state: dict) -> None:
        self.dummy_val = state.get("dummy_key")


def test_register_and_build_custom_model():
    @register_model("test_dummy_plugin")
    def _create_dummy(obs_dim=4, n_actions=2, **kwargs):
        return nn.Linear(obs_dim, n_actions)

    assert "test_dummy_plugin" in MODEL_REGISTRY
    m = build_model("test_dummy_plugin", obs_dim=8, n_actions=3)
    assert isinstance(m, nn.Linear)
    assert m.in_features == 8
    assert m.out_features == 3


def test_registered_standard_neural_models():
    from src.usr.models.neural.architectures import CNNActor, MLPQNetwork, NeuralBlenderMLP

    mlp = build_model("mlp", obs_dim=6, n_actions=3)
    assert isinstance(mlp, MLPQNetwork)
    assert mlp.n_actions == 3
    assert mlp.num_in_features == 6

    cnn = build_model("cnn", n_actions=4)
    assert isinstance(cnn, CNNActor)

    blender_mlp = build_model("neural_blender_mlp", obs_dim=8, n_actions=2)
    assert isinstance(blender_mlp, NeuralBlenderMLP)

    from src.usr.models.neural.resnet import DuelingResNetMLP
    resnet = build_model("dueling_resnet", obs_dim=46, n_actions=2)
    assert isinstance(resnet, DuelingResNetMLP)

    from src.usr.models.neural.transformer import SepsisTransformerPolicy, CrossAttentionSepsisPolicy
    transformer = build_model("transformer", obs_dim=46, n_actions=2)
    assert isinstance(transformer, SepsisTransformerPolicy)

    cross_attn = build_model("cross_attention", obs_dim=46, n_actions=2)
    assert isinstance(cross_attn, CrossAttentionSepsisPolicy)


def test_factories_get_neural_agent_delegates_to_build_model():
    from src.app.core.factories import get_neural_agent
    from src.usr.models.neural.architectures import MLPQNetwork

    agent_model = get_neural_agent("cartpole", n_actions=2, device="cpu", arch_name="mlp", num_in_features=4)
    assert isinstance(agent_model, MLPQNetwork)
    assert agent_model.n_actions == 2



def test_cew_implements_protocols():
    cew = CEWModel(n_inputs=4, n_actions=2)

    assert isinstance(cew, DynamicTopologyProtocol)
    assert isinstance(cew, ExtraStateProtocol)
    assert isinstance(cew, HasModelCallbacks)

    # Initial state
    assert not cew.has_topology_changed()
    cew._request_optimizer_rebind = True
    assert cew.has_topology_changed()
    cew.reset_topology_changed()
    assert not cew.has_topology_changed()

    # Callbacks
    cbs = cew.get_callbacks()
    assert len(cbs) == 1
    assert isinstance(cbs[0], CEWSelfOrganizationCallback)

    # Extra state
    state = cew.extra_state()
    assert "rules" in state
    assert "antecedents" in state


def test_walk_model_modules_standalone():
    dummy = DummyDynamicModel()
    modules = walk_model_modules(dummy)
    assert dummy in modules


class DummyComposite:
    def __init__(self, policy_modules):
        self.policy_modules = policy_modules


class DummyAgent:
    def __init__(self, model=None, q_model=None, target_q_model=None):
        self.model = model
        self.q_model = q_model
        self.target_q_model = target_q_model
        self.rebind_called = False

    def _rebind_optimizer(self):
        self.rebind_called = True


def test_walk_model_modules_composite():
    cew = CEWModel(n_inputs=4, n_actions=2)
    dummy_blender = DummyComposite(policy_modules=[cew])
    dummy_agent = DummyAgent(model=dummy_blender)

    discovered = walk_model_modules(dummy_agent)
    assert cew in discovered


def test_base_agent_callback_discovery_without_hardcoding():
    from src.usr.methods.base_agent import OfflineAgentBase

    cew = CEWModel(n_inputs=4, n_actions=2)
    dummy_agent = DummyAgent(q_model=cew)

    callbacks = OfflineAgentBase.configure_callbacks(dummy_agent)
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], CEWSelfOrganizationCallback)


def test_base_agent_dynamic_rebind_detection():
    from src.usr.methods.base_agent import OfflineAgentBase

    dynamic_model = DummyDynamicModel()
    target_model = DummyDynamicModel()
    dummy_agent = DummyAgent(q_model=dynamic_model, target_q_model=target_model)

    # Simulate topology change
    dynamic_model._changed = True
    assert dynamic_model.has_topology_changed()

    OfflineAgentBase._handle_optimizer_rebind(dummy_agent)

    # Verified that flag was reset and _rebind_optimizer was called
    assert not dynamic_model.has_topology_changed()
    assert dummy_agent.rebind_called


def test_standard_gymnasium_vector_env_resolution():
    from src.app.core.env_vectorized import VectorizedBaseEnv, StandardGymVectorEnv

    # Resolves any standard Gymnasium environment without needing an in/envs/ folder
    env = VectorizedBaseEnv.from_name("CartPole-v1", n_envs=2)
    assert isinstance(env, StandardGymVectorEnv)
    assert env.n_actions() == 2

    obs = env.reset()
    assert obs.shape == (2, 4)

    next_obs, rewards, term, trunc, infos = env.step(torch.tensor([0, 1]))
    assert next_obs.shape == (2, 4)
    assert len(rewards) == 2
    env.close()

