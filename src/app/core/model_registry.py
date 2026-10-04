"""Model Registry and Factory — Single source of truth for policy and Q-network models.

Models self-register using the @register_model decorator:
    @register_model("cew")
    class CEWModel(nn.Module):
        ...

    @register_model("blendrl")
    class BlenderActorCritic(nn.Module):
        ...

The pipeline resolves and instantiates models via build_model():
    model = build_model(cfg, env=env, device=device)
"""

from __future__ import annotations

import importlib
import logging
import os
import pkgutil
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

MODEL_REGISTRY: dict[str, Any] = {}


def register_model(*names: str):
    """Decorator to register a model class or factory function under one or more names.

    Usage:
        @register_model("cew")
        class CEWModel(nn.Module):
            ...

        @register_model("blendrl", "blender")
        class BlenderActorCritic(nn.Module):
            ...
    """

    def decorator(cls_or_fn: Any):
        for name in names:
            MODEL_REGISTRY[name.lower()] = cls_or_fn
        return cls_or_fn

    return decorator


def auto_discover_models():
    """Discover and import all model modules in src/usr/models to trigger @register_model."""
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "usr", "models")
    if not os.path.exists(models_dir):
        return

    for item in os.listdir(models_dir):
        sub_path = os.path.join(models_dir, item)
        if os.path.isdir(sub_path) and not item.startswith("_") and not item.startswith("."):
            try:
                importlib.import_module(f"src.usr.models.{item}")
            except Exception as e:
                logger.debug("Could not auto-import src.usr.models.%s: %s", item, e)


def get_model_class(model_name: str) -> Any:
    """Resolve a model name to its registered class or factory."""
    if not MODEL_REGISTRY:
        auto_discover_models()

    clean_name = model_name.lower().strip()
    if clean_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[clean_name]

    # Prefix match if exact match not found
    matches = [
        (prefix, cls)
        for prefix, cls in MODEL_REGISTRY.items()
        if clean_name == prefix or clean_name.startswith(prefix + "_")
    ]
    if matches:
        return max(matches, key=lambda x: len(x[0]))[1]

    raise ValueError(
        f"Unknown model architecture: '{model_name}'. Registered models: {sorted(MODEL_REGISTRY.keys())}"
    )


def build_model(
    model_spec: Any,
    env: Any = None,
    device: Any = None,
    n_actions: int | None = None,
    obs_dim: int | None = None,
    **kwargs,
) -> nn.Module:
    """Factory to build any registered model or composite model from configuration.

    Args:
        model_spec: Model identifier string, DictConfig, or dict.
        env: Environment instance or environment name.
        device: Torch device.
        n_actions: Number of discrete actions (inferred from env if None).
        obs_dim: Observation input dimension (inferred from env if None).
        kwargs: Overriding parameters passed to model constructor.
    """
    if not MODEL_REGISTRY:
        auto_discover_models()

    if device is None:
        device = torch.device("cpu")

    # If already instantiated module, return it directly
    if isinstance(model_spec, nn.Module):
        return model_spec.to(device)

    # Extract model name and parameters from spec
    model_name = "mlp"
    model_params = {}

    if isinstance(model_spec, str):
        model_name = model_spec
    elif isinstance(model_spec, (dict, DictConfig)):
        cfg_dict = OmegaConf.to_container(model_spec, resolve=True) if isinstance(model_spec, DictConfig) else dict(model_spec)
        # Check standard config naming conventions
        model_name = (
            cfg_dict.get("name")
            or cfg_dict.get("architecture")
            or cfg_dict.get("type")
            or (list(cfg_dict.keys())[0] if len(cfg_dict) == 1 and isinstance(list(cfg_dict.values())[0], dict) else "mlp")
        )
        # If wrapped under model name (e.g. {blendrl: {...}})
        if model_name in cfg_dict and isinstance(cfg_dict[model_name], dict):
            model_params = cfg_dict[model_name]
        else:
            model_params = {k: v for k, v in cfg_dict.items() if k not in ("name", "architecture", "type")}

    # Infer n_actions and obs_dim from env if needed
    env_name = getattr(env, "name", str(env)) if env is not None else "unknown"
    if n_actions is None and env is not None:
        raw_act = getattr(env, "n_actions", getattr(getattr(env, "action_space", None), "n", None))
        resolved_act = raw_act() if callable(raw_act) else raw_act
        if resolved_act is not None:
            n_actions = int(resolved_act)
    if obs_dim is None and env is not None:
        if hasattr(env, "observation_space"):
            obs_shape = getattr(env.observation_space, "shape", None)
            if obs_shape:
                obs_dim = int(np.prod(obs_shape))
        elif hasattr(env, "reset"):
            try:
                sample_obs = env.reset()
                if isinstance(sample_obs, tuple):
                    sample_obs = sample_obs[0]
                obs_dim = sample_obs.shape[-1]
            except Exception:
                pass

    merged_kwargs = {**model_params, **kwargs}

    # Handle blendrl / composite models
    if model_name.lower() in ("blendrl", "blender", "hybrid"):
        from src.usr.models.blendrl.agents.blender_agent import BlenderActorCritic

        # Standard defaults for BlendRL
        actor_mode = merged_kwargs.pop("actor_mode", "hybrid")
        blender_mode = merged_kwargs.pop("blender_mode", "neural")
        blend_function = merged_kwargs.pop("blend_function", "softmax")
        reasoner = merged_kwargs.pop("reasoner", "nsfr")
        rules = merged_kwargs.pop("rules", "default")
        modules = merged_kwargs.pop("modules", None)
        cfg = merged_kwargs.pop("cfg", None)

        return BlenderActorCritic(
            env=env,
            rules=rules,
            actor_mode=actor_mode,
            blender_mode=blender_mode,
            blend_function=blend_function,
            reasoner=reasoner,
            device=device,
            modules=modules,
            cfg=cfg,
            **merged_kwargs,
        ).to(device)

    # Handle CEW model
    if model_name.lower() == "cew":
        from src.usr.models.cew.cew_model import CEWModel

        return CEWModel(
            n_inputs=obs_dim or 1,
            n_actions=n_actions or 2,
            cql_alpha=float(merged_kwargs.get("cql_alpha", 1.0)),
            lr=float(merged_kwargs.get("lr", 3e-4)),
            ecm_dthr=float(merged_kwargs.get("ecm_dthr", 0.1)),
            eps=float(merged_kwargs.get("eps", 0.1)),
            kappa=float(merged_kwargs.get("kappa", 0.6)),
            fyd=bool(merged_kwargs.get("fyd", False)),
            fyd_top_k=merged_kwargs.get("fyd_top_k", None),
            stabilize=bool(merged_kwargs.get("stabilize", True)),
        ).to(device)

    # Resolve architecture from MODEL_REGISTRY
    cls_or_fn = get_model_class(model_name)
    import inspect

    target_fn = cls_or_fn.__init__ if isinstance(cls_or_fn, type) else cls_or_fn
    try:
        sig = inspect.signature(target_fn)
        params = sig.parameters
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        call_kwargs = {}
        if "obs_dim" in params or has_var_kw:
            call_kwargs["obs_dim"] = obs_dim
        if "n_inputs" in params or has_var_kw:
            call_kwargs["n_inputs"] = obs_dim
        if "num_in_features" in params or has_var_kw:
            call_kwargs["num_in_features"] = obs_dim
        if "n_actions" in params or has_var_kw:
            call_kwargs["n_actions"] = n_actions
        if "out_size" in params or has_var_kw:
            call_kwargs["out_size"] = n_actions
        if "device" in params or has_var_kw:
            call_kwargs["device"] = device
        if "env" in params or has_var_kw:
            call_kwargs["env"] = env
        for k, v in merged_kwargs.items():
            call_kwargs[k] = v

        if not has_var_kw:
            call_kwargs = {k: v for k, v in call_kwargs.items() if k in params}

        res = cls_or_fn(**call_kwargs)
    except Exception:
        res = cls_or_fn(obs_dim=obs_dim, n_actions=n_actions, **merged_kwargs)

    if isinstance(res, nn.Module):
        res = res.to(device)
    return res


def _safe_instantiate(module_class, **kwargs):
    """Instantiate a class by passing only parameters accepted by its __init__."""
    import inspect

    sig = inspect.signature(module_class.__init__)
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if has_var_keyword:
        return module_class(**kwargs)
    valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return module_class(**valid_kwargs)

