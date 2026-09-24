"""Single source of truth for training command construction.

Builds Hydra override lists from structured method config dicts.
"""

from src.app.pipeline.config import normalize_agent_name


def _format_hydra_val(x):
    """Format Python values into Hydra override syntax."""
    if isinstance(x, (list, tuple)):
        return "[" + ",".join(_format_hydra_val(item) for item in x) + "]"
    elif isinstance(x, dict):
        return "{" + ",".join(f"{k}:{_format_hydra_val(val)}" for k, val in x.items()) + "}"
    elif isinstance(x, bool):
        return "true" if x else "false"
    return str(x)


def build_method_overrides(
    method_name: str,
    method_cfg: dict,
    dataset_path=None,
    extra_args: list | None = None,
    cfg=None,
    study_name: str | None = None,
) -> list[str]:
    """Build Hydra override list from a structured methods: dict entry.

    Args:
        method_name:  Key in the experiment's methods: dict (e.g. 'cql_dnn').
        method_cfg:   The dict value for that entry. Must contain 'agent' and 'model'.
        dataset_path: Resolved filesystem path to the offline dataset (offline only).
        extra_args:   Additional Hydra overrides forwarded from the CLI.
        cfg:          Full Hydra config (used for experiment_name, paradigm).
        study_name:   Optuna study name (optional, for sweep runs).

    Returns:
        list[str]: Hydra override arguments ready to pass to train.py.
    """
    paradigm = cfg.get("paradigm", "offline_rl") if cfg is not None else "offline_rl"
    agent_val = method_cfg.get("agent")
    if isinstance(agent_val, dict):
        agent_algo = agent_val.get("name") or agent_val.get("type") or agent_val.get("algo")
        agent_subparams = {k: v for k, v in agent_val.items() if k not in ("name", "type", "algo")}
    else:
        agent_algo = agent_val
        agent_subparams = {}

    model_val = method_cfg.get("model")
    if isinstance(model_val, dict):
        model_arch = model_val.get("name") or model_val.get("type") or model_val.get("base")
        model_subparams = {k: v for k, v in model_val.items() if k not in ("name", "type", "base")}
    else:
        model_arch = model_val
        model_subparams = {}

    agent_name = method_cfg.get("name", normalize_agent_name(method_name))
    experiment_name = cfg.get("experiment_name", "") if cfg is not None else ""

    if not model_arch:
        raise ValueError(f"Method '{method_name}' is missing required key 'model' (or 'model.name').")

    overrides = [
        f"+experiment={experiment_name}",
        f"paradigm={paradigm}",
        f"model={model_arch}",
    ]

    def _flatten_overrides(prefix, d):
        for k, v in d.items():
            if isinstance(v, dict):
                _flatten_overrides(f"{prefix}.{k}", v)
            else:
                formatted_v = _format_hydra_val(v)
                overrides.append(f"++{prefix}.{k}={formatted_v}")

    _flatten_overrides("model", model_subparams)
    _flatten_overrides("agent", agent_subparams)

    if paradigm == "supervised":
        overrides.append(f"++model.name={agent_name}")
        overrides.append(f"++agent.name={agent_name}")
    else:
        if not agent_algo:
            raise ValueError(f"Method '{method_name}' is missing required key 'agent' for paradigm '{paradigm}'.")
        overrides.extend([
            f"agent={agent_algo}",
            f"++agent.name={agent_name}",
        ])

    if (paradigm in ("offline_rl", "supervised")) and dataset_path is not None:
        safe_ds_path = str(dataset_path)
        if any(c in safe_ds_path for c in "(), "):
            safe_ds_path = f'"{{safe_ds_path}}"'
        overrides.append(f"++dataset_path={safe_ds_path}")

    # Per-method hyperparameter overrides — only applied to this specific method.
    _STRUCTURAL_KEYS = {"agent", "model"}
    _MODEL_KEYS = {
        "architecture",
        "modules",
        "rules",
        "ecm_dthr",
        "fyd",
        "fyd_top_k",
        "actor_mode",
        "blender_mode",
        "blend_function",
        "blender",
        "neural_actor",
        "symbolic_actor",
        "hidden_sizes",
    }
    for k, v in method_cfg.items():
        if k in _STRUCTURAL_KEYS:
            continue
        formatted_v = _format_hydra_val(v)
        if isinstance(v, dict):
            target_ns = "model" if paradigm == "supervised" else "agent"
            for sub_k, sub_v in v.items():
                overrides.append(f"++{target_ns}.{k}.{sub_k}={_format_hydra_val(sub_v)}")
            if k in _MODEL_KEYS and target_ns != "model":
                for sub_k, sub_v in v.items():
                    overrides.append(f"++model.{k}.{sub_k}={_format_hydra_val(sub_v)}")
        else:
            if paradigm == "supervised":
                overrides.append(f"++model.{k}={formatted_v}")
                if k in ("lr", "batch_size", "epochs", "epochs_per_interval", "eval_interval_epochs", "weight_decay"):
                    overrides.append(f"++{k}={formatted_v}")
            else:
                overrides.append(f"++agent.{k}={formatted_v}")
                if k in _MODEL_KEYS:
                    overrides.append(f"++model.{k}={formatted_v}")

    if study_name:
        overrides.append(f"++hydra.sweeper.study_name={study_name}")

    if extra_args:
        overrides.extend(extra_args)

    return overrides


def get_sweep_direction(cfg, paradigm: str) -> str:
    """Single source of truth for Optuna sweep direction.

    Checks cfg.hydra.sweeper.direction first (explicit override).
    Defaults: online_rl → 'maximize' (reward), offline_rl → 'minimize' (loss).
    """
    if hasattr(cfg, "hydra") and hasattr(cfg.hydra, "sweeper"):
        explicit = cfg.hydra.sweeper.get("direction", None)
        if explicit:
            return str(explicit)
    return "maximize" if paradigm == "online_rl" else "minimize"
