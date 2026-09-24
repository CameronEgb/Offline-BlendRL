"""Single source of truth for training command construction.

Builds Hydra override lists from structured method config dicts.
"""

from src.app.pipeline.config import normalize_agent_name


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
    agent_algo = method_cfg.get("agent")
    model_arch = method_cfg.get("model")
    agent_name = method_cfg.get("name", normalize_agent_name(method_name))
    experiment_name = cfg.get("experiment_name", "") if cfg is not None else ""

    if not model_arch:
        raise ValueError(f"Method '{method_name}' is missing required key 'model'.")

    overrides = [
        f"+experiment={experiment_name}",
        f"paradigm={paradigm}",
        f"model={model_arch}",
    ]

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
    for k, v in method_cfg.items():
        if k in _STRUCTURAL_KEYS:
            continue
        if isinstance(v, dict):
            target_ns = "model" if paradigm == "supervised" else "agent"
            for sub_k, sub_v in v.items():
                overrides.append(f"++{target_ns}.{k}.{sub_k}={sub_v}")
        else:
            if paradigm == "supervised":
                overrides.append(f"++model.{k}={v}")
                if k in ("lr", "batch_size", "epochs", "epochs_per_interval", "eval_interval_epochs", "weight_decay"):
                    overrides.append(f"++{k}={v}")
            else:
                overrides.append(f"++agent.{k}={v}")

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
