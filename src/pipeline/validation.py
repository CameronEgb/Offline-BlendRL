"""Pre-flight configuration validation module.

Validates experiment configs against their declared paradigm before any training begins.
Raises ConfigurationError on fatal mismatches so the pipeline aborts cleanly at startup.
"""

from pathlib import Path
from typing import Any, List

from src.method_registry import METHOD_STYLE
from src.pipeline.config import normalize_agent_name, parse_method_list, resolve_experiment_config_name
from src.pipeline.datasets import resolve_dataset_path
from src.pipeline.exceptions import ConfigurationError

# Recognised paradigm names and the env/config constraints they enforce.
PARADIGM_CONSTRAINTS = {
    "online_v_offline": {
        "requires_offline_only": False,
        "allows_eval_episodes": True,
        "allows_intervals_count_gt1": True,
        "requires_online_methods": True,
    },
    "offline_only": {
        "requires_offline_only": True,
        "allows_eval_episodes": False,
        "allows_intervals_count_gt1": False,
        "requires_online_methods": False,
    },
}

# Task paradigms that bypass standard RL paradigm validation entirely.
TASK_PARADIGMS = {"early_prediction", "early_prediction_sweep", "reciprocal_refinement"}


def _load_raw_experiment_yaml(experiment_name: str) -> dict:
    """Return the raw (pre-Hydra-composition) experiment YAML as a dict, or {} on failure."""
    rel_path = resolve_experiment_config_name(experiment_name)
    exp_path = Path("in/config/experiment") / f"{rel_path}.yaml"
    if not exp_path.exists():
        return {}
    import yaml

    with open(exp_path) as f:
        return yaml.safe_load(f) or {}


def validate_experiment_config(cfg: Any, experiment_name: str, is_sweep: bool = False) -> list[str]:
    """Validate experiment configuration against its declared paradigm.

    Returns a list of non-fatal notice strings.
    Raises ConfigurationError on any fatal paradigm incompatibility.
    """
    notices: list[str] = []

    env_name = getattr(cfg.env, "name", "unknown") if hasattr(cfg, "env") else "unknown"
    is_offline_only = getattr(cfg.env, "offline_only", False)

    raw_exp = _load_raw_experiment_yaml(experiment_name)

    # --- Determine paradigm ---
    # Resolved cfg carries `paradigm` from the group _base.yaml (or experiment override).
    paradigm = cfg.get("paradigm", None)

    # Task-specific paradigms skip standard RL paradigm validation.
    task_name = cfg.get("task", "rl") or "rl"
    if task_name in TASK_PARADIGMS:
        # Still validate agent registrations and dataset paths.
        _validate_method_registrations(cfg, notices)
        _validate_offline_dataset_paths(cfg, notices)
        return notices

    if paradigm is None:
        raise ConfigurationError(
            f"[ConfigurationError] Experiment '{experiment_name}' has no 'paradigm' declared. "
            f"Add 'paradigm: online_v_offline' or 'paradigm: offline_only' to its group _base.yaml."
        )

    if paradigm not in PARADIGM_CONSTRAINTS:
        raise ConfigurationError(
            f"[ConfigurationError] Experiment '{experiment_name}' declares unknown paradigm '{paradigm}'. "
            f"Valid paradigms: {list(PARADIGM_CONSTRAINTS.keys())}."
        )

    constraints = PARADIGM_CONSTRAINTS[paradigm]

    # --- 1. offline_only env vs paradigm ---
    if constraints["requires_offline_only"] and not is_offline_only:
        raise ConfigurationError(
            f"[ConfigurationError] Paradigm '{paradigm}' requires an offline-only environment "
            f"(env.offline_only: true), but env '{env_name}' has offline_only: false."
        )
    if not constraints["requires_offline_only"] and is_offline_only:
        raise ConfigurationError(
            f"[ConfigurationError] Paradigm '{paradigm}' requires a live simulator environment "
            f"(env.offline_only: false), but env '{env_name}' has offline_only: true."
        )

    # --- 2. intervals_count ---
    # Check the raw YAML for an explicit override; composed cfg may have a default of 1.
    explicit_intervals = raw_exp.get("intervals_count", None)
    resolved_intervals = cfg.get("intervals_count", 1)
    intervals_to_check = explicit_intervals if explicit_intervals is not None else resolved_intervals

    if not constraints["allows_intervals_count_gt1"] and intervals_to_check > 1:
        raise ConfigurationError(
            f"[ConfigurationError] Paradigm '{paradigm}' (env '{env_name}') does not support "
            f"intervals_count > 1, but got intervals_count={intervals_to_check}. "
            f"Progressive dataset slicing requires the 'online_v_offline' paradigm."
        )

    # --- 3. eval_episodes on static datasets ---
    explicit_eval_episodes = raw_exp.get("eval_episodes", None)
    resolved_eval_episodes = cfg.get("eval_episodes", 0)
    eval_episodes_to_check = explicit_eval_episodes if explicit_eval_episodes is not None else resolved_eval_episodes

    if not constraints["allows_eval_episodes"] and eval_episodes_to_check and eval_episodes_to_check > 0:
        raise ConfigurationError(
            f"[ConfigurationError] Paradigm '{paradigm}' (env '{env_name}') disables simulated gym "
            f"rollouts — policy actions cannot alter historical trajectories. "
            f"Remove 'eval_episodes' from the experiment config or use 'online_v_offline'."
        )

    # --- 4. online_methods required ---
    online_list = parse_method_list(cfg.get("online_methods", ""))
    if constraints["requires_online_methods"] and not online_list:
        raise ConfigurationError(
            f"[ConfigurationError] Paradigm '{paradigm}' requires at least one entry in "
            f"'online_methods', but none were found in experiment '{experiment_name}'."
        )

    # --- 5. Method registry checks (non-fatal) ---
    _validate_method_registrations(cfg, notices)

    # --- 6. Offline dataset path checks (non-fatal) ---
    _validate_offline_dataset_paths(cfg, notices)

    # --- 7. Optuna sweep direction ---
    if is_sweep:
        sweeper = cfg.get("hydra", {}).get("sweeper", {})
        direction = sweeper.get("direction", None)
        if paradigm == "offline_only" and direction == "maximize":
            notices.append(
                f"Notice: Optuna direction is 'maximize' on offline_only experiment '{experiment_name}'. "
                f"Offline experiments monitor 'val/loss' and should use 'direction: minimize'."
            )

    return notices


def _validate_method_registrations(cfg: Any, notices: list[str]) -> None:
    """Append non-fatal notices for methods not found in METHOD_STYLE registry."""
    registered = set(METHOD_STYLE.keys())

    for method in parse_method_list(cfg.get("online_methods", [])):
        base_algo = method.split("/")[0]
        norm = method.replace("/", "_")
        if base_algo not in registered and norm not in registered:
            notices.append(f"Notice: Online method '{method}' might not match a registered agent style.")

    for method in parse_method_list(cfg.get("offline_methods", [])):
        base_algo = method.split("/")[0]
        norm = method.replace("/", "_")
        if base_algo not in registered and norm not in registered:
            notices.append(f"Notice: Offline method '{method}' might not match a registered agent style.")


def _validate_offline_dataset_paths(cfg: Any, notices: list[str]) -> None:
    """Append non-fatal notices for offline datasets that cannot be resolved."""
    online_list = parse_method_list(cfg.get("online_methods", []))
    offline_list = parse_method_list(cfg.get("offline_methods", []))
    if not offline_list:
        return

    offline_datasets = parse_method_list(cfg.get("offline_datasets", []))
    for ds in offline_datasets:
        norm_ds = normalize_agent_name(ds)
        norm_online = [normalize_agent_name(m) for m in online_list]
        if ds in online_list or norm_ds in norm_online:
            continue
        try:
            yaml_ds_path = cfg.mode.get("dataset_path", None) if hasattr(cfg, "mode") else None
            resolve_dataset_path(
                ds,
                group=cfg.get("group", ""),
                experiment_id=cfg.get("experiment_id", ""),
                yaml_ds_path=yaml_ds_path,
            )
        except Exception as e:
            notices.append(f"Dataset check: {e}")
