"""Pre-flight configuration validation module.

Validates experiment configs against their declared paradigm before any
training begins. Raises ConfigurationError on fatal mismatches so the
pipeline aborts cleanly at startup.

Paradigm definitions live in in/config/paradigms/base/ and
in/config/paradigms/meta/. Adding a new paradigm requires only a new YAML
file — no changes to this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


from src.method_registry import METHOD_STYLE
from src.pipeline.config import normalize_agent_name, parse_method_list, resolve_experiment_config_name
from src.pipeline.datasets import resolve_dataset_path
from src.pipeline.exceptions import ConfigurationError

# ---------------------------------------------------------------------------
# Paradigm registry loading
# ---------------------------------------------------------------------------

_PARADIGMS_DIR = Path("in/config/paradigms")


def _load_all_paradigms() -> dict[str, dict]:
    """Load all paradigm YAML files from in/config/paradigms/.

    Returns a dict mapping paradigm name → paradigm definition dict.
    Scans both base/ and meta/ subdirectories.
    """
    paradigms: dict[str, dict] = {}
    for yaml_path in sorted(_PARADIGMS_DIR.glob("**/*.yaml")):
        name = yaml_path.stem  # filename without extension = paradigm name
        with open(yaml_path) as f:
            defn = yaml.safe_load(f) or {}
        paradigms[name] = defn
    return paradigms


def list_paradigms() -> list[str]:
    """Return sorted list of registered paradigm names."""
    return sorted(_load_all_paradigms().keys())


def load_paradigm(paradigm_name: str) -> dict:
    """Load and return the definition dict for a named paradigm.

    Raises ConfigurationError if the paradigm is not found.
    """
    paradigms = _load_all_paradigms()
    if paradigm_name not in paradigms:
        raise ConfigurationError(
            f"[ConfigurationError] Unknown paradigm '{paradigm_name}'. "
            f"Available paradigms: {sorted(paradigms.keys())}. "
            f"To add a new paradigm create in/config/paradigms/base/<name>.yaml."
        )
    return paradigms[paradigm_name]


# ---------------------------------------------------------------------------
# Constraint checking helpers
# ---------------------------------------------------------------------------

def _deep_get(cfg: Any, dotted_key: str, default=None):
    """Resolve a dotted key path against a Hydra cfg or plain dict.

    E.g. _deep_get(cfg, 'env.offline_only') returns cfg.env.offline_only.
    """
    parts = dotted_key.split(".")
    obj = cfg
    for part in parts:
        if obj is None:
            return default
        if isinstance(obj, dict):
            obj = obj.get(part)
        else:
            # OmegaConf / MagicMock — use getattr then .get
            try:
                obj = getattr(obj, part)
            except AttributeError:
                try:
                    obj = obj.get(part)
                except Exception:
                    return default
    return obj if obj is not None else default


_CONSTRAINT_CHECKERS = {
    "non_empty": lambda v: bool(v),
    "non_zero": lambda v: bool(v) and v != 0,
    "true": lambda v: v is True,
    "false": lambda v: v is False,
    True: lambda v: v is True,
    False: lambda v: v is False,
}


def _check_paradigm_constraints(cfg: Any, paradigm_name: str, constraints: dict, env_name: str) -> None:
    """Apply requires/forbids rules from the paradigm definition.

    Raises ConfigurationError on any violation.
    """
    for key, rule in (constraints.get("requires") or {}).items():
        if rule not in _CONSTRAINT_CHECKERS:
            continue  # unknown rule type, skip silently
        val = _deep_get(cfg, key)
        if not _CONSTRAINT_CHECKERS[rule](val):
            raise ConfigurationError(
                f"[ConfigurationError] Paradigm '{paradigm_name}' requires '{key}' "
                f"to satisfy rule '{rule}', but got {val!r} "
                f"(env: '{env_name}')."
            )

    for key, rule in (constraints.get("forbids") or {}).items():
        if rule not in _CONSTRAINT_CHECKERS:
            continue
        val = _deep_get(cfg, key)
        if _CONSTRAINT_CHECKERS[rule](val):
            raise ConfigurationError(
                f"[ConfigurationError] Paradigm '{paradigm_name}' forbids '{key}' "
                f"from satisfying rule '{rule}', but got {val!r} "
                f"(env: '{env_name}')."
            )


# ---------------------------------------------------------------------------
# Raw experiment YAML loader (for explicit override checks)
# ---------------------------------------------------------------------------

def _load_raw_experiment_yaml(experiment_name: str) -> dict:
    """Return the raw (pre-Hydra-composition) experiment YAML as a dict, or {} on failure."""
    rel_path = resolve_experiment_config_name(experiment_name)
    exp_path = Path("in/config/experiment") / f"{rel_path}.yaml"
    if not exp_path.exists():
        return {}
    with open(exp_path) as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_experiment_config(cfg: Any, experiment_name: str, is_sweep: bool = False) -> list[str]:
    """Validate experiment configuration against its declared paradigm.

    Returns a list of non-fatal notice strings.
    Raises ConfigurationError on any fatal paradigm incompatibility.
    """
    notices: list[str] = []

    env_name = getattr(cfg.env, "name", "unknown") if hasattr(cfg, "env") else "unknown"
    raw_exp = _load_raw_experiment_yaml(experiment_name)

    # --- Paradigm lookup ---
    paradigm_name = cfg.get("paradigm", None)
    if not paradigm_name:
        raise ConfigurationError(
            f"[ConfigurationError] Experiment '{experiment_name}' has no 'paradigm' declared. "
            f"Add 'paradigm: <name>' to its group _base.yaml. "
            f"Available paradigms: {list_paradigms()}."
        )

    paradigm_def = load_paradigm(paradigm_name)  # raises ConfigurationError if unknown
    constraints = paradigm_def.get("constraints", {})

    
    # --- intervals_count: check raw YAML for explicit override ---
    # Only enforce if the paradigm does not allow intervals > 1.
    # Currently only base paradigms with offline data disallow intervals.
    # Meta-paradigms and paradigms with allows_intervals: true skip this.
    allows_intervals = paradigm_def.get("allows_intervals", False)
    if not allows_intervals:
        explicit_intervals = raw_exp.get("intervals_count", None)
        resolved_intervals = cfg.get("intervals_count", 1)
        intervals_to_check = explicit_intervals if explicit_intervals is not None else resolved_intervals
        if intervals_to_check and intervals_to_check > 1:
            raise ConfigurationError(
                f"[ConfigurationError] Paradigm '{paradigm_name}' (env '{env_name}') does not "
                f"support intervals_count > 1 (got {intervals_to_check}). "
                f"Progressive dataset slicing requires a paradigm with allows_intervals: true."
            )

    # --- eval_episodes: check raw YAML for explicit override ---
    allows_eval_episodes = paradigm_def.get("allows_eval_episodes", False)
    if not allows_eval_episodes:
        explicit_eval_ep = raw_exp.get("eval_episodes", None)
        resolved_eval_ep = cfg.get("eval_episodes", 0)
        eval_ep_to_check = explicit_eval_ep if explicit_eval_ep is not None else resolved_eval_ep
        if eval_ep_to_check and eval_ep_to_check > 0:
            raise ConfigurationError(
                f"[ConfigurationError] Paradigm '{paradigm_name}' (env '{env_name}') disables "
                f"simulated gym rollouts — policy actions cannot alter historical trajectories. "
                f"Remove 'eval_episodes' from the experiment config."
            )

    # --- Apply paradigm constraints ---
    _check_paradigm_constraints(cfg, paradigm_name, constraints, env_name)
# --- Optuna sweep direction notice ---
    if is_sweep:
        try:
            sweeper = cfg.get("hydra", {}).get("sweeper", {})
            direction = sweeper.get("direction", None)
            monitor = paradigm_def.get("monitor_metric", None)
            if direction == "maximize" and monitor and "loss" in monitor:
                notices.append(
                    f"Notice: Optuna direction is 'maximize' on paradigm '{paradigm_name}' "
                    f"which monitors '{monitor}'. Consider using 'direction: minimize'."
                )
        except Exception:
            pass

    # --- Method registry checks (non-fatal) ---
    _validate_method_registrations(cfg, notices)

    # --- Offline dataset path checks (non-fatal) ---
    _validate_offline_dataset_paths(cfg, notices)

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
