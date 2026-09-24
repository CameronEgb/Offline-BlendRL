"""Configuration utilities for the pipeline.

Provides functions for parsing method lists and normalizing agent names.
"""


def normalize_agent_name(agent_config: str) -> str:
    """Convert hierarchical agent config paths and dataset IDs to filesystem-safe and Hydra-safe names.
    e.g. 'blendrl_cql/human_cew' -> 'blendrl_cql_human_cew'
         'ex132(w)' -> 'ex132_w'
    This must match agent.name as set in the Hydra overrides."""
    return agent_config.replace("/", "_").replace("(", "_").replace(")", "").replace("__", "_").rstrip("_")


def parse_method_list(val):
    """Parse a method list from Hydra config.
    Hydra/YAML returns a Python list for `[a, b]` syntax but a string for `"a, b"` syntax.
    This function handles both forms."""
    if not val:
        return []
    if isinstance(val, (list, tuple)):
        return list(val)
    if hasattr(val, "__iter__") and not isinstance(val, str):
        return list(val)
    return [item.strip() for item in str(val).split(",") if item.strip()]


_RESERVED_METHOD_KEYS = {
    "params",
    "_params_",
    "defaults",
    "_defaults_",
    "shared",
    "_shared_",
    "common",
    "_common_",
    "common_params",
}


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries, with override taking precedence."""
    if not isinstance(base, dict):
        base = {}
    if not isinstance(override, dict):
        return base
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def parse_methods_dict(cfg) -> dict[str, dict]:
    """Parse and normalize the structured methods: dict from Hydra config.

    Supports shared parameters via:
      1. Top-level params: (or shared_params:, common_params:) in the experiment config.
      2. params: (or _params_, defaults, common) under methods:.

    Shared parameters are deep-merged as base values into every declared method.
    Individual method configurations override shared parameters.
    Special reserved keys are excluded from the returned dictionary so they are
    not treated as runnable methods.

    Returns:
        dict[str, dict]: Normalized method_name -> method_config mapping.
    """
    from omegaconf import DictConfig, OmegaConf

    raw_methods = getattr(cfg, "methods", None) if not isinstance(cfg, dict) else cfg.get("methods")
    if raw_methods is None and hasattr(cfg, "get"):
        raw_methods = cfg.get("methods", None)

    if not raw_methods:
        return {}

    if isinstance(raw_methods, DictConfig):
        raw_methods_dict = OmegaConf.to_container(raw_methods, resolve=True)
    elif hasattr(raw_methods, "items"):
        raw_methods_dict = dict(raw_methods)
    else:
        return {}

    # 1. Extract shared params from top-level config
    top_params = {}
    for top_key in ("params", "shared_params", "common_params"):
        val = getattr(cfg, top_key, None) if not isinstance(cfg, dict) else cfg.get(top_key)
        if val is None and hasattr(cfg, "get"):
            val = cfg.get(top_key, None)
        if val:
            if isinstance(val, DictConfig):
                val_dict = OmegaConf.to_container(val, resolve=True)
            elif hasattr(val, "items"):
                val_dict = dict(val)
            else:
                val_dict = {}
            top_params = deep_merge(top_params, val_dict)

    # 2. Extract shared params from within methods dict (e.g. methods.params)
    method_level_params = {}
    for res_key in _RESERVED_METHOD_KEYS:
        if res_key in raw_methods_dict:
            res_val = raw_methods_dict[res_key]
            if isinstance(res_val, DictConfig):
                res_dict = OmegaConf.to_container(res_val, resolve=True)
            elif isinstance(res_val, dict):
                res_dict = dict(res_val)
            else:
                res_dict = {}
            method_level_params = deep_merge(method_level_params, res_dict)

    shared_params = deep_merge(top_params, method_level_params)

    # 3. Build resolved method configurations
    result = {}
    for method_name, method_cfg in raw_methods_dict.items():
        if method_name in _RESERVED_METHOD_KEYS:
            continue

        if isinstance(method_cfg, DictConfig):
            m_dict = OmegaConf.to_container(method_cfg, resolve=True)
        elif isinstance(method_cfg, dict):
            m_dict = dict(method_cfg)
        else:
            m_dict = {}

        resolved_mcfg = deep_merge(shared_params, m_dict)
        result[str(method_name)] = resolved_mcfg

    return result



def resolve_experiment_config_name(exp_input: str) -> str:
    """Resolve an experiment name (e.g. 'mimic_cql' or 'mimic/mimic_cql')
    to its relative Hydra config path inside in/config/experiment/."""
    from pathlib import Path

    exp_dir = Path("in/config/experiment")
    if not exp_dir.exists():
        return exp_input

    clean_input = exp_input[:-5] if exp_input.endswith(".yaml") else exp_input
    direct_path = exp_dir / f"{clean_input}.yaml"
    if direct_path.exists():
        return clean_input

    # Search recursively in group subdirectories
    matches = list(exp_dir.glob(f"**/{clean_input}.yaml"))
    if not matches:
        raise ValueError(f"Experiment config '{clean_input}.yaml' not found in {exp_dir}")
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous experiment name '{clean_input}'. Found multiple matches: {[str(m) for m in matches]}. Please specify the exact group/experiment path."
        )

    rel = matches[0].relative_to(exp_dir)
    return str(rel.with_suffix(""))
