"""Unified method style registry for the NeSyRL project.

Single source of truth for method display names, colors, line styles, and markers.
Both the plotting system (plot/base.py) and EP evaluation (eval.py) import from here.

To add a new architecture, add ONE entry to METHOD_STYLE below.
"""
import re
from typing import Tuple, Optional

METHOD_STYLE = {
    # Clean harness/architecture keys
    "cql":                          {"label": "DNN",                  "color": "#1f77b4",    "marker": "o", "linestyle": "-"},
    "cql_dnn":                      {"label": "DNN",                  "color": "#1f77b4",    "marker": "o", "linestyle": "-"},
    "dnn":                          {"label": "DNN",                  "color": "#1f77b4",    "marker": "o", "linestyle": "-"},
    "cql_dueling_resnet":           {"label": "Dueling ResNet",       "color": "#08519c",    "marker": "D", "linestyle": "-"},
    "dueling_resnet":               {"label": "Dueling ResNet",       "color": "#08519c",    "marker": "D", "linestyle": "-"},
    "cql_transformer":              {"label": "Transformer",          "color": "#e377c2",    "marker": "p", "linestyle": "-"},
    "transformer":                  {"label": "Transformer",          "color": "#e377c2",    "marker": "p", "linestyle": "-"},
    "cql_blendrl_human_neural":     {"label": "BlendRL (MLP, Human, MLP)", "color": "#fdbf6f", "marker": "s", "linestyle": "-"},
    "blendrl_cql_human_neural":     {"label": "BlendRL (MLP, Human, MLP)", "color": "#fdbf6f", "marker": "s", "linestyle": "-"},
    "cql_blendrl_human_neural_logic": {"label": "BlendRL (MLP, Human, Human)", "color": "#b2df8a", "marker": "^", "linestyle": "--"},
    "blendrl_cql_human_neural_logic": {"label": "BlendRL (MLP, Human, Human)", "color": "#b2df8a", "marker": "^", "linestyle": "--"},
    "cql_blendrl_human_dueling_resnet": {"label": "BlendRL (ResNet, Human, MLP)", "color": "#41b6c4", "marker": "o", "linestyle": "-"},
    "blendrl_cql_human_dueling_resnet": {"label": "BlendRL (ResNet, Human, MLP)", "color": "#41b6c4", "marker": "o", "linestyle": "-"},
    "cql_blendrl_human_dueling_resnet_logic": {"label": "BlendRL (Dueling ResNet, Human, Human)", "color": "#bc80bd", "marker": "D", "linestyle": "-."},
    "blendrl_cql_human_dueling_resnet_logic": {"label": "BlendRL (Dueling ResNet, Human, Human)", "color": "#bc80bd", "marker": "D", "linestyle": "-."},
    "cql_blendrl_human_dueling_resnet_rigid": {"label": "BlendRL ResNet+Rigid", "color": "#d95f02", "marker": "^", "linestyle": "-"},
    "blendrl_cql_human_dueling_resnet_rigid": {"label": "BlendRL ResNet+Rigid", "color": "#d95f02", "marker": "^", "linestyle": "-"},
    "cql_blendrl_human_dueling_resnet_continuous": {"label": "BlendRL ResNet+Soft", "color": "#018571", "marker": "v", "linestyle": "-"},
    "blendrl_cql_human_dueling_resnet_continuous": {"label": "BlendRL ResNet+Soft", "color": "#018571", "marker": "v", "linestyle": "-"},
    "cql_blendrl_human_rigid":      {"label": "BlendRL Rigid Logic",  "color": "#ff7f0e",    "marker": "s", "linestyle": "-"},
    "cql_blendrl_human_continuous": {"label": "BlendRL Soft Logic",   "color": "#17becf",    "marker": "v", "linestyle": "-"},
    "cql_blendrl_human_transformer":{"label": "BlendRL Human+Transformer", "color": "#8c564b", "marker": "h", "linestyle": "-"},
    "blendrl_cql_human_transformer":{"label": "BlendRL Human+Transformer", "color": "#8c564b", "marker": "h", "linestyle": "-"},
    "cql_blendrl_cross_attention":  {"label": "BlendRL Cross-Attention", "color": "#9467bd",  "marker": "*", "linestyle": "-"},
    "blendrl_cql_cross_attention":  {"label": "BlendRL Cross-Attention", "color": "#9467bd",  "marker": "*", "linestyle": "-"},
    "cql_blendrl_human_cew":        {"label": "BlendRL Human+CEW",   "color": "#74c476",    "marker": "^", "linestyle": "--"},
    "blendrl_cql_human_cew":        {"label": "BlendRL Human+CEW",   "color": "#74c476",    "marker": "^", "linestyle": "--"},
    "cql_blendrl_cew_dueling_resnet": {"label": "BlendRL CEW+ResNet", "color": "#2ca02c",   "marker": "h", "linestyle": "-"},
    "blendrl_cql_cew_dueling_resnet": {"label": "BlendRL CEW+ResNet", "color": "#2ca02c",   "marker": "h", "linestyle": "-"},
    "cql_blendrl_human_cew_dueling_resnet": {"label": "BlendRL Human+CEW+ResNet", "color": "#6a3d9a", "marker": "p", "linestyle": "-"},
    "blendrl_cql_human_cew_dueling_resnet": {"label": "BlendRL Human+CEW+ResNet", "color": "#6a3d9a", "marker": "p", "linestyle": "-"},
    "cql_blendrl_cew_only":         {"label": "BlendRL CEW Only",     "color": "#d62728",    "marker": "D", "linestyle": "-"},
    "blendrl_cql_cew_only":         {"label": "BlendRL CEW Only",     "color": "#d62728",    "marker": "D", "linestyle": "-"},
    "iql_dnn":                      {"label": "IQL (Neural)",         "color": "#1f77b4",    "marker": "d", "linestyle": "-"},
    "iql_blendrl_human_neural":     {"label": "BlendRL Human+Neural", "color": "#d62728",    "marker": "s", "linestyle": "-"},
    "ppo_dnn":                      {"label": "PPO (Neural)",         "color": "black",      "marker": "o", "linestyle": "--"},
    "ppo_blendrl_human_neural":     {"label": "BlendRL Human+Neural", "color": "#2ca02c",    "marker": "^", "linestyle": "-"},
    "cew_base":                     {"label": "CEW",                  "color": "#e7298a",    "marker": "h", "linestyle": "-"},
    "cew_fyd":                      {"label": "CEW+FYD",              "color": "#bcbd22",    "marker": "p", "linestyle": "-"},
    "clinician":                    {"label": "Clinician (Dataset)",  "color": "#756bb1",    "marker": "X", "linestyle": "-"},
}

_DEFAULT_STYLE = {"label": None, "color": None, "marker": "o", "linestyle": "-"}


def get_style(name: str) -> dict:
    """Look up style by exact match, canonical name, or longest prefix match.
    
    Examples:
        get_style("cql")                  -> exact match (DNN)
        get_style("cql/dueling_resnet")   -> normalized match (Dueling ResNet)
        get_style("ppo_cp_tuned")          -> prefix match on "ppo"
        get_style("blendrl_iql_cp_tuned")  -> prefix match on "blendrl_iql"
        get_style("unknown_method")        -> default with label=name
    """
    raw = str(name)
    normalized = raw.replace("/", "_")
    canon = get_canonical_method_name(normalized)

    for cand in [canon, normalized, raw]:
        if cand in METHOD_STYLE:
            return METHOD_STYLE[cand]

    # Prefix match: longest key that is a prefix of candidate wins
    for cand in [canon, normalized, raw]:
        for key in sorted(METHOD_STYLE.keys(), key=len, reverse=True):
            if cand.startswith(key + "_") or cand == key:
                return METHOD_STYLE[key]

    return {**_DEFAULT_STYLE, "label": name}


def clean_label(name: str) -> str:
    """Return human-readable display label for a method name."""
    return get_style(name)["label"]


def get_style_info(name: str) -> Tuple[Optional[str], str, str]:
    """Return (color, linestyle, marker) tuple for matplotlib plotting."""
    s = get_style(name)
    return s["color"], s["linestyle"], s["marker"]


def get_canonical_method_name(name: str) -> str:
    """Map method aliases and historical display names to canonical registered name."""
    s = str(name).replace("/", "_")
    alias_map = {
        "blendrl_cql_human_neural": "cql_blendrl_human_neural",
        "blendrl_cql_human_transformer": "cql_blendrl_human_transformer",
        "blendrl_cql_human_cew": "cql_blendrl_human_cew",
        "blendrl_cql_cew_only": "cql_blendrl_cew_only",
        "blendrl_iql_human_neural": "iql_blendrl_human_neural",
        "blendrl_ppo_human_neural": "ppo_blendrl_human_neural",
        "cql": "cql_dnn",
        "dnn": "cql_dnn",
        "cql (standard mlp)": "cql_dnn",
        "cql (dueling resnet)": "cql_dueling_resnet",
        "cql (transformer)": "cql_transformer",
        "dueling_resnet": "cql_dueling_resnet",
        "cql_dueling_resnet": "cql_dueling_resnet",
        "transformer": "cql_transformer",
        "cql_transformer": "cql_transformer",
        "iql": "iql_dnn",
        "ppo": "ppo_dnn",
    }
    if s in alias_map:
        return alias_map[s]
    s_lower = s.lower()
    if s_lower in alias_map:
        return alias_map[s_lower]
    return s


def get_method_aliases(name: str) -> set:
    """Return all known aliases for a given method name."""
    canon = get_canonical_method_name(name)
    raw = str(name).replace("/", "_")
    aliases = {name, canon, raw}
    if "cql_blendrl_" in canon:
        aliases.add(canon.replace("cql_blendrl_", "blendrl_cql_"))
    elif "blendrl_cql_" in canon:
        aliases.add(canon.replace("blendrl_cql_", "cql_blendrl_"))
    if canon == "cql_dnn":
        aliases.update({"cql", "dnn", "cql/dnn"})
    elif canon == "cql_dueling_resnet":
        aliases.update({"dueling_resnet", "cql/dueling_resnet"})
    elif canon == "cql_transformer":
        aliases.update({"transformer", "cql/transformer"})
    elif canon == "iql_dnn":
        aliases.update({"iql", "iql/dnn"})
    elif canon == "ppo_dnn":
        aliases.update({"ppo", "ppo/dnn"})
    return aliases
