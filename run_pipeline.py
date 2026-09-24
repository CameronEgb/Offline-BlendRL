"""Unified experiment pipeline orchestrator for NeSyRL.

Coordinates online and offline RL methods, Optuna sweeps, and Slurm cluster submissions.
"""

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
for p in [
    PROJECT_ROOT,
    SRC_DIR,
    os.path.join(SRC_DIR, "app"),
    os.path.join(SRC_DIR, "usr"),
    os.path.join(SRC_DIR, "usr", "models"),
    os.path.join(SRC_DIR, "usr", "environments"),
    os.path.join(SRC_DIR, "usr", "eval"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

import hydra
from hydra import compose, initialize

from src.app.pipeline.config import normalize_agent_name, resolve_experiment_config_name
from src.app.pipeline.datasets import run_plotting
from src.app.pipeline.exceptions import ConfigurationError
from src.app.pipeline.optuna_utils import launch_optuna_dashboard
from src.app.pipeline.slurm import generate_sbatch_header, submit_sbatch
from src.app.pipeline.validation import validate_experiment_config


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python run_pipeline.py <group>/<experiment_id> [Hydra Overrides]")
        print("Standard Orchestration Overrides (use Hydra syntax, e.g. key=value):")
        print(
            "  site=local            Interactive CLI execution (default, for local machine or cluster interactive node)"
        )
        print("  site=ncshare          Slurm cluster execution on NCShare")
        print("  site=arc              Slurm cluster execution on ARC")
        print("  plot_only=true        Run plotting phase only (no training, no Slurm submission)")
        print("  no_plot=true          Skip automatic plotting")
        print("  no_online=true        Skip online training phase")
        print("  no_offline=true       Skip offline training phase")
        print("  dry_run=true          Validate config and exit")
        print("  sweep=true            Run Optuna hyperparameter sweep")
        print("  dash=true             Launch Optuna dashboard during run")
        print("  dash_only=true        Launch Optuna dashboard and exit")
        print("  remake=true           Force recalculation of summaries")
        print("  consolidate=true      Consolidate Slurm jobs into 1 single GPU job")
        print("  experiment_id=...     Override the experiment directory name")
        sys.exit(0)

    raw_experiment = sys.argv[1]
    extra_args = sys.argv[2:]

    # Resolve experiment config path
    experiment_arg = resolve_experiment_config_name(raw_experiment)

    # Prepare pure Hydra compose overrides (filtering out sweep logic that crashes compose API)
    overrides_for_compose = [f"+experiment={experiment_arg}", f"++experiment_name={experiment_arg}"]
    for arg in extra_args:
        # The compose API cannot parse Optuna sweep operators or hydra internal configs
        if any(sw in arg for sw in ["interval(", "choice(", "range(", "hydra.", "hydra/"]):
            continue
        if "=" in arg:
            overrides_for_compose.append(arg)

    # Subprocesses need everything PLUS the internal experiment tracking
    sanitized_extra_args = list(extra_args)
    sanitized_extra_args.append(f"++experiment_name={experiment_arg}")

    # Load configuration
    try:
        print(f"Loading Hydra configuration for '{experiment_arg}'...", flush=True)
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        initialize(version_base=None, config_path="in/config")
        cfg = compose(config_name="config", overrides=overrides_for_compose, return_hydra_config=True)
        exp_stem = Path(experiment_arg).stem
        if not cfg.get("experiment_id") or cfg.experiment_id == "default_exp":
            cfg.experiment_id = exp_stem

        exp_group = Path(experiment_arg).parent.name if "/" in experiment_arg else "ungrouped"
        if not cfg.get("group") or cfg.group == "ungrouped":
            cfg.group = exp_group

        # Ensure it's in extra args so it passes to children
        if not any("experiment_id=" in arg for arg in sanitized_extra_args):
            sanitized_extra_args.append(f"++experiment_id={cfg.experiment_id}")
        if not any("group=" in arg for arg in sanitized_extra_args):
            sanitized_extra_args.append(f"++group={cfg.group}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    is_sweep = cfg.get("sweep", False) or "--multirun" in sanitized_extra_args or "-m" in sanitized_extra_args
    if is_sweep and "--multirun" not in sanitized_extra_args and "-m" not in sanitized_extra_args:
        sanitized_extra_args.append("--multirun")

    # Pre-flight validation
    try:
        notices = validate_experiment_config(cfg, experiment_arg, is_sweep=is_sweep)
        for n in notices:
            print(f"[Config Notice] {n}")
    except ConfigurationError as e:
        print(f"\n{e}\n")
        sys.exit(1)

    if cfg.get("dry_run", False):
        print(f"\n[Validation Success] Experiment config '{experiment_arg}' is valid and ready to run.")
        sys.exit(0)

    if cfg.get("plot_only", False):
        print(f"\n=== Running Plotting Phase Only for '{experiment_arg}' ===")
        run_plotting(
            experiment=experiment_arg,
            style=cfg.get("plot_style", None),
            wipe=cfg.get("wipe", False),
            use_cache=cfg.get("use_cache", False),
        )
        sys.exit(0)

    # Load paradigm definition for component assembly
    paradigm_def = None
    try:
        from src.app.core.paradigm_loader import load_paradigm_definition
        paradigm_name = cfg.get("paradigm", None)
        if paradigm_name:
            paradigm_def = load_paradigm_definition(paradigm_name)
    except Exception as e:
        # Paradigm loading is best-effort during transition; log but don't abort
        print(f"[Notice] Could not load paradigm definition: {e}")

    # Execution mode is determined solely by the site profile: site=local -> interactive CLI, any other site -> Slurm cluster
    site_name = getattr(cfg.site, "name", "local") if hasattr(cfg, "site") else "local"
    is_interactive = site_name == "local"
    print(f"Execution Mode: {'Interactive (Local CLI)' if is_interactive else f'Slurm Cluster ({site_name})'}")

    storage_url = None
    if "hydra" in cfg and "sweeper" in cfg.hydra and "storage" in cfg.hydra.sweeper:
        storage_url = cfg.hydra.sweeper.storage
        if storage_url:
            storage_url = str(storage_url).replace("${experiment_id}", cfg.experiment_id)
        import os

        os.makedirs("results/optuna", exist_ok=True)

    if is_interactive and storage_url and (cfg.get("dash") or cfg.get("dash_only")):
        launch_optuna_dashboard(storage_url)
        if cfg.get("dash_only"):
            print("Dashboard running in persistent mode. Press Ctrl+C to exit.")
            import time

            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                sys.exit(0)

    if cfg.get("dash_only"):
        print("Error: Could not find Optuna storage URL in configuration.")
        sys.exit(1)

    # Parse structured methods: dict
    methods = cfg.get("methods", None)
    if not methods:
        raise ConfigurationError(
            f"[ConfigurationError] Experiment '{experiment_arg}' has no 'methods:' dict. "
            f"Declare methods as a dict with agent + model per entry."
        )

    # Reject legacy keys
    for legacy_key in ("online_methods", "offline_methods", "offline_datasets"):
        if cfg.get(legacy_key, None):
            raise ConfigurationError(
                f"[ConfigurationError] Legacy key '{legacy_key}' found in config. "
                f"Use the 'methods:' dict instead."
            )

    # Convert OmegaConf to plain dict
    from omegaconf import OmegaConf
    methods_dict = OmegaConf.to_container(methods, resolve=True)

    print(f"Declared Methods:")
    for name, mcfg in methods_dict.items():
        agent_str = f"agent={mcfg.get('agent')}, " if mcfg.get('agent') else ""
        print(f"  {name}: {agent_str}model={mcfg.get('model')}")

    # Build context for tasks
    context = {
        "is_interactive": is_interactive,
        "site_name": site_name,
        "sanitized_extra_args": sanitized_extra_args,
        "storage_url": storage_url,
        "is_sweep": is_sweep,
        "methods": methods_dict,
        "paradigm_def": paradigm_def,
    }

    # Extract task name
    task_name = cfg.get("task", "rl")
    if not task_name:
        task_name = "rl"

    from src.app.pipeline.task_registry import get_task

    task_fn = get_task(task_name)

    # Introspect task_fn to see if it accepts args (backwards compatibility for custom tasks)
    import inspect

    sig = inspect.signature(task_fn)
    if "args" in sig.parameters:
        task_fn(cfg, None, context)
    else:
        task_fn(cfg, context)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] not in ("-h", "--help"):
        print(f"Initializing BlendRL pipeline for: {sys.argv[1]} ...", flush=True)
    main()
