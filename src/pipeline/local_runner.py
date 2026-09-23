"""Local training phase runner.

Exposes four composable phase functions used by paradigm runner classes:
    _setup_output_dirs  – purge/recreate results directories
    run_online_phase    – online agent training (returns best_online_trial_ids)
    run_offline_phase   – offline agent training on collected/configured datasets
    run_plotting_phase  – automated post-training plotting

run_local_training() is a compatibility shim that calls all four in sequence.
"""

import sys
from pathlib import Path

from src.pipeline.commands import build_offline_overrides, build_online_overrides, get_sweep_direction
from src.pipeline.config import normalize_agent_name
from src.pipeline.datasets import ensure_online_dataset_path, fast_purge_dir, resolve_dataset_path, run_experiment
from src.pipeline.optuna_utils import (
    create_optuna_study,
    delete_optuna_study,
    get_next_study_name,
    promote_best_trial_checkpoint,
)


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

def _setup_output_dirs(cfg) -> None:
    """Purge and recreate checkpoint, log, and plot directories unless recovering."""
    if cfg.get("recover", False):
        return

    ckpt_dir = Path("results/checkpoints") / cfg.group / cfg.experiment_id
    fast_purge_dir(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    exp_log_dir = Path("results/logs") / cfg.group / cfg.experiment_id
    fast_purge_dir(exp_log_dir)
    exp_log_dir.mkdir(parents=True, exist_ok=True)

    clean_exp = Path(cfg.experiment_id).stem
    exp_plot_dir = Path("results/plots") / cfg.group / clean_exp
    fast_purge_dir(exp_plot_dir)
    exp_plot_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Phase 1: Online training
# ---------------------------------------------------------------------------

def run_online_phase(cfg, context) -> dict:
    """Execute online training for every online agent.

    Returns:
        best_online_trial_ids: mapping of agent_config → best trial id string.
    """
    online_list = context["online_list"]
    sanitized_extra_args = context["sanitized_extra_args"]
    storage_url = context["storage_url"]
    is_sweep = context["is_sweep"]

    best_online_trial_ids: dict = {}

    for agent_config in online_list:
        agent_name_internal = normalize_agent_name(agent_config)
        study_name = get_next_study_name(cfg.group, cfg.experiment_id, agent_name_internal)

        dataset_path, has_pkl = ensure_online_dataset_path(
            group=cfg.group,
            experiment_id=cfg.experiment_id,
            agent_name_internal=agent_name_internal,
            is_sweep=is_sweep,
        )

        if has_pkl:
            print(f"Dataset already exists at {dataset_path}. Skipping online training.")
            best_online_trial_ids[agent_config] = "0"
            continue

        print(f"\n=== Phase: Online Training ({agent_config}) ===")
        overrides = build_online_overrides(
            experiment=cfg.get("experiment_name", ""),
            agent_config=agent_config,
            agent_name=agent_name_internal,
            dataset_path=dataset_path,
            local_val=True,
            study_name=study_name,
            extra_args=sanitized_extra_args,
            cfg=cfg,
        )

        if is_sweep:
            delete_optuna_study(storage_url, study_name)
            direction = get_sweep_direction(cfg, "online")
            create_optuna_study(storage_url, study_name, direction=direction)

        run_experiment(overrides)

        if is_sweep:
            best_id = promote_best_trial_checkpoint(
                cfg.group, cfg.experiment_id, agent_name_internal, storage_url, study_name
            )
            best_online_trial_ids[agent_config] = best_id
        else:
            best_online_trial_ids[agent_config] = "0"

    return best_online_trial_ids


# ---------------------------------------------------------------------------
# Phase 2: Offline training (many-to-many: dataset × agent)
# ---------------------------------------------------------------------------

def run_offline_phase(cfg, context, best_online_trial_ids: dict | None = None) -> None:
    """Execute offline training for every (dataset, offline_agent) pair.

    Args:
        best_online_trial_ids: Mapping from online agent config → best trial id.
            Pass {} or None when running offline-only (no prior online phase).
    """
    offline_list = context["offline_list"]
    dataset_list = context["dataset_list"]
    sanitized_extra_args = context["sanitized_extra_args"]
    storage_url = context["storage_url"]
    is_sweep = context["is_sweep"]
    best_online_trial_ids = best_online_trial_ids or {}

    for dataset_id in dataset_list:
        dataset_name_internal = normalize_agent_name(dataset_id)

        best_id = best_online_trial_ids.get(dataset_id, "0")
        best_trial_path = (
            Path("in/datasets") / cfg.group / cfg.experiment_id / dataset_name_internal / best_id
        )
        yaml_ds_path = cfg.mode.get("dataset_path", None) if hasattr(cfg, "mode") else None

        if best_trial_path.exists() and any(best_trial_path.glob("*.pkl")):
            dataset_path = best_trial_path
        else:
            try:
                dataset_path = resolve_dataset_path(
                    dataset_id,
                    group=cfg.group,
                    experiment_id=cfg.experiment_id,
                    yaml_ds_path=yaml_ds_path,
                )
            except FileNotFoundError as e:
                print(f"Error: {e}")
                sys.exit(1)

        print(f"Using dataset from: {dataset_path}")

        for agent_config in offline_list:
            agent_name_internal = normalize_agent_name(agent_config)
            study_name = get_next_study_name(cfg.group, cfg.experiment_id, agent_name_internal)

            print(f"\n=== Phase: Offline Training ({agent_config}) on Dataset ({dataset_id}) ===")
            target_agent_name = (
                f"{agent_name_internal}_{dataset_name_internal}"
                if len(dataset_list) > 1
                else agent_name_internal
            )

            overrides = build_offline_overrides(
                experiment=cfg.get("experiment_name", ""),
                agent_config=agent_config,
                agent_name=target_agent_name,
                dataset_path=dataset_path,
                local_val=True,
                study_name=study_name,
                extra_args=sanitized_extra_args,
                cfg=cfg,
                dataset_id=dataset_id,
            )

            if is_sweep:
                if "--multirun" not in sanitized_extra_args and "-m" not in sanitized_extra_args:
                    overrides.append("--multirun")
                delete_optuna_study(storage_url, study_name)
                direction = get_sweep_direction(cfg, "offline")
                create_optuna_study(storage_url, study_name, direction=direction)

            run_experiment(overrides)

            if is_sweep:
                promote_best_trial_checkpoint(
                    cfg.group, cfg.experiment_id, target_agent_name, storage_url, study_name
                )


# ---------------------------------------------------------------------------
# Phase 3: Plotting
# ---------------------------------------------------------------------------

def run_plotting_phase(cfg, context) -> None:
    """Run automated plotting after training completes."""
    from src.pipeline.datasets import run_plotting

    site_cfg = getattr(cfg, "site", None)
    run_plotting(
        cfg.experiment_id,
        style=cfg.get("plot_style", None),
        base_experiment=cfg.get("experiment_name", ""),
        site_cfg=site_cfg,
    )


# ---------------------------------------------------------------------------
# Compatibility shim (used by Slurm runner and any legacy call-sites)
# ---------------------------------------------------------------------------

def run_local_training(cfg, context) -> None:
    """Execute all three phases sequentially (setup → online → offline → plot).

    This is the compatibility entry-point preserved for the Slurm runner and
    any tasks that haven't migrated to paradigm-driven dispatch yet.
    Paradigm-aware callers should invoke the individual phase functions directly
    via the runner classes in src/core/paradigm_impls/.
    """
    _setup_output_dirs(cfg)

    best_online_trial_ids: dict = {}
    if not cfg.get("no_online", False):
        best_online_trial_ids = run_online_phase(cfg, context)
    else:
        print("\n=== Skipping Online Training Phase ===")

    if not cfg.get("no_offline", False):
        run_offline_phase(cfg, context, best_online_trial_ids)
    else:
        print("\n=== Skipping Offline Training Phase ===")

    if not cfg.get("no_plot", False):
        run_plotting_phase(cfg, context)
