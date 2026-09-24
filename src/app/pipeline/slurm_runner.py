"""Slurm training phase runner.

Generates and submits training jobs to Slurm cluster.
"""

import shlex
import sys
from pathlib import Path

from src.app.pipeline.commands import build_method_overrides, get_sweep_direction
from src.app.pipeline.config import normalize_agent_name
from src.app.pipeline.datasets import ensure_online_dataset_path, fast_purge_dir, resolve_dataset_path
from src.app.pipeline.optuna_utils import create_optuna_study, delete_optuna_study, get_next_study_name
from src.app.pipeline.runtime import get_shell_env_block, get_shell_python_cmd
from src.app.pipeline.slurm import generate_sbatch_header, generate_sbatch_script, submit_sbatch


def _resolve_dataset_for_method(method_name, method_cfg, cfg):
    """Resolve the dataset path for an offline method."""
    explicit_ds = method_cfg.get("dataset_path") or cfg.get("dataset_path")
    if explicit_ds and Path(explicit_ds).exists():
        return Path(explicit_ds)

    env_dataset = None
    env_name = None
    if hasattr(cfg, "env"):
        env_dataset = cfg.env.get("dataset_name", None)
        env_name = cfg.env.get("name", None)
        if env_dataset:
            try:
                return resolve_dataset_path(
                    dataset_id=str(env_dataset).replace(".npz", ""),
                    group=env_name or cfg.get("group", ""),
                    experiment_id=cfg.get("experiment_id", ""),
                    yaml_ds_path=str(explicit_ds) if explicit_ds else None,
                )
            except FileNotFoundError:
                pass

    ds_root = Path("in/datasets") / cfg.group / cfg.experiment_id
    if ds_root.exists():
        return ds_root

    raise FileNotFoundError(
        f"Cannot resolve dataset for method '{method_name}'. "
        f"No dataset_name in env config and no datasets found at {ds_root}."
    )


def run_slurm_training(cfg, context):
    methods = context.get("methods", {})
    sanitized_extra_args = context.get("sanitized_extra_args", [])
    storage_url = context.get("storage_url", None)
    is_sweep = context.get("is_sweep", False)
    site_cfg = getattr(cfg, "site", None)
    paradigm = cfg.get("paradigm", "offline_rl")

    log_dir = Path("results/logs/slurm") / cfg.group / cfg.experiment_id
    fast_purge_dir(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.get("recover", False):
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

    cfg_consolidate = cfg.get("consolidate", None)
    site_consolidate = getattr(site_cfg, "consolidate", False) if site_cfg else False
    should_consolidate = cfg_consolidate if cfg_consolidate is not None else site_consolidate

    if should_consolidate:
        print(f"\n=== Preparing Consolidated Slurm Job ({cfg.experiment_id}) ===")
        job_name = f"all_{cfg.experiment_id}"
        script_content = generate_sbatch_header(job_name=job_name, log_dir=log_dir, cfg=cfg, is_consolidated=True)
        script_content += "\n" + get_shell_env_block(site_cfg) + "\n"
        python_cmd = get_shell_python_cmd(site_cfg)

        for method_name, method_cfg in methods.items():
            agent_name = normalize_agent_name(method_name)
            dataset_path = None
            
            if paradigm in ("offline_rl", "supervised"):
                try:
                    dataset_path = _resolve_dataset_for_method(method_name, method_cfg, cfg)
                except FileNotFoundError as e:
                    print(f"Error: {e}")
                    sys.exit(1)
            elif paradigm == "online_rl":
                dataset_path_str, has_pkl = ensure_online_dataset_path(
                    group=cfg.group,
                    experiment_id=cfg.experiment_id,
                    agent_name_internal=agent_name,
                    is_sweep=is_sweep,
                )
                if has_pkl:
                    script_content += f'echo "Dataset already exists for {method_name}. Skipping."\n\n'
                    continue
                dataset_path = dataset_path_str

            study_name = get_next_study_name(cfg.group, cfg.experiment_id, agent_name) if is_sweep else None
            
            cmd_args = build_method_overrides(
                method_name=method_name,
                method_cfg=method_cfg,
                dataset_path=dataset_path,
                extra_args=sanitized_extra_args,
                cfg=cfg,
                study_name=study_name,
            )
            
            if is_sweep:
                direction = get_sweep_direction(cfg, paradigm)
                create_optuna_study(storage_url, study_name, direction=direction)
                if "--multirun" not in sanitized_extra_args and "-m" not in sanitized_extra_args:
                    cmd_args.append("--multirun")

            train_cmd = " ".join(shlex.quote(arg) for arg in cmd_args)
            script_content += f'echo "=== [Phase: Training] {method_name} ==="\n'
            script_content += f"{python_cmd} src/app/train.py {train_cmd}\n\n"

            if is_sweep:
                storage_arg = storage_url if storage_url else ""
                script_content += f'echo "=== Promoting Winning Checkpoint for {method_name} ==="\n'
                script_content += f"{python_cmd} -c \"from src.app.pipeline.optuna_utils import promote_best_trial_checkpoint; promote_best_trial_checkpoint('{cfg.group}', '{cfg.experiment_id}', '{agent_name}', '{storage_arg}', '{study_name}')\"\n\n"

        if not cfg.get("no_plot", False):
            plot_cmd = f"{python_cmd} plot/manager.py {cfg.group}/{cfg.experiment_id}"
            if cfg.get("plot_style", None):
                plot_cmd += f" --style {cfg.get('plot_style', None)}"
            if cfg.get("use_cache", False):
                plot_cmd += " --use-cache"
            script_content += 'echo "=== [Generating Final Plots] ==="\n'
            script_content += f"{plot_cmd}\n\n"

        slurm_file = log_dir / f"consolidated_{cfg.experiment_id}.slurm"
        with open(slurm_file, "w") as f:
            f.write(script_content)

        print(f"Submitting Consolidated Slurm Job: {slurm_file}")
        job_id = submit_sbatch(script_content)
        return [job_id] if job_id else [], [], True

    job_ids = []
    submitted_idx = 0
    total_jobs = len(methods)

    for method_name, method_cfg in methods.items():
        agent_name = normalize_agent_name(method_name)
        dataset_path = None
        
        if paradigm in ("offline_rl", "supervised"):
            try:
                dataset_path = _resolve_dataset_for_method(method_name, method_cfg, cfg)
            except FileNotFoundError as e:
                print(f"Error: {e}")
                sys.exit(1)
        elif paradigm == "online_rl":
            dataset_path_str, has_pkl = ensure_online_dataset_path(
                group=cfg.group,
                experiment_id=cfg.experiment_id,
                agent_name_internal=agent_name,
                is_sweep=is_sweep,
            )
            if has_pkl:
                print(f"Dataset already exists for {method_name}. Skipping.", flush=True)
                continue
            dataset_path = dataset_path_str

        job_name = f"{agent_name}_{cfg.experiment_id}"
        study_name = get_next_study_name(cfg.group, cfg.experiment_id, agent_name) if is_sweep else None
        
        cmd_args = build_method_overrides(
            method_name=method_name,
            method_cfg=method_cfg,
            dataset_path=dataset_path,
            extra_args=sanitized_extra_args,
            cfg=cfg,
            study_name=study_name,
        )

        script_content = generate_sbatch_header(
            job_name=job_name,
            log_dir=log_dir,
            cfg=cfg,
            is_consolidated=False,
        )
        script_content += "\n" + get_shell_env_block(site_cfg) + "\n"
        python_cmd = get_shell_python_cmd(site_cfg)

        if is_sweep:
            direction = get_sweep_direction(cfg, paradigm)
            create_optuna_study(storage_url, study_name, direction=direction)
            if "--multirun" not in sanitized_extra_args and "-m" not in sanitized_extra_args:
                cmd_args.append("--multirun")

        train_cmd = " ".join(shlex.quote(arg) for arg in cmd_args)
        script_content += f'echo "=== [Phase: Training] {method_name} ==="\n'
        script_content += f"{python_cmd} src/app/train.py {train_cmd}\n\n"

        if is_sweep:
            storage_arg = storage_url if storage_url else ""
            script_content += f'echo "=== Promoting Winning Checkpoint for {method_name} ==="\n'
            script_content += f"{python_cmd} -c \"from src.app.pipeline.optuna_utils import promote_best_trial_checkpoint; promote_best_trial_checkpoint('{cfg.group}', '{cfg.experiment_id}', '{agent_name}', '{storage_arg}', '{study_name}')\"\n\n"

        slurm_file = log_dir / f"{job_name}.slurm"
        with open(slurm_file, "w") as f:
            f.write(script_content)

        submitted_idx += 1
        print(f"[{submitted_idx}/{total_jobs}] Submitting [{method_name}] ...", end="", flush=True)
        job_id = submit_sbatch(script_content)
        if job_id:
            job_ids.append(job_id)

    if not cfg.get("no_plot", False) and job_ids:
        plot_job_name = f"plot_{cfg.experiment_id}"
        dependency_str = ":".join(job_ids)
        plot_header = generate_sbatch_header(
            job_name=plot_job_name,
            log_dir=log_dir,
            cfg=cfg,
            dependency=dependency_str,
        )
        python_cmd = get_shell_python_cmd(site_cfg)
        plot_cmd = f"{python_cmd} plot/manager.py {cfg.group}/{cfg.experiment_id}"
        if cfg.get("plot_style", None):
            plot_cmd += f" --style {cfg.get('plot_style', None)}"
        if cfg.get("use_cache", False):
            plot_cmd += " --use-cache"
        plot_content = (
            plot_header
            + "\n"
            + get_shell_env_block(site_cfg)
            + f'\n\necho "=== [Generating Final Plots] ==="\n{plot_cmd}\n'
        )
        plot_slurm_file = log_dir / f"{plot_job_name}.slurm"
        with open(plot_slurm_file, "w") as f:
            f.write(plot_content)
        print(f"[Post-Process] Submitting Plotting Job (dependent on {len(job_ids)} jobs) ...", end="", flush=True)
        submit_sbatch(plot_content)

    return job_ids, [], False
