import argparse
import os
import subprocess
import sys
import time
import yaml
from pathlib import Path

# Configuration
CLUSTER_PARTITION = "rtx4060ti16g"
CLUSTER_NODES = 1
DATA_DIR = "results/experiments" 
DATASET_DIR = "results/datasets"

def parse_args():
    # Pre-parse to check for a config file
    initial_parser = argparse.ArgumentParser(add_help=False)
    initial_parser.add_argument("--config", type=str, help="Path to a YAML configuration file")
    initial_args, remaining_args = initial_parser.parse_known_args()
    
    config_values = {}
    if initial_args.config and os.path.exists(initial_args.config):
        print(f"Loading configuration from: {initial_args.config}")
        with open(initial_args.config, "r") as f:
            config_values = yaml.safe_load(f) or {}

    def get_default(key, default_val):
        # Precedence: config file > environment variable > hardcoded default
        if key in config_values:
            return config_values[key]
        env_val = os.getenv(key.upper())
        if env_val is not None:
            if isinstance(default_val, bool):
                return env_val.lower() == "true"
            if isinstance(default_val, int):
                return int(env_val)
            if isinstance(default_val, float):
                return float(env_val)
            return env_val
        return default_val

    parser = argparse.ArgumentParser(description="Run Full Cycle Experiments")
    parser.add_argument("--config", type=str, help="Path to a YAML configuration file")
    
    parser.add_argument("--experimentid", type=str, default=get_default("experimentid", "exp_001"))
    parser.add_argument("--environment", type=str, default=get_default("environment", "seaquest"))
    parser.add_argument("--online_methods", type=str, default=get_default("online_methods", "ppo,blendrl_ppo"))
    parser.add_argument("--online_steps", type=int, default=get_default("online_steps", 20000000))
    parser.add_argument("--offline_methods", type=str, default=get_default("offline_methods", "iql,blendrl_iql"))
    parser.add_argument("--offline_datasets", type=str, default=get_default("offline_datasets", ""))
    parser.add_argument("--offline_epochs", type=int, default=get_default("offline_epochs", 10))
    parser.add_argument("--intervals_count", type=int, default=get_default("intervals_count", 7))
    parser.add_argument("--eval_episodes", type=int, default=get_default("eval_episodes", 100))
    
    # Training Hyperparameters
    parser.add_argument("--lr", type=float, default=get_default("lr", 2.5e-4))
    parser.add_argument("--logic_lr", type=float, default=get_default("logic_lr", 2.5e-4))
    parser.add_argument("--blender_lr", type=float, default=get_default("blender_lr", 2.5e-4))
    parser.add_argument("--offline_lr", type=float, default=get_default("offline_lr", 3e-4))
    parser.add_argument("--gamma", type=float, default=get_default("gamma", 0.99))
    parser.add_argument("--gae_lambda", type=float, default=get_default("gae_lambda", 0.95))
    parser.add_argument("--clip_coef", type=float, default=get_default("clip_coef", 0.1))
    parser.add_argument("--ppo_epochs", type=int, default=get_default("ppo_epochs", 4))
    parser.add_argument("--num_minibatches", type=int, default=get_default("num_minibatches", 4))
    parser.add_argument("--ent_coef", type=float, default=get_default("ent_coef", 0.01))
    parser.add_argument("--blend_ent_coef", type=float, default=get_default("blend_ent_coef", 0.01))
    parser.add_argument("--max_grad_norm", type=float, default=get_default("max_grad_norm", 0.5))
    parser.add_argument("--iql_tau", type=float, default=get_default("iql_tau", 0.7))
    parser.add_argument("--iql_beta", type=float, default=get_default("iql_beta", 3.0))
    parser.add_argument("--batch_size", type=int, default=get_default("batch_size", 256))
    parser.add_argument("--num_envs", type=int, default=get_default("num_envs", 20))
    parser.add_argument("--num_blend_envs", type=int, default=get_default("num_blend_envs", 64))
    parser.add_argument("--num_steps", type=int, default=get_default("num_steps", 128))
    parser.add_argument("--reasoner", type=str, default=get_default("reasoner", "nsfr"))
    parser.add_argument("--algorithm", type=str, default=get_default("algorithm", "blender"))
    parser.add_argument("--blender_mode", type=str, default=get_default("blender_mode", "logic"))
    parser.add_argument("--blend_function", type=str, default=get_default("blend_function", "softmax"))
    parser.add_argument("--actor_mode", type=str, default=get_default("actor_mode", "hybrid"))
    parser.add_argument("--rules", type=str, default=get_default("rules", "default"))
    
    parser.add_argument("--pretrained", action="store_true", default=get_default("pretrained", False))
    parser.add_argument("--joint_training", action="store_true", default=get_default("joint_training", False))
    
    DEFAULT_LARGE_DATASET_PATH = "" 
    
    parser.add_argument("--use_large_dataset_path", action="store_true", default=get_default("use_large_dataset_path", True))
    parser.add_argument("--large_dataset_path", type=str, default=get_default("large_dataset_path", DEFAULT_LARGE_DATASET_PATH))
    
    parser.add_argument("--local", action="store_true", help="Run experiments locally instead of submitting to cluster")
    parser.add_argument("--recover", action="store_true", help="Recover training from last checkpoint")
    parser.add_argument("--no_overwrite", action="store_true", help="Automatically overwrite existing data without asking")
    
    parser.add_argument("--group", type=str, default=get_default("group", ""), help="Optional group name to organize results (e.g., results/experiments/group/exp_id)")
    parser.add_argument("--seed", type=int, default=get_default("seed", 1))
    parser.add_argument("--save_dataset", action="store_true", default=get_default("save_dataset", True))
    parser.add_argument("--no_save_dataset", action="store_false", dest="save_dataset", help="Disable dataset saving")
    return parser.parse_args()


def check_experiment_exists(path):
    return os.path.exists(path)

def ask_user_overwrite(run_name, no_overwrite=False):
    if no_overwrite:
        return True
    while True:
        choice = input(f"Run '{run_name}' already exists. Overwrite/Rerun? (y/n): ").lower()
        if choice in ['y', 'n']:
            return choice == 'y'

def submit_job(command, job_name, dependency=None, log_dir="logs", partition=CLUSTER_PARTITION, local=False):
    os.makedirs(log_dir, exist_ok=True)
    
    # Use unbuffered output for better logging
    if "python3 " in command:
        command = command.replace("python3 ", "python3 -u ")
    elif "python " in command:
        command = command.replace("python ", "python -u ")

    if local:
        log_file = os.path.join(log_dir, f"{job_name}.log")
        
        # Build the command string
        if dependency:
            # We still need a shell to handle the dependency check loop
            full_command = f"while kill -0 {dependency} 2>/dev/null; do sleep 5; done; {command}"
        else:
            full_command = command
            
        print(f"Running locally: {full_command}")
        print(f"Logging to: {log_file}")
        
        # Open log file in append mode to avoid overwriting issues if multiple things touch it
        # and to ensure the handle is valid
        log_f = open(log_file, "a")
        
        # Spawn the process. By passing the file handle to stdout/stderr, 
        # subprocess handles the duplication and ensures the FDs are valid in the child.
        # We also pass DEVNULL to stdin to completely detach from the TTY.
        proc = subprocess.Popen(
            full_command, 
            shell=True, 
            stdout=log_f, 
            stderr=subprocess.STDOUT, 
            stdin=subprocess.DEVNULL,
            start_new_session=True
        )
        
        # The child has its own copy of the FD now, we can close ours.
        log_f.close()
        
        print(f"Started local process {proc.pid}.")
        return str(proc.pid)
    else:
        sbatch_script = f"""#!/bin/bash
#SBATCH --partition={partition}
#SBATCH --nodes={CLUSTER_NODES}
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/{job_name}_%j.out
#SBATCH --error={log_dir}/{job_name}_%j.err
"""
        if dependency:
            sbatch_script += f"#SBATCH --dependency=afterok:{dependency}\n"
        
        sbatch_script += "\n"
        sbatch_script += "source venv/bin/activate\n"
        
        # Debug info
        sbatch_script += "echo \"Running on host $(hostname) with python $(which python3) version $(python3 --version)\"\n"
        
        # Comprehensive PYTHONPATH
        sbatch_script += "export PYTHONPATH=\".:nsfr:neumann:in/envs/seaquest:in/envs/mountaincar:$PYTHONPATH\"\n"
        sbatch_script += f"{command}"
        
        res = subprocess.run(["sbatch"], input=sbatch_script.encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0:
            job_id = res.stdout.decode().strip().split(" ")[-1]
            print(f"Submitted job {job_id} ({job_name})")
            return job_id
        else:
            print(f"Error submitting job: {res.stderr.decode()}")
            return None

def main():
    args = parse_args()
    
    experiment_id = args.experimentid
    group = args.group
    
    # Support listName/expID format for surgical reruns
    if "/" in experiment_id:
        parts = experiment_id.split("/")
        group = "/".join(parts[:-1])
        experiment_id = parts[-1]
    
    env_name = args.environment
    
    # Determine the base bucket for organization
    bucket = group if group else experiment_id
    
    # Redefine paths based on bucket
    global DATA_DIR, DATASET_DIR
    DATA_DIR = os.path.join("results/experiments", bucket)
    DATASET_DIR = os.path.join("results/datasets", bucket)
    LOG_BASE = os.path.join("results/logs", bucket)
    
    online_methods = args.online_methods.split(",") if args.online_methods else []
    offline_methods = args.offline_methods.split(",") if args.offline_methods else []
    offline_datasets = args.offline_datasets.split(",") if args.offline_datasets else []
    rulesets = args.rules.split(",") if args.rules else ["default"]
    
    dataset_root = args.large_dataset_path if args.use_large_dataset_path and args.large_dataset_path else "results/datasets"
    
    # We still want to save a summary in the specific experiment folder if it's different from the bucket
    exp_dir = Path(DATA_DIR)
    if group and experiment_id != bucket:
        exp_dir = exp_dir / experiment_id
    
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    with open(exp_dir / "hyperparameters.txt", "w") as f:
        f.write(f"====================================================\n")
        f.write(f"EXPERIMENT SUMMARY: {experiment_id}\n")
        f.write(f"====================================================\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Environment: {env_name}\n")
        f.write(f"Seed: {args.seed}\n\n")

        f.write(f"--- GLOBAL ALGORITHM SETTINGS ---\n")
        f.write(f"Algorithm: {args.algorithm}\n")
        f.write(f"Reasoner: {args.reasoner}\n")
        f.write(f"Actor Mode: {args.actor_mode}\n")
        f.write(f"Blender Mode: {args.blender_mode}\n")
        f.write(f"Blend Function: {args.blend_function}\n")
        f.write(f"Rules: {args.rules}\n")
        f.write(f"Pretrained: {args.pretrained}\n")
        f.write(f"Joint Training: {args.joint_training}\n\n")

        f.write(f"--- ONLINE TRAINING CONFIGURATION ---\n")
        f.write(f"Methods: {args.online_methods}\n")
        f.write(f"Steps per Method: {args.online_steps}\n")
        f.write(f"Num Envs: {args.num_envs}\n")
        f.write(f"Num Steps: {args.num_steps}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Logic LR: {args.logic_lr}\n")
        f.write(f"Blender LR: {args.blender_lr}\n")
        f.write(f"Gamma: {args.gamma}\n")
        f.write(f"GAE Lambda: {args.gae_lambda}\n")
        f.write(f"Entropy Coef: {args.ent_coef}\n")
        f.write(f"Blend Entropy Coef: {args.blend_ent_coef}\n")
        f.write(f"PPO Update Epochs: {args.ppo_epochs}\n")
        f.write(f"Num Minibatches: {args.num_minibatches}\n")
        f.write(f"Clip Coef: {args.clip_coef}\n")
        f.write(f"Evaluation Intervals: {args.intervals_count}\n")
        f.write(f"Eval Episodes per Interval: {args.eval_episodes}\n")
        f.write(f"Save Dataset: True\n\n")

        f.write(f"--- OFFLINE TRAINING CONFIGURATION ---\n")
        f.write(f"Methods: {args.offline_methods}\n")
        f.write(f"Dataset Sources: {args.offline_datasets}\n")
        f.write(f"Learning Rate: {args.offline_lr}\n")
        f.write(f"Gamma: {args.gamma}\n")
        f.write(f"IQL Tau: {args.iql_tau}\n")
        f.write(f"IQL Beta: {args.iql_beta}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Epochs per Interval: {args.offline_epochs}\n")
        f.write(f"Intervals: {args.intervals_count}\n")
        f.write(f"Eval Episodes per Interval: {args.eval_episodes}\n")
        f.write(f"====================================================\n")

    print(f"--- Starting Full Cycle: {experiment_id} ---")
    
    jobids_path = exp_dir / "jobids.txt"
    jobids_file = open(jobids_path, "w")
    python_cmd = "python3"
    
    # job_ids mapping: "method_ruleset" -> job_id
    online_job_ids = {} 
    
    # 1. Online Training
    for method in online_methods:
        method = method.strip()
        if not method: continue

        # Determine which rulesets to run for this method
        method_rulesets = [rulesets[0]] if method == "ppo" else rulesets

        for rs in method_rulesets:
            rs = rs.strip()
            
            # New Hierarchy logic
            actual_exp_id = f"{bucket}/online/{rs}"
            run_name = f"{method}_{experiment_id}" if group else method
            run_path = os.path.join("results/experiments", actual_exp_id, run_name)

            # Use unique job name for logging
            job_name = f"on_{method}_{rs}_{experiment_id}"
            log_dir = f"results/logs/{actual_exp_id}"

            if check_experiment_exists(run_path):
                if ask_user_overwrite(run_name, no_overwrite=args.no_overwrite):
                    import shutil
                    shutil.rmtree(run_path, ignore_errors=True)
                    import glob
                    for f in glob.glob(f"{log_dir}/{job_name}*"):
                        try: os.remove(f)
                        except: pass
                else:
                    print(f"Skipping online: {method} on ruleset {rs}")
                    online_job_ids[f"{method}_{rs}"] = None
                    continue

            # Localized Datasets (C)
            dataset_abs_path = os.path.join(run_path, "dataset")
            script_name = "train_neuralppo.py" if method == "ppo" else "train_blenderl.py"
            cmd = f"{python_cmd} {script_name} --env_name {env_name} --total_timesteps {args.online_steps} --seed {args.seed} --run_id {run_name} --exp_id {actual_exp_id} --intervals {args.intervals_count} --eval_episodes {args.eval_episodes} --learning_rate {args.lr} --gamma {args.gamma} --ent_coef {args.ent_coef} --num_envs {args.num_envs} --num_steps {args.num_steps} --batch_size {args.batch_size} --reasoner {args.reasoner} --algorithm {args.algorithm} --blender_mode {args.blender_mode} --blend_function {args.blend_function} --actor_mode {args.actor_mode} --rules {rs} --max_grad_norm {args.max_grad_norm}"
            if args.save_dataset: cmd += f" --save_dataset --dataset_path {dataset_abs_path}"
            if args.pretrained: cmd += " --pretrained"
            if args.joint_training: cmd += " --joint_training"
            if args.recover: cmd += " --recover"

            # Unified hyperparam flags
            ppo_flags = f" --update_epochs {args.ppo_epochs} --num_minibatches {args.num_minibatches} --gae_lambda {args.gae_lambda} --clip_coef {args.clip_coef}"
            if method == "ppo":
                cmd += ppo_flags
            elif method == "blendrl_ppo":
                cmd += f" --logic_learning_rate {args.logic_lr} --blender_learning_rate {args.blender_lr} --blend_ent_coef {args.blend_ent_coef} --num_blend_envs {args.num_blend_envs}" + ppo_flags

            jid = submit_job(cmd, job_name, local=args.local, log_dir=log_dir)
            if jid: 
                online_job_ids[f"{method}_{rs}"] = jid
                if not args.local:
                    jobids_file.write(f"{jid}\n")
                    jobids_file.flush()

    # 2. Offline Training
    for off_method in offline_methods:
        off_method = off_method.strip()
        if not off_method: continue

        for data_source_method in offline_datasets:
            data_source_method = data_source_method.strip()
            if not data_source_method: continue

            # PPO datasets can be used to train/eval offline against any ruleset
            source_rulesets = rulesets

            for rs in source_rulesets:
                rs = rs.strip()
                
                actual_exp_id = f"{bucket}/offline/{rs}"
                run_name = f"{off_method}_{data_source_method}_{experiment_id}" if group else f"{off_method}_{data_source_method}"
                run_path = os.path.join("results/experiments", actual_exp_id, run_name)
                
                job_name = f"off_{off_method}_{data_source_method}_{rs}_{experiment_id}"
                log_dir = f"results/logs/{actual_exp_id}"

                if check_experiment_exists(run_path):
                    if ask_user_overwrite(run_name):
                        import shutil
                        shutil.rmtree(run_path, ignore_errors=True)
                        import glob
                        for f in glob.glob(f"{log_dir}/{job_name}*"):
                            try: os.remove(f)
                            except: pass
                    else:
                        continue

                # Localized Datasets Mapping (C)
                # If source is standard ppo, it only exists in the first ruleset's folder
                source_rs = rulesets[0] if data_source_method == "ppo" else rs
                
                # Try preferred path (within current bucket)
                online_dataset_abs_path = os.path.join("results/experiments", bucket, "online", source_rs, f"{data_source_method}_{experiment_id}" if group else data_source_method, "dataset")
                
                # WORKAROUND: If not found, try the environment-named bucket (common for reusing online data)
                if not os.path.exists(online_dataset_abs_path):
                    alt_path = os.path.join("results/experiments", env_name, "online", source_rs, f"{data_source_method}_{experiment_id}" if group else data_source_method, "dataset")
                    if os.path.exists(alt_path):
                        print(f"Dataset not found in bucket '{bucket}', using alternate path: {alt_path}")
                        online_dataset_abs_path = alt_path
                    else:
                        # Final attempt: check without the experiment_id suffix even if group is present
                        alt_path_2 = os.path.join("results/experiments", env_name, "online", source_rs, data_source_method, "dataset")
                        if os.path.exists(alt_path_2):
                            print(f"Dataset found in alternate path (no suffix): {alt_path_2}")
                            online_dataset_abs_path = alt_path_2

                script = "train_iql.py" if off_method == "iql" else "train_blendrl_iql.py"
                dependency = online_job_ids.get(f"{data_source_method}_{rs}")

                # Build offline command
                off_cmd = f"{python_cmd} {script} --env_name {env_name} --dataset_path {online_dataset_abs_path} --run_id {run_name} --exp_id {actual_exp_id} --intervals {args.intervals_count} --epochs_per_interval {args.offline_epochs} --eval_episodes {args.eval_episodes} --learning_rate {args.offline_lr} --gamma {args.gamma} --batch_size {args.batch_size} --tau {args.iql_tau} --beta {args.iql_beta} --reasoner {args.reasoner} --algorithm {args.algorithm} --blender_mode {args.blender_mode} --blend_function {args.blend_function} --actor_mode {args.actor_mode} --rules {rs} --seed {args.seed} --total_timesteps {args.online_steps}"
                
                # Add BlendRL-IQL specific flags
                if off_method == "blendrl_iql":
                    off_cmd += f" --logic_learning_rate {args.logic_lr} --blender_learning_rate {args.blender_lr} --blend_ent_coef {args.blend_ent_coef}"
                
                jid = submit_job(off_cmd, job_name, dependency=dependency, local=args.local, log_dir=log_dir)
                if jid and not args.local:
                    jobids_file.write(f"{jid}\n")
                    jobids_file.flush()
    
    jobids_file.close()

if __name__ == "__main__":
    main()
