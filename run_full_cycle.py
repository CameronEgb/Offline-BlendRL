import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Configuration
CLUSTER_PARTITION = "rtx4060ti16g"
CLUSTER_NODES = 1
DATA_DIR = "out/runs" 
DATASET_DIR = "offline_dataset"

def parse_args():
    parser = argparse.ArgumentParser(description="Run Full Cycle Experiments")
    
    # Use environment variables as defaults for all parameters
    parser.add_argument("--experimentid", type=str, default=os.getenv("EXPERIMENT_ID", "exp_001"))
    parser.add_argument("--environment", type=str, default=os.getenv("ENVIRONMENT", "seaquest"))
    parser.add_argument("--online_methods", type=str, default=os.getenv("ONLINE_METHODS", "ppo,blendrl_ppo"))
    parser.add_argument("--online_steps", type=int, default=int(os.getenv("ONLINE_STEPS", "20000000")))
    parser.add_argument("--offline_methods", type=str, default=os.getenv("OFFLINE_METHODS", "iql,blendrl_iql"))
    parser.add_argument("--offline_datasets", type=str, default=os.getenv("OFFLINE_DATASETS", ""))
    parser.add_argument("--offline_epochs", type=int, default=int(os.getenv("OFFLINE_EPOCHS", "10")))
    parser.add_argument("--intervals_count", type=int, default=int(os.getenv("INTERVALS_COUNT", "7")))
    parser.add_argument("--eval_episodes", type=int, default=int(os.getenv("EVAL_EPISODES", "100")))
    
    # Training Hyperparameters
    parser.add_argument("--lr", type=float, default=float(os.getenv("LR", "2.5e-4")))
    parser.add_argument("--logic_lr", type=float, default=float(os.getenv("LOGIC_LR", "2.5e-4")))
    parser.add_argument("--blender_lr", type=float, default=float(os.getenv("BLENDER_LR", "2.5e-4")))
    parser.add_argument("--offline_lr", type=float, default=float(os.getenv("OFFLINE_LR", "3e-4")))
    parser.add_argument("--gamma", type=float, default=float(os.getenv("GAMMA", "0.99")))
    parser.add_argument("--ppo_epochs", type=int, default=int(os.getenv("PPO_EPOCHS", "4")))
    parser.add_argument("--ent_coef", type=float, default=float(os.getenv("ENT_COEF", "0.01")))
    parser.add_argument("--blend_ent_coef", type=float, default=float(os.getenv("BLEND_ENT_COEF", "0.01")))
    parser.add_argument("--iql_tau", type=float, default=float(os.getenv("IQL_TAU", "0.7")))
    parser.add_argument("--iql_beta", type=float, default=float(os.getenv("IQL_BETA", "3.0")))
    parser.add_argument("--batch_size", type=int, default=int(os.getenv("BATCH_SIZE", "256")))
    parser.add_argument("--num_envs", type=int, default=int(os.getenv("NUM_ENVS", "20")))
    parser.add_argument("--num_blend_envs", type=int, default=int(os.getenv("NUM_BLEND_ENVS", "64")))
    parser.add_argument("--num_steps", type=int, default=int(os.getenv("NUM_STEPS", "128")))
    parser.add_argument("--reasoner", type=str, default=os.getenv("REASONER", "nsfr"))
    parser.add_argument("--algorithm", type=str, default=os.getenv("ALGORITHM", "blender"))
    parser.add_argument("--blender_mode", type=str, default=os.getenv("BLENDER_MODE", "logic"))
    parser.add_argument("--blend_function", type=str, default=os.getenv("BLEND_FUNCTION", "softmax"))
    parser.add_argument("--actor_mode", type=str, default=os.getenv("ACTOR_MODE", "hybrid"))
    parser.add_argument("--rules", type=str, default=os.getenv("RULES", "default"))
    
    parser.add_argument("--pretrained", action="store_true", default=os.getenv("PRETRAINED", "false").lower() == "true")
    parser.add_argument("--joint_training", action="store_true", default=os.getenv("JOINT_TRAINING", "false").lower() == "true")
    
    use_large = os.getenv("USE_LARGE_DATASET_PATH", "false").lower() == "true"
    parser.add_argument("--use_large_dataset_path", action="store_true", default=use_large)
    parser.add_argument("--large_dataset_path", type=str, default=os.getenv("LARGE_DATASET_PATH", ""))
    
    parser.add_argument("--local", action="store_true", help="Run experiments locally instead of submitting to cluster")
    
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", "1")))
    return parser.parse_args()

def check_experiment_exists(path):
    return os.path.exists(path)

def ask_user_overwrite(run_name):
    while True:
        choice = input(f"Run '{run_name}' already exists. Overwrite/Rerun? (y/n): ").lower()
        if choice in ['y', 'n']:
            return choice == 'y'

def submit_job(command, job_name, dependency=None, log_dir="logs", partition=CLUSTER_PARTITION, local=False):
    os.makedirs(log_dir, exist_ok=True)
    
    if local:
        if dependency:
            command = f"while kill -0 {dependency} 2>/dev/null; do sleep 5; done; {command}"
            
        log_file = os.path.join(log_dir, f"{job_name}.log")
        print(f"Running locally: {command}")
        with open(log_file, "w") as f:
            proc = subprocess.Popen(command, shell=True, stdout=f, stderr=f, start_new_session=True)
        print(f"Started local process {proc.pid}. Log: {log_file}")
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
        sbatch_script += "if [ -d \"venv\" ]; then\n"
        sbatch_script += "    source venv/bin/activate\n"
        sbatch_script += "fi\n"
        sbatch_script += "export PYTHONPATH=$PYTHONPATH:.\n"
        sbatch_script += f"{command}"
        
        res = subprocess.run(["sbatch"], input=sbatch_script.encode(), capture_output=True)
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
    env_name = args.environment
    
    online_methods = args.online_methods.split(",") if args.online_methods else []
    offline_methods = args.offline_methods.split(",") if args.offline_methods else []
    offline_datasets = args.offline_datasets.split(",") if args.offline_datasets else []
    
    dataset_base_path = args.large_dataset_path if args.use_large_dataset_path and args.large_dataset_path else DATASET_DIR
    
    exp_dir = Path(DATA_DIR) / experiment_id
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
        f.write(f"Entropy Coef: {args.ent_coef}\n")
        f.write(f"Blend Entropy Coef: {args.blend_ent_coef}\n")
        f.write(f"PPO Update Epochs: {args.ppo_epochs}\n")
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
    
    online_job_ids = {} # method -> job_id
    python_cmd = sys.executable if args.local else "./venv/bin/python"
    
    # 1. Online Training
    for method in online_methods:
        method = method.strip()
        if not method: continue
        
        run_name = f"{env_name}_{method}_{experiment_id}"
        run_path = os.path.join(DATA_DIR, experiment_id, run_name)
        
        if check_experiment_exists(run_path):
            if ask_user_overwrite(run_name):
                print(f"Clearing old run data: {run_path}")
                import shutil
                shutil.rmtree(run_path, ignore_errors=True)
                if args.use_large_dataset_path and args.large_dataset_path:
                    dataset_path = os.path.join(args.large_dataset_path, experiment_id, run_name)
                else:
                    dataset_path = os.path.join(DATA_DIR, experiment_id, run_name, "offline_dataset")
                if os.path.exists(dataset_path):
                    shutil.rmtree(dataset_path, ignore_errors=True)
            else:
                continue
        
        script_name = "train_neuralppo.py" if method == "ppo" else "train_blenderl.py"
        if method not in ["ppo", "blendrl_ppo"]:
            print(f"Warning: Unknown method {method}")
            continue
            
        cmd = f"{python_cmd} {script_name} --env_name {env_name} --total_timesteps {args.online_steps} --seed {args.seed} --save_dataset --dataset_path {dataset_base_path} --run_id {run_name} --exp_id {experiment_id} --intervals {args.intervals_count} --eval_episodes {args.eval_episodes} --learning_rate {args.lr} --gamma {args.gamma} --ent_coef {args.ent_coef} --num_envs {args.num_envs} --num_steps {args.num_steps} --reasoner {args.reasoner} --algorithm {args.algorithm} --blender_mode {args.blender_mode} --blend_function {args.blend_function} --actor_mode {args.actor_mode} --rules {args.rules}"
        if args.pretrained: cmd += " --pretrained"
        if args.joint_training: cmd += " --joint_training"
            
        if method == "ppo":
            cmd += f" --update_epochs {args.ppo_epochs}"
        elif method == "blendrl_ppo":
            cmd += f" --logic_learning_rate {args.logic_lr} --blender_learning_rate {args.blender_lr} --blend_ent_coef {args.blend_ent_coef} --num_blend_envs {args.num_blend_envs}"
        
        jid = submit_job(cmd, f"on_{method}_{experiment_id}", local=args.local, log_dir=f"logs/{experiment_id}")
        if jid: online_job_ids[method] = jid

    # 2. Offline Training
    for off_method in offline_methods:
        off_method = off_method.strip()
        if not off_method: continue
        for data_source in offline_datasets:
            data_source = data_source.strip()
            if not data_source: continue
            
            run_name = f"off_{off_method}_{data_source}_{experiment_id}"
            run_path = os.path.join(DATA_DIR, experiment_id, run_name)
            
            if check_experiment_exists(run_path) and not ask_user_overwrite(run_name):
                continue
            if check_experiment_exists(run_path):
                import shutil
                shutil.rmtree(run_path, ignore_errors=True)

            dataset_run_id = f"{env_name}_{data_source}_{experiment_id}"
            script = "train_iql.py" if off_method == "iql" else "train_blendrl_iql.py"
            dependency = online_job_ids.get(data_source)

            cmd = f"{python_cmd} {script} --env_name {env_name} --dataset_path {dataset_base_path} --dataset_run_name {dataset_run_id} --intervals {args.intervals_count} --epochs_per_interval {args.offline_epochs} --seed {args.seed} --run_id {run_name} --exp_id {experiment_id} --eval_episodes {args.eval_episodes} --learning_rate {args.offline_lr} --gamma {args.gamma} --tau {args.iql_tau} --beta {args.iql_beta} --batch_size {args.batch_size} --algorithm {args.algorithm} --blender_mode {args.blender_mode} --blend_function {args.blend_function} --actor_mode {args.actor_mode} --rules {args.rules} --reasoner {args.reasoner}"
            submit_job(cmd, f"off_{off_method}_{data_source}_{experiment_id}", dependency=dependency, local=args.local, log_dir=f"logs/{experiment_id}")

if __name__ == "__main__":
    main()
