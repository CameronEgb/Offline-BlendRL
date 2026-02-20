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
    
    # Use unbuffered output for better logging
    if "python " in command:
        command = command.replace("python ", "python -u ")
    elif "python3 " in command:
        command = command.replace("python3 ", "python -u ")

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
        # Get absolute path for venv sourcing if possible
        cwd = os.getcwd()
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
        sbatch_script += f"cd {cwd}\n"
        sbatch_script += f"if [ -d \"{cwd}/venv\" ]; then\n"
        sbatch_script += f"    source {cwd}/venv/bin/activate\n"
        sbatch_script += "fi\n"
        
        # Comprehensive PYTHONPATH based on working outdated script
        sbatch_script += f"export PYTHONPATH=\"{cwd}:{cwd}/nsfr:{cwd}/neumann:{cwd}/in/envs/seaquest:{cwd}/in/envs/mountaincar:$PYTHONPATH\"\n"
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
    
    # We will save job IDs to a file for precise killing later
    jobids_path = exp_dir / "jobids.txt"
    jobids_file = open(jobids_path, "w")
    
    # Use 'python' and rely on venv sourcing in submit_job
    python_cmd = "python"
    
    online_job_ids = {} # method -> job_id
    
    # 1. Online Training
    for method in online_methods:
        # ... (rest of the loop)
        jid = submit_job(cmd, f"on_{method}_{experiment_id}", local=args.local, log_dir=f"logs/{experiment_id}")
        if jid: 
            online_job_ids[method] = jid
            if not args.local:
                jobids_file.write(f"{jid}\n")
                jobids_file.flush()

    # 2. Offline Training
    for off_method in offline_methods:
        # ... (rest of the loop)
        jid = submit_job(cmd, f"off_{off_method}_{data_source}_{experiment_id}", dependency=dependency, local=args.local, log_dir=f"logs/{experiment_id}")
        if jid and not args.local:
            jobids_file.write(f"{jid}\n")
            jobids_file.flush()
    
    jobids_file.close()

if __name__ == "__main__":
    main()
