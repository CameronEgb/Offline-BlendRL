import argparse
import os
import pickle
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def moving_average(a, n=10):
    if len(a) == 0:
        return np.array([])
    if len(a) < n:
        n = max(1, len(a))
    # Pad with the first value to avoid skipping the beginning
    a_padded = np.pad(a, (n-1, 0), mode='edge')
    ret = np.cumsum(a_padded, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_results(experiment_id, runs_dir="out/runs", output_dir="plots", num_envs_override=None):
    # Target directory for this experiment
    exp_path = Path(runs_dir) / experiment_id
    if not exp_path.exists():
        alt_path = Path(runs_dir) / f"exp_{experiment_id}"
        if alt_path.exists(): 
            exp_path = alt_path
        else:
            print(f"Warning: Experiment directory {exp_path} not found. Checking root...")
            exp_path = Path(runs_dir)
        
    output_path = Path(output_dir) / experiment_id
    output_path.mkdir(parents=True, exist_ok=True)
    
    # --- Try to find num_envs for step scaling ---
    num_envs = 1
    hp_path = exp_path / "hyperparameters.txt"
    if hp_path.exists():
        try:
            with open(hp_path, "r") as f:
                for line in f:
                    if "Num Envs:" in line:
                        num_envs = int(line.split(":")[1].strip())
                        break
        except: pass
    
    if num_envs_override is not None:
        num_envs = num_envs_override
        
    print(f"Scanning {exp_path} for experiment {experiment_id} (using num_envs={num_envs})...")
    
    online_shaped = {} 
    online_raw = {}
    online_lengths = {}
    eval_shaped = {}
    eval_raw = {}
    eval_limits = {}

    # Scan for sub-runs
    folders = list(exp_path.glob(f"*{experiment_id}*"))
    if not folders and exp_path.name == experiment_id:
        folders = [d for d in exp_path.iterdir() if d.is_dir()]

    for run_folder in folders:
        if not run_folder.is_dir(): continue
        folder_name = run_folder.name
        
        is_offline = folder_name.startswith("off_")
        clean_method = folder_name.replace(f"_{experiment_id}", "").replace("Seaquest-v4_", "").replace("seaquest_", "")
        
        # 1. Load Continuous Training Data (training_log.pkl)
        pkl_path = run_folder / "checkpoints" / "training_log.pkl"
        if pkl_path.exists():
            try:
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)
                    if data and len(data) >= 2:
                        returns = data[0]
                        lengths = data[1]
                        if returns:
                            online_shaped[clean_method] = returns
                            online_lengths[clean_method] = lengths
                            if len(data) >= 7: online_raw[clean_method] = data[6]
                        else:
                            print(f"  Note: {folder_name} has a pkl file but 0 completed episodes.")
            except Exception as e:
                print(f"  Error loading pkl from {run_folder.name}: {e}")

        # 2. Load Interval Eval Data (results.json)
        json_path = run_folder / "results.json"
        if not json_path.exists():
            json_path = run_folder / "checkpoints" / "results.json"
            
        if json_path.exists():
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                    if not data: continue
                    label = clean_method if is_offline else f"{clean_method} (Online)"
                    eval_limits[label] = [d["data_limit"] for d in data]
                    eval_shaped[label] = [d["avg_reward"] for d in data]
                    eval_raw[label] = [d.get("avg_raw_reward", 0.0) for d in data]
            except Exception as e:
                print(f"  Error loading json from {json_path}: {e}")

    # --- PLOTTING ---
    
    if not online_shaped:
        print("No continuous online data found to plot.")
    else:
        # Produce BOTH step-based and episode-based graphs
        for x_axis in ["steps", "episodes"]:
            plt.figure(figsize=(12, 5))
            
            # Subplot 1: Shaped Returns
            plt.subplot(1, 2, 1)
            for method, returns in online_shaped.items():
                if x_axis == "steps":
                    x = np.cumsum(online_lengths[method]) / float(num_envs)
                    label_x = f"Steps (per Env, total/{num_envs})"
                else:
                    x = np.arange(len(returns))
                    label_x = "Completed Episodes"
                
                n_smooth = min(20, max(1, len(returns)//10))
                y = moving_average(returns, n=n_smooth)
                plt.plot(x, y, label=method)
            plt.xlabel(label_x); plt.ylabel("Return (Shaped)"); plt.title(f"Continuous Training (Shaped, x={x_axis})"); plt.legend(); plt.grid(True)

            # Subplot 2: Raw Returns
            plt.subplot(1, 2, 2)
            if online_raw:
                for method, returns in online_raw.items():
                    if x_axis == "steps":
                        x = np.cumsum(online_lengths[method]) / float(num_envs)
                    else:
                        x = np.arange(len(returns))
                    
                    n_smooth = min(20, max(1, len(returns)//10))
                    y = moving_average(returns, n=n_smooth)
                    plt.plot(x, y, label=method)
                plt.xlabel(label_x); plt.ylabel("Atari Score"); plt.title(f"Continuous Training (Raw, x={x_axis})"); plt.legend(); plt.grid(True)
            else:
                plt.text(0.5, 0.5, "No raw data in pkl", ha='center')
            
            plt.tight_layout()
            plt.savefig(output_path / f"online_performance_{x_axis}.png")
            print(f"Saved online performance plot to {output_path / f'online_performance_{x_axis}.png'}")

    # FIGURE 2: Eval Comparison (Shaped)
    if eval_shaped:
        plt.figure(figsize=(10, 6))
        for label, rewards in eval_shaped.items():
            # For eval, we usually plot against data_limit (steps)
            x = np.array(eval_limits[label]) / float(num_envs)
            plt.plot(x, rewards, marker='o', label=label)
        plt.xlabel(f"Steps (per Env, assuming total/{num_envs})"); plt.ylabel("Avg Eval Return (Shaped)")
        plt.title(f"Evaluation Comparison - Shaped Rewards ({experiment_id})")
        plt.legend(); plt.grid(True); plt.savefig(output_path / "eval_shaped.png")

    # FIGURE 3: Eval Comparison (Raw)
    if eval_raw:
        plt.figure(figsize=(10, 6))
        for label, rewards in eval_raw.items():
            x = np.array(eval_limits[label]) / float(num_envs)
            plt.plot(x, rewards, marker='s', linestyle='--', label=label)
        plt.xlabel(f"Steps (per Env, assuming total/{num_envs})"); plt.ylabel("Avg Atari Score")
        plt.title(f"Evaluation Comparison - Raw Atari Score ({experiment_id})")
        plt.legend(); plt.grid(True); plt.savefig(output_path / "eval_raw.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experimentid", type=str)
    parser.add_argument("--runs_dir", type=str, default="out/runs")
    parser.add_argument("--output_dir", type=str, default="plots")
    parser.add_argument("--num_envs", type=int, default=None)
    args = parser.parse_args()
    plot_results(args.experimentid, args.runs_dir, args.output_dir, args.num_envs)
