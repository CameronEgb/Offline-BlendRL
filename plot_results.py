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
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Subplot 1: Shaped Returns
            ax1 = axes[0]
            lines = []
            labels = []
            for method, returns in online_shaped.items():
                if x_axis == "steps":
                    x = np.cumsum(online_lengths[method]) / float(num_envs)
                    label_x = f"Steps (per Env, total/{num_envs})"
                else:
                    x = np.arange(len(returns))
                    label_x = "Completed Episodes"
                
                n_smooth = min(20, max(1, len(returns)//10))
                y = moving_average(returns, n=n_smooth)
                line, = ax1.plot(x, y, label=method)
                if method not in labels:
                    lines.append(line)
                    labels.append(method)
            ax1.set_xlabel(label_x); ax1.set_ylabel("Return (Shaped)"); ax1.set_title(f"Continuous Training (Shaped)"); ax1.grid(True)

            # Subplot 2: Raw Returns
            ax2 = axes[1]
            if online_raw:
                for method, returns in online_raw.items():
                    if x_axis == "steps":
                        x = np.cumsum(online_lengths[method]) / float(num_envs)
                    else:
                        x = np.arange(len(returns))
                    
                    n_smooth = min(20, max(1, len(returns)//10))
                    y = moving_average(returns, n=n_smooth)
                    ax2.plot(x, y, label=method)
                ax2.set_xlabel(label_x); ax2.set_ylabel("Atari Score"); ax2.set_title(f"Continuous Training (Raw)"); ax2.grid(True)
            else:
                ax2.text(0.5, 0.5, "No raw data in pkl", ha='center')
            
            # Shared legend for continuous training
            fig.legend(lines, labels, loc='lower center', ncol=min(len(labels), 4), bbox_to_anchor=(0.5, -0.05))
            plt.tight_layout(rect=[0, 0.05, 1, 1])
            plt.savefig(output_path / f"online_performance_{x_axis}.png", bbox_inches='tight')
            print(f"Saved online performance plot to {output_path / f'online_performance_{x_axis}.png'}")

    # COMBINED FIGURE: Eval Comparison (Shaped and Raw)
    if eval_shaped:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        lines = []
        labels = []
        
        # Subplot 1: Shaped Eval
        ax1 = axes[0]
        for label, rewards in eval_shaped.items():
            x = np.array(eval_limits[label])
            line, = ax1.plot(x, rewards, marker='o', label=label)
            if label not in labels:
                lines.append(line)
                labels.append(label)
        ax1.set_xlabel("Total Training Steps / Dataset Size"); ax1.set_ylabel("Avg Eval Return (Shaped)")
        ax1.set_title(f"Evaluation Comparison - Shaped Rewards")
        ax1.grid(True)

        # Subplot 2: Raw Eval
        ax2 = axes[1]
        if eval_raw:
            for label, rewards in eval_raw.items():
                x = np.array(eval_limits[label])
                ax2.plot(x, rewards, marker='s', linestyle='--', label=label)
            ax2.set_xlabel("Total Training Steps / Dataset Size"); ax2.set_ylabel("Avg Atari Score")
            ax2.set_title(f"Evaluation Comparison - Raw Atari Score")
            ax2.grid(True)
        else:
            ax2.text(0.5, 0.5, "No raw eval data", ha='center')

        # Shared legend for eval comparison
        fig.legend(lines, labels, loc='lower center', ncol=min(len(labels), 3), bbox_to_anchor=(0.5, -0.05))
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(output_path / "eval_comparison.png", bbox_inches='tight')
        print(f"Saved combined evaluation plot to {output_path / 'eval_comparison.png'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experimentid", type=str)
    parser.add_argument("--runs_dir", type=str, default="out/runs")
    parser.add_argument("--output_dir", type=str, default="plots")
    parser.add_argument("--num_envs", type=int, default=None)
    args = parser.parse_args()
    plot_results(args.experimentid, args.runs_dir, args.output_dir, args.num_envs)
