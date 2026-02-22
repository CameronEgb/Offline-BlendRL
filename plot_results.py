import argparse
import os
import pickle
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def moving_average(a, n=100) :
    if len(a) < n:
        return a
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_results(experiment_id, runs_dir="out/runs", output_dir="plots"):
    # Each experiment now has its own subdirectory in out/runs
    exp_path = Path(runs_dir) / experiment_id
    if not exp_path.exists():
        alt_path = Path(runs_dir) / f"exp_{experiment_id}"
        if alt_path.exists():
            exp_path = alt_path
        else:
            print(f"Warning: Experiment directory {exp_path} not found.")
            exp_path = Path(runs_dir)
        
    output_path = Path(output_dir) / experiment_id
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning {exp_path} for experiment {experiment_id}...")
    
    online_data = {} # method -> (steps, returns)
    eval_data = {}   # method -> (limits, rewards) - shaped
    eval_raw_data = {} # method -> (limits, rewards) - raw
    
    for run_folder in exp_path.glob(f"*{experiment_id}*"):
        if not run_folder.is_dir():
            continue
            
        folder_name = run_folder.name
        print(f"Found run: {folder_name}")
        is_offline = folder_name.startswith("off_")
        
        # Heuristic to clean method name
        clean_method = folder_name.replace(f"_{experiment_id}", "")
        clean_method = clean_method.replace("Seaquest-v4_", "").replace("seaquest_", "")
        
        # Online Data
        pkl_path = run_folder / "checkpoints" / "training_log.pkl"
        if pkl_path.exists():
            try:
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)
                    if data and len(data) >= 2:
                        returns, lengths = data[0], data[1]
                        if returns:
                            online_data[clean_method] = (returns, lengths)
            except: pass

        # Eval Data
        json_path = run_folder / "results.json"
        if json_path.exists():
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                    rewards = [d["avg_reward"] for d in data]
                    raw_rewards = [d.get("avg_raw_reward", 0.0) for d in data]
                    limits = [d["data_limit"] for d in data]
                    
                    label = clean_method if is_offline else f"{clean_method} (Online Eval)"
                    eval_data[label] = (limits, rewards)
                    if any(r > 0 for r in raw_rewards):
                        eval_raw_data[label] = (limits, raw_rewards)
            except: pass

    # Plot Online Continuous Training
    if online_data:
        plt.figure(figsize=(10, 6))
        for method, (returns, lengths) in online_data.items():
            steps = np.cumsum(lengths)
            n_smooth = min(50, max(1, len(returns)//4))
            if n_smooth > 1:
                smoothed = moving_average(returns, n=n_smooth)
                plt.plot(steps[len(steps)-len(smoothed):], smoothed, label=method)
            else:
                plt.plot(steps, returns, label=method)
        plt.xlabel("Steps"); plt.ylabel("Episodic Return (Shaped)"); plt.title(f"Online Training Performance ({experiment_id})")
        plt.legend(); plt.grid(True); plt.savefig(output_path / "online_returns.png")
        print(f"Saved online plot to {output_path / 'online_returns.png'}")

    # Plot Eval Comparison (Shaped)
    if eval_data:
        plt.figure(figsize=(10, 6))
        for label, (limits, rewards) in eval_data.items():
            plt.plot(limits, rewards, marker='o', label=label)
        plt.xlabel("Training Steps / Dataset Size"); plt.ylabel("Average Eval Return (Shaped)")
        plt.title(f"Performance Comparison - Shaped Rewards ({experiment_id})")
        plt.legend(); plt.grid(True); plt.savefig(output_path / "eval_comparison_shaped.png")
        print(f"Saved shaped eval plot to {output_path / 'eval_comparison_shaped.png'}")

    # Plot Eval Comparison (Raw Atari)
    if eval_raw_data:
        plt.figure(figsize=(10, 6))
        for label, (limits, rewards) in eval_raw_data.items():
            plt.plot(limits, rewards, marker='s', linestyle='--', label=label)
        plt.xlabel("Training Steps / Dataset Size"); plt.ylabel("Average Atari Score")
        plt.title(f"Performance Comparison - Raw Score ({experiment_id})")
        plt.legend(); plt.grid(True); plt.savefig(output_path / "eval_comparison_raw.png")
        print(f"Saved raw eval plot to {output_path / 'eval_comparison_raw.png'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experimentid", type=str)
    parser.add_argument("--runs_dir", type=str, default="out/runs")
    parser.add_argument("--output_dir", type=str, default="plots")
    args = parser.parse_args()
    
    plot_results(args.experimentid, args.runs_dir, args.output_dir)
