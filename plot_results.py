import argparse
import os
import pickle
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def moving_average(a, n=100) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_results(experiment_id, runs_dir="out/runs", output_dir="plots"):
    # Each experiment now has its own subdirectory in out/runs
    # If the user gives "003", we might have "exp_003" or "003"
    exp_path = Path(runs_dir) / experiment_id
    if not exp_path.exists():
        # Try prepending "exp_"
        alt_path = Path(runs_dir) / f"exp_{experiment_id}"
        if alt_path.exists():
            exp_path = alt_path
        else:
            print(f"Warning: Experiment directory {exp_path} or {alt_path} not found. Checking root {runs_dir}...")
            exp_path = Path(runs_dir)
        
    output_path = Path(output_dir) / experiment_id
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning {exp_path} for experiment {experiment_id}...")
    
    online_data = {} # method -> (steps, returns)
    eval_data = {}   # method -> (limits, rewards) - for both online and offline comparison
    
    # Search for immediate subfolders matching the ID
    # Use glob instead of rglob to avoid deep nesting issues
    for run_folder in exp_path.glob(f"*{experiment_id}*"):
        if not run_folder.is_dir():
            continue
            
        folder_name = run_folder.name
        print(f"Found run: {folder_name}")
        
        is_offline = folder_name.startswith("off_")
        
        # Check for Online Data (Continuous logs)
        pkl_path = run_folder / "checkpoints" / "training_log.pkl"
        if pkl_path.exists():
            try:
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)
                    returns = data[0]
                    lengths = data[1]
                    
                    if not returns:
                        print(f"  Note: {folder_name} has a log file but 0 completed episodes.")
                    
                    # Heuristic to clean method name: remove exp_id and surrounding underscores
                    method_name = folder_name.replace(f"_{experiment_id}", "").replace(f"_{experiment_id.replace('exp_', '')}", "")
                    method_name = method_name.replace("Seaquest-v4_", "").replace("seaquest_", "")
                    online_data[method_name] = (returns, lengths)
            except Exception as e:
                print(f"Error loading pkl {pkl_path}: {e}")

        # Check for Eval Data (results.json)
        json_path = run_folder / "results.json"
        if json_path.exists():
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                    rewards = [d["avg_reward"] for d in data]
                    limits = [d["data_limit"] for d in data]
                    
                    # Clean up method name
                    method_name = folder_name.replace(f"_{experiment_id}", "").replace(f"_{experiment_id.replace('exp_', '')}", "")
                    method_name = method_name.replace("Seaquest-v4_", "").replace("seaquest_", "")
                    if not is_offline:
                        method_name = f"{method_name} (Online Eval)"
                    
                    eval_data[method_name] = (limits, rewards)
            except Exception as e:
                print(f"Error loading json {json_path}: {e}")

    # Plot Online Continuous Training
    if online_data:
        # ... (rest of the plotting logic)
        plt.figure(figsize=(10, 6))
        plot_count = 0
        for method, (returns, lengths) in online_data.items():
            if not returns or not lengths:
                print(f"Warning: Empty data for method {method}")
                continue
            
            steps = np.cumsum(lengths)
            # Smooth
            n_smooth = min(100, len(returns)//2)
            if n_smooth > 1:
                smoothed = moving_average(returns, n=n_smooth)
                smoothed_steps = steps[len(steps)-len(smoothed):]
                plt.plot(smoothed_steps, smoothed, label=method)
                plot_count += 1
            elif len(returns) > 0:
                plt.plot(steps, returns, label=method)
                plot_count += 1
        
        if plot_count > 0:
            plt.xlabel("Steps")
            plt.ylabel("Episodic Return")
            plt.title(f"Online Training Performance ({experiment_id})")
            plt.legend()
            plt.grid(True)
            plt.savefig(output_path / "online_returns.png")
            print(f"Saved online plot to {output_path / 'online_returns.png'}")
        else:
            print("No valid online data points to plot.")
        plt.close()
    else:
        print("No online data found.")

    # Plot Eval Comparison (Online vs Offline)
    if eval_data:
        plt.figure(figsize=(10, 6))
        for method, (limits, rewards) in eval_data.items():
            plt.plot(limits, rewards, marker='o', label=method)
            
        plt.xlabel("Training Steps / Dataset Size")
        plt.ylabel("Average Eval Return")
        plt.title(f"Performance Comparison ({experiment_id})")
        plt.legend()
        plt.grid(True)
        plt.savefig(output_path / "eval_comparison.png")
        plt.close()
        print(f"Saved eval comparison plot to {output_path / 'eval_comparison.png'}")
    else:
        print("No evaluation data (results.json) found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experimentid", type=str)
    parser.add_argument("--runs_dir", type=str, default="out/runs")
    parser.add_argument("--output_dir", type=str, default="plots")
    args = parser.parse_args()
    
    plot_results(args.experimentid, args.runs_dir, args.output_dir)
