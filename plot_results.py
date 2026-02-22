import argparse
import os
import pickle
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def moving_average(a, n=100):
    if len(a) < n:
        return np.array(a)
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_results(experiment_id, runs_dir="out/runs", output_dir="plots"):
    exp_path = Path(runs_dir) / experiment_id
    if not exp_path.exists():
        alt_path = Path(runs_dir) / f"exp_{experiment_id}"
        if alt_path.exists(): exp_path = alt_path
        else:
            print(f"Warning: Experiment directory {exp_path} not found.")
            exp_path = Path(runs_dir)
        
    output_path = Path(output_dir) / experiment_id
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Scanning {exp_path} for experiment {experiment_id}...")
    
    # Data structures
    # Continuous training data
    online_shaped = {} 
    online_raw = {}
    online_lengths = {}
    
    # Interval evaluation data
    eval_shaped = {}
    eval_raw = {}
    eval_limits = {}

    for run_folder in exp_path.glob(f"*{experiment_id}*"):
        if not run_folder.is_dir(): continue
        folder_name = run_folder.name
        is_offline = folder_name.startswith("off_")
        
        # Clean method name
        clean_method = folder_name.replace(f"_{experiment_id}", "")
        clean_method = clean_method.replace("Seaquest-v4_", "").replace("seaquest_", "")
        
        # 1. Load Continuous Training Data (Online only)
        pkl_path = run_folder / "checkpoints" / "training_log.pkl"
        if pkl_path.exists():
            try:
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)
                    if data and len(data) >= 2:
                        returns, lengths = data[0], data[1]
                        if returns:
                            online_shaped[clean_method] = returns
                            online_lengths[clean_method] = lengths
                            # Check if we have raw returns (new format)
                            if len(data) >= 7:
                                online_raw[clean_method] = data[6]
            except: pass

        # 2. Load Interval Eval Data (results.json)
        json_path = run_folder / "results.json"
        if json_path.exists():
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                    if not data: continue
                    label = clean_method if is_offline else f"{clean_method} (Online)"
                    eval_limits[label] = [d["data_limit"] for d in data]
                    eval_shaped[label] = [d["avg_reward"] for d in data]
                    if "avg_raw_reward" in data[0]:
                        eval_raw[label] = [d["avg_raw_reward"] for d in data]
            except: pass

    # --- FIGURE 1: Online Continuous Performance ---
    if online_shaped:
        # Shaped plot
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        for method, returns in online_shaped.items():
            steps = np.cumsum(online_lengths[method])
            n_smooth = min(50, max(1, len(returns)//5))
            y = moving_average(returns, n=n_smooth)
            x = steps[len(steps)-len(y):]
            plt.plot(x, y, label=method)
        plt.xlabel("Steps"); plt.ylabel("Episodic Return (Shaped)"); plt.title("Continuous Training (Shaped)"); plt.legend(); plt.grid(True)

        # Raw plot (if available)
        plt.subplot(1, 2, 2)
        if online_raw:
            for method, returns in online_raw.items():
                steps = np.cumsum(online_lengths[method])
                n_smooth = min(50, max(1, len(returns)//5))
                y = moving_average(returns, n=n_smooth)
                x = steps[len(steps)-len(y):]
                plt.plot(x, y, label=method)
            plt.xlabel("Steps"); plt.ylabel("Atari Score"); plt.title("Continuous Training (Raw Score)"); plt.legend(); plt.grid(True)
        else:
            plt.text(0.5, 0.5, "No raw data in old logs", ha='center')
            
        plt.tight_layout()
        plt.savefig(output_path / "online_performance.png")
        print(f"Saved online performance plot to {output_path / 'online_performance.png'}")

    # --- FIGURE 2: Eval Comparison (Shaped) ---
    if eval_shaped:
        plt.figure(figsize=(10, 6))
        for label, rewards in eval_shaped.items():
            plt.plot(eval_limits[label], rewards, marker='o', label=label)
        plt.xlabel("Training Steps / Dataset Size"); plt.ylabel("Avg Eval Return (Shaped)")
        plt.title(f"Evaluation Comparison - Shaped Rewards ({experiment_id})")
        plt.legend(); plt.grid(True); plt.savefig(output_path / "eval_shaped.png")
        print(f"Saved shaped eval plot to {output_path / 'eval_shaped.png'}")

    # --- FIGURE 3: Eval Comparison (Raw Atari) ---
    if eval_raw:
        plt.figure(figsize=(10, 6))
        for label, rewards in eval_raw.items():
            plt.plot(eval_limits[label], rewards, marker='s', linestyle='--', label=label)
        plt.xlabel("Training Steps / Dataset Size"); plt.ylabel("Avg Atari Score")
        plt.title(f"Evaluation Comparison - Raw Atari Score ({experiment_id})")
        plt.legend(); plt.grid(True); plt.savefig(output_path / "eval_raw.png")
        print(f"Saved raw eval plot to {output_path / 'eval_raw.png'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experimentid", type=str)
    parser.add_argument("--runs_dir", type=str, default="out/runs")
    parser.add_argument("--output_dir", type=str, default="plots")
    args = parser.parse_args()
    plot_results(args.experimentid, args.runs_dir, args.output_dir)
