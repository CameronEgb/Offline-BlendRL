import argparse
import os
import pickle
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import yaml

def get_style_info(label):
    l = label.lower()
    # PPO Baseline
    if l == "ppo":
        return "black", "--", "o"
    
    # Offline BlendIQL (BlendRL-IQL)
    if "blendiql" in l:
        if "on ppo" in l:
            return "#d62728", "-", "s" # Red square
        if "on blendrl" in l:
            return "#ff7f0e", "-", "D" # Orange diamond
            
    # Online BlendRL
    if "blendrl" in l and "on" not in l:
        return "#2ca02c", "-", "^" # Green triangle
        
    # Offline IQL
    if "iql" in l and "blendiql" not in l:
        if "on ppo" in l:
            return "#1f77b4", "-", "v" # Blue down-triangle
        if "on blendrl" in l:
            return "#9467bd", "-", "p" # Purple pentagon
            
    return None, "-", "o"

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

def load_run_data(run_folder, experiment_id, num_envs_default, ruleset_default="none"):
    folder_name = run_folder.name
    is_offline = "offline" in str(run_folder) or folder_name.startswith("off_")
    
    # Defaults
    run_num_envs = num_envs_default
    ruleset = ruleset_default
    method_name = "unknown"
    
    # Try to detect ruleset from new hierarchical structure first
    # Structure: .../online/[ruleset]/[method] or .../offline/[ruleset]/[method]
    parts = list(run_folder.parts)
    if "online" in parts:
        idx = parts.index("online")
        if idx + 1 < len(parts):
            ruleset = parts[idx + 1]
    elif "offline" in parts:
        idx = parts.index("offline")
        if idx + 1 < len(parts):
            ruleset = parts[idx + 1]
            
    # Try to load from config.yaml as backup
    config_path = run_folder / "config.yaml"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                if config:
                    if "num_envs" in config:
                        run_num_envs = int(config["num_envs"])
                    if "rules" in config:
                        ruleset = config["rules"]
                    if "exp_name" in config:
                        method_name = "blendrl_ppo" if "blenderl" in config["exp_name"] else "ppo"
        except: pass

    # If method or ruleset still unknown/default, parse folder name
    # NEW FORMAT: {method}_{rs} or {off_method}_{source}_{rs}
    if method_name == "unknown":
        if is_offline:
            if folder_name.startswith("blendrl_iql"): method_name = "blendrl_iql"
            else: method_name = "iql"
        else:
            if folder_name.startswith("blendrl_ppo"): method_name = "blendrl_ppo"
            else: method_name = "ppo"

    if ruleset == "none" or "," in ruleset: 
        # Attempt to strip the method and source to find the ruleset
        core_name = folder_name
        
        # Strip offline prefix
        if is_offline:
            if core_name.startswith("blendrl_iql_blendrl_ppo_"): core_name = core_name[len("blendrl_iql_blendrl_ppo_"):]
            elif core_name.startswith("blendrl_iql_ppo_"): core_name = core_name[len("blendrl_iql_ppo_"):]
            elif core_name.startswith("iql_blendrl_ppo_"): core_name = core_name[len("iql_blendrl_ppo_"):]
            elif core_name.startswith("iql_ppo_"): core_name = core_name[len("iql_ppo_"):]
        else:
            if core_name.startswith("blendrl_ppo_"): core_name = core_name[len("blendrl_ppo_"):]
            elif core_name.startswith("ppo_"): core_name = core_name[len("ppo_"):]
        
        # If the stripping didn't happen (maybe old format), use the previous logic
        if core_name == folder_name:
            # Step 1: Strip the experiment_id suffix
            suffix = f"_{experiment_id}"
            if folder_name.endswith(suffix):
                core_name = folder_name[:-len(suffix)]
            else:
                core_name = folder_name
                
            # Step 2: Strip environment prefix
            for env in ["seaquest_", "Seaquest-v4_", "mountaincar_", "cartpole_", "reverseMC_", "alien_"]:
                if core_name.startswith(env):
                    core_name = core_name[len(env):]
                    break
            
            # Step 3: Strip method prefix
            if is_offline:
                if core_name.startswith("blendrl_iql_blendrl_ppo_"): core_name = core_name[len("blendrl_iql_blendrl_ppo_"):]
                elif core_name.startswith("blendrl_iql_ppo_"): core_name = core_name[len("blendrl_iql_ppo_"):]
                elif core_name.startswith("iql_blendrl_ppo_"): core_name = core_name[len("iql_blendrl_ppo_"):]
                elif core_name.startswith("iql_ppo_"): core_name = core_name[len("iql_ppo_"):]
            else:
                if core_name.startswith("blendrl_ppo_"): core_name = core_name[len("blendrl_ppo_"):]
                elif core_name.startswith("ppo_"): core_name = core_name[len("ppo_"):]
            
        ruleset = core_name


    # Special case for PPO: it might be in a folder with "default" in the name but it uses no rules
    if "ppo" in method_name and "blendrl" not in method_name:
        ruleset = "default"

    # Ruleset Renaming Mapping
    ruleset_mapping = {
        "default": "full",
        "no_left_blender_right": "nolefteasy",
        "no_left": "nolefthard",
        "no_right_blender_left": "nolefteasy",
        "no_right": "nolefthard",
        "variant1": "v1",
        "variant2": "v2",
        "none": "none"
    }
    display_ruleset = ruleset_mapping.get(ruleset, ruleset)

    # Legend Label
    if is_offline:
        # Determine source method (ppo or blendrl_ppo)
        # Check for blendrl_ppo first because ppo is a substring
        if "blendrl_ppo" in folder_name:
            source = "blendrl"
        elif "ppo" in folder_name:
            source = "ppo"
        else:
            source = "unknown"
            
        off_m_label = "blendIQL" if "blendrl" in method_name else "iql"
        
        # Format: [ruleset] [iql|blendIQL] on [ppo|blendrl]
        if off_m_label == "blendIQL":
            legend_label = f"[{display_ruleset}] {off_m_label} on {source}"
        else:
            legend_label = f"{off_m_label} on {source}"
    else:
        if method_name == "ppo":
            legend_label = "ppo"
        else:
            # For online BlendRL: [ruleset] blendrl
            legend_label = f"[{display_ruleset}] blendrl"

    online_data = None
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
                        online_data = (returns, lengths)
        except Exception as e:
            print(f"  Error loading pkl from {run_folder.name}: {e}")

    eval_data = None
    # 2. Load Interval Eval Data (results.json)
    json_path = run_folder / "results.json"
    if not json_path.exists():
        json_path = run_folder / "checkpoints" / "results.json"
        
    if json_path.exists():
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
                if data:
                    eval_data = {
                        "limits": [d["data_limit"] for d in data],
                        "rewards": [d["avg_reward"] for d in data]
                    }
        except Exception as e:
            print(f"  Error loading json from {json_path}: {e}")
            
    return display_ruleset, legend_label, online_data, eval_data, folder_name

def plot_results(experiment_id=None, group=None, runs_dir="results/experiments", output_dir="results/plots", num_envs_override=None, smooth_window=None):
    if group:
        base_path = Path(runs_dir) / group
        output_path = Path(output_dir) / group
        print(f"Plotting group: {group} from {base_path}")
        if experiment_id:
            exp_folders = [base_path / experiment_id]
        else:
            exp_folders = [d for d in base_path.iterdir() if d.is_dir()]
    else:
        base_path = Path(runs_dir)
        output_path = Path(output_dir) / experiment_id
        exp_folders = [base_path / experiment_id]

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # all_online_shaped[display_ruleset][legend_label] = data
    all_online_shaped = {}
    all_online_lengths = {}
    all_eval_shaped = {}
    all_eval_limits = {}

    for exp_path in exp_folders:
        if not exp_path.exists():
            continue
            
        curr_exp_id = exp_path.name
        
        # 1. Try to find the common ruleset for this experiment from hyperparameters.txt
        exp_ruleset = "none"
        num_envs = 1
        hp_path = exp_path / "hyperparameters.txt"
        if hp_path.exists():
            try:
                with open(hp_path, "r") as f:
                    for line in f:
                        if "Rules:" in line:
                            exp_ruleset = line.split(":")[1].strip()
                        if "Num Envs:" in line:
                            num_envs = int(line.split(":")[1].strip())
            except: pass
        if num_envs_override is not None:
            num_envs = num_envs_override

        # 2. Scan for sub-runs recursively (New hierarchical structure)
        # Search for both interval evaluation data (results.json) AND continuous training data (training_log.pkl)
        run_folders = [f.parent for f in exp_path.rglob("results.json")]
        # Also check checkpoints/results.json just in case
        run_folders += [f.parent.parent for f in exp_path.rglob("checkpoints/results.json")]
        # Also include online training folders that only have training_log.pkl
        run_folders += [f.parent.parent for f in exp_path.rglob("training_log.pkl")]
        # Unique folders
        run_folders = list(set(run_folders))
        
        # WORKAROUND: Only process folders that are inside an 'online' or 'offline' subdirectory
        # to avoid legacy files causing duplicate plots (like eval_comparison_mc_default.png)
        run_folders = [f for f in run_folders if "online" in f.parts or "offline" in f.parts]

        for run_folder in run_folders:
            display_ruleset, legend_label, online, evaluation, folder_name = load_run_data(run_folder, curr_exp_id, num_envs, ruleset_default=exp_ruleset)
            
            # IMPROVED DETECTION: Try to extract the actual tuning ID from the folder_name
            # folder_name is like "blendrl_ppo_mc_easy_tune_1"
            # curr_exp_id is the top level directory name
            
            # If folder_name contains curr_exp_id, it's likely a sub-run and folder_name is more specific
            # But folder_name might also contain method prefixes.
            # Let's use a combination or the most specific part.
            
            specific_id = folder_name
            # If folder_name is just "online" or something generic, fallback to curr_exp_id
            if folder_name in ["online", "offline", "checkpoints"]:
                specific_id = curr_exp_id
            
            if display_ruleset not in all_online_shaped:
                all_online_shaped[display_ruleset] = {}
                all_online_lengths[display_ruleset] = {}
                all_eval_shaped[display_ruleset] = {}
                all_eval_limits[display_ruleset] = {}

            # Use the most specific ID we found
            plot_label = f"{legend_label} ({specific_id})"

            if online:
                print(f"  Detected Online run: {plot_label} (Ruleset: {display_ruleset})")
                all_online_shaped[display_ruleset][plot_label] = online[0]
                all_online_lengths[display_ruleset][plot_label] = online[1]
            if evaluation:
                print(f"  Detected Eval run: {plot_label} (Ruleset: {display_ruleset})")
                all_eval_shaped[display_ruleset][plot_label] = evaluation["rewards"]
                all_eval_limits[display_ruleset][plot_label] = evaluation["limits"]

    # --- PLOTTING ---
    
    # 1. Online Performance (Combined by group or separate)
    if all_online_shaped:
        for x_axis in ["steps", "episodes"]:
            fig, ax1 = plt.subplots(1, 1, figsize=(12, 7))
            found_any = False
            for ruleset in all_online_shaped:
                for label, returns in all_online_shaped[ruleset].items():
                    found_any = True
                    if x_axis == "steps":
                        x = np.cumsum(all_online_lengths[ruleset][label])
                        label_x = "Total Agent Steps"
                    else:
                        x = np.arange(len(returns))
                        label_x = "Completed Episodes"
                    
                    n_smooth = smooth_window if smooth_window else min(50, max(1, len(returns)//20))
                    y = moving_average(returns, n=n_smooth)
                    ax1.plot(x, y, label=label)
            
            if found_any:
                ax1.set_xlabel(label_x); ax1.set_ylabel("Return"); ax1.set_title(f"Continuous Training Performance ({group if group else experiment_id})")
                ax1.grid(True)
                ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
                plt.tight_layout()
                plt.savefig(output_path / f"online_performance_{x_axis}.png", bbox_inches='tight')
                print(f"Saved online plot to {output_path / f'online_performance_{x_axis}.png'}")
            plt.close()

    # 2. Evaluation Comparison (ONE GRAPH PER RULESET)
    if all_eval_shaped:
        # First, find PPO results if they exist (ruleset='none')
        ppo_evals = all_eval_shaped.get("none", {})
        ppo_limits = all_eval_limits.get("none", {})
        
        # Sort rulesets for consistent plot order
        sorted_rulesets = sorted([r for r in all_eval_shaped.keys() if r != "none"])

        for ruleset in sorted_rulesets:
            results = all_eval_shaped[ruleset]
            if not results: continue
            
            fig, ax1 = plt.subplots(1, 1, figsize=(12, 7))
            
            # Plot PPO baseline first
            for ppo_label, rewards in ppo_evals.items():
                x = np.array(ppo_limits[ppo_label])
                color, ls, marker = get_style_info(ppo_label)
                ax1.plot(x, rewards, marker=marker, label=ppo_label, linestyle=ls, color=color, linewidth=2)
                
            # Plot actual ruleset results
            # Sort labels to ensure consistent legend order
            sorted_labels = sorted(results.keys())
            for label in sorted_labels:
                rewards = results[label]
                x = np.array(all_eval_limits[ruleset][label])
                color, ls, marker = get_style_info(label)
                ax1.plot(x, rewards, marker=marker, label=label, linestyle=ls, color=color, linewidth=2)
            
            ax1.set_xlabel("Total Training Steps / Dataset Size"); ax1.set_ylabel("Avg Eval Return")
            ax1.set_title(f"Evaluation Comparison: Ruleset = {ruleset}")
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=True)
            plt.tight_layout()
            
            filename = f"eval_comparison_{ruleset}.png"
            plt.savefig(output_path / filename, bbox_inches='tight')
            print(f"Saved evaluation plot to {output_path / filename}")
            plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experimentid", type=str, nargs='?', default=None)
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--runs_dir", type=str, default="results/experiments")
    parser.add_argument("--output_dir", type=str, default="results/plots")
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--smooth", type=int, default=None)
    args = parser.parse_args()
    
    experiment_id = args.experimentid
    group = args.group

    # 1. Check if the input is a configList or config name
    if experiment_id and not group:
        list_path = f"in/config/configLists/{experiment_id}"
        if os.path.isfile(list_path):
            # If it's a configList, the list name is the group
            group = experiment_id
            experiment_id = None
            print(f"Resolved group '{group}' from configList '{list_path}'")
        else:
            config_path = None
            if os.path.isfile(f"in/config/{experiment_id}"):
                config_path = f"in/config/{experiment_id}"
            elif os.path.isfile(f"in/config/{experiment_id}.yaml"):
                config_path = f"in/config/{experiment_id}.yaml"
            
            if config_path:
                try:
                    with open(config_path, "r") as f:
                        cfg = yaml.safe_load(f)
                        if cfg and "experimentid" in cfg:
                            # If it's a config, we use its experimentid, and the config name as the group
                            group = experiment_id
                            experiment_id = cfg["experimentid"]
                            print(f"Resolved experiment_id '{experiment_id}' and group '{group}' from config '{config_path}'")
                except Exception as e:
                    print(f"Error reading config {config_path}: {e}")

    # 2. WORKAROUND: If the user provides a first argument that exists as a group folder, 
    # treat it as a group automatically.
    if experiment_id and not group:
        potential_group_path = os.path.join(args.runs_dir, experiment_id)
        if os.path.isdir(potential_group_path):
            # Check if it has sub-directories (likely a group) or just results (likely a single exp)
            subdirs = [d for d in os.listdir(potential_group_path) if os.path.isdir(os.path.join(potential_group_path, d))]
            if "online" in subdirs or "offline" in subdirs or len(subdirs) > 2:
                group = experiment_id
                experiment_id = None
                print(f"Auto-detected '{group}' as a group folder.")

    if not experiment_id and not group:
        parser.error("At least one of experimentid or --group must be provided.")
        
    plot_results(experiment_id, group, args.runs_dir, args.output_dir, args.num_envs, smooth_window=args.smooth)
