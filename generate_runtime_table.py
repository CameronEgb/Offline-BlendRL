import argparse
import json
import os
from pathlib import Path
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

def format_table(data, headers):
    if HAS_TABULATE:
        print(tabulate(data, headers=headers, tablefmt="github"))
        return

    # Simple table formatter fallback
    widths = [max(len(str(row[i])) for row in data + [headers]) for i in range(len(headers))]
    
    line = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    header_line = "| " + " | ".join(str(h).ljust(w) for h, w in zip(headers, widths)) + " |"
    
    print(line)
    print(header_line)
    print(line)
    
    # Find minimum runtime
    min_time = min(row[1] for row in data) if data else None
    
    for row in data:
        is_min = row[1] == min_time and min_time is not None
        
        formatted_row = []
        for i, val in enumerate(row):
            str_val = str(val)
            if is_min:
                str_val = f"**{str_val}**"
            formatted_row.append(str_val.ljust(widths[i]))
            
        print("| " + " | ".join(formatted_row) + " |")
    
    print(line)

def generate_table(experiment_id, runs_dir="results/experiments"):
    exp_path = Path(runs_dir) / experiment_id
    if not exp_path.exists():
        exp_path = Path(runs_dir) # Fallback to root
        
    print(f"Scanning {exp_path} for experiment {experiment_id} runtimes...")
    
    runtimes = []
    
    # Robustly find all runtime.json files within the experiment/group directory
    for runtime_file in exp_path.rglob("runtime.json"):
        run_folder = runtime_file.parent
        if "checkpoints" in run_folder.parts:
            continue
            
        try:
            with open(runtime_file, "r") as f:
                data = json.load(f)
                # Extract descriptive method name from folder structure
                # Handle group folders by including sub-parts if useful, but strip common suffixes
                method_name = run_folder.name
                for suffix in [f"_{experiment_id}", f"_{experiment_id.replace('exp_', '')}"]:
                    if method_name.endswith(suffix):
                        method_name = method_name[:-len(suffix)]
                
                # Strip environment prefixes
                for env in ["Seaquest-v4_", "seaquest_", "mountaincar_", "cartpole_"]:
                    if method_name.startswith(env):
                        method_name = method_name[len(env):]
                
                # If in a hierarchical structure, prepend the ruleset
                if "online" in run_folder.parts:
                    idx = run_folder.parts.index("online")
                    if idx + 1 < len(run_folder.parts) - 1:
                        rs = run_folder.parts[idx+1]
                        method_name = f"{rs}/{method_name}"
                elif "offline" in run_folder.parts:
                    idx = run_folder.parts.index("offline")
                    if idx + 1 < len(run_folder.parts) - 1:
                        rs = run_folder.parts[idx+1]
                        method_name = f"{rs}/{method_name}"

                avg_train = f"{data.get('avg_train_time_per_interval', 0):.2f}s"
                avg_eval = f"{data.get('avg_eval_time_per_interval', 0):.2f}s"
                
                runtimes.append([
                    method_name, 
                    int(data.get("runtime_seconds", 0)), 
                    data.get("runtime_formatted", "0:00:00"),
                    avg_train,
                    avg_eval
                ])
        except Exception as e:
            print(f"Error reading {runtime_file}: {e}")
    
    if not runtimes:
        print("No runtime data found.")
        return

    # Sort by runtime
    runtimes.sort(key=lambda x: x[1])
    
    headers = ["Method", "Seconds", "Formatted (HH:MM:SS)", "Avg Train/Int", "Avg Eval/Int"]
    format_table(runtimes, headers)
    
    # Save to file
    output_dir = Path("results/plots") / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "runtimes.txt"
    with open(output_file, "w") as f:
        # Re-capture output to file
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        
        format_table(runtimes, headers)
        
        sys.stdout = old_stdout
        f.write(mystdout.getvalue())
    
    print(f"Table saved to {output_file}")

if __name__ == "__main__":
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("experimentid", type=str)
    parser.add_argument("--runs_dir", type=str, default="results/experiments")
    args = parser.parse_args()

    experiment_id = args.experimentid
    
    # 1. Check if the input is a configList or config name
    list_path = f"in/config/configLists/{experiment_id}"
    if os.path.isfile(list_path):
        # If it's a configList, use the list name as experiment_id (which usually acts as the group folder)
        print(f"Using configList '{experiment_id}' as the folder to scan.")
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
                        experiment_id = cfg["experimentid"]
                        print(f"Resolved experiment_id '{experiment_id}' from config '{config_path}'")
            except Exception as e:
                print(f"Error reading config {config_path}: {e}")
            
    generate_table(experiment_id, args.runs_dir)
