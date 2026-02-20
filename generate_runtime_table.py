import argparse
import json
import os
from pathlib import Path
from tabulate import tabize # Not available, using simple formatting

def format_table(data, headers):
    # Simple table formatter
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

def generate_table(experiment_id, runs_dir="out/runs"):
    exp_path = Path(runs_dir) / experiment_id
    if not exp_path.exists():
        exp_path = Path(runs_dir) # Fallback to root
        
    print(f"Scanning {exp_path} for experiment {experiment_id} runtimes...")
    
    runtimes = []
    
    for run_folder in exp_path.rglob(f"*{experiment_id}*"):
        if not run_folder.is_dir() or "checkpoints" in run_folder.parts:
            continue
            
        runtime_file = run_folder / "runtime.json"
        if runtime_file.exists():
            try:
                with open(runtime_file, "r") as f:
                    data = json.load(f)
                    method_name = run_folder.name.replace(f"_{experiment_id}", "").replace(f"_{experiment_id.replace('exp_', '')}", "")
                    method_name = method_name.replace("Seaquest-v4_", "").replace("seaquest_", "")
                    runtimes.append([method_name, int(data["runtime_seconds"]), data["runtime_formatted"]])
            except Exception as e:
                print(f"Error reading {runtime_file}: {e}")
    
    if not runtimes:
        print("No runtime data found.")
        return

    # Sort by runtime
    runtimes.sort(key=lambda x: x[1])
    
    headers = ["Method", "Seconds", "Formatted (HH:MM:SS)"]
    format_table(runtimes, headers)
    
    # Save to file
    output_dir = Path("plots") / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "runtimes.txt", "w") as f:
        # Re-capture output to file
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        
        format_table(runtimes, headers)
        
        sys.stdout = old_stdout
        f.write(mystdout.getvalue())
    
    print(f"Table saved to {output_dir / 'runtimes.txt'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experimentid", type=str)
    parser.add_argument("--runs_dir", type=str, default="out/runs")
    args = parser.parse_args()
    
    generate_table(args.experimentid, args.runs_dir)
