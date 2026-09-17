import csv
import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

try:
    from src.method_registry import METHOD_STYLE
except ImportError:
    METHOD_STYLE = {}
try:
    from src.methods.registry import list_registered_agents
except ImportError:

    def list_registered_agents():
        return []


app = FastAPI(title="NeSyRL API")

# Enable CORS for frontend clients (Vite, Next.js, Electron, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs: dict[str, dict[str, Any]] = {}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/environments")
def list_environments():
    envs = []
    env_dir = Path("in/config/env")
    if env_dir.exists():
        for f in env_dir.glob("*.yaml"):
            if f.name.startswith("_"):
                continue
            try:
                with open(f) as yaml_file:
                    envs.append({"name": f.stem, "config": yaml.safe_load(yaml_file)})
            except Exception:
                pass
    return {"environments": envs}


@app.get("/api/methods")
def list_methods():
    return {
        "registered_agents": list_registered_agents(),
        "method_styles": METHOD_STYLE,
    }


@app.get("/api/experiments")
def list_experiments():
    exps = []
    exp_dir = Path("in/config/experiment")
    if exp_dir.exists():
        for f in sorted(exp_dir.glob("**/*.yaml")):
            # Ignore internal base / template configs
            if f.name.startswith("_"):
                continue
            rel_path = f.relative_to(exp_dir)
            try:
                with open(f) as yaml_file:
                    exps.append(
                        {
                            "path": str(rel_path),
                            "name": str(rel_path.with_suffix("")),
                            "config": yaml.safe_load(yaml_file),
                        }
                    )
            except Exception:
                pass
    return {"experiments": exps}


@app.get("/api/runs")
def list_runs():
    runs = []
    logs_dir = Path("results/logs")
    if logs_dir.exists():
        for run_dir in logs_dir.glob("*/*"):
            if run_dir.is_dir():
                runtime_file = run_dir / "runtime.json"
                metadata = {}
                if runtime_file.exists():
                    try:
                        with open(runtime_file) as f:
                            metadata = json.load(f)
                    except Exception:
                        pass
                runs.append(
                    {
                        "group": run_dir.parent.name,
                        "experiment_id": run_dir.name,
                        "metadata": metadata,
                    }
                )
    return {"runs": runs}


@app.get("/api/runs/{group}/{experiment_id}/{agent}/metrics")
def get_metrics(group: str, experiment_id: str, agent: str):
    agent_dir = Path(f"results/logs/{group}/{experiment_id}/{agent}")
    # Metrics live in version_N subdirectories
    metrics_file = None
    if agent_dir.exists():
        versions = sorted(agent_dir.glob("version_*/metrics.csv"), reverse=True)
        if versions:
            metrics_file = versions[0]
    if not metrics_file or not metrics_file.exists():
        raise HTTPException(status_code=404, detail="Metrics not found")

    metrics = []
    try:
        with open(metrics_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                metrics.append(row)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"metrics": metrics, "source": str(metrics_file)}


@app.get("/api/runs/{group}/{experiment_id}/plots")
def list_plots(group: str, experiment_id: str):
    plots = []
    plot_dir = Path(f"results/plots/{group}/{experiment_id}")
    if plot_dir.exists():
        for f in plot_dir.glob("*"):
            if f.is_file():
                plots.append(f.name)
    return {"plots": plots}


@app.get("/api/runs/{group}/{experiment_id}/plots/{filename}")
def get_plot_image(group: str, experiment_id: str, filename: str):
    file_path = Path(f"results/plots/{group}/{experiment_id}/{filename}")
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Plot not found")
    return FileResponse(file_path)


class LaunchRequest(BaseModel):
    experiment: str
    overrides: list[str] = []


def run_experiment_task(job_id: str, req: LaunchRequest):
    try:
        exp_name = req.experiment
        if exp_name.endswith(".yaml"):
            exp_name = exp_name[:-5]
        cmd = [sys.executable, "run_pipeline.py", exp_name] + req.overrides
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        jobs[job_id]["status"] = "running"
        jobs[job_id]["pid"] = process.pid
        jobs[job_id]["_process"] = process
        stdout, stderr = process.communicate()
        if jobs[job_id].get("status") != "cancelled":
            jobs[job_id]["status"] = "completed" if process.returncode == 0 else "failed"
        jobs[job_id]["returncode"] = process.returncode
        jobs[job_id]["stdout"] = stdout
        jobs[job_id]["stderr"] = stderr
    except Exception as e:
        if jobs[job_id].get("status") != "cancelled":
            jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)


@app.post("/api/experiments/launch")
def launch_experiment(req: LaunchRequest):
    import uuid

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "request": req.model_dump()}

    thread = threading.Thread(target=run_experiment_task, args=(job_id, req))
    thread.daemon = True
    thread.start()

    return {"job_id": job_id, "status": "pending"}


@app.get("/api/experiments/{job_id}/status")
def check_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job_info = {k: v for k, v in jobs[job_id].items() if k != "_process"}
    if "stdout" in job_info and isinstance(job_info["stdout"], str):
        job_info["stdout"] = job_info["stdout"][-1000:]
    return job_info


@app.post("/api/experiments/{job_id}/cancel")
def cancel_experiment(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job.get("status") not in ("pending", "running"):
        return {"status": job.get("status"), "message": f"Job is already {job.get('status')}"}

    process = job.get("_process")
    if process and process.poll() is None:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except Exception:
            try:
                process.terminate()
            except Exception:
                pass

    job["status"] = "cancelled"
    return {"job_id": job_id, "status": "cancelled"}
