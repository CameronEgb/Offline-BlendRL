# NeSyRL Backend API & Team Onboarding Guide

This document serves as the interface specification for integrating **ThetaIDE** (or any custom frontend/client) with the **NeSyRL** backend engine, as well as a guide for onboarding new team members to the repository.

---

## 1. Quick Start for Team Members

### 1.1 Development Setup

```bash
# 1. Clone repository
git clone https://github.com/CameronEgb/Offline-BlendRL.git
cd Offline-BlendRL

# 2. Virtual environment setup
python3 -m venv venv
source venv/bin/activate

# 3. Install core dependencies + dev tools + API service
pip install -e ".[dev,api]"

# 4. Verify test suite
pytest tests/ -v
```

### 1.2 Running a Smoke Test (Simulator vs. Offline Datasets)

Because `results/` and `in/datasets/` are `.gitignore`d to prevent committing large binary files, teammates should understand how datasets work:

1. **Simulator-Backed Benchmarks (`online_v_offline` paradigm):**
   Environments like `CartPole` or `MountainCar` **do not require any downloaded datasets**. 
   Running an experiment automatically generates the replay buffer in Phase 1 and trains offline agents on it in Phase 2:
   ```bash
   # Fast local smoke test (~15-30 seconds)
   python run_pipeline.py cp_final total_timesteps=1000 intervals_count=2 eval_episodes=5 site=local
   ```

2. **Static Clinical / Offline Datasets (`offline_only` paradigm):**
   Datasets like `MIMIC` or `Pyrenees` reside in `in/datasets/mimic/` or `in/datasets/pyrenees/`. 
   If working on clinical offline RL, obtain the dataset chunks (`.pkl`) from the team storage or run the preprocessing utilities in `scripts/preprocess_pyrenees.py`.

---

## 2. Launching the Backend API Server

The backend exposes a lightweight FastAPI service located in `src/api/app.py`.

```bash
# Start API server on localhost:8000 with hot-reloading
uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000
```

Once running:
- **Interactive Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Documentation:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 3. API Endpoint Reference

### 3.1 System Health
* **`GET /api/health`**
  * **Description:** Health-check endpoint to confirm server availability.
  * **Response:**
    ```json
    {"status": "ok"}
    ```

---

### 3.2 Metadata & Exploration (Experiment Builder Panel)

* **`GET /api/environments`**
  * **Description:** Discovers all declared environment YAML specifications under `in/config/env/`.
  * **Response:**
    ```json
    {
      "environments": [
        {
          "name": "cartpole",
          "config": {"name": "cartpole", "offline_only": false, ...}
        }
      ]
    }
    ```

* **`GET /api/methods`**
  * **Description:** Lists all registered agent architectures (`AGENT_REGISTRY`) and visualization style mappings (`METHOD_STYLE`).
  * **Response:**
    ```json
    {
      "registered_agents": ["blendrl_iql", "cql", "iql", "ppo"],
      "method_styles": {
        "ppo": {"label": "PPO (Neural)", "color": "#1f77b4", "linestyle": "-", "marker": "o"},
        "iql": {"label": "IQL (Neural)", "color": "#2ca02c", "linestyle": "-", "marker": "s"}
      }
    }
    ```

* **`GET /api/experiments`**
  * **Description:** Lists pre-configured experiment recipes in `in/config/experiment/**/*.yaml`.
  * **Response:**
    ```json
    {
      "experiments": [
        {
          "path": "cartpole/cp_final.yaml",
          "name": "cartpole/cp_final",
          "config": {"paradigm": "online_v_offline", "online_methods": "ppo/cp_tuned", ...}
        }
      ]
    }
    ```

---

### 3.3 Execution & Job Control (Training Monitor Panel)

* **`POST /api/experiments/launch`**
  * **Description:** Asynchronously spawns an experiment in a background worker thread using `run_pipeline.py`.
  * **Request Body:**
    ```json
    {
      "experiment": "cartpole/cp_final",
      "overrides": ["total_timesteps=2000", "intervals_count=2", "site=local"]
    }
    ```
  * **Response:**
    ```json
    {
      "job_id": "4b68e7b9-1f48-43d7-832f-488cb8e02d31",
      "status": "pending"
    }
    ```

* **`GET /api/experiments/{job_id}/status`**
  * **Description:** Queries the status of an active or finished job, including the tail of standard output.
  * **Response:**
    ```json
    {
      "status": "running",
      "pid": 48210,
      "stdout": "... [Epoch 1/5] eval/reward = 195.4 ...",
      "returncode": null
    }
    ```
  * *Possible Status Values:* `"pending"`, `"running"`, `"completed"`, `"failed"`, `"error"`.

---

### 3.4 Results & Visualization (Results Browser Panel)

* **`GET /api/runs`**
  * **Description:** Scans `results/logs/` and returns all completed experiment runs along with `runtime.json` metadata (git commit, hardware, timing).
  * **Response:**
    ```json
    {
      "runs": [
        {
          "group": "cartpole",
          "experiment_id": "cp_final",
          "metadata": {
            "agent": "ppo_cp_tuned",
            "training_time_seconds": 23.4,
            "git_commit": "31b56ad"
          }
        }
      ]
    }
    ```

* **`GET /api/runs/{group}/{experiment_id}/{agent}/metrics`**
  * **Description:** Parses the latest `metrics.csv` for the given agent and experiment into structured JSON for frontend graphing.
  * **Response:**
    ```json
    {
      "source": "results/logs/cartpole/cp_final/ppo_cp_tuned/version_0/metrics.csv",
      "metrics": [
        {"epoch": "0", "step": "0", "eval/reward": "18.2", "transitions": "0.0"},
        {"epoch": "1", "step": "256", "eval/reward": "194.5", "transitions": "50000.0"}
      ]
    }
    ```

* **`GET /api/runs/{group}/{experiment_id}/plots`**
  * **Description:** Lists all generated visual plots (PNG files) in `results/plots/{group}/{experiment_id}/`.
  * **Response:**
    ```json
    {
      "plots": ["convergence_reward.png", "losses_actor.png"]
    }
    ```

* **`GET /api/runs/{group}/{experiment_id}/plots/{filename}`**
  * **Description:** Streams the actual plot image file (e.g., `image/png`) directly to the client for rendering.
