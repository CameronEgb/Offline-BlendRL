# Installation & Setup Guide

This guide walks through setting up the environment to run the training pipeline, testing suites, and backend API service.

---

## 1. Prerequisites

- Python 3.10, 3.11, 3.12, or 3.13
- Git
- Virtual environment manager (`venv` or `conda`)

---

## 2. Installation Steps

### Step 1: Clone the Repository

```bash
git clone https://github.com/CameronEgb/Offline-BlendRL.git
cd Offline-BlendRL
```

### Step 2: Create and Activate Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows (PowerShell):

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

Install the core framework along with development tools and the FastAPI service:

```bash
pip install --upgrade pip
pip install -e ".[dev,api]"
```

*(Optional) Install Atari benchmarks:*
```bash
pip install -e ".[atari]"
```

---

## 3. Verifying Installation

Run the automated test suite to confirm all modules and registries function properly:

```bash
pytest tests/ -v
```

All 85+ unit and integration tests should pass cleanly in under 5 seconds.

---

## 4. Launching the Backend API Service

Start the local FastAPI development server on port 8000:

```bash
uvicorn src.app.api.app:app --reload --host 127.0.0.1 --port 8000
```

- **Interactive API Documentation (Swagger UI):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 5. Launching the Desktop Frontend

The PyQt6 frontend (`frontend/`) is standalone. It does not import the backend and uses its own lightweight virtual environment:

```bash
python3 -m venv frontend/.venv
source frontend/.venv/bin/activate        # Windows: .\frontend\.venv\Scripts\Activate.ps1
pip install -r frontend/requirements.txt
python -m frontend                        # run from the repository root
```

Start the backend API (section 4) first. The experiment builder sends each config to the backend, which composes and validates it with the same code as `run_pipeline.py`. The resolved config appears in the `config.yaml` tab. Without the backend, the builder still works but shows an unvalidated local draft. To use a different backend address, pass `--api-url <url>` or set `THETAIDE_API_URL`.

**Export recipe YAML…** writes a standalone experiment recipe. Save it in `in/config/experiment/thetaide/` and run it with `python run_pipeline.py thetaide/<name>`.

**Launch training** (F5) trains the builder's config on this machine through the backend. The Training monitor shows live reward, loss and progress, and the console streams the pipeline's output. **Stop** (Shift+F5) terminates the pipeline and its training processes. Each launch gets a unique experiment ID (`<name>_<YYYYmmdd-HHMMSS>`), so earlier results are never overwritten. Closing the window does not stop training; reopening the app reconnects to the run. **Run → Start simulated demo** works without a backend.

**Job queue.** **＋ Add to queue** (Ctrl+Shift+Q) adds the builder's config to the queue without starting it. When your list is ready, press **Start queue** in the Queue tab (or Ctrl+Shift+R). Queued experiments then train one at a time, in order, and the Training monitor switches to each one as it starts. **Pause queue** stops the next job from starting; one already training keeps running. The queue pauses itself when it runs out of jobs. The **Queue** tab lists the running, queued and recent jobs, with buttons to move a queued job up or down, remove it, or open it in the monitor. The queue is kept by the API server, so it continues while ThetaIDE is closed, but it is lost if the API server restarts.

**TensorBoard tab.** Runs launched with **Log to TensorBoard**, which is on by default, also write TensorBoard logs to `results/tensorboard/`. The first time you open the TensorBoard tab, the backend starts a local TensorBoard server (bound to 127.0.0.1) and the tab embeds it. The server keeps running while the API runs, so later visits reuse it. **Stop** ends it and **Open in browser** shows it in your browser. The embedded view needs `PyQt6-WebEngine`, which is in `frontend/requirements.txt`; without it, the tab offers only the browser option. Stop the API with Ctrl+C so it also shuts TensorBoard down. Force-killing the API process leaves TensorBoard running.

Run records are stored in `.thetaide/runs/` by default; override this with `--data-dir <path>`.
