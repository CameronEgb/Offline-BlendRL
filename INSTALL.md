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
uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000
```

- **Interactive API Documentation (Swagger UI):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
