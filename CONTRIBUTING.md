# Contributing to NeSyRL

Thank you for your interest in contributing! This guide will help you get started.

## Quick Setup

```bash
# 1. Fork and clone
git clone https://github.com/<YOUR_FORK>/NeSyRL.git
cd NeSyRL

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install in development mode
pip install -e ".[dev]"

# 4. Install pre-commit hooks
pre-commit install

# 5. Verify everything works
pytest tests/ -v
```

## Development Workflow

1. **Branch from `main`** — create a descriptive branch name (`feat/new-agent`, `fix/config-bug`).
2. **Write tests** for new features — add them to the `tests/` directory.
3. **Lint before committing** — `ruff check --fix .` and `ruff format .`
4. **CI must pass** before merge — the GitHub Actions workflow runs lint, tests, and type checking.

## Adding a New RL Method

See [WORKFLOW_GUIDE.md](docs/WORKFLOW_GUIDE.md) for step-by-step instructions on:
- Registering a new agent class
- Adding Hydra configuration YAML
- Wiring the architecture into the pipeline

## Adding a New Plotter

Plotters are auto-discovered by `plot/manager.py`. To add one:
1. Create a new Python file in `plot/` that subclasses `BasePlotter` from `plot/base.py`.
2. Implement the required `plot()` method.
3. Add a `_config.yaml` file for default plotter settings.
4. The manager will auto-discover and dispatch it.

## Project Structure

```
run_pipeline.py          # CLI entry point
src/                     # Core source code
  methods/               # RL agents (PPO, IQL, CQL, BlendRL variants)
  pipeline/              # Orchestration, config parsing, Slurm submission
  data/                  # Offline data modules
  core/                  # Factories, callbacks, Lightning builder
in/config/               # Hydra YAML configuration
plot/                    # Modular plotting framework
tests/                   # Test suite
```

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):
- `feat: add new plotting metric`
- `fix: resolve config override precedence`
- `docs: update installation instructions`
- `test: add dataset round-trip tests`
- `chore: update dependencies`

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run a specific test file
pytest tests/test_method_registry.py -v

# Skip slow tests
pytest tests/ -v -m "not slow"
```

## Code Style

- **Linter & Formatter:** [Ruff](https://github.com/astral-sh/ruff) (configured in `pyproject.toml`)
- **Type Checker:** [mypy](https://mypy-lang.org/) (configured in `pyproject.toml`)
- **Line length:** 120 characters
- **Python version:** 3.10+
