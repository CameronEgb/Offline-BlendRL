"""Shared test fixtures for the NeSyRL test suite."""

import os
from pathlib import Path

import pytest

collect_ignore_glob = ["in/envs/*", "src/fyd_repo/*", "src/cew_repo/*"]

PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def config_dir():
    """Return the Hydra config directory."""
    return PROJECT_ROOT / "in" / "config"


@pytest.fixture
def tmp_results(tmp_path):
    """Create a temporary results directory structure."""
    for sub in ["logs", "plots", "checkpoints", "datasets"]:
        (tmp_path / sub).mkdir()
    return tmp_path
