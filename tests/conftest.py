import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"

for p in [
    str(PROJECT_ROOT),
    str(SRC_DIR),
    str(SRC_DIR / "app"),
    str(SRC_DIR / "usr"),
    str(SRC_DIR / "usr" / "models"),
    str(SRC_DIR / "usr" / "environments"),
    str(SRC_DIR / "usr" / "eval"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

collect_ignore_glob = ["in/envs/*", "src/usr/models/fyd_repo/*", "src/usr/models/cew_repo/*"]

# Qt requires QtWebEngineWidgets to be imported before any QApplication exists.
# Test modules build one at import time, so whichever module pytest collects
# first would otherwise decide whether the web engine is usable for all of them
# — and the modules that need it fail to import and skip silently rather than
# failing. Importing it here, before any test module is collected, removes the
# ordering dependency.
try:
    from PyQt6 import QtWebEngineWidgets  # noqa: F401
except ImportError:
    pass


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
