"""Core utility functions for dynamic module loading and inspection."""

import importlib.util
import sys
from pathlib import Path


def load_module(path: str):
    """Dynamically load a Python module from a file path without polluting sys.modules."""
    p = Path(path)
    module_name = f"dynamic_{p.stem}"
    spec = importlib.util.spec_from_file_location(module_name, str(p))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from path: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
