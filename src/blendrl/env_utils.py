"""Backward-compatibility re-export of Atari environment wrappers.

Canonical implementation has moved to src.core.atari_wrappers.
"""

from src.core.atari_wrappers import make_atari_env, make_env

__all__ = ["make_atari_env", "make_env"]
