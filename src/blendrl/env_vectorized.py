"""Backward-compatibility re-export of VectorizedNudgeBaseEnv.

The canonical definition has moved to src.core.env_vectorized to decouple
general gym environment loading from the BlendRL model architecture.
"""
from src.core.env_vectorized import VectorizedNudgeBaseEnv

__all__ = ["VectorizedNudgeBaseEnv"]
