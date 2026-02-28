"""
Training utilities for Brain-Inspired AI System.

Provides helpers for torch.compile integration, gradient checkpointing,
and memory profiling.
"""

from .compile_utils import (
    setup_torch_compile,
    setup_gradient_checkpointing,
    profile_memory,
    get_compile_stats,
)

__all__ = [
    "setup_torch_compile",
    "setup_gradient_checkpointing",
    "profile_memory",
    "get_compile_stats",
]
