# brain_ai/memory/__init__.py
"""
Memory systems for Brain-Inspired AI.

Engram: O(1) conditional memory via N-gram hash lookup.
"""

from .tokenizer_compression import TokenizerCompression
from .hash_embedding import MultiHeadHash, OffloadableEmbedding
from .engram import (
    EngramConfig,
    RMSNorm,
    EngramEmbedding,
    ContextAwareGating,
    EngramModule,
)

__all__ = [
    "TokenizerCompression",
    "MultiHeadHash",
    "OffloadableEmbedding",
    "EngramConfig",
    "RMSNorm",
    "EngramEmbedding",
    "ContextAwareGating",
    "EngramModule",
]
