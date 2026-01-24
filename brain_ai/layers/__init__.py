# brain_ai/layers/__init__.py
"""
Neural network layers for Brain-Inspired AI.
"""

from .engram_layer import EngramAugmentedLayer, create_engram_layer

__all__ = [
    'EngramAugmentedLayer',
    'create_engram_layer',
]
