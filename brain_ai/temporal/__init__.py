"""
Temporal Processing Module

HTM-based sequence learning and prediction with fallback options.
"""

from .htm import (
    HTMLayer,
    HTMConfig,
    PytorchSpatialPooler,
    PytorchTemporalMemory,
    SparseTensor,
    create_htm_layer,
    HTM_CORE_AVAILABLE,
    # Accelerated HTM (2025)
    ReflexMemory,
    AcceleratedHTM,
    create_accelerated_htm,
)

from .sequence import (
    TemporalLayer,
    LSTMSequencePredictor,
    GRUSequencePredictor,
    TransformerSequencePredictor,
    SequenceConfig,
    create_temporal_layer,
)

__all__ = [
    # HTM
    "HTMLayer",
    "HTMConfig",
    "PytorchSpatialPooler",
    "PytorchTemporalMemory",
    "SparseTensor",
    "create_htm_layer",
    "HTM_CORE_AVAILABLE",
    # Accelerated HTM (2025)
    "ReflexMemory",
    "AcceleratedHTM",
    "create_accelerated_htm",
    # Sequence predictors
    "TemporalLayer",
    "LSTMSequencePredictor",
    "GRUSequencePredictor",
    "TransformerSequencePredictor",
    "SequenceConfig",
    "create_temporal_layer",
]
