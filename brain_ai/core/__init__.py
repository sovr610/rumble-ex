"""
SNN Core Module

Spiking neural network foundation for brain-inspired AI.
"""

from .neurons import (
    LIFNeuron,
    AdaptiveLIFNeuron,
    RecurrentLIFNeuron,
    get_surrogate,
    ATanSurrogate,
    FastSigmoidSurrogate,
    StraightThroughSurrogate,
)

from .encoding import (
    RateEncoder,
    TemporalEncoder,
    LatencyEncoder,
    PopulationEncoder,
    DeltaEncoder,
    SpikeDecoder,
)

from .snn import (
    SNNCore,
    ConvSNN,
    SNNLinear,
    SNNConv2d,
    ResidualSNNBlock,
    create_snn,
)

__all__ = [
    # Neurons
    "LIFNeuron",
    "AdaptiveLIFNeuron",
    "RecurrentLIFNeuron",
    "get_surrogate",
    "ATanSurrogate",
    "FastSigmoidSurrogate",
    "StraightThroughSurrogate",
    # Encoding
    "RateEncoder",
    "TemporalEncoder",
    "LatencyEncoder",
    "PopulationEncoder",
    "DeltaEncoder",
    "SpikeDecoder",
    # Networks
    "SNNCore",
    "ConvSNN",
    "SNNLinear",
    "SNNConv2d",
    "ResidualSNNBlock",
    "create_snn",
]
