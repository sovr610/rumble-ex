"""
SNN Core Module

Spiking neural network foundation for brain-inspired AI.
"""

from .neurons import (
    LIFNeuron,
    AdaptiveLIFNeuron,
    RecurrentLIFNeuron,
    AdvancedLIFNeuron,  # New: with learnable delays and heterogeneous tau
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

from .losses import (
    prob_spikes_loss,
    spike_rate_regularization,
    spike_rate_range_regularization,
    temporal_consistency_loss,
    temporal_sparsity_loss,
    inter_spike_interval_loss,
    membrane_potential_regularization,
    SNNLoss,
    compute_snn_metrics,
)

__all__ = [
    # Neurons
    "LIFNeuron",
    "AdaptiveLIFNeuron",
    "RecurrentLIFNeuron",
    "AdvancedLIFNeuron",
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
    # Losses (new)
    "prob_spikes_loss",
    "spike_rate_regularization",
    "spike_rate_range_regularization",
    "temporal_consistency_loss",
    "temporal_sparsity_loss",
    "inter_spike_interval_loss",
    "membrane_potential_regularization",
    "SNNLoss",
    "compute_snn_metrics",
]
