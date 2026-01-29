"""
Decision System Module

Active inference action selection and output heads.
"""

from .active_inference import (
    ActiveInferenceAgent,
    ActiveInferenceConfig,
    ContinuousActiveInference,
    StateEncoder,
    GenerativeModel,
    Preferences,
    create_active_inference_agent,
    # Improved EFE (2025)
    EmpowermentEstimator,
    ImprovedEFEComputation,
    ImprovedActiveInferenceAgent,
)
from .output_heads import (
    DecisionHeads,
    OutputHeadsConfig,
    ClassificationHead,
    TextDecoderHead,
    ContinuousControlHead,
    create_decision_heads,
)

__all__ = [
    # Active Inference
    'ActiveInferenceAgent',
    'ActiveInferenceConfig',
    'ContinuousActiveInference',
    'StateEncoder',
    'GenerativeModel',
    'Preferences',
    'create_active_inference_agent',
    # Improved EFE (2025)
    'EmpowermentEstimator',
    'ImprovedEFEComputation',
    'ImprovedActiveInferenceAgent',
    # Output Heads
    'DecisionHeads',
    'OutputHeadsConfig',
    'ClassificationHead',
    'TextDecoderHead',
    'ContinuousControlHead',
    'create_decision_heads',
]
