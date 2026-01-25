"""
Brain-Inspired AI System

A comprehensive brain-inspired artificial intelligence architecture that combines:
- Spiking Neural Networks (SNN)
- Hierarchical Temporal Memory (HTM)
- Global Workspace Theory
- Active Inference Decision-Making
- Neuro-Symbolic Reasoning
- Meta-Learning with Neuromodulation

Usage:
    from brain_ai import BrainAI, create_brain_ai

    # Create system
    brain = create_brain_ai(
        modalities=['vision', 'text'],
        output_type='classify',
        num_classes=10,
    )

    # Forward pass
    output = brain({
        'vision': images,
        'text': text_embeddings,
    })
"""

__version__ = "0.1.0"

# Core components
from .config import BrainAIConfig

# System
from .system import (
    BrainAI,
    SystemOutput,
    create_brain_ai,
    create_vision_classifier,
    create_multimodal_system,
    create_control_agent,
)

# Inference
from .inference import (
    BrainInference,
    InferenceResult,
)

# Core SNN
from .core import (
    SNNCore,
    ConvSNN,
    SNNLinear,
    SNNConv2d,
    ResidualSNNBlock,
    LIFNeuron,
    AdaptiveLIFNeuron,
    RecurrentLIFNeuron,
    RateEncoder,
    SpikeDecoder,
)

# Encoders
from .encoders import (
    VisionEncoder,
    TextEncoder,
    AudioEncoder,
    SensorEncoder,
    create_vision_encoder,
    create_text_encoder,
    create_audio_encoder,
    create_sensor_encoder,
)
from .encoders.engram_encoder import EngramTextEncoder, create_engram_encoder

# Memory systems (Engram)
from .memory import (
    EngramConfig,
    EngramModule,
    EngramEmbedding,
    ContextAwareGating,
    RMSNorm,
    TokenizerCompression,
    MultiHeadHash,
    OffloadableEmbedding,
)

# Layers
from .layers import EngramAugmentedLayer, create_engram_layer

# Temporal
from .temporal import (
    HTMLayer,
    HTMConfig,
    PytorchSpatialPooler,
    PytorchTemporalMemory,
    create_htm_layer,
)

# Workspace
from .workspace import (
    GlobalWorkspace,
    GlobalWorkspaceConfig,
    WorkingMemory,
    WorkingMemoryConfig,
    create_global_workspace,
    create_working_memory,
)

# Decision
from .decision import (
    ActiveInferenceAgent,
    ActiveInferenceConfig,
    DecisionHeads,
    OutputHeadsConfig,
    ClassificationHead,
    TextDecoderHead,
    ContinuousControlHead,
    create_active_inference_agent,
    create_decision_heads,
)

# Reasoning
from .reasoning import (
    SymbolicReasoner,
    SymbolicConfig,
    DualProcessReasoner,
    System2Config,
    FuzzyLogic,
    create_symbolic_reasoner,
    create_dual_process_reasoner,
)

# Meta-Learning
from .meta import (
    NeuromodulatoryGate,
    NeuromodulationConfig,
    MAML,
    MAMLConfig,
    FOMAML,
    Reptile,
    EligibilityTrace,
    EligibilityConfig,
    EligibilityMLP,
    create_neuromodulatory_gate,
    create_meta_learner,
    create_eligibility_network,
)

__all__ = [
    # Version
    '__version__',
    # Config
    'BrainAIConfig',
    # System
    'BrainAI',
    'SystemOutput',
    'create_brain_ai',
    'create_vision_classifier',
    'create_multimodal_system',
    'create_control_agent',
    # Core
    'SNNCore',
    'ConvSNN',
    'SNNLinear',
    'SNNConv2d',
    'ResidualSNNBlock',
    'LIFNeuron',
    'AdaptiveLIFNeuron',
    'RecurrentLIFNeuron',
    'RateEncoder',
    'SpikeDecoder',
    # Encoders
    'VisionEncoder',
    'TextEncoder',
    'AudioEncoder',
    'SensorEncoder',
    'create_vision_encoder',
    'create_text_encoder',
    'create_audio_encoder',
    'create_sensor_encoder',
    'EngramTextEncoder',
    'create_engram_encoder',
    # Memory (Engram)
    'EngramConfig',
    'EngramModule',
    'EngramEmbedding',
    'ContextAwareGating',
    'RMSNorm',
    'TokenizerCompression',
    'MultiHeadHash',
    'OffloadableEmbedding',
    # Layers
    'EngramAugmentedLayer',
    'create_engram_layer',
    # Temporal
    'HTMLayer',
    'HTMConfig',
    'PytorchSpatialPooler',
    'PytorchTemporalMemory',
    'create_htm_layer',
    # Workspace
    'GlobalWorkspace',
    'GlobalWorkspaceConfig',
    'WorkingMemory',
    'WorkingMemoryConfig',
    'create_global_workspace',
    'create_working_memory',
    # Decision
    'ActiveInferenceAgent',
    'ActiveInferenceConfig',
    'DecisionHeads',
    'OutputHeadsConfig',
    'ClassificationHead',
    'TextDecoderHead',
    'ContinuousControlHead',
    'create_active_inference_agent',
    'create_decision_heads',
    # Reasoning
    'SymbolicReasoner',
    'SymbolicConfig',
    'DualProcessReasoner',
    'System2Config',
    'FuzzyLogic',
    'create_symbolic_reasoner',
    'create_dual_process_reasoner',
    # Meta
    'NeuromodulatoryGate',
    'NeuromodulationConfig',
    'MAML',
    'MAMLConfig',
    'FOMAML',
    'Reptile',
    'EligibilityTrace',
    'EligibilityConfig',
    'EligibilityMLP',
    'create_neuromodulatory_gate',
    'create_meta_learner',
    'create_eligibility_network',
    # Inference
    'BrainInference',
    'InferenceResult',
]
