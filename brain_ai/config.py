"""
Configuration for Brain-Inspired AI System.

Centralized hyperparameters and feature flags for all modules.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SNNConfig:
    """Spiking Neural Network configuration."""
    beta: float = 0.9  # Membrane decay
    num_timesteps: int = 25
    surrogate: str = "atan"  # atan, fast_sigmoid, straight_through
    surrogate_alpha: float = 2.0
    dropout: float = 0.2
    hidden_sizes: list[int] = field(default_factory=lambda: [512, 256, 128])


@dataclass
class EncoderConfig:
    """Modality encoder configuration."""
    output_dim: int = 512

    # Vision
    vision_channels: list[int] = field(default_factory=lambda: [32, 64, 128])

    # Text
    text_vocab_size: int = 30000
    text_embed_dim: int = 256
    text_num_layers: int = 4
    text_num_heads: int = 4

    # Audio
    audio_n_mels: int = 80
    audio_sample_rate: int = 16000
    audio_channels: list[int] = field(default_factory=lambda: [128, 256])

    # Sensors
    sensor_hidden_dim: int = 256


@dataclass
class HTMConfig:
    """Hierarchical Temporal Memory configuration."""
    column_count: int = 2048
    cells_per_column: int = 32
    sparsity: float = 0.02
    permanence_inc: float = 0.1
    permanence_dec: float = 0.1
    activation_threshold: int = 13

    # Fallback to LSTM if htm.core unavailable
    use_fallback: bool = True
    lstm_hidden_dim: int = 256


@dataclass
class WorkspaceConfig:
    """Global Workspace configuration."""
    workspace_dim: int = 512
    num_heads: int = 8
    capacity_limit: int = 7  # Miller's Law

    # Working memory (Liquid NN)
    memory_hidden_dim: int = 512
    memory_mode: str = "cfc"  # cfc or ltc


@dataclass
class DecisionConfig:
    """Decision system configuration."""
    hidden_dim: int = 512

    # Active Inference
    planning_horizon: int = 3
    epistemic_weight: float = 1.0

    # Output heads
    num_classes: int = 10  # Override per task
    control_dim: int = 6  # Override per task
    text_vocab_size: int = 30000
    text_decoder_layers: int = 2


@dataclass
class ReasoningConfig:
    """Neuro-symbolic reasoning configuration."""
    hidden_dim: int = 512
    num_reasoning_steps: int = 5
    confidence_threshold: float = 0.7

    # Symbolic
    num_entities: int = 100
    num_predicates: int = 50
    logic_type: str = "product"  # product or godel


@dataclass
class MetaConfig:
    """Meta-learning configuration."""
    num_modulators: int = 4

    # MAML
    inner_lr: float = 0.01
    outer_lr: float = 0.001
    num_inner_steps: int = 5
    first_order: bool = False

    # Eligibility traces
    trace_decay: float = 0.95


@dataclass
class EngramConfig:
    """Engram conditional memory configuration."""
    # Vocabulary
    vocab_size: int = 50000
    compressed_vocab_size: int = 38500  # ~77% of vocab_size

    # Embeddings
    embedding_dim: int = 256
    ngram_orders: tuple = (2, 3)  # Bigrams and trigrams
    num_heads: int = 4
    table_size: int = 10_000_003  # Prime, production scale

    # Tokenizer
    tokenizer_mode: str = "shared"
    use_compression: bool = True

    # Convolution
    conv_kernel_size: int = 4
    conv_dilation: int = 3

    # Offloading
    offload_to_cpu: bool = False
    prefetch: bool = True

    # Gating
    gate_temperature: float = 1.0


@dataclass
class BrainAIConfig:
    """Complete system configuration."""

    # Module configs
    snn: SNNConfig = field(default_factory=SNNConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    htm: HTMConfig = field(default_factory=HTMConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    meta: MetaConfig = field(default_factory=MetaConfig)
    engram: EngramConfig = field(default_factory=EngramConfig)

    # Feature flags
    use_snn: bool = True
    use_htm: bool = True
    use_workspace: bool = True
    use_symbolic: bool = True
    use_meta: bool = True
    use_engram: bool = False
    engram_layer_idx: int = 1  # Which layer for Phase 2 integration

    # Training
    learning_rate: float = 1e-3
    batch_size: int = 32
    device: str = "cuda"  # cuda or cpu

    # Modalities enabled
    modalities: list[str] = field(default_factory=lambda: ["vision", "text", "audio", "sensors"])

    @classmethod
    def for_vision_only(cls) -> "BrainAIConfig":
        """Preset for vision-only experiments."""
        config = cls()
        config.modalities = ["vision"]
        return config

    @classmethod
    def minimal(cls) -> "BrainAIConfig":
        """Minimal config for testing."""
        config = cls()
        config.use_htm = False
        config.use_symbolic = False
        config.use_meta = False
        config.snn.num_timesteps = 10
        config.snn.hidden_sizes = [256, 128]
        return config
