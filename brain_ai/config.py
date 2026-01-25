"""
Configuration for Brain-Inspired AI System.

Centralized hyperparameters and feature flags for all modules.

Production Configuration (~7B parameters total):
================================================
This configuration is designed for production-scale training equivalent
to a 7B parameter model. Parameter distribution:

  - Vision Encoder (ViT-Large scale):     ~300M params
  - Text Encoder (BERT-Large scale):      ~340M params  
  - Audio Encoder (Wav2Vec2-Large):       ~300M params
  - SNN Core (Scaled):                    ~500M params
  - HTM Layer (Scaled):                   ~200M params
  - Global Workspace:                     ~1.5B params
  - Decision Heads:                       ~500M params
  - Symbolic Reasoning:                   ~800M params
  - Meta-Learning:                        ~100M params
  - Engram Memory:                        ~2.5B params
  -----------------------------------------------
  Total:                                  ~7B params

Training Data Requirements (Chinchilla Optimal):
================================================
For 7B parameters, minimum ~140B tokens recommended.
Modern practice (OLMo 2, Llama 3): 4T+ tokens for best results.

Per-modality data requirements:
  - Vision: ImageNet-21K (14M), LAION-400M subset, DataComp
  - Text: RedPajama, The Pile, C4, FineWeb (~1T+ tokens)
  - Audio: LibriSpeech (960h), Common Voice (>10K hours)
  - Multimodal: LAION-5B subset, CC-12M, WebVid-10M
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class SNNConfig:
    """Spiking Neural Network configuration.
    
    Production scale: ~500M parameters
    - Larger hidden layers for capacity
    - More timesteps for better temporal dynamics
    - Optimized surrogate gradient
    """
    beta: float = 0.95  # Higher membrane decay for longer memory
    num_timesteps: int = 50  # More timesteps for rich dynamics
    surrogate: str = "atan"  # atan is most stable for training
    surrogate_alpha: float = 2.0
    dropout: float = 0.1  # Lower dropout for large models
    # Scaled hidden sizes: ~500M params with 4 layers
    hidden_sizes: list[int] = field(default_factory=lambda: [4096, 4096, 2048, 2048])


@dataclass
class EncoderConfig:
    """Modality encoder configuration.
    
    Production scale: ~1B parameters across all encoders
    
    Vision (ViT-Large equivalent): ~300M params
      - 24 layers, 1024 hidden, 16 heads
    
    Text (BERT-Large equivalent): ~340M params
      - 24 layers, 1024 hidden, 16 heads
    
    Audio (Wav2Vec2-Large equivalent): ~300M params
      - 24 layers, 1024 hidden, 16 heads
    """
    output_dim: int = 4096  # Match 7B LLM hidden dim

    # Vision encoder (ViT-Large scale)
    vision_channels: list[int] = field(default_factory=lambda: [256, 512, 1024, 1024])
    vision_num_layers: int = 24
    vision_hidden_dim: int = 1024
    vision_num_heads: int = 16
    vision_mlp_dim: int = 4096
    vision_patch_size: int = 16
    vision_image_size: int = 384  # Higher resolution for production

    # Text encoder (BERT-Large scale -> scaled to 7B proportions)
    text_vocab_size: int = 128000  # Modern tokenizer (Llama 3 scale)
    text_embed_dim: int = 4096  # Match output_dim
    text_num_layers: int = 32  # Deep for 7B equivalent
    text_num_heads: int = 32
    text_ff_dim: int = 14336  # ~3.5x hidden (SwiGLU standard)
    text_max_seq_len: int = 8192  # Long context
    text_rope_theta: float = 500000.0  # Extended RoPE for long context

    # Audio encoder (Wav2Vec2-Large scale)
    audio_n_mels: int = 128  # Higher resolution
    audio_sample_rate: int = 16000
    audio_channels: list[int] = field(default_factory=lambda: [512, 512, 512, 512, 512, 512, 512])
    audio_num_layers: int = 24
    audio_hidden_dim: int = 1024
    audio_num_heads: int = 16

    # Sensors (scaled for RL/robotics)
    sensor_input_dim: int = 512  # Richer sensor input
    sensor_hidden_dim: int = 2048


@dataclass
class HTMConfig:
    """Hierarchical Temporal Memory configuration.
    
    Production scale: ~200M parameters
    - Larger column count for more patterns
    - More cells per column for longer sequences
    - LSTM fallback sized appropriately
    """
    column_count: int = 16384  # 8x increase for production
    cells_per_column: int = 64  # 2x for longer sequences
    sparsity: float = 0.02
    permanence_inc: float = 0.1
    permanence_dec: float = 0.1
    activation_threshold: int = 13

    # Fallback LSTM (production scale)
    use_fallback: bool = True
    lstm_hidden_dim: int = 4096  # Match workspace dim
    lstm_num_layers: int = 4  # Deeper LSTM


@dataclass
class WorkspaceConfig:
    """Global Workspace configuration.
    
    Production scale: ~1.5B parameters
    - Large workspace for multi-modal integration
    - More heads for fine-grained attention
    - Liquid NN for working memory
    """
    workspace_dim: int = 4096  # Match 7B LLM hidden
    num_heads: int = 32  # Multi-head attention
    capacity_limit: int = 7  # Miller's Law (keep biological constraint)

    # Working memory (Liquid NN / CfC)
    memory_hidden_dim: int = 4096
    memory_mode: str = "cfc"  # CfC for efficient dynamics
    memory_num_layers: int = 8  # Deeper for temporal reasoning

    # Cross-modal attention
    cross_modal_heads: int = 16
    cross_modal_layers: int = 6


@dataclass
class DecisionConfig:
    """Decision system configuration.
    
    Production scale: ~500M parameters
    - Active inference for principled action selection
    - Large output heads for many classes
    """
    hidden_dim: int = 4096

    # Active Inference (production scale)
    planning_horizon: int = 8  # Longer planning
    epistemic_weight: float = 1.0
    num_policies: int = 128  # More action options

    # Output heads
    num_classes: int = 10000  # Large-scale classification
    control_dim: int = 32  # Rich control actions
    text_vocab_size: int = 128000  # Match text encoder
    text_decoder_layers: int = 8  # Deeper decoder
    text_decoder_heads: int = 32


@dataclass
class ReasoningConfig:
    """Neuro-symbolic reasoning configuration.
    
    Production scale: ~800M parameters
    - More reasoning steps for complex problems
    - Larger entity/predicate space
    """
    hidden_dim: int = 4096
    num_reasoning_steps: int = 16  # Deeper reasoning
    confidence_threshold: float = 0.8  # Higher threshold for production

    # Symbolic components (production scale)
    num_entities: int = 10000  # Large knowledge base
    num_predicates: int = 1000  # Rich predicate space
    logic_type: str = "product"  # Product t-norm for differentiability
    
    # System 2 reasoning
    system2_layers: int = 8
    system2_heads: int = 16


@dataclass
class MetaConfig:
    """Meta-learning configuration.
    
    Production scale: ~100M parameters
    - Neuromodulation for plasticity control
    - MAML for few-shot adaptation
    """
    num_modulators: int = 8  # More neuromodulatory signals

    # MAML (production scale)
    inner_lr: float = 0.001  # Smaller for large models
    outer_lr: float = 0.0001  # Smaller for stability
    num_inner_steps: int = 10  # More adaptation steps
    first_order: bool = True  # First-order for efficiency at scale

    # Eligibility traces
    trace_decay: float = 0.97  # Longer traces for credit assignment
    
    # Continual learning
    ewc_lambda: float = 1000.0  # Elastic weight consolidation


@dataclass
class EngramConfig:
    """Engram conditional memory configuration.
    
    Production scale: ~2.5B parameters
    - Massive hash table for memory
    - Rich n-gram representations
    """
    # Vocabulary (modern scale)
    vocab_size: int = 128000  # Match modern tokenizers
    compressed_vocab_size: int = 98560  # ~77% of vocab_size

    # Embeddings (production scale)
    embedding_dim: int = 4096  # Match workspace
    ngram_orders: tuple = (2, 3, 4)  # Include 4-grams for production
    num_heads: int = 32
    table_size: int = 100_000_007  # Large prime for hash table (~1B entries effective)

    # Tokenizer
    tokenizer_mode: str = "shared"
    use_compression: bool = True

    # Convolution (scaled)
    conv_kernel_size: int = 7  # Larger receptive field
    conv_dilation: int = 4

    # Memory management
    offload_to_cpu: bool = False
    prefetch: bool = True

    # Gating (stable for production)
    gate_temperature: float = 0.5  # Sharper gating


@dataclass
class TrainingConfig:
    """Training configuration for production scale.
    
    Based on OLMo 2, Llama 3, and modern best practices.
    """
    # Optimizer
    learning_rate: float = 3e-4  # Peak LR
    min_learning_rate: float = 3e-5  # 10% of peak for cosine decay
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    # Schedule
    warmup_steps: int = 2000
    total_steps: int = 1_000_000  # Adjust based on compute
    lr_scheduler: str = "cosine"
    
    # Batch size (scale up with more GPUs)
    batch_size: int = 32  # Per GPU
    gradient_accumulation_steps: int = 16  # Effective batch ~512
    
    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "bfloat16"  # Preferred for modern GPUs
    
    # Distributed
    use_ddp: bool = True
    use_fsdp: bool = True  # For 7B scale
    
    # Checkpointing
    checkpoint_interval: int = 1000
    eval_interval: int = 500
    
    # Data
    num_workers: int = 8
    prefetch_factor: int = 4


@dataclass
class DatasetConfig:
    """Dataset configuration for production training.
    
    Chinchilla optimal for 7B: ~140B tokens minimum
    Modern practice: 1T-4T tokens for best results
    """
    # Text datasets
    text_datasets: list[str] = field(default_factory=lambda: [
        "redpajama",      # 1.2T tokens, diverse web text
        "the_pile",       # 800B tokens, curated mixture
        "c4",             # 365B tokens, cleaned Common Crawl
        "fineweb",        # High-quality filtered web
        "starcoder_data", # Code (for reasoning)
        "arxiv",          # Scientific papers
        "wikipedia",      # Encyclopedic knowledge
        "books3",         # Long-form text
    ])
    
    # Vision datasets
    vision_datasets: list[str] = field(default_factory=lambda: [
        "imagenet21k",    # 14M images, 21K classes
        "laion400m",      # 400M image-text pairs (subset)
        "datacomp_1b",    # 1B curated images
        "cc12m",          # 12M Conceptual Captions
        "coco",           # 330K images with dense annotations
        "visual_genome",  # 108K images with scene graphs
        "openimages",     # 9M images, rich annotations
    ])
    
    # Audio datasets  
    audio_datasets: list[str] = field(default_factory=lambda: [
        "librispeech",    # 960 hours, clean speech
        "common_voice",   # 10K+ hours, multilingual
        "voxpopuli",      # 400K hours, EU Parliament
        "audioset",       # 2M clips, sound events
        "music_caps",     # Music understanding
        "gigaspeech",     # 10K hours, diverse
    ])
    
    # Multimodal datasets
    multimodal_datasets: list[str] = field(default_factory=lambda: [
        "cmu_mosei",      # 23K video clips, sentiment
        "howto100m",      # 136M video clips, instructional
        "webvid10m",      # 10M video-text pairs
        "valor",          # Audio-visual reasoning
        "ego4d",          # Egocentric video understanding
    ])
    
    # Reasoning datasets
    reasoning_datasets: list[str] = field(default_factory=lambda: [
        "gsm8k",          # Grade school math
        "math",           # Competition math
        "arc",            # AI2 Reasoning Challenge
        "hellaswag",      # Commonsense reasoning
        "winogrande",     # Coreference resolution
        "babi",           # 20 reasoning tasks
        "proofwriter",    # Logical deduction
        "folio",          # First-order logic
        "clutrr",         # Compositional reasoning
    ])
    
    # Meta-learning datasets
    metalearning_datasets: list[str] = field(default_factory=lambda: [
        "omniglot",       # Few-shot character recognition
        "mini_imagenet",  # Few-shot image classification
        "tiered_imagenet", # Hierarchical few-shot
        "meta_dataset",   # Large-scale few-shot benchmark
    ])
    
    # RL/Control datasets
    rl_datasets: list[str] = field(default_factory=lambda: [
        "minari",         # Offline RL (D4RL successor)
        "d4rl_antmaze",   # Maze navigation
        "minigrid",       # Grid world planning
        "procgen",        # Procedural environments
    ])
    
    # Token budget (Chinchilla optimal for 7B)
    target_tokens: int = 140_000_000_000  # 140B minimum
    recommended_tokens: int = 1_000_000_000_000  # 1T for better results


@dataclass
class BrainAIConfig:
    """Complete system configuration for 7B-equivalent production model."""

    # Module configs
    snn: SNNConfig = field(default_factory=SNNConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    htm: HTMConfig = field(default_factory=HTMConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    meta: MetaConfig = field(default_factory=MetaConfig)
    engram: EngramConfig = field(default_factory=EngramConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    datasets: DatasetConfig = field(default_factory=DatasetConfig)

    # Feature flags (all enabled for production)
    use_snn: bool = True
    use_htm: bool = True
    use_workspace: bool = True
    use_symbolic: bool = True
    use_meta: bool = True
    use_engram: bool = True  # Enable engram for production
    engram_layer_idx: int = 2  # Deeper integration

    # Training
    learning_rate: float = 3e-4  # Peak learning rate
    batch_size: int = 32  # Per-GPU batch size
    device: str = "cuda"

    # Modalities enabled (all for production)
    modalities: list[str] = field(default_factory=lambda: ["vision", "text", "audio", "sensors"])

    @classmethod
    def for_vision_only(cls) -> "BrainAIConfig":
        """Preset for vision-only experiments (still production scale)."""
        config = cls()
        config.modalities = ["vision"]
        config.use_engram = False
        return config

    @classmethod
    def minimal(cls) -> "BrainAIConfig":
        """Minimal config for testing/debugging (NOT production)."""
        config = cls()
        config.use_htm = False
        config.use_symbolic = False
        config.use_meta = False
        config.use_engram = False
        config.snn.num_timesteps = 10
        config.snn.hidden_sizes = [256, 128]
        config.encoder.output_dim = 512
        config.encoder.text_num_layers = 4
        config.encoder.text_embed_dim = 256
        config.workspace.workspace_dim = 512
        config.decision.hidden_dim = 512
        config.reasoning.hidden_dim = 512
        config.training.batch_size = 8
        return config

    @classmethod  
    def production_7b(cls) -> "BrainAIConfig":
        """Full 7B production configuration (default)."""
        return cls()

    @classmethod
    def production_3b(cls) -> "BrainAIConfig":
        """Reduced 3B configuration for limited resources."""
        config = cls()
        # Scale down by ~half
        config.encoder.output_dim = 2048
        config.encoder.text_num_layers = 16
        config.encoder.text_embed_dim = 2048
        config.encoder.text_ff_dim = 8192
        config.encoder.text_num_heads = 16
        config.workspace.workspace_dim = 2048
        config.workspace.num_heads = 16
        config.decision.hidden_dim = 2048
        config.reasoning.hidden_dim = 2048
        config.snn.hidden_sizes = [2048, 2048, 1024]
        config.engram.embedding_dim = 2048
        return config

    @classmethod
    def production_1b(cls) -> "BrainAIConfig":
        """Compact 1B configuration for efficient deployment."""
        config = cls()
        config.encoder.output_dim = 1024
        config.encoder.text_num_layers = 12
        config.encoder.text_embed_dim = 1024
        config.encoder.text_ff_dim = 4096
        config.encoder.text_num_heads = 8
        config.workspace.workspace_dim = 1024
        config.workspace.num_heads = 8
        config.decision.hidden_dim = 1024
        config.reasoning.hidden_dim = 1024
        config.snn.hidden_sizes = [1024, 1024, 512]
        config.engram.embedding_dim = 1024
        config.engram.table_size = 10_000_003
        return config
