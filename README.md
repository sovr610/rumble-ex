# Brain-Inspired AI System

A comprehensive brain-inspired neural architecture that synthesizes cutting-edge 2024-2026 neuroscience and AI research into a unified cognitive system. The system combines spiking neural networks, hierarchical temporal memory, global workspace theory, active inference, neuro-symbolic reasoning, meta-learning, and engram conditional memory into a single modular, trainable framework.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Core Components](#core-components)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Training](#training)
- [Inference](#inference)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Testing](#testing)
- [2025-2026 Research Improvements](#2025-2026-research-improvements)
- [Research References](#research-references)
- [Current Status](#current-status)

---

## Overview

This project implements a multi-layered cognitive architecture that mirrors the organization and function of the human brain. Rather than relying on a single monolithic model, it composes specialized biologically-inspired modules -- each grounded in neuroscience theory -- into an integrated system capable of multi-modal perception, temporal reasoning, deliberate thought, and adaptive learning.

**Core design principles:**

- **Biological Plausibility** -- Components map to known brain regions and neuroscience theories (LIF neurons, neuromodulation, eligibility traces, sparse distributed representations)
- **Multi-Modal Integration** -- Unified processing of vision, text, audio, and sensor data through a shared workspace inspired by consciousness theories
- **Dual-Process Cognition** -- Fast System 1 (automatic) and slow System 2 (deliberate) reasoning paths, routing based on confidence
- **Scalable Architecture** -- Configurable from minimal test setups (~1M params) to production-scale 7B-parameter models
- **Modular Design** -- Each component can be independently enabled/disabled via feature flags and configuration

---

## Architecture

### Data Flow

```
Input -> Encoders -> Global Workspace -> HTM -> Reasoning -> Active Inference -> Output
         |              |                |         |              |
     [Vision]      [Competition]    [Sequence]  [Symbolic]    [Decision]
     [Text]        [Broadcast]     [Anomaly]   [System1/2]   [Heads]
     [Audio]       [Working Mem]   [Prediction] [Metacog]     [Actions]
     [Sensors]         |                                         |
     [Engram]     [Neuromodulation] <---- Feedback ----> [Meta-Learning]
```

### Seven Cognitive Layers

| Layer | Component | Brain Analog | Function | Key Classes |
|-------|-----------|-------------|----------|-------------|
| 1 | **Modality Encoders** | Visual/Auditory/Somatosensory Cortex | Process vision, text, audio, sensor inputs into unified representations | VisionEncoder, TextEncoder, AudioEncoder, SensorEncoder |
| 2 | **SNN Core** | Primary Cortex | Temporal computation with Leaky Integrate-and-Fire neurons, spike-based processing | SNNCore, ConvSNN, LIFNeuron, AdaptiveLIFNeuron |
| 3 | **HTM** | Hippocampus / Entorhinal Cortex | Online sequence learning, prediction, and anomaly detection via sparse distributed representations | HTMLayer, PytorchSpatialPooler, PytorchTemporalMemory |
| 4 | **Global Workspace** | Prefrontal Cortex / Thalamus | Multi-modal competition, information broadcast, and working memory integration | GlobalWorkspace, AttentionCompetition, WorkingMemory |
| 5 | **Active Inference** | Basal Ganglia / Motor Cortex | Goal-directed decision-making via Expected Free Energy minimization | ActiveInferenceAgent, GenerativeModel, Preferences |
| 6 | **Symbolic Reasoning** | Prefrontal / Parietal Cortex | Dual-process (System 1/2) deliberation with fuzzy logic and rule networks | DualProcessReasoner, SymbolicReasoner, FuzzyLogic |
| 7 | **Meta-Learning** | Neuromodulatory Systems (DA, ACh, NE, 5-HT) | Adaptive plasticity control, few-shot learning, eligibility traces | NeuromodulatoryGate, MAML, EligibilityTrace |

### Cognitive Mapping

| Brain Region | AI Component | Neuroscience Basis |
|--------------|--------------|-------------------|
| V1-V4 Visual Cortex | VisionEncoder (spiking CNN) | Hierarchical feature extraction with spike-based processing |
| Wernicke's / Broca's Area | TextEncoder (Transformer) | Language comprehension via multi-head attention |
| Auditory Cortex | AudioEncoder (spiking 1D CNN) | Mel-spectrogram processing with temporal dynamics |
| Somatosensory Cortex | SensorEncoder (Liquid NN) | Proprioceptive/interoceptive signals via continuous-time dynamics |
| Hippocampus | HTMLayer | Episodic memory and sequence learning via Sparse Distributed Representations |
| Prefrontal Cortex | GlobalWorkspace | Executive control, attention-based competition, working memory (Miller's Law: 7+/-2 items) |
| Basal Ganglia | ActiveInferenceAgent | Action selection via Free Energy minimization (Friston) |
| Semantic Memory | EngramModule | O(1) static pattern recall via N-gram hash lookup (DeepSeek) |
| Dopamine System | DopamineSystem | Reward prediction error, motivation, reinforcement |
| Acetylcholine System | AcetylcholineSystem | Novelty detection, attention modulation, learning rate control |
| Norepinephrine System | NorepinephrineSystem | Arousal, global gain modulation, fight-or-flight |
| Serotonin System | SerotoninSystem | Exploration/exploitation balance, temporal discounting |

---

## Core Components

### 1. Spiking Neural Networks (SNN)

The SNN core implements neuromorphic computing with biologically-plausible dynamics:

- **LIF Neurons** (LIFNeuron) -- Leaky Integrate-and-Fire with membrane dynamics: U[t+1] = beta*U[t] + I[t] - S[t]*V_thresh
- **Adaptive LIF** (AdaptiveLIFNeuron) -- Dynamic threshold with spike-frequency adaptation
- **Recurrent LIF** (RecurrentLIFNeuron) -- Lateral recurrent connections within layers
- **Surrogate Gradients** -- ATan, Fast Sigmoid, and Straight-Through estimators for backpropagation through discrete spikes
- **Spike Encoding** -- Rate coding (Bernoulli/Poisson/deterministic), temporal (time-to-first-spike), latency, and population coding
- **SNN Losses** -- ProbSpikes loss (cross-entropy on normalized spike counts), spike rate regularization, temporal consistency, inter-spike interval regularization

**2025 Improvements:**
- Learnable synaptic delays (AdvancedLIFNeuron) for temporal pattern recognition
- Heterogeneous per-neuron time constants for richer dynamics
- ProbSpikes loss for improved training vs. membrane potential loss

```python
from brain_ai.core import SNNCore, ConvSNN, LIFNeuron

# Feedforward SNN
snn = SNNCore(input_size=784, hidden_sizes=[256, 128], output_size=10, num_steps=25)

# Convolutional SNN for vision
conv_snn = ConvSNN(input_channels=1, channels=[32, 64, 128], num_classes=10)
```

### 2. Modality Encoders

Four specialized encoders project different input modalities into a common workspace representation:

| Encoder | Input | Architecture | Notes |
|---------|-------|-------------|-------|
| **VisionEncoder** | Images (B, C, H, W) | Spiking CNN -> Adaptive Pool -> MLP | Supports static images, DVS event streams, video frames. Includes EventVisionEncoder for neuromorphic cameras and MultiScaleVisionEncoder for pyramid pooling. |
| **TextEncoder** | Token IDs (B, seq) | Embedding -> Positional -> Transformer -> Pooling -> MLP | CLS/mean/max pooling. Includes SpikeTextEncoder for spike output and CharacterTextEncoder for character-level processing. |
| **AudioEncoder** | Waveforms (B, samples) | Mel Spectrogram -> Spiking 1D CNN -> Pool -> MLP | Uses torchaudio when available, otherwise manual STFT + mel filterbank. |
| **SensorEncoder** | Sensor streams (B, T, dim) | Liquid Time-Constant (LTC) or Closed-form Continuous-time (CfC) cells | Implements continuous-time RNNs with input-dependent time constants for irregular time series. |
| **EngramTextEncoder** | Token IDs (B, seq) | N-gram Hash -> Pool -> MLP | O(1) pattern retrieval, competes with Transformer in workspace. |

### 3. Hierarchical Temporal Memory (HTM)

Implementation of Numenta's HTM theory for online sequence learning:

- **Spatial Pooler** (PytorchSpatialPooler) -- Converts inputs to Sparse Distributed Representations (SDRs, ~2% sparsity) with competitive inhibition and Hebbian learning with boosting
- **Temporal Memory** (PytorchTemporalMemory) -- Learns sequences by forming connections between cells across columns. Cells represent different temporal contexts for the same spatial pattern. Uses sparse segment storage for memory efficiency.
- **Anomaly Detection** -- Prediction failure rate as anomaly score, with learned anomaly likelihood estimation
- **LSTM Fallback** -- LSTMSequencePredictor and GRUSequencePredictor when htm.core is unavailable, providing compatible interface with prediction and anomaly capabilities

**2025 Improvements:**
- Accelerated HTM with Reflex Memory -- LSH-based O(1) pattern lookup for frequently-seen patterns
- Automatic pattern promotion after configurable threshold matches
- TransformerSequencePredictor as additional fallback option

### 4. Global Workspace Theory (GWT)

Implements the cognitive architecture proposed by Baars (1988) and extended by Dehaene and Changeux (2011):

- **Modality Projections** (ModalityProjection) -- Project each modality into common workspace dimension with learned salience (importance) scores
- **Attention Competition** (AttentionCompetition) -- Multi-head self-attention among modalities with top-K gating for capacity-limited workspace access (Miller's Law: 7+/-2 items)
- **Working Memory** (WorkingMemory) -- Temporal integration using Liquid Neural Networks (CfC/LTC via ncps library) with GRU fallback. Maintains state across timesteps.
- **Information Broadcast** (InformationBroadcast) -- Winners broadcast information back to all specialist modules, enabling global information sharing
- **Integration Layer** -- Combines current workspace content with previous temporal context

**2025 Improvements:**
- Selection-Broadcast cycle with ignition dynamics
- Iterative competition rounds for robust workspace access
- Confidence-gated output

### 5. Active Inference

Decision-making based on Karl Friston's Free Energy Principle:

- **State Encoder** -- Approximate posterior q(s|o) over latent states given observations (Gaussian with learned mean/variance)
- **Generative Model** -- Likelihood model P(o|s) and transition model P(s'|s,a) for imagining consequences of actions
- **Expected Free Energy (EFE)** -- Actions minimize EFE which balances:
  - *Pragmatic value*: How well actions achieve goals (preference satisfaction)
  - *Epistemic value*: How much uncertainty is reduced (information gain)
- **Learnable Preferences** -- Goal specification via learned or fixed preferred observation patterns
- Optional integration with pymdp for discrete state-space active inference

**2025 Improvements:**
- Three-component EFE (pragmatic + epistemic + instrumental/empowerment)
- Amortized action selection for faster inference
- Empowerment estimation as intrinsic motivation

### 6. Neuro-Symbolic Reasoning

Dual-process cognitive architecture combining fast and slow reasoning:

**System 1** (System1Module):
- Fast, automatic, parallel feed-forward network
- Handles familiar patterns with low latency
- Returns output + confidence estimate

**System 2** (System2Module):
- Slow, deliberate, iterative reasoning with GRU-based refinement
- Engages when System 1 confidence < threshold (default 0.7)
- Multi-step reasoning with early stopping on convergence
- Integrates symbolic reasoning for logical inference

**Metacognition** (MetacognitionModule):
- Monitors uncertainty, novelty, and required effort
- Decides when to route between System 1 and System 2

**Symbolic Reasoning** (SymbolicReasoner):
- Differentiable fuzzy logic operations (AND, OR, NOT, IMPLIES, FORALL, EXISTS)
- Three t-norm types: Product, Goedel (min/max), Lukasiewicz
- Entity/predicate embeddings with unary P(x) and binary R(x,y) predicates
- Rule network with learned attention-weighted rule application

**2025 Improvements:**
- Logic Tensor Networks (LTN) for fully differentiable first-order logic
- Real Logic grounding with stable semantics

### 7. Meta-Learning and Neuromodulation

**Meta-Learning** -- Enables rapid adaptation to new tasks:

- **MAML** -- Model-Agnostic Meta-Learning with differentiable inner loop optimization
- **FOMAML** -- First-order approximation (ignores second-order gradients) for efficiency
- **Reptile** -- Simplified meta-learning: moves parameters toward task solutions
- **Eligibility Traces** (EligibilityTrace, EligibilityNetwork) -- Three-factor Hebbian learning (pre x post x neuromodulator) with accumulating, replacing, and Dutch trace types

**Neuromodulation** (NeuromodulatoryGate) -- Four neurotransmitter-inspired systems:

| Modulator | Analog | Signal | Effect |
|-----------|--------|--------|--------|
| **Dopamine** | DA | Reward prediction error | Controls reinforcement learning rate |
| **Acetylcholine** | ACh | Uncertainty / novelty | Increases attention and learning in novel situations |
| **Norepinephrine** | NE | Arousal / urgency | Global gain modulation |
| **Serotonin** | 5-HT | Patience / mood | Exploration vs exploitation balance |

**2025 Improvements:**
- MAML++ with per-layer per-step adaptive learning rates
- Multi-step loss for training stability
- Task2Vec for task embeddings and clustering

### 8. Engram Conditional Memory

Implementation based on DeepSeek's Engram paper (arXiv:2601.07372) -- "Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models":

- **N-gram Hashing** (MultiHeadHash) -- Deterministic multiplicative-XOR hash mapping N-gram token sequences to embedding indices with K independent hash functions per N-gram order to reduce collisions
- **Offloadable Embeddings** (OffloadableEmbedding) -- Embedding tables with optional CPU offload and async CUDA prefetching for large-scale deployment
- **Tokenizer Compression** (TokenizerCompression) -- Surjective vocabulary mapping via NFKC normalization, lowercasing, and whitespace normalization, achieving ~23% vocabulary reduction
- **Context-Aware Gating** (ContextAwareGating) -- Gate retrieved memory using hidden state: alpha = sigma(RMSNorm(h)^T * RMSNorm(Ke) / sqrt(d)), output is alpha * Ve
- **Engram Module** (EngramModule) -- Complete pipeline: N-gram extraction -> multi-head hash lookup -> context-aware gating -> depthwise causal convolution -> residual connection

**Integration modes:**
- *Phase 1* (Encoder-style): EngramTextEncoder competes with Transformer in the Global Workspace
- *Phase 2* (Layer-style): EngramAugmentedLayer embeds Engram at specific transformer layers (Engram -> SNN -> Attention -> FFN)

---

## Key Features

- **Multi-Modal Processing** -- Unified handling of vision (RGB, grayscale, DVS events), text (token-level, character-level), audio (waveforms, mel spectrograms), and sensor data (IMU, proprioception, time-series)
- **Biologically Plausible** -- Spiking neurons, eligibility traces, neuromodulation, sparse distributed representations, Hebbian learning rules
- **Configurable Scale** -- From BrainAIConfig.minimal() (~1M params for testing) through production_1b(), production_3b(), to production_7b() (~7B params)
- **Extensible** -- Modular design with feature flags (use_snn, use_htm, use_workspace, use_symbolic, use_meta, use_engram) for each component
- **Multiple Output Modes** -- Classification, text generation (autoregressive Transformer decoder with top-k/nucleus sampling), and continuous control (Gaussian policy)
- **Stateful Processing** -- Working memory, HTM temporal memory, and neuromodulatory systems maintain state across timesteps
- **CLI Interface** -- Interactive mode, single inference, batch processing, and server deployment
- **Experiment Tracking** -- TensorBoard and Weights & Biases integration

---

## Quick Start

```python
from brain_ai import create_brain_ai
import torch

# Create a multi-modal brain
brain = create_brain_ai(
    modalities=['vision', 'text'],
    output_type='classify',
    num_classes=10,
    device='auto'
)

# Forward pass
output = brain({
    'vision': torch.randn(4, 3, 224, 224),   # (batch, channels, height, width)
    'text': torch.randint(0, 50000, (4, 128))  # (batch, seq_len)
})

# Get detailed analysis with internals
result = brain(
    {'vision': images, 'text': text_ids},
    return_details=True
)
# Returns SystemOutput with: output, workspace, confidence, attention, reasoning_trace, modulators
```

### Convenience Factories

```python
from brain_ai import (
    create_vision_classifier,
    create_multimodal_system,
    create_control_agent
)

# Vision-only classifier (SNN + HTM + Workspace)
model = create_vision_classifier(num_classes=10)

# Multi-modal perception system
model = create_multimodal_system(['vision', 'text', 'audio'])

# Robotic control agent with sensor input
model = create_control_agent(state_dim=32, action_dim=6)
```

### Inference API

```python
from brain_ai import BrainInference

# Load from checkpoint
brain = BrainInference.load('checkpoints/brain_ai_v1.pth')

# Single-modality inference
result = brain.classify_image('path/to/image.jpg')
result = brain.classify_text("Some input text")
result = brain.classify_audio('path/to/audio.wav')

# Multi-modal inference
result = brain.infer({
    'vision': image_tensor,
    'text': "What is in this image?",
})

# Interactive REPL session
brain.interactive()

print(result)
# InferenceResult(
#   prediction=3,
#   confidence=87.5%,
#   modalities=['vision', 'text'],
#   reasoning_used=False
# )
```

---

## Installation

### Prerequisites

- Python 3.9+ (3.10-3.11 recommended for full compatibility)
- PyTorch 2.0+
- CUDA-capable GPU (optional but recommended)
- 8GB+ RAM (16GB+ recommended for full system)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd human-brain

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from brain_ai import create_brain_ai; print('Installation successful!')"
```

### Optional Dependencies

```bash
# HTM Core (native Hierarchical Temporal Memory - uses LSTM fallback if absent)
git clone https://github.com/htm-community/htm.core
cd htm.core && python setup.py install

# Liquid Neural Networks (for CfC/LTC working memory - uses GRU fallback if absent)
pip install ncps

# Active Inference (for discrete state-space models)
pip install inferactively-pymdp

# Meta-Learning (requires Python 3.10-3.11)
pip install learn2learn higher

# Neuromorphic datasets
pip install tonic snntorch spikingjelly
```

### Python Version Notes

| Python Version | Compatibility |
|---------------|---------------|
| 3.10-3.11 | Full compatibility with all packages |
| 3.12+ | Most packages work; learn2learn and avalanche-lib may have Cython issues. Project includes custom MAML implementation that does not require learn2learn. |
| 3.9 | Minimum supported; some type hints may need adjustment |

---

## Configuration

All configuration is centralized in BrainAIConfig which aggregates sub-configs for each module.

### Production Scale Presets

| Preset | Parameters | Use Case |
|--------|-----------|----------|
| BrainAIConfig.minimal() | ~1M | Unit testing, debugging |
| BrainAIConfig.production_1b() | ~1B | Efficient deployment, edge devices |
| BrainAIConfig.production_3b() | ~3B | Resource-limited training |
| BrainAIConfig.production_7b() | ~7B | Full production training |
| BrainAIConfig.for_vision_only() | ~7B (vision) | Vision-only experiments |

**7B Parameter Distribution:**

| Component | Parameters | Notes |
|-----------|-----------|-------|
| Vision Encoder (ViT-Large) | ~300M | 24 layers, 1024 hidden, 16 heads |
| Text Encoder (BERT-Large+) | ~340M | 32 layers, 4096 hidden, 32 heads |
| Audio Encoder (Wav2Vec2-Large) | ~300M | 24 layers, 1024 hidden, 16 heads |
| SNN Core | ~500M | 4 layers [4096, 4096, 2048, 2048] |
| HTM Layer | ~200M | 16K columns, 64 cells/column, LSTM fallback |
| Global Workspace | ~1.5B | 4096-dim workspace, 32 heads, CfC memory |
| Decision Heads | ~500M | Classification, text generation, control |
| Symbolic Reasoning | ~800M | System 1/2, fuzzy logic, rule networks |
| Meta-Learning | ~100M | 4 neuromodulators, MAML |
| Engram Memory | ~2.5B | Multi-head N-gram hash tables |
| **Total** | **~7B** | |

### Configuration Dataclasses

```python
from brain_ai.config import BrainAIConfig

config = BrainAIConfig(
    # Feature flags
    use_snn=True,           # Spiking Neural Networks
    use_htm=True,           # Hierarchical Temporal Memory
    use_workspace=True,     # Global Workspace Theory
    use_symbolic=True,      # Neuro-Symbolic Reasoning
    use_meta=True,          # Meta-Learning / Neuromodulation
    use_engram=True,        # Engram Conditional Memory

    # Modalities to enable
    modalities=['vision', 'text', 'audio', 'sensors'],

    # Training
    learning_rate=3e-4,
    batch_size=32,
    device='cuda',
)

# Sub-configs accessible as attributes:
# config.snn      - SNNConfig (beta, timesteps, surrogate, delays, etc.)
# config.encoder  - EncoderConfig (vision/text/audio/sensor dimensions)
# config.htm      - HTMConfig (column_count, cells_per_column, reflex memory)
# config.workspace - WorkspaceConfig (workspace_dim, heads, CfC/LTC mode)
# config.decision - DecisionConfig (EFE weights, planning horizon, output heads)
# config.reasoning - ReasoningConfig (System2 steps, LTN, fuzzy logic type)
# config.meta     - MetaConfig (MAML inner/outer LR, eligibility traces, Task2Vec)
# config.engram   - EngramConfig (vocab size, N-gram orders, hash table size)
# config.training - TrainingConfig (optimizer, scheduler, AMP, FSDP)
# config.datasets - DatasetConfig (per-modality dataset lists, token budgets)
```

---

## Training

### Seven-Phase Training Pipeline

The system trains incrementally across 7 phases, each building on the previous:

| Phase | Script | Component | Dataset Examples |
|-------|--------|-----------|-----------------|
| 1 | train_phase1.py | SNN Core | MNIST, CIFAR-10, ImageNet-21K |
| 2 | train_phase2.py | Modality Encoders | N-MNIST (DVS), SHD (spiking audio), Tonic datasets |
| 3 | train_phase3.py | HTM Layer | Synthetic sequences, time-series anomaly detection |
| 4 | train_phase4.py | Global Workspace | CMU-MOSEI, multi-modal fusion benchmarks |
| 5 | train_phase5.py | Active Inference | Minari (offline RL), MiniGrid, Gymnasium |
| 6 | train_phase6.py | Neuro-Symbolic Reasoning | bAbI, ProofWriter, FOLIO, GSM8K, ARC |
| 7 | train_phase7.py | Meta-Learning | Omniglot, mini-ImageNet, tiered-ImageNet, CIFAR-FS |

### Training Commands

```bash
# === Development (quick validation on MNIST) ===
python scripts/train_phase1.py --mode dev

# === Production 7B ===
python scripts/train_phase1.py --mode production --dataset imagenet21k --use-amp

# === Multi-GPU (distributed) ===
torchrun --nproc_per_node=8 scripts/train_phase1.py \
    --mode production --dataset imagenet21k --use-amp

# === Full pipeline (all 7 phases sequentially) ===
python scripts/train_full_pipeline.py --mode dev
python scripts/train_full_pipeline.py --mode production --use-amp

# === Resume from specific phase ===
python scripts/train_full_pipeline.py --mode production --start-phase 4
```

### Dataset Requirements

**Chinchilla-optimal for 7B parameters:** minimum ~140B tokens, recommended 1T+ tokens.

| Modality | Datasets | Scale |
|----------|----------|-------|
| **Text** | RedPajama (1.2T tokens), The Pile (800B), C4 (365B), FineWeb, StarCoder, ArXiv, Wikipedia, Books3 | 1T+ tokens |
| **Vision** | ImageNet-21K (14M imgs), LAION-400M, DataComp-1B, CC-12M, COCO, Visual Genome, OpenImages | 1B+ images |
| **Audio** | LibriSpeech (960h), Common Voice (10K+ h), VoxPopuli (400K h), AudioSet (2M clips), GigaSpeech | 10K+ hours |
| **Multimodal** | CMU-MOSEI, HowTo100M, WebVid-10M, VALOR, Ego4D | 10M+ samples |
| **Reasoning** | GSM8K, MATH, ARC, HellaSwag, WinoGrande, bAbI, ProofWriter, FOLIO, CLUTRR | - |
| **Meta-Learning** | Omniglot, mini-ImageNet, tiered-ImageNet, Meta-Dataset | - |
| **RL/Control** | Minari, D4RL-AntMaze, MiniGrid, ProcGen | - |

---

## Inference

### Python API

```python
from brain_ai import BrainInference, create_brain_ai

# Create or load model
brain = BrainInference.load('checkpoints/model.pth', device='auto')
# or: brain = BrainInference(model=create_brain_ai(...))

# Classification
result = brain.classify_image('photo.jpg', top_k=5)
result = brain.classify_text("Hello world", top_k=5)
result = brain.classify_audio('speech.wav', top_k=5)

# Text generation
output = brain.generate(prompt="Once upon a time", max_length=100, temperature=0.8)

# Batch processing
results = brain.batch_classify(['img1.jpg', 'img2.jpg'], batch_size=32)

# Multi-modal with detailed output
result = brain.infer({
    'vision': image_tensor,
    'text': "Describe this image",
}, return_details=True)

# Access detailed results
print(f"Prediction: {result.prediction}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Modalities used: {result.modalities_used}")
print(f"Reasoning used: {result.reasoning_used}")
print(f"Anomaly score: {result.anomaly_score}")
print(f"Inference time: {result.inference_time_ms:.1f}ms")
```

### CLI Interface

```bash
# Interactive REPL
python -m brain_ai.cli interactive

# Classify single inputs
python -m brain_ai.cli classify --image photo.jpg --top-k 5
python -m brain_ai.cli classify --text "Some text" --top-k 5
python -m brain_ai.cli classify --audio speech.wav

# Text generation
python -m brain_ai.cli generate --prompt "Once upon a time" --max-length 100 --temperature 0.8

# Batch processing
python -m brain_ai.cli batch --input "images/*.jpg" --output results.json --batch-size 32

# Start inference server
python -m brain_ai.cli serve --host 0.0.0.0 --port 8000

# Model info
python -m brain_ai.cli info

# Run demo
python -m brain_ai.cli demo --modality all
```

---

## Project Structure

```
brain_ai/
  __init__.py             # Public API exports (all major classes/factories)
  config.py               # Centralized configuration (10 dataclasses, presets)
  system.py               # BrainAI orchestrator (main nn.Module, forward pass)
  inference.py            # BrainInference high-level API (load, classify, generate)
  cli.py                  # Command-line interface (interactive, classify, serve, etc.)

  core/                   # Spiking Neural Network implementation
    neurons.py            # LIF neurons (Standard, Adaptive, Recurrent) + surrogate gradients
    snn.py                # SNNCore (feedforward), ConvSNN (convolutional), SNNLinear
    encoding.py           # Spike encoders (Rate, Temporal, Latency, Population) + decoder
    losses.py             # ProbSpikes loss, spike rate/temporal/ISI regularization

  encoders/               # Multi-modal input processing
    vision.py             # VisionEncoder, EventVisionEncoder, MultiScaleVisionEncoder
    text.py               # TextEncoder, SpikeTextEncoder, CharacterTextEncoder
    audio.py              # AudioEncoder, MelSpectrogramFrontend
    sensors.py            # SensorEncoder (LTC/CfC cells), LiquidTimeConstant
    engram_encoder.py     # EngramTextEncoder (Phase 1 integration)

  temporal/               # Temporal sequence processing
    htm.py                # HTMLayer, PytorchSpatialPooler, PytorchTemporalMemory, SparseTensor
    sequence.py           # LSTM/GRU/Transformer fallback sequence predictors

  workspace/              # Global Workspace Theory implementation
    global_workspace.py   # GlobalWorkspace, AttentionCompetition, InformationBroadcast
    working_memory.py     # WorkingMemory (CfC/LTC/GRU backends), LiquidWorkingMemory

  decision/               # Decision-making and output
    active_inference.py   # ActiveInferenceAgent, GenerativeModel, StateEncoder, Preferences
    output_heads.py       # ClassificationHead, TextDecoderHead, ContinuousControlHead

  reasoning/              # Neuro-symbolic reasoning
    symbolic.py           # SymbolicReasoner, FuzzyLogic, PredicateEncoder, RuleNetwork
    system2.py            # DualProcessReasoner, System1Module, System2Module, Metacognition

  meta/                   # Meta-learning and plasticity control
    neuromodulation.py    # NeuromodulatoryGate (DA, ACh, NE, 5-HT systems)
    maml.py               # MAML, FOMAML, Reptile, InnerLoopOptimizer
    eligibility.py        # EligibilityTrace, EligibilityNetwork, EligibilityMLP

  memory/                 # Engram conditional memory system
    engram.py             # EngramModule, EngramEmbedding, ContextAwareGating, RMSNorm
    hash_embedding.py     # MultiHeadHash, OffloadableEmbedding (CPU offload + prefetch)
    tokenizer_compression.py  # TokenizerCompression (NFKC normalization, ~23% reduction)

  layers/                 # Composite layers
    engram_layer.py       # EngramAugmentedLayer (Engram -> SNN -> Attention -> FFN)

  datasets/               # Dataset loaders for each training phase
    phase1_snn.py         # MNIST, CIFAR, ImageNet loaders
    phase2_event.py       # Neuromorphic/event-driven datasets (Tonic)
    phase3_htm.py         # Sequence/time-series datasets
    phase4_multimodal.py  # Multi-modal fusion datasets
    phase5_active_inference.py  # RL/control datasets (Minari, Gymnasium)
    phase6_neurosymbolic.py     # Reasoning datasets (bAbI, FOLIO, GSM8K)
    phase7_metalearning.py      # Few-shot datasets (Omniglot, mini-ImageNet)
    utils.py              # Shared data utilities

scripts/
  train_full_pipeline.py  # Orchestrate all 7 training phases
  train_phase1.py         # Phase 1: SNN Core (MNIST/ImageNet)
  train_phase2.py         # Phase 2: Modality Encoders
  train_phase3.py         # Phase 3: HTM Layer
  train_phase4.py         # Phase 4: Global Workspace
  train_phase5.py         # Phase 5: Active Inference
  train_phase6.py         # Phase 6: Neuro-Symbolic Reasoning
  train_phase7.py         # Phase 7: Meta-Learning

checkpoints/              # Saved model checkpoints
  snn_mnist.pth           # Phase 1 SNN trained on MNIST
  vision_encoder.pth      # Phase 2 vision encoder
  htm_layer.pth           # Phase 3 HTM
  global_workspace.pth    # Phase 4 workspace
  active_inference.pth    # Phase 5 decision system
  neuro_symbolic.pth      # Phase 6 reasoning
  meta_learning.pth       # Phase 7 meta-learning

tests/
  test_snn.py             # SNN unit tests
  test_engram.py          # Engram integration tests

docs/
  ENGRAM_INTEGRATION_GUIDE.md
  DATASET_LOADERS_GUIDE.md
  GPU_TRAINING_COMMANDS.md
  INFERENCE_COMMANDS.md
  INFERENCE_GUIDE.md
  PRODUCTION_TRAINING_GUIDE.md
  TRAINING_COMMANDS.md
  TRAINING_DATASETS_RESEARCH.md
  plans/
    2026-01-17-brain-ai-design.md
    2026-01-18-engram-implementation-plan.md
    2026-01-18-engram-integration-design.md
    2026-01-28-model-improvements.md

examples/
  inference_demo.py       # Complete inference demonstration
```

---

## Dependencies

### Core (Required)

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >=2.0.0 | Core deep learning framework |
| torchvision | >=0.15.0 | Vision datasets and transforms |
| torchaudio | >=2.0.0 | Audio processing |
| numpy | >=1.24.0 | Numerical computing |
| scipy | >=1.10.0 | Scientific computing |
| einops | >=0.6.0 | Tensor operations |

### Neuromorphic / SNN

| Package | Version | Purpose |
|---------|---------|---------|
| snntorch | >=0.7.0 | Spiking neural network utilities |
| spikingjelly | >=0.0.0.0.14 | SNN framework |
| tonic | >=1.0.0 | Neuromorphic datasets (N-MNIST, DVS-CIFAR, SHD) |

### Temporal / Working Memory

| Package | Version | Purpose |
|---------|---------|---------|
| ncps | >=0.0.7 | Liquid Neural Networks (CfC, LTC, NCP wirings) |
| pyod | >=1.0.0 | Anomaly detection utilities |

### Decision / RL

| Package | Version | Purpose |
|---------|---------|---------|
| inferactively-pymdp | >=0.0.7.1 | Active Inference (discrete POMDP) |
| minari | >=0.4.0 | Offline RL datasets |
| gymnasium | >=0.29.0 | RL environments |
| minigrid | >=2.3.0 | Grid-world planning |

### Data / NLP

| Package | Version | Purpose |
|---------|---------|---------|
| datasets | >=2.14.0 | HuggingFace datasets (bAbI, ProofWriter, etc.) |
| transformers | >=4.30.0 | Pretrained models and tokenizers |
| h5py | >=3.8.0 | HDF5 data loading |
| pandas | >=2.0.0 | Data manipulation |

### Meta-Learning

| Package | Version | Purpose |
|---------|---------|---------|
| higher | >=0.2.1 | Differentiable inner-loop optimization |
| learn2learn | optional | Meta-learning benchmarks (Python 3.10-3.11 only) |

### Training / Monitoring

| Package | Version | Purpose |
|---------|---------|---------|
| tensorboard | >=2.12.0 | Training visualization |
| wandb | >=0.15.0 | Experiment tracking |
| matplotlib | >=3.7.0 | Plotting |
| tqdm | >=4.65.0 | Progress bars |

### Testing

| Package | Version | Purpose |
|---------|---------|---------|
| pytest | >=7.3.0 | Test framework |
| pytest-cov | >=4.1.0 | Coverage reporting |

See requirements.txt for the complete dependency list.

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test files
python -m pytest tests/test_snn.py -v
python -m pytest tests/test_engram.py -v

# With coverage report
python -m pytest tests/ --cov=brain_ai --cov-report=html

# Quick smoke test
python -c "
from brain_ai import create_brain_ai
import torch
brain = create_brain_ai(modalities=['vision'], output_type='classify', num_classes=10)
out = brain({'vision': torch.randn(2, 1, 28, 28)})
print(f'Output shape: {out.shape}, Predictions: {out.argmax(dim=1)}')
print('All systems operational!')
"
```

---

## 2025-2026 Research Improvements

All improvements documented in docs/plans/2026-01-28-model-improvements.md have been implemented:

| Module | Improvement | Config Flag | Reference |
|--------|------------|-------------|-----------|
| **SNN** | Learnable synaptic delays | use_learnable_delays=True | Meszaros et al. (2024), Hammouamri et al. (2024) |
| **SNN** | Heterogeneous per-neuron time constants | use_heterogeneous_tau=True | Diversity in SNN dynamics |
| **SNN** | ProbSpikes loss | use_probspikes_loss=True | Shrestha and Orchard (2018), spike count CE |
| **HTM** | Accelerated HTM with Reflex Memory | use_reflex_memory=True | LSH-based O(1) pattern lookup |
| **Workspace** | Selection-Broadcast with ignition | use_selection_broadcast=True | Dehaene and Changeux (2011), GNW |
| **Active Inference** | Three-component EFE | use_improved_efe=True | Friston et al., pragmatic+epistemic+instrumental |
| **Active Inference** | Empowerment as intrinsic motivation | use_empowerment=True | Salge et al. (2014) |
| **Reasoning** | Logic Tensor Networks | use_ltn=True | Badreddine et al. (2022), Real Logic |
| **Meta** | MAML++ | use_maml_plus_plus=True | Antoniou et al. (2018), per-layer LRs |
| **Meta** | Task2Vec embeddings | use_task2vec=True | Achille et al. (2019) |

---

## Research References

This project synthesizes research from multiple domains:

### Spiking Neural Networks
- **LIF Neurons and Surrogate Gradients**: Neftci, E. O., Mostafa, H. and Zenke, F. "Surrogate gradient learning in spiking neural networks." *IEEE Signal Processing Magazine* 36, 51-63 (2019)
- **Surrogate Gradient Robustness**: Zenke, F. and Vogels, T. P. "The remarkable robustness of surrogate gradient learning for instilling complex function in spiking neural networks." *Neural Computation* 33, 899-925 (2021)
- **Learnable Delays**: Hammouamri, I., Khalfaoui-Hassani, I. and Masquelier, T. "Learning delays in spiking neural networks using dilated convolutions with learnable spacings." *ICLR* (2024)
- **Delay Learning via Gradients**: Meszaros, B., Knight, J. C. and Nowotny, T. "Learning delays through gradients and structure." *Frontiers in Computational Neuroscience* 18:1460309 (2024)
- **SLAYER**: Shrestha, S. B. and Orchard, G. "SLAYER: Spike layer error reassignment in time." *NeurIPS* (2018)
- **snntorch**: Eshraghian, J. K. et al. "Training spiking neural networks using lessons from deep learning." *Proceedings of the IEEE* (2023)

### Hierarchical Temporal Memory
- **HTM Theory**: Hawkins, J. and Blakeslee, S. *On Intelligence* (2004)
- **HTM Spatial Pooler**: Ahmad, S. and Hawkins, J. "Properties of sparse distributed representations and their application to hierarchical temporal memory." *arXiv:1503.07469* (2015)
- **HTM Temporal Memory**: Hawkins, J. and Ahmad, S. "Why neurons have thousands of synapses, a theory of sequence memory in neocortex." *Frontiers in Neural Circuits* 10:23 (2016)
- **htm.core**: https://github.com/htm-community/htm.core

### Global Workspace Theory
- **GWT Origin**: Baars, B. J. *A Cognitive Theory of Consciousness* (1988)
- **Global Neuronal Workspace**: Dehaene, S. and Changeux, J.-P. "Experimental and theoretical approaches to conscious processing." *Neuron* 70(2), 200-227 (2011)
- **Deep Learning and GWT**: VanRullen, R. and Kanai, R. "Deep learning and the Global Workspace Theory." *Trends in Cognitive Sciences* 25(11), 956-968 (2021)

### Active Inference and Free Energy Principle
- **Free Energy Principle**: Friston, K. "The free-energy principle: a unified brain theory?" *Nature Reviews Neuroscience* 11, 127-138 (2010)
- **Active Inference**: Friston, K. et al. "Active inference and epistemic value." *Cognitive Neuroscience* 6(4), 187-214 (2015)
- **pymdp**: Heins, C. et al. "pymdp: A Python library for active inference in discrete state spaces." *Journal of Open Source Software* 7(73), 4098 (2022). https://github.com/infer-actively/pymdp
- **Empowerment**: Salge, C., Glackin, C. and Polani, D. "Empowerment -- an introduction." *Guided Self-Organization: Inception* (2014)

### Liquid Neural Networks
- **LTC Networks**: Hasani, R. et al. "Liquid time-constant networks." *AAAI* 35(9), 7657-7666 (2021)
- **CfC Networks**: Hasani, R. et al. "Closed-form continuous-time neural networks." *Nature Machine Intelligence* 4, 992-1003 (2022)
- **Neural Circuit Policies**: Lechner, M. et al. "Neural circuit policies enabling auditable autonomy." *Nature Machine Intelligence* 2, 642-652 (2020)
- **ncps**: https://ncps.readthedocs.io

### Neuro-Symbolic Reasoning
- **Logic Tensor Networks**: Badreddine, S. et al. "Logic Tensor Networks." *Artificial Intelligence* 303, 103649 (2022). https://github.com/logictensornetworks/logictensornetworks
- **Dual-Process Theory**: Kahneman, D. *Thinking, Fast and Slow* (2011)
- **Neuro-Symbolic AI**: d'Avila Garcez, A. and Lamb, L. C. "Neurosymbolic AI: the 3rd wave." *Artificial Intelligence Review* 56, 12387-12406 (2023)

### Meta-Learning
- **MAML**: Finn, C., Abbeel, P. and Levine, S. "Model-agnostic meta-learning for fast adaptation of deep networks." *ICML* (2017)
- **MAML++**: Antoniou, A., Edwards, H. and Storkey, A. "How to train your MAML." *ICLR* (2019)
- **Reptile**: Nichol, A., Achiam, J. and Schulman, J. "On first-order meta-learning algorithms." *arXiv:1803.02999* (2018)
- **Task2Vec**: Achille, A. et al. "Task2Vec: Task embedding for meta-learning." *ICCV* (2019)

### Engram Conditional Memory
- **Engram Paper**: DeepSeek-AI. "Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models." *arXiv:2601.07372* (2026). https://github.com/deepseek-ai/Engram
- **Key insight**: N-gram hash tables provide O(1) static pattern retrieval, freeing transformer depth for compositional reasoning. Engram-27B demonstrates improvements over MoE baselines across knowledge, reasoning, code, and math domains under iso-parameter and iso-FLOPs constraints.

### Neuromodulation
- **Dopamine and RPE**: Schultz, W. "Dopamine reward prediction error coding." *Dialogues in Clinical Neuroscience* 18(1), 23-32 (2016)
- **Three-Factor Learning Rules**: Gerstner, W. et al. "Eligibility traces and plasticity on behavioral time scales." *Frontiers in Neural Circuits* 12:53 (2018)
- **Working Memory Capacity**: Miller, G. A. "The magical number seven, plus or minus two." *Psychological Review* 63(2), 81-97 (1956)

---

## Current Status

### Implemented and Tested
- Core SNN (SNNCore, ConvSNN, LIFNeuron variants, all surrogate gradients)
- All modality encoders (Vision, Text, Audio, Sensor, Engram)
- Multiple spike encoding schemes (Rate, Temporal, Latency, Population)
- SNN-specific losses (ProbSpikes, spike rate regularization, temporal consistency)
- HTM layer with pure PyTorch implementation (Spatial Pooler + Temporal Memory)
- LSTM/GRU/Transformer sequence predictor fallbacks
- Global Workspace with attention competition + working memory
- Liquid Neural Network working memory (CfC/LTC with GRU fallback)
- Active Inference decision system (state encoder, generative model, EFE)
- Output heads (Classification, Text Generation with sampling, Continuous Control)
- Dual-process symbolic reasoning (System 1/2 with metacognition)
- Neuro-symbolic reasoning (fuzzy logic, predicate encoders, rule networks)
- Meta-learning (MAML, FOMAML, Reptile)
- Neuromodulation (4 neurotransmitter systems)
- Eligibility traces (accumulating, replacing, Dutch)
- Engram memory (N-gram hashing, context-aware gating, tokenizer compression)
- Engram-augmented layer (Phase 2 layer-style integration)
- 7-phase training pipeline with dev and production modes
- CLI with interactive, classify, generate, batch, and serve commands
- Inference API with checkpoint loading and multi-modal support
- MNIST training pipeline validated

### Under Development
- End-to-end production training at 7B scale
- Benchmark evaluations across all reasoning datasets
- Deployment optimization (quantization, pruning, distillation)

---

## Documentation

- [INSTRUCTIONS.md](INSTRUCTIONS.md) - How to run, train, and perform inference
- [docs/ENGRAM_INTEGRATION_GUIDE.md](docs/ENGRAM_INTEGRATION_GUIDE.md) - Detailed Engram architecture
- [docs/DATASET_LOADERS_GUIDE.md](docs/DATASET_LOADERS_GUIDE.md) - Dataset loading for all phases
- [docs/GPU_TRAINING_COMMANDS.md](docs/GPU_TRAINING_COMMANDS.md) - GPU training reference
- [docs/PRODUCTION_TRAINING_GUIDE.md](docs/PRODUCTION_TRAINING_GUIDE.md) - Production-scale training
- [docs/INFERENCE_GUIDE.md](docs/INFERENCE_GUIDE.md) - Inference API documentation
- [docs/TRAINING_DATASETS_RESEARCH.md](docs/TRAINING_DATASETS_RESEARCH.md) - Dataset research and recommendations
- [docs/plans/](docs/plans/) - Architecture design documents and improvement plans

---

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
