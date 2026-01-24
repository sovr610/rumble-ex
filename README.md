# Brain-Inspired AI System

A comprehensive brain-inspired neural architecture that synthesizes cutting-edge 2024-2025 neuroscience and AI research into a unified cognitive system.

## Overview

This project implements a multi-layered cognitive architecture combining:

- **Spiking Neural Networks (SNNs)** - Neuromorphic computing with temporal dynamics
- **Hierarchical Temporal Memory (HTM)** - Online sequence learning and anomaly detection
- **Global Workspace Theory (GWT)** - Multi-modal information integration and attention
- **Active Inference** - Decision-making based on the Free Energy Principle
- **Neuro-Symbolic Reasoning** - Dual-process (System 1/2) deliberative reasoning
- **Meta-Learning** - Adaptive plasticity and few-shot learning
- **Engram Memory** - O(1) conditional memory retrieval (DeepSeek research)

## Architecture

```
Input → Encoders → SNN Core → HTM → Global Workspace → Active Inference → Reasoning → Output
         │                              │                    │               │
     [Vision]                      [Working]            [Decision]      [Symbolic]
     [Text]                        [Memory]             [Heads]         [System 1/2]
     [Audio]                          │                                     │
     [Sensors]                    [Engram]                           [Meta-Learning]
```

### Seven Cognitive Layers

| Layer | Component | Function |
|-------|-----------|----------|
| 1 | Modality Encoders | Process vision, text, audio, sensor inputs |
| 2 | SNN Core | Temporal computation with LIF neurons |
| 3 | HTM | Sequence learning and prediction |
| 4 | Global Workspace | Multi-modal competition and integration |
| 5 | Active Inference | Goal-directed decision making |
| 6 | Symbolic Reasoning | Logical inference and deliberation |
| 7 | Meta-Learning | Adaptive plasticity control |

## Key Features

- **Multi-Modal Processing**: Unified handling of vision, text, audio, and sensor data
- **Biologically Plausible**: Implements neuroscience-inspired mechanisms
- **Configurable**: Centralized configuration with feature flags for each component
- **Extensible**: Modular design allows easy addition of new components
- **Well-Tested**: 100+ unit tests covering all major components

## Quick Start

```python
from brain_ai import create_brain_ai

# Create a multi-modal brain
brain = create_brain_ai(
    modalities=['vision', 'text'],
    output_type='classify',
    num_classes=10,
    device='auto'
)

# Forward pass
output = brain({
    'vision': images,    # (batch, channels, height, width)
    'text': text_ids,    # (batch, seq_len)
})

# Get detailed analysis
result = brain(inputs, return_details=True)
# Returns: output, workspace, confidence, attention, reasoning_trace
```

## Convenience Factories

```python
from brain_ai import (
    create_vision_classifier,
    create_multimodal_system,
    create_control_agent
)

# Vision-only classifier
model = create_vision_classifier(num_classes=10)

# Multi-modal system
model = create_multimodal_system(['vision', 'text', 'audio'])

# Robotic control agent
model = create_control_agent(control_dim=6)
```

## Project Structure

```
brain_ai/
├── core/           # SNN implementation (LIF neurons, surrogate gradients)
├── encoders/       # Multi-modal input processing
├── temporal/       # HTM and sequence learning
├── workspace/      # Global Workspace and working memory
├── decision/       # Active Inference and output heads
├── reasoning/      # Neuro-symbolic and dual-process reasoning
├── meta/           # Meta-learning and neuromodulation
├── memory/         # Engram conditional memory system
├── layers/         # Composite layers (Engram-augmented)
├── config.py       # Centralized configuration
└── system.py       # Main BrainAI orchestrator

scripts/
└── train_phase1.py # MNIST training script

tests/
├── test_snn.py     # SNN unit tests
└── test_engram.py  # Engram integration tests

docs/
├── ENGRAM_INTEGRATION_GUIDE.md
└── plans/          # Implementation roadmaps
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

See `requirements.txt` for full dependencies.

## Installation

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

# Optional: Install HTM (requires building from source)
# See https://github.com/htm-community/htm.core
```

## Cognitive Mapping

The system maps biological cognitive processes to AI implementations:

| Brain Region | AI Component | Purpose |
|--------------|--------------|---------|
| Visual Cortex | VisionEncoder | Feature extraction |
| Wernicke's Area | TextEncoder | Language comprehension |
| Hippocampus | HTM Layer | Episodic memory |
| Prefrontal Cortex | Global Workspace | Executive control |
| Basal Ganglia | Active Inference | Action selection |
| Semantic Memory | Engram | Fast pattern recall |

## Configuration

All configuration through `BrainAIConfig`:

```python
from brain_ai.config import BrainAIConfig

# Full configuration
config = BrainAIConfig(
    use_snn=True,
    use_htm=True,
    use_workspace=True,
    use_symbolic=True,
    use_meta=True,
    use_engram=True,
)

# Minimal for testing
config = BrainAIConfig.minimal()

# Vision-only
config = BrainAIConfig.for_vision_only()
```

## Documentation

- [INSTRUCTIONS.md](INSTRUCTIONS.md) - How to run, train, and perform inference
- [docs/ENGRAM_INTEGRATION_GUIDE.md](docs/ENGRAM_INTEGRATION_GUIDE.md) - Detailed Engram architecture
- [claude.md](claude.md) - Comprehensive architecture guide (4000+ lines)

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test files
python -m pytest tests/test_snn.py -v
python -m pytest tests/test_engram.py -v

# With coverage
python -m pytest tests/ --cov=brain_ai
```

## Current Status

**Implemented & Tested:**
- Core SNN (SNNCore, ConvSNN, LIFNeuron)
- All modality encoders
- HTM layer with LSTM fallback
- Global Workspace Theory
- Active Inference decision system
- Dual-process symbolic reasoning
- Meta-learning components
- MNIST training pipeline

**Under Development:**
- Engram memory integration (Phase 1: encoder-style, Phase 2: layer-style)

## Research References

This project draws from:

- Spiking Neural Networks: LIF neurons, surrogate gradients
- HTM: Numenta's Hierarchical Temporal Memory
- Global Workspace Theory: Baars, Dehaene
- Active Inference: Friston's Free Energy Principle
- Liquid Neural Networks: Neural Circuit Policies (ncps)
- Logic Tensor Networks: Neuro-symbolic AI
- Engram: DeepSeek's conditional memory research

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
