# Brain-Inspired Multi-Modal AI System Design

**Date:** 2026-01-17
**Status:** Approved
**Author:** Human + Claude

---

## Overview

A general-purpose brain-inspired AI system supporting multiple input modalities (vision, text, audio, sensors) and output capabilities (classification, text generation, continuous control) with adaptive learning modes.

### Requirements

| Requirement | Choice |
|-------------|--------|
| Deployment | Cloud/Server (GPU clusters) |
| Input modalities | Vision, Text, Audio, Sensor streams |
| Output capabilities | Classification, Text generation, Control signals |
| Learning modes | Static inference, Online adaptation, Few-shot meta-learning |
| Development approach | Incremental by layer |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MULTI-MODAL BRAIN-INSPIRED AI SYSTEM                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUTS                          PROCESSING                      OUTPUTS   │
│                                                                             │
│  ┌─────────┐                                                               │
│  │ Vision  │──┐                  ┌─────────────────┐                       │
│  └─────────┘  │                  │ GLOBAL WORKSPACE│         ┌──────────┐  │
│  ┌─────────┐  │  ┌──────────┐    │   (Attention +  │────────▶│ Classify │  │
│  │  Text   │──┼─▶│ SNN Core │───▶│ Working Memory) │         └──────────┘  │
│  └─────────┘  │  └──────────┘    └────────┬────────┘         ┌──────────┐  │
│  ┌─────────┐  │        │                  │          ┌──────▶│ Generate │  │
│  │  Audio  │──┤        ▼                  ▼          │       └──────────┘  │
│  └─────────┘  │  ┌──────────┐    ┌─────────────────┐ │       ┌──────────┐  │
│  ┌─────────┐  │  │   HTM    │    │    DECISION     │─┼──────▶│ Control  │  │
│  │ Sensors │──┘  │(Temporal)│    │ (Active Infrnc) │ │       └──────────┘  │
│  └─────────┘     └──────────┘    └────────┬────────┘ │                     │
│                        │                  │          │                     │
│                        ▼                  ▼          │                     │
│                  ┌──────────┐    ┌─────────────────┐ │                     │
│                  │ SYMBOLIC │◀──▶│  META-LEARNING  │─┘                     │
│                  │(Reasoner)│    │ (Plasticity)    │                       │
│                  └──────────┘    └─────────────────┘                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: SNN Core

**Purpose:** Spike-based feature extraction foundation for all modalities.

**Framework:** snnTorch (PyTorch-based)

**Architecture:**
```python
SNNCore:
  - input_projection: Linear(modality_dim → 512)
  - snn_block_1: Linear(512) → LIF(β=0.9) → Dropout(0.2)
  - snn_block_2: Linear(256) → LIF(β=0.9) → Dropout(0.2)
  - snn_block_3: Linear(128) → LIF(β=0.9)
  - output: spike_sum over 25 timesteps → (batch, 128)
```

**Key Parameters:**
- Neuron model: Leaky Integrate-and-Fire
- Beta (decay): 0.9
- Surrogate gradient: Arctangent (alpha=2.0)
- Timesteps: 25
- Encoding: Rate coding

**Validation:** 98% accuracy on MNIST

---

## Phase 2: Modality Encoders

**Purpose:** Specialized preprocessing for each input type, outputting common 512-dim representation.

### Vision Encoder
```python
ConvSNN:
  - Conv2d(3→32, k=3) → LIF → MaxPool2d(2)
  - Conv2d(32→64, k=3) → LIF → MaxPool2d(2)
  - Conv2d(64→128, k=3) → LIF → AdaptiveAvgPool2d(4,4)
  - Flatten → Linear(128*4*4 → 512)
```
Validation: 90%+ CIFAR-10

### Text Encoder
```python
TextEncoder:
  - Embedding(vocab_size, 256)
  - TransformerEncoder(4 layers, 4 heads)
  - Linear(256 → 512)
```

### Audio Encoder
```python
AudioEncoder:
  - MelSpectrogram(n_mels=80)
  - Conv1d(80→128) → LIF
  - Conv1d(128→256) → LIF
  - AdaptiveAvgPool1d → Linear(256 → 512)
```
Validation: 90%+ Speech Commands

### Sensor Encoder
```python
SensorEncoder:
  - LiquidTimeConstant(input_dim, 256)  # ncps.LTC
  - Linear(256 → 512)
```

---

## Phase 3: HTM Temporal Layer

**Purpose:** Online sequence learning, prediction, and anomaly detection.

**Configuration:**
| Parameter | Value |
|-----------|-------|
| Column count | 2048 |
| Cells per column | 32 |
| Sparsity | 2% (~40 active) |
| Permanence increment | 0.1 |
| Activation threshold | 13 |

**Outputs:**
- `active_cells`: Current representation
- `predicted_cells`: Next-step prediction
- `anomaly`: Prediction failure score [0,1]

**Validation:** 90% sequence prediction, anomaly F1 > 0.85

**Note:** Requires htm.core library. Fallback: LSTM-based sequence predictor.

---

## Phase 4: Global Workspace

**Purpose:** Multi-modal integration via attention-based competition and working memory.

**Architecture:**
```python
GlobalWorkspace:
  - input_projections: {modality: Linear(dim → 512)}
  - competition: MultiheadAttention(512, 8 heads)
  - capacity_gate: Top-K selection (k=7, Miller's Law)
  - working_memory: CfC(512, 512)  # Liquid NN
  - broadcast_projections: {modality: Linear(512 → dim)}
```

**Mechanism:**
1. Project all modalities to common 512-dim space
2. Attention-based competition for workspace access
3. Top-K gating enforces capacity limit
4. Liquid NN maintains temporal working memory state
5. Broadcast winning information back to all specialists

**Validation:** Attention weights shift based on input salience

---

## Phase 5: Decision & Action System

**Purpose:** Action selection using Active Inference (minimize expected free energy).

**Components:**
- State inference: MLP encoder P(state|obs)
- Generative model: P(obs|state), P(state'|state,action)
- Preferences (C): Learnable goal specification
- Policy evaluation: EFE = -pragmatic - epistemic
- Action selection: π(a) ∝ softmax(-EFE)

**Output Heads:**
```python
DecisionHeads:
  - classifier: Linear(512 → num_classes) + Softmax
  - text_decoder: TransformerDecoder(512, vocab, 2 layers)
  - control: Linear(512 → μ), Linear(512 → log_σ) → Gaussian
```

**Validation:** T-maze navigation with exploration/exploitation balance

---

## Phase 6: Neuro-Symbolic Reasoning

**Purpose:** Verified multi-step reasoning when confidence is low.

**Dual-Process Integration:**
| System | Trigger | Behavior |
|--------|---------|----------|
| System 1 | confidence > 0.7 | Fast pass-through |
| System 2 | confidence ≤ 0.7 | Iterative refinement (up to 5 steps) |

**Symbolic Operations (Fuzzy Logic):**
```python
AND(a, b)     = a * b
OR(a, b)      = a + b - a*b
NOT(a)        = 1 - a
IMPLIES(a, b) = 1 - a + a*b
FORALL(x)     = product(x)
EXISTS(x)     = 1 - product(1 - x)
```

**Validation:** Transitive reasoning puzzles, System 2 engages appropriately

---

## Phase 7: Meta-Learning & Plasticity

**Purpose:** Control when and how learning occurs across the system.

**Neuromodulatory Gate:**
```python
NeuromodulatoryGate:
  - inputs: anomaly_score, confidence, prediction_error
  - 4 modulators (DA, ACh, NE, 5-HT analogs)
  - output: lr_multiplier ∈ [0, 2]
```

**Learning Modes:**
| Mode | Mechanism | Trigger |
|------|-----------|---------|
| Static | lr_mult → 0 | High confidence, familiar input |
| Online | Eligibility traces | Anomaly detected, novel input |
| Few-shot | MAML (5 inner steps) | Explicit new task |

**Eligibility Traces:**
- Decay: 0.95
- Update: trace = decay * trace + outer(post, pre)
- Learning: Δw = lr * reward_signal * trace

**Validation:** Few-shot adaptation, no catastrophic forgetting

---

## Project Structure

```
human-brain/
├── brain_ai/
│   ├── __init__.py
│   ├── core/
│   │   ├── snn.py
│   │   ├── neurons.py
│   │   └── encoding.py
│   ├── encoders/
│   │   ├── vision.py
│   │   ├── text.py
│   │   ├── audio.py
│   │   └── sensors.py
│   ├── temporal/
│   │   ├── htm.py
│   │   └── sequence.py
│   ├── workspace/
│   │   ├── global_workspace.py
│   │   └── working_memory.py
│   ├── decision/
│   │   ├── active_inference.py
│   │   └── output_heads.py
│   ├── reasoning/
│   │   ├── symbolic.py
│   │   └── system2.py
│   ├── meta/
│   │   ├── neuromodulation.py
│   │   ├── maml.py
│   │   └── eligibility.py
│   ├── system.py
│   └── config.py
├── tests/
├── scripts/
├── docs/plans/
├── requirements.txt
└── claude.md
```

---

## Implementation Order

| Phase | Deliverable | Validation Gate | Dependencies |
|-------|-------------|-----------------|--------------|
| 1 | SNN Core | 98% MNIST | torch, snntorch |
| 2 | Encoders | 90%+ benchmarks | Phase 1 |
| 3 | HTM | 90% seq pred, F1>0.85 | Phase 1, htm.core |
| 4 | Workspace | Attention behavior | Phases 1-3, ncps |
| 5 | Decision | T-maze, valid outputs | Phases 1-4, pymdp |
| 6 | Reasoning | Logic puzzles | Phases 1-5 |
| 7 | Meta | Few-shot, no forgetting | Phases 1-6 |
| **Full** | Integration | Multi-modal tasks | All phases |

---

## Dependencies

```
torch>=2.0.0
snntorch>=0.7.0
spikingjelly>=0.0.0.0.14
ncps>=0.0.7
inferactively-pymdp>=0.0.8
numpy>=1.24.0
scipy>=1.10.0
einops>=0.6.0
tensorboard>=2.12.0
```

Optional:
- `htm.core` (build from source for Phase 3)
- `torchaudio` (for audio encoder)
- `transformers` (for text encoder pretrained weights)
