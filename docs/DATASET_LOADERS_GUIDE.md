# Dataset Loaders - Production Implementation Guide

This document provides a comprehensive guide to the production-level dataset loaders for the 7-phase brain-inspired AI training pipeline.

## Quick Start

```python
from brain_ai.datasets import (
    # Phase 1: SNN Core
    get_snn_datasets, NMNISTDataset,
    
    # Phase 2: Event-Driven
    get_event_datasets, DVSGestureDataset,
    
    # Phase 3: HTM
    get_htm_datasets, NABDataset, TSBADDataset,
    
    # Phase 4: Multi-Modal
    get_multimodal_datasets, CMUMOSEIDataset,
    
    # Phase 5: Active Inference
    get_active_inference_datasets, MinariDataset,
    
    # Phase 6: Neuro-Symbolic
    get_neurosymbolic_datasets, BABIDataset,
    
    # Phase 7: Meta-Learning
    get_metalearning_datasets, MetaDatasetLoader,
)
```

## Installation

```bash
# Core dependencies
pip install torch torchvision torchaudio

# Phase 1-2: Neuromorphic
pip install tonic snntorch spikingjelly

# Phase 3: Time Series
pip install pyod scikit-learn

# Phase 4: Multi-Modal
pip install h5py pandas
pip install git+https://github.com/A2Zadeh/CMU-MultimodalSDK.git

# Phase 5: Active Inference / Offline RL
pip install minari[all] gymnasium minigrid inferactively-pymdp

# Phase 6: Neuro-Symbolic
pip install datasets Pillow

# Phase 7: Meta-Learning
pip install learn2learn avalanche-lib
```

---

## Phase 1: SNN Core Datasets

Neuromorphic vision and audio datasets for spiking neural network training.

### Datasets

| Dataset | Samples | Classes | Format | License |
|---------|---------|---------|--------|---------|
| N-MNIST | 70,000 | 10 | DVS events | CC BY-SA 4.0 |
| DVS-CIFAR10 | 10,000 | 10 | DVS events | CC BY 4.0 |
| SHD | 10,420 | 20 | Spike trains | CC BY 4.0 |
| SSC | 100,837 | 35 | Spike trains | CC BY 4.0 |

### Usage

```python
from brain_ai.datasets import get_snn_datasets, NMNISTDataset

# Quick setup with convenience function
data = get_snn_datasets(
    dataset_name='nmnist',
    batch_size=128,
    n_time_bins=10
)
train_loader = data['train_loader']

# Or use dataset class directly
dataset = NMNISTDataset(
    root='./data',
    train=True,
    download=True,
    n_time_bins=10
)

# Access NeuroBench metrics for evaluation
from brain_ai.datasets import get_neurobench_metrics
metrics = get_neurobench_metrics(model, test_loader, device='cuda')
print(f"Accuracy: {metrics['accuracy']:.2f}%")
print(f"Synaptic Ops/Inference: {metrics['synaptic_ops']}")
```

---

## Phase 2: Event-Driven Datasets

Event-based vision datasets for temporal processing.

### Datasets

| Dataset | Samples | Classes | Sensor | License |
|---------|---------|---------|--------|---------|
| DVS-Gesture | 1,342 | 11 | DVS128 | CC BY 4.0 |
| DailyDVS-200 | 22,000+ | 200 | DAVIS346 | Research |
| ASL-DVS | 100,800 | 24 | DVS128 | CC BY 4.0 |
| DSEC | 41 seq | - | Stereo | CC BY 4.0 |

### Usage

```python
from brain_ai.datasets import get_event_datasets, DVSGestureDataset

# Quick setup
data = get_event_datasets('dvs-gesture', batch_size=32)

# Direct usage with augmentation
from brain_ai.datasets import EventAugmentation, DVSGestureDataset

augment = EventAugmentation(
    noise_fraction=0.1,
    time_jitter=0.01,
    spatial_jitter=1
)

dataset = DVSGestureDataset(
    root='./data',
    train=True,
    transform=augment
)
```

---

## Phase 3: HTM Temporal Memory Datasets

Time series and anomaly detection for HTM training.

### Datasets

| Dataset | Series | Domain | Metric | License |
|---------|--------|--------|--------|---------|
| NAB | 58 | Mixed | NAB Score | AGPL-3.0 |
| TSB-AD | 1,070 | 40 sources | VUS-PR | MIT |
| UCR | 128 | Various | Accuracy | BSD |

### Usage

```python
from brain_ai.datasets import (
    get_htm_datasets, 
    NABDataset, 
    TSBADDataset,
    NABScorer,
    compute_vus_pr
)

# Load NAB for HTM
nab = NABDataset(root='./data', category='realTraffic')
for series_name, values, labels in nab:
    # Train HTM on each series
    pass

# Evaluate with NAB scoring
scorer = NABScorer(profile='standard')
score = scorer.score(predictions, labels, windows)

# TSB-AD with VUS-PR metric (NeurIPS 2024 recommendation)
tsb = TSBADDataset(root='./data', subset='synthetic')
vus_score = compute_vus_pr(anomaly_scores, labels, max_buffer=100)
```

---

## Phase 4: Multi-Modal Fusion Datasets

Multi-modal datasets for Global Workspace training.

### Datasets

| Dataset | Samples | Modalities | Task | License |
|---------|---------|------------|------|---------|
| CMU-MOSEI | 23,453 | Text+Audio+Video | Sentiment | CC BY 4.0 |
| CMU-MOSI | 2,199 | Text+Audio+Video | Sentiment | CC BY 4.0 |
| AV-MNIST | 70,000 | Image+Audio | Classification | - |
| How2 | 80,000 | Video+Audio+Text | Summarization | CC BY 4.0 |

### Usage

```python
from brain_ai.datasets import (
    get_multimodal_datasets,
    CMUMOSEIDataset,
    compute_multimodal_metrics
)

# Load CMU-MOSEI
data = get_multimodal_datasets('mosei', batch_size=32)
for batch in data['train_loader']:
    glove = batch['glove']      # Text embeddings
    covarep = batch['covarep']  # Audio features
    facet = batch['facet']      # Video features
    labels = batch['labels']    # Sentiment labels

# Evaluate multimodal model
metrics = compute_multimodal_metrics(predictions, labels, task='sentiment')
print(f"MAE: {metrics['mae']:.3f}")
print(f"Correlation: {metrics['corr']:.3f}")
print(f"Binary Acc: {metrics['binary_acc']:.1%}")
```

---

## Phase 5: Active Inference Datasets

Offline RL and decision-making datasets.

### Datasets

| Dataset | Episodes | Domain | Library | License |
|---------|----------|--------|---------|---------|
| Minari (D4RL) | Varies | MuJoCo/Maze | minari | MIT |
| MiniGrid | Generated | Grid World | minigrid | Apache |
| pymdp Tasks | Generated | POMDP | pymdp | MIT |

### Usage

```python
from brain_ai.datasets import (
    get_active_inference_datasets,
    MinariDataset,
    MiniGridDataset,
    ActiveInferenceDataset,
    compute_expected_free_energy
)

# Minari offline RL (replaces D4RL)
dataset = MinariDataset(
    env_name='antmaze-large-diverse-v1',
    download=True
)
for obs, action, reward, next_obs, done in dataset:
    # Offline RL training
    pass

# Active Inference with pymdp
ai_dataset = ActiveInferenceDataset(
    task='gridworld',
    n_episodes=500,
    collect_beliefs=True
)
for episode in ai_dataset:
    observations = episode['observations']
    actions = episode['actions']
    beliefs = episode['beliefs']  # For EFE training

# Compute Expected Free Energy for action selection
efe = compute_expected_free_energy(
    qs=belief_state,
    A=observation_model,
    B=transition_model,
    C=preferences,
    action=0
)
```

---

## Phase 6: Neuro-Symbolic Reasoning Datasets

Logical reasoning and inference datasets.

### Datasets

| Dataset | Samples | Task | Depth | License |
|---------|---------|------|-------|---------|
| bAbI | 10,000/task | QA | 1-5 | BSD |
| ProofWriter | Varies | Proof Gen | 0-5 | CC BY 4.0 |
| CLEVR | 865,000 | Visual QA | - | CC BY 4.0 |
| CLUTRR | Varies | Kinship | 2-10 | MIT |
| FOLIO | 1,435 | FOL | - | CC BY 4.0 |
| RuleTaker | Varies | Rules | 0-5 | Apache |

### Usage

```python
from brain_ai.datasets import (
    get_neurosymbolic_datasets,
    BABIDataset,
    ProofWriterDataset,
    FOLIODataset,
    compute_reasoning_accuracy
)

# bAbI QA tasks
data = get_neurosymbolic_datasets('babi', task=1, batch_size=32)
for stories, questions, answers, supporting in data['train_loader']:
    # Train reasoning model
    pass

# ProofWriter for proof generation
proofwriter = ProofWriterDataset(depth=3, split='train')
for context, question, answer, proof in proofwriter:
    # Train proof generator
    pass

# FOLIO for first-order logic
folio = FOLIODataset(split='train')
for premises, hypothesis, label, fol_premises in folio:
    # Train FOL reasoner
    pass

# Evaluate reasoning accuracy
metrics = compute_reasoning_accuracy(predictions, targets)
print(f"Accuracy: {metrics['accuracy']:.1%}")
```

---

## Phase 7: Meta-Learning & Continual Learning

Few-shot learning and plasticity evaluation.

### Datasets

| Dataset | Classes | Images/Class | Task | License |
|---------|---------|--------------|------|---------|
| Omniglot | 1,623 | 20 | Few-shot | MIT |
| mini-ImageNet | 100 | 600 | Few-shot | - |
| tiered-ImageNet | 608 | Varies | Few-shot | - |
| FC100 | 100 | 600 | Few-shot | - |
| CIFAR-FS | 100 | 600 | Few-shot | - |
| Split-CIFAR100 | 100 | 500 | Continual | MIT |

### Usage

```python
from brain_ai.datasets import (
    get_metalearning_datasets,
    MetaDatasetLoader,
    ContinualLearningBenchmark,
    maml_inner_loop,
    compute_few_shot_accuracy
)

# Few-shot with learn2learn
loader = MetaDatasetLoader(
    dataset_name='mini-imagenet',
    ways=5,
    shots=1,
    queries=15,
    download=True
)

for support, query in loader.get_task_iterator(1000, split='train'):
    support_images, support_labels = support
    query_images, query_labels = query
    
    # MAML inner loop adaptation
    adapted_model = maml_inner_loop(
        model, support,
        inner_lr=0.01,
        inner_steps=5
    )
    
    # Evaluate on query set
    accuracy = compute_few_shot_accuracy(adapted_model, query)

# Continual learning with Avalanche
benchmark = ContinualLearningBenchmark(
    benchmark_name='split-cifar100',
    n_experiences=5
)

for experience in benchmark.train_stream:
    # Train on experience
    train_on_experience(model, experience)
    
# Evaluate Loss of Plasticity (Nature 2024)
plasticity_metrics = benchmark.evaluate_plasticity(
    model, criterion, device='cuda'
)
print(f"Maintained Plasticity: {plasticity_metrics['maintained_plasticity']:.1%}")
```

---

## Key 2024-2025 Updates

### NeuroBench (Nature Communications 2025)
Gold standard for neuromorphic evaluation:
- Standardized metrics: accuracy, latency, energy, synaptic ops
- Hardware-agnostic benchmarking
- Reproducible comparisons

### TSB-AD (NeurIPS 2024)
New anomaly detection benchmark:
- 1,070 high-quality time series from 40 sources
- VUS-PR as recommended metric
- 40 algorithms benchmarked

### DailyDVS-200 (ECCV 2024)
Largest event-based action dataset:
- 22,000+ sequences across 200 classes
- Daily life actions captured with DAVIS346
- Challenges: fine-grained recognition, viewpoint variation

### Loss of Plasticity Benchmark (Nature 2024)
Continual learning evaluation:
- Measures learning capability maintenance
- Split-CIFAR100 as primary benchmark
- Plasticity score: ratio of learning speed

### Minari (Farama Foundation)
Official D4RL successor:
- D4RL namespace preserved for compatibility
- `minari.load_dataset("D4RL/antmaze/large-diverse-v1")`
- Active maintenance and new datasets

---

## Dataset Registry

All datasets are registered in `DatasetRegistry` with metadata:

```python
from brain_ai.datasets import DatasetRegistry

# List all registered datasets
for name, info in DatasetRegistry.DATASETS.items():
    print(f"{name}: {info['description']}")
    print(f"  License: {info['license']}")
    print(f"  URL: {info['url']}")
```

---

## Data Directory Structure

```
data/
├── neuromorphic/
│   ├── nmnist/
│   ├── dvs_cifar10/
│   ├── shd/
│   └── ssc/
├── event/
│   ├── dvs_gesture/
│   ├── daily_dvs_200/
│   └── asl_dvs/
├── htm/
│   ├── nab/
│   ├── tsb-ad/
│   └── ucr/
├── multimodal/
│   ├── cmu-mosei/
│   └── cmu-mosi/
├── rl/
│   └── minari/  # Minari manages this automatically
├── reasoning/
│   ├── babi/
│   ├── clevr/
│   └── proofwriter/
└── metalearning/
    ├── omniglot/
    ├── mini-imagenet/
    └── tiered-imagenet/
```

---

## Citation

If you use these dataset loaders, please cite the original dataset papers:

```bibtex
@article{neurobench2025,
  title={NeuroBench: A Framework for Benchmarking Neuromorphic Computing},
  journal={Nature Communications},
  year={2025}
}

@inproceedings{tsb-ad2024,
  title={TSB-AD: A Comprehensive Time Series Anomaly Detection Benchmark},
  booktitle={NeurIPS 2024},
  year={2024}
}

@article{loss-of-plasticity2024,
  title={Loss of Plasticity in Deep Continual Learning},
  journal={Nature},
  year={2024}
}
```
