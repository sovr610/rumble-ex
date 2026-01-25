# Training Commands Guide

> Quick reference for training each phase of the brain-inspired AI model with production-level datasets.

---

## Prerequisites

```bash
# Activate virtual environment
source human/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install neuromorphic dataset libraries
pip install tonic spikingjelly snntorch

# Install meta-learning libraries
# NOTE: torchmeta is incompatible with PyTorch 2.x (requires torch < 1.10)
# NOTE: learn2learn has Cython issues on Python 3.12+
# The project uses custom MAML implementation in brain_ai/meta/maml.py
pip install higher

# Install offline RL libraries
pip install minari gymnasium minigrid

# Install liquid neural networks
pip install ncps
```

---

## Phase 1: SNN Core

**Goal**: Train Spiking Neural Network on neuromorphic datasets  
**Target**: 98%+ accuracy on MNIST/N-MNIST

### Basic Training (MNIST)

```bash
python scripts/train_phase1.py
```

### Production Training (N-MNIST with Tonic)

```bash
# Standard N-MNIST training
python scripts/train_phase1.py \
    --epochs 20 \
    --batch-size 128 \
    --lr 1e-3 \
    --num-steps 25 \
    --model conv \
    --data-dir ./data

# High-performance settings
python scripts/train_phase1.py \
    --epochs 50 \
    --batch-size 64 \
    --lr 5e-4 \
    --num-steps 50 \
    --hidden 1024 512 \
    --model conv \
    --device cuda
```

### Available Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 10 | Number of training epochs |
| `--batch-size` | 128 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--beta` | 0.9 | Membrane decay constant |
| `--num-steps` | 25 | Simulation timesteps |
| `--hidden` | 800 400 | Hidden layer sizes |
| `--model` | ff | Model type (ff/conv) |
| `--device` | auto | Device (cuda/cpu/auto) |
| `--save-path` | checkpoints/snn_mnist.pth | Save path |

---

## Phase 2: Event-Driven Vision

**Goal**: Train Vision Encoder on DVS event streams  
**Target**: 75%+ accuracy on DVS-CIFAR10

### Basic Training (Synthetic Events)

```bash
python scripts/train_phase2.py
```

### Production Training (DVS-CIFAR10)

```bash
# Using SpikingJelly loader
python scripts/train_phase2.py \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-3 \
    --num-steps 16 \
    --use-spikingjelly \
    --data-dir ./data

# Extended training
python scripts/train_phase2.py \
    --epochs 100 \
    --batch-size 16 \
    --lr 5e-4 \
    --num-steps 32 \
    --use-spikingjelly \
    --device cuda
```

### Available Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 30 | Number of training epochs |
| `--batch-size` | 32 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--num-steps` | 16 | Simulation timesteps |
| `--use-spikingjelly` | false | Use SpikingJelly dataset loader |
| `--data-dir` | ./data | Data directory |
| `--save-path` | checkpoints/vision_encoder.pth | Save path |

---

## Phase 3: HTM Sequence Learning

**Goal**: Train HTM on sequence prediction and anomaly detection  
**Target**: 90%+ prediction accuracy

### Basic Training (Synthetic Sequences)

```bash
python scripts/train_phase3.py
```

### Production Training (NAB Anomaly Benchmark)

```bash
# Standard HTM training
python scripts/train_phase3.py \
    --sequences 1000 \
    --epochs 10 \
    --seq-length 50 \
    --column-count 512 \
    --cells-per-column 16

# Production-scale HTM
python scripts/train_phase3.py \
    --sequences 5000 \
    --epochs 20 \
    --seq-length 100 \
    --column-count 2048 \
    --cells-per-column 32 \
    --input-dim 256
```

### Available Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--sequences` | 200 | Number of training sequences |
| `--epochs` | 5 | Training epochs |
| `--seq-length` | 30 | Sequence length |
| `--column-count` | 256 | HTM column count |
| `--cells-per-column` | 8 | Cells per column |
| `--input-dim` | 128 | Input dimension |
| `--save-path` | checkpoints/htm_layer.pth | Save path |

---

## Phase 4: Global Workspace

**Goal**: Train multi-modal fusion with Liquid Neural Networks  
**Target**: Cross-modal temporal integration

### Basic Training (Synthetic Multi-Modal)

```bash
python scripts/train_phase4.py
```

### Production Training (CMU-MOSEI Style)

```bash
# Standard workspace training
python scripts/train_phase4.py \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-3 \
    --workspace-dim 512 \
    --num-modalities 3 \
    --seq-length 30

# Extended multi-modal training
python scripts/train_phase4.py \
    --epochs 100 \
    --batch-size 64 \
    --lr 5e-4 \
    --workspace-dim 512 \
    --num-modalities 5 \
    --seq-length 50 \
    --device cuda
```

### Available Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 30 | Number of epochs |
| `--batch-size` | 64 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--workspace-dim` | 256 | Workspace dimension |
| `--num-modalities` | 3 | Number of modalities |
| `--seq-length` | 20 | Sequence length |
| `--save-path` | checkpoints/global_workspace.pth | Save path |

---

## Phase 5: Active Inference

**Goal**: Train goal-directed decision making via Expected Free Energy  
**Target**: Optimal navigation policy

### Basic Training (Grid World)

```bash
python scripts/train_phase5.py
```

### Production Training (Minari/D4RL Style)

```bash
# Standard active inference
python scripts/train_phase5.py \
    --episodes 1000 \
    --max-steps 30 \
    --grid-size 5 \
    --planning-horizon 3 \
    --lr 3e-3

# Harder environment
python scripts/train_phase5.py \
    --episodes 5000 \
    --max-steps 50 \
    --grid-size 8 \
    --planning-horizon 5 \
    --lr 1e-3 \
    --epistemic-weight 0.3 \
    --device cuda
```

### Available Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--episodes` | 1000 | Training episodes |
| `--max-steps` | 30 | Max steps per episode |
| `--grid-size` | 5 | Grid world size |
| `--planning-horizon` | 3 | Planning horizon |
| `--lr` | 3e-3 | Learning rate |
| `--epistemic-weight` | 0.5 | Exploration weight |
| `--save-path` | checkpoints/active_inference.pth | Save path |

---

## Phase 6: Neuro-Symbolic Reasoning

**Goal**: Train System 1/System 2 reasoning with fuzzy logic  
**Target**: 85%+ accuracy with interpretable traces

### Basic Training (Synthetic Logic)

```bash
python scripts/train_phase6.py
```

### Production Training (bAbI/ProofWriter Style)

```bash
# Standard neuro-symbolic training
python scripts/train_phase6.py \
    --epochs 30 \
    --batch-size 64 \
    --lr 1e-3 \
    --reasoning-steps 5 \
    --hidden-dim 256

# Extended reasoning depth
python scripts/train_phase6.py \
    --epochs 50 \
    --batch-size 32 \
    --lr 5e-4 \
    --reasoning-steps 10 \
    --hidden-dim 512 \
    --device cuda
```

### Available Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 30 | Number of epochs |
| `--batch-size` | 64 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--reasoning-steps` | 5 | Max reasoning steps |
| `--hidden-dim` | 256 | Hidden dimension |
| `--save-path` | checkpoints/neuro_symbolic.pth | Save path |

---

## Phase 7: Meta-Learning

**Goal**: Train MAML for few-shot adaptation  
**Target**: 80%+ on 5-way 1-shot classification

### Basic Training (Synthetic Few-Shot)

```bash
python scripts/train_phase7.py
```

### Production Training (Omniglot/mini-ImageNet Style)

```bash
# Standard MAML training
python scripts/train_phase7.py \
    --meta-epochs 100 \
    --tasks-per-batch 4 \
    --n-way 5 \
    --k-shot 1 \
    --inner-lr 0.1 \
    --outer-lr 0.001 \
    --inner-steps 10

# 5-shot training (easier)
python scripts/train_phase7.py \
    --meta-epochs 200 \
    --tasks-per-batch 8 \
    --n-way 5 \
    --k-shot 5 \
    --q-query 15 \
    --inner-lr 0.05 \
    --outer-lr 5e-4 \
    --inner-steps 5 \
    --device cuda

# First-order MAML (faster)
python scripts/train_phase7.py \
    --meta-epochs 100 \
    --tasks-per-batch 8 \
    --first-order \
    --device cuda
```

### Available Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--meta-epochs` | 100 | Meta-training epochs |
| `--tasks-per-batch` | 4 | Tasks per meta-batch |
| `--n-way` | 5 | N-way classification |
| `--k-shot` | 1 | K-shot (support examples) |
| `--q-query` | 15 | Query examples per class |
| `--inner-lr` | 0.1 | Inner loop learning rate |
| `--outer-lr` | 0.001 | Outer loop learning rate |
| `--inner-steps` | 10 | Inner loop gradient steps |
| `--first-order` | false | Use first-order MAML |
| `--save-path` | checkpoints/meta_learning.pth | Save path |

---

## Full Pipeline Training

Train all phases sequentially:

```bash
#!/bin/bash
# full_train.sh - Train all phases

set -e  # Exit on error

echo "=== Phase 1: SNN Core ==="
python scripts/train_phase1.py --epochs 20 --model conv

echo "=== Phase 2: Event-Driven Vision ==="
python scripts/train_phase2.py --epochs 50

echo "=== Phase 3: HTM Sequence Learning ==="
python scripts/train_phase3.py --sequences 1000 --epochs 10

echo "=== Phase 4: Global Workspace ==="
python scripts/train_phase4.py --epochs 50

echo "=== Phase 5: Active Inference ==="
python scripts/train_phase5.py --episodes 2000

echo "=== Phase 6: Neuro-Symbolic Reasoning ==="
python scripts/train_phase6.py --epochs 30

echo "=== Phase 7: Meta-Learning ==="
python scripts/train_phase7.py --meta-epochs 100

echo "=== All phases complete! ==="
```

---

## GPU Memory Tips

If running out of GPU memory:

```bash
# Reduce batch size
--batch-size 16

# Use gradient checkpointing (if supported)
--gradient-checkpointing

# Use mixed precision
--fp16

# Reduce model size
--hidden 512 256
--workspace-dim 128
--column-count 128
```

---

## Monitoring Training

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# TensorBoard (if logging enabled)
tensorboard --logdir runs/

# Check checkpoint sizes
ls -lh checkpoints/
```

---

## Expected Checkpoints

After training, you should have:

| Phase | Checkpoint | Expected Size |
|-------|------------|---------------|
| 1 | `snn_mnist.pth` | ~5-20 MB |
| 2 | `vision_encoder.pth` | ~10-50 MB |
| 3 | `htm_layer.pth` | ~1-10 MB |
| 4 | `global_workspace.pth` | ~5-30 MB |
| 5 | `active_inference.pth` | ~5-20 MB |
| 6 | `neuro_symbolic.pth` | ~5-30 MB |
| 7 | `meta_learning.pth` | ~10-50 MB |
