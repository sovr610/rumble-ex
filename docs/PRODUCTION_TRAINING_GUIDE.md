# Production Training Guide: 7B Brain-Inspired AI

This guide covers the complete training pipeline for the 7B parameter Brain-Inspired AI system.

## Overview

### Parameter Distribution (~7B total)

| Layer | Parameters | Description |
|-------|-----------|-------------|
| Vision Encoder | ~300M | ViT-Large scale (24L, 1024d, 16h) |
| Text Encoder | ~2B | Scaled transformer (32L, 4096d, 32h) |
| Audio Encoder | ~300M | Wav2Vec2-Large scale |
| SNN Core | ~500M | 4-layer spiking network |
| HTM Layer | ~200M | Temporal memory (16K columns) |
| Global Workspace | ~1.5B | Multi-modal integration |
| Decision Heads | ~500M | Active inference + output |
| Symbolic Reasoning | ~800M | Neuro-symbolic System 2 |
| Meta-Learning | ~100M | Neuromodulation + MAML |
| Engram Memory | ~800M | Conditional memory |
| **Total** | **~7B** | |

### Training Data Requirements

**Chinchilla Optimal (minimum):** 140B tokens (20 tokens per parameter)
**Recommended (modern practice):** 1T-4T tokens

## Phase-by-Phase Training

### Phase 1: SNN Core (Vision)

**Datasets:**
1. **Primary:** ImageNet-21K (14M images, 21K classes)
   - Download: `huggingface-cli download imagenet-21k`
   - Size: ~1.2TB
   
2. **Secondary:** LAION-400M (filtered subset)
   - Use DataComp filtering pipeline
   - Target: 50M high-quality images
   
3. **Fine-tuning:** Domain-specific datasets
   - COCO (330K images, detection/segmentation)
   - Visual Genome (108K images, scene graphs)
   - OpenImages V7 (9M images)

**Training Configuration:**
```python
# config.py already configured for production
config = BrainAIConfig.production_7b()

# Or for limited resources:
config = BrainAIConfig.production_3b()
config = BrainAIConfig.production_1b()
```

**Training Command:**
```bash
# Single GPU (testing)
python scripts/train_phase1.py --config production

# Multi-GPU (production)
torchrun --nproc_per_node=8 scripts/train_phase1.py \
    --config production \
    --dataset imagenet21k \
    --epochs 90 \
    --batch-size 64 \
    --gradient-accumulation 4
```

### Phase 2: Event-Driven Processing

**Datasets:**
1. **DVS128-Gesture:** 1,342 samples, 11 gestures (SOTA: 98.78%)
2. **DailyDVS-200:** 22K samples, 200 actions (ECCV 2024)
3. **ASL-DVS:** 100K samples, 24 letters
4. **DSEC:** 150GB stereo driving events (for autonomous systems)

**Recommended Training:**
```bash
# Start with DVS-Gesture for validation
python scripts/train_phase2.py --dataset dvs_gesture --epochs 100

# Scale to DailyDVS-200 for production
python scripts/train_phase2.py --dataset dailydvs200 --epochs 200
```

### Phase 3: HTM Temporal Memory

**Datasets:**
1. **NAB (Numenta Anomaly Benchmark):** 58 time series
   - HTM native benchmark
   - Target: 70.5+ standard score
   
2. **TSB-AD (NeurIPS 2024):** 1,070 time series
   - Gold standard for anomaly detection
   - Metric: VUS-PR
   
3. **UCR Archive:** 128 univariate datasets

**Training Command:**
```bash
python scripts/train_phase3.py \
    --dataset nab \
    --columns 16384 \
    --cells-per-column 64 \
    --epochs 1000
```

### Phase 4: Global Workspace

**Datasets:**
1. **CMU-MOSEI:** 23K video clips, sentiment/emotion
2. **CMU-MOSI:** 2,199 clips, sentiment analysis
3. **How2:** 80K instructional videos
4. **MultiBench:** Unified multi-modal benchmark

**Training Configuration:**
```bash
# Multi-modal fusion training
python scripts/train_phase4.py \
    --dataset mosei \
    --modalities vision,text,audio \
    --workspace-dim 4096 \
    --epochs 100
```

### Phase 5: Active Inference

**Datasets:**
1. **Minari (D4RL successor):** Official offline RL benchmark
   - AntMaze, Hopper, Walker2d, HalfCheetah
   
2. **MiniGrid:** Procedural grid-world environments
3. **pymdp environments:** Active inference benchmarks

**Training Command:**
```bash
# Offline RL training
python scripts/train_phase5.py \
    --dataset minari \
    --env antmaze-large-diverse-v1 \
    --planning-horizon 8 \
    --epochs 500
```

### Phase 6: Neuro-Symbolic Reasoning

**Datasets:**
1. **bAbI:** 20 QA reasoning tasks
2. **ProofWriter:** Logical deduction (depths 0-5)
3. **CLEVR:** Visual reasoning
4. **FOLIO:** First-order logic inference
5. **ARC (AI2 Reasoning Challenge):** Common sense

**Training Command:**
```bash
# Start with bAbI for validation
python scripts/train_phase6.py --dataset babi --task all

# Scale to ProofWriter for depth
python scripts/train_phase6.py --dataset proofwriter --depth 5
```

### Phase 7: Meta-Learning

**Datasets:**
1. **Omniglot:** 1,623 characters, few-shot learning
2. **mini-ImageNet:** 100 classes, 600 images each
3. **tiered-ImageNet:** 608 classes, hierarchical
4. **Meta-Dataset:** Large-scale benchmark

**Training Command:**
```bash
# MAML training
python scripts/train_phase7.py \
    --dataset mini_imagenet \
    --ways 5 \
    --shots 1 \
    --inner-steps 10 \
    --outer-lr 0.0001
```

## Hardware Requirements

### Minimum (1B config)
- 1x NVIDIA A100 40GB or equivalent
- 256GB RAM
- 2TB NVMe storage

### Recommended (3B config)
- 4x NVIDIA A100 80GB
- 512GB RAM
- 10TB NVMe storage

### Full Scale (7B config)
- 8x NVIDIA H100 80GB (or 32x A100 40GB)
- 1TB RAM
- 50TB storage for datasets

## Training Tips

### Mixed Precision
Always use bfloat16 for modern GPUs:
```python
config.training.use_amp = True
config.training.amp_dtype = "bfloat16"
```

### Gradient Checkpointing
Enable for memory efficiency:
```python
model.gradient_checkpointing_enable()
```

### FSDP for Distributed Training
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
model = FSDP(model, ...)
```

### Learning Rate Schedule
Use warmup + cosine decay:
- Warmup: 2000 steps
- Peak LR: 3e-4
- Min LR: 3e-5 (10% of peak)

## Evaluation Metrics

### Per-Phase Targets

| Phase | Metric | Target |
|-------|--------|--------|
| 1. SNN | ImageNet Accuracy | 85%+ |
| 2. Event | DVS-Gesture Acc | 98%+ |
| 3. HTM | NAB Score | 70.5+ |
| 4. Workspace | MOSEI Acc-7 | 50%+ |
| 5. Active Inf | AntMaze Score | 90%+ |
| 6. Reasoning | bAbI Mean Acc | 95%+ |
| 7. Meta | 5-way 1-shot | 65%+ |

### End-to-End Benchmarks
- MMLU (general knowledge)
- ARC (reasoning)
- HellaSwag (commonsense)
- GSM8K (math)
- HumanEval (code)

## Dataset Download Scripts

```bash
# Install required libraries
pip install datasets huggingface_hub tonic minari gymnasium

# Download core datasets
python -c "from datasets import load_dataset; load_dataset('imagenet-1k')"
python -c "import minari; minari.download_dataset('D4RL/antmaze/large-diverse-v1')"

# Event camera datasets (via Tonic)
python -c "import tonic; tonic.datasets.DVSGesture('./data', train=True)"

# CMU multimodal datasets
pip install mmsdk
python -c "from mmsdk import mmdatasdk; mmdatasdk.cmu_mosei.highlevel.download('./data')"
```

## Checkpointing Strategy

Save checkpoints every 1000 steps:
```python
if step % config.training.checkpoint_interval == 0:
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': config,
    }, f'checkpoints/brain_ai_step_{step}.pth')
```

## Estimated Training Time

| Config | Hardware | Time |
|--------|----------|------|
| 1B | 1x A100 | ~1 week |
| 3B | 4x A100 | ~2 weeks |
| 7B | 8x H100 | ~3 weeks |

These estimates assume 1T tokens of training data.
