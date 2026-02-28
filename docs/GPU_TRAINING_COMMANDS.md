# GPU-Optimized Training Commands

**Date**: January 28, 2026  
**Project**: Brain-Inspired AI System

This guide provides optimized training commands for each phase, tailored to specific GPU hardware.

---

## GPU Specifications Reference

| GPU | VRAM | Memory Bandwidth | FP32 TFLOPS | Tensor TFLOPS | Architecture | Best For |
|-----|------|------------------|-------------|---------------|--------------|----------|
| **RTX PRO 6000 Blackwell WK** | 96 GB GDDR7 | ~2,000 GB/s | 125 | 4000 TOPS (AI) | Blackwell | Production 7B training |
| **NVIDIA H100 (80GB)** | 80 GB HBM3 | 3,350 GB/s | 67 | 1,979 | Hopper | Data center production |
| **NVIDIA A100 (80GB)** | 80 GB HBM2e | 1,935 GB/s | 19.5 | 312 | Ampere | Data center training |
| **NVIDIA A100 (40GB)** | 40 GB HBM2 | 1,555 GB/s | 19.5 | 312 | Ampere | Data center training |
| **RTX 6000 Ada** | 48 GB GDDR6 | 960 GB/s | 91.1 | 1,457 | Ada Lovelace | Professional workstation |
| **RTX 5090** | 32 GB GDDR7 | 1,792 GB/s | ~100 | ~1,600 | Blackwell | Consumer high-end |
| **RTX 4090** | 24 GB GDDR6X | 1,008 GB/s | 82.6 | 1,321 | Ada Lovelace | Consumer training |
| **RTX 3090** | 24 GB GDDR6X | 936 GB/s | 35.6 | 568 | Ampere | Budget training |

---

## Quick Reference: Batch Size by GPU

| GPU | Phase 1-3 | Phase 4-5 | Phase 6-7 | Gradient Accum |
|-----|-----------|-----------|-----------|----------------|
| RTX PRO 6000 Blackwell (96GB) | 256 | 128 | 64 | 1-2 |
| H100 (80GB) | 192 | 96 | 48 | 1-2 |
| A100 (80GB) | 192 | 96 | 48 | 2 |
| A100 (40GB) | 96 | 48 | 24 | 4 |
| RTX 6000 Ada (48GB) | 128 | 64 | 32 | 2-4 |
| RTX 5090 (32GB) | 96 | 48 | 24 | 4 |
| RTX 4090 (24GB) | 64 | 32 | 16 | 4-8 |
| RTX 3090 (24GB) | 48 | 24 | 12 | 8 |

---

## Phase 1: SNN Core Training

### RTX PRO 6000 Blackwell Workstation (96GB)

```bash
# Development mode (MNIST validation)
python scripts/train_phase1.py \
    --mode dev \
    --dataset mnist \
    --batch-size 512 \
    --epochs 20 \
    --lr 0.001 \
    --model conv \
    --use-amp \
    --compile

# Production mode (ImageNet-21K)
python scripts/train_phase1.py \
    --mode production \
    --dataset imagenet21k \
    --batch-size 256 \
    --epochs 90 \
    --lr 0.0003 \
    --gradient-accumulation 1 \
    --use-amp \
    --compile \
    --save-path checkpoints/phase1_rtxpro6000.pth

# Multi-GPU (if 2+ GPUs available)
torchrun --nproc_per_node=2 scripts/train_phase1.py \
    --mode production \
    --dataset imagenet21k \
    --batch-size 256 \
    --use-amp
```

### NVIDIA H100 (80GB)

```bash
# Development
python scripts/train_phase1.py \
    --mode dev \
    --batch-size 384 \
    --use-amp \
    --compile

# Production
torchrun --nproc_per_node=8 scripts/train_phase1.py \
    --mode production \
    --dataset imagenet21k \
    --batch-size 192 \
    --lr 0.0003 \
    --use-amp \
    --compile

# With FSDP for large model
torchrun --nproc_per_node=8 scripts/train_phase1.py \
    --mode production \
    --dataset imagenet21k \
    --batch-size 96 \
    --use-fsdp \
    --use-amp
```

### NVIDIA A100 (80GB)

```bash
# Development
python scripts/train_phase1.py \
    --mode dev \
    --batch-size 256 \
    --use-amp

# Production (single GPU)
python scripts/train_phase1.py \
    --mode production \
    --dataset imagenet21k \
    --batch-size 192 \
    --gradient-accumulation 2 \
    --use-amp \
    --compile

# Production (8x A100)
torchrun --nproc_per_node=8 scripts/train_phase1.py \
    --mode production \
    --dataset imagenet21k \
    --batch-size 96 \
    --use-amp
```

### NVIDIA A100 (40GB)

```bash
# Development
python scripts/train_phase1.py \
    --mode dev \
    --batch-size 128 \
    --use-amp

# Production
torchrun --nproc_per_node=4 scripts/train_phase1.py \
    --mode production \
    --dataset imagenet21k \
    --batch-size 96 \
    --gradient-accumulation 4 \
    --use-amp
```

### RTX 6000 Ada (48GB)

```bash
# Development
python scripts/train_phase1.py \
    --mode dev \
    --batch-size 256 \
    --use-amp \
    --compile

# Production
python scripts/train_phase1.py \
    --mode production_3b \
    --dataset imagenet21k \
    --batch-size 128 \
    --gradient-accumulation 4 \
    --use-amp \
    --compile
```

### RTX 5090 (32GB)

```bash
# Development
python scripts/train_phase1.py \
    --mode dev \
    --batch-size 256 \
    --use-amp \
    --compile

# Production (3B scale recommended)
python scripts/train_phase1.py \
    --mode production_3b \
    --dataset imagenet21k \
    --batch-size 96 \
    --gradient-accumulation 4 \
    --use-amp \
    --compile
```

### RTX 4090 (24GB)

```bash
# Development
python scripts/train_phase1.py \
    --mode dev \
    --batch-size 128 \
    --use-amp \
    --compile

# Production (1B scale)
python scripts/train_phase1.py \
    --mode production_1b \
    --dataset imagenet21k \
    --batch-size 64 \
    --gradient-accumulation 8 \
    --use-amp \
    --compile
```

### RTX 3090 (24GB)

```bash
# Development
python scripts/train_phase1.py \
    --mode dev \
    --batch-size 96 \
    --use-amp

# Production (1B scale)
python scripts/train_phase1.py \
    --mode production_1b \
    --dataset imagenet21k \
    --batch-size 48 \
    --gradient-accumulation 8 \
    --use-amp
```

---

## Phase 2: Event-Driven Vision Encoder

### RTX PRO 6000 Blackwell (96GB)

```bash
# Development
python scripts/train_phase2.py \
    --mode dev \
    --batch-size 256 \
    --use-amp \
    --compile

# Production
python scripts/train_phase2.py \
    --mode production \
    --batch-size 128 \
    --epochs 50 \
    --use-amp \
    --compile
```

### H100 / A100 (80GB)

```bash
torchrun --nproc_per_node=8 scripts/train_phase2.py \
    --mode production \
    --batch-size 96 \
    --use-amp \
    --compile
```

### RTX 4090 / RTX 3090 (24GB)

```bash
python scripts/train_phase2.py \
    --mode dev \
    --batch-size 64 \
    --gradient-accumulation 4 \
    --use-amp
```

---

## Phase 3: HTM Temporal Memory

### RTX PRO 6000 Blackwell (96GB)

```bash
# With new Accelerated HTM (AHTM)
python scripts/train_phase3.py \
    --mode production \
    --batch-size 128 \
    --use-reflex-memory \
    --column-count 32768 \
    --cells-per-column 128 \
    --use-amp \
    --compile

# Multi-GPU
torchrun --nproc_per_node=2 scripts/train_phase3.py \
    --mode production \
    --batch-size 128 \
    --use-amp
```

### H100 / A100 (80GB)

```bash
torchrun --nproc_per_node=8 scripts/train_phase3.py \
    --mode production \
    --batch-size 64 \
    --use-reflex-memory \
    --use-amp
```

### RTX 4090 / 5090 (24-32GB)

```bash
python scripts/train_phase3.py \
    --mode dev \
    --batch-size 32 \
    --use-reflex-memory \
    --gradient-accumulation 4 \
    --use-amp \
    --compile
```

---

## Phase 4: Multimodal Integration

### RTX PRO 6000 Blackwell (96GB)

```bash
# Full multimodal with Selection-Broadcast Workspace
python scripts/train_phase4.py \
    --mode production \
    --batch-size 64 \
    --workspace-dim 4096 \
    --use-selection-broadcast \
    --use-amp \
    --compile

# Development (vision + text only)
python scripts/train_phase4.py \
    --mode dev \
    --modalities vision,text \
    --batch-size 128 \
    --use-amp
```

### H100 (80GB)

```bash
torchrun --nproc_per_node=8 scripts/train_phase4.py \
    --mode production \
    --batch-size 48 \
    --workspace-dim 4096 \
    --use-selection-broadcast \
    --use-amp
```

### A100 (40-80GB)

```bash
torchrun --nproc_per_node=4 scripts/train_phase4.py \
    --mode production \
    --batch-size 48 \
    --gradient-accumulation 2 \
    --use-amp
```

### RTX 6000 Ada (48GB)

```bash
python scripts/train_phase4.py \
    --mode production_3b \
    --batch-size 32 \
    --gradient-accumulation 4 \
    --workspace-dim 2048 \
    --use-amp \
    --compile
```

### RTX 4090 (24GB)

```bash
python scripts/train_phase4.py \
    --mode dev \
    --modalities vision,text \
    --batch-size 16 \
    --gradient-accumulation 8 \
    --workspace-dim 1024 \
    --use-amp \
    --compile
```

---

## Phase 5: Active Inference

### RTX PRO 6000 Blackwell (96GB)

```bash
# With improved 3-component EFE + empowerment
python scripts/train_phase5.py \
    --mode production \
    --batch-size 64 \
    --use-improved-efe \
    --use-empowerment \
    --planning-horizon 16 \
    --use-amp \
    --compile
```

### H100 / A100 (80GB)

```bash
torchrun --nproc_per_node=8 scripts/train_phase5.py \
    --mode production \
    --batch-size 32 \
    --use-improved-efe \
    --use-amp
```

### RTX 4090 (24GB)

```bash
python scripts/train_phase5.py \
    --mode dev \
    --batch-size 16 \
    --use-improved-efe \
    --gradient-accumulation 8 \
    --use-amp \
    --compile
```

---

## Phase 6: Neuro-Symbolic Reasoning

### RTX PRO 6000 Blackwell (96GB)

```bash
# With Logic Tensor Networks
python scripts/train_phase6.py \
    --mode production \
    --batch-size 32 \
    --use-ltn \
    --ltn-embedding-dim 256 \
    --num-reasoning-steps 32 \
    --use-amp \
    --compile
```

### H100 (80GB)

```bash
torchrun --nproc_per_node=8 scripts/train_phase6.py \
    --mode production \
    --batch-size 24 \
    --use-ltn \
    --use-amp
```

### A100 (40-80GB)

```bash
python scripts/train_phase6.py \
    --mode production \
    --batch-size 24 \
    --use-ltn \
    --gradient-accumulation 4 \
    --use-amp
```

### RTX 4090 (24GB)

```bash
python scripts/train_phase6.py \
    --mode dev \
    --batch-size 8 \
    --use-ltn \
    --gradient-accumulation 8 \
    --use-amp
```

---

## Phase 7: Meta-Learning

### RTX PRO 6000 Blackwell (96GB)

```bash
# With MAML++ and Task2Vec
python scripts/train_phase7.py \
    --mode production \
    --batch-size 16 \
    --tasks-per-batch 8 \
    --use-maml-plus-plus \
    --use-task2vec \
    --inner-steps 10 \
    --use-amp \
    --compile
```

### H100 / A100 (80GB)

```bash
torchrun --nproc_per_node=4 scripts/train_phase7.py \
    --mode production \
    --batch-size 8 \
    --tasks-per-batch 4 \
    --use-maml-plus-plus \
    --use-amp
```

### RTX 4090 (24GB)

```bash
python scripts/train_phase7.py \
    --mode dev \
    --batch-size 4 \
    --tasks-per-batch 2 \
    --use-maml-plus-plus \
    --gradient-accumulation 8 \
    --use-amp
```

---

## Full Pipeline Training

### RTX PRO 6000 Blackwell (96GB) - RECOMMENDED

```bash
# Full 7B model training with all improvements
python scripts/train_full_pipeline.py \
    --mode production \
    --batch-size 64 \
    --use-all-improvements \
    --use-amp \
    --compile \
    --checkpoint-dir checkpoints/full_7b_rtxpro6000 \
    --log-dir logs/7b_training
```

### Multi-GPU H100 Cluster (8x H100)

```bash
torchrun --nproc_per_node=8 --nnodes=1 scripts/train_full_pipeline.py \
    --mode production \
    --batch-size 48 \
    --use-all-improvements \
    --use-fsdp \
    --use-amp \
    --checkpoint-dir checkpoints/full_7b_h100
```

### Multi-Node Training (4 nodes x 8 GPUs)

```bash
# On each node (modify RANK and MASTER_ADDR)
torchrun --nproc_per_node=8 \
    --nnodes=4 \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=29500 \
    scripts/train_full_pipeline.py \
    --mode production \
    --batch-size 32 \
    --use-fsdp \
    --use-amp
```

---

## Environment Variables for Optimization

```bash
# For RTX PRO 6000 Blackwell / RTX 5090 (Blackwell arch)
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="9.0"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# For H100 (Hopper arch)
export TORCH_CUDA_ARCH_LIST="9.0"
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5

# For A100 (Ampere arch)
export TORCH_CUDA_ARCH_LIST="8.0"

# For RTX 4090 / RTX 6000 Ada (Ada Lovelace arch)
export TORCH_CUDA_ARCH_LIST="8.9"

# For RTX 3090 (Ampere arch)
export TORCH_CUDA_ARCH_LIST="8.6"

# Memory optimization (all GPUs)
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# Performance optimization
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true
```

---

## Estimated Training Times

| GPU Config | Phase 1 (MNIST) | Phase 1 (ImageNet-21K) | Full Pipeline |
|------------|-----------------|------------------------|---------------|
| 1x RTX PRO 6000 Blackwell | ~5 min | ~2 days | ~2 weeks |
| 8x H100 | ~2 min | ~8 hours | ~3 days |
| 4x A100 (80GB) | ~3 min | ~18 hours | ~1 week |
| 1x RTX 6000 Ada | ~8 min | ~4 days | ~3 weeks |
| 1x RTX 5090 | ~8 min | ~5 days | ~4 weeks |
| 1x RTX 4090 | ~10 min | ~7 days (1B scale) | ~6 weeks (1B) |
| 1x RTX 3090 | ~15 min | ~10 days (1B scale) | ~8 weeks (1B) |

---

## Memory Optimization Tips

### For GPUs with < 48GB VRAM

1. **Enable gradient checkpointing**:
   ```bash
   --gradient-checkpointing
   ```

2. **Use FP16/BF16 mixed precision**:
   ```bash
   --use-amp --amp-dtype bfloat16  # for Ampere+
   ```

3. **Reduce model scale**:
   ```bash
   --mode production_1b  # or production_3b
   ```

4. **Increase gradient accumulation**:
   ```bash
   --gradient-accumulation 16
   ```

5. **Enable CPU offloading** (slower but works):
   ```bash
   --cpu-offload
   ```

### For Multi-GPU Setups

1. **Use FSDP for > 4 GPUs**:
   ```bash
   --use-fsdp --fsdp-sharding-strategy full
   ```

2. **Enable activation checkpointing**:
   ```bash
   --activation-checkpointing
   ```

---

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
--batch-size 16

# Increase gradient accumulation
--gradient-accumulation 16

# Use smaller model
--mode production_1b

# Enable gradient checkpointing
--gradient-checkpointing
```

### Slow Training

```bash
# Enable compilation (PyTorch 2.0+)
--compile

# Use more workers
--num-workers 16

# Enable prefetch
--prefetch-factor 4
```

### NaN Loss

```bash
# Reduce learning rate
--lr 0.0001

# Enable gradient clipping
--grad-clip 1.0

# Use FP32 instead of AMP for debugging
# (remove --use-amp)
```
