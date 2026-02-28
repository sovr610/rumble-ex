# DDP Setup for Self-Supervised Learning: Deep Reference

## The Online/Target Split

The fundamental insight for DDP in SSL: only the online encoder needs gradient synchronization.

```
Online Encoder + Predictor  →  wrapped with DDP (gradients all-reduced across ranks)
Target Encoder              →  NOT wrapped (updated only via EMA, no gradients)
```

### Why Wrapping the Target is Wrong

If you wrap the target encoder with DDP:

1. **Bandwidth waste**: DDP performs all-reduce on gradients. The target encoder has no gradients (requires_grad=False). DDP will detect this as "unused parameters" and either hang (find_unused_parameters=False) or waste bandwidth traversing the autograd graph to find them (find_unused_parameters=True).

2. **find_unused_parameters error**: With `find_unused_parameters=False` (default for performance), DDP expects ALL wrapped parameters to receive gradients each forward pass. Target encoder parameters don't participate in any computation graph, so DDP raises:
   ```
   RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one.
   ```

3. **EMA conflict**: DDP replaces the weight update mechanism with gradient all-reduce. For the target encoder, which uses EMA instead of gradient descent, DDP's all-reduce would fight with the EMA update.

### Correct Setup

```python
# Online encoder gets DDP wrapping
online_ddp = torch.nn.parallel.DistributedDataParallel(
    online_encoder,
    device_ids=[rank],
    find_unused_parameters=False
)

# Target encoder: plain module, no DDP
target_encoder = copy.deepcopy(online_encoder)
for param in target_encoder.parameters():
    param.requires_grad_(False)
```

---

## find_unused_parameters=False

Setting `find_unused_parameters=False` is correct when all DDP-wrapped parameters receive gradients on every forward pass. This is the normal case for SSL training.

### Why False is Better

With `find_unused_parameters=True`:
- DDP traverses the entire autograd graph after every backward pass
- For a 100M parameter model: adds ~5-15ms per step
- Over 100,000 steps: wastes ~25 minutes of compute

With `find_unused_parameters=False`:
- DDP uses bucket-based all-reduce without graph traversal
- No autograd graph traversal overhead
- Requires that all parameters receive gradients

If your SSL architecture has optional branches or conditional computation paths that might leave some parameters without gradients, set `find_unused_parameters=True` and accept the overhead.

---

## static_graph=True Optimization

When the same parameters receive gradients in the same order every iteration:

```python
online_ddp = torch.nn.parallel.DistributedDataParallel(
    online_encoder,
    device_ids=[rank],
    find_unused_parameters=False,
    static_graph=True   # optional: enables static graph optimization
)
```

With `static_graph=True`:
- DDP learns the gradient buckets during the first forward/backward pass
- Subsequent passes skip the bucket assignment step
- Saves ~1-3ms per step for large models

Requirements:
- The computation graph must be identical every iteration (no conditional branches)
- The set of parameters receiving gradients must not change
- Not compatible with gradient checkpointing in some configurations

---

## SyncBatchNorm: Critical Placement

SyncBatchNorm conversion MUST happen BEFORE DDP wrapping:

```python
# Step 1: Create model
online_encoder = MyEncoder().cuda(rank)

# Step 2: Convert BatchNorm → SyncBatchNorm (BEFORE DDP)
online_encoder = nn.SyncBatchNorm.convert_sync_batchnorm(online_encoder)

# Step 3: Wrap with DDP
online_ddp = torch.nn.parallel.DistributedDataParallel(
    online_encoder, device_ids=[rank]
)
```

### Why Order Matters

`nn.SyncBatchNorm.convert_sync_batchnorm(model)` replaces all `nn.BatchNorm*` layers with `nn.SyncBatchNorm` layers by modifying `model._modules` recursively.

After DDP wrapping, the model is stored inside `ddp_model._modules['module']`. Calling `convert_sync_batchnorm` on the DDP wrapper attempts to recurse into `_modules` which includes DDP metadata — this can succeed but may miss some submodules, or it may silently convert the wrong module.

The safe guarantee: convert before wrap, never after.

### When to Skip SyncBatchNorm

If the architecture uses LayerNorm (standard for ViTs) instead of BatchNorm, `convert_sync_batchnorm` is a no-op (LayerNorm operates per-sample, not per-batch, so no cross-rank synchronization is needed). Skip the conversion call for ViT-based architectures to avoid unnecessary overhead.

---

## DDP Memory Optimizations

### gradient_as_bucket_view=True

```python
online_ddp = torch.nn.parallel.DistributedDataParallel(
    online_encoder,
    device_ids=[rank],
    gradient_as_bucket_view=True   # saves memory
)
```

With `gradient_as_bucket_view=True`:
- Gradient tensors are views into the communication bucket directly
- Eliminates the copy from parameter gradients into buckets
- Saves memory proportional to gradient tensor size (~1x model parameters)

No correctness change: gradients are the same values, just stored in the bucket directly.

### bucket_cap_mb

```python
online_ddp = torch.nn.parallel.DistributedDataParallel(
    online_encoder,
    device_ids=[rank],
    bucket_cap_mb=25   # default: 25MB per bucket
)
```

DDP groups parameters into buckets and all-reduces each bucket as it fills. Larger buckets:
- Fewer all-reduce operations (lower overhead per byte)
- But increases latency before any bucket starts communicating (waits for bucket to fill)

For large models (>1B params): increase to `bucket_cap_mb=50` or `bucket_cap_mb=100`.
For small models (<100M params): default 25MB is fine.

---

## Process Group Lifecycle

### Initialization

```python
import torch.distributed as dist

def setup_distributed(rank, world_size, backend='nccl'):
    dist.init_process_group(
        backend=backend,
        init_method='env://',    # reads MASTER_ADDR, MASTER_PORT from env
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)  # MUST set before model creation
```

Always call `torch.cuda.set_device(rank)` BEFORE creating any CUDA tensors or models. Without this, all ranks use GPU 0, causing out-of-memory errors and incorrect DDP behavior.

### Cleanup

```python
def cleanup_distributed():
    dist.destroy_process_group()
```

Always call in a `finally` block:

```python
try:
    train(rank, world_size)
finally:
    cleanup_distributed()
```

Without `destroy_process_group()`, processes may hang on exit (NCCL maintains background threads that must be explicitly joined).

---

## Multi-Node Setup

For multi-node training with torchrun:

```bash
# On each node:
torchrun \
    --nproc_per_node=8 \
    --nnodes=4 \
    --node_rank=0 \
    --master_addr=<node0-ip> \
    --master_port=12345 \
    train.py
```

Environment variables set by torchrun:
- `LOCAL_RANK`: GPU index on this node (0-7 for 8 GPUs)
- `RANK`: global rank across all nodes
- `WORLD_SIZE`: total processes = nnodes * nproc_per_node
- `MASTER_ADDR`, `MASTER_PORT`: coordinator node address

In code:

```python
rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])

torch.cuda.set_device(local_rank)   # use LOCAL_RANK for device
dist.init_process_group(backend='nccl', init_method='env://')
```

---

## Accessing the Underlying Model

After DDP wrapping, the model is accessed via `.module`:

```python
# Access wrapped model
model_without_ddp = ddp_model.module

# Save checkpoint (always save the unwrapped model)
state_dict = ddp_model.module.state_dict()

# Access model-specific methods
ddp_model.module.some_custom_method()
```

Check for DDP wrapping at runtime:

```python
def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    return model
```

---

## W&B in DDP: Rank-0 Only

```python
import wandb

def setup_logging(rank, cfg):
    if rank == 0:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            config=vars(cfg),
            group="DDP"    # groups all ranks under one run
        )

def log_metrics(rank, metrics):
    if rank == 0:
        wandb.log(metrics)
```

Never call `wandb.init()` on non-zero ranks — this creates duplicate runs and causes confusion in the W&B dashboard.

---

## DistributedSampler

```python
from torch.utils.data import DistributedSampler, DataLoader

def create_distributed_dataloader(dataset, rank, world_size, batch_size, shuffle=True):
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=True   # ensures equal batch sizes across ranks
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    return loader, sampler
```

In the training loop, call `sampler.set_epoch(epoch)` at the start of each epoch to ensure different shuffling across epochs:

```python
for epoch in range(start_epoch, num_epochs):
    sampler.set_epoch(epoch)   # CRITICAL: different shuffle each epoch
    for batch in loader:
        train_step(batch)
```

Without `set_epoch()`, all epochs use the same shuffled order (the seed is fixed by rank), removing randomness from the data order.

---

## Common Pitfalls

### Pitfall 1: Wrapping Target Encoder

Symptom: `RuntimeError: Expected to have finished reduction in the prior iteration`

Cause: DDP-wrapped target encoder has no gradient flow, violating DDP's assumption that all parameters receive gradients.

Fix: Remove DDP wrapper from target encoder.

### Pitfall 2: SyncBatchNorm After DDP Wrap

Symptom: BatchNorm statistics diverge across ranks (training unstable, different losses per rank).

Cause: `convert_sync_batchnorm` was called after DDP wrapping, missed some modules.

Fix: Call `convert_sync_batchnorm` before `DistributedDataParallel()`.

### Pitfall 3: No CUDA Device Set Before Model Creation

Symptom: All ranks create tensors on GPU 0, OOM on GPU 0, GPU 1-7 idle.

Cause: `torch.cuda.set_device(rank)` not called before model creation.

Fix: Always set device as the first line of the distributed worker function.

### Pitfall 4: Forgetting set_epoch()

Symptom: Training data ordering is identical across epochs, reducing effective data diversity.

Cause: `sampler.set_epoch(epoch)` not called at start of each epoch.

Fix: Add `sampler.set_epoch(epoch)` as the first line of the epoch loop.
