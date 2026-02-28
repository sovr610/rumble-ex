# DDP Integration — Reference for Distributed Scaling Skill

This document specifies the process group initialization, model wrapping, metric aggregation, and gradient synchronization strategies for DistributedDataParallel (DDP) training of the `brain_ai` system. Use this as the canonical reference when implementing, auditing, or debugging DDP infrastructure.

---

## 1. Process Group Initialization

Every DDP training session begins with a process group — the communication fabric that coordinates gradient synchronization across ranks. The `torch.distributed` package provides this through `init_process_group`.

### 1.1 Core Initialization Sequence

```python
import os
import torch
import torch.distributed as dist

def init_process_group(rank: int, world_size: int, backend: str = "nccl",
                       master_addr: str = "localhost", master_port: str = "29500"):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)
```

Critical details:

- **NCCL backend** is mandatory for GPU-to-GPU communication. It uses NVIDIA's NCCL library for optimal bandwidth on NVLink and PCIe topologies. Use `"gloo"` only for CPU-only training or debugging on machines without GPUs.
- **`torch.cuda.set_device(rank)`** must be called before any CUDA allocation on that process. Without it, all processes default to GPU 0, causing memory contention and silent correctness errors.
- **MASTER_ADDR and MASTER_PORT** define the rendezvous endpoint. For single-node training, `localhost` works. For multi-node, set `MASTER_ADDR` to the IP of node 0 and ensure the port is open on all nodes.

### 1.2 torchrun vs Manual Launch

When using `torchrun` (the recommended launcher), environment variables `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR`, and `MASTER_PORT` are set automatically. The initialization simplifies to:

```python
def init_from_torchrun():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank
```

For SLURM-managed clusters, `SLURM_PROCID` maps to rank and `SLURM_NTASKS` maps to world size. The launcher must translate these:

```python
def init_from_slurm():
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    # SLURM sets MASTER_ADDR via scontrol or SLURM_NODELIST
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    return local_rank
```

### 1.3 Cleanup

Always call `dist.destroy_process_group()` at the end of training. Failing to do so can leave zombie NCCL communicators that consume GPU memory and cause hangs on subsequent launches.

```python
def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()
```

---

## 2. Wrapping BrainAI with DDP

The `BrainAI` model has conditional forward paths (symbolic reasoning is skipped when `use_symbolic=False`, HTM is optional, engram encoder competes in workspace only when enabled). This creates **unused parameters** — parameters allocated by modules that do not participate in every forward pass.

### 2.1 Basic Wrapping

```python
from torch.nn.parallel import DistributedDataParallel as DDP

def wrap_with_ddp(model: BrainAI, rank: int,
                  find_unused_parameters: bool = False,
                  sync_batchnorm: bool = True) -> DDP:
    device = torch.device(f"cuda:{rank}")
    model = model.to(device)

    if sync_batchnorm:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    ddp_model = DDP(
        model,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=find_unused_parameters,
    )
    return ddp_model
```

### 2.2 find_unused_parameters

When `BrainAI` is configured with `use_symbolic=True` but the reasoner's System 2 path is not taken (confidence above threshold), certain parameters in the System 2 GRU are not used in that forward pass. DDP normally expects all parameters to contribute to the backward pass. If any parameter has no gradient, DDP will hang waiting for its all_reduce bucket to be filled.

Setting `find_unused_parameters=True` tells DDP to scan the autograd graph after each forward pass and mark parameters that were not part of the computation. These are excluded from gradient synchronization for that step. The cost is roughly 5-10% overhead per iteration due to the graph traversal.

**When to enable:**
- `use_symbolic=True` (dual-process reasoner has conditional System 2 path)
- `use_engram=True` with dynamic workspace competition (engram may not win)
- Any configuration where `task` argument changes which output head is used

**When to disable (for better performance):**
- All modules are always active (all feature flags True, single task type)
- The forward pass is deterministic in which parameters participate

### 2.3 SyncBatchNorm

The `BrainAI` vision encoder uses BatchNorm layers in its convolutional backbone. In DDP, each rank computes batch statistics on its local micro-batch. With small per-GPU batch sizes (common when training 1B+ models), local statistics become noisy and degrade convergence.

`torch.nn.SyncBatchNorm.convert_sync_batchnorm()` replaces all `BatchNorm` layers with `SyncBatchNorm`, which aggregates mean and variance across all ranks before normalization. The communication cost is one all_reduce per BatchNorm layer per forward pass, but the statistical benefit is significant for batch sizes below 32 per GPU.

**Important:** Call `convert_sync_batchnorm` before wrapping with DDP. Applying it after DDP wrapping has no effect because DDP does not recursively convert modules inside its wrapper.

### 2.4 Accessing the Inner Model

DDP wraps the model inside a `.module` attribute. All direct access to model methods or state dict must go through `ddp_model.module`:

```python
# Correct
state = ddp_model.module.state_dict()
ddp_model.module.reset_state()

# Incorrect - accesses the DDP wrapper, not the model
state = ddp_model.state_dict()  # Keys will have 'module.' prefix
```

For checkpoint saving, always use `ddp_model.module.state_dict()` on rank 0 only. Saving from all ranks is wasteful and can cause filesystem contention.

---

## 3. All-Reduce for Metrics

During training, each rank computes local metrics (loss, accuracy, throughput). To report correct aggregated metrics, these must be reduced across all ranks.

### 3.1 Loss Aggregation

```python
def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Reduce a scalar tensor by averaging across all ranks."""
    if not dist.is_initialized():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt
```

This is used after computing loss on each rank:

```python
loss = criterion(output, target)
loss.backward()

# For logging only — do NOT use reduced loss for backward
reduced_loss = all_reduce_mean(loss.detach())
if rank == 0:
    logger.info(f"Step {step}: loss = {reduced_loss.item():.4f}")
```

**Critical:** The loss used for `backward()` must be the local, unreduced loss. DDP handles gradient synchronization automatically — calling `all_reduce` on the loss tensor and then `backward()` would double-count gradients.

### 3.2 Metric Dictionary Aggregation

For richer metrics (accuracy, spike rates, anomaly scores), reduce a dictionary:

```python
def all_reduce_metrics(metrics: Dict[str, float],
                       device: torch.device) -> Dict[str, float]:
    reduced = {}
    for key, value in metrics.items():
        tensor = torch.tensor(value, device=device, dtype=torch.float32)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        reduced[key] = tensor.item() / dist.get_world_size()
    return reduced
```

For count-based metrics (total correct, total samples), use SUM without division:

```python
def all_reduce_sum(metrics: Dict[str, float],
                   device: torch.device) -> Dict[str, float]:
    reduced = {}
    for key, value in metrics.items():
        tensor = torch.tensor(value, device=device, dtype=torch.float32)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        reduced[key] = tensor.item()
    return reduced
```

### 3.3 Non-Blocking Reduction

For overlapping communication with computation:

```python
work = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)
# Do other work here...
work.wait()
```

Use this when metric aggregation is not on the critical path (e.g., logging every N steps rather than every step).

---

## 4. Gradient Clipping with DDP

The `brain_ai` training config specifies `grad_clip: float = 1.0`. With DDP, gradient clipping must account for the fact that gradients are already synchronized (averaged) across ranks after the backward pass.

### 4.1 Correct Clipping Order

```python
loss.backward()  # DDP synchronizes gradients in backward hooks
# At this point, all ranks have identical averaged gradients

# Clip the synchronized gradients
torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)

# Step the optimizer — all ranks take identical steps
optimizer.step()
optimizer.zero_grad()
```

This is the correct order because:
1. `backward()` triggers DDP's gradient hooks, which all_reduce gradients across ranks.
2. After backward completes, all ranks have identical `.grad` tensors.
3. `clip_grad_norm_` computes the total norm and clips in-place. Since all ranks start with identical gradients, they compute identical norms and clip identically.
4. `optimizer.step()` applies identical updates, keeping model parameters synchronized.

### 4.2 Incorrect Patterns

**Clipping before backward completes:**
```python
loss.backward()
# BAD: some gradient hooks may not have fired yet if using gradient_as_bucket_view
clip_grad_norm_(model.parameters(), 1.0)
```

In practice, PyTorch's DDP implementation ensures all hooks fire before backward returns, so this is technically safe. But for clarity and to prevent subtle bugs with custom backward hooks, always clip after backward.

**Clipping different norms on different ranks:**
This cannot happen if the model and data pipeline are correct. If it does happen, it indicates a desynchronization bug — likely caused by non-deterministic operations or forgetting to set seeds identically across ranks.

---

## 5. Bucket Configuration and Communication Optimization

DDP groups parameters into buckets for all_reduce communication. Larger buckets amortize communication latency but delay the start of communication. The default bucket size is 25MB.

### 5.1 Bucket Size Tuning

```python
ddp_model = DDP(
    model,
    device_ids=[rank],
    bucket_cap_mb=25,  # Default: 25MB per bucket
)
```

For `BrainAI` at 1B parameters (~4GB in fp32), the model has roughly 160 buckets at the default size. Communication starts as soon as the first bucket's gradients are all computed (during backward pass), overlapping with computation of subsequent layers.

**Tuning guidelines:**
- **Increase** bucket size (50-100MB) when network bandwidth is high (NVLink, InfiniBand) and the model has many small parameters (typical of BrainAI's neuromodulatory gates).
- **Decrease** bucket size (10-15MB) when network is slow (PCIe between nodes) to start communication earlier.

### 5.2 gradient_as_bucket_view

```python
ddp_model = DDP(
    model,
    device_ids=[rank],
    gradient_as_bucket_view=True,  # Reduce memory by aliasing
)
```

When enabled, parameter `.grad` fields point directly into the communication buffer instead of allocating separate tensors. This saves approximately one model's worth of memory (4GB for 1B params). Enable this for all production training.

---

## 6. Multi-Node DDP

For training across multiple machines:

### 6.1 Network Requirements

- All nodes must be reachable from each other on the master port.
- NCCL requires either InfiniBand or RoCE for optimal performance. TCP/Ethernet works but is 5-10x slower for all_reduce.
- Set `NCCL_SOCKET_IFNAME` to the correct network interface if the default is wrong:

```bash
export NCCL_SOCKET_IFNAME=eth0
```

### 6.2 Launch Command

```bash
# Node 0
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
    --master_addr=10.0.0.1 --master_port=29500 train.py

# Node 1
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
    --master_addr=10.0.0.1 --master_port=29500 train.py
```

### 6.3 Fault Tolerance

For long-running 7B training jobs, use `torch.distributed.elastic` (via torchrun):

```bash
torchrun --nproc_per_node=8 --nnodes=2:4 --rdzv_backend=c10d \
    --rdzv_endpoint=10.0.0.1:29500 --max_restarts=3 train.py
```

The `--nnodes=2:4` syntax specifies a minimum of 2 and maximum of 4 nodes, allowing elastic scaling. The `--max_restarts=3` allows the job to recover from individual node failures.

---

## 7. BrainAI-Specific DDP Considerations

### 7.1 Stateful Modules

The HTM layer and workspace working memory maintain internal state across forward passes. In DDP, each rank maintains its own state. This is correct because each rank processes different data, so states should diverge. However:

- **Reset state identically** at the start of each epoch to prevent state divergence from compounding.
- **Do not all_reduce state tensors** — they are not gradients and should reflect local data patterns.

```python
# At epoch start
if hasattr(ddp_model.module, 'reset_state'):
    ddp_model.module.reset_state()
```

### 7.2 Neuromodulation Synchronization

The neuromodulatory gate produces modulator values (dopamine, acetylcholine, norepinephrine, serotonin) that influence learning rates and plasticity. In DDP, modulators are computed from local data and should remain local. The optimizer step is already synchronized through gradient synchronization.

### 7.3 Active Inference Sampling

When `task='active_inference'`, the forward pass samples actions stochastically. Different ranks will sample different actions. This is correct — each rank is processing different inputs and should take different actions. The gradient through the sampling (via reparameterization or REINFORCE) is correctly synchronized by DDP.

### 7.4 Checkpoint Saving Protocol

```python
if rank == 0:
    checkpoint = {
        'model': ddp_model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'step': global_step,
        'config': config,
    }
    torch.save(checkpoint, f"checkpoint_epoch{epoch}.pt")

# All ranks must wait for rank 0 to finish saving
dist.barrier()
```

The barrier ensures no rank starts the next epoch (and potentially overwrites data) before rank 0 has finished writing. For large models, consider saving asynchronously using a separate thread on rank 0.

---

## 8. Debugging DDP Issues

### 8.1 Gradient Synchronization Verification

To verify that all ranks have identical gradients after backward:

```python
for name, param in ddp_model.named_parameters():
    if param.grad is not None:
        grad_sum = param.grad.sum()
        gathered = [torch.zeros_like(grad_sum) for _ in range(world_size)]
        dist.all_gather(gathered, grad_sum)
        if rank == 0:
            maxdiff = max(abs(g - gathered[0]) for g in gathered)
            assert maxdiff < 1e-5, f"{name}: gradient mismatch {maxdiff}"
```

### 8.2 NCCL Debugging

Enable NCCL debugging output:

```bash
export NCCL_DEBUG=INFO          # Basic info
export NCCL_DEBUG=WARN          # Only warnings
export NCCL_DEBUG_SUBSYS=ALL    # All subsystems
```

### 8.3 Timeout Configuration

If training hangs on a collective operation:

```python
dist.init_process_group(
    backend="nccl",
    timeout=datetime.timedelta(minutes=30),  # Default is 30 min
)
```

For debugging, reduce the timeout to make hangs fail fast:

```python
timeout=datetime.timedelta(seconds=60)
```

### 8.4 Common Hang Causes

| Cause | Symptom | Fix |
|-------|---------|-----|
| Rank missing a collective | One rank hangs indefinitely | Ensure all ranks execute the same collective calls in the same order |
| OOM on one rank | Hangs after that rank crashes | Reduce batch size, enable gradient checkpointing |
| Network partition | Timeout error | Check firewall rules, verify MASTER_ADDR reachability |
| Deadlock in data loader | Hangs before backward | Set `num_workers=0` to diagnose, then fix worker initialization |

---

## 9. Performance Benchmarks

Expected DDP scaling efficiency for BrainAI (1B configuration):

| GPUs | Batch/GPU | Effective Batch | Throughput (samples/s) | Scaling Efficiency |
|------|-----------|-----------------|------------------------|--------------------|
| 1 | 32 | 32 | 100 | 100% |
| 2 | 32 | 64 | 192 | 96% |
| 4 | 32 | 128 | 370 | 93% |
| 8 | 32 | 256 | 710 | 89% |

Scaling efficiency drops with more GPUs due to:
- All_reduce communication time (scales as O(world_size) for ring-all_reduce)
- Synchronization overhead at each step boundary
- CPU bottlenecks in data loading and preprocessing

For the 7B model, FSDP is required instead of DDP (model does not fit in single GPU memory). See `fsdp-sharding.md` for details.
