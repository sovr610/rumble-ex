# FSDP Sharding — Reference for Distributed Scaling Skill

This document specifies the FullyShardedDataParallel (FSDP) wrapping strategy, sharding policies, activation checkpointing, CPU offloading, mixed precision, and distributed checkpointing for the `brain_ai` system at 3B-7B parameter scales. Use this as the canonical reference when implementing, auditing, or debugging FSDP infrastructure.

---

## 1. Why FSDP for BrainAI

At 7B parameters, the `BrainAI` model requires approximately 28GB in fp32 (or 14GB in fp16/bf16) just for parameters. Adding optimizer state (Adam stores two additional copies: momentum and variance), the memory requirement is approximately:

```
fp32 parameters:     7B x 4 bytes = 28 GB
fp32 gradients:      7B x 4 bytes = 28 GB
Adam momentum:       7B x 4 bytes = 28 GB
Adam variance:       7B x 4 bytes = 28 GB
Total per GPU (DDP): ~112 GB
```

No single GPU (even A100 80GB) can hold this. DDP replicates the entire model on each GPU, so it does not help with per-GPU memory.

FSDP solves this by sharding parameters, gradients, and optimizer states across GPUs. Each GPU holds only 1/N of the model state (where N is the world size), then all-gathers the full parameters just-in-time for each forward/backward computation.

With 8 GPUs and FULL_SHARD:
```
Per-GPU memory: ~112 GB / 8 = ~14 GB (parameters + optimizer)
+ activations:  ~5-15 GB (depends on batch size, sequence length)
+ buffers:      ~2-4 GB (temporary all-gather buffers)
Total:          ~21-33 GB per GPU
```

This fits comfortably in 40GB A100s and allows larger batch sizes on 80GB A100s/H100s.

---

## 2. FSDP Sharding Strategies

PyTorch FSDP provides three sharding strategies:

### 2.1 FULL_SHARD (Recommended for 7B)

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

fsdp_model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
)
```

This shards parameters, gradients, and optimizer states. Each GPU holds 1/N of each. Communication pattern per forward+backward step:

1. **Forward:** All-gather parameters for each FSDP unit before its forward. After forward, discard the gathered parameters (only keep the local shard).
2. **Backward:** All-gather parameters again (needed for gradient computation). After computing gradients, reduce-scatter gradients so each GPU gets 1/N of the gradient.

Communication volume per step: 2 all-gathers + 1 reduce-scatter per FSDP unit.

Memory savings: ~Nx reduction in parameter, gradient, and optimizer memory.

### 2.2 SHARD_GRAD_OP (Recommended for 3B)

```python
fsdp_model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
)
```

This shards gradients and optimizer states but keeps full parameters on each GPU after the forward pass. This reduces communication (no second all-gather in backward) at the cost of higher parameter memory.

Memory equation:
```
Per-GPU parameters: full model (28 GB for 7B)
Per-GPU gradients:  1/N of model (3.5 GB for 7B on 8 GPUs)
Per-GPU optimizer:  1/N of optimizer (7 GB for 7B on 8 GPUs)
Total:              ~38.5 GB for 7B on 8 GPUs
```

This fits in 40GB GPUs for 3B (approximately 16GB parameters + 4.5GB sharded), but not for 7B.

### 2.3 NO_SHARD (DDP Equivalent)

```python
fsdp_model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.NO_SHARD,
)
```

No sharding — equivalent to DDP. Use only for debugging or when comparing FSDP and DDP behavior.

### 2.4 Strategy Selection by Scale

| Scale | Parameters | Strategy | Min GPUs | Min GPU Memory |
|-------|-----------|----------|----------|----------------|
| 1M (minimal) | ~1M | NO_SHARD or DDP | 1 | Any |
| 1B | ~1B | SHARD_GRAD_OP | 2 | 24 GB |
| 3B | ~3B | SHARD_GRAD_OP | 4 | 40 GB |
| 7B | ~7B | FULL_SHARD | 8 | 40 GB (80 GB preferred) |

---

## 3. FSDP Wrapping Policy for BrainAI

FSDP wraps the model into units that are sharded independently. The granularity of wrapping affects both memory efficiency and communication patterns.

### 3.1 Module-Level Wrapping Policy

The `BrainAI` architecture maps directly to wrapping units:

```python
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
import functools

def get_brain_ai_wrapping_policy():
    """Return FSDP wrapping policy aligned with BrainAI architecture."""
    # These module classes are wrapped as individual FSDP units
    wrap_classes = set()

    # Each encoder is a wrapping unit
    # VisionEncoder, TextEncoder, AudioEncoder, SensorEncoder, EngramTextEncoder
    from brain_ai.encoders.vision import VisionEncoder
    from brain_ai.encoders.text import TextEncoder
    from brain_ai.encoders.audio import AudioEncoder
    from brain_ai.encoders.sensors import SensorEncoder
    wrap_classes.update([VisionEncoder, TextEncoder, AudioEncoder, SensorEncoder])

    # SNN Core
    from brain_ai.core.snn import SNNCore
    wrap_classes.add(SNNCore)

    # HTM, Workspace, Decision, Reasoning, Meta
    from brain_ai.temporal.htm import HTMLayer
    from brain_ai.workspace.global_workspace import GlobalWorkspace
    from brain_ai.decision.active_inference import ActiveInferenceAgent
    from brain_ai.decision.output_heads import DecisionHeads
    from brain_ai.reasoning.system2 import DualProcessReasoner
    from brain_ai.meta.neuromodulation import NeuromodulatoryGate
    wrap_classes.update([
        HTMLayer, GlobalWorkspace, ActiveInferenceAgent,
        DecisionHeads, DualProcessReasoner, NeuromodulatoryGate,
    ])

    return ModuleWrapPolicy(wrap_classes)
```

### 3.2 Why Module-Level Granularity

Each wrapping unit is sharded and all-gathered as a single block. The trade-offs:

**Too coarse (wrapping the entire BrainAI as one unit):**
- Memory savings are poor because the entire model must be all-gathered for a single forward call.
- Peak memory equals the full model size, defeating the purpose of FSDP.

**Too fine (wrapping every `nn.Linear`):**
- Excessive communication. Each tiny all-gather has high latency overhead.
- FSDP bookkeeping per unit consumes CPU time.
- Typically 10-30% slower than module-level wrapping.

**Module-level (recommended):**
- Each module (encoder, HTM, workspace, etc.) is 100M-2.5B parameters.
- At most 10-12 FSDP units for the full model.
- Communication is batched efficiently, and peak memory is bounded by the largest single module.

### 3.3 Size-Based Wrapping

An alternative policy wraps any module exceeding a parameter threshold:

```python
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

auto_wrap_policy = functools.partial(
    size_based_auto_wrap_policy,
    min_num_params=100_000_000,  # 100M parameter threshold
)
```

This wraps any submodule with 100M+ parameters. For BrainAI, this naturally captures the large modules (encoders, workspace, engram) while leaving small modules (neuromodulatory gate at ~1M params) unwrapped. The downside is less control over exactly which boundaries are chosen.

### 3.4 Lambda Wrapping for Conditional Modules

BrainAI has optional modules (HTM, reasoner, meta). A custom lambda policy can handle this:

```python
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

def brain_ai_lambda_policy(module: nn.Module) -> bool:
    """Return True if this module should be individually wrapped."""
    target_types = (
        "VisionEncoder", "TextEncoder", "AudioEncoder",
        "SensorEncoder", "SNNCore", "HTMLayer",
        "GlobalWorkspace", "DecisionHeads",
        "DualProcessReasoner", "NeuromodulatoryGate",
    )
    return type(module).__name__ in target_types

auto_wrap_policy = functools.partial(
    lambda_auto_wrap_policy,
    lambda_fn=brain_ai_lambda_policy,
)
```

---

## 4. Activation Checkpointing

Activation checkpointing (gradient checkpointing) trades compute for memory by recomputing intermediate activations during backward instead of storing them.

### 4.1 Per-Module Checkpointing

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

# After FSDP wrapping, apply checkpointing to large modules
apply_activation_checkpointing(
    fsdp_model,
    checkpoint_wrapper_fn=checkpoint_wrapper,
    check_fn=lambda module: isinstance(module, (GlobalWorkspace, DualProcessReasoner)),
)
```

### 4.2 Memory Savings

For the 7B model:
- Without checkpointing: ~15GB activations (batch=32, seq=512)
- With checkpointing on workspace + reasoning: ~6GB activations
- With checkpointing on all modules: ~3GB activations (but 30% slower)

### 4.3 Which Modules to Checkpoint

Prioritize modules with the highest activation-to-parameter ratio:

| Module | Parameters | Activations (batch=32) | Checkpoint Benefit |
|--------|-----------|----------------------|-------------------|
| GlobalWorkspace | 1.5B | ~4GB | High |
| DualProcessReasoner | 800M | ~3GB | High |
| TextEncoder | 340M | ~2GB | Medium |
| Engram | 2.5B | ~2GB | Medium |
| VisionEncoder | 300M | ~1GB | Low |
| SNNCore | 500M | ~1GB | Low |

---

## 5. CPU Offloading

FSDP can offload sharded parameters and gradients to CPU RAM, freeing GPU memory at the cost of PCIe transfer time.

### 5.1 Configuration

```python
from torch.distributed.fsdp import CPUOffload

fsdp_model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    cpu_offload=CPUOffload(offload_params=True),
)
```

### 5.2 When to Use

- **Enable** when GPU memory is insufficient even with FULL_SHARD (e.g., 7B on 4x 40GB GPUs).
- **Disable** when GPU memory is sufficient (e.g., 7B on 8x 80GB GPUs) — CPU offloading adds 20-40% overhead from PCIe data transfers.

### 5.3 Memory Equation with Offloading

```
Per-GPU (7B, 8 GPUs, FULL_SHARD + CPU offload):
  GPU: activations (~5-15GB) + all-gather buffers (~4GB) = ~9-19GB
  CPU: parameters + gradients + optimizer (~14GB per GPU)
```

This allows the 7B model to fit on 4x 24GB GPUs (tight) or 4x 40GB GPUs (comfortable), at the cost of slower training.

---

## 6. Mixed Precision Policy

FSDP natively supports mixed precision training through its `MixedPrecision` policy.

### 6.1 Configuration

```python
from torch.distributed.fsdp import MixedPrecision

# BFloat16 (recommended for A100/H100)
bf16_policy = MixedPrecision(
    param_dtype=torch.bfloat16,     # Parameters stored in bf16
    reduce_dtype=torch.bfloat16,    # Gradient reduction in bf16
    buffer_dtype=torch.bfloat16,    # Buffers (BatchNorm stats) in bf16
)

# Float16 (for older GPUs: V100, T4)
fp16_policy = MixedPrecision(
    param_dtype=torch.float16,
    reduce_dtype=torch.float16,
    buffer_dtype=torch.float16,
)

fsdp_model = FSDP(
    model,
    mixed_precision=bf16_policy,
)
```

### 6.2 Precision Considerations for BrainAI

- **SNN membrane potentials** are sensitive to precision. The LIF neuron accumulates small currents over many timesteps. bf16 (which has the same exponent range as fp32) is safer than fp16 (which can overflow on large potentials).
- **HTM permanence values** change by small increments (0.1 per connection per step). fp16 has limited mantissa precision and may lose these increments. Use fp32 computation for HTM internals.
- **Engram hash tables** use integer indices and are not affected by floating-point precision.
- **Global workspace attention** operates well in bf16 — attention logits are typically in a reasonable range.

### 6.3 Recommendation

Use bf16 mixed precision for all production training. If training on V100 GPUs (no bf16 hardware support), use fp16 with `GradScaler` for loss scaling. The `GradScaler` is not needed with bf16.

---

## 7. Distributed Checkpointing

FSDP requires special checkpointing because parameters are sharded across GPUs.

### 7.1 Sharded Checkpointing (Training Continuity)

For saving and loading with the same FSDP configuration:

```python
import torch.distributed.checkpoint as dcp

# Save
dcp.save(
    state_dict={"model": fsdp_model.state_dict(), "optimizer": optimizer.state_dict()},
    storage_writer=dcp.FileSystemWriter(checkpoint_dir),
)

# Load
dcp.load(
    state_dict={"model": fsdp_model.state_dict(), "optimizer": optimizer.state_dict()},
    storage_reader=dcp.FileSystemReader(checkpoint_dir),
)
```

Sharded checkpoints are saved as one file per rank and are the fastest to save/load because no gathering is needed.

### 7.2 Full State Dict (Portability)

For saving a checkpoint that can be loaded without FSDP (e.g., for inference, or for changing the number of GPUs):

```python
from torch.distributed.fsdp import FullStateDictConfig, StateDictType

# Configure full state dict gathering
full_state_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, full_state_config):
    state_dict = fsdp_model.state_dict()
    if rank == 0:
        torch.save(state_dict, "model_full.pt")
```

**Key parameters:**
- `offload_to_cpu=True`: Gather the full state dict to CPU to avoid GPU OOM when materializing the full 7B model.
- `rank0_only=True`: Only rank 0 gets the full state dict. Other ranks get empty dicts, saving memory.

### 7.3 Phase Boundary Checkpointing

The BrainAI training pipeline has 7 phases. At phase boundaries, save a full state dict for portability:

```python
def save_phase_checkpoint(fsdp_model, optimizer, phase: int, rank: int):
    """Save full-state checkpoint at phase boundary for config portability."""
    # Sharded checkpoint for fast resume within same config
    dcp.save(
        state_dict={"model": fsdp_model.state_dict()},
        storage_writer=dcp.FileSystemWriter(f"checkpoints/phase{phase}_sharded/"),
    )

    # Full checkpoint for portability (only at phase boundaries)
    full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, full_cfg):
        state = fsdp_model.state_dict()
        if rank == 0:
            torch.save(state, f"checkpoints/phase{phase}_full.pt")

    dist.barrier()
```

### 7.4 Loading a Full Checkpoint into FSDP

When starting a new FSDP configuration from a full checkpoint (e.g., changing world size between phases):

```python
# Load full state dict on CPU
if rank == 0:
    state_dict = torch.load("checkpoints/phase3_full.pt", map_location="cpu")
else:
    state_dict = {}

# Broadcast from rank 0 to all ranks (handled by FSDP internally)
full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, full_cfg):
    fsdp_model.load_state_dict(state_dict)
```

---

## 8. FSDP Debugging

### 8.1 Memory Tracking

```python
# After FSDP wrapping
if rank == 0:
    print(f"FSDP sharded params: {sum(p.numel() for p in fsdp_model.parameters()):,}")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

### 8.2 Verifying Sharding

```python
local_params = sum(p.numel() for p in fsdp_model.parameters())
total_params = torch.tensor(local_params, device=f"cuda:{rank}")
dist.all_reduce(total_params, op=dist.ReduceOp.SUM)
if rank == 0:
    print(f"Local params: {local_params:,}, Total params: {total_params.item():,}")
    print(f"Sharding ratio: {total_params.item() / local_params:.1f}x")
```

### 8.3 Common FSDP Errors

| Error | Cause | Fix |
|-------|-------|-----|
| "Parameters not sharded" | Module not wrapped by FSDP policy | Add module class to wrap policy |
| "CUDA OOM during all-gather" | Peak memory exceeds GPU capacity | Enable activation checkpointing or CPU offloading |
| "Incompatible state dict" | Loading checkpoint from different shard count | Use FULL_STATE_DICT for cross-config loading |
| "Gradient dtype mismatch" | Mixed precision and custom backward | Ensure all custom autograd functions respect mixed_precision policy |
| "Hangs on forward" | One rank has different model structure | Verify all ranks construct identical models before wrapping |

---

## 9. Performance Comparison: DDP vs FSDP

For BrainAI at various scales:

| Scale | Strategy | GPUs | Memory/GPU | Throughput | Notes |
|-------|----------|------|-----------|------------|-------|
| 1B | DDP | 4x A100 40GB | 30GB | 100% baseline | Full model fits |
| 1B | FSDP SHARD_GRAD_OP | 4x A100 40GB | 18GB | 95% | Slight comm overhead |
| 3B | DDP | 8x A100 80GB | 72GB | Baseline | Tight fit |
| 3B | FSDP SHARD_GRAD_OP | 4x A100 40GB | 28GB | 90% | Enables smaller GPUs |
| 7B | DDP | N/A | >112GB | N/A | Cannot fit |
| 7B | FSDP FULL_SHARD | 8x A100 80GB | 33GB | 85% vs ideal | Production config |
| 7B | FSDP FULL_SHARD + ckpt | 8x A100 40GB | 25GB | 70% vs ideal | With activation checkpointing |
