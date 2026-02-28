# Testing Matrix Reference

## Overview

This matrix defines the complete set of test scenarios for V-JEPA 2 distributed training infrastructure. Tests are organized by component and include both unit tests (no distributed setup required) and integration tests (require multi-process or simulated distributed environment).

---

## 1. Distributed Initialization Tests

### 1.1 SLURM Detection

| Scenario | Env Vars Set | Expected Result |
|----------|-------------|-----------------|
| SLURM job | `SLURM_NTASKS`, `SLURM_PROCID`, `SLURM_LOCALID` | `is_slurm()` returns `True` |
| torchrun (local) | `RANK`, `LOCAL_RANK`, `WORLD_SIZE` | `is_slurm()` returns `False` |
| Single process | None | `is_slurm()` returns `False` |
| Partial SLURM vars | Only `SLURM_PROCID` | `is_slurm()` returns `False` (incomplete) |

### 1.2 Init Return Values

| Scenario | Expected `(rank, local_rank, world_size)` |
|----------|------------------------------------------|
| Single-process fallback | `(0, 0, 1)` |
| SLURM rank 0 of 8 | `(0, 0, 8)` |
| SLURM rank 3 of 8 | `(3, 3, 8)` |
| Multi-node SLURM (rank 9, 2 nodes x 8 GPU) | `(9, 1, 16)` |

### 1.3 Cleanup

| Scenario | Expected Behavior |
|----------|------------------|
| Cleanup when not initialized | No exception (idempotent) |
| Cleanup after init | `dist.is_initialized()` returns `False` |
| Double cleanup | No exception (idempotent) |

### 1.4 CUDA Device Assignment

| Scenario | Expected |
|----------|----------|
| local_rank=0, CUDA available | `torch.cuda.current_device()` == 0 |
| local_rank=3, CUDA available | `torch.cuda.current_device()` == 3 |
| No CUDA available | No exception, CPU mode |

---

## 2. Custom Distributed Ops Tests

### 2.1 AllGather

#### Forward Pass (Single-Process)

| Input Shape | Expected Output Shape |
|-------------|----------------------|
| `[4, 128]` | `[4, 128]` (world_size=1, identity) |
| `[1, 512]` | `[1, 512]` |
| `[8, 32, 64]` | `[8, 32, 64]` |

#### Forward Pass (Multi-Process, world_size=4)

| Per-Rank Input Shape | Expected Output Shape |
|----------------------|----------------------|
| `[4, 128]` | `[16, 128]` |
| `[2, 256]` | `[8, 256]` |
| `[1, 64, 32]` | `[4, 64, 32]` |

#### Backward Pass (Gradient Flow)

| Test | Assertion |
|------|-----------|
| `requires_grad=True` input survives AllGather | `.grad` is not None after `.backward()` |
| Gradient shape after backward | Same as input shape |
| Gradient values correct (world_size=1) | `grad == upstream_grad` |
| No gradient blocking | `x.grad.sum()` != 0 for non-zero upstream grad |

### 2.2 AllReduceSum

| Scenario | Expected |
|----------|----------|
| Single process, input `[3.0]` | Output `[3.0]` |
| 4 processes, each with `[1.0]` | Each receives `[4.0]` |
| Forward is differentiable | `grad_fn` is not None |
| Backward is identity | `grad_output == grad_input` |

### 2.3 AllReduce

| Scenario | Expected |
|----------|----------|
| Single process, input `[3.0]` | Output `[3.0]` |
| 4 processes, each with `[4.0]` | Each receives `[4.0]` (sum/n = 16/4) |
| Forward is differentiable | `grad_fn` is not None |
| Backward is identity | `grad_output == grad_input` |

---

## 3. Checkpoint Tests

### 3.1 Save/Load Roundtrip

| Scenario | Assertion |
|----------|-----------|
| Save and load a simple state dict | All tensors equal after load |
| Save at epoch 5, reload | `ckpt["epoch"] == 5` |
| All checkpoint keys present | `{"epoch", "encoder", "predictor", "target_encoder", "opt", "scaler"} <= set(ckpt.keys())` |
| Reload optimizer state | `opt.state_dict()` matches before/after |
| Reload scaler state | `scaler.state_dict()` matches before/after |
| None scaler roundtrip | `ckpt["scaler"] is None` |

### 3.2 Retry Logic

| Scenario | Expected |
|----------|----------|
| File exists, loads on first try | No retry, returns dict |
| First 2 attempts raise IOError, 3rd succeeds | Returns dict (no exception) |
| All 5 attempts fail | Raises RuntimeError |
| Retry waits exponentially | Wait times: ~1s, ~2s, ~4s, ~8s |
| `max_retries=1` -- first failure raises | RuntimeError on first failure |

### 3.3 Prefix Stripping

| Input Key | Prefix | Expected Output Key |
|-----------|--------|-------------------|
| `"module.layer.weight"` | `"module."` | `"layer.weight"` |
| `"backbone.layer.bias"` | `"backbone."` | `"layer.bias"` |
| `"layer.weight"` (no prefix) | `"module."` | `"layer.weight"` (unchanged) |
| `"module.backbone.layer.weight"` | `"module."` | `"backbone.layer.weight"` |
| Empty state dict | Any prefix | Empty dict |

### 3.4 Pretrained Loading (strict=False)

| Scenario | Expected |
|----------|----------|
| Load with extra keys in checkpoint | Succeeds; extra keys in `unexpected_keys` |
| Load with missing keys | Succeeds with `strict=False`; missing in `missing_keys` |
| Load `pos_embed`-less checkpoint into RoPE model | Succeeds; `pos_embed` in missing_keys |
| Load with `module.` prefix | Auto-strips; loads cleanly |

---

## 4. SLURM / submitit Tests

### 4.1 Trainer Callable

| Scenario | Expected |
|----------|----------|
| `Trainer(args).__call__()` executes | Completes without exception |
| `Trainer.checkpoint()` returns `DelayedSubmission` | `isinstance(result, submitit.helpers.DelayedSubmission)` |
| `DelayedSubmission` contains same Trainer | Resubmission args preserved |
| Trainer is serializable | `import pickle; pickle.dumps(trainer)` succeeds |
| Trainer state preserved across serialization roundtrip | `args` identical |

### 4.2 Code Snapshotting

| Scenario | Expected |
|----------|----------|
| Snapshot copies `.py` files | All `.py` files present in dest dir |
| Snapshot preserves directory structure | Relative paths match source |
| Snapshot excludes `__pycache__` | No `__pycache__` dirs in dest |
| Snapshot excludes hidden dirs | No `.git`, `.mypy_cache` in dest |
| Snapshot is idempotent | Running twice produces same result |
| Custom extensions respected | Only specified extensions copied |

### 4.3 SLURMSubmitter Config

| Scenario | Expected |
|----------|----------|
| Default `SLURMConfig` values | nodes=1, gpus_per_node=8, timeout=4320 |
| `account=""` -- not set in executor params | No `slurm_account` kwarg |
| `qos=""` -- not set in executor params | No `slurm_qos` kwarg |
| Valid config serializes to dict | All fields present |

---

## 5. Performance Optimizer Tests

### 5.1 Activation Checkpointing

| Scenario | Expected |
|----------|----------|
| Enable checkpointing on a simple Sequential | No exception during forward |
| Forward pass with checkpointing produces same output | Output tensors equal (within floating point tolerance) |
| Memory estimate reduced | Reported peak memory lower with checkpointing |
| Backward pass still works | Gradients non-None after `.backward()` |

### 5.2 Mixed Precision

| Scenario | Expected |
|----------|----------|
| `enable_mixed_precision()` returns a GradScaler | `isinstance(scaler, GradScaler)` |
| GradScaler initial scale is positive | `scaler.get_scale() > 0` |
| bfloat16 autocast context applied | Tensors inside context are `torch.bfloat16` |

### 5.3 torch.compile

| Scenario | Expected |
|----------|----------|
| `compile_model()` returns an `nn.Module` | `isinstance(result, nn.Module)` |
| Compiled model forward produces same output | Outputs equal (within tolerance) |
| Compile on PyTorch < 2.0 gracefully falls back | Returns original model with warning |

### 5.4 Memory Measurement

| Scenario | Expected |
|----------|----------|
| `measure_memory()` returns dict with expected keys | `{"allocated_mb", "reserved_mb", "peak_mb"} <= set(result.keys())` |
| All values are non-negative floats | All values >= 0 |
| Without CUDA, returns CPU fallback | No exception |

---

## 6. Integration Tests

### 6.1 End-to-End Single-Process Training Step

| Scenario | Expected |
|----------|----------|
| Forward + backward with AllGather (world_size=1) | Gradients non-None |
| Checkpoint save after training step | File exists |
| Load checkpoint and continue | Loss continues decreasing |

### 6.2 Memory Optimization Combinations

| Combination | Expected |
|-------------|----------|
| Checkpointing + bfloat16 | Both work together |
| Checkpointing + compile | Both work together |
| bfloat16 + compile | Both work together |
| All enabled simultaneously | No exception |

---

## 7. Done-When Gate Validation Tests

These are the three acceptance tests that must pass for the skill to be considered complete:

### Gate 1: Distributed Init

```python
def test_gate_distributed_init():
    setup = DistributedSetup()
    rank, local_rank, world_size = setup.init()
    assert rank == 0
    assert local_rank == 0
    assert world_size == 1  # Single-process fallback
    setup.cleanup()  # Idempotent
    setup.cleanup()  # Must not raise
```

### Gate 2: AllGather Gradient Flow

```python
def test_gate_allgather_gradient():
    x = torch.randn(4, 32, requires_grad=True)
    y = AllGather.apply(x)
    # world_size=1: y == x, shape preserved
    assert y.shape == x.shape
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
```

### Gate 3: Checkpoint Robustness

```python
def test_gate_checkpoint_robustness(tmp_path):
    manager = CheckpointManager(str(tmp_path), max_retries=5)
    state = {
        "epoch": 42,
        "encoder": {"weight": torch.randn(16, 8)},
        "predictor": {"bias": torch.zeros(16)},
        "target_encoder": {"weight": torch.randn(16, 8)},
        "opt": {"state": {}, "param_groups": []},
        "scaler": None,
    }
    path = manager.save(state, epoch=42)
    loaded = manager.load(path)
    assert loaded["epoch"] == 42
    assert torch.allclose(loaded["encoder"]["weight"], state["encoder"]["weight"])
```

---

## Test Environment Requirements

| Test Category | Requirements |
|---------------|-------------|
| Unit tests (ops, checkpoint, setup detection) | Python 3.9+, PyTorch, no GPU, no distributed |
| Mixed precision tests | PyTorch >= 1.6 (GradScaler) |
| torch.compile tests | PyTorch >= 2.0 (graceful fallback if older) |
| Multi-process distributed tests | 2+ processes via `torch.multiprocessing.spawn` or `torchrun` |
| SLURM submitit tests | submitit installed; executor in "local" mode for CI |
| GPU memory tests | CUDA-capable GPU |

All unit tests must pass in single-process CPU mode to support CI environments without GPU access.
