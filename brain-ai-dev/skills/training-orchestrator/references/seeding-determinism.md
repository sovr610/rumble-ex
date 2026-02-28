# Seeding and Determinism Reference

This document defines the seeding policy, RNG stream isolation strategy, determinism
flags, tolerance definitions, and testing procedures for the brain_ai seven-phase
training pipeline. All training runs must follow these rules to achieve reproducibility
within the tolerances specified in section 5.

---

## 1. Seed Derivation Policy

### 1.1 Base Seed

Set the root seed in `SeedConfig.base_seed`. The default is **1337**. All other seeds
in the system derive from this single value through deterministic additive offsets.
Never use magic numbers; always derive from `base_seed`.

### 1.2 Per-Phase Seeds

Compute each phase seed by adding a fixed offset to the base seed:

```
phase_seed = base_seed + per_phase_offsets[phase - 1]
```

Default offsets (one per phase, phases 1 through 7):

```python
per_phase_offsets = [0, 100, 200, 300, 400, 500, 600]
```

Phase 1 uses the base seed directly. Phase 4 uses `base_seed + 300`. This spacing
leaves room for 100 component-level offsets within each phase without collision.

### 1.3 Per-Component Seeds

Within a phase, derive component-specific seeds from the phase seed:

```
component_seed = phase_seed + component_offset
```

Fixed component offsets:

| Component     | Offset |
|---------------|--------|
| augmentation  | 0      |
| dropout       | 10     |
| sampling      | 20     |
| dataloader    | 30     |

Example for phase 3, dropout:

```python
base_seed = 1337
phase_seed = 1337 + 200  # phase 3 offset
dropout_seed = 1537 + 10  # component offset
# dropout_seed = 1547
```

### 1.4 Implementation

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class SeedConfig:
    base_seed: int = 1337
    per_phase_offsets: List[int] = field(
        default_factory=lambda: [0, 100, 200, 300, 400, 500, 600]
    )

    # Component offsets within a phase
    AUGMENTATION_OFFSET: int = 0
    DROPOUT_OFFSET: int = 10
    SAMPLING_OFFSET: int = 20
    DATALOADER_OFFSET: int = 30

    def phase_seed(self, phase: int) -> int:
        """Return the seed for a given phase (1-indexed)."""
        return self.base_seed + self.per_phase_offsets[phase - 1]

    def component_seed(self, phase: int, component_offset: int) -> int:
        """Return the seed for a specific component within a phase."""
        return self.phase_seed(phase) + component_offset
```

All derivations are additive and deterministic. Given the same `base_seed` and the
same `per_phase_offsets`, every derived seed is identical across runs, machines, and
Python versions.

---

## 2. RNG Stream Isolation

### 2.1 Why Isolation Matters

PyTorch, NumPy, and Python's `random` module each maintain independent RNG states.
Within PyTorch, a single global generator is shared by default across all operations
that draw random numbers -- dropout, data augmentation, weight initialization,
RL sampling. If augmentation draws 50 random numbers in one run but 51 in another
(because an image was resized differently), every subsequent random draw shifts. Dropout
masks change. Sampled episodes differ. The entire training trajectory diverges.

Isolate RNG streams so that changes to one component's randomness do not propagate to
others.

### 2.2 Four Global RNG Sources

Seed all four at the start of each phase:

```python
import random
import numpy as np
import torch

def seed_global(seed: int) -> None:
    """Seed all global RNG sources."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

| Source              | Seeding Call                    | Scope                        |
|---------------------|--------------------------------|------------------------------|
| Python `random`     | `random.seed(seed)`            | General shuffling, sampling  |
| NumPy              | `np.random.seed(seed)`         | Data preprocessing, masking  |
| PyTorch CPU        | `torch.manual_seed(seed)`      | Weight init, CPU ops         |
| PyTorch CUDA       | `torch.cuda.manual_seed_all(seed)` | All GPU devices          |

### 2.3 Isolated `torch.Generator` Instances

Create separate `torch.Generator` objects for components that must not interfere:

```python
class IsolatedRNGStreams:
    """Maintain separate RNG streams for independent components."""

    def __init__(self, phase_seed: int, device: str = "cpu"):
        self.augmentation_gen = torch.Generator(device=device)
        self.augmentation_gen.manual_seed(phase_seed + SeedConfig.AUGMENTATION_OFFSET)

        self.dropout_gen = torch.Generator(device=device)
        self.dropout_gen.manual_seed(phase_seed + SeedConfig.DROPOUT_OFFSET)

        self.sampling_gen = torch.Generator(device=device)
        self.sampling_gen.manual_seed(phase_seed + SeedConfig.SAMPLING_OFFSET)
```

Use these generators explicitly in operations:

```python
# Data augmentation -- uses augmentation generator
noise = torch.randn(x.shape, generator=streams.augmentation_gen)

# Dropout -- uses dropout generator
mask = torch.bernoulli(
    torch.full(x.shape, 1 - dropout_rate),
    generator=streams.dropout_gen,
)

# RL episode sampling -- uses sampling generator
action_idx = torch.multinomial(
    policy_probs, num_samples=1,
    generator=streams.sampling_gen,
)
```

This guarantees that adding a new augmentation transform does not shift dropout
patterns, and changing the RL sampling strategy does not affect data augmentation.

### 2.4 Generator Lifecycle

Create new `IsolatedRNGStreams` at the start of each phase. Do not reuse generators
across phases -- the per-phase seed already encodes phase identity. Within a phase,
the generators persist for the full duration of training.

---

## 3. DataLoader Worker Seeding

### 3.1 The Problem

PyTorch DataLoader workers fork the parent process. Each worker inherits the same
RNG state. Without explicit seeding, all workers produce identical augmentation
sequences, destroying data diversity.

### 3.2 Worker Init Function

Define a `worker_init_fn` that seeds all three RNG sources per worker:

```python
def make_worker_init_fn(base_seed: int):
    """Return a worker_init_fn that seeds each DataLoader worker deterministically."""

    def worker_init_fn(worker_id: int):
        # Incorporate worker_id and epoch for unique-per-worker, unique-per-epoch seeds
        # epoch is injected via a closure or global; see epoch boundary re-seeding below
        epoch = getattr(worker_init_fn, "_epoch", 0)
        worker_seed = base_seed + worker_id + epoch * 1000

        random.seed(worker_seed)
        np.random.seed(worker_seed % (2**32))
        torch.manual_seed(worker_seed)

    return worker_init_fn
```

Attach the function to the DataLoader:

```python
worker_fn = make_worker_init_fn(seed_config.phase_seed(phase))

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    worker_init_fn=worker_fn,
    generator=torch.Generator().manual_seed(
        seed_config.component_seed(phase, SeedConfig.DATALOADER_OFFSET)
    ),
)
```

The `generator` argument controls batch sampling order. The `worker_init_fn` controls
per-worker augmentation randomness.

### 3.3 Epoch Boundary Re-Seeding

For persistent workers (`persistent_workers=True`), workers survive across epochs.
Update the epoch counter so that re-seeding produces different augmentation per epoch:

```python
for epoch in range(num_epochs):
    worker_fn._epoch = epoch
    # If persistent_workers=True, workers call worker_init_fn only once.
    # Re-seed manually via dataset or sampler hooks.
    if hasattr(dataloader.dataset, "set_epoch"):
        dataloader.dataset.set_epoch(epoch)
    for batch in dataloader:
        train_step(batch)
```

For non-persistent workers, `worker_init_fn` is called at every epoch start
automatically by PyTorch when workers are re-spawned.

### 3.4 Manifest Recording

Record the worker count in the manifest:

```json
{
  "seeds": {
    "base_seed": 1337,
    "num_workers": 8,
    "persistent_workers": true
  }
}
```

### 3.5 Changing `num_workers` Changes Results

Document this clearly: **changing `num_workers` changes the training outcome.** Each
worker processes a different subset of samples with a worker-specific seed. Altering
the worker count changes which worker processes which sample, shifting the augmentation
applied to each data point. This is expected and unavoidable. Record `num_workers` in
the manifest so that reproduction runs use the same value.

---

## 4. PyTorch Determinism Flags

### 4.1 Required Flags

Set these unconditionally in all training modes:

```python
import os
import torch

def set_determinism_flags(strict: bool = False) -> None:
    """Configure PyTorch for deterministic execution.

    Args:
        strict: If True, enable torch.use_deterministic_algorithms(True)
                which raises errors on nondeterministic ops.
    """
    # Required: seed the global RNG (done separately via seed_global)
    # torch.manual_seed(seed)           -- REQUIRED
    # torch.cuda.manual_seed_all(seed)  -- REQUIRED for multi-GPU

    # Required: deterministic cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Required: deterministic cuBLAS workspace
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Optional but recommended in dev mode
    if strict:
        torch.use_deterministic_algorithms(True)
```

| Flag | Effect | Required? |
|------|--------|-----------|
| `torch.manual_seed(seed)` | Seed CPU RNG | Yes |
| `torch.cuda.manual_seed_all(seed)` | Seed all GPU RNGs | Yes (multi-GPU) |
| `torch.backends.cudnn.deterministic = True` | Force deterministic cuDNN kernels | Yes |
| `torch.backends.cudnn.benchmark = False` | Disable kernel auto-tuning | Yes |
| `torch.use_deterministic_algorithms(True)` | Error on nondeterministic ops | Dev mode only |
| `CUBLAS_WORKSPACE_CONFIG=:4096:8` | Deterministic cuBLAS matrix multiply | Yes |

### 4.2 Why `benchmark = False`

When `benchmark` is `True`, cuDNN profiles multiple kernel implementations on the first
call and caches the fastest. The selected kernel may differ between runs due to system
load, thermal state, or CUDA driver version. This non-determinism is invisible --
numerically different kernels produce different floating-point rounding. Always disable
benchmarking for reproducible training.

### 4.3 Nondeterministic Operations

The following common PyTorch operations are nondeterministic by default. When
`torch.use_deterministic_algorithms(True)` is enabled, they raise `RuntimeError`:

| Operation | Why Nondeterministic | Workaround |
|-----------|---------------------|------------|
| `scatter_add` / `scatter_add_` | Atomic additions on GPU have undefined order | Use `index_add` where possible |
| `index_put_` with `accumulate=True` | Same atomic addition issue | Avoid accumulate mode or use CPU |
| `torch.Tensor.index_select` backward | Gradient accumulation order | Accept or use gather |
| `torch.nn.functional.interpolate` | Backward pass uses atomics | Use `align_corners=True` or CPU |
| `torch.nn.CTCLoss` backward | Nondeterministic on CUDA | Use CPU for CTC backward |
| `torch.bmm` backward | Nondeterministic accumulation | Set `CUBLAS_WORKSPACE_CONFIG` |
| Sparse tensor ops | Hash-based storage | Avoid sparse ops in deterministic mode |

### 4.4 Dev Mode vs. Production Mode

| Setting | Dev Mode | Production Mode |
|---------|----------|-----------------|
| `cudnn.deterministic` | `True` | `True` |
| `cudnn.benchmark` | `False` | `False` (or `True` if speed critical -- document deviation) |
| `use_deterministic_algorithms` | `True` | `False` (too restrictive for some ops) |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` | `:4096:8` |

In production mode, if `benchmark` is enabled for throughput, record this in the
manifest and accept wider tolerances (section 5.2).

---

## 5. Tolerance Definitions

### 5.1 Same Device, Deterministic Mode

When running twice on the **same GPU model**, with all determinism flags enabled and
identical `num_workers`:

- **Loss**: Exact match within `1e-6` absolute tolerance
- **Accuracy**: Exact match
- **Intermediate tensors**: Bitwise identical

This is the gold standard. Achieve this in dev mode for all unit and integration tests.

### 5.2 Same Device, Relaxed Mode

When `cudnn.benchmark = True` or `use_deterministic_algorithms = False`:

- **Loss**: Within **2% relative** tolerance
- **Accuracy**: Within **1% absolute** tolerance
- **Intermediate tensors**: Cosine similarity > 0.999

### 5.3 Cross-Device (Different GPU Model)

When comparing runs on, e.g., A100 vs. H100:

- **Loss**: Within **5% relative** tolerance
- **Accuracy**: Within **2% absolute** tolerance

Different GPU architectures use different floating-point units with different rounding
behavior. FP32 fused multiply-add on Ampere rounds differently than on Hopper. TF32
further widens this gap. This is fundamental to IEEE 754 and cannot be eliminated.

### 5.4 CPU vs. CUDA

The widest tolerance tier:

- **Intermediate tensors**: Cosine similarity > **0.99**
- **Loss trajectory**: Same convergence trend, not same values

CPU and CUDA use entirely different math libraries (MKL vs. cuBLAS), different
operation fusion, and different precision paths. Numerical agreement beyond cosine
similarity is not expected.

### 5.5 Computing Tolerances

Use relative tolerance with epsilon guard to avoid division by zero:

```python
def relative_tolerance(a: float, b: float, epsilon: float = 1e-8) -> float:
    """Compute relative difference between two values."""
    return abs(a - b) / max(abs(a), epsilon)

def absolute_tolerance(a: float, b: float) -> float:
    """Compute absolute difference between two values."""
    return abs(a - b)
```

For tensor comparison, use PyTorch's built-in:

```python
torch.testing.assert_close(
    tensor_a, tensor_b,
    rtol=1e-5,   # relative tolerance
    atol=1e-6,   # absolute tolerance
)
```

---

## 6. SNN-Specific Seeding Concerns

### 6.1 Surrogate Gradient Determinism

Surrogate gradient functions (ATan, FastSigmoid, StraightThrough) are pure mathematical
functions of their inputs. Given the same input tensor, they produce the same output.
They introduce no randomness themselves. Determinism depends entirely on the input to
the surrogate function being deterministic.

### 6.2 Spike Timing Sensitivity

LIF (Leaky Integrate-and-Fire) neurons accumulate membrane potential over timesteps.
A spike fires when potential exceeds a threshold. Small floating-point differences in
membrane potential can push a neuron above or below threshold on different timesteps,
producing entirely different spike trains. This makes SNN training more sensitive to
floating-point non-determinism than standard neural networks.

Mitigation: always enforce `cudnn.deterministic = True` and `CUBLAS_WORKSPACE_CONFIG`
when training SNN layers. In production mode, accept that cross-device spike timing
may differ and rely on statistical metrics (spike rate, mean firing time) rather than
exact spike-train comparison.

### 6.3 Heterogeneous Tau Values

When `use_heterogeneous_tau = True`, per-neuron time constants are initialized from the
phase seed:

```python
import math

def init_tau_values(num_neurons: int, seed: int) -> torch.Tensor:
    """Initialize heterogeneous tau values deterministically."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    # Log-uniform distribution between 2ms and 50ms
    log_tau = torch.empty(num_neurons).uniform_(
        math.log(2.0), math.log(50.0), generator=gen
    )
    return torch.exp(log_tau)
```

On resume, **do not re-derive tau values from the seed.** The tau values may have been
modified during training (if learnable) or may need to match the checkpoint exactly.
Always serialize tau values in the checkpoint:

```python
# Saving
checkpoint["snn_tau_values"] = model.snn.tau.data.clone()

# Loading (on resume)
model.snn.tau.data = checkpoint["snn_tau_values"]
# Do NOT call init_tau_values again
```

### 6.4 Learnable Delay Buffers

Learnable synaptic delays maintain internal ring buffers. On resume, restore the full
buffer state from the checkpoint, not just the delay parameters:

```python
checkpoint["snn_delay_buffers"] = model.snn.delay_layer.buffers
```

---

## 7. Meta-Learning Seeding

### 7.1 Episode Sampling RNG

Meta-learning (MAML) requires sampling episodes (support set + query set) from a task
distribution. Use a dedicated RNG stream keyed to three values:

```python
def episode_seed(global_seed: int, epoch: int, episode_idx: int) -> int:
    """Deterministic seed for a specific episode."""
    return global_seed + epoch * 10000 + episode_idx
```

This ensures:
- Same episode across re-runs (given same seed)
- Different episodes across epochs
- No collision between episodes within an epoch (up to 10,000 episodes)

```python
def sample_episode(dataset, seed: int, n_support: int, n_query: int):
    """Sample a single episode deterministically."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=gen)
    support = indices[:n_support]
    query = indices[n_support:n_support + n_query]
    return dataset[support], dataset[query]
```

### 7.2 Inner-Loop / Outer-Loop Isolation

The MAML inner loop performs gradient steps on the support set. These steps consume
random numbers (dropout in the adapted model, any stochastic layers). The inner loop
must not consume the outer loop's RNG state, or different numbers of inner steps would
shift the outer loop's randomness.

Isolation strategy: save and restore the global RNG state around the inner loop:

```python
def maml_inner_loop(model, support_data, num_steps, inner_lr):
    """Run MAML inner loop with RNG isolation."""
    # Save outer-loop RNG state
    cpu_state = torch.random.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    )
    np_state = np.random.get_state()
    py_state = random.getstate()

    # Inner loop uses its own RNG trajectory
    # (seeded deterministically by episode_seed, already set before this call)
    adapted_params = {n: p.clone() for n, p in model.named_parameters()}
    for step in range(num_steps):
        loss = compute_loss(model, support_data, adapted_params)
        grads = torch.autograd.grad(loss, adapted_params.values())
        adapted_params = {
            n: p - inner_lr * g
            for (n, p), g in zip(adapted_params.items(), grads)
        }

    # Restore outer-loop RNG state
    torch.random.set_rng_state(cpu_state)
    if cuda_states:
        torch.cuda.set_rng_state_all(cuda_states)
    np.random.set_state(np_state)
    random.setstate(py_state)

    return adapted_params
```

### 7.3 Second-Order MAML Consistency

When `first_order = False`, MAML computes second-order gradients through the inner
loop. The inner-loop trajectory must be identical regardless of the outer-loop
optimization state. This is guaranteed by the RNG isolation above: the inner loop always
sees the same random draws for a given episode seed, regardless of what the outer loop
did before it.

Verify this property in tests by running the inner loop with different outer-loop
parameter values and asserting the inner-loop loss trajectory is identical.

---

## 8. Manifest Recording

### 8.1 Seeds Section Schema

Record all seed-related values in `manifest.json`:

```json
{
  "seeds": {
    "base_seed": 1337,
    "per_phase_offsets": [0, 100, 200, 300, 400, 500, 600],
    "phase": 4,
    "phase_seed": 1637,
    "component_seeds": {
      "augmentation": 1637,
      "dropout": 1647,
      "sampling": 1657,
      "dataloader": 1667
    },
    "deterministic_flags": {
      "cudnn_deterministic": true,
      "cudnn_benchmark": false,
      "use_deterministic_algorithms": true,
      "cublas_workspace_config": ":4096:8"
    },
    "num_workers": 8,
    "persistent_workers": true,
    "torch_version": "2.4.0",
    "cuda_version": "12.4"
  }
}
```

### 8.2 RNG State in Checkpoints

On checkpoint save, capture the full RNG state for all four sources:

```python
def save_rng_state() -> dict:
    """Capture all RNG states for checkpoint."""
    state = {
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    return state


def restore_rng_state(state: dict) -> None:
    """Restore all RNG states from checkpoint."""
    torch.random.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
```

Include the RNG state in every checkpoint:

```python
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": epoch,
    "global_step": global_step,
    "rng_state": save_rng_state(),
    "snn_tau_values": (
        model.snn.tau.data.clone() if hasattr(model, "snn") else None
    ),
}
torch.save(checkpoint, path)
```

### 8.3 Resume Protocol

On resume, **restore RNG states from the checkpoint**, not from seed re-derivation.
Re-deriving from the seed would replay the RNG sequence from the beginning, producing
wrong random draws for the step being resumed:

```python
def resume_from_checkpoint(path, model, optimizer):
    """Resume training with full RNG state restoration."""
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Restore RNG -- do NOT re-seed from base_seed
    restore_rng_state(checkpoint["rng_state"])

    # Restore SNN tau values -- do NOT re-derive from seed
    if checkpoint.get("snn_tau_values") is not None and hasattr(model, "snn"):
        model.snn.tau.data = checkpoint["snn_tau_values"]

    return checkpoint["epoch"], checkpoint["global_step"]
```

---

## 9. Testing Determinism

### 9.1 Dual-Run Comparison

The primary determinism test: run the same configuration twice and compare outputs
tensor-by-tensor.

```python
import torch
from brain_ai import create_brain_ai

def test_forward_pass_determinism():
    """Verify that two forward passes with same seed produce identical output."""
    seed = 1337
    config = {
        "modalities": ["vision"],
        "output_type": "classify",
        "num_classes": 10,
    }

    # Run 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model_1 = create_brain_ai(**config)
    x = torch.randn(2, 1, 28, 28, generator=torch.Generator().manual_seed(42))
    with torch.no_grad():
        out_1 = model_1({"vision": x})

    # Run 2
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model_2 = create_brain_ai(**config)
    with torch.no_grad():
        out_2 = model_2({"vision": x})

    torch.testing.assert_close(out_1, out_2, rtol=0, atol=1e-6)
```

### 9.2 Gradient Determinism

Forward pass determinism does not guarantee backward pass determinism. Test gradients
explicitly:

```python
def test_gradient_determinism():
    """Verify that gradients are deterministic given same seed and input."""
    seed = 1337

    def run_once():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = create_brain_ai(
            modalities=["vision"], output_type="classify", num_classes=10
        )
        model.train()
        x = torch.randn(
            2, 1, 28, 28, generator=torch.Generator().manual_seed(42)
        )
        out = model({"vision": x})
        loss = out.sum()
        loss.backward()
        grads = {
            n: p.grad.clone()
            for n, p in model.named_parameters()
            if p.grad is not None
        }
        return grads

    grads_1 = run_once()
    grads_2 = run_once()

    for name in grads_1:
        torch.testing.assert_close(
            grads_1[name], grads_2[name],
            rtol=0, atol=1e-6,
            msg=f"Gradient mismatch for {name}",
        )
```

### 9.3 Training Loop Smoke Test

Run a 10-step training loop twice and compare loss curves:

```python
def test_training_loop_determinism():
    """10-step training loop must produce identical loss curves."""

    def run_training(seed: int, steps: int = 10):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        model = create_brain_ai(
            modalities=["vision"], output_type="classify", num_classes=10
        )
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        input_gen = torch.Generator().manual_seed(42)

        losses = []
        for step in range(steps):
            x = torch.randn(4, 1, 28, 28, generator=input_gen)
            target = torch.randint(0, 10, (4,), generator=input_gen)
            out = model({"vision": x})
            loss = torch.nn.functional.cross_entropy(out, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return losses

    losses_1 = run_training(seed=1337)
    losses_2 = run_training(seed=1337)

    for step, (l1, l2) in enumerate(zip(losses_1, losses_2)):
        assert abs(l1 - l2) < 1e-6, (
            f"Step {step}: loss_1={l1:.8f}, loss_2={l2:.8f}, "
            f"diff={abs(l1-l2):.2e}"
        )
```

### 9.4 Cross-Device Tolerance Test

When testing across different GPU models, relax tolerances:

```python
def test_cross_device_tolerance(
    losses_device_a: list, losses_device_b: list
):
    """Verify cross-device results are within 5% relative tolerance."""
    for step, (la, lb) in enumerate(zip(losses_device_a, losses_device_b)):
        rel = abs(la - lb) / max(abs(la), 1e-8)
        assert rel < 0.05, (
            f"Step {step}: loss_a={la:.6f}, loss_b={lb:.6f}, "
            f"relative_diff={rel:.4f} exceeds 5% threshold"
        )
```

### 9.5 Checklist

Run these tests as part of CI on every change to training infrastructure:

1. **Forward determinism**: Same seed, same model, same input produces identical output.
2. **Gradient determinism**: Backward pass gradients match across runs.
3. **10-step training loop**: Loss curves match within `1e-6` on same device.
4. **Worker seeding**: Two DataLoader iterations with same config yield same batches.
5. **Resume consistency**: Train 10 steps, checkpoint, resume, compare to uninterrupted
   20-step run -- loss at step 20 must match.
6. **RNG isolation**: Change augmentation seed only -- verify dropout masks unchanged.
7. **Meta-learning isolation**: Different `num_inner_steps` does not change outer-loop
   RNG state after the inner loop.

---

## Appendix A: Complete Seeding Setup Example

End-to-end seeding for a single phase:

```python
import os
import random
import math
import numpy as np
import torch


def setup_deterministic_training(
    seed_config: SeedConfig,
    phase: int,
    device: str = "cuda",
    strict: bool = True,
) -> IsolatedRNGStreams:
    """Complete deterministic setup for one training phase.

    Call this once at the start of each phase, before any model
    construction, data loading, or training.

    Args:
        seed_config: Seed configuration with base_seed and offsets.
        phase: Phase number (1-7).
        device: Device for torch generators.
        strict: Enable torch.use_deterministic_algorithms(True).

    Returns:
        IsolatedRNGStreams for augmentation, dropout, and sampling.
    """
    phase_seed = seed_config.phase_seed(phase)

    # 1. Seed all global RNG sources
    seed_global(phase_seed)

    # 2. Set determinism flags
    set_determinism_flags(strict=strict)

    # 3. Create isolated RNG streams
    streams = IsolatedRNGStreams(phase_seed, device=device)

    # 4. Log what was configured
    print(
        f"Phase {phase}: seeded with phase_seed={phase_seed} "
        f"(base={seed_config.base_seed}, "
        f"offset={seed_config.per_phase_offsets[phase-1]})"
    )
    print(f"  Deterministic mode: strict={strict}")
    print(
        f"  CUBLAS_WORKSPACE_CONFIG="
        f"{os.environ.get('CUBLAS_WORKSPACE_CONFIG', 'NOT SET')}"
    )

    return streams
```

## Appendix B: Quick Reference Table

| Item | Value / Formula |
|------|----------------|
| Default base_seed | 1337 |
| Phase seed | `base_seed + per_phase_offsets[phase - 1]` |
| Phase offsets | `[0, 100, 200, 300, 400, 500, 600]` |
| Augmentation seed | `phase_seed + 0` |
| Dropout seed | `phase_seed + 10` |
| Sampling seed | `phase_seed + 20` |
| DataLoader seed | `phase_seed + 30` |
| Worker seed | `base_seed + worker_id + epoch * 1000` |
| Episode seed | `global_seed + epoch * 10000 + episode_idx` |
| Same device, strict tolerance | `abs(a - b) < 1e-6` |
| Same device, relaxed tolerance | `rel_diff < 0.02`, `abs_diff < 0.01` (accuracy) |
| Cross-device tolerance | `rel_diff < 0.05` |
| CPU vs CUDA tolerance | `cosine_sim > 0.99` |
