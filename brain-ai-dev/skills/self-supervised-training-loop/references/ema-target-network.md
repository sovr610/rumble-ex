# EMA Target Network: Deep Reference

## Core EMA Formula

The exponential moving average update rule for target network parameters:

```
target_param = tau * target_param + (1 - tau) * online_param
```

Equivalently expressed as a convex combination where tau controls the "inertia" of the target:
- tau close to 1.0: target changes slowly (high inertia, stable)
- tau close to 0.0: target tracks online rapidly (low inertia, responsive)

For self-supervised learning (BYOL, DINO), tau is set very close to 1.0 (e.g., 0.996) to provide a slowly-moving, stable target that prevents representation collapse.

---

## Cosine Annealing Schedule

Rather than a fixed tau, cosine annealing gradually increases tau from tau_base to tau_final over training:

```
tau(step) = 1 - (1 - tau_base) * (cos(pi * step / total_steps) + 1) / 2
```

### Derivation

Let f(step) be a schedule that goes from 0 to 1 as step goes from 0 to total_steps, using a cosine shape:

```
f(step) = (1 - cos(pi * step / total_steps)) / 2
```

At step=0: f(0) = (1 - cos(0)) / 2 = (1 - 1) / 2 = 0
At step=total: f(total) = (1 - cos(pi)) / 2 = (1 - (-1)) / 2 = 1

We want tau to go from tau_base (at f=0) to tau_final (at f=1). Linear interpolation:

```
tau(step) = tau_base + f(step) * (tau_final - tau_base)
           = tau_base + (1 - cos(pi * step / total_steps)) / 2 * (tau_final - tau_base)
```

Rearranging to the standard form:

```
tau(step) = 1 - (1 - tau_base) * (cos(pi * step / total_steps) + 1) / 2
```

### Key Values

With tau_base=0.996, tau_final=0.9999, total_steps=100000:

| step      | cos term        | tau        |
|-----------|-----------------|------------|
| 0         | cos(0) = 1.0    | 0.996000   |
| 25000     | cos(pi/4) ≈ 0.707 | 0.998184 |
| 50000     | cos(pi/2) = 0.0 | 0.998950   |
| 75000     | cos(3pi/4) ≈ -0.707 | 0.999516 |
| 100000    | cos(pi) = -1.0  | 0.999900   |

Note: tau_final is approached but not quite reached due to cos(pi) = -1 exactly giving the upper bound.

### Why Cosine Annealing?

At the start of training:
- Online encoder weights are random and unstable
- Fast tracking (low tau) lets the target encoder adapt quickly
- Prevents the target from being stuck at random initialization for too long

At the end of training:
- Online encoder has converged to meaningful representations
- Slow tracking (high tau) provides a stable, slowly-changing target
- The target encoder acts as an exponentially-weighted average over recent checkpoints
- This temporal ensemble effect improves representation quality

The cosine shape provides a smooth transition with most of the tau change happening in the middle of training, avoiding abrupt transitions that could destabilize training.

---

## BYOL vs DINO Schedules

### BYOL Schedule (Bootstrap Your Own Latent)
- tau_base: 0.996
- tau_final: 1.0 (asymptotically, target becomes frozen)
- Schedule: cosine annealing from step 0
- No warmup period for tau

### DINO Schedule (Self-Distillation with No Labels)
- tau_base: 0.996
- tau_final: 1.0
- Schedule: cosine annealing
- Additional linear warmup of tau during first few epochs
  - During warmup: tau = tau_warmup_start + (tau_base - tau_warmup_start) * step / warmup_steps
  - Typical tau_warmup_start: 0.0 or lower (faster tracking during warmup)
- The warmup helps because DINO uses a centering operation that requires the teacher to be responsive early

---

## The @torch.no_grad() Requirement

EMA updates MUST be wrapped in `@torch.no_grad()` or `torch.no_grad()` context:

```python
@torch.no_grad()
def update(online, target, tau):
    for p_online, p_target in zip(online.parameters(), target.parameters()):
        p_target.data.lerp_(p_online.data, 1 - tau)
```

### Why This is Mandatory

Without `@torch.no_grad()`:

1. **Memory leak**: The lerp_ operation creates a computation graph node. Each EMA update step adds a node to the autograd graph connecting `p_target` to `p_online`. Over 100,000 steps, this graph grows to consume all available memory.

2. **Incorrect gradients**: If the target encoder's parameters have `requires_grad=True` (possible if they share the same class as online parameters), and EMA creates a computation graph, calling `loss.backward()` may incorrectly propagate gradients back through the EMA update chain into historical online parameters.

3. **Performance**: Even without the correctness issues, autograd tracking has overhead proportional to the number of parameters updated.

Solution: The target encoder should have `requires_grad=False` on all parameters, AND EMA updates should use `@torch.no_grad()` as a belt-and-suspenders guarantee.

```python
# Set up target with no gradient tracking
target_encoder = copy.deepcopy(online_encoder)
for param in target_encoder.parameters():
    param.requires_grad_(False)
```

---

## Initial Synchronization

Before training begins, synchronize target encoder weights with the online encoder:

```python
@torch.no_grad()
def initial_sync(online, target):
    for p_online, p_target in zip(online.parameters(), target.parameters()):
        p_target.data.copy_(p_online.data)
    for b_online, b_target in zip(online.buffers(), target.buffers()):
        b_target.data.copy_(b_online.data)
```

Why this is necessary:
- If target is independently initialized (e.g., deepcopy after separate init), they may diverge
- The EMA formula assumes target started at the same point as online
- Without initial sync, the first few thousand steps are wasted on correcting the initial mismatch

When to call `initial_sync`:
1. Before the first training step
2. After loading a checkpoint (NOT needed — checkpoint saves both states)
3. After DDP setup if online model was modified during wrapping

---

## Multi-Parameter-Group EMA

For models with multiple parameter groups (e.g., separate groups for encoder vs predictor), iterate over all parameters:

```python
@torch.no_grad()
def update_all_params(online, target, tau):
    # Update parameters
    for p_online, p_target in zip(online.parameters(), target.parameters()):
        p_target.data.lerp_(p_online.data, 1 - tau)

    # ALSO update buffers (BatchNorm running_mean, running_var)
    for b_online, b_target in zip(online.buffers(), target.buffers()):
        b_target.data.copy_(b_online.data)
```

### Buffer Handling

`nn.Module.buffers()` returns registered buffers (tensors that are not parameters). For BatchNorm layers, this includes:
- `running_mean`: exponential moving average of batch means
- `running_var`: exponential moving average of batch variances
- `num_batches_tracked`: counter used for momentum computation

These buffers should be COPIED (not EMA-updated) because:
- The target encoder should use the same running statistics as online
- EMA-ing the running stats would create stale statistics in the target

For LayerNorm: No buffers, nothing to copy.

---

## torch.lerp for Numerical Stability

Use `torch.lerp` instead of manual multiply-add:

```python
# LESS stable: manual computation
p_target.data = tau * p_target.data + (1 - tau) * p_online.data

# MORE stable: torch.lerp
p_target.data.lerp_(p_online.data, 1 - tau)
```

`torch.lerp(input, end, weight)` computes: `input + weight * (end - input)`

This is equivalent to: `(1 - weight) * input + weight * end`

With weight = (1 - tau):
- `input = p_target.data`
- `end = p_online.data`
- Result: `p_target.data + (1-tau) * (p_online.data - p_target.data)`
         = `p_target.data * tau + p_online.data * (1-tau)`

The numerical advantage of lerp: uses fused multiply-add (FMA) hardware instructions on modern CPUs/GPUs, reducing intermediate rounding errors compared to two separate multiplications.

---

## Diagnostic: Verifying EMA is Working

Log the L2 distance between target and online parameters periodically:

```python
@torch.no_grad()
def compute_ema_distance(online, target):
    total_sq_diff = 0.0
    for p_online, p_target in zip(online.parameters(), target.parameters()):
        diff = p_online.data - p_target.data
        total_sq_diff += diff.pow(2).sum().item()
    return total_sq_diff ** 0.5
```

Expected behavior:
- At step 0 (after initial_sync): distance = 0.0
- After first few steps: distance increases as online adapts
- During stable training: distance stabilizes (online and target move together)
- If tau is too low (target tracking too fast): distance stays near 0 (target and online are almost identical — no benefit from EMA)
- If tau is too high (target tracking too slow): distance grows without bound — representation collapse risk

Typical healthy distances for ViT-B: 0.1 to 5.0 (varies widely by layer norm choices and model scale).
