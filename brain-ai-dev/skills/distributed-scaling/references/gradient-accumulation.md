# Gradient Accumulation — Reference for Distributed Scaling Skill

This document specifies the mathematics, implementation, and integration of gradient accumulation with DDP, AMP, and learning rate scheduling for the `brain_ai` system. Use this as the canonical reference when implementing, auditing, or debugging gradient accumulation infrastructure.

---

## 1. Accumulation Mathematics

Gradient accumulation simulates a larger batch size by accumulating gradients across multiple forward-backward passes before taking an optimizer step. This is essential when the desired effective batch size exceeds GPU memory capacity.

### 1.1 Core Equation

```
effective_batch_size = micro_batch_size x accumulation_steps x world_size
```

Where:
- `micro_batch_size` is the per-GPU, per-step batch size that fits in memory
- `accumulation_steps` is the number of forward-backward passes between optimizer steps
- `world_size` is the number of GPUs (DDP or FSDP)

**Example for 7B BrainAI production training:**
```
micro_batch_size = 4        (7B model leaves little room for activations)
accumulation_steps = 16     (from TrainingConfig)
world_size = 8              (8x A100 80GB)
effective_batch_size = 4 x 16 x 8 = 512
```

### 1.2 Loss Scaling

When accumulating gradients, the loss must be divided by `accumulation_steps` so that the total gradient magnitude matches what a single forward pass with the full effective batch would produce:

```python
loss = criterion(output, target)
loss = loss / accumulation_steps  # Scale before backward
loss.backward()
```

**Why this is necessary:** Without scaling, after K accumulation steps, the gradient is K times larger than it would be with a single step at the effective batch size. This makes the effective learning rate K times larger, causing training instability.

**Mathematical proof:**

For a single step with batch B of size N:
```
g_full = (1/N) * sum_{i=1}^{N} grad(L_i)
```

For K accumulation steps with micro-batches of size M (where N = K*M):
```
g_accum = sum_{k=1}^{K} (1/M) * sum_{i=1}^{M} grad(L_{k,i})
        = (K/M) * (1/K) * sum_{k=1}^{K} sum_{i=1}^{M} grad(L_{k,i}) / M * M
        = K * g_full      (without scaling)
```

Dividing each micro-batch loss by K:
```
g_scaled = sum_{k=1}^{K} (1/(K*M)) * sum_{i=1}^{M} grad(L_{k,i})
         = (1/N) * sum_{all i} grad(L_i)
         = g_full          (correct)
```

### 1.3 Interaction with Mean vs Sum Reduction

PyTorch's standard loss functions use `reduction='mean'` by default, which averages over the micro-batch. With this reduction, dividing by `accumulation_steps` gives the correct gradient. If using `reduction='sum'`, divide by `effective_batch_size` instead.

---

## 2. Interaction with DDP

### 2.1 The no_sync Context Manager

DDP synchronizes (all_reduces) gradients at the end of every `backward()` call. During gradient accumulation, this synchronization is wasteful for all steps except the final accumulation step — intermediate gradients do not need to be synchronized because they will be further accumulated.

The `no_sync()` context manager disables gradient synchronization:

```python
for step_idx, (data, target) in enumerate(dataloader):
    is_accumulation_step = (step_idx + 1) % accumulation_steps != 0

    context = ddp_model.no_sync() if is_accumulation_step else nullcontext()

    with context:
        output = ddp_model(data)
        loss = criterion(output, target) / accumulation_steps
        loss.backward()

    if not is_accumulation_step:
        # Gradients are synchronized on this step
        optimizer.step()
        optimizer.zero_grad()
```

### 2.2 Communication Savings

Without `no_sync`: K all_reduces per optimizer step (one per accumulation step)
With `no_sync`: 1 all_reduce per optimizer step

For K=16 accumulation steps with 8 GPUs, this reduces communication by 16x. For the 7B model with approximately 28GB of gradients (in fp32), each all_reduce transfers approximately 56GB through the ring. Avoiding 15 of these saves approximately 840GB of network traffic per optimizer step.

### 2.3 FSDP with Accumulation

FSDP does not have a `no_sync()` context in the same way as DDP. Instead, FSDP's reduce-scatter happens during backward. For accumulation:

```python
# FSDP accumulates gradients correctly by default.
# The gradients in each shard accumulate across micro-steps.
# The reduce-scatter after each backward adds to existing grad shards.

for micro_step in range(accumulation_steps):
    output = fsdp_model(data[micro_step])
    loss = criterion(output, target[micro_step]) / accumulation_steps
    loss.backward()

optimizer.step()
optimizer.zero_grad()
```

With FSDP `FULL_SHARD`, gradients are reduce-scattered after each backward. Since reduce-scatter is an additive operation on the gradient shards, accumulation works naturally. However, this means communication happens on every micro-step (no way to defer it). To reduce this overhead, use `SHARD_GRAD_OP` if memory allows — it does support `no_sync`.

### 2.4 Gradient Accumulation with limit_all_gathers

For FSDP, enabling `limit_all_gathers=True` prevents FSDP from pre-fetching all-gathers for the next layer while the current layer is still computing. This reduces peak memory at the cost of slightly slower computation:

```python
fsdp_model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    limit_all_gathers=True,  # Essential for 7B on 40GB GPUs
)
```

---

## 3. Learning Rate Scaling

### 3.1 Linear Scaling Rule

When the effective batch size changes (due to accumulation or world size), the learning rate should scale proportionally (Goyal et al., 2017):

```
lr_scaled = lr_base x (effective_batch / reference_batch)
```

Where:
- `lr_base` is the baseline learning rate for `reference_batch`
- `reference_batch` is the batch size the baseline LR was tuned for

**BrainAI example:**
```
lr_base = 3e-4           (from TrainingConfig, tuned for batch=256)
reference_batch = 256
effective_batch = 512     (4 x 16 x 8)
lr_scaled = 3e-4 x (512 / 256) = 6e-4
```

### 3.2 Square Root Scaling Alternative

For very large batch sizes (>4096), linear scaling can cause divergence. The square root rule provides a more conservative scaling:

```
lr_scaled = lr_base x sqrt(effective_batch / reference_batch)
```

BrainAI uses linear scaling with warmup (below). The training config's `warmup_steps=2000` provides sufficient stabilization for batch sizes up to 2048.

### 3.3 Warmup Schedule

Warmup is critical when using scaled learning rates. Without warmup, the large initial learning rate causes gradient explosion in the first few steps.

```python
def get_lr(step: int, warmup_steps: int, total_steps: int,
           peak_lr: float, min_lr: float) -> float:
    """Cosine schedule with linear warmup."""
    if step < warmup_steps:
        # Linear warmup from 0 to peak_lr
        return peak_lr * step / warmup_steps
    else:
        # Cosine decay from peak_lr to min_lr
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))
```

### 3.4 Warmup Duration Scaling

When scaling the learning rate, also scale the warmup duration:

```
warmup_steps_scaled = warmup_steps_base x (effective_batch / reference_batch)
```

This ensures the warmup covers the same number of training samples regardless of batch size. For BrainAI with effective_batch=512 (2x the reference), use 4000 warmup steps.

---

## 4. Implementation Pattern

### 4.1 GradientAccumulator Class

```python
class GradientAccumulator:
    def __init__(self, accumulation_steps: int, scaler=None):
        self.accumulation_steps = accumulation_steps
        self.scaler = scaler  # GradScaler for AMP
        self._step_count = 0

    def effective_batch_size(self, micro_batch: int, world_size: int) -> int:
        return micro_batch * self.accumulation_steps * world_size

    def should_step(self) -> bool:
        """Return True if optimizer should step on this micro-batch."""
        return (self._step_count + 1) % self.accumulation_steps == 0

    def step(self, loss, optimizer, model, max_grad_norm=1.0):
        """
        Accumulate gradients and optionally step the optimizer.

        Returns True if optimizer stepped, False if still accumulating.
        """
        scaled_loss = loss / self.accumulation_steps

        if self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        self._step_count += 1

        if self.should_step():
            if self.scaler is not None:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            optimizer.zero_grad()
            return True

        return False
```

### 4.2 Full Training Loop

```python
accumulator = GradientAccumulator(
    accumulation_steps=config.training.gradient_accumulation_steps,
    scaler=GradScaler() if config.training.use_amp and config.training.amp_dtype == "float16" else None,
)

effective_bs = accumulator.effective_batch_size(
    micro_batch=config.training.batch_size,
    world_size=dist.get_world_size(),
)
lr_scale = effective_bs / reference_batch
peak_lr = config.training.learning_rate * lr_scale

for epoch in range(num_epochs):
    for step_idx, (data, target) in enumerate(dataloader):
        # DDP no_sync for non-final accumulation steps
        is_sync_step = accumulator.should_step()
        context = ddp_model.no_sync() if not is_sync_step else nullcontext()

        with context:
            with autocast(device_type='cuda', dtype=amp_dtype):
                output = ddp_model(data)
                loss = criterion(output, target)

            did_step = accumulator.step(loss, optimizer, ddp_model, max_grad_norm=1.0)

        if did_step:
            scheduler.step()
```

---

## 5. AMP Scaler with Accumulation

### 5.1 GradScaler Mechanics

The `GradScaler` maintains a scale factor that multiplies the loss before backward. This prevents gradients from underflowing in fp16. The scale is dynamically adjusted:

- If no inf/nan gradients: scale increases (by `growth_factor`, default 2.0)
- If inf/nan gradients detected: scale decreases (by `backoff_factor`, default 0.5), and the optimizer step is skipped

### 5.2 Scaler with Accumulation: Correct Pattern

```python
scaler = GradScaler()

for micro_step in range(accumulation_steps):
    with autocast(device_type='cuda', dtype=torch.float16):
        output = model(data[micro_step])
        loss = criterion(output, target[micro_step]) / accumulation_steps

    # Scale and backward (accumulates scaled gradients)
    scaler.scale(loss).backward()

# After all micro-steps: unscale, clip, step
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
optimizer.zero_grad()
```

**Critical:** Call `scaler.unscale_()` exactly once before clipping, after all micro-steps. Calling it on each micro-step would unscale partially accumulated gradients, producing incorrect magnitudes.

### 5.3 BFloat16: No Scaler Needed

BFloat16 has the same exponent range as fp32, so gradients do not underflow. The GradScaler is not needed:

```python
# bf16 training — no scaler
for micro_step in range(accumulation_steps):
    with autocast(device_type='cuda', dtype=torch.bfloat16):
        output = model(data[micro_step])
        loss = criterion(output, target[micro_step]) / accumulation_steps
    loss.backward()

torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
optimizer.zero_grad()
```

The BrainAI training config defaults to `amp_dtype='bfloat16'`, so production training does not use GradScaler. The scaler integration exists for compatibility with V100 GPUs.

---

## 6. Effective Batch Size Computation Table

Reference configurations for BrainAI training:

| Scale | Micro Batch | Accum Steps | World Size | Effective Batch | LR Scale (ref=256) |
|-------|------------|-------------|------------|----------------|---------------------|
| Minimal (dev) | 32 | 1 | 1 | 32 | 0.125x |
| 1B (small) | 16 | 4 | 4 | 256 | 1.0x |
| 3B (medium) | 8 | 8 | 8 | 512 | 2.0x |
| 7B (production) | 4 | 16 | 8 | 512 | 2.0x |
| 7B (large cluster) | 4 | 8 | 32 | 1024 | 4.0x |

### 6.1 Choosing Accumulation Steps

The primary constraint is GPU memory. Set `micro_batch_size` to the largest value that fits, then compute `accumulation_steps`:

```
accumulation_steps = target_effective_batch / (micro_batch_size x world_size)
```

If this is not an integer, round up and accept a slightly larger effective batch. Alternatively, reduce micro_batch_size by 1 and recalculate.

### 6.2 Maximum Effective Batch Size

Beyond a certain batch size, training quality degrades (the "large-batch training" problem). For LLM-scale models, the empirical limit is approximately:

- **Stable training:** up to 2048 effective batch size with proper warmup
- **Diminishing returns:** beyond 4096, more steps at smaller batch are better
- **Critical batch size** (McCandlish et al., 2018): for 7B models, approximately 1024-2048

BrainAI targets 512 effective batch for production, well within the stable regime.

---

## 7. Monitoring Accumulation

### 7.1 Gradient Norm Tracking

Track gradient norms across accumulation steps to detect instabilities:

```python
if accumulator.should_step():
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    if rank == 0:
        wandb.log({"grad_norm": grad_norm.item(), "step": global_step})
```

A sudden spike in grad_norm typically indicates:
- Bad data sample (corrupted input, extreme target value)
- Learning rate too high after warmup
- Numerical instability in fp16 (switch to bf16 or increase loss scale)

### 7.2 Loss Tracking

When logging loss during accumulation, track both the scaled micro-batch loss and the reconstructed full-batch loss:

```python
micro_losses = []
for micro_step in range(accumulation_steps):
    loss = criterion(output, target)
    micro_losses.append(loss.item())
    scaled_loss = loss / accumulation_steps
    scaled_loss.backward()

if rank == 0:
    avg_loss = sum(micro_losses) / len(micro_losses)
    wandb.log({"loss": avg_loss})
```

### 7.3 Throughput Calculation

When reporting samples per second, use effective batch size:

```python
elapsed = time.time() - start_time
effective_samples = global_step * effective_batch_size
throughput = effective_samples / elapsed
```

Do not count micro-steps as separate training steps — the learning dynamics are determined by optimizer steps, not forward-backward passes.
