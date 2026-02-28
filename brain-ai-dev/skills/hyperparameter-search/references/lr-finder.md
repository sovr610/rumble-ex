# Learning Rate Finder

## Overview

The learning rate (LR) is the single most impactful hyperparameter in neural network training. Choosing a learning rate that is too small leads to painfully slow convergence; choosing one that is too large causes divergence. The Learning Rate Range Test, introduced by Leslie Smith (2015, 2018), provides a systematic, cheap method for finding good learning rate bounds.

This document covers the Smith LR range test algorithm, exponential LR scheduling during the test, loss smoothing, divergence detection, suggested LR extraction heuristics, and cyclic LR policies that use the discovered bounds.

---

## 1. The Smith LR Range Test

### Core Idea

Train the model for a single pass through a subset of data while exponentially increasing the learning rate from a very small value (e.g., 1e-7) to a very large value (e.g., 10). Plot the smoothed loss against the learning rate on a log scale. The optimal learning rate region is where the loss is decreasing most steeply.

### Algorithm

```
Input: model, train_loader, criterion, start_lr=1e-7, end_lr=10, num_steps=100
Output: [(lr_i, loss_i) for i in 1..num_steps]

1. Save model state (weights, optimizer)
2. Set optimizer LR = start_lr
3. Compute LR multiplier: mult = (end_lr / start_lr) ^ (1 / num_steps)
4. For step = 1 to num_steps:
    a. Get next mini-batch (x, y) from train_loader
    b. Forward pass: loss = criterion(model(x), y)
    c. If loss > divergence_threshold * best_loss: STOP (diverged)
    d. Record (current_lr, loss)
    e. Backward pass: loss.backward()
    f. Optimizer step: optimizer.step()
    g. Update LR: optimizer.lr *= mult
5. Restore model state (weights, optimizer)
6. Return recorded (lr, loss) pairs
```

### Key Details

**State Preservation:** The LR finder modifies both the model weights and optimizer state. The original state must be saved before the test and restored afterward. Without this, the model is corrupted by training at extreme learning rates.

**Mini-batch Iteration:** The test uses a single pass through the data. If `num_steps` exceeds the number of batches in the train_loader, the loader wraps around. This is fine; the LR finder does not need to see unique data for every step.

**Stop-on-Divergence:** Once the loss exceeds `divergence_threshold` times the best observed loss (typically 4-5x), the test terminates. Continuing past divergence provides no useful information and wastes time.

---

## 2. Exponential LR Schedule

### Why Exponential?

The learning rate spans many orders of magnitude (e.g., 1e-7 to 10 = 8 orders). A linear schedule would spend most of its budget on the high end where training has already diverged. An exponential schedule distributes steps evenly across the log-scale, ensuring good resolution in both the low-LR and high-LR regimes.

### Mathematics

Given `start_lr`, `end_lr`, and `num_steps`:

```
gamma = (end_lr / start_lr) ^ (1 / num_steps)
lr_at_step_i = start_lr * gamma^i
```

This creates a geometric sequence: `lr_0 = start_lr, lr_1 = start_lr * gamma, ..., lr_n = end_lr`.

On a log scale, the learning rates are evenly spaced:

```
log(lr_i) = log(start_lr) + i * (log(end_lr) - log(start_lr)) / num_steps
```

### Choosing num_steps

- **Too few steps (< 50):** Low resolution; may miss the optimal region.
- **Too many steps (> 500):** The model drifts significantly from its initial state by the end, making late results unreliable.
- **Recommended: 100-200 steps.** This provides good resolution while keeping the test cheap (typically 1-2 minutes).

---

## 3. Loss Smoothing

### Why Smooth?

Raw per-batch losses are noisy due to mini-batch sampling. This noise obscures the underlying trend (loss decreasing, then increasing). Smoothing reveals the clean curve needed for LR selection.

### Exponential Moving Average (EMA)

The standard smoothing method uses an exponential moving average:

```
smoothed_loss_0 = raw_loss_0
smoothed_loss_i = beta * smoothed_loss_{i-1} + (1 - beta) * raw_loss_i
```

Where `beta` is the smoothing factor (typically 0.05-0.1 for the brain_ai context, where beta is actually `1 - smooth_factor`). Higher values of `smooth_factor` yield more smoothing.

**Bias correction:** Early smoothed values are biased toward zero because the EMA starts from the first raw value. Apply bias correction:

```
corrected_loss_i = smoothed_loss_i / (1 - beta^i)
```

This is the same correction used in Adam optimizer's moment estimates.

### Alternative: Simple Moving Average

A simple moving average over a window of k recent values:

```
smoothed_loss_i = mean(raw_loss_{i-k+1}, ..., raw_loss_i)
```

This is simpler but introduces lag. For LR finding, EMA is preferred.

### Recommended Settings

| Setting | Value | Rationale |
|---------|-------|-----------|
| smooth_factor | 0.05 | Light smoothing; preserves detail |
| smooth_factor | 0.1 | Moderate smoothing; good default |
| smooth_factor | 0.2 | Heavy smoothing; use for very noisy losses |

For the brain_ai system, use `smooth_factor=0.05` as the default. The multi-component losses (SNN spike rate, HTM reconstruction, workspace competition) can be noisy, but heavy smoothing risks hiding important transitions.

---

## 4. Divergence Detection

### Threshold Method

The simplest and most reliable method: track the best (minimum) smoothed loss seen so far. If the current smoothed loss exceeds `divergence_threshold * best_loss`, declare divergence and stop.

```
if smoothed_loss_i > divergence_threshold * best_loss:
    stop()
```

**Typical thresholds:**
- `divergence_threshold = 4.0` -- Conservative; stops well before NaN but may miss useful data in the transition region.
- `divergence_threshold = 5.0` -- Good default; captures the full useful range.
- `divergence_threshold = 10.0` -- Aggressive; only stops on clear divergence.

### NaN/Inf Detection

Always check for NaN or infinity in the raw loss. These indicate catastrophic divergence (gradient explosion) and require immediate stopping.

```
if isnan(loss) or isinf(loss):
    stop()
```

### Gradient Monitoring

For more nuanced divergence detection, monitor the gradient norm:

```
grad_norm = sqrt(sum(p.grad.norm()^2 for p in model.parameters()))
if grad_norm > grad_threshold:  # e.g., 1000
    stop()
```

This catches divergence earlier than loss monitoring, which lags because the loss is an average over the batch while gradients respond immediately.

---

## 5. Suggested LR Extraction

### Method 1: Steepest Descent Point

Find the learning rate where the loss is decreasing most rapidly (steepest negative gradient on the log-lr vs. loss curve).

```
gradients = [(loss[i+1] - loss[i]) / (log_lr[i+1] - log_lr[i]) for i in range(n-1)]
min_gradient_idx = argmin(gradients)
suggested_lr = lr[min_gradient_idx]
```

**Rationale:** At this point, the model is learning most efficiently. This is a good choice for the maximum learning rate in a cyclic schedule, or the peak learning rate with warmup.

### Method 2: One Order of Magnitude Before Minimum

Find the learning rate at which the loss reaches its minimum. Then divide by 10 to get the suggested learning rate.

```
min_loss_idx = argmin(smoothed_losses)
suggested_lr = lr[min_loss_idx] / 10
```

**Rationale:** The minimum loss point is where the learning rate is about to cause divergence. Training at that rate is unstable. One order of magnitude lower provides a safety margin.

### Method 3: Min-Max Bounds for Cyclic LR

Extract both minimum and maximum learning rate bounds for cyclic learning rate schedules:

```
# min_lr: 1-2 orders below the steepest descent point
# max_lr: the steepest descent point or slightly lower
steepest_idx = argmin(gradients)
max_lr = lr[steepest_idx]
min_lr = max_lr / 10  # or /100 for more conservative
```

### Recommended Approach for brain_ai

Use Method 1 (steepest descent) as the primary suggestion, with bounds:
- `max_lr = lr_at_steepest_descent`
- `min_lr = max_lr / 10`

These bounds feed directly into the `TrainingConfig.learning_rate` (peak) and `TrainingConfig.min_learning_rate` (floor for cosine decay).

### Edge Cases

- **Monotonically decreasing loss:** The test range did not reach divergence. Increase `end_lr` and re-run.
- **Monotonically increasing loss:** The model is already diverging at `start_lr`. Decrease `start_lr` by a few orders of magnitude.
- **Flat loss with sudden divergence:** The model is not learning at any reasonable rate. Check the model architecture, data, and loss function.
- **Multiple local minima in the loss curve:** Common with complex models like brain_ai. Use the first significant decrease region as the suggested range.

---

## 6. Cyclic Learning Rate Policies

The LR finder's output naturally feeds into cyclic LR schedules, which oscillate the learning rate between the discovered bounds.

### Triangular Policy (CLR)

The simplest cyclic policy: linearly increase LR from `min_lr` to `max_lr` over half a cycle, then linearly decrease back.

```
cycle = floor(1 + step / (2 * step_size))
x = abs(step / step_size - 2 * cycle + 1)
lr = min_lr + (max_lr - min_lr) * max(0, 1 - x)
```

**Step size:** Typically 2-8 epochs. Smith recommends 2-10 times the number of iterations in one epoch.

### Triangular2 Policy

Same as triangular but halves the amplitude each cycle:

```
lr = min_lr + (max_lr - min_lr) * max(0, 1 - x) / (2 ^ (cycle - 1))
```

This provides aggressive exploration early and fine-tuning late.

### One-Cycle Policy (Super-Convergence)

Smith's most effective policy: a single cycle that warms up to `max_lr`, then anneals below `min_lr` with a final fine-tuning phase.

```
Phase 1 (warm-up, ~30% of training): min_lr -> max_lr
Phase 2 (annealing, ~60% of training): max_lr -> min_lr
Phase 3 (fine-tuning, ~10% of training): min_lr -> min_lr/100
```

**Key insight:** The warm-up phase acts as regularization, exploring a wider region of the loss landscape. The rapid annealing finds the nearest good minimum. The fine-tuning phase sharpens convergence.

### Cosine Annealing with Warm Restarts

Used by default in brain_ai's `TrainingConfig`:

```
lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * step / total_steps))
```

Warm restarts periodically reset the LR to `max_lr`, allowing the optimizer to escape local minima:

```
T_i = T_0 * T_mult^i  (cycle length grows)
lr within cycle: cosine annealing from max_lr to min_lr
```

### Policy Selection for brain_ai

| Phase | Recommended Policy | Rationale |
|-------|-------------------|-----------|
| 1 (SNN) | One-Cycle | Super-convergence; SNN training benefits from large LR exploration |
| 2 (Encoders) | Cosine with Warmup | Standard for transformer training |
| 3 (HTM) | Triangular2 | HTM parameters need careful exploration; amplitude decay helps |
| 4 (Workspace) | Cosine with Warmup | Multi-head attention follows transformer conventions |
| 5 (Active Inference) | One-Cycle | Planning horizon optimization benefits from wide exploration |
| 6 (Reasoning) | Cosine with Warmup | Standard for symbolic + neural hybrid |
| 7 (Meta) | Low-amplitude Triangular | Meta-learning is sensitive; keep LR variations small |

---

## 7. Practical Considerations for brain_ai

### Multi-Loss Components

The brain_ai system has multiple loss components (spike rate regularization, temporal consistency, HTM reconstruction, workspace competition, active inference EFE, reasoning logic loss). The LR finder should use the **total weighted loss**, not individual components, because that is what the optimizer actually minimizes.

### Phase-Specific LR Finding

Run the LR finder separately for each training phase. The optimal LR for SNN training (Phase 1) is likely very different from the optimal LR for meta-learning (Phase 7) because:

1. Different parameter counts and scales.
2. Different loss landscapes (spiking dynamics vs. few-shot adaptation).
3. Different optimizer states (fresh vs. partially trained).

### Frozen Parameters

When running the LR finder for a later phase (e.g., Phase 4: Workspace), earlier layers should be frozen as they are in actual training. The LR finder must respect the same parameter freezing schedule as the training pipeline.

### Batch Size Interaction

The optimal learning rate scales approximately linearly with batch size (linear scaling rule). If the LR finder is run with batch_size=32 but training uses effective batch_size=512 (via gradient accumulation), multiply the suggested LR by `512/32 = 16`, or better yet, run the LR finder with the same effective batch size.

### Mixed Precision

When using AMP (as brain_ai does by default), run the LR finder with AMP enabled. The loss scaling in AMP affects the gradient magnitudes, which changes the effective learning rate landscape.

---

## 8. Implementation Notes

### Memory Management

The LR finder creates a copy of the model state at the beginning and restores it at the end. For large models (7B parameters), this requires significant memory. Options:

1. **State dict save/restore:** `torch.save(model.state_dict())` to a file, then `model.load_state_dict()`. Slower but uses disk instead of GPU memory.
2. **In-memory deepcopy:** `copy.deepcopy(model.state_dict())`. Fast but doubles GPU memory usage.
3. **CPU offload:** Save state dict to CPU memory. Middle ground between disk and GPU.

For the brain_ai system at production scale (7B params), use CPU offload. At minimal/testing scale, in-memory deepcopy is fine.

### DataLoader Handling

The LR finder needs a DataLoader that can be iterated for exactly `num_steps` batches. If the DataLoader is exhausted before `num_steps`, it should be restarted (wrap with `itertools.cycle` or similar). The LR finder should accept any DataLoader and handle exhaustion gracefully.

### Plotting

The `LearningRateFinder.plot()` method should produce:

1. A log-scale x-axis (learning rate).
2. The smoothed loss on the y-axis.
3. Vertical lines at the suggested min_lr and max_lr.
4. Annotations showing the suggested values.
5. The raw (unsmoothed) loss as a faint background trace.

If matplotlib is not available (headless server), the plot method should return the data arrays for external plotting.

---

## Summary

The LR finder is the essential first step before any hyperparameter search. It:

1. Eliminates the most impactful hyperparameter (learning rate) from the search space.
2. Provides bounds for cyclic LR policies.
3. Takes only 1-2 minutes even for large models.
4. Should be run per-phase, per-batch-size, and per-frozen-layer configuration.

The suggested workflow is:
1. Run LR finder with `start_lr=1e-7`, `end_lr=10`, `num_steps=100`.
2. Extract `max_lr` at the steepest descent point.
3. Set `min_lr = max_lr / 10`.
4. Use these bounds in the training config's LR scheduler.
5. Include learning rate in the HPO search space with a narrow range around the suggested values (e.g., `[max_lr/3, max_lr*3]` for fine-tuning).
