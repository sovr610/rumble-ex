# Task Boundary Detection for Continual Learning

## 1. Overview

Task boundary detection identifies when the data distribution shifts, signalling that the learning system has transitioned from one task to another. In traditional continual learning, task boundaries are provided explicitly (supervised). In more realistic settings -- task-free continual learning -- boundaries must be inferred from the data stream.

Detecting boundaries triggers critical actions:
- **EWC**: compute Fisher diagonal on completed-task data, snapshot optimal parameters
- **Replay**: add representative samples from the completed task to the buffer
- **Progressive**: freeze the current column, instantiate a new one
- **PackNet**: prune and freeze the current task's subnetwork
- **Metrics**: record per-task accuracy for forgetting/BWT computation

---

## 2. Detection Methods

### 2a. Loss Spike Detection

**Idea**: when the data distribution shifts, the model's loss on new data spikes because the model was optimized for the old distribution.

**Algorithm**:
```
Maintain a sliding window of recent losses: L_window = [l_1, l_2, ..., l_W]
Compute running statistics: mu = mean(L_window), sigma = std(L_window)

For each new loss value l_new:
    z_score = (l_new - mu) / (sigma + epsilon)
    if z_score > threshold:
        BOUNDARY DETECTED
    else:
        append l_new to L_window (sliding)
```

**Parameters**:
- `window_size`: number of recent losses to track (default: 50)
- `threshold`: z-score threshold for spike detection (default: 2.0)
- `epsilon`: stability constant for zero-variance windows (default: 1e-8)

**Pros**: simple, no model-specific knowledge required
**Cons**: may false-trigger on noisy loss landscapes; may miss gradual shifts

### 2b. Gradient Norm Monitoring

**Idea**: a distribution shift causes gradient norms to spike as the model encounters unexpected patterns.

**Algorithm**:
```
Maintain a sliding window of gradient norms: G_window = [g_1, ..., g_W]
Compute running statistics: mu_g = mean(G_window), sigma_g = std(G_window)

For each training step:
    g_new = norm(grad(L, theta))
    z_score = (g_new - mu_g) / (sigma_g + epsilon)
    if z_score > grad_norm_threshold:
        BOUNDARY DETECTED
    else:
        append g_new to G_window
```

**Parameters**:
- `window_size`: same sliding window (default: 50)
- `grad_norm_threshold`: z-score threshold (default: 5.0, higher than loss because gradients are noisier)

**Pros**: more sensitive to subtle distribution shifts (the model "notices" before loss spikes)
**Cons**: gradient norms are inherently noisy; requires careful thresholding

### 2c. Distribution Shift Detection

**Idea**: monitor the distribution of model activations or input features. A significant shift indicates a task change.

**Algorithm**:
```
Maintain running statistics of hidden activations: mu_h, sigma_h (per-layer)

For each mini-batch:
    h = model.extract_features(x)
    mu_batch = mean(h)
    sigma_batch = std(h)

    # Compute KL divergence or Wasserstein distance
    shift = wasserstein_1d(mu_h, sigma_h, mu_batch, sigma_batch)

    if shift > distribution_threshold:
        BOUNDARY DETECTED
    else:
        update mu_h, sigma_h with exponential moving average
```

**Parameters**:
- `distribution_threshold`: shift magnitude threshold (dataset-dependent)
- `ema_decay`: decay rate for running activation statistics (default: 0.99)
- `feature_layer`: which layer's activations to monitor (default: penultimate)

**Pros**: detects distribution shifts even before loss or gradient effects manifest
**Cons**: most complex to implement; threshold tuning is domain-dependent

### 2d. Manual Boundaries

**Idea**: the simplest approach -- the user or system explicitly signals task transitions.

**Usage**:
```python
detector = TaskBoundaryDetector(method="manual")
# ...training loop...
detector.signal_boundary(task_id=2)  # Explicit call
```

This is the baseline approach and is always supported alongside automatic detection.

---

## 3. Task-Free Continual Learning

In task-free continual learning, no explicit task labels or boundaries are provided. The system must:

1. Detect when the data distribution changes (using methods 2a-2c above)
2. Decide whether the change constitutes a "new task" or just natural variation
3. Trigger appropriate CL mechanisms without explicit task IDs

### Challenges

- **Gradual shifts**: tasks may blend into each other without a sharp boundary
- **Recurring tasks**: a previously seen task may return; detecting this avoids unnecessary new-task overhead
- **Multi-modal data**: different modalities may shift at different times
- **False positives**: triggering a boundary too often wastes computation (unnecessary Fisher recomputation, buffer updates)

### Recommended Approach

Use a two-stage detector:

1. **Fast detector** (loss spike): cheap, runs every step, catches large shifts
2. **Confirmation detector** (distribution shift): expensive, runs only when fast detector triggers, confirms the shift is real

```python
class TwoStageDetector:
    def __init__(self, fast_threshold=2.0, confirm_threshold=0.5):
        self.fast = LossSpikeDetector(threshold=fast_threshold)
        self.confirm = DistributionShiftDetector(threshold=confirm_threshold)
        self.pending_confirmation = False

    def step(self, loss, grad_norm, features):
        if self.pending_confirmation:
            if self.confirm.check(features):
                self.pending_confirmation = False
                return True  # Confirmed boundary
            else:
                self.pending_confirmation = False
                return False  # False alarm

        if self.fast.check(loss):
            self.pending_confirmation = True

        return False
```

---

## 4. Boundary Response Protocol

When a boundary is detected, the ContinualLearner should execute the following sequence:

```
1. SNAPSHOT
   - Store current model parameters as theta*_t
   - Record task metadata (task_id, duration, final metrics)

2. CONSOLIDATE (method-specific)
   - EWC: compute Fisher diagonal on recent data window
   - SI: finalize path integral, compute omega for completed task
   - PackNet: prune and freeze current task's subnetwork
   - Progressive: freeze current column

3. BUFFER UPDATE
   - Add representative samples from the completed task to the replay buffer
   - Rebalance buffer if using class-balanced strategy
   - Update priorities if using prioritized replay

4. PREPARE NEXT TASK
   - Progressive: instantiate new column with lateral connections
   - PackNet: re-initialize free (unpruned) weights
   - Distillation: snapshot teacher model

5. RESET MONITORS
   - Clear sliding windows for loss/gradient monitoring
   - Reset distribution statistics baselines
   - Increment internal task counter
```

---

## 5. Detection Latency

Detection latency is the number of steps between the actual task boundary and the detector firing. Lower is better, but zero-latency is only possible with manual boundaries.

| Method | Typical Latency | Factors |
|---|---|---|
| Manual | 0 steps | Exact timing known |
| Loss spike | 1-10 steps | Window size, threshold |
| Gradient norm | 1-15 steps | Gradient noise, threshold |
| Distribution shift | 5-20 steps | EMA decay, threshold |
| Two-stage | 5-25 steps | Sum of fast + confirm |

**Done-when gate (d)** requires detection within 10 steps of a synthetic boundary injection.

---

## 6. Handling Edge Cases

### Boundary at Epoch Start

If a boundary is detected at the very beginning of training (before the window is filled), use a conservative approach:
- Require the window to be at least 50% full before enabling detection
- Use a higher threshold during warmup

### Multiple Rapid Boundaries

If boundaries fire in quick succession (< window_size steps apart):
- Implement a cooldown period: ignore boundary signals for N steps after a detection
- Default cooldown: window_size // 2

### No Boundary (Single-Task Training)

If no boundary is ever detected, the CL mechanisms should be no-ops:
- EWC penalty = 0 (no prior Fisher/params)
- Replay buffer = empty (no prior samples)
- Progressive = single column (no laterals)

---

## 7. References

- Aljundi, R., et al. (2019). "Task-Free Continual Learning." CVPR 2019.
- He, X., & Jaeger, H. (2018). "Overcoming Catastrophic Interference using Conceptor-Aided Backpropagation." ICLR 2018.
- Lee, S., et al. (2020). "A Neural Dirichlet Process Mixture Model for Task-Free Continual Learning." ICLR 2020.
- Zeno, C., et al. (2018). "Task Agnostic Continual Learning Using Online Variational Bayes." arXiv:1803.10123.
