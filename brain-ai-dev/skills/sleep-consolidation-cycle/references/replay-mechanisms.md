# Replay Mechanisms Reference

## Table of Contents

1. [Overview](#1-overview)
2. [Sharp-Wave Ripple Biology](#2-sharp-wave-ripple-biology)
3. [Compressed Replay Implementation](#3-compressed-replay-implementation)
4. [Priority Sampling Mathematics](#4-priority-sampling-mathematics)
5. [Replay Buffer Design](#5-replay-buffer-design)
6. [Importance Sampling Correction](#6-importance-sampling-correction)
7. [Temporal Compression Strategies](#7-temporal-compression-strategies)
8. [Integration with NREM Phase](#8-integration-with-nrem-phase)
9. [Appendix: Troubleshooting](#appendix-a-troubleshooting)

---

## 1. Overview

### Purpose

The replay mechanism is the engine of the NREM-like consolidation phase. It selects, compresses,
and re-presents stored experiences to the model for offline gradient updates. The design is
inspired by hippocampal sharp-wave ripple (SWR) replay in biological brains, where recent
experiences are replayed at compressed timescales during slow-wave sleep.

### Key Properties

- **Priority-weighted**: high-surprise or high-reward experiences are replayed more frequently.
- **Temporally compressed**: sequential experiences are subsampled to reduce compute while
  preserving critical transitions.
- **Importance-sampling corrected**: priority sampling introduces bias that is corrected via
  importance sampling weights to maintain unbiased gradient estimates.
- **Deterministic given seed**: replay order is fully reproducible from a seed + priority state.

---

## 2. Sharp-Wave Ripple Biology

### Biological Background

During NREM sleep, the hippocampus generates sharp-wave ripple complexes (SWRs) -- brief
(50-100ms) high-frequency (150-250 Hz) oscillations that co-occur with the replay of
recently encoded spatial and episodic memories. Key findings:

- **Temporal compression**: sequences experienced over seconds are replayed in tens of
  milliseconds (compression ratios of 5-20x).
- **Reverse replay**: some SWR events replay experiences in reverse order, potentially
  supporting credit assignment.
- **Reward modulation**: experiences associated with reward are replayed more frequently.
- **Coordination**: SWRs are coordinated with cortical slow oscillations and thalamic
  spindles, suggesting a multi-system consolidation mechanism.

### Mapping to brain_ai

| Biological Feature | brain_ai Implementation |
|---|---|
| SWR event | Single replay batch from ReplayScheduler |
| Temporal compression | Sequence subsampling at compression_ratio |
| Reward modulation | Priority sampling weighted by TD-error |
| Reverse replay | Optional reverse-order sequence presentation |
| Cortical coordination | Systems consolidation running in tandem |

---

## 3. Compressed Replay Implementation

### Sequence Subsampling

Given an experience sequence of length L and compression ratio r, the compressed sequence
has length L' = ceil(L / r). Elements are selected using a strided sampling strategy that
preserves the first and last elements (boundary anchoring):

```python
def compress_sequence(sequence: Tensor, ratio: float) -> Tensor:
    """Compress a temporal sequence by subsampling.

    Parameters
    ----------
    sequence : Tensor
        Shape (T, *feature_dims) -- temporal sequence of experiences.
    ratio : float
        Compression ratio. ratio=5.0 means 5x compression.

    Returns
    -------
    Tensor
        Shape (T', *feature_dims) where T' = ceil(T / ratio).
    """
    T = sequence.shape[0]
    T_prime = max(2, math.ceil(T / ratio))  # at least 2 elements
    if T_prime >= T:
        return sequence

    # Always include first and last
    indices = torch.linspace(0, T - 1, T_prime).long()
    return sequence[indices]
```

### Key Transition Preservation

Not all timesteps are equally important. Key transitions are identified by:

1. **Reward transitions**: timesteps where reward signal changes significantly.
2. **State transitions**: timesteps with large change in hidden state representation.
3. **Action boundaries**: timesteps where action changes (for discrete action spaces).

The compression algorithm preferentially retains these key transitions:

```python
def compress_with_key_transitions(
    sequence: Tensor,
    rewards: Tensor,
    ratio: float,
    key_transition_bonus: float = 2.0,
) -> Tensor:
    """Compress sequence while preserving key transitions."""
    T = sequence.shape[0]
    T_prime = max(2, math.ceil(T / ratio))

    # Compute importance scores
    reward_deltas = torch.abs(rewards[1:] - rewards[:-1])
    state_deltas = torch.norm(sequence[1:] - sequence[:-1], dim=-1)
    importance = reward_deltas * key_transition_bonus + state_deltas

    # Pad to length T (first element gets mean importance)
    importance = torch.cat([importance.mean().unsqueeze(0), importance])

    # Select top-T' indices by importance, maintaining temporal order
    _, top_indices = importance.topk(min(T_prime, T))
    top_indices = top_indices.sort().values
    return sequence[top_indices]
```

---

## 4. Priority Sampling Mathematics

### Prioritized Experience Replay (PER)

Following Schaul et al. (2016), each experience i has a priority p_i and is sampled with
probability:

```
P(i) = p_i^alpha / sum_j(p_j^alpha)
```

Where alpha (priority_exponent) controls how much prioritization is used:
- alpha = 0: uniform random sampling
- alpha = 1: fully greedy (always sample highest priority)
- alpha = 0.6: default balance (empirically effective)

### Priority Assignment

Priorities are set based on the absolute TD-error (or equivalent surprise signal):

```
p_i = |delta_i| + epsilon
```

Where epsilon is a small constant (1e-6) to ensure all experiences have non-zero probability.

### Sum Tree Data Structure

For efficient O(log N) sampling, priorities are stored in a sum tree:

```
         42
       /    \
      29     13
     / \    / \
    16  13  6  7
   /\ /\  /\ /\
  10 6 8 5 4 2 3 4
```

Each leaf stores a priority value. Internal nodes store the sum of their children.
Sampling selects a random value in [0, total_priority] and traverses the tree to find
the corresponding leaf.

### Rank-Based Priority (Alternative)

Instead of proportional priority, rank-based priority assigns:

```
P(i) = 1 / rank(i)^alpha / sum_j(1 / rank(j)^alpha)
```

This is more robust to outlier TD-errors but slightly more expensive to compute.

---

## 5. Replay Buffer Design

### Circular Buffer with Priority Tree

The replay buffer combines a fixed-size circular array for experience storage with a sum
tree for priority-based sampling:

```python
class ReplayBuffer:
    def __init__(self, max_size: int, obs_shape: tuple, action_dim: int):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        # Storage arrays (pre-allocated)
        self.observations = torch.zeros(max_size, *obs_shape)
        self.actions = torch.zeros(max_size, action_dim)
        self.rewards = torch.zeros(max_size)
        self.next_observations = torch.zeros(max_size, *obs_shape)
        self.dones = torch.zeros(max_size, dtype=torch.bool)

        # Priority tree
        self.priorities = SumTree(max_size)
```

### Storage Format

Each experience is a tuple: (observation, action, reward, next_observation, done, metadata).
Metadata includes:

- `timestamp`: when the experience was collected (for recency weighting).
- `episode_id`: which episode the experience belongs to (for sequential replay).
- `td_error`: most recent TD-error (updated after each replay).
- `replay_count`: how many times this experience has been replayed.

### Eviction Policy

When the buffer is full, FIFO eviction removes the oldest experience. Optionally,
priority-based retention can protect high-priority experiences from eviction:

```python
def add(self, experience, initial_priority=None):
    if initial_priority is None:
        initial_priority = self.priorities.max() if self.size > 0 else 1.0

    # FIFO eviction at self.ptr
    self.observations[self.ptr] = experience.obs
    self.actions[self.ptr] = experience.action
    self.rewards[self.ptr] = experience.reward
    self.next_observations[self.ptr] = experience.next_obs
    self.dones[self.ptr] = experience.done

    self.priorities.update(self.ptr, initial_priority)
    self.ptr = (self.ptr + 1) % self.max_size
    self.size = min(self.size + 1, self.max_size)
```

---

## 6. Importance Sampling Correction

### Why Correction is Needed

Priority sampling oversamples high-priority experiences, introducing bias in the gradient
estimate. Importance sampling weights correct this bias:

```
w_i = (1 / (N * P(i)))^beta
```

Where beta (priority_correction) anneals from an initial value toward 1.0 over training:

```
beta_t = beta_0 + (1 - beta_0) * t / T_total
```

### Weight Normalization

To prevent large IS weights from causing gradient instability, weights are normalized:

```
w_i_normalized = w_i / max_j(w_j)
```

This ensures the maximum weight is 1.0 and all other weights are in (0, 1].

### Application to Gradient Updates

The IS-corrected loss for a replay batch is:

```
L_replay = (1/B) * sum_i(w_i * L(x_i, y_i))
```

Where L(x_i, y_i) is the per-sample loss and w_i is the normalized IS weight.

---

## 7. Temporal Compression Strategies

### Strategy 1: Uniform Stride

Simplest approach. Select every r-th timestep:

```
indices = range(0, T, stride)  where stride = int(ratio)
```

Pros: deterministic, fast, no additional computation.
Cons: may miss critical transitions.

### Strategy 2: Importance-Weighted Selection

Use per-timestep importance scores to select the most informative timesteps:

```
importance[t] = |reward_delta[t]| + lambda * |state_delta[t]|
```

Pros: preserves key transitions.
Cons: requires pre-computing importance scores.

### Strategy 3: Adaptive Compression

Vary compression ratio within a sequence based on local information density:

```
local_ratio[t] = base_ratio * (1 + information_density[t])
```

Regions with high information density (many state changes, reward transitions) get lower
compression (more samples retained), while low-density regions get higher compression.

### Recommended Default

Strategy 2 (importance-weighted selection) with uniform stride fallback when importance
scores are unavailable. This balances fidelity with compute efficiency.

---

## 8. Integration with NREM Phase

### Replay Loop

The NREM phase executes the following loop:

```python
def nrem_phase(model, replay_buffer, cfg):
    model.train(False)  # freeze batch norm, disable dropout for replay
    # ... but enable gradient computation for replay training

    for step in range(cfg.nrem_replay_steps):
        # 1. Sample prioritized batch
        batch, indices, is_weights = replay_scheduler.sample(
            replay_buffer, cfg.batch_size
        )

        # 2. Compress sequences
        compressed = replay_scheduler.compress(batch, cfg.compression_ratio)

        # 3. Forward pass
        output = model(compressed.observations)
        loss = compute_replay_loss(output, compressed, is_weights)

        # 4. Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 5. Update priorities with new TD-errors
        with torch.no_grad():
            td_errors = compute_td_errors(model, compressed)
        replay_scheduler.update_priorities(indices, td_errors)

    return PhaseResult(loss=loss.item(), steps=cfg.nrem_replay_steps)
```

### Learning Rate Considerations

Replay training typically uses a reduced learning rate compared to wake-phase training
(e.g., 0.1x to 0.5x of wake LR). This prevents replay from overwriting recent learning
while still enabling consolidation. The exact ratio is configurable via SleepConfig.

---

## Appendix A: Troubleshooting

| Issue | Cause | Resolution |
|---|---|---|
| All priorities equal | TD-errors not being updated | Call update_priorities after each replay batch |
| Sampling always returns same indices | Priority exponent too high with degenerate priorities | Check for NaN in priorities; reduce alpha |
| Compressed sequences lose reward signal | Compression ratio too high | Reduce ratio or use importance-weighted compression |
| Replay gradients explode | IS weights too large | Ensure weight normalization; clip IS weights |
| Buffer OOM | max_size too large for available memory | Reduce max_size; use memory-mapped storage |
| Slow sampling | Linear scan instead of sum tree | Verify sum tree implementation is used |
