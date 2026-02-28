# Experience Replay Strategies for Continual Learning

## 1. Overview

Experience replay maintains a fixed-size memory buffer of exemplars from previously learned tasks. During training on a new task, a fraction of each mini-batch is drawn from the buffer, forcing the model to maintain performance on old data while learning new patterns. This is the most intuitive defence against catastrophic forgetting: simply keep practicing on old examples.

The key design decisions are:
- **What to store**: raw (x, y) pairs, feature embeddings, or generated pseudo-samples
- **How to select**: reservoir sampling (uniform), prioritized (loss-based), or class-balanced
- **How to mix**: ratio of replay vs current-task data in each batch
- **When to update**: after every step, after every epoch, or at task boundaries

---

## 2. Reservoir Sampling (Algorithm R)

### Algorithm

Reservoir sampling (Vitter, 1985) maintains a buffer of size K from a stream of N items such that every item seen so far has an equal probability K/N of being in the buffer.

```
def reservoir_add(buffer, buffer_size, item, n_seen):
    """Add an item to the reservoir buffer.

    Args:
        buffer: List of stored items (max length = buffer_size)
        buffer_size: Maximum buffer capacity
        item: New item to potentially add
        n_seen: Total number of items seen so far (including this one)

    Returns:
        True if item was added, False if rejected
    """
    if len(buffer) < buffer_size:
        buffer.append(item)
        return True
    else:
        # Replace a random existing item with probability buffer_size / n_seen
        j = random.randint(0, n_seen - 1)
        if j < buffer_size:
            buffer[j] = item
            return True
        return False
```

### Properties

- **Uniform coverage**: after seeing N items, each item in the buffer was selected with probability K/N. For continual learning with T tasks of equal size, each task has approximately K/T exemplars.
- **O(1) per item**: the decision to keep or reject an item requires only a single random number.
- **No task labels required**: the algorithm works in a task-free streaming setting.
- **Memory**: exactly K items, regardless of how many items have been seen.

### Task Balance Analysis

For 5 tasks each contributing 1000 samples to a buffer of size 500:
- After task 1: 500 items from task 1 (100%)
- After task 2: ~250 from task 1, ~250 from task 2 (50% each)
- After task 5: ~100 from each task (20% each)

The distribution naturally balances across tasks over time.

---

## 3. Prioritized Replay

### Motivation

Not all exemplars are equally valuable for preventing forgetting. Prioritized replay assigns higher sampling probability to exemplars that the current model finds difficult (high loss) or uncertain (high entropy).

### Priority Computation

For each exemplar (x_i, y_i) in the buffer, compute a priority score:

    p_i = |L(x_i, y_i)|^alpha + epsilon

where L is the loss, alpha controls how much prioritization matters (alpha=0 is uniform, alpha=1 is fully proportional), and epsilon is a small constant ensuring non-zero probability.

The sampling probability is:

    P(i) = p_i / sum_j p_j

### Implementation

```python
class PrioritizedReplayBuffer:
    def __init__(self, max_size, alpha=0.6, epsilon=1e-6):
        self.max_size = max_size
        self.alpha = alpha
        self.epsilon = epsilon
        self.buffer = []       # (x, y, task_id) tuples
        self.priorities = []   # Priority scores

    def add(self, x, y, task_id, priority=1.0):
        if len(self.buffer) < self.max_size:
            self.buffer.append((x, y, task_id))
            self.priorities.append(priority)
        else:
            # Replace lowest-priority item
            min_idx = self.priorities.index(min(self.priorities))
            self.buffer[min_idx] = (x, y, task_id)
            self.priorities[min_idx] = priority

    def sample(self, batch_size):
        probs = np.array(self.priorities) ** self.alpha + self.epsilon
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), size=batch_size, p=probs)
        return [self.buffer[i] for i in indices]

    def update_priorities(self, indices, new_priorities):
        for idx, p in zip(indices, new_priorities):
            self.priorities[idx] = p
```

### Trade-offs

- **Pro**: focuses rehearsal on difficult examples, potentially more sample-efficient
- **Con**: requires periodic priority recomputation (model changes -> priorities change)
- **Con**: biased toward outliers; may overfit to unusual examples
- **Mitigation**: use importance sampling weights to correct for non-uniform sampling

---

## 4. Generative Replay (Pseudo-Rehearsal)

### Concept

Instead of storing real exemplars, train a generative model (VAE, GAN) alongside the main model. When learning a new task, generate pseudo-exemplars from previous tasks using the generator:

    x_pseudo ~ G(z), z ~ N(0, I)
    y_pseudo = Teacher(x_pseudo)

The main model trains on both real current-task data and generated pseudo-data.

### Architecture

```
Task 1: Train Main Model M1 + Generator G1 on task 1 data
Task 2:
  - Generate pseudo-data: x_pseudo ~ G1(z)
  - Label pseudo-data: y_pseudo = M1(x_pseudo)
  - Train M2 on (task 2 data) + (pseudo-data with pseudo-labels)
  - Train G2 on (task 2 data) + (pseudo-data from G1)
Task 3:
  - Generate pseudo-data: x_pseudo ~ G2(z)
  - [G2 has learned to generate data from tasks 1 and 2]
  - ...
```

### Advantages

- **No raw data storage**: privacy-preserving; suitable for medical/financial data
- **Unlimited replay**: can generate as many pseudo-samples as needed
- **Scales with task count**: generator capacity is fixed (does not grow with T)

### Disadvantages

- **Generator quality**: if the generator is poor, pseudo-data harms rather than helps
- **Training overhead**: must train a generator alongside the main model
- **Mode collapse**: GAN generators may forget modes from old tasks (the generator itself suffers from catastrophic forgetting)
- **Not suitable for all domains**: works well for images, less proven for structured data

### When to Use

Generative replay is most appropriate when:
1. Storing raw exemplars is prohibited (privacy constraints)
2. The data domain is well-suited to generative modeling (images, audio)
3. Model capacity for the generator is available

For most brain_ai applications, reservoir sampling is simpler and more reliable.

---

## 5. Class-Balanced Replay

### Problem

Reservoir sampling produces approximately uniform task coverage, but within each task, some classes may be over- or under-represented. For imbalanced datasets, this creates a biased buffer.

### Solution

Maintain explicit per-class quotas:

```python
class ClassBalancedBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.class_buffers = {}  # class_id -> list of (x, y)

    def add(self, x, y):
        class_id = y.item()
        if class_id not in self.class_buffers:
            self.class_buffers[class_id] = []

        quota = self.max_size // len(self.class_buffers)

        if len(self.class_buffers[class_id]) < quota:
            self.class_buffers[class_id].append((x, y))
        else:
            # Reservoir sampling within this class
            n = len(self.class_buffers[class_id])
            j = random.randint(0, n)
            if j < quota:
                self.class_buffers[class_id][j] = (x, y)

        # Rebalance if a new class was added
        self._rebalance()

    def _rebalance(self):
        num_classes = len(self.class_buffers)
        quota = self.max_size // num_classes
        for cid in self.class_buffers:
            if len(self.class_buffers[cid]) > quota:
                self.class_buffers[cid] = random.sample(
                    self.class_buffers[cid], quota
                )
```

---

## 6. Replay Mixing Strategies

### Fixed Ratio

Mix current-task and replay data at a fixed ratio:

    batch = concat(current_task_samples[:B//2], replay_samples[:B//2])

Typical ratio: 50% current, 50% replay. Adjust based on forgetting severity.

### Adaptive Ratio

Increase replay ratio when forgetting is detected:

    replay_ratio = base_ratio + forgetting_signal * sensitivity

where `forgetting_signal` is derived from validation accuracy on past tasks.

### Separate Losses

Instead of mixing in the same batch, compute separate losses and combine:

    L_total = L_current + beta * L_replay

This allows different loss weighting without changing batch composition.

---

## 7. Buffer Management at Task Boundaries

When a task boundary is detected (or manually triggered):

1. **Snapshot current data**: add representative samples from the ending task to the buffer
2. **Update priorities** (if using prioritized replay): recompute priorities for all buffered samples using the current model
3. **Rebalance** (if using class-balanced): adjust per-class quotas for the new total number of classes

The `ReplayBuffer.on_task_boundary(task_id)` method handles all three steps.

---

## 8. References

- Vitter, J. S. (1985). "Random sampling with a reservoir." ACM TOMS, 11(1), 37-57.
- Chaudhry, A., et al. (2019). "Tiny Episodic Memories in Continual Learning." arXiv:1902.10486.
- Shin, H., et al. (2017). "Continual Learning with Deep Generative Replay." NeurIPS 2017.
- Aljundi, R., et al. (2019). "Gradient based sample selection for online continual learning." NeurIPS 2019.
- Buzzega, P., et al. (2020). "Dark Experience for General Continual Learning." NeurIPS 2020.
- Schaul, T., et al. (2016). "Prioritized Experience Replay." ICLR 2016.
