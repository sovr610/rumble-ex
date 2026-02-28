---
name: Continual Learning Guard (EWC / SI / Progressive Networks / Replay)
description: >-
  This skill should be used when the user asks to "prevent catastrophic forgetting",
  "elastic weight consolidation", "EWC regularization", "progressive networks",
  "continual learning strategy", "knowledge distillation for retention",
  "replay buffer memory", "task boundary detection", "fisher information matrix",
  "synaptic intelligence", "PackNet pruning", "memory-aware synapses",
  "add continual learning guard", "implement EWC penalty", "add experience replay",
  "implement progressive columns", "add Fisher diagonal computation",
  "implement reservoir sampling", "add knowledge distillation loss",
  "implement task-free continual learning", "add online EWC",
  "implement generative replay", "add PackNet iterative pruning",
  "implement synaptic intelligence path integral",
  or mentions catastrophic forgetting, continual learning, lifelong learning,
  sequential task training, knowledge retention, task interference,
  Fisher information regularization, or Phase 8 continual-learning pipeline
  in the cognitive architecture.
version: 0.1.0
---

# Continual Learning Guard (EWC / SI / Progressive Networks / Replay)

## Purpose

This skill standardizes the "remember-while-learning" stack (Phase 8): a modular
continual-learning defence layer that prevents catastrophic forgetting when the
brain_ai system acquires new tasks sequentially. The non-negotiable goals are
measurable retention of prior-task performance and clean task-boundary management
across all six supported methods: Elastic Weight Consolidation (EWC), Synaptic
Intelligence (SI), Progressive Networks, Experience Replay, Knowledge Distillation,
and PackNet.

## Key Files

| Target Module | Template Asset | Purpose |
|---|---|---|
| `brain_ai/continual/ewc.py` | `assets/ewc_regularizer_template.py` | EWC with Fisher diagonal, quadratic penalty, online EWC |
| `brain_ai/continual/replay.py` | `assets/replay_buffer_template.py` | ReplayBuffer with reservoir sampling, priority queue |
| `brain_ai/continual/progressive.py` | `assets/progressive_net_template.py` | ProgressiveNet with lateral connections, column freezing |
| `brain_ai/continual/boundary.py` | `assets/task_boundary_template.py` | TaskBoundaryDetector with loss/gradient/distribution monitors |
| `brain_ai/continual/learner.py` | `assets/continual_learner_template.py` | ContinualLearner orchestrating all CL methods |
| `brain_ai/config.py` (extend) | `assets/cl_config_template.py` | ContinualLearningConfig dataclass hierarchy |

## Public Contract

```python
ewc_penalty(model, fisher_diag, star_params) -> scalar_loss
replay_sample(buffer, batch_size) -> (x, y)
progressive_forward(columns, task_id, x) -> logits
detect_boundary(monitor, metrics_window) -> bool
continual_step(learner, batch, task_id, *, config) -> ContinualOutput
```

`model` is a standard `nn.Module`. `fisher_diag` is a dict of diagonal Fisher
information values keyed by parameter name. `star_params` is a snapshot of
optimal parameters from previous tasks. `buffer` holds exemplars from prior
tasks via reservoir sampling or priority queue. `columns` is a list of frozen
prior-task networks plus the active column with lateral adapters.

## ContinualOutput Contract

| Field | Shape / Type | Description |
|---|---|---|
| `loss` | scalar | Combined task loss + CL regularization terms |
| `task_loss` | scalar | Raw cross-entropy / task-specific loss |
| `cl_penalty` | scalar | Sum of all CL regularization penalties |
| `metrics` | `Dict[str, float]` | `forgetting`, `bwt`, `fwt`, `avg_acc`, `replay_ratio` |
| `boundary_detected` | `bool` | Whether a task boundary was detected this step |

**Hard invariants**:
- EWC penalty is always non-negative (quadratic form with positive-semi-definite Fisher).
- Replay buffer never exceeds `max_size` samples; reservoir sampling maintains uniform coverage.
- Progressive network columns for completed tasks are fully frozen (zero gradient).
- Fisher diagonal values are computed with `module.train(False)`, not `.eval()`.

## Elastic Weight Consolidation (EWC)

The core regularization method. After completing task t, compute the diagonal of the
Fisher information matrix F_t over task-t data, and store the optimal parameters
theta*_t. For subsequent tasks, add a quadratic penalty:

    L_ewc = (lambda / 2) * sum_i F_t,i * (theta_i - theta*_t,i)^2

**Online EWC** (Schwarz et al. 2018) maintains a running Fisher diagonal instead of
storing per-task Fishers, using an exponential moving average:

    F_online = gamma * F_online + F_new

This bounds memory at O(|theta|) regardless of task count.

See `references/ewc-theory.md` for full derivation, multi-task extension, and lambda tuning.

## Synaptic Intelligence (SI)

An online alternative to EWC that tracks parameter importance during training
rather than computing it post-hoc. Each parameter accumulates an "importance
score" omega_i via path integral over the loss surface:

    omega_i = sum_t ( Delta_i(t) / (delta_i(t)^2 + xi) )

where Delta_i(t) is the total loss decrease attributable to parameter i during
task t, delta_i(t) is the total parameter change, and xi is a damping constant.
The penalty mirrors EWC: L_si = (c / 2) * sum_i omega_i * (theta_i - theta*_i)^2.

## Progressive Networks

Prevent forgetting by design: freeze all prior-task columns and add a new column
for each new task. Lateral adapter connections allow forward transfer from frozen
columns to the active column. No backward interference is possible because frozen
parameters receive zero gradient.

Trade-off: linear growth in parameters with task count. Suitable when task count
is bounded (< 20 tasks) and model capacity is available.

See `references/progressive-architectures.md` for lateral connection design, PackNet
alternative, and DEN.

## Experience Replay

Maintain a fixed-size buffer of exemplars from prior tasks. At each training step,
mix current-task data with replayed samples. Two variants:

- **Reservoir Sampling** (Algorithm R): uniform probability of retaining any sample
  seen so far. Memory-optimal for streaming; O(1) per sample decision.
- **Prioritized Replay**: weight samples by loss magnitude or uncertainty. Higher-loss
  samples are replayed more frequently to reinforce difficult examples.
- **Generative Replay** (pseudo-rehearsal): train a generative model to produce
  synthetic exemplars instead of storing real data. Privacy-preserving but adds
  model complexity.

See `references/replay-strategies.md` for buffer design and sampling algorithms.

## Knowledge Distillation

Use a frozen copy of the model after task t as a "teacher" and the current model as
a "student". The distillation loss penalizes divergence between teacher and student
soft outputs (logits / softmax with temperature T):

    L_kd = alpha * KL(softmax(z_teacher / T) || softmax(z_student / T))

This prevents the student from drifting too far from prior-task predictions.

## PackNet (Iterative Pruning + Freezing)

After training on each task, prune the network to identify a sparse subnetwork
sufficient for the task. Freeze those weights. The remaining (pruned) weights are
available for future tasks. This provides zero forgetting within the frozen
subnetwork while reusing a single model.

See `references/progressive-architectures.md` for PackNet pruning schedule and mask
management.

## Configuration Surface

### EWCConfig

| Field | Default | Purpose |
|---|---|---|
| `ewc_lambda` | 1000.0 | Regularization strength |
| `fisher_samples` | 200 | Samples for Fisher diagonal estimation |
| `online_ewc` | True | Use online (running) Fisher vs per-task |
| `online_gamma` | 0.95 | EMA decay for online Fisher |
| `normalize_fisher` | True | Normalize Fisher diagonal to unit max |

### SIConfig

| Field | Default | Purpose |
|---|---|---|
| `si_c` | 1.0 | SI regularization strength |
| `si_xi` | 0.1 | Damping constant for omega computation |

### ProgressiveConfig

| Field | Default | Purpose |
|---|---|---|
| `lateral_dim` | 64 | Lateral adapter hidden dimension |
| `max_columns` | 10 | Maximum number of task columns |
| `freeze_bn` | True | Freeze BatchNorm in frozen columns |

### ReplayConfig

| Field | Default | Purpose |
|---|---|---|
| `buffer_size` | 5000 | Maximum exemplar buffer size |
| `strategy` | `"reservoir"` | `"reservoir"`, `"priority"`, `"generative"` |
| `replay_batch_ratio` | 0.5 | Fraction of batch from replay buffer |
| `priority_alpha` | 0.6 | Priority exponent for prioritized replay |

### DistillationConfig

| Field | Default | Purpose |
|---|---|---|
| `distill_alpha` | 0.5 | Distillation loss weight |
| `distill_temperature` | 2.0 | Softmax temperature for soft targets |

### PackNetConfig

| Field | Default | Purpose |
|---|---|---|
| `prune_ratio` | 0.75 | Fraction of weights to prune per task |
| `prune_method` | `"magnitude"` | `"magnitude"`, `"random"` |

### BoundaryConfig

| Field | Default | Purpose |
|---|---|---|
| `detection_method` | `"loss_spike"` | `"loss_spike"`, `"grad_norm"`, `"distribution"`, `"manual"` |
| `loss_spike_threshold` | 2.0 | Relative loss increase to trigger boundary |
| `grad_norm_threshold` | 5.0 | Gradient norm spike multiplier |
| `window_size` | 50 | Sliding window for metric monitoring |

Presets: `ContinualLearningConfig.minimal()`, `.dev()`, `.production()`.

## Done-When Gates

| Gate | Test | Threshold |
|---|---|---|
| **(a) EWC retention** | Train 2-task sequence (split MNIST), measure Task-1 accuracy after Task-2 training with EWC vs without. | EWC retains >= 85% Task-1 accuracy; baseline drops below 50% |
| **(b) Replay coverage** | Fill 1000-sample buffer with 5 tasks (200 each), verify uniform task coverage via reservoir sampling. | Each task has 150--250 samples; chi-squared p > 0.05 |
| **(c) Progressive isolation** | Train 3-column progressive network; verify frozen columns have exactly zero gradient after backward pass. | All frozen param grads == 0 |
| **(d) Boundary detection** | Inject a distribution shift (switch task labels); verify boundary detector fires within 10 steps. | Detection latency <= 10 steps |
| **(e) Integration** | Run ContinualLearner on 3-task split MNIST: compute average accuracy, backward transfer (BWT), and forgetting metrics. | avg_acc >= 80%, BWT > -0.15 |

## Common Failure Modes

| Symptom | Cause | Fix |
|---|---|---|
| EWC penalty explodes | Lambda too high or un-normalized Fisher | Normalize Fisher to unit max; reduce lambda |
| EWC has no effect | Lambda too low or Fisher computed on wrong data | Increase lambda; verify Fisher uses task-t data with train(False) |
| Replay buffer skewed | Non-uniform sampling or broken reservoir logic | Verify Algorithm R: P(keep) = buffer_size / n_seen |
| Progressive OOM | Too many columns for available memory | Set max_columns; switch to PackNet |
| SI omega all zeros | Forgot to accumulate path integral during training | Call si.update_running_importance() at each step |
| Distillation diverges | Temperature too low (hard targets = no smoothing) | Use T >= 2.0; check teacher is frozen |
| PackNet masks overlap | Mask management bug across tasks | Verify masks are disjoint: mask_t AND mask_s == 0 for t != s |
| Boundary detector fires every step | Threshold too sensitive | Increase window_size; raise spike threshold |
| Fisher computed in train mode | Dropout / BN noise corrupts Fisher | Always use model.train(False) for Fisher computation |

## Anti-Patterns

- **Computing Fisher in training mode** -- dropout and BN noise corrupt the diagonal; always `model.train(False)`
- **Storing full Fisher matrix** -- O(|theta|^2) memory; use diagonal approximation
- **Single EWC snapshot** -- online EWC with EMA is strictly superior for > 2 tasks
- **Replay without task balancing** -- reservoir sampling naturally balances; priority queue needs per-task quotas
- **Unfreezing progressive columns** -- defeats the purpose; frozen means zero gradient, always
- **PackNet without disjoint mask check** -- overlapping masks corrupt prior-task subnetworks
- **Ignoring boundary detection in task-free settings** -- real-world streams lack explicit task labels
- **Using .eval() instead of .train(False)** -- both are equivalent but .train(False) is explicit and grep-friendly

## Additional Resources

### Reference Files

- **`references/ewc-theory.md`** -- Fisher information diagonal, lambda tuning, online EWC, multi-task extension
- **`references/replay-strategies.md`** -- Reservoir sampling, prioritized replay, generative replay, buffer management
- **`references/progressive-architectures.md`** -- Progressive Networks, PackNet, DEN, lateral adapters
- **`references/task-boundaries.md`** -- Task boundary detection: loss spike, gradient norm, distribution shift, task-free CL
- **`references/testing-matrix.md`** -- All test scenarios for EWC, SI, Progressive, Replay, Distillation, PackNet, Boundary, Integration

### Asset Templates

- **`assets/ewc_regularizer_template.py`** -- EWCRegularizer: Fisher diagonal, quadratic penalty, online EWC, self-test
- **`assets/replay_buffer_template.py`** -- ReplayBuffer: reservoir sampling, priority queue, generative replay stub, self-test
- **`assets/progressive_net_template.py`** -- ProgressiveNet: lateral connections, column freezing, forward transfer, self-test
- **`assets/task_boundary_template.py`** -- TaskBoundaryDetector: loss spike, gradient norm, distribution shift, self-test
- **`assets/continual_learner_template.py`** -- ContinualLearner: orchestrate EWC + Replay + Progressive + Distillation, self-test
- **`assets/cl_config_template.py`** -- All configs, presets, serialization, self-test

### Scripts

- **`scripts/validate_continual_learning.py`** -- Runtime contract validation (EWC retention, replay coverage, progressive isolation, boundary detection)
- **`scripts/gen_cl_tests.py`** -- Generates `tests/test_continual_learning.py` (~70+ test cases)
- **`scripts/forgetting_benchmark.py`** -- Benchmark forgetting on permuted MNIST sequence (5 permutations, measure BWT/FWT/avg_acc)
