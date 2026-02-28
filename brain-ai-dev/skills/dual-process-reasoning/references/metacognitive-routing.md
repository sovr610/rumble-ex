# Metacognitive Routing Policy

Reference document for the metacognitive router within Skill #9 (Dual-Process Reasoning).
Covers the routing decision pipeline, route score computation, novelty scoring methods,
determinism requirements, batch-level independence guarantees, budget management, and
integration code examples.

---

## Table of Contents

1. [MetacognitiveRouter Overview](#1-metacognitiverouter-overview)
2. [Routing Inputs](#2-routing-inputs)
3. [Route Score Computation](#3-route-score-computation)
4. [Routing Decision](#4-routing-decision)
5. [Steps Budget Allocation](#5-steps-budget-allocation)
6. [Novelty Scoring](#6-novelty-scoring)
7. [Determinism Requirements (Gate A)](#7-determinism-requirements-gate-a)
8. [Batch-Level Independence](#8-batch-level-independence)
9. [Budget Policy](#9-budget-policy)
10. [Code Examples](#10-code-examples)

Appendices:
- [A. Config Reference Table](#appendix-a-config-reference-table)
- [B. ASCII Decision Flow Diagram](#appendix-b-ascii-decision-flow-diagram)
- [C. Tuning Scenarios](#appendix-c-tuning-scenarios)
- [D. Integration Checklist](#appendix-d-integration-checklist)

---

## 1. MetacognitiveRouter Overview

### Purpose

The `MetacognitiveRouter` is the gating module that decides, for each item in a batch,
whether to accept the System 1 fast prediction or to escalate to System 2 deliberative
reasoning. It also determines how many reasoning steps System 2 receives when invoked.
The router sits between the System 1 output and the System 2 entry point in the
dual-process pipeline.

The router answers two questions per input:

1. **Route**: Send this item to System 1 (accept fast prediction) or System 2 (invoke
   slow deliberation)?
2. **Steps**: If System 2, how many iterative refinement steps does it receive?

Both decisions must be deterministic, per-item independent, and budget-aware.

### Module Signature

```python
class MetacognitiveRouter(nn.Module):
    """
    Decide per-item routing between System 1 (fast) and System 2 (slow).

    Args:
        config: DualProcessConfig containing routing hyperparameters.
    """

    def __init__(self, config: DualProcessConfig):
        super().__init__()
        self.config = config
        self.budget_tracker = BudgetTracker(config.total_s2_budget)
        self.novelty_scorer = NoveltyScorer(config.novelty_method, config.workspace_dim)

    def forward(
        self,
        calibrated_conf: torch.Tensor,   # (B,)
        novelty: torch.Tensor,            # (B,)
        anomaly: torch.Tensor,            # (B,)
        ignition: torch.Tensor,           # (B,)
        remaining_budget: float,          # scalar
    ) -> RoutingDecision:
        ...
```

### RoutingDecision Dataclass

```python
@dataclass
class RoutingDecision:
    """Output of the metacognitive router for a single forward pass."""

    route_scores: torch.Tensor        # (B,) float in [0, 1]
    use_s2: torch.BoolTensor          # (B,) True = send to System 2
    allocated_steps: torch.LongTensor  # (B,) int in [0, max_steps]
    budget_remaining: float            # scalar, updated after allocation
    debug_info: dict                   # optional diagnostics dict
```

Fields:

| Field | Dtype | Shape | Description |
|---|---|---|---|
| `route_scores` | float32 | `(B,)` | Composite urgency score per item, clamped to [0, 1] |
| `use_s2` | bool | `(B,)` | Whether this item is routed to System 2 |
| `allocated_steps` | int64 | `(B,)` | Number of System 2 GRU iterations allocated; 0 for S1 items |
| `budget_remaining` | float | scalar | Budget remaining after this batch's allocation |
| `debug_info` | dict | -- | Keys: `raw_scores`, `penalty`, `threshold_used`, `hard_skipped` |

### Design Principles

Adhere to these three invariants in every code path:

1. **Deterministic**: Given identical inputs and identical state, produce identical outputs.
   No sampling, no stochastic gating, no dropout in the routing path. See Section 7.

2. **Per-item independent**: The routing decision for item `i` must not depend on item `j`
   in the same batch. No cross-batch attention, no batch-level normalization, no sorting
   along batch dimension. See Section 8.

3. **No sampling**: Never draw from a Bernoulli or Gumbel-Softmax to decide routing.
   Use hard threshold comparison only. Probabilistic routing destroys reproducibility
   and complicates debugging.

---

## 2. Routing Inputs

The router consumes five signals. Four are per-item tensors and one is a global scalar.

| Input | Shape | Range | Source | Semantics |
|---|---|---|---|---|
| `calibrated_conf` | `(B,)` | [0, 1] | System 1 confidence head after temperature scaling | How confident S1 is in its own prediction. High = S1 is probably right. |
| `novelty` | `(B,)` | [0, 1] | `NoveltyScorer` (see Section 6) | How different this input is from training distribution. High = unfamiliar. |
| `anomaly` | `(B,)` | [0, 1] | HTM temporal memory anomaly score | How surprising the current timestep is given the learned temporal context. |
| `ignition` | `(B,)` | [0, 1] | Global workspace ignition strength | How strongly this input broadcast in the workspace competition. Low = weak representation. |
| `remaining_budget` | scalar | [0, total] | `BudgetTracker` | Number of System 2 invocations remaining in the current episode/epoch. |

### Input Validation

Validate inputs at the top of `forward()`. Fail fast on shape mismatches or out-of-range
values.

```python
def _validate_inputs(self, calibrated_conf, novelty, anomaly, ignition, remaining_budget):
    B = calibrated_conf.shape[0]
    assert calibrated_conf.shape == (B,), f"Expected (B,), got {calibrated_conf.shape}"
    assert novelty.shape == (B,), f"Expected (B,), got {novelty.shape}"
    assert anomaly.shape == (B,), f"Expected (B,), got {anomaly.shape}"
    assert ignition.shape == (B,), f"Expected (B,), got {ignition.shape}"

    for name, tensor in [("calibrated_conf", calibrated_conf),
                         ("novelty", novelty),
                         ("anomaly", anomaly),
                         ("ignition", ignition)]:
        assert tensor.min() >= 0.0 and tensor.max() <= 1.0, (
            f"{name} out of [0,1]: min={tensor.min().item()}, max={tensor.max().item()}"
        )

    assert remaining_budget >= 0, f"Budget cannot be negative: {remaining_budget}"
```

### Signal Provenance

Where each input originates in the pipeline:

- **calibrated_conf**: The System 1 confidence head outputs raw logits. These pass through
  temperature scaling (see `system1-confidence.md` Section 3) to yield calibrated
  probabilities. The maximum class probability is the scalar confidence.
- **novelty**: Computed by the `NoveltyScorer` module from the workspace representation.
  Three methods are available; see Section 6.
- **anomaly**: The HTM `TemporalMemory` produces a per-timestep anomaly score. If HTM is
  disabled (`use_htm=False`), default to 0.0 (never contributes).
- **ignition**: The global workspace `Competition` module computes ignition strength as
  the winning coalition's aggregate activation. If workspace is disabled, default to 1.0
  (never contributes negatively).
- **remaining_budget**: Managed by the `BudgetTracker` class. Reset at episode or epoch
  boundaries depending on configuration.

---

## 3. Route Score Computation

### Formula

Compute a composite route score for each item in the batch. Higher score means stronger
case for invoking System 2.

```
route_score = w_conf * (1 - conf) + w_novelty * novelty + w_anomaly * anomaly
              + w_ignition * (1 - ignition) - w_budget * penalty
```

where:

- `conf` = `calibrated_conf` -- inverted because low confidence argues for S2
- `novelty` = raw novelty score
- `anomaly` = HTM anomaly score
- `ignition` = workspace ignition -- inverted because weak ignition argues for S2
- `penalty` = budget penalty term (see below)
- `w_*` = learned or fixed weight coefficients

### Budget Penalty

The budget penalty discourages routing to System 2 as the budget depletes:

```
penalty = max(0, 1 - remaining_budget / total_budget)
```

When the budget is full (`remaining == total`), the penalty is 0. When the budget is
exhausted (`remaining == 0`), the penalty is 1. The penalty grows linearly as budget
decreases.

### Clamping

Clamp the final route score to `[0, 1]`:

```python
route_score = torch.clamp(route_score, 0.0, 1.0)
```

This ensures downstream code can treat the score as a pseudo-probability without
additional normalization.

### Weight Coefficients

Default weight values and their rationale:

| Weight | Default | Rationale |
|---|---|---|
| `w_conf` | 0.40 | Confidence is the primary routing signal |
| `w_novelty` | 0.25 | Novelty is the secondary signal; novel inputs need deliberation |
| `w_anomaly` | 0.20 | Anomaly is correlated with novelty but independent (temporal) |
| `w_ignition` | 0.10 | Ignition is a weaker signal; mainly catches workspace failures |
| `w_budget` | 0.15 | Budget penalty should temper but not dominate routing decisions |

Weights do not need to sum to 1.0. The clamping step absorbs any overshoot. Adjust
weights via `DualProcessConfig.routing_weights`.

### Vectorized Implementation

```python
def _compute_route_scores(
    self,
    conf: torch.Tensor,       # (B,)
    novelty: torch.Tensor,    # (B,)
    anomaly: torch.Tensor,    # (B,)
    ignition: torch.Tensor,   # (B,)
    remaining_budget: float,
) -> torch.Tensor:
    """Compute per-item route scores. Returns (B,) float tensor in [0, 1]."""
    w = self.config.routing_weights  # namespace with w_conf, w_novelty, etc.

    penalty = max(0.0, 1.0 - remaining_budget / self.config.total_s2_budget)

    score = (
        w.w_conf * (1.0 - conf)
        + w.w_novelty * novelty
        + w.w_anomaly * anomaly
        + w.w_ignition * (1.0 - ignition)
        - w.w_budget * penalty
    )

    return torch.clamp(score, 0.0, 1.0)
```

### Gradient Behavior

The route score computation is differentiable with respect to the input signals, but
the downstream threshold comparison (`score >= threshold`) is not differentiable. See
Section 9 for why the budget path is intentionally non-differentiable. If you need
gradient flow through the routing decision (for end-to-end training of the confidence
head), use the straight-through estimator on the threshold comparison:

```python
# Straight-through: forward uses hard threshold, backward uses identity
use_s2_hard = (route_score >= threshold).float()
use_s2_st = route_score + (use_s2_hard - route_score).detach()
```

---

## 4. Routing Decision

### Threshold Comparison

Compare each route score against the routing threshold to produce a boolean mask:

```python
use_s2 = route_score >= self.config.s2_threshold  # (B,) bool
```

Default threshold: `0.5`. Raise the threshold to route fewer items to S2 (more
conservative). Lower it to route more items to S2 (more aggressive).

### Hard Skip Override

Items with very high System 1 confidence bypass the route score entirely. This is a
safety rail: if S1 is almost certain, never waste budget on S2 regardless of other
signals.

```python
hard_skip_mask = calibrated_conf >= self.config.min_conf_to_skip  # (B,) bool
use_s2 = use_s2 & ~hard_skip_mask
```

Default `min_conf_to_skip`: `0.95`. Set to `1.0` to disable hard skip.

### Debug Override

For debugging and ablation, force all items through System 2:

```python
if self.config.always_run_s2:
    use_s2 = torch.ones_like(use_s2, dtype=torch.bool)
```

This flag is off by default. Enable it only for analysis or when profiling System 2
throughput. It ignores budget constraints entirely.

### Decision Assembly

Combine the above into the final `RoutingDecision`:

```python
def _make_decision(
    self,
    route_scores: torch.Tensor,
    calibrated_conf: torch.Tensor,
    remaining_budget: float,
) -> RoutingDecision:
    # Threshold
    use_s2 = route_scores >= self.config.s2_threshold

    # Hard skip
    hard_skip = calibrated_conf >= self.config.min_conf_to_skip
    use_s2 = use_s2 & ~hard_skip

    # Debug override
    if self.config.always_run_s2:
        use_s2 = torch.ones_like(use_s2, dtype=torch.bool)

    # Allocate steps
    allocated_steps = self._allocate_steps(route_scores, use_s2)

    # Update budget
    n_s2 = use_s2.sum().item()
    new_budget = max(0.0, remaining_budget - n_s2)

    return RoutingDecision(
        route_scores=route_scores,
        use_s2=use_s2,
        allocated_steps=allocated_steps,
        budget_remaining=new_budget,
        debug_info={
            "raw_scores": route_scores.detach(),
            "hard_skipped": hard_skip.sum().item(),
            "threshold_used": self.config.s2_threshold,
            "penalty": max(0.0, 1.0 - remaining_budget / self.config.total_s2_budget),
        },
    )
```

### Decision Priority Order

Apply overrides in this exact sequence:

1. Compute `route_score` (Section 3).
2. Apply threshold comparison: `use_s2 = score >= threshold`.
3. Apply hard skip: clear `use_s2` where `conf >= min_conf_to_skip`.
4. Apply debug override: set all `use_s2 = True` if `always_run_s2`.
5. Check budget: if `remaining_budget <= 0` and not `always_run_s2`, clear all `use_s2`.
6. Allocate steps for items where `use_s2 = True`.

Step 5 is the budget hard cutoff. It fires after all other overrides so that the debug
flag can still bypass budget exhaustion when needed.

---

## 5. Steps Budget Allocation

### Purpose

When an item is routed to System 2, the router must decide how many iterative GRU steps
S2 receives. More steps means deeper deliberation but higher cost. The allocation is
proportional to the route score: items with higher urgency get more steps.

### Formula

```
allocated_steps = clamp(round(base_steps + alpha * route_score), 1, max_steps)
```

where:

- `base_steps` = minimum number of steps every S2 item receives (default: 1)
- `alpha` = scaling factor mapping score to additional steps (default: 4)
- `route_score` = the per-item route score from Section 3
- `max_steps` = hard ceiling on System 2 iterations (default: 8)

For items where `use_s2 = False`, set `allocated_steps = 0`.

### Vectorized Implementation

```python
def _allocate_steps(
    self,
    route_scores: torch.Tensor,  # (B,)
    use_s2: torch.BoolTensor,    # (B,)
) -> torch.LongTensor:
    """Allocate System 2 steps proportional to route score."""
    base = self.config.base_steps     # default: 1
    alpha = self.config.step_alpha    # default: 4
    max_s = self.config.max_s2_steps  # default: 8

    raw = base + alpha * route_scores                     # (B,) float
    clamped = torch.clamp(torch.round(raw), 1, max_s)    # (B,) float
    steps = clamped.long()                                 # (B,) int64

    # Zero out steps for S1 items
    steps = steps * use_s2.long()

    return steps
```

### Worked Examples

Assume defaults: `base_steps=1`, `alpha=4`, `max_steps=8`.

| Item | route_score | use_s2 | raw = 1 + 4*score | round | clamp [1,8] | final steps |
|---|---|---|---|---|---|---|
| A | 0.00 | False | 1.0 | 1 | 1 | 0 (S1) |
| B | 0.50 | True | 3.0 | 3 | 3 | 3 |
| C | 0.72 | True | 3.88 | 4 | 4 | 4 |
| D | 0.95 | True | 4.80 | 5 | 5 | 5 |
| E | 1.00 | True | 5.0 | 5 | 5 | 5 |
| F | 0.30 | False | 2.2 | 2 | 2 | 0 (S1) |
| G | 0.55 | True | 3.2 | 3 | 3 | 3 |
| H | 0.99 | True | 4.96 | 5 | 5 | 5 |

Observe that with default `alpha=4`, the effective range is 1-5 steps. Increase `alpha`
to 7 to reach the full 1-8 range. The `max_steps` clamp prevents runaway computation.

### Cost Accounting

Each allocated step costs one unit of System 2 compute. The budget tracker decrements by
the number of items routed to S2 (not by total steps). This is a deliberate simplification:
the budget tracks *invocations*, not *step-seconds*. If step-level budgeting is required,
replace the invocation counter with a step accumulator:

```python
# Invocation-level (default):
budget_cost = use_s2.sum().item()

# Step-level (alternative):
budget_cost = allocated_steps.sum().item()
```

Document which policy is active in the config.

---

## 6. Novelty Scoring

The novelty scorer estimates how far an input is from the training distribution. Three
methods are supported. Select via `DualProcessConfig.novelty_method`.

### Method 1: Prototype Distance (Default)

Maintain a set of K prototype vectors representing the training distribution. Compute
novelty as the normalized distance from the input to its nearest prototype.

**Initialization**: Run K-means on the first N batches of training data to initialize
prototypes. Default K=64.

**Update rule**: Exponential moving average (EMA) on the nearest prototype:

```
prototype_k = (1 - decay) * prototype_k + decay * x
```

Default `decay = 0.01`. Update only during training, freeze during inference.

**Dead prototype reinit**: If a prototype is not the nearest neighbor for any input in
the last `reinit_window` batches (default: 100), reinitialize it to a random input from
the current batch. This prevents prototype collapse.

**Novelty computation**:

```python
def compute_novelty_prototype(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: (B, D) workspace representation.
    Returns:
        novelty: (B,) float in [0, 1].
    """
    # (B, K) pairwise L2 distances
    dists = torch.cdist(x.unsqueeze(0), self.prototypes.unsqueeze(0)).squeeze(0)

    # Nearest prototype distance
    min_dist, _ = dists.min(dim=1)  # (B,)

    # Normalize to [0, 1] using running statistics
    self._update_dist_stats(min_dist)
    novelty = (min_dist - self.dist_mean) / (self.dist_std + 1e-8)
    novelty = torch.sigmoid(novelty)  # squash to [0, 1]

    return novelty
```

**Running statistics**: Track the mean and standard deviation of nearest-prototype
distances with EMA (momentum=0.1). Use these to z-score normalize distances before
applying sigmoid.

### Method 2: HTM Anomaly Proxy

Reuse the HTM temporal memory anomaly score directly as a novelty proxy. This avoids
maintaining a separate novelty model but conflates temporal surprise with distributional
novelty.

```python
def compute_novelty_htm(self, anomaly: torch.Tensor) -> torch.Tensor:
    """Pass-through: treat HTM anomaly as novelty."""
    return anomaly  # already (B,) in [0, 1]
```

Use this method when:
- HTM is enabled (`use_htm=True`)
- Temporal surprise is a reasonable proxy for input novelty
- Minimizing parameter count is a priority

Do not use when:
- HTM is disabled (anomaly defaults to 0.0, making novelty useless)
- Inputs are non-sequential (anomaly has no meaningful temporal context)

### Method 3: Engram Miss Rate Proxy

Use the engram memory's retrieval miss rate as a novelty signal. The engram memory stores
N-gram hashes of previously seen patterns. If a new input does not match any stored hash,
it is novel.

```python
def compute_novelty_engram(self, engram_hits: torch.Tensor) -> torch.Tensor:
    """
    Args:
        engram_hits: (B,) float in [0, 1], fraction of N-gram hashes matched.
    Returns:
        novelty: (B,) float in [0, 1].
    """
    return 1.0 - engram_hits  # miss rate = novelty
```

Use this method when:
- Engram memory is enabled (`use_engram=True`)
- The engram table has been populated during training
- Pattern-level novelty (not distributional) is desired

Do not use when:
- Engram is disabled (hits default to 1.0, making novelty always 0.0)
- The engram table is cold (newly initialized, everything misses)

### Comparison Table

| Property | Prototype Distance | HTM Anomaly | Engram Miss Rate |
|---|---|---|---|
| Extra parameters | K * D (prototypes) | None | None |
| Extra compute | O(B * K * D) cdist | None | None |
| Requires HTM | No | Yes | No |
| Requires Engram | No | No | Yes |
| Captures distributional shift | Strong | Weak | Moderate |
| Captures temporal surprise | No | Strong | No |
| Captures pattern novelty | Moderate | No | Strong |
| Cold-start behavior | K-means init needed | Immediate | All-miss until populated |
| Recommended default | Yes | No | No |

### Novelty Scorer Dispatcher

```python
class NoveltyScorer(nn.Module):
    def __init__(self, method: str, workspace_dim: int, n_prototypes: int = 64):
        super().__init__()
        self.method = method
        if method == "prototype":
            self.prototypes = nn.Parameter(
                torch.randn(n_prototypes, workspace_dim), requires_grad=False
            )
            self.register_buffer("dist_mean", torch.tensor(0.0))
            self.register_buffer("dist_std", torch.tensor(1.0))
            self.register_buffer("hit_counts", torch.zeros(n_prototypes, dtype=torch.long))
            self.register_buffer("batch_counter", torch.tensor(0, dtype=torch.long))

    def forward(
        self,
        x: torch.Tensor,
        anomaly: Optional[torch.Tensor] = None,
        engram_hits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.method == "prototype":
            return self.compute_novelty_prototype(x)
        elif self.method == "htm_anomaly":
            assert anomaly is not None, "HTM anomaly required for htm_anomaly method"
            return self.compute_novelty_htm(anomaly)
        elif self.method == "engram_miss":
            assert engram_hits is not None, "Engram hits required for engram_miss method"
            return self.compute_novelty_engram(engram_hits)
        else:
            raise ValueError(f"Unknown novelty method: {self.method}")
```

---

## 7. Determinism Requirements (Gate A)

The router must produce bit-identical outputs across repeated runs with the same inputs.
This section specifies the constraints and test protocol.

### Prohibited Operations

Do not use the following in the router's forward path:

| Operation | Why it breaks determinism | Alternative |
|---|---|---|
| `torch.sort` on GPU (non-stable) | GPU sort is non-deterministic for equal elements | Use `torch.sort(stable=True)` or avoid sorting |
| `torch.multinomial` | Sampling is inherently stochastic | Use `torch.argmax` or threshold |
| `torch.bernoulli` | Random coin flip | Use hard threshold comparison |
| `F.dropout` | Random mask | Remove dropout from router path |
| `torch.scatter_add_` on GPU | Non-deterministic accumulation | Use `torch.scatter` with deterministic indexing |
| `torch.nn.functional.gumbel_softmax` | Adds Gumbel noise | Use hard argmax |
| Custom CUDA kernels with atomicAdd | Non-deterministic floating-point accumulation | Use serial accumulation or deterministic alternatives |

### Explicit Tie-Breaking

When comparing route scores against a threshold, ties are rare but possible. Define
explicit tie-breaking behavior:

```python
# >= means tie goes to S2 (deliberation-favoring)
use_s2 = route_score >= threshold

# > means tie goes to S1 (efficiency-favoring)
# use_s2 = route_score > threshold
```

Document which convention is active. Default: `>=` (favor deliberation on ties).

### CUDA Deterministic Mode

Enable PyTorch's deterministic mode in the training script:

```python
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

These settings may reduce performance (5-15% slowdown) but guarantee reproducibility.
Enable them for all router unit tests. In production training, enable them at least
during validation to verify determinism.

### Test Protocol

Run the following test to verify bit-identical routing:

```python
def test_router_determinism():
    """Run router 10 times with identical inputs; all outputs must match."""
    config = DualProcessConfig.minimal()
    router = MetacognitiveRouter(config)
    router.train(False)

    torch.manual_seed(42)
    B = 16
    inputs = {
        "calibrated_conf": torch.rand(B),
        "novelty": torch.rand(B),
        "anomaly": torch.rand(B),
        "ignition": torch.rand(B),
        "remaining_budget": 100.0,
    }

    reference = router(**inputs)

    for run in range(10):
        result = router(**inputs)
        assert torch.equal(result.route_scores, reference.route_scores), (
            f"route_scores differ on run {run}"
        )
        assert torch.equal(result.use_s2, reference.use_s2), (
            f"use_s2 differs on run {run}"
        )
        assert torch.equal(result.allocated_steps, reference.allocated_steps), (
            f"allocated_steps differ on run {run}"
        )
        assert result.budget_remaining == reference.budget_remaining, (
            f"budget_remaining differs on run {run}"
        )
```

### Seed Isolation

The router must not consume random state. Verify that calling the router does not
advance the global RNG:

```python
def test_router_no_rng_consumption():
    """Router must not advance random state."""
    router = MetacognitiveRouter(config)
    router.train(False)
    inputs = make_dummy_inputs(B=8)

    torch.manual_seed(123)
    state_before = torch.random.get_rng_state()
    _ = router(**inputs)
    state_after = torch.random.get_rng_state()

    assert torch.equal(state_before, state_after), "Router consumed RNG state"
```

---

## 8. Batch-Level Independence

### Formal Statement

For any batch of size B, the routing decision for item `i` must be identical regardless
of what items `j != i` are present in the same batch. Formally:

```
For all i in [0, B):
    decision(batch)[i] == decision(batch_with_item_i_only)[0]
```

This property is critical for:
- Reproducibility: Results must not change with batch size.
- Debugging: Isolate issues to individual inputs.
- Deployment: Batch size varies at inference time.

### Prohibited Operations

Do not use the following along the batch dimension (dim=0) in the router:

| Operation | Dimension | Why it violates independence | Status |
|---|---|---|---|
| `torch.sort(x, dim=0)` | batch | Reorders items based on peers | Prohibited |
| `torch.topk(x, k, dim=0)` | batch | Selection depends on peers | Prohibited |
| `F.softmax(x, dim=0)` | batch | Normalization depends on peers | Prohibited |
| `nn.BatchNorm1d(x)` | batch | Mean/var computed across batch | Prohibited |
| `nn.LayerNorm(x)` applied across batch | batch | Same as above if misapplied | Prohibited |
| `x - x.mean(dim=0)` | batch | Centering depends on peers | Prohibited |
| `x / x.std(dim=0)` | batch | Scaling depends on peers | Prohibited |
| `torch.cumsum(x, dim=0)` | batch | Prefix sums couple items | Prohibited |

Operations along `dim=1` (feature dimension) are safe, as they operate within each item.

### Allowed Operations

These are safe along dim=0:

| Operation | Why it is safe |
|---|---|
| Element-wise arithmetic (`+`, `*`, etc.) | Per-element, no cross-batch coupling |
| `torch.clamp(x, min, max)` | Per-element |
| `torch.sigmoid(x)` | Per-element |
| `torch.where(cond, a, b)` | Per-element conditional |
| `x >= threshold` | Per-element comparison |
| `torch.round(x)` | Per-element |
| Indexing `x[mask]` for gathering results | No coupling (read-only) |

### Per-Item Isolation Test

```python
def test_batch_independence():
    """Each item's routing must be independent of other items in the batch."""
    config = DualProcessConfig.minimal()
    router = MetacognitiveRouter(config)
    router.train(False)

    B = 8
    torch.manual_seed(0)
    conf = torch.rand(B)
    novelty = torch.rand(B)
    anomaly = torch.rand(B)
    ignition = torch.rand(B)
    budget = 100.0

    # Full batch
    full_result = router(conf, novelty, anomaly, ignition, budget)

    # Item-by-item
    for i in range(B):
        single_result = router(
            conf[i:i+1], novelty[i:i+1], anomaly[i:i+1], ignition[i:i+1], budget
        )
        assert torch.equal(
            full_result.route_scores[i:i+1], single_result.route_scores
        ), f"route_score mismatch for item {i}"
        assert torch.equal(
            full_result.use_s2[i:i+1], single_result.use_s2
        ), f"use_s2 mismatch for item {i}"
        assert torch.equal(
            full_result.allocated_steps[i:i+1], single_result.allocated_steps
        ), f"allocated_steps mismatch for item {i}"
```

Note: The budget remaining will differ in the per-item loop because each call deducts
budget independently. This is expected. The test validates that the routing *decision*
(score, route, steps) is independent, not the post-decision budget state.

---

## 9. Budget Policy

### Lifecycle

The System 2 budget follows a three-phase lifecycle:

```
INIT  ──────>  CONSUME  ──────>  RESET
  |                |                |
  |   Budget set   |  Items routed  |  Epoch/episode ends
  |   to total     |  to S2 deduct  |  Budget restored
  |                |  from budget   |
  └────────────────┘────────────────┘
```

1. **INIT**: At the start of each epoch (or episode, depending on config), set
   `remaining_budget = total_s2_budget`.
2. **CONSUME**: Each batch, the router deducts the number of items routed to S2.
3. **RESET**: At epoch/episode boundary, restore budget to `total_s2_budget`.

### BudgetTracker Pseudocode

```python
class BudgetTracker:
    """Track remaining System 2 invocation budget."""

    def __init__(self, total_budget: float):
        self.total_budget = total_budget
        self.remaining = total_budget
        self._history: list[float] = []

    def consume(self, n_items: int) -> float:
        """Deduct n_items from budget. Return new remaining."""
        self.remaining = max(0.0, self.remaining - n_items)
        self._history.append(n_items)
        return self.remaining

    def reset(self):
        """Restore budget to full. Call at epoch/episode boundary."""
        self.remaining = self.total_budget
        self._history.clear()

    @property
    def fraction_remaining(self) -> float:
        """Fraction of budget remaining, in [0, 1]."""
        if self.total_budget <= 0:
            return 0.0
        return self.remaining / self.total_budget

    @property
    def is_exhausted(self) -> bool:
        return self.remaining <= 0.0

    def summary(self) -> dict:
        """Return summary statistics for logging."""
        total_consumed = sum(self._history)
        return {
            "total_budget": self.total_budget,
            "remaining": self.remaining,
            "total_consumed": total_consumed,
            "n_batches": len(self._history),
            "avg_per_batch": total_consumed / max(1, len(self._history)),
        }
```

### Feedback Loop

The budget penalty in the route score formula (Section 3) creates a soft feedback loop:

1. As budget depletes, `penalty` increases.
2. Higher penalty lowers route scores.
3. Lower route scores mean fewer items cross the S2 threshold.
4. Fewer S2 items mean slower budget depletion.

This self-regulating mechanism spreads S2 invocations across the epoch rather than
exhausting the budget on the first few batches.

### Exhaustion Behavior

When the budget reaches zero, two behaviors are available:

**Soft cutoff (default)**: The budget penalty in the score formula drives route scores
down, but does not absolutely prevent S2 routing. Items with very high urgency (high
novelty + low confidence + high anomaly) can still breach the threshold even with
maximum penalty.

**Hard cutoff**: Force `use_s2 = False` for all items when `remaining_budget <= 0`.
Enable via `DualProcessConfig.hard_budget_cutoff = True`.

```python
# In _make_decision():
if self.config.hard_budget_cutoff and remaining_budget <= 0:
    if not self.config.always_run_s2:
        use_s2 = torch.zeros_like(use_s2, dtype=torch.bool)
```

Use hard cutoff when:
- Strict compute budgets must be honored (latency SLAs, cost caps).
- The soft penalty is insufficient to prevent overspend.

Use soft cutoff when:
- It is acceptable to slightly exceed the budget for high-urgency items.
- You prefer graceful degradation over hard walls.

### Gradient Non-Differentiability

The budget tracker is intentionally non-differentiable. Do not backpropagate through
the budget consumption path. Reasons:

1. The budget is a global scalar shared across the batch -- differentiating through it
   would violate per-item independence.
2. The budget represents an operational constraint, not a learned parameter.
3. Discrete consumption (integer item count) is inherently non-differentiable.

Detach any tensors before they interact with the budget:

```python
n_s2 = use_s2.detach().sum().item()  # .detach() + .item() breaks the graph
self.budget_tracker.consume(int(n_s2))
```

---

## 10. Code Examples

### Full Router Forward

Complete implementation of `MetacognitiveRouter.forward()` assembling all components:

```python
class MetacognitiveRouter(nn.Module):
    def __init__(self, config: DualProcessConfig):
        super().__init__()
        self.config = config
        self.budget_tracker = BudgetTracker(config.total_s2_budget)
        self.novelty_scorer = NoveltyScorer(
            method=config.novelty_method,
            workspace_dim=config.workspace_dim,
            n_prototypes=config.n_prototypes,
        )

    def forward(
        self,
        calibrated_conf: torch.Tensor,
        novelty: torch.Tensor,
        anomaly: torch.Tensor,
        ignition: torch.Tensor,
        remaining_budget: float,
    ) -> RoutingDecision:
        # 1. Validate inputs
        self._validate_inputs(calibrated_conf, novelty, anomaly, ignition, remaining_budget)

        # 2. Compute route scores
        route_scores = self._compute_route_scores(
            calibrated_conf, novelty, anomaly, ignition, remaining_budget
        )

        # 3. Threshold comparison
        use_s2 = route_scores >= self.config.s2_threshold

        # 4. Hard skip for high-confidence items
        hard_skip = calibrated_conf >= self.config.min_conf_to_skip
        use_s2 = use_s2 & ~hard_skip

        # 5. Debug override
        if self.config.always_run_s2:
            use_s2 = torch.ones_like(use_s2, dtype=torch.bool)

        # 6. Hard budget cutoff
        if self.config.hard_budget_cutoff and remaining_budget <= 0:
            if not self.config.always_run_s2:
                use_s2 = torch.zeros_like(use_s2, dtype=torch.bool)

        # 7. Allocate steps
        allocated_steps = self._allocate_steps(route_scores, use_s2)

        # 8. Consume budget (non-differentiable)
        n_s2 = use_s2.detach().sum().item()
        new_budget = self.budget_tracker.consume(int(n_s2))

        # 9. Assemble decision
        return RoutingDecision(
            route_scores=route_scores,
            use_s2=use_s2,
            allocated_steps=allocated_steps,
            budget_remaining=new_budget,
            debug_info={
                "raw_scores": route_scores.detach().clone(),
                "hard_skipped": hard_skip.sum().item(),
                "threshold_used": self.config.s2_threshold,
                "penalty": max(0.0, 1.0 - remaining_budget / self.config.total_s2_budget),
                "n_s2_this_batch": int(n_s2),
            },
        )

    def _validate_inputs(self, calibrated_conf, novelty, anomaly, ignition, remaining_budget):
        B = calibrated_conf.shape[0]
        for name, t in [("calibrated_conf", calibrated_conf), ("novelty", novelty),
                        ("anomaly", anomaly), ("ignition", ignition)]:
            assert t.shape == (B,), f"{name}: expected ({B},), got {t.shape}"
            assert t.min() >= 0.0 and t.max() <= 1.0, (
                f"{name} out of [0,1]: [{t.min().item():.4f}, {t.max().item():.4f}]"
            )
        assert remaining_budget >= 0, f"Negative budget: {remaining_budget}"

    def _compute_route_scores(self, conf, novelty, anomaly, ignition, remaining_budget):
        w = self.config.routing_weights
        penalty = max(0.0, 1.0 - remaining_budget / self.config.total_s2_budget)
        score = (
            w.w_conf * (1.0 - conf)
            + w.w_novelty * novelty
            + w.w_anomaly * anomaly
            + w.w_ignition * (1.0 - ignition)
            - w.w_budget * penalty
        )
        return torch.clamp(score, 0.0, 1.0)

    def _allocate_steps(self, route_scores, use_s2):
        base = self.config.base_steps
        alpha = self.config.step_alpha
        max_s = self.config.max_s2_steps
        raw = base + alpha * route_scores
        clamped = torch.clamp(torch.round(raw), 1, max_s).long()
        return clamped * use_s2.long()
```

### Isolated Route Score Computation

Standalone function for testing and visualization:

```python
def compute_route_score(
    conf: float,
    novelty: float,
    anomaly: float,
    ignition: float,
    budget_fraction: float,
    weights: dict = None,
) -> float:
    """
    Compute a single route score for debugging.

    Args:
        conf: Calibrated confidence in [0, 1].
        novelty: Novelty score in [0, 1].
        anomaly: HTM anomaly score in [0, 1].
        ignition: Workspace ignition strength in [0, 1].
        budget_fraction: remaining / total budget in [0, 1].
        weights: Optional dict with keys w_conf, w_novelty, w_anomaly,
                 w_ignition, w_budget. Defaults to standard weights.

    Returns:
        Route score clamped to [0, 1].
    """
    if weights is None:
        weights = {
            "w_conf": 0.40, "w_novelty": 0.25, "w_anomaly": 0.20,
            "w_ignition": 0.10, "w_budget": 0.15,
        }

    penalty = max(0.0, 1.0 - budget_fraction)

    score = (
        weights["w_conf"] * (1.0 - conf)
        + weights["w_novelty"] * novelty
        + weights["w_anomaly"] * anomaly
        + weights["w_ignition"] * (1.0 - ignition)
        - weights["w_budget"] * penalty
    )

    return max(0.0, min(1.0, score))
```

Example calls:

```python
# High confidence, low novelty -> S1
>>> compute_route_score(conf=0.95, novelty=0.1, anomaly=0.05, ignition=0.9, budget_fraction=1.0)
0.065  # well below 0.5 threshold -> S1

# Low confidence, high novelty -> S2
>>> compute_route_score(conf=0.3, novelty=0.8, anomaly=0.6, ignition=0.4, budget_fraction=1.0)
0.66   # above 0.5 threshold -> S2

# Moderate signals, depleted budget -> S1
>>> compute_route_score(conf=0.5, novelty=0.5, anomaly=0.5, ignition=0.5, budget_fraction=0.1)
0.3725 # budget penalty drags score below threshold -> S1
```

### Novelty Scorer with Prototype Distance

Full implementation with EMA updates and dead reinit:

```python
class PrototypeNoveltyScorer(nn.Module):
    def __init__(self, workspace_dim: int, n_prototypes: int = 64, decay: float = 0.01,
                 reinit_window: int = 100, dist_momentum: float = 0.1):
        super().__init__()
        self.n_prototypes = n_prototypes
        self.decay = decay
        self.reinit_window = reinit_window
        self.dist_momentum = dist_momentum

        # Prototypes (not learned via gradient)
        self.register_buffer("prototypes", torch.randn(n_prototypes, workspace_dim))
        self.register_buffer("hit_counts", torch.zeros(n_prototypes, dtype=torch.long))
        self.register_buffer("batch_counter", torch.tensor(0, dtype=torch.long))
        self.register_buffer("dist_mean", torch.tensor(0.0))
        self.register_buffer("dist_std", torch.tensor(1.0))
        self._initialized = False

    def initialize_from_data(self, data: torch.Tensor):
        """Run K-means to initialize prototypes. Call once with a large batch."""
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=self.n_prototypes, n_init=1, max_iter=50)
        km.fit(data.detach().cpu().numpy())
        self.prototypes.copy_(
            torch.from_numpy(km.cluster_centers_).to(self.prototypes.device)
        )
        self._initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D) workspace representations.
        Returns:
            novelty: (B,) float in [0, 1].
        """
        # Pairwise L2 distances: (B, K)
        dists = torch.cdist(x.unsqueeze(0), self.prototypes.unsqueeze(0)).squeeze(0)

        # Nearest prototype
        min_dist, nearest_idx = dists.min(dim=1)  # (B,), (B,)

        # Update prototypes (training only)
        if self.training:
            self._update_prototypes(x, nearest_idx)
            self._reinit_dead_prototypes(x)
            self.batch_counter += 1

        # Normalize distances
        self._update_dist_stats(min_dist)
        z = (min_dist - self.dist_mean) / (self.dist_std + 1e-8)
        novelty = torch.sigmoid(z)

        return novelty

    def _update_prototypes(self, x: torch.Tensor, nearest_idx: torch.Tensor):
        """EMA update of nearest prototypes."""
        for k in range(self.n_prototypes):
            mask = nearest_idx == k
            if mask.any():
                centroid = x[mask].mean(dim=0)
                self.prototypes[k] = (
                    (1 - self.decay) * self.prototypes[k] + self.decay * centroid
                )
                self.hit_counts[k] += mask.sum()

    def _reinit_dead_prototypes(self, x: torch.Tensor):
        """Reinitialize prototypes that have not been hit recently."""
        if self.batch_counter > 0 and self.batch_counter % self.reinit_window == 0:
            dead_mask = self.hit_counts == 0
            n_dead = dead_mask.sum().item()
            if n_dead > 0 and x.shape[0] > 0:
                # Pick random inputs as replacements
                indices = torch.randint(0, x.shape[0], (n_dead,), device=x.device)
                self.prototypes[dead_mask] = x[indices].detach()
            self.hit_counts.zero_()  # Reset counts for next window

    def _update_dist_stats(self, min_dist: torch.Tensor):
        """EMA update of distance running statistics."""
        batch_mean = min_dist.mean().detach()
        batch_std = min_dist.std().detach().clamp(min=1e-8)
        self.dist_mean = (
            (1 - self.dist_momentum) * self.dist_mean + self.dist_momentum * batch_mean
        )
        self.dist_std = (
            (1 - self.dist_momentum) * self.dist_std + self.dist_momentum * batch_std
        )
```

### Budget Tracking Integration

Integrate the budget tracker into a training loop:

```python
def train_epoch_with_budget(model, dataloader, optimizer, config):
    """Training loop demonstrating budget tracker integration."""
    router = model.dual_process.router
    router.budget_tracker.reset()

    epoch_stats = {"s1_count": 0, "s2_count": 0, "total_steps": 0}

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        optimizer.zero_grad()

        # System 1 forward pass (always runs)
        s1_out, s1_conf = model.system1(inputs)

        # Compute routing signals
        novelty = router.novelty_scorer(model.workspace_repr)
        anomaly = (
            model.htm.anomaly_score
            if config.use_htm
            else torch.zeros_like(s1_conf)
        )
        ignition = (
            model.workspace.ignition_strength
            if config.use_workspace
            else torch.ones_like(s1_conf)
        )

        # Route
        decision = router(
            calibrated_conf=s1_conf,
            novelty=novelty,
            anomaly=anomaly,
            ignition=ignition,
            remaining_budget=router.budget_tracker.remaining,
        )

        # System 2 forward pass (only for routed items)
        if decision.use_s2.any():
            s2_mask = decision.use_s2
            s2_steps = decision.allocated_steps[s2_mask]
            s2_inputs = inputs[s2_mask]

            s2_out = model.system2(s2_inputs, max_steps=s2_steps)

            # Merge S1 and S2 outputs
            final_out = s1_out.clone()
            final_out[s2_mask] = s2_out
        else:
            final_out = s1_out

        # Loss and backward
        loss = F.cross_entropy(final_out, targets)
        loss.backward()
        optimizer.step()

        # Track statistics
        n_s2 = decision.use_s2.sum().item()
        epoch_stats["s1_count"] += inputs.shape[0] - n_s2
        epoch_stats["s2_count"] += n_s2
        epoch_stats["total_steps"] += decision.allocated_steps.sum().item()

        # Log periodically
        if batch_idx % 100 == 0:
            budget_info = router.budget_tracker.summary()
            print(
                f"Batch {batch_idx}: "
                f"S2 rate={n_s2/inputs.shape[0]:.2%}, "
                f"budget={budget_info['remaining']:.0f}/{budget_info['total_budget']:.0f}, "
                f"avg_steps={decision.allocated_steps[decision.use_s2].float().mean():.1f}"
            )

    # End-of-epoch summary
    total = epoch_stats["s1_count"] + epoch_stats["s2_count"]
    print(
        f"Epoch complete: S2 rate={epoch_stats['s2_count']/total:.2%}, "
        f"total S2 steps={epoch_stats['total_steps']}"
    )
    return epoch_stats
```

---

## Appendix A: Config Reference Table

All configuration parameters related to metacognitive routing, collected in one place.

| Parameter | Type | Default | Location | Description |
|---|---|---|---|---|
| `s2_threshold` | float | 0.5 | `DualProcessConfig` | Route score threshold for S2 routing |
| `min_conf_to_skip` | float | 0.95 | `DualProcessConfig` | Hard skip S2 if S1 conf >= this |
| `always_run_s2` | bool | False | `DualProcessConfig` | Debug: force all items through S2 |
| `total_s2_budget` | float | 1000.0 | `DualProcessConfig` | Total S2 invocations per epoch |
| `hard_budget_cutoff` | bool | False | `DualProcessConfig` | Force S1 when budget is exhausted |
| `base_steps` | int | 1 | `DualProcessConfig` | Minimum S2 steps per invocation |
| `step_alpha` | float | 4.0 | `DualProcessConfig` | Steps scaling factor |
| `max_s2_steps` | int | 8 | `DualProcessConfig` | Hard ceiling on S2 iterations |
| `novelty_method` | str | "prototype" | `DualProcessConfig` | Novelty scorer method |
| `n_prototypes` | int | 64 | `DualProcessConfig` | Number of prototype vectors (prototype method) |
| `prototype_decay` | float | 0.01 | `DualProcessConfig` | EMA decay for prototype updates |
| `reinit_window` | int | 100 | `DualProcessConfig` | Batches between dead prototype checks |
| `dist_momentum` | float | 0.1 | `DualProcessConfig` | EMA momentum for distance statistics |
| `workspace_dim` | int | 4096 | `DualProcessConfig` | Workspace representation dimension |
| `routing_weights.w_conf` | float | 0.40 | `RoutingWeights` | Weight for (1 - confidence) term |
| `routing_weights.w_novelty` | float | 0.25 | `RoutingWeights` | Weight for novelty term |
| `routing_weights.w_anomaly` | float | 0.20 | `RoutingWeights` | Weight for anomaly term |
| `routing_weights.w_ignition` | float | 0.10 | `RoutingWeights` | Weight for (1 - ignition) term |
| `routing_weights.w_budget` | float | 0.15 | `RoutingWeights` | Weight for budget penalty term |
| `budget_reset_policy` | str | "epoch" | `DualProcessConfig` | When to reset budget: "epoch" or "episode" |

---

## Appendix B: ASCII Decision Flow Diagram

```
                          ┌─────────────────────┐
                          │   Routing Inputs     │
                          │  conf, novelty,      │
                          │  anomaly, ignition,  │
                          │  remaining_budget    │
                          └─────────┬───────────┘
                                    │
                                    ▼
                          ┌─────────────────────┐
                          │  Validate Inputs     │
                          │  shapes, ranges      │
                          └─────────┬───────────┘
                                    │
                                    ▼
                          ┌─────────────────────┐
                          │ Compute Route Score  │
                          │ w*(1-c) + w*n +     │
                          │ w*a + w*(1-i) - w*p │
                          │ clamp to [0, 1]      │
                          └─────────┬───────────┘
                                    │
                                    ▼
                          ┌─────────────────────┐
                          │  score >= threshold? │
                          └──────┬────────┬─────┘
                                 │ Yes    │ No
                                 ▼        ▼
                          ┌─────────┐  ┌──────┐
                          │ use_s2  │  │ S1   │
                          │ = True  │  │ only │
                          └────┬────┘  └──────┘
                               │
                               ▼
                    ┌────────────────────────┐
                    │ conf >= min_conf_skip? │
                    └─────┬───────────┬─────┘
                          │ Yes       │ No
                          ▼           ▼
                    ┌──────────┐  ┌─────────────┐
                    │ Override │  │  Keep S2     │
                    │ to S1    │  │  decision    │
                    └──────────┘  └──────┬──────┘
                                         │
                                         ▼
                              ┌────────────────────┐
                              │  always_run_s2?    │
                              └────┬──────────┬────┘
                                   │ Yes      │ No
                                   ▼          ▼
                            ┌──────────┐  ┌───────────┐
                            │ Force S2 │  │ Keep      │
                            │ for all  │  │ decision  │
                            └────┬─────┘  └─────┬─────┘
                                 │              │
                                 └──────┬───────┘
                                        │
                                        ▼
                           ┌──────────────────────┐
                           │ hard_cutoff AND      │
                           │ budget <= 0?         │
                           └────┬────────────┬────┘
                                │ Yes        │ No
                                ▼            ▼
                         ┌──────────┐  ┌───────────┐
                         │ Force S1 │  │ Keep      │
                         │ for all  │  │ decision  │
                         └──────────┘  └─────┬─────┘
                                             │
                                             ▼
                              ┌────────────────────────┐
                              │  Allocate Steps        │
                              │  clamp(round(b+a*s),   │
                              │        1, max)         │
                              │  zero for S1 items     │
                              └────────────┬───────────┘
                                           │
                                           ▼
                              ┌────────────────────────┐
                              │  Consume Budget        │
                              │  tracker.consume(n_s2) │
                              │  (non-differentiable)  │
                              └────────────┬───────────┘
                                           │
                                           ▼
                              ┌────────────────────────┐
                              │  Return                │
                              │  RoutingDecision       │
                              └────────────────────────┘
```

---

## Appendix C: Tuning Scenarios

### Scenario 1: Too Many Items Routed to S2

**Symptom**: S2 rate > 50%, budget exhausted by mid-epoch, high compute cost.

**Diagnosis**:
- System 1 confidence head is poorly calibrated (underconfident).
- Threshold is too low.
- Novelty scorer is over-sensitive (prototypes not converged).

**Fixes (apply in order)**:

| Fix | Parameter | Change | Expected Effect |
|---|---|---|---|
| Raise threshold | `s2_threshold` | 0.5 -> 0.65 | Direct reduction in S2 rate |
| Recalibrate S1 | (retrain calibration) | -- | Confidence scores become reliable |
| Lower novelty weight | `w_novelty` | 0.25 -> 0.15 | Novelty contributes less to score |
| Increase prototype count | `n_prototypes` | 64 -> 128 | Better coverage reduces false novelty |
| Enable hard cutoff | `hard_budget_cutoff` | False -> True | Enforce budget ceiling |

### Scenario 2: Too Few Items Routed to S2

**Symptom**: S2 rate < 5%, budget barely used, accuracy on hard examples is poor.

**Diagnosis**:
- System 1 is overconfident (miscalibrated high).
- Threshold is too high.
- Novelty and anomaly signals are suppressed (HTM disabled, engram cold).

**Fixes**:

| Fix | Parameter | Change | Expected Effect |
|---|---|---|---|
| Lower threshold | `s2_threshold` | 0.5 -> 0.35 | More items breach threshold |
| Recalibrate S1 | (retrain calibration) | -- | Reduce overconfidence |
| Raise novelty weight | `w_novelty` | 0.25 -> 0.35 | Novel items more likely to route |
| Lower hard skip floor | `min_conf_to_skip` | 0.95 -> 0.99 | Fewer items hard-skipped |
| Check HTM is enabled | `use_htm` | ensure True | Anomaly signal becomes active |

### Scenario 3: Budget Exhausted Early in Epoch

**Symptom**: Budget runs out in first 10-20% of epoch, remaining batches get no S2.

**Diagnosis**:
- Soft penalty ramp is too gradual (budget penalty weight too low).
- Burst of hard examples at epoch start (e.g., dataset ordering).

**Fixes**:

| Fix | Parameter | Change | Expected Effect |
|---|---|---|---|
| Raise budget penalty weight | `w_budget` | 0.15 -> 0.30 | Stronger depletion pushback |
| Increase total budget | `total_s2_budget` | 1000 -> 2000 | More headroom |
| Shuffle dataset | (training script) | -- | Distribute difficulty evenly |
| Enable hard cutoff | `hard_budget_cutoff` | True | Prevents overspend |
| Use step-level budgeting | (code change) | count steps, not items | Fairer accounting |

### Scenario 4: S2 Not Helping (S2 Accuracy ~= S1 Accuracy)

**Symptom**: Items routed to S2 show no accuracy improvement over S1 predictions.

**Diagnosis**:
- The router is sending the wrong items. S2 receives easy items, not hard ones.
- System 2 is undertrained or has too few steps.
- The confidence head is a poor discriminator of difficulty.

**Fixes**:

| Fix | Parameter | Change | Expected Effect |
|---|---|---|---|
| Raise confidence weight | `w_conf` | 0.40 -> 0.55 | Focus routing on low-conf items |
| Increase steps | `step_alpha` | 4 -> 7 | More deliberation per item |
| Increase max steps | `max_s2_steps` | 8 -> 12 | Higher ceiling for hardest items |
| Retrain S2 | (training script) | more S2 training epochs | S2 becomes more capable |
| Add S2 loss weighting | (training script) | weight S2 loss by (1-conf) | Prioritize learning on hard items |
| Audit confidence calibration | (reliability diagram) | -- | Ensure conf predicts difficulty |

---

## Appendix D: Integration Checklist

Use this checklist when integrating the metacognitive router into a new pipeline or
verifying an existing integration.

### Module Wiring

- [ ] `MetacognitiveRouter` is instantiated in `DualProcessReasoning.__init__()`.
- [ ] Router receives `DualProcessConfig` with all routing parameters set.
- [ ] `NoveltyScorer` is initialized with the correct `novelty_method` and `workspace_dim`.
- [ ] `BudgetTracker` is initialized with `total_s2_budget` from config.
- [ ] Router's `forward()` is called between System 1 output and System 2 input.

### Input Signals

- [ ] `calibrated_conf` comes from the temperature-scaled confidence head (not raw logits).
- [ ] `novelty` comes from `NoveltyScorer.forward()`, not a placeholder.
- [ ] `anomaly` defaults to `torch.zeros(B)` when `use_htm=False`.
- [ ] `ignition` defaults to `torch.ones(B)` when `use_workspace=False`.
- [ ] `remaining_budget` is read from `BudgetTracker.remaining` before the router call.

### Budget Management

- [ ] `BudgetTracker.reset()` is called at the correct boundary (epoch or episode).
- [ ] Budget consumption happens inside `router.forward()`, not externally.
- [ ] Budget is not double-deducted (router handles it internally).
- [ ] Budget summary is logged at epoch end for monitoring.

### Determinism

- [ ] Router forward path contains no `dropout`, `multinomial`, or `gumbel_softmax`.
- [ ] No `torch.sort` or `torch.topk` along `dim=0` in the router.
- [ ] No `BatchNorm` in the router.
- [ ] `test_router_determinism` passes (10 runs identical).
- [ ] `test_router_no_rng_consumption` passes.
- [ ] `torch.use_deterministic_algorithms(True)` is set in test configuration.

### Batch Independence

- [ ] `test_batch_independence` passes (single-item matches batch-item).
- [ ] No operations along `dim=0` that couple items (softmax, mean, std).
- [ ] Budget remaining is passed as a scalar, not derived from batch statistics.

### Output Handling

- [ ] `RoutingDecision.use_s2` is used to mask inputs for System 2.
- [ ] `RoutingDecision.allocated_steps` is passed to System 2 as `max_steps`.
- [ ] Items with `use_s2=False` retain their System 1 predictions unchanged.
- [ ] Items with `use_s2=True` have their outputs replaced by System 2 outputs.
- [ ] `debug_info` is logged during validation for monitoring.

### Training Integration

- [ ] Gradients flow through the route score computation to the confidence head (via
  straight-through estimator if needed).
- [ ] Budget consumption path is detached (`.detach().sum().item()`).
- [ ] Novelty scorer prototypes are updated only during `model.train()`.
- [ ] Novelty scorer prototypes are initialized from data before the first epoch.
- [ ] Prototype dead-reinit is active during training.

### Deployment

- [ ] `always_run_s2` is set to `False` in production config.
- [ ] `hard_budget_cutoff` is set based on latency requirements.
- [ ] `total_s2_budget` is tuned based on deployment compute constraints.
- [ ] Router is in inference mode during deployment (no prototype updates).
- [ ] Batch size variation does not change per-item routing decisions.
