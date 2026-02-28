# System 2 Iterative Refinement with Convergence Detection

Reference documentation for the System 2 reasoning loop in Skill #9 (Dual-Process Reasoning).
Specifies the iterative refinement architecture, convergence criteria, effort budget enforcement,
selective execution strategy, gradient flow, and step-level metric tracking. All designs target
`brain_ai/reasoning/system2.py` and its integration with `DualProcessReasoner`.

---

## Table of Contents

1. [System 2 Loop Structure](#1-system-2-loop-structure)
2. [Refinement Module Options](#2-refinement-module-options)
3. [Convergence Criteria](#3-convergence-criteria)
4. [Effort Budget Enforcement](#4-effort-budget-enforcement)
5. [Selective Execution (Scatter/Gather)](#5-selective-execution-scattergather)
6. [Gradient Flow Through S2](#6-gradient-flow-through-s2)
7. [Step-Level Metrics](#7-step-level-metrics)
8. [Code Examples](#8-code-examples)

---

## 1. System 2 Loop Structure

### 1.1 Overview

System 2 implements a recurrent iterative refinement process. When the metacognitive router
determines that System 1 lacks sufficient confidence, the item enters System 2 for deliberate,
multi-step reasoning. The loop refines both the output prediction and an internal hidden state
until convergence is detected or a budget limit is reached.

Core invariant: System 2 always starts from System 1's output. It never discards or ignores
the fast prediction. Instead, it treats `y1` as the initial proposal and iteratively improves it.

### 1.2 Initialization

The loop begins with two initialization steps:

```
y0 = y1                            # System 1 output as initial proposal
h0 = summary_net(x, context)       # Hidden state from a learned summary network
```

- **`y0`**: System 1 prediction, shape `(B_s2, output_dim)`, where `B_s2` is the count of
  items routed to System 2 (not the full batch).
- **`h0`**: Initial hidden state, shape `(B_s2, hidden_dim)`, produced by a 2-layer MLP
  that compresses the input representation and optional context into a fixed-size vector:

```python
summary_net = nn.Sequential(
    nn.Linear(input_dim + context_dim, hidden_dim),
    nn.GELU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.LayerNorm(hidden_dim),
)
```

If no context is provided, the summary network receives only `x_summary` and the context
dimension defaults to zero.

### 1.3 Main Loop

For each step `k` in `[1, 2, ..., steps_budget]`:

1. **Propose refinement**: `yk = refine(h_{k-1}, y_{k-1}, x_summary, context)`
2. **Update hidden state**: `hk = gru_cell(h_{k-1}, cat(yk, x_summary))`
   The GRU input concatenates the new output proposal and the original input summary,
   ensuring each step integrates both the evolving prediction and fixed input evidence.
3. **Compute step metrics**: `delta_kl = KL(p_k || p_{k-1})`, `conf_k`, `argmax_k`, etc.
   These serve dual purposes: convergence detection and diagnostic tracing.
4. **Check convergence**: if met, set `halt_reason = "converged"` and break;
   if budget exhausted, set `halt_reason = "budget_exhausted"` and break;
   if NaN detected, set `halt_reason = "nan_guard"` and break.

### 1.4 Loop Output

After the loop terminates, the module returns a `System2Result` dataclass:

```python
@dataclass
class System2Result:
    y2: Tensor              # (B_s2, output_dim) final refined output
    steps_used: Tensor      # (B_s2,) int per-item step count
    converged: Tensor       # (B_s2,) bool per-item convergence flag
    halt_reason: List[str]  # per-item halt reason string
```

Different items within the same mini-batch may converge at different steps. Items that
converge early have their outputs frozen via a per-item mask while the loop continues
for remaining items to maintain batch alignment.

### 1.5 System2Iterative Module Signature

```python
class System2Iterative(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        max_steps: int = 10,
        convergence_eps: float = 1e-3,
        convergence_patience: int = 2,
        nan_guard: bool = True,
        convergence_criterion: str = "kl_stability",
        refinement_type: str = "gru",
        deep_supervision: bool = False,
        supervision_discount: float = 0.9,
        grad_clip_per_step: Optional[float] = 1.0,
        summary_hidden_dim: Optional[int] = None,
        context_dim: int = 0,
    ): ...

    def forward(
        self,
        y1: Tensor,            # (B, output_dim) System 1 output
        x_summary: Tensor,     # (B, D) compressed input
        steps_budget: Union[int, Tensor],  # per-item or global budget
        context: Optional[Tensor] = None,  # (B, context_dim)
    ) -> System2Result: ...
```

Parameter reference:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `hidden_dim` | `int` | required | GRU hidden state dimension |
| `output_dim` | `int` | required | Output logit/embedding dimension |
| `max_steps` | `int` | `10` | Hard upper bound on refinement steps |
| `convergence_eps` | `float` | `1e-3` | Threshold for convergence criterion |
| `convergence_patience` | `int` | `2` | Consecutive stable steps before halting |
| `nan_guard` | `bool` | `True` | Enable NaN detection and early halt |
| `convergence_criterion` | `str` | `"kl_stability"` | Active criterion name |
| `refinement_type` | `str` | `"gru"` | Refinement module architecture |
| `deep_supervision` | `bool` | `False` | Apply loss at each step |
| `supervision_discount` | `float` | `0.9` | Discount factor for deep supervision |
| `grad_clip_per_step` | `float/None` | `1.0` | Per-step gradient norm clip |
| `summary_hidden_dim` | `int/None` | `None` | Summary net hidden dim (defaults to hidden_dim) |
| `context_dim` | `int` | `0` | Dimension of optional context vector |

### 1.6 Per-Item vs. Global Budget

`steps_budget` may be a scalar integer (broadcast to all items) or a per-item tensor `(B,)`.
When per-item, each item has its own budget, and the loop runs for `max(steps_budget)`
iterations total with per-item masking to freeze items that exhaust their budgets.

The effective budget for each item:

```
effective_budget_i = min(steps_budget_i, max_steps)
```

This guarantees `max_steps` is never exceeded regardless of what the metacognitive router
requests.

---

## 2. Refinement Module Options

### 2.1 GRU-Based Refinement (Primary)

The default and recommended refinement module uses a GRU cell. The architecture processes
the previous output, input summary, and optional context to produce a refined output.

```python
class GRURefinement(nn.Module):
    def __init__(self, hidden_dim, output_dim, input_summary_dim, context_dim=0):
        super().__init__()
        gru_input_dim = output_dim + input_summary_dim + context_dim
        self.gru_cell = nn.GRUCell(gru_input_dim, hidden_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h_prev, y_prev, x_summary, context=None):
        gru_input = torch.cat([y_prev, x_summary], dim=-1)
        if context is not None:
            gru_input = torch.cat([gru_input, context], dim=-1)
        h_new = self.layer_norm(self.gru_cell(gru_input, h_prev))
        delta = self.output_proj(h_new)
        y_new = y_prev + delta   # Residual refinement
        return y_new, h_new
```

Key design choices:

- **Residual refinement**: The output projection produces a *delta* added to the previous
  output. This encourages small, incremental corrections rather than wholesale replacement,
  which stabilizes the loop and makes convergence detection meaningful.
- **Layer normalization** on the hidden state prevents drift over many steps.
- **GELU activation** matches the activation used elsewhere in the brain-ai pipeline.

### 2.2 Transformer Reasoning Block (Alternative)

For tasks requiring attention over structured context (e.g., slot-based workspace
representations), a single transformer layer can serve as the refinement module. This layer
is applied recurrently -- the same parameters are shared across all refinement steps.

The transformer variant is more expressive when context contains multiple slots or tokens,
but it has higher memory cost per step and does not maintain a natural recurrent hidden state
(the hidden state must be reconstructed from the transformer output at each step).

### 2.3 Symbolic Constraint Integration (Optional)

Symbolic/fuzzy rules integrated as auxiliary loss term, not altering the forward pass:

```python
# During training only
if self.symbolic_constraints is not None and self.training:
    sat_score = self.symbolic_constraints(y_k, rules)
    aux_losses[f"symbolic_step_{k}"] = self.lambda_sym * (1.0 - sat_score)
```

### 2.4 Why GRU Is Preferred

| Property | GRU | Transformer |
|---|---|---|
| Gradient stability | Gating prevents vanishing/exploding | Requires careful LR scheduling |
| Natural hidden state | Built-in via GRU cell | Must reconstruct from output |
| Memory per step | `O(hidden_dim^2)` | `O(hidden_dim^2 + seq_len * hidden_dim)` |
| Parameter count | Lower | Higher (self-attn + cross-attn + FF) |
| Context integration | Via concatenation | Via cross-attention (more flexible) |
| Suitability for long loops | Excellent (designed for sequences) | Adequate but less natural |

Use the transformer variant only when attending over multi-slot context and when the expected
number of refinement steps is small (3-5), limiting memory overhead.

---

## 3. Convergence Criteria

### 3.1 Design Principles

Convergence detection determines when System 2 should stop iterating. The design follows
these principles:

1. **Deterministic**: Reproducible for fixed input, seed, and configuration. No stochastic halting.
2. **Per-item**: Each item in the mini-batch may converge independently.
3. **Patience-based**: Criterion must hold for `M` consecutive steps to trigger halting.
4. **Configurable**: Active criterion selected via configuration, not hard-coded.
5. **Composable**: All metrics computed at every step to support future multi-criterion rules.

### 3.2 Criterion Definitions

All criteria compare the current step's output to the previous step's output. Notation:
`p_k = softmax(y_k)` for probability distributions, `y_k` for raw logits.

#### 3.2.1 KL Stability (Primary, Default)

```python
def kl_stability(p_k: Tensor, p_prev: Tensor, eps: float = 1e-8) -> Tensor:
    """Per-item KL divergence. Returns shape (B,)."""
    p_k, p_prev = p_k.clamp(min=eps), p_prev.clamp(min=eps)
    return (p_k * (p_k.log() - p_prev.log())).sum(dim=-1)
```

Converges when `KL(p_k || p_{k-1}) < convergence_eps` for M consecutive steps.
Appropriate for classification (10-1000 classes). Default eps: `1e-3`. For larger output
spaces (language modeling), increase to `1e-2`.

#### 3.2.2 Logit Stability (Alternative)

```python
def logit_stability(y_k: Tensor, y_prev: Tensor) -> Tensor:
    """Per-item max absolute logit change. Returns shape (B,)."""
    return (y_k - y_prev).abs().max(dim=-1).values
```

Converges when `max|y_k - y_{k-1}| < eps` for M steps. Preferred when KL is noisy
due to peaky (near-one-hot) distributions. Operates on raw logits, avoids softmax.
Typical eps: `0.01` to `0.1` for logits in `[-10, 10]`.

#### 3.2.3 Argmax Stability (Classification Only)

```python
def argmax_stability(y_k: Tensor, argmax_history: List[Tensor], patience: int) -> Tensor:
    """Per-item argmax stability check. Returns shape (B,) bool."""
    current = y_k.argmax(dim=-1)
    if len(argmax_history) < patience:
        return torch.zeros(y_k.size(0), dtype=torch.bool, device=y_k.device)
    stable = torch.ones(y_k.size(0), dtype=torch.bool, device=y_k.device)
    for prev in argmax_history[-patience:]:
        stable = stable & (current == prev)
    return stable
```

Converges when same argmax for M consecutive steps. Most lenient criterion: ignores
changes in confidence or distribution shape. Use only when the final class label matters,
not calibrated confidence.

#### 3.2.4 Loss Proxy (Auxiliary Loss Available)

```python
def loss_proxy_stability(loss_k: Tensor, loss_prev: Tensor, eps: float) -> Tensor:
    """Per-item loss stability. Returns shape (B,) bool."""
    return (loss_k - loss_prev).abs() < eps
```

Converges when `|loss_k - loss_{k-1}| < eps` for M steps. Requires an auxiliary loss
computed without ground-truth labels (reconstruction loss, free energy, constraint
satisfaction, consistency loss between augmented views).

### 3.3 Criterion Comparison

| Criterion | Formula | When to Use | Pros | Cons |
|---|---|---|---|---|
| KL stability | `KL(p_k \|\| p_{k-1}) < eps` | Default | Full distributional change | Noisy when peaky |
| Logit stability | `max\|y_k - y_{k-1}\| < eps` | KL noisy | Numerically stable | Ignores distribution |
| Argmax stability | Same argmax M steps | Classification | Robust, simple | Ignores confidence |
| Loss proxy | `\|loss_k - loss_{k-1}\| < eps` | Auxiliary loss | Task-relevant | Extra computation |

### 3.4 Two-Part Halt Rule

The halt decision follows a strict priority order:

```
1. NaN guard       -> halt_reason = "nan_guard"        (highest priority, immediate halt)
2. Convergence     -> halt_reason = "converged"         (normal termination)
3. Budget exceeded -> halt_reason = "budget_exhausted"  (item budget spent)
4. Max steps       -> halt_reason = "max_steps"         (global hard limit)
```

### 3.5 Halt Reasons

```python
class HaltReason:
    CONVERGED = "converged"
    MAX_STEPS = "max_steps"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NAN_GUARD = "nan_guard"
```

String constants (not enum) for serialization simplicity. Each item in the batch gets its
own halt reason independently.

### 3.6 NaN Guard

```python
def check_nan(y_k: Tensor) -> Tensor:
    """Per-item NaN detection. Returns (B,) bool."""
    return torch.isnan(y_k).any(dim=-1)
```

When NaN is detected for an item:
1. Output reverts to `y_{k-1}` (last valid output).
2. Halt reason set to `"nan_guard"`, converged flag set to `False`.
3. Warning logged once per batch to avoid log spam.

Enabled by default (`nan_guard=True`). Disable only for debugging.

### 3.7 Patience Tracking

Patience requires tracking consecutive stable steps per item with a counter tensor:

```python
stable_count = torch.zeros(B, dtype=torch.long, device=device)

# At each step k:
criterion_met = evaluate_criterion(y_k, y_prev, ...)   # (B,) bool
stable_count = torch.where(
    criterion_met, stable_count + 1, torch.zeros_like(stable_count)
)
converged = stable_count >= convergence_patience
```

The counter resets to zero whenever the criterion is NOT met. `convergence_patience = 2`
requires two *consecutive* stable steps, not two stable steps out of the last N.

---

## 4. Effort Budget Enforcement

### 4.1 Budget Source

The metacognitive router computes `steps_budget` per item based on calibrated confidence
(lower -> more steps), novelty score (higher -> more steps), task complexity signals from
context, and global compute budget constraints.

### 4.2 Constraints

```python
steps_budget = steps_budget.clamp(min=1, max=self.max_steps)
```

Invariant: `1 <= steps_budget_i <= max_steps` for all items. If a scalar integer is provided,
it is broadcast to all items.

### 4.3 Budget Tracking

At each step `k`, remaining budget is computed per item:

```python
remaining = steps_budget - k
budget_exhausted = (remaining <= 0) & (~already_halted)
```

### 4.4 Interaction with Convergence

Budget enforcement takes priority over convergence patience. If an item's budget runs out
while the patience counter is accumulating, the halt reason is `"budget_exhausted"`, not
`"converged"`.

Decision tree per item at step `k`:

```
nan_detected(y_k)       -> "nan_guard",        use y_{k-1}
converged_at_step_k     -> "converged",         use y_k
k >= steps_budget_i     -> "budget_exhausted",  use y_k
k >= max_steps          -> "max_steps",         use y_k
otherwise               -> continue to step k+1
```

### 4.5 Budget Efficiency Metrics

| Metric | Formula | Purpose |
|---|---|---|
| `budget_utilization` | `steps_used / steps_budget` | Budget consumption rate |
| `early_stop_rate` | `count(converged) / count(total_s2)` | Early convergence fraction |
| `mean_steps` | `mean(steps_used)` | Average refinement depth |
| `max_steps_hit_rate` | `count(max_steps) / count(total_s2)` | Hard limit fraction |

These metrics help tune the metacognitive router's budget allocation policy.

---

## 5. Selective Execution (Scatter/Gather)

### 5.1 Motivation

Not every item in a batch needs System 2 processing. Only items with `used_system2=True`
enter the loop. This **scatter/gather** pattern is critical for efficiency: cost is
`O(B_s2 * max_steps)` instead of `O(B * max_steps)`.

### 5.2 Scatter Phase

Extract uncertain items into a compact mini-batch:

```python
s2_mask = used_system2                          # (B,) bool
s2_indices = torch.where(s2_mask)[0]            # (B_s2,) indices
y1_s2 = y1[s2_indices]                          # (B_s2, output_dim)
x_summary_s2 = x_summary[s2_indices]            # (B_s2, D)
context_s2 = context[s2_indices] if context is not None else None
```

### 5.3 Execute S2 on Mini-Batch

```python
s2_result = self.system2(y1=y1_s2, x_summary=x_summary_s2,
                         steps_budget=budget_s2, context=context_s2)
```

### 5.4 Gather Phase

Merge results back into the full batch:

```python
y_final = y1.clone()
y_final[s2_indices] = s2_result.y2

steps_used = torch.zeros(B, dtype=torch.long, device=device)
steps_used[s2_indices] = s2_result.steps_used

halt_reasons = ["s1_only"] * B
for i, idx in enumerate(s2_indices.tolist()):
    halt_reasons[idx] = s2_result.halt_reason[i]
```

### 5.5 Alternative: torch.where Masking

For traced/compiled models where advanced indexing is undesirable:

```python
y_final = torch.where(used_system2.unsqueeze(-1), y2_full, y1)
```

Simpler but wastes compute on confident items. Prefer scatter/gather for production.

### 5.6 Edge Cases

- **No S2 items** (`s2_mask.sum() == 0`): Skip loop entirely, return `y1`.
- **All S2** (`s2_mask.all()`): Run on full batch directly, skip scatter/gather.
- **Empty batch** (`B == 0`): Return empty tensors immediately.

### 5.7 Gradient Considerations

Advanced indexing (`y1[s2_indices]`) is differentiable in PyTorch. The `clone()` in gather
is necessary to avoid in-place modification breaking autograd. `torch.where` is also
differentiable and may be preferred for JIT compilation.

---

## 6. Gradient Flow Through S2

### 6.1 Training Mode

Gradients flow through the entire loop via BPTT:

```
loss -> y_K -> refine_K -> h_K -> gru_K -> h_{K-1} -> ... -> gru_1 -> h_0 -> summary_net
                                                                          \-> y_0 = y1 -> S1
```

### 6.2 Truncated BPTT

For long loops (`max_steps > 8`), detach hidden state periodically to limit memory:

```python
if self.truncation_window is not None and k % self.truncation_window == 0:
    h_k = h_k.detach()
```

Typical window: 4-6 steps. Default: `None` (full BPTT).

### 6.3 Per-Step Gradient Clipping

Register a backward hook on the hidden state to prevent explosion:

```python
def _clip_grad_hook(grad):
    return torch.clamp(grad, -self.grad_clip_per_step, self.grad_clip_per_step)

if self.grad_clip_per_step is not None and h_k.requires_grad:
    h_k.register_hook(_clip_grad_hook)
```

### 6.4 Loss Strategies

**Final output only**: `loss = criterion(y_K, target)` -- simple, single computation. No
intermediate training signal; can lead to slow refinement learning.

**Deep supervision**: Sum losses at each step with a discount factor:

```python
for k in range(K):
    weight = discount ** (K - 1 - k)  # Later steps weighted more
    total_loss += weight * criterion(y_k, target).mean()
```

Deep supervision provides training signal at every step and stabilizes refinement learning.
Controlled by `deep_supervision` and `supervision_discount` parameters.

### 6.5 Straight-Through for Halt Decision

The halt decision is non-differentiable. During training, the loop runs to completion for
gradient stability. During evaluation, the loop genuinely halts early for efficiency.

```python
if self.training:
    y_final = all_outputs[:, -1, :]                    # Last step
else:
    step_indices = steps_used - 1                      # 0-indexed
    y_final = all_outputs[torch.arange(B_s2), step_indices]  # Convergence step
```

### 6.6 Per-Item Masking

Frozen items get zero delta to prevent output drift after convergence:

```python
delta = self.output_proj(h_k)
delta = delta * active_mask.unsqueeze(-1).float()
y_k = y_prev + delta
```

---

## 7. Step-Level Metrics

### 7.1 Metrics Per Step

| Metric | Shape | Computation | Purpose |
|---|---|---|---|
| `delta_kl` | `(B_s2,)` | `KL(softmax(y_k) \|\| softmax(y_{k-1}))` | Convergence (KL) |
| `delta_max` | `(B_s2,)` | `max(\|y_k - y_{k-1}\|)` | Convergence (logit) |
| `conf_k` | `(B_s2,)` | `softmax(y_k).max(dim=-1)` | Confidence tracking |
| `argmax_k` | `(B_s2,)` | `y_k.argmax(dim=-1)` | Argmax stability |
| `h_norm_k` | `(B_s2,)` | `\|\|h_k\|\|_2` | Hidden state health |
| `y_norm_k` | `(B_s2,)` | `\|\|y_k\|\|_2` | Output health |
| `has_nan_k` | `(B_s2,)` | `isnan(y_k).any(dim=-1)` | NaN guard |

### 7.2 Integration with ReasoningTrace

```python
@dataclass
class StepTrace:
    step: int
    output: Optional[Tensor]     # y_k (only if full trace requested)
    delta_kl: float              # Scalar, per-item, detached
    delta_max: float
    confidence: float
    predicted_class: int
    hidden_norm: float

@dataclass
class ReasoningTrace:
    route_decision: str          # "system1" or "system2"
    s1_confidence: float
    novelty_score: float
    steps_budget: int
    steps_used: int
    converged: bool
    halt_reason: str
    step_traces: List[StepTrace]
```

### 7.3 Efficient Computation

Metrics only stored when `return_details=True`. Convergence-critical metrics always computed;
tracing-only metrics gated behind the flag to minimize memory overhead.

### 7.4 Metric Consumers

| Consumer | Metrics | Purpose |
|---|---|---|
| Convergence check | `delta_kl`, `delta_max`, `argmax` | Halt decision |
| NaN guard | `has_nan` | Emergency halt |
| ReasoningTrace | All | Diagnostics |
| Deep supervision | `y_k` | Per-step loss |
| TensorBoard | All | Training monitoring |

---

## 8. Code Examples

### 8.1 System2Iterative.forward (Complete)

```python
def forward(self, y1, x_summary, steps_budget, context=None, store_metrics=False):
    B, device = y1.size(0), y1.device

    # Normalize budget to per-item tensor
    if isinstance(steps_budget, int):
        steps_budget_t = torch.full((B,), steps_budget, dtype=torch.long, device=device)
    else:
        steps_budget_t = steps_budget.clone()
    steps_budget_t = steps_budget_t.clamp(min=1, max=self.max_steps)
    effective_max = steps_budget_t.max().item()

    # Initialize
    summary_in = torch.cat([x_summary, context], dim=-1) if context is not None else x_summary
    h = self.summary_net(summary_in)
    y_prev, p_prev = y1.clone(), torch.softmax(y1, dim=-1)

    # Per-item tracking
    halted = torch.zeros(B, dtype=torch.bool, device=device)
    converged = torch.zeros(B, dtype=torch.bool, device=device)
    steps_used = torch.zeros(B, dtype=torch.long, device=device)
    halt_reasons = [HaltReason.MAX_STEPS] * B
    stable_count = torch.zeros(B, dtype=torch.long, device=device)
    y_final = y1.clone()
    all_outputs, argmax_history = [], []

    def clip_grad_hook(grad):
        return torch.clamp(grad, -self._clip_value, self._clip_value) if self._clip_value else grad

    for k in range(1, effective_max + 1):
        # Step 1: GRU refinement
        gru_input = torch.cat([y_prev, x_summary], dim=-1)
        if context is not None:
            gru_input = torch.cat([gru_input, context], dim=-1)
        h_new = self.hidden_norm(self.gru_cell(gru_input, h))

        if self._clip_value and self.training and h_new.requires_grad:
            h_new.register_hook(clip_grad_hook)
        if self.truncation_window and k % self.truncation_window == 0 and self.training:
            h_new = h_new.detach()

        delta = self.output_proj(h_new)
        delta = delta * (~halted).unsqueeze(-1).float() if self.training else delta
        y_k = y_prev + delta

        # Step 2: NaN guard
        if self.nan_guard:
            has_nan = torch.isnan(y_k).any(dim=-1) & ~halted
            if has_nan.any():
                y_k = torch.where(has_nan.unsqueeze(-1), y_prev, y_k)
                halted = halted | has_nan
                for i in has_nan.nonzero(as_tuple=True)[0].tolist():
                    halt_reasons[i], steps_used[i] = HaltReason.NAN_GUARD, k

        # Step 3: Convergence check
        p_k = torch.softmax(y_k, dim=-1)
        newly_converged, stable_count, argmax_history = check_convergence(
            self.convergence_criterion, y_k, y_prev, p_k, p_prev,
            argmax_history, stable_count, halted,
            self.convergence_eps, self.convergence_patience,
        )

        # Step 4: Budget enforcement
        budget_exceeded = (k >= steps_budget_t) & ~halted & ~newly_converged

        # Step 5: Update halt status
        for i in newly_converged.nonzero(as_tuple=True)[0].tolist():
            halt_reasons[i], steps_used[i] = HaltReason.CONVERGED, k
        converged, halted = converged | newly_converged, halted | newly_converged

        for i in budget_exceeded.nonzero(as_tuple=True)[0].tolist():
            halt_reasons[i], steps_used[i] = HaltReason.BUDGET_EXHAUSTED, k
        halted = halted | budget_exceeded

        newly_done = newly_converged | budget_exceeded
        if newly_done.any():
            y_final = torch.where(newly_done.unsqueeze(-1), y_k, y_final)
        if self.deep_supervision or store_metrics:
            all_outputs.append(y_k.clone())

        h, y_prev, p_prev = h_new, y_k, p_k

        if not self.training and halted.all():
            break

    # Handle items that never halted
    still_active = ~halted
    if still_active.any():
        y_final = torch.where(still_active.unsqueeze(-1), y_prev, y_final)
        for i in still_active.nonzero(as_tuple=True)[0].tolist():
            halt_reasons[i], steps_used[i] = HaltReason.MAX_STEPS, effective_max

    return System2Result(y2=y_final, steps_used=steps_used,
                         converged=converged, halt_reason=halt_reasons)
```

### 8.2 Scatter/Gather Selective Execution

```python
def _run_system2_selective(self, y1, x_summary, used_system2, steps_budget, context=None):
    B, device = y1.size(0), y1.device

    if not used_system2.any():
        return y1, None, torch.zeros(B, dtype=torch.long, device=device), ["s1_only"] * B

    # Scatter
    s2_idx = torch.where(used_system2)[0]
    s2_result = self.system2(
        y1=y1[s2_idx], x_summary=x_summary[s2_idx],
        steps_budget=steps_budget[s2_idx],
        context=context[s2_idx] if context is not None else None,
    )

    # Gather
    y_final = y1.clone()
    y_final[s2_idx] = s2_result.y2
    steps_used = torch.zeros(B, dtype=torch.long, device=device)
    steps_used[s2_idx] = s2_result.steps_used
    halt_reasons = ["s1_only"] * B
    for i, idx in enumerate(s2_idx.tolist()):
        halt_reasons[idx] = s2_result.halt_reason[i]

    return y_final, s2_result, steps_used, halt_reasons
```

### 8.3 Deep Supervision Loss

```python
def compute_deep_supervision_loss(all_outputs, target, criterion, discount=0.9, masks=None):
    K = len(all_outputs)
    total_loss, total_weight = 0.0, 0.0
    for k, y_k in enumerate(all_outputs):
        weight = discount ** (K - 1 - k)
        step_loss = criterion(y_k, target)
        if masks is not None and masks[k] is not None:
            step_loss = (step_loss * masks[k].float()).sum() / masks[k].float().sum().clamp(min=1)
        else:
            step_loss = step_loss.mean()
        total_loss += weight * step_loss
        total_weight += weight
    return total_loss / max(total_weight, 1e-8)
```

### 8.4 Convergence Check Dispatcher

```python
def check_convergence(criterion, y_k, y_prev, p_k, p_prev, argmax_history,
                      stable_count, halted, eps, patience, loss_k=None, loss_prev=None):
    B, device = y_k.size(0), y_k.device

    if criterion == "kl_stability":
        p_ks, p_ps = p_k.clamp(min=1e-8), p_prev.clamp(min=1e-8)
        delta = (p_ks * (p_ks.log() - p_ps.log())).sum(dim=-1)
        criterion_met = delta < eps
    elif criterion == "logit_stability":
        criterion_met = (y_k - y_prev).abs().max(dim=-1).values < eps
    elif criterion == "argmax_stability":
        current = y_k.argmax(dim=-1)
        argmax_history.append(current)
        if len(argmax_history) >= patience:
            criterion_met = torch.ones(B, dtype=torch.bool, device=device)
            for prev in argmax_history[-patience:]:
                criterion_met = criterion_met & (current == prev)
        else:
            criterion_met = torch.zeros(B, dtype=torch.bool, device=device)
        return criterion_met & ~halted, stable_count, argmax_history
    elif criterion == "loss_proxy":
        criterion_met = (loss_k - loss_prev).abs() < eps
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    stable_count = torch.where(
        criterion_met & ~halted, stable_count + 1,
        torch.where(~halted, torch.zeros_like(stable_count), stable_count),
    )
    return (stable_count >= patience) & ~halted, stable_count, argmax_history
```

### 8.5 Configuration Integration

```python
@dataclass
class System2Config:
    hidden_dim: int = 512
    max_steps: int = 10
    convergence_eps: float = 1e-3
    convergence_patience: int = 2
    nan_guard: bool = True
    convergence_criterion: str = "kl_stability"
    refinement_type: str = "gru"
    deep_supervision: bool = False
    supervision_discount: float = 0.9
    grad_clip_per_step: Optional[float] = 1.0
    truncation_window: Optional[int] = None
```

Scale presets:

| Preset | `hidden_dim` | `max_steps` | `convergence_eps` | `patience` |
|---|---|---|---|---|
| `minimal()` | 64 | 3 | 1e-2 | 1 |
| `production_1b()` | 256 | 8 | 1e-3 | 2 |
| `production_3b()` | 512 | 10 | 1e-3 | 2 |
| `production_7b()` | 1024 | 12 | 5e-4 | 3 |

---

## Appendix A: Convergence Tuning Guidelines

### A.1 Choosing the Criterion

| Task Type | Criterion | Rationale |
|---|---|---|
| Classification (10-100 classes) | KL stability | Sensitive to class + confidence changes |
| Classification (1000+ classes) | Logit stability | Avoids peaky distribution numerics |
| Classification (label-only) | Argmax stability | Most lenient, fastest convergence |
| Regression / embedding | Logit stability | No softmax involved |
| With auxiliary loss | Loss proxy | Directly task-relevant |
| Safety-critical | KL stability + low eps | Conservative, thorough refinement |

### A.2 Tuning convergence_eps

Start with the default (`1e-3` for KL, `0.01` for logit stability) and adjust based on
the observed convergence curve:

- **Too tight** (eps too small): Items rarely converge, always hitting max_steps.
  Symptom: `early_stop_rate < 0.1`, high mean steps.
- **Too loose** (eps too large): Items converge after 1-2 steps, S2 adds no value.
  Symptom: `early_stop_rate > 0.95`, mean steps close to patience.
- **Target zone**: `early_stop_rate` in [0.3, 0.8], mean steps at 40-70% of max_steps.

### A.3 Tuning convergence_patience

- `patience = 1`: Single stable step triggers halt. Fast but can be premature if oscillating.
- `patience = 2` (default): Two consecutive stable steps. Good balance for most tasks.
- `patience = 3+`: More conservative. Use for safety-critical applications or observed
  oscillation in the refinement loop.

### A.4 Monitoring Convergence Health

1. **Steps histogram**: Distribution of `steps_used`. Expect roughly log-normal. Bimodal
   (many at 1 + many at max_steps) suggests miscalibrated threshold.
2. **KL trajectory**: Plot `delta_kl` across steps. Expect monotonic decrease. Non-monotonic
   behavior indicates refinement instability.
3. **Confidence trajectory**: Plot `conf_k` across steps. Expect monotonic increase.
   Decreasing confidence indicates the refinement module is degrading the output.
4. **NaN rate**: Should be zero in a healthy system. Non-zero triggers indicate numerical
   instability -- reduce learning rate or increase layer normalization.

---

## Appendix B: Memory and Compute Estimates

### B.1 Memory Per Item Per Step

| Component | Memory | Notes |
|---|---|---|
| Hidden state `h` | `hidden_dim * 4 bytes` | Float32 |
| Output `y` | `output_dim * 4 bytes` | Float32 |
| GRU activations | `~3 * hidden_dim * 4 bytes` | Gates: reset, update, candidate |
| Output projection | `~2 * hidden_dim * 4 bytes` | Two-layer MLP |
| Gradient buffers | `~2x` above | Training only (BPTT) |

For `hidden_dim=512`, `output_dim=1000`:
- Per item per step: ~16 KB forward, ~32 KB training
- 10 steps, 32 S2 items: ~10 MB forward, ~20 MB training
- ~2M FLOPs per item per step (negligible vs transformer layers)

### B.2 Scatter/Gather Overhead

- Index computation: O(B)
- Scatter indexing: O(B_s2 * D)
- Gather clone + write: O(B * D)

For typical batch sizes (B=128, B_s2=16), overhead is sub-millisecond.

---

## Appendix C: Testing Checklist

- [ ] Correct output shapes for all input combinations
- [ ] Convergence halts loop before max_steps on stable inputs
- [ ] Per-item convergence: different items halt at different steps
- [ ] Budget enforcement: items never exceed their steps_budget
- [ ] NaN guard: NaN in logits triggers immediate halt with correct reason
- [ ] All four halt reasons correctly assigned
- [ ] Deep supervision loss computed correctly with discount
- [ ] Gradients propagate through loop to summary network
- [ ] Per-step gradient clipping prevents explosion
- [ ] Truncated BPTT detaches at configured intervals
- [ ] Scatter/gather: S1-only items unmodified
- [ ] Scatter/gather: empty S2 set returns S1 output
- [ ] Scatter/gather: all items routed to S2 works correctly
- [ ] Eval mode exits early when all items converge
- [ ] Training mode runs to completion with masking
- [ ] All four convergence criteria produce correct booleans
- [ ] Patience counter resets on criterion failure
- [ ] Metrics only stored when store_metrics=True
- [ ] System2Result compatible with ReasoningTrace
- [ ] Memory scales linearly with B_s2 * max_steps
- [ ] Deterministic output for fixed seed and deterministic settings

---

## Appendix D: Related Documents

| Document | Location | Relationship |
|---|---|---|
| System 1 Confidence | `references/system1-confidence.md` | S1 architecture, confidence, calibration |
| Metacognitive Router | `references/metacognitive-routing.md` | Routing policy, budget allocation |
| Reasoning Trace | `references/reasoning-trace.md` | Trace serialization format |
| SKILL.md | `SKILL.md` | Top-level skill specification and public contract |
| System 2 Template | `assets/system2_template.py` | Code template for System2Iterative |
| Dual-Process Template | `assets/dual_process_template.py` | Code template for DualProcessReasoner |

---

*End of System 2 Convergence Reference Document*
