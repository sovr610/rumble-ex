# System 1 Fast Predictor & Confidence Calibration

Reference document for the System 1 subsystem within Skill #9 (Dual-Process Reasoning).
Covers architecture, confidence heads, calibration utilities, fitting protocols,
reliability diagnostics, and integration with the metacognitive router.

---

## Table of Contents

1. [System 1 Architecture](#1-system-1-architecture)
2. [Confidence Heads](#2-confidence-heads)
3. [Calibration Utilities](#3-calibration-utilities)
4. [Calibration Fitting Protocol](#4-calibration-fitting-protocol)
5. [Reliability Diagrams](#5-reliability-diagrams)
6. [Integration with Routing](#6-integration-with-routing)
7. [Code Examples](#7-code-examples)

---

## 1. System 1 Architecture

### Design Philosophy

System 1 implements the fast, parallel, low-latency prediction pathway inspired by
Kahneman's dual-process theory. It must produce an answer **and** a calibrated
confidence estimate in a single forward pass, with no iterative refinement. The
confidence estimate is the primary signal that determines whether the result is
accepted or whether System 2 (slow, deliberative) is invoked.

The core constraint: System 1 must be cheap enough to run on every input. System 2
is the expensive fallback. Therefore System 1 is a shallow feed-forward network,
not a deep autoregressive model.

### Single-Pass Feed-Forward Predictor

System 1 is a 1-to-3 layer MLP or a single shallow transformer block. It takes a
fixed-dimensional representation from the global workspace and produces task logits
plus uncertainty proxies in one pass.

**Architectural options (ordered by complexity):**

| Variant | Layers | Parameters | Latency | When to Use |
|---|---|---|---|---|
| Linear head | 1 linear | `D * C` | Minimal | Baseline, toy tasks |
| Shallow MLP | 2-3 linear + GELU | `D * H + H * C` | Low | Default for classification |
| Single transformer block | 1 self-attn + FFN | `~4 * D^2` | Medium | When slot structure matters |

For the default configuration (`D = 4096`, `H = 2048`, `C = num_classes`), the
shallow MLP variant has roughly 8M-12M parameters -- negligible compared to the
full pipeline.

### Input Format

System 1 accepts two input shapes:

1. **Pooled workspace vector**: `(B, D)` -- a single vector per batch element,
   already reduced from the workspace slots. This is the simplest and most common
   input format.

2. **Slot tensor**: `(B, K, D)` -- the full set of `K` workspace slots before
   pooling. When this format is used, System 1 applies its own pooling head
   (see [Pooling Head Options](#pooling-head-options) below) to reduce to `(B, D)`
   before the prediction layers.

An optional `context` tensor `(B, D_ctx)` can be concatenated or cross-attended
to provide task-specific conditioning (e.g., a task embedding or instruction
encoding).

### Output: System1Result

The forward pass returns a structured result object, not a raw tensor. This ensures
all downstream consumers (router, logger, loss functions) receive a consistent
interface.

```
@dataclass
class System1Result:
    y1: Tensor              # (B, C) -- raw logits (unnormalized)
    conf_raw: Tensor        # (B,)   -- max softmax probability
    conf_calibrated: Tensor # (B,)   -- calibrated confidence (after T-scaling)
    entropy: Tensor         # (B,)   -- predictive entropy
    margin: Tensor          # (B,)   -- top1 - top2 logit gap
    uncertainty_metrics: dict  # Additional metrics for logging
```

Fields:

| Field | Shape | Range | Description |
|---|---|---|---|
| `y1` | `(B, C)` | `(-inf, +inf)` | Raw logits before softmax |
| `conf_raw` | `(B,)` | `[1/C, 1.0]` | `softmax(y1).max(dim=-1).values` |
| `conf_calibrated` | `(B,)` | `[0, 1]` | Calibrated confidence after temperature scaling |
| `entropy` | `(B,)` | `[0, log(C)]` | Shannon entropy of softmax distribution |
| `margin` | `(B,)` | `[0, +inf)` | Difference between top-1 and top-2 logits |
| `uncertainty_metrics` | `dict` | varies | Aggregated dict for telemetry |

### System1Fast Module Design

```python
class System1Fast(nn.Module):
    def __init__(
        self,
        input_dim: int,           # D -- workspace dimensionality (e.g. 4096)
        hidden_dim: int,          # H -- hidden layer width (e.g. 2048)
        output_dim: int,          # C -- number of output classes or actions
        num_layers: int = 2,      # 1, 2, or 3 MLP layers
        confidence_head: bool = True,  # Dedicated learned confidence head
        dropout: float = 0.1,     # Dropout rate between layers
        slot_pooling: str = "mean",    # "mean", "attention", "cls"
        num_slots: int = 0,       # K -- if > 0, expect slot input (B, K, D)
    ):
        ...

    def forward(
        self,
        x: Tensor,               # (B, D) or (B, K, D)
        context: Optional[Tensor] = None,  # (B, D_ctx) optional conditioning
    ) -> System1Result:
        ...
```

**Constructor parameters explained:**

- `input_dim`: Must match the global workspace output dimensionality. In the
  default BrainAI configuration this is 4096.
- `hidden_dim`: Width of hidden layers. Typically `input_dim // 2` for a compact
  predictor. Increasing this beyond `input_dim` yields diminishing returns -- if
  the task needs that much capacity, System 2 should handle it.
- `output_dim`: Number of classes for classification, or action dimensionality for
  decision-making tasks. For regression, set to 1 and interpret `y1` as a scalar.
- `num_layers`: Number of linear layers in the prediction MLP. One layer is a
  linear probe. Two layers add a nonlinearity. Three layers are the maximum
  recommended -- beyond that, use System 2 instead.
- `confidence_head`: When `True`, instantiate a separate small MLP that predicts
  confidence from the penultimate hidden state. When `False`, confidence is derived
  purely from the logit distribution (max softmax, entropy, margin).
- `dropout`: Applied between hidden layers. Not applied in the final projection or
  confidence head. During inference, dropout is disabled (standard `model.eval()`
  behavior).
- `slot_pooling`: Pooling strategy when input is `(B, K, D)`. Ignored when input
  is already `(B, D)`.
- `num_slots`: Expected number of slots `K`. When set to 0, the module expects
  pre-pooled `(B, D)` input. When > 0, the module instantiates a pooling head.

### Pooling Head Options

When System 1 receives slot-structured input `(B, K, D)`, it must reduce to
`(B, D)` before the prediction layers. Three strategies are supported:

#### Mean Pooling

```python
# slot_pooling = "mean"
pooled = x.mean(dim=1)  # (B, K, D) -> (B, D)
```

Simplest option. Works well when all slots carry roughly equal information.
No additional parameters.

#### Attention Pooling

```python
# slot_pooling = "attention"
# Learned query vector attends over slots
query = self.pool_query.unsqueeze(0).expand(B, -1, -1)  # (B, 1, D)
attn_weights = torch.bmm(query, x.transpose(1, 2))       # (B, 1, K)
attn_weights = F.softmax(attn_weights / sqrt(D), dim=-1)
pooled = torch.bmm(attn_weights, x).squeeze(1)            # (B, D)
```

Learns which slots are most relevant. Adds `D` parameters (the query vector).
Preferred when slots have heterogeneous importance (e.g., one slot is the "winner"
of workspace competition).

#### CLS Token

```python
# slot_pooling = "cls"
# Prepend a learned CLS token, run through one self-attention layer, take CLS output
cls_token = self.cls_token.unsqueeze(0).expand(B, -1, -1)  # (B, 1, D)
x_with_cls = torch.cat([cls_token, x], dim=1)               # (B, K+1, D)
x_out = self.pool_attn_layer(x_with_cls)                    # (B, K+1, D)
pooled = x_out[:, 0, :]                                      # (B, D)
```

Most expensive pooling option. Adds a full self-attention layer. Use only when
slot interactions are important for the confidence estimate. Typically overkill
for System 1 -- if slot interactions matter that much, route to System 2.

### Forward Pass Walkthrough

The forward method follows this sequence:

```
1. Pool slots (if input is (B, K, D))
2. Concatenate context (if provided)
3. Project through MLP layers with GELU + dropout
4. Compute task logits from final hidden state
5. Compute confidence proxies (conf_raw, entropy, margin)
6. Compute learned confidence (if confidence_head is enabled)
7. Apply temperature scaling (if calibrator is fitted)
8. Pack everything into System1Result
```

Step-by-step in more detail:

**Step 1 -- Slot pooling.** If `x.ndim == 3`, apply the configured pooling head.
If `x.ndim == 2`, skip. Raise an error if `x.ndim` is anything else.

**Step 2 -- Context concatenation.** If `context` is provided, concatenate along
the feature dimension: `x = cat([x, context], dim=-1)`. The first linear layer
must accept `input_dim + context_dim` in this case. If no context is ever used,
the first layer accepts `input_dim`.

**Step 3 -- MLP layers.** For `num_layers = 2`:
```
h = dropout(gelu(linear1(x)))   # (B, H)
```
For `num_layers = 3`:
```
h = dropout(gelu(linear1(x)))   # (B, H)
h = dropout(gelu(linear2(h)))   # (B, H)
```
For `num_layers = 1`, skip hidden layers entirely (linear probe).

**Step 4 -- Task logits.** Apply the final projection:
```
logits = output_proj(h)  # (B, C)
```
where `output_proj` is `nn.Linear(H, C)` (or `nn.Linear(D, C)` if `num_layers=1`).

**Step 5 -- Confidence proxies.** Compute all three uncertainty metrics from the
raw logits. See [Section 2](#2-confidence-heads) for formulas.

**Step 6 -- Learned confidence.** If `confidence_head` is enabled, run the
penultimate hidden state through a small dedicated MLP:
```
conf_learned = sigmoid(conf_mlp(h.detach()))  # (B,)
```
Note the `.detach()` -- the confidence head does not backpropagate into the
prediction MLP. This prevents the model from learning to make confident-but-wrong
predictions to satisfy the confidence loss.

**Step 7 -- Temperature scaling.** If a fitted `TemperatureScaler` is attached,
apply it:
```
calibrated_logits = logits / T
conf_calibrated = softmax(calibrated_logits).max(dim=-1).values
```
If no calibrator is fitted yet (e.g., during initial training), set
`conf_calibrated = conf_raw` and emit a warning on the first call.

**Step 8 -- Pack result.** Assemble all outputs into a `System1Result` dataclass.

### Initialization Details

Weight initialization matters for confidence calibration. Use the following scheme:

- Hidden layers: Kaiming normal initialization (fan_in mode, GELU nonlinearity)
- Output projection: Xavier uniform initialization (prevents logit explosion at init)
- Confidence head: Xavier uniform with bias initialized to `0.0`
  (starts predicting ~0.5 confidence)
- Temperature parameter: Initialize to `1.5` (slightly conservative -- see
  [Section 3](#3-calibration-utilities) for rationale)

```python
def _init_weights(self):
    for name, module in self.named_modules():
        if isinstance(module, nn.Linear):
            if "conf" in name:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif "output_proj" in name:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            else:
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
```

### Latency Budget

System 1 must complete in under 10% of the total System 2 latency. For a System 2
that runs 8 GRU steps at ~2ms each (16ms total), System 1 must complete in under
1.6ms on the target hardware. The shallow MLP easily meets this constraint:

| Component | Approx. Time (A100, B=32, D=4096) |
|---|---|
| Mean pool (K=8 slots) | 0.01 ms |
| Linear 4096 -> 2048 + GELU | 0.05 ms |
| Linear 2048 -> 2048 + GELU | 0.03 ms |
| Linear 2048 -> C (C=1000) | 0.02 ms |
| Softmax + confidence metrics | 0.01 ms |
| **Total** | **~0.12 ms** |

This leaves a large margin. The actual bottleneck is memory bandwidth, not compute.

---

## 2. Confidence Heads

### Why Multiple Uncertainty Proxies

A single scalar confidence score is insufficient for robust routing. Consider these
failure modes:

| Scenario | conf_raw | entropy | margin | Correct Action |
|---|---|---|---|---|
| Confident and correct | 0.95 | 0.12 | 4.2 | Accept S1 |
| Confident but wrong | 0.92 | 0.18 | 3.8 | Route to S2 (but conf_raw says accept!) |
| Uniform uncertainty | 0.11 | 2.29 | 0.02 | Route to S2 |
| Bimodal (two likely classes) | 0.48 | 0.69 | 0.08 | Route to S2 |
| Peaked but fragile | 0.85 | 0.42 | 0.31 | Route to S2 (margin is small) |

The "confident but wrong" case is the critical failure mode. A model can assign
0.92 probability to the wrong class. The raw softmax maximum is high, so a
single-scalar router would accept the result. But the margin between the top-1 and
top-2 logits may be small (indicating a fragile decision), or the entropy over the
full distribution may reveal probability mass spread across many classes.

Using multiple proxies in combination makes routing more robust. The
MetacognitiveRouter (see [Section 6](#6-integration-with-routing)) learns a
nonlinear combination of all three signals.

### Proxy 1: Maximum Softmax Probability (conf_raw)

```
p = softmax(logits)              # (B, C)
conf_raw = p.max(dim=-1).values  # (B,)
```

**Properties:**
- Range: `[1/C, 1.0]`
- Monotonically related to the largest logit (given softmax is monotonic)
- Well-calibrated models produce `conf_raw` values that match empirical accuracy
  (i.e., among all inputs where `conf_raw ~ 0.8`, roughly 80% are correct)
- Modern deep networks are systematically overconfident -- `conf_raw` is typically
  too high. This is the primary motivation for post-hoc calibration.

**Known failure mode:** When the logit scale is inflated (e.g., due to training
with large learning rates or lack of weight decay), `conf_raw` saturates near 1.0
for almost all inputs, destroying its discriminative value as a routing signal.
Temperature scaling directly addresses this.

### Proxy 2: Predictive Entropy

```
p = softmax(logits)                             # (B, C)
log_p = torch.log(p + 1e-8)                     # Numerical stability
entropy = -(p * log_p).sum(dim=-1)              # (B,)
```

**Properties:**
- Range: `[0, log(C)]`
- `entropy = 0` when all probability mass is on one class (maximum certainty)
- `entropy = log(C)` when the distribution is uniform (maximum uncertainty)
- Captures the full distributional shape, not just the mode

**Normalization:** For routing purposes, normalize entropy to `[0, 1]`:
```
entropy_norm = entropy / math.log(num_classes)
```
This makes the threshold independent of the number of classes.

**Interpretation table:**

| entropy_norm | Interpretation |
|---|---|
| 0.00 - 0.10 | Extremely confident (near-degenerate distribution) |
| 0.10 - 0.30 | Confident (one dominant class) |
| 0.30 - 0.50 | Moderate uncertainty (2-3 plausible classes) |
| 0.50 - 0.70 | High uncertainty (many plausible classes) |
| 0.70 - 1.00 | Near-uniform (essentially guessing) |

### Proxy 3: Logit Margin

```
top2 = logits.topk(2, dim=-1).values            # (B, 2)
margin = top2[:, 0] - top2[:, 1]                 # (B,)
```

**Properties:**
- Range: `[0, +inf)` (in practice, bounded by logit scale)
- Measures the distance from the decision boundary between the two most likely
  classes
- High margin means the top prediction is well-separated from competitors
- Low margin means a small perturbation could flip the prediction

**Why margin complements conf_raw:** Softmax compresses large logit differences
into near-1.0 probabilities. Two predictions with `conf_raw = 0.95` may have
very different margins:
```
Prediction A: logits = [10.0, 7.0, ...]  -> conf_raw ~ 0.95, margin = 3.0
Prediction B: logits = [3.5,  3.2, ...]  -> conf_raw ~ 0.57, margin = 0.3
```
After temperature scaling (T=3), Prediction A remains confident but Prediction B
collapses. The margin reveals this fragility directly, without needing calibration.

**Normalization:** Margin has no natural upper bound. For routing, either:
- Apply a sigmoid: `margin_norm = sigmoid(margin - margin_threshold)`
- Use a running z-score: `margin_norm = (margin - mu) / sigma` with EMA statistics

The sigmoid approach is simpler and preferred for initial implementation.

### Dedicated Learned Confidence Head

In addition to the three distribution-based proxies, an optional dedicated
confidence head predicts confidence directly from the hidden representation:

```python
class ConfidenceHead(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, h: Tensor) -> Tensor:
        # h: (B, H) -- penultimate hidden state, DETACHED
        return self.net(h).squeeze(-1)  # (B,)
```

**Critical: detach the hidden state.** The confidence head receives `h.detach()`
so that the confidence loss does not flow back into the prediction pathway. Without
detachment, the model learns to inflate logits to make the confidence head happy,
rather than learning to predict accuracy.

**Training the confidence head:** The confidence head is trained with a binary
cross-entropy loss where the target is whether the prediction was correct:

```
correct = (logits.argmax(-1) == labels).float()  # (B,)
conf_loss = F.binary_cross_entropy(conf_learned, correct)
```

This loss is added to the total loss with a small weight (e.g., 0.1) so it does
not dominate training.

**When to enable:** The dedicated confidence head is most useful when:
- The logit distribution is unreliable (e.g., many classes with similar logits)
- The hidden representation contains information about difficulty that the logits
  do not capture (e.g., the model "knows" the input is out-of-distribution even
  before the final projection)
- The task involves regression (where softmax-based proxies do not apply)

### Combined Uncertainty Score

The router receives all proxies as a vector, not a single combined scalar. However,
for logging and simple thresholding, a combined score is useful:

```
uncertainty_combined = w1 * (1 - conf_calibrated) + w2 * entropy_norm + w3 * (1 - margin_norm)
```

Default weights: `w1 = 0.5, w2 = 0.3, w3 = 0.2`. These can be tuned on a
validation set. The combined score is in `[0, 1]` where higher means more uncertain.

**Do not use the combined score as the sole routing signal.** It discards
information. The router should receive the full vector `[conf_calibrated, entropy,
margin, conf_learned]` and learn its own combination. The combined score is for
human-readable logging only.

---

## 3. Calibration Utilities

### Why Calibrate

Modern deep neural networks produce poorly calibrated probabilities. Guo et al.
(2017) demonstrated that increased depth, width, batch normalization, and weight
decay all affect calibration -- and the net effect in modern architectures is
systematic overconfidence.

For the dual-process routing system, miscalibration is catastrophic:
- Overconfident System 1 -> never routes to System 2 -> misses hard cases
- Underconfident System 1 -> always routes to System 2 -> defeats the purpose
  of having a fast path

Post-hoc calibration fixes the probability scale without changing the model's
discriminative performance (accuracy, ranking). It is a monotonic transformation
of the logits.

### 3.1 TemperatureScaler

Temperature scaling is the simplest and most effective post-hoc calibration
method. It divides all logits by a single learned scalar `T > 0`:

```
calibrated_logits = logits / T
calibrated_probs = softmax(calibrated_logits)
```

**Effect of T on the distribution:**

| T value | Effect | When it helps |
|---|---|---|
| T < 1.0 | Sharpens distribution (more confident) | Underconfident model |
| T = 1.0 | No change (identity) | Already calibrated |
| T > 1.0 | Softens distribution (less confident) | Overconfident model |
| T -> inf | Uniform distribution | -- (degenerate) |

Most modern networks need `T > 1.0` (softening), typically in the range
`[1.2, 2.5]`.

**Key property:** Temperature scaling preserves the argmax. The predicted class
does not change, only the confidence. This means accuracy is unaffected by
calibration.

#### Implementation

```python
class TemperatureScaler(nn.Module):
    """
    Post-hoc temperature scaling for logit calibration.

    Reference: Guo et al. 2017, "On Calibration of Modern Neural Networks"
    """

    def __init__(self, initial_temperature: float = 1.5):
        super().__init__()
        # Store log(T) so that T = exp(log_T) is always positive
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(initial_temperature))
        )
        self._fitted = False

    @property
    def temperature(self) -> float:
        return self.log_temperature.exp().item()

    def fit(
        self,
        logits: Tensor,       # (N, C) -- collected logits from validation set
        labels: Tensor,       # (N,)   -- ground truth labels
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> dict:
        """
        Optimize temperature T to minimize NLL on validation data.

        Returns a dict with fitting diagnostics:
        {
            "temperature": float,
            "nll_before": float,
            "nll_after": float,
            "ece_before": float,
            "ece_after": float,
            "num_samples": int,
        }
        """
        ...

    def calibrate(self, logits: Tensor) -> Tensor:
        """Apply temperature scaling: logits / T."""
        T = self.log_temperature.exp()
        return logits / T

    def freeze(self):
        """Lock temperature -- no gradient updates."""
        self.log_temperature.requires_grad_(False)
        self._fitted = True

    def unfreeze(self):
        """Unlock temperature for re-fitting."""
        self.log_temperature.requires_grad_(True)
        self._fitted = False

    def state_dict_extra(self) -> dict:
        """Extra state for serialization."""
        return {
            "temperature": self.temperature,
            "fitted": self._fitted,
        }
```

#### Fitting Procedure

The fitting procedure optimizes `T` to minimize the negative log-likelihood (NLL)
on the validation set. NLL is the proper scoring rule that, when minimized, produces
calibrated probabilities.

```
NLL = - (1/N) * sum_i log softmax(logits_i / T)[labels_i]
```

Use L-BFGS (or Adam with a low learning rate) to find the optimal `T`. The
optimization landscape is convex in `log(T)` for the NLL objective, so convergence
is fast -- typically 20-50 iterations suffice.

**Step-by-step fitting:**

1. Collect logits and labels from the validation set. Do not use the training set
   (overfitting risk). Do not use the test set (leakage risk). A held-out
   calibration split of 5-10% of the validation set is ideal.

2. Initialize `log_temperature = log(1.5)`. Starting slightly above 1.0 biases
   toward softening, which is the common case for overconfident networks.

3. Set up the optimizer:
   ```python
   optimizer = torch.optim.LBFGS(
       [self.log_temperature],
       lr=lr,
       max_iter=max_iter,
   )
   ```

4. Define the closure:
   ```python
   def closure():
       optimizer.zero_grad()
       T = self.log_temperature.exp()
       scaled_logits = logits / T
       loss = F.cross_entropy(scaled_logits, labels)
       loss.backward()
       return loss
   ```

5. Run the optimizer:
   ```python
   optimizer.step(closure)
   ```

6. Freeze the parameter:
   ```python
   self.freeze()
   ```

7. Compute and return diagnostics (NLL before/after, ECE before/after).

#### Parameterization Choice: log(T) vs T

Store `log(T)` as the learnable parameter, not `T` directly. This ensures
`T = exp(log_T)` is always positive without requiring constrained optimization.
Gradient flow through `exp` is smooth and well-conditioned for values in the
typical range.

Alternative: store `T` directly with a softplus or clamp. This works but adds
an unnecessary nonlinearity to the gradient path.

#### Serialization

The temperature scalar must be saved and loaded as part of the model checkpoint.
Include it in the state dict:

```python
# Save
checkpoint = {
    "model": model.state_dict(),
    "temperature_scaler": scaler.state_dict(),
    "temperature_scaler_extra": scaler.state_dict_extra(),
}

# Load
scaler.load_state_dict(checkpoint["temperature_scaler"])
extra = checkpoint["temperature_scaler_extra"]
if extra["fitted"]:
    scaler.freeze()
```

Always save the `_fitted` flag. If a checkpoint is loaded without a fitted
calibrator, the system must know to fall back to raw confidence with a warning.

### 3.2 IsotonicCalibrator (Optional)

Isotonic regression provides a non-parametric, monotonic mapping from raw
confidence to calibrated confidence. Unlike temperature scaling, which applies
a single global transformation, isotonic regression can correct different
confidence ranges independently.

#### When to Use

Temperature scaling assumes that a single division by `T` is sufficient to fix
calibration across the entire confidence range. This assumption fails when
calibration errors are non-uniform -- for example, when the model is overconfident
at high confidence levels but underconfident at low confidence levels.

```
Calibration error profile:

  Confidence  |  Temperature Scaling  |  Isotonic Regression
  0.0 - 0.3   |  Over-corrects       |  Correct
  0.3 - 0.7   |  About right         |  Correct
  0.7 - 1.0   |  Under-corrects      |  Correct
```

If the reliability diagram (see [Section 5](#5-reliability-diagrams)) shows
non-uniform residuals after temperature scaling, switch to isotonic regression.

#### Implementation

```python
class IsotonicCalibrator:
    """
    Non-parametric monotonic calibration via isotonic regression.

    Fallback implementation that does not require sklearn.
    Uses the pool adjacent violators algorithm (PAVA).
    """

    def __init__(self):
        self._x_knots: Optional[Tensor] = None  # Confidence breakpoints
        self._y_knots: Optional[Tensor] = None  # Calibrated values at breakpoints
        self._fitted = False

    def fit(
        self,
        conf_raw: Tensor,   # (N,) -- raw confidence values
        correct: Tensor,    # (N,) -- binary: 1 if prediction was correct, 0 otherwise
    ) -> dict:
        """
        Fit isotonic regression from conf_raw -> calibrated_conf.

        Uses the pool adjacent violators algorithm (PAVA):
        1. Sort samples by conf_raw.
        2. Initialize calibrated values as the sorted correct labels.
        3. Iteratively merge adjacent blocks that violate monotonicity,
           replacing them with their weighted average.
        4. Store the resulting piecewise constant function as (x_knots, y_knots).

        Returns fitting diagnostics dict.
        """
        ...

    def calibrate(self, conf_raw: Tensor) -> Tensor:
        """
        Apply isotonic calibration via piecewise linear interpolation.

        For each input confidence value, find the two nearest knots
        and linearly interpolate.
        """
        ...

    def state_dict(self) -> dict:
        return {
            "x_knots": self._x_knots,
            "y_knots": self._y_knots,
            "fitted": self._fitted,
        }

    def load_state_dict(self, state: dict):
        self._x_knots = state["x_knots"]
        self._y_knots = state["y_knots"]
        self._fitted = state["fitted"]
```

#### Pool Adjacent Violators Algorithm (PAVA)

The PAVA is the standard algorithm for isotonic regression. It runs in `O(N)`
time after an `O(N log N)` sort.

```
Algorithm:
  1. Sort (conf_raw, correct) pairs by conf_raw ascending
  2. Initialize blocks: each sample is its own block with value = correct_i
  3. Scan left to right:
     - If block[i].value > block[i+1].value (monotonicity violation):
       - Merge blocks i and i+1
       - New block value = weighted average of merged blocks
       - Continue checking backwards (the merge may create new violations)
  4. Result: a piecewise constant monotonically non-decreasing function
```

To produce a smooth calibration curve, convert the piecewise constant function
to a piecewise linear function by using block midpoints as knots.

#### Piecewise Linear Interpolation

Given fitted knots `(x_k, y_k)` for `k = 0, ..., M-1`, calibrate a new
confidence value `c` by:

1. Find the interval: `x_{k} <= c < x_{k+1}`
2. Interpolate: `cal(c) = y_k + (c - x_k) * (y_{k+1} - y_k) / (x_{k+1} - x_k)`
3. Clamp to `[0, 1]`

For values below `x_0`, use `y_0`. For values above `x_{M-1}`, use `y_{M-1}`.

This can be vectorized using `torch.searchsorted`:

```python
def calibrate(self, conf_raw: Tensor) -> Tensor:
    idx = torch.searchsorted(self._x_knots, conf_raw.clamp(0, 1)) - 1
    idx = idx.clamp(0, len(self._x_knots) - 2)
    x_lo = self._x_knots[idx]
    x_hi = self._x_knots[idx + 1]
    y_lo = self._y_knots[idx]
    y_hi = self._y_knots[idx + 1]
    t = (conf_raw - x_lo) / (x_hi - x_lo + 1e-8)
    return (y_lo + t * (y_hi - y_lo)).clamp(0, 1)
```

#### No sklearn Dependency

The implementation must not depend on `sklearn.isotonic.IsotonicRegression`.
The PAVA is simple enough to implement in pure PyTorch. This avoids adding
a large dependency for a small utility and ensures the calibrator works in
environments where sklearn is not available.

If sklearn is available and the user opts in, a thin wrapper can delegate to
`sklearn.isotonic.IsotonicRegression` for numerical robustness. But the default
path must be dependency-free.

---

## 4. Calibration Fitting Protocol

### Overview

Calibration fitting is a post-training procedure. It must not be interleaved with
model training. The protocol is strict:

```
Phase 1: Train System 1 (normal training loop)
    |
Phase 2: Freeze System 1 weights (no gradient updates)
    |
Phase 3: Collect logits + labels on validation set (forward pass only)
    |
Phase 4: Fit calibrator (TemperatureScaler or IsotonicCalibrator)
    |
Phase 5: Freeze calibrator (lock T or knots)
    |
Phase 6: Deploy with frozen calibrator
```

### Step 1: Train System 1 Normally

Train the System 1 module as part of the full BrainAI pipeline. The loss includes:
- Task loss (cross-entropy for classification)
- Confidence head loss (optional, BCE against correctness)
- Any auxiliary losses from other modules

During training, the uncalibrated `conf_raw` is used for routing (with a lenient
threshold, e.g., 0.5). The calibrator is not active during training.

### Step 2: Freeze System 1 Weights

After training converges, freeze all System 1 parameters:

```python
for param in system1.parameters():
    param.requires_grad_(False)
system1.eval()
```

This is essential. If System 1 weights change after calibration, the calibration
becomes invalid. Temperature `T` was optimized for a specific set of logit
statistics -- if those statistics shift, `T` is no longer optimal.

### Step 3: Collect Logits on Validation Set

Run the frozen System 1 on the validation set (or a dedicated calibration split)
and collect all logits and labels:

```python
all_logits = []
all_labels = []

with torch.no_grad():
    for batch in val_loader:
        x, labels = batch
        result = system1(x)
        all_logits.append(result.y1.cpu())
        all_labels.append(labels.cpu())

all_logits = torch.cat(all_logits, dim=0)  # (N, C)
all_labels = torch.cat(all_labels, dim=0)  # (N,)
```

**Sample size requirements:**

| Calibration Method | Minimum Samples | Recommended Samples |
|---|---|---|
| TemperatureScaler | 1,000 | 5,000 - 10,000 |
| IsotonicCalibrator | 5,000 | 20,000+ |

Isotonic regression needs more samples because it is non-parametric -- each bin
needs sufficient data to estimate the calibrated probability reliably.

### Step 4: Fit the Calibrator

```python
# Option A: Temperature scaling (default)
scaler = TemperatureScaler(initial_temperature=1.5)
diagnostics = scaler.fit(all_logits, all_labels)
print(f"Fitted T = {diagnostics['temperature']:.3f}")
print(f"ECE: {diagnostics['ece_before']:.4f} -> {diagnostics['ece_after']:.4f}")

# Option B: Isotonic regression (if temperature scaling insufficient)
iso_cal = IsotonicCalibrator()
conf_raw = F.softmax(all_logits, dim=-1).max(dim=-1).values
correct = (all_logits.argmax(-1) == all_labels).float()
diagnostics = iso_cal.fit(conf_raw, correct)
```

**Decision criterion for choosing the calibration method:**

1. Fit temperature scaling first.
2. Compute the reliability diagram (Section 5).
3. If the maximum per-bin residual after temperature scaling exceeds 0.05
   (5% absolute calibration error in any bin), try isotonic regression.
4. If isotonic regression does not improve ECE by at least 10% relative,
   stick with temperature scaling (simpler, more robust to distribution shift).

### Step 5: Freeze the Calibrator

```python
scaler.freeze()  # Locks log_temperature, sets _fitted = True
```

After freezing, the calibrator performs inference only. No gradient computation
occurs for the temperature parameter during deployment.

### Why Not Online Recalibration

It may seem appealing to continuously update the temperature during deployment as
new data arrives. Do not do this. Online recalibration causes **routing drift**:

```
Cycle:
  1. T changes -> confidence changes
  2. Confidence changes -> routing threshold triggers differently
  3. Different routing -> different inputs reach System 2
  4. Different System 2 inputs -> different overall system behavior
  5. Different behavior -> different error patterns
  6. Different errors -> T needs to change again (back to step 1)
```

This feedback loop makes the system non-stationary and unpredictable. The
calibrator must be fitted once and frozen.

If the data distribution shifts significantly (detected via monitoring -- see
[Section 5](#5-reliability-diagrams)), re-run the full calibration protocol
from Step 2 with fresh validation data from the new distribution. Treat this as
a model update, not an online adaptation.

### Calibration During Multi-Phase Training

The BrainAI training pipeline has 7 phases. System 1 calibration fits into this
as follows:

| Phase | System 1 Status | Calibrator Status |
|---|---|---|
| Phase 1 (SNN Core) | Not yet instantiated | N/A |
| Phase 2 (Encoders) | Not yet instantiated | N/A |
| Phase 3 (HTM) | Not yet instantiated | N/A |
| Phase 4 (Workspace) | Instantiated, training | Inactive, conf_raw used |
| Phase 5 (Active Inference) | Training continues | Inactive, conf_raw used |
| Phase 6 (Reasoning) | Training continues | Inactive, conf_raw used |
| Phase 7 (Meta-Learning) | Frozen for calibration | **Fit and freeze here** |
| Deployment | Frozen | Frozen |

Calibration happens at the boundary between Phase 7 and deployment. If the
system undergoes fine-tuning after Phase 7, re-calibrate.

### Handling Missing Calibration

During phases 4-6, the calibrator is not yet fitted. The system must handle this
gracefully:

```python
if self.calibrator is not None and self.calibrator._fitted:
    conf_calibrated = F.softmax(
        self.calibrator.calibrate(logits), dim=-1
    ).max(dim=-1).values
else:
    conf_calibrated = conf_raw
    if not self._warned_uncalibrated:
        logger.warning(
            "TemperatureScaler not fitted. Using raw confidence for routing. "
            "This is expected during training phases 4-6."
        )
        self._warned_uncalibrated = True
```

The warning fires once to avoid log spam. The system continues to function with
raw confidence -- routing will be less optimal but not broken.

---

## 5. Reliability Diagrams

### Expected Calibration Error (ECE)

ECE is the primary scalar metric for calibration quality. It measures the
weighted average absolute difference between predicted confidence and empirical
accuracy across confidence bins.

**Formula:**

```
ECE = sum_{b=1}^{B} (n_b / N) * |accuracy_b - confidence_b|
```

Where:
- `B` = number of bins (default: 15)
- `n_b` = number of samples in bin `b`
- `N` = total number of samples
- `accuracy_b` = fraction of correct predictions in bin `b`
- `confidence_b` = mean predicted confidence in bin `b`

**Binning:** Use 15 equal-width bins over `[0, 1]`. Bin `b` contains all samples
with confidence in `[(b-1)/B, b/B)`. The last bin includes the right endpoint.

**Interpretation:**

| ECE Value | Calibration Quality |
|---|---|
| < 0.02 | Excellent |
| 0.02 - 0.05 | Good |
| 0.05 - 0.10 | Acceptable |
| 0.10 - 0.20 | Poor |
| > 0.20 | Very poor (likely no calibration applied) |

Modern uncalibrated networks typically have ECE in the 0.05-0.15 range. After
temperature scaling, ECE typically drops to 0.01-0.03.

### Maximum Calibration Error (MCE)

MCE captures the worst-case calibration error across bins:

```
MCE = max_{b=1}^{B} |accuracy_b - confidence_b|
```

MCE is important for safety-critical routing: even if the average calibration is
good, a single badly calibrated region can cause systematic routing failures.

**Target:** MCE < 0.05 after calibration. If MCE > 0.10 in any non-empty bin,
investigate that confidence range for systematic errors.

### Adaptive Calibration Error (ACE)

An alternative to ECE that uses adaptive binning (equal number of samples per bin
rather than equal width). More robust when the confidence distribution is
non-uniform (e.g., most samples have high confidence):

```
ACE = (1/R) * sum_{r=1}^{R} |accuracy_r - confidence_r|
```

Where each adaptive bin `r` contains exactly `N/R` samples, sorted by confidence.

Use ACE when the confidence distribution is highly skewed. Otherwise, standard ECE
with equal-width bins is sufficient.

### Computing the Reliability Diagram

```python
def compute_reliability_diagram(
    confidences: Tensor,  # (N,) -- predicted confidence
    correctness: Tensor,  # (N,) -- binary: 1 if correct, 0 if wrong
    num_bins: int = 15,
) -> dict:
    """
    Compute reliability diagram statistics.

    Returns:
    {
        "bin_boundaries": Tensor,   # (num_bins + 1,) -- bin edges
        "bin_accuracies": Tensor,   # (num_bins,) -- empirical accuracy per bin
        "bin_confidences": Tensor,  # (num_bins,) -- mean confidence per bin
        "bin_counts": Tensor,       # (num_bins,) -- samples per bin
        "ece": float,               # Expected Calibration Error
        "mce": float,               # Maximum Calibration Error
    }
    """
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_accuracies = torch.zeros(num_bins)
    bin_confidences = torch.zeros(num_bins)
    bin_counts = torch.zeros(num_bins, dtype=torch.long)

    for i in range(num_bins):
        lo = bin_boundaries[i]
        hi = bin_boundaries[i + 1]
        if i == num_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        bin_counts[i] = mask.sum()
        if bin_counts[i] > 0:
            bin_accuracies[i] = correctness[mask].float().mean()
            bin_confidences[i] = confidences[mask].mean()

    # ECE: weighted average of per-bin absolute errors
    N = confidences.shape[0]
    weights = bin_counts.float() / N
    abs_errors = (bin_accuracies - bin_confidences).abs()
    ece = (weights * abs_errors).sum().item()

    # MCE: maximum per-bin error (only non-empty bins)
    non_empty = bin_counts > 0
    mce = abs_errors[non_empty].max().item() if non_empty.any() else 0.0

    return {
        "bin_boundaries": bin_boundaries,
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
        "ece": ece,
        "mce": mce,
    }
```

### Logging to Telemetry

Log the following calibration metrics during validation:

```python
telemetry.log({
    "calibration/ece": ece,
    "calibration/mce": mce,
    "calibration/temperature": scaler.temperature,
    "calibration/mean_confidence": confidences.mean().item(),
    "calibration/mean_accuracy": correctness.float().mean().item(),
    "calibration/overconfidence_ratio": (
        (confidences > correctness.float()).float().mean().item()
    ),
})
```

The `overconfidence_ratio` is the fraction of samples where predicted confidence
exceeds actual accuracy. For a perfectly calibrated model, this should be
approximately 0.5 (half overconfident, half underconfident). Values significantly
above 0.5 indicate systematic overconfidence.

### Visual Reliability Diagram

For debugging and reporting, plot the reliability diagram:

```
Reliability Diagram (ASCII sketch):

  1.0 |                                    /
      |                                  /
  0.8 |                               ../
      |                            ..   /
  0.6 |                        ...    /
      |                     ...      /
  0.4 |                  ..         /
      |              ...           /
  0.2 |          ..               /
      |      ..                  /
  0.0 |___._____________________/________
      0.0  0.2  0.4  0.6  0.8  1.0
           Predicted Confidence

  Diagonal = perfect calibration
  Dots above diagonal = underconfident
  Dots below diagonal = overconfident (common case)
```

In the codebase, use matplotlib if available, otherwise log the bin statistics
as a table.

### Monitoring Calibration Drift

After deployment, periodically re-evaluate calibration on fresh labeled data.
If ECE increases by more than 50% relative to the calibration-time ECE, trigger
a recalibration alert:

```python
calibration_time_ece = 0.02  # Recorded at calibration time
current_ece = compute_ece(new_confidences, new_correctness)

if current_ece > calibration_time_ece * 1.5:
    logger.warning(
        f"Calibration drift detected: ECE {current_ece:.4f} "
        f"exceeds 1.5x calibration-time ECE {calibration_time_ece:.4f}. "
        f"Consider recalibrating."
    )
```

---

## 6. Integration with Routing

### MetacognitiveRouter Interface

The `MetacognitiveRouter` is the component that decides whether to accept the
System 1 result or invoke System 2. It receives the full `System1Result` and
produces a binary routing decision plus a routing confidence.

```python
class MetacognitiveRouter(nn.Module):
    def __init__(
        self,
        feature_dim: int = 4,      # Number of uncertainty features
        hidden_dim: int = 32,      # Small routing MLP
        threshold: float = 0.7,    # Default confidence threshold
        learnable_threshold: bool = True,
    ):
        ...

    def forward(self, result: System1Result) -> RoutingDecision:
        ...
```

```
@dataclass
class RoutingDecision:
    route_to_s2: Tensor      # (B,) -- bool: True = invoke System 2
    routing_confidence: Tensor  # (B,) -- how confident the router is in its decision
    features_used: dict        # For logging
```

### Feature Vector Construction

The router receives a feature vector constructed from the System1Result:

```python
def build_routing_features(result: System1Result) -> Tensor:
    """
    Construct the feature vector for the MetacognitiveRouter.

    Returns: (B, 4) tensor
    """
    features = torch.stack([
        result.conf_calibrated,                    # Primary signal
        result.entropy / math.log(num_classes),    # Normalized entropy
        torch.sigmoid(result.margin - 1.0),        # Normalized margin
        result.uncertainty_metrics.get(
            "conf_learned", result.conf_calibrated  # Learned conf (fallback)
        ),
    ], dim=-1)  # (B, 4)
    return features
```

### Routing Logic

The router is a small 2-layer MLP that maps the 4D feature vector to a routing
score:

```python
routing_score = sigmoid(router_mlp(features))  # (B,)
route_to_s2 = routing_score < threshold        # (B,) bool
```

The threshold is either fixed at 0.7 or learned as a parameter. A learnable
threshold is implemented as:

```python
self.threshold_logit = nn.Parameter(torch.tensor(0.847))  # sigmoid(0.847) ~ 0.7
threshold = torch.sigmoid(self.threshold_logit)
```

### conf_calibrated as Primary Signal

The calibrated confidence is the most informative single feature because:

1. It has been post-hoc adjusted to reflect true accuracy probabilities
2. It accounts for the full logit distribution (via softmax)
3. It is on a consistent scale `[0, 1]` with a meaningful interpretation

If only one feature were allowed, `conf_calibrated` alone would suffice. The
other features add robustness against edge cases where calibration is imperfect.

### entropy and margin as Secondary Signals

Entropy detects distributional uncertainty that `conf_calibrated` may miss:
- A nearly uniform distribution over 100 classes has `conf_raw ~ 0.01` but
  `entropy_norm ~ 1.0`. Both signals agree -- route to S2.
- A bimodal distribution (two classes at 0.45 each, rest near 0) has
  `conf_raw ~ 0.45` and `entropy_norm ~ 0.15`. Entropy says "fairly certain
  (low entropy)" but confidence says "uncertain (low conf_raw)." The router
  resolves this disagreement.

Margin detects decision-boundary proximity:
- Two predictions with identical `conf_raw = 0.6` may have very different margins.
  A margin of 0.1 means the decision is fragile; a margin of 2.0 means it is robust.

### Fallback When Calibrator Is Not Fitted

During training phases 4-6, the calibrator is not yet available. The router must
still function:

```python
if result.conf_calibrated is None or not calibrator_fitted:
    # Fall back to raw confidence
    features[..., 0] = result.conf_raw
    # Adjust threshold to be more conservative (raw conf is overconfident)
    effective_threshold = min(threshold + 0.1, 0.95)
```

The threshold adjustment (+0.1) compensates for the expected overconfidence of
raw softmax probabilities. This is a heuristic -- after calibration is fitted, the
normal threshold applies.

### Routing During Training vs Inference

| Aspect | Training | Inference |
|---|---|---|
| Routing | Soft (Gumbel-softmax or straight-through) | Hard (threshold) |
| Both paths | Always run (for loss computation) | Only selected path |
| Threshold | May use exploration (epsilon-greedy) | Fixed threshold |
| Gradient | Flows through routing decision | No gradient |

During training, both System 1 and System 2 are run for every input (at least
conceptually -- see the implementation in the DualProcessReasoning module for
efficiency tricks). The routing decision determines how the outputs are combined,
but both paths contribute to the loss. This ensures System 2 continues to improve
even on "easy" inputs.

During inference, only the selected path runs. If the router says "accept S1,"
System 2 is never invoked. This is the latency savings.

### Routing Statistics to Monitor

Log the following per-batch during training and per-epoch during validation:

```python
routing_stats = {
    "routing/s2_fraction": route_to_s2.float().mean().item(),
    "routing/mean_confidence": result.conf_calibrated.mean().item(),
    "routing/mean_entropy": result.entropy.mean().item(),
    "routing/mean_margin": result.margin.mean().item(),
    "routing/threshold": threshold,
    "routing/s1_accuracy_when_accepted": ...,  # Accuracy on S1-accepted inputs
    "routing/s2_accuracy_when_invoked": ...,   # Accuracy on S2-invoked inputs
}
```

Key health checks:
- `s2_fraction` should be 20-40% during training. If < 5%, the threshold is too
  lenient (S1 is overconfident). If > 80%, the threshold is too strict or S1 is
  undertrained.
- `s1_accuracy_when_accepted` should exceed `s2_accuracy_when_invoked`. If not,
  the router is making worse decisions than random.
- `mean_confidence` should track `s1_accuracy_when_accepted` within 5% (calibration
  check).

---

## 7. Code Examples

### 7.1 System1Fast -- Full Implementation

```python
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class System1Result:
    """Structured output from the System 1 fast predictor."""
    y1: torch.Tensor                    # (B, C) raw logits
    conf_raw: torch.Tensor              # (B,)   max softmax probability
    conf_calibrated: torch.Tensor       # (B,)   calibrated confidence
    entropy: torch.Tensor               # (B,)   predictive entropy
    margin: torch.Tensor                # (B,)   top1 - top2 logit gap
    uncertainty_metrics: dict = field(default_factory=dict)


class System1Fast(nn.Module):
    """
    Fast single-pass predictor (System 1) with confidence estimation.

    Produces task logits and multiple uncertainty proxies in a single
    forward pass. Designed to be cheap enough to run on every input.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        confidence_head: bool = True,
        dropout: float = 0.1,
        slot_pooling: str = "mean",
        num_slots: int = 0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_slots = num_slots

        # --- Pooling head (for slot input) ---
        if num_slots > 0:
            if slot_pooling == "attention":
                self.pool_query = nn.Parameter(torch.randn(1, input_dim))
                nn.init.xavier_uniform_(self.pool_query.unsqueeze(0))
            elif slot_pooling == "cls":
                self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))
                self.pool_attn_layer = nn.TransformerEncoderLayer(
                    d_model=input_dim,
                    nhead=8,
                    dim_feedforward=input_dim * 2,
                    dropout=dropout,
                    batch_first=True,
                )
            # "mean" pooling needs no parameters
        self.slot_pooling = slot_pooling

        # --- Prediction MLP ---
        layers = []
        current_dim = input_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()

        # Final projection to logits
        proj_input_dim = hidden_dim if num_layers > 1 else input_dim
        self.output_proj = nn.Linear(proj_input_dim, output_dim)

        # --- Confidence head (optional) ---
        self.has_confidence_head = confidence_head
        if confidence_head:
            conf_input_dim = hidden_dim if num_layers > 1 else input_dim
            self.conf_mlp = nn.Sequential(
                nn.Linear(conf_input_dim, 128),
                nn.GELU(),
                nn.Linear(128, 1),
                nn.Sigmoid(),
            )

        # --- Calibrator (attached after fitting) ---
        self.calibrator: Optional[nn.Module] = None
        self._warned_uncalibrated = False

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if "conf" in name:
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif "output_proj" in name:
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                else:
                    nn.init.kaiming_normal_(
                        module.weight, nonlinearity="relu"
                    )
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def _pool_slots(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce (B, K, D) -> (B, D) via configured pooling strategy."""
        B, K, D = x.shape
        if self.slot_pooling == "mean":
            return x.mean(dim=1)
        elif self.slot_pooling == "attention":
            query = self.pool_query.unsqueeze(0).expand(B, -1, -1)  # (B, 1, D)
            attn_weights = torch.bmm(
                query, x.transpose(1, 2)
            ) / math.sqrt(D)  # (B, 1, K)
            attn_weights = F.softmax(attn_weights, dim=-1)
            return torch.bmm(attn_weights, x).squeeze(1)  # (B, D)
        elif self.slot_pooling == "cls":
            cls = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
            x_with_cls = torch.cat([cls, x], dim=1)  # (B, K+1, D)
            x_out = self.pool_attn_layer(x_with_cls)  # (B, K+1, D)
            return x_out[:, 0, :]  # (B, D)
        else:
            raise ValueError(f"Unknown slot_pooling: {self.slot_pooling}")

    def _compute_confidence_metrics(
        self, logits: torch.Tensor
    ) -> tuple:
        """
        Compute three uncertainty proxies from logits.

        Returns: (conf_raw, entropy, margin)
        """
        probs = F.softmax(logits, dim=-1)  # (B, C)

        # Maximum softmax probability
        conf_raw = probs.max(dim=-1).values  # (B,)

        # Shannon entropy
        log_probs = torch.log(probs + 1e-8)
        entropy = -(probs * log_probs).sum(dim=-1)  # (B,)

        # Logit margin (top1 - top2)
        top2_vals = logits.topk(2, dim=-1).values  # (B, 2)
        margin = top2_vals[:, 0] - top2_vals[:, 1]  # (B,)

        return conf_raw, entropy, margin

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> System1Result:
        # Step 1: Pool slots if needed
        if x.ndim == 3:
            x = self._pool_slots(x)
        elif x.ndim != 2:
            raise ValueError(
                f"Expected input of shape (B, D) or (B, K, D), got {x.shape}"
            )

        # Step 2: Concatenate context if provided
        if context is not None:
            x = torch.cat([x, context], dim=-1)

        # Step 3: MLP hidden layers
        h = self.mlp(x)  # (B, H) or (B, D) if num_layers == 1

        # Step 4: Task logits
        logits = self.output_proj(h)  # (B, C)

        # Step 5: Confidence proxies from logits
        conf_raw, entropy, margin = self._compute_confidence_metrics(logits)

        # Step 6: Learned confidence head
        uncertainty_metrics = {}
        if self.has_confidence_head:
            conf_learned = self.conf_mlp(h.detach()).squeeze(-1)  # (B,)
            uncertainty_metrics["conf_learned"] = conf_learned

        # Step 7: Temperature scaling (if calibrator fitted)
        if (
            self.calibrator is not None
            and hasattr(self.calibrator, "_fitted")
            and self.calibrator._fitted
        ):
            calibrated_logits = self.calibrator.calibrate(logits)
            conf_calibrated = F.softmax(
                calibrated_logits, dim=-1
            ).max(dim=-1).values
        else:
            conf_calibrated = conf_raw
            if not self._warned_uncalibrated:
                import logging
                logging.getLogger(__name__).warning(
                    "TemperatureScaler not fitted. "
                    "Using raw confidence for routing."
                )
                self._warned_uncalibrated = True

        # Step 8: Pack result
        uncertainty_metrics.update({
            "entropy_norm": entropy / math.log(max(self.output_dim, 2)),
            "margin_sigmoid": torch.sigmoid(margin - 1.0),
        })

        return System1Result(
            y1=logits,
            conf_raw=conf_raw,
            conf_calibrated=conf_calibrated,
            entropy=entropy,
            margin=margin,
            uncertainty_metrics=uncertainty_metrics,
        )
```

### 7.2 TemperatureScaler -- Full Implementation

```python
class TemperatureScaler(nn.Module):
    """
    Post-hoc temperature scaling for logit calibration.

    Optimizes a single scalar temperature T such that
    softmax(logits / T) produces well-calibrated probabilities.

    Reference:
        Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).
        On calibration of modern neural networks. ICML.
    """

    def __init__(self, initial_temperature: float = 1.5):
        super().__init__()
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(initial_temperature))
        )
        self._fitted = False

    @property
    def temperature(self) -> float:
        """Current temperature value (always positive)."""
        return self.log_temperature.exp().item()

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.

        Args:
            logits: (B, C) or (N, C) raw logits

        Returns:
            Calibrated logits: logits / T
        """
        T = self.log_temperature.exp()
        return logits / T

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> dict:
        """
        Fit temperature T to minimize NLL on validation data.

        Args:
            logits: (N, C) -- pre-collected logits from frozen model
            labels: (N,)   -- ground truth class labels
            lr: Learning rate for L-BFGS
            max_iter: Maximum L-BFGS iterations

        Returns:
            Dict with fitting diagnostics.
        """
        # Record pre-calibration metrics
        with torch.no_grad():
            nll_before = F.cross_entropy(logits, labels).item()
            probs_before = F.softmax(logits, dim=-1)
            conf_before = probs_before.max(dim=-1).values
            correct = (logits.argmax(-1) == labels).float()
            ece_before = self._compute_ece(conf_before, correct)

        # Reset temperature
        self.log_temperature.data = torch.tensor(math.log(1.5))
        self.log_temperature.requires_grad_(True)

        # Optimize with L-BFGS
        optimizer = torch.optim.LBFGS(
            [self.log_temperature], lr=lr, max_iter=max_iter
        )

        def closure():
            optimizer.zero_grad()
            T = self.log_temperature.exp()
            loss = F.cross_entropy(logits / T, labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        # Record post-calibration metrics
        with torch.no_grad():
            T = self.log_temperature.exp()
            calibrated_logits = logits / T
            nll_after = F.cross_entropy(calibrated_logits, labels).item()
            probs_after = F.softmax(calibrated_logits, dim=-1)
            conf_after = probs_after.max(dim=-1).values
            ece_after = self._compute_ece(conf_after, correct)

        # Freeze
        self.freeze()

        return {
            "temperature": self.temperature,
            "nll_before": nll_before,
            "nll_after": nll_after,
            "ece_before": ece_before,
            "ece_after": ece_after,
            "num_samples": logits.shape[0],
        }

    def _compute_ece(
        self,
        confidences: torch.Tensor,
        correctness: torch.Tensor,
        num_bins: int = 15,
    ) -> float:
        """Compute Expected Calibration Error."""
        bin_boundaries = torch.linspace(0, 1, num_bins + 1)
        ece = 0.0
        N = confidences.shape[0]

        for i in range(num_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
            if i == num_bins - 1:
                mask = (confidences >= lo) & (confidences <= hi)
            else:
                mask = (confidences >= lo) & (confidences < hi)
            n_bin = mask.sum().item()
            if n_bin > 0:
                acc_bin = correctness[mask].mean().item()
                conf_bin = confidences[mask].mean().item()
                ece += (n_bin / N) * abs(acc_bin - conf_bin)

        return ece

    def freeze(self):
        """Lock temperature -- no gradient updates during inference."""
        self.log_temperature.requires_grad_(False)
        self._fitted = True

    def unfreeze(self):
        """Unlock temperature for re-fitting."""
        self.log_temperature.requires_grad_(True)
        self._fitted = False

    def state_dict_extra(self) -> dict:
        """Extra serialization state beyond nn.Module.state_dict."""
        return {
            "temperature": self.temperature,
            "fitted": self._fitted,
        }
```

### 7.3 Confidence Computation -- All Three Metrics

Standalone function for computing confidence metrics outside of the System1Fast
module (e.g., for evaluation scripts):

```python
def compute_confidence_metrics(
    logits: torch.Tensor,
    calibrator: Optional[TemperatureScaler] = None,
    num_classes: Optional[int] = None,
) -> dict:
    """
    Compute all confidence/uncertainty metrics from logits.

    Args:
        logits: (B, C) raw logits
        calibrator: Optional fitted TemperatureScaler
        num_classes: Number of classes (for entropy normalization).
                     If None, inferred from logits.shape[-1].

    Returns:
        Dict with keys:
            conf_raw, conf_calibrated, entropy, entropy_norm,
            margin, margin_sigmoid, uncertainty_combined
    """
    C = num_classes or logits.shape[-1]

    # Softmax probabilities
    probs = F.softmax(logits, dim=-1)

    # Proxy 1: Maximum softmax probability
    conf_raw = probs.max(dim=-1).values

    # Proxy 2: Predictive entropy
    log_probs = torch.log(probs + 1e-8)
    entropy = -(probs * log_probs).sum(dim=-1)
    entropy_norm = entropy / math.log(max(C, 2))

    # Proxy 3: Logit margin
    top2 = logits.topk(min(2, C), dim=-1).values
    if C >= 2:
        margin = top2[:, 0] - top2[:, 1]
    else:
        margin = torch.zeros_like(conf_raw)
    margin_sigmoid = torch.sigmoid(margin - 1.0)

    # Calibrated confidence
    if calibrator is not None and calibrator._fitted:
        cal_logits = calibrator.calibrate(logits)
        conf_calibrated = F.softmax(cal_logits, dim=-1).max(dim=-1).values
    else:
        conf_calibrated = conf_raw

    # Combined uncertainty score
    w1, w2, w3 = 0.5, 0.3, 0.2
    uncertainty_combined = (
        w1 * (1 - conf_calibrated)
        + w2 * entropy_norm
        + w3 * (1 - margin_sigmoid)
    )

    return {
        "conf_raw": conf_raw,
        "conf_calibrated": conf_calibrated,
        "entropy": entropy,
        "entropy_norm": entropy_norm,
        "margin": margin,
        "margin_sigmoid": margin_sigmoid,
        "uncertainty_combined": uncertainty_combined,
    }
```

### 7.4 Reliability Diagram -- Full Computation

```python
def compute_reliability_diagram(
    confidences: torch.Tensor,
    correctness: torch.Tensor,
    num_bins: int = 15,
) -> dict:
    """
    Compute reliability diagram statistics, ECE, and MCE.

    Args:
        confidences: (N,) predicted confidence values in [0, 1]
        correctness: (N,) binary tensor (1 = correct, 0 = wrong)
        num_bins: Number of equal-width bins (default 15)

    Returns:
        Dict with bin statistics, ECE, and MCE.
    """
    assert confidences.shape == correctness.shape
    assert confidences.ndim == 1

    N = confidences.shape[0]
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_accuracies = torch.zeros(num_bins)
    bin_confidences = torch.zeros(num_bins)
    bin_counts = torch.zeros(num_bins, dtype=torch.long)

    for i in range(num_bins):
        lo = bin_boundaries[i]
        hi = bin_boundaries[i + 1]
        # Include right endpoint in last bin
        if i == num_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        count = mask.sum().item()
        bin_counts[i] = count
        if count > 0:
            bin_accuracies[i] = correctness[mask].float().mean()
            bin_confidences[i] = confidences[mask].mean()

    # ECE: weighted average of per-bin |accuracy - confidence|
    weights = bin_counts.float() / max(N, 1)
    abs_errors = (bin_accuracies - bin_confidences).abs()
    ece = (weights * abs_errors).sum().item()

    # MCE: max per-bin error (non-empty bins only)
    non_empty = bin_counts > 0
    if non_empty.any():
        mce = abs_errors[non_empty].max().item()
    else:
        mce = 0.0

    # Per-bin gap (signed): positive = underconfident, negative = overconfident
    gaps = bin_accuracies - bin_confidences

    return {
        "bin_boundaries": bin_boundaries,
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
        "bin_gaps": gaps,
        "ece": ece,
        "mce": mce,
        "num_samples": N,
        "num_non_empty_bins": non_empty.sum().item(),
    }


def format_reliability_table(diagram: dict) -> str:
    """
    Format reliability diagram as a human-readable table.

    Example output:
        Bin        | Count | Accuracy | Confidence | Gap
        [0.00,0.07)|    12 |    0.083 |      0.042 | +0.042
        [0.07,0.13)|     8 |    0.125 |      0.098 | +0.027
        ...
    """
    lines = []
    lines.append(
        f"{'Bin':>14s} | {'Count':>5s} | {'Accuracy':>8s} | "
        f"{'Confidence':>10s} | {'Gap':>7s}"
    )
    lines.append("-" * 60)

    boundaries = diagram["bin_boundaries"]
    num_bins = len(diagram["bin_counts"])

    for i in range(num_bins):
        lo = boundaries[i].item()
        hi = boundaries[i + 1].item()
        count = diagram["bin_counts"][i].item()
        acc = diagram["bin_accuracies"][i].item()
        conf = diagram["bin_confidences"][i].item()
        gap = diagram["bin_gaps"][i].item()

        bracket = ")" if i < num_bins - 1 else "]"
        bin_label = f"[{lo:.2f},{hi:.2f}{bracket}"
        gap_str = f"{gap:+.3f}" if count > 0 else "   N/A"

        if count > 0:
            lines.append(
                f"{bin_label:>14s} | {count:5d} | {acc:8.3f} | "
                f"{conf:10.3f} | {gap_str:>7s}"
            )
        else:
            lines.append(
                f"{bin_label:>14s} | {count:5d} |      N/A | "
                f"       N/A | {gap_str:>7s}"
            )

    lines.append("-" * 60)
    lines.append(f"ECE = {diagram['ece']:.4f}    MCE = {diagram['mce']:.4f}")
    lines.append(f"Total samples: {diagram['num_samples']}")
    lines.append(
        f"Non-empty bins: {diagram['num_non_empty_bins']} / {num_bins}"
    )

    return "\n".join(lines)
```

### 7.5 End-to-End Calibration Script

Complete script showing the full calibration protocol:

```python
"""
calibrate_system1.py -- Post-hoc calibration of System 1 confidence.

Usage:
    python calibrate_system1.py \
        --checkpoint path/to/model.pt \
        --val-data path/to/val/ \
        --output path/to/calibrated_model.pt \
        --method temperature  # or "isotonic"
"""

import argparse
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def calibrate_system1(
    model,
    val_loader: DataLoader,
    method: str = "temperature",
    device: str = "cuda",
) -> dict:
    """
    Full calibration protocol for System 1.

    Steps:
        1. Freeze model
        2. Collect logits on validation set
        3. Fit calibrator
        4. Freeze calibrator
        5. Return diagnostics

    Args:
        model: BrainAI model with .system1 attribute
        val_loader: Validation data loader
        method: "temperature" or "isotonic"
        device: Device to use

    Returns:
        Diagnostics dict
    """
    system1 = model.system1

    # --- Step 1: Freeze System 1 ---
    logger.info("Step 1: Freezing System 1 weights")
    for param in system1.parameters():
        param.requires_grad_(False)
    system1.eval()

    # --- Step 2: Collect logits ---
    logger.info("Step 2: Collecting logits on validation set")
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            # Run through encoders + workspace to get System 1 input
            workspace_out = model.encode_and_workspace(inputs)
            result = system1(workspace_out)

            all_logits.append(result.y1.cpu())
            all_labels.append(labels.cpu())

            if (batch_idx + 1) % 100 == 0:
                logger.info(
                    f"  Collected {sum(l.shape[0] for l in all_logits)} samples"
                )

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    logger.info(f"  Total: {all_logits.shape[0]} samples collected")

    # --- Step 3: Fit calibrator ---
    if method == "temperature":
        logger.info("Step 3: Fitting TemperatureScaler")
        calibrator = TemperatureScaler(initial_temperature=1.5)
        diagnostics = calibrator.fit(all_logits, all_labels)
        logger.info(f"  Fitted T = {diagnostics['temperature']:.4f}")
        logger.info(
            f"  ECE: {diagnostics['ece_before']:.4f} -> "
            f"{diagnostics['ece_after']:.4f}"
        )
        logger.info(
            f"  NLL: {diagnostics['nll_before']:.4f} -> "
            f"{diagnostics['nll_after']:.4f}"
        )

    elif method == "isotonic":
        logger.info("Step 3: Fitting IsotonicCalibrator")
        calibrator = IsotonicCalibrator()
        probs = F.softmax(all_logits, dim=-1)
        conf_raw = probs.max(dim=-1).values
        correct = (all_logits.argmax(-1) == all_labels).float()
        diagnostics = calibrator.fit(conf_raw, correct)

    else:
        raise ValueError(f"Unknown calibration method: {method}")

    # --- Step 4: Attach and freeze ---
    logger.info("Step 4: Attaching calibrator to System 1")
    system1.calibrator = calibrator

    # --- Step 5: Validate calibration ---
    logger.info("Step 5: Validating calibration")
    with torch.no_grad():
        if method == "temperature":
            cal_logits = calibrator.calibrate(all_logits)
            cal_probs = F.softmax(cal_logits, dim=-1)
            cal_conf = cal_probs.max(dim=-1).values
        else:
            probs = F.softmax(all_logits, dim=-1)
            cal_conf = calibrator.calibrate(probs.max(dim=-1).values)

        correct = (all_logits.argmax(-1) == all_labels).float()
        diagram = compute_reliability_diagram(cal_conf, correct)

    logger.info(f"  Final ECE: {diagram['ece']:.4f}")
    logger.info(f"  Final MCE: {diagram['mce']:.4f}")
    logger.info("\n" + format_reliability_table(diagram))

    diagnostics["reliability_diagram"] = diagram
    return diagnostics
```

---

## Appendix A: Hyperparameter Defaults

Summary of all hyperparameters with defaults and recommended ranges:

| Parameter | Default | Range | Notes |
|---|---|---|---|
| `input_dim` | 4096 | 256-8192 | Must match workspace output |
| `hidden_dim` | 2048 | 128-4096 | Typically `input_dim // 2` |
| `num_layers` | 2 | 1-3 | Beyond 3, use System 2 |
| `dropout` | 0.1 | 0.0-0.3 | Higher for small datasets |
| `confidence_head` | True | -- | Disable for regression tasks |
| `slot_pooling` | "mean" | "mean"/"attention"/"cls" | "attention" for heterogeneous slots |
| `initial_temperature` | 1.5 | 1.0-3.0 | Starting point for optimization |
| `calibration_lr` | 0.01 | 0.001-0.1 | L-BFGS learning rate |
| `calibration_max_iter` | 50 | 20-200 | L-BFGS iterations |
| `ece_num_bins` | 15 | 10-20 | Standard: 15 equal-width |
| `routing_threshold` | 0.7 | 0.5-0.9 | Higher = more conservative (more S2) |
| `conf_loss_weight` | 0.1 | 0.01-0.5 | Weight of confidence head BCE loss |
| `uncertainty_weights` | [0.5, 0.3, 0.2] | -- | [conf, entropy, margin] for combined score |

---

## Appendix B: Common Failure Modes

| Symptom | Likely Cause | Fix |
|---|---|---|
| ECE > 0.10 after calibration | Too few calibration samples | Collect more validation data (>5000) |
| T < 0.5 (extreme sharpening) | Model is underconfident (rare) | Check for label noise; verify training loss |
| T > 5.0 (extreme softening) | Model is severely overconfident | Check for overfitting; add weight decay |
| MCE > 0.15 in one bin | Non-uniform miscalibration | Switch to isotonic regression |
| S2 fraction = 0% | Threshold too low or S1 always confident | Raise threshold; check for logit saturation |
| S2 fraction = 100% | Threshold too high or S1 never confident | Lower threshold; check S1 training |
| conf_learned disagrees with conf_raw | Confidence head sees different signal | Expected -- use both as router features |
| Calibration drifts after deployment | Data distribution shift | Re-run calibration protocol on new data |
| Routing oscillates between S1/S2 | Online recalibration or unstable threshold | Freeze calibrator; use fixed threshold |

---

## Appendix C: References

1. **Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q.** (2017).
   On calibration of modern neural networks. *ICML*.
   -- Foundation for temperature scaling.

2. **Niculescu-Mizil, A., & Caruana, R.** (2005).
   Predicting good probabilities with supervised learning. *ICML*.
   -- Platt scaling and isotonic regression for calibration.

3. **Kahneman, D.** (2011).
   *Thinking, Fast and Slow*. Farrar, Straus and Giroux.
   -- Dual-process theory: System 1 (fast) vs System 2 (slow).

4. **Lakshminarayanan, B., Pritzel, A., & Blundell, C.** (2017).
   Simple and scalable predictive uncertainty estimation using deep ensembles.
   *NeurIPS*.
   -- Ensemble-based uncertainty; motivates multiple confidence proxies.

5. **Nixon, J., Dusenberry, M. W., Zhang, L., Jerfel, G., & Tran, D.** (2019).
   Measuring calibration in deep learning. *CVPR Workshops*.
   -- Adaptive Calibration Error (ACE) and binning strategies.

6. **Hendrycks, D., & Gimpel, K.** (2017).
   A baseline for detecting misclassified and out-of-distribution examples
   in neural networks. *ICLR*.
   -- Maximum softmax probability as an uncertainty baseline.

7. **DeVries, T., & Taylor, G. W.** (2018).
   Learning confidence for out-of-distribution detection in neural networks.
   *arXiv:1802.04865*.
   -- Learned confidence heads with detached gradients.

---

*End of reference document. Target audience: another Claude instance implementing
the dual-process reasoning skill for brain-ai-dev.*
