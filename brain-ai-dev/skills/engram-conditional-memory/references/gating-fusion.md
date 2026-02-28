# Context-Aware Gating and Residual Fusion

Reference document for the Engram Conditional Memory subsystem. Covers the full
pipeline from retrieved N-gram embeddings through context-aware gating, depthwise
causal convolution, and residual fusion into the host backbone hidden states.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Context-Aware Gating Architecture](#2-context-aware-gating-architecture)
3. [RMSNorm for Stability](#3-rmsnorm-for-stability)
4. [Gate Initialization](#4-gate-initialization)
5. [Gate Statistics for Telemetry](#5-gate-statistics-for-telemetry)
6. [Depthwise Causal Convolution](#6-depthwise-causal-convolution)
7. [Dilation Configuration](#7-dilation-configuration)
8. [Residual Fusion](#8-residual-fusion)
9. [AMP Safety Patterns](#9-amp-safety-patterns)
10. [Forward Pass Pipeline](#10-forward-pass-pipeline)
11. [Configuration Parameters](#11-configuration-parameters)
12. [Ablation Guidance](#12-ablation-guidance)
13. [Common Failure Modes](#13-common-failure-modes)
14. [Implementation Checklist](#14-implementation-checklist)

---

## 1. Overview

### The Static Prior Problem

Retrieved Engram embeddings are **static priors**. Their values depend only on
the input token IDs and the frozen hash-to-embedding lookup tables. They carry no
information about the current hidden state, the attention context accumulated from
prior layers, or the dynamic state of the transformer backbone. This means:

- A retrieved embedding for the bigram "New York" is the same regardless of
  whether the surrounding context is about geography, sports teams, or
  a person's name.
- Without gating, the model has no way to suppress irrelevant or contradictory
  retrievals.
- Without fusion, the model has no way to smoothly integrate retrieval deltas
  into its residual stream.

### Solution: Gate-Then-Convolve-Then-Fuse

The gating and fusion pipeline addresses this with three stages:

```
Stage 1: Context-Aware Gating
    Use the current hidden state (which has dynamic context) as a query
    and the retrieved embedding as key/value. Produce a scalar gate
    alpha in [0, 1] that modulates how much of the retrieved value
    passes through.

Stage 2: Depthwise Causal Convolution
    Apply a parameter-efficient causal convolution to the gated output.
    This expands the receptive field beyond single positions and adds
    local nonlinearity before the signal enters the residual stream.

Stage 3: Residual Fusion
    Project the convolved output to match the backbone hidden dimension
    and add it to the hidden states as a residual delta.
```

### Design Principles

| Principle | Rationale |
|---|---|
| Conservative initialization | Early in training, gates default to ~0.12 (sigmoid of -2.0), keeping Engram contributions small until the model learns when to trust retrievals |
| RMSNorm on query and key | Prevents scale mismatch between hidden states (which grow with layer depth) and retrieved embeddings (which are fresh from embedding tables) |
| Depthwise convolution | Parameter-efficient (groups = channels) local mixing that respects causality |
| Dilation tied to N-gram order | Ensures the convolution "sees" across the same temporal span as the N-gram context window |
| AMP-safe throughout | All reduction operations (norms, sums) forced to fp32 even when inputs are fp16/bf16 |

### Data Flow Summary

```
hidden_states  (B, T, D)
     |
     |   retrieved  (B, T, D_emb)
     |       |
     |       +--- Wk --> k (B, T, D_gate) --> RMSNorm_k --> k_norm
     |       |
     |       +--- Wv --> v (B, T, D_out)
     |
     +--- Wq --> q (B, T, D_gate) --> RMSNorm_q --> q_norm
     |
     |   alpha = sigmoid(sum(q_norm * k_norm, dim=-1))  --> (B, T, 1)
     |
     |   gated = alpha * v                              --> (B, T, D_out)
     |
     |   conv_out = SiLU(CausalConv1d(gated))           --> (B, T, D_out)
     |
     |   delta = Wo(conv_out)                            --> (B, T, D)
     |
     +--- output = hidden_states + scale * delta         --> (B, T, D)
```

---

## 2. Context-Aware Gating Architecture

### 2.1 Inputs

The gating module receives two tensors:

| Input | Shape | Source | Properties |
|---|---|---|---|
| `hidden_states` | `(B, T, D)` | Host backbone (e.g., transformer hidden layer) | Dynamic, context-dependent, dtype matches backbone |
| `retrieved` | `(B, T, D_emb)` | Aggregated embeddings from N-gram hash tables | Static (depends only on token IDs), potentially different scale |

Here `B` is batch size, `T` is sequence length, `D` is the backbone hidden
dimension, and `D_emb` is the total embedding dimension from the hash tables
(sum of per-head dimensions across all N-gram orders and heads).

### 2.2 Projection Matrices

Three linear projections transform the inputs into a shared gating space:

```python
# Query projection: hidden_states -> gating space
Wq = nn.Linear(D, D_gate, bias=False)         # (D, D_gate)

# Key projection: retrieved embeddings -> gating space
Wk = nn.Linear(D_emb, D_gate, bias=False)     # (D_emb, D_gate)

# Value projection: retrieved embeddings -> output space
Wv = nn.Linear(D_emb, D_out, bias=False)      # (D_emb, D_out)
```

| Projection | Input Dim | Output Dim | Purpose |
|---|---|---|---|
| `Wq` | `D` | `D_gate` | Maps hidden states into gating query space |
| `Wk` | `D_emb` | `D_gate` | Maps retrieved embeddings into gating key space |
| `Wv` | `D_emb` | `D_out` | Maps retrieved embeddings into the output value space |

`D_gate` is typically `D // 4` or 128 (the gate only produces a scalar, so
full dimensionality is wasteful). `D_out` typically equals `D`, with `Wo`
mapping back if `D_out < D`.

### 2.3 RMSNorm on Query and Key

After projection, RMSNorm is applied **independently** to the query and key:

```python
q = Wq(hidden_states)            # (B, T, D_gate)
k = Wk(retrieved)                # (B, T, D_gate)

q_norm = RMSNorm_q(q)            # (B, T, D_gate), learnable gamma
k_norm = RMSNorm_k(k)            # (B, T, D_gate), learnable gamma
```

The two RMSNorm instances have **independent learnable scale parameters** (gamma).
This is critical: `q` and `k` come from different source distributions (backbone
hidden states vs. embedding table lookups) and will have different magnitude
profiles.

See [Section 3](#3-rmsnorm-for-stability) for detailed RMSNorm specification.

### 2.4 Gate Computation

The gate scalar alpha is computed as the element-wise dot product of the
normalized query and key, summed across the feature dimension and passed through
sigmoid:

```python
# Element-wise product, then sum across D_gate
gate_logits = torch.sum(q_norm * k_norm, dim=-1, keepdim=True)  # (B, T, 1)

# Sigmoid squashes to [0, 1]
alpha = torch.sigmoid(gate_logits)                                # (B, T, 1)
```

The dot product measures alignment between query and key. RMSNorm ensures
bounded magnitude so sigmoid operates in its sensitive region. Equivalent
einsum notation:

```python
alpha = torch.sigmoid(
    torch.einsum('btd,btd->bt', q_norm, k_norm)
).unsqueeze(-1)  # (B, T, 1)
```

### 2.5 Gated Output

The gate modulates the value projection:

```python
v = Wv(retrieved)             # (B, T, D_out)
gated = alpha * v             # (B, T, D_out) -- broadcasting alpha over D_out
```

Positions where `alpha ~ 0` (retrieval contradicts or is irrelevant to context)
produce near-zero gated output. Positions where `alpha ~ 1` (retrieval is
contextually relevant) pass the value through essentially unchanged.

### 2.6 Per-Head Gating (Alternative)

For multi-head retrieved embeddings where each hash head produces a separate
embedding vector, per-head gating allows independent suppression of individual
heads:

```python
# Per-head variant: H separate gates
# q_norm: (B, T, H, D_gate_per_head)
# k_norm: (B, T, H, D_gate_per_head)

gate_logits = torch.sum(q_norm * k_norm, dim=-1)  # (B, T, H)
alpha = torch.sigmoid(gate_logits)                  # (B, T, H)

# v: (B, T, H, D_out_per_head)
gated = alpha.unsqueeze(-1) * v                     # (B, T, H, D_out_per_head)

# Flatten heads back to single vector
gated = gated.reshape(B, T, H * D_out_per_head)    # (B, T, D_out)
```

| Gate Type | Alpha Shape | Advantages | Disadvantages |
|---|---|---|---|
| `"scalar"` | `(B, T, 1)` | Simpler, fewer parameters, easier to interpret | All-or-nothing: cannot selectively trust some heads |
| `"per_head"` | `(B, T, H)` | Fine-grained: can suppress noisy heads while keeping good ones | More parameters, harder to interpret telemetry |

**Recommendation:** Start with `"scalar"` gating. Switch to `"per_head"` only if
ablation experiments show that some hash heads are consistently noisy while
others are consistently useful, and per-head gating improves downstream metrics.

### 2.7 Temperature Scaling (Optional)

An optional temperature parameter can sharpen or soften the gate:

```python
gate_logits = gate_logits / temperature
alpha = torch.sigmoid(gate_logits)
```

| Temperature | Effect |
|---|---|
| `< 1.0` | Sharper gates (more binary, closer to 0 or 1) |
| `= 1.0` | Default behavior |
| `> 1.0` | Softer gates (more uncertain, closer to 0.5) |

Temperature is typically fixed at 1.0 and not learned. If needed for
calibration, it can be made a learnable scalar initialized to 1.0.

---

## 3. RMSNorm for Stability

### 3.1 Definition

RMSNorm (Root Mean Square Layer Normalization) normalizes a vector by its
root-mean-square value:

```
RMSNorm(x) = (x / RMS(x)) * gamma

where:
    RMS(x) = sqrt(mean(x^2) + eps)
    gamma  = learnable scale parameter (per-dimension)
    eps    = stability constant (default 1e-6)
```

Compared to LayerNorm, RMSNorm omits mean-centering and learnable bias,
making it ~15% cheaper while performing comparably.

### 3.2 PyTorch Implementation

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Force fp32 for numerical stability under AMP
        input_dtype = x.dtype
        x = x.float()

        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms

        # Cast back and apply learnable scale
        return (x_normed * self.gamma).to(input_dtype)
```

### 3.3 Why RMSNorm on Query and Key

Query (from hidden states) and key (from retrieved embeddings) come from
different distributions: hidden state magnitude grows with layer depth due to
residual accumulation, while retrieved embedding magnitude is fixed by
initialization. Without normalization, the dot product `sum(q * k)` can
saturate sigmoid at 0 or 1, killing gradient flow through the gate. RMSNorm
brings both to unit RMS, bounding the dot product magnitude and keeping
sigmoid in its sensitive region.

### 3.4 Independent Normalization

The two RMSNorm instances have **separate learnable gamma parameters**
(`gamma_q` and `gamma_k`, each of shape `(D_gate,)`). This allows each to
learn different per-dimension scaling, so query and key dimensions that matter
most for gating decisions can be up-weighted independently.

### 3.5 Epsilon and Gamma

- **Epsilon:** `1e-6` default (sufficient for fp32 and bf16). Increase to `1e-5`
  for pure fp16. Never use `1e-8` (below fp16 minimum representable).
- **Learnable gamma:** Initialized to ones, learns per-dimension scaling that
  controls which dimensions contribute most to the gate decision. Dimensions
  with large `gamma_q[i] * gamma_k[i]` contribute strongly; small products
  are effectively ignored. This provides soft feature selection within gating.

### 3.6 Interaction with AMP

RMSNorm is the most AMP-sensitive component in the gating pipeline. See
[Section 9](#9-amp-safety-patterns) for the complete AMP safety specification.
The critical requirement is:

> **All reduction operations inside RMSNorm MUST execute in fp32.**
>
> Cast input to fp32 before computing `mean(x^2)` and `sqrt(...)`.
> Cast back to the original dtype after normalization and gamma scaling.

---

## 4. Gate Initialization

### 4.1 Conservative Bias Initialization

The gate computation includes an implicit or explicit bias term:

```python
# Option A: Explicit bias on the gate logit
gate_logits = torch.sum(q_norm * k_norm, dim=-1, keepdim=True) + gate_bias
alpha = torch.sigmoid(gate_logits)

# Option B: Bias absorbed into the Wq or Wk projection
# (less common, harder to control initial gate value)
```

The bias is initialized to a **negative value** to produce a conservative
starting gate:

```python
self.gate_bias = nn.Parameter(torch.tensor(-2.0))
```

### 4.2 Why -2.0?

```python
>>> import torch
>>> torch.sigmoid(torch.tensor(-2.0))
tensor(0.1192)
```

With `gate_bias = -2.0`, the initial gate value is approximately **0.12**,
regardless of the projection outputs (assuming the dot product of randomly
initialized normalized vectors centers around zero).

| Bias Value | sigmoid(bias) | Initial Gate Behavior |
|---|---|---|
| `-4.0` | `0.018` | Almost no retrieval passes (too conservative, slow learning) |
| `-3.0` | `0.047` | Very conservative |
| `-2.0` | `0.119` | **Recommended**: small but nonzero contribution |
| `-1.0` | `0.269` | Moderate initial contribution |
| `0.0` | `0.500` | Fifty-fifty: too much random noise early in training |
| `+1.0` | `0.731` | Mostly open: injects random embeddings |
| `+2.0` | `0.881` | Nearly always open: destabilizes training |

### 4.3 Training Dynamics with Conservative Init

The conservative initialization creates a learning curriculum:

```
Phase 1 (early training):
    gate_bias ~ -2.0
    alpha ~ 0.12 everywhere
    The backbone learns primarily from its own representations.
    Engram contributes a small delta that is mostly ignored.
    Gradients from the backbone are stable because Engram noise is suppressed.

Phase 2 (mid training):
    gate_bias has been updated by gradients
    Some positions have alpha > 0.5 (the model has learned when retrievals help)
    Other positions still have alpha < 0.2 (the model has learned when to ignore)

Phase 3 (converged):
    Gate distribution is bimodal (many positions near 0 or 1)
    The model has learned a clear decision boundary for trust/suppress
```

### 4.4 Gate Bias as a Learnable Parameter

The gate bias is a **single scalar parameter** that is updated by standard
backpropagation:

```python
# Forward pass
alpha = sigmoid(dot_product + self.gate_bias)

# Backward pass
d_loss/d_gate_bias = d_loss/d_alpha * sigmoid'(logit) * 1.0
```

Because it is a scalar, it acts as a **global prior** on gate openness. The
per-position modulation comes from the dot product between `q_norm` and `k_norm`.

The gate bias often benefits from a 2-5x learning rate multiplier via a
separate parameter group. See [Section 14.2](#142-initialization) for the
complete initialization checklist covering all projection matrices.

---

## 5. Gate Statistics for Telemetry

Monitoring gate behavior during training is critical for diagnosing issues and
confirming that the gating mechanism is working as intended. All statistics are
computed when `return_details=True` and returned in the telemetry dictionary.

### 5.1 Mean Gate Value

```python
gate_mean = alpha.mean()  # scalar
```

| `gate_mean` Value | Interpretation |
|---|---|
| `< 0.1` | Model suppresses almost all retrievals (gates still conservative or retrieval is unhelpful) |
| `0.1 - 0.3` | Light retrieval usage (typical early-to-mid training) |
| `0.3 - 0.6` | Moderate retrieval usage (healthy mid-to-late training) |
| `0.6 - 0.8` | Heavy retrieval usage (model finds retrievals very useful) |
| `> 0.8` | Nearly always open (suspicious: check if gate is saturated) |

**Tracking over time:** Plot `gate_mean` per training step. Expect a gradual
increase from ~0.12 (initial) to a stable value, typically 0.2-0.5 depending
on the task and data.

### 5.2 Gate Standard Deviation

```python
gate_std = alpha.squeeze(-1).std()  # scalar, std across (B, T)
```

| `gate_std` Value | Interpretation |
|---|---|
| `< 0.05` | Very uniform gating (all positions treated similarly -- possibly not learning) |
| `0.05 - 0.15` | Mild variation (some position-dependent gating) |
| `0.15 - 0.30` | **Healthy**: strong position-dependent gating |
| `> 0.30` | Very high variation (approaching bimodal distribution) |

High standard deviation is generally desirable because it indicates the model
has learned to discriminate between positions where retrieval is helpful versus
positions where it should be suppressed.

### 5.3 Gate Sparsity

```python
gate_sparsity = (alpha.squeeze(-1) < sparsity_eps).float().mean()  # scalar
```

where `sparsity_eps = 0.01` (configurable).

This measures the **fraction of positions where retrieval is effectively
suppressed** (gate value below epsilon).

| `gate_sparsity` Value | Interpretation |
|---|---|
| `> 0.8` | Most positions suppress retrieval (normal early training, or data doesn't benefit from Engram) |
| `0.3 - 0.8` | Selective suppression (healthy) |
| `< 0.1` | Almost no positions are suppressed (all retrievals pass, possibly saturated) |

### 5.4 Gate Entropy Proxy

The gate value alpha can be viewed as a Bernoulli probability. Its entropy
measures how "uncertain" the gate is:

```python
eps_entropy = 1e-7  # small constant to avoid log(0)
gate_entropy = -(
    alpha * torch.log(alpha + eps_entropy)
    + (1 - alpha) * torch.log(1 - alpha + eps_entropy)
).mean()  # scalar
```

| `gate_entropy` Value | Interpretation |
|---|---|
| `~ 0` | Gates are very confident (near 0 or 1) -- bimodal, well-learned |
| `~ 0.3` | Moderate uncertainty |
| `~ 0.693` (ln 2) | Maximum entropy: gates are at 0.5 (completely uncertain, not learned) |

**Expected trajectory:** Entropy starts moderate (gates at 0.12 have low entropy),
may increase during mid-training as the model explores, then decreases as
gates converge to confident values.

### 5.5 Complete Telemetry Dictionary

```python
def compute_gate_telemetry(
    alpha: torch.Tensor,
    sparsity_eps: float = 0.01,
    entropy_eps: float = 1e-7,
) -> Dict[str, torch.Tensor]:
    """
    Compute gate statistics for monitoring.

    Args:
        alpha: (B, T, 1) or (B, T) gate values in [0, 1]
        sparsity_eps: threshold for counting a gate as "suppressed"
        entropy_eps: numerical stability for log computation

    Returns:
        Dictionary of scalar tensors for logging
    """
    a = alpha.squeeze(-1)  # (B, T)

    return {
        "gate/mean": a.mean(),
        "gate/std": a.std(),
        "gate/min": a.min(),
        "gate/max": a.max(),
        "gate/sparsity": (a < sparsity_eps).float().mean(),
        "gate/entropy": -(
            a * torch.log(a + entropy_eps)
            + (1 - a) * torch.log(1 - a + entropy_eps)
        ).mean(),
        "gate/fraction_above_half": (a > 0.5).float().mean(),
        "gate/bias": self.gate_bias.detach(),  # current learned bias
    }
```

### 5.7 Logging Frequency

Log `gate/mean`, `gate/std`, `gate/sparsity`, `gate/entropy`, `gate/min`,
`gate/max` every step (cheap single reductions). Log histograms every 1000
steps. Per-position heatmaps on validation only.

---

## 6. Depthwise Causal Convolution

### 6.1 Purpose

After gating, each position `t` has an independent gated vector `gated[t]`.
There is no communication between adjacent positions. The depthwise causal
convolution serves two purposes:

1. **Receptive field expansion**: Each output position becomes a function of
   `K * dilation` prior positions, allowing local pattern mixing.
2. **Local nonlinearity**: Combined with SiLU activation, adds a nonlinear
   transformation that can sharpen or smooth the gated signal before it enters
   the residual stream.

### 6.2 Architecture

```python
self.causal_conv = nn.Conv1d(
    in_channels=D_out,
    out_channels=D_out,
    kernel_size=K,                    # default K=4
    groups=D_out,                     # depthwise: each channel independent
    padding=dilation * (K - 1),       # left-padding for causality
    dilation=dilation,                # default matches max_ngram_order
    bias=True,
)
```

**Key properties:**

| Property | Value | Notes |
|---|---|---|
| `in_channels` | `D_out` | Matches gated output dimension |
| `out_channels` | `D_out` | Same as input (depthwise) |
| `kernel_size` | `K = 4` | Paper default |
| `groups` | `D_out` | **Depthwise**: each channel has its own 1D filter |
| `padding` | `dilation * (K - 1)` | Left-only causal padding |
| `dilation` | `max_ngram_order` (default 4) | Matches N-gram temporal span |
| `bias` | `True` | Per-channel bias |

### 6.3 Depthwise Convolution Explained

In a depthwise convolution (`groups = channels`), each output channel depends
only on its corresponding input channel: `output[c] = filter[c] * input[c]`.
Parameters: `C * K` versus `C^2 * K` for standard convolution. For
`D_out = 4096, K = 4`: depthwise uses 16,384 params versus 67M for standard
(~4000x more efficient). Each channel learns its own temporal filter.

### 6.4 Causal Padding

The convolution must be **causal**: output at position `t` depends only on
positions `<= t`, never on future positions `> t`.

**Recommended implementation: explicit left-padding with `padding=0` in Conv1d.**

```python
# Constructor: padding=0, causal padding applied manually
self.causal_conv = nn.Conv1d(
    D_out, D_out, kernel_size=K, groups=D_out,
    padding=0, dilation=dilation, bias=True,
)
self.pad_amount = dilation * (K - 1)

# Forward: left-pad, convolve, output is exactly length T
def causal_conv_forward(self, x: torch.Tensor) -> torch.Tensor:
    x_padded = F.pad(x, (self.pad_amount, 0), value=0.0)  # (left, right)
    return self.causal_conv(x_padded)
```

With `pad_amount = dilation * (K - 1)` zeros on the left:
- Position `t=0` sees `K-1` padding zeros plus itself
- Position `t >= pad_amount` sees only real data from `t - pad_amount` to `t`

This explicit approach avoids ambiguity about which side PyTorch applies
built-in padding to and makes the causal property clear in code.

### 6.6 Activation Function

After convolution, a nonlinear activation is applied:

```python
conv_out = activation(causal_conv_forward(gated.transpose(1, 2))).transpose(1, 2)
```

| Activation | Formula | Properties |
|---|---|---|
| `SiLU` (default) | `x * sigmoid(x)` | Smooth, non-monotonic, good gradient flow, AMP-safe |
| `GELU` (alternative) | `x * Phi(x)` | Similar to SiLU, slightly different shape, standard in transformers |

**SiLU is the default** because it is simpler, slightly faster, and commonly
used in modern architectures (LLaMA, Mamba).

### 6.7 Convolution Weight Initialization

**Zero initialization is recommended** (`zero_init_conv=True`). Combined with
the conservative gate bias, this means initial delta is exactly zero. The model
starts as if Engram does not exist and gradually learns to use it. Alternative:
Kaiming uniform, relying on the conservative gate alone to keep contributions
small.

---

## 7. Dilation Configuration

### 7.1 Dilation Tied to N-gram Order

The convolution dilation is set to `max_ngram_order`:

```python
dilation = max_ngram_order  # e.g., 4 for N-gram orders {1, 2, 3, 4}
```

**Rationale:** The N-gram hash lookup considers token windows of up to
`max_ngram_order` positions. The convolution should have a receptive field
that spans at least the same temporal range, so it can mix signals across
the full N-gram context window.

### 7.2 Effective Receptive Field

For a dilated causal convolution:

```
Effective receptive field = dilation * (kernel_size - 1) + 1
                          = dilation * (K - 1) + 1
```

| K | Dilation | Effective RF | Positions Covered |
|---|---|---|---|
| 4 | 1 | 4 | `t, t-1, t-2, t-3` |
| 4 | 2 | 7 | `t, t-2, t-4, t-6` |
| 4 | 3 | 10 | `t, t-3, t-6, t-9` |
| 4 | 4 | 13 | `t, t-4, t-8, t-12` |
| 4 | 8 | 25 | `t, t-8, t-16, t-24` |

With the default `K=4, dilation=4`, the convolution at position `t` depends
on positions `{t, t-4, t-8, t-12}`, spanning 13 positions. This covers
the temporal range of 4-gram contexts with gaps that allow the filter to
capture longer-range patterns.

### 7.3 Dilated Convolution Mechanics

With dilation `d`, the kernel is "spread out" with `d-1` zeros between taps.
Standard `(d=1)`: taps at `[t, t-1, t-2, t-3]`. Dilated `(d=4)`: taps at
`[t, t-4, t-8, t-12]`. Causal padding: `pad_amount = dilation * (K - 1)`.

### 7.4 Dilation and Sequence Length

Dilation does not change output length (always `T` after causal trimming). For
short sequences (`T < dilation * (K - 1) + 1`), some filter taps land on
padding zeros, which is handled correctly by the left-padding scheme.

---

## 8. Residual Fusion

### 8.1 Residual Addition

The final output is a residual addition:

```python
output = hidden_states + residual_scale * delta
```

where:
- `hidden_states`: `(B, T, D)` from the backbone
- `delta`: `(B, T, D)` from the Engram pipeline (gating + conv + output projection)
- `residual_scale`: scalar, default `1.0`

### 8.2 Output Projection

If `D_out != D` (the gating/conv dimension differs from the backbone hidden
dimension), an output projection maps back:

```python
self.Wo = nn.Linear(D_out, D, bias=False)

delta = self.Wo(conv_out)  # (B, T, D_out) -> (B, T, D)
```

If `D_out == D`, the output projection can be omitted (identity):

```python
delta = conv_out  # (B, T, D) already matches backbone
```

**Typical configuration:** Use `D_out = D` and include `Wo` anyway for
architectural flexibility. Initialize `Wo` with small weights.

### 8.3 Residual Scale Factor

The `residual_scale` parameter provides an additional knob:

```python
output = hidden_states + residual_scale * delta
```

| Scale Value | Purpose |
|---|---|
| `1.0` (default) | Standard residual, no scaling |
| `< 1.0` (e.g., 0.5) | Further attenuate Engram contribution (useful when inserting at many layers) |
| `> 1.0` | Amplify Engram contribution (rarely needed) |
| Learnable | Can be made a learnable parameter initialized to small value |

When Engram is injected at multiple backbone layers (Phase 2 layer-augmentation
mode with multiple insertion points), using `residual_scale < 1.0` prevents
the accumulated Engram deltas from dominating the residual stream.

**Rule of thumb:** If inserting at `L` layers, use `residual_scale = 1.0 / sqrt(L)`
as a starting point.

### 8.5 Delta Magnitude Monitoring

The magnitude of `delta` relative to `hidden_states` is a critical health
metric:

```python
delta_norm = delta.norm(dim=-1).mean()
hidden_norm = hidden_states.norm(dim=-1).mean()
delta_ratio = delta_norm / (hidden_norm + 1e-8)
```

| `delta_ratio` | Interpretation |
|---|---|
| `< 0.01` | Engram contribution is negligible (early training or unhelpful retrieval) |
| `0.01 - 0.05` | Small but meaningful contribution (healthy) |
| `0.05 - 0.15` | Moderate contribution (healthy, typically mid-to-late training) |
| `0.15 - 0.30` | Large contribution (check if model is becoming over-reliant on Engram) |
| `> 0.30` | Very large contribution (may destabilize backbone, investigate) |

### 8.6 Pre-Norm vs. Post-Norm Placement

**Pre-norm is recommended:** Apply backbone normalization before Engram, then
add the delta to the un-normed hidden states. This matches the standard
pre-norm transformer convention (LLaMA, etc.), provides normalized inputs to
the gate for stability, and maintains clean gradient flow through the
un-normed residual path.

The complete fusion pipeline (conv + projection + residual add) is shown in the
integrated `EngramGatingAndFusion` class in [Section 10.2](#102-complete-pipeline-pytorch).

---

## 9. AMP Safety Patterns

Automatic Mixed Precision (AMP) is essential for training at scale. The gating
and fusion pipeline must be safe under both `torch.cuda.amp.autocast` (fp16)
and `torch.amp.autocast('cuda', dtype=torch.bfloat16)`.

### 9.1 AMP-Sensitive Operations in the Pipeline

| Operation | AMP Risk | Mitigation |
|---|---|---|
| RMSNorm: `mean(x^2)` | Overflow in fp16 if values > 256 | Force fp32 |
| RMSNorm: `sqrt(mean + eps)` | Underflow if mean is very small | Force fp32, use eps=1e-6 |
| RMSNorm: `x / rms` | Loss of precision | Compute in fp32, cast back |
| Gate dot product: `sum(q * k)` | Accumulation error in fp16 | Force fp32 accumulation |
| Sigmoid | Safe (input/output range bounded) | No mitigation needed |
| Element-wise multiply: `alpha * v` | Safe | No mitigation needed |
| Conv1d | Safe (PyTorch AMP handles this) | No mitigation needed |
| SiLU/GELU | Safe (bounded gradients) | No mitigation needed |
| Linear (Wq, Wk, Wv, Wo) | Safe (PyTorch AMP handles matmuls) | No mitigation needed |

### 9.2 RMSNorm AMP-Safe Implementation

```python
class RMSNorm(nn.Module):
    """AMP-safe Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype

        # CRITICAL: Force fp32 for all reductions
        x_fp32 = x.float()

        # Compute RMS in fp32
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(variance + self.eps)
        x_normed = x_fp32 / rms

        # Apply learnable scale and cast back
        # gamma is always fp32 (parameter), result cast to input dtype
        output = (x_normed * self.gamma).to(input_dtype)

        return output
```

### 9.3 Gate Computation AMP Pattern

```python
def compute_gate(
    self,
    q_norm: torch.Tensor,  # (B, T, D_gate), may be fp16
    k_norm: torch.Tensor,  # (B, T, D_gate), may be fp16
) -> torch.Tensor:
    """AMP-safe gate computation."""

    # Force fp32 for the dot product accumulation
    gate_logits = torch.sum(
        q_norm.float() * k_norm.float(),
        dim=-1,
        keepdim=True,
    )  # (B, T, 1), fp32

    # Add learnable bias (fp32 parameter)
    gate_logits = gate_logits + self.gate_bias

    # Sigmoid is safe in any precision, but keep fp32 for consistency
    alpha = torch.sigmoid(gate_logits)

    # Cast back to input dtype for downstream operations
    return alpha.to(q_norm.dtype)
```

The complete AMP-safe forward pass integrating all patterns is shown in
[Section 10.2](#102-complete-pipeline-pytorch). The key rule: RMSNorm and gate
dot product force fp32 internally; all other operations are AMP-safe natively.

### 9.4 fp16 vs. bf16 Considerations

| Property | fp16 | bf16 |
|---|---|---|
| Mantissa bits | 10 | 7 |
| Exponent bits | 5 | 8 |
| Max value | 65504 | ~3.4e38 |
| Min positive normal | 6.1e-5 | ~1.2e-38 |
| RMSNorm risk | Overflow in `x^2` if `|x| > 256` | Very low risk (same range as fp32) |
| Gate risk | Low (sigmoid bounded) | Very low |
| Recommendation | Force fp32 for norms | Force fp32 for norms (still recommended for consistency) |

**bf16 is generally safer** for the gating pipeline because its wider exponent
range prevents overflow in `x^2` computations. However, the fp32-forcing pattern
in RMSNorm should still be used for consistency and correctness.

### 9.6 Gradient Scaling and torch.compile

- **GradScaler:** The gate bias gradient is a scalar and can be very small under
  fp16 scaling. Mitigation: use bf16, or place gate bias in a separate parameter
  group with higher `loss_scale`.
- **torch.compile:** The `.float()` / `.to(input_dtype)` pattern is
  trace-friendly. Avoid `@torch.cuda.amp.custom_fwd` decorators with
  `torch.compile`. All config-based `if` branches are static.

---

## 10. Forward Pass Pipeline

### 10.1 Complete Pipeline (Pseudocode)

```
Input:
    hidden_states: (B, T, D)      -- from backbone layer
    token_ids:     (B, T)         -- original token IDs
    attention_mask: (B, T)        -- 1 for real tokens, 0 for padding

Step 1: Tokenizer Compression
    canonical_ids = compress(token_ids)                         # (B, T)

Step 2: N-gram Extraction
    For each order n in {1, 2, ..., max_ngram_order}:
        ngrams_n = extract_suffix_ngrams(canonical_ids, n)      # (B, T, n)
        Apply attention_mask: zero out ngrams at padded positions

Step 3: Multi-Head Hashing
    For each order n, for each head h in {1, ..., H_per_order}:
        hash_ids[n][h] = hash_fn(ngrams_n, seed=salt[n][h])    # (B, T)

Step 4: Embedding Retrieval
    For each (n, h):
        emb[n][h] = embedding_table[n][h][hash_ids[n][h]]      # (B, T, D_per_head)
    retrieved = concat_and_aggregate(all emb)                   # (B, T, D_emb)

Step 5: Context-Aware Gating
    q = RMSNorm_q(Wq(hidden_states))                           # (B, T, D_gate)
    k = RMSNorm_k(Wk(retrieved))                               # (B, T, D_gate)
    v = Wv(retrieved)                                           # (B, T, D_out)
    alpha = sigmoid(sum(q * k, dim=-1, keepdim=True) + bias)   # (B, T, 1)
    gated = alpha * v                                           # (B, T, D_out)

Step 6: Depthwise Causal Convolution
    x = gated.transpose(1, 2)                                  # (B, D_out, T)
    x = left_pad(x, pad=dilation * (K - 1))                    # (B, D_out, T + pad)
    x = conv1d_depthwise(x)                                    # (B, D_out, T)
    x = x.transpose(1, 2)                                      # (B, T, D_out)
    conv_out = silu(x)                                          # (B, T, D_out)

Step 7: Output Projection
    delta = Wo(conv_out)                                        # (B, T, D)

Step 8: Residual Fusion
    output = hidden_states + residual_scale * delta             # (B, T, D)

Step 9: Mask Application
    output = output * attention_mask.unsqueeze(-1)              # zero padded positions

Step 10: Telemetry (if return_details=True)
    telemetry = compute_gate_telemetry(alpha)
    telemetry["delta_norm"] = delta.norm(dim=-1).mean()
    telemetry["delta_ratio"] = delta_norm / hidden_norm

Output:
    output:    (B, T, D)
    telemetry: Dict[str, Tensor] or None
```

### 10.2 Complete Pipeline (PyTorch)

```python
class EngramGatingAndFusion(nn.Module):
    """
    Complete gating and fusion pipeline for Engram conditional memory.

    Takes retrieved N-gram embeddings and backbone hidden states,
    produces a residual delta for the backbone.
    """

    def __init__(
        self,
        d_model: int,
        d_emb: int,
        d_gate: int = 128,
        d_out: Optional[int] = None,
        gate_type: str = "scalar",
        use_rmsnorm: bool = True,
        gate_init_bias: float = -2.0,
        conv_kernel_size: int = 4,
        conv_dilation: int = 4,
        activation: str = "silu",
        residual_scale: float = 1.0,
        num_heads: int = 1,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_emb = d_emb
        self.d_gate = d_gate
        self.d_out = d_out or d_model
        self.gate_type = gate_type
        self.use_rmsnorm = use_rmsnorm
        self.residual_scale = residual_scale
        self.num_heads = num_heads

        # ---------- Projections ----------
        self.Wq = nn.Linear(d_model, d_gate, bias=False)
        self.Wk = nn.Linear(d_emb, d_gate, bias=False)
        self.Wv = nn.Linear(d_emb, self.d_out, bias=False)

        # ---------- RMSNorm ----------
        if use_rmsnorm:
            self.rmsnorm_q = RMSNorm(d_gate, eps=eps)
            self.rmsnorm_k = RMSNorm(d_gate, eps=eps)

        # ---------- Gate bias ----------
        self.gate_bias = nn.Parameter(torch.tensor(gate_init_bias))

        # ---------- Depthwise causal convolution ----------
        self.conv_kernel_size = conv_kernel_size
        self.conv_dilation = conv_dilation
        self.pad_amount = conv_dilation * (conv_kernel_size - 1)

        self.causal_conv = nn.Conv1d(
            in_channels=self.d_out,
            out_channels=self.d_out,
            kernel_size=conv_kernel_size,
            groups=self.d_out,
            padding=0,
            dilation=conv_dilation,
            bias=True,
        )

        # ---------- Activation ----------
        if activation == "silu":
            self.act = nn.SiLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # ---------- Output projection ----------
        self.Wo = nn.Linear(self.d_out, d_model, bias=False)

        # ---------- Initialization ----------
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.Wq.weight)
        nn.init.xavier_uniform_(self.Wk.weight)
        nn.init.xavier_uniform_(self.Wv.weight)
        nn.init.xavier_uniform_(self.Wo.weight, gain=0.1)
        nn.init.zeros_(self.causal_conv.weight)
        nn.init.zeros_(self.causal_conv.bias)
        # gate_bias already initialized in Parameter constructor

    def forward(
        self,
        hidden_states: torch.Tensor,
        retrieved: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Args:
            hidden_states: (B, T, D) backbone hidden states
            retrieved: (B, T, D_emb) aggregated N-gram embeddings
            attention_mask: (B, T) optional, 1=real, 0=pad
            return_details: whether to compute and return telemetry

        Returns:
            output: (B, T, D) hidden_states + residual delta
            telemetry: dict of scalar tensors, or None
        """
        B, T, D = hidden_states.shape

        # --- Step 1: Projections ---
        q = self.Wq(hidden_states)      # (B, T, D_gate)
        k = self.Wk(retrieved)          # (B, T, D_gate)
        v = self.Wv(retrieved)          # (B, T, D_out)

        # --- Step 2: RMSNorm ---
        if self.use_rmsnorm:
            q = self.rmsnorm_q(q)
            k = self.rmsnorm_k(k)

        # --- Step 3: Gate computation (fp32 accumulation) ---
        gate_logits = torch.sum(
            q.float() * k.float(), dim=-1, keepdim=True
        )  # (B, T, 1), fp32

        gate_logits = gate_logits + self.gate_bias
        alpha = torch.sigmoid(gate_logits).to(v.dtype)  # (B, T, 1)

        # --- Step 4: Gated output ---
        gated = alpha * v               # (B, T, D_out)

        # --- Step 5: Depthwise causal convolution ---
        x = gated.transpose(1, 2)                          # (B, D_out, T)
        x = F.pad(x, (self.pad_amount, 0), value=0.0)      # left-pad
        x = self.causal_conv(x)                             # (B, D_out, T)
        x = x.transpose(1, 2)                              # (B, T, D_out)
        conv_out = self.act(x)                              # (B, T, D_out)

        # --- Step 6: Output projection ---
        delta = self.Wo(conv_out)                           # (B, T, D)

        # --- Step 7: Residual fusion ---
        output = hidden_states + self.residual_scale * delta  # (B, T, D)

        # --- Step 8: Mask application ---
        if attention_mask is not None:
            output = output * attention_mask.unsqueeze(-1)

        # --- Step 9: Telemetry ---
        telemetry = None
        if return_details:
            with torch.no_grad():
                a = alpha.squeeze(-1)  # (B, T)
                eps_ent = 1e-7
                telemetry = {
                    "gate/mean": a.mean(),
                    "gate/std": a.std(),
                    "gate/min": a.min(),
                    "gate/max": a.max(),
                    "gate/sparsity": (a < 0.01).float().mean(),
                    "gate/entropy": -(
                        a * torch.log(a + eps_ent)
                        + (1 - a) * torch.log(1 - a + eps_ent)
                    ).mean(),
                    "gate/fraction_above_half": (a > 0.5).float().mean(),
                    "gate/bias": self.gate_bias.detach().clone(),
                    "delta/norm": delta.norm(dim=-1).mean(),
                    "delta/ratio": (
                        delta.norm(dim=-1).mean()
                        / (hidden_states.norm(dim=-1).mean() + 1e-8)
                    ),
                }

        return output, telemetry
```

### 10.3 Tensor Shape Summary

| Variable | Shape | Dtype | Notes |
|---|---|---|---|
| `hidden_states` | `(B, T, D)` | fp16/bf16/fp32 | Backbone input |
| `retrieved` | `(B, T, D_emb)` | fp16/bf16/fp32 | From hash tables |
| `q` (after Wq) | `(B, T, D_gate)` | inherits from input | Query projection |
| `k` (after Wk) | `(B, T, D_gate)` | inherits from input | Key projection |
| `v` (after Wv) | `(B, T, D_out)` | inherits from input | Value projection |
| `q_norm` | `(B, T, D_gate)` | same as input (fp32 internal) | After RMSNorm |
| `k_norm` | `(B, T, D_gate)` | same as input (fp32 internal) | After RMSNorm |
| `gate_logits` | `(B, T, 1)` | fp32 | Dot product accumulation |
| `alpha` | `(B, T, 1)` | same as `v` | Sigmoid gate |
| `gated` | `(B, T, D_out)` | same as `v` | alpha * v |
| `conv_input` | `(B, D_out, T+pad)` | same as `v` | After transpose + left-pad |
| `conv_out` | `(B, T, D_out)` | same as `v` | After conv + activation |
| `delta` | `(B, T, D)` | same as `v` | After Wo |
| `output` | `(B, T, D)` | same as input | hidden + delta |

### 10.4 Parameter Count

For a configuration with `D=4096, D_emb=256, D_gate=128, D_out=4096, K=4`:

| Component | Parameters | Formula |
|---|---|---|
| `Wq` | 524,288 | `D * D_gate = 4096 * 128` |
| `Wk` | 32,768 | `D_emb * D_gate = 256 * 128` |
| `Wv` | 1,048,576 | `D_emb * D_out = 256 * 4096` |
| `RMSNorm_q` (gamma) | 128 | `D_gate` |
| `RMSNorm_k` (gamma) | 128 | `D_gate` |
| `gate_bias` | 1 | scalar |
| `causal_conv` (weight) | 16,384 | `D_out * K = 4096 * 4` |
| `causal_conv` (bias) | 4,096 | `D_out` |
| `Wo` | 16,777,216 | `D_out * D = 4096 * 4096` |
| **Total** | **18,403,585** | ~18.4M |

The output projection `Wo` dominates when `D_out = D`. If `D_out < D`
(e.g., `D_out = 512`), the parameter count drops to approximately 2.7M.

---

## 11. Configuration Parameters

### 11.1 GatingConfig Dataclass

```python
@dataclass
class GatingConfig:
    """Configuration for context-aware gating and residual fusion."""

    # Gate type
    gate_type: str = "scalar"
    """Gate granularity: "scalar" (one gate per position) or
    "per_head" (one gate per hash head per position)."""

    # RMSNorm
    use_rmsnorm: bool = True
    """Apply RMSNorm to query and key before gate computation.
    Disable only for ablation experiments."""

    rmsnorm_eps: float = 1e-6
    """Epsilon for RMSNorm stability. Increase to 1e-5 for pure fp16."""

    # Gate initialization
    gate_init_bias: float = -2.0
    """Initial gate bias. sigmoid(-2.0) ~ 0.12.
    Negative values are conservative (suppress retrieval early in training)."""

    # Gate dimensions
    gate_dim: int = 128
    """Dimension of the gating projection space (D_gate).
    Smaller than D for efficiency since gate only produces a scalar."""

    # Temperature
    temperature: float = 1.0
    """Gate temperature. < 1.0 sharpens, > 1.0 softens."""

    # Convolution
    conv_kernel_size: int = 4
    """Kernel size for depthwise causal convolution."""

    conv_dilation: int = 4
    """Dilation for causal convolution. Default matches max_ngram_order."""

    # Activation
    activation: str = "silu"
    """Post-convolution activation: "silu" or "gelu"."""

    # Residual
    residual_scale: float = 1.0
    """Scaling factor for the delta before residual addition.
    Use 1/sqrt(L) when inserting at L backbone layers."""

    # Output
    output_dim: Optional[int] = None
    """D_out for value projection. None defaults to d_model."""

    # Initialization
    output_proj_gain: float = 0.1
    """Xavier gain for output projection initialization."""

    zero_init_conv: bool = True
    """Zero-initialize convolution weights for conservative start."""
```

### 11.2 Parameter Interaction Matrix

Some parameters interact with each other. This table documents critical
interactions:

| Parameter A | Parameter B | Interaction |
|---|---|---|
| `gate_init_bias` | `zero_init_conv` | Both conservative: if both are used, initial delta is exactly zero. Relaxing one may require tightening the other. |
| `gate_init_bias` | `output_proj_gain` | Both control initial delta magnitude. With bias=-2.0 and gain=0.1, initial delta is very small (~0.012x of what it would be with bias=0 and gain=1.0). |
| `conv_dilation` | `conv_kernel_size` | Effective receptive field = dilation * (K-1) + 1. Increasing either expands the field. |
| `conv_dilation` | `max_ngram_order` | Default: dilation = max_ngram_order. Change dilation if changing N-gram orders. |
| `gate_dim` | `use_rmsnorm` | Smaller gate_dim means fewer RMSNorm parameters. With very small gate_dim (e.g., 16), RMSNorm may have limited effect. |
| `residual_scale` | `gate_init_bias` | Both attenuate the Engram contribution. Usually only one should be non-default. |
| `gate_type` | `num_heads_per_order` | "per_head" requires knowledge of total head count H = num_orders * num_heads_per_order. |
| `temperature` | `gate_init_bias` | Temperature affects the sigmoid slope. Higher temperature with negative bias means the gate opens more gradually. |

### 11.3 Preset Configurations

| Preset | `gate_type` | `gate_dim` | `conv_kernel_size` | `conv_dilation` | Notes |
|---|---|---|---|---|---|
| `minimal()` | `"scalar"` | 32 | 2 | 2 | Unit tests, ~1M param models |
| `dev()` | `"scalar"` | 64 | 4 | 4 | Fast iteration on MNIST-scale |
| `production()` | `"scalar"` | 128 | 4 | 4 | Full-scale training |
| `per_head_production()` | `"per_head"` | 128 | 4 | 4 | Per-head variant for ablation |

All presets share: `use_rmsnorm=True`, `gate_init_bias=-2.0`, `activation="silu"`,
`residual_scale=1.0`, `zero_init_conv=True`. Production presets additionally set
`output_proj_gain=0.1`.

### 11.4 Configuration Validation

Validate at construction time: `gate_type` in `{"scalar", "per_head"}`,
`activation` in `{"silu", "gelu"}`, positive values for `conv_kernel_size`,
`conv_dilation`, `gate_dim`, `residual_scale`, and `rmsnorm_eps`. Warn if
`gate_init_bias > 0` (initial gates > 0.5 may inject noise).

---

## 12. Ablation Guidance

| Ablation | Config Changes | Expected Effect |
|---|---|---|
| No RMSNorm | `use_rmsnorm=False` | Gate saturation/collapse if scale mismatch is large |
| No gating | Force `alpha=1.0` | Noise injection from unconditional retrieval |
| No conv | Skip causal conv | Lose local mixing, ~10-30% less perplexity improvement |
| No dilation | `conv_dilation=1` | Receptive field shrinks from 13 to 4 |
| No conservative init | `gate_init_bias=0.0` | Training instability in first ~1000 steps |
| Per-head vs. scalar | `gate_type="per_head"` | More parameters, often marginal gain |

**Recommended order:** Full model, then no-Engram baseline, then no-gating,
no-RMSNorm, no-conv, no-conservative-init, per-head. Measure training loss
curve, validation perplexity, gate statistics, throughput, and memory for each.

---

## 13. Common Failure Modes

### 13.1 NaN in Gate Under AMP

**Symptom:** `alpha` contains NaN values during mixed-precision training.

**Cause:** RMSNorm computed in fp16, where `x^2` overflows for values > 256.

**Fix:** Force fp32 for all reduction operations inside RMSNorm:

```python
# BAD: RMSNorm in fp16
rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

# GOOD: RMSNorm in fp32
x_fp32 = x.float()
rms = torch.sqrt(torch.mean(x_fp32 ** 2, dim=-1, keepdim=True) + self.eps)
x_normed = (x_fp32 / rms).to(x.dtype)
```

### 13.2 Gate Always Saturated at 1.0

**Symptom:** `gate/mean > 0.95` and `gate/std < 0.02` after initial training.

**Cause:** Gate bias initialized too high (e.g., 0.0 or positive), or gradient
signal pushing bias up too fast.

**Fix:**
1. Initialize `gate_init_bias = -2.0`
2. If already at -2.0, check if the backbone learning rate is appropriate
3. Inspect `gate/bias` telemetry: if bias is rapidly increasing, reduce LR
   for the gate bias parameter group

### 13.3 Gate Always Collapsed at 0.0

**Symptom:** `gate/mean < 0.05` and `gate/sparsity > 0.95` after many training
steps.

**Cause:** Retrieved embeddings are not useful for the task, or the key
projection is not learning.

**Fix:**
1. Check that the embedding tables are being trained (gradients flow through
   `Wv` and the embedding lookup)
2. Verify that tokenizer compression is not collapsing important distinctions
3. Check that N-gram orders cover the relevant pattern lengths for the task
4. Increase `gate_init_bias` slightly (e.g., from -2.0 to -1.5) to give
   the model more initial signal

### 13.4 Non-Causal Convolution Leakage

**Symptom:** Model achieves suspiciously good perplexity, especially on
autoregressive tasks.

**Cause:** Convolution padding is symmetric (default PyTorch behavior) instead
of left-only causal padding.

**Fix:** Use manual left-padding with `padding=0` in the Conv1d constructor:

```python
# BAD: symmetric padding allows future leakage
self.conv = nn.Conv1d(..., padding=dilation*(K-1)//2)

# GOOD: explicit causal padding
self.conv = nn.Conv1d(..., padding=0)
# In forward:
x = F.pad(x, (dilation * (K - 1), 0))
x = self.conv(x)
```

**Verification test:**

```python
def test_causality(model, B=2, T=16, D=64):
    """Verify output at position t is independent of inputs at positions > t."""
    x1 = torch.randn(B, T, D)
    x2 = x1.clone()
    x2[:, 8:, :] = torch.randn(B, T - 8, D)  # Change future positions

    # Outputs at position 7 should be identical
    y1 = model(x1)
    y2 = model(x2)
    assert torch.allclose(y1[:, :8, :], y2[:, :8, :], atol=1e-6), (
        "Causality violation: output at t < 8 changed when future was modified"
    )
```

### 13.5 Delta Too Large

**Symptom:** `delta/ratio > 0.3`, training loss oscillates.

**Cause:** Output projection gain too large, or residual scale too high.

**Fix:**
1. Reduce `output_proj_gain` (e.g., from 1.0 to 0.1)
2. Reduce `residual_scale` (e.g., from 1.0 to 0.5)
3. Enable `zero_init_conv` if not already enabled
4. Check that `gate_init_bias` is negative

### 13.6 Gradient Vanishing Through Gate

**Symptom:** Engram embedding tables and projection matrices have near-zero
gradients.

**Cause:** Gate is stuck near 0 (gradients through `alpha * v` vanish when
`alpha ~ 0`).

**Fix:**
1. Check `gate/mean` telemetry. If consistently < 0.05, the gate is too
   conservative.
2. Increase `gate_init_bias` slightly (e.g., -2.0 to -1.0)
3. Ensure gradients can flow through the gate: `sigmoid'(-2) = 0.105`,
   which is small but nonzero. If combined with very small `v` values,
   the product of gradients vanishes.
4. Consider a "straight-through" gradient estimator for the gate as a
   last resort (not recommended for standard use).

---

## 14. Implementation Checklist

### 14.1 Module Implementation

- [ ] `Wq`, `Wk`, `Wv` projections with correct input/output dimensions
- [ ] Separate `RMSNorm` instances for query and key with independent gamma
- [ ] Gate bias as `nn.Parameter` initialized to `gate_init_bias` (default -2.0)
- [ ] Gate computation with fp32 accumulation for dot product
- [ ] Sigmoid applied to gate logits (+ bias)
- [ ] Depthwise causal Conv1d with `groups=D_out` and `padding=0`
- [ ] Manual left-padding: `F.pad(x, (dilation * (K-1), 0))`
- [ ] Causal trim (if using built-in padding): `out[:, :, :T]`
- [ ] SiLU (or GELU) activation after convolution
- [ ] Output projection `Wo` with small initialization gain
- [ ] Residual addition with configurable scale
- [ ] Attention mask application on output

### 14.2 Initialization

- [ ] `Wq`: Xavier uniform
- [ ] `Wk`: Xavier uniform
- [ ] `Wv`: Xavier uniform
- [ ] `Wo`: Xavier uniform with `gain=0.1`
- [ ] `gate_bias`: constant at `-2.0`
- [ ] `causal_conv.weight`: zeros (if `zero_init_conv=True`)
- [ ] `causal_conv.bias`: zeros (if `zero_init_conv=True`)
- [ ] `RMSNorm.gamma`: ones (both instances)

### 14.3 AMP Safety

- [ ] RMSNorm computes `mean(x^2)` and `sqrt` in fp32
- [ ] RMSNorm casts output back to input dtype
- [ ] Gate dot product accumulates in fp32
- [ ] Gate sigmoid result cast back to value dtype
- [ ] No explicit `@autocast` decorators needed (handled by module-level pattern)
- [ ] Compatible with `torch.compile`

### 14.4 Telemetry

- [ ] `gate/mean`: mean of alpha across (B, T)
- [ ] `gate/std`: std of alpha across (B, T)
- [ ] `gate/min`, `gate/max`: range check
- [ ] `gate/sparsity`: fraction below epsilon
- [ ] `gate/entropy`: Bernoulli entropy proxy
- [ ] `gate/fraction_above_half`: fraction of open gates
- [ ] `gate/bias`: current learned bias value
- [ ] `delta/norm`: mean L2 norm of delta
- [ ] `delta/ratio`: delta norm relative to hidden state norm
- [ ] All computed under `torch.no_grad()` for efficiency
- [ ] All returned only when `return_details=True`

### 14.5 Testing

- [ ] Causality test: modifying future inputs does not change past outputs
- [ ] AMP test: forward pass under autocast produces no NaN/Inf
- [ ] Shape test: all intermediate and output shapes match specification
- [ ] Conservative init test: initial `alpha ~ 0.12` with default bias
- [ ] Determinism test: same inputs produce same outputs across runs
- [ ] Gate range test: `alpha` values always in `[0, 1]`
- [ ] Residual identity test: when gate is zero, output equals hidden_states
- [ ] Gradient flow test: all parameters receive non-zero gradients
- [ ] Mask test: padded positions produce zero output
- [ ] Telemetry test: all expected keys present in returned dict

---

## Appendix A: Notation Reference

| Symbol | Meaning | Typical Value |
|---|---|---|
| `B` | Batch size | 16-128 |
| `T` | Sequence length | 128-8192 |
| `D` | Backbone hidden dimension | 512-4096 |
| `D_emb` | Total retrieved embedding dimension | 256-1024 |
| `D_gate` | Gating projection dimension | 64-256 |
| `D_out` | Value projection / conv dimension | D or < D |
| `H` | Number of hash heads (total across all orders) | 4-16 |
| `K` | Convolution kernel size | 4 |
| `d` | Convolution dilation | 1-8 (default = max_ngram_order) |
| `alpha` | Gate value | [0, 1] |
| `gamma` | RMSNorm learnable scale | Per-dimension, init 1.0 |
| `eps` | RMSNorm epsilon | 1e-6 |

## Appendix B: Related Modules

| Module | Path | Relationship |
|---|---|---|
| `EngramModule` | `brain_ai/memory/engram.py` | Parent module that owns gating and fusion |
| `EngramEmbedding` | `brain_ai/memory/engram.py` | Provides `retrieved` tensor input to gating |
| `EngramAugmentedLayer` | `brain_ai/layers/engram_layer.py` | Phase 2 integration that calls EngramModule |
| `TokenizerCompression` | `brain_ai/memory/tokenizer_compression.py` | Upstream: compresses token IDs before hashing |
| `MultiHeadHash` | `brain_ai/memory/hash_embedding.py` | Upstream: hashes N-grams to embedding indices |
| `OffloadableEmbedding` | `brain_ai/memory/hash_embedding.py` | Upstream: retrieves embeddings from tables |
| `BrainAIConfig` | `brain_ai/config.py` | Root config that aggregates EngramConfig |

## Appendix C: References

- DeepSeek Engram paper -- Conditional memory via scalable N-gram hash lookup
- Zhang and Sennrich, 2019 -- RMSNorm
- Howard et al., 2017 -- Depthwise separable convolutions (MobileNets)
- ReZero / FixUp -- Conservative gate initialization patterns
