# Symlog Twohot Reference

## Overview

DreamerV3 introduces two numerical stability techniques for reward and value prediction: the
symlog transformation and twohot encoding. Together, they form a scale-invariant distribution
for regression targets that works without any environment-specific normalization.

The problem they solve: reward signals span wildly different scales across environments. Atari
rewards are integers in the thousands, Minecraft rewards are near zero, and continuous control
tasks produce real-valued rewards. A fixed-scale loss function (like MSE) would require separate
tuning for each environment. Symlog + twohot eliminates this requirement.

---

## Symlog Transform

### Formula

```
symlog(x) = sign(x) * ln(|x| + 1)
```

### Properties

- **Identity near origin**: For small `|x|`, `ln(|x| + 1) ≈ x`, so symlog ≈ identity near 0.
- **Logarithmic compression**: For large `|x|`, `ln(|x| + 1) ≈ ln(|x|)`, compressing large values.
- **Antisymmetry**: `symlog(-x) = -symlog(x)`, preserving the sign of the input.
- **Monotone**: Strictly increasing, so ordering is preserved.
- **Range**: Maps `(-inf, inf)` to `(-inf, inf)`, but with compressed scale.

### Example Values

| x | symlog(x) | ratio x/symlog(x) |
|---|-----------|-------------------|
| 0 | 0 | — |
| 1 | 0.693 | 1.44x |
| 10 | 2.398 | 4.17x |
| 100 | 4.615 | 21.7x |
| 1000 | 6.909 | 144.8x |
| 1,000,000 | 13.816 | 72,380x |
| -100 | -4.615 | same compression |

A reward of 1,000,000 becomes ~13.8 in symlog space — a factor of ~72,000x compression.
This is why a fixed bin range of [-20, 20] in symlog space covers rewards from roughly
-500 million to +500 million.

### PyTorch Implementation

```python
def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)
```

Note: Do NOT use `torch.log1p(torch.abs(x))` without the sign. That would always return
non-negative values.

---

## Symexp Transform (Inverse)

### Formula

```
symexp(x) = sign(x) * (exp(|x|) - 1)
```

### Properties

- **Inverse of symlog**: `symexp(symlog(x)) = x` for all `x`.
- **Identity near origin**: For small `|x|`, `exp(|x|) - 1 ≈ |x|`, so symexp ≈ identity.
- **Exponential expansion**: Inverts the logarithmic compression of symlog.
- **Antisymmetry**: `symexp(-x) = -symexp(x)`.

### Derivation

Starting from `y = symlog(x) = sign(x) * ln(|x| + 1)`:

```
|y| = ln(|x| + 1)
exp(|y|) = |x| + 1
|x| = exp(|y|) - 1
x = sign(y) * (exp(|y|) - 1)
```

Therefore `symexp(y) = sign(y) * (exp(|y|) - 1)`.

### PyTorch Implementation

```python
def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)
```

Warning: `torch.exp` can overflow for large inputs. The symlog output is always in a
reasonable range ([-20, 20] for rewards up to ~500M), so overflow is not a concern for
the decode path when applied to symlog outputs. However, apply symexp only to values
that have been through the bin range; never apply it to raw neural network outputs
without clamping.

---

## Twohot Encoding

### Motivation

Direct regression with MSE:
```
L = (output - target)^2
```

Problems:
1. Large target values produce enormous gradients, destabilizing training.
2. The loss is not scale-invariant: targets in [-1, 1] and targets in [-1e6, 1e6] require
   different learning rates.
3. Single-output regression has no notion of prediction uncertainty.

The twohot approach converts a scalar regression target into a probability distribution over
a discrete set of bins, then uses cross-entropy as the loss. Cross-entropy is naturally
scale-invariant because it operates on log-probabilities.

### Bin Layout

```
num_bins = 255
low = -20.0
high = 20.0
bin_centers = linspace(-20.0, 20.0, 255)
```

The 255 bin centers are:
```
b_0 = -20.0
b_1 = -19.843
b_2 = -19.686
...
b_127 = 0.0  (center bin at origin)
...
b_253 = 19.843
b_254 = 20.0
```

Bin spacing: `(20 - (-20)) / (255 - 1) = 40 / 254 ≈ 0.157` in symlog space.

The choice of 255 bins balances:
- **Resolution**: finer bins → better approximation of continuous values
- **Memory/compute**: 255-class categorical fits in standard buffers
- **Coverage**: [-20, 20] in symlog space covers |x| up to symexp(20) ≈ 485,165,195

### Algorithm: Encode Scalar to Twohot

Given a scalar `v`:

**Step 1**: Apply symlog compression.
```
v_log = symlog(v)
```

**Step 2**: Clamp to bin range (handles values outside the representable range).
```
v_clamped = clamp(v_log, low, high)
```

**Step 3**: Find the lower adjacent bin index.
```
# Bin spacing
delta = (high - low) / (num_bins - 1)  # ≈ 0.1575

# Continuous position in bin space
pos = (v_clamped - low) / delta  # in [0, num_bins - 1]

# Integer bin index (lower bound)
k = floor(pos)
k = clamp(k, 0, num_bins - 2)  # ensure upper bin k+1 exists
```

**Step 4**: Compute interpolation weights.
```
# Upper bin center and lower bin center
b_k   = bin_centers[k]
b_k1  = bin_centers[k + 1]

# Weight for lower bin: proportional to distance to upper bin
w_k  = (b_k1 - v_clamped) / (b_k1 - b_k)

# Weight for upper bin: proportional to distance to lower bin
w_k1 = 1.0 - w_k = (v_clamped - b_k) / (b_k1 - b_k)
```

**Step 5**: Build the twohot vector.
```
target = zeros(num_bins)
target[k]   = w_k
target[k+1] = w_k1
```

Result: a sparse vector with exactly two non-zero entries summing to 1.

### Batch Implementation

```python
def encode(self, x: Tensor) -> Tensor:
    # x: arbitrary shape (*, )
    x_log = symlog(x).clamp(self.low, self.high)  # (*)

    # Bin positions
    delta = (self.high - self.low) / (self.num_bins - 1)
    pos = (x_log - self.low) / delta           # (*)
    k = pos.long().clamp(0, self.num_bins - 2) # (*), integer lower bin

    # Fractional weights
    b_k  = self.bin_centers[k]       # (*)
    b_k1 = self.bin_centers[k + 1]   # (*)
    w_upper = (x_log - b_k) / (b_k1 - b_k)   # weight for upper bin
    w_lower = 1.0 - w_upper                   # weight for lower bin

    # Scatter into one-hot-like vector
    target = torch.zeros(*x.shape, self.num_bins, device=x.device)
    target.scatter_(-1, k.unsqueeze(-1), w_lower.unsqueeze(-1))
    target.scatter_(-1, (k + 1).unsqueeze(-1), w_upper.unsqueeze(-1))
    return target  # (*, num_bins), sums to 1
```

---

## Decode: Logits to Scalar

Given predicted logits (the output of a linear head), recover a scalar estimate:

**Step 1**: Convert logits to probabilities.
```
p = softmax(logits, dim=-1)   # (*, num_bins)
```

**Step 2**: Compute expected bin center in symlog space.
```
v_log = sum(p * bin_centers, dim=-1)   # (*)
      = (p @ bin_centers)
```

**Step 3**: Apply symexp to return to original scale.
```
v = symexp(v_log)   # (*)
```

### PyTorch Implementation

```python
def decode(self, logits: Tensor) -> Tensor:
    # logits: (*, num_bins)
    probs = torch.softmax(logits, dim=-1)
    v_log = (probs * self.bin_centers).sum(dim=-1)
    return symexp(v_log)
```

Note: `self.bin_centers` should be registered as a buffer (not a parameter) so it moves
with the module to the correct device.

---

## Cross-Entropy Loss

### Formula

```
L = -sum_k(target_k * log(softmax(logits)_k))
  = -sum_k(target_k * log_softmax(logits)_k)
```

For twohot targets (only two non-zero entries), this simplifies to:

```
L = -w_k * log_softmax(logits)_k - w_{k+1} * log_softmax(logits)_{k+1}
```

But in practice, use the full formula for numerical stability.

### PyTorch Implementation

```python
def loss(self, logits: Tensor, target: Tensor) -> Tensor:
    # logits: (*, num_bins) — raw network output
    # target: (*) — scalar regression target values
    twohot = self.encode(target)  # (*, num_bins)
    log_probs = torch.log_softmax(logits, dim=-1)  # (*, num_bins)
    return -(twohot * log_probs).sum(dim=-1)  # (*), one scalar loss per element
```

Return the per-element loss and aggregate (mean) in the caller. This preserves gradient
information for any sequence masking (e.g., ignoring padded timesteps).

### Numerical Stability

`torch.log_softmax` is more numerically stable than `torch.log(torch.softmax(...))` because
it avoids computing softmax first (which can overflow) and instead computes the log-sum-exp
directly.

---

## Why Twohot Works

### Cross-Entropy is Scale-Invariant

MSE loss: `(y_pred - y_target)^2`
- Gradient magnitude scales as `|y_pred - y_target|`
- Large targets → large gradients → instability
- Requires target normalization

Twohot cross-entropy: `-sum(target_k * log_softmax(logits)_k)`
- Gradient magnitude is bounded by the categorical probabilities
- Same loss function works for rewards ranging from -1 to 1e6
- No target normalization needed

### Gradient Flow

Twohot's two-bin interpolation means the loss is differentiable w.r.t. both adjacent bins'
logits, even for exact-integer values. Pure one-hot encoding would only backpropagate through
one bin's logit, wasting capacity.

The gradient of the cross-entropy loss w.r.t. logits is:

```
dL/d(logits_j) = softmax(logits)_j - target_j
```

This is the difference between predicted probability and target weight. The gradient is bounded
in [-1, 1] for all logit values, making training naturally stable.

---

## Comparison to Alternative Approaches

| Method | Scale-Invariant | Uncertainty | Gradient Stability | Notes |
|--------|----------------|-------------|-------------------|-------|
| MSE regression | No | No | Poor for large rewards | Requires target normalization |
| Normalized MSE | Partially | No | Better | Requires running statistics |
| Quantile regression | Partially | Yes (quantiles) | Good | More complex |
| Symlog + MSE | Yes | No | Good | Simpler than twohot |
| Symlog + Twohot | Yes | Yes (distribution) | Excellent | DreamerV3 approach |

The combination of symlog and twohot is uniquely suited to world models because:
1. World models must work across thousands of environments without tuning
2. Scale-invariance eliminates the most common hyperparameter (reward normalization)
3. The distributional nature provides a principled uncertainty estimate

---

## Numerical Edge Cases

### Very Large Positive Values

```
x = 1e9
symlog(x) = sign(1e9) * ln(1e9 + 1) ≈ 20.72
# Clamped to 20.0 in twohot encode → bin k=254 (maximum bin)
# Decode: symexp(20.0) ≈ 485,165,195 (some precision loss)
```

For extremely large values, precision is reduced but gradients remain stable.

### Very Large Negative Values

```
x = -1e9
symlog(x) ≈ -20.72
# Clamped to -20.0 → bin k=0 (minimum bin)
```

Symmetric handling due to antisymmetry of symlog.

### Zero

```
x = 0
symlog(0) = 0
# Maps exactly to bin k=127 with weight 1.0 (exact bin center)
# decode(encode(0)) = 0 exactly
```

### Small Values

```
x = 0.001
symlog(0.001) ≈ 0.0009995 ≈ 0.001
# Near-identity behavior: twohot puts weight in bins around 0
# decode(encode(0.001)) ≈ 0.001  (high precision)
```

Small values are handled with high precision because the bins around 0 are evenly spaced
in symlog space, and symlog is nearly linear near 0.

### Negative Values

```
x = -42.0
symlog(-42.0) = -sign(42) * ln(43) ≈ -3.761
# Maps to a bin in the negative half of the range
```

Negative values are fully supported by the antisymmetry design.

---

## Implementation Checklist

When implementing SymlogTwohot, verify:

1. `bin_centers` registered as a buffer (not parameter): moves with `.to(device)`
2. `encode` clamps after symlog (not before): ensures we're clamping in log-space
3. `encode` returns values summing to 1.0 within floating-point tolerance
4. `loss` uses `log_softmax` (not `log(softmax(...))`)
5. `decode` uses `softmax` (not `log_softmax`)
6. Round-trip `decode(logits_from_encode(encode(x)))` ≈ x within relative tolerance
7. For x=0, `encode(0)[127] ≈ 1.0` (lands exactly on center bin, possibly split between two)
8. Loss is finite (no nan/inf) for all inputs including x=0 and extreme values
