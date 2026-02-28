# HTM Spatial Pooler (SP) -- Implementation Reference

This reference covers the upgraded Spatial Pooler for `brain-ai-dev`, building on
the existing `PytorchSpatialPooler` in `brain_ai/temporal/htm.py`.  It is
intended for Claude context injection when the skill needs to implement, debug,
or extend SP internals.

---

## 1. Overview

The Spatial Pooler converts arbitrary real-valued or binary input vectors into
stable **Sparse Distributed Representations (SDRs)**.  An SDR is a fixed-width
binary vector in which only a small fraction (typically 2 %) of bits are active.
SDRs have several desirable properties for downstream sequence learning:

* **Noise robustness** -- overlapping SDRs indicate semantic similarity; small
  input perturbations produce overlapping outputs.
* **Capacity** -- the number of unique SDRs with ~2 % sparsity over 16 384
  columns exceeds 10^600, far beyond practical collision risk.
* **Union property** -- the bitwise OR of several SDRs retains information
  about each constituent, enabling set-like reasoning.

The SP achieves these properties through three mechanisms:

1. **Permanence-based synaptic connections.**  Each minicolumn maintains a
   *potential pool* of candidate input bits.  A permanence scalar in [0, 1]
   tracks connection strength; only synapses above a threshold contribute to
   overlap.  Permanences are updated via Hebbian learning (no backprop).

2. **Competitive inhibition.**  After computing overlap scores, only the top-K
   columns (where K enforces the target sparsity) become active.  All other
   columns are suppressed, producing a sparse output.

3. **Duty-cycle boosting.**  Columns that rarely win the competition receive
   multiplicative boost factors that increase their effective overlap,
   preventing dead columns and encouraging full utilization of the column
   space.

### Relationship to the Existing Codebase

| Item | Location |
|------|----------|
| Current implementation | `brain_ai/temporal/htm.py`, class `PytorchSpatialPooler` (lines 96-241) |
| HTM config dataclass | `brain_ai/config.py`, class `HTMConfig` (lines 124-154) |
| HTM layer wrapper | `brain_ai/temporal/htm.py`, class `HTMLayer` |
| Accelerated HTM | `brain_ai/temporal/htm.py`, class `AcceleratedHTM` |

The skill upgrade targets four areas the current implementation lacks:

* Deterministic tie-breaking during inhibition.
* Explicit, configurable input binarization (the current code hard-codes a
  0.5 threshold in `HTMLayer._forward_pytorch`).
* Mixed-precision safety for permanence updates and overlap counts.
* Batch-safe duty-cycle tracking that does not leak per-sample statistics
  across batch items.

---

## 2. Input Binarization

The SP operates on binary input vectors.  Real-valued encoder outputs must be
converted to binary before entering the overlap computation.  The upgraded SP
supports four binarization modes, selectable via `SPConfig.binarization_mode`.

### 2.1 Mode: `passthrough`

Assume the input is already binary (values in {0, 1}).  No transformation is
applied.  Use when upstream encoders produce explicit SDRs (e.g., another SP or
a binary hash encoder).

```python
def binarize_passthrough(x: torch.Tensor) -> torch.Tensor:
    """No-op binarization.  Input must already be binary."""
    return x
```

### 2.2 Mode: `threshold`

Apply a fixed scalar threshold.  All values strictly greater than the threshold
become 1; the rest become 0.  Deterministic for a given threshold.

```python
def binarize_threshold(x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Fixed-threshold binarization."""
    return (x > threshold).float()
```

**Config fields:** `binarization_threshold` (default 0.5).

**Determinism guarantee:** Identical float inputs always produce the same binary
pattern because `>` is a pure comparison -- no randomness is involved.

### 2.3 Mode: `topk`

Select the top-K largest values and set them to 1; all others become 0.  This
guarantees exactly K active bits regardless of input magnitude.

```python
def binarize_topk(x: torch.Tensor, k: int) -> torch.Tensor:
    """Top-K binarization.  Exactly k bits are set to 1.

    Uses stable sort for deterministic tie-breaking.
    """
    assert x.dim() == 1, "Operate on single sample; batch handled externally."
    # Stable descending argsort: negate values, then stable ascending sort
    order = torch.argsort(-x, stable=True)
    binary = torch.zeros_like(x)
    binary[order[:k]] = 1.0
    return binary
```

**Config fields:** `binarization_topk` (default: `int(input_size * 0.1)`).

**Determinism guarantee:** `torch.argsort` with `stable=True` preserves the
relative order of equal elements, producing the same output for the same input
across calls.

### 2.4 Mode: `learned_gate`

Apply a learned sigmoid gate followed by a hard threshold.  This allows the
model to adapt the binarization boundary during training (the sigmoid
parameters are updated via backprop through the straight-through estimator).

```python
class LearnedBinarizer(nn.Module):
    """Learned sigmoid gate for input binarization.

    Forward: sigmoid(scale * (x - bias)) -> threshold at 0.5
    Gradient: straight-through estimator (STE) for the threshold step.
    """

    def __init__(self, input_size: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(input_size))
        self.scale = nn.Parameter(torch.ones(input_size) * 5.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.scale * (x - self.bias))
        # Hard threshold with STE
        hard = (gate > 0.5).float()
        return hard - gate.detach() + gate  # STE trick
```

**Config fields:** `binarization_mode = "learned_gate"`.  No additional
hyperparameters; the gate parameters are part of the module state.

**Determinism guarantee:** For a fixed set of learned parameters, the same float
input produces the same binary output.  During training, parameter updates are
deterministic given the same gradient trajectory.

### 2.5 Selecting the Binarization Mode

```python
def _make_binarizer(cfg: "SPConfig") -> Callable:
    if cfg.binarization_mode == "passthrough":
        return binarize_passthrough
    elif cfg.binarization_mode == "threshold":
        return functools.partial(binarize_threshold, threshold=cfg.binarization_threshold)
    elif cfg.binarization_mode == "topk":
        return functools.partial(binarize_topk, k=cfg.binarization_topk)
    elif cfg.binarization_mode == "learned_gate":
        return LearnedBinarizer(cfg.input_size)
    else:
        raise ValueError(f"Unknown binarization mode: {cfg.binarization_mode}")
```

---

## 3. Data Structures

All persistent state is stored as `register_buffer` tensors so that:

* They move with the module across devices (`model.cuda()`, `model.cpu()`).
* They are included in `state_dict()` for checkpointing.
* They are excluded from `parameters()` and do not receive gradient updates.

### 3.1 Potential Pool Indices

```python
# (N_columns, P) -- int32 indices of candidate input bits per column
# P = int(input_size * potential_pct) or min(potential_radius * 2 + 1, input_size)
self.register_buffer(
    "potential_idx",
    torch.zeros(column_count, P, dtype=torch.int32),
)
```

Each column samples P input bits as its potential pool at initialization.
Sampling is done once and frozen.  This avoids a dense `(N_columns x N_inputs)`
matrix, which at production scale (16 384 columns x 4096 inputs) would consume
256 MB in float32 -- most of it zeros.  The sparse representation uses
`16384 x P x 4 bytes` (with P ~ 3481 at 85 % potential, ~215 MB), but this is
all useful storage with no wasted entries.

**Initialization:**

```python
def _init_potential_pools(
    column_count: int,
    input_size: int,
    potential_pct: float,
    potential_radius: Optional[int],
    rng: torch.Generator,
) -> torch.Tensor:
    """Sample potential pool indices for each column.

    If potential_radius is None or >= input_size, sample uniformly from all
    input bits.  Otherwise, sample from a neighborhood centered on the
    column's topographic position.
    """
    P = int(input_size * potential_pct)
    idx = torch.zeros(column_count, P, dtype=torch.int32)
    for c in range(column_count):
        if potential_radius is None or potential_radius >= input_size:
            pool = torch.randperm(input_size, generator=rng)[:P]
        else:
            center = int(c * input_size / column_count)
            lo = max(0, center - potential_radius)
            hi = min(input_size, center + potential_radius + 1)
            candidates = torch.arange(lo, hi)
            perm = torch.randperm(len(candidates), generator=rng)[:P]
            pool = candidates[perm]
            # Pad if neighborhood is smaller than P
            if len(pool) < P:
                remaining = torch.tensor(
                    [i for i in range(input_size) if i not in set(pool.tolist())]
                )
                extra = remaining[torch.randperm(len(remaining), generator=rng)[:P - len(pool)]]
                pool = torch.cat([pool, extra])
        idx[c] = pool.to(torch.int32)
    return idx
```

### 3.2 Permanences

```python
# (N_columns, P) -- float permanence values, clamped to [0, 1]
self.register_buffer(
    "perm",
    torch.full((column_count, P), initial_permanence, dtype=torch.float32),
)
```

Permanences track connection strength for each potential synapse.  A synapse is
"connected" when its permanence meets or exceeds `perm_threshold` (default
0.5).  The connected mask is derived, not stored:

```python
connected_mask = self.perm >= self.perm_threshold  # (N_columns, P) bool
```

**Initialization options:**

| Strategy | Description |
|----------|-------------|
| Constant | All permanences start at `initial_permanence` (default 0.21). Simple, predictable. |
| Random uniform | `U(initial_permanence - 0.05, initial_permanence + 0.05)`. Breaks symmetry faster. |
| Distance-based | Higher permanence for topographically closer inputs. Encourages local receptive fields. |

The default is constant initialization, matching the existing codebase.

### 3.3 Boost Factors

```python
# (N_columns,) -- multiplicative boost, >= 0
self.register_buffer(
    "boost",
    torch.ones(column_count, dtype=torch.float32),
)
```

### 3.4 Duty Cycles

```python
# (N_columns,) -- EMA of column activity frequency
self.register_buffer(
    "active_duty_cycle",
    torch.zeros(column_count, dtype=torch.float32),
)

# (N_columns,) -- EMA of non-zero overlap frequency
self.register_buffer(
    "overlap_duty_cycle",
    torch.zeros(column_count, dtype=torch.float32),
)
```

### 3.5 Iteration Counter

```python
# Scalar -- total forward calls processed (for EMA warmup)
self.register_buffer(
    "iteration_count",
    torch.tensor(0, dtype=torch.long),
)
```

### 3.6 Summary Table

| Buffer | Shape | Dtype | Purpose |
|--------|-------|-------|---------|
| `potential_idx` | `(N_col, P)` | int32 | Indices of candidate input bits per column |
| `perm` | `(N_col, P)` | float32 | Permanence values in [0, 1] |
| `boost` | `(N_col,)` | float32 | Multiplicative boost factors |
| `active_duty_cycle` | `(N_col,)` | float32 | EMA of column activation rate |
| `overlap_duty_cycle` | `(N_col,)` | float32 | EMA of non-zero overlap rate |
| `iteration_count` | `()` | int64 | Global step counter |

**Never use a dense `(N_columns x N_inputs)` permanence matrix.**  At
production scale (16 384 columns, 4096 inputs), that is 256 MB of float32 --
and with 85 % potential connectivity, 15 % of it is wasted zeros that still
consume memory and compute during matmuls.  The sparse `potential_idx + perm`
layout is always preferred.

---

## 4. Inference Step (Forward Pass)

The forward pass has three stages: overlap computation, inhibition, and output
formatting.  All stages must be deterministic for a given input.

### 4.1 Overlap Computation

Compute the number of connected, active input bits for each column.

```python
def compute_overlap(
    self,
    input_bits: torch.Tensor,  # (N_input,) binary
) -> torch.Tensor:
    """Compute overlap scores for all columns.

    Returns:
        boosted_overlap: (N_col,) float32 overlap scores with boost applied.
    """
    # Gather input bits for each column's potential pool
    # potential_idx: (N_col, P) int32
    # input_bits:    (N_input,) float {0, 1}
    active_inputs = input_bits[self.potential_idx.long()]  # (N_col, P)

    # Connected mask: permanence >= threshold
    connected = self.perm >= self.perm_threshold  # (N_col, P) bool

    # Raw overlap: count of connected AND active synapses
    overlap = (connected & (active_inputs > 0.5)).sum(dim=-1)  # (N_col,) int

    # Stimulus threshold: zero out columns with insufficient overlap
    overlap = overlap.float()
    overlap[overlap < self.min_overlap] = 0.0

    # Apply boost factors
    boosted_overlap = overlap * self.boost  # (N_col,) float32

    return boosted_overlap
```

**Critical details:**

* The overlap count is computed as an integer sum of booleans.  Cast to float32
  only after summing (see Section 6 on mixed-precision).
* `min_overlap` (stimulus threshold) filters out columns with too few
  connections to the input.  Without it, noise columns with 1-2 coincidental
  overlaps can win the competition.  Default: 1.
* Boost factors are always float32.  Multiplication happens after the
  float cast.

### 4.2 Inhibition

Select the winning columns from the boosted overlap scores.

#### Global Inhibition (Default)

```python
def inhibit_global(
    self,
    boosted_overlap: torch.Tensor,  # (N_col,) float32
) -> torch.Tensor:
    """Global inhibition: top-K columns win.

    Uses deterministic tie-breaking via stable sort.

    Returns:
        active_columns: (K,) int64 indices of winning columns.
    """
    K = self.num_active  # int(N_columns * sparsity)

    # Deterministic tie-breaking: sort by (-overlap, column_id)
    # stable=True preserves original order for equal values,
    # meaning lower column indices win ties.
    order = torch.argsort(-boosted_overlap, stable=True)
    active_columns = order[:K]

    return active_columns
```

**Why deterministic tie-breaking matters.**  `torch.topk` does not guarantee a
stable ordering when multiple columns share the same boosted overlap.  On GPU
especially, the selection of winners among tied columns can vary between runs.
For online Hebbian learning, non-deterministic column selection means the same
input can produce different SDRs on different runs, which breaks downstream
sequence learning that depends on stable representations.  Using
`torch.argsort(..., stable=True)` followed by slicing the top-K ensures
reproducibility.

**K computation:**

```python
K = int(self.column_count * self.sparsity)
# Example: 16384 columns * 0.02 = 328 active columns
```

K is computed once at initialization and stored as `self.num_active`.

#### Local Inhibition (Optional)

```python
def inhibit_local(
    self,
    boosted_overlap: torch.Tensor,  # (N_col,) float32
    local_radius: int,
) -> torch.Tensor:
    """Local inhibition: top-K within each column's neighborhood.

    Each column competes only with columns within `local_radius`
    positions.  A column wins if it is in the top-K of its neighborhood.

    Returns:
        active_columns: (variable,) int64 indices of winning columns.
    """
    N = len(boosted_overlap)
    K_local = max(1, int((2 * local_radius + 1) * self.sparsity))
    winners = []

    for c in range(N):
        lo = max(0, c - local_radius)
        hi = min(N, c + local_radius + 1)
        neighborhood = boosted_overlap[lo:hi]
        rank = (neighborhood > boosted_overlap[c]).sum()
        if rank < K_local and boosted_overlap[c] > 0:
            winners.append(c)

    return torch.tensor(winners, dtype=torch.long, device=boosted_overlap.device)
```

Local inhibition produces variable-density outputs (not exactly K active
columns).  It is useful for topographic maps but slower than global inhibition.
Default: global.

### 4.3 Output Formatting

The SP can return the active column SDR in two forms:

```python
def indices_to_dense(
    active_columns: torch.Tensor,  # (K,) int64
    N_columns: int,
) -> torch.Tensor:
    """Convert active column indices to a dense binary vector.

    Returns:
        sdr: (N_columns,) float {0, 1}
    """
    sdr = torch.zeros(N_columns, device=active_columns.device)
    sdr[active_columns] = 1.0
    return sdr
```

The forward method returns a dense tensor by default (for compatibility with
the existing `HTMLayer` which passes it to `PytorchTemporalMemory`).
An optional `return_indices=True` flag can return the sparse index form for
memory efficiency.

### 4.4 Complete Forward (Single Sample)

```python
def _forward_single(
    self,
    x: torch.Tensor,       # (N_input,) raw input
    learn: bool = True,
) -> torch.Tensor:
    """Full forward pass for a single input sample.

    Returns:
        active_columns_dense: (N_col,) binary SDR
    """
    # Step 1: Binarize input
    input_bits = self.binarizer(x)

    # Step 2: Compute overlap
    boosted_overlap = self.compute_overlap(input_bits)

    # Step 3: Inhibition
    if self.inhibition_mode == "local":
        active_idx = self.inhibit_local(boosted_overlap, self.local_radius)
    else:
        active_idx = self.inhibit_global(boosted_overlap)

    # Step 4: Convert to dense
    active_dense = indices_to_dense(active_idx, self.column_count)

    # Step 5: Online learning
    if learn and self.training:
        self._learn(input_bits, active_idx)
        self._update_duty_cycles(active_idx)
        self._update_boost()

    return active_dense
```

---

## 5. Online Learning (Hebbian, No Backprop)

SP learning is purely Hebbian: strengthen connections to co-active inputs,
weaken connections to inactive inputs.  No loss function, no optimizer, no
gradient tape.

### 5.1 Permanence Updates

For each active column, adjust the permanences of its potential pool based on
whether each corresponding input bit was active.

```python
def _learn(
    self,
    input_bits: torch.Tensor,   # (N_input,) binary
    active_idx: torch.Tensor,   # (K,) int64 -- indices of active columns
) -> None:
    """Hebbian permanence update for active columns.

    All math is performed in float32 regardless of storage dtype.
    """
    # Gather active input bits for each active column's pool
    # potential_idx[active_idx]: (K, P) int32
    pool_inputs = input_bits[self.potential_idx[active_idx].long()]  # (K, P)

    # Cast permanences to float32 for update math
    perm_slice = self.perm[active_idx].float()  # (K, P)

    # Hebbian update
    active_mask = pool_inputs > 0.5  # (K, P) bool
    perm_slice[active_mask] += self.perm_inc
    perm_slice[~active_mask] -= self.perm_dec

    # Clamp to [0, 1]
    perm_slice.clamp_(0.0, 1.0)

    # Write back (cast to storage dtype if using fp16 storage)
    self.perm[active_idx] = perm_slice.to(self.perm.dtype)
```

**Why update only active columns?**  In the existing codebase
(`PytorchSpatialPooler.learn`), the update is applied to all columns via a
broadcast mask (`active_mask = active_columns.unsqueeze(-1)`), but the mask
zeros out the contribution for inactive columns.  The result is equivalent, but
the broadcast approach processes all `N_col x N_input` entries.  Updating only
the K active columns processes `K x P` entries -- typically 100x fewer.

### 5.2 Duty Cycle Updates

Duty cycles are exponential moving averages (EMAs) that track how often each
column is active.

```python
def _update_duty_cycles(
    self,
    active_idx: torch.Tensor,  # (K,) int64
) -> None:
    """Update duty cycles with EMA.

    Duty cycles are GLOBAL statistics, not per-sample.  When processing
    a batch, call this once per batch with the mean activity, not once
    per sample.
    """
    self.iteration_count += 1
    period = min(self.duty_cycle_period, self.iteration_count.item())
    alpha = 1.0 / period

    # Build activity indicator
    is_active = torch.zeros(self.column_count, device=self.perm.device)
    is_active[active_idx] = 1.0

    # EMA update
    self.active_duty_cycle.mul_(1.0 - alpha).add_(is_active, alpha=alpha)
```

**EMA warmup:** During the first `duty_cycle_period` iterations, the effective
window is `min(period, iteration_count)`, which prevents the initial duty
cycles from being dominated by the very first inputs.

**Overlap duty cycle** (optional, for min-overlap enforcement):

```python
    # overlap_duty_cycle: fraction of iterations where a column had non-zero overlap
    has_overlap = torch.zeros(self.column_count, device=self.perm.device)
    # (computed from the overlap values before boost, stored from compute_overlap)
    has_overlap[self._raw_overlap > 0] = 1.0
    self.overlap_duty_cycle.mul_(1.0 - alpha).add_(has_overlap, alpha=alpha)
```

### 5.3 Boosting

Boost factors adjust the effective overlap of under-utilized columns upward
and over-utilized columns downward.

```python
def _update_boost(self) -> None:
    """Recompute boost factors from active duty cycles.

    boost = exp(-boost_strength * (active_duty_cycle - target_density))

    - Under-utilized columns (duty < target): boost > 1 (more competitive)
    - Over-utilized columns (duty > target): boost < 1 (less competitive)
    - Perfectly-utilized columns (duty == target): boost == 1
    """
    target_density = self.sparsity
    self.boost = torch.exp(
        -self.boost_strength * (self.active_duty_cycle - target_density)
    )
```

**Note on sign convention.**  The Numenta reference uses
`exp(boost_strength * (target - duty))`.  The negated form
`exp(-boost_strength * (duty - target))` is algebraically identical.  The
existing `PytorchSpatialPooler._update_boosting` in `brain_ai/temporal/htm.py`
uses `exp(boost_strength * (target - duty))` (line 202-204).  Both are correct.
Choose one form and remain consistent.

**`boost_strength` tuning:**

| Value | Behavior |
|-------|----------|
| 0.0 | Boosting disabled.  Duty cycles still tracked but have no effect. |
| 1.0 | Gentle.  Columns that are half as active as target get boost ~1.6. |
| 3.0 | Default.  Columns at half target get boost ~4.5.  Good balance. |
| 10.0 | Aggressive.  Can cause oscillation (columns alternate on/off). |

### 5.4 Min-Overlap Enforcement (Optional)

If a column's `overlap_duty_cycle` falls below a minimum threshold, increase
its permanences globally to make more synapses connected:

```python
def _enforce_min_overlap(self) -> None:
    """Raise permanences for columns with chronically low overlap."""
    min_duty = self.sparsity * 0.01  # e.g., 0.0002
    low_columns = self.overlap_duty_cycle < min_duty
    if low_columns.any():
        self.perm[low_columns] += 0.1 * self.perm_threshold
        self.perm.clamp_(0.0, 1.0)
```

This is a secondary recovery mechanism.  Boosting handles most duty-cycle
issues; min-overlap enforcement handles the extreme case where a column has so
few connected synapses that it never achieves any overlap at all.

---

## 6. Mixed-Precision Safety

When training with AMP (Automatic Mixed Precision), certain SP operations
must remain in float32 to prevent numerical issues.

### 6.1 Overlap Counts

Overlap counts are integer sums of boolean masks.  In float16, integers above
1024 lose precision (float16 has 10 mantissa bits).  With P ~ 3481 potential
synapses per column, the maximum overlap is 3481 -- well beyond float16's
exact integer range.

**Rule: Always compute overlap in int32 or float32.**

```python
# CORRECT
overlap = (connected & active_inputs).to(torch.int32).sum(dim=-1)

# ALSO CORRECT (bool sum promotes to int64 by default)
overlap = (connected & active_inputs).sum(dim=-1)

# WRONG -- float16 sum loses precision for counts > 1024
overlap = (connected.half() * active_inputs.half()).sum(dim=-1)
```

### 6.2 Permanence Storage vs. Update Math

Permanences can be stored in float16 to save memory (cuts permanence buffer
from 256 MB to 128 MB at production scale).  However, permanence updates must
be computed in float32.  Repeated small increments (0.1, -0.1) in float16
suffer from rounding drift: `half(0.21) + half(0.1)` may not equal
`half(0.31)` exactly, and over thousands of iterations the error accumulates.

```python
# CORRECT: cast to fp32 for math, cast back for storage
perm_fp32 = self.perm[active_idx].float()
perm_fp32[active_mask] += self.perm_inc
perm_fp32[~active_mask] -= self.perm_dec
perm_fp32.clamp_(0.0, 1.0)
self.perm[active_idx] = perm_fp32.half()  # or .to(self.perm.dtype)

# WRONG: update in fp16 directly
self.perm[active_idx, active_mask] += self.perm_inc  # cumulative drift
```

### 6.3 Boost Factors

Boost factors involve `torch.exp()`.  In float16, exp overflows at ~11.09
(`exp(11.09) > 65504`).  With `boost_strength=3.0` and `target_density=0.02`,
a completely dead column (`duty=0`) gets `exp(3.0 * 0.02) = exp(0.06) ~ 1.06`
-- safe.  But with `boost_strength=10.0` and a column at zero duty,
`exp(10.0 * 0.02) = exp(0.2) ~ 1.22` -- still safe.  The danger arises if
`target_density` is large (unlikely for SP) or if custom configurations push
the exponent high.

**Rule: Always compute and store boost factors in float32.**

### 6.4 Top-K Selection

The boosted overlap values used for `argsort` or `topk` must be float32.
Float16 quantization can change the ranking of columns with similar overlaps.

```python
# CORRECT
boosted_overlap = overlap.float() * self.boost  # both float32
order = torch.argsort(-boosted_overlap, stable=True)

# WRONG
boosted_overlap = overlap.half() * self.boost.half()  # precision loss
```

### 6.5 AMP Autocast Exclusion

Wrap the SP forward pass to ensure it is excluded from AMP autocasting:

```python
@torch.amp.custom_fwd(device_type="cuda", cast_output=torch.float32)
def forward(self, x: torch.Tensor, learn: bool = True) -> torch.Tensor:
    # All internal computation in float32
    ...
```

Alternatively, use `torch.amp.autocast` with `enabled=False` around the
critical sections:

```python
def forward(self, x, learn=True):
    with torch.amp.autocast("cuda", enabled=False):
        x = x.float()
        return self._forward_impl(x, learn)
```

---

## 7. Batch Processing

The SP is inherently a per-sample algorithm: each input may activate a
different set of columns, and Hebbian learning is sequential (the permanences
updated by sample N affect the overlap for sample N+1 in online learning).

### 7.1 Batched Forward with Explicit Loop

```python
def forward(
    self,
    x: torch.Tensor,       # (B, N_input) or (N_input,)
    learn: bool = True,
) -> torch.Tensor:
    """Process input through spatial pooler.

    For batched input, process each sample sequentially.
    Duty cycles are updated once per batch using mean activity.
    """
    if x.dim() == 1:
        return self._forward_single(x, learn)

    B = x.shape[0]
    outputs = []
    batch_active_counts = torch.zeros(
        self.column_count, device=x.device, dtype=torch.float32
    )

    for i in range(B):
        # Binarize
        input_bits = self.binarizer(x[i])

        # Overlap + inhibition
        boosted_overlap = self.compute_overlap(input_bits)
        active_idx = self.inhibit_global(boosted_overlap)
        active_dense = indices_to_dense(active_idx, self.column_count)
        outputs.append(active_dense)

        # Accumulate activity for batch-level duty cycle
        batch_active_counts[active_idx] += 1.0

        # Per-sample permanence update (if learning)
        if learn and self.training:
            self._learn(input_bits, active_idx)

    # Batch-level duty cycle update (once, using mean activity)
    if learn and self.training:
        mean_activity = batch_active_counts / B
        self._update_duty_cycles_from_mean(mean_activity)
        self._update_boost()

    return torch.stack(outputs)  # (B, N_col)
```

### 7.2 Batch-Safe Duty Cycle Accumulation

Duty cycles are global statistics that should reflect the overall distribution
of column usage, not be dominated by any single batch item.

```python
def _update_duty_cycles_from_mean(
    self,
    mean_activity: torch.Tensor,  # (N_col,) float, mean over batch
) -> None:
    """Update duty cycles using batch-mean activity.

    This ensures that duty cycles reflect the average behavior across
    the batch, not the last sample seen.
    """
    self.iteration_count += 1
    period = min(self.duty_cycle_period, self.iteration_count.item())
    alpha = 1.0 / period

    self.active_duty_cycle.mul_(1.0 - alpha).add_(mean_activity, alpha=alpha)
```

**Why not update per sample?**  If duty cycles are updated after each sample in
a batch, the duty cycles at the end of the batch reflect a sliding window that
overweights later samples.  For a batch of 32 samples with `period=1000`, the
last sample has ~31x more influence than the first.  Using the batch mean
treats all samples equally.

### 7.3 vmap Alternative (Experimental)

For inference only (no learning), `torch.vmap` can vectorize the overlap
computation:

```python
def _compute_overlap_vmapped(self, batch_input_bits):
    """Vectorized overlap for inference (no learning)."""

    def single_overlap(input_bits):
        active_inputs = input_bits[self.potential_idx.long()]
        connected = self.perm >= self.perm_threshold
        overlap = (connected & (active_inputs > 0.5)).sum(dim=-1).float()
        overlap[overlap < self.min_overlap] = 0.0
        return overlap * self.boost

    return torch.vmap(single_overlap)(batch_input_bits)
```

This does not work for learning because vmap does not support in-place mutation
of module state.

---

## 8. Migration from Existing Code

The existing `PytorchSpatialPooler` in `brain_ai/temporal/htm.py` (lines
96-241) uses a dense permanence matrix and lacks several features.  This
section details the required changes.

### 8.1 Current Implementation Issues

| Issue | Current Code | Upgrade |
|-------|-------------|---------|
| Dense permanence matrix | `permanences: (N_col, N_input)` float | Sparse: `potential_idx: (N_col, P)` int32 + `perm: (N_col, P)` float |
| Dense potential mask | `potential_mask: (N_col, N_input)` float | Eliminated; implicit in `potential_idx` |
| No binarization config | Hard-coded `(x > 0.5).float()` in `HTMLayer` | Configurable mode: passthrough, threshold, topk, learned_gate |
| Non-deterministic inhibition | `torch.topk(overlap, K)` | `torch.argsort(-overlap, stable=True)[:K]` |
| No stimulus threshold | All columns with any overlap compete | `overlap[overlap < min_overlap] = 0` |
| Plain attributes for state | `boost_factors`, `active_duty_cycles` as buffers (OK) but `permanences` also buffer (OK) | All state as buffers with explicit dtype specification |
| No fp32 enforcement | Update math uses whatever dtype input arrives as | Cast to fp32 for all update math |
| Per-sample duty cycles | `_update_boosting` called per sample | Batch-mean accumulation with single duty-cycle update |

### 8.2 Migration Steps

1. **Add `SPConfig` dataclass** with all new fields (see Section 9).  Wire it
   into the existing `HTMConfig` as a sub-config or extend `HTMConfig`.

2. **Replace dense permanence storage** with `potential_idx` + `perm`.  Update
   `compute_overlap` to use gather-based indexing instead of matmul.

3. **Add binarization dispatch.**  Create `_make_binarizer` and call it in
   `__init__`.  Remove the hard-coded threshold from `HTMLayer._forward_pytorch`.

4. **Replace `torch.topk` with `torch.argsort(..., stable=True)`** in
   the `inhibit` method.

5. **Add `min_overlap` parameter** and apply stimulus threshold in
   `compute_overlap`.

6. **Enforce fp32 in `_learn`.**  Cast permanence slice to float32 before
   update, cast back to storage dtype after.

7. **Add batch-safe duty cycle accumulation.**  Accumulate activity across
   batch, update duty cycle once per batch with mean activity.

8. **Register all state tensors with explicit dtypes.**  Ensure `potential_idx`
   is int32, `perm` is float32 (or float16 for storage with fp32 update),
   `boost` is float32, duty cycles are float32.

9. **Update `HTMLayer._init_pytorch`** to pass the new config fields to the
   upgraded SP constructor.

10. **Add unit tests** covering:
    - Deterministic output: same input produces same SDR across 100 calls.
    - Sparsity: output has exactly K active bits.
    - Binarization modes: each mode produces correct binary patterns.
    - Mixed precision: SP produces identical results under `torch.amp.autocast`.
    - Batch duty cycles: duty cycle after batch-of-32 equals duty cycle after
      32 sequential single-sample calls (within tolerance).

### 8.3 Backward Compatibility

The upgraded SP must remain a drop-in replacement.  The `forward` method
signature does not change: `forward(x, learn=True) -> Tensor`.  The output
shape `(N_col,)` or `(B, N_col)` does not change.  `state_dict()` keys will
change (new buffer names), so provide a migration function:

```python
@classmethod
def from_legacy_state_dict(cls, state_dict: dict, cfg: "SPConfig") -> "UpgradedSP":
    """Load weights from old PytorchSpatialPooler state_dict.

    Maps dense permanence matrix to sparse potential_idx + perm.
    """
    sp = cls(cfg)
    old_perm = state_dict["permanences"]          # (N_col, N_input)
    old_mask = state_dict["potential_mask"]        # (N_col, N_input)

    for c in range(cfg.column_count):
        pool = torch.where(old_mask[c] > 0.5)[0]
        P = sp.potential_idx.shape[1]
        if len(pool) > P:
            pool = pool[:P]
        elif len(pool) < P:
            remaining = torch.where(old_mask[c] <= 0.5)[0]
            extra = remaining[:P - len(pool)]
            pool = torch.cat([pool, extra])
        sp.potential_idx[c] = pool.to(torch.int32)
        sp.perm[c] = old_perm[c, pool.long()]

    sp.boost.copy_(state_dict.get("boost_factors", torch.ones(cfg.column_count)))
    sp.active_duty_cycle.copy_(state_dict.get("active_duty_cycles",
                                               torch.zeros(cfg.column_count)))
    sp.iteration_count.copy_(state_dict.get("iteration_count", torch.tensor(0)))

    return sp
```

---

## 9. Configuration Surface

### 9.1 SPConfig Dataclass

```python
from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class SPConfig:
    """Spatial Pooler configuration.

    All fields have sensible defaults matching Numenta's recommended
    parameters for a 2 % sparsity, 16384-column production SP.
    """

    # ----- Topology -----
    input_size: int = 4096
    """Number of input bits.  Must match encoder output dimension."""

    column_count: int = 16384
    """Number of minicolumns.  More columns = more capacity, more memory."""

    potential_radius: Optional[int] = None
    """Radius for topographic potential pool sampling.
    None means global (all inputs are candidates).  Set to a positive
    integer for local receptive fields."""

    potential_pct: float = 0.85
    """Fraction of inputs within radius to include in each column's pool.
    P = int(input_size * potential_pct)."""

    # ----- Sparsity -----
    sparsity: float = 0.02
    """Target fraction of active columns.  K = int(column_count * sparsity).
    Must be in (0, 1).  Typical values: 0.02 (2 %)."""

    # ----- Permanences -----
    perm_inc: float = 0.1
    """Permanence increment for connected + active synapses.
    Must be > 0.  Larger values = faster learning, less stability."""

    perm_dec: float = 0.05
    """Permanence decrement for connected + inactive synapses.
    Must be > 0.  Asymmetric with perm_inc (0.05 vs 0.1) for slower
    forgetting and better stability under online learning."""

    perm_threshold: float = 0.5
    """Permanence value at which a synapse is considered connected.
    Also called 'permanence_connected' in the existing codebase."""

    initial_permanence: float = 0.21
    """Initial permanence for all potential synapses.
    Should be below perm_threshold so columns start mostly disconnected
    and learn their receptive fields."""

    # ----- Stimulus -----
    min_overlap: int = 1
    """Minimum overlap score for a column to participate in inhibition.
    Columns with overlap < min_overlap are zeroed out.  Prevents noise
    columns from winning with 1-2 coincidental connections."""

    # ----- Boosting -----
    boost_strength: float = 3.0
    """Exponential boost factor.  Controls how aggressively under-utilized
    columns are boosted.  0 disables boosting."""

    duty_cycle_period: int = 1000
    """EMA window for duty cycle computation.  Larger = smoother, slower
    to respond to distribution shifts."""

    # ----- Binarization -----
    binarization_mode: Literal["passthrough", "threshold", "topk", "learned_gate"] = "topk"
    """How to convert real-valued input to binary.
    - passthrough: assume already binary
    - threshold: fixed threshold (binarization_threshold)
    - topk: top-K bits (binarization_topk)
    - learned_gate: sigmoid + STE threshold (trainable)"""

    binarization_topk: Optional[int] = None
    """Number of top bits to activate in topk mode.
    Default: int(input_size * 0.1)."""

    binarization_threshold: float = 0.5
    """Threshold for 'threshold' binarization mode."""

    # ----- Inhibition -----
    inhibition_mode: Literal["global", "local"] = "global"
    """Inhibition strategy.
    - global: top-K across all columns (fast, uniform density)
    - local: top-K within each column's neighborhood (slower, variable density)"""

    local_radius: Optional[int] = None
    """Neighborhood radius for local inhibition.  Only used when
    inhibition_mode == 'local'.  Default: column_count // 10."""

    def __post_init__(self):
        """Validate configuration and fill defaults."""
        assert 0 < self.sparsity < 1, (
            f"sparsity must be in (0,1), got {self.sparsity}"
        )
        assert self.column_count > 0, (
            f"column_count must be positive, got {self.column_count}"
        )
        assert self.input_size > 0, (
            f"input_size must be positive, got {self.input_size}"
        )
        assert self.perm_inc > 0, (
            f"perm_inc must be positive, got {self.perm_inc}"
        )
        assert self.perm_dec > 0, (
            f"perm_dec must be positive, got {self.perm_dec}"
        )
        assert 0 < self.perm_threshold < 1, (
            f"perm_threshold must be in (0,1), got {self.perm_threshold}"
        )
        assert 0 < self.initial_permanence <= 1, (
            f"initial_permanence must be in (0,1], got {self.initial_permanence}"
        )
        assert self.min_overlap >= 0, (
            f"min_overlap must be non-negative, got {self.min_overlap}"
        )
        assert self.boost_strength >= 0, (
            f"boost_strength must be non-negative, got {self.boost_strength}"
        )
        assert self.duty_cycle_period > 0, (
            f"duty_cycle_period must be positive, got {self.duty_cycle_period}"
        )

        if self.binarization_topk is None:
            self.binarization_topk = max(1, int(self.input_size * 0.1))

        if self.local_radius is None:
            self.local_radius = max(1, self.column_count // 10)

        if self.potential_radius is not None:
            assert self.potential_radius > 0, (
                f"potential_radius must be positive, got {self.potential_radius}"
            )
```

### 9.2 Field Defaults Summary

| Field | Default | Constraint |
|-------|---------|------------|
| `input_size` | 4096 | > 0 |
| `column_count` | 16384 | > 0 |
| `potential_radius` | None (global) | > 0 or None |
| `potential_pct` | 0.85 | (0, 1] |
| `sparsity` | 0.02 | (0, 1) |
| `perm_inc` | 0.1 | > 0 |
| `perm_dec` | 0.05 | > 0 |
| `perm_threshold` | 0.5 | (0, 1) |
| `initial_permanence` | 0.21 | (0, 1] |
| `min_overlap` | 1 | >= 0 |
| `boost_strength` | 3.0 | >= 0 |
| `duty_cycle_period` | 1000 | > 0 |
| `binarization_mode` | `"topk"` | one of 4 modes |
| `binarization_topk` | `input_size * 0.1` | > 0 |
| `binarization_threshold` | 0.5 | any float |
| `inhibition_mode` | `"global"` | `"global"` or `"local"` |
| `local_radius` | `column_count // 10` | > 0 |

### 9.3 Integration with Existing HTMConfig

The `SPConfig` can be embedded in the existing `HTMConfig` as a sub-field:

```python
@dataclass
class HTMConfig:
    # ... existing fields ...
    sp: SPConfig = field(default_factory=SPConfig)
```

Or, for backward compatibility, a factory function can construct `SPConfig`
from `HTMConfig` fields:

```python
def sp_config_from_htm(htm_cfg: HTMConfig, input_size: int) -> SPConfig:
    return SPConfig(
        input_size=input_size,
        column_count=htm_cfg.column_count,
        sparsity=htm_cfg.sparsity,
        perm_inc=htm_cfg.permanence_inc,
        perm_dec=htm_cfg.permanence_dec,
        perm_threshold=getattr(htm_cfg, "permanence_connected", 0.5),
    )
```

---

## 10. Anti-Patterns

Avoid the following mistakes when implementing or modifying the Spatial Pooler.

### 10.1 Dense Permanence Matrix

```python
# WRONG: Dense (N_col x N_input) matrix
self.register_buffer("permanences", torch.rand(column_count, input_size))
```

At production scale (16 384 columns, 4096 inputs), this is 256 MB of float32.
With 85 % potential connectivity, 15 % of entries are wasted zeros.  The matmul
in `compute_overlap` processes all entries, including zeros.  Use sparse
`potential_idx` + `perm` instead.

### 10.2 Non-Deterministic Tie-Breaking

```python
# WRONG: torch.topk does not have stable ordering for ties
_, top_indices = torch.topk(overlap, K)
```

Replace with:

```python
# CORRECT: stable argsort ensures reproducibility
order = torch.argsort(-overlap, stable=True)
top_indices = order[:K]
```

### 10.3 Duty Cycle Per Batch Item

```python
# WRONG: updating duty cycle inside the per-sample loop
for i in range(batch_size):
    active = self._forward_single(x[i])
    self._update_duty_cycles(active)  # leaks per-sample ordering
```

Later samples in the batch have disproportionate influence on duty cycles.
Instead, accumulate activity over the batch and update once:

```python
# CORRECT
for i in range(batch_size):
    active_idx = ...
    batch_activity[active_idx] += 1.0
self._update_duty_cycles_from_mean(batch_activity / batch_size)
```

### 10.4 Permanence Updates in fp16

```python
# WRONG: cumulative drift from repeated small updates
self.perm += 0.1  # if self.perm is half-precision
```

Always cast to float32 for update math:

```python
# CORRECT
p = self.perm.float()
p += 0.1
self.perm = p.half()
```

### 10.5 Missing Stimulus Threshold

```python
# WRONG: no minimum overlap check
boosted_overlap = overlap * self.boost
winners = torch.argsort(-boosted_overlap, stable=True)[:K]
```

Without a stimulus threshold, columns with 1-2 coincidental connections can
win when overall input activity is low.  Always apply:

```python
# CORRECT
overlap[overlap < self.min_overlap] = 0.0
boosted_overlap = overlap * self.boost
```

### 10.6 Unclamped Permanences

```python
# WRONG: permanences drift outside [0, 1]
self.perm[active_mask] += self.perm_inc
self.perm[~active_mask] -= self.perm_dec
# Missing: self.perm.clamp_(0.0, 1.0)
```

Negative permanences are meaningless and cause connected-mask logic errors.
Permanences above 1.0 create artificially strong connections that resist
unlearning.  Always clamp after update:

```python
# CORRECT
self.perm.clamp_(0.0, 1.0)
```

### 10.7 Boosting Without Duty Cycle Warmup

```python
# WRONG: boost from iteration 1 with alpha=1.0/1000
alpha = 1.0 / self.duty_cycle_period  # e.g., 0.001
self.active_duty_cycle = (1 - alpha) * self.active_duty_cycle + alpha * is_active
```

On the first iteration, `active_duty_cycle` is zero.  The EMA update produces
`0.001 * is_active`, which is far below the target density of 0.02.  This
means *all* columns get a massive boost on early iterations, then boost
collapses rapidly as the EMA warms up.

```python
# CORRECT: use adaptive alpha during warmup
period = min(self.duty_cycle_period, self.iteration_count.item())
alpha = 1.0 / period
```

On iteration 1, `alpha = 1.0` -- the duty cycle is set directly to the
activity pattern.  By iteration 1000, `alpha = 0.001` -- normal EMA behavior.

### 10.8 Learning During Inference

```python
# WRONG: permanences change during model inference mode
# model set to non-training mode, but learn=True by default
output = model(x)  # still calls learn() because learn=True by default
```

The SP `learn` parameter defaults to `True`.  During inference, always pass
`learn=False`, or guard the learning call with `self.training`:

```python
# CORRECT (in forward):
if learn and self.training:
    self._learn(input_bits, active_idx)
```

### 10.9 Ignoring potential_pct Interaction with potential_radius

```python
# WRONG: potential_pct applied to full input_size even with small radius
P = int(input_size * potential_pct)  # 4096 * 0.85 = 3481
# But potential_radius = 100 means only 201 candidates exist
```

When `potential_radius` limits the candidate pool, P may exceed the number of
available candidates.  The initialization code must handle this:

```python
# CORRECT
candidates_in_radius = 2 * potential_radius + 1
P = min(int(input_size * potential_pct), candidates_in_radius)
```

### 10.10 Modifying potential_idx After Initialization

The potential pool indices are sampled once at construction time and frozen.
They define the SP's topology.  Changing them after training has begun
invalidates all learned permanences (the permanence at position `[c, j]`
corresponds to `potential_idx[c, j]` -- if the index changes, the permanence
maps to a different input bit).

```python
# WRONG: reshuffling potential pools during training
self.potential_idx = new_random_indices  # breaks learned permanences
```

If the topology must change (e.g., pruning dead columns), reinitialize the
permanences for affected columns to `initial_permanence`.
