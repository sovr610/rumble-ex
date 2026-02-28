# Working Memory: Persistent State, Continuous-Time Dynamics, Ignition-Gated Writes

## 1. Overview

Working Memory maintains persistent state across timesteps, integrating workspace content over time. It is the temporal backbone of the Global Workspace: without it, each selection-broadcast cycle would be memoryless, unable to track task context, accumulate evidence, or detect novelty against a baseline.

Primary backend: CfC (Closed-form Continuous-time) from the ncps library, providing neural circuit policies with continuous-time dynamics, bounded gradients, and native support for irregular time sampling. Secondary backend: LTC (Liquid Time-Constant), also from ncps, offering greater expressiveness at the cost of speed. Fallback: standard PyTorch GRU, used when ncps is unavailable.

The module must provide seamless fallback with identical input/output contracts. No downstream code branches on backend type. The forward signature, state shape contract, and return dictionary are the same regardless of whether the backend is CfC, LTC, or GRU.

### Current state of `brain_ai/workspace/working_memory.py`

The existing implementation (424 lines) provides the core structure but has seven gaps that must be closed before the working memory module meets the full contract required by the global-workspace-ignition skill:

1. No `detach_state()` method for truncated BPTT
2. No `dt`/timespans handling wired through (CfC supports it but the plumbing is incomplete)
3. No ignition-gated memory write (all writes are unconditional)
4. No explicit `reset_state(batch_size, device, dtype)` with all three parameters
5. Memory buffer not tied to ignition (should write more persistently on ignition, suppress on non-ignition)
6. No state norms telemetry for drift detection
7. Memory buffer capacity hardcoded to 7 instead of configurable from config

---

## 2. Backend Selection

### 2.1 CfC (Closed-form Continuous-time)

CfC networks (Hasani et al., 2022) solve the ODE governing neural circuit dynamics in closed form, yielding a recurrent cell that:

- Handles irregular time sampling via the `timespans` parameter, making it suitable for event-driven or variable-rate inputs
- Uses AutoNCP wiring with configurable sparsity, creating a sparse recurrent connectivity pattern that mirrors biological neural circuits
- Produces bounded gradients by construction, avoiding the vanishing/exploding gradient problem that plagues vanilla RNNs over long sequences
- Runs faster than LTC because the ODE solution is analytical rather than numerical

CfC is the default and recommended backend. It provides the best balance of speed, stability, and temporal expressiveness.

### 2.2 LTC (Liquid Time-Constant)

LTC networks (Hasani et al., 2021) use a numerical ODE solver with adaptive time constants per neuron:

- More expressive than CfC due to the full ODE integration
- Slower at inference because each forward step requires numerical integration
- Better for very long-range dependencies where CfC's closed-form approximation may lose fidelity
- Also supports the `timespans` parameter, though the current code does not pass it through for LTC (a gap that must be fixed)

Use LTC when the task demands extremely long temporal dependencies (hundreds to thousands of steps) and inference speed is not the primary constraint.

### 2.3 GRU Fallback

Standard `torch.nn.GRU` provides temporal integration when ncps is unavailable:

- Identical input/output contract to CfC and LTC
- Does not natively support irregular time sampling (see Section 4.3 for dt handling)
- Well-understood optimization characteristics, wide hardware support
- Used automatically when `NCPS_AVAILABLE` is `False`

### 2.4 Seamless Fallback Requirement

Detect ncps availability exactly once, at import time. If missing, instantiate GRU and set `wm_backend="GRU_fallback"` in the dependencies report. All downstream code uses the same interface without branching.

```python
try:
    from ncps.torch import CfC, LTC
    from ncps.wirings import AutoNCP
    NCPS_AVAILABLE = True
except ImportError:
    NCPS_AVAILABLE = False
```

This pattern already exists in the current codebase at lines 17-22 of `brain_ai/workspace/working_memory.py`. Preserve it exactly. Do not add a second import attempt elsewhere. Do not catch exceptions beyond `ImportError`. Do not retry on failure.

The `WorkingMemory` class selects the backend in `__init__()`:

```python
class WorkingMemory(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if NCPS_AVAILABLE and config.mode in ("cfc", "ltc"):
            self.backend = LiquidWorkingMemory(
                input_dim=config.input_dim,
                hidden_dim=config.hidden_dim,
                output_dim=config.output_dim,
                mode=config.mode,
                num_units=config.num_units,
                sparsity=config.sparsity,
            )
            self.backend_type = config.mode
        else:
            self.backend = GRUWorkingMemory(
                input_dim=config.input_dim,
                hidden_dim=config.hidden_dim,
                output_dim=config.output_dim,
            )
            self.backend_type = "gru"
```

After this point, `self.backend_type` is used exclusively for telemetry and logging. No forward-path logic branches on `self.backend_type`. The forward pass calls `self.backend(...)` unconditionally.

---

## 3. Public Interface

### 3.1 forward(slots_t, state, timespans=None, ignition_gain=None) -> (wm_out, next_state)

The primary entry point. Process a single timestep of workspace content through working memory.

**Parameters:**

- `slots_t`: `(B, K, D)` or `(B, D)` -- workspace slots at the current timestep. If 3D, flatten or aggregate to `(B, D)` before passing to the backend. `K` is the number of workspace slots; `D` is `workspace_dim`.
- `state`: backend-specific hidden state, or `None`. If `None`, call `reset_state()` internally.
- `timespans`: optional `(B,)` or `(B, 1)` time deltas for CfC/LTC. See Section 4.
- `ignition_gain`: optional `(B,)` or `(B, 1)` effective gain from the ignition gate. See Section 5.

**Returns:**

- `wm_out`: `(B, D)` integrated output
- `next_state`: backend-specific hidden state to carry to the next timestep

**Implementation sketch:**

```python
def forward(
    self,
    slots_t: torch.Tensor,
    state: Optional[torch.Tensor] = None,
    timespans: Optional[torch.Tensor] = None,
    ignition_gain: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B = slots_t.shape[0]
    device = slots_t.device

    # Aggregate slots if 3D
    if slots_t.dim() == 3:
        slots_t = slots_t.mean(dim=1)  # (B, D)

    # Auto-reset if no state provided
    if state is None:
        state = self.reset_state(B, device, torch.float32)

    # Compute raw backend output
    raw_out, new_state_raw = self._backend_forward(slots_t, state, timespans)

    # Apply ignition gating if provided
    if ignition_gain is not None:
        gain = ignition_gain.view(B, 1)  # (B, 1)
        wm_out = gain * raw_out + (1.0 - gain) * self._prev_output(B, device)
        new_state = self._interpolate_state(state, new_state_raw, gain)
    else:
        wm_out = raw_out
        new_state = new_state_raw

    # Update memory buffer (ignition-gated)
    self._update_buffer(wm_out, ignition_gain)

    # Cache previous output for next weak-write interpolation
    self._cached_prev_output = wm_out.detach()

    return wm_out, new_state
```

### 3.2 reset_state(batch_size, device, dtype=torch.float32) -> state

Create a fresh initial state for the given batch configuration. All three parameters are required to avoid ambiguity.

**Parameters:**

- `batch_size`: `int` -- number of items in the batch
- `device`: `torch.device` -- target device (cpu, cuda:0, etc.)
- `dtype`: `torch.dtype` -- always `torch.float32` for state tensors (see Section 9)

**Returns:**

- Backend-specific hidden state initialized to zeros

**Implementation:**

```python
def reset_state(
    self,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create fresh zero-initialized state.

    CfC/LTC: zeros of shape determined by the wiring.
    GRU: zeros of shape (num_layers, batch_size, hidden_dim).
    """
    if self.backend_type in ("cfc", "ltc"):
        # CfC/LTC state shape is determined by the wiring
        # The ncps library expects state as (B, state_size)
        state_size = self.backend.liquid.state_size
        return torch.zeros(
            batch_size, state_size, device=device, dtype=dtype
        )
    else:
        # GRU state shape: (num_layers, batch_size, hidden_dim)
        num_layers = self.backend.gru.num_layers
        hidden_dim = self.backend.hidden_dim
        return torch.zeros(
            num_layers, batch_size, hidden_dim, device=device, dtype=dtype
        )
```

**Current gap:** The existing `reset_state()` on all three classes (`WorkingMemory`, `GRUWorkingMemory`, `LiquidWorkingMemory`) takes no arguments and simply sets `self.hidden_state = None`. This is the implicit-state pattern. Replace it with the explicit three-parameter version above. The existing parameterless `reset_state()` continues to work as a convenience that also clears the memory buffer:

```python
def reset(self):
    """Full reset: state + memory buffer. Convenience method."""
    self.backend.hidden_state = None
    self.memory_buffer = None
    self._cached_prev_output = None
```

### 3.3 detach_state(state) -> state

Detach hidden state from the computation graph. Essential for truncated BPTT: without detaching, the graph grows unboundedly across BPTT chunks, consuming all available memory.

**Implementation:**

```python
def detach_state(self, state: torch.Tensor) -> torch.Tensor:
    """Detach state from computation graph for truncated BPTT.

    Returns a new tensor with identical numerical values but
    grad_fn=None. The original state is not modified.

    Works with any backend: CfC state is a single tensor,
    GRU state is a single tensor, LTC state is a single tensor.
    All are handled identically by .detach().
    """
    if state is None:
        return None
    if isinstance(state, torch.Tensor):
        return state.detach()
    if isinstance(state, (tuple, list)):
        return type(state)(s.detach() for s in state)
    raise TypeError(f"Cannot detach state of type {type(state)}")
```

**Why this is currently missing:** The existing code stores state on `self.hidden_state` inside each backend class. When state is stored on `self`, detaching is impossible from outside. The fix is to return state from `forward()` and accept it as input, which the current code already does. Adding `detach_state()` as a standalone method completes the contract.

**Truncated BPTT usage pattern:**

```python
state = wm.reset_state(B, device, torch.float32)

for chunk_idx, chunk in enumerate(sequence_chunks):
    if chunk_idx > 0:
        state = wm.detach_state(state)

    for t in range(chunk_length):
        wm_out, state = wm(chunk[t], state, timespans=dt[t])

    loss = compute_loss(wm_out)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### 3.4 get_state_norms(state) -> Dict[str, float]

Return L2 norms of state components. Used for drift detection telemetry: if state norms grow without bound over long sequences, the model has a stability problem.

**Implementation:**

```python
def get_state_norms(self, state: torch.Tensor) -> Dict[str, float]:
    """Compute L2 norms of state components for drift monitoring.

    Returns:
        Dict with keys:
            - 'state_l2': overall L2 norm
            - 'state_max': maximum absolute value
            - 'state_mean': mean absolute value
    """
    if state is None:
        return {'state_l2': 0.0, 'state_max': 0.0, 'state_mean': 0.0}

    if isinstance(state, torch.Tensor):
        flat = state.detach().float().flatten()
    elif isinstance(state, (tuple, list)):
        flat = torch.cat([s.detach().float().flatten() for s in state])
    else:
        return {'state_l2': 0.0, 'state_max': 0.0, 'state_mean': 0.0}

    return {
        'state_l2': flat.norm(2).item(),
        'state_max': flat.abs().max().item(),
        'state_mean': flat.abs().mean().item(),
    }
```

**Telemetry integration:** Log these norms every N steps (e.g., every 100 steps) during training. Alert if `state_l2` exceeds 100x its initial value or if `state_max` exceeds 1e6. Both indicate that state is diverging and the model will produce NaN outputs within a few hundred more steps.

```python
# In the training loop
if step % 100 == 0:
    norms = wm.get_state_norms(state)
    logger.log({
        'wm/state_l2': norms['state_l2'],
        'wm/state_max': norms['state_max'],
        'wm/state_mean': norms['state_mean'],
    })
    if norms['state_l2'] > initial_l2 * 100:
        logger.warning(
            f"Working memory state norm grew "
            f"{norms['state_l2'] / initial_l2:.1f}x "
            f"from initial. Possible divergence."
        )
```

---

## 4. Time / dt Handling

CfC and LTC are continuous-time models. They can process inputs arriving at irregular intervals, but only if time delta information is passed to them. The current code accepts a `timespans` parameter in `LiquidWorkingMemory.forward()` but does not wire it from upstream. This section specifies the complete dt pipeline.

### 4.1 When EncoderOutput.time Exists

Some encoder outputs carry a timestamp indicating when the input was observed. Compute dt as the difference between consecutive timestamps:

```python
def compute_dt(
    time_current: torch.Tensor,  # (B,) timestamp of current input
    time_previous: torch.Tensor, # (B,) timestamp of previous input
    min_dt: float = 1e-4,        # floor to prevent division-by-zero downstream
    max_dt: float = 10.0,        # ceiling to prevent extreme scaling
) -> torch.Tensor:
    """Compute time delta between consecutive inputs.

    Returns:
        dt: (B, 1) clamped time delta in fp32
    """
    dt = (time_current - time_previous).float()
    dt = dt.clamp(min=min_dt, max=max_dt)
    return dt.unsqueeze(-1)  # (B, 1)
```

Store `time_previous` on the module as a registered buffer that updates each step:

```python
self.register_buffer('_time_previous', None)

def _get_dt(self, time_current):
    if time_current is None:
        return None
    if self._time_previous is None:
        self._time_previous = time_current.detach()
        return torch.ones(time_current.shape[0], 1,
                          device=time_current.device, dtype=torch.float32)
    dt = compute_dt(time_current, self._time_previous)
    self._time_previous = time_current.detach()
    return dt
```

### 4.2 CfC/LTC Timespans

CfC and LTC accept the `timespans` parameter in their forward call. The expected shape depends on whether `return_sequences` is True or False:

- `return_sequences=False` (current setting): `timespans` shape is `(B, seq_len, 1)` or `(B, 1)` for single-step mode
- Since working memory processes one timestep at a time with `seq_len=1`, pass `dt` as `(B, 1, 1)` or `(B, 1)` depending on the ncps version

Wire timespans through the backend forward call:

```python
def _backend_forward(self, x, state, timespans=None):
    """Dispatch to backend with correct timespans handling."""
    # Ensure sequence dimension exists
    if x.dim() == 2:
        x = x.unsqueeze(1)  # (B, 1, D)

    if self.backend_type in ("cfc", "ltc"):
        # Wire timespans for continuous-time backends
        if timespans is not None:
            # Ensure shape is (B, 1, 1) for single-step
            if timespans.dim() == 1:
                timespans = timespans.unsqueeze(-1).unsqueeze(-1)
            elif timespans.dim() == 2:
                timespans = timespans.unsqueeze(-1)

        output, new_state = self.backend.liquid(
            x, hx=state, timespans=timespans
        )
    else:
        output, new_state = self.backend.gru(x, state)

    # Project output
    if hasattr(self.backend, 'hidden_proj'):
        output = self.backend.hidden_proj(output)
    if hasattr(self.backend, 'output_proj'):
        output = self.backend.output_proj(output)

    # Squeeze sequence dim if needed
    if output.dim() == 3:
        output = output[:, -1, :]  # take last step

    return output, new_state
```

**Current gap:** `LiquidWorkingMemory.forward()` passes `timespans` to CfC but not to LTC (lines 234-244 of working_memory.py). Both CfC and LTC support timespans. Wire it for both:

```python
# Fix: pass timespans to both CfC and LTC
if timespans is not None:
    output, self.hidden_state = self.liquid(
        x, hx=self.hidden_state, timespans=timespans
    )
else:
    output, self.hidden_state = self.liquid(
        x, hx=self.hidden_state
    )
```

### 4.3 GRU dt Handling

GRU does not natively support irregular time sampling. Two approaches exist. Choose **one** and document the choice in the config.

**Option A: Concatenate dt as extra input channel.**

Append dt (or a learned embedding of dt) to the input features. This increases the input dimension by 1 (or by the embedding size).

```python
class GRUWorkingMemoryWithDt(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dt_input_mode="concat"):
        super().__init__()
        self.dt_input_mode = dt_input_mode

        if dt_input_mode == "concat":
            self.dt_proj = nn.Linear(1, 32)
            effective_input_dim = input_dim + 32
        else:
            effective_input_dim = input_dim
            self.dt_gate_scale = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.Sigmoid(),
            )

        self.gru = nn.GRU(
            input_size=effective_input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, state=None, timespans=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        if timespans is not None and self.dt_input_mode == "concat":
            dt_feat = self.dt_proj(timespans.view(-1, 1))  # (B, 32)
            dt_feat = dt_feat.unsqueeze(1).expand(-1, x.shape[1], -1)
            x = torch.cat([x, dt_feat], dim=-1)

        output, new_state = self.gru(x, state)

        if timespans is not None and self.dt_input_mode == "gate_scale":
            gate = self.dt_gate_scale(timespans.view(-1, 1))  # (B, H)
            output = output[:, -1, :] * gate
        else:
            output = output[:, -1, :]

        output = self.output_proj(output)
        return output, new_state
```

**Option B: Scale GRU update gate by dt.**

Multiply the update gate output by `sigmoid(linear(dt))`, making the GRU update more aggressively for larger time gaps and more conservatively for smaller ones.

**Recommended:** Option A (concat) for simplicity and because it does not require modifying GRU internals. Set `dt_input_mode="concat"` as the default in `WorkingMemoryConfig`.

### 4.4 Default dt

If no time information is available (no `timespans` argument, no `EncoderOutput.time` field), assume `dt=1.0` (regular sampling). This is the current implicit behavior: CfC/LTC default to `dt=1.0` when timespans is not provided.

Do not fabricate dt values. If the data does not have timing information, omit the timespans argument entirely and let the backend use its default.

```python
# Correct: omit timespans when no time information exists
if encoder_output.time is not None:
    dt = self._get_dt(encoder_output.time)
    wm_out, state = self.working_memory(slots, state, timespans=dt)
else:
    wm_out, state = self.working_memory(slots, state)
```

---

## 5. Ignition-Gated Memory Write

The core architectural gap in the current working memory. Today, every forward pass writes to the memory buffer unconditionally with equal weight. This ignores the ignition signal from the global workspace competition. The fix: modulate write strength and state update magnitude by the ignition gate's effective gain.

### 5.1 Committed Write (ignition=True)

When global ignition fires (effective_gain close to 1.0):

- Full-weight memory update: `wm_out = backend(slots, state)`
- Write to memory buffer at full persistence
- State update is complete: `next_state = new_state_raw`
- This represents a "conscious" workspace broadcast that should be remembered

```python
# When ignition_gain is approximately 1.0
wm_out = raw_out                    # full backend output
next_state = new_state_raw          # full state transition
buffer_write_weight = 1.0           # persistent write to buffer
```

### 5.2 Weak Write (ignition=False)

When ignition does not fire (effective_gain close to 0.0):

- Partial weight: `wm_out = gain * raw_out + (1 - gain) * prev_output`
- State update is conservative: `next_state = gain * new_state_raw + (1 - gain) * old_state`
- Memory buffer write is suppressed or weighted by gain
- This represents a "subliminal" input that does not reach global broadcast

```python
# When ignition_gain is approximately 0.0
gain = ignition_gain.view(B, 1)  # (B, 1)

# Interpolate output
wm_out = gain * raw_out + (1.0 - gain) * prev_output

# Interpolate state (conservative update)
if isinstance(new_state_raw, torch.Tensor) and isinstance(old_state, torch.Tensor):
    gain_state = gain.view(
        *([gain.shape[0]] + [1] * (new_state_raw.dim() - 1))
    )
    next_state = gain_state * new_state_raw + (1.0 - gain_state) * old_state
else:
    next_state = new_state_raw  # fallback: full update

# Suppress buffer write
buffer_write_weight = gain.squeeze(-1)  # (B,)
```

### 5.3 Integration with IgnitionGate

The `IterativeCompetition` module in `brain_ai/workspace/global_workspace.py` produces `competition_info['ignition']` as a `(B, 1)` tensor between 0 and 1, and `competition_info['global_ignition']` as a binary indicator. Use the continuous `ignition` value as `effective_gain`:

```python
# In SelectionBroadcastWorkspace.forward()
competition_info = ...  # from IterativeCompetition

# Extract effective gain for working memory
effective_gain = competition_info['ignition']  # (B, 1), continuous [0, 1]

# Pass to working memory
memory_result = self.working_memory(
    integrated,
    state=wm_state,
    timespans=dt,
    ignition_gain=effective_gain,
)
```

Using the continuous ignition value rather than the binary `global_ignition` is intentional: it allows gradients to flow through the ignition gate, making the threshold learnable. The binary version is used only for logging and metrics.

### 5.4 State Interpolation Implementation

```python
def _interpolate_state(
    self,
    old_state: torch.Tensor,
    new_state: torch.Tensor,
    gain: torch.Tensor,
) -> torch.Tensor:
    """Interpolate between old and new state based on ignition gain.

    Args:
        old_state: state before backend forward pass
        new_state: state produced by backend forward pass
        gain: (B, 1) ignition gain, 0 = keep old, 1 = use new

    Returns:
        Interpolated state tensor
    """
    if old_state is None:
        return new_state

    # Broadcast gain to match state shape
    # GRU state: (num_layers, B, H) -- gain must be (1, B, 1)
    # CfC state: (B, state_size) -- gain must be (B, 1)
    if old_state.dim() == 3:
        # GRU: (num_layers, B, H)
        gain_broadcast = gain.unsqueeze(0)  # (1, B, 1)
    elif old_state.dim() == 2:
        # CfC/LTC: (B, state_size)
        gain_broadcast = gain  # already (B, 1)
    else:
        return new_state  # unknown shape, skip interpolation

    return gain_broadcast * new_state.float() + \
           (1.0 - gain_broadcast) * old_state.float()
```

---

## 6. Memory Buffer

### 6.1 FIFO Buffer

The memory buffer is a capacity-limited queue storing recent working memory outputs. It serves as an explicit short-term memory analogous to the items held in human working memory.

**Current implementation:** A Python-level tensor that grows via `torch.cat` and is trimmed by slicing when capacity is exceeded. This works but has two problems:

1. Capacity is hardcoded to `self.capacity = 7` (line 298 of working_memory.py)
2. `torch.cat` allocates a new tensor every step, creating garbage collection pressure

**Improved implementation: Tensor ring buffer.**

```python
class MemoryBuffer(nn.Module):
    """Fixed-capacity ring buffer for working memory outputs.

    Uses a pre-allocated tensor and a write pointer instead of
    torch.cat + slice, eliminating per-step allocation.
    """

    def __init__(self, capacity: int, dim: int):
        super().__init__()
        self.capacity = capacity
        self.dim = dim

        # Pre-allocated storage
        self.register_buffer(
            'buffer', torch.zeros(1, capacity, dim)  # (1, C, D) placeholder
        )
        self.register_buffer(
            'write_ptr', torch.zeros(1, dtype=torch.long)
        )
        self.register_buffer(
            'count', torch.zeros(1, dtype=torch.long)
        )
        self._initialized = False

    def _ensure_initialized(self, batch_size: int, device: torch.device):
        """Lazily initialize buffer to match batch size."""
        if not self._initialized or self.buffer.shape[0] != batch_size:
            self.buffer = torch.zeros(
                batch_size, self.capacity, self.dim,
                device=device, dtype=torch.float32
            )
            self.write_ptr = torch.zeros(
                1, dtype=torch.long, device=device
            )
            self.count = torch.zeros(
                1, dtype=torch.long, device=device
            )
            self._initialized = True

    def write(
        self,
        item: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ):
        """Write item to buffer at current write pointer.

        Args:
            item: (B, D) tensor to store
            weight: optional (B,) write weight from ignition gate.
                    If provided, item is scaled by weight before storage.
                    Items with weight < 0.1 are skipped entirely.
        """
        B = item.shape[0]
        self._ensure_initialized(B, item.device)

        if weight is not None:
            # Skip write for items with very low ignition
            write_mask = (weight > 0.1).float()  # (B,)
            item = item * weight.unsqueeze(-1) * write_mask.unsqueeze(-1)

            # Only advance pointer if at least one item in batch has weight
            if write_mask.sum() == 0:
                return

        ptr = self.write_ptr.item() % self.capacity
        self.buffer[:, ptr, :] = item.float().detach()
        self.write_ptr += 1
        self.count = torch.clamp(self.count + 1, max=self.capacity)

    def read_all(self) -> Optional[torch.Tensor]:
        """Read all valid buffer entries.

        Returns:
            (B, N, D) tensor where N = min(count, capacity),
            or None if empty
        """
        if not self._initialized or self.count.item() == 0:
            return None

        n = min(self.count.item(), self.capacity)
        # Read entries in order from oldest to newest
        if self.write_ptr.item() >= self.capacity:
            # Buffer is full, read from write_ptr onward (wrapping)
            start = self.write_ptr.item() % self.capacity
            indices = [(start + i) % self.capacity for i in range(n)]
        else:
            indices = list(range(n))

        return self.buffer[:, indices, :]

    def reset(self):
        """Clear buffer."""
        if self._initialized:
            self.buffer.zero_()
            self.write_ptr.zero_()
            self.count.zero_()
```

### 6.2 Content-Based Retrieval

Retrieve from the memory buffer using dot-product attention between a query and the buffer contents. This is already implemented in `WorkingMemory.retrieve()` (line 364 of working_memory.py). The existing implementation is correct; preserve it but update to use the ring buffer:

```python
def retrieve(self, query: torch.Tensor) -> torch.Tensor:
    """Retrieve from memory buffer via content-based attention.

    Args:
        query: (B, D) query vector

    Returns:
        (B, D) attention-weighted retrieval from buffer
    """
    buffer_contents = self.memory_buffer.read_all()

    if buffer_contents is None:
        return query  # passthrough if buffer is empty

    # Dot-product attention
    # query: (B, D), buffer: (B, N, D)
    scores = torch.bmm(
        buffer_contents,
        query.unsqueeze(-1),
    ).squeeze(-1)  # (B, N)

    attention = torch.softmax(
        scores / (self.config.output_dim ** 0.5), dim=-1
    )

    retrieved = torch.bmm(
        attention.unsqueeze(1),
        buffer_contents,
    ).squeeze(1)  # (B, D)

    return retrieved
```

Note the addition of `sqrt(d)` scaling on the attention scores, which stabilizes the softmax at higher dimensions. The current code omits this scaling.

### 6.3 Buffer Write Policy

Tie buffer writes to the ignition signal:

| Ignition state | Buffer action | Rationale |
|---|---|---|
| `gain >= 0.5` | Write at full weight | Content reached global broadcast; persist it |
| `0.1 <= gain < 0.5` | Write at reduced weight (scaled by gain) | Partial activation; may be useful for retrieval |
| `gain < 0.1` | Skip write entirely | Subliminal; not worth occupying a buffer slot |

This policy is implemented in `MemoryBuffer.write()` above. The threshold values (0.1, 0.5) are implementation defaults; make them configurable if ablation studies require it.

```python
def _update_buffer(self, wm_out, ignition_gain=None):
    """Update memory buffer with ignition-gated write policy."""
    if ignition_gain is not None:
        self.memory_buffer.write(
            wm_out, weight=ignition_gain.squeeze(-1)
        )
    else:
        # No ignition info: unconditional write (backward compat)
        self.memory_buffer.write(wm_out)
```

---

## 7. State Persistence Tests

These tests validate that working memory state behaves correctly across timesteps, resets, and detach operations. Run them for each backend (CfC, LTC, GRU).

### 7.1 Sequential State Changes

Feed a sequence of timesteps and verify that state changes predictably at each step. The same input with carried state should produce different outputs than the same input with fresh state.

```python
def test_sequential_state_changes(backend_mode):
    """State must evolve: same input at step 1 and step 5
    produce different outputs."""
    config = WorkingMemoryConfig(
        input_dim=128, hidden_dim=128, output_dim=128,
        mode=backend_mode, num_units=32, sparsity=0.5,
        buffer_capacity=7,
    )
    wm = WorkingMemory(config)
    wm.eval()

    B, D = 4, 128
    x = torch.randn(B, D)
    state = wm.reset_state(B, torch.device('cpu'), torch.float32)

    outputs = []
    for t in range(5):
        out, state = wm(x, state)
        outputs.append(out.clone())

    # Output at step 0 and step 4 must differ
    # (state accumulated information)
    assert not torch.allclose(outputs[0], outputs[4], atol=1e-5), \
        "Same input at different timesteps must produce different " \
        "outputs when state carries"
```

### 7.2 Reset Baseline

After `reset_state()`, outputs should return to the baseline distribution. Verify that output after reset matches output from a fresh model initialization within numerical tolerance.

```python
def test_reset_baseline(backend_mode):
    """After reset, outputs must match fresh initialization."""
    config = WorkingMemoryConfig(
        input_dim=128, hidden_dim=128, output_dim=128,
        mode=backend_mode, num_units=32, sparsity=0.5,
        buffer_capacity=7,
    )
    wm = WorkingMemory(config)
    wm.eval()

    B, D = 4, 128
    x = torch.randn(B, D)

    # Run a few steps to accumulate state
    state = wm.reset_state(B, torch.device('cpu'), torch.float32)
    for _ in range(10):
        _, state = wm(x, state)

    # Reset and run one step
    state_reset = wm.reset_state(
        B, torch.device('cpu'), torch.float32
    )
    out_after_reset, _ = wm(x, state_reset)

    # Fresh model, fresh state, one step
    state_fresh = wm.reset_state(
        B, torch.device('cpu'), torch.float32
    )
    out_fresh, _ = wm(x, state_fresh)

    assert torch.allclose(out_after_reset, out_fresh, atol=1e-6), \
        "Output after reset must match output from fresh state"
```

### 7.3 Detach Does Not Affect Values

`detach_state(state)` must return tensors with identical numerical values but `requires_grad=False`.

```python
def test_detach_preserves_values(backend_mode):
    """Detach must preserve numerical values exactly."""
    config = WorkingMemoryConfig(
        input_dim=128, hidden_dim=128, output_dim=128,
        mode=backend_mode, num_units=32, sparsity=0.5,
        buffer_capacity=7,
    )
    wm = WorkingMemory(config)

    B, D = 4, 128
    x = torch.randn(B, D, requires_grad=True)
    state = wm.reset_state(
        B, torch.device('cpu'), torch.float32
    )

    _, state_after = wm(x, state)
    state_detached = wm.detach_state(state_after)

    if isinstance(state_after, torch.Tensor):
        assert torch.equal(
            state_after.data, state_detached.data
        ), "Detach must preserve state values"
        assert state_detached.grad_fn is None, \
            "Detached state must have no grad_fn"
    elif isinstance(state_after, (tuple, list)):
        for s_orig, s_det in zip(state_after, state_detached):
            assert torch.equal(s_orig.data, s_det.data)
            assert s_det.grad_fn is None
```

### 7.4 Detach Cuts Gradients

Verify that gradients do not flow through the detach boundary.

```python
def test_detach_cuts_gradients(backend_mode):
    """Gradients must not flow back through the detach boundary."""
    config = WorkingMemoryConfig(
        input_dim=128, hidden_dim=128, output_dim=128,
        mode=backend_mode, num_units=32, sparsity=0.5,
        buffer_capacity=7,
    )
    wm = WorkingMemory(config)

    B, D = 4, 128
    x1 = torch.randn(B, D, requires_grad=True)
    x2 = torch.randn(B, D, requires_grad=True)
    state = wm.reset_state(
        B, torch.device('cpu'), torch.float32
    )

    # Chunk 1
    _, state = wm(x1, state)

    # Detach at chunk boundary
    state = wm.detach_state(state)

    # Chunk 2
    out, _ = wm(x2, state)
    loss = out.sum()
    loss.backward()

    assert x1.grad is None, \
        "Gradient must not flow back through detach"
    assert x2.grad is not None, \
        "Gradient must flow for current chunk"
```

### 7.5 Backend Equivalence Contract

Verify that CfC and GRU backends produce the same output shape and dictionary structure.

```python
def test_backend_output_contract():
    """All backends must produce identical output structure."""
    for mode in ["cfc", "gru"]:
        config = WorkingMemoryConfig(
            input_dim=128, hidden_dim=128, output_dim=128,
            mode=mode, num_units=32, sparsity=0.5,
            buffer_capacity=7,
        )
        # Skip CfC if ncps unavailable
        if mode == "cfc" and not NCPS_AVAILABLE:
            continue

        wm = WorkingMemory(config)
        B, D = 4, 128
        x = torch.randn(B, D)
        state = wm.reset_state(
            B, torch.device('cpu'), torch.float32
        )

        out, new_state = wm(x, state)

        assert out.shape == (B, 128), \
            f"Output shape must be (B, D), got {out.shape}"
        assert new_state is not None, "State must not be None"
```

---

## 8. WMConfig Surface

Complete configuration surface for working memory. All fields with their types, defaults, and descriptions.

| Field | Type | Default | Description |
|---|---|---|---|
| `input_dim` | `int` | `4096` | Input dimension, must match `workspace_dim` |
| `hidden_dim` | `int` | `4096` | Internal hidden dimension of the backend |
| `output_dim` | `int` | `4096` | Output dimension, must match `workspace_dim` |
| `mode` | `str` | `"cfc"` | Backend: `"cfc"`, `"ltc"`, `"gru"`, or `"auto"` |
| `num_units` | `int` | `128` | Number of units in AutoNCP wiring |
| `sparsity` | `float` | `0.5` | Connection sparsity for AutoNCP wiring |
| `buffer_capacity` | `int` | `7` | Maximum items in memory buffer (Miller's Law) |
| `use_dt` | `bool` | `True` | Enable dt/timespans handling |
| `dt_input_mode` | `str` | `"concat"` | How GRU handles dt: `"concat"` or `"gate_scale"` |

**Current gap:** The existing `WorkingMemoryConfig` dataclass (line 26 of working_memory.py) has only six fields: `input_dim`, `hidden_dim`, `output_dim`, `mode`, `num_units`, `sparsity`. Add the three missing fields:

```python
@dataclass
class WorkingMemoryConfig:
    """Configuration for working memory."""
    input_dim: int = 4096        # Match workspace_dim
    hidden_dim: int = 4096       # Internal backend dimension
    output_dim: int = 4096       # Output dimension
    mode: str = "cfc"            # cfc, ltc, gru, auto
    num_units: int = 128         # AutoNCP wiring units
    sparsity: float = 0.5        # AutoNCP sparsity
    buffer_capacity: int = 7     # Memory buffer capacity
    use_dt: bool = True          # Enable dt handling
    dt_input_mode: str = "concat"  # GRU dt mode: concat or gate_scale
```

Note the dimension changes from the current defaults (512) to production defaults (4096). The current 512-dim defaults are from the minimal config. Production config in `brain_ai/config.py` already specifies `memory_hidden_dim=4096` (line 176). Align `WorkingMemoryConfig` defaults with the production config.

**Relationship to WorkspaceConfig:** The `WorkspaceConfig` in `brain_ai/config.py` has three working memory fields:

- `memory_hidden_dim: int = 4096` maps to `WorkingMemoryConfig.hidden_dim`
- `memory_mode: str = "cfc"` maps to `WorkingMemoryConfig.mode`
- `memory_num_layers: int = 8` is not currently used by `WorkingMemoryConfig`

Add `num_layers` to `WorkingMemoryConfig` to close this gap. For GRU, `num_layers` maps directly to `nn.GRU(num_layers=...)`. For CfC/LTC, `num_layers` means stacking multiple CfC/LTC cells (not currently supported; if stacking is implemented, do so by composing cells in a `nn.Sequential`).

### 8.1 Mode Auto-Selection

When `mode="auto"`:
1. If `NCPS_AVAILABLE`, use `"cfc"`
2. Otherwise, use `"gru"`

This is the existing behavior in `create_working_memory()` (line 413 of working_memory.py). Preserve it.

### 8.2 AutoNCP Constraints

The AutoNCP wiring has a constraint: `output_size < num_units - 2`. The current code handles this on line 155 with:

```python
ncp_output_size = min(hidden_dim, num_units - 4)
```

This constraint means `num_units` must be at least `output_size + 5` in practice. With `num_units=128` and the output projected separately, this is not a problem. Document this constraint so users do not set `num_units` too small.

---

## 9. Mixed-Precision Safety

Working memory state is a temporal accumulator. Small drift in state representation compounds over hundreds or thousands of steps. Mixed-precision must be handled with care.

### 9.1 State Updates: fp32

All state tensors (`hidden_state` for CfC/LTC/GRU, `_cached_prev_output`, `_time_previous`) must be stored and updated in `fp32`. Never cast state to `fp16` or `bf16`.

```python
def _backend_forward(self, x, state, timespans=None):
    # Cast input to fp32 for state accumulation
    x_fp32 = x.float()

    if timespans is not None:
        timespans = timespans.float()

    output, new_state = self.backend(x_fp32, state, timespans)

    # State is already fp32 from the backend
    # Cast output back to input dtype for downstream
    return output.to(x.dtype), new_state
```

### 9.2 Memory Buffer: fp32 Storage

The memory buffer stores integrated workspace outputs. These are used for content-based retrieval via attention. Store buffer contents in fp32 to prevent precision loss in the attention score computation:

```python
# In MemoryBuffer.write()
self.buffer[:, ptr, :] = item.float().detach()
```

### 9.3 Backend Forward Pass

The backend forward pass receives fp32 inputs and produces fp32 outputs. AMP's `autocast` context manager does not interfere because CfC/LTC/GRU internal operations are already fp32 when the input is fp32. Do not wrap the backend forward in `torch.cuda.amp.autocast()`.

```python
# Correct: no autocast inside working memory
def forward(self, slots_t, state, ...):
    x_fp32 = slots_t.float()
    output, state = self._backend_forward(x_fp32, state, timespans)
    return output, state  # fp32

# Wrong: autocast inside working memory causes state drift
# def forward(self, slots_t, state, ...):
#     with torch.cuda.amp.autocast():  # DO NOT DO THIS
#         output, state = self._backend_forward(slots_t, state)
#     return output, state
```

### 9.4 dt Computation: fp32

Time deltas must be computed in fp32. Small time differences (e.g., 1ms between events at 1000Hz) would underflow in fp16.

```python
dt = (time_current.float() - time_previous.float()).clamp(
    min=1e-4, max=10.0
)
```

### 9.5 Downstream Dtype Restoration

Working memory output is fp32. Downstream modules (broadcast, integration) may operate in mixed precision. Cast the output to match the downstream dtype at the boundary:

```python
# In SelectionBroadcastWorkspace.forward()
wm_out, wm_state = self.working_memory(integrated, ...)
wm_out = wm_out.to(integrated.dtype)  # restore dtype for downstream
```

---

## 10. Migration from Existing Code

### 10.1 From Current WorkingMemory

The `WorkingMemory` class in `brain_ai/workspace/working_memory.py` is the migration target. Apply these changes:

**Change 1: Add `reset_state(batch_size, device, dtype)`**

Current state: `reset_state()` takes no arguments, sets `self.backend.hidden_state = None` and `self.memory_buffer = None`.

Target state: `reset_state(batch_size, device, dtype)` returns a fresh state tensor. A separate `reset()` method clears buffer and cached state.

```python
# Before (line 300)
def reset_state(self):
    self.backend.reset_state()
    self.memory_buffer = None

# After
def reset_state(self, batch_size, device, dtype=torch.float32):
    """Create fresh state tensor for given batch configuration."""
    return self._create_fresh_state(batch_size, device, dtype)

def reset(self):
    """Full reset: clear state cache, memory buffer,
    and previous output."""
    self.backend.hidden_state = None
    self.memory_buffer.reset()
    self._cached_prev_output = None
```

**Change 2: Add `detach_state()`**

Currently missing entirely. Add as described in Section 3.3.

**Change 3: Wire timespans/dt**

Current state: `timespans` is accepted by `WorkingMemory.forward()` but only passed through when `self.backend_type == "cfc"` (line 347). LTC does not receive timespans even though it supports them.

Target state: Pass timespans to both CfC and LTC backends. Implement dt handling for GRU.

```python
# Before (line 347)
if self.backend_type == "cfc" and timespans is not None:
    output, new_state = self.backend(x, state, timespans)
else:
    output, new_state = self.backend(x, state)

# After
if self.backend_type in ("cfc", "ltc") and timespans is not None:
    output, new_state = self.backend(x, state, timespans)
elif (self.backend_type == "gru"
      and timespans is not None
      and self.config.use_dt):
    output, new_state = self.backend(x, state, timespans)
else:
    output, new_state = self.backend(x, state)
```

**Change 4: Make `buffer_capacity` configurable**

Current state: `self.capacity = 7` hardcoded on line 298.

Target state: `self.memory_buffer = MemoryBuffer(capacity=config.buffer_capacity, dim=config.output_dim)`.

**Change 5: Add ignition-gated write**

Current state: `update_buffer(output)` is called unconditionally on line 354.

Target state: Pass `ignition_gain` to `_update_buffer()`, which delegates to `MemoryBuffer.write()` with the weight parameter.

**Change 6: Add state norms telemetry**

Currently missing. Add `get_state_norms()` as described in Section 3.4.

### 10.2 From Current GRUWorkingMemory

The `GRUWorkingMemory` class (line 36 of working_memory.py) needs:

**Add dt handling:**

```python
# Before
def forward(self, x, state=None):
    ...

# After
def forward(self, x, state=None, timespans=None):
    ...
    if timespans is not None and self.dt_input_mode == "concat":
        dt_feat = self.dt_proj(timespans.view(-1, 1))
        ...
```

**Add explicit `reset_state(batch_size, device, dtype)`:**

```python
# Before (line 69)
def reset_state(self):
    self.hidden_state = None

# After
def reset_state(
    self,
    batch_size=None,
    device=None,
    dtype=torch.float32,
):
    if batch_size is None:
        self.hidden_state = None
        return None
    num_layers = self.gru.num_layers
    return torch.zeros(
        num_layers, batch_size, self.hidden_dim,
        device=device, dtype=dtype,
    )
```

**Add state norms:**

```python
def get_state_norms(self):
    if self.hidden_state is None:
        return {
            'state_l2': 0.0,
            'state_max': 0.0,
            'state_mean': 0.0,
        }
    flat = self.hidden_state.detach().float().flatten()
    return {
        'state_l2': flat.norm(2).item(),
        'state_max': flat.abs().max().item(),
        'state_mean': flat.abs().mean().item(),
    }
```

### 10.3 From Current LiquidWorkingMemory

The `LiquidWorkingMemory` class (line 112 of working_memory.py) needs:

**Wire timespans for LTC:**

```python
# Before (line 234)
if self.mode == "cfc":
    output, self.hidden_state = self.liquid(
        x, hx=self.hidden_state, timespans=timespans,
    )
else:
    output, self.hidden_state = self.liquid(
        x, hx=self.hidden_state,
    )

# After: pass timespans to both CfC and LTC
kwargs = {'hx': self.hidden_state}
if timespans is not None:
    kwargs['timespans'] = timespans
output, self.hidden_state = self.liquid(x, **kwargs)
```

**Add `detach_state` support:**

The hidden state for CfC/LTC is a tensor stored on `self.hidden_state`. Add a method to detach it:

```python
def detach_state(self, state=None):
    if state is not None:
        if isinstance(state, torch.Tensor):
            return state.detach()
        return state
    if self.hidden_state is not None:
        self.hidden_state = self.hidden_state.detach()
    return self.hidden_state
```

**Add explicit `reset_state(batch_size, device, dtype)`:**

```python
def reset_state(
    self,
    batch_size=None,
    device=None,
    dtype=torch.float32,
):
    if batch_size is None:
        self.hidden_state = None
        return None
    state_size = self.liquid.state_size
    return torch.zeros(
        batch_size, state_size, device=device, dtype=dtype
    )
```

---

## 11. Anti-Patterns

These patterns cause bugs that are difficult to diagnose. Avoid all of them.

**Forgetting `detach_state` in truncated BPTT.**
The computation graph grows without bound across BPTT chunks. Memory consumption increases linearly with sequence length. Training either OOMs or produces numerically meaningless gradients that have accumulated over thousands of unrelated timesteps. Always detach at chunk boundaries:

```python
# Wrong: no detach between chunks
for chunk in chunks:
    for t in range(chunk_len):
        out, state = wm(chunk[t], state)
    loss.backward()

# Correct: detach at chunk boundary
for chunk in chunks:
    state = wm.detach_state(state)
    for t in range(chunk_len):
        out, state = wm(chunk[t], state)
    loss.backward()
```

**Using fp16 for state storage.**
Membrane-like accumulators (and GRU/CfC hidden states are exactly this) compound small deltas over many steps. In fp16, the minimum representable increment above 1.0 is approximately 0.001. After 100 steps, the accumulated error can represent 10% of the true state value. After 1000 steps, the state is effectively random noise. Always use fp32 for state:

```python
# Wrong
state = wm.reset_state(B, device, torch.float16)

# Correct
state = wm.reset_state(B, device, torch.float32)
```

**Hardcoding buffer capacity.**
The capacity should come from config, not from a magic number in `__init__()`. Hardcoding prevents experiments that test the effect of buffer size on task performance, and it creates a maintenance burden when different deployment scenarios require different capacities:

```python
# Wrong (current code, line 298)
self.capacity = 7

# Correct
self.memory_buffer = MemoryBuffer(
    capacity=config.buffer_capacity,
    dim=config.output_dim,
)
```

**Branching on backend type outside the working memory module.**
The whole point of the seamless fallback design is that downstream code does not know or care which backend is running. If any module outside `working_memory.py` contains `if wm.backend_type == "cfc"`, the abstraction is broken:

```python
# Wrong: branching in global workspace
if self.working_memory.backend_type == "cfc":
    result = self.working_memory(x, state, timespans=dt)
else:
    result = self.working_memory(x, state)

# Correct: pass timespans unconditionally;
# backend ignores if unsupported
result = self.working_memory(x, state, timespans=dt)
```

**Not passing dt to CfC/LTC.**
CfC and LTC are continuous-time models. Their primary advantage over GRU is the ability to handle irregular time sampling. If dt is never passed, CfC degrades to a standard RNN with fixed timestep, wasting the expressiveness of the continuous-time formulation. Always compute and pass dt when time information is available:

```python
# Wrong: ignoring available time information
wm_out, state = wm(slots, state)  # dt available but not passed

# Correct: pass dt when available
dt = compute_dt(encoder_output.time, prev_time)
wm_out, state = wm(slots, state, timespans=dt)
```

**Unconditional memory write regardless of ignition.**
Writing to the memory buffer on every step regardless of ignition fills the buffer with subliminal, low-confidence content that dilutes the signal from genuine ignition events. The buffer then becomes a noisy FIFO of recent outputs rather than a curated set of "consciously attended" workspace states:

```python
# Wrong: unconditional write (current code, line 354)
if update_buffer:
    buffer = self.update_buffer(output)

# Correct: ignition-gated write
self._update_buffer(output, ignition_gain=ignition_gain)
```

**Not resetting state on batch size change.**
The existing code (lines 97-101 of working_memory.py) detects batch size mismatch and resets by setting `self.hidden_state = None`. This is correct in intent but loses state silently. A better approach is to raise an error if batch size changes without an explicit reset, because batch size changes mid-sequence indicate a bug in the data pipeline:

```python
# Current behavior: silent reset on batch size change
if self.hidden_state is not None:
    if self.hidden_state.shape[1] != batch_size:
        self.hidden_state = None  # silent data loss

# Better: explicit check
if state is not None:
    expected_batch = (
        state.shape[0] if state.dim() == 2 else state.shape[1]
    )
    if expected_batch != batch_size:
        raise ValueError(
            f"State batch size {expected_batch} does not match "
            f"input batch size {batch_size}. Call reset_state() "
            f"explicitly when batch size changes."
        )
```

**Using Python list for memory buffer.**
A Python list of tensors (`self.buffer = [t1, t2, t3, ...]`) causes repeated CPU-GPU synchronization and prevents batched operations. Use a pre-allocated tensor ring buffer as described in Section 6.1:

```python
# Wrong: Python list buffer
self.buffer = []
self.buffer.append(item)
if len(self.buffer) > capacity:
    self.buffer.pop(0)

# Correct: tensor ring buffer
self.memory_buffer = MemoryBuffer(capacity=capacity, dim=dim)
self.memory_buffer.write(item, weight=ignition_gain)
```

**Storing state on `self` inside backend classes.**
Both `GRUWorkingMemory` and `LiquidWorkingMemory` store `self.hidden_state` as an instance attribute. This creates implicit statefulness that makes the module non-reentrant and prevents parallel forward passes with different states. The fix is to return state from `forward()` and accept it as input, which the current code already does. Remove the `self.hidden_state` storage and use the returned state exclusively:

```python
# Wrong: storing state on self
def forward(self, x, state=None):
    if state is not None:
        self.hidden_state = state    # mutates self
    output, self.hidden_state = self.gru(
        x, self.hidden_state
    )  # reads/writes self
    return output, self.hidden_state

# Correct: pure function on state
def forward(self, x, state=None):
    if state is None:
        state = self.reset_state(
            x.shape[0], x.device, torch.float32
        )
    output, new_state = self.gru(x, state)
    return output, new_state  # no self mutation
```

**Not normalizing attention scores by sqrt(d) in retrieval.**
The current `retrieve()` method (line 379 of working_memory.py) computes raw dot-product attention without scaling. At dimension 4096, the dot products have variance proportional to 4096, causing the softmax to saturate and produce near-one-hot attention weights. Scale by `sqrt(d)`:

```python
# Wrong (current)
scores = torch.bmm(
    self.memory_buffer, query.unsqueeze(-1)
).squeeze(-1)
attention = torch.softmax(scores, dim=-1)

# Correct
scores = torch.bmm(
    self.memory_buffer, query.unsqueeze(-1)
).squeeze(-1)
scores = scores / (self.config.output_dim ** 0.5)
attention = torch.softmax(scores, dim=-1)
```
