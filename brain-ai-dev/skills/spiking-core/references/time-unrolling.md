# Time Unrolling: One Utility That Everything Uses

## The Problem

The existing codebase contains two separate, incompatible time unroll loops:

- `SNNCore.forward()` in `snn.py` lines 112-150 iterates over time with layout `(time, batch, feat)`.
- `ConvSNN.forward()` in `snn.py` lines 291-309 iterates over time with layout `(batch, C, H, W)` per step.

Neither implementation supports truncated backpropagation through time (BPTT) or configurable chunking. This divergence is the primary source of shape bugs, gradient explosion, and silent behavioral differences between the convolutional and MLP SNN paths.

The solution is a single `snn_unroll` utility that all SNN modules delegate to. This document specifies the semantics, interface, implementation logic, and integration path for that utility.

---

## A) Time Semantics: Two Modes

All SNN modules must support exactly two calling conventions. The mode is inferred from input rank.

### Step Mode

Input `x` has shape `(B, ...)` — no time dimension. The module processes a single timestep and returns the output for that step plus the updated state.

```
x:     (B, ...)
out:   (B, ...)
next_state: State
```

Step mode is used during autoregressive inference, real-time processing, and when an external loop controls time.

### Sequence Mode

Input `x` has shape `(B, T, ...)` — batch first, time second. The module processes the entire sequence and returns outputs for all timesteps plus the final state.

```
x:     (B, T, ...)
out:   (B, T, ...)
final_state: State
```

Sequence mode is used during training and batch inference.

### Canonical Layout Convention

All internal code uses **batch-first, time-second**: `(B, T, ...)`.

This is a hard convention. Any module that internally uses `(T, B, ...)` must transpose at its boundaries, not propagate the non-standard layout outward.

**ConvSNN layouts:**
- Sequence mode input: `(B, T, C, H, W)`
- Sequence mode output: `(B, T, C', H', W')`
- Step mode input: `(B, C, H, W)`
- Step mode output: `(B, C', H', W')`

**MLP SNN layouts:**
- Sequence mode input: `(B, T, D)`
- Sequence mode output: `(B, T, D')`
- Step mode input: `(B, D)`
- Step mode output: `(B, D')`

---

## B) The Unroll Utility

### Interface

```python
from typing import Callable, Dict, Optional, Tuple
from torch import Tensor

def snn_unroll(
    cell_fn: Callable[[Tensor, State], Tuple[Tensor, State]],
    inputs: Tensor,                    # (B, T, ...)
    initial_state: State,
    chunk_size: Optional[int] = None,  # truncated BPTT window; None = full BPTT
    return_traces: bool = False,       # return membrane and current traces
) -> Tuple[Tensor, State, Optional[Dict[str, Tensor]]]:
    """
    Unroll a spiking cell over a time sequence using chunked BPTT.

    Args:
        cell_fn:       Callable that maps (x_t, state_t) -> (spike_t, state_{t+1}).
                       x_t has shape (B, ...) — no time dimension.
        inputs:        Sequence input of shape (B, T, ...).
        initial_state: SpikingState (or compatible dataclass) holding initial v, i, a, ref.
        chunk_size:    Number of timesteps per BPTT chunk. If None, uses full T (no truncation).
        return_traces: If True, collect membrane potential and synaptic current over time.

    Returns:
        spikes:      Tensor of shape (B, T, ...) — spike output at each timestep.
        final_state: SpikingState after the last timestep.
        traces:      Dict with keys 'membrane', 'current' if return_traces=True, else None.
    """
```

### Implementation Logic

Walk through the unroll in this order:

**Step 1 — Extract dimensions.**

```python
B = inputs.shape[0]
T = inputs.shape[1]
spatial = inputs.shape[2:]   # everything after (B, T)
```

**Step 2 — Resolve chunk size.**

```python
if chunk_size is None:
    chunk_size = T
```

A `chunk_size` of `T` is identical to full BPTT. No special-casing is needed.

**Step 3 — Pre-allocate output buffer.**

Do not accumulate spikes in a Python list. Pre-allocate:

```python
spike_record = torch.zeros(B, T, *spatial,
                           device=inputs.device, dtype=inputs.dtype)
```

Optionally pre-allocate trace buffers if `return_traces=True`:

```python
if return_traces:
    membrane_record = torch.zeros(B, T, *initial_state.v.shape[1:],
                                  device=inputs.device, dtype=inputs.dtype)
    current_record  = torch.zeros_like(membrane_record)
```

**Step 4 — Outer loop over chunks.**

```python
state = initial_state
for chunk_start in range(0, T, chunk_size):
    chunk_end = min(chunk_start + chunk_size, T)
    chunk_inputs = inputs[:, chunk_start:chunk_end]  # (B, chunk_len, ...)
```

**Step 5 — Inner loop over timesteps within chunk.**

```python
    for t_local, t_global in enumerate(range(chunk_start, chunk_end)):
        x_t = chunk_inputs[:, t_local]              # (B, ...)
        spike_t, state = cell_fn(x_t, state)
        spike_record[:, t_global] = spike_t
        if return_traces:
            membrane_record[:, t_global] = state.v
            if state.i is not None:
                current_record[:, t_global] = state.i
```

**Step 6 — Detach state between chunks.**

```python
    state = state.detach()
```

This call cuts the computation graph between chunks. Gradients will not flow across chunk boundaries. The numerical values of state are preserved; only the gradient history is severed.

**Step 7 — Assemble return value.**

```python
traces = None
if return_traces:
    traces = {'membrane': membrane_record, 'current': current_record}

return spike_record, state, traces
```

---

## C) Truncated BPTT Chunking

### Why Truncated BPTT Is Necessary

Full BPTT through spikes has memory cost O(T * N) where N is the number of neurons. For T=50 with a large SNN, the intermediate activations stored for gradient computation consume significant GPU memory and slow backward passes.

Truncated BPTT trades gradient quality for memory and speed by limiting the horizon over which gradients are computed.

### How Chunking Works

1. Unroll for `chunk_size` timesteps while retaining the computation graph.
2. After the chunk completes, call `state.detach()` to cut the graph.
3. Continue unrolling from the detached state.
4. Repeat until `T` timesteps are exhausted.

Gradients flow only within each chunk. The state carries forward its numerical values across chunk boundaries, so temporal dynamics (membrane potential, adaptation) are continuous even though gradient flow is not.

### Memory and Gradient Trade-offs

| chunk\_size | Memory usage | Gradient quality | Training speed |
|---|---|---|---|
| Full T (no truncation) | Highest (O(T)) | Best — full temporal credit assignment | Slowest backward |
| T/2 | Moderate | Good for most sequences | Moderate |
| T/4 | Low | Adequate for local patterns | Fast |
| T/10 (recommended) | Minimal | Sufficient for most SNN tasks | Fastest |

### When to Use Each Mode

**Use full BPTT** (`chunk_size=None`) when:
- T is 25 or fewer timesteps.
- The model and activations fit comfortably in GPU memory.
- Training requires precise long-range temporal credit assignment.

**Use truncated BPTT** when:
- T exceeds 25 timesteps.
- Memory is constrained (batch size must be large, or model is large).
- Training is memory-bound and chunk_size=10 provides a practical default.

**Recommended default:** `chunk_size=10` for T=50. This limits gradient backprop to 10-step windows, reduces memory by approximately 5x compared to full BPTT, and retains sufficient temporal context for most SNN tasks.

### State Detach Implementation

The detach must cover all state components. Use `pytree_map` or an explicit method on the state container:

```python
state = tree_map(lambda t: t.detach(), state)
```

Or, using the `SpikingState.detach()` method defined in section E:

```python
state = state.detach()
```

Do not detach only `v`. If `i`, `a`, `ref`, or `spike_history` are present and not detached, their gradient histories will accumulate across the full sequence, defeating the purpose of chunking and causing memory growth proportional to T.

---

## D) ConvSNN Time Batching

Two approaches exist for applying convolutional layers across the time dimension.

### Approach 1: Loop Over Time (Recommended)

Pass `(B, C, H, W)` through the conv layer at each timestep inside the unroll loop.

```python
# cell_fn receives (B, C, H, W), applies conv, returns (B, C', H', W') and next state
spike_t, state = cell_fn(x_t, state)  # x_t is (B, C, H, W)
```

Characteristics:
- Simple and correct.
- No special handling for batch normalization or other stateful layers.
- Conv operations are already parallelized over the spatial dimensions (H, W) and channels (C) by CUDA. Looping over T does not reduce GPU utilization for typical spatial sizes.

### Approach 2: Flatten B and T (Avoid Unless Benchmarked)

Reshape input from `(B, T, C, H, W)` to `(B*T, C, H, W)`, apply conv, then reshape back to `(B, T, C', H', W')`.

```python
BT = B * T
x_flat = inputs.view(BT, C, H, W)
out_flat = conv(x_flat)               # (B*T, C', H', W')
out = out_flat.view(B, T, C_prime, H_prime, W_prime)
```

Characteristics:
- Potentially faster when T is small and spatial dims are tiny, because a single large CUDA kernel replaces T small kernels.
- Breaks batch normalization: BN running stats are computed over the wrong batch dimension.
- Incompatible with spiking state (state cannot be accumulated in a single forward pass).
- Requires careful re-partitioning of outputs back into per-timestep tensors before state update.

**Recommendation:** Use approach 1 inside `snn_unroll`. The conv operation is already spatially parallelized by CUDA. Flattening B*T provides meaningful speedup only in narrow regimes (very small T, very small spatial dims) that are atypical for SNN workloads. Approach 2 is a valid micro-optimization only after profiling confirms it is the bottleneck.

The `snn_unroll` utility abstracts this choice. ConvSNN provides a `cell_fn` that processes `(B, C, H, W)`; the unroll utility handles the time dimension. ConvSNN does not need to know whether approach 1 or 2 is used internally.

---

## E) State Containers

Replace the implicit `self.mem` pattern used in the current `SNNCore` and `ConvSNN` with an explicit, external state container.

### SpikingState Definition

```python
from dataclasses import dataclass
from typing import Optional
import torch
from torch import Tensor

@dataclass
class SpikingState:
    v: Tensor                          # membrane potential, shape (B, *neuron_shape)
    i: Optional[Tensor] = None         # synaptic current, shape (B, *neuron_shape)
    a: Optional[Tensor] = None         # adaptation variable, shape (B, *neuron_shape)
    ref: Optional[Tensor] = None       # refractory timer, shape (B, *neuron_shape)
    spike_history: Optional[Tensor] = None  # for delay models, shape (B, T_delay, *neuron_shape)

    def detach(self) -> 'SpikingState':
        """Detach all non-None fields from the computation graph."""
        return SpikingState(
            v=self.v.detach(),
            i=self.i.detach() if self.i is not None else None,
            a=self.a.detach() if self.a is not None else None,
            ref=self.ref.detach() if self.ref is not None else None,
            spike_history=(
                self.spike_history.detach()
                if self.spike_history is not None else None
            ),
        )

    @staticmethod
    def zeros(
        batch_size: int,
        neuron_shape: tuple,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        use_current: bool = False,
        use_adaptation: bool = False,
        use_refractory: bool = False,
        spike_history_len: int = 0,
    ) -> 'SpikingState':
        """Create a zero-initialized SpikingState."""
        shape = (batch_size, *neuron_shape)
        return SpikingState(
            v=torch.zeros(shape, device=device, dtype=dtype),
            i=torch.zeros(shape, device=device, dtype=dtype) if use_current else None,
            a=torch.zeros(shape, device=device, dtype=dtype) if use_adaptation else None,
            ref=torch.zeros(shape, device=device, dtype=dtype) if use_refractory else None,
            spike_history=(
                torch.zeros(batch_size, spike_history_len, *neuron_shape,
                            device=device, dtype=dtype)
                if spike_history_len > 0 else None
            ),
        )

    def to(self, device: torch.device) -> 'SpikingState':
        """Move all tensors to device."""
        return SpikingState(
            v=self.v.to(device),
            i=self.i.to(device) if self.i is not None else None,
            a=self.a.to(device) if self.a is not None else None,
            ref=self.ref.to(device) if self.ref is not None else None,
            spike_history=(
                self.spike_history.to(device)
                if self.spike_history is not None else None
            ),
        )
```

### Why External State

Storing state on `self` (as `self.mem`, `self.syn`) couples the module to a specific batch size and sequence position. This prevents:
- Running multiple independent sequences through the same module.
- Passing state across module boundaries without side effects.
- Correctly resetting state at sequence boundaries during dataloader iteration.

External state makes the module a stateless function parameterized by weights alone. State is passed in, updated, and returned. The caller controls state lifecycle.

---

## F) Integration with Existing Code

### Refactoring SNNCore

1. Extract the per-timestep logic from lines 112-150 of `snn.py` into a standalone `forward_step` method:

```python
def forward_step(self, x: Tensor, state: SpikingState) -> Tuple[Tensor, SpikingState]:
    # existing per-timestep computation here
    # return (spike, updated_state)
```

2. Define `cell_fn` as a closure or bound method:

```python
def cell_fn(x_t: Tensor, state: SpikingState) -> Tuple[Tensor, SpikingState]:
    return self.forward_step(x_t, state)
```

3. Replace the manual time loop in `forward` with `snn_unroll`:

```python
def forward(
    self,
    x: Tensor,
    state: Optional[SpikingState] = None,
    chunk_size: Optional[int] = None,
    return_traces: bool = False,
) -> Tuple[Tensor, SpikingState, Optional[Dict]]:
    if state is None:
        state = SpikingState.zeros(x.shape[0], (self.hidden_size,),
                                   device=x.device, dtype=x.dtype)
    if x.dim() == 2:
        # Step mode: (B, D) — process single timestep
        spike, state = self.forward_step(x, state)
        return spike, state, None
    # Sequence mode: (B, T, D)
    return snn_unroll(self.forward_step, x, state,
                      chunk_size=chunk_size, return_traces=return_traces)
```

### Refactoring ConvSNN

Apply the same pattern to `ConvSNN.forward()` at lines 291-309:

1. Extract per-timestep conv+spiking logic into `forward_step(self, x: Tensor, state: SpikingState)`.
2. In `forward`, detect sequence vs step mode by checking `x.dim()`.
3. For sequence mode, call `snn_unroll` with `self.forward_step` as `cell_fn`.

The `snn_unroll` utility is agnostic to the spatial structure of the tensor. ConvSNN's `cell_fn` receives `(B, C, H, W)` at each step; `snn_unroll` slices `inputs[:, t]` to produce this shape automatically because `inputs` is `(B, T, C, H, W)`.

---

## G) Recording Modes

The `return_traces` parameter controls how much internal state is collected during the unroll. Four operational modes arise from the combination of call mode and recording:

### Spikes Only (Default)

```python
spikes, state, traces = snn_unroll(cell_fn, x, state)
# traces is None
# spikes is (B, T, ...)
```

Minimal memory overhead. Use during standard training when membrane traces are not needed for the loss.

### Spikes + Membrane

```python
spikes, state, traces = snn_unroll(cell_fn, x, state, return_traces=True)
# traces['membrane'] is (B, T, *neuron_shape)
# traces['current'] is (B, T, *neuron_shape) — zeros if state.i is None
```

Use for visualization, debugging threshold calibration, or rate-coded decoding losses that operate on membrane values.

### Spikes + Membrane + Current

When `state.i` is non-None (synaptic current is modeled), `traces['current']` is populated. No additional flag is needed; the presence of `state.i` determines whether current is traced.

### Step Mode (No Traces)

In step mode, `snn_unroll` is not called. The module calls `forward_step` directly and returns `(spike, state)`. Trace collection is not supported in step mode because there is no sequence over which to accumulate traces.

---

## H) Common Anti-Patterns

Avoid all of the following. Each item describes a failure mode observed in or likely to emerge from the current dual-loop structure.

**Different unroll loops for Conv vs MLP.**
Running separate manual loops in `SNNCore` and `ConvSNN` means any fix to one (e.g., adding chunk support) must be duplicated. A single divergence creates silent behavioral differences. Delegate both to `snn_unroll`.

**Not detaching state between BPTT chunks.**
If `state.detach()` is omitted after each chunk, the computation graph grows to cover the full T steps regardless of `chunk_size`. Memory usage becomes identical to full BPTT. Gradient norms from early timesteps can explode or vanish. Always call `state.detach()` at chunk boundaries.

**Detaching spikes but not state.**
Detaching `spike_record[:, chunk_end - 1]` or the output tensor while leaving `state.v` attached is incorrect. The state is the only thing that carries gradient information forward between chunks. Spikes are outputs; state is the gradient-carrying signal.

**Using time-first convention in some modules and batch-first in others.**
A module that returns `(T, B, D)` when called from a trainer expecting `(B, T, D)` will silently transpose batch and time during training. Shape checks may not catch this if B == T. Enforce batch-first everywhere without exception.

**Accumulating spike records in a Python list without pre-allocation.**
```python
# Wrong
spikes = []
for t in range(T):
    s, state = cell_fn(x[:, t], state)
    spikes.append(s)
spikes = torch.stack(spikes, dim=1)
```
`torch.stack` on a Python list performs T memory allocations and a final copy. Pre-allocate `spike_record = torch.zeros(B, T, ...)` and assign `spike_record[:, t] = s` in-place. This is faster and avoids the copy.

**Not handling variable-length sequences.**
If batches contain sequences of different T (e.g., padded to the longest), the unroll must mask padded positions. Either pad with zeros and mask the loss, or use `torch.nn.utils.rnn.pack_padded_sequence` conventions. Do not pass padded inputs through cell_fn without masking — the cell accumulates state from padding timesteps, corrupting the final state.

**Resetting state inside cell_fn at sequence start.**
Cell_fn must not call `self.reset()` or modify `self.mem` internally. State is external. Any reset logic belongs to the caller before the `snn_unroll` call.

---

## I) Performance Considerations

### Pre-Allocated Tensors vs torch.stack

Pre-allocate the output buffer before the time loop and assign in-place:

```python
spike_record = torch.zeros(B, T, *spatial, device=inputs.device, dtype=inputs.dtype)
for t in range(T):
    spike_record[:, t] = cell_fn(inputs[:, t], state)[0]
```

This avoids T intermediate tensor allocations and the final `torch.stack` copy. For T=50 and large spatial dims, this reduces peak memory during the unroll by the size of one intermediate activation tensor per timestep.

### JIT Compilation

The unroll loop can be compiled with `torch.jit.script` if `cell_fn` is scriptable. Constraints:

- `cell_fn` must be a `torch.nn.Module` with a typed `forward` method, not a lambda or closure.
- `SpikingState` must be annotated with `@torch.jit.script` or declared as a named tuple.
- Optional fields (`i`, `a`, `ref`) must use `Optional[Tensor]` typing explicitly.

```python
@torch.jit.script
def snn_unroll_scripted(
    cell: torch.nn.Module,
    inputs: Tensor,
    state: SpikingState,
    chunk_size: int,
) -> Tuple[Tensor, SpikingState]:
    ...
```

JIT compilation eliminates Python interpreter overhead in the inner loop. For T=50 with small per-step ops, this can reduce unroll time by 20-40%.

### Gradient Checkpointing

For very long sequences (T > 100) where even truncated BPTT is memory-constrained, combine chunking with `torch.utils.checkpoint`:

```python
def checkpointed_chunk(cell_fn, chunk_inputs, state):
    def run_chunk(inputs, v, i):
        # reconstruct state, run chunk, return outputs and new state tensors
        ...
    return torch.utils.checkpoint.checkpoint(run_chunk, chunk_inputs, state.v, state.i)
```

Gradient checkpointing recomputes the forward pass during backward, trading compute for memory. Combined with `chunk_size=10`, this allows training on sequences of T=200+ with constant peak memory (relative to chunk size), at the cost of approximately 2x recomputation overhead.

Use gradient checkpointing only when:
- T exceeds 100 and chunked BPTT alone still causes OOM.
- Recomputation overhead is acceptable (batch throughput is memory-bound, not compute-bound).

### Mixed Precision

The `snn_unroll` utility is compatible with `torch.cuda.amp.autocast`. Wrap the call site:

```python
with torch.autocast(device_type='cuda', dtype=torch.float16):
    spikes, state, traces = snn_unroll(cell_fn, inputs, state, chunk_size=10)
```

Membrane potential arithmetic is sensitive to float16 precision. If instability is observed (NaN spikes, diverging membrane), add `torch.autocast` exclusions for the threshold comparison inside `cell_fn` using `with torch.autocast(enabled=False):`.

---

## Summary

The single `snn_unroll` utility defined in this document:

- Eliminates the dual-loop inconsistency between `SNNCore` and `ConvSNN`.
- Enforces the `(B, T, ...)` batch-first layout as the canonical convention.
- Provides configurable truncated BPTT via `chunk_size`, with correct state detachment at chunk boundaries covering all state fields.
- Supports optional membrane and current trace collection without changing the calling convention.
- Accepts any `cell_fn` that maps `(Tensor, SpikingState) -> (Tensor, SpikingState)`, making it agnostic to conv vs MLP topology.
- Replaces implicit `self.mem` state with explicit `SpikingState` containers that are always passed in and returned.

All SNN modules must delegate their time loop to this utility. No module may implement its own unroll loop.
