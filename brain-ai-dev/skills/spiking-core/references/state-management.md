# State Management: Reset vs Carry — Make It Explicit, Test It, Never Guess

## The Problem

Silent state leakage across batch items is the #1 source of irreproducible SNN training bugs. When membrane potential from batch N leaks into batch N+1:

- Training appears to work but converges to a worse solution
- Results differ between batch size 1 and batch size 32
- CPU and CUDA give different results due to different accumulation order

The existing codebase stores state implicitly on neuron instances (`self.mem`, `self.adaptation`, `self.prev_spk`, `self.spike_history`). This pattern is the root cause. The fix is not subtle — eliminate implicit state entirely and replace it with an explicit contract that every layer must satisfy.

The system-orchestrator skill defines `snn_state` as `Optional[Dict[str, Tensor]]` in `BrainAIState`. The gap between that definition and the current layer implementations is:

- No `reset_state(B, device, dtype)` method on any layer
- No `detach_state()` method on any layer
- No `carry_state` flag in `forward()`
- State shapes are not validated against batch size

Closing this gap is the primary obligation of this skill.

---

## A) The Explicit State Contract

Every spiking layer must provide the following interface. No exceptions.

```python
from dataclasses import dataclass
from typing import Optional, Set, Tuple
import torch
from torch import Tensor


@dataclass
class SpikingState:
    v: Tensor                              # membrane potential, always fp32
    i: Optional[Tensor] = None            # synaptic current (if modeled)
    a: Optional[Tensor] = None            # adaptation variable
    ref: Optional[Tensor] = None          # refractory counter
    spike_history: Optional[Tensor] = None  # (B, max_delay, N) for delay models


class SpikingLayer(torch.nn.Module):

    def reset_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> SpikingState:
        """Return fresh zero-initialized state. Never store on self.

        The returned SpikingState owns its tensors. The layer holds no
        reference to them. Calling reset_state twice returns two independent
        objects with no shared storage.
        """
        raise NotImplementedError

    def detach_state(self, state: SpikingState) -> SpikingState:
        """Return state with identical values but detached from computation graph.

        Used for truncated BPTT between chunks. The detached state is a new set
        of leaf tensors. Numerical values are byte-for-byte identical to the
        input state; the computation graph is severed.
        """
        raise NotImplementedError

    def forward(
        self,
        x: Tensor,
        state: Optional[SpikingState] = None,
        carry_state: bool = False,
    ) -> Tuple[Tensor, SpikingState]:
        """Run one timestep.

        Args:
            x: input tensor
            state: explicit state from previous step, or None to auto-reset
            carry_state: if True, the layer is in streaming mode; caller owns
                         state lifecycle. If False, the returned state is only
                         valid for the duration of the current forward pass.

        Returns:
            (spike_output, new_state)
        """
        raise NotImplementedError
```

These three methods form the complete contract. A layer that implements `forward()` without also implementing `reset_state()` and `detach_state()` is incomplete and must not be merged.

---

## B) State Shape Rules

All state tensors must be batch-aligned. The batch dimension `B` is always the first dimension. No exceptions. The specific shapes are:

| Layer type      | State fields and shapes                              |
|----------------|------------------------------------------------------|
| `LIFNeuron` (linear)  | `v: (B, N)`                                 |
| `LIFNeuron` (conv)    | `v: (B, C, H, W)`                          |
| `AdaptiveLIF`         | `v: (B, N)`, `a: (B, N)`                   |
| `RecurrentLIF`        | `v: (B, N)`, `prev_spk: (B, N)` (as `i`)  |
| `AdvancedLIF`         | `v: (B, N)`, `spike_history: (B, max_delay, N)`, `a: (B, N)` |

Do not use `(N, B)` for any reason, even if the internal weight matrix is transposed. Keep the API surface consistent: `B` first everywhere. Mixing conventions between layers causes subtle correctness bugs when the orchestrator reconstructs state from its flat dictionary representation.

When implementing `reset_state()`, derive the spatial dimensions from the layer's stored weight shapes, not from the input tensor. Using the input tensor for shape inference inherits the wrong shape when `x` has a time dimension prepended.

```python
# Correct: derive shape from weight
def reset_state(self, batch_size, device, dtype):
    N = self.weight.shape[0]
    v = torch.zeros(batch_size, N, device=device, dtype=torch.float32)
    return SpikingState(v=v)

# Wrong: derive shape from x (x may have time dim)
def reset_state_bad(self, x):
    v = torch.zeros_like(x)  # shape may be (T, B, N) — wrong
    return SpikingState(v=v)
```

---

## C) Training Mode (Default Behavior)

In training mode, follow this sequence on every forward call:

1. If no `state` argument is provided, call `reset_state(B, device, dtype)` to create fresh zeros. `B` is inferred from `x.shape[0]`.
2. State carries across timesteps within a sequence. This is correct and intentional — membrane potential must accumulate within a sequence.
3. Between batches and between optimizer steps: state is discarded. The next batch call gets a fresh `reset_state()`.
4. Between BPTT chunks within a long sequence: state is detached. Values are preserved; the computation graph is cut.

```python
def forward(self, x, state=None, carry_state=False):
    B = x.shape[0]
    device = x.device

    if state is None:
        state = self.reset_state(B, device, torch.float32)

    x_fp32 = x.float()
    v = self.beta * state.v + x_fp32
    spike = self.surrogate(v - self.threshold)
    v = v * (1.0 - spike.detach())  # reset only, don't block grad through spike

    new_state = SpikingState(v=v)
    return spike.to(x.dtype), new_state
```

This is the standard pattern. No state leakage across batch iterations occurs because `state=None` causes a hard reset at the start of every independent forward call.

---

## D) Streaming Inference Mode

In inference mode with `carry_state=True`:

1. The caller passes in the state returned by the previous call.
2. State carries across calls, enabling online and streaming processing.
3. The caller is responsible for managing state lifecycle — including deciding when to reset.
4. Wrap the call in `torch.no_grad()` to prevent memory growth.

```python
state = layer.reset_state(B=1, device=device, dtype=torch.float32)

with torch.no_grad():
    for frame in audio_stream:
        spike, state = layer(frame, state=state, carry_state=True)
        process(spike)
```

Example use cases:

- Processing an audio stream frame by frame, maintaining SNN state across frames
- Online anomaly detection where the neuron must remember context from thousands of steps prior
- Real-time sensory integration where batch boundaries do not correspond to sequence boundaries

When `carry_state=True`, the layer must not zero the state at the start of `forward()`. The `carry_state` flag is how the caller signals this intent explicitly.

---

## E) The detach_state() Operation

Implement `detach_state()` as a pure function that creates new leaf tensors:

```python
def detach_state(self, state: SpikingState) -> SpikingState:
    return SpikingState(
        v=state.v.detach(),
        i=state.i.detach() if state.i is not None else None,
        a=state.a.detach() if state.a is not None else None,
        ref=state.ref.detach() if state.ref is not None else None,
        spike_history=(
            state.spike_history.detach()
            if state.spike_history is not None
            else None
        ),
    )
```

What detach does:

- Keeps numerical values byte-for-byte identical to the input state
- Severs the computation graph — the returned tensors are leaf tensors with `grad_fn is None`
- Allows the memory of the previous computation graph to be freed by the garbage collector
- Makes the detached state safe to pass as input to the next BPTT chunk

Critical distinction: `detach()` and `clone()` are not interchangeable. `clone()` creates a copy but preserves `grad_fn`, meaning the gradient can still flow back through it. Using `clone()` instead of `detach()` in truncated BPTT causes the gradient to flow through the detach boundary, defeating the purpose of chunking. This is a frequent mistake. Use `detach()`.

After calling `detach_state()`, the returned state is a new set of leaf tensors. The original `state` object is unchanged. Neither object holds a reference to the other.

---

## F) Partial Reset

Partial reset is occasionally useful for controlled experiments: reset only some state components while preserving others.

```python
def partial_reset(
    self,
    state: SpikingState,
    components: Set[str],
    batch_size: int,
    device: torch.device,
) -> SpikingState:
    """Reset specified components to zero; keep others unchanged.

    Args:
        state: current state
        components: set of field names to zero, e.g. {'v', 'a'}
        batch_size: used to construct fresh zero tensors for reset fields
        device: target device for fresh tensors
    """
    fresh = self.reset_state(batch_size, device, torch.float32)
    return SpikingState(
        v=fresh.v if 'v' in components else state.v,
        i=fresh.i if 'i' in components else state.i,
        a=fresh.a if 'a' in components else state.a,
        ref=fresh.ref if 'ref' in components else state.ref,
        spike_history=(
            fresh.spike_history
            if 'spike_history' in components
            else state.spike_history
        ),
    )
```

Use case: reset membrane potential but preserve adaptation history, to test whether adaptation state carries useful information across sequences independent of membrane dynamics. This is an experimental tool. Production training code should use full reset or full carry, not partial reset.

---

## G) State and the Orchestrator

The system-orchestrator skill stores SNN state as:

```python
snn_state: Optional[Dict[str, Tensor]]  # {layer_name: membrane_potential}
```

This flat dictionary is the orchestrator's uniform state representation across all module types. Layers must map between their internal `SpikingState` dataclass and this flat dictionary format on every orchestrator boundary crossing.

### Flattening (layer to orchestrator)

Use a deterministic naming scheme with layer prefix and field name separated by a period:

```python
def state_to_dict(self, layer_name: str, state: SpikingState) -> Dict[str, Tensor]:
    out = {}
    out[f"{layer_name}.v"] = state.v
    if state.i is not None:
        out[f"{layer_name}.i"] = state.i
    if state.a is not None:
        out[f"{layer_name}.a"] = state.a
    if state.ref is not None:
        out[f"{layer_name}.ref"] = state.ref
    if state.spike_history is not None:
        out[f"{layer_name}.spike_history"] = state.spike_history
    return out
```

### Reconstructing (orchestrator to layer)

```python
def dict_to_state(self, layer_name: str, d: Dict[str, Tensor]) -> SpikingState:
    return SpikingState(
        v=d[f"{layer_name}.v"],
        i=d.get(f"{layer_name}.i"),
        a=d.get(f"{layer_name}.a"),
        ref=d.get(f"{layer_name}.ref"),
        spike_history=d.get(f"{layer_name}.spike_history"),
    )
```

The orchestrator does not inspect the tensor contents. It stores, passes through, and eventually discards the state. The layer is entirely responsible for the semantics of its own state fields. The orchestrator only needs the flat dictionary to function — the `SpikingState` dataclass is an internal layer detail.

When the orchestrator's `snn_state` is `None` (first call, or after explicit reset), layers receive `state=None` in their `forward()` call and must call `reset_state()` internally.

---

## H) Batch-Wise State Considerations

Some scenarios require per-sample state management within a batch:

- Different samples in the batch may have different sequence lengths
- Some samples may be at the start of a new sequence (need reset) while others continue a running sequence
- Padding tokens at the end of short sequences must not corrupt membrane state

Handle this via a boolean reset mask applied after state update:

```python
def apply_reset_mask(
    self,
    new_state: SpikingState,
    old_state: SpikingState,
    reset_mask: Tensor,  # (B,) bool, True = reset this sample
) -> SpikingState:
    """Selectively reset state for samples indicated by reset_mask."""
    mask = reset_mask.float()
    # Reshape mask to broadcast over state dimensions
    mask_v = mask.view(-1, *([1] * (new_state.v.dim() - 1)))

    fresh_v = torch.zeros_like(new_state.v)
    v = new_state.v * (1.0 - mask_v) + fresh_v * mask_v

    a = None
    if new_state.a is not None:
        mask_a = mask.view(-1, *([1] * (new_state.a.dim() - 1)))
        a = new_state.a * (1.0 - mask_a)

    return SpikingState(v=v, i=new_state.i, a=a, ref=new_state.ref,
                        spike_history=new_state.spike_history)
```

This allows a single batched forward pass to correctly handle a mix of continuing and resetting sequences without splitting the batch. The mask is computed by the data loader or sequence packer and passed alongside the input tensor.

---

## I) Mixed Precision State Policy

Hard rule: state tensors (`v`, `i`, `a`) stay in `fp32` even when the model is running under AMP (Automatic Mixed Precision).

```python
def forward(self, x, state=None, carry_state=False):
    B = x.shape[0]
    device = x.device

    if state is None:
        state = self.reset_state(B, device, torch.float32)

    # Cast input for accumulation; state is always fp32
    x_fp32 = x.float()
    v = self.beta * state.v + x_fp32          # fp32 accumulation
    spike = self.surrogate(v - self.threshold) # fp32 spike computation
    v = v * (1.0 - spike.detach())

    new_state = SpikingState(v=v)              # fp32 state returned
    spike_out = spike.to(x.dtype)             # cast output back for downstream
    return spike_out, new_state
```

Why fp32 state is mandatory: membrane potential accumulates small increments over many timesteps. In `fp16`, the minimum representable positive value above 1.0 is approximately 0.001. Input currents below this threshold are silently rounded to zero, causing the membrane to stall — it stops changing even when input is nonzero. In `bf16`, the dynamic range is wider but precision at small increments is similarly limited.

This failure mode appears as training loss plateauing mysteriously around step 5000-10000, long after AMP appears to be working correctly. The plateau is not due to learning rate, batch size, or optimizer — it is due to membrane dynamics collapsing to a fixed point because state updates fall below representable precision.

The fix is categorical: keep `v`, `i`, and `a` in `fp32` always. Cast the output spike tensor back to the caller's dtype so downstream layers receive the expected dtype. AMP's `autocast` context will not automatically cast `fp32` tensors to `fp16`, so this works correctly without any explicit AMP interaction.

---

## J) Testing State Semantics

The following tests must pass for every spiking layer implementation. Run them as part of the unit test suite for any new or modified layer.

### Test 1: Reset Determinism

Two calls to `reset_state()` followed by identical inputs must produce identical outputs.

```python
def test_reset_determinism(layer, x):
    state_a = layer.reset_state(B=4, device=torch.device('cpu'), dtype=torch.float32)
    y1, s1 = layer(x, state_a)

    state_b = layer.reset_state(B=4, device=torch.device('cpu'), dtype=torch.float32)
    y2, s2 = layer(x, state_b)

    assert torch.equal(y1, y2), "Outputs must be identical after fresh reset"
    assert torch.equal(s1.v, s2.v), "States must be identical after fresh reset"
```

### Test 2: Carry Changes Output

Running the layer with a carried (non-zero) state must produce a different output than running from a fresh reset state.

```python
def test_carry_changes_output(layer, x):
    state = layer.reset_state(B=4, device=torch.device('cpu'), dtype=torch.float32)
    y1, s1 = layer(x, state)

    y2, s2 = layer(x, s1)           # carry from previous step
    y3, _ = layer(x, layer.reset_state(B=4, device=torch.device('cpu'),
                                        dtype=torch.float32))  # fresh reset

    # y2 used carried state; y3 used fresh state — they must differ
    # (if they happen to be equal due to zero input, the test is vacuous)
    if not torch.equal(x, torch.zeros_like(x)):
        assert not torch.equal(y2, y3), "Carried state must change output"
```

### Test 3: Detach Preserves Values

`detach_state()` must return tensors with identical numerical values.

```python
def test_detach_preserves_values(layer, x):
    state = layer.reset_state(B=4, device=torch.device('cpu'), dtype=torch.float32)
    y1, s1 = layer(x, state)

    s1_detached = layer.detach_state(s1)

    assert torch.equal(s1.v, s1_detached.v), "Detach must preserve membrane values"
    if s1.a is not None:
        assert torch.equal(s1.a, s1_detached.a), "Detach must preserve adaptation values"
```

### Test 4: Detach Cuts Gradients

Gradients must not flow through the detach boundary.

```python
def test_detach_cuts_gradients(layer, x, x2):
    x = x.requires_grad_(True)
    x2 = x2.requires_grad_(True)

    state = layer.reset_state(B=4, device=torch.device('cpu'), dtype=torch.float32)
    y1, s1 = layer(x, state)

    s1_d = layer.detach_state(s1)
    assert s1_d.v.grad_fn is None, "Detached state must have no grad_fn"

    y2, s2 = layer(x2, s1_d)
    loss = y2.sum()
    loss.backward()

    assert x.grad is None, "Gradient must not flow back through detach boundary"
    assert x2.grad is not None, "Gradient must flow for current chunk input"
```

### Test 5: State Shape Consistency

State shapes must match the batch size passed to `reset_state()`, regardless of input shape.

```python
def test_state_shape(layer):
    for B in [1, 4, 16, 32]:
        state = layer.reset_state(B=B, device=torch.device('cpu'), dtype=torch.float32)
        assert state.v.shape[0] == B, f"State batch dim must equal {B}"
```

### Test 6: No Self State Mutation

Calling `forward()` twice with the same state object must not silently mutate it.

```python
def test_no_self_mutation(layer, x):
    state = layer.reset_state(B=4, device=torch.device('cpu'), dtype=torch.float32)
    v_before = state.v.clone()

    layer(x, state)

    assert torch.equal(state.v, v_before), \
        "forward() must not mutate the input state object"
```

---

## K) Migration from Implicit State

The existing code uses the `self.mem` pattern. Migrate in three stages without breaking existing callers.

### Stage 1: Add the explicit interface, keep backward compatibility

Add `reset_state()` and `detach_state()` methods. In `forward()`, if `state=None` and the layer still has `self.mem`, use `self.mem` to construct a `SpikingState` but emit a deprecation warning.

```python
import warnings

def forward(self, x, state=None, carry_state=False):
    if state is None:
        if hasattr(self, 'mem') and self.mem is not None:
            warnings.warn(
                "Implicit self.mem state is deprecated. Pass state explicitly "
                "via reset_state(). self.mem will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )
            state = SpikingState(v=self.mem)
        else:
            state = self.reset_state(x.shape[0], x.device, torch.float32)
    # ... rest of forward
```

### Stage 2: Update all callers

Update training loops, test fixtures, and orchestrator integration points to pass state explicitly. Verify all deprecation warnings are gone from the test suite output.

### Stage 3: Remove self.mem

Delete `self.mem` initialization from `__init__()` and the backward-compatibility branch from `forward()`. The deprecation warning machinery is no longer needed.

Do not skip Stage 1. Removing `self.mem` before updating all callers will cause silent failures in code paths that relied on implicit state without being aware of it.

---

## L) Anti-Patterns

These patterns cause bugs that are difficult to diagnose. Avoid all of them.

**Storing state on self (`self.mem`) instead of passing explicitly.** This is the root cause of the leakage problem. The layer becomes stateful in a way that is invisible to the caller. Two calls that should be independent silently share state.

**Forgetting to detach between BPTT chunks.** The computation graph grows unboundedly across chunks. Training consumes all available memory and eventually OOMs, or the gradient becomes numerically meaningless because it has accumulated over thousands of steps.

**Using `.clone()` instead of `.detach()`.** `clone()` creates a copy of the data but preserves `grad_fn`. The gradient can still flow back through the cloned tensor. This defeats the purpose of the detach operation in BPTT chunking. Always use `.detach()` when the intent is to cut the graph.

**Not resetting `spike_history` in `AdvancedLIF`.** The delay model's spike history tensor holds spikes from up to `max_delay` previous timesteps. If the history is not zeroed in `reset_state()`, stale spikes from the previous sequence contaminate the delay line of the current sequence from the very first timestep.

**Different state initialization for Conv vs Linear variants of the same layer.** If `LIFNeuron` initializes `v` as `(B, N)` in linear mode and `(B, C, H, W)` in conv mode, but the orchestrator always stores it under the same key `"layer_0.v"`, shape mismatches will occur when switching between architectures. Ensure the naming and structure of state dictionaries is consistent regardless of the layer's spatial mode.

**Using `torch.zeros_like(x)` for state initialization.** If `x` has a time dimension prepended (e.g., shape `(T, B, N)`), the resulting state tensor has the wrong shape. Always derive state shapes from the layer's stored weight dimensions, not from the input tensor.

**Initializing state in `__init__()` with a hardcoded batch size.** State must be created at the point of use, with the actual batch size of the current input. Hardcoding batch size in `__init__()` is incompatible with dynamic batching and breaks when batch sizes vary between training and inference.

**Carrying state across sequences without detaching.** In multi-sequence training (e.g., language modeling over concatenated documents), state must be detached at sequence boundaries even if the underlying data is contiguous in memory. Failing to detach at sequence boundaries allows gradients to flow backward into preceding sequences, violating the intended independence of training examples.
