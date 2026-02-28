# Reasoning Trace Schema Reference

## Table of Contents

1. [Trace Purpose](#1-trace-purpose)
2. [ReasoningTrace Schema](#2-reasoningtrace-schema)
3. [RouteTrace Schema](#3-routetrace-schema)
4. [System1Trace Schema](#4-system1trace-schema)
5. [System2Trace Schema](#5-system2trace-schema)
6. [StepTrace Schema](#6-steptrace-schema)
7. [Serialization](#7-serialization)
8. [Storage Policy](#8-storage-policy)
9. [Performance Considerations](#9-performance-considerations)
10. [Trace Comparison Utilities](#10-trace-comparison-utilities)
11. [Code Examples](#11-code-examples)

---

## 1. Trace Purpose

The reasoning trace is an inspectable, structured record of the entire dual-process
reasoning pipeline for a single batch item. It captures every decision point from
initial confidence estimation through System 1 fast-path or System 2 deliberative
reasoning, all the way to the final output.

### Design Goals

- **Inspectability**: Provide a complete audit trail of how the model arrived at its
  output. Every routing decision, every iterative refinement step, and every
  convergence check is recorded with the numeric values that drove the decision.

- **Zero overhead when disabled**: The trace is returned only when the caller sets
  `return_details=True`. When this flag is `False` (the default), no trace objects
  are allocated, no tensor copies are made, and no per-step metric dictionaries are
  constructed. The hot path pays nothing.

- **JSON-serializable by default**: All fields are Python primitives (`int`, `float`,
  `bool`, `str`, `None`) or containers thereof (`List`, `Dict`). No PyTorch tensors,
  NumPy arrays, or other non-serializable objects appear in the trace. This makes it
  trivial to log traces to JSON files, databases, or monitoring systems.

- **Memory-bounded**: Rather than storing full `(B, C)` logit tensors, the trace
  retains only top-k logit values and their indices at each step. This keeps memory
  usage proportional to `O(steps * top_k)` rather than `O(steps * num_classes)`.

### Use Cases

| Use Case | Description |
|---|---|
| **Debugging** | Inspect why a particular input was routed to System 2, or why System 2 failed to converge. Examine per-step confidence deltas to identify oscillation or stalling. |
| **Interpretability** | Present a human-readable narrative of the reasoning process: "Input had confidence 0.42 (below threshold 0.70), routed to System 2, converged after 3 steps with final confidence 0.91." |
| **Logging** | Stream traces to a logging backend for offline analysis. Aggregate routing statistics (percent routed to S2, average steps used, convergence rate) across batches. |
| **Regression testing** | Compare traces across code changes, hardware migrations, or quantization levels. Verify that routing decisions remain stable and that logit distributions stay within tolerance. |
| **Curriculum analysis** | Track how routing patterns evolve during training. Early in training, most items route to System 2; as the model improves, more items resolve via System 1. |

### When Traces Are Created

Traces are created at the end of the `DualProcessReasoning.forward()` call, after
all computation is complete. The trace construction code runs outside the main
computational graph and does not participate in gradient computation. This is
enforced by wrapping trace construction in `torch.no_grad()`.

```python
# In DualProcessReasoning.forward():
if return_details:
    with torch.no_grad():
        trace = self._build_trace(route_info, s1_result, s2_result, metadata)
    return output, trace
else:
    return output
```

The two-return-value pattern means callers can destructure the result when they
need traces, and receive a single tensor when they do not.

---

## 2. ReasoningTrace Schema

The top-level trace object aggregates all sub-traces into a single dataclass.

```python
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class ReasoningTrace:
    """Complete reasoning trace for a single batch item."""

    route: RouteTrace
    """Routing decision details: confidence scores, threshold, and
    the binary decision of whether System 2 was invoked."""

    system1: System1Trace
    """System 1 (fast path) output summary. Always populated, even
    when System 2 is used, because S1 runs unconditionally to produce
    the initial logits that feed the routing decision."""

    system2: Optional[System2Trace]
    """System 2 (deliberative path) trace. None if the router decided
    that System 1 output was sufficient (confidence above threshold).
    When present, contains per-step traces of the iterative refinement."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Auxiliary metadata. Includes timing information, device placement,
    a snapshot of the relevant config fields, and any custom annotations
    added by downstream code."""
```

### Metadata Fields

The `metadata` dictionary contains the following standard keys. Implementations
may add additional keys as needed.

| Key | Type | Description |
|---|---|---|
| `timestamp` | `str` | ISO 8601 timestamp of when the trace was created. |
| `device` | `str` | PyTorch device string (e.g., `"cuda:0"`, `"cpu"`). |
| `dtype` | `str` | Tensor dtype used during computation (e.g., `"float32"`). |
| `batch_index` | `int` | Index of this item within the batch (0-based). |
| `batch_size` | `int` | Total batch size. |
| `config` | `Dict[str, Any]` | Flat dictionary snapshot of the reasoning config fields that influenced this trace (threshold, max_steps, top_k, etc.). |
| `wall_time_ms` | `float` | Total wall-clock time for this item's reasoning path, in milliseconds. |
| `s1_time_ms` | `float` | Wall-clock time for the System 1 forward pass. |
| `s2_time_ms` | `Optional[float]` | Wall-clock time for the System 2 loop. `None` if S2 was not invoked. |
| `model_version` | `Optional[str]` | Version string of the model checkpoint, if available. |

### One Trace Per Batch Item

A batch of size `B` produces a list of `B` independent `ReasoningTrace` objects.
Each trace corresponds to exactly one input item and contains no references to other
items in the batch. This makes it safe to serialize, filter, or discard individual
traces without affecting the rest.

```python
# Batch inference with traces
outputs, traces = model(batch_input, return_details=True)
assert len(traces) == batch_input.shape[0]

# Access trace for item 3
trace_3 = traces[3]
print(trace_3.route.used_system2)  # True or False
```

---

## 3. RouteTrace Schema

The route trace records the inputs and output of the routing decision. The router
examines the System 1 output logits and computes several confidence metrics to
decide whether the fast-path result is adequate or whether System 2 deliberation
is required.

```python
@dataclass
class RouteTrace:
    """Captures the routing decision and all signals that contributed to it."""

    used_system2: bool
    """Final binary decision: True if System 2 was invoked."""

    route_score: float
    """Composite routing score combining all confidence signals.
    This is the single scalar compared against `threshold` to make
    the routing decision. Values range from 0.0 (no confidence,
    definitely route to S2) to 1.0 (full confidence, stay in S1)."""

    threshold: float
    """The confidence threshold in effect at the time of the decision.
    When route_score < threshold, the item routes to System 2.
    Default is 0.7. May vary during training via curriculum schedules."""

    conf_raw: float
    """Raw confidence: softmax probability of the argmax class.
    Computed as max(softmax(logits)). Range [1/C, 1.0] where C is
    the number of classes."""

    conf_calibrated: float
    """Calibrated confidence after temperature scaling or Platt scaling.
    More reliable than conf_raw for threshold comparisons. Range [0, 1]."""

    entropy: float
    """Shannon entropy of the softmax distribution, normalized to [0, 1].
    High entropy indicates uncertainty (uniform distribution). Low entropy
    indicates the model is confident in one or a few classes.
    Computed as: -sum(p * log(p)) / log(C)."""

    margin: float
    """Margin between the top-1 and top-2 softmax probabilities.
    Large margin indicates clear separation between the best and
    second-best predictions. Range [0, 1]."""

    novelty: Optional[float]
    """Novelty score from the HTM anomaly detector or an auxiliary
    novelty network. None if novelty detection is not enabled.
    High novelty suggests the input is out-of-distribution and may
    benefit from deliberative reasoning. Range [0, 1] when present."""

    anomaly: Optional[float]
    """Anomaly score from the global workspace or upstream modules.
    None if anomaly scoring is not available in the current config.
    Semantically distinct from novelty: anomaly captures structural
    irregularities, novelty captures distributional distance.
    Range [0, 1] when present."""

    ignition: Optional[float]
    """Global workspace ignition strength. None if the workspace module
    is disabled. Low ignition suggests the input failed to achieve
    broad cortical broadcast, which may indicate ambiguity.
    Range [0, 1] when present."""

    steps_budget: int
    """Maximum number of System 2 steps allocated for this item.
    Determined by the config's max_steps field, possibly adjusted by
    a dynamic budget allocator based on estimated difficulty.
    When used_system2 is False, this field still records the budget
    that would have been available."""
```

### Routing Score Composition

The `route_score` is computed as a weighted combination of the individual signals.
The exact formula is configurable, but the default is:

```python
route_score = (
    w_conf * conf_calibrated
    + w_entropy * (1.0 - entropy)
    + w_margin * margin
    + w_novelty * (1.0 - novelty) if novelty is not None else 0.0
    + w_anomaly * (1.0 - anomaly) if anomaly is not None else 0.0
    + w_ignition * ignition if ignition is not None else 0.0
) / sum_of_active_weights
```

The weights (`w_conf`, `w_entropy`, etc.) are stored in the config and may be
learned or hand-tuned. The normalization by `sum_of_active_weights` ensures the
score stays in `[0, 1]` regardless of which optional signals are available.

### Threshold Interpretation

| Condition | Interpretation |
|---|---|
| `route_score >= threshold` | System 1 output is accepted. `used_system2 = False`. |
| `route_score < threshold` | System 1 output is insufficient. Route to System 2. `used_system2 = True`. |
| `threshold = 0.0` | Never route to System 2 (pure S1 mode). |
| `threshold = 1.0` | Always route to System 2 (pure S2 mode). |

During training, a curriculum schedule may gradually lower the threshold as the
model's System 1 path becomes more capable, reducing the computational cost of
inference over time.

---

## 4. System1Trace Schema

System 1 (the fast path) always executes, even when System 2 is subsequently
invoked. The System 1 trace captures a summary of the fast-path output, storing
only the top-k logit values rather than the full logit tensor.

```python
from typing import List


@dataclass
class System1Trace:
    """Summary of System 1 (fast path) output."""

    top_k_indices: List[int]
    """Class indices of the top-k logits, sorted descending by value.
    Example: [7, 3, 1] means class 7 had the highest logit, class 3
    the second highest, etc."""

    top_k_values: List[float]
    """Logit values (pre-softmax) corresponding to top_k_indices.
    Example: [4.21, 2.87, 1.05]. These are raw logits, not probabilities.
    Stored as Python floats, not tensors."""

    top_k: int
    """The value of k used for this trace. Matches len(top_k_indices)
    and len(top_k_values). Default is 5. Configurable up to 20 for
    detailed debugging."""
```

### Why Top-K Only

Storing the full logit vector for every traced item would consume
`B * C * sizeof(float)` bytes per forward pass, where `C` can be large (e.g.,
21,000 for ImageNet-21k). For a batch of 256 with C=21,000:

```
256 * 21,000 * 4 bytes = ~20.6 MB per batch
```

With top-k=5, the storage drops to:

```
256 * 5 * (4 + 4) bytes = ~10.2 KB per batch
```

This is a 2000x reduction. For most debugging and interpretability purposes,
knowing the top 5-10 predictions is sufficient.

### Configuring k

The `top_k` parameter is set in the reasoning config:

```python
@dataclass
class DualProcessConfig:
    trace_top_k: int = 5       # Default: store top 5
    trace_top_k_max: int = 20  # Hard cap to prevent accidental bloat
```

Setting `trace_top_k` to a value greater than `trace_top_k_max` raises a
`ValueError` at config validation time.

### Invariants

- `len(top_k_indices) == len(top_k_values) == top_k`
- `top_k_indices` contains no duplicates
- `top_k_values` is sorted in descending order
- `top_k_indices[0]` is always the argmax class
- All values in `top_k_values` are finite (no `inf`, no `nan`)

If the forward pass produces `nan` or `inf` in the logits, the trace builder
replaces them with sentinel values (`-999.0` for `nan`, `+/-998.0` for `+/-inf`)
and sets `metadata["logit_sanitized"] = True` as a warning flag.

---

## 5. System2Trace Schema

The System 2 trace is present only when the router decided to invoke deliberative
reasoning (`route.used_system2 == True`). It captures the iterative refinement
loop: how many steps were taken, whether convergence was achieved, why the loop
terminated, and per-step details.

```python
from typing import List, Optional


@dataclass
class System2Trace:
    """Trace of the System 2 (deliberative) iterative refinement loop."""

    steps_used: int
    """Number of refinement steps actually executed. Always >= 1 (at
    least one step runs before the first convergence check). Always
    <= route.steps_budget."""

    converged: bool
    """True if the loop terminated because the convergence criterion
    was satisfied. False if terminated for any other reason (max steps,
    budget exhaustion, NaN guard)."""

    halt_reason: str
    """Human-readable string describing why the loop terminated.
    One of the following values:
      - "converged": KL divergence and max-logit delta both fell
        below their respective thresholds for `patience` consecutive
        steps.
      - "max_steps": The step counter reached the allocated budget
        without satisfying the convergence criterion.
      - "budget_exhausted": An external budget controller (e.g., a
        latency-aware scheduler) signaled early termination.
      - "nan_guard": A NaN or Inf was detected in the logits during
        a refinement step. The loop terminated immediately to prevent
        propagation. The last valid logits are used as the output.
    """

    steps: List[StepTrace]
    """Per-step traces, one for each refinement step executed. Length
    equals steps_used. See StepTrace schema for details."""

    final_top_k_indices: List[int]
    """Top-k class indices of the final (post-S2) logits. These may
    differ from the System1Trace top-k if deliberation changed the
    model's prediction."""

    final_top_k_values: List[float]
    """Top-k logit values of the final (post-S2) logits, corresponding
    to final_top_k_indices."""
```

### Halt Reasons in Detail

#### "converged"

The convergence criterion checks two conditions simultaneously:

1. **KL divergence** between step `k` and step `k-1` logit distributions is below
   `convergence_kl_threshold` (default: 0.001).
2. **Max-logit delta** (absolute change in the argmax logit value) is below
   `convergence_delta_threshold` (default: 0.01).

Both conditions must hold for `patience` consecutive steps (default: 1). This
prevents false convergence from a single lucky step.

```python
# Convergence check pseudocode
if delta_kl < kl_threshold and delta_max < delta_threshold:
    patience_counter += 1
    if patience_counter >= patience:
        halt_reason = "converged"
        break
else:
    patience_counter = 0
```

#### "max_steps"

The step counter reached `steps_budget` without convergence. This is the most
common halt reason during early training when the model's System 2 path has not
yet learned to refine efficiently.

#### "budget_exhausted"

An external signal (set via a flag on the module or passed through the metadata
dict) instructed the loop to terminate early. This is used by latency-aware
inference servers that need to bound wall-clock time.

#### "nan_guard"

A NaN or Inf was detected in the logits at step `k`. The loop terminates
immediately and the output uses the logits from step `k-1`. If NaN occurs at
step 0 (the first refinement step), the System 1 logits are used as fallback.

The trace records which step triggered the NaN guard:

```python
metadata["nan_guard_step"] = step_idx
```

### Comparing S1 and S2 Outputs

A common analysis pattern is comparing the System 1 prediction with the final
System 2 prediction to see if deliberation changed the answer:

```python
s1_pred = trace.system1.top_k_indices[0]
s2_pred = trace.system2.final_top_k_indices[0]
prediction_changed = s1_pred != s2_pred
```

When `prediction_changed` is True, the trace provides a full audit trail of how
and when the prediction shifted during the refinement loop (visible in the per-step
`argmax_k` field of `StepTrace`).

---

## 6. StepTrace Schema

Each iteration of the System 2 refinement loop produces a `StepTrace` recording
the key metrics for that step. These traces are collected into the
`System2Trace.steps` list.

```python
from typing import List, Optional


@dataclass
class StepTrace:
    """Metrics for a single System 2 refinement step."""

    step_idx: int
    """Zero-based index of this step within the refinement loop."""

    conf_k: float
    """Confidence (calibrated softmax probability of the argmax class)
    at this step. Tracks how confidence evolves across steps. Expected
    to increase monotonically in well-trained models, though oscillation
    is possible and informative for debugging."""

    delta_kl: float
    """KL divergence between the logit distribution at this step and
    the previous step. Measures how much the distribution changed.
    At step 0, this is the KL divergence between the S2 step-0 output
    and the original S1 output. Decreasing delta_kl indicates the
    model is converging."""

    delta_max: float
    """Absolute change in the argmax logit value between this step and
    the previous step. A complementary convergence signal to delta_kl.
    Large delta_max with small delta_kl can indicate that the top
    prediction is shifting while the overall distribution stays similar."""

    argmax_k: int
    """Class index of the argmax at this step. Tracking this across
    steps reveals whether the model's top prediction is stable or
    oscillating between classes."""

    halt_check_passed: bool
    """True if the convergence criterion was satisfied at this step.
    When patience > 1, halt_check_passed can be True for several
    steps before the loop actually terminates."""

    top_k_indices: Optional[List[int]]
    """Top-k class indices at this step. Only populated when
    full_trace=True in the config. None otherwise, to save memory
    in the default trace mode."""

    top_k_values: Optional[List[float]]
    """Top-k logit values at this step. Only populated when
    full_trace=True. None otherwise."""
```

### Step 0 Semantics

Step 0 is the first refinement step. Its `delta_kl` and `delta_max` are computed
relative to the System 1 output, not relative to a "step -1". This means step 0's
deltas measure how much the first deliberative step changed the prediction from
the fast-path baseline.

```
delta_kl[0] = KL(softmax(logits_s2_step0) || softmax(logits_s1))
delta_max[0] = |max(logits_s2_step0) - max(logits_s1)|
```

### Full Trace Mode

By default, `StepTrace` omits the per-step `top_k_indices` and `top_k_values` to
keep memory usage low. Enable full trace mode by setting `full_trace=True` in the
reasoning config:

```python
config = DualProcessConfig(
    full_trace=True,       # Enable per-step top-k logging
    trace_top_k=20,        # Increase k for more detail
)
```

Full trace mode is intended for development and debugging sessions, not for
production inference. The additional memory cost is:

```
per_item_cost = steps_used * top_k * 2 * sizeof(float)
```

For `steps_used=8` and `top_k=20`, this is `8 * 20 * 2 * 4 = 1280 bytes` per
item -- modest in absolute terms, but it adds up across large batches and long
training runs.

### Interpreting Step Metrics

| Pattern | Diagnosis |
|---|---|
| `conf_k` increases monotonically, `delta_kl` decreases | Healthy convergence. The model is becoming more confident with each step. |
| `conf_k` oscillates, `argmax_k` alternates between two values | The model is indecisive between two classes. May indicate an ambiguous input or undertrained S2 path. |
| `delta_kl` is large at every step, never decreases | The S2 GRU is not learning to refine. Check S2 learning rate and hidden state initialization. |
| `conf_k` drops below System 1 confidence | Deliberation is making things worse. The routing threshold may be too low, or S2 is undertrained. |
| `halt_check_passed` is True at step 2 but loop runs to step 5 | Patience is set > 1 and the convergence criterion was not sustained. The model briefly appeared converged but then changed its mind. |
| `delta_max` is large but `delta_kl` is small | The top logit is shifting but the overall distribution is stable. Often seen when two classes have similar logits and the argmax flips. |

---

## 7. Serialization

All trace dataclasses support bidirectional conversion between Python objects and
JSON-compatible dictionaries. This enables logging to files, databases, message
queues, and REST APIs without custom serialization logic.

### Methods

Every trace dataclass (`ReasoningTrace`, `RouteTrace`, `System1Trace`,
`System2Trace`, `StepTrace`) implements three methods:

#### `to_dict() -> dict`

Convert the trace to a nested dictionary. All values are JSON-compatible Python
primitives: `int`, `float`, `bool`, `str`, `None`, `list`, or `dict`.

```python
def to_dict(self) -> dict:
    """Convert trace to a JSON-compatible nested dictionary.

    Tensors are converted to Python scalars or lists.
    Optional fields that are None are included as null.
    Nested traces are recursively converted.
    """
    result = {}
    for f in fields(self):
        value = getattr(self, f.name)
        if hasattr(value, 'to_dict'):
            result[f.name] = value.to_dict()
        elif isinstance(value, list) and value and hasattr(value[0], 'to_dict'):
            result[f.name] = [item.to_dict() for item in value]
        elif isinstance(value, torch.Tensor):
            result[f.name] = value.detach().cpu().tolist()
        elif isinstance(value, (np.integer, np.floating)):
            result[f.name] = value.item()
        else:
            result[f.name] = value
    return result
```

#### `to_json(indent=None) -> str`

Convenience wrapper around `to_dict()` that returns a JSON string.

```python
def to_json(self, indent: Optional[int] = None) -> str:
    """Convert trace to a JSON string.

    Args:
        indent: JSON indentation level. None for compact output,
                2 or 4 for human-readable output.
    """
    return json.dumps(self.to_dict(), indent=indent)
```

#### `from_dict(d: dict) -> ReasoningTrace` (classmethod)

Reconstruct a trace object from a dictionary (as produced by `to_dict()`).

```python
@classmethod
def from_dict(cls, d: dict) -> 'ReasoningTrace':
    """Reconstruct a ReasoningTrace from a dictionary.

    Nested trace objects are recursively reconstructed.
    Missing optional fields default to None.
    """
    route = RouteTrace.from_dict(d['route'])
    system1 = System1Trace.from_dict(d['system1'])
    system2 = (
        System2Trace.from_dict(d['system2'])
        if d.get('system2') is not None
        else None
    )
    metadata = d.get('metadata', {})
    return cls(
        route=route,
        system1=system1,
        system2=system2,
        metadata=metadata,
    )
```

### Tensor Conversion Rules

Before any value is placed into the trace dataclass fields, tensors must be
converted to Python primitives. The following rules apply:

| Tensor Shape | Conversion | Result Type |
|---|---|---|
| Scalar (`[]`) | `.item()` | `float` or `int` |
| 1-D (`[K]`) | `.tolist()` | `List[float]` or `List[int]` |
| 2-D or higher | Not allowed | Raise `ValueError` |

The trace builder enforces these conversions at construction time, not at
serialization time. This means the dataclass fields are always Python-native and
there is no risk of accidentally holding a reference to a GPU tensor.

```python
def _to_scalar(t: torch.Tensor) -> float:
    """Convert a scalar tensor to a Python float."""
    return t.detach().cpu().item()

def _to_list(t: torch.Tensor) -> list:
    """Convert a 1-D tensor to a Python list."""
    return t.detach().cpu().tolist()
```

### Timestamp Format

All timestamps in the `metadata` dictionary use ISO 8601 format with timezone:

```python
from datetime import datetime, timezone

metadata['timestamp'] = datetime.now(timezone.utc).isoformat()
# Example: "2026-02-19T14:32:07.123456+00:00"
```

### Config Snapshot Format

The config snapshot stored in `metadata["config"]` is a flat dictionary with
string keys and primitive values. Nested config objects are flattened with dot
notation:

```python
metadata['config'] = {
    'threshold': 0.7,
    'max_steps': 8,
    'convergence_kl_threshold': 0.001,
    'convergence_delta_threshold': 0.01,
    'patience': 1,
    'trace_top_k': 5,
    'full_trace': False,
    'system2.hidden_dim': 512,
    'system2.num_layers': 2,
}
```

---

## 8. Storage Policy

The trace schema is designed around a tiered storage policy that balances
observability against memory cost. The policy is controlled by two config flags:
`return_details` (master switch) and `full_trace` (detail level).

### Tier 0: No Trace (`return_details=False`)

- No `ReasoningTrace` objects are created.
- No tensor copies or `.item()` calls are made.
- No per-step metric dictionaries are allocated.
- The forward pass returns only the output tensor.
- **Memory overhead: zero.**

This is the default mode for training and production inference.

### Tier 1: Default Trace (`return_details=True`, `full_trace=False`)

- `ReasoningTrace` is created with all fields populated.
- `System1Trace` stores top-k=5 logit indices and values.
- `System2Trace` stores final top-k=5 and per-step scalar metrics.
- `StepTrace.top_k_indices` and `StepTrace.top_k_values` are `None`.
- **Memory per item: ~200-500 bytes** (depending on steps_used).

This is the recommended mode for debugging and logging.

### Tier 2: Full Trace (`return_details=True`, `full_trace=True`)

- Same as Tier 1, plus per-step top-k indices and values.
- `trace_top_k` can be increased up to 20.
- `StepTrace.top_k_indices` and `StepTrace.top_k_values` are populated.
- **Memory per item: ~1-3 KB** (depending on steps_used and top_k).

This is for detailed debugging sessions and interpretability research.

### What Is Never Stored

Regardless of tier, the following are never stored in the trace:

- Full logit tensors of shape `(C,)` or `(B, C)`.
- GRU hidden states from the System 2 refinement loop.
- Intermediate activation tensors from any module.
- Gradient information.
- Input tensors or embeddings.

If full logit tensors are needed for analysis, capture them separately outside
the trace system using PyTorch hooks or by saving the logits tensor directly.

### Batch Traces

A batch of `B` items produces `B` independent trace objects, returned as a
plain Python list:

```python
outputs, traces = model(batch_input, return_details=True)
# traces: List[ReasoningTrace], len == B
```

Each trace is independent and can be serialized, filtered, or discarded without
affecting others. There is no shared state between traces in the list.

For large batches where only a subset of traces are needed (e.g., only items
that routed to System 2), filter after the forward pass:

```python
s2_traces = [t for t in traces if t.route.used_system2]
```

Do not attempt to conditionally create traces for only some batch items during
the forward pass. The branching logic would complicate the batched computation
and is not worth the savings.

---

## 9. Performance Considerations

### Zero-Cost When Disabled

The most critical performance property: when `return_details=False`, the trace
system imposes exactly zero overhead. Verify this by examining the forward path:

```python
def forward(self, x, return_details=False):
    # ... all computation ...

    if return_details:
        # This entire block is skipped when return_details=False
        with torch.no_grad():
            trace = self._build_trace(...)
        return output, trace
    return output
```

No conditional branches, no metric accumulators, no tensor copies exist outside
the `if return_details` block. The flag is checked once, at the end.

### Overhead When Enabled

When `return_details=True`, the overhead consists of:

1. **Tensor-to-scalar conversions**: `O(top_k)` calls to `.item()` per step, per
   item. Each `.item()` call synchronizes the CUDA stream (if on GPU), which can
   add latency.

2. **Top-k computation**: `torch.topk(logits, k)` is called once for S1 and once
   per S2 step. The cost is `O(C * log(k))` per call.

3. **Dictionary construction**: Python dict and list allocations for the trace
   object. Negligible compared to tensor operations.

4. **KL divergence for trace**: If delta_kl is already computed for the
   convergence check (which it is), no additional computation is needed. The trace
   simply records the value.

### Avoiding CUDA Synchronization

The most significant performance pitfall is calling `.item()` on GPU tensors
inside the inner loop. Each `.item()` forces a CUDA synchronization, stalling the
CPU until the GPU completes all queued operations.

To mitigate this, collect raw metrics as CPU tensors in a list during the loop,
then convert them to Python scalars in a single batch after the loop completes:

```python
# BAD: .item() inside the loop causes per-step synchronization
for step in range(max_steps):
    logits = refine(hidden, logits)
    conf = softmax(logits).max().item()  # CUDA sync here!
    step_metrics.append({'conf': conf})

# GOOD: Collect tensors, convert after loop
raw_confs = []
for step in range(max_steps):
    logits = refine(hidden, logits)
    raw_confs.append(softmax(logits).max().detach())

# Single sync point after the loop
if return_details:
    confs = [c.cpu().item() for c in raw_confs]
```

However, note that the convergence check itself already requires reading
`delta_kl` and `delta_max` to decide whether to halt. If these are already on
CPU (which they must be for the comparison), then recording them into the trace
adds no additional synchronization.

### Memory Allocation Pattern

Avoid allocating trace objects inside the inner loop. Instead, collect raw
numeric values into pre-allocated lists and construct the `StepTrace` objects
after the loop:

```python
# Pre-allocate metric collectors
step_confs = []
step_delta_kls = []
step_delta_maxs = []
step_argmaxs = []
step_halt_checks = []

# Inner loop: collect raw values only
for step in range(max_steps):
    logits, hidden = self.s2_cell(logits, hidden)
    # ... convergence check ...
    step_confs.append(conf_value)
    step_delta_kls.append(delta_kl_value)
    step_delta_maxs.append(delta_max_value)
    step_argmaxs.append(argmax_value)
    step_halt_checks.append(halt_passed)

# After loop: build trace objects
if return_details:
    step_traces = [
        StepTrace(
            step_idx=i,
            conf_k=step_confs[i],
            delta_kl=step_delta_kls[i],
            delta_max=step_delta_maxs[i],
            argmax_k=step_argmaxs[i],
            halt_check_passed=step_halt_checks[i],
            top_k_indices=None,
            top_k_values=None,
        )
        for i in range(steps_used)
    ]
```

### Benchmarks

Approximate overhead measurements on a single A100 GPU, batch size 256,
C=1000 classes, max_steps=8:

| Mode | Overhead per batch | Relative to base forward |
|---|---|---|
| `return_details=False` | 0 ms | 0% |
| `return_details=True`, `full_trace=False` | ~2-4 ms | ~1-2% |
| `return_details=True`, `full_trace=True`, k=20 | ~5-8 ms | ~2-4% |

The overhead is dominated by CUDA synchronization for `.item()` calls, not by
Python object construction.

---

## 10. Trace Comparison Utilities

The trace comparison system supports regression testing by comparing two traces
and producing a structured diff. This is useful for verifying that code changes,
hardware migrations, or quantization do not alter model behavior beyond acceptable
tolerances.

### TraceDiff Schema

```python
@dataclass
class TraceDiff:
    """Structured comparison between two ReasoningTrace objects."""

    same_route: bool
    """True if both traces made the same routing decision
    (both used S2 or both stayed in S1)."""

    same_prediction: bool
    """True if both traces produced the same argmax prediction."""

    same_halt_reason: bool
    """True if both traces have the same halt_reason. Only meaningful
    when both used System 2. Always True when both used System 1."""

    route_score_delta: float
    """Absolute difference in route_score between the two traces."""

    conf_delta: float
    """Absolute difference in final calibrated confidence."""

    steps_delta: int
    """Difference in steps_used (trace2.steps_used - trace1.steps_used).
    Zero when both used System 1."""

    logit_deltas: List[float]
    """Per-position absolute differences in the top-k logit values
    of the final output. Length equals min(k1, k2)."""

    max_logit_delta: float
    """Maximum value in logit_deltas. Convenience field for threshold
    comparisons."""

    details: Dict[str, Any]
    """Additional comparison details: per-step argmax alignment,
    convergence step comparison, etc."""
```

### compare_traces Function

```python
def compare_traces(
    t1: ReasoningTrace,
    t2: ReasoningTrace,
    logit_atol: float = 0.01,
    conf_atol: float = 0.005,
    route_score_atol: float = 0.01,
) -> TraceDiff:
    """Compare two reasoning traces and produce a structured diff.

    Args:
        t1: First trace (typically the reference/baseline).
        t2: Second trace (typically the candidate/new version).
        logit_atol: Absolute tolerance for logit value comparisons.
        conf_atol: Absolute tolerance for confidence comparisons.
        route_score_atol: Absolute tolerance for route score comparisons.

    Returns:
        TraceDiff with all comparison fields populated.
    """
```

### Comparison Logic

The comparison proceeds in stages:

1. **Route comparison**: Check whether both traces made the same routing decision.
   Compare `route_score` values within tolerance.

2. **Prediction comparison**: Compare the argmax of the final output (S1 top-k
   if S2 was not used, S2 final top-k if S2 was used).

3. **Halt reason comparison**: If both traces used System 2, check that they
   terminated for the same reason.

4. **Logit comparison**: Align the top-k lists by class index and compute
   per-position deltas. Classes present in one top-k but not the other are
   flagged in `details["unmatched_classes"]`.

5. **Step-level comparison** (when both used S2): Compare the per-step argmax
   sequences to detect divergence points. Record the first step where the two
   traces disagree on argmax.

```python
# Stage 5: Step-level argmax alignment
if t1.system2 and t2.system2:
    min_steps = min(t1.system2.steps_used, t2.system2.steps_used)
    first_divergence = None
    for i in range(min_steps):
        if t1.system2.steps[i].argmax_k != t2.system2.steps[i].argmax_k:
            first_divergence = i
            break
    details['first_argmax_divergence_step'] = first_divergence
    details['argmax_sequences'] = {
        't1': [s.argmax_k for s in t1.system2.steps],
        't2': [s.argmax_k for s in t2.system2.steps],
    }
```

### Tolerance-Based Pass/Fail

For automated regression testing, define a pass criterion based on the `TraceDiff`:

```python
def trace_matches(diff: TraceDiff, strict: bool = False) -> bool:
    """Determine if two traces match within acceptable tolerances.

    Args:
        diff: The TraceDiff produced by compare_traces.
        strict: If True, require exact match on all discrete fields
                (same_route, same_prediction, same_halt_reason) AND
                all continuous fields within tolerance. If False,
                only require same_prediction.
    """
    if strict:
        return (
            diff.same_route
            and diff.same_prediction
            and diff.same_halt_reason
            and diff.max_logit_delta < logit_atol
        )
    else:
        return diff.same_prediction
```

### Use Cases for Comparison

| Scenario | Comparison Strategy |
|---|---|
| **Code refactor** | Strict comparison. All fields must match within tolerance. Any deviation indicates a regression. |
| **Quantization (FP32 to FP16)** | Relaxed tolerances (`logit_atol=0.05`). Expect small numeric differences but same predictions. |
| **Hardware migration (GPU A to GPU B)** | Moderate tolerances. Different GPU architectures may produce slightly different floating-point results. |
| **Training checkpoint comparison** | Compare traces across epochs. Expect prediction changes (the model is learning), but verify convergence behavior improves (fewer steps, higher convergence rate). |
| **Determinism verification** | Run the same input twice on the same device with the same seed. Strict comparison with `logit_atol=0.0`. Any difference indicates non-determinism. |

---

## 11. Code Examples

### Example 1: ReasoningTrace.to_dict() and to_json()

```python
import json
from datetime import datetime, timezone
from brain_ai.reasoning.trace import ReasoningTrace, RouteTrace, System1Trace


# Build a minimal trace (System 1 only)
trace = ReasoningTrace(
    route=RouteTrace(
        used_system2=False,
        route_score=0.85,
        threshold=0.70,
        conf_raw=0.82,
        conf_calibrated=0.85,
        entropy=0.12,
        margin=0.67,
        novelty=None,
        anomaly=None,
        ignition=0.91,
        steps_budget=8,
    ),
    system1=System1Trace(
        top_k_indices=[7, 3, 1, 9, 0],
        top_k_values=[4.21, 2.87, 1.05, 0.44, 0.12],
        top_k=5,
    ),
    system2=None,
    metadata={
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'device': 'cuda:0',
        'dtype': 'float32',
        'batch_index': 0,
        'batch_size': 32,
        'wall_time_ms': 1.23,
        's1_time_ms': 1.23,
        's2_time_ms': None,
        'config': {
            'threshold': 0.70,
            'max_steps': 8,
            'trace_top_k': 5,
        },
    },
)

# Convert to dict
d = trace.to_dict()
assert isinstance(d, dict)
assert d['route']['used_system2'] is False
assert d['system2'] is None
assert isinstance(d['metadata']['timestamp'], str)

# Convert to JSON string
json_str = trace.to_json(indent=2)
print(json_str)

# Round-trip: reconstruct from dict
trace_restored = ReasoningTrace.from_dict(d)
assert trace_restored.route.route_score == trace.route.route_score
assert trace_restored.system1.top_k_indices == trace.system1.top_k_indices
```

### Example 2: Building a Trace from S1 and S2 Results

```python
import time
import torch
from datetime import datetime, timezone
from brain_ai.reasoning.trace import (
    ReasoningTrace, RouteTrace, System1Trace, System2Trace, StepTrace,
)


def build_trace(
    s1_logits: torch.Tensor,       # (C,) logits from System 1
    s2_result: dict,               # Output from the S2 loop (or None)
    route_info: dict,              # Output from the router
    config: dict,                  # Reasoning config as flat dict
    batch_idx: int,
    batch_size: int,
    device: str,
    start_time: float,
) -> ReasoningTrace:
    """Build a complete ReasoningTrace from computation results.

    All tensor-to-scalar conversions happen here, not in the forward pass.
    This function runs under torch.no_grad().
    """
    top_k = config.get('trace_top_k', 5)

    # System 1 trace: always built
    s1_topk_vals, s1_topk_idx = torch.topk(s1_logits.detach().cpu(), top_k)
    system1 = System1Trace(
        top_k_indices=s1_topk_idx.tolist(),
        top_k_values=s1_topk_vals.tolist(),
        top_k=top_k,
    )

    # Route trace
    route = RouteTrace(
        used_system2=route_info['used_system2'],
        route_score=float(route_info['route_score']),
        threshold=float(route_info['threshold']),
        conf_raw=float(route_info['conf_raw']),
        conf_calibrated=float(route_info['conf_calibrated']),
        entropy=float(route_info['entropy']),
        margin=float(route_info['margin']),
        novelty=_maybe_float(route_info.get('novelty')),
        anomaly=_maybe_float(route_info.get('anomaly')),
        ignition=_maybe_float(route_info.get('ignition')),
        steps_budget=int(route_info['steps_budget']),
    )

    # System 2 trace: built only if S2 was invoked
    system2 = None
    s2_time = None
    if s2_result is not None:
        final_logits = s2_result['final_logits'].detach().cpu()
        final_topk_vals, final_topk_idx = torch.topk(final_logits, top_k)

        step_traces = _build_step_traces(
            s2_result['step_metrics'],
            full_trace=config.get('full_trace', False),
            top_k=top_k,
        )

        system2 = System2Trace(
            steps_used=s2_result['steps_used'],
            converged=s2_result['converged'],
            halt_reason=s2_result['halt_reason'],
            steps=step_traces,
            final_top_k_indices=final_topk_idx.tolist(),
            final_top_k_values=final_topk_vals.tolist(),
        )
        s2_time = s2_result.get('wall_time_ms')

    wall_time = (time.perf_counter() - start_time) * 1000.0

    metadata = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'device': device,
        'dtype': 'float32',
        'batch_index': batch_idx,
        'batch_size': batch_size,
        'config': config,
        'wall_time_ms': round(wall_time, 3),
        's1_time_ms': round(route_info.get('s1_time_ms', 0.0), 3),
        's2_time_ms': round(s2_time, 3) if s2_time is not None else None,
    }

    return ReasoningTrace(
        route=route,
        system1=system1,
        system2=system2,
        metadata=metadata,
    )


def _maybe_float(val):
    """Convert a value to float if not None."""
    if val is None:
        return None
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().item()
    return float(val)
```

### Example 3: StepTrace Collection in the S2 Loop

```python
import torch
import torch.nn.functional as F


def run_system2_loop(
    logits: torch.Tensor,       # (C,) initial logits from S1
    hidden: torch.Tensor,       # (H,) initial GRU hidden state
    s2_cell: torch.nn.Module,   # GRU refinement cell
    max_steps: int,
    kl_threshold: float,
    delta_threshold: float,
    patience: int,
    return_details: bool,
    full_trace: bool,
    top_k: int,
) -> dict:
    """Execute the System 2 iterative refinement loop.

    Returns a dict with final logits, convergence info, and
    optionally per-step metrics for trace construction.
    """
    prev_logits = logits.detach().clone()
    prev_dist = F.softmax(prev_logits, dim=-1)
    patience_counter = 0
    converged = False
    halt_reason = "max_steps"
    steps_used = 0

    # Metric collectors (no trace objects in the loop)
    step_confs = []
    step_delta_kls = []
    step_delta_maxs = []
    step_argmaxs = []
    step_halt_checks = []
    step_topk_indices = []  # Only used if full_trace
    step_topk_values = []   # Only used if full_trace

    for step in range(max_steps):
        # Refinement step
        logits, hidden = s2_cell(logits, hidden)
        steps_used = step + 1

        # Compute convergence metrics (needed regardless of trace)
        curr_dist = F.softmax(logits, dim=-1)
        delta_kl = F.kl_div(
            curr_dist.log(), prev_dist, reduction='sum'
        ).item()
        delta_max = abs(logits.max().item() - prev_logits.max().item())
        conf_k = curr_dist.max().item()
        argmax_k = logits.argmax().item()

        # NaN guard
        if not (torch.isfinite(logits).all()):
            halt_reason = "nan_guard"
            logits = prev_logits  # Revert to last valid
            break

        # Convergence check
        halt_passed = (delta_kl < kl_threshold and delta_max < delta_threshold)
        if halt_passed:
            patience_counter += 1
            if patience_counter >= patience:
                converged = True
                halt_reason = "converged"
                # Record metrics for this final step before breaking
                if return_details:
                    step_confs.append(conf_k)
                    step_delta_kls.append(delta_kl)
                    step_delta_maxs.append(delta_max)
                    step_argmaxs.append(argmax_k)
                    step_halt_checks.append(True)
                    if full_trace:
                        topk_v, topk_i = torch.topk(
                            logits.detach().cpu(), top_k
                        )
                        step_topk_indices.append(topk_i.tolist())
                        step_topk_values.append(topk_v.tolist())
                break
        else:
            patience_counter = 0

        # Collect raw metrics (no object allocation)
        if return_details:
            step_confs.append(conf_k)
            step_delta_kls.append(delta_kl)
            step_delta_maxs.append(delta_max)
            step_argmaxs.append(argmax_k)
            step_halt_checks.append(halt_passed)
            if full_trace:
                topk_v, topk_i = torch.topk(logits.detach().cpu(), top_k)
                step_topk_indices.append(topk_i.tolist())
                step_topk_values.append(topk_v.tolist())

        # Update previous state for next step
        prev_logits = logits.detach().clone()
        prev_dist = curr_dist.detach()

    # Build result dict
    result = {
        'final_logits': logits,
        'final_hidden': hidden,
        'steps_used': steps_used,
        'converged': converged,
        'halt_reason': halt_reason,
    }

    # Attach step metrics for trace construction
    if return_details:
        result['step_metrics'] = {
            'confs': step_confs,
            'delta_kls': step_delta_kls,
            'delta_maxs': step_delta_maxs,
            'argmaxs': step_argmaxs,
            'halt_checks': step_halt_checks,
            'topk_indices': step_topk_indices if full_trace else None,
            'topk_values': step_topk_values if full_trace else None,
        }

    return result


def _build_step_traces(
    step_metrics: dict,
    full_trace: bool,
    top_k: int,
) -> list:
    """Convert raw step metrics into StepTrace objects.

    Called after the S2 loop completes, outside the computation graph.
    """
    n_steps = len(step_metrics['confs'])
    traces = []
    for i in range(n_steps):
        traces.append(StepTrace(
            step_idx=i,
            conf_k=step_metrics['confs'][i],
            delta_kl=step_metrics['delta_kls'][i],
            delta_max=step_metrics['delta_maxs'][i],
            argmax_k=step_metrics['argmaxs'][i],
            halt_check_passed=step_metrics['halt_checks'][i],
            top_k_indices=(
                step_metrics['topk_indices'][i]
                if full_trace and step_metrics['topk_indices']
                else None
            ),
            top_k_values=(
                step_metrics['topk_values'][i]
                if full_trace and step_metrics['topk_values']
                else None
            ),
        ))
    return traces
```

### Example 4: TraceDiff Comparison

```python
from brain_ai.reasoning.trace import (
    ReasoningTrace, TraceDiff, compare_traces, trace_matches,
)


def compare_traces(
    t1: ReasoningTrace,
    t2: ReasoningTrace,
    logit_atol: float = 0.01,
    conf_atol: float = 0.005,
    route_score_atol: float = 0.01,
) -> TraceDiff:
    """Compare two reasoning traces and return a structured diff."""

    # Route comparison
    same_route = t1.route.used_system2 == t2.route.used_system2
    route_score_delta = abs(t1.route.route_score - t2.route.route_score)

    # Prediction comparison (use S2 final if available, else S1)
    pred1 = (
        t1.system2.final_top_k_indices[0]
        if t1.system2 is not None
        else t1.system1.top_k_indices[0]
    )
    pred2 = (
        t2.system2.final_top_k_indices[0]
        if t2.system2 is not None
        else t2.system1.top_k_indices[0]
    )
    same_prediction = pred1 == pred2

    # Halt reason comparison
    hr1 = t1.system2.halt_reason if t1.system2 else "s1_only"
    hr2 = t2.system2.halt_reason if t2.system2 else "s1_only"
    same_halt_reason = hr1 == hr2

    # Confidence delta (final confidence)
    conf1 = t1.route.conf_calibrated
    conf2 = t2.route.conf_calibrated
    if t1.system2 and t1.system2.steps:
        conf1 = t1.system2.steps[-1].conf_k
    if t2.system2 and t2.system2.steps:
        conf2 = t2.system2.steps[-1].conf_k
    conf_delta = abs(conf1 - conf2)

    # Steps delta
    s1 = t1.system2.steps_used if t1.system2 else 0
    s2 = t2.system2.steps_used if t2.system2 else 0
    steps_delta = s2 - s1

    # Logit deltas (align by position in top-k)
    vals1 = (
        t1.system2.final_top_k_values
        if t1.system2 is not None
        else t1.system1.top_k_values
    )
    vals2 = (
        t2.system2.final_top_k_values
        if t2.system2 is not None
        else t2.system1.top_k_values
    )
    min_k = min(len(vals1), len(vals2))
    logit_deltas = [abs(vals1[i] - vals2[i]) for i in range(min_k)]
    max_logit_delta = max(logit_deltas) if logit_deltas else 0.0

    # Step-level comparison details
    details = {}
    if t1.system2 and t2.system2:
        min_steps = min(t1.system2.steps_used, t2.system2.steps_used)
        first_divergence = None
        for i in range(min_steps):
            if t1.system2.steps[i].argmax_k != t2.system2.steps[i].argmax_k:
                first_divergence = i
                break
        details['first_argmax_divergence_step'] = first_divergence
        details['argmax_sequences'] = {
            't1': [s.argmax_k for s in t1.system2.steps],
            't2': [s.argmax_k for s in t2.system2.steps],
        }

    return TraceDiff(
        same_route=same_route,
        same_prediction=same_prediction,
        same_halt_reason=same_halt_reason,
        route_score_delta=route_score_delta,
        conf_delta=conf_delta,
        steps_delta=steps_delta,
        logit_deltas=logit_deltas,
        max_logit_delta=max_logit_delta,
        details=details,
    )


# --- Usage in regression tests ---

def test_determinism(model, test_input, seed=42):
    """Verify that the model produces identical traces on repeated runs."""
    torch.manual_seed(seed)
    _, trace1 = model(test_input, return_details=True)

    torch.manual_seed(seed)
    _, trace2 = model(test_input, return_details=True)

    for i in range(len(trace1)):
        diff = compare_traces(trace1[i], trace2[i], logit_atol=0.0)
        assert diff.same_route, f"Item {i}: routing decision changed"
        assert diff.same_prediction, f"Item {i}: prediction changed"
        assert diff.max_logit_delta == 0.0, (
            f"Item {i}: logit delta {diff.max_logit_delta} != 0.0"
        )


def test_quantization_stability(model_fp32, model_fp16, test_input):
    """Verify that FP16 quantization preserves predictions."""
    _, traces_fp32 = model_fp32(test_input, return_details=True)
    _, traces_fp16 = model_fp16(test_input.half(), return_details=True)

    mismatches = 0
    for i in range(len(traces_fp32)):
        diff = compare_traces(
            traces_fp32[i], traces_fp16[i],
            logit_atol=0.05,      # Relaxed for FP16
            conf_atol=0.02,
            route_score_atol=0.02,
        )
        if not diff.same_prediction:
            mismatches += 1

    mismatch_rate = mismatches / len(traces_fp32)
    assert mismatch_rate < 0.01, (
        f"Quantization changed {mismatch_rate:.1%} of predictions"
    )
```

### Example 5: Logging Traces to a JSON File

```python
import json
import gzip
from pathlib import Path
from typing import List


def save_traces(
    traces: List[ReasoningTrace],
    path: str,
    compress: bool = True,
) -> None:
    """Save a batch of traces to a JSON file.

    Args:
        traces: List of ReasoningTrace objects from a forward pass.
        path: Output file path. If compress=True, '.gz' is appended.
        compress: Whether to gzip the output.
    """
    data = [t.to_dict() for t in traces]

    if compress:
        path = path if path.endswith('.gz') else path + '.gz'
        with gzip.open(path, 'wt', encoding='utf-8') as f:
            json.dump(data, f, separators=(',', ':'))
    else:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)


def load_traces(path: str) -> List[ReasoningTrace]:
    """Load traces from a JSON file.

    Args:
        path: Input file path. Handles both plain and gzipped JSON.

    Returns:
        List of ReasoningTrace objects.
    """
    open_fn = gzip.open if path.endswith('.gz') else open
    with open_fn(path, 'rt', encoding='utf-8') as f:
        data = json.load(f)

    return [ReasoningTrace.from_dict(d) for d in data]


# Usage
outputs, traces = model(batch_input, return_details=True)
save_traces(traces, 'traces/epoch_12_batch_0045.json')

# Later: load and analyze
loaded = load_traces('traces/epoch_12_batch_0045.json.gz')
s2_count = sum(1 for t in loaded if t.route.used_system2)
avg_steps = (
    sum(t.system2.steps_used for t in loaded if t.system2)
    / max(s2_count, 1)
)
print(f"S2 routing rate: {s2_count}/{len(loaded)}")
print(f"Average S2 steps: {avg_steps:.1f}")
```

### Example 6: Aggregating Trace Statistics Across Batches

```python
from collections import defaultdict
from typing import List


class TraceAggregator:
    """Accumulate trace statistics across multiple batches.

    Lightweight: stores only scalar counters, not full traces.
    """

    def __init__(self):
        self.total_items = 0
        self.s2_items = 0
        self.converged_items = 0
        self.total_steps = 0
        self.halt_reasons = defaultdict(int)
        self.conf_sum = 0.0
        self.conf_min = float('inf')
        self.conf_max = float('-inf')

    def update(self, traces: List[ReasoningTrace]) -> None:
        """Ingest a batch of traces."""
        for t in traces:
            self.total_items += 1
            conf = t.route.conf_calibrated
            self.conf_sum += conf
            self.conf_min = min(self.conf_min, conf)
            self.conf_max = max(self.conf_max, conf)

            if t.route.used_system2 and t.system2 is not None:
                self.s2_items += 1
                self.total_steps += t.system2.steps_used
                self.halt_reasons[t.system2.halt_reason] += 1
                if t.system2.converged:
                    self.converged_items += 1
            else:
                self.halt_reasons['s1_only'] += 1

    def summary(self) -> dict:
        """Return aggregated statistics as a dictionary."""
        return {
            'total_items': self.total_items,
            's2_routing_rate': (
                self.s2_items / self.total_items
                if self.total_items > 0 else 0.0
            ),
            's2_convergence_rate': (
                self.converged_items / self.s2_items
                if self.s2_items > 0 else 0.0
            ),
            'avg_s2_steps': (
                self.total_steps / self.s2_items
                if self.s2_items > 0 else 0.0
            ),
            'avg_confidence': (
                self.conf_sum / self.total_items
                if self.total_items > 0 else 0.0
            ),
            'min_confidence': self.conf_min,
            'max_confidence': self.conf_max,
            'halt_reasons': dict(self.halt_reasons),
        }


# Usage during training
aggregator = TraceAggregator()
for batch in dataloader:
    outputs, traces = model(batch, return_details=True)
    aggregator.update(traces)

stats = aggregator.summary()
print(f"S2 routing rate: {stats['s2_routing_rate']:.1%}")
print(f"S2 convergence rate: {stats['s2_convergence_rate']:.1%}")
print(f"Average S2 steps: {stats['avg_s2_steps']:.1f}")
print(f"Halt reasons: {stats['halt_reasons']}")
```

---

## Appendix: Quick Reference Card

### Trace Object Hierarchy

```
ReasoningTrace
+-- route: RouteTrace
|   +-- used_system2: bool
|   +-- route_score: float
|   +-- threshold: float
|   +-- conf_raw: float
|   +-- conf_calibrated: float
|   +-- entropy: float
|   +-- margin: float
|   +-- novelty: Optional[float]
|   +-- anomaly: Optional[float]
|   +-- ignition: Optional[float]
|   +-- steps_budget: int
+-- system1: System1Trace
|   +-- top_k_indices: List[int]
|   +-- top_k_values: List[float]
|   +-- top_k: int
+-- system2: Optional[System2Trace]
|   +-- steps_used: int
|   +-- converged: bool
|   +-- halt_reason: str
|   +-- steps: List[StepTrace]
|   |   +-- step_idx: int
|   |   +-- conf_k: float
|   |   +-- delta_kl: float
|   |   +-- delta_max: float
|   |   +-- argmax_k: int
|   |   +-- halt_check_passed: bool
|   |   +-- top_k_indices: Optional[List[int]]
|   |   +-- top_k_values: Optional[List[float]]
|   +-- final_top_k_indices: List[int]
|   +-- final_top_k_values: List[float]
+-- metadata: Dict[str, Any]
    +-- timestamp: str (ISO 8601)
    +-- device: str
    +-- dtype: str
    +-- batch_index: int
    +-- batch_size: int
    +-- config: Dict[str, Any]
    +-- wall_time_ms: float
    +-- s1_time_ms: float
    +-- s2_time_ms: Optional[float]
    +-- model_version: Optional[str]
```

### Storage Tiers at a Glance

| Tier | Flag | Per-Item Cost | Use Case |
|---|---|---|---|
| 0 | `return_details=False` | 0 bytes | Training, production |
| 1 | `return_details=True` | ~200-500 B | Debugging, logging |
| 2 | `full_trace=True` | ~1-3 KB | Deep debugging, research |

### Halt Reasons

| Value | Meaning |
|---|---|
| `"converged"` | KL + delta below threshold for `patience` steps |
| `"max_steps"` | Reached `steps_budget` without convergence |
| `"budget_exhausted"` | External early-termination signal |
| `"nan_guard"` | NaN/Inf detected; reverted to last valid logits |
