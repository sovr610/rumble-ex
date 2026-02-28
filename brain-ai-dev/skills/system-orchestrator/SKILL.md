---
name: BrainAI System Orchestrator
description: >
  This skill should be used when the user asks to "refactor the orchestrator",
  "refactor forward pass", "wire up feature flags", "add a module to the pipeline",
  "add a new cognitive module", "fix type contracts", "add telemetry",
  "implement state management", "add bypass adapters", "validate module contracts",
  "check pipeline determinism", "debug module routing", "add a new modality",
  "fix pipeline context", or mentions system.py, PipelinePlan, PipelineContext,
  BrainAIState, SystemOutput, Stage, or feature-flag routing in the BrainAI
  orchestrator or brain_ai cognitive architecture.
version: 0.1.0
---

# BrainAI System Orchestrator

## Purpose

Guide the design and maintenance of the BrainAI orchestrator — the control plane
that wires all cognitive modules (SNN, HTM, Workspace, Reasoning, Active Inference,
Meta-Learning, Engram) into a single deterministic, type-safe, reproducible pipeline.

The orchestrator does NOT invent ML modules. It makes every module behave like a
well-typed, swappable component with guaranteed fallbacks when disabled.

## Key Files

| File | Role |
|------|------|
| `brain_ai/system.py` | `BrainAI(nn.Module)` orchestrator, forward pass, factory functions |
| `brain_ai/types.py` | All typed contracts: `ModalityBatch`, `EncoderOutput`, `SystemOutput`, etc. |
| `brain_ai/pipeline.py` | `PipelinePlan` with ordered `Stage` objects and `PipelineContext` |
| `brain_ai/utils/repro.py` | Seed management, `stable_topk`, deterministic helpers |
| `brain_ai/config.py` | `BrainAIConfig` with feature flags and presets |

## Core Architecture

### Pipeline Execution Flow

```
ctx = normalize_inputs(inputs, config, device, dtype)
enc_outs = run_encoders(ctx)
if use_engram: add engram encoder to competition set
if use_snn: apply SNN transform to encoder outputs
if use_workspace:
    ws = workspace.compete(enc_outs) -> ctx.repr = ws.slots (B,K,D)
else:
    ctx.repr = pool(enc_outs) -> (B,T,D)
if use_htm: htm(ctx.repr) -> ctx.anomaly
if use_symbolic: reasoner(ctx.repr) -> ctx.reasoning
if use_meta: meta(ctx) -> ctx.modulators
if use_active_inference: aif(ctx.repr) -> ctx.action
out = heads(ctx.repr, ctx.reasoning, ctx.action)
assemble SystemOutput + BrainAIState
```

### The PipelinePlan Pattern

Build the plan ONCE at `__init__` from config + installed deps. Execute it in
`forward()`. No scattered conditionals.

```python
class Stage:
    name: str
    run: Callable[[PipelineContext], PipelineContext]
    enabled: bool
    bypass: Callable[[PipelineContext], PipelineContext]  # fallback when disabled
```

See **`references/pipeline-plan.md`** for the full Stage interface and context design.

### Type Contracts at Every Boundary

Every module boundary has a typed contract. The orchestrator asserts (cheaply):
- `D` matches `workspace_dim` everywhere
- Tensors are on the same device
- Masks match `T` dimension
- No silent dtype drift

| Stage | Contract Type | Shape |
|-------|--------------|-------|
| Encoders → | `EncoderOutput` | feats: `(B,T,D)`, mask: `(B,T)`, salience: `(B,T)` |
| → Workspace | `WorkspaceOutput` | slots: `(B,K,D)`, winners, attn |
| → HTM | `HTMOutput` | pred_sdr, anomaly_score, states |
| → Reasoning | `ReasoningOutput` | y_sys1, conf, y_sys2, used_sys2, trace |
| → Decision | `DecisionOutput` | action_dist, action, efe_terms |
| → Output | `SystemOutput` | output, confidence, details |

Full schemas in **`references/type-contracts.md`**.

### Feature Flags as Routing Graph

Disabling a module MUST still produce valid output with a stable schema. Every
module has a bypass adapter:

| Flag | If Disabled | Bypass |
|------|------------|--------|
| `use_workspace` | `concat_pool(encoder_feats) → (B,T,D)` | Shape-compatible |
| `use_htm` | `anomaly_score = zeros(B)`, `pred = None` | Neutral values |
| `use_symbolic` | `reasoning_trace = None`, `used_sys2 = False` | Pass-through |
| `use_meta` | `modulators = {DA:1, ACh:1, NE:1, 5HT:1}` | Fixed neutral |
| `use_engram` | Retrieval branch omitted, `None` fields | Schema preserved |
| `use_snn` | Raw encoder output (no spike transform) | Identity |

See **`references/feature-flags.md`** for bypass adapter implementations.

### State Management

Stateful modules (SNN membrane, workspace WM, HTM TM, belief state, eligibility
traces) are consolidated into one container:

```python
BrainAIState(wm_state, htm_state, snn_state, belief_state, eligibility_state, rng_state)
```

Two forward modes:
- **Stateless**: `forward(batch, state=None) → (SystemOutput, new_state)`
- **Streaming**: `step(batch_t, state) → (SystemOutput_t, new_state)`

See **`references/state-management.md`** for checkpointing and streaming patterns.

### return_details Two-Tier Design

**Always collected** (cheap, constant cost): winners, confidences, anomaly_score,
used_sys2, modulators, inference_time_ms.

**Only when requested** (return_details=True): attention maps, full reasoning trace,
HTM synapse stats, spike rasters.

```python
SystemOutput(output, confidence, modalities_used, reasoning_used, anomaly_score,
             inference_time_ms, details=SystemDetails(...) if return_details else None)
```

### Dependency Resolution

Record what was selected when optional deps are missing:

```python
brain.deps_report() → {"ncps": "missing → GRU", "htm_core": "present", ...}
```

Save into run manifests for reproducibility. See **`references/reproducibility.md`**.

### Telemetry

Structured event hooks, not ad-hoc prints:

```python
TelemetrySink: on_forward_start, on_module_start, on_module_metrics, on_module_end, on_forward_end
```

See **`references/telemetry.md`** for the interface and integration patterns.

## Anti-Patterns

- Do NOT scatter `if config.use_X` throughout `forward()`. Use `Stage.bypass` instead.
- Do NOT let modules hold hidden state. Externalize into `BrainAIState`.
- Do NOT iterate `dict.items()` on encoder outputs. Always use `sorted()` keys.
- Do NOT silently substitute defaults for missing values. Mark `UNRESOLVED` or raise.
- Do NOT mix devices/dtypes across module boundaries. The orchestrator enforces uniformity.

## Additional Resources

### Reference Files

- **`references/type-contracts.md`** — Full dataclass schemas for all typed contracts
- **`references/pipeline-plan.md`** — PipelinePlan, Stage interface, PipelineContext
- **`references/feature-flags.md`** — Routing graph, bypass adapters, fallback rules
- **`references/state-management.md`** — BrainAIState, streaming, checkpointing
- **`references/telemetry.md`** — TelemetrySink interface, W&B/TensorBoard integration
- **`references/reproducibility.md`** — Determinism rules, seed management, manifests
- **`references/testing-matrix.md`** — Parameterized tests, "done when" criteria

### Scripts

- **`scripts/validate_contracts.py`** — Runtime contract validation
- **`scripts/gen_flag_tests.py`** — Generate pairwise flag combination tests
- **`scripts/deps_report.py`** — Optional dependency audit

### Assets

- **`assets/types_template.py`** — Starter template for brain_ai/types.py
