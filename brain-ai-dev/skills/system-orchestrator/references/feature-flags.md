# Feature Flag Routing

## Principle

Feature flags are NOT "if statements everywhere." They define a routing graph with
guaranteed fallbacks. Disabling a module MUST still produce valid output with a
stable schema. No downstream module may crash or produce different-shaped output
because an upstream flag changed.

## The Routing Graph

```
use_snn ─────────┐
                  v
encoders ──> [SNN transform?] ──> encoder_outputs
                                       │
use_workspace ───────────────┐         │
                             v         v
                        [workspace.compete] ──> repr (B,K,D)
                             │                     │
                        OR bypass: pool ──> repr (B,T,D)
                                               │
           ┌───────────────┬──────────────────┤
           v               v                  v
use_htm: [HTM]    use_symbolic: [reason]  use_meta: [neuromod]
    │                  │                      │
    v                  v                      v
anomaly_score    reasoning_out          modulators
           │               │                  │
           └───────────────┴──────────────────┘
                              │
                              v
                    use_active_inference: [AIF]
                              │
                              v
                         output_heads
```

## Bypass Adapter Specifications

### use_snn = False

**Effect**: Encoder outputs pass through without spike-based transform.

```python
def snn_bypass(ctx):
    # Identity: encoder outputs already in (B, T, D) format
    return ctx
```

**Contract preserved**: `EncoderOutput` shape unchanged. Downstream workspace
receives the same `(B, T, D)` feats regardless of SNN flag.

### use_workspace = False

**Effect**: No competition. All encoder features are pooled.

```python
def workspace_bypass(ctx):
    all_feats = [eo.feats for eo in ctx.encoder_outputs.values()]
    if not all_feats:
        ctx.repr = torch.zeros(B, 1, workspace_dim, device=ctx.device)
        return ctx

    # Concatenate along time dimension
    concat = torch.cat(all_feats, dim=1)  # (B, T_total, D_enc)

    # Project to workspace_dim if needed
    if D_enc != workspace_dim:
        concat = fallback_proj(concat)

    ctx.repr = concat  # (B, T_total, workspace_dim)

    # Provide neutral workspace details
    ctx.details["workspace"] = {
        "winners": None,
        "slot_mask": None,
        "attn": None,
        "ignition_steps": 0,
        "modality_contributions": {},
    }
    return ctx
```

**Contract preserved**: `ctx.repr` is always `(B, *, workspace_dim)`. HTM,
reasoning, and decision stages consume `ctx.repr` regardless of source.

### use_htm = False

**Effect**: No temporal prediction or anomaly detection.

```python
def htm_bypass(ctx):
    B = ctx.repr.shape[0]
    ctx.anomaly_score = torch.zeros(B, device=ctx.device)
    ctx.details["htm"] = {
        "anomaly_score": ctx.anomaly_score,
        "pred_sdr_stats": None,
        "promoted_patterns_count": 0,
    }
    return ctx
```

**Contract preserved**: `anomaly_score` is `(B,)` float tensor, always present.
Neuromodulation stage (which uses anomaly as input) gets a neutral zero signal.

### use_symbolic = False

**Effect**: No System 2 reasoning. System 1 pass-through.

```python
def reasoning_bypass(ctx):
    ctx.reasoning = ReasoningOutput(
        y_sys1=ctx.repr.mean(dim=1),  # Pool to (B, D)
        conf_sys1=torch.ones(B, 1, device=ctx.device),
        y_sys2=None,
        used_sys2=False,
        output=ctx.repr.mean(dim=1),
        trace=None,
        symbols=None,
    )
    ctx.details["reasoning"] = {
        "used_sys2": False,
        "sys1_conf": ctx.reasoning.conf_sys1,
        "trace": None,
        "symbolic_facts": None,
    }
    return ctx
```

**Contract preserved**: `ReasoningOutput` has all fields. `used_sys2=False`.
Output heads receive `reasoning.output` which is always `(B, D)`.

### use_meta = False

**Effect**: Fixed neutral neuromodulator values.

```python
NEUTRAL_MODULATORS = {"DA": 1.0, "ACh": 1.0, "NE": 1.0, "5HT": 1.0}

def meta_bypass(ctx):
    B = ctx.repr.shape[0]
    ctx.modulators = {
        name: torch.full((B, 1), val, device=ctx.device)
        for name, val in NEUTRAL_MODULATORS.items()
    }
    ctx.details["meta"] = {
        **{k: v for k, v in ctx.modulators.items()},
        "trace_updates": None,
    }
    return ctx
```

**Contract preserved**: `modulators` dict always has DA/ACh/NE/5HT keys with
`(B, 1)` tensors. Value of 1.0 means "no modulation" (multiplicative identity).

### use_engram = False

**Effect**: Engram encoder not added to competition set. Retrieval omitted.

```python
def engram_bypass(ctx):
    # Simply don't add engram to encoder_outputs
    ctx.details["engram"] = {
        "hits": 0,
        "collision_rate_estimate": 0.0,
        "gating_alpha_stats": None,
    }
    return ctx
```

**Contract preserved**: `SystemDetails.engram` dict has stable keys.

## Verifying Bypass Correctness

For each flag, the test suite verifies:

1. **Schema stability**: Output fields are identical (same keys, same types)
2. **Shape stability**: Tensor shapes don't change based on flags
3. **Device stability**: No CPU tensors appearing in CUDA pipeline
4. **Gradient flow**: Bypasses don't break backprop (use `detach()` only where intended)

## Interaction Matrix

Some flag combinations require special handling:

| Flag A | Flag B | Interaction |
|--------|--------|-------------|
| `use_workspace=F` | `use_htm=T` | HTM receives pooled features instead of slots |
| `use_symbolic=F` | `use_meta=T` | Meta gets no confidence signal; use anomaly only |
| `use_snn=T` | `use_workspace=T` | SNN output feeds workspace competition |
| `use_engram=T` | `use_workspace=F` | Engram features pooled with others, no competition |

The PipelinePlan handles these via stage ordering and the context object.
Each stage reads from `ctx` whatever its upstream set, without knowing
which specific module populated it.
