# Type Contracts Reference

## Overview

Every module boundary in the BrainAI pipeline has a typed contract enforced by
the orchestrator. Contracts use `@dataclass` for compatibility with PyTorch's
JIT and `torch.utils.checkpoint`. All tensor fields include expected shape
annotations in comments.

## Canonical Input: ModalityBatch

```python
@dataclass
class ModalityBatch:
    """Normalized input batch consumed by all downstream modules."""
    # Core modalities (None if not provided)
    vision: Optional[Tensor] = None           # (B,C,H,W) or (B,T,C,H,W)
    vision_mask: Optional[Tensor] = None      # (B,) or (B,T) boolean
    text: Optional[Tensor] = None             # (B,L) int64 token ids
    attention_mask: Optional[Tensor] = None   # (B,L) boolean
    audio: Optional[Tensor] = None            # (B,S) raw waveform or (B,T,F) features
    audio_mask: Optional[Tensor] = None       # (B,) or (B,T)
    sensors: Optional[Tensor] = None          # (B,T,D)
    sensor_mask: Optional[Tensor] = None      # (B,T)

    # Control / RL signals (for active inference / neuromodulation)
    reward: Optional[Tensor] = None           # (B,) or (B,1)
    done: Optional[Tensor] = None             # (B,) boolean
    action: Optional[Tensor] = None           # (B,A) previous action

    # Metadata
    modalities_present: List[str] = field(default_factory=list)
    device: torch.device = torch.device("cpu")
    compute_dtype: torch.dtype = torch.float32
```

### Normalization Rules

The orchestrator converts raw user inputs into `ModalityBatch`:

1. **Device placement**: All tensors moved to `config.device`
2. **Dtype policy**: Parameters stay fp32; compute uses bf16/fp16; token ids stay int64
3. **Padding**: Variable-length sequences padded to batch max with corresponding masks
4. **Sorted modality order**: `modalities_present` always sorted alphabetically
   to avoid nondeterministic dict iteration
5. **Missing modalities**: Set to `None` with corresponding mask as `None`

## Encoder Output Contract

```python
@dataclass
class EncoderOutput:
    """Output from any modality encoder."""
    modality: str                             # "vision", "text", "audio", "sensors", "engram"
    feats: Tensor                             # (B, T, D) — always 3D even if T=1
    mask: Tensor                              # (B, T) boolean
    salience: Tensor                          # (B, T) or (B, 1) scalar weight for workspace
    aux: Optional[Dict[str, Tensor]] = None   # Encoder-specific auxiliary outputs
```

### Invariants

- `feats.shape[-1] == config.encoder.output_dim` (always matches workspace_dim after projection)
- `feats.shape[0] == mask.shape[0]` (batch dimension)
- `feats.shape[1] == mask.shape[1]` (time dimension)
- `feats.device == mask.device == salience.device` (same device)
- `salience` values are non-negative (used as competition weights)

### Ensuring T=1 for Static Inputs

Vision encoder on single images: `feats = feats.unsqueeze(1)` to make `(B,1,D)`.
This uniformity simplifies workspace competition and HTM processing.

## Workspace Output Contract

```python
@dataclass
class WorkspaceOutput:
    """Output from Global Workspace competition."""
    slots: Tensor                             # (B, K, D) — K winning slots
    slot_mask: Tensor                         # (B, K) boolean
    winners: Tensor                           # (B, K) indices into input modalities
    winner_scores: Tensor                     # (B, K) competition scores
    attn: Optional[Tensor] = None             # (B, H, K, T_total) attention maps
    broadcast: Optional[Tensor] = None        # (B, D) broadcast signal
    wm_state: Optional[Any] = None            # Working memory updated state
    modality_contributions: Optional[Dict[str, Tensor]] = None  # Per-modality scores
```

### Invariants

- `slots.shape == (B, K, D)` where K ≤ `config.workspace.capacity_limit`
- `D == config.workspace.workspace_dim`
- `winners` contains valid indices into the concatenated encoder outputs
- `winner_scores` is sorted descending (highest competition score first)

## HTM Output Contract

```python
@dataclass
class HTMOutput:
    """Output from HTM temporal layer."""
    prediction: Optional[Tensor]              # (B, D) predicted next representation
    anomaly_score: Tensor                     # (B,) in [0, 1]
    tm_state: Optional[Any] = None            # Temporal Memory state for streaming
    sp_state: Optional[Any] = None            # Spatial Pooler state
    promoted_patterns: int = 0                # Count of newly promoted reflex patterns
```

### Invariants

- `anomaly_score.shape == (B,)` and values in `[0, 1]`
- `prediction`, if present, has `shape[-1] == workspace_dim`

## Reasoning Output Contract

```python
@dataclass
class ReasoningOutput:
    """Output from Dual-Process Reasoner."""
    y_sys1: Tensor                            # (B, D) System 1 fast output
    conf_sys1: Tensor                         # (B, 1) System 1 confidence
    y_sys2: Optional[Tensor] = None           # (B, D) System 2 slow output (if triggered)
    used_sys2: bool = False                   # Whether System 2 was engaged
    output: Optional[Tensor] = None           # (B, D) final selected output
    trace: Optional[List[Tensor]] = None      # Reasoning step activations
    symbols: Optional[Dict[str, Tensor]] = None  # Extracted symbolic facts
```

### Invariants

- `y_sys1.shape[-1] == workspace_dim`
- If `used_sys2`, then `y_sys2` is not None
- `output = y_sys2 if used_sys2 else y_sys1`
- `conf_sys1` in `[0, 1]`; System 2 triggers when `conf_sys1 < confidence_threshold`

## Decision Output Contract

```python
@dataclass
class DecisionOutput:
    """Output from Active Inference decision system."""
    action_dist: Optional[Any] = None         # torch.distributions.Distribution
    action: Optional[Tensor] = None           # (B, A) selected action
    efe_terms: Optional[Dict[str, Tensor]] = None  # EFE decomposition
    belief_state: Optional[Tensor] = None     # (B, S) updated belief
    action_logits: Optional[Tensor] = None    # (B, num_actions) for discrete
```

## System Output Contract

```python
@dataclass
class SystemOutput:
    """Final output from BrainAI forward pass."""
    # Always present
    output: Tensor                            # Task-dependent shape
    confidence: Tensor                        # (B, 1)
    modalities_used: List[str]                # Which modalities were in the batch
    reasoning_used: bool                      # Whether System 2 engaged
    anomaly_score: Optional[Tensor]           # (B,) from HTM
    inference_time_ms: float                  # Wall-clock forward time

    # Present only when return_details=True
    details: Optional['SystemDetails'] = None

@dataclass
class SystemDetails:
    """Detailed introspection — only populated when return_details=True."""
    encoder: Optional[Dict[str, Dict]] = None
    workspace: Optional[Dict] = None
    htm: Optional[Dict] = None
    reasoning: Optional[Dict] = None
    decision: Optional[Dict] = None
    meta: Optional[Dict] = None
    engram: Optional[Dict] = None
```

### SystemDetails Field Contracts

Each sub-dict has a stable schema. Fields may be None but keys are always present:

```python
# encoder[modality]
{"salience": Tensor, "pooled_feats_stats": Dict, "mask_sum": int}

# workspace
{"winners": Tensor, "slot_mask": Tensor, "attn": Optional[Tensor],
 "ignition_steps": int, "modality_contributions": Dict}

# htm
{"anomaly_score": Tensor, "pred_sdr_stats": Optional[Dict],
 "promoted_patterns_count": int}

# reasoning
{"used_sys2": bool, "sys1_conf": Tensor, "trace": Optional[List],
 "symbolic_facts": Optional[Dict]}

# decision
{"efe_terms": Optional[Dict], "action_dist": Optional[Any]}

# meta
{"DA": Tensor, "ACh": Tensor, "NE": Tensor, "5HT": Tensor,
 "trace_updates": Optional[Dict]}

# engram
{"hits": int, "collision_rate_estimate": float,
 "gating_alpha_stats": Optional[Dict]}
```

## BrainAIState Contract

```python
@dataclass
class BrainAIState:
    """Consolidated state for all stateful modules."""
    wm_state: Optional[Any] = None            # Working memory (CfC/LTC/GRU)
    htm_state: Optional[Tuple] = None         # (tm_state, sp_state)
    snn_state: Optional[Dict[str, Tensor]] = None  # Membrane potentials per layer
    belief_state: Optional[Tensor] = None     # Active inference belief
    eligibility_state: Optional[Tensor] = None  # Eligibility traces
    rng_state: Optional[Dict] = None          # Per-module RNG states
    step_count: int = 0                       # Steps since last reset
```

## Contract Assertion Helpers

Cheap runtime checks (disabled in production via `torch.no_grad()` + flag):

```python
def assert_contract(output, contract_class, workspace_dim, device):
    """Verify output matches contract. Raises ContractViolation if not."""
    assert isinstance(output, contract_class)
    for field_name, expected_dim in contract_class.__shape_hints__.items():
        tensor = getattr(output, field_name)
        if tensor is not None:
            assert tensor.shape[-1] == expected_dim, f"{field_name} dim mismatch"
            assert tensor.device == device, f"{field_name} device mismatch"
```
