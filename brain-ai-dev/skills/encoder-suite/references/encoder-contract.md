# Encoder Contract Reference

## Overview

Every modality encoder in the brain-ai system must emit an `EncoderOutput` dataclass,
ensuring the invariant **"any modality in, same-shaped tokens out"**. This contract
is the single mechanism that allows the global workspace, HTM temporal layer, and
downstream reasoning modules to consume encoder outputs without any modality-specific
branching. Enforce this contract at encoder exit so violations surface immediately
during development rather than propagating silent shape bugs into the workspace.

---

## EncoderOutput Dataclass

Define all encoder outputs using this canonical dataclass. Every field beyond `modality`,
`feats`, and `mask` is optional, but when present it must obey the shape rules below.

```python
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import torch
from torch import Tensor


@dataclass
class EncoderOutput:
    modality: str                             # "vision", "text", "audio", "sensors", "engram"
    feats: Tensor                             # (B, T, D) float -- always 3D even if T=1
    mask: Tensor                              # (B, T) bool -- True=valid, False=padding
    salience: Optional[Tensor] = None         # (B, T) or (B, 1) competition weight
    pos_ids: Optional[Tensor] = None          # (B, T) int64 if needed
    time: Optional[Tensor] = None             # (B, T) float32 timestamps
    spike: Optional[Tensor] = None            # (B, T, *) optional spike-domain tensor
    aux: Dict[str, Any] = field(default_factory=dict)
```

### Field Descriptions

| Field | Required | Purpose |
|---|---|---|
| `modality` | Yes | String tag identifying the source modality. Used for logging, workspace routing, and salience weighting. |
| `feats` | Yes | The primary feature tensor. Always rank-3: batch, time/token, embedding dimension. Embedding dimension must equal `workspace_dim` from config. |
| `mask` | Yes | Boolean attention mask. `True` marks a valid token; `False` marks padding. Shape must match `feats[:2]`. |
| `salience` | No | Non-negative scalar or per-token weight that the global workspace uses during competition. When absent, the workspace assigns uniform salience. |
| `pos_ids` | No | Integer position identifiers. Supply when the downstream consumer needs explicit positional indices (e.g., for rotary embeddings applied after encoding). |
| `time` | No | Float timestamps in seconds. Required by the HTM layer when temporal ordering matters (audio, sensor streams). |
| `spike` | No | Spike-domain representation for the SNN core. Leading dimensions must be `(B, T)`, with trailing dimensions free (e.g., `(B, T, N_neurons)`). |
| `aux` | No | Modality-specific diagnostics and metadata. Never consumed by the forward path; used only for logging, debugging, and metric collection. |

---

## Hard Invariants

The orchestrator (`BrainAI.forward`) asserts these invariants cheaply on every forward
pass when contracts are enabled. Each invariant exists to prevent a specific class of
silent failure.

### 1. `feats.ndim == 3`

Guarantee a uniform `(B, T, D)` layout. A 2D tensor would indicate a missing time
dimension; a 4D tensor would indicate unreduced spatial dims (e.g., a raw CNN feature
map). Both break downstream `torch.cat` over the time axis.

### 2. `mask.ndim == 2`

The mask is `(B, T)` -- one boolean per token per batch element. A 3D mask suggests
per-head or per-feature masking, which belongs inside the encoder, not at the contract
boundary.

### 3. `feats.shape[:2] == mask.shape`

Batch and time dimensions of `feats` and `mask` must agree exactly. A mismatch means
the encoder produced tokens that have no corresponding validity flag, or vice versa.

### 4. `feats.shape[2] == workspace_dim`

The last dimension of `feats` must equal the configured `workspace_dim` (typically 4096
at production scale). This is the critical integration point: every encoder projects into
the same embedding space so the workspace can treat all modalities uniformly.

### 5. `feats.device == mask.device == target_device`

All tensors must reside on the same device. Mixed-device tensors cause silent copies or
hard crashes during `torch.cat`. Validate against the target device passed from the
orchestrator.

### 6. `feats.dtype in {torch.float32, torch.bfloat16, torch.float16}`

Only floating-point dtypes are valid for features. The specific dtype is controlled by
the AMP configuration. Integer or complex dtypes indicate a bug in the projection head.

### 7. `mask.dtype is torch.bool`

The mask must be boolean, not integer. PyTorch attention functions interpret `int` masks
differently from `bool` masks (additive vs. multiplicative), so a dtype mismatch causes
wrong attention scores without raising an error.

### 8. Salience values are non-negative (if present)

Salience represents competition weight for the global workspace. Negative values are
nonsensical and would invert the winner-take-all dynamics. Assert `(salience >= 0).all()`
when the field is not `None`.

### 9. Spike batch and time dims match feats (if present)

When `spike` is provided, `spike.shape[0]` must equal `B` and `spike.shape[1]` must
equal `T`. Trailing dimensions are unconstrained (they depend on SNN architecture), but
the leading two axes must align with `feats` so the SNN core can pair spikes with their
corresponding workspace tokens.

---

## T=1 Rule for Static Inputs

When an encoder processes a static input (e.g., a single image, a single sensor
snapshot), unsqueeze the output to `(B, 1, D)` rather than emitting `(B, D)`. This
applies to:

- **Vision encoder** on single frames: pool spatially, then unsqueeze `dim=1`.
- **Sensor encoder** on single-timestep readings: unsqueeze `dim=1`.
- **Engram encoder** on a single memory retrieval: unsqueeze `dim=1`.

The `T=1` convention is non-negotiable. It keeps the workspace concatenation path
branch-free and allows the HTM layer to treat every modality as a sequence (even a
length-1 sequence). Omitting the unsqueeze forces the workspace to add a special case
for 2D tensors, which defeats the purpose of the contract.

```python
# Correct -- vision encoder final step
pooled = spatial_pool(feature_map)          # (B, D)
feats = pooled.unsqueeze(1)                 # (B, 1, D)
mask = torch.ones(B, 1, dtype=torch.bool, device=feats.device)
```

---

## Contract Assertion Helper

Call `assert_encoder_output` at the end of every encoder's `forward` method. Guard
the assertions behind a global flag so they can be disabled in production inference
for zero overhead.

```python
_CONTRACTS_ENABLED = True   # Set False in production inference for speed


class ContractViolation(AssertionError):
    """Raised when an EncoderOutput violates the shared contract."""
    pass


def assert_encoder_output(
    output: EncoderOutput,
    workspace_dim: int,
    device: torch.device,
) -> None:
    """Validate EncoderOutput contract. Raise ContractViolation if violated.

    Call at the end of every encoder forward pass. Guarded by
    _CONTRACTS_ENABLED so production inference pays zero cost.

    Args:
        output: The EncoderOutput to validate.
        workspace_dim: Expected embedding dimension from BrainAIConfig.
        device: Expected device from the orchestrator.
    """
    if not _CONTRACTS_ENABLED:
        return

    # --- shape invariants ---
    assert output.feats.ndim == 3, (
        f"feats must be 3D (B, T, D), got {output.feats.ndim}D"
    )
    assert output.mask.ndim == 2, (
        f"mask must be 2D (B, T), got {output.mask.ndim}D"
    )
    assert output.feats.shape[:2] == output.mask.shape, (
        f"feats/mask batch/time mismatch: "
        f"feats {output.feats.shape[:2]} vs mask {output.mask.shape}"
    )
    assert output.feats.shape[-1] == workspace_dim, (
        f"feats dim {output.feats.shape[-1]} != workspace_dim {workspace_dim}"
    )

    # --- device invariant ---
    assert output.feats.device == device, (
        f"feats on {output.feats.device}, expected {device}"
    )
    assert output.mask.device == device, (
        f"mask on {output.mask.device}, expected {device}"
    )

    # --- dtype invariants ---
    assert output.feats.dtype in (torch.float32, torch.bfloat16, torch.float16), (
        f"feats dtype {output.feats.dtype} not in {{fp32, bf16, fp16}}"
    )
    assert output.mask.dtype == torch.bool, (
        f"mask dtype {output.mask.dtype}, expected bool"
    )

    # --- optional field invariants ---
    if output.salience is not None:
        assert (output.salience >= 0).all(), (
            "salience must be non-negative"
        )

    if output.spike is not None:
        assert output.spike.shape[0] == output.feats.shape[0], (
            f"spike batch dim {output.spike.shape[0]} != "
            f"feats batch dim {output.feats.shape[0]}"
        )
        assert output.spike.shape[1] == output.feats.shape[1], (
            f"spike time dim {output.spike.shape[1]} != "
            f"feats time dim {output.feats.shape[1]}"
        )
```

### Disabling Contracts at Runtime

Toggle contracts off for production inference to avoid per-batch assertion overhead:

```python
import brain_ai.encoders.contract as contract_mod
contract_mod._CONTRACTS_ENABLED = False
```

Alternatively, gate on an environment variable:

```python
import os
_CONTRACTS_ENABLED = os.environ.get("BRAIN_AI_CONTRACTS", "1") == "1"
```

---

## aux Dict Conventions

The `aux` field carries modality-specific diagnostics that never enter the forward
computation path. Populate it for logging, debugging, and metric dashboards. Use the
standard keys below for consistency across encoders.

### Standard Keys

| Key | Type | Populated By | Description |
|---|---|---|---|
| `"pos_applied"` | `bool` | All encoders | Whether positional encoding has already been added into `feats`. If `True`, the workspace must not re-apply positional encoding. |
| `"audio_frontend"` | `str` | Audio encoder | `"torchaudio"` when torchaudio is available, `"torch_stft_fallback"` when using the manual STFT path. Useful for diagnosing spectrogram differences. |
| `"tokenization_stats"` | `dict` | Text encoder | Contains `"token_count"` (int) and `"padding_ratio"` (float). Helps monitor tokenizer efficiency and sequence packing. |
| `"event_binning_stats"` | `dict` | Event vision encoder | Contains `"num_events"`, `"bin_width_ms"`, `"bins_used"`. Tracks how event camera data was discretized into time bins. |
| `"mel_params"` | `dict` | Audio encoder | Contains `"sample_rate"`, `"n_fft"`, `"hop_length"`, `"n_mels"`. Records the exact mel spectrogram parameters used, critical for reproducibility. |
| `"spike_stats"` | `dict` | SNN-backed encoders | Contains `"mean_firing_rate"` (float) and `"sparsity"` (float). Monitors SNN health -- firing rates that are too high or too low indicate training issues. |

### Guidelines for aux Usage

- Never read `aux` values in the forward path. They exist purely for observability.
- Keep values JSON-serializable (no tensors, no callables) so they can be logged to
  TensorBoard, W&B, or plain JSON files without conversion.
- Prefix custom keys with the modality name to avoid collisions
  (e.g., `"vision_crop_scale"`, `"audio_vad_active"`).
- Do not store large arrays. Summarize statistics instead (means, counts, ratios).

---

## Concatenation Guarantee

The contract exists to make modality-agnostic concatenation trivially correct. The
global workspace must be able to merge outputs from an arbitrary set of active encoders
with no modality-specific branches:

```python
# Inside GlobalWorkspace.forward
sorted_outs = sorted(encoder_outputs, key=lambda o: o.modality)

all_feats = torch.cat([out.feats for out in sorted_outs], dim=1)   # (B, T_total, D)
all_masks = torch.cat([out.mask for out in sorted_outs], dim=1)    # (B, T_total)

if any(out.salience is not None for out in sorted_outs):
    all_salience = torch.cat(
        [out.salience if out.salience is not None
         else torch.ones(out.feats.shape[:2], device=out.feats.device)
         for out in sorted_outs],
        dim=1,
    )  # (B, T_total)
else:
    all_salience = None
```

This works because:

1. **Dimension 0 (batch)** is identical across all encoders -- they process the same
   batch.
2. **Dimension 1 (time/tokens)** varies per modality but concatenation along this axis
   is valid because every token lives in the same `D`-dimensional embedding space.
3. **Dimension 2 (embedding)** is fixed at `workspace_dim` by invariant 4, so the
   concatenated tensor is a well-formed `(B, T_total, D)` sequence.
4. **Mask alignment** is guaranteed by invariant 3, so `all_masks[b, t]` always
   corresponds to `all_feats[b, t]`.

No `if modality == "vision"` branches. No reshaping. No special cases. The contract is
what makes this possible.

---

## Summary Checklist

Use this checklist when implementing or reviewing an encoder:

- [ ] Return an `EncoderOutput` dataclass, not a raw tensor.
- [ ] Project final features to `workspace_dim` (the config value, typically 4096).
- [ ] Ensure `feats` is always 3D. Apply `.unsqueeze(1)` for static/single-token inputs.
- [ ] Produce a `bool` mask with `True` for valid tokens, `False` for padding.
- [ ] Place all tensors on the correct device before returning.
- [ ] Set `salience` to a non-negative tensor or leave it as `None`.
- [ ] Populate `aux["pos_applied"]` to signal whether positional encoding is baked in.
- [ ] Call `assert_encoder_output(output, workspace_dim, device)` as the last line of `forward`.
- [ ] Run the smoke test to confirm shapes propagate through the workspace correctly.
