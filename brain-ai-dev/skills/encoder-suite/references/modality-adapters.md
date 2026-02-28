# Modality Adapters Reference

## Overview

The Modality Adapter layer is the glue between each encoder's native representation
and the shared `EncoderOutput` contract. Each encoder can have its own internal
representation (patches, subword tokens, mel frames, irregular sensor ticks). The
adapter standardizes all outputs to `workspace_dim`, producing a uniform
`(B, T, D)` tensor regardless of origin modality.

All adapters inherit from `ModalityAdapter`, override `forward`, and return a
fully populated `EncoderOutput`. Register new adapters in `ADAPTER_REGISTRY`.

## ModalityAdapter Base Interface

```python
class ModalityAdapter(nn.Module):
    """Base adapter: raw encoder output -> EncoderOutput contract."""

    def __init__(self, encoder_dim: int, workspace_dim: int, modality: str):
        super().__init__()
        self.modality = modality
        self.workspace_dim = workspace_dim
        self.projection = (
            nn.Linear(encoder_dim, workspace_dim)
            if encoder_dim != workspace_dim
            else nn.Identity()
        )

    def forward(self, raw_output, **kwargs) -> EncoderOutput:
        raise NotImplementedError
```

### Contract Guarantees

Every adapter must satisfy the following postconditions:

- `feats.shape == (B, T, workspace_dim)` -- always 3-D, even when T=1.
- `mask.shape == (B, T)` -- boolean, True for valid tokens.
- `salience.shape` is `(B, T)` or `(B, 1)` -- non-negative weights for workspace
  competition.
- `feats.device == mask.device == salience.device` -- all on the same device.
- `aux["pos_applied"]` is set to `True` or `False` to signal whether positional
  encoding has already been applied inside the adapter.

## Per-Modality Adapter Specifications

### VisionAdapter

**Input formats**

- Single image: `(B, C, H, W)`
- Video: `(B, T_frames, C, H, W)`

**Processing steps**

1. Extract spatial feature maps BEFORE the encoder's final pooling layer.
2. Single images: reshape to patch tokens `(B, N_patches, D)`, mask = all ones.
3. Video: flatten per-frame patches `(B, T_frames * N_patches, D)`, populate
   `time` with repeated frame indices.
4. Apply 2-D spatial positional encoding internally (sinusoidal or learned grid).
5. Project through `self.projection` to reach `workspace_dim`.

```python
class VisionAdapter(ModalityAdapter):
    def forward(self, raw_output, **kwargs) -> EncoderOutput:
        # raw_output: (B, N_patches, encoder_dim) from backbone
        feats = self.projection(raw_output)        # (B, N_patches, D)
        B, T, _ = feats.shape
        mask = torch.ones(B, T, dtype=torch.bool, device=feats.device)
        return EncoderOutput(
            modality=self.modality,
            feats=feats,
            mask=mask,
            salience=torch.ones(B, 1, device=feats.device),
            aux={"pos_applied": True},
        )
```

**Auxiliary fields**

- `aux["pos_applied"] = True`
- For video: `time` tensor of shape `(B, T_frames * N_patches)` with frame indices.

---

### TextAdapter

**Input format**

- Transformer hidden states: `(B, L, embed_dim)` -- intercept BEFORE the
  encoder's final pooling head.
- Attention mask: `(B, L)` boolean from the tokenizer.

**Processing steps**

1. Project hidden states: `feats = self.projection(hidden_states)` -> `(B, L, D)`.
2. Pass the tokenizer attention mask through directly as `mask`.
3. Optionally compute per-token salience from attention weights or a learned head.

```python
class TextAdapter(ModalityAdapter):
    def forward(self, raw_output, attention_mask=None, **kwargs) -> EncoderOutput:
        feats = self.projection(raw_output)        # (B, L, D)
        B, L, _ = feats.shape
        if attention_mask is None:
            attention_mask = torch.ones(B, L, dtype=torch.bool, device=feats.device)
        return EncoderOutput(
            modality=self.modality,
            feats=feats,
            mask=attention_mask,
            salience=torch.ones(B, 1, device=feats.device),
            aux={"pos_applied": True},
        )
```

**Auxiliary fields**

- `aux["pos_applied"] = True` -- transformer applies positional embeddings internally.
- `salience`: optional, derive from attention weights or a learned projection.

---

### AudioAdapter

**Input format**

- Mel features after convolutional front-end: `(B, channels, frames)` -- intercept
  BEFORE `AdaptiveAvgPool1d`.

**Processing steps**

1. Transpose to `(B, frames, channels)`.
2. Project to `(B, frames, D)`.
3. Build `mask` from original waveform lengths (mask out padding frames).
4. Optionally populate `time` as `frame_index * hop_length / sample_rate`.

```python
class AudioAdapter(ModalityAdapter):
    def __init__(self, encoder_dim, workspace_dim, modality="audio",
                 sample_rate=16000, n_fft=1024, hop_length=512, n_mels=128):
        super().__init__(encoder_dim, workspace_dim, modality)
        self.mel_params = {
            "sample_rate": sample_rate,
            "n_fft": n_fft,
            "hop_length": hop_length,
            "n_mels": n_mels,
        }

    def forward(self, raw_output, frame_lengths=None, **kwargs) -> EncoderOutput:
        # raw_output: (B, channels, frames)
        x = raw_output.transpose(1, 2)             # (B, frames, channels)
        feats = self.projection(x)                  # (B, frames, D)
        B, T, _ = feats.shape
        if frame_lengths is not None:
            mask = torch.arange(T, device=feats.device).unsqueeze(0) < frame_lengths.unsqueeze(1)
        else:
            mask = torch.ones(B, T, dtype=torch.bool, device=feats.device)
        return EncoderOutput(
            modality=self.modality,
            feats=feats,
            mask=mask,
            salience=torch.ones(B, 1, device=feats.device),
            aux={
                "pos_applied": True,
                "mel_params": self.mel_params,
                "audio_frontend": "torchaudio",  # or "torch_stft_fallback"
            },
        )
```

**Auxiliary fields**

- `aux["mel_params"]` = `{sample_rate, n_fft, hop_length, n_mels}`
- `aux["audio_frontend"]` = `"torchaudio"` or `"torch_stft_fallback"`
- `aux["pos_applied"] = True`
- `time`: optional `(B, frames)` float tensor of frame timestamps.

---

### SensorAdapter

**Input format**

- Liquid layer outputs: `(B, T_steps, hidden_dim)` -- obtain by calling the
  liquid/CfC/LTC layer with `return_sequence=True`.

**Processing steps**

1. For irregular timestamps: embed inter-step `dt` as an extra feature channel,
   concatenate with hidden states, then project to `workspace_dim`.
2. For uniform spacing: zero-fill the dt channel and project directly.
3. Build `mask` from per-sequence lengths.
4. Populate `time` from explicit timestamps or synthetic `dt * step_index`.

```python
class SensorAdapter(ModalityAdapter):
    def __init__(self, encoder_dim, workspace_dim, modality="sensors", dt_embed_dim=16):
        super().__init__(encoder_dim + dt_embed_dim, workspace_dim, modality)
        self.dt_embed = nn.Linear(1, dt_embed_dim)

    def forward(self, raw_output, seq_lengths=None, timestamps=None, **kwargs):
        B, T, _ = raw_output.shape
        if timestamps is not None:
            dt = F.pad(timestamps[:, 1:] - timestamps[:, :-1], (1, 0), value=0.0)
            dt_feats = self.dt_embed(dt.unsqueeze(-1))           # (B, T, dt_embed_dim)
            combined = torch.cat([raw_output, dt_feats], dim=-1) # (B, T, enc+dt)
        else:
            zeros = torch.zeros(B, T, self.dt_embed.in_features, device=raw_output.device)
            combined = torch.cat([raw_output, zeros], dim=-1)
        feats = self.projection(combined)                        # (B, T, D)
        if seq_lengths is not None:
            mask = torch.arange(T, device=feats.device).unsqueeze(0) < seq_lengths.unsqueeze(1)
        else:
            mask = torch.ones(B, T, dtype=torch.bool, device=feats.device)
        return EncoderOutput(
            modality=self.modality, feats=feats, mask=mask,
            salience=torch.ones(B, 1, device=feats.device),
            time=timestamps, aux={"pos_applied": True},
        )
```

**Auxiliary fields**

- `time`: `(B, T_steps)` float tensor -- real-valued timestamps or synthetic
  `dt * step_index`.
- `aux["pos_applied"] = True` -- time encoding applied via `dt_embed`.

---

### EngramAdapter

**Input format**

- Engram embeddings: `(B, L, embed_dim)` -- intercept BEFORE the engram module's
  pooling layer.

**Processing steps**

1. Project: `feats = self.projection(raw_output)` -> `(B, L, D)`.
2. Build `mask` as the n-gram validity mask (first `n-1` positions lack full
   context window and are masked out).
3. Align token positions with the text encoder's grid for workspace fusion.

```python
class EngramAdapter(ModalityAdapter):
    def __init__(self, encoder_dim, workspace_dim, modality="engram",
                 ngram_order=3):
        super().__init__(encoder_dim, workspace_dim, modality)
        self.ngram_order = ngram_order

    def forward(self, raw_output, seq_lengths=None, **kwargs) -> EncoderOutput:
        feats = self.projection(raw_output)        # (B, L, D)
        B, L, _ = feats.shape
        # N-gram validity: first (ngram_order - 1) positions lack full context
        mask = torch.ones(B, L, dtype=torch.bool, device=feats.device)
        mask[:, : self.ngram_order - 1] = False
        if seq_lengths is not None:
            length_mask = torch.arange(L, device=feats.device).unsqueeze(0) < seq_lengths.unsqueeze(1)
            mask = mask & length_mask
        return EncoderOutput(
            modality=self.modality,
            feats=feats,
            mask=mask,
            salience=torch.ones(B, 1, device=feats.device),
            aux={"pos_applied": True},
        )
```

**Auxiliary fields**

- `aux["pos_applied"] = True` -- follows the text encoder's positional scheme.

## T Semantics Policy

### Token-Axis Semantics (Recommended)

Treat T as "competition tokens": patches for vision, subwords for text, mel frames
for audio, timesteps for sensors, n-gram positions for engram. If a modality has a
natural time axis (video, sensors), optionally populate `time` aligned to tokens.

### Time-Axis Alternative

Treat T as "real time steps" for all modalities. Vision and text become T=1 unless
chunked or streamed, reducing token-level competition richness.

### Recommendation

Given the architecture (workspace competition + HTM + SNN), use **token-axis
semantics** as the default. Supply `time` only when the modality naturally involves
a time dimension.

## Positional / Time Encoding Policy

### Recommended Compromise

| Modality | Encoding Strategy | `pos_applied` |
|----------|------------------|---------------|
| Text | Keep standard transformer positional embedding internally. | `True` |
| Vision | Apply 2-D / relative spatial encoding internally (patch models). | `True` |
| Audio | Apply frame-index or relative position encoding internally. | `True` |
| Sensors | Return explicit timestamps in `time` field (float). Apply shared Time2Vec / Fourier time features projected to D inside the adapter. | `True` |
| Engram | Follow text token positions. | `True` |

All adapters set `aux["pos_applied"] = True`. The workspace must not re-apply
positional encoding when this flag is set.

## Spiking Outputs (Without Polluting Downstream)

If an encoder internally uses spiking dynamics (LIF neurons, surrogate gradients):

1. Compute spike state internally (spike raster, membrane potentials, spike counts).
2. Return through the adapter:
   - `feats`: projected continuous embedding `(B, T, D)` -- consumed by workspace.
   - `spike`: optional spike tensor for debugging and training losses.
   - `aux`: spike statistics (mean firing rate, sparsity).
3. Keep spike training tricks local; the workspace remains modality-agnostic.

```python
# Inside a spiking adapter's forward():
spike_raster = encoder.get_spike_raster()          # (B, T, D_spike) or None
feats = self.projection(continuous_output)          # (B, T, D)

spike_aux = {}
if spike_raster is not None:
    spike_aux = {
        "spike_stats": {
            "mean_rate": spike_raster.mean().item(),
            "sparsity": (spike_raster == 0).float().mean().item(),
        }
    }

return EncoderOutput(
    modality=self.modality,
    feats=feats,
    mask=mask,
    salience=salience,
    spike=spike_raster,
    aux={"pos_applied": True, **spike_aux},
)
```

Attach spike-specific losses at the encoder level. The adapter exposes
`spike_stats` through `aux` for logging; the workspace never reads `spike`.

## Adapter Registration Pattern

```python
ADAPTER_REGISTRY: Dict[str, Type[ModalityAdapter]] = {
    "vision": VisionAdapter,
    "text": TextAdapter,
    "audio": AudioAdapter,
    "sensors": SensorAdapter,
    "engram": EngramAdapter,
}
```

### Adding a New Modality

1. Subclass `ModalityAdapter`.
2. Implement `forward` returning a valid `EncoderOutput`.
3. Add the entry to `ADAPTER_REGISTRY`.
4. Ensure `aux["pos_applied"]` is set.
5. Write a test asserting all contract invariants (shape, device, dtype, mask).

### Runtime Adapter Resolution

```python
def build_adapter(modality: str, encoder_dim: int, workspace_dim: int,
                  **kwargs) -> ModalityAdapter:
    """Look up and instantiate the correct adapter for a modality."""
    if modality not in ADAPTER_REGISTRY:
        raise ValueError(f"Unknown modality '{modality}'. "
                         f"Registered: {list(ADAPTER_REGISTRY.keys())}")
    return ADAPTER_REGISTRY[modality](encoder_dim, workspace_dim,
                                      modality=modality, **kwargs)
```

Call `build_adapter` once per modality during `__init__` and store the result as a
submodule so parameters are tracked by the optimizer.
