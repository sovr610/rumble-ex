# Multi-Modal Encoder Batching Rules

## Overview

Variable-length batching is where systems usually break. Define one standardized
batching story across all encoders: pad to max T within the batch, mask marks
valid tokens, store lengths in aux for debugging, never allow ragged tensors past
the adapter. Every encoder adapter must accept `(padded, mask)` and return
`(workspace_tensor, mask)` in that same contract so downstream layers never
reason about modality-specific padding logic.

## Universal Batching Rules

1. Pad to max T within the batch.
2. `mask` marks valid tokens (`True` = valid, `False` = padding).
3. Store original lengths in `aux["lengths"]` for debugging and loss masking.
4. Never allow ragged tensors past the adapter boundary.
5. All padding uses zeros (not special tokens, not NaN, not sentinel values).
6. Return shape from every adapter: `(B, T, D_workspace)` plus `(B, T)` bool mask.
7. Treat the `pad_and_mask` utility as the single canonical implementation --
   do not rewrite padding logic per encoder.
8. Assert mask shape matches padded tensor shape at adapter exit in debug mode.
9. Propagate the mask through every layer that operates on the T dimension --
   attention, pooling, loss computation. Dropping the mask mid-pipeline is a bug.
10. When collating batches in a DataLoader, use a custom `collate_fn` that calls
    `pad_and_mask` rather than relying on PyTorch default collation (which
    requires uniform tensor shapes and will error on variable-length inputs).

## Per-Modality Batching Specifications

### Text

- T = max sequence length in batch.
- `mask` = `attention_mask` (True for real tokens, False for padding).
- If accepting raw strings: perform tokenization in preprocessing, not inside
  `forward()`. If intentionally supporting in-forward tokenization, control
  determinism and caching explicitly.
- Padding token id = 0 by convention (matching `nn.Embedding(padding_idx=0)`).
- Truncate sequences exceeding `max_seq_len` before padding, never silently
  drop tokens without recording the truncation in `aux["truncated"]`.
- When combining subword tokenizers with engram n-gram extraction, align on
  token-index space after tokenization, not on raw character offsets.
- For causal (autoregressive) tasks, apply a causal mask on top of the padding
  mask. Combine them with logical AND: `final_mask = causal_mask & padding_mask`.
- BOS/EOS tokens count as valid (mask = True). Include them in `aux["lengths"]`.

### Audio

- If waveform input varies in length:
  - **Preferred**: compute mel spectrogram per item, pad mel frames to max
    frame count in batch -- yields a stable T = frames.
  - **Alternative**: pad waveform to max samples, then compute mel frames and
    derive `frame_mask` from original sample counts.
- `mask` is based on frame count (not sample count). One frame covers
  `hop_length` samples, so `valid_frames = ceil(num_samples / hop_length)`.
- STFT must stay in fp32 even under AMP to avoid numerical issues; cast to
  compute dtype after mel feature extraction.
- `aux["mel_params"]` records `sample_rate`, `n_fft`, `hop_length`, `n_mels`
  for frame-count reproducibility.
- When batch items span very different durations (e.g., 0.5 s vs 30 s), consider
  bucketed batching to reduce wasted padding compute.
- For stereo or multi-channel audio, mix down to mono before mel extraction
  unless the encoder explicitly handles channel dimension. Record
  `aux["original_channels"]` if mixing down.
- Normalize waveform amplitude to [-1, 1] before STFT. Unnormalized waveforms
  cause scale-dependent mel magnitudes that destabilize training.

### Vision

- **Images**: fixed T = `N_patches` (or T = 1 if globally pooled), mask = all
  ones. No padding variance within a batch for same-resolution images because
  patch count is deterministic from `(H, W, patch_size)`.
- **Video**: variable frame counts require padding.
  - Option A: pad raw frames to `max_frames`, patch each frame, concatenate.
  - Option B: patch each frame independently, concatenate token sequences, pad
    the resulting token sequence.
  - `mask` for tokens corresponding to missing frames = False.
- For mixed-resolution image batches: resize to a canonical resolution before
  patching so T stays uniform and no per-item mask is needed.
- Normalization (mean/std per channel) happens before patching. Record the
  normalization constants in `aux["vision_norm"]` so inference can invert if
  needed for visualization.
- Data augmentation (random crop, flip, color jitter) must occur before the
  adapter boundary. The adapter receives deterministic, normalized patches only.

### Sensors

- Options for irregular time series:
  1. **Resample to fixed dt** (simplifies T) -- recommended for training.
  2. Bin into fixed windows and embed aggregates.
  3. Keep native timestamps but still pad sequences to max length.
- `mask` comes from per-item length in timesteps or bins.
- When using CfC/LTC: pass `dt` explicitly to the cell. Do not assume uniform
  sampling -- the cell needs real inter-sample intervals to integrate correctly.
- For multi-sensor fusion where channels arrive at different rates, resample
  each channel to a common rate before stacking into a single tensor.
- Store `aux["sensor_dt"]` with the actual time step used, so downstream
  temporal models (HTM, CfC) can condition on it.
- If sensor readings include NaN (from dropped packets or faulty sensors),
  replace NaN with zero and mark those timesteps as False in the mask before
  reaching the adapter.

### Engram

- T aligns to text token positions or n-gram positions.
- `mask` = valid n-gram extraction positions (positions where the full n-gram
  window exists).
- Pad to max L in batch.
- For n-gram order k: the first (k - 1) positions may be invalid if no left
  context is available. Set those mask entries to False.
- Hash collisions in the engram table do not affect padding -- they affect
  embedding quality. Keep collision tracking in `aux["engram_collisions"]`
  separate from length tracking in `aux["lengths"]`.
- When engram operates alongside the text encoder on the same input, ensure
  both produce the same T after padding so workspace competition can align
  their representations token-by-token.

## `pad_and_mask` Utility

```python
def pad_and_mask(
    sequences: List[Tensor],
    pad_value: float = 0.0,
    max_len: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """Pad variable-length sequences and create mask.

    Args:
        sequences: list of (T_i, D) tensors
        pad_value: value for padding positions
        max_len: optional max length cap (default: max in batch)

    Returns:
        padded: (B, T_max, D) tensor
        mask: (B, T_max) bool tensor, True=valid
    """
    B = len(sequences)
    D = sequences[0].shape[-1]
    T_max = max_len or max(s.shape[0] for s in sequences)
    device = sequences[0].device
    dtype = sequences[0].dtype

    padded = torch.full((B, T_max, D), pad_value, device=device, dtype=dtype)
    mask = torch.zeros(B, T_max, device=device, dtype=torch.bool)

    for i, seq in enumerate(sequences):
        T_i = min(seq.shape[0], T_max)
        padded[i, :T_i] = seq[:T_i]
        mask[i, :T_i] = True

    return padded, mask
```

Use this function at the boundary between raw feature extraction and the adapter
projection. Do not scatter equivalent logic across individual encoder files.

When `max_len` is provided and a sequence exceeds it, the sequence is silently
truncated. Log or record the truncation in `aux` so that downstream analysis
can detect information loss.

For 1-D inputs (e.g., scalar time series with shape `(T_i,)`), unsqueeze to
`(T_i, 1)` before passing to `pad_and_mask`. The utility expects a feature
dimension D >= 1.

### Integration with DataLoader

```python
def collate_fn(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """Custom collation that uses pad_and_mask for variable-length fields."""
    sequences = [item["features"] for item in batch]
    padded, mask = pad_and_mask(sequences)
    lengths = torch.tensor([s.shape[0] for s in sequences], dtype=torch.long)
    return {"features": padded, "mask": mask, "lengths": lengths}
```

Pass this `collate_fn` to `torch.utils.data.DataLoader`. Do not rely on default
collation for any modality that can produce variable-length sequences.

## Half-Precision Safety

| Operation             | Safe dtype       | Notes                                        |
|-----------------------|------------------|----------------------------------------------|
| STFT                  | fp32 only        | Complex arithmetic unstable in fp16/bf16     |
| Mel filterbank multiply | fp32 or bf16  | After STFT, safe to cast                     |
| Embedding lookup      | any              | Integer indices, output inherits param dtype |
| Conv2d / Conv1d       | bf16 safe        | Under AMP autocast                           |
| Linear projection     | bf16 safe        | Under AMP autocast                           |
| LayerNorm             | fp32 accumulation| PyTorch handles internally                   |
| Softmax (for salience)| fp32 recommended | Numerical stability                          |

Pattern for safe audio frontend:

```python
with torch.amp.autocast(device_type='cuda', enabled=False):
    mel = self.mel_frontend(waveform.float())  # Force fp32 for STFT
mel = mel.to(compute_dtype)  # Cast back to compute dtype
```

Apply the same `autocast(enabled=False)` guard to any operation listed as
"fp32 only" in the table above. Never rely on AMP to silently keep an operation
in fp32 -- guard it explicitly.

When mixing modalities in a single forward pass under AMP, each encoder may
need its own autocast context. The system-level forward in `BrainAI.forward()`
should not wrap the entire call in one autocast block if individual encoders
need fp32 segments internally. Instead, let each encoder manage its own
precision boundaries and enter AMP only for the workspace and downstream layers.

## Deterministic Preprocessing

- Same input must produce same tokenization/features for a fixed config and seed.
- Caching must not change results. A cached result and a freshly computed result
  must be bit-identical for the same input and config.
- Record preprocessing params in `aux` for reproducibility audit:
  - Text: tokenizer name/version, vocab size, max_seq_len.
  - Audio: sample_rate, n_fft, hop_length, n_mels, window function.
  - Vision: resolution, patch_size, normalization mean/std.
  - Sensors: resampling rate, bin width, interpolation method.
  - Engram: n-gram order, hash table size, hash function id.
- When adding a new modality encoder, include a determinism unit test that runs
  the same input twice and asserts bitwise equality of the output tensor and mask.
- Set `torch.use_deterministic_algorithms(True)` in test environments to catch
  non-deterministic operations early. Some operations (e.g., scatter_add on CUDA)
  are non-deterministic by default and need explicit deterministic alternatives.
- Pin random seeds for any stochastic preprocessing (e.g., SpecAugment for audio,
  random crop for vision) when reproducibility is required. Store the seed in
  `aux["preprocess_seed"]`.

## Common Failure Modes

- **Ragged tensors leaking past the adapter**: causes shape mismatches in the
  workspace competition layer. The adapter boundary is the firewall -- enforce
  the `(B, T, D)` + `(B, T)` contract there.
- **Mask dtype wrong**: downstream attention layers expect `bool` masks. Passing
  `float` or `int` masks can silently produce wrong attention scores.
- **Padding with NaN**: propagates NaN through every downstream matmul. Always
  pad with zeros.
- **Forgetting to mask the loss**: padding tokens contribute zero-information
  gradient if not masked out of the loss function. Use `aux["lengths"]` or
  the returned mask to build the loss mask.
- **Mixed devices**: `pad_and_mask` inherits device from `sequences[0]`. Ensure
  all items in the list reside on the same device before calling.
- **Empty sequences**: a sequence with T = 0 produces an all-False mask row.
  Downstream layers must handle all-False mask rows gracefully (e.g., attention
  should not divide by zero when computing weighted averages over zero valid
  tokens). Guard against this with a minimum sequence length of 1 or by
  filtering empty sequences before batching.
- **Batch size of 1**: `pad_and_mask` still works correctly, but T_max equals
  the single item's length and no actual padding occurs. Verify that the mask
  is still generated (all True) so downstream code does not special-case B = 1.
- **Gradient through padding**: zero-padded positions produce zero gradients
  naturally for additive operations, but multiplicative or normalization
  operations can still be affected. Always apply the mask before any operation
  that aggregates across the T dimension.
