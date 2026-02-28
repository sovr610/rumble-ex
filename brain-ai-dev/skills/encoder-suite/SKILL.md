---
name: BrainAI Multi-Modal Encoder Suite
description: >
  This skill should be used when the user asks to "add a new encoder",
  "add a new modality", "create an encoder adapter", "fix encoder output shapes",
  "standardize encoder outputs", "make encoders return (B,T,D)", "add vision encoder",
  "add audio encoder", "add sensor encoder", "add text encoder", "add engram encoder",
  "fix encoder contract", "validate encoder outputs", "write encoder tests",
  "check encoder dependencies", "add spiking output to encoder", "fix variable-length
  batching", "fix encoder mask", "add salience to encoder", "debug encoder shapes",
  or mentions EncoderOutput, ModalityAdapter, workspace_dim, encoder contract,
  modality adapter, or encoder fallback in the BrainAI cognitive architecture.
version: 0.1.0
---

# BrainAI Multi-Modal Encoder Suite

## Purpose

Enforce the guarantee that every modality encoder outputs identically-shaped tokens
to the workspace. The workspace, HTM, reasoning, and all downstream modules never
see modality-specific types — only `EncoderOutput(feats=(B,T,D), mask=(B,T), ...)`.

This skill does NOT design encoder internals (that is modality-specific ML).
It standardizes the boundary between encoders and the rest of the pipeline.

## Key Files

| File | Role |
|------|------|
| `brain_ai/encoders/schema.py` | `EncoderOutput` dataclass — single source of truth |
| `brain_ai/encoders/adapters.py` | `ModalityAdapter` base + per-modality adapters |
| `brain_ai/encoders/__init__.py` | Public API, factory functions |
| `brain_ai/encoders/vision.py` | Vision encoder (spiking CNN, ViT, event) |
| `brain_ai/encoders/text.py` | Text encoder (transformer, spike, character) |
| `brain_ai/encoders/audio.py` | Audio encoder (mel + spiking 1D conv) |
| `brain_ai/encoders/sensors.py` | Sensor encoder (CfC/LTC/GRU) |
| `brain_ai/encoders/engram_encoder.py` | Engram n-gram hash encoder |

## The Shared Contract

Every encoder + adapter pair must produce:

```python
@dataclass
class EncoderOutput:
    modality: str           # "vision" | "text" | "audio" | "sensors" | "engram"
    feats: Tensor           # (B, T, D) float — always 3D, even T=1
    mask: Tensor            # (B, T) bool — True=valid, False=pad
    salience: Optional[Tensor] = None  # (B, T) or (B, 1) — workspace competition weight
    time: Optional[Tensor] = None     # (B, T) float — optional timestamps
    spike: Optional[Tensor] = None    # (B, T, *) — optional spike domain
    aux: dict               # modality-specific diagnostics
```

Hard invariants the orchestrator asserts cheaply:

| Rule | Check |
|------|-------|
| 3D feats | `feats.ndim == 3` |
| 2D mask | `mask.ndim == 2` |
| Aligned dims | `feats.shape[:2] == mask.shape` |
| Correct D | `feats.shape[-1] == workspace_dim` |
| Same device | `feats.device == mask.device == target_device` |
| Valid dtype | `feats.dtype in {fp32, bf16, fp16}` |
| Bool mask | `mask.dtype == torch.bool` |
| Non-negative salience | `(salience >= 0).all()` if present |

See **`references/encoder-contract.md`** for the full schema and assertion helpers.

## The Adapter Pattern

Each encoder has its own internal representation. The adapter standardizes it:

| Modality | Internal Shape | Adapter Produces |
|----------|---------------|-----------------|
| Vision | `(B, C, H', W')` spatial features | `(B, N_patches, D)` + ones mask |
| Text | `(B, L, embed_dim)` hidden states | `(B, L, D)` + attention_mask |
| Audio | `(B, channels, frames)` conv features | `(B, frames, D)` + frame_mask |
| Sensors | `(B, T, hidden_dim)` sequence output | `(B, T, D)` + length_mask |
| Engram | `(B, L, embed_dim)` embeddings | `(B, L, D)` + n-gram_mask |

Key point: the workspace does a single `torch.cat([out.feats for out in outs], dim=1)`
with no modality-specific branches.

See **`references/modality-adapters.md`** for the full adapter interface and per-modality
specs including spiking output handling.

## T Semantics

T means "competition tokens" (token-axis semantics):
- Vision: patches
- Text: subword tokens
- Audio: mel frames
- Sensors: timesteps
- Engram: n-gram positions

Static inputs use T=1 (unsqueeze). Time metadata returned in `time` field for
modalities with natural temporal axes.

## Variable-Length Batching

Universal rules:
1. Pad to max T within batch
2. `mask` marks valid tokens (`True`=valid)
3. Store lengths in `aux["lengths"]`
4. Never allow ragged tensors past the adapter
5. STFT stays fp32 even under AMP

See **`references/batching-rules.md`** for per-modality padding, half-precision
safety table, and the `pad_and_mask` utility.

## Dependency Fallbacks

Every optional dep degrades to a working baseline with identical output shapes:

| Dependency | Fallback | aux Key |
|-----------|----------|---------|
| torchaudio | `torch.stft` + manual mel | `audio_frontend` |
| ncps | Built-in CfC/LTC | `sensor_backend` |
| transformers | `nn.TransformerEncoder` | `text_backend` |
| tonic | Simple event binning | `event_frontend` |
| snntorch | Built-in LIFNeuron | `snn_backend` |

Fallback selections are recorded in `aux` and in the dependency report.

See **`references/dependency-fallbacks.md`** for fallback implementations and
monkeypatch testing patterns.

## Common Failure Modes to Prevent

- "Vision returns `(B,D)` but text returns `(B,T,D)`" — enforce T=1 rule + adapters
- "Mask uses 0/1 int64 sometimes" — enforce bool dtype
- "Audio fallback changes frame count" — use mask, never assume T
- "Half precision breaks STFT" — keep STFT in fp32, cast after
- "Engram aligns to bytes but text to tokens" — enforce token-level alignment

## Anti-Patterns

- Do NOT pool to `(B, D)` inside the encoder. Return per-token features `(B, T, D)`.
- Do NOT use int masks. Always `torch.bool`.
- Do NOT assume T is fixed across batches. Use mask.
- Do NOT branch on modality name in the workspace. Use the contract.
- Do NOT skip the adapter for "simple" modalities. Every modality goes through an adapter.

## Additional Resources

### Reference Files

- **`references/encoder-contract.md`** — Full EncoderOutput schema, invariants, aux conventions
- **`references/modality-adapters.md`** — Adapter interface, per-modality specs, spiking outputs
- **`references/batching-rules.md`** — Padding, masking, half-precision safety, T semantics
- **`references/dependency-fallbacks.md`** — Fallback matrix, testing patterns, dep report
- **`references/testing-matrix.md`** — Test categories, pytest examples, "done when" checklist

### Scripts

- **`scripts/validate_encoders.py`** — Runtime contract validation for all encoders
- **`scripts/gen_encoder_tests.py`** — Generate parameterized pytest test suite
- **`scripts/encoder_deps_report.py`** — Encoder-specific dependency audit

### Assets

- **`assets/adapter_template.py`** — Starter template for `brain_ai/encoders/adapters.py`
- **`assets/encoder_output_schema.py`** — Starter template for `brain_ai/encoders/schema.py`
