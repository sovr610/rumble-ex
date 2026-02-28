# Encoder Test Matrix

## Overview

Tests must fail loudly the moment a modality breaks the contract. This reference
defines the minimum test matrix and "done when" criteria for every encoder in the
suite. Treat each category as a hard gate: an encoder that skips a category is not
shippable.

---

## Test Category A: Shape/Device/Dtype Invariants

Run each encoder on CPU and CUDA (skip CUDA if not available). Test fp32 and
mixed-precision modes. Every encoder must satisfy the `EncoderOutput` contract
regardless of device or dtype.

```python
MODALITIES = ["vision", "text", "audio", "sensors", "engram"]
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
DTYPES = [torch.float32, torch.bfloat16]

@pytest.mark.parametrize("modality", MODALITIES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_encoder_contract(modality, device, dtype):
    """Every encoder must produce (B,T,D) feats and (B,T) mask."""
    encoder = create_encoder(modality, device=device, dtype=dtype)
    batch = make_sample_batch(modality, B=2, device=device)
    output = encoder(batch)

    assert isinstance(output, EncoderOutput)
    assert output.feats.ndim == 3, f"feats must be 3D, got {output.feats.ndim}"
    assert output.mask.ndim == 2, f"mask must be 2D, got {output.mask.ndim}"
    assert output.feats.shape[:2] == output.mask.shape
    assert output.feats.shape[-1] == WORKSPACE_DIM
    assert output.feats.device.type == device
    assert output.mask.dtype == torch.bool
    assert not torch.isnan(output.feats).any()
    assert not torch.isinf(output.feats).any()
```

### Key assertions

- `feats` is always rank-3: `(batch, tokens, dim)`.
- `mask` is always rank-2: `(batch, tokens)`, dtype `torch.bool`.
- The last dim of `feats` equals `WORKSPACE_DIM` (4096 at production scale).
- No NaN or Inf values appear under normal inputs.
- Device placement matches the requested device exactly.

---

## Test Category B: Variable-Length Handling

Verify that each modality handles heterogeneous sequence lengths within a single
batch and produces correctly aligned masks.

```python
def test_text_variable_lengths():
    """Different text lengths must produce aligned masks."""
    lengths = [10, 25, 5]
    # Create batch with different lengths, padded to max
    # Verify mask[i, :lengths[i]] == True
    # Verify mask[i, lengths[i]:] == False
    # Verify padded positions don't affect output of non-padded items

def test_audio_variable_lengths():
    """Different waveform lengths produce correct frame masks."""
    # Create waveforms of 8000, 16000, 12000 samples
    # Compute mel frames per item
    # Verify frame_mask aligns with actual frame counts

def test_sensor_variable_lengths():
    """Different sensor sequence lengths produce correct masks."""
    # Create sequences of lengths 50, 100, 75
    # Verify mask alignment
```

### Verification strategy

1. Construct a batch where every item has a different raw length.
2. Pad to the maximum length in the batch.
3. Pass through the encoder.
4. Assert `mask[i, :len_i]` is all `True`.
5. Assert `mask[i, len_i:]` is all `False`.
6. Re-run with only the shortest item and confirm that non-padded positions
   produce the same feature values (tolerance `atol=1e-5`).

---

## Test Category C: Concatenation Compatibility

The workspace consumes encoder outputs by concatenating them along the token
dimension. These tests ensure that concatenation requires zero modality-specific
branching.

```python
def test_cross_modal_concatenation():
    """Outputs from 2-4 modalities must concatenate without branching."""
    outputs = []
    for modality in ["vision", "text", "audio"]:
        enc = create_encoder(modality)
        batch = make_sample_batch(modality, B=4)
        outputs.append(enc(batch))

    # This must work with no modality-specific logic:
    all_feats = torch.cat([o.feats for o in outputs], dim=1)
    all_masks = torch.cat([o.mask for o in outputs], dim=1)

    assert all_feats.ndim == 3
    assert all_feats.shape[0] == 4  # batch
    assert all_feats.shape[2] == WORKSPACE_DIM
    assert all_masks.shape == all_feats.shape[:2]

def test_all_modality_pairs():
    """Every pair of modalities must concatenate cleanly."""
    for m1, m2 in itertools.combinations(MODALITIES, 2):
        out1 = run_encoder(m1, B=2)
        out2 = run_encoder(m2, B=2)
        combined_feats = torch.cat([out1.feats, out2.feats], dim=1)
        combined_mask = torch.cat([out1.mask, out2.mask], dim=1)
        assert combined_feats.shape[:2] == combined_mask.shape
```

### Why this matters

If any encoder emits a shape that deviates from `(B, T, WORKSPACE_DIM)`, the
`torch.cat` on the token axis will raise a dimension mismatch. Catching this in
a pairwise sweep prevents silent breakage when a new modality is added.

---

## Test Category D: Missing Dependency Simulation

Use `monkeypatch` to remove optional packages from `sys.modules` before
importing the encoder. Confirm the encoder still initializes, runs, and
returns a valid `EncoderOutput` using its fallback backend.

```python
@pytest.fixture
def no_torchaudio(monkeypatch):
    """Simulate torchaudio not installed."""
    import sys
    monkeypatch.setitem(sys.modules, 'torchaudio', None)
    monkeypatch.setitem(sys.modules, 'torchaudio.transforms', None)

@pytest.fixture
def no_ncps(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'ncps', None)
    monkeypatch.setitem(sys.modules, 'ncps.torch', None)
    monkeypatch.setitem(sys.modules, 'ncps.wirings', None)

@pytest.fixture
def no_tonic(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'tonic', None)

def test_audio_fallback(no_torchaudio):
    """Audio encoder works without torchaudio."""
    encoder = create_audio_encoder(output_dim=WORKSPACE_DIM)
    output = encoder(torch.randn(2, 16000))
    assert output.feats.shape[-1] == WORKSPACE_DIM
    assert output.aux.get("audio_frontend") == "torch_stft_fallback"

def test_sensor_fallback(no_ncps):
    """Sensor encoder works without ncps."""
    encoder = create_sensor_encoder(input_dim=6, output_dim=WORKSPACE_DIM)
    output = encoder(torch.randn(2, 50, 6))
    assert output.feats.shape[-1] == WORKSPACE_DIM
```

### Rules for fallback tests

- The `aux` dict must record which backend was selected (e.g.,
  `"audio_frontend": "torch_stft_fallback"`).
- The returned `EncoderOutput` schema must be identical to the primary-backend
  output: same fields, same shapes, same dtypes.
- No `ImportError` or `ModuleNotFoundError` may propagate to the caller.

---

## Test Category E: Deterministic Preprocessing

Confirm that the same input with the same random seed produces bit-identical
(or near-identical for floating-point ops) output. This catches hidden
statefulness in preprocessing layers such as dropout or stochastic augmentation
that should be disabled in eval mode.

```python
def test_text_deterministic():
    """Same input produces identical output with same seed."""
    torch.manual_seed(42)
    out1 = run_encoder("text", input_ids=sample_ids)
    torch.manual_seed(42)
    out2 = run_encoder("text", input_ids=sample_ids)
    assert torch.equal(out1.feats, out2.feats)

def test_audio_deterministic():
    """Same waveform produces identical mel features."""
    waveform = torch.randn(2, 16000)
    out1 = run_audio_encoder(waveform)
    out2 = run_audio_encoder(waveform)
    assert torch.allclose(out1.feats, out2.feats, atol=1e-6)
```

### Notes

- Set the encoder to `.eval()` mode before running determinism checks.
- For audio, the mel-spectrogram path is fully deterministic; use `torch.equal`
  when no learned layers are involved, `torch.allclose` otherwise.
- Caching layers (e.g., engram hash lookups) must not alter results between
  the first and second call for the same input.

---

## Test Category F: Salience and Spike Fields

Optional fields on `EncoderOutput` (salience, spike) still carry invariants
when present. Enforce those invariants unconditionally.

```python
def test_salience_non_negative():
    """Salience values must be non-negative if present."""
    for modality in MODALITIES:
        output = run_encoder(modality)
        if output.salience is not None:
            assert (output.salience >= 0).all()

def test_spike_alignment():
    """Spike tensor must align with feats batch and time dims."""
    for modality in ["vision", "audio"]:  # spiking modalities
        output = run_encoder(modality, spiking=True)
        if output.spike is not None:
            assert output.spike.shape[0] == output.feats.shape[0]
            assert output.spike.shape[1] == output.feats.shape[1]
```

### Salience constraints

- Non-negative (zero is valid; it means "no salience signal").
- Shape must broadcast with `feats` along the token dimension: either
  `(B, T)` or `(B, T, 1)`.

### Spike constraints

- Binary or soft-binary values in `[0, 1]`.
- First two dimensions `(B, T)` must match `feats`.
- Third dimension, if present, equals the number of spike channels.

---

## "Done When" Checklist

### Per-encoder gates

- [ ] Shape tests: feats `(B, T, D)`, mask `(B, T)` -- all modalities
- [ ] Device tests: CPU and CUDA (if available)
- [ ] Dtype tests: fp32 + AMP compute (bf16)
- [ ] Variable-length tests: different `T` per batch item
- [ ] Mask alignment: `mask[i, :len_i] == True`, `mask[i, len_i:] == False`
- [ ] No NaN/Inf in feats under normal inputs
- [ ] Salience non-negative (if present)
- [ ] Spike aligned to feats (if present)
- [ ] `aux` contains expected keys

### Cross-modal batch gates

- [ ] Concatenate feats/masks across any 2+ enabled modalities
- [ ] No special-casing in workspace input prep
- [ ] Combined feats shape is `(B, sum(T_i), D)`

### Optional-dependency gates

- [ ] No runtime crash when torchaudio/ncps/tonic/transformers missing
- [ ] Fallback mode selected and recorded in `aux`
- [ ] Same `EncoderOutput` schema returned regardless of backend
- [ ] Dependency report reflects actual state

### Determinism gates

- [ ] Same input + same seed = same output
- [ ] Caching does not change results

---

## Running the Matrix

Execute the full encoder test matrix:

```bash
python -m pytest tests/test_encoders.py -v --tb=short
```

Run a single category in isolation:

```bash
python -m pytest tests/test_encoders.py -v -k "test_encoder_contract"
python -m pytest tests/test_encoders.py -v -k "variable_length"
python -m pytest tests/test_encoders.py -v -k "fallback"
```

Skip CUDA tests on CPU-only machines (handled automatically by the parametrize
decorator, but can also be forced):

```bash
python -m pytest tests/test_encoders.py -v -k "not cuda"
```

Generate a coverage report scoped to the encoder package:

```bash
python -m pytest tests/test_encoders.py --cov=brain_ai.encoders --cov-report=term-missing
```

Mark a test matrix run as the release gate: all categories must pass with zero
skips (other than hardware-gated CUDA tests) before merging encoder changes.
