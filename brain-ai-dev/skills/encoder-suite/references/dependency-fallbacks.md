# Encoder Dependency Fallbacks

## Overview

Every optional dependency must degrade to a working baseline that produces the
same output shapes as the primary implementation. The fallback must be testable
in isolation and recorded in `aux` metadata so that training runs remain fully
explainable. Never silently change behaviour -- always surface which backend is
active.

## Fallback Matrix

| Dependency | Used By | Fallback | Shape Impact | aux Key |
|-----------|---------|----------|-------------|---------|
| torchaudio | AudioEncoder mel frontend | `torch.stft` + triangular mel filterbank | Frame count may differ slightly; downstream uses mask, never assumes T | `audio_frontend` |
| transformers | TextEncoder (optional pretrained) | Built-in `nn.TransformerEncoder` | Same `(B, L, D)` | `text_backend` |
| ncps | SensorEncoder CfC/LTC | Built-in CfC/LTC or GRU | Same `(B, T, D)` | `sensor_backend` |
| tonic | EventVisionEncoder | Simple event binning or disable event input | Same `(B, N, D)` or modality disabled | `event_frontend` |
| snntorch | Spiking layers (optional enhanced) | Built-in `LIFNeuron` | Same shapes | `snn_backend` |

## Concrete Fallback Implementations

### Audio: torchaudio Missing

```python
class MelSpectrogramFrontend(nn.Module):
    def __init__(self, ...):
        try:
            import torchaudio
            self.mel_transform = torchaudio.transforms.MelSpectrogram(...)
            self._backend = "torchaudio"
        except ImportError:
            self._backend = "torch_stft_fallback"
            self.register_buffer('mel_fb', self._create_mel_filterbank(...))

    def forward(self, waveform):
        if self._backend == "torchaudio":
            mel = self.mel_transform(waveform)
        else:
            # Force fp32 for STFT numerical stability
            with torch.amp.autocast(device_type='cuda', enabled=False):
                stft = torch.stft(waveform.float(), n_fft=self.n_fft,
                                  hop_length=self.hop_length, return_complex=True)
            power_spec = stft.abs() ** 2
            mel = torch.matmul(self.mel_fb, power_spec.transpose(-1, -2)).transpose(-1, -2)
        return torch.log(mel + 1e-9)
```

Important: frame count may differ between torchaudio and manual STFT due to
padding conventions. Accept the difference but ensure downstream always uses
the mask and never assumes a fixed T. Record the active backend in `aux`.

### Sensors: ncps Missing

```python
# In sensors.py, already implemented:
try:
    from ncps.torch import CfC, LTC
    from ncps.wirings import AutoNCP
    NCPS_AVAILABLE = True
except ImportError:
    NCPS_AVAILABLE = False

# Fallback: use built-in ClosedFormContinuous or LiquidTimeConstant.
# These are simplified versions but produce the same output shapes.
```

Keep timestamp features: embed `dt` as an extra channel in input features even
without ncps. Do not drop temporal spacing information just because the primary
library is absent.

### Text: transformers Missing

```python
try:
    from transformers import AutoModel
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Fallback: built-in TextEncoder with nn.TransformerEncoder.
# Tokenization fallback: enforce token_ids only (no raw-string support
# without transformers).
```

When `transformers` is missing, raise a clear `ValueError` if the caller
passes raw strings instead of pre-tokenized IDs. Never silently truncate or
ignore the input.

### Event Vision: tonic Missing

```python
try:
    import tonic
    TONIC_AVAILABLE = True
except ImportError:
    TONIC_AVAILABLE = False

# Fallback: simple event binning if events are provided.
# Or disable event input with a clear warning, allowing other modalities
# to proceed.
```

If tonic is absent and event data is the *only* modality supplied, raise an
error rather than returning an empty tensor.

## Recording Fallback Selections

Every encoder adapter must populate `aux` with backend information so that
logs, checkpoints, and analysis scripts can determine exactly which code path
executed.

```python
aux = {
    "audio_frontend": self.mel_frontend._backend,   # "torchaudio" | "torch_stft_fallback"
    "sensor_backend": "ncps_cfc" if NCPS_AVAILABLE else "builtin_cfc",
    "text_backend":   "hf_transformers" if HF_AVAILABLE else "builtin_transformer",
    "event_frontend": "tonic" if TONIC_AVAILABLE else "simple_binning",
    "snn_backend":    "snntorch" if SNNTORCH_AVAILABLE else "builtin_lif",
}
```

Persist these keys alongside training metrics. If a fallback changes between
runs (e.g., a library is installed mid-project), the change will be visible in
the `aux` history.

## Dependency Report Integration

Provide a single entry-point function that summarises all encoder-specific
optional dependencies and their current status.

```python
def encoder_deps_report() -> Dict[str, Dict]:
    """Report encoder-specific optional dependencies."""
    return {
        "torchaudio": {
            "status": "present" if _check("torchaudio") else "missing",
            "fallback": "torch.stft + manual mel filterbank",
        },
        "ncps": {
            "status": "present" if _check("ncps") else "missing",
            "fallback": "built-in CfC/LTC or GRU",
        },
        "transformers": {
            "status": "present" if _check("transformers") else "missing",
            "fallback": "built-in nn.TransformerEncoder",
        },
        "tonic": {
            "status": "present" if _check("tonic") else "missing",
            "fallback": "simple event binning or disabled",
        },
        "snntorch": {
            "status": "present" if _check("snntorch") else "missing",
            "fallback": "built-in LIFNeuron",
        },
    }
```

Call `encoder_deps_report()` at the start of every training run and log the
result. Include it in checkpoint metadata so that model provenance is complete.

## Testing Fallback Paths

Use monkeypatching to simulate missing dependencies. Each fallback must have a
dedicated test that forces the fallback path and verifies the output contract.

```python
@pytest.fixture
def no_torchaudio(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'torchaudio', None)
    monkeypatch.setitem(sys.modules, 'torchaudio.transforms', None)


def test_audio_fallback(no_torchaudio):
    encoder = create_audio_encoder(output_dim=512)
    waveform = torch.randn(2, 16000)
    output = encoder(waveform)
    assert output.feats.shape[0] == 2
    assert output.feats.shape[-1] == 512     # D is exact
    # T varies but shape contract holds -- verify via mask
    assert output.mask is not None
    assert output.aux["audio_frontend"] == "torch_stft_fallback"


@pytest.fixture
def no_ncps(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'ncps', None)
    monkeypatch.setitem(sys.modules, 'ncps.torch', None)
    monkeypatch.setitem(sys.modules, 'ncps.wirings', None)


def test_sensor_fallback(no_ncps):
    encoder = create_sensor_encoder(input_dim=32, output_dim=512)
    x = torch.randn(2, 50, 32)
    output = encoder(x)
    assert output.feats.shape == (2, 50, 512)
    assert output.aux["sensor_backend"] == "builtin_cfc"
```

Follow the same monkeypatch pattern for `transformers`, `tonic`, and
`snntorch`. Always assert both shape correctness and the expected `aux` key
value.

## Common Failure Mode Prevention

| Failure | Cause | Prevention |
|---------|-------|-----------|
| Different frame counts in fallback | STFT padding conventions differ between torchaudio and `torch.stft` | Always use `mask`; never assume a fixed T |
| Half-precision breaks STFT | Complex arithmetic unstable in fp16 | Force fp32 for STFT (`autocast disabled`), cast back after |
| Missing tokenizer with raw strings | `transformers` not installed | Enforce `token_ids` only in no-transformers mode; raise on raw strings |
| ncps wiring shape mismatch | `AutoNCP` output dimensionality != expected dim | Always project through an adapter linear layer after the recurrent cell |
| Silent modality dropout | Fallback returns zeros instead of erroring | Check tensor norms; raise if the sole modality produces all zeros |
| Inconsistent RNG across backends | Fallback initialises weights differently | Seed explicitly in tests; accept small numerical differences in training |

## Checklist for Adding a New Fallback

1. Add the dependency to the fallback matrix table above.
2. Implement the `try/except ImportError` guard at module level.
3. Ensure the fallback produces identical output shapes (or document any
   axis that may vary and confirm masking handles it).
4. Record the active backend string in `aux` under a descriptive key.
5. Add the dependency to `encoder_deps_report()`.
6. Write a monkeypatch test that forces the fallback and asserts shape +
   `aux` value.
7. Update this reference document.
