# Preprocessing Standards — Reference for Data Pipeline & Loaders Skill

This document specifies the normalization conventions, tokenization protocols, spectrogram computation, masking strategies, dtype requirements, and device placement rules for all modalities in the brain_ai data pipeline. Use this as the canonical reference when implementing preprocessing functions or auditing data quality.

---

## 1. Vision Preprocessing

### 1.1 Normalization

All vision data passes through a two-stage normalization pipeline:

**Stage 1 -- Value Range Normalization:**
Raw pixel values (typically uint8 in `[0, 255]`) are converted to float32 in `[0, 1]` by dividing by 255.0. If the data is already float32 in `[0, 1]`, this step is a no-op.

```python
def normalize_image_range(x: Tensor) -> Tensor:
    if x.dtype == torch.uint8:
        return x.float() / 255.0
    if x.max() > 1.0:
        return x / 255.0
    return x
```

**Stage 2 -- Channel-wise Normalization (production only):**
For production mode with pretrained encoders, apply ImageNet channel-wise normalization:

```
mean = [0.485, 0.456, 0.406]  # RGB channels
std  = [0.229, 0.224, 0.225]  # RGB channels
normalized = (x - mean) / std
```

For grayscale (dev mode, single channel), use global normalization:
```
mean = [0.5]
std  = [0.5]
```

This produces values approximately in `[-2, 2]` for natural images.

### 1.2 Spatial Preprocessing

**Resize:** Images are resized to the target resolution using bilinear interpolation. Dev mode target is 28x28 (MNIST-scale). Production target is 224x224 or 384x384 (matching `config.encoder.vision_image_size`).

**Center Crop (eval only):** For evaluation, images are center-cropped to the target size after resizing to a slightly larger resolution (e.g., resize to 256x256 then center-crop to 224x224). This avoids edge artifacts.

**Channel Order:** All vision tensors use CHW format (channels first). If source data is HWC (numpy convention), transpose to CHW before any processing.

```python
def ensure_chw(x: Tensor) -> Tensor:
    if x.ndim == 3 and x.shape[-1] in (1, 3, 4):
        return x.permute(2, 0, 1)
    return x
```

### 1.3 Grayscale Handling

Dev mode datasets (MNIST-like) produce single-channel images `[1, H, W]`. If a production encoder expects 3 channels, repeat the grayscale channel:

```python
def grayscale_to_rgb(x: Tensor) -> Tensor:
    if x.shape[0] == 1:
        return x.expand(3, -1, -1)
    return x
```

---

## 2. Text Preprocessing

### 2.1 Tokenization

**Dev mode tokenizer:** A simple character-level tokenizer that maps ASCII characters to integer IDs. Vocabulary size is limited to 256 (one ID per byte). This is fast, requires no external dependencies, and produces valid token ID tensors.

```python
def simple_tokenize(text: str, max_len: int = 128) -> Tensor:
    token_ids = [ord(c) % 256 for c in text[:max_len]]
    # Pad to max_len
    token_ids = token_ids + [0] * (max_len - len(token_ids))
    return torch.tensor(token_ids, dtype=torch.long)
```

**Production mode tokenizer:** Uses a BPE tokenizer with `vocab_size=128000` matching `config.encoder.text_vocab_size`. The tokenizer should be loaded from a saved vocabulary file. Compatible tokenizers include SentencePiece, tiktoken, or HuggingFace tokenizers.

### 2.2 Padding and Truncation

All text sequences are fixed-length: either `max_len=128` (dev) or `max_len=512` / `max_len=8192` (production, matching `config.encoder.text_max_seq_len`).

**Truncation:** Sequences longer than `max_len` are truncated from the right (last tokens removed).

**Padding:** Sequences shorter than `max_len` are right-padded with the pad token ID (0).

**Attention mask:** A boolean tensor of shape `[seq_len]` where `True` indicates real tokens and `False` indicates padding:

```python
def create_attention_mask(token_ids: Tensor, pad_id: int = 0) -> Tensor:
    return token_ids != pad_id
```

### 2.3 Special Tokens

| Token | ID | Purpose |
|---|---|---|
| PAD | 0 | Padding token |
| BOS | 1 | Beginning of sequence |
| EOS | 2 | End of sequence |
| UNK | 3 | Unknown/out-of-vocabulary token |
| MASK | 4 | Mask token (for masked language modeling) |

Dev mode uses these fixed IDs. Production mode uses the tokenizer's native special token IDs.

---

## 3. Audio Preprocessing

### 3.1 Log-Mel Spectrogram Computation

Raw audio waveforms are converted to log-mel spectrograms using the following pipeline:

```
waveform [1, num_samples]
  -> STFT [n_fft, T_frames]
  -> Mel filterbank [n_mels, T_frames]
  -> Log transform [n_mels, T_frames]
  -> Normalization [n_mels, T_frames]
```

**STFT parameters:**
| Parameter | Dev Value | Production Value | Source |
|---|---|---|---|
| `n_fft` | 512 | 1024 | Window size in samples |
| `hop_length` | 256 | 512 | Stride between frames |
| `win_length` | 512 | 1024 | Window function length |
| `window` | Hann | Hann | Window function type |

**Mel filterbank parameters:**
| Parameter | Dev Value | Production Value |
|---|---|---|
| `n_mels` | 64 | 128 (from `config.encoder.audio_n_mels`) |
| `f_min` | 0 Hz | 0 Hz |
| `f_max` | 8000 Hz | 8000 Hz |
| `sample_rate` | 16000 Hz | 16000 Hz (from `config.encoder.audio_sample_rate`) |

### 3.2 Log Transform

Apply log compression to prevent dynamic range issues:

```python
def log_mel_transform(mel_spec: Tensor, eps: float = 1e-9) -> Tensor:
    return torch.log(mel_spec + eps)
```

The epsilon prevents `log(0)` which would produce `-inf`.

### 3.3 Spectrogram Normalization

After log transform, apply per-channel (per-mel-bin) normalization:

```python
def normalize_spectrogram(log_mel: Tensor) -> Tensor:
    mean = log_mel.mean(dim=-1, keepdim=True)
    std = log_mel.std(dim=-1, keepdim=True).clamp(min=1e-6)
    return (log_mel - mean) / std
```

For production mode with known dataset statistics, use precomputed global mean and std.

### 3.4 Synthetic Audio Generation (Dev Mode)

For dev mode, generate synthetic spectrograms directly without requiring real audio:

```python
def generate_synthetic_spectrogram(n_mels: int = 64, T: int = 100) -> Tensor:
    # Generate a spectrogram with realistic structure
    freq_pattern = torch.linspace(0, 1, n_mels).unsqueeze(1)
    time_pattern = torch.sin(torch.linspace(0, 4 * math.pi, T)).unsqueeze(0)
    base = freq_pattern * time_pattern
    noise = torch.randn(n_mels, T) * 0.1
    return base + noise
```

---

## 4. Sequence Preprocessing

### 4.1 Normalization

Sequence data `[T, D]` is normalized per-feature (per-dimension) across the temporal axis:

```python
def normalize_sequence(x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    if mask is not None:
        # Only compute stats on valid timesteps
        valid = x[mask]
        mean = valid.mean(dim=0, keepdim=True)
        std = valid.std(dim=0, keepdim=True).clamp(min=1e-6)
    else:
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True).clamp(min=1e-6)
    return (x - mean) / std
```

### 4.2 Masking Conventions

**Boolean masks:** Shape `[T]` where `True` = valid, `False` = padding/invalid. This matches PyTorch's `nn.TransformerEncoder` convention for `src_key_padding_mask` (after negation).

**Integer masks:** Shape `[T]` where `1` = valid, `0` = padding. Used when boolean operations are inconvenient.

**Conversion:**
```python
bool_mask = int_mask.bool()
int_mask = bool_mask.long()
```

### 4.3 Variable-Length Handling

Sequences of different lengths within a batch are padded to the maximum length. The padding value is 0.0 for float tensors and 0 for integer tensors.

```python
def pad_sequences(sequences: List[Tensor], pad_value: float = 0.0) -> Tuple[Tensor, Tensor]:
    max_len = max(s.shape[0] for s in sequences)
    feat_dim = sequences[0].shape[-1] if sequences[0].ndim > 1 else 1
    padded = torch.full((len(sequences), max_len, feat_dim), pad_value)
    masks = torch.zeros(len(sequences), max_len, dtype=torch.bool)
    for i, s in enumerate(sequences):
        length = s.shape[0]
        padded[i, :length] = s
        masks[i, :length] = True
    return padded, masks
```

---

## 5. RL Episode Preprocessing

### 5.1 State Normalization

RL states are normalized using running statistics (online mean/std estimation):

```python
class RunningNormalizer:
    def __init__(self, shape: Tuple[int, ...], eps: float = 1e-6):
        self.mean = torch.zeros(shape)
        self.var = torch.ones(shape)
        self.count = 0
        self.eps = eps

    def update(self, x: Tensor):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0)
        batch_count = x.shape[0]
        self._update_stats(batch_mean, batch_var, batch_count)

    def normalize(self, x: Tensor) -> Tensor:
        return (x - self.mean) / torch.sqrt(self.var + self.eps)
```

### 5.2 Reward Scaling

Rewards are optionally scaled to have unit variance across the dataset. This helps stabilize policy gradient methods:

```python
def scale_rewards(rewards: Tensor) -> Tensor:
    std = rewards.std().clamp(min=1e-6)
    return rewards / std
```

### 5.3 Trajectory Structure

Each RL sample is a dictionary with these keys:

| Key | Shape | dtype | Description |
|---|---|---|---|
| `state` | `[state_dim]` | `float32` | Current observation |
| `action` | `[]` or `[action_dim]` | `int64` or `float32` | Discrete or continuous action |
| `reward` | `[]` | `float32` | Scalar reward |
| `next_state` | `[state_dim]` | `float32` | Next observation |
| `done` | `[]` | `float32` | Terminal flag (0.0 or 1.0) |

---

## 6. dtype Requirements

### 6.1 Input Tensors

| Modality | dtype | Rationale |
|---|---|---|
| Vision | `torch.float32` | Standard for neural network inputs |
| Text (token IDs) | `torch.int64` (`torch.long`) | Required by `nn.Embedding` |
| Text (attention mask) | `torch.bool` | Memory efficient, native mask type |
| Audio | `torch.float32` | Standard for neural network inputs |
| Sequences | `torch.float32` | Standard for neural network inputs |
| Sequence masks | `torch.bool` | Memory efficient, native mask type |
| RL states | `torch.float32` | Standard for neural network inputs |
| RL actions (discrete) | `torch.int64` | Index type |
| RL actions (continuous) | `torch.float32` | Continuous values |

### 6.2 Target Tensors

| Task Type | dtype | Rationale |
|---|---|---|
| Classification | `torch.int64` | Required by `nn.CrossEntropyLoss` |
| Regression | `torch.float32` | Required by `nn.MSELoss` |
| Next-token prediction | `torch.int64` | Required by `nn.CrossEntropyLoss` |
| Sequence prediction | `torch.float32` | Continuous target sequences |
| RL rewards | `torch.float32` | Continuous reward signal |

### 6.3 Mixed Precision Compatibility

When `config.training.use_amp` is True, the model runs in bfloat16 or float16. However, data loaders must ALWAYS produce float32 tensors. The autocast context manager handles the dtype conversion inside the model.

**Do NOT produce float16 tensors from data loaders.** This can cause precision issues in loss computation and gradient accumulation.

---

## 7. Device Placement

### 7.1 CPU Loading

All data loading and preprocessing happens on CPU. DataLoader workers run on CPU and produce CPU tensors. GPU transfer happens inside the training loop:

```python
for batch in train_loader:
    # Transfer to GPU inside the loop
    batch = {k: v.to(device) for k, v in batch.items()}
    output = model(batch)
```

### 7.2 Pin Memory

When `config.pin_memory` is True (default), DataLoader allocates CPU tensors in pinned (page-locked) memory. This enables faster asynchronous CPU-to-GPU transfers via `tensor.to(device, non_blocking=True)`.

**Requirements for pin_memory:**
- All tensors returned by the dataset/collate must be on CPU.
- A CUDA device must be available.
- The system must have sufficient pinned memory (controlled by OS limits).

### 7.3 Non-Blocking Transfers

For maximum throughput, use non-blocking transfers in the training loop:

```python
for batch in train_loader:
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    # CUDA operations are queued and synchronized automatically
    output = model(batch)
    loss = criterion(output, batch["target"])
```

This overlaps data transfer with computation on the GPU.

---

## 8. Preprocessing Function Signatures

All preprocessing functions follow a consistent signature pattern:

```python
def preprocess_<modality>(
    raw_data: Union[Tensor, np.ndarray, str],
    config: DataConfig,
    mode: str = "dev",
) -> Dict[str, Tensor]:
```

Each function returns a dictionary of tensors with standardized keys. This enables uniform handling in the data pipeline regardless of modality.

### 8.1 Vision Preprocessing Function

```python
def preprocess_vision(image, config, mode="dev") -> Dict[str, Tensor]:
    # Returns: {"input": [C, H, W] float32, "target": [] int64}
```

### 8.2 Text Preprocessing Function

```python
def preprocess_text(text, config, mode="dev") -> Dict[str, Tensor]:
    # Returns: {"input": [seq_len] int64, "attention_mask": [seq_len] bool, "target": [] int64}
```

### 8.3 Audio Preprocessing Function

```python
def preprocess_audio(waveform, config, mode="dev") -> Dict[str, Tensor]:
    # Returns: {"input": [n_mels, T] float32, "target": [] int64}
```

### 8.4 Sequence Preprocessing Function

```python
def preprocess_sequence(sequence, config, mode="dev") -> Dict[str, Tensor]:
    # Returns: {"input": [T, D] float32, "mask": [T] bool, "target": [T, D] float32 or [] int64}
```

---

## 9. Data Quality Checks

All preprocessing functions should include optional validation that can be enabled via `config` or a `validate=True` flag:

1. **Range check:** Verify output values are in expected range (e.g., normalized images in `[-3, 3]`).
2. **dtype check:** Verify all tensors have the correct dtype.
3. **Shape check:** Verify output shapes match the contract.
4. **NaN/Inf check:** Verify no NaN or Inf values in output tensors.
5. **Mask consistency:** For masked data, verify mask shape matches data shape.

```python
def validate_preprocessed(data: Dict[str, Tensor], modality: str) -> List[str]:
    errors = []
    for key, tensor in data.items():
        if torch.isnan(tensor).any():
            errors.append(f"{key} contains NaN values")
        if torch.isinf(tensor).any():
            errors.append(f"{key} contains Inf values")
    return errors
```
