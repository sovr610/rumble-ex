"""
brain_ai/data/preprocessing.py -- Preprocessing functions for all modalities.

Provides standardized preprocessing for vision, text, audio, and sequence data.
All functions return dict-of-tensors with consistent keys and shapes matching
the preprocessing contract.

Key functions:
    normalize_image       -- Normalize vision tensors to [0,1] or channel-wise
    ensure_chw            -- Convert HWC to CHW format
    grayscale_to_rgb      -- Expand single-channel to 3-channel
    resize_image          -- Resize with bilinear interpolation
    tokenize_text         -- Character-level tokenizer (dev mode)
    create_attention_mask -- Create boolean mask from token IDs
    compute_spectrogram   -- Synthetic or real spectrogram generation
    log_mel_transform     -- Log-compress mel spectrogram
    normalize_spectrogram -- Per-mel-bin normalization
    normalize_sequence    -- Per-feature sequence normalization
    pad_sequences         -- Pad variable-length sequences with masking
    validate_preprocessed -- Validate preprocessed output quality

Usage:
    img = normalize_image(raw_image)
    tokens = tokenize_text("hello world", max_len=128)
    spec = compute_spectrogram(n_mels=64, T=100)
    seq = normalize_sequence(raw_sequence)
"""

from __future__ import annotations

import math
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SECTION 1: Vision Preprocessing
# ---------------------------------------------------------------------------

def normalize_image(x: Tensor, mean: Optional[List[float]] = None,
                    std: Optional[List[float]] = None) -> Tensor:
    """Normalize image tensor to [0, 1] range, then optionally apply channel-wise norm.

    Args:
        x: Image tensor of shape [C, H, W] or [H, W, C].
        mean: Per-channel mean (e.g., ImageNet: [0.485, 0.456, 0.406]).
        std: Per-channel std (e.g., ImageNet: [0.229, 0.224, 0.225]).

    Returns:
        Normalized float32 tensor of shape [C, H, W].
    """
    # Convert uint8 to float
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    elif x.dtype != torch.float32:
        x = x.float()

    # Scale if values are outside [0, 1]
    if x.max() > 1.0:
        x = x / 255.0

    # Clamp to valid range
    x = x.clamp(0.0, 1.0)

    # Apply channel-wise normalization if provided
    if mean is not None and std is not None:
        m = torch.tensor(mean, dtype=x.dtype).view(-1, 1, 1)
        s = torch.tensor(std, dtype=x.dtype).view(-1, 1, 1)
        x = (x - m) / s

    return x


def ensure_chw(x: Tensor) -> Tensor:
    """Convert HWC tensor to CHW format if necessary.

    Args:
        x: Image tensor of shape [H, W, C] or [C, H, W].

    Returns:
        Tensor in [C, H, W] format.
    """
    if x.ndim == 3 and x.shape[-1] in (1, 3, 4):
        # Likely HWC format, check if last dim is channels
        if x.shape[0] not in (1, 3, 4) or x.shape[-1] < x.shape[0]:
            return x.permute(2, 0, 1)
    return x


def grayscale_to_rgb(x: Tensor) -> Tensor:
    """Expand single-channel image to 3-channel by repeating.

    Args:
        x: Tensor of shape [1, H, W].

    Returns:
        Tensor of shape [3, H, W].
    """
    if x.ndim == 3 and x.shape[0] == 1:
        return x.expand(3, -1, -1)
    return x


def resize_image(x: Tensor, target_size: Tuple[int, int]) -> Tensor:
    """Resize image using bilinear interpolation.

    Args:
        x: Tensor of shape [C, H, W].
        target_size: (target_H, target_W).

    Returns:
        Resized tensor of shape [C, target_H, target_W].
    """
    if x.shape[1] == target_size[0] and x.shape[2] == target_size[1]:
        return x
    return torch.nn.functional.interpolate(
        x.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False
    ).squeeze(0)


def center_crop(x: Tensor, crop_size: Tuple[int, int]) -> Tensor:
    """Center crop an image tensor.

    Args:
        x: Tensor of shape [C, H, W].
        crop_size: (crop_H, crop_W).

    Returns:
        Cropped tensor of shape [C, crop_H, crop_W].
    """
    C, H, W = x.shape
    crop_h, crop_w = crop_size
    if crop_h >= H and crop_w >= W:
        return x
    top = (H - crop_h) // 2
    left = (W - crop_w) // 2
    return x[:, top:top + crop_h, left:left + crop_w]


def preprocess_vision(image: Tensor, mode: str = "dev",
                      target_size: Optional[Tuple[int, int]] = None,
                      channel_norm: bool = False) -> Dict[str, Tensor]:
    """Full vision preprocessing pipeline.

    Args:
        image: Raw image tensor.
        mode: "dev" or "production".
        target_size: Optional resize target (H, W).
        channel_norm: Whether to apply ImageNet normalization.

    Returns:
        Dict with "input" key containing [C, H, W] float32 tensor.
    """
    x = ensure_chw(image)

    if target_size is not None:
        x = resize_image(x, target_size)

    if channel_norm and mode == "production" and x.shape[0] == 3:
        x = normalize_image(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif x.shape[0] == 1:
        x = normalize_image(x, mean=[0.5], std=[0.5])
    else:
        x = normalize_image(x)

    return {"input": x}


# ---------------------------------------------------------------------------
# SECTION 2: Text Preprocessing
# ---------------------------------------------------------------------------

# Special token IDs
PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
UNK_TOKEN_ID = 3
MASK_TOKEN_ID = 4


def tokenize_text(text: str, max_len: int = 128, vocab_size: int = 256,
                  add_special_tokens: bool = True) -> Tensor:
    """Simple character-level tokenizer for dev mode.

    Maps each character to its ordinal value mod vocab_size. Pads or truncates
    to max_len.

    Args:
        text: Input string.
        max_len: Maximum sequence length.
        vocab_size: Vocabulary size (default 256 for byte-level).
        add_special_tokens: Whether to prepend BOS and append EOS.

    Returns:
        Token ID tensor of shape [max_len], dtype int64.
    """
    # Convert characters to token IDs (skip special token range)
    token_ids = []
    if add_special_tokens:
        token_ids.append(BOS_TOKEN_ID)

    for c in text:
        tid = ord(c) % vocab_size
        # Avoid special token IDs (0-4)
        if tid < 5:
            tid = tid + 5
        token_ids.append(tid)

    if add_special_tokens:
        token_ids.append(EOS_TOKEN_ID)

    # Truncate
    if len(token_ids) > max_len:
        token_ids = token_ids[:max_len]

    # Pad
    pad_length = max_len - len(token_ids)
    token_ids = token_ids + [PAD_TOKEN_ID] * pad_length

    return torch.tensor(token_ids, dtype=torch.long)


def create_attention_mask(token_ids: Tensor, pad_id: int = 0) -> Tensor:
    """Create boolean attention mask from token IDs.

    Args:
        token_ids: Tensor of shape [seq_len] or [B, seq_len].
        pad_id: Padding token ID (default 0).

    Returns:
        Boolean tensor where True indicates real tokens.
    """
    return token_ids != pad_id


def batch_tokenize(texts: List[str], max_len: int = 128,
                   vocab_size: int = 256) -> Dict[str, Tensor]:
    """Tokenize a batch of texts.

    Args:
        texts: List of input strings.
        max_len: Maximum sequence length.
        vocab_size: Vocabulary size.

    Returns:
        Dict with "input" (token IDs) and "attention_mask" tensors.
    """
    tokens = torch.stack([tokenize_text(t, max_len, vocab_size) for t in texts])
    masks = create_attention_mask(tokens)
    return {"input": tokens, "attention_mask": masks}


def preprocess_text(text: str, max_len: int = 128, vocab_size: int = 256,
                    add_special_tokens: bool = True) -> Dict[str, Tensor]:
    """Full text preprocessing pipeline.

    Returns:
        Dict with "input" [seq_len] int64 and "attention_mask" [seq_len] bool.
    """
    tokens = tokenize_text(text, max_len, vocab_size, add_special_tokens)
    mask = create_attention_mask(tokens)
    return {"input": tokens, "attention_mask": mask}


# ---------------------------------------------------------------------------
# SECTION 3: Audio Preprocessing
# ---------------------------------------------------------------------------

def compute_spectrogram(n_mels: int = 64, T: int = 100, seed: int = 42) -> Tensor:
    """Generate a synthetic log-mel spectrogram with realistic structure.

    For dev mode, generates spectrograms directly without requiring real audio.
    Produces harmonically-structured patterns that approximate real speech.

    Args:
        n_mels: Number of mel frequency bins.
        T: Number of time frames.
        seed: Random seed for reproducibility.

    Returns:
        Log-mel spectrogram tensor of shape [n_mels, T], dtype float32.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    # Base harmonic pattern
    freq_pattern = torch.linspace(0, 1, n_mels).unsqueeze(1)
    time_pattern = torch.sin(torch.linspace(0, 4 * math.pi, T)).unsqueeze(0)
    base = freq_pattern * time_pattern

    # Add formant-like peaks
    formant_freqs = [int(n_mels * 0.2), int(n_mels * 0.4), int(n_mels * 0.7)]
    for ff in formant_freqs:
        if ff < n_mels:
            envelope = torch.exp(-0.5 * ((torch.arange(n_mels).float() - ff) / 5.0) ** 2)
            base = base + envelope.unsqueeze(1) * 0.5

    # Add noise
    noise = torch.randn(n_mels, T, generator=gen) * 0.1
    spectrogram = base + noise

    return spectrogram


def log_mel_transform(mel_spec: Tensor, eps: float = 1e-9) -> Tensor:
    """Apply log compression to mel spectrogram.

    Args:
        mel_spec: Mel spectrogram tensor of shape [n_mels, T].
        eps: Small value to prevent log(0).

    Returns:
        Log-compressed tensor, same shape.
    """
    return torch.log(mel_spec.clamp(min=eps))


def normalize_spectrogram(log_mel: Tensor) -> Tensor:
    """Per-channel (per-mel-bin) normalization.

    Args:
        log_mel: Log-mel spectrogram of shape [n_mels, T].

    Returns:
        Normalized tensor with zero mean and unit std per mel bin.
    """
    mean = log_mel.mean(dim=-1, keepdim=True)
    std = log_mel.std(dim=-1, keepdim=True).clamp(min=1e-6)
    return (log_mel - mean) / std


def preprocess_audio(n_mels: int = 64, T: int = 100,
                     normalize: bool = True, seed: int = 42) -> Dict[str, Tensor]:
    """Full audio preprocessing pipeline (synthetic).

    Returns:
        Dict with "input" [n_mels, T] float32 tensor.
    """
    spec = compute_spectrogram(n_mels, T, seed)
    spec = log_mel_transform(spec)
    if normalize:
        spec = normalize_spectrogram(spec)
    return {"input": spec}


# ---------------------------------------------------------------------------
# SECTION 4: Sequence Preprocessing
# ---------------------------------------------------------------------------

def normalize_sequence(x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """Per-feature normalization for sequence data.

    Normalizes each feature dimension to zero mean and unit standard deviation
    across the temporal axis.

    Args:
        x: Sequence tensor of shape [T, D].
        mask: Optional boolean mask of shape [T] where True = valid.

    Returns:
        Normalized tensor, same shape as input.
    """
    if mask is not None:
        # Only compute stats on valid timesteps
        valid_mask = mask.bool()
        if valid_mask.sum() < 2:
            return x
        valid = x[valid_mask]
        mean = valid.mean(dim=0, keepdim=True)
        std = valid.std(dim=0, keepdim=True).clamp(min=1e-6)
        result = x.clone()
        result[valid_mask] = (x[valid_mask] - mean) / std
        return result
    else:
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True).clamp(min=1e-6)
        return (x - mean) / std


def pad_sequences(sequences: List[Tensor], pad_value: float = 0.0
                  ) -> Tuple[Tensor, Tensor]:
    """Pad variable-length sequences to the maximum length in the batch.

    Args:
        sequences: List of tensors, each of shape [T_i, D] or [T_i].
        pad_value: Value used for padding.

    Returns:
        Tuple of (padded_tensor [B, max_T, D], mask_tensor [B, max_T] bool).
    """
    max_len = max(s.shape[0] for s in sequences)
    feat_dim = sequences[0].shape[-1] if sequences[0].ndim > 1 else 1
    B = len(sequences)

    padded = torch.full((B, max_len, feat_dim), pad_value, dtype=sequences[0].dtype)
    masks = torch.zeros(B, max_len, dtype=torch.bool)

    for i, s in enumerate(sequences):
        length = s.shape[0]
        if s.ndim == 1:
            padded[i, :length, 0] = s
        else:
            padded[i, :length] = s
        masks[i, :length] = True

    return padded, masks


def generate_synthetic_sequence(T: int, D: int, pattern: str = "sinusoidal",
                                seed: int = 42) -> Tensor:
    """Generate a synthetic sequence for dev mode testing.

    Args:
        T: Sequence length (timesteps).
        D: Feature dimension.
        pattern: "sinusoidal", "random_walk", or "step".
        seed: Random seed.

    Returns:
        Tensor of shape [T, D].
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    if pattern == "sinusoidal":
        t = torch.linspace(0, 4 * math.pi, T).unsqueeze(-1)
        freqs = torch.rand(1, D, generator=gen) * 2.0 + 0.5
        phases = torch.rand(1, D, generator=gen) * 2.0 * math.pi
        seq = torch.sin(t * freqs + phases)
        noise = torch.randn(T, D, generator=gen) * 0.05
        return seq + noise

    elif pattern == "random_walk":
        steps = torch.randn(T, D, generator=gen) * 0.1
        return torch.cumsum(steps, dim=0)

    elif pattern == "step":
        seq = torch.zeros(T, D)
        n_steps = torch.randint(2, 6, (1,), generator=gen).item()
        step_positions = sorted(torch.randint(0, T, (n_steps,), generator=gen).tolist())
        values = torch.randn(n_steps + 1, D, generator=gen)
        prev_pos = 0
        for idx, pos in enumerate(step_positions):
            seq[prev_pos:pos] = values[idx].unsqueeze(0)
            prev_pos = pos
        seq[prev_pos:] = values[-1].unsqueeze(0)
        return seq

    else:
        return torch.randn(T, D, generator=gen)


def preprocess_sequence(x: Tensor, mask: Optional[Tensor] = None,
                        normalize: bool = True) -> Dict[str, Tensor]:
    """Full sequence preprocessing pipeline.

    Returns:
        Dict with "input" [T, D] float32 and "mask" [T] bool.
    """
    if mask is None:
        mask = torch.ones(x.shape[0], dtype=torch.bool)
    if normalize:
        x = normalize_sequence(x, mask)
    return {"input": x, "mask": mask}


# ---------------------------------------------------------------------------
# SECTION 5: Data Quality Validation
# ---------------------------------------------------------------------------

def validate_preprocessed(data: Dict[str, Tensor], modality: str) -> List[str]:
    """Validate preprocessed data quality.

    Checks for NaN, Inf, dtype correctness, and shape validity.

    Args:
        data: Dict of tensor outputs from a preprocessing function.
        modality: One of "vision", "text", "audio", "sequence".

    Returns:
        List of error messages (empty if all checks pass).
    """
    errors = []

    for key, tensor in data.items():
        if not isinstance(tensor, Tensor):
            errors.append(f"{key} is not a Tensor (got {type(tensor).__name__})")
            continue

        # NaN check
        if tensor.is_floating_point() and torch.isnan(tensor).any():
            errors.append(f"{key} contains NaN values")

        # Inf check
        if tensor.is_floating_point() and torch.isinf(tensor).any():
            errors.append(f"{key} contains Inf values")

    # Modality-specific checks
    if modality == "vision" and "input" in data:
        inp = data["input"]
        if inp.ndim != 3:
            errors.append(f"Vision input must be 3D [C,H,W], got {inp.ndim}D")
        elif inp.shape[0] not in (1, 3, 4):
            errors.append(f"Vision channels must be 1, 3, or 4, got {inp.shape[0]}")
        if inp.dtype != torch.float32:
            errors.append(f"Vision input must be float32, got {inp.dtype}")

    if modality == "text" and "input" in data:
        inp = data["input"]
        if inp.dtype != torch.long:
            errors.append(f"Text input must be int64, got {inp.dtype}")
        if inp.ndim != 1:
            errors.append(f"Text input must be 1D, got {inp.ndim}D")
        if "attention_mask" in data:
            mask = data["attention_mask"]
            if mask.shape != inp.shape:
                errors.append(f"Attention mask shape {mask.shape} != input shape {inp.shape}")

    if modality == "audio" and "input" in data:
        inp = data["input"]
        if inp.ndim != 2:
            errors.append(f"Audio input must be 2D [n_mels,T], got {inp.ndim}D")
        if inp.dtype != torch.float32:
            errors.append(f"Audio input must be float32, got {inp.dtype}")

    if modality == "sequence" and "input" in data:
        inp = data["input"]
        if inp.ndim != 2:
            errors.append(f"Sequence input must be 2D [T,D], got {inp.ndim}D")
        if inp.dtype != torch.float32:
            errors.append(f"Sequence input must be float32, got {inp.dtype}")
        if "mask" in data:
            mask = data["mask"]
            if mask.shape[0] != inp.shape[0]:
                errors.append(f"Mask length {mask.shape[0]} != sequence length {inp.shape[0]}")

    return errors


# ---------------------------------------------------------------------------
# SECTION 6: Running Normalizer (for RL states)
# ---------------------------------------------------------------------------

class RunningNormalizer:
    """Online running mean/std normalizer for RL states.

    Uses Welford's online algorithm for numerically stable computation.
    """

    def __init__(self, shape: Tuple[int, ...], eps: float = 1e-6):
        self.mean = torch.zeros(shape)
        self.var = torch.ones(shape)
        self.count = 0
        self.eps = eps

    def update(self, x: Tensor) -> None:
        """Update statistics with a new batch.

        Args:
            x: Tensor of shape [B, *shape].
        """
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        total_count = self.count + batch_count
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_count / max(total_count, 1)

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / max(total_count, 1)
        new_var = m2 / max(total_count, 1)

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x: Tensor) -> Tensor:
        """Normalize a tensor using running statistics.

        Args:
            x: Tensor of shape [B, *shape] or [*shape].

        Returns:
            Normalized tensor.
        """
        return (x - self.mean) / torch.sqrt(self.var + self.eps)

    def reset(self) -> None:
        """Reset all statistics."""
        self.mean.zero_()
        self.var.fill_(1.0)
        self.count = 0


# ============================================================================
# SELF-TESTS
# ============================================================================

if __name__ == "__main__":
    import sys
    import traceback

    passed = 0
    failed = 0
    test_results = []

    def run_test(name, fn):
        global passed, failed
        try:
            fn()
            passed += 1
            test_results.append(("PASS", name))
        except Exception as e:
            failed += 1
            test_results.append(("FAIL", name, str(e)))
            traceback.print_exc()

    # ---- Vision preprocessing ----

    def test_normalize_image_uint8():
        x = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
        y = normalize_image(x)
        assert y.dtype == torch.float32
        assert y.min() >= 0.0
        assert y.max() <= 1.0
    run_test("normalize_image uint8", test_normalize_image_uint8)

    def test_normalize_image_float_noop():
        x = torch.rand(3, 32, 32)
        y = normalize_image(x)
        assert torch.allclose(x, y, atol=1e-6)
    run_test("normalize_image float [0,1] noop", test_normalize_image_float_noop)

    def test_normalize_image_high_values():
        x = torch.rand(3, 32, 32) * 255.0
        y = normalize_image(x)
        assert y.max() <= 1.0
    run_test("normalize_image high values", test_normalize_image_high_values)

    def test_normalize_image_channel_wise():
        x = torch.rand(3, 32, 32)
        y = normalize_image(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        assert y.dtype == torch.float32
        assert y.shape == (3, 32, 32)
    run_test("normalize_image channel-wise", test_normalize_image_channel_wise)

    def test_normalize_image_shape_preserved():
        x = torch.rand(1, 28, 28)
        y = normalize_image(x)
        assert y.shape == x.shape
    run_test("normalize_image shape preserved", test_normalize_image_shape_preserved)

    def test_ensure_chw_hwc():
        x = torch.rand(32, 32, 3)
        y = ensure_chw(x)
        assert y.shape == (3, 32, 32)
    run_test("ensure_chw converts HWC", test_ensure_chw_hwc)

    def test_ensure_chw_already_chw():
        x = torch.rand(3, 32, 32)
        y = ensure_chw(x)
        assert y.shape == (3, 32, 32)
    run_test("ensure_chw noop for CHW", test_ensure_chw_already_chw)

    def test_grayscale_to_rgb():
        x = torch.rand(1, 28, 28)
        y = grayscale_to_rgb(x)
        assert y.shape == (3, 28, 28)
    run_test("grayscale_to_rgb expansion", test_grayscale_to_rgb)

    def test_grayscale_to_rgb_noop():
        x = torch.rand(3, 28, 28)
        y = grayscale_to_rgb(x)
        assert y.shape == (3, 28, 28)
    run_test("grayscale_to_rgb noop for RGB", test_grayscale_to_rgb_noop)

    def test_resize_image():
        x = torch.rand(3, 32, 32)
        y = resize_image(x, (64, 64))
        assert y.shape == (3, 64, 64)
    run_test("resize_image", test_resize_image)

    def test_center_crop():
        x = torch.rand(3, 64, 64)
        y = center_crop(x, (32, 32))
        assert y.shape == (3, 32, 32)
    run_test("center_crop", test_center_crop)

    def test_preprocess_vision_dev():
        img = torch.rand(1, 28, 28)
        result = preprocess_vision(img, mode="dev")
        assert "input" in result
        assert result["input"].dtype == torch.float32
    run_test("preprocess_vision dev mode", test_preprocess_vision_dev)

    # ---- Text preprocessing ----

    def test_tokenize_text_output_dtype():
        tokens = tokenize_text("hello world", max_len=128)
        assert tokens.dtype == torch.long
    run_test("tokenize_text dtype", test_tokenize_text_output_dtype)

    def test_tokenize_text_max_len():
        tokens = tokenize_text("hello", max_len=128)
        assert tokens.shape == (128,)
    run_test("tokenize_text max_len", test_tokenize_text_max_len)

    def test_tokenize_text_padding():
        tokens = tokenize_text("hi", max_len=128)
        # BOS + 2 chars + EOS = 4 tokens, rest should be padding (0)
        assert (tokens[4:] == 0).all()
    run_test("tokenize_text padding", test_tokenize_text_padding)

    def test_tokenize_text_truncation():
        long_text = "x" * 200
        tokens = tokenize_text(long_text, max_len=128)
        assert tokens.shape == (128,)
    run_test("tokenize_text truncation", test_tokenize_text_truncation)

    def test_tokenize_text_special_tokens():
        tokens = tokenize_text("a", max_len=128, add_special_tokens=True)
        assert tokens[0] == BOS_TOKEN_ID
    run_test("tokenize_text special tokens", test_tokenize_text_special_tokens)

    def test_tokenize_text_no_special():
        tokens = tokenize_text("abc", max_len=128, add_special_tokens=False)
        assert tokens[0] != BOS_TOKEN_ID or tokens[0] == ord('a') % 256
    run_test("tokenize_text no special tokens", test_tokenize_text_no_special)

    def test_create_attention_mask():
        tokens = torch.tensor([1, 5, 10, 0, 0], dtype=torch.long)
        mask = create_attention_mask(tokens)
        assert mask.dtype == torch.bool
        assert mask.tolist() == [True, True, True, False, False]
    run_test("create_attention_mask", test_create_attention_mask)

    def test_preprocess_text():
        result = preprocess_text("hello world")
        assert "input" in result
        assert "attention_mask" in result
        assert result["input"].dtype == torch.long
        assert result["attention_mask"].dtype == torch.bool
    run_test("preprocess_text pipeline", test_preprocess_text)

    def test_batch_tokenize():
        result = batch_tokenize(["hello", "world"], max_len=64)
        assert result["input"].shape == (2, 64)
        assert result["attention_mask"].shape == (2, 64)
    run_test("batch_tokenize", test_batch_tokenize)

    # ---- Audio preprocessing ----

    def test_compute_spectrogram_shape():
        spec = compute_spectrogram(n_mels=64, T=100)
        assert spec.shape == (64, 100)
    run_test("compute_spectrogram shape", test_compute_spectrogram_shape)

    def test_compute_spectrogram_no_nan():
        spec = compute_spectrogram()
        assert not torch.isnan(spec).any()
    run_test("compute_spectrogram no NaN", test_compute_spectrogram_no_nan)

    def test_compute_spectrogram_reproducible():
        s1 = compute_spectrogram(seed=42)
        s2 = compute_spectrogram(seed=42)
        assert torch.allclose(s1, s2)
    run_test("compute_spectrogram reproducible", test_compute_spectrogram_reproducible)

    def test_log_mel_transform():
        spec = torch.rand(64, 100)
        log_spec = log_mel_transform(spec)
        assert log_spec.shape == spec.shape
        assert not torch.isnan(log_spec).any()
    run_test("log_mel_transform", test_log_mel_transform)

    def test_normalize_spectrogram_shape():
        spec = torch.randn(64, 100)
        y = normalize_spectrogram(spec)
        assert y.shape == spec.shape
    run_test("normalize_spectrogram shape", test_normalize_spectrogram_shape)

    def test_preprocess_audio():
        result = preprocess_audio(n_mels=64, T=100)
        assert "input" in result
        assert result["input"].shape == (64, 100)
    run_test("preprocess_audio pipeline", test_preprocess_audio)

    # ---- Sequence preprocessing ----

    def test_normalize_sequence_zero_mean():
        x = torch.randn(100, 16)
        y = normalize_sequence(x)
        assert abs(y.mean().item()) < 0.01
    run_test("normalize_sequence zero mean", test_normalize_sequence_zero_mean)

    def test_normalize_sequence_unit_std():
        x = torch.randn(100, 16) * 5.0 + 3.0
        y = normalize_sequence(x)
        per_feat_std = y.std(dim=0)
        assert (per_feat_std - 1.0).abs().max() < 0.1
    run_test("normalize_sequence unit std", test_normalize_sequence_unit_std)

    def test_normalize_sequence_with_mask():
        x = torch.randn(50, 8)
        mask = torch.ones(50, dtype=torch.bool)
        mask[40:] = False
        y = normalize_sequence(x, mask)
        assert y.shape == x.shape
    run_test("normalize_sequence with mask", test_normalize_sequence_with_mask)

    def test_pad_sequences():
        seqs = [torch.randn(10, 4), torch.randn(15, 4), torch.randn(8, 4)]
        padded, masks = pad_sequences(seqs)
        assert padded.shape == (3, 15, 4)
        assert masks.shape == (3, 15)
        assert masks[0, 9].item() is True
        assert masks[0, 14].item() is False
        assert masks[1, 14].item() is True
    run_test("pad_sequences", test_pad_sequences)

    def test_generate_synthetic_sinusoidal():
        seq = generate_synthetic_sequence(50, 16, pattern="sinusoidal")
        assert seq.shape == (50, 16)
        assert not torch.isnan(seq).any()
    run_test("generate_synthetic sinusoidal", test_generate_synthetic_sinusoidal)

    def test_generate_synthetic_random_walk():
        seq = generate_synthetic_sequence(50, 16, pattern="random_walk")
        assert seq.shape == (50, 16)
    run_test("generate_synthetic random_walk", test_generate_synthetic_random_walk)

    def test_generate_synthetic_step():
        seq = generate_synthetic_sequence(50, 16, pattern="step")
        assert seq.shape == (50, 16)
    run_test("generate_synthetic step", test_generate_synthetic_step)

    def test_preprocess_sequence():
        x = torch.randn(50, 16)
        result = preprocess_sequence(x)
        assert "input" in result
        assert "mask" in result
        assert result["input"].shape == (50, 16)
        assert result["mask"].shape == (50,)
    run_test("preprocess_sequence pipeline", test_preprocess_sequence)

    # ---- Validation ----

    def test_validate_vision_ok():
        data = {"input": torch.rand(3, 32, 32)}
        errors = validate_preprocessed(data, "vision")
        assert len(errors) == 0
    run_test("validate_preprocessed vision OK", test_validate_vision_ok)

    def test_validate_vision_nan():
        data = {"input": torch.tensor([float("nan"), 0.5, 0.5]).reshape(1, 1, 3).expand(1, 3, 3)}
        errors = validate_preprocessed(data, "vision")
        assert any("NaN" in e for e in errors)
    run_test("validate_preprocessed vision NaN", test_validate_vision_nan)

    def test_validate_text_ok():
        data = {"input": torch.randint(0, 256, (128,), dtype=torch.long),
                "attention_mask": torch.ones(128, dtype=torch.bool)}
        errors = validate_preprocessed(data, "text")
        assert len(errors) == 0
    run_test("validate_preprocessed text OK", test_validate_text_ok)

    def test_validate_audio_ok():
        data = {"input": torch.randn(64, 100)}
        errors = validate_preprocessed(data, "audio")
        assert len(errors) == 0
    run_test("validate_preprocessed audio OK", test_validate_audio_ok)

    def test_validate_sequence_ok():
        data = {"input": torch.randn(50, 16), "mask": torch.ones(50, dtype=torch.bool)}
        errors = validate_preprocessed(data, "sequence")
        assert len(errors) == 0
    run_test("validate_preprocessed sequence OK", test_validate_sequence_ok)

    # ---- RunningNormalizer ----

    def test_running_normalizer_basic():
        norm = RunningNormalizer(shape=(4,))
        x = torch.randn(100, 4) * 3.0 + 2.0
        norm.update(x)
        assert norm.count == 100
        y = norm.normalize(x)
        assert abs(y.mean().item()) < 0.5
    run_test("RunningNormalizer basic", test_running_normalizer_basic)

    def test_running_normalizer_reset():
        norm = RunningNormalizer(shape=(4,))
        norm.update(torch.randn(50, 4))
        norm.reset()
        assert norm.count == 0
        assert torch.allclose(norm.mean, torch.zeros(4))
    run_test("RunningNormalizer reset", test_running_normalizer_reset)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print(f"PREPROCESSING TEMPLATE SELF-TESTS: {passed} passed, {failed} failed")
    print("=" * 60)
    for result in test_results:
        status = result[0]
        name = result[1]
        extra = f" -- {result[2]}" if len(result) > 2 else ""
        print(f"  [{status}] {name}{extra}")

    sys.exit(0 if failed == 0 else 1)
