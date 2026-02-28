"""
brain_ai/data/augmentation.py -- AugmentationPipeline with per-modality transforms.

Provides composable, deterministic augmentation transforms for vision, text, audio,
and sequence modalities. Supports four strength presets (none, light, standard, heavy)
and custom transform injection.

Key classes:
    AugmentationPipeline   -- Central augmentation controller
    RandomTransform        -- Probabilistic transform wrapper

Usage:
    pipeline = AugmentationPipeline("vision", strength="standard", seed=42)
    train_tfm = pipeline.get_train_transforms()
    eval_tfm = pipeline.get_eval_transforms()
    augmented = train_tfm(image_tensor)
"""

from __future__ import annotations

import math
import logging
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SECTION 1: Strength presets per modality
# ---------------------------------------------------------------------------

VISION_PRESETS: Dict[str, Dict[str, Any]] = {
    "none": {
        "crop_padding": 0, "flip_prob": 0.0,
        "brightness": 0.0, "contrast": 0.0, "saturation": 0.0, "hue": 0.0,
        "cutout_holes": 0, "cutout_size": 0.0,
        "randaug_n": 0, "randaug_m": 0,
    },
    "light": {
        "crop_padding": 2, "flip_prob": 0.3,
        "brightness": 0.1, "contrast": 0.1, "saturation": 0.1, "hue": 0.02,
        "cutout_holes": 1, "cutout_size": 0.125,
        "randaug_n": 1, "randaug_m": 5,
    },
    "standard": {
        "crop_padding": 4, "flip_prob": 0.5,
        "brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.05,
        "cutout_holes": 1, "cutout_size": 0.25,
        "randaug_n": 2, "randaug_m": 9,
    },
    "heavy": {
        "crop_padding": 8, "flip_prob": 0.5,
        "brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1,
        "cutout_holes": 2, "cutout_size": 0.25,
        "randaug_n": 3, "randaug_m": 15,
    },
}

TEXT_PRESETS: Dict[str, Dict[str, Any]] = {
    "none": {"token_dropout": 0.0, "synonym_prob": 0.0, "shuffle_window": 0},
    "light": {"token_dropout": 0.05, "synonym_prob": 0.05, "shuffle_window": 2},
    "standard": {"token_dropout": 0.10, "synonym_prob": 0.10, "shuffle_window": 3},
    "heavy": {"token_dropout": 0.20, "synonym_prob": 0.15, "shuffle_window": 5},
}

AUDIO_PRESETS: Dict[str, Dict[str, Any]] = {
    "none": {
        "freq_mask_width": 0, "num_freq_masks": 0,
        "time_mask_width": 0, "num_time_masks": 0,
        "time_stretch_range": (1.0, 1.0),
        "noise_level": 0.0, "pitch_shift_max": 0,
    },
    "light": {
        "freq_mask_width": 10, "num_freq_masks": 1,
        "time_mask_width": 10, "num_time_masks": 1,
        "time_stretch_range": (0.95, 1.05),
        "noise_level": 0.005, "pitch_shift_max": 2,
    },
    "standard": {
        "freq_mask_width": 20, "num_freq_masks": 2,
        "time_mask_width": 30, "num_time_masks": 2,
        "time_stretch_range": (0.9, 1.1),
        "noise_level": 0.01, "pitch_shift_max": 4,
    },
    "heavy": {
        "freq_mask_width": 30, "num_freq_masks": 3,
        "time_mask_width": 50, "num_time_masks": 3,
        "time_stretch_range": (0.8, 1.25),
        "noise_level": 0.02, "pitch_shift_max": 8,
    },
}

SEQUENCE_PRESETS: Dict[str, Dict[str, Any]] = {
    "none": {"jitter_sigma": 0.0, "noise_std": 0.0, "subsample_min": 1.0, "feat_dropout": 0.0},
    "light": {"jitter_sigma": 0.5, "noise_std": 0.01, "subsample_min": 0.9, "feat_dropout": 0.05},
    "standard": {"jitter_sigma": 1.0, "noise_std": 0.03, "subsample_min": 0.7, "feat_dropout": 0.1},
    "heavy": {"jitter_sigma": 2.0, "noise_std": 0.05, "subsample_min": 0.5, "feat_dropout": 0.2},
}


# ---------------------------------------------------------------------------
# SECTION 2: Individual transform functions
# ---------------------------------------------------------------------------

# -- Vision transforms --

def random_crop_with_padding(x: Tensor, padding: int, gen: Optional[torch.Generator] = None) -> Tensor:
    """Random crop with reflection padding. x: [C, H, W]."""
    if padding <= 0:
        return x
    C, H, W = x.shape
    padded = torch.nn.functional.pad(x, [padding] * 4, mode="reflect")
    top = torch.randint(0, 2 * padding + 1, (1,), generator=gen).item()
    left = torch.randint(0, 2 * padding + 1, (1,), generator=gen).item()
    return padded[:, top:top + H, left:left + W]


def random_horizontal_flip(x: Tensor, prob: float, gen: Optional[torch.Generator] = None) -> Tensor:
    """Flip image horizontally with probability prob. x: [C, H, W]."""
    if prob <= 0.0:
        return x
    if torch.rand(1, generator=gen).item() < prob:
        return x.flip(-1)
    return x


def color_jitter(x: Tensor, brightness: float, contrast: float,
                 saturation: float, hue: float,
                 gen: Optional[torch.Generator] = None) -> Tensor:
    """Apply color jitter. x: [C, H, W], only for C >= 3."""
    if x.shape[0] < 3:
        return x
    if brightness <= 0 and contrast <= 0 and saturation <= 0 and hue <= 0:
        return x

    # Brightness
    if brightness > 0:
        factor = 1.0 + (torch.rand(1, generator=gen).item() * 2 - 1) * brightness
        x = x * factor

    # Contrast
    if contrast > 0:
        factor = 1.0 + (torch.rand(1, generator=gen).item() * 2 - 1) * contrast
        mean = x.mean()
        x = (x - mean) * factor + mean

    # Simple saturation adjustment
    if saturation > 0 and x.shape[0] >= 3:
        factor = 1.0 + (torch.rand(1, generator=gen).item() * 2 - 1) * saturation
        gray = x[:3].mean(dim=0, keepdim=True)
        x = torch.lerp(gray.expand_as(x[:3]), x[:3], factor)
        if x.shape[0] > 3:
            x = torch.cat([x, x[3:]], dim=0)

    return x.clamp(0, 1)


def cutout(x: Tensor, num_holes: int, hole_size_frac: float,
           gen: Optional[torch.Generator] = None) -> Tensor:
    """Apply cutout (random erasing) to image. x: [C, H, W]."""
    if num_holes <= 0 or hole_size_frac <= 0:
        return x
    C, H, W = x.shape
    result = x.clone()
    hole_h = int(H * hole_size_frac)
    hole_w = int(W * hole_size_frac)
    for _ in range(num_holes):
        cy = torch.randint(0, H, (1,), generator=gen).item()
        cx = torch.randint(0, W, (1,), generator=gen).item()
        y1 = max(0, cy - hole_h // 2)
        y2 = min(H, cy + hole_h // 2)
        x1 = max(0, cx - hole_w // 2)
        x2 = min(W, cx + hole_w // 2)
        result[:, y1:y2, x1:x2] = 0.0
    return result


def normalize_image(x: Tensor, mean: Optional[List[float]] = None,
                    std: Optional[List[float]] = None) -> Tensor:
    """Normalize image tensor. x: [C, H, W]."""
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    elif x.max() > 1.0:
        x = x / 255.0
    if mean is not None and std is not None:
        m = torch.tensor(mean, dtype=x.dtype).view(-1, 1, 1)
        s = torch.tensor(std, dtype=x.dtype).view(-1, 1, 1)
        x = (x - m) / s
    return x


# -- Text transforms --

def token_dropout(tokens: Tensor, prob: float, mask_token_id: int = 4,
                  gen: Optional[torch.Generator] = None) -> Tensor:
    """Randomly replace tokens with mask token. tokens: [seq_len] int64."""
    if prob <= 0:
        return tokens
    result = tokens.clone()
    # Do not modify padding tokens (id=0)
    non_pad = tokens != 0
    drop_mask = torch.rand(tokens.shape, generator=gen) < prob
    replace = non_pad & drop_mask
    result[replace] = mask_token_id
    return result


def synonym_replacement(tokens: Tensor, prob: float, vocab_size: int = 256,
                        gen: Optional[torch.Generator] = None) -> Tensor:
    """Replace tokens with nearby tokens (simple synonym approximation)."""
    if prob <= 0:
        return tokens
    result = tokens.clone()
    non_pad = tokens != 0
    replace_mask = torch.rand(tokens.shape, generator=gen) < prob
    replace = non_pad & replace_mask
    offsets = torch.randint(-1, 2, tokens.shape, generator=gen)
    replaced = (tokens + offsets).clamp(1, vocab_size - 1)
    result[replace] = replaced[replace]
    return result


def token_shuffle(tokens: Tensor, window_size: int,
                  gen: Optional[torch.Generator] = None) -> Tensor:
    """Shuffle tokens within a local window."""
    if window_size <= 1:
        return tokens
    result = tokens.clone()
    seq_len = tokens.shape[0]
    for i in range(0, seq_len, window_size):
        end = min(i + window_size, seq_len)
        segment = result[i:end]
        # Only shuffle non-padding tokens
        non_pad_mask = segment != 0
        if non_pad_mask.sum() > 1:
            indices = torch.randperm(non_pad_mask.sum().item(), generator=gen)
            non_pad_vals = segment[non_pad_mask]
            segment[non_pad_mask] = non_pad_vals[indices]
            result[i:end] = segment
    return result


# -- Audio transforms --

def spec_augment_freq(x: Tensor, mask_width: int, num_masks: int,
                      gen: Optional[torch.Generator] = None) -> Tensor:
    """Frequency masking for SpecAugment. x: [n_mels, T]."""
    if mask_width <= 0 or num_masks <= 0:
        return x
    result = x.clone()
    n_mels = x.shape[0]
    for _ in range(num_masks):
        f = torch.randint(0, min(mask_width, n_mels), (1,), generator=gen).item()
        f0 = torch.randint(0, max(1, n_mels - f), (1,), generator=gen).item()
        result[f0:f0 + f, :] = 0.0
    return result


def spec_augment_time(x: Tensor, mask_width: int, num_masks: int,
                      gen: Optional[torch.Generator] = None) -> Tensor:
    """Time masking for SpecAugment. x: [n_mels, T]."""
    if mask_width <= 0 or num_masks <= 0:
        return x
    result = x.clone()
    T = x.shape[1]
    for _ in range(num_masks):
        t = torch.randint(0, min(mask_width, T), (1,), generator=gen).item()
        t0 = torch.randint(0, max(1, T - t), (1,), generator=gen).item()
        result[:, t0:t0 + t] = 0.0
    return result


def time_stretch(x: Tensor, rate_range: Tuple[float, float],
                 gen: Optional[torch.Generator] = None) -> Tensor:
    """Stretch time axis by a random factor, then resample. x: [n_mels, T]."""
    lo, hi = rate_range
    if abs(lo - hi) < 1e-6:
        return x
    rate = lo + torch.rand(1, generator=gen).item() * (hi - lo)
    n_mels, T = x.shape
    new_T = max(1, int(T * rate))
    # Use interpolate for resampling
    stretched = torch.nn.functional.interpolate(
        x.unsqueeze(0).unsqueeze(0), size=(n_mels, new_T),
        mode="bilinear", align_corners=False
    ).squeeze(0).squeeze(0)
    # Resample back to original length
    resampled = torch.nn.functional.interpolate(
        stretched.unsqueeze(0).unsqueeze(0), size=(n_mels, T),
        mode="bilinear", align_corners=False
    ).squeeze(0).squeeze(0)
    return resampled


def pitch_shift(x: Tensor, max_shift: int,
                gen: Optional[torch.Generator] = None) -> Tensor:
    """Shift frequency bins up/down. x: [n_mels, T]."""
    if max_shift <= 0:
        return x
    shift = torch.randint(-max_shift, max_shift + 1, (1,), generator=gen).item()
    if shift == 0:
        return x
    return torch.roll(x, shift, dims=0)


def audio_noise_injection(x: Tensor, noise_level: float,
                          gen: Optional[torch.Generator] = None) -> Tensor:
    """Add Gaussian noise to spectrogram. x: [n_mels, T]."""
    if noise_level <= 0:
        return x
    noise = torch.randn(x.shape, generator=gen) * noise_level
    return x + noise


def normalize_spectrogram(x: Tensor) -> Tensor:
    """Per-channel (per-mel-bin) normalization. x: [n_mels, T]."""
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
    return (x - mean) / std


# -- Sequence transforms --

def temporal_jitter(x: Tensor, sigma: float,
                    gen: Optional[torch.Generator] = None) -> Tensor:
    """Add small random jitter to temporal positions. x: [T, D]."""
    if sigma <= 0:
        return x
    T, D = x.shape
    indices = torch.arange(T, dtype=torch.float32)
    offsets = torch.randn(T, generator=gen) * sigma if gen else torch.randn(T) * sigma
    new_indices = (indices + offsets).clamp(0, T - 1)
    # Linear interpolation
    lower = new_indices.long().clamp(0, T - 2)
    upper = (lower + 1).clamp(0, T - 1)
    frac = (new_indices - lower.float()).unsqueeze(-1)
    return x[lower] * (1 - frac) + x[upper] * frac


def sequence_noise(x: Tensor, noise_std: float,
                   gen: Optional[torch.Generator] = None) -> Tensor:
    """Add Gaussian noise to all features at all timesteps. x: [T, D]."""
    if noise_std <= 0:
        return x
    noise = torch.randn(x.shape, generator=gen) * noise_std if gen else torch.randn_like(x) * noise_std
    return x + noise


def subsequence_sample(x: Tensor, min_ratio: float, max_ratio: float,
                       gen: Optional[torch.Generator] = None) -> Tensor:
    """Sample a contiguous subsequence. x: [T, D]."""
    if min_ratio >= 1.0:
        return x
    T = x.shape[0]
    ratio = min_ratio + torch.rand(1, generator=gen).item() * (max_ratio - min_ratio)
    new_T = max(1, int(T * ratio))
    start = torch.randint(0, max(1, T - new_T + 1), (1,), generator=gen).item()
    return x[start:start + new_T]


def feature_dropout(x: Tensor, prob: float,
                    gen: Optional[torch.Generator] = None) -> Tensor:
    """Zero out entire feature dimensions. x: [T, D]."""
    if prob <= 0:
        return x
    D = x.shape[-1]
    drop_mask = torch.rand(D, generator=gen) < prob if gen else torch.rand(D) < prob
    result = x.clone()
    result[:, drop_mask] = 0.0
    return result


def normalize_sequence_fn(x: Tensor) -> Tensor:
    """Per-feature normalization to zero mean, unit std. x: [T, D]."""
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp(min=1e-6)
    return (x - mean) / std


# ---------------------------------------------------------------------------
# SECTION 3: RandomTransform wrapper
# ---------------------------------------------------------------------------

class RandomTransform:
    """Wraps a transform to apply it with a given probability."""

    def __init__(self, transform: Callable, prob: float = 1.0):
        self.transform = transform
        self.prob = prob

    def __call__(self, x: Tensor, gen: Optional[torch.Generator] = None) -> Tensor:
        if self.prob >= 1.0 or torch.rand(1, generator=gen).item() < self.prob:
            sig = inspect.signature(self.transform)
            if "gen" in sig.parameters:
                return self.transform(x, gen=gen)
            return self.transform(x)
        return x


# ---------------------------------------------------------------------------
# SECTION 4: AugmentationPipeline
# ---------------------------------------------------------------------------

class AugmentationPipeline:
    """Central augmentation controller with modality-aware transforms.

    Provides composable, deterministic augmentation chains for training and
    evaluation. Supports vision, text, audio, sequence, and rl modalities.
    """

    VALID_MODALITIES = {"vision", "text", "audio", "sequence", "rl"}
    VALID_STRENGTHS = {"none", "light", "standard", "heavy"}

    def __init__(self, modality: str, strength: str = "standard",
                 seed: Optional[int] = None):
        if modality not in self.VALID_MODALITIES:
            raise ValueError(f"modality must be one of {self.VALID_MODALITIES}, got '{modality}'")
        if strength not in self.VALID_STRENGTHS:
            raise ValueError(f"strength must be one of {self.VALID_STRENGTHS}, got '{strength}'")

        self.modality = modality
        self.strength = strength
        self.seed = seed
        self._gen: Optional[torch.Generator] = None
        if seed is not None:
            self._gen = torch.Generator()
            self._gen.manual_seed(seed)

        self._custom_transforms: List[Tuple[Callable, float]] = []

    def set_seed(self, seed: int) -> None:
        """Reset the random seed for all stochastic transforms."""
        self.seed = seed
        self._gen = torch.Generator()
        self._gen.manual_seed(seed)

    def add_transform(self, transform: Callable, prob: float = 1.0) -> None:
        """Append a custom transform to the pipeline."""
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"prob must be in [0, 1], got {prob}")
        self._custom_transforms.append((transform, prob))

    def get_train_transforms(self) -> Callable:
        """Return a composed callable for training augmentation."""
        if self.modality == "vision":
            return self._vision_train()
        elif self.modality == "text":
            return self._text_train()
        elif self.modality == "audio":
            return self._audio_train()
        elif self.modality == "sequence":
            return self._sequence_train()
        elif self.modality == "rl":
            return self._rl_train()
        raise ValueError(f"Unknown modality: {self.modality}")

    def get_eval_transforms(self) -> Callable:
        """Return a composed callable for validation/test (deterministic, no randomness)."""
        if self.modality == "vision":
            return self._vision_eval()
        elif self.modality == "text":
            return self._text_eval()
        elif self.modality == "audio":
            return self._audio_eval()
        elif self.modality == "sequence":
            return self._sequence_eval()
        elif self.modality == "rl":
            return self._rl_eval()
        raise ValueError(f"Unknown modality: {self.modality}")

    def _apply_custom(self, x: Tensor) -> Tensor:
        """Apply custom transforms with their probabilities."""
        gen = self._gen
        for tfm, prob in self._custom_transforms:
            if prob >= 1.0 or torch.rand(1, generator=gen).item() < prob:
                x = tfm(x)
        return x

    # ---- Vision ----

    def _vision_train(self) -> Callable:
        p = VISION_PRESETS[self.strength]
        gen = self._gen
        custom_transforms = self._custom_transforms

        def transform(x: Tensor) -> Tensor:
            # 1. RandomCrop
            x = random_crop_with_padding(x, p["crop_padding"], gen)
            # 2. HorizontalFlip
            x = random_horizontal_flip(x, p["flip_prob"], gen)
            # 3. ColorJitter
            x = color_jitter(x, p["brightness"], p["contrast"],
                             p["saturation"], p["hue"], gen)
            # 4. Cutout
            x = cutout(x, p["cutout_holes"], p["cutout_size"], gen)
            # 5. Normalize to [0, 1]
            x = normalize_image(x)
            # 6. Custom transforms
            for tfm, prob in custom_transforms:
                if prob >= 1.0 or torch.rand(1, generator=gen).item() < prob:
                    x = tfm(x)
            return x
        return transform

    def _vision_eval(self) -> Callable:
        def transform(x: Tensor) -> Tensor:
            return normalize_image(x)
        return transform

    # ---- Text ----

    def _text_train(self) -> Callable:
        p = TEXT_PRESETS[self.strength]
        gen = self._gen

        def transform(x: Tensor) -> Tensor:
            x = token_dropout(x, p["token_dropout"], gen=gen)
            x = synonym_replacement(x, p["synonym_prob"], gen=gen)
            x = token_shuffle(x, p["shuffle_window"], gen=gen)
            return x
        return transform

    def _text_eval(self) -> Callable:
        def transform(x: Tensor) -> Tensor:
            return x  # No transform for text during validation/test
        return transform

    # ---- Audio ----

    def _audio_train(self) -> Callable:
        p = AUDIO_PRESETS[self.strength]
        gen = self._gen
        custom_transforms = self._custom_transforms

        def transform(x: Tensor) -> Tensor:
            # 1. Time stretch
            x = time_stretch(x, p["time_stretch_range"], gen)
            # 2. Pitch shift
            x = pitch_shift(x, p["pitch_shift_max"], gen)
            # 3. SpecAugment freq
            x = spec_augment_freq(x, p["freq_mask_width"], p["num_freq_masks"], gen)
            # 4. SpecAugment time
            x = spec_augment_time(x, p["time_mask_width"], p["num_time_masks"], gen)
            # 5. Noise injection
            x = audio_noise_injection(x, p["noise_level"], gen)
            # 6. Normalize
            x = normalize_spectrogram(x)
            # 7. Custom transforms
            for tfm, prob in custom_transforms:
                if prob >= 1.0 or torch.rand(1, generator=gen).item() < prob:
                    x = tfm(x)
            return x
        return transform

    def _audio_eval(self) -> Callable:
        def transform(x: Tensor) -> Tensor:
            return normalize_spectrogram(x)
        return transform

    # ---- Sequence ----

    def _sequence_train(self) -> Callable:
        p = SEQUENCE_PRESETS[self.strength]
        gen = self._gen
        custom_transforms = self._custom_transforms

        def transform(x: Tensor) -> Tensor:
            # 1. Subsequence sampling (only for standard+heavy)
            if p["subsample_min"] < 1.0:
                x = subsequence_sample(x, p["subsample_min"], 1.0, gen)
            # 2. Temporal jittering
            x = temporal_jitter(x, p["jitter_sigma"], gen)
            # 3. Gaussian noise
            x = sequence_noise(x, p["noise_std"], gen)
            # 4. Feature dropout
            x = feature_dropout(x, p["feat_dropout"], gen)
            # 5. Normalize
            x = normalize_sequence_fn(x)
            # 6. Custom transforms
            for tfm, prob in custom_transforms:
                if prob >= 1.0 or torch.rand(1, generator=gen).item() < prob:
                    x = tfm(x)
            return x
        return transform

    def _sequence_eval(self) -> Callable:
        def transform(x: Tensor) -> Tensor:
            return normalize_sequence_fn(x)
        return transform

    # ---- RL ----

    def _rl_train(self) -> Callable:
        """RL episodes should NOT be augmented to preserve trajectory integrity."""
        def transform(x: Tensor) -> Tensor:
            return x
        return transform

    def _rl_eval(self) -> Callable:
        def transform(x: Tensor) -> Tensor:
            return x
        return transform


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

    # ---- Pipeline construction ----

    def test_pipeline_valid_modalities():
        for mod in ("vision", "text", "audio", "sequence", "rl"):
            p = AugmentationPipeline(mod, strength="standard")
            assert p.modality == mod
    run_test("Pipeline valid modalities", test_pipeline_valid_modalities)

    def test_pipeline_invalid_modality():
        try:
            AugmentationPipeline("smell")
            assert False, "Should raise ValueError"
        except ValueError:
            pass
    run_test("Pipeline invalid modality", test_pipeline_invalid_modality)

    def test_pipeline_valid_strengths():
        for s in ("none", "light", "standard", "heavy"):
            p = AugmentationPipeline("vision", strength=s)
            assert p.strength == s
    run_test("Pipeline valid strengths", test_pipeline_valid_strengths)

    def test_pipeline_invalid_strength():
        try:
            AugmentationPipeline("vision", strength="extreme")
            assert False, "Should raise ValueError"
        except ValueError:
            pass
    run_test("Pipeline invalid strength", test_pipeline_invalid_strength)

    # ---- Vision augmentation ----

    def test_vision_train_shape():
        p = AugmentationPipeline("vision", strength="standard")
        tfm = p.get_train_transforms()
        x = torch.rand(3, 32, 32)
        y = tfm(x)
        assert y.shape == x.shape
    run_test("Vision train preserves shape", test_vision_train_shape)

    def test_vision_eval_shape():
        p = AugmentationPipeline("vision", strength="standard")
        tfm = p.get_eval_transforms()
        x = torch.rand(3, 32, 32)
        y = tfm(x)
        assert y.shape == x.shape
    run_test("Vision eval preserves shape", test_vision_eval_shape)

    def test_vision_eval_deterministic():
        p = AugmentationPipeline("vision", strength="standard")
        tfm = p.get_eval_transforms()
        x = torch.rand(1, 28, 28)
        results = [tfm(x.clone()) for _ in range(5)]
        for r in results[1:]:
            assert torch.allclose(results[0], r)
    run_test("Vision eval deterministic", test_vision_eval_deterministic)

    def test_vision_seed_determinism():
        x = torch.rand(3, 32, 32)
        p1 = AugmentationPipeline("vision", strength="standard", seed=42)
        y1 = p1.get_train_transforms()(x.clone())
        p2 = AugmentationPipeline("vision", strength="standard", seed=42)
        y2 = p2.get_train_transforms()(x.clone())
        assert torch.allclose(y1, y2)
    run_test("Vision seeded determinism", test_vision_seed_determinism)

    def test_vision_different_seeds():
        x = torch.rand(3, 32, 32)
        p1 = AugmentationPipeline("vision", strength="heavy", seed=1)
        y1 = p1.get_train_transforms()(x.clone())
        p2 = AugmentationPipeline("vision", strength="heavy", seed=999)
        y2 = p2.get_train_transforms()(x.clone())
        assert not torch.equal(y1, y2)
    run_test("Vision different seeds differ", test_vision_different_seeds)

    def test_vision_none_no_random_change():
        p = AugmentationPipeline("vision", strength="none")
        tfm = p.get_train_transforms()
        x = torch.rand(1, 28, 28)
        y = tfm(x.clone())
        # With none strength, only normalization is applied
        assert y.shape == x.shape
    run_test("Vision none strength minimal change", test_vision_none_no_random_change)

    def test_vision_preserves_dtype():
        p = AugmentationPipeline("vision", strength="standard")
        x = torch.rand(3, 32, 32)
        y = p.get_train_transforms()(x)
        assert y.dtype == torch.float32
    run_test("Vision preserves float32 dtype", test_vision_preserves_dtype)

    def test_vision_grayscale():
        p = AugmentationPipeline("vision", strength="standard", seed=42)
        x = torch.rand(1, 28, 28)
        y = p.get_train_transforms()(x)
        assert y.shape == (1, 28, 28)
    run_test("Vision grayscale (C=1) works", test_vision_grayscale)

    # ---- Text augmentation ----

    def test_text_train_shape():
        p = AugmentationPipeline("text", strength="standard")
        tfm = p.get_train_transforms()
        x = torch.randint(1, 256, (128,), dtype=torch.long)
        y = tfm(x)
        assert y.shape == x.shape
    run_test("Text train preserves shape", test_text_train_shape)

    def test_text_eval_deterministic():
        p = AugmentationPipeline("text", strength="standard")
        tfm = p.get_eval_transforms()
        x = torch.randint(1, 256, (128,), dtype=torch.long)
        y1 = tfm(x.clone())
        y2 = tfm(x.clone())
        assert torch.equal(y1, y2)
    run_test("Text eval deterministic", test_text_eval_deterministic)

    def test_text_preserves_padding():
        p = AugmentationPipeline("text", strength="heavy", seed=42)
        tfm = p.get_train_transforms()
        x = torch.randint(1, 256, (128,), dtype=torch.long)
        x[100:] = 0  # pad last 28 tokens
        y = tfm(x.clone())
        # Padding positions should remain 0 (token_dropout, synonym_replacement skip padding)
        assert (y[100:] == 0).all(), "Padding tokens should remain 0"
    run_test("Text augmentation preserves padding", test_text_preserves_padding)

    def test_text_preserves_dtype():
        p = AugmentationPipeline("text", strength="standard")
        x = torch.randint(1, 256, (64,), dtype=torch.long)
        y = p.get_train_transforms()(x)
        assert y.dtype == torch.long
    run_test("Text preserves int64 dtype", test_text_preserves_dtype)

    def test_text_none_identity():
        p = AugmentationPipeline("text", strength="none")
        x = torch.randint(1, 256, (64,), dtype=torch.long)
        y = p.get_train_transforms()(x.clone())
        assert torch.equal(x, y)
    run_test("Text none strength is identity", test_text_none_identity)

    def test_text_seed_determinism():
        x = torch.randint(1, 256, (128,), dtype=torch.long)
        p1 = AugmentationPipeline("text", strength="standard", seed=42)
        y1 = p1.get_train_transforms()(x.clone())
        p2 = AugmentationPipeline("text", strength="standard", seed=42)
        y2 = p2.get_train_transforms()(x.clone())
        assert torch.equal(y1, y2)
    run_test("Text seeded determinism", test_text_seed_determinism)

    def test_text_valid_token_range():
        p = AugmentationPipeline("text", strength="heavy", seed=42)
        x = torch.randint(1, 256, (128,), dtype=torch.long)
        y = p.get_train_transforms()(x)
        assert y.min() >= 0
        assert y.max() < 256
    run_test("Text tokens in valid range after augmentation", test_text_valid_token_range)

    # ---- Audio augmentation ----

    def test_audio_train_shape():
        p = AugmentationPipeline("audio", strength="standard", seed=42)
        tfm = p.get_train_transforms()
        x = torch.randn(64, 100)
        y = tfm(x)
        assert y.shape == x.shape
    run_test("Audio train preserves shape", test_audio_train_shape)

    def test_audio_eval_deterministic():
        p = AugmentationPipeline("audio", strength="standard")
        tfm = p.get_eval_transforms()
        x = torch.randn(64, 100)
        y1 = tfm(x.clone())
        y2 = tfm(x.clone())
        assert torch.allclose(y1, y2)
    run_test("Audio eval deterministic", test_audio_eval_deterministic)

    def test_audio_preserves_dtype():
        p = AugmentationPipeline("audio", strength="heavy", seed=42)
        x = torch.randn(64, 100)
        y = p.get_train_transforms()(x)
        assert y.dtype == torch.float32
    run_test("Audio preserves float32 dtype", test_audio_preserves_dtype)

    def test_audio_seed_determinism():
        x = torch.randn(64, 100)
        p1 = AugmentationPipeline("audio", strength="standard", seed=42)
        y1 = p1.get_train_transforms()(x.clone())
        p2 = AugmentationPipeline("audio", strength="standard", seed=42)
        y2 = p2.get_train_transforms()(x.clone())
        assert torch.allclose(y1, y2)
    run_test("Audio seeded determinism", test_audio_seed_determinism)

    def test_audio_none_no_augmentation():
        p = AugmentationPipeline("audio", strength="none", seed=42)
        x = torch.randn(64, 100)
        y = p.get_train_transforms()(x.clone())
        # None strength: only normalization
        expected = normalize_spectrogram(x)
        assert torch.allclose(y, expected)
    run_test("Audio none = normalize only", test_audio_none_no_augmentation)

    # ---- Sequence augmentation ----

    def test_seq_train_shape_none():
        p = AugmentationPipeline("sequence", strength="none")
        x = torch.randn(50, 16)
        y = p.get_train_transforms()(x)
        assert y.shape == x.shape
    run_test("Sequence train preserves shape (none)", test_seq_train_shape_none)

    def test_seq_eval_deterministic():
        p = AugmentationPipeline("sequence", strength="standard")
        tfm = p.get_eval_transforms()
        x = torch.randn(50, 16)
        y1 = tfm(x.clone())
        y2 = tfm(x.clone())
        assert torch.allclose(y1, y2)
    run_test("Sequence eval deterministic", test_seq_eval_deterministic)

    def test_seq_seed_determinism():
        x = torch.randn(50, 16)
        p1 = AugmentationPipeline("sequence", strength="light", seed=42)
        y1 = p1.get_train_transforms()(x.clone())
        p2 = AugmentationPipeline("sequence", strength="light", seed=42)
        y2 = p2.get_train_transforms()(x.clone())
        assert torch.allclose(y1, y2)
    run_test("Sequence seeded determinism", test_seq_seed_determinism)

    def test_seq_preserves_dtype():
        p = AugmentationPipeline("sequence", strength="standard")
        x = torch.randn(50, 16)
        y = p.get_train_transforms()(x)
        assert y.dtype == torch.float32
    run_test("Sequence preserves float32 dtype", test_seq_preserves_dtype)

    # ---- RL augmentation ----

    def test_rl_no_modification():
        p = AugmentationPipeline("rl", strength="heavy")
        x = torch.randn(4)
        y_train = p.get_train_transforms()(x.clone())
        y_eval_out = p.get_eval_transforms()(x.clone())
        assert torch.equal(x, y_train)
        assert torch.equal(x, y_eval_out)
    run_test("RL no modification (preserves trajectory)", test_rl_no_modification)

    # ---- Custom transforms ----

    def test_add_custom_transform():
        p = AugmentationPipeline("vision", strength="none")
        call_count = [0]

        def my_tfm(x):
            call_count[0] += 1
            return x * 0.5

        p.add_transform(my_tfm, prob=1.0)
        tfm = p.get_train_transforms()
        x = torch.ones(1, 8, 8)
        y = tfm(x.clone())
        assert call_count[0] == 1
    run_test("Custom transform is called", test_add_custom_transform)

    def test_add_transform_invalid_prob():
        p = AugmentationPipeline("vision", strength="none")
        try:
            p.add_transform(lambda x: x, prob=1.5)
            assert False, "Should raise ValueError"
        except ValueError:
            pass
    run_test("add_transform rejects invalid prob", test_add_transform_invalid_prob)

    # ---- set_seed ----

    def test_set_seed_resets():
        p = AugmentationPipeline("vision", strength="standard", seed=42)
        x = torch.rand(3, 32, 32)
        y1 = p.get_train_transforms()(x.clone())
        p.set_seed(42)
        y2 = p.get_train_transforms()(x.clone())
        assert torch.allclose(y1, y2)
    run_test("set_seed resets determinism", test_set_seed_resets)

    # ---- Individual transform tests ----

    def test_random_crop_padding():
        x = torch.rand(1, 8, 8)
        y = random_crop_with_padding(x, 2)
        assert y.shape == (1, 8, 8)
    run_test("random_crop_with_padding shape", test_random_crop_padding)

    def test_horizontal_flip_prob_zero():
        x = torch.rand(1, 4, 4)
        y = random_horizontal_flip(x, 0.0)
        assert torch.equal(x, y)
    run_test("horizontal_flip prob=0 identity", test_horizontal_flip_prob_zero)

    def test_cutout_shape():
        x = torch.rand(3, 32, 32)
        y = cutout(x, 1, 0.25)
        assert y.shape == x.shape
    run_test("cutout preserves shape", test_cutout_shape)

    def test_spec_augment_freq_shape():
        x = torch.randn(64, 100)
        y = spec_augment_freq(x, 10, 2)
        assert y.shape == x.shape
    run_test("spec_augment_freq preserves shape", test_spec_augment_freq_shape)

    def test_spec_augment_time_shape():
        x = torch.randn(64, 100)
        y = spec_augment_time(x, 10, 2)
        assert y.shape == x.shape
    run_test("spec_augment_time preserves shape", test_spec_augment_time_shape)

    def test_time_stretch_shape():
        x = torch.randn(64, 100)
        y = time_stretch(x, (0.9, 1.1))
        assert y.shape == x.shape
    run_test("time_stretch preserves shape", test_time_stretch_shape)

    def test_temporal_jitter_shape():
        x = torch.randn(50, 16)
        y = temporal_jitter(x, 1.0)
        assert y.shape == x.shape
    run_test("temporal_jitter preserves shape", test_temporal_jitter_shape)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print(f"AUGMENTATION TEMPLATE SELF-TESTS: {passed} passed, {failed} failed")
    print("=" * 60)
    for result in test_results:
        status = result[0]
        name = result[1]
        extra = f" -- {result[2]}" if len(result) > 2 else ""
        print(f"  [{status}] {name}{extra}")

    sys.exit(0 if failed == 0 else 1)
