"""Canonical encoder output schema for brain_ai.

This module defines ``EncoderOutput`` — the **single source of truth** for
the data contract that every modality encoder must satisfy.  The global
workspace, downstream reasoning modules, and the ``BrainAI`` orchestrator
all consume ``EncoderOutput`` exclusively; no raw tensors should leak
across module boundaries.

Typical import::

    from brain_ai.encoders.schema import (
        EncoderOutput,
        assert_encoder_output,
        sorted_encoder_outputs,
        concat_encoder_outputs,
    )

Copy this file to ``brain_ai/encoders/schema.py`` when integrating into the
main package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Contract enforcement toggle
# ---------------------------------------------------------------------------

_CONTRACTS_ENABLED: bool = True
"""Global flag controlling whether ``assert_encoder_output`` performs
runtime checks.  Disable in tight inference loops for speed."""


def disable_contracts() -> None:
    """Turn off all ``assert_encoder_output`` runtime checks.

    Useful during latency-critical inference where the encoder outputs
    have already been validated during development / testing.
    """
    global _CONTRACTS_ENABLED
    _CONTRACTS_ENABLED = False


def enable_contracts() -> None:
    """Re-enable ``assert_encoder_output`` runtime checks."""
    global _CONTRACTS_ENABLED
    _CONTRACTS_ENABLED = True


# ---------------------------------------------------------------------------
# EncoderOutput dataclass
# ---------------------------------------------------------------------------

@dataclass
class EncoderOutput:
    """Output from any modality encoder.

    This is the shared contract that ALL encoders must satisfy.
    The workspace and all downstream modules consume this type exclusively.

    Invariants (enforced by ``assert_encoder_output``):
        - ``feats.ndim == 3``
        - ``mask.ndim == 2``
        - ``feats.shape[:2] == mask.shape``
        - ``feats.shape[-1] == workspace_dim``
        - ``feats.device == mask.device``
        - ``feats.dtype`` in ``{float32, bfloat16, float16}``
        - ``mask.dtype == torch.bool``
        - ``salience >= 0`` (if present)
        - ``spike`` batch/time dims match ``feats`` (if present)
        - No NaN or Inf in ``feats``
    """

    modality: str
    """Modality tag: ``"vision"``, ``"text"``, ``"audio"``, ``"sensors"``,
    or ``"engram"``."""

    feats: Tensor
    """(B, T, D) float — always 3-D even when T=1."""

    mask: Tensor
    """(B, T) bool — ``True`` = valid token, ``False`` = padding."""

    salience: Optional[Tensor] = None
    """(B, T) or (B, 1) competition weight used by the global workspace."""

    pos_ids: Optional[Tensor] = None
    """(B, T) int64 positional indices, when the encoder wants to supply
    explicit positions to downstream attention layers."""

    time: Optional[Tensor] = None
    """(B, T) float32 timestamps in seconds.  Optional; used for temporal
    alignment across modalities with different sampling rates."""

    spike: Optional[Tensor] = None
    """(B, T, \\*) optional spike-domain representation carried alongside
    the continuous ``feats`` for modules that can exploit it."""

    aux: Dict[str, Any] = field(default_factory=dict)
    """Modality-specific diagnostics and metadata (see standard keys below)."""


# ---------------------------------------------------------------------------
# Standard aux keys (by convention, not enforced)
# ---------------------------------------------------------------------------
# "pos_applied": bool        — whether positional encoding is already in feats
# "audio_frontend": str      — "torchaudio" | "torch_stft_fallback"
# "mel_params": dict         — {sample_rate, n_fft, hop_length, n_mels}
# "tokenization_stats": dict — {token_count, padding_ratio}
# "event_binning_stats": dict
# "spike_stats": dict        — {mean_firing_rate, sparsity}
# "lengths": Tensor          — original sequence lengths before padding
# "sensor_backend": str      — "ncps_cfc" | "builtin_cfc" | "gru"
# "text_backend": str        — "hf_transformers" | "builtin_transformer"


# ---------------------------------------------------------------------------
# Contract validator
# ---------------------------------------------------------------------------

def assert_encoder_output(
    out: EncoderOutput,
    workspace_dim: int,
    device: Optional[torch.device] = None,
) -> None:
    """Validate every invariant of *out* at runtime.

    Parameters
    ----------
    out:
        The ``EncoderOutput`` to check.
    workspace_dim:
        Expected last dimension of ``out.feats`` (e.g. 4096).
    device:
        If provided, asserts that all tensors live on this device.
        When ``None`` the device of ``out.feats`` is used as the reference.

    Raises
    ------
    AssertionError
        If any invariant is violated.  The message describes which check
        failed and includes the actual values for easy debugging.
    """
    if not _CONTRACTS_ENABLED:
        return

    ref_device = device if device is not None else out.feats.device

    # --- feats ---
    assert out.feats.ndim == 3, (
        f"feats.ndim must be 3, got {out.feats.ndim} "
        f"(shape={out.feats.shape})"
    )
    assert out.feats.shape[-1] == workspace_dim, (
        f"feats last dim must be {workspace_dim}, "
        f"got {out.feats.shape[-1]}"
    )
    assert out.feats.device == ref_device, (
        f"feats on {out.feats.device}, expected {ref_device}"
    )
    _ALLOWED_DTYPES = {torch.float32, torch.bfloat16, torch.float16}
    assert out.feats.dtype in _ALLOWED_DTYPES, (
        f"feats.dtype must be one of {_ALLOWED_DTYPES}, got {out.feats.dtype}"
    )
    assert not torch.isnan(out.feats).any(), "feats contains NaN"
    assert not torch.isinf(out.feats).any(), "feats contains Inf"

    # --- mask ---
    assert out.mask.ndim == 2, (
        f"mask.ndim must be 2, got {out.mask.ndim} "
        f"(shape={out.mask.shape})"
    )
    assert out.mask.dtype == torch.bool, (
        f"mask.dtype must be torch.bool, got {out.mask.dtype}"
    )
    assert out.mask.device == ref_device, (
        f"mask on {out.mask.device}, expected {ref_device}"
    )
    assert out.feats.shape[:2] == out.mask.shape, (
        f"feats batch/time {out.feats.shape[:2]} != "
        f"mask shape {out.mask.shape}"
    )

    # --- salience (optional) ---
    if out.salience is not None:
        assert (out.salience >= 0).all(), "salience must be non-negative"

    # --- spike (optional) ---
    if out.spike is not None:
        assert out.spike.shape[0] == out.feats.shape[0], (
            f"spike batch dim {out.spike.shape[0]} != "
            f"feats batch dim {out.feats.shape[0]}"
        )
        assert out.spike.shape[1] == out.feats.shape[1], (
            f"spike time dim {out.spike.shape[1]} != "
            f"feats time dim {out.feats.shape[1]}"
        )


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def sorted_encoder_outputs(
    outputs: Dict[str, EncoderOutput],
) -> List[EncoderOutput]:
    """Return encoder outputs sorted by modality name for deterministic ordering.

    Sorting ensures that concatenation order is reproducible across runs
    regardless of ``dict`` insertion order.
    """
    return [outputs[k] for k in sorted(outputs.keys())]


def concat_encoder_outputs(
    outputs: List[EncoderOutput],
) -> Tuple[Tensor, Tensor]:
    """Concatenate feats and masks across modalities along the time axis.

    Parameters
    ----------
    outputs:
        One or more ``EncoderOutput`` instances whose ``feats`` share the
        same batch size and last dimension.

    Returns
    -------
    all_feats:
        ``(B, sum(T_i), D)`` concatenated features.
    all_masks:
        ``(B, sum(T_i))`` concatenated masks.
    """
    all_feats = torch.cat([o.feats for o in outputs], dim=1)
    all_masks = torch.cat([o.mask for o in outputs], dim=1)
    return all_feats, all_masks
