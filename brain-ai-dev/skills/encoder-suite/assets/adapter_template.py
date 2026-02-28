"""
brain_ai/encoders/adapters.py — Modality adapter layer.

Adapters sit between each encoder's native representation and the shared
EncoderOutput contract.  Every encoder today returns (B, output_dim) — a single
pooled vector.  The adapters intercept BEFORE final pooling and emit per-token
features (B, T, D) instead, giving the global workspace fine-grained tokens to
compete over.

Usage:
    from brain_ai.encoders.adapters import (
        EncoderOutput,
        VisionAdapter, TextAdapter, AudioAdapter, SensorAdapter, EngramAdapter,
        ADAPTER_REGISTRY, create_adapter, assert_encoder_output,
    )

Canonical data-flow:
    raw_encoder_output (modality-specific) -> ModalityAdapter.forward()
    -> EncoderOutput(feats=(B,T,D), mask=(B,T), ...)

NOTE: This is a TEMPLATE file.  Integration points where the actual encoder
internals must be wired in are marked with ``# TODO:`` comments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# EncoderOutput — single source of truth
# ---------------------------------------------------------------------------

@dataclass
class EncoderOutput:
    """Output from any modality encoder — the shared contract.

    Every adapter's ``forward`` must return an instance of this dataclass.
    The global workspace, HTM layer, and downstream modules consume
    ``feats`` and ``mask`` without any modality-specific branching.

    Shape rules:
        feats   — (B, T, D)    float,   always 3-D even if T=1
        mask    — (B, T)        bool,    True=valid, False=padding
        salience — (B, T) or (B, 1) float, non-negative competition weight
        pos_ids — (B, T)        int64,   optional positional indices
        time    — (B, T)        float32, optional real-valued timestamps
        spike   — (B, T, *)     optional spike-domain tensor
        aux     — dict          never consumed by forward path, observability only
    """
    modality: str                                # "vision" | "text" | "audio" | "sensors" | "engram"
    feats: Tensor                                # (B, T, D)   float
    mask: Tensor                                 # (B, T)      bool — True = valid
    salience: Optional[Tensor] = None            # (B, T) or (B, 1)  competition weight
    pos_ids: Optional[Tensor] = None             # (B, T)      int64
    time: Optional[Tensor] = None                # (B, T)      float32 timestamps
    spike: Optional[Tensor] = None               # (B, T, *)   spike-domain
    aux: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Contract assertion
# ---------------------------------------------------------------------------

_CONTRACTS_ENABLED: bool = True


def disable_contracts() -> None:
    """Turn off runtime contract checks for production inference (zero overhead)."""
    global _CONTRACTS_ENABLED
    _CONTRACTS_ENABLED = False


def enable_contracts() -> None:
    """Re-enable runtime contract checks (useful in test harnesses)."""
    global _CONTRACTS_ENABLED
    _CONTRACTS_ENABLED = True


def assert_encoder_output(
    output: EncoderOutput,
    workspace_dim: int,
    device: torch.device,
) -> None:
    """Validate EncoderOutput contract cheaply.

    Call at the end of every adapter ``forward``.  Guarded by
    ``_CONTRACTS_ENABLED`` so production inference pays zero cost.

    Args:
        output:        The EncoderOutput to validate.
        workspace_dim: Expected last-dim of ``feats`` (from BrainAIConfig).
        device:        Expected device for all tensors.

    Raises:
        AssertionError with a descriptive message on contract violation.
    """
    if not _CONTRACTS_ENABLED:
        return

    # --- shape ---
    assert output.feats.ndim == 3, (
        f"[{output.modality}] feats must be 3-D (B, T, D), got {output.feats.ndim}-D"
    )
    assert output.mask.ndim == 2, (
        f"[{output.modality}] mask must be 2-D (B, T), got {output.mask.ndim}-D"
    )
    assert output.feats.shape[:2] == output.mask.shape, (
        f"[{output.modality}] feats/mask batch-time mismatch: "
        f"feats {tuple(output.feats.shape[:2])} vs mask {tuple(output.mask.shape)}"
    )
    assert output.feats.shape[-1] == workspace_dim, (
        f"[{output.modality}] feats dim {output.feats.shape[-1]} != workspace_dim {workspace_dim}"
    )

    # --- device ---
    assert output.feats.device == device, (
        f"[{output.modality}] feats on {output.feats.device}, expected {device}"
    )
    assert output.mask.device == device, (
        f"[{output.modality}] mask on {output.mask.device}, expected {device}"
    )

    # --- dtype ---
    assert output.feats.dtype in (torch.float32, torch.bfloat16, torch.float16), (
        f"[{output.modality}] feats dtype {output.feats.dtype} not in {{fp32, bf16, fp16}}"
    )
    assert output.mask.dtype == torch.bool, (
        f"[{output.modality}] mask dtype {output.mask.dtype}, expected bool"
    )

    # --- optional fields ---
    if output.salience is not None:
        assert (output.salience >= 0).all(), (
            f"[{output.modality}] salience contains negative values"
        )

    if output.spike is not None:
        assert output.spike.shape[0] == output.feats.shape[0], (
            f"[{output.modality}] spike batch dim {output.spike.shape[0]} != "
            f"feats batch dim {output.feats.shape[0]}"
        )
        assert output.spike.shape[1] == output.feats.shape[1], (
            f"[{output.modality}] spike time dim {output.spike.shape[1]} != "
            f"feats time dim {output.feats.shape[1]}"
        )


# ---------------------------------------------------------------------------
# Utility: variable-length padding
# ---------------------------------------------------------------------------

def pad_and_mask(
    sequences: List[Tensor],
    pad_value: float = 0.0,
    max_len: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """Pad variable-length (T_i, D) tensors to a uniform (B, T_max, D) batch.

    Args:
        sequences: List of B tensors, each shaped (T_i, D).  T_i may differ
                   across elements.
        pad_value: Value used to fill padding positions.
        max_len:   If given, clamp T_max to this value (truncates longer
                   sequences).  If ``None``, T_max = max(T_i).

    Returns:
        padded: (B, T_max, D) float tensor.
        mask:   (B, T_max) bool tensor — True for valid positions.
    """
    B = len(sequences)
    D = sequences[0].shape[-1]
    device = sequences[0].device

    lengths = [seq.shape[0] for seq in sequences]
    T_max = max(lengths) if max_len is None else min(max(lengths), max_len)

    padded = torch.full((B, T_max, D), pad_value, device=device, dtype=sequences[0].dtype)
    mask = torch.zeros(B, T_max, dtype=torch.bool, device=device)

    for i, seq in enumerate(sequences):
        length = min(seq.shape[0], T_max)
        padded[i, :length] = seq[:length]
        mask[i, :length] = True

    return padded, mask


# ---------------------------------------------------------------------------
# Base adapter
# ---------------------------------------------------------------------------

class ModalityAdapter(nn.Module):
    """Base adapter: raw encoder internals -> EncoderOutput contract.

    Subclasses override ``forward`` to handle the modality-specific raw
    output shape and return a fully populated ``EncoderOutput``.

    Args:
        encoder_dim:   Dimensionality of the encoder's native feature vectors.
        workspace_dim: Target dimensionality for the global workspace.
        modality:      String tag (``"vision"``, ``"text"``, etc.).
    """

    def __init__(self, encoder_dim: int, workspace_dim: int, modality: str):
        super().__init__()
        self.modality = modality
        self.workspace_dim = workspace_dim
        self.encoder_dim = encoder_dim

        # Linear projection when encoder dim differs from workspace dim;
        # identity passthrough otherwise.
        if encoder_dim != workspace_dim:
            self.projection = nn.Linear(encoder_dim, workspace_dim)
        else:
            self.projection = nn.Identity()

    def forward(self, raw_output: Tensor, **kwargs) -> EncoderOutput:
        """Subclasses must override this method.

        Args:
            raw_output: Encoder-internal tensor intercepted BEFORE final
                        pooling.  Shape is modality-dependent.
            **kwargs:   Additional signals (attention masks, lengths, etc.).

        Returns:
            EncoderOutput with ``feats``, ``mask``, and optionally other
            fields populated.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement forward()"
        )


# ---------------------------------------------------------------------------
# 2-D sinusoidal positional encoding for spatial patches
# ---------------------------------------------------------------------------

class SpatialPositionalEncoding(nn.Module):
    """Additive 2-D sinusoidal positional encoding for patch grids.

    Given a (B, H'*W', D) token sequence produced by flattening a spatial
    feature map, adds row/column-aware sinusoidal embeddings so that the
    workspace can distinguish spatial relationships.
    """

    def __init__(self, dim: int, max_h: int = 64, max_w: int = 64):
        super().__init__()
        assert dim % 2 == 0, "dim must be even for 2-D sinusoidal PE"
        half = dim // 2

        # Row encoding — (max_h, half)
        pe_h = self._sinusoidal_table(max_h, half)
        # Column encoding — (max_w, half)
        pe_w = self._sinusoidal_table(max_w, half)

        self.register_buffer("pe_h", pe_h)  # (max_h, half)
        self.register_buffer("pe_w", pe_w)  # (max_w, half)

    @staticmethod
    def _sinusoidal_table(length: int, dim: int) -> Tensor:
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
        )
        pe = torch.zeros(length, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, feats: Tensor, h: int, w: int) -> Tensor:
        """Add 2-D positional encoding.

        Args:
            feats: (B, H'*W', D)
            h: Height of the spatial grid.
            w: Width of the spatial grid.

        Returns:
            feats + positional encoding, same shape (B, H'*W', D).
        """
        # Build (H', W', D) encoding from row + col halves
        row_pe = self.pe_h[:h]                   # (H', half)
        col_pe = self.pe_w[:w]                   # (W', half)

        # Outer combination: (H', 1, half) + (1, W', half) -> broadcast
        row_pe = row_pe.unsqueeze(1).expand(-1, w, -1)   # (H', W', half)
        col_pe = col_pe.unsqueeze(0).expand(h, -1, -1)   # (H', W', half)
        pe_2d = torch.cat([row_pe, col_pe], dim=-1)       # (H', W', D)
        pe_2d = pe_2d.reshape(h * w, -1)                  # (N_patches, D)

        return feats + pe_2d.unsqueeze(0)                  # broadcast over B


# ===========================================================================
# Concrete adapters
# ===========================================================================

class VisionAdapter(ModalityAdapter):
    """Adapter for vision encoders (VisionEncoder, MultiScaleVisionEncoder).

    **Input**: Conv feature maps BEFORE AdaptiveAvgPool2d.
        - Single image : (B, C_out, H', W')
        - Video        : (B, T_frames, C_out, H', W')

    **Output**: EncoderOutput with feats (B, N_patches, D) where
        N_patches = H' * W'  (single image)   or
        N_patches = T_frames * H' * W'  (video).
    """

    def __init__(
        self,
        encoder_dim: int,
        workspace_dim: int,
        modality: str = "vision",
        max_h: int = 64,
        max_w: int = 64,
    ):
        super().__init__(encoder_dim, workspace_dim, modality)

        # Learned per-token salience head: (D,) -> scalar
        self.salience_head = nn.Sequential(
            nn.Linear(workspace_dim, 1),
            nn.Sigmoid(),
        )

        # 2-D spatial positional encoding
        self.spatial_pe = SpatialPositionalEncoding(workspace_dim, max_h, max_w)

    def forward(
        self,
        raw_output: Tensor,
        **kwargs,
    ) -> EncoderOutput:
        """
        Args:
            raw_output: (B, C_out, H', W') from the vision encoder's last
                        conv block, BEFORE adaptive pooling.
                        For video: (B, T_frames, C_out, H', W').

        Returns:
            EncoderOutput with per-patch tokens.
        """
        # TODO: Wire this to the actual vision encoder by intercepting
        #       VisionEncoder.forward_step / MultiScaleVisionEncoder.backbone
        #       output BEFORE self.adaptive_pool.

        is_video = raw_output.ndim == 5
        if is_video:
            B, T_frames, C, H, W = raw_output.shape
            # Merge time into batch for projection, then restore
            raw_output = raw_output.reshape(B * T_frames, C, H, W)
        else:
            B, C, H, W = raw_output.shape
            T_frames = 1

        # (B', C, H, W) -> (B', H*W, C) — flatten spatial dims to tokens
        N_patches = H * W
        tokens = raw_output.flatten(2).transpose(1, 2)   # (B', N_patches, C)

        # Project to workspace_dim: (B', N_patches, C) -> (B', N_patches, D)
        feats = self.projection(tokens)                   # (B', N_patches, D)

        # Add 2-D spatial positional encoding
        feats = self.spatial_pe(feats, H, W)              # (B', N_patches, D)

        if is_video:
            # Reshape back: (B, T_frames * N_patches, D)
            feats = feats.reshape(B, T_frames * N_patches, feats.shape[-1])
        else:
            # feats already (B, N_patches, D)
            pass

        B_out = feats.shape[0]
        T = feats.shape[1]

        # All patches are valid for fixed-size images
        mask = torch.ones(B_out, T, dtype=torch.bool, device=feats.device)

        # Learned salience per token
        salience = self.salience_head(feats).squeeze(-1)  # (B, T)

        # Build time tensor for video (frame indices repeated per patch)
        time_tensor = None
        if is_video:
            frame_ids = torch.arange(T_frames, device=feats.device, dtype=torch.float32)
            # Each frame has N_patches tokens
            time_tensor = frame_ids.repeat_interleave(N_patches)  # (T_frames*N_patches,)
            time_tensor = time_tensor.unsqueeze(0).expand(B_out, -1)  # (B, T)

        # Optional: capture spike raster from spiking encoder
        # TODO: If VisionEncoder uses SNN conv blocks, call
        #       encoder.get_spike_raster() and pass spike=(B,T,D_spike) here.
        spike = None

        aux: Dict[str, Any] = {"pos_applied": True}
        if is_video:
            aux["is_video"] = True
            aux["num_frames"] = T_frames
            aux["patches_per_frame"] = N_patches

        return EncoderOutput(
            modality=self.modality,
            feats=feats,
            mask=mask,
            salience=salience,
            time=time_tensor,
            spike=spike,
            aux=aux,
        )


# ---------------------------------------------------------------------------

class TextAdapter(ModalityAdapter):
    """Adapter for text encoders (TextEncoder, SpikeTextEncoder).

    **Input**: Transformer hidden states (B, L, embed_dim) BEFORE the
    encoder's final pooling and projection head.

    **Output**: EncoderOutput with feats (B, L, D) — one token per subword.
    """

    def __init__(
        self,
        encoder_dim: int,
        workspace_dim: int,
        modality: str = "text",
    ):
        super().__init__(encoder_dim, workspace_dim, modality)

        # Optional attention-based salience head
        self.salience_head = nn.Sequential(
            nn.Linear(workspace_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        raw_output: Tensor,
        attention_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> EncoderOutput:
        """
        Args:
            raw_output:     (B, L, embed_dim) — transformer hidden states
                            BEFORE pooling.
            attention_mask: (B, L) int or bool from the tokenizer.
                            1 / True = valid token, 0 / False = padding.

        Returns:
            EncoderOutput with per-token features.
        """
        # TODO: Wire this to the actual TextEncoder by intercepting the
        #       output of self.transformer BEFORE the pooling step in
        #       TextEncoder.forward (after line ``x = self.transformer(...)``).

        feats = self.projection(raw_output)               # (B, L, D)
        B, L, _ = feats.shape

        # Convert attention mask to bool
        if attention_mask is not None:
            mask = attention_mask.bool()                   # (B, L)
        else:
            mask = torch.ones(B, L, dtype=torch.bool, device=feats.device)

        # Attention-weighted salience
        salience = self.salience_head(feats).squeeze(-1)  # (B, L)

        aux: Dict[str, Any] = {"pos_applied": True}

        return EncoderOutput(
            modality=self.modality,
            feats=feats,
            mask=mask,
            salience=salience,
            aux=aux,
        )


# ---------------------------------------------------------------------------

class AudioAdapter(ModalityAdapter):
    """Adapter for audio encoders (AudioEncoder, StreamingAudioEncoder).

    **Input**: Conv feature maps (B, channels, frames) from the spiking 1-D
    conv stack, BEFORE AdaptiveAvgPool1d.

    **Output**: EncoderOutput with feats (B, frames, D) — one token per
    mel-spectrogram frame.

    The STFT computation inside MelSpectrogramFrontend stays in fp32 to
    preserve numerical precision; only the projected features adopt the
    training dtype (bf16 / fp16 via AMP).
    """

    def __init__(
        self,
        encoder_dim: int,
        workspace_dim: int,
        modality: str = "audio",
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
    ):
        super().__init__(encoder_dim, workspace_dim, modality)

        self.mel_params: Dict[str, int] = {
            "sample_rate": sample_rate,
            "n_fft": n_fft,
            "hop_length": hop_length,
            "n_mels": n_mels,
        }

    def forward(
        self,
        raw_output: Tensor,
        frame_lengths: Optional[Tensor] = None,
        **kwargs,
    ) -> EncoderOutput:
        """
        Args:
            raw_output:    (B, channels, frames) — spiking conv features
                           BEFORE AdaptiveAvgPool1d.
            frame_lengths: (B,) int — number of valid frames per sample
                           (derived from original waveform lengths).

        Returns:
            EncoderOutput with per-frame tokens.
        """
        # TODO: Wire this to the actual AudioEncoder by intercepting the
        #       conv output BEFORE self.pool in AudioEncoder.forward.
        #       Also need to track frame_lengths through the mel frontend.

        # (B, channels, frames) -> (B, frames, channels)
        x = raw_output.transpose(1, 2)                    # (B, frames, channels)
        feats = self.projection(x)                         # (B, frames, D)
        B, T, _ = feats.shape

        # Build mask from frame lengths (padding mask for variable-length audio)
        if frame_lengths is not None:
            # frame_lengths: (B,) — number of valid frames per sample
            indices = torch.arange(T, device=feats.device).unsqueeze(0)  # (1, T)
            mask = indices < frame_lengths.unsqueeze(1)                   # (B, T)
        else:
            mask = torch.ones(B, T, dtype=torch.bool, device=feats.device)

        # Compute frame timestamps: frame_index * hop_length / sample_rate
        hop = self.mel_params["hop_length"]
        sr = self.mel_params["sample_rate"]
        frame_indices = torch.arange(T, device=feats.device, dtype=torch.float32)
        time_seconds = frame_indices * (hop / sr)          # (T,)
        time_tensor = time_seconds.unsqueeze(0).expand(B, -1)  # (B, T)

        # Determine which audio frontend was used
        # TODO: Read this from the encoder instance at adapter construction
        # time.  For now, default to "unknown".
        audio_frontend = kwargs.get("audio_frontend", "unknown")

        aux: Dict[str, Any] = {
            "pos_applied": True,
            "mel_params": self.mel_params,
            "audio_frontend": audio_frontend,
        }

        return EncoderOutput(
            modality=self.modality,
            feats=feats,
            mask=mask,
            salience=None,  # uniform salience — workspace assigns equal weight
            time=time_tensor,
            aux=aux,
        )


# ---------------------------------------------------------------------------

class SensorAdapter(ModalityAdapter):
    """Adapter for sensor encoders (SensorEncoder, IMUEncoder).

    **Input**: Liquid layer outputs with ``return_sequence=True``:
    (B, T_steps, hidden_dim).

    **Output**: EncoderOutput with feats (B, T_steps, D) — one token per
    sensor time-step.

    For irregular-timestamp sensors, pass ``timestamps`` (B, T_steps) as a
    kwarg.  The adapter embeds inter-step dt as an extra feature channel,
    concatenates with the hidden states, then projects the combined tensor
    to workspace_dim.
    """

    def __init__(
        self,
        encoder_dim: int,
        workspace_dim: int,
        modality: str = "sensors",
        dt: float = 1.0,
        dt_embed_dim: int = 16,
    ):
        # When timestamps are provided, the projection input is
        # encoder_dim + dt_embed_dim.
        super().__init__(encoder_dim + dt_embed_dim, workspace_dim, modality)
        self.dt_default = dt
        self.dt_embed_dim = dt_embed_dim
        self.dt_embed = nn.Linear(1, dt_embed_dim)

        # Also keep a plain-encoder-dim projection for the no-timestamp path
        if encoder_dim != workspace_dim:
            self._proj_no_dt = nn.Linear(encoder_dim, workspace_dim)
        else:
            self._proj_no_dt = nn.Identity()

    def forward(
        self,
        raw_output: Tensor,
        seq_lengths: Optional[Tensor] = None,
        timestamps: Optional[Tensor] = None,
        **kwargs,
    ) -> EncoderOutput:
        """
        Args:
            raw_output:  (B, T_steps, hidden_dim) — liquid-layer sequence.
            seq_lengths: (B,) int — valid length per sample.
            timestamps:  (B, T_steps) float — real-valued timestamps in
                         seconds.  If None, synthetic time is computed from
                         ``dt * step_index``.

        Returns:
            EncoderOutput with per-timestep tokens.
        """
        # TODO: Wire this to SensorEncoder.forward with
        #       ``return_sequence=True`` to get full (B, T, hidden_dim)
        #       instead of last-step-only output.

        B, T, H = raw_output.shape

        if timestamps is not None:
            # Embed inter-step dt
            dt_vals = F.pad(
                timestamps[:, 1:] - timestamps[:, :-1],    # (B, T-1)
                (1, 0),                                     # left-pad first step with 0
                value=0.0,
            )                                               # (B, T)
            dt_feats = self.dt_embed(dt_vals.unsqueeze(-1)) # (B, T, dt_embed_dim)
            combined = torch.cat([raw_output, dt_feats], dim=-1)  # (B, T, H+dt_embed_dim)
            feats = self.projection(combined)               # (B, T, D)
            time_tensor = timestamps                        # (B, T)
        else:
            # No explicit timestamps — use default dt spacing
            feats = self._proj_no_dt(raw_output)            # (B, T, D)
            step_indices = torch.arange(T, device=raw_output.device, dtype=torch.float32)
            time_tensor = (step_indices * self.dt_default).unsqueeze(0).expand(B, -1)

        # Mask from sequence lengths
        if seq_lengths is not None:
            indices = torch.arange(T, device=feats.device).unsqueeze(0)
            mask = indices < seq_lengths.unsqueeze(1)       # (B, T) bool
        else:
            mask = torch.ones(B, T, dtype=torch.bool, device=feats.device)

        aux: Dict[str, Any] = {"pos_applied": True}

        return EncoderOutput(
            modality=self.modality,
            feats=feats,
            mask=mask,
            salience=None,
            time=time_tensor,
            aux=aux,
        )


# ---------------------------------------------------------------------------

class EngramAdapter(ModalityAdapter):
    """Adapter for engram encoders (EngramTextEncoder).

    **Input**: Engram embeddings (B, L, embed_dim) BEFORE the encoder's
    mean-pooling step.

    **Output**: EncoderOutput with feats (B, L, D) — one token per n-gram
    position.

    The first ``ngram_order - 1`` positions lack a full n-gram context
    window and are masked out.
    """

    def __init__(
        self,
        encoder_dim: int,
        workspace_dim: int,
        modality: str = "engram",
        ngram_order: int = 3,
    ):
        super().__init__(encoder_dim, workspace_dim, modality)
        self.ngram_order = ngram_order

    def forward(
        self,
        raw_output: Tensor,
        seq_lengths: Optional[Tensor] = None,
        **kwargs,
    ) -> EncoderOutput:
        """
        Args:
            raw_output:  (B, L, embed_dim) — engram embeddings (with
                         positional encoding already applied inside
                         EngramTextEncoder) BEFORE pooling.
            seq_lengths: (B,) int — number of valid tokens per sample.

        Returns:
            EncoderOutput with per-position tokens.
        """
        # TODO: Wire this to EngramTextEncoder by intercepting the output
        #       AFTER positional encoding addition but BEFORE the masked
        #       mean pooling step in EngramTextEncoder.forward.

        feats = self.projection(raw_output)               # (B, L, D)
        B, L, _ = feats.shape

        # N-gram validity mask: first (ngram_order - 1) positions lack
        # full context and are invalid.
        mask = torch.ones(B, L, dtype=torch.bool, device=feats.device)
        if self.ngram_order > 1:
            mask[:, : self.ngram_order - 1] = False

        # Additionally mask out padding positions
        if seq_lengths is not None:
            indices = torch.arange(L, device=feats.device).unsqueeze(0)
            length_mask = indices < seq_lengths.unsqueeze(1)  # (B, L)
            mask = mask & length_mask

        aux: Dict[str, Any] = {"pos_applied": True}

        return EncoderOutput(
            modality=self.modality,
            feats=feats,
            mask=mask,
            salience=None,
            aux=aux,
        )


# ---------------------------------------------------------------------------
# Registry and factory
# ---------------------------------------------------------------------------

ADAPTER_REGISTRY: Dict[str, Type[ModalityAdapter]] = {
    "vision": VisionAdapter,
    "text": TextAdapter,
    "audio": AudioAdapter,
    "sensors": SensorAdapter,
    "engram": EngramAdapter,
}


def create_adapter(
    modality: str,
    encoder_dim: int,
    workspace_dim: int,
    **kwargs,
) -> ModalityAdapter:
    """Look up and instantiate the correct adapter for a modality.

    This is the primary entry point used by the orchestrator during
    ``__init__`` to build per-modality adapter submodules.

    Args:
        modality:      One of ``"vision"``, ``"text"``, ``"audio"``,
                       ``"sensors"``, ``"engram"``.
        encoder_dim:   Dimensionality of the encoder's native output
                       (e.g., 128 for the default AudioEncoder conv stack,
                       256 for TextEncoder embed_dim).
        workspace_dim: Target dimensionality for the global workspace
                       (typically 4096 at production scale).
        **kwargs:      Forwarded to the adapter constructor (e.g.,
                       ``sample_rate``, ``ngram_order``).

    Returns:
        An initialized ``ModalityAdapter`` subclass instance.

    Raises:
        ValueError: If ``modality`` is not in ``ADAPTER_REGISTRY``.

    Example::

        adapter = create_adapter(
            modality="vision",
            encoder_dim=128,       # last conv channel count
            workspace_dim=4096,
        )
        enc_out = adapter(conv_features)
        assert_encoder_output(enc_out, workspace_dim=4096, device=conv_features.device)
    """
    if modality not in ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown modality '{modality}'. "
            f"Registered: {sorted(ADAPTER_REGISTRY.keys())}"
        )
    return ADAPTER_REGISTRY[modality](
        encoder_dim=encoder_dim,
        workspace_dim=workspace_dim,
        modality=modality,
        **kwargs,
    )
