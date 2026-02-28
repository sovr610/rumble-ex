#!/usr/bin/env python3
"""System 1 Fast Predictor Module for Dual-Process Reasoning.

This module implements the fast, parallel "System 1" pathway of the dual-process
reasoning architecture. System 1 operates as a shallow MLP that produces rapid
predictions along with calibrated confidence metrics. When confidence falls below
a configurable threshold, the slower System 2 pathway is engaged.

Architecture overview::

    Input (B, D) or (B, K, D)
         |
    [ Pooling Head ]  (if slots input)
         |
    [ MLP Body ]      (1-3 layers, LayerNorm, dropout)
         |
    +----+----+
    |         |
  Logits   Confidence
  Head       Head (optional)
    |         |
    v         v
  System1Result

The module is designed for production use with:
- Configurable pooling strategies (mean, attention, CLS token)
- Multiple uncertainty metrics (softmax confidence, entropy, margin)
- Optional learned confidence head for calibrated estimates
- Full fp32 precision for confidence computations
- Deterministic forward pass (no stochastic components beyond dropout)

Typical usage::

    config = System1Config(input_dim=4096, output_dim=256)
    model = System1Fast(config)
    result = model(workspace_tensor)  # (B, 4096)
    print(result.conf_calibrated)     # (B,) calibrated confidence

Dependencies: torch (no external dependencies).
"""

from __future__ import annotations

import math
import sys
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS: float = 1e-8
_SUPPORTED_ACTIVATIONS: Tuple[str, ...] = ("relu", "gelu", "silu")
_SUPPORTED_POOL_MODES: Tuple[str, ...] = ("mean", "attention", "cls")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class System1Config:
    """Configuration for the System 1 fast predictor.

    Attributes:
        input_dim: Dimensionality of the input workspace representation.
            Must match the upstream workspace output (typically 4096).
        hidden_dim: Width of the hidden layers in the MLP body.
        output_dim: Dimensionality of the output logits / embedding.
        num_layers: Number of MLP blocks in the body (1-3).
        pool_mode: Pooling strategy for 3-D slot inputs.  One of
            ``"mean"`` (average over slots), ``"attention"`` (learned
            attention weights), or ``"cls"`` (select first slot).
        confidence_head: Whether to attach a learned confidence head
            alongside the logits head.  The learned confidence is
            blended with the softmax-derived raw confidence.
        activation: Nonlinearity for the MLP body.  One of ``"gelu"``,
            ``"relu"``, or ``"silu"``.
        dropout: Dropout probability applied after each MLP block.
        num_slots: Optional hint for the number of slots (used by the
            attention pooling head for documentation; not enforced).
    """

    input_dim: int = 4096
    hidden_dim: int = 512
    output_dim: int = 256
    num_layers: int = 2
    pool_mode: str = "mean"
    confidence_head: bool = True
    activation: str = "gelu"
    dropout: float = 0.1
    num_slots: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {self.output_dim}")
        if self.num_layers < 1 or self.num_layers > 3:
            raise ValueError(f"num_layers must be in [1, 3], got {self.num_layers}")
        if self.pool_mode not in _SUPPORTED_POOL_MODES:
            raise ValueError(
                f"pool_mode must be one of {_SUPPORTED_POOL_MODES}, "
                f"got '{self.pool_mode}'"
            )
        if self.activation not in _SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {_SUPPORTED_ACTIVATIONS}, "
                f"got '{self.activation}'"
            )
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}")

    @classmethod
    def minimal(cls) -> "System1Config":
        """Create a minimal configuration for unit testing."""
        return cls(
            input_dim=32, hidden_dim=16, output_dim=8,
            num_layers=1, confidence_head=True, dropout=0.0,
        )

    @classmethod
    def small(cls) -> "System1Config":
        """Create a small configuration for development."""
        return cls(
            input_dim=256, hidden_dim=128, output_dim=64,
            num_layers=2, confidence_head=True, dropout=0.1,
        )

    @classmethod
    def production(cls) -> "System1Config":
        """Create a production-scale configuration."""
        return cls(
            input_dim=4096, hidden_dim=512, output_dim=256,
            num_layers=2, confidence_head=True, dropout=0.1,
        )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class System1Result:
    """Output container for System 1 forward pass.

    Attributes:
        y1: Prediction logits of shape ``(B, output_dim)``.
        conf_raw: Raw confidence (max softmax probability) of shape ``(B,)``.
        conf_calibrated: Calibrated confidence of shape ``(B,)``.
            When no learned confidence head is active, this equals
            ``conf_raw``.  Otherwise it is the blended learned+raw value.
        entropy: Shannon entropy of the softmax distribution, shape ``(B,)``.
        margin: Difference between top-1 and top-2 logit values, shape ``(B,)``.
        uncertainty_metrics: Dictionary containing ``"conf_raw"``,
            ``"entropy"``, and ``"margin"`` tensors (and optionally
            ``"learned_confidence"``).
        hidden: Intermediate hidden representation of shape
            ``(B, hidden_dim)`` -- useful for passing to System 2.
    """

    y1: Tensor
    conf_raw: Tensor
    conf_calibrated: Tensor
    entropy: Tensor
    margin: Tensor
    uncertainty_metrics: Dict[str, Tensor]
    hidden: Tensor

    # -- Convenience methods ------------------------------------------------

    def is_confident(self, threshold: float = 0.7) -> Tensor:
        """Return boolean mask ``(B,)`` for predictions above *threshold*."""
        return self.conf_calibrated >= threshold

    def top_k_predictions(self, k: int = 5) -> Tuple[Tensor, Tensor]:
        """Return top-k logit values and their indices."""
        k = min(k, self.y1.shape[-1])
        return torch.topk(self.y1, k, dim=-1)

    def detach(self) -> "System1Result":
        """Return a new ``System1Result`` with all tensors detached."""
        return System1Result(
            y1=self.y1.detach(),
            conf_raw=self.conf_raw.detach(),
            conf_calibrated=self.conf_calibrated.detach(),
            entropy=self.entropy.detach(),
            margin=self.margin.detach(),
            uncertainty_metrics={
                k: v.detach() for k, v in self.uncertainty_metrics.items()
            },
            hidden=self.hidden.detach(),
        )

    def to(self, device: Union[str, torch.device]) -> "System1Result":
        """Move all tensors to *device*."""
        return System1Result(
            y1=self.y1.to(device),
            conf_raw=self.conf_raw.to(device),
            conf_calibrated=self.conf_calibrated.to(device),
            entropy=self.entropy.to(device),
            margin=self.margin.to(device),
            uncertainty_metrics={
                k: v.to(device) for k, v in self.uncertainty_metrics.items()
            },
            hidden=self.hidden.to(device),
        )


# ---------------------------------------------------------------------------
# Pure confidence functions
# ---------------------------------------------------------------------------


def compute_conf_raw(logits: Tensor) -> Tensor:
    """Compute raw confidence as the maximum softmax probability.

    All arithmetic is performed in fp32 regardless of input dtype.

    Args:
        logits: Tensor of shape ``(B, C)`` where *C* is the number of
            output classes / dimensions.

    Returns:
        Tensor of shape ``(B,)`` with values in ``[0, 1]``.
    """
    if logits.shape[-1] == 1:
        return torch.ones(logits.shape[0], device=logits.device, dtype=torch.float32)
    probs = F.softmax(logits.float(), dim=-1)
    conf, _ = probs.max(dim=-1)
    return conf


def compute_entropy(logits: Tensor) -> Tensor:
    """Compute Shannon entropy of the softmax distribution.

    Uses the numerically stable identity ``H = -sum(p * log_softmax)``
    evaluated in fp32.

    Args:
        logits: Tensor of shape ``(B, C)``.

    Returns:
        Tensor of shape ``(B,)`` with non-negative entropy values.
    """
    if logits.shape[-1] == 1:
        return torch.zeros(logits.shape[0], device=logits.device, dtype=torch.float32)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = F.softmax(logits.float(), dim=-1)
    return -torch.sum(probs * log_probs, dim=-1)


def compute_margin(logits: Tensor) -> Tensor:
    """Compute the margin between the top-1 and top-2 logit values.

    A large margin indicates high separation between the winning class
    and the runner-up, which is an independent confidence signal that
    does not depend on the softmax temperature.

    Args:
        logits: Tensor of shape ``(B, C)``.

    Returns:
        Tensor of shape ``(B,)`` with non-negative margin values.
    """
    logits_f = logits.float()
    if logits_f.shape[-1] == 1:
        return torch.zeros(
            logits_f.shape[0], device=logits_f.device, dtype=torch.float32
        )
    if logits_f.shape[-1] == 2:
        return (logits_f[:, 0] - logits_f[:, 1]).abs()
    top2_vals, _ = torch.topk(logits_f, k=2, dim=-1)
    return top2_vals[:, 0] - top2_vals[:, 1]


def compute_uncertainty_metrics(logits: Tensor) -> Dict[str, Tensor]:
    """Compute all three uncertainty metrics from logits.

    Returns:
        Dictionary with keys ``"conf_raw"``, ``"entropy"``, ``"margin"``.
    """
    return {
        "conf_raw": compute_conf_raw(logits),
        "entropy": compute_entropy(logits),
        "margin": compute_margin(logits),
    }


# ---------------------------------------------------------------------------
# Activation factory
# ---------------------------------------------------------------------------


def _get_activation(name: str) -> nn.Module:
    """Create an activation module from its string name.

    Raises:
        ValueError: If *name* is not in :data:`_SUPPORTED_ACTIVATIONS`.
    """
    name = name.lower().strip()
    if name == "relu":
        return nn.ReLU(inplace=False)
    elif name == "gelu":
        return nn.GELU()
    elif name == "silu":
        return nn.SiLU(inplace=False)
    else:
        raise ValueError(
            f"Unsupported activation '{name}'. "
            f"Choose from {_SUPPORTED_ACTIVATIONS}."
        )


# ---------------------------------------------------------------------------
# Pooling heads
# ---------------------------------------------------------------------------


class MeanPoolHead(nn.Module):
    """Pool slot representations by averaging over the slot dimension.

    Given ``(B, K, D)`` returns ``(B, D)`` via ``mean(dim=1)``.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, slots: Tensor) -> Tensor:
        """Compute the mean over the slot dimension."""
        return slots.mean(dim=1)

    def extra_repr(self) -> str:
        return "mode=mean"


class AttentionPoolHead(nn.Module):
    """Pool slot representations using learned attention weights.

    Computes a scalar attention score for each slot via a learned query
    vector, normalises with softmax, and returns the weighted sum.

    Args:
        dim: Feature dimensionality (last dim of the slots tensor).
        num_slots: Optional number-of-slots hint for documentation.
    """

    def __init__(self, dim: int, num_slots: Optional[int] = None) -> None:
        super().__init__()
        self.dim: int = dim
        self.num_slots: Optional[int] = num_slots
        self.query = nn.Parameter(torch.randn(dim, 1) * (dim ** -0.5))
        self.scale: float = dim ** -0.5
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, slots: Tensor) -> Tensor:
        """Compute attention-weighted sum over slots.

        Args:
            slots: Tensor of shape ``(B, K, D)``.

        Returns:
            Pooled representation ``(B, D)``.
        """
        normed = self.layer_norm(slots)  # (B, K, D)
        attn_logits = torch.matmul(normed, self.query) * self.scale  # (B, K, 1)
        attn_weights = F.softmax(attn_logits.float(), dim=1).to(slots.dtype)
        return (slots * attn_weights).sum(dim=1)  # (B, D)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, num_slots={self.num_slots}, "
            f"scale={self.scale:.6f}"
        )


class CLSPoolHead(nn.Module):
    """Pool slot representations by selecting the first slot (CLS token).

    Given ``(B, K, D)`` returns ``(B, D)`` by taking ``[:, 0, :]``.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, slots: Tensor) -> Tensor:
        """Select the first slot as the CLS representation."""
        return slots[:, 0, :]

    def extra_repr(self) -> str:
        return "mode=cls, index=0"


def create_pool_head(config: System1Config) -> nn.Module:
    """Factory: build the appropriate pooling head from a config.

    Args:
        config: System1Config with ``pool_mode``, ``input_dim``, and
            ``num_slots`` fields.

    Returns:
        An ``nn.Module`` that maps ``(B, K, D) -> (B, D)``.

    Raises:
        ValueError: If ``config.pool_mode`` is unrecognised.
    """
    if config.pool_mode == "mean":
        return MeanPoolHead()
    elif config.pool_mode == "attention":
        return AttentionPoolHead(dim=config.input_dim, num_slots=config.num_slots)
    elif config.pool_mode == "cls":
        return CLSPoolHead()
    else:
        raise ValueError(
            f"Unknown pool_mode '{config.pool_mode}'. "
            f"Choose from {_SUPPORTED_POOL_MODES}."
        )


# ---------------------------------------------------------------------------
# MLP building blocks
# ---------------------------------------------------------------------------


class MLPBlock(nn.Module):
    """Single MLP block: Linear -> LayerNorm -> Activation -> Dropout.

    This is the repeating unit of the System 1 MLP body.

    Args:
        in_features: Input dimensionality.
        out_features: Output dimensionality.
        activation: Activation function name.
        dropout: Dropout probability.
        bias: Whether to include bias in the linear layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = "gelu",
        dropout: float = 0.1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.norm = nn.LayerNorm(out_features)
        self.act = _get_activation(activation)
        self.drop: nn.Module = (
            nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply linear -> norm -> activation -> dropout."""
        return self.drop(self.act(self.norm(self.linear(x))))


class MLPBody(nn.Module):
    """Multi-layer MLP body for System 1 processing.

    Stacks ``num_layers`` :class:`MLPBlock` instances.  The first block
    projects from ``input_dim`` to ``hidden_dim``; subsequent blocks
    maintain ``hidden_dim``.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Hidden layer width (and output width).
        num_layers: Number of :class:`MLPBlock` layers.
        activation: Activation function name.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation: str = "gelu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        blocks: List[nn.Module] = [
            MLPBlock(input_dim, hidden_dim, activation, dropout)
        ]
        for _ in range(num_layers - 1):
            blocks.append(
                MLPBlock(hidden_dim, hidden_dim, activation, dropout)
            )
        self.layers = nn.Sequential(*blocks)
        self.output_dim: int = hidden_dim

    def forward(self, x: Tensor) -> Tensor:
        """Pass input through all MLP blocks."""
        return self.layers(x)


class LogitsHead(nn.Module):
    """Linear projection from hidden space to output logits.

    Args:
        hidden_dim: Input feature dimension.
        output_dim: Output (logits) dimension.
        bias: Whether to include bias.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, output_dim, bias=bias)

    def forward(self, hidden: Tensor) -> Tensor:
        """Project hidden state to logits."""
        return self.linear(hidden)


class ConfidenceHead(nn.Module):
    """Learned confidence predictor from hidden states.

    A small MLP that maps the hidden representation to a scalar
    confidence in ``[0, 1]`` via sigmoid.

    Architecture::

        hidden -> Linear(hidden_dim, hidden_dim//2) -> LayerNorm
               -> activation -> Dropout
               -> Linear(hidden_dim//2, 1)
               -> Sigmoid -> (B,)

    All internal computation is in fp32 to avoid numerical issues with
    half-precision confidence values.

    Args:
        hidden_dim: Input dimensionality (matches the MLP body output).
        activation: Activation function name.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        hidden_dim: int,
        activation: str = "gelu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        mid_dim: int = max(hidden_dim // 2, 1)
        layers: List[nn.Module] = [
            nn.Linear(hidden_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            _get_activation(activation),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(mid_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, hidden: Tensor) -> Tensor:
        """Predict confidence from hidden state.

        Args:
            hidden: Tensor of shape ``(B, hidden_dim)``.

        Returns:
            Confidence tensor of shape ``(B,)`` in ``[0, 1]``.
        """
        return torch.sigmoid(self.net(hidden.float())).squeeze(-1)


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------


def _validate_input_shape(
    x: Tensor, expected_dim: int, name: str = "input"
) -> None:
    """Validate that *x* is 2-D ``(B, D)`` or 3-D ``(B, K, D)``.

    Raises:
        ValueError: If shape is invalid.
    """
    if x.ndim not in (2, 3):
        raise ValueError(
            f"{name} must be 2D (B, D) or 3D (B, K, D), "
            f"got {x.ndim}D with shape {tuple(x.shape)}"
        )
    if x.shape[-1] != expected_dim:
        raise ValueError(
            f"{name} last dimension must be {expected_dim}, "
            f"got {x.shape[-1]} (shape: {tuple(x.shape)})"
        )


def _is_slots_input(x: Tensor) -> bool:
    """Return ``True`` if *x* is a 3-D slots tensor."""
    return x.ndim == 3


# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------


def _init_weights(module: nn.Module) -> None:
    """Apply Kaiming uniform init to linear layers, ones/zeros to norms."""
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
        if module.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(module.bias, -bound, bound)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------------
# System1Fast -- main module
# ---------------------------------------------------------------------------


class System1Fast(nn.Module):
    """System 1 fast predictor: shallow MLP with confidence estimation.

    Accepts either pooled workspace vectors ``(B, D)`` or raw slot
    representations ``(B, K, D)`` and produces logits together with
    calibrated confidence scores.  When confidence falls below a
    configurable threshold the caller should route the input to the
    slower System 2 pathway.

    Parameters
    ----------
    config : System1Config, optional
        Full configuration.  If ``None`` a default config is used.

    Examples
    --------
    >>> cfg = System1Config(input_dim=512, output_dim=64)
    >>> model = System1Fast(cfg)
    >>> x = torch.randn(4, 512)
    >>> result = model(x)
    >>> result.y1.shape
    torch.Size([4, 64])
    """

    def __init__(self, config: Optional[System1Config] = None) -> None:
        super().__init__()
        self.config: System1Config = config or System1Config()

        # ---- pooling (only used when input is 3-D slots) ----
        self.pool_head: nn.Module = create_pool_head(self.config)

        # ---- MLP body ----
        self.body = MLPBody(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            activation=self.config.activation,
            dropout=self.config.dropout,
        )

        # ---- logits head ----
        self.logits_head = LogitsHead(
            self.config.hidden_dim,
            self.config.output_dim,
        )

        # ---- optional learned confidence head ----
        self.confidence_head_module: Optional[ConfidenceHead] = None
        if self.config.confidence_head:
            self.confidence_head_module = ConfidenceHead(
                self.config.hidden_dim,
                activation=self.config.activation,
                dropout=self.config.dropout,
            )

        # ---- weight init ----
        self.apply(_init_weights)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> System1Result:
        """Run the System 1 forward pass.

        Parameters
        ----------
        x : Tensor
            Either ``(B, D)`` or ``(B, K, D)`` where *D* must equal
            ``config.input_dim``.
        context : Tensor, optional
            Reserved for future use (e.g. workspace context).  Currently
            ignored.

        Returns
        -------
        System1Result
            Named result with logits, confidence, entropy, margin,
            hidden, and auxiliary uncertainty metrics.
        """
        _validate_input_shape(x, self.config.input_dim, name="System1Fast.input")

        # ---- Step 1: pool if 3-D slots ----
        if _is_slots_input(x):
            x = self.pool_head(x)  # (B, K, D) -> (B, D)

        # ---- Step 2: MLP body ----
        hidden: Tensor = self.body(x)  # (B, hidden_dim)

        # ---- Step 3: compute logits ----
        logits: Tensor = self.logits_head(hidden)  # (B, output_dim)

        # ---- Step 4: confidence metrics (always fp32) ----
        conf_raw: Tensor = compute_conf_raw(logits)
        entropy: Tensor = compute_entropy(logits)
        margin: Tensor = compute_margin(logits)
        uncertainty_metrics: Dict[str, Tensor] = compute_uncertainty_metrics(logits)

        # ---- Step 5: optional learned confidence head ----
        conf_calibrated: Tensor = conf_raw
        if self.confidence_head_module is not None:
            learned_conf: Tensor = self.confidence_head_module(hidden)  # (B,)
            # Blend: average of softmax-derived conf and learned conf
            conf_calibrated = 0.5 * conf_raw + 0.5 * learned_conf
            uncertainty_metrics["learned_confidence"] = learned_conf.detach()

        # ---- Step 6: assemble result ----
        return System1Result(
            y1=logits,
            conf_raw=conf_raw,
            conf_calibrated=conf_calibrated,
            entropy=entropy,
            margin=margin,
            uncertainty_metrics=uncertainty_metrics,
            hidden=hidden,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_param_count(self) -> Dict[str, int]:
        """Return parameter counts broken down by component."""
        counts: Dict[str, int] = {}
        for name, child in self.named_children():
            counts[name] = sum(p.numel() for p in child.parameters())
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts

    def get_hidden_dim(self) -> int:
        """Return the hidden dimension of the MLP body."""
        return self.config.hidden_dim

    def get_output_dim(self) -> int:
        """Return the output (logits) dimension."""
        return self.config.output_dim

    def freeze_body(self) -> None:
        """Freeze the MLP body parameters (for fine-tuning heads only)."""
        for p in self.body.parameters():
            p.requires_grad = False

    def unfreeze_body(self) -> None:
        """Unfreeze the MLP body parameters."""
        for p in self.body.parameters():
            p.requires_grad = True

    def __repr__(self) -> str:
        counts = self.get_param_count()
        return (
            f"System1Fast(in={self.config.input_dim}, "
            f"hidden={self.config.hidden_dim}, "
            f"out={self.config.output_dim}, "
            f"layers={self.config.num_layers}, "
            f"pool={self.config.pool_mode}, "
            f"conf_head={self.config.confidence_head}, "
            f"params={counts.get('total', 0):,})"
        )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def batch_predict(
    model: System1Fast,
    data: Tensor,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> System1Result:
    """Run *model* over *data* in mini-batches and concatenate results.

    This avoids OOM for large datasets while still returning a single
    ``System1Result`` covering every input sample.

    Parameters
    ----------
    model : System1Fast
        The System 1 model.
    data : Tensor
        Full dataset tensor ``(N, D)`` or ``(N, K, D)``.
    batch_size : int
        Micro-batch size.
    device : torch.device, optional
        Move each batch to *device* before forward.

    Returns
    -------
    System1Result
        Concatenated results across all batches.
    """
    was_training = model.training
    model.train(False)
    results: List[System1Result] = []
    n: int = data.shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = data[start : start + batch_size]
            if device is not None:
                batch = batch.to(device)
            results.append(model(batch))
    model.train(was_training)
    # Concatenate all tensor fields
    y1 = torch.cat([r.y1 for r in results], dim=0)
    conf_raw = torch.cat([r.conf_raw for r in results], dim=0)
    conf_cal = torch.cat([r.conf_calibrated for r in results], dim=0)
    ent = torch.cat([r.entropy for r in results], dim=0)
    mar = torch.cat([r.margin for r in results], dim=0)
    hid = torch.cat([r.hidden for r in results], dim=0)
    # Merge uncertainty dictionaries
    all_keys = results[0].uncertainty_metrics.keys()
    merged: Dict[str, Tensor] = {}
    for k in all_keys:
        merged[k] = torch.cat(
            [r.uncertainty_metrics[k] for r in results], dim=0
        )
    return System1Result(
        y1=y1,
        conf_raw=conf_raw,
        conf_calibrated=conf_cal,
        entropy=ent,
        margin=mar,
        uncertainty_metrics=merged,
        hidden=hid,
    )


def system1_loss(
    result: System1Result,
    targets: Tensor,
    ce_weight: float = 1.0,
    conf_weight: float = 0.1,
    entropy_weight: float = 0.01,
) -> Tensor:
    """Composite loss for System 1 training.

    Combines:
    * Cross-entropy on logits
    * Confidence calibration loss (MSE between conf_raw and per-sample
      accuracy indicator)
    * Entropy regulariser (penalises high entropy on correct predictions)

    Parameters
    ----------
    result : System1Result
        Output from ``System1Fast.forward``.
    targets : Tensor
        Ground-truth class indices ``(B,)``.
    ce_weight, conf_weight, entropy_weight : float
        Weighting coefficients for each loss component.

    Returns
    -------
    Tensor
        Scalar combined loss.
    """
    ce = F.cross_entropy(result.y1, targets)
    with torch.no_grad():
        preds = result.y1.argmax(dim=-1)
        correct = (preds == targets).float()
    conf_loss = F.mse_loss(result.conf_raw, correct)
    entropy_reg = (result.entropy * correct).mean()
    return ce_weight * ce + conf_weight * conf_loss + entropy_weight * entropy_reg


# ---------------------------------------------------------------------------
# System 2 hand-off
# ---------------------------------------------------------------------------


@dataclass
class System2Handoff:
    """Information packet passed from System 1 to System 2.

    Contains the hidden representation, raw logits, and all confidence
    signals so that System 2 can decide how much iterative refinement
    is needed.
    """

    hidden: Tensor
    logits: Tensor
    conf_calibrated: Tensor
    entropy: Tensor
    margin: Tensor
    deferred_mask: Tensor
    uncertainty_metrics: Dict[str, Tensor] = field(default_factory=dict)
    metadata: Dict[str, Union[float, int, str]] = field(default_factory=dict)


def prepare_system2_handoff(
    result: System1Result,
    threshold: float = 0.7,
) -> System2Handoff:
    """Create a hand-off packet for samples that need System 2 processing.

    Parameters
    ----------
    result : System1Result
        Full System 1 output.
    threshold : float
        Confidence threshold below which a sample is deferred.

    Returns
    -------
    System2Handoff
        Packet with a boolean ``deferred_mask`` indicating which samples
        in the batch are deferred.
    """
    deferred = result.conf_calibrated < threshold
    return System2Handoff(
        hidden=result.hidden,
        logits=result.y1,
        conf_calibrated=result.conf_calibrated,
        entropy=result.entropy,
        margin=result.margin,
        deferred_mask=deferred,
        uncertainty_metrics=result.uncertainty_metrics,
        metadata={
            "threshold": threshold,
            "n_deferred": int(deferred.sum().item()),
        },
    )


def merge_system1_system2(
    s1_result: System1Result,
    s2_logits: Tensor,
    deferred_mask: Tensor,
) -> Tensor:
    """Merge System 1 and System 2 logits based on deferral mask.

    For samples where ``deferred_mask`` is ``True``, use *s2_logits*;
    otherwise keep the System 1 logits.

    Parameters
    ----------
    s1_result : System1Result
        System 1 predictions.
    s2_logits : Tensor
        System 2 predictions ``(B, C)`` or ``(N_deferred, C)``.
    deferred_mask : Tensor
        Boolean mask ``(B,)`` indicating deferred samples.

    Returns
    -------
    Tensor
        Merged logits ``(B, C)``.
    """
    merged = s1_result.y1.clone()
    if s2_logits.shape[0] == deferred_mask.sum().item():
        merged[deferred_mask] = s2_logits
    else:
        merged[deferred_mask] = s2_logits[deferred_mask]
    return merged


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def save_system1(
    model: System1Fast,
    path: str,
    extra_metadata: Optional[Dict] = None,
) -> None:
    """Save model state together with its configuration.

    Parameters
    ----------
    model : System1Fast
        The model to save.
    path : str
        File path (typically ``.pt`` or ``.pth``).
    extra_metadata : dict, optional
        Arbitrary metadata to bundle with the checkpoint.
    """
    import dataclasses as _dc

    payload = {
        "state_dict": model.state_dict(),
        "config": _dc.asdict(model.config),
        "metadata": extra_metadata or {},
    }
    torch.save(payload, path)


def load_system1(
    path: str,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> System1Fast:
    """Load a System 1 model from a checkpoint.

    Parameters
    ----------
    path : str
        Checkpoint file path.
    device : torch.device, optional
        Device to map tensors to.
    strict : bool
        Whether to enforce strict key matching.

    Returns
    -------
    System1Fast
        Restored model.
    """
    payload = torch.load(path, map_location=device or "cpu", weights_only=False)
    cfg = System1Config(**payload["config"])
    model = System1Fast(cfg)
    model.load_state_dict(payload["state_dict"], strict=strict)
    if device is not None:
        model = model.to(device)
    return model


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------


def profile_forward_pass(
    model: System1Fast,
    input_shape: Tuple[int, ...] = (32, 4096),
    n_warmup: int = 5,
    n_runs: int = 20,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Profile the forward-pass latency.

    Parameters
    ----------
    model : System1Fast
        Model to profile.
    input_shape : tuple
        Shape of the synthetic input tensor.
    n_warmup : int
        Warmup iterations (not timed).
    n_runs : int
        Timed iterations.
    device : torch.device, optional
        Device for profiling.

    Returns
    -------
    dict
        Keys: ``mean_ms``, ``std_ms``, ``min_ms``, ``max_ms``.
    """
    import time as _time

    if device is not None:
        model = model.to(device)
    was_training = model.training
    model.train(False)
    x = torch.randn(*input_shape)
    if device is not None:
        x = x.to(device)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize()

        times: List[float] = []
        for _ in range(n_runs):
            t0 = _time.perf_counter()
            model(x)
            if device is not None and device.type == "cuda":
                torch.cuda.synchronize()
            t1 = _time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    model.train(was_training)

    import statistics as _stats

    return {
        "mean_ms": _stats.mean(times),
        "std_ms": _stats.stdev(times) if len(times) > 1 else 0.0,
        "min_ms": min(times),
        "max_ms": max(times),
    }


# ===========================================================================
# Self-tests
# ===========================================================================


def _run_self_tests() -> None:
    """Comprehensive self-tests for the System 1 fast predictor.

    Runs 12 tests covering shape correctness, confidence bounds, entropy
    non-negativity, determinism, pooling modes, gradient flow, confidence
    head toggling, activation types, and edge cases.

    Prints ``X/Y self-tests passed`` at the end.
    """
    torch.manual_seed(42)
    device: torch.device = torch.device("cpu")

    passed: int = 0
    total: int = 12
    failures: List[str] = []

    def _report(name: str, ok: bool, detail: str = "") -> None:
        nonlocal passed
        status: str = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failures.append(f"{name}: {detail}")
        msg: str = f"  [{status}] {name}"
        if detail and not ok:
            msg += f" -- {detail}"
        print(msg)

    print("=" * 70)
    print("System 1 Fast Predictor -- Self Tests")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Test 1: Forward (B, D) shape
    # -----------------------------------------------------------------------
    try:
        cfg = System1Config.minimal()
        model = System1Fast(cfg).to(device)
        model.train(False)

        B = 4
        x = torch.randn(B, cfg.input_dim, device=device)
        result = model(x)

        ok = (
            result.y1.shape == (B, cfg.output_dim)
            and result.conf_raw.shape == (B,)
            and result.conf_calibrated.shape == (B,)
            and result.entropy.shape == (B,)
            and result.margin.shape == (B,)
            and result.hidden.shape == (B, cfg.hidden_dim)
            and isinstance(result.uncertainty_metrics, dict)
        )
        _report(
            "Forward (B, D) shape",
            ok,
            f"y1={result.y1.shape}, conf_raw={result.conf_raw.shape}, "
            f"hidden={result.hidden.shape}",
        )
    except Exception as e:
        _report("Forward (B, D) shape", False, f"Exception: {e}")
        traceback.print_exc()

    # -----------------------------------------------------------------------
    # Test 2: Forward (B, K, D) slots shape
    # -----------------------------------------------------------------------
    try:
        cfg = System1Config.minimal()
        model = System1Fast(cfg).to(device)
        model.train(False)

        B, K = 3, 5
        x = torch.randn(B, K, cfg.input_dim, device=device)
        result = model(x)

        ok = (
            result.y1.shape == (B, cfg.output_dim)
            and result.hidden.shape == (B, cfg.hidden_dim)
            and result.conf_raw.shape == (B,)
        )
        _report(
            "Forward (B, K, D) slots shape",
            ok,
            f"y1={result.y1.shape}, hidden={result.hidden.shape}",
        )
    except Exception as e:
        _report("Forward (B, K, D) slots shape", False, f"Exception: {e}")
        traceback.print_exc()

    # -----------------------------------------------------------------------
    # Test 3: conf_raw in [0, 1]
    # -----------------------------------------------------------------------
    try:
        cfg = System1Config.minimal()
        model = System1Fast(cfg).to(device)
        model.train(False)

        B = 16
        x = torch.randn(B, cfg.input_dim, device=device)
        result = model(x)

        in_range = (result.conf_raw >= 0.0).all() and (result.conf_raw <= 1.0).all()
        ok = bool(in_range.item())
        _report(
            "conf_raw in [0, 1]",
            ok,
            f"min={result.conf_raw.min().item():.6f}, "
            f"max={result.conf_raw.max().item():.6f}",
        )
    except Exception as e:
        _report("conf_raw in [0, 1]", False, f"Exception: {e}")
        traceback.print_exc()

    # -----------------------------------------------------------------------
    # Test 4: entropy >= 0
    # -----------------------------------------------------------------------
    try:
        cfg = System1Config.minimal()
        model = System1Fast(cfg).to(device)
        model.train(False)

        B = 16
        x = torch.randn(B, cfg.input_dim, device=device)
        result = model(x)

        non_negative = (result.entropy >= 0.0).all()
        ok = bool(non_negative.item())
        _report(
            "entropy >= 0",
            ok,
            f"min={result.entropy.min().item():.6f}, "
            f"max={result.entropy.max().item():.6f}",
        )
    except Exception as e:
        _report("entropy >= 0", False, f"Exception: {e}")
        traceback.print_exc()

    # -----------------------------------------------------------------------
    # Test 5: Determinism
    # -----------------------------------------------------------------------
    try:
        cfg = System1Config.minimal()
        model = System1Fast(cfg).to(device)
        model.train(False)

        B = 4
        torch.manual_seed(123)
        x = torch.randn(B, cfg.input_dim, device=device)

        result1 = model(x)
        result2 = model(x)

        y_match = torch.allclose(result1.y1, result2.y1, atol=1e-6)
        conf_match = torch.allclose(result1.conf_raw, result2.conf_raw, atol=1e-6)
        ent_match = torch.allclose(result1.entropy, result2.entropy, atol=1e-6)

        ok = y_match and conf_match and ent_match
        _report(
            "Determinism",
            ok,
            f"y_match={y_match}, conf_match={conf_match}, "
            f"ent_match={ent_match}",
        )
    except Exception as e:
        _report("Determinism", False, f"Exception: {e}")
        traceback.print_exc()

    # -----------------------------------------------------------------------
    # Test 6: All 3 pool modes
    # -----------------------------------------------------------------------
    try:
        all_pool_ok = True
        pool_details: List[str] = []

        for mode in _SUPPORTED_POOL_MODES:
            cfg = System1Config(
                input_dim=32, hidden_dim=16, output_dim=8,
                num_layers=1, pool_mode=mode, dropout=0.0,
                confidence_head=False,
            )
            m = System1Fast(cfg).to(device)
            m.train(False)

            B, K = 2, 4
            x_3d = torch.randn(B, K, cfg.input_dim, device=device)
            r = m(x_3d)

            valid = (
                r.y1.shape == (B, cfg.output_dim)
                and not torch.isnan(r.y1).any()
            )
            pool_details.append(f"{mode}: ok={valid}")
            if not valid:
                all_pool_ok = False

        ok = all_pool_ok
        _report(
            "All 3 pool modes",
            ok,
            "; ".join(pool_details),
        )
    except Exception as e:
        _report("All 3 pool modes", False, f"Exception: {e}")
        traceback.print_exc()

    # -----------------------------------------------------------------------
    # Test 7: Gradient flow
    # -----------------------------------------------------------------------
    try:
        cfg = System1Config.minimal()
        model = System1Fast(cfg).to(device)
        model.train(True)

        B = 4
        x = torch.randn(B, cfg.input_dim, device=device, requires_grad=True)
        result = model(x)
        loss = result.y1.sum()
        loss.backward()

        input_grad_ok = x.grad is not None and x.grad.abs().sum().item() > 0

        body_grads: List[bool] = []
        for name, p in model.body.named_parameters():
            if p.grad is not None:
                body_grads.append(p.grad.abs().sum().item() > 0)

        logit_grads: List[bool] = []
        for name, p in model.logits_head.named_parameters():
            if p.grad is not None:
                logit_grads.append(p.grad.abs().sum().item() > 0)

        ok = input_grad_ok and all(body_grads) and all(logit_grads)
        _report(
            "Gradient flow",
            ok,
            f"input_grad={input_grad_ok}, "
            f"body_grads={body_grads}, "
            f"logit_grads={logit_grads}",
        )
    except Exception as e:
        _report("Gradient flow", False, f"Exception: {e}")
        traceback.print_exc()

    # -----------------------------------------------------------------------
    # Test 8: With and without confidence head
    # -----------------------------------------------------------------------
    try:
        results_info: Dict[str, bool] = {}

        # With confidence head
        cfg_with = System1Config(
            input_dim=32, hidden_dim=16, output_dim=8,
            num_layers=1, confidence_head=True, dropout=0.0,
        )
        m_with = System1Fast(cfg_with).to(device)
        m_with.train(False)
        r_with = m_with(torch.randn(2, 32, device=device))
        has_learned = "learned_confidence" in r_with.uncertainty_metrics
        has_module = m_with.confidence_head_module is not None
        results_info["with_head"] = has_learned and has_module

        # Without confidence head
        cfg_without = System1Config(
            input_dim=32, hidden_dim=16, output_dim=8,
            num_layers=1, confidence_head=False, dropout=0.0,
        )
        m_without = System1Fast(cfg_without).to(device)
        m_without.train(False)
        r_without = m_without(torch.randn(2, 32, device=device))
        no_learned = "learned_confidence" not in r_without.uncertainty_metrics
        no_module = m_without.confidence_head_module is None
        results_info["without_head"] = no_learned and no_module

        # conf_calibrated should equal conf_raw when head is off
        cal_eq_raw_off = torch.allclose(
            r_without.conf_calibrated, r_without.conf_raw, atol=1e-6
        )
        results_info["cal_eq_raw_when_off"] = cal_eq_raw_off

        ok = all(results_info.values())
        _report(
            "With/without confidence head",
            ok,
            str(results_info),
        )
    except Exception as e:
        _report("With/without confidence head", False, f"Exception: {e}")
        traceback.print_exc()

    # -----------------------------------------------------------------------
    # Test 9: All activation types
    # -----------------------------------------------------------------------
    try:
        all_act_ok = True
        act_details: List[str] = []

        for act_name in _SUPPORTED_ACTIVATIONS:
            cfg = System1Config(
                input_dim=32, hidden_dim=16, output_dim=8,
                num_layers=1, activation=act_name, dropout=0.0,
                confidence_head=False,
            )
            m = System1Fast(cfg).to(device)
            m.train(False)

            r = m(torch.randn(2, 32, device=device))
            valid = (
                r.y1.shape == (2, 8)
                and not torch.isnan(r.y1).any()
            )
            act_details.append(f"{act_name}: ok={valid}")
            if not valid:
                all_act_ok = False

        ok = all_act_ok
        _report(
            "All activation types",
            ok,
            "; ".join(act_details),
        )
    except Exception as e:
        _report("All activation types", False, f"Exception: {e}")
        traceback.print_exc()

    # -----------------------------------------------------------------------
    # Test 10: Edge case -- B=1
    # -----------------------------------------------------------------------
    try:
        cfg = System1Config.minimal()
        model = System1Fast(cfg).to(device)
        model.train(False)

        x = torch.randn(1, cfg.input_dim, device=device)
        result = model(x)

        ok = (
            result.y1.shape == (1, cfg.output_dim)
            and result.conf_raw.shape == (1,)
            and result.entropy.shape == (1,)
            and result.margin.shape == (1,)
            and result.hidden.shape == (1, cfg.hidden_dim)
        )
        _report(
            "Edge case: B=1",
            ok,
            f"y1={result.y1.shape}, conf_raw={result.conf_raw.shape}",
        )
    except Exception as e:
        _report("Edge case: B=1", False, f"Exception: {e}")
        traceback.print_exc()

    # -----------------------------------------------------------------------
    # Test 11: Edge case -- single class (output_dim=1)
    # -----------------------------------------------------------------------
    try:
        cfg = System1Config(
            input_dim=32, hidden_dim=16, output_dim=1,
            num_layers=1, confidence_head=False, dropout=0.0,
        )
        model = System1Fast(cfg).to(device)
        model.train(False)

        B = 4
        x = torch.randn(B, cfg.input_dim, device=device)
        result = model(x)

        # With single class: conf_raw should be 1.0, entropy should be 0.0,
        # margin should be 0.0
        conf_ok = torch.allclose(
            result.conf_raw,
            torch.ones(B, device=device),
            atol=1e-6,
        )
        ent_ok = torch.allclose(
            result.entropy,
            torch.zeros(B, device=device),
            atol=1e-6,
        )
        margin_ok = torch.allclose(
            result.margin,
            torch.zeros(B, device=device),
            atol=1e-6,
        )

        ok = conf_ok and ent_ok and margin_ok
        _report(
            "Edge case: single class (output_dim=1)",
            ok,
            f"conf_ok={conf_ok}, ent_ok={ent_ok}, margin_ok={margin_ok}",
        )
    except Exception as e:
        _report("Edge case: single class (output_dim=1)", False, f"Exception: {e}")
        traceback.print_exc()

    # -----------------------------------------------------------------------
    # Test 12: Pure confidence functions on known logits
    # -----------------------------------------------------------------------
    try:
        # Construct logits where class 0 clearly dominates
        logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]], device=device)

        cr = compute_conf_raw(logits)
        ent = compute_entropy(logits)
        mar = compute_margin(logits)
        metrics = compute_uncertainty_metrics(logits)

        cr_ok = cr.item() > 0.99  # softmax heavily peaked
        ent_ok = ent.item() < 0.1  # very low entropy
        mar_ok = mar.item() > 9.0  # 10 - ~0 = ~10
        metrics_ok = (
            "conf_raw" in metrics
            and "entropy" in metrics
            and "margin" in metrics
        )

        ok = cr_ok and ent_ok and mar_ok and metrics_ok
        _report(
            "Pure confidence functions on known logits",
            ok,
            f"conf_raw={cr.item():.4f}, entropy={ent.item():.4f}, "
            f"margin={mar.item():.4f}, metrics_keys={list(metrics.keys())}",
        )
    except Exception as e:
        _report(
            "Pure confidence functions on known logits",
            False,
            f"Exception: {e}",
        )
        traceback.print_exc()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("=" * 70)
    print(f"{passed}/{total} self-tests passed")
    if failures:
        print("Failures:")
        for f in failures:
            print(f"  - {f}")
    print("=" * 70)


# ===========================================================================
# Entry point
# ===========================================================================


if __name__ == "__main__":
    _run_self_tests()
