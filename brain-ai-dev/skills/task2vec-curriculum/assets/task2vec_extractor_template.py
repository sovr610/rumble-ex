"""
Task2Vec Embedding Extractor — Complete Self-Contained Module.

Implements the Task2Vec pipeline for computing fixed-dimensional Fisher-information-based
embeddings of few-shot learning episodes. These embeddings enable curriculum ordering,
diversity-constrained meta-batch composition, and offline task-space analysis.

Three-stage pipeline:
    1. Probe forward pass: Fixed pretrained backbone + ephemeral N-way head (frozen)
    2. Diagonal Fisher estimation: Per-sample gradient squared, accumulated (fp32)
    3. Projection: Layer subset selection, per-group aggregation, log1p + L2-norm

The probe network is NOT the meta-learner -- it is a fixed reference model. Embeddings are
relative to this probe, making them comparable across different meta-learning runs.

Usage::

    from task2vec_extractor_template import (
        Task2VecConfig,
        Task2VecExtractor,
        TaskEpisode,
    )

    config = Task2VecConfig(probe_model="conv4", embedding_dim=512)
    extractor = Task2VecExtractor(config)

    episode = TaskEpisode(
        task_id="auto",
        x_support=x_s,
        y_support=y_s,
        x_query=x_q,
        y_query=y_q,
        dataset="miniImageNet",
        split="train",
        n_way=5,
        k_shot=5,
        q_query=15,
        class_ids=[0, 3, 7, 12, 45],
    )
    result = extractor.extract(episode, seed=42, device="cpu")
    print(result.embedding.shape)  # (512,)

Reference:
    Achille et al., "Task2Vec: Task Embedding for Meta-Learning", ICCV 2019.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Optional imports (guarded)
# ---------------------------------------------------------------------------
_HAS_TORCHVISION = False
try:
    import torchvision  # noqa: F401
    import torchvision.models as tv_models  # noqa: F401

    _HAS_TORCHVISION = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ============================================================================
# Section 1 — Data Classes
# ============================================================================


@dataclass
class TaskEpisode:
    """Represents a single few-shot learning episode.

    Attributes:
        task_id: Deterministic hash of the episode contents.  Set to ``"auto"``
            to have it computed automatically via :func:`canonicalize_task_id`.
        x_support: Support-set inputs ``(N*K, C, H, W)`` or ``(N*K, D)``.
        y_support: Support-set labels ``(N*K,)`` with values in ``[0, N)``.
        x_query: Query-set inputs (same spatial shape as support).
        y_query: Query-set labels.
        dataset: Name of the originating dataset (e.g. ``"miniImageNet"``).
        split: Data split — ``"train"``, ``"val"``, or ``"test"``.
        n_way: Number of classes in the episode.
        k_shot: Shots per class in the support set.
        q_query: Queries per class in the query set.
        class_ids: Original class indices drawn from the dataset.
        support_indices: Optional per-sample indices into the dataset.
        transforms_signature: Optional hash / description of applied transforms.
    """

    task_id: str
    x_support: torch.Tensor
    y_support: torch.Tensor
    x_query: Optional[torch.Tensor] = None
    y_query: Optional[torch.Tensor] = None
    dataset: str = "unknown"
    split: str = "train"
    n_way: int = 5
    k_shot: int = 5
    q_query: int = 15
    class_ids: Optional[List[int]] = None
    support_indices: Optional[List[int]] = None
    transforms_signature: Optional[str] = None


@dataclass
class ExtractionDiagnostics:
    """Diagnostics emitted after a single embedding extraction.

    Attributes:
        fisher_norm: Frobenius norm of the (projected) diagonal-Fisher vector.
        sparsity: Fraction of Fisher entries that are exactly zero.
        probe_loss: Cross-entropy loss of the probe on the support set.
        extraction_time_ms: Wall-clock time for the entire extraction in ms.
        n_samples: Number of support samples used for Fisher estimation.
        n_params: Number of probe parameters included in the Fisher.
    """

    fisher_norm: float = 0.0
    sparsity: float = 0.0
    probe_loss: float = 0.0
    extraction_time_ms: float = 0.0
    n_samples: int = 0
    n_params: int = 0

    def to_dict(self) -> Dict[str, float]:
        """Serialize to a flat dictionary of floats."""
        return {
            "fisher_norm": self.fisher_norm,
            "sparsity": self.sparsity,
            "probe_loss": self.probe_loss,
            "extraction_time_ms": self.extraction_time_ms,
            "n_samples": float(self.n_samples),
            "n_params": float(self.n_params),
        }


@dataclass
class TaskEmbedding:
    """Result of extracting a Task2Vec embedding for one episode.

    Attributes:
        embedding: L2-normalized Fisher-derived task vector of shape ``(E,)``.
        task_id: Deterministic hash identifying this episode.
        diagnostics: Dictionary of extraction diagnostics.
        probe_signature: Hash identifying the probe model + config used.
    """

    embedding: torch.Tensor
    task_id: str
    diagnostics: Dict[str, float]
    probe_signature: str


# ============================================================================
# Section 2 — Configuration
# ============================================================================


@dataclass
class Task2VecConfig:
    """Configuration for the Task2Vec embedding extractor.

    Attributes:
        probe_model: Probe backbone name — ``"conv4"``, ``"resnet12"``.
        pretrained: Whether to load pretrained weights (requires torchvision).
        layer_subset: Which layers to include in the Fisher —
            ``"last_block"``, ``"per_stage"``, ``"all"``.
        embedding_dim: Target embedding dimension E.
        aggregation: Fisher aggregation strategy —
            ``"per_channel"``, ``"per_layer"``.
        normalize: Normalization pipeline —
            ``"log1p_l2"``, ``"l2"``, ``"whiten"``.
        num_fisher_samples: Number of support samples for Fisher estimation.
            ``None`` means use all support samples.
        input_channels: Number of input channels for the probe (e.g. 3 for RGB).
        probe_seed: Seed for probe weight initialization (for reproducibility).
    """

    probe_model: str = "conv4"
    pretrained: bool = False
    layer_subset: str = "last_block"
    embedding_dim: int = 512
    aggregation: str = "per_channel"
    normalize: str = "log1p_l2"
    num_fisher_samples: Optional[int] = None
    input_channels: int = 3
    probe_seed: int = 0

    def signature(self) -> str:
        """Compute a deterministic signature of this configuration."""
        parts = [
            self.probe_model,
            str(self.pretrained),
            self.layer_subset,
            str(self.embedding_dim),
            self.aggregation,
            self.normalize,
            str(self.input_channels),
            str(self.probe_seed),
        ]
        raw = "|".join(parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# ============================================================================
# Section 3 — Task ID Canonicalization
# ============================================================================


def canonicalize_task_id(
    dataset: str,
    split: str,
    class_ids: Optional[List[int]],
    support_indices: Optional[List[int]] = None,
    transforms_signature: Optional[str] = None,
) -> str:
    """Compute a deterministic task ID from episode metadata.

    The hash is computed from:
        - dataset name (lowercased, stripped)
        - split name (lowercased, stripped)
        - sorted class IDs
        - sorted support indices (if provided)
        - transforms signature (if provided)

    Given identical inputs the output is always the same SHA-256 hex digest
    (first 32 characters).

    Args:
        dataset: Name of the originating dataset.
        split: Data split.
        class_ids: List of original class indices.
        support_indices: Optional per-sample indices.
        transforms_signature: Optional hash of applied transforms.

    Returns:
        A 32-character hex string uniquely identifying this task.
    """
    parts: List[str] = []
    parts.append(f"dataset={dataset.strip().lower()}")
    parts.append(f"split={split.strip().lower()}")

    if class_ids is not None:
        sorted_cids = sorted(int(c) for c in class_ids)
        parts.append(f"class_ids={sorted_cids}")
    else:
        parts.append("class_ids=none")

    if support_indices is not None:
        sorted_sidx = sorted(int(s) for s in support_indices)
        parts.append(f"support_indices={sorted_sidx}")
    else:
        parts.append("support_indices=none")

    if transforms_signature is not None:
        parts.append(f"transforms={transforms_signature.strip()}")
    else:
        parts.append("transforms=none")

    raw = "|".join(parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return digest[:32]


# ============================================================================
# Section 4 — Probe Networks
# ============================================================================


class Conv4Probe(nn.Module):
    """4-layer convolutional probe network.

    Standard few-shot backbone with four convolutional blocks, each consisting
    of a 3x3 convolution, batch normalization, ReLU activation, and 2x2 max
    pooling.  All four blocks use 64 output channels.

    For a 28x28 input the output spatial size is 1x1 (28 -> 14 -> 7 -> 3 -> 1).
    For an 84x84 input the output spatial size is 5x5 (84 -> 42 -> 21 -> 10 -> 5).

    The ``feature_dim`` attribute gives the flattened feature dimension.

    Args:
        in_channels: Number of input channels.
    """

    _STAGE_NAMES = ["block1", "block2", "block3", "block4"]

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.in_channels = in_channels

        self.block1 = self._make_block(in_channels, 64)
        self.block2 = self._make_block(64, 64)
        self.block3 = self._make_block(64, 64)
        self.block4 = self._make_block(64, 64)

        # Feature dim will be set lazily on first forward pass
        self._feature_dim: Optional[int] = None

    @staticmethod
    def _make_block(in_ch: int, out_ch: int) -> nn.Sequential:
        """Create one convolutional block: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    @property
    def feature_dim(self) -> int:
        """Return the flattened feature dimension.

        If not yet computed, run a dummy forward pass to determine it.
        """
        if self._feature_dim is None:
            with torch.no_grad():
                dummy = torch.zeros(1, self.in_channels, 28, 28)
                out = self._forward_features(dummy)
                self._feature_dim = out.shape[1]
        return self._feature_dim

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run all four blocks and flatten."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.view(x.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning flattened features.

        Args:
            x: Input tensor ``(B, C, H, W)``.

        Returns:
            Feature tensor ``(B, feature_dim)``.
        """
        feat = self._forward_features(x)
        if self._feature_dim is None:
            self._feature_dim = feat.shape[1]
        return feat

    def get_layer_subset(self, mode: str) -> List[str]:
        """Return parameter name prefixes for the requested layer subset.

        Args:
            mode: One of ``"last_block"``, ``"per_stage"``, ``"all"``.

        Returns:
            List of parameter name prefixes.
        """
        if mode == "last_block":
            return ["block4"]
        elif mode == "per_stage":
            return ["block1", "block2", "block3", "block4"]
        elif mode == "all":
            return ["block1", "block2", "block3", "block4"]
        else:
            raise ValueError(f"Unknown layer subset mode: {mode!r}")

    def get_stage_names(self) -> List[str]:
        """Return ordered stage/block names."""
        return list(self._STAGE_NAMES)


class _ResNetBasicBlock(nn.Module):
    """Basic residual block for ResNet-12.

    Two-layer block: conv3x3-BN-ReLU-conv3x3-BN with a skip connection
    that optionally applies a 1x1 convolution for channel matching.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity, inplace=True)
        return out


class ResNet12Probe(nn.Module):
    """ResNet-12 probe network.

    A stronger few-shot backbone with four stages of increasing width:
    64 -> 160 -> 320 -> 640. Each stage consists of a single residual block
    followed by 2x2 max pooling.

    For an 84x84 input the output spatial size is 5x5 after four pooling
    layers (84 -> 42 -> 21 -> 10 -> 5), giving feature_dim = 640 * 5 * 5 = 16000.
    For a 28x28 input: 28 -> 14 -> 7 -> 3 -> 1, giving feature_dim = 640.

    Args:
        in_channels: Number of input channels.
    """

    _STAGE_NAMES = ["stage1", "stage2", "stage3", "stage4"]
    _STAGE_CHANNELS = [64, 160, 320, 640]

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.in_channels = in_channels

        channels = self._STAGE_CHANNELS
        self.stage1 = self._make_stage(in_channels, channels[0])
        self.stage2 = self._make_stage(channels[0], channels[1])
        self.stage3 = self._make_stage(channels[1], channels[2])
        self.stage4 = self._make_stage(channels[2], channels[3])

        self._feature_dim: Optional[int] = None

    @staticmethod
    def _make_stage(in_ch: int, out_ch: int) -> nn.Sequential:
        """Create one stage: BasicBlock -> MaxPool2d."""
        return nn.Sequential(
            _ResNetBasicBlock(in_ch, out_ch, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    @property
    def feature_dim(self) -> int:
        """Return the flattened feature dimension (lazy computation)."""
        if self._feature_dim is None:
            with torch.no_grad():
                dummy = torch.zeros(1, self.in_channels, 28, 28)
                out = self._forward_features(dummy)
                self._feature_dim = out.shape[1]
        return self._feature_dim

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run all four stages and flatten."""
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x.view(x.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning flattened features.

        Args:
            x: Input tensor ``(B, C, H, W)``.

        Returns:
            Feature tensor ``(B, feature_dim)``.
        """
        feat = self._forward_features(x)
        if self._feature_dim is None:
            self._feature_dim = feat.shape[1]
        return feat

    def get_layer_subset(self, mode: str) -> List[str]:
        """Return parameter name prefixes for the requested layer subset.

        Args:
            mode: One of ``"last_block"``, ``"per_stage"``, ``"all"``.

        Returns:
            List of parameter name prefixes.
        """
        if mode == "last_block":
            return ["stage4"]
        elif mode == "per_stage":
            return ["stage1", "stage2", "stage3", "stage4"]
        elif mode == "all":
            return ["stage1", "stage2", "stage3", "stage4"]
        else:
            raise ValueError(f"Unknown layer subset mode: {mode!r}")

    def get_stage_names(self) -> List[str]:
        """Return ordered stage/block names."""
        return list(self._STAGE_NAMES)


class ProbeWithHead(nn.Module):
    """Wraps a probe backbone with an ephemeral linear classification head.

    The backbone is always kept frozen (inference mode, requires_grad=False).
    Only the head is trainable during Fisher estimation (though it is
    discarded immediately after).

    Args:
        backbone: A probe backbone (e.g. :class:`Conv4Probe`).
        n_way: Number of classes for the classification head.
        freeze_backbone: If True (default), freeze backbone parameters.
    """

    def __init__(
        self,
        backbone: nn.Module,
        n_way: int,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.n_way = n_way

        # Determine feature dimension
        feat_dim = backbone.feature_dim
        self.head = nn.Linear(feat_dim, n_way)

        if freeze_backbone:
            self._set_backbone_frozen()

    def _set_backbone_frozen(self) -> None:
        """Put backbone into inference mode with all grads disabled."""
        self.backbone.requires_grad_(False)
        # Use train(False) to disable dropout and use running stats in BN
        self.backbone.train(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: backbone features -> linear head logits.

        When backbone parameters have ``requires_grad=True`` (e.g. during
        Fisher computation), gradients flow through the backbone.  Otherwise
        the backbone forward is wrapped in ``torch.no_grad()`` for efficiency.

        Args:
            x: Input tensor ``(B, C, H, W)``.

        Returns:
            Logits tensor ``(B, n_way)``.
        """
        # Check if any backbone param needs grad (Fisher computation mode)
        any_grad = any(p.requires_grad for p in self.backbone.parameters())
        if any_grad:
            features = self.backbone(x)
        else:
            with torch.no_grad():
                features = self.backbone(x)
        logits = self.head(features)
        return logits

    def get_fisher_params(self, layer_subset: str) -> Dict[str, nn.Parameter]:
        """Return the parameters to use for Fisher computation.

        This includes both the backbone parameters matching the layer subset
        AND the head parameters.

        Args:
            layer_subset: Layer subset mode (passed to backbone's ``get_layer_subset``).

        Returns:
            Dictionary mapping parameter name to parameter tensor.
        """
        prefixes = self.backbone.get_layer_subset(layer_subset)
        params: Dict[str, nn.Parameter] = {}

        for name, param in self.backbone.named_parameters():
            for prefix in prefixes:
                if name.startswith(prefix):
                    params[f"backbone.{name}"] = param
                    break

        # Always include head parameters
        for name, param in self.head.named_parameters():
            params[f"head.{name}"] = param

        return params


def create_probe(
    name: str,
    *,
    in_channels: int = 3,
    pretrained: bool = False,
    seed: int = 0,
) -> nn.Module:
    """Factory function for creating probe backbone networks.

    Supported probe names:
        - ``"conv4"``: 4-layer convolutional network (64 channels each).
        - ``"resnet12"``: ResNet-12 with stages 64->160->320->640.

    Args:
        name: Probe backbone name.
        in_channels: Number of input channels.
        pretrained: Whether to load pretrained weights (requires torchvision).
        seed: Random seed for weight initialization.

    Returns:
        A probe backbone module.

    Raises:
        ValueError: If the probe name is not recognized.
    """
    # Seed for reproducible initialization
    gen = torch.Generator()
    gen.manual_seed(seed)

    if name == "conv4":
        probe = Conv4Probe(in_channels=in_channels)
    elif name == "resnet12":
        probe = ResNet12Probe(in_channels=in_channels)
    else:
        raise ValueError(
            f"Unknown probe model: {name!r}. Supported: 'conv4', 'resnet12'."
        )

    # Re-initialize weights with the seeded generator for reproducibility
    _reinit_parameters(probe, gen)

    if pretrained and _HAS_TORCHVISION:
        logger.info("Pretrained weights requested but custom probe used; skipping.")

    return probe


def _reinit_parameters(module: nn.Module, generator: torch.Generator) -> None:
    """Re-initialize all parameters of a module using the given generator.

    Applies Kaiming uniform for Conv2d weights, ones/zeros for BatchNorm,
    and Xavier uniform for Linear layers.

    Args:
        module: Module whose parameters will be re-initialized.
        generator: Random number generator for reproducibility.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
            std = math.sqrt(2.0 / fan_in)
            with torch.no_grad():
                m.weight.normal_(0, std, generator=generator)
            if m.bias is not None:
                with torch.no_grad():
                    m.bias.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                with torch.no_grad():
                    m.weight.fill_(1.0)
            if m.bias is not None:
                with torch.no_grad():
                    m.bias.zero_()
            if m.running_mean is not None:
                m.running_mean.zero_()
            if m.running_var is not None:
                m.running_var.fill_(1.0)
        elif isinstance(m, nn.Linear):
            fan_in = m.in_features
            std = 1.0 / math.sqrt(fan_in)
            with torch.no_grad():
                m.weight.uniform_(-std, std, generator=generator)
            if m.bias is not None:
                with torch.no_grad():
                    m.bias.uniform_(-std, std, generator=generator)


# ============================================================================
# Section 5 — Diagonal Fisher Computation
# ============================================================================


def compute_diagonal_fisher(
    probe: nn.Module,
    x_support: torch.Tensor,
    y_support: torch.Tensor,
    *,
    param_names: Optional[List[str]] = None,
    num_samples: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Compute the diagonal Fisher information matrix for the probe.

    For each sample in the support set, computes the gradient of the
    negative log-likelihood (cross-entropy loss) with respect to the
    specified parameters, then accumulates the squared gradients:

        F_diag[p] = (1/N) * sum_i (d loss_i / d p)^2

    All computation is forced to fp32 to ensure numerical reproducibility
    even under AMP autocast contexts.

    Args:
        probe: The probe model (with head) in inference mode.
        x_support: Support-set inputs ``(N_total, C, H, W)``.
        y_support: Support-set labels ``(N_total,)``.
        param_names: List of parameter names to compute Fisher for.
            If ``None``, all parameters with ``requires_grad=True`` are used.
        num_samples: Number of samples to use (``None`` means all).

    Returns:
        Dictionary mapping parameter name to diagonal Fisher tensor (same
        shape as the parameter, dtype=float32).
    """
    device = x_support.device

    # Force fp32
    x_fp32 = x_support.float()
    y_labels = y_support.long()

    # Determine which samples to use
    n_total = x_fp32.shape[0]
    if num_samples is not None and num_samples < n_total:
        indices = torch.randperm(n_total, device=device)[:num_samples]
        x_fp32 = x_fp32[indices]
        y_labels = y_labels[indices]
        n_total = num_samples

    # Collect parameters
    if param_names is not None:
        all_params = dict(probe.named_parameters())
        params: Dict[str, nn.Parameter] = {}
        for pn in param_names:
            if pn in all_params:
                params[pn] = all_params[pn]
            else:
                logger.warning("Parameter %r not found in probe; skipping.", pn)
    else:
        params = {
            name: p for name, p in probe.named_parameters() if p.requires_grad
        }

    if not params:
        logger.warning("No parameters selected for Fisher computation.")
        return {}

    # Temporarily enable gradients on selected parameters
    original_requires_grad: Dict[str, bool] = {}
    for name, p in params.items():
        original_requires_grad[name] = p.requires_grad
        p.requires_grad_(True)

    # Initialize Fisher accumulators
    fisher_diag: Dict[str, torch.Tensor] = {}
    for name, p in params.items():
        fisher_diag[name] = torch.zeros_like(p, dtype=torch.float32, device=device)

    # Accumulate per-sample squared gradients
    probe_was_training = probe.training
    probe.train(False)

    param_list = list(params.values())
    param_name_list = list(params.keys())

    total_loss = 0.0

    for i in range(n_total):
        xi = x_fp32[i : i + 1]
        yi = y_labels[i : i + 1]

        # Forward pass (force fp32)
        with torch.amp.autocast("cuda", enabled=False):
            logits = probe(xi)
            logits_fp32 = logits.float()
            loss = F.cross_entropy(logits_fp32, yi)

        total_loss += loss.item()

        # Compute gradients
        grads = torch.autograd.grad(
            loss,
            param_list,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        # Accumulate squared gradients
        for j, (g, pname) in enumerate(zip(grads, param_name_list)):
            if g is not None:
                fisher_diag[pname] += g.float().detach().pow(2)

    # Normalize by number of samples
    if n_total > 0:
        for name in fisher_diag:
            fisher_diag[name] /= float(n_total)

    # Restore original requires_grad state
    for name, p in params.items():
        p.requires_grad_(original_requires_grad[name])

    if probe_was_training:
        probe.train(True)

    avg_loss = total_loss / max(n_total, 1)
    logger.debug(
        "Fisher computation: %d samples, %d params, avg loss=%.4f",
        n_total,
        len(params),
        avg_loss,
    )

    return fisher_diag


def _compute_probe_loss(
    probe: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    """Compute average cross-entropy loss of the probe on the given data.

    Args:
        probe: Probe model in inference mode.
        x: Input tensor.
        y: Label tensor.

    Returns:
        Scalar loss value.
    """
    was_training = probe.training
    probe.train(False)
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=False):
            logits = probe(x.float())
            loss = F.cross_entropy(logits.float(), y.long())
    if was_training:
        probe.train(True)
    return loss.item()


# ============================================================================
# Section 6 — Embedding Projection
# ============================================================================


def project_fisher_to_embedding(
    fisher_diag: Dict[str, torch.Tensor],
    *,
    layer_subset: str = "last_block",
    aggregation: str = "per_channel",
    embedding_dim: int = 512,
    probe: Optional[nn.Module] = None,
) -> torch.Tensor:
    """Project the diagonal Fisher dictionary into a fixed-dimensional embedding.

    Three steps:
        1. **Layer subset selection** — filter Fisher entries by parameter name
           prefix matching the requested subset (requires ``probe``).
        2. **Aggregation** — reduce each parameter's Fisher to a summary vector:
           - ``"per_channel"``: mean over spatial dims per output channel.
           - ``"per_layer"``: single scalar mean per parameter tensor.
        3. **Normalization** — concatenate, pad/truncate to ``embedding_dim``,
           apply ``log1p``, then L2-normalize.

    Args:
        fisher_diag: Dictionary mapping parameter name to diagonal Fisher tensor.
        layer_subset: Which layers to include (``"last_block"``, ``"per_stage"``,
            ``"all"``).  If ``probe`` is ``None``, all entries are used regardless.
        aggregation: Aggregation strategy.
        embedding_dim: Target embedding dimension E.
        probe: Optional probe module for layer subset resolution.  If ``None``,
            all Fisher entries are used.

    Returns:
        L2-normalized embedding tensor of shape ``(embedding_dim,)``, dtype float32.
    """
    # --- Step 1: Layer subset selection ---
    if probe is not None and hasattr(probe, "backbone"):
        backbone = probe.backbone
        if hasattr(backbone, "get_layer_subset"):
            prefixes = backbone.get_layer_subset(layer_subset)
            # Filter Fisher entries whose name matches any prefix (with backbone. prefix)
            filtered: Dict[str, torch.Tensor] = {}
            for name, val in fisher_diag.items():
                # Check with backbone. prefix (ProbeWithHead names)
                bare_name = name.replace("backbone.", "")
                for prefix in prefixes:
                    if bare_name.startswith(prefix):
                        filtered[name] = val
                        break
                # Always include head parameters
                if name.startswith("head."):
                    filtered[name] = val
            fisher_diag = filtered if filtered else fisher_diag
    elif probe is not None and hasattr(probe, "get_layer_subset"):
        prefixes = probe.get_layer_subset(layer_subset)
        filtered = {}
        for name, val in fisher_diag.items():
            for prefix in prefixes:
                if name.startswith(prefix):
                    filtered[name] = val
                    break
        fisher_diag = filtered if filtered else fisher_diag

    if not fisher_diag:
        logger.warning("No Fisher entries after layer subset filtering; returning zeros.")
        return torch.zeros(embedding_dim, dtype=torch.float32)

    # --- Step 2: Aggregation ---
    aggregated_parts: List[torch.Tensor] = []

    # Sort keys for deterministic ordering
    sorted_names = sorted(fisher_diag.keys())

    for name in sorted_names:
        f = fisher_diag[name].float()

        if aggregation == "per_channel":
            if f.dim() >= 2:
                # For conv weights: (out_ch, in_ch, kH, kW) -> mean over dims 1,2,3
                # For linear weights: (out, in) -> mean over dim 1
                agg = f.mean(dim=tuple(range(1, f.dim())))  # shape: (out_ch,)
            else:
                # Bias or 1-D param: keep as-is
                agg = f
        elif aggregation == "per_layer":
            # Single scalar per parameter tensor
            agg = f.mean().unsqueeze(0)  # shape: (1,)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation!r}")

        aggregated_parts.append(agg.flatten())

    # Concatenate all aggregated parts
    raw_embedding = torch.cat(aggregated_parts, dim=0)

    # --- Step 3: Pad or truncate to embedding_dim ---
    current_dim = raw_embedding.shape[0]
    if current_dim < embedding_dim:
        # Pad with zeros
        padding = torch.zeros(
            embedding_dim - current_dim,
            dtype=torch.float32,
            device=raw_embedding.device,
        )
        raw_embedding = torch.cat([raw_embedding, padding], dim=0)
    elif current_dim > embedding_dim:
        # Truncate (keep first embedding_dim elements)
        raw_embedding = raw_embedding[:embedding_dim]

    # --- Step 4: log1p + L2-normalize ---
    # log1p is applied to non-negative Fisher values
    embedding = torch.log1p(raw_embedding)

    # L2 normalize
    norm = embedding.norm(p=2)
    if norm > 0:
        embedding = embedding / norm

    return embedding.detach()


# ============================================================================
# Section 7 — Whitening Transform
# ============================================================================


class WhiteningTransform:
    """Whitening transform for Task2Vec embeddings.

    Fits a whitening transformation from a set of reference embeddings, then
    applies it to new embeddings. This removes correlations and normalizes
    variance, making cosine-distance comparisons more meaningful.

    The transform is:
        z_white = W @ (z - mu)

    where ``mu`` is the mean of the reference embeddings and ``W`` is the
    inverse square root of the covariance matrix (with Tikhonov regularization
    for numerical stability).

    Args:
        reg: Regularization coefficient added to the covariance diagonal.
    """

    def __init__(self, reg: float = 1e-5) -> None:
        self.reg = reg
        self._mean: Optional[torch.Tensor] = None
        self._whiten_matrix: Optional[torch.Tensor] = None
        self._is_fitted: bool = False
        self._n_ref: int = 0

    @property
    def is_fitted(self) -> bool:
        """Whether the transform has been fitted."""
        return self._is_fitted

    def fit(self, embeddings: torch.Tensor) -> "WhiteningTransform":
        """Fit the whitening transform from reference embeddings.

        Args:
            embeddings: Reference embeddings of shape ``(N, E)`` where N >= 2.

        Returns:
            Self (for chaining).

        Raises:
            ValueError: If fewer than 2 embeddings are provided.
        """
        if embeddings.dim() != 2:
            raise ValueError(
                f"Expected 2-D embeddings (N, E), got shape {embeddings.shape}"
            )
        n, e = embeddings.shape
        if n < 2:
            raise ValueError(
                f"Need at least 2 reference embeddings for whitening, got {n}"
            )

        emb = embeddings.float()
        self._mean = emb.mean(dim=0)  # (E,)
        centered = emb - self._mean  # (N, E)

        # Covariance matrix (E, E)
        cov = (centered.T @ centered) / (n - 1)

        # Regularize
        cov += self.reg * torch.eye(e, dtype=torch.float32, device=cov.device)

        # Inverse square root via eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        # Clamp eigenvalues for numerical stability
        eigenvalues = eigenvalues.clamp(min=self.reg)
        inv_sqrt_eigenvalues = 1.0 / eigenvalues.sqrt()
        # W = V @ diag(1/sqrt(lambda)) @ V^T
        self._whiten_matrix = (
            eigenvectors @ torch.diag(inv_sqrt_eigenvalues) @ eigenvectors.T
        )

        self._is_fitted = True
        self._n_ref = n
        logger.info(
            "WhiteningTransform fitted on %d embeddings of dim %d.", n, e
        )
        return self

    def transform(self, embedding: torch.Tensor) -> torch.Tensor:
        """Apply the whitening transform to an embedding.

        Args:
            embedding: Embedding tensor of shape ``(E,)`` or ``(N, E)``.

        Returns:
            Whitened embedding of the same shape, L2-normalized.

        Raises:
            RuntimeError: If the transform has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "WhiteningTransform has not been fitted. Call fit() first."
            )
        assert self._mean is not None
        assert self._whiten_matrix is not None

        emb = embedding.float()
        squeeze = False
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
            squeeze = True

        mean = self._mean.to(emb.device)
        W = self._whiten_matrix.to(emb.device)

        centered = emb - mean  # (N, E)
        whitened = centered @ W.T  # (N, E)

        # L2 normalize each embedding
        norms = whitened.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
        whitened = whitened / norms

        if squeeze:
            whitened = whitened.squeeze(0)

        return whitened.detach()

    def fit_transform(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Fit on reference embeddings and transform them.

        Args:
            embeddings: Reference embeddings ``(N, E)``.

        Returns:
            Whitened embeddings ``(N, E)``.
        """
        self.fit(embeddings)
        return self.transform(embeddings)


# ============================================================================
# Section 8 — Task2VecExtractor
# ============================================================================


class Task2VecExtractor:
    """Main extractor class for computing Task2Vec embeddings.

    Orchestrates the full pipeline:
        1. Create / reuse a frozen probe backbone.
        2. Attach an ephemeral N-way head for the episode.
        3. Compute the diagonal Fisher on the support set.
        4. Project Fisher to a fixed-dimensional embedding.
        5. Return the embedding with diagnostics.

    The probe backbone is created once and reused across extractions.
    The classification head is re-created for each episode (since N-way
    may differ).

    Args:
        config: Configuration for the extractor.
    """

    def __init__(self, config: Task2VecConfig) -> None:
        self.config = config
        self._probe_backbone: Optional[nn.Module] = None
        self._probe_signature: str = config.signature()
        self._whitening: Optional[WhiteningTransform] = None

    @property
    def probe_signature(self) -> str:
        """Return the probe configuration signature."""
        return self._probe_signature

    def _get_or_create_backbone(self, device: torch.device) -> nn.Module:
        """Get the cached backbone or create a new one.

        Args:
            device: Device for the backbone.

        Returns:
            Frozen probe backbone on the specified device.
        """
        if self._probe_backbone is None:
            logger.info(
                "Creating probe backbone: %s (seed=%d)",
                self.config.probe_model,
                self.config.probe_seed,
            )
            self._probe_backbone = create_probe(
                self.config.probe_model,
                in_channels=self.config.input_channels,
                pretrained=self.config.pretrained,
                seed=self.config.probe_seed,
            )
            # Freeze and set to inference mode
            self._probe_backbone.train(False)
            for p in self._probe_backbone.parameters():
                p.requires_grad = False

        backbone = self._probe_backbone.to(device)
        backbone.train(False)
        return backbone

    def _snapshot_params(self, module: nn.Module) -> Dict[str, torch.Tensor]:
        """Take a snapshot of all parameter values for later comparison.

        Args:
            module: Module to snapshot.

        Returns:
            Dictionary of parameter name -> cloned tensor.
        """
        return {name: p.data.clone() for name, p in module.named_parameters()}

    def _verify_frozen(
        self,
        module: nn.Module,
        snapshot: Dict[str, torch.Tensor],
    ) -> bool:
        """Verify that no parameters have changed since the snapshot.

        Args:
            module: Module to check.
            snapshot: Previous snapshot from :meth:`_snapshot_params`.

        Returns:
            True if all parameters match, False otherwise.
        """
        for name, p in module.named_parameters():
            if name in snapshot:
                if not torch.equal(p.data, snapshot[name]):
                    logger.error("Parameter %r changed during extraction!", name)
                    return False
        return True

    def extract(
        self,
        episode: TaskEpisode,
        *,
        seed: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> TaskEmbedding:
        """Extract a Task2Vec embedding from a single episode.

        Full pipeline:
            1. Resolve task ID (auto-canonicalize if needed).
            2. Create probe with ephemeral head for this episode's N-way.
            3. Compute diagonal Fisher on support set.
            4. Project Fisher to embedding.
            5. Package result with diagnostics.

        Args:
            episode: The few-shot episode to embed.
            seed: Random seed for reproducibility.
            device: Device for computation.

        Returns:
            A :class:`TaskEmbedding` with the L2-normalized embedding and metadata.
        """
        t_start = time.perf_counter()
        dev = torch.device(device)

        # --- Seed for determinism ---
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # --- Resolve task ID ---
        task_id = episode.task_id
        if task_id == "auto":
            task_id = canonicalize_task_id(
                dataset=episode.dataset,
                split=episode.split,
                class_ids=episode.class_ids,
                support_indices=episode.support_indices,
                transforms_signature=episode.transforms_signature,
            )

        # --- Move data to device ---
        x_support = episode.x_support.to(dev).float()
        y_support = episode.y_support.to(dev).long()

        # --- Create probe with head ---
        backbone = self._get_or_create_backbone(dev)
        backbone_snapshot = self._snapshot_params(backbone)

        probe = ProbeWithHead(
            backbone=backbone,
            n_way=episode.n_way,
            freeze_backbone=True,
        ).to(dev)
        probe.train(False)

        # Initialize head with seed
        if seed is not None:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(seed + 1)
            _reinit_linear(probe.head, gen, dev)

        # --- Compute probe loss ---
        probe_loss = _compute_probe_loss(probe, x_support, y_support)

        # --- Get parameters for Fisher ---
        fisher_params = probe.get_fisher_params(self.config.layer_subset)
        param_names = list(fisher_params.keys())

        # Enable gradients on selected backbone params for Fisher
        for name in param_names:
            parts = name.split(".")
            module: Any = probe
            for part in parts[:-1]:
                module = getattr(module, part)
            param = getattr(module, parts[-1])
            param.requires_grad_(True)

        # --- Compute diagonal Fisher ---
        fisher_diag = compute_diagonal_fisher(
            probe,
            x_support,
            y_support,
            param_names=param_names,
            num_samples=self.config.num_fisher_samples,
        )

        # Restore requires_grad state on backbone
        for p in backbone.parameters():
            p.requires_grad_(False)

        # --- Project to embedding ---
        embedding = project_fisher_to_embedding(
            fisher_diag,
            layer_subset=self.config.layer_subset,
            aggregation=self.config.aggregation,
            embedding_dim=self.config.embedding_dim,
            probe=probe,
        )

        # Ensure fp32
        embedding = embedding.float().to("cpu")

        # --- Apply whitening if configured and fitted ---
        if self.config.normalize == "whiten" and self._whitening is not None:
            embedding = self._whitening.transform(embedding)

        # --- Diagnostics ---
        fisher_norm = 0.0
        sparsity = 0.0
        n_params = 0
        for name, fv in fisher_diag.items():
            fisher_norm += fv.float().norm().item() ** 2
            n_params += fv.numel()
            sparsity += (fv == 0).sum().item()
        fisher_norm = math.sqrt(fisher_norm)
        sparsity = sparsity / max(n_params, 1)

        t_end = time.perf_counter()
        extraction_time_ms = (t_end - t_start) * 1000.0

        diagnostics_obj = ExtractionDiagnostics(
            fisher_norm=fisher_norm,
            sparsity=sparsity,
            probe_loss=probe_loss,
            extraction_time_ms=extraction_time_ms,
            n_samples=x_support.shape[0],
            n_params=n_params,
        )

        # --- Verify backbone was not modified ---
        assert self._verify_frozen(backbone, backbone_snapshot), (
            "Backbone parameters were modified during extraction!"
        )

        logger.info(
            "Extracted embedding for task %s: dim=%d, fisher_norm=%.4f, "
            "sparsity=%.3f, probe_loss=%.4f, time=%.1fms",
            task_id[:12],
            embedding.shape[0],
            fisher_norm,
            sparsity,
            probe_loss,
            extraction_time_ms,
        )

        return TaskEmbedding(
            embedding=embedding,
            task_id=task_id,
            diagnostics=diagnostics_obj.to_dict(),
            probe_signature=self._probe_signature,
        )

    def extract_batch(
        self,
        episodes: List[TaskEpisode],
        *,
        seed: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> List[TaskEmbedding]:
        """Extract embeddings for a batch of episodes.

        The probe backbone is shared across all episodes. Each episode
        gets its own ephemeral classification head.

        Args:
            episodes: List of episodes to embed.
            seed: Base random seed.  Episode ``i`` uses seed ``seed + i``
                if seed is not ``None``.
            device: Device for computation.

        Returns:
            List of :class:`TaskEmbedding` results in the same order.
        """
        results: List[TaskEmbedding] = []
        for i, episode in enumerate(episodes):
            ep_seed = (seed + i) if seed is not None else None
            result = self.extract(episode, seed=ep_seed, device=device)
            results.append(result)
        return results

    def set_whitening(self, transform: WhiteningTransform) -> None:
        """Set a whitening transform to apply after embedding projection.

        Args:
            transform: A fitted :class:`WhiteningTransform`.
        """
        if not transform.is_fitted:
            raise ValueError("WhiteningTransform must be fitted before setting.")
        self._whitening = transform
        logger.info("Whitening transform set on extractor.")

    def reset_probe(self) -> None:
        """Discard the cached probe backbone.

        The next call to :meth:`extract` will create a fresh probe.
        """
        self._probe_backbone = None
        logger.info("Probe backbone cache cleared.")


def _reinit_linear(
    linear: nn.Linear,
    generator: torch.Generator,
    device: torch.device,
) -> None:
    """Re-initialize a Linear layer with a seeded generator.

    Args:
        linear: Linear layer to re-initialize.
        generator: Random number generator.
        device: Device for the parameters.
    """
    fan_in = linear.in_features
    std = 1.0 / math.sqrt(fan_in)
    with torch.no_grad():
        # Generate on CPU then move (generator is CPU-only)
        w = torch.empty_like(linear.weight, device="cpu")
        w.uniform_(-std, std, generator=generator)
        linear.weight.copy_(w.to(device))
        if linear.bias is not None:
            b = torch.empty_like(linear.bias, device="cpu")
            b.uniform_(-std, std, generator=generator)
            linear.bias.copy_(b.to(device))


# ============================================================================
# Section 9 — Utility Functions
# ============================================================================


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine distance between two vectors.

    cosine_distance(a, b) = 1 - cos(a, b)

    Args:
        a: First vector ``(D,)``.
        b: Second vector ``(D,)``.

    Returns:
        Cosine distance in [0, 2].
    """
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    cos_sim = F.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).item()
    return 1.0 - cos_sim


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector ``(D,)``.
        b: Second vector ``(D,)``.

    Returns:
        Cosine similarity in [-1, 1].
    """
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return F.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).item()


def pairwise_cosine_distances(embeddings: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine distance matrix.

    Args:
        embeddings: Embedding matrix ``(N, E)``.

    Returns:
        Distance matrix ``(N, N)`` with zeros on the diagonal.
    """
    emb = embeddings.float()
    # Normalize
    norms = emb.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
    normed = emb / norms
    # Cosine similarity matrix
    sim = normed @ normed.T
    # Cosine distance
    dist = 1.0 - sim
    # Ensure diagonal is zero and values are non-negative
    dist = dist.clamp(min=0.0)
    dist.fill_diagonal_(0.0)
    return dist


def embedding_statistics(embeddings: torch.Tensor) -> Dict[str, float]:
    """Compute summary statistics for a set of embeddings.

    Args:
        embeddings: Embedding matrix ``(N, E)``.

    Returns:
        Dictionary with statistics: mean_norm, std_norm, mean_pairwise_dist,
        min_pairwise_dist, max_pairwise_dist.
    """
    emb = embeddings.float()
    norms = emb.norm(p=2, dim=1)
    dist_mat = pairwise_cosine_distances(emb)

    # Extract upper triangle (excluding diagonal)
    n = dist_mat.shape[0]
    if n < 2:
        return {
            "mean_norm": norms.mean().item(),
            "std_norm": norms.std().item() if n > 1 else 0.0,
            "mean_pairwise_dist": 0.0,
            "min_pairwise_dist": 0.0,
            "max_pairwise_dist": 0.0,
        }

    upper_tri = dist_mat[torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)]

    return {
        "mean_norm": norms.mean().item(),
        "std_norm": norms.std().item(),
        "mean_pairwise_dist": upper_tri.mean().item(),
        "min_pairwise_dist": upper_tri.min().item(),
        "max_pairwise_dist": upper_tri.max().item(),
    }


def validate_episode(episode: TaskEpisode) -> List[str]:
    """Validate a TaskEpisode for correctness.

    Checks shapes, label ranges, and consistency between metadata and tensors.

    Args:
        episode: Episode to validate.

    Returns:
        List of warning/error messages (empty if valid).
    """
    issues: List[str] = []

    # Check support shapes
    if episode.x_support.dim() < 2:
        issues.append(
            f"x_support should be at least 2-D, got {episode.x_support.dim()}-D"
        )
    if episode.y_support.dim() != 1:
        issues.append(
            f"y_support should be 1-D, got {episode.y_support.dim()}-D"
        )

    n_support = episode.x_support.shape[0]
    if episode.y_support.shape[0] != n_support:
        issues.append(
            f"x_support ({n_support}) and y_support ({episode.y_support.shape[0]}) "
            f"have different batch sizes"
        )

    # Check expected support size
    expected_support = episode.n_way * episode.k_shot
    if n_support != expected_support:
        issues.append(
            f"Expected {expected_support} support samples (n_way={episode.n_way} * "
            f"k_shot={episode.k_shot}), got {n_support}"
        )

    # Check label range
    unique_labels = episode.y_support.unique()
    if unique_labels.max().item() >= episode.n_way:
        issues.append(
            f"Labels should be in [0, {episode.n_way}), "
            f"but max label is {unique_labels.max().item()}"
        )
    if unique_labels.min().item() < 0:
        issues.append(f"Labels should be non-negative, got min {unique_labels.min().item()}")

    # Check class_ids count
    if episode.class_ids is not None and len(episode.class_ids) != episode.n_way:
        issues.append(
            f"class_ids has {len(episode.class_ids)} entries but n_way={episode.n_way}"
        )

    # Check query set if present
    if episode.x_query is not None:
        if episode.x_query.dim() != episode.x_support.dim():
            issues.append("x_query and x_support have different number of dimensions")
        if episode.y_query is not None:
            if episode.y_query.shape[0] != episode.x_query.shape[0]:
                issues.append("x_query and y_query have different batch sizes")

    return issues


# ============================================================================
# Section 10 — Synthetic Episode Helpers (for testing)
# ============================================================================


def make_synthetic_episode(
    n_way: int = 5,
    k_shot: int = 5,
    q_query: int = 15,
    channels: int = 3,
    height: int = 28,
    width: int = 28,
    dataset: str = "synthetic",
    split: str = "train",
    seed: Optional[int] = None,
) -> TaskEpisode:
    """Create a synthetic few-shot episode for testing.

    Generates random image tensors and sequential class labels.

    Args:
        n_way: Number of classes.
        k_shot: Shots per class in the support set.
        q_query: Queries per class in the query set.
        channels: Number of image channels.
        height: Image height.
        width: Image width.
        dataset: Dataset name for the episode.
        split: Split name.
        seed: Random seed.

    Returns:
        A :class:`TaskEpisode` with random data and auto task ID.
    """
    if seed is not None:
        torch.manual_seed(seed)

    n_support = n_way * k_shot
    n_query = n_way * q_query

    x_support = torch.randn(n_support, channels, height, width)
    y_support = torch.arange(n_way).repeat_interleave(k_shot)

    x_query = torch.randn(n_query, channels, height, width)
    y_query = torch.arange(n_way).repeat_interleave(q_query)

    class_ids = list(range(n_way))

    return TaskEpisode(
        task_id="auto",
        x_support=x_support,
        y_support=y_support,
        x_query=x_query,
        y_query=y_query,
        dataset=dataset,
        split=split,
        n_way=n_way,
        k_shot=k_shot,
        q_query=q_query,
        class_ids=class_ids,
    )


def make_distinct_episodes(
    n_episodes: int = 2,
    n_way: int = 5,
    k_shot: int = 5,
    channels: int = 3,
    height: int = 28,
    width: int = 28,
    base_seed: int = 0,
) -> List[TaskEpisode]:
    """Create multiple distinct synthetic episodes.

    Each episode uses a different seed and different class IDs to ensure
    they represent genuinely different tasks.

    Args:
        n_episodes: Number of episodes to generate.
        n_way: Number of classes per episode.
        k_shot: Shots per class.
        channels: Number of image channels.
        height: Image height.
        width: Image width.
        base_seed: Base seed (episode i uses seed base_seed + i * 1000).

    Returns:
        List of distinct :class:`TaskEpisode` instances.
    """
    episodes: List[TaskEpisode] = []
    for i in range(n_episodes):
        ep_seed = base_seed + i * 1000
        class_ids = list(range(i * n_way, (i + 1) * n_way))

        torch.manual_seed(ep_seed)
        n_support = n_way * k_shot
        n_query = n_way * 15

        x_support = torch.randn(n_support, channels, height, width)
        y_support = torch.arange(n_way).repeat_interleave(k_shot)
        x_query = torch.randn(n_query, channels, height, width)
        y_query = torch.arange(n_way).repeat_interleave(15)

        ep = TaskEpisode(
            task_id="auto",
            x_support=x_support,
            y_support=y_support,
            x_query=x_query,
            y_query=y_query,
            dataset="synthetic",
            split="train",
            n_way=n_way,
            k_shot=k_shot,
            q_query=15,
            class_ids=class_ids,
        )
        episodes.append(ep)
    return episodes


# ============================================================================
# Section 11 — Self-Test Block
# ============================================================================


def _run_self_tests() -> None:
    """Run comprehensive self-tests for the Task2Vec extractor module.

    All tests use synthetic data (random tensors) and print PASS/FAIL per test.
    Target: 20+ test cases covering probes, canonicalization, Fisher computation,
    embedding projection, determinism, whitening, and batch extraction.
    """
    results: List[Tuple[str, bool, str]] = []

    def record(name: str, passed: bool, detail: str = "") -> None:
        results.append((name, passed, detail))
        status = "PASS" if passed else "FAIL"
        msg = f"  [{status}] {name}"
        if detail:
            msg += f"  -- {detail}"
        print(msg)

    print("=" * 72)
    print("Task2Vec Extractor Self-Tests")
    print("=" * 72)

    device = torch.device("cpu")
    in_channels = 1
    img_h, img_w = 28, 28

    # ----------------------------------------------------------------
    # Test 1: Conv4Probe forward pass shape
    # ----------------------------------------------------------------
    try:
        probe_c4 = Conv4Probe(in_channels=in_channels)
        x_test = torch.randn(4, in_channels, img_h, img_w)
        out = probe_c4(x_test)
        expected_feat_dim = probe_c4.feature_dim
        ok = out.shape == (4, expected_feat_dim)
        record(
            "Conv4Probe forward shape",
            ok,
            f"output={out.shape}, expected=(4, {expected_feat_dim})",
        )
    except Exception as e:
        record("Conv4Probe forward shape", False, str(e))

    # ----------------------------------------------------------------
    # Test 2: ResNet12Probe forward pass shape
    # ----------------------------------------------------------------
    try:
        probe_r12 = ResNet12Probe(in_channels=in_channels)
        x_test = torch.randn(4, in_channels, img_h, img_w)
        out = probe_r12(x_test)
        expected_feat_dim = probe_r12.feature_dim
        ok = out.shape == (4, expected_feat_dim)
        record(
            "ResNet12Probe forward shape",
            ok,
            f"output={out.shape}, expected=(4, {expected_feat_dim})",
        )
    except Exception as e:
        record("ResNet12Probe forward shape", False, str(e))

    # ----------------------------------------------------------------
    # Test 3: ProbeWithHead creates correct head size
    # ----------------------------------------------------------------
    try:
        backbone = Conv4Probe(in_channels=in_channels)
        n_way = 7
        pwh = ProbeWithHead(backbone, n_way=n_way)
        head_out = pwh.head.out_features
        ok = head_out == n_way
        record(
            "ProbeWithHead head size for N-way",
            ok,
            f"head.out_features={head_out}, n_way={n_way}",
        )
    except Exception as e:
        record("ProbeWithHead head size for N-way", False, str(e))

    # ----------------------------------------------------------------
    # Test 4: ProbeWithHead forward pass shape
    # ----------------------------------------------------------------
    try:
        backbone = Conv4Probe(in_channels=in_channels)
        n_way = 5
        pwh = ProbeWithHead(backbone, n_way=n_way)
        x_test = torch.randn(8, in_channels, img_h, img_w)
        logits = pwh(x_test)
        ok = logits.shape == (8, n_way)
        record(
            "ProbeWithHead forward shape",
            ok,
            f"output={logits.shape}, expected=(8, {n_way})",
        )
    except Exception as e:
        record("ProbeWithHead forward shape", False, str(e))

    # ----------------------------------------------------------------
    # Test 5: canonicalize_task_id is deterministic
    # ----------------------------------------------------------------
    try:
        cids = [5, 3, 1, 7, 9]
        sid = [100, 50, 200, 10, 150]
        tid1 = canonicalize_task_id("miniImageNet", "train", cids, sid, "resize_84")
        tid2 = canonicalize_task_id("miniImageNet", "train", cids, sid, "resize_84")
        ok = tid1 == tid2
        record(
            "canonicalize_task_id deterministic",
            ok,
            f"id1={tid1[:16]}, id2={tid2[:16]}",
        )
    except Exception as e:
        record("canonicalize_task_id deterministic", False, str(e))

    # ----------------------------------------------------------------
    # Test 6: canonicalize_task_id differs for different inputs
    # ----------------------------------------------------------------
    try:
        cids_a = [0, 1, 2, 3, 4]
        cids_b = [5, 6, 7, 8, 9]
        tid_a = canonicalize_task_id("ds", "train", cids_a)
        tid_b = canonicalize_task_id("ds", "train", cids_b)
        ok = tid_a != tid_b
        record(
            "canonicalize_task_id differs for diff inputs",
            ok,
            f"a={tid_a[:12]}, b={tid_b[:12]}",
        )
    except Exception as e:
        record("canonicalize_task_id differs for diff inputs", False, str(e))

    # ----------------------------------------------------------------
    # Test 7: canonicalize_task_id invariant to class_id order
    # ----------------------------------------------------------------
    try:
        tid_sorted = canonicalize_task_id("ds", "train", [1, 2, 3])
        tid_unsorted = canonicalize_task_id("ds", "train", [3, 1, 2])
        ok = tid_sorted == tid_unsorted
        record(
            "canonicalize_task_id order invariant",
            ok,
            f"sorted={tid_sorted[:12]}, unsorted={tid_unsorted[:12]}",
        )
    except Exception as e:
        record("canonicalize_task_id order invariant", False, str(e))

    # ----------------------------------------------------------------
    # Test 8: compute_diagonal_fisher returns non-negative values
    # ----------------------------------------------------------------
    try:
        backbone = Conv4Probe(in_channels=in_channels)
        n_way = 3
        k_shot = 2
        probe = ProbeWithHead(backbone, n_way=n_way, freeze_backbone=False)
        probe.train(False)

        n_support = n_way * k_shot
        x_s = torch.randn(n_support, in_channels, img_h, img_w)
        y_s = torch.arange(n_way).repeat_interleave(k_shot)

        fisher = compute_diagonal_fisher(probe, x_s, y_s)
        all_non_neg = all(
            (v >= 0).all().item() for v in fisher.values()
        )
        record(
            "Fisher non-negative values",
            all_non_neg,
            f"n_params={len(fisher)}",
        )
    except Exception as e:
        record("Fisher non-negative values", False, str(e))

    # ----------------------------------------------------------------
    # Test 9: compute_diagonal_fisher is deterministic
    # ----------------------------------------------------------------
    try:
        torch.manual_seed(42)
        backbone1 = Conv4Probe(in_channels=in_channels)
        probe1 = ProbeWithHead(backbone1, n_way=3, freeze_backbone=False)
        probe1.train(False)
        x_s = torch.randn(6, in_channels, img_h, img_w)
        y_s = torch.tensor([0, 0, 1, 1, 2, 2])

        torch.manual_seed(42)
        backbone2 = Conv4Probe(in_channels=in_channels)
        probe2 = ProbeWithHead(backbone2, n_way=3, freeze_backbone=False)
        probe2.train(False)

        fisher1 = compute_diagonal_fisher(probe1, x_s, y_s)
        fisher2 = compute_diagonal_fisher(probe2, x_s, y_s)

        max_diff = 0.0
        for key in fisher1:
            if key in fisher2:
                diff = (fisher1[key] - fisher2[key]).abs().max().item()
                max_diff = max(max_diff, diff)

        ok = max_diff < 1e-7
        record(
            "Fisher deterministic (same model+data)",
            ok,
            f"max_diff={max_diff:.2e}",
        )
    except Exception as e:
        record("Fisher deterministic (same model+data)", False, str(e))

    # ----------------------------------------------------------------
    # Test 10: Fisher normalized by sample count
    # ----------------------------------------------------------------
    try:
        backbone = Conv4Probe(in_channels=in_channels)
        probe = ProbeWithHead(backbone, n_way=2, freeze_backbone=False)
        probe.train(False)
        x_s = torch.randn(4, in_channels, img_h, img_w)
        y_s = torch.tensor([0, 0, 1, 1])

        fisher = compute_diagonal_fisher(probe, x_s, y_s)
        # Fisher values should be finite and reasonable in magnitude
        all_finite = all(torch.isfinite(v).all().item() for v in fisher.values())
        record(
            "Fisher normalized by sample count (finite values)",
            all_finite,
            f"n_params={len(fisher)}",
        )
    except Exception as e:
        record("Fisher normalized by sample count (finite values)", False, str(e))

    # ----------------------------------------------------------------
    # Test 11: project_fisher_to_embedding produces correct E-dim
    # ----------------------------------------------------------------
    try:
        backbone = Conv4Probe(in_channels=in_channels)
        probe = ProbeWithHead(backbone, n_way=3, freeze_backbone=False)
        probe.train(False)
        x_s = torch.randn(6, in_channels, img_h, img_w)
        y_s = torch.tensor([0, 0, 1, 1, 2, 2])

        fisher = compute_diagonal_fisher(probe, x_s, y_s)
        emb_dim = 256
        embedding = project_fisher_to_embedding(
            fisher,
            layer_subset="all",
            aggregation="per_channel",
            embedding_dim=emb_dim,
            probe=probe,
        )
        ok = embedding.shape == (emb_dim,)
        record(
            "Embedding has correct E-dim",
            ok,
            f"shape={embedding.shape}, expected=({emb_dim},)",
        )
    except Exception as e:
        record("Embedding has correct E-dim", False, str(e))

    # ----------------------------------------------------------------
    # Test 12: Embedding is L2-normalized (norm approximately 1.0)
    # ----------------------------------------------------------------
    try:
        backbone = Conv4Probe(in_channels=in_channels)
        probe = ProbeWithHead(backbone, n_way=3, freeze_backbone=False)
        probe.train(False)
        x_s = torch.randn(6, in_channels, img_h, img_w)
        y_s = torch.tensor([0, 0, 1, 1, 2, 2])

        fisher = compute_diagonal_fisher(probe, x_s, y_s)
        embedding = project_fisher_to_embedding(
            fisher,
            layer_subset="all",
            aggregation="per_channel",
            embedding_dim=512,
            probe=probe,
        )
        norm_val = embedding.norm(p=2).item()
        ok = abs(norm_val - 1.0) < 1e-5
        record(
            "Embedding L2-normalized (norm ~ 1.0)",
            ok,
            f"norm={norm_val:.6f}",
        )
    except Exception as e:
        record("Embedding L2-normalized (norm ~ 1.0)", False, str(e))

    # ----------------------------------------------------------------
    # Test 13: log1p applied (all embedding values >= 0)
    # ----------------------------------------------------------------
    try:
        backbone = Conv4Probe(in_channels=in_channels)
        probe = ProbeWithHead(backbone, n_way=3, freeze_backbone=False)
        probe.train(False)
        x_s = torch.randn(6, in_channels, img_h, img_w)
        y_s = torch.tensor([0, 0, 1, 1, 2, 2])

        fisher = compute_diagonal_fisher(probe, x_s, y_s)
        embedding = project_fisher_to_embedding(
            fisher,
            layer_subset="all",
            aggregation="per_channel",
            embedding_dim=512,
            probe=probe,
        )
        # After log1p of non-negative Fisher + L2 norm, values should be >= 0
        all_non_neg = (embedding >= -1e-7).all().item()
        record(
            "log1p applied (all values >= 0 after normalization)",
            all_non_neg,
            f"min={embedding.min().item():.6e}",
        )
    except Exception as e:
        record("log1p applied (all values >= 0 after normalization)", False, str(e))

    # ----------------------------------------------------------------
    # Test 14: Layer subset "last_block" selects fewer params than "all"
    # ----------------------------------------------------------------
    try:
        backbone = Conv4Probe(in_channels=in_channels)
        last_block_params = backbone.get_layer_subset("last_block")
        all_params = backbone.get_layer_subset("all")
        # Count actual matching parameters
        n_last = sum(
            1
            for n, _ in backbone.named_parameters()
            if any(n.startswith(p) for p in last_block_params)
        )
        n_all = sum(
            1
            for n, _ in backbone.named_parameters()
            if any(n.startswith(p) for p in all_params)
        )
        ok = n_last <= n_all
        record(
            "Layer subset last_block <= all params",
            ok,
            f"last_block={n_last}, all={n_all}",
        )
    except Exception as e:
        record("Layer subset last_block <= all params", False, str(e))

    # ----------------------------------------------------------------
    # Test 15: Task2VecExtractor end-to-end produces TaskEmbedding
    # ----------------------------------------------------------------
    try:
        config = Task2VecConfig(
            probe_model="conv4",
            embedding_dim=128,
            layer_subset="all",
            aggregation="per_channel",
            input_channels=in_channels,
        )
        extractor = Task2VecExtractor(config)
        episode = make_synthetic_episode(
            n_way=3,
            k_shot=2,
            channels=in_channels,
            height=img_h,
            width=img_w,
            seed=42,
        )
        result = extractor.extract(episode, seed=42, device="cpu")

        checks = [
            isinstance(result, TaskEmbedding),
            result.embedding.shape == (128,),
            isinstance(result.task_id, str) and len(result.task_id) == 32,
            isinstance(result.diagnostics, dict),
            isinstance(result.probe_signature, str),
        ]
        ok = all(checks)
        record(
            "Task2VecExtractor end-to-end",
            ok,
            f"emb_shape={result.embedding.shape}, task_id_len={len(result.task_id)}",
        )
    except Exception as e:
        record("Task2VecExtractor end-to-end", False, str(e))

    # ----------------------------------------------------------------
    # Test 16: Determinism: same episode + seed -> same embedding
    # ----------------------------------------------------------------
    try:
        config = Task2VecConfig(
            probe_model="conv4",
            embedding_dim=128,
            input_channels=in_channels,
        )
        extractor1 = Task2VecExtractor(config)
        extractor2 = Task2VecExtractor(config)

        ep = make_synthetic_episode(
            n_way=3,
            k_shot=2,
            channels=in_channels,
            height=img_h,
            width=img_w,
            seed=99,
        )
        r1 = extractor1.extract(ep, seed=42, device="cpu")
        r2 = extractor2.extract(ep, seed=42, device="cpu")

        diff = (r1.embedding - r2.embedding).abs().max().item()
        ok = diff < 1e-6
        record(
            "Determinism: same episode+seed -> same embedding",
            ok,
            f"max_diff={diff:.2e}",
        )
    except Exception as e:
        record("Determinism: same episode+seed -> same embedding", False, str(e))

    # ----------------------------------------------------------------
    # Test 17: Different episodes -> different embeddings
    # ----------------------------------------------------------------
    try:
        config = Task2VecConfig(
            probe_model="conv4",
            embedding_dim=128,
            input_channels=in_channels,
        )
        extractor = Task2VecExtractor(config)

        episodes = make_distinct_episodes(
            n_episodes=2,
            n_way=3,
            k_shot=2,
            channels=in_channels,
            height=img_h,
            width=img_w,
        )
        r1 = extractor.extract(episodes[0], seed=42, device="cpu")
        r2 = extractor.extract(episodes[1], seed=42, device="cpu")

        cos_dist = cosine_distance(r1.embedding, r2.embedding)
        ok = cos_dist > 0.0
        record(
            "Different episodes -> different embeddings (cos_dist > 0)",
            ok,
            f"cosine_distance={cos_dist:.6f}",
        )
    except Exception as e:
        record(
            "Different episodes -> different embeddings (cos_dist > 0)",
            False,
            str(e),
        )

    # ----------------------------------------------------------------
    # Test 18: Probe stays frozen after extraction
    # ----------------------------------------------------------------
    try:
        config = Task2VecConfig(
            probe_model="conv4",
            embedding_dim=128,
            input_channels=in_channels,
        )
        extractor = Task2VecExtractor(config)

        # Force backbone creation
        ep = make_synthetic_episode(
            n_way=3,
            k_shot=2,
            channels=in_channels,
            height=img_h,
            width=img_w,
            seed=10,
        )
        _ = extractor.extract(ep, seed=10, device="cpu")

        # Snapshot backbone params
        backbone = extractor._probe_backbone
        assert backbone is not None
        snap_before = {n: p.data.clone() for n, p in backbone.named_parameters()}

        # Extract again with different episode
        ep2 = make_synthetic_episode(
            n_way=5,
            k_shot=3,
            channels=in_channels,
            height=img_h,
            width=img_w,
            seed=20,
        )
        _ = extractor.extract(ep2, seed=20, device="cpu")

        # Check params unchanged
        all_same = all(
            torch.equal(p.data, snap_before[n])
            for n, p in backbone.named_parameters()
        )
        ok = all_same
        record("Probe stays frozen after extraction", ok)
    except Exception as e:
        record("Probe stays frozen after extraction", False, str(e))

    # ----------------------------------------------------------------
    # Test 19: WhiteningTransform fit + transform produces zero-mean
    # ----------------------------------------------------------------
    try:
        n_ref, e_dim = 50, 64
        torch.manual_seed(123)
        ref_embeddings = torch.randn(n_ref, e_dim)
        # L2 normalize
        ref_embeddings = ref_embeddings / ref_embeddings.norm(p=2, dim=1, keepdim=True)

        wt = WhiteningTransform(reg=1e-4)
        whitened = wt.fit_transform(ref_embeddings)

        # Check approximately zero mean
        mean_abs = whitened.mean(dim=0).abs().mean().item()
        ok = mean_abs < 0.15  # Approximate zero-mean after whitening + renormalization
        record(
            "WhiteningTransform approximately zero-mean",
            ok,
            f"mean(|mean|)={mean_abs:.4f}",
        )
    except Exception as e:
        record("WhiteningTransform approximately zero-mean", False, str(e))

    # ----------------------------------------------------------------
    # Test 20: WhiteningTransform single embedding transform
    # ----------------------------------------------------------------
    try:
        n_ref, e_dim = 30, 32
        torch.manual_seed(456)
        ref_emb = torch.randn(n_ref, e_dim)
        ref_emb = ref_emb / ref_emb.norm(p=2, dim=1, keepdim=True)

        wt = WhiteningTransform(reg=1e-4)
        wt.fit(ref_emb)

        single = torch.randn(e_dim)
        single = single / single.norm(p=2)
        transformed = wt.transform(single)

        ok = transformed.shape == (e_dim,)
        norm_val = transformed.norm(p=2).item()
        ok = ok and abs(norm_val - 1.0) < 1e-5
        record(
            "WhiteningTransform single embedding output shape + normalized",
            ok,
            f"shape={transformed.shape}, norm={norm_val:.6f}",
        )
    except Exception as e:
        record(
            "WhiteningTransform single embedding output shape + normalized",
            False,
            str(e),
        )

    # ----------------------------------------------------------------
    # Test 21: fp32 enforcement: embedding dtype is float32
    # ----------------------------------------------------------------
    try:
        config = Task2VecConfig(
            probe_model="conv4",
            embedding_dim=64,
            input_channels=in_channels,
        )
        extractor = Task2VecExtractor(config)
        ep = make_synthetic_episode(
            n_way=2,
            k_shot=2,
            channels=in_channels,
            height=img_h,
            width=img_w,
            seed=77,
        )
        result = extractor.extract(ep, seed=77, device="cpu")
        ok = result.embedding.dtype == torch.float32
        record(
            "fp32 enforcement: embedding dtype is float32",
            ok,
            f"dtype={result.embedding.dtype}",
        )
    except Exception as e:
        record("fp32 enforcement: embedding dtype is float32", False, str(e))

    # ----------------------------------------------------------------
    # Test 22: Batch extraction produces list of correct length
    # ----------------------------------------------------------------
    try:
        config = Task2VecConfig(
            probe_model="conv4",
            embedding_dim=64,
            input_channels=in_channels,
        )
        extractor = Task2VecExtractor(config)
        episodes = make_distinct_episodes(
            n_episodes=4,
            n_way=3,
            k_shot=2,
            channels=in_channels,
            height=img_h,
            width=img_w,
        )
        batch_results = extractor.extract_batch(episodes, seed=42, device="cpu")
        ok = len(batch_results) == 4
        all_correct_shape = all(r.embedding.shape == (64,) for r in batch_results)
        ok = ok and all_correct_shape
        record(
            "Batch extraction: correct length and shapes",
            ok,
            f"n_results={len(batch_results)}",
        )
    except Exception as e:
        record("Batch extraction: correct length and shapes", False, str(e))

    # ----------------------------------------------------------------
    # Test 23: Diagnostics include required fields
    # ----------------------------------------------------------------
    try:
        config = Task2VecConfig(
            probe_model="conv4",
            embedding_dim=64,
            input_channels=in_channels,
        )
        extractor = Task2VecExtractor(config)
        ep = make_synthetic_episode(
            n_way=3,
            k_shot=2,
            channels=in_channels,
            height=img_h,
            width=img_w,
            seed=55,
        )
        result = extractor.extract(ep, seed=55, device="cpu")
        required_keys = {
            "fisher_norm",
            "sparsity",
            "probe_loss",
            "extraction_time_ms",
        }
        present = required_keys.issubset(set(result.diagnostics.keys()))
        fisher_norm_ok = result.diagnostics["fisher_norm"] >= 0
        sparsity_ok = 0.0 <= result.diagnostics["sparsity"] <= 1.0
        time_ok = result.diagnostics["extraction_time_ms"] > 0
        ok = present and fisher_norm_ok and sparsity_ok and time_ok
        record(
            "Diagnostics include fisher_norm, sparsity, probe_loss, time",
            ok,
            f"keys={sorted(result.diagnostics.keys())}",
        )
    except Exception as e:
        record(
            "Diagnostics include fisher_norm, sparsity, probe_loss, time",
            False,
            str(e),
        )

    # ----------------------------------------------------------------
    # Test 24: create_probe factory function
    # ----------------------------------------------------------------
    try:
        p1 = create_probe("conv4", in_channels=1, seed=0)
        p2 = create_probe("resnet12", in_channels=1, seed=0)
        ok = isinstance(p1, Conv4Probe) and isinstance(p2, ResNet12Probe)
        record("create_probe factory", ok)
    except Exception as e:
        record("create_probe factory", False, str(e))

    # ----------------------------------------------------------------
    # Test 25: create_probe unknown name raises ValueError
    # ----------------------------------------------------------------
    try:
        raised = False
        try:
            create_probe("unknown_probe")
        except ValueError:
            raised = True
        record("create_probe unknown name raises ValueError", raised)
    except Exception as e:
        record("create_probe unknown name raises ValueError", False, str(e))

    # ----------------------------------------------------------------
    # Test 26: validate_episode detects issues
    # ----------------------------------------------------------------
    try:
        # Valid episode
        ep_valid = make_synthetic_episode(
            n_way=3, k_shot=2, channels=in_channels
        )
        issues_valid = validate_episode(ep_valid)
        ok_valid = len(issues_valid) == 0

        # Invalid episode: wrong label range
        ep_bad = TaskEpisode(
            task_id="test",
            x_support=torch.randn(6, in_channels, img_h, img_w),
            y_support=torch.tensor([0, 0, 1, 1, 5, 5]),
            n_way=3,
            k_shot=2,
        )
        issues_bad = validate_episode(ep_bad)
        ok_bad = len(issues_bad) > 0

        ok = ok_valid and ok_bad
        record(
            "validate_episode detects issues",
            ok,
            f"valid_issues={len(issues_valid)}, bad_issues={len(issues_bad)}",
        )
    except Exception as e:
        record("validate_episode detects issues", False, str(e))

    # ----------------------------------------------------------------
    # Test 27: Per-layer aggregation produces different result than per-channel
    # ----------------------------------------------------------------
    try:
        backbone = Conv4Probe(in_channels=in_channels)
        probe = ProbeWithHead(backbone, n_way=3, freeze_backbone=False)
        probe.train(False)
        x_s = torch.randn(6, in_channels, img_h, img_w)
        y_s = torch.tensor([0, 0, 1, 1, 2, 2])
        fisher = compute_diagonal_fisher(probe, x_s, y_s)

        emb_per_ch = project_fisher_to_embedding(
            fisher,
            layer_subset="all",
            aggregation="per_channel",
            embedding_dim=512,
            probe=probe,
        )
        emb_per_layer = project_fisher_to_embedding(
            fisher,
            layer_subset="all",
            aggregation="per_layer",
            embedding_dim=512,
            probe=probe,
        )

        # Both should have same final dim but different internal structure
        ok = emb_per_ch.shape == (512,) and emb_per_layer.shape == (512,)
        # They should produce different embeddings (different aggregation)
        diff = (emb_per_ch - emb_per_layer).abs().max().item()
        ok = ok and diff > 1e-6
        record(
            "per_channel vs per_layer aggregation differ",
            ok,
            f"max_diff={diff:.6f}",
        )
    except Exception as e:
        record("per_channel vs per_layer aggregation differ", False, str(e))

    # ----------------------------------------------------------------
    # Test 28: Task2VecConfig signature is deterministic
    # ----------------------------------------------------------------
    try:
        cfg1 = Task2VecConfig(probe_model="conv4", embedding_dim=512)
        cfg2 = Task2VecConfig(probe_model="conv4", embedding_dim=512)
        cfg3 = Task2VecConfig(probe_model="resnet12", embedding_dim=512)
        ok = (
            cfg1.signature() == cfg2.signature()
            and cfg1.signature() != cfg3.signature()
        )
        record(
            "Task2VecConfig signature deterministic + discriminative",
            ok,
            f"conv4={cfg1.signature()[:12]}, resnet12={cfg3.signature()[:12]}",
        )
    except Exception as e:
        record(
            "Task2VecConfig signature deterministic + discriminative",
            False,
            str(e),
        )

    # ----------------------------------------------------------------
    # Test 29: Cosine distance / similarity utilities
    # ----------------------------------------------------------------
    try:
        a = torch.tensor([1.0, 0.0, 0.0])
        b = torch.tensor([0.0, 1.0, 0.0])
        c = torch.tensor([1.0, 0.0, 0.0])

        dist_ab = cosine_distance(a, b)
        dist_ac = cosine_distance(a, c)
        sim_ac = cosine_similarity(a, c)

        ok = (
            abs(dist_ab - 1.0) < 1e-5
            and abs(dist_ac) < 1e-5
            and abs(sim_ac - 1.0) < 1e-5
        )
        record(
            "Cosine distance/similarity utilities",
            ok,
            f"dist(ortho)={dist_ab:.4f}, dist(same)={dist_ac:.4f}, "
            f"sim(same)={sim_ac:.4f}",
        )
    except Exception as e:
        record("Cosine distance/similarity utilities", False, str(e))

    # ----------------------------------------------------------------
    # Test 30: Pairwise cosine distance matrix
    # ----------------------------------------------------------------
    try:
        embs = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
        dist_mat = pairwise_cosine_distances(embs)
        ok = dist_mat.shape == (3, 3)
        ok = ok and abs(dist_mat[0, 0].item()) < 1e-5  # self-distance = 0
        ok = ok and abs(dist_mat[0, 2].item()) < 1e-5  # same vector = 0
        ok = ok and abs(dist_mat[0, 1].item() - 1.0) < 1e-5  # orthogonal = 1
        record(
            "Pairwise cosine distance matrix",
            ok,
            f"shape={dist_mat.shape}, d(0,1)={dist_mat[0, 1]:.4f}, "
            f"d(0,2)={dist_mat[0, 2]:.4f}",
        )
    except Exception as e:
        record("Pairwise cosine distance matrix", False, str(e))

    # ----------------------------------------------------------------
    # Test 31: ExtractionDiagnostics to_dict
    # ----------------------------------------------------------------
    try:
        diag = ExtractionDiagnostics(
            fisher_norm=1.23,
            sparsity=0.5,
            probe_loss=0.9,
            extraction_time_ms=42.0,
            n_samples=10,
            n_params=1000,
        )
        d = diag.to_dict()
        ok = (
            d["fisher_norm"] == 1.23
            and d["sparsity"] == 0.5
            and d["probe_loss"] == 0.9
            and d["extraction_time_ms"] == 42.0
            and d["n_samples"] == 10.0
            and d["n_params"] == 1000.0
        )
        record("ExtractionDiagnostics to_dict", ok)
    except Exception as e:
        record("ExtractionDiagnostics to_dict", False, str(e))

    # ----------------------------------------------------------------
    # Test 32: embedding_statistics utility
    # ----------------------------------------------------------------
    try:
        torch.manual_seed(789)
        embs = torch.randn(10, 32)
        embs = embs / embs.norm(p=2, dim=1, keepdim=True)
        stats = embedding_statistics(embs)
        required = {
            "mean_norm",
            "std_norm",
            "mean_pairwise_dist",
            "min_pairwise_dist",
            "max_pairwise_dist",
        }
        ok = required.issubset(set(stats.keys()))
        ok = ok and abs(stats["mean_norm"] - 1.0) < 0.1  # all L2 normalized
        ok = ok and stats["min_pairwise_dist"] >= 0.0
        record(
            "embedding_statistics utility",
            ok,
            f"mean_norm={stats['mean_norm']:.4f}, "
            f"mean_dist={stats['mean_pairwise_dist']:.4f}",
        )
    except Exception as e:
        record("embedding_statistics utility", False, str(e))

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print()
    print("=" * 72)
    n_pass = sum(1 for _, passed, _ in results if passed)
    n_fail = sum(1 for _, passed, _ in results if not passed)
    n_total = len(results)
    print(f"Results: {n_pass}/{n_total} passed, {n_fail}/{n_total} failed")

    if n_fail > 0:
        print("\nFailed tests:")
        for name, passed, detail in results:
            if not passed:
                print(f"  - {name}: {detail}")
    else:
        print("\nAll tests passed.")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _run_self_tests()
