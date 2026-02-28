# Copyright (c) Meta Platforms, Inc. and affiliates.
# MIT License
#
# frozen_assessor_template.py
#
# FrozenBackboneAssessor: manages a frozen encoder with N trainable probe heads.
# Supports multi-head hyperparameter search by training all heads in a single pass.

from __future__ import annotations

import copy
import unittest
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from attentive_classifier_template import AttentiveClassifier


# ---------------------------------------------------------------------------
# AssessmentConfig (minimal, imported from assessment_config_template in prod)
# ---------------------------------------------------------------------------

class _MinimalConfig:
    """Minimal config used when AssessmentConfig is not available."""
    task: str = "video_classification"
    num_classes: int = 10
    num_queries: int = 1
    probe_depth: int = 1
    num_segments: int = 1
    num_views: int = 1
    val_only: bool = False


# ---------------------------------------------------------------------------
# FrozenBackboneAssessor
# ---------------------------------------------------------------------------

class FrozenBackboneAssessor:
    """
    Multi-head frozen backbone assessment pipeline.

    Holds a frozen encoder and manages N trainable AttentiveClassifier heads,
    one per (lr, wd) combination in ``multihead_kwargs``. Trains all heads
    simultaneously in a single forward pass over each batch.

    Public contract::

        assessor = FrozenBackboneAssessor(encoder, config)
        assessor.train_probes(train_loader, multihead_kwargs)
        metrics = assessor.run_validation(val_loader)
        idx, best_metrics = assessor.best_head()

    Args:
        encoder: Pretrained encoder. Its parameters will be frozen immediately.
        config:  AssessmentConfig (or any object with the required attributes).
        device:  Target device (defaults to CPU).
    """

    def __init__(
        self,
        encoder: nn.Module,
        config,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config
        self.device = device or torch.device("cpu")

        # Freeze encoder — must happen before any training
        self.encoder = encoder.to(self.device)
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.train(False)   # inference mode: disables dropout, batchnorm updates

        # Probe heads are created when train_probes() is called
        self._probes: List[nn.Module]              = []
        self._optimizers: List[torch.optim.Optimizer] = []
        self._head_metrics: List[Dict[str, float]] = []
        self._multihead_kwargs: List[Dict]         = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_probes(
        self,
        train_loader: DataLoader,
        multihead_kwargs: List[Dict],
        num_epochs: int = 1,
    ) -> None:
        """
        Train N probe heads simultaneously.

        Args:
            train_loader:     DataLoader yielding (features, labels) pairs where
                              features have shape [B, N, D] and labels are [B].
            multihead_kwargs: List of dicts, each with keys "lr" and optionally
                              "wd" (weight decay). One probe per dict.
            num_epochs:       Number of training epochs.
        """
        if self.config.val_only:
            return

        self._multihead_kwargs = multihead_kwargs
        self._build_probes(multihead_kwargs)

        for epoch in range(num_epochs):
            self._run_train_epoch(train_loader)

    def run_validation(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Run validation for all probes and return per-head metrics.

        Returns:
            Dict mapping "head_{i}" -> {"accuracy": float, "loss": float}
        """
        if not self._probes:
            raise RuntimeError("Call train_probes() before run_validation().")

        all_metrics: Dict[str, float] = {}
        for i, probe in enumerate(self._probes):
            metrics = self._validate_single_probe(probe, val_loader)
            all_metrics[f"head_{i}"] = metrics
            self._head_metrics.append(metrics)

        return all_metrics

    def best_head(self) -> Tuple[int, Dict[str, float]]:
        """
        Select the probe head with the highest validation accuracy.

        Returns:
            (index, metrics_dict) for the best head.
        """
        if not self._head_metrics:
            # No validation run: default to first head
            return 0, {}

        best_idx = max(
            range(len(self._head_metrics)),
            key=lambda i: self._head_metrics[i].get("accuracy", 0.0),
        )
        return best_idx, self._head_metrics[best_idx]

    def get_probe(self, idx: int) -> nn.Module:
        """Return the probe head at the given index."""
        return self._probes[idx]

    def save_best_probe(self, path: str) -> None:
        """Save the best probe's state dict to ``path``."""
        idx, _ = self.best_head()
        torch.save(self._probes[idx].state_dict(), path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_probes(self, multihead_kwargs: List[Dict]) -> None:
        """Create one AttentiveClassifier per hyperparameter combination."""
        embed_dim   = self._infer_embed_dim()
        num_classes = getattr(self.config, "num_classes", 10)
        num_queries = getattr(self.config, "num_queries", 1)
        probe_depth = getattr(self.config, "probe_depth", 1)

        self._probes     = []
        self._optimizers = []
        self._head_metrics = []

        for kw in multihead_kwargs:
            probe = AttentiveClassifier(
                embed_dim=embed_dim,
                num_classes=num_classes,
                num_queries=num_queries,
                depth=probe_depth,
            ).to(self.device)
            self._probes.append(probe)

            lr  = kw.get("lr", 1e-3)
            wd  = kw.get("wd", kw.get("weight_decay", 1e-4))
            mom = kw.get("momentum", 0.9)
            opt = torch.optim.SGD(probe.parameters(), lr=lr, weight_decay=wd, momentum=mom)
            self._optimizers.append(opt)

    def _infer_embed_dim(self) -> int:
        """
        Determine encoder output dimension by running a dummy forward pass.
        Falls back to config.embed_dim if the encoder has that attribute.
        """
        if hasattr(self.encoder, "embed_dim"):
            return self.encoder.embed_dim

        # Try a tiny synthetic forward pass to get the output shape
        try:
            dummy = torch.zeros(1, 4, 16, device=self.device)   # [B, N, D] guess
            with torch.no_grad():
                out = self.encoder(dummy)
            if out.dim() == 3:
                return out.shape[-1]
        except Exception:
            pass

        # Last resort
        return getattr(self.config, "embed_dim", 128)

    def _run_train_epoch(self, train_loader: DataLoader) -> None:
        """One epoch: freeze encoder, train all probes simultaneously."""
        self.encoder.train(False)   # keep frozen throughout
        for probe in self._probes:
            probe.train()

        for batch in train_loader:
            frames, labels = self._unpack_batch(batch)
            frames = frames.to(self.device)
            labels = labels.to(self.device)

            # Single encoder pass — shared across all probe heads
            with torch.no_grad():
                features = self._encode(frames)   # [B, N, D]

            # Independent probe updates
            for probe, opt in zip(self._probes, self._optimizers):
                logits = probe(features)
                loss   = F.cross_entropy(logits, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()

    def _validate_single_probe(
        self, probe: nn.Module, val_loader: DataLoader
    ) -> Dict[str, float]:
        """Compute accuracy and average loss for a single probe."""
        probe.train(False)

        total_loss     = 0.0
        total_correct  = 0
        total_samples  = 0

        with torch.no_grad():
            for batch in val_loader:
                frames, labels = self._unpack_batch(batch)
                frames = frames.to(self.device)
                labels = labels.to(self.device)

                features = self._encode(frames)
                logits   = probe(features)
                loss     = F.cross_entropy(logits, labels, reduction="sum")

                preds          = logits.argmax(dim=-1)
                total_correct  += (preds == labels).sum().item()
                total_loss     += loss.item()
                total_samples  += labels.shape[0]

        accuracy = total_correct / max(total_samples, 1)
        avg_loss = total_loss   / max(total_samples, 1)

        probe.train()   # restore training mode for next epoch
        return {"accuracy": accuracy, "loss": avg_loss, "samples": total_samples}

    def _encode(self, frames: Tensor) -> Tensor:
        """
        Run encoder in non-training, no-grad mode.
        Handles both 3D [B, N, D] and 5D [B, C, T, H, W] inputs.
        If encoder output is 2D [B, D], unsqueeze to [B, 1, D].
        """
        out = self.encoder(frames)
        if out.dim() == 2:
            out = out.unsqueeze(1)  # [B, D] -> [B, 1, D]
        return out  # [B, N, D]

    @staticmethod
    def _unpack_batch(batch) -> Tuple[Tensor, Tensor]:
        """Unpack a DataLoader batch into (frames, labels)."""
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            return batch[0], batch[1]
        raise ValueError(f"Unexpected batch format: {type(batch)}")


# ---------------------------------------------------------------------------
# Multi-Segment Assessment Helper
# ---------------------------------------------------------------------------

def multi_segment_predict(
    encoder: nn.Module,
    probe: nn.Module,
    clips: Tensor,
    num_segments: int,
    num_views: int,
) -> Tensor:
    """
    Average logits across (num_segments * num_views) clips per video.

    Args:
        encoder:      Frozen encoder.
        probe:        Trained classifier head.
        clips:        [B * num_segments * num_views, ...] batch of clips.
        num_segments: Number of temporal segments per video.
        num_views:    Number of spatial views per segment.

    Returns:
        avg_logits: [B, num_classes] averaged over all clips.
    """
    total_clips = num_segments * num_views
    B_total     = clips.shape[0]
    B           = B_total // total_clips

    encoder.train(False)
    probe.train(False)

    with torch.no_grad():
        features = encoder(clips)          # [B*total_clips, N, D]
        logits   = probe(features)         # [B*total_clips, C]

    logits     = logits.view(B, total_clips, -1)  # [B, total_clips, C]
    avg_logits = logits.mean(dim=1)               # [B, C]
    return avg_logits


# ---------------------------------------------------------------------------
# Self-Tests
# ---------------------------------------------------------------------------

class _TinyEncoder(nn.Module):
    """Minimal encoder for testing: maps [B, N, D_in] -> [B, N, D_out]."""

    def __init__(self, embed_dim: int = 64) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.linear    = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


def _make_synthetic_loader(
    num_samples: int = 32,
    num_patches: int = 49,
    embed_dim: int = 64,
    num_classes: int = 10,
    batch_size: int = 8,
) -> DataLoader:
    features = torch.randn(num_samples, num_patches, embed_dim)
    labels   = torch.randint(0, num_classes, (num_samples,))
    dataset  = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class _TestFrozenBackboneAssessor(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.embed_dim   = 64
        self.num_classes = 10
        self.encoder     = _TinyEncoder(self.embed_dim)
        self.config      = _MinimalConfig()
        self.config.num_classes = self.num_classes
        self.config.embed_dim   = self.embed_dim
        self.assessor    = FrozenBackboneAssessor(self.encoder, self.config)
        self.train_loader = _make_synthetic_loader(
            embed_dim=self.embed_dim, num_classes=self.num_classes
        )
        self.val_loader   = _make_synthetic_loader(
            num_samples=16, embed_dim=self.embed_dim, num_classes=self.num_classes
        )
        self.multihead_kwargs = [
            {"lr": 1e-2, "wd": 1e-4},
            {"lr": 5e-3, "wd": 1e-4},
        ]

    # -- Freeze tests -------------------------------------------------------

    def test_encoder_requires_grad_is_false(self):
        for p in self.assessor.encoder.parameters():
            self.assertFalse(p.requires_grad)

    def test_encoder_gradients_stay_none_after_training(self):
        self.assessor.train_probes(
            self.train_loader, self.multihead_kwargs, num_epochs=1
        )
        # Verify no gradients accumulated on encoder
        for name, p in self.assessor.encoder.named_parameters():
            self.assertIsNone(p.grad, f"Encoder param '{name}' has gradient")

    def test_probe_has_gradients_after_backward(self):
        self.assessor.train_probes(
            self.train_loader, self.multihead_kwargs, num_epochs=1
        )
        probe = self.assessor.get_probe(0)
        # Manually run one forward + backward to check gradient state
        features = torch.randn(4, 49, self.embed_dim)
        labels   = torch.randint(0, self.num_classes, (4,))
        logits   = probe(features)
        loss     = F.cross_entropy(logits, labels)
        loss.backward()
        has_any_grad = any(p.grad is not None for p in probe.parameters())
        self.assertTrue(has_any_grad)

    # -- Training tests -----------------------------------------------------

    def test_probe_loss_decreases(self):
        """
        Track loss across steps during a single training run.
        First-5-step average should be higher than last-5-step average,
        demonstrating the probe is learning on fixed synthetic features.
        """
        torch.manual_seed(1)
        # Pre-compute fixed features (no random encoder involved)
        num_patches = 49
        features = torch.randn(64, num_patches, self.embed_dim)
        labels   = torch.randint(0, self.num_classes, (64,))

        # Create a probe and optimizer directly (bypass the assessor encoder path)
        probe = AttentiveClassifier(
            embed_dim=self.embed_dim,
            num_classes=self.num_classes,
        )
        optimizer = torch.optim.SGD(probe.parameters(), lr=1e-1, weight_decay=0.0, momentum=0.9)
        probe.train()

        step_losses = []
        for _epoch in range(10):
            perm = torch.randperm(64)
            for i in range(0, 64, 8):
                idx    = perm[i : i + 8]
                feat   = features[idx]
                lbl    = labels[idx]
                logits = probe(feat)
                loss   = F.cross_entropy(logits, lbl)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step_losses.append(loss.item())

        n = len(step_losses)
        quarter = max(1, n // 4)
        first_avg = sum(step_losses[:quarter]) / quarter
        last_avg  = sum(step_losses[-quarter:]) / quarter

        self.assertLess(
            last_avg, first_avg,
            f"Last-quarter loss ({last_avg:.4f}) should be < first-quarter loss ({first_avg:.4f})"
        )

    def test_n_probes_created(self):
        self.assessor.train_probes(
            self.train_loader, self.multihead_kwargs, num_epochs=1
        )
        self.assertEqual(len(self.assessor._probes), len(self.multihead_kwargs))

    # -- Validation tests ---------------------------------------------------

    def test_run_validation_returns_n_metrics(self):
        self.assessor.train_probes(
            self.train_loader, self.multihead_kwargs, num_epochs=1
        )
        metrics = self.assessor.run_validation(self.val_loader)
        self.assertEqual(len(metrics), len(self.multihead_kwargs))

    def test_run_validation_contains_accuracy_key(self):
        self.assessor.train_probes(
            self.train_loader, self.multihead_kwargs, num_epochs=1
        )
        metrics = self.assessor.run_validation(self.val_loader)
        for key, m in metrics.items():
            self.assertIn("accuracy", m, f"Missing 'accuracy' in {key}")

    def test_best_head_index_in_range(self):
        self.assessor.train_probes(
            self.train_loader, self.multihead_kwargs, num_epochs=1
        )
        self.assessor.run_validation(self.val_loader)
        idx, _ = self.assessor.best_head()
        self.assertGreaterEqual(idx, 0)
        self.assertLess(idx, len(self.multihead_kwargs))

    def test_val_only_skips_training(self):
        self.config.val_only = True
        assessor = FrozenBackboneAssessor(self.encoder, self.config)
        assessor.train_probes(self.train_loader, self.multihead_kwargs, num_epochs=5)
        # No probes should be created
        self.assertEqual(len(assessor._probes), 0)


import math


if __name__ == "__main__":
    print("Running FrozenBackboneAssessor self-tests...")
    suite  = unittest.TestLoader().loadTestsFromTestCase(_TestFrozenBackboneAssessor)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\nAll self-tests passed.")
    else:
        raise SystemExit(1)
