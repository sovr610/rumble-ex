"""
phase_runner_template.py -- Per-Phase Training Runner for Brain-AI Pipeline

Provides a comprehensive, self-contained template for running each of the
seven training phases in the brain-inspired AI system:

    Phase 1: SNN Core (LIF neurons, surrogate gradients)
    Phase 2: Modality Encoders (vision, text, audio -> 4096-dim workspace)
    Phase 3: HTM (spatial pooler, temporal memory, anomaly detection)
    Phase 4: Global Workspace (competition, broadcast, ignition, working memory)
    Phase 5: Active Inference (generative model, EFE, planning)
    Phase 6: Reasoning (dual-process System 1/2, fuzzy logic, symbolic KB)
    Phase 7: Meta-Learning (MAML/FOMAML/Reptile, episodic few-shot)

Classes:
    PhaseConfig          -- Per-phase training parameter defaults and merging.
    PhaseRunner          -- Full lifecycle: setup, build, train, validate, finalize.
    TrainingLoop         -- Generic training loop with AMP, grad clipping, hooks.
    DevModeAdapter       -- Shrink datasets, epochs, and batch sizes for fast iteration.
    PhaseDatasetRegistry -- Map each phase to its expected dataset(s).
    CLIParser            -- Argparse builder for train_phase{N}.py scripts.
    MockModel            -- Simple nn.Module for self-testing (no brain_ai imports).

Self-contained: no brain_ai imports required.  Uses torch for model/optimizer/
dataloader.  All seven phases can run in dev mode with synthetic data.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HAS_CUDA = torch.cuda.is_available()
_DEVICE = "cuda" if _HAS_CUDA else "cpu"

_PHASE_NAMES: Dict[int, str] = {
    1: "SNN Core",
    2: "Modality Encoders",
    3: "HTM",
    4: "Global Workspace",
    5: "Active Inference",
    6: "Reasoning",
    7: "Meta-Learning",
}


# =========================================================================
# PhaseConfig
# =========================================================================

PHASE_DEFAULTS: Dict[int, Dict[str, Any]] = {
    1: {
        "lr": 1e-3,
        "epochs": 50,
        "dev_epochs": 2,
        "batch_size": 64,
        "optimizer": "adam",
        "use_amp": False,
        "grad_clip_norm": 1.0,
        "log_every_n_steps": 50,
        "save_every_n_steps": 500,
        "early_stopping_patience": 10,
        "weight_decay": 0.0,
        "scheduler": "cosine",
        "warmup_steps": 100,
    },
    2: {
        "lr": 5e-4,
        "epochs": 30,
        "dev_epochs": 2,
        "batch_size": 32,
        "optimizer": "adam",
        "use_amp": False,
        "grad_clip_norm": 1.0,
        "log_every_n_steps": 50,
        "save_every_n_steps": 500,
        "early_stopping_patience": 8,
        "weight_decay": 0.0,
        "scheduler": "cosine",
        "warmup_steps": 100,
    },
    3: {
        "lr": 1e-3,
        "epochs": 20,
        "dev_epochs": 2,
        "batch_size": 128,
        "optimizer": "adam",
        "use_amp": False,
        "grad_clip_norm": 5.0,
        "log_every_n_steps": 20,
        "save_every_n_steps": 200,
        "early_stopping_patience": 5,
        "weight_decay": 0.0,
        "scheduler": "step",
        "warmup_steps": 0,
        "htm_custom_update": True,
    },
    4: {
        "lr": 3e-4,
        "epochs": 40,
        "dev_epochs": 2,
        "batch_size": 32,
        "optimizer": "adamw",
        "use_amp": True,
        "grad_clip_norm": 1.0,
        "log_every_n_steps": 50,
        "save_every_n_steps": 500,
        "early_stopping_patience": 10,
        "weight_decay": 1e-2,
        "scheduler": "cosine",
        "warmup_steps": 200,
    },
    5: {
        "lr": 1e-4,
        "epochs": 30,
        "dev_epochs": 2,
        "batch_size": 16,
        "optimizer": "adamw",
        "use_amp": False,
        "grad_clip_norm": 0.5,
        "log_every_n_steps": 25,
        "save_every_n_steps": 300,
        "early_stopping_patience": 8,
        "weight_decay": 1e-2,
        "scheduler": "cosine",
        "warmup_steps": 100,
    },
    6: {
        "lr": 1e-4,
        "epochs": 25,
        "dev_epochs": 2,
        "batch_size": 16,
        "optimizer": "adamw",
        "use_amp": False,
        "grad_clip_norm": 1.0,
        "log_every_n_steps": 25,
        "save_every_n_steps": 300,
        "early_stopping_patience": 8,
        "weight_decay": 1e-2,
        "scheduler": "cosine",
        "warmup_steps": 100,
    },
    7: {
        "lr": 1e-2,
        "outer_lr": 1e-4,
        "epochs": 50,
        "dev_epochs": 2,
        "batch_size": 4,
        "optimizer": "adam",
        "use_amp": False,
        "grad_clip_norm": 2.0,
        "log_every_n_steps": 10,
        "save_every_n_steps": 100,
        "early_stopping_patience": 15,
        "weight_decay": 0.0,
        "scheduler": "cosine",
        "warmup_steps": 50,
        "episodes_per_epoch": 600,
        "dev_episodes_per_epoch": 10,
        "inner_steps": 5,
        "n_way": 5,
        "k_shot": 1,
    },
}


def get_phase_config(phase: int, mode: str = "dev") -> Dict[str, Any]:
    """Return training config for a specific phase and mode.

    Parameters
    ----------
    phase : int
        Training phase number, 1 through 7.
    mode : str
        ``"dev"`` for fast iteration or ``"production"`` for full training.

    Returns
    -------
    dict
        Training configuration with mode-appropriate values applied.
    """
    if phase not in PHASE_DEFAULTS:
        raise ValueError(f"Unknown phase {phase}. Must be 1-7.")
    mode = mode.lower()
    if mode not in ("dev", "production"):
        raise ValueError(f"mode must be 'dev' or 'production', got '{mode}'")

    config = dict(PHASE_DEFAULTS[phase])
    config["phase"] = phase
    config["mode"] = mode
    config["phase_name"] = _PHASE_NAMES[phase]

    if mode == "dev":
        config["epochs"] = config.get("dev_epochs", 2)
        config["batch_size"] = min(16, config["batch_size"])
        config["subset_size"] = 1000
        config["use_amp"] = False
        if phase == 7:
            config["episodes_per_epoch"] = config.get("dev_episodes_per_epoch", 10)
    else:
        config["subset_size"] = None
        if phase == 7:
            config["episodes_per_epoch"] = PHASE_DEFAULTS[7]["episodes_per_epoch"]

    # Remove internal dev-only keys
    config.pop("dev_epochs", None)
    config.pop("dev_episodes_per_epoch", None)

    return config


# =========================================================================
# DevModeAdapter
# =========================================================================

class DevModeAdapter:
    """Adapter for shrinking datasets, dataloaders, and configs for fast dev iteration."""

    @staticmethod
    def wrap_dataset(
        dataset: Dataset, subset_size: int = 1000, seed: int = 42
    ) -> Subset:
        """Create a deterministic small subset of a dataset.

        Parameters
        ----------
        dataset : Dataset
            The full dataset to subset.
        subset_size : int
            Maximum number of samples to include.
        seed : int
            Random seed for reproducible index selection.

        Returns
        -------
        Subset
            A subset containing at most *subset_size* samples.
        """
        n = len(dataset)
        actual_size = min(subset_size, n)
        gen = torch.Generator()
        gen.manual_seed(seed)
        indices = torch.randperm(n, generator=gen)[:actual_size].tolist()
        return Subset(dataset, indices)

    @staticmethod
    def wrap_dataloader(dataloader: DataLoader, max_batches: int = 50) -> "_LimitedDataLoader":
        """Wrap a DataLoader to limit the number of batches per epoch.

        Parameters
        ----------
        dataloader : DataLoader
            Original dataloader.
        max_batches : int
            Maximum number of batches to yield per iteration.

        Returns
        -------
        _LimitedDataLoader
            A wrapper that stops after *max_batches*.
        """
        return _LimitedDataLoader(dataloader, max_batches)

    @staticmethod
    def adjust_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Reduce config values for fast dev iteration.

        Parameters
        ----------
        config : dict
            Training configuration to adjust.

        Returns
        -------
        dict
            Adjusted copy of the configuration.
        """
        adjusted = dict(config)
        adjusted["epochs"] = min(2, adjusted.get("epochs", 2))
        adjusted["batch_size"] = min(16, adjusted.get("batch_size", 16))
        adjusted["subset_size"] = adjusted.get("subset_size", 1000)
        adjusted["use_amp"] = False
        if "episodes_per_epoch" in adjusted:
            adjusted["episodes_per_epoch"] = min(10, adjusted["episodes_per_epoch"])
        return adjusted

    @staticmethod
    def is_dev_mode(config: Dict[str, Any]) -> bool:
        """Check whether the config represents dev mode.

        Parameters
        ----------
        config : dict
            Training configuration with a ``mode`` key.

        Returns
        -------
        bool
        """
        return config.get("mode", "dev").lower() == "dev"


class _LimitedDataLoader:
    """Wraps a DataLoader to yield at most ``max_batches`` per iteration."""

    def __init__(self, dataloader: DataLoader, max_batches: int) -> None:
        self._dataloader = dataloader
        self._max_batches = max_batches

    def __iter__(self):
        for i, batch in enumerate(self._dataloader):
            if i >= self._max_batches:
                break
            yield batch

    def __len__(self) -> int:
        return min(self._max_batches, len(self._dataloader))

    @property
    def dataset(self) -> Dataset:
        return self._dataloader.dataset


# =========================================================================
# PhaseDatasetRegistry
# =========================================================================

class PhaseDatasetRegistry:
    """Maps each training phase to its expected datasets and provides
    synthetic mock datasets for testing without file I/O."""

    _PHASE_DATASETS: Dict[int, Dict[str, Dict[str, str]]] = {
        1: {
            "dev": {"name": "MNIST", "split": "train"},
            "production": {"name": "ImageNet", "split": "train"},
        },
        2: {
            "dev": {"name": "MNIST", "split": "train"},
            "production": {"name": "ImageNet", "split": "train"},
        },
        3: {
            "dev": {"name": "MNIST", "split": "train"},
            "production": {"name": "ImageNet", "split": "train"},
        },
        4: {
            "dev": {"name": "MNIST", "split": "train"},
            "production": {"name": "ImageNet", "split": "train"},
        },
        5: {
            "dev": {"name": "MNIST", "split": "train"},
            "production": {"name": "ImageNet", "split": "train"},
        },
        6: {
            "dev": {"name": "MNIST", "split": "train"},
            "production": {"name": "ImageNet", "split": "train"},
        },
        7: {
            "dev": {"name": "Omniglot", "split": "train"},
            "production": {"name": "mini-ImageNet", "split": "train"},
        },
    }

    # Synthetic data shapes per phase
    _MOCK_SHAPES: Dict[int, Tuple[Tuple[int, ...], int]] = {
        1: ((1, 28, 28), 10),      # SNN: image-like input, 10 classes
        2: ((1, 28, 28), 10),      # Encoders: multi-modal, simplified to image
        3: ((128,), 10),           # HTM: pre-encoded features
        4: ((4096,), 10),          # GW: workspace-dim input
        5: ((4096,), 10),          # Active Inference: workspace-dim
        6: ((4096,), 10),          # Reasoning: workspace-dim
        7: ((1, 28, 28), 20),     # Meta-learning: image, more classes
    }

    @classmethod
    def get_datasets(cls, phase: int, mode: str = "dev") -> Dict[str, str]:
        """Return the dataset configuration for a given phase and mode.

        Parameters
        ----------
        phase : int
            Training phase (1-7).
        mode : str
            ``"dev"`` or ``"production"``.

        Returns
        -------
        dict
            Dataset name and split information.
        """
        if phase not in cls._PHASE_DATASETS:
            raise ValueError(f"Unknown phase {phase}. Must be 1-7.")
        mode = mode.lower()
        if mode not in ("dev", "production"):
            raise ValueError(f"mode must be 'dev' or 'production', got '{mode}'")
        return dict(cls._PHASE_DATASETS[phase][mode])

    @classmethod
    def build_mock_dataset(
        cls,
        phase: int,
        mode: str = "dev",
        num_samples: int = 500,
    ) -> TensorDataset:
        """Create a synthetic TensorDataset for testing (no file I/O).

        Parameters
        ----------
        phase : int
            Training phase (1-7).
        mode : str
            ``"dev"`` or ``"production"`` (only affects default *num_samples*
            if not specified).
        num_samples : int
            Number of synthetic samples.

        Returns
        -------
        TensorDataset
            Dataset of (input_tensor, label_tensor) pairs.
        """
        if phase not in cls._MOCK_SHAPES:
            raise ValueError(f"Unknown phase {phase}. Must be 1-7.")
        input_shape, num_classes = cls._MOCK_SHAPES[phase]
        x = torch.randn(num_samples, *input_shape)
        y = torch.randint(0, num_classes, (num_samples,))
        return TensorDataset(x, y)


# =========================================================================
# CLIParser
# =========================================================================

class CLIParser:
    """Argument parser builder for per-phase training scripts."""

    @staticmethod
    def build_parser(phase: int) -> argparse.ArgumentParser:
        """Build an argparse.ArgumentParser for ``train_phase{phase}.py``.

        Parameters
        ----------
        phase : int
            Training phase number (1-7).

        Returns
        -------
        argparse.ArgumentParser
        """
        phase_name = _PHASE_NAMES.get(phase, f"Phase {phase}")
        parser = argparse.ArgumentParser(
            description=f"Train Phase {phase}: {phase_name}",
        )
        parser.add_argument(
            "--mode", type=str, default="dev",
            choices=["dev", "production"],
            help="Training mode: dev (fast) or production (full)",
        )
        parser.add_argument("--resume-from", type=str, default=None,
                            help="Path to checkpoint to resume from")
        parser.add_argument("--run-dir", type=str, default=None,
                            help="Directory for run outputs (created if needed)")
        parser.add_argument("--seed", type=int, default=1337,
                            help="Random seed for reproducibility")
        parser.add_argument("--use-amp", action="store_true", default=False,
                            help="Enable automatic mixed precision")
        parser.add_argument("--epochs", type=int, default=None,
                            help="Override number of epochs")
        parser.add_argument("--batch-size", type=int, default=None,
                            help="Override batch size")
        parser.add_argument("--lr", type=float, default=None,
                            help="Override learning rate")
        parser.add_argument("--device", type=str, default=None,
                            help="Device (cpu, cuda, cuda:0, etc.)")
        parser.add_argument("--num-workers", type=int, default=2,
                            help="DataLoader worker count")

        # Phase-7-specific arguments
        if phase == 7:
            parser.add_argument("--inner-steps", type=int, default=None,
                                help="Number of inner MAML adaptation steps")
            parser.add_argument("--n-way", type=int, default=None,
                                help="N-way classification for few-shot")
            parser.add_argument("--k-shot", type=int, default=None,
                                help="K-shot examples per class")
            parser.add_argument("--outer-lr", type=float, default=None,
                                help="Outer loop learning rate")
            parser.add_argument("--episodes-per-epoch", type=int, default=None,
                                help="Number of episodes per epoch")

        return parser

    @staticmethod
    def parse_and_merge(
        parser: argparse.ArgumentParser,
        config: Dict[str, Any],
        args: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Parse CLI arguments and override config values.

        Parameters
        ----------
        parser : argparse.ArgumentParser
            Parser built by ``build_parser``.
        config : dict
            Base training config (e.g. from ``get_phase_config``).
        args : list[str] or None
            Argument strings.  ``None`` uses ``sys.argv[1:]``.

        Returns
        -------
        dict
            Merged config with CLI overrides applied.
        """
        parsed = parser.parse_args(args if args is not None else [])
        merged = dict(config)

        # Direct overrides -- only apply if CLI value is not None / not default
        cli_map = {
            "epochs": "epochs",
            "batch_size": "batch_size",
            "lr": "lr",
            "seed": "seed",
            "device": "device",
            "num_workers": "num_workers",
        }
        for cli_key, config_key in cli_map.items():
            val = getattr(parsed, cli_key, None)
            if val is not None:
                merged[config_key] = val

        # use_amp is a store_true flag -- only override if explicitly set
        if parsed.use_amp:
            merged["use_amp"] = True

        # String / path overrides
        if parsed.resume_from is not None:
            merged["resume_from"] = parsed.resume_from
        if parsed.run_dir is not None:
            merged["run_dir"] = parsed.run_dir
        if parsed.mode is not None:
            merged["mode"] = parsed.mode

        # Phase 7 specific
        phase7_map = {
            "inner_steps": "inner_steps",
            "n_way": "n_way",
            "k_shot": "k_shot",
            "outer_lr": "outer_lr",
            "episodes_per_epoch": "episodes_per_epoch",
        }
        for cli_key, config_key in phase7_map.items():
            val = getattr(parsed, cli_key, None)
            if val is not None:
                merged[config_key] = val

        return merged


# =========================================================================
# MockModel
# =========================================================================

class MockModel(nn.Module):
    """Simple nn.Module for self-testing without brain_ai imports.

    Accepts an input tensor, passes through a linear layer + optional hidden
    layer, and returns a loss computed against a target.

    Parameters
    ----------
    input_dim : int
        Flattened input dimension.
    hidden_dim : int
        Hidden layer dimension.
    output_dim : int
        Number of output classes.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 128,
        output_dim: int = 10,
    ) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self, x: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Returns
        -------
        dict
            ``"logits"`` always present; ``"loss"`` present when *targets*
            is provided.
        """
        flat = self.flatten(x)
        logits = self.layers(flat)
        result: Dict[str, torch.Tensor] = {"logits": logits}
        if targets is not None:
            result["loss"] = self.loss_fn(logits, targets)
        return result


# =========================================================================
# TrainingLoop
# =========================================================================

class TrainingLoop:
    """Generic training loop with AMP support, gradient clipping, and hooks.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    optimizer : torch.optim.Optimizer
        Optimizer instance.
    scheduler : optional
        Learning rate scheduler (must support ``step()``).
    logger : logging.Logger or None
        Logger for metric output.
    config : dict
        Training configuration containing at least ``use_amp``,
        ``grad_clip_norm``, ``log_every_n_steps``, ``save_every_n_steps``.
    device : str
        Device string (``"cpu"`` or ``"cuda"``).
    checkpoint_dir : str or Path or None
        Directory to save checkpoints.  ``None`` disables checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
        config: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
        checkpoint_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger or logging.getLogger("TrainingLoop")
        self.config = config or {}
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        # AMP
        self.use_amp = self.config.get("use_amp", False) and _HAS_CUDA
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Gradient clipping
        self.grad_clip_norm = self.config.get("grad_clip_norm", 0.0)

        # Logging frequency
        self.log_every_n_steps = self.config.get("log_every_n_steps", 50)
        self.save_every_n_steps = self.config.get("save_every_n_steps", 500)

        # Early stopping
        self.early_stopping_patience = self.config.get("early_stopping_patience", 10)

        # Tracking
        self.global_step: int = 0
        self.best_val_loss: float = float("inf")
        self.patience_counter: int = 0

    def train_step(self, batch: Tuple[torch.Tensor, ...]) -> Dict[str, float]:
        """Execute a single training step: forward + loss + backward + optimizer.

        Parameters
        ----------
        batch : tuple of tensors
            ``(inputs, targets)`` pair.

        Returns
        -------
        dict
            Metrics including ``"loss"`` and ``"lr"``.
        """
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        self.optimizer.zero_grad()

        if self.use_amp:
            with torch.amp.autocast("cuda"):
                output = self.model(inputs, targets)
                loss = output["loss"]
            self.scaler.scale(loss).backward()
            if self.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            output = self.model(inputs, targets)
            loss = output["loss"]
            loss.backward()
            if self.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm
                )
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        self.global_step += 1

        # Compute accuracy
        with torch.no_grad():
            preds = output["logits"].argmax(dim=-1)
            correct = (preds == targets).float().mean().item()

        current_lr = self.optimizer.param_groups[0]["lr"]
        return {
            "loss": loss.item(),
            "accuracy": correct,
            "lr": current_lr,
        }

    def train_epoch(
        self, dataloader: Union[DataLoader, _LimitedDataLoader], epoch: int
    ) -> Dict[str, float]:
        """Run one full training epoch.

        Parameters
        ----------
        dataloader : DataLoader
            Training data iterator.
        epoch : int
            Current epoch number (for logging).

        Returns
        -------
        dict
            Averaged metrics over the epoch.
        """
        self.model.train()
        epoch_metrics: Dict[str, List[float]] = {}
        num_batches = 0

        for batch in dataloader:
            step_metrics = self.train_step(batch)
            num_batches += 1

            for key, val in step_metrics.items():
                epoch_metrics.setdefault(key, []).append(val)

            if self.global_step % self.log_every_n_steps == 0:
                window = self.log_every_n_steps
                avg_loss = np.mean(epoch_metrics.get("loss", [0.0])[-window:])
                self.logger.info(
                    f"  [epoch {epoch}] step {self.global_step}: "
                    f"loss={avg_loss:.4f}, lr={step_metrics['lr']:.2e}"
                )

            if self.should_checkpoint(self.global_step, epoch):
                self._save_checkpoint(epoch, "step")

        # Compute averages
        averaged: Dict[str, float] = {}
        for key, vals in epoch_metrics.items():
            averaged[key] = float(np.mean(vals))
        averaged["num_batches"] = float(num_batches)
        return averaged

    @torch.no_grad()
    def validate(
        self, dataloader: Union[DataLoader, _LimitedDataLoader]
    ) -> Dict[str, float]:
        """Run a validation pass.

        Parameters
        ----------
        dataloader : DataLoader
            Validation data iterator.

        Returns
        -------
        dict
            Averaged validation metrics.
        """
        self.model.eval()
        val_metrics: Dict[str, List[float]] = {}

        for batch in dataloader:
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    output = self.model(inputs, targets)
            else:
                output = self.model(inputs, targets)

            loss = output["loss"].item()
            preds = output["logits"].argmax(dim=-1)
            accuracy = (preds == targets).float().mean().item()

            val_metrics.setdefault("val_loss", []).append(loss)
            val_metrics.setdefault("val_accuracy", []).append(accuracy)

        averaged: Dict[str, float] = {}
        for key, vals in val_metrics.items():
            averaged[key] = float(np.mean(vals))
        return averaged

    def should_checkpoint(self, step: int, epoch: int) -> bool:
        """Determine if a checkpoint should be saved at this step.

        Parameters
        ----------
        step : int
            Current global step.
        epoch : int
            Current epoch number.

        Returns
        -------
        bool
        """
        if self.checkpoint_dir is None:
            return False
        if self.save_every_n_steps <= 0:
            return False
        return step > 0 and step % self.save_every_n_steps == 0

    def _save_checkpoint(self, epoch: int, tag: str = "step") -> Optional[Path]:
        """Save a checkpoint to disk.

        Parameters
        ----------
        epoch : int
            Current epoch.
        tag : str
            Checkpoint tag (``"step"``, ``"epoch"``, ``"best"``).

        Returns
        -------
        Path or None
            Path to the saved checkpoint, or ``None`` if checkpointing
            is disabled.
        """
        if self.checkpoint_dir is None:
            return None
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        filename = f"checkpoint_{tag}_e{epoch}_s{self.global_step}.pt"
        path = self.checkpoint_dir / filename
        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }
        if self.scaler is not None:
            state["scaler_state_dict"] = self.scaler.state_dict()
        if self.scheduler is not None and hasattr(self.scheduler, "state_dict"):
            state["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(state, path)
        self.logger.info(f"  Checkpoint saved: {path}")
        return path

    def _check_early_stopping(self, val_loss: float) -> bool:
        """Check early stopping criterion.

        Parameters
        ----------
        val_loss : float
            Current validation loss.

        Returns
        -------
        bool
            ``True`` if training should stop.
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            self._save_checkpoint(epoch=-1, tag="best")
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.early_stopping_patience:
                self.logger.info(
                    f"  Early stopping triggered after {self.patience_counter} "
                    f"epochs without improvement."
                )
                return True
            return False

    def run_training(
        self,
        train_loader: Union[DataLoader, _LimitedDataLoader],
        val_loader: Union[DataLoader, _LimitedDataLoader],
        num_epochs: int,
    ) -> Dict[str, Any]:
        """Run the full training loop with validation and checkpointing.

        Parameters
        ----------
        train_loader : DataLoader
            Training data iterator.
        val_loader : DataLoader
            Validation data iterator.
        num_epochs : int
            Number of training epochs.

        Returns
        -------
        dict
            Final training summary with per-epoch metrics history.
        """
        self.logger.info(f"Starting training for {num_epochs} epochs...")
        history: Dict[str, List[Dict[str, float]]] = {
            "train": [],
            "val": [],
        }

        for epoch in range(1, num_epochs + 1):
            self.logger.info(f"Epoch {epoch}/{num_epochs}")

            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            history["train"].append(train_metrics)
            self.logger.info(
                f"  Train: loss={train_metrics['loss']:.4f}, "
                f"acc={train_metrics.get('accuracy', 0.0):.4f}"
            )

            # Validate
            val_metrics = self.validate(val_loader)
            history["val"].append(val_metrics)
            self.logger.info(
                f"  Val:   loss={val_metrics['val_loss']:.4f}, "
                f"acc={val_metrics.get('val_accuracy', 0.0):.4f}"
            )

            # Epoch checkpoint
            self._save_checkpoint(epoch, tag="epoch")

            # Early stopping
            if self._check_early_stopping(val_metrics["val_loss"]):
                break

        return {
            "final_train": history["train"][-1] if history["train"] else {},
            "final_val": history["val"][-1] if history["val"] else {},
            "history": history,
            "total_steps": self.global_step,
            "best_val_loss": self.best_val_loss,
        }


# =========================================================================
# PhaseRunner
# =========================================================================

class PhaseRunner:
    """Full lifecycle runner for a single training phase.

    Orchestrates: setup -> build_model -> build_optimizer -> build_dataloader
    -> train_epoch/validate loop -> finalize.

    Parameters
    ----------
    phase : int
        Training phase number (1-7).
    config : dict or None
        Override config.  If ``None``, ``get_phase_config`` is called.
    run_dir : str or Path or None
        Root directory for run outputs.  If ``None``, a temp directory
        is created.
    mode : str
        ``"dev"`` or ``"production"``.
    seed : int
        Base random seed.
    """

    def __init__(
        self,
        phase: int,
        config: Optional[Dict[str, Any]] = None,
        run_dir: Optional[Union[str, Path]] = None,
        mode: str = "dev",
        seed: int = 1337,
    ) -> None:
        if phase not in range(1, 8):
            raise ValueError(f"phase must be 1-7, got {phase}")
        self.phase = phase
        self.mode = mode.lower()
        self.seed = seed

        # Merge provided config with defaults
        base_config = get_phase_config(phase, self.mode)
        if config is not None:
            base_config.update(config)
        self.config = base_config
        self.config["seed"] = self.seed

        # Run directory
        if run_dir is None:
            self.run_dir = Path(f"runs/phase{phase}_{self.mode}_{int(time.time())}")
        else:
            self.run_dir = Path(run_dir)

        # Determined at setup
        self.logger: Optional[logging.Logger] = None
        self.device: str = self.config.get("device", _DEVICE)
        self.manifest: Dict[str, Any] = {}

        # Set at build time
        self._model: Optional[nn.Module] = None
        self._optimizer: Optional[optim.Optimizer] = None
        self._scheduler: Optional[Any] = None
        self._training_loop: Optional[TrainingLoop] = None

    def setup(self) -> None:
        """Create run directory, seed RNG, set up logging, capture manifest."""
        # Create directories
        self.run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = self.run_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        # Logging
        self.logger = logging.getLogger(f"Phase{self.phase}")
        self.logger.setLevel(logging.INFO)
        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()

        formatter = logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler(self.run_dir / "training.log")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        # Seed everything
        self._seed_everything()

        # Manifest
        self.manifest = {
            "phase": self.phase,
            "phase_name": _PHASE_NAMES[self.phase],
            "mode": self.mode,
            "seed": self.seed,
            "config": dict(self.config),
            "device": self.device,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pytorch_version": torch.__version__,
            "cuda_available": _HAS_CUDA,
        }
        self._save_manifest()

        self.logger.info(f"Phase {self.phase} ({_PHASE_NAMES[self.phase]}) setup complete.")
        self.logger.info(f"  Mode: {self.mode}, Device: {self.device}")
        self.logger.info(f"  Run directory: {self.run_dir}")

    def _seed_everything(self) -> None:
        """Set all RNG seeds for reproducibility."""
        seed = self.seed
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)
        if _HAS_CUDA:
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = (self.mode == "production")

    def _save_manifest(self) -> None:
        """Write the manifest dict to JSON."""
        path = self.run_dir / "manifest.json"
        with open(path, "w") as f:
            json.dump(self.manifest, f, indent=2, default=str)

    def build_model(
        self, resume_boundary: Optional[str] = None
    ) -> nn.Module:
        """Instantiate the model for this phase.

        In this template, a MockModel is used.  In production, this would
        dispatch to the appropriate brain_ai module based on ``self.phase``.

        Parameters
        ----------
        resume_boundary : str or None
            Path to a checkpoint from a previous phase to load as boundary
            weights.

        Returns
        -------
        nn.Module
            The model, moved to ``self.device``.
        """
        # Determine dimensions based on phase
        shapes = PhaseDatasetRegistry._MOCK_SHAPES[self.phase]
        input_shape, num_classes = shapes
        input_dim = 1
        for s in input_shape:
            input_dim *= s

        model = MockModel(
            input_dim=input_dim,
            hidden_dim=128,
            output_dim=num_classes,
        )

        # Load boundary weights if provided
        if resume_boundary is not None and os.path.exists(resume_boundary):
            state = torch.load(resume_boundary, map_location="cpu", weights_only=True)
            if "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"], strict=False)
            if self.logger:
                self.logger.info(f"  Loaded boundary weights from {resume_boundary}")

        model = model.to(self.device)
        self._model = model

        param_count = sum(p.numel() for p in model.parameters())
        if self.logger:
            self.logger.info(f"  Model built: {param_count:,} parameters")
        self.manifest["param_count"] = param_count
        return model

    def build_optimizer(
        self, model: nn.Module
    ) -> Tuple[optim.Optimizer, Optional[Any]]:
        """Create optimizer and scheduler based on phase config.

        Parameters
        ----------
        model : nn.Module
            The model whose parameters will be optimized.

        Returns
        -------
        tuple of (Optimizer, scheduler or None)
        """
        lr = self.config["lr"]
        weight_decay = self.config.get("weight_decay", 0.0)
        opt_name = self.config.get("optimizer", "adam").lower()

        if opt_name == "adam":
            optimizer = optim.Adam(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif opt_name == "adamw":
            optimizer = optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif opt_name == "sgd":
            optimizer = optim.SGD(
                model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

        # Scheduler
        sched_name = self.config.get("scheduler", "cosine").lower()
        total_steps = self._estimate_total_steps()

        if sched_name == "cosine" and total_steps > 0:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(total_steps, 1)
            )
        elif sched_name == "step":
            step_size = max(self.config.get("epochs", 10) // 3, 1)
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=0.1
            )
        else:
            scheduler = None

        self._optimizer = optimizer
        self._scheduler = scheduler

        if self.logger:
            self.logger.info(
                f"  Optimizer: {opt_name}, lr={lr}, weight_decay={weight_decay}"
            )
            self.logger.info(f"  Scheduler: {sched_name}")
        return optimizer, scheduler

    def _estimate_total_steps(self) -> int:
        """Estimate total training steps from config for scheduler setup."""
        epochs = self.config.get("epochs", 10)
        batch_size = self.config.get("batch_size", 32)
        subset_size = self.config.get("subset_size")
        if subset_size is not None:
            steps_per_epoch = max(subset_size // batch_size, 1)
        else:
            # Rough estimate for production
            steps_per_epoch = 1000
        return epochs * steps_per_epoch

    def build_dataloader(
        self,
        split: str = "train",
        dataset: Optional[Dataset] = None,
    ) -> DataLoader:
        """Create a DataLoader with proper worker seeding.

        Parameters
        ----------
        split : str
            ``"train"`` or ``"val"``.
        dataset : Dataset or None
            Override dataset.  If ``None``, a mock dataset is built.

        Returns
        -------
        DataLoader
        """
        if dataset is None:
            num_samples = 500 if self.mode == "dev" else 5000
            dataset = PhaseDatasetRegistry.build_mock_dataset(
                self.phase, self.mode, num_samples=num_samples
            )

        # Apply dev mode subsetting
        if DevModeAdapter.is_dev_mode(self.config):
            subset_size = self.config.get("subset_size", 1000)
            dataset = DevModeAdapter.wrap_dataset(
                dataset, subset_size=subset_size, seed=self.seed
            )

        batch_size = self.config.get("batch_size", 32)
        num_workers = self.config.get("num_workers", 0)
        local_seed = self.seed

        # Worker init function for deterministic seeding
        def worker_init_fn(worker_id: int) -> None:
            worker_seed = local_seed + worker_id
            random.seed(worker_seed)
            np.random.seed(worker_seed % (2 ** 32))
            torch.manual_seed(worker_seed)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            drop_last=(split == "train"),
            pin_memory=_HAS_CUDA,
        )

        if self.logger:
            self.logger.info(
                f"  DataLoader ({split}): {len(dataset)} samples, "
                f"batch_size={batch_size}, workers={num_workers}"
            )
        return loader

    def train_epoch(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        dataloader: Union[DataLoader, _LimitedDataLoader],
        epoch: int,
    ) -> Dict[str, float]:
        """Run a single training epoch with metric logging.

        Parameters
        ----------
        model : nn.Module
            The model to train.
        optimizer : Optimizer
            The optimizer.
        dataloader : DataLoader
            Training data iterator.
        epoch : int
            Current epoch number.

        Returns
        -------
        dict
            Averaged metrics for the epoch.
        """
        if self._training_loop is None:
            raise RuntimeError("Call run() or build the TrainingLoop first.")
        return self._training_loop.train_epoch(dataloader, epoch)

    def validate(
        self,
        model: nn.Module,
        val_dataloader: Union[DataLoader, _LimitedDataLoader],
    ) -> Dict[str, float]:
        """Run a validation pass and return metrics.

        Parameters
        ----------
        model : nn.Module
            The model to validate.
        val_dataloader : DataLoader
            Validation data iterator.

        Returns
        -------
        dict
            Averaged validation metrics.
        """
        if self._training_loop is None:
            raise RuntimeError("Call run() or build the TrainingLoop first.")
        return self._training_loop.validate(val_dataloader)

    def run(self) -> Dict[str, Any]:
        """Execute the full training pipeline for this phase.

        Returns
        -------
        dict
            Training summary with metrics history.
        """
        # Setup
        self.setup()

        # Build
        model = self.build_model(
            resume_boundary=self.config.get("resume_from")
        )
        optimizer, scheduler = self.build_optimizer(model)

        # Data
        train_loader = self.build_dataloader(split="train")
        val_loader = self.build_dataloader(split="val")

        # Dev mode wrapping
        if DevModeAdapter.is_dev_mode(self.config):
            train_loader = DevModeAdapter.wrap_dataloader(train_loader, max_batches=50)
            val_loader = DevModeAdapter.wrap_dataloader(val_loader, max_batches=10)

        # Training loop
        checkpoint_dir = self.run_dir / "checkpoints"
        self._training_loop = TrainingLoop(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            logger=self.logger,
            config=self.config,
            device=self.device,
            checkpoint_dir=checkpoint_dir,
        )

        num_epochs = self.config.get("epochs", 2)
        results = self._training_loop.run_training(
            train_loader, val_loader, num_epochs
        )

        # Finalize
        self.finalize(results)
        return results

    def finalize(self, results: Optional[Dict[str, Any]] = None) -> None:
        """Save phase boundary checkpoint, finalize manifest, close logger.

        Parameters
        ----------
        results : dict or None
            Training results to store in the manifest.
        """
        # Save phase boundary
        if self._model is not None:
            boundary_path = self.run_dir / f"phase{self.phase}_boundary.pt"
            torch.save(
                {"model_state_dict": self._model.state_dict()},
                boundary_path,
            )
            if self.logger:
                self.logger.info(f"  Phase boundary saved: {boundary_path}")

        # Update manifest
        self.manifest["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if results:
            self.manifest["results"] = {
                "best_val_loss": results.get("best_val_loss", None),
                "total_steps": results.get("total_steps", 0),
            }
            # Store serializable final metrics
            for key in ("final_train", "final_val"):
                if key in results:
                    self.manifest["results"][key] = {
                        k: float(v) for k, v in results[key].items()
                    }
        self._save_manifest()

        # Close logger handlers
        if self.logger:
            self.logger.info("Phase finalized.")
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)


# =========================================================================
# Self-test suite
# =========================================================================

def _banner(text: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}")


def _pass(name: str) -> None:
    print(f"  [PASS] {name}")


def _fail(name: str, detail: str = "") -> None:
    msg = f"  [FAIL] {name}"
    if detail:
        msg += f" -- {detail}"
    print(msg)


def _run_tests() -> None:
    """Execute the full self-test suite."""
    import tempfile

    passed = 0
    failed = 0
    failure_details: List[str] = []

    def check(condition: bool, name: str, detail: str = "") -> None:
        nonlocal passed, failed
        if condition:
            _pass(name)
            passed += 1
        else:
            _fail(name, detail)
            failed += 1
            failure_details.append(name)

    # Disable deterministic algorithms for tests
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass

    # ==================================================================
    # PhaseConfig tests
    # ==================================================================
    _banner("PhaseConfig -- defaults for all 7 phases")

    # 1-7: Dev mode config for each phase
    for phase in range(1, 8):
        cfg = get_phase_config(phase, "dev")
        check(cfg["phase"] == phase, f"Phase {phase} dev: phase number correct")
        check(cfg["mode"] == "dev", f"Phase {phase} dev: mode is dev")
        check(cfg["epochs"] == 2, f"Phase {phase} dev: epochs=2")
        check(
            cfg["batch_size"] <= 16,
            f"Phase {phase} dev: batch_size <= 16",
            f"got {cfg['batch_size']}",
        )
        check(
            cfg.get("subset_size") == 1000,
            f"Phase {phase} dev: subset_size=1000",
        )
        check(
            cfg["use_amp"] is False,
            f"Phase {phase} dev: AMP disabled",
        )

    # 8-14: Production mode config for each phase
    for phase in range(1, 8):
        cfg = get_phase_config(phase, "production")
        check(cfg["mode"] == "production", f"Phase {phase} prod: mode is production")
        check(
            cfg["epochs"] == PHASE_DEFAULTS[phase]["epochs"],
            f"Phase {phase} prod: epochs={PHASE_DEFAULTS[phase]['epochs']}",
        )
        check(
            cfg["batch_size"] == PHASE_DEFAULTS[phase]["batch_size"],
            f"Phase {phase} prod: batch_size={PHASE_DEFAULTS[phase]['batch_size']}",
        )

    # 15: Phase 1 specific defaults
    cfg1 = get_phase_config(1, "production")
    check(cfg1["lr"] == 1e-3, "Phase 1 prod: lr=1e-3")
    check(cfg1["optimizer"] == "adam", "Phase 1 prod: optimizer=adam")

    # 16: Phase 4 production has AMP enabled
    cfg4 = get_phase_config(4, "production")
    check(cfg4["use_amp"] is True, "Phase 4 prod: use_amp=True")
    check(cfg4["optimizer"] == "adamw", "Phase 4 prod: optimizer=adamw")

    # 17: Phase 7 meta-learning specific
    cfg7_dev = get_phase_config(7, "dev")
    check(cfg7_dev["episodes_per_epoch"] == 10, "Phase 7 dev: episodes_per_epoch=10")
    check("inner_steps" in cfg7_dev, "Phase 7 dev: inner_steps present")
    check("n_way" in cfg7_dev, "Phase 7 dev: n_way present")
    check("k_shot" in cfg7_dev, "Phase 7 dev: k_shot present")

    cfg7_prod = get_phase_config(7, "production")
    check(cfg7_prod["episodes_per_epoch"] == 600, "Phase 7 prod: episodes_per_epoch=600")
    check(cfg7_prod["lr"] == 1e-2, "Phase 7 prod: inner lr=1e-2")
    check(cfg7_prod["outer_lr"] == 1e-4, "Phase 7 prod: outer_lr=1e-4")

    # 18: Invalid phase raises
    raised = False
    try:
        get_phase_config(0, "dev")
    except ValueError:
        raised = True
    check(raised, "get_phase_config: phase=0 raises ValueError")

    # 19: Invalid mode raises
    raised = False
    try:
        get_phase_config(1, "invalid")
    except ValueError:
        raised = True
    check(raised, "get_phase_config: invalid mode raises ValueError")

    # ==================================================================
    # PhaseRunner -- setup
    # ==================================================================
    _banner("PhaseRunner -- setup creates run directory")

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "test_run_phase1"
        runner = PhaseRunner(phase=1, run_dir=run_dir, mode="dev")
        runner.setup()

        # 20: Run directory exists
        check(run_dir.exists(), "PhaseRunner setup: run directory created")

        # 21: Checkpoints subdirectory exists
        check(
            (run_dir / "checkpoints").exists(),
            "PhaseRunner setup: checkpoints dir created",
        )

        # 22: Manifest file written
        check(
            (run_dir / "manifest.json").exists(),
            "PhaseRunner setup: manifest.json created",
        )

        # 23: Log file written
        check(
            (run_dir / "training.log").exists(),
            "PhaseRunner setup: training.log created",
        )

        # 24: Manifest content
        with open(run_dir / "manifest.json") as f:
            manifest = json.load(f)
        check(manifest["phase"] == 1, "PhaseRunner setup: manifest phase=1")
        check(manifest["mode"] == "dev", "PhaseRunner setup: manifest mode=dev")
        check(manifest["seed"] == 1337, "PhaseRunner setup: manifest seed=1337")

        # Cleanup logger
        runner.finalize()

    # ==================================================================
    # PhaseRunner -- build_model with mock
    # ==================================================================
    _banner("PhaseRunner -- build_model (mock)")

    with tempfile.TemporaryDirectory() as tmpdir:
        runner = PhaseRunner(phase=1, run_dir=Path(tmpdir) / "bm", mode="dev")
        runner.setup()
        model = runner.build_model()

        # 25: Model is nn.Module
        check(isinstance(model, nn.Module), "build_model: returns nn.Module")

        # 26: Model is on correct device
        first_param = next(model.parameters())
        expected_device = torch.device(runner.device)
        check(
            first_param.device.type == expected_device.type,
            "build_model: model on correct device",
        )

        # 27: Model can forward pass
        x = torch.randn(2, 1, 28, 28).to(runner.device)
        y = torch.randint(0, 10, (2,)).to(runner.device)
        out = model(x, y)
        check("logits" in out, "build_model: forward returns logits")
        check("loss" in out, "build_model: forward returns loss")

        runner.finalize()

    # ==================================================================
    # TrainingLoop -- 2 epochs on synthetic data
    # ==================================================================
    _banner("TrainingLoop -- 2 epochs synthetic data")

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir) / "ckpts"
        model = MockModel(input_dim=784, hidden_dim=64, output_dim=10)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        config = {
            "use_amp": False,
            "grad_clip_norm": 1.0,
            "log_every_n_steps": 5,
            "save_every_n_steps": 999999,
            "early_stopping_patience": 100,
        }
        loop = TrainingLoop(
            model=model,
            optimizer=optimizer,
            config=config,
            device="cpu",
            checkpoint_dir=ckpt_dir,
        )

        dataset = TensorDataset(
            torch.randn(100, 1, 28, 28),
            torch.randint(0, 10, (100,)),
        )
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=16, shuffle=False)

        results = loop.run_training(train_loader, val_loader, num_epochs=2)

        # 28: Training completes
        check(results is not None, "TrainingLoop: run_training returns results")

        # 29: History has 2 entries
        check(
            len(results["history"]["train"]) == 2,
            "TrainingLoop: 2 train epochs in history",
        )
        check(
            len(results["history"]["val"]) == 2,
            "TrainingLoop: 2 val epochs in history",
        )

        # 30: Loss is a finite number
        final_loss = results["final_train"]["loss"]
        check(
            np.isfinite(final_loss),
            "TrainingLoop: final train loss is finite",
            f"got {final_loss}",
        )

        # 31: Total steps > 0
        check(
            results["total_steps"] > 0,
            "TrainingLoop: total_steps > 0",
        )

        # 32: Best val loss tracked
        check(
            results["best_val_loss"] < float("inf"),
            "TrainingLoop: best_val_loss updated",
        )

    # ==================================================================
    # TrainingLoop -- AMP (if CUDA available)
    # ==================================================================
    _banner("TrainingLoop -- AMP")

    if _HAS_CUDA:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_amp = MockModel(input_dim=784, hidden_dim=64, output_dim=10).cuda()
            optimizer_amp = optim.Adam(model_amp.parameters(), lr=1e-3)
            config_amp = {
                "use_amp": True,
                "grad_clip_norm": 1.0,
                "log_every_n_steps": 5,
                "save_every_n_steps": 999999,
                "early_stopping_patience": 100,
            }
            loop_amp = TrainingLoop(
                model=model_amp,
                optimizer=optimizer_amp,
                config=config_amp,
                device="cuda",
                checkpoint_dir=Path(tmpdir) / "amp_ckpts",
            )
            dataset_amp = TensorDataset(
                torch.randn(50, 1, 28, 28),
                torch.randint(0, 10, (50,)),
            )
            tl = DataLoader(dataset_amp, batch_size=8)
            vl = DataLoader(dataset_amp, batch_size=8)
            results_amp = loop_amp.run_training(tl, vl, num_epochs=1)
            check(
                results_amp is not None and np.isfinite(results_amp["final_train"]["loss"]),
                "AMP training: completes with finite loss",
            )
    else:
        _pass("AMP training: SKIPPED (no CUDA)")
        passed += 1

    # ==================================================================
    # DevModeAdapter
    # ==================================================================
    _banner("DevModeAdapter")

    # 33: wrap_dataset creates subset
    full_dataset = TensorDataset(
        torch.randn(5000, 10),
        torch.randint(0, 5, (5000,)),
    )
    subset = DevModeAdapter.wrap_dataset(full_dataset, subset_size=200, seed=42)
    check(len(subset) == 200, "DevModeAdapter wrap_dataset: correct subset size")

    # 34: Subset is deterministic
    subset2 = DevModeAdapter.wrap_dataset(full_dataset, subset_size=200, seed=42)
    x1, _ = subset[0]
    x2, _ = subset2[0]
    check(torch.equal(x1, x2), "DevModeAdapter wrap_dataset: deterministic")

    # 35: wrap_dataset with size larger than dataset
    small_ds = TensorDataset(torch.randn(50, 10), torch.randint(0, 5, (50,)))
    small_sub = DevModeAdapter.wrap_dataset(small_ds, subset_size=1000)
    check(len(small_sub) == 50, "DevModeAdapter wrap_dataset: caps at dataset size")

    # 36: wrap_dataloader limits batches
    dl = DataLoader(full_dataset, batch_size=10)
    limited = DevModeAdapter.wrap_dataloader(dl, max_batches=3)
    count = sum(1 for _ in limited)
    check(count == 3, "DevModeAdapter wrap_dataloader: limits to 3 batches")

    # 37: adjust_config reduces values
    config_big = {"epochs": 100, "batch_size": 256, "use_amp": True, "episodes_per_epoch": 600}
    adjusted = DevModeAdapter.adjust_config(config_big)
    check(adjusted["epochs"] == 2, "DevModeAdapter adjust_config: epochs=2")
    check(adjusted["batch_size"] == 16, "DevModeAdapter adjust_config: batch_size=16")
    check(adjusted["use_amp"] is False, "DevModeAdapter adjust_config: use_amp=False")
    check(
        adjusted["episodes_per_epoch"] == 10,
        "DevModeAdapter adjust_config: episodes_per_epoch=10",
    )

    # 38: is_dev_mode checks
    check(DevModeAdapter.is_dev_mode({"mode": "dev"}), "is_dev_mode: True for dev")
    check(
        not DevModeAdapter.is_dev_mode({"mode": "production"}),
        "is_dev_mode: False for production",
    )
    check(DevModeAdapter.is_dev_mode({}), "is_dev_mode: True when mode missing (defaults dev)")

    # ==================================================================
    # CLIParser
    # ==================================================================
    _banner("CLIParser")

    # 39: Basic parser builds
    parser = CLIParser.build_parser(phase=1)
    check(parser is not None, "CLIParser build_parser: returns parser for phase 1")

    # 40: Parse default args
    parsed_config = CLIParser.parse_and_merge(
        parser, {"lr": 1e-3, "epochs": 50}, args=[]
    )
    check(parsed_config["lr"] == 1e-3, "CLIParser: default lr preserved")
    check(parsed_config["epochs"] == 50, "CLIParser: default epochs preserved")

    # 41: CLI overrides
    parsed_override = CLIParser.parse_and_merge(
        parser,
        {"lr": 1e-3, "epochs": 50, "batch_size": 64},
        args=["--lr", "5e-4", "--epochs", "10", "--batch-size", "32"],
    )
    check(
        parsed_override["lr"] == 5e-4,
        "CLIParser: --lr override",
    )
    check(parsed_override["epochs"] == 10, "CLIParser: --epochs override")
    check(parsed_override["batch_size"] == 32, "CLIParser: --batch-size override")

    # 42: Mode override
    parsed_mode = CLIParser.parse_and_merge(
        parser, {"mode": "dev"}, args=["--mode", "production"]
    )
    check(parsed_mode["mode"] == "production", "CLIParser: --mode override")

    # 43: AMP flag
    parsed_amp = CLIParser.parse_and_merge(
        parser, {"use_amp": False}, args=["--use-amp"]
    )
    check(parsed_amp["use_amp"] is True, "CLIParser: --use-amp flag")

    # 44: Phase 7 specific args
    parser7 = CLIParser.build_parser(phase=7)
    parsed7 = CLIParser.parse_and_merge(
        parser7,
        {"inner_steps": 5, "n_way": 5, "k_shot": 1},
        args=["--inner-steps", "10", "--n-way", "20", "--k-shot", "5"],
    )
    check(parsed7["inner_steps"] == 10, "CLIParser Phase 7: --inner-steps")
    check(parsed7["n_way"] == 20, "CLIParser Phase 7: --n-way")
    check(parsed7["k_shot"] == 5, "CLIParser Phase 7: --k-shot")

    # 45: Resume and run-dir
    parsed_resume = CLIParser.parse_and_merge(
        parser, {}, args=["--resume-from", "/tmp/ckpt.pt", "--run-dir", "/tmp/run"]
    )
    check(
        parsed_resume.get("resume_from") == "/tmp/ckpt.pt",
        "CLIParser: --resume-from",
    )
    check(parsed_resume.get("run_dir") == "/tmp/run", "CLIParser: --run-dir")

    # ==================================================================
    # PhaseDatasetRegistry
    # ==================================================================
    _banner("PhaseDatasetRegistry")

    # 46-47: Dataset config for all phases
    for phase in range(1, 8):
        ds_dev = PhaseDatasetRegistry.get_datasets(phase, "dev")
        check("name" in ds_dev, f"PhaseDatasetRegistry phase {phase} dev: has name")

        ds_prod = PhaseDatasetRegistry.get_datasets(phase, "production")
        check("name" in ds_prod, f"PhaseDatasetRegistry phase {phase} prod: has name")

    # 48: Phase 7 uses Omniglot/mini-ImageNet
    ds7_dev = PhaseDatasetRegistry.get_datasets(7, "dev")
    check(ds7_dev["name"] == "Omniglot", "PhaseDatasetRegistry: Phase 7 dev uses Omniglot")
    ds7_prod = PhaseDatasetRegistry.get_datasets(7, "production")
    check(
        ds7_prod["name"] == "mini-ImageNet",
        "PhaseDatasetRegistry: Phase 7 prod uses mini-ImageNet",
    )

    # 49: build_mock_dataset returns TensorDataset
    for phase in range(1, 8):
        mock_ds = PhaseDatasetRegistry.build_mock_dataset(phase, "dev", num_samples=100)
        check(
            isinstance(mock_ds, TensorDataset),
            f"build_mock_dataset phase {phase}: returns TensorDataset",
        )
        check(len(mock_ds) == 100, f"build_mock_dataset phase {phase}: correct length")

    # 50: Mock dataset shapes
    mock1 = PhaseDatasetRegistry.build_mock_dataset(1, "dev", num_samples=10)
    x, y = mock1[0]
    check(x.shape == (1, 28, 28), "build_mock_dataset phase 1: input shape (1,28,28)")
    check(y.dim() == 0, "build_mock_dataset phase 1: label is scalar")

    mock4 = PhaseDatasetRegistry.build_mock_dataset(4, "dev", num_samples=10)
    x4, _ = mock4[0]
    check(x4.shape == (4096,), "build_mock_dataset phase 4: input shape (4096,)")

    # ==================================================================
    # Checkpoint saving during training loop
    # ==================================================================
    _banner("Checkpoint saving")

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir) / "ckpt_test"
        model = MockModel(input_dim=100, hidden_dim=32, output_dim=5)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        config = {
            "use_amp": False,
            "grad_clip_norm": 1.0,
            "log_every_n_steps": 999,
            "save_every_n_steps": 3,
            "early_stopping_patience": 100,
        }
        loop = TrainingLoop(
            model=model,
            optimizer=optimizer,
            config=config,
            device="cpu",
            checkpoint_dir=ckpt_dir,
        )

        dataset = TensorDataset(
            torch.randn(30, 100), torch.randint(0, 5, (30,))
        )
        train_loader = DataLoader(dataset, batch_size=5)
        val_loader = DataLoader(dataset, batch_size=5)
        loop.run_training(train_loader, val_loader, num_epochs=1)

        # 51: Checkpoints created
        ckpt_files = list(ckpt_dir.glob("checkpoint_step_*.pt"))
        check(
            len(ckpt_files) >= 1,
            "Checkpoint saving: at least 1 step checkpoint created",
            f"found {len(ckpt_files)} files",
        )

        # 52: Checkpoint is loadable
        if ckpt_files:
            ckpt = torch.load(ckpt_files[0], map_location="cpu", weights_only=False)
            check(
                "model_state_dict" in ckpt,
                "Checkpoint saving: contains model_state_dict",
            )
            check(
                "optimizer_state_dict" in ckpt,
                "Checkpoint saving: contains optimizer_state_dict",
            )
            check("epoch" in ckpt, "Checkpoint saving: contains epoch")
            check("global_step" in ckpt, "Checkpoint saving: contains global_step")
        else:
            _fail("Checkpoint saving: no files to load")
            failed += 1

    # ==================================================================
    # Validation metrics computation
    # ==================================================================
    _banner("Validation metrics")

    model = MockModel(input_dim=100, hidden_dim=32, output_dim=5)
    loop = TrainingLoop(
        model=model,
        optimizer=optim.Adam(model.parameters(), lr=1e-3),
        config={"use_amp": False, "grad_clip_norm": 0.0, "log_every_n_steps": 999,
                "save_every_n_steps": 999999, "early_stopping_patience": 100},
        device="cpu",
    )
    val_ds = TensorDataset(torch.randn(50, 100), torch.randint(0, 5, (50,)))
    val_dl = DataLoader(val_ds, batch_size=10)
    val_metrics = loop.validate(val_dl)

    # 53: val_loss present
    check("val_loss" in val_metrics, "Validation: val_loss in metrics")

    # 54: val_accuracy present
    check("val_accuracy" in val_metrics, "Validation: val_accuracy in metrics")

    # 55: val_loss is finite
    check(np.isfinite(val_metrics["val_loss"]), "Validation: val_loss is finite")

    # 56: val_accuracy in [0, 1]
    check(
        0.0 <= val_metrics["val_accuracy"] <= 1.0,
        "Validation: val_accuracy in [0,1]",
        f"got {val_metrics['val_accuracy']}",
    )

    # ==================================================================
    # Gradient clipping
    # ==================================================================
    _banner("Gradient clipping")

    model_gc = MockModel(input_dim=100, hidden_dim=32, output_dim=5)
    optimizer_gc = optim.Adam(model_gc.parameters(), lr=1e-3)
    loop_gc = TrainingLoop(
        model=model_gc,
        optimizer=optimizer_gc,
        config={
            "use_amp": False,
            "grad_clip_norm": 0.01,
            "log_every_n_steps": 999,
            "save_every_n_steps": 999999,
            "early_stopping_patience": 100,
        },
        device="cpu",
    )

    # Create a batch with large values to produce large gradients
    big_x = torch.randn(8, 100) * 100.0
    big_y = torch.randint(0, 5, (8,))
    step_metrics = loop_gc.train_step((big_x, big_y))

    # 57: Training step completes with grad clipping
    check(np.isfinite(step_metrics["loss"]), "Grad clipping: loss is finite after clip")

    # 58: Gradients are clipped (check norm is finite after step)
    total_norm = 0.0
    for p in model_gc.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    check(
        np.isfinite(total_norm),
        "Grad clipping: gradient norm is finite after step",
    )

    # ==================================================================
    # MockModel tests
    # ==================================================================
    _banner("MockModel")

    # 59: Basic construction
    mm = MockModel(input_dim=256, hidden_dim=64, output_dim=20)
    check(isinstance(mm, nn.Module), "MockModel: is nn.Module")

    # 60: Parameter count
    param_count = sum(p.numel() for p in mm.parameters())
    check(param_count > 0, "MockModel: has parameters")

    # 61: Forward without targets
    out_no_tgt = mm(torch.randn(4, 256))
    check("logits" in out_no_tgt, "MockModel: forward without targets has logits")
    check("loss" not in out_no_tgt, "MockModel: forward without targets has no loss")

    # 62: Forward with targets
    out_tgt = mm(torch.randn(4, 256), torch.randint(0, 20, (4,)))
    check("loss" in out_tgt, "MockModel: forward with targets has loss")
    check(
        out_tgt["logits"].shape == (4, 20),
        "MockModel: logits shape matches (batch, num_classes)",
    )

    # 63: state_dict save/load round-trip
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "mock_model.pt"
        torch.save(mm.state_dict(), path)
        mm2 = MockModel(input_dim=256, hidden_dim=64, output_dim=20)
        mm2.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        x_test = torch.randn(2, 256)
        check(
            torch.equal(mm(x_test)["logits"], mm2(x_test)["logits"]),
            "MockModel: state_dict save/load produces identical output",
        )

    # ==================================================================
    # All 7 phases instantiate in dev mode
    # ==================================================================
    _banner("All 7 phases -- dev mode instantiation")

    with tempfile.TemporaryDirectory() as tmpdir:
        for phase in range(1, 8):
            try:
                run_dir = Path(tmpdir) / f"phase_{phase}"
                runner = PhaseRunner(
                    phase=phase, run_dir=run_dir, mode="dev", seed=42
                )
                runner.setup()
                model = runner.build_model()
                optimizer, scheduler = runner.build_optimizer(model)
                train_dl = runner.build_dataloader(split="train")
                val_dl = runner.build_dataloader(split="val")

                # Run a single training step manually
                runner._training_loop = TrainingLoop(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    logger=runner.logger,
                    config=runner.config,
                    device=runner.device,
                    checkpoint_dir=run_dir / "checkpoints",
                )
                train_metrics = runner.train_epoch(model, optimizer, train_dl, epoch=1)
                val_metrics = runner.validate(model, val_dl)

                check(
                    np.isfinite(train_metrics.get("loss", float("nan"))),
                    f"Phase {phase} dev: train epoch produces finite loss",
                )
                check(
                    np.isfinite(val_metrics.get("val_loss", float("nan"))),
                    f"Phase {phase} dev: validation produces finite loss",
                )
                runner.finalize()
            except Exception as e:
                _fail(f"Phase {phase} dev: instantiation", str(e))
                failed += 1

    # ==================================================================
    # PhaseRunner.run() full pipeline (phase 1 only, dev mode)
    # ==================================================================
    _banner("PhaseRunner.run() -- full pipeline (phase 1 dev)")

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "full_run"
        runner = PhaseRunner(phase=1, run_dir=run_dir, mode="dev", seed=42)
        results = runner.run()

        # 64: Results returned
        check(results is not None, "PhaseRunner.run(): returns results")

        # 65: History present
        check("history" in results, "PhaseRunner.run(): history in results")

        # 66: Phase boundary saved
        boundary = run_dir / "phase1_boundary.pt"
        check(boundary.exists(), "PhaseRunner.run(): phase boundary checkpoint saved")

        # 67: Manifest finalized with end_time
        with open(run_dir / "manifest.json") as f:
            final_manifest = json.load(f)
        check(
            "end_time" in final_manifest,
            "PhaseRunner.run(): manifest has end_time",
        )
        check(
            "results" in final_manifest,
            "PhaseRunner.run(): manifest has results",
        )

    # ==================================================================
    # PhaseRunner -- invalid phase
    # ==================================================================
    _banner("PhaseRunner -- edge cases")

    # 68: Invalid phase raises
    raised = False
    try:
        PhaseRunner(phase=0)
    except ValueError:
        raised = True
    check(raised, "PhaseRunner: phase=0 raises ValueError")

    raised = False
    try:
        PhaseRunner(phase=8)
    except ValueError:
        raised = True
    check(raised, "PhaseRunner: phase=8 raises ValueError")

    # ==================================================================
    # TrainingLoop -- early stopping
    # ==================================================================
    _banner("TrainingLoop -- early stopping")

    model_es = MockModel(input_dim=50, hidden_dim=16, output_dim=3)
    loop_es = TrainingLoop(
        model=model_es,
        optimizer=optim.Adam(model_es.parameters(), lr=1e-3),
        config={
            "use_amp": False,
            "grad_clip_norm": 1.0,
            "log_every_n_steps": 999,
            "save_every_n_steps": 999999,
            "early_stopping_patience": 2,
        },
        device="cpu",
    )

    # 69: Early stopping triggers
    loop_es.best_val_loss = 0.001  # Set an artificially low best
    stopped = loop_es._check_early_stopping(0.5)
    check(not stopped, "Early stopping: first bad epoch, patience not exhausted")
    stopped = loop_es._check_early_stopping(0.5)
    check(stopped, "Early stopping: patience=2 exhausted, triggers stop")

    # 70: Improvement resets patience
    loop_es.patience_counter = 0
    loop_es.best_val_loss = 1.0
    stopped = loop_es._check_early_stopping(0.5)
    check(not stopped, "Early stopping: improvement resets counter")
    check(loop_es.patience_counter == 0, "Early stopping: patience_counter reset to 0")
    check(
        loop_es.best_val_loss == 0.5,
        "Early stopping: best_val_loss updated",
    )

    # ==================================================================
    # _LimitedDataLoader
    # ==================================================================
    _banner("_LimitedDataLoader")

    full_dl = DataLoader(
        TensorDataset(torch.randn(200, 10), torch.randint(0, 5, (200,))),
        batch_size=10,
    )

    # 71: Limits correctly
    limited = _LimitedDataLoader(full_dl, max_batches=5)
    count = sum(1 for _ in limited)
    check(count == 5, "_LimitedDataLoader: yields exactly max_batches")

    # 72: len() is correct
    check(len(limited) == 5, "_LimitedDataLoader: __len__ returns max_batches")

    # 73: dataset property
    check(
        limited.dataset is full_dl.dataset,
        "_LimitedDataLoader: dataset property returns original dataset",
    )

    # ==================================================================
    # PhaseRunner -- build_optimizer variations
    # ==================================================================
    _banner("PhaseRunner -- build_optimizer variations")

    with tempfile.TemporaryDirectory() as tmpdir:
        # 74: AdamW optimizer
        runner = PhaseRunner(phase=4, run_dir=Path(tmpdir) / "adamw", mode="dev")
        runner.setup()
        model = runner.build_model()
        opt, sched = runner.build_optimizer(model)
        check(
            isinstance(opt, optim.AdamW),
            "Phase 4 build_optimizer: creates AdamW",
        )
        check(sched is not None, "Phase 4 build_optimizer: creates scheduler")
        runner.finalize()

        # 75: Adam optimizer for phase 1
        runner1 = PhaseRunner(phase=1, run_dir=Path(tmpdir) / "adam", mode="dev")
        runner1.setup()
        model1 = runner1.build_model()
        opt1, _ = runner1.build_optimizer(model1)
        check(isinstance(opt1, optim.Adam), "Phase 1 build_optimizer: creates Adam")
        runner1.finalize()

    # ==================================================================
    # Summary
    # ==================================================================
    _banner("SUMMARY")

    total = passed + failed
    print(f"\n  Total : {total}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")

    if failure_details:
        print("\n  Failed tests:")
        for name in failure_details:
            print(f"    - {name}")

    if failed > 0:
        print(f"\n  EXIT CODE: 1 ({failed} failure(s))")
        sys.exit(1)
    else:
        print("\n  All tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    _run_tests()
