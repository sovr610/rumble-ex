#!/usr/bin/env python3
"""
training_config_template.py -- Unified Training Configuration for Brain-AI Pipeline

Comprehensive configuration dataclasses covering the seven-phase brain-inspired
AI training pipeline.  All config surfaces are unified here: training control,
manifest capture, seeding, checkpointing, logging, ablation, per-phase hyper-
parameters, and presets.

Classes:
    TrainingConfig        -- Top-level training control (mode, phases, AMP, etc.)
    ManifestConfig        -- Run provenance capture settings
    SeedConfig            -- Determinism and seeding parameters
    CheckpointConfig      -- Checkpoint retention and saving policy
    LoggingConfig         -- TensorBoard, W&B, JSONL logging settings
    AblationConfig        -- Ablation study configuration
    PhaseSpecificConfig   -- Per-phase hyperparameters with sensible defaults
    TrainingFullConfig    -- Aggregates all sub-configs with serialization
    ConfigResolver        -- Replaces "auto" sentinel values with concrete values

Self-contained: no brain_ai imports required.  Uses json for serialization.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import shutil
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------

VALID_MODES = ("dev", "production")
VALID_PHASES = tuple(range(1, 8))


@dataclass
class TrainingConfig:
    """Top-level training control.

    Attributes
    ----------
    mode : str
        ``"dev"`` (MNIST, fast iteration) or ``"production"`` (full datasets, AMP).
    phases : list[int]
        Which phases to include in the pipeline run.
    start_phase : int
        First phase to execute (inclusive).
    end_phase : int
        Last phase to execute (inclusive).
    run_dir : str
        Base directory for storing run outputs.
    use_amp : bool
        Enable automatic mixed precision training.
    gradient_clip_norm : float
        Maximum gradient norm for clipping.
    early_stopping_patience : int
        Number of evaluations without improvement before stopping.
    eval_every_n_steps : int
        Run validation every N training steps.
    """

    mode: str = "dev"
    phases: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7])
    start_phase: int = 1
    end_phase: int = 7
    run_dir: str = "runs/"
    use_amp: bool = False
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 10
    eval_every_n_steps: int = 500

    def validate(self) -> List[str]:
        """Validate configuration values.  Returns list of error strings."""
        errors: List[str] = []
        if self.mode not in VALID_MODES:
            errors.append(
                f"TrainingConfig.mode must be one of {VALID_MODES}, got {self.mode!r}"
            )
        if self.start_phase > self.end_phase:
            errors.append(
                f"TrainingConfig.start_phase ({self.start_phase}) must be "
                f"<= end_phase ({self.end_phase})"
            )
        for p in self.phases:
            if p < 1 or p > 7:
                errors.append(
                    f"TrainingConfig.phases contains invalid phase {p}; must be 1-7"
                )
        if self.start_phase < 1 or self.start_phase > 7:
            errors.append(
                f"TrainingConfig.start_phase must be 1-7, got {self.start_phase}"
            )
        if self.end_phase < 1 or self.end_phase > 7:
            errors.append(
                f"TrainingConfig.end_phase must be 1-7, got {self.end_phase}"
            )
        if self.gradient_clip_norm <= 0:
            errors.append(
                f"TrainingConfig.gradient_clip_norm must be > 0, got {self.gradient_clip_norm}"
            )
        if self.early_stopping_patience < 1:
            errors.append(
                f"TrainingConfig.early_stopping_patience must be >= 1, "
                f"got {self.early_stopping_patience}"
            )
        if self.eval_every_n_steps < 1:
            errors.append(
                f"TrainingConfig.eval_every_n_steps must be >= 1, "
                f"got {self.eval_every_n_steps}"
            )
        return errors


# ---------------------------------------------------------------------------
# ManifestConfig
# ---------------------------------------------------------------------------

VALID_FINGERPRINT_TIERS = ("auto", "tier1", "tier2", "tier3")


@dataclass
class ManifestConfig:
    """Run provenance capture settings.

    Attributes
    ----------
    capture_git_diff : bool
        Save ``git diff HEAD`` as a patch file.
    capture_pip_freeze : bool
        Save ``pip freeze`` output.
    capture_hardware : bool
        Save hardware (GPU, CPU, RAM) snapshot.
    dataset_fingerprint_tier : str
        ``"auto"``, ``"tier1"`` (fast hash), ``"tier2"`` (sample hash),
        ``"tier3"`` (full content hash).
    schema_version : str
        Manifest schema version string.
    """

    capture_git_diff: bool = True
    capture_pip_freeze: bool = True
    capture_hardware: bool = True
    dataset_fingerprint_tier: str = "auto"
    schema_version: str = "1.0"

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.dataset_fingerprint_tier not in VALID_FINGERPRINT_TIERS:
            errors.append(
                f"ManifestConfig.dataset_fingerprint_tier must be one of "
                f"{VALID_FINGERPRINT_TIERS}, got {self.dataset_fingerprint_tier!r}"
            )
        return errors


# ---------------------------------------------------------------------------
# SeedConfig
# ---------------------------------------------------------------------------

@dataclass
class SeedConfig:
    """Determinism and seeding parameters.

    Attributes
    ----------
    base_seed : int
        Root seed for the entire training run.
    per_phase_offsets : list[int]
        Seven offsets (one per phase), added to base_seed for per-phase seeds.
    enforce_deterministic : bool
        Strict determinism (True in dev, typically False in production).
    cudnn_benchmark : bool
        Enable cuDNN auto-tuner (faster but nondeterministic).
    cudnn_deterministic : bool
        Force cuDNN to use deterministic algorithms.
    use_deterministic_algorithms : bool
        Call ``torch.use_deterministic_algorithms(True)``.
    """

    base_seed: int = 1337
    per_phase_offsets: List[int] = field(
        default_factory=lambda: [0, 100, 200, 300, 400, 500, 600]
    )
    enforce_deterministic: bool = True
    cudnn_benchmark: bool = False
    cudnn_deterministic: bool = True
    use_deterministic_algorithms: bool = True

    def validate(self) -> List[str]:
        errors: List[str] = []
        if len(self.per_phase_offsets) != 7:
            errors.append(
                f"SeedConfig.per_phase_offsets must have exactly 7 entries, "
                f"got {len(self.per_phase_offsets)}"
            )
        if self.base_seed < 0:
            errors.append(
                f"SeedConfig.base_seed must be >= 0, got {self.base_seed}"
            )
        if self.use_deterministic_algorithms and self.cudnn_benchmark:
            errors.append(
                "SeedConfig: use_deterministic_algorithms=True requires "
                "cudnn_benchmark=False"
            )
        return errors


# ---------------------------------------------------------------------------
# CheckpointConfig
# ---------------------------------------------------------------------------

VALID_METRIC_MODES = ("min", "max")
VALID_RETENTION_POLICIES = ("keep_all", "keep_best_and_boundary", "keep_last_n")


@dataclass
class CheckpointConfig:
    """Checkpoint retention and saving policy.

    Attributes
    ----------
    save_every_n_steps : int
        Save a checkpoint every N training steps.
    keep_best : bool
        Keep the best checkpoint by ``best_metric_key``.
    best_metric_key : str
        Metric key for determining "best" checkpoint.
    best_metric_mode : str
        ``"min"`` (lower is better) or ``"max"`` (higher is better).
    keep_last_n : int
        Number of most recent checkpoints to retain.
    save_phase_boundary : bool
        Save a checkpoint at the end of each phase.
    retention_policy : str
        One of ``"keep_all"``, ``"keep_best_and_boundary"``, ``"keep_last_n"``.
    """

    save_every_n_steps: int = 1000
    keep_best: bool = True
    best_metric_key: str = "val_loss"
    best_metric_mode: str = "min"
    keep_last_n: int = 3
    save_phase_boundary: bool = True
    retention_policy: str = "keep_best_and_boundary"

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.best_metric_mode not in VALID_METRIC_MODES:
            errors.append(
                f"CheckpointConfig.best_metric_mode must be one of "
                f"{VALID_METRIC_MODES}, got {self.best_metric_mode!r}"
            )
        if self.retention_policy not in VALID_RETENTION_POLICIES:
            errors.append(
                f"CheckpointConfig.retention_policy must be one of "
                f"{VALID_RETENTION_POLICIES}, got {self.retention_policy!r}"
            )
        if self.save_every_n_steps < 1:
            errors.append(
                f"CheckpointConfig.save_every_n_steps must be >= 1, "
                f"got {self.save_every_n_steps}"
            )
        if self.keep_last_n < 1:
            errors.append(
                f"CheckpointConfig.keep_last_n must be >= 1, got {self.keep_last_n}"
            )
        return errors


# ---------------------------------------------------------------------------
# LoggingConfig
# ---------------------------------------------------------------------------

VALID_WANDB_RESUME_MODES = ("must", "allow", "never")


@dataclass
class LoggingConfig:
    """TensorBoard, Weights & Biases, and JSONL logging settings.

    Attributes
    ----------
    tensorboard_enabled : bool
        Enable TensorBoard logging.
    wandb_enabled : bool
        Enable Weights & Biases logging.
    wandb_project : str
        W&B project name.
    wandb_entity : str or None
        W&B entity (team or user).
    wandb_resume_mode : str
        ``"must"``, ``"allow"``, or ``"never"``.
    jsonl_enabled : bool
        Enable JSONL metric logging.
    log_histograms_every : int
        Log weight/gradient histograms every N steps.
    log_images_every : int
        Log sample images every N steps.
    log_system_every : int
        Log system metrics (GPU util, memory) every N steps.
    flush_every : int
        Flush log buffers every N steps.
    """

    tensorboard_enabled: bool = True
    wandb_enabled: bool = False
    wandb_project: str = "brain_ai"
    wandb_entity: Optional[str] = None
    wandb_resume_mode: str = "never"
    jsonl_enabled: bool = True
    log_histograms_every: int = 500
    log_images_every: int = 1000
    log_system_every: int = 100
    flush_every: int = 100

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.wandb_resume_mode not in VALID_WANDB_RESUME_MODES:
            errors.append(
                f"LoggingConfig.wandb_resume_mode must be one of "
                f"{VALID_WANDB_RESUME_MODES}, got {self.wandb_resume_mode!r}"
            )
        if self.log_histograms_every < 1:
            errors.append(
                f"LoggingConfig.log_histograms_every must be >= 1, "
                f"got {self.log_histograms_every}"
            )
        if self.log_images_every < 1:
            errors.append(
                f"LoggingConfig.log_images_every must be >= 1, "
                f"got {self.log_images_every}"
            )
        return errors


# ---------------------------------------------------------------------------
# AblationConfig
# ---------------------------------------------------------------------------

VALID_ABLATION_MODES = ("full", "pairwise")


@dataclass
class AblationConfig:
    """Ablation study configuration.

    Attributes
    ----------
    spec_file : str or None
        Path to ablation spec YAML/JSON file.
    mode : str
        ``"full"`` (all combinations) or ``"pairwise"`` (pairwise toggling).
    parallel : bool
        Run ablation variants in parallel.
    max_workers : int
        Maximum parallel workers.
    phases : list[int]
        Which phases to run ablation on.
    seeds : list[int]
        Seeds to sweep over for each ablation variant.
    baseline_run_id : str or None
        Run ID of baseline to compare against.
    """

    spec_file: Optional[str] = None
    mode: str = "full"
    parallel: bool = False
    max_workers: int = 4
    phases: List[int] = field(default_factory=lambda: [4])
    seeds: List[int] = field(default_factory=lambda: [1337])
    baseline_run_id: Optional[str] = None

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.mode not in VALID_ABLATION_MODES:
            errors.append(
                f"AblationConfig.mode must be one of {VALID_ABLATION_MODES}, "
                f"got {self.mode!r}"
            )
        if self.max_workers < 1:
            errors.append(
                f"AblationConfig.max_workers must be >= 1, got {self.max_workers}"
            )
        for p in self.phases:
            if p < 1 or p > 7:
                errors.append(
                    f"AblationConfig.phases contains invalid phase {p}; must be 1-7"
                )
        return errors


# ---------------------------------------------------------------------------
# PhaseSpecificConfig
# ---------------------------------------------------------------------------

@dataclass
class PhaseSpecificConfig:
    """Per-phase training hyperparameters.

    Attributes
    ----------
    lr : float
        Learning rate.
    epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size.
    optimizer : str
        Optimizer name (e.g. ``"adamw"``, ``"sgd"``).
    warmup_steps : int
        Number of warmup steps for the learning rate scheduler.
    weight_decay : float
        L2 weight decay coefficient.
    scheduler : str
        LR scheduler type: ``"cosine"``, ``"linear"``, ``"step"``, ``"none"``.
    dataset : str
        Dataset name, or ``"auto"`` for mode-dependent resolution.
    """

    lr: float = 1e-3
    epochs: int = 10
    batch_size: int = 32
    optimizer: str = "adamw"
    warmup_steps: int = 0
    weight_decay: float = 0.01
    scheduler: str = "cosine"
    dataset: str = "auto"

    VALID_SCHEDULERS: ClassVar[Tuple[str, ...]] = ("cosine", "linear", "step", "none")
    VALID_OPTIMIZERS: ClassVar[Tuple[str, ...]] = ("adamw", "adam", "sgd", "rmsprop")

    # Sensible defaults per phase -- keys are phase numbers 1-7.
    PHASE_DEFAULTS: ClassVar[Dict[int, Dict[str, Any]]] = {
        1: {  # SNN Core
            "lr": 1e-3, "epochs": 20, "batch_size": 64, "optimizer": "adamw",
            "warmup_steps": 500, "weight_decay": 0.01, "scheduler": "cosine",
            "dataset": "auto",
        },
        2: {  # Modality Encoders
            "lr": 5e-4, "epochs": 15, "batch_size": 64, "optimizer": "adamw",
            "warmup_steps": 300, "weight_decay": 0.01, "scheduler": "cosine",
            "dataset": "auto",
        },
        3: {  # HTM Sequence Learning
            "lr": 1e-3, "epochs": 25, "batch_size": 32, "optimizer": "adam",
            "warmup_steps": 200, "weight_decay": 0.005, "scheduler": "cosine",
            "dataset": "auto",
        },
        4: {  # Global Workspace
            "lr": 3e-4, "epochs": 30, "batch_size": 32, "optimizer": "adamw",
            "warmup_steps": 1000, "weight_decay": 0.01, "scheduler": "cosine",
            "dataset": "auto",
        },
        5: {  # Active Inference
            "lr": 1e-4, "epochs": 20, "batch_size": 16, "optimizer": "adamw",
            "warmup_steps": 500, "weight_decay": 0.01, "scheduler": "linear",
            "dataset": "auto",
        },
        6: {  # Dual-Process Reasoning
            "lr": 5e-5, "epochs": 15, "batch_size": 16, "optimizer": "adamw",
            "warmup_steps": 300, "weight_decay": 0.005, "scheduler": "cosine",
            "dataset": "auto",
        },
        7: {  # Meta-Learning
            "lr": 1e-4, "epochs": 10, "batch_size": 8, "optimizer": "adam",
            "warmup_steps": 100, "weight_decay": 0.0, "scheduler": "step",
            "dataset": "auto",
        },
    }

    @classmethod
    def for_phase(cls, phase: int, **overrides: Any) -> "PhaseSpecificConfig":
        """Create a PhaseSpecificConfig with defaults for a given phase.

        Parameters
        ----------
        phase : int
            Training phase 1-7.
        **overrides
            Keyword arguments to override defaults.
        """
        if phase not in cls.PHASE_DEFAULTS:
            raise ValueError(f"phase must be 1-7, got {phase}")
        defaults = dict(cls.PHASE_DEFAULTS[phase])
        defaults.update(overrides)
        return cls(**defaults)

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.lr <= 0:
            errors.append(f"PhaseSpecificConfig.lr must be > 0, got {self.lr}")
        if self.epochs < 1:
            errors.append(f"PhaseSpecificConfig.epochs must be >= 1, got {self.epochs}")
        if self.batch_size < 1:
            errors.append(
                f"PhaseSpecificConfig.batch_size must be >= 1, got {self.batch_size}"
            )
        if self.scheduler not in self.VALID_SCHEDULERS:
            errors.append(
                f"PhaseSpecificConfig.scheduler must be one of "
                f"{self.VALID_SCHEDULERS}, got {self.scheduler!r}"
            )
        if self.optimizer not in self.VALID_OPTIMIZERS:
            errors.append(
                f"PhaseSpecificConfig.optimizer must be one of "
                f"{self.VALID_OPTIMIZERS}, got {self.optimizer!r}"
            )
        if self.warmup_steps < 0:
            errors.append(
                f"PhaseSpecificConfig.warmup_steps must be >= 0, got {self.warmup_steps}"
            )
        if self.weight_decay < 0:
            errors.append(
                f"PhaseSpecificConfig.weight_decay must be >= 0, got {self.weight_decay}"
            )
        return errors


# ---------------------------------------------------------------------------
# TrainingFullConfig
# ---------------------------------------------------------------------------

@dataclass
class TrainingFullConfig:
    """Aggregates all sub-configs into a single unified configuration.

    Provides serialization (to_dict / from_dict), file I/O (save_json /
    load_json), override merging, validation, and factory presets.
    """

    training: TrainingConfig = field(default_factory=TrainingConfig)
    manifest: ManifestConfig = field(default_factory=ManifestConfig)
    seed: SeedConfig = field(default_factory=SeedConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    phase_configs: Dict[int, PhaseSpecificConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Populate phase_configs with defaults for any missing phases.
        for phase in range(1, 8):
            if phase not in self.phase_configs:
                self.phase_configs[phase] = PhaseSpecificConfig.for_phase(phase)

    # -- validation ---------------------------------------------------------

    def validate(self) -> List[str]:
        """Validate all sub-configs and cross-field constraints.

        Returns list of error strings.  Empty means valid.
        """
        errors: List[str] = []
        errors.extend(self.training.validate())
        errors.extend(self.manifest.validate())
        errors.extend(self.seed.validate())
        errors.extend(self.checkpoint.validate())
        errors.extend(self.logging.validate())
        errors.extend(self.ablation.validate())
        for phase, pcfg in self.phase_configs.items():
            phase_errors = pcfg.validate()
            for e in phase_errors:
                errors.append(f"phase_configs[{phase}]: {e}")
        # Cross-field: AMP in dev mode is a warning (not error), but AMP
        # without CUDA availability should be flagged.
        if self.training.use_amp:
            try:
                import torch
                if not torch.cuda.is_available():
                    errors.append(
                        "TrainingConfig.use_amp=True but CUDA is not available"
                    )
            except ImportError:
                errors.append(
                    "TrainingConfig.use_amp=True but torch is not installed"
                )
        return errors

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict suitable for JSON."""
        d: Dict[str, Any] = {
            "training": asdict(self.training),
            "manifest": asdict(self.manifest),
            "seed": asdict(self.seed),
            "checkpoint": asdict(self.checkpoint),
            "logging": asdict(self.logging),
            "ablation": asdict(self.ablation),
            "phase_configs": {
                str(k): asdict(v) for k, v in self.phase_configs.items()
            },
        }
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingFullConfig":
        """Reconstruct from a dict produced by ``to_dict``."""
        training = TrainingConfig(**d.get("training", {}))
        manifest = ManifestConfig(**d.get("manifest", {}))
        seed = SeedConfig(**d.get("seed", {}))
        checkpoint = CheckpointConfig(**d.get("checkpoint", {}))
        logging_cfg = LoggingConfig(**d.get("logging", {}))
        ablation = AblationConfig(**d.get("ablation", {}))
        phase_configs: Dict[int, PhaseSpecificConfig] = {}
        for k, v in d.get("phase_configs", {}).items():
            phase_configs[int(k)] = PhaseSpecificConfig(**v)
        obj = cls.__new__(cls)
        obj.training = training
        obj.manifest = manifest
        obj.seed = seed
        obj.checkpoint = checkpoint
        obj.logging = logging_cfg
        obj.ablation = ablation
        obj.phase_configs = phase_configs
        # Ensure all 7 phases are present.
        for phase in range(1, 8):
            if phase not in obj.phase_configs:
                obj.phase_configs[phase] = PhaseSpecificConfig.for_phase(phase)
        return obj

    # -- file I/O -----------------------------------------------------------

    def save_json(self, path: Union[str, Path]) -> Path:
        """Write config to a JSON file.  Returns the path."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return p

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> "TrainingFullConfig":
        """Load config from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_dict(d)

    # -- override merging ---------------------------------------------------

    def merge_overrides(self, overrides: Dict[str, Any]) -> None:
        """Apply CLI / ablation overrides to the resolved config.

        Supports dotted keys for nested access, e.g.:
            ``{"training.use_amp": True, "seed.base_seed": 42,
              "phase_configs.4.lr": 1e-4}``
        Also supports flat keys for sub-config direct assignment, e.g.:
            ``{"mode": "production"}`` sets ``training.mode``.
        """
        for key, value in overrides.items():
            parts = key.split(".")
            if len(parts) == 1:
                # Try to find the field on TrainingConfig first (common CLI args)
                if hasattr(self.training, parts[0]):
                    setattr(self.training, parts[0], value)
                elif hasattr(self.seed, parts[0]):
                    setattr(self.seed, parts[0], value)
                elif hasattr(self.checkpoint, parts[0]):
                    setattr(self.checkpoint, parts[0], value)
                elif hasattr(self.logging, parts[0]):
                    setattr(self.logging, parts[0], value)
                elif hasattr(self.ablation, parts[0]):
                    setattr(self.ablation, parts[0], value)
                elif hasattr(self.manifest, parts[0]):
                    setattr(self.manifest, parts[0], value)
            elif len(parts) == 2:
                section, attr = parts
                sub = self._get_section(section)
                if sub is not None and hasattr(sub, attr):
                    setattr(sub, attr, value)
            elif len(parts) == 3 and parts[0] == "phase_configs":
                phase = int(parts[1])
                attr = parts[2]
                if phase in self.phase_configs and hasattr(self.phase_configs[phase], attr):
                    setattr(self.phase_configs[phase], attr, value)

    def _get_section(self, name: str) -> Any:
        """Return sub-config object by section name."""
        mapping = {
            "training": self.training,
            "manifest": self.manifest,
            "seed": self.seed,
            "checkpoint": self.checkpoint,
            "logging": self.logging,
            "ablation": self.ablation,
        }
        return mapping.get(name)

    # -- presets ------------------------------------------------------------

    @classmethod
    def minimal(cls) -> "TrainingFullConfig":
        """Minimal config for unit tests: 2 epochs, CPU, no logging, seed=42."""
        cfg = cls(
            training=TrainingConfig(
                mode="dev",
                phases=[1, 2, 3, 4, 5, 6, 7],
                start_phase=1,
                end_phase=7,
                run_dir="runs/minimal/",
                use_amp=False,
                gradient_clip_norm=1.0,
                early_stopping_patience=3,
                eval_every_n_steps=50,
            ),
            manifest=ManifestConfig(
                capture_git_diff=False,
                capture_pip_freeze=False,
                capture_hardware=False,
                dataset_fingerprint_tier="tier1",
                schema_version="1.0",
            ),
            seed=SeedConfig(
                base_seed=42,
                per_phase_offsets=[0, 10, 20, 30, 40, 50, 60],
                enforce_deterministic=True,
                cudnn_benchmark=False,
                cudnn_deterministic=True,
                use_deterministic_algorithms=True,
            ),
            checkpoint=CheckpointConfig(
                save_every_n_steps=100,
                keep_best=True,
                best_metric_key="val_loss",
                best_metric_mode="min",
                keep_last_n=1,
                save_phase_boundary=True,
                retention_policy="keep_best_and_boundary",
            ),
            logging=LoggingConfig(
                tensorboard_enabled=False,
                wandb_enabled=False,
                wandb_project="brain_ai_test",
                wandb_entity=None,
                wandb_resume_mode="never",
                jsonl_enabled=False,
                log_histograms_every=100,
                log_images_every=100,
                log_system_every=50,
                flush_every=50,
            ),
            ablation=AblationConfig(),
        )
        # Override all phases to 2 epochs with small batch size
        for phase in range(1, 8):
            cfg.phase_configs[phase] = PhaseSpecificConfig.for_phase(
                phase, epochs=2, batch_size=8
            )
        return cfg

    @classmethod
    def dev(cls) -> "TrainingFullConfig":
        """Dev mode: small datasets, TensorBoard logging, deterministic."""
        cfg = cls(
            training=TrainingConfig(
                mode="dev",
                phases=[1, 2, 3, 4, 5, 6, 7],
                start_phase=1,
                end_phase=7,
                run_dir="runs/dev/",
                use_amp=False,
                gradient_clip_norm=1.0,
                early_stopping_patience=10,
                eval_every_n_steps=500,
            ),
            manifest=ManifestConfig(
                capture_git_diff=True,
                capture_pip_freeze=True,
                capture_hardware=True,
                dataset_fingerprint_tier="auto",
                schema_version="1.0",
            ),
            seed=SeedConfig(
                base_seed=1337,
                per_phase_offsets=[0, 100, 200, 300, 400, 500, 600],
                enforce_deterministic=True,
                cudnn_benchmark=False,
                cudnn_deterministic=True,
                use_deterministic_algorithms=True,
            ),
            checkpoint=CheckpointConfig(
                save_every_n_steps=1000,
                keep_best=True,
                best_metric_key="val_loss",
                best_metric_mode="min",
                keep_last_n=3,
                save_phase_boundary=True,
                retention_policy="keep_best_and_boundary",
            ),
            logging=LoggingConfig(
                tensorboard_enabled=True,
                wandb_enabled=False,
                wandb_project="brain_ai",
                wandb_entity=None,
                wandb_resume_mode="never",
                jsonl_enabled=True,
                log_histograms_every=500,
                log_images_every=1000,
                log_system_every=100,
                flush_every=100,
            ),
            ablation=AblationConfig(),
        )
        return cfg

    @classmethod
    def production(cls) -> "TrainingFullConfig":
        """Production mode: full datasets, TB+W&B, AMP, relaxed determinism."""
        cfg = cls(
            training=TrainingConfig(
                mode="production",
                phases=[1, 2, 3, 4, 5, 6, 7],
                start_phase=1,
                end_phase=7,
                run_dir="runs/production/",
                use_amp=True,
                gradient_clip_norm=1.0,
                early_stopping_patience=15,
                eval_every_n_steps=1000,
            ),
            manifest=ManifestConfig(
                capture_git_diff=True,
                capture_pip_freeze=True,
                capture_hardware=True,
                dataset_fingerprint_tier="tier2",
                schema_version="1.0",
            ),
            seed=SeedConfig(
                base_seed=1337,
                per_phase_offsets=[0, 100, 200, 300, 400, 500, 600],
                enforce_deterministic=False,
                cudnn_benchmark=True,
                cudnn_deterministic=False,
                use_deterministic_algorithms=False,
            ),
            checkpoint=CheckpointConfig(
                save_every_n_steps=2000,
                keep_best=True,
                best_metric_key="val_loss",
                best_metric_mode="min",
                keep_last_n=5,
                save_phase_boundary=True,
                retention_policy="keep_best_and_boundary",
            ),
            logging=LoggingConfig(
                tensorboard_enabled=True,
                wandb_enabled=True,
                wandb_project="brain_ai",
                wandb_entity=None,
                wandb_resume_mode="allow",
                jsonl_enabled=True,
                log_histograms_every=1000,
                log_images_every=2000,
                log_system_every=200,
                flush_every=200,
            ),
            ablation=AblationConfig(),
        )
        # Production phases get more epochs and larger batches
        for phase in range(1, 8):
            defaults = dict(PhaseSpecificConfig.PHASE_DEFAULTS[phase])
            defaults["epochs"] = defaults["epochs"] * 3
            defaults["batch_size"] = defaults["batch_size"] * 4
            cfg.phase_configs[phase] = PhaseSpecificConfig(**defaults)
        return cfg

    @classmethod
    def ablation_dev(cls) -> "TrainingFullConfig":
        """Dev mode with ablation: small matrix for quick ablation sweeps."""
        cfg = cls.dev()
        cfg.ablation = AblationConfig(
            spec_file=None,
            mode="pairwise",
            parallel=False,
            max_workers=2,
            phases=[4],
            seeds=[1337, 42],
            baseline_run_id=None,
        )
        # Shorter epochs for ablation
        for phase in range(1, 8):
            cfg.phase_configs[phase] = PhaseSpecificConfig.for_phase(
                phase, epochs=5, batch_size=16
            )
        return cfg


# ---------------------------------------------------------------------------
# ConfigResolver
# ---------------------------------------------------------------------------

# Dataset mappings per phase and mode.
_DATASET_MAP: Dict[str, Dict[int, str]] = {
    "dev": {
        1: "mnist",
        2: "mnist",
        3: "sequential_mnist",
        4: "mnist",
        5: "cartpole",
        6: "babi",
        7: "omniglot",
    },
    "production": {
        1: "imagenet",
        2: "imagenet21k",
        3: "wikitext103",
        4: "multimodal_combined",
        5: "dm_control",
        6: "arc_challenge",
        7: "meta_dataset",
    },
}

# Fingerprint tier resolution based on mode.
_FINGERPRINT_TIER_MAP: Dict[str, str] = {
    "dev": "tier1",
    "production": "tier2",
}


class ConfigResolver:
    """Replace all ``"auto"`` sentinel values with concrete values.

    Given a ``TrainingFullConfig``, the resolver inspects the mode and phase
    to replace ``"auto"`` datasets, fingerprint tiers, and any other
    mode-dependent auto-resolved fields.

    The resolved config should contain no ``"auto"`` values and is suitable
    for writing to ``manifest.lock.json``.
    """

    @classmethod
    def resolve(cls, config: TrainingFullConfig) -> TrainingFullConfig:
        """Return a new config with all ``"auto"`` values resolved.

        Parameters
        ----------
        config : TrainingFullConfig
            Input config (not modified in place).

        Returns
        -------
        TrainingFullConfig
            A deep copy with all ``"auto"`` values replaced.
        """
        resolved = cls._deep_copy_config(config)
        mode = resolved.training.mode

        # Resolve dataset fingerprint tier
        if resolved.manifest.dataset_fingerprint_tier == "auto":
            resolved.manifest.dataset_fingerprint_tier = _FINGERPRINT_TIER_MAP.get(
                mode, "tier1"
            )

        # Resolve per-phase datasets
        dataset_map = _DATASET_MAP.get(mode, _DATASET_MAP["dev"])
        for phase, pcfg in resolved.phase_configs.items():
            if pcfg.dataset == "auto":
                pcfg.dataset = dataset_map.get(phase, "unknown")

        return resolved

    @staticmethod
    def _deep_copy_config(config: TrainingFullConfig) -> TrainingFullConfig:
        """Create a deep copy of the config via serialization round-trip."""
        d = config.to_dict()
        return TrainingFullConfig.from_dict(copy.deepcopy(d))

    @classmethod
    def has_auto_values(cls, config: TrainingFullConfig) -> bool:
        """Return True if the config still contains any ``"auto"`` values."""
        if config.manifest.dataset_fingerprint_tier == "auto":
            return True
        for pcfg in config.phase_configs.values():
            if pcfg.dataset == "auto":
                return True
        return False


# ===========================================================================
# Self-test block
# ===========================================================================

if __name__ == "__main__":
    _pass_count = 0
    _fail_count = 0
    _errors: List[str] = []

    def _assert(condition: bool, name: str, detail: str = "") -> None:
        global _pass_count, _fail_count
        if condition:
            _pass_count += 1
            print(f"  PASS  {name}")
        else:
            _fail_count += 1
            msg = f"  FAIL  {name}"
            if detail:
                msg += f"  -- {detail}"
            print(msg)
            _errors.append(msg)

    print("=" * 72)
    print("TrainingConfig template self-test")
    print("=" * 72)

    # -------------------------------------------------------------------
    # 1. TrainingConfig creation and defaults
    # -------------------------------------------------------------------
    print("\n--- TrainingConfig ---")

    tc = TrainingConfig()
    _assert(tc.mode == "dev", "TrainingConfig: default mode is dev")
    _assert(tc.phases == [1, 2, 3, 4, 5, 6, 7], "TrainingConfig: default phases 1-7")
    _assert(tc.start_phase == 1, "TrainingConfig: default start_phase is 1")
    _assert(tc.end_phase == 7, "TrainingConfig: default end_phase is 7")
    _assert(tc.use_amp is False, "TrainingConfig: default use_amp is False")
    _assert(tc.gradient_clip_norm == 1.0, "TrainingConfig: default gradient_clip_norm")
    _assert(tc.early_stopping_patience == 10, "TrainingConfig: default patience")
    _assert(tc.eval_every_n_steps == 500, "TrainingConfig: default eval_every")
    _assert(len(tc.validate()) == 0, "TrainingConfig: defaults validate cleanly")

    # 1b. Invalid mode
    tc_bad = TrainingConfig(mode="turbo")
    errs = tc_bad.validate()
    _assert(any("mode" in e for e in errs), "TrainingConfig: rejects mode=turbo")

    # 1c. Invalid phase range
    tc_bad2 = TrainingConfig(start_phase=5, end_phase=2)
    errs = tc_bad2.validate()
    _assert(
        any("start_phase" in e for e in errs),
        "TrainingConfig: rejects start_phase > end_phase",
    )

    # 1d. Invalid phase number
    tc_bad3 = TrainingConfig(phases=[0, 8])
    errs = tc_bad3.validate()
    _assert(
        any("invalid phase" in e.lower() for e in errs),
        "TrainingConfig: rejects phase 0 and 8",
    )

    # 1e. Invalid gradient_clip_norm
    tc_bad4 = TrainingConfig(gradient_clip_norm=-1.0)
    errs = tc_bad4.validate()
    _assert(
        any("gradient_clip_norm" in e for e in errs),
        "TrainingConfig: rejects negative gradient_clip_norm",
    )

    # -------------------------------------------------------------------
    # 2. ManifestConfig
    # -------------------------------------------------------------------
    print("\n--- ManifestConfig ---")

    mc = ManifestConfig()
    _assert(mc.capture_git_diff is True, "ManifestConfig: default capture_git_diff")
    _assert(mc.capture_pip_freeze is True, "ManifestConfig: default capture_pip_freeze")
    _assert(
        mc.dataset_fingerprint_tier == "auto",
        "ManifestConfig: default tier is auto",
    )
    _assert(mc.schema_version == "1.0", "ManifestConfig: default schema_version")
    _assert(len(mc.validate()) == 0, "ManifestConfig: defaults validate cleanly")

    mc_bad = ManifestConfig(dataset_fingerprint_tier="tier99")
    errs = mc_bad.validate()
    _assert(
        any("fingerprint_tier" in e for e in errs),
        "ManifestConfig: rejects invalid tier",
    )

    # -------------------------------------------------------------------
    # 3. SeedConfig
    # -------------------------------------------------------------------
    print("\n--- SeedConfig ---")

    sc = SeedConfig()
    _assert(sc.base_seed == 1337, "SeedConfig: default base_seed")
    _assert(
        len(sc.per_phase_offsets) == 7,
        "SeedConfig: default offsets have 7 entries",
    )
    _assert(sc.enforce_deterministic is True, "SeedConfig: default enforce_deterministic")
    _assert(sc.cudnn_benchmark is False, "SeedConfig: default cudnn_benchmark")
    _assert(len(sc.validate()) == 0, "SeedConfig: defaults validate cleanly")

    # 3b. Wrong offset count
    sc_bad = SeedConfig(per_phase_offsets=[0, 1, 2])
    errs = sc_bad.validate()
    _assert(
        any("7 entries" in e for e in errs),
        "SeedConfig: rejects 3-element offsets",
    )

    # 3c. Negative base_seed
    sc_bad2 = SeedConfig(base_seed=-1)
    errs = sc_bad2.validate()
    _assert(
        any("base_seed" in e for e in errs),
        "SeedConfig: rejects negative base_seed",
    )

    # 3d. Deterministic + benchmark conflict
    sc_bad3 = SeedConfig(use_deterministic_algorithms=True, cudnn_benchmark=True)
    errs = sc_bad3.validate()
    _assert(
        any("benchmark" in e for e in errs),
        "SeedConfig: rejects deterministic + benchmark",
    )

    # -------------------------------------------------------------------
    # 4. CheckpointConfig
    # -------------------------------------------------------------------
    print("\n--- CheckpointConfig ---")

    cc = CheckpointConfig()
    _assert(cc.save_every_n_steps == 1000, "CheckpointConfig: default save_every")
    _assert(cc.keep_best is True, "CheckpointConfig: default keep_best")
    _assert(cc.best_metric_key == "val_loss", "CheckpointConfig: default metric_key")
    _assert(cc.best_metric_mode == "min", "CheckpointConfig: default metric_mode")
    _assert(cc.keep_last_n == 3, "CheckpointConfig: default keep_last_n")
    _assert(cc.save_phase_boundary is True, "CheckpointConfig: default phase_boundary")
    _assert(
        cc.retention_policy == "keep_best_and_boundary",
        "CheckpointConfig: default retention_policy",
    )
    _assert(len(cc.validate()) == 0, "CheckpointConfig: defaults validate cleanly")

    cc_bad = CheckpointConfig(best_metric_mode="median")
    errs = cc_bad.validate()
    _assert(
        any("metric_mode" in e for e in errs),
        "CheckpointConfig: rejects mode=median",
    )

    cc_bad2 = CheckpointConfig(retention_policy="discard_all")
    errs = cc_bad2.validate()
    _assert(
        any("retention_policy" in e for e in errs),
        "CheckpointConfig: rejects invalid retention_policy",
    )

    # -------------------------------------------------------------------
    # 5. LoggingConfig
    # -------------------------------------------------------------------
    print("\n--- LoggingConfig ---")

    lc = LoggingConfig()
    _assert(lc.tensorboard_enabled is True, "LoggingConfig: default tensorboard")
    _assert(lc.wandb_enabled is False, "LoggingConfig: default wandb")
    _assert(lc.wandb_project == "brain_ai", "LoggingConfig: default wandb_project")
    _assert(lc.wandb_entity is None, "LoggingConfig: default wandb_entity")
    _assert(lc.wandb_resume_mode == "never", "LoggingConfig: default resume_mode")
    _assert(lc.jsonl_enabled is True, "LoggingConfig: default jsonl")
    _assert(lc.log_histograms_every == 500, "LoggingConfig: default histograms_every")
    _assert(lc.log_images_every == 1000, "LoggingConfig: default images_every")
    _assert(lc.log_system_every == 100, "LoggingConfig: default system_every")
    _assert(lc.flush_every == 100, "LoggingConfig: default flush_every")
    _assert(len(lc.validate()) == 0, "LoggingConfig: defaults validate cleanly")

    lc_bad = LoggingConfig(wandb_resume_mode="force")
    errs = lc_bad.validate()
    _assert(
        any("resume_mode" in e for e in errs),
        "LoggingConfig: rejects invalid resume_mode",
    )

    # -------------------------------------------------------------------
    # 6. AblationConfig
    # -------------------------------------------------------------------
    print("\n--- AblationConfig ---")

    ac = AblationConfig()
    _assert(ac.spec_file is None, "AblationConfig: default spec_file")
    _assert(ac.mode == "full", "AblationConfig: default mode")
    _assert(ac.parallel is False, "AblationConfig: default parallel")
    _assert(ac.max_workers == 4, "AblationConfig: default max_workers")
    _assert(ac.phases == [4], "AblationConfig: default phases")
    _assert(ac.seeds == [1337], "AblationConfig: default seeds")
    _assert(ac.baseline_run_id is None, "AblationConfig: default baseline_run_id")
    _assert(len(ac.validate()) == 0, "AblationConfig: defaults validate cleanly")

    ac_bad = AblationConfig(mode="random")
    errs = ac_bad.validate()
    _assert(
        any("mode" in e for e in errs),
        "AblationConfig: rejects invalid mode",
    )

    ac_bad2 = AblationConfig(phases=[0, 9])
    errs = ac_bad2.validate()
    _assert(
        any("invalid phase" in e.lower() for e in errs),
        "AblationConfig: rejects invalid phases",
    )

    # -------------------------------------------------------------------
    # 7. PhaseSpecificConfig
    # -------------------------------------------------------------------
    print("\n--- PhaseSpecificConfig ---")

    # 7a. Defaults for all 7 phases
    for phase in range(1, 8):
        pcfg = PhaseSpecificConfig.for_phase(phase)
        errs = pcfg.validate()
        _assert(
            len(errs) == 0,
            f"PhaseSpecificConfig: phase {phase} defaults validate cleanly",
            f"errors: {errs}",
        )

    # 7b. Phase 1 defaults
    p1 = PhaseSpecificConfig.for_phase(1)
    _assert(p1.lr == 1e-3, "Phase1: lr=1e-3")
    _assert(p1.epochs == 20, "Phase1: epochs=20")
    _assert(p1.batch_size == 64, "Phase1: batch_size=64")
    _assert(p1.optimizer == "adamw", "Phase1: optimizer=adamw")
    _assert(p1.warmup_steps == 500, "Phase1: warmup_steps=500")
    _assert(p1.dataset == "auto", "Phase1: dataset=auto")

    # 7c. Phase 7 defaults
    p7 = PhaseSpecificConfig.for_phase(7)
    _assert(p7.optimizer == "adam", "Phase7: optimizer=adam")
    _assert(p7.scheduler == "step", "Phase7: scheduler=step")
    _assert(p7.batch_size == 8, "Phase7: batch_size=8")
    _assert(p7.weight_decay == 0.0, "Phase7: weight_decay=0.0")

    # 7d. Override via for_phase
    p4_custom = PhaseSpecificConfig.for_phase(4, lr=1e-5, epochs=100)
    _assert(p4_custom.lr == 1e-5, "PhaseSpecific: for_phase override lr")
    _assert(p4_custom.epochs == 100, "PhaseSpecific: for_phase override epochs")
    _assert(
        p4_custom.optimizer == "adamw",
        "PhaseSpecific: non-overridden field uses phase default",
    )

    # 7e. Invalid phase
    try:
        PhaseSpecificConfig.for_phase(0)
        _assert(False, "PhaseSpecific: for_phase(0) should raise")
    except ValueError:
        _assert(True, "PhaseSpecific: for_phase(0) raises ValueError")

    # 7f. Invalid scheduler
    p_bad = PhaseSpecificConfig(scheduler="triangular")
    errs = p_bad.validate()
    _assert(
        any("scheduler" in e for e in errs),
        "PhaseSpecific: rejects invalid scheduler",
    )

    # 7g. Invalid optimizer
    p_bad2 = PhaseSpecificConfig(optimizer="lion")
    errs = p_bad2.validate()
    _assert(
        any("optimizer" in e for e in errs),
        "PhaseSpecific: rejects invalid optimizer",
    )

    # 7h. Negative lr
    p_bad3 = PhaseSpecificConfig(lr=-0.01)
    errs = p_bad3.validate()
    _assert(any("lr" in e for e in errs), "PhaseSpecific: rejects negative lr")

    # -------------------------------------------------------------------
    # 8. TrainingFullConfig aggregation
    # -------------------------------------------------------------------
    print("\n--- TrainingFullConfig ---")

    full = TrainingFullConfig()
    _assert(full.training.mode == "dev", "FullConfig: default training mode")
    _assert(full.seed.base_seed == 1337, "FullConfig: default seed")
    _assert(len(full.phase_configs) == 7, "FullConfig: auto-populates 7 phase_configs")
    errs = full.validate()
    _assert(len(errs) == 0, "FullConfig: defaults validate cleanly", f"errors: {errs}")

    # 8b. All 7 phases populated
    for phase in range(1, 8):
        _assert(
            phase in full.phase_configs,
            f"FullConfig: phase_configs contains phase {phase}",
        )

    # -------------------------------------------------------------------
    # 9. Presets
    # -------------------------------------------------------------------
    print("\n--- Presets ---")

    # 9a. minimal
    cfg_min = TrainingFullConfig.minimal()
    _assert(cfg_min.training.mode == "dev", "minimal: mode is dev")
    _assert(cfg_min.seed.base_seed == 42, "minimal: seed is 42")
    _assert(
        cfg_min.logging.tensorboard_enabled is False,
        "minimal: tensorboard disabled",
    )
    _assert(cfg_min.logging.wandb_enabled is False, "minimal: wandb disabled")
    _assert(cfg_min.logging.jsonl_enabled is False, "minimal: jsonl disabled")
    _assert(
        all(cfg_min.phase_configs[p].epochs == 2 for p in range(1, 8)),
        "minimal: all phases have 2 epochs",
    )
    errs = cfg_min.validate()
    _assert(len(errs) == 0, "minimal: validates cleanly", f"errors: {errs}")

    # 9b. dev
    cfg_dev = TrainingFullConfig.dev()
    _assert(cfg_dev.training.mode == "dev", "dev: mode is dev")
    _assert(cfg_dev.seed.enforce_deterministic is True, "dev: deterministic")
    _assert(cfg_dev.logging.tensorboard_enabled is True, "dev: tensorboard enabled")
    _assert(cfg_dev.logging.wandb_enabled is False, "dev: wandb disabled")
    _assert(cfg_dev.training.use_amp is False, "dev: no AMP")
    errs = cfg_dev.validate()
    _assert(len(errs) == 0, "dev: validates cleanly", f"errors: {errs}")

    # 9c. production
    cfg_prod = TrainingFullConfig.production()
    _assert(cfg_prod.training.mode == "production", "production: mode is production")
    _assert(cfg_prod.training.use_amp is True, "production: AMP enabled")
    _assert(cfg_prod.logging.wandb_enabled is True, "production: wandb enabled")
    _assert(
        cfg_prod.seed.enforce_deterministic is False,
        "production: relaxed determinism",
    )
    _assert(
        cfg_prod.seed.cudnn_benchmark is True,
        "production: cudnn_benchmark enabled",
    )
    _assert(
        cfg_prod.logging.wandb_resume_mode == "allow",
        "production: wandb resume_mode is allow",
    )
    # Production should have more epochs than dev defaults
    _assert(
        cfg_prod.phase_configs[1].epochs > PhaseSpecificConfig.PHASE_DEFAULTS[1]["epochs"],
        "production: phase 1 has more epochs than default",
    )
    # Note: production validate may flag AMP without CUDA -- that is expected
    # on CPU-only test machines.

    # 9d. ablation_dev
    cfg_abl = TrainingFullConfig.ablation_dev()
    _assert(cfg_abl.ablation.mode == "pairwise", "ablation_dev: pairwise mode")
    _assert(cfg_abl.ablation.seeds == [1337, 42], "ablation_dev: seeds")
    _assert(cfg_abl.ablation.phases == [4], "ablation_dev: phases")
    _assert(
        all(cfg_abl.phase_configs[p].epochs == 5 for p in range(1, 8)),
        "ablation_dev: all phases have 5 epochs",
    )
    errs = cfg_abl.validate()
    _assert(len(errs) == 0, "ablation_dev: validates cleanly", f"errors: {errs}")

    # -------------------------------------------------------------------
    # 10. Serialization round-trip (to_dict / from_dict)
    # -------------------------------------------------------------------
    print("\n--- Serialization ---")

    cfg_orig = TrainingFullConfig.dev()
    d = cfg_orig.to_dict()
    _assert(isinstance(d, dict), "to_dict: returns dict")
    _assert("training" in d, "to_dict: has training key")
    _assert("manifest" in d, "to_dict: has manifest key")
    _assert("seed" in d, "to_dict: has seed key")
    _assert("checkpoint" in d, "to_dict: has checkpoint key")
    _assert("logging" in d, "to_dict: has logging key")
    _assert("ablation" in d, "to_dict: has ablation key")
    _assert("phase_configs" in d, "to_dict: has phase_configs key")
    _assert(len(d["phase_configs"]) == 7, "to_dict: phase_configs has 7 entries")

    cfg_restored = TrainingFullConfig.from_dict(d)
    _assert(
        cfg_restored.training.mode == cfg_orig.training.mode,
        "from_dict: training.mode round-trips",
    )
    _assert(
        cfg_restored.seed.base_seed == cfg_orig.seed.base_seed,
        "from_dict: seed.base_seed round-trips",
    )
    _assert(
        cfg_restored.checkpoint.retention_policy == cfg_orig.checkpoint.retention_policy,
        "from_dict: checkpoint.retention_policy round-trips",
    )
    _assert(
        cfg_restored.logging.wandb_project == cfg_orig.logging.wandb_project,
        "from_dict: logging.wandb_project round-trips",
    )
    _assert(
        cfg_restored.phase_configs[4].lr == cfg_orig.phase_configs[4].lr,
        "from_dict: phase_configs[4].lr round-trips",
    )
    _assert(
        len(cfg_restored.phase_configs) == 7,
        "from_dict: phase_configs has 7 entries after round-trip",
    )

    # -------------------------------------------------------------------
    # 11. File I/O (save_json / load_json)
    # -------------------------------------------------------------------
    print("\n--- File I/O ---")

    tmpdir = tempfile.mkdtemp(prefix="training_config_test_")
    try:
        json_path = os.path.join(tmpdir, "config.json")
        cfg_save = TrainingFullConfig.dev()
        saved_path = cfg_save.save_json(json_path)
        _assert(os.path.exists(str(saved_path)), "save_json: file written")

        cfg_loaded = TrainingFullConfig.load_json(json_path)
        _assert(
            cfg_loaded.training.mode == cfg_save.training.mode,
            "load_json: mode matches after round-trip",
        )
        _assert(
            cfg_loaded.seed.base_seed == cfg_save.seed.base_seed,
            "load_json: base_seed matches after round-trip",
        )
        _assert(
            cfg_loaded.phase_configs[3].lr == cfg_save.phase_configs[3].lr,
            "load_json: phase_configs[3].lr matches after round-trip",
        )
        _assert(
            cfg_loaded.logging.tensorboard_enabled == cfg_save.logging.tensorboard_enabled,
            "load_json: tensorboard_enabled matches",
        )

        # Verify JSON is valid and readable
        with open(json_path, "r") as f:
            raw = json.load(f)
        _assert(isinstance(raw, dict), "save_json: produces valid JSON dict")
        _assert("training" in raw, "save_json: JSON has training key")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # -------------------------------------------------------------------
    # 12. merge_overrides
    # -------------------------------------------------------------------
    print("\n--- merge_overrides ---")

    cfg_merge = TrainingFullConfig.dev()
    cfg_merge.merge_overrides({
        "training.use_amp": True,
        "seed.base_seed": 42,
        "phase_configs.4.lr": 1e-5,
        "logging.wandb_enabled": True,
        "checkpoint.keep_last_n": 10,
    })
    _assert(
        cfg_merge.training.use_amp is True,
        "merge_overrides: training.use_amp applied",
    )
    _assert(
        cfg_merge.seed.base_seed == 42,
        "merge_overrides: seed.base_seed applied",
    )
    _assert(
        cfg_merge.phase_configs[4].lr == 1e-5,
        "merge_overrides: phase_configs.4.lr applied",
    )
    _assert(
        cfg_merge.logging.wandb_enabled is True,
        "merge_overrides: logging.wandb_enabled applied",
    )
    _assert(
        cfg_merge.checkpoint.keep_last_n == 10,
        "merge_overrides: checkpoint.keep_last_n applied",
    )

    # Flat key override
    cfg_merge2 = TrainingFullConfig.dev()
    cfg_merge2.merge_overrides({"mode": "production"})
    _assert(
        cfg_merge2.training.mode == "production",
        "merge_overrides: flat key 'mode' applied to training",
    )

    # -------------------------------------------------------------------
    # 13. ConfigResolver
    # -------------------------------------------------------------------
    print("\n--- ConfigResolver ---")

    # 13a. Dev mode resolution
    cfg_auto = TrainingFullConfig.dev()
    _assert(
        ConfigResolver.has_auto_values(cfg_auto),
        "has_auto_values: True before resolve (dev)",
    )
    resolved = ConfigResolver.resolve(cfg_auto)
    _assert(
        not ConfigResolver.has_auto_values(resolved),
        "has_auto_values: False after resolve",
    )
    _assert(
        resolved.manifest.dataset_fingerprint_tier == "tier1",
        "resolve: dev fingerprint tier -> tier1",
    )
    _assert(
        resolved.phase_configs[1].dataset == "mnist",
        "resolve: dev phase 1 dataset -> mnist",
    )
    _assert(
        resolved.phase_configs[5].dataset == "cartpole",
        "resolve: dev phase 5 dataset -> cartpole",
    )
    _assert(
        resolved.phase_configs[7].dataset == "omniglot",
        "resolve: dev phase 7 dataset -> omniglot",
    )

    # 13b. Production mode resolution
    cfg_auto_prod = TrainingFullConfig.production()
    resolved_prod = ConfigResolver.resolve(cfg_auto_prod)
    _assert(
        not ConfigResolver.has_auto_values(resolved_prod),
        "has_auto_values: False after resolve (production)",
    )
    _assert(
        resolved_prod.manifest.dataset_fingerprint_tier == "tier2",
        "resolve: production fingerprint tier -> tier2",
    )
    _assert(
        resolved_prod.phase_configs[1].dataset == "imagenet",
        "resolve: production phase 1 dataset -> imagenet",
    )
    _assert(
        resolved_prod.phase_configs[4].dataset == "multimodal_combined",
        "resolve: production phase 4 dataset -> multimodal_combined",
    )
    _assert(
        resolved_prod.phase_configs[7].dataset == "meta_dataset",
        "resolve: production phase 7 dataset -> meta_dataset",
    )

    # 13c. Resolve does not mutate original
    _assert(
        cfg_auto.phase_configs[1].dataset == "auto",
        "resolve: original config not mutated",
    )

    # 13d. Already-resolved config passes through unchanged
    resolved2 = ConfigResolver.resolve(resolved)
    _assert(
        resolved2.phase_configs[1].dataset == resolved.phase_configs[1].dataset,
        "resolve: already-resolved config is idempotent",
    )

    # -------------------------------------------------------------------
    # 14. PhaseSpecificConfig PHASE_DEFAULTS coverage
    # -------------------------------------------------------------------
    print("\n--- PhaseDefaults Coverage ---")

    _assert(
        len(PhaseSpecificConfig.PHASE_DEFAULTS) == 7,
        "PHASE_DEFAULTS: has 7 entries",
    )
    for phase in range(1, 8):
        defaults = PhaseSpecificConfig.PHASE_DEFAULTS[phase]
        required_keys = {"lr", "epochs", "batch_size", "optimizer",
                         "warmup_steps", "weight_decay", "scheduler", "dataset"}
        _assert(
            required_keys.issubset(defaults.keys()),
            f"PHASE_DEFAULTS[{phase}]: has all required keys",
            f"missing: {required_keys - defaults.keys()}",
        )

    # -------------------------------------------------------------------
    # 15. Cross-field validation
    # -------------------------------------------------------------------
    print("\n--- Cross-field Validation ---")

    # 15a. AMP without CUDA (if torch available and no CUDA)
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    if not has_cuda:
        cfg_amp = TrainingFullConfig(
            training=TrainingConfig(use_amp=True),
        )
        errs = cfg_amp.validate()
        _assert(
            any("amp" in e.lower() or "cuda" in e.lower() for e in errs),
            "Cross-field: AMP without CUDA flagged",
        )
    else:
        # On CUDA machines, AMP should be fine
        cfg_amp = TrainingFullConfig(
            training=TrainingConfig(use_amp=True),
        )
        errs = cfg_amp.validate()
        amp_errs = [e for e in errs if "amp" in e.lower() or "cuda" in e.lower()]
        _assert(
            len(amp_errs) == 0,
            "Cross-field: AMP with CUDA does not produce AMP errors",
        )

    # 15b. Invalid phase_config within full config
    cfg_bad_phase = TrainingFullConfig()
    cfg_bad_phase.phase_configs[3] = PhaseSpecificConfig(lr=-0.01)
    errs = cfg_bad_phase.validate()
    _assert(
        any("phase_configs[3]" in e for e in errs),
        "Cross-field: invalid phase_config detected in validate",
    )

    # -------------------------------------------------------------------
    # 16. TrainingFullConfig with partial phase_configs
    # -------------------------------------------------------------------
    print("\n--- Partial Phase Configs ---")

    # Create with only 2 phase_configs specified; rest auto-populated
    cfg_partial = TrainingFullConfig(
        phase_configs={
            1: PhaseSpecificConfig(lr=0.1, epochs=5, batch_size=16, optimizer="sgd"),
            4: PhaseSpecificConfig(lr=0.001, epochs=50, batch_size=8, optimizer="adamw"),
        }
    )
    _assert(
        len(cfg_partial.phase_configs) == 7,
        "Partial: auto-populates missing phases to 7 total",
    )
    _assert(
        cfg_partial.phase_configs[1].lr == 0.1,
        "Partial: specified phase 1 preserved",
    )
    _assert(
        cfg_partial.phase_configs[4].epochs == 50,
        "Partial: specified phase 4 preserved",
    )
    _assert(
        cfg_partial.phase_configs[2].lr == PhaseSpecificConfig.PHASE_DEFAULTS[2]["lr"],
        "Partial: unspecified phase 2 uses defaults",
    )

    # -------------------------------------------------------------------
    # 17. to_dict / from_dict with overrides and resolve
    # -------------------------------------------------------------------
    print("\n--- Serialize + Resolve Integration ---")

    cfg_integ = TrainingFullConfig.dev()
    cfg_integ.merge_overrides({"seed.base_seed": 999, "phase_configs.2.epochs": 77})
    resolved_integ = ConfigResolver.resolve(cfg_integ)

    d_integ = resolved_integ.to_dict()
    cfg_integ_restored = TrainingFullConfig.from_dict(d_integ)
    _assert(
        cfg_integ_restored.seed.base_seed == 999,
        "Integ: base_seed=999 survives serialize+resolve+restore",
    )
    _assert(
        cfg_integ_restored.phase_configs[2].epochs == 77,
        "Integ: phase 2 epochs=77 survives serialize+resolve+restore",
    )
    _assert(
        cfg_integ_restored.phase_configs[1].dataset == "mnist",
        "Integ: resolved dataset survives round-trip",
    )
    _assert(
        not ConfigResolver.has_auto_values(cfg_integ_restored),
        "Integ: no auto values after full round-trip",
    )

    # -------------------------------------------------------------------
    # 18. File I/O + resolve round-trip
    # -------------------------------------------------------------------
    print("\n--- File I/O + Resolve ---")

    tmpdir2 = tempfile.mkdtemp(prefix="training_config_integ_")
    try:
        cfg_io = TrainingFullConfig.dev()
        resolved_io = ConfigResolver.resolve(cfg_io)
        io_path = os.path.join(tmpdir2, "resolved_config.json")
        resolved_io.save_json(io_path)

        loaded_io = TrainingFullConfig.load_json(io_path)
        _assert(
            not ConfigResolver.has_auto_values(loaded_io),
            "IO+Resolve: no auto values after save/load",
        )
        _assert(
            loaded_io.phase_configs[3].dataset == "sequential_mnist",
            "IO+Resolve: phase 3 dataset preserved",
        )
        errs = loaded_io.validate()
        _assert(
            len(errs) == 0,
            "IO+Resolve: loaded config validates cleanly",
            f"errors: {errs}",
        )
    finally:
        shutil.rmtree(tmpdir2, ignore_errors=True)

    # -------------------------------------------------------------------
    # 19. Edge cases
    # -------------------------------------------------------------------
    print("\n--- Edge Cases ---")

    # 19a. Empty overrides dict does nothing
    cfg_noop = TrainingFullConfig.dev()
    original_seed = cfg_noop.seed.base_seed
    cfg_noop.merge_overrides({})
    _assert(
        cfg_noop.seed.base_seed == original_seed,
        "Edge: empty overrides dict is a no-op",
    )

    # 19b. Unknown override key is silently ignored
    cfg_unknown = TrainingFullConfig.dev()
    cfg_unknown.merge_overrides({"nonexistent_field": 42})
    _assert(True, "Edge: unknown override key does not raise")

    # 19c. from_dict with empty dict still produces valid config
    cfg_empty = TrainingFullConfig.from_dict({})
    _assert(
        len(cfg_empty.phase_configs) == 7,
        "Edge: from_dict({}) auto-populates 7 phase_configs",
    )

    # 19d. Preset configs are independent (modifying one does not affect another)
    cfg_a = TrainingFullConfig.dev()
    cfg_b = TrainingFullConfig.dev()
    cfg_a.seed.base_seed = 99999
    _assert(
        cfg_b.seed.base_seed == 1337,
        "Edge: preset configs are independent instances",
    )

    # 19e. Phase configs with for_phase overrides validate
    p_custom = PhaseSpecificConfig.for_phase(5, lr=1e-6, epochs=100, scheduler="linear")
    errs = p_custom.validate()
    _assert(
        len(errs) == 0,
        "Edge: for_phase with valid overrides validates cleanly",
    )

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"Results: {_pass_count} passed, {_fail_count} failed, "
          f"{_pass_count + _fail_count} total")
    print("=" * 72)
    if _fail_count > 0:
        print("\nFailed tests:")
        for e in _errors:
            print(e)
        sys.exit(1)
    else:
        print("\nAll tests passed.")
        sys.exit(0)
