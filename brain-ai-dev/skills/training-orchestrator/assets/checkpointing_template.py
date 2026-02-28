#!/usr/bin/env python3
"""
checkpointing_template.py
=========================

Comprehensive checkpoint management for the brain-inspired AI seven-phase
training pipeline.  Provides:

    - CheckpointManager      -- top-level save / load / resume / cleanup API
    - AtomicSaver             -- crash-safe save via temp-file + rename
    - CheckpointNaming        -- deterministic naming conventions
    - PhaseBoundaryBuilder    -- cross-phase artifact construction
    - ResumeManager           -- find + validate + restore in one call
    - CheckpointRetentionPolicy -- disk-space-aware cleanup
    - validate_checkpoint     -- schema enforcement

Self-contained: no brain_ai imports.  Uses torch, dataclasses, pathlib, etc.
Run ``python checkpointing_template.py`` for 50+ self-tests.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

CHECKPOINT_SCHEMA_VERSION = "1.0"

PERIODIC_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "rng_states",
        "global_step",
        "phase",
        "phase_step",
        "epoch",
        "best_metrics",
        "config_hash",
    }
)

BOUNDARY_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "source_phase",
        "target_phase",
        "source_run_id",
        "model_state_dict",
        "config_snapshot",
        "feature_flags",
        "compatibility",
        "metadata",
    }
)


def validate_checkpoint(
    ckpt_dict: Dict[str, Any],
    expected_type: Literal["periodic", "boundary"] = "periodic",
) -> List[str]:
    """Return a list of validation errors (empty == valid).

    Checks:
        - Required keys present
        - schema_version matches CHECKPOINT_SCHEMA_VERSION
    """
    errors: List[str] = []
    required = PERIODIC_REQUIRED_KEYS if expected_type == "periodic" else BOUNDARY_REQUIRED_KEYS
    missing = required - set(ckpt_dict.keys())
    if missing:
        errors.append(f"Missing keys for {expected_type} checkpoint: {sorted(missing)}")
    ver = ckpt_dict.get("schema_version")
    if ver is not None and ver != CHECKPOINT_SCHEMA_VERSION:
        errors.append(
            f"Schema version mismatch: expected {CHECKPOINT_SCHEMA_VERSION!r}, got {ver!r}"
        )
    return errors


# ---------------------------------------------------------------------------
# AtomicSaver
# ---------------------------------------------------------------------------


class AtomicSaver:
    """Save ``torch.save``-compatible data atomically using temp + rename.

    On POSIX systems ``os.rename`` on the same filesystem is atomic, so a
    crash during ``torch.save`` can never leave a half-written checkpoint at
    the final path.
    """

    @staticmethod
    def save_atomic(data: Any, path: Union[str, Path]) -> Path:
        """Save *data* to *path* atomically.

        1. Write to ``<path>.tmp.<random>`` in the same directory.
        2. ``os.replace`` (atomic on POSIX same-fs) to final *path*.
        3. Return the resolved final ``Path``.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = f".tmp.{uuid.uuid4().hex[:12]}"
        tmp_path = path.with_name(path.name + suffix)
        try:
            torch.save(data, str(tmp_path))
            os.replace(str(tmp_path), str(path))
            logger.debug("Atomic save complete: %s", path)
            return path
        except BaseException:
            # Clean up partial temp file on any failure.
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    @staticmethod
    def save_atomic_json(data: Any, path: Union[str, Path]) -> Path:
        """Atomic JSON write (useful for lightweight metadata files)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = f".tmp.{uuid.uuid4().hex[:12]}"
        tmp_path = path.with_name(path.name + suffix)
        try:
            with open(tmp_path, "w") as fh:
                json.dump(data, fh, indent=2, default=str)
            os.replace(str(tmp_path), str(path))
            return path
        except BaseException:
            if tmp_path.exists():
                tmp_path.unlink()
            raise


# ---------------------------------------------------------------------------
# CheckpointNaming
# ---------------------------------------------------------------------------


class CheckpointNaming:
    """Deterministic, parseable checkpoint naming conventions."""

    _STEP_RE = re.compile(r"ckpt_step(\d{8})\.pt$")

    @staticmethod
    def periodic_name(global_step: int) -> str:
        """``ckpt_step00000042.pt``"""
        return f"ckpt_step{global_step:08d}.pt"

    @staticmethod
    def best_name(metric_key: str) -> str:
        """``ckpt_best_val_loss.pt``"""
        safe_key = re.sub(r"[^a-zA-Z0-9_]", "_", metric_key)
        return f"ckpt_best_{safe_key}.pt"

    @staticmethod
    def boundary_name() -> str:
        """``phase_boundary.pt``"""
        return "phase_boundary.pt"

    @staticmethod
    def final_name() -> str:
        """``ckpt_final.pt``"""
        return "ckpt_final.pt"

    @classmethod
    def parse_step_from_name(cls, filename: str) -> Optional[int]:
        """Extract global step from a periodic checkpoint filename.

        Returns ``None`` if *filename* does not match the expected pattern.
        """
        m = cls._STEP_RE.search(str(filename))
        if m:
            return int(m.group(1))
        return None

    @staticmethod
    def phase_dir(run_dir: Union[str, Path], phase: int) -> Path:
        """``run_dir / checkpoints / phase{phase}``"""
        return Path(run_dir) / "checkpoints" / f"phase{phase}"


# ---------------------------------------------------------------------------
# PhaseBoundaryBuilder
# ---------------------------------------------------------------------------


class PhaseBoundaryBuilder:
    """Constructs a ``phase_boundary.pt`` dict for cross-phase transfer.

    Phase boundary checkpoints are intentionally *lightweight*: they carry
    only the model weights and enough metadata for the next phase to
    validate compatibility.  Optimizer / scheduler / RNG states are
    excluded so that each phase starts with a fresh optimiser.
    """

    @staticmethod
    def build(
        model: nn.Module,
        config: Dict[str, Any],
        phase: int,
        run_id: str,
        *,
        best_metrics: Optional[Dict[str, float]] = None,
        total_steps: int = 0,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build the phase-boundary dict.

        Parameters
        ----------
        model : nn.Module
            The trained model at the end of *phase*.
        config : dict
            Full config snapshot (serialisable).
        phase : int
            The phase that just completed (1-7).
        run_id : str
            Unique identifier for this training run.
        best_metrics : dict, optional
            Best validation metrics achieved during the phase.
        total_steps : int
            Total training steps completed in this phase.
        extra_metadata : dict, optional
            Arbitrary extra metadata to attach.
        """
        compatibility = PhaseBoundaryBuilder._extract_compatibility(config)
        feature_flags = PhaseBoundaryBuilder._extract_feature_flags(config)
        metadata: Dict[str, Any] = {
            "best_metrics": best_metrics or {},
            "training_steps": total_steps,
            "timestamp": time.time(),
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "source_phase": phase,
            "target_phase": phase + 1,
            "source_run_id": run_id,
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "config_snapshot": config,
            "feature_flags": feature_flags,
            "compatibility": compatibility,
            "metadata": metadata,
        }

    @staticmethod
    def _extract_compatibility(config: Dict[str, Any]) -> Dict[str, Any]:
        """Pull dimension / vocab info needed by downstream phases."""
        return {
            "workspace_dim": config.get("workspace_dim", 4096),
            "vocab_size": config.get("vocab_size", 32000),
            "num_encoders": config.get("num_encoders", 4),
            "snn_hidden": config.get("snn_hidden", 1024),
            "htm_columns": config.get("htm_columns", 2048),
        }

    @staticmethod
    def _extract_feature_flags(config: Dict[str, Any]) -> Dict[str, bool]:
        """Extract boolean feature flags from config."""
        flag_keys = [
            "use_snn",
            "use_htm",
            "use_workspace",
            "use_symbolic",
            "use_meta",
            "use_engram",
        ]
        return {k: config.get(k, True) for k in flag_keys}

    @staticmethod
    def extract_model_subset(
        state_dict: Dict[str, torch.Tensor],
        phase: int,
    ) -> Dict[str, torch.Tensor]:
        """Optionally extract only the weights needed by the *next* phase.

        The default implementation returns the full state dict.  Override
        or extend with phase-specific prefix filters when model sizes are
        very large.
        """
        # Phase-prefix mapping (illustrative — adjust to real module names).
        phase_prefixes: Dict[int, List[str]] = {
            1: ["core.", "encoder."],
            2: ["encoder."],
            3: ["temporal.", "htm."],
            4: ["workspace.", "global_workspace."],
            5: ["decision.", "active_inference."],
            6: ["reasoning.", "dual_process."],
            7: ["meta.", "neuromod."],
        }
        prefixes = phase_prefixes.get(phase)
        if prefixes is None:
            return state_dict
        subset = {
            k: v
            for k, v in state_dict.items()
            if any(k.startswith(p) for p in prefixes)
        }
        # Fall back to full dict when no keys matched (safety).
        return subset if subset else state_dict


# ---------------------------------------------------------------------------
# ResumeInfo dataclass
# ---------------------------------------------------------------------------


@dataclass
class ResumeInfo:
    """Information returned after a successful resume."""

    checkpoint_path: Path
    global_step: int
    phase_step: int
    epoch: int
    was_best: bool
    metrics: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ResumeManager
# ---------------------------------------------------------------------------


class ResumeManager:
    """High-level helper: find, validate, and restore a checkpoint in one call."""

    @staticmethod
    def find_resume_point(
        run_dir: Union[str, Path],
        phase: Optional[int] = None,
    ) -> Optional[Path]:
        """Find the latest periodic checkpoint for *phase* (or any phase).

        Search order within a phase directory:
            1. ``ckpt_final.pt``
            2. Most recent ``ckpt_step*.pt`` by step number
        """
        run_dir = Path(run_dir)
        if phase is not None:
            phases_to_check = [phase]
        else:
            ckpt_root = run_dir / "checkpoints"
            if not ckpt_root.exists():
                return None
            phases_to_check = sorted(
                (
                    int(d.name.replace("phase", ""))
                    for d in ckpt_root.iterdir()
                    if d.is_dir() and d.name.startswith("phase")
                ),
                reverse=True,
            )

        for ph in phases_to_check:
            pdir = CheckpointNaming.phase_dir(run_dir, ph)
            if not pdir.exists():
                continue
            final = pdir / CheckpointNaming.final_name()
            if final.exists():
                return final
            periodics = sorted(pdir.glob("ckpt_step*.pt"))
            if periodics:
                return periodics[-1]
        return None

    @staticmethod
    def validate_resume_compatibility(
        checkpoint: Dict[str, Any],
        current_config: Dict[str, Any],
    ) -> List[str]:
        """Check that checkpoint is compatible with current config.

        Returns a list of warnings (empty == fully compatible).
        """
        warnings: List[str] = []

        # Config hash check (informational — not blocking).
        ckpt_hash = checkpoint.get("config_hash")
        if ckpt_hash is not None:
            cur_hash = _config_hash(current_config)
            if ckpt_hash != cur_hash:
                warnings.append(
                    f"Config hash mismatch (ckpt={ckpt_hash[:12]}.. vs current={cur_hash[:12]}..)"
                )

        # Model key check.
        ckpt_keys = set(checkpoint.get("model_state_dict", {}).keys())
        if not ckpt_keys:
            warnings.append("Checkpoint model_state_dict is empty")

        return warnings

    @staticmethod
    def resume(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        run_dir: Union[str, Path],
        phase: Optional[int] = None,
        current_config: Optional[Dict[str, Any]] = None,
        strict_load: bool = True,
    ) -> Optional[ResumeInfo]:
        """Full resume flow: find -> load -> validate -> restore.

        Returns ``None`` if no checkpoint is found.
        """
        ckpt_path = ResumeManager.find_resume_point(run_dir, phase)
        if ckpt_path is None:
            logger.info("No checkpoint found for resume (run_dir=%s, phase=%s)", run_dir, phase)
            return None

        logger.info("Resuming from %s", ckpt_path)
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

        # Validate.
        if current_config is not None:
            warns = ResumeManager.validate_resume_compatibility(ckpt, current_config)
            for w in warns:
                logger.warning("Resume compatibility: %s", w)

        # Restore model.
        model.load_state_dict(ckpt["model_state_dict"], strict=strict_load)

        # Restore optimizer.
        if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # Restore scheduler.
        if (
            scheduler is not None
            and "scheduler_state_dict" in ckpt
            and ckpt["scheduler_state_dict"] is not None
        ):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        # Restore RNG states.
        rng = ckpt.get("rng_states")
        if rng is not None:
            _restore_rng_states(rng)

        best_metrics = ckpt.get("best_metrics", {})
        return ResumeInfo(
            checkpoint_path=ckpt_path,
            global_step=ckpt.get("global_step", 0),
            phase_step=ckpt.get("phase_step", 0),
            epoch=ckpt.get("epoch", 0),
            was_best=False,
            metrics=best_metrics,
        )


# ---------------------------------------------------------------------------
# CheckpointRetentionPolicy
# ---------------------------------------------------------------------------


class CheckpointRetentionPolicy:
    """Disk-space-aware cleanup for periodic checkpoints.

    "Protected" files are never deleted:
        - ``phase_boundary.pt``
        - ``ckpt_best_*.pt``
        - ``ckpt_final.pt``
    """

    def __init__(
        self,
        keep_best: bool = True,
        keep_boundary: bool = True,
        keep_last: bool = True,
        keep_every_n: Optional[int] = None,
        max_checkpoints: int = 5,
    ) -> None:
        self.keep_best = keep_best
        self.keep_boundary = keep_boundary
        self.keep_last = keep_last
        self.keep_every_n = keep_every_n
        self.max_checkpoints = max_checkpoints

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_protected(path: Path) -> bool:
        name = path.name
        return (
            name == "phase_boundary.pt"
            or name.startswith("ckpt_best_")
            or name == "ckpt_final.pt"
        )

    def get_deletable(self, checkpoint_dir: Union[str, Path]) -> List[Path]:
        """Return list of checkpoints eligible for deletion."""
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            return []

        # Collect periodic checkpoints sorted by step ascending.
        periodics: List[Tuple[int, Path]] = []
        for p in sorted(checkpoint_dir.glob("ckpt_step*.pt")):
            step = CheckpointNaming.parse_step_from_name(p.name)
            if step is not None:
                periodics.append((step, p))
        periodics.sort(key=lambda x: x[0])

        if not periodics:
            return []

        # Build the "keep" set.
        keep_indices: set[int] = set()

        # Always keep the latest.
        if self.keep_last:
            keep_indices.add(len(periodics) - 1)

        # Keep every N-th.
        if self.keep_every_n and self.keep_every_n > 0:
            for idx, (step, _) in enumerate(periodics):
                if step % self.keep_every_n == 0:
                    keep_indices.add(idx)

        # Keep last max_checkpoints.
        remaining = [i for i in range(len(periodics)) if i not in keep_indices]
        slots = max(0, self.max_checkpoints - len(keep_indices))
        if slots > 0 and remaining:
            # Keep the most recent `slots` from remaining.
            for idx in remaining[-slots:]:
                keep_indices.add(idx)

        deletable: List[Path] = []
        for idx, (_, path) in enumerate(periodics):
            if idx not in keep_indices and not self._is_protected(path):
                deletable.append(path)

        return deletable

    def apply(self, checkpoint_dir: Union[str, Path]) -> List[Path]:
        """Delete checkpoints not matching the retention policy.

        Returns list of paths that were deleted.
        """
        deletable = self.get_deletable(checkpoint_dir)
        deleted: List[Path] = []
        for p in deletable:
            try:
                p.unlink()
                deleted.append(p)
                logger.debug("Deleted checkpoint: %s", p)
            except OSError as exc:
                logger.warning("Failed to delete %s: %s", p, exc)
        return deleted


# ---------------------------------------------------------------------------
# RNG helpers
# ---------------------------------------------------------------------------


def _capture_rng_states() -> Dict[str, Any]:
    """Capture Python, NumPy, and PyTorch RNG states."""
    import random

    states: Dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.random.get_rng_state(),
    }
    try:
        import numpy as np

        states["numpy"] = np.random.get_state()
    except ImportError:
        pass
    if torch.cuda.is_available():
        states["torch_cuda"] = [
            torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())
        ]
    return states


def _restore_rng_states(states: Dict[str, Any]) -> None:
    """Restore previously captured RNG states."""
    import random

    if "python" in states:
        random.setstate(states["python"])
    if "torch_cpu" in states:
        torch.random.set_rng_state(states["torch_cpu"])
    try:
        import numpy as np

        if "numpy" in states:
            np.random.set_state(states["numpy"])
    except ImportError:
        pass
    if torch.cuda.is_available() and "torch_cuda" in states:
        for i, s in enumerate(states["torch_cuda"]):
            torch.cuda.set_rng_state(s, i)


def _config_hash(config: Dict[str, Any]) -> str:
    """Deterministic SHA-256 of a JSON-serialisable config dict."""
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------


class CheckpointManager:
    """Top-level save / load / resume / cleanup API.

    Organises checkpoints under::

        <run_dir>/checkpoints/phase<N>/
            ckpt_step00000100.pt
            ckpt_step00000200.pt
            ckpt_best_val_loss.pt
            phase_boundary.pt
            ckpt_final.pt
    """

    def __init__(
        self,
        run_dir: Union[str, Path],
        phase: int,
        config: Dict[str, Any],
        *,
        retention_policy: Optional[CheckpointRetentionPolicy] = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.phase = phase
        self.config = config
        self.config_hash = _config_hash(config)
        self.phase_dir = CheckpointNaming.phase_dir(self.run_dir, self.phase)
        self.phase_dir.mkdir(parents=True, exist_ok=True)
        self._saver = AtomicSaver()
        self._naming = CheckpointNaming
        self._retention = retention_policy or CheckpointRetentionPolicy()

        # Track best metrics seen so far for save_best.
        self._best: Dict[str, float] = {}

        logger.info(
            "CheckpointManager initialised: phase=%d  dir=%s", self.phase, self.phase_dir
        )

    # ------------------------------------------------------------------
    # Save helpers
    # ------------------------------------------------------------------

    def _build_periodic_dict(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        rng_states: Optional[Dict[str, Any]],
        global_step: int,
        phase_step: int,
        epoch: int,
        metrics: Optional[Dict[str, float]],
    ) -> Dict[str, Any]:
        sched_sd = None
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            sched_sd = scheduler.state_dict()
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": sched_sd,
            "rng_states": rng_states if rng_states is not None else _capture_rng_states(),
            "global_step": global_step,
            "phase": self.phase,
            "phase_step": phase_step,
            "epoch": epoch,
            "best_metrics": dict(self._best),
            "config_hash": self.config_hash,
            "metrics": metrics or {},
        }

    def save_periodic(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        rng_states: Optional[Dict[str, Any]],
        global_step: int,
        phase_step: int,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Path:
        """Save a periodic (step) checkpoint atomically."""
        ckpt = self._build_periodic_dict(
            model, optimizer, scheduler, rng_states, global_step, phase_step, epoch, metrics
        )
        fname = self._naming.periodic_name(global_step)
        path = self.phase_dir / fname
        self._saver.save_atomic(ckpt, path)
        logger.info("Saved periodic checkpoint: %s (step=%d)", path.name, global_step)
        return path

    def save_best(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        rng_states: Optional[Dict[str, Any]],
        global_step: int,
        phase_step: int,
        epoch: int,
        metrics: Dict[str, float],
        metric_key: str = "val_loss",
        mode: Literal["min", "max"] = "min",
    ) -> Optional[Path]:
        """Save a best-metric checkpoint only if *metric_key* improved.

        Returns the checkpoint path if saved, else ``None``.
        """
        current = metrics.get(metric_key)
        if current is None:
            logger.warning("metric_key %r not found in metrics dict; skipping save_best", metric_key)
            return None

        prev = self._best.get(metric_key)
        improved = False
        if prev is None:
            improved = True
        elif mode == "min" and current < prev:
            improved = True
        elif mode == "max" and current > prev:
            improved = True

        if not improved:
            logger.debug(
                "Metric %s did not improve (prev=%s, cur=%s); skipping save_best",
                metric_key,
                prev,
                current,
            )
            return None

        self._best[metric_key] = current
        ckpt = self._build_periodic_dict(
            model, optimizer, scheduler, rng_states, global_step, phase_step, epoch, metrics
        )
        fname = self._naming.best_name(metric_key)
        path = self.phase_dir / fname
        self._saver.save_atomic(ckpt, path)
        logger.info(
            "Saved best checkpoint for %s=%.6f: %s", metric_key, current, path.name
        )
        return path

    def save_phase_boundary(
        self,
        model: nn.Module,
        config: Optional[Dict[str, Any]] = None,
        compatibility_info: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        run_id: Optional[str] = None,
        best_metrics: Optional[Dict[str, float]] = None,
        total_steps: int = 0,
    ) -> Path:
        """Save a phase-boundary checkpoint (no optimizer / scheduler / RNG).

        Parameters
        ----------
        model : nn.Module
        config : dict, optional
            Config snapshot; defaults to ``self.config``.
        compatibility_info : dict, optional
            Override auto-extracted compatibility dict.
        metadata : dict, optional
            Extra metadata to embed.
        run_id : str, optional
            Run identifier; defaults to a UUID.
        best_metrics : dict, optional
        total_steps : int
        """
        cfg = config or self.config
        rid = run_id or uuid.uuid4().hex[:16]
        ckpt = PhaseBoundaryBuilder.build(
            model,
            cfg,
            self.phase,
            rid,
            best_metrics=best_metrics or dict(self._best),
            total_steps=total_steps,
            extra_metadata=metadata,
        )
        if compatibility_info is not None:
            ckpt["compatibility"].update(compatibility_info)

        fname = self._naming.boundary_name()
        path = self.phase_dir / fname
        self._saver.save_atomic(ckpt, path)
        logger.info("Saved phase boundary: %s", path)
        return path

    def save_final(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        rng_states: Optional[Dict[str, Any]],
        global_step: int,
        phase_step: int,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Path:
        """Save a ``ckpt_final.pt`` periodic checkpoint at the end of a phase."""
        ckpt = self._build_periodic_dict(
            model, optimizer, scheduler, rng_states, global_step, phase_step, epoch, metrics
        )
        path = self.phase_dir / self._naming.final_name()
        self._saver.save_atomic(ckpt, path)
        logger.info("Saved final checkpoint: %s (step=%d)", path.name, global_step)
        return path

    # ------------------------------------------------------------------
    # Load helpers
    # ------------------------------------------------------------------

    def load_checkpoint(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load and validate a checkpoint from *path*."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        # Determine type.
        if "source_phase" in ckpt:
            errors = validate_checkpoint(ckpt, expected_type="boundary")
        else:
            errors = validate_checkpoint(ckpt, expected_type="periodic")
        if errors:
            logger.warning("Checkpoint validation warnings for %s: %s", path, errors)
        return ckpt

    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Load the most recent periodic checkpoint in the phase directory.

        Checks ``ckpt_final.pt`` first, then ``ckpt_step*.pt`` by step.
        """
        final = self.phase_dir / self._naming.final_name()
        if final.exists():
            return self.load_checkpoint(final)
        periodics = sorted(self.phase_dir.glob("ckpt_step*.pt"))
        if not periodics:
            return None
        return self.load_checkpoint(periodics[-1])

    def load_best(self, metric_key: str = "val_loss") -> Optional[Dict[str, Any]]:
        """Load ``ckpt_best_{metric_key}.pt``."""
        path = self.phase_dir / self._naming.best_name(metric_key)
        if not path.exists():
            return None
        return self.load_checkpoint(path)

    def load_phase_boundary(self, phase: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Load ``phase_boundary.pt`` from the given (or current) phase directory."""
        ph = phase if phase is not None else self.phase
        pdir = CheckpointNaming.phase_dir(self.run_dir, ph)
        path = pdir / self._naming.boundary_name()
        if not path.exists():
            return None
        return self.load_checkpoint(path)

    # ------------------------------------------------------------------
    # Resume helper
    # ------------------------------------------------------------------

    def resume_from_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        checkpoint: Dict[str, Any],
        strict_load: bool = True,
    ) -> ResumeInfo:
        """Restore all state from a previously loaded checkpoint dict.

        Restores model weights, optimizer state, scheduler state, and RNG
        states.
        """
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict_load)

        if "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if (
            scheduler is not None
            and "scheduler_state_dict" in checkpoint
            and checkpoint["scheduler_state_dict"] is not None
        ):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        rng = checkpoint.get("rng_states")
        if rng is not None:
            _restore_rng_states(rng)

        # Restore tracked best metrics.
        self._best = dict(checkpoint.get("best_metrics", {}))

        return ResumeInfo(
            checkpoint_path=Path("(from dict)"),
            global_step=checkpoint.get("global_step", 0),
            phase_step=checkpoint.get("phase_step", 0),
            epoch=checkpoint.get("epoch", 0),
            was_best=False,
            metrics=dict(self._best),
        )

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """Return sorted list of checkpoint metadata dicts in phase directory.

        Each entry has keys: ``path``, ``name``, ``type``, ``step`` (if periodic),
        ``size_mb``, ``mtime``.
        """
        results: List[Dict[str, Any]] = []
        if not self.phase_dir.exists():
            return results

        for p in sorted(self.phase_dir.glob("*.pt")):
            entry: Dict[str, Any] = {
                "path": p,
                "name": p.name,
                "size_mb": p.stat().st_size / (1024 * 1024),
                "mtime": p.stat().st_mtime,
            }
            step = self._naming.parse_step_from_name(p.name)
            if step is not None:
                entry["type"] = "periodic"
                entry["step"] = step
            elif p.name.startswith("ckpt_best_"):
                entry["type"] = "best"
            elif p.name == "phase_boundary.pt":
                entry["type"] = "boundary"
            elif p.name == "ckpt_final.pt":
                entry["type"] = "final"
            else:
                entry["type"] = "unknown"
            results.append(entry)

        # Sort periodic checkpoints by step, others by name.
        results.sort(key=lambda e: (e.get("step", float("inf")), e["name"]))
        return results

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(
        self,
        keep_best: bool = True,
        keep_boundary: bool = True,
        keep_last: bool = True,
        keep_n: int = 3,
    ) -> List[Path]:
        """Remove old periodic checkpoints, respecting retention rules.

        Parameters
        ----------
        keep_best : bool
            Keep ``ckpt_best_*.pt`` files (always True by default).
        keep_boundary : bool
            Keep ``phase_boundary.pt`` (always True by default).
        keep_last : bool
            Keep the most recent ``ckpt_step*.pt``.
        keep_n : int
            Number of most-recent periodic checkpoints to keep.

        Returns
        -------
        list[Path]
            Paths that were deleted.
        """
        policy = CheckpointRetentionPolicy(
            keep_best=keep_best,
            keep_boundary=keep_boundary,
            keep_last=keep_last,
            max_checkpoints=keep_n,
        )
        return policy.apply(self.phase_dir)


# ===================================================================
#  SELF-TESTS
# ===================================================================

if __name__ == "__main__":
    import random
    import sys
    import traceback

    # ------------------------------------------------------------------
    # Tiny mock model for testing
    # ------------------------------------------------------------------

    class _MockModel(nn.Module):
        def __init__(self, dim: int = 32) -> None:
            super().__init__()
            self.fc1 = nn.Linear(dim, dim)
            self.fc2 = nn.Linear(dim, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc2(torch.relu(self.fc1(x)))

    # ------------------------------------------------------------------
    # Test harness
    # ------------------------------------------------------------------

    _pass_count = 0
    _fail_count = 0
    _test_names: List[str] = []

    def _run_test(name: str, fn):
        global _pass_count, _fail_count
        _test_names.append(name)
        try:
            fn()
            _pass_count += 1
            print(f"  PASS  {name}")
        except Exception as exc:
            _fail_count += 1
            print(f"  FAIL  {name}: {exc}")
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Temp directory for all tests
    # ------------------------------------------------------------------

    _tmpdir = Path(tempfile.mkdtemp(prefix="ckpt_test_"))

    def _fresh_dir(name: str) -> Path:
        d = _tmpdir / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ===== 1. Atomic save round-trip =====

    def test_atomic_save_roundtrip():
        d = _fresh_dir("atomic_rt")
        data = {"key": torch.tensor([1.0, 2.0, 3.0]), "step": 42}
        path = AtomicSaver.save_atomic(data, d / "test.pt")
        assert path.exists(), "File should exist after atomic save"
        loaded = torch.load(str(path), map_location="cpu", weights_only=False)
        assert torch.equal(loaded["key"], data["key"])
        assert loaded["step"] == 42

    _run_test("atomic_save_roundtrip", test_atomic_save_roundtrip)

    # ===== 2. Atomic save creates no leftover temp files =====

    def test_atomic_no_temp_files():
        d = _fresh_dir("atomic_no_tmp")
        AtomicSaver.save_atomic({"a": 1}, d / "clean.pt")
        files = list(d.iterdir())
        assert len(files) == 1, f"Expected 1 file, got {len(files)}: {files}"
        assert files[0].name == "clean.pt"

    _run_test("atomic_no_temp_files", test_atomic_no_temp_files)

    # ===== 3. Atomic JSON save =====

    def test_atomic_json_save():
        d = _fresh_dir("atomic_json")
        AtomicSaver.save_atomic_json({"hello": "world"}, d / "meta.json")
        with open(d / "meta.json") as f:
            data = json.load(f)
        assert data["hello"] == "world"

    _run_test("atomic_json_save", test_atomic_json_save)

    # ===== 4. Naming: periodic =====

    def test_naming_periodic():
        assert CheckpointNaming.periodic_name(0) == "ckpt_step00000000.pt"
        assert CheckpointNaming.periodic_name(42) == "ckpt_step00000042.pt"
        assert CheckpointNaming.periodic_name(12345678) == "ckpt_step12345678.pt"

    _run_test("naming_periodic", test_naming_periodic)

    # ===== 5. Naming: best =====

    def test_naming_best():
        assert CheckpointNaming.best_name("val_loss") == "ckpt_best_val_loss.pt"
        assert CheckpointNaming.best_name("val/acc") == "ckpt_best_val_acc.pt"

    _run_test("naming_best", test_naming_best)

    # ===== 6. Naming: boundary / final =====

    def test_naming_boundary_final():
        assert CheckpointNaming.boundary_name() == "phase_boundary.pt"
        assert CheckpointNaming.final_name() == "ckpt_final.pt"

    _run_test("naming_boundary_final", test_naming_boundary_final)

    # ===== 7. Naming: parse step =====

    def test_naming_parse_step():
        assert CheckpointNaming.parse_step_from_name("ckpt_step00000042.pt") == 42
        assert CheckpointNaming.parse_step_from_name("ckpt_step12345678.pt") == 12345678
        assert CheckpointNaming.parse_step_from_name("ckpt_best_val_loss.pt") is None
        assert CheckpointNaming.parse_step_from_name("random.pt") is None

    _run_test("naming_parse_step", test_naming_parse_step)

    # ===== 8. Naming: phase_dir =====

    def test_naming_phase_dir():
        p = CheckpointNaming.phase_dir("/tmp/run1", 3)
        assert p == Path("/tmp/run1/checkpoints/phase3")

    _run_test("naming_phase_dir", test_naming_phase_dir)

    # ===== 9. Schema validation: valid periodic =====

    def test_schema_valid_periodic():
        ckpt = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "scheduler_state_dict": None,
            "rng_states": {},
            "global_step": 0,
            "phase": 1,
            "phase_step": 0,
            "epoch": 0,
            "best_metrics": {},
            "config_hash": "abc",
        }
        errors = validate_checkpoint(ckpt, "periodic")
        assert errors == [], f"Expected no errors: {errors}"

    _run_test("schema_valid_periodic", test_schema_valid_periodic)

    # ===== 10. Schema validation: missing keys =====

    def test_schema_missing_keys():
        ckpt = {"schema_version": CHECKPOINT_SCHEMA_VERSION, "model_state_dict": {}}
        errors = validate_checkpoint(ckpt, "periodic")
        assert len(errors) == 1
        assert "Missing keys" in errors[0]

    _run_test("schema_missing_keys", test_schema_missing_keys)

    # ===== 11. Schema validation: wrong version =====

    def test_schema_wrong_version():
        ckpt = {
            "schema_version": "0.1",
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "scheduler_state_dict": None,
            "rng_states": {},
            "global_step": 0,
            "phase": 1,
            "phase_step": 0,
            "epoch": 0,
            "best_metrics": {},
            "config_hash": "abc",
        }
        errors = validate_checkpoint(ckpt, "periodic")
        assert any("version mismatch" in e for e in errors), f"Expected version error: {errors}"

    _run_test("schema_wrong_version", test_schema_wrong_version)

    # ===== 12. Schema validation: valid boundary =====

    def test_schema_valid_boundary():
        ckpt = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "source_phase": 1,
            "target_phase": 2,
            "source_run_id": "abc123",
            "model_state_dict": {},
            "config_snapshot": {},
            "feature_flags": {},
            "compatibility": {},
            "metadata": {},
        }
        errors = validate_checkpoint(ckpt, "boundary")
        assert errors == [], f"Expected no errors: {errors}"

    _run_test("schema_valid_boundary", test_schema_valid_boundary)

    # ===== 13. Schema validation: boundary missing keys =====

    def test_schema_boundary_missing():
        ckpt = {"schema_version": CHECKPOINT_SCHEMA_VERSION, "source_phase": 1}
        errors = validate_checkpoint(ckpt, "boundary")
        assert len(errors) == 1
        assert "Missing keys" in errors[0]

    _run_test("schema_boundary_missing", test_schema_boundary_missing)

    # ===== 14. CheckpointManager: save_periodic round-trip =====

    def test_manager_save_periodic_roundtrip():
        d = _fresh_dir("mgr_periodic")
        model = _MockModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        cfg = {"workspace_dim": 4096}
        mgr = CheckpointManager(d, phase=1, config=cfg)

        path = mgr.save_periodic(model, opt, None, None, global_step=100, phase_step=50, epoch=1)
        assert path.exists()
        assert path.name == "ckpt_step00000100.pt"

        ckpt = mgr.load_checkpoint(path)
        assert ckpt["global_step"] == 100
        assert ckpt["phase_step"] == 50
        assert ckpt["epoch"] == 1
        assert ckpt["phase"] == 1
        assert ckpt["schema_version"] == CHECKPOINT_SCHEMA_VERSION

        # Model weights should round-trip exactly.
        model2 = _MockModel()
        model2.load_state_dict(ckpt["model_state_dict"])
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.equal(p1, p2)

    _run_test("manager_save_periodic_roundtrip", test_manager_save_periodic_roundtrip)

    # ===== 15. CheckpointManager: save_best only when improved =====

    def test_manager_save_best_only_improved():
        d = _fresh_dir("mgr_best")
        model = _MockModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        cfg = {"workspace_dim": 4096}
        mgr = CheckpointManager(d, phase=1, config=cfg)

        # First call: should always save (no previous best).
        r1 = mgr.save_best(model, opt, None, None, 100, 50, 1, {"val_loss": 1.0})
        assert r1 is not None, "First save_best should always save"

        # Second call with worse metric: should NOT save.
        r2 = mgr.save_best(model, opt, None, None, 200, 100, 2, {"val_loss": 1.5})
        assert r2 is None, "Worse metric should not trigger save"

        # Third call with better metric: should save.
        r3 = mgr.save_best(model, opt, None, None, 300, 150, 3, {"val_loss": 0.5})
        assert r3 is not None, "Better metric should trigger save"

    _run_test("manager_save_best_only_improved", test_manager_save_best_only_improved)

    # ===== 16. save_best with mode="max" =====

    def test_manager_save_best_max_mode():
        d = _fresh_dir("mgr_best_max")
        model = _MockModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        mgr = CheckpointManager(d, phase=1, config={})

        r1 = mgr.save_best(model, opt, None, None, 100, 50, 1, {"val_acc": 0.5}, metric_key="val_acc", mode="max")
        assert r1 is not None

        r2 = mgr.save_best(model, opt, None, None, 200, 100, 2, {"val_acc": 0.3}, metric_key="val_acc", mode="max")
        assert r2 is None, "Lower acc should not save in max mode"

        r3 = mgr.save_best(model, opt, None, None, 300, 150, 3, {"val_acc": 0.9}, metric_key="val_acc", mode="max")
        assert r3 is not None, "Higher acc should save in max mode"

    _run_test("manager_save_best_max_mode", test_manager_save_best_max_mode)

    # ===== 17. save_best with missing metric key =====

    def test_manager_save_best_missing_key():
        d = _fresh_dir("mgr_best_miss")
        model = _MockModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        mgr = CheckpointManager(d, phase=1, config={})

        r = mgr.save_best(model, opt, None, None, 100, 50, 1, {"train_loss": 1.0}, metric_key="val_loss")
        assert r is None, "Missing metric key should return None"

    _run_test("manager_save_best_missing_key", test_manager_save_best_missing_key)

    # ===== 18. Phase boundary creation (no optimizer state) =====

    def test_manager_phase_boundary():
        d = _fresh_dir("mgr_boundary")
        model = _MockModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        cfg = {"workspace_dim": 4096, "use_snn": True, "use_htm": False}
        mgr = CheckpointManager(d, phase=2, config=cfg)

        path = mgr.save_phase_boundary(model, total_steps=5000)
        assert path.exists()
        assert path.name == "phase_boundary.pt"

        ckpt = mgr.load_checkpoint(path)
        assert ckpt["source_phase"] == 2
        assert ckpt["target_phase"] == 3
        assert "optimizer_state_dict" not in ckpt
        assert "scheduler_state_dict" not in ckpt
        assert "rng_states" not in ckpt
        assert ckpt["compatibility"]["workspace_dim"] == 4096
        assert ckpt["feature_flags"]["use_snn"] is True
        assert ckpt["feature_flags"]["use_htm"] is False
        assert ckpt["metadata"]["training_steps"] == 5000

    _run_test("manager_phase_boundary", test_manager_phase_boundary)

    # ===== 19. load_latest finds most recent step =====

    def test_manager_load_latest():
        d = _fresh_dir("mgr_latest")
        model = _MockModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        mgr = CheckpointManager(d, phase=1, config={})

        mgr.save_periodic(model, opt, None, None, 100, 50, 1)
        mgr.save_periodic(model, opt, None, None, 200, 100, 2)
        mgr.save_periodic(model, opt, None, None, 300, 150, 3)

        ckpt = mgr.load_latest()
        assert ckpt is not None
        assert ckpt["global_step"] == 300

    _run_test("manager_load_latest", test_manager_load_latest)

    # ===== 20. load_latest prefers ckpt_final.pt =====

    def test_manager_load_latest_prefers_final():
        d = _fresh_dir("mgr_latest_final")
        model = _MockModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        mgr = CheckpointManager(d, phase=1, config={})

        mgr.save_periodic(model, opt, None, None, 100, 50, 1)
        mgr.save_final(model, opt, None, None, 200, 100, 2)

        ckpt = mgr.load_latest()
        assert ckpt is not None
        assert ckpt["global_step"] == 200

    _run_test("manager_load_latest_prefers_final", test_manager_load_latest_prefers_final)

    # ===== 21. load_latest returns None for empty dir =====

    def test_manager_load_latest_empty():
        d = _fresh_dir("mgr_latest_empty")
        mgr = CheckpointManager(d, phase=1, config={})
        assert mgr.load_latest() is None

    _run_test("manager_load_latest_empty", test_manager_load_latest_empty)

    # ===== 22. load_best =====

    def test_manager_load_best():
        d = _fresh_dir("mgr_load_best")
        model = _MockModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        mgr = CheckpointManager(d, phase=1, config={})

        mgr.save_best(model, opt, None, None, 100, 50, 1, {"val_loss": 0.5})
        ckpt = mgr.load_best("val_loss")
        assert ckpt is not None
        assert ckpt["metrics"]["val_loss"] == 0.5

    _run_test("manager_load_best", test_manager_load_best)

    # ===== 23. load_best returns None for non-existent =====

    def test_manager_load_best_nonexistent():
        d = _fresh_dir("mgr_load_best_ne")
        mgr = CheckpointManager(d, phase=1, config={})
        assert mgr.load_best("val_loss") is None

    _run_test("manager_load_best_nonexistent", test_manager_load_best_nonexistent)

    # ===== 24. load_phase_boundary =====

    def test_manager_load_phase_boundary():
        d = _fresh_dir("mgr_load_bnd")
        model = _MockModel()
        cfg = {"workspace_dim": 2048}
        mgr = CheckpointManager(d, phase=3, config=cfg)
        mgr.save_phase_boundary(model)

        ckpt = mgr.load_phase_boundary(phase=3)
        assert ckpt is not None
        assert ckpt["source_phase"] == 3

    _run_test("manager_load_phase_boundary", test_manager_load_phase_boundary)

    # ===== 25. load_phase_boundary returns None for missing =====

    def test_manager_load_phase_boundary_missing():
        d = _fresh_dir("mgr_load_bnd_miss")
        mgr = CheckpointManager(d, phase=1, config={})
        assert mgr.load_phase_boundary(phase=99) is None

    _run_test("manager_load_phase_boundary_missing", test_manager_load_phase_boundary_missing)

    # ===== 26. resume_from_checkpoint restores all state =====

    def test_manager_resume_restores_state():
        d = _fresh_dir("mgr_resume")
        model = _MockModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        cfg = {"workspace_dim": 4096}
        mgr = CheckpointManager(d, phase=1, config=cfg)

        # Do a training step to change optimizer state.
        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        path = mgr.save_periodic(model, opt, None, None, 500, 250, 5, {"val_loss": 0.3})
        ckpt = mgr.load_checkpoint(path)

        # Create fresh model/opt and resume.
        model2 = _MockModel()
        opt2 = torch.optim.SGD(model2.parameters(), lr=0.01)
        info = mgr.resume_from_checkpoint(model2, opt2, None, ckpt)

        assert info.global_step == 500
        assert info.phase_step == 250
        assert info.epoch == 5

        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.equal(p1, p2), "Model params should match after resume"

        # Optimizer state should also match.
        for pg1, pg2 in zip(opt.param_groups, opt2.param_groups):
            assert pg1["lr"] == pg2["lr"]

    _run_test("manager_resume_restores_state", test_manager_resume_restores_state)

    # ===== 27. resume restores best metrics tracking =====

    def test_manager_resume_restores_best_tracking():
        d = _fresh_dir("mgr_resume_best")
        model = _MockModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        mgr = CheckpointManager(d, phase=1, config={})

        # Save best, then periodic.
        mgr.save_best(model, opt, None, None, 100, 50, 1, {"val_loss": 0.5})
        path = mgr.save_periodic(model, opt, None, None, 200, 100, 2, {"val_loss": 0.7})

        # Resume into new manager.
        mgr2 = CheckpointManager(d, phase=1, config={})
        ckpt = mgr2.load_checkpoint(path)
        mgr2.resume_from_checkpoint(model, opt, None, ckpt)

        # Should NOT save because 0.7 > 0.5 (the tracked best).
        r = mgr2.save_best(model, opt, None, None, 300, 150, 3, {"val_loss": 0.7})
        assert r is None, "Should not save when metric hasn't improved over resumed best"

        # SHOULD save for genuinely better metric.
        r2 = mgr2.save_best(model, opt, None, None, 400, 200, 4, {"val_loss": 0.3})
        assert r2 is not None, "Should save when metric improves over resumed best"

    _run_test("manager_resume_restores_best_tracking", test_manager_resume_restores_best_tracking)

    # ===== 28. list_checkpoints =====

    def test_manager_list_checkpoints():
        d = _fresh_dir("mgr_list")
        model = _MockModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        mgr = CheckpointManager(d, phase=1, config={})

        mgr.save_periodic(model, opt, None, None, 100, 50, 1)
        mgr.save_periodic(model, opt, None, None, 200, 100, 2)
        mgr.save_best(model, opt, None, None, 200, 100, 2, {"val_loss": 0.5})
        mgr.save_phase_boundary(model)

        entries = mgr.list_checkpoints()
        assert len(entries) == 4
        types = {e["type"] for e in entries}
        assert "periodic" in types
        assert "best" in types
        assert "boundary" in types

    _run_test("manager_list_checkpoints", test_manager_list_checkpoints)

    # ===== 29. cleanup keeps best/boundary/last, removes old =====

    def test_manager_cleanup():
        d = _fresh_dir("mgr_cleanup")
        model = _MockModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        mgr = CheckpointManager(d, phase=1, config={})

        for step in range(100, 1100, 100):
            mgr.save_periodic(model, opt, None, None, step, step // 2, step // 100)

        mgr.save_best(model, opt, None, None, 1000, 500, 10, {"val_loss": 0.1})
        mgr.save_phase_boundary(model)

        # 10 periodic + 1 best + 1 boundary = 12 files.
        before = list(mgr.phase_dir.glob("*.pt"))
        assert len(before) == 12

        deleted = mgr.cleanup(keep_n=2)
        assert len(deleted) > 0

        after = list(mgr.phase_dir.glob("*.pt"))
        # Best + boundary + at most 2 periodic should remain.
        remaining_names = {p.name for p in after}
        assert "ckpt_best_val_loss.pt" in remaining_names, "Best should be kept"
        assert "phase_boundary.pt" in remaining_names, "Boundary should be kept"
        # At most 2 periodic + best + boundary.
        periodic_remaining = [p for p in after if p.name.startswith("ckpt_step")]
        assert len(periodic_remaining) <= 2, f"Expected <=2 periodic, got {len(periodic_remaining)}"

    _run_test("manager_cleanup", test_manager_cleanup)

    # ===== 30. cleanup preserves ckpt_final.pt =====

    def test_manager_cleanup_preserves_final():
        d = _fresh_dir("mgr_cleanup_final")
        model = _MockModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        mgr = CheckpointManager(d, phase=1, config={})

        for step in range(100, 600, 100):
            mgr.save_periodic(model, opt, None, None, step, step // 2, step // 100)
        mgr.save_final(model, opt, None, None, 600, 300, 6)

        mgr.cleanup(keep_n=1)
        remaining = {p.name for p in mgr.phase_dir.glob("*.pt")}
        assert "ckpt_final.pt" in remaining, "ckpt_final.pt should be preserved"

    _run_test("manager_cleanup_preserves_final", test_manager_cleanup_preserves_final)

    # ===== 31. PhaseBoundaryBuilder: build + validate =====

    def test_boundary_builder_build():
        model = _MockModel()
        cfg = {"workspace_dim": 4096, "vocab_size": 32000, "use_snn": True, "use_engram": False}
        ckpt = PhaseBoundaryBuilder.build(model, cfg, phase=1, run_id="run123", total_steps=10000)

        errors = validate_checkpoint(ckpt, "boundary")
        assert errors == [], f"Boundary should be valid: {errors}"
        assert ckpt["source_phase"] == 1
        assert ckpt["target_phase"] == 2
        assert ckpt["compatibility"]["workspace_dim"] == 4096
        assert ckpt["feature_flags"]["use_snn"] is True
        assert ckpt["feature_flags"]["use_engram"] is False
        assert ckpt["metadata"]["training_steps"] == 10000

    _run_test("boundary_builder_build", test_boundary_builder_build)

    # ===== 32. PhaseBoundaryBuilder: compatibility extraction =====

    def test_boundary_builder_compatibility():
        cfg = {
            "workspace_dim": 2048,
            "vocab_size": 50000,
            "num_encoders": 6,
            "snn_hidden": 512,
            "htm_columns": 1024,
        }
        compat = PhaseBoundaryBuilder._extract_compatibility(cfg)
        assert compat["workspace_dim"] == 2048
        assert compat["vocab_size"] == 50000
        assert compat["num_encoders"] == 6
        assert compat["snn_hidden"] == 512
        assert compat["htm_columns"] == 1024

    _run_test("boundary_builder_compatibility", test_boundary_builder_compatibility)

    # ===== 33. PhaseBoundaryBuilder: feature flag extraction =====

    def test_boundary_builder_flags():
        cfg = {"use_snn": False, "use_htm": True}
        flags = PhaseBoundaryBuilder._extract_feature_flags(cfg)
        assert flags["use_snn"] is False
        assert flags["use_htm"] is True
        # Defaults to True for missing flags.
        assert flags["use_workspace"] is True

    _run_test("boundary_builder_flags", test_boundary_builder_flags)

    # ===== 34. PhaseBoundaryBuilder: extract_model_subset =====

    def test_boundary_builder_subset():
        sd = {
            "core.weight": torch.randn(4),
            "encoder.bias": torch.randn(4),
            "temporal.weight": torch.randn(4),
        }
        subset = PhaseBoundaryBuilder.extract_model_subset(sd, phase=1)
        assert "core.weight" in subset
        assert "encoder.bias" in subset
        assert "temporal.weight" not in subset

    _run_test("boundary_builder_subset", test_boundary_builder_subset)

    # ===== 35. extract_model_subset fallback for unknown phase =====

    def test_boundary_builder_subset_unknown():
        sd = {"a": torch.randn(4), "b": torch.randn(4)}
        subset = PhaseBoundaryBuilder.extract_model_subset(sd, phase=99)
        assert subset is sd, "Unknown phase should return full state dict"

    _run_test("boundary_builder_subset_unknown", test_boundary_builder_subset_unknown)

    # ===== 36. extract_model_subset fallback when no keys match =====

    def test_boundary_builder_subset_no_match():
        sd = {"unrelated.weight": torch.randn(4)}
        subset = PhaseBoundaryBuilder.extract_model_subset(sd, phase=1)
        assert subset is sd, "No matching keys should return full state dict"

    _run_test("boundary_builder_subset_no_match", test_boundary_builder_subset_no_match)

    # ===== 37. ResumeManager: find_resume_point =====

    def test_resume_manager_find():
        d = _fresh_dir("resume_find")
        model = _MockModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        mgr = CheckpointManager(d, phase=2, config={})
        mgr.save_periodic(model, opt, None, None, 100, 50, 1)
        mgr.save_periodic(model, opt, None, None, 200, 100, 2)

        found = ResumeManager.find_resume_point(d, phase=2)
        assert found is not None
        assert found.name == "ckpt_step00000200.pt"

    _run_test("resume_manager_find", test_resume_manager_find)

    # ===== 38. ResumeManager: find across phases =====

    def test_resume_manager_find_across_phases():
        d = _fresh_dir("resume_across")
        model = _MockModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)

        mgr1 = CheckpointManager(d, phase=1, config={})
        mgr1.save_periodic(model, opt, None, None, 100, 50, 1)

        mgr2 = CheckpointManager(d, phase=2, config={})
        mgr2.save_periodic(model, opt, None, None, 500, 250, 5)

        # Without specifying phase, should find latest across all phases.
        found = ResumeManager.find_resume_point(d)
        assert found is not None
        assert "phase2" in str(found)

    _run_test("resume_manager_find_across_phases", test_resume_manager_find_across_phases)

    # ===== 39. ResumeManager: find returns None for empty =====

    def test_resume_manager_find_empty():
        d = _fresh_dir("resume_empty")
        assert ResumeManager.find_resume_point(d) is None

    _run_test("resume_manager_find_empty", test_resume_manager_find_empty)

    # ===== 40. ResumeManager: find prefers ckpt_final =====

    def test_resume_manager_find_prefers_final():
        d = _fresh_dir("resume_final")
        model = _MockModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        mgr = CheckpointManager(d, phase=1, config={})
        mgr.save_periodic(model, opt, None, None, 100, 50, 1)
        mgr.save_final(model, opt, None, None, 200, 100, 2)

        found = ResumeManager.find_resume_point(d, phase=1)
        assert found is not None
        assert found.name == "ckpt_final.pt"

    _run_test("resume_manager_find_prefers_final", test_resume_manager_find_prefers_final)

    # ===== 41. ResumeManager: validate compatibility =====

    def test_resume_manager_validate():
        cfg = {"lr": 0.01, "dim": 64}
        ckpt = {"config_hash": _config_hash(cfg), "model_state_dict": {"a": torch.randn(4)}}

        # Same config: no warnings.
        warns = ResumeManager.validate_resume_compatibility(ckpt, cfg)
        assert warns == []

        # Different config: warning.
        warns2 = ResumeManager.validate_resume_compatibility(ckpt, {"lr": 0.001})
        assert any("hash mismatch" in w for w in warns2)

    _run_test("resume_manager_validate", test_resume_manager_validate)

    # ===== 42. ResumeManager: validate empty model_state_dict =====

    def test_resume_manager_validate_empty():
        ckpt = {"config_hash": "abc", "model_state_dict": {}}
        warns = ResumeManager.validate_resume_compatibility(ckpt, {})
        assert any("empty" in w for w in warns)

    _run_test("resume_manager_validate_empty", test_resume_manager_validate_empty)

    # ===== 43. ResumeManager: full resume flow =====

    def test_resume_manager_full_flow():
        d = _fresh_dir("resume_full")
        model = _MockModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        cfg = {"dim": 32}
        mgr = CheckpointManager(d, phase=1, config=cfg)

        # Train briefly.
        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

        mgr.save_periodic(model, opt, None, None, 42, 21, 1)

        # Full resume.
        model2 = _MockModel()
        opt2 = torch.optim.SGD(model2.parameters(), lr=0.01)
        info = ResumeManager.resume(model2, opt2, None, d, phase=1, current_config=cfg)

        assert info is not None
        assert info.global_step == 42
        assert info.phase_step == 21
        assert info.epoch == 1

        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.equal(p1, p2)

    _run_test("resume_manager_full_flow", test_resume_manager_full_flow)

    # ===== 44. ResumeManager: resume returns None for no checkpoint =====

    def test_resume_manager_returns_none():
        d = _fresh_dir("resume_none")
        model = _MockModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        info = ResumeManager.resume(model, opt, None, d, phase=1)
        assert info is None

    _run_test("resume_manager_returns_none", test_resume_manager_returns_none)

    # ===== 45. CheckpointRetentionPolicy: get_deletable =====

    def test_retention_get_deletable():
        d = _fresh_dir("retention")
        model = _MockModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        mgr = CheckpointManager(d, phase=1, config={})

        for step in range(100, 1100, 100):
            mgr.save_periodic(model, opt, None, None, step, step // 2, step // 100)

        policy = CheckpointRetentionPolicy(max_checkpoints=3)
        deletable = policy.get_deletable(mgr.phase_dir)
        # 10 periodic - 3 kept = 7 deletable.
        assert len(deletable) == 7
        # Latest should NOT be in deletable.
        deletable_names = {p.name for p in deletable}
        assert "ckpt_step00001000.pt" not in deletable_names

    _run_test("retention_get_deletable", test_retention_get_deletable)

    # ===== 46. Retention: protected files never deleted =====

    def test_retention_protected():
        d = _fresh_dir("retention_prot")
        model = _MockModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        mgr = CheckpointManager(d, phase=1, config={})

        for step in range(100, 600, 100):
            mgr.save_periodic(model, opt, None, None, step, step // 2, step // 100)
        mgr.save_best(model, opt, None, None, 500, 250, 5, {"val_loss": 0.1})
        mgr.save_phase_boundary(model)
        mgr.save_final(model, opt, None, None, 600, 300, 6)

        policy = CheckpointRetentionPolicy(max_checkpoints=1)
        deletable = policy.get_deletable(mgr.phase_dir)
        deletable_names = {p.name for p in deletable}

        assert "ckpt_best_val_loss.pt" not in deletable_names
        assert "phase_boundary.pt" not in deletable_names
        assert "ckpt_final.pt" not in deletable_names

    _run_test("retention_protected", test_retention_protected)

    # ===== 47. Retention: keep_every_n =====

    def test_retention_keep_every_n():
        d = _fresh_dir("retention_every_n")
        model = _MockModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        mgr = CheckpointManager(d, phase=1, config={})

        for step in range(100, 1100, 100):
            mgr.save_periodic(model, opt, None, None, step, step // 2, step // 100)

        policy = CheckpointRetentionPolicy(keep_every_n=500, max_checkpoints=0)
        deletable = policy.get_deletable(mgr.phase_dir)
        kept_names = {p.name for p in mgr.phase_dir.glob("ckpt_step*.pt")} - {p.name for p in deletable}
        # Steps 500 and 1000 should be kept (divisible by 500), plus last (1000).
        assert "ckpt_step00000500.pt" in kept_names
        assert "ckpt_step00001000.pt" in kept_names

    _run_test("retention_keep_every_n", test_retention_keep_every_n)

    # ===== 48. Retention: apply actually deletes files =====

    def test_retention_apply_deletes():
        d = _fresh_dir("retention_apply")
        model = _MockModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        mgr = CheckpointManager(d, phase=1, config={})

        for step in range(100, 600, 100):
            mgr.save_periodic(model, opt, None, None, step, step // 2, step // 100)

        before_count = len(list(mgr.phase_dir.glob("ckpt_step*.pt")))
        assert before_count == 5

        policy = CheckpointRetentionPolicy(max_checkpoints=2)
        deleted = policy.apply(mgr.phase_dir)
        assert len(deleted) > 0

        after_count = len(list(mgr.phase_dir.glob("ckpt_step*.pt")))
        assert after_count <= 2

    _run_test("retention_apply_deletes", test_retention_apply_deletes)

    # ===== 49. Retention: empty directory =====

    def test_retention_empty_dir():
        d = _fresh_dir("retention_empty")
        d.mkdir(parents=True, exist_ok=True)
        policy = CheckpointRetentionPolicy()
        assert policy.get_deletable(d) == []
        assert policy.apply(d) == []

    _run_test("retention_empty_dir", test_retention_empty_dir)

    # ===== 50. Retention: non-existent directory =====

    def test_retention_nonexistent():
        policy = CheckpointRetentionPolicy()
        assert policy.get_deletable("/nonexistent/path") == []

    _run_test("retention_nonexistent", test_retention_nonexistent)

    # ===== 51. RNG state capture and restore =====

    def test_rng_capture_restore():
        torch.manual_seed(42)
        random.seed(42)

        states = _capture_rng_states()
        # Generate some random numbers.
        vals1 = [torch.randn(1).item() for _ in range(5)]
        py_vals1 = [random.random() for _ in range(5)]

        # Restore and regenerate: should match.
        _restore_rng_states(states)
        vals2 = [torch.randn(1).item() for _ in range(5)]
        py_vals2 = [random.random() for _ in range(5)]

        assert vals1 == vals2, "Torch RNG should be reproducible after restore"
        assert py_vals1 == py_vals2, "Python RNG should be reproducible after restore"

    _run_test("rng_capture_restore", test_rng_capture_restore)

    # ===== 52. Config hash determinism =====

    def test_config_hash_deterministic():
        cfg1 = {"a": 1, "b": [2, 3], "c": {"d": 4}}
        cfg2 = {"c": {"d": 4}, "a": 1, "b": [2, 3]}  # Same, different order.
        assert _config_hash(cfg1) == _config_hash(cfg2)

        cfg3 = {"a": 2, "b": [2, 3], "c": {"d": 4}}
        assert _config_hash(cfg1) != _config_hash(cfg3)

    _run_test("config_hash_deterministic", test_config_hash_deterministic)

    # ===== 53. save_periodic with scheduler =====

    def test_manager_save_with_scheduler():
        d = _fresh_dir("mgr_sched")
        model = _MockModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)

        for _ in range(15):
            sched.step()

        mgr = CheckpointManager(d, phase=1, config={})
        path = mgr.save_periodic(model, opt, sched, None, 15, 15, 1)
        ckpt = mgr.load_checkpoint(path)

        assert ckpt["scheduler_state_dict"] is not None

        # Restore scheduler.
        opt2 = torch.optim.SGD(model.parameters(), lr=0.1)
        sched2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=10, gamma=0.5)
        sched2.load_state_dict(ckpt["scheduler_state_dict"])

        assert sched.get_last_lr() == sched2.get_last_lr()

    _run_test("manager_save_with_scheduler", test_manager_save_with_scheduler)

    # ===== 54. Multiple best metrics tracked independently =====

    def test_manager_multiple_best_metrics():
        d = _fresh_dir("mgr_multi_best")
        model = _MockModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        mgr = CheckpointManager(d, phase=1, config={})

        # Save best for val_loss.
        r1 = mgr.save_best(model, opt, None, None, 100, 50, 1, {"val_loss": 0.5, "val_acc": 0.8}, metric_key="val_loss")
        assert r1 is not None

        # Save best for val_acc (independent tracking).
        r2 = mgr.save_best(model, opt, None, None, 100, 50, 1, {"val_loss": 0.5, "val_acc": 0.8}, metric_key="val_acc", mode="max")
        assert r2 is not None

        # Check both files exist.
        assert (mgr.phase_dir / "ckpt_best_val_loss.pt").exists()
        assert (mgr.phase_dir / "ckpt_best_val_acc.pt").exists()

    _run_test("manager_multiple_best_metrics", test_manager_multiple_best_metrics)

    # ===== 55. Checkpoint load with FileNotFoundError =====

    def test_manager_load_nonexistent():
        d = _fresh_dir("mgr_load_ne")
        mgr = CheckpointManager(d, phase=1, config={})
        try:
            mgr.load_checkpoint(d / "nonexistent.pt")
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass

    _run_test("manager_load_nonexistent", test_manager_load_nonexistent)

    # ===== 56. Phase boundary cross-phase load =====

    def test_cross_phase_boundary_load():
        d = _fresh_dir("cross_phase")
        model = _MockModel()

        # Phase 1 saves boundary.
        mgr1 = CheckpointManager(d, phase=1, config={"workspace_dim": 4096})
        mgr1.save_phase_boundary(model, total_steps=1000)

        # Phase 2 loads phase 1 boundary.
        mgr2 = CheckpointManager(d, phase=2, config={"workspace_dim": 4096})
        ckpt = mgr2.load_phase_boundary(phase=1)
        assert ckpt is not None
        assert ckpt["source_phase"] == 1
        assert ckpt["target_phase"] == 2

        # Can load model from boundary.
        model2 = _MockModel()
        model2.load_state_dict(ckpt["model_state_dict"])
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.equal(p1, p2)

    _run_test("cross_phase_boundary_load", test_cross_phase_boundary_load)

    # ===== 57. list_checkpoints sorting =====

    def test_manager_list_sorting():
        d = _fresh_dir("mgr_sort")
        model = _MockModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        mgr = CheckpointManager(d, phase=1, config={})

        # Save out of order to check sorting.
        mgr.save_periodic(model, opt, None, None, 300, 150, 3)
        mgr.save_periodic(model, opt, None, None, 100, 50, 1)
        mgr.save_periodic(model, opt, None, None, 200, 100, 2)

        entries = mgr.list_checkpoints()
        steps = [e["step"] for e in entries if e["type"] == "periodic"]
        assert steps == sorted(steps), f"Steps should be sorted: {steps}"

    _run_test("manager_list_sorting", test_manager_list_sorting)

    # ===== 58. save_final round-trip =====

    def test_manager_save_final():
        d = _fresh_dir("mgr_final_rt")
        model = _MockModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        mgr = CheckpointManager(d, phase=1, config={})

        path = mgr.save_final(model, opt, None, None, 999, 499, 10, {"val_loss": 0.1})
        assert path.name == "ckpt_final.pt"
        ckpt = mgr.load_checkpoint(path)
        assert ckpt["global_step"] == 999

    _run_test("manager_save_final", test_manager_save_final)

    # ===== 59. PhaseBoundaryBuilder deep-copies state dict =====

    def test_boundary_builder_deep_copy():
        model = _MockModel()
        original_weight = model.fc1.weight.clone()
        ckpt = PhaseBoundaryBuilder.build(model, {}, phase=1, run_id="x")

        # Mutate model after build.
        with torch.no_grad():
            model.fc1.weight.fill_(999.0)

        # Boundary state dict should still have original weights.
        assert torch.equal(ckpt["model_state_dict"]["fc1.weight"], original_weight)

    _run_test("boundary_builder_deep_copy", test_boundary_builder_deep_copy)

    # ===== 60. Metadata timestamp is present =====

    def test_boundary_metadata_timestamp():
        model = _MockModel()
        ckpt = PhaseBoundaryBuilder.build(model, {}, phase=1, run_id="t")
        md = ckpt["metadata"]
        assert "timestamp" in md
        assert "timestamp_iso" in md
        assert isinstance(md["timestamp"], float)
        assert "T" in md["timestamp_iso"]

    _run_test("boundary_metadata_timestamp", test_boundary_metadata_timestamp)

    # ===== 61. ResumeManager with scheduler =====

    def test_resume_with_scheduler():
        d = _fresh_dir("resume_sched")
        model = _MockModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)

        for _ in range(12):
            sched.step()
        lr_before = sched.get_last_lr()[0]

        mgr = CheckpointManager(d, phase=1, config={})
        mgr.save_periodic(model, opt, sched, None, 12, 12, 1)

        model2 = _MockModel()
        opt2 = torch.optim.SGD(model2.parameters(), lr=0.1)
        sched2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=5, gamma=0.5)

        info = ResumeManager.resume(model2, opt2, sched2, d, phase=1)
        assert info is not None
        lr_after = sched2.get_last_lr()[0]
        assert lr_before == lr_after, f"LR mismatch: {lr_before} vs {lr_after}"

    _run_test("resume_with_scheduler", test_resume_with_scheduler)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print()
    print(f"{'=' * 60}")
    print(f"  {_pass_count} passed, {_fail_count} failed out of {len(_test_names)} tests")
    print(f"{'=' * 60}")

    # Cleanup temp dir.
    shutil.rmtree(_tmpdir, ignore_errors=True)

    sys.exit(0 if _fail_count == 0 else 1)
