#!/usr/bin/env python3
"""
logging_utils_template.py
=========================
Comprehensive dual-backend training logging system for the brain-inspired AI
seven-phase training pipeline.

Backends: TensorBoard + Weights & Biases + append-only JSONL.
All backends degrade gracefully when their dependencies are absent.

Usage:
    from logging_utils_template import MetricLogger, LoggingConfig

    cfg = LoggingConfig(tensorboard_enabled=True, wandb_enabled=False)
    with MetricLogger(run_dir="runs/exp01", config=cfg, phase=1) as logger:
        for step in range(1000):
            logger.log_scalar("train/loss", loss_val, step)
            logger.log_gradient_norms(model.named_parameters(), step)
        logger.log_phase_transition(from_phase=1, to_phase=2, step=1000)

Self-contained: no brain_ai imports required.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Optional dependency probes
# ---------------------------------------------------------------------------
try:
    from torch.utils.tensorboard import SummaryWriter

    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False

try:
    import wandb  # type: ignore[import-untyped]

    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


# ===================================================================
# 5. MetricNamespace  (~40 lines)
# ===================================================================
class MetricNamespace:
    """Standard metric tag constants and constructors.

    All tags follow the convention ``group/subgroup/name`` with no spaces
    and no leading slash.
    """

    # -- Fixed tags --
    TRAIN_LOSS = "train/loss"
    TRAIN_ACC = "train/acc"
    VAL_LOSS = "val/loss"
    VAL_ACC = "val/acc"
    LEARNING_RATE = "lr/base"
    GLOBAL_GRAD_NORM = "grad_norm/global"

    _TAG_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_/.-]*$")

    @classmethod
    def validate_tag(cls, tag: str) -> str:
        """Validate *tag* format. Raises ``ValueError`` on bad input."""
        if not tag:
            raise ValueError("Metric tag must be a non-empty string.")
        if tag.startswith("/"):
            raise ValueError(f"Metric tag must not start with '/': {tag!r}")
        if " " in tag:
            raise ValueError(f"Metric tag must not contain spaces: {tag!r}")
        if not cls._TAG_RE.match(tag):
            raise ValueError(
                f"Metric tag contains invalid characters: {tag!r}"
            )
        return tag

    @staticmethod
    def phase_metric(phase: int, name: str) -> str:
        """Return ``phase<N>/<name>``."""
        return f"phase{phase}/{name}"

    @staticmethod
    def module_metric(phase: int, module: str, name: str) -> str:
        """Return ``phase<N>/<module>/<name>``."""
        return f"phase{phase}/{module}/{name}"

    @staticmethod
    def system_metric(name: str) -> str:
        """Return ``system/<name>``."""
        return f"system/{name}"

    @staticmethod
    def lr_metric(name: str = "base") -> str:
        """Return ``lr/<name>``."""
        return f"lr/{name}"

    @staticmethod
    def grad_norm_metric(module: str = "global") -> str:
        """Return ``grad_norm/<module>``."""
        return f"grad_norm/{module}"


# ===================================================================
# 7. LoggingConfig dataclass  (~40 lines)
# ===================================================================
@dataclass
class LoggingConfig:
    """Runtime configuration for the logging system."""

    tensorboard_enabled: bool = True
    wandb_enabled: bool = False
    wandb_project: str = "brain_ai"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_run_id: Optional[str] = None
    wandb_resume_mode: str = "never"  # "must" | "allow" | "never"
    jsonl_enabled: bool = True
    log_histograms_every: int = 500
    log_images_every: int = 1000
    log_system_every: int = 100
    flush_every: int = 100

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (JSON-safe)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LoggingConfig":
        """Deserialize from a plain dict, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


# ===================================================================
# 2. TensorBoardBackend  (~100 lines)
# ===================================================================
class TensorBoardBackend:
    """Thin wrapper around ``SummaryWriter`` with graceful no-op fallback."""

    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir
        self._writer: Optional[Any] = None
        self._active = False

        if not _TB_AVAILABLE:
            logger.warning(
                "TensorBoard not installed — TensorBoardBackend is no-op. "
                "Install with: pip install tensorboard"
            )
            return

        os.makedirs(log_dir, exist_ok=True)
        try:
            self._writer = SummaryWriter(log_dir=log_dir)
            self._active = True
        except Exception as exc:
            logger.warning("Failed to create SummaryWriter: %s", exc)

    # -- public API -------------------------------------------------------

    @property
    def active(self) -> bool:
        return self._active

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        if self._active:
            self._writer.add_scalar(tag, value, global_step=step)

    def add_scalars(
        self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int
    ) -> None:
        if self._active:
            self._writer.add_scalars(
                main_tag, tag_scalar_dict, global_step=step
            )

    def add_histogram(
        self, tag: str, values: torch.Tensor, step: int
    ) -> None:
        if self._active:
            self._writer.add_histogram(tag, values, global_step=step)

    def add_image(
        self, tag: str, img_tensor: torch.Tensor, step: int
    ) -> None:
        """Log an image tensor (C, H, W). Resized to max 256x256."""
        if not self._active:
            return
        if img_tensor.dim() == 2:
            img_tensor = img_tensor.unsqueeze(0)
        _, h, w = img_tensor.shape
        if max(h, w) > 256:
            scale = 256.0 / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img_tensor = torch.nn.functional.interpolate(
                img_tensor.unsqueeze(0),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        self._writer.add_image(tag, img_tensor, global_step=step)

    def add_text(self, tag: str, text: str, step: int) -> None:
        if self._active:
            self._writer.add_text(tag, text, global_step=step)

    def add_custom_scalars_layout(self, layout: Dict[str, Any]) -> None:
        """Configure a custom dashboard layout for phase-aware views.

        *layout* follows the TensorBoard custom-scalars protobuf schema:

            {"Category": {"Chart": ["Multiline", ["tag1", "tag2"]]}}
        """
        if self._active:
            try:
                self._writer.add_custom_scalars(layout)
            except Exception as exc:
                logger.warning("Failed to set custom scalars layout: %s", exc)

    def flush(self) -> None:
        if self._active:
            self._writer.flush()

    def close(self) -> None:
        if self._active:
            self._writer.flush()
            self._writer.close()
            self._active = False


# ===================================================================
# 3. WandBBackend  (~120 lines)
# ===================================================================
class WandBBackend:
    """Wrapper around the Weights & Biases Python SDK with graceful fallback."""

    def __init__(
        self,
        config: LoggingConfig,
        run_id: Optional[str] = None,
        resume_mode: str = "never",
    ) -> None:
        self._active = False
        self._run: Optional[Any] = None

        if not _WANDB_AVAILABLE:
            logger.warning(
                "wandb not installed — WandBBackend is no-op. "
                "Install with: pip install wandb"
            )
            return

        effective_id = run_id or config.wandb_run_id
        effective_resume = resume_mode if resume_mode != "never" else config.wandb_resume_mode

        init_kwargs: Dict[str, Any] = {
            "project": config.wandb_project,
            "config": config.to_dict(),
        }
        if config.wandb_entity:
            init_kwargs["entity"] = config.wandb_entity
        if config.wandb_run_name:
            init_kwargs["name"] = config.wandb_run_name
        if effective_id:
            init_kwargs["id"] = effective_id
        if effective_resume and effective_resume != "never":
            init_kwargs["resume"] = effective_resume

        try:
            self._run = wandb.init(**init_kwargs)
            self._active = True
        except Exception as exc:
            logger.warning("wandb.init failed: %s — backend is no-op", exc)

    # -- public API -------------------------------------------------------

    @property
    def active(self) -> bool:
        return self._active

    def log(self, metrics_dict: Dict[str, Any], step: int) -> None:
        if self._active:
            wandb.log(metrics_dict, step=step)

    def log_artifact(
        self, path: str, name: str, artifact_type: str = "model"
    ) -> None:
        """Log a file or directory as a W&B Artifact."""
        if not self._active:
            return
        try:
            art = wandb.Artifact(name=name, type=artifact_type)
            if os.path.isdir(path):
                art.add_dir(path)
            else:
                art.add_file(path)
            wandb.log_artifact(art)
        except Exception as exc:
            logger.warning("wandb log_artifact failed: %s", exc)

    def log_table(
        self, table_name: str, data: List[List[Any]], columns: Optional[List[str]] = None
    ) -> None:
        """Log tabular data as a ``wandb.Table``."""
        if not self._active:
            return
        try:
            tbl = wandb.Table(data=data, columns=columns or [])
            wandb.log({table_name: tbl})
        except Exception as exc:
            logger.warning("wandb log_table failed: %s", exc)

    def alert(
        self, title: str, text: str, level: str = "WARN"
    ) -> None:
        """Send a W&B alert (email/Slack notification)."""
        if not self._active:
            return
        level_map = {
            "INFO": wandb.AlertLevel.INFO,
            "WARN": wandb.AlertLevel.WARN,
            "ERROR": wandb.AlertLevel.ERROR,
        }
        try:
            wandb.alert(
                title=title,
                text=text,
                level=level_map.get(level.upper(), wandb.AlertLevel.WARN),
            )
        except Exception as exc:
            logger.warning("wandb alert failed: %s", exc)

    def get_run_id(self) -> Optional[str]:
        """Return the wandb run ID, or ``None`` if inactive."""
        if self._active and self._run is not None:
            return self._run.id
        return None

    def finish(self, exit_code: int = 0) -> None:
        if self._active:
            try:
                wandb.finish(exit_code=exit_code)
            except Exception:
                pass
            self._active = False


# ===================================================================
# 4. JSONLMetricWriter  (~60 lines)
# ===================================================================
class JSONLMetricWriter:
    """Append-only JSONL metric log at ``<run_dir>/metrics.jsonl``."""

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        self._fh = open(filepath, "a", encoding="utf-8")

    def write(self, entry: Dict[str, Any]) -> None:
        """Serialize *entry* to a single JSON line and flush."""
        line = json.dumps(entry, default=str, ensure_ascii=False)
        self._fh.write(line + "\n")
        self._fh.flush()

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        self._fh.flush()
        self._fh.close()

    # -- class / static helpers ------------------------------------------

    @classmethod
    def read_all(cls, filepath: str) -> List[Dict[str, Any]]:
        """Load every entry from *filepath* into a list of dicts."""
        entries: List[Dict[str, Any]] = []
        with open(filepath, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    @classmethod
    def summarize(cls, filepath: str) -> Dict[str, Dict[str, float]]:
        """Compute min / max / mean / final for every metric key in *filepath*.

        Returns ``{metric_key: {"min": ..., "max": ..., "mean": ..., "final": ...}}``.
        Only considers entries that have a ``"metrics"`` sub-dict.
        """
        entries = cls.read_all(filepath)
        accumulators: Dict[str, List[float]] = {}
        for entry in entries:
            metrics = entry.get("metrics")
            if not isinstance(metrics, dict):
                continue
            for key, val in metrics.items():
                if isinstance(val, (int, float)) and math.isfinite(val):
                    accumulators.setdefault(key, []).append(float(val))

        summary: Dict[str, Dict[str, float]] = {}
        for key, vals in accumulators.items():
            summary[key] = {
                "min": min(vals),
                "max": max(vals),
                "mean": sum(vals) / len(vals),
                "final": vals[-1],
            }
        return summary


# ===================================================================
# 6. GradientMonitor  (~60 lines)
# ===================================================================
class GradientMonitor:
    """Utility for computing, logging, and auditing gradient norms."""

    @staticmethod
    def compute_grad_norms(
        named_parameters: Iterator[Tuple[str, nn.Parameter]],
    ) -> Dict[str, float]:
        """Return ``{module_prefix: l2_grad_norm}`` for all parameters with gradients.

        Module prefix is derived by stripping the final ``.weight`` / ``.bias``
        suffix from the parameter name.
        """
        norms: Dict[str, float] = {}
        for name, param in named_parameters:
            if param.grad is not None:
                norm_val = param.grad.data.norm(2).item()
                # Strip trailing .weight / .bias for module-level grouping
                module_key = re.sub(r"\.(weight|bias)$", "", name)
                # If multiple params map to same module, take the max
                if module_key in norms:
                    norms[module_key] = max(norms[module_key], norm_val)
                else:
                    norms[module_key] = norm_val
        return norms

    @staticmethod
    def compute_global_grad_norm(
        named_parameters: Iterator[Tuple[str, nn.Parameter]],
    ) -> float:
        """Compute the combined L2 norm across all parameter gradients."""
        total_norm_sq = 0.0
        for _name, param in named_parameters:
            if param.grad is not None:
                total_norm_sq += param.grad.data.norm(2).item() ** 2
        return math.sqrt(total_norm_sq)

    @staticmethod
    def detect_anomalies(
        grad_norms: Dict[str, float],
        max_norm: float = 100.0,
        min_norm: float = 1e-7,
    ) -> List[str]:
        """Return a list of human-readable warnings for exploding / vanishing gradients."""
        warnings_list: List[str] = []
        for module, norm in grad_norms.items():
            if norm > max_norm:
                warnings_list.append(
                    f"EXPLODING gradient in '{module}': norm={norm:.4e} > max_norm={max_norm}"
                )
            elif norm < min_norm:
                warnings_list.append(
                    f"VANISHING gradient in '{module}': norm={norm:.4e} < min_norm={min_norm}"
                )
        return warnings_list

    @classmethod
    def log_all(
        cls,
        metric_logger: "MetricLogger",
        named_parameters: Iterator[Tuple[str, nn.Parameter]],
        step: int,
    ) -> None:
        """Compute norms, log them, and emit warnings for anomalies."""
        # We need to iterate named_parameters twice, so materialise the list
        params_list = list(named_parameters)

        grad_norms = cls.compute_grad_norms(iter(params_list))
        global_norm = cls.compute_global_grad_norm(iter(params_list))

        # Log per-module norms
        for module, norm in grad_norms.items():
            tag = MetricNamespace.grad_norm_metric(module)
            metric_logger.log_scalar(tag, norm, step)

        # Log global norm
        metric_logger.log_scalar(MetricNamespace.GLOBAL_GRAD_NORM, global_norm, step)

        # Detect and warn about anomalies
        anomalies = cls.detect_anomalies(grad_norms)
        for warning_msg in anomalies:
            logger.warning("[step %d] %s", step, warning_msg)
            # Also log a text note in TB
            metric_logger.log_text("grad_anomalies", warning_msg, step)


# ===================================================================
# 1. MetricLogger  (~250 lines)
# ===================================================================
class MetricLogger:
    """Unified metric logger dispatching to TensorBoard, W&B, and JSONL.

    Typical usage::

        cfg = LoggingConfig(tensorboard_enabled=True)
        with MetricLogger("runs/exp01", cfg, phase=1) as ml:
            ml.log_scalar("train/loss", 0.5, step=10)
    """

    def __init__(
        self,
        run_dir: str,
        config: LoggingConfig,
        phase: Optional[int] = None,
    ) -> None:
        self.run_dir = run_dir
        self.config = config
        self.phase = phase
        self._step_count = 0
        self._start_time = time.monotonic()
        self._last_log_time = self._start_time

        os.makedirs(run_dir, exist_ok=True)

        # -- TensorBoard ---------------------------------------------------
        self._tb: Optional[TensorBoardBackend] = None
        if config.tensorboard_enabled:
            tb_dir = os.path.join(run_dir, "tb")
            self._tb = TensorBoardBackend(tb_dir)
            if self._tb.active:
                self._setup_tb_layout()

        # -- W&B -----------------------------------------------------------
        self._wb: Optional[WandBBackend] = None
        if config.wandb_enabled:
            self._wb = WandBBackend(
                config,
                run_id=config.wandb_run_id,
                resume_mode=config.wandb_resume_mode,
            )

        # -- JSONL ----------------------------------------------------------
        self._jsonl: Optional[JSONLMetricWriter] = None
        if config.jsonl_enabled:
            jsonl_path = os.path.join(run_dir, "metrics.jsonl")
            self._jsonl = JSONLMetricWriter(jsonl_path)

    # -- Context manager ---------------------------------------------------

    def __enter__(self) -> "MetricLogger":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    # -- Scalar logging ----------------------------------------------------

    def log_scalar(
        self,
        tag: str,
        value: float,
        step: int,
        phase_step: Optional[int] = None,
    ) -> None:
        """Log a single scalar to all active backends."""
        MetricNamespace.validate_tag(tag)
        value = float(value)

        if self._tb is not None:
            self._tb.add_scalar(tag, value, step)

        if self._wb is not None and self._wb.active:
            self._wb.log({tag: value}, step)

        if self._jsonl is not None:
            self._jsonl.write(
                {
                    "step": step,
                    "phase": self.phase,
                    "phase_step": phase_step,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "metrics": {tag: value},
                }
            )

        self._step_count += 1
        self._maybe_flush(step)

    def log_scalars(
        self,
        tag_value_dict: Dict[str, float],
        step: int,
        phase_step: Optional[int] = None,
    ) -> None:
        """Log multiple scalars in a single call."""
        coerced: Dict[str, float] = {}
        for tag, value in tag_value_dict.items():
            MetricNamespace.validate_tag(tag)
            coerced[tag] = float(value)

        # TensorBoard — individual scalars
        if self._tb is not None:
            for tag, value in coerced.items():
                self._tb.add_scalar(tag, value, step)

        # W&B — single log call
        if self._wb is not None and self._wb.active:
            self._wb.log(coerced, step)

        # JSONL — single entry
        if self._jsonl is not None:
            self._jsonl.write(
                {
                    "step": step,
                    "phase": self.phase,
                    "phase_step": phase_step,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "metrics": coerced,
                }
            )

        self._step_count += 1
        self._maybe_flush(step)

    # -- Histogram ---------------------------------------------------------

    def log_histogram(
        self, tag: str, values: torch.Tensor, step: int
    ) -> None:
        """Log a histogram to TB and W&B."""
        MetricNamespace.validate_tag(tag)
        if self._tb is not None:
            self._tb.add_histogram(tag, values, step)
        if self._wb is not None and self._wb.active:
            try:
                self._wb.log({tag: wandb.Histogram(values.cpu().numpy())}, step)
            except Exception:
                pass  # numpy conversion may fail for some dtypes

    # -- Image -------------------------------------------------------------

    def log_image(
        self, tag: str, image_tensor: torch.Tensor, step: int
    ) -> None:
        """Log an image tensor (C, H, W) to TB. Resized to max 256x256."""
        MetricNamespace.validate_tag(tag)
        if self._tb is not None:
            self._tb.add_image(tag, image_tensor, step)
        if self._wb is not None and self._wb.active:
            try:
                self._wb.log(
                    {tag: wandb.Image(image_tensor.permute(1, 2, 0).cpu().numpy())},
                    step,
                )
            except Exception:
                pass

    # -- Text --------------------------------------------------------------

    def log_text(self, tag: str, text: str, step: int) -> None:
        """Log text to TB and W&B."""
        MetricNamespace.validate_tag(tag)
        if self._tb is not None:
            self._tb.add_text(tag, text, step)
        if self._wb is not None and self._wb.active:
            self._wb.log({tag: text}, step)

    # -- Phase transition --------------------------------------------------

    def log_phase_transition(
        self,
        from_phase: int,
        to_phase: int,
        step: int,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Record a phase boundary as a special event in all backends."""
        marker_tag = f"phase_transition/{from_phase}_to_{to_phase}"
        transition_text = (
            f"Phase transition: {from_phase} -> {to_phase} at step {step}"
        )

        # TB
        if self._tb is not None:
            self._tb.add_scalar(marker_tag, 1.0, step)
            self._tb.add_text("phase_transitions", transition_text, step)

        # W&B
        wb_payload: Dict[str, Any] = {
            marker_tag: 1.0,
            "phase": to_phase,
        }
        if metrics:
            wb_payload.update(metrics)
        if self._wb is not None and self._wb.active:
            self._wb.log(wb_payload, step)

        # JSONL
        if self._jsonl is not None:
            entry: Dict[str, Any] = {
                "event": "phase_transition",
                "from_phase": from_phase,
                "to_phase": to_phase,
                "step": step,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if metrics:
                entry["metrics"] = metrics
            self._jsonl.write(entry)

        self.phase = to_phase
        logger.info(transition_text)

    # -- Gradient norms ----------------------------------------------------

    def log_gradient_norms(
        self,
        named_parameters: Iterator[Tuple[str, nn.Parameter]],
        step: int,
    ) -> None:
        """Compute and log per-module + global gradient norms."""
        GradientMonitor.log_all(self, named_parameters, step)

    # -- Weight norms ------------------------------------------------------

    def log_weight_norms(
        self,
        named_parameters: Iterator[Tuple[str, nn.Parameter]],
        step: int,
    ) -> None:
        """Compute and log per-module weight norms."""
        for name, param in named_parameters:
            module_key = re.sub(r"\.(weight|bias)$", "", name)
            tag = f"weight_norm/{module_key}"
            norm_val = param.data.norm(2).item()
            self.log_scalar(tag, norm_val, step)

    # -- System metrics ----------------------------------------------------

    def log_system_metrics(self, step: int) -> None:
        """Log GPU utilisation, memory, and throughput when available."""
        metrics: Dict[str, float] = {}

        # GPU metrics via torch.cuda
        if torch.cuda.is_available():
            try:
                dev = torch.cuda.current_device()
                mem_alloc = torch.cuda.memory_allocated(dev) / (1024 ** 3)
                mem_reserved = torch.cuda.memory_reserved(dev) / (1024 ** 3)
                mem_max = torch.cuda.max_memory_allocated(dev) / (1024 ** 3)
                metrics[MetricNamespace.system_metric("gpu_mem_alloc_gb")] = mem_alloc
                metrics[MetricNamespace.system_metric("gpu_mem_reserved_gb")] = mem_reserved
                metrics[MetricNamespace.system_metric("gpu_mem_peak_gb")] = mem_max

                # Utilisation (requires pynvml behind the scenes — best effort)
                try:
                    utilisation = torch.cuda.utilization(dev)
                    metrics[MetricNamespace.system_metric("gpu_util_pct")] = float(utilisation)
                except Exception:
                    pass
            except Exception:
                pass

        # Throughput: steps per second since last system-metrics log
        now = time.monotonic()
        elapsed = now - self._last_log_time
        if elapsed > 0:
            metrics[MetricNamespace.system_metric("steps_per_sec")] = (
                self._step_count / elapsed
            )
        self._last_log_time = now
        self._step_count = 0

        if metrics:
            self.log_scalars(metrics, step)

    # -- Flush / close -----------------------------------------------------

    def flush(self) -> None:
        """Flush all active backends."""
        if self._tb is not None:
            self._tb.flush()
        if self._jsonl is not None:
            self._jsonl.flush()
        # W&B auto-flushes on log()

    def close(self) -> None:
        """Flush and close all backends."""
        if self._tb is not None:
            self._tb.close()
            self._tb = None
        if self._wb is not None and self._wb.active:
            self._wb.finish()
            self._wb = None
        if self._jsonl is not None:
            self._jsonl.close()
            self._jsonl = None

    # -- Internals ---------------------------------------------------------

    def _maybe_flush(self, step: int) -> None:
        """Auto-flush every ``config.flush_every`` steps."""
        if step > 0 and step % self.config.flush_every == 0:
            self.flush()

    def _setup_tb_layout(self) -> None:
        """Pre-configure TensorBoard custom scalars dashboard layout."""
        if self._tb is None:
            return
        layout = {
            "Training": {
                "Loss": [
                    "Multiline",
                    [r"train/loss", r"val/loss"],
                ],
                "Accuracy": [
                    "Multiline",
                    [r"train/acc", r"val/acc"],
                ],
            },
            "Phases": {
                "Phase Loss": [
                    "Multiline",
                    [r"phase\d+/loss"],
                ],
                "Phase Accuracy": [
                    "Multiline",
                    [r"phase\d+/acc"],
                ],
            },
            "System": {
                "GPU Memory": [
                    "Multiline",
                    [r"system/gpu_mem_alloc_gb", r"system/gpu_mem_peak_gb"],
                ],
                "Throughput": [
                    "Multiline",
                    [r"system/steps_per_sec"],
                ],
            },
            "Gradients": {
                "Global Norm": [
                    "Multiline",
                    [r"grad_norm/global"],
                ],
            },
        }
        self._tb.add_custom_scalars_layout(layout)


# ===================================================================
# 8. Self-test block  (~200 lines, 50+ tests)
# ===================================================================
if __name__ == "__main__":
    import shutil
    import sys
    import tempfile
    import traceback
    from unittest.mock import MagicMock, patch

    # -- helpers -----------------------------------------------------------
    _passed = 0
    _failed = 0
    _errors: List[str] = []

    def _run_test(name: str, fn):
        global _passed, _failed
        try:
            fn()
            _passed += 1
            print(f"  PASS  {name}")
        except Exception as exc:
            _failed += 1
            _errors.append(f"{name}: {exc}")
            print(f"  FAIL  {name}: {exc}")
            traceback.print_exc()

    def _tmpdir() -> str:
        return tempfile.mkdtemp(prefix="logtest_")

    # =====================================================================
    # Tests
    # =====================================================================

    # -- MetricNamespace ---------------------------------------------------

    def test_ns_fixed_tags():
        assert MetricNamespace.TRAIN_LOSS == "train/loss"
        assert MetricNamespace.TRAIN_ACC == "train/acc"
        assert MetricNamespace.VAL_LOSS == "val/loss"
        assert MetricNamespace.VAL_ACC == "val/acc"

    def test_ns_phase_metric():
        assert MetricNamespace.phase_metric(3, "loss") == "phase3/loss"

    def test_ns_module_metric():
        assert MetricNamespace.module_metric(2, "snn", "fire_rate") == "phase2/snn/fire_rate"

    def test_ns_system_metric():
        assert MetricNamespace.system_metric("gpu_util") == "system/gpu_util"

    def test_ns_lr_metric():
        assert MetricNamespace.lr_metric() == "lr/base"
        assert MetricNamespace.lr_metric("encoder") == "lr/encoder"

    def test_ns_grad_norm_metric():
        assert MetricNamespace.grad_norm_metric() == "grad_norm/global"
        assert MetricNamespace.grad_norm_metric("fc1") == "grad_norm/fc1"

    def test_ns_validate_ok():
        MetricNamespace.validate_tag("train/loss")
        MetricNamespace.validate_tag("phase1/snn/fire_rate")
        MetricNamespace.validate_tag("a.b-c_d")

    def test_ns_validate_empty():
        try:
            MetricNamespace.validate_tag("")
            assert False, "Should have raised"
        except ValueError:
            pass

    def test_ns_validate_leading_slash():
        try:
            MetricNamespace.validate_tag("/bad")
            assert False, "Should have raised"
        except ValueError:
            pass

    def test_ns_validate_spaces():
        try:
            MetricNamespace.validate_tag("bad tag")
            assert False, "Should have raised"
        except ValueError:
            pass

    def test_ns_validate_invalid_chars():
        try:
            MetricNamespace.validate_tag("bad@tag")
            assert False, "Should have raised"
        except ValueError:
            pass

    # -- LoggingConfig -----------------------------------------------------

    def test_config_defaults():
        cfg = LoggingConfig()
        assert cfg.tensorboard_enabled is True
        assert cfg.wandb_enabled is False
        assert cfg.wandb_project == "brain_ai"
        assert cfg.jsonl_enabled is True
        assert cfg.log_histograms_every == 500
        assert cfg.log_images_every == 1000
        assert cfg.log_system_every == 100
        assert cfg.flush_every == 100

    def test_config_to_dict():
        cfg = LoggingConfig(wandb_project="test_proj")
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert d["wandb_project"] == "test_proj"
        assert d["tensorboard_enabled"] is True

    def test_config_from_dict():
        d = {"wandb_project": "abc", "wandb_enabled": True, "unknown_key": 42}
        cfg = LoggingConfig.from_dict(d)
        assert cfg.wandb_project == "abc"
        assert cfg.wandb_enabled is True

    def test_config_roundtrip():
        cfg = LoggingConfig(
            tensorboard_enabled=False,
            wandb_enabled=True,
            wandb_project="rt",
            flush_every=50,
        )
        cfg2 = LoggingConfig.from_dict(cfg.to_dict())
        assert cfg == cfg2

    # -- JSONLMetricWriter -------------------------------------------------

    def test_jsonl_write_read():
        d = _tmpdir()
        path = os.path.join(d, "metrics.jsonl")
        w = JSONLMetricWriter(path)
        w.write({"step": 1, "metrics": {"loss": 0.5}})
        w.write({"step": 2, "metrics": {"loss": 0.3}})
        w.close()
        entries = JSONLMetricWriter.read_all(path)
        assert len(entries) == 2
        assert entries[0]["step"] == 1
        assert entries[1]["metrics"]["loss"] == 0.3
        shutil.rmtree(d)

    def test_jsonl_append_mode():
        d = _tmpdir()
        path = os.path.join(d, "metrics.jsonl")
        w1 = JSONLMetricWriter(path)
        w1.write({"step": 1, "metrics": {"a": 1}})
        w1.close()
        w2 = JSONLMetricWriter(path)
        w2.write({"step": 2, "metrics": {"a": 2}})
        w2.close()
        entries = JSONLMetricWriter.read_all(path)
        assert len(entries) == 2
        shutil.rmtree(d)

    def test_jsonl_summarize():
        d = _tmpdir()
        path = os.path.join(d, "metrics.jsonl")
        w = JSONLMetricWriter(path)
        for i in range(5):
            w.write({"step": i, "metrics": {"loss": float(4 - i), "acc": float(i) / 4}})
        w.close()
        summary = JSONLMetricWriter.summarize(path)
        assert "loss" in summary
        assert summary["loss"]["min"] == 0.0
        assert summary["loss"]["max"] == 4.0
        assert abs(summary["loss"]["mean"] - 2.0) < 1e-6
        assert summary["loss"]["final"] == 0.0
        assert "acc" in summary
        assert summary["acc"]["final"] == 1.0
        shutil.rmtree(d)

    def test_jsonl_summarize_skips_events():
        d = _tmpdir()
        path = os.path.join(d, "metrics.jsonl")
        w = JSONLMetricWriter(path)
        w.write({"event": "phase_transition", "from_phase": 1, "to_phase": 2})
        w.write({"step": 1, "metrics": {"x": 5.0}})
        w.close()
        summary = JSONLMetricWriter.summarize(path)
        assert "x" in summary
        assert summary["x"]["final"] == 5.0
        shutil.rmtree(d)

    def test_jsonl_special_event_entry():
        d = _tmpdir()
        path = os.path.join(d, "metrics.jsonl")
        w = JSONLMetricWriter(path)
        w.write({"event": "phase_transition", "from_phase": 1, "to_phase": 2, "step": 100})
        w.close()
        entries = JSONLMetricWriter.read_all(path)
        assert len(entries) == 1
        assert entries[0]["event"] == "phase_transition"
        assert entries[0]["from_phase"] == 1
        shutil.rmtree(d)

    # -- TensorBoardBackend ------------------------------------------------

    def test_tb_backend_creation():
        d = _tmpdir()
        tb = TensorBoardBackend(os.path.join(d, "tb"))
        if _TB_AVAILABLE:
            assert tb.active
        tb.close()
        shutil.rmtree(d)

    def test_tb_backend_scalar():
        d = _tmpdir()
        tb = TensorBoardBackend(os.path.join(d, "tb"))
        tb.add_scalar("test/val", 1.23, 0)
        tb.flush()
        tb.close()
        # Check event file exists (if TB was available)
        if _TB_AVAILABLE:
            tb_dir = os.path.join(d, "tb")
            files = os.listdir(tb_dir)
            assert any("events" in f for f in files), f"No event file found in {files}"
        shutil.rmtree(d)

    def test_tb_backend_noop_when_unavailable():
        """Simulate missing tensorboard by forcing _active = False."""
        d = _tmpdir()
        tb = TensorBoardBackend(os.path.join(d, "tb"))
        tb._active = False  # force no-op
        tb.add_scalar("x", 1, 0)  # should not raise
        tb.add_histogram("x", torch.randn(100), 0)
        tb.add_text("x", "hello", 0)
        tb.flush()
        tb.close()
        shutil.rmtree(d)

    def test_tb_backend_image():
        d = _tmpdir()
        tb = TensorBoardBackend(os.path.join(d, "tb"))
        img = torch.randn(3, 64, 64)
        tb.add_image("test/img", img, 0)
        tb.flush()
        tb.close()
        shutil.rmtree(d)

    def test_tb_backend_image_resize():
        d = _tmpdir()
        tb = TensorBoardBackend(os.path.join(d, "tb"))
        img = torch.randn(3, 512, 512)
        tb.add_image("test/big_img", img, 0)  # should resize to 256x256
        tb.flush()
        tb.close()
        shutil.rmtree(d)

    def test_tb_backend_image_2d():
        d = _tmpdir()
        tb = TensorBoardBackend(os.path.join(d, "tb"))
        img = torch.randn(28, 28)  # 2D grayscale
        tb.add_image("test/gray", img, 0)
        tb.flush()
        tb.close()
        shutil.rmtree(d)

    def test_tb_custom_layout():
        d = _tmpdir()
        tb = TensorBoardBackend(os.path.join(d, "tb"))
        layout = {"Cat": {"Chart": ["Multiline", ["t1", "t2"]]}}
        tb.add_custom_scalars_layout(layout)
        tb.close()
        shutil.rmtree(d)

    # -- WandBBackend ------------------------------------------------------

    def test_wandb_noop_fallback():
        """WandBBackend should be no-op when wandb is not installed or init fails."""
        cfg = LoggingConfig(wandb_enabled=True, wandb_project="test")
        # Force unavailable path
        with patch.dict("sys.modules", {"wandb": None}):
            wb = WandBBackend.__new__(WandBBackend)
            wb._active = False
            wb._run = None
        wb.log({"x": 1}, 0)
        wb.log_artifact("/tmp/fake", "art", "model")
        wb.log_table("t", [[1, 2]], ["a", "b"])
        wb.alert("title", "text")
        assert wb.get_run_id() is None
        wb.finish()

    def test_wandb_backend_properties():
        wb = WandBBackend.__new__(WandBBackend)
        wb._active = False
        wb._run = None
        assert wb.active is False
        assert wb.get_run_id() is None

    # -- MetricLogger ------------------------------------------------------

    def test_logger_creation_tb_only():
        d = _tmpdir()
        cfg = LoggingConfig(tensorboard_enabled=True, wandb_enabled=False, jsonl_enabled=True)
        ml = MetricLogger(d, cfg, phase=1)
        assert ml.phase == 1
        assert ml.run_dir == d
        ml.close()
        shutil.rmtree(d)

    def test_logger_scalar_logging():
        d = _tmpdir()
        cfg = LoggingConfig(tensorboard_enabled=True, wandb_enabled=False, jsonl_enabled=True)
        ml = MetricLogger(d, cfg, phase=1)
        ml.log_scalar("train/loss", 0.5, step=10)
        ml.log_scalar("train/loss", 0.3, step=20)
        ml.flush()
        ml.close()
        # Check JSONL
        jsonl_path = os.path.join(d, "metrics.jsonl")
        entries = JSONLMetricWriter.read_all(jsonl_path)
        assert len(entries) == 2
        assert entries[0]["metrics"]["train/loss"] == 0.5
        assert entries[1]["metrics"]["train/loss"] == 0.3
        # Check TB event file
        if _TB_AVAILABLE:
            tb_dir = os.path.join(d, "tb")
            assert any("events" in f for f in os.listdir(tb_dir))
        shutil.rmtree(d)

    def test_logger_batch_scalar_logging():
        d = _tmpdir()
        cfg = LoggingConfig(tensorboard_enabled=True, wandb_enabled=False, jsonl_enabled=True)
        ml = MetricLogger(d, cfg, phase=1)
        ml.log_scalars({"train/loss": 0.5, "train/acc": 0.8}, step=10)
        ml.close()
        jsonl_path = os.path.join(d, "metrics.jsonl")
        entries = JSONLMetricWriter.read_all(jsonl_path)
        assert len(entries) == 1
        assert entries[0]["metrics"]["train/loss"] == 0.5
        assert entries[0]["metrics"]["train/acc"] == 0.8
        shutil.rmtree(d)

    def test_logger_phase_transition():
        d = _tmpdir()
        cfg = LoggingConfig(tensorboard_enabled=True, wandb_enabled=False, jsonl_enabled=True)
        ml = MetricLogger(d, cfg, phase=1)
        ml.log_phase_transition(
            from_phase=1, to_phase=2, step=1000,
            metrics={"val/loss": 0.1, "val/acc": 0.95},
        )
        assert ml.phase == 2
        ml.close()
        jsonl_path = os.path.join(d, "metrics.jsonl")
        entries = JSONLMetricWriter.read_all(jsonl_path)
        assert len(entries) == 1
        assert entries[0]["event"] == "phase_transition"
        assert entries[0]["from_phase"] == 1
        assert entries[0]["to_phase"] == 2
        assert entries[0]["metrics"]["val/loss"] == 0.1
        shutil.rmtree(d)

    def test_logger_phase_transition_no_metrics():
        d = _tmpdir()
        cfg = LoggingConfig(tensorboard_enabled=False, wandb_enabled=False, jsonl_enabled=True)
        ml = MetricLogger(d, cfg, phase=3)
        ml.log_phase_transition(from_phase=3, to_phase=4, step=5000)
        ml.close()
        entries = JSONLMetricWriter.read_all(os.path.join(d, "metrics.jsonl"))
        assert entries[0]["event"] == "phase_transition"
        assert "metrics" not in entries[0]
        shutil.rmtree(d)

    def test_logger_histogram():
        d = _tmpdir()
        cfg = LoggingConfig(tensorboard_enabled=True, wandb_enabled=False, jsonl_enabled=False)
        ml = MetricLogger(d, cfg)
        ml.log_histogram("weights/fc1", torch.randn(100), step=5)
        ml.close()
        shutil.rmtree(d)

    def test_logger_image():
        d = _tmpdir()
        cfg = LoggingConfig(tensorboard_enabled=True, wandb_enabled=False, jsonl_enabled=False)
        ml = MetricLogger(d, cfg)
        ml.log_image("samples/gen", torch.randn(3, 64, 64), step=5)
        ml.close()
        shutil.rmtree(d)

    def test_logger_text():
        d = _tmpdir()
        cfg = LoggingConfig(tensorboard_enabled=True, wandb_enabled=False, jsonl_enabled=False)
        ml = MetricLogger(d, cfg)
        ml.log_text("debug/msg", "Hello from phase 1", step=5)
        ml.close()
        shutil.rmtree(d)

    def test_logger_context_manager():
        d = _tmpdir()
        cfg = LoggingConfig(tensorboard_enabled=True, wandb_enabled=False, jsonl_enabled=True)
        with MetricLogger(d, cfg, phase=1) as ml:
            ml.log_scalar("train/loss", 0.5, step=0)
        # After __exit__, backends should be closed
        assert ml._tb is None
        assert ml._jsonl is None
        shutil.rmtree(d)

    def test_logger_no_backends():
        d = _tmpdir()
        cfg = LoggingConfig(
            tensorboard_enabled=False, wandb_enabled=False, jsonl_enabled=False
        )
        ml = MetricLogger(d, cfg, phase=1)
        ml.log_scalar("train/loss", 0.5, step=0)
        ml.log_scalars({"a": 1}, step=1)
        ml.log_histogram("h", torch.randn(10), step=2)
        ml.log_text("t", "text", step=3)
        ml.log_phase_transition(1, 2, 100)
        ml.flush()
        ml.close()
        shutil.rmtree(d)

    def test_logger_auto_flush():
        d = _tmpdir()
        cfg = LoggingConfig(
            tensorboard_enabled=False, wandb_enabled=False,
            jsonl_enabled=True, flush_every=5,
        )
        ml = MetricLogger(d, cfg, phase=1)
        for i in range(10):
            ml.log_scalar("x", float(i), step=i)
        ml.close()
        entries = JSONLMetricWriter.read_all(os.path.join(d, "metrics.jsonl"))
        assert len(entries) == 10
        shutil.rmtree(d)

    def test_logger_phase_step():
        d = _tmpdir()
        cfg = LoggingConfig(
            tensorboard_enabled=False, wandb_enabled=False, jsonl_enabled=True
        )
        ml = MetricLogger(d, cfg, phase=2)
        ml.log_scalar("train/loss", 0.3, step=1000, phase_step=50)
        ml.close()
        entries = JSONLMetricWriter.read_all(os.path.join(d, "metrics.jsonl"))
        assert entries[0]["phase_step"] == 50
        assert entries[0]["phase"] == 2
        shutil.rmtree(d)

    # -- GradientMonitor ---------------------------------------------------

    def _make_model_with_grads():
        """Create a small model, run a forward/backward pass, return model."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )
        x = torch.randn(4, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        return model

    def test_grad_monitor_norms():
        model = _make_model_with_grads()
        norms = GradientMonitor.compute_grad_norms(model.named_parameters())
        assert len(norms) > 0
        for v in norms.values():
            assert isinstance(v, float)
            assert v >= 0

    def test_grad_monitor_global_norm():
        model = _make_model_with_grads()
        g = GradientMonitor.compute_global_grad_norm(model.named_parameters())
        assert isinstance(g, float)
        assert g > 0

    def test_grad_monitor_no_grads():
        model = nn.Linear(5, 3)
        norms = GradientMonitor.compute_grad_norms(model.named_parameters())
        assert len(norms) == 0
        g = GradientMonitor.compute_global_grad_norm(model.named_parameters())
        assert g == 0.0

    def test_grad_anomaly_exploding():
        norms = {"fc1": 200.0, "fc2": 0.5}
        warnings_list = GradientMonitor.detect_anomalies(norms, max_norm=100.0)
        assert len(warnings_list) == 1
        assert "EXPLODING" in warnings_list[0]
        assert "fc1" in warnings_list[0]

    def test_grad_anomaly_vanishing():
        norms = {"fc1": 1e-10, "fc2": 0.5}
        warnings_list = GradientMonitor.detect_anomalies(norms, min_norm=1e-7)
        assert len(warnings_list) == 1
        assert "VANISHING" in warnings_list[0]
        assert "fc1" in warnings_list[0]

    def test_grad_anomaly_both():
        norms = {"fc1": 500.0, "fc2": 1e-12}
        warnings_list = GradientMonitor.detect_anomalies(norms, max_norm=100.0, min_norm=1e-7)
        assert len(warnings_list) == 2

    def test_grad_anomaly_none():
        norms = {"fc1": 1.0, "fc2": 0.5}
        warnings_list = GradientMonitor.detect_anomalies(norms)
        assert len(warnings_list) == 0

    def test_grad_monitor_log_all():
        d = _tmpdir()
        cfg = LoggingConfig(
            tensorboard_enabled=False, wandb_enabled=False, jsonl_enabled=True
        )
        ml = MetricLogger(d, cfg, phase=1)
        model = _make_model_with_grads()
        GradientMonitor.log_all(ml, model.named_parameters(), step=5)
        ml.close()
        entries = JSONLMetricWriter.read_all(os.path.join(d, "metrics.jsonl"))
        tags = set()
        for e in entries:
            tags.update(e.get("metrics", {}).keys())
        assert MetricNamespace.GLOBAL_GRAD_NORM in tags
        shutil.rmtree(d)

    # -- Weight norms via MetricLogger -------------------------------------

    def test_logger_weight_norms():
        d = _tmpdir()
        cfg = LoggingConfig(
            tensorboard_enabled=False, wandb_enabled=False, jsonl_enabled=True
        )
        ml = MetricLogger(d, cfg, phase=1)
        model = nn.Linear(10, 5)
        ml.log_weight_norms(model.named_parameters(), step=0)
        ml.close()
        entries = JSONLMetricWriter.read_all(os.path.join(d, "metrics.jsonl"))
        assert len(entries) > 0
        has_weight_norm = any(
            any(k.startswith("weight_norm/") for k in e.get("metrics", {}))
            for e in entries
        )
        assert has_weight_norm
        shutil.rmtree(d)

    # -- System metrics (best effort) --------------------------------------

    def test_logger_system_metrics():
        d = _tmpdir()
        cfg = LoggingConfig(
            tensorboard_enabled=False, wandb_enabled=False, jsonl_enabled=True
        )
        ml = MetricLogger(d, cfg, phase=1)
        ml.log_system_metrics(step=0)
        ml.close()
        entries = JSONLMetricWriter.read_all(os.path.join(d, "metrics.jsonl"))
        # At minimum we should have steps_per_sec
        tags = set()
        for e in entries:
            tags.update(e.get("metrics", {}).keys())
        assert MetricNamespace.system_metric("steps_per_sec") in tags
        shutil.rmtree(d)

    # -- MetricLogger gradient norms shortcut ------------------------------

    def test_logger_log_gradient_norms():
        d = _tmpdir()
        cfg = LoggingConfig(
            tensorboard_enabled=False, wandb_enabled=False, jsonl_enabled=True
        )
        ml = MetricLogger(d, cfg, phase=1)
        model = _make_model_with_grads()
        ml.log_gradient_norms(model.named_parameters(), step=10)
        ml.close()
        entries = JSONLMetricWriter.read_all(os.path.join(d, "metrics.jsonl"))
        tags = set()
        for e in entries:
            tags.update(e.get("metrics", {}).keys())
        assert MetricNamespace.GLOBAL_GRAD_NORM in tags
        shutil.rmtree(d)

    # -- Validation edge cases ---------------------------------------------

    def test_log_scalar_validates_tag():
        d = _tmpdir()
        cfg = LoggingConfig(
            tensorboard_enabled=False, wandb_enabled=False, jsonl_enabled=False
        )
        ml = MetricLogger(d, cfg)
        try:
            ml.log_scalar("bad tag", 1.0, 0)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
        ml.close()
        shutil.rmtree(d)

    def test_log_scalars_validates_tags():
        d = _tmpdir()
        cfg = LoggingConfig(
            tensorboard_enabled=False, wandb_enabled=False, jsonl_enabled=False
        )
        ml = MetricLogger(d, cfg)
        try:
            ml.log_scalars({"/bad": 1.0}, 0)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
        ml.close()
        shutil.rmtree(d)

    def test_log_scalar_coerces_int():
        d = _tmpdir()
        cfg = LoggingConfig(
            tensorboard_enabled=False, wandb_enabled=False, jsonl_enabled=True
        )
        ml = MetricLogger(d, cfg)
        ml.log_scalar("train/loss", 1, step=0)  # int, not float
        ml.close()
        entries = JSONLMetricWriter.read_all(os.path.join(d, "metrics.jsonl"))
        assert isinstance(entries[0]["metrics"]["train/loss"], float)
        shutil.rmtree(d)

    # -- Resumability: JSONL survives restart --------------------------------

    def test_jsonl_resumability():
        d = _tmpdir()
        jsonl_path = os.path.join(d, "metrics.jsonl")
        cfg = LoggingConfig(
            tensorboard_enabled=False, wandb_enabled=False, jsonl_enabled=True
        )
        # Session 1
        ml1 = MetricLogger(d, cfg, phase=1)
        ml1.log_scalar("train/loss", 0.9, step=0)
        ml1.log_scalar("train/loss", 0.7, step=1)
        ml1.close()
        # Session 2 (resume)
        ml2 = MetricLogger(d, cfg, phase=1)
        ml2.log_scalar("train/loss", 0.5, step=2)
        ml2.close()
        entries = JSONLMetricWriter.read_all(jsonl_path)
        assert len(entries) == 3
        assert entries[2]["step"] == 2
        shutil.rmtree(d)

    # -- JSONL timestamp presence ------------------------------------------

    def test_jsonl_has_timestamp():
        d = _tmpdir()
        cfg = LoggingConfig(
            tensorboard_enabled=False, wandb_enabled=False, jsonl_enabled=True
        )
        ml = MetricLogger(d, cfg, phase=1)
        ml.log_scalar("x", 1.0, step=0)
        ml.close()
        entries = JSONLMetricWriter.read_all(os.path.join(d, "metrics.jsonl"))
        assert "timestamp" in entries[0]
        # Should be parseable ISO
        ts = entries[0]["timestamp"]
        datetime.fromisoformat(ts)
        shutil.rmtree(d)

    # -- JSONL summarize edge: no metrics entries --------------------------

    def test_jsonl_summarize_empty():
        d = _tmpdir()
        path = os.path.join(d, "metrics.jsonl")
        w = JSONLMetricWriter(path)
        w.write({"event": "phase_transition", "from_phase": 1, "to_phase": 2})
        w.close()
        summary = JSONLMetricWriter.summarize(path)
        assert summary == {}
        shutil.rmtree(d)

    # -- Multiple phases via MetricLogger ----------------------------------

    def test_multi_phase_logging():
        d = _tmpdir()
        cfg = LoggingConfig(
            tensorboard_enabled=False, wandb_enabled=False, jsonl_enabled=True
        )
        ml = MetricLogger(d, cfg, phase=1)
        ml.log_scalar(MetricNamespace.phase_metric(1, "loss"), 0.5, step=0)
        ml.log_phase_transition(1, 2, step=100, metrics={"val/loss": 0.2})
        ml.log_scalar(MetricNamespace.phase_metric(2, "loss"), 0.3, step=101)
        ml.close()
        entries = JSONLMetricWriter.read_all(os.path.join(d, "metrics.jsonl"))
        assert len(entries) == 3
        assert entries[0]["phase"] == 1
        assert entries[1]["event"] == "phase_transition"
        assert entries[2]["phase"] == 2
        shutil.rmtree(d)

    # -- TB + JSONL consistency -------------------------------------------

    def test_tb_jsonl_consistency():
        d = _tmpdir()
        cfg = LoggingConfig(
            tensorboard_enabled=True, wandb_enabled=False, jsonl_enabled=True
        )
        ml = MetricLogger(d, cfg, phase=1)
        for i in range(5):
            ml.log_scalar("train/loss", 1.0 - i * 0.2, step=i)
        ml.close()
        entries = JSONLMetricWriter.read_all(os.path.join(d, "metrics.jsonl"))
        assert len(entries) == 5
        if _TB_AVAILABLE:
            tb_files = os.listdir(os.path.join(d, "tb"))
            assert len(tb_files) > 0
        shutil.rmtree(d)

    # -- GradientMonitor module grouping -----------------------------------

    def test_grad_norms_module_grouping():
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )
        x = torch.randn(2, 10)
        loss = model(x).sum()
        loss.backward()
        norms = GradientMonitor.compute_grad_norms(model.named_parameters())
        # Should have grouped weight and bias under module names like "0", "2"
        assert "0" in norms  # nn.Sequential uses integer keys
        assert "2" in norms

    # -- Close idempotency -------------------------------------------------

    def test_close_idempotent():
        d = _tmpdir()
        cfg = LoggingConfig(
            tensorboard_enabled=True, wandb_enabled=False, jsonl_enabled=True
        )
        ml = MetricLogger(d, cfg)
        ml.close()
        ml.close()  # second close should not raise
        shutil.rmtree(d)

    # -- Flush explicit call -----------------------------------------------

    def test_explicit_flush():
        d = _tmpdir()
        cfg = LoggingConfig(
            tensorboard_enabled=True, wandb_enabled=False, jsonl_enabled=True
        )
        ml = MetricLogger(d, cfg)
        ml.log_scalar("x", 1.0, step=0)
        ml.flush()
        ml.close()
        shutil.rmtree(d)

    # -- WandBBackend inactive methods should not raise --------------------

    def test_wandb_inactive_all_methods():
        wb = WandBBackend.__new__(WandBBackend)
        wb._active = False
        wb._run = None
        wb.log({"a": 1}, 0)
        wb.log_artifact("/nonexistent", "n", "model")
        wb.log_table("t", [[1]], ["c"])
        wb.alert("t", "msg", "INFO")
        wb.alert("t", "msg", "ERROR")
        assert wb.get_run_id() is None
        wb.finish(exit_code=1)

    # -- LoggingConfig wandb_resume_mode values ----------------------------

    def test_config_resume_modes():
        for mode in ("must", "allow", "never"):
            cfg = LoggingConfig(wandb_resume_mode=mode)
            assert cfg.wandb_resume_mode == mode

    # -- MetricNamespace.validate_tag additional cases ---------------------

    def test_ns_validate_dot_and_dash():
        MetricNamespace.validate_tag("train.loss-v2")

    def test_ns_validate_single_char():
        MetricNamespace.validate_tag("x")

    # =====================================================================
    # Run all tests
    # =====================================================================

    print("=" * 60)
    print("logging_utils_template.py  —  self-test suite")
    print("=" * 60)
    print(f"TensorBoard available: {_TB_AVAILABLE}")
    print(f"wandb available:       {_WANDB_AVAILABLE}")
    print("-" * 60)

    test_functions = [
        # MetricNamespace (11 tests)
        ("MetricNamespace.fixed_tags", test_ns_fixed_tags),
        ("MetricNamespace.phase_metric", test_ns_phase_metric),
        ("MetricNamespace.module_metric", test_ns_module_metric),
        ("MetricNamespace.system_metric", test_ns_system_metric),
        ("MetricNamespace.lr_metric", test_ns_lr_metric),
        ("MetricNamespace.grad_norm_metric", test_ns_grad_norm_metric),
        ("MetricNamespace.validate_ok", test_ns_validate_ok),
        ("MetricNamespace.validate_empty", test_ns_validate_empty),
        ("MetricNamespace.validate_leading_slash", test_ns_validate_leading_slash),
        ("MetricNamespace.validate_spaces", test_ns_validate_spaces),
        ("MetricNamespace.validate_invalid_chars", test_ns_validate_invalid_chars),
        # LoggingConfig (4 tests)
        ("LoggingConfig.defaults", test_config_defaults),
        ("LoggingConfig.to_dict", test_config_to_dict),
        ("LoggingConfig.from_dict", test_config_from_dict),
        ("LoggingConfig.roundtrip", test_config_roundtrip),
        # JSONLMetricWriter (5 tests)
        ("JSONL.write_read", test_jsonl_write_read),
        ("JSONL.append_mode", test_jsonl_append_mode),
        ("JSONL.summarize", test_jsonl_summarize),
        ("JSONL.summarize_skips_events", test_jsonl_summarize_skips_events),
        ("JSONL.special_event_entry", test_jsonl_special_event_entry),
        # TensorBoardBackend (7 tests)
        ("TB.creation", test_tb_backend_creation),
        ("TB.scalar", test_tb_backend_scalar),
        ("TB.noop_when_unavailable", test_tb_backend_noop_when_unavailable),
        ("TB.image", test_tb_backend_image),
        ("TB.image_resize", test_tb_backend_image_resize),
        ("TB.image_2d", test_tb_backend_image_2d),
        ("TB.custom_layout", test_tb_custom_layout),
        # WandBBackend (3 tests)
        ("WandB.noop_fallback", test_wandb_noop_fallback),
        ("WandB.properties", test_wandb_backend_properties),
        ("WandB.inactive_all_methods", test_wandb_inactive_all_methods),
        # MetricLogger (17 tests)
        ("Logger.creation_tb_only", test_logger_creation_tb_only),
        ("Logger.scalar_logging", test_logger_scalar_logging),
        ("Logger.batch_scalar_logging", test_logger_batch_scalar_logging),
        ("Logger.phase_transition", test_logger_phase_transition),
        ("Logger.phase_transition_no_metrics", test_logger_phase_transition_no_metrics),
        ("Logger.histogram", test_logger_histogram),
        ("Logger.image", test_logger_image),
        ("Logger.text", test_logger_text),
        ("Logger.context_manager", test_logger_context_manager),
        ("Logger.no_backends", test_logger_no_backends),
        ("Logger.auto_flush", test_logger_auto_flush),
        ("Logger.phase_step", test_logger_phase_step),
        ("Logger.weight_norms", test_logger_weight_norms),
        ("Logger.system_metrics", test_logger_system_metrics),
        ("Logger.log_gradient_norms", test_logger_log_gradient_norms),
        ("Logger.validates_tag", test_log_scalar_validates_tag),
        ("Logger.validates_tags_batch", test_log_scalars_validates_tags),
        # Additional tests (9 tests)
        ("Logger.coerces_int", test_log_scalar_coerces_int),
        ("Logger.resumability", test_jsonl_resumability),
        ("Logger.jsonl_has_timestamp", test_jsonl_has_timestamp),
        ("JSONL.summarize_empty", test_jsonl_summarize_empty),
        ("Logger.multi_phase", test_multi_phase_logging),
        ("Logger.tb_jsonl_consistency", test_tb_jsonl_consistency),
        ("GradMonitor.module_grouping", test_grad_norms_module_grouping),
        ("Logger.close_idempotent", test_close_idempotent),
        ("Logger.explicit_flush", test_explicit_flush),
        # GradientMonitor (6 tests)
        ("GradMonitor.norms", test_grad_monitor_norms),
        ("GradMonitor.global_norm", test_grad_monitor_global_norm),
        ("GradMonitor.no_grads", test_grad_monitor_no_grads),
        ("GradMonitor.anomaly_exploding", test_grad_anomaly_exploding),
        ("GradMonitor.anomaly_vanishing", test_grad_anomaly_vanishing),
        ("GradMonitor.anomaly_both", test_grad_anomaly_both),
        ("GradMonitor.anomaly_none", test_grad_anomaly_none),
        ("GradMonitor.log_all", test_grad_monitor_log_all),
        # Config edge cases (3 tests)
        ("Config.resume_modes", test_config_resume_modes),
        ("Namespace.dot_and_dash", test_ns_validate_dot_and_dash),
        ("Namespace.single_char", test_ns_validate_single_char),
    ]

    for name, fn in test_functions:
        _run_test(name, fn)

    print("-" * 60)
    total = _passed + _failed
    print(f"Results: {_passed}/{total} passed, {_failed} failed")
    if _errors:
        print("\nFailures:")
        for err in _errors:
            print(f"  - {err}")
    print("=" * 60)
    sys.exit(1 if _failed else 0)
