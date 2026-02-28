"""
brain_ai/meta/meta_metrics.py -- Meta-Learning Diagnostic Metrics & Checkpoint I/O

Comprehensive diagnostics for meta-learning (MAML / MAML++ / Reptile) training
loops.  This module provides:

* **AdaptationCurve** -- Per-task accuracy/loss trajectories across inner-loop
  adaptation steps, including AUAC (Area Under Adaptation Curve).
* **GradientDiagnostics / GradientTracker** -- Per-step gradient norms, update
  magnitudes, clipping events, and zero-gradient counts.
* **LSLRStatistics** -- Summary statistics for MAML++'s learned per-layer
  per-step learning rates.
* **EpochMetrics / MetricsAggregator** -- Episode-level collection with
  epoch-level aggregation.
* **MetaCheckpoint** -- Save/load checkpoints with full hyperparameter state,
  MAML++ extensions (LSLR, MSL), RNG state, and metrics history.
* **MetricsLogger** -- JSON-based metrics logging for post-hoc analysis.

Design principles:
    1. No external dependencies beyond PyTorch and the standard library.
    2. Every dataclass is JSON-serialisable via .to_dict().
    3. Numerical stability: NaN/Inf detection with explicit warnings.
    4. All computation is done in fp32 regardless of input tensor dtype.
    5. Graceful handling of empty/degenerate cases (zero episodes, single step).

References:
    Finn et al. (2017) "Model-Agnostic Meta-Learning for Fast Adaptation"
    Antoniou et al. (2019) "How to Train Your MAML" (MAML++)
"""

from __future__ import annotations

import copy
import json
import math
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NAN = float('nan')
_CHECKPOINT_FORMAT_VERSION = "1.0"
_MAX_METRICS_HISTORY_IN_CHECKPOINT = 10


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _safe_mean(values: Sequence[float]) -> float:
    """Compute mean of a sequence, returning 0.0 for empty sequences."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _safe_div(numerator: float, denominator: float) -> float:
    """Safe division returning 0.0 when denominator is zero."""
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def _check_finite(value: float, name: str) -> float:
    """Warn if value is NaN or Inf, return value unchanged."""
    if math.isnan(value) or math.isinf(value):
        warnings.warn(
            f"Non-finite metric detected: {name} = {value}",
            RuntimeWarning, stacklevel=3,
        )
    return value


def _trapezoidal_area(values: List[float]) -> float:
    """Compute normalised trapezoidal area under unit-spaced y-values."""
    n = len(values)
    if n == 0:
        return 0.0
    if n == 1:
        return values[0]
    raw_area = sum(
        (values[i] + values[i + 1]) / 2.0 for i in range(n - 1)
    )
    return raw_area / (n - 1)


# ---------------------------------------------------------------------------
# 1. Adaptation Curve
# ---------------------------------------------------------------------------

@dataclass
class AdaptationCurve:
    """Per-step accuracy and loss trajectory during inner-loop adaptation.

    Tracks accuracy at step 0 (pre-adaptation) through step S (post-adaptation).

    Attributes:
        step_accuracies: [acc_step0, acc_step1, ..., acc_stepS].
        step_losses: [loss_step0, loss_step1, ..., loss_stepS].
    """
    step_accuracies: List[float]
    step_losses: List[float]

    def __post_init__(self) -> None:
        if len(self.step_accuracies) != len(self.step_losses):
            warnings.warn(
                f"AdaptationCurve: step_accuracies length "
                f"({len(self.step_accuracies)}) != step_losses length "
                f"({len(self.step_losses)}).",
                RuntimeWarning, stacklevel=2,
            )
        if not self.step_accuracies:
            warnings.warn(
                "AdaptationCurve created with empty step_accuracies.",
                RuntimeWarning, stacklevel=2,
            )

    @property
    def num_steps(self) -> int:
        return len(self.step_accuracies)

    @property
    def pre_adapt_acc(self) -> float:
        """Accuracy at step 0 (before any inner-loop updates)."""
        return self.step_accuracies[0] if self.step_accuracies else 0.0

    @property
    def post_adapt_acc(self) -> float:
        """Accuracy at the final inner-loop step."""
        return self.step_accuracies[-1] if self.step_accuracies else 0.0

    @property
    def pre_adapt_loss(self) -> float:
        return self.step_losses[0] if self.step_losses else 0.0

    @property
    def post_adapt_loss(self) -> float:
        return self.step_losses[-1] if self.step_losses else 0.0

    @property
    def fast_adaptation_gain(self) -> float:
        """acc(step 1) - acc(step 0). Measures how much the first step helps."""
        if len(self.step_accuracies) < 2:
            return 0.0
        return self.step_accuracies[1] - self.step_accuracies[0]

    @property
    def total_gain(self) -> float:
        """acc(final) - acc(step 0)."""
        return self.post_adapt_acc - self.pre_adapt_acc

    @property
    def loss_reduction(self) -> float:
        """loss(step 0) - loss(final). Positive means loss decreased."""
        return self.pre_adapt_loss - self.post_adapt_loss

    @property
    def auac(self) -> float:
        """Area Under Adaptation Curve (trapezoidal rule, normalised to [0,1])."""
        return _trapezoidal_area(self.step_accuracies)

    @property
    def auac_loss(self) -> float:
        """Area Under Loss Curve (trapezoidal, normalised)."""
        return _trapezoidal_area(self.step_losses)

    @property
    def is_monotonic_increasing(self) -> bool:
        """Whether accuracy is monotonically non-decreasing across steps."""
        for i in range(len(self.step_accuracies) - 1):
            if self.step_accuracies[i + 1] < self.step_accuracies[i] - 1e-7:
                return False
        return True

    @property
    def adaptation_efficiency(self) -> float:
        """Ratio of fast_adaptation_gain to total_gain."""
        return _safe_div(self.fast_adaptation_gain, self.total_gain)

    def to_dict(self) -> dict:
        return {
            'step_accuracies': list(self.step_accuracies),
            'step_losses': list(self.step_losses),
            'num_steps': self.num_steps,
            'pre_adapt_acc': self.pre_adapt_acc,
            'post_adapt_acc': self.post_adapt_acc,
            'pre_adapt_loss': self.pre_adapt_loss,
            'post_adapt_loss': self.post_adapt_loss,
            'fast_adaptation_gain': self.fast_adaptation_gain,
            'total_gain': self.total_gain,
            'loss_reduction': self.loss_reduction,
            'auac': self.auac,
            'auac_loss': self.auac_loss,
            'is_monotonic_increasing': self.is_monotonic_increasing,
            'adaptation_efficiency': self.adaptation_efficiency,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AdaptationCurve:
        return cls(
            step_accuracies=d['step_accuracies'],
            step_losses=d['step_losses'],
        )

    def __repr__(self) -> str:
        return (
            f"AdaptationCurve(steps={self.num_steps}, "
            f"pre={self.pre_adapt_acc:.4f}, post={self.post_adapt_acc:.4f}, "
            f"gain={self.total_gain:.4f}, auac={self.auac:.4f})"
        )


# ---------------------------------------------------------------------------
# 2. Gradient Diagnostics
# ---------------------------------------------------------------------------

@dataclass
class GradientDiagnostics:
    """Per-step gradient statistics for inner-loop debugging."""
    step: int
    grad_norm: float         # L2 norm of all gradients
    update_norm: float       # L2 norm of parameter updates (lr * grad)
    grad_max: float          # max absolute gradient value
    grad_min: float          # min absolute gradient value (non-zero)
    grad_mean: float = 0.0
    clipped: bool = False    # whether gradient clipping was applied
    num_zero_grads: int = 0  # number of parameters with zero gradients
    num_params: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_healthy(self) -> bool:
        """Heuristic: norm finite and in (1e-8, 1000), <20% zero grads."""
        if math.isnan(self.grad_norm) or math.isinf(self.grad_norm):
            return False
        if self.grad_norm < 1e-8 or self.grad_norm > 1000.0:
            return False
        if self.num_params > 0 and self.num_zero_grads / self.num_params > 0.2:
            return False
        return True

    def __repr__(self) -> str:
        clip_str = " CLIPPED" if self.clipped else ""
        return (
            f"GradDiag(step={self.step}, norm={self.grad_norm:.4f}, "
            f"update={self.update_norm:.6f}, zeros={self.num_zero_grads}{clip_str})"
        )


class GradientTracker:
    """Track gradient statistics across inner-loop adaptation steps."""

    def __init__(self) -> None:
        self.step_diagnostics: List[GradientDiagnostics] = []

    def record_step(
        self, step: int,
        grads: Union[Tuple[Optional[Tensor], ...], List[Optional[Tensor]]],
        lr: float, clipped: bool = False,
    ) -> GradientDiagnostics:
        """Record gradient statistics for one inner-loop step."""
        non_none = [g for g in grads if g is not None]
        total_params = len(non_none)

        if not non_none:
            diag = GradientDiagnostics(
                step=step, grad_norm=0.0, update_norm=0.0,
                grad_max=0.0, grad_min=0.0, grad_mean=0.0,
                clipped=clipped, num_zero_grads=0, num_params=0,
            )
            self.step_diagnostics.append(diag)
            return diag

        grad_norms = [g.float().norm().item() for g in non_none]
        total_norm = math.sqrt(sum(n ** 2 for n in grad_norms))
        abs_vals = torch.cat([g.float().abs().flatten() for g in non_none])
        grad_max = abs_vals.max().item()
        grad_mean = abs_vals.mean().item()
        positive_mask = abs_vals > 0
        grad_min = abs_vals[positive_mask].min().item() if positive_mask.any() else 0.0
        num_zero = sum(1 for g in non_none if g.abs().sum().item() == 0.0)

        diag = GradientDiagnostics(
            step=step,
            grad_norm=_check_finite(total_norm, f"grad_norm[step={step}]"),
            update_norm=_check_finite(total_norm * lr, f"update_norm[step={step}]"),
            grad_max=grad_max, grad_min=grad_min, grad_mean=grad_mean,
            clipped=clipped, num_zero_grads=num_zero, num_params=total_params,
        )
        self.step_diagnostics.append(diag)
        return diag

    def get_summary(self) -> Dict[str, float]:
        """Aggregate gradient statistics across all recorded steps."""
        if not self.step_diagnostics:
            return {}
        norms = [d.grad_norm for d in self.step_diagnostics]
        update_norms = [d.update_norm for d in self.step_diagnostics]
        return {
            'mean_grad_norm': _safe_mean(norms),
            'max_grad_norm': max(norms),
            'min_grad_norm': min(norms),
            'mean_update_norm': _safe_mean(update_norms),
            'max_update_norm': max(update_norms),
            'mean_grad_max': _safe_mean([d.grad_max for d in self.step_diagnostics]),
            'mean_grad_mean': _safe_mean([d.grad_mean for d in self.step_diagnostics]),
            'num_clip_events': sum(1 for d in self.step_diagnostics if d.clipped),
            'total_zero_grads': sum(d.num_zero_grads for d in self.step_diagnostics),
            'num_steps_recorded': len(self.step_diagnostics),
            'all_healthy': all(d.is_healthy for d in self.step_diagnostics),
        }

    def get_norm_trajectory(self) -> List[float]:
        return [d.grad_norm for d in self.step_diagnostics]

    def has_exploding_gradients(self, threshold: float = 100.0) -> bool:
        return any(d.grad_norm > threshold for d in self.step_diagnostics)

    def has_vanishing_gradients(self, threshold: float = 1e-7) -> bool:
        return any(
            d.grad_norm < threshold and d.num_params > 0
            for d in self.step_diagnostics
        )

    def clear(self) -> None:
        self.step_diagnostics.clear()

    def __len__(self) -> int:
        return len(self.step_diagnostics)

    def __repr__(self) -> str:
        if not self.step_diagnostics:
            return "GradientTracker(empty)"
        norms = [d.grad_norm for d in self.step_diagnostics]
        return f"GradientTracker(steps={len(self)}, norm_range=[{min(norms):.4f}, {max(norms):.4f}])"


# ---------------------------------------------------------------------------
# 3. LSLR (Learned Per-Layer Per-Step Learning Rate) Statistics
# ---------------------------------------------------------------------------

@dataclass
class LSLRStatistics:
    """Statistics about learned per-layer per-step learning rates."""
    lr_mean: float
    lr_min: float
    lr_max: float
    lr_std: float
    per_layer_means: Dict[str, float]
    per_step_means: Dict[int, float]
    num_layers: int = 0
    num_steps: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d['per_step_means'] = {str(k): v for k, v in self.per_step_means.items()}
        return d

    @property
    def lr_range(self) -> float:
        return self.lr_max - self.lr_min

    @property
    def has_negative_lrs(self) -> bool:
        return self.lr_min < 0.0

    def __repr__(self) -> str:
        return (
            f"LSLRStatistics(mean={self.lr_mean:.6f}, "
            f"range=[{self.lr_min:.6f}, {self.lr_max:.6f}], "
            f"std={self.lr_std:.6f})"
        )


def compute_lslr_statistics(lslr_module: Any) -> LSLRStatistics:
    """Compute statistics from an LSLR module.

    The lslr_module must have .layer_names, .num_steps, and .get_lr(layer, step).
    """
    all_lrs: List[float] = []
    per_layer: Dict[str, float] = {}
    per_step: Dict[int, List[float]] = {}

    for layer in lslr_module.layer_names:
        layer_lrs: List[float] = []
        for step in range(lslr_module.num_steps):
            lr_val = lslr_module.get_lr(layer, step).item()
            all_lrs.append(lr_val)
            layer_lrs.append(lr_val)
            per_step.setdefault(step, []).append(lr_val)
        per_layer[layer] = _safe_mean(layer_lrs)

    per_step_means = {step: _safe_mean(vals) for step, vals in per_step.items()}

    if not all_lrs:
        return LSLRStatistics(
            lr_mean=0.0, lr_min=0.0, lr_max=0.0, lr_std=0.0,
            per_layer_means={}, per_step_means={}, num_layers=0, num_steps=0,
        )

    lr_tensor = torch.tensor(all_lrs, dtype=torch.float32)
    return LSLRStatistics(
        lr_mean=lr_tensor.mean().item(),
        lr_min=lr_tensor.min().item(),
        lr_max=lr_tensor.max().item(),
        lr_std=lr_tensor.std().item() if len(all_lrs) > 1 else 0.0,
        per_layer_means=per_layer,
        per_step_means=per_step_means,
        num_layers=len(lslr_module.layer_names),
        num_steps=lslr_module.num_steps,
    )


# ---------------------------------------------------------------------------
# 4. Epoch Metrics Aggregator
# ---------------------------------------------------------------------------

@dataclass
class EpochMetrics:
    """Aggregated metrics for one meta-training epoch."""
    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    pre_adapt_acc: float = 0.0
    post_adapt_acc: float = 0.0
    fast_gain: float = 0.0
    total_gain: float = 0.0
    auac: float = 0.0
    grad_summary: Dict[str, float] = field(default_factory=dict)
    lslr_stats: Optional[LSLRStatistics] = None
    s2_fraction: Optional[float] = None  # fraction using System 2
    wall_time: float = 0.0               # seconds for this epoch
    num_episodes: int = 0
    val_accuracy: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            'epoch': self.epoch,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'pre_adapt_acc': self.pre_adapt_acc,
            'post_adapt_acc': self.post_adapt_acc,
            'fast_gain': self.fast_gain,
            'total_gain': self.total_gain,
            'auac': self.auac,
            'grad_summary': dict(self.grad_summary),
            'lslr_stats': self.lslr_stats.to_dict() if self.lslr_stats else None,
            's2_fraction': self.s2_fraction,
            'wall_time': self.wall_time,
            'num_episodes': self.num_episodes,
            'val_accuracy': self.val_accuracy,
        }

    @classmethod
    def from_dict(cls, d: dict) -> EpochMetrics:
        lslr_data = d.get('lslr_stats')
        lslr_stats = None
        if lslr_data is not None:
            per_step = {int(k): v for k, v in lslr_data.get('per_step_means', {}).items()}
            lslr_stats = LSLRStatistics(
                lr_mean=lslr_data['lr_mean'], lr_min=lslr_data['lr_min'],
                lr_max=lslr_data['lr_max'], lr_std=lslr_data['lr_std'],
                per_layer_means=lslr_data.get('per_layer_means', {}),
                per_step_means=per_step,
                num_layers=lslr_data.get('num_layers', 0),
                num_steps=lslr_data.get('num_steps', 0),
            )
        return cls(
            epoch=d['epoch'], train_loss=d['train_loss'],
            val_loss=d.get('val_loss'),
            pre_adapt_acc=d.get('pre_adapt_acc', 0.0),
            post_adapt_acc=d.get('post_adapt_acc', 0.0),
            fast_gain=d.get('fast_gain', 0.0),
            total_gain=d.get('total_gain', 0.0),
            auac=d.get('auac', 0.0),
            grad_summary=d.get('grad_summary', {}),
            lslr_stats=lslr_stats,
            s2_fraction=d.get('s2_fraction'),
            wall_time=d.get('wall_time', 0.0),
            num_episodes=d.get('num_episodes', 0),
            val_accuracy=d.get('val_accuracy'),
        )

    @property
    def improvement_ratio(self) -> float:
        return _safe_div(self.post_adapt_acc, self.pre_adapt_acc)

    @property
    def episodes_per_second(self) -> float:
        return _safe_div(float(self.num_episodes), self.wall_time)

    def __repr__(self) -> str:
        return (
            f"EpochMetrics(epoch={self.epoch}, loss={self.train_loss:.4f}, "
            f"pre={self.pre_adapt_acc:.4f}, post={self.post_adapt_acc:.4f}, "
            f"auac={self.auac:.4f}, eps={self.num_episodes})"
        )


class MetricsAggregator:
    """Collect per-episode metrics and aggregate into epoch-level summaries."""

    def __init__(self) -> None:
        self.episode_curves: List[AdaptationCurve] = []
        self.episode_losses: List[float] = []
        self.episode_val_losses: List[float] = []
        self.grad_trackers: List[GradientTracker] = []
        self.s2_flags: List[bool] = []
        self.start_time: Optional[float] = None
        self._epoch_started: bool = False

    def start_epoch(self) -> None:
        self.episode_curves = []
        self.episode_losses = []
        self.episode_val_losses = []
        self.grad_trackers = []
        self.s2_flags = []
        self.start_time = time.time()
        self._epoch_started = True

    def record_episode(
        self, curve: AdaptationCurve, loss: float,
        grad_tracker: Optional[GradientTracker] = None,
        val_loss: Optional[float] = None,
        used_system2: Optional[bool] = None,
    ) -> None:
        if not self._epoch_started:
            warnings.warn(
                "MetricsAggregator.record_episode() called without start_epoch(). "
                "Auto-starting.", RuntimeWarning, stacklevel=2,
            )
            self.start_epoch()

        self.episode_curves.append(curve)
        self.episode_losses.append(loss)
        if grad_tracker is not None:
            self.grad_trackers.append(grad_tracker)
        if val_loss is not None:
            self.episode_val_losses.append(val_loss)
        if used_system2 is not None:
            self.s2_flags.append(used_system2)

    def _aggregate_grad_summaries(self) -> Dict[str, float]:
        if not self.grad_trackers:
            return {}
        summaries = [t.get_summary() for t in self.grad_trackers if t.get_summary()]
        if not summaries:
            return {}

        result: Dict[str, float] = {}
        for key in ['mean_grad_norm', 'mean_update_norm', 'mean_grad_max', 'mean_grad_mean']:
            vals = [s[key] for s in summaries if key in s]
            if vals:
                result[key] = _safe_mean(vals)
        for key in ['max_grad_norm', 'max_update_norm']:
            vals = [s[key] for s in summaries if key in s]
            if vals:
                result[key] = max(vals)
        for key in ['min_grad_norm']:
            vals = [s.get(key, 0) for s in summaries]
            if vals:
                result[key] = min(vals)
        for key in ['num_clip_events', 'total_zero_grads']:
            result[key] = sum(s.get(key, 0) for s in summaries)
        result['num_trackers'] = float(len(summaries))
        result['all_healthy'] = float(all(s.get('all_healthy', True) for s in summaries))
        return result

    def finalize_epoch(
        self, epoch: int,
        lslr_stats: Optional[LSLRStatistics] = None,
        val_accuracy: Optional[float] = None,
    ) -> EpochMetrics:
        n = len(self.episode_curves)
        wall_time = time.time() - self.start_time if self.start_time else 0.0
        s2_fraction = None
        if self.s2_flags:
            s2_fraction = sum(1 for f in self.s2_flags if f) / len(self.s2_flags)
        val_loss = _safe_mean(self.episode_val_losses) if self.episode_val_losses else None

        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=_safe_mean(self.episode_losses),
            val_loss=val_loss,
            pre_adapt_acc=_safe_mean([c.pre_adapt_acc for c in self.episode_curves]),
            post_adapt_acc=_safe_mean([c.post_adapt_acc for c in self.episode_curves]),
            fast_gain=_safe_mean([c.fast_adaptation_gain for c in self.episode_curves]),
            total_gain=_safe_mean([c.total_gain for c in self.episode_curves]),
            auac=_safe_mean([c.auac for c in self.episode_curves]),
            grad_summary=self._aggregate_grad_summaries(),
            lslr_stats=lslr_stats, s2_fraction=s2_fraction,
            wall_time=wall_time, num_episodes=n, val_accuracy=val_accuracy,
        )
        self._epoch_started = False
        return metrics

    @property
    def current_episode_count(self) -> int:
        return len(self.episode_curves)


# ---------------------------------------------------------------------------
# 5. Checkpoint I/O
# ---------------------------------------------------------------------------

class MetaCheckpoint:
    """Save and load meta-learning checkpoints with full hyperparameter state."""

    FORMAT_VERSION = _CHECKPOINT_FORMAT_VERSION

    @staticmethod
    def save(
        path: Union[str, Path],
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Any,
        epoch: int,
        metrics_history: List[EpochMetrics],
        *,
        scheduler: Optional[Any] = None,
        lslr_module: Optional[nn.Module] = None,
        msl_module: Optional[nn.Module] = None,
        sampler_config: Optional[dict] = None,
        best_val_accuracy: float = 0.0,
        rng_states: Optional[dict] = None,
        extra: Optional[dict] = None,
    ) -> None:
        """Save a meta-learning checkpoint."""
        import random
        try:
            import numpy as np
            has_numpy = True
        except ImportError:
            has_numpy = False

        def _cfg(attr: str, default: Any = None) -> Any:
            if hasattr(config, 'maml'):
                return getattr(config.maml, attr, default)
            return getattr(config, attr, default)

        if rng_states is None:
            rng_states = {
                'torch': torch.random.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                'python': random.getstate(),
            }
            if has_numpy:
                rng_states['numpy'] = np.random.get_state()

        maml_plus = None
        if lslr_module is not None or msl_module is not None:
            maml_plus = {}
            if lslr_module is not None:
                maml_plus['lslr_state_dict'] = lslr_module.state_dict()
            if msl_module is not None:
                maml_plus['msl_state_dict'] = msl_module.state_dict()

        recent_metrics = metrics_history[-_MAX_METRICS_HISTORY_IN_CHECKPOINT:]

        checkpoint = {
            'format_version': MetaCheckpoint.FORMAT_VERSION,
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'model_state_dict': model.state_dict(),
            'algo': _cfg('algo', 'maml'),
            'second_order': _cfg('second_order', True),
            'inner_steps': _cfg('inner_steps', 5),
            'inner_lr': _cfg('inner_lr', 0.01),
            'inner_clip': _cfg('inner_clip', 10.0),
            'backend': _cfg('backend', 'auto'),
            'maml_plus': maml_plus,
            'outer_optimizer_state_dict': optimizer.state_dict(),
            'outer_scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'sampler_config': sampler_config,
            'epoch': epoch,
            'best_val_accuracy': best_val_accuracy,
            'rng_state': rng_states,
            'metrics_history': [m.to_dict() for m in recent_metrics],
        }
        if extra is not None:
            checkpoint['extra'] = extra

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix('.tmp')
        torch.save(checkpoint, tmp_path)
        tmp_path.rename(path)

    @staticmethod
    def load(
        path: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        lslr_module: Optional[nn.Module] = None,
        msl_module: Optional[nn.Module] = None,
        strict: bool = True,
        map_location: str = 'cpu',
        restore_rng: bool = False,
    ) -> dict:
        """Load a meta-learning checkpoint and restore state."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        ckpt = torch.load(path, map_location=map_location, weights_only=False)

        version = ckpt.get('format_version', '0.0')
        if version.split('.')[0] != MetaCheckpoint.FORMAT_VERSION.split('.')[0]:
            raise ValueError(
                f"Incompatible checkpoint format version: {version} "
                f"(expected major version {MetaCheckpoint.FORMAT_VERSION.split('.')[0]})"
            )

        model.load_state_dict(ckpt['model_state_dict'], strict=strict)

        if optimizer and 'outer_optimizer_state_dict' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['outer_optimizer_state_dict'])
            except (ValueError, KeyError) as e:
                warnings.warn(f"Could not restore optimizer state: {e}",
                              RuntimeWarning, stacklevel=2)

        if scheduler and ckpt.get('outer_scheduler_state_dict'):
            try:
                scheduler.load_state_dict(ckpt['outer_scheduler_state_dict'])
            except (ValueError, KeyError) as e:
                warnings.warn(f"Could not restore scheduler state: {e}",
                              RuntimeWarning, stacklevel=2)

        maml_plus = ckpt.get('maml_plus') or {}
        if lslr_module and 'lslr_state_dict' in maml_plus:
            try:
                lslr_module.load_state_dict(maml_plus['lslr_state_dict'])
            except (ValueError, KeyError) as e:
                warnings.warn(f"Could not restore LSLR state: {e}",
                              RuntimeWarning, stacklevel=2)

        if msl_module and 'msl_state_dict' in maml_plus:
            try:
                msl_module.load_state_dict(maml_plus['msl_state_dict'])
            except (ValueError, KeyError) as e:
                warnings.warn(f"Could not restore MSL state: {e}",
                              RuntimeWarning, stacklevel=2)

        if restore_rng and 'rng_state' in ckpt:
            import random
            rng = ckpt['rng_state']
            if rng.get('torch') is not None:
                torch.random.set_rng_state(rng['torch'])
            if rng.get('cuda') is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng['cuda'])
            if rng.get('python') is not None:
                random.setstate(rng['python'])

        return ckpt

    @staticmethod
    def diff_config(ckpt: dict, current_config: Any) -> Dict[str, Tuple[Any, Any]]:
        """Compare checkpoint config with current config, return differences."""
        diffs: Dict[str, Tuple[Any, Any]] = {}
        for key in ['algo', 'inner_steps', 'inner_lr', 'second_order', 'backend', 'inner_clip']:
            ckpt_val = ckpt.get(key)
            if hasattr(current_config, 'maml'):
                config_val = getattr(current_config.maml, key, None)
            else:
                config_val = getattr(current_config, key, None)
            if ckpt_val is not None and config_val is not None and ckpt_val != config_val:
                diffs[key] = (ckpt_val, config_val)
        return diffs

    @staticmethod
    def get_metrics_history(ckpt: dict) -> List[EpochMetrics]:
        """Extract and reconstruct EpochMetrics from checkpoint."""
        raw = ckpt.get('metrics_history', [])
        result = []
        for entry in raw:
            try:
                result.append(EpochMetrics.from_dict(entry))
            except (KeyError, TypeError) as e:
                warnings.warn(f"Could not reconstruct EpochMetrics: {e}",
                              RuntimeWarning, stacklevel=2)
        return result

    @staticmethod
    def get_info(path: Union[str, Path], map_location: str = 'cpu') -> dict:
        """Quick inspection of a checkpoint without loading model weights."""
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        return {
            'format_version': ckpt.get('format_version'),
            'created_at': ckpt.get('created_at'),
            'epoch': ckpt.get('epoch'),
            'best_val_accuracy': ckpt.get('best_val_accuracy'),
            'algo': ckpt.get('algo'),
            'inner_steps': ckpt.get('inner_steps'),
            'inner_lr': ckpt.get('inner_lr'),
            'second_order': ckpt.get('second_order'),
            'has_lslr': bool(ckpt.get('maml_plus', {}).get('lslr_state_dict')),
            'has_msl': bool(ckpt.get('maml_plus', {}).get('msl_state_dict')),
            'num_metrics_entries': len(ckpt.get('metrics_history', [])),
            'model_param_keys': len(ckpt.get('model_state_dict', {})),
        }


# ---------------------------------------------------------------------------
# 6. Metrics JSON Logger
# ---------------------------------------------------------------------------

class MetricsLogger:
    """Write epoch metrics to a JSON file for post-hoc analysis."""

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: List[dict] = []

    def log_epoch(self, metrics: EpochMetrics) -> None:
        self.entries.append(metrics.to_dict())
        self._flush()

    def _flush(self) -> None:
        tmp_path = self.path.with_suffix('.tmp')
        with open(tmp_path, 'w') as f:
            json.dump(self.entries, f, indent=2, default=str)
        tmp_path.rename(self.path)

    def load(self) -> List[dict]:
        if self.path.exists():
            with open(self.path) as f:
                return json.load(f)
        return []

    def load_as_epoch_metrics(self) -> List[EpochMetrics]:
        return [EpochMetrics.from_dict(d) for d in self.load()]

    def get_best_epoch(self, key: str = 'post_adapt_acc', maximize: bool = True) -> Optional[dict]:
        valid = [e for e in self.entries if key in e and e[key] is not None]
        if not valid:
            return None
        return max(valid, key=lambda e: e[key]) if maximize else min(valid, key=lambda e: e[key])

    @property
    def num_entries(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return f"MetricsLogger(path={self.path}, entries={self.num_entries})"


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def summarize_epoch(metrics: EpochMetrics) -> str:
    """Format an EpochMetrics into a human-readable one-line summary."""
    parts = [
        f"Epoch {metrics.epoch:3d}",
        f"loss={metrics.train_loss:.4f}",
        f"pre={metrics.pre_adapt_acc:.3f}",
        f"post={metrics.post_adapt_acc:.3f}",
        f"gain={metrics.total_gain:+.3f}",
        f"auac={metrics.auac:.3f}",
        f"eps={metrics.num_episodes}",
        f"time={metrics.wall_time:.1f}s",
    ]
    if metrics.val_accuracy is not None:
        parts.append(f"val={metrics.val_accuracy:.3f}")
    if metrics.s2_fraction is not None:
        parts.append(f"s2={metrics.s2_fraction:.2f}")
    grad = metrics.grad_summary
    if grad:
        parts.append(f"gnorm={grad.get('mean_grad_norm', 0.0):.3f}")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    'AdaptationCurve',
    'GradientDiagnostics',
    'GradientTracker',
    'LSLRStatistics',
    'EpochMetrics',
    'MetricsAggregator',
    'MetaCheckpoint',
    'MetricsLogger',
    'compute_lslr_statistics',
    'summarize_epoch',
]


# ===================================================================
# SELF-TEST
# ===================================================================

if __name__ == '__main__':
    import sys
    import tempfile
    import os

    _pass_count = 0
    _fail_count = 0

    def _report(name: str, passed: bool, detail: str = "") -> None:
        global _pass_count, _fail_count
        if passed:
            _pass_count += 1
            print(f"  PASS  {name}")
        else:
            _fail_count += 1
            msg = f"  FAIL  {name}"
            if detail:
                msg += f"  -- {detail}"
            print(msg)

    print("=" * 72)
    print("meta_metrics.py  --  Self-Test Suite")
    print("=" * 72)

    # Helper: mock config for checkpoint tests
    class _MockMAMLConfig:
        algo = 'maml'
        second_order = True
        inner_steps = 5
        inner_lr = 0.01
        inner_clip = 10.0
        backend = 'auto'

    class _MockConfig:
        maml = _MockMAMLConfig()

    # ------------------------------------------------------------------
    # Test 1: AdaptationCurve properties
    # ------------------------------------------------------------------
    print("\n--- Test 1: AdaptationCurve properties ---")
    try:
        curve = AdaptationCurve(
            step_accuracies=[0.2, 0.5, 0.7, 0.85, 0.9],
            step_losses=[2.0, 1.5, 1.0, 0.6, 0.3],
        )
        _report("pre_adapt_acc", curve.pre_adapt_acc == 0.2,
                f"expected 0.2, got {curve.pre_adapt_acc}")
        _report("post_adapt_acc", curve.post_adapt_acc == 0.9,
                f"expected 0.9, got {curve.post_adapt_acc}")
        _report("fast_adaptation_gain", abs(curve.fast_adaptation_gain - 0.3) < 1e-9,
                f"expected 0.3, got {curve.fast_adaptation_gain}")
        _report("total_gain", abs(curve.total_gain - 0.7) < 1e-9,
                f"expected 0.7, got {curve.total_gain}")
    except Exception as e:
        _report("AdaptationCurve properties", False, str(e))

    # ------------------------------------------------------------------
    # Test 2: AUAC computation with known values
    # ------------------------------------------------------------------
    print("\n--- Test 2: AUAC trapezoidal correctness ---")
    try:
        # Linear [0, 0.25, 0.5, 0.75, 1.0] -> area = 2.0 / 4 = 0.5
        linear = AdaptationCurve([0.0, 0.25, 0.5, 0.75, 1.0], [1.0, 0.75, 0.5, 0.25, 0.0])
        _report("AUAC linear", abs(linear.auac - 0.5) < 1e-9,
                f"expected 0.5, got {linear.auac}")

        # Constant at 0.8 -> AUAC = 0.8
        const = AdaptationCurve([0.8, 0.8, 0.8], [0.5, 0.5, 0.5])
        _report("AUAC constant", abs(const.auac - 0.8) < 1e-9,
                f"expected 0.8, got {const.auac}")

        # Step function [0, 0, 0, 1, 1] -> 1.5 / 4 = 0.375
        step_fn = AdaptationCurve([0.0, 0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0, 0.0])
        _report("AUAC step function", abs(step_fn.auac - 0.375) < 1e-9,
                f"expected 0.375, got {step_fn.auac}")

        # Perfect [1, 1, 1] -> 1.0
        perfect = AdaptationCurve([1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
        _report("AUAC perfect", abs(perfect.auac - 1.0) < 1e-9,
                f"expected 1.0, got {perfect.auac}")
    except Exception as e:
        _report("AUAC trapezoidal", False, str(e))

    # ------------------------------------------------------------------
    # Test 3: AUAC with single step
    # ------------------------------------------------------------------
    print("\n--- Test 3: AUAC single step ---")
    try:
        single = AdaptationCurve([0.42], [1.5])
        _report("AUAC single step value", abs(single.auac - 0.42) < 1e-9,
                f"expected 0.42, got {single.auac}")
        _report("fast_gain single step", single.fast_adaptation_gain == 0.0)
    except Exception as e:
        _report("AUAC single step", False, str(e))

    # ------------------------------------------------------------------
    # Test 4: GradientDiagnostics creation
    # ------------------------------------------------------------------
    print("\n--- Test 4: GradientDiagnostics creation ---")
    try:
        diag = GradientDiagnostics(
            step=0, grad_norm=1.5, update_norm=0.015,
            grad_max=3.2, grad_min=0.001, grad_mean=0.5,
            clipped=False, num_zero_grads=2, num_params=10,
        )
        d = diag.to_dict()
        _report("to_dict keys", d['step'] == 0 and d['grad_norm'] == 1.5)
        _report("is_healthy normal", diag.is_healthy is True)

        bad = GradientDiagnostics(
            step=1, grad_norm=5000.0, update_norm=50.0,
            grad_max=10000.0, grad_min=0.0, clipped=True,
            num_zero_grads=0, num_params=5,
        )
        _report("is_healthy exploding", bad.is_healthy is False)

        zero_heavy = GradientDiagnostics(
            step=2, grad_norm=0.5, update_norm=0.005,
            grad_max=1.0, grad_min=0.01, clipped=False,
            num_zero_grads=8, num_params=10,
        )
        _report("is_healthy zero grads", zero_heavy.is_healthy is False)
    except Exception as e:
        _report("GradientDiagnostics", False, str(e))

    # ------------------------------------------------------------------
    # Test 5: GradientTracker records steps
    # ------------------------------------------------------------------
    print("\n--- Test 5: GradientTracker records steps ---")
    try:
        tracker = GradientTracker()
        g1 = torch.randn(10, 5)
        g2 = torch.randn(3, 3)
        d = tracker.record_step(0, (g1, g2), lr=0.01)
        _report("record returns GradientDiagnostics",
                isinstance(d, GradientDiagnostics) and d.step == 0)
        _report("len after 1 record", len(tracker) == 1)

        tracker.record_step(1, (g1, None, g2), lr=0.01)
        _report("handles None grads", len(tracker) == 2)

        d3 = tracker.record_step(2, (None, None), lr=0.01)
        _report("all-None grads", d3.grad_norm == 0.0 and d3.num_params == 0)
    except Exception as e:
        _report("GradientTracker records", False, str(e))

    # ------------------------------------------------------------------
    # Test 6: GradientTracker summary
    # ------------------------------------------------------------------
    print("\n--- Test 6: GradientTracker summary ---")
    try:
        tracker = GradientTracker()
        g_small = torch.ones(10) * 0.1
        g_large = torch.ones(10) * 1.0
        g_zero = torch.zeros(5)

        tracker.record_step(0, (g_small,), lr=0.01, clipped=False)
        tracker.record_step(1, (g_large,), lr=0.01, clipped=True)
        tracker.record_step(2, (g_small, g_zero), lr=0.01, clipped=False)

        summary = tracker.get_summary()
        _report("summary has expected keys",
                'mean_grad_norm' in summary and 'max_grad_norm' in summary)
        _report("num_clip_events", summary['num_clip_events'] == 1,
                f"expected 1, got {summary['num_clip_events']}")
        _report("total_zero_grads >= 1", summary['total_zero_grads'] >= 1)
        _report("num_steps_recorded", summary['num_steps_recorded'] == 3)
        _report("norm_trajectory length", len(tracker.get_norm_trajectory()) == 3)
    except Exception as e:
        _report("GradientTracker summary", False, str(e))

    # ------------------------------------------------------------------
    # Test 7: EpochMetrics aggregation
    # ------------------------------------------------------------------
    print("\n--- Test 7: EpochMetrics construction ---")
    try:
        em = EpochMetrics(
            epoch=5, train_loss=0.42,
            pre_adapt_acc=0.3, post_adapt_acc=0.85,
            fast_gain=0.2, total_gain=0.55, auac=0.65,
            grad_summary={'mean_grad_norm': 1.2},
            wall_time=120.5, num_episodes=100, val_accuracy=0.82,
        )
        d = em.to_dict()
        _report("to_dict", d['epoch'] == 5 and d['train_loss'] == 0.42)
        _report("improvement_ratio", abs(em.improvement_ratio - 0.85 / 0.3) < 1e-6)
        _report("episodes_per_second", abs(em.episodes_per_second - 100 / 120.5) < 1e-4)

        em2 = EpochMetrics.from_dict(d)
        _report("from_dict round-trip",
                em2.epoch == em.epoch and abs(em2.train_loss - em.train_loss) < 1e-9)
    except Exception as e:
        _report("EpochMetrics", False, str(e))

    # ------------------------------------------------------------------
    # Test 8: MetricsAggregator start/record/finalize
    # ------------------------------------------------------------------
    print("\n--- Test 8: MetricsAggregator cycle ---")
    try:
        agg = MetricsAggregator()
        agg.start_epoch()

        for i in range(5):
            base_acc = 0.1 * i
            curve = AdaptationCurve(
                [base_acc, base_acc + 0.1, base_acc + 0.2],
                [1.0 - base_acc, 0.8 - base_acc, 0.6 - base_acc],
            )
            tracker = GradientTracker()
            tracker.record_step(0, (torch.randn(10, 10),), lr=0.01)
            agg.record_episode(curve, 0.5 - 0.05 * i, tracker)

        em = agg.finalize_epoch(epoch=0)
        _report("num_episodes", em.num_episodes == 5)
        _report("train_loss mean", abs(em.train_loss - 0.4) < 1e-6,
                f"expected 0.4, got {em.train_loss}")
        _report("wall_time >= 0", em.wall_time >= 0)
        _report("has grad_summary", len(em.grad_summary) > 0)
        _report("fast_gain", em.fast_gain == 0.1,
                f"expected 0.1, got {em.fast_gain}")
    except Exception as e:
        _report("MetricsAggregator cycle", False, str(e))

    # ------------------------------------------------------------------
    # Test 9: MetaCheckpoint save
    # ------------------------------------------------------------------
    print("\n--- Test 9: MetaCheckpoint save ---")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, 'test_ckpt.pt')
            model = nn.Linear(10, 5)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            history = [
                EpochMetrics(epoch=0, train_loss=0.5, num_episodes=10),
                EpochMetrics(epoch=1, train_loss=0.3, num_episodes=10),
            ]

            MetaCheckpoint.save(
                path=ckpt_path, model=model, optimizer=optimizer,
                config=_MockConfig(), epoch=1, metrics_history=history,
            )

            _report("file created", os.path.exists(ckpt_path))
            raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            _report("format_version", raw['format_version'] == '1.0')
            _report("has model_state_dict", 'model_state_dict' in raw)
            _report("epoch correct", raw['epoch'] == 1)
            _report("metrics_history len", len(raw['metrics_history']) == 2)
    except Exception as e:
        _report("MetaCheckpoint save", False, str(e))

    # ------------------------------------------------------------------
    # Test 10: MetaCheckpoint load restores model
    # ------------------------------------------------------------------
    print("\n--- Test 10: MetaCheckpoint load ---")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, 'test_ckpt.pt')
            model = nn.Linear(10, 5)
            optimizer = torch.optim.Adam(model.parameters())
            with torch.no_grad():
                model.weight.fill_(0.42)
                model.bias.fill_(-0.13)

            MetaCheckpoint.save(
                path=ckpt_path, model=model, optimizer=optimizer,
                config=_MockConfig(), epoch=5,
                metrics_history=[EpochMetrics(epoch=5, train_loss=0.1)],
            )

            model2 = nn.Linear(10, 5)
            nn.init.zeros_(model2.weight)
            nn.init.zeros_(model2.bias)
            ckpt = MetaCheckpoint.load(ckpt_path, model=model2)

            _report("restores weights",
                    torch.allclose(model2.weight, torch.full_like(model2.weight, 0.42)))
            _report("restores bias",
                    torch.allclose(model2.bias, torch.full_like(model2.bias, -0.13)))
            _report("returns epoch", ckpt['epoch'] == 5)
    except Exception as e:
        _report("MetaCheckpoint load", False, str(e))

    # ------------------------------------------------------------------
    # Test 11: MetaCheckpoint round-trip
    # ------------------------------------------------------------------
    print("\n--- Test 11: MetaCheckpoint round-trip ---")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, 'roundtrip.pt')
            torch.manual_seed(12345)
            model = nn.Sequential(nn.Linear(20, 15), nn.ReLU(), nn.Linear(15, 5))
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

            x = torch.randn(4, 20)
            model(x).sum().backward()
            optimizer.step()

            original_params = {n: p.clone() for n, p in model.named_parameters()}

            class _ReptileConfig:
                class maml:
                    algo = 'reptile'
                    second_order = False
                    inner_steps = 3
                    inner_lr = 0.05
                    inner_clip = 5.0
                    backend = 'custom'

            metrics = [EpochMetrics(epoch=i, train_loss=1.0 - 0.3 * i) for i in range(3)]

            MetaCheckpoint.save(
                path=ckpt_path, model=model, optimizer=optimizer,
                config=_ReptileConfig(), epoch=2, metrics_history=metrics,
                best_val_accuracy=0.88,
            )

            torch.manual_seed(99999)
            model2 = nn.Sequential(nn.Linear(20, 15), nn.ReLU(), nn.Linear(15, 5))
            optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.1, momentum=0.9)
            ckpt = MetaCheckpoint.load(ckpt_path, model=model2, optimizer=optimizer2)

            all_match = all(
                torch.allclose(p, original_params[n])
                for n, p in model2.named_parameters()
            )
            _report("parameters match", all_match)
            _report("best_val_accuracy", ckpt['best_val_accuracy'] == 0.88)
            _report("algo preserved", ckpt['algo'] == 'reptile')
            _report("metrics history len", len(ckpt['metrics_history']) == 3)
    except Exception as e:
        _report("MetaCheckpoint round-trip", False, str(e))

    # ------------------------------------------------------------------
    # Test 12: MetaCheckpoint version validation
    # ------------------------------------------------------------------
    print("\n--- Test 12: MetaCheckpoint version validation ---")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Incompatible major version
            bad_path = os.path.join(tmpdir, 'bad_v.pt')
            torch.save({'format_version': '2.0', 'model_state_dict': {}}, bad_path)
            model = nn.Linear(5, 3)
            try:
                MetaCheckpoint.load(bad_path, model=model)
                _report("rejects incompatible version", False, "should raise ValueError")
            except ValueError:
                _report("rejects incompatible version", True)

            # Compatible minor version
            ok_path = os.path.join(tmpdir, 'ok_v.pt')
            torch.save({'format_version': '1.1', 'model_state_dict': model.state_dict()}, ok_path)
            try:
                model2 = nn.Linear(5, 3)
                MetaCheckpoint.load(ok_path, model=model2)
                _report("accepts compatible minor version", True)
            except ValueError:
                _report("accepts compatible minor version", False)

            # Missing file
            try:
                MetaCheckpoint.load('/nonexistent/path.pt', model=model)
                _report("raises FileNotFoundError", False)
            except FileNotFoundError:
                _report("raises FileNotFoundError", True)
    except Exception as e:
        _report("MetaCheckpoint version", False, str(e))

    # ------------------------------------------------------------------
    # Test 13: MetaCheckpoint diff_config
    # ------------------------------------------------------------------
    print("\n--- Test 13: MetaCheckpoint diff_config ---")
    try:
        ckpt_dict = {
            'algo': 'maml', 'inner_steps': 5, 'inner_lr': 0.01,
            'second_order': True, 'backend': 'auto', 'inner_clip': 10.0,
        }

        # Same config
        diffs_same = MetaCheckpoint.diff_config(ckpt_dict, _MockConfig())
        _report("no changes detected", len(diffs_same) == 0)

        # Different config
        class _DiffMAML:
            algo = 'reptile'
            inner_steps = 10
            inner_lr = 0.05
            second_order = False
            backend = 'auto'
            inner_clip = 10.0

        class _DiffConfig:
            maml = _DiffMAML()

        diffs = MetaCheckpoint.diff_config(ckpt_dict, _DiffConfig())
        _report("detects algo change", 'algo' in diffs and diffs['algo'] == ('maml', 'reptile'))
        _report("detects inner_steps change", 'inner_steps' in diffs and diffs['inner_steps'] == (5, 10))
        _report("detects inner_lr change", 'inner_lr' in diffs and diffs['inner_lr'] == (0.01, 0.05))
        _report("detects second_order change",
                'second_order' in diffs and diffs['second_order'] == (True, False))
        _report("total diffs count", len(diffs) == 4,
                f"expected 4, got {len(diffs)}")
    except Exception as e:
        _report("diff_config", False, str(e))

    # ------------------------------------------------------------------
    # Test 14: MetricsLogger writes valid JSON
    # ------------------------------------------------------------------
    print("\n--- Test 14: MetricsLogger JSON ---")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, 'metrics.json')
            logger = MetricsLogger(log_path)

            for ep in range(3):
                logger.log_epoch(EpochMetrics(
                    epoch=ep, train_loss=1.0 - 0.2 * ep,
                    pre_adapt_acc=0.2 + 0.1 * ep, post_adapt_acc=0.6 + 0.1 * ep,
                    auac=0.5 + 0.05 * ep, num_episodes=20, wall_time=10.0 + ep,
                ))

            _report("num_entries", logger.num_entries == 3)

            loaded = logger.load()
            _report("load returns correct list",
                    isinstance(loaded, list) and len(loaded) == 3)

            with open(log_path) as f:
                parsed = json.loads(f.read())
            _report("valid JSON", len(parsed) == 3 and parsed[0]['epoch'] == 0)

            epoch_list = logger.load_as_epoch_metrics()
            _report("load_as_epoch_metrics",
                    len(epoch_list) == 3 and isinstance(epoch_list[0], EpochMetrics))

            best = logger.get_best_epoch('post_adapt_acc', maximize=True)
            _report("get_best_epoch", best is not None and best['epoch'] == 2)
    except Exception as e:
        _report("MetricsLogger JSON", False, str(e))

    # ------------------------------------------------------------------
    # Test 15: AdaptationCurve.to_dict() JSON-serialisable
    # ------------------------------------------------------------------
    print("\n--- Test 15: AdaptationCurve JSON serialisability ---")
    try:
        curve = AdaptationCurve([0.1, 0.3, 0.5, 0.7], [2.5, 1.8, 1.2, 0.7])
        d = curve.to_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)

        _report("JSON-serialisable", isinstance(parsed, dict) and 'auac' in parsed)
        _report("values round-trip",
                parsed['pre_adapt_acc'] == 0.1 and parsed['post_adapt_acc'] == 0.7)
        _report("has adaptation_efficiency", 'adaptation_efficiency' in parsed)

        curve2 = AdaptationCurve.from_dict(parsed)
        _report("from_dict after JSON",
                curve2.step_accuracies == curve.step_accuracies)
    except Exception as e:
        _report("AdaptationCurve JSON", False, str(e))

    # ------------------------------------------------------------------
    # Test 16: Empty metrics don't crash aggregator
    # ------------------------------------------------------------------
    print("\n--- Test 16: Empty metrics robustness ---")
    try:
        # Empty aggregator
        agg = MetricsAggregator()
        agg.start_epoch()
        em = agg.finalize_epoch(epoch=0)
        _report("empty aggregator", em.num_episodes == 0 and em.train_loss == 0.0)

        # Empty adaptation curve
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            empty_curve = AdaptationCurve([], [])
        _report("empty curve pre_adapt_acc", empty_curve.pre_adapt_acc == 0.0)
        _report("empty curve auac", empty_curve.auac == 0.0)
        _report("empty curve fast_gain", empty_curve.fast_adaptation_gain == 0.0)

        # Empty gradient tracker
        _report("empty tracker summary", GradientTracker().get_summary() == {})

        # None fields serialise
        em_none = EpochMetrics(epoch=0, train_loss=0.0)
        json.dumps(em_none.to_dict())
        _report("None fields serialise", True)

        # Auto-start on record_episode
        agg2 = MetricsAggregator()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            agg2.record_episode(AdaptationCurve([0.5], [1.0]), 0.5)
        _report("auto-start", agg2.finalize_epoch(0).num_episodes == 1)
    except Exception as e:
        _report("Empty metrics robustness", False, str(e))

    # ------------------------------------------------------------------
    # Bonus: LSLRStatistics
    # ------------------------------------------------------------------
    print("\n--- Bonus: LSLRStatistics ---")
    try:
        class _MockLSLR:
            layer_names = ['layer0', 'layer1', 'layer2']
            num_steps = 3
            _lrs = {
                'layer0': {0: 0.01, 1: 0.02, 2: 0.015},
                'layer1': {0: 0.005, 1: 0.008, 2: 0.012},
                'layer2': {0: 0.02, 1: 0.025, 2: 0.03},
            }
            def get_lr(self, layer, step):
                return torch.tensor(self._lrs[layer][step])

        stats = compute_lslr_statistics(_MockLSLR())
        _report("num_layers", stats.num_layers == 3)
        _report("lr_min", abs(stats.lr_min - 0.005) < 1e-6)
        _report("lr_max", abs(stats.lr_max - 0.03) < 1e-6)
        _report("per_layer_means count", len(stats.per_layer_means) == 3)
        _report("per_step_means count", len(stats.per_step_means) == 3)
        json.dumps(stats.to_dict())
        _report("JSON-serialisable", True)
        _report("has_negative_lrs", stats.has_negative_lrs is False)
    except Exception as e:
        _report("LSLRStatistics", False, str(e))

    # ------------------------------------------------------------------
    # Bonus: MetaCheckpoint MAML++ extensions
    # ------------------------------------------------------------------
    print("\n--- Bonus: MetaCheckpoint MAML++ extensions ---")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, 'maml_plus.pt')
            model = nn.Linear(10, 5)
            optimizer = torch.optim.Adam(model.parameters())
            lslr = nn.Linear(3, 3)
            msl = nn.Linear(5, 5)
            with torch.no_grad():
                lslr.weight.fill_(0.01)
                msl.weight.fill_(0.5)

            MetaCheckpoint.save(
                path=ckpt_path, model=model, optimizer=optimizer,
                config=_MockConfig(), epoch=3,
                metrics_history=[EpochMetrics(epoch=3, train_loss=0.2)],
                lslr_module=lslr, msl_module=msl,
            )

            model2 = nn.Linear(10, 5)
            lslr2 = nn.Linear(3, 3)
            msl2 = nn.Linear(5, 5)
            ckpt = MetaCheckpoint.load(ckpt_path, model=model2,
                                       lslr_module=lslr2, msl_module=msl2)

            _report("LSLR restored",
                    torch.allclose(lslr2.weight, torch.full_like(lslr2.weight, 0.01)))
            _report("MSL restored",
                    torch.allclose(msl2.weight, torch.full_like(msl2.weight, 0.5)))

            info = MetaCheckpoint.get_info(ckpt_path)
            _report("get_info", info['epoch'] == 3 and info['has_lslr'] is True)
    except Exception as e:
        _report("MAML++ extensions", False, str(e))

    # ------------------------------------------------------------------
    # Bonus: MetaCheckpoint get_metrics_history
    # ------------------------------------------------------------------
    print("\n--- Bonus: MetaCheckpoint get_metrics_history ---")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, 'hist.pt')
            model = nn.Linear(5, 3)
            optimizer = torch.optim.Adam(model.parameters())
            metrics = [
                EpochMetrics(epoch=0, train_loss=1.0, post_adapt_acc=0.5),
                EpochMetrics(epoch=1, train_loss=0.7, post_adapt_acc=0.65),
            ]
            MetaCheckpoint.save(
                path=ckpt_path, model=model, optimizer=optimizer,
                config=_MockConfig(), epoch=1, metrics_history=metrics,
            )
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            restored = MetaCheckpoint.get_metrics_history(ckpt)
            _report("restores EpochMetrics",
                    len(restored) == 2 and isinstance(restored[0], EpochMetrics))
            _report("preserves values", abs(restored[1].post_adapt_acc - 0.65) < 1e-6)
    except Exception as e:
        _report("get_metrics_history", False, str(e))

    # ------------------------------------------------------------------
    # Bonus: summarize_epoch
    # ------------------------------------------------------------------
    print("\n--- Bonus: summarize_epoch ---")
    try:
        em = EpochMetrics(
            epoch=10, train_loss=0.25, pre_adapt_acc=0.4,
            post_adapt_acc=0.9, total_gain=0.5, auac=0.72,
            num_episodes=100, wall_time=60.0, val_accuracy=0.88,
            s2_fraction=0.1, grad_summary={'mean_grad_norm': 1.5},
        )
        s = summarize_epoch(em)
        _report("returns string with Epoch", isinstance(s, str) and 'Epoch' in s)
    except Exception as e:
        _report("summarize_epoch", False, str(e))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    total = _pass_count + _fail_count
    print(f"Results: {_pass_count}/{total} passed, {_fail_count} failed")
    print("=" * 72)

    sys.exit(0 if _fail_count == 0 else 1)
