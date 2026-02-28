"""
Plasticity Diagnostics Template
================================

Comprehensive diagnostics, logging, and trace inspection for the
neuromodulation + eligibility traces (three-factor learning) system.

This module provides:
- Structured data classes for capturing plasticity snapshots at each step
- A PlasticityLogger for recording and querying diagnostics over time
- A ModulatorAnalyzer for statistical analysis of neuromodulator signals
- A TraceAnalyzer for detecting trace pathologies (explosion, vanishing)
- A PlasticityReport generator for human-readable diagnostic output
- Checkpoint integration for saving/loading diagnostics alongside model state

All code is self-contained with no external dependencies beyond torch,
json, pathlib, and Python stdlib.

Target file: brain_ai/meta/plasticity_diagnostics.py

Usage::

    from brain_ai.meta.plasticity_diagnostics import (
        PlasticityLogger,
        ModulatorAnalyzer,
        TraceAnalyzer,
        PlasticityReport,
        save_diagnostics_with_checkpoint,
        load_diagnostics_from_checkpoint,
    )

    logger = PlasticityLogger(log_every_n_steps=1, max_history=10000)

    # Inside training loop:
    logger.log_step(
        step=step_idx,
        modulators=ModulatorSnapshot(da=0.5, ach=0.3, ne=0.2, sht=0.6),
        traces=[TraceSnapshot(layer_name="snn.w1", norm=1.2, mean=0.01,
                              max_val=0.8, min_val=-0.3, sparsity=0.15)],
        updates=[UpdateSnapshot(layer_name="snn.w1", delta_w_norm=0.003,
                                clamp_hits=0, effective_lr=0.001)],
    )

    # After epoch:
    summary = logger.get_epoch_summary(range(0, 1000))
    report = PlasticityReport.generate_report(logger.get_trace())
    print(report)
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Data Classes
# ---------------------------------------------------------------------------


@dataclass
class ModulatorSnapshot:
    """Snapshot of all four neuromodulator values at a single timestep.

    Attributes:
        da:  Dopamine level (reward prediction error proxy).  Range typically [-1, 1].
        ach: Acetylcholine level (novelty / attention gate).  Range typically [0, 1].
        ne:  Norepinephrine level (urgency / arousal).        Range typically [0, 1].
        sht: Serotonin (5-HT) level (patience / exploitation). Range typically [0, 1].
    """

    da: float = 0.0
    ach: float = 0.0
    ne: float = 0.0
    sht: float = 0.0

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, float]:
        """Serialize to a plain dictionary."""
        return {"da": self.da, "ach": self.ach, "ne": self.ne, "sht": self.sht}

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "ModulatorSnapshot":
        """Deserialize from a plain dictionary."""
        return cls(
            da=float(d.get("da", 0.0)),
            ach=float(d.get("ach", 0.0)),
            ne=float(d.get("ne", 0.0)),
            sht=float(d.get("sht", 0.0)),
        )

    def as_list(self) -> List[float]:
        """Return modulator values as an ordered list [DA, ACh, NE, 5-HT]."""
        return [self.da, self.ach, self.ne, self.sht]

    def magnitude(self) -> float:
        """L2 norm of the modulator vector."""
        return math.sqrt(self.da ** 2 + self.ach ** 2 + self.ne ** 2 + self.sht ** 2)

    def is_saturated(self, threshold: float = 0.95) -> Dict[str, bool]:
        """Check whether each modulator is near its boundary.

        For DA (range [-1,1]) saturation means |da| > threshold.
        For ACh, NE, 5-HT (range [0,1]) saturation means value > threshold.
        """
        return {
            "da": abs(self.da) > threshold,
            "ach": self.ach > threshold,
            "ne": self.ne > threshold,
            "sht": self.sht > threshold,
        }


@dataclass
class TraceSnapshot:
    """Snapshot of eligibility trace statistics for one layer at one timestep.

    Attributes:
        layer_name: Fully-qualified layer name (e.g. ``"snn_core.layer1.weight"``).
        norm:       Frobenius / L2 norm of the trace tensor.
        mean:       Mean value of the trace tensor.
        max_val:    Maximum element in the trace tensor.
        min_val:    Minimum element in the trace tensor.
        sparsity:   Fraction of elements with |value| < 1e-6 (near-zero).
    """

    layer_name: str = ""
    norm: float = 0.0
    mean: float = 0.0
    max_val: float = 0.0
    min_val: float = 0.0
    sparsity: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "layer_name": self.layer_name,
            "norm": self.norm,
            "mean": self.mean,
            "max_val": self.max_val,
            "min_val": self.min_val,
            "sparsity": self.sparsity,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TraceSnapshot":
        """Deserialize from a plain dictionary."""
        return cls(
            layer_name=str(d.get("layer_name", "")),
            norm=float(d.get("norm", 0.0)),
            mean=float(d.get("mean", 0.0)),
            max_val=float(d.get("max_val", 0.0)),
            min_val=float(d.get("min_val", 0.0)),
            sparsity=float(d.get("sparsity", 0.0)),
        )

    @classmethod
    def from_tensor(cls, layer_name: str, tensor: torch.Tensor) -> "TraceSnapshot":
        """Create snapshot from a live eligibility trace tensor.

        This is the primary way to build a ``TraceSnapshot`` during training.

        Args:
            layer_name: Name of the layer / parameter.
            tensor:     The eligibility trace tensor (any shape).

        Returns:
            A populated ``TraceSnapshot``.
        """
        with torch.no_grad():
            flat = tensor.float().flatten()
            norm_val = torch.norm(flat).item()
            mean_val = flat.mean().item()
            max_val = flat.max().item()
            min_val = flat.min().item()
            near_zero = (flat.abs() < 1e-6).float().mean().item()
        return cls(
            layer_name=layer_name,
            norm=norm_val,
            mean=mean_val,
            max_val=max_val,
            min_val=min_val,
            sparsity=near_zero,
        )


@dataclass
class UpdateSnapshot:
    """Snapshot of a single weight-update event for one layer.

    Attributes:
        layer_name:    Fully-qualified layer name.
        delta_w_norm:  L2 norm of the weight delta applied.
        clamp_hits:    Number of elements that hit the clamp boundary.
        effective_lr:  The effective learning rate used (plasticity_gain * base_lr).
    """

    layer_name: str = ""
    delta_w_norm: float = 0.0
    clamp_hits: int = 0
    effective_lr: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "layer_name": self.layer_name,
            "delta_w_norm": self.delta_w_norm,
            "clamp_hits": self.clamp_hits,
            "effective_lr": self.effective_lr,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "UpdateSnapshot":
        """Deserialize from a plain dictionary."""
        return cls(
            layer_name=str(d.get("layer_name", "")),
            delta_w_norm=float(d.get("delta_w_norm", 0.0)),
            clamp_hits=int(d.get("clamp_hits", 0)),
            effective_lr=float(d.get("effective_lr", 0.0)),
        )

    @classmethod
    def from_delta_tensor(
        cls,
        layer_name: str,
        delta_w: torch.Tensor,
        clamp_lo: float = -1.0,
        clamp_hi: float = 1.0,
        effective_lr: float = 0.0,
    ) -> "UpdateSnapshot":
        """Create snapshot from a weight-delta tensor.

        Args:
            layer_name:    Name of the layer.
            delta_w:       The weight delta tensor.
            clamp_lo:      Lower clamp bound (used to count clamp hits).
            clamp_hi:      Upper clamp bound.
            effective_lr:  Effective learning rate applied.

        Returns:
            A populated ``UpdateSnapshot``.
        """
        with torch.no_grad():
            flat = delta_w.float().flatten()
            norm_val = torch.norm(flat).item()
            hits = int(((flat <= clamp_lo) | (flat >= clamp_hi)).sum().item())
        return cls(
            layer_name=layer_name,
            delta_w_norm=norm_val,
            clamp_hits=hits,
            effective_lr=effective_lr,
        )


@dataclass
class StepDiagnostics:
    """Complete diagnostics for a single training step.

    Bundles modulator state, eligibility-trace statistics for every
    tracked layer, and weight-update statistics for that step.

    Attributes:
        step_idx:    Global step index.
        modulators:  Neuromodulator snapshot.
        traces:      Per-layer trace snapshots.
        updates:     Per-layer update snapshots.
        timestamp:   Wall-clock time (seconds since epoch) when the step was recorded.
    """

    step_idx: int = 0
    modulators: ModulatorSnapshot = field(default_factory=ModulatorSnapshot)
    traces: List[TraceSnapshot] = field(default_factory=list)
    updates: List[UpdateSnapshot] = field(default_factory=list)
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "step_idx": self.step_idx,
            "modulators": self.modulators.to_dict(),
            "traces": [t.to_dict() for t in self.traces],
            "updates": [u.to_dict() for u in self.updates],
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StepDiagnostics":
        """Deserialize from a plain dictionary."""
        return cls(
            step_idx=int(d.get("step_idx", 0)),
            modulators=ModulatorSnapshot.from_dict(d.get("modulators", {})),
            traces=[TraceSnapshot.from_dict(t) for t in d.get("traces", [])],
            updates=[UpdateSnapshot.from_dict(u) for u in d.get("updates", [])],
            timestamp=float(d.get("timestamp", 0.0)),
        )

    def has_nonzero_update(self) -> bool:
        """Return True if any layer received a non-zero weight update."""
        return any(u.delta_w_norm > 0.0 for u in self.updates)


@dataclass
class PlasticityTrace:
    """Container for a sequence of step diagnostics with summary statistics.

    Attributes:
        steps:       Ordered list of per-step diagnostics.
        total_steps: Total number of steps captured (may differ from len(steps)
                     if some steps were skipped due to ``log_every_n_steps``).
        summary:     Arbitrary summary dictionary (populated lazily).
    """

    steps: List[StepDiagnostics] = field(default_factory=list)
    total_steps: int = 0
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full trace to a JSON-compatible dictionary."""
        return {
            "steps": [s.to_dict() for s in self.steps],
            "total_steps": self.total_steps,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PlasticityTrace":
        """Deserialize from a JSON-compatible dictionary."""
        return cls(
            steps=[StepDiagnostics.from_dict(s) for s in d.get("steps", [])],
            total_steps=int(d.get("total_steps", 0)),
            summary=dict(d.get("summary", {})),
        )

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "PlasticityTrace":
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(s))

    def step_range(self) -> Tuple[int, int]:
        """Return (min_step, max_step) covered by this trace, or (0, 0) if empty."""
        if not self.steps:
            return (0, 0)
        return (self.steps[0].step_idx, self.steps[-1].step_idx)

    def layer_names(self) -> List[str]:
        """Return sorted list of unique layer names across all trace snapshots."""
        names: set = set()
        for s in self.steps:
            for t in s.traces:
                names.add(t.layer_name)
        return sorted(names)


@dataclass
class EpochSummary:
    """Aggregated diagnostics for a single epoch.

    Attributes:
        epoch:                     Epoch number.
        mean_da:                   Mean dopamine across the epoch.
        mean_ach:                  Mean acetylcholine across the epoch.
        mean_ne:                   Mean norepinephrine across the epoch.
        mean_sht:                  Mean serotonin across the epoch.
        mean_trace_norm_per_layer: {layer_name: mean_norm} across the epoch.
        mean_update_norm_per_layer:{layer_name: mean_delta_w_norm} across the epoch.
        total_clamp_hits:          Sum of clamp hits across all layers and steps.
        plasticity_utilization:    Fraction of steps with at least one non-zero update.
    """

    epoch: int = 0
    mean_da: float = 0.0
    mean_ach: float = 0.0
    mean_ne: float = 0.0
    mean_sht: float = 0.0
    mean_trace_norm_per_layer: Dict[str, float] = field(default_factory=dict)
    mean_update_norm_per_layer: Dict[str, float] = field(default_factory=dict)
    total_clamp_hits: int = 0
    plasticity_utilization: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "epoch": self.epoch,
            "mean_da": self.mean_da,
            "mean_ach": self.mean_ach,
            "mean_ne": self.mean_ne,
            "mean_sht": self.mean_sht,
            "mean_trace_norm_per_layer": dict(self.mean_trace_norm_per_layer),
            "mean_update_norm_per_layer": dict(self.mean_update_norm_per_layer),
            "total_clamp_hits": self.total_clamp_hits,
            "plasticity_utilization": self.plasticity_utilization,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EpochSummary":
        """Deserialize from a plain dictionary."""
        return cls(
            epoch=int(d.get("epoch", 0)),
            mean_da=float(d.get("mean_da", 0.0)),
            mean_ach=float(d.get("mean_ach", 0.0)),
            mean_ne=float(d.get("mean_ne", 0.0)),
            mean_sht=float(d.get("mean_sht", 0.0)),
            mean_trace_norm_per_layer=dict(d.get("mean_trace_norm_per_layer", {})),
            mean_update_norm_per_layer=dict(d.get("mean_update_norm_per_layer", {})),
            total_clamp_hits=int(d.get("total_clamp_hits", 0)),
            plasticity_utilization=float(d.get("plasticity_utilization", 0.0)),
        )


# ---------------------------------------------------------------------------
# 2. PlasticityLogger
# ---------------------------------------------------------------------------


class PlasticityLogger:
    """Main logging class for tracking plasticity over training.

    Records ``StepDiagnostics`` every ``log_every_n_steps`` steps and
    provides retrieval, aggregation, and persistence utilities.

    Args:
        log_every_n_steps: Record one step every *n* steps (default 1 = every step).
        max_history:       Maximum number of step records to retain in memory.
                           Oldest records are evicted when this limit is reached.
        detailed:          If False, skip trace/update snapshots (modulator-only logging).
                           Useful for reducing overhead in production runs.
    """

    def __init__(
        self,
        log_every_n_steps: int = 1,
        max_history: int = 10_000,
        detailed: bool = False,
    ) -> None:
        self._log_every: int = max(1, log_every_n_steps)
        self._max_history: int = max_history
        self._detailed: bool = detailed

        self._history: List[StepDiagnostics] = []
        self._total_steps_seen: int = 0

        _logger.debug(
            "PlasticityLogger initialized: log_every=%d, max_history=%d, detailed=%s",
            self._log_every,
            self._max_history,
            self._detailed,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_recorded(self) -> int:
        """Number of step records currently stored."""
        return len(self._history)

    @property
    def total_steps_seen(self) -> int:
        """Total number of steps observed (including skipped ones)."""
        return self._total_steps_seen

    @property
    def detailed(self) -> bool:
        """Whether detailed logging (traces/updates) is enabled."""
        return self._detailed

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def log_step(
        self,
        step: int,
        modulators: ModulatorSnapshot,
        traces: Optional[List[TraceSnapshot]] = None,
        updates: Optional[List[UpdateSnapshot]] = None,
    ) -> None:
        """Record one step's diagnostics.

        Args:
            step:       Global step index.
            modulators: Current neuromodulator snapshot.
            traces:     Per-layer eligibility trace snapshots (optional).
            updates:    Per-layer weight-update snapshots (optional).
        """
        self._total_steps_seen += 1

        # Respect sampling interval
        if self._total_steps_seen % self._log_every != 0:
            return

        # If not detailed, strip trace/update data to save memory
        if not self._detailed:
            traces = []
            updates = []

        diag = StepDiagnostics(
            step_idx=step,
            modulators=modulators,
            traces=traces if traces is not None else [],
            updates=updates if updates is not None else [],
            timestamp=time.time(),
        )

        self._history.append(diag)

        # Evict oldest if over capacity
        if len(self._history) > self._max_history:
            overflow = len(self._history) - self._max_history
            self._history = self._history[overflow:]
            _logger.debug(
                "PlasticityLogger evicted %d oldest records (capacity=%d)",
                overflow,
                self._max_history,
            )

    def get_trace(
        self,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
    ) -> PlasticityTrace:
        """Retrieve a ``PlasticityTrace`` for a step range.

        Args:
            start_step: Inclusive lower bound on ``step_idx`` (None = beginning).
            end_step:   Inclusive upper bound on ``step_idx`` (None = end).

        Returns:
            A ``PlasticityTrace`` containing the matching steps.
        """
        filtered: List[StepDiagnostics] = []
        for diag in self._history:
            if start_step is not None and diag.step_idx < start_step:
                continue
            if end_step is not None and diag.step_idx > end_step:
                continue
            filtered.append(diag)

        trace = PlasticityTrace(
            steps=filtered,
            total_steps=len(filtered),
        )

        # Populate a lightweight summary
        if filtered:
            da_vals = [s.modulators.da for s in filtered]
            ach_vals = [s.modulators.ach for s in filtered]
            ne_vals = [s.modulators.ne for s in filtered]
            sht_vals = [s.modulators.sht for s in filtered]
            trace.summary = {
                "num_steps": len(filtered),
                "mean_da": _safe_mean(da_vals),
                "mean_ach": _safe_mean(ach_vals),
                "mean_ne": _safe_mean(ne_vals),
                "mean_sht": _safe_mean(sht_vals),
                "step_range": [filtered[0].step_idx, filtered[-1].step_idx],
            }

        return trace

    def get_epoch_summary(
        self,
        epoch_steps: range,
        epoch: int = 0,
    ) -> EpochSummary:
        """Aggregate statistics for a contiguous range of steps (an epoch).

        Args:
            epoch_steps: A ``range`` of step indices belonging to this epoch.
            epoch:       Epoch number for labeling.

        Returns:
            An ``EpochSummary`` with aggregated statistics.
        """
        step_set = set(epoch_steps)
        matching = [d for d in self._history if d.step_idx in step_set]

        if not matching:
            return EpochSummary(epoch=epoch)

        # Modulator means
        da_vals = [s.modulators.da for s in matching]
        ach_vals = [s.modulators.ach for s in matching]
        ne_vals = [s.modulators.ne for s in matching]
        sht_vals = [s.modulators.sht for s in matching]

        # Trace norms per layer
        trace_norms: Dict[str, List[float]] = {}
        for s in matching:
            for t in s.traces:
                trace_norms.setdefault(t.layer_name, []).append(t.norm)
        mean_trace_norm = {k: _safe_mean(v) for k, v in trace_norms.items()}

        # Update norms per layer
        update_norms: Dict[str, List[float]] = {}
        total_clamp = 0
        for s in matching:
            for u in s.updates:
                update_norms.setdefault(u.layer_name, []).append(u.delta_w_norm)
                total_clamp += u.clamp_hits
        mean_update_norm = {k: _safe_mean(v) for k, v in update_norms.items()}

        # Plasticity utilization: fraction of steps with non-zero updates
        nonzero_steps = sum(1 for s in matching if s.has_nonzero_update())
        utilization = nonzero_steps / len(matching) if matching else 0.0

        return EpochSummary(
            epoch=epoch,
            mean_da=_safe_mean(da_vals),
            mean_ach=_safe_mean(ach_vals),
            mean_ne=_safe_mean(ne_vals),
            mean_sht=_safe_mean(sht_vals),
            mean_trace_norm_per_layer=mean_trace_norm,
            mean_update_norm_per_layer=mean_update_norm,
            total_clamp_hits=total_clamp,
            plasticity_utilization=utilization,
        )

    def clear(self) -> None:
        """Reset all logged history."""
        self._history.clear()
        self._total_steps_seen = 0
        _logger.debug("PlasticityLogger cleared")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_json(self, path: Union[str, Path]) -> None:
        """Persist all diagnostics to a JSON file.

        Args:
            path: File path (will be created / overwritten).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "log_every_n_steps": self._log_every,
            "max_history": self._max_history,
            "detailed": self._detailed,
            "total_steps_seen": self._total_steps_seen,
            "history": [d.to_dict() for d in self._history],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        _logger.info("PlasticityLogger saved %d records to %s", len(self._history), path)

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> "PlasticityLogger":
        """Restore a ``PlasticityLogger`` from a JSON file.

        Args:
            path: Path to a file previously written by :meth:`save_json`.

        Returns:
            A fully restored ``PlasticityLogger`` instance.
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger_obj = cls(
            log_every_n_steps=int(data.get("log_every_n_steps", 1)),
            max_history=int(data.get("max_history", 10_000)),
            detailed=bool(data.get("detailed", False)),
        )
        logger_obj._total_steps_seen = int(data.get("total_steps_seen", 0))
        logger_obj._history = [
            StepDiagnostics.from_dict(d) for d in data.get("history", [])
        ]
        _logger.info(
            "PlasticityLogger loaded %d records from %s",
            len(logger_obj._history),
            path,
        )
        return logger_obj

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def last_step(self) -> Optional[StepDiagnostics]:
        """Return the most recently recorded step, or None if empty."""
        return self._history[-1] if self._history else None

    def modulator_history(self) -> List[ModulatorSnapshot]:
        """Return a flat list of all recorded modulator snapshots."""
        return [d.modulators for d in self._history]

    def __repr__(self) -> str:
        return (
            f"PlasticityLogger(recorded={self.num_recorded}, "
            f"total_seen={self._total_steps_seen}, "
            f"log_every={self._log_every})"
        )


# ---------------------------------------------------------------------------
# 3. ModulatorAnalyzer
# ---------------------------------------------------------------------------


class ModulatorAnalyzer:
    """Analysis utilities for neuromodulator signal histories.

    All methods are static / classmethod -- no internal state.
    """

    @staticmethod
    def compute_modulator_stats(
        history: List[ModulatorSnapshot],
    ) -> Dict[str, Dict[str, float]]:
        """Compute per-modulator summary statistics.

        For each of DA, ACh, NE, 5-HT computes: mean, std, min, max,
        and saturation_rate (fraction of steps where the modulator is
        near its output bounds).

        Args:
            history: Ordered list of modulator snapshots.

        Returns:
            ``{"da": {"mean": ..., "std": ..., "min": ..., "max": ...,
            "saturation_rate": ...}, ...}``
        """
        if not history:
            empty = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "saturation_rate": 0.0}
            return {"da": dict(empty), "ach": dict(empty), "ne": dict(empty), "sht": dict(empty)}

        da_vals = [h.da for h in history]
        ach_vals = [h.ach for h in history]
        ne_vals = [h.ne for h in history]
        sht_vals = [h.sht for h in history]

        def _stats(vals: List[float], name: str) -> Dict[str, float]:
            n = len(vals)
            mean = _safe_mean(vals)
            std = _safe_std(vals)
            lo = min(vals)
            hi = max(vals)
            # Saturation: DA in [-1,1] so |v| > 0.9; others in [0,1] so v > 0.9
            if name == "da":
                sat_count = sum(1 for v in vals if abs(v) > 0.9)
            else:
                sat_count = sum(1 for v in vals if v > 0.9)
            sat_rate = sat_count / n if n > 0 else 0.0
            return {
                "mean": mean,
                "std": std,
                "min": lo,
                "max": hi,
                "saturation_rate": sat_rate,
            }

        return {
            "da": _stats(da_vals, "da"),
            "ach": _stats(ach_vals, "ach"),
            "ne": _stats(ne_vals, "ne"),
            "sht": _stats(sht_vals, "sht"),
        }

    @staticmethod
    def compute_da_reward_correlation(
        da_history: List[float],
        reward_history: List[float],
    ) -> float:
        """Compute Pearson correlation between DA signal and actual reward.

        Args:
            da_history:     Ordered DA values over time.
            reward_history: Corresponding reward values.

        Returns:
            Pearson r in [-1, 1].  Returns 0.0 if inputs are degenerate
            (zero variance, mismatched lengths, etc.).
        """
        n = min(len(da_history), len(reward_history))
        if n < 2:
            return 0.0

        da = da_history[:n]
        rw = reward_history[:n]

        mean_da = _safe_mean(da)
        mean_rw = _safe_mean(rw)

        cov = 0.0
        var_da = 0.0
        var_rw = 0.0
        for i in range(n):
            d_da = da[i] - mean_da
            d_rw = rw[i] - mean_rw
            cov += d_da * d_rw
            var_da += d_da * d_da
            var_rw += d_rw * d_rw

        denom = math.sqrt(var_da * var_rw)
        if denom < 1e-12:
            return 0.0
        return cov / denom

    @staticmethod
    def detect_saturation(
        history: List[ModulatorSnapshot],
        threshold: float = 0.95,
    ) -> Dict[str, float]:
        """Compute the fraction of steps where each modulator is saturated.

        Saturation means the modulator is near its output bound.
        For DA (range [-1,1]): ``|da| > threshold``.
        For ACh/NE/5-HT (range [0,1]): ``value > threshold``.

        Args:
            history:   Ordered list of modulator snapshots.
            threshold: Saturation boundary (default 0.95).

        Returns:
            ``{"da": frac, "ach": frac, "ne": frac, "sht": frac}``
        """
        if not history:
            return {"da": 0.0, "ach": 0.0, "ne": 0.0, "sht": 0.0}

        n = len(history)
        da_sat = sum(1 for h in history if abs(h.da) > threshold) / n
        ach_sat = sum(1 for h in history if h.ach > threshold) / n
        ne_sat = sum(1 for h in history if h.ne > threshold) / n
        sht_sat = sum(1 for h in history if h.sht > threshold) / n

        return {"da": da_sat, "ach": ach_sat, "ne": ne_sat, "sht": sht_sat}

    @staticmethod
    def modulator_temporal_profile(
        history: List[ModulatorSnapshot],
        window: int = 100,
    ) -> Dict[str, List[float]]:
        """Compute a sliding-window mean for each modulator.

        Args:
            history: Ordered modulator snapshots.
            window:  Window size in steps.

        Returns:
            ``{"da": [mean_0, mean_1, ...], "ach": [...], ...}`` where each
            list has ``len(history) - window + 1`` entries (or empty if
            history is shorter than window).
        """
        result: Dict[str, List[float]] = {"da": [], "ach": [], "ne": [], "sht": []}

        n = len(history)
        if n < window or window < 1:
            return result

        # Compute prefix sums for efficiency
        da_prefix = _prefix_sum([h.da for h in history])
        ach_prefix = _prefix_sum([h.ach for h in history])
        ne_prefix = _prefix_sum([h.ne for h in history])
        sht_prefix = _prefix_sum([h.sht for h in history])

        for i in range(n - window + 1):
            result["da"].append((da_prefix[i + window] - da_prefix[i]) / window)
            result["ach"].append((ach_prefix[i + window] - ach_prefix[i]) / window)
            result["ne"].append((ne_prefix[i + window] - ne_prefix[i]) / window)
            result["sht"].append((sht_prefix[i + window] - sht_prefix[i]) / window)

        return result

    @staticmethod
    def modulator_range_check(
        history: List[ModulatorSnapshot],
    ) -> Dict[str, Dict[str, bool]]:
        """Check whether modulator values stay within expected ranges.

        DA should be in [-1, 1], ACh/NE/5-HT should be in [0, 1].

        Returns:
            ``{"da": {"in_range": True/False, "violations": count}, ...}``
        """
        results: Dict[str, Dict[str, Any]] = {}
        if not history:
            for name in ("da", "ach", "ne", "sht"):
                results[name] = {"in_range": True, "violations": 0}
            return results

        da_viols = sum(1 for h in history if h.da < -1.0 or h.da > 1.0)
        ach_viols = sum(1 for h in history if h.ach < 0.0 or h.ach > 1.0)
        ne_viols = sum(1 for h in history if h.ne < 0.0 or h.ne > 1.0)
        sht_viols = sum(1 for h in history if h.sht < 0.0 or h.sht > 1.0)

        results["da"] = {"in_range": da_viols == 0, "violations": da_viols}
        results["ach"] = {"in_range": ach_viols == 0, "violations": ach_viols}
        results["ne"] = {"in_range": ne_viols == 0, "violations": ne_viols}
        results["sht"] = {"in_range": sht_viols == 0, "violations": sht_viols}
        return results


# ---------------------------------------------------------------------------
# 4. TraceAnalyzer
# ---------------------------------------------------------------------------


class TraceAnalyzer:
    """Analysis utilities for eligibility trace snapshots.

    All methods are static -- no internal state.
    """

    @staticmethod
    def compute_trace_decay_curve(
        snapshots: List[List[TraceSnapshot]],
        layer_name: str,
    ) -> List[float]:
        """Extract the norm of a specific layer's trace over time.

        Args:
            snapshots: Outer list is timesteps; inner list is per-layer
                       snapshots at that timestep.  Typically obtained from
                       ``[step.traces for step in trace.steps]``.
            layer_name: Name of the layer to extract.

        Returns:
            List of trace norms, one per timestep that contains the layer.
        """
        curve: List[float] = []
        for step_traces in snapshots:
            for t in step_traces:
                if t.layer_name == layer_name:
                    curve.append(t.norm)
                    break
        return curve

    @staticmethod
    def detect_trace_explosion(
        snapshots: List[List[TraceSnapshot]],
        threshold: float = 100.0,
    ) -> List[str]:
        """Detect layers where the eligibility trace norm exceeds a threshold.

        Checks the *most recent* snapshot for each layer.

        Args:
            snapshots: Same structure as :meth:`compute_trace_decay_curve`.
            threshold: Norm threshold above which a trace is "exploding".

        Returns:
            Sorted list of layer names with exploding traces.
        """
        latest: Dict[str, float] = {}
        for step_traces in snapshots:
            for t in step_traces:
                latest[t.layer_name] = t.norm

        exploding = sorted(name for name, norm in latest.items() if norm > threshold)
        return exploding

    @staticmethod
    def detect_trace_vanishing(
        snapshots: List[List[TraceSnapshot]],
        threshold: float = 1e-6,
    ) -> List[str]:
        """Detect layers where the eligibility trace norm is near zero.

        Checks the *most recent* snapshot for each layer.

        Args:
            snapshots: Same structure as :meth:`compute_trace_decay_curve`.
            threshold: Norm threshold below which a trace is "vanishing".

        Returns:
            Sorted list of layer names with vanishing traces.
        """
        latest: Dict[str, float] = {}
        for step_traces in snapshots:
            for t in step_traces:
                latest[t.layer_name] = t.norm

        vanishing = sorted(name for name, norm in latest.items() if norm < threshold)
        return vanishing

    @staticmethod
    def top_updated_units(
        update_snapshots: List[List[UpdateSnapshot]],
        k: int = 10,
    ) -> List[Tuple[str, int, float]]:
        """Find the top-k layers/units by |delta_w| magnitude.

        Since ``UpdateSnapshot`` is per-layer (not per-unit), the "unit index"
        is the timestep index at which the largest update occurred for that layer.
        This gives an approximation: (layer_name, step_of_max_update, max_norm).

        Args:
            update_snapshots: Outer list is timesteps; inner list is per-layer
                              update snapshots.
            k: Number of top entries to return.

        Returns:
            Sorted list of (layer_name, timestep_index, delta_w_norm) tuples,
            descending by magnitude.
        """
        # Track per-layer maximum update
        layer_max: Dict[str, Tuple[int, float]] = {}
        for step_idx, step_updates in enumerate(update_snapshots):
            for u in step_updates:
                if u.layer_name not in layer_max or u.delta_w_norm > layer_max[u.layer_name][1]:
                    layer_max[u.layer_name] = (step_idx, u.delta_w_norm)

        ranked = sorted(
            [(name, idx, mag) for name, (idx, mag) in layer_max.items()],
            key=lambda x: x[2],
            reverse=True,
        )
        return ranked[:k]

    @staticmethod
    def compute_sparsity_trend(
        snapshots: List[List[TraceSnapshot]],
        layer_name: str,
    ) -> List[float]:
        """Extract the sparsity of a specific layer's trace over time.

        Args:
            snapshots: Per-step trace snapshot lists.
            layer_name: Target layer.

        Returns:
            List of sparsity values (fraction near zero).
        """
        curve: List[float] = []
        for step_traces in snapshots:
            for t in step_traces:
                if t.layer_name == layer_name:
                    curve.append(t.sparsity)
                    break
        return curve

    @staticmethod
    def compute_trace_statistics(
        snapshots: List[List[TraceSnapshot]],
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate statistics per layer across all timesteps.

        Returns:
            ``{"layer_name": {"mean_norm": ..., "max_norm": ...,
            "mean_sparsity": ..., "num_snapshots": ...}, ...}``
        """
        layer_data: Dict[str, Dict[str, List[float]]] = {}
        for step_traces in snapshots:
            for t in step_traces:
                entry = layer_data.setdefault(t.layer_name, {"norms": [], "sparsities": []})
                entry["norms"].append(t.norm)
                entry["sparsities"].append(t.sparsity)

        result: Dict[str, Dict[str, float]] = {}
        for name, data in layer_data.items():
            norms = data["norms"]
            sparsities = data["sparsities"]
            result[name] = {
                "mean_norm": _safe_mean(norms),
                "max_norm": max(norms) if norms else 0.0,
                "min_norm": min(norms) if norms else 0.0,
                "std_norm": _safe_std(norms),
                "mean_sparsity": _safe_mean(sparsities),
                "num_snapshots": float(len(norms)),
            }
        return result


# ---------------------------------------------------------------------------
# 5. PlasticityReport
# ---------------------------------------------------------------------------


class PlasticityReport:
    """Generate human-readable diagnostic reports from plasticity traces.

    All methods are static -- this class is a namespace for report generation.
    """

    @staticmethod
    def generate_report(trace: PlasticityTrace) -> str:
        """Generate a comprehensive human-readable diagnostic report.

        Sections:
        - Overview: step range, total steps, recording timestamps
        - Modulator Summary: per-modulator mean/std/min/max
        - Trace Health: per-layer norm statistics, explosion/vanishing warnings
        - Update Statistics: per-layer update norms, clamp hits
        - Warnings: actionable alerts for common pathologies

        Args:
            trace: A ``PlasticityTrace`` to analyze.

        Returns:
            Multi-line report string.
        """
        lines: List[str] = []
        sep = "=" * 72

        # ---- Header ----
        lines.append(sep)
        lines.append("  PLASTICITY DIAGNOSTICS REPORT")
        lines.append(sep)
        lines.append("")

        # ---- Overview ----
        lines.append("OVERVIEW")
        lines.append("-" * 40)
        lines.append(f"  Total steps recorded : {trace.total_steps}")
        if trace.steps:
            step_lo, step_hi = trace.step_range()
            lines.append(f"  Step range           : [{step_lo}, {step_hi}]")
            t0 = trace.steps[0].timestamp
            t1 = trace.steps[-1].timestamp
            if t1 > t0:
                elapsed = t1 - t0
                lines.append(f"  Wall-clock elapsed   : {elapsed:.2f}s")
                steps_per_sec = len(trace.steps) / elapsed if elapsed > 0 else 0
                lines.append(f"  Steps / sec          : {steps_per_sec:.1f}")
        else:
            lines.append("  (no steps recorded)")
        lines.append("")

        if not trace.steps:
            lines.append("No data to analyze.")
            lines.append(sep)
            return "\n".join(lines)

        # ---- Modulator Summary ----
        mod_history = [s.modulators for s in trace.steps]
        mod_stats = ModulatorAnalyzer.compute_modulator_stats(mod_history)

        lines.append("MODULATOR SUMMARY")
        lines.append("-" * 40)
        for mod_name in ("da", "ach", "ne", "sht"):
            st = mod_stats[mod_name]
            label = {"da": "DA (Dopamine)", "ach": "ACh (Acetylcholine)",
                     "ne": "NE (Norepinephrine)", "sht": "5-HT (Serotonin)"}[mod_name]
            lines.append(f"  {label}:")
            lines.append(
                f"    mean={st['mean']:.4f}  std={st['std']:.4f}  "
                f"min={st['min']:.4f}  max={st['max']:.4f}  "
                f"sat_rate={st['saturation_rate']:.2%}"
            )
        lines.append("")

        # ---- Trace Health ----
        all_trace_snapshots = [s.traces for s in trace.steps]
        trace_stats = TraceAnalyzer.compute_trace_statistics(all_trace_snapshots)
        exploding = TraceAnalyzer.detect_trace_explosion(all_trace_snapshots)
        vanishing = TraceAnalyzer.detect_trace_vanishing(all_trace_snapshots)

        lines.append("TRACE HEALTH")
        lines.append("-" * 40)
        if trace_stats:
            for layer_name in sorted(trace_stats.keys()):
                ts = trace_stats[layer_name]
                lines.append(f"  {layer_name}:")
                lines.append(
                    f"    mean_norm={ts['mean_norm']:.6f}  "
                    f"max_norm={ts['max_norm']:.6f}  "
                    f"mean_sparsity={ts['mean_sparsity']:.2%}"
                )
        else:
            lines.append("  (no trace data recorded)")
        lines.append("")

        # ---- Update Statistics ----
        all_update_snapshots = [s.updates for s in trace.steps]
        total_clamp = 0
        update_layer_norms: Dict[str, List[float]] = {}
        update_layer_lrs: Dict[str, List[float]] = {}
        for step_updates in all_update_snapshots:
            for u in step_updates:
                update_layer_norms.setdefault(u.layer_name, []).append(u.delta_w_norm)
                update_layer_lrs.setdefault(u.layer_name, []).append(u.effective_lr)
                total_clamp += u.clamp_hits

        lines.append("UPDATE STATISTICS")
        lines.append("-" * 40)
        if update_layer_norms:
            for layer_name in sorted(update_layer_norms.keys()):
                norms = update_layer_norms[layer_name]
                lrs = update_layer_lrs.get(layer_name, [])
                lines.append(f"  {layer_name}:")
                lines.append(
                    f"    mean_dw_norm={_safe_mean(norms):.6f}  "
                    f"max_dw_norm={max(norms):.6f}  "
                    f"mean_eff_lr={_safe_mean(lrs):.6f}"
                )
            lines.append(f"  Total clamp hits: {total_clamp}")
        else:
            lines.append("  (no update data recorded)")
        lines.append("")

        # ---- Plasticity Utilization ----
        nonzero_steps = sum(1 for s in trace.steps if s.has_nonzero_update())
        utilization = nonzero_steps / len(trace.steps) if trace.steps else 0.0
        lines.append("PLASTICITY UTILIZATION")
        lines.append("-" * 40)
        lines.append(
            f"  Steps with non-zero update: {nonzero_steps} / {len(trace.steps)} "
            f"({utilization:.2%})"
        )
        lines.append("")

        # ---- Warnings ----
        warnings: List[str] = []

        # Saturated modulators
        sat_fracs = ModulatorAnalyzer.detect_saturation(mod_history)
        for mod_name, frac in sat_fracs.items():
            if frac > 0.10:
                label = {"da": "DA", "ach": "ACh", "ne": "NE", "sht": "5-HT"}[mod_name]
                warnings.append(
                    f"[WARN] {label} saturated in {frac:.1%} of steps "
                    f"(threshold=10%). Consider normalizing inputs."
                )

        # Exploding traces
        for layer in exploding:
            warnings.append(
                f"[WARN] Trace explosion in '{layer}' "
                f"(norm > 100). Check tau_e and clamp_range."
            )

        # Vanishing traces
        for layer in vanishing:
            warnings.append(
                f"[WARN] Trace vanishing in '{layer}' "
                f"(norm < 1e-6). Check tau_e and input activity."
            )

        # Excessive clamping
        if total_clamp > len(trace.steps) * 0.5:
            warnings.append(
                f"[WARN] Excessive clamping ({total_clamp} total hits). "
                f"Consider widening weight_clamp range."
            )

        # Low utilization
        if utilization < 0.1 and len(trace.steps) > 10:
            warnings.append(
                f"[WARN] Low plasticity utilization ({utilization:.1%}). "
                f"Modulator signal may not be reaching eligible layers."
            )

        lines.append("WARNINGS")
        lines.append("-" * 40)
        if warnings:
            for w in warnings:
                lines.append(f"  {w}")
        else:
            lines.append("  No warnings. All diagnostics look healthy.")
        lines.append("")
        lines.append(sep)

        return "\n".join(lines)

    @staticmethod
    def generate_compact_summary(trace: PlasticityTrace) -> str:
        """Generate a one-line compact summary.

        Useful for progress bar annotations or log lines.

        Args:
            trace: A ``PlasticityTrace`` to summarize.

        Returns:
            A single-line string summary.
        """
        if not trace.steps:
            return "PlasticityTrace(empty)"

        mod_vals = [s.modulators for s in trace.steps]
        mean_da = _safe_mean([m.da for m in mod_vals])
        mean_ach = _safe_mean([m.ach for m in mod_vals])

        all_norms: List[float] = []
        for s in trace.steps:
            for t in s.traces:
                all_norms.append(t.norm)
        mean_tnorm = _safe_mean(all_norms) if all_norms else 0.0

        nonzero = sum(1 for s in trace.steps if s.has_nonzero_update())
        util = nonzero / len(trace.steps)

        return (
            f"steps={len(trace.steps)} "
            f"DA={mean_da:.3f} ACh={mean_ach:.3f} "
            f"trace_norm={mean_tnorm:.4f} "
            f"util={util:.1%}"
        )

    @staticmethod
    def generate_layer_report(
        trace: PlasticityTrace,
        layer_name: str,
    ) -> str:
        """Generate a detailed report for a single layer.

        Args:
            trace:      The plasticity trace to analyze.
            layer_name: Which layer to focus on.

        Returns:
            Multi-line report string focused on the specified layer.
        """
        lines: List[str] = []
        lines.append(f"Layer Report: {layer_name}")
        lines.append("=" * 60)

        # Collect trace data for this layer
        norms: List[float] = []
        sparsities: List[float] = []
        dw_norms: List[float] = []
        clamp_hits_total = 0
        eff_lrs: List[float] = []

        for s in trace.steps:
            for t in s.traces:
                if t.layer_name == layer_name:
                    norms.append(t.norm)
                    sparsities.append(t.sparsity)
            for u in s.updates:
                if u.layer_name == layer_name:
                    dw_norms.append(u.delta_w_norm)
                    clamp_hits_total += u.clamp_hits
                    eff_lrs.append(u.effective_lr)

        if norms:
            lines.append(f"  Trace norm   : mean={_safe_mean(norms):.6f} "
                         f"max={max(norms):.6f} min={min(norms):.6f}")
            lines.append(f"  Sparsity     : mean={_safe_mean(sparsities):.2%}")
        else:
            lines.append("  (no trace data for this layer)")

        if dw_norms:
            lines.append(f"  Update norm  : mean={_safe_mean(dw_norms):.6f} "
                         f"max={max(dw_norms):.6f}")
            lines.append(f"  Clamp hits   : {clamp_hits_total}")
            lines.append(f"  Effective LR : mean={_safe_mean(eff_lrs):.6f}")
        else:
            lines.append("  (no update data for this layer)")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. Checkpoint Integration
# ---------------------------------------------------------------------------


def save_diagnostics_with_checkpoint(
    trace: PlasticityTrace,
    checkpoint_path: Union[str, Path],
) -> Path:
    """Save plasticity diagnostics alongside a model checkpoint.

    Creates a ``.plasticity.json`` file next to the checkpoint.

    Args:
        trace:           The ``PlasticityTrace`` to save.
        checkpoint_path: Path to the model checkpoint (``.pt`` / ``.pth``).

    Returns:
        Path to the saved diagnostics file.
    """
    checkpoint_path = Path(checkpoint_path)
    diag_path = checkpoint_path.with_suffix(".plasticity.json")

    data = trace.to_dict()
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    _logger.info(
        "Saved plasticity diagnostics (%d steps) to %s",
        len(trace.steps),
        diag_path,
    )
    return diag_path


def load_diagnostics_from_checkpoint(
    checkpoint_path: Union[str, Path],
) -> PlasticityTrace:
    """Load plasticity diagnostics that were saved alongside a checkpoint.

    Args:
        checkpoint_path: Path to the model checkpoint (``.pt`` / ``.pth``).

    Returns:
        The restored ``PlasticityTrace``.

    Raises:
        FileNotFoundError: If the diagnostics file does not exist.
    """
    checkpoint_path = Path(checkpoint_path)
    diag_path = checkpoint_path.with_suffix(".plasticity.json")

    if not diag_path.exists():
        raise FileNotFoundError(
            f"No plasticity diagnostics found at {diag_path}. "
            f"Ensure save_diagnostics_with_checkpoint was called."
        )

    with open(diag_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    trace = PlasticityTrace.from_dict(data)
    _logger.info(
        "Loaded plasticity diagnostics (%d steps) from %s",
        len(trace.steps),
        diag_path,
    )
    return trace


# ---------------------------------------------------------------------------
# 7. Helper utilities (module-private)
# ---------------------------------------------------------------------------


def _safe_mean(values: List[float]) -> float:
    """Compute mean of a list, returning 0.0 for empty lists."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _safe_std(values: List[float]) -> float:
    """Compute population standard deviation, returning 0.0 for empty/singleton."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return math.sqrt(variance)


def _prefix_sum(values: List[float]) -> List[float]:
    """Compute prefix sums for sliding-window calculations.

    Returns a list of length ``len(values) + 1`` where ``result[0] = 0``
    and ``result[i] = sum(values[:i])``.
    """
    result = [0.0]
    running = 0.0
    for v in values:
        running += v
        result.append(running)
    return result


# ---------------------------------------------------------------------------
# 8. Convenience factory functions
# ---------------------------------------------------------------------------


def create_modulator_snapshot_from_dict(
    modulators: Dict[str, float],
) -> ModulatorSnapshot:
    """Create a ``ModulatorSnapshot`` from a dictionary of modulator values.

    Accepts keys: ``"da"``, ``"ach"``, ``"ne"``, ``"sht"`` (or
    ``"dopamine"``, ``"acetylcholine"``, ``"norepinephrine"``, ``"serotonin"``).

    Args:
        modulators: Mapping of modulator names to values.

    Returns:
        A populated ``ModulatorSnapshot``.
    """
    def _get(key: str, alt: str) -> float:
        return float(modulators.get(key, modulators.get(alt, 0.0)))

    return ModulatorSnapshot(
        da=_get("da", "dopamine"),
        ach=_get("ach", "acetylcholine"),
        ne=_get("ne", "norepinephrine"),
        sht=_get("sht", "serotonin"),
    )


def create_modulator_snapshot_from_tensors(
    modulators: Dict[str, torch.Tensor],
) -> ModulatorSnapshot:
    """Create a ``ModulatorSnapshot`` from a dictionary of tensors.

    Takes the mean of each tensor (collapsing batch dims).

    Args:
        modulators: ``{"da": Tensor, "ach": Tensor, ...}`` or full names.

    Returns:
        A populated ``ModulatorSnapshot``.
    """
    def _get(key: str, alt: str) -> float:
        t = modulators.get(key, modulators.get(alt, None))
        if t is None:
            return 0.0
        return float(t.detach().float().mean().item())

    return ModulatorSnapshot(
        da=_get("da", "dopamine"),
        ach=_get("ach", "acetylcholine"),
        ne=_get("ne", "norepinephrine"),
        sht=_get("sht", "serotonin"),
    )


def collect_trace_snapshots(
    traces: Dict[str, torch.Tensor],
) -> List[TraceSnapshot]:
    """Build ``TraceSnapshot`` objects from a dictionary of trace tensors.

    Args:
        traces: ``{layer_name: trace_tensor}``

    Returns:
        List of ``TraceSnapshot`` objects.
    """
    snapshots: List[TraceSnapshot] = []
    for name, tensor in traces.items():
        snapshots.append(TraceSnapshot.from_tensor(name, tensor))
    return snapshots


def collect_update_snapshots(
    deltas: Dict[str, torch.Tensor],
    clamp_lo: float = -1.0,
    clamp_hi: float = 1.0,
    effective_lr: float = 0.0,
) -> List[UpdateSnapshot]:
    """Build ``UpdateSnapshot`` objects from a dictionary of weight deltas.

    Args:
        deltas:       ``{layer_name: delta_w_tensor}``
        clamp_lo:     Lower clamp bound.
        clamp_hi:     Upper clamp bound.
        effective_lr: Effective learning rate used.

    Returns:
        List of ``UpdateSnapshot`` objects.
    """
    snapshots: List[UpdateSnapshot] = []
    for name, delta in deltas.items():
        snapshots.append(
            UpdateSnapshot.from_delta_tensor(name, delta, clamp_lo, clamp_hi, effective_lr)
        )
    return snapshots


# ---------------------------------------------------------------------------
# 9. Self-Test Block
# ---------------------------------------------------------------------------


def _run_self_tests() -> None:
    """Run comprehensive self-tests for all diagnostics components.

    Prints PASS/FAIL for each test case.  Target: 18+ tests.
    """
    import tempfile
    import os

    passed = 0
    failed = 0
    total = 0

    def _test(name: str, condition: bool, detail: str = "") -> None:
        nonlocal passed, failed, total
        total += 1
        if condition:
            passed += 1
            print(f"  [{total:2d}] PASS: {name}")
        else:
            failed += 1
            msg = f"  [{total:2d}] FAIL: {name}"
            if detail:
                msg += f" -- {detail}"
            print(msg)

    print("=" * 72)
    print("  Plasticity Diagnostics Self-Test")
    print("=" * 72)
    print()

    # ---- Test 1: PlasticityLogger log_step records correctly ----
    logger = PlasticityLogger(log_every_n_steps=1, max_history=100, detailed=True)
    mod = ModulatorSnapshot(da=0.5, ach=0.3, ne=0.2, sht=0.6)
    tr = [TraceSnapshot(layer_name="layer1", norm=1.0, mean=0.1,
                        max_val=0.5, min_val=-0.1, sparsity=0.2)]
    up = [UpdateSnapshot(layer_name="layer1", delta_w_norm=0.01,
                         clamp_hits=0, effective_lr=0.001)]
    logger.log_step(step=0, modulators=mod, traces=tr, updates=up)
    _test(
        "PlasticityLogger: log_step records correctly",
        logger.num_recorded == 1
        and logger.last_step() is not None
        and logger.last_step().modulators.da == 0.5
        and len(logger.last_step().traces) == 1
        and logger.last_step().traces[0].layer_name == "layer1",
    )

    # ---- Test 2: PlasticityLogger get_trace returns correct step range ----
    for i in range(1, 20):
        logger.log_step(
            step=i,
            modulators=ModulatorSnapshot(da=i * 0.01, ach=0.1, ne=0.1, sht=0.1),
            traces=[TraceSnapshot(layer_name="layer1", norm=float(i))],
            updates=[],
        )
    trace_obj = logger.get_trace(start_step=5, end_step=10)
    step_indices = [s.step_idx for s in trace_obj.steps]
    _test(
        "PlasticityLogger: get_trace returns correct step range",
        all(5 <= idx <= 10 for idx in step_indices)
        and len(step_indices) == 6,
        f"got steps={step_indices}",
    )

    # ---- Test 3: PlasticityLogger clear resets history ----
    logger.clear()
    _test(
        "PlasticityLogger: clear resets history",
        logger.num_recorded == 0 and logger.total_steps_seen == 0,
    )

    # ---- Test 4: Epoch summary mean values computed correctly ----
    logger2 = PlasticityLogger(log_every_n_steps=1, max_history=1000, detailed=True)
    for i in range(10):
        logger2.log_step(
            step=i,
            modulators=ModulatorSnapshot(da=0.5, ach=0.3, ne=0.2, sht=0.4),
            traces=[TraceSnapshot(layer_name="w1", norm=1.0)],
            updates=[UpdateSnapshot(layer_name="w1", delta_w_norm=0.01, clamp_hits=1,
                                    effective_lr=0.001)],
        )
    summary = logger2.get_epoch_summary(range(0, 10), epoch=1)
    _test(
        "Epoch summary: mean values computed correctly",
        abs(summary.mean_da - 0.5) < 1e-6
        and abs(summary.mean_ach - 0.3) < 1e-6
        and abs(summary.mean_ne - 0.2) < 1e-6
        and abs(summary.mean_sht - 0.4) < 1e-6
        and summary.epoch == 1,
        f"da={summary.mean_da}, ach={summary.mean_ach}",
    )

    # ---- Test 5: Modulator stats mean/std/min/max correct ----
    history = [
        ModulatorSnapshot(da=0.2, ach=0.1, ne=0.5, sht=0.3),
        ModulatorSnapshot(da=0.4, ach=0.3, ne=0.5, sht=0.7),
        ModulatorSnapshot(da=0.6, ach=0.5, ne=0.5, sht=0.5),
    ]
    stats = ModulatorAnalyzer.compute_modulator_stats(history)
    _test(
        "Modulator stats: mean/std/min/max correct for known data",
        abs(stats["da"]["mean"] - 0.4) < 1e-6
        and abs(stats["da"]["min"] - 0.2) < 1e-6
        and abs(stats["da"]["max"] - 0.6) < 1e-6
        and abs(stats["ne"]["std"] - 0.0) < 1e-6,
        f"da_mean={stats['da']['mean']}, ne_std={stats['ne']['std']}",
    )

    # ---- Test 6: Saturation detection ----
    sat_history = [
        ModulatorSnapshot(da=0.99, ach=0.98, ne=0.1, sht=0.05),
        ModulatorSnapshot(da=0.99, ach=0.99, ne=0.1, sht=0.05),
        ModulatorSnapshot(da=0.3, ach=0.96, ne=0.1, sht=0.05),
    ]
    sat = ModulatorAnalyzer.detect_saturation(sat_history, threshold=0.95)
    _test(
        "Saturation detection: detects when modulator near bounds",
        sat["da"] > 0.5  # 2/3 are > 0.95
        and sat["ach"] > 0.5  # all 3 are > 0.95
        and sat["ne"] < 0.01
        and sat["sht"] < 0.01,
        f"da_sat={sat['da']:.2f}, ach_sat={sat['ach']:.2f}",
    )

    # ---- Test 7: DA-reward correlation ----
    import random
    random.seed(42)
    # Correlated: DA = reward + noise
    rewards_corr = [float(i) / 50.0 for i in range(50)]
    da_corr = [r + random.gauss(0, 0.01) for r in rewards_corr]
    corr_pos = ModulatorAnalyzer.compute_da_reward_correlation(da_corr, rewards_corr)

    # Uncorrelated: DA random, reward increasing
    da_uncorr = [random.gauss(0, 1) for _ in range(50)]
    rewards_uncorr = [float(i) / 50.0 for i in range(50)]
    corr_zero = ModulatorAnalyzer.compute_da_reward_correlation(da_uncorr, rewards_uncorr)

    _test(
        "DA-reward correlation: positive for correlated, near zero for uncorrelated",
        corr_pos > 0.9 and abs(corr_zero) < 0.5,
        f"corr_pos={corr_pos:.3f}, corr_zero={corr_zero:.3f}",
    )

    # ---- Test 8: Trace decay curve ----
    decay_snapshots: List[List[TraceSnapshot]] = []
    for i in range(20):
        norm_val = 10.0 * (0.9 ** i)  # Exponential decay
        decay_snapshots.append([
            TraceSnapshot(layer_name="decay_layer", norm=norm_val),
        ])
    curve = TraceAnalyzer.compute_trace_decay_curve(decay_snapshots, "decay_layer")
    _test(
        "Trace decay curve: decreasing for decaying traces",
        len(curve) == 20
        and all(curve[i] >= curve[i + 1] for i in range(len(curve) - 1)),
        f"curve_start={curve[0]:.2f}, curve_end={curve[-1]:.4f}",
    )

    # ---- Test 9: Trace explosion detection ----
    explosion_snapshots: List[List[TraceSnapshot]] = [
        [
            TraceSnapshot(layer_name="safe_layer", norm=5.0),
            TraceSnapshot(layer_name="exploding_layer", norm=200.0),
            TraceSnapshot(layer_name="another_safe", norm=50.0),
        ]
    ]
    exploded = TraceAnalyzer.detect_trace_explosion(explosion_snapshots, threshold=100.0)
    _test(
        "Trace explosion detection: flags large norms",
        exploded == ["exploding_layer"],
        f"exploded={exploded}",
    )

    # ---- Test 10: Trace vanishing detection ----
    vanishing_snapshots: List[List[TraceSnapshot]] = [
        [
            TraceSnapshot(layer_name="active_layer", norm=1.0),
            TraceSnapshot(layer_name="dead_layer", norm=1e-8),
            TraceSnapshot(layer_name="also_dead", norm=0.0),
        ]
    ]
    vanished = TraceAnalyzer.detect_trace_vanishing(vanishing_snapshots, threshold=1e-6)
    _test(
        "Trace vanishing detection: flags small norms",
        "dead_layer" in vanished and "also_dead" in vanished
        and "active_layer" not in vanished,
        f"vanished={vanished}",
    )

    # ---- Test 11: Top updated units ----
    update_snapshots_list: List[List[UpdateSnapshot]] = [
        [
            UpdateSnapshot(layer_name="layer_a", delta_w_norm=0.5),
            UpdateSnapshot(layer_name="layer_b", delta_w_norm=2.0),
            UpdateSnapshot(layer_name="layer_c", delta_w_norm=0.1),
        ],
        [
            UpdateSnapshot(layer_name="layer_a", delta_w_norm=0.3),
            UpdateSnapshot(layer_name="layer_b", delta_w_norm=1.0),
            UpdateSnapshot(layer_name="layer_d", delta_w_norm=3.0),
        ],
    ]
    top = TraceAnalyzer.top_updated_units(update_snapshots_list, k=3)
    _test(
        "Top updated units: returns k items sorted by magnitude",
        len(top) == 3
        and top[0][0] == "layer_d"
        and top[0][2] == 3.0
        and top[1][0] == "layer_b"
        and top[1][2] == 2.0,
        f"top={top}",
    )

    # ---- Test 12: Plasticity report ----
    report_logger = PlasticityLogger(log_every_n_steps=1, max_history=100, detailed=True)
    for i in range(5):
        report_logger.log_step(
            step=i,
            modulators=ModulatorSnapshot(da=0.5, ach=0.3, ne=0.2, sht=0.4),
            traces=[TraceSnapshot(layer_name="fc1", norm=1.0 + i * 0.1,
                                  mean=0.01, max_val=0.5, min_val=-0.1,
                                  sparsity=0.3)],
            updates=[UpdateSnapshot(layer_name="fc1", delta_w_norm=0.005,
                                    clamp_hits=0, effective_lr=0.001)],
        )
    report_trace = report_logger.get_trace()
    report = PlasticityReport.generate_report(report_trace)
    _test(
        "Plasticity report: non-empty string with expected sections",
        len(report) > 100
        and "OVERVIEW" in report
        and "MODULATOR SUMMARY" in report
        and "TRACE HEALTH" in report
        and "UPDATE STATISTICS" in report
        and "WARNINGS" in report,
    )

    # ---- Test 13: JSON save/load round-trip ----
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "test_diag.json")
        report_logger_copy = PlasticityLogger(
            log_every_n_steps=1, max_history=100, detailed=True
        )
        for i in range(5):
            report_logger_copy.log_step(
                step=i,
                modulators=ModulatorSnapshot(da=0.1 * i, ach=0.2, ne=0.3, sht=0.4),
                traces=[TraceSnapshot(layer_name="fc1", norm=float(i))],
                updates=[UpdateSnapshot(layer_name="fc1", delta_w_norm=0.01 * i)],
            )
        report_logger_copy.save_json(json_path)
        loaded = PlasticityLogger.load_json(json_path)
        _test(
            "JSON save/load round-trip: data preserved",
            loaded.num_recorded == report_logger_copy.num_recorded
            and loaded.last_step().modulators.da == report_logger_copy.last_step().modulators.da
            and loaded.last_step().traces[0].norm == report_logger_copy.last_step().traces[0].norm,
            f"recorded={loaded.num_recorded}, "
            f"da={loaded.last_step().modulators.da}",
        )

    # ---- Test 14: return_details=False scenario ----
    no_detail_logger = PlasticityLogger(
        log_every_n_steps=1, max_history=100, detailed=False
    )
    for i in range(5):
        no_detail_logger.log_step(
            step=i,
            modulators=ModulatorSnapshot(da=0.5, ach=0.3, ne=0.2, sht=0.4),
            traces=[TraceSnapshot(layer_name="fc1", norm=1.0)],
            updates=[UpdateSnapshot(layer_name="fc1", delta_w_norm=0.01)],
        )
    nd_trace = no_detail_logger.get_trace()
    # When detailed=False, traces and updates should be empty
    all_traces_empty = all(len(s.traces) == 0 for s in nd_trace.steps)
    all_updates_empty = all(len(s.updates) == 0 for s in nd_trace.steps)
    _test(
        "return_details=False scenario: no overhead (empty trace/update data)",
        all_traces_empty and all_updates_empty and nd_trace.total_steps == 5,
        f"traces_empty={all_traces_empty}, updates_empty={all_updates_empty}",
    )

    # ---- Test 15: Modulator temporal profile ----
    profile_history = [
        ModulatorSnapshot(da=float(i), ach=0.0, ne=0.0, sht=0.0)
        for i in range(10)
    ]
    profile = ModulatorAnalyzer.modulator_temporal_profile(profile_history, window=3)
    # Window of size 3 over [0,1,2,3,...,9] => means: [1.0, 2.0, 3.0, ..., 8.0]
    expected_da = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    profile_correct = (
        len(profile["da"]) == 8
        and all(abs(profile["da"][i] - expected_da[i]) < 1e-6 for i in range(8))
    )
    _test(
        "Modulator temporal profile: correct window averaging",
        profile_correct,
        f"da_profile={profile['da'][:4]}... expected={expected_da[:4]}...",
    )

    # ---- Test 16: PlasticityTrace JSON-serializable ----
    test_trace = PlasticityTrace(
        steps=[
            StepDiagnostics(
                step_idx=0,
                modulators=ModulatorSnapshot(da=0.5, ach=0.3, ne=0.2, sht=0.1),
                traces=[TraceSnapshot(layer_name="l1", norm=1.5)],
                updates=[UpdateSnapshot(layer_name="l1", delta_w_norm=0.02)],
                timestamp=1000.0,
            )
        ],
        total_steps=1,
        summary={"test_key": "test_value"},
    )
    json_str = test_trace.to_json()
    restored = PlasticityTrace.from_json(json_str)
    _test(
        "PlasticityTrace: JSON-serializable",
        restored.total_steps == 1
        and restored.steps[0].modulators.da == 0.5
        and restored.steps[0].traces[0].layer_name == "l1"
        and restored.summary.get("test_key") == "test_value",
    )

    # ---- Test 17: EpochSummary all fields populated ----
    epoch_logger = PlasticityLogger(log_every_n_steps=1, max_history=100, detailed=True)
    for i in range(20):
        epoch_logger.log_step(
            step=i,
            modulators=ModulatorSnapshot(da=0.3, ach=0.4, ne=0.5, sht=0.6),
            traces=[
                TraceSnapshot(layer_name="l1", norm=2.0),
                TraceSnapshot(layer_name="l2", norm=3.0),
            ],
            updates=[
                UpdateSnapshot(layer_name="l1", delta_w_norm=0.01, clamp_hits=2,
                               effective_lr=0.001),
                UpdateSnapshot(layer_name="l2", delta_w_norm=0.02, clamp_hits=1,
                               effective_lr=0.002),
            ],
        )
    epoch_sum = epoch_logger.get_epoch_summary(range(0, 20), epoch=3)
    _test(
        "EpochSummary: all fields populated",
        epoch_sum.epoch == 3
        and abs(epoch_sum.mean_da - 0.3) < 1e-6
        and "l1" in epoch_sum.mean_trace_norm_per_layer
        and "l2" in epoch_sum.mean_trace_norm_per_layer
        and abs(epoch_sum.mean_trace_norm_per_layer["l1"] - 2.0) < 1e-6
        and "l1" in epoch_sum.mean_update_norm_per_layer
        and epoch_sum.total_clamp_hits == 60  # 20 steps * (2+1) clamp hits
        and epoch_sum.plasticity_utilization > 0.0,
        f"epoch={epoch_sum.epoch}, da={epoch_sum.mean_da}, "
        f"clamp={epoch_sum.total_clamp_hits}, util={epoch_sum.plasticity_utilization}",
    )

    # ---- Test 18: Plasticity utilization correct fraction ----
    util_logger = PlasticityLogger(log_every_n_steps=1, max_history=100, detailed=True)
    for i in range(10):
        dw = 0.01 if i % 2 == 0 else 0.0  # 5 out of 10 have non-zero
        util_logger.log_step(
            step=i,
            modulators=ModulatorSnapshot(da=0.5),
            traces=[],
            updates=[UpdateSnapshot(layer_name="l1", delta_w_norm=dw)],
        )
    util_summary = util_logger.get_epoch_summary(range(0, 10))
    _test(
        "Plasticity utilization: correct fraction",
        abs(util_summary.plasticity_utilization - 0.5) < 1e-6,
        f"utilization={util_summary.plasticity_utilization}",
    )

    # ---- Test 19: TraceSnapshot.from_tensor ----
    test_tensor = torch.randn(10, 20)
    snap = TraceSnapshot.from_tensor("test_layer", test_tensor)
    _test(
        "TraceSnapshot.from_tensor: correct statistics",
        snap.layer_name == "test_layer"
        and snap.norm > 0
        and snap.max_val > snap.min_val
        and 0.0 <= snap.sparsity <= 1.0,
        f"norm={snap.norm:.4f}, sparsity={snap.sparsity:.4f}",
    )

    # ---- Test 20: Checkpoint save/load round-trip ----
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "model.pt")
        # Create a dummy checkpoint
        torch.save({"epoch": 1}, ckpt_path)

        save_trace = PlasticityTrace(
            steps=[
                StepDiagnostics(
                    step_idx=42,
                    modulators=ModulatorSnapshot(da=0.7, ach=0.2, ne=0.1, sht=0.9),
                    traces=[TraceSnapshot(layer_name="ckpt_layer", norm=3.14)],
                    updates=[],
                    timestamp=9999.0,
                ),
            ],
            total_steps=1,
            summary={"checkpoint_test": True},
        )
        diag_file = save_diagnostics_with_checkpoint(save_trace, ckpt_path)
        loaded_trace = load_diagnostics_from_checkpoint(ckpt_path)
        _test(
            "Checkpoint save/load round-trip: data preserved",
            loaded_trace.total_steps == 1
            and loaded_trace.steps[0].step_idx == 42
            and abs(loaded_trace.steps[0].modulators.da - 0.7) < 1e-6
            and loaded_trace.steps[0].traces[0].layer_name == "ckpt_layer"
            and abs(loaded_trace.steps[0].traces[0].norm - 3.14) < 1e-6,
        )

    # ---- Test 21: max_history eviction ----
    evict_logger = PlasticityLogger(log_every_n_steps=1, max_history=5, detailed=True)
    for i in range(10):
        evict_logger.log_step(
            step=i,
            modulators=ModulatorSnapshot(da=float(i)),
        )
    _test(
        "max_history eviction: oldest records dropped",
        evict_logger.num_recorded == 5
        and evict_logger.last_step().step_idx == 9
        and evict_logger._history[0].step_idx == 5,
        f"recorded={evict_logger.num_recorded}, "
        f"first_step={evict_logger._history[0].step_idx}",
    )

    # ---- Test 22: log_every_n_steps sampling ----
    sample_logger = PlasticityLogger(log_every_n_steps=3, max_history=100, detailed=True)
    for i in range(9):
        sample_logger.log_step(
            step=i,
            modulators=ModulatorSnapshot(da=float(i)),
        )
    _test(
        "log_every_n_steps: only records every Nth step",
        sample_logger.num_recorded == 3
        and sample_logger.total_steps_seen == 9,
        f"recorded={sample_logger.num_recorded}, seen={sample_logger.total_steps_seen}",
    )

    # ---- Test 23: compact summary ----
    compact_trace = report_logger.get_trace()
    compact = PlasticityReport.generate_compact_summary(compact_trace)
    _test(
        "Compact summary: produces one-line string",
        isinstance(compact, str)
        and "steps=" in compact
        and "DA=" in compact
        and len(compact.split("\n")) == 1,
        f"compact='{compact[:80]}...'",
    )

    # ---- Test 24: collect_trace_snapshots from tensors ----
    tensor_dict = {
        "layer_a": torch.randn(5, 5),
        "layer_b": torch.zeros(3, 3),
    }
    collected = collect_trace_snapshots(tensor_dict)
    _test(
        "collect_trace_snapshots: correct snapshots from tensors",
        len(collected) == 2
        and any(c.layer_name == "layer_a" and c.norm > 0 for c in collected)
        and any(c.layer_name == "layer_b" and c.sparsity == 1.0 for c in collected),
        f"names={[c.layer_name for c in collected]}",
    )

    # ---- Test 25: ModulatorSnapshot convenience methods ----
    ms = ModulatorSnapshot(da=0.9, ach=0.1, ne=0.5, sht=0.3)
    _test(
        "ModulatorSnapshot: convenience methods work",
        len(ms.as_list()) == 4
        and ms.magnitude() > 0
        and not ms.is_saturated(0.95)["da"]
        and ms.is_saturated(0.85)["da"],
    )

    # ---- Test 26: Empty trace report ----
    empty_trace = PlasticityTrace()
    empty_report = PlasticityReport.generate_report(empty_trace)
    _test(
        "Empty trace: report handles gracefully",
        "No data to analyze" in empty_report or "no steps recorded" in empty_report,
    )

    # ---- Summary ----
    print()
    print("=" * 72)
    print(f"  Results: {passed} passed, {failed} failed, {total} total")
    if failed == 0:
        print("  All tests PASSED.")
    else:
        print(f"  {failed} test(s) FAILED.")
    print("=" * 72)


if __name__ == "__main__":
    _run_self_tests()
