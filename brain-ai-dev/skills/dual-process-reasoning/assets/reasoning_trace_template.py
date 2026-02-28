#!/usr/bin/env python3
"""Reasoning Trace: JSON-serializable audit trail for dual-process reasoning.

Every item that passes through the dual-process pipeline produces a
``ReasoningTrace`` recording *why* the routing decision was made, what
System 1 predicted, and (if invoked) the full iterative trajectory of
System 2.  Traces are designed for:

* **Debugging** -- reproduce routing anomalies offline.
* **Evaluation** -- compute system-2 engagement rate, convergence rate,
  average deliberation steps across a dataset.
* **Logging** -- stream JSONL to disk for later analysis.
* **Comparison** -- ``TraceDiff`` / ``compare_traces`` enables regression
  testing between model versions.

Dataclass hierarchy::

    ReasoningTrace
    +-- RouteTrace       (metacognitive routing decision)
    +-- System1Trace     (System 1 top-k predictions)
    +-- System2Trace?    (iterative GRU loop, only when routed)
    |   +-- StepTrace[]  (per-step convergence metrics)
    +-- metadata         (arbitrary key-value pairs)

    BatchTrace           (list of ReasoningTrace for a full batch)

Helper utilities::

    TraceBuilder   -- conditional builder (no-op when disabled)
    TraceDiff      -- structured diff between two traces
    TraceLogger    -- JSONL file / stream logger with buffer
    extract_top_k  -- torch.topk wrapper returning plain lists
    make_s1_only_trace / make_s2_trace  -- factory shortcuts
    filter_traces / trace_statistics / merge_batch_traces

Dependencies: Python stdlib (dataclasses, json, typing).  ``torch`` is an
optional import used only by ``extract_top_k``.

Copy this template to ``brain_ai/reasoning/trace.py`` when integrating.
"""

from __future__ import annotations

import io
import json
import math
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

# ---------------------------------------------------------------------------
# Optional torch import -- only needed for extract_top_k
# ---------------------------------------------------------------------------
try:
    import torch
    from torch import Tensor

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False
    Tensor = Any  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
__all__ = [
    "RouteTrace",
    "System1Trace",
    "StepTrace",
    "System2Trace",
    "ReasoningTrace",
    "BatchTrace",
    "TraceBuilder",
    "TraceDiff",
    "compare_traces",
    "TraceLogger",
    "extract_top_k",
    "make_s1_only_trace",
    "make_s2_trace",
    "filter_traces",
    "trace_statistics",
    "merge_batch_traces",
]

_EPS: float = 1e-8


# ============================================================================
# SECTION 1 -- RouteTrace
# ============================================================================


@dataclass
class RouteTrace:
    """Metacognitive routing decision for a single input item.

    Records the raw and calibrated confidence values, uncertainty metrics,
    and the final routing decision (System 1 only vs. System 2 engaged).

    Attributes:
        used_system2:    Whether the item was routed to System 2.
        route_score:     Composite routing score (higher = more uncertain).
        threshold:       Routing threshold applied.
        conf_raw:        Raw softmax confidence from System 1.
        conf_calibrated: Calibrated confidence after temperature scaling.
        entropy:         Softmax entropy of System 1 logits.
        margin:          Gap between top-1 and top-2 softmax probabilities.
        novelty:         Optional novelty score from HTM / prototype distance.
        anomaly:         Optional anomaly score from HTM temporal memory.
        ignition:        Optional global workspace ignition strength.
        steps_budget:    Maximum number of System 2 steps allocated.
    """

    used_system2: bool
    route_score: float
    threshold: float
    conf_raw: float
    conf_calibrated: float
    entropy: float
    margin: float
    novelty: Optional[float] = None
    anomaly: Optional[float] = None
    ignition: Optional[float] = None
    steps_budget: int = 0

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dictionary suitable for JSON serialization."""
        d: Dict[str, Any] = {
            "used_system2": self.used_system2,
            "route_score": self.route_score,
            "threshold": self.threshold,
            "conf_raw": self.conf_raw,
            "conf_calibrated": self.conf_calibrated,
            "entropy": self.entropy,
            "margin": self.margin,
            "steps_budget": self.steps_budget,
        }
        if self.novelty is not None:
            d["novelty"] = self.novelty
        if self.anomaly is not None:
            d["anomaly"] = self.anomaly
        if self.ignition is not None:
            d["ignition"] = self.ignition
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RouteTrace":
        """Reconstruct a ``RouteTrace`` from a dictionary."""
        return cls(
            used_system2=bool(d["used_system2"]),
            route_score=float(d["route_score"]),
            threshold=float(d["threshold"]),
            conf_raw=float(d["conf_raw"]),
            conf_calibrated=float(d["conf_calibrated"]),
            entropy=float(d["entropy"]),
            margin=float(d["margin"]),
            novelty=float(d["novelty"]) if d.get("novelty") is not None else None,
            anomaly=float(d["anomaly"]) if d.get("anomaly") is not None else None,
            ignition=float(d["ignition"]) if d.get("ignition") is not None else None,
            steps_budget=int(d.get("steps_budget", 0)),
        )


# ============================================================================
# SECTION 2 -- System1Trace
# ============================================================================


@dataclass
class System1Trace:
    """Prediction summary from System 1 (fast path).

    Stores the top-k class indices and their corresponding softmax
    probabilities (or logit values) for quick inspection.

    Attributes:
        top_k_indices: Class indices of the top-k predictions.
        top_k_values:  Corresponding values (softmax probs or logits).
        top_k:         The k used for extraction.
    """

    top_k_indices: List[int]
    top_k_values: List[float]
    top_k: int

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dictionary."""
        return {
            "top_k_indices": list(self.top_k_indices),
            "top_k_values": [float(v) for v in self.top_k_values],
            "top_k": self.top_k,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "System1Trace":
        """Reconstruct from a dictionary."""
        return cls(
            top_k_indices=[int(i) for i in d["top_k_indices"]],
            top_k_values=[float(v) for v in d["top_k_values"]],
            top_k=int(d["top_k"]),
        )


# ============================================================================
# SECTION 3 -- StepTrace (System 2 per-step record)
# ============================================================================


@dataclass
class StepTrace:
    """Per-step convergence record for a single System 2 iteration.

    Each iteration of the GRU deliberation loop produces a ``StepTrace``
    recording convergence metrics used by the halting criterion.

    Attributes:
        step_idx:          Zero-based step index.
        conf_k:            Confidence at this step (softmax top-1).
        delta_kl:          KL divergence between this step and previous.
        delta_max:         Max absolute logit change from previous step.
        argmax_k:          Argmax class index at this step.
        halt_check_passed: Whether halting criterion was satisfied.
        top_k_indices:     Optional top-k class indices at this step.
        top_k_values:      Optional top-k values at this step.
    """

    step_idx: int
    conf_k: float
    delta_kl: float
    delta_max: float
    argmax_k: int
    halt_check_passed: bool
    top_k_indices: Optional[List[int]] = None
    top_k_values: Optional[List[float]] = None

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dictionary."""
        d: Dict[str, Any] = {
            "step_idx": self.step_idx,
            "conf_k": self.conf_k,
            "delta_kl": self.delta_kl,
            "delta_max": self.delta_max,
            "argmax_k": self.argmax_k,
            "halt_check_passed": self.halt_check_passed,
        }
        if self.top_k_indices is not None:
            d["top_k_indices"] = list(self.top_k_indices)
        if self.top_k_values is not None:
            d["top_k_values"] = [float(v) for v in self.top_k_values]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StepTrace":
        """Reconstruct from a dictionary."""
        return cls(
            step_idx=int(d["step_idx"]),
            conf_k=float(d["conf_k"]),
            delta_kl=float(d["delta_kl"]),
            delta_max=float(d["delta_max"]),
            argmax_k=int(d["argmax_k"]),
            halt_check_passed=bool(d["halt_check_passed"]),
            top_k_indices=(
                [int(i) for i in d["top_k_indices"]]
                if d.get("top_k_indices") is not None
                else None
            ),
            top_k_values=(
                [float(v) for v in d["top_k_values"]]
                if d.get("top_k_values") is not None
                else None
            ),
        )


# ============================================================================
# SECTION 4 -- System2Trace
# ============================================================================


@dataclass
class System2Trace:
    """Full trajectory of the System 2 iterative deliberation loop.

    Contains per-step records and summary statistics for convergence
    analysis.

    Attributes:
        steps_used:          Number of GRU steps actually executed.
        converged:           Whether the halting criterion was met before
                             exhausting the budget.
        halt_reason:         Why the loop stopped: ``"converged"``,
                             ``"budget_exhausted"``, ``"confidence_met"``,
                             or ``"argmax_stable"``.
        steps:               Ordered list of per-step traces.
        final_top_k_indices: Top-k class indices from the final step.
        final_top_k_values:  Top-k values from the final step.
    """

    steps_used: int
    converged: bool
    halt_reason: str
    steps: List[StepTrace]
    final_top_k_indices: List[int]
    final_top_k_values: List[float]

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dictionary."""
        return {
            "steps_used": self.steps_used,
            "converged": self.converged,
            "halt_reason": self.halt_reason,
            "steps": [s.to_dict() for s in self.steps],
            "final_top_k_indices": list(self.final_top_k_indices),
            "final_top_k_values": [float(v) for v in self.final_top_k_values],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "System2Trace":
        """Reconstruct from a dictionary."""
        return cls(
            steps_used=int(d["steps_used"]),
            converged=bool(d["converged"]),
            halt_reason=str(d["halt_reason"]),
            steps=[StepTrace.from_dict(s) for s in d.get("steps", [])],
            final_top_k_indices=[int(i) for i in d["final_top_k_indices"]],
            final_top_k_values=[float(v) for v in d["final_top_k_values"]],
        )


# ============================================================================
# SECTION 5 -- ReasoningTrace
# ============================================================================


@dataclass
class ReasoningTrace:
    """Complete reasoning audit trail for a single input item.

    Combines the routing decision, System 1 prediction, optional System 2
    deliberation trajectory, and arbitrary metadata into a single
    JSON-serializable record.

    Attributes:
        route:    Metacognitive routing decision.
        system1:  System 1 fast prediction summary.
        system2:  System 2 deliberation trajectory (None if System 1 only).
        metadata: Arbitrary key-value pairs (model version, timestamp, etc.).
    """

    route: RouteTrace
    system1: System1Trace
    system2: Optional[System2Trace] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert the full trace to a nested dictionary."""
        d: Dict[str, Any] = {
            "route": self.route.to_dict(),
            "system1": self.system1.to_dict(),
        }
        if self.system2 is not None:
            d["system2"] = self.system2.to_dict()
        else:
            d["system2"] = None
        if self.metadata:
            d["metadata"] = dict(self.metadata)
        else:
            d["metadata"] = {}
        return d

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serialize to a JSON string.

        Args:
            indent: JSON indentation level.  ``None`` for compact output.

        Returns:
            JSON string representation of the trace.
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReasoningTrace":
        """Reconstruct a ``ReasoningTrace`` from a nested dictionary."""
        s2 = d.get("system2")
        return cls(
            route=RouteTrace.from_dict(d["route"]),
            system1=System1Trace.from_dict(d["system1"]),
            system2=System2Trace.from_dict(s2) if s2 is not None else None,
            metadata=dict(d.get("metadata", {})),
        )

    @classmethod
    def from_json(cls, s: str) -> "ReasoningTrace":
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(s))

    # -- convenience properties ----------------------------------------------

    @property
    def used_system2(self) -> bool:
        """Whether System 2 was engaged for this item."""
        return self.route.used_system2

    @property
    def final_class(self) -> int:
        """Predicted class index from the final (active) system.

        If System 2 was used, returns the argmax from System 2's final
        top-k; otherwise returns System 1's top prediction.
        """
        if self.system2 is not None and len(self.system2.final_top_k_indices) > 0:
            return self.system2.final_top_k_indices[0]
        if len(self.system1.top_k_indices) > 0:
            return self.system1.top_k_indices[0]
        return -1

    @property
    def final_confidence(self) -> float:
        """Confidence from the final (active) system.

        Returns the top-1 value from System 2 if used, else from System 1.
        """
        if self.system2 is not None and len(self.system2.final_top_k_values) > 0:
            return self.system2.final_top_k_values[0]
        if len(self.system1.top_k_values) > 0:
            return self.system1.top_k_values[0]
        return 0.0

    @property
    def total_steps(self) -> int:
        """Total deliberation steps.

        Returns 0 if only System 1 was used; otherwise the number of
        System 2 steps executed.
        """
        if self.system2 is not None:
            return self.system2.steps_used
        return 0


# ============================================================================
# SECTION 6 -- BatchTrace
# ============================================================================


@dataclass
class BatchTrace:
    """Collection of ``ReasoningTrace`` objects for a full batch.

    Provides indexing, iteration, batch-level statistics, and
    JSON serialization.

    Attributes:
        traces:     List of per-item reasoning traces.
        batch_size: Number of items in the batch.
    """

    traces: List[ReasoningTrace]
    batch_size: int

    # -- container protocol --------------------------------------------------

    def __getitem__(self, idx: int) -> ReasoningTrace:
        """Index into the batch of traces."""
        return self.traces[idx]

    def __len__(self) -> int:
        """Number of traces in the batch."""
        return len(self.traces)

    def __iter__(self) -> Iterator[ReasoningTrace]:
        """Iterate over traces."""
        return iter(self.traces)

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert the full batch trace to a dictionary."""
        return {
            "traces": [t.to_dict() for t in self.traces],
            "batch_size": self.batch_size,
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BatchTrace":
        """Reconstruct from a dictionary."""
        traces = [ReasoningTrace.from_dict(t) for t in d["traces"]]
        return cls(
            traces=traces,
            batch_size=int(d.get("batch_size", len(traces))),
        )

    @classmethod
    def from_json(cls, s: str) -> "BatchTrace":
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(s))

    # -- batch-level statistics ----------------------------------------------

    @property
    def system2_fraction(self) -> float:
        """Fraction of items that were routed to System 2."""
        if len(self.traces) == 0:
            return 0.0
        count = sum(1 for t in self.traces if t.used_system2)
        return count / len(self.traces)

    @property
    def avg_steps(self) -> float:
        """Average number of System 2 steps across all items.

        Items that did not use System 2 contribute 0 steps.
        """
        if len(self.traces) == 0:
            return 0.0
        total = sum(t.total_steps for t in self.traces)
        return total / len(self.traces)

    @property
    def convergence_rate(self) -> float:
        """Fraction of System-2 items that converged before budget.

        Returns 0.0 if no items used System 2.
        """
        s2_traces = [t for t in self.traces if t.system2 is not None]
        if len(s2_traces) == 0:
            return 0.0
        converged = sum(1 for t in s2_traces if t.system2.converged)  # type: ignore[union-attr]
        return converged / len(s2_traces)

    def summary_dict(self) -> Dict[str, Any]:
        """Return a concise summary of batch statistics.

        Returns:
            Dictionary with keys: batch_size, system2_fraction, avg_steps,
            convergence_rate, total_system2_items.
        """
        s2_count = sum(1 for t in self.traces if t.used_system2)
        return {
            "batch_size": self.batch_size,
            "num_traces": len(self.traces),
            "system2_fraction": round(self.system2_fraction, 4),
            "avg_steps": round(self.avg_steps, 4),
            "convergence_rate": round(self.convergence_rate, 4),
            "total_system2_items": s2_count,
        }


# ============================================================================
# SECTION 7 -- TraceBuilder
# ============================================================================


class TraceBuilder:
    """Conditional trace builder that becomes a no-op when disabled.

    In production, tracing can be expensive.  ``TraceBuilder`` wraps all
    trace construction so that when ``enabled=False``, every ``build_*``
    method returns ``None`` immediately with negligible overhead.

    Args:
        enabled:   Whether to actually build traces.
        top_k:     Default k for top-k extraction.
        full_mode: If True, record per-step top-k inside ``StepTrace``
                   (increases trace size but improves debuggability).
    """

    def __init__(
        self,
        enabled: bool = True,
        top_k: int = 5,
        full_mode: bool = False,
    ) -> None:
        self.enabled = enabled
        self.top_k = top_k
        self.full_mode = full_mode

    # -- individual builders -------------------------------------------------

    def build_route_trace(
        self,
        *,
        used_system2: bool,
        route_score: float,
        threshold: float,
        conf_raw: float,
        conf_calibrated: float,
        entropy: float,
        margin: float,
        novelty: Optional[float] = None,
        anomaly: Optional[float] = None,
        ignition: Optional[float] = None,
        steps_budget: int = 0,
    ) -> Optional[RouteTrace]:
        """Build a ``RouteTrace`` if tracing is enabled.

        Returns:
            ``RouteTrace`` or ``None`` if disabled.
        """
        if not self.enabled:
            return None
        return RouteTrace(
            used_system2=used_system2,
            route_score=route_score,
            threshold=threshold,
            conf_raw=conf_raw,
            conf_calibrated=conf_calibrated,
            entropy=entropy,
            margin=margin,
            novelty=novelty,
            anomaly=anomaly,
            ignition=ignition,
            steps_budget=steps_budget,
        )

    def build_s1_trace(
        self,
        *,
        top_k_indices: List[int],
        top_k_values: List[float],
        top_k: Optional[int] = None,
    ) -> Optional[System1Trace]:
        """Build a ``System1Trace`` if tracing is enabled.

        Args:
            top_k_indices: Class indices of top-k predictions.
            top_k_values:  Corresponding values.
            top_k:         Override for k (defaults to ``self.top_k``).

        Returns:
            ``System1Trace`` or ``None`` if disabled.
        """
        if not self.enabled:
            return None
        k = top_k if top_k is not None else self.top_k
        return System1Trace(
            top_k_indices=list(top_k_indices[:k]),
            top_k_values=[float(v) for v in top_k_values[:k]],
            top_k=k,
        )

    def build_step_trace(
        self,
        *,
        step_idx: int,
        conf_k: float,
        delta_kl: float,
        delta_max: float,
        argmax_k: int,
        halt_check_passed: bool,
        top_k_indices: Optional[List[int]] = None,
        top_k_values: Optional[List[float]] = None,
    ) -> Optional[StepTrace]:
        """Build a ``StepTrace`` for one System 2 iteration.

        In non-full mode, top-k data is omitted from step traces to save
        space.

        Returns:
            ``StepTrace`` or ``None`` if disabled.
        """
        if not self.enabled:
            return None
        # In non-full mode, strip per-step top-k to reduce trace size
        if not self.full_mode:
            top_k_indices = None
            top_k_values = None
        return StepTrace(
            step_idx=step_idx,
            conf_k=conf_k,
            delta_kl=delta_kl,
            delta_max=delta_max,
            argmax_k=argmax_k,
            halt_check_passed=halt_check_passed,
            top_k_indices=(
                [int(i) for i in top_k_indices] if top_k_indices is not None else None
            ),
            top_k_values=(
                [float(v) for v in top_k_values] if top_k_values is not None else None
            ),
        )

    def build_s2_trace(
        self,
        *,
        steps_used: int,
        converged: bool,
        halt_reason: str,
        steps: List[StepTrace],
        final_top_k_indices: List[int],
        final_top_k_values: List[float],
    ) -> Optional[System2Trace]:
        """Build a ``System2Trace`` from collected step traces.

        Returns:
            ``System2Trace`` or ``None`` if disabled.
        """
        if not self.enabled:
            return None
        return System2Trace(
            steps_used=steps_used,
            converged=converged,
            halt_reason=halt_reason,
            steps=list(steps),
            final_top_k_indices=[int(i) for i in final_top_k_indices],
            final_top_k_values=[float(v) for v in final_top_k_values],
        )

    def build_trace(
        self,
        *,
        route: Optional[RouteTrace],
        system1: Optional[System1Trace],
        system2: Optional[System2Trace] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[ReasoningTrace]:
        """Assemble a full ``ReasoningTrace`` from sub-traces.

        If tracing is disabled or required sub-traces are ``None``, returns
        ``None``.

        Returns:
            ``ReasoningTrace`` or ``None`` if disabled or inputs are None.
        """
        if not self.enabled:
            return None
        if route is None or system1 is None:
            return None
        return ReasoningTrace(
            route=route,
            system1=system1,
            system2=system2,
            metadata=metadata if metadata is not None else {},
        )

    def build_batch_trace(
        self,
        traces: List[Optional[ReasoningTrace]],
    ) -> Optional[BatchTrace]:
        """Build a ``BatchTrace`` from a list of optional traces.

        Filters out ``None`` entries (which occur when tracing is disabled
        for some items or when building failed).

        Returns:
            ``BatchTrace`` or ``None`` if disabled or no valid traces.
        """
        if not self.enabled:
            return None
        valid = [t for t in traces if t is not None]
        if len(valid) == 0:
            return None
        return BatchTrace(
            traces=valid,
            batch_size=len(traces),
        )


# ============================================================================
# SECTION 8 -- TraceDiff and compare_traces
# ============================================================================


@dataclass
class TraceDiff:
    """Structured diff between two ``ReasoningTrace`` objects.

    Useful for regression testing -- compare traces produced by two model
    checkpoints on the same input to detect behavioral changes.

    Attributes:
        same_routing:    Whether both traces made the same routing decision.
        same_halt_reason: Whether System 2 halt reasons match (True if
                         neither used System 2).
        logit_max_diff:  Absolute difference in final top-1 values.
        confidence_diff: Absolute difference in calibrated confidences.
        steps_diff:      Absolute difference in System 2 steps used.
        details:         Additional diff information.
    """

    same_routing: bool
    same_halt_reason: bool
    logit_max_diff: float
    confidence_diff: float
    steps_diff: int
    details: Dict[str, Any] = field(default_factory=dict)

    def is_identical(self, atol: float = 1e-5) -> bool:
        """Check whether the two traces are functionally identical.

        Args:
            atol: Absolute tolerance for floating-point comparisons.

        Returns:
            True if routing, halt reason, and numerical values all match
            within tolerance.
        """
        return (
            self.same_routing
            and self.same_halt_reason
            and self.logit_max_diff <= atol
            and self.confidence_diff <= atol
            and self.steps_diff == 0
        )


def compare_traces(
    t1: ReasoningTrace,
    t2: ReasoningTrace,
    atol: float = 1e-5,
) -> TraceDiff:
    """Compare two ``ReasoningTrace`` objects and return a ``TraceDiff``.

    Args:
        t1:   First trace (reference).
        t2:   Second trace (candidate).
        atol: Absolute tolerance for numerical comparisons.

    Returns:
        ``TraceDiff`` summarizing the differences.
    """
    same_routing = t1.route.used_system2 == t2.route.used_system2

    # Halt reason comparison
    hr1 = t1.system2.halt_reason if t1.system2 is not None else None
    hr2 = t2.system2.halt_reason if t2.system2 is not None else None
    same_halt_reason = hr1 == hr2

    # Final top-1 value diff
    val1 = t1.final_confidence
    val2 = t2.final_confidence
    logit_max_diff = abs(val1 - val2)

    # Calibrated confidence diff
    confidence_diff = abs(t1.route.conf_calibrated - t2.route.conf_calibrated)

    # Steps diff
    steps1 = t1.total_steps
    steps2 = t2.total_steps
    steps_diff = abs(steps1 - steps2)

    # Extra details
    details: Dict[str, Any] = {
        "class_match": t1.final_class == t2.final_class,
        "route_score_diff": abs(t1.route.route_score - t2.route.route_score),
        "entropy_diff": abs(t1.route.entropy - t2.route.entropy),
    }

    return TraceDiff(
        same_routing=same_routing,
        same_halt_reason=same_halt_reason,
        logit_max_diff=logit_max_diff,
        confidence_diff=confidence_diff,
        steps_diff=steps_diff,
        details=details,
    )


# ============================================================================
# SECTION 9 -- TraceLogger
# ============================================================================


class TraceLogger:
    """JSONL logger for reasoning traces.

    Supports buffered writes to a file and/or a stream (e.g. ``sys.stdout``).
    Traces are written one-per-line in JSONL format for easy streaming
    ingestion.

    Args:
        output_path: Optional file path for JSONL output.
        max_traces:  Maximum number of traces to buffer before auto-flush.
        stream:      Optional writable stream (e.g. ``sys.stdout``) for
                     real-time output.
    """

    def __init__(
        self,
        output_path: Optional[str] = None,
        max_traces: int = 1000,
        stream: Optional[Any] = None,
    ) -> None:
        self.output_path = output_path
        self.max_traces = max_traces
        self.stream = stream
        self._buffer: List[Dict[str, Any]] = []
        self._total_logged: int = 0
        self._file_handle: Optional[Any] = None
        if output_path is not None:
            parent = os.path.dirname(output_path)
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)
            self._file_handle = open(output_path, "a", encoding="utf-8")

    def log(self, trace: ReasoningTrace) -> None:
        """Log a single ``ReasoningTrace``.

        Writes immediately to the stream (if set), and buffers for file
        output.  Auto-flushes the file buffer when ``max_traces`` is
        reached.

        Args:
            trace: The trace to log.
        """
        d = trace.to_dict()
        self._buffer.append(d)
        self._total_logged += 1
        if self.stream is not None:
            line = json.dumps(d)
            self.stream.write(line + "\n")
        if len(self._buffer) >= self.max_traces:
            self.flush()

    def log_batch(self, batch_trace: BatchTrace) -> None:
        """Log all traces in a ``BatchTrace``.

        Args:
            batch_trace: Batch of traces to log.
        """
        for trace in batch_trace.traces:
            self.log(trace)

    def flush(self) -> None:
        """Flush the buffer to the output file.

        Does nothing if no output_path was configured.
        """
        if self._file_handle is not None and self._buffer:
            for d in self._buffer:
                line = json.dumps(d)
                self._file_handle.write(line + "\n")
            self._file_handle.flush()
        self._buffer.clear()

    def close(self) -> None:
        """Flush remaining buffer and close the file handle."""
        self.flush()
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None

    def summary(self) -> Dict[str, Any]:
        """Return a summary of logging activity.

        Returns:
            Dictionary with total_logged, buffer_size, output_path.
        """
        return {
            "total_logged": self._total_logged,
            "buffer_size": len(self._buffer),
            "output_path": self.output_path,
        }

    def __del__(self) -> None:
        """Best-effort cleanup on garbage collection."""
        try:
            self.close()
        except Exception:
            pass


# ============================================================================
# SECTION 10 -- extract_top_k utility
# ============================================================================


def extract_top_k(
    logits: "Tensor",
    k: int,
) -> Tuple[List[int], List[float]]:
    """Extract top-k indices and values from a 1-D logit tensor.

    Uses ``torch.topk`` for efficient extraction.  Handles edge cases:

    * ``k <= 0`` returns empty lists.
    * ``k > num_classes`` is clamped to ``num_classes``.

    Args:
        logits: 1-D tensor of shape ``(num_classes,)``.
        k:      Number of top entries to extract.

    Returns:
        Tuple of (indices as List[int], values as List[float]).

    Raises:
        RuntimeError: If torch is not available.
    """
    if not _HAS_TORCH:
        raise RuntimeError(
            "extract_top_k requires torch. Install with: pip install torch"
        )
    if k <= 0:
        return [], []
    num_classes = logits.shape[-1]
    k_clamped = min(k, num_classes)
    if k_clamped == 0:
        return [], []
    values, indices = torch.topk(logits.detach().float(), k_clamped, dim=-1)
    return indices.tolist(), [float(v) for v in values.tolist()]


# ============================================================================
# SECTION 11 -- Helper factories
# ============================================================================


def make_s1_only_trace(
    *,
    conf_raw: float,
    conf_calibrated: float,
    entropy: float,
    margin: float,
    route_score: float,
    threshold: float,
    top_k_indices: List[int],
    top_k_values: List[float],
    top_k: int = 5,
    novelty: Optional[float] = None,
    anomaly: Optional[float] = None,
    ignition: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ReasoningTrace:
    """Create a ``ReasoningTrace`` for an item handled by System 1 only.

    Convenience factory that builds all sub-traces in one call.

    Args:
        conf_raw:        Raw softmax confidence.
        conf_calibrated: Calibrated confidence.
        entropy:         Softmax entropy.
        margin:          Top-1 vs top-2 probability gap.
        route_score:     Composite routing score.
        threshold:       Routing threshold used.
        top_k_indices:   Top-k class indices from System 1.
        top_k_values:    Top-k values from System 1.
        top_k:           k used for extraction.
        novelty:         Optional novelty score.
        anomaly:         Optional anomaly score.
        ignition:        Optional ignition strength.
        metadata:        Optional metadata dict.

    Returns:
        A ``ReasoningTrace`` with ``system2=None``.
    """
    route = RouteTrace(
        used_system2=False,
        route_score=route_score,
        threshold=threshold,
        conf_raw=conf_raw,
        conf_calibrated=conf_calibrated,
        entropy=entropy,
        margin=margin,
        novelty=novelty,
        anomaly=anomaly,
        ignition=ignition,
        steps_budget=0,
    )
    s1 = System1Trace(
        top_k_indices=list(top_k_indices),
        top_k_values=[float(v) for v in top_k_values],
        top_k=top_k,
    )
    return ReasoningTrace(
        route=route,
        system1=s1,
        system2=None,
        metadata=metadata if metadata is not None else {},
    )


def make_s2_trace(
    *,
    conf_raw: float,
    conf_calibrated: float,
    entropy: float,
    margin: float,
    route_score: float,
    threshold: float,
    s1_top_k_indices: List[int],
    s1_top_k_values: List[float],
    s1_top_k: int = 5,
    steps_used: int,
    converged: bool,
    halt_reason: str,
    step_traces: List[StepTrace],
    final_top_k_indices: List[int],
    final_top_k_values: List[float],
    steps_budget: int = 10,
    novelty: Optional[float] = None,
    anomaly: Optional[float] = None,
    ignition: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ReasoningTrace:
    """Create a ``ReasoningTrace`` for an item that used System 2.

    Convenience factory that builds all sub-traces in one call.

    Args:
        conf_raw:             Raw softmax confidence from System 1.
        conf_calibrated:      Calibrated confidence from System 1.
        entropy:              Softmax entropy from System 1.
        margin:               Top-1 vs top-2 probability gap.
        route_score:          Composite routing score.
        threshold:            Routing threshold used.
        s1_top_k_indices:     System 1 top-k class indices.
        s1_top_k_values:      System 1 top-k values.
        s1_top_k:             k used for System 1 extraction.
        steps_used:           Number of System 2 GRU steps executed.
        converged:            Whether halting criterion was met.
        halt_reason:          Reason for halting.
        step_traces:          List of per-step ``StepTrace`` objects.
        final_top_k_indices:  System 2 final top-k class indices.
        final_top_k_values:   System 2 final top-k values.
        steps_budget:         Maximum steps allocated.
        novelty:              Optional novelty score.
        anomaly:              Optional anomaly score.
        ignition:             Optional ignition strength.
        metadata:             Optional metadata dict.

    Returns:
        A ``ReasoningTrace`` with ``system2`` populated.
    """
    route = RouteTrace(
        used_system2=True,
        route_score=route_score,
        threshold=threshold,
        conf_raw=conf_raw,
        conf_calibrated=conf_calibrated,
        entropy=entropy,
        margin=margin,
        novelty=novelty,
        anomaly=anomaly,
        ignition=ignition,
        steps_budget=steps_budget,
    )
    s1 = System1Trace(
        top_k_indices=list(s1_top_k_indices),
        top_k_values=[float(v) for v in s1_top_k_values],
        top_k=s1_top_k,
    )
    s2 = System2Trace(
        steps_used=steps_used,
        converged=converged,
        halt_reason=halt_reason,
        steps=list(step_traces),
        final_top_k_indices=[int(i) for i in final_top_k_indices],
        final_top_k_values=[float(v) for v in final_top_k_values],
    )
    return ReasoningTrace(
        route=route,
        system1=s1,
        system2=s2,
        metadata=metadata if metadata is not None else {},
    )


# ============================================================================
# SECTION 12 -- filter_traces, trace_statistics, merge_batch_traces
# ============================================================================


def filter_traces(
    traces: Sequence[ReasoningTrace],
    *,
    system2_only: bool = False,
    system1_only: bool = False,
) -> List[ReasoningTrace]:
    """Filter a list of traces by routing decision.

    Args:
        traces:       Sequence of traces to filter.
        system2_only: If True, keep only traces that used System 2.
        system1_only: If True, keep only traces that used System 1 only.

    Returns:
        Filtered list.  If both flags are False, returns all traces.
        If both flags are True, returns an empty list (contradictory).
    """
    if system2_only and system1_only:
        return []
    if system2_only:
        return [t for t in traces if t.used_system2]
    if system1_only:
        return [t for t in traces if not t.used_system2]
    return list(traces)


def trace_statistics(traces: Sequence[ReasoningTrace]) -> Dict[str, Any]:
    """Compute aggregate statistics over a collection of traces.

    Args:
        traces: Sequence of reasoning traces.

    Returns:
        Dictionary with keys:
            total, system1_count, system2_count, system2_fraction,
            avg_confidence, avg_entropy, avg_margin, avg_steps,
            convergence_rate, avg_route_score.
    """
    n = len(traces)
    if n == 0:
        return {
            "total": 0,
            "system1_count": 0,
            "system2_count": 0,
            "system2_fraction": 0.0,
            "avg_confidence": 0.0,
            "avg_entropy": 0.0,
            "avg_margin": 0.0,
            "avg_steps": 0.0,
            "convergence_rate": 0.0,
            "avg_route_score": 0.0,
        }

    s2_traces = [t for t in traces if t.used_system2]
    s1_count = n - len(s2_traces)
    s2_count = len(s2_traces)

    avg_conf = sum(t.route.conf_calibrated for t in traces) / n
    avg_ent = sum(t.route.entropy for t in traces) / n
    avg_mar = sum(t.route.margin for t in traces) / n
    avg_steps = sum(t.total_steps for t in traces) / n
    avg_rs = sum(t.route.route_score for t in traces) / n

    conv_rate = 0.0
    if s2_count > 0:
        converged = sum(
            1 for t in s2_traces
            if t.system2 is not None and t.system2.converged
        )
        conv_rate = converged / s2_count

    return {
        "total": n,
        "system1_count": s1_count,
        "system2_count": s2_count,
        "system2_fraction": round(s2_count / n, 4),
        "avg_confidence": round(avg_conf, 4),
        "avg_entropy": round(avg_ent, 4),
        "avg_margin": round(avg_mar, 4),
        "avg_steps": round(avg_steps, 4),
        "convergence_rate": round(conv_rate, 4),
        "avg_route_score": round(avg_rs, 4),
    }


def merge_batch_traces(batches: Sequence[BatchTrace]) -> BatchTrace:
    """Merge multiple ``BatchTrace`` objects into a single ``BatchTrace``.

    Args:
        batches: Sequence of batch traces to merge.

    Returns:
        A new ``BatchTrace`` containing all traces and the sum of all
        batch sizes.
    """
    all_traces: List[ReasoningTrace] = []
    total_batch_size = 0
    for bt in batches:
        all_traces.extend(bt.traces)
        total_batch_size += bt.batch_size
    return BatchTrace(traces=all_traces, batch_size=total_batch_size)


# ============================================================================
# SELF-TESTS
# ============================================================================


def _run_self_tests() -> None:
    """Run 12 self-tests and print results."""
    passed = 0
    total = 12

    # -- helpers to build test data ------------------------------------------

    def _make_route(used_s2: bool = False) -> RouteTrace:
        return RouteTrace(
            used_system2=used_s2,
            route_score=0.65,
            threshold=0.7,
            conf_raw=0.82,
            conf_calibrated=0.78,
            entropy=1.23,
            margin=0.35,
            novelty=0.1,
            anomaly=0.05,
            ignition=0.9,
            steps_budget=10 if used_s2 else 0,
        )

    def _make_s1() -> System1Trace:
        return System1Trace(
            top_k_indices=[3, 7, 1, 0, 5],
            top_k_values=[0.45, 0.20, 0.12, 0.08, 0.05],
            top_k=5,
        )

    def _make_step(idx: int, halt: bool = False) -> StepTrace:
        return StepTrace(
            step_idx=idx,
            conf_k=0.5 + idx * 0.1,
            delta_kl=0.05 / (idx + 1),
            delta_max=0.02 / (idx + 1),
            argmax_k=3,
            halt_check_passed=halt,
            top_k_indices=[3, 7, 1],
            top_k_values=[0.6 + idx * 0.05, 0.2, 0.1],
        )

    def _make_s2(steps: int = 3, converged: bool = True) -> System2Trace:
        step_list = [_make_step(i, halt=(i == steps - 1 and converged)) for i in range(steps)]
        return System2Trace(
            steps_used=steps,
            converged=converged,
            halt_reason="converged" if converged else "budget_exhausted",
            steps=step_list,
            final_top_k_indices=[3, 7, 1, 0, 5],
            final_top_k_values=[0.75, 0.10, 0.06, 0.04, 0.03],
        )

    def _make_trace(used_s2: bool = False) -> ReasoningTrace:
        route = _make_route(used_s2)
        s1 = _make_s1()
        s2 = _make_s2() if used_s2 else None
        return ReasoningTrace(
            route=route,
            system1=s1,
            system2=s2,
            metadata={"model_version": "0.1.0", "item_id": 42},
        )

    # -- Test 1: Manual trace -> to_dict -> to_json -> valid JSON ------------
    try:
        t = _make_trace(used_s2=True)
        d = t.to_dict()
        j = t.to_json(indent=2)
        parsed = json.loads(j)
        assert isinstance(parsed, dict), "JSON parse should produce dict"
        assert "route" in parsed, "JSON should contain route"
        assert "system1" in parsed, "JSON should contain system1"
        assert "system2" in parsed, "JSON should contain system2"
        assert parsed["system2"] is not None, "system2 should not be None"
        passed += 1
        print("  [PASS] Test  1: Manual trace -> to_dict -> to_json -> valid JSON")
    except Exception as e:
        print(f"  [FAIL] Test  1: Manual trace -> to_dict -> to_json -> valid JSON: {e}")

    # -- Test 2: Roundtrip: to_dict -> from_dict -> compare ------------------
    try:
        t_orig = _make_trace(used_s2=True)
        d = t_orig.to_dict()
        t_rt = ReasoningTrace.from_dict(d)
        assert t_rt.route.used_system2 == t_orig.route.used_system2
        assert t_rt.route.route_score == t_orig.route.route_score
        assert t_rt.route.novelty == t_orig.route.novelty
        assert t_rt.system1.top_k_indices == t_orig.system1.top_k_indices
        assert t_rt.system1.top_k_values == t_orig.system1.top_k_values
        assert t_rt.system2 is not None
        assert t_rt.system2.steps_used == t_orig.system2.steps_used  # type: ignore
        assert t_rt.system2.converged == t_orig.system2.converged  # type: ignore
        assert t_rt.system2.halt_reason == t_orig.system2.halt_reason  # type: ignore
        assert len(t_rt.system2.steps) == len(t_orig.system2.steps)  # type: ignore
        assert t_rt.metadata == t_orig.metadata
        # Also test JSON roundtrip
        t_json_rt = ReasoningTrace.from_json(t_orig.to_json())
        assert t_json_rt.final_class == t_orig.final_class
        assert abs(t_json_rt.final_confidence - t_orig.final_confidence) < 1e-7
        passed += 1
        print("  [PASS] Test  2: Roundtrip: to_dict -> from_dict -> compare")
    except Exception as e:
        print(f"  [FAIL] Test  2: Roundtrip: to_dict -> from_dict -> compare: {e}")

    # -- Test 3: BatchTrace indexing and serialization -----------------------
    try:
        traces = [_make_trace(i % 2 == 0) for i in range(6)]
        bt = BatchTrace(traces=traces, batch_size=6)
        assert len(bt) == 6
        assert bt[0].route.used_system2 is True
        assert bt[1].route.used_system2 is False
        count = 0
        for t in bt:
            count += 1
        assert count == 6
        # Serialization roundtrip
        bt_d = bt.to_dict()
        bt_j = bt.to_json()
        bt_rt = BatchTrace.from_json(bt_j)
        assert len(bt_rt) == len(bt)
        assert bt_rt.batch_size == bt.batch_size
        assert bt_rt[2].final_class == bt[2].final_class
        passed += 1
        print("  [PASS] Test  3: BatchTrace indexing and serialization")
    except Exception as e:
        print(f"  [FAIL] Test  3: BatchTrace indexing and serialization: {e}")

    # -- Test 4: TraceBuilder enabled=True produces traces -------------------
    try:
        builder = TraceBuilder(enabled=True, top_k=3, full_mode=True)
        rt = builder.build_route_trace(
            used_system2=False,
            route_score=0.4,
            threshold=0.7,
            conf_raw=0.9,
            conf_calibrated=0.88,
            entropy=0.5,
            margin=0.6,
        )
        assert rt is not None, "Route trace should not be None when enabled"
        s1t = builder.build_s1_trace(
            top_k_indices=[3, 7, 1, 0, 5],
            top_k_values=[0.45, 0.20, 0.12, 0.08, 0.05],
        )
        assert s1t is not None
        assert len(s1t.top_k_indices) == 3, "Should truncate to top_k=3"
        trace = builder.build_trace(route=rt, system1=s1t)
        assert trace is not None
        assert trace.system2 is None
        assert trace.used_system2 is False
        passed += 1
        print("  [PASS] Test  4: TraceBuilder enabled=True produces traces")
    except Exception as e:
        print(f"  [FAIL] Test  4: TraceBuilder enabled=True produces traces: {e}")

    # -- Test 5: TraceBuilder enabled=False returns None ---------------------
    try:
        builder_off = TraceBuilder(enabled=False)
        assert builder_off.build_route_trace(
            used_system2=False, route_score=0.4, threshold=0.7,
            conf_raw=0.9, conf_calibrated=0.88, entropy=0.5, margin=0.6,
        ) is None
        assert builder_off.build_s1_trace(
            top_k_indices=[1], top_k_values=[0.9],
        ) is None
        assert builder_off.build_step_trace(
            step_idx=0, conf_k=0.5, delta_kl=0.01,
            delta_max=0.02, argmax_k=3, halt_check_passed=False,
        ) is None
        assert builder_off.build_s2_trace(
            steps_used=1, converged=True, halt_reason="converged",
            steps=[], final_top_k_indices=[3], final_top_k_values=[0.9],
        ) is None
        assert builder_off.build_trace(route=None, system1=None) is None
        assert builder_off.build_batch_trace([]) is None
        passed += 1
        print("  [PASS] Test  5: TraceBuilder enabled=False returns None")
    except Exception as e:
        print(f"  [FAIL] Test  5: TraceBuilder enabled=False returns None: {e}")

    # -- Test 6: StepTrace list building -------------------------------------
    try:
        steps = [_make_step(i, halt=(i == 4)) for i in range(5)]
        assert len(steps) == 5
        assert steps[0].step_idx == 0
        assert steps[4].halt_check_passed is True
        assert steps[3].halt_check_passed is False
        # Roundtrip each step
        for s in steps:
            s_rt = StepTrace.from_dict(s.to_dict())
            assert s_rt.step_idx == s.step_idx
            assert s_rt.conf_k == s.conf_k
            assert s_rt.argmax_k == s.argmax_k
            assert s_rt.halt_check_passed == s.halt_check_passed
        passed += 1
        print("  [PASS] Test  6: StepTrace list building")
    except Exception as e:
        print(f"  [FAIL] Test  6: StepTrace list building: {e}")

    # -- Test 7: TraceDiff identical -> diffs zero ---------------------------
    try:
        t_a = _make_trace(used_s2=True)
        t_b = _make_trace(used_s2=True)
        diff = compare_traces(t_a, t_b)
        assert diff.same_routing is True
        assert diff.same_halt_reason is True
        assert diff.logit_max_diff < 1e-7
        assert diff.confidence_diff < 1e-7
        assert diff.steps_diff == 0
        assert diff.is_identical(atol=1e-5) is True
        passed += 1
        print("  [PASS] Test  7: TraceDiff identical -> diffs zero")
    except Exception as e:
        print(f"  [FAIL] Test  7: TraceDiff identical -> diffs zero: {e}")

    # -- Test 8: TraceDiff different -> diffs nonzero ------------------------
    try:
        t_c = _make_trace(used_s2=False)
        t_d = _make_trace(used_s2=True)
        diff2 = compare_traces(t_c, t_d)
        assert diff2.same_routing is False, "Routing should differ"
        assert diff2.same_halt_reason is False, "Halt reasons should differ"
        assert diff2.is_identical() is False
        # Also check numerical diffs when confidence differs
        t_e = _make_trace(used_s2=False)
        t_f = _make_trace(used_s2=False)
        t_f.route.conf_calibrated = 0.50  # modify to create a diff
        t_f.system1.top_k_values[0] = 0.99
        diff3 = compare_traces(t_e, t_f)
        assert diff3.confidence_diff > 0.1, "Confidence diff should be > 0.1"
        assert diff3.logit_max_diff > 0.4, "Logit max diff should be > 0.4"
        assert diff3.is_identical() is False
        passed += 1
        print("  [PASS] Test  8: TraceDiff different -> diffs nonzero")
    except Exception as e:
        print(f"  [FAIL] Test  8: TraceDiff different -> diffs nonzero: {e}")

    # -- Test 9: TraceLogger: log, flush, file -------------------------------
    try:
        tmp_path = os.path.join(tempfile.gettempdir(), "_trace_test.jsonl")
        # Clean up any previous run
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        logger = TraceLogger(output_path=tmp_path, max_traces=100)
        for i in range(5):
            logger.log(_make_trace(i % 2 == 0))
        summary = logger.summary()
        assert summary["total_logged"] == 5
        assert summary["buffer_size"] == 5
        logger.flush()
        assert logger.summary()["buffer_size"] == 0
        logger.close()
        # Verify file contents
        with open(tmp_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 5, f"Expected 5 lines, got {len(lines)}"
        for line in lines:
            parsed = json.loads(line.strip())
            assert "route" in parsed
        # Clean up
        os.remove(tmp_path)
        # Test stream logging
        buf = io.StringIO()
        logger2 = TraceLogger(stream=buf)
        logger2.log(_make_trace())
        logger2.close()
        assert len(buf.getvalue().strip()) > 0
        passed += 1
        print("  [PASS] Test  9: TraceLogger: log, flush, file")
    except Exception as e:
        print(f"  [FAIL] Test  9: TraceLogger: log, flush, file: {e}")

    # -- Test 10: extract_top_k various k -----------------------------------
    try:
        if not _HAS_TORCH:
            print("  [SKIP] Test 10: extract_top_k (torch not available)")
            # Count as passed since torch is optional
            passed += 1
        else:
            logits = torch.tensor([0.1, 0.5, 0.3, 0.9, 0.2, 0.7, 0.4])
            # k=3
            idx, vals = extract_top_k(logits, k=3)
            assert len(idx) == 3
            assert len(vals) == 3
            assert idx[0] == 3, f"Top index should be 3 (0.9), got {idx[0]}"
            assert abs(vals[0] - 0.9) < 1e-5
            # k=0
            idx0, vals0 = extract_top_k(logits, k=0)
            assert idx0 == []
            assert vals0 == []
            # k > num_classes
            idx_big, vals_big = extract_top_k(logits, k=100)
            assert len(idx_big) == 7
            assert len(vals_big) == 7
            # k=1
            idx1, vals1 = extract_top_k(logits, k=1)
            assert len(idx1) == 1
            assert idx1[0] == 3
            passed += 1
            print("  [PASS] Test 10: extract_top_k various k")
    except Exception as e:
        print(f"  [FAIL] Test 10: extract_top_k various k: {e}")

    # -- Test 11: Edge cases: empty S2, single step --------------------------
    try:
        # S1-only trace
        t_s1 = make_s1_only_trace(
            conf_raw=0.95,
            conf_calibrated=0.92,
            entropy=0.3,
            margin=0.7,
            route_score=0.2,
            threshold=0.7,
            top_k_indices=[5, 2, 8],
            top_k_values=[0.6, 0.2, 0.1],
            top_k=3,
        )
        assert t_s1.system2 is None
        assert t_s1.used_system2 is False
        assert t_s1.final_class == 5
        assert t_s1.total_steps == 0
        # Roundtrip
        t_s1_rt = ReasoningTrace.from_dict(t_s1.to_dict())
        assert t_s1_rt.system2 is None
        # S2 with single step
        t_s2_single = make_s2_trace(
            conf_raw=0.4,
            conf_calibrated=0.35,
            entropy=2.1,
            margin=0.05,
            route_score=0.85,
            threshold=0.7,
            s1_top_k_indices=[1],
            s1_top_k_values=[0.3],
            s1_top_k=1,
            steps_used=1,
            converged=True,
            halt_reason="converged",
            step_traces=[_make_step(0, halt=True)],
            final_top_k_indices=[3],
            final_top_k_values=[0.8],
            steps_budget=5,
        )
        assert t_s2_single.used_system2 is True
        assert t_s2_single.total_steps == 1
        assert t_s2_single.final_class == 3
        # S2 with empty steps list (edge case: 0 steps recorded)
        t_empty_s2 = make_s2_trace(
            conf_raw=0.4,
            conf_calibrated=0.35,
            entropy=2.1,
            margin=0.05,
            route_score=0.85,
            threshold=0.7,
            s1_top_k_indices=[1],
            s1_top_k_values=[0.3],
            s1_top_k=1,
            steps_used=0,
            converged=False,
            halt_reason="budget_exhausted",
            step_traces=[],
            final_top_k_indices=[1],
            final_top_k_values=[0.3],
            steps_budget=5,
        )
        assert t_empty_s2.system2 is not None
        assert len(t_empty_s2.system2.steps) == 0
        # filter_traces
        mixed = [t_s1, t_s2_single, t_empty_s2]
        s2_only = filter_traces(mixed, system2_only=True)
        assert len(s2_only) == 2
        s1_only = filter_traces(mixed, system1_only=True)
        assert len(s1_only) == 1
        both_flags = filter_traces(mixed, system2_only=True, system1_only=True)
        assert len(both_flags) == 0
        no_filter = filter_traces(mixed)
        assert len(no_filter) == 3
        # trace_statistics
        stats = trace_statistics(mixed)
        assert stats["total"] == 3
        assert stats["system1_count"] == 1
        assert stats["system2_count"] == 2
        # Empty traces statistics
        empty_stats = trace_statistics([])
        assert empty_stats["total"] == 0
        passed += 1
        print("  [PASS] Test 11: Edge cases: empty S2, single step")
    except Exception as e:
        print(f"  [FAIL] Test 11: Edge cases: empty S2, single step: {e}")

    # -- Test 12: build_batch_trace end-to-end -------------------------------
    try:
        builder = TraceBuilder(enabled=True, top_k=5, full_mode=True)
        all_traces: List[Optional[ReasoningTrace]] = []

        for i in range(8):
            use_s2 = i % 3 == 0
            rt = builder.build_route_trace(
                used_system2=use_s2,
                route_score=0.3 + i * 0.05,
                threshold=0.7,
                conf_raw=0.9 - i * 0.05,
                conf_calibrated=0.85 - i * 0.05,
                entropy=0.5 + i * 0.1,
                margin=0.6 - i * 0.05,
                novelty=0.1 * i,
                steps_budget=10 if use_s2 else 0,
            )
            s1t = builder.build_s1_trace(
                top_k_indices=[i, (i + 1) % 10, (i + 2) % 10],
                top_k_values=[0.5, 0.3, 0.1],
                top_k=3,
            )
            s2t = None
            if use_s2:
                step_traces: List[StepTrace] = []
                for si in range(3):
                    st = builder.build_step_trace(
                        step_idx=si,
                        conf_k=0.5 + si * 0.1,
                        delta_kl=0.05 / (si + 1),
                        delta_max=0.02 / (si + 1),
                        argmax_k=i,
                        halt_check_passed=(si == 2),
                    )
                    if st is not None:
                        step_traces.append(st)
                s2t = builder.build_s2_trace(
                    steps_used=3,
                    converged=True,
                    halt_reason="converged",
                    steps=step_traces,
                    final_top_k_indices=[i, (i + 1) % 10],
                    final_top_k_values=[0.7, 0.2],
                )
            trace = builder.build_trace(
                route=rt,
                system1=s1t,
                system2=s2t,
                metadata={"idx": i},
            )
            all_traces.append(trace)

        bt = builder.build_batch_trace(all_traces)
        assert bt is not None, "BatchTrace should not be None"
        assert len(bt) == 8
        assert bt.batch_size == 8

        # Properties
        frac = bt.system2_fraction
        assert 0.0 < frac < 1.0, f"system2_fraction should be between 0 and 1, got {frac}"
        avg = bt.avg_steps
        assert avg > 0.0, f"avg_steps should be > 0, got {avg}"
        cr = bt.convergence_rate
        assert cr == 1.0, f"All S2 items converged, expected rate=1.0, got {cr}"
        sd = bt.summary_dict()
        assert sd["batch_size"] == 8
        assert sd["num_traces"] == 8

        # Serialization roundtrip
        bt_json = bt.to_json()
        bt_rt = BatchTrace.from_json(bt_json)
        assert len(bt_rt) == len(bt)
        assert bt_rt.batch_size == bt.batch_size

        # merge_batch_traces
        bt2 = BatchTrace(
            traces=[_make_trace(True), _make_trace(False)],
            batch_size=2,
        )
        merged = merge_batch_traces([bt, bt2])
        assert len(merged) == 10
        assert merged.batch_size == 10

        passed += 1
        print("  [PASS] Test 12: build_batch_trace end-to-end")
    except Exception as e:
        print(f"  [FAIL] Test 12: build_batch_trace end-to-end: {e}")

    # -- summary -------------------------------------------------------------
    print(f"\n{passed}/{total} self-tests passed")
    if passed < total:
        sys.exit(1)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    _run_self_tests()
