"""
brain_ai/reasoning/dual_process.py -- Dual-Process Reasoner (System 1/2 + Metacognitive Control)

Main integration module for dual-process reasoning.  Wires together:

  - System 1 (fast, single-pass MLP): produces prediction + calibrated confidence
  - System 2 (slow, iterative GRU loop): refines uncertain predictions
  - Metacognitive router: deterministic routing based on confidence, novelty, anomaly
  - Calibrator: temperature scaling / isotonic regression for confidence calibration
  - Reasoning trace: JSON-serializable per-item reasoning audit trail

Pipeline:
    x  ->  System 1  ->  calibrate confidence  ->  compute novelty
       ->  metacognitive routing decision
       ->  if any uncertain: scatter -> System 2 -> gather
       ->  merge S1 + S2 outputs  ->  build trace  ->  ReasoningOutput

Scatter/gather ensures that only uncertain items enter the S2 loop, so cost
scales with the fraction of uncertain items rather than the full batch.

Brain analog: prefrontal cortex (metacognition), basal ganglia (routing),
parietal cortex (System 2 deliberation).

References:
    Kahneman (2011) "Thinking, Fast and Slow"
    Evans & Stanovich (2013) "Dual-Process Theories of Higher Cognition"
    Bengio (2017) "The Consciousness Prior"
    Dehaene et al. (2021) "What is consciousness, and could machines have it?"

Usage:
    from brain_ai.reasoning.dual_process import (
        DualProcessReasoner,
        DualProcessFullConfig,
        ReasoningOutput,
        create_dual_process_reasoner,
    )

    config = DualProcessFullConfig.minimal()
    model = DualProcessReasoner(config)
    out = model(torch.randn(4, 256))
    print(out.y.shape, out.used_system2)

Copy this template to brain_ai/reasoning/dual_process.py when integrating.
"""

from __future__ import annotations

import io
import json
import logging
import math
import time
import warnings
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ============================================================================
# SECTION 1: Sub-module result dataclasses
# ============================================================================


@dataclass
class System1Result:
    """Output produced by the System 1 fast path.

    Attributes:
        y1: Prediction logits/embedding from System 1, shape ``(B, output_dim)``.
        hidden: Intermediate hidden representation, shape ``(B, hidden_dim)``.
            Used as initial hidden state for System 2 if routed.
        conf_raw: Raw maximum softmax probability, shape ``(B,)``.
        conf_calibrated: Calibrated confidence after temperature / isotonic, shape ``(B,)``.
        entropy: Shannon entropy of the output distribution, shape ``(B,)``.
        margin: Difference between top-1 and top-2 logits, shape ``(B,)``.
        logits_topk_vals: Top-k logit values, shape ``(B, k)``.
        logits_topk_idx: Top-k logit indices, shape ``(B, k)``.
    """
    y1: Tensor
    hidden: Tensor
    conf_raw: Tensor
    conf_calibrated: Tensor
    entropy: Tensor
    margin: Tensor
    logits_topk_vals: Tensor
    logits_topk_idx: Tensor


@dataclass
class StepTrace:
    """Single refinement step trace for System 2.

    Attributes:
        step: Step index (0-based).
        conf: Confidence at this step.
        kl_delta: KL divergence from previous step distribution.
        logits_topk_vals: Top-k logit values at this step.
        logits_topk_idx: Top-k logit indices at this step.
        halt_check: Whether halt criteria were met at this step.
    """
    step: int
    conf: float
    kl_delta: float
    logits_topk_vals: List[float]
    logits_topk_idx: List[int]
    halt_check: bool


@dataclass
class System2Result:
    """Output produced by the System 2 iterative refinement loop.

    Attributes:
        y2: Refined prediction logits/embedding, shape ``(N_s2, output_dim)``
            where N_s2 is the number of items routed to System 2.
        steps_used: Number of refinement steps actually executed per item,
            shape ``(N_s2,)`` int.
        converged: Per-item convergence flag, shape ``(N_s2,)`` bool.
        halt_reason: Per-item halt reason string list, length N_s2.
        hidden_final: Final GRU hidden state, shape ``(N_s2, hidden_dim)``
            for single-layer GRU, or ``(num_layers, N_s2, hidden_dim)`` for
            stacked GRU.
        per_step_traces: Optional list of per-step traces (only when tracing).
            Outer list indexed by item, inner list by step.
        deep_supervision_logits: Optional list of per-step logits for deep
            supervision loss, each ``(N_s2, output_dim)``.
    """
    y2: Tensor
    steps_used: Tensor
    converged: Tensor
    halt_reason: List[str]
    hidden_final: Tensor
    per_step_traces: Optional[List[List[StepTrace]]] = None
    deep_supervision_logits: Optional[List[Tensor]] = None


@dataclass
class RoutingDecision:
    """Output of the metacognitive router.

    Attributes:
        used_system2: Per-item boolean mask, shape ``(B,)``.
        route_score: Per-item routing score, shape ``(B,)``.
        steps_budget: Per-item step budget for S2, shape ``(B,)`` int.
        novelty: Per-item novelty score, shape ``(B,)``.
    """
    used_system2: Tensor
    route_score: Tensor
    steps_budget: Tensor
    novelty: Tensor


@dataclass
class ReasoningTrace:
    """Full reasoning trace for a single batch item.

    JSON-serializable when converted via ``to_dict()``.

    Attributes:
        used_system2: Whether System 2 was invoked.
        route_score: Routing score that triggered the decision.
        route_threshold: Threshold used for routing.
        conf_raw: Raw confidence from System 1.
        conf_calibrated: Calibrated confidence.
        entropy: Entropy of System 1 distribution.
        margin: Logit margin from System 1.
        novelty: Novelty score.
        s1_topk_vals: Top-k logit values from System 1.
        s1_topk_idx: Top-k logit indices from System 1.
        s2_steps_used: Steps used by System 2 (0 if not invoked).
        s2_converged: Whether System 2 converged (False if not invoked).
        s2_halt_reason: Halt reason string (empty if not invoked).
        s2_step_traces: Per-step traces from System 2 (empty if not invoked).
    """
    used_system2: bool
    route_score: float
    route_threshold: float
    conf_raw: float
    conf_calibrated: float
    entropy: float
    margin: float
    novelty: float
    s1_topk_vals: List[float]
    s1_topk_idx: List[int]
    s2_steps_used: int = 0
    s2_converged: bool = False
    s2_halt_reason: str = ""
    s2_step_traces: List[StepTrace] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        d: Dict[str, Any] = {
            "used_system2": self.used_system2,
            "route_score": self.route_score,
            "route_threshold": self.route_threshold,
            "conf_raw": self.conf_raw,
            "conf_calibrated": self.conf_calibrated,
            "entropy": self.entropy,
            "margin": self.margin,
            "novelty": self.novelty,
            "s1_topk_vals": self.s1_topk_vals,
            "s1_topk_idx": self.s1_topk_idx,
            "s2_steps_used": self.s2_steps_used,
            "s2_converged": self.s2_converged,
            "s2_halt_reason": self.s2_halt_reason,
            "s2_step_traces": [
                {
                    "step": st.step,
                    "conf": st.conf,
                    "kl_delta": st.kl_delta,
                    "logits_topk_vals": st.logits_topk_vals,
                    "logits_topk_idx": st.logits_topk_idx,
                    "halt_check": st.halt_check,
                }
                for st in self.s2_step_traces
            ],
        }
        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReasoningTrace":
        """Deserialize from dictionary."""
        step_traces = [
            StepTrace(
                step=st["step"],
                conf=st["conf"],
                kl_delta=st["kl_delta"],
                logits_topk_vals=st["logits_topk_vals"],
                logits_topk_idx=st["logits_topk_idx"],
                halt_check=st["halt_check"],
            )
            for st in d.get("s2_step_traces", [])
        ]
        return cls(
            used_system2=d["used_system2"],
            route_score=d["route_score"],
            route_threshold=d["route_threshold"],
            conf_raw=d["conf_raw"],
            conf_calibrated=d["conf_calibrated"],
            entropy=d["entropy"],
            margin=d["margin"],
            novelty=d["novelty"],
            s1_topk_vals=d["s1_topk_vals"],
            s1_topk_idx=d["s1_topk_idx"],
            s2_steps_used=d.get("s2_steps_used", 0),
            s2_converged=d.get("s2_converged", False),
            s2_halt_reason=d.get("s2_halt_reason", ""),
            s2_step_traces=step_traces,
        )


@dataclass
class ReasoningOutput:
    """Top-level output of the DualProcessReasoner.

    Attributes:
        y: Final output logits/embedding, shape ``(B, output_dim)``.
        used_system2: Per-item boolean mask, shape ``(B,)``.
        s1: System 1 results for the full batch.
        s2: System 2 results for the subset routed to S2, or None.
        trace: List of per-item ReasoningTrace objects, or None if
            ``return_details=False``.
        aux: Dictionary of auxiliary scalar/tensor metrics:
            ``s2_fraction``, ``mean_route_score``, ``mean_novelty``,
            ``mean_s2_steps``, ``convergence_rate``.
    """
    y: Tensor
    used_system2: Tensor
    s1: System1Result
    s2: Optional[System2Result]
    trace: Optional[List[ReasoningTrace]]
    aux: Dict[str, Tensor]


# ============================================================================
# SECTION 2: Recurrent state for streaming / sequential processing
# ============================================================================


@dataclass
class DualProcessState:
    """Carries recurrent state across sequential forward calls.

    Attributes:
        novelty_prototypes: Running prototype bank for novelty scoring,
            shape ``(num_prototypes, hidden_dim)``.
        novelty_counts: Per-prototype usage counts, shape ``(num_prototypes,)``.
        s2_hidden_carry: Optional carry-over hidden state from System 2,
            shape depends on GRU layer count.  ``None`` initially.
        calibrator_temperature: Current calibrator temperature scalar.
        step_count: Number of forward steps executed with this state.
    """
    novelty_prototypes: Tensor
    novelty_counts: Tensor
    s2_hidden_carry: Optional[Tensor]
    calibrator_temperature: float
    step_count: int


# ============================================================================
# SECTION 3: Configuration dataclasses
# ============================================================================


@dataclass
class System1Config:
    """Configuration for the System 1 fast predictor.

    Attributes:
        input_dim: Input dimensionality from workspace.
        hidden_dim: Width of MLP hidden layers.
        output_dim: Output logits/embedding dimensionality.
        num_layers: Number of MLP hidden layers.
        confidence_head: Whether to use a dedicated confidence head
            (vs deriving confidence from logits alone).
        dropout: Dropout rate for regularization.
        activation: Activation function name (``"relu"``, ``"gelu"``, ``"silu"``).
    """
    input_dim: int = 4096
    hidden_dim: int = 512
    output_dim: int = 256
    num_layers: int = 2
    confidence_head: bool = True
    dropout: float = 0.1
    activation: str = "gelu"


@dataclass
class System2Config:
    """Configuration for the System 2 iterative refinement loop.

    Attributes:
        hidden_dim: GRU hidden state dimensionality.
        max_steps: Maximum refinement iterations.
        convergence_eps: KL stability threshold for convergence.
        convergence_patience: Consecutive stable steps required before halt.
        nan_guard: Whether to halt on NaN detection.
        output_dim: Output logits/embedding dimensionality (must match S1).
        gru_layers: Number of stacked GRU layers.
        dropout: Dropout rate within GRU.
        gradient_clip_value: Output clamp range (0 disables).
    """
    hidden_dim: int = 512
    max_steps: int = 10
    convergence_eps: float = 1e-3
    convergence_patience: int = 2
    nan_guard: bool = True
    output_dim: int = 256
    gru_layers: int = 1
    dropout: float = 0.1
    gradient_clip_value: float = 1.0


@dataclass
class MetacognitionConfig:
    """Configuration for the metacognitive router.

    Attributes:
        route_threshold: Score threshold for S2 activation.
        w_conf: Weight for (1 - calibrated_conf) in route score.
        w_novelty: Weight for novelty score in route score.
        w_anomaly: Weight for anomaly score in route score.
        w_budget: Weight for remaining budget penalty in route score.
        min_conf_to_skip_s2: Hard confidence skip threshold. Items with
            calibrated confidence above this never go to S2.
        base_steps: Base S2 step allocation.
        step_scale_alpha: Scaling factor from route score to step budget.
        always_run_s2: Debug flag to force all items through S2.
        novelty_method: Novelty scoring method (``"prototype"``, ``"entropy"``,
            ``"none"``).
        num_prototypes: Number of prototypes for prototype-based novelty.
        novelty_ema_decay: Exponential moving average decay for prototype updates.
    """
    route_threshold: float = 0.5
    w_conf: float = 1.0
    w_novelty: float = 0.5
    w_anomaly: float = 0.3
    w_budget: float = 0.1
    min_conf_to_skip_s2: float = 0.95
    base_steps: float = 3.0
    step_scale_alpha: float = 5.0
    always_run_s2: bool = False
    novelty_method: str = "prototype"
    num_prototypes: int = 64
    novelty_ema_decay: float = 0.99


@dataclass
class CalibrationConfig:
    """Configuration for confidence calibration.

    Attributes:
        method: Calibration method (``"temperature"``, ``"isotonic"``, ``"none"``).
        initial_temperature: Starting temperature for temperature scaling.
        freeze_after_fit: Lock calibrator parameters after fitting.
        isotonic_bins: Number of bins for isotonic regression.
    """
    method: str = "temperature"
    initial_temperature: float = 1.5
    freeze_after_fit: bool = True
    isotonic_bins: int = 15


@dataclass
class DualProcessFullConfig:
    """Aggregated configuration for the full dual-process reasoning module.

    Attributes:
        system1: System 1 configuration.
        system2: System 2 configuration.
        metacognition: Metacognitive router configuration.
        calibration: Calibration configuration.
        trace_top_k: Number of top-k logit entries to store in traces.
        enable_deep_supervision: Whether to produce per-step logits from S2
            for auxiliary deep supervision loss.
    """
    system1: System1Config = field(default_factory=System1Config)
    system2: System2Config = field(default_factory=System2Config)
    metacognition: MetacognitionConfig = field(default_factory=MetacognitionConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    trace_top_k: int = 5
    enable_deep_supervision: bool = False

    # -- Preset constructors -------------------------------------------------

    @classmethod
    def minimal(cls) -> "DualProcessFullConfig":
        """Minimal config for unit tests (~100K params)."""
        return cls(
            system1=System1Config(
                input_dim=256,
                hidden_dim=128,
                output_dim=64,
                num_layers=1,
                dropout=0.0,
            ),
            system2=System2Config(
                hidden_dim=128,
                max_steps=4,
                convergence_eps=1e-2,
                convergence_patience=1,
                output_dim=64,
                gru_layers=1,
                dropout=0.0,
            ),
            metacognition=MetacognitionConfig(
                num_prototypes=8,
                base_steps=2.0,
                step_scale_alpha=2.0,
            ),
            calibration=CalibrationConfig(
                method="temperature",
                initial_temperature=1.5,
            ),
            trace_top_k=3,
            enable_deep_supervision=False,
        )

    @classmethod
    def dev(cls) -> "DualProcessFullConfig":
        """Dev config for rapid iteration (~10M params)."""
        return cls(
            system1=System1Config(
                input_dim=512,
                hidden_dim=256,
                output_dim=128,
                num_layers=2,
                dropout=0.1,
            ),
            system2=System2Config(
                hidden_dim=256,
                max_steps=8,
                convergence_eps=1e-3,
                convergence_patience=2,
                output_dim=128,
                gru_layers=1,
                dropout=0.1,
            ),
            metacognition=MetacognitionConfig(
                num_prototypes=32,
            ),
            calibration=CalibrationConfig(
                method="temperature",
                initial_temperature=1.5,
            ),
            trace_top_k=5,
            enable_deep_supervision=False,
        )

    @classmethod
    def production_1b(cls) -> "DualProcessFullConfig":
        """Production 1B config."""
        return cls(
            system1=System1Config(
                input_dim=4096,
                hidden_dim=1024,
                output_dim=512,
                num_layers=3,
                dropout=0.1,
                activation="gelu",
            ),
            system2=System2Config(
                hidden_dim=1024,
                max_steps=12,
                convergence_eps=5e-4,
                convergence_patience=2,
                output_dim=512,
                gru_layers=2,
                dropout=0.1,
            ),
            metacognition=MetacognitionConfig(
                num_prototypes=128,
                base_steps=4.0,
                step_scale_alpha=6.0,
            ),
            calibration=CalibrationConfig(
                method="temperature",
                initial_temperature=1.5,
            ),
            trace_top_k=10,
            enable_deep_supervision=True,
        )

    @classmethod
    def production_3b(cls) -> "DualProcessFullConfig":
        """Production 3B config."""
        return cls(
            system1=System1Config(
                input_dim=4096,
                hidden_dim=2048,
                output_dim=1024,
                num_layers=3,
                dropout=0.1,
                activation="gelu",
            ),
            system2=System2Config(
                hidden_dim=2048,
                max_steps=16,
                convergence_eps=5e-4,
                convergence_patience=2,
                output_dim=1024,
                gru_layers=2,
                dropout=0.1,
                gradient_clip_value=1.0,
            ),
            metacognition=MetacognitionConfig(
                num_prototypes=256,
                base_steps=5.0,
                step_scale_alpha=7.0,
            ),
            calibration=CalibrationConfig(
                method="temperature",
                initial_temperature=1.5,
            ),
            trace_top_k=10,
            enable_deep_supervision=True,
        )

    @classmethod
    def production_7b(cls) -> "DualProcessFullConfig":
        """Production 7B config -- full scale."""
        return cls(
            system1=System1Config(
                input_dim=4096,
                hidden_dim=4096,
                output_dim=2048,
                num_layers=4,
                dropout=0.1,
                activation="gelu",
            ),
            system2=System2Config(
                hidden_dim=4096,
                max_steps=20,
                convergence_eps=1e-4,
                convergence_patience=3,
                output_dim=2048,
                gru_layers=3,
                dropout=0.1,
                gradient_clip_value=1.0,
            ),
            metacognition=MetacognitionConfig(
                num_prototypes=512,
                base_steps=6.0,
                step_scale_alpha=8.0,
            ),
            calibration=CalibrationConfig(
                method="temperature",
                initial_temperature=1.5,
            ),
            trace_top_k=15,
            enable_deep_supervision=True,
        )


# ============================================================================
# SECTION 4: Activation helper
# ============================================================================


def _get_activation(name: str) -> nn.Module:
    """Return an activation module by name.

    Args:
        name: One of ``"relu"``, ``"gelu"``, ``"silu"``, ``"tanh"``.

    Returns:
        Corresponding ``nn.Module`` activation.

    Raises:
        ValueError: If the name is unrecognized.
    """
    name_lower = name.lower()
    if name_lower == "relu":
        return nn.ReLU()
    elif name_lower == "gelu":
        return nn.GELU()
    elif name_lower == "silu":
        return nn.SiLU()
    elif name_lower == "tanh":
        return nn.Tanh()
    else:
        raise ValueError(
            f"Unknown activation: {name!r}. Expected relu/gelu/silu/tanh."
        )


# ============================================================================
# SECTION 5: System 1 -- Fast predictor with confidence heads
# ============================================================================


class System1Fast(nn.Module):
    """System 1: single-pass fast predictor with multiple uncertainty proxies.

    Produces a prediction ``y1``, an intermediate ``hidden`` representation,
    and three uncertainty metrics: ``conf_raw``, ``entropy``, ``margin``.

    If ``confidence_head`` is enabled, a separate MLP head predicts confidence
    directly; otherwise confidence is derived from the output logits.

    Args:
        config: System1Config with architecture parameters.
    """

    def __init__(self, config: System1Config) -> None:
        super().__init__()
        self.config = config

        # Build MLP backbone
        layers: List[nn.Module] = []
        in_dim = config.input_dim
        for _i in range(config.num_layers):
            layers.append(nn.Linear(in_dim, config.hidden_dim))
            layers.append(_get_activation(config.activation))
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = config.hidden_dim
        self.backbone = nn.Sequential(*layers)

        # Output projection
        self.output_proj = nn.Linear(config.hidden_dim, config.output_dim)

        # Optional dedicated confidence head
        if config.confidence_head:
            self.conf_head: Optional[nn.Module] = nn.Sequential(
                nn.Linear(config.hidden_dim, max(config.hidden_dim // 2, 1)),
                _get_activation(config.activation),
                nn.Linear(max(config.hidden_dim // 2, 1), 1),
                nn.Sigmoid(),
            )
        else:
            self.conf_head = None

    def forward(
        self,
        x: Tensor,
        top_k: int = 5,
    ) -> System1Result:
        """Run System 1 fast path.

        Args:
            x: Input tensor, shape ``(B, input_dim)`` or ``(B, K, input_dim)``.
                If 3-D, it is mean-pooled over the slot dimension to ``(B, input_dim)``.
            top_k: Number of top logit entries to store.

        Returns:
            System1Result with predictions and uncertainty metrics.
        """
        # Handle (B, K, D) slot input by mean pooling
        if x.dim() == 3:
            x_2d = x.mean(dim=1)  # (B, D)
        else:
            x_2d = x  # (B, D)

        hidden: Tensor = self.backbone(x_2d)       # (B, hidden_dim)
        y1: Tensor = self.output_proj(hidden)       # (B, output_dim)

        # -- Uncertainty metrics in fp32 for numerical stability --
        y1_f32 = y1.float()
        probs = F.softmax(y1_f32, dim=-1)           # (B, output_dim)

        # Raw confidence: max probability
        conf_raw_logit = probs.max(dim=-1).values   # (B,)

        # Entropy: -sum(p * log(p))
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=-1)  # (B,)

        # Margin: top1 - top2 logit difference
        out_dim = y1_f32.shape[-1]
        if out_dim >= 2:
            topk2_vals, _topk2_idx = torch.topk(y1_f32, k=2, dim=-1)
            margin = topk2_vals[:, 0] - topk2_vals[:, 1]  # (B,)
        else:
            margin = torch.ones(y1_f32.shape[0], device=y1.device, dtype=torch.float32)

        # If dedicated confidence head exists, use it instead of max-prob
        if self.conf_head is not None:
            conf_raw = self.conf_head(hidden).squeeze(-1)  # (B,)
        else:
            conf_raw = conf_raw_logit

        # Top-k logits for trace storage
        actual_k = min(top_k, out_dim)
        logits_topk_vals, logits_topk_idx = torch.topk(y1_f32, k=actual_k, dim=-1)

        return System1Result(
            y1=y1,
            hidden=hidden,
            conf_raw=conf_raw,
            conf_calibrated=conf_raw.clone(),  # placeholder; overwritten by calibrator
            entropy=entropy,
            margin=margin,
            logits_topk_vals=logits_topk_vals,
            logits_topk_idx=logits_topk_idx,
        )


# ============================================================================
# SECTION 6: Calibrator -- Temperature scaling / Isotonic regression
# ============================================================================


class TemperatureScaler(nn.Module):
    """Platt scaling (temperature scaling) for confidence calibration.

    Learns a single temperature parameter ``T`` such that
    ``calibrated = sigmoid(logit(conf_raw) / T)``.

    The logit transform maps [0, 1] to (-inf, +inf), dividing by T > 1
    compresses the distribution toward 0.5, and sigmoid maps back to [0, 1].

    Args:
        initial_temperature: Starting temperature value (must be > 0).
    """

    def __init__(self, initial_temperature: float = 1.5) -> None:
        super().__init__()
        # Store log-temperature for unconstrained optimisation
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(max(initial_temperature, 1e-6)), dtype=torch.float32)
        )

    @property
    def temperature(self) -> Tensor:
        """Current temperature (always positive)."""
        return self.log_temperature.exp()

    def forward(self, conf_raw: Tensor) -> Tensor:
        """Apply temperature scaling to raw confidence.

        Args:
            conf_raw: Raw confidence scores, shape ``(B,)``, values in [0, 1].

        Returns:
            Calibrated confidence, shape ``(B,)``, values in [0, 1].
        """
        conf_f32 = conf_raw.float()
        T = self.temperature.float()
        # Clamp to avoid log(0) in logit transform
        conf_clamped = conf_f32.clamp(1e-7, 1.0 - 1e-7)
        logit = torch.log(conf_clamped / (1.0 - conf_clamped))
        return torch.sigmoid(logit / T)


class IsotonicCalibrator(nn.Module):
    """Binned isotonic regression calibrator.

    Fits a piecewise-constant monotonic mapping from raw confidence to
    calibrated confidence using histogram bins.  After fitting, the mapping
    is stored as a buffer and applied via linear interpolation.

    Args:
        num_bins: Number of histogram bins.
    """

    def __init__(self, num_bins: int = 15) -> None:
        super().__init__()
        self.num_bins = num_bins
        # Bin edges: evenly spaced in [0, 1]
        edges = torch.linspace(0.0, 1.0, num_bins + 1)
        self.register_buffer("bin_edges", edges)
        # Calibrated values per bin (initialised to identity mapping)
        bin_centers = (edges[:-1] + edges[1:]) / 2.0
        self.register_buffer("bin_values", bin_centers.clone())
        self.register_buffer("fitted", torch.tensor(False))

    def fit(self, conf_raw: Tensor, labels: Tensor) -> None:
        """Fit isotonic calibrator on validation data.

        Uses pool adjacent violators (PAV) to enforce monotonicity.

        Args:
            conf_raw: Raw confidence values, shape ``(N,)``.
            labels: Binary correctness labels, shape ``(N,)`` float.
        """
        conf_raw_d = conf_raw.float().detach()
        labels_d = labels.float().detach()

        bin_values: List[float] = []
        for i in range(self.num_bins):
            lo = self.bin_edges[i]
            hi = self.bin_edges[i + 1]
            if i == self.num_bins - 1:
                mask = (conf_raw_d >= lo) & (conf_raw_d <= hi)
            else:
                mask = (conf_raw_d >= lo) & (conf_raw_d < hi)
            if mask.any():
                bin_values.append(labels_d[mask].mean().item())
            else:
                bin_values.append(((lo + hi) / 2.0).item())

        # Enforce monotonicity via PAV
        values = bin_values[:]
        for _pav_iter in range(self.num_bins):
            changed = False
            for j in range(len(values) - 1):
                if values[j] > values[j + 1]:
                    avg = (values[j] + values[j + 1]) / 2.0
                    values[j] = avg
                    values[j + 1] = avg
                    changed = True
            if not changed:
                break

        self.bin_values.copy_(torch.tensor(values, dtype=torch.float32))
        self.fitted.fill_(True)

    def forward(self, conf_raw: Tensor) -> Tensor:
        """Apply isotonic calibration via linear interpolation.

        Args:
            conf_raw: Raw confidence, shape ``(B,)``.

        Returns:
            Calibrated confidence, shape ``(B,)``.
        """
        if not self.fitted.item():
            return conf_raw  # Identity if not fitted

        conf_f32 = conf_raw.float()
        bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0

        result = torch.zeros_like(conf_f32)
        for b_idx in range(conf_f32.shape[0]):
            val = conf_f32[b_idx].item()
            if val <= bin_centers[0].item():
                result[b_idx] = self.bin_values[0]
            elif val >= bin_centers[-1].item():
                result[b_idx] = self.bin_values[-1]
            else:
                for j in range(len(bin_centers) - 1):
                    if bin_centers[j].item() <= val <= bin_centers[j + 1].item():
                        lo_idx = j
                        hi_idx = j + 1
                        frac = (val - bin_centers[lo_idx].item()) / (
                            bin_centers[hi_idx].item() - bin_centers[lo_idx].item() + 1e-10
                        )
                        result[b_idx] = (
                            self.bin_values[lo_idx] * (1.0 - frac)
                            + self.bin_values[hi_idx] * frac
                        )
                        break
        return result


class Calibrator(nn.Module):
    """Dispatcher that selects temperature or isotonic calibration.

    Args:
        config: CalibrationConfig specifying which method to use.
    """

    def __init__(self, config: CalibrationConfig) -> None:
        super().__init__()
        self.config = config
        self.method = config.method.lower()

        if self.method == "temperature":
            self.scaler: Optional[nn.Module] = TemperatureScaler(config.initial_temperature)
        elif self.method == "isotonic":
            self.scaler = IsotonicCalibrator(config.isotonic_bins)
        elif self.method == "none":
            self.scaler = None
        else:
            raise ValueError(
                f"Unknown calibration method: {config.method!r}. "
                "Expected 'temperature', 'isotonic', or 'none'."
            )

    def forward(self, conf_raw: Tensor) -> Tensor:
        """Apply calibration to raw confidence scores.

        Args:
            conf_raw: Raw confidence, shape ``(B,)``.

        Returns:
            Calibrated confidence, shape ``(B,)``.
        """
        if self.scaler is None:
            return conf_raw
        return self.scaler(conf_raw)

    def freeze(self) -> None:
        """Freeze calibrator parameters (no gradient updates)."""
        if self.scaler is not None:
            for p in self.scaler.parameters():
                p.requires_grad_(False)

    def unfreeze(self) -> None:
        """Unfreeze calibrator parameters."""
        if self.scaler is not None:
            for p in self.scaler.parameters():
                p.requires_grad_(True)


# ============================================================================
# SECTION 7: Novelty scorer
# ============================================================================


class NoveltyScorer(nn.Module):
    """Computes novelty score for input representations.

    Supports multiple methods:
      - ``"prototype"``: Cosine distance to nearest prototype in a learned bank.
        Prototypes are updated via exponential moving average during training.
      - ``"entropy"``: Uses the entropy from System 1 as a novelty proxy,
        normalised via sigmoid centred at 1.0.
      - ``"none"``: Returns zeros.

    Args:
        config: MetacognitionConfig with novelty parameters.
        hidden_dim: Dimensionality of the hidden representation.
    """

    def __init__(self, config: MetacognitionConfig, hidden_dim: int) -> None:
        super().__init__()
        self.config = config
        self.method = config.novelty_method.lower()
        self.hidden_dim = hidden_dim

        if self.method == "prototype":
            self.register_buffer(
                "prototypes",
                torch.randn(config.num_prototypes, hidden_dim) * 0.01,
            )
            self.register_buffer(
                "prototype_counts",
                torch.zeros(config.num_prototypes),
            )
            self.ema_decay = config.novelty_ema_decay
        elif self.method in ("entropy", "none"):
            pass
        else:
            raise ValueError(
                f"Unknown novelty method: {config.novelty_method!r}. "
                "Expected 'prototype', 'entropy', or 'none'."
            )

    def forward(
        self,
        hidden: Tensor,
        entropy: Optional[Tensor] = None,
        state: Optional[DualProcessState] = None,
    ) -> Tensor:
        """Compute novelty score.

        Args:
            hidden: Hidden representation from System 1, shape ``(B, hidden_dim)``.
            entropy: Optional entropy from System 1, shape ``(B,)``.
                Required when method is ``"entropy"``.
            state: Optional state carrying prototype bank across time steps.

        Returns:
            Novelty scores, shape ``(B,)``, values in [0, 1].
        """
        B = hidden.shape[0]
        device = hidden.device

        if self.method == "prototype":
            protos = (
                state.novelty_prototypes
                if state is not None and state.novelty_prototypes is not None
                else self.prototypes
            )

            hidden_f32 = hidden.float()
            protos_f32 = protos.float()

            # Cosine distance: 1 - cosine_similarity
            hidden_norm = F.normalize(hidden_f32, dim=-1)   # (B, D)
            protos_norm = F.normalize(protos_f32, dim=-1)   # (P, D)
            similarity = torch.mm(hidden_norm, protos_norm.t())  # (B, P)
            max_sim, nearest_idx = similarity.max(dim=-1)   # (B,)
            novelty = (1.0 - max_sim).clamp(0.0, 1.0)

            # Update prototypes during training via EMA
            if self.training:
                self._update_prototypes(hidden_f32, nearest_idx, state)

            return novelty

        elif self.method == "entropy":
            if entropy is None:
                return torch.zeros(B, device=device, dtype=torch.float32)
            entropy_f32 = entropy.float()
            novelty = torch.sigmoid(entropy_f32 - 1.0)
            return novelty

        else:  # "none"
            return torch.zeros(B, device=device, dtype=torch.float32)

    @torch.no_grad()
    def _update_prototypes(
        self,
        hidden: Tensor,
        nearest_idx: Tensor,
        state: Optional[DualProcessState],
    ) -> None:
        """Update prototypes via exponential moving average.

        Args:
            hidden: Current batch hidden representations, shape ``(B, D)``.
            nearest_idx: Index of nearest prototype per item, shape ``(B,)``.
            state: Optional state to update in-place.
        """
        target = (
            state.novelty_prototypes
            if state is not None and state.novelty_prototypes is not None
            else self.prototypes
        )
        target_counts = (
            state.novelty_counts
            if state is not None and state.novelty_counts is not None
            else self.prototype_counts
        )

        for i in range(hidden.shape[0]):
            idx = nearest_idx[i].item()
            target[idx] = (
                self.ema_decay * target[idx]
                + (1.0 - self.ema_decay) * hidden[i]
            )
            target_counts[idx] += 1


# ============================================================================
# SECTION 8: Metacognitive router
# ============================================================================


class MetacognitiveRouter(nn.Module):
    """Deterministic routing policy combining multiple uncertainty signals.

    Route score formula::

        route_score = w_conf * (1 - calibrated_conf)
                    + w_novelty * novelty
                    + w_anomaly * anomaly
                    - w_budget * (1 - remaining_budget)

        used_system2 = route_score >= route_threshold
        steps_budget = clamp(round(base_steps + alpha * route_score), 1, max_steps)

    The ``min_conf_to_skip_s2`` threshold provides a hard override: items
    whose calibrated confidence exceeds it are never routed to S2.

    All computation is in fp32 to ensure deterministic routing.

    Args:
        config: MetacognitionConfig with routing parameters.
        max_steps: Maximum S2 steps from System2Config.
    """

    def __init__(self, config: MetacognitionConfig, max_steps: int) -> None:
        super().__init__()
        self.config = config
        self.max_steps = max_steps

    def forward(
        self,
        s1_result: System1Result,
        novelty: Tensor,
        anomaly: Optional[Tensor] = None,
        ignition: Optional[Tensor] = None,
        remaining_budget: Optional[Tensor] = None,
    ) -> RoutingDecision:
        """Compute deterministic routing decision.

        Args:
            s1_result: System 1 result with calibrated confidence.
            novelty: Novelty scores, shape ``(B,)``.
            anomaly: Optional anomaly scores from HTM, shape ``(B,)``.
                Defaults to zeros if not provided.
            ignition: Optional ignition scores from global workspace, shape ``(B,)``.
                Not used in route score directly but available for extensions.
            remaining_budget: Optional remaining compute budget, shape ``(B,)``.
                Defaults to ones (full budget) if not provided.

        Returns:
            RoutingDecision with per-item routing masks and step budgets.
        """
        B = s1_result.conf_calibrated.shape[0]
        device = s1_result.conf_calibrated.device
        cfg = self.config

        # All computation in fp32 for determinism
        conf = s1_result.conf_calibrated.float()
        nov = novelty.float()
        anom = (
            anomaly.float()
            if anomaly is not None
            else torch.zeros(B, device=device, dtype=torch.float32)
        )
        budget = (
            remaining_budget.float()
            if remaining_budget is not None
            else torch.ones(B, device=device, dtype=torch.float32)
        )

        # Route score
        route_score: Tensor = (
            cfg.w_conf * (1.0 - conf)
            + cfg.w_novelty * nov
            + cfg.w_anomaly * anom
            - cfg.w_budget * (1.0 - budget)
        )

        # Determine routing mask
        if cfg.always_run_s2:
            used_system2 = torch.ones(B, device=device, dtype=torch.bool)
        else:
            used_system2 = route_score >= cfg.route_threshold
            # Hard skip: items with very high confidence always stay in S1
            high_conf_mask = conf >= cfg.min_conf_to_skip_s2
            used_system2 = used_system2 & (~high_conf_mask)

        # Step budget per item
        steps_float = cfg.base_steps + cfg.step_scale_alpha * route_score
        steps_budget = steps_float.round().clamp(1, self.max_steps).to(torch.long)

        return RoutingDecision(
            used_system2=used_system2,
            route_score=route_score,
            steps_budget=steps_budget,
            novelty=nov,
        )


# ============================================================================
# SECTION 9: System 2 -- Iterative GRU refinement with convergence
# ============================================================================


class System2Iterative(nn.Module):
    """System 2: iterative GRU-based refinement loop with convergence detection.

    Starting from the System 1 prediction, iteratively refines the output using
    a GRU cell.  Halts when KL divergence between consecutive steps falls below
    ``convergence_eps`` for ``convergence_patience`` consecutive steps, or when
    the per-item step budget is exhausted.

    Halt reasons:
      - ``"converged"``: KL stability criterion met.
      - ``"max_steps"``: Budget exhausted.
      - ``"nan_guard"``: NaN detected in output.

    Args:
        config: System2Config with loop parameters.
        input_dim: Dimensionality of the System 1 hidden representation.
    """

    def __init__(self, config: System2Config, input_dim: int) -> None:
        super().__init__()
        self.config = config

        # Input projection (from S1 output_dim to S2 hidden_dim)
        self.input_proj = nn.Linear(config.output_dim, config.hidden_dim)

        # Context projection (from S1 hidden_dim to S2 hidden_dim)
        self.context_proj = nn.Linear(input_dim, config.hidden_dim)

        # GRU cell(s) for iterative refinement
        self._use_gru_cell = config.gru_layers == 1
        if self._use_gru_cell:
            self.gru: nn.Module = nn.GRUCell(config.hidden_dim, config.hidden_dim)
        else:
            self.gru = nn.GRU(
                input_size=config.hidden_dim,
                hidden_size=config.hidden_dim,
                num_layers=config.gru_layers,
                batch_first=True,
                dropout=config.dropout if config.gru_layers > 1 else 0.0,
            )

        # Output projection back to output_dim
        self.output_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

        # Dropout in the loop
        self.loop_dropout: nn.Module = (
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        )

        # Flag wired by DualProcessReasoner
        self._enable_deep_supervision: bool = False

    def _gru_step(self, x: Tensor, h: Tensor) -> Tensor:
        """Single GRU step abstracting over GRUCell vs stacked GRU.

        Args:
            x: Input tensor, shape ``(N, hidden_dim)``.
            h: Hidden state.  GRUCell: ``(N, hidden_dim)``.
                Stacked GRU: ``(num_layers, N, hidden_dim)``.

        Returns:
            Updated hidden state, same layout as ``h``.
        """
        if self._use_gru_cell:
            return self.gru(x, h)
        else:
            _out, h_new = self.gru(x.unsqueeze(1), h)
            return h_new

    def _extract_top_hidden(self, h: Tensor) -> Tensor:
        """Extract the top-layer hidden state for output projection.

        Args:
            h: Hidden state from GRU.

        Returns:
            Hidden state of the last GRU layer, shape ``(N, hidden_dim)``.
        """
        if self._use_gru_cell:
            return h  # already (N, hidden_dim)
        else:
            return h[-1]  # last layer: (N, hidden_dim)

    def _init_hidden(self, context: Tensor) -> Tensor:
        """Initialise hidden state from context.

        Args:
            context: Context from S1 hidden, shape ``(N, input_dim)``.

        Returns:
            Initial hidden state for GRU.
        """
        h0 = self.context_proj(context)  # (N, hidden_dim)
        if self._use_gru_cell:
            return h0
        else:
            return h0.unsqueeze(0).expand(
                self.config.gru_layers, -1, -1
            ).contiguous()

    def forward(
        self,
        y1: Tensor,
        x_summary: Tensor,
        steps_budget: Tensor,
        return_traces: bool = False,
        top_k: int = 5,
        h_carry: Optional[Tensor] = None,
    ) -> System2Result:
        """Run System 2 iterative refinement.

        Only called on the subset of items routed to S2 (after scatter).

        Args:
            y1: System 1 prediction for uncertain items, shape ``(N, output_dim)``.
            x_summary: S1 hidden representation, shape ``(N, input_dim)``.
            steps_budget: Per-item step budget, shape ``(N,)`` long.
            return_traces: Whether to build per-step traces.
            top_k: Number of top-k logits to store in traces.
            h_carry: Optional carry-over hidden state from a previous time step.

        Returns:
            System2Result with refined predictions and convergence info.
        """
        N = y1.shape[0]
        device = y1.device
        cfg = self.config

        # Initialise
        y_current = self.input_proj(y1)  # (N, hidden_dim)
        if h_carry is not None:
            h = h_carry
        else:
            h = self._init_hidden(x_summary)

        # Upper bound on iteration count
        max_budget = int(steps_budget.max().item()) if N > 0 else 0
        max_budget = min(max_budget, cfg.max_steps)

        # Per-item tracking
        steps_used = torch.ones(N, device=device, dtype=torch.long)
        converged = torch.zeros(N, device=device, dtype=torch.bool)
        halt_reason: List[str] = ["" for _ in range(N)]
        active_mask = torch.ones(N, device=device, dtype=torch.bool)

        # Previous distribution for KL computation
        prev_logits = self.output_proj(self._extract_top_hidden(h)).float()
        prev_probs = F.softmax(prev_logits, dim=-1)

        # Optional collectors
        per_step_traces: Optional[List[List[StepTrace]]] = (
            [[] for _ in range(N)] if return_traces else None
        )
        deep_supervision_logits: Optional[List[Tensor]] = (
            [] if self.training and self._enable_deep_supervision else None
        )

        # Convergence patience counters
        stable_counts = torch.zeros(N, device=device, dtype=torch.long)

        for step in range(max_budget):
            # Determine which items are still running
            budget_active = steps_budget > step
            still_active = active_mask & budget_active

            if not still_active.any():
                break

            # GRU step
            y_input = self.loop_dropout(y_current)
            h = self._gru_step(y_input, h)
            h_out = self._extract_top_hidden(h)

            # Project to output space
            current_logits = self.output_proj(h_out)  # (N, output_dim)

            # Collect deep supervision logits
            if deep_supervision_logits is not None:
                deep_supervision_logits.append(current_logits)

            # -- Convergence check (fp32) --
            current_logits_f32 = current_logits.float()
            current_probs = F.softmax(current_logits_f32, dim=-1)

            # KL(current || prev) per item
            kl_div = F.kl_div(
                torch.log(current_probs + 1e-10),
                prev_probs,
                reduction="none",
            ).sum(dim=-1)  # (N,)

            # NaN guard
            if cfg.nan_guard:
                nan_mask = (
                    torch.isnan(current_logits_f32).any(dim=-1)
                    | torch.isnan(kl_div)
                )
                if nan_mask.any():
                    for idx_t in torch.where(nan_mask & still_active)[0]:
                        idx = idx_t.item()
                        if active_mask[idx]:
                            active_mask[idx] = False
                            halt_reason[idx] = "nan_guard"
                            steps_used[idx] = step + 1

            # Stability check
            stable = kl_div < cfg.convergence_eps
            stable_counts = torch.where(
                stable & still_active,
                stable_counts + 1,
                torch.zeros_like(stable_counts),
            )

            converged_now = (
                (stable_counts >= cfg.convergence_patience)
                & still_active
                & active_mask
            )
            if converged_now.any():
                for idx_t in torch.where(converged_now)[0]:
                    idx = idx_t.item()
                    if active_mask[idx]:
                        active_mask[idx] = False
                        converged[idx] = True
                        halt_reason[idx] = "converged"
                        steps_used[idx] = step + 1

            # Build per-step trace
            if return_traces and per_step_traces is not None:
                actual_k = min(top_k, current_logits_f32.shape[-1])
                tk_vals, tk_idx = torch.topk(
                    current_logits_f32, k=actual_k, dim=-1
                )
                conf_step = F.softmax(current_logits_f32, dim=-1).max(dim=-1).values

                for i in range(N):
                    if still_active[i]:
                        per_step_traces[i].append(
                            StepTrace(
                                step=step,
                                conf=conf_step[i].item(),
                                kl_delta=kl_div[i].item(),
                                logits_topk_vals=tk_vals[i].tolist(),
                                logits_topk_idx=tk_idx[i].tolist(),
                                halt_check=bool(stable[i].item()),
                            )
                        )

            # Prepare for next iteration
            if self.training:
                prev_probs = current_probs
            else:
                prev_probs = current_probs.detach()
            y_current = h_out  # feed GRU output back as next input

            # Update steps_used for items still running
            still_running = active_mask & budget_active
            steps_used = torch.where(
                still_running,
                torch.tensor(step + 1, device=device, dtype=torch.long),
                steps_used,
            )

        # Mark remaining active items as budget-exhausted
        for i in range(N):
            if active_mask[i] and halt_reason[i] == "":
                halt_reason[i] = "max_steps"
                steps_used[i] = min(int(steps_budget[i].item()), max_budget)

        # Final output
        h_final_out = self._extract_top_hidden(h)
        y2 = self.output_proj(h_final_out)  # (N, output_dim)

        # Clamp for stability during training
        if self.training and cfg.gradient_clip_value > 0:
            clamp_range = cfg.gradient_clip_value * 100
            y2 = y2.clamp(-clamp_range, clamp_range)

        return System2Result(
            y2=y2,
            steps_used=steps_used,
            converged=converged,
            halt_reason=halt_reason,
            hidden_final=h,
            per_step_traces=per_step_traces,
            deep_supervision_logits=deep_supervision_logits,
        )


# ============================================================================
# SECTION 10: DualProcessReasoner -- Main integration module
# ============================================================================


class DualProcessReasoner(nn.Module):
    """Dual-Process Reasoner integrating System 1, System 2, metacognition,
    calibration, novelty scoring, and reasoning trace.

    This is the main entry point for the dual-process reasoning pipeline.
    Given a workspace representation ``x``, it:

    1. Runs System 1 for fast prediction + uncertainty metrics.
    2. Calibrates the raw confidence using temperature/isotonic scaling.
    3. Computes novelty scores.
    4. Routes items via metacognitive policy (deterministic).
    5. Runs System 2 on uncertain items only (scatter/gather).
    6. Merges S1 and S2 outputs.
    7. Optionally builds a reasoning trace.
    8. Computes auxiliary metrics.

    Hard invariants:
      - Routing is deterministic for a fixed seed and deterministic settings.
      - ``return_details=False`` produces ``trace=None`` with near-zero overhead.
      - Per-item routing is computed independently (no cross-batch sorting).

    Args:
        config: DualProcessFullConfig with all sub-module parameters.

    Raises:
        ValueError: If ``system1.output_dim != system2.output_dim``.
    """

    def __init__(self, config: DualProcessFullConfig) -> None:
        super().__init__()
        self.config = config

        # Validate output_dim consistency
        if config.system1.output_dim != config.system2.output_dim:
            raise ValueError(
                f"System 1 output_dim ({config.system1.output_dim}) must match "
                f"System 2 output_dim ({config.system2.output_dim}). "
                "Both systems must produce outputs of the same dimensionality."
            )

        # ----- System 1 -----
        self.system1 = System1Fast(config.system1)

        # ----- Calibrator -----
        self.calibrator = Calibrator(config.calibration)

        # ----- Novelty scorer -----
        self.novelty_scorer = NoveltyScorer(
            config.metacognition,
            hidden_dim=config.system1.hidden_dim,
        )

        # ----- Metacognitive router -----
        self.router = MetacognitiveRouter(
            config.metacognition,
            max_steps=config.system2.max_steps,
        )

        # ----- System 2 -----
        self.system2 = System2Iterative(
            config.system2,
            input_dim=config.system1.hidden_dim,
        )
        # Wire deep supervision flag
        self.system2._enable_deep_supervision = config.enable_deep_supervision

        self._output_dim: int = config.system1.output_dim
        self._trace_top_k: int = config.trace_top_k

    @property
    def output_dim(self) -> int:
        """Output dimensionality of the reasoner."""
        return self._output_dim

    def param_count(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # -- State management ----------------------------------------------------

    def reset(
        self,
        batch_size: int,
        device: torch.device,
    ) -> DualProcessState:
        """Initialise a fresh recurrent state for sequential processing.

        Args:
            batch_size: Batch size (informational; not all state is batch-indexed).
            device: Target device.

        Returns:
            Fresh DualProcessState with cloned prototypes and zero counts.
        """
        hidden_dim = self.config.system1.hidden_dim
        num_protos = self.config.metacognition.num_prototypes

        if hasattr(self.novelty_scorer, "prototypes"):
            proto_init = self.novelty_scorer.prototypes.clone().to(device)
        else:
            proto_init = torch.randn(num_protos, hidden_dim, device=device) * 0.01

        cal_temp = 1.0
        if isinstance(self.calibrator.scaler, TemperatureScaler):
            cal_temp = self.calibrator.scaler.temperature.item()

        return DualProcessState(
            novelty_prototypes=proto_init,
            novelty_counts=torch.zeros(num_protos, device=device),
            s2_hidden_carry=None,
            calibrator_temperature=cal_temp,
            step_count=0,
        )

    @staticmethod
    def detach_state(state: DualProcessState) -> DualProcessState:
        """Detach all tensors in the state for use across time steps.

        Prevents gradient flow from future steps into past computations.

        Args:
            state: State to detach.

        Returns:
            New DualProcessState with detached tensors.
        """
        return DualProcessState(
            novelty_prototypes=state.novelty_prototypes.detach(),
            novelty_counts=state.novelty_counts.detach(),
            s2_hidden_carry=(
                state.s2_hidden_carry.detach()
                if state.s2_hidden_carry is not None
                else None
            ),
            calibrator_temperature=state.calibrator_temperature,
            step_count=state.step_count,
        )

    # -- Forward pass ---------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        *,
        context: Optional[Dict[str, Tensor]] = None,
        return_details: bool = False,
        state: Optional[DualProcessState] = None,
    ) -> ReasoningOutput:
        """Full dual-process reasoning forward pass.

        Args:
            x: Input tensor from the workspace.
                Shape ``(B, input_dim)`` for pooled representations, or
                ``(B, K, input_dim)`` for slot representations (mean-pooled internally).
            context: Optional context dictionary.  Recognised keys:

                - ``"anomaly"``: HTM anomaly scores, shape ``(B,)``.
                - ``"ignition"``: Global workspace ignition scores, shape ``(B,)``.
                - ``"budget"``: Remaining compute budget, shape ``(B,)``.

            return_details: If True, build and return per-item reasoning traces.
            state: Optional recurrent state for sequential/streaming processing.

        Returns:
            ReasoningOutput with final predictions, routing decisions, traces,
            and auxiliary metrics.
        """
        context = context or {}

        # ----------------------------------------------------------------
        # Step 1: System 1 fast path
        # ----------------------------------------------------------------
        s1_result: System1Result = self.system1(x, top_k=self._trace_top_k)
        B: int = s1_result.y1.shape[0]
        device: torch.device = s1_result.y1.device

        # ----------------------------------------------------------------
        # Step 2: Calibrate confidence
        # ----------------------------------------------------------------
        conf_calibrated: Tensor = self.calibrator(s1_result.conf_raw)
        s1_result.conf_calibrated = conf_calibrated

        # ----------------------------------------------------------------
        # Step 3: Compute novelty
        # ----------------------------------------------------------------
        novelty: Tensor = self.novelty_scorer(
            s1_result.hidden,
            entropy=s1_result.entropy,
            state=state,
        )

        # ----------------------------------------------------------------
        # Step 4: Metacognitive routing
        # ----------------------------------------------------------------
        routing_decision: RoutingDecision = self.router(
            s1_result,
            novelty=novelty,
            anomaly=context.get("anomaly", None),
            ignition=context.get("ignition", None),
            remaining_budget=context.get("budget", None),
        )

        s2_mask: Tensor = routing_decision.used_system2  # (B,) bool

        # ----------------------------------------------------------------
        # Step 5: Selective System 2 execution (scatter / gather)
        # ----------------------------------------------------------------
        s2_result: Optional[System2Result] = None
        y: Tensor = s1_result.y1.clone()

        if s2_mask.any():
            s2_indices: Tensor = torch.where(s2_mask)[0]
            N_s2: int = s2_indices.shape[0]

            # -- Scatter: extract uncertain items --
            y1_s2: Tensor = s1_result.y1[s2_indices]          # (N_s2, output_dim)
            x_summary_s2: Tensor = s1_result.hidden[s2_indices]  # (N_s2, hidden_dim)
            steps_s2: Tensor = routing_decision.steps_budget[s2_indices]

            # Handle hidden state carry-over for streaming
            h_carry: Optional[Tensor] = None
            if state is not None and state.s2_hidden_carry is not None:
                carry = state.s2_hidden_carry
                # Attempt to index into carry-over state
                if self.system2._use_gru_cell:
                    # (prev_N, hidden_dim) -> select matching indices
                    if carry.shape[0] >= B:
                        h_carry = carry[:B][s2_indices]
                    elif carry.shape[0] == N_s2:
                        h_carry = carry
                else:
                    # (num_layers, prev_N, hidden_dim)
                    if carry.dim() == 3 and carry.shape[1] >= B:
                        h_carry = carry[:, :B, :][:, s2_indices, :]
                    elif carry.dim() == 3 and carry.shape[1] == N_s2:
                        h_carry = carry

            # -- Run System 2 on the subset --
            s2_result = self.system2(
                y1_s2,
                x_summary_s2,
                steps_s2,
                return_traces=return_details,
                top_k=self._trace_top_k,
                h_carry=h_carry,
            )

            # -- Gather: merge S2 results back into full batch --
            y[s2_indices] = s2_result.y2

            # Update state with S2 hidden carry
            if state is not None:
                state.s2_hidden_carry = s2_result.hidden_final

        # ----------------------------------------------------------------
        # Step 6: Update state step count
        # ----------------------------------------------------------------
        if state is not None:
            state.step_count += 1

        # ----------------------------------------------------------------
        # Step 7: Build reasoning traces (only if requested)
        # ----------------------------------------------------------------
        trace: Optional[List[ReasoningTrace]] = None

        if return_details:
            trace = self._build_traces(
                B=B,
                s1_result=s1_result,
                s2_mask=s2_mask,
                s2_result=s2_result,
                routing_decision=routing_decision,
                novelty=novelty,
            )

        # ----------------------------------------------------------------
        # Step 8: Compute auxiliary metrics (always, cheap)
        # ----------------------------------------------------------------
        aux: Dict[str, Tensor] = self._compute_aux(
            B=B,
            device=device,
            s2_mask=s2_mask,
            s2_result=s2_result,
            routing_decision=routing_decision,
            novelty=novelty,
        )

        # ----------------------------------------------------------------
        # Step 9: Return ReasoningOutput
        # ----------------------------------------------------------------
        return ReasoningOutput(
            y=y,
            used_system2=s2_mask,
            s1=s1_result,
            s2=s2_result,
            trace=trace,
            aux=aux,
        )

    # -- Trace builder (private) ---------------------------------------------

    def _build_traces(
        self,
        B: int,
        s1_result: System1Result,
        s2_mask: Tensor,
        s2_result: Optional[System2Result],
        routing_decision: RoutingDecision,
        novelty: Tensor,
    ) -> List[ReasoningTrace]:
        """Build per-item reasoning traces.

        Args:
            B: Batch size.
            s1_result: Full-batch System 1 results.
            s2_mask: Boolean mask indicating S2 routing.
            s2_result: S2 results for the routed subset (or None).
            routing_decision: Routing decision from metacognitive router.
            novelty: Novelty scores.

        Returns:
            List of ``B`` ReasoningTrace objects.
        """
        route_threshold = self.router.config.route_threshold
        traces: List[ReasoningTrace] = []

        # Pre-compute s2_indices for fast lookup
        s2_indices_list: List[int] = []
        if s2_mask.any():
            s2_indices_list = torch.where(s2_mask)[0].tolist()
        s2_index_map: Dict[int, int] = {
            global_idx: local_idx
            for local_idx, global_idx in enumerate(s2_indices_list)
        }

        for i in range(B):
            item_trace = ReasoningTrace(
                used_system2=bool(s2_mask[i].item()),
                route_score=float(routing_decision.route_score[i].item()),
                route_threshold=float(route_threshold),
                conf_raw=float(s1_result.conf_raw[i].item()),
                conf_calibrated=float(s1_result.conf_calibrated[i].item()),
                entropy=float(s1_result.entropy[i].item()),
                margin=float(s1_result.margin[i].item()),
                novelty=float(novelty[i].item()),
                s1_topk_vals=[float(v) for v in s1_result.logits_topk_vals[i].tolist()],
                s1_topk_idx=[int(v) for v in s1_result.logits_topk_idx[i].tolist()],
            )

            # Add S2 trace if this item was routed to S2
            if s2_mask[i].item() and s2_result is not None and i in s2_index_map:
                j = s2_index_map[i]
                item_trace.s2_steps_used = int(s2_result.steps_used[j].item())
                item_trace.s2_converged = bool(s2_result.converged[j].item())
                item_trace.s2_halt_reason = s2_result.halt_reason[j]
                if s2_result.per_step_traces is not None and j < len(s2_result.per_step_traces):
                    item_trace.s2_step_traces = s2_result.per_step_traces[j]

            traces.append(item_trace)

        return traces

    # -- Aux metrics (private) -----------------------------------------------

    @staticmethod
    def _compute_aux(
        B: int,
        device: torch.device,
        s2_mask: Tensor,
        s2_result: Optional[System2Result],
        routing_decision: RoutingDecision,
        novelty: Tensor,
    ) -> Dict[str, Tensor]:
        """Compute auxiliary metrics dictionary.

        Args:
            B: Batch size.
            device: Device for tensor creation.
            s2_mask: Boolean S2 routing mask.
            s2_result: S2 results (or None).
            routing_decision: Routing decision.
            novelty: Novelty scores.

        Returns:
            Dictionary of scalar metrics.
        """
        s2_count = s2_mask.float().sum()
        s2_fraction = s2_count / max(B, 1)

        mean_route_score = routing_decision.route_score.mean()
        mean_novelty = novelty.mean()

        if s2_result is not None and s2_mask.any():
            mean_s2_steps = s2_result.steps_used.float().mean()
            convergence_rate = s2_result.converged.float().mean()
        else:
            mean_s2_steps = torch.tensor(0.0, device=device)
            convergence_rate = torch.tensor(0.0, device=device)

        return {
            "s2_fraction": (
                s2_fraction.detach()
                if isinstance(s2_fraction, Tensor)
                else torch.tensor(float(s2_fraction), device=device)
            ),
            "mean_route_score": mean_route_score.detach(),
            "mean_novelty": mean_novelty.detach(),
            "mean_s2_steps": (
                mean_s2_steps.detach()
                if isinstance(mean_s2_steps, Tensor)
                else torch.tensor(float(mean_s2_steps), device=device)
            ),
            "convergence_rate": (
                convergence_rate.detach()
                if isinstance(convergence_rate, Tensor)
                else torch.tensor(float(convergence_rate), device=device)
            ),
        }


# ============================================================================
# SECTION 11: Factory functions
# ============================================================================


def create_dual_process_reasoner(
    config: Optional[DualProcessFullConfig] = None,
    **kwargs: Any,
) -> DualProcessReasoner:
    """Factory function for creating a DualProcessReasoner.

    Args:
        config: Full configuration. If None, uses DualProcessFullConfig.minimal().
        **kwargs: Keyword arguments forwarded to DualProcessFullConfig if config
            is not provided. Recognised top-level keys: ``system1``, ``system2``,
            ``metacognition``, ``calibration``, ``trace_top_k``,
            ``enable_deep_supervision``.

    Returns:
        Configured DualProcessReasoner instance.

    Examples:
        >>> model = create_dual_process_reasoner()
        >>> model = create_dual_process_reasoner(DualProcessFullConfig.dev())
        >>> model = create_dual_process_reasoner(trace_top_k=10)
    """
    if config is None:
        if kwargs:
            s1_kw = kwargs.pop("system1", {})
            s2_kw = kwargs.pop("system2", {})
            mc_kw = kwargs.pop("metacognition", {})
            cal_kw = kwargs.pop("calibration", {})

            config = DualProcessFullConfig(
                system1=(
                    System1Config(**s1_kw) if isinstance(s1_kw, dict) else s1_kw
                ),
                system2=(
                    System2Config(**s2_kw) if isinstance(s2_kw, dict) else s2_kw
                ),
                metacognition=(
                    MetacognitionConfig(**mc_kw) if isinstance(mc_kw, dict) else mc_kw
                ),
                calibration=(
                    CalibrationConfig(**cal_kw) if isinstance(cal_kw, dict) else cal_kw
                ),
                trace_top_k=kwargs.get("trace_top_k", 5),
                enable_deep_supervision=kwargs.get("enable_deep_supervision", False),
            )
        else:
            config = DualProcessFullConfig.minimal()

    return DualProcessReasoner(config)


def create_from_brain_ai_config(
    reasoning_config: Any,
    input_dim: int = 4096,
    output_dim: int = 256,
) -> DualProcessReasoner:
    """Create a DualProcessReasoner from a legacy BrainAIConfig.reasoning dataclass.

    This adapter translates the legacy BrainAIConfig fields into the new
    DualProcessFullConfig format.

    Args:
        reasoning_config: Object with attributes like ``hidden_dim``,
            ``confidence_threshold``, ``max_iterations``, etc.
        input_dim: Workspace input dimensionality.
        output_dim: Output dimensionality.

    Returns:
        Configured DualProcessReasoner.
    """
    hidden_dim = getattr(reasoning_config, "hidden_dim", 512)
    confidence_threshold = getattr(reasoning_config, "confidence_threshold", 0.7)
    max_iterations = getattr(reasoning_config, "max_iterations", 10)
    num_reasoning_steps = getattr(reasoning_config, "num_reasoning_steps", 5)

    config = DualProcessFullConfig(
        system1=System1Config(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=getattr(reasoning_config, "system1_layers", 2),
            dropout=0.1,
        ),
        system2=System2Config(
            hidden_dim=hidden_dim,
            max_steps=max(max_iterations, num_reasoning_steps),
            convergence_eps=1e-3,
            convergence_patience=2,
            output_dim=output_dim,
        ),
        metacognition=MetacognitionConfig(
            route_threshold=1.0 - confidence_threshold,
            min_conf_to_skip_s2=0.95,
        ),
        calibration=CalibrationConfig(method="temperature"),
    )
    return DualProcessReasoner(config)


# ============================================================================
# SECTION 12: Utility / loss functions
# ============================================================================


def compute_deep_supervision_loss(
    s2_result: System2Result,
    targets: Tensor,
    loss_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    decay: float = 0.8,
) -> Tensor:
    """Compute deep supervision loss over S2 per-step logits.

    Applies a decaying weight to earlier steps so the final step contributes
    the most to the loss.

    Args:
        s2_result: System2Result with ``deep_supervision_logits``.
        targets: Target labels for the S2 subset, shape ``(N_s2,)`` long
            or ``(N_s2, output_dim)`` for soft targets.
        loss_fn: Loss function.  Defaults to ``F.cross_entropy`` for integer
            targets or ``F.mse_loss`` for soft targets.
        decay: Exponential decay factor for step weights (1.0 = equal weight).

    Returns:
        Scalar loss tensor (requires grad).
    """
    if (
        s2_result.deep_supervision_logits is None
        or len(s2_result.deep_supervision_logits) == 0
    ):
        return torch.tensor(0.0, requires_grad=True)

    if loss_fn is None:
        if targets.dim() == 1:
            loss_fn = F.cross_entropy
        else:
            loss_fn = lambda pred, tgt: F.mse_loss(pred, tgt)

    device = targets.device
    total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    num_steps = len(s2_result.deep_supervision_logits)

    for step_idx, logits in enumerate(s2_result.deep_supervision_logits):
        weight = decay ** (num_steps - 1 - step_idx)
        step_loss = loss_fn(logits, targets)
        total_loss = total_loss + weight * step_loss

    weight_sum = sum(decay ** (num_steps - 1 - i) for i in range(num_steps))
    return total_loss / max(weight_sum, 1e-8)


def compute_routing_loss(
    routing_decision: RoutingDecision,
    s1_correct: Tensor,
    s2_improved: Optional[Tensor] = None,
    target_s2_fraction: float = 0.2,
    fraction_weight: float = 0.1,
) -> Tensor:
    """Compute auxiliary routing loss to encourage efficient routing.

    Penalises routing correct S1 items to S2 (wasted compute) and not routing
    incorrect S1 items to S2 (missed corrections).  Also optionally regularises
    the S2 fraction toward a target.

    Args:
        routing_decision: RoutingDecision from the router.
        s1_correct: Boolean mask of items where S1 was correct, shape ``(B,)``.
        s2_improved: Optional boolean mask of items where S2 improved over S1.
        target_s2_fraction: Target fraction of items to route to S2.
        fraction_weight: Weight for the fraction regularisation term.

    Returns:
        Scalar loss tensor.
    """
    used_s2 = routing_decision.used_system2.float()
    s1_correct_f = s1_correct.float()

    # Penalise routing correct items to S2 (unnecessary compute)
    wasted = (used_s2 * s1_correct_f).mean()

    # Penalise not routing incorrect items to S2 (missed corrections)
    missed = ((1.0 - used_s2) * (1.0 - s1_correct_f)).mean()

    # Fraction regularisation
    actual_fraction = used_s2.mean()
    fraction_reg = (actual_fraction - target_s2_fraction) ** 2

    return wasted + missed + fraction_weight * fraction_reg


# ============================================================================
# SECTION 13: Self-tests
# ============================================================================


def _run_self_tests() -> bool:
    """Run comprehensive self-tests for the DualProcessReasoner.

    Tests cover:
      1.  Forward with (B, D) input
      2.  Forward with (B, K, D) slots
      3.  S1-only path (all confident)
      4.  S2 path (all uncertain)
      5.  Mixed batch routing
      6.  return_details=True -> trace not None
      7.  return_details=False -> trace is None
      8.  Trace JSON serialisation
      9.  Determinism across runs
      10. State management
      11. All config presets instantiate
      12. Selective S2 execution
      13. Gradient flow in training mode
      14. Aux metrics computed correctly
      15. Checkpoint save/load roundtrip
      16. Batch size 1
      17. Empty context handling
      18. Context with anomaly/ignition/budget
      19. Factory function
      20. No NaN in output
      21. S2 convergence detection
      22. Calibrator freeze/unfreeze
      23. Deep supervision loss
      24. Routing loss
      25. S1 uncertainty metrics valid
      26. Novelty prototype update
      27. Novelty entropy method
      28. Novelty none method
      29. All calibrator methods
      30. Temperature scaler behaviour
      31. Isotonic calibrator fit/apply
      32. S2 budget enforcement
      33. Gradient flow S1-only
      34. Activation functions
      35. Legacy adapter
      36. Param count
      37. Multi-GRU layers
      38. S2 trace structure
      39. High confidence skip
      40. Output dim mismatch error

    Returns:
        True if all tests pass, False otherwise.
    """
    passed = 0
    failed = 0
    errors: List[str] = []

    def _test(name: str, fn: Callable[[], None]) -> None:
        nonlocal passed, failed
        try:
            fn()
            passed += 1
            print(f"  [PASS] {name}")
        except Exception as e:
            failed += 1
            msg = f"  [FAIL] {name}: {e}"
            errors.append(msg)
            print(msg)

    print("=" * 70)
    print("DualProcessReasoner Self-Tests")
    print("=" * 70)

    device = torch.device("cpu")

    def _cfg() -> DualProcessFullConfig:
        return DualProcessFullConfig.minimal()

    def _make_model(cfg: Optional[DualProcessFullConfig] = None) -> DualProcessReasoner:
        return DualProcessReasoner(cfg or _cfg())

    # -------------------------------------------------------------------- #
    # Test 1: Forward with (B, D) input -- shapes correct
    # -------------------------------------------------------------------- #
    def test_forward_2d() -> None:
        torch.manual_seed(42)
        cfg = _cfg()
        model = _make_model(cfg)
        model.eval()
        B, D = 4, cfg.system1.input_dim
        x = torch.randn(B, D)
        out = model(x)

        assert out.y.shape == (B, cfg.system1.output_dim), \
            f"Expected y shape ({B}, {cfg.system1.output_dim}), got {out.y.shape}"
        assert out.used_system2.shape == (B,)
        assert out.s1 is not None
        assert out.s1.y1.shape == (B, cfg.system1.output_dim)
        assert out.s1.conf_raw.shape == (B,)
        assert out.s1.conf_calibrated.shape == (B,)
        assert out.s1.entropy.shape == (B,)
        assert out.s1.margin.shape == (B,)

    _test("1. Forward with (B, D) input: shapes correct", test_forward_2d)

    # -------------------------------------------------------------------- #
    # Test 2: Forward with (B, K, D) slots -- shapes correct
    # -------------------------------------------------------------------- #
    def test_forward_3d() -> None:
        torch.manual_seed(42)
        cfg = _cfg()
        model = _make_model(cfg)
        model.eval()
        B, K, D = 4, 7, cfg.system1.input_dim
        x = torch.randn(B, K, D)
        out = model(x)
        assert out.y.shape == (B, cfg.system1.output_dim)
        assert out.used_system2.shape == (B,)

    _test("2. Forward with (B, K, D) slots: shapes correct", test_forward_3d)

    # -------------------------------------------------------------------- #
    # Test 3: S1-only path -- all confident -> used_system2 all False
    # -------------------------------------------------------------------- #
    def test_s1_only() -> None:
        torch.manual_seed(42)
        cfg = _cfg()
        cfg.metacognition.route_threshold = 100.0
        cfg.metacognition.always_run_s2 = False
        model = _make_model(cfg)
        model.eval()
        B, D = 8, cfg.system1.input_dim
        x = torch.randn(B, D)
        out = model(x)
        assert not out.used_system2.any(), \
            f"Expected no S2 routing, got {out.used_system2.sum().item()} items"
        assert out.s2 is None

    _test("3. S1-only path: all confident -> used_system2 all False", test_s1_only)

    # -------------------------------------------------------------------- #
    # Test 4: S2 path -- all uncertain -> used_system2 all True
    # -------------------------------------------------------------------- #
    def test_all_s2() -> None:
        torch.manual_seed(42)
        cfg = _cfg()
        cfg.metacognition.always_run_s2 = True
        model = _make_model(cfg)
        model.eval()
        B, D = 4, cfg.system1.input_dim
        x = torch.randn(B, D)
        out = model(x)
        assert out.used_system2.all(), \
            f"Expected all S2, got {out.used_system2.sum().item()}/{B}"
        assert out.s2 is not None
        assert out.s2.y2.shape == (B, cfg.system1.output_dim)
        assert out.s2.steps_used.shape == (B,)
        assert out.s2.converged.shape == (B,)
        assert len(out.s2.halt_reason) == B

    _test("4. S2 path: all uncertain -> used_system2 all True", test_all_s2)

    # -------------------------------------------------------------------- #
    # Test 5: Mixed batch routing
    # -------------------------------------------------------------------- #
    def test_mixed_batch() -> None:
        torch.manual_seed(42)
        cfg = _cfg()
        cfg.metacognition.route_threshold = 0.5
        cfg.metacognition.min_conf_to_skip_s2 = 0.99
        cfg.metacognition.always_run_s2 = False
        model = _make_model(cfg)
        model.eval()
        B, D = 16, cfg.system1.input_dim
        x = torch.randn(B, D)
        out = model(x)
        n_s2 = out.used_system2.sum().item()
        assert out.y.shape == (B, cfg.system1.output_dim)
        if n_s2 > 0:
            assert out.s2 is not None
        if n_s2 == 0:
            assert out.s2 is None

    _test("5. Mixed batch: some confident, some uncertain", test_mixed_batch)

    # -------------------------------------------------------------------- #
    # Test 6: return_details=True -> trace is not None
    # -------------------------------------------------------------------- #
    def test_trace_returned() -> None:
        torch.manual_seed(42)
        cfg = _cfg()
        cfg.metacognition.always_run_s2 = True
        model = _make_model(cfg)
        model.eval()
        B, D = 4, cfg.system1.input_dim
        x = torch.randn(B, D)
        out = model(x, return_details=True)
        assert out.trace is not None
        assert len(out.trace) == B
        for t in out.trace:
            assert isinstance(t, ReasoningTrace)
            assert isinstance(t.used_system2, bool)
            assert isinstance(t.route_score, float)

    _test("6. return_details=True: trace is not None", test_trace_returned)

    # -------------------------------------------------------------------- #
    # Test 7: return_details=False -> trace is None
    # -------------------------------------------------------------------- #
    def test_no_trace() -> None:
        torch.manual_seed(42)
        model = _make_model()
        model.eval()
        B, D = 4, _cfg().system1.input_dim
        x = torch.randn(B, D)
        out = model(x, return_details=False)
        assert out.trace is None

    _test("7. return_details=False: trace is None", test_no_trace)

    # -------------------------------------------------------------------- #
    # Test 8: Trace JSON serialisation
    # -------------------------------------------------------------------- #
    def test_trace_json() -> None:
        torch.manual_seed(42)
        cfg = _cfg()
        cfg.metacognition.always_run_s2 = True
        model = _make_model(cfg)
        model.eval()
        B, D = 2, cfg.system1.input_dim
        x = torch.randn(B, D)
        out = model(x, return_details=True)
        assert out.trace is not None
        for t in out.trace:
            json_str = t.to_json()
            parsed = json.loads(json_str)
            assert isinstance(parsed, dict)
            assert "used_system2" in parsed
            assert "route_score" in parsed
            assert "s1_topk_vals" in parsed
            assert "s2_step_traces" in parsed
            assert isinstance(parsed["used_system2"], bool)
            assert isinstance(parsed["route_score"], float)
            assert isinstance(parsed["s1_topk_vals"], list)
            # Roundtrip
            rt = ReasoningTrace.from_dict(parsed)
            assert rt.used_system2 == parsed["used_system2"]

    _test("8. Trace JSON serialisation roundtrip", test_trace_json)

    # -------------------------------------------------------------------- #
    # Test 9: Determinism -- same input -> same output
    # -------------------------------------------------------------------- #
    def test_determinism() -> None:
        cfg = _cfg()
        cfg.metacognition.always_run_s2 = True
        B, D = 4, cfg.system1.input_dim

        results: List[ReasoningOutput] = []
        for _ in range(3):
            torch.manual_seed(99)
            model = _make_model(cfg)
            model.eval()
            x = torch.randn(B, D)
            torch.manual_seed(99)
            out = model(x)
            results.append(out)

        for i in range(1, len(results)):
            assert torch.allclose(results[0].y, results[i].y, atol=1e-6), \
                f"Run 0 vs {i}: max delta = {(results[0].y - results[i].y).abs().max()}"
            assert torch.equal(results[0].used_system2, results[i].used_system2)

    _test("9. Determinism: same input -> same output across 3 runs", test_determinism)

    # -------------------------------------------------------------------- #
    # Test 10: State management: reset -> forward with state
    # -------------------------------------------------------------------- #
    def test_state_management() -> None:
        torch.manual_seed(42)
        cfg = _cfg()
        cfg.metacognition.always_run_s2 = True
        model = _make_model(cfg)
        model.eval()
        B, D = 4, cfg.system1.input_dim
        state = model.reset(B, device)
        assert state.step_count == 0
        assert state.novelty_prototypes.shape == (
            cfg.metacognition.num_prototypes,
            cfg.system1.hidden_dim,
        )
        assert state.s2_hidden_carry is None

        x1 = torch.randn(B, D)
        _out1 = model(x1, state=state)
        assert state.step_count == 1

        x2 = torch.randn(B, D)
        _out2 = model(x2, state=state)
        assert state.step_count == 2

        detached = DualProcessReasoner.detach_state(state)
        assert detached.step_count == 2
        assert not detached.novelty_prototypes.requires_grad

    _test("10. State management: reset -> forward -> detach", test_state_management)

    # -------------------------------------------------------------------- #
    # Test 11: All config presets instantiate
    # -------------------------------------------------------------------- #
    def test_config_presets() -> None:
        presets = {
            "minimal": DualProcessFullConfig.minimal,
            "dev": DualProcessFullConfig.dev,
            "production_1b": DualProcessFullConfig.production_1b,
            "production_3b": DualProcessFullConfig.production_3b,
            "production_7b": DualProcessFullConfig.production_7b,
        }
        for name, factory in presets.items():
            cfg = factory()
            assert isinstance(cfg, DualProcessFullConfig)
            assert cfg.system1.output_dim == cfg.system2.output_dim, \
                f"{name}: S1 out ({cfg.system1.output_dim}) != S2 out ({cfg.system2.output_dim})"
            model = DualProcessReasoner(cfg)
            assert model.param_count() > 0

    _test("11. All config presets instantiate without error", test_config_presets)

    # -------------------------------------------------------------------- #
    # Test 12: Selective execution -- only uncertain items processed by S2
    # -------------------------------------------------------------------- #
    def test_selective_execution() -> None:
        torch.manual_seed(42)
        cfg = _cfg()
        cfg.metacognition.always_run_s2 = False
        cfg.metacognition.route_threshold = -100.0
        cfg.metacognition.min_conf_to_skip_s2 = 1.0
        model = _make_model(cfg)
        model.eval()
        B, D = 8, cfg.system1.input_dim
        x = torch.randn(B, D)
        out = model(x)
        assert out.used_system2.all()
        assert out.s2 is not None
        assert out.s2.y2.shape[0] == B

        cfg2 = _cfg()
        cfg2.metacognition.route_threshold = 100.0
        model2 = _make_model(cfg2)
        model2.eval()
        out2 = model2(x)
        assert not out2.used_system2.any()
        assert out2.s2 is None

    _test("12. Selective execution: S2 processes only routed items", test_selective_execution)

    # -------------------------------------------------------------------- #
    # Test 13: Gradient flow in training mode
    # -------------------------------------------------------------------- #
    def test_gradient_flow() -> None:
        torch.manual_seed(42)
        cfg = _cfg()
        cfg.metacognition.always_run_s2 = True
        model = _make_model(cfg)
        model.train()
        B, D = 4, cfg.system1.input_dim
        x = torch.randn(B, D, requires_grad=True)
        out = model(x)
        loss = out.y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
            if p.requires_grad
        )
        assert has_grad

    _test("13. Gradient flow in training mode (S2 active)", test_gradient_flow)

    # -------------------------------------------------------------------- #
    # Test 14: Aux metrics computed correctly
    # -------------------------------------------------------------------- #
    def test_aux_metrics() -> None:
        torch.manual_seed(42)
        cfg = _cfg()
        cfg.metacognition.always_run_s2 = True
        model = _make_model(cfg)
        model.eval()
        B, D = 8, cfg.system1.input_dim
        x = torch.randn(B, D)
        out = model(x)

        expected_keys = {
            "s2_fraction", "mean_route_score", "mean_novelty",
            "mean_s2_steps", "convergence_rate",
        }
        assert expected_keys.issubset(set(out.aux.keys()))
        assert abs(out.aux["s2_fraction"].item() - 1.0) < 1e-6
        assert out.aux["mean_s2_steps"].item() > 0
        cr = out.aux["convergence_rate"].item()
        assert 0.0 <= cr <= 1.0

        cfg2 = _cfg()
        cfg2.metacognition.route_threshold = 100.0
        model2 = _make_model(cfg2)
        model2.eval()
        out2 = model2(x)
        assert abs(out2.aux["s2_fraction"].item()) < 1e-6
        assert abs(out2.aux["mean_s2_steps"].item()) < 1e-6
        assert abs(out2.aux["convergence_rate"].item()) < 1e-6

    _test("14. Aux metrics computed correctly", test_aux_metrics)

    # -------------------------------------------------------------------- #
    # Test 15: Checkpoint save/load roundtrip
    # -------------------------------------------------------------------- #
    def test_checkpoint_roundtrip() -> None:
        torch.manual_seed(42)
        cfg = _cfg()
        cfg.metacognition.always_run_s2 = True
        model = _make_model(cfg)
        model.eval()
        B, D = 4, cfg.system1.input_dim
        x = torch.randn(B, D)
        out_before = model(x)

        buf = io.BytesIO()
        torch.save(
            {"model_state_dict": model.state_dict(), "config": cfg},
            buf,
        )
        buf.seek(0)
        checkpoint = torch.load(buf, map_location="cpu", weights_only=False)
        model2 = DualProcessReasoner(checkpoint["config"])
        model2.load_state_dict(checkpoint["model_state_dict"])
        model2.eval()

        out_after = model2(x)
        assert torch.allclose(out_before.y, out_after.y, atol=1e-6)
        assert torch.equal(out_before.used_system2, out_after.used_system2)

    _test("15. Checkpoint save/load roundtrip", test_checkpoint_roundtrip)

    # -------------------------------------------------------------------- #
    # Test 16: Batch size 1 edge case
    # -------------------------------------------------------------------- #
    def test_batch_size_one() -> None:
        torch.manual_seed(42)
        cfg = _cfg()
        cfg.metacognition.always_run_s2 = True
        model = _make_model(cfg)
        model.eval()
        D = cfg.system1.input_dim
        x = torch.randn(1, D)
        out = model(x, return_details=True)
        assert out.y.shape == (1, cfg.system1.output_dim)
        assert out.used_system2.shape == (1,)
        assert out.trace is not None and len(out.trace) == 1

    _test("16. Batch size 1 edge case", test_batch_size_one)

    # -------------------------------------------------------------------- #
    # Test 17: Empty context handling
    # -------------------------------------------------------------------- #
    def test_empty_context() -> None:
        torch.manual_seed(42)
        model = _make_model()
        model.eval()
        B, D = 4, _cfg().system1.input_dim
        x = torch.randn(B, D)
        out1 = model(x, context=None)
        assert out1.y.shape[0] == B
        out2 = model(x, context={})
        assert out2.y.shape[0] == B

    _test("17. Empty context handling (None and empty dict)", test_empty_context)

    # -------------------------------------------------------------------- #
    # Test 18: Context with anomaly, ignition, budget
    # -------------------------------------------------------------------- #
    def test_context_signals() -> None:
        torch.manual_seed(42)
        cfg = _cfg()
        model = _make_model(cfg)
        model.eval()
        B, D = 4, cfg.system1.input_dim
        x = torch.randn(B, D)
        context = {
            "anomaly": torch.rand(B),
            "ignition": torch.rand(B),
            "budget": torch.ones(B) * 0.5,
        }
        out = model(x, context=context)
        assert out.y.shape == (B, cfg.system1.output_dim)

    _test("18. Context with anomaly, ignition, budget signals", test_context_signals)

    # -------------------------------------------------------------------- #
    # Test 19: Factory function
    # -------------------------------------------------------------------- #
    def test_factory() -> None:
        model1 = create_dual_process_reasoner()
        assert isinstance(model1, DualProcessReasoner)
        cfg = DualProcessFullConfig.dev()
        model2 = create_dual_process_reasoner(cfg)
        assert model2.config.system1.input_dim == 512
        model3 = create_dual_process_reasoner(trace_top_k=10)
        assert model3._trace_top_k == 10

    _test("19. Factory function create_dual_process_reasoner", test_factory)

    # -------------------------------------------------------------------- #
    # Test 20: No NaN in output
    # -------------------------------------------------------------------- #
    def test_no_nan() -> None:
        torch.manual_seed(42)
        cfg = _cfg()
        cfg.metacognition.always_run_s2 = True
        model = _make_model(cfg)
        model.eval()
        D = cfg.system1.input_dim
        out1 = model(torch.randn(4, D))
        assert not torch.isnan(out1.y).any(), "NaN for normal input"
        out2 = model(torch.zeros(4, D))
        assert not torch.isnan(out2.y).any(), "NaN for zero input"
        out3 = model(torch.randn(4, D) * 10.0)
        assert not torch.isnan(out3.y).any(), "NaN for large input"

    _test("20. No NaN in output for normal/zero/large inputs", test_no_nan)

    # -------------------------------------------------------------------- #
    # Test 21: S2 convergence detection
    # -------------------------------------------------------------------- #
    def test_s2_convergence() -> None:
        torch.manual_seed(42)
        cfg = _cfg()
        cfg.metacognition.always_run_s2 = True
        cfg.system2.max_steps = 20
        cfg.system2.convergence_eps = 0.1
        cfg.system2.convergence_patience = 1
        model = _make_model(cfg)
        model.eval()
        B, D = 4, cfg.system1.input_dim
        x = torch.randn(B, D) * 0.01
        out = model(x)
        assert out.s2 is not None
        valid_reasons = {"converged", "max_steps", "nan_guard"}
        for reason in out.s2.halt_reason:
            assert reason in valid_reasons, f"Unexpected halt reason: {reason!r}"

    _test("21. S2 convergence detection: valid halt reasons", test_s2_convergence)

    # -------------------------------------------------------------------- #
    # Test 22: Calibrator freeze/unfreeze
    # -------------------------------------------------------------------- #
    def test_calibrator_freeze() -> None:
        cfg = _cfg()
        model = _make_model(cfg)
        cal_params = list(model.calibrator.parameters())
        if len(cal_params) > 0:
            model.calibrator.freeze()
            for p in model.calibrator.parameters():
                assert not p.requires_grad
            model.calibrator.unfreeze()
            for p in model.calibrator.parameters():
                assert p.requires_grad

    _test("22. Calibrator freeze/unfreeze", test_calibrator_freeze)

    # -------------------------------------------------------------------- #
    # Test 23: Deep supervision loss
    # -------------------------------------------------------------------- #
    def test_deep_supervision_loss() -> None:
        torch.manual_seed(42)
        cfg = _cfg()
        cfg.metacognition.always_run_s2 = True
        cfg.enable_deep_supervision = True
        model = _make_model(cfg)
        model.system2._enable_deep_supervision = True
        model.train()
        B, D = 4, cfg.system1.input_dim
        x = torch.randn(B, D)
        out = model(x)
        if out.s2 is not None and out.s2.deep_supervision_logits is not None:
            targets = torch.randint(0, cfg.system1.output_dim, (B,))
            loss = compute_deep_supervision_loss(out.s2, targets)
            assert loss.requires_grad
            assert not torch.isnan(loss)

    _test("23. Deep supervision loss computation", test_deep_supervision_loss)

    # -------------------------------------------------------------------- #
    # Test 24: Routing loss
    # -------------------------------------------------------------------- #
    def test_routing_loss() -> None:
        torch.manual_seed(42)
        cfg = _cfg()
        cfg.metacognition.always_run_s2 = True
        model = _make_model(cfg)
        model.eval()
        B, D = 8, cfg.system1.input_dim
        x = torch.randn(B, D)
        out = model(x)
        routing_dec = RoutingDecision(
            used_system2=out.used_system2,
            route_score=out.aux["mean_route_score"].expand(B),
            steps_budget=torch.full((B,), 3, dtype=torch.long),
            novelty=out.aux["mean_novelty"].expand(B),
        )
        s1_correct = torch.rand(B) > 0.5
        loss = compute_routing_loss(routing_dec, s1_correct)
        assert not torch.isnan(loss)

    _test("24. Routing loss computation", test_routing_loss)

    # -------------------------------------------------------------------- #
    # Test 25: S1 uncertainty metrics valid
    # -------------------------------------------------------------------- #
    def test_s1_metrics() -> None:
        torch.manual_seed(42)
        cfg = _cfg()
        model = _make_model(cfg)
        model.eval()
        B, D = 8, cfg.system1.input_dim
        x = torch.randn(B, D)
        out = model(x)
        assert (out.s1.conf_raw >= 0).all() and (out.s1.conf_raw <= 1).all()
        assert (out.s1.conf_calibrated >= 0).all() and (out.s1.conf_calibrated <= 1).all()
        assert (out.s1.entropy >= -1e-6).all()
        expected_k = min(cfg.trace_top_k, cfg.system1.output_dim)
        assert out.s1.logits_topk_vals.shape == (B, expected_k)
        assert out.s1.logits_topk_idx.shape == (B, expected_k)

    _test("25. S1 uncertainty metrics have valid ranges", test_s1_metrics)

    # -------------------------------------------------------------------- #
    # Test 26: Novelty prototype update
    # -------------------------------------------------------------------- #
    def test_novelty_update() -> None:
        torch.manual_seed(42)
        cfg = _cfg()
        cfg.metacognition.novelty_method = "prototype"
        model = _make_model(cfg)
        model.train()
        protos_before = model.novelty_scorer.prototypes.clone()
        B, D = 4, cfg.system1.input_dim
        _ = model(torch.randn(B, D))
        delta = (model.novelty_scorer.prototypes - protos_before).abs().sum().item()
        assert delta > 0, "Prototypes did not update"

    _test("26. Novelty prototype update during training", test_novelty_update)

    # -------------------------------------------------------------------- #
    # Test 27: Novelty entropy method
    # -------------------------------------------------------------------- #
    def test_novelty_entropy() -> None:
        torch.manual_seed(42)
        cfg = _cfg()
        cfg.metacognition.novelty_method = "entropy"
        model = _make_model(cfg)
        model.eval()
        B, D = 4, cfg.system1.input_dim
        out = model(torch.randn(B, D))
        assert out.aux["mean_novelty"].item() >= 0

    _test("27. Novelty scorer with entropy method", test_novelty_entropy)

    # -------------------------------------------------------------------- #
    # Test 28: Novelty none method
    # -------------------------------------------------------------------- #
    def test_novelty_none() -> None:
        torch.manual_seed(42)
        cfg = _cfg()
        cfg.metacognition.novelty_method = "none"
        model = _make_model(cfg)
        model.eval()
        out = model(torch.randn(4, cfg.system1.input_dim))
        assert abs(out.aux["mean_novelty"].item()) < 1e-6

    _test("28. Novelty scorer 'none' returns zeros", test_novelty_none)

    # -------------------------------------------------------------------- #
    # Test 29: All calibrator methods
    # -------------------------------------------------------------------- #
    def test_calibrator_methods() -> None:
        for method in ("temperature", "isotonic", "none"):
            cfg = _cfg()
            cfg.calibration.method = method
            model = _make_model(cfg)
            model.eval()
            B, D = 4, cfg.system1.input_dim
            out = model(torch.randn(B, D))
            assert out.y.shape == (B, cfg.system1.output_dim)

    _test("29. All calibrator methods (temperature, isotonic, none)", test_calibrator_methods)

    # -------------------------------------------------------------------- #
    # Test 30: Temperature scaler behaviour
    # -------------------------------------------------------------------- #
    def test_temperature_scaler() -> None:
        scaler = TemperatureScaler(initial_temperature=1.5)
        conf = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        calibrated = scaler(conf)
        assert calibrated.shape == conf.shape
        assert (calibrated >= 0).all() and (calibrated <= 1).all()
        assert calibrated[0] > conf[0], "Low conf should increase with T > 1"
        assert calibrated[4] < conf[4], "High conf should decrease with T > 1"

    _test("30. Temperature scaler calibration behaviour", test_temperature_scaler)

    # -------------------------------------------------------------------- #
    # Test 31: Isotonic calibrator fit/apply
    # -------------------------------------------------------------------- #
    def test_isotonic_calibrator() -> None:
        cal = IsotonicCalibrator(num_bins=10)
        torch.manual_seed(42)
        conf_raw = torch.rand(100)
        labels = (conf_raw + torch.randn(100) * 0.2 > 0.5).float()
        cal.fit(conf_raw, labels)
        assert cal.fitted.item()
        calibrated = cal(conf_raw[:10])
        assert calibrated.shape == (10,)
        assert (calibrated >= 0).all() and (calibrated <= 1).all()

    _test("31. Isotonic calibrator fit and apply", test_isotonic_calibrator)

    # -------------------------------------------------------------------- #
    # Test 32: S2 budget enforcement
    # -------------------------------------------------------------------- #
    def test_s2_budget_enforcement() -> None:
        torch.manual_seed(42)
        cfg = _cfg()
        cfg.metacognition.always_run_s2 = True
        cfg.system2.max_steps = 5
        model = _make_model(cfg)
        model.eval()
        B, D = 8, cfg.system1.input_dim
        out = model(torch.randn(B, D))
        assert out.s2 is not None
        assert (out.s2.steps_used <= cfg.system2.max_steps).all()
        assert (out.s2.steps_used >= 1).all()

    _test("32. S2 steps used within budget", test_s2_budget_enforcement)

    # -------------------------------------------------------------------- #
    # Test 33: Gradient flow S1-only
    # -------------------------------------------------------------------- #
    def test_gradient_s1_only() -> None:
        torch.manual_seed(42)
        cfg = _cfg()
        cfg.metacognition.route_threshold = 100.0
        model = _make_model(cfg)
        model.train()
        B, D = 4, cfg.system1.input_dim
        x = torch.randn(B, D, requires_grad=True)
        out = model(x)
        out.y.sum().backward()
        assert x.grad is not None and x.grad.abs().sum() > 0

    _test("33. Gradient flow through S1-only path", test_gradient_s1_only)

    # -------------------------------------------------------------------- #
    # Test 34: Activation functions
    # -------------------------------------------------------------------- #
    def test_activations() -> None:
        for act_name in ("relu", "gelu", "silu", "tanh"):
            cfg = _cfg()
            cfg.system1.activation = act_name
            model = _make_model(cfg)
            B, D = 2, cfg.system1.input_dim
            out = model(torch.randn(B, D))
            assert out.y.shape == (B, cfg.system1.output_dim)

    _test("34. All activation functions (relu, gelu, silu, tanh)", test_activations)

    # -------------------------------------------------------------------- #
    # Test 35: Legacy adapter
    # -------------------------------------------------------------------- #
    def test_legacy_adapter() -> None:
        @dataclass
        class LegacyCfg:
            hidden_dim: int = 256
            confidence_threshold: float = 0.7
            max_iterations: int = 8
            num_reasoning_steps: int = 5
            system1_layers: int = 2

        model = create_from_brain_ai_config(LegacyCfg(), input_dim=512, output_dim=128)
        assert isinstance(model, DualProcessReasoner)
        out = model(torch.randn(4, 512))
        assert out.y.shape == (4, 128)

    _test("35. create_from_brain_ai_config legacy adapter", test_legacy_adapter)

    # -------------------------------------------------------------------- #
    # Test 36: Param count
    # -------------------------------------------------------------------- #
    def test_param_count() -> None:
        model = _make_model()
        count = model.param_count()
        assert count > 0
        assert count < 10_000_000, f"Minimal config has {count} params (>10M)"

    _test("36. param_count is reasonable for minimal config", test_param_count)

    # -------------------------------------------------------------------- #
    # Test 37: Multi-GRU layers
    # -------------------------------------------------------------------- #
    def test_multi_gru_layers() -> None:
        torch.manual_seed(42)
        cfg = _cfg()
        cfg.system2.gru_layers = 2
        cfg.metacognition.always_run_s2 = True
        model = _make_model(cfg)
        model.eval()
        B, D = 4, cfg.system1.input_dim
        out = model(torch.randn(B, D))
        assert out.y.shape == (B, cfg.system1.output_dim)
        assert out.s2 is not None

    _test("37. Multiple S2 GRU layers (gru_layers=2)", test_multi_gru_layers)

    # -------------------------------------------------------------------- #
    # Test 38: S2 trace structure
    # -------------------------------------------------------------------- #
    def test_s2_traces_structure() -> None:
        torch.manual_seed(42)
        cfg = _cfg()
        cfg.metacognition.always_run_s2 = True
        cfg.system2.max_steps = 6
        model = _make_model(cfg)
        model.eval()
        B, D = 2, cfg.system1.input_dim
        out = model(torch.randn(B, D), return_details=True)
        assert out.trace is not None
        for i, t in enumerate(out.trace):
            assert t.used_system2
            assert t.s2_steps_used > 0
            assert len(t.s2_step_traces) > 0
            for st in t.s2_step_traces:
                assert isinstance(st.step, int)
                assert isinstance(st.conf, float)
                assert isinstance(st.kl_delta, float)
                assert isinstance(st.logits_topk_vals, list)
                assert isinstance(st.halt_check, bool)

    _test("38. S2 traces structure with return_details", test_s2_traces_structure)

    # -------------------------------------------------------------------- #
    # Test 39: High confidence skip
    # -------------------------------------------------------------------- #
    def test_high_conf_skip() -> None:
        torch.manual_seed(42)
        cfg = _cfg()
        cfg.metacognition.min_conf_to_skip_s2 = 0.0
        cfg.metacognition.route_threshold = -100.0
        cfg.metacognition.always_run_s2 = False
        model = _make_model(cfg)
        model.eval()
        out = model(torch.randn(4, cfg.system1.input_dim))
        assert not out.used_system2.any(), \
            f"Expected all skipped, got {out.used_system2.sum().item()} routed"

    _test("39. High confidence skip prevents S2 routing", test_high_conf_skip)

    # -------------------------------------------------------------------- #
    # Test 40: Output dim mismatch error
    # -------------------------------------------------------------------- #
    def test_output_dim_mismatch() -> None:
        cfg = _cfg()
        cfg.system1.output_dim = 64
        cfg.system2.output_dim = 128
        try:
            DualProcessReasoner(cfg)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "output_dim" in str(e).lower()

    _test("40. Output dim mismatch raises ValueError", test_output_dim_mismatch)

    # -------------------------------------------------------------------- #
    # Summary
    # -------------------------------------------------------------------- #
    print("=" * 70)
    print(f"Results: {passed}/{passed + failed} self-tests passed")
    if errors:
        print("\nFailed tests:")
        for e in errors:
            print(e)
    print("=" * 70)

    return failed == 0


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    success = _run_self_tests()
    raise SystemExit(0 if success else 1)
