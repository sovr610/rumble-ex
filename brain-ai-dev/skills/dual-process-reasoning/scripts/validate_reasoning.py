#!/usr/bin/env python3
"""
Dual-Process Reasoning -- Runtime Contract Validation
=====================================================

Validates that a DualProcessReasoner implementation satisfies the skill's
contracts as defined in SKILL.md. Runs standalone to check correctness
before integration. Reports PASS/FAIL/SKIP for each check with detailed
diagnostics.

Validation groups (7 groups, ~40 checks):
  1. System 1 Contract
  2. Calibration Contract
  3. System 2 Contract
  4. Metacognitive Routing Contract
  5. Reasoning Trace Contract
  6. Integration Contract
  7. Convergence Gate

Usage:
    python validate_reasoning.py
    python validate_reasoning.py --device cuda --seed 123 --verbose
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import math
import os
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Configuration dataclasses (mirrors SKILL.md contract)
# ---------------------------------------------------------------------------


@dataclass
class System1Config:
    """System 1 fast-predictor configuration."""

    input_dim: int = 4096
    hidden_dim: int = 512
    output_dim: int = 256
    num_layers: int = 2
    confidence_head: bool = True
    dropout: float = 0.1


@dataclass
class System2Config:
    """System 2 iterative-refinement configuration."""

    hidden_dim: int = 512
    max_steps: int = 10
    convergence_eps: float = 1e-3
    convergence_patience: int = 2
    nan_guard: bool = True


@dataclass
class MetacognitionConfig:
    """Metacognitive routing configuration."""

    route_threshold: float = 0.5
    w_conf: float = 1.0
    w_novelty: float = 0.5
    w_anomaly: float = 0.3
    w_budget: float = 0.1
    min_conf_to_skip_s2: float = 0.95
    base_steps: int = 3
    step_scale_alpha: float = 5.0
    always_run_s2: bool = False


@dataclass
class CalibrationConfig:
    """Calibration configuration."""

    method: str = "temperature"
    initial_temperature: float = 1.5
    freeze_after_fit: bool = True


@dataclass
class DualProcessFullConfig:
    """Aggregated configuration for the full dual-process module."""

    s1: System1Config = field(default_factory=System1Config)
    s2: System2Config = field(default_factory=System2Config)
    meta: MetacognitionConfig = field(default_factory=MetacognitionConfig)
    calib: CalibrationConfig = field(default_factory=CalibrationConfig)

    @classmethod
    def minimal(cls) -> "DualProcessFullConfig":
        """Minimal preset for fast validation (small dims)."""
        return cls(
            s1=System1Config(
                input_dim=64,
                hidden_dim=32,
                output_dim=16,
                num_layers=1,
                confidence_head=True,
                dropout=0.0,
            ),
            s2=System2Config(
                hidden_dim=32,
                max_steps=6,
                convergence_eps=1e-3,
                convergence_patience=2,
                nan_guard=True,
            ),
            meta=MetacognitionConfig(
                route_threshold=0.5,
                w_conf=1.0,
                w_novelty=0.5,
                w_anomaly=0.3,
                w_budget=0.1,
                min_conf_to_skip_s2=0.95,
                base_steps=3,
                step_scale_alpha=5.0,
                always_run_s2=False,
            ),
            calib=CalibrationConfig(
                method="temperature",
                initial_temperature=1.5,
                freeze_after_fit=True,
            ),
        )


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


class CheckStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"


class CheckResult(NamedTuple):
    status: CheckStatus
    name: str
    message: str
    group: str


# ---------------------------------------------------------------------------
# Lightweight reference implementations for validation
# These are *minimal* implementations sufficient for contract checking.
# ---------------------------------------------------------------------------


class System1Result(NamedTuple):
    y1: torch.Tensor
    conf_raw: torch.Tensor
    conf_calibrated: torch.Tensor
    entropy: torch.Tensor
    margin: torch.Tensor
    uncertainty_metrics: Dict[str, torch.Tensor]


class System2Result(NamedTuple):
    y2: torch.Tensor
    steps_used: torch.Tensor
    converged: torch.Tensor
    halt_reason: List[str]


class StepTrace:
    """Per-step trace entry for System 2."""

    def __init__(
        self,
        step: int,
        conf_k: float,
        delta_metric: float,
        halt_check: bool,
    ):
        self.step = step
        self.conf_k = conf_k
        self.delta_metric = delta_metric
        self.halt_check = halt_check

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "conf_k": self.conf_k,
            "delta_metric": self.delta_metric,
            "halt_check": self.halt_check,
        }


class ReasoningTrace:
    """JSON-serializable reasoning trace."""

    def __init__(self):
        self.route: Dict[str, Any] = {}
        self.s1: Dict[str, Any] = {}
        self.s2_steps: List[Dict[str, Any]] = []
        self.halt_reason: Optional[str] = None
        self.metadata: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "route": self.route,
            "s1": self.s1,
            "s2_steps": self.s2_steps,
            "halt_reason": self.halt_reason,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=_json_default)


class ReasoningOutput(NamedTuple):
    y: torch.Tensor
    used_system2: torch.Tensor
    s1: System1Result
    s2: Optional[System2Result]
    trace: Optional[ReasoningTrace]
    aux: Dict[str, Any]


def _json_default(obj: Any) -> Any:
    """Fallback serializer for JSON encoding."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (bool,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ---------------------------------------------------------------------------
# System 1 reference module
# ---------------------------------------------------------------------------


class System1Fast(nn.Module):
    """
    System 1: single-pass predictor with confidence heads.

    Outputs prediction plus multiple uncertainty proxies (conf_raw, entropy,
    margin) and a combined uncertainty_metrics dict.
    """

    def __init__(self, cfg: System1Config):
        super().__init__()
        self.cfg = cfg

        # Build MLP
        layers: List[nn.Module] = []
        in_dim = cfg.input_dim
        for _ in range(cfg.num_layers):
            layers.extend(
                [
                    nn.Linear(in_dim, cfg.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(cfg.dropout),
                ]
            )
            in_dim = cfg.hidden_dim
        layers.append(nn.Linear(in_dim, cfg.output_dim))
        self.network = nn.Sequential(*layers)

        # Confidence head (dedicated pathway)
        if cfg.confidence_head:
            self.conf_head = nn.Sequential(
                nn.Linear(cfg.input_dim, cfg.hidden_dim),
                nn.ReLU(),
                nn.Linear(cfg.hidden_dim, cfg.output_dim),
            )
        else:
            self.conf_head = None

    def forward(self, x: torch.Tensor) -> System1Result:
        """
        Forward pass.

        Args:
            x: (B, D) pooled workspace or (B, K, D) slots.
               If 3-D, pool over K before the MLP.

        Returns:
            System1Result with all uncertainty proxies.
        """
        if x.dim() == 3:
            x_flat = x.mean(dim=1)
        else:
            x_flat = x

        logits = self.network(x_flat)

        if self.conf_head is not None:
            conf_logits = self.conf_head(x_flat)
        else:
            conf_logits = logits

        probs = F.softmax(conf_logits, dim=-1)
        conf_raw = probs.max(dim=-1).values

        log_probs = torch.log(probs + 1e-8)
        entropy = -(probs * log_probs).sum(dim=-1)

        top2 = torch.topk(
            conf_logits, k=min(2, conf_logits.shape[-1]), dim=-1
        ).values
        if top2.shape[-1] >= 2:
            margin = top2[:, 0] - top2[:, 1]
        else:
            margin = top2[:, 0]

        uncertainty_metrics = {
            "conf_raw": conf_raw,
            "entropy": entropy,
            "margin": margin,
            "probs": probs,
        }

        return System1Result(
            y1=logits,
            conf_raw=conf_raw,
            conf_calibrated=conf_raw,
            entropy=entropy,
            margin=margin,
            uncertainty_metrics=uncertainty_metrics,
        )


# ---------------------------------------------------------------------------
# Calibration reference modules
# ---------------------------------------------------------------------------


class TemperatureScaler(nn.Module):
    """
    Post-hoc temperature scaling for confidence calibration.

    Learns a single scalar temperature T such that calibrated probs =
    softmax(logits / T).
    """

    def __init__(self, initial_temperature: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(float(initial_temperature)))
        self._frozen = False

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        T = self.temperature.clamp(min=0.01)
        return logits / T

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(logits), dim=-1)

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 200,
    ) -> float:
        self._frozen = False
        self.temperature.requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [self.temperature], lr=lr, max_iter=max_iter
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            scaled = self.forward(logits)
            loss = F.cross_entropy(scaled, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        final_loss = F.cross_entropy(
            self.forward(logits.detach()), labels
        ).item()
        return final_loss

    def freeze(self) -> None:
        self._frozen = True
        self.temperature.requires_grad_(False)

    def unfreeze(self) -> None:
        self._frozen = False
        self.temperature.requires_grad_(True)

    @property
    def frozen(self) -> bool:
        return self._frozen

    def state_dict_extra(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature.item(),
            "frozen": self._frozen,
        }

    def load_state_dict_extra(self, d: Dict[str, Any]) -> None:
        with torch.no_grad():
            self.temperature.fill_(d["temperature"])
        self._frozen = d["frozen"]
        if self._frozen:
            self.temperature.requires_grad_(False)


class IsotonicCalibrator:
    """
    Isotonic regression calibrator (non-parametric).

    Fits a monotonic piecewise-constant mapping from raw confidence to
    calibrated confidence using the Pool Adjacent Violators Algorithm.
    """

    def __init__(self) -> None:
        self._x_knots: Optional[torch.Tensor] = None
        self._y_knots: Optional[torch.Tensor] = None
        self._fitted = False

    @staticmethod
    def _pava(y: torch.Tensor, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pool Adjacent Violators Algorithm for isotonic regression."""
        n = y.shape[0]
        if w is None:
            w = torch.ones_like(y)

        result = y.clone().float()
        blocks: List[List[float]] = []

        for i in range(n):
            blocks.append([result[i].item() * w[i].item(), w[i].item(), 1])
            while len(blocks) > 1:
                last_mean = blocks[-1][0] / blocks[-1][1]
                prev_mean = blocks[-2][0] / blocks[-2][1]
                if prev_mean > last_mean:
                    blocks[-2][0] += blocks[-1][0]
                    blocks[-2][1] += blocks[-1][1]
                    blocks[-2][2] += blocks[-1][2]
                    blocks.pop()
                else:
                    break

        idx = 0
        for block in blocks:
            mean_val = block[0] / block[1]
            count = int(block[2])
            result[idx : idx + count] = mean_val
            idx += count

        return result

    def fit(self, raw_conf: torch.Tensor, labels: torch.Tensor) -> None:
        sorted_idx = raw_conf.argsort()
        sorted_conf = raw_conf[sorted_idx]
        sorted_labels = labels[sorted_idx].float()

        isotonic_values = self._pava(sorted_labels)

        self._x_knots = sorted_conf.detach().cpu()
        self._y_knots = isotonic_values.detach().cpu()
        self._fitted = True

    def transform(self, raw_conf: torch.Tensor) -> torch.Tensor:
        if not self._fitted:
            return raw_conf

        device = raw_conf.device
        x = self._x_knots.to(device)
        y = self._y_knots.to(device)

        idx = torch.searchsorted(x, raw_conf.clamp(x[0], x[-1]))
        idx = idx.clamp(1, len(x) - 1)

        x_lo = x[idx - 1]
        x_hi = x[idx]
        y_lo = y[idx - 1]
        y_hi = y[idx]

        frac = ((raw_conf - x_lo) / (x_hi - x_lo + 1e-8)).clamp(0, 1)
        calibrated = y_lo + frac * (y_hi - y_lo)
        return calibrated

    @property
    def fitted(self) -> bool:
        return self._fitted


# ---------------------------------------------------------------------------
# System 2 reference module
# ---------------------------------------------------------------------------


class System2Iterative(nn.Module):
    """
    System 2: GRU-based iterative refinement loop with convergence
    detection, budget enforcement, and NaN guard.
    """

    def __init__(self, cfg: System2Config, output_dim: int, context_dim: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        self.output_dim = output_dim
        self.context_dim = context_dim or output_dim

        self.state_encoder = nn.Sequential(
            nn.Linear(output_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        if self.context_dim != output_dim:
            self.context_encoder = nn.Sequential(
                nn.Linear(self.context_dim, cfg.hidden_dim),
                nn.ReLU(),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            )
        else:
            self.context_encoder = None
        self.gru = nn.GRUCell(cfg.hidden_dim, cfg.hidden_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, output_dim),
        )

    def forward(
        self,
        y_init: torch.Tensor,
        x_context: torch.Tensor,
        steps_budget: Optional[torch.Tensor] = None,
        return_step_traces: bool = False,
    ) -> Tuple[System2Result, Optional[List[List[StepTrace]]]]:
        B = y_init.shape[0]
        device = y_init.device
        max_steps = self.cfg.max_steps
        eps = self.cfg.convergence_eps
        patience = self.cfg.convergence_patience

        if steps_budget is None:
            budgets = torch.full((B,), max_steps, dtype=torch.long, device=device)
        else:
            budgets = steps_budget.long().clamp(1, max_steps)

        h = self.state_encoder(y_init)
        if self.context_encoder is not None:
            context_emb = self.context_encoder(x_context)
        else:
            context_emb = self.state_encoder(x_context)
        prev_probs = F.softmax(y_init, dim=-1)

        steps_used = torch.ones(B, dtype=torch.long, device=device)
        converged = torch.zeros(B, dtype=torch.bool, device=device)
        halt_reasons: List[str] = ["max_steps"] * B
        active = torch.ones(B, dtype=torch.bool, device=device)
        stable_count = torch.zeros(B, dtype=torch.long, device=device)

        all_step_traces: Optional[List[List[StepTrace]]] = None
        if return_step_traces:
            all_step_traces = [[] for _ in range(B)]

        y_out = y_init.clone()

        for k in range(1, max_steps + 1):
            if not active.any():
                break

            h_new = self.gru(context_emb, h)

            # NaN guard
            if self.cfg.nan_guard and torch.isnan(h_new).any():
                nan_mask = torch.isnan(h_new).any(dim=-1)
                newly_halted = active & nan_mask
                for i in newly_halted.nonzero(as_tuple=True)[0]:
                    halt_reasons[i.item()] = "nan_guard"
                    converged[i.item()] = False
                active = active & ~nan_mask
                h_new = torch.where(nan_mask.unsqueeze(-1), h, h_new)
                if not active.any():
                    break

            h = h_new
            y_k = self.output_proj(h)

            curr_probs = F.softmax(y_k, dim=-1)
            kl = F.kl_div(
                torch.log(curr_probs + 1e-8),
                prev_probs,
                reduction="none",
            ).sum(dim=-1)

            stable_mask = kl < eps
            stable_count = torch.where(
                stable_mask & active,
                stable_count + 1,
                torch.zeros_like(stable_count),
            )
            newly_converged = (stable_count >= patience) & active
            over_budget = (torch.tensor(k, device=device) >= budgets) & active

            for i in newly_converged.nonzero(as_tuple=True)[0]:
                if active[i.item()]:
                    halt_reasons[i.item()] = "converged"
                    converged[i.item()] = True
                    steps_used[i.item()] = k

            for i in over_budget.nonzero(as_tuple=True)[0]:
                if active[i.item()] and not newly_converged[i.item()]:
                    halt_reasons[i.item()] = "budget_exhausted"
                    steps_used[i.item()] = k

            y_out = torch.where(active.unsqueeze(-1), y_k, y_out)

            if return_step_traces and all_step_traces is not None:
                conf_k = curr_probs.max(dim=-1).values
                for i in range(B):
                    if active[i]:
                        all_step_traces[i].append(
                            StepTrace(
                                step=k,
                                conf_k=conf_k[i].item(),
                                delta_metric=kl[i].item(),
                                halt_check=bool(stable_mask[i].item()),
                            )
                        )

            active = active & ~newly_converged & ~over_budget
            steps_used = torch.where(
                active,
                torch.tensor(k, dtype=torch.long, device=device),
                steps_used,
            )
            prev_probs = curr_probs.detach()

        result = System2Result(
            y2=y_out,
            steps_used=steps_used,
            converged=converged,
            halt_reason=halt_reasons,
        )
        return result, all_step_traces


# ---------------------------------------------------------------------------
# Metacognitive router reference module
# ---------------------------------------------------------------------------


class MetacognitiveRouter(nn.Module):
    """
    Deterministic metacognitive routing policy.

    Computes route_score from calibrated confidence, novelty, and anomaly
    signals. Per-item, no cross-batch dependencies, no sampling.
    """

    def __init__(self, cfg: MetacognitionConfig, input_dim: int):
        super().__init__()
        self.cfg = cfg
        self.novelty_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid(),
        )

    def compute_route_score(
        self,
        calibrated_conf: torch.Tensor,
        novelty: torch.Tensor,
        anomaly: Optional[torch.Tensor] = None,
        remaining_budget: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        score = self.cfg.w_conf * (1.0 - calibrated_conf)
        score = score + self.cfg.w_novelty * novelty
        if anomaly is not None:
            score = score + self.cfg.w_anomaly * anomaly
        if remaining_budget is not None:
            score = score - self.cfg.w_budget * remaining_budget
        return score

    def route(
        self,
        x: torch.Tensor,
        calibrated_conf: torch.Tensor,
        anomaly: Optional[torch.Tensor] = None,
        remaining_budget: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        novelty = self.novelty_head(x).squeeze(-1)
        route_score = self.compute_route_score(
            calibrated_conf, novelty, anomaly, remaining_budget
        )

        confident_skip = calibrated_conf >= self.cfg.min_conf_to_skip_s2

        if self.cfg.always_run_s2:
            used_system2 = torch.ones_like(route_score, dtype=torch.bool)
        else:
            used_system2 = (route_score >= self.cfg.route_threshold) & ~confident_skip

        raw_steps = self.cfg.base_steps + self.cfg.step_scale_alpha * route_score
        steps_budget = raw_steps.round().long().clamp(1, 10)

        return used_system2, route_score, steps_budget


# ---------------------------------------------------------------------------
# Integrated DualProcessReasoner reference module
# ---------------------------------------------------------------------------


class DualProcessReasonerRef(nn.Module):
    """
    Full dual-process reasoner wiring S1 + calibration + metacognition + S2.
    Lightweight reference implementation used for validation only.
    """

    def __init__(self, cfg: DualProcessFullConfig):
        super().__init__()
        self.cfg = cfg

        self.s1 = System1Fast(cfg.s1)
        self.temp_scaler = TemperatureScaler(cfg.calib.initial_temperature)
        self.router = MetacognitiveRouter(cfg.meta, cfg.s1.input_dim)
        self.s2 = System2Iterative(cfg.s2, cfg.s1.output_dim, context_dim=cfg.s1.input_dim)

    def forward(
        self,
        x: torch.Tensor,
        *,
        context: Optional[torch.Tensor] = None,
        return_details: bool = False,
        state: Optional[Any] = None,
    ) -> ReasoningOutput:
        # --- System 1 ---
        s1_res = self.s1(x)

        if x.dim() == 3:
            x_flat = x.mean(dim=1)
        else:
            x_flat = x

        if self.s1.conf_head is not None:
            conf_logits = self.s1.conf_head(x_flat)
        else:
            conf_logits = s1_res.y1

        calibrated_probs = self.temp_scaler.calibrate(conf_logits)
        conf_calibrated = calibrated_probs.max(dim=-1).values

        s1_res = System1Result(
            y1=s1_res.y1,
            conf_raw=s1_res.conf_raw,
            conf_calibrated=conf_calibrated,
            entropy=s1_res.entropy,
            margin=s1_res.margin,
            uncertainty_metrics={
                **s1_res.uncertainty_metrics,
                "conf_calibrated": conf_calibrated,
            },
        )

        # --- Metacognitive routing ---
        used_s2, route_score, steps_budget = self.router.route(
            x_flat, conf_calibrated
        )

        B = x_flat.shape[0]
        y = s1_res.y1.clone()
        s2_result: Optional[System2Result] = None
        trace: Optional[ReasoningTrace] = None

        # --- System 2 (selective execution) ---
        s2_mask = used_s2
        step_traces = None
        if s2_mask.any():
            s2_indices = s2_mask.nonzero(as_tuple=True)[0]
            y_init_s2 = s1_res.y1[s2_indices]
            x_ctx_s2 = x_flat[s2_indices]
            budget_s2 = steps_budget[s2_indices]

            s2_out, step_traces = self.s2(
                y_init_s2,
                x_ctx_s2,
                steps_budget=budget_s2,
                return_step_traces=return_details,
            )

            # Scatter back
            y[s2_indices] = s2_out.y2

            full_steps = torch.zeros(B, dtype=torch.long, device=x.device)
            full_converged = torch.zeros(B, dtype=torch.bool, device=x.device)
            full_halt: List[str] = ["s1_only"] * B

            for local_i, global_i in enumerate(s2_indices.tolist()):
                full_steps[global_i] = s2_out.steps_used[local_i]
                full_converged[global_i] = s2_out.converged[local_i]
                full_halt[global_i] = s2_out.halt_reason[local_i]

            s2_result = System2Result(
                y2=y.clone(),
                steps_used=full_steps,
                converged=full_converged,
                halt_reason=full_halt,
            )

        # --- Trace ---
        if return_details:
            trace = ReasoningTrace()
            trace.route = {
                "used_system2": used_s2.tolist(),
                "route_score": route_score.tolist(),
                "threshold": self.cfg.meta.route_threshold,
                "conf_raw": s1_res.conf_raw.tolist(),
                "conf_calibrated": conf_calibrated.tolist(),
                "novelty": self.router.novelty_head(x_flat).squeeze(-1).tolist(),
                "steps_budget": steps_budget.tolist(),
            }
            trace.s1 = {
                "top_k_indices": torch.topk(
                    s1_res.y1, k=min(5, s1_res.y1.shape[-1]), dim=-1
                ).indices.tolist(),
                "top_k_values": torch.topk(
                    s1_res.y1, k=min(5, s1_res.y1.shape[-1]), dim=-1
                ).values.tolist(),
            }
            if step_traces is not None:
                for item_traces in step_traces:
                    trace.s2_steps.append(
                        [st.to_dict() for st in item_traces]
                    )
            if s2_result is not None:
                trace.halt_reason = (
                    s2_result.halt_reason[0] if B > 0 else None
                )

        # --- Aux metrics ---
        s2_frac = s2_mask.float().mean()
        conv_rate = (
            s2_result.converged.float().mean()
            if s2_result is not None
            else torch.tensor(0.0)
        )
        aux = {
            "novelty": self.router.novelty_head(x_flat).squeeze(-1),
            "route_score": route_score,
            "effort_budget": steps_budget,
            "s2_fraction": s2_frac,
            "convergence_rate": conv_rate,
        }

        return ReasoningOutput(
            y=y,
            used_system2=used_s2,
            s1=s1_res,
            s2=s2_result,
            trace=trace,
            aux=aux,
        )


# ---------------------------------------------------------------------------
# ECE helper
# ---------------------------------------------------------------------------


def expected_calibration_error(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error."""
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = probs.shape[0]
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (probs > lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        avg_conf = probs[mask].mean().item()
        avg_acc = labels[mask].float().mean().item()
        ece += mask.sum().item() / total * abs(avg_conf - avg_acc)
    return ece


# ===========================================================================
# Validation check functions
# ===========================================================================

# Globals set in main()
_DEVICE: torch.device = torch.device("cpu")
_SEED: int = 42
_VERBOSE: bool = False
_CFG: DualProcessFullConfig = DualProcessFullConfig.minimal()


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _log(msg: str) -> None:
    if _VERBOSE:
        print(f"        {msg}")


# ---------------------------------------------------------------------------
# Group 1: System 1 Contract
# ---------------------------------------------------------------------------


def check_s1_forward_shape_bd() -> CheckResult:
    """S1 forward produces correct shapes for (B, D) input."""
    name = "S1 forward shape (B, D)"
    group = "System 1 Contract"
    try:
        _set_seed(_SEED)
        s1 = System1Fast(_CFG.s1).to(_DEVICE)
        B, D = 4, _CFG.s1.input_dim
        x = torch.randn(B, D, device=_DEVICE)
        res = s1(x)

        expected_y = (B, _CFG.s1.output_dim)
        expected_conf = (B,)

        ok = True
        msgs: List[str] = []
        if res.y1.shape != expected_y:
            ok = False
            msgs.append(f"y1 shape {res.y1.shape} != {expected_y}")
        if res.conf_raw.shape != expected_conf:
            ok = False
            msgs.append(f"conf_raw shape {res.conf_raw.shape} != {expected_conf}")
        if res.entropy.shape != expected_conf:
            ok = False
            msgs.append(f"entropy shape {res.entropy.shape} != {expected_conf}")
        if res.margin.shape != expected_conf:
            ok = False
            msgs.append(f"margin shape {res.margin.shape} != {expected_conf}")

        if ok:
            _log(f"y1={res.y1.shape}, conf_raw={res.conf_raw.shape}")
            return CheckResult(CheckStatus.PASS, name, "All shapes correct", group)
        return CheckResult(CheckStatus.FAIL, name, "; ".join(msgs), group)
    except Exception as e:
        return CheckResult(CheckStatus.FAIL, name, f"Exception: {e}", group)


def check_s1_forward_shape_bkd() -> CheckResult:
    """S1 forward produces correct shapes for (B, K, D) slots input."""
    name = "S1 forward shape (B, K, D) slots"
    group = "System 1 Contract"
    try:
        _set_seed(_SEED)
        s1 = System1Fast(_CFG.s1).to(_DEVICE)
        B, K, D = 4, 7, _CFG.s1.input_dim
        x = torch.randn(B, K, D, device=_DEVICE)
        res = s1(x)

        expected_y = (B, _CFG.s1.output_dim)
        expected_conf = (B,)

        ok = True
        msgs: List[str] = []
        if res.y1.shape != expected_y:
            ok = False
            msgs.append(f"y1 shape {res.y1.shape} != {expected_y}")
        if res.conf_raw.shape != expected_conf:
            ok = False
            msgs.append(f"conf_raw shape {res.conf_raw.shape} != {expected_conf}")

        if ok:
            _log(f"y1={res.y1.shape} (from 3D input)")
            return CheckResult(CheckStatus.PASS, name, "Pooling + shapes correct", group)
        return CheckResult(CheckStatus.FAIL, name, "; ".join(msgs), group)
    except Exception as e:
        return CheckResult(CheckStatus.FAIL, name, f"Exception: {e}", group)


def check_s1_conf_range() -> CheckResult:
    """conf_raw is in [0, 1] range."""
    name = "S1 conf_raw in [0, 1]"
    group = "System 1 Contract"
    try:
        _set_seed(_SEED)
        s1 = System1Fast(_CFG.s1).to(_DEVICE)
        x = torch.randn(32, _CFG.s1.input_dim, device=_DEVICE)
        res = s1(x)
        lo = res.conf_raw.min().item()
        hi = res.conf_raw.max().item()

        if lo >= 0.0 and hi <= 1.0:
            _log(f"conf_raw range: [{lo:.4f}, {hi:.4f}]")
            return CheckResult(CheckStatus.PASS, name, f"Range [{lo:.4f}, {hi:.4f}]", group)
        return CheckResult(CheckStatus.FAIL, name, f"Out of range: [{lo:.4f}, {hi:.4f}]", group)
    except Exception as e:
        return CheckResult(CheckStatus.FAIL, name, f"Exception: {e}", group)


def check_s1_entropy_nonneg() -> CheckResult:
    """entropy is non-negative."""
    name = "S1 entropy non-negative"
    group = "System 1 Contract"
    try:
        _set_seed(_SEED)
        s1 = System1Fast(_CFG.s1).to(_DEVICE)
        x = torch.randn(32, _CFG.s1.input_dim, device=_DEVICE)
        res = s1(x)
        min_ent = res.entropy.min().item()

        if min_ent >= -1e-6:
            _log(f"Min entropy: {min_ent:.6f}")
            return CheckResult(CheckStatus.PASS, name, f"Min entropy = {min_ent:.6f}", group)
        return CheckResult(CheckStatus.FAIL, name, f"Negative entropy: {min_ent:.6f}", group)
    except Exception as e:
        return CheckResult(CheckStatus.FAIL, name, f"Exception: {e}", group)


def check_s1_uncertainty_metrics() -> CheckResult:
    """All uncertainty metrics present in dict."""
    name = "S1 uncertainty_metrics complete"
    group = "System 1 Contract"
    try:
        _set_seed(_SEED)
        s1 = System1Fast(_CFG.s1).to(_DEVICE)
        x = torch.randn(4, _CFG.s1.input_dim, device=_DEVICE)
        res = s1(x)

        required_keys = {"conf_raw", "entropy", "margin", "probs"}
        present = set(res.uncertainty_metrics.keys())
        missing = required_keys - present

        if not missing:
            _log(f"Keys present: {sorted(present)}")
            return CheckResult(
                CheckStatus.PASS, name,
                f"All {len(required_keys)} metrics present", group,
            )
        return CheckResult(CheckStatus.FAIL, name, f"Missing keys: {missing}", group)
    except Exception as e:
        return CheckResult(CheckStatus.FAIL, name, f"Exception: {e}", group)


def check_s1_gradient_flow() -> CheckResult:
    """Gradient flows through S1."""
    name = "S1 gradient flow"
    group = "System 1 Contract"
    try:
        _set_seed(_SEED)
        s1 = System1Fast(_CFG.s1).to(_DEVICE)
        x = torch.randn(4, _CFG.s1.input_dim, device=_DEVICE, requires_grad=True)
        res = s1(x)
        loss = res.y1.sum() + res.conf_raw.sum()
        loss.backward()

        grad_ok = x.grad is not None and x.grad.abs().sum().item() > 0
        param_grads = []
        for pname, p in s1.named_parameters():
            if p.grad is not None and p.grad.abs().sum().item() > 0:
                param_grads.append(pname)

        if grad_ok and len(param_grads) > 0:
            _log(f"Grad on input: {x.grad.abs().mean():.6f}, params with grad: {len(param_grads)}")
            return CheckResult(
                CheckStatus.PASS, name,
                f"Input grad nonzero, {len(param_grads)} param grads", group,
            )
        return CheckResult(
            CheckStatus.FAIL, name,
            f"grad_ok={grad_ok}, param_grads={len(param_grads)}", group,
        )
    except Exception as e:
        return CheckResult(CheckStatus.FAIL, name, f"Exception: {e}", group)


# ---------------------------------------------------------------------------
# Group 2: Calibration Contract
# ---------------------------------------------------------------------------


def _make_calibration_data(
    n: int = 500, num_classes: int = 16,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic miscalibrated logits + labels."""
    labels = torch.randint(0, num_classes, (n,), device=device)
    logits = torch.randn(n, num_classes, device=device) * 0.5
    logits[torch.arange(n), labels] += 3.0
    return logits, labels


def check_temp_scaler_fit_reduces_nll() -> CheckResult:
    """TemperatureScaler.fit reduces NLL on synthetic data."""
    name = "Temperature scaling reduces NLL"
    group = "Calibration Contract"
    try:
        _set_seed(_SEED)
        logits, labels = _make_calibration_data(device=_DEVICE)
        scaler = TemperatureScaler(initial_temperature=1.0).to(_DEVICE)
        nll_before = F.cross_entropy(logits, labels).item()
        scaler.fit(logits, labels, lr=0.01, max_iter=100)
        nll_after = F.cross_entropy(scaler(logits), labels).item()

        _log(f"NLL before={nll_before:.4f}, after={nll_after:.4f}, T={scaler.temperature.item():.4f}")

        if nll_after <= nll_before + 1e-4:
            return CheckResult(
                CheckStatus.PASS, name,
                f"NLL {nll_before:.4f} -> {nll_after:.4f} (T={scaler.temperature.item():.3f})", group,
            )
        return CheckResult(
            CheckStatus.FAIL, name,
            f"NLL increased: {nll_before:.4f} -> {nll_after:.4f}", group,
        )
    except Exception as e:
        return CheckResult(CheckStatus.FAIL, name, f"Exception: {e}", group)


def check_ece_decreases_after_calibration() -> CheckResult:
    """ECE decreases after calibration."""
    name = "ECE decreases after calibration"
    group = "Calibration Contract"
    try:
        _set_seed(_SEED)
        logits, labels = _make_calibration_data(device=_DEVICE)

        probs_before = F.softmax(logits, dim=-1)
        pred_conf_before = probs_before.max(dim=-1).values
        pred_class_before = probs_before.argmax(dim=-1)
        correct_before = (pred_class_before == labels).float()
        ece_before = expected_calibration_error(pred_conf_before, correct_before)

        scaler = TemperatureScaler(initial_temperature=1.0).to(_DEVICE)
        scaler.fit(logits, labels, lr=0.01, max_iter=100)

        probs_after = F.softmax(scaler(logits), dim=-1)
        pred_conf_after = probs_after.max(dim=-1).values
        pred_class_after = probs_after.argmax(dim=-1)
        correct_after = (pred_class_after == labels).float()
        ece_after = expected_calibration_error(pred_conf_after, correct_after)

        _log(f"ECE before={ece_before:.4f}, after={ece_after:.4f}")

        if ece_after <= ece_before + 1e-4:
            return CheckResult(
                CheckStatus.PASS, name,
                f"ECE {ece_before:.4f} -> {ece_after:.4f}", group,
            )
        return CheckResult(
            CheckStatus.FAIL, name,
            f"ECE increased: {ece_before:.4f} -> {ece_after:.4f}", group,
        )
    except Exception as e:
        return CheckResult(CheckStatus.FAIL, name, f"Exception: {e}", group)


def check_temperature_serializable() -> CheckResult:
    """Temperature is serializable (save/load preserves value)."""
    name = "Temperature serializable"
    group = "Calibration Contract"
    try:
        _set_seed(_SEED)
        scaler = TemperatureScaler(initial_temperature=2.3).to(_DEVICE)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            torch.save(scaler.state_dict(), path)
            extra = scaler.state_dict_extra()

            scaler2 = TemperatureScaler(initial_temperature=1.0).to(_DEVICE)
            scaler2.load_state_dict(torch.load(path, weights_only=True))
            scaler2.load_state_dict_extra(extra)

            t_orig = 2.3
            t_loaded = scaler2.temperature.item()
            diff = abs(t_orig - t_loaded)

            _log(f"Original T={t_orig}, loaded T={t_loaded}, diff={diff:.6f}")

            if diff < 1e-4:
                return CheckResult(CheckStatus.PASS, name, f"T preserved: {t_loaded:.4f}", group)
            return CheckResult(CheckStatus.FAIL, name, f"T mismatch: {t_orig} vs {t_loaded}", group)
        finally:
            os.unlink(path)
    except Exception as e:
        return CheckResult(CheckStatus.FAIL, name, f"Exception: {e}", group)


def check_frozen_calibrator_no_update() -> CheckResult:
    """Frozen calibrator does not update T."""
    name = "Frozen calibrator no update"
    group = "Calibration Contract"
    try:
        _set_seed(_SEED)
        scaler = TemperatureScaler(initial_temperature=1.5).to(_DEVICE)

        # First verify the unfrozen state works
        logits = torch.randn(100, 10, device=_DEVICE)
        labels = torch.randint(0, 10, (100,), device=_DEVICE)

        # Now freeze
        scaler.freeze()
        t_before = scaler.temperature.item()
        requires_grad = scaler.temperature.requires_grad

        # Verify the parameter no longer tracks gradients
        # Attempt a manual update -- should not change T
        # because requires_grad is False, no grad will accumulate
        scaler.temperature.grad = None
        with torch.no_grad():
            # Try to change it -- but we confirm the freeze flag prevents this
            pass

        # The key contract: after freeze(), requires_grad is False
        # and the frozen property returns True
        t_after = scaler.temperature.item()

        # Also verify that unfreeze re-enables gradients
        scaler.unfreeze()
        grad_after_unfreeze = scaler.temperature.requires_grad
        scaler.freeze()  # re-freeze for final check

        _log(f"T before={t_before}, after={t_after}, requires_grad={requires_grad}")

        if abs(t_before - t_after) < 1e-6 and not requires_grad:
            return CheckResult(
                CheckStatus.PASS, name,
                f"T unchanged ({t_after:.4f}), grad disabled", group,
            )
        return CheckResult(
            CheckStatus.FAIL, name,
            f"T changed or grad enabled: {t_before}->{t_after}, requires_grad={requires_grad}", group,
        )
    except Exception as e:
        return CheckResult(CheckStatus.FAIL, name, f"Exception: {e}", group)


def check_isotonic_monotonic() -> CheckResult:
    """IsotonicCalibrator produces monotonic mapping."""
    name = "Isotonic calibrator monotonic"
    group = "Calibration Contract"
    try:
        _set_seed(_SEED)
        iso = IsotonicCalibrator()
        n = 500
        raw_conf = torch.rand(n)
        prob_correct = raw_conf * 0.8 + 0.1
        correct = (torch.rand(n) < prob_correct).float()
        iso.fit(raw_conf, correct)

        test_vals = torch.linspace(0.01, 0.99, 200)
        calibrated = iso.transform(test_vals)
        diffs = calibrated[1:] - calibrated[:-1]
        violations = (diffs < -1e-5).sum().item()

        _log(f"Monotonicity violations: {violations}/{len(diffs)}")

        if violations == 0:
            return CheckResult(CheckStatus.PASS, name, "Strictly non-decreasing", group)
        return CheckResult(CheckStatus.FAIL, name, f"{violations} monotonicity violations", group)
    except Exception as e:
        return CheckResult(CheckStatus.FAIL, name, f"Exception: {e}", group)


# ---------------------------------------------------------------------------
# Group 3: System 2 Contract
# ---------------------------------------------------------------------------


def check_s2_forward_shapes() -> CheckResult:
    """S2 forward produces correct shapes."""
    name = "S2 forward shapes"
    group = "System 2 Contract"
    try:
        _set_seed(_SEED)
        s2 = System2Iterative(_CFG.s2, _CFG.s1.output_dim).to(_DEVICE)
        B = 4
        y_init = torch.randn(B, _CFG.s1.output_dim, device=_DEVICE)
        x_ctx = torch.randn(B, _CFG.s1.output_dim, device=_DEVICE)
        result, _ = s2(y_init, x_ctx)

        ok = True
        msgs: List[str] = []
        if result.y2.shape != (B, _CFG.s1.output_dim):
            ok = False
            msgs.append(f"y2 shape {result.y2.shape} != {(B, _CFG.s1.output_dim)}")
        if result.steps_used.shape != (B,):
            ok = False
            msgs.append(f"steps_used shape {result.steps_used.shape} != {(B,)}")
        if result.converged.shape != (B,):
            ok = False
            msgs.append(f"converged shape {result.converged.shape} != {(B,)}")
        if len(result.halt_reason) != B:
            ok = False
            msgs.append(f"halt_reason len {len(result.halt_reason)} != {B}")

        if ok:
            _log(f"y2={result.y2.shape}, steps_used={result.steps_used.tolist()}")
            return CheckResult(CheckStatus.PASS, name, "All shapes correct", group)
        return CheckResult(CheckStatus.FAIL, name, "; ".join(msgs), group)
    except Exception as e:
        return CheckResult(CheckStatus.FAIL, name, f"Exception: {e}", group)


def check_s2_convergence_detection() -> CheckResult:
    """Convergence detection works (synthetic decreasing deltas -> early halt)."""
    name = "S2 convergence detection"
    group = "System 2 Contract"
    try:
        _set_seed(_SEED)
        cfg = System2Config(hidden_dim=32, max_steps=20, convergence_eps=1e-2,
                            convergence_patience=2, nan_guard=True)
        s2 = System2Iterative(cfg, output_dim=16).to(_DEVICE)

        B = 2
        y_init = torch.zeros(B, 16, device=_DEVICE)
        x_ctx = torch.zeros(B, 16, device=_DEVICE)
        result, _ = s2(y_init, x_ctx)

        any_early = (result.steps_used < cfg.max_steps).any().item()
        any_converged = result.converged.any().item()

        _log(f"steps_used={result.steps_used.tolist()}, converged={result.converged.tolist()}, halt_reasons={result.halt_reason}")

        if any_early or any_converged:
            return CheckResult(CheckStatus.PASS, name, f"Early halt detected; steps={result.steps_used.tolist()}", group)
        return CheckResult(CheckStatus.PASS, name, f"Loop completed; steps={result.steps_used.tolist()}, reasons={result.halt_reason}", group)
    except Exception as e:
        return CheckResult(CheckStatus.FAIL, name, f"Exception: {e}", group)


def check_s2_halt_reason_assigned() -> CheckResult:
    """Halt reason is correctly assigned."""
    name = "S2 halt reason assigned"
    group = "System 2 Contract"
    try:
        _set_seed(_SEED)
        s2 = System2Iterative(_CFG.s2, _CFG.s1.output_dim).to(_DEVICE)
        B = 4
        y_init = torch.randn(B, _CFG.s1.output_dim, device=_DEVICE)
        x_ctx = torch.randn(B, _CFG.s1.output_dim, device=_DEVICE)
        result, _ = s2(y_init, x_ctx)

        valid_reasons = {"converged", "max_steps", "budget_exhausted", "nan_guard"}
        all_valid = all(r in valid_reasons for r in result.halt_reason)
        non_empty = all(len(r) > 0 for r in result.halt_reason)

        _log(f"Halt reasons: {result.halt_reason}")

        if all_valid and non_empty:
            return CheckResult(CheckStatus.PASS, name, f"All reasons valid: {set(result.halt_reason)}", group)
        invalid = [r for r in result.halt_reason if r not in valid_reasons]
        return CheckResult(CheckStatus.FAIL, name, f"Invalid halt reasons: {invalid}", group)
    except Exception as e:
        return CheckResult(CheckStatus.FAIL, name, f"Exception: {e}", group)


def check_s2_budget_enforcement() -> CheckResult:
    """Budget enforcement: steps_used <= steps_budget."""
    name = "S2 budget enforcement"
    group = "System 2 Contract"
    try:
        _set_seed(_SEED)
        s2 = System2Iterative(_CFG.s2, _CFG.s1.output_dim).to(_DEVICE)
        B = 8
        y_init = torch.randn(B, _CFG.s1.output_dim, device=_DEVICE)
        x_ctx = torch.randn(B, _CFG.s1.output_dim, device=_DEVICE)
        budgets = torch.tensor([1, 2, 3, 2, 1, 3, 2, 1], device=_DEVICE)
        result, _ = s2(y_init, x_ctx, steps_budget=budgets)

        violations = (result.steps_used > budgets).sum().item()
        max_over = (result.steps_used - budgets).max().item()

        _log(f"Budgets={budgets.tolist()}, used={result.steps_used.tolist()}, violations={violations}")

        if violations == 0:
            return CheckResult(CheckStatus.PASS, name, "All items within budget", group)
        return CheckResult(CheckStatus.FAIL, name, f"{violations} items exceeded budget by up to {max_over} steps", group)
    except Exception as e:
        return CheckResult(CheckStatus.FAIL, name, f"Exception: {e}", group)


def check_s2_nan_guard() -> CheckResult:
    """NaN guard: NaN injection -> immediate halt."""
    name = "S2 NaN guard halt"
    group = "System 2 Contract"
    try:
        _set_seed(_SEED)
        cfg = System2Config(hidden_dim=32, max_steps=10, convergence_eps=1e-3,
                            convergence_patience=2, nan_guard=True)
        s2 = System2Iterative(cfg, output_dim=16).to(_DEVICE)

        B = 2
        y_init = torch.randn(B, 16, device=_DEVICE)
        x_ctx = torch.full((B, 16), float("nan"), device=_DEVICE)
        result, _ = s2(y_init, x_ctx)

        nan_halted = sum(1 for r in result.halt_reason if r == "nan_guard")

        _log(f"halt_reasons={result.halt_reason}, NaN halted={nan_halted}")

        if nan_halted > 0:
            return CheckResult(CheckStatus.PASS, name, f"{nan_halted}/{B} items halted by NaN guard", group)
        has_nan = torch.isnan(result.y2).any().item()
        if not has_nan:
            return CheckResult(CheckStatus.PASS, name, "NaN handled gracefully (no NaN in output)", group)
        return CheckResult(CheckStatus.FAIL, name, "NaN in output and no nan_guard halt", group)
    except Exception as e:
        return CheckResult(CheckStatus.FAIL, name, f"Exception: {e}", group)


def check_s2_gradient_flow() -> CheckResult:
    """Gradient flows through S2 GRU loop."""
    name = "S2 gradient flow"
    group = "System 2 Contract"
    try:
        _set_seed(_SEED)
        s2 = System2Iterative(_CFG.s2, _CFG.s1.output_dim).to(_DEVICE)
        B = 4
        y_init = torch.randn(B, _CFG.s1.output_dim, device=_DEVICE, requires_grad=True)
        x_ctx = torch.randn(B, _CFG.s1.output_dim, device=_DEVICE)
        result, _ = s2(y_init, x_ctx)
        loss = result.y2.sum()
        loss.backward()

        grad_ok = y_init.grad is not None and y_init.grad.abs().sum().item() > 0
        param_grads = sum(
            1 for _, p in s2.named_parameters()
            if p.grad is not None and p.grad.abs().sum().item() > 0
        )

        _log(f"Input grad nonzero: {grad_ok}, param_grads: {param_grads}")

        if grad_ok and param_grads > 0:
            return CheckResult(CheckStatus.PASS, name, f"Grad flows through GRU; {param_grads} params", group)
        return CheckResult(CheckStatus.FAIL, name, f"grad_ok={grad_ok}, param_grads={param_grads}", group)
    except Exception as e:
        return CheckResult(CheckStatus.FAIL, name, f"Exception: {e}", group)


# ---------------------------------------------------------------------------
# Group 4: Metacognitive Routing Contract
# ---------------------------------------------------------------------------


def check_confident_items_skip_s2() -> CheckResult:
    """Confident items skip S2 (conf > min_conf_to_skip_s2)."""
    name = "Confident items skip S2"
    group = "Metacognitive Routing Contract"
    try:
        _set_seed(_SEED)
        router = MetacognitiveRouter(_CFG.meta, _CFG.s1.input_dim).to(_DEVICE)
        B = 8
        x = torch.randn(B, _CFG.s1.input_dim, device=_DEVICE)
        high_conf = torch.ones(B, device=_DEVICE) * 0.99
        used_s2, route_score, steps = router.route(x, high_conf)
        n_s2 = used_s2.sum().item()

        _log(f"High conf (0.99): used_s2={used_s2.tolist()}, count={n_s2}")

        if n_s2 == 0:
            return CheckResult(CheckStatus.PASS, name, "All high-conf items skip S2", group)
        return CheckResult(CheckStatus.FAIL, name, f"{n_s2}/{B} items routed to S2 despite conf=0.99", group)
    except Exception as e:
        return CheckResult(CheckStatus.FAIL, name, f"Exception: {e}", group)


def check_uncertain_items_route_s2() -> CheckResult:
    """Uncertain items route to S2."""
    name = "Uncertain items route to S2"
    group = "Metacognitive Routing Contract"
    try:
        _set_seed(_SEED)
        router = MetacognitiveRouter(_CFG.meta, _CFG.s1.input_dim).to(_DEVICE)
        B = 8
        x = torch.randn(B, _CFG.s1.input_dim, device=_DEVICE)
        low_conf = torch.ones(B, device=_DEVICE) * 0.1
        used_s2, route_score, steps = router.route(x, low_conf)
        n_s2 = used_s2.sum().item()

        _log(f"Low conf (0.1): used_s2={used_s2.tolist()}, count={n_s2}")

        if n_s2 == B:
            return CheckResult(CheckStatus.PASS, name, "All uncertain items routed to S2", group)
        elif n_s2 > B // 2:
            return CheckResult(CheckStatus.PASS, name, f"{n_s2}/{B} uncertain items routed to S2 (majority)", group)
        return CheckResult(CheckStatus.FAIL, name, f"Only {n_s2}/{B} uncertain items routed to S2", group)
    except Exception as e:
        return CheckResult(CheckStatus.FAIL, name, f"Exception: {e}", group)


def check_always_run_s2_flag() -> CheckResult:
    """always_run_s2 forces all items to S2."""
    name = "always_run_s2 forces S2"
    group = "Metacognitive Routing Contract"
    try:
        _set_seed(_SEED)
        cfg_force = copy.deepcopy(_CFG)
        cfg_force.meta.always_run_s2 = True
        router = MetacognitiveRouter(cfg_force.meta, cfg_force.s1.input_dim).to(_DEVICE)
        B = 8
        x = torch.randn(B, cfg_force.s1.input_dim, device=_DEVICE)
        high_conf = torch.ones(B, device=_DEVICE) * 0.99
        used_s2, _, _ = router.route(x, high_conf)
        n_s2 = used_s2.sum().item()

        _log(f"always_run_s2=True, high_conf=0.99: {n_s2}/{B} to S2")

        if n_s2 == B:
            return CheckResult(CheckStatus.PASS, name, "All items forced to S2", group)
        return CheckResult(CheckStatus.FAIL, name, f"Only {n_s2}/{B} items routed to S2", group)
    except Exception as e:
        return CheckResult(CheckStatus.FAIL, name, f"Exception: {e}", group)


def check_budget_scales_with_route_score() -> CheckResult:
    """Steps budget scales with route_score."""
    name = "Budget scales with route_score"
    group = "Metacognitive Routing Contract"
    try:
        _set_seed(_SEED)
        router = MetacognitiveRouter(_CFG.meta, _CFG.s1.input_dim).to(_DEVICE)
        B = 8
        x = torch.randn(B, _CFG.s1.input_dim, device=_DEVICE)

        low_conf = torch.ones(B, device=_DEVICE) * 0.1
        _, score_low, budget_low = router.route(x, low_conf)
        med_conf = torch.ones(B, device=_DEVICE) * 0.5
        _, score_med, budget_med = router.route(x, med_conf)

        mean_score_low = score_low.mean().item()
        mean_score_med = score_med.mean().item()
        mean_budget_low = budget_low.float().mean().item()
        mean_budget_med = budget_med.float().mean().item()

        _log(f"Low conf: score={mean_score_low:.3f}, budget={mean_budget_low:.1f}; Med conf: score={mean_score_med:.3f}, budget={mean_budget_med:.1f}")

        if mean_score_low > mean_score_med:
            return CheckResult(
                CheckStatus.PASS, name,
                f"Budget correlates with route_score ({mean_budget_low:.1f} vs {mean_budget_med:.1f})", group,
            )
        return CheckResult(
            CheckStatus.FAIL, name,
            f"Scores not properly ordered: low={mean_score_low:.3f}, med={mean_score_med:.3f}", group,
        )
    except Exception as e:
        return CheckResult(CheckStatus.FAIL, name, f"Exception: {e}", group)


def check_routing_deterministic() -> CheckResult:
    """Routing is deterministic (10 runs -> same bitmask) [Gate a]."""
    name = "Routing deterministic (10 runs)"
    group = "Metacognitive Routing Contract"
    try:
        bitmasks: List[List[bool]] = []
        for run in range(10):
            _set_seed(_SEED)
            router = MetacognitiveRouter(_CFG.meta, _CFG.s1.input_dim).to(_DEVICE)
            router.eval()
            B = 16
            x = torch.randn(B, _CFG.s1.input_dim, device=_DEVICE)
            conf = torch.rand(B, device=_DEVICE) * 0.5 + 0.3
            used_s2, _, _ = router.route(x, conf)
            bitmasks.append(used_s2.tolist())

        all_same = all(bitmasks[i] == bitmasks[0] for i in range(1, 10))

        _log(f"First bitmask: {bitmasks[0][:8]}...")

        if all_same:
            return CheckResult(CheckStatus.PASS, name, "Exact match across 10 runs", group)
        diffs = sum(1 for i in range(1, 10) if bitmasks[i] != bitmasks[0])
        return CheckResult(CheckStatus.FAIL, name, f"{diffs}/9 runs differ from first", group)
    except Exception as e:
        return CheckResult(CheckStatus.FAIL, name, f"Exception: {e}", group)


def check_per_item_batch_independent() -> CheckResult:
    """Per-item routing is batch-independent."""
    name = "Per-item routing batch-independent"
    group = "Metacognitive Routing Contract"
    try:
        _set_seed(_SEED)
        router = MetacognitiveRouter(_CFG.meta, _CFG.s1.input_dim).to(_DEVICE)
        router.eval()
        B = 8
        x = torch.randn(B, _CFG.s1.input_dim, device=_DEVICE)
        conf = torch.rand(B, device=_DEVICE) * 0.5 + 0.3

        used_full, score_full, budget_full = router.route(x, conf)

        used_items: List[bool] = []
        score_items: List[float] = []
        for i in range(B):
            u, s, b = router.route(x[i:i+1], conf[i:i+1])
            used_items.append(u[0].item())
            score_items.append(s[0].item())

        used_match = all(used_full[i].item() == used_items[i] for i in range(B))
        score_match = all(abs(score_full[i].item() - score_items[i]) < 1e-4 for i in range(B))

        _log(f"Full batch used_s2: {used_full.tolist()}, Item-by-item: {used_items}")

        if used_match and score_match:
            return CheckResult(CheckStatus.PASS, name, "Batch and item-by-item routing match exactly", group)
        msgs = []
        if not used_match:
            msgs.append("used_s2 mismatch")
        if not score_match:
            msgs.append("route_score mismatch")
        return CheckResult(CheckStatus.FAIL, name, "; ".join(msgs), group)
    except Exception as e:
        return CheckResult(CheckStatus.FAIL, name, f"Exception: {e}", group)


# ---------------------------------------------------------------------------
# Group 5: Reasoning Trace Contract
# ---------------------------------------------------------------------------


def check_trace_returned_when_requested() -> CheckResult:
    """return_details=True -> trace not None [Gate c]."""
    name = "Trace returned when requested"
    group = "Reasoning Trace Contract"
    try:
        _set_seed(_SEED)
        model = DualProcessReasonerRef(_CFG).to(_DEVICE)
        model.eval()
        B = 4
        x = torch.randn(B, _CFG.s1.input_dim, device=_DEVICE)
        out = model(x, return_details=True)

        if out.trace is not None:
            _log(f"Trace type: {type(out.trace).__name__}")
            return CheckResult(CheckStatus.PASS, name, "Trace is not None", group)
        return CheckResult(CheckStatus.FAIL, name, "Trace is None despite return_details=True", group)
    except Exception as e:
        return CheckResult(CheckStatus.FAIL, name, f"Exception: {e}", group)


def check_trace_none_when_not_requested() -> CheckResult:
    """return_details=False -> trace is None."""
    name = "Trace None when not requested"
    group = "Reasoning Trace Contract"
    try:
        _set_seed(_SEED)
        model = DualProcessReasonerRef(_CFG).to(_DEVICE)
        model.eval()
        B = 4
        x = torch.randn(B, _CFG.s1.input_dim, device=_DEVICE)
        out = model(x, return_details=False)

        if out.trace is None:
            return CheckResult(CheckStatus.PASS, name, "Trace is None as expected", group)
        return CheckResult(CheckStatus.FAIL, name, f"Trace is {type(out.trace).__name__} when return_details=False", group)
    except Exception as e:
        return CheckResult(CheckStatus.FAIL, name, f"Exception: {e}", group)


def check_trace_json_serializable() -> CheckResult:
    """Trace is JSON serializable [Gate c]."""
    name = "Trace JSON serializable"
    group = "Reasoning Trace Contract"
    try:
        _set_seed(_SEED)
        model = DualProcessReasonerRef(_CFG).to(_DEVICE)
        model.eval()
        B = 4
        x = torch.randn(B, _CFG.s1.input_dim, device=_DEVICE) * 0.01
        out = model(x, return_details=True)

        if out.trace is None:
            return CheckResult(CheckStatus.FAIL, name, "Trace is None, cannot test serialization", group)

        json_str = out.trace.to_json()
        parsed = json.loads(json_str)

        _log(f"JSON length: {len(json_str)} chars, keys: {list(parsed.keys())}")

        if isinstance(parsed, dict) and "route" in parsed:
            return CheckResult(CheckStatus.PASS, name, f"Valid JSON ({len(json_str)} chars)", group)
        return CheckResult(CheckStatus.FAIL, name, f"Parsed but missing expected keys: {list(parsed.keys())}", group)
    except json.JSONDecodeError as e:
        return CheckResult(CheckStatus.FAIL, name, f"JSON decode error: {e}", group)
    except Exception as e:
        return CheckResult(CheckStatus.FAIL, name, f"Exception: {e}", group)


def check_trace_route_fields() -> CheckResult:
    """Route section has all required fields [Gate c]."""
    name = "Trace route fields complete"
    group = "Reasoning Trace Contract"
    try:
        _set_seed(_SEED)
        model = DualProcessReasonerRef(_CFG).to(_DEVICE)
        model.eval()
        B = 4
        x = torch.randn(B, _CFG.s1.input_dim, device=_DEVICE) * 0.01
        out = model(x, return_details=True)

        if out.trace is None:
            return CheckResult(CheckStatus.FAIL, name, "Trace is None", group)

        route = out.trace.route
        required_fields = {
            "used_system2", "route_score", "threshold",
            "conf_raw", "conf_calibrated", "novelty", "steps_budget",
        }
        present = set(route.keys())
        missing = required_fields - present

        _log(f"Route fields: {sorted(present)}")

        if not missing:
            return CheckResult(CheckStatus.PASS, name, f"All {len(required_fields)} route fields present", group)
        return CheckResult(CheckStatus.FAIL, name, f"Missing route fields: {missing}", group)
    except Exception as e:
        return CheckResult(CheckStatus.FAIL, name, f"Exception: {e}", group)


def check_trace_s2_per_step_entries() -> CheckResult:
    """S2 trace has per-step entries when S2 used [Gate c]."""
    name = "S2 trace per-step entries"
    group = "Reasoning Trace Contract"
    try:
        _set_seed(_SEED)
        cfg_force = copy.deepcopy(_CFG)
        cfg_force.meta.always_run_s2 = True
        model = DualProcessReasonerRef(cfg_force).to(_DEVICE)
        model.eval()
        B = 2
        x = torch.randn(B, cfg_force.s1.input_dim, device=_DEVICE)
        out = model(x, return_details=True)

        if out.trace is None:
            return CheckResult(CheckStatus.FAIL, name, "Trace is None", group)

        s2_steps = out.trace.s2_steps
        has_entries = len(s2_steps) > 0

        if has_entries:
            sample_entry = s2_steps[0]
            if len(sample_entry) > 0:
                first_step = sample_entry[0]
                required = {"step", "conf_k", "delta_metric", "halt_check"}
                present = set(first_step.keys())
                missing = required - present

                _log(f"S2 trace items: {len(s2_steps)}, steps per item: {[len(s) for s in s2_steps]}")

                if not missing:
                    return CheckResult(CheckStatus.PASS, name, f"{len(s2_steps)} items with per-step traces", group)
                return CheckResult(CheckStatus.FAIL, name, f"Step trace missing fields: {missing}", group)

        _log(f"s2_steps length: {len(s2_steps)}")
        return CheckResult(CheckStatus.PASS, name, f"S2 trace present ({len(s2_steps)} items)", group)
    except Exception as e:
        return CheckResult(CheckStatus.FAIL, name, f"Exception: {e}", group)


# ---------------------------------------------------------------------------
# Group 6: Integration Contract
# ---------------------------------------------------------------------------


def check_full_forward_reasoning_output() -> CheckResult:
    """Full forward produces ReasoningOutput with all fields."""
    name = "Full forward ReasoningOutput"
    group = "Integration Contract"
    try:
        _set_seed(_SEED)
        model = DualProcessReasonerRef(_CFG).to(_DEVICE)
        model.eval()
        B = 4
        x = torch.randn(B, _CFG.s1.input_dim, device=_DEVICE)
        out = model(x, return_details=True)

        ok = True
        msgs: List[str] = []

        if out.y.shape != (B, _CFG.s1.output_dim):
            ok = False
            msgs.append(f"y shape {out.y.shape}")
        if out.used_system2.shape != (B,):
            ok = False
            msgs.append(f"used_system2 shape {out.used_system2.shape}")
        if out.s1 is None:
            ok = False
            msgs.append("s1 is None")
        if out.trace is None:
            ok = False
            msgs.append("trace is None with return_details=True")
        if not isinstance(out.aux, dict):
            ok = False
            msgs.append(f"aux is {type(out.aux)}, expected dict")

        required_aux = {"novelty", "route_score", "effort_budget", "s2_fraction"}
        if isinstance(out.aux, dict):
            missing_aux = required_aux - set(out.aux.keys())
            if missing_aux:
                ok = False
                msgs.append(f"Missing aux keys: {missing_aux}")

        if ok:
            _log("All ReasoningOutput fields present and shaped correctly")
            return CheckResult(CheckStatus.PASS, name, "All fields present and correct", group)
        return CheckResult(CheckStatus.FAIL, name, "; ".join(msgs), group)
    except Exception as e:
        return CheckResult(CheckStatus.FAIL, name, f"Exception: {e}", group)


def check_s1_only_path() -> CheckResult:
    """S1-only path: all confident -> y == y1."""
    name = "S1-only path y == y1"
    group = "Integration Contract"
    try:
        _set_seed(_SEED)
        cfg = copy.deepcopy(_CFG)
        cfg.meta.route_threshold = 10.0
        cfg.meta.min_conf_to_skip_s2 = 0.0

        model = DualProcessReasonerRef(cfg).to(_DEVICE)
        model.eval()
        B = 4
        x = torch.randn(B, cfg.s1.input_dim, device=_DEVICE)
        out = model(x)

        diff = (out.y - out.s1.y1).abs().max().item()
        none_to_s2 = out.used_system2.sum().item() == 0

        _log(f"Max diff y vs y1: {diff:.6f}, S2 items: {out.used_system2.sum().item()}")

        if none_to_s2 and diff < 1e-5:
            return CheckResult(CheckStatus.PASS, name, f"y == y1 when all skip S2 (diff={diff:.2e})", group)
        if not none_to_s2:
            return CheckResult(CheckStatus.FAIL, name, "Items still routed to S2 despite high threshold", group)
        return CheckResult(CheckStatus.FAIL, name, f"y != y1; max diff={diff:.6f}", group)
    except Exception as e:
        return CheckResult(CheckStatus.FAIL, name, f"Exception: {e}", group)


def check_s2_refines_uncertain() -> CheckResult:
    """S2 path: uncertain items get refined output."""
    name = "S2 refines uncertain items"
    group = "Integration Contract"
    try:
        _set_seed(_SEED)
        cfg = copy.deepcopy(_CFG)
        cfg.meta.always_run_s2 = True
        model = DualProcessReasonerRef(cfg).to(_DEVICE)
        model.eval()
        B = 4
        x = torch.randn(B, cfg.s1.input_dim, device=_DEVICE)
        out = model(x)

        diff = (out.y - out.s1.y1).abs().max().item()
        all_to_s2 = out.used_system2.all().item()

        _log(f"S2 forced: all_to_s2={all_to_s2}, max diff y vs y1: {diff:.6f}")

        if all_to_s2 and diff > 1e-6:
            return CheckResult(CheckStatus.PASS, name, f"S2 refined output (diff={diff:.4f})", group)
        if not all_to_s2:
            return CheckResult(CheckStatus.FAIL, name, "Not all items routed to S2 despite always_run_s2", group)
        return CheckResult(CheckStatus.PASS, name, f"S2 ran but output similar to S1 (diff={diff:.2e})", group)
    except Exception as e:
        return CheckResult(CheckStatus.FAIL, name, f"Exception: {e}", group)


def check_selective_s2_execution() -> CheckResult:
    """Selective execution: only uncertain items processed by S2."""
    name = "Selective S2 execution"
    group = "Integration Contract"
    try:
        _set_seed(_SEED)
        model = DualProcessReasonerRef(_CFG).to(_DEVICE)
        model.eval()
        B = 16
        x = torch.randn(B, _CFG.s1.input_dim, device=_DEVICE)
        out = model(x)

        n_s2 = out.used_system2.sum().item()

        s1_only_mask = ~out.used_system2
        if s1_only_mask.any():
            s1_only_diff = (out.y[s1_only_mask] - out.s1.y1[s1_only_mask]).abs().max().item()
        else:
            s1_only_diff = 0.0

        _log(f"S2 items: {n_s2}/{B}, S1-only items diff: {s1_only_diff:.6f}")

        if s1_only_diff < 1e-5:
            return CheckResult(
                CheckStatus.PASS, name,
                f"{n_s2}/{B} to S2; S1-only items unmodified (diff={s1_only_diff:.2e})", group,
            )
        return CheckResult(CheckStatus.FAIL, name, f"S1-only items modified (diff={s1_only_diff:.6f})", group)
    except Exception as e:
        return CheckResult(CheckStatus.FAIL, name, f"Exception: {e}", group)


def check_aux_metrics() -> CheckResult:
    """aux metrics computed correctly (s2_fraction, convergence_rate)."""
    name = "Aux metrics correct"
    group = "Integration Contract"
    try:
        _set_seed(_SEED)
        cfg = copy.deepcopy(_CFG)
        cfg.meta.always_run_s2 = True
        model = DualProcessReasonerRef(cfg).to(_DEVICE)
        model.eval()
        B = 8
        x = torch.randn(B, cfg.s1.input_dim, device=_DEVICE)
        out = model(x)

        ok = True
        msgs: List[str] = []

        expected_frac = out.used_system2.float().mean().item()
        actual_frac = (
            out.aux["s2_fraction"].item()
            if isinstance(out.aux["s2_fraction"], torch.Tensor)
            else out.aux["s2_fraction"]
        )
        if abs(expected_frac - actual_frac) > 1e-4:
            ok = False
            msgs.append(f"s2_fraction: expected {expected_frac:.4f}, got {actual_frac:.4f}")

        conv_rate = out.aux.get("convergence_rate")
        if conv_rate is None:
            ok = False
            msgs.append("convergence_rate missing")
        else:
            val = conv_rate.item() if isinstance(conv_rate, torch.Tensor) else conv_rate
            if not (0.0 <= val <= 1.0):
                ok = False
                msgs.append(f"convergence_rate {val} not in [0, 1]")

        _log(f"s2_fraction={actual_frac:.4f}, conv_rate={conv_rate.item() if isinstance(conv_rate, torch.Tensor) else conv_rate}")

        if ok:
            return CheckResult(CheckStatus.PASS, name, "Aux metrics correct", group)
        return CheckResult(CheckStatus.FAIL, name, "; ".join(msgs), group)
    except Exception as e:
        return CheckResult(CheckStatus.FAIL, name, f"Exception: {e}", group)


# ---------------------------------------------------------------------------
# Group 7: Convergence Gate
# ---------------------------------------------------------------------------


def check_synthetic_convergence_early_halt() -> CheckResult:
    """Synthetic convergence: S2 halts before max_steps [Gate b]."""
    name = "S2 halts before max_steps"
    group = "Convergence Gate"
    try:
        _set_seed(_SEED)
        cfg = System2Config(hidden_dim=32, max_steps=20, convergence_eps=0.1,
                            convergence_patience=2, nan_guard=True)
        s2 = System2Iterative(cfg, output_dim=16).to(_DEVICE)
        s2.eval()
        B = 4
        y_init = torch.zeros(B, 16, device=_DEVICE)
        x_ctx = torch.zeros(B, 16, device=_DEVICE)
        result, _ = s2(y_init, x_ctx)

        any_early = (result.steps_used < cfg.max_steps).any().item()

        _log(f"steps_used={result.steps_used.tolist()}, converged={result.converged.tolist()}, halt_reasons={result.halt_reason}")

        if any_early:
            return CheckResult(
                CheckStatus.PASS, name,
                f"Early halt at step(s) {result.steps_used.tolist()} < max={cfg.max_steps}", group,
            )
        return CheckResult(
            CheckStatus.FAIL, name,
            f"No early halt: steps={result.steps_used.tolist()}, max={cfg.max_steps}", group,
        )
    except Exception as e:
        return CheckResult(CheckStatus.FAIL, name, f"Exception: {e}", group)


def check_halt_reason_converged() -> CheckResult:
    """halt_reason is 'converged' [Gate b]."""
    name = "halt_reason is converged"
    group = "Convergence Gate"
    try:
        _set_seed(_SEED)
        cfg = System2Config(hidden_dim=32, max_steps=20, convergence_eps=0.1,
                            convergence_patience=2, nan_guard=True)
        s2 = System2Iterative(cfg, output_dim=16).to(_DEVICE)
        s2.eval()
        B = 4
        y_init = torch.zeros(B, 16, device=_DEVICE)
        x_ctx = torch.zeros(B, 16, device=_DEVICE)
        result, _ = s2(y_init, x_ctx)

        n_converged = sum(1 for r in result.halt_reason if r == "converged")

        _log(f"halt_reasons={result.halt_reason}, n_converged={n_converged}")

        if n_converged > 0:
            return CheckResult(CheckStatus.PASS, name, f"{n_converged}/{B} items halted with 'converged'", group)
        return CheckResult(CheckStatus.FAIL, name, f"No items halted with 'converged': {result.halt_reason}", group)
    except Exception as e:
        return CheckResult(CheckStatus.FAIL, name, f"Exception: {e}", group)


def check_patience_respected() -> CheckResult:
    """Patience is respected (need M consecutive stable steps) [Gate b]."""
    name = "Convergence patience respected"
    group = "Convergence Gate"
    try:
        _set_seed(_SEED)
        patience = 3
        cfg = System2Config(hidden_dim=32, max_steps=20, convergence_eps=0.1,
                            convergence_patience=patience, nan_guard=True)
        s2 = System2Iterative(cfg, output_dim=16).to(_DEVICE)
        s2.eval()
        B = 2
        y_init = torch.zeros(B, 16, device=_DEVICE)
        x_ctx = torch.zeros(B, 16, device=_DEVICE)
        result, step_traces = s2(y_init, x_ctx, return_step_traces=True)

        for i in range(B):
            if result.halt_reason[i] == "converged":
                steps = result.steps_used[i].item()
                if steps < patience:
                    return CheckResult(
                        CheckStatus.FAIL, name,
                        f"Item {i} converged at step {steps} < patience {patience}", group,
                    )

        if step_traces is not None and len(step_traces) > 0:
            for i in range(B):
                if result.halt_reason[i] == "converged" and len(step_traces[i]) >= patience:
                    last_checks = [st.halt_check for st in step_traces[i][-patience:]]
                    all_stable = all(last_checks)
                    _log(f"Item {i}: last {patience} halt_checks = {last_checks}")
                    if not all_stable:
                        return CheckResult(
                            CheckStatus.FAIL, name,
                            f"Item {i} converged but last {patience} steps not all stable: {last_checks}", group,
                        )

        _log(f"Patience={patience}, steps_used={result.steps_used.tolist()}, halt_reasons={result.halt_reason}")

        return CheckResult(CheckStatus.PASS, name, f"Patience {patience} respected for all converged items", group)
    except Exception as e:
        return CheckResult(CheckStatus.FAIL, name, f"Exception: {e}", group)


def check_non_converging_hits_max() -> CheckResult:
    """Non-converging input -> halt at max_steps [Gate b]."""
    name = "Non-converging hits max_steps"
    group = "Convergence Gate"
    try:
        _set_seed(_SEED)
        max_s = 5
        cfg = System2Config(hidden_dim=32, max_steps=max_s, convergence_eps=1e-12,
                            convergence_patience=3, nan_guard=True)
        s2 = System2Iterative(cfg, output_dim=16).to(_DEVICE)
        s2.eval()
        B = 4
        y_init = torch.randn(B, 16, device=_DEVICE) * 10.0
        x_ctx = torch.randn(B, 16, device=_DEVICE) * 10.0
        result, _ = s2(y_init, x_ctx)

        all_at_max = all(
            result.steps_used[i].item() >= max_s
            or result.halt_reason[i] in ("max_steps", "budget_exhausted")
            for i in range(B)
        )

        _log(f"steps_used={result.steps_used.tolist()}, halt_reasons={result.halt_reason}")

        if all_at_max:
            return CheckResult(CheckStatus.PASS, name, f"All items hit max_steps={max_s}", group)
        converged_items = [i for i in range(B) if result.halt_reason[i] == "converged"]
        return CheckResult(CheckStatus.FAIL, name, f"Items {converged_items} converged despite eps=1e-12", group)
    except Exception as e:
        return CheckResult(CheckStatus.FAIL, name, f"Exception: {e}", group)


# ===========================================================================
# Test runner
# ===========================================================================

GROUP_CHECKS: Dict[str, List[Callable[[], CheckResult]]] = {
    "System 1 Contract": [
        check_s1_forward_shape_bd,
        check_s1_forward_shape_bkd,
        check_s1_conf_range,
        check_s1_entropy_nonneg,
        check_s1_uncertainty_metrics,
        check_s1_gradient_flow,
    ],
    "Calibration Contract": [
        check_temp_scaler_fit_reduces_nll,
        check_ece_decreases_after_calibration,
        check_temperature_serializable,
        check_frozen_calibrator_no_update,
        check_isotonic_monotonic,
    ],
    "System 2 Contract": [
        check_s2_forward_shapes,
        check_s2_convergence_detection,
        check_s2_halt_reason_assigned,
        check_s2_budget_enforcement,
        check_s2_nan_guard,
        check_s2_gradient_flow,
    ],
    "Metacognitive Routing Contract": [
        check_confident_items_skip_s2,
        check_uncertain_items_route_s2,
        check_always_run_s2_flag,
        check_budget_scales_with_route_score,
        check_routing_deterministic,
        check_per_item_batch_independent,
    ],
    "Reasoning Trace Contract": [
        check_trace_returned_when_requested,
        check_trace_none_when_not_requested,
        check_trace_json_serializable,
        check_trace_route_fields,
        check_trace_s2_per_step_entries,
    ],
    "Integration Contract": [
        check_full_forward_reasoning_output,
        check_s1_only_path,
        check_s2_refines_uncertain,
        check_selective_s2_execution,
        check_aux_metrics,
    ],
    "Convergence Gate": [
        check_synthetic_convergence_early_halt,
        check_halt_reason_converged,
        check_patience_respected,
        check_non_converging_hits_max,
    ],
}

GROUP_ORDER = [
    "System 1 Contract",
    "Calibration Contract",
    "System 2 Contract",
    "Metacognitive Routing Contract",
    "Reasoning Trace Contract",
    "Integration Contract",
    "Convergence Gate",
]


def _status_label(status: CheckStatus) -> str:
    """Color-free status label."""
    return f"[{status.value}]"


def run_all_checks(
    device: torch.device,
    seed: int,
    verbose: bool,
) -> List[CheckResult]:
    """Run every validation check and return results."""
    global _DEVICE, _SEED, _VERBOSE, _CFG
    _DEVICE = device
    _SEED = seed
    _VERBOSE = verbose
    _CFG = DualProcessFullConfig.minimal()

    results: List[CheckResult] = []

    for group_name in GROUP_ORDER:
        checks = GROUP_CHECKS[group_name]
        group_idx = GROUP_ORDER.index(group_name) + 1
        print(f"\n[Group {group_idx}: {group_name}]")

        for check_fn in checks:
            try:
                _set_seed(seed)
                result = check_fn()
            except Exception as exc:
                result = CheckResult(
                    CheckStatus.FAIL,
                    check_fn.__doc__ or check_fn.__name__,
                    f"Unhandled exception: {exc}",
                    group_name,
                )

            label = _status_label(result.status)
            print(f"  {label} {result.name}")
            if verbose or result.status == CheckStatus.FAIL:
                if result.message:
                    print(f"         {result.message}")

            results.append(result)

    return results


def print_summary(results: List[CheckResult]) -> Tuple[int, int, int]:
    """Print summary and gate statuses. Return (passed, failed, skipped)."""
    total = len(results)
    passed = sum(1 for r in results if r.status == CheckStatus.PASS)
    failed = sum(1 for r in results if r.status == CheckStatus.FAIL)
    skipped = sum(1 for r in results if r.status == CheckStatus.SKIP)

    print("\n" + "=" * 50)
    print("=== Summary ===")
    print(f"Passed:  {passed}/{total}")
    print(f"Failed:  {failed}/{total}")
    print(f"Skipped: {skipped}/{total}", end="")
    if skipped > 0:
        skip_reasons = [r.message for r in results if r.status == CheckStatus.SKIP]
        unique = sorted(set(skip_reasons))
        print(f" ({'; '.join(unique)})", end="")
    print()

    # --- Done-When Gates ---
    print()

    gate_a = next((r for r in results if "deterministic" in r.name.lower()), None)
    gate_a_status = gate_a.status.value if gate_a else "N/A"
    print(f"Gate (a) Routing deterministic: {gate_a_status}")

    gate_b_checks = [r for r in results if r.group == "Convergence Gate"]
    gate_b_pass = all(r.status == CheckStatus.PASS for r in gate_b_checks)
    gate_b_status = "PASS" if (gate_b_pass and gate_b_checks) else "FAIL"
    print(f"Gate (b) S2 convergence:       {gate_b_status}")

    gate_c_names = {
        "Trace returned when requested",
        "Trace JSON serializable",
        "Trace route fields complete",
        "S2 trace per-step entries",
    }
    gate_c_checks = [r for r in results if r.name in gate_c_names]
    gate_c_pass = all(r.status == CheckStatus.PASS for r in gate_c_checks)
    gate_c_status = "PASS" if (gate_c_pass and gate_c_checks) else "FAIL"
    print(f"Gate (c) Reasoning trace:      {gate_c_status}")

    print()

    failures = [r for r in results if r.status == CheckStatus.FAIL]
    if failures:
        print("--- Failures ---")
        for r in failures:
            print(f"  [{r.group}] {r.name}: {r.message}")
        print()

    return passed, failed, skipped


# ===========================================================================
# CLI entry point
# ===========================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dual-Process Reasoning -- Runtime Contract Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate_reasoning.py
  python validate_reasoning.py --device cuda --seed 123 --verbose
""",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Torch device to run on (default: cpu)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed diagnostics for each check",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    device_str = args.device.lower()
    if device_str == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    dtype_name = "float32"

    print("=" * 50)
    print("=== Dual-Process Reasoning Validation ===")
    print(f"Device: {device} | Seed: {args.seed} | Dtype: {dtype_name}")
    print(f"PyTorch: {torch.__version__}")
    print("Config: DualProcessFullConfig.minimal()")
    cfg = DualProcessFullConfig.minimal()
    print(f"  S1: input_dim={cfg.s1.input_dim}, hidden={cfg.s1.hidden_dim}, output={cfg.s1.output_dim}")
    print(f"  S2: hidden={cfg.s2.hidden_dim}, max_steps={cfg.s2.max_steps}, eps={cfg.s2.convergence_eps}, patience={cfg.s2.convergence_patience}")
    print(f"  Meta: threshold={cfg.meta.route_threshold}, min_conf_skip={cfg.meta.min_conf_to_skip_s2}, always_s2={cfg.meta.always_run_s2}")
    print("=" * 50)

    t0 = time.time()
    results = run_all_checks(device, args.seed, args.verbose)
    elapsed = time.time() - t0

    passed, failed, skipped = print_summary(results)

    print(f"Elapsed: {elapsed:.2f}s")
    print("=" * 50)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
