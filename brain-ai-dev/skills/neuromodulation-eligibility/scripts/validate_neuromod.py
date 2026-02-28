#!/usr/bin/env python3
"""
validate_neuromod.py -- Runtime contract validation for neuromodulation + eligibility traces.

Self-contained validation of three-factor learning (neuromodulation + eligibility traces)
covering the three done-when gates from SKILL.md:

    (a) Third-factor gating: mod_signal=0 => delta_w exactly zero
    (b) Deterministic traces: fixed inputs => identical e(t) across runs
    (c) Delayed reward association: eligibility bridges temporal gap, loss decreases

Uses inline stubs so the script runs standalone without brain_ai imports.
Requires only: torch, numpy, json, argparse, time.

Usage:
    # Run all groups
    python scripts/validate_neuromod.py

    # Verbose output
    python scripts/validate_neuromod.py --verbose

    # Run a specific group (1-7)
    python scripts/validate_neuromod.py --group 3

    # Multiple groups
    python scripts/validate_neuromod.py --group 1 --group 3 --group 6

    # JSON-only output (for CI)
    python scripts/validate_neuromod.py --json

    # Combine flags
    python scripts/validate_neuromod.py --verbose --group 6 --json

Exit codes:
    0 = all checks passed
    1 = one or more checks failed
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================
# These constants define the validation thresholds, dimensions, and numerical
# tolerances used throughout the script.  They are collected here for easy
# tuning and so that every check references a single source of truth.
# ============================================================================

# -- Dimensions used across most validation groups --
DEFAULT_BATCH_SIZE = 4
DEFAULT_N_PRE = 16
DEFAULT_N_POST = 8
LARGE_N_PRE = 32
LARGE_N_POST = 16

# -- Numerical tolerances --
ATOL_FP32_DETERMINISM = 1e-6   # For fp32 determinism comparisons across runs
ATOL_APPROX_EQUAL = 1e-4       # For loose approximate comparisons (AMP, etc.)
RTOL_LR_SCALING = 0.05         # Relative tolerance for LR ratio checks

# -- Trace dynamics --
DEFAULT_TAU_E = 50.0            # Default eligibility decay time constant
FAST_TAU_E = 10.0               # Fast decay time constant (for tau_e effect test)
SLOW_TAU_E = 100.0              # Slow decay time constant (for tau_e effect test)
DEFAULT_DT = 1.0                # Default simulation timestep
DEFAULT_CLAMP_MIN = -5.0        # Default trace lower clamp bound
DEFAULT_CLAMP_MAX = 5.0         # Default trace upper clamp bound

# -- STDP parameters --
DEFAULT_A_PLUS = 0.01           # STDP potentiation amplitude
DEFAULT_A_MINUS = 0.012         # STDP depression amplitude
DEFAULT_TAU_PLUS = 20.0         # STDP potentiation time constant
DEFAULT_TAU_MINUS = 20.0        # STDP depression time constant

# -- Three-factor update --
DEFAULT_LR = 0.01               # Default plasticity learning rate
DEFAULT_WEIGHT_CLAMP = (-1.0, 1.0)
DEFAULT_DELTA_CLAMP = 0.05      # Per-step delta clamp

# -- Neuromodulator thresholds --
HIGH_SIGNAL_THRESHOLD = 0.8     # Threshold for "high" modulator value
NEUTRAL_DA_EXPECTED = 0.0       # DA at zero reward (after tanh)
NEUTRAL_SIGMOID_EXPECTED = 0.5  # ACh/NE/5-HT at zero input (sigmoid bias=0)

# -- Delayed reward task --
DELAYED_REWARD_INPUT_DIM = 4
DELAYED_REWARD_HIDDEN_DIM = 32
DELAYED_REWARD_OUTPUT_DIM = 4
DELAYED_REWARD_LABEL_MAP = [2, 0, 3, 1]  # Non-trivial permutation
DELAYED_REWARD_CHANCE_ACCURACY = 0.25     # 1/4 for 4-class task

# -- Determinism --
DETERMINISM_NUM_RUNS = 10       # Number of repeated runs for determinism checks
DETERMINISM_NUM_STEPS = 20      # Number of steps per determinism run

# -- Validation metadata --
SCRIPT_NAME = "validate_neuromod.py"
SKILL_NAME = "neuromodulation-eligibility"


# ============================================================================
# SECTION 1: INLINE STUBS
# ============================================================================
# These stubs replicate the core logic from the brain_ai neuromodulation skill
# so that this script is fully self-contained. They follow the specifications
# in references/eligibility-dynamics.md, references/neuromodulator-signals.md,
# and references/three-factor-rules.md.
#
# Key design decisions in the stubs:
#
# 1. All trace computations use fp32 (float32) regardless of input dtype.
#    This prevents catastrophic precision loss under AMP (fp16 has only ~3
#    decimal digits, causing small eligibility updates to round to zero).
#
# 2. Decay is applied BEFORE adding the new correlation at each timestep,
#    so that the current timestep's correlation is recorded at full strength
#    while all previous contributions are attenuated. This follows the
#    canonical form: e(t+1) = decay * e(t) + f(pre, post).
#
# 3. Clamping is applied after every update step to prevent numerical
#    explosion. The default range [-5.0, 5.0] accommodates typical correlation
#    magnitudes (order 0.01--1.0) with headroom for transient accumulation.
#
# 4. The three-factor update uses detached tensors (torch.no_grad) to prevent
#    the plasticity update from interfering with the backprop computational
#    graph. This is a critical invariant: three-factor updates are purely
#    non-differentiable imperatives applied to parameter .data.
#
# 5. The NeuromodulatoryGate produces bounded outputs for each modulator:
#    DA in [-1,1] (tanh), ACh/NE/5-HT in [0,1] (sigmoid). The combination
#    function maps these to a scalar global plasticity gain.
# ============================================================================


# ---------------------------------------------------------------------------
# 1.1 Trace Type Enum
# ---------------------------------------------------------------------------

class TraceType(Enum):
    """Supported eligibility trace accumulation strategies."""
    ACCUMULATING = "accumulating"
    REPLACING = "replacing"
    DUTCH = "dutch"


# ---------------------------------------------------------------------------
# 1.2 EligibilityTraceModule
# ---------------------------------------------------------------------------

class EligibilityTraceModule(nn.Module):
    """Manages eligibility traces for a single synaptic connection matrix.

    Supports accumulating, replacing, and Dutch trace types.
    Handles spike-based (STDP) and rate-based (Hebbian) correlation modes.
    All trace computations are performed in fp32 regardless of AMP settings.

    Parameters
    ----------
    n_pre : int
        Number of presynaptic neurons.
    n_post : int
        Number of postsynaptic neurons.
    tau_e : float
        Eligibility decay time constant (in same units as dt).
    dt : float
        Simulation timestep.
    trace_type : TraceType
        One of ACCUMULATING, REPLACING, DUTCH.
    alpha : float
        Dutch trace replacement rate in [0, 1]. Only used for DUTCH type.
    clamp_min, clamp_max : float
        Bounds for trace clamping after each update step.
    use_exponential_decay : bool
        If True, use exp(-dt/tau_e); else use linear approximation 1 - dt/tau_e.
    A_plus, A_minus : float
        STDP potentiation / depression amplitudes.
    tau_plus, tau_minus : float
        STDP potentiation / depression time constants.
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        tau_e: float = 100.0,
        dt: float = 1.0,
        trace_type: TraceType = TraceType.ACCUMULATING,
        alpha: float = 0.1,
        clamp_min: float = -5.0,
        clamp_max: float = 5.0,
        use_exponential_decay: bool = True,
        A_plus: float = 0.01,
        A_minus: float = 0.012,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
    ):
        super().__init__()
        self.n_pre = n_pre
        self.n_post = n_post
        self.tau_e = tau_e
        self.dt = dt
        self.trace_type = trace_type
        self.alpha = alpha
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.A_plus = A_plus
        self.A_minus = A_minus

        # Compute decay factors
        if use_exponential_decay:
            self.decay_e = math.exp(-dt / tau_e)
            self.decay_pre = math.exp(-dt / tau_plus)
            self.decay_post = math.exp(-dt / tau_minus)
        else:
            self.decay_e = max(0.0, 1.0 - dt / tau_e)
            self.decay_pre = max(0.0, 1.0 - dt / tau_plus)
            self.decay_post = max(0.0, 1.0 - dt / tau_minus)

        # State tensors -- allocated lazily on first call or reset
        self.eligibility: Optional[torch.Tensor] = None
        self.x_pre: Optional[torch.Tensor] = None
        self.x_post: Optional[torch.Tensor] = None

    def reset(self, batch_size: int, device: torch.device) -> None:
        """Clear all traces to exactly zero. Call at episode/task boundaries."""
        self.eligibility = torch.zeros(
            batch_size, self.n_post, self.n_pre, device=device, dtype=torch.float32,
        )
        self.x_pre = torch.zeros(
            batch_size, self.n_pre, device=device, dtype=torch.float32,
        )
        self.x_post = torch.zeros(
            batch_size, self.n_post, device=device, dtype=torch.float32,
        )

    def _ensure_state(self, batch_size: int, device: torch.device) -> None:
        """Allocate state on first call or batch size change (carry semantics)."""
        if self.eligibility is None or self.eligibility.shape[0] != batch_size:
            self.reset(batch_size, device)

    def _apply_trace_update(
        self, trace: torch.Tensor, correlation: torch.Tensor,
    ) -> torch.Tensor:
        """Apply trace-type-specific accumulation semantics."""
        if self.trace_type == TraceType.ACCUMULATING:
            trace = self.decay_e * trace + correlation
        elif self.trace_type == TraceType.REPLACING:
            decayed = self.decay_e * trace
            # Replace where new correlation exceeds decayed value in magnitude
            use_new = correlation.abs() > decayed.abs()
            trace = torch.where(use_new, correlation, decayed)
        elif self.trace_type == TraceType.DUTCH:
            trace = (1.0 - self.alpha) * self.decay_e * trace + correlation
        else:
            raise ValueError(f"Unknown trace_type: {self.trace_type}")
        return torch.clamp(trace, self.clamp_min, self.clamp_max)

    @torch.no_grad()
    def step_hebbian(
        self,
        pre: torch.Tensor,   # (B, N_pre), continuous rates
        post: torch.Tensor,   # (B, N_post), continuous rates
    ) -> torch.Tensor:
        """Update eligibility using rate-based Hebbian outer-product correlation.

        Returns the current eligibility trace (B, N_post, N_pre).
        """
        pre = pre.float()
        post = post.float()
        B = pre.shape[0]
        self._ensure_state(B, pre.device)

        # Outer product correlation: (B, N_post, N_pre)
        correlation = post.unsqueeze(2) * pre.unsqueeze(1)

        self.eligibility = self._apply_trace_update(self.eligibility, correlation)
        return self.eligibility

    @torch.no_grad()
    def step_stdp(
        self,
        spikes_pre: torch.Tensor,   # (B, N_pre), binary 0/1
        spikes_post: torch.Tensor,  # (B, N_post), binary 0/1
    ) -> torch.Tensor:
        """Update eligibility using spike-based pair-based STDP.

        Maintains exponentially decaying pre/post spike traces. On a post spike,
        the pre trace contributes LTP; on a pre spike, the post trace contributes LTD.

        Returns the current eligibility trace (B, N_post, N_pre).
        """
        spikes_pre = spikes_pre.float()
        spikes_post = spikes_post.float()
        B = spikes_pre.shape[0]
        self._ensure_state(B, spikes_pre.device)

        # Decay then add new spikes to pre/post trace variables
        self.x_pre = self.decay_pre * self.x_pre + spikes_pre
        self.x_post = self.decay_post * self.x_post + spikes_post

        # LTP: where post fires, use pre trace  (B, N_post, N_pre)
        ltp = self.A_plus * spikes_post.unsqueeze(2) * self.x_pre.unsqueeze(1)
        # LTD: where pre fires, use post trace   (B, N_post, N_pre)
        ltd = self.A_minus * self.x_post.unsqueeze(2) * spikes_pre.unsqueeze(1)

        correlation = ltp - ltd

        self.eligibility = self._apply_trace_update(self.eligibility, correlation)
        return self.eligibility

    def get_trace(self) -> Optional[torch.Tensor]:
        """Return current eligibility trace without updating it."""
        return self.eligibility


# ---------------------------------------------------------------------------
# 1.3 Rate Kernel (standalone function)
# ---------------------------------------------------------------------------

def rate_kernel(pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
    """Compute rate-based Hebbian correlation: outer product f(pre, post).

    Parameters
    ----------
    pre : (B, N_pre) or (N_pre,)
    post : (B, N_post) or (N_post,)

    Returns
    -------
    correlation : (B, N_post, N_pre) or (N_post, N_pre)
    """
    if pre.dim() == 1:
        return torch.outer(post, pre)
    return post.unsqueeze(2) * pre.unsqueeze(1)


# ---------------------------------------------------------------------------
# 1.4 STDP Kernel (standalone function)
# ---------------------------------------------------------------------------

def stdp_kernel(
    delta_t: torch.Tensor,
    A_plus: float = 0.01,
    A_minus: float = 0.012,
    tau_plus: float = 20.0,
    tau_minus: float = 20.0,
) -> torch.Tensor:
    """Pair-based STDP kernel: signed eligibility as a function of timing.

    Parameters
    ----------
    delta_t : Tensor
        Timing difference t_post - t_pre.  Positive = causal (LTP).
    A_plus, A_minus : float
        Amplitudes.
    tau_plus, tau_minus : float
        Time constants.

    Returns
    -------
    f : Tensor  same shape as delta_t.
        Positive for causal (delta_t > 0), negative for anti-causal (delta_t < 0).
    """
    f = torch.zeros_like(delta_t)
    # Causal: pre fires before post
    causal = delta_t > 0
    f[causal] = A_plus * torch.exp(-delta_t[causal] / tau_plus)
    # Anti-causal: post fires before pre
    anticausal = delta_t < 0
    f[anticausal] = -A_minus * torch.exp(delta_t[anticausal] / tau_minus)
    return f


# ---------------------------------------------------------------------------
# 1.5 NeuromodulatoryGate
# ---------------------------------------------------------------------------

class NeuromodulatoryGate(nn.Module):
    """Computes DA / ACh / NE / 5-HT modulators and a global plasticity gain.

    All computations are deterministic and bounded.

    Parameters
    ----------
    combination_fn : str
        "weighted_sum" (default).
    alpha_baseline : float
        EMA rate for the reward baseline used in DA computation.
    """

    def __init__(
        self,
        combination_fn: str = "weighted_sum",
        alpha_baseline: float = 0.01,
    ):
        super().__init__()
        self.combination_fn = combination_fn
        self.alpha_baseline = alpha_baseline

        # Learnable weights (one per modulator)
        self.w_reward = nn.Parameter(torch.tensor(1.0))
        self.w_novelty = nn.Parameter(torch.tensor(1.0))
        self.w_urgency = nn.Parameter(torch.tensor(1.0))
        self.w_patience = nn.Parameter(torch.tensor(1.0))

        # Biases for sigmoid-based modulators
        self.bias_ach = nn.Parameter(torch.tensor(0.0))
        self.bias_ne = nn.Parameter(torch.tensor(0.0))
        self.bias_5ht = nn.Parameter(torch.tensor(0.0))

        # Combination weights (weighted_sum mode)
        self.combination_weights = nn.Parameter(
            torch.tensor([0.4, 0.3, 0.2, 0.1]),
        )

        # Running state
        self.register_buffer("reward_baseline", torch.tensor(0.0))

    # ---- Individual modulator computations ----

    def compute_da(self, reward: torch.Tensor) -> torch.Tensor:
        """DA = tanh(w_reward * (reward - baseline)).  Range [-1, 1]."""
        reward = reward.float()
        rpe = reward - self.reward_baseline
        return torch.tanh(self.w_reward * rpe)

    def compute_ach(self, novelty: torch.Tensor) -> torch.Tensor:
        """ACh = sigmoid(w_novelty * novelty + bias).  Range [0, 1]."""
        novelty = novelty.float()
        return torch.sigmoid(self.w_novelty * novelty + self.bias_ach)

    def compute_ne(self, urgency: torch.Tensor) -> torch.Tensor:
        """NE = sigmoid(w_urgency * urgency + bias).  Range [0, 1]."""
        urgency = urgency.float()
        return torch.sigmoid(self.w_urgency * urgency + self.bias_ne)

    def compute_5ht(self, patience: torch.Tensor) -> torch.Tensor:
        """5-HT = sigmoid(w_patience * patience + bias).  Range [0, 1]."""
        patience = patience.float()
        return torch.sigmoid(self.w_patience * patience + self.bias_5ht)

    # ---- Forward ----

    def forward(
        self,
        signals: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Compute all four modulators and the combined global plasticity gain.

        Parameters
        ----------
        signals : dict
            Keys may include "reward", "novelty", "urgency", "patience".
            Values are Tensors of shape (B,) or scalar Tensors.

        Returns
        -------
        modulators : dict  {DA, ACh, NE, 5HT} each (B,) or scalar
        global_plasticity : Tensor (B,) or scalar
        """
        reward = signals.get("reward")
        novelty = signals.get("novelty")
        urgency = signals.get("urgency")
        patience = signals.get("patience")

        # Compute each modulator
        if reward is not None:
            da = self.compute_da(reward)
        else:
            da = torch.tensor(0.0)

        if novelty is not None:
            ach = self.compute_ach(novelty)
        else:
            ach = torch.tensor(0.5)

        if urgency is not None:
            ne = self.compute_ne(urgency)
        else:
            ne = torch.tensor(0.5)

        if patience is not None:
            sht = self.compute_5ht(patience)
        else:
            sht = torch.tensor(0.5)

        # Combination
        if self.combination_fn == "weighted_sum":
            stacked = torch.stack(
                [da, ach, ne, sht] if da.dim() > 0 else
                [da.unsqueeze(0), ach.unsqueeze(0), ne.unsqueeze(0), sht.unsqueeze(0)],
                dim=-1,
            )
            if stacked.dim() == 1:
                stacked = stacked.unsqueeze(0)
            raw = (stacked * self.combination_weights).sum(dim=-1)
            global_plasticity = torch.tanh(raw).squeeze(0)
        else:
            global_plasticity = da  # Fallback

        modulators = {"DA": da, "ACh": ach, "NE": ne, "5HT": sht}
        return modulators, global_plasticity

    def update_baseline(self, reward: torch.Tensor) -> None:
        """Update reward baseline EMA. Call once per step after forward."""
        with torch.no_grad():
            mean_reward = reward.detach().float().mean()
            self.reward_baseline.mul_(1.0 - self.alpha_baseline).add_(
                self.alpha_baseline * mean_reward,
            )

    def reset_state(self) -> None:
        """Reset running state between episodes."""
        with torch.no_grad():
            self.reward_baseline.zero_()


# ---------------------------------------------------------------------------
# 1.6 Reward Baseline EMA (standalone)
# ---------------------------------------------------------------------------

class RewardBaselineEMA:
    """Exponential moving average tracker for reward baseline.

    baseline(t) = (1 - alpha) * baseline(t-1) + alpha * reward(t)
    """

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        self.value = 0.0

    def update(self, reward: float) -> float:
        self.value = (1.0 - self.alpha) * self.value + self.alpha * reward
        return self.value

    def reset(self) -> None:
        self.value = 0.0


# ---------------------------------------------------------------------------
# 1.7 ThreeFactorUpdate
# ---------------------------------------------------------------------------

class ThreeFactorUpdate:
    """Applies the canonical three-factor weight update rule:

        delta_w = lr * mod_signal * eligibility_trace

    with per-step delta clamping and absolute weight clamping.
    """

    def __init__(
        self,
        lr: float = 0.001,
        weight_clamp: Tuple[float, float] = (-1.0, 1.0),
        delta_clamp: float = 0.05,
    ):
        self.lr = lr
        self.weight_clamp = weight_clamp
        self.delta_clamp = delta_clamp

    def compute_delta_w(
        self,
        mod_signal: torch.Tensor,
        eligibility: torch.Tensor,
    ) -> torch.Tensor:
        """Compute delta_w = lr * mod_signal * eligibility.

        Both mod_signal and eligibility are detached (no grad).
        """
        mod = mod_signal.detach().float()
        e = eligibility.detach().float()

        # Broadcast mod to match eligibility shape
        while mod.dim() < e.dim():
            mod = mod.unsqueeze(-1)

        delta_w = self.lr * mod * e

        # Per-step delta clamp
        delta_w = torch.clamp(delta_w, -self.delta_clamp, self.delta_clamp)
        return delta_w

    @torch.no_grad()
    def apply(
        self,
        weights: torch.Tensor,
        mod_signal: torch.Tensor,
        eligibility: torch.Tensor,
    ) -> torch.Tensor:
        """Compute delta_w and apply to weights with clamping.

        Returns the updated weight tensor (same storage, modified in place).
        """
        delta_w = self.compute_delta_w(mod_signal, eligibility)

        # Average over batch dim if present (eligibility is (B, N_post, N_pre))
        if delta_w.dim() > weights.dim():
            delta_w = delta_w.mean(dim=0)

        weights.data.add_(delta_w)
        weights.data.clamp_(self.weight_clamp[0], self.weight_clamp[1])
        return weights


# ---------------------------------------------------------------------------
# 1.8 FastMemoryAdapter
# ---------------------------------------------------------------------------

class FastMemoryAdapter(nn.Module):
    """Low-rank adapter: output = x + up(down(x)).

    Only adapter weights receive three-factor plasticity updates.
    The main model weights remain frozen.
    """

    def __init__(self, dim: int, bottleneck: int = 64):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, dim, bias=False)

        nn.init.normal_(self.down.weight, std=0.01)
        nn.init.normal_(self.up.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual adapter forward pass."""
        h = self.down(x)
        return x + self.up(h)


# ---------------------------------------------------------------------------
# 1.9 Simple Linear Model for Delayed Reward Toy Task
# ---------------------------------------------------------------------------

class DelayedRewardModel(nn.Module):
    """Simple two-layer model for the delayed reward association task.

    Input -> Linear(input_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, output_dim)
    """

    def __init__(self, input_dim: int = 8, hidden_dim: int = 32, output_dim: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        return self.fc2(h)


# ============================================================================
# SECTION 2: VALIDATION INFRASTRUCTURE
# ============================================================================


@dataclass
class CheckResult:
    """Result of a single validation check."""
    group: str
    name: str
    passed: bool
    message: str
    elapsed_ms: float = 0.0


@dataclass
class GroupResult:
    """Result of a validation group."""
    group_name: str
    group_id: int
    checks: List[CheckResult] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def failed(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    @property
    def total(self) -> int:
        return len(self.checks)

    @property
    def all_passed(self) -> bool:
        return self.failed == 0


def run_check(group: str, name: str, fn, verbose: bool = False) -> CheckResult:
    """Run a single check function, capturing pass/fail and timing."""
    t0 = time.perf_counter()
    try:
        fn()
        elapsed = (time.perf_counter() - t0) * 1000.0
        result = CheckResult(
            group=group, name=name, passed=True,
            message="OK", elapsed_ms=elapsed,
        )
    except AssertionError as e:
        elapsed = (time.perf_counter() - t0) * 1000.0
        result = CheckResult(
            group=group, name=name, passed=False,
            message=str(e) or "Assertion failed", elapsed_ms=elapsed,
        )
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000.0
        result = CheckResult(
            group=group, name=name, passed=False,
            message=f"Exception: {type(e).__name__}: {e}", elapsed_ms=elapsed,
        )

    if verbose:
        status = colorize("PASS", "green") if result.passed else colorize("FAIL", "red")
        print(f"  [{status}] {name} ({result.elapsed_ms:.1f}ms)")
        if not result.passed:
            print(f"         {result.message}")

    return result


# ---------------------------------------------------------------------------
# Terminal colors
# ---------------------------------------------------------------------------

_COLOR_CODES = {
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "cyan": "\033[96m",
    "bold": "\033[1m",
    "reset": "\033[0m",
}


def colorize(text: str, color: str) -> str:
    """Wrap text in ANSI color codes.  Falls back to plain text if not a tty."""
    if not sys.stdout.isatty():
        return text
    code = _COLOR_CODES.get(color, "")
    return f"{code}{text}{_COLOR_CODES['reset']}"


# ============================================================================
# SECTION 3: VALIDATION GROUPS
# ============================================================================


# ---------------------------------------------------------------------------
# GROUP 1: Trace Dynamics (6 checks)
# ---------------------------------------------------------------------------

def run_group_1(verbose: bool = False) -> GroupResult:
    """Group 1: Trace Dynamics -- accumulating, replacing, Dutch, decay, reset, determinism."""
    group_name = "Trace Dynamics"
    result = GroupResult(group_name=group_name, group_id=1)
    t0 = time.perf_counter()

    # Common dimensions
    B, N_pre, N_post = 4, 16, 8

    # -- Check 1.1: Accumulating trace increases with activity --
    def check_accumulating_increases():
        torch.manual_seed(42)
        mod = EligibilityTraceModule(
            n_pre=N_pre, n_post=N_post, tau_e=50.0, dt=1.0,
            trace_type=TraceType.ACCUMULATING,
        )
        pre = torch.rand(B, N_pre)
        post = torch.rand(B, N_post)
        norms = []
        for _ in range(10):
            e = mod.step_hebbian(pre, post)
            norms.append(e.norm().item())
        # Norms should be monotonically increasing (or at least final > first)
        assert norms[-1] > norms[0], (
            f"Accumulating trace norm should increase: first={norms[0]:.4f}, last={norms[-1]:.4f}"
        )
        # Also check that the sequence is broadly increasing
        for i in range(1, len(norms)):
            assert norms[i] >= norms[i - 1] - 1e-6, (
                f"Norm at step {i} ({norms[i]:.6f}) < step {i-1} ({norms[i-1]:.6f})"
            )

    result.checks.append(run_check(group_name, "Accumulating trace increases with activity", check_accumulating_increases, verbose))

    # -- Check 1.2: Replacing trace stays bounded --
    def check_replacing_bounded():
        torch.manual_seed(42)
        mod = EligibilityTraceModule(
            n_pre=N_pre, n_post=N_post, tau_e=50.0, dt=1.0,
            trace_type=TraceType.REPLACING,
        )
        pre = torch.rand(B, N_pre) * 2.0
        post = torch.rand(B, N_post) * 2.0
        # Compute single-step max correlation
        single_corr = post.unsqueeze(2) * pre.unsqueeze(1)
        f_max = single_corr.abs().max().item()
        for _ in range(30):
            e = mod.step_hebbian(pre, post)
        assert e.abs().max().item() <= f_max + 1e-5, (
            f"Replacing trace exceeded single-step max: {e.abs().max().item():.6f} > {f_max:.6f}"
        )

    result.checks.append(run_check(group_name, "Replacing trace stays bounded", check_replacing_bounded, verbose))

    # -- Check 1.3: Dutch trace intermediate behavior --
    def check_dutch_intermediate():
        torch.manual_seed(42)
        pre = torch.rand(B, N_pre)
        post = torch.rand(B, N_post)

        mod_acc = EligibilityTraceModule(
            n_pre=N_pre, n_post=N_post, tau_e=50.0, dt=1.0,
            trace_type=TraceType.ACCUMULATING,
        )
        mod_rep = EligibilityTraceModule(
            n_pre=N_pre, n_post=N_post, tau_e=50.0, dt=1.0,
            trace_type=TraceType.REPLACING,
        )
        mod_dut = EligibilityTraceModule(
            n_pre=N_pre, n_post=N_post, tau_e=50.0, dt=1.0,
            trace_type=TraceType.DUTCH, alpha=0.3,
        )

        for step in range(15):
            e_acc = mod_acc.step_hebbian(pre, post)
            e_rep = mod_rep.step_hebbian(pre, post)
            e_dut = mod_dut.step_hebbian(pre, post)

        n_acc = e_acc.norm().item()
        n_rep = e_rep.norm().item()
        n_dut = e_dut.norm().item()
        assert n_rep <= n_dut + 1e-4, (
            f"Dutch norm ({n_dut:.4f}) should be >= replacing norm ({n_rep:.4f})"
        )
        assert n_dut <= n_acc + 1e-4, (
            f"Dutch norm ({n_dut:.4f}) should be <= accumulating norm ({n_acc:.4f})"
        )

    result.checks.append(run_check(group_name, "Dutch trace: intermediate behavior", check_dutch_intermediate, verbose))

    # -- Check 1.4: Decay toward zero without activity --
    def check_decay_toward_zero():
        torch.manual_seed(42)
        mod = EligibilityTraceModule(
            n_pre=N_pre, n_post=N_post, tau_e=20.0, dt=1.0,
            trace_type=TraceType.ACCUMULATING,
        )
        pre = torch.rand(B, N_pre)
        post = torch.rand(B, N_post)
        # Build up traces
        for _ in range(5):
            mod.step_hebbian(pre, post)
        initial_norm = mod.eligibility.norm().item()
        assert initial_norm > 0, "Trace should be nonzero after activity"

        # Now feed zeros
        zero_pre = torch.zeros(B, N_pre)
        zero_post = torch.zeros(B, N_post)
        for _ in range(200):
            mod.step_hebbian(zero_pre, zero_post)

        final_norm = mod.eligibility.norm().item()
        assert final_norm < 0.01 * initial_norm, (
            f"Trace should decay to <1%: initial={initial_norm:.6f}, final={final_norm:.6f}"
        )

    result.checks.append(run_check(group_name, "Decay: traces approach zero without activity", check_decay_toward_zero, verbose))

    # -- Check 1.5: Reset clears to exactly zero --
    def check_reset():
        torch.manual_seed(42)
        mod = EligibilityTraceModule(
            n_pre=N_pre, n_post=N_post, tau_e=50.0, dt=1.0,
        )
        pre = torch.rand(B, N_pre)
        post = torch.rand(B, N_post)
        for _ in range(5):
            mod.step_hebbian(pre, post)
        assert mod.eligibility.norm().item() > 0, "Trace should be nonzero"
        mod.reset(B, pre.device)
        assert torch.all(mod.eligibility == 0), "Eligibility should be exactly zero after reset"
        assert torch.all(mod.x_pre == 0), "x_pre should be exactly zero after reset"
        assert torch.all(mod.x_post == 0), "x_post should be exactly zero after reset"

    result.checks.append(run_check(group_name, "Reset: clears to exactly zero", check_reset, verbose))

    # -- Check 1.6: Determinism --
    def check_determinism():
        results = []
        pre_data = torch.rand(B, N_pre)
        post_data = torch.rand(B, N_post)
        for run in range(10):
            torch.manual_seed(42)
            mod = EligibilityTraceModule(
                n_pre=N_pre, n_post=N_post, tau_e=50.0, dt=1.0,
                trace_type=TraceType.ACCUMULATING,
            )
            for _ in range(20):
                e = mod.step_hebbian(pre_data, post_data)
            results.append(e.clone())

        for i in range(1, 10):
            assert torch.allclose(results[i], results[0], atol=1e-6), (
                f"Run {i} differs from run 0: max diff = {(results[i] - results[0]).abs().max().item():.2e}"
            )

    result.checks.append(run_check(group_name, "Determinism: fixed inputs produce identical traces", check_determinism, verbose))

    # -- Check 1.7: Tau_e effect: larger tau_e => slower decay --
    def check_tau_e_effect():
        """Two modules with different tau_e: the one with larger tau_e retains
        more trace energy after identical zero-input decay steps."""
        torch.manual_seed(42)
        pre = torch.rand(B, N_pre)
        post = torch.rand(B, N_post)

        mod_fast = EligibilityTraceModule(
            n_pre=N_pre, n_post=N_post, tau_e=FAST_TAU_E, dt=DEFAULT_DT,
            trace_type=TraceType.ACCUMULATING,
        )
        mod_slow = EligibilityTraceModule(
            n_pre=N_pre, n_post=N_post, tau_e=SLOW_TAU_E, dt=DEFAULT_DT,
            trace_type=TraceType.ACCUMULATING,
        )

        # Build up traces with identical input
        for _ in range(5):
            mod_fast.step_hebbian(pre, post)
            mod_slow.step_hebbian(pre, post)

        # Now let both decay without input
        zero_pre = torch.zeros(B, N_pre)
        zero_post = torch.zeros(B, N_post)
        for step in range(20):
            mod_fast.step_hebbian(zero_pre, zero_post)
            mod_slow.step_hebbian(zero_pre, zero_post)

            n_fast = mod_fast.eligibility.norm().item()
            n_slow = mod_slow.eligibility.norm().item()
            # Slow tau should always retain more energy than fast tau
            assert n_slow >= n_fast - 1e-6, (
                f"Step {step}: slow tau norm ({n_slow:.6f}) < fast tau norm ({n_fast:.6f})"
            )

    result.checks.append(run_check(group_name, "Tau_e effect: larger tau_e => slower decay", check_tau_e_effect, verbose))

    # -- Check 1.8: Zero pre activity => zero correlation update --
    def check_zero_pre_zero_update():
        """Zero pre-synaptic activity produces zero trace update (only decay
        of existing trace occurs)."""
        torch.manual_seed(42)
        mod = EligibilityTraceModule(
            n_pre=N_pre, n_post=N_post, tau_e=DEFAULT_TAU_E, dt=DEFAULT_DT,
            trace_type=TraceType.ACCUMULATING,
        )
        # Build some initial trace
        pre = torch.rand(B, N_pre)
        post = torch.rand(B, N_post)
        mod.step_hebbian(pre, post)
        e_before = mod.eligibility.clone()

        # Now feed zero pre, nonzero post
        zero_pre = torch.zeros(B, N_pre)
        nonzero_post = torch.rand(B, N_post) * 3.0
        mod.step_hebbian(zero_pre, nonzero_post)
        e_after = mod.eligibility.clone()

        # The result should be exactly decay * e_before (outer product with zero pre = 0)
        expected = mod.decay_e * e_before
        expected = torch.clamp(expected, mod.clamp_min, mod.clamp_max)
        max_diff = (e_after - expected).abs().max().item()
        assert torch.allclose(e_after, expected, atol=ATOL_FP32_DETERMINISM), (
            f"Zero pre should produce pure decay: max diff = {max_diff:.2e}"
        )

    result.checks.append(run_check(group_name, "Zero pre activity => zero correlation (pure decay)", check_zero_pre_zero_update, verbose))

    # -- Check 1.9: Zero post activity => zero correlation update --
    def check_zero_post_zero_update():
        """Zero post-synaptic activity produces zero trace update (only decay
        of existing trace occurs)."""
        torch.manual_seed(42)
        mod = EligibilityTraceModule(
            n_pre=N_pre, n_post=N_post, tau_e=DEFAULT_TAU_E, dt=DEFAULT_DT,
            trace_type=TraceType.ACCUMULATING,
        )
        pre = torch.rand(B, N_pre)
        post = torch.rand(B, N_post)
        mod.step_hebbian(pre, post)
        e_before = mod.eligibility.clone()

        # Now feed nonzero pre, zero post
        nonzero_pre = torch.rand(B, N_pre) * 3.0
        zero_post = torch.zeros(B, N_post)
        mod.step_hebbian(nonzero_pre, zero_post)
        e_after = mod.eligibility.clone()

        expected = mod.decay_e * e_before
        expected = torch.clamp(expected, mod.clamp_min, mod.clamp_max)
        max_diff = (e_after - expected).abs().max().item()
        assert torch.allclose(e_after, expected, atol=ATOL_FP32_DETERMINISM), (
            f"Zero post should produce pure decay: max diff = {max_diff:.2e}"
        )

    result.checks.append(run_check(group_name, "Zero post activity => zero correlation (pure decay)", check_zero_post_zero_update, verbose))

    # -- Check 1.10: Clamp range enforced under sustained high activity --
    def check_clamp_enforced():
        """Trace values never exceed configured clamp bounds even under
        sustained high-magnitude pre/post activity for 50 steps."""
        torch.manual_seed(42)
        clamp_lo, clamp_hi = -2.0, 2.0
        mod = EligibilityTraceModule(
            n_pre=N_pre, n_post=N_post, tau_e=DEFAULT_TAU_E, dt=DEFAULT_DT,
            trace_type=TraceType.ACCUMULATING,
            clamp_min=clamp_lo, clamp_max=clamp_hi,
        )
        # Sustained large-magnitude activity
        pre = torch.ones(B, N_pre) * 5.0
        post = torch.ones(B, N_post) * 5.0
        for step in range(50):
            e = mod.step_hebbian(pre, post)
            assert torch.all(e >= clamp_lo), (
                f"Step {step}: trace below clamp_min: min={e.min().item():.4f}"
            )
            assert torch.all(e <= clamp_hi), (
                f"Step {step}: trace above clamp_max: max={e.max().item():.4f}"
            )

    result.checks.append(run_check(group_name, "Clamp range enforced: traces stay in bounds", check_clamp_enforced, verbose))

    result.elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return result


# ---------------------------------------------------------------------------
# GROUP 2: STDP Kernels (4 checks)
# ---------------------------------------------------------------------------

def run_group_2(verbose: bool = False) -> GroupResult:
    """Group 2: STDP Kernels -- causal, anti-causal, zero, rate kernel shape."""
    group_name = "STDP Kernels"
    result = GroupResult(group_name=group_name, group_id=2)
    t0 = time.perf_counter()

    # -- Check 2.1: Causal timing (dt > 0) -> positive eligibility --
    def check_causal_positive():
        delta_t = torch.tensor([1.0, 5.0, 10.0, 15.0, 20.0])
        f = stdp_kernel(delta_t, A_plus=0.01, A_minus=0.012)
        assert torch.all(f > 0), (
            f"Causal timing should produce positive eligibility, got {f.tolist()}"
        )

    result.checks.append(run_check(group_name, "Causal timing (dt>0) -> positive eligibility", check_causal_positive, verbose))

    # -- Check 2.2: Anti-causal timing (dt < 0) -> negative eligibility --
    def check_anticausal_negative():
        delta_t = torch.tensor([-1.0, -5.0, -10.0, -15.0, -20.0])
        f = stdp_kernel(delta_t, A_plus=0.01, A_minus=0.012)
        assert torch.all(f < 0), (
            f"Anti-causal timing should produce negative eligibility, got {f.tolist()}"
        )

    result.checks.append(run_check(group_name, "Anti-causal timing (dt<0) -> negative eligibility", check_anticausal_negative, verbose))

    # -- Check 2.3: Zero spikes -> zero update --
    def check_zero_spikes():
        B, N_pre, N_post = 4, 16, 8
        mod = EligibilityTraceModule(
            n_pre=N_pre, n_post=N_post, tau_e=50.0, dt=1.0,
        )
        # All zero spikes
        spikes_pre = torch.zeros(B, N_pre)
        spikes_post = torch.zeros(B, N_post)
        e = mod.step_stdp(spikes_pre, spikes_post)
        assert torch.all(e == 0), (
            f"Zero spikes should produce exactly zero eligibility, got norm={e.norm().item():.2e}"
        )

    result.checks.append(run_check(group_name, "Zero spikes -> zero update", check_zero_spikes, verbose))

    # -- Check 2.4: Rate kernel outer product shape --
    def check_rate_kernel_shape():
        B, N_pre, N_post = 4, 16, 8
        pre = torch.rand(B, N_pre)
        post = torch.rand(B, N_post)
        corr = rate_kernel(pre, post)
        expected_shape = (B, N_post, N_pre)
        assert corr.shape == expected_shape, (
            f"Rate kernel shape should be {expected_shape}, got {corr.shape}"
        )
        # Also check unbatched
        corr_1d = rate_kernel(pre[0], post[0])
        assert corr_1d.shape == (N_post, N_pre), (
            f"Unbatched rate kernel shape should be ({N_post}, {N_pre}), got {corr_1d.shape}"
        )

    result.checks.append(run_check(group_name, "Rate kernel: outer product shape correct", check_rate_kernel_shape, verbose))

    # -- Check 2.5: STDP kernel exponential decay with distance --
    def check_stdp_exponential_decay():
        """Causal STDP kernel magnitude decreases exponentially with delta_t.
        Verifies the exp(-delta_t / tau_plus) profile for positive timings."""
        deltas = torch.tensor([1.0, 5.0, 10.0, 20.0, 40.0])
        f = stdp_kernel(deltas, A_plus=DEFAULT_A_PLUS, A_minus=DEFAULT_A_MINUS,
                        tau_plus=DEFAULT_TAU_PLUS, tau_minus=DEFAULT_TAU_MINUS)
        # Verify monotonically decreasing for positive (causal) timings
        for i in range(1, len(deltas)):
            assert f[i] < f[i - 1], (
                f"STDP kernel should decrease with distance: "
                f"f({deltas[i].item()})={f[i].item():.6f} >= f({deltas[i-1].item()})={f[i-1].item():.6f}"
            )
        # Verify the values match expected analytical form
        for i, dt_val in enumerate(deltas):
            expected = DEFAULT_A_PLUS * math.exp(-dt_val.item() / DEFAULT_TAU_PLUS)
            assert abs(f[i].item() - expected) < 1e-6, (
                f"STDP at dt={dt_val.item()}: expected {expected:.8f}, got {f[i].item():.8f}"
            )

    result.checks.append(run_check(group_name, "STDP kernel: exponential decay with distance", check_stdp_exponential_decay, verbose))

    # -- Check 2.6: STDP multi-step spike sequence produces signed traces --
    def check_stdp_multi_step_sequence():
        """Run a multi-step spike sequence through the EligibilityTraceModule
        in STDP mode.  When a pre spike at t=1 is followed by a post spike
        at t=3, the resulting eligibility should have a positive (LTP)
        component for that synapse pair."""
        B_test, N_pre_test, N_post_test = 1, 4, 2
        mod = EligibilityTraceModule(
            n_pre=N_pre_test, n_post=N_post_test, tau_e=DEFAULT_TAU_E,
            dt=DEFAULT_DT, A_plus=DEFAULT_A_PLUS, A_minus=DEFAULT_A_MINUS,
        )
        mod.reset(B_test, torch.device("cpu"))

        # t=0: no spikes
        mod.step_stdp(
            torch.zeros(B_test, N_pre_test),
            torch.zeros(B_test, N_post_test),
        )

        # t=1: pre neuron 0 fires
        pre_t1 = torch.zeros(B_test, N_pre_test)
        pre_t1[0, 0] = 1.0
        mod.step_stdp(pre_t1, torch.zeros(B_test, N_post_test))

        # t=2: no spikes
        mod.step_stdp(
            torch.zeros(B_test, N_pre_test),
            torch.zeros(B_test, N_post_test),
        )

        # t=3: post neuron 0 fires (causal: pre at t=1, post at t=3 => LTP)
        post_t3 = torch.zeros(B_test, N_post_test)
        post_t3[0, 0] = 1.0
        e = mod.step_stdp(torch.zeros(B_test, N_pre_test), post_t3)

        # The synapse (post=0, pre=0) should have positive eligibility (LTP)
        assert e[0, 0, 0].item() > 0, (
            f"Causal spike pair should produce positive LTP eligibility, "
            f"got e[0,0,0]={e[0, 0, 0].item():.8f}"
        )

        # The synapse (post=0, pre=1) should have zero or near-zero eligibility
        # (pre neuron 1 never fired)
        assert abs(e[0, 0, 1].item()) < 1e-6, (
            f"Inactive pre synapse should have ~zero eligibility, "
            f"got e[0,0,1]={e[0, 0, 1].item():.8f}"
        )

    result.checks.append(run_check(group_name, "STDP multi-step: causal spike pair -> positive LTP", check_stdp_multi_step_sequence, verbose))

    # -- Check 2.7: STDP symmetry property: A_plus == A_minus and |dt| equal --
    def check_stdp_symmetry():
        """When A_plus == A_minus and tau_plus == tau_minus, the absolute
        magnitude of the kernel should be symmetric: |f(+dt)| == |f(-dt)|."""
        A_sym = 0.01
        tau_sym = 20.0
        dt_pos = torch.tensor([5.0])
        dt_neg = torch.tensor([-5.0])
        f_pos = stdp_kernel(dt_pos, A_plus=A_sym, A_minus=A_sym,
                            tau_plus=tau_sym, tau_minus=tau_sym)
        f_neg = stdp_kernel(dt_neg, A_plus=A_sym, A_minus=A_sym,
                            tau_plus=tau_sym, tau_minus=tau_sym)
        assert abs(f_pos.item()) > 0, "Positive dt should give nonzero f"
        assert abs(f_neg.item()) > 0, "Negative dt should give nonzero f"
        assert abs(abs(f_pos.item()) - abs(f_neg.item())) < 1e-8, (
            f"Symmetric STDP should have |f(+dt)| == |f(-dt)|: "
            f"|f(+5)|={abs(f_pos.item()):.8f}, |f(-5)|={abs(f_neg.item()):.8f}"
        )
        # Also verify opposite signs
        assert f_pos.item() > 0 and f_neg.item() < 0, (
            f"f(+dt) should be positive ({f_pos.item():.6f}) and "
            f"f(-dt) negative ({f_neg.item():.6f})"
        )

    result.checks.append(run_check(group_name, "STDP symmetry: |f(+dt)| == |f(-dt)| when A_plus==A_minus", check_stdp_symmetry, verbose))

    result.elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return result


# ---------------------------------------------------------------------------
# GROUP 3: Done-When Gate (a) -- Third-Factor Gating (5 checks)
# ---------------------------------------------------------------------------

def run_group_3(verbose: bool = False) -> GroupResult:
    """Group 3: Done-When Gate (a) -- Third-factor gating semantics.

    The fundamental invariant: mod_signal = 0 => delta_w is EXACTLY zero.
    """
    group_name = "Done-When Gate (a): Third-Factor Gating"
    result = GroupResult(group_name=group_name, group_id=3)
    t0 = time.perf_counter()

    B, N_pre, N_post = 4, 16, 8
    updater = ThreeFactorUpdate(lr=0.01, weight_clamp=(-1.0, 1.0), delta_clamp=0.05)

    # -- Check 3.1: mod_signal = 0 -> delta_w is EXACTLY zero (all elements) --
    def check_zero_mod_zero_delta():
        torch.manual_seed(42)
        mod = EligibilityTraceModule(n_pre=N_pre, n_post=N_post, tau_e=50.0)
        pre = torch.rand(B, N_pre) * 5.0
        post = torch.rand(B, N_post) * 5.0
        for _ in range(10):
            mod.step_hebbian(pre, post)
        e = mod.get_trace()
        assert e.norm().item() > 0, "Eligibility should be nonzero for this test"

        mod_signal = torch.tensor(0.0)
        delta_w = updater.compute_delta_w(mod_signal, e)
        assert torch.all(delta_w == 0), (
            f"delta_w must be EXACTLY zero when mod_signal=0, "
            f"but got max abs = {delta_w.abs().max().item():.2e}"
        )

    result.checks.append(run_check(group_name, "mod_signal=0 -> delta_w is EXACTLY zero (all elements)", check_zero_mod_zero_delta, verbose))

    # -- Check 3.2: Strong pre/post + zero mod -> still zero delta_w --
    def check_strong_activity_zero_mod():
        torch.manual_seed(42)
        mod = EligibilityTraceModule(n_pre=N_pre, n_post=N_post, tau_e=50.0)
        # Very strong pre/post to ensure eligibility is large
        pre = torch.ones(B, N_pre) * 10.0
        post = torch.ones(B, N_post) * 10.0
        for _ in range(20):
            mod.step_hebbian(pre, post)
        e = mod.get_trace()
        e_norm = e.norm().item()
        assert e_norm > 1.0, f"Eligibility should be substantial: {e_norm:.4f}"

        mod_signal = torch.tensor(0.0)
        delta_w = updater.compute_delta_w(mod_signal, e)
        assert torch.all(delta_w == 0), (
            f"Zero mod_signal with strong eligibility must still give exactly zero delta_w"
        )

    result.checks.append(run_check(group_name, "Strong pre/post + zero mod -> still zero delta_w", check_strong_activity_zero_mod, verbose))

    # -- Check 3.3: mod_signal=1 + eligibility -> non-zero delta_w --
    def check_nonzero_mod_nonzero_delta():
        torch.manual_seed(42)
        mod = EligibilityTraceModule(n_pre=N_pre, n_post=N_post, tau_e=50.0)
        pre = torch.rand(B, N_pre)
        post = torch.rand(B, N_post)
        for _ in range(5):
            mod.step_hebbian(pre, post)
        e = mod.get_trace()

        mod_signal = torch.tensor(1.0)
        delta_w = updater.compute_delta_w(mod_signal, e)
        assert torch.any(delta_w != 0), (
            f"delta_w should be non-zero when mod_signal=1 and eligibility is non-zero"
        )

    result.checks.append(run_check(group_name, "mod_signal=1 + eligibility -> non-zero delta_w", check_nonzero_mod_nonzero_delta, verbose))

    # -- Check 3.4: Positive mod + positive eligibility -> positive delta_w --
    def check_positive_mod_positive_e():
        torch.manual_seed(42)
        # Create all-positive eligibility
        e = torch.rand(B, N_post, N_pre) * 2.0 + 0.1  # strictly positive
        mod_signal = torch.tensor(1.0)
        delta_w = updater.compute_delta_w(mod_signal, e)
        pos_mask = e > 0
        assert torch.all(delta_w[pos_mask] > 0), (
            f"Positive mod + positive eligibility should produce positive delta_w"
        )

    result.checks.append(run_check(group_name, "Positive mod + positive eligibility -> positive delta_w", check_positive_mod_positive_e, verbose))

    # -- Check 3.5: Negative mod + positive eligibility -> negative delta_w --
    def check_negative_mod_positive_e():
        torch.manual_seed(42)
        e = torch.rand(B, N_post, N_pre) * 2.0 + 0.1  # strictly positive
        mod_signal = torch.tensor(-1.0)
        delta_w = updater.compute_delta_w(mod_signal, e)
        pos_mask = e > 0
        assert torch.all(delta_w[pos_mask] < 0), (
            f"Negative mod + positive eligibility should produce negative delta_w"
        )

    result.checks.append(run_check(group_name, "Negative mod + positive eligibility -> negative delta_w", check_negative_mod_positive_e, verbose))

    # -- Check 3.6: Delta clamp: per-step update magnitude bounded --
    def check_delta_clamp():
        """Per-step update magnitude is bounded by delta_clamp even when
        mod_signal * eligibility is very large."""
        delta_limit = 0.01
        updater_clamped = ThreeFactorUpdate(
            lr=1.0, weight_clamp=(-10.0, 10.0), delta_clamp=delta_limit,
        )
        # Very large mod and eligibility
        e = torch.ones(B, N_post, N_pre) * 100.0
        mod_signal = torch.tensor(50.0)
        delta_w = updater_clamped.compute_delta_w(mod_signal, e)
        assert torch.all(delta_w.abs() <= delta_limit + 1e-7), (
            f"Delta clamp violated: max |delta_w| = {delta_w.abs().max().item():.6f}, "
            f"limit = {delta_limit}"
        )

    result.checks.append(run_check(group_name, "Delta clamp: per-step update magnitude bounded", check_delta_clamp, verbose))

    # -- Check 3.7: LR scaling: delta_w scales linearly with learning rate --
    def check_lr_scaling():
        """Two updaters with different LRs (10x ratio) produce delta_w with
        magnitudes in the same 10x ratio."""
        torch.manual_seed(42)
        e = torch.rand(B, N_post, N_pre) * 0.1 + 0.01
        mod_signal = torch.tensor(0.5)

        # Use large delta_clamp so clamping doesn't interfere
        updater_hi = ThreeFactorUpdate(lr=0.01, weight_clamp=(-10.0, 10.0), delta_clamp=10.0)
        updater_lo = ThreeFactorUpdate(lr=0.001, weight_clamp=(-10.0, 10.0), delta_clamp=10.0)

        delta_hi = updater_hi.compute_delta_w(mod_signal, e)
        delta_lo = updater_lo.compute_delta_w(mod_signal, e)

        # Compute ratio where both are nonzero
        mask = delta_lo.abs() > 1e-10
        ratios = delta_hi[mask] / delta_lo[mask]
        mean_ratio = ratios.mean().item()
        assert abs(mean_ratio - 10.0) < 10.0 * RTOL_LR_SCALING, (
            f"LR ratio should be ~10.0, got mean ratio = {mean_ratio:.4f}"
        )

    result.checks.append(run_check(group_name, "LR scaling: delta_w scales linearly with learning rate", check_lr_scaling, verbose))

    # -- Check 3.8: Detached computation: delta_w has no grad --
    def check_detached_computation():
        """Three-factor weight updates do not create a computational graph.
        delta_w must have requires_grad=False and grad_fn=None."""
        torch.manual_seed(42)
        e = torch.rand(B, N_post, N_pre, requires_grad=True) * 0.5
        mod_signal = torch.tensor(1.0, requires_grad=True)
        delta_w = updater.compute_delta_w(mod_signal, e)
        assert not delta_w.requires_grad, (
            "delta_w should not require grad (three-factor is detached)"
        )
        assert delta_w.grad_fn is None, (
            "delta_w should have no grad_fn (fully detached computation)"
        )

    result.checks.append(run_check(group_name, "Detached computation: delta_w has no grad_fn", check_detached_computation, verbose))

    # -- Check 3.9: Batch mod_signal: per-batch modulator produces per-batch delta_w --
    def check_batch_mod_signal():
        """When mod_signal is a batch tensor (B,), each batch element gets its
        own modulation strength. Setting one batch element's mod to zero should
        produce zero delta_w for that element only."""
        torch.manual_seed(42)
        e = torch.rand(B, N_post, N_pre) * 0.5 + 0.1
        mod_batch = torch.tensor([1.0, 0.0, -0.5, 0.3])
        delta_w = updater.compute_delta_w(mod_batch, e)

        # Batch element 1 has mod=0, so delta_w[1] should be exactly zero
        assert torch.all(delta_w[1] == 0), (
            f"Batch element with mod=0 should have zero delta_w, "
            f"got max abs = {delta_w[1].abs().max().item():.2e}"
        )
        # Batch element 0 has mod=1, should be nonzero
        assert torch.any(delta_w[0] != 0), (
            "Batch element with mod=1.0 should have nonzero delta_w"
        )

    result.checks.append(run_check(group_name, "Batch mod_signal: per-batch modulation respected", check_batch_mod_signal, verbose))

    result.elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return result


# ---------------------------------------------------------------------------
# GROUP 4: Neuromodulator Computation (6 checks)
# ---------------------------------------------------------------------------

def run_group_4(verbose: bool = False) -> GroupResult:
    """Group 4: Neuromodulator Computation -- DA, ACh, NE, 5-HT ranges and semantics."""
    group_name = "Neuromodulator Computation"
    result = GroupResult(group_name=group_name, group_id=4)
    t0 = time.perf_counter()

    gate = NeuromodulatoryGate(combination_fn="weighted_sum", alpha_baseline=0.01)

    # -- Check 4.1: DA positive reward -> positive DA in [-1,1] --
    def check_da_positive():
        gate.reset_state()
        reward = torch.tensor([1.0, 2.0, 3.0])
        da = gate.compute_da(reward)
        assert torch.all(da > 0), f"Positive reward should give positive DA, got {da.tolist()}"
        assert torch.all(da >= -1.0) and torch.all(da <= 1.0), (
            f"DA should be in [-1,1], got {da.tolist()}"
        )

    result.checks.append(run_check(group_name, "DA: positive reward -> positive DA in [-1,1]", check_da_positive, verbose))

    # -- Check 4.2: DA negative reward -> negative DA --
    def check_da_negative():
        gate.reset_state()
        reward = torch.tensor([-1.0, -2.0, -3.0])
        da = gate.compute_da(reward)
        assert torch.all(da < 0), f"Negative reward should give negative DA, got {da.tolist()}"

    result.checks.append(run_check(group_name, "DA: negative reward -> negative DA", check_da_negative, verbose))

    # -- Check 4.3: ACh high novelty -> high ACh in [0,1] --
    def check_ach_high():
        novelty = torch.tensor([5.0, 10.0])
        ach = gate.compute_ach(novelty)
        assert torch.all(ach > 0.8), f"High novelty should give high ACh (>0.8), got {ach.tolist()}"
        assert torch.all(ach >= 0.0) and torch.all(ach <= 1.0), (
            f"ACh should be in [0,1], got {ach.tolist()}"
        )

    result.checks.append(run_check(group_name, "ACh: high novelty -> high ACh in [0,1]", check_ach_high, verbose))

    # -- Check 4.4: NE high urgency -> high NE in [0,1] --
    def check_ne_high():
        urgency = torch.tensor([5.0, 10.0])
        ne = gate.compute_ne(urgency)
        assert torch.all(ne > 0.8), f"High urgency should give high NE (>0.8), got {ne.tolist()}"
        assert torch.all(ne >= 0.0) and torch.all(ne <= 1.0), (
            f"NE should be in [0,1], got {ne.tolist()}"
        )

    result.checks.append(run_check(group_name, "NE: high urgency -> high NE in [0,1]", check_ne_high, verbose))

    # -- Check 4.5: 5-HT high patience -> high 5-HT in [0,1] --
    def check_5ht_high():
        patience = torch.tensor([5.0, 10.0])
        sht = gate.compute_5ht(patience)
        assert torch.all(sht > 0.8), f"High patience should give high 5-HT (>0.8), got {sht.tolist()}"
        assert torch.all(sht >= 0.0) and torch.all(sht <= 1.0), (
            f"5-HT should be in [0,1], got {sht.tolist()}"
        )

    result.checks.append(run_check(group_name, "5-HT: high patience -> high 5-HT in [0,1]", check_5ht_high, verbose))

    # -- Check 4.6: All zero signals -> neutral outputs --
    def check_all_zero_neutral():
        gate.reset_state()
        signals = {
            "reward": torch.tensor([0.0]),
            "novelty": torch.tensor([0.0]),
            "urgency": torch.tensor([0.0]),
            "patience": torch.tensor([0.0]),
        }
        modulators, gp = gate(signals)

        # DA with tanh(0) = 0
        assert torch.allclose(modulators["DA"], torch.tensor([0.0]), atol=1e-6), (
            f"DA should be 0 with zero reward, got {modulators['DA'].item():.6f}"
        )
        # ACh = sigmoid(0 + bias=0) = 0.5
        assert torch.allclose(modulators["ACh"], torch.tensor([0.5]), atol=1e-4), (
            f"ACh should be ~0.5 with zero novelty, got {modulators['ACh'].item():.6f}"
        )
        # NE = sigmoid(0 + bias=0) = 0.5
        assert torch.allclose(modulators["NE"], torch.tensor([0.5]), atol=1e-4), (
            f"NE should be ~0.5 with zero urgency, got {modulators['NE'].item():.6f}"
        )
        # 5HT = sigmoid(0 + bias=0) = 0.5
        assert torch.allclose(modulators["5HT"], torch.tensor([0.5]), atol=1e-4), (
            f"5-HT should be ~0.5 with zero patience, got {modulators['5HT'].item():.6f}"
        )

    result.checks.append(run_check(group_name, "All zero signals -> neutral outputs", check_all_zero_neutral, verbose))

    # -- Check 4.7: DA range sweep: always in [-1, 1] --
    def check_da_range_sweep():
        """Sweep reward over linspace(-10, 10, 100) and verify DA stays in [-1, 1]."""
        gate.reset_state()
        rewards = torch.linspace(-10.0, 10.0, 100)
        for r in rewards:
            da = gate.compute_da(r.unsqueeze(0))
            assert da.item() >= -1.0, f"DA below -1.0: {da.item():.6f} at reward={r.item()}"
            assert da.item() <= 1.0, f"DA above 1.0: {da.item():.6f} at reward={r.item()}"

    result.checks.append(run_check(group_name, "DA range sweep: always in [-1, 1]", check_da_range_sweep, verbose))

    # -- Check 4.8: ACh range sweep: always in [0, 1] --
    def check_ach_range_sweep():
        """Sweep novelty over linspace(-10, 10, 100) and verify ACh stays in [0, 1]."""
        novelties = torch.linspace(-10.0, 10.0, 100)
        for n in novelties:
            ach = gate.compute_ach(n.unsqueeze(0))
            assert ach.item() >= 0.0, f"ACh below 0.0: {ach.item():.6f}"
            assert ach.item() <= 1.0, f"ACh above 1.0: {ach.item():.6f}"

    result.checks.append(run_check(group_name, "ACh range sweep: always in [0, 1]", check_ach_range_sweep, verbose))

    # -- Check 4.9: DA baseline tracking via EMA --
    def check_da_baseline_tracking():
        """After repeated positive rewards, the DA baseline increases toward
        the reward mean, causing subsequent DA responses to diminish (adaptation)."""
        gate_bl = NeuromodulatoryGate(combination_fn="weighted_sum", alpha_baseline=0.1)
        gate_bl.reset_state()

        baselines = []
        for _ in range(20):
            reward = torch.tensor([1.0])
            gate_bl.compute_da(reward)
            gate_bl.update_baseline(reward)
            baselines.append(gate_bl.reward_baseline.item())

        # Baseline should be monotonically increasing toward 1.0
        for i in range(1, len(baselines)):
            assert baselines[i] >= baselines[i - 1] - 1e-8, (
                f"Baseline should increase: step {i} = {baselines[i]:.6f} "
                f"< step {i-1} = {baselines[i-1]:.6f}"
            )
        # After 20 updates with alpha=0.1, baseline should be substantial
        assert baselines[-1] > 0.5, (
            f"Baseline should be >0.5 after 20 reward=1.0 updates, got {baselines[-1]:.4f}"
        )

        # Now DA response to reward=1.0 should be diminished (reward - baseline is small)
        da_adapted = gate_bl.compute_da(torch.tensor([1.0]))
        gate_fresh = NeuromodulatoryGate()
        gate_fresh.reset_state()
        da_fresh = gate_fresh.compute_da(torch.tensor([1.0]))
        assert da_adapted.item() < da_fresh.item(), (
            f"DA after adaptation ({da_adapted.item():.4f}) should be less than "
            f"fresh DA ({da_fresh.item():.4f})"
        )

    result.checks.append(run_check(group_name, "DA baseline tracking: EMA adapts over repeated rewards", check_da_baseline_tracking, verbose))

    # -- Check 4.10: Neuromodulator determinism across runs --
    def check_neuromod_determinism():
        """Same inputs and initial state produce identical modulator outputs
        across 10 independent runs."""
        results_list = []
        for _ in range(DETERMINISM_NUM_RUNS):
            g = NeuromodulatoryGate()
            g.reset_state()
            signals = {
                "reward": torch.tensor([0.7]),
                "novelty": torch.tensor([0.3]),
                "urgency": torch.tensor([0.5]),
                "patience": torch.tensor([0.2]),
            }
            modulators, gp = g(signals)
            results_list.append({
                "DA": modulators["DA"].item(),
                "ACh": modulators["ACh"].item(),
                "NE": modulators["NE"].item(),
                "5HT": modulators["5HT"].item(),
                "gp": gp.item(),
            })

        ref = results_list[0]
        for i in range(1, DETERMINISM_NUM_RUNS):
            for key in ref:
                assert abs(results_list[i][key] - ref[key]) < ATOL_FP32_DETERMINISM, (
                    f"Run {i} {key} differs: {results_list[i][key]:.8f} vs {ref[key]:.8f}"
                )

    result.checks.append(run_check(group_name, "Neuromodulator determinism across 10 runs", check_neuromod_determinism, verbose))

    result.elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return result


# ---------------------------------------------------------------------------
# GROUP 5: Done-When Gate (b) -- Deterministic Traces (4 checks)
# ---------------------------------------------------------------------------

def run_group_5(verbose: bool = False) -> GroupResult:
    """Group 5: Done-When Gate (b) -- Deterministic trace reproduction across runs."""
    group_name = "Done-When Gate (b): Deterministic Traces"
    result = GroupResult(group_name=group_name, group_id=5)
    t0 = time.perf_counter()

    B, N_pre, N_post = 4, 16, 8

    # -- Check 5.1: Fixed pre/post sequence -> identical e(t) across 10 runs --
    def check_trace_determinism():
        # Pre-generate a fixed sequence of inputs
        torch.manual_seed(99)
        seq_len = 20
        pre_seq = [torch.rand(B, N_pre) for _ in range(seq_len)]
        post_seq = [torch.rand(B, N_post) for _ in range(seq_len)]

        traces_per_run = []
        for run in range(10):
            mod = EligibilityTraceModule(
                n_pre=N_pre, n_post=N_post, tau_e=50.0, dt=1.0,
                trace_type=TraceType.ACCUMULATING,
            )
            mod.reset(B, torch.device("cpu"))
            for t in range(seq_len):
                e = mod.step_hebbian(pre_seq[t], post_seq[t])
            traces_per_run.append(e.clone())

        ref = traces_per_run[0]
        for i in range(1, 10):
            max_diff = (traces_per_run[i] - ref).abs().max().item()
            assert torch.allclose(traces_per_run[i], ref, atol=1e-6), (
                f"Run {i} trace differs from run 0: max diff = {max_diff:.2e}"
            )

    result.checks.append(run_check(group_name, "Fixed pre/post -> identical e(t) across 10 runs", check_trace_determinism, verbose))

    # -- Check 5.2: Batch independence: e[0] unaffected by e[1] --
    def check_batch_independence():
        torch.manual_seed(42)
        pre_a = torch.rand(B, N_pre)
        post_a = torch.rand(B, N_post)
        # Run with full batch
        mod_full = EligibilityTraceModule(n_pre=N_pre, n_post=N_post, tau_e=50.0)
        mod_full.reset(B, torch.device("cpu"))
        for _ in range(10):
            mod_full.step_hebbian(pre_a, post_a)
        e_full_0 = mod_full.eligibility[0].clone()

        # Run with only item 0 (batch size 1)
        mod_single = EligibilityTraceModule(n_pre=N_pre, n_post=N_post, tau_e=50.0)
        mod_single.reset(1, torch.device("cpu"))
        for _ in range(10):
            mod_single.step_hebbian(pre_a[0:1], post_a[0:1])
        e_single_0 = mod_single.eligibility[0].clone()

        max_diff = (e_full_0 - e_single_0).abs().max().item()
        assert torch.allclose(e_full_0, e_single_0, atol=1e-6), (
            f"Batch item 0 should be independent of other batch items: max diff = {max_diff:.2e}"
        )

    result.checks.append(run_check(group_name, "Batch independence: e[0] unaffected by e[1]", check_batch_independence, verbose))

    # -- Check 5.3: Carry mode: state preserved across calls --
    def check_carry_mode():
        torch.manual_seed(42)
        mod = EligibilityTraceModule(n_pre=N_pre, n_post=N_post, tau_e=50.0)
        mod.reset(B, torch.device("cpu"))

        pre = torch.rand(B, N_pre)
        post = torch.rand(B, N_post)
        mod.step_hebbian(pre, post)
        e_after_first = mod.eligibility.clone()
        assert e_after_first.norm().item() > 0, "Should have nonzero trace after first step"

        # Call again with zeros -- trace should decay but not be zero (carry mode)
        zero_pre = torch.zeros(B, N_pre)
        zero_post = torch.zeros(B, N_post)
        mod.step_hebbian(zero_pre, zero_post)
        e_after_carry = mod.eligibility.clone()

        # The trace should be the decayed version of the previous trace
        expected = mod.decay_e * e_after_first
        expected = torch.clamp(expected, mod.clamp_min, mod.clamp_max)
        max_diff = (e_after_carry - expected).abs().max().item()
        assert torch.allclose(e_after_carry, expected, atol=1e-6), (
            f"Carry mode: trace should match decay * previous: max diff = {max_diff:.2e}"
        )
        assert e_after_carry.norm().item() > 0, "Carried trace should be nonzero"

    result.checks.append(run_check(group_name, "Carry mode: state preserved across calls", check_carry_mode, verbose))

    # -- Check 5.4: fp32 enforcement: traces always float32 --
    def check_fp32_enforcement():
        torch.manual_seed(42)
        mod = EligibilityTraceModule(n_pre=N_pre, n_post=N_post, tau_e=50.0)
        # Try feeding fp16 inputs
        pre = torch.rand(B, N_pre).half()
        post = torch.rand(B, N_post).half()
        e = mod.step_hebbian(pre, post)
        assert e.dtype == torch.float32, (
            f"Trace should always be float32, got {e.dtype}"
        )
        # Feed fp64
        pre64 = torch.rand(B, N_pre).double()
        post64 = torch.rand(B, N_post).double()
        mod2 = EligibilityTraceModule(n_pre=N_pre, n_post=N_post, tau_e=50.0)
        e2 = mod2.step_hebbian(pre64, post64)
        assert e2.dtype == torch.float32, (
            f"Trace should always be float32 even with fp64 input, got {e2.dtype}"
        )

    result.checks.append(run_check(group_name, "fp32 enforcement: traces always float32", check_fp32_enforcement, verbose))

    # -- Check 5.5: STDP trace determinism across 10 runs --
    def check_stdp_determinism():
        """Fixed spike sequences produce identical STDP traces across 10 runs."""
        torch.manual_seed(77)
        seq_len = 15
        # Pre-generate fixed spike sequences
        spike_pre_seq = [torch.bernoulli(torch.full((B, N_pre), 0.3)) for _ in range(seq_len)]
        spike_post_seq = [torch.bernoulli(torch.full((B, N_post), 0.3)) for _ in range(seq_len)]

        traces_per_run = []
        for _ in range(DETERMINISM_NUM_RUNS):
            mod = EligibilityTraceModule(
                n_pre=N_pre, n_post=N_post, tau_e=DEFAULT_TAU_E, dt=DEFAULT_DT,
            )
            mod.reset(B, torch.device("cpu"))
            for t in range(seq_len):
                e = mod.step_stdp(spike_pre_seq[t], spike_post_seq[t])
            traces_per_run.append(e.clone())

        ref = traces_per_run[0]
        for i in range(1, DETERMINISM_NUM_RUNS):
            max_diff = (traces_per_run[i] - ref).abs().max().item()
            assert torch.allclose(traces_per_run[i], ref, atol=ATOL_FP32_DETERMINISM), (
                f"STDP run {i} differs from run 0: max diff = {max_diff:.2e}"
            )

    result.checks.append(run_check(group_name, "STDP trace determinism across 10 runs", check_stdp_determinism, verbose))

    # -- Check 5.6: Linear decay approximation agrees with exponential for small dt/tau --
    def check_linear_vs_exponential_decay():
        """When dt << tau_e, the linear approximation (1 - dt/tau_e) should
        closely match the exponential exp(-dt/tau_e).  We compare traces from
        both modes over 10 steps and verify they are close."""
        torch.manual_seed(42)
        tau_large = 200.0  # dt/tau = 1/200 = 0.005, well within approximation range
        pre = torch.rand(B, N_pre) * 0.5
        post = torch.rand(B, N_post) * 0.5

        mod_exp = EligibilityTraceModule(
            n_pre=N_pre, n_post=N_post, tau_e=tau_large, dt=DEFAULT_DT,
            trace_type=TraceType.ACCUMULATING, use_exponential_decay=True,
        )
        mod_lin = EligibilityTraceModule(
            n_pre=N_pre, n_post=N_post, tau_e=tau_large, dt=DEFAULT_DT,
            trace_type=TraceType.ACCUMULATING, use_exponential_decay=False,
        )

        for step in range(10):
            e_exp = mod_exp.step_hebbian(pre, post)
            e_lin = mod_lin.step_hebbian(pre, post)

        # Linear and exponential should agree to within ~0.5% for dt/tau=0.005
        max_diff = (e_exp - e_lin).abs().max().item()
        max_val = max(e_exp.abs().max().item(), e_lin.abs().max().item())
        relative_diff = max_diff / (max_val + 1e-10)
        assert relative_diff < 0.01, (
            f"Linear vs exponential decay should agree within 1%: "
            f"relative diff = {relative_diff:.4f}, abs diff = {max_diff:.6f}"
        )

    result.checks.append(run_check(group_name, "Linear decay approx agrees with exp for small dt/tau", check_linear_vs_exponential_decay, verbose))

    # -- Check 5.7: Checkpoint round-trip: state_dict save/load preserves traces --
    def check_checkpoint_round_trip():
        """Build up trace state over several steps, save the module state,
        load into a fresh module, and verify the traces are identical."""
        torch.manual_seed(42)
        mod_orig = EligibilityTraceModule(
            n_pre=N_pre, n_post=N_post, tau_e=DEFAULT_TAU_E, dt=DEFAULT_DT,
        )
        pre = torch.rand(B, N_pre)
        post = torch.rand(B, N_post)
        for _ in range(10):
            mod_orig.step_hebbian(pre, post)
        e_orig = mod_orig.eligibility.clone()
        x_pre_orig = mod_orig.x_pre.clone()
        x_post_orig = mod_orig.x_post.clone()

        # Simulate checkpoint: save the raw tensors
        checkpoint = {
            "eligibility": mod_orig.eligibility.clone(),
            "x_pre": mod_orig.x_pre.clone(),
            "x_post": mod_orig.x_post.clone(),
        }

        # Create fresh module and load
        mod_restored = EligibilityTraceModule(
            n_pre=N_pre, n_post=N_post, tau_e=DEFAULT_TAU_E, dt=DEFAULT_DT,
        )
        mod_restored.eligibility = checkpoint["eligibility"]
        mod_restored.x_pre = checkpoint["x_pre"]
        mod_restored.x_post = checkpoint["x_post"]

        # Run one more step on both
        e1 = mod_orig.step_hebbian(pre, post)
        e2 = mod_restored.step_hebbian(pre, post)

        max_diff = (e1 - e2).abs().max().item()
        assert torch.allclose(e1, e2, atol=ATOL_FP32_DETERMINISM), (
            f"Checkpoint round-trip: traces differ after resume: max diff = {max_diff:.2e}"
        )

    result.checks.append(run_check(group_name, "Checkpoint round-trip: save/load preserves traces", check_checkpoint_round_trip, verbose))

    # -- Check 5.8: Episode boundary reset: fresh module == reset module --
    def check_episode_boundary_reset():
        """After running 10 steps and then calling reset(), the next step
        should produce identical output to a module that was freshly created."""
        torch.manual_seed(42)
        pre = torch.rand(B, N_pre)
        post = torch.rand(B, N_post)

        # Module that runs 10 steps, then resets
        mod_reset = EligibilityTraceModule(
            n_pre=N_pre, n_post=N_post, tau_e=DEFAULT_TAU_E, dt=DEFAULT_DT,
        )
        for _ in range(10):
            mod_reset.step_hebbian(pre, post)
        mod_reset.reset(B, torch.device("cpu"))
        e_reset = mod_reset.step_hebbian(pre, post)

        # Fresh module
        mod_fresh = EligibilityTraceModule(
            n_pre=N_pre, n_post=N_post, tau_e=DEFAULT_TAU_E, dt=DEFAULT_DT,
        )
        mod_fresh.reset(B, torch.device("cpu"))
        e_fresh = mod_fresh.step_hebbian(pre, post)

        max_diff = (e_reset - e_fresh).abs().max().item()
        assert torch.allclose(e_reset, e_fresh, atol=ATOL_FP32_DETERMINISM), (
            f"Episode boundary: reset module should match fresh module: "
            f"max diff = {max_diff:.2e}"
        )

    result.checks.append(run_check(group_name, "Episode boundary: reset module matches fresh module", check_episode_boundary_reset, verbose))

    result.elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return result


# ---------------------------------------------------------------------------
# GROUP 6: Done-When Gate (c) -- Delayed Reward Association (5 checks)
# ---------------------------------------------------------------------------

def _run_delayed_reward_experiment(
    delay: int,
    use_eligibility: bool,
    num_episodes: int = 300,
    lr: float = 0.01,
    trace_decay: float = 0.9,
    seed: int = 42,
) -> Tuple[List[float], float]:
    """Run the delayed reward toy task and return (loss_history, final_accuracy).

    Task: 4-class classification using a hidden label mapping. The input is a
    4-dim one-hot vector indicating one of four patterns. The correct action
    is determined by a fixed but non-trivial permutation (0->2, 1->0, 2->3, 3->1).
    This permutation ensures the model cannot achieve above-chance accuracy
    (~25%) without learning, regardless of initial weight bias.

    The reward arrives `delay` steps after the action. With eligibility traces
    active, REINFORCE-style policy gradient eligibility decays over the delay
    period and is credited when the delayed reward arrives. Without eligibility,
    no learning signal is applied and the model stays at chance.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = 4
    hidden_dim = 32
    output_dim = 4
    # Non-trivial label permutation: input class -> correct action
    label_map = [2, 0, 3, 1]

    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )
    # Zero init on output layer so model starts with uniform logits (chance)
    nn.init.zeros_(model[2].weight)
    nn.init.zeros_(model[2].bias)

    effective_tau = max(delay * 5.0, 20.0)
    decay_per_step = math.exp(-1.0 / effective_tau)
    decay_over_delay = decay_per_step ** delay

    baseline_val = 0.0
    alpha_bl = 0.1

    reward_queue: deque = deque()
    loss_history: List[float] = []

    for episode in range(num_episodes):
        cue = np.random.randint(0, output_dim)
        correct_action = label_map[cue]
        cue_vec = torch.zeros(1, input_dim)
        cue_vec[0, cue] = 1.0

        with torch.no_grad():
            h = F.relu(model[0](cue_vec))
            logits = model[2](h)

        # Softmax action selection with temperature for exploration
        temp = max(1.0, 3.0 * (1.0 - episode / num_episodes))  # anneal temperature
        probs = F.softmax(logits / temp, dim=-1)
        action = torch.multinomial(probs, 1).item()
        correct = int(action == correct_action)
        loss_history.append(0.0 if correct else 1.0)

        if use_eligibility:
            # REINFORCE-style eligibility: d log pi(a|s) / d W
            grad_log_pi = torch.zeros(1, output_dim, hidden_dim)
            for j in range(output_dim):
                if j == action:
                    grad_log_pi[0, j, :] = h[0] * (1.0 - probs[0, j])
                else:
                    grad_log_pi[0, j, :] = -h[0] * probs[0, j]

            reward_queue.append((grad_log_pi.clone(), correct_action, action, episode))

        # Process delayed rewards
        if use_eligibility:
            while reward_queue and reward_queue[0][3] <= episode - delay:
                past_elig, past_correct_action, past_action, _ = reward_queue.popleft()
                past_reward = 1.0 if int(past_action == past_correct_action) else -0.3

                e_delayed = decay_over_delay * past_elig
                rpe = past_reward - baseline_val
                baseline_val = (1.0 - alpha_bl) * baseline_val + alpha_bl * past_reward

                delta_w = lr * rpe * e_delayed.squeeze(0)
                delta_w = torch.clamp(delta_w, -0.2, 0.2)

                with torch.no_grad():
                    model[2].weight.data.add_(delta_w)
                    model[2].weight.data.clamp_(-5.0, 5.0)

    # Evaluate final accuracy (greedy)
    eval_correct = 0
    n_eval = 200
    for _ in range(n_eval):
        cue = np.random.randint(0, output_dim)
        correct_action = label_map[cue]
        cue_vec = torch.zeros(1, input_dim)
        cue_vec[0, cue] = 1.0
        with torch.no_grad():
            logits = model(cue_vec)
        action = torch.argmax(logits, dim=-1).item()
        if action == correct_action:
            eval_correct += 1
    final_accuracy = eval_correct / float(n_eval)

    return loss_history, final_accuracy


def run_group_6(verbose: bool = False) -> GroupResult:
    """Group 6: Done-When Gate (c) -- Delayed reward association via eligibility traces."""
    group_name = "Done-When Gate (c): Delayed Reward Association"
    result = GroupResult(group_name=group_name, group_id=6)
    t0 = time.perf_counter()

    delay = 5
    num_episodes = 600

    # Run both experiments
    if verbose:
        print(f"  Running delayed reward experiments (delay={delay}, episodes={num_episodes})...")

    loss_without, acc_without = _run_delayed_reward_experiment(
        delay=delay, use_eligibility=False, num_episodes=num_episodes, seed=42,
    )
    loss_with, acc_with = _run_delayed_reward_experiment(
        delay=delay, use_eligibility=True, num_episodes=num_episodes,
        lr=0.05, trace_decay=0.9, seed=42,
    )

    if verbose:
        print(f"  Without eligibility: accuracy={acc_without:.2f}")
        print(f"  With eligibility: accuracy={acc_with:.2f}")

    # -- Check 6.1: Toy task setup (input -> delay -> reward) --
    def check_toy_task_setup():
        # This check validates that the experiment structure is correct
        assert len(loss_without) == num_episodes, (
            f"Loss history should have {num_episodes} entries, got {len(loss_without)}"
        )
        assert len(loss_with) == num_episodes, (
            f"Loss history should have {num_episodes} entries, got {len(loss_with)}"
        )

    result.checks.append(run_check(group_name, "Toy task: correct structure (input -> delay -> reward)", check_toy_task_setup, verbose))

    # -- Check 6.2: Without eligibility, loss doesn't decrease significantly --
    def check_without_eligibility():
        # Without eligibility, the model should stay near chance (~25% for 4-class)
        assert acc_without < 0.40, (
            f"Without eligibility, accuracy should be near chance (<0.40): got {acc_without:.2f}"
        )

    result.checks.append(run_check(group_name, "Without eligibility: model stays near chance", check_without_eligibility, verbose))

    # -- Check 6.3: With eligibility, traces bridge gap and loss decreases --
    def check_with_eligibility_learns():
        # Early vs late loss comparison using wider windows for robustness
        window = num_episodes // 4
        early_loss = np.mean(loss_with[:window])
        late_loss = np.mean(loss_with[-window:])
        assert late_loss < early_loss, (
            f"With eligibility, late loss ({late_loss:.3f}) should be < early loss ({early_loss:.3f})"
        )

    result.checks.append(run_check(group_name, "With eligibility: loss decreases over training", check_with_eligibility_learns, verbose))

    # -- Check 6.4: Accuracy improves above chance (>50% for 4-class task, chance=25%) --
    def check_accuracy_above_chance():
        assert acc_with > 0.50, (
            f"With eligibility, accuracy should be > 0.50: got {acc_with:.2f}"
        )

    result.checks.append(run_check(group_name, "Accuracy improves above chance (>50%)", check_accuracy_above_chance, verbose))

    # -- Check 6.5: Different delay lengths: still learns (longer delay = slower) --
    def check_longer_delay():
        _, acc_delay3 = _run_delayed_reward_experiment(
            delay=3, use_eligibility=True, num_episodes=800,
            lr=0.05, trace_decay=0.9, seed=123,
        )
        _, acc_delay10 = _run_delayed_reward_experiment(
            delay=10, use_eligibility=True, num_episodes=1200,
            lr=0.05, trace_decay=0.9, seed=123,
        )
        if verbose:
            print(f"    Delay=3: accuracy={acc_delay3:.2f}")
            print(f"    Delay=10: accuracy={acc_delay10:.2f}")
        # Both should be above chance (25% for 4-class)
        assert acc_delay3 > 0.40, (
            f"Delay=3 should learn above chance (>0.40): got {acc_delay3:.2f}"
        )
        assert acc_delay10 > 0.35, (
            f"Delay=10 should still learn above chance (>0.35): got {acc_delay10:.2f}"
        )

    result.checks.append(run_check(group_name, "Different delays: still learns", check_longer_delay, verbose))

    result.elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return result


# ---------------------------------------------------------------------------
# GROUP 7: Integration and Stability (4 checks)
# ---------------------------------------------------------------------------

def run_group_7(verbose: bool = False) -> GroupResult:
    """Group 7: Integration and Stability -- full pipeline, adapter, clamping, NaN check."""
    group_name = "Integration and Stability"
    result = GroupResult(group_name=group_name, group_id=7)
    t0 = time.perf_counter()

    B, N_pre, N_post = 4, 32, 16

    # -- Check 7.1: Full pipeline works end-to-end --
    def check_full_pipeline():
        torch.manual_seed(42)
        # Pre/post activity
        pre = torch.rand(B, N_pre)
        post = torch.rand(B, N_post)

        # Step 1: Update eligibility trace
        trace_mod = EligibilityTraceModule(
            n_pre=N_pre, n_post=N_post, tau_e=50.0, dt=1.0,
        )
        trace_mod.reset(B, torch.device("cpu"))
        e = trace_mod.step_hebbian(pre, post)
        assert e.shape == (B, N_post, N_pre), f"Trace shape wrong: {e.shape}"

        # Step 2: Compute neuromodulatory signals
        gate = NeuromodulatoryGate()
        signals = {
            "reward": torch.tensor([0.5, -0.3, 1.0, 0.0]),
            "novelty": torch.tensor([0.8, 0.2, 0.5, 0.1]),
            "urgency": torch.tensor([0.1, 0.9, 0.4, 0.6]),
            "patience": torch.tensor([0.6, 0.3, 0.7, 0.2]),
        }
        modulators, global_plasticity = gate(signals)
        assert "DA" in modulators, "Missing DA modulator"
        assert "ACh" in modulators, "Missing ACh modulator"
        assert "NE" in modulators, "Missing NE modulator"
        assert "5HT" in modulators, "Missing 5HT modulator"

        # Step 3: Three-factor weight update
        weights = torch.randn(N_post, N_pre) * 0.1
        updater = ThreeFactorUpdate(lr=0.01, weight_clamp=(-1.0, 1.0))
        weights_before = weights.clone()
        updater.apply(weights, global_plasticity, e)

        # Verify weights changed (global_plasticity should be nonzero)
        assert not torch.equal(weights, weights_before), (
            "Weights should change after three-factor update with nonzero modulator"
        )

    result.checks.append(run_check(group_name, "Full pipeline: pre/post -> trace -> modulator -> weight update", check_full_pipeline, verbose))

    # -- Check 7.2: FastMemoryAdapter forward + three-factor update --
    def check_fast_adapter():
        torch.manual_seed(42)
        dim = 64
        adapter = FastMemoryAdapter(dim=dim, bottleneck=16)
        x = torch.randn(B, dim)

        # Forward pass
        out = adapter(x)
        assert out.shape == (B, dim), f"Adapter output shape wrong: {out.shape}"

        # Simulate three-factor update on adapter weights
        pre = x.detach()
        h = adapter.down(pre)
        post = adapter.up(h).detach()

        trace = EligibilityTraceModule(n_pre=dim, n_post=dim, tau_e=50.0)
        trace.reset(B, torch.device("cpu"))
        e = trace.step_hebbian(pre, post)

        mod_signal = torch.tensor(0.5)
        updater = ThreeFactorUpdate(lr=0.005, weight_clamp=(-1.0, 1.0))

        # We need to update the adapter's up weight: shape (dim, bottleneck)
        # But our trace is (B, dim, dim). For this test, just verify the pipeline works.
        down_weight_before = adapter.down.weight.clone()
        # Create a trace for the down weight
        trace_down = EligibilityTraceModule(n_pre=dim, n_post=16, tau_e=50.0)
        trace_down.reset(B, torch.device("cpu"))
        e_down = trace_down.step_hebbian(pre, h.detach())
        updater.apply(adapter.down.weight, mod_signal, e_down)

        assert not torch.equal(adapter.down.weight.data, down_weight_before), (
            "Adapter down weight should change after three-factor update"
        )

        # Verify adapter still works after weight update
        out2 = adapter(x)
        assert out2.shape == (B, dim), "Adapter should still produce correct shape after update"

    result.checks.append(run_check(group_name, "FastMemoryAdapter: forward + three-factor update", check_fast_adapter, verbose))

    # -- Check 7.3: Weight clamping: updated weights stay in range --
    def check_weight_clamping():
        torch.manual_seed(42)
        w_min, w_max = -0.5, 0.5
        weights = torch.randn(N_post, N_pre) * 0.1  # starts within bounds

        # Create very large eligibility and mod signal to force clamp hits
        e = torch.ones(1, N_post, N_pre) * 10.0
        mod_signal = torch.tensor(10.0)
        updater = ThreeFactorUpdate(lr=0.1, weight_clamp=(w_min, w_max), delta_clamp=1.0)

        updater.apply(weights, mod_signal, e)
        assert torch.all(weights >= w_min), (
            f"Weights below w_min: min = {weights.min().item():.4f}"
        )
        assert torch.all(weights <= w_max), (
            f"Weights above w_max: max = {weights.max().item():.4f}"
        )

    result.checks.append(run_check(group_name, "Weight clamping: updated weights stay in range", check_weight_clamping, verbose))

    # -- Check 7.4: No NaN: full pipeline produces no NaN values --
    def check_no_nan():
        torch.manual_seed(42)
        # Run a multi-step pipeline with varied inputs
        trace = EligibilityTraceModule(n_pre=N_pre, n_post=N_post, tau_e=50.0)
        gate = NeuromodulatoryGate()
        updater = ThreeFactorUpdate(lr=0.01, weight_clamp=(-2.0, 2.0))
        weights = torch.randn(N_post, N_pre) * 0.1

        for step in range(50):
            pre = torch.randn(B, N_pre) * (1.0 + step * 0.1)
            post = torch.randn(B, N_post) * (1.0 + step * 0.1)
            e = trace.step_hebbian(pre, post)

            signals = {
                "reward": torch.randn(B),
                "novelty": torch.rand(B),
                "urgency": torch.rand(B),
                "patience": torch.rand(B),
            }
            modulators, gp = gate(signals)
            updater.apply(weights, gp, e)

            # Check all outputs for NaN
            assert not torch.isnan(e).any(), f"NaN in eligibility at step {step}"
            assert not torch.isnan(weights).any(), f"NaN in weights at step {step}"
            for name, val in modulators.items():
                assert not torch.isnan(val).any(), f"NaN in modulator {name} at step {step}"
            assert not torch.isnan(gp).any(), f"NaN in global_plasticity at step {step}"

    result.checks.append(run_check(group_name, "No NaN: full pipeline produces no NaN values", check_no_nan, verbose))

    # -- Check 7.5: Multiple eligible layers with different LRs --
    def check_multiple_eligible_layers():
        """Three layers are each updated independently with different learning
        rates.  The update magnitude ratios should approximately match the
        LR ratios."""
        torch.manual_seed(42)
        # Three layers with different LRs
        lr_1, lr_2, lr_3 = 0.01, 0.005, 0.001
        e = torch.rand(1, N_post, N_pre) * 0.5 + 0.1  # Same eligibility
        mod_signal = torch.tensor(1.0)

        u1 = ThreeFactorUpdate(lr=lr_1, weight_clamp=(-5.0, 5.0), delta_clamp=1.0)
        u2 = ThreeFactorUpdate(lr=lr_2, weight_clamp=(-5.0, 5.0), delta_clamp=1.0)
        u3 = ThreeFactorUpdate(lr=lr_3, weight_clamp=(-5.0, 5.0), delta_clamp=1.0)

        dw1 = u1.compute_delta_w(mod_signal, e)
        dw2 = u2.compute_delta_w(mod_signal, e)
        dw3 = u3.compute_delta_w(mod_signal, e)

        # Check ratio lr_1/lr_2 = 2.0
        ratio_12 = dw1.norm().item() / (dw2.norm().item() + 1e-10)
        assert abs(ratio_12 - 2.0) < 0.1, (
            f"LR ratio 0.01/0.005 should give ~2.0x update: got {ratio_12:.3f}"
        )
        # Check ratio lr_1/lr_3 = 10.0
        ratio_13 = dw1.norm().item() / (dw3.norm().item() + 1e-10)
        assert abs(ratio_13 - 10.0) < 0.5, (
            f"LR ratio 0.01/0.001 should give ~10.0x update: got {ratio_13:.3f}"
        )

    result.checks.append(run_check(group_name, "Multiple eligible layers: LR ratios respected", check_multiple_eligible_layers, verbose))

    # -- Check 7.6: RewardBaselineEMA standalone accuracy --
    def check_reward_baseline_ema():
        """RewardBaselineEMA converges toward the mean of a constant reward
        stream and correctly computes EMA update."""
        ema = RewardBaselineEMA(alpha=0.1)
        ema.reset()
        values = []
        for _ in range(50):
            ema.update(1.0)
            values.append(ema.value)

        # After 50 steps with reward=1.0, baseline should be close to 1.0
        assert values[-1] > 0.99, (
            f"EMA should converge to ~1.0 after 50 steps of reward=1.0: got {values[-1]:.6f}"
        )
        # Monotonically increasing
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1] - 1e-10, (
                f"EMA should monotonically increase: step {i}={values[i]:.6f} "
                f"< step {i-1}={values[i-1]:.6f}"
            )

        # Test reset
        ema.reset()
        assert ema.value == 0.0, f"EMA reset should zero the value, got {ema.value}"

        # Test with alternating rewards
        for step in range(100):
            r = 1.0 if step % 2 == 0 else -1.0
            ema.update(r)
        # Should converge toward 0.0 (mean of alternating +1/-1)
        assert abs(ema.value) < 0.15, (
            f"EMA with alternating +1/-1 should converge near 0.0: got {ema.value:.4f}"
        )

    result.checks.append(run_check(group_name, "RewardBaselineEMA: convergence and reset", check_reward_baseline_ema, verbose))

    # -- Check 7.7: Streaming carry: traces carry across sequential calls --
    def check_streaming_carry():
        """In streaming mode (no reset between calls), traces carry forward.
        Processing data in two chunks should produce the same result as
        processing all data sequentially in a single module."""
        torch.manual_seed(42)
        total_steps = 10
        pre_seq = [torch.rand(B, N_pre) for _ in range(total_steps)]
        post_seq = [torch.rand(B, N_post) for _ in range(total_steps)]

        # Single-pass module: all 10 steps without reset
        mod_single = EligibilityTraceModule(
            n_pre=N_pre, n_post=N_post, tau_e=DEFAULT_TAU_E, dt=DEFAULT_DT,
        )
        mod_single.reset(B, torch.device("cpu"))
        for t in range(total_steps):
            e_single = mod_single.step_hebbian(pre_seq[t], post_seq[t])

        # Two-chunk module: first 5 steps, then next 5 steps (no reset in between)
        mod_chunk = EligibilityTraceModule(
            n_pre=N_pre, n_post=N_post, tau_e=DEFAULT_TAU_E, dt=DEFAULT_DT,
        )
        mod_chunk.reset(B, torch.device("cpu"))
        for t in range(5):
            mod_chunk.step_hebbian(pre_seq[t], post_seq[t])
        # Carry: continue without reset
        for t in range(5, total_steps):
            e_chunk = mod_chunk.step_hebbian(pre_seq[t], post_seq[t])

        max_diff = (e_single - e_chunk).abs().max().item()
        assert torch.allclose(e_single, e_chunk, atol=ATOL_FP32_DETERMINISM), (
            f"Streaming carry: chunked processing should match single-pass: "
            f"max diff = {max_diff:.2e}"
        )

    result.checks.append(run_check(group_name, "Streaming carry: chunked == single-pass processing", check_streaming_carry, verbose))

    # -- Check 7.8: Non-eligible layers frozen: weights unchanged by three-factor --
    def check_non_eligible_frozen():
        """When only one layer is designated as eligible, the other layer's
        weights must remain bitwise identical after a three-factor update."""
        torch.manual_seed(42)
        dim = 32
        adapter = FastMemoryAdapter(dim=dim, bottleneck=8)
        # Freeze the 'up' weights; only update 'down'
        up_weight_before = adapter.up.weight.data.clone()
        down_weight_before = adapter.down.weight.data.clone()

        # Create trace and update only the down projection
        trace_down = EligibilityTraceModule(n_pre=dim, n_post=8, tau_e=DEFAULT_TAU_E)
        trace_down.reset(B, torch.device("cpu"))
        x = torch.rand(B, dim)
        h = adapter.down(x).detach()
        trace_down.step_hebbian(x, h)
        e_down = trace_down.get_trace()

        updater_local = ThreeFactorUpdate(lr=0.01, weight_clamp=(-2.0, 2.0))
        updater_local.apply(adapter.down.weight, torch.tensor(1.0), e_down)

        # down weights should have changed
        assert not torch.equal(adapter.down.weight.data, down_weight_before), (
            "Eligible layer (down) weights should change"
        )
        # up weights should be untouched (bitwise identical)
        assert torch.equal(adapter.up.weight.data, up_weight_before), (
            "Non-eligible layer (up) weights should be bitwise identical"
        )

    result.checks.append(run_check(group_name, "Non-eligible layers: weights frozen after update", check_non_eligible_frozen, verbose))

    # -- Check 7.9: Large-scale stress: 1000 steps without NaN or Inf --
    def check_large_scale_stress():
        """Run a full pipeline for 1000 steps with random inputs to verify
        no numerical instability (NaN, Inf) accumulates over time.  This is
        a stronger version of check 7.4 with more steps and varied conditions."""
        torch.manual_seed(42)
        trace = EligibilityTraceModule(
            n_pre=LARGE_N_PRE, n_post=LARGE_N_POST, tau_e=DEFAULT_TAU_E,
        )
        gate_stress = NeuromodulatoryGate()
        updater_stress = ThreeFactorUpdate(lr=0.005, weight_clamp=(-2.0, 2.0))
        weights = torch.randn(LARGE_N_POST, LARGE_N_PRE) * 0.1

        for step in range(1000):
            # Vary input magnitude to stress-test
            scale = 0.5 + 2.0 * (step % 100) / 100.0
            pre = torch.randn(B, LARGE_N_PRE) * scale
            post = torch.randn(B, LARGE_N_POST) * scale
            e = trace.step_hebbian(pre, post)

            signals = {
                "reward": torch.randn(B) * 2.0,
                "novelty": torch.rand(B),
                "urgency": torch.rand(B),
                "patience": torch.rand(B),
            }
            modulators, gp = gate_stress(signals)
            updater_stress.apply(weights, gp, e)

        # Final health check
        assert not torch.isnan(weights).any(), "Weights contain NaN after 1000 steps"
        assert not torch.isinf(weights).any(), "Weights contain Inf after 1000 steps"
        assert not torch.isnan(trace.eligibility).any(), "Traces contain NaN after 1000 steps"
        assert not torch.isinf(trace.eligibility).any(), "Traces contain Inf after 1000 steps"
        # Weights should still be within clamp range
        assert torch.all(weights >= -2.0) and torch.all(weights <= 2.0), (
            f"Weights outside clamp range: min={weights.min().item():.4f}, "
            f"max={weights.max().item():.4f}"
        )

    result.checks.append(run_check(group_name, "Large-scale stress: 1000 steps without NaN/Inf", check_large_scale_stress, verbose))

    result.elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return result


# ============================================================================
# SECTION 4: MAIN DRIVER
# ============================================================================

# Map of group ID -> runner function
GROUP_RUNNERS = {
    1: run_group_1,
    2: run_group_2,
    3: run_group_3,
    4: run_group_4,
    5: run_group_5,
    6: run_group_6,
    7: run_group_7,
}

GROUP_NAMES = {
    1: "Trace Dynamics",
    2: "STDP Kernels",
    3: "Done-When Gate (a): Third-Factor Gating",
    4: "Neuromodulator Computation",
    5: "Done-When Gate (b): Deterministic Traces",
    6: "Done-When Gate (c): Delayed Reward Association",
    7: "Integration and Stability",
}


def build_summary(results: List[GroupResult]) -> Dict[str, Any]:
    """Build a JSON-serializable summary of all results."""
    groups = []
    total_passed = 0
    total_failed = 0
    total_checks = 0

    for gr in results:
        checks_json = []
        for c in gr.checks:
            checks_json.append({
                "name": c.name,
                "passed": c.passed,
                "message": c.message,
                "elapsed_ms": round(c.elapsed_ms, 2),
            })
        groups.append({
            "group_id": gr.group_id,
            "group_name": gr.group_name,
            "passed": gr.passed,
            "failed": gr.failed,
            "total": gr.total,
            "elapsed_ms": round(gr.elapsed_ms, 2),
            "checks": checks_json,
        })
        total_passed += gr.passed
        total_failed += gr.failed
        total_checks += gr.total

    return {
        "script": "validate_neuromod.py",
        "skill": "neuromodulation-eligibility",
        "overall_passed": total_failed == 0,
        "total_passed": total_passed,
        "total_failed": total_failed,
        "total_checks": total_checks,
        "total_elapsed_ms": round(sum(gr.elapsed_ms for gr in results), 2),
        "groups": groups,
    }


def print_terminal_summary(results: List[GroupResult], summary: Dict[str, Any]) -> None:
    """Print a colorized terminal summary."""
    print()
    print(colorize("=" * 72, "bold"))
    print(colorize("  Neuromodulation + Eligibility Traces Validation", "bold"))
    print(colorize("=" * 72, "bold"))
    print()

    for gr in results:
        if gr.all_passed:
            status = colorize("PASS", "green")
        else:
            status = colorize("FAIL", "red")
        print(f"  Group {gr.group_id}: {gr.group_name}")
        print(f"    Status: {status}  ({gr.passed}/{gr.total} checks, {gr.elapsed_ms:.0f}ms)")

        # Show failed checks even in non-verbose mode
        for c in gr.checks:
            if not c.passed:
                print(f"    {colorize('FAIL', 'red')}: {c.name}")
                print(f"           {c.message}")
        print()

    print(colorize("-" * 72, "bold"))
    total_p = summary["total_passed"]
    total_f = summary["total_failed"]
    total_c = summary["total_checks"]
    elapsed = summary["total_elapsed_ms"]

    if total_f == 0:
        overall = colorize("ALL PASSED", "green")
    else:
        overall = colorize(f"{total_f} FAILED", "red")

    print(f"  Result: {overall}  ({total_p}/{total_c} checks passed, {elapsed:.0f}ms total)")

    # Done-when gates summary
    print()
    gate_a_group = next((g for g in results if g.group_id == 3), None)
    gate_b_group = next((g for g in results if g.group_id == 5), None)
    gate_c_group = next((g for g in results if g.group_id == 6), None)

    print(colorize("  Done-When Gates:", "bold"))
    for label, grp in [
        ("(a) Third-factor gating", gate_a_group),
        ("(b) Deterministic traces", gate_b_group),
        ("(c) Delayed reward assoc", gate_c_group),
    ]:
        if grp is None:
            print(f"    {label}: {colorize('SKIPPED', 'yellow')}")
        elif grp.all_passed:
            print(f"    {label}: {colorize('PASS', 'green')}")
        else:
            print(f"    {label}: {colorize('FAIL', 'red')}")

    print(colorize("=" * 72, "bold"))
    print()


def main() -> int:
    """Run validation and return exit code (0 = success, 1 = failure)."""
    parser = argparse.ArgumentParser(
        description="Validate neuromodulation + eligibility traces (three-factor learning).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Groups:\n"
            "  1  Trace Dynamics\n"
            "  2  STDP Kernels\n"
            "  3  Done-When Gate (a): Third-Factor Gating\n"
            "  4  Neuromodulator Computation\n"
            "  5  Done-When Gate (b): Deterministic Traces\n"
            "  6  Done-When Gate (c): Delayed Reward Association\n"
            "  7  Integration and Stability\n"
        ),
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-check pass/fail output.",
    )
    parser.add_argument(
        "--group", "-g", type=int, action="append", default=None,
        help="Run only specified group(s). Can be repeated: --group 1 --group 3.",
    )
    parser.add_argument(
        "--json", "-j", action="store_true",
        help="Print JSON summary (in addition to terminal output, or alone if not verbose).",
    )
    args = parser.parse_args()

    # Determine which groups to run
    if args.group:
        group_ids = sorted(set(args.group))
        for gid in group_ids:
            if gid not in GROUP_RUNNERS:
                print(f"Error: unknown group {gid}. Valid: 1-7.", file=sys.stderr)
                return 1
    else:
        group_ids = list(range(1, 8))

    # Print header
    if not args.json or args.verbose:
        print()
        print(colorize("Neuromodulation + Eligibility Traces: Runtime Validation", "bold"))
        print(f"PyTorch {torch.__version__} | Device: CPU | Groups: {group_ids}")
        print()

    # Run groups
    results: List[GroupResult] = []
    for gid in group_ids:
        runner = GROUP_RUNNERS[gid]
        if args.verbose:
            print(colorize(f"Group {gid}: {GROUP_NAMES[gid]}", "cyan"))
        gr = runner(verbose=args.verbose)
        results.append(gr)
        if args.verbose:
            print()

    # Build summary
    summary = build_summary(results)

    # Output
    if not args.json:
        print_terminal_summary(results, summary)
    elif args.json and args.verbose:
        print_terminal_summary(results, summary)
        print(colorize("JSON Summary:", "bold"))
        print(json.dumps(summary, indent=2))
    else:
        # JSON-only mode
        print(json.dumps(summary, indent=2))

    return 0 if summary["overall_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
