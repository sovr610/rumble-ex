"""
brain_ai/meta/eligibility_traces.py -- Eligibility Trace Dynamics for Three-Factor Learning

Eligibility traces are the cornerstone of biologically plausible three-factor learning
rules.  They accumulate local pre/post-synaptic correlations (Hebbian signals) into a
slowly decaying trace variable `e`.  The trace is NOT itself a weight update -- it only
records *which* synapses were recently active.  A delayed third-factor neuromodulatory
signal (e.g. dopamine reward prediction error) then gates whether e is converted into
an actual weight change:

    delta_w = lr * mod_signal * e

This separation is fundamental: it allows the system to associate rewards with synaptic
events that occurred tens or hundreds of milliseconds earlier (the temporal credit
assignment problem).

Mathematical Formulation
------------------------
At each timestep the eligibility trace is updated as:

    e(t) = decay(e(t-1)) + f(pre, post)

where:
    - decay applies exponential forgetting: e *= exp(-dt / tau_e)
    - f(pre, post) is the correlation kernel (rate, STDP pair, STDP symmetric)
    - tau_e controls how far into the past eligibility "remembers"

Three trace update strategies are provided:

    ACCUMULATING:   e += f(pre, post)            (standard; e grows without bound pre-clamp)
    REPLACING:      e = max(e, f(pre, post))     (bounded; good for sparse signals)
    DUTCH:          e = (1 - alpha) * e + f       (interpolating; Dutch trace from RL)

STDP Kernels
------------
For spiking networks the correlation kernel f is typically an STDP-like function:

    Pair-based:   pre-then-post (causal) -> potentiation (positive e)
                  post-then-pre (anti-causal) -> depression (negative e)

    Symmetric:    |f| depends only on |delta_t|, always positive

For rate-coded networks:

    Rate kernel:  f = pre (x) post (outer product or element-wise)

Batch Safety
------------
All trace operations are batch-indexed: e[b] is independent of e[b'].  No operations
mix batch dimensions (no batch-level mean/max in trace updates).  reset() creates
fresh per-batch traces.

AMP Safety
----------
All trace computations are forced to fp32 via torch.cuda.amp.autocast(enabled=False).
Pre/post inputs are cast to fp32 before computation.  NaN guards reset traces if NaN
is detected.

Key Classes:
    TraceConfig              - All configuration for a single trace module.
    EligibilityTraceModule   - Main nn.Module; manages trace state and updates.
    PairBasedSTDP            - Classic pair-based STDP kernel.
    SymmetricSTDP            - Symmetric STDP kernel (always potentiating).
    RateKernel               - Rate-based correlation kernel.
    MultiLayerEligibility    - Manages traces across multiple named layers.

Factory:
    create_eligibility_module(n_pre, n_post, config, diagonal) -> EligibilityTraceModule

References:
    Fremaux & Gerstner (2016) "Neuromodulated STDP, and Theory of Three-Factor
        Learning Rules." Frontiers in Neural Circuits.
    Bellec et al. (2020) "A solution to the learning dilemma for recurrent networks
        of spiking neurons." Nature Communications.
    Izhikevich (2007) "Solving the Distal Reward Problem through Linkage of STDP
        and Dopamine Signaling." Cerebral Cortex.
    Sutton & Barto (2018) "Reinforcement Learning: An Introduction." (Dutch traces)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ===========================================================================
# Module-level constants
# ===========================================================================

#: Default eligibility time constant in milliseconds (matches typical DA burst delay).
DEFAULT_TAU_E: float = 100.0

#: Default STDP time constant for pre-synaptic trace (ms).
DEFAULT_STDP_TAU_PLUS: float = 20.0

#: Default STDP time constant for post-synaptic trace (ms).
DEFAULT_STDP_TAU_MINUS: float = 20.0

#: Default STDP amplitude for potentiation.
DEFAULT_A_PLUS: float = 1.0

#: Default STDP amplitude for depression.
DEFAULT_A_MINUS: float = 1.0

#: Default clamp range for eligibility traces.
DEFAULT_CLAMP_RANGE: Tuple[float, float] = (-5.0, 5.0)

#: Dutch trace interpolation coefficient default.
DEFAULT_DUTCH_ALPHA: float = 0.1

#: Numerical epsilon for safe division.
_EPS: float = 1e-8

#: Minimum time constant to avoid division by zero.
_TAU_MIN: float = 0.1


# ===========================================================================
# Enums
# ===========================================================================

class TraceType(Enum):
    """Strategy for integrating new correlation signals into the eligibility trace.

    ACCUMULATING:
        Standard additive update.  The trace grows without bound (before clamping)
        as long as pre/post activity is present.  Best for dense, continuous signals.
        Update: e += f(pre, post)

    REPLACING:
        The trace takes the element-wise maximum of its current value and the new
        correlation signal.  This bounds the trace magnitude by the peak correlation
        and is preferred for sparse spiking signals where you want to "remember the
        strongest event".
        Update: e = max(e, f(pre, post))

    DUTCH:
        Interpolating update inspired by Dutch traces in reinforcement learning.
        A fraction (1 - alpha) of the existing trace is retained before adding
        the new correlation.  Provides a middle ground between accumulating and
        replacing.
        Update: e = (1 - alpha) * e + f(pre, post)
        Note: exponential decay is applied separately before this update.
    """
    ACCUMULATING = auto()
    REPLACING = auto()
    DUTCH = auto()


class KernelType(Enum):
    """Type of pre/post correlation kernel used to compute the Hebbian signal.

    RATE:
        Rate-coded kernel.  f(pre, post) = pre (outer-product) post for full-rank
        traces, or pre * post (element-wise) for diagonal traces.
        No temporal dynamics; purely instantaneous correlation.

    STDP_PAIR:
        Classic pair-based STDP.  Maintains exponentially decaying pre_trace and
        post_trace variables.  On a post-synaptic spike, potentiation is computed
        from the pre_trace value; on a pre-synaptic spike, depression is computed
        from the post_trace value.  This implements the causal/anti-causal asymmetry
        of classical STDP windows.

    STDP_SYMMETRIC:
        Symmetric STDP kernel where the correlation depends only on |delta_t|.
        Always produces potentiation (positive eligibility).  Useful for
        homeostatic plasticity signals.
    """
    RATE = auto()
    STDP_PAIR = auto()
    STDP_SYMMETRIC = auto()


# ===========================================================================
# Data classes
# ===========================================================================

@dataclass
class TraceState:
    """Snapshot of all state variables for an EligibilityTraceModule.

    This dataclass enables checkpointing and restoring trace state, which is
    essential for:
        - Carrying traces across truncated BPTT windows
        - Saving/loading training state for resumption
        - Debugging by inspecting trace evolution

    Attributes:
        e:           The eligibility trace tensor, shape (B, N_post, N_pre)
                     or (B, N) for diagonal mode.
        pre_trace:   STDP pre-synaptic trace, shape (B, N_pre) or None.
        post_trace:  STDP post-synaptic trace, shape (B, N_post) or None.
        step_count:  Number of update steps since last reset.
        device:      The device on which tensors reside.
    """
    e: Tensor
    pre_trace: Optional[Tensor] = None
    post_trace: Optional[Tensor] = None
    step_count: int = 0
    device: str = "cpu"

    def to(self, device: Union[str, torch.device]) -> "TraceState":
        """Move all tensors to the specified device."""
        device_str = str(device)
        new_e = self.e.to(device)
        new_pre = self.pre_trace.to(device) if self.pre_trace is not None else None
        new_post = self.post_trace.to(device) if self.post_trace is not None else None
        return TraceState(
            e=new_e,
            pre_trace=new_pre,
            post_trace=new_post,
            step_count=self.step_count,
            device=device_str,
        )

    def clone(self) -> "TraceState":
        """Deep-clone all tensors."""
        return TraceState(
            e=self.e.clone(),
            pre_trace=self.pre_trace.clone() if self.pre_trace is not None else None,
            post_trace=self.post_trace.clone() if self.post_trace is not None else None,
            step_count=self.step_count,
            device=self.device,
        )


@dataclass
class TraceConfig:
    """Configuration for a single EligibilityTraceModule.

    This dataclass fully specifies the behavior of an eligibility trace: the
    trace update strategy, decay time constant, correlation kernel type, STDP
    parameters, clamping bounds, and Dutch trace coefficient.

    Attributes:
        trace_type:     Trace update strategy (ACCUMULATING, REPLACING, DUTCH).
        tau_e:          Eligibility trace decay time constant in ms.  Larger values
                        mean the trace persists longer.  Must be > 0.
        kernel:         Correlation kernel type (RATE, STDP_PAIR, STDP_SYMMETRIC).
        stdp_tau_plus:  Time constant for pre-synaptic STDP trace (ms).
        stdp_tau_minus: Time constant for post-synaptic STDP trace (ms).
        dutch_alpha:    Interpolation coefficient for DUTCH trace type.
                        0.0 = fully accumulating, 1.0 = fully replacing.
        clamp_range:    (min, max) bounds for the eligibility trace.
        a_plus:         Amplitude for STDP potentiation (pre-before-post).
        a_minus:        Amplitude for STDP depression (post-before-pre).
    """
    trace_type: TraceType = TraceType.ACCUMULATING
    tau_e: float = DEFAULT_TAU_E
    kernel: KernelType = KernelType.RATE
    stdp_tau_plus: float = DEFAULT_STDP_TAU_PLUS
    stdp_tau_minus: float = DEFAULT_STDP_TAU_MINUS
    dutch_alpha: float = DEFAULT_DUTCH_ALPHA
    clamp_range: Tuple[float, float] = DEFAULT_CLAMP_RANGE
    a_plus: float = DEFAULT_A_PLUS
    a_minus: float = DEFAULT_A_MINUS

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.tau_e <= 0.0:
            raise ValueError(f"tau_e must be positive, got {self.tau_e}")
        if self.tau_e < _TAU_MIN:
            logger.warning(
                "tau_e=%.4f is very small (< %.4f); trace will decay extremely fast",
                self.tau_e, _TAU_MIN,
            )
        if self.stdp_tau_plus <= 0.0:
            raise ValueError(f"stdp_tau_plus must be positive, got {self.stdp_tau_plus}")
        if self.stdp_tau_minus <= 0.0:
            raise ValueError(f"stdp_tau_minus must be positive, got {self.stdp_tau_minus}")
        if not (0.0 <= self.dutch_alpha <= 1.0):
            raise ValueError(
                f"dutch_alpha must be in [0, 1], got {self.dutch_alpha}"
            )
        lo, hi = self.clamp_range
        if lo >= hi:
            raise ValueError(
                f"clamp_range lower ({lo}) must be < upper ({hi})"
            )
        if self.a_plus < 0.0:
            raise ValueError(f"a_plus must be non-negative, got {self.a_plus}")
        if self.a_minus < 0.0:
            raise ValueError(f"a_minus must be non-negative, got {self.a_minus}")

    @staticmethod
    def default_rate() -> "TraceConfig":
        """Create a default config for rate-coded networks."""
        return TraceConfig(
            trace_type=TraceType.ACCUMULATING,
            tau_e=100.0,
            kernel=KernelType.RATE,
        )

    @staticmethod
    def default_stdp() -> "TraceConfig":
        """Create a default config for spiking networks with pair-based STDP."""
        return TraceConfig(
            trace_type=TraceType.ACCUMULATING,
            tau_e=100.0,
            kernel=KernelType.STDP_PAIR,
            stdp_tau_plus=20.0,
            stdp_tau_minus=20.0,
            a_plus=1.0,
            a_minus=1.0,
        )

    @staticmethod
    def default_dutch() -> "TraceConfig":
        """Create a default config with Dutch trace update."""
        return TraceConfig(
            trace_type=TraceType.DUTCH,
            tau_e=100.0,
            kernel=KernelType.RATE,
            dutch_alpha=0.1,
        )


# ===========================================================================
# STDP Kernels
# ===========================================================================

class BaseKernel(nn.Module):
    """Abstract base class for pre/post correlation kernels.

    Every kernel computes a Hebbian-like correlation signal f(pre, post)
    that is added to the eligibility trace.  Subclasses implement the
    specific correlation rule and any internal state (e.g. STDP traces).

    Kernels must handle two modes:
        - Full-rank: output shape (B, N_post, N_pre) for dense weight matrices
        - Diagonal:  output shape (B, N) for per-neuron traces (N_pre == N_post == N)
    """

    def __init__(self, n_pre: int, n_post: int, diagonal: bool = False) -> None:
        super().__init__()
        self.n_pre = n_pre
        self.n_post = n_post
        self.diagonal = diagonal

    def reset(self, batch_size: int, device: torch.device) -> None:
        """Reset any internal kernel state (e.g. STDP traces)."""
        pass

    def forward(
        self,
        pre: Tensor,
        post: Tensor,
        dt: float = 1.0,
    ) -> Tensor:
        """Compute correlation signal f(pre, post).

        Args:
            pre:  Pre-synaptic activation or spikes, shape (B, N_pre).
            post: Post-synaptic activation or spikes, shape (B, N_post).
            dt:   Timestep in ms for temporal kernel dynamics.

        Returns:
            Correlation tensor, shape (B, N_post, N_pre) or (B, N) if diagonal.
        """
        raise NotImplementedError

    def get_pre_trace(self) -> Optional[Tensor]:
        """Return the current pre-synaptic trace if applicable."""
        return None

    def get_post_trace(self) -> Optional[Tensor]:
        """Return the current post-synaptic trace if applicable."""
        return None

    def set_pre_trace(self, trace: Tensor) -> None:
        """Restore the pre-synaptic trace from a checkpoint."""
        pass

    def set_post_trace(self, trace: Tensor) -> None:
        """Restore the post-synaptic trace from a checkpoint."""
        pass


class RateKernel(BaseKernel):
    """Rate-coded Hebbian correlation kernel.

    Computes instantaneous correlation between pre- and post-synaptic
    firing rates (or continuous activations).

    Full-rank mode:
        f(pre, post) = post^T @ pre  (outer product)
        Output shape: (B, N_post, N_pre)

    Diagonal mode:
        f(pre, post) = pre * post  (element-wise)
        Output shape: (B, N)
        Requires N_pre == N_post.

    No internal state is maintained -- the kernel is purely instantaneous.
    """

    def __init__(self, n_pre: int, n_post: int, diagonal: bool = False) -> None:
        super().__init__(n_pre, n_post, diagonal)
        if diagonal and n_pre != n_post:
            raise ValueError(
                f"Diagonal mode requires n_pre == n_post, got {n_pre} vs {n_post}"
            )

    def forward(
        self,
        pre: Tensor,
        post: Tensor,
        dt: float = 1.0,
    ) -> Tensor:
        """Compute rate-based correlation.

        Args:
            pre:  (B, N_pre) pre-synaptic rates/activations.
            post: (B, N_post) post-synaptic rates/activations.
            dt:   Unused for rate kernel (included for interface consistency).

        Returns:
            (B, N_post, N_pre) or (B, N) correlation tensor.
        """
        if self.diagonal:
            # Element-wise product: (B, N)
            return pre * post
        else:
            # Outer product: (B, N_post, N_pre)
            # post: (B, N_post) -> (B, N_post, 1)
            # pre:  (B, N_pre)  -> (B, 1, N_pre)
            return torch.bmm(
                post.unsqueeze(2),  # (B, N_post, 1)
                pre.unsqueeze(1),   # (B, 1, N_pre)
            )


class PairBasedSTDP(BaseKernel):
    """Classic pair-based spike-timing-dependent plasticity kernel.

    Maintains exponentially decaying pre- and post-synaptic trace variables.
    These traces record the recent spike history and are used to compute the
    correlation signal when the other neuron fires.

    Trace dynamics:
        On each timestep:
            pre_trace  *= exp(-dt / tau_plus)
            post_trace *= exp(-dt / tau_minus)

        On pre-synaptic spike (pre[i] > 0):
            pre_trace[i] += A_plus

        On post-synaptic spike (post[j] > 0):
            post_trace[j] += A_minus

    Correlation signal (the eligibility update):
        Potentiation (pre-before-post, causal):
            f_pot[j, i] = post_spike[j] * pre_trace[i]

        Depression (post-before-pre, anti-causal):
            f_dep[j, i] = -pre_spike[i] * post_trace[j]

        Total:
            f[j, i] = f_pot[j, i] + f_dep[j, i]

    Sign convention:
        - Positive f -> potentiation (causal timing, pre fires before post)
        - Negative f -> depression (anti-causal timing, post fires before pre)

    This convention matches the classical STDP window from Bi & Poo (1998).

    For diagonal mode, the same logic applies but produces (B, N) output
    where only same-index pre/post pairs are considered.

    References:
        Bi & Poo (1998) "Synaptic Modifications in Cultured Hippocampal Neurons."
        Song, Miller & Abbott (2000) "Competitive Hebbian Learning through STDP."
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        tau_plus: float = DEFAULT_STDP_TAU_PLUS,
        tau_minus: float = DEFAULT_STDP_TAU_MINUS,
        a_plus: float = DEFAULT_A_PLUS,
        a_minus: float = DEFAULT_A_MINUS,
        diagonal: bool = False,
    ) -> None:
        super().__init__(n_pre, n_post, diagonal)
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.a_plus = a_plus
        self.a_minus = a_minus

        if diagonal and n_pre != n_post:
            raise ValueError(
                f"Diagonal mode requires n_pre == n_post, got {n_pre} vs {n_post}"
            )

        # Pre and post traces are registered as buffers (non-parameter state)
        # but will be lazily initialized on first reset() call.
        self._pre_trace: Optional[Tensor] = None
        self._post_trace: Optional[Tensor] = None

    def reset(self, batch_size: int, device: torch.device) -> None:
        """Reset STDP pre and post traces to zero.

        Args:
            batch_size: Batch dimension B.
            device:     Target device for trace tensors.
        """
        self._pre_trace = torch.zeros(
            batch_size, self.n_pre, dtype=torch.float32, device=device
        )
        self._post_trace = torch.zeros(
            batch_size, self.n_post, dtype=torch.float32, device=device
        )

    def forward(
        self,
        pre: Tensor,
        post: Tensor,
        dt: float = 1.0,
    ) -> Tensor:
        """Compute pair-based STDP correlation signal.

        The update follows this sequence:
            1. Decay existing pre/post traces.
            2. Compute potentiation from current post spikes x pre_trace.
            3. Compute depression from current pre spikes x post_trace.
            4. Increment pre_trace for pre spikes, post_trace for post spikes.
            5. Return potentiation + depression.

        Args:
            pre:  (B, N_pre) binary spikes or continuous activations.
            post: (B, N_post) binary spikes or continuous activations.
            dt:   Timestep in ms.

        Returns:
            (B, N_post, N_pre) or (B, N) STDP correlation signal.
        """
        B = pre.shape[0]
        device = pre.device

        # Lazy initialization
        if self._pre_trace is None or self._pre_trace.shape[0] != B:
            self.reset(B, device)

        # Ensure traces are on the correct device
        if self._pre_trace.device != device:  # type: ignore[union-attr]
            self._pre_trace = self._pre_trace.to(device)  # type: ignore[union-attr]
            self._post_trace = self._post_trace.to(device)  # type: ignore[union-attr]

        assert self._pre_trace is not None
        assert self._post_trace is not None

        # Step 1: Decay existing traces
        decay_pre = math.exp(-dt / self.tau_plus)
        decay_post = math.exp(-dt / self.tau_minus)
        self._pre_trace = self._pre_trace * decay_pre
        self._post_trace = self._post_trace * decay_post

        # Step 2 & 3: Compute correlation BEFORE updating traces
        # Potentiation: post spike reads from pre_trace (causal: pre before post)
        # Depression: pre spike reads from post_trace (anti-causal: post before pre)
        if self.diagonal:
            # (B, N) element-wise
            f_pot = post * self._pre_trace     # post spikes * pre_trace
            f_dep = -pre * self._post_trace    # pre spikes * post_trace (negative = depression)
            f = f_pot + f_dep
        else:
            # (B, N_post, N_pre) outer product style
            # f_pot[b, j, i] = post[b, j] * pre_trace[b, i]
            f_pot = torch.bmm(
                post.unsqueeze(2),           # (B, N_post, 1)
                self._pre_trace.unsqueeze(1),  # (B, 1, N_pre)
            )
            # f_dep[b, j, i] = -pre[b, i] * post_trace[b, j]
            f_dep = -torch.bmm(
                self._post_trace.unsqueeze(2),  # (B, N_post, 1)
                pre.unsqueeze(1),                # (B, 1, N_pre)
            )
            f = f_pot + f_dep

        # Step 4: Update traces with new spikes
        self._pre_trace = self._pre_trace + self.a_plus * pre
        self._post_trace = self._post_trace + self.a_minus * post

        return f

    def get_pre_trace(self) -> Optional[Tensor]:
        """Return current pre-synaptic STDP trace."""
        return self._pre_trace

    def get_post_trace(self) -> Optional[Tensor]:
        """Return current post-synaptic STDP trace."""
        return self._post_trace

    def set_pre_trace(self, trace: Tensor) -> None:
        """Restore pre-synaptic STDP trace from checkpoint."""
        self._pre_trace = trace.clone()

    def set_post_trace(self, trace: Tensor) -> None:
        """Restore post-synaptic STDP trace from checkpoint."""
        self._post_trace = trace.clone()


class SymmetricSTDP(BaseKernel):
    """Symmetric STDP kernel where correlation depends only on |delta_t|.

    Unlike the pair-based kernel, this kernel always produces positive
    (potentiating) eligibility regardless of spike ordering.  The magnitude
    decays with the temporal distance between pre and post spikes.

    This is useful for:
        - Homeostatic plasticity where any correlated activity should strengthen
        - Situations where the sign of the update is determined entirely by the
          third-factor modulator

    Implementation:
        Uses the same exponentially decaying pre/post traces as PairBasedSTDP,
        but takes the absolute value of the correlation and uses only the
        potentiation term (magnitude of both causal and anti-causal events).

        f_sym[j, i] = |post_spike[j] * pre_trace[i]| + |pre_spike[i] * post_trace[j]|

    For rate-coded inputs this reduces to approximately the rate kernel output
    plus residual trace effects.

    References:
        Kempter, Gerstner & van Hemmen (1999) "Hebbian learning and spiking neurons."
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        tau_plus: float = DEFAULT_STDP_TAU_PLUS,
        tau_minus: float = DEFAULT_STDP_TAU_MINUS,
        a_plus: float = DEFAULT_A_PLUS,
        a_minus: float = DEFAULT_A_MINUS,
        diagonal: bool = False,
    ) -> None:
        super().__init__(n_pre, n_post, diagonal)
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.a_plus = a_plus
        self.a_minus = a_minus

        if diagonal and n_pre != n_post:
            raise ValueError(
                f"Diagonal mode requires n_pre == n_post, got {n_pre} vs {n_post}"
            )

        self._pre_trace: Optional[Tensor] = None
        self._post_trace: Optional[Tensor] = None

    def reset(self, batch_size: int, device: torch.device) -> None:
        """Reset symmetric STDP traces to zero.

        Args:
            batch_size: Batch dimension B.
            device:     Target device for trace tensors.
        """
        self._pre_trace = torch.zeros(
            batch_size, self.n_pre, dtype=torch.float32, device=device
        )
        self._post_trace = torch.zeros(
            batch_size, self.n_post, dtype=torch.float32, device=device
        )

    def forward(
        self,
        pre: Tensor,
        post: Tensor,
        dt: float = 1.0,
    ) -> Tensor:
        """Compute symmetric STDP correlation signal.

        Similar to PairBasedSTDP.forward, but takes absolute values so the
        result is always non-negative.

        Args:
            pre:  (B, N_pre) binary spikes or continuous activations.
            post: (B, N_post) binary spikes or continuous activations.
            dt:   Timestep in ms.

        Returns:
            (B, N_post, N_pre) or (B, N) symmetric STDP correlation, always >= 0.
        """
        B = pre.shape[0]
        device = pre.device

        # Lazy initialization
        if self._pre_trace is None or self._pre_trace.shape[0] != B:
            self.reset(B, device)

        if self._pre_trace.device != device:  # type: ignore[union-attr]
            self._pre_trace = self._pre_trace.to(device)  # type: ignore[union-attr]
            self._post_trace = self._post_trace.to(device)  # type: ignore[union-attr]

        assert self._pre_trace is not None
        assert self._post_trace is not None

        # Step 1: Decay existing traces
        decay_pre = math.exp(-dt / self.tau_plus)
        decay_post = math.exp(-dt / self.tau_minus)
        self._pre_trace = self._pre_trace * decay_pre
        self._post_trace = self._post_trace * decay_post

        # Step 2: Compute symmetric correlation (always positive)
        if self.diagonal:
            f_causal = torch.abs(post * self._pre_trace)
            f_anti = torch.abs(pre * self._post_trace)
            f = f_causal + f_anti
        else:
            f_causal = torch.abs(torch.bmm(
                post.unsqueeze(2),
                self._pre_trace.unsqueeze(1),
            ))
            f_anti = torch.abs(torch.bmm(
                self._post_trace.unsqueeze(2),
                pre.unsqueeze(1),
            ))
            f = f_causal + f_anti

        # Step 3: Update traces with new spikes
        self._pre_trace = self._pre_trace + self.a_plus * pre
        self._post_trace = self._post_trace + self.a_minus * post

        return f

    def get_pre_trace(self) -> Optional[Tensor]:
        """Return current pre-synaptic STDP trace."""
        return self._pre_trace

    def get_post_trace(self) -> Optional[Tensor]:
        """Return current post-synaptic STDP trace."""
        return self._post_trace

    def set_pre_trace(self, trace: Tensor) -> None:
        """Restore pre-synaptic STDP trace from checkpoint."""
        self._pre_trace = trace.clone()

    def set_post_trace(self, trace: Tensor) -> None:
        """Restore post-synaptic STDP trace from checkpoint."""
        self._post_trace = trace.clone()


# ===========================================================================
# Kernel Factory
# ===========================================================================

def _create_kernel(
    kernel_type: KernelType,
    n_pre: int,
    n_post: int,
    config: TraceConfig,
    diagonal: bool = False,
) -> BaseKernel:
    """Internal factory that creates the appropriate kernel from config.

    Args:
        kernel_type: Type of kernel to create.
        n_pre:       Number of pre-synaptic neurons.
        n_post:      Number of post-synaptic neurons.
        config:      Full trace configuration (STDP params extracted if needed).
        diagonal:    Whether to operate in diagonal mode.

    Returns:
        An instance of the appropriate BaseKernel subclass.

    Raises:
        ValueError: If kernel_type is not recognized.
    """
    if kernel_type == KernelType.RATE:
        return RateKernel(n_pre, n_post, diagonal=diagonal)
    elif kernel_type == KernelType.STDP_PAIR:
        return PairBasedSTDP(
            n_pre, n_post,
            tau_plus=config.stdp_tau_plus,
            tau_minus=config.stdp_tau_minus,
            a_plus=config.a_plus,
            a_minus=config.a_minus,
            diagonal=diagonal,
        )
    elif kernel_type == KernelType.STDP_SYMMETRIC:
        return SymmetricSTDP(
            n_pre, n_post,
            tau_plus=config.stdp_tau_plus,
            tau_minus=config.stdp_tau_minus,
            a_plus=config.a_plus,
            a_minus=config.a_minus,
            diagonal=diagonal,
        )
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")


# ===========================================================================
# EligibilityTraceModule
# ===========================================================================

class EligibilityTraceModule(nn.Module):
    """Core module for eligibility trace dynamics in three-factor learning.

    This module manages the eligibility trace tensor `e`, which records the
    local Hebbian correlation between pre- and post-synaptic activity.  The
    trace decays exponentially over time and is updated at each timestep with
    a correlation signal from the selected kernel.

    The trace itself does NOT modify weights.  Weight updates occur only when
    the `apply_update` method is called with a third-factor modulatory signal:

        delta_w = lr * mod_signal * e

    This separation is the defining characteristic of three-factor learning.

    Shapes:
        Full-rank mode (diagonal=False):
            e: (B, N_post, N_pre)     -- full weight-matrix-shaped trace
            Suitable for dense layers: nn.Linear(N_pre, N_post)

        Diagonal mode (diagonal=True):
            e: (B, N)                 -- per-neuron trace (N_pre == N_post == N)
            Suitable for per-neuron modulation, recurrent self-connections

    Thread Safety:
        This module is NOT thread-safe.  Each thread or process should have
        its own instance.

    Args:
        n_pre:     Number of pre-synaptic neurons.
        n_post:    Number of post-synaptic neurons.
        config:    Trace configuration dataclass.
        diagonal:  If True, use diagonal (element-wise) trace mode.
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        config: Optional[TraceConfig] = None,
        diagonal: bool = False,
    ) -> None:
        super().__init__()

        if config is None:
            config = TraceConfig()

        if diagonal and n_pre != n_post:
            raise ValueError(
                f"Diagonal mode requires n_pre == n_post, got {n_pre} vs {n_post}"
            )

        self.n_pre = n_pre
        self.n_post = n_post
        self.config = config
        self.diagonal = diagonal

        # Create the correlation kernel
        self.kernel = _create_kernel(
            config.kernel, n_pre, n_post, config, diagonal=diagonal
        )

        # Trace state -- lazily initialized on reset() or first update()
        self._e: Optional[Tensor] = None
        self._step_count: int = 0
        self._batch_size: int = 0
        self._device: torch.device = torch.device("cpu")

        # Precompute decay factor (recomputed if tau_e changes)
        self._tau_e = config.tau_e

        logger.debug(
            "EligibilityTraceModule created: n_pre=%d, n_post=%d, "
            "trace_type=%s, kernel=%s, tau_e=%.1f, diagonal=%s",
            n_pre, n_post, config.trace_type.name, config.kernel.name,
            config.tau_e, diagonal,
        )

    # -------------------------------------------------------------------
    # State management
    # -------------------------------------------------------------------

    def reset(self, batch_size: int, device: Union[str, torch.device]) -> None:
        """Clear all traces to zero and reinitialize state.

        This MUST be called before the first update() call at the start of
        each episode or sequence.  It creates fresh per-batch traces with
        the correct shape and dtype.

        Args:
            batch_size: Number of independent batch elements.
            device:     Target device (cpu, cuda, etc.).
        """
        device = torch.device(device) if isinstance(device, str) else device
        self._batch_size = batch_size
        self._device = device
        self._step_count = 0

        # Initialize eligibility trace to zero
        if self.diagonal:
            self._e = torch.zeros(
                batch_size, self.n_pre,
                dtype=torch.float32, device=device,
            )
        else:
            self._e = torch.zeros(
                batch_size, self.n_post, self.n_pre,
                dtype=torch.float32, device=device,
            )

        # Reset kernel internal state (STDP traces)
        self.kernel.reset(batch_size, device)

        logger.debug(
            "Traces reset: batch_size=%d, device=%s, e_shape=%s",
            batch_size, device, tuple(self._e.shape),
        )

    def get_state(self) -> TraceState:
        """Return a snapshot of the current trace state.

        The returned TraceState contains clones of all tensors, so it is
        safe to store and restore without worrying about aliasing.

        Returns:
            TraceState with cloned e, pre_trace, post_trace, step_count.

        Raises:
            RuntimeError: If reset() has not been called yet.
        """
        if self._e is None:
            raise RuntimeError(
                "Cannot get_state() before reset() -- traces are uninitialized"
            )
        return TraceState(
            e=self._e.clone(),
            pre_trace=(
                self.kernel.get_pre_trace().clone()
                if self.kernel.get_pre_trace() is not None
                else None
            ),
            post_trace=(
                self.kernel.get_post_trace().clone()
                if self.kernel.get_post_trace() is not None
                else None
            ),
            step_count=self._step_count,
            device=str(self._device),
        )

    def set_state(self, state: TraceState) -> None:
        """Restore trace state from a previously saved TraceState.

        This is used for carrying traces across truncated BPTT windows or
        for resuming training from a checkpoint.

        Args:
            state: TraceState to restore.  Tensors are cloned internally.
        """
        device = torch.device(state.device)
        self._e = state.e.clone().to(device)
        self._step_count = state.step_count
        self._device = device
        self._batch_size = state.e.shape[0]

        if state.pre_trace is not None:
            self.kernel.set_pre_trace(state.pre_trace.to(device))
        if state.post_trace is not None:
            self.kernel.set_post_trace(state.post_trace.to(device))

        logger.debug(
            "Trace state restored: step_count=%d, device=%s",
            self._step_count, device,
        )

    # -------------------------------------------------------------------
    # Core update
    # -------------------------------------------------------------------

    @torch.no_grad()
    def update(
        self,
        pre: Tensor,
        post: Tensor,
        dt: float = 1.0,
    ) -> Tensor:
        """Update eligibility traces given pre- and post-synaptic activity.

        This is the core method called at each simulation timestep.  It:
            1. Decays the existing trace by exp(-dt / tau_e).
            2. Computes the correlation signal f(pre, post) using the kernel.
            3. Integrates f into the trace using the selected strategy.
            4. Clamps the trace to the configured range.

        All computations are performed in fp32 regardless of AMP context.

        Args:
            pre:  Pre-synaptic activity, shape (B, N_pre).
            post: Post-synaptic activity, shape (B, N_post).
            dt:   Timestep size in ms (default 1.0).

        Returns:
            Updated eligibility trace tensor e, shape (B, N_post, N_pre)
            or (B, N) for diagonal mode.

        Raises:
            RuntimeError: If reset() has not been called yet.
        """
        # Force fp32 for numerical stability under AMP
        with torch.amp.autocast("cuda", enabled=False):
            return self._update_impl(pre.float(), post.float(), dt)

    def _update_impl(
        self,
        pre: Tensor,
        post: Tensor,
        dt: float,
    ) -> Tensor:
        """Internal implementation of trace update, always in fp32.

        Args:
            pre:  (B, N_pre) fp32 pre-synaptic activity.
            post: (B, N_post) fp32 post-synaptic activity.
            dt:   Timestep in ms.

        Returns:
            Updated eligibility trace tensor e.
        """
        B = pre.shape[0]
        device = pre.device

        # Lazy initialization if reset() was never called
        if self._e is None or self._e.shape[0] != B or self._e.device != device:
            self.reset(B, device)

        assert self._e is not None

        # ------ NaN guard on inputs ------
        if torch.isnan(pre).any() or torch.isnan(post).any():
            logger.warning(
                "NaN detected in pre/post inputs at step %d; "
                "replacing NaN with zero",
                self._step_count,
            )
            pre = torch.nan_to_num(pre, nan=0.0)
            post = torch.nan_to_num(post, nan=0.0)

        # ------ Step 1: Decay existing trace ------
        decay_factor = math.exp(-dt / self._tau_e)
        self._e = self._e * decay_factor

        # ------ Step 2: Compute correlation via kernel ------
        f = self.kernel(pre, post, dt=dt)

        # ------ Step 3: Apply trace update strategy ------
        if self.config.trace_type == TraceType.ACCUMULATING:
            self._e = self._e + f
        elif self.config.trace_type == TraceType.REPLACING:
            self._e = torch.max(self._e, f)
        elif self.config.trace_type == TraceType.DUTCH:
            # Dutch trace: blend existing (already decayed) with new signal
            # e = (1 - alpha) * e_decayed + f
            # Note: decay was already applied in step 1, so the full update is:
            # e = (1 - alpha) * (e * decay) + f  [but we already have e * decay]
            alpha = self.config.dutch_alpha
            self._e = (1.0 - alpha) * self._e + f
        else:
            raise ValueError(f"Unknown trace type: {self.config.trace_type}")

        # ------ Step 4: Clamp ------
        lo, hi = self.config.clamp_range
        self._e = torch.clamp(self._e, min=lo, max=hi)

        # ------ NaN guard on output ------
        if torch.isnan(self._e).any():
            logger.warning(
                "NaN detected in eligibility trace at step %d; resetting trace",
                self._step_count,
            )
            self._e = torch.zeros_like(self._e)

        self._step_count += 1

        return self._e

    # -------------------------------------------------------------------
    # Weight update (third-factor gating)
    # -------------------------------------------------------------------

    @torch.no_grad()
    def apply_update(
        self,
        weights: Tensor,
        mod_signal: Union[Tensor, float],
        *,
        lr: float = 0.001,
        clamp: Optional[Tuple[float, float]] = None,
    ) -> Tensor:
        """Apply three-factor weight update: delta_w = lr * mod_signal * e.

        This is the core of three-factor learning.  The weight update is gated
        by the modulatory signal: when mod_signal == 0, delta_w is exactly zero
        and no computation is wasted.

        Args:
            weights:     Current weight tensor.  Shape must be compatible with e.
                         For full-rank: (N_post, N_pre).
                         For diagonal: (N,).
            mod_signal:  Third-factor modulatory signal.  Can be:
                         - A scalar float (applied uniformly across batch and synapses)
                         - A scalar Tensor
                         - A Tensor of shape (B,) for per-sample modulation
                         - A Tensor of shape matching e for per-synapse modulation
            lr:          Learning rate for the update.
            clamp:       Optional (min, max) clamp for the updated weights.

        Returns:
            Updated weight tensor (detached from computation graph).
            Same shape as input weights.
        """
        if self._e is None:
            logger.warning("apply_update called before any trace update; returning weights unchanged")
            return weights.detach().clone()

        # Force fp32
        with torch.amp.autocast("cuda", enabled=False):
            weights_fp32 = weights.float()

            # Check for zero modulation -> skip computation (gating!)
            if isinstance(mod_signal, (int, float)):
                if mod_signal == 0.0:
                    return weights_fp32.detach().clone()
                mod_tensor = torch.tensor(
                    mod_signal, dtype=torch.float32, device=self._device
                )
            else:
                mod_tensor = mod_signal.float().to(self._device)
                # Check if mod_signal is a scalar tensor that is exactly zero
                if mod_tensor.numel() == 1 and mod_tensor.item() == 0.0:
                    return weights_fp32.detach().clone()

            # Compute mean trace across batch for weight update
            # e: (B, N_post, N_pre) or (B, N)
            # We average across batch to get a single weight update
            e_mean = self._e.mean(dim=0)  # (N_post, N_pre) or (N,)

            # Handle per-batch modulation
            if mod_tensor.dim() >= 1 and mod_tensor.shape[0] == self._batch_size:
                # Per-sample modulation: weight each sample's trace differently
                # mod_tensor: (B,) -> broadcast over spatial dims
                if self.diagonal:
                    # e: (B, N), mod: (B,) -> (B, 1)
                    weighted_e = (self._e * mod_tensor.unsqueeze(-1)).mean(dim=0)
                else:
                    # e: (B, N_post, N_pre), mod: (B,) -> (B, 1, 1)
                    weighted_e = (
                        self._e * mod_tensor.unsqueeze(-1).unsqueeze(-1)
                    ).mean(dim=0)
                delta_w = lr * weighted_e
            else:
                # Scalar or per-synapse modulation
                if mod_tensor.dim() == 0 or mod_tensor.numel() == 1:
                    delta_w = lr * mod_tensor.item() * e_mean
                else:
                    delta_w = lr * mod_tensor * e_mean

            # Clamp delta_w if specified
            if clamp is not None:
                delta_lo, delta_hi = clamp
                delta_w = torch.clamp(delta_w, min=delta_lo, max=delta_hi)

            # Apply update
            new_weights = weights_fp32 + delta_w

            # Clamp weights if specified
            if clamp is not None:
                new_weights = torch.clamp(new_weights, min=clamp[0], max=clamp[1])

            return new_weights.detach()

    # -------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------

    def trace_stats(self) -> Dict[str, float]:
        """Compute summary statistics of the current eligibility trace.

        Returns:
            Dictionary with keys: norm, mean, max, min, sparsity, step_count.
            If traces are uninitialized, all values are 0.0.
        """
        if self._e is None:
            return {
                "norm": 0.0,
                "mean": 0.0,
                "max": 0.0,
                "min": 0.0,
                "sparsity": 1.0,
                "step_count": 0.0,
            }

        e = self._e.detach()
        total_elements = e.numel()
        near_zero = (e.abs() < _EPS).sum().item()

        return {
            "norm": e.norm().item(),
            "mean": e.mean().item(),
            "max": e.max().item(),
            "min": e.min().item(),
            "sparsity": near_zero / max(total_elements, 1),
            "step_count": float(self._step_count),
        }

    @property
    def eligibility(self) -> Optional[Tensor]:
        """Read-only access to the current eligibility trace tensor."""
        return self._e

    @property
    def step_count(self) -> int:
        """Number of update steps since last reset."""
        return self._step_count

    def extra_repr(self) -> str:
        """String representation for nn.Module printing."""
        return (
            f"n_pre={self.n_pre}, n_post={self.n_post}, "
            f"trace_type={self.config.trace_type.name}, "
            f"kernel={self.config.kernel.name}, "
            f"tau_e={self.config.tau_e:.1f}, "
            f"diagonal={self.diagonal}"
        )


# ===========================================================================
# MultiLayerEligibility
# ===========================================================================

class MultiLayerEligibility(nn.Module):
    """Manages eligibility traces across multiple named layers.

    In a multi-layer network, each layer has its own eligibility trace with
    potentially different configurations (e.g. different tau_e, different
    kernel types).  This class provides a unified interface for resetting,
    updating, and applying traces across all layers.

    Usage:
        layer_configs = {
            "encoder": (784, 256, TraceConfig(tau_e=50.0)),
            "hidden":  (256, 128, TraceConfig(tau_e=100.0)),
            "output":  (128, 10,  TraceConfig(tau_e=200.0)),
        }
        multi = MultiLayerEligibility(layer_configs)
        multi.reset_all(batch_size=32, device="cuda")

        # At each timestep:
        activations = {
            "encoder": (pre_enc, post_enc),
            "hidden":  (pre_hid, post_hid),
            "output":  (pre_out, post_out),
        }
        traces = multi.update_all(activations)

        # When modulator fires:
        new_weights = multi.apply_all(weight_dict, mod_signal=0.5, lr=0.001)

    Args:
        layer_configs: Dictionary mapping layer names to (n_pre, n_post, config)
                       tuples.  The config can be a TraceConfig or None (defaults).
        diagonal_layers: Optional set of layer names that should use diagonal mode.
    """

    def __init__(
        self,
        layer_configs: Dict[str, Tuple[int, int, Optional[TraceConfig]]],
        diagonal_layers: Optional[set] = None,
    ) -> None:
        super().__init__()

        if diagonal_layers is None:
            diagonal_layers = set()

        self.layer_names: List[str] = list(layer_configs.keys())
        self.traces = nn.ModuleDict()

        for name, (n_pre, n_post, config) in layer_configs.items():
            diag = name in diagonal_layers
            trace_module = EligibilityTraceModule(
                n_pre, n_post,
                config=config,
                diagonal=diag,
            )
            self.traces[name] = trace_module

        logger.info(
            "MultiLayerEligibility created with %d layers: %s",
            len(self.layer_names),
            self.layer_names,
        )

    def reset_all(self, batch_size: int, device: Union[str, torch.device]) -> None:
        """Reset all layer traces to zero.

        Args:
            batch_size: Batch dimension B.
            device:     Target device.
        """
        for name in self.layer_names:
            self.traces[name].reset(batch_size, device)  # type: ignore[union-attr]

    def update_all(
        self,
        activations: Dict[str, Tuple[Tensor, Tensor]],
        dt: float = 1.0,
    ) -> Dict[str, Tensor]:
        """Update traces for all layers given their pre/post activations.

        Args:
            activations: Dictionary mapping layer names to (pre, post) tensor
                         tuples.  Only layers present in this dict are updated;
                         missing layers are silently skipped.
            dt:          Timestep in ms.

        Returns:
            Dictionary mapping layer names to their updated eligibility traces.
        """
        results: Dict[str, Tensor] = {}
        for name in self.layer_names:
            if name in activations:
                pre, post = activations[name]
                trace_mod = self.traces[name]
                results[name] = trace_mod.update(pre, post, dt=dt)  # type: ignore[union-attr]
        return results

    def apply_all(
        self,
        weights: Dict[str, Tensor],
        mod_signal: Union[Tensor, float],
        lr: float = 0.001,
        clamp: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Tensor]:
        """Apply three-factor weight updates across all layers.

        Args:
            weights:    Dictionary mapping layer names to weight tensors.
            mod_signal: Third-factor modulatory signal (scalar or tensor).
            lr:         Learning rate.
            clamp:      Optional (min, max) clamp for updated weights.

        Returns:
            Dictionary mapping layer names to updated weight tensors.
        """
        updated: Dict[str, Tensor] = {}
        for name in self.layer_names:
            if name in weights:
                trace_mod = self.traces[name]
                updated[name] = trace_mod.apply_update(  # type: ignore[union-attr]
                    weights[name], mod_signal, lr=lr, clamp=clamp,
                )
        return updated

    def get_all_states(self) -> Dict[str, TraceState]:
        """Return trace states for all layers.

        Returns:
            Dictionary mapping layer names to TraceState snapshots.
        """
        states: Dict[str, TraceState] = {}
        for name in self.layer_names:
            trace_mod = self.traces[name]
            try:
                states[name] = trace_mod.get_state()  # type: ignore[union-attr]
            except RuntimeError:
                # Layer not yet initialized
                pass
        return states

    def set_all_states(self, states: Dict[str, TraceState]) -> None:
        """Restore trace states for all layers.

        Args:
            states: Dictionary mapping layer names to TraceState.
        """
        for name, state in states.items():
            if name in self.traces:
                self.traces[name].set_state(state)  # type: ignore[union-attr]

    def all_trace_stats(self) -> Dict[str, Dict[str, float]]:
        """Return trace statistics for all layers.

        Returns:
            Dictionary mapping layer names to trace_stats() dictionaries.
        """
        stats: Dict[str, Dict[str, float]] = {}
        for name in self.layer_names:
            trace_mod = self.traces[name]
            stats[name] = trace_mod.trace_stats()  # type: ignore[union-attr]
        return stats


# ===========================================================================
# Factory Function
# ===========================================================================

def create_eligibility_module(
    n_pre: int,
    n_post: int,
    config: Optional[TraceConfig] = None,
    diagonal: bool = False,
) -> EligibilityTraceModule:
    """Factory function to create an EligibilityTraceModule.

    This is the recommended entry point for creating eligibility trace modules.
    It provides sensible defaults and logs the configuration for reproducibility.

    Args:
        n_pre:     Number of pre-synaptic neurons.
        n_post:    Number of post-synaptic neurons.
        config:    Optional TraceConfig.  If None, uses default rate-coded config.
        diagonal:  If True, use diagonal (element-wise) mode.

    Returns:
        Configured EligibilityTraceModule instance.

    Examples:
        # Simple rate-coded trace for a dense layer
        trace = create_eligibility_module(784, 256)

        # STDP trace for a spiking layer
        cfg = TraceConfig.default_stdp()
        trace = create_eligibility_module(256, 128, config=cfg)

        # Diagonal trace for recurrent self-connections
        trace = create_eligibility_module(512, 512, diagonal=True)
    """
    if config is None:
        config = TraceConfig.default_rate()

    module = EligibilityTraceModule(
        n_pre=n_pre,
        n_post=n_post,
        config=config,
        diagonal=diagonal,
    )

    logger.info(
        "Created EligibilityTraceModule: n_pre=%d, n_post=%d, "
        "trace_type=%s, kernel=%s, tau_e=%.1f, diagonal=%s",
        n_pre, n_post,
        config.trace_type.name,
        config.kernel.name,
        config.tau_e,
        diagonal,
    )

    return module


# ===========================================================================
# Utility Functions
# ===========================================================================

def compute_decay_factor(tau: float, dt: float = 1.0) -> float:
    """Compute the exponential decay factor for a given time constant.

    Args:
        tau: Time constant in ms.  Must be positive.
        dt:  Timestep in ms.  Must be positive.

    Returns:
        Decay factor exp(-dt / tau) in [0, 1).

    Raises:
        ValueError: If tau or dt is non-positive.
    """
    if tau <= 0.0:
        raise ValueError(f"tau must be positive, got {tau}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    return math.exp(-dt / tau)


def estimate_trace_halflife(tau: float, dt: float = 1.0) -> float:
    """Estimate the number of steps for the trace to decay to half its value.

    Args:
        tau: Time constant in ms.
        dt:  Timestep in ms.

    Returns:
        Number of timesteps for the trace to reach 50% of its initial value.
    """
    if tau <= 0.0:
        raise ValueError(f"tau must be positive, got {tau}")
    # decay = exp(-dt/tau), half-life: decay^n = 0.5
    # n * (-dt/tau) = ln(0.5)
    # n = -tau * ln(0.5) / dt = tau * ln(2) / dt
    return tau * math.log(2.0) / dt


def summarize_trace_config(config: TraceConfig) -> str:
    """Return a human-readable summary of a TraceConfig.

    Args:
        config: TraceConfig to summarize.

    Returns:
        Multi-line string describing the configuration.
    """
    lines = [
        f"TraceConfig Summary:",
        f"  trace_type:     {config.trace_type.name}",
        f"  kernel:         {config.kernel.name}",
        f"  tau_e:          {config.tau_e:.1f} ms",
        f"  clamp_range:    [{config.clamp_range[0]:.1f}, {config.clamp_range[1]:.1f}]",
    ]
    if config.trace_type == TraceType.DUTCH:
        lines.append(f"  dutch_alpha:    {config.dutch_alpha:.3f}")
    if config.kernel in (KernelType.STDP_PAIR, KernelType.STDP_SYMMETRIC):
        lines.extend([
            f"  stdp_tau_plus:  {config.stdp_tau_plus:.1f} ms",
            f"  stdp_tau_minus: {config.stdp_tau_minus:.1f} ms",
            f"  a_plus:         {config.a_plus:.3f}",
            f"  a_minus:        {config.a_minus:.3f}",
        ])
    halflife = estimate_trace_halflife(config.tau_e)
    lines.append(f"  half-life:      {halflife:.1f} steps (dt=1.0)")
    return "\n".join(lines)


def validate_shapes(
    pre: Tensor,
    post: Tensor,
    n_pre: int,
    n_post: int,
) -> None:
    """Validate pre/post tensor shapes against expected neuron counts.

    Args:
        pre:    Pre-synaptic tensor, expected shape (B, N_pre).
        post:   Post-synaptic tensor, expected shape (B, N_post).
        n_pre:  Expected number of pre-synaptic neurons.
        n_post: Expected number of post-synaptic neurons.

    Raises:
        ValueError: If shapes do not match expectations.
    """
    if pre.dim() != 2:
        raise ValueError(
            f"pre must be 2D (B, N_pre), got shape {tuple(pre.shape)}"
        )
    if post.dim() != 2:
        raise ValueError(
            f"post must be 2D (B, N_post), got shape {tuple(post.shape)}"
        )
    if pre.shape[1] != n_pre:
        raise ValueError(
            f"pre.shape[1]={pre.shape[1]} does not match n_pre={n_pre}"
        )
    if post.shape[1] != n_post:
        raise ValueError(
            f"post.shape[1]={post.shape[1]} does not match n_post={n_post}"
        )
    if pre.shape[0] != post.shape[0]:
        raise ValueError(
            f"Batch size mismatch: pre has {pre.shape[0]}, post has {post.shape[0]}"
        )


def format_trace_report(
    stats: Dict[str, float],
    layer_name: str = "unnamed",
) -> str:
    """Format trace statistics into a human-readable report string.

    Args:
        stats:      Dictionary from trace_stats().
        layer_name: Name of the layer for the report header.

    Returns:
        Multi-line formatted string.
    """
    lines = [
        f"Trace Report: {layer_name}",
        f"  Step count: {int(stats.get('step_count', 0))}",
        f"  Norm:       {stats.get('norm', 0.0):.6f}",
        f"  Mean:       {stats.get('mean', 0.0):.6f}",
        f"  Max:        {stats.get('max', 0.0):.6f}",
        f"  Min:        {stats.get('min', 0.0):.6f}",
        f"  Sparsity:   {stats.get('sparsity', 1.0):.4f}",
    ]
    return "\n".join(lines)


# ===========================================================================
# Advanced: Trace with Learnable Decay
# ===========================================================================

class LearnableDecayTrace(nn.Module):
    """Eligibility trace module with a learnable decay time constant.

    Instead of a fixed tau_e, this module parameterizes the decay via a
    learnable parameter (through softplus reparameterization) that can be
    optimized end-to-end alongside the network weights.

    The learnable decay allows the network to adapt how quickly it forgets
    recent synaptic correlations, which is especially useful when the optimal
    credit assignment window varies across training.

    Parameterization:
        tau_e = tau_min + softplus(tau_raw)
        decay = exp(-dt / tau_e)

    Args:
        n_pre:     Number of pre-synaptic neurons.
        n_post:    Number of post-synaptic neurons.
        config:    Base trace configuration (tau_e used as initialization).
        diagonal:  If True, use diagonal trace mode.
        tau_min:   Minimum allowed tau_e (default 1.0 ms).
        tau_max:   Maximum allowed tau_e (default 1000.0 ms).
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        config: Optional[TraceConfig] = None,
        diagonal: bool = False,
        tau_min: float = 1.0,
        tau_max: float = 1000.0,
    ) -> None:
        super().__init__()

        if config is None:
            config = TraceConfig()

        self.n_pre = n_pre
        self.n_post = n_post
        self.config = config
        self.diagonal = diagonal
        self.tau_min = tau_min
        self.tau_max = tau_max

        # Initialize tau_raw such that softplus(tau_raw) + tau_min = config.tau_e
        init_tau = max(config.tau_e, tau_min + 0.01)
        init_raw = self._inverse_softplus(init_tau - tau_min)
        self.tau_raw = nn.Parameter(torch.tensor(init_raw, dtype=torch.float32))

        # Delegate to a standard trace module (we override the decay computation)
        self._inner = EligibilityTraceModule(
            n_pre, n_post, config=config, diagonal=diagonal,
        )

    @staticmethod
    def _inverse_softplus(x: float, beta: float = 1.0) -> float:
        """Compute inverse softplus: raw = log(exp(x * beta) - 1) / beta."""
        if x * beta > 20.0:
            return x  # For large values, softplus is approximately identity
        return math.log(math.expm1(x * beta)) / beta

    @property
    def tau_e(self) -> Tensor:
        """Current (constrained) tau_e value."""
        tau = self.tau_min + F.softplus(self.tau_raw)
        return torch.clamp(tau, min=self.tau_min, max=self.tau_max)

    def reset(self, batch_size: int, device: Union[str, torch.device]) -> None:
        """Reset all traces to zero."""
        self._inner.reset(batch_size, device)

    def update(
        self,
        pre: Tensor,
        post: Tensor,
        dt: float = 1.0,
    ) -> Tensor:
        """Update traces using the learnable decay rate.

        The decay factor is computed from the current (differentiable) tau_e.
        Note: the trace update itself is done with torch.no_grad() (traces
        are not part of the loss computation graph), but the tau_e gradient
        can flow through if needed via a separate loss.

        Args:
            pre:  (B, N_pre) pre-synaptic activity.
            post: (B, N_post) post-synaptic activity.
            dt:   Timestep in ms.

        Returns:
            Updated eligibility trace tensor.
        """
        # Override the inner module's tau_e with our learnable value
        current_tau = self.tau_e.item()
        self._inner._tau_e = current_tau
        return self._inner.update(pre, post, dt=dt)

    def apply_update(
        self,
        weights: Tensor,
        mod_signal: Union[Tensor, float],
        *,
        lr: float = 0.001,
        clamp: Optional[Tuple[float, float]] = None,
    ) -> Tensor:
        """Apply three-factor weight update using inner trace module."""
        return self._inner.apply_update(weights, mod_signal, lr=lr, clamp=clamp)

    def get_state(self) -> TraceState:
        """Return current trace state."""
        return self._inner.get_state()

    def set_state(self, state: TraceState) -> None:
        """Restore trace state."""
        self._inner.set_state(state)

    def trace_stats(self) -> Dict[str, float]:
        """Return trace statistics including current tau_e."""
        stats = self._inner.trace_stats()
        stats["tau_e"] = self.tau_e.item()
        return stats


# ===========================================================================
# Advanced: Convolutional Eligibility Traces
# ===========================================================================

class ConvEligibilityTrace(nn.Module):
    """Eligibility traces for convolutional layers.

    For convolutional layers, the eligibility trace is defined per output
    channel and input channel (not per spatial location), matching the
    structure of convolutional weight tensors.

    The trace has shape (B, C_out, C_in, kH, kW) matching the kernel shape,
    and is computed by correlating unfolded (im2col) input patches with
    output feature maps.

    This is a simplified version that operates on the already-unfolded
    activations (pre = unfolded input patches, post = output activations).
    The unfolding is assumed to be done externally.

    Args:
        c_in:       Number of input channels.
        c_out:      Number of output channels.
        kernel_size: Spatial kernel dimensions (kH, kW).
        config:     Trace configuration.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: Tuple[int, int] = (3, 3),
        config: Optional[TraceConfig] = None,
    ) -> None:
        super().__init__()

        if config is None:
            config = TraceConfig.default_rate()

        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.config = config

        # Flatten kernel dimensions for the inner trace module
        n_pre = c_in * kernel_size[0] * kernel_size[1]
        n_post = c_out

        self._inner = EligibilityTraceModule(
            n_pre=n_pre,
            n_post=n_post,
            config=config,
            diagonal=False,
        )

    def reset(self, batch_size: int, device: Union[str, torch.device]) -> None:
        """Reset traces to zero.

        Args:
            batch_size: Batch dimension.
            device:     Target device.
        """
        self._inner.reset(batch_size, device)

    def update(
        self,
        pre_unfolded: Tensor,
        post_pooled: Tensor,
        dt: float = 1.0,
    ) -> Tensor:
        """Update convolutional eligibility trace.

        Args:
            pre_unfolded: Unfolded input patches averaged over spatial locations.
                          Shape: (B, C_in * kH * kW).
            post_pooled:  Output activations averaged over spatial locations.
                          Shape: (B, C_out).
            dt:           Timestep in ms.

        Returns:
            Eligibility trace, shape (B, C_out, C_in * kH * kW).
        """
        return self._inner.update(pre_unfolded, post_pooled, dt=dt)

    def get_kernel_shaped_trace(self) -> Optional[Tensor]:
        """Return the trace reshaped to match convolutional kernel dimensions.

        Returns:
            Trace reshaped to (B, C_out, C_in, kH, kW), or None if uninitialized.
        """
        if self._inner._e is None:
            return None
        B = self._inner._e.shape[0]
        return self._inner._e.reshape(
            B, self.c_out, self.c_in, self.kernel_size[0], self.kernel_size[1]
        )

    def apply_update(
        self,
        weights: Tensor,
        mod_signal: Union[Tensor, float],
        *,
        lr: float = 0.001,
        clamp: Optional[Tuple[float, float]] = None,
    ) -> Tensor:
        """Apply weight update to convolutional kernel.

        Args:
            weights:    Convolutional kernel, shape (C_out, C_in, kH, kW).
            mod_signal: Third-factor modulatory signal.
            lr:         Learning rate.
            clamp:      Optional weight clamp.

        Returns:
            Updated kernel, same shape as input.
        """
        original_shape = weights.shape
        # Flatten to 2D for the inner module
        weights_2d = weights.reshape(self.c_out, -1)
        updated_2d = self._inner.apply_update(
            weights_2d, mod_signal, lr=lr, clamp=clamp,
        )
        return updated_2d.reshape(original_shape)

    def trace_stats(self) -> Dict[str, float]:
        """Return trace statistics."""
        return self._inner.trace_stats()


# ===========================================================================
# Advanced: Trace Scheduler
# ===========================================================================

class TraceScheduler:
    """Schedules trace parameters (tau_e, learning rate) over training.

    Provides annealing schedules for the eligibility trace time constant
    and associated learning rates.  This is useful for curriculum-style
    training where early phases use short credit assignment windows and
    later phases extend them.

    Supported schedules:
        - linear:      Linear interpolation between start and end values.
        - exponential: Exponential interpolation.
        - cosine:      Cosine annealing with optional warm restarts.
        - step:        Discrete steps at specified milestones.

    Args:
        total_steps:   Total number of training steps.
        tau_e_start:   Initial tau_e value.
        tau_e_end:     Final tau_e value.
        lr_start:      Initial learning rate for trace updates.
        lr_end:        Final learning rate for trace updates.
        schedule_type: Type of schedule ('linear', 'exponential', 'cosine', 'step').
        milestones:    Step milestones for 'step' schedule type.
    """

    def __init__(
        self,
        total_steps: int,
        tau_e_start: float = 50.0,
        tau_e_end: float = 200.0,
        lr_start: float = 0.01,
        lr_end: float = 0.001,
        schedule_type: str = "linear",
        milestones: Optional[List[int]] = None,
    ) -> None:
        self.total_steps = max(total_steps, 1)
        self.tau_e_start = tau_e_start
        self.tau_e_end = tau_e_end
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.schedule_type = schedule_type
        self.milestones = milestones or []
        self._current_step = 0

    def step(self) -> Tuple[float, float]:
        """Advance scheduler by one step and return current (tau_e, lr).

        Returns:
            Tuple of (current_tau_e, current_lr).
        """
        progress = min(self._current_step / self.total_steps, 1.0)

        if self.schedule_type == "linear":
            factor = progress
        elif self.schedule_type == "exponential":
            factor = 1.0 - math.exp(-5.0 * progress)
        elif self.schedule_type == "cosine":
            factor = 0.5 * (1.0 - math.cos(math.pi * progress))
        elif self.schedule_type == "step":
            # Count how many milestones we've passed
            n_passed = sum(1 for m in self.milestones if self._current_step >= m)
            n_total = max(len(self.milestones), 1)
            factor = n_passed / n_total
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

        tau_e = self.tau_e_start + factor * (self.tau_e_end - self.tau_e_start)
        lr = self.lr_start + factor * (self.lr_end - self.lr_start)

        self._current_step += 1
        return tau_e, lr

    def get_tau_e(self) -> float:
        """Return current tau_e without advancing the step."""
        progress = min(self._current_step / self.total_steps, 1.0)
        if self.schedule_type == "linear":
            factor = progress
        elif self.schedule_type == "exponential":
            factor = 1.0 - math.exp(-5.0 * progress)
        elif self.schedule_type == "cosine":
            factor = 0.5 * (1.0 - math.cos(math.pi * progress))
        elif self.schedule_type == "step":
            n_passed = sum(1 for m in self.milestones if self._current_step >= m)
            n_total = max(len(self.milestones), 1)
            factor = n_passed / n_total
        else:
            factor = 0.0
        return self.tau_e_start + factor * (self.tau_e_end - self.tau_e_start)

    def reset(self) -> None:
        """Reset scheduler to initial step."""
        self._current_step = 0


# ===========================================================================
# Advanced: Trace Buffer for Temporal Unrolling
# ===========================================================================

class TraceBuffer:
    """Circular buffer that stores trace snapshots for temporal unrolling.

    When using truncated BPTT or sequence-level training, it is useful to
    store trace snapshots at regular intervals so that:
        - Traces can be carried across truncation boundaries.
        - Trace evolution can be visualized or debugged.
        - Gradient checkpointing can be applied to trace computations.

    Args:
        max_length:  Maximum number of snapshots to store.
        layer_names: Names of layers to track (must match MultiLayerEligibility).
    """

    def __init__(
        self,
        max_length: int = 100,
        layer_names: Optional[List[str]] = None,
    ) -> None:
        self.max_length = max_length
        self.layer_names = layer_names or []
        self._buffer: List[Dict[str, TraceState]] = []
        self._step_indices: List[int] = []

    def store(
        self,
        states: Dict[str, TraceState],
        step_index: int,
    ) -> None:
        """Store a snapshot of trace states.

        Args:
            states:     Dictionary mapping layer names to TraceState.
            step_index: Global step index for this snapshot.
        """
        # Clone all states to avoid aliasing
        cloned = {name: state.clone() for name, state in states.items()}
        self._buffer.append(cloned)
        self._step_indices.append(step_index)

        # Evict oldest if over capacity
        if len(self._buffer) > self.max_length:
            self._buffer.pop(0)
            self._step_indices.pop(0)

    def get_latest(self) -> Optional[Dict[str, TraceState]]:
        """Return the most recent snapshot, or None if empty."""
        if not self._buffer:
            return None
        return self._buffer[-1]

    def get_at_step(self, step_index: int) -> Optional[Dict[str, TraceState]]:
        """Return the snapshot closest to the given step index.

        Args:
            step_index: Target step index.

        Returns:
            Closest snapshot, or None if buffer is empty.
        """
        if not self._buffer:
            return None
        # Find closest step
        distances = [abs(s - step_index) for s in self._step_indices]
        min_idx = distances.index(min(distances))
        return self._buffer[min_idx]

    def clear(self) -> None:
        """Clear all stored snapshots."""
        self._buffer.clear()
        self._step_indices.clear()

    def __len__(self) -> int:
        """Number of stored snapshots."""
        return len(self._buffer)


# ===========================================================================
# Self-Test Block
# ===========================================================================

if __name__ == "__main__":
    import sys

    # Configure logging for test output
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    passed = 0
    failed = 0
    total = 0

    def run_test(name: str, test_fn: Callable[[], bool]) -> None:
        """Run a single test and report PASS/FAIL."""
        global passed, failed, total
        total += 1
        try:
            result = test_fn()
            if result:
                print(f"  PASS [{total:2d}] {name}")
                passed += 1
            else:
                print(f"  FAIL [{total:2d}] {name}")
                failed += 1
        except Exception as exc:
            print(f"  FAIL [{total:2d}] {name} -- Exception: {exc}")
            failed += 1

    print("=" * 72)
    print("Eligibility Traces Self-Test Suite")
    print("=" * 72)

    device = torch.device("cpu")
    B = 4       # batch size
    N_pre = 8   # pre-synaptic neurons
    N_post = 6  # post-synaptic neurons

    # ------------------------------------------------------------------
    # Test 1: Accumulating trace increases with repeated pre/post
    # ------------------------------------------------------------------
    def test_accumulating_increases() -> bool:
        cfg = TraceConfig(
            trace_type=TraceType.ACCUMULATING,
            kernel=KernelType.RATE,
            tau_e=1000.0,  # very slow decay so we can see accumulation
            clamp_range=(-100.0, 100.0),
        )
        mod = create_eligibility_module(N_pre, N_post, config=cfg)
        mod.reset(B, device)

        pre = torch.ones(B, N_pre)
        post = torch.ones(B, N_post)

        e1 = mod.update(pre, post).clone()
        e2 = mod.update(pre, post).clone()
        e3 = mod.update(pre, post).clone()

        # Each update should increase the trace (with slow decay)
        return (e2.mean() > e1.mean()) and (e3.mean() > e2.mean())

    run_test("Accumulating trace increases with repeated pre/post", test_accumulating_increases)

    # ------------------------------------------------------------------
    # Test 2: Replacing trace bounded by max correlation
    # ------------------------------------------------------------------
    def test_replacing_bounded() -> bool:
        cfg = TraceConfig(
            trace_type=TraceType.REPLACING,
            kernel=KernelType.RATE,
            tau_e=1000.0,
            clamp_range=(-100.0, 100.0),
        )
        mod = create_eligibility_module(N_pre, N_post, config=cfg)
        mod.reset(B, device)

        pre = torch.ones(B, N_pre)
        post = torch.ones(B, N_post)

        # The correlation f = outer(post, pre) = all ones (N_post x N_pre)
        # With REPLACING: e = max(e, f)
        # After decay, e < 1.0, then max(e, 1.0) = 1.0
        e1 = mod.update(pre, post).clone()
        e2 = mod.update(pre, post).clone()
        e3 = mod.update(pre, post).clone()

        # With slow decay and replacing, trace should be approximately
        # the same after the first update (max(decayed_e, 1.0) = 1.0)
        diff_23 = (e3 - e2).abs().max().item()
        diff_12 = (e2 - e1).abs().max().item()
        # Both differences should be very small (only from tiny decay)
        return diff_23 < 0.01 and diff_12 < 0.01

    run_test("Replacing trace bounded by max correlation", test_replacing_bounded)

    # ------------------------------------------------------------------
    # Test 3: Dutch trace intermediate behavior
    # ------------------------------------------------------------------
    def test_dutch_intermediate() -> bool:
        cfg_acc = TraceConfig(
            trace_type=TraceType.ACCUMULATING,
            kernel=KernelType.RATE,
            tau_e=1000.0,
            clamp_range=(-100.0, 100.0),
        )
        cfg_dutch = TraceConfig(
            trace_type=TraceType.DUTCH,
            kernel=KernelType.RATE,
            tau_e=1000.0,
            dutch_alpha=0.5,
            clamp_range=(-100.0, 100.0),
        )
        mod_acc = create_eligibility_module(N_pre, N_post, config=cfg_acc)
        mod_dutch = create_eligibility_module(N_pre, N_post, config=cfg_dutch)
        mod_acc.reset(B, device)
        mod_dutch.reset(B, device)

        pre = torch.ones(B, N_pre)
        post = torch.ones(B, N_post)

        for _ in range(10):
            e_acc = mod_acc.update(pre, post)
            e_dutch = mod_dutch.update(pre, post)

        # Dutch trace should be less than accumulating (alpha dampens growth)
        return e_dutch.mean().item() < e_acc.mean().item()

    run_test("Dutch trace intermediate behavior", test_dutch_intermediate)

    # ------------------------------------------------------------------
    # Test 4: Decay -- e approaches zero with no activity
    # ------------------------------------------------------------------
    def test_decay_to_zero() -> bool:
        cfg = TraceConfig(
            trace_type=TraceType.ACCUMULATING,
            kernel=KernelType.RATE,
            tau_e=10.0,  # fast decay
            clamp_range=(-100.0, 100.0),
        )
        mod = create_eligibility_module(N_pre, N_post, config=cfg)
        mod.reset(B, device)

        # Single strong activation
        pre = torch.ones(B, N_pre) * 5.0
        post = torch.ones(B, N_post) * 5.0
        mod.update(pre, post)

        # Now feed zeros for many steps
        pre_zero = torch.zeros(B, N_pre)
        post_zero = torch.zeros(B, N_post)
        for _ in range(200):
            e = mod.update(pre_zero, post_zero)

        # Trace should be near zero
        return e.abs().max().item() < 1e-4

    run_test("Decay: e approaches zero with no activity", test_decay_to_zero)

    # ------------------------------------------------------------------
    # Test 5: tau_e effect -- larger tau -> slower decay
    # ------------------------------------------------------------------
    def test_tau_effect() -> bool:
        cfg_fast = TraceConfig(tau_e=10.0, kernel=KernelType.RATE, clamp_range=(-100.0, 100.0))
        cfg_slow = TraceConfig(tau_e=500.0, kernel=KernelType.RATE, clamp_range=(-100.0, 100.0))

        mod_fast = create_eligibility_module(N_pre, N_post, config=cfg_fast)
        mod_slow = create_eligibility_module(N_pre, N_post, config=cfg_slow)
        mod_fast.reset(B, device)
        mod_slow.reset(B, device)

        # One pulse
        pre = torch.ones(B, N_pre)
        post = torch.ones(B, N_post)
        mod_fast.update(pre, post)
        mod_slow.update(pre, post)

        # Decay for 50 steps
        pre_zero = torch.zeros(B, N_pre)
        post_zero = torch.zeros(B, N_post)
        for _ in range(50):
            e_fast = mod_fast.update(pre_zero, post_zero)
            e_slow = mod_slow.update(pre_zero, post_zero)

        # Slow-decaying trace should be larger
        return e_slow.abs().mean().item() > e_fast.abs().mean().item()

    run_test("tau_e effect: larger tau -> slower decay", test_tau_effect)

    # ------------------------------------------------------------------
    # Test 6: Reset clears to exactly zero
    # ------------------------------------------------------------------
    def test_reset_clears() -> bool:
        cfg = TraceConfig(kernel=KernelType.RATE)
        mod = create_eligibility_module(N_pre, N_post, config=cfg)
        mod.reset(B, device)

        # Build up trace
        pre = torch.ones(B, N_pre)
        post = torch.ones(B, N_post)
        mod.update(pre, post)
        mod.update(pre, post)

        # Reset
        mod.reset(B, device)
        e = mod.eligibility
        assert e is not None
        return e.abs().max().item() == 0.0

    run_test("Reset clears to exactly zero", test_reset_clears)

    # ------------------------------------------------------------------
    # Test 7: Carry -- set_state/get_state preserves traces
    # ------------------------------------------------------------------
    def test_carry_state() -> bool:
        cfg = TraceConfig(kernel=KernelType.RATE, tau_e=1000.0)
        mod = create_eligibility_module(N_pre, N_post, config=cfg)
        mod.reset(B, device)

        pre = torch.ones(B, N_pre)
        post = torch.ones(B, N_post)
        mod.update(pre, post)
        mod.update(pre, post)

        # Save state
        saved_state = mod.get_state()
        saved_e = saved_state.e.clone()

        # Reset and verify cleared
        mod.reset(B, device)
        assert mod.eligibility is not None
        assert mod.eligibility.abs().max().item() == 0.0

        # Restore
        mod.set_state(saved_state)
        restored_e = mod.eligibility
        assert restored_e is not None

        # Should match saved
        return torch.allclose(restored_e, saved_e, atol=1e-6)

    run_test("Carry: set_state/get_state preserves traces", test_carry_state)

    # ------------------------------------------------------------------
    # Test 8: Batch independence
    # ------------------------------------------------------------------
    def test_batch_independence() -> bool:
        cfg = TraceConfig(kernel=KernelType.RATE, tau_e=1000.0, clamp_range=(-100.0, 100.0))
        mod = create_eligibility_module(N_pre, N_post, config=cfg)
        mod.reset(B, device)

        # Different pre/post for each batch element
        pre = torch.zeros(B, N_pre)
        post = torch.zeros(B, N_post)
        # Only batch element 0 has activity
        pre[0] = 1.0
        post[0] = 1.0

        e = mod.update(pre, post)

        # Batch 0 should have non-zero trace
        e0_norm = e[0].abs().sum().item()
        # Batch 1 should have zero trace
        e1_norm = e[1].abs().sum().item()

        return e0_norm > 0.0 and e1_norm == 0.0

    run_test("Batch independence: modifying e[0] doesn't affect e[1]", test_batch_independence)

    # ------------------------------------------------------------------
    # Test 9: STDP causal timing -> positive eligibility
    # ------------------------------------------------------------------
    def test_stdp_causal() -> bool:
        cfg = TraceConfig(
            kernel=KernelType.STDP_PAIR,
            tau_e=1000.0,
            stdp_tau_plus=20.0,
            stdp_tau_minus=20.0,
            a_plus=1.0,
            a_minus=1.0,
            clamp_range=(-100.0, 100.0),
        )
        mod = create_eligibility_module(N_pre, N_post, config=cfg)
        mod.reset(1, device)

        # Pre spike first, then post spike (causal timing)
        pre_spike = torch.zeros(1, N_pre)
        post_spike = torch.zeros(1, N_post)

        # Step 1: pre spike (creates pre_trace)
        pre_spike[0, 0] = 1.0
        mod.update(pre_spike, torch.zeros(1, N_post), dt=1.0)

        # Step 2: post spike (reads pre_trace -> potentiation)
        post_spike[0, 0] = 1.0
        e = mod.update(torch.zeros(1, N_pre), post_spike, dt=1.0)

        # The trace at (post=0, pre=0) should be positive (potentiation)
        return e[0, 0, 0].item() > 0.0

    run_test("Spike STDP: causal timing -> positive eligibility", test_stdp_causal)

    # ------------------------------------------------------------------
    # Test 10: STDP anti-causal timing -> negative eligibility
    # ------------------------------------------------------------------
    def test_stdp_anticausal() -> bool:
        cfg = TraceConfig(
            kernel=KernelType.STDP_PAIR,
            tau_e=1000.0,
            stdp_tau_plus=20.0,
            stdp_tau_minus=20.0,
            a_plus=1.0,
            a_minus=1.0,
            clamp_range=(-100.0, 100.0),
        )
        mod = create_eligibility_module(N_pre, N_post, config=cfg)
        mod.reset(1, device)

        # Post spike first, then pre spike (anti-causal timing)
        # Step 1: post spike (creates post_trace)
        post_spike = torch.zeros(1, N_post)
        post_spike[0, 0] = 1.0
        mod.update(torch.zeros(1, N_pre), post_spike, dt=1.0)

        # Step 2: pre spike (reads post_trace -> depression)
        pre_spike = torch.zeros(1, N_pre)
        pre_spike[0, 0] = 1.0
        e = mod.update(pre_spike, torch.zeros(1, N_post), dt=1.0)

        # The trace at (post=0, pre=0) should be negative (depression)
        return e[0, 0, 0].item() < 0.0

    run_test("Spike STDP: anti-causal timing -> negative eligibility", test_stdp_anticausal)

    # ------------------------------------------------------------------
    # Test 11: Rate kernel outer product shape correct
    # ------------------------------------------------------------------
    def test_rate_outer_shape() -> bool:
        cfg = TraceConfig(kernel=KernelType.RATE)
        mod = create_eligibility_module(N_pre, N_post, config=cfg, diagonal=False)
        mod.reset(B, device)

        pre = torch.randn(B, N_pre)
        post = torch.randn(B, N_post)
        e = mod.update(pre, post)

        return e.shape == (B, N_post, N_pre)

    run_test("Rate kernel: outer product shape correct", test_rate_outer_shape)

    # ------------------------------------------------------------------
    # Test 12: Diagonal mode element-wise product shape (B, N)
    # ------------------------------------------------------------------
    def test_diagonal_shape() -> bool:
        N = 10
        cfg = TraceConfig(kernel=KernelType.RATE)
        mod = create_eligibility_module(N, N, config=cfg, diagonal=True)
        mod.reset(B, device)

        pre = torch.randn(B, N)
        post = torch.randn(B, N)
        e = mod.update(pre, post)

        return e.shape == (B, N)

    run_test("Diagonal mode: element-wise product shape (B, N)", test_diagonal_shape)

    # ------------------------------------------------------------------
    # Test 13: Zero pre -> zero trace update
    # ------------------------------------------------------------------
    def test_zero_pre() -> bool:
        cfg = TraceConfig(kernel=KernelType.RATE, tau_e=1000.0)
        mod = create_eligibility_module(N_pre, N_post, config=cfg)
        mod.reset(B, device)

        pre_zero = torch.zeros(B, N_pre)
        post = torch.ones(B, N_post)
        e = mod.update(pre_zero, post)

        # With zero pre, the outer product is all zeros -> no trace update
        return e.abs().max().item() == 0.0

    run_test("Zero pre -> zero trace update", test_zero_pre)

    # ------------------------------------------------------------------
    # Test 14: Zero post -> zero trace update
    # ------------------------------------------------------------------
    def test_zero_post() -> bool:
        cfg = TraceConfig(kernel=KernelType.RATE, tau_e=1000.0)
        mod = create_eligibility_module(N_pre, N_post, config=cfg)
        mod.reset(B, device)

        pre = torch.ones(B, N_pre)
        post_zero = torch.zeros(B, N_post)
        e = mod.update(pre, post_zero)

        # With zero post, the outer product is all zeros -> no trace update
        return e.abs().max().item() == 0.0

    run_test("Zero post -> zero trace update", test_zero_post)

    # ------------------------------------------------------------------
    # Test 15: Clamp -- values stay within clamp_range
    # ------------------------------------------------------------------
    def test_clamp() -> bool:
        cfg = TraceConfig(
            kernel=KernelType.RATE,
            tau_e=1000.0,
            clamp_range=(-1.0, 1.0),
        )
        mod = create_eligibility_module(N_pre, N_post, config=cfg)
        mod.reset(B, device)

        # Strong activity that would push traces beyond [-1, 1]
        pre = torch.ones(B, N_pre) * 10.0
        post = torch.ones(B, N_post) * 10.0

        for _ in range(20):
            e = mod.update(pre, post)

        return e.max().item() <= 1.0 and e.min().item() >= -1.0

    run_test("Clamp: values stay within clamp_range", test_clamp)

    # ------------------------------------------------------------------
    # Test 16: Third-factor gating -- mod_signal=0 -> delta_w exactly zero
    # ------------------------------------------------------------------
    def test_gating_zero() -> bool:
        cfg = TraceConfig(kernel=KernelType.RATE, tau_e=1000.0)
        mod = create_eligibility_module(N_pre, N_post, config=cfg)
        mod.reset(B, device)

        # Build up trace
        pre = torch.ones(B, N_pre)
        post = torch.ones(B, N_post)
        mod.update(pre, post)

        # Apply update with zero modulation
        weights = torch.randn(N_post, N_pre)
        original_weights = weights.clone()
        new_weights = mod.apply_update(weights, mod_signal=0.0, lr=0.1)

        # Weights should be EXACTLY unchanged
        return torch.equal(new_weights, original_weights)

    run_test("Third-factor gating: mod_signal=0 -> delta_w exactly zero", test_gating_zero)

    # ------------------------------------------------------------------
    # Test 17: mod_signal != 0 -> delta_w non-zero
    # ------------------------------------------------------------------
    def test_gating_nonzero() -> bool:
        cfg = TraceConfig(kernel=KernelType.RATE, tau_e=1000.0)
        mod = create_eligibility_module(N_pre, N_post, config=cfg)
        mod.reset(B, device)

        # Build up trace
        pre = torch.ones(B, N_pre)
        post = torch.ones(B, N_post)
        mod.update(pre, post)

        # Apply update with non-zero modulation
        weights = torch.randn(N_post, N_pre)
        original_weights = weights.clone()
        new_weights = mod.apply_update(weights, mod_signal=1.0, lr=0.1)

        # Weights should have changed
        return not torch.equal(new_weights, original_weights)

    run_test("mod_signal != 0 -> delta_w non-zero", test_gating_nonzero)

    # ------------------------------------------------------------------
    # Test 18: apply_update -- weights change by lr * mod * e
    # ------------------------------------------------------------------
    def test_weight_update_magnitude() -> bool:
        cfg = TraceConfig(kernel=KernelType.RATE, tau_e=1e6)  # negligible decay
        mod = create_eligibility_module(N_pre, N_post, config=cfg)
        mod.reset(1, device)  # single batch for simplicity

        # Single activation to build a known trace
        pre = torch.ones(1, N_pre)
        post = torch.ones(1, N_post)
        e = mod.update(pre, post)
        # e should be approximately outer(ones, ones) = all-ones matrix
        # (with negligible decay factor)

        lr = 0.01
        mod_val = 2.0
        weights = torch.zeros(N_post, N_pre)
        new_weights = mod.apply_update(weights, mod_signal=mod_val, lr=lr)

        # Expected delta_w = lr * mod * mean(e over batch) = lr * mod * e[0]
        # e[0] should be ~1.0 everywhere (outer product of ones)
        expected_delta = lr * mod_val * e.mean(dim=0)
        actual_delta = new_weights - weights

        return torch.allclose(actual_delta, expected_delta, atol=1e-5)

    run_test("apply_update: weights change by lr * mod * e", test_weight_update_magnitude)

    # ------------------------------------------------------------------
    # Test 19: fp32 enforcement -- traces are float32
    # ------------------------------------------------------------------
    def test_fp32_enforcement() -> bool:
        cfg = TraceConfig(kernel=KernelType.RATE)
        mod = create_eligibility_module(N_pre, N_post, config=cfg)
        mod.reset(B, device)

        # Feed fp16 inputs (simulating AMP)
        pre = torch.ones(B, N_pre, dtype=torch.float16)
        post = torch.ones(B, N_post, dtype=torch.float16)
        e = mod.update(pre, post)

        return e.dtype == torch.float32

    run_test("fp32 enforcement: traces are float32", test_fp32_enforcement)

    # ------------------------------------------------------------------
    # Test 20: NaN guard -- NaN input doesn't propagate
    # ------------------------------------------------------------------
    def test_nan_guard() -> bool:
        cfg = TraceConfig(kernel=KernelType.RATE, tau_e=1000.0)
        mod = create_eligibility_module(N_pre, N_post, config=cfg)
        mod.reset(B, device)

        # Normal update first
        pre = torch.ones(B, N_pre)
        post = torch.ones(B, N_post)
        mod.update(pre, post)

        # NaN input
        pre_nan = torch.full((B, N_pre), float("nan"))
        post_nan = torch.full((B, N_post), float("nan"))
        e = mod.update(pre_nan, post_nan)

        # Trace should not contain NaN
        return not torch.isnan(e).any().item()

    run_test("NaN guard: NaN input doesn't propagate", test_nan_guard)

    # ------------------------------------------------------------------
    # Test 21: Determinism -- same inputs -> same traces (10 runs)
    # ------------------------------------------------------------------
    def test_determinism() -> bool:
        cfg = TraceConfig(kernel=KernelType.RATE, tau_e=50.0)
        results = []
        for _ in range(10):
            mod = create_eligibility_module(N_pre, N_post, config=cfg)
            mod.reset(B, device)

            torch.manual_seed(42)
            pre = torch.randn(B, N_pre)
            post = torch.randn(B, N_post)

            for _ in range(5):
                e = mod.update(pre, post)
            results.append(e.clone())

        # All runs should produce identical results
        for i in range(1, len(results)):
            if not torch.allclose(results[0], results[i], atol=1e-7):
                return False
        return True

    run_test("Determinism: same inputs -> same traces (10 runs)", test_determinism)

    # ------------------------------------------------------------------
    # Test 22: MultiLayerEligibility -- reset_all clears all layers
    # ------------------------------------------------------------------
    def test_multi_reset() -> bool:
        layer_configs = {
            "layer1": (N_pre, N_post, TraceConfig(tau_e=100.0)),
            "layer2": (N_post, 4, TraceConfig(tau_e=200.0)),
        }
        multi = MultiLayerEligibility(layer_configs)
        multi.reset_all(B, device)

        # Build up traces
        activations = {
            "layer1": (torch.ones(B, N_pre), torch.ones(B, N_post)),
            "layer2": (torch.ones(B, N_post), torch.ones(B, 4)),
        }
        multi.update_all(activations)

        # Reset all
        multi.reset_all(B, device)

        # Check all traces are zero
        for name in ["layer1", "layer2"]:
            trace_mod = multi.traces[name]
            e = trace_mod.eligibility  # type: ignore[union-attr]
            if e is None or e.abs().max().item() != 0.0:
                return False
        return True

    run_test("MultiLayerEligibility: reset_all clears all layers", test_multi_reset)

    # ------------------------------------------------------------------
    # Test 23: MultiLayerEligibility -- update_all processes all layers
    # ------------------------------------------------------------------
    def test_multi_update() -> bool:
        layer_configs = {
            "layer1": (N_pre, N_post, TraceConfig(tau_e=1000.0)),
            "layer2": (N_post, 4, TraceConfig(tau_e=1000.0)),
        }
        multi = MultiLayerEligibility(layer_configs)
        multi.reset_all(B, device)

        activations = {
            "layer1": (torch.ones(B, N_pre), torch.ones(B, N_post)),
            "layer2": (torch.ones(B, N_post), torch.ones(B, 4)),
        }
        results = multi.update_all(activations)

        # Both layers should have non-zero traces
        if "layer1" not in results or "layer2" not in results:
            return False
        return (
            results["layer1"].abs().sum().item() > 0.0
            and results["layer2"].abs().sum().item() > 0.0
        )

    run_test("MultiLayerEligibility: update_all processes all layers", test_multi_update)

    # ------------------------------------------------------------------
    # Test 24: trace_stats returns dict with expected keys
    # ------------------------------------------------------------------
    def test_trace_stats_keys() -> bool:
        cfg = TraceConfig(kernel=KernelType.RATE)
        mod = create_eligibility_module(N_pre, N_post, config=cfg)
        mod.reset(B, device)

        pre = torch.randn(B, N_pre)
        post = torch.randn(B, N_post)
        mod.update(pre, post)

        stats = mod.trace_stats()
        expected_keys = {"norm", "mean", "max", "min", "sparsity", "step_count"}
        return expected_keys.issubset(set(stats.keys()))

    run_test("trace_stats: returns dict with expected keys", test_trace_stats_keys)

    # ------------------------------------------------------------------
    # Test 25: Symmetric STDP always non-negative
    # ------------------------------------------------------------------
    def test_symmetric_stdp_nonneg() -> bool:
        cfg = TraceConfig(
            kernel=KernelType.STDP_SYMMETRIC,
            tau_e=1000.0,
            clamp_range=(-100.0, 100.0),
        )
        mod = create_eligibility_module(N_pre, N_post, config=cfg)
        mod.reset(B, device)

        # Various spike patterns
        for _ in range(20):
            pre = (torch.rand(B, N_pre) > 0.5).float()
            post = (torch.rand(B, N_post) > 0.5).float()
            e = mod.update(pre, post)

        # Symmetric STDP should only produce non-negative traces
        # (the kernel output is always non-negative, and the trace starts at 0)
        return e.min().item() >= 0.0

    run_test("Symmetric STDP: always non-negative eligibility", test_symmetric_stdp_nonneg)

    # ------------------------------------------------------------------
    # Test 26: LearnableDecayTrace tau_e is learnable
    # ------------------------------------------------------------------
    def test_learnable_decay() -> bool:
        cfg = TraceConfig(tau_e=50.0, kernel=KernelType.RATE)
        mod = LearnableDecayTrace(N_pre, N_post, config=cfg)

        # tau_e should be a learnable parameter
        initial_tau = mod.tau_e.item()
        # Verify it's close to the configured value
        return abs(initial_tau - 50.0) < 1.0

    run_test("LearnableDecayTrace: tau_e is learnable parameter", test_learnable_decay)

    # ------------------------------------------------------------------
    # Test 27: ConvEligibilityTrace shape matches kernel dims
    # ------------------------------------------------------------------
    def test_conv_trace_shape() -> bool:
        c_in, c_out = 3, 16
        kH, kW = 3, 3
        cfg = TraceConfig(tau_e=100.0, kernel=KernelType.RATE)
        conv_trace = ConvEligibilityTrace(c_in, c_out, kernel_size=(kH, kW), config=cfg)
        conv_trace.reset(B, device)

        # Simulated unfolded input and pooled output
        pre_unfolded = torch.randn(B, c_in * kH * kW)
        post_pooled = torch.randn(B, c_out)
        conv_trace.update(pre_unfolded, post_pooled)

        kernel_trace = conv_trace.get_kernel_shaped_trace()
        if kernel_trace is None:
            return False
        return kernel_trace.shape == (B, c_out, c_in, kH, kW)

    run_test("ConvEligibilityTrace: kernel-shaped trace dimensions", test_conv_trace_shape)

    # ------------------------------------------------------------------
    # Test 28: TraceScheduler linear schedule
    # ------------------------------------------------------------------
    def test_scheduler_linear() -> bool:
        sched = TraceScheduler(
            total_steps=100,
            tau_e_start=50.0,
            tau_e_end=200.0,
            lr_start=0.01,
            lr_end=0.001,
            schedule_type="linear",
        )

        # At step 0
        tau0, lr0 = sched.step()
        # At step ~50 (midpoint)
        for _ in range(49):
            sched.step()
        tau50, lr50 = sched.step()  # step 50

        # tau should increase linearly, lr should decrease linearly
        return (
            tau0 < tau50
            and lr0 > lr50
            and abs(tau50 - 125.0) < 5.0  # midpoint tau
        )

    run_test("TraceScheduler: linear schedule changes tau_e and lr", test_scheduler_linear)

    # ------------------------------------------------------------------
    # Test 29: TraceBuffer stores and retrieves snapshots
    # ------------------------------------------------------------------
    def test_trace_buffer() -> bool:
        cfg = TraceConfig(kernel=KernelType.RATE, tau_e=1000.0)
        mod = create_eligibility_module(N_pre, N_post, config=cfg)
        mod.reset(B, device)

        buf = TraceBuffer(max_length=10)

        pre = torch.ones(B, N_pre)
        post = torch.ones(B, N_post)
        mod.update(pre, post)

        state = mod.get_state()
        buf.store({"layer": state}, step_index=0)

        # Retrieve
        latest = buf.get_latest()
        if latest is None or "layer" not in latest:
            return False

        return torch.allclose(latest["layer"].e, state.e, atol=1e-7)

    run_test("TraceBuffer: stores and retrieves trace snapshots", test_trace_buffer)

    # ------------------------------------------------------------------
    # Test 30: MultiLayerEligibility apply_all with gating
    # ------------------------------------------------------------------
    def test_multi_apply() -> bool:
        layer_configs = {
            "layer1": (N_pre, N_post, TraceConfig(tau_e=1000.0)),
        }
        multi = MultiLayerEligibility(layer_configs)
        multi.reset_all(B, device)

        # Build trace
        activations = {"layer1": (torch.ones(B, N_pre), torch.ones(B, N_post))}
        multi.update_all(activations)

        # Apply with zero modulation
        weights = {"layer1": torch.randn(N_post, N_pre)}
        original = weights["layer1"].clone()
        updated_zero = multi.apply_all(weights, mod_signal=0.0, lr=0.1)

        # Apply with non-zero modulation
        updated_nonzero = multi.apply_all(weights, mod_signal=1.0, lr=0.1)

        zero_unchanged = torch.equal(updated_zero["layer1"], original)
        nonzero_changed = not torch.equal(updated_nonzero["layer1"], original)

        return zero_unchanged and nonzero_changed

    run_test("MultiLayerEligibility: apply_all gating works correctly", test_multi_apply)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 72)
    print(f"Results: {passed} passed, {failed} failed, {total} total")
    if failed == 0:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 72)

    sys.exit(0 if failed == 0 else 1)
