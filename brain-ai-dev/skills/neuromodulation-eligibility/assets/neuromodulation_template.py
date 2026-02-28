"""
Neuromodulatory Gate Template -- DA / ACh / NE / 5-HT Computation

Complete, self-contained module for computing four neuromodulatory signals
from observable pipeline signals (reward, novelty, entropy, anomaly, urgency,
surprise, patience, horizon_value) and combining them into a single
global_plasticity gain that gates three-factor weight updates.

Brain analogs:
    - Dopamine  (DA)  : Reward prediction error     -> [-1, 1]  via tanh
    - Acetylcholine (ACh) : Novelty / uncertainty    -> [ 0, 1]  via sigmoid
    - Norepinephrine (NE) : Urgency / surprise       -> [ 0, 1]  via sigmoid
    - Serotonin (5-HT)    : Patience / horizon value -> [ 0, 1]  via sigmoid

Architecture summary
--------------------
The module is organized into clearly separated components:

1. **Enums** (``ModulatorType``, ``CombinationFn``) define the valid
   modulator identifiers and combination strategies.

2. **Data classes** (``ModulatorOutput``, ``ModulatorState``) carry
   the outputs and mutable state through the pipeline without
   requiring global variables or side-effects.

3. **Configuration** (``NeuromodConfig``) exposes every tuneable knob
   with validated defaults and factory presets for minimal, dev, and
   production scales.

4. **Individual modulator modules** (``DopamineModule``,
   ``AcetylcholineModule``, ``NorepinephrineModule``,
   ``SerotoninModule``) each implement a single signal-to-modulator
   mapping with bounded activation functions.

5. **Combination functions** (``WeightedSumCombination``,
   ``GatedProductCombination``, ``MLPCombination``) reduce the four
   modulator scalars to a single plasticity gain.

6. **Signal normalisation** (``RunningNormalizer``) optionally
   stabilises input signals via online running statistics.

7. **NeuromodulatoryGate** ties everything together: it extracts
   signals from a dictionary, feeds them to the correct modules,
   combines the outputs, and updates persistent state.

8. **Utility helpers** (``ModulatorHistory``,
   ``compute_effective_eligibility_decay``,
   ``modulator_output_to_dict``) simplify downstream integration
   with eligibility traces and diagnostics logging.

Design principles:
    - Deterministic: identical inputs + state produce identical outputs.
    - Bounded: every modulator is provably within its output range.
    - Pure torch: no external dependencies beyond PyTorch standard library.
    - fp32 safe: all modulator computations forced to fp32 under AMP.
    - Batch-independent: batch items never interact during computation.
    - Graceful defaults: missing signals default to zero (neutral).
    - Serialisable: state can be saved / loaded for checkpointing.

Usage:
    >>> from neuromodulation_template import NeuromodulatoryGate, NeuromodConfig
    >>> config = NeuromodConfig()
    >>> gate = NeuromodulatoryGate(config)
    >>> signals = {"reward": torch.tensor([1.0, -0.5])}
    >>> output, state = gate(signals)
    >>> print(output.da, output.global_plasticity)

Integration with eligibility traces:
    The ``global_plasticity`` field is the third-factor modulator ``M(t)``
    in the three-factor weight update rule ``delta_w = lr * M(t) * e(t)``.
    When ``global_plasticity`` is zero, no weight update occurs regardless
    of eligibility trace magnitude.

    For per-modulator integration (e.g. ACh modulating workspace
    temperature, NE modulating HTM learning rate), access individual
    modulator values via ``output.da``, ``output.ach``, ``output.ne``,
    ``output.sht``.
"""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ===================================================================
# Section 1 -- Enums
# ===================================================================

class ModulatorType(Enum):
    """Identifiers for the four neuromodulatory systems."""

    DA = "dopamine"
    ACH = "acetylcholine"
    NE = "norepinephrine"
    SHT = "serotonin"


class CombinationFn(Enum):
    """Strategy for combining four modulators into a scalar plasticity gain."""

    WEIGHTED_SUM = "weighted_sum"
    GATED_PRODUCT = "gated_product"
    MLP = "mlp"


# ===================================================================
# Section 2 -- Data Classes
# ===================================================================

@dataclass
class ModulatorOutput:
    """Container for all modulator values produced by a single forward pass.

    Attributes:
        da: Dopamine signal, shape ``(B,)`` or ``(B, 1)``, range ``[-1, 1]``.
        ach: Acetylcholine signal, shape ``(B,)`` or ``(B, 1)``, range ``[0, 1]``.
        ne: Norepinephrine signal, shape ``(B,)`` or ``(B, 1)``, range ``[0, 1]``.
        sht: Serotonin (5-HT) signal, shape ``(B,)`` or ``(B, 1)``, range ``[0, 1]``.
        global_plasticity: Combined scalar plasticity gain, shape ``(B,)``.
        metadata: Arbitrary diagnostic information attached by the gate.
    """

    da: torch.Tensor
    ach: torch.Tensor
    ne: torch.Tensor
    sht: torch.Tensor
    global_plasticity: torch.Tensor
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModulatorState:
    """Mutable state carried across forward passes.

    Attributes:
        reward_baseline: Exponential moving average of observed rewards,
            shape ``(B,)`` in fp32.
        novelty_history: Ring-buffer of recent novelty values used for
            adaptive normalisation, shape ``(history_len,)`` in fp32.
        arousal_state: Smoothed NE value from the previous step, ``(B,)``.
        step_count: Number of forward passes since last reset.
    """

    reward_baseline: torch.Tensor
    novelty_history: torch.Tensor
    arousal_state: torch.Tensor
    step_count: int = 0

    # -- serialisation helpers ------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary (tensors are detached)."""
        return {
            "reward_baseline": self.reward_baseline.detach().clone(),
            "novelty_history": self.novelty_history.detach().clone(),
            "arousal_state": self.arousal_state.detach().clone(),
            "step_count": self.step_count,
        }

    @staticmethod
    def from_state_dict(d: Dict[str, Any]) -> "ModulatorState":
        """Reconstruct from a dictionary produced by :meth:`state_dict`."""
        return ModulatorState(
            reward_baseline=d["reward_baseline"],
            novelty_history=d["novelty_history"],
            arousal_state=d["arousal_state"],
            step_count=d["step_count"],
        )


# ===================================================================
# Section 3 -- Configuration
# ===================================================================

@dataclass
class NeuromodConfig:
    """Full configuration for the :class:`NeuromodulatoryGate`.

    All fields have sensible defaults that produce a working gate
    straight out of the box.

    Attributes:
        combination_fn: How to combine four modulators into
            ``global_plasticity``.
        modulator_hidden_dim: Hidden dimension for the MLP combiner
            (only used when ``combination_fn == "mlp"``).
        da_ema_alpha: EMA rate for the reward baseline in the dopamine
            module.  Smaller values make the baseline more sluggish.
        ne_smoothing_beta: Temporal smoothing coefficient for NE.
            ``NE = (1-beta)*NE_prev + beta*NE_raw``.
        normalise_inputs: Whether to apply running-statistics adaptive
            normalisation to incoming signals.
        normalise_momentum: Momentum for the running mean / var
            statistics (only used when ``normalise_inputs`` is True).
        novelty_history_len: Length of the ring-buffer used by
            :class:`ModulatorState` for novelty history.
        clamp_input_range: Hard clamp applied to every input signal
            before modulator computation.
        w_da_init: Initial value for the learnable ``w_reward`` scale
            inside the DA module.
        w_ach_novelty_init: Initial weight for novelty in ACh.
        w_ach_entropy_init: Initial weight for entropy in ACh.
        w_ach_anomaly_init: Initial weight for HTM anomaly in ACh.
        ach_bias_init: Bias initialisation for the ACh linear gate.
        w_ne_urgency_init: Initial weight for urgency in NE.
        w_ne_surprise_init: Initial weight for surprise in NE.
        ne_bias_init: Bias initialisation for the NE linear gate.
        w_sht_patience_init: Initial weight for patience in 5-HT.
        w_sht_horizon_init: Initial weight for horizon value in 5-HT.
        sht_bias_init: Bias initialisation for the 5-HT linear gate.
        ws_da: Weight for DA in the weighted-sum combiner.
        ws_ach: Weight for ACh in the weighted-sum combiner.
        ws_ne: Weight for NE in the weighted-sum combiner.
        ws_sht: Weight for 5-HT in the weighted-sum combiner.
        gp_ach_weight: Mixing weight for ACh in the gated-product
            combiner.
        gp_baseline_ach: Baseline ACh fallback for gated-product.
        learnable_weights: When ``True``, modulator input weights
            (w_reward, w_novelty, ...) are ``nn.Parameter``.
            When ``False``, they are plain floats (not trained).
    """

    # Combination
    combination_fn: str = "weighted_sum"
    modulator_hidden_dim: int = 64

    # DA
    da_ema_alpha: float = 0.1
    w_da_init: float = 1.0

    # ACh
    w_ach_novelty_init: float = 1.0
    w_ach_entropy_init: float = 0.5
    w_ach_anomaly_init: float = 0.5
    ach_bias_init: float = 0.0

    # NE
    w_ne_urgency_init: float = 1.0
    w_ne_surprise_init: float = 0.5
    ne_bias_init: float = 0.0
    ne_smoothing_beta: float = 0.3

    # 5-HT
    w_sht_patience_init: float = 1.0
    w_sht_horizon_init: float = 0.5
    sht_bias_init: float = 0.0

    # Combination weights (weighted_sum)
    ws_da: float = 0.4
    ws_ach: float = 0.25
    ws_ne: float = 0.2
    ws_sht: float = 0.15

    # Gated product parameters
    gp_ach_weight: float = 0.5
    gp_baseline_ach: float = 0.5

    # Normalisation
    normalise_inputs: bool = False
    normalise_momentum: float = 0.1
    novelty_history_len: int = 100
    clamp_input_range: Tuple[float, float] = (-10.0, 10.0)

    # Misc
    learnable_weights: bool = True

    # ----- validation -------------------------------------------------------

    def __post_init__(self) -> None:
        valid_fns = {e.value for e in CombinationFn}
        if self.combination_fn not in valid_fns:
            raise ValueError(
                f"combination_fn must be one of {valid_fns}, "
                f"got '{self.combination_fn}'"
            )
        if self.da_ema_alpha <= 0.0 or self.da_ema_alpha > 1.0:
            raise ValueError(
                f"da_ema_alpha must be in (0, 1], got {self.da_ema_alpha}"
            )
        if self.ne_smoothing_beta <= 0.0 or self.ne_smoothing_beta > 1.0:
            raise ValueError(
                f"ne_smoothing_beta must be in (0, 1], got {self.ne_smoothing_beta}"
            )
        lo, hi = self.clamp_input_range
        if lo >= hi:
            raise ValueError(
                f"clamp_input_range[0] ({lo}) must be < clamp_input_range[1] ({hi})"
            )

    # ----- presets -----------------------------------------------------------

    @classmethod
    def minimal(cls) -> "NeuromodConfig":
        """Minimal config for unit tests and smoke tests."""
        return cls(
            combination_fn="weighted_sum",
            modulator_hidden_dim=16,
            normalise_inputs=False,
            learnable_weights=False,
        )

    @classmethod
    def dev(cls) -> "NeuromodConfig":
        """Development config with learnable weights and MLP combiner."""
        return cls(
            combination_fn="mlp",
            modulator_hidden_dim=32,
            normalise_inputs=True,
            learnable_weights=True,
        )

    @classmethod
    def production(cls) -> "NeuromodConfig":
        """Full production config with adaptive normalisation."""
        return cls(
            combination_fn="mlp",
            modulator_hidden_dim=64,
            normalise_inputs=True,
            learnable_weights=True,
        )


# ===================================================================
# Section 4 -- Signal Normalisation
# ===================================================================

class RunningNormalizer(nn.Module):
    """Tracks running mean and variance for adaptive input normalisation.

    Uses Welford-style online statistics with configurable momentum.
    The normaliser maps inputs to approximate zero mean and unit variance,
    then clamps outputs to the configured range.

    Args:
        num_signals: Number of independent signals to normalise.
        momentum: Weight given to the new batch statistics.
        clamp_range: Hard clamp applied after normalisation.
    """

    def __init__(
        self,
        num_signals: int,
        momentum: float = 0.1,
        clamp_range: Tuple[float, float] = (-10.0, 10.0),
    ) -> None:
        super().__init__()
        self.momentum = momentum
        self.clamp_lo = clamp_range[0]
        self.clamp_hi = clamp_range[1]
        self.register_buffer("running_mean", torch.zeros(num_signals))
        self.register_buffer("running_var", torch.ones(num_signals))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise ``x`` of shape ``(B, num_signals)`` in-place.

        During training the running statistics are updated.  During eval
        the statistics are frozen.
        """
        if self.training and x.shape[0] > 1:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False).clamp(min=1e-8)
            self.running_mean.mul_(1.0 - self.momentum).add_(
                self.momentum * batch_mean
            )
            self.running_var.mul_(1.0 - self.momentum).add_(
                self.momentum * batch_var
            )
            self.count.add_(1)

        normed = (x - self.running_mean) / (self.running_var.sqrt() + 1e-8)
        return torch.clamp(normed, self.clamp_lo, self.clamp_hi)


# ===================================================================
# Section 5 -- Individual Modulator Modules
#
# Each modulator is an independent nn.Module that maps one or more
# observable signals to a bounded scalar output.  They share common
# design traits:
#
# - All computation is wrapped in torch.cuda.amp.autocast(enabled=False)
#   to guarantee fp32 precision even when the rest of the model runs
#   under automatic mixed precision.
#
# - Activation functions are chosen to guarantee bounded outputs:
#       DA  uses tanh  -> [-1, 1]
#       ACh uses sigmoid -> [0, 1]
#       NE  uses sigmoid -> [0, 1]
#       5-HT uses sigmoid -> [0, 1]
#
# - Each module is stateless within a single call; any persistent
#   state (e.g. EMA baselines, smoothed values) is passed in and
#   returned explicitly via the ModulatorState dataclass.
#
# - Weights can be either learnable nn.Parameters (trained end-to-end
#   through backprop) or fixed buffers (set from config).  The choice
#   is controlled by the ``learnable`` flag.
# ===================================================================


class DopamineModule(nn.Module):
    """Computes the dopamine (DA) modulatory signal.

    DA encodes reward prediction error (RPE).  An exponential moving
    average (EMA) baseline tracks the expected reward; DA is the
    tanh-squashed difference between the observed reward and this
    baseline.

    .. math::

        \\text{DA} = \\tanh(w_{reward} \\cdot (r - \\bar{r}))

    The baseline updates as:

    .. math::

        \\bar{r} \\leftarrow (1 - \\alpha) \\bar{r} + \\alpha \\, r

    Args:
        w_reward_init: Initial scale for the reward-baseline difference.
        alpha: EMA rate for baseline tracking.
        learnable: If ``True``, ``w_reward`` is an ``nn.Parameter``.
    """

    def __init__(
        self,
        w_reward_init: float = 1.0,
        alpha: float = 0.1,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        self.alpha = alpha

        if learnable:
            self.w_reward = nn.Parameter(torch.tensor(w_reward_init))
        else:
            self.register_buffer("w_reward", torch.tensor(w_reward_init))

    # ---- forward -----------------------------------------------------------

    def forward(
        self,
        reward: torch.Tensor,
        baseline: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute DA from a reward signal and an EMA baseline.

        Args:
            reward: Shape ``(B,)`` -- the current reward observation.
            baseline: Shape ``(B,)`` -- the current EMA baseline.

        Returns:
            da: Dopamine signal in ``[-1, 1]``, shape ``(B,)``.
            new_baseline: Updated EMA baseline, shape ``(B,)``.
        """
        with torch.cuda.amp.autocast(enabled=False):
            reward = reward.float()
            baseline = baseline.float()
            rpe = reward - baseline
            da = torch.tanh(self.w_reward * rpe)
            new_baseline = (1.0 - self.alpha) * baseline + self.alpha * reward
        return da, new_baseline


class AcetylcholineModule(nn.Module):
    """Computes the acetylcholine (ACh) modulatory signal.

    ACh encodes a combined novelty / uncertainty / anomaly gate.
    High ACh signals that the current input is novel or uncertain,
    promoting learning and attention.

    .. math::

        \\text{ACh} = \\sigma(w_n \\cdot n + w_e \\cdot e + w_a \\cdot a + b)

    where *n* = novelty, *e* = entropy, *a* = HTM anomaly.

    Args:
        w_novelty_init: Initial weight for the novelty signal.
        w_entropy_init: Initial weight for the entropy signal.
        w_anomaly_init: Initial weight for the HTM anomaly signal.
        bias_init: Initial bias (shifts the sigmoid baseline).
        learnable: If ``True``, weights/bias are ``nn.Parameter``.
    """

    def __init__(
        self,
        w_novelty_init: float = 1.0,
        w_entropy_init: float = 0.5,
        w_anomaly_init: float = 0.5,
        bias_init: float = 0.0,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        if learnable:
            self.w_novelty = nn.Parameter(torch.tensor(w_novelty_init))
            self.w_entropy = nn.Parameter(torch.tensor(w_entropy_init))
            self.w_anomaly = nn.Parameter(torch.tensor(w_anomaly_init))
            self.bias = nn.Parameter(torch.tensor(bias_init))
        else:
            self.register_buffer("w_novelty", torch.tensor(w_novelty_init))
            self.register_buffer("w_entropy", torch.tensor(w_entropy_init))
            self.register_buffer("w_anomaly", torch.tensor(w_anomaly_init))
            self.register_buffer("bias", torch.tensor(bias_init))

    # ---- forward -----------------------------------------------------------

    def forward(
        self,
        novelty: torch.Tensor,
        entropy: torch.Tensor,
        anomaly: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ACh from novelty, entropy and anomaly signals.

        Args:
            novelty: Shape ``(B,)`` -- how novel the current input is.
            entropy: Shape ``(B,)`` -- prediction entropy / uncertainty.
            anomaly: Shape ``(B,)`` -- HTM anomaly score (0 = expected).

        Returns:
            ach: Acetylcholine signal in ``[0, 1]``, shape ``(B,)``.
        """
        with torch.cuda.amp.autocast(enabled=False):
            novelty = novelty.float()
            entropy = entropy.float()
            anomaly = anomaly.float()
            logit = (
                self.w_novelty * novelty
                + self.w_entropy * entropy
                + self.w_anomaly * anomaly
                + self.bias
            )
            ach = torch.sigmoid(logit)
        return ach


class NorepinephrineModule(nn.Module):
    """Computes the norepinephrine (NE) modulatory signal.

    NE encodes urgency and surprise, modulating arousal and
    exploration.  A temporal smoothing filter prevents abrupt
    oscillations.

    .. math::

        \\text{NE}_{raw} = \\sigma(w_u \\cdot u + w_s \\cdot s + b)

        \\text{NE} = (1 - \\beta) \\, \\text{NE}_{prev} + \\beta \\, \\text{NE}_{raw}

    Args:
        w_urgency_init: Initial weight for the urgency signal.
        w_surprise_init: Initial weight for the surprise signal.
        bias_init: Initial bias.
        beta: Temporal smoothing coefficient.
        learnable: If ``True``, weights/bias are ``nn.Parameter``.
    """

    def __init__(
        self,
        w_urgency_init: float = 1.0,
        w_surprise_init: float = 0.5,
        bias_init: float = 0.0,
        beta: float = 0.3,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        self.beta = beta
        if learnable:
            self.w_urgency = nn.Parameter(torch.tensor(w_urgency_init))
            self.w_surprise = nn.Parameter(torch.tensor(w_surprise_init))
            self.bias = nn.Parameter(torch.tensor(bias_init))
        else:
            self.register_buffer("w_urgency", torch.tensor(w_urgency_init))
            self.register_buffer("w_surprise", torch.tensor(w_surprise_init))
            self.register_buffer("bias", torch.tensor(bias_init))

    # ---- forward -----------------------------------------------------------

    def forward(
        self,
        urgency: torch.Tensor,
        surprise: torch.Tensor,
        ne_prev: torch.Tensor,
    ) -> torch.Tensor:
        """Compute NE with temporal smoothing.

        Args:
            urgency: Shape ``(B,)`` -- urgency magnitude.
            surprise: Shape ``(B,)`` -- surprise magnitude.
            ne_prev: Shape ``(B,)`` -- NE value from the previous step.

        Returns:
            ne: Smoothed NE signal in ``[0, 1]``, shape ``(B,)``.
        """
        with torch.cuda.amp.autocast(enabled=False):
            urgency = urgency.float()
            surprise = surprise.float()
            ne_prev = ne_prev.float()
            logit = (
                self.w_urgency * urgency
                + self.w_surprise * surprise
                + self.bias
            )
            ne_raw = torch.sigmoid(logit)
            ne = (1.0 - self.beta) * ne_prev + self.beta * ne_raw
        return ne


class SerotoninModule(nn.Module):
    """Computes the serotonin (5-HT) modulatory signal.

    5-HT encodes patience and long-horizon value, biasing the
    system toward exploitation (high 5-HT) or exploration (low
    5-HT).  It also modulates the effective eligibility window:
    high 5-HT lengthens the effective trace decay.

    .. math::

        5\\text{HT} = \\sigma(w_p \\cdot p + w_h \\cdot h + b)

    Args:
        w_patience_init: Initial weight for the patience signal.
        w_horizon_init: Initial weight for the horizon value signal.
        bias_init: Initial bias.
        learnable: If ``True``, weights/bias are ``nn.Parameter``.
    """

    def __init__(
        self,
        w_patience_init: float = 1.0,
        w_horizon_init: float = 0.5,
        bias_init: float = 0.0,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        if learnable:
            self.w_patience = nn.Parameter(torch.tensor(w_patience_init))
            self.w_horizon = nn.Parameter(torch.tensor(w_horizon_init))
            self.bias = nn.Parameter(torch.tensor(bias_init))
        else:
            self.register_buffer("w_patience", torch.tensor(w_patience_init))
            self.register_buffer("w_horizon", torch.tensor(w_horizon_init))
            self.register_buffer("bias", torch.tensor(bias_init))

    # ---- forward -----------------------------------------------------------

    def forward(
        self,
        patience: torch.Tensor,
        horizon_value: torch.Tensor,
    ) -> torch.Tensor:
        """Compute 5-HT from patience and horizon value.

        Args:
            patience: Shape ``(B,)`` -- how patient the agent is.
            horizon_value: Shape ``(B,)`` -- expected long-horizon value.

        Returns:
            sht: Serotonin signal in ``[0, 1]``, shape ``(B,)``.
        """
        with torch.cuda.amp.autocast(enabled=False):
            patience = patience.float()
            horizon_value = horizon_value.float()
            logit = (
                self.w_patience * patience
                + self.w_horizon * horizon_value
                + self.bias
            )
            sht = torch.sigmoid(logit)
        return sht


# ===================================================================
# Section 6 -- Combination Functions
#
# The combination function reduces four modulator scalars (DA, ACh,
# NE, 5-HT) into a single global_plasticity gain.  Three strategies
# are provided:
#
#   WEIGHTED_SUM    -- Linear blend; transparent, easy to interpret.
#                      Good default for most pipelines.
#
#   GATED_PRODUCT   -- DA gates everything.  When DA is zero,
#                      global_plasticity is zero regardless of the
#                      other three modulators.  Biologically,
#                      dopamine is the primary "go" signal.
#
#   MLP             -- Learned nonlinear mapping.  Most flexible
#                      but less interpretable.  Initialised near
#                      zero so the gate starts neutral.
#
# All three produce outputs bounded to [-1, 1].
# ===================================================================


class WeightedSumCombination(nn.Module):
    """Weighted linear combination of four modulators.

    .. math::

        g = \\text{clamp}(w_{da} \\cdot DA + w_{ach} \\cdot ACh
                          + w_{ne} \\cdot NE + w_{sht} \\cdot 5HT, \\; -1, 1)

    Args:
        w_da: Weight for DA.
        w_ach: Weight for ACh.
        w_ne: Weight for NE.
        w_sht: Weight for 5-HT.
    """

    def __init__(
        self,
        w_da: float = 0.4,
        w_ach: float = 0.25,
        w_ne: float = 0.2,
        w_sht: float = 0.15,
    ) -> None:
        super().__init__()
        self.register_buffer("w_da", torch.tensor(w_da))
        self.register_buffer("w_ach", torch.tensor(w_ach))
        self.register_buffer("w_ne", torch.tensor(w_ne))
        self.register_buffer("w_sht", torch.tensor(w_sht))

    def forward(
        self,
        da: torch.Tensor,
        ach: torch.Tensor,
        ne: torch.Tensor,
        sht: torch.Tensor,
    ) -> torch.Tensor:
        """Return the weighted sum, clamped to ``[-1, 1]``."""
        g = (
            self.w_da * da
            + self.w_ach * ach
            + self.w_ne * ne
            + self.w_sht * sht
        )
        return torch.clamp(g, -1.0, 1.0)


class GatedProductCombination(nn.Module):
    """DA-gated product combination.

    DA gates the contribution of the other modulators:

    .. math::

        g = \\text{clamp}\\bigl(
            DA \\cdot (w_{ach} \\cdot ACh + (1 - w_{ach}) \\cdot b_{ach}), \\; -1, 1
        \\bigr)

    where ``b_ach`` is a constant baseline.  When ``DA = 0``, ``g = 0``
    regardless of the other modulator values.  NE and 5-HT contribute
    as additive bias (scaled to small magnitude) so the signal is not
    entirely zero-information when DA is small.

    Args:
        w_ach: Mixing weight for ACh (vs. baseline).
        baseline_ach: Fallback ACh level when DA is present but ACh
            is missing.
        ne_scale: Small additive contribution of NE.
        sht_scale: Small additive contribution of 5-HT.
    """

    def __init__(
        self,
        w_ach: float = 0.5,
        baseline_ach: float = 0.5,
        ne_scale: float = 0.1,
        sht_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.register_buffer("w_ach", torch.tensor(w_ach))
        self.register_buffer("baseline_ach", torch.tensor(baseline_ach))
        self.register_buffer("ne_scale", torch.tensor(ne_scale))
        self.register_buffer("sht_scale", torch.tensor(sht_scale))

    def forward(
        self,
        da: torch.Tensor,
        ach: torch.Tensor,
        ne: torch.Tensor,
        sht: torch.Tensor,
    ) -> torch.Tensor:
        """Return the gated product, clamped to ``[-1, 1]``."""
        ach_blend = self.w_ach * ach + (1.0 - self.w_ach) * self.baseline_ach
        additive = self.ne_scale * ne + self.sht_scale * sht
        g = da * ach_blend + additive * da.abs()
        return torch.clamp(g, -1.0, 1.0)


class MLPCombination(nn.Module):
    """Learned MLP combination of four modulators.

    A small two-layer MLP maps ``[DA, ACh, NE, 5HT]`` to a scalar
    plasticity gain with ``tanh`` output activation.

    Args:
        hidden_dim: Width of the hidden layer.
    """

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),
        )
        # Initialise last layer near zero so that the gate starts neutral
        nn.init.zeros_(self.net[-2].bias)  # type: ignore[index]
        nn.init.normal_(self.net[-2].weight, std=0.01)  # type: ignore[index]

    def forward(
        self,
        da: torch.Tensor,
        ach: torch.Tensor,
        ne: torch.Tensor,
        sht: torch.Tensor,
    ) -> torch.Tensor:
        """Return the MLP output, inherently in ``[-1, 1]`` via tanh."""
        x = torch.stack([da, ach, ne, sht], dim=-1)  # (B, 4)
        return self.net(x).squeeze(-1)  # (B,)


# ===================================================================
# Section 7 -- Utility: Input Signal Preparation
# ===================================================================

# Canonical signal keys expected by the gate.
_SIGNAL_KEYS: List[str] = [
    "reward",
    "novelty",
    "entropy",
    "anomaly",
    "urgency",
    "surprise",
    "patience",
    "horizon_value",
]


def _prepare_signal(
    signals: Dict[str, torch.Tensor],
    key: str,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Extract a signal from the dict, defaulting to zero if absent.

    Handles:
    - Missing keys -> zero tensor.
    - Scalar tensors -> expanded to ``(batch_size,)``.
    - ``(B, 1)`` tensors -> squeezed to ``(B,)``.
    - Already ``(B,)`` -> returned as-is.

    Returns:
        Tensor of shape ``(B,)``, dtype ``float32``.
    """
    val = signals.get(key, None)
    if val is None:
        return torch.zeros(batch_size, device=device, dtype=torch.float32)

    val = val.float().to(device)

    if val.dim() == 0:
        val = val.unsqueeze(0).expand(batch_size)
    elif val.dim() == 2 and val.shape[-1] == 1:
        val = val.squeeze(-1)
    elif val.dim() == 1 and val.shape[0] == 1 and batch_size > 1:
        val = val.expand(batch_size)

    if val.shape[0] != batch_size:
        raise ValueError(
            f"Signal '{key}' has batch dim {val.shape[0]} but expected {batch_size}"
        )
    return val


def _infer_batch_size(signals: Dict[str, torch.Tensor]) -> int:
    """Infer the batch size from the first non-scalar signal.

    Falls back to 1 if all signals are scalar or the dict is empty.
    """
    for val in signals.values():
        if val is not None:
            if val.dim() == 0:
                continue
            if val.dim() == 2:
                return val.shape[0]
            if val.dim() == 1:
                return val.shape[0]
    return 1


def _infer_device(signals: Dict[str, torch.Tensor]) -> torch.device:
    """Infer the device from the first available signal tensor."""
    for val in signals.values():
        if val is not None:
            return val.device
    return torch.device("cpu")


# ===================================================================
# Section 8 -- NeuromodulatoryGate (main module)
# ===================================================================


class NeuromodulatoryGate(nn.Module):
    """Main module that computes DA, ACh, NE, 5-HT from observable signals
    and combines them into a single ``global_plasticity`` gain.

    This is the template version of the neuromodulatory gate designed to
    operate on explicit signal dictionaries rather than learned workspace
    representations.  It is intended as the building block for three-factor
    learning.

    Args:
        config: A :class:`NeuromodConfig` instance.  Defaults to
            :meth:`NeuromodConfig.minimal`.

    Example::

        gate = NeuromodulatoryGate(NeuromodConfig())
        signals = {
            "reward": torch.tensor([1.0]),
            "novelty": torch.tensor([0.5]),
        }
        output, state = gate(signals)
        print(output.da, output.global_plasticity)
    """

    def __init__(self, config: Optional[NeuromodConfig] = None) -> None:
        super().__init__()
        self.config = config or NeuromodConfig.minimal()

        # ---- individual modulators -----------------------------------------
        self.da_module = DopamineModule(
            w_reward_init=self.config.w_da_init,
            alpha=self.config.da_ema_alpha,
            learnable=self.config.learnable_weights,
        )
        self.ach_module = AcetylcholineModule(
            w_novelty_init=self.config.w_ach_novelty_init,
            w_entropy_init=self.config.w_ach_entropy_init,
            w_anomaly_init=self.config.w_ach_anomaly_init,
            bias_init=self.config.ach_bias_init,
            learnable=self.config.learnable_weights,
        )
        self.ne_module = NorepinephrineModule(
            w_urgency_init=self.config.w_ne_urgency_init,
            w_surprise_init=self.config.w_ne_surprise_init,
            bias_init=self.config.ne_bias_init,
            beta=self.config.ne_smoothing_beta,
            learnable=self.config.learnable_weights,
        )
        self.sht_module = SerotoninModule(
            w_patience_init=self.config.w_sht_patience_init,
            w_horizon_init=self.config.w_sht_horizon_init,
            bias_init=self.config.sht_bias_init,
            learnable=self.config.learnable_weights,
        )

        # ---- combination function ------------------------------------------
        combo = CombinationFn(self.config.combination_fn)
        if combo == CombinationFn.WEIGHTED_SUM:
            self.combiner: nn.Module = WeightedSumCombination(
                w_da=self.config.ws_da,
                w_ach=self.config.ws_ach,
                w_ne=self.config.ws_ne,
                w_sht=self.config.ws_sht,
            )
        elif combo == CombinationFn.GATED_PRODUCT:
            self.combiner = GatedProductCombination(
                w_ach=self.config.gp_ach_weight,
                baseline_ach=self.config.gp_baseline_ach,
            )
        elif combo == CombinationFn.MLP:
            self.combiner = MLPCombination(
                hidden_dim=self.config.modulator_hidden_dim,
            )
        else:
            raise ValueError(f"Unknown combination_fn: {self.config.combination_fn}")

        # ---- optional signal normaliser ------------------------------------
        if self.config.normalise_inputs:
            self.normaliser: Optional[RunningNormalizer] = RunningNormalizer(
                num_signals=len(_SIGNAL_KEYS),
                momentum=self.config.normalise_momentum,
                clamp_range=self.config.clamp_input_range,
            )
        else:
            self.normaliser = None

        logger.info(
            "NeuromodulatoryGate initialised: combination_fn=%s, "
            "learnable=%s, normalise=%s",
            self.config.combination_fn,
            self.config.learnable_weights,
            self.config.normalise_inputs,
        )

    # ---- state management --------------------------------------------------

    def reset_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> ModulatorState:
        """Create a fresh :class:`ModulatorState` for the given batch.

        Call this at episode boundaries or when starting a new sequence.

        Args:
            batch_size: Number of independent items in the batch.
            device: Target device for all state tensors.

        Returns:
            A zeroed-out :class:`ModulatorState`.
        """
        return ModulatorState(
            reward_baseline=torch.zeros(batch_size, device=device, dtype=torch.float32),
            novelty_history=torch.zeros(
                self.config.novelty_history_len,
                device=device,
                dtype=torch.float32,
            ),
            arousal_state=torch.full(
                (batch_size,), 0.5, device=device, dtype=torch.float32
            ),
            step_count=0,
        )

    # ---- forward -----------------------------------------------------------

    def forward(
        self,
        signals: Dict[str, torch.Tensor],
        state: Optional[ModulatorState] = None,
    ) -> Tuple[ModulatorOutput, ModulatorState]:
        """Compute all four modulators and the combined plasticity gain.

        Args:
            signals: Dictionary of observable signals.  Expected keys
                (all optional, missing ones default to zero):

                - ``"reward"``       : ``(B,)`` or ``(B,1)``
                - ``"novelty"``      : ``(B,)``
                - ``"entropy"``      : ``(B,)``
                - ``"anomaly"``      : ``(B,)``
                - ``"urgency"``      : ``(B,)``
                - ``"surprise"``     : ``(B,)``
                - ``"patience"``     : ``(B,)``
                - ``"horizon_value"`` : ``(B,)``

            state: Mutable state from a previous call.  If ``None``,
                a fresh state is created.

        Returns:
            output: :class:`ModulatorOutput` containing the four modulator
                values, the ``global_plasticity`` gain, and metadata.
            state: Updated :class:`ModulatorState`.
        """
        # -- infer batch / device -------------------------------------------
        batch_size = _infer_batch_size(signals)
        device = _infer_device(signals)

        # -- ensure state exists --------------------------------------------
        if state is None:
            state = self.reset_state(batch_size, device)
        elif state.reward_baseline.shape[0] != batch_size:
            logger.warning(
                "Batch size changed (%d -> %d); resetting modulator state.",
                state.reward_baseline.shape[0],
                batch_size,
            )
            state = self.reset_state(batch_size, device)

        # Move state tensors to the correct device if needed
        if state.reward_baseline.device != device:
            state = ModulatorState(
                reward_baseline=state.reward_baseline.to(device),
                novelty_history=state.novelty_history.to(device),
                arousal_state=state.arousal_state.to(device),
                step_count=state.step_count,
            )

        # -- extract and normalise signals ----------------------------------
        raw: Dict[str, torch.Tensor] = {}
        for key in _SIGNAL_KEYS:
            raw[key] = _prepare_signal(signals, key, batch_size, device)

        # Clamp raw inputs to prevent extreme values
        lo, hi = self.config.clamp_input_range
        for key in raw:
            raw[key] = torch.clamp(raw[key], lo, hi)

        if self.normaliser is not None:
            stacked = torch.stack([raw[k] for k in _SIGNAL_KEYS], dim=-1)
            stacked = self.normaliser(stacked)
            for idx, key in enumerate(_SIGNAL_KEYS):
                raw[key] = stacked[:, idx]

        # -- compute individual modulators ----------------------------------
        with torch.cuda.amp.autocast(enabled=False):
            # Dopamine
            da, new_baseline = self.da_module(
                reward=raw["reward"],
                baseline=state.reward_baseline,
            )

            # Acetylcholine
            ach = self.ach_module(
                novelty=raw["novelty"],
                entropy=raw["entropy"],
                anomaly=raw["anomaly"],
            )

            # Norepinephrine (uses previous arousal for smoothing)
            ne = self.ne_module(
                urgency=raw["urgency"],
                surprise=raw["surprise"],
                ne_prev=state.arousal_state,
            )

            # Serotonin
            sht = self.sht_module(
                patience=raw["patience"],
                horizon_value=raw["horizon_value"],
            )

            # -- combine into global plasticity ------------------------------
            global_plasticity = self.combiner(da, ach, ne, sht)

        # -- update state ---------------------------------------------------
        # Update novelty history ring buffer
        hist_idx = state.step_count % self.config.novelty_history_len
        new_novelty_history = state.novelty_history.clone()
        new_novelty_history[hist_idx] = raw["novelty"].mean().detach()

        new_state = ModulatorState(
            reward_baseline=new_baseline.detach(),
            novelty_history=new_novelty_history,
            arousal_state=ne.detach(),
            step_count=state.step_count + 1,
        )

        # -- build output ---------------------------------------------------
        output = ModulatorOutput(
            da=da,
            ach=ach,
            ne=ne,
            sht=sht,
            global_plasticity=global_plasticity,
            metadata={
                "step": new_state.step_count,
                "reward_baseline_mean": new_baseline.mean().item(),
            },
        )

        return output, new_state

    # ---- diagnostics -------------------------------------------------------

    @staticmethod
    def get_modulator_stats(
        output: ModulatorOutput,
    ) -> Dict[str, Dict[str, float]]:
        """Compute per-modulator summary statistics.

        Returns a nested dictionary keyed by modulator name, each
        containing ``mean``, ``std``, ``min``, ``max`` and
        ``saturation_rate`` (fraction of values within 0.05 of the
        output bounds).

        Args:
            output: A :class:`ModulatorOutput` from a forward pass.

        Returns:
            Nested dict of statistics.
        """

        def _stats(
            tensor: torch.Tensor,
            lo: float,
            hi: float,
        ) -> Dict[str, float]:
            t = tensor.detach().float()
            near_lo = (t <= lo + 0.05).float().mean().item()
            near_hi = (t >= hi - 0.05).float().mean().item()
            return {
                "mean": t.mean().item(),
                "std": t.std().item() if t.numel() > 1 else 0.0,
                "min": t.min().item(),
                "max": t.max().item(),
                "saturation_rate": near_lo + near_hi,
            }

        return {
            ModulatorType.DA.value: _stats(output.da, -1.0, 1.0),
            ModulatorType.ACH.value: _stats(output.ach, 0.0, 1.0),
            ModulatorType.NE.value: _stats(output.ne, 0.0, 1.0),
            ModulatorType.SHT.value: _stats(output.sht, 0.0, 1.0),
            "global_plasticity": _stats(output.global_plasticity, -1.0, 1.0),
        }


# ===================================================================
# Section 9 -- Factory Helpers
# ===================================================================


def create_neuromodulatory_gate_template(
    combination_fn: str = "weighted_sum",
    learnable: bool = True,
    normalise: bool = False,
    **overrides: Any,
) -> NeuromodulatoryGate:
    """Convenience factory for :class:`NeuromodulatoryGate`.

    Args:
        combination_fn: One of ``"weighted_sum"``, ``"gated_product"``,
            ``"mlp"``.
        learnable: Whether modulator weights are trainable parameters.
        normalise: Whether to apply running-statistics normalisation.
        **overrides: Extra fields forwarded to :class:`NeuromodConfig`.

    Returns:
        A configured :class:`NeuromodulatoryGate` instance.
    """
    config = NeuromodConfig(
        combination_fn=combination_fn,
        learnable_weights=learnable,
        normalise_inputs=normalise,
        **overrides,
    )
    return NeuromodulatoryGate(config)


# ===================================================================
# Section 10 -- Utility Helpers for Downstream Integration
# ===================================================================


class ModulatorHistory:
    """Accumulates modulator values over multiple forward passes for
    diagnostics and logging.

    The history stores per-step modulator values in a ring buffer.
    When the buffer is full, the oldest entries are overwritten.

    Args:
        max_steps: Maximum number of steps to retain.

    Example::

        history = ModulatorHistory(max_steps=200)
        for step in range(1000):
            output, state = gate(signals, state)
            history.record(output)
        stats = history.summary()
    """

    def __init__(self, max_steps: int = 500) -> None:
        self.max_steps = max_steps
        self._da: List[float] = []
        self._ach: List[float] = []
        self._ne: List[float] = []
        self._sht: List[float] = []
        self._gp: List[float] = []

    def record(self, output: ModulatorOutput) -> None:
        """Record mean values from a single forward-pass output.

        Args:
            output: :class:`ModulatorOutput` from the gate.
        """
        self._da.append(output.da.detach().mean().item())
        self._ach.append(output.ach.detach().mean().item())
        self._ne.append(output.ne.detach().mean().item())
        self._sht.append(output.sht.detach().mean().item())
        self._gp.append(output.global_plasticity.detach().mean().item())
        # Trim to ring buffer length
        if len(self._da) > self.max_steps:
            self._da = self._da[-self.max_steps:]
            self._ach = self._ach[-self.max_steps:]
            self._ne = self._ne[-self.max_steps:]
            self._sht = self._sht[-self.max_steps:]
            self._gp = self._gp[-self.max_steps:]

    @property
    def length(self) -> int:
        """Number of recorded steps."""
        return len(self._da)

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Compute summary statistics over the recorded history.

        Returns:
            Nested dict with keys ``da``, ``ach``, ``ne``, ``sht``,
            ``global_plasticity``, each containing ``mean``, ``std``,
            ``min``, ``max``.
        """
        def _summarise(values: List[float]) -> Dict[str, float]:
            if not values:
                return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
            t = torch.tensor(values)
            return {
                "mean": t.mean().item(),
                "std": t.std().item() if len(values) > 1 else 0.0,
                "min": t.min().item(),
                "max": t.max().item(),
            }

        return {
            "da": _summarise(self._da),
            "ach": _summarise(self._ach),
            "ne": _summarise(self._ne),
            "sht": _summarise(self._sht),
            "global_plasticity": _summarise(self._gp),
        }

    def clear(self) -> None:
        """Discard all recorded history."""
        self._da.clear()
        self._ach.clear()
        self._ne.clear()
        self._sht.clear()
        self._gp.clear()

    def to_dict(self) -> Dict[str, List[float]]:
        """Serialise raw history arrays to a plain dictionary.

        Useful for JSON logging or TensorBoard custom scalars.
        """
        return {
            "da": list(self._da),
            "ach": list(self._ach),
            "ne": list(self._ne),
            "sht": list(self._sht),
            "global_plasticity": list(self._gp),
        }


def compute_effective_eligibility_decay(
    base_tau_e: float,
    sht: torch.Tensor,
    scale_factor: float = 2.0,
) -> torch.Tensor:
    """Compute a 5-HT-modulated eligibility decay constant.

    High serotonin (patience) extends the effective eligibility window
    by multiplying the base time constant:

    .. math::

        \\tau_{eff} = \\tau_e \\cdot (1 + \\text{scale} \\cdot 5HT)

    The returned value is the per-element effective ``tau_e`` which can
    be used to compute a per-sample decay factor.

    Args:
        base_tau_e: The baseline eligibility time constant.
        sht: Serotonin signal, shape ``(B,)`` in ``[0, 1]``.
        scale_factor: How strongly 5-HT modulates the time constant.
            A value of 2.0 means that at ``5HT = 1`` the effective
            time constant is tripled (``tau_e * 3``).

    Returns:
        Effective tau_e per batch item, shape ``(B,)``, all positive.

    Example::

        sht = torch.tensor([0.0, 0.5, 1.0])
        tau_eff = compute_effective_eligibility_decay(20.0, sht)
        # tau_eff: tensor([20., 40., 60.])
    """
    sht = sht.float().clamp(0.0, 1.0)
    tau_eff = base_tau_e * (1.0 + scale_factor * sht)
    return tau_eff


def modulator_output_to_dict(
    output: ModulatorOutput,
) -> Dict[str, torch.Tensor]:
    """Convert a :class:`ModulatorOutput` to a flat tensor dictionary.

    This is useful for broadcasting modulator values to consuming
    modules that expect a ``Dict[str, Tensor]`` interface (e.g.
    the integration hooks for workspace, reasoning, and HTM).

    Args:
        output: The modulator output dataclass.

    Returns:
        Dictionary with string keys mapped to tensor values.

    Example::

        output, state = gate(signals)
        mod_dict = modulator_output_to_dict(output)
        workspace.set_ach_signal(mod_dict["acetylcholine"])
    """
    return {
        ModulatorType.DA.value: output.da,
        ModulatorType.ACH.value: output.ach,
        ModulatorType.NE.value: output.ne,
        ModulatorType.SHT.value: output.sht,
        "global_plasticity": output.global_plasticity,
    }


def clamp_and_log_signals(
    signals: Dict[str, torch.Tensor],
    clamp_range: Tuple[float, float] = (-10.0, 10.0),
    tag: str = "neuromod",
) -> Dict[str, torch.Tensor]:
    """Clamp every signal tensor and log warnings for extreme values.

    This is a defensive preprocessing step that prevents numerical
    explosion from malformed upstream signals.  It logs a warning
    (once per key per call) if any value is clipped.

    Args:
        signals: Raw signal dictionary.
        clamp_range: ``(lo, hi)`` clamp bounds.
        tag: Prefix for log messages.

    Returns:
        New dictionary with clamped tensors (original dict is not
        mutated).
    """
    lo, hi = clamp_range
    clamped: Dict[str, torch.Tensor] = {}
    for key, val in signals.items():
        if val is None:
            continue
        if val.numel() > 0:
            was_clipped = bool((val < lo).any() or (val > hi).any())
            if was_clipped:
                logger.warning(
                    "[%s] Signal '%s' had values outside [%.1f, %.1f]; "
                    "clamped (min=%.3f, max=%.3f).",
                    tag, key, lo, hi,
                    val.min().item(), val.max().item(),
                )
        clamped[key] = torch.clamp(val, lo, hi)
    return clamped


def compute_modulator_agreement(output: ModulatorOutput) -> float:
    """Compute a scalar measure of how much the modulators agree.

    Agreement is defined as the negative pairwise variance of the
    four modulator mean values.  When all modulators are at similar
    levels, agreement is high (close to 0).  When they diverge,
    agreement is more negative.

    This can be used as a diagnostic to detect conflicting
    neuromodulatory signals (e.g. high DA but low ACh).

    Args:
        output: Modulator output from a forward pass.

    Returns:
        Agreement score (higher = more agreement, max 0.0).
    """
    values = torch.tensor([
        output.da.detach().mean().item(),
        output.ach.detach().mean().item(),
        output.ne.detach().mean().item(),
        output.sht.detach().mean().item(),
    ])
    return -values.var().item()


# ===================================================================
# Section 11 -- Self-Test Block
# ===================================================================

def _run_self_tests() -> None:  # noqa: C901 -- complexity is intentional
    """Execute 22+ self-tests and print PASS/FAIL for each.

    Designed to be invoked as ``python neuromodulation_template.py``.
    """

    results: List[Tuple[str, bool, str]] = []

    def _test(name: str, condition: bool, detail: str = "") -> None:
        results.append((name, condition, detail))
        tag = "PASS" if condition else "FAIL"
        msg = f"  [{tag}] {name}"
        if detail:
            msg += f"  ({detail})"
        print(msg)

    print("=" * 72)
    print("NeuromodulatoryGate -- Self-Test Suite")
    print("=" * 72)

    torch.manual_seed(42)
    B = 4

    # ------------------------------------------------------------------
    # 1. DA: positive reward -> positive DA
    # ------------------------------------------------------------------
    da_mod = DopamineModule(w_reward_init=1.0, alpha=0.1, learnable=False)
    reward = torch.ones(B)
    baseline = torch.zeros(B)
    da_val, _ = da_mod(reward, baseline)
    _test(
        "DA: positive reward -> positive DA",
        bool(torch.all(da_val > 0).item()),
        f"DA range: [{da_val.min().item():.4f}, {da_val.max().item():.4f}]",
    )

    # ------------------------------------------------------------------
    # 2. DA: negative reward -> negative DA
    # ------------------------------------------------------------------
    reward_neg = -torch.ones(B)
    da_neg, _ = da_mod(reward_neg, baseline)
    _test(
        "DA: negative reward -> negative DA",
        bool(torch.all(da_neg < 0).item()),
        f"DA range: [{da_neg.min().item():.4f}, {da_neg.max().item():.4f}]",
    )

    # ------------------------------------------------------------------
    # 3. DA: zero reward after baseline -> near zero DA
    # ------------------------------------------------------------------
    # After baseline has tracked to some value, zero reward minus baseline
    # should be zero when baseline is also zero.
    reward_zero = torch.zeros(B)
    da_zero, _ = da_mod(reward_zero, baseline)
    _test(
        "DA: zero reward after baseline=0 -> near zero DA",
        bool(torch.allclose(da_zero, torch.zeros(B), atol=1e-6)),
        f"DA values: {da_zero.tolist()}",
    )

    # ------------------------------------------------------------------
    # 4. DA baseline EMA tracking
    # ------------------------------------------------------------------
    bl = torch.zeros(B)
    baselines = [bl.mean().item()]
    for _ in range(20):
        _, bl = da_mod(torch.ones(B), bl)
        baselines.append(bl.mean().item())
    _test(
        "DA baseline: EMA tracks reward correctly",
        baselines[-1] > baselines[0] and baselines[-1] > 0.8,
        f"baseline[0]={baselines[0]:.3f}, baseline[-1]={baselines[-1]:.3f}",
    )

    # ------------------------------------------------------------------
    # 5. ACh: high novelty -> high ACh
    # ------------------------------------------------------------------
    ach_mod = AcetylcholineModule(learnable=False)
    ach_val = ach_mod(
        novelty=5.0 * torch.ones(B),
        entropy=torch.zeros(B),
        anomaly=torch.zeros(B),
    )
    _test(
        "ACh: high novelty -> high ACh",
        bool(torch.all(ach_val > 0.8).item()),
        f"ACh={ach_val.mean().item():.4f}",
    )

    # ------------------------------------------------------------------
    # 6. ACh: zero signals -> moderate ACh near sigmoid(0)
    # ------------------------------------------------------------------
    ach_zero = ach_mod(
        novelty=torch.zeros(B),
        entropy=torch.zeros(B),
        anomaly=torch.zeros(B),
    )
    sigmoid_0 = torch.sigmoid(torch.tensor(0.0)).item()
    _test(
        "ACh: zero signals -> moderate ACh near sigmoid(0)",
        bool(torch.allclose(ach_zero, torch.full((B,), sigmoid_0), atol=1e-4)),
        f"ACh={ach_zero.mean().item():.4f}, sigmoid(0)={sigmoid_0:.4f}",
    )

    # ------------------------------------------------------------------
    # 7. NE: high urgency -> high NE
    # ------------------------------------------------------------------
    ne_mod = NorepinephrineModule(beta=1.0, learnable=False)  # beta=1 -> no smoothing
    ne_val = ne_mod(
        urgency=5.0 * torch.ones(B),
        surprise=torch.zeros(B),
        ne_prev=torch.zeros(B),
    )
    _test(
        "NE: high urgency -> high NE",
        bool(torch.all(ne_val > 0.8).item()),
        f"NE={ne_val.mean().item():.4f}",
    )

    # ------------------------------------------------------------------
    # 8. NE: temporal smoothing works (NE changes gradually)
    # ------------------------------------------------------------------
    ne_mod_smooth = NorepinephrineModule(beta=0.3, learnable=False)
    ne_prev = torch.zeros(B)
    ne_step1 = ne_mod_smooth(
        urgency=5.0 * torch.ones(B),
        surprise=torch.zeros(B),
        ne_prev=ne_prev,
    )
    ne_raw_expected = torch.sigmoid(torch.tensor(5.0))
    # With beta=0.3: NE = 0.7 * 0 + 0.3 * sigmoid(5) < sigmoid(5)
    _test(
        "NE: temporal smoothing (NE < raw sigmoid)",
        bool(torch.all(ne_step1 < ne_raw_expected).item()),
        f"smoothed={ne_step1.mean().item():.4f}, raw={ne_raw_expected.item():.4f}",
    )

    # ------------------------------------------------------------------
    # 9. 5-HT: high patience -> high 5-HT
    # ------------------------------------------------------------------
    sht_mod = SerotoninModule(learnable=False)
    sht_val = sht_mod(
        patience=5.0 * torch.ones(B),
        horizon_value=torch.zeros(B),
    )
    _test(
        "5-HT: high patience -> high 5-HT",
        bool(torch.all(sht_val > 0.8).item()),
        f"5-HT={sht_val.mean().item():.4f}",
    )

    # ------------------------------------------------------------------
    # 10. All modulators: bounded output ranges
    # ------------------------------------------------------------------
    # Sweep extreme values through each modulator
    extremes = torch.linspace(-10, 10, 50)
    da_sweep, _ = da_mod(extremes, torch.zeros(50))
    ach_sweep = ach_mod(extremes, torch.zeros(50), torch.zeros(50))
    ne_sweep = ne_mod(extremes, torch.zeros(50), torch.zeros(50))
    sht_sweep = sht_mod(extremes, torch.zeros(50))

    da_bounded = bool((da_sweep >= -1.0).all() and (da_sweep <= 1.0).all())
    ach_bounded = bool((ach_sweep >= 0.0).all() and (ach_sweep <= 1.0).all())
    ne_bounded = bool((ne_sweep >= 0.0).all() and (ne_sweep <= 1.0).all())
    sht_bounded = bool((sht_sweep >= 0.0).all() and (sht_sweep <= 1.0).all())
    _test(
        "All modulators: bounded output ranges verified",
        da_bounded and ach_bounded and ne_bounded and sht_bounded,
        f"DA=[-1,1]:{da_bounded} ACh=[0,1]:{ach_bounded} "
        f"NE=[0,1]:{ne_bounded} 5-HT=[0,1]:{sht_bounded}",
    )

    # ------------------------------------------------------------------
    # 11. All zero signals: neutral outputs
    # ------------------------------------------------------------------
    gate = NeuromodulatoryGate(NeuromodConfig.minimal())
    zero_signals: Dict[str, torch.Tensor] = {
        k: torch.zeros(B) for k in _SIGNAL_KEYS
    }
    out_zero, st_zero = gate(zero_signals)
    da_neutral = torch.allclose(out_zero.da, torch.zeros(B), atol=1e-6)
    ach_neutral_val = torch.sigmoid(torch.tensor(0.0)).item()
    ach_neutral = torch.allclose(
        out_zero.ach, torch.full((B,), ach_neutral_val), atol=1e-4
    )
    _test(
        "All zero signals: neutral outputs (DA=0, ACh~sigmoid(0))",
        bool(da_neutral and ach_neutral),
        f"DA={out_zero.da.mean().item():.4f}, ACh={out_zero.ach.mean().item():.4f}",
    )

    # ------------------------------------------------------------------
    # 12. Weighted sum: correct weighted combination
    # ------------------------------------------------------------------
    ws = WeightedSumCombination(w_da=0.4, w_ach=0.25, w_ne=0.2, w_sht=0.15)
    da_t = torch.tensor([0.5])
    ach_t = torch.tensor([0.8])
    ne_t = torch.tensor([0.3])
    sht_t = torch.tensor([0.6])
    g_ws = ws(da_t, ach_t, ne_t, sht_t)
    expected_ws = 0.4 * 0.5 + 0.25 * 0.8 + 0.2 * 0.3 + 0.15 * 0.6
    _test(
        "Weighted sum: correct weighted combination",
        bool(torch.allclose(g_ws, torch.tensor([expected_ws]), atol=1e-5)),
        f"got={g_ws.item():.5f}, expected={expected_ws:.5f}",
    )

    # ------------------------------------------------------------------
    # 13. Gated product: DA=0 gates output to near zero
    # ------------------------------------------------------------------
    gp = GatedProductCombination()
    g_gp_zero = gp(
        da=torch.tensor([0.0]),
        ach=torch.tensor([1.0]),
        ne=torch.tensor([1.0]),
        sht=torch.tensor([1.0]),
    )
    _test(
        "Gated product: DA=0 gates output to near zero",
        bool(torch.allclose(g_gp_zero, torch.tensor([0.0]), atol=1e-6)),
        f"gated_product={g_gp_zero.item():.6f}",
    )

    # ------------------------------------------------------------------
    # 14. MLP combination: produces bounded output
    # ------------------------------------------------------------------
    torch.manual_seed(42)
    mlp_comb = MLPCombination(hidden_dim=32)
    g_mlp = mlp_comb(
        da=torch.randn(B),
        ach=torch.rand(B),
        ne=torch.rand(B),
        sht=torch.rand(B),
    )
    _test(
        "MLP combination: produces bounded output",
        bool((g_mlp >= -1.0).all() and (g_mlp <= 1.0).all() and torch.isfinite(g_mlp).all()),
        f"MLP range: [{g_mlp.min().item():.4f}, {g_mlp.max().item():.4f}]",
    )

    # ------------------------------------------------------------------
    # 15. NeuromodulatoryGate: end-to-end forward pass
    # ------------------------------------------------------------------
    torch.manual_seed(42)
    gate_e2e = NeuromodulatoryGate(NeuromodConfig())
    e2e_signals = {
        "reward": torch.randn(B),
        "novelty": torch.rand(B),
        "entropy": torch.rand(B),
        "anomaly": torch.rand(B),
        "urgency": torch.rand(B),
        "surprise": torch.rand(B),
        "patience": torch.rand(B),
        "horizon_value": torch.rand(B),
    }
    out_e2e, st_e2e = gate_e2e(e2e_signals)
    _test(
        "NeuromodulatoryGate: end-to-end forward pass",
        (
            out_e2e.da.shape == (B,)
            and out_e2e.ach.shape == (B,)
            and out_e2e.ne.shape == (B,)
            and out_e2e.sht.shape == (B,)
            and out_e2e.global_plasticity.shape == (B,)
        ),
        f"shapes: da={out_e2e.da.shape}, gp={out_e2e.global_plasticity.shape}",
    )

    # ------------------------------------------------------------------
    # 16. State update: baseline changes after reward
    # ------------------------------------------------------------------
    gate_state = NeuromodulatoryGate(NeuromodConfig.minimal())
    s0 = gate_state.reset_state(B, torch.device("cpu"))
    bl_before = s0.reward_baseline.clone()
    _, s1 = gate_state({"reward": torch.ones(B)}, state=s0)
    bl_after = s1.reward_baseline
    _test(
        "State update: baseline changes after reward",
        bool(torch.all(bl_after > bl_before).item()),
        f"before={bl_before.mean().item():.4f}, after={bl_after.mean().item():.4f}",
    )

    # ------------------------------------------------------------------
    # 17. State reset: clears to initial values
    # ------------------------------------------------------------------
    s_reset = gate_state.reset_state(B, torch.device("cpu"))
    _test(
        "State reset: clears to initial values",
        bool(
            torch.all(s_reset.reward_baseline == 0).item()
            and s_reset.step_count == 0
            and torch.all(s_reset.novelty_history == 0).item()
        ),
        f"baseline={s_reset.reward_baseline.mean().item():.4f}, step={s_reset.step_count}",
    )

    # ------------------------------------------------------------------
    # 18. Missing signals: handled gracefully (defaults to zero)
    # ------------------------------------------------------------------
    gate_missing = NeuromodulatoryGate(NeuromodConfig.minimal())
    # Only provide reward -- all others should default to zero
    out_miss, _ = gate_missing({"reward": torch.ones(B)})
    _test(
        "Missing signals: handled gracefully (defaults to zero)",
        (
            out_miss.da.shape == (B,)
            and out_miss.ach.shape == (B,)
            and torch.isfinite(out_miss.global_plasticity).all().item()
        ),
        f"gp={out_miss.global_plasticity.tolist()}",
    )

    # ------------------------------------------------------------------
    # 19. Determinism: same inputs + state -> same outputs
    # ------------------------------------------------------------------
    torch.manual_seed(123)
    gate_det = NeuromodulatoryGate(NeuromodConfig.minimal())
    det_signals = {"reward": torch.tensor([0.5, -0.3, 0.0, 1.0])}
    s_det = gate_det.reset_state(4, torch.device("cpu"))
    out_a, _ = gate_det(det_signals, state=s_det)
    out_b, _ = gate_det(det_signals, state=s_det)  # same state, same signals
    _test(
        "Determinism: same inputs + state -> same outputs",
        bool(
            torch.allclose(out_a.da, out_b.da, atol=1e-7)
            and torch.allclose(out_a.ach, out_b.ach, atol=1e-7)
            and torch.allclose(out_a.global_plasticity, out_b.global_plasticity, atol=1e-7)
        ),
        f"da_diff={torch.abs(out_a.da - out_b.da).max().item():.1e}",
    )

    # ------------------------------------------------------------------
    # 20. Modulator stats: contains expected keys
    # ------------------------------------------------------------------
    stats = NeuromodulatoryGate.get_modulator_stats(out_a)
    expected_keys = {"dopamine", "acetylcholine", "norepinephrine", "serotonin", "global_plasticity"}
    stat_fields = {"mean", "std", "min", "max", "saturation_rate"}
    has_keys = expected_keys.issubset(set(stats.keys()))
    has_fields = all(stat_fields.issubset(set(v.keys())) for v in stats.values())
    _test(
        "Modulator stats: contains expected keys",
        has_keys and has_fields,
        f"keys={set(stats.keys())}",
    )

    # ------------------------------------------------------------------
    # 21. Batch independence: different batch items independent
    # ------------------------------------------------------------------
    gate_bi = NeuromodulatoryGate(NeuromodConfig.minimal())
    bi_signals = {
        "reward": torch.tensor([1.0, 0.0, -1.0, 0.5]),
        "novelty": torch.tensor([0.0, 1.0, 0.0, 0.0]),
    }
    out_bi, _ = gate_bi(bi_signals)
    # Item 0 has reward=1 -> DA should be positive
    # Item 2 has reward=-1 -> DA should be negative
    # They should differ
    _test(
        "Batch independence: different batch items independent",
        bool(
            out_bi.da[0].item() > 0
            and out_bi.da[2].item() < 0
            and not torch.allclose(out_bi.da[0:1], out_bi.da[2:3])
        ),
        f"DA[0]={out_bi.da[0].item():.4f}, DA[2]={out_bi.da[2].item():.4f}",
    )

    # ------------------------------------------------------------------
    # 22. fp32: outputs are float32
    # ------------------------------------------------------------------
    _test(
        "fp32: outputs are float32",
        (
            out_bi.da.dtype == torch.float32
            and out_bi.ach.dtype == torch.float32
            and out_bi.ne.dtype == torch.float32
            and out_bi.sht.dtype == torch.float32
            and out_bi.global_plasticity.dtype == torch.float32
        ),
        f"dtypes: da={out_bi.da.dtype}, gp={out_bi.global_plasticity.dtype}",
    )

    # ------------------------------------------------------------------
    # 23. Config validation: invalid combination_fn raises ValueError
    # ------------------------------------------------------------------
    config_error = False
    try:
        NeuromodConfig(combination_fn="invalid")
    except ValueError:
        config_error = True
    _test(
        "Config validation: invalid combination_fn raises ValueError",
        config_error,
    )

    # ------------------------------------------------------------------
    # 24. Gated product with nonzero DA produces nonzero output
    # ------------------------------------------------------------------
    g_gp_nonzero = gp(
        da=torch.tensor([0.8]),
        ach=torch.tensor([0.7]),
        ne=torch.tensor([0.5]),
        sht=torch.tensor([0.3]),
    )
    _test(
        "Gated product: nonzero DA -> nonzero output",
        bool(g_gp_nonzero.abs().item() > 1e-6),
        f"gated_product={g_gp_nonzero.item():.6f}",
    )

    # ------------------------------------------------------------------
    # 25. State serialisation round-trip
    # ------------------------------------------------------------------
    s_orig = ModulatorState(
        reward_baseline=torch.tensor([0.3, 0.5, 0.1, 0.9]),
        novelty_history=torch.randn(100),
        arousal_state=torch.tensor([0.6, 0.4, 0.7, 0.2]),
        step_count=42,
    )
    d = s_orig.state_dict()
    s_loaded = ModulatorState.from_state_dict(d)
    _test(
        "State serialisation round-trip",
        bool(
            torch.allclose(s_orig.reward_baseline, s_loaded.reward_baseline)
            and torch.allclose(s_orig.novelty_history, s_loaded.novelty_history)
            and torch.allclose(s_orig.arousal_state, s_loaded.arousal_state)
            and s_orig.step_count == s_loaded.step_count
        ),
        f"step_count={s_loaded.step_count}",
    )

    # ------------------------------------------------------------------
    # 26. Running normaliser does not crash and preserves shape
    # ------------------------------------------------------------------
    rn = RunningNormalizer(num_signals=8)
    rn_input = torch.randn(B, 8)
    rn_output = rn(rn_input)
    _test(
        "RunningNormalizer: preserves shape and produces finite output",
        rn_output.shape == rn_input.shape and bool(torch.isfinite(rn_output).all()),
        f"shape={rn_output.shape}",
    )

    # ------------------------------------------------------------------
    # 27. Gate with normalisation enabled runs without error
    # ------------------------------------------------------------------
    gate_norm = NeuromodulatoryGate(NeuromodConfig(normalise_inputs=True))
    gate_norm.train()
    out_norm, _ = gate_norm(e2e_signals)
    _test(
        "Gate with normalisation: runs without error",
        bool(torch.isfinite(out_norm.global_plasticity).all()),
        f"gp={out_norm.global_plasticity.tolist()}",
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_pass = sum(1 for _, ok, _ in results if ok)
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print()
    print("=" * 72)
    print(f"Results: {n_pass} passed, {n_fail} failed, {len(results)} total")
    print("=" * 72)

    if n_fail > 0:
        print("\nFailed tests:")
        for name, ok, detail in results:
            if not ok:
                print(f"  - {name}: {detail}")
        sys.exit(1)
    else:
        print("\nAll tests passed.")


# ===================================================================
# Entry point
# ===================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    _run_self_tests()
