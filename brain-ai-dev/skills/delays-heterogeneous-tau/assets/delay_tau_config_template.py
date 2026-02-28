"""
brain_ai/core/delay_tau_config.py — Configuration for learnable delays and heterogeneous tau.

This module provides structured configuration dataclasses that replace the flat
boolean flags in SNNConfig for the delay and heterogeneous-tau features of
AdvancedLIFNeuron. It is designed to be drop-in compatible with the existing
SNNConfig while enabling fine-grained ablation studies.

Key design decisions:
  - DelayConfig encapsulates all DCLS-style learnable synaptic delay parameters.
  - TauConfig encapsulates all heterogeneous time-constant parameters.
  - AblationConfig bundles one of each for a single controlled experiment.
  - Preset factory functions produce the canonical 2x2 ablation matrix.
  - upgrade_snn_config() bridges old flat flags to new structured objects.
  - All configs are serializable to/from plain dicts for JSON experiment logging.

Integration path into brain_ai/config.py::

    from brain_ai.core.delay_tau_config import DelayConfig, TauConfig

    @dataclass
    class SNNConfig:
        ...
        delay: DelayConfig = field(default_factory=DelayConfig)
        tau: TauConfig = field(default_factory=TauConfig)

References:
    Hammouamri et al. (2024) "Learning Delays in Spiking Neural Networks
        Using Dilated Convolutions with Learnable Spacings (DCLS)."
    Perez-Nieves et al. (2021) "Neural heterogeneity promotes robust learning."
    Yin et al. (2021) "Accurate and efficient time-domain classification with
        adaptive spiking recurrent neural networks."
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Type, Union


# ---------------------------------------------------------------------------
# Section 1: DelayConfig
# ---------------------------------------------------------------------------

#: All legal delay modes.
_DELAY_MODES = ("off", "fixed_random", "learnable_dcls")

#: All legal granularity options for delays.
_DELAY_GRANULARITIES = ("per_synapse", "per_output", "per_input", "per_block")

#: All legal sigma schedules.
_SIGMA_SCHEDULES = ("constant", "decreasing")

#: All legal initialization strategies for delays.
_DELAY_INIT_STRATEGIES = ("uniform", "normal_center", "zeros")

#: All legal implementation backends.
_DELAY_IMPLEMENTATIONS = ("bin_accumulation", "ring_buffer")


@dataclass
class DelayConfig:
    """Configuration for DCLS-style learnable synaptic delays.

    Controls how the spike-history buffer in AdvancedLIFNeuron is parameterised,
    initialised, and used during training and evaluation.

    Attributes:
        enabled:
            Master switch.  When False the neuron skips all delay logic
            regardless of all other settings.  Maps to
            ``SNNConfig.use_learnable_delays``.
        mode:
            Delay mode:

            - ``"off"``           — No delay; equivalent to enabled=False.
            - ``"fixed_random"``  — Delays sampled once at init, not trained.
            - ``"learnable_dcls"``— DCLS soft-attention delays, gradient-trained.

        max_delay:
            Maximum delay depth in simulation timesteps (Td). Equivalent to
            ``SNNConfig.max_delay``.  Must be >= 1.
        num_bins:
            Number of Gaussian interpolation bins (K) used by DCLS to form
            the soft delay distribution.  Ignored when mode != "learnable_dcls".
        granularity:
            Level at which separate delay parameters are maintained:

            - ``"per_synapse"`` — one learnable value per (input, output) pair.
            - ``"per_output"``  — one value per output neuron  [default].
            - ``"per_input"``   — one value per input channel.
            - ``"per_block"``   — one value per block of ``block_size`` outputs.

        block_size:
            Number of output neurons per parameter block when
            ``granularity="per_block"``.  Ignored otherwise.
        sigma_start:
            Initial standard deviation of the Gaussian interpolation kernel.
            Larger values give wider, more exploratory initialisation.
        sigma_end:
            Target sigma at the end of ``sigma_decay_epochs``.  Must be > 0
            and <= sigma_start.
        sigma_decay_epochs:
            Number of training epochs over which sigma anneals from
            ``sigma_start`` to ``sigma_end``.  Irrelevant when
            ``sigma_schedule="constant"``.
        sigma_schedule:
            Annealing schedule for sigma:

            - ``"constant"``   — sigma stays at sigma_start.
            - ``"decreasing"`` — exponential decay from sigma_start to sigma_end.

        eval_discretize:
            When True, delay weights are rounded to the nearest integer bin
            at evaluation time (no Gaussian interpolation), producing crisper
            and faster inference.
        init_strategy:
            How initial delay values are drawn:

            - ``"uniform"``        — U[0, max_delay).
            - ``"normal_center"``  — N(max_delay/2, sigma_start).
            - ``"zeros"``          — All delays start at 0.

        init_seed:
            Optional random seed for reproducible initialisation.  None means
            use the global PyTorch RNG state.
        memory_warn_threshold:
            Element count above which a warning is emitted about potential
            VRAM pressure from the spike history buffer
            (B * max_delay * N_neurons elements).
        implementation:
            Backend used to compute the delayed input:

            - ``"bin_accumulation"`` — Iterate over K bins; differentiable.
            - ``"ring_buffer"``      — Pre-allocated circular buffer;
              faster but requires careful gradient handling.
    """

    # Master switch
    enabled: bool = True
    mode: str = "learnable_dcls"

    # Delay range
    max_delay: int = 16

    # Gaussian interpolation
    num_bins: int = 3

    # Granularity
    granularity: str = "per_output"
    block_size: int = 64

    # Sigma annealing
    sigma_start: float = 1.0
    sigma_end: float = 0.5
    sigma_decay_epochs: int = 50
    sigma_schedule: str = "decreasing"

    # Eval behaviour
    eval_discretize: bool = True

    # Initialisation
    init_strategy: str = "uniform"
    init_seed: Optional[int] = None

    # Memory safety
    memory_warn_threshold: int = 100_000_000

    # Backend
    implementation: str = "bin_accumulation"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Check all field constraints and raise ValueError on violation.

        Call this after constructing or deserializing a DelayConfig to catch
        configuration errors before they surface as cryptic runtime failures.

        Raises:
            ValueError: Describes the first constraint that is violated.
        """
        if self.mode not in _DELAY_MODES:
            raise ValueError(
                f"DelayConfig.mode must be one of {_DELAY_MODES}, "
                f"got '{self.mode}'"
            )
        if self.max_delay < 1:
            raise ValueError(
                f"DelayConfig.max_delay must be >= 1, got {self.max_delay}"
            )
        if self.num_bins < 1:
            raise ValueError(
                f"DelayConfig.num_bins must be >= 1, got {self.num_bins}"
            )
        if self.granularity not in _DELAY_GRANULARITIES:
            raise ValueError(
                f"DelayConfig.granularity must be one of {_DELAY_GRANULARITIES}, "
                f"got '{self.granularity}'"
            )
        if self.block_size < 1:
            raise ValueError(
                f"DelayConfig.block_size must be >= 1, got {self.block_size}"
            )
        if self.sigma_start <= 0.0:
            raise ValueError(
                f"DelayConfig.sigma_start must be > 0, got {self.sigma_start}"
            )
        if self.sigma_end <= 0.0:
            raise ValueError(
                f"DelayConfig.sigma_end must be > 0, got {self.sigma_end}"
            )
        if self.sigma_end > self.sigma_start:
            raise ValueError(
                f"DelayConfig.sigma_end ({self.sigma_end}) must be "
                f"<= sigma_start ({self.sigma_start})"
            )
        if self.sigma_decay_epochs < 1:
            raise ValueError(
                f"DelayConfig.sigma_decay_epochs must be >= 1, "
                f"got {self.sigma_decay_epochs}"
            )
        if self.sigma_schedule not in _SIGMA_SCHEDULES:
            raise ValueError(
                f"DelayConfig.sigma_schedule must be one of {_SIGMA_SCHEDULES}, "
                f"got '{self.sigma_schedule}'"
            )
        if self.init_strategy not in _DELAY_INIT_STRATEGIES:
            raise ValueError(
                f"DelayConfig.init_strategy must be one of "
                f"{_DELAY_INIT_STRATEGIES}, got '{self.init_strategy}'"
            )
        if self.init_seed is not None and self.init_seed < 0:
            raise ValueError(
                f"DelayConfig.init_seed must be non-negative or None, "
                f"got {self.init_seed}"
            )
        if self.memory_warn_threshold < 1:
            raise ValueError(
                f"DelayConfig.memory_warn_threshold must be >= 1, "
                f"got {self.memory_warn_threshold}"
            )
        if self.implementation not in _DELAY_IMPLEMENTATIONS:
            raise ValueError(
                f"DelayConfig.implementation must be one of "
                f"{_DELAY_IMPLEMENTATIONS}, got '{self.implementation}'"
            )
        # Logical consistency: mode="off" should also have enabled=False
        if self.mode == "off" and self.enabled:
            raise ValueError(
                "DelayConfig.mode='off' is inconsistent with enabled=True. "
                "Set enabled=False or change mode."
            )

    def sigma_at_epoch(self, epoch: int) -> float:
        """Return the sigma value for a given training epoch.

        Implements the configured annealing schedule.  Does not modify self.

        Args:
            epoch: Zero-indexed training epoch number.

        Returns:
            Sigma value clipped to [sigma_end, sigma_start].
        """
        if self.sigma_schedule == "constant":
            return self.sigma_start
        # Exponential decay (matches DCLS paper)
        import math
        sigma = (
            self.sigma_end
            + (self.sigma_start - self.sigma_end)
            * math.exp(-epoch / max(1, self.sigma_decay_epochs))
        )
        return float(max(self.sigma_end, min(self.sigma_start, sigma)))

    def estimate_buffer_elements(self, batch_size: int, num_neurons: int) -> int:
        """Estimate spike history buffer element count for memory planning.

        Args:
            batch_size: Training batch size B.
            num_neurons: Number of neurons N in the layer.

        Returns:
            Approximate number of float32 elements: B * max_delay * N.
        """
        return batch_size * self.max_delay * num_neurons


# ---------------------------------------------------------------------------
# Section 2: TauConfig
# ---------------------------------------------------------------------------

#: All legal tau modes.
_TAU_MODES = (
    "homogeneous_fixed",
    "heterogeneous_fixed",
    "heterogeneous_learnable",
)

#: All legal granularity levels for tau.
_TAU_GRANULARITIES = ("per_neuron", "per_channel", "per_layer")

#: All legal tau initialization strategies.
_TAU_INIT_STRATEGIES = (
    "homogeneous",
    "heterogeneous_gamma",
    "heterogeneous_loguniform",
    "preset_bank",
)


@dataclass
class TauConfig:
    """Configuration for heterogeneous membrane time constants (tau / beta).

    The LIF membrane time constant tau determines how quickly the membrane
    potential decays: ``beta = exp(-dt / tau)``.  Making tau heterogeneous
    across a neuron population — either fixed or learned — improves the
    network's ability to represent multi-scale temporal patterns.

    Attributes:
        enabled:
            Master switch.  When False the SNN uses a single homogeneous
            scalar beta from ``SNNConfig.beta``.
        mode:
            Tau mode:

            - ``"homogeneous_fixed"``      — Single shared beta (baseline).
            - ``"heterogeneous_fixed"``    — Per-neuron beta, not trained.
            - ``"heterogeneous_learnable"``— Per-neuron beta, gradient-trained.

        granularity:
            Sharing level for tau parameters:

            - ``"per_neuron"``  — Independent tau per neuron [default].
            - ``"per_channel"`` — One tau per channel (for conv layers).
            - ``"per_layer"``   — One tau per layer (very coarse).

        tau_0:
            Default / reference tau in the same units as ``dt``.
            beta equivalent: ``exp(-dt / tau_0)``.
            At dt=1.0, tau_0=20 gives beta ~= 0.951.
        tau_min:
            Minimum allowed tau (fastest time constant). Must be > 0.
        tau_max:
            Maximum allowed tau (slowest time constant). Must be > tau_min.
        dt:
            Simulation timestep duration (e.g. 1.0 ms).  Used to convert
            between tau and beta via ``beta = exp(-dt / tau)``.
        init_strategy:
            Strategy for initialising the per-neuron tau values:

            - ``"homogeneous"``             — All neurons start at tau_0.
            - ``"heterogeneous_gamma"``     — Draw from Gamma(k, theta).
            - ``"heterogeneous_loguniform"``— Draw from LogUniform[tau_min, tau_max].
            - ``"preset_bank"``             — Assign neurons to preset_values bins.

        gamma_shape:
            Shape parameter k of the Gamma distribution (when
            init_strategy="heterogeneous_gamma").  Mean = k * theta.
        gamma_scale:
            Scale parameter theta of the Gamma distribution.
            With defaults k=4, theta=5, mean tau = 20.
        preset_values:
            Tuple of discrete tau values used by ``init_strategy="preset_bank"``.
            Neurons are assigned round-robin to these values.
        init_seed:
            Optional random seed for reproducible tau initialisation.
        beta_min:
            Hard lower bound on the beta value derived from tau.  Prevents
            membrane potential from resetting too aggressively.
        beta_max:
            Hard upper bound on beta.  Matches ``BETA_MAX`` in neurons_template.py.
    """

    # Master switch
    enabled: bool = True
    mode: str = "heterogeneous_learnable"

    # Granularity
    granularity: str = "per_neuron"

    # Tau range and default
    tau_0: float = 20.0
    tau_min: float = 1.0
    tau_max: float = 100.0
    dt: float = 1.0

    # Initialization
    init_strategy: str = "heterogeneous_loguniform"

    # Gamma distribution params
    gamma_shape: float = 4.0
    gamma_scale: float = 5.0

    # Preset bank
    preset_values: Tuple[float, ...] = (2.0, 5.0, 10.0, 20.0, 50.0)

    # Seed
    init_seed: Optional[int] = None

    # Safety bounds on beta
    beta_min: float = 0.001
    beta_max: float = 0.999

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def beta_0(self) -> float:
        """Convert tau_0 to the corresponding beta value.

        Returns:
            ``exp(-dt / tau_0)`` clamped to [beta_min, beta_max].
        """
        raw = math.exp(-self.dt / self.tau_0)
        return float(max(self.beta_min, min(self.beta_max, raw)))

    @property
    def beta_range(self) -> Tuple[float, float]:
        """Return (beta_for_tau_max, beta_for_tau_min) — i.e. (slow, fast).

        Note that beta increases with tau (slower decay = higher beta), so
        beta_for_tau_max > beta_for_tau_min.
        """
        b_slow = math.exp(-self.dt / self.tau_max)
        b_fast = math.exp(-self.dt / self.tau_min)
        return (
            float(max(self.beta_min, min(self.beta_max, b_slow))),
            float(max(self.beta_min, min(self.beta_max, b_fast))),
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Check all field constraints and raise ValueError on violation.

        Raises:
            ValueError: Describes the first constraint that is violated.
        """
        if self.mode not in _TAU_MODES:
            raise ValueError(
                f"TauConfig.mode must be one of {_TAU_MODES}, got '{self.mode}'"
            )
        if self.granularity not in _TAU_GRANULARITIES:
            raise ValueError(
                f"TauConfig.granularity must be one of {_TAU_GRANULARITIES}, "
                f"got '{self.granularity}'"
            )
        if self.tau_0 <= 0.0:
            raise ValueError(
                f"TauConfig.tau_0 must be > 0, got {self.tau_0}"
            )
        if self.tau_min <= 0.0:
            raise ValueError(
                f"TauConfig.tau_min must be > 0, got {self.tau_min}"
            )
        if self.tau_max <= self.tau_min:
            raise ValueError(
                f"TauConfig.tau_max ({self.tau_max}) must be > "
                f"tau_min ({self.tau_min})"
            )
        if not (self.tau_min <= self.tau_0 <= self.tau_max):
            raise ValueError(
                f"TauConfig.tau_0 ({self.tau_0}) must be in "
                f"[tau_min={self.tau_min}, tau_max={self.tau_max}]"
            )
        if self.dt <= 0.0:
            raise ValueError(
                f"TauConfig.dt must be > 0, got {self.dt}"
            )
        if self.init_strategy not in _TAU_INIT_STRATEGIES:
            raise ValueError(
                f"TauConfig.init_strategy must be one of "
                f"{_TAU_INIT_STRATEGIES}, got '{self.init_strategy}'"
            )
        if self.gamma_shape <= 0.0:
            raise ValueError(
                f"TauConfig.gamma_shape must be > 0, got {self.gamma_shape}"
            )
        if self.gamma_scale <= 0.0:
            raise ValueError(
                f"TauConfig.gamma_scale must be > 0, got {self.gamma_scale}"
            )
        if len(self.preset_values) < 1:
            raise ValueError(
                "TauConfig.preset_values must contain at least one value"
            )
        if any(v <= 0 for v in self.preset_values):
            raise ValueError(
                "All values in TauConfig.preset_values must be > 0"
            )
        if self.init_seed is not None and self.init_seed < 0:
            raise ValueError(
                f"TauConfig.init_seed must be non-negative or None, "
                f"got {self.init_seed}"
            )
        if not (0.0 < self.beta_min < self.beta_max < 1.0):
            raise ValueError(
                f"TauConfig requires 0 < beta_min ({self.beta_min}) < "
                f"beta_max ({self.beta_max}) < 1"
            )
        # Note: mode="homogeneous_fixed" with enabled=True is valid — it means
        # "the tau module exists but all neurons share the same fixed tau."
        # This is useful as an ablation baseline with the module plumbed in.


# ---------------------------------------------------------------------------
# Section 3: AblationConfig
# ---------------------------------------------------------------------------

#: All legal benchmark identifiers.
_BENCHMARKS = ("shd", "ssc", "smnist", "psmnist", "mnist")


@dataclass
class AblationConfig:
    """Configuration for a single delay/tau ablation experiment.

    Bundles a DelayConfig and TauConfig with experiment metadata: name,
    seeds, benchmark, training duration, and output path.  Designed to be
    serialized to JSON for reproducible experiment management.

    Attributes:
        name:
            Short human-readable identifier (e.g. ``"baseline"``,
            ``"delays_only"``).  Used as the subdirectory name under
            ``output_dir``.
        delay_config:
            DelayConfig instance controlling the delay feature.
        tau_config:
            TauConfig instance controlling the heterogeneous-tau feature.
        seeds:
            Tuple of random seeds to run.  Results are averaged across seeds
            for statistical significance.
        benchmark:
            Dataset / task to evaluate on:

            - ``"shd"``    — Spiking Heidelberg Digits (audio classification).
            - ``"ssc"``    — Spiking Speech Commands.
            - ``"smnist"`` — Sequential permuted MNIST (standard).
            - ``"psmnist"``— Permuted Sequential MNIST.
            - ``"mnist"``  — Plain MNIST (sanity-check baseline).

        epochs:
            Number of training epochs per seed.
        output_dir:
            Base directory for writing results, logs, and checkpoints.
            Each seed creates a subdirectory: ``{output_dir}/{name}/seed_{s}/``.
    """

    name: str = "baseline"
    delay_config: DelayConfig = field(default_factory=DelayConfig)
    tau_config: TauConfig = field(default_factory=TauConfig)
    seeds: Tuple[int, ...] = (42, 123, 456)
    benchmark: str = "shd"
    epochs: int = 100
    output_dir: str = "results/ablation"

    def validate(self) -> None:
        """Validate the ablation config and its sub-configs.

        Raises:
            ValueError: Describes the first constraint violated.
        """
        if not self.name:
            raise ValueError("AblationConfig.name must be a non-empty string")
        if self.benchmark not in _BENCHMARKS:
            raise ValueError(
                f"AblationConfig.benchmark must be one of {_BENCHMARKS}, "
                f"got '{self.benchmark}'"
            )
        if self.epochs < 1:
            raise ValueError(
                f"AblationConfig.epochs must be >= 1, got {self.epochs}"
            )
        if len(self.seeds) < 1:
            raise ValueError(
                "AblationConfig.seeds must contain at least one seed"
            )
        if any(s < 0 for s in self.seeds):
            raise ValueError(
                "All values in AblationConfig.seeds must be non-negative"
            )
        if not self.output_dir:
            raise ValueError(
                "AblationConfig.output_dir must be a non-empty string"
            )
        # Validate sub-configs
        self.delay_config.validate()
        self.tau_config.validate()


# ---------------------------------------------------------------------------
# Section 4: Preset factory functions
# ---------------------------------------------------------------------------

def ablation_baseline() -> AblationConfig:
    """Return a pure baseline: no learnable delays, no heterogeneous tau.

    This is the control condition for the 2x2 ablation matrix.  The SNN
    runs with a single homogeneous beta and no spike history buffer.

    Returns:
        AblationConfig with both features disabled.
    """
    return AblationConfig(
        name="baseline",
        delay_config=DelayConfig(enabled=False, mode="off"),
        tau_config=TauConfig(enabled=False, mode="homogeneous_fixed"),
    )


def ablation_delays_only() -> AblationConfig:
    """Return a config with DCLS learnable delays and homogeneous tau.

    Isolates the contribution of the synaptic delay mechanism by keeping
    tau fixed and homogeneous.

    Returns:
        AblationConfig with delays enabled, tau homogeneous.
    """
    return AblationConfig(
        name="delays_only",
        delay_config=DelayConfig(
            enabled=True,
            mode="learnable_dcls",
        ),
        tau_config=TauConfig(enabled=False, mode="homogeneous_fixed"),
    )


def ablation_hetero_only() -> AblationConfig:
    """Return a config with heterogeneous learnable tau and no delays.

    Isolates the contribution of per-neuron time constants.

    Returns:
        AblationConfig with tau heterogeneous and learnable, delays off.
    """
    return AblationConfig(
        name="hetero_only",
        delay_config=DelayConfig(enabled=False, mode="off"),
        tau_config=TauConfig(
            enabled=True,
            mode="heterogeneous_learnable",
            init_strategy="heterogeneous_loguniform",
        ),
    )


def ablation_both() -> AblationConfig:
    """Return the full-featured config: DCLS delays AND heterogeneous tau.

    This represents the complete set of 2024-2025 improvements to the SNN
    core and is expected to perform best on temporal benchmarks.

    Returns:
        AblationConfig with both features enabled.
    """
    return AblationConfig(
        name="both",
        delay_config=DelayConfig(
            enabled=True,
            mode="learnable_dcls",
        ),
        tau_config=TauConfig(
            enabled=True,
            mode="heterogeneous_learnable",
            init_strategy="heterogeneous_loguniform",
        ),
    )


def get_ablation_matrix() -> Dict[str, AblationConfig]:
    """Return the canonical 2x2 ablation matrix as a named dict.

    Keys correspond to the four conditions:
      - ``"baseline"``    — Neither delay nor heterogeneous tau.
      - ``"delays_only"`` — Delays only.
      - ``"hetero_only"`` — Heterogeneous tau only.
      - ``"both"``        — Both features.

    Returns:
        Dict mapping condition name to AblationConfig.
    """
    return {
        "baseline": ablation_baseline(),
        "delays_only": ablation_delays_only(),
        "hetero_only": ablation_hetero_only(),
        "both": ablation_both(),
    }


# ---------------------------------------------------------------------------
# Section 5: SNNConfig integration helper
# ---------------------------------------------------------------------------

def upgrade_snn_config(
    snn_config: Any,
    delay_config: Optional[DelayConfig] = None,
    tau_config: Optional[TauConfig] = None,
) -> Any:
    """Upgrade an existing SNNConfig with structured delay and tau configs.

    Maps the legacy flat boolean flags onto the new structured config objects.
    The original SNNConfig object is modified in-place and returned.

    Flag mapping:
      - ``use_learnable_delays`` → ``delay_config.enabled``
      - ``max_delay``            → ``delay_config.max_delay``
      - ``use_heterogeneous_tau``→ ``tau_config.enabled``
      - ``beta``                 → ``tau_config.tau_0`` (via tau = -dt / ln(beta))

    Args:
        snn_config:
            An instance of ``brain_ai.config.SNNConfig`` (or any object with
            the expected flat-flag attributes).  Modified in-place.
        delay_config:
            If provided, overrides the delay sub-config.  If None, a new
            DelayConfig is constructed from the flat flags on snn_config.
        tau_config:
            If provided, overrides the tau sub-config.  If None, a new
            TauConfig is constructed from the flat flags on snn_config.

    Returns:
        The upgraded snn_config (same object, mutated).

    Example::

        from brain_ai.config import SNNConfig
        from brain_ai.core.delay_tau_config import upgrade_snn_config

        snn = SNNConfig()
        snn = upgrade_snn_config(snn)
        print(snn.delay.max_delay)   # 16
        print(snn.tau.beta_0)        # ~0.951
    """
    # Build DelayConfig from flat flags if not supplied
    if delay_config is None:
        use_delays = getattr(snn_config, "use_learnable_delays", True)
        max_delay = getattr(snn_config, "max_delay", 16)
        delay_config = DelayConfig(
            enabled=use_delays,
            mode="learnable_dcls" if use_delays else "off",
            max_delay=max_delay,
        )

    # Build TauConfig from flat flags if not supplied
    if tau_config is None:
        use_hetero = getattr(snn_config, "use_heterogeneous_tau", True)
        beta = getattr(snn_config, "beta", 0.95)
        # Convert beta to tau_0: tau_0 = -dt / ln(beta), dt assumed 1.0
        tau_0 = _beta_to_tau(beta, dt=1.0)
        tau_config = TauConfig(
            enabled=use_hetero,
            mode="heterogeneous_learnable" if use_hetero else "homogeneous_fixed",
            tau_0=tau_0,
        )

    # Validate before attaching
    delay_config.validate()
    tau_config.validate()

    # Attach structured configs
    snn_config.delay = delay_config
    snn_config.tau = tau_config

    return snn_config


def _beta_to_tau(beta: float, dt: float = 1.0) -> float:
    """Convert a membrane decay factor beta to a time constant tau.

    Args:
        beta: Decay factor in (0, 1).  Typical value: 0.95.
        dt: Simulation timestep (same units as desired tau).

    Returns:
        tau = -dt / ln(beta).  Clipped to [1e-6, 1e6] to avoid
        numerical issues near beta=0 or beta=1.
    """
    if not (0.0 < beta < 1.0):
        raise ValueError(
            f"beta must be in (0, 1) for tau conversion, got {beta}"
        )
    return float(max(1e-6, min(1e6, -dt / math.log(beta))))


# ---------------------------------------------------------------------------
# Section 6: Backward compatibility
# ---------------------------------------------------------------------------

def legacy_to_new_config(
    use_learnable_delays: bool,
    max_delay: int,
    use_heterogeneous_tau: bool,
    beta: float,
) -> Tuple[DelayConfig, TauConfig]:
    """Convert old-style flat flags to new structured configs.

    This is the canonical migration function for code that previously read
    delay/tau settings directly from the flat SNNConfig fields.

    Args:
        use_learnable_delays:
            Former ``SNNConfig.use_learnable_delays``.
        max_delay:
            Former ``SNNConfig.max_delay``.
        use_heterogeneous_tau:
            Former ``SNNConfig.use_heterogeneous_tau``.
        beta:
            Former ``SNNConfig.beta``.

    Returns:
        ``(DelayConfig, TauConfig)`` — two fully validated structured configs.

    Example::

        delay_cfg, tau_cfg = legacy_to_new_config(
            use_learnable_delays=True,
            max_delay=16,
            use_heterogeneous_tau=True,
            beta=0.95,
        )
    """
    # Delays
    delay_cfg = DelayConfig(
        enabled=use_learnable_delays,
        mode="learnable_dcls" if use_learnable_delays else "off",
        max_delay=max_delay,
    )

    # Tau — convert beta to tau_0 using dt=1.0 (default simulation step)
    tau_0 = _beta_to_tau(beta, dt=1.0)
    tau_cfg = TauConfig(
        enabled=use_heterogeneous_tau,
        mode=(
            "heterogeneous_learnable"
            if use_heterogeneous_tau
            else "homogeneous_fixed"
        ),
        tau_0=tau_0,
    )

    delay_cfg.validate()
    tau_cfg.validate()

    return delay_cfg, tau_cfg


# ---------------------------------------------------------------------------
# Section 7: Config serialization
# ---------------------------------------------------------------------------

def config_to_dict(
    config: Union[DelayConfig, TauConfig, AblationConfig],
) -> Dict[str, Any]:
    """Serialize a config dataclass to a plain dict suitable for JSON logging.

    Tuples are converted to lists so the result is JSON-serializable by the
    standard ``json`` module.  Nested dataclasses (e.g. AblationConfig's
    sub-configs) are recursively serialized.

    Args:
        config:
            A DelayConfig, TauConfig, or AblationConfig instance.

    Returns:
        Dict with the same key/value structure as the dataclass.

    Example::

        d = config_to_dict(ablation_both())
        with open("experiment.json", "w") as f:
            json.dump(d, f, indent=2)
    """
    d: Dict[str, Any] = {}
    for f in dataclasses.fields(config):
        value = getattr(config, f.name)
        if dataclasses.is_dataclass(value):
            d[f.name] = config_to_dict(value)  # type: ignore[arg-type]
        elif isinstance(value, tuple):
            d[f.name] = list(value)
        else:
            d[f.name] = value
    return d


def config_from_dict(
    d: Dict[str, Any],
    config_class: Type[Union[DelayConfig, TauConfig, AblationConfig]],
) -> Union[DelayConfig, TauConfig, AblationConfig]:
    """Deserialize a config dataclass from a plain dict.

    Performs type coercion for fields that JSON cannot represent natively
    (lists -> tuples where the dataclass field is typed as Tuple).

    Args:
        d:
            A dict previously produced by ``config_to_dict()``, or loaded
            from a JSON file.
        config_class:
            The dataclass class to instantiate: DelayConfig, TauConfig,
            or AblationConfig.

    Returns:
        A fully constructed and populated instance of ``config_class``.

    Raises:
        KeyError:   If a required field is missing from ``d``.
        ValueError: If a field value fails the config's validate() call.

    Example::

        raw = json.loads(json_string)
        cfg = config_from_dict(raw, AblationConfig)
        cfg.validate()
    """
    kwargs: Dict[str, Any] = {}

    for f in dataclasses.fields(config_class):
        if f.name not in d:
            continue  # Use dataclass default for missing optional fields

        value = d[f.name]

        # Reconstruct nested dataclasses
        if f.name == "delay_config" and isinstance(value, dict):
            value = config_from_dict(value, DelayConfig)
        elif f.name == "tau_config" and isinstance(value, dict):
            value = config_from_dict(value, TauConfig)
        # Restore tuples from JSON lists
        elif isinstance(value, list) and _field_is_tuple(f):
            value = tuple(value)

        kwargs[f.name] = value

    return config_class(**kwargs)


def _field_is_tuple(f: dataclasses.Field) -> bool:  # type: ignore[type-arg]
    """Return True if the dataclass field's type annotation includes Tuple."""
    hint = str(f.type) if isinstance(f.type, str) else repr(f.type)
    return "Tuple" in hint or "tuple" in hint


# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = [
    # Dataclasses
    "DelayConfig",
    "TauConfig",
    "AblationConfig",
    # Preset factories
    "ablation_baseline",
    "ablation_delays_only",
    "ablation_hetero_only",
    "ablation_both",
    "get_ablation_matrix",
    # Integration helpers
    "upgrade_snn_config",
    "legacy_to_new_config",
    # Serialization
    "config_to_dict",
    "config_from_dict",
    # Constants
    "_DELAY_MODES",
    "_DELAY_GRANULARITIES",
    "_SIGMA_SCHEDULES",
    "_DELAY_INIT_STRATEGIES",
    "_DELAY_IMPLEMENTATIONS",
    "_TAU_MODES",
    "_TAU_GRANULARITIES",
    "_TAU_INIT_STRATEGIES",
    "_BENCHMARKS",
]


# ---------------------------------------------------------------------------
# Self-test (__main__)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    _PASS = "PASS"
    _FAIL = "FAIL"

    def _section(title: str) -> None:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")

    def _check(label: str, condition: bool) -> None:
        status = _PASS if condition else _FAIL
        print(f"  [{status}] {label}")
        if not condition:
            sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Create and validate all individual configs
    # ------------------------------------------------------------------
    _section("1. Create and validate all configs")

    delay_cfg = DelayConfig()
    delay_cfg.validate()
    _check("DelayConfig default validates", True)

    tau_cfg = TauConfig()
    tau_cfg.validate()
    _check("TauConfig default validates", True)

    ablation_cfg = AblationConfig()
    ablation_cfg.validate()
    _check("AblationConfig default validates", True)

    # Check derived properties
    _check(
        f"TauConfig.beta_0 in (0, 1): {tau_cfg.beta_0:.4f}",
        0.0 < tau_cfg.beta_0 < 1.0,
    )
    b_slow, b_fast = tau_cfg.beta_range
    _check(
        f"TauConfig.beta_range ordered (slow={b_slow:.3f} > fast={b_fast:.3f})",
        b_slow > b_fast,
    )

    # sigma annealing
    s0 = delay_cfg.sigma_at_epoch(0)
    s25 = delay_cfg.sigma_at_epoch(25)
    s50 = delay_cfg.sigma_at_epoch(50)
    _check(
        f"sigma annealing: epoch 0={s0:.2f}, 25={s25:.2f}, 50={s50:.2f}",
        s0 >= s25 >= s50,
    )

    # buffer size estimation
    elems = delay_cfg.estimate_buffer_elements(batch_size=32, num_neurons=512)
    _check(
        f"buffer element estimate: {elems:,} = 32*16*512",
        elems == 32 * 16 * 512,
    )

    # ------------------------------------------------------------------
    # 2. Test all ablation presets
    # ------------------------------------------------------------------
    _section("2. Ablation presets")

    matrix = get_ablation_matrix()
    _check("get_ablation_matrix returns 4 configs", len(matrix) == 4)

    expected_names = {"baseline", "delays_only", "hetero_only", "both"}
    _check("All expected names present", set(matrix.keys()) == expected_names)

    for name, cfg in matrix.items():
        cfg.validate()
        _check(f"  AblationConfig '{name}' validates", True)

    # Check baseline has both features off
    bl = matrix["baseline"]
    _check(
        "baseline: delay disabled",
        not bl.delay_config.enabled and bl.delay_config.mode == "off",
    )
    _check(
        "baseline: tau homogeneous fixed",
        not bl.tau_config.enabled and bl.tau_config.mode == "homogeneous_fixed",
    )

    # Check delays_only
    do = matrix["delays_only"]
    _check("delays_only: delay enabled", do.delay_config.enabled)
    _check("delays_only: tau disabled", not do.tau_config.enabled)

    # Check hetero_only
    ho = matrix["hetero_only"]
    _check("hetero_only: delay disabled", not ho.delay_config.enabled)
    _check("hetero_only: tau enabled", ho.tau_config.enabled)

    # Check both
    bt = matrix["both"]
    _check("both: delay enabled", bt.delay_config.enabled)
    _check("both: tau enabled", bt.tau_config.enabled)

    # ------------------------------------------------------------------
    # 3. Test legacy conversion
    # ------------------------------------------------------------------
    _section("3. Legacy conversion")

    d_new, t_new = legacy_to_new_config(
        use_learnable_delays=True,
        max_delay=8,
        use_heterogeneous_tau=True,
        beta=0.9,
    )
    _check("legacy_to_new_config returns DelayConfig", isinstance(d_new, DelayConfig))
    _check("legacy_to_new_config returns TauConfig", isinstance(t_new, TauConfig))
    _check("DelayConfig.max_delay=8", d_new.max_delay == 8)
    _check("DelayConfig.enabled=True", d_new.enabled)
    _check("TauConfig.enabled=True", t_new.enabled)

    # tau_0 from beta=0.9: -1.0 / ln(0.9) = 9.49
    expected_tau = -1.0 / math.log(0.9)
    _check(
        f"TauConfig.tau_0 ~= {expected_tau:.2f}",
        abs(t_new.tau_0 - expected_tau) < 0.01,
    )

    # Disabled version
    d_off, t_off = legacy_to_new_config(
        use_learnable_delays=False,
        max_delay=16,
        use_heterogeneous_tau=False,
        beta=0.95,
    )
    _check("legacy off: delay.mode='off'", d_off.mode == "off")
    _check("legacy off: tau.mode='homogeneous_fixed'", t_off.mode == "homogeneous_fixed")

    # upgrade_snn_config smoke test (using a plain namespace)
    class _FakeSNNConfig:
        use_learnable_delays = True
        max_delay = 16
        use_heterogeneous_tau = True
        beta = 0.95

    fake_snn = _FakeSNNConfig()
    upgraded = upgrade_snn_config(fake_snn)
    _check("upgrade_snn_config: .delay attached", hasattr(upgraded, "delay"))
    _check("upgrade_snn_config: .tau attached", hasattr(upgraded, "tau"))
    _check(
        "upgrade_snn_config: delay.max_delay=16",
        upgraded.delay.max_delay == 16,
    )

    # ------------------------------------------------------------------
    # 4. Test serialization round-trip
    # ------------------------------------------------------------------
    _section("4. Serialization round-trip")

    for name, orig in matrix.items():
        d = config_to_dict(orig)
        _check(f"config_to_dict '{name}' produces dict", isinstance(d, dict))
        _check(
            f"config_to_dict '{name}' has 'delay_config' key",
            "delay_config" in d,
        )

        # JSON round-trip
        json_str = json.dumps(d)
        loaded_d = json.loads(json_str)
        restored = config_from_dict(loaded_d, AblationConfig)
        restored.validate()
        _check(
            f"config_from_dict '{name}' validates after JSON round-trip",
            True,
        )
        _check(
            f"round-trip name preserved '{name}'",
            restored.name == orig.name,
        )
        _check(
            f"round-trip delay.enabled preserved",
            restored.delay_config.enabled == orig.delay_config.enabled,
        )
        _check(
            f"round-trip tau.enabled preserved",
            restored.tau_config.enabled == orig.tau_config.enabled,
        )
        _check(
            f"round-trip seeds preserved",
            restored.seeds == orig.seeds,
        )

    # Individual config round-trips
    d_dict = config_to_dict(DelayConfig())
    d_restored = config_from_dict(d_dict, DelayConfig)
    _check("DelayConfig round-trip", d_restored.max_delay == 16)

    t_dict = config_to_dict(TauConfig())
    t_restored = config_from_dict(t_dict, TauConfig)
    _check("TauConfig round-trip", t_restored.tau_0 == 20.0)

    # ------------------------------------------------------------------
    # 5. Test validation catches bad values
    # ------------------------------------------------------------------
    _section("5. Validation catches bad values")

    def _expect_error(label: str, fn: Any) -> None:
        try:
            fn()
            print(f"  [{_FAIL}] {label} — no error raised")
            sys.exit(1)
        except ValueError as exc:
            print(f"  [{_PASS}] {label} — caught: {exc}")

    _expect_error(
        "DelayConfig: max_delay=0",
        lambda: DelayConfig(max_delay=0).validate(),
    )
    _expect_error(
        "DelayConfig: bad mode",
        lambda: DelayConfig(mode="invalid").validate(),
    )
    _expect_error(
        "DelayConfig: sigma_end > sigma_start",
        lambda: DelayConfig(sigma_start=0.5, sigma_end=1.0).validate(),
    )
    _expect_error(
        "DelayConfig: mode='off' but enabled=True",
        lambda: DelayConfig(enabled=True, mode="off").validate(),
    )
    _expect_error(
        "TauConfig: tau_min >= tau_max",
        lambda: TauConfig(tau_min=50.0, tau_max=20.0).validate(),
    )
    _expect_error(
        "TauConfig: tau_0 outside [tau_min, tau_max]",
        lambda: TauConfig(tau_0=200.0, tau_max=100.0).validate(),
    )
    _expect_error(
        "TauConfig: bad mode",
        lambda: TauConfig(mode="unknown").validate(),
    )
    # homogeneous_fixed + enabled=True is now valid (ablation baseline)
    TauConfig(enabled=True, mode="homogeneous_fixed").validate()
    _check("TauConfig: homogeneous_fixed + enabled=True is valid", True)
    _expect_error(
        "AblationConfig: bad benchmark",
        lambda: AblationConfig(benchmark="invalid").validate(),
    )
    _expect_error(
        "AblationConfig: epochs=0",
        lambda: AblationConfig(epochs=0).validate(),
    )
    _expect_error(
        "AblationConfig: empty seeds",
        lambda: AblationConfig(seeds=()).validate(),
    )

    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  All self-tests passed.")
    print(f"{'=' * 60}\n")
