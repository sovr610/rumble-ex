"""
Amortized Policy Distillation for Active Inference.

Planning under active inference is expensive: each decision requires evaluating
N candidate action sequences over H-step rollouts through the generative model,
computing Expected Free Energy (EFE) for each trajectory.  Cost scales as
O(N * H) forward passes per decision step.

This module implements **amortized policy distillation** -- training a fast
feed-forward policy pi_theta(a | o, ctx) that approximates the planner output
in a single forward pass.  The amortized policy is trained online via
distillation from the planner's action distribution, so that during deployment
the full planning loop is bypassed.  EFE is still computed periodically for
auditing to ensure the amortized policy has not drifted.

Architecture:
    obs (+ optional ctx) --> MLP backbone --> action distribution head
                                              |
                                discrete: categorical logits (B, action_dim)
                                continuous: Gaussian (mu, log_std) (B, action_dim)

Distillation modes:
    - **Hard**: Cross-entropy (discrete) or MSE (continuous) on argmax action.
    - **Soft**: KL divergence matching the full planner action distribution.
    - **Offline**: From a stored Minari-style dataset, optionally reweighted by
      EFE to concentrate on high-quality trajectories.

Integration with the brain-inspired AI system:
    The decision layer (basal ganglia analog) in ``brain_ai/decision/`` uses
    active inference for action selection.  At production scale (7B params), the
    planner is invoked during training to collect distillation targets.  At
    inference time the amortized policy runs in a single pass, with periodic
    auditing to detect distribution shift.

References:
    - Fountas et al., "Deep Active Inference Agents Using Monte-Carlo
      Methods", NeurIPS 2020
    - Millidge et al., "Whence the Expected Free Energy?", Neural Computation 2021
    - Catal et al., "Learning Generative State Space Models for Active
      Inference", Frontiers in Computational Neuroscience 2021
    - Mazzaglia et al., "The Free Energy Principle for Perception and Action:
      A Deep Learning Perspective", Entropy 2022
    - Tschantz et al., "Scaling Active Inference", ICLR 2023

Typical usage:
    >>> from amortized_policy_template import AmortizedPolicy, PlannerDistiller
    >>> policy = AmortizedPolicy(AmortizedPolicyConfig())
    >>> distiller = PlannerDistiller(policy, AmortizedPolicyConfig())
    >>> # During training: collect planner outputs
    >>> distiller.add_experience(obs, ctx, planner_action, planner_dist)
    >>> loss = distiller.train_step(optimizer)
    >>> # During inference: single forward pass
    >>> action = policy.sample(obs, deterministic=True)
"""

from __future__ import annotations

import collections
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Independent, Normal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOG_2PI: float = math.log(2.0 * math.pi)
MIN_LOG_STD: float = -5.0
MAX_LOG_STD: float = 2.0
EFE_SUM_INVARIANT_TOL: float = 1e-5
DEFAULT_BUFFER_CAPACITY: int = 50_000


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class AmortizedPolicyConfig:
    """Configuration for the amortized (distilled) policy network.

    The amortized policy pi_theta(a | o, ctx) approximates the full active
    inference planner via distillation.  This config controls architecture,
    distillation strategy, action space type, and auditing schedule.

    Attributes:
        enabled: Whether to build and use the amortized policy.  When False,
            the agent always falls back to the full planner.
        hidden_dim: Width of hidden layers in the policy MLP.
        num_layers: Number of hidden layers in the policy MLP.  Minimum 1.
        distill_lr: Learning rate for online distillation.  Used only when
            the PlannerDistiller manages its own optimizer.
        distill_mode: Distillation strategy.  One of:
            - ``"soft"``: KL divergence matching the full planner distribution.
            - ``"hard"``: Cross-entropy (discrete) or MSE (continuous) on the
              argmax action from the planner.
            - ``"offline"``: Distillation from a pre-collected dataset
              (Minari-style), optionally reweighted by EFE.
        refresh_interval: Number of agent steps between audits that compare
            the amortized policy against the full planner.
        action_type: ``"continuous"`` for Gaussian output or ``"discrete"``
            for categorical logits.
        action_dim: Dimensionality of the action space.  For discrete actions,
            this is the number of categories.  For continuous actions, this is
            the dimensionality of the action vector.
        obs_dim: Dimensionality of the observation input from the global
            workspace.  Default 4096 matches the unified workspace
            representation.
        ctx_dim: Dimensionality of optional context input.  Set to 0 to
            disable context conditioning.
        buffer_capacity: Maximum number of experiences stored in the replay
            buffer used by PlannerDistiller.  FIFO eviction.
        batch_size: Mini-batch size for distillation training steps.
        grad_clip_norm: Maximum gradient norm for distillation updates.
        efe_reweight_temperature: Temperature for EFE-based importance
            reweighting in offline distillation.  Lower values concentrate
            weight on low-EFE (high-quality) transitions.  Set to ``inf`` to
            disable reweighting.
        dropout: Dropout rate applied in the MLP backbone.
        activation: Activation function name.  One of ``"silu"``, ``"relu"``,
            ``"gelu"``, ``"tanh"``.
        use_layer_norm: Apply LayerNorm after each hidden layer.
        init_std: Standard deviation for weight initialization of the output
            heads.  Smaller values produce less noisy initial policies.
    """

    enabled: bool = True
    hidden_dim: int = 512
    num_layers: int = 3
    distill_lr: float = 1e-4
    distill_mode: str = "soft"  # "soft" | "hard" | "offline"
    refresh_interval: int = 100
    action_type: str = "continuous"  # "continuous" | "discrete"
    action_dim: int = 128
    obs_dim: int = 4096
    ctx_dim: int = 0
    buffer_capacity: int = DEFAULT_BUFFER_CAPACITY
    batch_size: int = 64
    grad_clip_norm: float = 1.0
    efe_reweight_temperature: float = 1.0
    dropout: float = 0.0
    activation: str = "silu"
    use_layer_norm: bool = False
    init_std: float = 0.01

    def __post_init__(self) -> None:
        """Validate configuration values."""
        assert self.num_layers >= 1, "num_layers must be >= 1"
        assert self.hidden_dim > 0, "hidden_dim must be > 0"
        assert self.action_dim > 0, "action_dim must be > 0"
        assert self.obs_dim > 0, "obs_dim must be > 0"
        assert self.ctx_dim >= 0, "ctx_dim must be >= 0"
        assert self.distill_mode in ("soft", "hard", "offline"), (
            f"distill_mode must be 'soft', 'hard', or 'offline', got '{self.distill_mode}'"
        )
        assert self.action_type in ("continuous", "discrete"), (
            f"action_type must be 'continuous' or 'discrete', got '{self.action_type}'"
        )
        assert self.buffer_capacity > 0, "buffer_capacity must be > 0"
        assert self.batch_size > 0, "batch_size must be > 0"
        assert self.activation in ("silu", "relu", "gelu", "tanh"), (
            f"activation must be one of silu/relu/gelu/tanh, got '{self.activation}'"
        )


# ============================================================================
# Data containers
# ============================================================================


@dataclass
class ActionDistribution:
    """Container for an action distribution produced by the amortized policy.

    For continuous actions, holds the mean and log-std of a diagonal Gaussian.
    For discrete actions, holds the categorical logits.

    Attributes:
        action_type: ``"continuous"`` or ``"discrete"``.
        logits: Categorical logits, shape ``(B, action_dim)``.  Only present
            when ``action_type == "discrete"``.
        mu: Gaussian mean, shape ``(B, action_dim)``.  Only present when
            ``action_type == "continuous"``.
        log_std: Gaussian log standard deviation, shape ``(B, action_dim)``.
            Only present when ``action_type == "continuous"``.
    """

    action_type: str
    logits: Optional[torch.Tensor] = None
    mu: Optional[torch.Tensor] = None
    log_std: Optional[torch.Tensor] = None

    @property
    def batch_size(self) -> int:
        """Return the batch dimension."""
        if self.logits is not None:
            return self.logits.shape[0]
        if self.mu is not None:
            return self.mu.shape[0]
        return 0

    @property
    def device(self) -> torch.device:
        """Return the device of the underlying tensors."""
        if self.logits is not None:
            return self.logits.device
        if self.mu is not None:
            return self.mu.device
        return torch.device("cpu")

    def sample(self, deterministic: bool = False) -> torch.Tensor:
        """Draw an action from this distribution.

        Args:
            deterministic: If True, return the mode (argmax for discrete,
                mean for continuous) instead of sampling.

        Returns:
            Action tensor.  Shape ``(B, action_dim)`` for continuous,
            ``(B,)`` for discrete.
        """
        if self.action_type == "discrete":
            assert self.logits is not None, "Discrete distribution requires logits"
            if deterministic:
                return self.logits.argmax(dim=-1)
            return Categorical(logits=self.logits).sample()
        else:
            assert self.mu is not None and self.log_std is not None, (
                "Continuous distribution requires mu and log_std"
            )
            if deterministic:
                return self.mu
            std = torch.exp(self.log_std)
            return self.mu + std * torch.randn_like(std)

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """Compute log-probability of an action under this distribution.

        Args:
            action: Action tensor.  Shape ``(B, action_dim)`` for continuous
                or ``(B,)`` for discrete.

        Returns:
            Log-probability per batch element, shape ``(B,)``.
        """
        if self.action_type == "discrete":
            assert self.logits is not None
            return Categorical(logits=self.logits).log_prob(action.long())
        else:
            assert self.mu is not None and self.log_std is not None
            std = torch.exp(self.log_std)
            dist = Independent(Normal(self.mu, std), reinterpreted_batch_ndims=1)
            return dist.log_prob(action)

    def entropy(self) -> torch.Tensor:
        """Compute the entropy of this distribution.

        Returns:
            Entropy per batch element, shape ``(B,)``.
        """
        if self.action_type == "discrete":
            assert self.logits is not None
            return Categorical(logits=self.logits).entropy()
        else:
            assert self.mu is not None and self.log_std is not None
            std = torch.exp(self.log_std)
            dist = Independent(Normal(self.mu, std), reinterpreted_batch_ndims=1)
            return dist.entropy()

    def kl_divergence(self, other: "ActionDistribution") -> torch.Tensor:
        """Compute KL divergence KL(self || other).

        Args:
            other: Another ActionDistribution of the same type.

        Returns:
            KL divergence per batch element, shape ``(B,)``.

        Raises:
            ValueError: If action types differ.
        """
        if self.action_type != other.action_type:
            raise ValueError(
                f"Cannot compute KL between {self.action_type} and "
                f"{other.action_type} distributions"
            )

        if self.action_type == "discrete":
            assert self.logits is not None and other.logits is not None
            p = Categorical(logits=self.logits)
            q = Categorical(logits=other.logits)
            return torch.distributions.kl_divergence(p, q)
        else:
            assert self.mu is not None and self.log_std is not None
            assert other.mu is not None and other.log_std is not None
            p = Independent(
                Normal(self.mu, torch.exp(self.log_std)),
                reinterpreted_batch_ndims=1,
            )
            q = Independent(
                Normal(other.mu, torch.exp(other.log_std)),
                reinterpreted_batch_ndims=1,
            )
            return torch.distributions.kl_divergence(p, q)

    def detach(self) -> "ActionDistribution":
        """Return a copy with all tensors detached from the computation graph."""
        return ActionDistribution(
            action_type=self.action_type,
            logits=self.logits.detach() if self.logits is not None else None,
            mu=self.mu.detach() if self.mu is not None else None,
            log_std=self.log_std.detach() if self.log_std is not None else None,
        )


class Experience(NamedTuple):
    """A single experience tuple stored in the replay buffer.

    Fields:
        obs: Observation tensor, shape ``(obs_dim,)``.
        ctx: Context tensor, shape ``(ctx_dim,)`` or empty ``(0,)``.
        planner_action: Action selected by the planner, shape
            ``(action_dim,)`` for continuous or scalar for discrete.
        planner_dist: Serialized planner distribution as a dict with keys
            matching ActionDistribution fields.  Tensors are on CPU and
            represent a single sample (no batch dim).
        efe_total: Scalar EFE value for this transition.
    """

    obs: torch.Tensor
    ctx: torch.Tensor
    planner_action: torch.Tensor
    planner_dist: Dict[str, torch.Tensor]
    efe_total: float


@dataclass
class AuditResult:
    """Result of a policy audit comparing amortized vs planner outputs.

    Attributes:
        agreement_rate: Fraction of batch elements where the amortized policy
            selects the same action as the planner.  For continuous actions,
            agreement is defined as cosine similarity > 0.95.
        efe_gap_mean: Mean difference in EFE between the amortized policy
            action and the planner action.  Positive means the amortized
            action has higher (worse) EFE.
        efe_gap_std: Standard deviation of the EFE gap across the batch.
        distribution_kl: Mean KL divergence KL(planner || amortized) across
            the batch.
        batch_size: Number of observations in the audit batch.
        planner_time_ms: Wall-clock time for the planner (ms).
        amortized_time_ms: Wall-clock time for the amortized policy (ms).
        speedup: Ratio planner_time / amortized_time.
    """

    agreement_rate: float
    efe_gap_mean: float
    efe_gap_std: float
    distribution_kl: float
    batch_size: int
    planner_time_ms: float
    amortized_time_ms: float
    speedup: float

    def is_healthy(
        self,
        min_agreement: float = 0.8,
        max_efe_gap: float = 1.0,
        max_kl: float = 2.0,
    ) -> bool:
        """Check whether the amortized policy meets quality thresholds.

        Args:
            min_agreement: Minimum acceptable agreement rate.
            max_efe_gap: Maximum acceptable mean EFE gap.
            max_kl: Maximum acceptable distribution KL divergence.

        Returns:
            True if all thresholds are satisfied.
        """
        return (
            self.agreement_rate >= min_agreement
            and self.efe_gap_mean <= max_efe_gap
            and self.distribution_kl <= max_kl
        )


# ============================================================================
# Utility functions
# ============================================================================


def _get_activation(name: str) -> nn.Module:
    """Return an activation module by name.

    Args:
        name: One of ``"silu"``, ``"relu"``, ``"gelu"``, ``"tanh"``.

    Returns:
        PyTorch activation module.

    Raises:
        ValueError: If the name is not recognized.
    """
    activations = {
        "silu": nn.SiLU,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
    }
    if name not in activations:
        raise ValueError(f"Unknown activation '{name}', choose from {list(activations)}")
    return activations[name]()


def _build_mlp(
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    activation: str = "silu",
    dropout: float = 0.0,
    use_layer_norm: bool = False,
) -> nn.Sequential:
    """Build a multi-layer perceptron backbone.

    Constructs ``num_layers`` hidden layers of width ``hidden_dim`` with the
    specified activation, optional LayerNorm, and optional dropout.

    Args:
        input_dim: Input feature dimensionality.
        hidden_dim: Width of each hidden layer.
        num_layers: Number of hidden layers.
        activation: Activation function name.
        dropout: Dropout probability.  0.0 disables dropout.
        use_layer_norm: If True, apply LayerNorm after each linear layer
            (before activation).

    Returns:
        Sequential module mapping ``(B, input_dim)`` to ``(B, hidden_dim)``.
    """
    layers: List[nn.Module] = []
    for i in range(num_layers):
        in_d = input_dim if i == 0 else hidden_dim
        layers.append(nn.Linear(in_d, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(_get_activation(activation))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def _serialize_dist(dist: ActionDistribution) -> Dict[str, torch.Tensor]:
    """Serialize an ActionDistribution to a dict of CPU tensors without batch dim.

    This is used for storing distributions in the experience replay buffer.

    Args:
        dist: The action distribution to serialize.

    Returns:
        Dictionary with detached CPU tensors.
    """
    result: Dict[str, torch.Tensor] = {}
    if dist.logits is not None:
        result["logits"] = dist.logits.detach().cpu()
    if dist.mu is not None:
        result["mu"] = dist.mu.detach().cpu()
    if dist.log_std is not None:
        result["log_std"] = dist.log_std.detach().cpu()
    return result


def _deserialize_dist(
    data: Dict[str, torch.Tensor],
    action_type: str,
    device: torch.device,
) -> ActionDistribution:
    """Reconstruct an ActionDistribution from serialized data.

    Args:
        data: Dictionary produced by ``_serialize_dist``.
        action_type: ``"continuous"`` or ``"discrete"``.
        device: Target device for tensors.

    Returns:
        Reconstructed ActionDistribution on the specified device.
    """
    return ActionDistribution(
        action_type=action_type,
        logits=data["logits"].to(device) if "logits" in data else None,
        mu=data["mu"].to(device) if "mu" in data else None,
        log_std=data["log_std"].to(device) if "log_std" in data else None,
    )


# ============================================================================
# AmortizedPolicy
# ============================================================================


class AmortizedPolicy(nn.Module):
    """Amortized policy network pi_theta(a | o, ctx).

    Maps observations (and optional context) to an action distribution via an
    MLP backbone.  Supports both discrete (categorical logits) and continuous
    (diagonal Gaussian) action spaces.

    The policy is trained via distillation from a full active inference planner
    (see ``PlannerDistiller``), so that inference is a single forward pass
    instead of an expensive planning loop.

    Architecture:
        Input: obs (obs_dim) [+ ctx (ctx_dim)] --> MLP backbone (num_layers x
        hidden_dim) --> distribution head.

        - Discrete: logit_head outputs ``(B, action_dim)`` logits.
        - Continuous: mu_head and log_std_head each output ``(B, action_dim)``.

    Args:
        config: AmortizedPolicyConfig controlling architecture and behaviour.

    Example:
        >>> cfg = AmortizedPolicyConfig(obs_dim=64, action_dim=8, hidden_dim=32)
        >>> policy = AmortizedPolicy(cfg)
        >>> obs = torch.randn(4, 64)
        >>> dist = policy(obs)
        >>> action = dist.sample()
    """

    def __init__(self, config: AmortizedPolicyConfig) -> None:
        super().__init__()
        self.config = config

        # Compute input dimensionality: obs + optional ctx
        input_dim = config.obs_dim + config.ctx_dim

        # Build MLP backbone
        self.backbone = _build_mlp(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            activation=config.activation,
            dropout=config.dropout,
            use_layer_norm=config.use_layer_norm,
        )

        # Distribution heads
        self.is_continuous = config.action_type == "continuous"
        if self.is_continuous:
            self.mu_head = nn.Linear(config.hidden_dim, config.action_dim)
            self.log_std_head = nn.Linear(config.hidden_dim, config.action_dim)
            # Initialize output heads with small weights for stable initial policy
            nn.init.normal_(self.mu_head.weight, std=config.init_std)
            nn.init.zeros_(self.mu_head.bias)
            nn.init.normal_(self.log_std_head.weight, std=config.init_std)
            nn.init.constant_(self.log_std_head.bias, -1.0)  # start with low std
        else:
            self.logit_head = nn.Linear(config.hidden_dim, config.action_dim)
            nn.init.normal_(self.logit_head.weight, std=config.init_std)
            nn.init.zeros_(self.logit_head.bias)

        logger.info(
            "AmortizedPolicy: obs_dim=%d, ctx_dim=%d, action_dim=%d, "
            "action_type=%s, hidden=%d x %d layers, params=%d",
            config.obs_dim,
            config.ctx_dim,
            config.action_dim,
            config.action_type,
            config.hidden_dim,
            config.num_layers,
            sum(p.numel() for p in self.parameters()),
        )

    def forward(
        self,
        obs: torch.Tensor,
        ctx: Optional[torch.Tensor] = None,
    ) -> ActionDistribution:
        """Compute the action distribution for given observations.

        Args:
            obs: Observations from the global workspace, shape
                ``(B, obs_dim)``.
            ctx: Optional context tensor, shape ``(B, ctx_dim)``.  Must be
                provided if ``config.ctx_dim > 0``.

        Returns:
            ActionDistribution with the policy's output distribution.

        Raises:
            ValueError: If ctx is required but not provided.
        """
        # Validate input shapes
        B = obs.shape[0]
        assert obs.shape == (B, self.config.obs_dim), (
            f"Expected obs shape (B, {self.config.obs_dim}), got {obs.shape}"
        )

        # Concatenate context if present
        if self.config.ctx_dim > 0:
            if ctx is None:
                raise ValueError(
                    f"Context required (ctx_dim={self.config.ctx_dim}) but ctx is None"
                )
            assert ctx.shape == (B, self.config.ctx_dim), (
                f"Expected ctx shape (B, {self.config.ctx_dim}), got {ctx.shape}"
            )
            x = torch.cat([obs, ctx], dim=-1)
        else:
            x = obs

        # Forward through backbone
        h = self.backbone(x)  # (B, hidden_dim)

        # Compute distribution parameters
        if self.is_continuous:
            mu = self.mu_head(h)  # (B, action_dim)
            log_std = self.log_std_head(h).clamp(MIN_LOG_STD, MAX_LOG_STD)
            return ActionDistribution(
                action_type="continuous",
                mu=mu,
                log_std=log_std,
            )
        else:
            logits = self.logit_head(h)  # (B, action_dim)
            return ActionDistribution(
                action_type="discrete",
                logits=logits,
            )

    def sample(
        self,
        obs: torch.Tensor,
        ctx: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Sample an action from the policy.

        Convenience method that calls ``forward`` and then samples from the
        resulting distribution.

        Args:
            obs: Observations, shape ``(B, obs_dim)``.
            ctx: Optional context, shape ``(B, ctx_dim)``.
            deterministic: If True, return the mode instead of sampling.

        Returns:
            Action tensor.  Shape ``(B, action_dim)`` for continuous actions,
            ``(B,)`` for discrete actions.
        """
        dist = self.forward(obs, ctx)
        return dist.sample(deterministic=deterministic)

    def log_prob(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        ctx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute log-probability of actions under the policy.

        Args:
            obs: Observations, shape ``(B, obs_dim)``.
            action: Actions to score.  Shape ``(B, action_dim)`` for
                continuous or ``(B,)`` for discrete.
            ctx: Optional context, shape ``(B, ctx_dim)``.

        Returns:
            Log-probability per batch element, shape ``(B,)``.
        """
        dist = self.forward(obs, ctx)
        return dist.log_prob(action)

    def get_distribution(
        self,
        obs: torch.Tensor,
        ctx: Optional[torch.Tensor] = None,
    ) -> ActionDistribution:
        """Alias for ``forward`` with a more descriptive name.

        Args:
            obs: Observations, shape ``(B, obs_dim)``.
            ctx: Optional context, shape ``(B, ctx_dim)``.

        Returns:
            ActionDistribution from the policy.
        """
        return self.forward(obs, ctx)

    def parameter_count(self) -> int:
        """Return the total number of trainable parameters.

        Returns:
            Integer count of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Experience Replay Buffer
# ============================================================================


class ReplayBuffer:
    """Fixed-capacity FIFO experience replay buffer for distillation.

    Stores ``Experience`` tuples collected from planner rollouts and provides
    random mini-batch sampling for training the amortized policy.

    Args:
        capacity: Maximum number of experiences.  Once full, oldest
            experiences are evicted in FIFO order.

    Example:
        >>> buf = ReplayBuffer(capacity=1000)
        >>> buf.add(Experience(obs, ctx, action, dist_dict, efe))
        >>> batch = buf.sample(batch_size=32)
    """

    def __init__(self, capacity: int = DEFAULT_BUFFER_CAPACITY) -> None:
        self.capacity = capacity
        self._buffer: Deque[Experience] = collections.deque(maxlen=capacity)

    def add(self, experience: Experience) -> None:
        """Add a single experience to the buffer.

        If the buffer is full, the oldest experience is evicted.

        Args:
            experience: Experience tuple to store.
        """
        self._buffer.append(experience)

    def add_batch(
        self,
        obs: torch.Tensor,
        ctx: torch.Tensor,
        planner_action: torch.Tensor,
        planner_dist: ActionDistribution,
        efe_totals: torch.Tensor,
    ) -> None:
        """Add a batch of experiences to the buffer.

        Unbatches the tensors and stores each element individually.

        Args:
            obs: Observations, shape ``(B, obs_dim)``.
            ctx: Context, shape ``(B, ctx_dim)`` or ``(B, 0)``.
            planner_action: Planner actions, shape ``(B, action_dim)`` or
                ``(B,)`` for discrete.
            planner_dist: Planner's ActionDistribution (batched).
            efe_totals: EFE values, shape ``(B,)``.
        """
        B = obs.shape[0]
        serialized = _serialize_dist(planner_dist)

        for i in range(B):
            # Extract per-element serialized distribution
            per_elem: Dict[str, torch.Tensor] = {}
            for k, v in serialized.items():
                per_elem[k] = v[i]

            exp = Experience(
                obs=obs[i].detach().cpu(),
                ctx=ctx[i].detach().cpu() if ctx.numel() > 0 else torch.empty(0),
                planner_action=planner_action[i].detach().cpu(),
                planner_dist=per_elem,
                efe_total=efe_totals[i].item(),
            )
            self._buffer.append(exp)

    def sample(self, batch_size: int) -> List[Experience]:
        """Sample a random mini-batch from the buffer.

        Args:
            batch_size: Number of experiences to sample.

        Returns:
            List of Experience tuples.

        Raises:
            ValueError: If the buffer contains fewer than ``batch_size``
                experiences.
        """
        if len(self._buffer) < batch_size:
            raise ValueError(
                f"Buffer has {len(self._buffer)} experiences, "
                f"but requested batch_size={batch_size}"
            )
        indices = torch.randint(0, len(self._buffer), (batch_size,))
        return [self._buffer[idx.item()] for idx in indices]

    def sample_tensors(
        self,
        batch_size: int,
        device: torch.device,
        action_type: str,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        ActionDistribution,
        torch.Tensor,
    ]:
        """Sample a mini-batch and collate into stacked tensors.

        Args:
            batch_size: Number of experiences to sample.
            device: Target device for output tensors.
            action_type: ``"continuous"`` or ``"discrete"`` to reconstruct
                the planner distribution.

        Returns:
            Tuple of (obs, ctx, planner_action, planner_dist, efe_totals)
            with batch dimension B = batch_size.
        """
        experiences = self.sample(batch_size)

        obs = torch.stack([e.obs for e in experiences]).to(device)
        ctx = torch.stack([e.ctx for e in experiences]).to(device)
        actions = torch.stack([e.planner_action for e in experiences]).to(device)
        efe_totals = torch.tensor(
            [e.efe_total for e in experiences],
            dtype=torch.float32,
            device=device,
        )

        # Reconstruct batched distribution
        dist_dicts = [e.planner_dist for e in experiences]
        batched_dist_data: Dict[str, torch.Tensor] = {}
        for key in dist_dicts[0]:
            batched_dist_data[key] = torch.stack(
                [d[key] for d in dist_dicts]
            ).to(device)
        planner_dist = _deserialize_dist(batched_dist_data, action_type, device)

        return obs, ctx, actions, planner_dist, efe_totals

    def __len__(self) -> int:
        """Return the current number of stored experiences."""
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        """Return True if the buffer is at capacity."""
        return len(self._buffer) >= self.capacity

    def clear(self) -> None:
        """Remove all experiences from the buffer."""
        self._buffer.clear()


# ============================================================================
# PlannerDistiller
# ============================================================================


class PlannerDistiller:
    """Manages online distillation of the amortized policy from planner outputs.

    The distiller collects experience tuples ``(obs, ctx, planner_action,
    planner_distribution, efe)`` into a replay buffer and trains the amortized
    policy to match the planner's behavior.

    Three distillation modes are supported:

    - **Hard distillation**: Minimize cross-entropy (discrete) or MSE
      (continuous) between the amortized policy output and the planner's
      argmax action.  Simple and stable but discards distribution information.

    - **Soft distillation**: Minimize KL divergence between the planner's
      action distribution and the amortized policy's output distribution.
      Preserves uncertainty information from the planner.

    - **Offline distillation**: Train from a pre-collected dataset, optionally
      reweighting experiences by their EFE values so the policy focuses on
      high-quality transitions.

    Args:
        policy: The AmortizedPolicy to train.
        config: AmortizedPolicyConfig controlling distillation strategy.

    Example:
        >>> policy = AmortizedPolicy(config)
        >>> distiller = PlannerDistiller(policy, config)
        >>> optimizer = torch.optim.Adam(policy.parameters(), lr=config.distill_lr)
        >>> distiller.add_experience(obs, ctx, planner_action, planner_dist, efe)
        >>> loss = distiller.train_step(optimizer)
    """

    def __init__(
        self,
        policy: AmortizedPolicy,
        config: AmortizedPolicyConfig,
    ) -> None:
        self.policy = policy
        self.config = config
        self.buffer = ReplayBuffer(capacity=config.buffer_capacity)
        self._total_train_steps: int = 0

        logger.info(
            "PlannerDistiller: mode=%s, buffer_capacity=%d, batch_size=%d",
            config.distill_mode,
            config.buffer_capacity,
            config.batch_size,
        )

    def add_experience(
        self,
        obs: torch.Tensor,
        ctx: Optional[torch.Tensor],
        planner_action: torch.Tensor,
        planner_dist: Optional[ActionDistribution] = None,
        efe_total: Optional[torch.Tensor] = None,
    ) -> None:
        """Add planner experiences to the replay buffer.

        Accepts both single observations ``(obs_dim,)`` and batched
        observations ``(B, obs_dim)``.  Missing context is replaced with
        a zero-width tensor.

        Args:
            obs: Observations, shape ``(obs_dim,)`` or ``(B, obs_dim)``.
            ctx: Optional context, shape ``(ctx_dim,)`` or ``(B, ctx_dim)``.
                ``None`` produces a zero-width tensor.
            planner_action: Planner action, shape matching obs batch.
            planner_dist: Full planner distribution for soft distillation.
                Required for soft mode; ignored in hard mode but stored
                if available.
            efe_total: Per-element EFE values, shape ``(B,)`` or scalar.
                Used for offline reweighting.
        """
        # Ensure batch dimension
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        B = obs.shape[0]

        if planner_action.dim() == 0:
            planner_action = planner_action.unsqueeze(0)
        if planner_action.dim() == 1 and self.config.action_type == "continuous":
            # Could be (action_dim,) for single sample or (B,) for discrete
            if planner_action.shape[0] == self.config.action_dim and B == 1:
                planner_action = planner_action.unsqueeze(0)

        # Handle context
        if ctx is None:
            ctx = torch.zeros(B, 0)
        elif ctx.dim() == 1:
            if self.config.ctx_dim > 0:
                ctx = ctx.unsqueeze(0)
            else:
                ctx = torch.zeros(B, 0)

        # Handle EFE
        if efe_total is None:
            efe_total = torch.zeros(B)
        elif efe_total.dim() == 0:
            efe_total = efe_total.unsqueeze(0).expand(B)

        # Handle distribution
        if planner_dist is None:
            # Create a dummy distribution for hard distillation
            if self.config.action_type == "discrete":
                planner_dist = ActionDistribution(
                    action_type="discrete",
                    logits=torch.zeros(B, self.config.action_dim),
                )
            else:
                planner_dist = ActionDistribution(
                    action_type="continuous",
                    mu=planner_action.clone().detach(),
                    log_std=torch.zeros_like(planner_action) - 2.0,
                )

        self.buffer.add_batch(obs, ctx, planner_action, planner_dist, efe_total)

    def train_step(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Perform one distillation training step.

        Samples a mini-batch from the replay buffer and updates the amortized
        policy using the configured distillation mode.

        Args:
            optimizer: Optimizer for the amortized policy parameters.

        Returns:
            Scalar loss value for this step.

        Raises:
            ValueError: If the buffer contains fewer experiences than the
                configured batch_size.
        """
        batch_size = min(self.config.batch_size, len(self.buffer))
        if batch_size == 0:
            logger.warning("PlannerDistiller.train_step: buffer is empty")
            return 0.0

        device = next(self.policy.parameters()).device
        obs, ctx, target_actions, planner_dist, efe_totals = (
            self.buffer.sample_tensors(
                batch_size=batch_size,
                device=device,
                action_type=self.config.action_type,
            )
        )

        # Context handling: pass None if ctx_dim == 0
        ctx_input = ctx if self.config.ctx_dim > 0 else None

        # Compute loss based on distillation mode
        if self.config.distill_mode == "hard":
            loss = self._hard_distillation_loss(obs, ctx_input, target_actions)
        elif self.config.distill_mode == "soft":
            loss = self._soft_distillation_loss(obs, ctx_input, planner_dist)
        elif self.config.distill_mode == "offline":
            loss = self._offline_distillation_loss(
                obs, ctx_input, target_actions, planner_dist, efe_totals
            )
        else:
            raise ValueError(f"Unknown distill_mode: {self.config.distill_mode}")

        # Backprop and update
        optimizer.zero_grad()
        loss.backward()
        if self.config.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.config.grad_clip_norm
            )
        optimizer.step()

        self._total_train_steps += 1
        return loss.item()

    def _hard_distillation_loss(
        self,
        obs: torch.Tensor,
        ctx: Optional[torch.Tensor],
        target_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute hard distillation loss: match the planner's argmax action.

        For continuous actions, uses MSE between the policy mean and the
        planner's selected action.

        For discrete actions, uses cross-entropy between the policy logits
        and the planner's selected action index.

        Args:
            obs: Observations, shape ``(B, obs_dim)``.
            ctx: Optional context, shape ``(B, ctx_dim)`` or None.
            target_actions: Planner actions to match.

        Returns:
            Scalar loss tensor.
        """
        policy_dist = self.policy(obs, ctx)

        if self.config.action_type == "continuous":
            assert policy_dist.mu is not None
            loss = F.mse_loss(policy_dist.mu, target_actions)
        else:
            assert policy_dist.logits is not None
            target_idx = target_actions.long()
            if target_idx.dim() > 1:
                target_idx = target_idx.squeeze(-1)
            loss = F.cross_entropy(policy_dist.logits, target_idx)
        return loss

    def _soft_distillation_loss(
        self,
        obs: torch.Tensor,
        ctx: Optional[torch.Tensor],
        planner_dist: ActionDistribution,
    ) -> torch.Tensor:
        """Compute soft distillation loss: match the full planner distribution.

        Minimizes KL(planner || amortized) so the amortized policy covers
        the planner's distribution.

        Args:
            obs: Observations, shape ``(B, obs_dim)``.
            ctx: Optional context, shape ``(B, ctx_dim)`` or None.
            planner_dist: Full action distribution from the planner.

        Returns:
            Scalar loss tensor (mean KL across batch).
        """
        policy_dist = self.policy(obs, ctx)

        # KL(planner || policy) -- the amortized policy is the "q" being fitted
        # to the planner's "p".  We use this direction so the policy covers all
        # modes of the planner distribution.
        kl = planner_dist.kl_divergence(policy_dist)  # (B,)
        return kl.mean()

    def _offline_distillation_loss(
        self,
        obs: torch.Tensor,
        ctx: Optional[torch.Tensor],
        target_actions: torch.Tensor,
        planner_dist: ActionDistribution,
        efe_totals: torch.Tensor,
    ) -> torch.Tensor:
        """Compute offline distillation loss with optional EFE reweighting.

        Combines hard or soft loss with importance weights derived from EFE
        values.  Lower-EFE (better) transitions receive higher weight.

        The importance weights are computed as:
            w_i = softmax(-efe_i / temperature)

        Args:
            obs: Observations, shape ``(B, obs_dim)``.
            ctx: Optional context, shape ``(B, ctx_dim)`` or None.
            target_actions: Planner actions.
            planner_dist: Planner distributions.
            efe_totals: Per-element EFE values, shape ``(B,)``.

        Returns:
            Scalar loss tensor (weighted mean across batch).
        """
        policy_dist = self.policy(obs, ctx)

        # Compute per-element loss
        if self.config.action_type == "continuous":
            assert policy_dist.mu is not None
            per_elem_loss = (
                (policy_dist.mu - target_actions).pow(2).mean(dim=-1)
            )  # (B,)
        else:
            assert policy_dist.logits is not None
            target_idx = target_actions.long()
            if target_idx.dim() > 1:
                target_idx = target_idx.squeeze(-1)
            per_elem_loss = F.cross_entropy(
                policy_dist.logits, target_idx, reduction="none"
            )  # (B,)

        # Compute EFE-based importance weights
        temperature = self.config.efe_reweight_temperature
        if temperature < float("inf"):
            # Lower EFE -> higher weight (negate because EFE is minimized)
            weights = F.softmax(-efe_totals / temperature, dim=0)  # (B,)
            weights = weights * len(weights)  # normalize so E[w] = 1
        else:
            weights = torch.ones_like(per_elem_loss)

        loss = (weights.detach() * per_elem_loss).mean()
        return loss

    @property
    def total_train_steps(self) -> int:
        """Return the total number of training steps performed."""
        return self._total_train_steps

    @property
    def buffer_size(self) -> int:
        """Return the current number of experiences in the buffer."""
        return len(self.buffer)

    def load_offline_dataset(
        self,
        obs: torch.Tensor,
        ctx: Optional[torch.Tensor],
        actions: torch.Tensor,
        efe_totals: Optional[torch.Tensor] = None,
    ) -> int:
        """Load a pre-collected dataset into the replay buffer.

        Intended for offline distillation from a Minari-style dataset.
        Each row is treated as a single experience.

        Args:
            obs: Observations, shape ``(N, obs_dim)``.
            ctx: Optional context, shape ``(N, ctx_dim)`` or None.
            actions: Actions, shape ``(N, action_dim)`` or ``(N,)``.
            efe_totals: Optional EFE values, shape ``(N,)``.

        Returns:
            Number of experiences loaded.
        """
        N = obs.shape[0]
        if efe_totals is None:
            efe_totals = torch.zeros(N)

        for i in range(N):
            obs_i = obs[i].detach().cpu()
            ctx_i = ctx[i].detach().cpu() if ctx is not None else torch.empty(0)
            action_i = actions[i].detach().cpu()

            # Create a point distribution around the stored action
            if self.config.action_type == "continuous":
                dist_data: Dict[str, torch.Tensor] = {
                    "mu": action_i.clone(),
                    "log_std": torch.full_like(action_i, -2.0),
                }
            else:
                logits = torch.full(
                    (self.config.action_dim,), -10.0
                )
                action_idx = int(action_i.item()) if action_i.dim() == 0 else int(action_i[0].item())
                logits[action_idx] = 10.0
                dist_data = {"logits": logits}

            exp = Experience(
                obs=obs_i,
                ctx=ctx_i,
                planner_action=action_i,
                planner_dist=dist_data,
                efe_total=efe_totals[i].item(),
            )
            self.buffer.add(exp)

        logger.info("Loaded %d experiences for offline distillation", N)
        return N


# ============================================================================
# PolicyAuditor
# ============================================================================


class PolicyAuditor:
    """Periodically compares the amortized policy with the full planner.

    The auditor runs the full planner on a batch of observations and compares
    the resulting actions and distributions with those from the amortized
    policy.  This detects distribution shift and degradation over time.

    Metrics computed:
        - **Agreement rate**: Fraction of batch elements where amortized and
          planner actions match (discrete: exact match; continuous: cosine
          similarity > threshold).
        - **EFE gap**: Difference in EFE between amortized and planner
          actions, measuring the quality gap.
        - **Distribution KL**: KL divergence between the two action
          distributions.

    Args:
        config: AmortizedPolicyConfig (uses ``refresh_interval``).
        agreement_threshold: Cosine similarity threshold for continuous
            action agreement.  Default 0.95.

    Example:
        >>> auditor = PolicyAuditor(config)
        >>> if auditor.should_audit(step_count=200):
        ...     result = auditor.audit(agent, obs_batch)
        ...     print(f"Agreement: {result.agreement_rate:.2%}")
    """

    def __init__(
        self,
        config: AmortizedPolicyConfig,
        agreement_threshold: float = 0.95,
    ) -> None:
        self.config = config
        self.agreement_threshold = agreement_threshold
        self._audit_history: List[AuditResult] = []

    def should_audit(self, step_count: int) -> bool:
        """Check whether an audit should be performed at this step.

        Args:
            step_count: Current agent step count.

        Returns:
            True if step_count is a multiple of refresh_interval (and > 0).
        """
        if self.config.refresh_interval <= 0:
            return False
        return step_count > 0 and step_count % self.config.refresh_interval == 0

    @torch.no_grad()
    def audit(
        self,
        planner_fn: Any,
        amortized_policy: AmortizedPolicy,
        obs_batch: torch.Tensor,
        ctx_batch: Optional[torch.Tensor] = None,
        efe_fn: Optional[Any] = None,
    ) -> AuditResult:
        """Run an audit comparing the amortized policy against the planner.

        This method runs both the full planner and the amortized policy on
        the same observation batch, then computes agreement, EFE gap, and
        distribution KL metrics.

        Args:
            planner_fn: Callable that takes ``(obs, ctx)`` and returns a
                tuple ``(action, ActionDistribution, efe_total)`` where
                efe_total is a ``(B,)`` tensor.  This represents the full
                planning loop.
            amortized_policy: The amortized policy to audit.
            obs_batch: Observations, shape ``(B, obs_dim)``.
            ctx_batch: Optional context, shape ``(B, ctx_dim)``.
            efe_fn: Optional callable that takes ``(obs, action)`` and
                returns EFE values ``(B,)``.  Used to independently score
                the amortized policy's actions when the planner does not
                provide separate EFE scoring.

        Returns:
            AuditResult with computed metrics.
        """
        device = obs_batch.device
        B = obs_batch.shape[0]

        # --- Planner ---
        t0 = time.perf_counter()
        planner_action, planner_dist, planner_efe = planner_fn(obs_batch, ctx_batch)
        planner_time = (time.perf_counter() - t0) * 1000.0

        # --- Amortized policy ---
        t0 = time.perf_counter()
        amortized_dist = amortized_policy(obs_batch, ctx_batch)
        amortized_action = amortized_dist.sample(deterministic=True)
        amortized_time = (time.perf_counter() - t0) * 1000.0

        # --- Agreement rate ---
        agreement = self._compute_agreement(
            planner_action, amortized_action, amortized_policy.config.action_type
        )

        # --- EFE gap ---
        if efe_fn is not None:
            amortized_efe = efe_fn(obs_batch, amortized_action)
        else:
            # Use planner EFE as proxy (gap = 0 when not evaluable)
            amortized_efe = planner_efe

        efe_gap = amortized_efe - planner_efe  # positive = worse
        efe_gap_mean = efe_gap.mean().item()
        efe_gap_std = efe_gap.std().item() if B > 1 else 0.0

        # --- Distribution KL ---
        kl = planner_dist.kl_divergence(amortized_dist)  # (B,)
        distribution_kl = kl.mean().item()

        # --- Speedup ---
        speedup = planner_time / max(amortized_time, 1e-6)

        result = AuditResult(
            agreement_rate=agreement,
            efe_gap_mean=efe_gap_mean,
            efe_gap_std=efe_gap_std,
            distribution_kl=distribution_kl,
            batch_size=B,
            planner_time_ms=planner_time,
            amortized_time_ms=amortized_time,
            speedup=speedup,
        )

        self._audit_history.append(result)
        logger.info(
            "Audit: agreement=%.2f, efe_gap=%.4f +/- %.4f, kl=%.4f, speedup=%.1fx",
            agreement,
            efe_gap_mean,
            efe_gap_std,
            distribution_kl,
            speedup,
        )
        return result

    def _compute_agreement(
        self,
        planner_action: torch.Tensor,
        amortized_action: torch.Tensor,
        action_type: str,
    ) -> float:
        """Compute agreement rate between planner and amortized actions.

        Args:
            planner_action: Planner-selected actions.
            amortized_action: Amortized policy actions.
            action_type: ``"continuous"`` or ``"discrete"``.

        Returns:
            Agreement rate in [0, 1].
        """
        if action_type == "discrete":
            # Exact match for discrete actions
            if planner_action.dim() != amortized_action.dim():
                # Handle shape mismatches (e.g. one is (B,1) and other is (B,))
                planner_action = planner_action.reshape(-1)
                amortized_action = amortized_action.reshape(-1)
            matches = (planner_action.long() == amortized_action.long()).float()
            return matches.mean().item()
        else:
            # Cosine similarity for continuous actions
            cos_sim = F.cosine_similarity(planner_action, amortized_action, dim=-1)
            matches = (cos_sim > self.agreement_threshold).float()
            return matches.mean().item()

    @property
    def audit_history(self) -> List[AuditResult]:
        """Return the list of all past audit results."""
        return self._audit_history

    @property
    def last_audit(self) -> Optional[AuditResult]:
        """Return the most recent audit result, or None if no audits."""
        return self._audit_history[-1] if self._audit_history else None


# ============================================================================
# Integration helper: wraps everything into a cohesive facade
# ============================================================================


class AmortizedPolicyManager:
    """High-level manager that wires together policy, distiller, and auditor.

    This facade simplifies integration with the ActiveInferenceAgent by
    providing a single object that handles:
    1. Forwarding through the amortized policy for fast inference.
    2. Collecting distillation data from the planner.
    3. Periodically training the amortized policy.
    4. Auditing the amortized policy against the planner.

    Args:
        config: AmortizedPolicyConfig for all sub-components.
        device: Target device for the policy network.

    Example:
        >>> manager = AmortizedPolicyManager(config, device=torch.device("cpu"))
        >>> # Fast action selection
        >>> action = manager.act(obs, ctx, deterministic=True)
        >>> # After planner produces output, record it
        >>> manager.record_planner_output(obs, ctx, planner_action, planner_dist, efe)
        >>> # Periodically train
        >>> if manager.should_train():
        ...     loss = manager.train()
        >>> # Periodically audit
        >>> if manager.should_audit(step):
        ...     result = manager.audit(planner_fn, obs_batch)
    """

    def __init__(
        self,
        config: AmortizedPolicyConfig,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.config = config
        self.device = device

        # Build components
        self.policy = AmortizedPolicy(config).to(device)
        self.distiller = PlannerDistiller(self.policy, config)
        self.auditor = PolicyAuditor(config)

        # Internal optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.distill_lr
        )

        self._step_count: int = 0
        self._train_loss_history: List[float] = []

    def act(
        self,
        obs: torch.Tensor,
        ctx: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Select an action via the amortized policy.

        Args:
            obs: Observations, shape ``(B, obs_dim)``.
            ctx: Optional context, shape ``(B, ctx_dim)``.
            deterministic: If True, return the mode of the distribution.

        Returns:
            Action tensor.
        """
        was_training = self.policy.training
        self.policy.train(False)
        with torch.no_grad():
            action = self.policy.sample(obs, ctx, deterministic=deterministic)
        self.policy.train(was_training)
        return action

    def get_distribution(
        self,
        obs: torch.Tensor,
        ctx: Optional[torch.Tensor] = None,
    ) -> ActionDistribution:
        """Get the full action distribution from the amortized policy.

        Args:
            obs: Observations, shape ``(B, obs_dim)``.
            ctx: Optional context, shape ``(B, ctx_dim)``.

        Returns:
            ActionDistribution from the policy.
        """
        was_training = self.policy.training
        self.policy.train(False)
        with torch.no_grad():
            dist = self.policy(obs, ctx)
        self.policy.train(was_training)
        return dist

    def record_planner_output(
        self,
        obs: torch.Tensor,
        ctx: Optional[torch.Tensor],
        planner_action: torch.Tensor,
        planner_dist: Optional[ActionDistribution] = None,
        efe_total: Optional[torch.Tensor] = None,
    ) -> None:
        """Record planner output for future distillation.

        Args:
            obs: Observations from the planning step.
            ctx: Context from the planning step.
            planner_action: Action selected by the planner.
            planner_dist: Full planner distribution (for soft distillation).
            efe_total: EFE values (for offline reweighting).
        """
        self.distiller.add_experience(obs, ctx, planner_action, planner_dist, efe_total)
        self._step_count += 1

    def should_train(self, min_buffer_size: Optional[int] = None) -> bool:
        """Check whether we have enough data for a training step.

        Args:
            min_buffer_size: Minimum buffer size before training starts.
                Defaults to batch_size.

        Returns:
            True if training should proceed.
        """
        threshold = min_buffer_size or self.config.batch_size
        return self.distiller.buffer_size >= threshold

    def train(self) -> float:
        """Perform one distillation training step.

        Returns:
            Scalar loss value.
        """
        self.policy.train(True)
        loss = self.distiller.train_step(self.optimizer)
        self._train_loss_history.append(loss)
        return loss

    def should_audit(self, step_count: Optional[int] = None) -> bool:
        """Check whether an audit is due.

        Args:
            step_count: Step count to check.  Defaults to internal counter.

        Returns:
            True if an audit should be performed.
        """
        step = step_count if step_count is not None else self._step_count
        return self.auditor.should_audit(step)

    def audit(
        self,
        planner_fn: Any,
        obs_batch: torch.Tensor,
        ctx_batch: Optional[torch.Tensor] = None,
        efe_fn: Optional[Any] = None,
    ) -> AuditResult:
        """Run an audit comparing amortized policy vs planner.

        Args:
            planner_fn: See PolicyAuditor.audit.
            obs_batch: Observation batch for auditing.
            ctx_batch: Optional context batch.
            efe_fn: Optional EFE evaluation function.

        Returns:
            AuditResult with computed metrics.
        """
        return self.auditor.audit(
            planner_fn=planner_fn,
            amortized_policy=self.policy,
            obs_batch=obs_batch,
            ctx_batch=ctx_batch,
            efe_fn=efe_fn,
        )

    @property
    def step_count(self) -> int:
        """Return the total number of planner outputs recorded."""
        return self._step_count

    @property
    def train_loss_history(self) -> List[float]:
        """Return the history of training losses."""
        return self._train_loss_history


# ============================================================================
# Factory functions
# ============================================================================


def create_amortized_policy(
    obs_dim: int = 4096,
    action_dim: int = 128,
    action_type: str = "continuous",
    hidden_dim: int = 512,
    num_layers: int = 3,
    ctx_dim: int = 0,
    **kwargs: Any,
) -> AmortizedPolicy:
    """Create an AmortizedPolicy with the given parameters.

    Convenience factory that constructs an AmortizedPolicyConfig and builds
    the policy network.

    Args:
        obs_dim: Observation dimensionality.
        action_dim: Action dimensionality.
        action_type: ``"continuous"`` or ``"discrete"``.
        hidden_dim: MLP hidden layer width.
        num_layers: Number of hidden layers.
        ctx_dim: Context dimensionality.  0 disables context.
        **kwargs: Additional AmortizedPolicyConfig fields.

    Returns:
        Configured AmortizedPolicy instance.
    """
    config = AmortizedPolicyConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_type=action_type,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        ctx_dim=ctx_dim,
        **kwargs,
    )
    return AmortizedPolicy(config)


def create_amortized_policy_manager(
    obs_dim: int = 4096,
    action_dim: int = 128,
    action_type: str = "continuous",
    hidden_dim: int = 512,
    num_layers: int = 3,
    ctx_dim: int = 0,
    device: str = "cpu",
    **kwargs: Any,
) -> AmortizedPolicyManager:
    """Create a full AmortizedPolicyManager with policy, distiller, and auditor.

    Args:
        obs_dim: Observation dimensionality.
        action_dim: Action dimensionality.
        action_type: ``"continuous"`` or ``"discrete"``.
        hidden_dim: MLP hidden layer width.
        num_layers: Number of hidden layers.
        ctx_dim: Context dimensionality.  0 disables context.
        device: Device string (e.g. ``"cpu"`` or ``"cuda:0"``).
        **kwargs: Additional AmortizedPolicyConfig fields.

    Returns:
        Configured AmortizedPolicyManager instance.
    """
    config = AmortizedPolicyConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_type=action_type,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        ctx_dim=ctx_dim,
        **kwargs,
    )
    return AmortizedPolicyManager(config, device=torch.device(device))


# ============================================================================
# Mock planner for testing
# ============================================================================


class _MockPlanner:
    """Deterministic mock planner for testing distillation and auditing.

    Implements a simple policy: the action is a deterministic function of the
    observation (linear projection + tanh for continuous, argmax of a linear
    projection for discrete).  This provides a stable target for verifying
    that distillation converges.

    Args:
        obs_dim: Observation dimensionality.
        action_dim: Action dimensionality.
        action_type: ``"continuous"`` or ``"discrete"``.
        ctx_dim: Context dimensionality.
        delay_ms: Artificial delay to simulate planning cost.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_type: str = "continuous",
        ctx_dim: int = 0,
        delay_ms: float = 5.0,
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_type = action_type
        self.ctx_dim = ctx_dim
        self.delay_ms = delay_ms

        # Deterministic linear projection as the "planner's policy"
        torch.manual_seed(42)
        self._proj = torch.randn(obs_dim, action_dim) * 0.1

    def __call__(
        self,
        obs: torch.Tensor,
        ctx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ActionDistribution, torch.Tensor]:
        """Run the mock planner.

        Args:
            obs: Observations, shape ``(B, obs_dim)``.
            ctx: Unused context for API compatibility.

        Returns:
            Tuple of (action, distribution, efe_total).
        """
        # Simulate planning delay
        if self.delay_ms > 0:
            time.sleep(self.delay_ms / 1000.0)

        B = obs.shape[0]
        proj = self._proj.to(obs.device)

        raw = obs @ proj  # (B, action_dim)

        if self.action_type == "continuous":
            mu = torch.tanh(raw)
            log_std = torch.full_like(mu, -1.0)
            action = mu  # deterministic mode
            dist = ActionDistribution(
                action_type="continuous",
                mu=mu,
                log_std=log_std,
            )
        else:
            logits = raw
            action = logits.argmax(dim=-1)
            dist = ActionDistribution(
                action_type="discrete",
                logits=logits,
            )

        # Mock EFE: lower is better, use negative norm as proxy
        efe_total = torch.norm(raw, dim=-1)  # (B,)

        return action, dist, efe_total

    def efe_fn(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Score arbitrary actions by distance from the planner's preferred action.

        Args:
            obs: Observations, shape ``(B, obs_dim)``.
            action: Actions to score.

        Returns:
            EFE values, shape ``(B,)``.
        """
        # Simple EFE: distance from planner's preferred action
        proj = self._proj.to(obs.device)
        preferred = torch.tanh(obs @ proj)
        if self.action_type == "continuous":
            efe = (action - preferred).pow(2).sum(dim=-1)
        else:
            efe = torch.zeros(obs.shape[0], device=obs.device)
        return efe


# ============================================================================
# Self-tests
# ============================================================================


def _run_self_tests() -> None:
    """Run comprehensive self-tests covering all components.

    Tests are grouped by component:
        01-07: AmortizedPolicy (shapes, discrete/continuous, deterministic, gradients, ctx)
        08-11: PlannerDistiller (add experience, train step, buffer management)
        12-15: PolicyAuditor (agreement, EFE gap, should_audit, is_healthy)
        16-18: Integration (distill from mock planner, audit after training)
        19-20: Performance (amortized faster than planner)
        21-23: Additional coverage (log_prob, KL divergence, entropy)

    Each test prints PASS or FAIL.  Raises AssertionError on failure.
    """
    import traceback
    passed = 0
    failed = 0
    total_start = time.perf_counter()

    def _test(name: str, fn: Any) -> None:
        nonlocal passed, failed
        try:
            fn()
            print(f"  PASS: {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {name}: {e}")
            traceback.print_exc()
            failed += 1

    print("=" * 72)
    print("Amortized Policy Distillation -- Self-Tests")
    print("=" * 72)

    device = torch.device("cpu")

    # ------------------------------------------------------------------
    # Test 01: AmortizedPolicy continuous output shapes
    # ------------------------------------------------------------------
    def test_01_continuous_output_shapes() -> None:
        cfg = AmortizedPolicyConfig(
            obs_dim=64, action_dim=8, hidden_dim=32, num_layers=2,
            action_type="continuous",
        )
        policy = AmortizedPolicy(cfg)
        obs = torch.randn(4, 64)
        dist = policy(obs)
        assert dist.action_type == "continuous"
        assert dist.mu is not None and dist.mu.shape == (4, 8), (
            f"Expected mu shape (4, 8), got {dist.mu.shape}"
        )
        assert dist.log_std is not None and dist.log_std.shape == (4, 8), (
            f"Expected log_std shape (4, 8), got {dist.log_std.shape}"
        )
        assert dist.logits is None

    _test("01_continuous_output_shapes", test_01_continuous_output_shapes)

    # ------------------------------------------------------------------
    # Test 02: AmortizedPolicy discrete output shapes
    # ------------------------------------------------------------------
    def test_02_discrete_output_shapes() -> None:
        cfg = AmortizedPolicyConfig(
            obs_dim=64, action_dim=10, hidden_dim=32, num_layers=2,
            action_type="discrete",
        )
        policy = AmortizedPolicy(cfg)
        obs = torch.randn(4, 64)
        dist = policy(obs)
        assert dist.action_type == "discrete"
        assert dist.logits is not None and dist.logits.shape == (4, 10), (
            f"Expected logits shape (4, 10), got {dist.logits.shape}"
        )
        assert dist.mu is None
        assert dist.log_std is None

    _test("02_discrete_output_shapes", test_02_discrete_output_shapes)

    # ------------------------------------------------------------------
    # Test 03: Deterministic sampling returns consistent results
    # ------------------------------------------------------------------
    def test_03_deterministic_sampling() -> None:
        cfg = AmortizedPolicyConfig(
            obs_dim=32, action_dim=8, hidden_dim=16, num_layers=1,
            action_type="continuous",
        )
        policy = AmortizedPolicy(cfg)
        obs = torch.randn(2, 32)
        a1 = policy.sample(obs, deterministic=True)
        a2 = policy.sample(obs, deterministic=True)
        assert torch.allclose(a1, a2, atol=1e-6), "Deterministic samples differ"

    _test("03_deterministic_sampling", test_03_deterministic_sampling)

    # ------------------------------------------------------------------
    # Test 04: Deterministic discrete returns argmax
    # ------------------------------------------------------------------
    def test_04_discrete_deterministic() -> None:
        cfg = AmortizedPolicyConfig(
            obs_dim=32, action_dim=5, hidden_dim=16, num_layers=1,
            action_type="discrete",
        )
        policy = AmortizedPolicy(cfg)
        obs = torch.randn(3, 32)
        dist = policy(obs)
        det_action = dist.sample(deterministic=True)
        expected = dist.logits.argmax(dim=-1)
        assert torch.equal(det_action, expected), (
            f"Deterministic discrete should be argmax: {det_action} vs {expected}"
        )

    _test("04_discrete_deterministic", test_04_discrete_deterministic)

    # ------------------------------------------------------------------
    # Test 05: Gradient flow through continuous policy
    # ------------------------------------------------------------------
    def test_05_gradient_flow_continuous() -> None:
        cfg = AmortizedPolicyConfig(
            obs_dim=32, action_dim=8, hidden_dim=16, num_layers=2,
            action_type="continuous",
        )
        policy = AmortizedPolicy(cfg)
        obs = torch.randn(4, 32)
        dist = policy(obs)
        loss = dist.mu.pow(2).sum()
        loss.backward()
        grad_found = False
        for p in policy.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                grad_found = True
                break
        assert grad_found, "No gradients flowed through continuous policy"

    _test("05_gradient_flow_continuous", test_05_gradient_flow_continuous)

    # ------------------------------------------------------------------
    # Test 06: Gradient flow through discrete policy
    # ------------------------------------------------------------------
    def test_06_gradient_flow_discrete() -> None:
        cfg = AmortizedPolicyConfig(
            obs_dim=32, action_dim=5, hidden_dim=16, num_layers=2,
            action_type="discrete",
        )
        policy = AmortizedPolicy(cfg)
        obs = torch.randn(4, 32)
        dist = policy(obs)
        loss = dist.logits.pow(2).sum()
        loss.backward()
        grad_found = False
        for p in policy.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                grad_found = True
                break
        assert grad_found, "No gradients flowed through discrete policy"

    _test("06_gradient_flow_discrete", test_06_gradient_flow_discrete)

    # ------------------------------------------------------------------
    # Test 07: Context conditioning
    # ------------------------------------------------------------------
    def test_07_context_conditioning() -> None:
        cfg = AmortizedPolicyConfig(
            obs_dim=32, action_dim=8, hidden_dim=16, num_layers=1,
            action_type="continuous", ctx_dim=16,
        )
        policy = AmortizedPolicy(cfg)
        obs = torch.randn(2, 32)
        ctx = torch.randn(2, 16)
        dist_with_ctx = policy(obs, ctx)
        assert dist_with_ctx.mu.shape == (2, 8)

        # Different context should produce different output
        ctx2 = torch.randn(2, 16)
        dist_with_ctx2 = policy(obs, ctx2)
        assert not torch.allclose(dist_with_ctx.mu, dist_with_ctx2.mu, atol=1e-4), (
            "Different contexts produced identical outputs"
        )

    _test("07_context_conditioning", test_07_context_conditioning)

    # ------------------------------------------------------------------
    # Test 08: PlannerDistiller add_experience fills buffer
    # ------------------------------------------------------------------
    def test_08_add_experience() -> None:
        cfg = AmortizedPolicyConfig(
            obs_dim=32, action_dim=8, hidden_dim=16, num_layers=1,
            action_type="continuous", buffer_capacity=100, batch_size=4,
        )
        policy = AmortizedPolicy(cfg)
        distiller = PlannerDistiller(policy, cfg)

        obs = torch.randn(10, 32)
        actions = torch.randn(10, 8)
        dist = ActionDistribution(
            action_type="continuous",
            mu=actions.clone(),
            log_std=torch.zeros(10, 8),
        )
        efe = torch.randn(10)
        distiller.add_experience(obs, None, actions, dist, efe)

        assert distiller.buffer_size == 10, (
            f"Expected buffer size 10, got {distiller.buffer_size}"
        )

    _test("08_add_experience", test_08_add_experience)

    # ------------------------------------------------------------------
    # Test 09: PlannerDistiller train step reduces loss (hard mode)
    # ------------------------------------------------------------------
    def test_09_train_step_hard() -> None:
        cfg = AmortizedPolicyConfig(
            obs_dim=32, action_dim=8, hidden_dim=64, num_layers=2,
            action_type="continuous", distill_mode="hard",
            buffer_capacity=500, batch_size=32, distill_lr=3e-3,
        )
        policy = AmortizedPolicy(cfg)
        distiller = PlannerDistiller(policy, cfg)
        optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.distill_lr)

        # Generate fixed target: action = tanh(obs @ W)
        W = torch.randn(32, 8) * 0.5
        obs = torch.randn(200, 32)
        target = torch.tanh(obs @ W)
        dist = ActionDistribution(
            action_type="continuous",
            mu=target.clone(),
            log_std=torch.full((200, 8), -2.0),
        )
        distiller.add_experience(obs, None, target, dist, torch.zeros(200))

        # Train for enough steps and check loss decreases
        losses = []
        for _ in range(100):
            loss = distiller.train_step(optimizer)
            losses.append(loss)

        assert losses[-1] < losses[0] * 0.5, (
            f"Loss did not decrease sufficiently: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    _test("09_train_step_hard", test_09_train_step_hard)

    # ------------------------------------------------------------------
    # Test 10: PlannerDistiller train step (soft mode)
    # ------------------------------------------------------------------
    def test_10_train_step_soft() -> None:
        cfg = AmortizedPolicyConfig(
            obs_dim=32, action_dim=8, hidden_dim=32, num_layers=2,
            action_type="continuous", distill_mode="soft",
            buffer_capacity=200, batch_size=16, distill_lr=1e-3,
        )
        policy = AmortizedPolicy(cfg)
        distiller = PlannerDistiller(policy, cfg)
        optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.distill_lr)

        W = torch.randn(32, 8) * 0.5
        obs = torch.randn(100, 32)
        target_mu = torch.tanh(obs @ W)
        target_log_std = torch.full((100, 8), -1.0)
        dist = ActionDistribution(
            action_type="continuous",
            mu=target_mu,
            log_std=target_log_std,
        )
        distiller.add_experience(obs, None, target_mu, dist, torch.zeros(100))

        losses = []
        for _ in range(50):
            loss = distiller.train_step(optimizer)
            losses.append(loss)

        assert losses[-1] < losses[0] * 0.8, (
            f"Soft loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    _test("10_train_step_soft", test_10_train_step_soft)

    # ------------------------------------------------------------------
    # Test 11: Buffer FIFO eviction
    # ------------------------------------------------------------------
    def test_11_buffer_fifo() -> None:
        cfg = AmortizedPolicyConfig(
            obs_dim=8, action_dim=4, hidden_dim=8, num_layers=1,
            action_type="continuous", buffer_capacity=10,
        )
        policy = AmortizedPolicy(cfg)
        distiller = PlannerDistiller(policy, cfg)

        # Add 20 experiences to a buffer of capacity 10
        for i in range(20):
            obs = torch.full((1, 8), float(i))
            action = torch.randn(1, 4)
            distiller.add_experience(obs, None, action)

        assert distiller.buffer_size == 10, (
            f"Expected buffer size 10, got {distiller.buffer_size}"
        )
        # Oldest should be evicted -- buffer[0] should have obs value >= 10
        oldest_val = distiller.buffer._buffer[0].obs[0].item()
        assert oldest_val >= 10.0, (
            f"Expected oldest obs >= 10.0, got {oldest_val}"
        )

    _test("11_buffer_fifo", test_11_buffer_fifo)

    # ------------------------------------------------------------------
    # Test 12: PolicyAuditor agreement (discrete)
    # ------------------------------------------------------------------
    def test_12_agreement_discrete() -> None:
        cfg = AmortizedPolicyConfig(
            obs_dim=32, action_dim=5, hidden_dim=16, num_layers=1,
            action_type="discrete", refresh_interval=10,
        )
        auditor = PolicyAuditor(cfg)

        # Create a mock planner that returns fixed logits
        def planner_fn(obs, ctx):
            B = obs.shape[0]
            logits = torch.randn(B, 5)
            action = logits.argmax(dim=-1)
            dist = ActionDistribution(action_type="discrete", logits=logits)
            efe = torch.zeros(B)
            return action, dist, efe

        policy = AmortizedPolicy(cfg)

        obs = torch.randn(8, 32)
        result = auditor.audit(planner_fn, policy, obs)

        assert isinstance(result, AuditResult)
        assert 0.0 <= result.agreement_rate <= 1.0
        assert result.batch_size == 8

    _test("12_agreement_discrete", test_12_agreement_discrete)

    # ------------------------------------------------------------------
    # Test 13: PolicyAuditor EFE gap computation
    # ------------------------------------------------------------------
    def test_13_efe_gap() -> None:
        cfg = AmortizedPolicyConfig(
            obs_dim=16, action_dim=4, hidden_dim=8, num_layers=1,
            action_type="continuous", refresh_interval=5,
        )
        mock = _MockPlanner(
            obs_dim=16, action_dim=4, action_type="continuous", delay_ms=0.0,
        )
        policy = AmortizedPolicy(cfg)
        auditor = PolicyAuditor(cfg)

        obs = torch.randn(4, 16)
        result = auditor.audit(
            planner_fn=mock,
            amortized_policy=policy,
            obs_batch=obs,
            efe_fn=mock.efe_fn,
        )

        # EFE gap should be a finite float
        assert isinstance(result.efe_gap_mean, float)
        assert isinstance(result.efe_gap_std, float)
        assert math.isfinite(result.efe_gap_mean)

    _test("13_efe_gap", test_13_efe_gap)

    # ------------------------------------------------------------------
    # Test 14: should_audit respects refresh_interval
    # ------------------------------------------------------------------
    def test_14_should_audit() -> None:
        cfg = AmortizedPolicyConfig(refresh_interval=50)
        auditor = PolicyAuditor(cfg)

        assert not auditor.should_audit(0), "Should not audit at step 0"
        assert not auditor.should_audit(25), "Should not audit at step 25"
        assert auditor.should_audit(50), "Should audit at step 50"
        assert auditor.should_audit(100), "Should audit at step 100"
        assert not auditor.should_audit(99), "Should not audit at step 99"

    _test("14_should_audit", test_14_should_audit)

    # ------------------------------------------------------------------
    # Test 15: AuditResult.is_healthy
    # ------------------------------------------------------------------
    def test_15_audit_healthy() -> None:
        good = AuditResult(
            agreement_rate=0.9, efe_gap_mean=0.1, efe_gap_std=0.05,
            distribution_kl=0.5, batch_size=10,
            planner_time_ms=50.0, amortized_time_ms=1.0, speedup=50.0,
        )
        assert good.is_healthy(), "Good result should be healthy"

        bad = AuditResult(
            agreement_rate=0.3, efe_gap_mean=5.0, efe_gap_std=2.0,
            distribution_kl=10.0, batch_size=10,
            planner_time_ms=50.0, amortized_time_ms=1.0, speedup=50.0,
        )
        assert not bad.is_healthy(), "Bad result should not be healthy"

    _test("15_audit_healthy", test_15_audit_healthy)

    # ------------------------------------------------------------------
    # Test 16: Integration -- distill from mock planner (continuous)
    # ------------------------------------------------------------------
    def test_16_integration_continuous() -> None:
        obs_dim, action_dim = 32, 8
        cfg = AmortizedPolicyConfig(
            obs_dim=obs_dim, action_dim=action_dim, hidden_dim=64,
            num_layers=2, action_type="continuous", distill_mode="hard",
            buffer_capacity=500, batch_size=32, distill_lr=3e-3,
        )
        mock = _MockPlanner(
            obs_dim=obs_dim, action_dim=action_dim,
            action_type="continuous", delay_ms=0.0,
        )
        manager = AmortizedPolicyManager(cfg)

        # Collect data from planner
        for _ in range(100):
            obs = torch.randn(4, obs_dim)
            action, dist, efe = mock(obs)
            manager.record_planner_output(obs, None, action, dist, efe)

        # Train
        initial_loss = None
        final_loss = 0.0
        for step in range(100):
            loss = manager.train()
            if initial_loss is None:
                initial_loss = loss
            final_loss = loss

        assert final_loss < initial_loss * 0.5, (
            f"Integration loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"
        )

    _test("16_integration_continuous", test_16_integration_continuous)

    # ------------------------------------------------------------------
    # Test 17: Integration -- distill from mock planner (discrete)
    # ------------------------------------------------------------------
    def test_17_integration_discrete() -> None:
        obs_dim, action_dim = 32, 5
        cfg = AmortizedPolicyConfig(
            obs_dim=obs_dim, action_dim=action_dim, hidden_dim=64,
            num_layers=2, action_type="discrete", distill_mode="hard",
            buffer_capacity=500, batch_size=32, distill_lr=3e-3,
        )
        mock = _MockPlanner(
            obs_dim=obs_dim, action_dim=action_dim,
            action_type="discrete", delay_ms=0.0,
        )
        manager = AmortizedPolicyManager(cfg)

        for _ in range(100):
            obs = torch.randn(4, obs_dim)
            action, dist, efe = mock(obs)
            manager.record_planner_output(obs, None, action, dist, efe)

        initial_loss = None
        final_loss = 0.0
        for step in range(100):
            loss = manager.train()
            if initial_loss is None:
                initial_loss = loss
            final_loss = loss

        assert final_loss < initial_loss * 0.7, (
            f"Discrete integration loss did not decrease: "
            f"{initial_loss:.4f} -> {final_loss:.4f}"
        )

    _test("17_integration_discrete", test_17_integration_discrete)

    # ------------------------------------------------------------------
    # Test 18: Audit after training shows improved EFE gap
    # ------------------------------------------------------------------
    def test_18_audit_after_training() -> None:
        obs_dim, action_dim = 32, 8
        cfg = AmortizedPolicyConfig(
            obs_dim=obs_dim, action_dim=action_dim, hidden_dim=64,
            num_layers=2, action_type="continuous", distill_mode="hard",
            buffer_capacity=500, batch_size=32, distill_lr=3e-3,
            refresh_interval=50,
        )
        mock = _MockPlanner(
            obs_dim=obs_dim, action_dim=action_dim,
            action_type="continuous", delay_ms=0.0,
        )
        manager = AmortizedPolicyManager(cfg)

        # Audit before training (random policy, high EFE gap)
        audit_obs = torch.randn(16, obs_dim)
        result_before = manager.audit(mock, audit_obs, efe_fn=mock.efe_fn)

        # Collect data and train
        for _ in range(200):
            obs = torch.randn(4, obs_dim)
            action, dist, efe = mock(obs)
            manager.record_planner_output(obs, None, action, dist, efe)

        for _ in range(200):
            manager.train()

        # Audit after training (should have improved)
        result_after = manager.audit(mock, audit_obs, efe_fn=mock.efe_fn)

        # EFE gap should decrease after training (with some tolerance)
        assert result_after.efe_gap_mean <= result_before.efe_gap_mean + 0.5, (
            f"EFE gap did not improve: before={result_before.efe_gap_mean:.4f}, "
            f"after={result_after.efe_gap_mean:.4f}"
        )

    _test("18_audit_after_training", test_18_audit_after_training)

    # ------------------------------------------------------------------
    # Test 19: Amortized policy is faster than planner (wall clock)
    # ------------------------------------------------------------------
    def test_19_speed_advantage() -> None:
        obs_dim, action_dim = 64, 16
        cfg = AmortizedPolicyConfig(
            obs_dim=obs_dim, action_dim=action_dim, hidden_dim=32,
            num_layers=2, action_type="continuous",
        )
        policy = AmortizedPolicy(cfg)
        mock = _MockPlanner(
            obs_dim=obs_dim, action_dim=action_dim,
            action_type="continuous", delay_ms=5.0,  # 5ms simulated delay
        )

        obs = torch.randn(8, obs_dim)

        # Time the planner
        t0 = time.perf_counter()
        for _ in range(10):
            mock(obs)
        planner_time = time.perf_counter() - t0

        # Time the amortized policy
        was_training = policy.training
        policy.train(False)
        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(10):
                policy.sample(obs, deterministic=True)
            amortized_time = time.perf_counter() - t0
        policy.train(was_training)

        speedup = planner_time / max(amortized_time, 1e-9)
        assert speedup > 2.0, (
            f"Amortized policy not fast enough: speedup={speedup:.1f}x "
            f"(planner={planner_time*1000:.1f}ms, "
            f"amortized={amortized_time*1000:.1f}ms)"
        )

    _test("19_speed_advantage", test_19_speed_advantage)

    # ------------------------------------------------------------------
    # Test 20: Offline distillation with EFE reweighting
    # ------------------------------------------------------------------
    def test_20_offline_distillation() -> None:
        obs_dim, action_dim = 32, 8
        cfg = AmortizedPolicyConfig(
            obs_dim=obs_dim, action_dim=action_dim, hidden_dim=32,
            num_layers=2, action_type="continuous", distill_mode="offline",
            buffer_capacity=500, batch_size=32, distill_lr=3e-3,
            efe_reweight_temperature=0.5,
        )
        policy = AmortizedPolicy(cfg)
        distiller = PlannerDistiller(policy, cfg)
        optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.distill_lr)

        # Create offline dataset
        W = torch.randn(obs_dim, action_dim) * 0.3
        obs = torch.randn(200, obs_dim)
        actions = torch.tanh(obs @ W)
        # EFE: good actions have low EFE
        efe_totals = (actions - torch.tanh(obs @ W)).pow(2).sum(dim=-1)

        distiller.load_offline_dataset(obs, None, actions, efe_totals)
        assert distiller.buffer_size == 200

        losses = []
        for _ in range(50):
            loss = distiller.train_step(optimizer)
            losses.append(loss)

        assert losses[-1] < losses[0] * 0.8, (
            f"Offline loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    _test("20_offline_distillation", test_20_offline_distillation)

    # ------------------------------------------------------------------
    # Test 21: log_prob computation
    # ------------------------------------------------------------------
    def test_21_log_prob() -> None:
        cfg = AmortizedPolicyConfig(
            obs_dim=16, action_dim=4, hidden_dim=16, num_layers=1,
            action_type="continuous",
        )
        policy = AmortizedPolicy(cfg)
        obs = torch.randn(3, 16)
        action = torch.randn(3, 4)
        lp = policy.log_prob(obs, action)
        assert lp.shape == (3,), f"Expected log_prob shape (3,), got {lp.shape}"
        assert torch.isfinite(lp).all(), "log_prob contains non-finite values"

        # Discrete log_prob
        cfg_d = AmortizedPolicyConfig(
            obs_dim=16, action_dim=5, hidden_dim=16, num_layers=1,
            action_type="discrete",
        )
        policy_d = AmortizedPolicy(cfg_d)
        obs_d = torch.randn(3, 16)
        action_d = torch.randint(0, 5, (3,))
        lp_d = policy_d.log_prob(obs_d, action_d)
        assert lp_d.shape == (3,), (
            f"Expected discrete log_prob shape (3,), got {lp_d.shape}"
        )
        assert (lp_d <= 0).all(), "Discrete log_prob should be <= 0"

    _test("21_log_prob", test_21_log_prob)

    # ------------------------------------------------------------------
    # Test 22: ActionDistribution KL divergence
    # ------------------------------------------------------------------
    def test_22_kl_divergence() -> None:
        # Continuous: KL between two different Gaussians
        dist_a = ActionDistribution(
            action_type="continuous",
            mu=torch.zeros(4, 8),
            log_std=torch.zeros(4, 8),
        )
        dist_b = ActionDistribution(
            action_type="continuous",
            mu=torch.ones(4, 8),
            log_std=torch.zeros(4, 8),
        )
        kl = dist_a.kl_divergence(dist_b)
        assert kl.shape == (4,), f"Expected KL shape (4,), got {kl.shape}"
        assert (kl > 0).all(), "KL should be positive for different distributions"

        # KL with itself should be ~0
        kl_self = dist_a.kl_divergence(dist_a)
        assert torch.allclose(kl_self, torch.zeros(4), atol=1e-5), (
            f"KL(p||p) should be ~0, got {kl_self}"
        )

        # Discrete KL
        dist_c = ActionDistribution(
            action_type="discrete",
            logits=torch.randn(3, 5),
        )
        dist_d = ActionDistribution(
            action_type="discrete",
            logits=torch.randn(3, 5),
        )
        kl_disc = dist_c.kl_divergence(dist_d)
        assert kl_disc.shape == (3,)
        assert (kl_disc >= 0).all(), "KL should be non-negative"

    _test("22_kl_divergence", test_22_kl_divergence)

    # ------------------------------------------------------------------
    # Test 23: ActionDistribution entropy
    # ------------------------------------------------------------------
    def test_23_entropy() -> None:
        dist = ActionDistribution(
            action_type="continuous",
            mu=torch.zeros(2, 4),
            log_std=torch.zeros(2, 4),
        )
        ent = dist.entropy()
        assert ent.shape == (2,)
        assert (ent > 0).all(), "Gaussian entropy should be positive"

        # Higher std -> higher entropy
        dist_high_std = ActionDistribution(
            action_type="continuous",
            mu=torch.zeros(2, 4),
            log_std=torch.ones(2, 4),
        )
        ent_high = dist_high_std.entropy()
        assert (ent_high > ent).all(), "Higher std should give higher entropy"

    _test("23_entropy", test_23_entropy)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = (time.perf_counter() - total_start) * 1000.0
    print("=" * 72)
    print(
        f"Results: {passed} passed, {failed} failed, "
        f"{passed + failed} total ({elapsed:.0f}ms)"
    )
    print("=" * 72)

    if failed > 0:
        raise AssertionError(f"{failed} test(s) failed")


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    _run_self_tests()
