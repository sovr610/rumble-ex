"""
Generative Model Components for Active Inference.

Implements the four learned components of the generative model stack required
by the Active Inference decision layer:

    1. Latent Encoder q(s|o)          -- approximate posterior over hidden states
    2. Likelihood Decoder P(o|s)      -- observation model
    3. Transition Model P(s'|s,a)     -- dynamics model (ensemble)
    4. Preferences C                  -- prior over desired observations

Observations arrive as (B, obs_dim=4096) tensors from the global workspace.
Default dimensions: state_dim=256, action_dim=128, hidden_dim=512.

All distributions use diagonal Gaussians in the continuous case. For the
discrete latent case, the encoder produces Gumbel-Softmax categoricals.
Log-variance is always clamped to ``GenerativeModelConfig.log_var_clamp``
for numerical stability.

The top-level ``GenerativeModel`` module wires all four components together
and provides convenience methods including ELBO computation for end-to-end
variational training.

References:
    - Friston et al., "Active Inference: A Process Theory", Neural Computation 2017
    - Fountas et al., "Deep Active Inference Agents Using Monte-Carlo Methods", NeurIPS 2020
    - Catal et al., "Learning Generative State Space Models for Active Inference", 2021
    - Mazzaglia et al., "The Free Energy Principle for Perception and Action", 2022
    - Ha & Schmidhuber, "World Models", NeurIPS 2018 (ensemble transitions)

Typical usage::

    >>> from generative_model_template import GenerativeModel, GenerativeModelConfig
    >>> cfg = GenerativeModelConfig()
    >>> model = GenerativeModel(cfg)
    >>> obs = torch.randn(4, 4096)
    >>> elbo, components = model.compute_elbo(obs)
    >>> print(elbo.shape)  # (4,)
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOG_2PI: float = math.log(2.0 * math.pi)
LOG_2PIE: float = math.log(2.0 * math.pi * math.e)


# ===========================================================================
# Configuration
# ===========================================================================


@dataclass
class GenerativeModelConfig:
    """Configuration for the four-component generative model stack.

    Attributes:
        obs_dim: Workspace observation dimensionality (from global workspace).
        state_dim: Latent state dimensionality.
        action_dim: Action space size (continuous width or discrete count).
        hidden_dim: Width of hidden layers in encoder/decoder/transition MLPs.
        latent_type: ``"continuous"`` for Gaussian, ``"discrete"`` for
            categorical (Gumbel-Softmax).
        num_discrete_states: Number of categories when ``latent_type == "discrete"``.
        encoder_layers: Depth of the latent encoder backbone.
        decoder_layers: Depth of the likelihood decoder backbone.
        transition_ensemble_size: Number of ensemble members K for the
            transition model. K=1 disables ensemble uncertainty estimation.
        transition_layers: Depth of each transition ensemble member.
        use_residual_transition: If True, the transition model predicts
            delta-state (residual) rather than absolute next state.
        log_var_clamp: Tuple ``(min, max)`` for log-variance clamping across
            all distribution heads.
        kl_weight: Beta weight for KL divergence in the ELBO.
        ctx_mode: How to integrate context: ``"concat"``, ``"cross_attention"``,
            or ``"film"`` (Feature-wise Linear Modulation).
        ctx_dim: Context dimensionality. ``None`` or 0 disables context
            conditioning in the encoder.
        preference_mode: ``"fixed"`` (no grad), ``"learned"`` (nn.Parameter),
            or ``"reward_derived"`` (MLP from reward signal).
        preference_reward_dim: Input dimensionality for reward-derived mode.
        action_type: ``"continuous"`` or ``"discrete"``.
        gumbel_temperature: Temperature for Gumbel-Softmax sampling when
            ``latent_type == "discrete"``.
        gumbel_hard: Whether to use straight-through Gumbel-Softmax.
    """

    obs_dim: int = 4096
    state_dim: int = 256
    action_dim: int = 128
    hidden_dim: int = 512

    latent_type: str = "continuous"
    num_discrete_states: int = 32

    encoder_layers: int = 3
    decoder_layers: int = 3
    transition_ensemble_size: int = 5
    transition_layers: int = 2
    use_residual_transition: bool = True

    log_var_clamp: Tuple[float, float] = (-10.0, 2.0)
    kl_weight: float = 1.0

    ctx_mode: str = "concat"
    ctx_dim: Optional[int] = None

    preference_mode: str = "learned"
    preference_reward_dim: int = 1

    action_type: str = "continuous"
    gumbel_temperature: float = 1.0
    gumbel_hard: bool = False

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    @property
    def effective_ctx_dim(self) -> int:
        """Return the effective context dimension (0 if disabled)."""
        if self.ctx_dim is None or self.ctx_dim <= 0:
            return 0
        return self.ctx_dim


# ===========================================================================
# Distribution containers
# ===========================================================================


@dataclass
class LatentDistribution:
    """Container for a diagonal Gaussian or categorical latent distribution.

    For continuous latent spaces, holds ``mu`` and ``log_var`` and provides
    reparameterized sampling and closed-form KL divergence to a standard
    normal prior.

    For discrete latent spaces, ``mu`` stores the logits and ``log_var`` is
    unused (set to zeros).

    Attributes:
        mu: Mean (continuous) or logits (discrete), shape ``(B, state_dim)``
            or ``(B, num_categories)``.
        log_var: Log-variance of the distribution, shape ``(B, state_dim)``.
            Zeros for discrete distributions.
        sample: A reparameterized sample, shape ``(B, state_dim)``.
        is_discrete: Whether this is a discrete (Gumbel-Softmax) distribution.
    """

    mu: torch.Tensor
    log_var: torch.Tensor
    sample: torch.Tensor
    is_discrete: bool = False

    def kl_divergence(self, prior_mu: Optional[torch.Tensor] = None,
                      prior_log_var: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute KL divergence from this distribution to a prior.

        For continuous distributions, computes the closed-form KL divergence
        KL(q || p) between two diagonal Gaussians. If no prior is specified,
        the standard normal N(0, I) is used.

        For discrete distributions, computes KL from the categorical to a
        uniform prior over categories.

        Args:
            prior_mu: Prior mean, shape ``(B, state_dim)``. Defaults to zeros.
            prior_log_var: Prior log-variance, shape ``(B, state_dim)``.
                Defaults to zeros (unit variance).

        Returns:
            KL divergence per batch element, shape ``(B,)``.
        """
        if self.is_discrete:
            # KL from categorical to uniform
            logits = self.mu
            q = F.softmax(logits, dim=-1)
            log_q = F.log_softmax(logits, dim=-1)
            num_categories = logits.shape[-1]
            log_uniform = -math.log(num_categories)
            kl = (q * (log_q - log_uniform)).sum(dim=-1)
            return kl

        # Continuous: closed-form KL for diagonal Gaussians
        mu_q = self.mu
        lv_q = self.log_var

        if prior_mu is None:
            mu_p = torch.zeros_like(mu_q)
        else:
            mu_p = prior_mu
        if prior_log_var is None:
            lv_p = torch.zeros_like(lv_q)
        else:
            lv_p = prior_log_var

        # KL(q || p) = 0.5 * sum[ exp(lv_q - lv_p)
        #                         + (mu_p - mu_q)^2 / exp(lv_p)
        #                         - 1
        #                         + lv_p - lv_q ]
        var_ratio = torch.exp(lv_q - lv_p)
        mean_sq = (mu_p - mu_q).pow(2) * torch.exp(-lv_p)
        kl_per_dim = 0.5 * (var_ratio + mean_sq - 1.0 + lv_p - lv_q)
        return kl_per_dim.sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        """Compute differential entropy of the distribution.

        For a diagonal Gaussian: H = 0.5 * sum_d [log(2*pi*e) + log_var_d]

        Returns:
            Entropy per batch element, shape ``(B,)``.
        """
        if self.is_discrete:
            logits = self.mu
            q = F.softmax(logits, dim=-1)
            log_q = F.log_softmax(logits, dim=-1)
            return -(q * log_q).sum(dim=-1)

        return 0.5 * (LOG_2PIE + self.log_var).sum(dim=-1)


@dataclass
class TransitionDistribution:
    """Container for the ensemble transition model output.

    Stores predictions from K ensemble members and provides methods
    for sampling, uncertainty decomposition, and sequence prediction.

    Attributes:
        means: Predicted means from each ensemble member,
            shape ``(K, B, state_dim)``.
        log_vars: Predicted log-variances from each ensemble member,
            shape ``(K, B, state_dim)``.
        ensemble_size: Number of ensemble members K.
    """

    means: torch.Tensor
    log_vars: torch.Tensor
    ensemble_size: int

    def sample(self, member_idx: Optional[int] = None) -> torch.Tensor:
        """Draw a reparameterized sample from the transition distribution.

        If ``member_idx`` is specified, samples from that single ensemble
        member. Otherwise, randomly selects one member per batch element
        (Thompson sampling).

        Args:
            member_idx: Optional index of a specific ensemble member.

        Returns:
            Sampled next state, shape ``(B, state_dim)``.
        """
        if member_idx is not None:
            mu = self.means[member_idx]
            lv = self.log_vars[member_idx]
        else:
            # Thompson sampling: random member per batch element
            K, B, D = self.means.shape
            indices = torch.randint(0, K, (B,), device=self.means.device)
            mu = self.means[indices, torch.arange(B, device=self.means.device)]
            lv = self.log_vars[indices, torch.arange(B, device=self.means.device)]

        std = torch.exp(0.5 * lv)
        eps = torch.randn_like(std)
        return mu + std * eps

    def mean_prediction(self) -> torch.Tensor:
        """Return the mean of the ensemble means, shape ``(B, state_dim)``."""
        return self.means.mean(dim=0)

    def epistemic_uncertainty(self) -> torch.Tensor:
        """Compute epistemic (model) uncertainty.

        Epistemic uncertainty is the variance of the ensemble means --
        it captures disagreement between models about the expected
        next state. High epistemic uncertainty indicates the model is
        uncertain because of insufficient data.

        Returns:
            Per-dimension epistemic uncertainty, shape ``(B, state_dim)``.
        """
        return self.means.var(dim=0)

    def aleatoric_uncertainty(self) -> torch.Tensor:
        """Compute aleatoric (data) uncertainty.

        Aleatoric uncertainty is the mean of the ensemble variances --
        it captures irreducible noise in the transition dynamics that
        persists regardless of the amount of training data.

        Returns:
            Per-dimension aleatoric uncertainty, shape ``(B, state_dim)``.
        """
        variances = torch.exp(self.log_vars)
        return variances.mean(dim=0)

    def total_uncertainty(self) -> torch.Tensor:
        """Compute total predictive uncertainty (epistemic + aleatoric).

        Returns:
            Per-dimension total uncertainty, shape ``(B, state_dim)``.
        """
        return self.epistemic_uncertainty() + self.aleatoric_uncertainty()

    def scalar_epistemic_uncertainty(self) -> torch.Tensor:
        """Scalar epistemic uncertainty per batch element, shape ``(B,)``."""
        return self.epistemic_uncertainty().sum(dim=-1)

    def scalar_aleatoric_uncertainty(self) -> torch.Tensor:
        """Scalar aleatoric uncertainty per batch element, shape ``(B,)``."""
        return self.aleatoric_uncertainty().sum(dim=-1)


# ===========================================================================
# Building blocks
# ===========================================================================


def _build_residual_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    num_layers: int,
    activation: str = "silu",
) -> nn.Module:
    """Build an MLP with residual connections and LayerNorm.

    Architecture for each hidden layer::

        x -> Linear -> LayerNorm -> Activation -> residual add

    The first layer projects from ``input_dim`` to ``hidden_dim``, and
    subsequent layers maintain ``hidden_dim``. The final linear projects
    from ``hidden_dim`` to ``output_dim``.

    Args:
        input_dim: Input feature dimensionality.
        output_dim: Output feature dimensionality.
        hidden_dim: Width of hidden layers.
        num_layers: Number of hidden layers (minimum 1).
        activation: Activation function name (``"silu"`` or ``"relu"``).

    Returns:
        An ``nn.Sequential`` module.
    """
    act_fn = nn.SiLU if activation == "silu" else nn.ReLU
    layers: List[nn.Module] = []

    # Input projection
    layers.append(nn.Linear(input_dim, hidden_dim))
    layers.append(nn.LayerNorm(hidden_dim))
    layers.append(act_fn())

    # Residual hidden layers
    for _ in range(num_layers - 1):
        layers.append(_ResidualBlock(hidden_dim, act_fn))

    # Output projection
    layers.append(nn.Linear(hidden_dim, output_dim))

    return nn.Sequential(*layers)


class _ResidualBlock(nn.Module):
    """Single residual block: Linear -> LayerNorm -> Activation -> add.

    Used internally by ``_build_residual_mlp`` to construct deep MLPs with
    skip connections that mitigate vanishing gradients.
    """

    def __init__(self, dim: int, act_cls: type = nn.SiLU) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            act_cls(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class _CrossAttentionConditioner(nn.Module):
    """Cross-attention context conditioning module.

    Uses the latent hidden representation as queries and the context
    as keys/values via standard scaled dot-product attention.

    Args:
        hidden_dim: Dimensionality of the hidden representation (query).
        ctx_dim: Dimensionality of the context (key/value).
        num_heads: Number of attention heads.
    """

    def __init__(self, hidden_dim: int, ctx_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            kdim=ctx_dim,
            vdim=ctx_dim,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """Apply cross-attention conditioning.

        Args:
            hidden: Query tensor, shape ``(B, hidden_dim)``.
            ctx: Key/value tensor, shape ``(B, ctx_dim)``.

        Returns:
            Conditioned hidden tensor, shape ``(B, hidden_dim)``.
        """
        # Add sequence dimension for MHA: (B, 1, D)
        h = hidden.unsqueeze(1)
        c = ctx.unsqueeze(1)
        attended, _ = self.attn(h, c, c)
        out = self.norm(hidden + attended.squeeze(1))
        return out


class _FiLMConditioner(nn.Module):
    """Feature-wise Linear Modulation (FiLM) context conditioning.

    Projects the context into scale (gamma) and shift (beta) parameters
    that modulate the hidden representation element-wise:

        output = gamma * hidden + beta

    Args:
        hidden_dim: Dimensionality of the hidden representation.
        ctx_dim: Dimensionality of the context.
    """

    def __init__(self, hidden_dim: int, ctx_dim: int) -> None:
        super().__init__()
        self.gamma_proj = nn.Linear(ctx_dim, hidden_dim)
        self.beta_proj = nn.Linear(ctx_dim, hidden_dim)

        # Initialize gamma near 1 and beta near 0 for near-identity init
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)

    def forward(self, hidden: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """Apply FiLM conditioning.

        Args:
            hidden: Feature tensor, shape ``(B, hidden_dim)``.
            ctx: Context tensor, shape ``(B, ctx_dim)``.

        Returns:
            Modulated feature tensor, shape ``(B, hidden_dim)``.
        """
        gamma = self.gamma_proj(ctx)
        beta = self.beta_proj(ctx)
        return gamma * hidden + beta


# ===========================================================================
# Component 1: Latent Encoder q(s|o)
# ===========================================================================


class LatentEncoder(nn.Module):
    """Approximate posterior encoder q(s|o) mapping observations to latent states.

    Implements a multi-layer MLP backbone with residual connections, LayerNorm,
    and SiLU activation. Produces parameters of a diagonal Gaussian (continuous)
    or categorical (discrete, via Gumbel-Softmax) distribution over the latent
    state space.

    Optional context conditioning supports three modes:
        - ``"concat"``: Concatenate context to the input before encoding.
        - ``"cross_attention"``: Apply cross-attention between hidden features
          and the context.
        - ``"film"``: Feature-wise Linear Modulation of hidden features.

    Args:
        config: Generative model configuration.
    """

    def __init__(self, config: GenerativeModelConfig) -> None:
        super().__init__()
        self.config = config
        self.obs_dim = config.obs_dim
        self.state_dim = config.state_dim
        self.hidden_dim = config.hidden_dim
        self.latent_type = config.latent_type
        self.ctx_mode = config.ctx_mode
        self.ctx_dim = config.effective_ctx_dim

        # Determine input dimension based on context mode
        input_dim = self.obs_dim
        if self.ctx_mode == "concat" and self.ctx_dim > 0:
            input_dim += self.ctx_dim

        # Backbone MLP with residual connections
        act_fn = nn.SiLU
        backbone_layers: List[nn.Module] = []
        backbone_layers.append(nn.Linear(input_dim, self.hidden_dim))
        backbone_layers.append(nn.LayerNorm(self.hidden_dim))
        backbone_layers.append(act_fn())
        for _ in range(config.encoder_layers - 1):
            backbone_layers.append(_ResidualBlock(self.hidden_dim, act_fn))
        self.backbone = nn.Sequential(*backbone_layers)

        # Context conditioning modules (non-concat modes)
        self.ctx_conditioner: Optional[nn.Module] = None
        if self.ctx_dim > 0 and self.ctx_mode == "cross_attention":
            self.ctx_conditioner = _CrossAttentionConditioner(
                self.hidden_dim, self.ctx_dim
            )
        elif self.ctx_dim > 0 and self.ctx_mode == "film":
            self.ctx_conditioner = _FiLMConditioner(self.hidden_dim, self.ctx_dim)

        # Distribution heads
        if self.latent_type == "continuous":
            self.mu_head = nn.Linear(self.hidden_dim, self.state_dim)
            self.log_var_head = nn.Linear(self.hidden_dim, self.state_dim)
            # Initialize log_var head to output small variances
            nn.init.zeros_(self.log_var_head.weight)
            nn.init.constant_(self.log_var_head.bias, -2.0)
        elif self.latent_type == "discrete":
            self.logits_head = nn.Linear(
                self.hidden_dim, config.num_discrete_states
            )
        else:
            raise ValueError(
                f"Unknown latent_type: {self.latent_type}. "
                f"Must be 'continuous' or 'discrete'."
            )

    def forward(
        self,
        obs: torch.Tensor,
        ctx: Optional[torch.Tensor] = None,
    ) -> LatentDistribution:
        """Encode observations into a latent distribution.

        Args:
            obs: Observations from the workspace, shape ``(B, obs_dim)``.
            ctx: Optional context tensor, shape ``(B, ctx_dim)``. Ignored if
                ``ctx_dim`` is 0 in the config.

        Returns:
            A ``LatentDistribution`` containing the posterior parameters and
            a reparameterized sample.
        """
        x = obs

        # Context conditioning: concat mode
        if self.ctx_mode == "concat" and self.ctx_dim > 0 and ctx is not None:
            x = torch.cat([x, ctx], dim=-1)

        # Backbone forward pass
        h = self.backbone(x)

        # Context conditioning: cross-attention or FiLM
        if self.ctx_conditioner is not None and ctx is not None:
            h = self.ctx_conditioner(h, ctx)

        # Produce distribution parameters and sample
        if self.latent_type == "continuous":
            mu = self.mu_head(h)
            log_var_raw = self.log_var_head(h)
            log_var = torch.clamp(
                log_var_raw,
                min=self.config.log_var_clamp[0],
                max=self.config.log_var_clamp[1],
            )
            # Reparameterization trick
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            sample = mu + std * eps
            return LatentDistribution(
                mu=mu, log_var=log_var, sample=sample, is_discrete=False
            )
        else:
            # Discrete: Gumbel-Softmax
            logits = self.logits_head(h)
            if self.training:
                sample = F.gumbel_softmax(
                    logits,
                    tau=self.config.gumbel_temperature,
                    hard=self.config.gumbel_hard,
                )
            else:
                # Hard argmax at eval time
                idx = logits.argmax(dim=-1)
                sample = F.one_hot(
                    idx, num_classes=self.config.num_discrete_states
                ).float()

            log_var = torch.zeros_like(logits)
            return LatentDistribution(
                mu=logits, log_var=log_var, sample=sample, is_discrete=True
            )


# ===========================================================================
# Component 2: Likelihood Decoder P(o|s)
# ===========================================================================


class LikelihoodDecoder(nn.Module):
    """Observation likelihood model P(o|s) mapping latent states to observations.

    Mirror architecture of the encoder: multi-layer MLP with residual
    connections, LayerNorm, and SiLU activation. Outputs diagonal Gaussian
    parameters (mean, log_var) over the observation space.

    Provides methods for:
        - Forward pass producing distribution parameters
        - Log-probability evaluation
        - Reconstruction loss computation

    Args:
        config: Generative model configuration.
    """

    def __init__(self, config: GenerativeModelConfig) -> None:
        super().__init__()
        self.config = config
        self.obs_dim = config.obs_dim
        self.state_dim = config.state_dim
        self.hidden_dim = config.hidden_dim

        # Determine input dimension (continuous state or discrete one-hot)
        if config.latent_type == "discrete":
            input_dim = config.num_discrete_states
        else:
            input_dim = config.state_dim

        # Backbone MLP with residual connections
        act_fn = nn.SiLU
        backbone_layers: List[nn.Module] = []
        backbone_layers.append(nn.Linear(input_dim, self.hidden_dim))
        backbone_layers.append(nn.LayerNorm(self.hidden_dim))
        backbone_layers.append(act_fn())
        for _ in range(config.decoder_layers - 1):
            backbone_layers.append(_ResidualBlock(self.hidden_dim, act_fn))
        self.backbone = nn.Sequential(*backbone_layers)

        # Output heads
        self.mean_head = nn.Linear(self.hidden_dim, self.obs_dim)
        self.log_var_head = nn.Linear(self.hidden_dim, self.obs_dim)
        # Initialize log_var to moderate values
        nn.init.zeros_(self.log_var_head.weight)
        nn.init.constant_(self.log_var_head.bias, -1.0)

    def forward(
        self, latent_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode latent state to observation distribution parameters.

        Args:
            latent_state: Latent state sample, shape ``(B, state_dim)`` or
                ``(B, num_discrete_states)`` for discrete latents.

        Returns:
            Tuple of (mean, log_var), each shape ``(B, obs_dim)``.
        """
        h = self.backbone(latent_state)
        mean = self.mean_head(h)
        log_var_raw = self.log_var_head(h)
        log_var = torch.clamp(
            log_var_raw,
            min=self.config.log_var_clamp[0],
            max=self.config.log_var_clamp[1],
        )
        return mean, log_var

    def log_prob(
        self, obs: torch.Tensor, latent_state: torch.Tensor
    ) -> torch.Tensor:
        """Compute log-probability of observations under the likelihood model.

        Evaluates log P(o|s) using the diagonal Gaussian density:

            log p(o_d | s) = -0.5 * [log(2*pi) + log_var_d + (o_d - mu_d)^2 / var_d]

        Summed over observation dimensions.

        Args:
            obs: Observed data, shape ``(B, obs_dim)``.
            latent_state: Latent state, shape ``(B, state_dim)``.

        Returns:
            Log-probability per batch element, shape ``(B,)``.
        """
        mean, log_var = self.forward(latent_state)
        var = torch.exp(log_var)

        # Gaussian log-probability per dimension
        log_prob_per_dim = -0.5 * (
            LOG_2PI + log_var + (obs - mean).pow(2) / (var + 1e-8)
        )
        return log_prob_per_dim.sum(dim=-1)

    def reconstruction_loss(
        self, obs: torch.Tensor, latent_state: torch.Tensor
    ) -> torch.Tensor:
        """Compute reconstruction loss (negative log-likelihood).

        This is the mean negative log-likelihood across the batch, suitable
        as a training loss. It is equivalent to the negative of the mean
        log_prob.

        Args:
            obs: Target observations, shape ``(B, obs_dim)``.
            latent_state: Latent state, shape ``(B, state_dim)``.

        Returns:
            Scalar reconstruction loss.
        """
        log_p = self.log_prob(obs, latent_state)
        return -log_p.mean()

    def sample(self, latent_state: torch.Tensor) -> torch.Tensor:
        """Sample an observation from the likelihood model.

        Args:
            latent_state: Latent state, shape ``(B, state_dim)``.

        Returns:
            Sampled observation, shape ``(B, obs_dim)``.
        """
        mean, log_var = self.forward(latent_state)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + std * eps


# ===========================================================================
# Component 3: Transition Model P(s'|s,a) -- Ensemble
# ===========================================================================


class _TransitionMember(nn.Module):
    """Single member of the transition ensemble.

    Predicts next-state distribution parameters (mean, log_var) given
    the current state and action. Optionally predicts a residual delta
    rather than the absolute next state.

    Args:
        config: Generative model configuration.
    """

    def __init__(self, config: GenerativeModelConfig) -> None:
        super().__init__()
        self.config = config
        self.use_residual = config.use_residual_transition

        # Action embedding for discrete actions
        if config.action_type == "discrete":
            self.action_embed = nn.Embedding(config.action_dim, config.hidden_dim)
            input_dim = config.state_dim + config.hidden_dim
        else:
            input_dim = config.state_dim + config.action_dim

        # MLP backbone
        act_fn = nn.SiLU
        layers: List[nn.Module] = []
        layers.append(nn.Linear(input_dim, config.hidden_dim))
        layers.append(nn.LayerNorm(config.hidden_dim))
        layers.append(act_fn())
        for _ in range(config.transition_layers - 1):
            layers.append(_ResidualBlock(config.hidden_dim, act_fn))
        self.backbone = nn.Sequential(*layers)

        # Output heads
        self.mu_head = nn.Linear(config.hidden_dim, config.state_dim)
        self.log_var_head = nn.Linear(config.hidden_dim, config.state_dim)

        # Initialize for small initial predictions
        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.log_var_head.weight)
        nn.init.constant_(self.log_var_head.bias, -2.0)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict next-state distribution for a single ensemble member.

        Args:
            state: Current latent state, shape ``(B, state_dim)``.
            action: Action tensor. For continuous actions, shape
                ``(B, action_dim)``. For discrete actions, shape ``(B,)``
                with integer indices.

        Returns:
            Tuple of (mean, log_var) for the predicted next state,
            each shape ``(B, state_dim)``.
        """
        if self.config.action_type == "discrete":
            action_feat = self.action_embed(action.long())
        else:
            action_feat = action

        x = torch.cat([state, action_feat], dim=-1)
        h = self.backbone(x)

        mu = self.mu_head(h)
        log_var_raw = self.log_var_head(h)
        log_var = torch.clamp(
            log_var_raw,
            min=self.config.log_var_clamp[0],
            max=self.config.log_var_clamp[1],
        )

        # Residual prediction: next_state = state + delta
        if self.use_residual:
            mu = state + mu

        return mu, log_var


class TransitionModel(nn.Module):
    """Ensemble transition model P(s'|s,a).

    Maintains K independent transition networks (ensemble members) that
    each predict a Gaussian distribution over the next state. The ensemble
    provides:

        - **Epistemic uncertainty** via disagreement between members (variance
          of the means).
        - **Aleatoric uncertainty** via the average predicted variance.
        - **Thompson sampling** for exploration by randomly selecting a member.
        - **Multi-step rollouts** via ``predict_sequence``.

    Args:
        config: Generative model configuration.
    """

    def __init__(self, config: GenerativeModelConfig) -> None:
        super().__init__()
        self.config = config
        self.ensemble_size = config.transition_ensemble_size

        # Create ensemble members
        self.members = nn.ModuleList([
            _TransitionMember(config) for _ in range(self.ensemble_size)
        ])

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        deterministic: bool = False,
    ) -> TransitionDistribution:
        """Predict next-state distribution using the full ensemble.

        Args:
            state: Current latent state, shape ``(B, state_dim)``.
            action: Action tensor. Continuous: ``(B, action_dim)``.
                Discrete: ``(B,)`` integer indices.
            deterministic: If True, the ``sample`` method of the returned
                distribution will use the ensemble mean instead of
                Thompson sampling.

        Returns:
            A ``TransitionDistribution`` containing predictions from all
            ensemble members.
        """
        means_list = []
        log_vars_list = []

        for member in self.members:
            mu, lv = member(state, action)
            means_list.append(mu)
            log_vars_list.append(lv)

        means = torch.stack(means_list, dim=0)      # (K, B, state_dim)
        log_vars = torch.stack(log_vars_list, dim=0)  # (K, B, state_dim)

        return TransitionDistribution(
            means=means,
            log_vars=log_vars,
            ensemble_size=self.ensemble_size,
        )

    def predict_sequence(
        self,
        initial_state: torch.Tensor,
        action_sequence: torch.Tensor,
        member_idx: Optional[int] = None,
    ) -> Tuple[List[torch.Tensor], List[TransitionDistribution]]:
        """Predict a multi-step state trajectory by rolling out the model.

        At each step, a state is sampled from the transition distribution
        and used as input for the next step. This enables planning over
        extended horizons.

        Args:
            initial_state: Starting state, shape ``(B, state_dim)``.
            action_sequence: Sequence of actions, shape ``(T, B, action_dim)``
                for continuous actions or ``(T, B)`` for discrete.
            member_idx: If specified, use a fixed ensemble member for all
                steps. Otherwise, Thompson sampling is used at each step.

        Returns:
            Tuple of:
                - List of sampled states for each timestep ``[s_1, ..., s_T]``,
                  each shape ``(B, state_dim)``.
                - List of ``TransitionDistribution`` objects for each step.
        """
        state = initial_state
        states: List[torch.Tensor] = []
        distributions: List[TransitionDistribution] = []

        T = action_sequence.shape[0]
        for t in range(T):
            action_t = action_sequence[t]
            dist = self.forward(state, action_t)
            distributions.append(dist)
            state = dist.sample(member_idx=member_idx)
            states.append(state)

        return states, distributions


# ===========================================================================
# Component 4: Preferences C
# ===========================================================================


class Preferences(nn.Module):
    """Prior preferences C over observations.

    Represents the agent's desired observation distribution. Preferences
    define what constitutes a "good" outcome and drive the pragmatic
    component of Expected Free Energy.

    Three modes of operation:

        - ``"fixed"``: Preference parameters are registered as buffers
          (no gradient). Useful for hand-specified goals.
        - ``"learned"``: Preference parameters are ``nn.Parameter`` objects
          that receive gradients. The model discovers what it prefers.
        - ``"reward_derived"``: An MLP maps external reward signals to
          preference parameters. Bridges RL reward shaping with Active
          Inference.

    The preference distribution is a diagonal Gaussian N(mean, 1/precision).

    Args:
        config: Generative model configuration.
    """

    def __init__(self, config: GenerativeModelConfig) -> None:
        super().__init__()
        self.config = config
        self.obs_dim = config.obs_dim
        self.mode = config.preference_mode

        if self.mode == "fixed":
            # No gradient: registered as buffers
            self.register_buffer(
                "pref_mean", torch.zeros(config.obs_dim)
            )
            self.register_buffer(
                "pref_log_precision", torch.zeros(config.obs_dim)
            )

        elif self.mode == "learned":
            # Learnable parameters
            self.pref_mean = nn.Parameter(
                torch.randn(config.obs_dim) * 0.01
            )
            self.pref_log_precision = nn.Parameter(
                torch.zeros(config.obs_dim)
            )

        elif self.mode == "reward_derived":
            # MLP mapping reward -> preference parameters
            reward_dim = config.preference_reward_dim
            self.reward_encoder = nn.Sequential(
                nn.Linear(reward_dim, config.hidden_dim),
                nn.SiLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.SiLU(),
            )
            self.reward_to_mean = nn.Linear(config.hidden_dim, config.obs_dim)
            self.reward_to_log_prec = nn.Linear(
                config.hidden_dim, config.obs_dim
            )
            # Buffer for the most recent reward-derived preferences
            self.register_buffer(
                "_cached_mean", torch.zeros(config.obs_dim)
            )
            self.register_buffer(
                "_cached_log_precision", torch.zeros(config.obs_dim)
            )

        else:
            raise ValueError(
                f"Unknown preference_mode: {self.mode}. "
                f"Must be 'fixed', 'learned', or 'reward_derived'."
            )

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return current preference distribution parameters.

        Returns:
            Tuple of (mean, precision):
                - mean: Preferred observation mean, shape ``(obs_dim,)``.
                - precision: Preference precision (inverse variance),
                  shape ``(obs_dim,)``.
        """
        if self.mode == "fixed" or self.mode == "learned":
            mean = self.pref_mean
            precision = torch.exp(self.pref_log_precision)
        elif self.mode == "reward_derived":
            mean = self._cached_mean
            precision = torch.exp(self._cached_log_precision)
        else:
            raise RuntimeError(f"Invalid preference mode: {self.mode}")

        return mean, precision

    def log_prob(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute log-probability of observations under the preference prior.

        Evaluates log C(o) = log N(o; pref_mean, 1/precision) summed over
        observation dimensions.

        Args:
            obs: Observations, shape ``(B, obs_dim)``.

        Returns:
            Log-probability per batch element, shape ``(B,)``.
        """
        mean, precision = self.forward()

        # Gaussian log-probability per dimension
        # log N(o; mu, sigma^2) = -0.5 * [log(2*pi) - log(prec) + prec*(o-mu)^2]
        sq_diff = (obs - mean.unsqueeze(0)).pow(2)
        log_p_per_dim = -0.5 * (
            LOG_2PI - torch.log(precision + 1e-8).unsqueeze(0) + precision.unsqueeze(0) * sq_diff
        )
        return log_p_per_dim.sum(dim=-1)

    def update_from_reward(self, reward: torch.Tensor) -> None:
        """Update cached preferences from an external reward signal.

        Only functional when ``preference_mode == "reward_derived"``.
        The reward is passed through the reward MLP to produce new
        preference parameters, which are cached for subsequent
        ``forward()`` and ``log_prob()`` calls.

        Args:
            reward: Reward tensor, shape ``(reward_dim,)`` or ``(B, reward_dim)``.
                If batched, the mean across the batch is used.

        Raises:
            RuntimeError: If called in a mode other than ``"reward_derived"``.
        """
        if self.mode != "reward_derived":
            raise RuntimeError(
                f"update_from_reward() is only valid for 'reward_derived' mode, "
                f"but current mode is '{self.mode}'."
            )

        # Ensure 2D input
        if reward.dim() == 1:
            reward = reward.unsqueeze(0)

        # Average over batch if needed
        if reward.shape[0] > 1:
            reward = reward.mean(dim=0, keepdim=True)

        h = self.reward_encoder(reward)
        new_mean = self.reward_to_mean(h).squeeze(0)
        new_log_prec = self.reward_to_log_prec(h).squeeze(0)

        self._cached_mean.copy_(new_mean.detach())
        self._cached_log_precision.copy_(new_log_prec.detach())

    def set_fixed_preference(
        self, mean: torch.Tensor, precision: Optional[torch.Tensor] = None
    ) -> None:
        """Manually set preference values (works for fixed and learned modes).

        Args:
            mean: Target preference mean, shape ``(obs_dim,)``.
            precision: Optional precision, shape ``(obs_dim,)``. If not given,
                existing precision is kept unchanged.
        """
        with torch.no_grad():
            if self.mode in ("fixed", "learned"):
                self.pref_mean.copy_(mean)
                if precision is not None:
                    self.pref_log_precision.copy_(torch.log(precision + 1e-8))
            elif self.mode == "reward_derived":
                self._cached_mean.copy_(mean)
                if precision is not None:
                    self._cached_log_precision.copy_(
                        torch.log(precision + 1e-8)
                    )

    def save_preferences(self, path: str) -> None:
        """Save current preference parameters to disk.

        Saves a dictionary containing the mean, log_precision, and mode.

        Args:
            path: File path for the saved checkpoint.
        """
        mean, precision = self.forward()
        state = {
            "mode": self.mode,
            "mean": mean.detach().cpu(),
            "log_precision": torch.log(precision + 1e-8).detach().cpu(),
        }
        torch.save(state, path)
        logger.info("Preferences saved to %s", path)

    def load_preferences(self, path: str) -> None:
        """Load preference parameters from disk.

        Args:
            path: File path to load from.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Preference file not found: {path}")

        state = torch.load(path, map_location="cpu", weights_only=True)
        loaded_mean = state["mean"]
        loaded_log_prec = state["log_precision"]

        with torch.no_grad():
            if self.mode in ("fixed", "learned"):
                self.pref_mean.copy_(loaded_mean)
                self.pref_log_precision.copy_(loaded_log_prec)
            elif self.mode == "reward_derived":
                self._cached_mean.copy_(loaded_mean)
                self._cached_log_precision.copy_(loaded_log_prec)

        logger.info("Preferences loaded from %s (mode=%s)", path, state["mode"])


# ===========================================================================
# Top-Level Generative Model
# ===========================================================================


class GenerativeModel(nn.Module):
    """Complete generative model combining encoder, decoder, transition, and preferences.

    Wires together the four components of the Active Inference generative
    model stack and provides convenience methods for encoding, decoding,
    state prediction, preference evaluation, and ELBO computation.

    The ELBO (Evidence Lower BOund) decomposes as:

        ELBO = E_q(s|o)[log P(o|s)] - beta * KL[q(s|o) || p(s)]

    where beta is ``config.kl_weight`` and p(s) is the standard normal prior
    (continuous) or uniform prior (discrete).

    Args:
        config: Generative model configuration. If ``None``, default config
            is used.
    """

    def __init__(self, config: Optional[GenerativeModelConfig] = None) -> None:
        super().__init__()
        self.config = config or GenerativeModelConfig()

        # Build the four components
        self.encoder = LatentEncoder(self.config)
        self.decoder = LikelihoodDecoder(self.config)
        self.transition = TransitionModel(self.config)
        self.preferences = Preferences(self.config)

    def encode(
        self,
        obs: torch.Tensor,
        ctx: Optional[torch.Tensor] = None,
    ) -> LatentDistribution:
        """Encode observations into a latent distribution.

        Args:
            obs: Observations, shape ``(B, obs_dim)``.
            ctx: Optional context, shape ``(B, ctx_dim)``.

        Returns:
            ``LatentDistribution`` with posterior parameters and sample.
        """
        return self.encoder(obs, ctx=ctx)

    def decode(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode latent state to observation distribution.

        Args:
            state: Latent state, shape ``(B, state_dim)``.

        Returns:
            Tuple of (mean, log_var), each shape ``(B, obs_dim)``.
        """
        return self.decoder(state)

    def predict_transition(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> TransitionDistribution:
        """Predict next-state distribution given state and action.

        Args:
            state: Current state, shape ``(B, state_dim)``.
            action: Action tensor.

        Returns:
            ``TransitionDistribution`` from the ensemble.
        """
        return self.transition(state, action)

    def preference_log_prob(self, obs: torch.Tensor) -> torch.Tensor:
        """Evaluate log-probability of observations under preferences.

        Args:
            obs: Observations, shape ``(B, obs_dim)``.

        Returns:
            Log-probability per batch element, shape ``(B,)``.
        """
        return self.preferences.log_prob(obs)

    def compute_elbo(
        self,
        obs: torch.Tensor,
        ctx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the Evidence Lower BOund (ELBO) for training.

        ELBO = E_q(s|o)[log P(o|s)] - beta * KL[q(s|o) || p(s)]

        A higher ELBO indicates a better fit. The negative ELBO is the
        variational free energy (VFE) that the model minimizes.

        Args:
            obs: Observations, shape ``(B, obs_dim)``.
            ctx: Optional context, shape ``(B, ctx_dim)``.

        Returns:
            Tuple of:
                - ELBO per batch element, shape ``(B,)``.
                - Dictionary of component values:
                    - ``"reconstruction"``: log P(o|s) per batch, shape ``(B,)``.
                    - ``"kl"``: KL divergence per batch, shape ``(B,)``.
                    - ``"elbo"``: ELBO per batch, shape ``(B,)``.
                    - ``"loss"``: Negative mean ELBO (scalar, for optimization).
        """
        # Encode
        dist = self.encoder(obs, ctx=ctx)

        # Reconstruction term: E_q[log P(o|s)]
        reconstruction = self.decoder.log_prob(obs, dist.sample)

        # KL divergence: KL[q(s|o) || p(s)]
        kl = dist.kl_divergence()

        # ELBO
        elbo = reconstruction - self.config.kl_weight * kl

        components = {
            "reconstruction": reconstruction.detach(),
            "kl": kl.detach(),
            "elbo": elbo.detach(),
            "loss": -elbo.mean().detach(),
        }

        return elbo, components

    def reconstruct(
        self,
        obs: torch.Tensor,
        ctx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode then decode observations (reconstruction).

        Args:
            obs: Input observations, shape ``(B, obs_dim)``.
            ctx: Optional context, shape ``(B, ctx_dim)``.

        Returns:
            Reconstructed observation means, shape ``(B, obs_dim)``.
        """
        dist = self.encoder(obs, ctx=ctx)
        mean, _ = self.decoder(dist.sample)
        return mean

    def imagine_trajectory(
        self,
        obs: torch.Tensor,
        action_sequence: torch.Tensor,
        ctx: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Imagine a future trajectory given an observation and action sequence.

        Encodes the current observation into a latent state, then rolls out
        the transition model for each action in the sequence, decoding
        predicted observations at each step.

        Args:
            obs: Current observation, shape ``(B, obs_dim)``.
            action_sequence: Future actions, shape ``(T, B, action_dim)``
                or ``(T, B)`` for discrete.
            ctx: Optional context, shape ``(B, ctx_dim)``.

        Returns:
            Tuple of:
                - List of predicted state samples ``[s_1, ..., s_T]``.
                - List of predicted observation means ``[o_1, ..., o_T]``.
        """
        dist = self.encoder(obs, ctx=ctx)
        states, _ = self.transition.predict_sequence(
            dist.sample, action_sequence
        )
        pred_obs = []
        for s in states:
            mean, _ = self.decoder(s)
            pred_obs.append(mean)
        return states, pred_obs


# ===========================================================================
# Factory function
# ===========================================================================


def create_generative_model(
    obs_dim: int = 4096,
    state_dim: int = 256,
    action_dim: int = 128,
    hidden_dim: int = 512,
    **kwargs: Any,
) -> GenerativeModel:
    """Create a GenerativeModel with the given dimensions.

    This is the recommended entry point for constructing the generative
    model. Keyword arguments are forwarded to ``GenerativeModelConfig``.

    Args:
        obs_dim: Workspace observation dimension.
        state_dim: Latent state dimension.
        action_dim: Action space dimension.
        hidden_dim: Hidden layer width.
        **kwargs: Additional config overrides.

    Returns:
        A fully initialized ``GenerativeModel``.
    """
    config = GenerativeModelConfig(
        obs_dim=obs_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        **kwargs,
    )
    return GenerativeModel(config)


# ===========================================================================
# Self-Test Suite
# ===========================================================================


def _run_self_tests() -> None:
    """Run comprehensive self-tests for all generative model components.

    Validates shapes, gradient flow, determinism, distribution properties,
    uncertainty decomposition, serialization, and ELBO computation. Prints
    results for each test and raises AssertionError on any failure.

    This block is designed to be executed directly:

        python generative_model_template.py
    """
    import sys
    import traceback

    torch.manual_seed(42)

    passed = 0
    failed = 0
    total = 0
    failures: List[str] = []

    def _test(name: str, fn):
        nonlocal passed, failed, total
        total += 1
        try:
            fn()
            passed += 1
            print(f"  [PASS] {name}")
        except Exception as e:
            failed += 1
            tb = traceback.format_exc()
            failures.append(f"{name}: {e}\n{tb}")
            print(f"  [FAIL] {name}: {e}")

    # ------------------------------------------------------------------
    # Config for tests: small dimensions for speed
    # ------------------------------------------------------------------
    B = 4
    test_cfg = GenerativeModelConfig(
        obs_dim=64,
        state_dim=16,
        action_dim=8,
        hidden_dim=32,
        encoder_layers=2,
        decoder_layers=2,
        transition_ensemble_size=3,
        transition_layers=2,
        latent_type="continuous",
        ctx_dim=None,
        kl_weight=1.0,
        use_residual_transition=True,
    )

    obs = torch.randn(B, test_cfg.obs_dim)
    state = torch.randn(B, test_cfg.state_dim)
    action_cont = torch.randn(B, test_cfg.action_dim)

    # ==================================================================
    # 1. GenerativeModelConfig tests
    # ==================================================================
    print("\n=== GenerativeModelConfig ===")

    def test_config_defaults():
        cfg = GenerativeModelConfig()
        assert cfg.obs_dim == 4096
        assert cfg.state_dim == 256
        assert cfg.action_dim == 128
        assert cfg.hidden_dim == 512
        assert cfg.latent_type == "continuous"
        assert cfg.log_var_clamp == (-10.0, 2.0)
        assert cfg.kl_weight == 1.0
    _test("Config defaults", test_config_defaults)

    def test_config_effective_ctx_dim():
        cfg1 = GenerativeModelConfig(ctx_dim=None)
        assert cfg1.effective_ctx_dim == 0
        cfg2 = GenerativeModelConfig(ctx_dim=0)
        assert cfg2.effective_ctx_dim == 0
        cfg3 = GenerativeModelConfig(ctx_dim=64)
        assert cfg3.effective_ctx_dim == 64
    _test("Config effective_ctx_dim", test_config_effective_ctx_dim)

    # ==================================================================
    # 2. LatentDistribution tests
    # ==================================================================
    print("\n=== LatentDistribution ===")

    def test_latent_dist_kl_to_standard_normal():
        mu = torch.zeros(B, 16)
        log_var = torch.zeros(B, 16)
        sample = torch.randn(B, 16)
        dist = LatentDistribution(mu=mu, log_var=log_var, sample=sample)
        kl = dist.kl_divergence()
        assert kl.shape == (B,)
        # KL of N(0,I) to N(0,I) should be ~0
        assert kl.abs().max().item() < 1e-5, f"Expected ~0, got {kl}"
    _test("KL to standard normal is ~0", test_latent_dist_kl_to_standard_normal)

    def test_latent_dist_kl_nonzero():
        mu = torch.ones(B, 16) * 2.0
        log_var = torch.ones(B, 16) * 0.5
        sample = torch.randn(B, 16)
        dist = LatentDistribution(mu=mu, log_var=log_var, sample=sample)
        kl = dist.kl_divergence()
        assert kl.shape == (B,)
        assert (kl > 0).all(), "KL should be positive for non-standard Gaussian"
    _test("KL is positive for non-standard Gaussian", test_latent_dist_kl_nonzero)

    def test_latent_dist_kl_with_prior():
        mu = torch.randn(B, 16)
        log_var = torch.randn(B, 16) * 0.1
        prior_mu = torch.randn(B, 16)
        prior_lv = torch.randn(B, 16) * 0.1
        sample = torch.randn(B, 16)
        dist = LatentDistribution(mu=mu, log_var=log_var, sample=sample)
        kl = dist.kl_divergence(prior_mu=prior_mu, prior_log_var=prior_lv)
        assert kl.shape == (B,)
        assert (kl >= 0).all(), "KL should be non-negative"
    _test("KL with non-standard prior", test_latent_dist_kl_with_prior)

    def test_latent_dist_entropy():
        mu = torch.zeros(B, 16)
        log_var = torch.zeros(B, 16)
        sample = torch.randn(B, 16)
        dist = LatentDistribution(mu=mu, log_var=log_var, sample=sample)
        h = dist.entropy()
        assert h.shape == (B,)
        # Entropy of N(0,1)^16 = 16 * 0.5 * log(2*pi*e)
        expected = 16 * 0.5 * LOG_2PIE
        assert (h - expected).abs().max().item() < 1e-4
    _test("Entropy of standard Gaussian", test_latent_dist_entropy)

    def test_latent_dist_discrete_kl():
        logits = torch.randn(B, 32)
        sample = F.gumbel_softmax(logits, tau=1.0, hard=False)
        dist = LatentDistribution(
            mu=logits, log_var=torch.zeros_like(logits),
            sample=sample, is_discrete=True
        )
        kl = dist.kl_divergence()
        assert kl.shape == (B,)
        assert (kl >= 0).all()
    _test("Discrete KL to uniform is non-negative", test_latent_dist_discrete_kl)

    # ==================================================================
    # 3. LatentEncoder tests
    # ==================================================================
    print("\n=== LatentEncoder ===")

    def test_encoder_output_shape():
        enc = LatentEncoder(test_cfg)
        dist = enc(obs)
        assert dist.mu.shape == (B, test_cfg.state_dim)
        assert dist.log_var.shape == (B, test_cfg.state_dim)
        assert dist.sample.shape == (B, test_cfg.state_dim)
    _test("Encoder output shapes", test_encoder_output_shape)

    def test_encoder_gradient_flow():
        enc = LatentEncoder(test_cfg)
        dist = enc(obs)
        loss = dist.sample.sum()
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in enc.parameters()
        )
        assert has_grad, "No gradients flowed through the encoder"
    _test("Encoder gradient flow", test_encoder_gradient_flow)

    def test_encoder_reparameterization():
        enc = LatentEncoder(test_cfg)
        enc.train()
        torch.manual_seed(1)
        d1 = enc(obs)
        torch.manual_seed(2)
        d2 = enc(obs)
        # Same input, different random noise -> different samples
        assert not torch.allclose(d1.sample, d2.sample, atol=1e-6), \
            "Samples should differ with different seeds"
        # But means should be the same (deterministic network)
        assert torch.allclose(d1.mu, d2.mu, atol=1e-6), \
            "Means should be identical for same input"
    _test("Reparameterization produces different samples", test_encoder_reparameterization)

    def test_encoder_log_var_clamping():
        enc = LatentEncoder(test_cfg)
        # Use extreme input to try to produce extreme log_var
        extreme_obs = torch.randn(B, test_cfg.obs_dim) * 100
        dist = enc(extreme_obs)
        assert dist.log_var.min().item() >= test_cfg.log_var_clamp[0]
        assert dist.log_var.max().item() <= test_cfg.log_var_clamp[1]
    _test("Encoder log-var clamping", test_encoder_log_var_clamping)

    def test_encoder_kl_output():
        enc = LatentEncoder(test_cfg)
        dist = enc(obs)
        kl = dist.kl_divergence()
        assert kl.shape == (B,)
        assert torch.isfinite(kl).all()
    _test("Encoder KL divergence computation", test_encoder_kl_output)

    def test_encoder_context_concat():
        ctx_dim = 16
        cfg_ctx = GenerativeModelConfig(
            obs_dim=64, state_dim=16, hidden_dim=32,
            encoder_layers=2, ctx_dim=ctx_dim, ctx_mode="concat",
        )
        enc = LatentEncoder(cfg_ctx)
        ctx = torch.randn(B, ctx_dim)
        dist = enc(obs, ctx=ctx)
        assert dist.mu.shape == (B, cfg_ctx.state_dim)
    _test("Encoder context mode: concat", test_encoder_context_concat)

    def test_encoder_context_cross_attention():
        ctx_dim = 16
        cfg_ctx = GenerativeModelConfig(
            obs_dim=64, state_dim=16, hidden_dim=32,
            encoder_layers=2, ctx_dim=ctx_dim, ctx_mode="cross_attention",
        )
        enc = LatentEncoder(cfg_ctx)
        ctx = torch.randn(B, ctx_dim)
        dist = enc(obs, ctx=ctx)
        assert dist.mu.shape == (B, cfg_ctx.state_dim)
    _test("Encoder context mode: cross_attention", test_encoder_context_cross_attention)

    def test_encoder_context_film():
        ctx_dim = 16
        cfg_ctx = GenerativeModelConfig(
            obs_dim=64, state_dim=16, hidden_dim=32,
            encoder_layers=2, ctx_dim=ctx_dim, ctx_mode="film",
        )
        enc = LatentEncoder(cfg_ctx)
        ctx = torch.randn(B, ctx_dim)
        dist = enc(obs, ctx=ctx)
        assert dist.mu.shape == (B, cfg_ctx.state_dim)
    _test("Encoder context mode: FiLM", test_encoder_context_film)

    def test_encoder_discrete():
        cfg_disc = GenerativeModelConfig(
            obs_dim=64, state_dim=16, hidden_dim=32,
            encoder_layers=2, latent_type="discrete", num_discrete_states=32,
        )
        enc = LatentEncoder(cfg_disc)
        enc.train()
        dist = enc(obs)
        assert dist.is_discrete
        assert dist.mu.shape == (B, 32), f"Expected logits shape (B,32), got {dist.mu.shape}"
        assert dist.sample.shape == (B, 32)
        # Gumbel-softmax samples should sum to ~1 along last dim
        sums = dist.sample.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(B), atol=0.1)
    _test("Encoder discrete (Gumbel-Softmax)", test_encoder_discrete)

    # ==================================================================
    # 4. LikelihoodDecoder tests
    # ==================================================================
    print("\n=== LikelihoodDecoder ===")

    def test_decoder_output_shape():
        dec = LikelihoodDecoder(test_cfg)
        mean, log_var = dec(state)
        assert mean.shape == (B, test_cfg.obs_dim)
        assert log_var.shape == (B, test_cfg.obs_dim)
    _test("Decoder output shapes", test_decoder_output_shape)

    def test_decoder_log_var_clamping():
        dec = LikelihoodDecoder(test_cfg)
        extreme_state = torch.randn(B, test_cfg.state_dim) * 100
        _, log_var = dec(extreme_state)
        assert log_var.min().item() >= test_cfg.log_var_clamp[0]
        assert log_var.max().item() <= test_cfg.log_var_clamp[1]
    _test("Decoder log-var clamping", test_decoder_log_var_clamping)

    def test_decoder_reconstruction_loss():
        dec = LikelihoodDecoder(test_cfg)
        loss = dec.reconstruction_loss(obs, state)
        assert loss.dim() == 0, "Reconstruction loss should be scalar"
        assert torch.isfinite(loss)
    _test("Decoder reconstruction loss", test_decoder_reconstruction_loss)

    def test_decoder_log_prob_shape():
        dec = LikelihoodDecoder(test_cfg)
        lp = dec.log_prob(obs, state)
        assert lp.shape == (B,)
        assert torch.isfinite(lp).all()
    _test("Decoder log_prob shape", test_decoder_log_prob_shape)

    def test_decoder_gradient_flow():
        dec = LikelihoodDecoder(test_cfg)
        loss = dec.reconstruction_loss(obs, state)
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in dec.parameters()
        )
        assert has_grad, "No gradients through decoder"
    _test("Decoder gradient flow", test_decoder_gradient_flow)

    def test_decoder_sample():
        dec = LikelihoodDecoder(test_cfg)
        sampled = dec.sample(state)
        assert sampled.shape == (B, test_cfg.obs_dim)
    _test("Decoder sample shape", test_decoder_sample)

    # ==================================================================
    # 5. TransitionModel tests
    # ==================================================================
    print("\n=== TransitionModel ===")

    def test_transition_output_shape():
        trans = TransitionModel(test_cfg)
        dist = trans(state, action_cont)
        assert dist.means.shape == (test_cfg.transition_ensemble_size, B, test_cfg.state_dim)
        assert dist.log_vars.shape == (test_cfg.transition_ensemble_size, B, test_cfg.state_dim)
    _test("Transition output shapes", test_transition_output_shape)

    def test_transition_ensemble_size():
        trans = TransitionModel(test_cfg)
        dist = trans(state, action_cont)
        assert dist.ensemble_size == test_cfg.transition_ensemble_size
    _test("Transition ensemble size", test_transition_ensemble_size)

    def test_transition_log_var_clamping():
        trans = TransitionModel(test_cfg)
        extreme_state = torch.randn(B, test_cfg.state_dim) * 100
        extreme_action = torch.randn(B, test_cfg.action_dim) * 100
        dist = trans(extreme_state, extreme_action)
        assert dist.log_vars.min().item() >= test_cfg.log_var_clamp[0]
        assert dist.log_vars.max().item() <= test_cfg.log_var_clamp[1]
    _test("Transition log-var clamping", test_transition_log_var_clamping)

    def test_transition_sample_shape():
        trans = TransitionModel(test_cfg)
        dist = trans(state, action_cont)
        s = dist.sample()
        assert s.shape == (B, test_cfg.state_dim)
    _test("Transition sample shape", test_transition_sample_shape)

    def test_transition_sample_specific_member():
        trans = TransitionModel(test_cfg)
        dist = trans(state, action_cont)
        s = dist.sample(member_idx=0)
        assert s.shape == (B, test_cfg.state_dim)
    _test("Transition sample from specific member", test_transition_sample_specific_member)

    def test_transition_epistemic_uncertainty():
        trans = TransitionModel(test_cfg)
        dist = trans(state, action_cont)
        eu = dist.epistemic_uncertainty()
        assert eu.shape == (B, test_cfg.state_dim)
        assert (eu >= 0).all(), "Epistemic uncertainty should be non-negative"
    _test("Transition epistemic uncertainty", test_transition_epistemic_uncertainty)

    def test_transition_aleatoric_uncertainty():
        trans = TransitionModel(test_cfg)
        dist = trans(state, action_cont)
        au = dist.aleatoric_uncertainty()
        assert au.shape == (B, test_cfg.state_dim)
        assert (au > 0).all(), "Aleatoric uncertainty should be positive"
    _test("Transition aleatoric uncertainty", test_transition_aleatoric_uncertainty)

    def test_transition_total_uncertainty():
        trans = TransitionModel(test_cfg)
        dist = trans(state, action_cont)
        tu = dist.total_uncertainty()
        eu = dist.epistemic_uncertainty()
        au = dist.aleatoric_uncertainty()
        assert torch.allclose(tu, eu + au, atol=1e-6)
    _test("Transition total = epistemic + aleatoric", test_transition_total_uncertainty)

    def test_transition_scalar_uncertainties():
        trans = TransitionModel(test_cfg)
        dist = trans(state, action_cont)
        seu = dist.scalar_epistemic_uncertainty()
        sau = dist.scalar_aleatoric_uncertainty()
        assert seu.shape == (B,)
        assert sau.shape == (B,)
    _test("Transition scalar uncertainty shapes", test_transition_scalar_uncertainties)

    def test_transition_mean_prediction():
        trans = TransitionModel(test_cfg)
        dist = trans(state, action_cont)
        mp = dist.mean_prediction()
        assert mp.shape == (B, test_cfg.state_dim)
    _test("Transition mean prediction shape", test_transition_mean_prediction)

    def test_transition_gradient_flow():
        trans = TransitionModel(test_cfg)
        dist = trans(state, action_cont)
        loss = dist.sample(member_idx=0).sum()
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in trans.parameters()
        )
        assert has_grad, "No gradients through transition model"
    _test("Transition gradient flow", test_transition_gradient_flow)

    def test_transition_predict_sequence():
        trans = TransitionModel(test_cfg)
        T = 5
        action_seq = torch.randn(T, B, test_cfg.action_dim)
        states, dists = trans.predict_sequence(state, action_seq)
        assert len(states) == T
        assert len(dists) == T
        for i, s in enumerate(states):
            assert s.shape == (B, test_cfg.state_dim), \
                f"State {i} shape {s.shape} != expected"
    _test("Transition predict_sequence", test_transition_predict_sequence)

    def test_transition_predict_sequence_fixed_member():
        trans = TransitionModel(test_cfg)
        T = 3
        action_seq = torch.randn(T, B, test_cfg.action_dim)
        states, dists = trans.predict_sequence(state, action_seq, member_idx=1)
        assert len(states) == T
    _test("Transition predict_sequence with fixed member", test_transition_predict_sequence_fixed_member)

    def test_transition_discrete_action():
        cfg_disc_act = GenerativeModelConfig(
            obs_dim=64, state_dim=16, action_dim=8,
            hidden_dim=32, transition_ensemble_size=2,
            transition_layers=2, action_type="discrete",
        )
        trans = TransitionModel(cfg_disc_act)
        disc_action = torch.randint(0, cfg_disc_act.action_dim, (B,))
        dist = trans(state, disc_action)
        assert dist.means.shape == (2, B, 16)
    _test("Transition with discrete actions", test_transition_discrete_action)

    # ==================================================================
    # 6. Preferences tests
    # ==================================================================
    print("\n=== Preferences ===")

    def test_preferences_fixed_clean():
        cfg_f = GenerativeModelConfig(
            obs_dim=64, state_dim=16, hidden_dim=32, preference_mode="fixed"
        )
        pref = Preferences(cfg_f)
        params = list(pref.parameters())
        assert len(params) == 0, f"Expected 0 learnable params, got {len(params)}"
        mean, prec = pref.forward()
        assert mean.shape == (64,)
        lp = pref.log_prob(obs)
        assert lp.shape == (B,)
    _test("Preferences fixed mode", test_preferences_fixed_clean)

    def test_preferences_learned():
        cfg_l = GenerativeModelConfig(
            obs_dim=64, state_dim=16, hidden_dim=32, preference_mode="learned"
        )
        pref = Preferences(cfg_l)
        params = list(pref.parameters())
        assert len(params) == 2, f"Expected 2 learnable params, got {len(params)}"
        lp = pref.log_prob(obs)
        assert lp.shape == (B,)
        # Gradients should flow
        lp.sum().backward()
        assert pref.pref_mean.grad is not None
    _test("Preferences learned mode", test_preferences_learned)

    def test_preferences_reward_derived():
        cfg_r = GenerativeModelConfig(
            obs_dim=64, state_dim=16, hidden_dim=32,
            preference_mode="reward_derived", preference_reward_dim=1,
        )
        pref = Preferences(cfg_r)
        # Before update, preferences are zero buffers
        mean, prec = pref.forward()
        assert mean.shape == (64,)

        # Update from reward
        reward = torch.tensor([1.5])
        pref.update_from_reward(reward)
        mean_new, prec_new = pref.forward()
        # After update, cached values should have changed
        assert not torch.allclose(mean_new, torch.zeros(64), atol=1e-6), \
            "Preferences should change after reward update"

        lp = pref.log_prob(obs)
        assert lp.shape == (B,)
    _test("Preferences reward_derived mode", test_preferences_reward_derived)

    def test_preferences_reward_derived_wrong_mode():
        cfg_l = GenerativeModelConfig(
            obs_dim=64, state_dim=16, hidden_dim=32, preference_mode="learned"
        )
        pref = Preferences(cfg_l)
        try:
            pref.update_from_reward(torch.tensor([1.0]))
            assert False, "Should have raised RuntimeError"
        except RuntimeError:
            pass
    _test("Preferences reward update raises in wrong mode", test_preferences_reward_derived_wrong_mode)

    def test_preferences_serialization():
        cfg_l = GenerativeModelConfig(
            obs_dim=64, state_dim=16, hidden_dim=32, preference_mode="learned"
        )
        pref = Preferences(cfg_l)
        # Set specific values
        with torch.no_grad():
            pref.pref_mean.fill_(1.5)
            pref.pref_log_precision.fill_(0.3)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            pref.save_preferences(path)

            # Load into a new instance
            pref2 = Preferences(cfg_l)
            pref2.load_preferences(path)

            m1, p1 = pref.forward()
            m2, p2 = pref2.forward()
            assert torch.allclose(m1, m2, atol=1e-5)
            assert torch.allclose(p1, p2, atol=1e-5)
        finally:
            os.remove(path)
    _test("Preferences save/load roundtrip", test_preferences_serialization)

    def test_preferences_set_fixed():
        cfg_f = GenerativeModelConfig(
            obs_dim=64, state_dim=16, hidden_dim=32, preference_mode="fixed"
        )
        pref = Preferences(cfg_f)
        target = torch.randn(64)
        pref.set_fixed_preference(target)
        mean, _ = pref.forward()
        assert torch.allclose(mean, target, atol=1e-6)
    _test("Preferences set_fixed_preference", test_preferences_set_fixed)

    # ==================================================================
    # 7. GenerativeModel (integrated) tests
    # ==================================================================
    print("\n=== GenerativeModel (integrated) ===")

    def test_gm_construction():
        gm = GenerativeModel(test_cfg)
        assert isinstance(gm.encoder, LatentEncoder)
        assert isinstance(gm.decoder, LikelihoodDecoder)
        assert isinstance(gm.transition, TransitionModel)
        assert isinstance(gm.preferences, Preferences)
    _test("GenerativeModel construction", test_gm_construction)

    def test_gm_encode():
        gm = GenerativeModel(test_cfg)
        dist = gm.encode(obs)
        assert dist.mu.shape == (B, test_cfg.state_dim)
        assert dist.sample.shape == (B, test_cfg.state_dim)
    _test("GenerativeModel encode", test_gm_encode)

    def test_gm_decode():
        gm = GenerativeModel(test_cfg)
        mean, log_var = gm.decode(state)
        assert mean.shape == (B, test_cfg.obs_dim)
        assert log_var.shape == (B, test_cfg.obs_dim)
    _test("GenerativeModel decode", test_gm_decode)

    def test_gm_transition():
        gm = GenerativeModel(test_cfg)
        td = gm.predict_transition(state, action_cont)
        assert td.means.shape[0] == test_cfg.transition_ensemble_size
    _test("GenerativeModel predict_transition", test_gm_transition)

    def test_gm_preference_log_prob():
        gm = GenerativeModel(test_cfg)
        lp = gm.preference_log_prob(obs)
        assert lp.shape == (B,)
    _test("GenerativeModel preference_log_prob", test_gm_preference_log_prob)

    def test_gm_elbo():
        gm = GenerativeModel(test_cfg)
        elbo, components = gm.compute_elbo(obs)
        assert elbo.shape == (B,)
        assert "reconstruction" in components
        assert "kl" in components
        assert "elbo" in components
        assert "loss" in components
        assert components["loss"].dim() == 0
        # ELBO = reconstruction - kl_weight * kl
        expected_elbo = components["reconstruction"] - test_cfg.kl_weight * components["kl"]
        assert torch.allclose(elbo.detach(), expected_elbo, atol=1e-4), \
            f"ELBO decomposition mismatch"
    _test("GenerativeModel ELBO computation", test_gm_elbo)

    def test_gm_elbo_gradient():
        gm = GenerativeModel(test_cfg)
        elbo, _ = gm.compute_elbo(obs)
        loss = -elbo.mean()
        loss.backward()
        grad_count = sum(
            1 for p in gm.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        assert grad_count > 0, "No gradients through ELBO"
    _test("GenerativeModel ELBO gradient flow", test_gm_elbo_gradient)

    def test_gm_reconstruct():
        gm = GenerativeModel(test_cfg)
        recon = gm.reconstruct(obs)
        assert recon.shape == (B, test_cfg.obs_dim)
    _test("GenerativeModel reconstruct", test_gm_reconstruct)

    def test_gm_encode_decode_roundtrip():
        gm = GenerativeModel(test_cfg)
        dist = gm.encode(obs)
        mean, _ = gm.decode(dist.sample)
        assert mean.shape == obs.shape
        # After training this should improve; for now just check shape & finite
        assert torch.isfinite(mean).all()
    _test("GenerativeModel encode-decode roundtrip", test_gm_encode_decode_roundtrip)

    def test_gm_imagine_trajectory():
        gm = GenerativeModel(test_cfg)
        T = 4
        action_seq = torch.randn(T, B, test_cfg.action_dim)
        states, pred_obs = gm.imagine_trajectory(obs, action_seq)
        assert len(states) == T
        assert len(pred_obs) == T
        for i in range(T):
            assert states[i].shape == (B, test_cfg.state_dim)
            assert pred_obs[i].shape == (B, test_cfg.obs_dim)
    _test("GenerativeModel imagine_trajectory", test_gm_imagine_trajectory)

    def test_gm_fp32_verification():
        gm = GenerativeModel(test_cfg)
        obs_fp32 = obs.float()
        elbo, components = gm.compute_elbo(obs_fp32)
        assert elbo.dtype == torch.float32, f"ELBO dtype {elbo.dtype} != float32"
        assert components["reconstruction"].dtype == torch.float32
        assert components["kl"].dtype == torch.float32
    _test("GenerativeModel fp32 verification", test_gm_fp32_verification)

    def test_gm_batch_independence():
        gm = GenerativeModel(test_cfg)
        gm.eval()
        torch.manual_seed(99)
        # Process full batch
        with torch.no_grad():
            elbo_full, _ = gm.compute_elbo(obs)
        # Process each element individually
        for i in range(B):
            torch.manual_seed(99 + i)
            single = obs[i:i+1]
            with torch.no_grad():
                dist = gm.encode(single)
                recon_lp = gm.decoder.log_prob(single, dist.sample)
            # Just verify it produces finite values independently
            assert torch.isfinite(recon_lp).all(), f"Non-finite for element {i}"
    _test("GenerativeModel batch independence (finite)", test_gm_batch_independence)

    def test_gm_default_construction():
        # Test that default config works (full 4096-dim)
        gm = GenerativeModel()
        assert gm.config.obs_dim == 4096
        assert gm.config.state_dim == 256
    _test("GenerativeModel default construction", test_gm_default_construction)

    def test_factory_function():
        gm = create_generative_model(obs_dim=128, state_dim=32, action_dim=16)
        assert gm.config.obs_dim == 128
        dist = gm.encode(torch.randn(2, 128))
        assert dist.sample.shape == (2, 32)
    _test("Factory function create_generative_model", test_factory_function)

    # ==================================================================
    # 8. TransitionDistribution edge cases
    # ==================================================================
    print("\n=== TransitionDistribution edge cases ===")

    def test_transition_dist_single_member():
        cfg_single = GenerativeModelConfig(
            obs_dim=64, state_dim=16, action_dim=8,
            hidden_dim=32, transition_ensemble_size=1,
            transition_layers=1,
        )
        trans = TransitionModel(cfg_single)
        dist = trans(state, action_cont)
        assert dist.ensemble_size == 1
        s = dist.sample()
        assert s.shape == (B, 16)
        # Epistemic uncertainty with 1 member should be 0
        eu = dist.epistemic_uncertainty()
        assert torch.allclose(eu, torch.zeros_like(eu), atol=1e-8), \
            "Single member should have 0 epistemic uncertainty"
    _test("Single ensemble member: 0 epistemic uncertainty", test_transition_dist_single_member)

    def test_transition_residual_vs_absolute():
        cfg_res = GenerativeModelConfig(
            obs_dim=64, state_dim=16, action_dim=8,
            hidden_dim=32, transition_ensemble_size=1,
            transition_layers=1, use_residual_transition=True,
        )
        cfg_abs = GenerativeModelConfig(
            obs_dim=64, state_dim=16, action_dim=8,
            hidden_dim=32, transition_ensemble_size=1,
            transition_layers=1, use_residual_transition=False,
        )
        trans_res = TransitionModel(cfg_res)
        trans_abs = TransitionModel(cfg_abs)

        dist_res = trans_res(state, action_cont)
        dist_abs = trans_abs(state, action_cont)

        # Both should produce valid shapes
        assert dist_res.means.shape == dist_abs.means.shape

        # Residual: mean should be close to state at init (since weights init to 0)
        # The mean prediction is state + 0 = state
        res_mean = dist_res.means[0]
        assert torch.allclose(res_mean, state, atol=0.1), \
            "Residual transition should start near identity at init"
    _test("Residual vs absolute transition", test_transition_residual_vs_absolute)

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n{'='*60}")
    print(f"Self-test results: {passed}/{total} passed, {failed} failed")
    print(f"{'='*60}")

    if failures:
        print("\nFailed tests:")
        for f in failures:
            print(f"  {f}")

    if failed > 0:
        sys.exit(1)
    else:
        print("\nAll tests passed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _run_self_tests()
