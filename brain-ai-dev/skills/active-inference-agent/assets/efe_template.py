"""
Expected Free Energy (EFE) Computation Template.

Implements the three-component EFE decomposition for Active Inference:

    EFE(pi) = w_p * Pragmatic(pi) + w_e * Epistemic(pi) + w_i * Instrumental(pi)

Note: Instrumental returns **negative empowerment** (non-positive), so adding it
increases EFE when empowerment is low and decreases EFE when empowerment is high.

Where:
    - Pragmatic:    E_{q(o|pi)}[ -log p_pref(o) ]
                    How far predicted observations deviate from preferences.
    - Epistemic:    KL[ q(s|o,pi) || q(s|pi) ]
                    Information gain about hidden states.
    - Instrumental: E[ log q(a|s,s') - log q(a|s) ]
                    Empowerment -- how many future options an action enables.

Hard invariant:
    |compute_efe_total(p, e, i, w) - (w.pragmatic * p + w.epistemic * e - w.instrumental * i)| < 1e-5

All EFE computation is performed in fp32 regardless of input dtype.

References:
    - Parr, T., Pezzulo, G., & Friston, K. J. (2022). Active Inference.
    - Berseth et al. (2021). SMiRL + Empowerment for intrinsic motivation.
    - Sajid, N. et al. (2021). Active Inference: Demystified and Compared.
    - Millidge, B. et al. (2024). Deep Active Inference with three-term EFE.

This module is part of the brain-inspired AI system described in CLAUDE.md.
The decision layer (basal ganglia analog) uses EFE to select actions that
jointly minimize prediction error, resolve uncertainty, and maintain agency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import math
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOG_2PI: float = math.log(2.0 * math.pi)
LOG_2PIE: float = math.log(2.0 * math.pi * math.e)
EFE_SUM_INVARIANT_TOL: float = 1e-5


# ===========================================================================
# Configuration
# ===========================================================================


@dataclass
class EFEConfig:
    """Configuration for Expected Free Energy computation.

    Attributes:
        pragmatic_weight: Scalar weight for the pragmatic (goal-directed) term.
            Higher values bias the agent toward achieving preferences.
        epistemic_weight: Scalar weight for the epistemic (information-seeking)
            term.  Higher values bias the agent toward reducing uncertainty.
        instrumental_weight: Scalar weight for the instrumental (empowerment)
            term.  Higher values bias the agent toward maintaining options.
        num_samples: Number of Monte Carlo samples for expectations.
        discount_factor: Temporal discount gamma applied per planning step.
        normalize_by_horizon: If True, divide trajectory EFE by horizon length
            so that the magnitude is comparable across different horizons.
        normalize_terms: If True, apply running normalization to each term
            before weighting, which stabilizes training when term magnitudes
            differ by orders of magnitude.
        term_norm_mode: Normalization strategy when ``normalize_terms`` is True.
            - ``"running"``: running mean / variance whitening.
            - ``"sigmoid"``: squash each term through sigmoid.
            - ``"none"``: no normalization (same as normalize_terms=False).
        state_dim: Dimensionality of the latent state space.
        action_dim: Dimensionality of the action space.
        hidden_dim: Hidden layer size for empowerment MLPs.
    """

    pragmatic_weight: float = 1.0
    epistemic_weight: float = 1.0
    instrumental_weight: float = 0.1
    num_samples: int = 32
    discount_factor: float = 0.99
    normalize_by_horizon: bool = True
    normalize_terms: bool = False
    term_norm_mode: str = "running"  # "running" | "sigmoid" | "none"

    # Architecture dimensions (used by EFEComputer / EmpowermentEstimator)
    state_dim: int = 64
    action_dim: int = 10
    hidden_dim: int = 256


# ===========================================================================
# Pure Functions -- No side effects, no self, deterministic given inputs.
# ===========================================================================


def compute_pragmatic(
    predicted_obs_mean: torch.Tensor,
    predicted_obs_log_var: torch.Tensor,
    preference_mean: torch.Tensor,
    preference_log_var: torch.Tensor,
) -> torch.Tensor:
    """Compute the pragmatic (goal-directed) term of EFE.

    Evaluates how well predicted observations match the agent's preferences
    under a Gaussian preference model.  This is the expected negative
    log-likelihood of predicted observations under the preference distribution:

        E_{q(o|pi)}[ -log p_pref(o) ]

    For a Gaussian preference p_pref = N(pref_mean, diag(exp(pref_log_var))):

        -log p_pref(o) = 0.5 * [ exp(-pref_lv) * (o - pref_mean)^2
                                  + log(2*pi)
                                  + pref_lv
                                  + exp(-pref_lv) * var(o) ]

    The ``exp(-pref_lv) * var(o)`` term accounts for the spread of the predictive
    distribution q(o|pi), not just the mean.

    Args:
        predicted_obs_mean: Predicted observation means, shape ``(B, obs_dim)``.
        predicted_obs_log_var: Log-variance of predicted observations,
            shape ``(B, obs_dim)``.
        preference_mean: Mean of the Gaussian preference distribution,
            shape ``(obs_dim,)`` or ``(B, obs_dim)``.
        preference_log_var: Log-variance of the preference distribution,
            shape ``(obs_dim,)`` or ``(B, obs_dim)``.

    Returns:
        Pragmatic value per batch element, shape ``(B,)``.
        Higher values indicate worse alignment with preferences.
    """
    # Promote to fp32 for numerical stability
    pred_mean = predicted_obs_mean.float()
    pred_log_var = predicted_obs_log_var.float()
    pref_mean = preference_mean.float()
    pref_lv = preference_log_var.float()

    pred_var = torch.exp(pred_log_var)
    # precision = 1/variance = exp(-log_var)
    pref_log_var = torch.exp(-pref_lv)

    # Negative log-likelihood under Gaussian preference, per dimension
    # nll_d = 0.5 * [precision * (pred_mean - pref_mean)^2
    #                + log(2*pi)
    #                + log_var  (= -log(precision))
    #                + precision * pred_var]
    sq_diff = (pred_mean - pref_mean).pow(2)
    nll_per_dim = 0.5 * (
        pref_log_var * sq_diff
        + LOG_2PI
        + pref_lv
        + pref_log_var * pred_var
    )

    # Sum over observation dimensions -> (B,)
    pragmatic = nll_per_dim.sum(dim=-1)
    return pragmatic


def compute_epistemic(
    posterior_mean: torch.Tensor,
    posterior_log_var: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_log_var: torch.Tensor,
) -> torch.Tensor:
    """Compute the epistemic (information-seeking) term of EFE.

    This is the KL divergence between the posterior over hidden states given
    the policy's predicted observations and the prior (predictive) distribution
    over states under the policy alone:

        KL[ q(s|o,pi) || q(s|pi) ]

    For diagonal Gaussians the closed-form KL is:

        KL = 0.5 * sum_d [ exp(post_lv - prior_lv)
                           + (prior_mu - post_mu)^2 / exp(prior_lv)
                           - 1
                           + prior_lv - post_lv ]

    The sum is over the state dimensions; the result is per batch element.

    Args:
        posterior_mean: Mean of q(s|o,pi), shape ``(B, state_dim)``.
        posterior_log_var: Log-variance of q(s|o,pi), shape ``(B, state_dim)``.
        prior_mean: Mean of q(s|pi), shape ``(B, state_dim)``.
        prior_log_var: Log-variance of q(s|pi), shape ``(B, state_dim)``.

    Returns:
        KL divergence per batch element, shape ``(B,)``.
        Higher values indicate greater expected information gain.
    """
    post_mu = posterior_mean.float()
    post_lv = posterior_log_var.float()
    pri_mu = prior_mean.float()
    pri_lv = prior_log_var.float()

    # Closed-form KL for diagonal Gaussians
    var_ratio = torch.exp(post_lv - pri_lv)
    mean_sq_diff = (pri_mu - post_mu).pow(2) * torch.exp(-pri_lv)
    kl_per_dim = 0.5 * (var_ratio + mean_sq_diff - 1.0 + pri_lv - post_lv)

    # Sum over state dimensions -> (B,)
    epistemic = kl_per_dim.sum(dim=-1)
    return epistemic


def compute_instrumental(
    state: torch.Tensor,
    action: torch.Tensor,
    next_state: torch.Tensor,
    source_log_probs: torch.Tensor,
    planning_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute the instrumental (empowerment) term of EFE.

    Empowerment measures the agent's capacity to influence future states.
    It is approximated via a variational bound on the mutual information
    I(a; s' | s):

        Empowerment approx = E[ log q(a|s,s') - log q(a|s) ]

    where q(a|s,s') is the planning distribution (which action caused this
    transition) and q(a|s) is the source distribution (marginal action
    prior from this state).

    When the planning network can confidently infer which action caused a
    transition, empowerment is high -- the agent has diverse effects on the
    world.

    The sign convention is: instrumental returns **negative empowerment**
    (non-positive), so that all three EFE terms sum directly:

        efe_total = w_p * pragmatic + w_e * epistemic + w_i * instrumental

    High empowerment -> more negative instrumental -> lower EFE -> better action.

    Args:
        state: Current latent state, shape ``(B, state_dim)``.  Not used
            in the pure computation but included for API consistency and
            potential future extensions.
        action: Action taken, shape ``(B, action_dim)`` or ``(B,)``.  Not
            used directly; the log-probs are pre-computed by the caller.
        next_state: Resulting latent state, shape ``(B, state_dim)``.  Not
            used directly; same rationale as ``state``.
        source_log_probs: Log q(a|s) for the taken action, shape ``(B,)``.
        planning_log_probs: Log q(a|s,s') for the taken action, shape ``(B,)``.

    Returns:
        Negative empowerment per batch element, shape ``(B,)``.
        Non-positive: 0 when no empowerment, negative when empowered.
    """
    src_lp = source_log_probs.float()
    plan_lp = planning_log_probs.float()

    # Variational lower bound on empowerment (positive value)
    empowerment = (plan_lp - src_lp).clamp(min=0.0)

    # Return negative empowerment so all EFE terms sum directly
    return -empowerment


def compute_efe_total(
    pragmatic: torch.Tensor,
    epistemic: torch.Tensor,
    instrumental: torch.Tensor,
    weights: EFEConfig,
) -> torch.Tensor:
    """Combine the three EFE terms into a single scalar per batch element.

    The combination formula is:

        total = w_p * pragmatic + w_e * epistemic + w_i * instrumental

    All terms are summed directly.  Instrumental is already negative empowerment
    (non-positive), so adding it with a positive weight reduces EFE when
    empowerment is high.

    This is the authoritative formula for the sum invariant:

        |total - (w_p * pragmatic + w_e * epistemic + w_i * instrumental)| < 1e-5

    Args:
        pragmatic: Pragmatic term, shape ``(B,)``.
        epistemic: Epistemic term, shape ``(B,)``.
        instrumental: Instrumental term, shape ``(B,)``.
        weights: EFEConfig providing ``pragmatic_weight``, ``epistemic_weight``,
            and ``instrumental_weight``.

    Returns:
        Total EFE per batch element, shape ``(B,)``.
    """
    p = pragmatic.float()
    e = epistemic.float()
    i = instrumental.float()

    total = (
        weights.pragmatic_weight * p
        + weights.epistemic_weight * e
        + weights.instrumental_weight * i
    )
    return total


# ===========================================================================
# Analytical / Closed-Form Test Utilities
# ===========================================================================


def analytical_gaussian_pragmatic(
    pred_mean: torch.Tensor,
    pred_var: torch.Tensor,
    pref_mean: torch.Tensor,
    pref_log_var: torch.Tensor,
) -> torch.Tensor:
    """Exact negative log-likelihood of a Gaussian observation under Gaussian
    preferences.

    This provides a reference implementation using variance (not log-variance)
    for direct comparison with ``compute_pragmatic``.

    Formula per dimension d:
        nll_d = 0.5 * [ exp(-lv_d) * (mu_o_d - mu_pref_d)^2
                       + log(2*pi)
                       + lv_d
                       + exp(-lv_d) * var_o_d ]

    Args:
        pred_mean: ``(B, D)`` predicted observation mean.
        pred_var: ``(B, D)`` predicted observation variance (NOT log-var).
        pref_mean: ``(D,)`` or ``(B, D)`` preference mean.
        pref_log_var: ``(D,)`` or ``(B, D)`` preference log-variance.

    Returns:
        ``(B,)`` exact pragmatic value.
    """
    pm = pred_mean.float()
    pv = pred_var.float()
    pfm = pref_mean.float()
    pf_lv = pref_log_var.float()
    pfp = torch.exp(-pf_lv)  # precision = 1/var = exp(-log_var)

    sq = (pm - pfm).pow(2)
    nll = 0.5 * (pfp * sq + LOG_2PI + pf_lv + pfp * pv)
    return nll.sum(dim=-1)


def analytical_gaussian_kl(
    mu1: torch.Tensor,
    var1: torch.Tensor,
    mu2: torch.Tensor,
    var2: torch.Tensor,
) -> torch.Tensor:
    """Exact KL divergence KL(N(mu1,var1) || N(mu2,var2)) for diagonal Gaussians.

    Uses variance (not log-variance) for clarity.

    Formula per dimension d:
        kl_d = 0.5 * [ var1_d / var2_d + (mu2_d - mu1_d)^2 / var2_d - 1 + log(var2_d / var1_d) ]

    Args:
        mu1: ``(B, D)`` mean of q.
        var1: ``(B, D)`` variance of q (NOT log-var).
        mu2: ``(B, D)`` mean of p.
        var2: ``(B, D)`` variance of p (NOT log-var).

    Returns:
        ``(B,)`` exact KL divergence.
    """
    m1 = mu1.float()
    v1 = var1.float()
    m2 = mu2.float()
    v2 = var2.float()

    kl = 0.5 * (
        v1 / (v2 + 1e-8)
        + (m2 - m1).pow(2) / (v2 + 1e-8)
        - 1.0
        + torch.log((v2 + 1e-8) / (v1 + 1e-8))
    )
    return kl.sum(dim=-1)


# ===========================================================================
# Running Normalization Helper
# ===========================================================================


class RunningNormalization(nn.Module):
    """Track running mean and variance per EFE term for online whitening.

    After a warmup period the normalized output has approximately zero mean
    and unit variance, which stabilizes gradient magnitudes across terms with
    very different scales.

    Statistics are stored as registered buffers so that they survive
    ``state_dict`` round-trips and ``model.to(device)`` calls.

    Usage::

        rn = RunningNormalization(term_names=["pragmatic", "epistemic", "instrumental"])
        # During training:
        normed = rn.normalize(raw_value, "pragmatic")
        rn.update(raw_value, "pragmatic")

    Args:
        term_names: Names of the terms to track.
        momentum: Exponential moving average momentum. Smaller values give
            slower but more stable adaptation.
        warmup_steps: Number of update calls before normalization is active.
            During warmup the raw value is returned unchanged.
        eps: Small constant added to the standard deviation to avoid division
            by zero.
    """

    def __init__(
        self,
        term_names: Optional[List[str]] = None,
        momentum: float = 0.01,
        warmup_steps: int = 100,
        eps: float = 1e-6,
    ):
        super().__init__()
        if term_names is None:
            term_names = ["pragmatic", "epistemic", "instrumental"]
        self.term_names = term_names
        self.momentum = momentum
        self.warmup_steps = warmup_steps
        self.eps = eps

        for name in self.term_names:
            self.register_buffer(f"{name}_mean", torch.zeros(1))
            self.register_buffer(f"{name}_var", torch.ones(1))
            self.register_buffer(f"{name}_count", torch.zeros(1, dtype=torch.long))

    def _get_buffers(self, term_name: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve running buffers for a given term."""
        mean = getattr(self, f"{term_name}_mean")
        var = getattr(self, f"{term_name}_var")
        count = getattr(self, f"{term_name}_count")
        return mean, var, count

    @torch.no_grad()
    def update(self, value: torch.Tensor, term_name: str) -> None:
        """Update running statistics for ``term_name`` with new values.

        Args:
            value: Raw (un-normalized) term values, shape ``(B,)`` or scalar.
            term_name: Which EFE term these values belong to.
        """
        if term_name not in self.term_names:
            raise ValueError(
                f"Unknown term '{term_name}'. "
                f"Registered terms: {self.term_names}"
            )
        mean_buf, var_buf, count_buf = self._get_buffers(term_name)

        v = value.float().detach()
        batch_mean = v.mean()
        batch_var = v.var() if v.numel() > 1 else torch.zeros_like(batch_mean)

        alpha = self.momentum
        mean_buf.lerp_(batch_mean.unsqueeze(0), alpha)
        var_buf.lerp_(batch_var.unsqueeze(0), alpha)
        count_buf.add_(1)

    def normalize(self, value: torch.Tensor, term_name: str) -> torch.Tensor:
        """Normalize ``value`` using running statistics.

        During warmup (fewer than ``warmup_steps`` updates) the raw value
        is returned unchanged to avoid unstable early normalization.

        Args:
            value: Raw term values, shape ``(B,)``.
            term_name: Which EFE term to normalize.

        Returns:
            Normalized values with approximately zero mean and unit variance
            (after warmup), shape ``(B,)``.
        """
        if term_name not in self.term_names:
            raise ValueError(
                f"Unknown term '{term_name}'. "
                f"Registered terms: {self.term_names}"
            )
        mean_buf, var_buf, count_buf = self._get_buffers(term_name)

        if count_buf.item() < self.warmup_steps:
            return value

        std = torch.sqrt(var_buf + self.eps)
        return (value - mean_buf) / std

    def get_statistics(self, term_name: str) -> Dict[str, float]:
        """Return current running statistics as plain Python floats.

        Args:
            term_name: Which term to query.

        Returns:
            Dictionary with ``"mean"``, ``"var"``, ``"std"``, and ``"count"``.
        """
        mean_buf, var_buf, count_buf = self._get_buffers(term_name)
        return {
            "mean": mean_buf.item(),
            "var": var_buf.item(),
            "std": math.sqrt(var_buf.item() + self.eps),
            "count": int(count_buf.item()),
        }


# ===========================================================================
# Empowerment Estimator
# ===========================================================================


class EmpowermentEstimator(nn.Module):
    """Variational empowerment estimator using source and planning networks.

    Empowerment is the mutual information I(a; s' | s) between the agent's
    action and the resulting next state, conditioned on the current state.
    It measures how much *control* the agent has from a given state.

    The variational bound is:
        I(a; s' | s) >= E_{q(a|s)} E_{p(s'|s,a)}[ log q(a|s,s') - log q(a|s) ]

    Two networks parameterize the bound:
        - **source_net** q(a|s): marginal action distribution from state.
        - **planning_net** q(a|s,s'): action posterior given the transition.

    When the planning network can confidently identify which action caused a
    transition (high planning log-prob) while the source network is uncertain
    (low source log-prob), empowerment is high.

    Args:
        state_dim: Dimensionality of the latent state.
        action_dim: Number of discrete actions.
        hidden_dim: Width of hidden layers in source and planning MLPs.
        num_hidden_layers: Number of hidden layers in each MLP.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Source network: q(a|s)
        source_layers: List[nn.Module] = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            source_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        source_layers.append(nn.Linear(hidden_dim, action_dim))
        self.source_net = nn.Sequential(*source_layers)

        # Planning network: q(a|s,s')
        planning_layers: List[nn.Module] = [
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
        ]
        for _ in range(num_hidden_layers - 1):
            planning_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        planning_layers.append(nn.Linear(hidden_dim, action_dim))
        self.planning_net = nn.Sequential(*planning_layers)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute log-probs from both networks for the taken action.

        Args:
            state: ``(B, state_dim)``.
            action: ``(B,)`` integer action indices.
            next_state: ``(B, state_dim)``.

        Returns:
            Tuple of (empowerment, source_log_prob, planning_log_prob),
            each shape ``(B,)``.
        """
        source_lp, planning_lp = self.get_log_probs(state, action, next_state)
        empowerment = planning_lp - source_lp
        return empowerment, source_lp, planning_lp

    def get_log_probs(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute source and planning log-probs for the taken action.

        Args:
            state: ``(B, state_dim)``.
            action: ``(B,)`` integer action indices.
            next_state: ``(B, state_dim)``.

        Returns:
            Tuple of (source_log_prob, planning_log_prob), each ``(B,)``.
        """
        s = state.float()
        ns = next_state.float()
        a = action.long()

        # Source: q(a|s)
        source_logits = self.source_net(s)
        source_log_probs = F.log_softmax(source_logits, dim=-1)
        source_lp = source_log_probs.gather(-1, a.unsqueeze(-1)).squeeze(-1)

        # Planning: q(a|s,s')
        combined = torch.cat([s, ns], dim=-1)
        planning_logits = self.planning_net(combined)
        planning_log_probs = F.log_softmax(planning_logits, dim=-1)
        planning_lp = planning_log_probs.gather(-1, a.unsqueeze(-1)).squeeze(-1)

        return source_lp, planning_lp

    def compute_empowerment(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
    ) -> torch.Tensor:
        """Compute empowerment for a (state, action, next_state) tuple.

        Convenience wrapper returning only the empowerment scalar.

        Args:
            state: ``(B, state_dim)``.
            action: ``(B,)`` integer actions.
            next_state: ``(B, state_dim)``.

        Returns:
            Empowerment estimate, shape ``(B,)``.
        """
        emp, _, _ = self.forward(state, action, next_state)
        return emp

    def estimate_state_empowerment(
        self,
        state: torch.Tensor,
        forward_model: nn.Module,
        num_samples: int = 32,
    ) -> torch.Tensor:
        """Estimate empowerment of a state by averaging over sampled actions.

        Samples actions from the source distribution, rolls them through a
        forward model to get next states, then averages the per-sample
        empowerment.

        Args:
            state: ``(B, state_dim)``.
            forward_model: Module with ``predict_next_state(state, action_onehot)``
                returning ``(next_mu, next_log_var)``.
            num_samples: Number of action samples for the Monte Carlo estimate.

        Returns:
            State empowerment estimate, shape ``(B,)``.
        """
        B = state.shape[0]
        device = state.device

        # Get action distribution from source network
        source_logits = self.source_net(state.float())
        action_probs = F.softmax(source_logits, dim=-1)

        # Sample actions: (B, num_samples)
        actions = torch.multinomial(action_probs, num_samples, replacement=True)

        total_emp = torch.zeros(B, device=device)
        for i in range(num_samples):
            a_i = actions[:, i]  # (B,)
            a_onehot = F.one_hot(a_i, self.action_dim).float()

            with torch.no_grad():
                next_mu, _ = forward_model.predict_next_state(state, a_onehot)

            emp_i = self.compute_empowerment(state, a_i, next_mu)
            total_emp = total_emp + emp_i

        return total_emp / num_samples


# ===========================================================================
# EFE Computer (nn.Module wrapper)
# ===========================================================================


class EFEComputer(nn.Module):
    """Compute Expected Free Energy by calling the pure functions.

    This module provides:
        - Single-step EFE computation with all three terms.
        - Multi-step (trajectory) EFE with discounting and horizon norm.
        - Optional running normalization of individual terms.
        - Empowerment estimation via learned source/planning networks.
        - Validation of the sum invariant on every call.

    The core computation is delegated to the module-level pure functions
    ``compute_pragmatic``, ``compute_epistemic``, ``compute_instrumental``,
    and ``compute_efe_total``.  This class is responsible for:
        1. Owning the learnable empowerment parameters.
        2. Managing running normalization buffers.
        3. Providing trajectory-level aggregation.

    Args:
        config: EFEConfig controlling weights, normalization, and architecture.

    Example::

        config = EFEConfig(pragmatic_weight=1.0, epistemic_weight=1.0)
        computer = EFEComputer(config)
        efe, terms = computer.compute_single_step(
            predicted_obs_params=(pred_mean, pred_log_var),
            posterior_params=(post_mean, post_log_var),
            prior_params=(prior_mean, prior_log_var),
            state=state, action=action, next_state=next_state,
            preference_params=(pref_mean, pref_log_var),
        )
    """

    def __init__(self, config: EFEConfig):
        super().__init__()
        self.config = config

        # Empowerment estimator (for instrumental term)
        self.empowerment_estimator = EmpowermentEstimator(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
        )

        # Running normalization (if enabled)
        if config.normalize_terms and config.term_norm_mode == "running":
            self.running_norm = RunningNormalization(
                term_names=["pragmatic", "epistemic", "instrumental"],
                momentum=0.01,
                warmup_steps=100,
            )
        else:
            self.running_norm = None

    def _normalize_term(
        self,
        value: torch.Tensor,
        term_name: str,
        update: bool = True,
    ) -> torch.Tensor:
        """Apply normalization to a single EFE term.

        Args:
            value: Raw term values, shape ``(B,)``.
            term_name: Name of the term.
            update: Whether to update running statistics (set False at eval).

        Returns:
            Normalized value, shape ``(B,)``.
        """
        mode = self.config.term_norm_mode
        if not self.config.normalize_terms or mode == "none":
            return value

        if mode == "sigmoid":
            return torch.sigmoid(value)

        if mode == "running" and self.running_norm is not None:
            if update and self.training:
                self.running_norm.update(value, term_name)
            return self.running_norm.normalize(value, term_name)

        return value

    def _validate_sum_invariant(
        self,
        total: torch.Tensor,
        pragmatic: torch.Tensor,
        epistemic: torch.Tensor,
        instrumental: torch.Tensor,
    ) -> None:
        """Check the hard invariant: |total - formula| < 1e-5.

        Logs a warning if violated.  Uses the weighted formula from
        ``compute_efe_total`` as ground truth.
        """
        expected = (
            self.config.pragmatic_weight * pragmatic.float()
            + self.config.epistemic_weight * epistemic.float()
            - self.config.instrumental_weight * instrumental.float()
        )
        max_err = (total.float() - expected).abs().max().item()
        if max_err > EFE_SUM_INVARIANT_TOL:
            logger.warning(
                "EFE sum invariant violated: max |total - expected| = %.8f "
                "(tolerance = %.1e). This may indicate numerical issues.",
                max_err,
                EFE_SUM_INVARIANT_TOL,
            )

    def compute_single_step(
        self,
        predicted_obs_params: Tuple[torch.Tensor, torch.Tensor],
        posterior_params: Tuple[torch.Tensor, torch.Tensor],
        prior_params: Tuple[torch.Tensor, torch.Tensor],
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        preference_params: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute EFE for a single planning step.

        Calls the three pure functions, applies optional normalization and
        weighting, validates the sum invariant, and returns the total EFE
        along with a breakdown dictionary.

        Args:
            predicted_obs_params: Tuple of ``(pred_mean, pred_log_var)`` for
                predicted observations, each ``(B, obs_dim)``.
            posterior_params: Tuple of ``(post_mean, post_log_var)`` for the
                posterior over states, each ``(B, state_dim)``.
            prior_params: Tuple of ``(prior_mean, prior_log_var)`` for the
                prior over states, each ``(B, state_dim)``.
            state: Current latent state, ``(B, state_dim)``.
            action: Action index taken, ``(B,)`` integer tensor.
            next_state: Predicted next state, ``(B, state_dim)``.
            preference_params: Tuple of ``(pref_mean, pref_log_var)`` for
                the preference distribution, each ``(obs_dim,)`` or ``(B, obs_dim)``.

        Returns:
            Tuple of:
                - ``efe_step``: Total EFE, shape ``(B,)``.
                - ``terms_dict``: Dictionary with keys ``"pragmatic"``,
                  ``"epistemic"``, ``"instrumental"``, ``"total"``,
                  ``"pragmatic_raw"``, ``"epistemic_raw"``, ``"instrumental_raw"``.
        """
        pred_mean, pred_log_var = predicted_obs_params
        post_mean, post_log_var = posterior_params
        prior_mean, prior_log_var = prior_params
        pref_mean, pref_log_var = preference_params

        # 1. Pragmatic term
        pragmatic_raw = compute_pragmatic(
            pred_mean, pred_log_var, pref_mean, pref_log_var
        )

        # 2. Epistemic term
        epistemic_raw = compute_epistemic(
            post_mean, post_log_var, prior_mean, prior_log_var
        )

        # 3. Instrumental term (empowerment)
        source_lp, planning_lp = self.empowerment_estimator.get_log_probs(
            state, action, next_state
        )
        instrumental_raw = compute_instrumental(
            state, action, next_state, source_lp, planning_lp
        )

        # Optional normalization
        pragmatic_normed = self._normalize_term(pragmatic_raw, "pragmatic")
        epistemic_normed = self._normalize_term(epistemic_raw, "epistemic")
        instrumental_normed = self._normalize_term(instrumental_raw, "instrumental")

        # Compute total
        total = compute_efe_total(
            pragmatic_normed, epistemic_normed, instrumental_normed, self.config
        )

        # Validate sum invariant (on the normed values which are what was summed)
        self._validate_sum_invariant(
            total, pragmatic_normed, epistemic_normed, instrumental_normed
        )

        terms_dict = {
            "pragmatic": pragmatic_normed.detach(),
            "epistemic": epistemic_normed.detach(),
            "instrumental": instrumental_normed.detach(),
            "pragmatic_raw": pragmatic_raw.detach(),
            "epistemic_raw": epistemic_raw.detach(),
            "instrumental_raw": instrumental_raw.detach(),
            "total": total.detach(),
        }

        return total, terms_dict

    def compute_trajectory(
        self,
        trajectory_states: torch.Tensor,
        trajectory_obs_params: List[Tuple[torch.Tensor, torch.Tensor]],
        trajectory_actions: torch.Tensor,
        preference_params: Tuple[torch.Tensor, torch.Tensor],
        posterior_params_list: List[Tuple[torch.Tensor, torch.Tensor]],
        prior_params_list: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """Compute discounted EFE over a multi-step trajectory.

        Iterates over planning steps, applies per-step discount factors, and
        optionally normalizes by the horizon length.

        Args:
            trajectory_states: States along the trajectory, shape
                ``(B, H+1, state_dim)`` where H is the horizon.  Index 0 is
                the current state; index H is the final predicted state.
            trajectory_obs_params: List of H tuples, each containing
                ``(pred_mean, pred_log_var)`` for step t, shapes ``(B, obs_dim)``.
            trajectory_actions: Actions taken at each step, shape ``(B, H)``.
            preference_params: Tuple of ``(pref_mean, pref_log_var)`` shared
                across all steps.
            posterior_params_list: List of H tuples ``(post_mean, post_log_var)``
                for the posterior at each step.
            prior_params_list: List of H tuples ``(prior_mean, prior_log_var)``
                for the prior at each step.

        Returns:
            Tuple of:
                - ``efe_total``: Discounted sum of per-step EFEs, shape ``(B,)``.
                - ``per_step_terms``: List of H dictionaries, one per step,
                  each identical in structure to the output of
                  ``compute_single_step``.
        """
        H = len(trajectory_obs_params)
        if H == 0:
            B = trajectory_states.shape[0]
            device = trajectory_states.device
            return torch.zeros(B, device=device), []

        efe_accumulated = torch.zeros(
            trajectory_states.shape[0], device=trajectory_states.device
        )
        per_step_terms: List[Dict[str, torch.Tensor]] = []
        discount = 1.0

        for t in range(H):
            state_t = trajectory_states[:, t, :]
            next_state_t = trajectory_states[:, t + 1, :]
            action_t = trajectory_actions[:, t]
            obs_params_t = trajectory_obs_params[t]
            post_params_t = posterior_params_list[t]
            prior_params_t = prior_params_list[t]

            efe_t, terms_t = self.compute_single_step(
                predicted_obs_params=obs_params_t,
                posterior_params=post_params_t,
                prior_params=prior_params_t,
                state=state_t,
                action=action_t,
                next_state=next_state_t,
                preference_params=preference_params,
            )

            efe_accumulated = efe_accumulated + discount * efe_t
            terms_t["discount"] = torch.tensor(discount)
            per_step_terms.append(terms_t)

            discount *= self.config.discount_factor

        # Horizon normalization
        if self.config.normalize_by_horizon and H > 0:
            efe_accumulated = efe_accumulated / float(H)

        return efe_accumulated, per_step_terms

    def forward(
        self,
        predicted_obs_params: Tuple[torch.Tensor, torch.Tensor],
        posterior_params: Tuple[torch.Tensor, torch.Tensor],
        prior_params: Tuple[torch.Tensor, torch.Tensor],
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        preference_params: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Standard nn.Module forward pass, delegates to ``compute_single_step``.

        This makes EFEComputer compatible with standard PyTorch training loops
        that call ``model(inputs)``.

        Args:
            Same as ``compute_single_step``.

        Returns:
            Same as ``compute_single_step``.
        """
        return self.compute_single_step(
            predicted_obs_params=predicted_obs_params,
            posterior_params=posterior_params,
            prior_params=prior_params,
            state=state,
            action=action,
            next_state=next_state,
            preference_params=preference_params,
        )


# ===========================================================================
# Factory
# ===========================================================================


def create_efe_computer(
    state_dim: int = 64,
    action_dim: int = 10,
    hidden_dim: int = 256,
    pragmatic_weight: float = 1.0,
    epistemic_weight: float = 1.0,
    instrumental_weight: float = 0.1,
    normalize_terms: bool = False,
    **kwargs,
) -> EFEComputer:
    """Create an EFEComputer with the given configuration.

    This is the recommended entry point for downstream modules that need
    EFE computation (e.g. ``ActiveInferenceAgent``).

    Args:
        state_dim: Latent state dimension.
        action_dim: Number of discrete actions.
        hidden_dim: MLP width for empowerment networks.
        pragmatic_weight: Weight for the pragmatic term.
        epistemic_weight: Weight for the epistemic term.
        instrumental_weight: Weight for the instrumental term.
        normalize_terms: Whether to normalize terms before weighting.
        **kwargs: Additional keyword arguments passed to ``EFEConfig``.

    Returns:
        Configured ``EFEComputer`` instance.
    """
    config = EFEConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        pragmatic_weight=pragmatic_weight,
        epistemic_weight=epistemic_weight,
        instrumental_weight=instrumental_weight,
        normalize_terms=normalize_terms,
        **kwargs,
    )
    return EFEComputer(config)


# ===========================================================================
# Self-Test Suite
# ===========================================================================


def _make_gaussian_pair(
    B: int,
    D: int,
    device: torch.device,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a deterministic pair of Gaussian distributions for testing.

    Returns (mu1, log_var1, mu2, log_var2) all shaped (B, D).
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    mu1 = torch.randn(B, D, generator=gen).to(device)
    lv1 = torch.randn(B, D, generator=gen).to(device) * 0.5
    mu2 = torch.randn(B, D, generator=gen).to(device)
    lv2 = torch.randn(B, D, generator=gen).to(device) * 0.5
    return mu1, lv1, mu2, lv2


def _run_self_tests() -> None:
    """Run comprehensive self-tests for the EFE module.

    Verifies pure function properties, closed-form correctness, monotonicity,
    precision guarantees, gradient flow, and module-level integration.

    All tests are printed to stdout.  A summary of passed / failed / total
    is printed at the end.
    """
    device = torch.device("cpu")
    passed = 0
    failed = 0
    total_tests = 0

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal passed, failed, total_tests
        total_tests += 1
        if condition:
            passed += 1
            print(f"  [PASS] {name}")
        else:
            failed += 1
            msg = f"  [FAIL] {name}"
            if detail:
                msg += f"  -- {detail}"
            print(msg)

    B, obs_dim, state_dim, action_dim = 8, 16, 12, 5

    print("=" * 72)
    print("EFE Template Self-Test Suite")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 1. Pure function: deterministic (same input -> same output)
    # ------------------------------------------------------------------
    print("\n--- 1. Pure Function: Determinism ---")

    pred_mean = torch.randn(B, obs_dim, device=device)
    pred_log_var = torch.randn(B, obs_dim, device=device) * 0.5
    pref_mean = torch.randn(obs_dim, device=device)
    pref_log_var = torch.zeros(obs_dim, device=device) - 0.7  # precision ~2.0

    p1 = compute_pragmatic(pred_mean, pred_log_var, pref_mean, pref_log_var)
    p2 = compute_pragmatic(pred_mean, pred_log_var, pref_mean, pref_log_var)
    check("compute_pragmatic deterministic", torch.allclose(p1, p2, atol=1e-7))

    mu1, lv1, mu2, lv2 = _make_gaussian_pair(B, state_dim, device)
    e1 = compute_epistemic(mu1, lv1, mu2, lv2)
    e2 = compute_epistemic(mu1, lv1, mu2, lv2)
    check("compute_epistemic deterministic", torch.allclose(e1, e2, atol=1e-7))

    src_lp = torch.randn(B, device=device)
    plan_lp = torch.randn(B, device=device)
    state_ = torch.randn(B, state_dim, device=device)
    action_ = torch.randn(B, action_dim, device=device)
    next_state_ = torch.randn(B, state_dim, device=device)
    i1 = compute_instrumental(state_, action_, next_state_, src_lp, plan_lp)
    i2 = compute_instrumental(state_, action_, next_state_, src_lp, plan_lp)
    check("compute_instrumental deterministic", torch.allclose(i1, i2, atol=1e-7))

    # ------------------------------------------------------------------
    # 2. Pure function: no side effects (inputs not modified)
    # ------------------------------------------------------------------
    print("\n--- 2. Pure Function: No Side Effects ---")

    pm_clone = pred_mean.clone()
    plv_clone = pred_log_var.clone()
    _ = compute_pragmatic(pred_mean, pred_log_var, pref_mean, pref_log_var)
    check(
        "compute_pragmatic no side effects on pred_mean",
        torch.equal(pred_mean, pm_clone),
    )
    check(
        "compute_pragmatic no side effects on pred_log_var",
        torch.equal(pred_log_var, plv_clone),
    )

    mu1_clone = mu1.clone()
    lv1_clone = lv1.clone()
    _ = compute_epistemic(mu1, lv1, mu2, lv2)
    check(
        "compute_epistemic no side effects on posterior_mean",
        torch.equal(mu1, mu1_clone),
    )
    check(
        "compute_epistemic no side effects on posterior_log_var",
        torch.equal(lv1, lv1_clone),
    )

    slp_clone = src_lp.clone()
    plp_clone = plan_lp.clone()
    _ = compute_instrumental(state_, action_, next_state_, src_lp, plan_lp)
    check(
        "compute_instrumental no side effects on source_log_probs",
        torch.equal(src_lp, slp_clone),
    )
    check(
        "compute_instrumental no side effects on planning_log_probs",
        torch.equal(plan_lp, plp_clone),
    )

    # ------------------------------------------------------------------
    # 3. Pure function: independence (each term computed alone)
    # ------------------------------------------------------------------
    print("\n--- 3. Pure Function: Independence ---")

    p_solo = compute_pragmatic(pred_mean, pred_log_var, pref_mean, pref_log_var)
    e_solo = compute_epistemic(mu1, lv1, mu2, lv2)
    i_solo = compute_instrumental(state_, action_, next_state_, src_lp, plan_lp)

    # Compute pragmatic again after epistemic -- should be identical
    p_after_e = compute_pragmatic(pred_mean, pred_log_var, pref_mean, pref_log_var)
    check(
        "pragmatic independent of epistemic call order",
        torch.allclose(p_solo, p_after_e, atol=1e-7),
    )

    # Compute epistemic again after instrumental -- should be identical
    e_after_i = compute_epistemic(mu1, lv1, mu2, lv2)
    check(
        "epistemic independent of instrumental call order",
        torch.allclose(e_solo, e_after_i, atol=1e-7),
    )

    # ------------------------------------------------------------------
    # 4. Sum invariant
    # ------------------------------------------------------------------
    print("\n--- 4. Sum Invariant ---")

    config_test = EFEConfig(
        pragmatic_weight=1.3,
        epistemic_weight=0.8,
        instrumental_weight=0.2,
    )
    prag = torch.randn(B, device=device)
    epist = torch.randn(B, device=device).abs()
    instr = torch.randn(B, device=device)

    total_efe = compute_efe_total(prag, epist, instr, config_test)
    expected_efe = (
        config_test.pragmatic_weight * prag.float()
        + config_test.epistemic_weight * epist.float()
        - config_test.instrumental_weight * instr.float()
    )
    max_diff = (total_efe - expected_efe).abs().max().item()
    check(
        f"sum invariant holds (max diff = {max_diff:.2e})",
        max_diff < EFE_SUM_INVARIANT_TOL,
        f"max_diff={max_diff:.2e}, tol={EFE_SUM_INVARIANT_TOL:.1e}",
    )

    # Also test with default weights
    config_default = EFEConfig()
    total_default = compute_efe_total(prag, epist, instr, config_default)
    expected_default = (
        config_default.pragmatic_weight * prag.float()
        + config_default.epistemic_weight * epist.float()
        - config_default.instrumental_weight * instr.float()
    )
    diff_default = (total_default - expected_default).abs().max().item()
    check(
        f"sum invariant with default weights (max diff = {diff_default:.2e})",
        diff_default < EFE_SUM_INVARIANT_TOL,
    )

    # ------------------------------------------------------------------
    # 5. Closed-form verification: pragmatic
    # ------------------------------------------------------------------
    print("\n--- 5. Closed-Form Verification: Pragmatic ---")

    pm2 = torch.randn(B, obs_dim, device=device)
    plv2 = torch.randn(B, obs_dim, device=device) * 0.5
    pfm2 = torch.randn(obs_dim, device=device)
    pfp2 = torch.ones(obs_dim, device=device) * 1.5

    from_fn = compute_pragmatic(pm2, plv2, pfm2, pfp2)
    from_analytical = analytical_gaussian_pragmatic(
        pm2, torch.exp(plv2), pfm2, pfp2
    )
    max_prag_diff = (from_fn - from_analytical).abs().max().item()
    check(
        f"pragmatic matches analytical (max diff = {max_prag_diff:.2e})",
        max_prag_diff < 1e-4,
        f"max_diff={max_prag_diff}",
    )

    # ------------------------------------------------------------------
    # 6. Closed-form verification: epistemic (KL)
    # ------------------------------------------------------------------
    print("\n--- 6. Closed-Form Verification: Epistemic (KL) ---")

    mu_a, lv_a, mu_b, lv_b = _make_gaussian_pair(B, state_dim, device, seed=99)
    from_fn_kl = compute_epistemic(mu_a, lv_a, mu_b, lv_b)
    from_analytical_kl = analytical_gaussian_kl(
        mu_a, torch.exp(lv_a), mu_b, torch.exp(lv_b)
    )
    max_kl_diff = (from_fn_kl - from_analytical_kl).abs().max().item()
    check(
        f"epistemic matches analytical KL (max diff = {max_kl_diff:.2e})",
        max_kl_diff < 1e-4,
        f"max_diff={max_kl_diff}",
    )

    # KL of identical distributions should be zero
    kl_zero = compute_epistemic(mu_a, lv_a, mu_a, lv_a)
    max_zero = kl_zero.abs().max().item()
    check(
        f"KL(q||q) = 0 (max = {max_zero:.2e})",
        max_zero < 1e-5,
    )

    # ------------------------------------------------------------------
    # 7. Monotonicity: worse preference match -> higher pragmatic
    # ------------------------------------------------------------------
    print("\n--- 7. Monotonicity: Pragmatic ---")

    close_mean = pref_mean.unsqueeze(0).expand(B, -1) + 0.01 * torch.randn(B, obs_dim)
    far_mean = pref_mean.unsqueeze(0).expand(B, -1) + 5.0 * torch.randn(B, obs_dim)
    zero_lv = torch.zeros(B, obs_dim, device=device)
    unit_prec = torch.ones(obs_dim, device=device)

    p_close = compute_pragmatic(close_mean, zero_lv, pref_mean, unit_prec)
    p_far = compute_pragmatic(far_mean, zero_lv, pref_mean, unit_prec)
    # On average, far observations should yield higher pragmatic value
    check(
        "farther from preference -> higher pragmatic (batch mean)",
        p_far.mean().item() > p_close.mean().item(),
        f"close={p_close.mean().item():.4f}, far={p_far.mean().item():.4f}",
    )

    # ------------------------------------------------------------------
    # 8. Monotonicity: more uncertainty -> higher epistemic
    # ------------------------------------------------------------------
    print("\n--- 8. Monotonicity: Epistemic ---")

    mu_same = torch.zeros(B, state_dim, device=device)
    lv_low = torch.full((B, state_dim), -2.0, device=device)   # low variance
    lv_high = torch.full((B, state_dim), 2.0, device=device)   # high variance
    lv_prior = torch.zeros(B, state_dim, device=device)         # unit prior variance

    e_low = compute_epistemic(mu_same, lv_low, mu_same, lv_prior)
    e_high = compute_epistemic(mu_same, lv_high, mu_same, lv_prior)
    # More posterior variance -> higher KL (farther from prior)
    check(
        "higher posterior variance -> higher epistemic",
        e_high.mean().item() > e_low.mean().item(),
        f"low_var={e_low.mean().item():.4f}, high_var={e_high.mean().item():.4f}",
    )

    # ------------------------------------------------------------------
    # 9. fp32 precision: inputs in fp16, output in fp32
    # ------------------------------------------------------------------
    print("\n--- 9. fp32 Precision ---")

    pm_16 = pred_mean.half()
    plv_16 = pred_log_var.half()
    pfm_16 = pref_mean.half()
    pfp_16 = pref_log_var.half()

    p_from_16 = compute_pragmatic(pm_16, plv_16, pfm_16, pfp_16)
    check(
        "pragmatic output dtype is fp32 from fp16 inputs",
        p_from_16.dtype == torch.float32,
        f"got {p_from_16.dtype}",
    )

    e_from_16 = compute_epistemic(
        mu1.half(), lv1.half(), mu2.half(), lv2.half()
    )
    check(
        "epistemic output dtype is fp32 from fp16 inputs",
        e_from_16.dtype == torch.float32,
        f"got {e_from_16.dtype}",
    )

    i_from_16 = compute_instrumental(
        state_.half(), action_.half(), next_state_.half(),
        src_lp.half(), plan_lp.half(),
    )
    check(
        "instrumental output dtype is fp32 from fp16 inputs",
        i_from_16.dtype == torch.float32,
        f"got {i_from_16.dtype}",
    )

    # ------------------------------------------------------------------
    # 10. Gradient flow through each term independently
    # ------------------------------------------------------------------
    print("\n--- 10. Gradient Flow ---")

    pm_g = torch.randn(B, obs_dim, requires_grad=True, device=device)
    plv_g = torch.randn(B, obs_dim, requires_grad=True, device=device) * 0.5
    p_g = compute_pragmatic(pm_g, plv_g, pref_mean, pref_log_var)
    loss_p = p_g.sum()
    loss_p.backward()
    check(
        "gradient flows through pragmatic to pred_mean",
        pm_g.grad is not None and pm_g.grad.abs().sum().item() > 0,
    )
    check(
        "gradient flows through pragmatic to pred_log_var",
        plv_g.grad is not None and plv_g.grad.abs().sum().item() > 0,
    )

    mu_g = torch.randn(B, state_dim, requires_grad=True, device=device)
    lv_g = torch.randn(B, state_dim, requires_grad=True, device=device) * 0.5
    mu_p = torch.randn(B, state_dim, device=device)
    lv_p = torch.randn(B, state_dim, device=device) * 0.5
    e_g = compute_epistemic(mu_g, lv_g, mu_p, lv_p)
    loss_e = e_g.sum()
    loss_e.backward()
    check(
        "gradient flows through epistemic to posterior_mean",
        mu_g.grad is not None and mu_g.grad.abs().sum().item() > 0,
    )
    check(
        "gradient flows through epistemic to posterior_log_var",
        lv_g.grad is not None and lv_g.grad.abs().sum().item() > 0,
    )

    slp_g = torch.randn(B, requires_grad=True, device=device)
    plp_g = torch.randn(B, requires_grad=True, device=device)
    i_g = compute_instrumental(
        state_, action_, next_state_, slp_g, plp_g,
    )
    loss_i = i_g.sum()
    loss_i.backward()
    check(
        "gradient flows through instrumental to source_log_probs",
        slp_g.grad is not None and slp_g.grad.abs().sum().item() > 0,
    )
    check(
        "gradient flows through instrumental to planning_log_probs",
        plp_g.grad is not None and plp_g.grad.abs().sum().item() > 0,
    )

    # ------------------------------------------------------------------
    # 11. EFEComputer single step via module
    # ------------------------------------------------------------------
    print("\n--- 11. EFEComputer Single Step ---")

    config_comp = EFEConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=64,
        pragmatic_weight=1.0,
        epistemic_weight=1.0,
        instrumental_weight=0.1,
    )
    computer = EFEComputer(config_comp).to(device)

    pred_obs_p = (
        torch.randn(B, obs_dim, device=device),
        torch.randn(B, obs_dim, device=device) * 0.5,
    )
    post_p = (
        torch.randn(B, state_dim, device=device),
        torch.randn(B, state_dim, device=device) * 0.5,
    )
    pri_p = (
        torch.randn(B, state_dim, device=device),
        torch.randn(B, state_dim, device=device) * 0.5,
    )
    st = torch.randn(B, state_dim, device=device)
    act = torch.randint(0, action_dim, (B,), device=device)
    nst = torch.randn(B, state_dim, device=device)
    pref_p = (
        torch.randn(obs_dim, device=device),
        torch.ones(obs_dim, device=device),
    )

    efe_val, terms = computer.compute_single_step(
        pred_obs_p, post_p, pri_p, st, act, nst, pref_p
    )
    check("EFEComputer returns (B,) total", efe_val.shape == (B,))
    check("EFEComputer terms dict has pragmatic", "pragmatic" in terms)
    check("EFEComputer terms dict has epistemic", "epistemic" in terms)
    check("EFEComputer terms dict has instrumental", "instrumental" in terms)
    check("EFEComputer terms dict has total", "total" in terms)

    # Verify sum invariant on module output
    recomputed = (
        config_comp.pragmatic_weight * terms["pragmatic"]
        + config_comp.epistemic_weight * terms["epistemic"]
        - config_comp.instrumental_weight * terms["instrumental"]
    )
    inv_diff = (terms["total"] - recomputed).abs().max().item()
    check(
        f"EFEComputer sum invariant (max diff = {inv_diff:.2e})",
        inv_diff < EFE_SUM_INVARIANT_TOL,
    )

    # ------------------------------------------------------------------
    # 12. EFEComputer trajectory computation
    # ------------------------------------------------------------------
    print("\n--- 12. EFEComputer Trajectory ---")

    H = 5  # horizon
    traj_states = torch.randn(B, H + 1, state_dim, device=device)
    traj_obs_params = [
        (torch.randn(B, obs_dim, device=device),
         torch.randn(B, obs_dim, device=device) * 0.5)
        for _ in range(H)
    ]
    traj_actions = torch.randint(0, action_dim, (B, H), device=device)
    traj_post = [
        (torch.randn(B, state_dim, device=device),
         torch.randn(B, state_dim, device=device) * 0.5)
        for _ in range(H)
    ]
    traj_prior = [
        (torch.randn(B, state_dim, device=device),
         torch.randn(B, state_dim, device=device) * 0.5)
        for _ in range(H)
    ]

    efe_traj, step_terms = computer.compute_trajectory(
        traj_states, traj_obs_params, traj_actions, pref_p, traj_post, traj_prior
    )
    check("trajectory returns (B,) total", efe_traj.shape == (B,))
    check(f"trajectory returns {H} step term dicts", len(step_terms) == H)

    # ------------------------------------------------------------------
    # 13. Horizon normalization: EFE/H consistent across H
    # ------------------------------------------------------------------
    print("\n--- 13. Horizon Normalization ---")

    config_hn = EFEConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=64,
        discount_factor=1.0,      # no discount to isolate horizon effect
        normalize_by_horizon=True,
    )
    comp_hn = EFEComputer(config_hn).to(device)

    def _make_uniform_trajectory(horizon: int):
        """Build a trajectory where every step has the same params."""
        ts = torch.randn(B, 1, state_dim, device=device).expand(B, horizon + 1, state_dim).contiguous()
        obs_p = [
            (torch.randn(B, obs_dim, device=device),
             torch.zeros(B, obs_dim, device=device))
        ] * horizon
        acts = torch.randint(0, action_dim, (B, horizon), device=device)
        po = [(torch.randn(B, state_dim, device=device),
               torch.zeros(B, state_dim, device=device))] * horizon
        pr = [(torch.randn(B, state_dim, device=device),
               torch.zeros(B, state_dim, device=device))] * horizon
        return ts, obs_p, acts, po, pr

    # Generate consistent random data shared across horizons
    torch.manual_seed(123)
    base_state = torch.randn(B, state_dim, device=device)
    base_obs_mean = torch.randn(B, obs_dim, device=device)
    base_obs_lv = torch.zeros(B, obs_dim, device=device)
    base_post_mean = torch.randn(B, state_dim, device=device)
    base_post_lv = torch.zeros(B, state_dim, device=device)
    base_prior_mean = torch.randn(B, state_dim, device=device)
    base_prior_lv = torch.zeros(B, state_dim, device=device)
    base_act = torch.randint(0, action_dim, (B,), device=device)

    def _make_constant_traj(horizon: int):
        ts = base_state.unsqueeze(1).expand(B, horizon + 1, state_dim).contiguous()
        obs_p = [(base_obs_mean, base_obs_lv)] * horizon
        acts = base_act.unsqueeze(1).expand(B, horizon).contiguous()
        po = [(base_post_mean, base_post_lv)] * horizon
        pr = [(base_prior_mean, base_prior_lv)] * horizon
        return ts, obs_p, acts, po, pr

    efe_h3, _ = comp_hn.compute_trajectory(*_make_constant_traj(3), pref_p, *_make_constant_traj(3)[3:])
    efe_h10, _ = comp_hn.compute_trajectory(*_make_constant_traj(10), pref_p, *_make_constant_traj(10)[3:])

    # With horizon normalization and identical steps, EFE/H should be
    # roughly the same for different H (both are mean of same per-step value)
    ratio = (efe_h3.mean() / (efe_h10.mean() + 1e-8)).abs().item()
    check(
        f"horizon normalization: ratio H=3/H=10 ~ 1.0 (got {ratio:.3f})",
        0.7 < ratio < 1.4,
        f"ratio={ratio:.4f}",
    )

    # ------------------------------------------------------------------
    # 14. Discount factor: discounted < undiscounted
    # ------------------------------------------------------------------
    print("\n--- 14. Discount Factor ---")

    config_disc = EFEConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=64,
        discount_factor=0.9,
        normalize_by_horizon=False,
    )
    comp_disc = EFEComputer(config_disc).to(device)

    config_nodisc = EFEConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=64,
        discount_factor=1.0,
        normalize_by_horizon=False,
    )
    comp_nodisc = EFEComputer(config_nodisc).to(device)

    # Copy weights so networks are identical
    comp_nodisc.load_state_dict(comp_disc.state_dict(), strict=False)

    H_disc = 8
    ts_d = torch.randn(B, H_disc + 1, state_dim, device=device)
    obs_d = [
        (torch.randn(B, obs_dim, device=device).abs(),
         torch.randn(B, obs_dim, device=device) * 0.5)
        for _ in range(H_disc)
    ]
    acts_d = torch.randint(0, action_dim, (B, H_disc), device=device)
    po_d = [
        (torch.randn(B, state_dim, device=device),
         torch.randn(B, state_dim, device=device) * 0.5)
        for _ in range(H_disc)
    ]
    pr_d = [
        (torch.randn(B, state_dim, device=device),
         torch.randn(B, state_dim, device=device) * 0.5)
        for _ in range(H_disc)
    ]

    efe_disc, _ = comp_disc.compute_trajectory(
        ts_d, obs_d, acts_d, pref_p, po_d, pr_d
    )
    efe_nodisc, _ = comp_nodisc.compute_trajectory(
        ts_d, obs_d, acts_d, pref_p, po_d, pr_d
    )

    # Discounted magnitude should be smaller on average because future
    # terms are down-weighted (gamma < 1).
    disc_mag = efe_disc.abs().mean().item()
    nodisc_mag = efe_nodisc.abs().mean().item()
    check(
        f"discounted magnitude ({disc_mag:.4f}) <= undiscounted ({nodisc_mag:.4f})",
        disc_mag <= nodisc_mag + 1e-4,
        f"disc={disc_mag:.4f}, nodisc={nodisc_mag:.4f}",
    )

    # ------------------------------------------------------------------
    # 15. Running normalization: mean ~ 0, std ~ 1 after warmup
    # ------------------------------------------------------------------
    print("\n--- 15. Running Normalization ---")

    rn = RunningNormalization(
        term_names=["test_term"],
        momentum=0.05,
        warmup_steps=50,
    )

    # Feed 200 samples from N(10, 4) (mean=10, std=2)
    for _ in range(200):
        sample = 10.0 + 2.0 * torch.randn(32)
        rn.update(sample, "test_term")

    stats = rn.get_statistics("test_term")
    check(
        f"running mean near 10.0 (got {stats['mean']:.2f})",
        abs(stats["mean"] - 10.0) < 2.0,
    )
    check(
        f"running std near 2.0 (got {stats['std']:.2f})",
        abs(stats["std"] - 2.0) < 1.5,
    )

    # Normalize and check output statistics
    test_batch = 10.0 + 2.0 * torch.randn(1000)
    normed = rn.normalize(test_batch, "test_term")
    check(
        f"normalized mean near 0 (got {normed.mean().item():.3f})",
        abs(normed.mean().item()) < 1.0,
    )
    check(
        f"normalized std near 1 (got {normed.std().item():.3f})",
        abs(normed.std().item() - 1.0) < 1.0,
    )

    # ------------------------------------------------------------------
    # 16. Empowerment estimator: positive for controllable states
    # ------------------------------------------------------------------
    print("\n--- 16. Empowerment Estimator ---")

    emp_est = EmpowermentEstimator(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=64,
    ).to(device)

    s_emp = torch.randn(B, state_dim, device=device)
    a_emp = torch.randint(0, action_dim, (B,), device=device)
    ns_emp = torch.randn(B, state_dim, device=device)

    emp_val, src_lp_out, plan_lp_out = emp_est(s_emp, a_emp, ns_emp)
    check("empowerment returns (B,)", emp_val.shape == (B,))
    check("source log probs returns (B,)", src_lp_out.shape == (B,))
    check("planning log probs returns (B,)", plan_lp_out.shape == (B,))

    # Train empowerment estimator briefly on distinguishable transitions
    # Action 0 -> next_state moves right; Action 1 -> next_state moves left
    # This should yield positive empowerment after training.
    opt_emp = torch.optim.Adam(emp_est.parameters(), lr=1e-3)
    for _ in range(200):
        s_t = torch.randn(32, state_dim, device=device)
        a_t = torch.randint(0, action_dim, (32,), device=device)
        # Make transitions distinguishable per action
        ns_t = s_t.clone()
        for ai in range(action_dim):
            mask = (a_t == ai)
            if mask.any():
                ns_t[mask, ai % state_dim] += 2.0 * (ai + 1)

        _, src_lp_t, plan_lp_t = emp_est(s_t, a_t, ns_t)
        # Maximize empowerment (maximize planning_lp - source_lp)
        # Equivalent: minimize -planning_lp (make planning net better)
        loss_emp = -plan_lp_t.mean()
        opt_emp.zero_grad()
        loss_emp.backward()
        opt_emp.step()

    # After training, empowerment should be positive on average
    with torch.no_grad():
        s_eval = torch.randn(64, state_dim, device=device)
        a_eval = torch.randint(0, action_dim, (64,), device=device)
        ns_eval = s_eval.clone()
        for ai in range(action_dim):
            mask = (a_eval == ai)
            if mask.any():
                ns_eval[mask, ai % state_dim] += 2.0 * (ai + 1)
        emp_eval, _, _ = emp_est(s_eval, a_eval, ns_eval)

    check(
        f"empowerment positive after training (mean={emp_eval.mean().item():.3f})",
        emp_eval.mean().item() > 0.0,
        f"mean={emp_eval.mean().item():.4f}",
    )

    # ------------------------------------------------------------------
    # 17. Batch independence
    # ------------------------------------------------------------------
    print("\n--- 17. Batch Independence ---")

    # Compute EFE for batch of 4 vs individual elements
    B_ind = 4
    pm_ind = torch.randn(B_ind, obs_dim, device=device)
    plv_ind = torch.randn(B_ind, obs_dim, device=device) * 0.5
    pfm_ind = torch.randn(obs_dim, device=device)
    pfp_ind = torch.ones(obs_dim, device=device)

    p_batch = compute_pragmatic(pm_ind, plv_ind, pfm_ind, pfp_ind)
    p_individuals = []
    for b in range(B_ind):
        p_i = compute_pragmatic(
            pm_ind[b:b+1], plv_ind[b:b+1], pfm_ind, pfp_ind
        )
        p_individuals.append(p_i)
    p_cat = torch.cat(p_individuals, dim=0)
    batch_diff = (p_batch - p_cat).abs().max().item()
    check(
        f"pragmatic batch independent (max diff = {batch_diff:.2e})",
        batch_diff < 1e-6,
    )

    mu_ind1, lv_ind1, mu_ind2, lv_ind2 = _make_gaussian_pair(B_ind, state_dim, device, seed=77)
    e_batch = compute_epistemic(mu_ind1, lv_ind1, mu_ind2, lv_ind2)
    e_individuals = []
    for b in range(B_ind):
        e_i = compute_epistemic(
            mu_ind1[b:b+1], lv_ind1[b:b+1],
            mu_ind2[b:b+1], lv_ind2[b:b+1],
        )
        e_individuals.append(e_i)
    e_cat = torch.cat(e_individuals, dim=0)
    e_batch_diff = (e_batch - e_cat).abs().max().item()
    check(
        f"epistemic batch independent (max diff = {e_batch_diff:.2e})",
        e_batch_diff < 1e-6,
    )

    # ------------------------------------------------------------------
    # 18. Output shapes
    # ------------------------------------------------------------------
    print("\n--- 18. Output Shapes ---")

    for test_B in [1, 4, 16]:
        p_shape = compute_pragmatic(
            torch.randn(test_B, obs_dim), torch.randn(test_B, obs_dim),
            torch.randn(obs_dim), torch.ones(obs_dim),
        )
        check(f"pragmatic shape (B={test_B})", p_shape.shape == (test_B,))

    for test_B in [1, 4, 16]:
        e_shape = compute_epistemic(
            torch.randn(test_B, state_dim), torch.randn(test_B, state_dim),
            torch.randn(test_B, state_dim), torch.randn(test_B, state_dim),
        )
        check(f"epistemic shape (B={test_B})", e_shape.shape == (test_B,))

    # ------------------------------------------------------------------
    # 19. Instrumental sign convention
    # ------------------------------------------------------------------
    print("\n--- 19. Instrumental Sign Convention ---")

    # When planning_log_prob > source_log_prob, instrumental should be positive
    high_plan = torch.tensor([0.0, 0.0, 0.0, 0.0])
    low_source = torch.tensor([-3.0, -3.0, -3.0, -3.0])
    dummy_s = torch.randn(4, state_dim)
    dummy_a = torch.randn(4, action_dim)
    dummy_ns = torch.randn(4, state_dim)

    inst_pos = compute_instrumental(dummy_s, dummy_a, dummy_ns, low_source, high_plan)
    check(
        "instrumental positive when planning > source",
        (inst_pos > 0).all().item(),
        f"values={inst_pos.tolist()}",
    )

    # When planning_log_prob < source_log_prob, instrumental should be negative
    low_plan = torch.tensor([-3.0, -3.0, -3.0, -3.0])
    high_source = torch.tensor([0.0, 0.0, 0.0, 0.0])
    inst_neg = compute_instrumental(dummy_s, dummy_a, dummy_ns, high_source, low_plan)
    check(
        "instrumental negative when planning < source",
        (inst_neg < 0).all().item(),
        f"values={inst_neg.tolist()}",
    )

    # ------------------------------------------------------------------
    # 20. Total EFE: all terms summed (instrumental is negative empowerment)
    # ------------------------------------------------------------------
    print("\n--- 20. Total EFE: All Terms Summed ---")

    cfg20 = EFEConfig(
        pragmatic_weight=1.0,
        epistemic_weight=1.0,
        instrumental_weight=1.0,
    )
    fixed_p = torch.tensor([5.0, 5.0])
    fixed_e = torch.tensor([3.0, 3.0])
    fixed_i = torch.tensor([-10.0, -10.0])  # negative empowerment

    total20 = compute_efe_total(fixed_p, fixed_e, fixed_i, cfg20)
    # total = 1.0*5 + 1.0*3 + 1.0*(-10) = -2.0
    check(
        "total = w_p*p + w_e*e + w_i*i",
        torch.allclose(total20, torch.tensor([-2.0, -2.0])),
        f"expected [-2, -2], got {total20.tolist()}",
    )

    # ------------------------------------------------------------------
    # 21. EFEComputer with normalization
    # ------------------------------------------------------------------
    print("\n--- 21. EFEComputer with Running Normalization ---")

    config_norm = EFEConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=64,
        normalize_terms=True,
        term_norm_mode="running",
    )
    comp_norm = EFEComputer(config_norm).to(device)
    comp_norm.train()

    # Run warmup
    for _ in range(120):
        _pred = (
            torch.randn(B, obs_dim, device=device),
            torch.randn(B, obs_dim, device=device) * 0.5,
        )
        _post = (
            torch.randn(B, state_dim, device=device),
            torch.randn(B, state_dim, device=device) * 0.5,
        )
        _pri = (
            torch.randn(B, state_dim, device=device),
            torch.randn(B, state_dim, device=device) * 0.5,
        )
        _s = torch.randn(B, state_dim, device=device)
        _a = torch.randint(0, action_dim, (B,), device=device)
        _ns = torch.randn(B, state_dim, device=device)

        comp_norm.compute_single_step(
            _pred, _post, _pri, _s, _a, _ns, pref_p
        )

    check(
        "running_norm exists after warmup",
        comp_norm.running_norm is not None,
    )
    prag_stats = comp_norm.running_norm.get_statistics("pragmatic")
    check(
        f"pragmatic running norm count >= 100 (got {prag_stats['count']})",
        prag_stats["count"] >= 100,
    )

    # ------------------------------------------------------------------
    # 22. EFEComputer with sigmoid normalization
    # ------------------------------------------------------------------
    print("\n--- 22. EFEComputer with Sigmoid Normalization ---")

    config_sig = EFEConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=64,
        normalize_terms=True,
        term_norm_mode="sigmoid",
    )
    comp_sig = EFEComputer(config_sig).to(device)
    efe_sig, terms_sig = comp_sig.compute_single_step(
        pred_obs_p, post_p, pri_p, st, act, nst, pref_p
    )
    # Sigmoid normalization means pragmatic/epistemic/instrumental are in [0,1]
    check(
        "sigmoid-normed pragmatic in [0, 1]",
        (terms_sig["pragmatic"] >= 0).all().item()
        and (terms_sig["pragmatic"] <= 1).all().item(),
    )
    check(
        "sigmoid-normed epistemic in [0, 1]",
        (terms_sig["epistemic"] >= 0).all().item()
        and (terms_sig["epistemic"] <= 1).all().item(),
    )

    # ------------------------------------------------------------------
    # 23. Forward pass (nn.Module interface)
    # ------------------------------------------------------------------
    print("\n--- 23. nn.Module Forward Pass ---")

    efe_fwd, terms_fwd = computer(pred_obs_p, post_p, pri_p, st, act, nst, pref_p)
    check("forward() returns same shape as compute_single_step", efe_fwd.shape == (B,))
    check("forward() returns terms dict", "total" in terms_fwd)

    # ------------------------------------------------------------------
    # 24. Factory function
    # ------------------------------------------------------------------
    print("\n--- 24. Factory Function ---")

    comp_factory = create_efe_computer(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=32,
        pragmatic_weight=2.0,
    )
    check("factory returns EFEComputer", isinstance(comp_factory, EFEComputer))
    check(
        "factory config has pragmatic_weight=2.0",
        comp_factory.config.pragmatic_weight == 2.0,
    )

    # ------------------------------------------------------------------
    # 25. Analytical utilities: edge cases
    # ------------------------------------------------------------------
    print("\n--- 25. Analytical Utilities: Edge Cases ---")

    # Identical distributions: KL = 0
    mu_id = torch.randn(B, state_dim)
    var_id = torch.rand(B, state_dim) + 0.1
    kl_id = analytical_gaussian_kl(mu_id, var_id, mu_id, var_id)
    check(
        f"analytical KL of identical = 0 (max={kl_id.abs().max().item():.2e})",
        kl_id.abs().max().item() < 1e-5,
    )

    # Pragmatic at exact preference: should be minimum
    exact_pred = pref_mean.unsqueeze(0).expand(B, -1)
    exact_var = torch.zeros(B, obs_dim)
    p_exact = analytical_gaussian_pragmatic(exact_pred, exact_var, pref_mean, pref_log_var)
    # This is the NLL at the mean, which is 0.5 * (log(2pi) + log_var) per dim
    expected_min_per_dim = 0.5 * (LOG_2PI + pref_log_var)
    expected_min = expected_min_per_dim.sum().item()
    check(
        f"analytical pragmatic at preference = theoretical min ({p_exact[0].item():.3f} vs {expected_min:.3f})",
        abs(p_exact[0].item() - expected_min) < 0.01,
    )

    # ------------------------------------------------------------------
    # 26. EmpowermentEstimator state empowerment (with simple forward model)
    # ------------------------------------------------------------------
    print("\n--- 26. EmpowermentEstimator State Empowerment ---")

    class SimpleForwardModel(nn.Module):
        """Minimal forward model for testing estimate_state_empowerment."""
        def __init__(self, s_dim, a_dim):
            super().__init__()
            self.net = nn.Linear(s_dim + a_dim, s_dim)

        def predict_next_state(self, state, action_onehot):
            x = torch.cat([state, action_onehot], dim=-1)
            mu = self.net(x)
            lv = torch.zeros_like(mu)
            return mu, lv

    fwd_model = SimpleForwardModel(state_dim, action_dim).to(device)
    emp_est2 = EmpowermentEstimator(state_dim, action_dim, 64).to(device)
    s_se = torch.randn(B, state_dim, device=device)
    state_emp = emp_est2.estimate_state_empowerment(s_se, fwd_model, num_samples=16)
    check("state empowerment returns (B,)", state_emp.shape == (B,))

    # ------------------------------------------------------------------
    # 27. Large batch stability
    # ------------------------------------------------------------------
    print("\n--- 27. Large Batch Stability ---")

    big_B = 256
    p_big = compute_pragmatic(
        torch.randn(big_B, obs_dim), torch.randn(big_B, obs_dim) * 0.5,
        torch.randn(obs_dim), torch.ones(obs_dim),
    )
    check("large batch (256) no NaN", not torch.isnan(p_big).any().item())
    check("large batch (256) no Inf", not torch.isinf(p_big).any().item())

    e_big = compute_epistemic(
        torch.randn(big_B, state_dim), torch.randn(big_B, state_dim) * 0.5,
        torch.randn(big_B, state_dim), torch.randn(big_B, state_dim) * 0.5,
    )
    check("large batch epistemic no NaN", not torch.isnan(e_big).any().item())
    check("large batch epistemic no Inf", not torch.isinf(e_big).any().item())

    # ------------------------------------------------------------------
    # 28. Extreme input robustness
    # ------------------------------------------------------------------
    print("\n--- 28. Extreme Input Robustness ---")

    # Very large log-variance
    extreme_lv = torch.full((B, obs_dim), 20.0)
    p_extreme = compute_pragmatic(
        torch.zeros(B, obs_dim), extreme_lv,
        torch.zeros(obs_dim), torch.ones(obs_dim),
    )
    check("extreme log_var=20: no NaN", not torch.isnan(p_extreme).any().item())

    # Very small log-variance
    small_lv = torch.full((B, obs_dim), -20.0)
    p_small = compute_pragmatic(
        torch.zeros(B, obs_dim), small_lv,
        torch.zeros(obs_dim), torch.ones(obs_dim),
    )
    check("extreme log_var=-20: no NaN", not torch.isnan(p_small).any().item())

    # Zero precision (edge case for pragmatic)
    p_zero_prec = compute_pragmatic(
        torch.randn(B, obs_dim), torch.randn(B, obs_dim),
        torch.randn(obs_dim), torch.full((obs_dim,), 1e-8),
    )
    check("near-zero precision: no NaN", not torch.isnan(p_zero_prec).any().item())

    # ------------------------------------------------------------------
    # 29. Empty trajectory
    # ------------------------------------------------------------------
    print("\n--- 29. Empty Trajectory ---")

    efe_empty, terms_empty = computer.compute_trajectory(
        torch.randn(B, 1, state_dim, device=device),  # H+1=1 -> H=0
        [],   # no obs params
        torch.zeros(B, 0, dtype=torch.long, device=device),
        pref_p,
        [],   # no posterior params
        [],   # no prior params
    )
    check("empty trajectory returns zeros", torch.allclose(efe_empty, torch.zeros(B, device=device)))
    check("empty trajectory returns no step terms", len(terms_empty) == 0)

    # ------------------------------------------------------------------
    # 30. Config default values
    # ------------------------------------------------------------------
    print("\n--- 30. Config Default Values ---")

    cfg_def = EFEConfig()
    check("default pragmatic_weight = 1.0", cfg_def.pragmatic_weight == 1.0)
    check("default epistemic_weight = 1.0", cfg_def.epistemic_weight == 1.0)
    check("default instrumental_weight = 0.1", cfg_def.instrumental_weight == 0.1)
    check("default num_samples = 32", cfg_def.num_samples == 32)
    check("default discount_factor = 0.99", cfg_def.discount_factor == 0.99)
    check("default normalize_by_horizon = True", cfg_def.normalize_by_horizon is True)
    check("default normalize_terms = False", cfg_def.normalize_terms is False)
    check("default term_norm_mode = 'running'", cfg_def.term_norm_mode == "running")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"RESULTS: {passed} passed, {failed} failed, {total_tests} total")
    print("=" * 72)
    if failed == 0:
        print("ALL TESTS PASSED.")
    else:
        print(f"WARNING: {failed} test(s) FAILED.")


# ===========================================================================
# Main
# ===========================================================================


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _run_self_tests()
