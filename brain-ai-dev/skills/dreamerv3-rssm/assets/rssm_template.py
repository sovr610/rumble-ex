"""
RSSM Template — DreamerV3-Style Recurrent State Space Model

Implements:
  RSSM: Full world model with observe (encode) and imagine (rollout) methods.
        Combines BlockGRU (deterministic) + categorical state (stochastic).

Dependencies (from other asset templates):
  block_gru_template.py      -> BlockGRU
  categorical_state_template -> PriorNet, PosteriorNet, sample_straight_through
  symlog_twohot_template.py  -> SymlogTwohot
  rssm_config_template.py    -> RSSMConfig, RSSMState, ImaginedTrajectory
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Inline implementations of dependencies (so this file is self-contained)
# ---------------------------------------------------------------------------

def _symlog(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def _symexp(x: Tensor) -> Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def _unimix_probs(logits: Tensor, unimix: float = 0.01) -> Tensor:
    num_classes = logits.shape[-1]
    soft = torch.softmax(logits, dim=-1)
    return (1.0 - unimix) * soft + unimix / num_classes


def _sample_straight_through(logits: Tensor, unimix: float = 0.01) -> Tensor:
    import torch.nn.functional as F
    probs = _unimix_probs(logits, unimix)
    indices = probs.argmax(dim=-1)
    z_hard = F.one_hot(indices, logits.shape[-1]).to(probs.dtype)
    return z_hard - probs.detach() + probs


def _make_rmsnorm(dim: int) -> nn.Module:
    if hasattr(nn, "RMSNorm"):
        return nn.RMSNorm(dim)
    return _RMSNormFallback(dim)


class _RMSNormFallback(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


def _make_mlp(in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    cur = in_dim
    for _ in range(num_layers):
        layers += [nn.Linear(cur, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()]
        cur = hidden_dim
    layers.append(nn.Linear(cur, out_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Config and output dataclasses (inline for self-containment)
# ---------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Any


@dataclass
class RSSMConfig:
    deter_dim: int = 1024
    stoch_dim: int = 32
    num_classes: int = 32
    hidden_dim: int = 1024
    num_layers: int = 2
    activation: str = "silu"
    norm: str = "layernorm"
    unimix: float = 0.01

    @property
    def stoch_flat_dim(self) -> int:
        return self.stoch_dim * self.num_classes

    @property
    def feature_dim(self) -> int:
        return self.deter_dim + self.stoch_flat_dim


@dataclass
class RSSMState:
    deter: Tensor   # (batch, deter_dim)
    stoch: Tensor   # (batch, stoch_dim, num_classes)
    logits: Tensor  # (batch, stoch_dim, num_classes)

    @property
    def features(self) -> Tensor:
        return torch.cat([self.deter, self.stoch.flatten(start_dim=-2)], dim=-1)

    @property
    def batch_size(self) -> int:
        return self.deter.shape[0]

    def detach(self) -> "RSSMState":
        return RSSMState(self.deter.detach(), self.stoch.detach(), self.logits.detach())


@dataclass
class ImaginedTrajectory:
    features: Tensor        # (horizon, batch, feature_dim)
    actions: Tensor         # (horizon, batch, action_dim)
    reward_logits: Tensor   # (horizon, batch, num_bins)
    continue_logits: Tensor # (horizon, batch, 1)

    @property
    def horizon(self) -> int:
        return self.features.shape[0]

    @property
    def batch_size(self) -> int:
        return self.features.shape[1]

    @property
    def continue_probs(self) -> Tensor:
        return torch.sigmoid(self.continue_logits.squeeze(-1))


# ---------------------------------------------------------------------------
# BlockGRU (self-contained version)
# ---------------------------------------------------------------------------

class BlockGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 1024) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        gate_in = hidden_dim * 2
        self.gate_r = nn.Linear(gate_in, hidden_dim)
        self.gate_z = nn.Linear(gate_in, hidden_dim)
        self.gate_n = nn.Linear(gate_in, hidden_dim)
        self.norm_out = _make_rmsnorm(hidden_dim)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        x_proj = self.input_proj(x)
        xh = torch.cat([x_proj, h], dim=-1)
        r = torch.sigmoid(self.gate_r(xh))
        z = torch.sigmoid(self.gate_z(xh))
        xrh = torch.cat([x_proj, r * h], dim=-1)
        n = torch.tanh(self.gate_n(xrh))
        h_new = (1.0 - z) * h + z * n
        return self.norm_out(h_new)


# ---------------------------------------------------------------------------
# Prior and Posterior Networks (self-contained)
# ---------------------------------------------------------------------------

class PriorNet(nn.Module):
    def __init__(self, deter_dim: int, stoch_dim: int, num_classes: int,
                 hidden_dim: int = 256, num_layers: int = 2) -> None:
        super().__init__()
        self.stoch_dim = stoch_dim
        self.num_classes = num_classes
        self.mlp = _make_mlp(deter_dim, hidden_dim, stoch_dim * num_classes, num_layers)

    def forward(self, h: Tensor) -> Tensor:
        return self.mlp(h).view(h.shape[0], self.stoch_dim, self.num_classes)


class PosteriorNet(nn.Module):
    def __init__(self, deter_dim: int, embed_dim: int, stoch_dim: int,
                 num_classes: int, hidden_dim: int = 256, num_layers: int = 2) -> None:
        super().__init__()
        self.stoch_dim = stoch_dim
        self.num_classes = num_classes
        self.mlp = _make_mlp(deter_dim + embed_dim, hidden_dim,
                              stoch_dim * num_classes, num_layers)

    def forward(self, h: Tensor, embed: Tensor) -> Tensor:
        inp = torch.cat([h, embed], dim=-1)
        return self.mlp(inp).view(h.shape[0], self.stoch_dim, self.num_classes)


# ---------------------------------------------------------------------------
# SymlogTwohot (self-contained)
# ---------------------------------------------------------------------------

class SymlogTwohot(nn.Module):
    def __init__(self, num_bins: int = 255, low: float = -20.0, high: float = 20.0) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.low = low
        self.high = high
        self.register_buffer("bin_centers", torch.linspace(low, high, num_bins))

    def encode(self, x: Tensor) -> Tensor:
        import torch.nn.functional as F_
        x_log = _symlog(x).clamp(self.low, self.high)
        delta = (self.high - self.low) / (self.num_bins - 1)
        pos = (x_log - self.low) / delta
        k = pos.long().clamp(0, self.num_bins - 2)
        b_k = self.bin_centers[k]
        b_k1 = self.bin_centers[k + 1]
        w_upper = ((x_log - b_k) / (b_k1 - b_k + 1e-8)).clamp(0.0, 1.0)
        w_lower = 1.0 - w_upper
        target = torch.zeros(*x.shape, self.num_bins, device=x.device, dtype=x.dtype)
        target.scatter_(-1, k.unsqueeze(-1), w_lower.unsqueeze(-1))
        target.scatter_(-1, (k + 1).unsqueeze(-1), w_upper.unsqueeze(-1))
        return target

    def decode(self, logits: Tensor) -> Tensor:
        probs = torch.softmax(logits, dim=-1)
        return _symexp((probs * self.bin_centers).sum(dim=-1))

    def loss(self, logits: Tensor, target: Tensor) -> Tensor:
        import torch.nn.functional as F_
        twohot = self.encode(target)
        return -(twohot * torch.log_softmax(logits, dim=-1)).sum(dim=-1)


# ---------------------------------------------------------------------------
# RSSM
# ---------------------------------------------------------------------------

class RSSM(nn.Module):
    """
    DreamerV3-Style Recurrent State Space Model.

    Maintains a latent world state (h_t, z_t) where:
      h_t: deterministic state from BlockGRU
      z_t: stochastic categorical state (stoch_dim x num_classes)

    The observe() method encodes observations into posterior states.
    The imagine() method rolls out the prior for actor-critic training.

    Args:
        cfg:       RSSMConfig with architecture hyperparameters.
        embed_dim: Dimensionality of the observation encoder output.
        action_dim: Dimensionality of the action space.
        twohot_bins: Number of bins for symlog twohot prediction heads.
    """

    def __init__(
        self,
        cfg: RSSMConfig,
        embed_dim: int,
        action_dim: int,
        twohot_bins: int = 255,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_dim = embed_dim
        self.action_dim = action_dim

        # GRU input = flattened previous stochastic state + previous action
        gru_input_dim = cfg.stoch_flat_dim + action_dim

        self.gru = BlockGRU(input_dim=gru_input_dim, hidden_dim=cfg.deter_dim)

        self.prior_net = PriorNet(
            deter_dim=cfg.deter_dim,
            stoch_dim=cfg.stoch_dim,
            num_classes=cfg.num_classes,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
        )

        self.posterior_net = PosteriorNet(
            deter_dim=cfg.deter_dim,
            embed_dim=embed_dim,
            stoch_dim=cfg.stoch_dim,
            num_classes=cfg.num_classes,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
        )

        # Prediction heads on the feature vector
        feature_dim = cfg.feature_dim
        self.reward_head = nn.Linear(feature_dim, twohot_bins)
        self.continue_head = nn.Linear(feature_dim, 1)
        self.twohot = SymlogTwohot(num_bins=twohot_bins)

    # -----------------------------------------------------------------------
    # State utilities
    # -----------------------------------------------------------------------

    def initial_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> RSSMState:
        """
        Return all-zeros initial RSSM state.

        Args:
            batch_size: Number of parallel sequences.
            device:     Target device. If None, uses CPU.

        Returns:
            RSSMState with zeros for deter, stoch, and logits.
        """
        if device is None:
            device = next(self.parameters()).device
        return RSSMState(
            deter=torch.zeros(batch_size, self.cfg.deter_dim, device=device),
            stoch=torch.zeros(batch_size, self.cfg.stoch_dim, self.cfg.num_classes, device=device),
            logits=torch.zeros(batch_size, self.cfg.stoch_dim, self.cfg.num_classes, device=device),
        )

    def get_features(self, state: RSSMState) -> Tensor:
        """
        Concatenate deterministic and flattened stochastic state.

        Returns:
            Tensor of shape (batch, deter_dim + stoch_dim * num_classes).
        """
        return state.features

    # -----------------------------------------------------------------------
    # Single observe step
    # -----------------------------------------------------------------------

    def observe_step(
        self,
        embed: Tensor,
        action: Tensor,
        state: RSSMState,
    ) -> Tuple[RSSMState, Tensor]:
        """
        Process one timestep of observed data.

        Computes:
          1. h_t = BlockGRU(concat(z_{t-1}_flat, a_{t-1}), h_{t-1})
          2. prior_logits = prior_net(h_t)
          3. posterior_logits = posterior_net(concat(h_t, embed_t))
          4. z_t ~ categorical(unimix(posterior_logits))  [straight-through]

        Args:
            embed:  Observation embedding, shape (batch, embed_dim).
            action: Previous action, shape (batch, action_dim).
            state:  Previous state (h_{t-1}, z_{t-1}).

        Returns:
            Tuple of:
              - posterior_state: RSSMState(h_t, z_t, posterior_logits)
              - prior_logits:    Tensor (batch, stoch_dim, num_classes)
        """
        # 1. Form GRU input: concat previous z (flattened) and previous action
        z_flat = state.stoch.flatten(start_dim=-2)  # (batch, stoch_flat_dim)
        gru_in = torch.cat([z_flat, action], dim=-1)  # (batch, stoch_flat_dim + action_dim)

        # 2. Advance deterministic state
        h_new = self.gru(gru_in, state.deter)  # (batch, deter_dim)

        # 3. Compute prior distribution (for KL computation)
        prior_logits = self.prior_net(h_new)  # (batch, stoch_dim, num_classes)

        # 4. Compute posterior distribution (conditions on observation)
        posterior_logits = self.posterior_net(h_new, embed)  # (batch, stoch_dim, num_classes)

        # 5. Sample posterior state with straight-through
        z_new = _sample_straight_through(posterior_logits, unimix=self.cfg.unimix)

        new_state = RSSMState(deter=h_new, stoch=z_new, logits=posterior_logits)
        return new_state, prior_logits

    # -----------------------------------------------------------------------
    # Sequence observe
    # -----------------------------------------------------------------------

    def observe(
        self,
        embed_seq: Tensor,
        action_seq: Tensor,
        state: RSSMState,
    ) -> Tuple[List[RSSMState], List[Tensor]]:
        """
        Encode a full sequence of observations.

        Args:
            embed_seq:  Observation embeddings, shape (T, batch, embed_dim).
            action_seq: Actions at each step, shape (T, batch, action_dim).
                        action_seq[t] is the action taken at timestep t
                        (used to transition from state_{t-1} to state_t).
            state:      Initial state.

        Returns:
            Tuple of:
              - posteriors: List[RSSMState] of length T (posterior states).
              - priors:     List[Tensor] of length T (prior logits for KL).
        """
        T = embed_seq.shape[0]
        posteriors: List[RSSMState] = []
        priors: List[Tensor] = []

        for t in range(T):
            state, prior_logits = self.observe_step(
                embed=embed_seq[t],
                action=action_seq[t],
                state=state,
            )
            posteriors.append(state)
            priors.append(prior_logits)

        return posteriors, priors

    # -----------------------------------------------------------------------
    # Single imagine step
    # -----------------------------------------------------------------------

    def imagine_step(
        self,
        state: RSSMState,
        action: Tensor,
    ) -> RSSMState:
        """
        Advance the world model by one step using only the prior.

        No observation is used — this is the imagination (rollout) step.

        Args:
            state:  Current state (h_t, z_t).
            action: Action taken, shape (batch, action_dim).

        Returns:
            Next state (h_{t+1}, z_{t+1}) sampled from prior.
        """
        z_flat = state.stoch.flatten(start_dim=-2)
        gru_in = torch.cat([z_flat, action], dim=-1)
        h_new = self.gru(gru_in, state.deter)
        prior_logits = self.prior_net(h_new)
        z_new = _sample_straight_through(prior_logits, unimix=self.cfg.unimix)
        return RSSMState(deter=h_new, stoch=z_new, logits=prior_logits)

    # -----------------------------------------------------------------------
    # Imagination rollout
    # -----------------------------------------------------------------------

    def imagine(
        self,
        policy: Callable[[Tensor], Tensor],
        state: RSSMState,
        horizon: int,
    ) -> ImaginedTrajectory:
        """
        Unroll the world model for `horizon` steps using only the prior.

        At each step:
          1. Extract feature vector from current state.
          2. Query policy for action.
          3. Advance to next state via prior.
          4. Collect reward and continue logits.

        Gradients flow through all operations (no torch.no_grad).

        Args:
            policy:  Callable mapping (batch, feature_dim) -> (batch, action_dim).
            state:   Starting state (typically from observe step).
            horizon: Number of imagination steps.

        Returns:
            ImaginedTrajectory with all tensors of shape (horizon, batch, *).
        """
        features_list: List[Tensor] = []
        actions_list: List[Tensor] = []
        reward_logits_list: List[Tensor] = []
        cont_logits_list: List[Tensor] = []

        for _ in range(horizon):
            # Current feature vector
            feat = self.get_features(state)  # (batch, feature_dim)

            # Query policy
            action = policy(feat)  # (batch, action_dim)

            # Advance state via prior
            state = self.imagine_step(state, action)

            # Prediction heads on the new state's features
            next_feat = self.get_features(state)
            reward_logits = self.reward_head(next_feat)   # (batch, num_bins)
            cont_logits = self.continue_head(next_feat)   # (batch, 1)

            features_list.append(feat)
            actions_list.append(action)
            reward_logits_list.append(reward_logits)
            cont_logits_list.append(cont_logits)

        return ImaginedTrajectory(
            features=torch.stack(features_list, dim=0),         # (H, B, feature_dim)
            actions=torch.stack(actions_list, dim=0),           # (H, B, action_dim)
            reward_logits=torch.stack(reward_logits_list, dim=0),  # (H, B, num_bins)
            continue_logits=torch.stack(cont_logits_list, dim=0),  # (H, B, 1)
        )

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def count_parameters(self) -> dict[str, int]:
        """Return parameter counts per submodule."""
        def count(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters())
        return {
            "gru": count(self.gru),
            "prior_net": count(self.prior_net),
            "posterior_net": count(self.posterior_net),
            "reward_head": count(self.reward_head),
            "continue_head": count(self.continue_head),
            "total": count(self),
        }


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("RSSM self-tests")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}\n")

    FAILURES: list[str] = []

    def check(condition: bool, name: str, detail: str = "") -> None:
        status = "PASS" if condition else "FAIL"
        msg = f"  [{status}] {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)
        if not condition:
            FAILURES.append(name)

    # Use a small config for fast tests
    cfg = RSSMConfig(
        deter_dim=64,
        stoch_dim=8,
        num_classes=8,
        hidden_dim=64,
        num_layers=2,
        unimix=0.01,
    )
    EMBED_DIM = 32
    ACTION_DIM = 4
    BATCH = 4
    T = 10
    H = 15

    rssm = RSSM(cfg, embed_dim=EMBED_DIM, action_dim=ACTION_DIM).to(device)

    # -----------------------------------------------------------------------
    # Test 1: Initial state
    # -----------------------------------------------------------------------
    print("Test 1: Initial state")
    state0 = rssm.initial_state(BATCH, device=device)
    check(state0.deter.shape == (BATCH, 64), "initial deter shape", str(state0.deter.shape))
    check(state0.stoch.shape == (BATCH, 8, 8), "initial stoch shape", str(state0.stoch.shape))
    check(torch.all(state0.deter == 0).item(), "initial deter is zeros")
    check(torch.all(state0.stoch == 0).item(), "initial stoch is zeros")

    # -----------------------------------------------------------------------
    # Test 2: observe_step shapes
    # -----------------------------------------------------------------------
    print("\nTest 2: observe_step shapes")
    embed = torch.randn(BATCH, EMBED_DIM, device=device)
    action = torch.randn(BATCH, ACTION_DIM, device=device)
    new_state, prior_logits = rssm.observe_step(embed, action, state0)
    check(new_state.deter.shape == (BATCH, 64), "observe_step deter shape")
    check(new_state.stoch.shape == (BATCH, 8, 8), "observe_step stoch shape")
    check(new_state.logits.shape == (BATCH, 8, 8), "observe_step posterior logits shape")
    check(prior_logits.shape == (BATCH, 8, 8), "observe_step prior logits shape")

    # -----------------------------------------------------------------------
    # Test 3: observe sequence shapes
    # -----------------------------------------------------------------------
    print("\nTest 3: observe sequence")
    embed_seq = torch.randn(T, BATCH, EMBED_DIM, device=device)
    action_seq = torch.randn(T, BATCH, ACTION_DIM, device=device)
    state_init = rssm.initial_state(BATCH, device=device)
    posteriors, priors = rssm.observe(embed_seq, action_seq, state_init)
    check(len(posteriors) == T, f"observe: {T} posteriors")
    check(len(priors) == T, f"observe: {T} priors")
    check(posteriors[0].deter.shape == (BATCH, 64), "observe posterior[0] deter shape")
    check(priors[0].shape == (BATCH, 8, 8), "observe prior[0] shape")

    # -----------------------------------------------------------------------
    # Test 4: imagine shapes
    # -----------------------------------------------------------------------
    print("\nTest 4: imagine shapes")

    def random_policy(feat: Tensor) -> Tensor:
        return torch.randn(feat.shape[0], ACTION_DIM, device=feat.device)

    state_for_imagine = rssm.initial_state(BATCH, device=device)
    traj = rssm.imagine(random_policy, state_for_imagine, horizon=H)

    expected_feature_dim = cfg.deter_dim + cfg.stoch_dim * cfg.num_classes
    check(traj.features.shape == (H, BATCH, expected_feature_dim),
          "imagine features shape", str(traj.features.shape))
    check(traj.actions.shape == (H, BATCH, ACTION_DIM),
          "imagine actions shape", str(traj.actions.shape))
    check(traj.reward_logits.shape[0] == H, "imagine reward_logits horizon")
    check(traj.continue_logits.shape == (H, BATCH, 1),
          "imagine continue_logits shape", str(traj.continue_logits.shape))

    # -----------------------------------------------------------------------
    # Test 5: feature vector concatenation
    # -----------------------------------------------------------------------
    print("\nTest 5: get_features concatenation")
    state5 = rssm.initial_state(BATCH, device=device)
    feat5 = rssm.get_features(state5)
    check(feat5.shape == (BATCH, expected_feature_dim),
          "get_features shape", str(feat5.shape))
    # Verify first deter_dim elements match state.deter
    check(torch.allclose(feat5[:, :cfg.deter_dim], state5.deter), "features[:, :deter] == deter")

    # -----------------------------------------------------------------------
    # Test 6: gradient flow through observe
    # Observe_step: stoch comes from posterior (gradient flows to posterior + GRU).
    # Prior gets gradients via the KL loss (not tested here).
    # We verify the components that DO receive gradient via stoch.
    # -----------------------------------------------------------------------
    print("\nTest 6: gradient flow through observe")
    rssm_grad = RSSM(cfg, embed_dim=EMBED_DIM, action_dim=ACTION_DIM).to(device)
    embed_grad = torch.randn(BATCH, EMBED_DIM, device=device, requires_grad=True)
    action_grad = torch.randn(BATCH, ACTION_DIM, device=device)
    state_grad0 = rssm_grad.initial_state(BATCH, device=device)
    new_state_grad, _ = rssm_grad.observe_step(embed_grad, action_grad, state_grad0)
    loss_grad = new_state_grad.stoch.sum()
    loss_grad.backward()
    check(embed_grad.grad is not None, "gradient flows to embed")
    check(torch.all(torch.isfinite(embed_grad.grad)).item(), "embed gradient finite")
    # Verify GRU and posterior receive gradients (they are in the stoch computation path)
    gru_params_with_grad = [p for n, p in rssm_grad.named_parameters() if "gru" in n and p.grad is not None]
    post_params_with_grad = [p for n, p in rssm_grad.named_parameters() if "posterior" in n and p.grad is not None]
    check(len(gru_params_with_grad) > 0, "GRU parameters receive gradient from observe")
    check(len(post_params_with_grad) > 0, "posterior_net parameters receive gradient from observe")

    # -----------------------------------------------------------------------
    # Test 7: gradient flow through imagine
    # Imagine uses prior and prediction heads — all should receive gradients.
    # The policy must propagate gradients (use feat directly, not a fresh randn).
    # -----------------------------------------------------------------------
    print("\nTest 7: gradient flow through imagine")
    rssm_img = RSSM(cfg, embed_dim=EMBED_DIM, action_dim=ACTION_DIM).to(device)
    state_img = rssm_img.initial_state(BATCH, device=device)

    def differentiable_policy(feat: Tensor) -> Tensor:
        # Use a slice of the feature vector so gradients flow through it
        return torch.tanh(feat[:, :ACTION_DIM])

    traj_img = rssm_img.imagine(differentiable_policy, state_img, horizon=5)
    reward_loss = traj_img.reward_logits.mean()
    reward_loss.backward()
    # Prior net, reward head, continue head, and GRU should all receive gradients
    prior_ok = all(p.grad is not None for n, p in rssm_img.named_parameters() if "prior" in n)
    reward_ok = all(p.grad is not None for n, p in rssm_img.named_parameters() if "reward" in n)
    gru_ok = all(p.grad is not None for n, p in rssm_img.named_parameters() if "gru" in n)
    check(prior_ok, "prior_net parameters receive gradient from imagine")
    check(reward_ok, "reward_head parameters receive gradient from imagine")
    check(gru_ok, "GRU parameters receive gradient from imagine")

    # -----------------------------------------------------------------------
    # Test 8: stochastic state uses straight-through (z is approximately one-hot)
    # -----------------------------------------------------------------------
    print("\nTest 8: straight-through one-hot in observe")
    embed8 = torch.randn(BATCH, EMBED_DIM, device=device)
    action8 = torch.randn(BATCH, ACTION_DIM, device=device)
    state8_init = rssm.initial_state(BATCH, device=device)
    state8, _ = rssm.observe_step(embed8, action8, state8_init)
    # In forward pass, stoch should be approximately one-hot per row
    stoch_sums = state8.stoch.sum(dim=-1)  # (batch, stoch_dim), should be ≈ 1
    check(
        torch.allclose(stoch_sums, torch.ones_like(stoch_sums), atol=1e-5),
        "stoch sums to 1 per distribution (one-hot via straight-through)",
        f"max_dev={(stoch_sums - 1.0).abs().max().item():.2e}",
    )

    # -----------------------------------------------------------------------
    # Test 9: batch size 1
    # -----------------------------------------------------------------------
    print("\nTest 9: batch size 1")
    state_b1 = rssm.initial_state(1, device=device)
    embed_b1 = torch.randn(1, EMBED_DIM, device=device)
    action_b1 = torch.randn(1, ACTION_DIM, device=device)
    new_state_b1, _ = rssm.observe_step(embed_b1, action_b1, state_b1)
    check(new_state_b1.deter.shape == (1, 64), "batch_size=1 observe shape")
    traj_b1 = rssm.imagine(lambda f: torch.zeros(1, ACTION_DIM, device=f.device), state_b1, horizon=3)
    check(traj_b1.features.shape[1] == 1, "batch_size=1 imagine shape")

    # -----------------------------------------------------------------------
    # Test 10: horizon=1 imagination
    # -----------------------------------------------------------------------
    print("\nTest 10: horizon=1 imagination")
    traj_h1 = rssm.imagine(random_policy, rssm.initial_state(BATCH, device=device), horizon=1)
    check(traj_h1.horizon == 1, "horizon=1: trajectory horizon")
    check(traj_h1.features.shape[0] == 1, "horizon=1: features.shape[0]")

    # -----------------------------------------------------------------------
    # Test 11: parameter count
    # -----------------------------------------------------------------------
    print("\nTest 11: parameter count")
    counts = rssm.count_parameters()
    check("total" in counts, "count_parameters returns total")
    check(counts["total"] > 0, "total params > 0", f"total={counts['total']}")
    check(counts["gru"] > 0, "gru params > 0")
    check(counts["prior_net"] > 0, "prior_net params > 0")
    check(counts["posterior_net"] > 0, "posterior_net params > 0")

    # -----------------------------------------------------------------------
    # Test 12: all-zero embed input (numerical stability)
    # -----------------------------------------------------------------------
    print("\nTest 12: all-zero embed input")
    embed_zero = torch.zeros(BATCH, EMBED_DIM, device=device)
    action_zero = torch.zeros(BATCH, ACTION_DIM, device=device)
    state_zero_init = rssm.initial_state(BATCH, device=device)
    state_zero_out, _ = rssm.observe_step(embed_zero, action_zero, state_zero_init)
    check(torch.all(torch.isfinite(state_zero_out.deter)).item(), "zero input: deter finite")
    check(torch.all(torch.isfinite(state_zero_out.stoch)).item(), "zero input: stoch finite")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} test(s): {', '.join(FAILURES)}")
        sys.exit(1)
    else:
        print("All tests PASSED.")
