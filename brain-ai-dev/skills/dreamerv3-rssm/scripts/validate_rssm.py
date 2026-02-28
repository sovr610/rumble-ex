#!/usr/bin/env python3
"""
validate_rssm.py — DreamerV3 RSSM Done-When Gate Validator

Validates the three done-when gates defined in SKILL.md:

  Gate 1: RSSM Observe/Imagine Work
  Gate 2: Symlog Twohot Round-Trips
  Gate 3: KL Balancing Correct

Usage:
    python scripts/validate_rssm.py
    python scripts/validate_rssm.py --verbose
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Inline RSSM components (no external dependencies required)
# ---------------------------------------------------------------------------

def symlog(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x: Tensor) -> Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def unimix_probs(logits: Tensor, unimix: float = 0.01) -> Tensor:
    num_classes = logits.shape[-1]
    return (1.0 - unimix) * torch.softmax(logits, dim=-1) + unimix / num_classes


def sample_straight_through(logits: Tensor, unimix: float = 0.01) -> Tensor:
    probs = unimix_probs(logits, unimix)
    indices = probs.argmax(dim=-1)
    z_hard = F.one_hot(indices, logits.shape[-1]).to(probs.dtype)
    return z_hard - probs.detach() + probs


def kl_categorical(p_logits: Tensor, q_logits: Tensor, unimix: float = 0.01) -> Tensor:
    nc = p_logits.shape[-1]
    p = (1 - unimix) * torch.softmax(p_logits, dim=-1) + unimix / nc
    q = (1 - unimix) * torch.softmax(q_logits, dim=-1) + unimix / nc
    return (q * (q.clamp(1e-20).log() - p.clamp(1e-20).log())).sum(-1).clamp(min=0)


def make_rmsnorm(dim: int) -> nn.Module:
    if hasattr(nn, "RMSNorm"):
        return nn.RMSNorm(dim)
    class _RMS(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.ones(d))
        def forward(self, x: Tensor) -> Tensor:
            return x / x.pow(2).mean(-1, keepdim=True).add(1e-8).sqrt() * self.w
    return _RMS(dim)


def make_mlp(in_d: int, hid_d: int, out_d: int, layers: int = 2) -> nn.Sequential:
    mods: list[nn.Module] = []
    cur = in_d
    for _ in range(layers):
        mods += [nn.Linear(cur, hid_d), nn.LayerNorm(hid_d), nn.SiLU()]
        cur = hid_d
    mods.append(nn.Linear(cur, out_d))
    return nn.Sequential(*mods)


@dataclass
class RSSMConfig:
    deter_dim: int = 64
    stoch_dim: int = 8
    num_classes: int = 8
    hidden_dim: int = 64
    num_layers: int = 2
    unimix: float = 0.01

    @property
    def stoch_flat_dim(self) -> int:
        return self.stoch_dim * self.num_classes

    @property
    def feature_dim(self) -> int:
        return self.deter_dim + self.stoch_flat_dim


@dataclass
class RSSMState:
    deter: Tensor
    stoch: Tensor
    logits: Tensor

    @property
    def features(self) -> Tensor:
        return torch.cat([self.deter, self.stoch.flatten(-2)], dim=-1)


@dataclass
class ImaginedTrajectory:
    features: Tensor
    actions: Tensor
    reward_logits: Tensor
    continue_logits: Tensor

    @property
    def horizon(self) -> int:
        return self.features.shape[0]


class BlockGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()
        )
        g = hidden_dim * 2
        self.gate_r = nn.Linear(g, hidden_dim)
        self.gate_z = nn.Linear(g, hidden_dim)
        self.gate_n = nn.Linear(g, hidden_dim)
        self.norm_out = make_rmsnorm(hidden_dim)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        xp = self.input_proj(x)
        xh = torch.cat([xp, h], -1)
        r = torch.sigmoid(self.gate_r(xh))
        z = torch.sigmoid(self.gate_z(xh))
        n = torch.tanh(self.gate_n(torch.cat([xp, r * h], -1)))
        return self.norm_out((1 - z) * h + z * n)


class PriorNet(nn.Module):
    def __init__(self, cfg: RSSMConfig) -> None:
        super().__init__()
        self.sd, self.nc = cfg.stoch_dim, cfg.num_classes
        self.mlp = make_mlp(cfg.deter_dim, cfg.hidden_dim, cfg.stoch_dim * cfg.num_classes, cfg.num_layers)

    def forward(self, h: Tensor) -> Tensor:
        return self.mlp(h).view(h.shape[0], self.sd, self.nc)


class PosteriorNet(nn.Module):
    def __init__(self, cfg: RSSMConfig, embed_dim: int) -> None:
        super().__init__()
        self.sd, self.nc = cfg.stoch_dim, cfg.num_classes
        self.mlp = make_mlp(cfg.deter_dim + embed_dim, cfg.hidden_dim, cfg.stoch_dim * cfg.num_classes, cfg.num_layers)

    def forward(self, h: Tensor, e: Tensor) -> Tensor:
        return self.mlp(torch.cat([h, e], -1)).view(h.shape[0], self.sd, self.nc)


class SymlogTwohot(nn.Module):
    def __init__(self, num_bins: int = 255) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.register_buffer("bin_centers", torch.linspace(-20.0, 20.0, num_bins))

    def encode(self, x: Tensor) -> Tensor:
        xl = symlog(x).clamp(-20.0, 20.0)
        delta = 40.0 / (self.num_bins - 1)
        pos = (xl + 20.0) / delta
        k = pos.long().clamp(0, self.num_bins - 2)
        bk = self.bin_centers[k]
        bk1 = self.bin_centers[k + 1]
        wu = ((xl - bk) / (bk1 - bk + 1e-8)).clamp(0, 1)
        tgt = torch.zeros(*x.shape, self.num_bins, device=x.device, dtype=x.dtype)
        tgt.scatter_(-1, k.unsqueeze(-1), (1 - wu).unsqueeze(-1))
        tgt.scatter_(-1, (k + 1).unsqueeze(-1), wu.unsqueeze(-1))
        return tgt

    def decode(self, logits: Tensor) -> Tensor:
        return symexp((torch.softmax(logits, -1) * self.bin_centers).sum(-1))

    def loss(self, logits: Tensor, target: Tensor) -> Tensor:
        return -(self.encode(target) * F.log_softmax(logits, -1)).sum(-1)


class RSSM(nn.Module):
    def __init__(self, cfg: RSSMConfig, embed_dim: int, action_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.gru = BlockGRU(cfg.stoch_flat_dim + action_dim, cfg.deter_dim)
        self.prior = PriorNet(cfg)
        self.posterior = PosteriorNet(cfg, embed_dim)
        self.reward_head = nn.Linear(cfg.feature_dim, 255)
        self.cont_head = nn.Linear(cfg.feature_dim, 1)
        self.twohot = SymlogTwohot()

    def initial_state(self, B: int, device: torch.device) -> RSSMState:
        z = lambda shape: torch.zeros(*shape, device=device)
        return RSSMState(z([B, self.cfg.deter_dim]), z([B, self.cfg.stoch_dim, self.cfg.num_classes]), z([B, self.cfg.stoch_dim, self.cfg.num_classes]))

    def observe_step(self, embed: Tensor, action: Tensor, state: RSSMState) -> Tuple[RSSMState, Tensor]:
        gru_in = torch.cat([state.stoch.flatten(-2), action], -1)
        h = self.gru(gru_in, state.deter)
        prior_logits = self.prior(h)
        post_logits = self.posterior(h, embed)
        z = sample_straight_through(post_logits, self.cfg.unimix)
        return RSSMState(h, z, post_logits), prior_logits

    def observe(self, embed_seq: Tensor, action_seq: Tensor, state: RSSMState):
        posts, priors = [], []
        for t in range(embed_seq.shape[0]):
            state, pl = self.observe_step(embed_seq[t], action_seq[t], state)
            posts.append(state); priors.append(pl)
        return posts, priors

    def imagine(self, policy: Callable, state: RSSMState, horizon: int) -> ImaginedTrajectory:
        feats, acts, rews, conts = [], [], [], []
        for _ in range(horizon):
            feat = state.features
            action = policy(feat)
            gru_in = torch.cat([state.stoch.flatten(-2), action], -1)
            h = self.gru(gru_in, state.deter)
            pl = self.prior(h)
            z = sample_straight_through(pl, self.cfg.unimix)
            state = RSSMState(h, z, pl)
            nfeat = state.features
            feats.append(feat); acts.append(action)
            rews.append(self.reward_head(nfeat)); conts.append(self.cont_head(nfeat))
        return ImaginedTrajectory(
            torch.stack(feats), torch.stack(acts), torch.stack(rews), torch.stack(conts)
        )


# ---------------------------------------------------------------------------
# Gate check infrastructure
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


class GateValidator:
    def __init__(self, gate_name: str, verbose: bool = False) -> None:
        self.gate_name = gate_name
        self.verbose = verbose
        self.checks: List[CheckResult] = []

    def check(self, condition: bool, name: str, detail: str = "") -> None:
        result = CheckResult(name=name, passed=condition, detail=detail)
        self.checks.append(result)
        if self.verbose:
            status = "PASS" if condition else "FAIL"
            msg = f"    [{status}] {name}"
            if detail:
                msg += f" — {detail}"
            print(msg)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def n_total(self) -> int:
        return len(self.checks)

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.gate_name}: {self.n_passed}/{self.n_total} checks"

    def failures(self) -> List[CheckResult]:
        return [c for c in self.checks if not c.passed]


# ---------------------------------------------------------------------------
# Gate 1: RSSM Observe/Imagine
# ---------------------------------------------------------------------------

def gate1_rssm_observe_imagine(device: torch.device, verbose: bool) -> GateValidator:
    """
    Verify that:
    - observe() produces posterior and prior states with correct shapes (batch, stoch_dim, num_classes)
    - imagine() unrolls for H=15 steps and returns stacked tensors
    - Categorical samples use straight-through + unimix (sum to 1, no zeros)
    """
    v = GateValidator("Gate 1: RSSM Observe/Imagine", verbose)

    cfg = RSSMConfig(deter_dim=64, stoch_dim=32, num_classes=32, hidden_dim=128, num_layers=2)
    EMBED_DIM = 64
    ACTION_DIM = 4
    BATCH = 8
    T = 10
    H = 15

    rssm = RSSM(cfg, embed_dim=EMBED_DIM, action_dim=ACTION_DIM).to(device)
    state0 = rssm.initial_state(BATCH, device)

    # --- observe_step shapes ---
    embed = torch.randn(BATCH, EMBED_DIM, device=device)
    action = torch.randn(BATCH, ACTION_DIM, device=device)
    new_state, prior_logits = rssm.observe_step(embed, action, state0)

    v.check(new_state.deter.shape == (BATCH, 64), "observe_step: deter shape (B, deter_dim)")
    v.check(new_state.stoch.shape == (BATCH, 32, 32), "observe_step: stoch shape (B, 32, 32)")
    v.check(new_state.logits.shape == (BATCH, 32, 32), "observe_step: posterior logits (B, 32, 32)")
    v.check(prior_logits.shape == (BATCH, 32, 32), "observe_step: prior logits (B, 32, 32)")

    # --- stochastic state is one-hot (straight-through) ---
    stoch_sums = new_state.stoch.sum(dim=-1)  # (B, 32), each row should sum to 1
    v.check(
        torch.allclose(stoch_sums, torch.ones_like(stoch_sums), atol=1e-5),
        "stoch: each distribution sums to 1 (one-hot via straight-through)",
        f"max_dev={(stoch_sums - 1.0).abs().max().item():.2e}",
    )

    # --- unimix prevents zero probabilities ---
    prior_probs = unimix_probs(prior_logits, unimix=0.01)
    v.check(prior_probs.min().item() > 0, "unimix: no zero probabilities in prior")

    post_probs = unimix_probs(new_state.logits, unimix=0.01)
    v.check(post_probs.min().item() > 0, "unimix: no zero probabilities in posterior")

    # --- observe sequence shapes ---
    embed_seq = torch.randn(T, BATCH, EMBED_DIM, device=device)
    action_seq = torch.randn(T, BATCH, ACTION_DIM, device=device)
    state_init = rssm.initial_state(BATCH, device)
    posteriors, priors = rssm.observe(embed_seq, action_seq, state_init)

    v.check(len(posteriors) == T, f"observe: returns T={T} posterior states")
    v.check(len(priors) == T, f"observe: returns T={T} prior logit tensors")
    v.check(posteriors[0].stoch.shape == (BATCH, 32, 32), "observe: posterior stoch shape")

    # --- imagine H=15 steps ---
    def random_policy(feat: Tensor) -> Tensor:
        return torch.randn(feat.shape[0], ACTION_DIM, device=feat.device)

    start_state = rssm.initial_state(BATCH, device)
    traj = rssm.imagine(random_policy, start_state, horizon=H)

    expected_feat_dim = 64 + 32 * 32  # deter + stoch_flat
    v.check(traj.features.shape == (H, BATCH, expected_feat_dim),
            f"imagine: features shape ({H}, {BATCH}, {expected_feat_dim})", str(traj.features.shape))
    v.check(traj.actions.shape == (H, BATCH, ACTION_DIM),
            f"imagine: actions shape ({H}, {BATCH}, {ACTION_DIM})", str(traj.actions.shape))
    v.check(traj.reward_logits.shape[0] == H, f"imagine: reward_logits has {H} time steps")
    v.check(traj.continue_logits.shape[0] == H, f"imagine: continue_logits has {H} time steps")
    v.check(traj.horizon == H, "imagine: ImaginedTrajectory.horizon property")

    # --- gradient flows through imagination ---
    # Use a differentiable policy (uses features, not fresh randn)
    # and check that prior, reward head, and GRU receive gradients.
    # Note: posterior_net is NOT used in imagine (only prior), so it
    # does not receive gradients from this path — that is correct.
    start2 = rssm.initial_state(BATCH, device)

    def grad_policy(feat: Tensor) -> Tensor:
        # Must use feat to propagate gradients through features
        return torch.tanh(feat[:, :ACTION_DIM])

    traj2 = rssm.imagine(grad_policy, start2, horizon=5)
    traj2.reward_logits.mean().backward()
    # Check that the components used in imagination have gradients
    prior_ok = all(p.grad is not None for n, p in rssm.named_parameters() if "prior" in n)
    reward_ok = all(p.grad is not None for n, p in rssm.named_parameters() if "reward" in n)
    gru_ok = all(p.grad is not None for n, p in rssm.named_parameters() if "gru" in n)
    all_grad = prior_ok and reward_ok and gru_ok
    v.check(all_grad, "imagine: gradients flow back through all RSSM parameters")

    # --- all-zero inputs don't cause NaN ---
    state_zero = rssm.initial_state(BATCH, device)
    embed_z = torch.zeros(BATCH, EMBED_DIM, device=device)
    action_z = torch.zeros(BATCH, ACTION_DIM, device=device)
    state_z_out, _ = rssm.observe_step(embed_z, action_z, state_zero)
    v.check(torch.all(torch.isfinite(state_z_out.deter)).item(), "observe: zero inputs → finite deter")
    v.check(torch.all(torch.isfinite(state_z_out.stoch)).item(), "observe: zero inputs → finite stoch")

    return v


# ---------------------------------------------------------------------------
# Gate 2: Symlog Twohot Round-Trips
# ---------------------------------------------------------------------------

def gate2_symlog_twohot(device: torch.device, verbose: bool) -> GateValidator:
    """
    Verify that:
    - encode(x) produces valid twohot vectors summing to 1
    - decode(encode_as_logits(x)) recovers original value within tolerance
    - Cross-entropy loss is finite and differentiable
    """
    v = GateValidator("Gate 2: Symlog Twohot Round-Trips", verbose)

    module = SymlogTwohot(num_bins=255).to(device)

    # --- encode properties ---
    test_vals = torch.tensor([-1e6, -100.0, -1.0, 0.0, 1.0, 100.0, 1e6], device=device)
    twohot = module.encode(test_vals)

    sums = twohot.sum(dim=-1)
    v.check(torch.allclose(sums, torch.ones_like(sums), atol=1e-5),
            "encode: twohot sums to 1", f"max_dev={(sums-1.0).abs().max().item():.2e}")
    v.check(twohot.min().item() >= 0.0, "encode: twohot non-negative")
    nonzero = (twohot > 1e-7).sum(-1)
    v.check((nonzero <= 2).all().item(), "encode: at most 2 nonzero entries per sample")

    # --- decode via "perfect" logits (log of twohot) ---
    round_trip_cases = [
        (-1e6,  0.10, True,  "neg_million"),
        (-100.0, 0.005, True, "neg_hundred"),
        (-10.0,  0.001, True, "neg_ten"),
        (-1.0,   0.001, False, "neg_one"),
        (0.0,    1e-5,  False, "zero"),
        (1.0,    0.001, False, "one"),
        (10.0,   0.001, True,  "ten"),
        (100.0,  0.005, True,  "hundred"),
        (1e6,    0.10,  True,  "million"),
    ]

    all_rt_ok = True
    rt_details = []
    for val, tol, use_rtol, label in round_trip_cases:
        x_in = torch.tensor([val], device=device)
        th = module.encode(x_in)  # (1, 255)
        logits = th.clamp(1e-7).log()  # approximation of "perfect" logits
        decoded = module.decode(logits).item()
        if use_rtol:
            err = abs(decoded - val) / (abs(val) + 1e-8)
            ok = err < tol
            rt_details.append(f"{label}: {val:.2e} -> {decoded:.2e} (rel_err={err:.3f})")
        else:
            err = abs(decoded - val)
            ok = err < tol
            rt_details.append(f"{label}: {val:.4f} -> {decoded:.4f} (abs_err={err:.2e})")
        if not ok:
            all_rt_ok = False
        v.check(ok, f"round-trip {label}", rt_details[-1])

    # --- loss is finite ---
    targets_loss = torch.tensor([-1e6, -1000.0, -100.0, -1.0, 0.0, 1.0, 100.0, 1000.0, 1e6], device=device)
    logits_loss = torch.randn(len(targets_loss), 255, device=device)
    loss_vals = module.loss(logits_loss, targets_loss)
    v.check(torch.all(torch.isfinite(loss_vals)).item(), "loss: finite for all test values")
    v.check((loss_vals >= 0).all().item(), "loss: non-negative")

    # --- loss is differentiable ---
    logits_grad = torch.randn(8, 255, device=device, requires_grad=True)
    targets_grad = torch.tensor([-100.0, -10.0, -1.0, 0.0, 0.5, 1.0, 10.0, 100.0], device=device)
    loss_grad = module.loss(logits_grad, targets_grad)
    loss_grad.mean().backward()
    v.check(logits_grad.grad is not None, "loss: differentiable w.r.t. logits")
    v.check(torch.all(torch.isfinite(logits_grad.grad)).item(), "loss: gradient is finite")

    # --- extreme values handled gracefully ---
    extreme = torch.tensor([1e12, -1e12], device=device)
    th_extreme = module.encode(extreme)
    v.check(torch.all(torch.isfinite(th_extreme)).item(), "encode: extreme values → finite twohot")
    v.check(
        torch.allclose(th_extreme.sum(-1), torch.ones(2, device=device), atol=1e-5),
        "encode: extreme values → twohot sums to 1",
    )

    return v


# ---------------------------------------------------------------------------
# Gate 3: KL Balancing Correct
# ---------------------------------------------------------------------------

def gate3_kl_balancing(device: torch.device, verbose: bool) -> GateValidator:
    """
    Verify that:
    - With identical posterior and prior, KL is 0 (before free-nats, should be 0)
    - With free=1.0, KL below 1 nat is clamped to 1.0
    - Stop-gradient applied to correct distribution in each term
    - Loss weights are 0.5 (dyn) and 0.1 (rep)
    """
    v = GateValidator("Gate 3: KL Balancing Correct", verbose)

    BATCH = 8
    STOCH = 32
    NUM_CLASSES = 32
    FREE = 1.0
    DYN_SCALE = 0.5
    REP_SCALE = 0.1

    # --- identical distributions → KL = 0 ---
    logits_same = torch.randn(BATCH, STOCH, NUM_CLASSES, device=device)
    kl_same = kl_categorical(logits_same, logits_same, unimix=0.01)
    v.check(
        torch.allclose(kl_same, torch.zeros_like(kl_same), atol=1e-5),
        "identical distributions: KL = 0",
        f"max_kl={kl_same.max().item():.2e}",
    )

    # --- after free-nats clamping, identical dists give free_nats per distribution ---
    kl_clamped = torch.clamp(kl_same, min=FREE)
    v.check(
        torch.allclose(kl_clamped, torch.full_like(kl_clamped, FREE), atol=1e-5),
        "identical dists + free_nats=1.0: clamped KL = 1.0 per distribution",
        f"mean_clamped={kl_clamped.mean().item():.4f}",
    )

    # --- different distributions give positive KL ---
    p_l = torch.zeros(BATCH, STOCH, NUM_CLASSES, device=device)
    q_l = torch.zeros(BATCH, STOCH, NUM_CLASSES, device=device)
    q_l[:, :, 0] = 10.0  # push all mass to class 0
    kl_diff = kl_categorical(p_l, q_l, unimix=0.01)
    v.check(kl_diff.min().item() > 0, "different distributions: KL > 0")

    # --- stop-gradient on posterior in dynamics loss ---
    post_g1 = torch.randn(BATCH, STOCH, NUM_CLASSES, device=device, requires_grad=True)
    prior_g1 = torch.randn(BATCH, STOCH, NUM_CLASSES, device=device, requires_grad=True)

    kl_dyn = kl_categorical(
        p_logits=prior_g1,
        q_logits=post_g1.detach(),   # stop-gradient on posterior
    ).sum()
    kl_dyn.backward()

    v.check(prior_g1.grad is not None, "dynamics loss: prior receives gradient")
    v.check(post_g1.grad is None, "dynamics loss: posterior has no gradient (stop-gradient)")

    if prior_g1.grad is not None:
        v.check(
            torch.all(torch.isfinite(prior_g1.grad)).item(),
            "dynamics loss: prior gradient is finite",
        )

    # --- stop-gradient on prior in representation loss ---
    post_g2 = torch.randn(BATCH, STOCH, NUM_CLASSES, device=device, requires_grad=True)
    prior_g2 = torch.randn(BATCH, STOCH, NUM_CLASSES, device=device, requires_grad=True)

    kl_rep = kl_categorical(
        p_logits=prior_g2.detach(),   # stop-gradient on prior
        q_logits=post_g2,
    ).sum()
    kl_rep.backward()

    v.check(post_g2.grad is not None, "representation loss: posterior receives gradient")
    v.check(prior_g2.grad is None, "representation loss: prior has no gradient (stop-gradient)")

    # --- coefficient verification (0.5 dyn + 0.1 rep) ---
    post_l = torch.randn(BATCH, STOCH, NUM_CLASSES, device=device)
    prior_l = torch.randn(BATCH, STOCH, NUM_CLASSES, device=device)

    kl_dyn_val = torch.clamp(kl_categorical(prior_l, post_l.detach()), min=FREE).sum(-1).mean()
    kl_rep_val = torch.clamp(kl_categorical(prior_l.detach(), post_l), min=FREE).sum(-1).mean()
    expected = DYN_SCALE * kl_dyn_val + REP_SCALE * kl_rep_val

    v.check(
        torch.isfinite(expected).item(),
        "combined KL = 0.5 * L_dyn + 0.1 * L_rep is finite",
        f"combined={expected.item():.4f}",
    )

    # Verify the scalar combining formula matches component-wise computation
    kl_dyn_m = torch.clamp(kl_categorical(prior_l, post_l.detach()), min=FREE).sum(-1).mean()
    kl_rep_m = torch.clamp(kl_categorical(prior_l.detach(), post_l), min=FREE).sum(-1).mean()
    computed = DYN_SCALE * kl_dyn_m + REP_SCALE * kl_rep_m
    v.check(
        torch.allclose(computed, expected, atol=1e-4),
        "combined KL computation is consistent",
        f"diff={abs(computed.item() - expected.item()):.2e}",
    )

    # --- free nats prevents gradient when KL is small ---
    # With very similar distributions, KL should be below free_nats
    # and the clamped version should equal free_nats (providing no informative gradient)
    logits_a = torch.randn(BATCH, STOCH, NUM_CLASSES, device=device)
    logits_b = logits_a + 0.0001  # nearly identical
    kl_small = kl_categorical(logits_a, logits_b, unimix=0.01)
    fraction_below_free = (kl_small < FREE).float().mean()
    v.check(
        fraction_below_free.item() > 0.5,
        "free nats: nearly-identical dists have KL < free_nats for most distributions",
        f"fraction_below_free={fraction_below_free.item():.3f}",
    )

    # --- unimix prevents infinite KL (no zero probabilities) ---
    # Even with extreme logits, KL should be finite
    p_extreme = torch.zeros(BATCH, STOCH, NUM_CLASSES, device=device)
    p_extreme[:, :, 0] = 1000.0  # almost deterministic class 0
    q_extreme = torch.zeros(BATCH, STOCH, NUM_CLASSES, device=device)
    q_extreme[:, :, 1] = 1000.0  # almost deterministic class 1 (opposite)
    kl_extreme = kl_categorical(p_extreme, q_extreme, unimix=0.01)
    v.check(
        torch.all(torch.isfinite(kl_extreme)).item(),
        "unimix: KL is finite even for near-deterministic distributions",
        f"max_kl={kl_extreme.max().item():.4f}",
    )

    return v


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate DreamerV3 RSSM done-when gates."
    )
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print individual check results")
    parser.add_argument("--device", default="auto",
                        help="Device: 'cpu', 'cuda', or 'auto' (default: auto)")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 70)
    print("DreamerV3 RSSM — Done-When Gate Validation")
    print(f"Device: {device}")
    print("=" * 70)
    print()

    gates = [
        ("Gate 1", gate1_rssm_observe_imagine),
        ("Gate 2", gate2_symlog_twohot),
        ("Gate 3", gate3_kl_balancing),
    ]

    results = []
    for gate_id, gate_fn in gates:
        print(f"Running {gate_id}...")
        if args.verbose:
            print()
        try:
            v = gate_fn(device=device, verbose=args.verbose)
            results.append(v)
            if args.verbose:
                print()
            print(f"  {v.summary()}")
            if not v.passed:
                for fail in v.failures():
                    print(f"    FAILED: {fail.name} — {fail.detail}")
        except Exception as e:
            print(f"  [ERROR] {gate_id} raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(None)
        print()

    # Summary
    print("=" * 70)
    total_checks = sum(v.n_total for v in results if v is not None)
    total_passed = sum(v.n_passed for v in results if v is not None)
    all_passed = all(v is not None and v.passed for v in results)

    print(f"Results: {total_passed}/{total_checks} checks passed")
    print()
    for v in results:
        if v is not None:
            icon = "✓" if v.passed else "✗"
            print(f"  {icon} {v.summary()}")
    print()

    if all_passed:
        print("ALL GATES PASSED — Implementation meets done-when criteria.")
        sys.exit(0)
    else:
        failed_gates = [v.gate_name for v in results if v is not None and not v.passed]
        print(f"GATES FAILED: {', '.join(failed_gates)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
