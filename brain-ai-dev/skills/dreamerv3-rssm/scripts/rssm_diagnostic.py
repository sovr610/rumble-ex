#!/usr/bin/env python3
"""
rssm_diagnostic.py — DreamerV3 RSSM Diagnostic Tool

Instantiates an RSSM with a specified configuration, runs forward passes,
and prints detailed diagnostic information including:
  - Model configuration and parameter counts
  - State shapes from observe pass
  - Trajectory shapes from imagine pass
  - Symlog twohot encode/decode round-trip errors
  - KL loss components (L_dyn, L_rep, total)
  - Feature vector dimensionality
  - Memory usage estimates

Usage:
    python scripts/rssm_diagnostic.py
    python scripts/rssm_diagnostic.py --model_size 50M
    python scripts/rssm_diagnostic.py --model_size 50M --batch_size 4 --seq_len 32 --horizon 15
    python scripts/rssm_diagnostic.py --batch_size 8 --action_dim 18 --embed_dim 1024
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Inline RSSM implementation (self-contained)
# ---------------------------------------------------------------------------

def symlog(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

def symexp(x: Tensor) -> Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

def unimix_probs(logits: Tensor, unimix: float = 0.01) -> Tensor:
    nc = logits.shape[-1]
    return (1.0 - unimix) * torch.softmax(logits, -1) + unimix / nc

def sample_straight_through(logits: Tensor, unimix: float = 0.01) -> Tensor:
    probs = unimix_probs(logits, unimix)
    idx = probs.argmax(-1)
    z_hard = F.one_hot(idx, logits.shape[-1]).to(probs.dtype)
    return z_hard - probs.detach() + probs

def kl_categorical(p_logits: Tensor, q_logits: Tensor, unimix: float = 0.01) -> Tensor:
    nc = p_logits.shape[-1]
    p = (1 - unimix) * torch.softmax(p_logits, -1) + unimix / nc
    q = (1 - unimix) * torch.softmax(q_logits, -1) + unimix / nc
    return (q * (q.clamp(1e-20).log() - p.clamp(1e-20).log())).sum(-1).clamp(min=0)

def make_rmsnorm(dim: int) -> nn.Module:
    if hasattr(nn, "RMSNorm"):
        return nn.RMSNorm(dim)
    class _R(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.ones(d))
        def forward(self, x: Tensor) -> Tensor:
            return x / x.pow(2).mean(-1, keepdim=True).add(1e-8).sqrt() * self.w
    return _R(dim)

def make_mlp(i: int, h: int, o: int, n: int = 2) -> nn.Sequential:
    mods: list[nn.Module] = []
    cur = i
    for _ in range(n):
        mods += [nn.Linear(cur, h), nn.LayerNorm(h), nn.SiLU()]
        cur = h
    mods.append(nn.Linear(cur, o))
    return nn.Sequential(*mods)


@dataclass
class RSSMConfig:
    deter_dim: int = 1024
    stoch_dim: int = 32
    num_classes: int = 32
    hidden_dim: int = 1024
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
        return torch.cat([self.deter, self.stoch.flatten(-2)], -1)


@dataclass
class ImaginedTrajectory:
    features: Tensor
    actions: Tensor
    reward_logits: Tensor
    continue_logits: Tensor

    @property
    def horizon(self) -> int:
        return self.features.shape[0]

    @property
    def batch_size(self) -> int:
        return self.features.shape[1]

    @property
    def continue_probs(self) -> Tensor:
        return torch.sigmoid(self.continue_logits.squeeze(-1))


class BlockGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
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
        bk, bk1 = self.bin_centers[k], self.bin_centers[k + 1]
        wu = ((xl - bk) / (bk1 - bk + 1e-8)).clamp(0, 1)
        t = torch.zeros(*x.shape, self.num_bins, device=x.device, dtype=x.dtype)
        t.scatter_(-1, k.unsqueeze(-1), (1 - wu).unsqueeze(-1))
        t.scatter_(-1, (k + 1).unsqueeze(-1), wu.unsqueeze(-1))
        return t

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
        z = lambda s: torch.zeros(*s, device=device)
        return RSSMState(z([B, self.cfg.deter_dim]), z([B, self.cfg.stoch_dim, self.cfg.num_classes]), z([B, self.cfg.stoch_dim, self.cfg.num_classes]))

    def observe_step(self, embed: Tensor, action: Tensor, state: RSSMState):
        gru_in = torch.cat([state.stoch.flatten(-2), action], -1)
        h = self.gru(gru_in, state.deter)
        pl = self.prior(h)
        ql = self.posterior(h, embed)
        z = sample_straight_through(ql, self.cfg.unimix)
        return RSSMState(h, z, ql), pl

    def observe(self, embed_seq: Tensor, action_seq: Tensor, state: RSSMState):
        posts, priors = [], []
        for t in range(embed_seq.shape[0]):
            state, pl = self.observe_step(embed_seq[t], action_seq[t], state)
            posts.append(state); priors.append(pl)
        return posts, priors

    def imagine(self, policy: Callable, state: RSSMState, horizon: int) -> ImaginedTrajectory:
        fs, acs, rs, cs = [], [], [], []
        for _ in range(horizon):
            feat = state.features
            action = policy(feat)
            gru_in = torch.cat([state.stoch.flatten(-2), action], -1)
            h = self.gru(gru_in, state.deter)
            pl = self.prior(h)
            z = sample_straight_through(pl, self.cfg.unimix)
            state = RSSMState(h, z, pl)
            nf = state.features
            fs.append(feat); acs.append(action); rs.append(self.reward_head(nf)); cs.append(self.cont_head(nf))
        return ImaginedTrajectory(torch.stack(fs), torch.stack(acs), torch.stack(rs), torch.stack(cs))

    def count_parameters(self) -> Dict[str, int]:
        def c(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters())
        return {
            "gru": c(self.gru),
            "prior_net": c(self.prior),
            "posterior_net": c(self.posterior),
            "reward_head": c(self.reward_head),
            "cont_head": c(self.cont_head),
            "total": c(self),
        }


# ---------------------------------------------------------------------------
# Model size table
# ---------------------------------------------------------------------------

MODEL_SIZES: Dict[str, Dict] = {
    "12M": {"deter_dim": 512,  "hidden_dim": 256,  "num_classes": 16, "stoch_dim": 32},
    "25M": {"deter_dim": 1024, "hidden_dim": 384,  "num_classes": 24, "stoch_dim": 32},
    "50M": {"deter_dim": 2048, "hidden_dim": 512,  "num_classes": 32, "stoch_dim": 32},
    "100M": {"deter_dim": 3072, "hidden_dim": 768, "num_classes": 48, "stoch_dim": 32},
    "200M": {"deter_dim": 4096, "hidden_dim": 1024, "num_classes": 64, "stoch_dim": 32},
}


def get_config_for_size(size: str) -> RSSMConfig:
    if size not in MODEL_SIZES:
        raise ValueError(f"Unknown model size: {size!r}. Available: {list(MODEL_SIZES.keys())}")
    overrides = MODEL_SIZES[size]
    cfg = RSSMConfig()
    for k, v in overrides.items():
        object.__setattr__(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def fmt_size(n: int) -> str:
    """Format a parameter count as a human-readable string."""
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def fmt_shape(t: Tensor) -> str:
    return "(" + ", ".join(str(d) for d in t.shape) + ")"


def fmt_bytes(n_bytes: int) -> str:
    if n_bytes >= 1024**3:
        return f"{n_bytes/1024**3:.2f} GB"
    elif n_bytes >= 1024**2:
        return f"{n_bytes/1024**2:.2f} MB"
    elif n_bytes >= 1024:
        return f"{n_bytes/1024:.1f} KB"
    return f"{n_bytes} B"


def separator(char: str = "-", width: int = 60) -> None:
    print(char * width)


def header(title: str, width: int = 60) -> None:
    separator("=", width)
    print(f"  {title}")
    separator("=", width)


def section(title: str) -> None:
    print()
    separator("-", 60)
    print(f"  {title}")
    separator("-", 60)


# ---------------------------------------------------------------------------
# Main diagnostic runner
# ---------------------------------------------------------------------------

def run_diagnostics(
    model_size: Optional[str],
    batch_size: int,
    seq_len: int,
    horizon: int,
    action_dim: int,
    embed_dim: int,
    device_str: str,
) -> None:
    """Run all diagnostic checks and print results."""

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    # Build config
    if model_size is not None:
        cfg = get_config_for_size(model_size)
        size_label = model_size
    else:
        cfg = RSSMConfig()
        size_label = "custom"

    header(f"DreamerV3 RSSM Diagnostic — Size: {size_label}")
    print()
    print(f"  Device:          {device}")
    print(f"  PyTorch version: {torch.__version__}")
    print()

    # ===========================================================
    section("Configuration")
    # ===========================================================

    print(f"  deter_dim:   {cfg.deter_dim}")
    print(f"  stoch_dim:   {cfg.stoch_dim}")
    print(f"  num_classes: {cfg.num_classes}")
    print(f"  hidden_dim:  {cfg.hidden_dim}")
    print(f"  num_layers:  {cfg.num_layers}")
    print(f"  unimix:      {cfg.unimix}")
    print(f"  embed_dim:   {embed_dim}  (observation encoder output)")
    print(f"  action_dim:  {action_dim}")
    print()
    print(f"  stoch_flat_dim: {cfg.stoch_flat_dim}")
    print(f"  feature_dim:    {cfg.feature_dim}  (deter + stoch_flat)")
    print()
    print(f"  Batch size: {batch_size}")
    print(f"  Seq length: {seq_len}")
    print(f"  Horizon:    {horizon}")

    # ===========================================================
    section("Model Construction")
    # ===========================================================

    t0 = time.perf_counter()
    rssm = RSSM(cfg, embed_dim=embed_dim, action_dim=action_dim).to(device)
    rssm.train(False)
    t_build = time.perf_counter() - t0
    print(f"  Build time: {t_build*1000:.1f} ms")

    # Parameter counts
    counts = rssm.count_parameters()
    print()
    print("  Parameter counts:")
    for name, n in counts.items():
        print(f"    {name:<20} {fmt_size(n):>10} ({n:,})")

    # Memory estimate (float32)
    total_bytes = counts["total"] * 4
    print()
    print(f"  Estimated model memory (fp32): {fmt_bytes(total_bytes)}")
    print(f"  Estimated model memory (fp16): {fmt_bytes(total_bytes // 2)}")

    # ===========================================================
    section("Observe Forward Pass")
    # ===========================================================

    state0 = rssm.initial_state(batch_size, device)
    print(f"  initial_state shapes:")
    print(f"    deter:  {fmt_shape(state0.deter)}")
    print(f"    stoch:  {fmt_shape(state0.stoch)}")
    print(f"    logits: {fmt_shape(state0.logits)}")

    # Single observe step
    embed_single = torch.randn(batch_size, embed_dim, device=device)
    action_single = torch.randn(batch_size, action_dim, device=device)

    t0 = time.perf_counter()
    with torch.no_grad():
        new_state, prior_logits = rssm.observe_step(embed_single, action_single, state0)
    t_obs_step = time.perf_counter() - t0

    print()
    print(f"  observe_step output shapes (B={batch_size}):")
    print(f"    posterior.deter:  {fmt_shape(new_state.deter)}")
    print(f"    posterior.stoch:  {fmt_shape(new_state.stoch)}")
    print(f"    posterior.logits: {fmt_shape(new_state.logits)}")
    print(f"    prior_logits:     {fmt_shape(prior_logits)}")
    print(f"    features:         {fmt_shape(new_state.features)}")
    print(f"  Time: {t_obs_step*1000:.2f} ms")

    # Verify stoch is approximately one-hot
    stoch_sums = new_state.stoch.sum(-1)
    stoch_max_dev = (stoch_sums - 1.0).abs().max().item()
    print()
    print(f"  Stoch sum-to-1 max deviation: {stoch_max_dev:.2e} (should be < 1e-5)")

    # Unimix floor verification
    prior_probs = unimix_probs(prior_logits, cfg.unimix)
    min_prior_prob = prior_probs.min().item()
    print(f"  Prior min probability (unimix floor): {min_prior_prob:.6f} (should be > 0)")

    # Observe full sequence
    embed_seq = torch.randn(seq_len, batch_size, embed_dim, device=device)
    action_seq = torch.randn(seq_len, batch_size, action_dim, device=device)

    t0 = time.perf_counter()
    with torch.no_grad():
        posteriors, priors = rssm.observe(embed_seq, action_seq, rssm.initial_state(batch_size, device))
    t_obs_seq = time.perf_counter() - t0

    print()
    print(f"  observe(T={seq_len}, B={batch_size}) timing: {t_obs_seq*1000:.1f} ms")
    print(f"  Returned {len(posteriors)} posteriors, {len(priors)} prior tensors")
    print(f"  Throughput: {seq_len * batch_size / t_obs_seq:.0f} steps/sec")

    # ===========================================================
    section("Imagine Rollout")
    # ===========================================================

    def random_policy(feat: Tensor) -> Tensor:
        return torch.randn(feat.shape[0], action_dim, device=feat.device)

    start_state = rssm.initial_state(batch_size, device)

    t0 = time.perf_counter()
    with torch.no_grad():
        traj = rssm.imagine(random_policy, start_state, horizon=horizon)
    t_imagine = time.perf_counter() - t0

    print(f"  imagine(H={horizon}, B={batch_size}) output shapes:")
    print(f"    features:        {fmt_shape(traj.features)}")
    print(f"    actions:         {fmt_shape(traj.actions)}")
    print(f"    reward_logits:   {fmt_shape(traj.reward_logits)}")
    print(f"    continue_logits: {fmt_shape(traj.continue_logits)}")
    print()
    print(f"  Time: {t_imagine*1000:.1f} ms")
    print(f"  Throughput: {horizon * batch_size / t_imagine:.0f} steps/sec")

    # Continue probability statistics
    cont_probs = traj.continue_probs  # (H, B)
    print()
    print(f"  Continue probability stats (sigmoid of logits):")
    print(f"    mean: {cont_probs.mean().item():.4f}")
    print(f"    min:  {cont_probs.min().item():.4f}")
    print(f"    max:  {cont_probs.max().item():.4f}")

    # Reward statistics (decoded from logits)
    reward_decoded = rssm.twohot.decode(traj.reward_logits.view(-1, 255))  # (H*B,)
    print()
    print(f"  Decoded reward stats (from random policy):")
    print(f"    mean: {reward_decoded.mean().item():.4f}")
    print(f"    std:  {reward_decoded.std().item():.4f}")
    print(f"    min:  {reward_decoded.min().item():.4f}")
    print(f"    max:  {reward_decoded.max().item():.4f}")

    # ===========================================================
    section("Symlog Twohot Round-Trip")
    # ===========================================================

    test_values = [-1e6, -1000.0, -100.0, -10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0, 100.0, 1000.0, 1e6]
    twohot_module = rssm.twohot

    print(f"  {'Value':>15}  {'Encoded as':>20}  {'Decoded':>15}  {'Abs Error':>12}  {'Rel Error':>12}")
    print(f"  {'-'*15}  {'-'*20}  {'-'*15}  {'-'*12}  {'-'*12}")

    max_rel_err = 0.0
    for v in test_values:
        x = torch.tensor([v], device=device)
        with torch.no_grad():
            th = twohot_module.encode(x)
            k_max = th.argmax(-1).item()
            logits = th.clamp(1e-7).log()
            decoded = twohot_module.decode(logits).item()

        abs_err = abs(decoded - v)
        rel_err = abs_err / (abs(v) + 1e-8)
        max_rel_err = max(max_rel_err, rel_err)

        print(f"  {v:>15.4g}  bin {k_max:>3d}/{twohot_module.num_bins-1}            {decoded:>15.4g}  {abs_err:>12.4g}  {rel_err:>12.4f}")

    print()
    print(f"  Max relative error: {max_rel_err:.4f}")
    print(f"  Note: large values (|x| > symexp(20) ≈ 485M) are clamped, introducing error.")

    # Loss on random logits
    logits_rand = torch.randn(16, 255, device=device)
    targets_rand = torch.tensor([-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0, 1e6,
                                  -1e6, 0.5, -0.5, 42.0, -42.0, 0.001, -0.001, 3.14],
                                device=device)
    with torch.no_grad():
        loss_vals = twohot_module.loss(logits_rand, targets_rand)
    print()
    print(f"  Twohot loss statistics (random logits):")
    print(f"    mean: {loss_vals.mean().item():.4f}")
    print(f"    min:  {loss_vals.min().item():.4f}")
    print(f"    max:  {loss_vals.max().item():.4f}")
    print(f"    all finite: {torch.all(torch.isfinite(loss_vals)).item()}")

    # ===========================================================
    section("KL Balancing")
    # ===========================================================

    # Use posteriors and priors from the observe pass
    batch_post_logits = posteriors[-1].logits  # last timestep posterior logits
    batch_prior_logits = priors[-1]            # last timestep prior logits

    with torch.no_grad():
        # Per-distribution KL (batch, stoch_dim)
        kl_dyn_per = kl_categorical(batch_prior_logits, batch_post_logits.detach())
        kl_rep_per = kl_categorical(batch_prior_logits.detach(), batch_post_logits)

        FREE_NATS = 1.0
        kl_dyn = torch.clamp(kl_dyn_per, min=FREE_NATS).sum(-1).mean()
        kl_rep = torch.clamp(kl_rep_per, min=FREE_NATS).sum(-1).mean()
        kl_total = 0.5 * kl_dyn + 0.1 * kl_rep

    print(f"  KL statistics (last observe timestep):")
    print(f"    kl_dyn_per (before clamp):")
    print(f"      mean: {kl_dyn_per.mean().item():.4f} nats/distribution")
    print(f"      max:  {kl_dyn_per.max().item():.4f} nats/distribution")
    print(f"      fraction < free_nats: {(kl_dyn_per < FREE_NATS).float().mean().item():.3f}")
    print(f"    kl_rep_per (before clamp):")
    print(f"      mean: {kl_rep_per.mean().item():.4f} nats/distribution")
    print(f"      max:  {kl_rep_per.max().item():.4f} nats/distribution")
    print()
    print(f"    L_dyn (after free_nats={FREE_NATS}, sum over stoch_dim, mean over batch):")
    print(f"      {kl_dyn.item():.4f}")
    print(f"    L_rep (after free_nats={FREE_NATS}, sum over stoch_dim, mean over batch):")
    print(f"      {kl_rep.item():.4f}")
    print(f"    L_KL = 0.5 * L_dyn + 0.1 * L_rep:")
    print(f"      {kl_total.item():.4f}")

    # Entropy of distributions
    post_probs = unimix_probs(batch_post_logits)
    post_entropy = -(post_probs * post_probs.clamp(1e-10).log()).sum(-1)  # (B, stoch_dim)
    max_entropy = math.log(cfg.num_classes)

    print()
    print(f"  Posterior entropy (max = ln({cfg.num_classes}) ≈ {max_entropy:.3f}):")
    print(f"    mean: {post_entropy.mean().item():.4f} nats")
    print(f"    min:  {post_entropy.min().item():.4f} nats  (collapse risk if < 0.1)")
    print(f"    max:  {post_entropy.max().item():.4f} nats")
    print()

    collapse_risk = post_entropy.mean().item() < 0.1
    print(f"  Posterior collapse risk: {'YES — very low entropy' if collapse_risk else 'No'}")

    # ===========================================================
    section("Feature Vector")
    # ===========================================================

    feat = posteriors[-1].features
    print(f"  Feature vector shape: {fmt_shape(feat)}")
    print(f"  Feature dim breakdown:")
    print(f"    deter_dim:           {cfg.deter_dim}")
    print(f"    stoch_flat_dim:      {cfg.stoch_flat_dim}  ({cfg.stoch_dim} x {cfg.num_classes})")
    print(f"    total feature_dim:   {cfg.feature_dim}")
    print()
    print(f"  Feature statistics (after observe):")
    print(f"    mean: {feat.mean().item():.4f}")
    print(f"    std:  {feat.std().item():.4f}")
    print(f"    min:  {feat.min().item():.4f}")
    print(f"    max:  {feat.max().item():.4f}")

    # ===========================================================
    section("Memory Estimates (forward pass)")
    # ===========================================================

    def tensor_bytes(t: Tensor) -> int:
        return t.element_size() * t.nelement()

    state_bytes = tensor_bytes(state0.deter) + tensor_bytes(state0.stoch) + tensor_bytes(state0.logits)
    traj_bytes = (tensor_bytes(traj.features) + tensor_bytes(traj.actions) +
                  tensor_bytes(traj.reward_logits) + tensor_bytes(traj.continue_logits))
    seq_bytes = (tensor_bytes(embed_seq) + tensor_bytes(action_seq))

    print(f"  Single RSSM state (B={batch_size}):  {fmt_bytes(state_bytes)}")
    print(f"  Input sequence (T={seq_len}, B={batch_size}): {fmt_bytes(seq_bytes)}")
    print(f"  Imagined trajectory (H={horizon}, B={batch_size}): {fmt_bytes(traj_bytes)}")

    # ===========================================================
    section("Summary")
    # ===========================================================

    all_ok = True
    checks = [
        ("State shapes correct", new_state.deter.shape == (batch_size, cfg.deter_dim)),
        ("Stoch shape correct", new_state.stoch.shape == (batch_size, cfg.stoch_dim, cfg.num_classes)),
        ("Trajectory horizon correct", traj.horizon == horizon),
        ("Stoch sums to 1", stoch_max_dev < 1e-4),
        ("Prior probs > 0 (unimix)", min_prior_prob > 0),
        ("KL finite", torch.isfinite(kl_total).item()),
        ("Twohot loss finite", torch.all(torch.isfinite(loss_vals)).item()),
    ]

    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_ok = False

    print()
    separator("=")
    if all_ok:
        print("  All diagnostic checks passed.")
    else:
        print("  Some diagnostic checks FAILED — see above.")
    separator("=")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DreamerV3 RSSM diagnostic tool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/rssm_diagnostic.py
  python scripts/rssm_diagnostic.py --model_size 50M
  python scripts/rssm_diagnostic.py --model_size 12M --batch_size 8 --seq_len 64 --horizon 20
  python scripts/rssm_diagnostic.py --batch_size 1 --action_dim 4 --embed_dim 512
        """,
    )
    parser.add_argument(
        "--model_size", type=str, default=None,
        choices=list(MODEL_SIZES.keys()),
        help="Model size preset. If omitted, uses default RSSMConfig (deter=1024, hidden=1024, classes=32).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size for diagnostic runs. Default: 4.",
    )
    parser.add_argument(
        "--seq_len", type=int, default=16,
        help="Sequence length for observe pass. Default: 16.",
    )
    parser.add_argument(
        "--horizon", type=int, default=15,
        help="Imagination horizon. Default: 15.",
    )
    parser.add_argument(
        "--action_dim", type=int, default=4,
        help="Action space dimensionality. Default: 4.",
    )
    parser.add_argument(
        "--embed_dim", type=int, default=512,
        help="Observation encoder embedding dimension. Default: 512.",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: 'cpu', 'cuda', or 'auto'. Default: auto.",
    )
    args = parser.parse_args()

    run_diagnostics(
        model_size=args.model_size,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        horizon=args.horizon,
        action_dim=args.action_dim,
        embed_dim=args.embed_dim,
        device_str=args.device,
    )


# ---------------------------------------------------------------------------
# Self-test block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
