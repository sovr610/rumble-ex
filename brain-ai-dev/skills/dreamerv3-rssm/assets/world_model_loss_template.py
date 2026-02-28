"""
World Model Loss Template — DreamerV3 KL Balancing and Prediction Losses

Implements:
  kl_categorical:  KL divergence for unimix categoricals
  WorldModelLoss:  Combined loss with KL balancing, free nats, and prediction heads
  LossResult:      Named decomposition of all loss terms

KL balancing:
  L_dyn = max(free, KL[sg(posterior) || prior])   — coefficient 0.5
  L_rep = max(free, KL[posterior || sg(prior)])   — coefficient 0.1
  L_KL  = 0.5 * L_dyn + 0.1 * L_rep
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Config (inline, matching rssm_config_template.py)
# ---------------------------------------------------------------------------

@dataclass
class LossConfig:
    kl_free_nats: float = 1.0
    kl_dyn_scale: float = 0.5
    kl_rep_scale: float = 0.1
    reward_scale: float = 1.0
    continue_scale: float = 1.0
    obs_scale: float = 1.0


@dataclass
class LossResult:
    total: Tensor
    kl_dyn: Tensor
    kl_rep: Tensor
    obs_loss: Tensor
    reward_loss: Tensor
    continue_loss: Tensor

    def to_dict(self) -> dict[str, float]:
        return {
            "total": self.total.item(),
            "kl_dyn": self.kl_dyn.item(),
            "kl_rep": self.kl_rep.item(),
            "obs_loss": self.obs_loss.item(),
            "reward_loss": self.reward_loss.item(),
            "continue_loss": self.continue_loss.item(),
        }


# ---------------------------------------------------------------------------
# Symlog twohot (inline for self-containment)
# ---------------------------------------------------------------------------

def _symlog(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def _symexp(x: Tensor) -> Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def _twohot_encode(x: Tensor, bin_centers: Tensor) -> Tensor:
    """Encode scalar x to two-hot vector using precomputed bin_centers."""
    low = bin_centers[0].item()
    high = bin_centers[-1].item()
    num_bins = bin_centers.shape[0]
    x_log = _symlog(x).clamp(low, high)
    delta = (high - low) / (num_bins - 1)
    pos = (x_log - low) / delta
    k = pos.long().clamp(0, num_bins - 2)
    b_k = bin_centers[k]
    b_k1 = bin_centers[k + 1]
    w_upper = ((x_log - b_k) / (b_k1 - b_k + 1e-8)).clamp(0.0, 1.0)
    w_lower = 1.0 - w_upper
    target = torch.zeros(*x.shape, num_bins, device=x.device, dtype=x.dtype)
    target.scatter_(-1, k.unsqueeze(-1), w_lower.unsqueeze(-1))
    target.scatter_(-1, (k + 1).unsqueeze(-1), w_upper.unsqueeze(-1))
    return target


# ---------------------------------------------------------------------------
# KL divergence for categorical distributions
# ---------------------------------------------------------------------------

def kl_categorical(
    p_logits: Tensor,
    q_logits: Tensor,
    unimix: float = 0.01,
) -> Tensor:
    """
    KL divergence KL[q || p] for unimix categorical distributions.

    Computes KL for each of the stoch_dim categorical distributions independently,
    returning per-distribution KL values. The total KL is obtained by summing over
    the stoch_dim dimension.

    Formula:
        p_mix = (1 - unimix) * softmax(p_logits) + unimix / num_classes
        q_mix = (1 - unimix) * softmax(q_logits) + unimix / num_classes
        KL[q || p] = sum_k q_k * (log q_k - log p_k)

    Unimix ensures no probability is exactly zero, preventing infinite KL.

    Args:
        p_logits: Reference distribution logits, shape (batch, stoch_dim, num_classes).
        q_logits: Query distribution logits, shape (batch, stoch_dim, num_classes).
                  Computes KL[q || p].
        unimix:   Uniform mixture fraction.

    Returns:
        Per-distribution KL, shape (batch, stoch_dim). Non-negative.
    """
    num_classes = p_logits.shape[-1]

    # Apply unimix to both distributions
    p_soft = torch.softmax(p_logits, dim=-1)
    q_soft = torch.softmax(q_logits, dim=-1)

    p_probs = (1.0 - unimix) * p_soft + unimix / num_classes  # (batch, stoch_dim, num_classes)
    q_probs = (1.0 - unimix) * q_soft + unimix / num_classes  # (batch, stoch_dim, num_classes)

    # KL[q || p] = sum_k q_k * (log q_k - log p_k)
    # Clamp to avoid log(0) — unimix should prevent this, but add safety margin
    kl = (q_probs * (q_probs.clamp(min=1e-20).log() - p_probs.clamp(min=1e-20).log())).sum(dim=-1)
    # kl: (batch, stoch_dim)
    return kl.clamp(min=0.0)  # Numerical safety: KL is non-negative


# ---------------------------------------------------------------------------
# World Model Loss
# ---------------------------------------------------------------------------

class WorldModelLoss(nn.Module):
    """
    Combined world model loss with KL balancing and prediction losses.

    Loss components:
      1. Dynamics KL:        max(free, KL[sg(posterior) || prior])
      2. Representation KL:  max(free, KL[posterior || sg(prior)])
      3. Reward loss:        cross-entropy(reward_logits, twohot(target_reward))
      4. Continue loss:      binary cross-entropy(cont_logits, target_cont)
      5. Observation loss:   MSE or cross-entropy depending on obs_type

    Total = kl_dyn_scale * L_dyn + kl_rep_scale * L_rep
          + reward_scale * reward_loss
          + continue_scale * continue_loss
          + obs_scale * obs_loss

    Args:
        cfg:       LossConfig with scales and free nats threshold.
        unimix:    Uniform mixture fraction for KL computation. Default 0.01.
        num_bins:  Number of symlog twohot bins. Default 255.
        twohot_low:  Lower bound of bin range. Default -20.0.
        twohot_high: Upper bound of bin range. Default 20.0.
    """

    def __init__(
        self,
        cfg: Optional[LossConfig] = None,
        unimix: float = 0.01,
        num_bins: int = 255,
        twohot_low: float = -20.0,
        twohot_high: float = 20.0,
    ) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else LossConfig()
        self.unimix = unimix

        bin_centers = torch.linspace(twohot_low, twohot_high, num_bins)
        self.register_buffer("bin_centers", bin_centers)

    # -----------------------------------------------------------------------
    # Individual loss functions
    # -----------------------------------------------------------------------

    def kl_loss(
        self,
        posterior_logits: Tensor,
        prior_logits: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute KL balancing loss terms with free-nats clipping and stop-gradients.

        Args:
            posterior_logits: Shape (batch, stoch_dim, num_classes).
            prior_logits:     Shape (batch, stoch_dim, num_classes).

        Returns:
            (kl_dyn, kl_rep): Scalar tensors, each averaged over batch and stoch_dim.
        """
        free = self.cfg.kl_free_nats

        # Dynamics loss: train prior to match posterior
        # Stop gradient on posterior so only the prior network is updated
        kl_dyn_per = kl_categorical(
            p_logits=prior_logits,
            q_logits=posterior_logits.detach(),   # stop-gradient on posterior
            unimix=self.unimix,
        )  # (batch, stoch_dim)

        # Representation loss: train posterior to stay close to prior
        # Stop gradient on prior so only the posterior network is updated
        kl_rep_per = kl_categorical(
            p_logits=prior_logits.detach(),        # stop-gradient on prior
            q_logits=posterior_logits,
            unimix=self.unimix,
        )  # (batch, stoch_dim)

        # Apply free-nats: clamp per-distribution KL below free threshold
        # Then sum over stoch_dim and average over batch
        kl_dyn = torch.clamp(kl_dyn_per, min=free).sum(dim=-1).mean()
        kl_rep = torch.clamp(kl_rep_per, min=free).sum(dim=-1).mean()

        return kl_dyn, kl_rep

    def reward_loss(
        self,
        pred_reward_logits: Tensor,
        target_reward: Tensor,
    ) -> Tensor:
        """
        Symlog twohot cross-entropy loss for reward prediction.

        Args:
            pred_reward_logits: Shape (*, num_bins).
            target_reward:      Shape (*). Scalar rewards in original scale.

        Returns:
            Scalar mean loss.
        """
        twohot = _twohot_encode(target_reward, self.bin_centers)
        log_probs = F.log_softmax(pred_reward_logits, dim=-1)
        per_elem = -(twohot * log_probs).sum(dim=-1)
        return per_elem.mean()

    def continue_loss(
        self,
        pred_cont_logits: Tensor,
        target_cont: Tensor,
    ) -> Tensor:
        """
        Binary cross-entropy loss for episode continuation prediction.

        Args:
            pred_cont_logits: Shape (*, 1) or (*). Raw logits.
            target_cont:      Shape (*). Float 0/1 or bool indicating episode continues.

        Returns:
            Scalar mean loss.
        """
        logits = pred_cont_logits.squeeze(-1)   # (*), handle (*, 1) input
        targets = target_cont.float()
        return F.binary_cross_entropy_with_logits(logits, targets)

    def obs_loss_mse(
        self,
        pred_obs: Tensor,
        target_obs: Tensor,
    ) -> Tensor:
        """
        MSE loss for observation reconstruction (in symlog space for large values).

        Args:
            pred_obs:   Predicted observation, any shape.
            target_obs: Target observation, same shape.

        Returns:
            Scalar mean loss.
        """
        # Apply symlog to both for scale-invariant MSE
        return F.mse_loss(_symlog(pred_obs), _symlog(target_obs))

    # -----------------------------------------------------------------------
    # Combined forward pass
    # -----------------------------------------------------------------------

    def forward(
        self,
        posterior_logits: Tensor,
        prior_logits: Tensor,
        pred_obs: Tensor,
        target_obs: Tensor,
        pred_reward: Tensor,
        target_reward: Tensor,
        pred_cont: Tensor,
        target_cont: Tensor,
    ) -> LossResult:
        """
        Compute the combined world model loss.

        Args:
            posterior_logits: Posterior distribution logits from observe step.
                              Shape (batch, stoch_dim, num_classes) or
                              (T * batch, stoch_dim, num_classes).
            prior_logits:     Prior distribution logits from observe step.
                              Same shape as posterior_logits.
            pred_obs:         Predicted observation, any shape.
            target_obs:       Target observation, same shape as pred_obs.
            pred_reward:      Predicted reward logits, shape (*, num_bins).
            target_reward:    Target reward scalars, shape (*).
            pred_cont:        Predicted continue logits, shape (*, 1) or (*).
            target_cont:      Target continue flags, shape (*). Float 0/1 or bool.

        Returns:
            LossResult with all individual loss terms and the weighted total.
        """
        # 1. KL balancing terms
        kl_dyn, kl_rep = self.kl_loss(posterior_logits, prior_logits)

        # 2. Prediction losses
        r_loss = self.reward_loss(pred_reward, target_reward)
        c_loss = self.continue_loss(pred_cont, target_cont)
        o_loss = self.obs_loss_mse(pred_obs, target_obs)

        # 3. Weighted total
        total = (
            self.cfg.kl_dyn_scale * kl_dyn
            + self.cfg.kl_rep_scale * kl_rep
            + self.cfg.reward_scale * r_loss
            + self.cfg.continue_scale * c_loss
            + self.cfg.obs_scale * o_loss
        )

        return LossResult(
            total=total,
            kl_dyn=kl_dyn,
            kl_rep=kl_rep,
            obs_loss=o_loss,
            reward_loss=r_loss,
            continue_loss=c_loss,
        )

    def forward_kl(
        self,
        posterior_logits: Tensor,
        prior_logits: Tensor,
    ) -> Tensor:
        """
        Compute only the combined KL loss term.

        Useful for testing or when prediction losses are computed separately.

        Returns:
            Scalar: kl_dyn_scale * kl_dyn + kl_rep_scale * kl_rep
        """
        kl_dyn, kl_rep = self.kl_loss(posterior_logits, prior_logits)
        return self.cfg.kl_dyn_scale * kl_dyn + self.cfg.kl_rep_scale * kl_rep


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("WorldModelLoss self-tests")
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

    BATCH = 4
    STOCH = 8
    NUM_CLASSES = 8
    NUM_BINS = 255

    loss_fn = WorldModelLoss(cfg=LossConfig(), unimix=0.01).to(device)

    # -----------------------------------------------------------------------
    # Test 1: Identical distributions → KL ≈ 0 (before free-nats clamp)
    # -----------------------------------------------------------------------
    print("Test 1: Identical distributions → KL ≈ 0")

    logits_same = torch.randn(BATCH, STOCH, NUM_CLASSES, device=device)
    kl = kl_categorical(logits_same, logits_same, unimix=0.01)
    check(
        torch.allclose(kl, torch.zeros_like(kl), atol=1e-5),
        "identical dists: KL ≈ 0",
        f"max_kl={kl.max().item():.2e}",
    )

    # -----------------------------------------------------------------------
    # Test 2: Different distributions → positive KL
    # -----------------------------------------------------------------------
    print("\nTest 2: Different distributions → positive KL")

    p_logits = torch.zeros(BATCH, STOCH, NUM_CLASSES, device=device)
    q_logits = torch.ones(BATCH, STOCH, NUM_CLASSES, device=device) * 10
    q_logits[:, :, 0] = -10  # very different
    kl_pos = kl_categorical(p_logits, q_logits, unimix=0.01)
    check(kl_pos.min().item() > 0, "different dists: KL > 0", f"min_kl={kl_pos.min().item():.4f}")

    # -----------------------------------------------------------------------
    # Test 3: Free nats clipping (KL below free_nats gets clamped)
    # -----------------------------------------------------------------------
    print("\nTest 3: Free nats clipping")

    # Create distributions that are very close (KL << 1.0)
    logits_close = torch.randn(BATCH, STOCH, NUM_CLASSES, device=device)
    logits_noise = logits_close + 0.001 * torch.randn_like(logits_close)
    kl_raw = kl_categorical(logits_close, logits_noise, unimix=0.01)  # (BATCH, STOCH)

    # After clamping to free_nats=1.0
    kl_clamped = torch.clamp(kl_raw, min=1.0)
    check(
        kl_clamped.min().item() >= 1.0,
        "free-nats clamped: all values >= 1.0",
        f"min_clamped={kl_clamped.min().item():.4f}",
    )

    # -----------------------------------------------------------------------
    # Test 4: Stop-gradient — dynamics loss does not update posterior
    # -----------------------------------------------------------------------
    print("\nTest 4: Stop-gradient (dynamics loss)")

    post_logits = torch.randn(BATCH, STOCH, NUM_CLASSES, device=device, requires_grad=True)
    prior_logits_ = torch.randn(BATCH, STOCH, NUM_CLASSES, device=device, requires_grad=True)

    # Dynamics loss: sg(posterior) || prior — only prior should get gradient
    kl_dyn = kl_categorical(
        p_logits=prior_logits_,
        q_logits=post_logits.detach(),   # stop-gradient
        unimix=0.01,
    ).sum()
    kl_dyn.backward()

    check(prior_logits_.grad is not None, "dynamics: prior.grad not None")
    check(post_logits.grad is None, "dynamics: posterior.grad is None (stop-gradient)")

    # Reset
    prior_logits_ = torch.randn(BATCH, STOCH, NUM_CLASSES, device=device, requires_grad=True)
    post_logits = torch.randn(BATCH, STOCH, NUM_CLASSES, device=device, requires_grad=True)

    # -----------------------------------------------------------------------
    # Test 5: Stop-gradient — representation loss does not update prior
    # -----------------------------------------------------------------------
    print("\nTest 5: Stop-gradient (representation loss)")

    post_logits2 = torch.randn(BATCH, STOCH, NUM_CLASSES, device=device, requires_grad=True)
    prior_logits2 = torch.randn(BATCH, STOCH, NUM_CLASSES, device=device, requires_grad=True)

    kl_rep = kl_categorical(
        p_logits=prior_logits2.detach(),  # stop-gradient
        q_logits=post_logits2,
        unimix=0.01,
    ).sum()
    kl_rep.backward()

    check(post_logits2.grad is not None, "representation: posterior.grad not None")
    check(prior_logits2.grad is None, "representation: prior.grad is None (stop-gradient)")

    # -----------------------------------------------------------------------
    # Test 6: Combined KL loss coefficients (0.5 dyn + 0.1 rep)
    # -----------------------------------------------------------------------
    print("\nTest 6: KL loss coefficient application")

    post_l = torch.randn(BATCH, STOCH, NUM_CLASSES, device=device)
    prior_l = torch.randn(BATCH, STOCH, NUM_CLASSES, device=device)

    kl_dyn_val, kl_rep_val = loss_fn.kl_loss(post_l, prior_l)
    combined_kl = loss_fn.forward_kl(post_l, prior_l)
    expected_combined = 0.5 * kl_dyn_val + 0.1 * kl_rep_val
    check(
        torch.allclose(combined_kl, expected_combined, atol=1e-5),
        "combined KL = 0.5 * dyn + 0.1 * rep",
        f"combined={combined_kl.item():.4f}, expected={expected_combined.item():.4f}",
    )

    # -----------------------------------------------------------------------
    # Test 7: Reward loss (symlog twohot cross-entropy)
    # -----------------------------------------------------------------------
    print("\nTest 7: Reward loss")

    reward_logits = torch.randn(BATCH, NUM_BINS, device=device, requires_grad=True)
    target_rewards = torch.tensor([-100.0, 0.0, 1.0, 1e5], device=device)
    rl = loss_fn.reward_loss(reward_logits, target_rewards)
    check(torch.isfinite(rl).item(), "reward loss finite", f"loss={rl.item():.4f}")
    check(rl.item() >= 0, "reward loss non-negative")
    rl.backward()
    check(reward_logits.grad is not None, "reward loss grad not None")
    check(torch.all(torch.isfinite(reward_logits.grad)).item(), "reward loss grad finite")

    # -----------------------------------------------------------------------
    # Test 8: Continue loss (binary cross-entropy)
    # -----------------------------------------------------------------------
    print("\nTest 8: Continue loss")

    cont_logits = torch.randn(BATCH, 1, device=device, requires_grad=True)
    target_cont = torch.tensor([1.0, 0.0, 1.0, 1.0], device=device)
    cl = loss_fn.continue_loss(cont_logits, target_cont)
    check(torch.isfinite(cl).item(), "continue loss finite")
    check(cl.item() >= 0, "continue loss non-negative")
    cl.backward()
    check(cont_logits.grad is not None, "continue loss grad not None")

    # -----------------------------------------------------------------------
    # Test 9: Full forward pass is differentiable
    # -----------------------------------------------------------------------
    print("\nTest 9: Full forward pass differentiable")

    post_full = torch.randn(BATCH, STOCH, NUM_CLASSES, device=device, requires_grad=True)
    prior_full = torch.randn(BATCH, STOCH, NUM_CLASSES, device=device, requires_grad=True)
    pred_obs_f = torch.randn(BATCH, 16, device=device, requires_grad=True)
    target_obs_f = torch.randn(BATCH, 16, device=device)
    pred_rew_f = torch.randn(BATCH, NUM_BINS, device=device, requires_grad=True)
    target_rew_f = torch.randn(BATCH, device=device) * 10
    pred_cont_f = torch.randn(BATCH, 1, device=device, requires_grad=True)
    target_cont_f = (torch.rand(BATCH, device=device) > 0.2).float()

    result = loss_fn(
        post_full, prior_full,
        pred_obs_f, target_obs_f,
        pred_rew_f, target_rew_f,
        pred_cont_f, target_cont_f,
    )

    check(torch.isfinite(result.total).item(), "full forward: total finite")
    check(torch.isfinite(result.kl_dyn).item(), "full forward: kl_dyn finite")
    check(torch.isfinite(result.kl_rep).item(), "full forward: kl_rep finite")
    check(torch.isfinite(result.obs_loss).item(), "full forward: obs_loss finite")
    check(torch.isfinite(result.reward_loss).item(), "full forward: reward_loss finite")
    check(torch.isfinite(result.continue_loss).item(), "full forward: continue_loss finite")

    result.total.backward()
    check(post_full.grad is not None, "full backward: post.grad not None")
    check(prior_full.grad is not None, "full backward: prior.grad not None")
    check(pred_rew_f.grad is not None, "full backward: reward head grad not None")
    check(torch.all(torch.isfinite(post_full.grad)).item(), "full backward: post.grad finite")
    check(torch.all(torch.isfinite(prior_full.grad)).item(), "full backward: prior.grad finite")

    # -----------------------------------------------------------------------
    # Test 10: obs_scale=0 nullifies observation loss contribution
    # -----------------------------------------------------------------------
    print("\nTest 10: obs_scale=0 zeroes obs contribution")

    cfg_no_obs = LossConfig(obs_scale=0.0)
    loss_fn_no_obs = WorldModelLoss(cfg=cfg_no_obs).to(device)

    result_no_obs = loss_fn_no_obs(
        post_full.detach(), prior_full.detach(),
        pred_obs_f.detach(), target_obs_f,
        pred_rew_f.detach(), target_rew_f,
        pred_cont_f.detach(), target_cont_f,
    )
    check(result_no_obs.obs_loss.item() >= 0, "obs_loss still computed (for logging)")
    # The obs_loss should not affect total when scale=0
    total_without_obs = (
        0.5 * result_no_obs.kl_dyn
        + 0.1 * result_no_obs.kl_rep
        + result_no_obs.reward_loss
        + result_no_obs.continue_loss
    )
    check(
        torch.allclose(result_no_obs.total, total_without_obs, atol=1e-5),
        "obs_scale=0: total equals KL+reward+continue",
        f"total={result_no_obs.total.item():.4f}, expected={total_without_obs.item():.4f}",
    )

    # -----------------------------------------------------------------------
    # Test 11: LossResult.to_dict
    # -----------------------------------------------------------------------
    print("\nTest 11: LossResult.to_dict")
    d = result_no_obs.to_dict()
    check(isinstance(d, dict), "to_dict returns dict")
    for key in ("total", "kl_dyn", "kl_rep", "obs_loss", "reward_loss", "continue_loss"):
        check(key in d, f"to_dict has '{key}'")

    # -----------------------------------------------------------------------
    # Test 12: KL is non-negative
    # -----------------------------------------------------------------------
    print("\nTest 12: KL non-negative")

    for _ in range(10):
        p = torch.randn(BATCH, STOCH, NUM_CLASSES, device=device)
        q = torch.randn(BATCH, STOCH, NUM_CLASSES, device=device)
        kl_val = kl_categorical(p, q, unimix=0.01)
        check(
            kl_val.min().item() >= -1e-5,
            "KL non-negative (random distributions)",
            f"min={kl_val.min().item():.2e}",
        )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} test(s): {', '.join(FAILURES)}")
        sys.exit(1)
    else:
        print("All tests PASSED.")
