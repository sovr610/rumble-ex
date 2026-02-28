"""
Categorical State Template — DreamerV3-Style Stochastic State

Components:
  - UnimixCategorical: categorical distribution with 1% uniform mixture
  - sample_straight_through: differentiable discrete sampling
  - PriorNet: 2-layer MLP (h_t -> logits)
  - PosteriorNet: 2-layer MLP (concat(h_t, embed_t) -> logits)

The stochastic state z_t is (batch, stoch_dim, num_classes) — stoch_dim independent
categorical distributions, each over num_classes classes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Unimix helpers
# ---------------------------------------------------------------------------

def unimix_probs(logits: Tensor, unimix: float = 0.01) -> Tensor:
    """
    Compute unimix categorical probabilities.

    Formula:
        p = (1 - unimix) * softmax(logits, dim=-1) + unimix / num_classes

    Args:
        logits: Raw logits of shape (*, num_classes).
        unimix: Uniform mixture fraction. Default 0.01.

    Returns:
        Probabilities of shape (*, num_classes). Always sums to 1 and is > 0.
    """
    num_classes = logits.shape[-1]
    soft = torch.softmax(logits, dim=-1)
    uniform = torch.ones_like(soft) / num_classes
    return (1.0 - unimix) * soft + unimix * uniform


def sample_straight_through(logits: Tensor, unimix: float = 0.01) -> Tensor:
    """
    Sample from the unimix categorical distribution using the straight-through estimator.

    Forward pass: returns one-hot(argmax(logits)).
    Backward pass: gradients flow through the soft unimix probabilities.

    This makes the discrete sampling step differentiable: parameters that produce
    `logits` receive gradients as if the sampling were soft.

    Args:
        logits: Shape (*, stoch_dim, num_classes).
        unimix: Uniform mixture fraction.

    Returns:
        z: Shape (*, stoch_dim, num_classes). One-hot in forward pass; differentiable.
    """
    # Soft probabilities — differentiable path for gradients
    probs = unimix_probs(logits, unimix=unimix)  # (*, stoch_dim, num_classes)

    # Hard sample: argmax of the soft probabilities (or of raw logits; equivalent for unimix)
    indices = probs.argmax(dim=-1)  # (*, stoch_dim)
    num_classes = logits.shape[-1]
    z_hard = F.one_hot(indices, num_classes).to(probs.dtype)  # (*, stoch_dim, num_classes)

    # Straight-through trick: forward uses z_hard, backward uses probs
    # z = z_hard - probs.detach() + probs
    # ↑ In the forward pass this equals z_hard.
    # ↑ In the backward pass, d(z)/d(probs) = 1, so gradients flow to logits via probs.
    z = z_hard - probs.detach() + probs
    return z


# ---------------------------------------------------------------------------
# MLP building block
# ---------------------------------------------------------------------------

def _make_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    num_layers: int = 2,
) -> nn.Sequential:
    """
    Build a 2-layer MLP with LayerNorm + SiLU on intermediate layers.

    Structure (num_layers=2):
        Linear(in, hidden) -> LayerNorm(hidden) -> SiLU
        -> Linear(hidden, hidden) -> LayerNorm(hidden) -> SiLU
        -> Linear(hidden, out)

    Args:
        in_dim:     Input feature dimension.
        hidden_dim: Hidden layer width.
        out_dim:    Output feature dimension.
        num_layers: Number of hidden layers before the final linear.

    Returns:
        nn.Sequential MLP.
    """
    layers: list[nn.Module] = []
    current = in_dim
    for _ in range(num_layers):
        layers.append(nn.Linear(current, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.SiLU())
        current = hidden_dim
    layers.append(nn.Linear(current, out_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Prior Network
# ---------------------------------------------------------------------------

class PriorNet(nn.Module):
    """
    Prior distribution network: p(z_t | h_t).

    Predicts the distribution over the stochastic state given only the
    deterministic state. Used during imagination rollouts (no observations).

    Args:
        deter_dim:   Dimensionality of the deterministic state h_t.
        stoch_dim:   Number of independent categorical distributions.
        num_classes: Number of classes per distribution.
        hidden_dim:  MLP hidden layer width.
        num_layers:  Number of MLP hidden layers.
        unimix:      Uniform mixture fraction for sampling.
    """

    def __init__(
        self,
        deter_dim: int,
        stoch_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        unimix: float = 0.01,
    ) -> None:
        super().__init__()
        self.stoch_dim = stoch_dim
        self.num_classes = num_classes
        self.unimix = unimix

        out_dim = stoch_dim * num_classes
        self.mlp = _make_mlp(deter_dim, hidden_dim, out_dim, num_layers)

    def forward(self, h: Tensor) -> Tensor:
        """
        Compute prior logits.

        Args:
            h: Deterministic state, shape (batch, deter_dim).

        Returns:
            logits: Shape (batch, stoch_dim, num_classes).
        """
        batch = h.shape[0]
        logits_flat = self.mlp(h)  # (batch, stoch_dim * num_classes)
        return logits_flat.view(batch, self.stoch_dim, self.num_classes)

    def probs(self, h: Tensor) -> Tensor:
        """Return unimix probabilities. Shape: (batch, stoch_dim, num_classes)."""
        return unimix_probs(self.forward(h), unimix=self.unimix)

    def sample(self, h: Tensor) -> tuple[Tensor, Tensor]:
        """
        Sample z_t from the prior using straight-through estimator.

        Returns:
            z:      Sample, shape (batch, stoch_dim, num_classes).
            logits: Raw logits, shape (batch, stoch_dim, num_classes).
        """
        logits = self.forward(h)
        z = sample_straight_through(logits, unimix=self.unimix)
        return z, logits


# ---------------------------------------------------------------------------
# Posterior Network
# ---------------------------------------------------------------------------

class PosteriorNet(nn.Module):
    """
    Posterior distribution network: q(z_t | h_t, embed_t).

    Predicts the distribution over the stochastic state given both the
    deterministic state and the observation embedding. Used during the
    observe (encoding) step.

    Args:
        deter_dim:   Dimensionality of the deterministic state h_t.
        embed_dim:   Dimensionality of the observation embedding.
        stoch_dim:   Number of independent categorical distributions.
        num_classes: Number of classes per distribution.
        hidden_dim:  MLP hidden layer width.
        num_layers:  Number of MLP hidden layers.
        unimix:      Uniform mixture fraction for sampling.
    """

    def __init__(
        self,
        deter_dim: int,
        embed_dim: int,
        stoch_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        unimix: float = 0.01,
    ) -> None:
        super().__init__()
        self.stoch_dim = stoch_dim
        self.num_classes = num_classes
        self.unimix = unimix

        in_dim = deter_dim + embed_dim
        out_dim = stoch_dim * num_classes
        self.mlp = _make_mlp(in_dim, hidden_dim, out_dim, num_layers)

    def forward(self, h: Tensor, embed: Tensor) -> Tensor:
        """
        Compute posterior logits.

        Args:
            h:     Deterministic state, shape (batch, deter_dim).
            embed: Observation embedding, shape (batch, embed_dim).

        Returns:
            logits: Shape (batch, stoch_dim, num_classes).
        """
        batch = h.shape[0]
        inp = torch.cat([h, embed], dim=-1)  # (batch, deter_dim + embed_dim)
        logits_flat = self.mlp(inp)          # (batch, stoch_dim * num_classes)
        return logits_flat.view(batch, self.stoch_dim, self.num_classes)

    def probs(self, h: Tensor, embed: Tensor) -> Tensor:
        """Return unimix probabilities. Shape: (batch, stoch_dim, num_classes)."""
        return unimix_probs(self.forward(h, embed), unimix=self.unimix)

    def sample(self, h: Tensor, embed: Tensor) -> tuple[Tensor, Tensor]:
        """
        Sample z_t from the posterior using straight-through estimator.

        Returns:
            z:      Sample, shape (batch, stoch_dim, num_classes).
            logits: Raw logits, shape (batch, stoch_dim, num_classes).
        """
        logits = self.forward(h, embed)
        z = sample_straight_through(logits, unimix=self.unimix)
        return z, logits


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Categorical State self-tests")
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

    # -----------------------------------------------------------------------
    # Test unimix_probs
    # -----------------------------------------------------------------------
    print("Tests: unimix_probs")

    logits = torch.randn(8, 32, 32, device=device)

    # Probabilities sum to 1
    probs = unimix_probs(logits, unimix=0.01)
    sums = probs.sum(dim=-1)
    check(
        torch.allclose(sums, torch.ones_like(sums), atol=1e-5),
        "probs sum to 1",
        f"max_deviation={( sums - 1.0).abs().max().item():.2e}",
    )

    # No zero probabilities
    check(
        probs.min().item() > 0,
        "no zero probabilities",
        f"min_prob={probs.min().item():.2e}",
    )

    # Minimum probability at least unimix / num_classes
    expected_min = 0.01 / 32
    check(
        probs.min().item() >= expected_min - 1e-7,
        f"min_prob >= {expected_min:.4f}",
        f"actual_min={probs.min().item():.6f}",
    )

    # Extreme logits still have unimix floor
    logits_extreme = torch.zeros(4, 32, 32, device=device)
    logits_extreme[:, :, 0] = 100.0  # push all mass to class 0
    probs_ext = unimix_probs(logits_extreme, unimix=0.01)
    check(
        probs_ext[:, :, 1:].min().item() >= expected_min - 1e-7,
        "extreme logits: non-argmax classes still have floor probability",
        f"min_other={probs_ext[:, :, 1:].min().item():.6f}",
    )

    # -----------------------------------------------------------------------
    # Test sample_straight_through
    # -----------------------------------------------------------------------
    print("\nTests: sample_straight_through")

    logits_st = torch.randn(4, 32, 32, device=device, requires_grad=True)
    z = sample_straight_through(logits_st, unimix=0.01)

    # Shape correct
    check(z.shape == (4, 32, 32), "straight-through shape", str(z.shape))

    # Forward pass gives one-hot-like values (sum per row = 1)
    z_sums = z.sum(dim=-1)
    check(
        torch.allclose(z_sums, torch.ones_like(z_sums), atol=1e-5),
        "straight-through sums to 1 per row",
    )

    # Forward pass: each row has one entry ≈ 1 (one-hot)
    z_max = z.max(dim=-1).values
    check(
        torch.allclose(z_max, torch.ones_like(z_max), atol=1e-5),
        "straight-through max ≈ 1 (one-hot)",
    )

    # Backward pass: gradients flow through logits
    loss = z.sum()
    loss.backward()
    check(logits_st.grad is not None, "straight-through gradient not None")
    check(
        torch.all(torch.isfinite(logits_st.grad)).item(),
        "straight-through gradient finite",
    )
    check(
        not torch.all(logits_st.grad == 0).item(),
        "straight-through gradient non-zero",
    )

    # -----------------------------------------------------------------------
    # Test PriorNet
    # -----------------------------------------------------------------------
    print("\nTests: PriorNet")

    prior = PriorNet(
        deter_dim=64,
        stoch_dim=8,
        num_classes=8,
        hidden_dim=32,
        num_layers=2,
        unimix=0.01,
    ).to(device)

    h = torch.randn(4, 64, device=device)

    # Forward shape
    logits_p = prior(h)
    check(logits_p.shape == (4, 8, 8), "PriorNet logits shape", str(logits_p.shape))

    # Sample shapes
    z_p, logits_ps = prior.sample(h)
    check(z_p.shape == (4, 8, 8), "PriorNet sample shape", str(z_p.shape))
    check(logits_ps.shape == (4, 8, 8), "PriorNet sample logits shape")

    # Probs shape
    probs_p = prior.probs(h)
    check(probs_p.shape == (4, 8, 8), "PriorNet probs shape")
    check(
        torch.allclose(probs_p.sum(dim=-1), torch.ones(4, 8, device=device), atol=1e-5),
        "PriorNet probs sum to 1",
    )

    # Gradient flows into prior parameters
    h_g = torch.randn(4, 64, device=device, requires_grad=True)
    z_g, _ = prior.sample(h_g)
    z_g.sum().backward()
    check(h_g.grad is not None, "PriorNet gradient flows to h")
    for name, param in prior.named_parameters():
        check(
            param.grad is not None,
            f"PriorNet param grad: {name}",
        )
    prior.zero_grad()

    # -----------------------------------------------------------------------
    # Test PosteriorNet
    # -----------------------------------------------------------------------
    print("\nTests: PosteriorNet")

    posterior = PosteriorNet(
        deter_dim=64,
        embed_dim=32,
        stoch_dim=8,
        num_classes=8,
        hidden_dim=32,
        num_layers=2,
        unimix=0.01,
    ).to(device)

    h_post = torch.randn(4, 64, device=device)
    embed = torch.randn(4, 32, device=device)

    # Forward shape
    logits_q = posterior(h_post, embed)
    check(logits_q.shape == (4, 8, 8), "PosteriorNet logits shape", str(logits_q.shape))

    # Sample shapes
    z_q, logits_qs = posterior.sample(h_post, embed)
    check(z_q.shape == (4, 8, 8), "PosteriorNet sample shape")

    # Probs sum to 1
    probs_q = posterior.probs(h_post, embed)
    check(
        torch.allclose(probs_q.sum(dim=-1), torch.ones(4, 8, device=device), atol=1e-5),
        "PosteriorNet probs sum to 1",
    )

    # Gradient flows into posterior parameters
    h_qg = torch.randn(4, 64, device=device, requires_grad=True)
    embed_qg = torch.randn(4, 32, device=device, requires_grad=True)
    z_qg, _ = posterior.sample(h_qg, embed_qg)
    z_qg.sum().backward()
    check(h_qg.grad is not None, "PosteriorNet gradient flows to h")
    check(embed_qg.grad is not None, "PosteriorNet gradient flows to embed")
    posterior.zero_grad()

    # -----------------------------------------------------------------------
    # Test: codebook diversity
    # -----------------------------------------------------------------------
    print("\nTests: Codebook diversity")

    prior_div = PriorNet(deter_dim=64, stoch_dim=8, num_classes=8, hidden_dim=32).to(device)
    prior_div.train(False)
    h_div = torch.randn(64, 64, device=device)  # large batch for diversity
    z_div, _ = prior_div.sample(h_div)
    indices_div = z_div.argmax(dim=-1)  # (64, 8)

    all_diverse = True
    for dist_idx in range(8):
        unique_classes = indices_div[:, dist_idx].unique()
        if len(unique_classes) <= 1:
            all_diverse = False
            break
    check(all_diverse, "codebook: multiple classes used across batch")

    # -----------------------------------------------------------------------
    # Test: batch size 1
    # -----------------------------------------------------------------------
    print("\nTests: Batch size 1")

    prior_b1 = PriorNet(deter_dim=64, stoch_dim=8, num_classes=8, hidden_dim=32).to(device)
    h_b1 = torch.randn(1, 64, device=device)
    z_b1, _ = prior_b1.sample(h_b1)
    check(z_b1.shape == (1, 8, 8), "PriorNet batch_size=1 shape")

    posterior_b1 = PosteriorNet(deter_dim=64, embed_dim=32, stoch_dim=8,
                                 num_classes=8, hidden_dim=32).to(device)
    embed_b1 = torch.randn(1, 32, device=device)
    z_b1_post, _ = posterior_b1.sample(h_b1, embed_b1)
    check(z_b1_post.shape == (1, 8, 8), "PosteriorNet batch_size=1 shape")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} test(s): {', '.join(FAILURES)}")
        sys.exit(1)
    else:
        print("All tests PASSED.")
