"""
Symlog Twohot Template — DreamerV3 Scale-Invariant Prediction Heads

Provides:
  symlog(x) = sign(x) * ln(|x| + 1)
  symexp(x) = sign(x) * (exp(|x|) - 1)
  SymlogTwohot: encode scalars to 255-bin twohot vectors, decode logits to scalars,
                compute cross-entropy loss.

Used for all prediction heads: reward, continue, observation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Symlog / Symexp
# ---------------------------------------------------------------------------

def symlog(x: Tensor) -> Tensor:
    """
    Symmetric logarithm transform.

    Formula: sign(x) * ln(|x| + 1)

    Properties:
    - Monotone, antisymmetric, identity near origin.
    - Maps (-inf, inf) -> (-inf, inf) with logarithmic compression for large |x|.
    - symlog(0) = 0 exactly.

    Args:
        x: Tensor of any shape.

    Returns:
        Tensor of the same shape.
    """
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x: Tensor) -> Tensor:
    """
    Symmetric exponential transform. Exact inverse of symlog.

    Formula: sign(x) * (exp(|x|) - 1)

    Args:
        x: Tensor of any shape (expected to be in a reasonable range, e.g. [-20, 20]).

    Returns:
        Tensor of the same shape.
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


# ---------------------------------------------------------------------------
# SymlogTwohot
# ---------------------------------------------------------------------------

class SymlogTwohot(nn.Module):
    """
    Scale-invariant distribution for scalar regression.

    Combines symlog compression with two-hot (interpolated one-hot) encoding
    and cross-entropy loss. Eliminates the need for environment-specific
    reward normalization.

    Bin layout:
        255 bin centers uniformly spaced in [low, high] = [-20.0, 20.0].
        These cover symlog values for |x| up to ~485 million.

    Args:
        num_bins: Number of bins. Default 255.
        low:      Minimum bin center in symlog space. Default -20.0.
        high:     Maximum bin center in symlog space. Default 20.0.
    """

    def __init__(
        self,
        num_bins: int = 255,
        low: float = -20.0,
        high: float = 20.0,
    ) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.low = low
        self.high = high

        # Register bin centers as a buffer so they move with the module (.to(device)).
        bin_centers = torch.linspace(low, high, num_bins)
        self.register_buffer("bin_centers", bin_centers)

    def encode(self, x: Tensor) -> Tensor:
        """
        Encode a scalar tensor to a two-hot vector in symlog space.

        For scalar value v:
          1. Apply symlog: v_log = symlog(v)
          2. Clamp to [low, high]
          3. Find the two adjacent bins bracketing v_log
          4. Assign interpolated weights summing to 1

        Args:
            x: Scalar values of shape (*).

        Returns:
            twohot: Shape (*, num_bins). Values >= 0, sums to 1 over last dim.
        """
        # Step 1 & 2: symlog + clamp
        x_log = symlog(x).clamp(self.low, self.high)  # (*)

        # Step 3: find lower bin index
        # Bin spacing
        delta = (self.high - self.low) / (self.num_bins - 1)
        # Continuous position in bin grid [0, num_bins - 1]
        pos = (x_log - self.low) / delta              # (*)
        # Integer lower bin index, clamped so upper bin k+1 always exists
        k = pos.long().clamp(0, self.num_bins - 2)    # (*)

        # Step 4: interpolation weights
        b_k  = self.bin_centers[k]         # lower bin center, shape (*)
        b_k1 = self.bin_centers[k + 1]     # upper bin center, shape (*)
        # Weight for upper bin: proportional to distance from lower bin center
        w_upper = (x_log - b_k) / (b_k1 - b_k + 1e-8)  # (*)
        w_upper = w_upper.clamp(0.0, 1.0)                # numerical safety
        w_lower = 1.0 - w_upper                          # (*)

        # Step 5: scatter into twohot vector
        # Preserve leading dimensions; final dim is num_bins
        target = torch.zeros(*x.shape, self.num_bins, device=x.device, dtype=x.dtype)
        target.scatter_(-1, k.unsqueeze(-1), w_lower.unsqueeze(-1))
        target.scatter_(-1, (k + 1).unsqueeze(-1), w_upper.unsqueeze(-1))
        return target  # (*, num_bins)

    def decode(self, logits: Tensor) -> Tensor:
        """
        Decode predicted logits to a scalar estimate.

        Steps:
          1. softmax(logits) -> probability distribution over bins
          2. Expected bin center: sum(p * bin_centers)
          3. Apply symexp to recover original scale

        Args:
            logits: Shape (*, num_bins). Raw linear head output.

        Returns:
            Scalar estimates of shape (*).
        """
        probs = torch.softmax(logits, dim=-1)               # (*, num_bins)
        v_log = (probs * self.bin_centers).sum(dim=-1)      # (*)
        return symexp(v_log)                                  # (*)

    def loss(self, logits: Tensor, target: Tensor) -> Tensor:
        """
        Compute cross-entropy loss between logits and two-hot encoded target.

        Formula: L = -sum_k(twohot_k * log_softmax(logits)_k)

        This is the categorical cross-entropy where the target is a soft distribution
        (two-hot) rather than a one-hot. log_softmax is used for numerical stability.

        Args:
            logits: Shape (*, num_bins). Raw linear head output.
            target: Shape (*). Scalar regression targets (in original scale).

        Returns:
            Per-element loss of shape (*). Non-negative. Differentiable w.r.t. logits.
        """
        twohot = self.encode(target)                          # (*, num_bins)
        log_probs = F.log_softmax(logits, dim=-1)             # (*, num_bins)
        # Cross-entropy: negative sum over bins
        return -(twohot * log_probs).sum(dim=-1)             # (*)

    def extra_repr(self) -> str:
        return (
            f"num_bins={self.num_bins}, low={self.low}, high={self.high}, "
            f"symlog_coverage=(-{symexp(torch.tensor(self.high)).item():.2e}, "
            f"{symexp(torch.tensor(self.high)).item():.2e})"
        )


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import math

    print("=" * 60)
    print("SymlogTwohot self-tests")
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

    module = SymlogTwohot(num_bins=255, low=-20.0, high=20.0).to(device)
    module.train(False)

    # -----------------------------------------------------------------------
    # Test 1: symlog and symexp are inverses
    # -----------------------------------------------------------------------
    print("Test 1: symlog / symexp invertibility")

    test_values = torch.tensor(
        [-1e6, -1000.0, -100.0, -10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0, 100.0, 1000.0, 1e6],
        device=device,
    )

    recovered = symexp(symlog(test_values))
    for i, (v, r) in enumerate(zip(test_values.tolist(), recovered.tolist())):
        if abs(v) > 1e-8:
            rel_err = abs(r - v) / abs(v)
            ok = rel_err < 1e-5
        else:
            ok = abs(r - v) < 1e-7
        check(ok, f"symexp(symlog({v:.4g})) ≈ {v:.4g}", f"got {r:.6g}")

    # -----------------------------------------------------------------------
    # Test 2: encode produces valid two-hot vectors
    # -----------------------------------------------------------------------
    print("\nTest 2: encode properties")

    vals = torch.tensor([-1e6, -100.0, -1.0, 0.0, 1.0, 100.0, 1e6], device=device)
    twohot = module.encode(vals)

    # Sums to 1
    sums = twohot.sum(dim=-1)
    check(
        torch.allclose(sums, torch.ones_like(sums), atol=1e-5),
        "twohot sums to 1",
        f"max_dev={( sums - 1.0).abs().max().item():.2e}",
    )

    # Non-negative
    check(twohot.min().item() >= 0.0, "twohot non-negative", f"min={twohot.min().item():.4f}")

    # Exactly 2 non-zero entries (unless x lands exactly on a bin center)
    nonzero_count = (twohot > 1e-7).sum(dim=-1)
    check(
        (nonzero_count <= 2).all().item(),
        "twohot has <= 2 nonzero entries",
        f"counts={nonzero_count.tolist()}",
    )

    # Zero maps near center bin
    zero_twohot = module.encode(torch.tensor([0.0], device=device))
    center_bin = (module.num_bins - 1) // 2  # bin 127
    check(
        zero_twohot[0, center_bin].item() > 0.5,
        "encode(0) has nonzero weight at center bin",
        f"center_bin={center_bin}, weight={zero_twohot[0, center_bin].item():.4f}",
    )

    # -----------------------------------------------------------------------
    # Test 3: round-trip decode(encode_as_logits(x)) ≈ x
    # -----------------------------------------------------------------------
    print("\nTest 3: encode/decode round-trip")

    # Create logits from twohot by using log of twohot (with small floor to avoid -inf)
    round_trip_vals = [
        (0.0,    1e-5, 0.0,   "zero"),
        (1.0,    1e-3, 0.0,   "one"),
        (-1.0,   1e-3, 0.0,   "neg_one"),
        (10.0,   5e-2, 0.0,   "ten"),
        (-10.0,  5e-2, 0.0,   "neg_ten"),
        (100.0,  0.0,  5e-3,  "hundred"),
        (-100.0, 0.0,  5e-3,  "neg_hundred"),
        (1e6,    0.0,  0.10,  "million"),   # clamped; lossy
        (-1e6,   0.0,  0.10,  "neg_million"),
    ]

    for val, atol, rtol, label in round_trip_vals:
        x_in = torch.tensor([val], device=device)
        th = module.encode(x_in)  # (1, 255)
        # Use log of twohot as "perfect" logits
        logits_perfect = th.clamp(min=1e-7).log()  # (1, 255)
        decoded = module.decode(logits_perfect)      # (1,)
        got = decoded.item()
        if abs(val) < 1e-8:
            ok = abs(got - val) < atol
            detail = f"got={got:.6f}, atol={atol}"
        elif rtol > 0:
            rel = abs(got - val) / (abs(val) + 1e-8)
            ok = rel < rtol
            detail = f"got={got:.4e}, rel_err={rel:.4f}, rtol={rtol}"
        else:
            ok = abs(got - val) < atol
            detail = f"got={got:.6f}, atol={atol}"
        check(ok, f"round-trip {label}", detail)

    # -----------------------------------------------------------------------
    # Test 4: loss is finite and differentiable
    # -----------------------------------------------------------------------
    print("\nTest 4: loss finite and differentiable")

    targets_loss = torch.tensor(
        [-1e6, -1000.0, -100.0, -1.0, 0.0, 1.0, 100.0, 1000.0, 1e6,
         -0.5, 0.5, 42.0, -42.0, 0.001, -0.001],
        device=device,
    )
    logits_loss = torch.randn(len(targets_loss), 255, device=device, requires_grad=True)
    loss = module.loss(logits_loss, targets_loss)

    check(torch.all(torch.isfinite(loss)).item(), "loss finite for all targets")
    check((loss >= 0).all().item(), "loss non-negative")

    loss.mean().backward()
    check(logits_loss.grad is not None, "logits.grad not None")
    check(
        torch.all(torch.isfinite(logits_loss.grad)).item(),
        "logits.grad finite",
    )

    # -----------------------------------------------------------------------
    # Test 5: loss with batch dimensions
    # -----------------------------------------------------------------------
    print("\nTest 5: batched loss")

    B, T = 4, 16
    logits_bt = torch.randn(B, T, 255, device=device, requires_grad=True)
    targets_bt = torch.randn(B, T, device=device) * 50  # various rewards
    loss_bt = module.loss(logits_bt, targets_bt)
    check(loss_bt.shape == (B, T), "batched loss shape", str(loss_bt.shape))
    check(torch.all(torch.isfinite(loss_bt)).item(), "batched loss finite")
    loss_bt.sum().backward()
    check(torch.all(torch.isfinite(logits_bt.grad)).item(), "batched loss grad finite")

    # -----------------------------------------------------------------------
    # Test 6: extreme values clamped gracefully
    # -----------------------------------------------------------------------
    print("\nTest 6: extreme value clamping")

    extreme = torch.tensor([1e10, -1e10], device=device)
    th_extreme = module.encode(extreme)
    check(torch.all(torch.isfinite(th_extreme)).item(), "extreme values: twohot finite")
    check(
        torch.allclose(th_extreme.sum(dim=-1), torch.ones(2, device=device), atol=1e-5),
        "extreme values: twohot sums to 1",
    )
    # Large positive should land at last bin
    check(
        th_extreme[0, -1].item() > 0.5,
        "very large positive → last bin has weight",
        f"last_bin_weight={th_extreme[0, -1].item():.4f}",
    )
    # Large negative should land at first bin
    check(
        th_extreme[1, 0].item() > 0.5,
        "very large negative → first bin has weight",
        f"first_bin_weight={th_extreme[1, 0].item():.4f}",
    )

    # -----------------------------------------------------------------------
    # Test 7: symlog properties
    # -----------------------------------------------------------------------
    print("\nTest 7: symlog properties")

    # symlog(0) == 0
    check(symlog(torch.tensor(0.0)).item() == 0.0, "symlog(0) = 0")

    # antisymmetry: symlog(-x) = -symlog(x)
    x_test = torch.tensor([0.5, 1.0, 10.0, 100.0])
    check(
        torch.allclose(symlog(-x_test), -symlog(x_test), atol=1e-6),
        "symlog antisymmetric",
    )

    # small x: symlog(x) ≈ x
    x_small = torch.tensor([0.001, 0.01, 0.1])
    approx_ratio = (symlog(x_small) / x_small - 1.0).abs()
    check(
        approx_ratio.max().item() < 0.1,
        "symlog ≈ identity near 0",
        f"max_rel_err={approx_ratio.max().item():.4f}",
    )

    # -----------------------------------------------------------------------
    # Test 8: bin_centers is a registered buffer (moves with module)
    # -----------------------------------------------------------------------
    print("\nTest 8: buffer registration")

    module_test = SymlogTwohot()
    buffer_names = [name for name, _ in module_test.named_buffers()]
    check("bin_centers" in buffer_names, "bin_centers registered as buffer")

    # Verify linspace covers [-20, 20]
    check(
        abs(module_test.bin_centers[0].item() - (-20.0)) < 1e-5,
        "first bin center = -20.0",
        f"actual={module_test.bin_centers[0].item()}",
    )
    check(
        abs(module_test.bin_centers[-1].item() - 20.0) < 1e-5,
        "last bin center = 20.0",
        f"actual={module_test.bin_centers[-1].item()}",
    )
    check(len(module_test.bin_centers) == 255, "255 bin centers")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} test(s): {', '.join(FAILURES)}")
        sys.exit(1)
    else:
        print("All tests PASSED.")
