"""
Block GRU Template — DreamerV3-Style Modified GRU Cell

Architecture:
  Input projection: Linear(input_dim, hidden_dim) -> LayerNorm(hidden_dim) -> SiLU
  GRU gates:        reset, update, candidate (standard GRU mechanics)
  Output:           RMSNorm(hidden_dim)(h_new)

Use nn.RMSNorm (PyTorch 2.4+). For earlier versions see the manual fallback below.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# RMSNorm fallback for PyTorch < 2.4
# ---------------------------------------------------------------------------

def _make_rmsnorm(dim: int) -> nn.Module:
    """Return nn.RMSNorm if available, otherwise a manual implementation."""
    if hasattr(nn, "RMSNorm"):
        return nn.RMSNorm(dim)
    return _RMSNorm(dim)


class _RMSNorm(nn.Module):
    """Manual RMSNorm for PyTorch versions that lack nn.RMSNorm."""

    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # x: (*, dim)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


# ---------------------------------------------------------------------------
# BlockGRU
# ---------------------------------------------------------------------------

class BlockGRU(nn.Module):
    """
    DreamerV3-style Block GRU cell.

    Differences from standard GRUCell:
    - Input is first projected through Linear -> LayerNorm -> SiLU before gate computation.
    - Output hidden state is normalized via RMSNorm before being returned.

    Args:
        input_dim:  Dimensionality of the input tensor x.
        hidden_dim: Dimensionality of the hidden state. Default 1024.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 1024) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input projection: compress and normalize input before gate computation.
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # GRU gates. Each takes concat(x_proj, h) as input.
        # concat dimension: hidden_dim + hidden_dim = 2 * hidden_dim
        gate_in = hidden_dim * 2

        self.gate_r = nn.Linear(gate_in, hidden_dim, bias=True)  # reset gate
        self.gate_z = nn.Linear(gate_in, hidden_dim, bias=True)  # update gate
        self.gate_n = nn.Linear(gate_in, hidden_dim, bias=True)  # candidate

        # Output normalization.
        self.norm_out = _make_rmsnorm(hidden_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for stable training."""
        for gate in [self.gate_r, self.gate_z, self.gate_n]:
            nn.init.orthogonal_(gate.weight)
            nn.init.zeros_(gate.bias)
        # Input projection Linear
        nn.init.xavier_uniform_(self.input_proj[0].weight)
        nn.init.zeros_(self.input_proj[0].bias)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        """
        Advance the GRU by one step.

        Args:
            x: Input tensor of shape (batch, input_dim).
            h: Previous hidden state of shape (batch, hidden_dim).

        Returns:
            h_new: Updated hidden state of shape (batch, hidden_dim),
                   normalized via RMSNorm.
        """
        # 1. Project and normalize input.
        x_proj = self.input_proj(x)          # (batch, hidden_dim)

        # 2. Concatenate projected input with previous hidden for gate inputs.
        xh = torch.cat([x_proj, h], dim=-1)  # (batch, 2 * hidden_dim)

        # 3. Compute reset and update gates.
        r = torch.sigmoid(self.gate_r(xh))   # (batch, hidden_dim)
        z = torch.sigmoid(self.gate_z(xh))   # (batch, hidden_dim)

        # 4. Compute candidate hidden state (reset gate applied to h).
        xrh = torch.cat([x_proj, r * h], dim=-1)  # (batch, 2 * hidden_dim)
        n = torch.tanh(self.gate_n(xrh))           # (batch, hidden_dim)

        # 5. Interpolate between previous and candidate.
        h_new = (1.0 - z) * h + z * n             # (batch, hidden_dim)

        # 6. Normalize output.
        return self.norm_out(h_new)                # (batch, hidden_dim)

    def initial_state(self, batch_size: int, device: torch.device = torch.device("cpu")) -> Tensor:
        """Return all-zeros initial hidden state."""
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}"


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("BlockGRU self-tests")
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

    # --- Test 1: Output shape ---
    print("Test 1: Output shape")
    gru = BlockGRU(input_dim=64, hidden_dim=128).to(device)
    x = torch.randn(4, 64, device=device)
    h = torch.zeros(4, 128, device=device)
    h_new = gru(x, h)
    check(h_new.shape == (4, 128), "shape=(4,128)", str(h_new.shape))

    # --- Test 2: No NaN in output ---
    print("\nTest 2: No NaN in output")
    check(torch.all(torch.isfinite(h_new)).item(), "output finite")

    # --- Test 3: Batch size 1 ---
    print("\nTest 3: Batch size 1")
    x1 = torch.randn(1, 64, device=device)
    h1 = torch.zeros(1, 128, device=device)
    h1_new = gru(x1, h1)
    check(h1_new.shape == (1, 128), "batch_size=1 shape")
    check(torch.all(torch.isfinite(h1_new)).item(), "batch_size=1 finite")

    # --- Test 4: Large batch ---
    print("\nTest 4: Large batch")
    x_big = torch.randn(256, 64, device=device)
    h_big = torch.zeros(256, 128, device=device)
    h_big_new = gru(x_big, h_big)
    check(h_big_new.shape == (256, 128), "large batch shape")

    # --- Test 5: Different hidden dims ---
    print("\nTest 5: Different hidden dims")
    for hdim in [256, 512, 1024]:
        gru_h = BlockGRU(input_dim=64, hidden_dim=hdim).to(device)
        x_h = torch.randn(4, 64, device=device)
        h_h = torch.zeros(4, hdim, device=device)
        out = gru_h(x_h, h_h)
        check(out.shape == (4, hdim), f"hidden_dim={hdim} shape")

    # --- Test 6: Gradient flow through x ---
    print("\nTest 6: Gradient flow through x")
    gru2 = BlockGRU(input_dim=64, hidden_dim=128).to(device)
    x_g = torch.randn(4, 64, device=device, requires_grad=True)
    h_g = torch.zeros(4, 128, device=device, requires_grad=True)
    h_g_new = gru2(x_g, h_g)
    loss = h_g_new.sum()
    loss.backward()
    check(x_g.grad is not None, "x.grad not None")
    check(h_g.grad is not None, "h.grad not None")
    check(torch.all(torch.isfinite(x_g.grad)).item(), "x.grad finite")
    check(torch.all(torch.isfinite(h_g.grad)).item(), "h.grad finite")

    # --- Test 7: Gate parameters receive gradients ---
    print("\nTest 7: Gate parameters receive gradients")
    gru3 = BlockGRU(input_dim=64, hidden_dim=128).to(device)
    x3 = torch.randn(4, 64, device=device)
    h3 = torch.zeros(4, 128, device=device)
    out3 = gru3(x3, h3)
    out3.sum().backward()
    check(gru3.gate_r.weight.grad is not None, "gate_r.weight.grad")
    check(gru3.gate_z.weight.grad is not None, "gate_z.weight.grad")
    check(gru3.gate_n.weight.grad is not None, "gate_n.weight.grad")

    # --- Test 8: Different inputs produce different outputs ---
    print("\nTest 8: Input sensitivity")
    gru4 = BlockGRU(input_dim=64, hidden_dim=128).to(device)
    gru4.train(False)
    h_fixed = torch.zeros(2, 128, device=device)
    xa = torch.randn(2, 64, device=device)
    xb = torch.randn(2, 64, device=device)
    ha = gru4(xa, h_fixed)
    hb = gru4(xb, h_fixed)
    check(not torch.allclose(ha, hb), "different inputs → different outputs")

    # --- Test 9: Sequential steps are consistent ---
    print("\nTest 9: Sequential steps")
    gru5 = BlockGRU(input_dim=16, hidden_dim=32).to(device)
    gru5.train(False)
    h_seq = torch.zeros(1, 32, device=device)
    for step in range(5):
        x_step = torch.randn(1, 16, device=device)
        h_seq = gru5(x_step, h_seq)
    check(h_seq.shape == (1, 32), "sequential steps output shape")
    check(torch.all(torch.isfinite(h_seq)).item(), "sequential steps finite")

    # --- Test 10: Input projection has LayerNorm and SiLU ---
    print("\nTest 10: Input projection structure")
    gru6 = BlockGRU(input_dim=64, hidden_dim=128)
    proj_types = [type(m).__name__ for m in gru6.input_proj]
    check("Linear" in proj_types, "input_proj contains Linear")
    check("LayerNorm" in proj_types, "input_proj contains LayerNorm")
    check("SiLU" in proj_types, "input_proj contains SiLU")

    # --- Test 11: RMSNorm on output ---
    print("\nTest 11: RMSNorm output normalization")
    gru7 = BlockGRU(input_dim=64, hidden_dim=128).to(device)
    gru7.train(False)
    x7 = torch.randn(32, 64, device=device) * 100  # large scale input
    h7 = torch.zeros(32, 128, device=device)
    h7_new = gru7(x7, h7)
    # With RMSNorm scale=1, the RMS of the output should be close to 1
    rms = (h7_new.pow(2).mean(dim=-1)).sqrt()
    check(
        (rms - 1.0).abs().mean().item() < 0.5,
        "RMSNorm output has bounded RMS",
        f"mean_rms={rms.mean().item():.3f}",
    )

    # --- Test 12: All-zero input ---
    print("\nTest 12: All-zero inputs")
    gru8 = BlockGRU(input_dim=64, hidden_dim=128).to(device)
    gru8.train(False)
    x_zero = torch.zeros(4, 64, device=device)
    h_zero = torch.zeros(4, 128, device=device)
    h_zero_out = gru8(x_zero, h_zero)
    check(torch.all(torch.isfinite(h_zero_out)).item(), "zero input → finite output")

    # --- Test 13: initial_state helper ---
    print("\nTest 13: initial_state helper")
    gru9 = BlockGRU(input_dim=16, hidden_dim=64)
    init_h = gru9.initial_state(batch_size=8)
    check(init_h.shape == (8, 64), "initial_state shape")
    check(torch.all(init_h == 0).item(), "initial_state is zeros")

    # Summary
    print("\n" + "=" * 60)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} test(s): {', '.join(FAILURES)}")
        sys.exit(1)
    else:
        print(f"All tests PASSED.")
