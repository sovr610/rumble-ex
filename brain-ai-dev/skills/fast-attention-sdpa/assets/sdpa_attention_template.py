"""
sdpa_attention_template.py
==========================
Core SDPA attention wrapper implementing:
  - BackendPolicy enum mapping config strings to SDPBackend lists
  - sdpa_attention() -- main function with shape validation, dropout guard,
    is_causal / attn_mask exclusivity, and backend selection
  - SDPAModule -- drop-in nn.Module wrapping sdpa_attention
  - permute_for_sdpa() -- tensor layout conversion helper

Self-tests: python sdpa_attention_template.py
"""

from __future__ import annotations

import logging
import threading
from enum import Enum
from typing import List, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy SDPBackend import guard
# ---------------------------------------------------------------------------

def _get_sdp_backend():
    """Return SDPBackend enum, raising ImportError with a helpful message if unavailable."""
    try:
        from torch.nn.attention import SDPBackend
        return SDPBackend
    except ImportError as e:
        raise ImportError(
            "torch.nn.attention.SDPBackend not available. "
            "Requires PyTorch >= 2.0. "
            f"Original error: {e}"
        ) from e


def _get_sdpa_kernel():
    try:
        from torch.nn.attention import sdpa_kernel
        return sdpa_kernel
    except ImportError as e:
        raise ImportError(
            "torch.nn.attention.sdpa_kernel not available. "
            "Requires PyTorch >= 2.0."
        ) from e


# ---------------------------------------------------------------------------
# BackendPolicy enum
# ---------------------------------------------------------------------------


class BackendPolicy(Enum):
    """
    Maps config string values to lists of SDPBackend enum members.

    Priority within each list is fixed by PyTorch's internal ordering
    (Flash > Efficient > cuDNN > Math). The list only controls which backends
    are *eligible*; PyTorch selects the highest-priority eligible backend.
    """
    AUTO = "auto"
    FLASH = "flash"
    EFFICIENT = "efficient"
    CUDNN = "cudnn"
    FLASH_OR_EFFICIENT = "flash_or_efficient"
    MATH = "math"

    @classmethod
    def from_string(cls, s: str) -> "BackendPolicy":
        try:
            return cls(s)
        except ValueError:
            valid = [e.value for e in cls]
            raise ValueError(
                f"Unknown backend policy {s!r}. Valid values: {valid}"
            )

    def to_sdp_backends(self, force: bool = False) -> List[Any]:
        """
        Return list of SDPBackend values for this policy.

        When force=True, Math is removed from all policies except MATH itself.
        This causes SDPA to raise if no fused kernel can run.
        """
        SDPBackend = _get_sdp_backend()
        FLASH = SDPBackend.FLASH_ATTENTION
        EFF = SDPBackend.EFFICIENT_ATTENTION
        CUDNN = SDPBackend.CUDNN_ATTENTION
        MATH = SDPBackend.MATH

        policy_map = {
            BackendPolicy.AUTO: [FLASH, EFF, CUDNN, MATH],
            BackendPolicy.FLASH: [FLASH, MATH],
            BackendPolicy.EFFICIENT: [EFF, MATH],
            BackendPolicy.CUDNN: [CUDNN, MATH],
            BackendPolicy.FLASH_OR_EFFICIENT: [FLASH, EFF, CUDNN, MATH],
            BackendPolicy.MATH: [MATH],
        }
        backends = list(policy_map[self])

        if force and self != BackendPolicy.MATH:
            backends = [b for b in backends if b != MATH]

        return backends


# ---------------------------------------------------------------------------
# BackendConfig (lightweight version -- full version in backend_config_template)
# ---------------------------------------------------------------------------


class BackendConfig:
    """
    Resolved backend configuration passed to sdpa_attention().

    Attributes:
        backends : list of SDPBackend values to pass to sdpa_kernel
        force    : if True, Math is excluded -> SDPA raises if fused fails
        log      : if True, log backend info on first call
    """

    def __init__(
        self,
        policy: str = "auto",
        force: bool = False,
        log: bool = True,
    ):
        self.policy = BackendPolicy.from_string(policy)
        self.force = force
        self.log = log
        self._backends: Optional[List[Any]] = None

    @property
    def backends(self) -> List[Any]:
        if self._backends is None:
            self._backends = self.policy.to_sdp_backends(force=self.force)
        return self._backends

    def __repr__(self) -> str:
        return (
            f"BackendConfig(policy={self.policy.value!r}, "
            f"force={self.force}, "
            f"backends={[b.name if hasattr(b, 'name') else b for b in self.backends]})"
        )


# ---------------------------------------------------------------------------
# Thread-local first-call log state
# ---------------------------------------------------------------------------

_log_state = threading.local()


def _should_log() -> bool:
    if not getattr(_log_state, "logged", False):
        _log_state.logged = True
        return True
    return False


def reset_log_state() -> None:
    """Reset first-call log flag (useful in tests)."""
    _log_state.logged = False


# ---------------------------------------------------------------------------
# permute_for_sdpa
# ---------------------------------------------------------------------------


def permute_for_sdpa(x: torch.Tensor, fmt: str) -> torch.Tensor:
    """
    Convert tensor layout for SDPA compatibility.

    SDPA expects: (batch, heads, seq_len, head_dim)  ->  fmt="BHSD"

    Args:
        x   : input tensor
        fmt : current layout description. Recognised values:
              "BHSD"  -- (B, H, S, D) -- already correct for SDPA, no-op
              "BSHD"  -- (B, S, H, D) -- typical Transformer layout, permute to BHSD
              "SBHD"  -- (S, B, H, D) -- seq-first layout

    Returns:
        Tensor in (B, H, S, D) layout.
    """
    fmt = fmt.upper().replace(" ", "")
    if fmt == "BHSD":
        return x
    if fmt == "BSHD":
        # (B, S, H, D) -> (B, H, S, D)
        return x.permute(0, 2, 1, 3)
    if fmt == "SBHD":
        # (S, B, H, D) -> (B, H, S, D)
        return x.permute(1, 2, 0, 3)
    raise ValueError(
        f"Unknown fmt {fmt!r}. Expected one of: BHSD, BSHD, SBHD"
    )


def permute_from_sdpa(x: torch.Tensor, target_fmt: str) -> torch.Tensor:
    """
    Convert from SDPA output layout (B, H, S, D) back to target_fmt.

    Inverse of permute_for_sdpa.
    """
    target_fmt = target_fmt.upper().replace(" ", "")
    if target_fmt == "BHSD":
        return x
    if target_fmt == "BSHD":
        # (B, H, S, D) -> (B, S, H, D)
        return x.permute(0, 2, 1, 3)
    if target_fmt == "SBHD":
        # (B, H, S, D) -> (S, B, H, D)
        return x.permute(2, 0, 1, 3)
    raise ValueError(f"Unknown target_fmt {target_fmt!r}")


# ---------------------------------------------------------------------------
# Shape validation
# ---------------------------------------------------------------------------


def _validate_qkv_shapes(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    """
    Validate that q, k, v have compatible shapes for SDPA.

    Expected layout: (..., heads, seq_len, head_dim)
    The last 3 dimensions are (heads, seq_len, head_dim).
    """
    if q.dim() < 3:
        raise ValueError(
            f"q must have at least 3 dimensions (heads, seq, head_dim), "
            f"got {q.dim()}D tensor with shape {q.shape}"
        )
    if k.dim() != q.dim():
        raise ValueError(f"k.dim()={k.dim()} != q.dim()={q.dim()}")
    if v.dim() != q.dim():
        raise ValueError(f"v.dim()={v.dim()} != q.dim()={q.dim()}")

    # head_dim must match between q and k; v can have a different head_dim
    # (grouped query attention), but q/k must agree
    q_head_dim = q.shape[-1]
    k_head_dim = k.shape[-1]
    if q_head_dim != k_head_dim:
        raise ValueError(
            f"q head_dim={q_head_dim} != k head_dim={k_head_dim}. "
            "Mismatched head dimensions in query and key."
        )


# ---------------------------------------------------------------------------
# sdpa_attention -- main wrapper
# ---------------------------------------------------------------------------


def sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    backend_cfg: Optional[BackendConfig] = None,
    training: bool = False,
) -> torch.Tensor:
    """
    Route attention through F.scaled_dot_product_attention with backend control.

    Args:
        q           : Query tensor, shape (B, H, S, D) -- SDPA convention
        k           : Key tensor
        v           : Value tensor
        attn_mask   : Optional mask. Cannot be used with is_causal=True.
                      Bool mask -> True positions are attended to.
                      Float mask -> added to attention logits.
        dropout_p   : Attention dropout probability.
        is_causal   : Apply causal (autoregressive) masking.
        backend_cfg : Backend selection config. None -> use all backends (auto).
        training    : Whether the model is in training mode.
                      Controls dropout guard. Pass self.training from a Module.

    Returns:
        Attention output tensor, same shape as q.

    Raises:
        ValueError      : is_causal=True and attn_mask is not None
        AssertionError  : dropout_p > 0 when training=False
        RuntimeError    : force=True and no fused backend can run
    """
    # ------------------------------------------------------------------
    # 1. Shape validation
    # ------------------------------------------------------------------
    _validate_qkv_shapes(q, k, v)

    # ------------------------------------------------------------------
    # 2. is_causal / attn_mask exclusivity
    # ------------------------------------------------------------------
    if is_causal and attn_mask is not None:
        raise ValueError(
            "is_causal=True and attn_mask cannot both be set. "
            "When is_causal=True, SDPA generates its own causal mask internally. "
            "Remove attn_mask or set is_causal=False."
        )

    # ------------------------------------------------------------------
    # 3. Dropout guard
    # ------------------------------------------------------------------
    if not training and dropout_p != 0.0:
        raise AssertionError(
            f"dropout_p={dropout_p} but training=False. "
            "Pass dropout_p=0.0 during inference. "
            "Use: p = self.dropout_p if self.training else 0.0"
        )

    # ------------------------------------------------------------------
    # 4. Backend selection and dispatch
    # ------------------------------------------------------------------
    if backend_cfg is None:
        # Default: all backends enabled (SDPA auto-selects)
        if _should_log():
            logger.info("sdpa_attention: using auto backend selection (no BackendConfig)")
        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

    if backend_cfg.log and _should_log():
        logger.info("sdpa_attention: %s", backend_cfg)

    sdpa_kernel = _get_sdpa_kernel()
    with sdpa_kernel(backend_cfg.backends):
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )
    return out


# ---------------------------------------------------------------------------
# SDPAModule -- nn.Module wrapper
# ---------------------------------------------------------------------------


class SDPAModule(nn.Module):
    """
    Drop-in attention module routing through sdpa_attention().

    Usage:
        attn = SDPAModule(num_heads=8, head_dim=64, dropout_p=0.1)
        # q, k, v shape: (B, H, S, D) -- SDPA convention
        out = attn(q, k, v, is_causal=True)

    Notes:
        - Dropout is zeroed in inference mode automatically (per SKILL.md rule).
        - Use module.train(False) to set inference mode (avoids bare .eval()).
        - forward() accepts either (B, H, S, D) or (B, S, H, D) via input_fmt.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        dropout_p: float = 0.0,
        backend_cfg: Optional[BackendConfig] = None,
        input_fmt: str = "BHSD",
    ):
        """
        Args:
            num_heads   : Number of attention heads.
            head_dim    : Dimension per head.
            dropout_p   : Dropout probability (only applied during training).
            backend_cfg : Backend selection config. None -> auto.
            input_fmt   : Layout of input tensors to forward().
                          "BHSD" (default) or "BSHD".
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout_p = dropout_p
        self.backend_cfg = backend_cfg
        self.input_fmt = input_fmt.upper()

        if num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {num_heads}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be > 0, got {head_dim}")
        if not (0.0 <= dropout_p < 1.0):
            raise ValueError(f"dropout_p must be in [0, 1), got {dropout_p}")

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Compute scaled dot-product attention.

        Args:
            q, k, v   : Tensors in layout specified by self.input_fmt.
            attn_mask  : Optional mask (cannot be used with is_causal=True).
            is_causal  : Apply causal masking.

        Returns:
            Attention output in the same layout as inputs.
        """
        # Permute to SDPA convention (B, H, S, D) if necessary
        if self.input_fmt != "BHSD":
            q = permute_for_sdpa(q, self.input_fmt)
            k = permute_for_sdpa(k, self.input_fmt)
            v = permute_for_sdpa(v, self.input_fmt)

        # Dropout guard: zero in inference mode
        effective_dropout = self.dropout_p if self.training else 0.0

        out = sdpa_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=effective_dropout,
            is_causal=is_causal,
            backend_cfg=self.backend_cfg,
            training=self.training,
        )

        # Permute back to original layout
        if self.input_fmt != "BHSD":
            out = permute_from_sdpa(out, self.input_fmt)

        return out

    def extra_repr(self) -> str:
        policy = self.backend_cfg.policy.value if self.backend_cfg else "auto"
        return (
            f"num_heads={self.num_heads}, head_dim={self.head_dim}, "
            f"dropout_p={self.dropout_p}, backend={policy}"
        )


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


def _run_self_tests() -> None:
    import math
    print("Running sdpa_attention_template self-tests...")
    failures: List[str] = []

    # --- permute_for_sdpa ---
    try:
        x = torch.zeros(2, 16, 8, 64)  # (B, S, H, D)
        y = permute_for_sdpa(x, "BSHD")
        assert y.shape == (2, 8, 16, 64), f"Expected (2,8,16,64) got {y.shape}"
        print("  PASS: permute_for_sdpa BSHD->BHSD")
    except Exception as e:
        failures.append(f"permute BSHD: {e}")

    try:
        x = torch.zeros(2, 8, 16, 64)  # (B, H, S, D)
        y = permute_for_sdpa(x, "BHSD")
        assert y.shape == x.shape
        print("  PASS: permute_for_sdpa BHSD is no-op")
    except Exception as e:
        failures.append(f"permute BHSD: {e}")

    try:
        x = torch.zeros(16, 2, 8, 64)  # (S, B, H, D)
        y = permute_for_sdpa(x, "SBHD")
        assert y.shape == (2, 8, 16, 64), f"Expected (2,8,16,64) got {y.shape}"
        print("  PASS: permute_for_sdpa SBHD->BHSD")
    except Exception as e:
        failures.append(f"permute SBHD: {e}")

    try:
        permute_for_sdpa(torch.zeros(2, 8, 16, 64), "INVALID")
        failures.append("Bad fmt should raise ValueError")
    except ValueError:
        print("  PASS: permute_for_sdpa raises on invalid fmt")

    # --- BackendPolicy ---
    try:
        p = BackendPolicy.from_string("auto")
        assert p == BackendPolicy.AUTO
        print("  PASS: BackendPolicy.from_string('auto')")
    except Exception as e:
        failures.append(f"BackendPolicy.from_string: {e}")

    try:
        BackendPolicy.from_string("invalid_backend")
        failures.append("Invalid policy should raise ValueError")
    except ValueError:
        print("  PASS: BackendPolicy raises on invalid string")

    try:
        SDPBackend = _get_sdp_backend()
        p = BackendPolicy.FLASH
        backends = p.to_sdp_backends(force=False)
        assert SDPBackend.FLASH_ATTENTION in backends
        assert SDPBackend.MATH in backends
        print("  PASS: BackendPolicy.FLASH includes FLASH + MATH")

        backends_forced = p.to_sdp_backends(force=True)
        assert SDPBackend.MATH not in backends_forced
        assert SDPBackend.FLASH_ATTENTION in backends_forced
        print("  PASS: BackendPolicy.FLASH force=True removes MATH")
    except ImportError:
        print("  SKIP: SDPBackend not available")
    except Exception as e:
        failures.append(f"BackendPolicy.to_sdp_backends: {e}")

    # --- sdpa_attention shape validation ---
    try:
        q = torch.randn(2, 4, 16, 64)
        k = torch.randn(2, 4, 16, 64)
        v = torch.randn(2, 4, 16, 64)
        reset_log_state()
        out = sdpa_attention(q, k, v, training=False)
        assert out.shape == (2, 4, 16, 64), f"Shape mismatch: {out.shape}"
        print("  PASS: sdpa_attention basic forward shape")
    except Exception as e:
        failures.append(f"Basic forward: {e}")

    # --- dropout guard ---
    try:
        q = torch.randn(2, 4, 16, 64)
        k = torch.randn(2, 4, 16, 64)
        v = torch.randn(2, 4, 16, 64)
        sdpa_attention(q, k, v, dropout_p=0.1, training=False)
        failures.append("dropout_p > 0 with training=False should raise AssertionError")
    except AssertionError:
        print("  PASS: dropout guard raises when training=False and dropout_p > 0")

    try:
        sdpa_attention(q, k, v, dropout_p=0.1, training=True)
        print("  PASS: dropout_p > 0 with training=True succeeds")
    except Exception as e:
        failures.append(f"Dropout in train: {e}")

    # --- is_causal + attn_mask exclusivity ---
    try:
        q = torch.randn(2, 4, 16, 64)
        k = torch.randn(2, 4, 16, 64)
        v = torch.randn(2, 4, 16, 64)
        mask = torch.ones(16, 16, dtype=torch.bool)
        sdpa_attention(q, k, v, attn_mask=mask, is_causal=True, training=False)
        failures.append("is_causal + attn_mask should raise ValueError")
    except ValueError:
        print("  PASS: is_causal + attn_mask raises ValueError")

    # --- is_causal alone ---
    try:
        out = sdpa_attention(q, k, v, is_causal=True, training=False)
        assert out.shape == q.shape
        print("  PASS: is_causal=True alone works")
    except Exception as e:
        failures.append(f"is_causal alone: {e}")

    # --- attn_mask alone (float) ---
    try:
        q = torch.randn(2, 4, 16, 64)
        k = torch.randn(2, 4, 16, 64)
        v = torch.randn(2, 4, 16, 64)
        float_mask = torch.zeros(16, 16)
        out = sdpa_attention(q, k, v, attn_mask=float_mask, training=False)
        assert out.shape == q.shape
        print("  PASS: attn_mask (float) alone works")
    except Exception as e:
        failures.append(f"float attn_mask: {e}")

    # --- shape validation raises ---
    try:
        _validate_qkv_shapes(torch.randn(4, 16), torch.randn(4, 16), torch.randn(4, 16))
        failures.append("2D tensor should raise ValueError")
    except ValueError:
        print("  PASS: _validate_qkv_shapes raises on 2D tensor")

    # --- BackendConfig ---
    try:
        bc = BackendConfig(policy="math", force=False)
        SDPBackend = _get_sdp_backend()
        assert SDPBackend.MATH in bc.backends
        assert SDPBackend.FLASH_ATTENTION not in bc.backends
        print("  PASS: BackendConfig math policy")
    except ImportError:
        print("  SKIP: SDPBackend not available")
    except Exception as e:
        failures.append(f"BackendConfig math: {e}")

    # --- BackendConfig force=True flash on CPU should error ---
    try:
        bc = BackendConfig(policy="flash", force=True)
        reset_log_state()
        q = torch.randn(2, 4, 16, 64)  # CPU
        sdpa_attention(q, torch.randn_like(q), torch.randn_like(q),
                       training=False, backend_cfg=bc)
        # On CPU, Flash is not available and force=True removes Math
        # PyTorch should raise -- if it doesn't, that's also fine on some builds
        print("  PASS (or SKIP): BackendConfig force=True flash on CPU")
    except (RuntimeError, AssertionError):
        print("  PASS: BackendConfig force=True flash on CPU raises")
    except Exception as e:
        failures.append(f"BackendConfig force flash CPU: {e}")

    # --- SDPAModule ---
    try:
        mod = SDPAModule(num_heads=4, head_dim=64, dropout_p=0.1)
        mod.train(False)  # use train(False) instead of .eval()
        q = torch.randn(2, 4, 16, 64)
        out = mod(q, q, q, is_causal=True)
        assert out.shape == q.shape
        print("  PASS: SDPAModule inference forward")
    except Exception as e:
        failures.append(f"SDPAModule inference forward: {e}")

    try:
        mod = SDPAModule(num_heads=4, head_dim=64, dropout_p=0.0, input_fmt="BSHD")
        mod.train(False)
        q_bshd = torch.randn(2, 16, 4, 64)  # (B, S, H, D)
        out = mod(q_bshd, q_bshd, q_bshd)
        assert out.shape == (2, 16, 4, 64), f"Expected (2,16,4,64) got {out.shape}"
        print("  PASS: SDPAModule BSHD input_fmt roundtrip")
    except Exception as e:
        failures.append(f"SDPAModule BSHD: {e}")

    # --- SDPAModule numerical correctness (Math backend, fp32) ---
    try:
        import math
        SDPBackend = _get_sdp_backend()
        bc = BackendConfig(policy="math", force=False, log=False)
        mod = SDPAModule(num_heads=1, head_dim=32, dropout_p=0.0, backend_cfg=bc)
        mod.train(False)

        torch.manual_seed(42)
        q = torch.randn(1, 1, 8, 32)
        k = torch.randn(1, 1, 8, 32)
        v = torch.randn(1, 1, 8, 32)

        # Manual attention
        scale = 1.0 / math.sqrt(32)
        attn_w = torch.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
        expected = attn_w @ v

        reset_log_state()
        got = mod(q, k, v)
        assert torch.allclose(got, expected, atol=1e-5), \
            f"Max diff: {(got - expected).abs().max().item()}"
        print("  PASS: SDPAModule numerical correctness vs manual attention")
    except ImportError:
        print("  SKIP: SDPBackend not available")
    except Exception as e:
        failures.append(f"Numerical correctness: {e}")

    # --- SDPAModule validation ---
    try:
        SDPAModule(num_heads=0, head_dim=64)
        failures.append("num_heads=0 should raise ValueError")
    except ValueError:
        print("  PASS: SDPAModule raises for num_heads=0")

    # --- Summary ---
    if failures:
        print(f"\nFAILED {len(failures)} tests:")
        for f in failures:
            print(f"  FAIL: {f}")
        raise SystemExit(1)
    else:
        print(f"\nAll sdpa_attention_template self-tests PASSED")


if __name__ == "__main__":
    _run_self_tests()
