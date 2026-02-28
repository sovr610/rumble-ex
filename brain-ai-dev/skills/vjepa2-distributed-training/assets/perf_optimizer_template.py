"""
perf_optimizer_template.py
============================
PerformanceOptimizer: applies memory and compute optimizations to
V-JEPA 2 models for distributed training.

Supported optimizations:
  - Activation checkpointing (torch.utils.checkpoint): trades compute for
    memory by recomputing activations during backward pass (~50% memory saving)
  - Mixed precision (GradScaler + bfloat16 autocast): reduces memory and
    improves throughput on modern GPUs
  - torch.compile: JIT compilation for 10-30% throughput improvement
    (PyTorch >= 2.0 only; gracefully falls back on older versions)
  - SDPA (scaled_dot_product_attention): auto-selects Flash Attention or
    memory-efficient attention kernels
  - Memory measurement: reports allocated / reserved / peak CUDA memory

Usage:
    perf = PerformanceOptimizer(model, PerfConfig(
        use_activation_checkpointing=True,
        use_bfloat16=True,
        compile_model=False,
        use_sdpa=True,
    ))
    perf.enable_activation_checkpointing()
    scaler = perf.enable_mixed_precision()
    perf.measure_memory()
"""

from __future__ import annotations

import functools
import logging
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Generator, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PerfConfig:
    """
    Configuration for performance optimizations.

    Attributes:
        use_activation_checkpointing: Recompute activations during backward
            to save ~50% activation memory at ~20% throughput cost.
        use_bfloat16: Use bfloat16 mixed precision (AMP) for forward pass.
            Faster on A100/H100; more numerically stable than float16.
        compile_model: Apply torch.compile() for JIT kernel fusion.
            Requires PyTorch >= 2.0. Provides 10-30% speedup after warmup.
        use_sdpa: Use F.scaled_dot_product_attention for fused attention
            kernels (Flash Attention v2 or memory-efficient attention).
        activation_checkpoint_granularity: Which modules to checkpoint.
            "block" = checkpoint each transformer block (recommended).
            "full" = checkpoint entire model (maximum memory saving).
    """
    use_activation_checkpointing: bool = False
    use_bfloat16: bool = True
    compile_model: bool = False
    use_sdpa: bool = True
    activation_checkpoint_granularity: str = "block"  # "block" or "full"


# ---------------------------------------------------------------------------
# PerformanceOptimizer
# ---------------------------------------------------------------------------

class PerformanceOptimizer:
    """
    Applies memory and compute optimizations to an nn.Module.

    Each optimization is independently togglable and safe to combine.
    All methods modify the model in-place and/or return required objects
    (e.g. GradScaler for mixed precision).

    Args:
        model: The nn.Module to optimize.
        config: PerfConfig specifying which optimizations to enable.
    """

    def __init__(self, model: nn.Module, config: Optional[PerfConfig] = None) -> None:
        self.model = model
        self.config = config or PerfConfig()
        self._compiled_model: Optional[nn.Module] = None
        self._scaler: Optional[GradScaler] = None

    # ------------------------------------------------------------------
    # Activation checkpointing
    # ------------------------------------------------------------------

    def enable_activation_checkpointing(self) -> None:
        """
        Enable gradient checkpointing to trade compute for memory.

        Wraps the forward pass (or each transformer block) with
        torch.utils.checkpoint.checkpoint, which discards intermediate
        activations and recomputes them during the backward pass.

        Memory saving: ~50% reduction in activation memory.
        Throughput cost: ~20% slower due to recomputation.

        Modifies model in-place by wrapping the forward method.
        """
        granularity = self.config.activation_checkpoint_granularity

        if granularity == "full":
            self._wrap_full_model_with_checkpoint()
        else:
            # Default: wrap individual transformer blocks
            self._wrap_blocks_with_checkpoint()

        logger.info(
            "Activation checkpointing enabled (granularity=%s)", granularity
        )

    def _wrap_full_model_with_checkpoint(self) -> None:
        """Wrap entire model forward with torch.utils.checkpoint."""
        original_forward = self.model.forward

        @functools.wraps(original_forward)
        def checkpointed_forward(*args: Any, **kwargs: Any) -> Any:
            # use_reentrant=False is the modern recommended setting
            return torch.utils.checkpoint.checkpoint(
                original_forward, *args, use_reentrant=False, **kwargs
            )

        self.model.forward = checkpointed_forward  # type: ignore[method-assign]

    def _wrap_blocks_with_checkpoint(self) -> None:
        """
        Wrap each transformer block (or Sequential child) with checkpoint.

        Heuristic: wraps children that are named 'blocks', 'layers',
        or 'encoder_layers', or any Sequential container at the top level.
        Falls back to full-model wrapping if no blocks found.
        """
        block_attrs = ["blocks", "layers", "encoder_layers", "transformer"]
        found = False

        for attr in block_attrs:
            container = getattr(self.model, attr, None)
            if container is None:
                continue
            if not isinstance(container, (nn.ModuleList, nn.Sequential)):
                continue

            for i, block in enumerate(container):
                original_forward = block.forward

                @functools.wraps(original_forward)
                def make_checkpointed(orig_fwd: Any) -> Any:
                    @functools.wraps(orig_fwd)
                    def ckpt_fwd(*args: Any, **kwargs: Any) -> Any:
                        return torch.utils.checkpoint.checkpoint(
                            orig_fwd, *args, use_reentrant=False, **kwargs
                        )
                    return ckpt_fwd

                block.forward = make_checkpointed(original_forward)  # type: ignore
            found = True
            logger.debug("Wrapped %d blocks in '%s'", len(container), attr)

        if not found:
            logger.debug("No transformer blocks found, falling back to full-model checkpoint")
            self._wrap_full_model_with_checkpoint()

    # ------------------------------------------------------------------
    # Mixed precision
    # ------------------------------------------------------------------

    def enable_mixed_precision(self) -> GradScaler:
        """
        Create and return a GradScaler for AMP training.

        The GradScaler works alongside torch.autocast to:
        1. Scale the loss to prevent underflow in float16/bfloat16
        2. Unscale gradients before the optimizer step
        3. Skip the optimizer step if NaN/Inf gradients are detected

        Note: bfloat16 rarely needs scaling (it has the same exponent range
        as float32) but GradScaler is still recommended for compatibility.

        Returns:
            Configured GradScaler instance.

        Usage:
            scaler = perf.enable_mixed_precision()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(x)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        """
        if not self.config.use_bfloat16:
            logger.warning(
                "enable_mixed_precision called but use_bfloat16=False in config. "
                "Returning GradScaler anyway — caller should decide whether to use it."
            )

        # For bfloat16: enabled=False is safe (no scaling needed) but True works too
        # For float16: enabled=True is required to prevent gradient underflow
        dtype = torch.bfloat16 if self.config.use_bfloat16 else torch.float16
        # bfloat16 has float32-level exponent range: scaling less critical
        enabled = dtype == torch.float16

        self._scaler = GradScaler(enabled=enabled)
        logger.info("GradScaler created (enabled=%s, dtype=%s)", enabled, dtype)
        return self._scaler

    @contextmanager
    def autocast_context(self) -> Generator[None, None, None]:
        """
        Context manager for automatic mixed precision.

        Usage:
            with perf.autocast_context():
                output = model(input)
        """
        if self.config.use_bfloat16 and torch.cuda.is_available():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                yield
        else:
            yield

    # ------------------------------------------------------------------
    # torch.compile
    # ------------------------------------------------------------------

    def compile_model(self, mode: str = "default") -> nn.Module:
        """
        Compile the model with torch.compile for kernel fusion.

        Requires PyTorch >= 2.0. Falls back gracefully to the original
        model on older PyTorch versions.

        Args:
            mode: Compilation mode:
                "default"      — Balance between compilation time and runtime speed.
                "reduce-overhead" — Minimize overhead for small batches.
                "max-autotune"    — Maximize runtime speed (longer compile time).

        Returns:
            Compiled nn.Module (or original model if compile unavailable).
        """
        if not hasattr(torch, "compile"):
            logger.warning(
                "torch.compile not available (requires PyTorch >= 2.0). "
                "Returning original model unchanged."
            )
            warnings.warn(
                "torch.compile requires PyTorch >= 2.0. "
                "Using original (uncompiled) model.",
                UserWarning,
                stacklevel=2,
            )
            return self.model

        try:
            self._compiled_model = torch.compile(self.model, mode=mode)
            logger.info("Model compiled with torch.compile(mode=%r)", mode)
            return self._compiled_model
        except Exception as e:
            logger.warning("torch.compile failed: %s. Using original model.", e)
            return self.model

    # ------------------------------------------------------------------
    # SDPA
    # ------------------------------------------------------------------

    @staticmethod
    def sdpa_is_available() -> bool:
        """Return True if F.scaled_dot_product_attention is available."""
        try:
            from torch.nn.functional import scaled_dot_product_attention  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def apply_sdpa(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Apply scaled dot-product attention using PyTorch's fused kernel.

        Automatically selects Flash Attention, memory-efficient attention,
        or math attention based on hardware and input shape.

        Args:
            query: [B, H, S, D] query tensor.
            key:   [B, H, S, D] key tensor.
            value: [B, H, S, D] value tensor.
            attn_mask: Optional attention mask.
            dropout_p: Dropout probability (0.0 during inference).
            is_causal: Whether to apply causal masking.

        Returns:
            [B, H, S, D] attention output.
        """
        import torch.nn.functional as F
        return F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

    # ------------------------------------------------------------------
    # Memory measurement
    # ------------------------------------------------------------------

    def measure_memory(self) -> Dict[str, float]:
        """
        Measure current CUDA memory usage in megabytes.

        Returns a dict with keys:
          - "allocated_mb": Currently allocated memory.
          - "reserved_mb": Memory reserved by the allocator (may exceed allocated).
          - "peak_mb": Peak allocated memory since last reset.

        Returns 0.0 for all values if CUDA is not available.
        """
        if not torch.cuda.is_available():
            logger.debug("CUDA not available — returning zero memory stats")
            return {
                "allocated_mb": 0.0,
                "reserved_mb": 0.0,
                "peak_mb": 0.0,
            }

        to_mb = 1.0 / (1024 ** 2)
        device = torch.cuda.current_device()

        return {
            "allocated_mb": torch.cuda.memory_allocated(device) * to_mb,
            "reserved_mb": torch.cuda.memory_reserved(device) * to_mb,
            "peak_mb": torch.cuda.max_memory_allocated(device) * to_mb,
        }

    def reset_peak_memory(self) -> None:
        """Reset the peak memory counter for fresh benchmarking."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    # ------------------------------------------------------------------
    # Convenience: apply all enabled optimizations at once
    # ------------------------------------------------------------------

    def apply_all(self) -> GradScaler:
        """
        Apply all optimizations enabled in config.

        Returns:
            GradScaler (always returned for use in training loop).
        """
        if self.config.use_activation_checkpointing:
            self.enable_activation_checkpointing()

        scaler = self.enable_mixed_precision()

        if self.config.compile_model:
            self.compile_model()

        return scaler

    def __repr__(self) -> str:
        return (
            f"PerformanceOptimizer("
            f"model={type(self.model).__name__}, "
            f"use_activation_checkpointing={self.config.use_activation_checkpointing}, "
            f"use_bfloat16={self.config.use_bfloat16}, "
            f"compile={self.config.compile_model})"
        )


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("PerformanceOptimizer self-tests")
    print("=" * 60)

    # Helper: create a simple model with transformer-like blocks
    class TransformerBlock(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.linear = nn.Linear(d, d)
            self.norm = nn.LayerNorm(d)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.norm(self.linear(x) + x)

    class SimpleTransformer(nn.Module):
        def __init__(self, d: int = 32, n_blocks: int = 3) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([TransformerBlock(d) for _ in range(n_blocks)])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for block in self.blocks:
                x = block(x)
            return x

    # ------------------------------------------------------------------
    # Test 1: Activation checkpointing — forward produces same output
    # ------------------------------------------------------------------
    print("\n[Test 1] Activation checkpointing: output correctness")

    torch.manual_seed(42)
    model_ref = SimpleTransformer(d=32, n_blocks=3)
    model_ckpt = SimpleTransformer(d=32, n_blocks=3)
    # Make weights identical
    model_ckpt.load_state_dict(model_ref.state_dict())

    x = torch.randn(2, 8, 32)
    ref_out = model_ref(x)

    perf = PerformanceOptimizer(model_ckpt, PerfConfig(use_activation_checkpointing=True))
    perf.enable_activation_checkpointing()
    ckpt_out = model_ckpt(x)

    assert torch.allclose(ref_out, ckpt_out, atol=1e-5), (
        f"Output mismatch: max diff={( ref_out - ckpt_out).abs().max().item():.2e}"
    )
    print("  Outputs equal with/without checkpointing  PASS")

    # ------------------------------------------------------------------
    # Test 2: Activation checkpointing — backward works
    # ------------------------------------------------------------------
    print("\n[Test 2] Activation checkpointing: backward pass")

    x2 = torch.randn(2, 8, 32, requires_grad=True)
    model_ckpt2 = SimpleTransformer(d=32, n_blocks=2)
    perf2 = PerformanceOptimizer(model_ckpt2, PerfConfig(use_activation_checkpointing=True))
    perf2.enable_activation_checkpointing()

    out = model_ckpt2(x2)
    out.sum().backward()

    for name, param in model_ckpt2.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
    print("  All parameter gradients present after backward  PASS")

    # ------------------------------------------------------------------
    # Test 3: enable_mixed_precision returns GradScaler
    # ------------------------------------------------------------------
    print("\n[Test 3] enable_mixed_precision returns GradScaler")

    model3 = nn.Linear(8, 4)
    perf3 = PerformanceOptimizer(model3, PerfConfig(use_bfloat16=True))
    scaler = perf3.enable_mixed_precision()

    assert isinstance(scaler, GradScaler), f"Expected GradScaler, got {type(scaler)}"
    assert scaler.get_scale() > 0, "GradScaler initial scale should be positive"
    print(f"  GradScaler returned, scale={scaler.get_scale()}  PASS")

    # ------------------------------------------------------------------
    # Test 4: compile_model returns nn.Module
    # ------------------------------------------------------------------
    print("\n[Test 4] compile_model returns nn.Module")

    model4 = nn.Sequential(nn.Linear(8, 8), nn.ReLU())
    perf4 = PerformanceOptimizer(model4, PerfConfig(compile_model=True))
    compiled = perf4.compile_model()

    assert isinstance(compiled, nn.Module), f"Expected nn.Module, got {type(compiled)}"
    print(f"  compile_model returned {type(compiled).__name__}  PASS")

    # ------------------------------------------------------------------
    # Test 5: compile_model output matches uncompiled
    # ------------------------------------------------------------------
    print("\n[Test 5] compile_model output correctness")

    torch.manual_seed(0)
    model5 = nn.Sequential(nn.Linear(16, 16), nn.GELU(), nn.Linear(16, 8))
    x5 = torch.randn(4, 16)
    ref5 = model5(x5)

    perf5 = PerformanceOptimizer(model5, PerfConfig(compile_model=True))
    compiled5 = perf5.compile_model()
    out5 = compiled5(x5)

    # Note: compiled output should be numerically identical on CPU
    assert torch.allclose(ref5, out5, atol=1e-5), (
        f"Compiled output mismatch: max diff={(ref5 - out5).abs().max().item():.2e}"
    )
    print("  Compiled output matches uncompiled  PASS")

    # ------------------------------------------------------------------
    # Test 6: measure_memory returns dict with correct keys and non-negative values
    # ------------------------------------------------------------------
    print("\n[Test 6] measure_memory returns correct structure")

    model6 = nn.Linear(8, 4)
    perf6 = PerformanceOptimizer(model6)
    mem = perf6.measure_memory()

    expected_keys = {"allocated_mb", "reserved_mb", "peak_mb"}
    assert expected_keys <= set(mem.keys()), f"Missing keys in memory dict: {set(mem.keys())}"
    for k, v in mem.items():
        assert v >= 0.0, f"Memory value {k}={v} should be non-negative"
    print(f"  Memory stats: {mem}  PASS")

    # ------------------------------------------------------------------
    # Test 7: autocast_context (CPU no-op)
    # ------------------------------------------------------------------
    print("\n[Test 7] autocast_context (CPU: no-op)")

    model7 = nn.Linear(4, 2)
    perf7 = PerformanceOptimizer(model7, PerfConfig(use_bfloat16=True))
    x7 = torch.randn(3, 4)

    try:
        with perf7.autocast_context():
            out7 = model7(x7)
        assert out7.shape == (3, 2)
        print("  autocast_context runs without error (CPU)  PASS")
    except Exception as e:
        print(f"  FAIL: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Test 8: apply_all returns GradScaler
    # ------------------------------------------------------------------
    print("\n[Test 8] apply_all returns GradScaler")

    model8 = SimpleTransformer(d=16, n_blocks=2)
    perf8 = PerformanceOptimizer(model8, PerfConfig(
        use_activation_checkpointing=True,
        use_bfloat16=True,
        compile_model=False,
    ))
    scaler8 = perf8.apply_all()
    assert isinstance(scaler8, GradScaler)
    print("  apply_all() returned GradScaler  PASS")

    # ------------------------------------------------------------------
    # Test 9: SDPA availability check
    # ------------------------------------------------------------------
    print("\n[Test 9] SDPA availability")

    sdpa_avail = PerformanceOptimizer.sdpa_is_available()
    print(f"  sdpa_is_available() = {sdpa_avail}  PASS")

    # ------------------------------------------------------------------
    # Test 10: repr
    # ------------------------------------------------------------------
    print("\n[Test 10] __repr__")

    model10 = nn.Linear(4, 2)
    perf10 = PerformanceOptimizer(model10, PerfConfig(use_bfloat16=True))
    r = repr(perf10)
    assert "Linear" in r and "use_bfloat16=True" in r
    print(f"  repr: {r}  PASS")

    print("\n" + "=" * 60)
    print("All PerformanceOptimizer self-tests PASSED")
    print("=" * 60)
