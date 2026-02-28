"""
brain_ai/inference/memory.py — Memory Optimizer for Inference

This module provides the MemoryOptimizer for reducing GPU/CPU memory footprint
during inference. Supports inference mode, dtype conversion, selective CPU
offloading, gradient checkpointing, and memory measurement.

Key classes:
    MemoryReport     — Structured memory measurement report
    MemoryOptimizer  — Main class: enable_inference_mode(), offload_to_cpu(), measure_memory()

Design principles:
    1. inference_mode is the default — eliminates autograd overhead (30-50% memory).
    2. Dtype conversion is selective — sensitive modules (HTM, active inference) stay FP32.
    3. CPU offloading is module-level — only specified modules are moved.
    4. Memory measurement is non-intrusive — uses torch memory stats.
    5. Recommendations are generated automatically based on model size and budget.

References:
    references/memory-optimization.md — Full memory optimization rationale
    SKILL.md § MemoryOptimizer contract
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


# ===========================================================================
# SECTION 1: MemoryReport
# ===========================================================================

@dataclass
class MemoryReport:
    """Report produced by MemoryOptimizer.measure_memory().

    Attributes:
        model_size_mb: Total model parameters + buffers in MB.
        param_size_mb: Parameters only in MB.
        buffer_size_mb: Buffers only in MB.
        peak_inference_mb: Peak memory during inference.
        activation_mb: Activation memory (peak - model).
        dtype: Current model dtype string.
        device: Current device string.
        per_module: Per-module memory breakdown (module_name -> MB).
        recommendations: Optimization suggestions.
    """

    model_size_mb: float = 0.0
    param_size_mb: float = 0.0
    buffer_size_mb: float = 0.0
    peak_inference_mb: float = 0.0
    activation_mb: float = 0.0
    dtype: str = "fp32"
    device: str = "cpu"
    per_module: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Model: {self.model_size_mb:.1f}MB ({self.dtype} on {self.device})",
            f"  Params: {self.param_size_mb:.1f}MB  "
            f"Buffers: {self.buffer_size_mb:.1f}MB",
            f"Peak inference: {self.peak_inference_mb:.1f}MB",
            f"Activations: {self.activation_mb:.1f}MB",
        ]
        if self.per_module:
            lines.append("Per-module memory:")
            for name, mb in sorted(
                self.per_module.items(), key=lambda x: x[1], reverse=True
            ):
                lines.append(f"  {name}: {mb:.2f}MB")
        if self.recommendations:
            lines.append("Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")
        return "\n".join(lines)


# ===========================================================================
# SECTION 2: Memory measurement utilities
# ===========================================================================

def measure_model_memory(model: nn.Module) -> Dict[str, float]:
    """Measure memory used by model parameters and buffers.

    Returns:
        Dict with 'param_mb', 'buffer_mb', 'total_mb'.
    """
    param_bytes = sum(
        p.nelement() * p.element_size() for p in model.parameters()
    )
    buffer_bytes = sum(
        b.nelement() * b.element_size() for b in model.buffers()
    )
    return {
        "param_mb": param_bytes / (1024 * 1024),
        "buffer_mb": buffer_bytes / (1024 * 1024),
        "total_mb": (param_bytes + buffer_bytes) / (1024 * 1024),
    }


def per_module_memory(model: nn.Module) -> Dict[str, float]:
    """Measure memory contribution of each top-level module.

    Returns:
        Dict mapping module name to total memory in MB.
    """
    breakdown = {}
    for name, module in model.named_children():
        param_bytes = sum(
            p.nelement() * p.element_size() for p in module.parameters()
        )
        buffer_bytes = sum(
            b.nelement() * b.element_size() for b in module.buffers()
        )
        total_mb = (param_bytes + buffer_bytes) / (1024 * 1024)
        if total_mb > 0:
            breakdown[name] = total_mb
    return breakdown


def get_model_dtype(model: nn.Module) -> str:
    """Detect the primary dtype of a model's parameters."""
    for p in model.parameters():
        if p.dtype == torch.float16:
            return "fp16"
        elif p.dtype == torch.bfloat16:
            return "bf16"
        elif p.dtype == torch.float32:
            return "fp32"
    return "fp32"


def get_model_device(model: nn.Module) -> str:
    """Detect the device of a model's parameters."""
    for p in model.parameters():
        return str(p.device)
    return "cpu"


# ===========================================================================
# SECTION 3: MemoryOptimizer
# ===========================================================================

class MemoryOptimizer:
    """Memory optimizer for inference.

    Provides methods to reduce memory footprint through inference mode,
    dtype conversion, CPU offloading, and memory measurement.

    Args:
        model: The model to optimize.
        device: Target device.
        dtype: Target dtype string ("fp32", "fp16", "bf16").
        mixed_precision_modules: Module names to keep in FP32 when using FP16.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        dtype: str = "fp32",
        mixed_precision_modules: Optional[List[str]] = None,
    ):
        self.model = model
        self.device = device
        self.target_dtype = dtype
        self.mixed_precision_modules = mixed_precision_modules or [
            "htm", "active_inference", "neuromodulation"
        ]
        self._inference_mode_enabled = False
        self._offloaded_modules: Dict[str, str] = {}  # name -> original device

    def enable_inference_mode(self) -> nn.Module:
        """Set model to inference-optimized mode.

        - Sets model.eval() to disable dropout and batch norm updates.
        - Disables gradient computation for all parameters.
        - Returns the model for chaining.

        Note: torch.inference_mode() context should be used at call sites
        around the actual forward pass, not stored on the model.
        """
        self.model.eval()
        self._inference_mode_enabled = True

        # Disable gradient computation for all parameters
        for param in self.model.parameters():
            param.requires_grad_(False)

        return self.model

    def convert_dtype(self, dtype: Optional[str] = None) -> nn.Module:
        """Convert model to the specified dtype.

        Args:
            dtype: Target dtype ("fp32", "fp16", "bf16"). Defaults to self.target_dtype.

        Returns:
            The model (modified in-place).
        """
        dtype = dtype or self.target_dtype
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        if dtype not in dtype_map:
            raise ValueError(f"Invalid dtype: {dtype}")

        torch_dtype = dtype_map[dtype]

        if dtype in ("fp16", "bf16"):
            # Selective conversion: keep sensitive modules in FP32
            for name, module in self.model.named_modules():
                keep_fp32 = any(
                    fp32_name in name for fp32_name in self.mixed_precision_modules
                )
                if not keep_fp32:
                    module.to(dtype=torch_dtype)
        else:
            self.model.to(dtype=torch_dtype)

        self.target_dtype = dtype
        return self.model

    def offload_to_cpu(self, modules: List[str]) -> None:
        """Move specified modules to CPU to free GPU memory.

        Args:
            modules: List of top-level module attribute names to offload.
        """
        for name in modules:
            if hasattr(self.model, name):
                module = getattr(self.model, name)
                if module is not None:
                    # Record original device
                    for p in module.parameters():
                        self._offloaded_modules[name] = str(p.device)
                        break
                    else:
                        self._offloaded_modules[name] = "cpu"
                    module.cpu()
                    logger.info(f"Offloaded module '{name}' to CPU")
            else:
                logger.warning(f"Module '{name}' not found on model")

    def load_to_device(self, module_name: str, device: Optional[str] = None) -> None:
        """Load an offloaded module back to the target device.

        Args:
            module_name: Name of the module to load.
            device: Target device (defaults to self.device).
        """
        device = device or self.device
        if hasattr(self.model, module_name):
            module = getattr(self.model, module_name)
            if module is not None:
                module.to(device)

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing on supported modules.

        This is primarily useful for training but can reduce memory
        during inference for very long sequences.
        """
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        else:
            logger.info(
                "Model does not support gradient_checkpointing_enable(). "
                "Skipping."
            )

    def measure_memory(
        self, sample_input: Optional[Dict[str, Tensor]] = None
    ) -> MemoryReport:
        """Measure model memory and optionally profile inference memory.

        Args:
            sample_input: If provided, run a forward pass to measure
                peak inference memory.

        Returns:
            MemoryReport with detailed breakdown.
        """
        mem = measure_model_memory(self.model)
        per_mod = per_module_memory(self.model)
        model_dtype = get_model_dtype(self.model)
        model_device = get_model_device(self.model)

        peak_mb = 0.0
        activation_mb = 0.0

        if sample_input is not None and torch.cuda.is_available():
            device = model_device if "cuda" in model_device else "cuda"
            try:
                self.model.to(device)
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.empty_cache()

                before = torch.cuda.memory_allocated(device) / (1024 * 1024)

                device_input = {
                    k: v.to(device) for k, v in sample_input.items()
                }
                with torch.inference_mode():
                    _ = self.model(device_input)

                peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
                activation_mb = peak_mb - before
            except Exception as e:
                logger.warning(f"Could not measure GPU memory: {e}")
        elif sample_input is not None:
            # CPU-only: estimate activation memory via model size
            peak_mb = mem["total_mb"] * 1.5  # rough estimate
            activation_mb = mem["total_mb"] * 0.5

        # Generate recommendations
        recommendations = self._generate_recommendations(mem, model_dtype)

        return MemoryReport(
            model_size_mb=mem["total_mb"],
            param_size_mb=mem["param_mb"],
            buffer_size_mb=mem["buffer_mb"],
            peak_inference_mb=peak_mb,
            activation_mb=activation_mb,
            dtype=model_dtype,
            device=model_device,
            per_module=per_mod,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self, mem: Dict[str, float], dtype: str
    ) -> List[str]:
        """Generate optimization recommendations based on model state."""
        recs = []

        if dtype == "fp32" and mem["total_mb"] > 100:
            recs.append(
                "Consider converting to FP16 or BF16 to halve memory usage"
            )

        if mem["total_mb"] > 4000:
            recs.append(
                "Model exceeds 4GB -- consider CPU offloading for "
                "infrequently-used modules"
            )

        if not self._inference_mode_enabled:
            recs.append(
                "Call enable_inference_mode() to disable gradients "
                "and reduce memory"
            )

        if mem["total_mb"] > 10000 and dtype == "fp32":
            recs.append(
                "Model exceeds 10GB in FP32 -- strongly recommend "
                "reduced precision"
            )

        return recs

    def auto_optimize(self, gpu_memory_mb: float = 0.0) -> List[str]:
        """Automatically apply memory optimizations.

        Args:
            gpu_memory_mb: Available GPU memory budget. 0 = auto-detect.

        Returns:
            List of optimizations applied.
        """
        applied = []

        # Step 1: Enable inference mode
        self.enable_inference_mode()
        applied.append("Enabled inference mode (no grad + eval)")

        # Step 2: Detect GPU memory if auto
        if gpu_memory_mb <= 0 and torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.get_device_properties(0).total_mem / (1024 * 1024)
        elif gpu_memory_mb <= 0:
            gpu_memory_mb = float('inf')

        mem = measure_model_memory(self.model)
        model_mb = mem["total_mb"]

        # Step 3: Convert dtype if needed
        if model_mb > gpu_memory_mb * 0.5 and self.target_dtype == "fp32":
            if torch.cuda.is_available():
                cap = torch.cuda.get_device_capability()
                if cap >= (8, 0):
                    self.convert_dtype("bf16")
                    applied.append("Converted to BF16")
                else:
                    self.convert_dtype("fp16")
                    applied.append("Converted to FP16 (selective)")
            else:
                # On CPU, FP16 is slower; keep FP32
                pass

        return applied


# ===========================================================================
# SECTION 4: DictModel for testing
# ===========================================================================

class DictModel(nn.Module):
    """Wrapper that makes a model accept dict inputs."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, inputs):
        if isinstance(inputs, dict):
            x = next(iter(inputs.values()))
        else:
            x = inputs
        return self.inner(x)


# ===========================================================================
# SECTION 5: Self-tests
# ===========================================================================

def _run_self_tests():
    """Run self-tests for memory optimizer."""
    import traceback

    passed = 0
    failed = 0
    test_results = []

    def _test(name, fn):
        nonlocal passed, failed
        try:
            fn()
            passed += 1
            test_results.append(f"  PASS: {name}")
        except Exception as e:
            failed += 1
            test_results.append(f"  FAIL: {name} -- {e}")
            traceback.print_exc()

    def make_model(in_f=256, out_f=10):
        return DictModel(nn.Sequential(
            nn.Linear(in_f, 512),
            nn.ReLU(),
            nn.Linear(512, out_f),
        ))

    def make_compound_model():
        """Model with named submodules for offloading tests."""
        model = nn.Module()
        model.encoder = nn.Linear(64, 128)
        model.decoder = nn.Linear(128, 32)
        model.head = nn.Linear(32, 10)
        return model

    # --- measure_model_memory tests ---
    def test_measure_model_memory():
        model = nn.Linear(1000, 1000)
        mem = measure_model_memory(model)
        # 1000*1000*4 bytes (weights) + 1000*4 bytes (bias) ~= 3.8MB
        assert mem["param_mb"] > 3.0
        assert mem["param_mb"] < 5.0
        assert mem["total_mb"] > 0

    def test_measure_model_memory_small():
        model = nn.Linear(10, 10)
        mem = measure_model_memory(model)
        assert mem["param_mb"] > 0
        assert mem["total_mb"] == mem["param_mb"] + mem["buffer_mb"]

    def test_per_module_memory():
        model = make_compound_model()
        breakdown = per_module_memory(model)
        assert "encoder" in breakdown
        assert "decoder" in breakdown
        assert "head" in breakdown
        assert all(v > 0 for v in breakdown.values())

    def test_get_model_dtype_fp32():
        model = nn.Linear(10, 10)
        assert get_model_dtype(model) == "fp32"

    def test_get_model_dtype_fp16():
        model = nn.Linear(10, 10).half()
        assert get_model_dtype(model) == "fp16"

    def test_get_model_device_cpu():
        model = nn.Linear(10, 10)
        assert "cpu" in get_model_device(model)

    # --- MemoryOptimizer tests ---
    def test_enable_inference_mode():
        model = make_model()
        opt = MemoryOptimizer(model)
        result = opt.enable_inference_mode()
        assert result.training is False
        for p in model.parameters():
            assert p.requires_grad is False

    def test_enable_inference_mode_returns_model():
        model = make_model()
        opt = MemoryOptimizer(model)
        returned = opt.enable_inference_mode()
        assert returned is model

    def test_convert_dtype_fp16():
        model = make_model()
        opt = MemoryOptimizer(model, dtype="fp16", mixed_precision_modules=[])
        opt.convert_dtype("fp16")
        for p in model.parameters():
            assert p.dtype == torch.float16

    def test_convert_dtype_fp32():
        model = make_model()
        opt = MemoryOptimizer(model)
        opt.convert_dtype("fp32")
        for p in model.parameters():
            assert p.dtype == torch.float32

    def test_convert_dtype_invalid():
        model = make_model()
        opt = MemoryOptimizer(model)
        try:
            opt.convert_dtype("int4")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_fp16_inference_produces_finite():
        model = make_model(in_f=32, out_f=5)
        opt = MemoryOptimizer(model, mixed_precision_modules=[])
        opt.convert_dtype("fp16")
        inp = {"features": torch.randn(1, 32).half()}
        with torch.inference_mode():
            out = model(inp)
        assert torch.isfinite(out).all()

    def test_offload_to_cpu():
        model = make_compound_model()
        opt = MemoryOptimizer(model)
        opt.offload_to_cpu(["encoder"])
        for p in model.encoder.parameters():
            assert str(p.device) == "cpu"

    def test_offload_nonexistent_module():
        model = make_compound_model()
        opt = MemoryOptimizer(model)
        # Should not raise, just log a warning
        opt.offload_to_cpu(["nonexistent_module"])

    def test_offload_preserves_others():
        model = make_compound_model()
        opt = MemoryOptimizer(model, device="cpu")
        opt.offload_to_cpu(["encoder"])
        for p in model.decoder.parameters():
            assert str(p.device) == "cpu"

    def test_load_to_device():
        model = make_compound_model()
        opt = MemoryOptimizer(model, device="cpu")
        opt.offload_to_cpu(["encoder"])
        opt.load_to_device("encoder", "cpu")
        for p in model.encoder.parameters():
            assert str(p.device) == "cpu"

    def test_measure_memory_basic():
        model = make_model(in_f=64, out_f=10)
        opt = MemoryOptimizer(model)
        report = opt.measure_memory()
        assert report.model_size_mb > 0
        assert report.param_size_mb > 0
        assert report.dtype == "fp32"
        assert "cpu" in report.device

    def test_measure_memory_with_input():
        model = make_model(in_f=64, out_f=10)
        opt = MemoryOptimizer(model)
        sample = {"features": torch.randn(1, 64)}
        report = opt.measure_memory(sample_input=sample)
        assert report.model_size_mb > 0
        assert report.peak_inference_mb >= 0

    def test_memory_report_summary():
        report = MemoryReport(
            model_size_mb=100.0, param_size_mb=90.0, buffer_size_mb=10.0,
            dtype="fp32", device="cpu",
            per_module={"encoder": 50.0, "decoder": 40.0},
            recommendations=["Use FP16"],
        )
        s = report.summary()
        assert "100.0MB" in s
        assert "encoder" in s
        assert "Use FP16" in s

    def test_memory_report_model_size_sum():
        report = MemoryReport(
            model_size_mb=100.0, param_size_mb=90.0, buffer_size_mb=10.0,
        )
        assert abs(report.model_size_mb - report.param_size_mb - report.buffer_size_mb) < 1e-6

    def test_memory_report_nonnegative():
        model = make_model()
        opt = MemoryOptimizer(model)
        report = opt.measure_memory()
        assert report.model_size_mb >= 0
        assert report.param_size_mb >= 0
        assert report.buffer_size_mb >= 0

    def test_recommendations_for_large_fp32():
        model = nn.Linear(10000, 10000)  # ~400MB in FP32
        opt = MemoryOptimizer(model)
        report = opt.measure_memory()
        assert any("FP16" in r or "BF16" in r for r in report.recommendations)

    def test_recommendations_for_no_inference_mode():
        model = nn.Linear(1000, 1000)
        opt = MemoryOptimizer(model)
        report = opt.measure_memory()
        assert any("inference_mode" in r for r in report.recommendations)

    def test_auto_optimize_enables_inference():
        model = make_model()
        opt = MemoryOptimizer(model)
        applied = opt.auto_optimize()
        assert any("inference mode" in a for a in applied)
        assert model.training is False

    def test_mixed_precision_selective():
        """When converting to FP16, modules in mixed_precision_modules stay FP32."""
        model = make_compound_model()
        opt = MemoryOptimizer(model, mixed_precision_modules=["encoder"])
        opt.convert_dtype("fp16")
        for p in model.encoder.parameters():
            assert p.dtype == torch.float32
        for p in model.decoder.parameters():
            assert p.dtype == torch.float16

    def test_gradient_checkpointing():
        model = make_model()
        opt = MemoryOptimizer(model)
        opt.enable_gradient_checkpointing()

    def test_per_module_breakdown_content():
        model = make_compound_model()
        breakdown = per_module_memory(model)
        assert breakdown["encoder"] > 0
        assert breakdown["encoder"] < 1.0

    def test_measure_memory_dtype_field():
        model = nn.Linear(10, 10).half()
        opt = MemoryOptimizer(model)
        report = opt.measure_memory()
        assert report.dtype == "fp16"

    # Run all tests
    tests = [
        ("measure_model_memory large", test_measure_model_memory),
        ("measure_model_memory small", test_measure_model_memory_small),
        ("per_module_memory", test_per_module_memory),
        ("get_model_dtype fp32", test_get_model_dtype_fp32),
        ("get_model_dtype fp16", test_get_model_dtype_fp16),
        ("get_model_device cpu", test_get_model_device_cpu),
        ("enable_inference_mode", test_enable_inference_mode),
        ("enable_inference_mode returns model", test_enable_inference_mode_returns_model),
        ("convert_dtype fp16", test_convert_dtype_fp16),
        ("convert_dtype fp32", test_convert_dtype_fp32),
        ("convert_dtype invalid", test_convert_dtype_invalid),
        ("fp16 inference produces finite", test_fp16_inference_produces_finite),
        ("offload_to_cpu", test_offload_to_cpu),
        ("offload nonexistent module", test_offload_nonexistent_module),
        ("offload preserves others", test_offload_preserves_others),
        ("load_to_device", test_load_to_device),
        ("measure_memory basic", test_measure_memory_basic),
        ("measure_memory with input", test_measure_memory_with_input),
        ("memory report summary", test_memory_report_summary),
        ("memory report model_size sum", test_memory_report_model_size_sum),
        ("memory report nonnegative", test_memory_report_nonnegative),
        ("recommendations for large fp32", test_recommendations_for_large_fp32),
        ("recommendations for no inference mode", test_recommendations_for_no_inference_mode),
        ("auto_optimize enables inference", test_auto_optimize_enables_inference),
        ("mixed precision selective", test_mixed_precision_selective),
        ("gradient checkpointing", test_gradient_checkpointing),
        ("per_module breakdown content", test_per_module_breakdown_content),
        ("measure_memory dtype field", test_measure_memory_dtype_field),
    ]

    print(f"Running {len(tests)} self-tests for memory_optimizer_template...")
    for name, fn in tests:
        _test(name, fn)

    print("\n".join(test_results))
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    if failed == 0:
        print("ALL TESTS PASSED")
    return failed == 0


if __name__ == "__main__":
    _run_self_tests()
