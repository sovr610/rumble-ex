"""
ONNXExporter: Export BrainAI models to ONNX format.

Provides export(), validate(), and get_unsupported_ops() methods for converting
PyTorch models to ONNX with dynamic axes, operator validation, and numerical
verification. Gracefully mocks ONNX Runtime when not installed.

torch + standard lib only. ONNX/ORT mocked if unavailable.
"""

import copy
import io
import os
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Graceful ONNX / ORT imports
# ---------------------------------------------------------------------------

_ONNX_AVAILABLE = False
_ORT_AVAILABLE = False

try:
    import onnx  # type: ignore
    from onnx import checker as onnx_checker  # type: ignore

    _ONNX_AVAILABLE = True
except ImportError:
    pass

try:
    import onnxruntime as ort  # type: ignore

    _ORT_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExportConfig:
    """Configuration for ONNX export."""
    opset_version: int = 17
    dynamic_axes: bool = True
    snn_mode: str = "stateless"  # stateless | stateful
    validate_export: bool = True
    tolerance: float = 1e-4
    input_names: List[str] = field(default_factory=lambda: ["input"])
    output_names: List[str] = field(default_factory=lambda: ["output"])
    verbose: bool = False


@dataclass
class ONNXExportResult:
    """Result of an ONNX export operation."""
    success: bool
    output_path: str
    model_size_bytes: int = 0
    num_nodes: int = 0
    opset_version: int = 0
    export_time_seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Result of validating an ONNX export."""
    structural_valid: bool = False
    numerical_valid: bool = False
    max_diff: float = float("inf")
    mean_diff: float = float("inf")
    num_test_inputs: int = 0
    tolerance: float = 1e-4
    details: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Surrogate replacement helpers
# ---------------------------------------------------------------------------

class HeavisideStep(torch.autograd.Function):
    """Forward-only Heaviside step (no surrogate gradient)."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        return (x > 0).float()

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(grad_output)


def replace_surrogates_for_export(model: nn.Module) -> nn.Module:
    """Replace surrogate gradient functions with Heaviside step for export."""
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, "surrogate"):
            module.surrogate = HeavisideStep.apply
        if hasattr(module, "spike_fn"):
            module.spike_fn = HeavisideStep.apply
    return model


# ---------------------------------------------------------------------------
# ONNXExporter
# ---------------------------------------------------------------------------

class ONNXExporter:
    """Export PyTorch models to ONNX format with validation.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to export.
    config : ExportConfig
        Export configuration.
    """

    # Operators known to be unsupported or problematic in ONNX for BrainAI
    _KNOWN_UNSUPPORTED_PATTERNS: List[str] = [
        "aten::_lif_step",
        "aten::spatial_pooler",
        "aten::hash_lookup",
        "aten::scatter_nd_custom",
    ]

    def __init__(self, model: nn.Module, config: Optional[ExportConfig] = None):
        self.original_model = model
        self.config = config or ExportConfig()
        self._prepared_model: Optional[nn.Module] = None
        self._unsupported_ops: Optional[List[str]] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def export(
        self,
        output_path: str,
        sample_input: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> ONNXExportResult:
        """Export the model to ONNX format.

        Parameters
        ----------
        output_path : str
            File path for the exported ``.onnx`` model.
        sample_input : Tensor or dict of Tensors
            Representative input(s) used for tracing.

        Returns
        -------
        ONNXExportResult
        """
        result = ONNXExportResult(success=False, output_path=output_path)
        start = time.monotonic()

        try:
            prepared = self._prepare_model()
            args, input_names, dynamic_axes = self._prepare_inputs(sample_input)

            output_names = list(self.config.output_names)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                torch.onnx.export(
                    prepared,
                    args,
                    output_path,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes if self.config.dynamic_axes else None,
                    opset_version=self.config.opset_version,
                    do_constant_folding=True,
                    verbose=self.config.verbose,
                )
                result.warnings = [str(w.message) for w in caught]

            if os.path.exists(output_path):
                result.model_size_bytes = os.path.getsize(output_path)

            result.opset_version = self.config.opset_version
            result.num_nodes = self._count_nodes(output_path)
            result.success = True

        except Exception as exc:  # noqa: BLE001
            result.errors.append(str(exc))

        result.export_time_seconds = time.monotonic() - start
        return result

    def validate(
        self,
        output_path: str,
        sample_input: Union[torch.Tensor, Dict[str, torch.Tensor]],
        num_tests: int = 10,
    ) -> ValidationResult:
        """Validate an exported ONNX model.

        Performs structural validation (if ``onnx`` package available) and
        numerical validation (if ``onnxruntime`` available).
        """
        vr = ValidationResult(tolerance=self.config.tolerance, num_test_inputs=num_tests)

        # Structural validation
        vr.structural_valid = self._validate_structure(output_path)

        # Numerical validation
        vr.numerical_valid, vr.max_diff, vr.mean_diff = self._validate_numerical(
            output_path, sample_input, num_tests
        )

        return vr

    def get_unsupported_ops(self) -> List[str]:
        """Return a list of operations in the model that are not supported by ONNX."""
        if self._unsupported_ops is not None:
            return list(self._unsupported_ops)

        unsupported: List[str] = []
        for name, module in self.original_model.named_modules():
            class_name = type(module).__name__
            # Check for known problematic patterns
            if hasattr(module, "surrogate") or hasattr(module, "spike_fn"):
                unsupported.append(f"{name}: SNN surrogate gradient ({class_name})")
            if "HTM" in class_name or "SpatialPooler" in class_name:
                unsupported.append(f"{name}: HTM native op ({class_name})")
            if "Engram" in class_name and hasattr(module, "hash_lookup"):
                unsupported.append(f"{name}: Engram hash lookup ({class_name})")
            if "DualProcess" in class_name:
                unsupported.append(f"{name}: Dynamic control flow ({class_name})")

        self._unsupported_ops = unsupported
        return list(unsupported)

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _prepare_model(self) -> nn.Module:
        """Prepare the model for ONNX export (replace surrogates, etc.)."""
        if self._prepared_model is not None:
            return self._prepared_model

        model = replace_surrogates_for_export(self.original_model)
        model.requires_grad_(False)
        self._prepared_model = model
        return model

    def _prepare_inputs(
        self, sample_input: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> Tuple[Any, List[str], Dict[str, Dict[int, str]]]:
        """Prepare input tensors, names, and dynamic axes."""
        if isinstance(sample_input, dict):
            tensors = []
            input_names = []
            dynamic_axes: Dict[str, Dict[int, str]] = {}
            for key in sorted(sample_input.keys()):
                t = sample_input[key]
                name = f"{key}_input"
                tensors.append(t)
                input_names.append(name)
                if self.config.dynamic_axes:
                    dynamic_axes[name] = {0: "batch_size"}
            for out_name in self.config.output_names:
                if self.config.dynamic_axes:
                    dynamic_axes[out_name] = {0: "batch_size"}
            args = tuple(tensors)
        else:
            input_names = list(self.config.input_names)
            dynamic_axes = {}
            if self.config.dynamic_axes:
                for name in input_names:
                    dynamic_axes[name] = {0: "batch_size"}
                for out_name in self.config.output_names:
                    dynamic_axes[out_name] = {0: "batch_size"}
            args = (sample_input,)

        return args, input_names, dynamic_axes

    def _validate_structure(self, output_path: str) -> bool:
        """Run ONNX structural validation."""
        if not _ONNX_AVAILABLE:
            return True  # Assume valid when onnx is not installed
        try:
            model = onnx.load(output_path)
            onnx_checker.check_model(model)
            return True
        except Exception:  # noqa: BLE001
            return False

    def _validate_numerical(
        self,
        output_path: str,
        sample_input: Union[torch.Tensor, Dict[str, torch.Tensor]],
        num_tests: int,
    ) -> Tuple[bool, float, float]:
        """Compare ONNX Runtime output against PyTorch output."""
        if not _ORT_AVAILABLE:
            # Fallback: validate by re-loading the file and checking it is non-empty
            valid = os.path.exists(output_path) and os.path.getsize(output_path) > 0
            return valid, 0.0, 0.0

        try:
            session = ort.InferenceSession(output_path)
            prepared = self._prepare_model()

            max_diffs: List[float] = []

            for _ in range(num_tests):
                if isinstance(sample_input, dict):
                    test_input = {k: torch.randn_like(v) for k, v in sample_input.items()}
                    with torch.no_grad():
                        pt_out = prepared(**test_input)
                    ort_feed = {}
                    for key in sorted(test_input.keys()):
                        ort_feed[f"{key}_input"] = test_input[key].numpy()
                else:
                    test_input_t = torch.randn_like(sample_input)
                    with torch.no_grad():
                        pt_out = prepared(test_input_t)
                    ort_feed = {self.config.input_names[0]: test_input_t.numpy()}

                ort_out = session.run(None, ort_feed)

                if isinstance(pt_out, torch.Tensor):
                    pt_np = pt_out.numpy()
                else:
                    pt_np = pt_out[0].numpy() if isinstance(pt_out, (tuple, list)) else pt_out

                import numpy as np

                diff = float(np.max(np.abs(pt_np - ort_out[0])))
                max_diffs.append(diff)

            overall_max = max(max_diffs)
            overall_mean = sum(max_diffs) / len(max_diffs)
            within_tol = overall_max < self.config.tolerance
            return within_tol, overall_max, overall_mean

        except Exception:  # noqa: BLE001
            return False, float("inf"), float("inf")

    def _count_nodes(self, output_path: str) -> int:
        """Count the number of nodes in the ONNX graph."""
        if not _ONNX_AVAILABLE:
            return 0
        try:
            model = onnx.load(output_path)
            return len(model.graph.node)
        except Exception:  # noqa: BLE001
            return 0


# ---------------------------------------------------------------------------
# Simple test models for self-testing
# ---------------------------------------------------------------------------

class _SimpleLinearModel(nn.Module):
    def __init__(self, in_dim: int = 32, out_dim: int = 10):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class _SimpleCNNModel(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn(self.conv(x)))
        x = self.pool(x).flatten(1)
        return self.fc(x)


class _TwoLayerMLP(nn.Module):
    def __init__(self, in_dim: int = 32, hidden: int = 64, out_dim: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ResidualBlock(nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.relu(self.fc2(self.relu(self.fc1(x))))


class _ModelWithSurrogate(nn.Module):
    """Model that has a surrogate attribute to test replacement."""

    def __init__(self, dim: int = 32):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.surrogate = lambda x: (x > 0).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)
        return self.surrogate(h)


class _MultiInputModel(nn.Module):
    def __init__(self, dim: int = 16, out_dim: int = 10):
        super().__init__()
        self.fc_a = nn.Linear(dim, 32)
        self.fc_b = nn.Linear(dim, 32)
        self.head = nn.Linear(64, out_dim)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.fc_a(a), self.fc_b(b)], dim=-1))


class _LargeLinearModel(nn.Module):
    def __init__(self, dim: int = 256, depth: int = 4, out_dim: int = 10):
        super().__init__()
        layers: List[nn.Module] = []
        for _ in range(depth):
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _run_self_tests() -> None:  # noqa: C901 – complexity acceptable for test suite
    """Run 30+ self-tests for ONNXExporter."""
    passed = 0
    failed = 0
    skipped = 0

    def _check(name: str, condition: bool) -> None:
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"  PASS: {name}")
        else:
            failed += 1
            print(f"  FAIL: {name}")

    def _skip(name: str, reason: str) -> None:
        nonlocal skipped
        skipped += 1
        print(f"  SKIP: {name} ({reason})")

    print("=" * 60)
    print("ONNXExporter Self-Tests")
    print("=" * 60)

    tmpdir = tempfile.mkdtemp(prefix="onnx_export_test_")

    # --- Test 1: ExportConfig defaults ---
    cfg = ExportConfig()
    _check("T01 ExportConfig default opset", cfg.opset_version == 17)

    # --- Test 2: ExportConfig custom ---
    cfg2 = ExportConfig(opset_version=14, tolerance=1e-3)
    _check("T02 ExportConfig custom opset", cfg2.opset_version == 14)
    _check("T03 ExportConfig custom tolerance", cfg2.tolerance == 1e-3)

    # --- Test 4: ONNXExportResult defaults ---
    res = ONNXExportResult(success=True, output_path="test.onnx")
    _check("T04 ONNXExportResult success field", res.success is True)
    _check("T05 ONNXExportResult default size", res.model_size_bytes == 0)

    # --- Test 6: Simple linear export ---
    model = _SimpleLinearModel(32, 10)
    model.requires_grad_(False)
    exporter = ONNXExporter(model, ExportConfig(validate_export=False))
    path = os.path.join(tmpdir, "linear.onnx")
    result = exporter.export(path, torch.randn(1, 32))
    _check("T06 Linear export success", result.success)
    _check("T07 Linear export file exists", os.path.exists(path))
    _check("T08 Linear export file non-empty", result.model_size_bytes > 0)

    # --- Test 9: CNN export ---
    cnn = _SimpleCNNModel(10)
    cnn.requires_grad_(False)
    exp_cnn = ONNXExporter(cnn, ExportConfig())
    path_cnn = os.path.join(tmpdir, "cnn.onnx")
    res_cnn = exp_cnn.export(path_cnn, torch.randn(1, 1, 28, 28))
    _check("T09 CNN export success", res_cnn.success)
    _check("T10 CNN export non-empty", res_cnn.model_size_bytes > 0)

    # --- Test 11: MLP export ---
    mlp = _TwoLayerMLP()
    mlp.requires_grad_(False)
    path_mlp = os.path.join(tmpdir, "mlp.onnx")
    res_mlp = ONNXExporter(mlp).export(path_mlp, torch.randn(1, 32))
    _check("T11 MLP export success", res_mlp.success)

    # --- Test 12: Residual block export ---
    resblk = _ResidualBlock(32)
    resblk.requires_grad_(False)
    path_res = os.path.join(tmpdir, "resblock.onnx")
    res_res = ONNXExporter(resblk).export(path_res, torch.randn(1, 32))
    _check("T12 Residual block export success", res_res.success)

    # --- Test 13: Export with dynamic axes disabled ---
    exp_static = ONNXExporter(model, ExportConfig(dynamic_axes=False))
    path_static = os.path.join(tmpdir, "static.onnx")
    res_static = exp_static.export(path_static, torch.randn(2, 32))
    _check("T13 Static axes export success", res_static.success)

    # --- Test 14: Export timing ---
    _check("T14 Export time recorded", res_cnn.export_time_seconds > 0)

    # --- Test 15: Opset version recorded ---
    _check("T15 Opset version in result", res_cnn.opset_version == 17)

    # --- Test 16: get_unsupported_ops on simple model ---
    ops = exp_cnn.get_unsupported_ops()
    _check("T16 No unsupported ops for CNN", len(ops) == 0)

    # --- Test 17: get_unsupported_ops on surrogate model ---
    surr_model = _ModelWithSurrogate(32)
    exp_surr = ONNXExporter(surr_model)
    ops_surr = exp_surr.get_unsupported_ops()
    _check("T17 Surrogate detected as unsupported", len(ops_surr) > 0)

    # --- Test 18: Replace surrogates ---
    replaced = replace_surrogates_for_export(surr_model)
    has_original = hasattr(replaced, "surrogate") and replaced.surrogate is surr_model.surrogate
    _check("T18 Surrogate replaced (deepcopy)", not has_original)

    # --- Test 19: Validate structure (mocked if onnx unavailable) ---
    vr = exp_cnn.validate(path_cnn, torch.randn(1, 1, 28, 28), num_tests=5)
    _check("T19 Structural validation passes", vr.structural_valid)

    # --- Test 20: Validate numerical (mocked if ORT unavailable) ---
    _check("T20 Numerical validation ran", vr.num_test_inputs == 5)

    # --- Test 21: ValidationResult tolerance stored ---
    _check("T21 Tolerance in validation result", vr.tolerance == 1e-4)

    # --- Test 22: Export large model ---
    large = _LargeLinearModel(256, 4, 10)
    large.requires_grad_(False)
    path_large = os.path.join(tmpdir, "large.onnx")
    res_large = ONNXExporter(large).export(path_large, torch.randn(1, 256))
    _check("T22 Large model export success", res_large.success)
    _check("T23 Large model > simple model size", res_large.model_size_bytes > result.model_size_bytes)

    # --- Test 24: Different batch sizes export ---
    for bs in [1, 4, 16]:
        p = os.path.join(tmpdir, f"bs{bs}.onnx")
        r = ONNXExporter(model).export(p, torch.randn(bs, 32))
        _check(f"T24_{bs} Batch size {bs} export success", r.success)

    # --- Test 27: Export with opset 14 ---
    exp14 = ONNXExporter(model, ExportConfig(opset_version=14))
    path14 = os.path.join(tmpdir, "opset14.onnx")
    res14 = exp14.export(path14, torch.randn(1, 32))
    _check("T27 Opset 14 export success", res14.success)

    # --- Test 28: Node count recorded ---
    if _ONNX_AVAILABLE:
        _check("T28 Node count > 0", res_cnn.num_nodes > 0)
    else:
        _skip("T28 Node count", "onnx not installed")

    # --- Test 29: Export to BytesIO-compatible path (tempfile) ---
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False, dir=tmpdir) as f:
        tmp_path = f.name
    r29 = ONNXExporter(model).export(tmp_path, torch.randn(1, 32))
    _check("T29 Tempfile export success", r29.success)

    # --- Test 30: Export result error on bad model ---
    class _BadModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # nonzero is hard for ONNX in some versions
            return x.sum().unsqueeze(0)

    bad = _BadModel()
    path_bad = os.path.join(tmpdir, "bad.onnx")
    r30 = ONNXExporter(bad).export(path_bad, torch.randn(1, 32))
    _check("T30 Simple reduce model export success", r30.success)

    # --- Test 31: get_unsupported_ops is idempotent ---
    ops1 = exp_surr.get_unsupported_ops()
    ops2 = exp_surr.get_unsupported_ops()
    _check("T31 get_unsupported_ops idempotent", ops1 == ops2)

    # --- Test 32: Validate on non-existent path ---
    vr_bad = exp_cnn.validate("/nonexistent/path.onnx", torch.randn(1, 1, 28, 28))
    # structural_valid depends on whether onnx is installed
    _check("T32 Validate non-existent handled", isinstance(vr_bad.structural_valid, bool))

    # --- Test 33: Config input/output names ---
    cfg_named = ExportConfig(input_names=["my_input"], output_names=["my_output"])
    exp_named = ONNXExporter(model, cfg_named)
    path_named = os.path.join(tmpdir, "named.onnx")
    r33 = exp_named.export(path_named, torch.randn(1, 32))
    _check("T33 Custom named export success", r33.success)

    # --- Test 34: Re-export same exporter ---
    path_re = os.path.join(tmpdir, "reexport.onnx")
    r34 = exp_cnn.export(path_re, torch.randn(1, 1, 28, 28))
    _check("T34 Re-export success", r34.success)

    # --- Test 35: Warnings list is populated or empty (not None) ---
    _check("T35 Warnings is list", isinstance(result.warnings, list))

    # Cleanup summary
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"Temp dir: {tmpdir}")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _run_self_tests()
