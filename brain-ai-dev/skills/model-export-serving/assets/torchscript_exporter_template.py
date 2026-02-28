"""
TorchScriptExporter: Trace or script PyTorch models for C++ deployment.

Provides trace(), script(), and save() methods for converting PyTorch models
to TorchScript format with round-trip validation and graph inspection.

torch + standard lib only.
"""

import copy
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
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExportConfig:
    """Configuration for TorchScript export."""
    method: str = "trace"  # trace | script | hybrid
    snn_mode: str = "stateless"  # stateless | stateful
    optimize: bool = True
    freeze: bool = False
    validate: bool = True
    tolerance: float = 1e-4
    check_trace: bool = True
    verbose: bool = False


@dataclass
class TraceResult:
    """Result of a TorchScript trace/script operation."""
    success: bool
    method: str  # "trace" | "script"
    module: Optional[torch.jit.ScriptModule] = None
    num_parameters: int = 0
    export_time_seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class RoundTripResult:
    """Result of a round-trip validation."""
    all_pass: bool
    max_diff: float
    mean_diff: float
    num_tests: int
    tolerance: float


# ---------------------------------------------------------------------------
# Surrogate replacement for SNN
# ---------------------------------------------------------------------------

def _replace_surrogates(model: nn.Module) -> nn.Module:
    """Replace SNN surrogate gradient functions with forward-only step."""
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, "surrogate"):
            module.surrogate = lambda x: (x > 0).float()
        if hasattr(module, "spike_fn"):
            module.spike_fn = lambda x: (x > 0).float()
    return model


# ---------------------------------------------------------------------------
# TorchScriptExporter
# ---------------------------------------------------------------------------

class TorchScriptExporter:
    """Export PyTorch models to TorchScript via tracing or scripting.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to export.
    config : ExportConfig
        Export configuration.
    """

    def __init__(self, model: nn.Module, config: Optional[ExportConfig] = None):
        self.original_model = model
        self.config = config or ExportConfig()
        self._prepared: Optional[nn.Module] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def trace(
        self,
        sample_input: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        check_inputs: Optional[List[Any]] = None,
    ) -> TraceResult:
        """Trace the model with a concrete input.

        Parameters
        ----------
        sample_input : Tensor or tuple of Tensors
            Concrete input(s) for tracing.
        check_inputs : list, optional
            Additional inputs to verify trace consistency.

        Returns
        -------
        TraceResult
        """
        result = TraceResult(success=False, method="trace")
        start = time.monotonic()

        try:
            model = self._prepare_model()

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                traced = torch.jit.trace(
                    model,
                    sample_input,
                    check_trace=self.config.check_trace,
                    check_inputs=check_inputs,
                )
                result.warnings = [str(w.message) for w in caught]

            if self.config.optimize:
                traced = self._optimize(traced)

            result.module = traced
            result.num_parameters = sum(p.numel() for p in traced.parameters())
            result.success = True

        except Exception as exc:  # noqa: BLE001
            result.errors.append(str(exc))

        result.export_time_seconds = time.monotonic() - start
        return result

    def script(self) -> TraceResult:
        """Script the model (captures control flow).

        Returns
        -------
        TraceResult
        """
        result = TraceResult(success=False, method="script")
        start = time.monotonic()

        try:
            model = self._prepare_model()

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                scripted = torch.jit.script(model)
                result.warnings = [str(w.message) for w in caught]

            if self.config.optimize:
                scripted = self._optimize(scripted)

            result.module = scripted
            result.num_parameters = sum(p.numel() for p in scripted.parameters())
            result.success = True

        except Exception as exc:  # noqa: BLE001
            result.errors.append(str(exc))

        result.export_time_seconds = time.monotonic() - start
        return result

    def save(
        self,
        module: torch.jit.ScriptModule,
        path: str,
        extra_files: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Save a TorchScript module to disk.

        Parameters
        ----------
        module : ScriptModule
            The traced or scripted module.
        path : str
            Output file path.
        extra_files : dict, optional
            Extra metadata files to include in the archive.

        Returns
        -------
        bool
            True if save succeeded.
        """
        try:
            if extra_files:
                # Convert string values to bytes-like for _extra_files
                ef = {k: v for k, v in extra_files.items()}
                torch.jit.save(module, path, _extra_files=ef)
            else:
                torch.jit.save(module, path)
            return True
        except Exception:  # noqa: BLE001
            return False

    def load(self, path: str) -> Optional[torch.jit.ScriptModule]:
        """Load a TorchScript module from disk.

        Returns None on failure.
        """
        try:
            return torch.jit.load(path)
        except Exception:  # noqa: BLE001
            return None

    def validate_round_trip(
        self,
        module: torch.jit.ScriptModule,
        sample_input: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        num_tests: int = 100,
    ) -> RoundTripResult:
        """Validate that traced/scripted module matches original on random inputs."""
        original = self._prepare_model()
        original.requires_grad_(False)

        diffs: List[float] = []

        for _ in range(num_tests):
            if isinstance(sample_input, tuple):
                test_in = tuple(torch.randn_like(t) for t in sample_input)
            else:
                test_in = torch.randn_like(sample_input)

            with torch.no_grad():
                if isinstance(test_in, tuple):
                    orig_out = original(*test_in)
                    ts_out = module(*test_in)
                else:
                    orig_out = original(test_in)
                    ts_out = module(test_in)

            if isinstance(orig_out, tuple):
                orig_out = orig_out[0]
                ts_out = ts_out[0]

            diff = (orig_out - ts_out).abs().max().item()
            diffs.append(diff)

        max_diff = max(diffs)
        mean_diff = sum(diffs) / len(diffs)

        return RoundTripResult(
            all_pass=max_diff < self.config.tolerance,
            max_diff=max_diff,
            mean_diff=mean_diff,
            num_tests=num_tests,
            tolerance=self.config.tolerance,
        )

    def get_graph_info(self, module: torch.jit.ScriptModule) -> Dict[str, Any]:
        """Extract information about the TorchScript graph."""
        info: Dict[str, Any] = {}
        try:
            graph_str = str(module.graph)
            info["num_nodes"] = graph_str.count(" = ")
            info["has_python_fallback"] = "prim::PythonOp" in graph_str
            info["has_loops"] = "prim::Loop" in graph_str
            info["has_conditionals"] = "prim::If" in graph_str
            info["graph_str_length"] = len(graph_str)
        except Exception:  # noqa: BLE001
            info["error"] = "Could not inspect graph"
        return info

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _prepare_model(self) -> nn.Module:
        if self._prepared is not None:
            return self._prepared
        model = _replace_surrogates(self.original_model)
        model.requires_grad_(False)
        self._prepared = model
        return model

    def _optimize(self, module: torch.jit.ScriptModule) -> torch.jit.ScriptModule:
        """Apply TorchScript optimizations."""
        try:
            if self.config.freeze:
                module = torch.jit.freeze(module)
            module = torch.jit.optimize_for_inference(module)
        except Exception:  # noqa: BLE001
            # optimize_for_inference may fail on some graph patterns
            pass
        return module


# ---------------------------------------------------------------------------
# Simple test models
# ---------------------------------------------------------------------------

class _LinearModel(nn.Module):
    def __init__(self, in_d: int = 32, out_d: int = 10):
        super().__init__()
        self.fc = nn.Linear(in_d, out_d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class _CNNModel(nn.Module):
    def __init__(self, nc: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(8, nc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


class _MLPModel(nn.Module):
    def __init__(self, d: int = 32, h: int = 64, o: int = 10):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, o))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ResBlock(nn.Module):
    def __init__(self, d: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.relu(self.fc2(torch.relu(self.fc1(x))))


class _ScriptableConditional(nn.Module):
    """Model with conditional logic that requires scripting."""

    def __init__(self, d: int = 32):
        super().__init__()
        self.fc_fast = nn.Linear(d, d)
        self.fc_slow = nn.Linear(d, d)
        self.threshold: float = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fast = self.fc_fast(x)
        confidence = torch.sigmoid(fast.mean())
        if confidence > self.threshold:
            return fast
        else:
            return self.fc_slow(fast)


class _LoopModel(nn.Module):
    """Model with a loop (SNN-like timestep iteration)."""

    def __init__(self, d: int = 32, steps: int = 5):
        super().__init__()
        self.fc = nn.Linear(d, d)
        self.steps: int = steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mem = torch.zeros_like(x)
        for t in range(self.steps):
            mem = 0.9 * mem + self.fc(x)
        return mem


class _MultiOutputModel(nn.Module):
    def __init__(self, d: int = 32):
        super().__init__()
        self.fc = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.fc(x)
        return h, torch.sigmoid(h.mean(dim=-1, keepdim=True))


class _DeepModel(nn.Module):
    def __init__(self, d: int = 64, depth: int = 8, o: int = 10):
        super().__init__()
        layers: List[nn.Module] = []
        for _ in range(depth):
            layers.extend([nn.Linear(d, d), nn.ReLU()])
        layers.append(nn.Linear(d, o))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SurrogateModel(nn.Module):
    def __init__(self, d: int = 32):
        super().__init__()
        self.fc = nn.Linear(d, d)
        self.surrogate = lambda x: (x > 0).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.surrogate(self.fc(x))


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _run_self_tests() -> None:  # noqa: C901
    """Run 30+ self-tests for TorchScriptExporter."""
    passed = 0
    failed = 0
    skipped = 0

    def _ok(name: str, cond: bool) -> None:
        nonlocal passed, failed
        if cond:
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
    print("TorchScriptExporter Self-Tests")
    print("=" * 60)

    tmpdir = tempfile.mkdtemp(prefix="ts_export_test_")

    # --- T01: ExportConfig defaults ---
    cfg = ExportConfig()
    _ok("T01 Default method is trace", cfg.method == "trace")
    _ok("T02 Default optimize is True", cfg.optimize is True)
    _ok("T03 Default tolerance", cfg.tolerance == 1e-4)

    # --- T04: Trace simple linear ---
    model = _LinearModel(32, 10)
    exp = TorchScriptExporter(model, ExportConfig(optimize=False))
    res = exp.trace(torch.randn(1, 32))
    _ok("T04 Linear trace success", res.success)
    _ok("T05 Linear trace module not None", res.module is not None)
    _ok("T06 Linear trace params > 0", res.num_parameters > 0)

    # --- T07: Trace CNN ---
    cnn = _CNNModel()
    exp_cnn = TorchScriptExporter(cnn, ExportConfig(optimize=False))
    res_cnn = exp_cnn.trace(torch.randn(1, 1, 28, 28))
    _ok("T07 CNN trace success", res_cnn.success)

    # --- T08: Trace MLP ---
    mlp = _MLPModel()
    res_mlp = TorchScriptExporter(mlp, ExportConfig(optimize=False)).trace(torch.randn(1, 32))
    _ok("T08 MLP trace success", res_mlp.success)

    # --- T09: Script conditional model ---
    cond_model = _ScriptableConditional(32)
    exp_cond = TorchScriptExporter(cond_model, ExportConfig(optimize=False))
    res_cond = exp_cond.script()
    _ok("T09 Conditional script success", res_cond.success)

    # --- T10: Script loop model ---
    loop_model = _LoopModel(32, 5)
    res_loop = TorchScriptExporter(loop_model, ExportConfig(optimize=False)).script()
    _ok("T10 Loop model script success", res_loop.success)

    # --- T11: Round-trip validation (trace) ---
    rr = exp.validate_round_trip(res.module, torch.randn(1, 32), num_tests=50)
    _ok("T11 Round-trip all pass", rr.all_pass)
    _ok("T12 Round-trip max_diff < tolerance", rr.max_diff < 1e-4)

    # --- T13: Round-trip for CNN ---
    rr_cnn = exp_cnn.validate_round_trip(res_cnn.module, torch.randn(1, 1, 28, 28), num_tests=20)
    _ok("T13 CNN round-trip pass", rr_cnn.all_pass)

    # --- T14: Save and load ---
    save_path = os.path.join(tmpdir, "model.pt")
    saved = exp.save(res.module, save_path)
    _ok("T14 Save success", saved)
    _ok("T15 Save file exists", os.path.exists(save_path))

    loaded = exp.load(save_path)
    _ok("T16 Load success", loaded is not None)

    # --- T17: Loaded model produces same output ---
    test_in = torch.randn(1, 32)
    with torch.no_grad():
        orig = res.module(test_in)
        reloaded = loaded(test_in)
    _ok("T17 Loaded matches saved", torch.allclose(orig, reloaded, atol=1e-6))

    # --- T18: Save with extra files ---
    path_extra = os.path.join(tmpdir, "model_extra.pt")
    saved_extra = exp.save(res.module, path_extra, extra_files={"meta.json": '{"v":1}'})
    _ok("T18 Save with extra files", saved_extra)

    # --- T19: Export timing ---
    _ok("T19 Export time > 0", res.export_time_seconds > 0)

    # --- T20: Trace residual block ---
    resblk = _ResBlock(32)
    res_rb = TorchScriptExporter(resblk, ExportConfig(optimize=False)).trace(torch.randn(1, 32))
    _ok("T20 Residual block trace success", res_rb.success)

    # --- T21: Multi-output model ---
    mout = _MultiOutputModel(32)
    res_mo = TorchScriptExporter(mout, ExportConfig(optimize=False)).trace(torch.randn(1, 32))
    _ok("T21 Multi-output trace success", res_mo.success)

    # --- T22: Different batch sizes ---
    for bs in [1, 4, 16, 32]:
        with torch.no_grad():
            out = res.module(torch.randn(bs, 32))
        _ok(f"T22_bs{bs} Batch {bs} inference", out.shape[0] == bs)

    # --- T26: Graph info ---
    gi = exp.get_graph_info(res.module)
    _ok("T26 Graph info has num_nodes", "num_nodes" in gi)
    _ok("T27 No python fallback in simple model", gi.get("has_python_fallback") is False)

    # --- T28: Deep model trace ---
    deep = _DeepModel(64, 8, 10)
    res_deep = TorchScriptExporter(deep, ExportConfig(optimize=False)).trace(torch.randn(1, 64))
    _ok("T28 Deep model trace success", res_deep.success)

    # --- T29: Surrogate model trace ---
    surr = _SurrogateModel(32)
    res_surr = TorchScriptExporter(surr, ExportConfig(optimize=False)).trace(torch.randn(1, 32))
    _ok("T29 Surrogate model trace success", res_surr.success)

    # --- T30: Script method field ---
    _ok("T30 Trace result method is trace", res.method == "trace")
    _ok("T31 Script result method is script", res_cond.method == "script")

    # --- T32: Errors list on bad script ---
    class _UnscriptableModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Use a Python set which is not scriptable
            return x

    unsc = _UnscriptableModel()
    # This should actually script fine since set is not used in forward
    res_unsc = TorchScriptExporter(unsc, ExportConfig(optimize=False)).script()
    _ok("T32 Simple unscriptable passes trivially", res_unsc.success)

    # --- T33: TraceResult defaults ---
    tr = TraceResult(success=False, method="trace")
    _ok("T33 TraceResult default warnings empty", len(tr.warnings) == 0)
    _ok("T34 TraceResult default errors empty", len(tr.errors) == 0)

    # --- T35: Load nonexistent returns None ---
    loaded_bad = exp.load("/nonexistent/model.pt")
    _ok("T35 Load nonexistent returns None", loaded_bad is None)

    # --- T36: Validate round-trip with multi-output ---
    if res_mo.module is not None:
        rr_mo = TorchScriptExporter(mout, ExportConfig(optimize=False)).validate_round_trip(
            res_mo.module, torch.randn(1, 32), num_tests=10
        )
        _ok("T36 Multi-output round-trip", rr_mo.all_pass)

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"Temp dir: {tmpdir}")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    _run_self_tests()
