"""
memory_profiler_template.py
---------------------------
MemoryProfiler that measures per-layer peak activation memory to guide
selective gradient checkpointing decisions.

Uses CUDA memory statistics and forward hooks to attribute memory to
individual layers. Falls back to parameter-count estimation on CPU.

Usage:
    from memory_profiler_template import MemoryProfiler, ProfileReport

    profiler = MemoryProfiler(model, device=torch.device('cuda'))
    report = profiler.profile({"input_ids": sample_input})
    expensive = profiler.recommend_layers(memory_budget_mb=50.0)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class LayerProfile:
    """Memory and compute profile for a single layer."""

    name: str
    activation_memory_mb: float
    param_count: int
    param_memory_mb: float
    module_type: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProfileReport:
    """Aggregated profiling report for all layers."""

    layers: List[LayerProfile] = field(default_factory=list)
    total_activation_mb: float = 0.0
    peak_memory_mb: float = 0.0
    model_param_mb: float = 0.0
    timestamp: str = ""
    device: str = "cpu"
    num_runs: int = 1

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["layers"] = [lp.to_dict() for lp in self.layers]
        return d

    def summary(self) -> str:
        lines = [
            f"ProfileReport ({self.device}, {self.num_runs} runs)",
            f"  Total activation memory: {self.total_activation_mb:.2f} MB",
            f"  Peak memory: {self.peak_memory_mb:.2f} MB",
            f"  Model parameters: {self.model_param_mb:.2f} MB",
            f"  Layers profiled: {len(self.layers)}",
            "",
        ]
        for lp in self.layers[:10]:
            lines.append(
                f"  {lp.name:40s}  {lp.activation_memory_mb:8.2f} MB  "
                f"({lp.module_type}, {lp.param_count:,} params)"
            )
        if len(self.layers) > 10:
            lines.append(f"  ... and {len(self.layers) - 10} more layers")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# MemoryProfiler
# ---------------------------------------------------------------------------


class MemoryProfiler:
    """
    Profile per-layer activation memory for a model.

    On CUDA, uses torch.cuda.memory_stats() with forward hooks to measure
    actual memory deltas. On CPU, falls back to estimating activation size
    from output tensor shapes.

    Parameters
    ----------
    model : nn.Module
        The model to profile.
    device : torch.device
        Device to run profiling on.
    target_layers : list of str, optional
        If provided, only profile these layers (by name). If None, profile
        all direct children.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        target_layers: Optional[List[str]] = None,
    ) -> None:
        self.model = model
        self.device = device or next(
            (p.device for p in model.parameters()), torch.device("cpu")
        )
        self.target_layers = target_layers
        self._last_report: Optional[ProfileReport] = None

    def profile(
        self,
        sample_input: Dict[str, torch.Tensor],
        num_runs: int = 3,
    ) -> ProfileReport:
        """
        Run forward passes and measure per-layer activation memory.

        Parameters
        ----------
        sample_input : dict
            Keyword arguments to pass to model.forward().
        num_runs : int
            Number of forward passes to average. More runs reduce noise.

        Returns
        -------
        ProfileReport
            Sorted by activation_memory_mb descending.
        """
        use_cuda = self.device.type == "cuda"

        # Determine which layers to profile
        layer_names = []
        layer_modules = {}
        for name, mod in self.model.named_children():
            if self.target_layers is not None and name not in self.target_layers:
                continue
            layer_names.append(name)
            layer_modules[name] = mod

        if not layer_names:
            logger.warning("No layers to profile.")
            return ProfileReport(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))

        # Move sample input to device
        sample_input = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in sample_input.items()
        }

        # Accumulate measurements
        accumulated: Dict[str, List[float]] = {name: [] for name in layer_names}

        self.model.train(False)
        self.model.to(self.device)

        for run_idx in range(num_runs):
            if use_cuda:
                layer_mem = self._profile_cuda_run(
                    layer_names, layer_modules, sample_input
                )
            else:
                layer_mem = self._profile_cpu_run(
                    layer_names, layer_modules, sample_input
                )

            for name, mem_mb in layer_mem.items():
                accumulated[name].append(mem_mb)

        # Build report
        layers = []
        for name in layer_names:
            values = accumulated[name]
            avg_mem = sum(values) / len(values) if values else 0.0
            mod = layer_modules[name]
            param_count = sum(p.numel() for p in mod.parameters())
            param_mb = sum(
                p.numel() * p.element_size() for p in mod.parameters()
            ) / (1024 * 1024)
            layers.append(
                LayerProfile(
                    name=name,
                    activation_memory_mb=avg_mem,
                    param_count=param_count,
                    param_memory_mb=param_mb,
                    module_type=type(mod).__name__,
                )
            )

        # Sort descending by activation memory
        layers.sort(key=lambda lp: lp.activation_memory_mb, reverse=True)

        total_act = sum(lp.activation_memory_mb for lp in layers)
        model_param_mb = sum(
            p.numel() * p.element_size() for p in self.model.parameters()
        ) / (1024 * 1024)

        peak_mb = 0.0
        if use_cuda:
            peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)

        report = ProfileReport(
            layers=layers,
            total_activation_mb=total_act,
            peak_memory_mb=peak_mb,
            model_param_mb=model_param_mb,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            device=str(self.device),
            num_runs=num_runs,
        )

        self._last_report = report
        return report

    def recommend_layers(
        self,
        memory_budget_mb: float,
        report: Optional[ProfileReport] = None,
    ) -> List[str]:
        """
        Return layer names whose activation memory exceeds the budget threshold.

        Parameters
        ----------
        memory_budget_mb : float
            Layers above this threshold (in MB) are recommended for checkpointing.
        report : ProfileReport, optional
            If not provided, uses the last profile() result.

        Returns
        -------
        list of str
            Layer names to checkpoint, sorted by memory descending.
        """
        rpt = report or self._last_report
        if rpt is None:
            raise RuntimeError(
                "No profile report available. Call profile() first or provide a report."
            )
        return [
            lp.name
            for lp in rpt.layers
            if lp.activation_memory_mb > memory_budget_mb
        ]

    # -------------------------------------------------------------------
    # Internal profiling methods
    # -------------------------------------------------------------------

    def _profile_cuda_run(
        self,
        layer_names: List[str],
        layer_modules: Dict[str, nn.Module],
        sample_input: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Profile one forward pass using CUDA memory stats."""
        mem_before: Dict[str, int] = {}
        mem_after: Dict[str, int] = {}
        hooks = []

        def make_pre_hook(name):
            def hook(module, inputs):
                torch.cuda.synchronize(self.device)
                mem_before[name] = torch.cuda.memory_allocated(self.device)
            return hook

        def make_post_hook(name):
            def hook(module, inputs, outputs):
                torch.cuda.synchronize(self.device)
                mem_after[name] = torch.cuda.memory_allocated(self.device)
            return hook

        # Register hooks
        for name in layer_names:
            mod = layer_modules[name]
            hooks.append(mod.register_forward_pre_hook(make_pre_hook(name)))
            hooks.append(mod.register_forward_hook(make_post_hook(name)))

        # Clear cache and reset stats
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        # Forward pass
        with torch.no_grad():
            self.model(**sample_input)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Compute deltas
        result = {}
        for name in layer_names:
            before = mem_before.get(name, 0)
            after = mem_after.get(name, 0)
            delta_bytes = max(0, after - before)
            result[name] = delta_bytes / (1024 * 1024)

        return result

    def _profile_cpu_run(
        self,
        layer_names: List[str],
        layer_modules: Dict[str, nn.Module],
        sample_input: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Estimate activation memory from output tensor shapes (CPU fallback)."""
        output_sizes: Dict[str, float] = {}
        hooks = []

        def make_hook(name):
            def hook(module, inputs, outputs):
                if isinstance(outputs, torch.Tensor):
                    size_bytes = outputs.numel() * outputs.element_size()
                elif isinstance(outputs, (tuple, list)):
                    size_bytes = sum(
                        o.numel() * o.element_size()
                        for o in outputs
                        if isinstance(o, torch.Tensor)
                    )
                else:
                    size_bytes = 0
                output_sizes[name] = size_bytes / (1024 * 1024)
            return hook

        for name in layer_names:
            hooks.append(layer_modules[name].register_forward_hook(make_hook(name)))

        with torch.no_grad():
            self.model(**sample_input)

        for h in hooks:
            h.remove()

        return output_sizes


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    failures: List[str] = []

    def _check(name: str, condition: bool, msg: str = "") -> None:
        if condition:
            print(f"  PASS  {name}")
        else:
            print(f"  FAIL  {name}: {msg}")
            failures.append(name)

    print("=" * 60)
    print("MemoryProfiler self-tests")
    print("=" * 60)

    torch.manual_seed(42)

    # --- Build a test model ---
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, 32)
            self.layer1 = nn.Linear(32, 128)
            self.layer2 = nn.Linear(128, 256)
            self.layer3 = nn.Linear(256, 64)
            self.head = nn.Linear(64, 10)

        def forward(self, input_ids):
            x = self.embed(input_ids)
            x = torch.relu(self.layer1(x))
            x = torch.relu(self.layer2(x))
            x = torch.relu(self.layer3(x))
            x = self.head(x.mean(dim=1))
            return x

    model = TestModel()
    sample = {"input_ids": torch.randint(0, 100, (4, 16))}

    # --- Test 1: Profile returns a ProfileReport ---
    try:
        profiler = MemoryProfiler(model, device=torch.device("cpu"))
        report = profiler.profile(sample, num_runs=2)
        _check("profile_returns_report", isinstance(report, ProfileReport))
    except Exception as e:
        _check("profile_returns_report", False, str(e))

    # --- Test 2: Layers have positive activation memory ---
    try:
        non_trivial = [lp for lp in report.layers if lp.param_count > 0]
        all_positive = all(lp.activation_memory_mb > 0 for lp in non_trivial)
        _check(
            "positive_activation_memory",
            all_positive,
            f"layers: {[(lp.name, lp.activation_memory_mb) for lp in non_trivial]}",
        )
    except Exception as e:
        _check("positive_activation_memory", False, str(e))

    # --- Test 3: Report is sorted descending ---
    try:
        mems = [lp.activation_memory_mb for lp in report.layers]
        sorted_desc = all(mems[i] >= mems[i + 1] for i in range(len(mems) - 1))
        _check("sorted_descending", sorted_desc, f"mems: {mems}")
    except Exception as e:
        _check("sorted_descending", False, str(e))

    # --- Test 4: recommend_layers with high threshold returns empty ---
    try:
        rec = profiler.recommend_layers(memory_budget_mb=1e6)
        _check("high_threshold_empty", len(rec) == 0, f"got {rec}")
    except Exception as e:
        _check("high_threshold_empty", False, str(e))

    # --- Test 5: recommend_layers with 0 threshold returns all with memory ---
    try:
        rec_all = profiler.recommend_layers(memory_budget_mb=0.0)
        positive_layers = [lp.name for lp in report.layers if lp.activation_memory_mb > 0]
        _check(
            "zero_threshold_all",
            set(rec_all) == set(positive_layers),
            f"got {rec_all}, expected {positive_layers}",
        )
    except Exception as e:
        _check("zero_threshold_all", False, str(e))

    # --- Test 6: Profile is reproducible ---
    try:
        report2 = profiler.profile(sample, num_runs=2)
        # Check that the layer ordering is the same
        names1 = [lp.name for lp in report.layers]
        names2 = [lp.name for lp in report2.layers]
        _check("reproducible_ordering", names1 == names2, f"{names1} != {names2}")
    except Exception as e:
        _check("reproducible_ordering", False, str(e))

    # --- Test 7: to_dict roundtrip ---
    try:
        d = report.to_dict()
        _check(
            "to_dict_has_keys",
            "layers" in d and "total_activation_mb" in d and "peak_memory_mb" in d,
        )
    except Exception as e:
        _check("to_dict_has_keys", False, str(e))

    # --- Test 8: summary string ---
    try:
        s = report.summary()
        _check(
            "summary_nonempty",
            len(s) > 50 and "ProfileReport" in s,
            f"summary too short or missing header: {s[:80]}",
        )
    except Exception as e:
        _check("summary_nonempty", False, str(e))

    # --- Test 9: target_layers filter ---
    try:
        profiler2 = MemoryProfiler(
            model, device=torch.device("cpu"), target_layers=["layer2"]
        )
        report3 = profiler2.profile(sample, num_runs=1)
        _check(
            "target_layers_filter",
            len(report3.layers) == 1 and report3.layers[0].name == "layer2",
            f"got {[lp.name for lp in report3.layers]}",
        )
    except Exception as e:
        _check("target_layers_filter", False, str(e))

    # --- Test 10: LayerProfile fields ---
    try:
        lp = report.layers[0]
        _check(
            "layer_profile_fields",
            lp.name != ""
            and lp.param_count >= 0
            and lp.module_type != ""
            and lp.param_memory_mb >= 0,
        )
    except Exception as e:
        _check("layer_profile_fields", False, str(e))

    # --- Test 11: No report raises on recommend ---
    try:
        fresh_profiler = MemoryProfiler(model, torch.device("cpu"))
        try:
            fresh_profiler.recommend_layers(10.0)
            _check("no_report_raises", False, "Should have raised RuntimeError")
        except RuntimeError:
            _check("no_report_raises", True)
    except Exception as e:
        _check("no_report_raises", False, str(e))

    print()
    if failures:
        print(f"FAILED: {len(failures)} tests: {failures}")
        sys.exit(1)
    else:
        print("All 11 tests passed.")
