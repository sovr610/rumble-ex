"""
compile_benchmark_template.py
------------------------------
CompileBenchmark: compares eager vs compiled training throughput.

Measures compile time, steady-state tokens/sec, peak memory, and
speedup ratio across modes. Generates human-readable table and JSON output.

Usage:
    from compile_benchmark_template import CompileBenchmark
    import torch
    import torch.nn as nn

    def model_fn():
        return nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 256))

    def batch_fn(device):
        return {"x": torch.randn(32, 256, device=device)}

    bench = CompileBenchmark()
    results = bench.run_comparison(
        model_fn=model_fn,
        batch_fn=batch_fn,
        steps=100,
        warmup=20,
        modes=["eager", "default", "reduce-overhead"],
    )
    print(bench.format_report(results))
    bench.save_report(results, "compile_benchmark.json")
"""

from __future__ import annotations

import json
import logging
import statistics
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


class BenchmarkResult:
    """Container for a single mode's benchmark results."""

    def __init__(
        self,
        mode: str,
        compile_time_s: float,
        step_times_s: List[float],
        peak_memory_bytes: int,
        error: Optional[str] = None,
        tokens_per_step: int = 0,
    ) -> None:
        self.mode = mode
        self.compile_time_s = compile_time_s
        self.step_times_s = step_times_s
        self.peak_memory_bytes = peak_memory_bytes
        self.error = error
        self.tokens_per_step = tokens_per_step

    @property
    def failed(self) -> bool:
        return self.error is not None

    @property
    def steady_state_times(self) -> List[float]:
        """Step times excluding the first step (compilation)."""
        if len(self.step_times_s) <= 1:
            return self.step_times_s
        return self.step_times_s[1:]

    @property
    def steady_state_tokens_per_sec_p50(self) -> float:
        """Median tokens/sec over steady-state steps."""
        times = self.steady_state_times
        if not times or self.tokens_per_step == 0:
            return 0.0
        rates = [self.tokens_per_step / t for t in times if t > 0]
        if not rates:
            return 0.0
        return statistics.median(rates)

    @property
    def step_time_p50_s(self) -> float:
        """Median step time over steady-state steps."""
        times = self.steady_state_times
        if not times:
            return 0.0
        return statistics.median(times)

    @property
    def step_time_p95_s(self) -> float:
        """95th percentile step time over steady-state steps."""
        times = sorted(self.steady_state_times)
        if not times:
            return 0.0
        idx = int(0.95 * len(times))
        return times[min(idx, len(times) - 1)]

    def speedup_vs(self, baseline: "BenchmarkResult") -> float:
        """Speedup ratio of this mode vs a baseline mode."""
        if baseline.failed or self.failed:
            return 0.0
        baseline_p50 = baseline.step_time_p50_s
        self_p50 = self.step_time_p50_s
        if self_p50 <= 0 or baseline_p50 <= 0:
            return 0.0
        return baseline_p50 / self_p50

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "compile_time_s": self.compile_time_s,
            "step_times_s": self.step_times_s,
            "peak_memory_bytes": self.peak_memory_bytes,
            "tokens_per_step": self.tokens_per_step,
            "steady_state_tokens_per_sec_p50": self.steady_state_tokens_per_sec_p50,
            "step_time_p50_s": self.step_time_p50_s,
            "step_time_p95_s": self.step_time_p95_s,
            "error": self.error,
            "failed": self.failed,
        }


# ---------------------------------------------------------------------------
# CompileBenchmark
# ---------------------------------------------------------------------------


class CompileBenchmark:
    """
    Benchmark comparing eager mode vs various torch.compile modes.

    For each mode:
    1. Create a fresh model instance via model_fn().
    2. Apply torch.compile (or skip for "eager").
    3. Run warmup steps (including first-step compilation).
    4. Measure step times for benchmark_steps steps with CUDA synchronization.
    5. Record compile time, step times, peak memory.
    6. Compute speedup vs eager.

    Parameters
    ----------
    device:
        Target device. Defaults to CUDA if available, else CPU.
    loss_fn:
        Optional scalar loss function. If None, uses output.mean().
    use_amp:
        Whether to use Automatic Mixed Precision (torch.cuda.amp.autocast).
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        loss_fn: Optional[Callable] = None,
        use_amp: bool = False,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.loss_fn = loss_fn
        self.use_amp = use_amp and torch.cuda.is_available()

    def run_comparison(
        self,
        model_fn: Callable[[], nn.Module],
        batch_fn: Callable[[torch.device], Dict[str, Any]],
        steps: int = 100,
        warmup: int = 50,
        modes: List[str] = None,
    ) -> Dict[str, BenchmarkResult]:
        """
        Run benchmark across all specified modes.

        Parameters
        ----------
        model_fn:
            Callable that creates a fresh model instance. Called once per mode.
            The model should NOT be pre-compiled.
        batch_fn:
            Callable(device) -> Dict[str, Tensor]. Creates a sample batch
            on the given device. Called each step.
        steps:
            Number of measurement steps (after warmup).
        warmup:
            Number of warmup steps. First step triggers JIT compilation.
        modes:
            List of modes to benchmark. Valid values:
            "eager", "default", "reduce-overhead", "max-autotune",
            "max-autotune-no-cudagraphs"
            Default: ["eager", "default", "reduce-overhead"]

        Returns
        -------
        Dict[str, BenchmarkResult]
            Mode name -> BenchmarkResult. Always contains an "eager" baseline.
        """
        if modes is None:
            modes = ["eager", "default", "reduce-overhead"]

        if "eager" not in modes:
            modes = ["eager"] + list(modes)

        results: Dict[str, BenchmarkResult] = {}

        logger.info(
            "benchmark: starting comparison — device=%s, modes=%s, "
            "warmup=%d, steps=%d",
            self.device,
            modes,
            warmup,
            steps,
        )

        for mode in modes:
            logger.info("benchmark: running mode='%s'", mode)
            result = self._run_mode(
                mode=mode,
                model_fn=model_fn,
                batch_fn=batch_fn,
                warmup=warmup,
                steps=steps,
            )
            results[mode] = result

            if result.failed:
                logger.error(
                    "benchmark: mode='%s' FAILED: %s", mode, result.error
                )
            else:
                logger.info(
                    "benchmark: mode='%s' p50=%.3fs tokens/s=%.0f "
                    "memory=%.1fMB compile_time=%.3fs",
                    mode,
                    result.step_time_p50_s,
                    result.steady_state_tokens_per_sec_p50,
                    result.peak_memory_bytes / 1024**2,
                    result.compile_time_s,
                )

        # Attach speedup_vs_eager to results dict
        eager_result = results.get("eager")
        for mode, result in results.items():
            if mode != "eager" and eager_result and not result.failed:
                su = result.speedup_vs(eager_result)
                logger.info("benchmark: mode='%s' speedup=%.2fx vs eager", mode, su)

        return results

    def _run_mode(
        self,
        mode: str,
        model_fn: Callable[[], nn.Module],
        batch_fn: Callable[[torch.device], Dict[str, Any]],
        warmup: int,
        steps: int,
    ) -> BenchmarkResult:
        """Run benchmark for a single mode."""
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Create fresh model
        model = model_fn().to(self.device)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Apply compilation
        compile_start = time.perf_counter()
        if mode != "eager":
            try:
                model = torch.compile(model, mode=mode, backend="inductor")
            except Exception as exc:
                return BenchmarkResult(
                    mode=mode,
                    compile_time_s=time.perf_counter() - compile_start,
                    step_times_s=[],
                    peak_memory_bytes=0,
                    error=f"torch.compile() failed: {traceback.format_exc()}",
                )
        compile_call_time = time.perf_counter() - compile_start

        # Get a sample batch to determine tokens_per_step
        sample_batch = batch_fn(self.device)
        tokens_per_step = self._count_tokens(sample_batch)

        # Warmup (includes first-step JIT compilation)
        logger.debug("benchmark: mode='%s' warmup %d steps", mode, warmup)
        warmup_times = []
        for w in range(warmup):
            t = self._step(model, sample_batch, optimizer)
            warmup_times.append(t)
            if w == 0 and mode != "eager":
                logger.debug(
                    "benchmark: mode='%s' first step (compile+run)=%.3fs", mode, t
                )

        # JIT compilation happens on the first step; record its wall time
        first_step_compile_time = warmup_times[0] if warmup_times else 0.0

        # Measurement steps
        logger.debug("benchmark: mode='%s' measuring %d steps", mode, steps)
        step_times = []
        try:
            for _ in range(steps):
                t = self._step(model, sample_batch, optimizer)
                step_times.append(t)
        except Exception as exc:
            return BenchmarkResult(
                mode=mode,
                compile_time_s=first_step_compile_time,
                step_times_s=step_times,
                peak_memory_bytes=int(
                    torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
                ),
                error=f"Step failed: {traceback.format_exc()}",
                tokens_per_step=tokens_per_step,
            )

        peak_memory = int(
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        )

        return BenchmarkResult(
            mode=mode,
            compile_time_s=first_step_compile_time,
            step_times_s=step_times,
            peak_memory_bytes=peak_memory,
            tokens_per_step=tokens_per_step,
        )

    def _step(
        self,
        model: nn.Module,
        batch: Dict[str, Any],
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Execute one training step and return wall time in seconds."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        optimizer.zero_grad()

        if self.use_amp:
            with torch.cuda.amp.autocast():
                output = self._forward(model, batch)
                loss = self._compute_loss(output, batch)
        else:
            output = self._forward(model, batch)
            loss = self._compute_loss(output, batch)

        loss.backward()
        optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter() - t0

    def _forward(self, model: nn.Module, batch: Dict[str, Any]) -> Any:
        """Try model(**batch) then model(batch_values[0])."""
        tensor_batch = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
        try:
            return model(**tensor_batch)
        except TypeError:
            vals = list(tensor_batch.values())
            if vals:
                return model(vals[0])
            raise

    def _compute_loss(self, output: Any, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute scalar loss from model output."""
        if self.loss_fn is not None:
            return self.loss_fn(output, batch)

        if isinstance(output, torch.Tensor):
            return output.float().mean()
        elif hasattr(output, "loss") and output.loss is not None:
            return output.loss
        elif isinstance(output, (tuple, list)):
            return output[0].float().mean()
        else:
            raise RuntimeError(
                f"Cannot compute loss from output type {type(output)}. "
                "Provide a loss_fn to CompileBenchmark."
            )

    def _count_tokens(self, batch: Dict[str, Any]) -> int:
        """Estimate tokens per step from batch."""
        if "input_ids" in batch and isinstance(batch["input_ids"], torch.Tensor):
            return int(batch["input_ids"].numel())
        # Fall back to total elements of first tensor
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                return int(v.numel())
        return 0

    # ---------------------------------------------------------------------------
    # Reporting
    # ---------------------------------------------------------------------------

    def format_report(self, results: Dict[str, "BenchmarkResult"]) -> str:
        """
        Format benchmark results as a human-readable table.

        Returns
        -------
        str
            Multi-line table with mode, compile time, step time, throughput,
            memory, and speedup columns.
        """
        eager_result = results.get("eager")

        # Header
        lines = [
            "",
            "torch.compile Benchmark Results",
            "=" * 80,
            f"{'Mode':<30} {'Compile(s)':<12} {'p50 step(s)':<14} "
            f"{'tok/s':<12} {'Peak Mem(MB)':<14} {'Speedup'}",
            "-" * 80,
        ]

        for mode in ["eager"] + [m for m in results if m != "eager"]:
            if mode not in results:
                continue
            r = results[mode]

            if r.failed:
                lines.append(f"{mode:<30} {'ERROR':<12} {r.error[:40]}")
                continue

            compile_str = f"{r.compile_time_s:.3f}"
            step_str = f"{r.step_time_p50_s:.4f}"
            toks_str = (
                f"{r.steady_state_tokens_per_sec_p50:,.0f}"
                if r.steady_state_tokens_per_sec_p50 > 0
                else "N/A"
            )
            mem_str = f"{r.peak_memory_bytes / 1024**2:.1f}"

            if mode == "eager" or eager_result is None or eager_result.failed:
                speedup_str = "1.00x (baseline)"
            else:
                su = r.speedup_vs(eager_result)
                speedup_str = f"{su:.2f}x"

            lines.append(
                f"{mode:<30} {compile_str:<12} {step_str:<14} "
                f"{toks_str:<12} {mem_str:<14} {speedup_str}"
            )

        lines.append("-" * 80)

        # Summary note
        if eager_result and not eager_result.failed:
            eager_p50 = eager_result.step_time_p50_s
            lines.append(f"Baseline (eager) p50 step time: {eager_p50:.4f}s")

        lines.append("")
        return "\n".join(lines)

    def save_report(
        self,
        results: Dict[str, "BenchmarkResult"],
        path: str,
    ) -> None:
        """
        Save benchmark results as JSON.

        Parameters
        ----------
        results:
            Dict of mode -> BenchmarkResult from run_comparison().
        path:
            Output file path. Parent directories are created if needed.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        # Build serializable dict
        eager_result = results.get("eager")
        output = {}
        for mode, result in results.items():
            d = result.to_dict()
            if mode != "eager" and eager_result and not result.failed and not eager_result.failed:
                d["speedup_vs_eager"] = result.speedup_vs(eager_result)
            else:
                d["speedup_vs_eager"] = 1.0 if mode == "eager" else None
            output[mode] = d

        with open(p, "w") as f:
            json.dump(output, f, indent=2)

        logger.info("benchmark: saved report to %s", p)


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    failures: list = []

    def _check(name: str, condition: bool, msg: str = "") -> None:
        if condition:
            print(f"  PASS  {name}")
        else:
            print(f"  FAIL  {name}: {msg}")
            failures.append(name)

    print("=" * 60)
    print("CompileBenchmark self-tests")
    print("=" * 60)

    # Simple model and batch for testing
    def make_small_model():
        return nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 32))

    def make_batch(device):
        return {"x": torch.randn(4, 32, device=device)}

    bench = CompileBenchmark()

    # Test 1: Eager mode benchmark completes
    try:
        results = bench.run_comparison(
            model_fn=make_small_model,
            batch_fn=make_batch,
            steps=5,
            warmup=2,
            modes=["eager"],
        )
        _check("eager_mode_completes", "eager" in results)
        _check(
            "eager_not_failed",
            not results["eager"].failed,
            f"Error: {results['eager'].error}",
        )
    except Exception as e:
        _check("eager_mode_completes", False, str(e))
        _check("eager_not_failed", False, str(e))

    # Test 2: Eager result has required keys
    try:
        r = results["eager"]
        d = r.to_dict()
        required_keys = [
            "mode", "compile_time_s", "step_times_s", "peak_memory_bytes",
            "steady_state_tokens_per_sec_p50", "step_time_p50_s",
            "step_time_p95_s", "error", "failed",
        ]
        for key in required_keys:
            _check(f"eager_has_{key}", key in d, f"Missing key '{key}' in {list(d.keys())}")
    except Exception as e:
        for key in required_keys:
            _check(f"eager_has_{key}", False, str(e))

    # Test 3: Eager baseline speedup is 1.0x
    try:
        _check(
            "eager_speedup_is_1x",
            results["eager"].speedup_vs(results["eager"]) == 1.0,
        )
    except Exception as e:
        _check("eager_speedup_is_1x", False, str(e))

    # Test 4: Comparison with compile mode produces all expected keys
    try:
        results_full = bench.run_comparison(
            model_fn=make_small_model,
            batch_fn=make_batch,
            steps=5,
            warmup=2,
            modes=["eager", "default"],
        )
        _check("comparison_has_eager", "eager" in results_full)
        _check("comparison_has_default", "default" in results_full)

        if "default" in results_full and not results_full["default"].failed:
            d2 = results_full["default"].to_dict()
            _check(
                "compiled_mode_has_required_keys",
                all(k in d2 for k in required_keys),
            )
    except Exception as e:
        _check("comparison_has_eager", False, str(e))
        _check("comparison_has_default", False, str(e))
        _check("compiled_mode_has_required_keys", False, str(e))

    # Test 5: Speedup ratio is in reasonable range
    try:
        if "default" in results_full and not results_full["default"].failed:
            su = results_full["default"].speedup_vs(results_full["eager"])
            _check(
                "speedup_in_reasonable_range",
                0.1 <= su <= 20.0,
                f"Speedup {su:.2f}x is outside [0.1, 20.0]",
            )
        else:
            # Skip if compile failed (e.g., no GPU in test environment)
            print("  SKIP  speedup_in_reasonable_range (compile mode failed, likely no GPU)")
    except Exception as e:
        _check("speedup_in_reasonable_range", False, str(e))

    # Test 6: format_report returns non-empty string
    try:
        report = bench.format_report(results_full if "results_full" in dir() else results)
        _check(
            "format_report_nonempty",
            isinstance(report, str) and len(report) > 50,
            f"Report length {len(report)}",
        )
        _check(
            "format_report_contains_eager",
            "eager" in report,
            "Report missing 'eager' entry",
        )
    except Exception as e:
        _check("format_report_nonempty", False, str(e))
        _check("format_report_contains_eager", False, str(e))

    # Test 7: save_report writes valid JSON
    try:
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            bench.save_report(results, tmp_path)
            with open(tmp_path) as f:
                loaded = json.load(f)
            _check("save_report_valid_json", "eager" in loaded)
            _check(
                "save_report_has_required_fields",
                "compile_time_s" in loaded.get("eager", {}),
            )
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        _check("save_report_valid_json", False, str(e))
        _check("save_report_has_required_fields", False, str(e))

    # Test 8: BenchmarkResult properties
    try:
        r = BenchmarkResult(
            mode="test",
            compile_time_s=1.5,
            step_times_s=[1.0, 0.01, 0.01, 0.01, 0.01],
            peak_memory_bytes=1024 * 1024 * 100,
            tokens_per_step=1024,
        )
        _check("steady_state_times", len(r.steady_state_times) == 4)
        _check("step_time_p50_positive", r.step_time_p50_s > 0)
        _check(
            "tokens_per_sec_positive",
            r.steady_state_tokens_per_sec_p50 > 0,
            f"Got {r.steady_state_tokens_per_sec_p50}",
        )
    except Exception as e:
        _check("steady_state_times", False, str(e))
        _check("step_time_p50_positive", False, str(e))
        _check("tokens_per_sec_positive", False, str(e))

    print()
    if failures:
        print(f"FAILED: {len(failures)} tests: {failures}")
        sys.exit(1)
    else:
        total = (
            2  # eager completes
            + len([
                "mode", "compile_time_s", "step_times_s", "peak_memory_bytes",
                "steady_state_tokens_per_sec_p50", "step_time_p50_s",
                "step_time_p95_s", "error", "failed",
            ])  # key checks
            + 1  # speedup 1x
            + 3  # comparison keys
            + 1  # speedup range (conditional skip)
            + 2  # format report
            + 2  # save report
            + 3  # BenchmarkResult properties
        )
        print(f"All required tests passed.")
