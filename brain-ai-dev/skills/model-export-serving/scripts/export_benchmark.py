#!/usr/bin/env python3
"""
export_benchmark.py -- Latency and throughput benchmarks for exported models.

Benchmarks:
- FP32 baseline inference latency/throughput
- TorchScript traced latency/throughput
- TorchScript optimized (freeze + optimize_for_inference)
- Dynamic INT8 quantized latency/throughput
- FP16 latency/throughput (CPU emulation or GPU)
- Batch size scaling
- Model size comparison
- Export time measurement

torch + standard lib only. All benchmarks self-contained.
"""

import argparse
import copy
import gc
import json
import os
import sys
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""
    warmup_iterations: int = 20
    benchmark_iterations: int = 100
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8, 16, 32])
    model_configs: List[str] = field(default_factory=lambda: ["small", "medium", "large"])
    include_quantized: bool = True
    include_fp16: bool = True
    include_torchscript: bool = True
    device: str = "cpu"
    output_format: str = "table"  # table | json
    verbose: bool = False


# ---------------------------------------------------------------------------
# Benchmark result data classes
# ---------------------------------------------------------------------------

@dataclass
class LatencyResult:
    """Latency measurements for a single configuration."""
    name: str
    batch_size: int
    mean_ms: float = 0.0
    median_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    std_ms: float = 0.0
    throughput_samples_per_sec: float = 0.0
    num_iterations: int = 0


@dataclass
class SizeResult:
    """Model size measurements."""
    name: str
    param_count: int = 0
    size_bytes: int = 0
    size_mb: float = 0.0


@dataclass
class ExportTimeResult:
    """Export time measurements."""
    name: str
    export_time_seconds: float = 0.0


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""
    latency_results: List[LatencyResult] = field(default_factory=list)
    size_results: List[SizeResult] = field(default_factory=list)
    export_times: List[ExportTimeResult] = field(default_factory=list)
    device: str = "cpu"
    torch_version: str = ""
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------

class SmallModel(nn.Module):
    """~5K parameters."""
    def __init__(self, d: int = 32, o: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, o),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MediumModel(nn.Module):
    """~100K parameters."""
    def __init__(self, d: int = 128, o: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, o),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LargeModel(nn.Module):
    """~2M parameters."""
    def __init__(self, d: int = 512, depth: int = 6, o: int = 10):
        super().__init__()
        layers: list = []
        for _ in range(depth):
            layers.extend([nn.Linear(d, d), nn.ReLU()])
        layers.append(nn.Linear(d, o))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNModel(nn.Module):
    """CNN for vision benchmarks."""
    def __init__(self, nc: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, nc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


def create_model(config_name: str) -> Tuple[nn.Module, int]:
    """Create a model by config name. Returns (model, input_dim)."""
    if config_name == "small":
        return SmallModel(32, 10), 32
    elif config_name == "medium":
        return MediumModel(128, 10), 128
    elif config_name == "large":
        return LargeModel(512, 6, 10), 512
    elif config_name == "cnn":
        return CNNModel(10), -1  # Special handling for CNN
    else:
        return SmallModel(32, 10), 32


# ---------------------------------------------------------------------------
# Benchmarking functions
# ---------------------------------------------------------------------------

def measure_latency(
    model: nn.Module,
    input_fn: callable,
    warmup: int = 20,
    iterations: int = 100,
) -> List[float]:
    """Measure inference latency in milliseconds."""
    model.requires_grad_(False)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            model(input_fn())

    # Synchronize if on GPU
    if next(model.parameters(), torch.tensor(0.0)).is_cuda:
        torch.cuda.synchronize()

    latencies: List[float] = []
    for _ in range(iterations):
        inp = input_fn()
        if inp.is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            model(inp)
        if inp.is_cuda:
            torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)  # ms

    return latencies


def compute_latency_stats(
    name: str, batch_size: int, latencies: List[float]
) -> LatencyResult:
    """Compute statistics from raw latency measurements."""
    sorted_lat = sorted(latencies)
    n = len(sorted_lat)

    return LatencyResult(
        name=name,
        batch_size=batch_size,
        mean_ms=statistics.mean(sorted_lat),
        median_ms=statistics.median(sorted_lat),
        p95_ms=sorted_lat[int(n * 0.95)] if n > 1 else sorted_lat[0],
        p99_ms=sorted_lat[int(n * 0.99)] if n > 1 else sorted_lat[0],
        min_ms=sorted_lat[0],
        max_ms=sorted_lat[-1],
        std_ms=statistics.stdev(sorted_lat) if n > 1 else 0.0,
        throughput_samples_per_sec=batch_size * 1000.0 / statistics.mean(sorted_lat)
        if statistics.mean(sorted_lat) > 0 else 0.0,
        num_iterations=n,
    )


def measure_model_size(model: nn.Module, name: str) -> SizeResult:
    """Measure model parameter count and disk size."""
    param_count = sum(p.numel() for p in model.parameters())

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        torch.save(model.state_dict(), path)
        size_bytes = os.path.getsize(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)

    return SizeResult(
        name=name,
        param_count=param_count,
        size_bytes=size_bytes,
        size_mb=size_bytes / (1024 * 1024),
    )


def measure_export_time(
    model: nn.Module, sample_input: torch.Tensor, method: str
) -> ExportTimeResult:
    """Measure time to export a model."""
    start = time.monotonic()

    if method == "torchscript_trace":
        _ = torch.jit.trace(model, sample_input)
    elif method == "torchscript_script":
        try:
            _ = torch.jit.script(model)
        except Exception:
            pass
    elif method == "onnx":
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=True) as f:
            try:
                torch.onnx.export(model, sample_input, f.name, opset_version=17)
            except Exception:
                pass
    elif method == "dynamic_quant":
        _ = torch.quantization.quantize_dynamic(
            copy.deepcopy(model), {nn.Linear}, dtype=torch.qint8
        )

    elapsed = time.monotonic() - start
    return ExportTimeResult(name=method, export_time_seconds=elapsed)


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def benchmark_model_variant(
    model: nn.Module,
    name: str,
    input_fn: callable,
    batch_size: int,
    config: BenchmarkConfig,
) -> LatencyResult:
    """Benchmark a single model variant."""
    latencies = measure_latency(
        model, input_fn,
        warmup=config.warmup_iterations,
        iterations=config.benchmark_iterations,
    )
    return compute_latency_stats(name, batch_size, latencies)


def run_benchmarks(config: BenchmarkConfig) -> BenchmarkReport:
    """Run all benchmarks and produce a report."""
    import datetime

    report = BenchmarkReport(
        device=config.device,
        torch_version=torch.__version__,
        timestamp=datetime.datetime.now().isoformat(),
    )

    for model_name in config.model_configs:
        model_orig, input_dim = create_model(model_name)
        model_orig.requires_grad_(False)

        if config.verbose:
            print(f"\n{'='*60}")
            print(f"Benchmarking: {model_name} (input_dim={input_dim})")
            print(f"{'='*60}")

        # Measure model sizes
        report.size_results.append(measure_model_size(model_orig, f"{model_name}_fp32"))

        # Prepare variants
        variants: Dict[str, nn.Module] = {"fp32": model_orig}

        # TorchScript
        if config.include_torchscript:
            try:
                sample = torch.randn(1, input_dim)
                traced = torch.jit.trace(model_orig, sample)
                variants["torchscript"] = traced

                # Optimized TorchScript
                try:
                    frozen = torch.jit.freeze(copy.deepcopy(traced))
                    optimized = torch.jit.optimize_for_inference(frozen)
                    variants["ts_optimized"] = optimized
                except Exception:
                    pass

                # Measure export times
                report.export_times.append(
                    measure_export_time(model_orig, sample, "torchscript_trace")
                )
            except Exception as e:
                if config.verbose:
                    print(f"  TorchScript trace failed: {e}")

        # Dynamic INT8
        if config.include_quantized:
            try:
                q_model = torch.quantization.quantize_dynamic(
                    copy.deepcopy(model_orig), {nn.Linear}, dtype=torch.qint8
                )
                variants["dynamic_int8"] = q_model
                report.size_results.append(
                    measure_model_size(q_model, f"{model_name}_int8")
                )
                report.export_times.append(
                    measure_export_time(model_orig, torch.randn(1, input_dim), "dynamic_quant")
                )
            except Exception as e:
                if config.verbose:
                    print(f"  Dynamic INT8 failed: {e}")

        # FP16
        if config.include_fp16:
            try:
                fp16_model = copy.deepcopy(model_orig).half()
                variants["fp16"] = fp16_model
            except Exception as e:
                if config.verbose:
                    print(f"  FP16 failed: {e}")

        # Benchmark each variant at each batch size
        for variant_name, variant_model in variants.items():
            for bs in config.batch_sizes:
                if variant_name == "fp16":
                    input_fn = lambda: torch.randn(bs, input_dim).half()
                else:
                    input_fn = lambda: torch.randn(bs, input_dim)

                try:
                    result = benchmark_model_variant(
                        variant_model,
                        f"{model_name}/{variant_name}",
                        input_fn,
                        bs,
                        config,
                    )
                    report.latency_results.append(result)

                    if config.verbose:
                        print(
                            f"  {variant_name:15s} bs={bs:3d}  "
                            f"mean={result.mean_ms:7.2f}ms  "
                            f"p99={result.p99_ms:7.2f}ms  "
                            f"throughput={result.throughput_samples_per_sec:8.0f} samples/s"
                        )
                except Exception as e:
                    if config.verbose:
                        print(f"  {variant_name} bs={bs} FAILED: {e}")

        # ONNX export time
        try:
            report.export_times.append(
                measure_export_time(model_orig, torch.randn(1, input_dim), "onnx")
            )
        except Exception:
            pass

        gc.collect()

    return report


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_table(report: BenchmarkReport) -> str:
    """Format benchmark report as a human-readable table."""
    lines: List[str] = []

    lines.append("=" * 100)
    lines.append("EXPORT BENCHMARK REPORT")
    lines.append(f"PyTorch: {report.torch_version}  Device: {report.device}  Time: {report.timestamp}")
    lines.append("=" * 100)

    # Latency table
    lines.append("")
    lines.append("LATENCY RESULTS")
    lines.append("-" * 100)
    header = (
        f"{'Model/Variant':30s} {'BS':>4s} {'Mean':>8s} {'Median':>8s} "
        f"{'P95':>8s} {'P99':>8s} {'Std':>8s} {'Throughput':>12s}"
    )
    lines.append(header)
    lines.append("-" * 100)

    for r in report.latency_results:
        line = (
            f"{r.name:30s} {r.batch_size:4d} "
            f"{r.mean_ms:7.2f}ms {r.median_ms:7.2f}ms "
            f"{r.p95_ms:7.2f}ms {r.p99_ms:7.2f}ms "
            f"{r.std_ms:7.2f}ms "
            f"{r.throughput_samples_per_sec:10.0f}/s"
        )
        lines.append(line)

    # Size table
    lines.append("")
    lines.append("MODEL SIZES")
    lines.append("-" * 60)
    lines.append(f"{'Name':30s} {'Params':>12s} {'Size':>10s}")
    lines.append("-" * 60)

    for s in report.size_results:
        lines.append(
            f"{s.name:30s} {s.param_count:12,d} {s.size_mb:8.2f} MB"
        )

    # Export times
    if report.export_times:
        lines.append("")
        lines.append("EXPORT TIMES")
        lines.append("-" * 40)
        for et in report.export_times:
            lines.append(f"  {et.name:25s} {et.export_time_seconds:.3f}s")

    lines.append("")
    lines.append("=" * 100)
    return "\n".join(lines)


def format_json(report: BenchmarkReport) -> str:
    """Format benchmark report as JSON."""
    data = {
        "device": report.device,
        "torch_version": report.torch_version,
        "timestamp": report.timestamp,
        "latency_results": [asdict(r) for r in report.latency_results],
        "size_results": [asdict(s) for s in report.size_results],
        "export_times": [asdict(e) for e in report.export_times],
    }
    return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# Speedup analysis
# ---------------------------------------------------------------------------

def compute_speedups(report: BenchmarkReport) -> str:
    """Compute speedup ratios relative to FP32 baseline."""
    lines: List[str] = []
    lines.append("\nSPEEDUP ANALYSIS (vs FP32 baseline)")
    lines.append("-" * 70)

    # Group by model name and batch size
    baselines: Dict[Tuple[str, int], float] = {}
    for r in report.latency_results:
        parts = r.name.split("/")
        if len(parts) == 2 and parts[1] == "fp32":
            baselines[(parts[0], r.batch_size)] = r.mean_ms

    for r in report.latency_results:
        parts = r.name.split("/")
        if len(parts) == 2:
            model_name, variant = parts
            key = (model_name, r.batch_size)
            if key in baselines and baselines[key] > 0:
                speedup = baselines[key] / r.mean_ms
                lines.append(
                    f"  {r.name:30s} bs={r.batch_size:3d}  "
                    f"speedup={speedup:.2f}x"
                )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark exported models")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--iterations", "-n", type=int, default=100, help="Benchmark iterations")
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16, 32],
        help="Batch sizes to benchmark",
    )
    parser.add_argument(
        "--models", nargs="+", default=["small", "medium", "large"],
        help="Model configs to benchmark",
    )
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--format", choices=["table", "json"], default="table")
    parser.add_argument("--output", "-o", default=None, help="Output file path")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--no-quant", action="store_true", help="Skip quantization benchmarks")
    parser.add_argument("--no-fp16", action="store_true", help="Skip FP16 benchmarks")
    parser.add_argument("--no-torchscript", action="store_true", help="Skip TorchScript benchmarks")
    args = parser.parse_args()

    config = BenchmarkConfig(
        warmup_iterations=args.warmup,
        benchmark_iterations=args.iterations,
        batch_sizes=args.batch_sizes,
        model_configs=args.models,
        include_quantized=not args.no_quant,
        include_fp16=not args.no_fp16,
        include_torchscript=not args.no_torchscript,
        device=args.device,
        output_format=args.format,
        verbose=args.verbose,
    )

    print(f"Running benchmarks on {config.device}...")
    print(f"Models: {config.model_configs}")
    print(f"Batch sizes: {config.batch_sizes}")
    print(f"Iterations: {config.benchmark_iterations} (warmup: {config.warmup_iterations})")
    print()

    report = run_benchmarks(config)

    # Format output
    if config.output_format == "json":
        output = format_json(report)
    else:
        output = format_table(report)
        output += "\n" + compute_speedups(report)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"\nReport saved to: {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
