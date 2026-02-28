#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# MIT License
#
# assessment_benchmark.py
#
# Throughput and latency benchmarks for V-JEPA 2 assessment components.
# Measures:
#   1. Frozen probing throughput (samples/sec) vs number of probe heads
#   2. Focal loss computation throughput at EPIC-Kitchens scale
#   3. ClassMeanRecall update throughput
#   4. AttentiveClassifier latency vs embed_dim
#   5. ActionAnticipationClassifier throughput
#   6. Multi-head overhead as a function of N heads
#   7. Factory model construction time
#   8. Preprocessor throughput (if torchvision is available)
#
# Usage:
#   python scripts/assessment_benchmark.py                      # all benchmarks
#   python scripts/assessment_benchmark.py --task probing       # single task
#   python scripts/assessment_benchmark.py --device cuda        # GPU benchmarks
#   python scripts/assessment_benchmark.py --warmup 5           # warmup iterations
#   python scripts/assessment_benchmark.py --iters 50           # measurement iterations
#   python scripts/assessment_benchmark.py --output results.json # save JSON

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Import templates
# ---------------------------------------------------------------------------

SKILL_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = SKILL_ROOT / "assets"
sys.path.insert(0, str(ASSETS_DIR))

from attentive_classifier_template import AttentiveClassifier, AttentivePooler
from focal_loss_template import FocalLoss, ClassMeanRecall, ActionAnticipationClassifier
from frozen_assessor_template import FrozenBackboneAssessor
from model_hub_template import vjepa2_vit_large, _VideoViTEncoder, _VJEPAPredictor
from assessment_config_template import AssessmentConfig


# ---------------------------------------------------------------------------
# Benchmark infrastructure
# ---------------------------------------------------------------------------

class BenchmarkResult:
    """Holds timing and throughput results for a single benchmark."""

    def __init__(
        self,
        name: str,
        latency_ms: float,
        throughput: Optional[float] = None,
        unit: str = "samples/s",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name        = name
        self.latency_ms  = latency_ms
        self.throughput  = throughput
        self.unit        = unit
        self.extra       = extra or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name":        self.name,
            "latency_ms":  round(self.latency_ms, 3),
            "throughput":  round(self.throughput, 2) if self.throughput is not None else None,
            "unit":        self.unit,
            **self.extra,
        }

    def __repr__(self) -> str:
        tpt = f"{self.throughput:.1f} {self.unit}" if self.throughput is not None else ""
        return (
            f"[{self.name}]  latency={self.latency_ms:.2f}ms  {tpt}"
        )


def timed_run(
    fn: Callable,
    warmup: int = 3,
    iters: int = 20,
    device: Optional[torch.device] = None,
) -> float:
    """
    Run ``fn`` with ``warmup`` warmup calls, then ``iters`` timed calls.
    Returns the mean wall-clock time in milliseconds.
    """
    for _ in range(warmup):
        fn()
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)

    times = []
    for _ in range(iters):
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        fn()
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append((time.perf_counter() - t0) * 1000.0)

    return sum(times) / len(times)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def make_encoder_features(
    batch_size: int, num_patches: int, embed_dim: int, device: torch.device
) -> Tensor:
    return torch.randn(batch_size, num_patches, embed_dim, device=device)


# ---------------------------------------------------------------------------
# Benchmark 1: Frozen probing throughput
# ---------------------------------------------------------------------------

def benchmark_frozen_probing(
    device: torch.device,
    warmup: int,
    iters: int,
    batch_size: int = 32,
    num_patches: int = 196,
    embed_dim: int = 128,
    num_classes: int = 174,
) -> BenchmarkResult:
    """Measure samples/sec for single-head frozen probing (encoder + probe forward)."""

    class _Enc(nn.Module):
        def __init__(self, d: int):
            super().__init__()
            self.embed_dim = d
            self.fc = nn.Linear(d, d)
        def forward(self, x: Tensor) -> Tensor:
            return self.fc(x)

    encoder = _Enc(embed_dim).to(device)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.train(False)

    probe = AttentiveClassifier(embed_dim, num_classes).to(device)
    probe.train(False)

    features = make_encoder_features(batch_size, num_patches, embed_dim, device)

    def fn():
        with torch.no_grad():
            enc_out = encoder(features)
            _       = probe(enc_out)

    lat_ms  = timed_run(fn, warmup=warmup, iters=iters, device=device)
    samples_per_sec = (batch_size * 1000.0) / lat_ms

    return BenchmarkResult(
        name="frozen_probing_single_head",
        latency_ms=lat_ms,
        throughput=samples_per_sec,
        unit="samples/s",
        extra={"batch_size": batch_size, "embed_dim": embed_dim, "num_classes": num_classes},
    )


# ---------------------------------------------------------------------------
# Benchmark 2: Multi-head overhead
# ---------------------------------------------------------------------------

def benchmark_multihead_overhead(
    device: torch.device,
    warmup: int,
    iters: int,
    embed_dim: int = 128,
    num_patches: int = 196,
    num_classes: int = 174,
    batch_size: int = 32,
) -> List[BenchmarkResult]:
    """Measure time per step for 1, 2, 4, 8 simultaneous probe heads."""

    class _Enc(nn.Module):
        def __init__(self, d: int):
            super().__init__()
            self.embed_dim = d
            self.fc = nn.Linear(d, d)
        def forward(self, x: Tensor) -> Tensor:
            return self.fc(x)

    encoder = _Enc(embed_dim).to(device)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.train(False)

    features = make_encoder_features(batch_size, num_patches, embed_dim, device)
    labels   = torch.randint(0, num_classes, (batch_size,), device=device)

    results = []
    for n_heads in [1, 2, 4, 8]:
        probes = [
            AttentiveClassifier(embed_dim, num_classes).to(device)
            for _ in range(n_heads)
        ]
        optimizers = [
            torch.optim.SGD(p.parameters(), lr=1e-2) for p in probes
        ]

        def fn(probes=probes, optimizers=optimizers):
            with torch.no_grad():
                enc_out = encoder(features)
            for probe, opt in zip(probes, optimizers):
                logits = probe(enc_out)
                loss   = F.cross_entropy(logits, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()

        lat_ms = timed_run(fn, warmup=warmup, iters=iters, device=device)
        samples_per_sec = (batch_size * 1000.0) / lat_ms

        results.append(BenchmarkResult(
            name=f"multihead_n{n_heads}",
            latency_ms=lat_ms,
            throughput=samples_per_sec,
            unit="samples/s",
            extra={"num_heads": n_heads, "embed_dim": embed_dim},
        ))

    return results


# ---------------------------------------------------------------------------
# Benchmark 3: Focal loss throughput
# ---------------------------------------------------------------------------

def benchmark_focal_loss(
    device: torch.device,
    warmup: int,
    iters: int,
    batch_size: int = 64,
    num_classes: int = 3806,
) -> BenchmarkResult:
    """FocalLoss throughput at EPIC-Kitchens 100 scale (3806 classes)."""
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0).to(device)
    logits  = torch.randn(batch_size, num_classes, device=device)
    targets = torch.randint(0, num_classes, (batch_size,), device=device)

    def fn():
        _ = loss_fn(logits, targets)

    lat_ms  = timed_run(fn, warmup=warmup, iters=iters, device=device)
    samples_per_sec = (batch_size * 1000.0) / lat_ms

    return BenchmarkResult(
        name="focal_loss_epic_scale",
        latency_ms=lat_ms,
        throughput=samples_per_sec,
        unit="samples/s",
        extra={"batch_size": batch_size, "num_classes": num_classes},
    )


# ---------------------------------------------------------------------------
# Benchmark 4: ClassMeanRecall update
# ---------------------------------------------------------------------------

def benchmark_class_mean_recall(
    device: torch.device,
    warmup: int,
    iters: int,
    batch_size: int = 64,
    num_classes: int = 300,
) -> BenchmarkResult:
    """ClassMeanRecall.update_topk() throughput."""
    metric  = ClassMeanRecall(num_classes=num_classes)
    logits  = torch.randn(batch_size, num_classes)   # CPU (metric is CPU-based)
    targets = torch.randint(0, num_classes, (batch_size,))

    def fn():
        metric.update_topk(logits, targets, k=5)

    lat_ms  = timed_run(fn, warmup=warmup, iters=iters, device=None)
    samples_per_sec = (batch_size * 1000.0) / lat_ms

    return BenchmarkResult(
        name="class_mean_recall_update",
        latency_ms=lat_ms,
        throughput=samples_per_sec,
        unit="samples/s",
        extra={"batch_size": batch_size, "num_classes": num_classes, "topk": 5},
    )


# ---------------------------------------------------------------------------
# Benchmark 5: AttentiveClassifier latency vs embed_dim
# ---------------------------------------------------------------------------

def benchmark_attentive_classifier_vs_embed_dim(
    device: torch.device,
    warmup: int,
    iters: int,
    batch_size: int = 16,
    num_patches: int = 196,
    num_classes: int = 174,
) -> List[BenchmarkResult]:
    """AttentiveClassifier forward latency for different embed_dims."""
    results = []
    for embed_dim in [128, 256, 512, 1024, 1408]:
        model = AttentiveClassifier(embed_dim=embed_dim, num_classes=num_classes).to(device)
        model.train(False)
        x = make_encoder_features(batch_size, num_patches, embed_dim, device)

        def fn(m=model, x=x):
            with torch.no_grad():
                return m(x)

        lat_ms  = timed_run(fn, warmup=warmup, iters=iters, device=device)
        samples_per_sec = (batch_size * 1000.0) / lat_ms

        results.append(BenchmarkResult(
            name=f"attentive_classifier_D{embed_dim}",
            latency_ms=lat_ms,
            throughput=samples_per_sec,
            unit="samples/s",
            extra={"embed_dim": embed_dim, "num_patches": num_patches},
        ))

    return results


# ---------------------------------------------------------------------------
# Benchmark 6: ActionAnticipationClassifier
# ---------------------------------------------------------------------------

def benchmark_action_anticipation(
    device: torch.device,
    warmup: int,
    iters: int,
    batch_size: int = 16,
    num_patches: int = 64,
    embed_dim: int = 128,
) -> BenchmarkResult:
    """ActionAnticipationClassifier 3-output forward throughput."""
    model = ActionAnticipationClassifier(
        embed_dim=embed_dim,
        num_verbs=97,
        num_nouns=300,
        num_actions=3806,
    ).to(device)
    model.train(False)
    x = make_encoder_features(batch_size, num_patches, embed_dim, device)

    def fn():
        with torch.no_grad():
            return model(x)

    lat_ms  = timed_run(fn, warmup=warmup, iters=iters, device=device)
    samples_per_sec = (batch_size * 1000.0) / lat_ms

    return BenchmarkResult(
        name="action_anticipation_classifier",
        latency_ms=lat_ms,
        throughput=samples_per_sec,
        unit="samples/s",
        extra={"embed_dim": embed_dim, "num_patches": num_patches},
    )


# ---------------------------------------------------------------------------
# Benchmark 7: Factory model construction time
# ---------------------------------------------------------------------------

def benchmark_factory_construction(
    device: torch.device,
    warmup: int,
    iters: int,
) -> List[BenchmarkResult]:
    """Time to construct encoder + predictor from factory (no pretrained weights)."""
    results = []

    def build_large():
        return vjepa2_vit_large(pretrained=False)

    def build_small():
        enc  = _VideoViTEncoder(embed_dim=64, depth=2, num_heads=4)
        pred = _VJEPAPredictor(context_embed_dim=64, embed_dim=32, depth=2, num_heads=4)
        return enc, pred

    for name, fn in [("factory_vit_large", build_large), ("factory_small_proxy", build_small)]:
        lat_ms = timed_run(fn, warmup=max(1, warmup // 2), iters=max(5, iters // 4), device=None)
        results.append(BenchmarkResult(
            name=name,
            latency_ms=lat_ms,
            throughput=None,
            unit="",
            extra={"construction_time_ms": round(lat_ms, 2)},
        ))

    return results


# ---------------------------------------------------------------------------
# Benchmark 8: Preprocessor
# ---------------------------------------------------------------------------

def benchmark_preprocessor(
    device: torch.device,
    warmup: int,
    iters: int,
    batch_size: int = 8,
    frames_per_clip: int = 16,
    crop_size: int = 224,
) -> Optional[BenchmarkResult]:
    """vjepa2_preprocessor throughput for [C, T, H, W] video tensors."""
    try:
        from model_hub_template import vjepa2_preprocessor
        prep = vjepa2_preprocessor(crop_size=crop_size, frames_per_clip=frames_per_clip)
    except ImportError:
        return None

    # Simulate a batch: process each video independently
    videos = [torch.rand(3, 32, 256, 256) for _ in range(batch_size)]

    def fn():
        _ = [prep(v) for v in videos]

    lat_ms  = timed_run(fn, warmup=warmup, iters=iters, device=None)
    clips_per_sec = (batch_size * 1000.0) / lat_ms

    return BenchmarkResult(
        name="preprocessor",
        latency_ms=lat_ms,
        throughput=clips_per_sec,
        unit="clips/s",
        extra={"batch_size": batch_size, "frames_per_clip": frames_per_clip, "crop_size": crop_size},
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

TASK_MAP = {
    "probing":         benchmark_frozen_probing,
    "multihead":       benchmark_multihead_overhead,
    "focal":           benchmark_focal_loss,
    "recall":          benchmark_class_mean_recall,
    "classifier":      benchmark_attentive_classifier_vs_embed_dim,
    "anticipation":    benchmark_action_anticipation,
    "factory":         benchmark_factory_construction,
    "preprocessor":    benchmark_preprocessor,
}


def print_results(results: List[BenchmarkResult]) -> None:
    col_w = max(len(r.name) for r in results) + 2
    print(f"\n{'Benchmark':<{col_w}}  {'Latency (ms)':>14}  {'Throughput':>16}")
    print("-" * (col_w + 36))
    for r in results:
        tpt = f"{r.throughput:.1f} {r.unit}" if r.throughput is not None else "—"
        print(f"{r.name:<{col_w}}  {r.latency_ms:>14.2f}  {tpt:>16}")
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark V-JEPA 2 assessment components"
    )
    p.add_argument(
        "--task", choices=list(TASK_MAP.keys()) + ["all"], default="all",
        help="Which benchmark to run (default: all)."
    )
    p.add_argument(
        "--device", choices=["cpu", "cuda"], default="cpu",
        help="Device to run benchmarks on (default: cpu)."
    )
    p.add_argument(
        "--warmup", type=int, default=3,
        help="Number of warmup iterations (default: 3)."
    )
    p.add_argument(
        "--iters", type=int, default=20,
        help="Number of timed iterations (default: 20)."
    )
    p.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for throughput benchmarks (default: 32)."
    )
    p.add_argument(
        "--output", "-o", type=str, default=None,
        help="Save results to JSON file."
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Falling back to CPU.")
        args.device = "cpu"

    device  = torch.device(args.device)
    warmup  = args.warmup
    iters   = args.iters
    bs      = args.batch_size

    print(f"\nV-JEPA 2 Assessment Benchmarks")
    print(f"Device:  {device}  ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Warmup:  {warmup} iterations")
    print(f"Measure: {iters} iterations")
    print(f"Batch:   {bs} samples")

    all_results: List[BenchmarkResult] = []

    tasks_to_run = list(TASK_MAP.keys()) if args.task == "all" else [args.task]

    for task_name in tasks_to_run:
        print(f"\nRunning: {task_name} ...", flush=True)
        try:
            fn = TASK_MAP[task_name]

            # Call with compatible arguments
            if task_name in ("factory", "preprocessor"):
                out = fn(device=device, warmup=warmup, iters=iters)
            elif task_name == "recall":
                out = fn(device=device, warmup=warmup, iters=iters, batch_size=bs)
            else:
                out = fn(device=device, warmup=warmup, iters=iters, batch_size=bs)

            if out is None:
                print(f"  Skipped (dependency not available).")
                continue

            if isinstance(out, list):
                for r in out:
                    all_results.append(r)
                    print(f"  {r}")
            else:
                all_results.append(out)
                print(f"  {out}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print_results(all_results)

    if args.output:
        data = {
            "device":   args.device,
            "warmup":   warmup,
            "iters":    iters,
            "results":  [r.to_dict() for r in all_results],
        }
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
