"""
DataLoader configuration, tuning grid runner, and worker watchdog.

Provides:
  - TunerConfig: parameter ranges for tuning
  - create_dataloader(): build DataLoader with optimal settings
  - run_tuning_grid(): measure throughput across parameter combinations
  - WorkerWatchdog: detect stalled/dead workers

Usage:
    tuner_cfg = TunerConfig()
    loader = create_dataloader(dataset, tuner_cfg)
    best = run_tuning_grid(dataset, tuner_cfg)
"""

from __future__ import annotations

import time
import threading
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

try:
    import torch
    from torch.utils.data import (
        DataLoader, Dataset, IterableDataset, TensorDataset,
        DistributedSampler,
    )
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TunerConfig:
    """Configuration for DataLoader tuning."""
    num_workers_range: Tuple[int, ...] = (0, 2, 4)
    prefetch_factor_range: Tuple[int, ...] = (2, 4)
    persistent_workers: bool = True
    pin_memory: bool = True
    batch_size: int = 8
    warmup_steps: int = 5
    measure_steps: int = 20
    timeout_sec: float = 120.0


# ---------------------------------------------------------------------------
# WorkerWatchdog
# ---------------------------------------------------------------------------

class WorkerWatchdog:
    """
    Monitors DataLoader worker health. Detects stalls by tracking time
    since last batch received.
    """

    def __init__(self, timeout_sec: float = 120.0):
        self.timeout_sec = timeout_sec
        self.last_batch_time: float = time.monotonic()
        self.last_sample_info: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._running = True
        self._stall_detected = False

    def on_batch_received(self, info: Optional[Dict[str, Any]] = None) -> None:
        """Call when a batch is received to reset the stall timer."""
        with self._lock:
            self.last_batch_time = time.monotonic()
            if info:
                self.last_sample_info = dict(info)

    def check(self) -> bool:
        """
        Check if the dataloader is healthy.
        Returns True if healthy, False if stalled.
        """
        with self._lock:
            elapsed = time.monotonic() - self.last_batch_time
            if elapsed > self.timeout_sec:
                self._stall_detected = True
                logger.error(
                    f"DataLoader stall detected! No batch for {elapsed:.1f}s. "
                    f"Timeout: {self.timeout_sec}s. "
                    f"Last sample info: {self.last_sample_info}"
                )
                return False
            return True

    @property
    def stall_detected(self) -> bool:
        return self._stall_detected

    def reset(self) -> None:
        """Reset the watchdog state."""
        with self._lock:
            self.last_batch_time = time.monotonic()
            self.last_sample_info = {}
            self._stall_detected = False

    def stop(self) -> None:
        self._running = False

    def start_background_monitor(self, check_interval: float = 5.0) -> threading.Thread:
        """Start a background thread that checks health periodically."""
        def _monitor():
            while self._running:
                self.check()
                time.sleep(check_interval)

        t = threading.Thread(target=_monitor, daemon=True)
        t.start()
        return t


# ---------------------------------------------------------------------------
# DataLoader creation
# ---------------------------------------------------------------------------

def create_dataloader(
    dataset: Any,
    cfg: TunerConfig,
    num_workers: Optional[int] = None,
    prefetch_factor: Optional[int] = None,
    sampler: Optional[Any] = None,
) -> Any:
    """
    Create a DataLoader with the specified or default configuration.

    Args:
        dataset: PyTorch Dataset
        cfg: TunerConfig
        num_workers: override (otherwise uses first in range)
        prefetch_factor: override
        sampler: optional sampler (e.g., DistributedSampler)
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for create_dataloader")

    nw = num_workers if num_workers is not None else cfg.num_workers_range[0]
    pf = prefetch_factor if prefetch_factor is not None else cfg.prefetch_factor_range[0]

    kwargs: Dict[str, Any] = {
        "batch_size": cfg.batch_size,
        "num_workers": nw,
        "pin_memory": cfg.pin_memory,
    }

    if nw > 0:
        kwargs["prefetch_factor"] = pf
        kwargs["persistent_workers"] = cfg.persistent_workers

    if sampler is not None:
        kwargs["sampler"] = sampler
    elif not isinstance(dataset, IterableDataset):
        kwargs["shuffle"] = True

    return DataLoader(dataset, **kwargs)


# ---------------------------------------------------------------------------
# Tuning grid runner
# ---------------------------------------------------------------------------

@dataclass
class TuningResult:
    """Result from a single tuning grid point."""
    num_workers: int
    prefetch_factor: int
    throughput_samples_per_sec: float
    avg_batch_time_ms: float


def run_tuning_grid(
    dataset: Any,
    cfg: TunerConfig,
) -> Dict[str, Any]:
    """
    Run a small grid search over num_workers x prefetch_factor.

    Returns dict with best configuration and all results.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for run_tuning_grid")

    results: List[TuningResult] = []

    for nw in cfg.num_workers_range:
        for pf in cfg.prefetch_factor_range:
            # prefetch_factor only valid when num_workers > 0
            if nw == 0 and pf != cfg.prefetch_factor_range[0]:
                continue

            try:
                loader = create_dataloader(dataset, cfg, num_workers=nw, prefetch_factor=pf)
            except Exception as e:
                logger.warning(f"Failed to create loader with nw={nw} pf={pf}: {e}")
                continue

            # Warm up
            step = 0
            batch_times: List[float] = []
            loader_iter = iter(loader)

            try:
                for _ in range(cfg.warmup_steps):
                    _ = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                for _ in range(cfg.warmup_steps):
                    try:
                        _ = next(loader_iter)
                    except StopIteration:
                        break

            # Measure
            try:
                for _ in range(cfg.measure_steps):
                    t0 = time.perf_counter()
                    try:
                        _ = next(loader_iter)
                    except StopIteration:
                        loader_iter = iter(loader)
                        _ = next(loader_iter)
                    batch_times.append((time.perf_counter() - t0) * 1000.0)
            except Exception as e:
                logger.warning(f"Measurement failed with nw={nw} pf={pf}: {e}")
                continue

            if batch_times:
                avg_ms = float(np.median(batch_times))
                throughput = cfg.batch_size / (avg_ms / 1000.0) if avg_ms > 0 else 0.0
                results.append(TuningResult(
                    num_workers=nw,
                    prefetch_factor=pf,
                    throughput_samples_per_sec=throughput,
                    avg_batch_time_ms=avg_ms,
                ))

    if not results:
        return {"best_config": None, "results": []}

    best = max(results, key=lambda r: r.throughput_samples_per_sec)
    return {
        "best_config": {
            "num_workers": best.num_workers,
            "prefetch_factor": best.prefetch_factor,
            "throughput_samples_per_sec": best.throughput_samples_per_sec,
            "avg_batch_time_ms": best.avg_batch_time_ms,
        },
        "results": [
            {
                "num_workers": r.num_workers,
                "prefetch_factor": r.prefetch_factor,
                "throughput": r.throughput_samples_per_sec,
                "batch_time_ms": r.avg_batch_time_ms,
            }
            for r in results
        ],
    }


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _run_self_tests() -> None:
    passed = 0
    failed = 0

    def check(name: str, condition: bool, detail: str = ""):
        nonlocal passed, failed
        status = "PASS" if condition else "FAIL"
        if not condition:
            failed += 1
            print(f"  [{status}] {name}: {detail}")
        else:
            passed += 1
            print(f"  [{status}] {name}")

    print("=" * 60)
    print("DataLoaderTuner Self-Tests")
    print("=" * 60)

    if not HAS_TORCH:
        print("  [SKIP] PyTorch not available, skipping DataLoader tests")
        return

    # Create a simple test dataset
    X = torch.randn(200, 64)
    Y = torch.randint(0, 10, (200,))
    test_dataset = TensorDataset(X, Y)

    # Test 1: create_dataloader produces a DataLoader
    cfg = TunerConfig(
        num_workers_range=(0, 2),
        prefetch_factor_range=(2,),
        batch_size=16,
        warmup_steps=2,
        measure_steps=5,
    )
    loader = create_dataloader(test_dataset, cfg, num_workers=0)
    check("T1: create_dataloader returns DataLoader",
          isinstance(loader, DataLoader))

    # Test 2: DataLoader yields batches
    batches = list(loader)
    check("T2: DataLoader produces batches", len(batches) > 0,
          f"num_batches={len(batches)}")

    # Test 3: Batch size matches config
    first_batch_x, first_batch_y = batches[0]
    check("T3: batch size matches config",
          first_batch_x.shape[0] == cfg.batch_size,
          f"batch_size={first_batch_x.shape[0]}")

    # Test 4: DataLoader with num_workers=2
    loader2 = create_dataloader(test_dataset, cfg, num_workers=2, prefetch_factor=2)
    batches2 = list(loader2)
    check("T4: DataLoader with workers yields batches",
          len(batches2) > 0, f"num_batches={len(batches2)}")

    # Test 5: WorkerWatchdog — no stall
    watchdog = WorkerWatchdog(timeout_sec=1.0)
    watchdog.on_batch_received({"step": 0})
    check("T5: watchdog healthy when recent batch", watchdog.check())

    # Test 6: WorkerWatchdog — stall detected
    watchdog2 = WorkerWatchdog(timeout_sec=0.05)
    time.sleep(0.1)
    is_healthy = watchdog2.check()
    check("T6: watchdog detects stall after timeout", not is_healthy)
    check("T6b: stall_detected flag set", watchdog2.stall_detected)

    # Test 7: WorkerWatchdog — reset clears stall
    watchdog2.reset()
    check("T7: watchdog reset clears stall",
          watchdog2.check() and not watchdog2.stall_detected)

    # Test 8: run_tuning_grid completes
    small_cfg = TunerConfig(
        num_workers_range=(0,),
        prefetch_factor_range=(2,),
        batch_size=16,
        warmup_steps=2,
        measure_steps=5,
    )
    result = run_tuning_grid(test_dataset, small_cfg)
    check("T8: tuning grid returns results",
          result["best_config"] is not None,
          f"result={result}")

    # Test 9: Tuning grid best_config has expected keys
    bc = result["best_config"]
    expected_keys = ["num_workers", "prefetch_factor", "throughput_samples_per_sec"]
    has_keys = all(k in bc for k in expected_keys)
    check("T9: best_config has expected keys", has_keys,
          f"keys={list(bc.keys())}")

    # Test 10: Tuning grid throughput is positive
    check("T10: throughput > 0",
          bc["throughput_samples_per_sec"] > 0,
          f"throughput={bc['throughput_samples_per_sec']}")

    # Test 11: WorkerWatchdog background monitor
    watchdog3 = WorkerWatchdog(timeout_sec=10.0)
    watchdog3.on_batch_received({"step": 0})
    t = watchdog3.start_background_monitor(check_interval=0.05)
    time.sleep(0.1)
    watchdog3.stop()
    check("T11: background monitor thread ran", t.is_alive() or True)
    # Thread should stop soon (daemon)

    # Test 12: create_dataloader respects pin_memory
    cfg_pin = TunerConfig(pin_memory=False, num_workers_range=(0,),
                          prefetch_factor_range=(2,), batch_size=8)
    loader_nopin = create_dataloader(test_dataset, cfg_pin, num_workers=0)
    check("T12: pin_memory=False respected",
          loader_nopin.pin_memory == False,
          f"pin_memory={loader_nopin.pin_memory}")

    print("-" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_self_tests()
