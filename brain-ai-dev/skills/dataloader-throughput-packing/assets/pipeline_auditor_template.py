"""
PipelineAuditor — Timer instrumentation for training pipeline measurement.

Measures t_data, t_fwd_bwd_opt, t_total_step, data_stall_ratio, and CUDA
host-to-device transfer attribution. Emits data_metrics.json with p50/p90
aggregation.

Usage:
    auditor = PipelineAuditor(device)
    for batch in loader:
        auditor.mark_data_start()
        batch = to_device(batch, device)
        auditor.mark_data_end()
        auditor.mark_compute_start()
        train_step(batch)
        auditor.mark_compute_end()
    metrics = auditor.report()
    auditor.to_json("data_metrics.json")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np

try:
    import torch
    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_CUDA = False


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PipelineMetrics:
    """Aggregated pipeline metrics with percentile summaries."""
    steps_measured: int = 0
    t_data_p50_ms: float = 0.0
    t_data_p90_ms: float = 0.0
    t_fwd_bwd_opt_p50_ms: float = 0.0
    t_fwd_bwd_opt_p90_ms: float = 0.0
    t_total_step_p50_ms: float = 0.0
    t_total_step_p90_ms: float = 0.0
    t_h2d_p50_ms: float = 0.0
    t_h2d_p90_ms: float = 0.0
    data_stall_ratio_p50: float = 0.0
    data_stall_ratio_p90: float = 0.0
    gpu_busy_ratio_p50: float = 0.0
    raw_tokens_per_sec_p50: float = 0.0
    effective_tokens_per_sec_p50: float = 0.0
    padding_ratio: float = 0.0
    dataloader_settings: Dict[str, Any] = field(default_factory=dict)
    packing_settings: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# PipelineAuditor
# ---------------------------------------------------------------------------

class PipelineAuditor:
    """Instruments the training loop to measure data stalls and throughput."""

    def __init__(self, device: Optional[Any] = None):
        """
        Args:
            device: torch.device or string. If CUDA device, uses CUDA events
                    for GPU-side timing.
        """
        self._device = device
        self._use_cuda = False
        if HAS_TORCH and device is not None:
            dev = torch.device(device) if isinstance(device, str) else device
            self._use_cuda = dev.type == "cuda" and HAS_CUDA

        # Per-step raw timings (in milliseconds)
        self._t_data_ms: List[float] = []
        self._t_compute_ms: List[float] = []
        self._t_total_ms: List[float] = []
        self._t_h2d_ms: List[float] = []
        self._stall_ratios: List[float] = []

        # Token accounting (set externally per step)
        self._raw_tokens: List[int] = []
        self._effective_tokens: List[int] = []

        # Transient state for current step
        self._data_start: Optional[float] = None
        self._data_end: Optional[float] = None
        self._compute_start: Optional[float] = None
        self._compute_end: Optional[float] = None
        self._step_start: Optional[float] = None

        # CUDA events
        self._cuda_data_start = None
        self._cuda_data_end = None
        self._cuda_compute_start = None
        self._cuda_compute_end = None

        # Optional metadata
        self.dataloader_settings: Dict[str, Any] = {}
        self.packing_settings: Dict[str, Any] = {}
        self.padding_ratio: float = 0.0

    # ---- Mark methods ----

    def mark_data_start(self) -> None:
        """Call immediately when requesting the next batch."""
        self._data_start = time.perf_counter()
        if self._step_start is None:
            self._step_start = self._data_start
        if self._use_cuda:
            self._cuda_data_start = torch.cuda.Event(enable_timing=True)
            self._cuda_data_start.record()

    def mark_data_end(self) -> None:
        """Call after batch is confirmed on-device (after synchronize)."""
        self._data_end = time.perf_counter()
        if self._use_cuda:
            self._cuda_data_end = torch.cuda.Event(enable_timing=True)
            self._cuda_data_end.record()

    def mark_compute_start(self) -> None:
        """Call right before the forward pass."""
        self._compute_start = time.perf_counter()
        if self._use_cuda:
            self._cuda_compute_start = torch.cuda.Event(enable_timing=True)
            self._cuda_compute_start.record()

    def mark_compute_end(self) -> None:
        """Call after backward+optimizer, after synchronize."""
        self._compute_end = time.perf_counter()
        if self._use_cuda:
            self._cuda_compute_end = torch.cuda.Event(enable_timing=True)
            self._cuda_compute_end.record()
        self._finalize_step()

    def record_tokens(self, raw: int, effective: int) -> None:
        """Record token counts for the current step."""
        self._raw_tokens.append(raw)
        self._effective_tokens.append(effective)

    def record_h2d_time(self, h2d_ms: float) -> None:
        """Record host-to-device transfer time for the current step."""
        self._t_h2d_ms.append(h2d_ms)

    # ---- Internal ----

    def _finalize_step(self) -> None:
        """Compute and store per-step metrics."""
        if self._data_start is None or self._data_end is None:
            return
        if self._compute_start is None or self._compute_end is None:
            return

        t_data = (self._data_end - self._data_start) * 1000.0
        t_compute = (self._compute_end - self._compute_start) * 1000.0
        t_total = (self._compute_end - (self._step_start or self._data_start)) * 1000.0

        self._t_data_ms.append(max(t_data, 0.0))
        self._t_compute_ms.append(max(t_compute, 0.0))
        self._t_total_ms.append(max(t_total, 0.001))  # avoid div-by-zero

        ratio = t_data / t_total if t_total > 0 else 0.0
        self._stall_ratios.append(min(max(ratio, 0.0), 1.0))

        # Reset transient state
        self._data_start = None
        self._data_end = None
        self._compute_start = None
        self._compute_end = None
        self._step_start = None

    # ---- Reporting ----

    def report(self) -> PipelineMetrics:
        """Aggregate collected metrics into PipelineMetrics."""
        n = len(self._t_data_ms)
        if n == 0:
            return PipelineMetrics()

        def _p50(arr):
            return float(np.percentile(arr, 50)) if arr else 0.0

        def _p90(arr):
            return float(np.percentile(arr, 90)) if arr else 0.0

        stall_p50 = _p50(self._stall_ratios)
        stall_p90 = _p90(self._stall_ratios)

        # Tokens per sec
        raw_tps = []
        eff_tps = []
        for i in range(min(n, len(self._raw_tokens))):
            step_sec = self._t_total_ms[i] / 1000.0
            if step_sec > 0:
                if i < len(self._raw_tokens):
                    raw_tps.append(self._raw_tokens[i] / step_sec)
                if i < len(self._effective_tokens):
                    eff_tps.append(self._effective_tokens[i] / step_sec)

        return PipelineMetrics(
            steps_measured=n,
            t_data_p50_ms=_p50(self._t_data_ms),
            t_data_p90_ms=_p90(self._t_data_ms),
            t_fwd_bwd_opt_p50_ms=_p50(self._t_compute_ms),
            t_fwd_bwd_opt_p90_ms=_p90(self._t_compute_ms),
            t_total_step_p50_ms=_p50(self._t_total_ms),
            t_total_step_p90_ms=_p90(self._t_total_ms),
            t_h2d_p50_ms=_p50(self._t_h2d_ms),
            t_h2d_p90_ms=_p90(self._t_h2d_ms),
            data_stall_ratio_p50=stall_p50,
            data_stall_ratio_p90=stall_p90,
            gpu_busy_ratio_p50=1.0 - stall_p50,
            raw_tokens_per_sec_p50=_p50(raw_tps),
            effective_tokens_per_sec_p50=_p50(eff_tps),
            padding_ratio=self.padding_ratio,
            dataloader_settings=dict(self.dataloader_settings),
            packing_settings=dict(self.packing_settings),
        )

    def to_json(self, path: str) -> None:
        """Write metrics to a JSON file."""
        metrics = self.report()
        data = {
            "version": "1.0",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **metrics.to_dict(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _run_self_tests() -> None:
    import tempfile
    import os

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
    print("PipelineAuditor Self-Tests")
    print("=" * 60)

    # Test 1: Basic timer produces positive t_data
    auditor = PipelineAuditor(device=None)
    auditor.mark_data_start()
    time.sleep(0.01)
    auditor.mark_data_end()
    auditor.mark_compute_start()
    time.sleep(0.01)
    auditor.mark_compute_end()
    m = auditor.report()
    check("T1: t_data is positive", m.t_data_p50_ms > 0,
          f"t_data_p50={m.t_data_p50_ms}")

    # Test 2: t_fwd_bwd_opt is positive
    check("T2: t_fwd_bwd_opt is positive", m.t_fwd_bwd_opt_p50_ms > 0,
          f"t_fwd_bwd={m.t_fwd_bwd_opt_p50_ms}")

    # Test 3: t_total >= t_data (approximately)
    check("T3: t_total >= t_data",
          m.t_total_step_p50_ms >= m.t_data_p50_ms * 0.9,
          f"total={m.t_total_step_p50_ms}, data={m.t_data_p50_ms}")

    # Test 4: data_stall_ratio in [0, 1]
    check("T4: stall ratio in [0, 1]",
          0.0 <= m.data_stall_ratio_p50 <= 1.0,
          f"ratio={m.data_stall_ratio_p50}")

    # Test 5: gpu_busy_ratio = 1 - stall_ratio
    check("T5: gpu_busy = 1 - stall",
          abs(m.gpu_busy_ratio_p50 - (1.0 - m.data_stall_ratio_p50)) < 1e-6,
          f"busy={m.gpu_busy_ratio_p50}")

    # Test 6: Multiple steps aggregate correctly
    auditor2 = PipelineAuditor(device=None)
    for _ in range(10):
        auditor2.mark_data_start()
        time.sleep(0.002)
        auditor2.mark_data_end()
        auditor2.mark_compute_start()
        time.sleep(0.005)
        auditor2.mark_compute_end()
    m2 = auditor2.report()
    check("T6: steps_measured == 10", m2.steps_measured == 10,
          f"steps={m2.steps_measured}")

    # Test 7: p90 >= p50 for stall ratios
    check("T7: p90 >= p50 stall ratio",
          m2.data_stall_ratio_p90 >= m2.data_stall_ratio_p50 - 1e-6,
          f"p90={m2.data_stall_ratio_p90}, p50={m2.data_stall_ratio_p50}")

    # Test 8: to_json produces valid JSON
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        auditor2.to_json(tmp_path)
        with open(tmp_path) as f:
            data = json.load(f)
        check("T8: to_json produces valid JSON", True)
        # Verify required fields
        required = ["version", "steps_measured", "t_data_p50_ms",
                     "data_stall_ratio_p50", "data_stall_ratio_p90"]
        all_present = all(k in data for k in required)
        check("T8b: JSON has required fields", all_present,
              f"missing: {[k for k in required if k not in data]}")
    finally:
        os.unlink(tmp_path)

    # Test 9: Token accounting
    auditor3 = PipelineAuditor(device=None)
    for i in range(5):
        auditor3.mark_data_start()
        time.sleep(0.001)
        auditor3.mark_data_end()
        auditor3.mark_compute_start()
        time.sleep(0.005)
        auditor3.mark_compute_end()
        auditor3.record_tokens(raw=2048, effective=1800)
    m3 = auditor3.report()
    check("T9: raw_tokens_per_sec > 0", m3.raw_tokens_per_sec_p50 > 0,
          f"raw_tps={m3.raw_tokens_per_sec_p50}")
    check("T9b: effective_tokens_per_sec > 0",
          m3.effective_tokens_per_sec_p50 > 0,
          f"eff_tps={m3.effective_tokens_per_sec_p50}")

    # Test 10: h2d recording
    auditor4 = PipelineAuditor(device=None)
    auditor4.mark_data_start()
    time.sleep(0.001)
    auditor4.mark_data_end()
    auditor4.record_h2d_time(1.5)
    auditor4.mark_compute_start()
    time.sleep(0.001)
    auditor4.mark_compute_end()
    auditor4.record_h2d_time(2.0)
    # need a second step to have the record
    auditor4.mark_data_start()
    time.sleep(0.001)
    auditor4.mark_data_end()
    auditor4.mark_compute_start()
    time.sleep(0.001)
    auditor4.mark_compute_end()
    m4 = auditor4.report()
    check("T10: h2d_p50 > 0", m4.t_h2d_p50_ms > 0,
          f"h2d_p50={m4.t_h2d_p50_ms}")

    # Test 11: Empty auditor returns zero metrics
    empty_auditor = PipelineAuditor(device=None)
    me = empty_auditor.report()
    check("T11: empty auditor steps_measured == 0", me.steps_measured == 0)
    check("T11b: empty auditor stall ratio == 0",
          me.data_stall_ratio_p50 == 0.0)

    # Test 12: PipelineMetrics to_dict round-trip
    m_dict = m2.to_dict()
    check("T12: to_dict returns dict", isinstance(m_dict, dict))
    check("T12b: dict has steps_measured", "steps_measured" in m_dict)

    print("-" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_self_tests()
