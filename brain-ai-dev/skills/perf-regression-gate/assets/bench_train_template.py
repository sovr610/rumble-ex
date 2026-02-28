"""
bench_train_template.py
========================
BenchTrain class for measuring training throughput with correct CUDA synchronization,
MFU estimation, and aggregated statistics.

Usage:
    from bench_train_template import BenchTrain
    from perf_gate_config_template import BenchConfig

    config = BenchConfig(warmup_steps=10, measure_steps=20, mode="synthetic")
    bench = BenchTrain(config)
    result = bench.run()
    bench.save("artifacts/metrics.json")

Self-test:
    python bench_train_template.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# Import config/result types (allow running standalone too)
try:
    from perf_gate_config_template import (
        BenchConfig, BenchResult, Status, GPU_PEAK_TFLOPS, get_peak_tflops
    )
except ImportError:
    # Minimal stubs for standalone testing
    from dataclasses import dataclass, field
    from enum import Enum
    from typing import Optional as Opt

    @dataclass
    class BenchConfig:
        warmup_steps: int = 5
        measure_steps: int = 10
        mode: str = "synthetic"
        world_size: int = 1
        profile: str = "off"
        repeat: int = 1
        peak_tflops: Optional[float] = None
        mfu_estimator: str = "6ND"
        per_device_batch_size: int = 2
        seq_len: int = 64
        grad_accum_steps: int = 1
        dtype: str = "bfloat16"
        vocab_size: int = 1000
        hidden_dim: int = 128
        num_layers: int = 2

        @property
        def tokens_per_step(self):
            return self.per_device_batch_size * max(1, self.world_size) * self.seq_len * self.grad_accum_steps

        def validate(self):
            pass

    @dataclass
    class BenchResult:
        step_times: List[float]
        step_time_mean: float
        step_time_p50: float
        step_time_p90: float
        step_time_min: float
        step_time_max: float
        tokens_per_sec_mean: float
        tokens_per_sec_p50: float
        tokens_per_sec_p10: float
        tokens_per_step: int
        peak_allocated_bytes: int
        peak_allocated_gb: float
        peak_reserved_bytes: int
        peak_reserved_gb: float
        mfu_p50: Optional[float]
        mfu_mean: Optional[float]
        n_non_embedding_params: int
        peak_tflops_per_gpu: Optional[float]
        mfu_estimator: str
        loss_values: List[float]
        final_loss: Optional[float]
        loss_slope: Optional[float]
        machine_profile: str
        run_config: dict

        def to_dict(self):
            import time as t
            return {
                "schema_version": "1.0",
                "machine_profile": self.machine_profile,
                "run_config": self.run_config,
                "throughput": {
                    "tokens_per_sec_mean": self.tokens_per_sec_mean,
                    "tokens_per_sec_p50": self.tokens_per_sec_p50,
                    "tokens_per_sec_p10": self.tokens_per_sec_p10,
                    "tokens_per_step": self.tokens_per_step,
                },
                "timing": {
                    "step_time_mean_s": self.step_time_mean,
                    "step_time_p50_s": self.step_time_p50,
                    "step_time_p90_s": self.step_time_p90,
                    "step_time_min_s": self.step_time_min,
                    "step_time_max_s": self.step_time_max,
                },
                "memory": {
                    "peak_allocated_bytes": self.peak_allocated_bytes,
                    "peak_allocated_gb": self.peak_allocated_gb,
                    "peak_reserved_bytes": self.peak_reserved_bytes,
                    "peak_reserved_gb": self.peak_reserved_gb,
                },
                "mfu": {
                    "mfu_p50": self.mfu_p50,
                    "mfu_mean": self.mfu_mean,
                    "n_non_embedding_params": self.n_non_embedding_params,
                    "peak_tflops_per_gpu": self.peak_tflops_per_gpu,
                    "estimator": self.mfu_estimator,
                },
                "loss": {
                    "final_loss": self.final_loss,
                    "loss_slope": self.loss_slope,
                    "loss_values": self.loss_values,
                },
                "timestamp": t.strftime("%Y-%m-%dT%H:%M:%SZ", t.gmtime()),
            }


def get_peak_tflops(device_name: str, dtype: str = "bf16") -> Optional[float]:
    """Fallback lookup for standalone use."""
    table = {
        "H100 SXM": 989.0, "H100": 989.0, "A100": 312.0,
        "L40S": 362.0, "RTX 4090": 330.0, "V100": 125.0,
    }
    for k, v in table.items():
        if k in device_name:
            return v
    return None


# ===========================================================================
# Minimal model for synthetic benchmarking
# ===========================================================================

def _build_minimal_model(
    vocab_size: int = 1000,
    hidden_dim: int = 128,
    num_layers: int = 2,
) -> "nn.Module":
    """Build a minimal transformer-like model for synthetic benchmarking."""
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for BenchTrain")

    class MinimalTransformerLayer(nn.Module):
        def __init__(self, hidden: int) -> None:
            super().__init__()
            self.attn = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
            self.ff1 = nn.Linear(hidden, hidden * 4)
            self.ff2 = nn.Linear(hidden * 4, hidden)
            self.norm1 = nn.LayerNorm(hidden)
            self.norm2 = nn.LayerNorm(hidden)
            self.act = nn.GELU()

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            attn_out, _ = self.attn(x, x, x)
            x = self.norm1(x + attn_out)
            ff_out = self.ff2(self.act(self.ff1(x)))
            x = self.norm2(x + ff_out)
            return x

    class MinimalLM(nn.Module):
        def __init__(self, vocab: int, hidden: int, layers: int) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocab, hidden)
            self.layers = nn.ModuleList([
                MinimalTransformerLayer(hidden) for _ in range(layers)
            ])
            self.head = nn.Linear(hidden, vocab, bias=False)

        def forward(
            self,
            input_ids: "torch.Tensor",
            labels: Optional["torch.Tensor"] = None,
        ) -> Dict[str, "torch.Tensor"]:
            x = self.embed(input_ids)
            for layer in self.layers:
                x = layer(x)
            logits = self.head(x)
            result = {"logits": logits}
            if labels is not None:
                loss_fn = nn.CrossEntropyLoss()
                # Flatten for cross-entropy
                result["loss"] = loss_fn(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                )
            return result

    return MinimalLM(vocab_size, hidden_dim, num_layers)


def _count_non_embedding_params(model: "nn.Module") -> int:
    """Count all parameters excluding Embedding layers."""
    total = sum(p.numel() for p in model.parameters())
    embed_params = sum(
        p.numel()
        for m in model.modules()
        if isinstance(m, nn.Embedding)
        for p in m.parameters()
    )
    return total - embed_params


# ===========================================================================
# BenchTrain
# ===========================================================================

class BenchTrain:
    """
    Training throughput benchmark with proper CUDA synchronization timing.

    Protocol:
        1. Reset peak memory stats
        2. Run warmup_steps (discarded)
        3. CUDA sync before each timed step
        4. Measure step_time = time after step - time before step
        5. CUDA sync after each step
        6. Aggregate statistics
        7. Compute MFU

    Example:
        config = BenchConfig(warmup_steps=50, measure_steps=100)
        bench = BenchTrain(config)
        result = bench.run()
        print(f"Throughput: {result.tokens_per_sec_p50:.0f} tok/s")
        print(f"MFU: {result.mfu_p50:.3f}")
    """

    def __init__(
        self,
        config: BenchConfig,
        model: Optional["nn.Module"] = None,
        optimizer: Optional["torch.optim.Optimizer"] = None,
        data_iter: Optional[Iterator] = None,
        machine_profile: str = "unknown",
    ) -> None:
        """
        Args:
            config:          Benchmark configuration.
            model:           Pre-built model. If None, a minimal model is created.
            optimizer:       Pre-built optimizer. If None, AdamW is created.
            data_iter:       Data iterator for e2e mode. Required if mode="e2e".
            machine_profile: Machine profile string for output tagging.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for BenchTrain")

        config.validate()
        self.config = config
        self.machine_profile = machine_profile
        self.data_iter = data_iter

        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Determine effective world_size
        self._effective_world_size = self._resolve_world_size()

        # Model and optimizer
        if model is None:
            self.model = _build_minimal_model(
                vocab_size=config.vocab_size,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
            ).to(self.device)
        else:
            self.model = model

        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=1e-4, weight_decay=0.1
            )
        else:
            self.optimizer = optimizer

        # Peak TFLOPS
        self._peak_tflops = self._resolve_peak_tflops()

        # Storage for result
        self._result: Optional[BenchResult] = None

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def run(self) -> BenchResult:
        """
        Execute the full benchmark: warmup -> measure -> aggregate.

        Returns:
            BenchResult with all timing, throughput, memory, and MFU metrics.
        """
        best_result: Optional[BenchResult] = None

        for repeat_idx in range(self.config.repeat):
            result = self._single_run()
            if best_result is None or result.step_time_p50 < best_result.step_time_p50:
                best_result = result

        self._result = best_result
        return best_result

    def save(self, path: str) -> None:
        """
        Write metrics.json atomically to path.

        Args:
            path: Destination file. Parent directories are created.
        """
        if self._result is None:
            raise RuntimeError("Call run() before save()")

        metrics = self._result.to_dict()
        metrics["env"] = self._collect_env_summary()

        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=dest.parent, suffix=".tmp", prefix=dest.stem
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(metrics, f, indent=2, default=str)
            os.replace(tmp_path, dest)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # -----------------------------------------------------------------------
    # Internal: single benchmark run
    # -----------------------------------------------------------------------

    def _single_run(self) -> BenchResult:
        """Run one warmup + measure cycle."""
        self.model.train()

        # Step 1: Reset memory stats before warmup
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # Step 2: Warmup (discarded from measurement)
        self._warmup()

        # Step 3: Reset memory again after warmup (warmup may hold temp allocs)
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # Step 4: Measurement
        step_times, loss_values = self._measure()

        # Step 5: Read memory
        peak_alloc, peak_reserved = self._read_memory()

        # Step 6: Aggregate
        return self._build_result(step_times, loss_values, peak_alloc, peak_reserved)

    def _warmup(self) -> None:
        """Run warmup steps without timing."""
        for _ in range(self.config.warmup_steps):
            batch = self._get_batch()
            self._training_step(batch)

        if self.device == "cuda":
            torch.cuda.synchronize()

    def _measure(self) -> Tuple[List[float], List[float]]:
        """
        Run measure_steps with CUDA-synchronized timing.

        Returns:
            (step_times_seconds, loss_values)
        """
        step_times: List[float] = []
        loss_values: List[float] = []

        for _ in range(self.config.measure_steps):
            batch = self._get_batch()

            # BEFORE: synchronize to flush any pending GPU work
            if self.device == "cuda":
                torch.cuda.synchronize()

            t0 = time.perf_counter()

            loss_val = self._training_step(batch)

            # AFTER: synchronize to wait for step kernels to complete
            if self.device == "cuda":
                torch.cuda.synchronize()

            step_time = time.perf_counter() - t0
            step_times.append(step_time)

            if loss_val is not None:
                loss_values.append(float(loss_val))

        return step_times, loss_values

    def _training_step(self, batch: Dict[str, Any]) -> Optional[float]:
        """
        Execute one full training step (includes gradient accumulation).

        Returns:
            Loss value as float, or None if no loss computed.
        """
        accumulated_loss: Optional["torch.Tensor"] = None

        for accum_idx in range(self.config.grad_accum_steps):
            micro_batch = self._get_micro_batch(batch, accum_idx)

            # Determine if this is the last accumulation step
            is_last = (accum_idx == self.config.grad_accum_steps - 1)

            with torch.autocast(
                device_type=self.device if self.device != "cpu" else "cpu",
                dtype=self._get_torch_dtype(),
                enabled=(self.device == "cuda"),
            ):
                outputs = self.model(
                    input_ids=micro_batch["input_ids"],
                    labels=micro_batch["labels"],
                )

            if "loss" in outputs and outputs["loss"] is not None:
                loss = outputs["loss"] / self.config.grad_accum_steps
                loss.backward()
                if accumulated_loss is None:
                    accumulated_loss = loss.detach()
                else:
                    accumulated_loss = accumulated_loss + loss.detach()

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        return float(accumulated_loss.cpu()) if accumulated_loss is not None else None

    # -----------------------------------------------------------------------
    # Internal: batch generation
    # -----------------------------------------------------------------------

    def _get_batch(self) -> Dict[str, Any]:
        """Get a batch according to config.mode."""
        if self.config.mode == "synthetic":
            return self._create_synthetic_batch()
        else:  # e2e
            if self.data_iter is None:
                raise RuntimeError(
                    "data_iter is required for mode='e2e'. "
                    "Pass a data iterator to BenchTrain.__init__()."
                )
            batch = next(self.data_iter)
            return {
                k: v.to(self.device, non_blocking=True)
                for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }

    def _create_synthetic_batch(self) -> Dict[str, "torch.Tensor"]:
        """
        Generate a random batch of shape (batch_size, seq_len) directly on device.
        Bypasses dataloader I/O entirely for pure-compute measurement.
        """
        bs = self.config.per_device_batch_size
        seq = self.config.seq_len
        vocab = self.config.vocab_size

        input_ids = torch.randint(
            0, vocab, (bs, seq),
            device=self.device,
            dtype=torch.long,
        )
        labels = torch.randint(
            0, vocab, (bs, seq),
            device=self.device,
            dtype=torch.long,
        )
        return {"input_ids": input_ids, "labels": labels}

    def _get_micro_batch(self, batch: Dict[str, Any], accum_idx: int) -> Dict[str, Any]:
        """
        For gradient accumulation: slice batch into micro-batches.
        If already per-device sized, return as-is.
        """
        # For simplicity in synthetic mode, return the whole batch each time.
        # In production, you'd slice: batch[accum_idx * micro_bs:(accum_idx+1) * micro_bs]
        return batch

    # -----------------------------------------------------------------------
    # Internal: MFU computation
    # -----------------------------------------------------------------------

    def _compute_mfu(
        self,
        step_time_s: float,
        tokens_per_step: int,
        n_non_embed_params: int,
    ) -> Optional[float]:
        """
        Compute Model FLOPs Utilization for a given step time.

        Returns:
            MFU in [0, 1] or None if peak_tflops not available.
        """
        if self._peak_tflops is None:
            return None

        if self.config.mfu_estimator == "6ND":
            flops = 6.0 * n_non_embed_params * tokens_per_step
        else:  # transformer_aware
            base_flops = 6.0 * n_non_embed_params * tokens_per_step
            # Attention FLOPs: 12 * seq^2 * hidden * num_layers * batch_tokens/seq
            batch_seqs = tokens_per_step // max(1, self.config.seq_len)
            attn_flops = (
                12.0
                * (self.config.seq_len ** 2)
                * self.config.hidden_dim
                * self.config.num_layers
                * batch_seqs
            )
            flops = base_flops + attn_flops

        achieved_tflops = flops / step_time_s / 1e12
        total_peak_tflops = self._effective_world_size * self._peak_tflops
        if total_peak_tflops <= 0:
            return None
        mfu = achieved_tflops / total_peak_tflops
        return max(0.0, mfu)  # Clamp to non-negative (noise can produce tiny negatives)

    # -----------------------------------------------------------------------
    # Internal: aggregation and result building
    # -----------------------------------------------------------------------

    def _aggregate(
        self,
        step_times: List[float],
        tokens_per_step: int,
    ) -> Dict[str, float]:
        """Compute mean/p50/p90/p10 statistics from step times."""
        if _NUMPY_AVAILABLE:
            import numpy as np
            arr = np.array(step_times, dtype=np.float64)
            tps_arr = np.array([tokens_per_step / t for t in step_times], dtype=np.float64)
            return {
                "step_time_mean": float(np.mean(arr)),
                "step_time_p50": float(np.percentile(arr, 50)),
                "step_time_p90": float(np.percentile(arr, 90)),
                "step_time_min": float(np.min(arr)),
                "step_time_max": float(np.max(arr)),
                "tps_mean": float(np.mean(tps_arr)),
                "tps_p50": float(np.percentile(tps_arr, 50)),
                "tps_p10": float(np.percentile(tps_arr, 10)),
            }
        else:
            # Pure Python fallback
            sorted_times = sorted(step_times)
            n = len(sorted_times)

            def pct(lst, p):
                idx = int(math.ceil(p / 100.0 * n)) - 1
                return lst[max(0, min(idx, n - 1))]

            tps_list = [tokens_per_step / t for t in step_times]
            sorted_tps = sorted(tps_list)
            return {
                "step_time_mean": sum(step_times) / n,
                "step_time_p50": pct(sorted_times, 50),
                "step_time_p90": pct(sorted_times, 90),
                "step_time_min": sorted_times[0],
                "step_time_max": sorted_times[-1],
                "tps_mean": sum(tps_list) / n,
                "tps_p50": pct(sorted_tps, 50),
                "tps_p10": pct(sorted_tps, 10),
            }

    def _compute_loss_slope(self, loss_values: List[float]) -> Optional[float]:
        """Estimate slope of loss curve via linear regression."""
        if len(loss_values) < 2:
            return None
        n = len(loss_values)
        x_mean = (n - 1) / 2.0
        y_mean = sum(loss_values) / n
        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(loss_values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def _read_memory(self) -> Tuple[int, int]:
        """Return (peak_allocated_bytes, peak_reserved_bytes)."""
        if self.device == "cuda" and _TORCH_AVAILABLE:
            return (
                torch.cuda.max_memory_allocated(),
                torch.cuda.max_memory_reserved(),
            )
        return 0, 0

    def _build_result(
        self,
        step_times: List[float],
        loss_values: List[float],
        peak_alloc: int,
        peak_reserved: int,
    ) -> BenchResult:
        """Assemble BenchResult from raw measurements."""
        tokens_per_step = self.config.tokens_per_step
        stats = self._aggregate(step_times, tokens_per_step)

        n_non_embed = _count_non_embedding_params(self.model) if _TORCH_AVAILABLE else 0
        mfu_per_step = [
            self._compute_mfu(t, tokens_per_step, n_non_embed)
            for t in step_times
        ]
        mfu_values = [m for m in mfu_per_step if m is not None]

        if mfu_values and _NUMPY_AVAILABLE:
            import numpy as np
            mfu_p50 = float(np.percentile(mfu_values, 50))
            mfu_mean = float(np.mean(mfu_values))
        elif mfu_values:
            mfu_p50 = sorted(mfu_values)[len(mfu_values) // 2]
            mfu_mean = sum(mfu_values) / len(mfu_values)
        else:
            mfu_p50 = None
            mfu_mean = None

        return BenchResult(
            step_times=step_times,
            step_time_mean=stats["step_time_mean"],
            step_time_p50=stats["step_time_p50"],
            step_time_p90=stats["step_time_p90"],
            step_time_min=stats["step_time_min"],
            step_time_max=stats["step_time_max"],
            tokens_per_sec_mean=stats["tps_mean"],
            tokens_per_sec_p50=stats["tps_p50"],
            tokens_per_sec_p10=stats["tps_p10"],
            tokens_per_step=tokens_per_step,
            peak_allocated_bytes=peak_alloc,
            peak_allocated_gb=round(peak_alloc / (1024 ** 3), 3),
            peak_reserved_bytes=peak_reserved,
            peak_reserved_gb=round(peak_reserved / (1024 ** 3), 3),
            mfu_p50=mfu_p50,
            mfu_mean=mfu_mean,
            n_non_embedding_params=n_non_embed,
            peak_tflops_per_gpu=self._peak_tflops,
            mfu_estimator=self.config.mfu_estimator,
            loss_values=loss_values,
            final_loss=loss_values[-1] if loss_values else None,
            loss_slope=self._compute_loss_slope(loss_values),
            machine_profile=self.machine_profile,
            run_config=self._build_run_config(),
        )

    # -----------------------------------------------------------------------
    # Internal: utilities
    # -----------------------------------------------------------------------

    def _resolve_world_size(self) -> int:
        """Resolve world_size=-1 to actual GPU count."""
        if self.config.world_size == -1:
            if _TORCH_AVAILABLE and torch.cuda.is_available():
                return torch.cuda.device_count()
            return 1
        return max(1, self.config.world_size)

    def _resolve_peak_tflops(self) -> Optional[float]:
        """Determine peak TFLOPS from config override or registry."""
        if self.config.peak_tflops is not None:
            return self.config.peak_tflops
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            dtype_key = "bf16" if self.config.dtype == "bfloat16" else "fp16"
            return get_peak_tflops(device_name, dtype_key)
        return None

    def _get_torch_dtype(self) -> "torch.dtype":
        """Convert config dtype string to torch dtype."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.config.dtype, torch.bfloat16)

    def _build_run_config(self) -> dict:
        """Build run config dict for metrics.json."""
        return {
            "mode": self.config.mode,
            "world_size": self._effective_world_size,
            "per_device_batch_size": self.config.per_device_batch_size,
            "seq_len": self.config.seq_len,
            "global_batch_size": self.config.tokens_per_step // self.config.seq_len,
            "grad_accum_steps": self.config.grad_accum_steps,
            "warmup_steps": self.config.warmup_steps,
            "measure_steps": self.config.measure_steps,
            "repeat": self.config.repeat,
            "profile": self.config.profile,
            "mfu_estimator": self.config.mfu_estimator,
            "dtype": self.config.dtype,
        }

    def _collect_env_summary(self) -> dict:
        """Lightweight env snapshot for metrics.json."""
        torch_ver = torch.__version__ if _TORCH_AVAILABLE else "N/A"
        cuda_ver = getattr(torch.version, "cuda", "N/A") if _TORCH_AVAILABLE else "N/A"
        return {
            "torch_version": torch_ver,
            "cuda_version": cuda_ver,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        }


# ===========================================================================
# Self-Tests
# ===========================================================================

def _run_self_tests() -> None:
    """Run all self-tests."""
    print("Running bench_train_template self-tests...")
    failures: List[str] = []

    def check(name: str, condition: bool, msg: str = "") -> None:
        if not condition:
            failures.append(f"FAIL [{name}]: {msg}")
        else:
            print(f"  PASS  {name}")

    if not _TORCH_AVAILABLE:
        print("  WARN  PyTorch not available; skipping GPU tests")
    else:
        # Use very small config for fast tests
        config = BenchConfig(
            warmup_steps=2,
            measure_steps=5,
            mode="synthetic",
            per_device_batch_size=2,
            seq_len=32,
            grad_accum_steps=1,
            vocab_size=100,
            hidden_dim=64,
            num_layers=2,
            repeat=1,
            peak_tflops=989.0,  # Fixed for testing
        )

        bench = BenchTrain(config, machine_profile="test_H100x1_sm90")
        result = bench.run()

        # --- step_times has exactly measure_steps entries ---
        check(
            "bench.step_times_count",
            len(result.step_times) == config.measure_steps,
            f"Expected {config.measure_steps}, got {len(result.step_times)}",
        )

        # --- All step times are positive ---
        check(
            "bench.step_times_positive",
            all(t > 0 for t in result.step_times),
            f"Non-positive step times: {[t for t in result.step_times if t <= 0]}",
        )

        # --- tokens/sec > 0 ---
        check(
            "bench.tokens_per_sec_positive",
            result.tokens_per_sec_p50 > 0,
            f"tokens_per_sec_p50 = {result.tokens_per_sec_p50}",
        )

        # --- tokens_per_step computation ---
        expected_tokens = config.per_device_batch_size * config.seq_len * max(1, config.world_size)
        check(
            "bench.tokens_per_step",
            result.tokens_per_step == expected_tokens,
            f"Expected {expected_tokens}, got {result.tokens_per_step}",
        )

        # --- MFU in valid range if available ---
        if result.mfu_p50 is not None:
            check(
                "bench.mfu_in_range",
                0 < result.mfu_p50 <= 1.0,
                f"MFU = {result.mfu_p50} out of (0, 1]",
            )
        else:
            print("  INFO  MFU is None (GPU not in registry or CPU run)")

        # --- p50 <= p90 for step times ---
        check(
            "bench.p50_le_p90",
            result.step_time_p50 <= result.step_time_p90,
            f"p50={result.step_time_p50} > p90={result.step_time_p90}",
        )

        # --- min <= p50 <= max ---
        check(
            "bench.min_le_p50_le_max",
            result.step_time_min <= result.step_time_p50 <= result.step_time_max,
            f"min={result.step_time_min}, p50={result.step_time_p50}, max={result.step_time_max}",
        )

        # --- Synthetic batch shape ---
        batch = bench._create_synthetic_batch()
        check(
            "bench.synthetic_batch_input_shape",
            batch["input_ids"].shape == (config.per_device_batch_size, config.seq_len),
            f"Expected ({config.per_device_batch_size}, {config.seq_len}), got {batch['input_ids'].shape}",
        )
        check(
            "bench.synthetic_batch_labels_shape",
            batch["labels"].shape == (config.per_device_batch_size, config.seq_len),
            f"Shape mismatch for labels",
        )

        # --- metrics.json schema ---
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "metrics.json")
            bench.save(save_path)
            check("bench.save_creates_file", os.path.isfile(save_path))

            with open(save_path) as f:
                loaded = json.load(f)

            required_sections = [
                "schema_version", "machine_profile", "run_config",
                "throughput", "timing", "memory", "mfu", "loss",
            ]
            for section in required_sections:
                check(
                    f"bench.metrics_json.has_{section}",
                    section in loaded,
                    f"Missing section '{section}'",
                )

            # Throughput fields
            for field_name in ["tokens_per_sec_mean", "tokens_per_sec_p50", "tokens_per_sec_p10"]:
                check(
                    f"bench.throughput.has_{field_name}",
                    field_name in loaded["throughput"],
                    f"Missing {field_name}",
                )

        # --- 6ND MFU formula unit test ---
        n_params = 7_000_000_000
        tokens = 131072
        step_time_s = 1.0
        world = 8
        peak_tflops = 989.0
        flops = 6 * n_params * tokens
        achieved = flops / step_time_s / 1e12
        expected_mfu = achieved / (world * peak_tflops)
        # Verify formula using a mock config
        cfg_mfu = BenchConfig(
            warmup_steps=0,
            measure_steps=1,
            world_size=world,
            peak_tflops=peak_tflops,
            seq_len=512,
            per_device_batch_size=4,  # 4 * 8 * 512 = 16384 tokens
        )
        bench2 = BenchTrain(cfg_mfu, machine_profile="test")
        bench2._peak_tflops = peak_tflops
        bench2._effective_world_size = world
        calc_mfu = bench2._compute_mfu(step_time_s, tokens, n_params)
        check(
            "bench.mfu_6nd_formula",
            calc_mfu is not None and abs(calc_mfu - expected_mfu) < 1e-6,
            f"Expected {expected_mfu:.6f}, got {calc_mfu}",
        )

        # --- loss slope computation ---
        slope = bench._compute_loss_slope([2.5, 2.4, 2.3, 2.2, 2.1])
        check(
            "bench.loss_slope_negative",
            slope is not None and slope < 0,
            f"Expected negative slope, got {slope}",
        )
        flat_slope = bench._compute_loss_slope([2.0, 2.0, 2.0])
        check(
            "bench.loss_slope_flat",
            flat_slope is not None and abs(flat_slope) < 1e-9,
            f"Expected ~0 slope, got {flat_slope}",
        )

        # --- repeat=2 takes best ---
        config_repeat = BenchConfig(
            warmup_steps=1,
            measure_steps=3,
            mode="synthetic",
            per_device_batch_size=2,
            seq_len=32,
            vocab_size=100,
            hidden_dim=64,
            num_layers=2,
            repeat=2,
            peak_tflops=989.0,
        )
        bench_rep = BenchTrain(config_repeat, machine_profile="test")
        result_rep = bench_rep.run()
        check("bench.repeat_produces_result", result_rep is not None)

    # --- Aggregation correctness (pure Python) ---
    from perf_gate_config_template import BenchConfig as BC
    fake_bench = object.__new__(BenchTrain)
    fake_bench.config = BenchConfig(
        warmup_steps=0, measure_steps=5, seq_len=100, per_device_batch_size=1,
        vocab_size=10, hidden_dim=8, num_layers=1
    )
    fake_bench._peak_tflops = None
    fake_bench._effective_world_size = 1

    times = [1.0, 1.1, 1.2, 1.3, 2.0]
    stats = fake_bench._aggregate(times, tokens_per_step=1000)
    check(
        "bench.aggregate_p50_correct",
        abs(stats["step_time_p50"] - 1.2) < 0.01,
        f"Expected ~1.2, got {stats['step_time_p50']}",
    )
    check(
        "bench.aggregate_min_correct",
        abs(stats["step_time_min"] - 1.0) < 1e-9,
        f"Expected 1.0, got {stats['step_time_min']}",
    )
    check(
        "bench.aggregate_tps_positive",
        stats["tps_p50"] > 0,
        f"tps_p50 = {stats['tps_p50']}",
    )

    # Summary
    print()
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    else:
        print("All bench_train_template self-tests passed.")


if __name__ == "__main__":
    _run_self_tests()
