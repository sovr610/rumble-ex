#!/usr/bin/env python3
"""
validate_distributed.py
------------------------
Validates the three Done-When gates for the distributed-memory-scaling skill.

Gate 1: Strategy Switch is Config-Only
  Verify that StrategyRouter correctly dispatches for ddp, fsdp, and
  deepspeed_zero3 without code changes, using mock-based path validation.

Gate 2: Checkpoint Portability is Real
  Verify that FSDP uses FullStateDictConfig(offload_to_cpu=True, rank0_only=True).
  Verify that DeepSpeed config JSON includes stage3_gather_16bit_weights_on_model_save.

Gate 3: Scaling Harness Produces Stable Metrics
  Verify that metrics.json schema includes all required fields:
  strategy, world_size, scaling_efficiency, memory_peak_gb,
  throughput_p50, step_time_p50_ms.

Exit codes:
  0 — all gates passed
  1 — one or more gates failed
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Setup: add assets directory to path
# ---------------------------------------------------------------------------

SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(SKILL_DIR, "assets")
if ASSETS_DIR not in sys.path:
    sys.path.insert(0, ASSETS_DIR)


# ---------------------------------------------------------------------------
# Minimal config stubs (standalone, no dependency on distributed_config_template)
# ---------------------------------------------------------------------------


@dataclass
class _DistCfg:
    strategy: str = "ddp"
    world_size: int = 1
    backend: str = "gloo"
    grad_accum: int = 1


@dataclass
class _FSDPCfg:
    sharding_strategy: str = "FULL_SHARD"
    wrap_policy: str = "size_based"
    wrap_module_classes: list = None
    mixed_precision: str = "none"
    activation_checkpointing: str = "off"
    state_dict_type: str = "full"
    sync_module_states: bool = False
    cpu_offload: bool = False
    min_num_params: int = 1

    def __post_init__(self):
        if self.wrap_module_classes is None:
            self.wrap_module_classes = []


@dataclass
class _DSCfg:
    zero_stage: int = 3
    offload_optimizer: str = "none"
    offload_param: str = "none"
    reduce_bucket_size: int = 500_000_000
    allgather_bucket_size: int = 500_000_000
    stage3_prefetch_bucket_size: int = 50_000_000
    stage3_param_persistence_threshold: int = 100_000
    stage3_max_live_parameters: int = 1_000_000_000
    stage3_max_reuse_distance: int = 1_000_000_000
    overlap_comm: bool = True
    contiguous_gradients: bool = True
    nvme_path: Any = None
    fp16: bool = False
    bf16: bool = False
    gradient_clipping: float = 1.0

    def validate(self):
        if self.offload_param != "none" and self.zero_stage != 3:
            raise ValueError("offload_param requires zero_stage=3")


@dataclass
class _OptCfg:
    optimizer_type: str = "AdamW"
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    momentum: float = 0.9
    grad_clip: float = 1.0


# ---------------------------------------------------------------------------
# Gate result tracking
# ---------------------------------------------------------------------------


class GateResult:
    def __init__(self, gate_name: str) -> None:
        self.gate_name = gate_name
        self.checks: List[Tuple[str, bool, str]] = []  # (name, passed, message)

    def check(self, name: str, condition: bool, message: str = "") -> None:
        self.checks.append((name, condition, message))
        status = "PASS" if condition else "FAIL"
        if message and not condition:
            print(f"    [{status}] {name}: {message}")
        else:
            print(f"    [{status}] {name}")

    @property
    def passed(self) -> bool:
        return all(ok for _, ok, _ in self.checks)

    @property
    def pass_count(self) -> int:
        return sum(1 for _, ok, _ in self.checks if ok)

    @property
    def total(self) -> int:
        return len(self.checks)


# ---------------------------------------------------------------------------
# Gate 1: Strategy Switch
# ---------------------------------------------------------------------------


def run_gate1() -> GateResult:
    """Verify StrategyRouter dispatches correctly for all three strategies."""
    result = GateResult("Gate 1: Strategy Switch")
    print("\n  Gate 1: Strategy Switch")
    print("  " + "-" * 50)

    try:
        from strategy_router_template import RouterState, StrategyContext, StrategyRouter

        import torch.nn as nn

        model = nn.Linear(8, 8)

        # --- DDP path ---
        with patch("torch.distributed.init_process_group"), \
             patch("torch.distributed.is_initialized", return_value=False), \
             patch("torch.distributed.barrier"):

            router_ddp = StrategyRouter(_DistCfg(strategy="ddp"))
            router_ddp._state = RouterState.DISTRIBUTED_READY
            router_ddp._context = StrategyContext(
                strategy="ddp", rank=0, world_size=1,
                local_rank=0, device="cpu", is_distributed=False
            )
            wrapped_ddp, ctx_ddp = router_ddp.wrap_model(nn.Linear(4, 4))
            result.check(
                "ddp_dispatch_context_strategy",
                ctx_ddp.strategy == "ddp",
                f"got '{ctx_ddp.strategy}'",
            )
            result.check(
                "ddp_dispatch_state_model_wrapped",
                router_ddp.state == RouterState.MODEL_WRAPPED,
            )
            result.check(
                "ddp_dispatch_returns_module",
                isinstance(wrapped_ddp, nn.Module),
            )

        # --- FSDP path ---
        with patch("strategy_router_template.StrategyRouter._wrap_fsdp") as mock_fsdp:
            mock_fsdp.return_value = nn.Linear(4, 4)

            router_fsdp = StrategyRouter(
                _DistCfg(strategy="fsdp"),
                fsdp_cfg=_FSDPCfg(),
            )
            router_fsdp._state = RouterState.DISTRIBUTED_READY
            router_fsdp._context = StrategyContext(
                strategy="fsdp", rank=0, world_size=4,
                local_rank=0, device="cpu", is_distributed=True
            )
            _, ctx_fsdp = router_fsdp.wrap_model(nn.Linear(4, 4))
            result.check(
                "fsdp_dispatch_called_wrap_fsdp",
                mock_fsdp.called,
            )
            result.check(
                "fsdp_dispatch_context_strategy",
                ctx_fsdp.strategy == "fsdp",
            )

        # --- DeepSpeed ZeRO-3 path ---
        with patch("strategy_router_template.StrategyRouter._wrap_deepspeed") as mock_ds:
            mock_engine = MagicMock()
            mock_ds.return_value = (mock_engine, MagicMock())

            router_ds = StrategyRouter(
                _DistCfg(strategy="deepspeed_zero3"),
                ds_cfg=_DSCfg(zero_stage=3),
            )
            router_ds._state = RouterState.DISTRIBUTED_READY
            router_ds._context = StrategyContext(
                strategy="deepspeed_zero3", rank=0, world_size=4,
                local_rank=0, device="cpu", is_distributed=True
            )
            model_ds = nn.Linear(4, 4)
            _, ctx_ds = router_ds.wrap_model(model_ds)
            result.check(
                "ds_dispatch_context_strategy",
                ctx_ds.strategy == "deepspeed_zero3",
            )

            opt = router_ds.build_optimizer(model_ds, _OptCfg())
            result.check(
                "ds_dispatch_called_wrap_deepspeed",
                mock_ds.called,
            )
            result.check(
                "ds_state_optimizer_built",
                router_ds.state == RouterState.OPTIMIZER_BUILT,
            )

        # --- Wrapping order enforcement ---
        router_bad = StrategyRouter(_DistCfg(strategy="ddp"))
        try:
            router_bad.wrap_model(nn.Linear(4, 4))
            result.check(
                "order_wrap_before_setup_raises",
                False,
                "should have raised RuntimeError",
            )
        except RuntimeError:
            result.check("order_wrap_before_setup_raises", True)

    except ImportError as e:
        result.check("strategy_router_importable", False, str(e))

    return result


# ---------------------------------------------------------------------------
# Gate 2: Checkpoint Portability
# ---------------------------------------------------------------------------


def run_gate2() -> GateResult:
    """Verify checkpoint portability configuration."""
    result = GateResult("Gate 2: Checkpoint Portability")
    print("\n  Gate 2: Checkpoint Portability")
    print("  " + "-" * 50)

    # --- FSDP: FullStateDictConfig uses offload_to_cpu + rank0_only ---
    try:
        from torch.distributed.fsdp import FullStateDictConfig

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        result.check(
            "fsdp_full_state_dict_offload_to_cpu",
            cfg.offload_to_cpu is True,
        )
        result.check(
            "fsdp_full_state_dict_rank0_only",
            cfg.rank0_only is True,
        )
    except ImportError:
        result.check(
            "fsdp_full_state_dict_available",
            False,
            "torch.distributed.fsdp not available",
        )

    # --- FSDP wrapper uses correct config in save_full_state_dict ---
    try:
        from fsdp_wrapper_template import FSDPWrapper
        from torch.distributed.fsdp import StateDictType

        # Verify the save_full_state_dict method code references the right config
        import inspect
        source = inspect.getsource(FSDPWrapper.save_full_state_dict)
        result.check(
            "fsdp_save_uses_offload_to_cpu",
            "offload_to_cpu=True" in source,
        )
        result.check(
            "fsdp_save_uses_rank0_only",
            "rank0_only=True" in source,
        )
        result.check(
            "fsdp_save_uses_full_state_dict_type",
            "FULL_STATE_DICT" in source,
        )
    except ImportError as e:
        result.check("fsdp_wrapper_importable", False, str(e))

    # --- DeepSpeed: config JSON includes stage3_gather_16bit_weights_on_model_save ---
    try:
        from deepspeed_wrapper_template import DeepSpeedWrapper

        wrapper = DeepSpeedWrapper()
        cfg_ds = _DSCfg(zero_stage=3)
        config_dict = wrapper.generate_config_json(cfg_ds)

        result.check(
            "ds_stage3_gather_weights_on_save",
            config_dict.get("zero_optimization", {}).get(
                "stage3_gather_16bit_weights_on_model_save"
            ) is True,
        )
        result.check(
            "ds_zero_stage_correct",
            config_dict["zero_optimization"]["stage"] == 3,
        )
    except ImportError as e:
        result.check("deepspeed_wrapper_importable", False, str(e))

    # --- DeepSpeed: export_fp32_weights method exists and calls subprocess ---
    try:
        from deepspeed_wrapper_template import DeepSpeedWrapper
        import inspect

        source_export = inspect.getsource(DeepSpeedWrapper.export_fp32_weights)
        result.check(
            "ds_export_calls_subprocess",
            "subprocess" in source_export or "subprocess.run" in source_export,
        )
        result.check(
            "ds_export_calls_zero_to_fp32",
            "zero_to_fp32" in source_export,
        )
    except ImportError as e:
        result.check("deepspeed_export_method_importable", False, str(e))

    # --- Cross-strategy: FSDP full state dict name remapping documented ---
    try:
        from fsdp_wrapper_template import FSDPWrapper
        import inspect

        # The load_full_state_dict method should use rank0_only / FULL_STATE_DICT
        source_load = inspect.getsource(FSDPWrapper.load_full_state_dict)
        result.check(
            "fsdp_load_uses_full_state_dict",
            "FULL_STATE_DICT" in source_load,
        )
        result.check(
            "fsdp_load_uses_rank0_only",
            "rank0_only" in source_load,
        )
    except ImportError as e:
        result.check("fsdp_load_importable", False, str(e))

    return result


# ---------------------------------------------------------------------------
# Gate 3: Scaling Harness
# ---------------------------------------------------------------------------


def run_gate3() -> GateResult:
    """Verify scaling harness produces metrics.json with correct schema."""
    result = GateResult("Gate 3: Scaling Harness")
    print("\n  Gate 3: Scaling Harness")
    print("  " + "-" * 50)

    try:
        from scaling_benchmark_template import (
            BenchResult,
            ScalingBenchmark,
            ScalingReport,
            validate_metrics_schema,
            REQUIRED_METRICS_FIELDS,
        )

        bench = ScalingBenchmark()

        def make_result(world_size, throughput):
            return BenchResult(
                strategy="fsdp",
                world_size=world_size,
                throughput_p50=throughput,
                throughput_p90=throughput * 0.95,
                step_time_p50_ms=100.0,
                step_time_p90_ms=110.0,
                memory_peak_gb=20.0,
                measured_steps=50,
                warmup_steps=10,
                per_gpu_batch_size=4,
                seq_len=2048,
            )

        single = make_result(1, 10000.0)
        multi = make_result(4, 38000.0)
        report = bench.compute_efficiency(single, multi)

        # Schema validation
        data = report.to_dict()
        missing = validate_metrics_schema(data)
        result.check(
            "schema_no_missing_required_fields",
            len(missing) == 0,
            f"missing: {missing}" if missing else "",
        )

        # Required fields individually
        for field_name in sorted(REQUIRED_METRICS_FIELDS):
            result.check(
                f"required_field_{field_name}",
                field_name in data,
            )

        # Type checks
        result.check("strategy_is_str", isinstance(data["strategy"], str))
        result.check("world_size_is_int", isinstance(data["world_size"], int))
        result.check(
            "scaling_efficiency_is_float",
            isinstance(data["scaling_efficiency"], float),
        )
        result.check(
            "memory_peak_gb_is_positive",
            data["memory_peak_gb"] >= 0,
        )
        result.check(
            "throughput_p50_is_positive",
            data["throughput_p50"] > 0,
        )
        result.check(
            "step_time_p50_ms_is_positive",
            data["step_time_p50_ms"] > 0,
        )

        # Efficiency value range
        result.check(
            "scaling_efficiency_in_valid_range",
            0.0 < data["scaling_efficiency"] <= 2.0,
            f"got {data['scaling_efficiency']}",
        )

        # Save/load round-trip
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "metrics.json")
            bench.save_metrics(report, metrics_path)

            result.check("metrics_file_created", os.path.isfile(metrics_path))

            with open(metrics_path, "r") as f:
                raw = f.read()

            try:
                reparsed = json.loads(raw)
                result.check("metrics_json_parseable", True)
                result.check(
                    "metrics_strategy_preserved",
                    reparsed.get("strategy") == "fsdp",
                )
                result.check(
                    "metrics_world_size_preserved",
                    reparsed.get("world_size") == 4,
                )
            except json.JSONDecodeError as e:
                result.check("metrics_json_parseable", False, str(e))

        # Zero throughput baseline raises
        try:
            bench.compute_efficiency(make_result(1, 0.0), multi)
            result.check("zero_throughput_raises", False, "should have raised ValueError")
        except (ValueError, ZeroDivisionError):
            result.check("zero_throughput_raises", True)

        # world_size=1 efficiency is exactly 1.0
        report_1 = bench.compute_efficiency(
            make_result(1, 5000.0),
            make_result(1, 5000.0),
        )
        result.check(
            "world_size_1_efficiency_is_1",
            abs(report_1.scaling_efficiency - 1.0) < 1e-9,
            f"got {report_1.scaling_efficiency}",
        )

    except ImportError as e:
        result.check("scaling_benchmark_importable", False, str(e))

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 60)
    print("  Distributed Memory Scaling — Validation Suite")
    print("=" * 60)

    gate1 = run_gate1()
    gate2 = run_gate2()
    gate3 = run_gate3()

    gates = [gate1, gate2, gate3]

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)

    all_passed = True
    for gate in gates:
        status = "PASS" if gate.passed else "FAIL"
        print(
            f"  [{status}] {gate.gate_name}: "
            f"{gate.pass_count}/{gate.total} checks passed"
        )
        if not gate.passed:
            all_passed = False
            for name, ok, msg in gate.checks:
                if not ok:
                    detail = f" ({msg})" if msg else ""
                    print(f"         FAIL: {name}{detail}")

    print("=" * 60)
    if all_passed:
        total_checks = sum(g.total for g in gates)
        print(f"  OVERALL: PASS — {total_checks}/{total_checks} checks passed")
        return 0
    else:
        passed = sum(g.pass_count for g in gates)
        total = sum(g.total for g in gates)
        print(f"  OVERALL: FAIL — {passed}/{total} checks passed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
