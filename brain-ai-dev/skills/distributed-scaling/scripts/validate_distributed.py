"""
validate_distributed.py — Validates the three done-when gates for the
distributed-scaling skill using simulated multi-process testing.

Done-When Gates:
    1. DDP Training — DDPWrapper produces correct aggregated metrics in
       single-process mode; all_reduce_metrics returns passthrough values.
    2. FSDP Sharding — FSDPWrapper single-process setup/checkpoint
       round-trips correctly; sharding policy presets are consistent.
    3. Rank-Aware Loading — Each simulated rank processes unique samples;
       set_epoch() changes shuffle order; no duplicate samples across world.

All tests run in single-process on CPU (no GPU required).
Uses gloo backend where process groups are needed.
"""

from __future__ import annotations

import copy
import logging
import math
import os
import sys
import tempfile
import time
import traceback
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

# Add parent assets directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "assets")
sys.path.insert(0, ASSETS_DIR)

# Import templates
from ddp_wrapper_template import DDPWrapper, ProcessGroupManager, get_device_for_rank
from fsdp_wrapper_template import FSDPWrapper, ShardingPolicy, create_fsdp_wrapper
from gradient_accumulator_template import (
    GradientAccumulator,
    compute_scaled_lr,
    get_cosine_lr,
)
from distributed_launcher_template import (
    DistributedLauncher,
    LaunchConfig,
    EnvironmentDetector,
)
from rank_aware_loader_template import (
    RankAwareDataLoader,
    SamplerVerifier,
    WorkerSeedManager,
)
from distributed_config_template import (
    DistributedConfig,
    MemoryConfig,
    CommunicationConfig,
    CheckpointConfig,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

_DIST_SAMPLER_AVAILABLE = True
try:
    from torch.utils.data.distributed import DistributedSampler
except ImportError:
    _DIST_SAMPLER_AVAILABLE = False


# ===========================================================================
# Test infrastructure
# ===========================================================================

class ValidationRunner:
    """Runs validation tests and tracks results by gate."""

    def __init__(self):
        self.results: Dict[str, List[Dict[str, Any]]] = {
            "gate1_ddp": [],
            "gate2_fsdp": [],
            "gate3_rank_aware": [],
        }
        self._current_gate = "gate1_ddp"

    def set_gate(self, gate: str) -> None:
        self._current_gate = gate

    def run_test(self, name: str, fn: Callable) -> bool:
        try:
            fn()
            self.results[self._current_gate].append({
                "name": name, "passed": True, "error": None,
            })
            print(f"  PASS  {name}")
            return True
        except Exception as e:
            self.results[self._current_gate].append({
                "name": name, "passed": False, "error": str(e),
            })
            print(f"  FAIL  {name}: {e}")
            traceback.print_exc()
            return False

    def summary(self) -> Dict[str, Any]:
        summary = {}
        total_pass = 0
        total_fail = 0
        for gate, tests in self.results.items():
            n_pass = sum(1 for t in tests if t["passed"])
            n_fail = sum(1 for t in tests if not t["passed"])
            total_pass += n_pass
            total_fail += n_fail
            summary[gate] = {
                "passed": n_pass,
                "failed": n_fail,
                "total": len(tests),
                "gate_passed": n_fail == 0,
            }
        summary["total"] = {
            "passed": total_pass,
            "failed": total_fail,
            "total": total_pass + total_fail,
            "all_gates_passed": total_fail == 0,
        }
        return summary


def _make_model(in_dim=32, hidden=64, out_dim=10):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


def _make_dataset(n_samples=200, in_dim=32, n_classes=10):
    x = torch.randn(n_samples, in_dim)
    y = torch.randint(0, n_classes, (n_samples,))
    return TensorDataset(x, y)


# ===========================================================================
# GATE 1: DDP Training Validation
# ===========================================================================

def validate_gate1_ddp(runner: ValidationRunner) -> None:
    """Validate DDP done-when gate.

    DDPWrapper.setup() on single-GPU produces correct per-step loss;
    all_reduce_metrics() returns correct aggregated values.
    """
    runner.set_gate("gate1_ddp")

    # Test 1: DDPWrapper setup in single-process returns unwrapped model
    def test_ddp_setup_single():
        model = _make_model()
        wrapper = DDPWrapper(model, backend="gloo")
        returned = wrapper.setup(rank=0, world_size=1)
        assert returned is model, "Setup should return unwrapped model in single-process"
        wrapper.cleanup()

    runner.run_test("DDP: setup returns unwrapped model", test_ddp_setup_single)

    # Test 2: Forward + backward produces gradients
    def test_ddp_forward_backward():
        model = _make_model()
        wrapper = DDPWrapper(model, backend="gloo")
        wrapper.setup(rank=0, world_size=1)
        x = torch.randn(8, 32)
        out = wrapper.wrapped_model(x)
        assert out.shape == (8, 10)
        loss = out.sum()
        loss.backward()
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad, "No gradients after backward"
        wrapper.cleanup()

    runner.run_test("DDP: forward+backward produces gradients", test_ddp_forward_backward)

    # Test 3: all_reduce_metrics returns passthrough in single-process
    def test_ddp_all_reduce_passthrough():
        model = _make_model()
        wrapper = DDPWrapper(model, backend="gloo")
        wrapper.setup(rank=0, world_size=1)
        metrics = {"loss": 0.5, "accuracy": 0.95, "lr": 1e-3}
        reduced = wrapper.all_reduce_metrics(metrics)
        for k, v in metrics.items():
            assert abs(reduced[k] - v) < 1e-6, f"{k}: {reduced[k]} != {v}"
        wrapper.cleanup()

    runner.run_test("DDP: all_reduce_metrics passthrough", test_ddp_all_reduce_passthrough)

    # Test 4: Training step changes parameters
    def test_ddp_training_step():
        model = _make_model()
        wrapper = DDPWrapper(model, backend="gloo")
        wrapper.setup(rank=0, world_size=1)
        params_before = {n: p.clone() for n, p in model.named_parameters()}
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.randn(8, 32)
        out = wrapper.wrapped_model(x)
        loss = out.sum()
        loss.backward()
        wrapper.clip_gradients(max_norm=1.0)
        optimizer.step()
        changed = any(
            not torch.equal(p, params_before[n])
            for n, p in model.named_parameters()
        )
        assert changed, "Optimizer step did not change parameters"
        wrapper.cleanup()

    runner.run_test("DDP: training step changes parameters", test_ddp_training_step)

    # Test 5: Checkpoint save/load round-trip
    def test_ddp_checkpoint_roundtrip():
        model = _make_model()
        wrapper = DDPWrapper(model, backend="gloo")
        wrapper.setup(rank=0, world_size=1)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        wrapper.save_checkpoint(path, epoch=5, step=100)
        assert os.path.exists(path)
        # Modify model
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(999.0)
        # Load
        wrapper.load_checkpoint(path)
        first_param = next(model.parameters())
        assert not torch.all(first_param == 999.0), "Checkpoint did not restore"
        os.unlink(path)
        wrapper.cleanup()

    runner.run_test("DDP: checkpoint round-trip", test_ddp_checkpoint_roundtrip)

    # Test 6: GradientAccumulator equivalence
    def test_ddp_gradient_accumulation():
        K, M = 4, 8
        torch.manual_seed(42)
        # Use a model WITHOUT BatchNorm -- BatchNorm computes different
        # statistics for different batch sizes, breaking the equivalence.
        model_a = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 4))
        model_b = copy.deepcopy(model_a)
        data = torch.randn(K * M, 32)
        target = torch.randn(K * M, 4)
        # Single large batch
        out_a = model_a(data)
        loss_a = nn.functional.mse_loss(out_a, target)
        loss_a.backward()
        # Accumulated
        acc = GradientAccumulator(accumulation_steps=K)
        for k in range(K):
            s, e = k * M, (k + 1) * M
            out_b = model_b(data[s:e])
            loss_b = nn.functional.mse_loss(out_b, target[s:e])
            acc.backward_only(loss_b)
        for (na, pa), (nb, pb) in zip(
            model_a.named_parameters(), model_b.named_parameters()
        ):
            torch.testing.assert_close(pa.grad, pb.grad, atol=1e-5, rtol=1e-5)

    runner.run_test("DDP: gradient accumulation equivalence", test_ddp_gradient_accumulation)

    # Test 7: ProcessGroupManager defaults
    def test_pgm_defaults():
        pgm = ProcessGroupManager()
        assert pgm.rank == 0
        assert pgm.world_size == 1
        assert pgm.is_main is True
        assert pgm.initialized is False

    runner.run_test("DDP: ProcessGroupManager defaults", test_pgm_defaults)

    # Test 8: no_sync context manager in single process
    def test_ddp_no_sync():
        model = _make_model()
        wrapper = DDPWrapper(model, backend="gloo")
        wrapper.setup(rank=0, world_size=1)
        with wrapper.no_sync():
            x = torch.randn(4, 32)
            out = model(x)
            out.sum().backward()
        wrapper.cleanup()

    runner.run_test("DDP: no_sync context single-process", test_ddp_no_sync)


# ===========================================================================
# GATE 2: FSDP Sharding Validation
# ===========================================================================

def validate_gate2_fsdp(runner: ValidationRunner) -> None:
    """Validate FSDP done-when gate.

    FSDPWrapper single-process setup; distributed checkpoint round-trips;
    sharding policy presets are consistent.
    """
    runner.set_gate("gate2_fsdp")

    # Test 1: FSDP setup returns unwrapped model in single-process
    def test_fsdp_setup_single():
        model = _make_model()
        wrapper = FSDPWrapper(model, backend="gloo")
        returned = wrapper.setup(rank=0, world_size=1)
        assert returned is model
        wrapper.cleanup()

    runner.run_test("FSDP: setup returns unwrapped model", test_fsdp_setup_single)

    # Test 2: Forward/backward in single-process
    def test_fsdp_forward_backward():
        model = _make_model()
        wrapper = FSDPWrapper(model, backend="gloo")
        wrapper.setup(rank=0, world_size=1)
        x = torch.randn(4, 32)
        out = wrapper.wrapped_model(x)
        assert out.shape == (4, 10)
        loss = out.sum()
        loss.backward()
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad
        wrapper.cleanup()

    runner.run_test("FSDP: forward+backward", test_fsdp_forward_backward)

    # Test 3: Checkpoint round-trip
    def test_fsdp_checkpoint_roundtrip():
        model = _make_model()
        wrapper = FSDPWrapper(model, backend="gloo")
        wrapper.setup(rank=0, world_size=1)
        with tempfile.TemporaryDirectory() as tmp_dir:
            wrapper.save_distributed_checkpoint(tmp_dir)
            assert os.path.exists(os.path.join(tmp_dir, "model.pt"))
            # Modify
            with torch.no_grad():
                for p in model.parameters():
                    p.fill_(999.0)
            # Load
            wrapper.load_distributed_checkpoint(tmp_dir)
            first_param = next(model.parameters())
            assert not torch.all(first_param == 999.0)
        wrapper.cleanup()

    runner.run_test("FSDP: checkpoint round-trip", test_fsdp_checkpoint_roundtrip)

    # Test 4: Phase checkpoint creates both formats
    def test_fsdp_phase_checkpoint():
        model = _make_model()
        wrapper = FSDPWrapper(model, backend="gloo")
        wrapper.setup(rank=0, world_size=1)
        with tempfile.TemporaryDirectory() as tmp_dir:
            wrapper.save_phase_checkpoint(phase=1, base_dir=tmp_dir)
            assert os.path.isdir(os.path.join(tmp_dir, "phase1_sharded"))
            assert os.path.exists(os.path.join(tmp_dir, "phase1_full.pt"))
        wrapper.cleanup()

    runner.run_test("FSDP: phase checkpoint both formats", test_fsdp_phase_checkpoint)

    # Test 5: ShardingPolicy presets are consistent
    def test_fsdp_policy_presets():
        for name, factory in [
            ("1b", ShardingPolicy.for_1b),
            ("3b", ShardingPolicy.for_3b),
            ("7b", ShardingPolicy.for_7b),
            ("7b_constrained", ShardingPolicy.for_7b_constrained),
        ]:
            policy = factory()
            assert policy.strategy in ("full_shard", "shard_grad_op", "no_shard"), \
                f"{name}: invalid strategy {policy.strategy}"

    runner.run_test("FSDP: all policy presets valid", test_fsdp_policy_presets)

    # Test 6: create_fsdp_wrapper factory
    def test_fsdp_factory():
        model = _make_model()
        for scale in ("1b", "3b", "7b"):
            wrapper = create_fsdp_wrapper(model, scale=scale, backend="gloo")
            assert wrapper.policy is not None

    runner.run_test("FSDP: create_fsdp_wrapper factory", test_fsdp_factory)

    # Test 7: Parameter summary
    def test_fsdp_param_summary():
        model = _make_model()
        wrapper = FSDPWrapper(model, backend="gloo")
        wrapper.setup(rank=0, world_size=1)
        summary = wrapper.get_parameter_summary()
        assert summary["total_parameters"] > 0
        assert summary["trainable_parameters"] > 0
        wrapper.cleanup()

    runner.run_test("FSDP: parameter summary", test_fsdp_param_summary)

    # Test 8: DistributedConfig FSDP presets validate
    def test_fsdp_config_presets():
        for name, factory in [
            ("3b", DistributedConfig.for_3b),
            ("7b", DistributedConfig.for_7b),
            ("7b_constrained", DistributedConfig.for_7b_constrained),
        ]:
            cfg = factory()
            errors = cfg.validate()
            assert len(errors) == 0, f"{name}: {errors}"
            assert cfg.strategy == "fsdp"

    runner.run_test("FSDP: config presets validate", test_fsdp_config_presets)


# ===========================================================================
# GATE 3: Rank-Aware Loading Validation
# ===========================================================================

def validate_gate3_rank_aware(runner: ValidationRunner) -> None:
    """Validate rank-aware loading done-when gate.

    Each rank processes unique samples; set_epoch() changes shuffle order;
    no duplicate samples across the world.
    """
    runner.set_gate("gate3_rank_aware")

    # Test 1: No duplicate samples (drop_last=True)
    def test_no_duplicates_basic():
        for ws in [1, 2, 4, 8]:
            valid, dups = SamplerVerifier.verify_no_duplicates(
                1000, world_size=ws, drop_last=True,
            )
            assert valid, f"Duplicates at world_size={ws}: {dups}"

    runner.run_test("Rank: no duplicates (various world_sizes)", test_no_duplicates_basic)

    # Test 2: Full coverage
    def test_full_coverage():
        for ws in [1, 2, 4]:
            valid, missing = SamplerVerifier.verify_full_coverage(1000, world_size=ws)
            assert valid, f"Missing at world_size={ws}: {missing}"

    runner.run_test("Rank: full coverage", test_full_coverage)

    # Test 3: Balanced partition
    def test_balanced():
        for ds_size in [100, 101, 1000, 1023]:
            for ws in [1, 2, 3, 4, 7, 8]:
                valid, counts = SamplerVerifier.verify_balanced_partition(ds_size, ws)
                assert valid, f"Unbalanced: size={ds_size}, ws={ws}, counts={counts}"

    runner.run_test("Rank: balanced partition (many configs)", test_balanced)

    # Test 4: set_epoch changes order
    def test_epoch_changes():
        result = SamplerVerifier.verify_epoch_changes_order(
            dataset_size=200, world_size=2, rank=0,
        )
        assert result, "set_epoch did not change shuffle order"

    runner.run_test("Rank: set_epoch changes order", test_epoch_changes)

    # Test 5: RankAwareDataLoader single-process iteration
    def test_ral_iteration():
        ds = _make_dataset(100)
        ral = RankAwareDataLoader(ds, rank=0, world_size=1, batch_size=10, drop_last=True)
        loader = ral.get_loader()
        count = 0
        for batch in loader:
            x, y = batch
            assert x.shape[0] == 10
            count += 1
        assert count == 10

    runner.run_test("Rank: RankAwareDataLoader iteration", test_ral_iteration)

    # Test 6: Simulated multi-rank unique indices
    def test_multi_rank_unique():
        if not _DIST_SAMPLER_AVAILABLE:
            return
        ds = list(range(200))
        world_size = 4
        all_indices = []
        for rank in range(world_size):
            sampler = DistributedSampler(
                ds, num_replicas=world_size, rank=rank,
                shuffle=False, drop_last=True,
            )
            all_indices.extend(list(sampler))
        counts = Counter(all_indices)
        dups = {i: c for i, c in counts.items() if c > 1}
        assert len(dups) == 0, f"Duplicates: {dups}"

    runner.run_test("Rank: simulated 4-rank unique indices", test_multi_rank_unique)

    # Test 7: set_epoch preserves per-rank sample count and valid indices
    def test_epoch_preserves_count():
        ds = _make_dataset(200)
        ral = RankAwareDataLoader(ds, rank=0, world_size=2, shuffle=True)
        ral.get_sampler()
        ral.set_epoch(0)
        indices_0 = ral.get_all_indices_for_rank()
        n0 = len(indices_0)
        assert all(0 <= i < 200 for i in indices_0), "Invalid index"
        ral.set_epoch(1)
        indices_1 = ral.get_all_indices_for_rank()
        n1 = len(indices_1)
        # Same number of samples per rank across epochs
        assert n0 == n1, f"Per-rank count changed: {n0} vs {n1}"

    runner.run_test("Rank: set_epoch preserves count", test_epoch_preserves_count)

    # Test 8: Different ranks get different indices
    def test_different_ranks():
        ds = _make_dataset(200)
        ral0 = RankAwareDataLoader(ds, rank=0, world_size=4, shuffle=False)
        ral0.get_sampler()
        idx0 = set(ral0.get_all_indices_for_rank())

        ral1 = RankAwareDataLoader(ds, rank=1, world_size=4, shuffle=False)
        ral1.get_sampler()
        idx1 = set(ral1.get_all_indices_for_rank())

        overlap = idx0 & idx1
        # With drop_last=True at sampler level, these should not overlap
        # but default is drop_last=False; still, indices should differ
        assert idx0 != idx1, "Rank 0 and rank 1 got identical indices"

    runner.run_test("Rank: different ranks get different indices", test_different_ranks)

    # Test 9: Uneven dataset size
    def test_uneven_dataset():
        valid, dups = SamplerVerifier.verify_no_duplicates(
            dataset_size=103, world_size=7, drop_last=True,
        )
        assert valid, f"Duplicates for 103 samples / 7 ranks: {dups}"

    runner.run_test("Rank: uneven dataset (103 / 7 ranks)", test_uneven_dataset)

    # Test 10: Large dataset
    def test_large_dataset():
        valid, dups = SamplerVerifier.verify_no_duplicates(
            dataset_size=50000, world_size=8, drop_last=True,
        )
        assert valid, f"Duplicates in large dataset: {len(dups)} dups"

    runner.run_test("Rank: large dataset (50000 / 8 ranks)", test_large_dataset)

    # Test 11: WorkerSeedManager produces different outputs per rank
    def test_worker_seed_ranks():
        mgr0 = WorkerSeedManager(42, rank=0)
        mgr1 = WorkerSeedManager(42, rank=1)
        gen0 = mgr0.get_generator()
        gen1 = mgr1.get_generator()
        v0 = torch.randn(10, generator=gen0)
        v1 = torch.randn(10, generator=gen1)
        assert not torch.equal(v0, v1)

    runner.run_test("Rank: worker seeds differ across ranks", test_worker_seed_ranks)


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 70)
    print("Distributed Scaling — Done-When Gate Validation")
    print("=" * 70)
    print()

    runner = ValidationRunner()

    print("-" * 70)
    print("GATE 1: DDP Training")
    print("-" * 70)
    validate_gate1_ddp(runner)
    print()

    print("-" * 70)
    print("GATE 2: FSDP Sharding")
    print("-" * 70)
    validate_gate2_fsdp(runner)
    print()

    print("-" * 70)
    print("GATE 3: Rank-Aware Loading")
    print("-" * 70)
    validate_gate3_rank_aware(runner)
    print()

    # Summary
    summary = runner.summary()
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    for gate, info in summary.items():
        if gate == "total":
            continue
        status = "PASSED" if info["gate_passed"] else "FAILED"
        print(f"  {gate}: {info['passed']}/{info['total']} tests — {status}")
    print()
    total = summary["total"]
    overall = "ALL GATES PASSED" if total["all_gates_passed"] else "SOME GATES FAILED"
    print(f"  Total: {total['passed']}/{total['total']} tests — {overall}")
    print("=" * 70)

    return 0 if total["all_gates_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
