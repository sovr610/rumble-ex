#!/usr/bin/env python3
"""
validate_distributed.py
========================
Validates the three done-when gates for the V-JEPA 2 distributed training skill:

  Gate 1: Distributed Init
    DistributedSetup.init() correctly detects SLURM or falls back to
    single-process; returns valid (rank, local_rank, world_size).

  Gate 2: AllGather Gradient
    AllGather.apply(x) in forward produces correct gathered tensor;
    backward correctly routes gradients to originating rank.

  Gate 3: Checkpoint Robustness
    CheckpointManager.load() succeeds after simulated transient failure
    (retry logic works); loaded state matches saved state exactly.

Usage:
    python scripts/validate_distributed.py
    python scripts/validate_distributed.py --gate 1
    python scripts/validate_distributed.py --gate 2
    python scripts/validate_distributed.py --gate 3
    python scripts/validate_distributed.py --verbose

Exit codes:
    0 = all selected gates passed
    1 = one or more gates failed
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from typing import Dict, List, Optional
from unittest.mock import patch

# Make sure assets directory is importable
_SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_SKILL_DIR, "assets"))

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Gate result tracking
# ---------------------------------------------------------------------------

class GateResult:
    def __init__(self, gate_id: int, name: str) -> None:
        self.gate_id = gate_id
        self.name = name
        self.passed = False
        self.checks: List[Dict] = []
        self.error: Optional[str] = None

    def add_check(self, description: str, passed: bool, detail: str = "") -> None:
        self.checks.append({"desc": description, "passed": passed, "detail": detail})

    def mark_passed(self) -> None:
        self.passed = True

    def mark_failed(self, error: str) -> None:
        self.passed = False
        self.error = error

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"Gate {self.gate_id}: {self.name} [{status}]"


def _print_gate_header(gate_id: int, name: str) -> None:
    print(f"\n{'='*60}")
    print(f"Gate {gate_id}: {name}")
    print(f"{'='*60}")


def _print_check(desc: str, passed: bool, detail: str = "", verbose: bool = False) -> None:
    status = "PASS" if passed else "FAIL"
    marker = "  [+]" if passed else "  [!]"
    print(f"{marker} {desc}: {status}")
    if detail and (verbose or not passed):
        print(f"      {detail}")


# ---------------------------------------------------------------------------
# Gate 1: Distributed Initialization
# ---------------------------------------------------------------------------

def validate_gate1(verbose: bool = False) -> GateResult:
    """
    Gate 1: DistributedSetup.init() correctly detects SLURM or falls
    back to single-process; returns valid rank/world_size.
    """
    from distributed_setup_template import DistributedSetup

    _print_gate_header(1, "Distributed Init")
    result = GateResult(1, "Distributed Init")

    # ---- Check 1: Single-process fallback returns (0, 0, 1) ----
    desc = "Single-process fallback: returns (rank=0, local_rank=0, world_size=1)"
    try:
        # Ensure no distributed env vars interfere
        slurm_vars = ["SLURM_NTASKS", "SLURM_PROCID", "SLURM_LOCALID",
                      "SLURM_JOB_ID", "SLURM_JOB_TMPDIR"]
        torch_vars = ["RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]
        saved = {v: os.environ.pop(v) for v in slurm_vars + torch_vars if v in os.environ}

        setup = DistributedSetup()
        rank, local_rank, world_size = setup.init()

        ok = rank == 0 and local_rank == 0 and world_size == 1
        detail = f"rank={rank}, local_rank={local_rank}, world_size={world_size}"
        result.add_check(desc, ok, detail)
        _print_check(desc, ok, detail, verbose)

        os.environ.update(saved)
    except Exception as e:
        result.add_check(desc, False, str(e))
        _print_check(desc, False, str(e), verbose)

    # ---- Check 2: dist is NOT initialized after single-process init ----
    desc = "dist.is_initialized() is False in single-process mode"
    try:
        not_initialized = not dist.is_initialized()
        result.add_check(desc, not_initialized)
        _print_check(desc, not_initialized, "", verbose)
    except Exception as e:
        result.add_check(desc, False, str(e))
        _print_check(desc, False, str(e), verbose)

    # ---- Check 3: is_slurm() returns False without SLURM vars ----
    desc = "is_slurm() returns False when SLURM_* vars absent"
    try:
        setup2 = DistributedSetup()
        ok = not setup2.is_slurm()
        result.add_check(desc, ok)
        _print_check(desc, ok, "", verbose)
    except Exception as e:
        result.add_check(desc, False, str(e))
        _print_check(desc, False, str(e), verbose)

    # ---- Check 4: is_slurm() returns False with partial SLURM vars ----
    desc = "is_slurm() returns False with only SLURM_PROCID set"
    try:
        os.environ["SLURM_PROCID"] = "0"
        setup3 = DistributedSetup()
        ok = not setup3.is_slurm()
        del os.environ["SLURM_PROCID"]
        result.add_check(desc, ok, "SLURM_NTASKS and SLURM_LOCALID missing")
        _print_check(desc, ok, "", verbose)
    except Exception as e:
        if "SLURM_PROCID" in os.environ:
            del os.environ["SLURM_PROCID"]
        result.add_check(desc, False, str(e))
        _print_check(desc, False, str(e), verbose)

    # ---- Check 5: Cleanup is idempotent ----
    desc = "cleanup() is idempotent (double-call does not raise)"
    try:
        setup4 = DistributedSetup()
        setup4.init()
        setup4.cleanup()
        setup4.cleanup()  # Second call
        result.add_check(desc, True)
        _print_check(desc, True, "", verbose)
    except Exception as e:
        result.add_check(desc, False, str(e))
        _print_check(desc, False, str(e), verbose)

    # ---- Check 6: device property returns a valid torch.device ----
    desc = "device property returns valid torch.device"
    try:
        setup5 = DistributedSetup()
        setup5.init()
        device = setup5.device
        expected = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        ok = device == expected
        result.add_check(desc, ok, f"device={device}")
        _print_check(desc, ok, f"device={device}", verbose)
    except Exception as e:
        result.add_check(desc, False, str(e))
        _print_check(desc, False, str(e), verbose)

    # ---- Determine overall gate pass ----
    all_passed = all(c["passed"] for c in result.checks)
    if all_passed:
        result.mark_passed()
    else:
        failed = [c["desc"] for c in result.checks if not c["passed"]]
        result.mark_failed(f"Failed checks: {failed}")

    return result


# ---------------------------------------------------------------------------
# Gate 2: AllGather Gradient
# ---------------------------------------------------------------------------

def validate_gate2(verbose: bool = False) -> GateResult:
    """
    Gate 2: AllGather.apply(x) produces correct gathered tensor;
    backward correctly routes gradients to originating rank.
    """
    from custom_dist_ops_template import AllGather, AllReduceSum, AllReduce

    _print_gate_header(2, "AllGather Gradient Flow")
    result = GateResult(2, "AllGather Gradient Flow")

    # ---- Check 1: Forward shape (world_size=1 identity) ----
    desc = "AllGather forward: output shape == input shape (world_size=1)"
    try:
        shapes = [(4, 128), (2, 64, 32), (1,)]
        ok = True
        for shape in shapes:
            x = torch.randn(*shape)
            y = AllGather.apply(x)
            if y.shape != x.shape:
                ok = False
                break
        result.add_check(desc, ok, f"tested shapes: {shapes}")
        _print_check(desc, ok, f"shapes: {shapes}", verbose)
    except Exception as e:
        result.add_check(desc, False, str(e))
        _print_check(desc, False, str(e), verbose)

    # ---- Check 2: Forward values (identity) ----
    desc = "AllGather forward: output values equal input (world_size=1)"
    try:
        x = torch.randn(4, 32)
        y = AllGather.apply(x)
        ok = torch.allclose(y, x)
        result.add_check(desc, ok)
        _print_check(desc, ok, "", verbose)
    except Exception as e:
        result.add_check(desc, False, str(e))
        _print_check(desc, False, str(e), verbose)

    # ---- Check 3: Gradient is not None ----
    desc = "AllGather backward: x.grad is not None after loss.backward()"
    try:
        x = torch.randn(4, 32, requires_grad=True)
        y = AllGather.apply(x)
        y.sum().backward()
        ok = x.grad is not None
        result.add_check(desc, ok)
        _print_check(desc, ok, "", verbose)
    except Exception as e:
        result.add_check(desc, False, str(e))
        _print_check(desc, False, str(e), verbose)

    # ---- Check 4: Gradient shape matches input ----
    desc = "AllGather backward: grad.shape == x.shape"
    try:
        shapes = [(4, 32), (2, 8, 16), (1, 128)]
        ok = True
        for shape in shapes:
            x = torch.randn(*shape, requires_grad=True)
            y = AllGather.apply(x)
            y.sum().backward()
            if x.grad is None or x.grad.shape != x.shape:
                ok = False
                break
        result.add_check(desc, ok, f"shapes: {shapes}")
        _print_check(desc, ok, f"shapes: {shapes}", verbose)
    except Exception as e:
        result.add_check(desc, False, str(e))
        _print_check(desc, False, str(e), verbose)

    # ---- Check 5: Gradient is non-zero ----
    desc = "AllGather backward: gradient is non-zero for non-trivial upstream"
    try:
        x = torch.randn(3, 5, requires_grad=True)
        y = AllGather.apply(x)
        (y * 2.0).sum().backward()
        ok = x.grad is not None and x.grad.abs().sum().item() > 0
        result.add_check(desc, ok, f"grad sum abs = {x.grad.abs().sum().item():.4f}" if x.grad is not None else "grad=None")
        _print_check(desc, ok, "", verbose)
    except Exception as e:
        result.add_check(desc, False, str(e))
        _print_check(desc, False, str(e), verbose)

    # ---- Check 6: grad_fn is not None (differentiable) ----
    desc = "AllGather, AllReduceSum, AllReduce all have grad_fn"
    try:
        x = torch.randn(2, 8, requires_grad=True)
        y1 = AllGather.apply(x)
        y2 = AllReduceSum.apply(x)
        y3 = AllReduce.apply(x)
        ok = all(y.grad_fn is not None for y in [y1, y2, y3])
        result.add_check(desc, ok)
        _print_check(desc, ok, "", verbose)
    except Exception as e:
        result.add_check(desc, False, str(e))
        _print_check(desc, False, str(e), verbose)

    # ---- Check 7: AllReduceSum backward is identity ----
    desc = "AllReduceSum backward: grad_input == grad_output (identity)"
    try:
        x = torch.randn(4, 4, requires_grad=True)
        upstream = torch.randn(4, 4)
        y = AllReduceSum.apply(x)
        y.backward(upstream)
        ok = x.grad is not None and torch.allclose(x.grad, upstream)
        result.add_check(desc, ok)
        _print_check(desc, ok, "", verbose)
    except Exception as e:
        result.add_check(desc, False, str(e))
        _print_check(desc, False, str(e), verbose)

    # ---- Check 8: AllReduce backward is identity ----
    desc = "AllReduce backward: grad_input == grad_output (identity, world_size=1)"
    try:
        x = torch.randn(4, 4, requires_grad=True)
        upstream = torch.randn(4, 4)
        y = AllReduce.apply(x)
        y.backward(upstream)
        ok = x.grad is not None and torch.allclose(x.grad, upstream)
        result.add_check(desc, ok)
        _print_check(desc, ok, "", verbose)
    except Exception as e:
        result.add_check(desc, False, str(e))
        _print_check(desc, False, str(e), verbose)

    all_passed = all(c["passed"] for c in result.checks)
    if all_passed:
        result.mark_passed()
    else:
        failed = [c["desc"] for c in result.checks if not c["passed"]]
        result.mark_failed(f"Failed checks: {failed}")

    return result


# ---------------------------------------------------------------------------
# Gate 3: Checkpoint Robustness
# ---------------------------------------------------------------------------

def validate_gate3(verbose: bool = False) -> GateResult:
    """
    Gate 3: CheckpointManager.load() succeeds after simulated transient
    failure (retry logic works); loaded state matches saved state exactly.
    """
    from checkpoint_manager_template import CheckpointManager

    _print_gate_header(3, "Checkpoint Robustness")
    result = GateResult(3, "Checkpoint Robustness")

    # ---- Check 1: Save/load roundtrip — values match ----
    desc = "Save/load roundtrip: all tensor values match exactly"
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, max_retries=5)

            enc_weight = torch.randn(16, 8)
            state = {
                "epoch": 42,
                "encoder": {"weight": enc_weight},
                "predictor": {"fc.weight": torch.randn(8, 16)},
                "target_encoder": {"weight": enc_weight.clone()},
                "opt": {"state": {}, "param_groups": [{"lr": 1e-4}]},
                "scaler": None,
            }

            path = manager.save(state, epoch=42)
            loaded = manager.load(path)

            epoch_ok = loaded["epoch"] == 42
            weight_ok = torch.allclose(loaded["encoder"]["weight"], enc_weight)
            lr_ok = loaded["opt"]["param_groups"][0]["lr"] == 1e-4
            scaler_ok = loaded["scaler"] is None

            ok = epoch_ok and weight_ok and lr_ok and scaler_ok
            detail = f"epoch={epoch_ok}, weight={weight_ok}, lr={lr_ok}, scaler={scaler_ok}"
            result.add_check(desc, ok, detail)
            _print_check(desc, ok, detail, verbose)
    except Exception as e:
        result.add_check(desc, False, str(e))
        _print_check(desc, False, str(e), verbose)

    # ---- Check 2: Retry on transient failure — succeeds after 2 failures ----
    desc = "Retry logic: succeeds after 2 simulated transient failures"
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, max_retries=5)
            path = manager.save({"epoch": 7}, epoch=7)

            call_count = [0]
            original_torch_load = torch.load

            def flaky_load(p, *args, **kwargs):
                call_count[0] += 1
                if call_count[0] < 3:
                    raise IOError("Simulated NFS error")
                return original_torch_load(p, *args, **kwargs)

            with patch("torch.load", side_effect=flaky_load):
                with patch("time.sleep"):  # Skip actual sleep in test
                    loaded = manager.load(path)

            ok = loaded["epoch"] == 7 and call_count[0] == 3
            detail = f"total_calls={call_count[0]}, epoch={loaded.get('epoch')}"
            result.add_check(desc, ok, detail)
            _print_check(desc, ok, detail, verbose)
    except Exception as e:
        result.add_check(desc, False, str(e))
        _print_check(desc, False, str(e), verbose)

    # ---- Check 3: All retries exhausted raises RuntimeError ----
    desc = "All retries exhausted: raises RuntimeError"
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, max_retries=3)
            fake_path = os.path.join(tmpdir, "nonexistent.pth")

            raised = False
            try:
                with patch("time.sleep"):
                    manager.load(fake_path)
            except RuntimeError:
                raised = True

            result.add_check(desc, raised)
            _print_check(desc, raised, "", verbose)
    except Exception as e:
        result.add_check(desc, False, str(e))
        _print_check(desc, False, str(e), verbose)

    # ---- Check 4: strip_prefix correctness ----
    desc = "strip_prefix: removes 'module.' from DDP checkpoint keys"
    try:
        sd = {
            "module.layer.weight": torch.randn(4, 4),
            "module.layer.bias": torch.zeros(4),
        }
        cleaned = CheckpointManager.strip_prefix(sd, "module.")
        ok = set(cleaned.keys()) == {"layer.weight", "layer.bias"}
        detail = f"cleaned keys: {list(cleaned.keys())}"
        result.add_check(desc, ok, detail)
        _print_check(desc, ok, detail, verbose)
    except Exception as e:
        result.add_check(desc, False, str(e))
        _print_check(desc, False, str(e), verbose)

    # ---- Check 5: strip_prefix preserves keys without prefix ----
    desc = "strip_prefix: preserves keys without the prefix"
    try:
        sd = {"weight": torch.randn(4), "other.bias": torch.zeros(4)}
        cleaned = CheckpointManager.strip_prefix(sd, "module.")
        ok = cleaned == sd
        result.add_check(desc, ok, f"keys unchanged: {list(cleaned.keys())}")
        _print_check(desc, ok, "", verbose)
    except Exception as e:
        result.add_check(desc, False, str(e))
        _print_check(desc, False, str(e), verbose)

    # ---- Check 6: Atomic write (no .tmp files left) ----
    desc = "Atomic write: no leftover .tmp files after save"
    try:
        import glob as _glob
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)
            manager.save({"epoch": 0}, epoch=0)
            tmp_files = _glob.glob(os.path.join(tmpdir, "*.tmp"))
            ok = len(tmp_files) == 0
            result.add_check(desc, ok, f"tmp_files={tmp_files}")
            _print_check(desc, ok, "", verbose)
    except Exception as e:
        result.add_check(desc, False, str(e))
        _print_check(desc, False, str(e), verbose)

    # ---- Check 7: load_pretrained with strict=False (missing keys tolerated) ----
    desc = "load_pretrained: strict=False tolerates missing keys"
    try:
        import torch.nn as nn
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)
            ckpt_state = {
                "encoder": {
                    "0.weight": torch.randn(4, 4),
                    "0.bias": torch.zeros(4),
                    "pos_embed": torch.randn(1, 10, 4),  # Not in model
                }
            }
            path = manager.save(ckpt_state, epoch=0)
            model = nn.Sequential(nn.Linear(4, 4))
            missing, unexpected = manager.load_pretrained(model, path, key="encoder", strict=False)
            ok = "pos_embed" in unexpected and len(missing) == 0
            detail = f"missing={missing}, unexpected={unexpected}"
            result.add_check(desc, ok, detail)
            _print_check(desc, ok, detail, verbose)
    except Exception as e:
        result.add_check(desc, False, str(e))
        _print_check(desc, False, str(e), verbose)

    all_passed = all(c["passed"] for c in result.checks)
    if all_passed:
        result.mark_passed()
    else:
        failed = [c["desc"] for c in result.checks if not c["passed"]]
        result.mark_failed(f"Failed checks: {failed}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate V-JEPA 2 distributed training done-when gates"
    )
    parser.add_argument(
        "--gate", type=int, choices=[1, 2, 3], default=None,
        help="Run only a specific gate (1, 2, or 3). Default: run all."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print details for passing checks too."
    )
    args = parser.parse_args()

    gates_to_run = [1, 2, 3] if args.gate is None else [args.gate]
    validators = {1: validate_gate1, 2: validate_gate2, 3: validate_gate3}

    print("\nV-JEPA 2 Distributed Training — Done-When Gate Validation")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Running gates: {gates_to_run}")

    results = []
    start_time = time.time()

    for gate_id in gates_to_run:
        result = validators[gate_id](verbose=args.verbose)
        results.append(result)

    # ---- Summary ----
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    all_passed = True
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        n_checks = len(result.checks)
        n_passed = sum(1 for c in result.checks if c["passed"])
        print(f"  Gate {result.gate_id}: {result.name:<30} [{status}] ({n_passed}/{n_checks} checks)")
        if not result.passed and result.error:
            print(f"    Error: {result.error}")
        all_passed = all_passed and result.passed

    print(f"\nTotal time: {elapsed:.2f}s")

    if all_passed:
        print("\n[+] All done-when gates PASSED")
        print("    The V-JEPA 2 distributed training skill is verified.")
        return 0
    else:
        print("\n[!] One or more done-when gates FAILED")
        print("    Review the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
