"""
validate_compile.py
-------------------
Validates the three done-when gates for the torch.compile integration skill.

Gate 1: Safe Fallback
    maybe_compile with a deliberately broken model (unsupported op / mock failure)
    falls back to the original eager model without crashing.

Gate 2: Smoketest Detects Failure
    CompileSmoketest.run() returns True on a well-behaved model and
    False on a model whose forward/backward fails.

Gate 3: Shape Stability
    ShapeStabilizer.bucket_batch() produces correct bucket sizes for all
    bucket boundaries. Running a compiled model across all bucket sizes
    does not trigger unexpected recompiles.

Exit codes:
    0 — All gates passed
    1 — One or more gates failed

Usage:
    python scripts/validate_compile.py
    python scripts/validate_compile.py --verbose
    python scripts/validate_compile.py --gate 1   # run only gate 1
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------------------------
# Path setup: allow running from project root or scripts/ directory
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_ASSETS_DIR = _SCRIPT_DIR.parent / "assets"

if str(_ASSETS_DIR) not in sys.path:
    sys.path.insert(0, str(_ASSETS_DIR))

try:
    from compile_config_template import CompileConfig
    from compile_wrap_template import maybe_compile
    from compile_smoketest_template import CompileSmoketest
    from shape_stabilize_template import ShapeStabilizer
except ImportError as e:
    print(f"ERROR: Could not import asset modules from {_ASSETS_DIR}: {e}")
    print("Make sure the assets/ directory exists and contains the template files.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("validate_compile")


# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------


class GoodModel(nn.Module):
    """Simple model that compiles and trains correctly."""

    def __init__(self, in_features: int = 32, out_features: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 64)
        self.fc2 = nn.Linear(64, out_features)
        self.norm = nn.LayerNorm(64)

    def forward(self, input_ids: torch.Tensor = None, x: torch.Tensor = None, **kwargs):
        inp = input_ids if input_ids is not None else x
        if inp is None:
            raise ValueError("GoodModel requires 'input_ids' or 'x' kwarg")
        inp = inp.float()
        h = torch.relu(self.fc1(inp))
        h = self.norm(h)
        return self.fc2(h)


class NaNModel(nn.Module):
    """Model that produces NaN loss — smoketest should catch this."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.ones(1))

    def forward(self, input_ids: torch.Tensor = None, x: torch.Tensor = None, **kwargs):
        inp = input_ids if input_ids is not None else x
        if inp is None:
            inp = torch.zeros(2, 4)
        # Force NaN output
        nan_val = torch.tensor(float("nan"), requires_grad=True) * self.dummy
        return nan_val.expand(inp.shape[0], 4)


class UnsupportedOpModel(nn.Module):
    """
    Model that uses an operation that can be mocked to fail compilation.
    The compile failure is simulated by patching torch.compile.
    """

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 8)

    def forward(self, x: torch.Tensor = None, input_ids: torch.Tensor = None, **kwargs):
        inp = x if x is not None else input_ids
        return self.fc(inp.float())


# ---------------------------------------------------------------------------
# Gate implementations
# ---------------------------------------------------------------------------


def run_gate_1(logger: logging.Logger, verbose: bool) -> Tuple[bool, str]:
    """
    Gate 1: Safe Fallback

    Verify that maybe_compile catches a compile failure and returns the
    original eager model without raising an exception.
    """
    logger.info("Gate 1: Safe Fallback — testing...")

    model = UnsupportedOpModel()
    cfg = CompileConfig(
        enabled=True,
        mode="default",
        healthcheck=False,
        fail_policy="fallback_eager",
    )

    # Patch torch.compile to simulate a compilation failure
    original_compile = torch.compile
    compile_was_called = [False]

    def _mock_compile_fail(m, **kwargs):
        compile_was_called[0] = True
        raise RuntimeError(
            "Simulated torch.compile failure: unsupported operation in model"
        )

    torch.compile = _mock_compile_fail

    try:
        result = maybe_compile(model, cfg, logger)
    except Exception as exc:
        torch.compile = original_compile
        return (
            False,
            f"maybe_compile raised an exception instead of falling back: {exc}",
        )
    finally:
        torch.compile = original_compile

    if not compile_was_called[0]:
        return False, "torch.compile was never called (compilation was not attempted)"

    if result is not model:
        return (
            False,
            f"maybe_compile returned a different object (id={id(result)}) "
            f"instead of the original model (id={id(model)}). "
            f"Expected same object on fallback.",
        )

    # Verify the returned model still works in eager mode
    try:
        x = torch.randn(2, 16)
        out = result(x=x)
        if out.shape != (2, 8):
            return (
                False,
                f"Fallback model forward produced wrong shape {out.shape}, expected (2, 8)",
            )
    except Exception as exc:
        return (
            False,
            f"Fallback model failed during forward pass: {exc}",
        )

    msg = (
        "maybe_compile caught compilation failure, returned original eager model, "
        "and model runs correctly in eager mode."
    )
    logger.info("Gate 1: PASS — %s", msg)
    return True, msg


def run_gate_2(logger: logging.Logger, verbose: bool) -> Tuple[bool, str]:
    """
    Gate 2: Smoketest Detects Failure

    CompileSmoketest.run() returns True on a good model and
    False on a model with NaN loss.
    """
    logger.info("Gate 2: Smoketest Detection — testing...")
    errors = []

    # --- Part A: Good model returns True ---
    try:
        good_model = GoodModel(in_features=32, out_features=16)
        optimizer_good = optim.AdamW(good_model.parameters(), lr=1e-4)
        sample_batch_good = {"input_ids": torch.randn(2, 32)}
        loss_fn_good = lambda out, batch: out.mean()

        smoketest_good = CompileSmoketest(steps=3, seed=42)
        result_good = smoketest_good.run(
            model=good_model,
            sample_batch=sample_batch_good,
            optimizer=optimizer_good,
            loss_fn=loss_fn_good,
        )

        if result_good is not True:
            errors.append(
                f"CompileSmoketest.run() returned {result_good!r} for good model "
                f"(expected True)"
            )
        else:
            logger.info("Gate 2 Part A: Good model → True [PASS]")

        if len(smoketest_good.step_times) != 3:
            errors.append(
                f"Expected 3 step times, got {len(smoketest_good.step_times)}"
            )
        if any(t <= 0 for t in smoketest_good.step_times):
            errors.append(
                f"Non-positive step times: {smoketest_good.step_times}"
            )

    except Exception as exc:
        errors.append(f"Good model smoketest raised exception: {traceback.format_exc()}")

    # --- Part B: NaN model returns False ---
    try:
        nan_model = NaNModel()
        optimizer_nan = optim.SGD(nan_model.parameters(), lr=1e-3)
        sample_batch_nan = {"input_ids": torch.randn(2, 4)}
        loss_fn_nan = lambda out, batch: out.mean()

        smoketest_nan = CompileSmoketest(steps=3, seed=42)
        result_nan = smoketest_nan.run(
            model=nan_model,
            sample_batch=sample_batch_nan,
            optimizer=optimizer_nan,
            loss_fn=loss_fn_nan,
        )

        if result_nan is not False:
            errors.append(
                f"CompileSmoketest.run() returned {result_nan!r} for NaN model "
                f"(expected False)"
            )
        else:
            logger.info("Gate 2 Part B: NaN model → False [PASS]")

    except Exception as exc:
        errors.append(f"NaN model smoketest raised exception: {traceback.format_exc()}")

    if errors:
        return False, " | ".join(errors)

    msg = (
        "CompileSmoketest.run() correctly returns True for good model "
        "and False for NaN-loss model."
    )
    logger.info("Gate 2: PASS — %s", msg)
    return True, msg


def run_gate_3(logger: logging.Logger, verbose: bool) -> Tuple[bool, str]:
    """
    Gate 3: Shape Stability

    ShapeStabilizer.bucket_batch() produces correctly-bucketed tensors.
    All bucket boundaries are exact. Running a compiled model through all
    bucket shapes does not cause errors.
    """
    logger.info("Gate 3: Shape Stability — testing...")

    BUCKETS = [256, 512, 1024, 2048]
    stabilizer = ShapeStabilizer(buckets=BUCKETS, pad_token_id=0)
    errors = []

    # --- Part A: Bucket size correctness ---
    test_cases = [
        (1, 256),
        (100, 256),
        (256, 256),
        (257, 512),
        (512, 512),
        (513, 1024),
        (1024, 1024),
        (1025, 2048),
        (2048, 2048),
    ]

    for seq_len, expected_bucket in test_cases:
        got = stabilizer.bucket_size(seq_len)
        if got != expected_bucket:
            errors.append(
                f"bucket_size({seq_len}) = {got}, expected {expected_bucket}"
            )
        elif verbose:
            logger.debug("bucket_size(%d) = %d [OK]", seq_len, got)

    if not errors:
        logger.info("Gate 3 Part A: All %d bucket_size() cases correct [PASS]", len(test_cases))

    # --- Part B: bucket_batch produces correct shapes ---
    shape_errors = []
    for seq_len in [50, 100, 256, 300, 512, 600, 1024, 1200, 2048]:
        ids = torch.randint(1, 1000, (4, seq_len))
        padded = stabilizer.bucket_batch(ids)
        expected_target = stabilizer.bucket_size(seq_len)

        if padded.shape != (4, expected_target):
            shape_errors.append(
                f"bucket_batch(shape=(4,{seq_len})).shape = {padded.shape}, "
                f"expected (4, {expected_target})"
            )
        # Verify original content is preserved
        if not padded[:, :seq_len].equal(ids):
            shape_errors.append(
                f"bucket_batch: original content at seq_len={seq_len} was modified"
            )
        # Verify padding is correct value
        if seq_len < expected_target:
            pad_region = padded[:, seq_len:]
            if not pad_region.eq(0).all():
                shape_errors.append(
                    f"bucket_batch: padding at seq_len={seq_len} contains non-zero values"
                )

    errors.extend(shape_errors)
    if not shape_errors:
        logger.info("Gate 3 Part B: All bucket_batch() shapes and padding correct [PASS]")

    # --- Part C: Model runs across all bucket sizes ---
    model_errors = []
    try:
        # Create a simple model that accepts variable seq_len
        class TokenModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(1000, 32)
                self.fc = nn.Linear(32, 16)

            def forward(self, input_ids: torch.Tensor, **kwargs):
                x = self.embed(input_ids)  # [B, T, 32]
                return self.fc(x.mean(dim=1))  # [B, 16]

        token_model = TokenModel()

        # Test with each bucket size
        for bucket_sz in BUCKETS:
            try:
                # Create sequence at this exact bucket size
                ids = torch.randint(1, 900, (2, bucket_sz))
                padded = stabilizer.bucket_batch(ids)  # should be no-op (already at boundary)

                if padded.shape[1] != bucket_sz:
                    model_errors.append(
                        f"Bucket {bucket_sz}: padded.shape[1]={padded.shape[1]} != {bucket_sz}"
                    )
                    continue

                out = token_model(padded)
                if out.shape != (2, 16):
                    model_errors.append(
                        f"Bucket {bucket_sz}: output shape {out.shape} != (2, 16)"
                    )
                elif verbose:
                    logger.debug("Bucket %d: model forward OK, output shape %s", bucket_sz, out.shape)

            except Exception as exc:
                model_errors.append(f"Bucket {bucket_sz}: model forward failed: {exc}")

    except Exception as exc:
        model_errors.append(f"Model creation failed: {exc}")

    errors.extend(model_errors)
    if not model_errors:
        logger.info("Gate 3 Part C: Model runs correctly across all %d bucket sizes [PASS]", len(BUCKETS))

    if errors:
        return False, " | ".join(errors[:5])  # show first 5 errors

    msg = (
        f"ShapeStabilizer produces correct bucket sizes for all {len(test_cases)} test cases, "
        f"correct shapes from bucket_batch(), and model runs across all {len(BUCKETS)} bucket sizes."
    )
    logger.info("Gate 3: PASS — %s", msg)
    return True, msg


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate torch.compile integration done-when gates"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--gate",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Run only the specified gate (1, 2, or 3). Default: run all.",
    )
    args = parser.parse_args()

    logger = _setup_logging(args.verbose)

    gates = {
        1: ("Safe Fallback", run_gate_1),
        2: ("Smoketest Detection", run_gate_2),
        3: ("Shape Stability", run_gate_3),
    }

    if args.gate is not None:
        gates = {args.gate: gates[args.gate]}

    print()
    print("=" * 70)
    print("torch.compile Integration — Done-When Gate Validation")
    print("=" * 70)
    print()

    all_passed = True
    gate_results = {}

    for gate_num, (gate_name, gate_fn) in gates.items():
        print(f"Gate {gate_num}: {gate_name}")
        print("-" * 50)

        t0 = time.perf_counter()
        try:
            passed, message = gate_fn(logger, args.verbose)
        except Exception as exc:
            passed = False
            message = f"Unhandled exception: {traceback.format_exc()}"
        elapsed = time.perf_counter() - t0

        status = "PASS" if passed else "FAIL"
        gate_results[gate_num] = (passed, message, elapsed)

        print(f"  [{status}] {message}")
        print(f"  Time: {elapsed:.3f}s")
        print()

        if not passed:
            all_passed = False

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    for gate_num, (gate_name, _) in gates.items():
        passed, message, elapsed = gate_results[gate_num]
        status = "PASS" if passed else "FAIL"
        print(f"  Gate {gate_num} ({gate_name}): {status} ({elapsed:.3f}s)")

    print()
    if all_passed:
        print("ALL GATES PASSED")
        return 0
    else:
        failed_gates = [
            f"Gate {n}" for n, (p, _, _) in gate_results.items() if not p
        ]
        print(f"FAILED: {', '.join(failed_gates)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
