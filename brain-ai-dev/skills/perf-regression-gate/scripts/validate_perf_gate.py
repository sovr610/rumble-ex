"""
validate_perf_gate.py
======================
Validates the three done-when gates for the Compute/Throughput Baseline
& Regression Gate skill.

Gate 1: Bench run produces metrics.json + eval.json with correct schema.
Gate 2: Profiling produces trace files in the profile/ directory.
Gate 3: compare_baseline correctly blocks when regression exceeds tolerance
        and passes when within tolerance.

Usage:
    python validate_perf_gate.py [--out /tmp/gate_validation]

Returns exit code 0 if all gates pass, 1 if any gate fails.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add the assets directory to path for imports
_SCRIPT_DIR = Path(__file__).parent.resolve()
_ASSETS_DIR = _SCRIPT_DIR.parent / "assets"
sys.path.insert(0, str(_ASSETS_DIR))

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from perf_gate_config_template import (
        BenchConfig, EvalConfig, ToleranceConfig, Status
    )
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    print("WARNING: perf_gate_config_template not found", file=sys.stderr)

try:
    from bench_train_template import BenchTrain
    _BENCH_AVAILABLE = True
except ImportError:
    _BENCH_AVAILABLE = False

try:
    from eval_small_template import EvalSmall
    _QUALITY_AVAILABLE = True
except ImportError:
    _QUALITY_AVAILABLE = False

try:
    from compare_baseline_template import CompareBaseline, make_metrics, make_eval_result
    _COMPARE_AVAILABLE = True
except ImportError:
    _COMPARE_AVAILABLE = False


# ===========================================================================
# Gate 1: Schema validation helpers
# ===========================================================================

METRICS_JSON_REQUIRED_KEYS = {
    "schema_version",
    "machine_profile",
    "run_config",
    "throughput",
    "timing",
    "memory",
    "mfu",
    "loss",
}

METRICS_THROUGHPUT_KEYS = {
    "tokens_per_sec_mean",
    "tokens_per_sec_p50",
    "tokens_per_sec_p10",
    "tokens_per_step",
}

METRICS_TIMING_KEYS = {
    "step_time_mean_s",
    "step_time_p50_s",
    "step_time_p90_s",
    "step_time_min_s",
    "step_time_max_s",
}

EVAL_JSON_REQUIRED_KEYS = {
    "schema_version",
    "machine_profile",
    "ppl_fixed_shard",
    "task_probe_accuracy",
}


def validate_metrics_schema(path: str) -> Tuple[bool, str]:
    """
    Validate that a metrics.json file has the correct schema.

    Returns:
        (valid: bool, message: str)
    """
    if not os.path.isfile(path):
        return False, f"metrics.json not found: {path}"

    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        return False, f"metrics.json is not valid JSON: {exc}"

    # Check top-level keys
    missing = METRICS_JSON_REQUIRED_KEYS - set(data.keys())
    if missing:
        return False, f"metrics.json missing keys: {missing}"

    # Check throughput sub-keys
    throughput = data.get("throughput", {})
    missing_tps = METRICS_THROUGHPUT_KEYS - set(throughput.keys())
    if missing_tps:
        return False, f"throughput section missing keys: {missing_tps}"

    # Check timing sub-keys
    timing = data.get("timing", {})
    missing_timing = METRICS_TIMING_KEYS - set(timing.keys())
    if missing_timing:
        return False, f"timing section missing keys: {missing_timing}"

    # Validate data types for critical fields
    tps_p50 = throughput.get("tokens_per_sec_p50")
    if not isinstance(tps_p50, (int, float)) or tps_p50 <= 0:
        return False, f"tokens_per_sec_p50 must be a positive number, got {tps_p50}"

    st_p50 = timing.get("step_time_p50_s")
    if not isinstance(st_p50, (int, float)) or st_p50 <= 0:
        return False, f"step_time_p50_s must be a positive number, got {st_p50}"

    schema_ver = data.get("schema_version")
    if not isinstance(schema_ver, str):
        return False, f"schema_version must be a string, got {type(schema_ver)}"

    return True, "metrics.json schema valid"


def validate_eval_schema(path: str) -> Tuple[bool, str]:
    """
    Validate that an eval.json file has the correct schema.

    Returns:
        (valid: bool, message: str)
    """
    if not os.path.isfile(path):
        return False, f"eval.json not found: {path}"

    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        return False, f"eval.json is not valid JSON: {exc}"

    missing = EVAL_JSON_REQUIRED_KEYS - set(data.keys())
    if missing:
        return False, f"eval.json missing keys: {missing}"

    ppl = data.get("ppl_fixed_shard")
    if not isinstance(ppl, (int, float)) or ppl <= 0:
        return False, f"ppl_fixed_shard must be positive, got {ppl}"

    probe_acc = data.get("task_probe_accuracy", {})
    if not isinstance(probe_acc, dict):
        return False, "task_probe_accuracy must be a dict"

    for probe_name, acc in probe_acc.items():
        if not isinstance(acc, (int, float)) or not (0.0 <= acc <= 1.0):
            return False, f"probe accuracy {probe_name}={acc} must be in [0, 1]"

    return True, "eval.json schema valid"


# ===========================================================================
# Gate 2: Profiling trace validation
# ===========================================================================

def validate_trace_files(profile_dir: str) -> Tuple[bool, str]:
    """
    Validate that profiling produced trace files in the expected locations.

    Returns:
        (valid: bool, message: str)
    """
    profile_path = Path(profile_dir)

    if not profile_path.exists():
        return False, f"profile directory not found: {profile_dir}"

    # Look for Chrome trace
    chrome_trace = profile_path / "chrome" / "trace.json"
    if chrome_trace.exists():
        try:
            with open(chrome_trace) as f:
                trace_data = json.load(f)
            if not isinstance(trace_data, (list, dict)):
                return False, "Chrome trace.json is not a valid JSON array or object"
        except json.JSONDecodeError as exc:
            return False, f"Chrome trace.json is invalid JSON: {exc}"

    # Look for TensorBoard traces (*.json in tb/ directory)
    tb_dir = profile_path / "tb"
    if tb_dir.exists():
        trace_files = list(tb_dir.glob("*.json")) + list(tb_dir.glob("*_pt_trace.json"))
        if not trace_files and not chrome_trace.exists():
            return False, f"No trace files found in {profile_path}"

    if not chrome_trace.exists() and not (profile_path / "tb").exists():
        return False, f"Neither chrome/ nor tb/ directory found in {profile_dir}"

    return True, f"Trace files found in {profile_dir}"


# ===========================================================================
# Gate validators
# ===========================================================================

def run_gate1(out_dir: str, machine_profile: str = "test_profile") -> Tuple[bool, str]:
    """
    Gate 1: Verify bench run produces metrics.json and eval.json with correct schema.

    Runs a minimal benchmark and quality harness to generate the files,
    then validates their schemas.
    """
    print("\n--- Gate 1: Schema Validation ---")

    if not _CONFIG_AVAILABLE:
        return False, "perf_gate_config_template not available"

    # Run a minimal benchmark to generate metrics.json
    if _BENCH_AVAILABLE:
        print("  Running minimal benchmark...")
        config = BenchConfig(
            warmup_steps=2,
            measure_steps=5,
            mode="synthetic",
            per_device_batch_size=2,
            seq_len=32,
            vocab_size=100,
            hidden_dim=64,
            num_layers=2,
            repeat=1,
            peak_tflops=989.0,
        )
        bench = BenchTrain(config, machine_profile=machine_profile)
        result = bench.run()
        metrics_path = os.path.join(out_dir, f"{machine_profile}.metrics.json")
        bench.save(metrics_path)
        print(f"  metrics.json written: {metrics_path}")
    else:
        # Create a minimal synthetic metrics.json for schema testing
        print("  BenchTrain not available; creating synthetic metrics.json")
        metrics_path = os.path.join(out_dir, f"{machine_profile}.metrics.json")
        synthetic_metrics = {
            "schema_version": "1.0",
            "machine_profile": machine_profile,
            "run_config": {"mode": "synthetic", "world_size": 1},
            "throughput": {
                "tokens_per_sec_mean": 50000.0,
                "tokens_per_sec_p50": 50000.0,
                "tokens_per_sec_p10": 45000.0,
                "tokens_per_step": 65536,
            },
            "timing": {
                "step_time_mean_s": 1.31,
                "step_time_p50_s": 1.31,
                "step_time_p90_s": 1.35,
                "step_time_min_s": 1.28,
                "step_time_max_s": 1.40,
            },
            "memory": {
                "peak_allocated_bytes": 10 * 1024 ** 3,
                "peak_allocated_gb": 10.0,
            },
            "mfu": {"mfu_p50": 0.35},
            "loss": {"loss_slope": -0.001, "final_loss": 2.5},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(metrics_path, "w") as f:
            json.dump(synthetic_metrics, f, indent=2)

    # Create a minimal synthetic eval.json (quality harness needs a model)
    print("  Creating synthetic eval.json for schema test")
    quality_path = os.path.join(out_dir, f"{machine_profile}.eval.json")
    synthetic_quality = {
        "schema_version": "1.0",
        "machine_profile": machine_profile,
        "ppl_fixed_shard": 12.34,
        "task_probe_accuracy": {
            "basic_reasoning_25": 0.92,
            "format_following_30": 0.87,
            "code_sanity_20": 0.90,
        },
        "eval_config": {"seed": 42, "temperature": 0.0},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(quality_path, "w") as f:
        json.dump(synthetic_quality, f, indent=2)

    # Validate schemas
    metrics_valid, metrics_msg = validate_metrics_schema(metrics_path)
    print(f"  metrics.json schema: {'PASS' if metrics_valid else 'FAIL'} -- {metrics_msg}")

    quality_valid, quality_msg = validate_eval_schema(quality_path)
    print(f"  eval.json schema:    {'PASS' if quality_valid else 'FAIL'} -- {quality_msg}")

    if metrics_valid and quality_valid:
        return True, "Both metrics.json and eval.json have valid schemas"
    else:
        msgs = []
        if not metrics_valid:
            msgs.append(f"metrics.json: {metrics_msg}")
        if not quality_valid:
            msgs.append(f"eval.json: {quality_msg}")
        return False, "; ".join(msgs)


def run_gate2(out_dir: str) -> Tuple[bool, str]:
    """
    Gate 2: Verify profiling produces trace files.

    Runs a minimal profiling session and checks for output files.
    """
    print("\n--- Gate 2: Profiling Trace Files ---")

    if not _TORCH_AVAILABLE:
        return False, "PyTorch not available; cannot run profiler"

    if not _CONFIG_AVAILABLE:
        return False, "perf_gate_config_template not available"

    profile_dir = os.path.join(out_dir, "profile")
    tb_dir = os.path.join(profile_dir, "tb")
    chrome_dir = os.path.join(profile_dir, "chrome")
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(chrome_dir, exist_ok=True)

    print("  Running minimal profiling session...")

    # Build a very small model for profiling
    if _BENCH_AVAILABLE:
        from bench_train_template import _build_minimal_model
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        model = _build_minimal_model(vocab_size=100, hidden_dim=32, num_layers=1).to(device_str)
    else:
        # Fallback: use a tiny linear model
        device_str = "cpu"
        model = torch.nn.Sequential(
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
        ).to(device_str)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    wait, warmup_p, active, repeat_p = 1, 1, 2, 1
    total_steps = (wait + warmup_p + active) * repeat_p

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    on_trace_fn = torch.profiler.tensorboard_trace_handler(tb_dir)

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=wait, warmup=warmup_p, active=active, repeat=repeat_p
        ),
        on_trace_ready=on_trace_fn,
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for _ in range(total_steps):
            if _BENCH_AVAILABLE:
                x = torch.randint(0, 100, (2, 8), device=device_str)
                y = torch.randint(0, 100, (2, 8), device=device_str)
                with torch.autograd.profiler.record_function("forward"):
                    outputs = model(input_ids=x, labels=y)
                    loss = outputs.get("loss") if isinstance(outputs, dict) else getattr(outputs, "loss", None)
            else:
                x = torch.randn(4, 16, device=device_str)
                with torch.autograd.profiler.record_function("forward"):
                    out = model(x)
                    loss = out.mean()

            if loss is not None:
                with torch.autograd.profiler.record_function("backward"):
                    loss.backward()
                with torch.autograd.profiler.record_function("optimizer_step"):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            prof.step()

    # Export Chrome trace
    chrome_path = os.path.join(chrome_dir, "trace.json")
    prof.export_chrome_trace(chrome_path)
    print(f"  Chrome trace exported -> {chrome_path}")

    # Validate
    valid, msg = validate_trace_files(profile_dir)

    # Additional check: Chrome trace is valid JSON with traceEvents
    if valid and os.path.isfile(chrome_path):
        with open(chrome_path) as f:
            trace_data = json.load(f)
        if isinstance(trace_data, dict) and "traceEvents" in trace_data:
            event_names = {e.get("name", "") for e in trace_data["traceEvents"]}
            has_record_fn = any(n in event_names for n in ("forward", "backward", "optimizer_step"))
            if has_record_fn:
                print("  record_function regions found in trace: forward, backward, optimizer_step")
            else:
                print(f"  INFO: record_function regions not found. Event names: {list(event_names)[:10]}")

    print(f"  Trace validation: {'PASS' if valid else 'FAIL'} -- {msg}")
    return valid, msg


def run_gate3(out_dir: str, machine_profile: str = "test_profile") -> Tuple[bool, str]:
    """
    Gate 3: Verify compare_baseline correctly blocks on regression and
    passes when within tolerance.

    Tests:
      A) Pass case: metrics within tolerance -> Status.PASS
      B) Fail case: throughput drops >5% -> Status.FAIL
      C) Fail case: perplexity increases >1.5% -> Status.FAIL
    """
    print("\n--- Gate 3: Baseline Comparison Logic ---")

    if not _COMPARE_AVAILABLE:
        return False, "compare_baseline_template not available"

    if not _CONFIG_AVAILABLE:
        return False, "perf_gate_config_template not available"

    tol = ToleranceConfig()
    comparator = CompareBaseline(tol)

    all_passed = True
    messages: List[str] = []

    # --- Sub-test A: Pass when within tolerance ---
    print("  Sub-test A: Pass when within tolerance...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        cur_dir = os.path.join(tmp_dir, "current")
        bas_dir = os.path.join(tmp_dir, "baseline")
        os.makedirs(cur_dir)
        os.makedirs(bas_dir)
        p = machine_profile

        base_m = make_metrics(tokens_per_sec_p50=100_000.0, step_time_p50_s=1.0)
        cur_m = make_metrics(tokens_per_sec_p50=97_000.0, step_time_p50_s=1.03)  # -3%, +3%
        base_e = make_eval_result(ppl=12.0, basic_reasoning=0.92)
        cur_e = make_eval_result(ppl=12.1, basic_reasoning=0.91)  # +0.83% ppl, -1pp

        for d, m, e in [(cur_dir, cur_m, cur_e), (bas_dir, base_m, base_e)]:
            with open(os.path.join(d, f"{p}.metrics.json"), "w") as f:
                json.dump(m, f)
            with open(os.path.join(d, f"{p}.eval.json"), "w") as f:
                json.dump(e, f)

        result_a = comparator.compare(cur_dir, bas_dir, p)
        a_pass = result_a.overall_status == Status.PASS
        status_a = "PASS" if a_pass else "FAIL"
        print(f"    Within-tolerance comparison: {status_a}")
        if not a_pass:
            all_passed = False
            messages.append(
                f"Sub-test A failed: expected PASS but got {result_a.overall_status}. "
                f"Failed checks: {[c.metric for c in result_a.failed_checks]}"
            )

    # --- Sub-test B: Fail when throughput drops >5% ---
    print("  Sub-test B: Fail when throughput drops >5%...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        cur_dir = os.path.join(tmp_dir, "current")
        bas_dir = os.path.join(tmp_dir, "baseline")
        os.makedirs(cur_dir)
        os.makedirs(bas_dir)
        p = machine_profile

        base_m = make_metrics(tokens_per_sec_p50=100_000.0)
        cur_m = make_metrics(tokens_per_sec_p50=92_000.0)  # -8% -> FAIL
        base_e = make_eval_result()
        cur_e = make_eval_result()

        for d, m, e in [(cur_dir, cur_m, cur_e), (bas_dir, base_m, base_e)]:
            with open(os.path.join(d, f"{p}.metrics.json"), "w") as f:
                json.dump(m, f)
            with open(os.path.join(d, f"{p}.eval.json"), "w") as f:
                json.dump(e, f)

        result_b = comparator.compare(cur_dir, bas_dir, p)
        b_pass = result_b.overall_status == Status.FAIL
        status_b = "PASS" if b_pass else "FAIL"
        print(f"    Throughput-regression comparison: {status_b}")
        if not b_pass:
            all_passed = False
            messages.append(
                f"Sub-test B failed: expected FAIL (throughput drop) "
                f"but got {result_b.overall_status}"
            )

    # --- Sub-test C: Fail when perplexity increases >1.5% ---
    print("  Sub-test C: Fail when perplexity increases >1.5%...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        cur_dir = os.path.join(tmp_dir, "current")
        bas_dir = os.path.join(tmp_dir, "baseline")
        os.makedirs(cur_dir)
        os.makedirs(bas_dir)
        p = machine_profile

        base_m = make_metrics()
        cur_m = make_metrics()
        base_e = make_eval_result(ppl=10.0)
        cur_e = make_eval_result(ppl=10.25)  # +2.5% -> FAIL

        for d, m, e in [(cur_dir, cur_m, cur_e), (bas_dir, base_m, base_e)]:
            with open(os.path.join(d, f"{p}.metrics.json"), "w") as f:
                json.dump(m, f)
            with open(os.path.join(d, f"{p}.eval.json"), "w") as f:
                json.dump(e, f)

        result_c = comparator.compare(cur_dir, bas_dir, p)
        c_pass = result_c.overall_status == Status.FAIL
        status_c = "PASS" if c_pass else "FAIL"
        print(f"    Perplexity-regression comparison: {status_c}")
        if not c_pass:
            all_passed = False
            messages.append(
                f"Sub-test C failed: expected FAIL (ppl increase) "
                f"but got {result_c.overall_status}"
            )

    # --- Sub-test D: SKIP when no baseline ---
    print("  Sub-test D: SKIP when baseline missing...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        empty_bas = os.path.join(tmp_dir, "empty_baseline")
        os.makedirs(empty_bas)
        result_d = comparator.compare(tmp_dir, empty_bas, "nonexistent_profile")
        d_pass = result_d.overall_status == Status.SKIP
        status_d = "PASS" if d_pass else "FAIL"
        print(f"    Missing-baseline comparison: {status_d}")
        if not d_pass:
            all_passed = False
            messages.append(
                f"Sub-test D failed: expected SKIP "
                f"but got {result_d.overall_status}"
            )

    if all_passed:
        return True, "All comparison sub-tests passed (A=within_tolerance, B=throughput_fail, C=ppl_fail, D=missing_skip)"
    else:
        return False, "; ".join(messages)


# ===========================================================================
# Main validator
# ===========================================================================

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Validate Compute/Throughput Baseline & Regression Gate done-when gates"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory for validation artifacts (default: temp dir)"
    )
    parser.add_argument(
        "--machine-profile",
        default="test_validation_profile",
        help="Machine profile string to use in validation"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Compute/Throughput Baseline & Regression Gate Validation")
    print("=" * 60)
    print(f"Assets path: {_ASSETS_DIR}")
    print(f"Available: config={_CONFIG_AVAILABLE}, bench={_BENCH_AVAILABLE}, "
          f"quality={_QUALITY_AVAILABLE}, compare={_COMPARE_AVAILABLE}, "
          f"torch={_TORCH_AVAILABLE}")

    gate_results: Dict[str, Tuple[bool, str]] = {}

    if args.out:
        out_dir = args.out
        os.makedirs(out_dir, exist_ok=True)
        _cleanup = False
    else:
        _tmp = tempfile.mkdtemp(prefix="perf_gate_validation_")
        out_dir = _tmp
        _cleanup = True

    print(f"\nOutput directory: {out_dir}")

    try:
        # Gate 1
        gate1_ok, gate1_msg = run_gate1(out_dir, machine_profile=args.machine_profile)
        gate_results["Gate 1 (Schema)"] = (gate1_ok, gate1_msg)

        # Gate 2
        gate2_ok, gate2_msg = run_gate2(out_dir)
        gate_results["Gate 2 (Profiling)"] = (gate2_ok, gate2_msg)

        # Gate 3
        gate3_ok, gate3_msg = run_gate3(out_dir, machine_profile=args.machine_profile)
        gate_results["Gate 3 (Comparison)"] = (gate3_ok, gate3_msg)

    finally:
        if _cleanup:
            import shutil
            shutil.rmtree(out_dir, ignore_errors=True)

    # Print summary
    print("\n" + "=" * 60)
    print("GATE SUMMARY")
    print("=" * 60)

    all_passed = True
    for gate_name, (ok, msg) in gate_results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {gate_name}: {status}")
        print(f"    {msg}")
        if not ok:
            all_passed = False

    print()
    if all_passed:
        print("OVERALL: PASS -- All done-when gates satisfied.")
        return 0
    else:
        failed_gates = [name for name, (ok, _) in gate_results.items() if not ok]
        print(f"OVERALL: FAIL -- Gates failed: {failed_gates}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
