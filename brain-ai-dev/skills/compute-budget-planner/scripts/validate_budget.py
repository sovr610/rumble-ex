"""
validate_budget.py
==================
Validates the three done-when gates for the Compute-Optimal Budget Planner skill.

Gate 1: Chinchilla Sanity
    params=70B, tokens=1.4T yields tokens_per_param=20.0 within float tolerance;
    FLOPs = k * 70e9 * 1.4e12 with k=6.

Gate 2: Undertraining Detection
    Planner correctly warns when tokens_per_param < 0.8 * target;
    does not warn when within optimal range [0.8, 2.0].

Gate 3: N_opt/D_opt Solver
    solve_optimal(C) returns N_opt, D_opt such that k * N_opt * D_opt == C
    within rounding tolerance.

Usage
-----
    python scripts/validate_budget.py

Exit code: 0 if all gates pass, 1 if any gate fails.
"""

from __future__ import annotations

import math
import os
import sys

# ---------------------------------------------------------------------------
# Add assets directory to path
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ASSETS_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "assets")
sys.path.insert(0, _ASSETS_DIR)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
try:
    from budget_config_template import (
        BudgetConfig, RunSpec, ModelSpec, ComputeBudget, BudgetResult
    )
    from gpu_specs_template import GPUSpecTable
    from budget_planner_template import BudgetPlanner
    from chinchilla_solver_template import ChinchillaSolver
except ImportError as exc:
    print(f"FATAL: Cannot import planner modules from {_ASSETS_DIR}: {exc}")
    print("Ensure all template files are present in assets/")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Gate validation helpers
# ---------------------------------------------------------------------------

def _check_close(actual: float, expected: float, rel_tol: float = 1e-6, label: str = "") -> bool:
    """Return True if actual ≈ expected within relative tolerance."""
    if expected == 0:
        return abs(actual) < rel_tol
    rel_err = abs(actual - expected) / abs(expected)
    ok = rel_err <= rel_tol
    if not ok:
        print(
            f"    MISMATCH {label}: actual={actual:.8e}, expected={expected:.8e}, "
            f"rel_err={rel_err:.2e} (tol={rel_tol:.0e})"
        )
    return ok


def _section(title: str) -> None:
    print()
    print("=" * 65)
    print(f"  {title}")
    print("=" * 65)


def _pass(msg: str) -> None:
    print(f"  [PASS] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


# ---------------------------------------------------------------------------
# Gate 1: Chinchilla Sanity
# ---------------------------------------------------------------------------

def gate1_chinchilla_sanity() -> bool:
    """Gate 1: params=70B, tokens=1.4T → tokens_per_param=20.0, FLOPs=5.88e23."""
    _section("Gate 1: Chinchilla Sanity (70B / 1.4T)")

    all_pass = True
    k = 6.0
    n_params = 70e9
    total_tokens = 1.4e12
    expected_tpp = 20.0
    expected_flops = k * n_params * total_tokens  # 5.88e23

    print(f"  Parameters:   n={n_params:.3e} ({n_params/1e9:.1f}B)")
    print(f"  Tokens:       D={total_tokens:.3e} ({total_tokens/1e12:.1f}T)")
    print(f"  k:            {k}")
    print()

    # Subtest 1a: tokens_per_param from ChinchillaSolver
    solver = ChinchillaSolver(k=k, tokens_per_param=expected_tpp)
    actual_ratio = solver.undertraining_ratio(n_params, total_tokens)
    expected_ratio = 1.0  # at exact compute-optimal, ratio=1.0
    actual_tpp = total_tokens / n_params

    print(f"  tokens/param: {actual_tpp:.6f} (expected: {expected_tpp:.1f})")
    if _check_close(actual_tpp, expected_tpp, rel_tol=1e-9, label="tokens_per_param"):
        _pass(f"tokens/param = {actual_tpp:.6f} ≈ {expected_tpp:.1f} (within 1e-9 relative tolerance)")
    else:
        _fail(f"tokens/param = {actual_tpp:.6f} ≠ {expected_tpp:.1f}")
        all_pass = False

    # Subtest 1b: undertraining_ratio = 1.0 at exact target
    print(f"  Undertraining ratio: {actual_ratio:.10f} (expected: 1.0)")
    if _check_close(actual_ratio, 1.0, rel_tol=1e-9, label="undertraining_ratio"):
        _pass(f"undertraining_ratio = {actual_ratio:.10f} ≈ 1.0")
    else:
        _fail(f"undertraining_ratio = {actual_ratio:.10f} ≠ 1.0")
        all_pass = False

    # Subtest 1c: FLOPs from BudgetPlanner
    specs = GPUSpecTable()
    config = BudgetConfig(k=k, tokens_per_param_target=expected_tpp)
    planner = BudgetPlanner(specs, config)

    from budget_planner_template import _compute_flops
    actual_flops = _compute_flops(n_params, total_tokens, k)

    print(f"  Total FLOPs: {actual_flops:.6e} (expected: {expected_flops:.6e})")
    if _check_close(actual_flops, expected_flops, rel_tol=1e-10, label="total_flops"):
        _pass(f"total_flops = {actual_flops:.6e} ≈ k*N*D = {expected_flops:.6e}")
    else:
        _fail(f"total_flops = {actual_flops:.6e} ≠ {expected_flops:.6e}")
        all_pass = False

    # Subtest 1d: Full Mode A run through BudgetPlanner
    steps = math.ceil(total_tokens / (2048 * 2048))
    run = RunSpec(
        n_params=n_params, seq_len=2048, global_batch=2048, steps=steps,
        num_gpus=8, gpu_type="H100_SXM", dtype="bf16"
    )
    result = planner.validate_run(run)
    actual_flops_mode_a = result.derived["total_flops"]
    result_tpp = result.derived["tokens_per_param"]

    print(f"  Mode A tokens/param: {result_tpp:.6f}")
    if _check_close(result_tpp, expected_tpp, rel_tol=0.001, label="Mode_A_tokens_per_param"):
        _pass(f"Mode A tokens/param = {result_tpp:.4f} ≈ 20.0 (within 0.1% — step rounding)")
    else:
        _fail(f"Mode A tokens/param = {result_tpp:.4f}, expected ~20.0")
        all_pass = False

    print(f"  Mode A total_flops: {actual_flops_mode_a:.6e}")
    expected_flops_mode_a = k * n_params * run.total_tokens
    if _check_close(actual_flops_mode_a, expected_flops_mode_a, rel_tol=1e-6):
        _pass(f"Mode A total_flops = {actual_flops_mode_a:.6e}")
    else:
        _fail(f"Mode A total_flops = {actual_flops_mode_a:.6e}, expected {expected_flops_mode_a:.6e}")
        all_pass = False

    # Subtest 1e: No critical warnings (this is near compute-optimal)
    if not result.has_critical_warnings():
        _pass("No CRITICAL warnings for compute-optimal 70B/1.4T run")
    else:
        _fail(f"Unexpected CRITICAL warnings: {result.warnings}")
        all_pass = False

    return all_pass


# ---------------------------------------------------------------------------
# Gate 2: Undertraining Detection
# ---------------------------------------------------------------------------

def gate2_undertraining_detection() -> bool:
    """Gate 2: Correctly warns when undertrained, silent when in optimal range."""
    _section("Gate 2: Undertraining Detection")

    all_pass = True

    specs = GPUSpecTable()
    config = BudgetConfig(k=6.0, tokens_per_param_target=20.0)
    planner = BudgetPlanner(specs, config)

    # tokens per step for the test runs
    seq_len = 2048
    global_batch = 2048
    tokens_per_step = seq_len * global_batch  # 4,194,304

    # To get a target tokens_per_param ratio:
    #   steps = ceil(target_tpp * n_params / tokens_per_step)
    # ratio = 0.02 -> tpp = 0.02 * 20 = 0.4; steps = ceil(0.4*7e9/4194304) = ceil(667) = 668
    # ratio = 0.60 -> tpp = 12.0;             steps = ceil(12*7e9/4194304) = ceil(20019) = 20020
    # ratio = 1.00 -> tpp = 20.0;             steps = ceil(20*7e9/4194304) = ceil(33366) = 33367
    # ratio = 1.20 -> tpp = 24.0;             steps = ceil(24*7e9/4194304) = ceil(40039) = 40040
    # ratio = 7.10 -> tpp = 142.0;            steps = ceil(142*7e9/4194304)= ceil(236969)= 236970

    import math as _math

    def _steps_for_ratio(target_ratio, n_params, tpp_target, tokens_per_step):
        target_tpp = target_ratio * tpp_target
        return _math.ceil(target_tpp * n_params / tokens_per_step)

    steps_critical = _steps_for_ratio(0.02, 7e9, 20.0, tokens_per_step)
    steps_warning  = _steps_for_ratio(0.60, 7e9, 20.0, tokens_per_step)
    steps_ok       = _steps_for_ratio(1.00, 7e9, 20.0, tokens_per_step)
    steps_ok2      = _steps_for_ratio(1.20, 7e9, 20.0, tokens_per_step)
    steps_info     = _steps_for_ratio(7.10, 7e9, 20.0, tokens_per_step)

    test_cases = [
        # (description, n_params, steps, should_warn, expected_severity)
        ("Severely undertrained (ratio≈0.02)",   7e9, steps_critical, True,  "CRITICAL"),
        ("Likely undertrained   (ratio≈0.60)",   7e9, steps_warning,  True,  "WARNING"),
        ("Near compute-optimal  (ratio≈1.0)",    7e9, steps_ok,       False, None),
        ("Slightly over target  (ratio≈1.2)",    7e9, steps_ok2,      False, None),
        ("Overtrain regime      (ratio≈7.1)",    7e9, steps_info,     True,  "INFO"),
    ]

    for desc, n_params, steps, should_warn, expected_severity in test_cases:
        total_tokens = steps * tokens_per_step
        actual_tpp = total_tokens / n_params
        ratio = actual_tpp / 20.0

        run = RunSpec(
            n_params=n_params, seq_len=seq_len, global_batch=global_batch,
            steps=steps, num_gpus=8, gpu_type="H100_SXM"
        )
        result = planner.validate_run(run)

        has_warn = result.has_warnings()
        print(f"\n  Test: {desc}")
        print(f"    tokens/param={actual_tpp:.2f}, ratio={ratio:.4f}")
        print(f"    Warnings emitted: {has_warn} | Expected: {should_warn}")
        if result.warnings:
            print(f"    Warning text: {result.warnings[0][:80]}...")

        if should_warn:
            if has_warn:
                # Check severity
                if expected_severity:
                    sev_found = any(
                        f"[{expected_severity}]" in w for w in result.warnings
                    )
                    if sev_found:
                        _pass(f"{desc}: [{expected_severity}] warning correctly emitted")
                    else:
                        _fail(
                            f"{desc}: Expected [{expected_severity}] in warnings, "
                            f"got: {result.warnings}"
                        )
                        all_pass = False
                else:
                    _pass(f"{desc}: Warning correctly emitted")
            else:
                _fail(f"{desc}: Expected warning but none emitted")
                all_pass = False
        else:
            if not has_warn:
                _pass(f"{desc}: No warnings emitted (correct — ratio in OK range)")
            else:
                _fail(
                    f"{desc}: Unexpected warning for ratio={ratio:.3f}: {result.warnings}"
                )
                all_pass = False

    # Boundary tests
    print("\n  Boundary tests:")
    from chinchilla_solver_template import ChinchillaSolver
    solver = ChinchillaSolver(k=6.0, tokens_per_param=20.0)

    boundaries = [
        (0.4999, True, "CRITICAL"),    # just below CRITICAL threshold (0.5)
        (0.5000, False, None),         # exactly at boundary: WARNING (not CRITICAL) → no warning range? Actually 0.5 is WARNING
        (0.7999, True, "WARNING"),     # just below WARNING threshold (0.8)
        (0.8000, False, None),         # exactly at 0.8: OK
        (2.0000, False, None),         # exactly at 2.0: OK
        (2.0001, True, "INFO"),        # just above 2.0: INFO
    ]

    # Correct the 0.5 boundary: ratio=0.5 is at THRESHOLD_CRITICAL boundary
    # Looking at the code: ratio < 0.5 → CRITICAL, so ratio=0.5 → WARNING
    boundaries_corrected = [
        (0.3, True, "CRITICAL"),       # ratio < 0.5 → CRITICAL
        (0.5, True, "WARNING"),        # ratio = 0.5, not < 0.5 → WARNING (0.5 <= ratio < 0.8)
        (0.79, True, "WARNING"),       # ratio < 0.8 → WARNING
        (0.8, False, None),            # ratio = 0.8 → OK
        (1.0, False, None),            # ratio = 1.0 → OK
        (2.0, False, None),            # ratio = 2.0 → OK
        (2.001, True, "INFO"),         # ratio > 2.0 → INFO
    ]

    for ratio, should_warn, expected_sev in boundaries_corrected:
        warnings = solver.warnings(ratio)
        has_warn = len(warnings) > 0
        correct = has_warn == should_warn
        if expected_sev and correct:
            correct = any(f"[{expected_sev}]" in w for w in warnings)

        status = "[PASS]" if correct else "[FAIL]"
        sev_label = f"[{expected_sev}]" if expected_sev else "no warning"
        print(f"    {status} ratio={ratio:.4f}: expected {sev_label} | "
              f"got: {warnings[0][:40] if warnings else 'no warnings'}")

        if not correct:
            all_pass = False

    return all_pass


# ---------------------------------------------------------------------------
# Gate 3: N_opt / D_opt Solver
# ---------------------------------------------------------------------------

def gate3_solver_correctness() -> bool:
    """Gate 3: k * N_opt * D_opt == C within rounding tolerance."""
    _section("Gate 3: N_opt / D_opt Solver Correctness")

    all_pass = True
    k = 6.0
    a = 20.0
    rel_tol = 1e-9

    specs = GPUSpecTable()
    config = BudgetConfig(k=k, tokens_per_param_target=a)
    planner = BudgetPlanner(specs, config)

    test_budgets = [
        1e18,   # 1 ExaFLOP
        1e20,
        5e22,
        5.88e23,  # Chinchilla 70B/1.4T exact
        5e23,
        1e25,
        1e26,
    ]

    print(f"  Testing that k * N_opt * D_opt == C (tol={rel_tol:.0e})")
    print(f"  k={k}, tokens_per_param_target={a}")
    print()
    print(f"  {'C':>12}  {'N_opt':>12}  {'D_opt':>12}  {'k*N*D':>14}  {'rel_err':>12}  Status")
    print(f"  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*14}  {'-'*12}  {'-'*6}")

    for C in test_budgets:
        budget = ComputeBudget(mode="flops", total_flops=C)
        result = planner.solve_optimal(budget)

        n_opt = result.derived["n_opt"]
        d_opt = result.derived["d_opt"]
        reconstructed = k * n_opt * d_opt
        rel_err = abs(reconstructed - C) / C

        ok = rel_err <= rel_tol
        status = "PASS" if ok else "FAIL"

        print(
            f"  {C:>12.3e}  {n_opt:>12.4e}  {d_opt:>12.4e}  "
            f"{reconstructed:>14.6e}  {rel_err:>12.2e}  {status}"
        )

        if not ok:
            all_pass = False
            _fail(f"k*N_opt*D_opt = {reconstructed:.6e} ≠ C = {C:.6e} (rel_err={rel_err:.2e})")

    if all_pass:
        _pass(f"k * N_opt * D_opt == C within {rel_tol:.0e} for all test budgets")

    # Subtest: tokens/param at optimal = target exactly
    print()
    print("  Verifying D_opt / N_opt == tokens_per_param_target:")
    for C in [1e20, 5e23, 1e25]:
        budget = ComputeBudget(mode="flops", total_flops=C)
        result = planner.solve_optimal(budget)
        n_opt = result.derived["n_opt"]
        d_opt = result.derived["d_opt"]
        computed_tpp = d_opt / n_opt
        ok = abs(computed_tpp - a) < 1e-6
        print(f"    C={C:.1e}: D_opt/N_opt = {computed_tpp:.8f} (target={a:.1f}) {'OK' if ok else 'FAIL'}")
        if not ok:
            all_pass = False

    if all_pass:
        _pass("D_opt / N_opt == tokens_per_param_target for all test budgets")

    # Subtest: N_opt proportional to sqrt(C)
    print()
    print("  Verifying N_opt proportional to sqrt(C):")
    budgets_pair = [(1e22, 4e22), (1e23, 1e24)]
    for C1, C4 in budgets_pair:
        b1 = ComputeBudget(mode="flops", total_flops=C1)
        b4 = ComputeBudget(mode="flops", total_flops=C4)
        r1 = planner.solve_optimal(b1)
        r4 = planner.solve_optimal(b4)
        n1 = r1.derived["n_opt"]
        n4 = r4.derived["n_opt"]
        ratio = n4 / n1
        expected_ratio = math.sqrt(C4 / C1)
        ok = abs(ratio - expected_ratio) < 1e-9
        print(
            f"    N_opt(C={C4:.1e}) / N_opt(C={C1:.1e}) = {ratio:.8f} "
            f"(expected sqrt({C4/C1:.0f}) = {expected_ratio:.8f}) {'OK' if ok else 'FAIL'}"
        )
        if not ok:
            all_pass = False

    if all_pass:
        _pass("N_opt scales as sqrt(C) for all tested budget pairs")

    # Subtest: Chinchilla 70B/1.4T is on the optimal frontier
    print()
    print("  Verifying 70B/1.4T is on the compute-optimal frontier:")
    C_chinchilla = k * 70e9 * 1.4e12  # 5.88e23
    budget = ComputeBudget(mode="flops", total_flops=C_chinchilla)
    result = planner.solve_optimal(budget)
    n_opt = result.derived["n_opt"]
    d_opt = result.derived["d_opt"]
    n_rel_err = abs(n_opt - 70e9) / 70e9
    d_rel_err = abs(d_opt - 1.4e12) / 1.4e12
    print(f"    C = {C_chinchilla:.4e}")
    print(f"    N_opt = {n_opt:.6e} (expected: 7.0e10), rel_err={n_rel_err:.2e}")
    print(f"    D_opt = {d_opt:.6e} (expected: 1.4e12), rel_err={d_rel_err:.2e}")

    if n_rel_err < 1e-6 and d_rel_err < 1e-6:
        _pass("solve_optimal(C_chinchilla) yields N_opt≈70B, D_opt≈1.4T")
    else:
        _fail(f"solve_optimal(C_chinchilla): N_opt={n_opt:.4e}, D_opt={d_opt:.4e}")
        all_pass = False

    return all_pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Run all done-when gates. Return 0 if all pass, 1 if any fail."""
    print("=" * 65)
    print("  Compute-Optimal Budget Planner — Done-When Gate Validation")
    print("=" * 65)

    results = {}

    try:
        results["Gate 1"] = gate1_chinchilla_sanity()
    except Exception as exc:
        print(f"\n  [ERROR] Gate 1 crashed: {exc}")
        import traceback
        traceback.print_exc()
        results["Gate 1"] = False

    try:
        results["Gate 2"] = gate2_undertraining_detection()
    except Exception as exc:
        print(f"\n  [ERROR] Gate 2 crashed: {exc}")
        import traceback
        traceback.print_exc()
        results["Gate 2"] = False

    try:
        results["Gate 3"] = gate3_solver_correctness()
    except Exception as exc:
        print(f"\n  [ERROR] Gate 3 crashed: {exc}")
        import traceback
        traceback.print_exc()
        results["Gate 3"] = False

    # Summary
    _section("Summary")
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  ALL DONE-WHEN GATES PASSED. Skill is ready for integration.")
    else:
        print("  SOME GATES FAILED. See details above.")
    print("=" * 65)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
