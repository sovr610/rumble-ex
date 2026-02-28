"""
chinchilla_solver_template.py
==============================
ChinchillaSolver: compute-optimal frontier calculations.

Implements the Chinchilla-style scaling law solver:
    N_opt = sqrt(C / (k * a))
    D_opt = a * N_opt

Where:
    C = total training FLOPs
    k = FLOPs coefficient (default 6, from C = k*N*D)
    a = tokens_per_param target (default 20, from Chinchilla 70B/1.4T)

Provides:
    - N_opt, D_opt from a compute budget C
    - Undertraining ratio: planned tokens/param vs target
    - Severity-based warnings (CRITICAL / WARNING / INFO)
    - Target tokens given model size
    - Required FLOPs given (N, D)

Usage
-----
    from chinchilla_solver_template import ChinchillaSolver

    solver = ChinchillaSolver(k=6.0, tokens_per_param=20.0)
    n = solver.n_opt(5e23)   # -> ~64.6B
    d = solver.d_opt(5e23)   # -> ~1.29T
    ratio = solver.undertraining_ratio(7e9, 1e11)  # -> 0.71 (undertrained)
    warnings = solver.warnings(ratio)
"""

from __future__ import annotations

import math
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Warning severity thresholds
# ---------------------------------------------------------------------------

# undertraining_ratio thresholds:
#   ratio = planned_tokens_per_param / tokens_per_param_target
THRESHOLD_CRITICAL = 0.5   # ratio < 0.5  → CRITICAL (severely undertrained)
THRESHOLD_WARNING  = 0.8   # ratio < 0.8  → WARNING  (likely undertrained)
THRESHOLD_OK_HIGH  = 2.0   # ratio > 2.0  → INFO     (overtrain regime)
# 0.8 <= ratio <= 2.0 → OK, no warning


class ChinchillaSolver:
    """Chinchilla compute-optimal frontier solver and undertraining detector.

    Parameters
    ----------
    k : float
        FLOPs coefficient in C = k * N * D. Default 6.0.
    tokens_per_param : float
        Target tokens per non-embedding parameter. Default 20.0 (Chinchilla).

    Notes
    -----
    The N_opt/D_opt formulas assume:
        C = k * N * D,   D = a * N   (where a = tokens_per_param)
    Substituting: C = k * a * N^2
        N_opt = sqrt(C / (k * a))
        D_opt = a * N_opt = sqrt(C * a / k)

    Validation: k * N_opt * D_opt == C (to floating-point precision).
    """

    def __init__(self, k: float = 6.0, tokens_per_param: float = 20.0) -> None:
        if not math.isfinite(k) or k <= 0:
            raise ValueError(f"k must be a finite positive number, got {k!r}")
        if not math.isfinite(tokens_per_param) or tokens_per_param <= 0:
            raise ValueError(f"tokens_per_param must be a finite positive number, got {tokens_per_param!r}")
        self.k = k
        self.tokens_per_param = tokens_per_param
        logger.debug(
            "ChinchillaSolver initialized: k=%.1f, tokens_per_param=%.1f",
            self.k, self.tokens_per_param
        )

    # ------------------------------------------------------------------
    # Core solver methods
    # ------------------------------------------------------------------

    def n_opt(self, compute_flops: float) -> float:
        """Compute optimal parameter count for a given compute budget.

        Parameters
        ----------
        compute_flops : float
            Total training FLOPs budget C.

        Returns
        -------
        float
            Compute-optimal number of parameters N_opt = sqrt(C / (k * a)).

        Raises
        ------
        ValueError
            If compute_flops is not a finite positive number.
        """
        if not math.isfinite(compute_flops) or compute_flops <= 0:
            raise ValueError(f"compute_flops must be a finite positive number, got {compute_flops!r}")

        n = math.sqrt(compute_flops / (self.k * self.tokens_per_param))
        logger.debug("n_opt(C=%.3e) = %.3e  [k=%.1f, a=%.1f]",
                     compute_flops, n, self.k, self.tokens_per_param)
        return n

    def d_opt(self, compute_flops: float) -> float:
        """Compute optimal token count for a given compute budget.

        Parameters
        ----------
        compute_flops : float
            Total training FLOPs budget C.

        Returns
        -------
        float
            Compute-optimal token count D_opt = a * N_opt.
        """
        n = self.n_opt(compute_flops)
        d = self.tokens_per_param * n
        logger.debug("d_opt(C=%.3e) = %.3e", compute_flops, d)
        return d

    def n_d_opt(self, compute_flops: float) -> Tuple[float, float]:
        """Return (N_opt, D_opt) pair for a given compute budget.

        Returns
        -------
        tuple of (float, float)
            (N_opt, D_opt)
        """
        n = self.n_opt(compute_flops)
        d = self.tokens_per_param * n
        return n, d

    # ------------------------------------------------------------------
    # Undertraining analysis
    # ------------------------------------------------------------------

    def undertraining_ratio(self, n_params: float, total_tokens: float) -> float:
        """Compute the undertraining ratio: planned tokens/param vs target.

        Parameters
        ----------
        n_params : float
            Non-embedding parameter count of the model.
        total_tokens : float
            Total planned training tokens.

        Returns
        -------
        float
            ratio = (total_tokens / n_params) / tokens_per_param_target
            ratio = 1.0 means exactly at the Chinchilla compute-optimal point.
            ratio < 1.0 means undertrained.
            ratio > 1.0 means overtrained (LLaMA-style).

        Raises
        ------
        ValueError
            If n_params <= 0 or total_tokens < 0.
        """
        if not math.isfinite(n_params) or n_params <= 0:
            raise ValueError(f"n_params must be a finite positive number, got {n_params!r}")
        if not math.isfinite(total_tokens) or total_tokens < 0:
            raise ValueError(f"total_tokens must be a finite non-negative number, got {total_tokens!r}")

        planned_ratio = total_tokens / n_params
        ratio = planned_ratio / self.tokens_per_param
        logger.debug(
            "undertraining_ratio: tokens/param=%.2f, target=%.1f, ratio=%.4f",
            planned_ratio, self.tokens_per_param, ratio
        )
        return ratio

    def tokens_per_param_actual(self, n_params: float, total_tokens: float) -> float:
        """Return actual tokens/param for a given run.

        Parameters
        ----------
        n_params : float
        total_tokens : float

        Returns
        -------
        float
            total_tokens / n_params
        """
        if not math.isfinite(n_params) or n_params <= 0:
            raise ValueError(f"n_params must be a finite positive number, got {n_params!r}")
        return total_tokens / n_params

    # ------------------------------------------------------------------
    # Warning and suggestion generation
    # ------------------------------------------------------------------

    def warnings(self, ratio: float) -> List[str]:
        """Generate severity-based warning messages for an undertraining ratio.

        Parameters
        ----------
        ratio : float
            Output of undertraining_ratio().

        Returns
        -------
        list of str
            Zero or more warning strings with [SEVERITY] prefix.
            Empty list if ratio is in the OK range [0.8, 2.0].
        """
        msgs: List[str] = []

        if ratio < THRESHOLD_CRITICAL:
            msgs.append(
                f"[CRITICAL] Severely undertrained: tokens/param ratio is {ratio:.3f}x the target "
                f"(target={self.tokens_per_param:.1f}). The model has seen far too few tokens. "
                f"Either increase training tokens significantly or reduce model size."
            )
        elif ratio < THRESHOLD_WARNING:
            msgs.append(
                f"[WARNING] Likely undertrained: tokens/param ratio is {ratio:.3f}x the target "
                f"(target={self.tokens_per_param:.1f}). Consider training for more tokens "
                f"to approach the Chinchilla compute-optimal point."
            )
        elif ratio > THRESHOLD_OK_HIGH:
            msgs.append(
                f"[INFO] Overtrain regime: tokens/param ratio is {ratio:.3f}x the target "
                f"(target={self.tokens_per_param:.1f}). This is likely intentional (LLaMA-style "
                f"inference-optimal training). Confirm this is deliberate."
            )
        # else: 0.8 <= ratio <= 2.0 → OK, no warning

        return msgs

    def suggestions(
        self,
        n_params: float,
        total_tokens: float,
        ratio: float,
        num_gpus: Optional[int] = None,
        peak_tflops: Optional[float] = None,
        utilization: Optional[float] = None,
        seq_len: Optional[int] = None,
        global_batch: Optional[int] = None,
    ) -> List[str]:
        """Generate actionable suggestions for correcting undertraining.

        Parameters
        ----------
        n_params : float
        total_tokens : float
        ratio : float
            Undertraining ratio from undertraining_ratio().
        num_gpus : int, optional
            Number of GPUs (for wallclock suggestion).
        peak_tflops : float, optional
            Peak GPU TFLOPS (for wallclock suggestion).
        utilization : float, optional
            MFU (for wallclock suggestion).
        seq_len : int, optional
        global_batch : int, optional

        Returns
        -------
        list of str
        """
        msgs: List[str] = []
        target_tokens = self.target_tokens(n_params)

        if ratio < THRESHOLD_WARNING:
            token_gap = target_tokens - total_tokens
            msgs.append(
                f"To reach {self.tokens_per_param:.1f} tokens/param for this "
                f"{n_params/1e9:.2f}B model, increase to "
                f"{target_tokens:.3e} tokens (add {token_gap:.3e} more tokens)."
            )
            if seq_len and global_batch:
                tokens_per_step = seq_len * global_batch
                extra_steps = math.ceil(token_gap / tokens_per_step)
                msgs.append(
                    f"At seq_len={seq_len}, global_batch={global_batch}: "
                    f"add {extra_steps:,} optimizer steps."
                )
            if num_gpus and peak_tflops and utilization:
                extra_flops = self.k * n_params * token_gap
                extra_s = extra_flops / (num_gpus * peak_tflops * 1e12 * utilization)
                extra_h = extra_s / 3600
                msgs.append(
                    f"Estimated additional wallclock at {utilization*100:.0f}% MFU "
                    f"on {num_gpus}x GPUs: +{extra_h:.1f} hours."
                )

        elif ratio > THRESHOLD_OK_HIGH:
            compute_opt_n, compute_opt_d = self.n_d_opt(
                self.required_flops(n_params, total_tokens)
            )
            msgs.append(
                f"tokens/param={total_tokens/n_params:.1f} >> target={self.tokens_per_param:.1f}. "
                f"If inference cost is the concern, this LLaMA-style overtrain is justified. "
                f"Compute-optimal for this FLOP budget: N_opt={compute_opt_n/1e9:.2f}B, "
                f"D_opt={compute_opt_d/1e12:.3f}T tokens."
            )

        return msgs

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def target_tokens(self, n_params: float) -> float:
        """Compute target token count for compute-optimal training.

        Parameters
        ----------
        n_params : float
            Non-embedding parameter count.

        Returns
        -------
        float
            tokens_per_param_target * n_params
        """
        if not math.isfinite(n_params) or n_params <= 0:
            raise ValueError(f"n_params must be a finite positive number, got {n_params!r}")
        return self.tokens_per_param * n_params

    def required_flops(self, n_params: float, total_tokens: float) -> float:
        """Compute total training FLOPs for a given (N, D) pair.

        Parameters
        ----------
        n_params : float
        total_tokens : float

        Returns
        -------
        float
            k * n_params * total_tokens
        """
        if not math.isfinite(n_params) or n_params <= 0:
            raise ValueError(f"n_params must be a finite positive number, got {n_params!r}")
        if not math.isfinite(total_tokens) or total_tokens < 0:
            raise ValueError(f"total_tokens must be a finite non-negative number, got {total_tokens!r}")
        return self.k * n_params * total_tokens

    def frontier_points(
        self, c_values: Optional[List[float]] = None
    ) -> List[Tuple[float, float, float]]:
        """Generate (C, N_opt, D_opt) points on the compute-optimal frontier.

        Parameters
        ----------
        c_values : list of float, optional
            Compute budgets to evaluate. If None, uses a default log-spaced range.

        Returns
        -------
        list of (C, N_opt, D_opt) tuples
        """
        if c_values is None:
            c_values = [10 ** exp for exp in range(18, 27)]

        points = []
        for c in c_values:
            try:
                n, d = self.n_d_opt(c)
                points.append((c, n, d))
            except ValueError:
                continue
        return points

    def summary_str(self, n_params: float, total_tokens: float) -> str:
        """Return a human-readable summary of the Chinchilla analysis.

        Parameters
        ----------
        n_params : float
        total_tokens : float

        Returns
        -------
        str
        """
        actual_tpp = total_tokens / n_params
        ratio = self.undertraining_ratio(n_params, total_tokens)
        total_flops = self.required_flops(n_params, total_tokens)
        n_opt_val, d_opt_val = self.n_d_opt(total_flops)

        status_map = [
            (THRESHOLD_CRITICAL, "CRITICAL — severely undertrained"),
            (THRESHOLD_WARNING, "WARNING — likely undertrained"),
            (THRESHOLD_OK_HIGH, "OK — near compute-optimal"),
            (float("inf"), "INFO — overtrain regime"),
        ]
        status = "INFO — overtrain regime"
        for threshold, label in status_map:
            if ratio < threshold:
                status = label
                break

        lines = [
            f"Chinchilla Analysis",
            f"  Model params:        {n_params/1e9:.3f}B",
            f"  Training tokens:     {total_tokens/1e12:.3f}T",
            f"  tokens/param:        {actual_tpp:.2f} (target: {self.tokens_per_param:.1f})",
            f"  Undertraining ratio: {ratio:.4f}x",
            f"  Status:              {status}",
            f"  Total FLOPs:         {total_flops:.3e}",
            f"  N_opt (this budget): {n_opt_val/1e9:.3f}B",
            f"  D_opt (this budget): {d_opt_val/1e12:.3f}T",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _run_self_tests() -> None:
    print("Running chinchilla_solver_template.py self-tests...")
    errors = []

    solver = ChinchillaSolver(k=6.0, tokens_per_param=20.0)

    # Test 1: N_opt * D_opt * k == C (fundamental correctness)
    try:
        test_budgets = [1e20, 5e22, 5e23, 1e25, 1e26]
        for C in test_budgets:
            n = solver.n_opt(C)
            d = solver.d_opt(C)
            reconstructed = solver.k * n * d
            rel_err = abs(reconstructed - C) / C
            assert rel_err < 1e-9, (
                f"k*N_opt*D_opt != C for C={C:.2e}: "
                f"k*N*D={reconstructed:.6e}, C={C:.6e}, rel_err={rel_err:.2e}"
            )
        print("  [PASS] k * N_opt * D_opt == C within tolerance for all test budgets")
    except Exception as e:
        errors.append(f"  [FAIL] N_opt/D_opt correctness: {e}")

    # Test 2: Chinchilla 70B/1.4T sanity
    try:
        # 6 * 70e9 * 1.4e12 = 5.88e23
        C_chinchilla = 6 * 70e9 * 1.4e12
        n = solver.n_opt(C_chinchilla)
        d = solver.d_opt(C_chinchilla)
        assert abs(n - 70e9) / 70e9 < 1e-6, f"N_opt != 70B: {n:.4e}"
        assert abs(d - 1.4e12) / 1.4e12 < 1e-6, f"D_opt != 1.4T: {d:.4e}"
        print("  [PASS] Chinchilla 70B/1.4T sanity: N_opt ≈ 70B, D_opt ≈ 1.4T")
    except Exception as e:
        errors.append(f"  [FAIL] Chinchilla 70B/1.4T sanity: {e}")

    # Test 3: tokens/param ratio
    try:
        ratio_20 = solver.undertraining_ratio(70e9, 1.4e12)
        assert abs(ratio_20 - 1.0) < 1e-10, f"Expected ratio=1.0, got {ratio_20}"

        ratio_10 = solver.undertraining_ratio(70e9, 0.7e12)  # half tokens
        assert abs(ratio_10 - 0.5) < 1e-10, f"Expected ratio=0.5, got {ratio_10}"

        print("  [PASS] undertraining_ratio correct for 70B/1.4T and 70B/0.7T")
    except Exception as e:
        errors.append(f"  [FAIL] undertraining_ratio: {e}")

    # Test 4: Warning thresholds
    try:
        # CRITICAL: ratio < 0.5
        w = solver.warnings(0.3)
        assert any("[CRITICAL]" in msg for msg in w), f"Expected CRITICAL, got {w}"

        # WARNING: 0.5 <= ratio < 0.8
        w = solver.warnings(0.65)
        assert any("[WARNING]" in msg for msg in w), f"Expected WARNING, got {w}"
        assert not any("[CRITICAL]" in msg for msg in w)

        # OK: 0.8 <= ratio <= 2.0
        for r in [0.8, 1.0, 1.5, 2.0]:
            w = solver.warnings(r)
            assert len(w) == 0, f"Expected no warnings for ratio={r}, got {w}"

        # INFO: ratio > 2.0
        w = solver.warnings(7.1)
        assert any("[INFO]" in msg for msg in w), f"Expected INFO, got {w}"
        assert not any("[CRITICAL]" in msg for msg in w)
        assert not any("[WARNING]" in msg for msg in w)

        print("  [PASS] Warning thresholds CRITICAL/WARNING/OK/INFO correct")
    except Exception as e:
        errors.append(f"  [FAIL] Warning thresholds: {e}")

    # Test 5: N_opt proportional to sqrt(C)
    try:
        C1 = 1e23
        C4 = 4e23
        n1 = solver.n_opt(C1)
        n4 = solver.n_opt(C4)
        ratio_n = n4 / n1
        assert abs(ratio_n - 2.0) < 1e-9, f"N_opt(4C)/N_opt(C) should be 2.0, got {ratio_n}"
        print("  [PASS] N_opt proportional to sqrt(C): N_opt(4C) = 2 * N_opt(C)")
    except Exception as e:
        errors.append(f"  [FAIL] N_opt sqrt(C) proportionality: {e}")

    # Test 6: target_tokens
    try:
        t = solver.target_tokens(7e9)
        assert abs(t - 140e9) < 1, f"Expected target_tokens(7B)=140B, got {t}"
        t70 = solver.target_tokens(70e9)
        assert abs(t70 - 1.4e12) / 1.4e12 < 1e-10
        print("  [PASS] target_tokens: 7B->140B, 70B->1.4T")
    except Exception as e:
        errors.append(f"  [FAIL] target_tokens: {e}")

    # Test 7: required_flops
    try:
        flops = solver.required_flops(70e9, 1.4e12)
        expected = 6 * 70e9 * 1.4e12
        assert abs(flops - expected) / expected < 1e-10, f"Expected {expected:.3e}, got {flops:.3e}"
        print("  [PASS] required_flops(70B, 1.4T) = 5.88e23")
    except Exception as e:
        errors.append(f"  [FAIL] required_flops: {e}")

    # Test 8: Positive values for very large and very small budgets
    try:
        for C in [1e18, 1e22, 1e26, 1e30]:
            n = solver.n_opt(C)
            d = solver.d_opt(C)
            assert n > 0 and math.isfinite(n), f"n_opt not positive/finite for C={C:.1e}"
            assert d > 0 and math.isfinite(d), f"d_opt not positive/finite for C={C:.1e}"
        print("  [PASS] n_opt/d_opt positive and finite for C in [1e18, 1e30]")
    except Exception as e:
        errors.append(f"  [FAIL] Positive values for extreme C: {e}")

    # Test 9: ValueError on bad inputs
    try:
        try:
            solver.n_opt(-1.0)
            errors.append("  [FAIL] n_opt(-1) should raise ValueError")
        except ValueError:
            pass
        try:
            solver.undertraining_ratio(0, 1e12)
            errors.append("  [FAIL] undertraining_ratio(n=0) should raise ValueError")
        except ValueError:
            pass
        try:
            solver.undertraining_ratio(7e9, -1)
            errors.append("  [FAIL] undertraining_ratio(tokens=-1) should raise ValueError")
        except ValueError:
            pass
        print("  [PASS] ValueError raised for invalid inputs")
    except Exception as e:
        errors.append(f"  [FAIL] ValueError checks: {e}")

    # Test 10: Different k values
    try:
        solver_k8 = ChinchillaSolver(k=8.0, tokens_per_param=20.0)
        C = 5e23
        n8 = solver_k8.n_opt(C)
        n6 = solver.n_opt(C)
        # k=8 means more FLOPs per param, so N_opt should be smaller
        assert n8 < n6, f"With higher k, N_opt should be smaller: n8={n8:.3e}, n6={n6:.3e}"
        # Verify k*N*D = C for k=8
        d8 = solver_k8.d_opt(C)
        reconstructed = 8.0 * n8 * d8
        assert abs(reconstructed - C) / C < 1e-9
        print("  [PASS] ChinchillaSolver with k=8.0 produces correct N_opt/D_opt")
    except Exception as e:
        errors.append(f"  [FAIL] Different k values: {e}")

    # Test 11: frontier_points returns monotonically increasing values
    try:
        points = solver.frontier_points()
        ns = [p[1] for p in points]
        ds = [p[2] for p in points]
        for i in range(1, len(ns)):
            assert ns[i] > ns[i-1], f"N_opt not monotonic at index {i}: {ns[i-1]:.3e} >= {ns[i]:.3e}"
            assert ds[i] > ds[i-1], f"D_opt not monotonic at index {i}: {ds[i-1]:.3e} >= {ds[i]:.3e}"
        print(f"  [PASS] frontier_points returns {len(points)} monotonically increasing (N, D) pairs")
    except Exception as e:
        errors.append(f"  [FAIL] frontier_points monotonicity: {e}")

    # Test 12: Constructor validation
    try:
        try:
            ChinchillaSolver(k=-1.0)
            errors.append("  [FAIL] ChinchillaSolver(k=-1) should raise ValueError")
        except ValueError:
            pass
        try:
            ChinchillaSolver(tokens_per_param=0.0)
            errors.append("  [FAIL] ChinchillaSolver(tokens_per_param=0) should raise ValueError")
        except ValueError:
            pass
        print("  [PASS] ChinchillaSolver constructor validates k and tokens_per_param")
    except Exception as e:
        errors.append(f"  [FAIL] Constructor validation: {e}")

    # Test 13: summary_str includes key fields
    try:
        s = solver.summary_str(70e9, 1.4e12)
        assert "70.000B" in s or "70" in s
        assert "1.4T" in s or "1.400T" in s
        assert "20.0" in s  # target
        assert "1.0000x" in s or "1.0" in s  # ratio
        assert "OK" in s or "compute-optimal" in s
        print("  [PASS] summary_str contains expected fields")
    except Exception as e:
        errors.append(f"  [FAIL] summary_str: {e}")

    # Summary
    if errors:
        print("\nFailed tests:")
        for err in errors:
            print(err)
        raise SystemExit(1)
    else:
        print("\nAll self-tests passed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    _run_self_tests()
