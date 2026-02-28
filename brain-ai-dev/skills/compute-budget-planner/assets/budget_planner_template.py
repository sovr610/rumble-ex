"""
budget_planner_template.py
==========================
BudgetPlanner: core planner with three operational modes.

Mode A (validate_run):    Given an existing run config, validate it against
                          Chinchilla compute-optimal targets and estimate wallclock.
Mode B (compute_required): Given a model size, compute required tokens, steps,
                           wallclock, and cost to train compute-optimally.
Mode C (solve_optimal):   Given a compute budget (FLOPs, time, or money),
                          solve for optimal N and D.

Usage
-----
    from gpu_specs_template import GPUSpecTable
    from budget_config_template import BudgetConfig, RunSpec, ModelSpec, ComputeBudget
    from budget_planner_template import BudgetPlanner

    specs  = GPUSpecTable()
    config = BudgetConfig()
    planner = BudgetPlanner(specs, config)

    # Mode A
    run = RunSpec(n_params=7e9, seq_len=2048, global_batch=2048,
                  steps=488281, num_gpus=8, gpu_type="H100_SXM")
    result = planner.validate_run(run)

    # Mode B
    model = ModelSpec(n_params=70e9)
    result = planner.compute_required(model)

    # Mode C
    budget = ComputeBudget(mode="flops", total_flops=5e23)
    result = planner.solve_optimal(budget)
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Local imports (templates are self-contained; handle missing deps gracefully)
# ---------------------------------------------------------------------------

try:
    from gpu_specs_template import GPUSpecTable, GPUSpec
except ImportError:
    GPUSpecTable = None
    GPUSpec = None

try:
    from budget_config_template import (
        BudgetConfig, RunSpec, ModelSpec, ComputeBudget, BudgetResult,
    )
except ImportError:
    BudgetConfig = None
    RunSpec = None
    ModelSpec = None
    ComputeBudget = None
    BudgetResult = None

try:
    from chinchilla_solver_template import ChinchillaSolver
except ImportError:
    ChinchillaSolver = None


# ---------------------------------------------------------------------------
# Internal helpers (usable even if imports above fail in isolation)
# ---------------------------------------------------------------------------

def _compute_flops(n_params: float, total_tokens: float, k: float = 6.0) -> float:
    """Compute total training FLOPs: C = k * N * D.

    Parameters
    ----------
    n_params : float
        Non-embedding parameter count.
    total_tokens : float
        Total training tokens.
    k : float
        FLOPs coefficient (default 6.0).

    Returns
    -------
    float
        Total training FLOPs.
    """
    if n_params <= 0:
        raise ValueError(f"n_params must be > 0, got {n_params}")
    if total_tokens < 0:
        raise ValueError(f"total_tokens must be >= 0, got {total_tokens}")
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")

    result = k * n_params * total_tokens
    logger.debug(
        "_compute_flops: k=%.1f * N=%.3e * D=%.3e = %.3e",
        k, n_params, total_tokens, result
    )
    return result


def _compute_wallclock(
    total_flops: float,
    num_gpus: int,
    peak_tflops: float,
    utilization: float,
) -> float:
    """Compute predicted wallclock in hours.

    Parameters
    ----------
    total_flops : float
        Total training FLOPs.
    num_gpus : int
        Number of GPUs.
    peak_tflops : float
        Peak TFLOPS per GPU (dense, for the training dtype).
    utilization : float
        MFU (Model FLOPs Utilization) in (0, 1].

    Returns
    -------
    float
        Predicted wallclock in hours (elapsed time for the cluster).
    """
    if total_flops <= 0:
        raise ValueError(f"total_flops must be > 0, got {total_flops}")
    if num_gpus < 1:
        raise ValueError(f"num_gpus must be >= 1, got {num_gpus}")
    if peak_tflops <= 0:
        raise ValueError(f"peak_tflops must be > 0, got {peak_tflops}")
    if utilization <= 0 or utilization > 1.0:
        raise ValueError(f"utilization must be in (0, 1], got {utilization}")

    achieved_flops_per_s = num_gpus * peak_tflops * 1e12 * utilization
    wallclock_s = total_flops / achieved_flops_per_s
    wallclock_h = wallclock_s / 3600.0

    logger.debug(
        "_compute_wallclock: achieved=%.3e FLOPs/s -> %.2f hours",
        achieved_flops_per_s, wallclock_h
    )
    return wallclock_h


def _compute_cost(
    wallclock_hours: float,
    num_gpus: int,
    cost_per_gpu_hour: Optional[float],
) -> Optional[float]:
    """Compute total training cost in USD.

    Parameters
    ----------
    wallclock_hours : float
        Elapsed cluster hours.
    num_gpus : int
        Number of GPUs.
    cost_per_gpu_hour : float or None
        Per-GPU-hour cost. None disables cost estimation.

    Returns
    -------
    float or None
    """
    if cost_per_gpu_hour is None or cost_per_gpu_hour <= 0:
        return None
    cost = wallclock_hours * num_gpus * cost_per_gpu_hour
    logger.debug(
        "_compute_cost: %.2f hours * %d GPUs * $%.2f/hr = $%.2f",
        wallclock_hours, num_gpus, cost_per_gpu_hour, cost
    )
    return cost


def _n_opt(compute_flops: float, k: float, tokens_per_param: float) -> float:
    """Chinchilla N_opt = sqrt(C / (k * a))."""
    return math.sqrt(compute_flops / (k * tokens_per_param))


def _d_opt(n_opt_val: float, tokens_per_param: float) -> float:
    """Chinchilla D_opt = a * N_opt."""
    return tokens_per_param * n_opt_val


def _undertraining_ratio(
    total_tokens: float,
    n_params: float,
    tokens_per_param_target: float,
) -> float:
    """ratio = (total_tokens / n_params) / tokens_per_param_target."""
    planned = total_tokens / n_params
    return planned / tokens_per_param_target


def _build_warnings_and_suggestions(
    ratio: float,
    n_params: float,
    total_tokens: float,
    tokens_per_param_target: float,
    k: float,
    num_gpus: Optional[int] = None,
    peak_tflops: Optional[float] = None,
    utilization: Optional[float] = None,
    seq_len: Optional[int] = None,
    global_batch: Optional[int] = None,
    n_opt_val: Optional[float] = None,
    d_opt_val: Optional[float] = None,
) -> Tuple[List[str], List[str]]:
    """Build warning and suggestion lists from undertraining ratio."""
    warnings: List[str] = []
    suggestions: List[str] = []
    actual_tpp = total_tokens / n_params if n_params > 0 else 0

    if ratio < 0.5:
        warnings.append(
            f"[CRITICAL] Severely undertrained: tokens/param={actual_tpp:.2f} "
            f"(ratio={ratio:.3f}x target={tokens_per_param_target:.1f}). "
            f"The model has seen far too few training tokens."
        )
    elif ratio < 0.8:
        warnings.append(
            f"[WARNING] Likely undertrained: tokens/param={actual_tpp:.2f} "
            f"(ratio={ratio:.3f}x target={tokens_per_param_target:.1f}). "
            f"Consider training for more tokens."
        )
    elif ratio > 2.0:
        warnings.append(
            f"[INFO] Overtrain regime: tokens/param={actual_tpp:.2f} "
            f"(ratio={ratio:.3f}x target={tokens_per_param_target:.1f}). "
            f"This appears intentional (LLaMA-style inference-optimal). Confirm."
        )

    # Build suggestions for undertraining cases
    if ratio < 0.8:
        target_tokens = tokens_per_param_target * n_params
        token_gap = target_tokens - total_tokens
        suggestions.append(
            f"To reach {tokens_per_param_target:.1f} tokens/param for "
            f"{n_params/1e9:.2f}B params: increase to {target_tokens:.3e} tokens "
            f"(add {token_gap:.3e} more)."
        )
        if seq_len and global_batch and seq_len > 0 and global_batch > 0:
            tokens_per_step = seq_len * global_batch
            extra_steps = math.ceil(token_gap / tokens_per_step)
            suggestions.append(
                f"At seq_len={seq_len}, global_batch={global_batch}: "
                f"need {extra_steps:,} additional optimizer steps."
            )
        if num_gpus and peak_tflops and utilization:
            extra_flops = k * n_params * token_gap
            extra_h = extra_flops / (num_gpus * peak_tflops * 1e12 * utilization * 3600)
            suggestions.append(
                f"Estimated additional wallclock at {utilization*100:.0f}% MFU "
                f"on {num_gpus}x GPUs: +{extra_h:.1f} hours."
            )
    elif ratio > 2.0:
        if n_opt_val is not None and d_opt_val is not None:
            suggestions.append(
                f"Compute-optimal for this FLOP budget: "
                f"N_opt={n_opt_val/1e9:.2f}B params on "
                f"D_opt={d_opt_val/1e12:.3f}T tokens."
            )
        suggestions.append(
            "If inference cost is the driver, LLaMA-style overtrain is justified. "
            "No action required if this is intentional."
        )

    return warnings, suggestions


# ---------------------------------------------------------------------------
# BudgetPlanner
# ---------------------------------------------------------------------------

class BudgetPlanner:
    """Core planner with three operational modes.

    Parameters
    ----------
    gpu_specs : GPUSpecTable
        GPU hardware specification registry.
    config : BudgetConfig
        Global planner configuration.

    Methods
    -------
    validate_run(run: RunSpec) -> BudgetResult
        Mode A: validate an existing or planned run against Chinchilla targets.
    compute_required(model: ModelSpec) -> BudgetResult
        Mode B: compute required training budget for a model.
    solve_optimal(budget: ComputeBudget) -> BudgetResult
        Mode C: solve optimal (N, D) for a given compute budget.
    """

    def __init__(self, gpu_specs: Any, config: Any) -> None:
        self.gpu_specs = gpu_specs
        self.config = config
        logger.info(
            "BudgetPlanner initialized: k=%.1f, tokens_per_param_target=%.1f, "
            "default_utilization=%.2f, gpu_type=%s",
            config.k, config.tokens_per_param_target,
            config.default_utilization, config.gpu_type
        )

    # ------------------------------------------------------------------
    # Mode A: Validate Run
    # ------------------------------------------------------------------

    def validate_run(self, run: Any) -> Any:
        """Mode A: Validate an existing or planned training run.

        Given a complete run specification (model size, batch, steps, GPUs),
        compute total_tokens, tokens_per_param, total_FLOPs, predicted_wallclock,
        and generate undertraining warnings and suggestions.

        Parameters
        ----------
        run : RunSpec
            Complete training run specification.

        Returns
        -------
        BudgetResult
        """
        logger.info("Mode A: validate_run called")
        run._validate()

        k = self.config.k
        a = self.config.tokens_per_param_target
        utilization = run.utilization if run.utilization is not None else self.config.default_utilization
        cost_per_gpu_hour = (
            run.cost_per_gpu_hour
            if run.cost_per_gpu_hour is not None
            else self.config.cost_per_gpu_hour
        )

        # Step 1: Token accounting
        total_tokens = run.total_tokens
        tokens_per_param = total_tokens / run.n_params
        logger.debug("total_tokens=%.3e, tokens_per_param=%.2f", total_tokens, tokens_per_param)

        # Step 2: FLOPs
        total_flops = _compute_flops(run.n_params, total_tokens, k)

        # Step 3: GPU specs
        gpu_spec = self.gpu_specs.validate_or_fallback(run.gpu_type, dtype=run.dtype)
        peak_tflops = gpu_spec.peak_tflops

        # Log spec source
        spec_source = "user_supplied" if gpu_spec.is_user_supplied else "table"
        logger.info(
            "GPU spec: %s | dtype=%s | peak_tflops=%.1f | is_sparse=%s | source=%s",
            gpu_spec.name, run.dtype, peak_tflops, gpu_spec.is_sparse, spec_source
        )
        if gpu_spec.is_sparse:
            logger.warning(
                "GPU spec is_sparse=True. Wallclock estimates assume sparse utilization. "
                "Most training runs are DENSE — verify this is correct."
            )

        # Step 4: Wallclock and cost
        wallclock_h = _compute_wallclock(total_flops, run.num_gpus, peak_tflops, utilization)
        cost_usd = _compute_cost(wallclock_h, run.num_gpus, cost_per_gpu_hour)

        # Step 5: Chinchilla check
        ratio = _undertraining_ratio(total_tokens, run.n_params, a)
        n_opt_val = _n_opt(total_flops, k, a)
        d_opt_val = _d_opt(n_opt_val, a)

        # Step 6: Warnings and suggestions
        warnings, suggestions = _build_warnings_and_suggestions(
            ratio=ratio,
            n_params=run.n_params,
            total_tokens=total_tokens,
            tokens_per_param_target=a,
            k=k,
            num_gpus=run.num_gpus,
            peak_tflops=peak_tflops,
            utilization=utilization,
            seq_len=run.seq_len,
            global_batch=run.global_batch,
            n_opt_val=n_opt_val,
            d_opt_val=d_opt_val,
        )

        # Step 7: Build result
        result = BudgetResult(
            mode="validate_run",
            inputs=run.to_dict(),
            assumptions={
                "k": k,
                "tokens_per_param_target": a,
                "utilization": utilization,
                "gpu_type": run.gpu_type,
                "gpu_spec_source": spec_source,
                "peak_tflops": peak_tflops,
                "is_sparse": gpu_spec.is_sparse,
                "dtype": run.dtype,
            },
            derived={
                "total_tokens": total_tokens,
                "tokens_per_param": tokens_per_param,
                "total_flops": total_flops,
                "predicted_wallclock_hours": wallclock_h,
                "predicted_cost_usd": cost_usd,
                "undertraining_ratio": ratio,
                "n_opt": n_opt_val,
                "d_opt": d_opt_val,
            },
            warnings=warnings,
            suggestions=suggestions,
        )

        self._log_result_summary(result)
        return result

    # ------------------------------------------------------------------
    # Mode B: Compute Required
    # ------------------------------------------------------------------

    def compute_required(self, model: Any) -> Any:
        """Mode B: Compute required training budget for a model.

        Given a model specification and token-per-param target, compute the
        number of tokens, optimizer steps, wallclock hours, and cost needed
        for compute-optimal training.

        Parameters
        ----------
        model : ModelSpec
            Model specification.

        Returns
        -------
        BudgetResult
        """
        logger.info("Mode B: compute_required called")
        model._validate()

        k = self.config.k
        a = model.tokens_per_param_target or self.config.tokens_per_param_target
        num_gpus = model.num_gpus or self.config.num_gpus
        gpu_type = model.gpu_type or self.config.gpu_type
        dtype = model.dtype
        utilization = model.utilization if model.utilization is not None else self.config.default_utilization
        cost_per_gpu_hour = (
            model.cost_per_gpu_hour
            if model.cost_per_gpu_hour is not None
            else self.config.cost_per_gpu_hour
        )

        # Step 1: Target tokens
        target_tokens = a * model.n_params
        logger.debug(
            "Mode B: target_tokens = %.1f * %.3e = %.3e",
            a, model.n_params, target_tokens
        )

        # Step 2: Steps (round up to complete all target tokens)
        tokens_per_step = model.global_batch * model.seq_len
        if tokens_per_step <= 0:
            raise ValueError(
                f"tokens_per_step = global_batch * seq_len = "
                f"{model.global_batch} * {model.seq_len} = {tokens_per_step} must be > 0"
            )
        steps = math.ceil(target_tokens / tokens_per_step)
        actual_tokens = steps * tokens_per_step  # may slightly exceed target due to ceil

        # Step 3: FLOPs
        total_flops = _compute_flops(model.n_params, actual_tokens, k)

        # Step 4: GPU specs
        gpu_spec = self.gpu_specs.validate_or_fallback(gpu_type, dtype=dtype)
        peak_tflops = gpu_spec.peak_tflops
        spec_source = "user_supplied" if gpu_spec.is_user_supplied else "table"

        # Step 5: Wallclock and cost
        wallclock_h = _compute_wallclock(total_flops, num_gpus, peak_tflops, utilization)
        cost_usd = _compute_cost(wallclock_h, num_gpus, cost_per_gpu_hour)

        # Step 6: Chinchilla check (should be ratio ≈ 1.0 by construction)
        ratio = _undertraining_ratio(actual_tokens, model.n_params, a)
        n_opt_val = model.n_params          # by definition in Mode B
        d_opt_val = target_tokens

        # Warnings only if rounding caused deviation
        warnings, suggestions = _build_warnings_and_suggestions(
            ratio=ratio,
            n_params=model.n_params,
            total_tokens=actual_tokens,
            tokens_per_param_target=a,
            k=k,
        )

        result = BudgetResult(
            mode="compute_required",
            inputs=model.to_dict(),
            assumptions={
                "k": k,
                "tokens_per_param_target": a,
                "utilization": utilization,
                "num_gpus": num_gpus,
                "gpu_type": gpu_type,
                "gpu_spec_source": spec_source,
                "peak_tflops": peak_tflops,
                "is_sparse": gpu_spec.is_sparse,
                "dtype": dtype,
            },
            derived={
                "target_tokens": target_tokens,
                "actual_tokens": actual_tokens,
                "total_tokens": actual_tokens,
                "tokens_per_param": actual_tokens / model.n_params,
                "required_steps": steps,
                "tokens_per_step": tokens_per_step,
                "total_flops": total_flops,
                "predicted_wallclock_hours": wallclock_h,
                "predicted_cost_usd": cost_usd,
                "undertraining_ratio": ratio,
                "n_opt": n_opt_val,
                "d_opt": d_opt_val,
            },
            warnings=warnings,
            suggestions=suggestions,
        )

        self._log_result_summary(result)
        return result

    # ------------------------------------------------------------------
    # Mode C: Solve Optimal
    # ------------------------------------------------------------------

    def solve_optimal(self, budget: Any) -> Any:
        """Mode C: Solve optimal (N, D) for a given compute budget.

        Given a compute budget specified as FLOPs, GPU-hours, or dollars,
        solve for the Chinchilla-optimal model size and token count.

        Parameters
        ----------
        budget : ComputeBudget
            Compute budget specification.

        Returns
        -------
        BudgetResult
        """
        logger.info("Mode C: solve_optimal called (mode=%s)", budget.mode)
        budget._validate()

        k = budget.k or self.config.k
        a = budget.tokens_per_param_target or self.config.tokens_per_param_target
        utilization = budget.utilization

        # Step 1: Resolve total_flops from budget mode
        gpu_spec = None
        gpu_spec_source = "not_used"
        peak_tflops = None
        num_gpus = None
        wallclock_h = None
        cost_usd = None

        if budget.mode == "flops":
            total_flops = budget.total_flops
            logger.debug("Mode C (flops): total_flops=%.3e", total_flops)

        elif budget.mode == "time":
            gpu_type = budget.gpu_type or self.config.gpu_type
            gpu_spec = self.gpu_specs.validate_or_fallback(gpu_type, dtype="bf16")
            gpu_spec_source = "user_supplied" if gpu_spec.is_user_supplied else "table"
            peak_tflops = gpu_spec.peak_tflops
            num_gpus = budget.num_gpus or self.config.num_gpus
            total_flops = (
                budget.hours * 3600.0 * num_gpus * peak_tflops * 1e12 * utilization
            )
            wallclock_h = budget.hours
            logger.debug(
                "Mode C (time): %g hours * %d GPUs * %.1f TFLOPS * %.2f util = %.3e FLOPs",
                budget.hours, num_gpus, peak_tflops, utilization, total_flops
            )

        elif budget.mode == "money":
            gpu_type = budget.gpu_type or self.config.gpu_type
            gpu_spec = self.gpu_specs.validate_or_fallback(gpu_type, dtype="bf16")
            gpu_spec_source = "user_supplied" if gpu_spec.is_user_supplied else "table"
            peak_tflops = gpu_spec.peak_tflops
            num_gpus = budget.num_gpus or self.config.num_gpus
            cost_per_gpu_hour = budget.cost_per_gpu_hour or self.config.cost_per_gpu_hour
            if not cost_per_gpu_hour:
                raise ValueError(
                    "cost_per_gpu_hour must be set in ComputeBudget or BudgetConfig for mode='money'"
                )
            wallclock_h = budget.budget_dollars / (num_gpus * cost_per_gpu_hour)
            total_flops = wallclock_h * 3600.0 * num_gpus * peak_tflops * 1e12 * utilization
            cost_usd = budget.budget_dollars
            logger.debug(
                "Mode C (money): $%.2f / (%d * $%.2f) = %.2f hours -> %.3e FLOPs",
                budget.budget_dollars, num_gpus, cost_per_gpu_hour, wallclock_h, total_flops
            )

        else:
            raise ValueError(f"Unknown budget mode: {budget.mode!r}")

        # Step 2: Solve N_opt and D_opt
        if total_flops <= 0:
            raise ValueError(f"Resolved total_flops={total_flops} must be > 0")

        n_opt_val = _n_opt(total_flops, k, a)
        d_opt_val = _d_opt(n_opt_val, a)

        logger.info(
            "Mode C solved: N_opt=%.3e (%.2fB), D_opt=%.3e (%.3fT)",
            n_opt_val, n_opt_val/1e9, d_opt_val, d_opt_val/1e12
        )

        # Step 3: Predict wallclock if GPU info is available
        if num_gpus and peak_tflops:
            if wallclock_h is None:
                wallclock_h = _compute_wallclock(total_flops, num_gpus, peak_tflops, utilization)
            cost_usd = cost_usd or _compute_cost(wallclock_h, num_gpus,
                                                  budget.cost_per_gpu_hour or self.config.cost_per_gpu_hour)

        # Step 4: Verification (k * N_opt * D_opt should equal total_flops)
        reconstructed = k * n_opt_val * d_opt_val
        rel_err = abs(reconstructed - total_flops) / total_flops
        if rel_err > 1e-6:
            logger.warning(
                "Solver precision warning: k*N_opt*D_opt=%.6e != C=%.6e (rel_err=%.2e)",
                reconstructed, total_flops, rel_err
            )

        warnings = [
            "N_opt/D_opt are heuristic Chinchilla estimates. "
            "Actual optimal may differ based on architecture, tokenizer, and data distribution."
        ]
        suggestions = [
            f"Train a ~{n_opt_val/1e9:.2f}B parameter model on ~{d_opt_val/1e12:.3f}T tokens.",
            "Use compute_required mode to estimate step count, wallclock, and cost for this config.",
        ]

        result = BudgetResult(
            mode="solve_optimal",
            inputs=budget.to_dict(),
            assumptions={
                "k": k,
                "tokens_per_param_target": a,
                "utilization": utilization,
                "gpu_spec_source": gpu_spec_source,
                "peak_tflops": peak_tflops,
                "num_gpus": num_gpus,
            },
            derived={
                "total_flops": total_flops,
                "total_tokens": d_opt_val,
                "tokens_per_param": a,
                "n_opt": n_opt_val,
                "d_opt": d_opt_val,
                "predicted_wallclock_hours": wallclock_h,
                "predicted_cost_usd": cost_usd,
                "undertraining_ratio": 1.0,    # by definition at optimal
            },
            warnings=warnings,
            suggestions=suggestions,
        )

        self._log_result_summary(result)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_flops(self, n_params: float, total_tokens: float, k: float) -> float:
        """Thin wrapper for module-level _compute_flops with logging."""
        return _compute_flops(n_params, total_tokens, k)

    def _compute_wallclock(
        self,
        total_flops: float,
        num_gpus: int,
        peak_tflops: float,
        utilization: float,
    ) -> float:
        """Thin wrapper for module-level _compute_wallclock."""
        return _compute_wallclock(total_flops, num_gpus, peak_tflops, utilization)

    def _compute_cost(
        self,
        wallclock_hours: float,
        num_gpus: int,
        cost_per_gpu_hour: Optional[float],
    ) -> Optional[float]:
        """Thin wrapper for module-level _compute_cost."""
        return _compute_cost(wallclock_hours, num_gpus, cost_per_gpu_hour)

    def _log_result_summary(self, result: Any) -> None:
        """Log a brief summary of a BudgetResult."""
        d = result.derived
        logger.info(
            "Result [%s]: total_tokens=%.3e, tokens/param=%.2f, "
            "total_flops=%.3e, wallclock=%.1fh, cost=%s, ratio=%.3f",
            result.mode,
            d.get("total_tokens") or 0,
            d.get("tokens_per_param") or 0,
            d.get("total_flops") or 0,
            d.get("predicted_wallclock_hours") or 0,
            f"${d.get('predicted_cost_usd'):.0f}" if d.get("predicted_cost_usd") else "N/A",
            d.get("undertraining_ratio") or 0,
        )
        if result.warnings:
            for w in result.warnings:
                logger.warning("Budget warning: %s", w)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_planner(
    gpu_type: str = "H100_SXM",
    k: float = 6.0,
    tokens_per_param_target: float = 20.0,
    utilization: float = 0.35,
    cost_per_gpu_hour: Optional[float] = None,
    num_gpus: int = 8,
) -> BudgetPlanner:
    """Create a BudgetPlanner with sensible defaults.

    Parameters
    ----------
    gpu_type : str
    k : float
    tokens_per_param_target : float
    utilization : float
    cost_per_gpu_hour : float or None
    num_gpus : int

    Returns
    -------
    BudgetPlanner
    """
    from gpu_specs_template import GPUSpecTable
    from budget_config_template import BudgetConfig

    specs = GPUSpecTable()
    config = BudgetConfig(
        k=k,
        tokens_per_param_target=tokens_per_param_target,
        default_utilization=utilization,
        cost_per_gpu_hour=cost_per_gpu_hour,
        num_gpus=num_gpus,
        gpu_type=gpu_type,
    )
    return BudgetPlanner(specs, config)


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _run_self_tests() -> None:
    print("Running budget_planner_template.py self-tests...")
    errors = []

    # Import inline to support running this file directly
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from gpu_specs_template import GPUSpecTable
        from budget_config_template import (
            BudgetConfig, RunSpec, ModelSpec, ComputeBudget, BudgetResult
        )
    except ImportError as e:
        print(f"  [SKIP] Cannot import dependencies: {e}")
        return

    specs = GPUSpecTable()
    config = BudgetConfig(
        k=6.0,
        tokens_per_param_target=20.0,
        default_utilization=0.35,
        num_gpus=8,
        gpu_type="H100_SXM",
    )
    planner = BudgetPlanner(specs, config)

    # Test 1: Mode A — Chinchilla 70B/1.4T sanity
    try:
        # 1.4T tokens / 2048 seq / 2048 batch = 333862.3 -> 333863 steps
        # actual_tokens = 333863 * 2048 * 2048 = 1400000094208 ≈ 1.4T
        steps = math.ceil(1.4e12 / (2048 * 2048))
        run = RunSpec(
            n_params=70e9, seq_len=2048, global_batch=2048, steps=steps,
            num_gpus=8, gpu_type="H100_SXM", dtype="bf16"
        )
        result = planner.validate_run(run)
        assert isinstance(result, BudgetResult)
        assert result.mode == "validate_run"
        tpp = result.derived["tokens_per_param"]
        assert abs(tpp - 20.0) < 0.01, f"tokens_per_param should be ~20.0, got {tpp:.4f}"
        expected_flops = 6 * 70e9 * run.total_tokens
        assert abs(result.derived["total_flops"] - expected_flops) / expected_flops < 1e-6
        assert result.derived["undertraining_ratio"] >= 0.99
        assert result.derived["undertraining_ratio"] <= 1.01
        assert result.predicted_wallclock_hours > 0
        print(f"  [PASS] Mode A Chinchilla 70B/1.4T sanity: tokens/param={tpp:.4f}, ratio≈1.0")
    except Exception as e:
        errors.append(f"  [FAIL] Mode A Chinchilla sanity: {e}")

    # Test 2: Mode A — undertraining detection
    try:
        run_under = RunSpec(
            n_params=7e9, seq_len=2048, global_batch=2048, steps=500,
            num_gpus=8, gpu_type="H100_SXM"
        )
        result_under = planner.validate_run(run_under)
        # 500 steps * 2048 * 2048 = ~2B tokens; 2B / 7B = ~0.286 tokens/param
        # ratio = 0.286/20 = ~0.014 → CRITICAL
        ratio = result_under.derived["undertraining_ratio"]
        assert ratio < 0.5, f"Expected critical undertraining ratio < 0.5, got {ratio:.4f}"
        assert result_under.has_critical_warnings(), "Expected CRITICAL warning"
        print(f"  [PASS] Mode A undertraining detection: ratio={ratio:.4f}, CRITICAL warning emitted")
    except Exception as e:
        errors.append(f"  [FAIL] Mode A undertraining detection: {e}")

    # Test 3: Mode B — compute_required 70B model
    try:
        model = ModelSpec(n_params=70e9, seq_len=2048, global_batch=2048)
        result_b = planner.compute_required(model)
        assert result_b.mode == "compute_required"
        tpp = result_b.derived["tokens_per_param"]
        assert abs(tpp - 20.0) < 0.001, f"tokens/param should be ~20.0, got {tpp}"
        steps = result_b.derived["required_steps"]
        assert steps > 0
        assert result_b.predicted_wallclock_hours > 0
        # Verify steps produce near-target tokens
        actual_tokens = result_b.derived["actual_tokens"]
        assert abs(actual_tokens - 70e9 * 20) / (70e9 * 20) < 0.001
        print(f"  [PASS] Mode B compute_required 70B: steps={steps:,}, tokens={actual_tokens:.3e}, tokens/param≈20")
    except Exception as e:
        errors.append(f"  [FAIL] Mode B compute_required: {e}")

    # Test 4: Mode C — solve_optimal with FLOPs
    try:
        budget = ComputeBudget(mode="flops", total_flops=5e23)
        result_c = planner.solve_optimal(budget)
        assert result_c.mode == "solve_optimal"
        n_opt = result_c.derived["n_opt"]
        d_opt = result_c.derived["d_opt"]
        assert n_opt > 0 and d_opt > 0
        # Verify k * N_opt * D_opt == 5e23
        reconstructed = 6.0 * n_opt * d_opt
        rel_err = abs(reconstructed - 5e23) / 5e23
        assert rel_err < 1e-6, f"k*N*D={reconstructed:.6e} != 5e23={5e23:.6e}, rel_err={rel_err:.2e}"
        # Verify tokens_per_param == target
        assert abs(d_opt / n_opt - 20.0) < 1e-6
        print(f"  [PASS] Mode C solve_optimal(5e23 FLOPs): N_opt={n_opt/1e9:.2f}B, D_opt={d_opt/1e12:.3f}T")
    except Exception as e:
        errors.append(f"  [FAIL] Mode C solve_optimal (flops): {e}")

    # Test 5: Mode C — solve_optimal with time budget
    try:
        budget_time = ComputeBudget(
            mode="time", hours=1000.0, num_gpus=8,
            gpu_type="H100_SXM", utilization=0.35
        )
        result_ct = planner.solve_optimal(budget_time)
        n_opt = result_ct.derived["n_opt"]
        d_opt = result_ct.derived["d_opt"]
        assert n_opt > 0 and d_opt > 0
        assert result_ct.predicted_wallclock_hours == 1000.0
        print(f"  [PASS] Mode C solve_optimal(1000h on 8xH100): N_opt={n_opt/1e9:.2f}B, D_opt={d_opt/1e12:.3f}T")
    except Exception as e:
        errors.append(f"  [FAIL] Mode C solve_optimal (time): {e}")

    # Test 6: FLOPs monotonicity
    try:
        # Doubling tokens doubles FLOPs
        f1 = _compute_flops(70e9, 1.4e12, 6)
        f2 = _compute_flops(70e9, 2.8e12, 6)
        assert abs(f2 / f1 - 2.0) < 1e-10, f"Expected ratio 2.0, got {f2/f1}"
        # Doubling params doubles FLOPs
        f3 = _compute_flops(70e9, 1.4e12, 6)
        f4 = _compute_flops(140e9, 1.4e12, 6)
        assert abs(f4 / f3 - 2.0) < 1e-10, f"Expected ratio 2.0, got {f4/f3}"
        print("  [PASS] FLOPs monotonicity: doubling tokens or params doubles FLOPs")
    except Exception as e:
        errors.append(f"  [FAIL] FLOPs monotonicity: {e}")

    # Test 7: Wallclock scaling
    try:
        wc_8 = _compute_wallclock(5e23, 8, 989.0, 0.35)
        wc_16 = _compute_wallclock(5e23, 16, 989.0, 0.35)
        assert abs(wc_8 / wc_16 - 2.0) < 1e-10, f"8 GPUs should take 2x longer than 16: {wc_8/wc_16}"
        wc_35 = _compute_wallclock(5e23, 8, 989.0, 0.35)
        wc_70 = _compute_wallclock(5e23, 8, 989.0, 0.70)
        assert abs(wc_35 / wc_70 - 2.0) < 1e-10, f"35% util should take 2x longer than 70%: {wc_35/wc_70}"
        print("  [PASS] Wallclock scaling: doubling GPUs or util halves wallclock")
    except Exception as e:
        errors.append(f"  [FAIL] Wallclock scaling: {e}")

    # Test 8: Cost computation
    try:
        cost = _compute_cost(100.0, 8, 4.0)
        assert abs(cost - 3200.0) < 1e-6, f"Expected $3200.0, got ${cost}"
        cost_none = _compute_cost(100.0, 8, None)
        assert cost_none is None
        print("  [PASS] Cost computation: 100h * 8GPUs * $4/hr = $3200; None when cost_per_gpu_hour not set")
    except Exception as e:
        errors.append(f"  [FAIL] Cost computation: {e}")

    # Test 9: All modes produce valid BudgetResult
    try:
        run = RunSpec(n_params=7e9, seq_len=2048, global_batch=2048, steps=10000, num_gpus=8)
        r_a = planner.validate_run(run)
        r_b = planner.compute_required(ModelSpec(n_params=7e9))
        r_c = planner.solve_optimal(ComputeBudget(mode="flops", total_flops=1e22))
        for r, mode in [(r_a, "validate_run"), (r_b, "compute_required"), (r_c, "solve_optimal")]:
            assert isinstance(r, BudgetResult), f"Mode {mode} did not return BudgetResult"
            assert r.mode == mode
            assert r.warnings is not None
            assert r.suggestions is not None
            assert isinstance(r.derived, dict)
            assert "total_flops" in r.derived
        print("  [PASS] All three modes return valid BudgetResult with required fields")
    except Exception as e:
        errors.append(f"  [FAIL] All modes produce BudgetResult: {e}")

    # Test 10: Mode C with money budget
    try:
        config_with_cost = BudgetConfig(cost_per_gpu_hour=4.0)
        planner2 = BudgetPlanner(specs, config_with_cost)
        budget_money = ComputeBudget(
            mode="money",
            budget_dollars=50000.0,
            cost_per_gpu_hour=4.0,
            num_gpus=8,
            gpu_type="H100_SXM",
            utilization=0.35,
        )
        result_m = planner2.solve_optimal(budget_money)
        assert result_m.derived["n_opt"] > 0
        assert result_m.derived["predicted_cost_usd"] == 50000.0
        print(f"  [PASS] Mode C money budget: N_opt={result_m.derived['n_opt']/1e9:.2f}B")
    except Exception as e:
        errors.append(f"  [FAIL] Mode C money budget: {e}")

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
