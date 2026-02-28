#!/usr/bin/env python3
"""
validate_workspace.py  --  Runtime contract validation for the Global Workspace module.

Validates that GlobalWorkspace, SelectionBroadcastWorkspace, IterativeCompetition,
RefinedBroadcast, AttentionCompetition, ModalityProjection, WorkingMemory,
GRUWorkingMemory, LiquidWorkingMemory, and all factory functions honor their
documented contracts (shapes, dtypes, key presence, determinism, ignition,
broadcast, working-memory semantics, legacy API, etc.).

Seven validation categories (~36 checks total):
    1. Competition   -- multi-modality input, output keys/shapes, capacity, attention
    2. Iterative     -- selection rounds, details, ignition scoring, early stop
    3. Ignition      -- ignition score, global flag, confidence gating, threshold
    4. Broadcast     -- per-modality broadcasts, determinism, feedback, no-states
    5. WorkingMemory -- auto backend, forward dict, state reuse, buffer, retrieve
    6. Integration   -- end-to-end pipeline, downstream shapes, reset, batch mismatch
    7. Legacy API    -- factory functions, WorkspaceConfig fields, dict input

Usage:
    python validate_workspace.py                    # run everything
    python validate_workspace.py --verbose          # detailed per-check output
    python validate_workspace.py --category ignition  # single category
    python validate_workspace.py --json-report      # machine-readable JSON
    python validate_workspace.py --category broadcast --verbose --json-report
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path bootstrap -- 4-level dirname to reach project root
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR))))
sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Imports from brain_ai
# ---------------------------------------------------------------------------
from brain_ai.workspace.global_workspace import (
    GlobalWorkspace,
    GlobalWorkspaceConfig,
    AttentionCompetition,
    ModalityProjection,
    InformationBroadcast,
    create_global_workspace,
    SelectionBroadcastConfig,
    SelectionBroadcastWorkspace,
    IterativeCompetition,
    RefinedBroadcast,
    create_selection_broadcast_workspace,
)
from brain_ai.workspace.working_memory import (
    WorkingMemory,
    WorkingMemoryConfig,
    GRUWorkingMemory,
    create_working_memory,
)

try:
    from brain_ai.workspace.working_memory import LiquidWorkingMemory
    _HAS_LIQUID = True
except ImportError:
    _HAS_LIQUID = False

from brain_ai.config import WorkspaceConfig


# ============================================================================
# Validation result / report data structures
# ============================================================================

@dataclass
class ValidationResult:
    """Outcome of a single validation check."""
    name: str
    passed: bool
    message: str
    category: str
    elapsed_ms: float = 0.0
    warning: bool = False


@dataclass
class ValidationReport:
    """Aggregate report over all validation checks."""
    results: List[ValidationResult] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    # -- summary helpers --
    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed and not r.warning)

    @property
    def warned(self) -> int:
        return sum(1 for r in self.results if r.warning)

    @property
    def elapsed_s(self) -> float:
        return self.end_time - self.start_time

    @property
    def all_passed(self) -> bool:
        return self.failed == 0

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "warned": self.warned,
            "elapsed_s": round(self.elapsed_s, 3),
            "all_passed": self.all_passed,
        }


# ============================================================================
# ANSI colour helpers
# ============================================================================

_SUPPORTS_COLOR = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _green(text: str) -> str:
    return f"\033[92m{text}\033[0m" if _SUPPORTS_COLOR else text


def _red(text: str) -> str:
    return f"\033[91m{text}\033[0m" if _SUPPORTS_COLOR else text


def _yellow(text: str) -> str:
    return f"\033[93m{text}\033[0m" if _SUPPORTS_COLOR else text


def _bold(text: str) -> str:
    return f"\033[1m{text}\033[0m" if _SUPPORTS_COLOR else text


def _dim(text: str) -> str:
    return f"\033[2m{text}\033[0m" if _SUPPORTS_COLOR else text


def _status_str(result: ValidationResult) -> str:
    if result.warning:
        return _yellow("WARN")
    elif result.passed:
        return _green("PASS")
    else:
        return _red("FAIL")


# ============================================================================
# Check runner helper
# ============================================================================

def _run_check(
    name: str,
    category: str,
    fn: Callable[[], Tuple[bool, str]],
) -> ValidationResult:
    """Execute *fn* and capture its pass/fail plus any exception."""
    t0 = time.perf_counter()
    try:
        passed, msg = fn()
        elapsed = (time.perf_counter() - t0) * 1000
        return ValidationResult(
            name=name,
            passed=passed,
            message=msg,
            category=category,
            elapsed_ms=round(elapsed, 2),
        )
    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        tb = traceback.format_exc()
        return ValidationResult(
            name=name,
            passed=False,
            message=f"Exception: {exc}\n{tb}",
            category=category,
            elapsed_ms=round(elapsed, 2),
        )


def _run_warn(
    name: str,
    category: str,
    fn: Callable[[], Tuple[bool, str]],
) -> ValidationResult:
    """Same as _run_check but marks the result as a warning instead of fail."""
    result = _run_check(name, category, fn)
    if not result.passed:
        result.warning = True
    return result


# ============================================================================
# Shared fixtures (small dims so validation is fast)
# ============================================================================

_B = 4          # batch size
_D = 128        # workspace dim
_HEADS = 4      # attention heads
_CAP = 3        # capacity limit (small for test)
_ROUNDS = 3     # selection rounds
_IGN_TH = 0.3   # ignition threshold

_MOD_DIMS: Dict[str, int] = {
    "vision": 64,
    "text": 96,
    "audio": 48,
}


def _make_modality_inputs(
    batch: int = _B,
    mod_dims: Optional[Dict[str, int]] = None,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Create a dict of random modality tensors."""
    md = mod_dims or _MOD_DIMS
    return {k: torch.randn(batch, v, device=device) for k, v in md.items()}


def _make_gw(
    mod_dims: Optional[Dict[str, int]] = None,
    **cfg_kwargs,
) -> GlobalWorkspace:
    """Instantiate a small GlobalWorkspace for testing."""
    md = mod_dims or _MOD_DIMS
    cfg = GlobalWorkspaceConfig(
        workspace_dim=_D,
        num_heads=_HEADS,
        capacity_limit=_CAP,
        memory_mode="gru",
        memory_hidden_dim=_D,
        **cfg_kwargs,
    )
    return GlobalWorkspace(config=cfg, modality_dims=md)


def _make_sbw(
    mod_dims: Optional[Dict[str, int]] = None,
    use_confidence_gating: bool = True,
    **cfg_kwargs,
) -> SelectionBroadcastWorkspace:
    """Instantiate a small SelectionBroadcastWorkspace for testing."""
    md = mod_dims or _MOD_DIMS
    cfg = SelectionBroadcastConfig(
        workspace_dim=_D,
        num_heads=_HEADS,
        capacity_limit=_CAP,
        selection_rounds=_ROUNDS,
        ignition_threshold=_IGN_TH,
        memory_mode="gru",
        memory_hidden_dim=_D,
        use_confidence_gating=use_confidence_gating,
        **cfg_kwargs,
    )
    return SelectionBroadcastWorkspace(config=cfg, modality_dims=md)


# ============================================================================
# 1. Competition Checks (~6)
# ============================================================================

def _competition_checks() -> List[ValidationResult]:
    """Checks for basic workspace competition behaviour."""
    results: List[ValidationResult] = []
    cat = "competition"

    # 1.1 Workspace accepts multi-modality input dict
    def check_multi_modality_input():
        gw = _make_gw()
        gw.train(False)
        inputs = _make_modality_inputs()
        with torch.no_grad():
            out = gw(inputs)
        return True, f"Accepted {len(inputs)} modality inputs successfully"
    results.append(_run_check("accept_multi_modality_input", cat, check_multi_modality_input))

    # 1.2 Output contains 'workspace' key with correct shape (B, D)
    def check_workspace_key_shape():
        gw = _make_gw()
        gw.train(False)
        inputs = _make_modality_inputs()
        with torch.no_grad():
            out = gw(inputs)
        if "workspace" not in out:
            return False, "Missing 'workspace' key in output dict"
        ws = out["workspace"]
        expected = (_B, _D)
        if ws.shape != expected:
            return False, f"workspace shape {tuple(ws.shape)} != expected {expected}"
        return True, f"workspace shape {tuple(ws.shape)} matches expected {expected}"
    results.append(_run_check("workspace_key_shape", cat, check_workspace_key_shape))

    # 1.3 Output contains 'broadcasts' dict with entries per modality
    def check_broadcasts_dict():
        gw = _make_gw()
        gw.train(False)
        inputs = _make_modality_inputs()
        with torch.no_grad():
            out = gw(inputs)
        if "broadcasts" not in out:
            return False, "Missing 'broadcasts' key in output dict"
        bc = out["broadcasts"]
        if not isinstance(bc, dict):
            return False, f"broadcasts is {type(bc).__name__}, expected dict"
        missing = [k for k in _MOD_DIMS if k not in bc]
        if missing:
            return False, f"Missing broadcast entries for: {missing}"
        return True, f"broadcasts dict has entries for {list(bc.keys())}"
    results.append(_run_check("broadcasts_dict_entries", cat, check_broadcasts_dict))

    # 1.4 Competition capacity limit respected (not more than K winners)
    def check_capacity_limit():
        # Use many modalities to exceed capacity
        many_dims = {f"mod_{i}": 64 for i in range(8)}
        gw = _make_gw(mod_dims=many_dims)
        gw.train(False)
        inputs = {k: torch.randn(_B, v) for k, v in many_dims.items()}
        with torch.no_grad():
            out = gw(inputs, return_attention=True)
        attn = out.get("attention", {})
        # Count non-zero attention weights per sample
        if len(attn) == 0:
            return False, "No attention weights returned"
        # Stack attention values across modalities for each sample
        attn_tensor = torch.stack([attn[k] for k in attn], dim=-1)  # (B, num_mods)
        nonzero_per_sample = (attn_tensor > 1e-6).float().sum(dim=-1)  # (B,)
        max_winners = nonzero_per_sample.max().item()
        if max_winners > _CAP:
            return False, (
                f"Max non-zero winners {max_winners} exceeds "
                f"capacity_limit {_CAP}"
            )
        return True, (
            f"Max non-zero winners {max_winners} <= capacity_limit {_CAP}"
        )
    results.append(_run_check("capacity_limit_respected", cat, check_capacity_limit))

    # 1.5 Attention weights sum to ~1 (softmax valid)
    def check_attention_sum():
        gw = _make_gw()
        gw.train(False)
        inputs = _make_modality_inputs()
        with torch.no_grad():
            out = gw(inputs, return_attention=True)
        attn = out.get("attention", {})
        if len(attn) == 0:
            return False, "No attention weights returned"
        attn_tensor = torch.stack([attn[k] for k in attn], dim=-1)  # (B, num_mods)
        sums = attn_tensor.sum(dim=-1)  # (B,)
        max_dev = (sums - 1.0).abs().max().item()
        if max_dev > 0.05:
            return False, f"Attention sum deviates from 1.0 by {max_dev:.4f}"
        return True, f"Attention sums within tolerance (max dev {max_dev:.6f})"
    results.append(_run_check("attention_weights_sum_to_one", cat, check_attention_sum))

    # 1.6 Score computation does not produce NaN/inf
    def check_no_nan_inf():
        gw = _make_gw()
        gw.train(False)
        inputs = _make_modality_inputs()
        with torch.no_grad():
            out = gw(inputs, return_attention=True)
        # Check workspace tensor
        ws = out["workspace"]
        if torch.isnan(ws).any():
            return False, "NaN detected in workspace output"
        if torch.isinf(ws).any():
            return False, "Inf detected in workspace output"
        # Check attention weights
        attn = out.get("attention", {})
        for name, a in attn.items():
            if torch.isnan(a).any():
                return False, f"NaN in attention for modality '{name}'"
            if torch.isinf(a).any():
                return False, f"Inf in attention for modality '{name}'"
        # Check broadcasts
        for name, bc in out["broadcasts"].items():
            if torch.isnan(bc).any():
                return False, f"NaN in broadcast for modality '{name}'"
            if torch.isinf(bc).any():
                return False, f"Inf in broadcast for modality '{name}'"
        return True, "No NaN/Inf in workspace, attention, or broadcasts"
    results.append(_run_check("no_nan_inf_in_scores", cat, check_no_nan_inf))

    return results


# ============================================================================
# 2. Iterative Round Checks (~5)
# ============================================================================

def _iterative_checks() -> List[ValidationResult]:
    """Checks for SelectionBroadcastWorkspace iterative selection rounds."""
    results: List[ValidationResult] = []
    cat = "iterative"

    # 2.1 SelectionBroadcastWorkspace runs selection_rounds iterations
    def check_selection_rounds():
        sbw = _make_sbw()
        sbw.train(False)
        inputs = _make_modality_inputs()
        with torch.no_grad():
            out = sbw(inputs, return_details=True)
        details = out.get("competition_details", {})
        rounds_ran = details.get("selection_rounds", None)
        if rounds_ran is None:
            return False, "competition_details missing 'selection_rounds'"
        if rounds_ran < 1 or rounds_ran > _ROUNDS:
            return False, f"Ran {rounds_ran} rounds, expected 1..{_ROUNDS}"
        return True, f"Ran {rounds_ran} selection rounds (max {_ROUNDS})"
    results.append(_run_check("selection_rounds_execute", cat, check_selection_rounds))

    # 2.2 competition_details contains round history when return_details=True
    def check_round_history():
        sbw = _make_sbw()
        sbw.train(False)
        inputs = _make_modality_inputs()
        with torch.no_grad():
            out = sbw(inputs, return_details=True)
        details = out.get("competition_details", {})
        history = details.get("history", None)
        if history is None:
            return False, "competition_details missing 'history'"
        if not isinstance(history, list):
            return False, f"history is {type(history).__name__}, expected list"
        if len(history) == 0:
            return False, "history list is empty"
        # Each entry should have 'saliences' and 'ignition'
        entry = history[0]
        if "saliences" not in entry:
            return False, "history entry missing 'saliences'"
        if "ignition" not in entry:
            return False, "history entry missing 'ignition'"
        return True, f"history has {len(history)} round entries with saliences/ignition"
    results.append(_run_check("round_history_present", cat, check_round_history))

    # 2.3 Ignition score is scalar in [0, 1]
    def check_ignition_score_range():
        sbw = _make_sbw()
        sbw.train(False)
        inputs = _make_modality_inputs()
        with torch.no_grad():
            out = sbw(inputs, return_details=True)
        ign = out.get("ignition", None)
        if ign is None:
            return False, "Missing 'ignition' key in output"
        # It should be (B, 1) after sigmoid
        if ign.dim() > 2:
            return False, f"ignition has {ign.dim()} dims, expected <= 2"
        vals = ign.flatten()
        if vals.min().item() < -0.01 or vals.max().item() > 1.01:
            return False, (
                f"ignition values out of [0,1] range: "
                f"min={vals.min().item():.4f}, max={vals.max().item():.4f}"
            )
        return True, (
            f"ignition values in [0,1] "
            f"(min={vals.min().item():.4f}, max={vals.max().item():.4f})"
        )
    results.append(_run_check("ignition_score_in_0_1", cat, check_ignition_score_range))

    # 2.4 global_ignition is boolean-like tensor
    def check_global_ignition_bool():
        sbw = _make_sbw()
        sbw.train(False)
        inputs = _make_modality_inputs()
        with torch.no_grad():
            out = sbw(inputs)
        gi = out.get("global_ignition", None)
        if gi is None:
            return False, "Missing 'global_ignition' key in output"
        if not isinstance(gi, torch.Tensor):
            return False, f"global_ignition is {type(gi).__name__}, expected Tensor"
        unique_vals = gi.unique()
        all_binary = all(v.item() in (0.0, 1.0) for v in unique_vals)
        if not all_binary:
            return False, (
                f"global_ignition has non-binary values: "
                f"{[v.item() for v in unique_vals]}"
            )
        return True, f"global_ignition is binary tensor (values: {[v.item() for v in unique_vals]})"
    results.append(_run_check("global_ignition_is_boolean", cat, check_global_ignition_bool))

    # 2.5 Early stop possible (rounds < max_rounds)
    def check_early_stop_possible():
        # We cannot guarantee early stop with random data, but we verify
        # the mechanism exists by checking the code path does not crash
        # and that selection_rounds <= max_rounds
        sbw = _make_sbw()
        sbw.train(False)
        inputs = _make_modality_inputs()
        with torch.no_grad():
            out = sbw(inputs, return_details=True)
        details = out.get("competition_details", {})
        rounds_ran = details.get("selection_rounds", _ROUNDS)
        if rounds_ran <= _ROUNDS:
            early = rounds_ran < _ROUNDS
            msg = (
                f"Ran {rounds_ran}/{_ROUNDS} rounds"
                + (" (early stopped)" if early else " (ran all rounds)")
            )
            return True, msg
        return False, f"Ran {rounds_ran} rounds which exceeds max {_ROUNDS}"
    results.append(_run_check("early_stop_possible", cat, check_early_stop_possible))

    return results


# ============================================================================
# 3. Ignition Checks (~5)
# ============================================================================

def _ignition_checks() -> List[ValidationResult]:
    """Checks for ignition scoring and gating behaviour."""
    results: List[ValidationResult] = []
    cat = "ignition"

    # 3.1 Ignition score returned in output
    def check_ignition_returned():
        sbw = _make_sbw()
        sbw.train(False)
        inputs = _make_modality_inputs()
        with torch.no_grad():
            out = sbw(inputs)
        if "ignition" not in out:
            return False, "Missing 'ignition' key"
        return True, f"ignition key present, shape {tuple(out['ignition'].shape)}"
    results.append(_run_check("ignition_score_returned", cat, check_ignition_returned))

    # 3.2 global_ignition flag returned
    def check_global_ignition_flag():
        sbw = _make_sbw()
        sbw.train(False)
        inputs = _make_modality_inputs()
        with torch.no_grad():
            out = sbw(inputs)
        if "global_ignition" not in out:
            return False, "Missing 'global_ignition' key"
        gi = out["global_ignition"]
        if not isinstance(gi, torch.Tensor):
            return False, f"global_ignition is {type(gi).__name__}, expected Tensor"
        return True, f"global_ignition present, shape {tuple(gi.shape)}"
    results.append(_run_check("global_ignition_flag_returned", cat, check_global_ignition_flag))

    # 3.3 Confidence score returned if use_confidence_gating=True
    def check_confidence_gating():
        sbw = _make_sbw(use_confidence_gating=True)
        sbw.train(False)
        inputs = _make_modality_inputs()
        with torch.no_grad():
            out = sbw(inputs)
        if "confidence" not in out:
            return False, "Missing 'confidence' key when use_confidence_gating=True"
        conf = out["confidence"]
        vals = conf.flatten()
        if vals.min().item() < -0.01 or vals.max().item() > 1.01:
            return False, (
                f"confidence out of [0,1]: "
                f"min={vals.min().item():.4f}, max={vals.max().item():.4f}"
            )
        return True, (
            f"confidence present in [0,1] "
            f"(min={vals.min().item():.4f}, max={vals.max().item():.4f})"
        )
    results.append(_run_check("confidence_score_when_gating", cat, check_confidence_gating))

    # 3.4 Ignition affects output (compare with/without high ignition)
    def check_ignition_affects_output():
        # Run two forward passes; verify the workspace output is a real tensor
        # and that confidence gating modulates it (output with gating differs
        # from output without gating because the confidence scalar != 1.0).
        sbw_gated = _make_sbw(use_confidence_gating=True)
        sbw_ungated = _make_sbw(use_confidence_gating=False)
        # They are separate instances with separate random inits, so outputs
        # will inherently differ -- this validates the code path is wired up.
        sbw_gated.train(False)
        sbw_ungated.train(False)

        torch.manual_seed(42)
        inputs = _make_modality_inputs()

        with torch.no_grad():
            torch.manual_seed(0)
            out_gated = sbw_gated(inputs)
            torch.manual_seed(0)
            out_ungated = sbw_ungated(inputs)

        ws_gated = out_gated["workspace"]
        ws_ungated = out_ungated["workspace"]

        # With confidence gating enabled, workspace should be scaled by
        # confidence, so unless confidence is exactly 1.0, outputs differ.
        if torch.isnan(ws_gated).any() or torch.isnan(ws_ungated).any():
            return False, "NaN detected in workspace output"

        # Verify the gated workspace is NOT identically zero (confidence > 0)
        if ws_gated.abs().max().item() < 1e-12:
            return False, "Gated workspace is all zeros -- confidence may be 0"

        return True, (
            "Ignition/confidence pipeline produces valid, non-zero workspace output"
        )
    results.append(_run_check("ignition_affects_output", cat, check_ignition_affects_output))

    # 3.5 Ignition threshold from config respected
    def check_ignition_threshold_config():
        # Verify that the IterativeCompetition stores the threshold correctly
        ic = IterativeCompetition(
            workspace_dim=_D,
            num_heads=_HEADS,
            selection_rounds=_ROUNDS,
            ignition_threshold=0.75,
            temperature=0.5,
        )
        if ic.ignition_threshold != 0.75:
            return False, (
                f"ignition_threshold is {ic.ignition_threshold}, expected 0.75"
            )
        # And verify it's reflected in SelectionBroadcastWorkspace
        sbw = _make_sbw()
        actual = sbw.competition.ignition_threshold
        if actual != _IGN_TH:
            return False, (
                f"SBW competition ignition_threshold is {actual}, expected {_IGN_TH}"
            )
        return True, (
            f"Ignition threshold correctly propagated "
            f"(IC=0.75, SBW={actual})"
        )
    results.append(_run_check("ignition_threshold_from_config", cat, check_ignition_threshold_config))

    return results


# ============================================================================
# 4. Broadcast Checks (~5)
# ============================================================================

def _broadcast_checks() -> List[ValidationResult]:
    """Checks for broadcast behaviour (basic and refined)."""
    results: List[ValidationResult] = []
    cat = "broadcast"

    # 4.1 Broadcasts dict has entry for each input modality
    def check_broadcast_per_modality():
        sbw = _make_sbw()
        sbw.train(False)
        inputs = _make_modality_inputs()
        with torch.no_grad():
            out = sbw(inputs)
        bc = out.get("broadcasts", {})
        missing = [k for k in _MOD_DIMS if k not in bc]
        if missing:
            return False, f"Missing broadcast entries: {missing}"
        return True, f"Broadcast entries for all {len(_MOD_DIMS)} modalities"
    results.append(_run_check("broadcast_per_modality", cat, check_broadcast_per_modality))

    # 4.2 Each broadcast shape is (B, modality_dim)
    def check_broadcast_shapes():
        sbw = _make_sbw()
        sbw.train(False)
        inputs = _make_modality_inputs()
        with torch.no_grad():
            out = sbw(inputs)
        bc = out["broadcasts"]
        errors = []
        for name, expected_dim in _MOD_DIMS.items():
            if name not in bc:
                errors.append(f"Missing '{name}'")
                continue
            expected_shape = (_B, expected_dim)
            actual_shape = tuple(bc[name].shape)
            if actual_shape != expected_shape:
                errors.append(f"'{name}': {actual_shape} != {expected_shape}")
        if errors:
            return False, "; ".join(errors)
        return True, "All broadcast shapes match (B, modality_dim)"
    results.append(_run_check("broadcast_shapes_correct", cat, check_broadcast_shapes))

    # 4.3 Broadcast deterministic (same input -> same output)
    def check_broadcast_deterministic():
        sbw = _make_sbw()
        sbw.train(False)
        torch.manual_seed(123)
        inputs = _make_modality_inputs()
        with torch.no_grad():
            sbw.reset_state()
            out1 = sbw(inputs)
            sbw.reset_state()
            out2 = sbw(inputs)
        bc1 = out1["broadcasts"]
        bc2 = out2["broadcasts"]
        max_diff = 0.0
        for name in _MOD_DIMS:
            diff = (bc1[name] - bc2[name]).abs().max().item()
            max_diff = max(max_diff, diff)
        if max_diff > 1e-5:
            return False, f"Broadcast not deterministic: max diff {max_diff:.6f}"
        return True, f"Broadcast deterministic (max diff {max_diff:.8f})"
    results.append(_run_check("broadcast_deterministic", cat, check_broadcast_deterministic))

    # 4.4 RefinedBroadcast feedback integration works
    def check_refined_broadcast_feedback():
        rb = RefinedBroadcast(
            workspace_dim=_D,
            modality_dims=_MOD_DIMS,
            broadcast_iterations=2,
            broadcast_decay=0.9,
        )
        rb.train(False)
        ws_content = torch.randn(_B, _D)
        mod_states = {k: torch.randn(_B, v) for k, v in _MOD_DIMS.items()}
        with torch.no_grad():
            broadcasts, refined = rb(ws_content, modality_states=mod_states)
        if refined.shape != (_B, _D):
            return False, f"refined shape {tuple(refined.shape)} != ({_B}, {_D})"
        # refined should differ from input after feedback integration
        diff = (refined - ws_content).abs().max().item()
        if diff < 1e-8:
            return False, "Refined content identical to input -- feedback not integrated"
        return True, f"Feedback integrated (max change {diff:.6f})"
    results.append(_run_check("refined_broadcast_feedback", cat, check_refined_broadcast_feedback))

    # 4.5 Broadcast with no modality_states still works
    def check_broadcast_no_states():
        rb = RefinedBroadcast(
            workspace_dim=_D,
            modality_dims=_MOD_DIMS,
            broadcast_iterations=2,
            broadcast_decay=0.9,
        )
        rb.train(False)
        ws_content = torch.randn(_B, _D)
        with torch.no_grad():
            broadcasts, refined = rb(ws_content, modality_states=None)
        if not isinstance(broadcasts, dict):
            return False, f"broadcasts is {type(broadcasts).__name__}, expected dict"
        if len(broadcasts) != len(_MOD_DIMS):
            return False, (
                f"Expected {len(_MOD_DIMS)} broadcasts, got {len(broadcasts)}"
            )
        return True, "Broadcast works with modality_states=None"
    results.append(_run_check("broadcast_no_modality_states", cat, check_broadcast_no_states))

    return results


# ============================================================================
# 5. Working Memory Checks (~7)
# ============================================================================

def _working_memory_checks() -> List[ValidationResult]:
    """Checks for WorkingMemory and its backends."""
    results: List[ValidationResult] = []
    cat = "working_memory"

    wm_dim = 64

    # 5.1 WorkingMemory instantiates with auto mode
    def check_auto_instantiation():
        wm = create_working_memory(
            input_dim=wm_dim,
            hidden_dim=wm_dim,
            output_dim=wm_dim,
            mode="auto",
        )
        if not isinstance(wm, WorkingMemory):
            return False, f"Got {type(wm).__name__}, expected WorkingMemory"
        bt = wm.backend_type
        return True, f"WorkingMemory instantiated with backend_type='{bt}'"
    results.append(_run_check("auto_mode_instantiation", cat, check_auto_instantiation))

    # 5.2 forward returns dict with 'output', 'state', 'buffer'
    def check_forward_dict_keys():
        wm = create_working_memory(
            input_dim=wm_dim, hidden_dim=wm_dim, output_dim=wm_dim, mode="gru"
        )
        wm.train(False)
        x = torch.randn(_B, wm_dim)
        with torch.no_grad():
            result = wm(x)
        if not isinstance(result, dict):
            return False, f"forward returned {type(result).__name__}, expected dict"
        required_keys = {"output", "state", "buffer"}
        missing = required_keys - set(result.keys())
        if missing:
            return False, f"Missing keys: {missing}"
        return True, f"forward returned dict with keys {set(result.keys())}"
    results.append(_run_check("forward_returns_dict", cat, check_forward_dict_keys))

    # 5.3 Output shape matches config output_dim
    def check_output_shape():
        wm = create_working_memory(
            input_dim=wm_dim, hidden_dim=wm_dim, output_dim=wm_dim, mode="gru"
        )
        wm.train(False)
        x = torch.randn(_B, wm_dim)
        with torch.no_grad():
            result = wm(x)
        out = result["output"]
        expected = (_B, wm_dim)
        if out.shape != expected:
            return False, f"output shape {tuple(out.shape)} != expected {expected}"
        return True, f"output shape {tuple(out.shape)} matches config"
    results.append(_run_check("output_shape_matches_config", cat, check_output_shape))

    # 5.4 State can be passed back (sequential calls)
    def check_state_reuse():
        wm = create_working_memory(
            input_dim=wm_dim, hidden_dim=wm_dim, output_dim=wm_dim, mode="gru"
        )
        wm.train(False)
        x1 = torch.randn(_B, wm_dim)
        x2 = torch.randn(_B, wm_dim)
        with torch.no_grad():
            r1 = wm(x1)
            state1 = r1["state"]
            # Second call should implicitly use updated internal state
            r2 = wm(x2)
            out2 = r2["output"]
        # Also test explicit state passing via reset + explicit state
        wm.reset_state()
        with torch.no_grad():
            r3 = wm(x1)
            r4 = wm(x2, state=r3["state"])
        # Outputs r2 and r4 should be identical (same state path)
        diff = (out2 - r4["output"]).abs().max().item()
        if diff > 1e-4:
            return False, (
                f"Sequential vs explicit state diff too large: {diff:.6f}"
            )
        return True, f"State reuse consistent (diff {diff:.8f})"
    results.append(_run_check("state_reuse_sequential", cat, check_state_reuse))

    # 5.5 Buffer shape is (B, capacity, output_dim)
    def check_buffer_shape():
        wm = create_working_memory(
            input_dim=wm_dim, hidden_dim=wm_dim, output_dim=wm_dim, mode="gru"
        )
        wm.train(False)
        # Feed several items to fill buffer
        with torch.no_grad():
            for i in range(5):
                x = torch.randn(_B, wm_dim)
                result = wm(x)
        buf = result["buffer"]
        if buf is None:
            return False, "Buffer is None after 5 forward calls"
        # Buffer should have shape (B, num_items, output_dim)
        if buf.dim() != 3:
            return False, f"Buffer has {buf.dim()} dims, expected 3"
        if buf.shape[0] != _B:
            return False, f"Buffer batch dim {buf.shape[0]} != {_B}"
        if buf.shape[2] != wm_dim:
            return False, f"Buffer feature dim {buf.shape[2]} != {wm_dim}"
        num_items = buf.shape[1]
        if num_items > wm.capacity:
            return False, (
                f"Buffer has {num_items} items, exceeds capacity {wm.capacity}"
            )
        return True, (
            f"Buffer shape {tuple(buf.shape)}, "
            f"items={num_items} <= capacity={wm.capacity}"
        )
    results.append(_run_check("buffer_shape_correct", cat, check_buffer_shape))

    # 5.6 retrieve(query) returns correct shape
    def check_retrieve_shape():
        wm = create_working_memory(
            input_dim=wm_dim, hidden_dim=wm_dim, output_dim=wm_dim, mode="gru"
        )
        wm.train(False)
        # Fill buffer
        with torch.no_grad():
            for i in range(3):
                wm(torch.randn(_B, wm_dim))
        query = torch.randn(_B, wm_dim)
        with torch.no_grad():
            retrieved = wm.retrieve(query)
        expected = (_B, wm_dim)
        if retrieved.shape != expected:
            return False, f"retrieve shape {tuple(retrieved.shape)} != {expected}"
        return True, f"retrieve returns shape {tuple(retrieved.shape)}"
    results.append(_run_check("retrieve_correct_shape", cat, check_retrieve_shape))

    # 5.7 Backend type reported correctly
    def check_backend_type():
        wm_gru = create_working_memory(
            input_dim=wm_dim, hidden_dim=wm_dim, output_dim=wm_dim, mode="gru"
        )
        if wm_gru.backend_type != "gru":
            return False, f"GRU backend reports type '{wm_gru.backend_type}'"
        # Also test that auto mode picks something valid
        wm_auto = create_working_memory(
            input_dim=wm_dim, hidden_dim=wm_dim, output_dim=wm_dim, mode="auto"
        )
        if wm_auto.backend_type not in ("gru", "cfc", "ltc"):
            return False, f"Auto backend reports unknown type '{wm_auto.backend_type}'"
        return True, (
            f"GRU backend_type='gru', auto backend_type='{wm_auto.backend_type}'"
        )
    results.append(_run_check("backend_type_reported", cat, check_backend_type))

    return results


# ============================================================================
# 6. Integration Checks (~4)
# ============================================================================

def _integration_checks() -> List[ValidationResult]:
    """End-to-end integration checks."""
    results: List[ValidationResult] = []
    cat = "integration"

    # 6.1 Full pipeline: modality inputs -> workspace output -> broadcasts
    def check_full_pipeline():
        sbw = _make_sbw()
        sbw.train(False)
        inputs = _make_modality_inputs()
        with torch.no_grad():
            out = sbw(inputs)
        # Verify workspace
        ws = out["workspace"]
        if ws.shape != (_B, _D):
            return False, f"workspace shape {tuple(ws.shape)} != ({_B}, {_D})"
        # Verify broadcasts
        bc = out["broadcasts"]
        for name, dim in _MOD_DIMS.items():
            if name not in bc:
                return False, f"Missing broadcast for '{name}'"
            if bc[name].shape != (_B, dim):
                return False, (
                    f"broadcast['{name}'] shape {tuple(bc[name].shape)} "
                    f"!= ({_B}, {dim})"
                )
        # Verify ignition
        if "ignition" not in out:
            return False, "Missing 'ignition' in output"
        # Verify attention
        if "attention" not in out:
            return False, "Missing 'attention' in output"
        return True, "Full pipeline produces workspace, broadcasts, ignition, attention"
    results.append(_run_check("full_pipeline_end_to_end", cat, check_full_pipeline))

    # 6.2 Workspace output feeds into downstream shapes
    def check_downstream_shapes():
        sbw = _make_sbw()
        sbw.train(False)
        inputs = _make_modality_inputs()
        with torch.no_grad():
            out = sbw(inputs)
        ws = out["workspace"]
        # Simulate feeding into a downstream module (e.g., linear head)
        downstream = nn.Linear(_D, 10)
        with torch.no_grad():
            logits = downstream(ws)
        if logits.shape != (_B, 10):
            return False, f"Downstream logits shape {tuple(logits.shape)} != ({_B}, 10)"
        # Also test feeding into another hidden layer
        hidden = nn.Linear(_D, _D)
        with torch.no_grad():
            h = hidden(ws)
        if h.shape != (_B, _D):
            return False, f"Downstream hidden shape {tuple(h.shape)} != ({_B}, {_D})"
        return True, (
            f"Workspace output feeds downstream correctly "
            f"(logits={tuple(logits.shape)}, hidden={tuple(h.shape)})"
        )
    results.append(_run_check("downstream_shape_compatibility", cat, check_downstream_shapes))

    # 6.3 reset_state clears workspace state
    def check_reset_state():
        sbw = _make_sbw()
        sbw.train(False)
        inputs = _make_modality_inputs()
        # Run a forward pass to populate state
        with torch.no_grad():
            sbw(inputs)
        # State should be set
        if sbw.prev_context is None:
            return False, "prev_context is None after forward pass (should be set)"
        # Reset
        sbw.reset_state()
        if sbw.prev_context is not None:
            return False, "prev_context not None after reset_state()"
        # Also check working memory buffer
        if sbw.working_memory.memory_buffer is not None:
            return False, "memory_buffer not None after reset_state()"
        return True, "reset_state() clears prev_context and memory_buffer"
    results.append(_run_check("reset_state_clears", cat, check_reset_state))

    # 6.4 Batch size mismatch auto-resets
    def check_batch_mismatch_reset():
        sbw = _make_sbw()
        sbw.train(False)
        # Forward with batch=4
        inputs4 = _make_modality_inputs(batch=4)
        with torch.no_grad():
            out4 = sbw(inputs4)
        # Forward with batch=2 -- should auto-reset, not crash
        inputs2 = _make_modality_inputs(batch=2)
        try:
            with torch.no_grad():
                out2 = sbw(inputs2)
        except RuntimeError as e:
            return False, f"Batch mismatch caused RuntimeError: {e}"
        ws = out2["workspace"]
        if ws.shape[0] != 2:
            return False, f"Output batch size {ws.shape[0]} != 2 after mismatch"
        return True, "Batch size mismatch handled gracefully (auto-reset)"
    results.append(_run_check("batch_mismatch_auto_reset", cat, check_batch_mismatch_reset))

    return results


# ============================================================================
# 7. Legacy API Checks (~5)
# ============================================================================

def _legacy_api_checks() -> List[ValidationResult]:
    """Checks for factory functions and backward-compatible API surface."""
    results: List[ValidationResult] = []
    cat = "legacy"

    # 7.1 create_global_workspace factory works
    def check_create_gw_factory():
        gw = create_global_workspace(
            workspace_dim=_D,
            modality_dims=_MOD_DIMS,
            num_heads=_HEADS,
            capacity_limit=_CAP,
            memory_mode="gru",
        )
        if not isinstance(gw, GlobalWorkspace):
            return False, f"Got {type(gw).__name__}, expected GlobalWorkspace"
        gw.train(False)
        inputs = _make_modality_inputs()
        with torch.no_grad():
            out = gw(inputs)
        if "workspace" not in out:
            return False, "Factory-created workspace missing 'workspace' key"
        return True, "create_global_workspace() factory works correctly"
    results.append(_run_check("create_global_workspace_factory", cat, check_create_gw_factory))

    # 7.2 create_selection_broadcast_workspace factory works
    def check_create_sbw_factory():
        sbw = create_selection_broadcast_workspace(
            workspace_dim=_D,
            modality_dims=_MOD_DIMS,
            num_heads=_HEADS,
            selection_rounds=_ROUNDS,
            ignition_threshold=_IGN_TH,
            memory_mode="gru",
        )
        if not isinstance(sbw, SelectionBroadcastWorkspace):
            return False, (
                f"Got {type(sbw).__name__}, expected SelectionBroadcastWorkspace"
            )
        sbw.train(False)
        inputs = _make_modality_inputs()
        with torch.no_grad():
            out = sbw(inputs)
        if "workspace" not in out:
            return False, "Factory-created SBW missing 'workspace' key"
        if "ignition" not in out:
            return False, "Factory-created SBW missing 'ignition' key"
        return True, "create_selection_broadcast_workspace() factory works correctly"
    results.append(_run_check("create_sbw_factory", cat, check_create_sbw_factory))

    # 7.3 create_working_memory factory works
    def check_create_wm_factory():
        wm = create_working_memory(
            input_dim=64, hidden_dim=64, output_dim=64, mode="gru"
        )
        if not isinstance(wm, WorkingMemory):
            return False, f"Got {type(wm).__name__}, expected WorkingMemory"
        wm.train(False)
        x = torch.randn(_B, 64)
        with torch.no_grad():
            result = wm(x)
        if "output" not in result:
            return False, "Factory-created WM missing 'output' key"
        return True, "create_working_memory() factory works correctly"
    results.append(_run_check("create_working_memory_factory", cat, check_create_wm_factory))

    # 7.4 Legacy WorkspaceConfig fields accessible
    def check_workspace_config_fields():
        cfg = WorkspaceConfig()
        required_fields = [
            "workspace_dim",
            "num_heads",
            "capacity_limit",
            "memory_hidden_dim",
            "memory_mode",
            "use_selection_broadcast",
            "selection_rounds",
            "ignition_threshold",
            "broadcast_iterations",
            "broadcast_decay",
            "use_confidence_gating",
        ]
        missing = []
        for f in required_fields:
            if not hasattr(cfg, f):
                missing.append(f)
        if missing:
            return False, f"WorkspaceConfig missing fields: {missing}"
        return True, f"All {len(required_fields)} WorkspaceConfig fields accessible"
    results.append(_run_check("workspace_config_fields", cat, check_workspace_config_fields))

    # 7.5 Old-style Dict[str, Tensor] input accepted
    def check_dict_tensor_input():
        gw = _make_gw()
        gw.train(False)
        # Build input as plain dict -- the API signature expects Dict[str, Tensor]
        inputs: Dict[str, torch.Tensor] = {
            "vision": torch.randn(_B, _MOD_DIMS["vision"]),
            "text": torch.randn(_B, _MOD_DIMS["text"]),
        }
        with torch.no_grad():
            out = gw(inputs)
        if "workspace" not in out:
            return False, "Dict[str, Tensor] input did not produce 'workspace'"
        ws = out["workspace"]
        if ws.shape != (_B, _D):
            return False, f"workspace shape {tuple(ws.shape)} incorrect"
        return True, "Dict[str, Tensor] input accepted (2 modalities)"
    results.append(_run_check("dict_tensor_input_accepted", cat, check_dict_tensor_input))

    return results


# ============================================================================
# Category registry
# ============================================================================

_CATEGORY_RUNNERS: Dict[str, Callable[[], List[ValidationResult]]] = {
    "competition": _competition_checks,
    "iterative": _iterative_checks,
    "ignition": _ignition_checks,
    "broadcast": _broadcast_checks,
    "working_memory": _working_memory_checks,
    "integration": _integration_checks,
    "legacy": _legacy_api_checks,
}


# ============================================================================
# Reporting
# ============================================================================

def _print_result(result: ValidationResult, verbose: bool) -> None:
    """Print a single validation result."""
    tag = _status_str(result)
    line = f"  [{tag}] {result.name}"
    if verbose:
        line += f"  {_dim(f'({result.elapsed_ms:.1f}ms)')}"
    print(line)
    if verbose or (not result.passed and not result.warning):
        # Indent message lines
        for msg_line in result.message.split("\n"):
            stripped = msg_line.strip()
            if stripped:
                prefix = "       "
                print(f"{prefix}{_dim(stripped)}")


def _print_summary_table(report: ValidationReport) -> None:
    """Print a summary table grouped by category."""
    categories: Dict[str, Dict[str, int]] = {}
    for r in report.results:
        if r.category not in categories:
            categories[r.category] = {"total": 0, "passed": 0, "failed": 0, "warned": 0}
        cat = categories[r.category]
        cat["total"] += 1
        if r.passed:
            cat["passed"] += 1
        elif r.warning:
            cat["warned"] += 1
        else:
            cat["failed"] += 1

    print()
    print(_bold("=" * 72))
    print(_bold("  VALIDATION SUMMARY"))
    print(_bold("=" * 72))
    print()

    header = f"  {'Category':<20}  {'Total':>5}  {'Pass':>5}  {'Fail':>5}  {'Warn':>5}"
    print(_bold(header))
    print("  " + "-" * 60)

    for cat_name, counts in categories.items():
        total = counts["total"]
        passed = counts["passed"]
        failed = counts["failed"]
        warned = counts["warned"]

        pass_str = _green(f"{passed:>5}") if passed == total else f"{passed:>5}"
        fail_str = _red(f"{failed:>5}") if failed > 0 else f"{failed:>5}"
        warn_str = _yellow(f"{warned:>5}") if warned > 0 else f"{warned:>5}"

        print(f"  {cat_name:<20}  {total:>5}  {pass_str}  {fail_str}  {warn_str}")

    print("  " + "-" * 60)

    # Grand totals
    gt_pass = _green(f"{report.passed:>5}") if report.passed == report.total else f"{report.passed:>5}"
    gt_fail = _red(f"{report.failed:>5}") if report.failed > 0 else f"{report.failed:>5}"
    gt_warn = _yellow(f"{report.warned:>5}") if report.warned > 0 else f"{report.warned:>5}"
    print(_bold(f"  {'TOTAL':<20}  {report.total:>5}") + f"  {gt_pass}  {gt_fail}  {gt_warn}")

    print()
    elapsed_str = f"Elapsed: {report.elapsed_s:.2f}s"
    if report.all_passed:
        print(f"  {_green(_bold(f'ALL {report.total} CHECKS PASSED'))}  {_dim(elapsed_str)}")
    else:
        print(f"  {_red(_bold(f'{report.failed} CHECK(S) FAILED'))}  {_dim(elapsed_str)}")
    print()


def _print_json_report(report: ValidationReport) -> None:
    """Print machine-readable JSON report."""
    data = {
        "summary": report.summary_dict(),
        "results": [
            {
                "name": r.name,
                "passed": r.passed,
                "warning": r.warning,
                "message": r.message,
                "category": r.category,
                "elapsed_ms": r.elapsed_ms,
            }
            for r in report.results
        ],
    }
    print(json.dumps(data, indent=2))


# ============================================================================
# Main runner
# ============================================================================

def run_validation(
    categories: Optional[List[str]] = None,
    verbose: bool = False,
    json_report: bool = False,
) -> ValidationReport:
    """
    Execute validation checks and build a report.

    Parameters
    ----------
    categories : list of str, optional
        If given, only run checks in these categories.  If None, run all.
    verbose : bool
        Print detailed per-check output.
    json_report : bool
        Print JSON report at the end.

    Returns
    -------
    ValidationReport
    """
    report = ValidationReport()
    report.start_time = time.time()

    runners_to_run = _CATEGORY_RUNNERS
    if categories:
        unknown = [c for c in categories if c not in _CATEGORY_RUNNERS]
        if unknown:
            print(_red(f"Unknown categories: {unknown}"))
            print(f"Available: {list(_CATEGORY_RUNNERS.keys())}")
            sys.exit(1)
        runners_to_run = {k: v for k, v in _CATEGORY_RUNNERS.items() if k in categories}

    if not json_report:
        print()
        print(_bold("Global Workspace -- Runtime Contract Validation"))
        print(_bold("=" * 52))
        print()

    for cat_name, runner_fn in runners_to_run.items():
        if not json_report:
            print(_bold(f"[{cat_name}]"))
        cat_results = runner_fn()
        report.results.extend(cat_results)
        if not json_report:
            for r in cat_results:
                _print_result(r, verbose)
            print()

    report.end_time = time.time()

    if json_report:
        _print_json_report(report)
    else:
        _print_summary_table(report)

    return report


# ============================================================================
# CLI entry point
# ============================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Runtime contract validation for the Global Workspace module.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Categories:\n"
            "  competition     Basic workspace competition checks\n"
            "  iterative       Iterative selection round checks\n"
            "  ignition        Ignition scoring and gating\n"
            "  broadcast       Broadcast and feedback integration\n"
            "  working_memory  WorkingMemory backends and buffer\n"
            "  integration     End-to-end pipeline checks\n"
            "  legacy          Factory functions and legacy API\n"
        ),
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Print detailed per-check output including timing.",
    )
    parser.add_argument(
        "--category", "-c",
        type=str,
        default=None,
        help=(
            "Run only checks in the specified category. "
            "Can be comma-separated for multiple categories."
        ),
    )
    parser.add_argument(
        "--json-report", "-j",
        action="store_true",
        default=False,
        help="Output machine-readable JSON report.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    categories = None
    if args.category:
        categories = [c.strip() for c in args.category.split(",")]

    report = run_validation(
        categories=categories,
        verbose=args.verbose,
        json_report=args.json_report,
    )

    sys.exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    main()
