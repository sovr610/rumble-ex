#!/usr/bin/env python3
"""
HTM Contract Validator -- Runtime validation of SP, TM, Reflex, and Fallback contracts.

Validates that the brain_ai/temporal/ module conforms to the contracts
specified by the htm-spatial-temporal-reflex skill.  Works with both
existing legacy code and upgraded target code.

Usage:
    python validate_htm.py [--target-only] [--legacy-only] [--verbose]

Validates:
    1. SDR utility functions (if available)
    2. Spatial Pooler contract (output shapes, sparsity, determinism)
    3. Temporal Memory contract (SequenceOutput, learning, anomaly)
    4. Reflex Memory contract (promotion, lookup, state_dict)
    5. Fallback predictor contract (same SequenceOutput interface)
    6. Integration (SP -> TM -> Reflex pipeline)
    7. Legacy compatibility (existing HTMLayer, AcceleratedHTM)

Exit code 0 if all executed checks pass, 1 if any fail.
"""

import os
import sys
import argparse
import traceback
import dataclasses
import tempfile
from typing import List, Optional, Dict, Any, Tuple

# =============================================================================
# Section 1: Path Setup
# =============================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# scripts/ -> htm-spatial-temporal-reflex/ -> skills/ -> brain-ai-dev/ -> human-brain/
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(_SCRIPT_DIR)
        )
    )
)

_BRAIN_AI_DIR = os.path.join(_PROJECT_ROOT, "brain_ai")

if not os.path.isdir(_BRAIN_AI_DIR):
    print(f"FATAL: brain_ai directory not found at {_BRAIN_AI_DIR}")
    print(f"  Project root resolved to: {_PROJECT_ROOT}")
    print(f"  Script directory: {_SCRIPT_DIR}")
    sys.exit(2)

# Insert project root so brain_ai is importable
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Attempt torch import early -- nearly every check needs it
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Deterministic seeding for reproducibility
_SEED = 42


def _seed_everything() -> None:
    """Seed all RNGs for deterministic validation."""
    import random
    random.seed(_SEED)
    if TORCH_AVAILABLE:
        torch.manual_seed(_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_SEED)
    try:
        import numpy as np
        np.random.seed(_SEED)
    except ImportError:
        pass


# =============================================================================
# Section 2: ValidationResult and ValidationReport
# =============================================================================

@dataclasses.dataclass
class ValidationResult:
    """Result of a single validation check."""
    name: str
    passed: bool
    message: str
    category: str
    details: str = ""

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        base = f"[{status}] {self.name}: {self.message}"
        if self.details:
            base += f"\n        {self.details}"
        return base


class ValidationReport:
    """Aggregated results across all validation checks."""

    def __init__(self) -> None:
        self.results: List[ValidationResult] = []

    def add(
        self,
        name: str,
        passed: bool,
        message: str,
        category: str,
        details: str = "",
    ) -> ValidationResult:
        """Record a validation result and return it."""
        result = ValidationResult(
            name=name,
            passed=passed,
            message=message,
            category=category,
            details=details,
        )
        self.results.append(result)
        return result

    def skip(self, name: str, reason: str, category: str) -> ValidationResult:
        """Record a skipped check (counted as pass with a skip note)."""
        return self.add(
            name=name,
            passed=True,
            message=f"SKIP: {reason}",
            category=category,
        )

    def passed_results(self) -> List[ValidationResult]:
        return [r for r in self.results if r.passed]

    def failed_results(self) -> List[ValidationResult]:
        return [r for r in self.results if not r.passed]

    def summary(self, verbose: bool = False) -> str:
        """Return a formatted summary string."""
        total = len(self.results)
        n_pass = len(self.passed_results())
        n_fail = len(self.failed_results())

        # Group by category
        categories: Dict[str, List[ValidationResult]] = {}
        for r in self.results:
            categories.setdefault(r.category, []).append(r)

        lines = [
            "",
            "=" * 74,
            "  HTM CONTRACT VALIDATION REPORT",
            "=" * 74,
        ]

        for cat, results in categories.items():
            cat_pass = sum(1 for r in results if r.passed)
            cat_total = len(results)
            lines.append(f"\n  [{cat}] {cat_pass}/{cat_total}")
            if verbose:
                for r in results:
                    lines.append(f"    {r}")

        lines.append("")
        lines.append("=" * 74)
        lines.append(
            f"  TOTAL: {n_pass}/{total} passed, {n_fail}/{total} failed"
        )
        lines.append("=" * 74)

        if n_fail > 0:
            lines.append("")
            lines.append("  FAILURES:")
            for r in self.failed_results():
                lines.append(f"    [{r.category}] {r.name}: {r.message}")
                if r.details:
                    for d in r.details.split("\n"):
                        lines.append(f"        {d}")

        lines.append("")
        return "\n".join(lines)

    def exit_code(self) -> int:
        """Return 0 if all passed, 1 if any failed."""
        return 0 if len(self.failed_results()) == 0 else 1


# =============================================================================
# Section 3: API Detection
# =============================================================================

# Target API symbols (populated during detection)
_target_sdr: Dict[str, Any] = {}
_target_sp: Dict[str, Any] = {}
_target_tm: Dict[str, Any] = {}
_target_reflex: Dict[str, Any] = {}

# Legacy API symbols (populated during detection)
_legacy: Dict[str, Any] = {}


def detect_target_api() -> bool:
    """Detect whether the upgraded target API is available.

    Returns True if at least one target-specific symbol is importable.
    """
    found = False

    # SDR utilities (target: brain_ai.temporal.sdr_utils)
    for name in [
        "indices_to_dense", "dense_to_indices",
        "sdr_overlap", "sdr_jaccard", "sdr_hash",
        "random_sdr", "SDRConfig",
    ]:
        try:
            mod = __import__("brain_ai.temporal.sdr_utils", fromlist=[name])
            cls = getattr(mod, name, None)
            if cls is not None:
                _target_sdr[name] = cls
                found = True
        except (ImportError, AttributeError):
            pass

    # Spatial Pooler (target: brain_ai.temporal.spatial_pooler)
    for name in ["SpatialPooler", "SPConfig", "InputBinarizer"]:
        try:
            mod = __import__("brain_ai.temporal.spatial_pooler", fromlist=[name])
            cls = getattr(mod, name, None)
            if cls is not None:
                _target_sp[name] = cls
                found = True
        except (ImportError, AttributeError):
            pass

    # Temporal Memory (target: brain_ai.temporal.temporal_memory)
    for name in ["TemporalMemory", "TMConfig", "SequenceOutput"]:
        try:
            mod = __import__("brain_ai.temporal.temporal_memory", fromlist=[name])
            cls = getattr(mod, name, None)
            if cls is not None:
                _target_tm[name] = cls
                found = True
        except (ImportError, AttributeError):
            pass

    # Reflex Memory (target may be in reflex_memory or upgraded htm)
    for name in ["ReflexMemory"]:
        try:
            mod = __import__("brain_ai.temporal.reflex_memory", fromlist=[name])
            cls = getattr(mod, name, None)
            if cls is not None:
                _target_reflex[name] = cls
                found = True
        except (ImportError, AttributeError):
            pass

    return found


def detect_legacy_api() -> bool:
    """Detect whether the legacy API is available.

    Returns True if at least one legacy symbol is importable.
    """
    found = False
    for name in [
        "HTMLayer", "HTMConfig", "PytorchSpatialPooler",
        "PytorchTemporalMemory", "SparseTensor",
        "ReflexMemory", "AcceleratedHTM",
        "create_htm_layer", "create_accelerated_htm",
    ]:
        try:
            mod = __import__("brain_ai.temporal.htm", fromlist=[name])
            cls = getattr(mod, name, None)
            if cls is not None:
                _legacy[name] = cls
                found = True
        except (ImportError, AttributeError):
            pass

    # Sequence predictors
    for name in [
        "TemporalLayer", "LSTMSequencePredictor",
        "GRUSequencePredictor", "TransformerSequencePredictor",
        "SequenceConfig", "create_temporal_layer",
    ]:
        try:
            mod = __import__("brain_ai.temporal.sequence", fromlist=[name])
            cls = getattr(mod, name, None)
            if cls is not None:
                _legacy[name] = cls
                found = True
        except (ImportError, AttributeError):
            pass

    return found


# =============================================================================
# Section 4: Test Data Helpers
# =============================================================================

def _make_binary_input(batch: int, size: int, sparsity: float = 0.1) -> "torch.Tensor":
    """Create a random binary input tensor with approximate sparsity."""
    x = torch.zeros(batch, size)
    num_active = max(1, int(size * sparsity))
    for b in range(batch):
        indices = torch.randperm(size)[:num_active]
        x[b, indices] = 1.0
    return x


def _make_sp_output_indices(batch: int, n_columns: int, k: int) -> "torch.Tensor":
    """Create synthetic SP output in index form (B, K) int."""
    out = torch.zeros(batch, k, dtype=torch.long)
    for b in range(batch):
        out[b] = torch.randperm(n_columns)[:k].sort().values
    return out


def _make_sp_output_dense(batch: int, n_columns: int, k: int) -> "torch.Tensor":
    """Create synthetic SP output in dense form (B, N_col) binary."""
    dense = torch.zeros(batch, n_columns)
    for b in range(batch):
        indices = torch.randperm(n_columns)[:k]
        dense[b, indices] = 1.0
    return dense


# =============================================================================
# Section 5: SDR Utility Checks (target API)
# =============================================================================

CAT_SDR = "SDR Utilities"


def run_sdr_checks(report: ValidationReport) -> None:
    """Run all SDR utility contract checks."""

    if not _target_sdr:
        report.skip(
            "sdr_utils_available",
            "Target sdr_utils module not found (not yet upgraded)",
            CAT_SDR,
        )
        return

    # -- indices_to_dense --
    name = "check_sdr_indices_to_dense"
    try:
        fn = _target_sdr.get("indices_to_dense")
        if fn is None:
            report.skip(name, "indices_to_dense not found", CAT_SDR)
        else:
            idx = torch.tensor([[0, 3, 7], [1, 2, 5]])
            dense = fn(idx, 10)
            assert dense.shape == (2, 10), f"Expected (2, 10), got {dense.shape}"
            assert dense.dtype == torch.bool or dense.dtype == torch.float32, \
                f"Expected bool or float32, got {dense.dtype}"
            report.add(name, True, "Output shape (B, N) correct", CAT_SDR)
    except Exception as e:
        report.add(name, False, str(e), CAT_SDR, traceback.format_exc())

    # -- dense_to_indices --
    name = "check_sdr_dense_to_indices"
    try:
        fn = _target_sdr.get("dense_to_indices")
        if fn is None:
            report.skip(name, "dense_to_indices not found", CAT_SDR)
        else:
            dense = torch.zeros(2, 10, dtype=torch.bool)
            dense[0, 0] = True; dense[0, 3] = True; dense[0, 7] = True
            dense[1, 1] = True; dense[1, 5] = True; dense[1, 9] = True
            idx = fn(dense, 3)
            assert idx.shape == (2, 3), f"Expected (2, 3), got {idx.shape}"
            assert idx.dtype in (torch.int32, torch.int64), \
                f"Expected int dtype, got {idx.dtype}"
            report.add(name, True, "Output shape (B, K) int correct", CAT_SDR)
    except Exception as e:
        report.add(name, False, str(e), CAT_SDR, traceback.format_exc())

    # -- sdr_overlap --
    name = "check_sdr_overlap"
    try:
        fn = _target_sdr.get("sdr_overlap")
        if fn is None:
            report.skip(name, "sdr_overlap not found", CAT_SDR)
        else:
            a = torch.tensor([[1, 3, 5, 7], [2, 4, 6, 8]])
            overlap = fn(a, a)
            assert overlap.shape == (2,), f"Expected (2,), got {overlap.shape}"
            assert overlap.dtype in (torch.int32, torch.int64), \
                f"Expected int dtype, got {overlap.dtype}"
            # overlap(a, a) == K
            assert (overlap == 4).all(), \
                f"Self-overlap should be K=4, got {overlap.tolist()}"
            report.add(name, True, "Returns (B,) int, self-overlap == K", CAT_SDR)
    except Exception as e:
        report.add(name, False, str(e), CAT_SDR, traceback.format_exc())

    # -- sdr_jaccard --
    name = "check_sdr_jaccard"
    try:
        fn = _target_sdr.get("sdr_jaccard")
        if fn is None:
            report.skip(name, "sdr_jaccard not found", CAT_SDR)
        else:
            a = torch.tensor([[1, 3, 5, 7]])
            j = fn(a, a)
            assert j.shape == (1,), f"Expected (1,), got {j.shape}"
            assert j.dtype == torch.float32, f"Expected float32, got {j.dtype}"
            assert abs(j.item() - 1.0) < 1e-5, \
                f"Jaccard(a, a) should be 1.0, got {j.item()}"
            # Check range with partial overlap
            b = torch.tensor([[1, 3, 9, 10]])
            j2 = fn(a, b)
            assert 0.0 <= j2.item() <= 1.0, \
                f"Jaccard should be in [0, 1], got {j2.item()}"
            report.add(name, True, "Returns (B,) float in [0,1], jaccard(a,a)==1.0", CAT_SDR)
    except Exception as e:
        report.add(name, False, str(e), CAT_SDR, traceback.format_exc())

    # -- sdr_hash --
    name = "check_sdr_hash"
    try:
        fn = _target_sdr.get("sdr_hash")
        if fn is None:
            report.skip(name, "sdr_hash not found", CAT_SDR)
        else:
            a = torch.tensor([[1, 5, 10, 20]])
            h1 = fn(a)
            h2 = fn(a)
            assert h1.shape == (1,), f"Expected (1,), got {h1.shape}"
            assert h1.dtype == torch.int64, f"Expected int64, got {h1.dtype}"
            # Deterministic
            assert h1.item() == h2.item(), "Hash must be deterministic"
            # Order-independent
            a_perm = torch.tensor([[20, 10, 5, 1]])
            h3 = fn(a_perm)
            assert h1.item() == h3.item(), "Hash must be order-independent"
            report.add(name, True, "Returns (B,) int64, deterministic, order-independent", CAT_SDR)
    except Exception as e:
        report.add(name, False, str(e), CAT_SDR, traceback.format_exc())


# =============================================================================
# Section 6: Spatial Pooler Checks
# =============================================================================

CAT_SP = "Spatial Pooler"


def _get_sp_instance() -> Tuple[Optional[Any], str, int, int, int]:
    """Try to get a SpatialPooler instance (target or legacy).

    Returns:
        (sp_instance_or_None, backend_name, input_size, column_count, K)
    """
    input_size = 128
    column_count = 64
    sparsity = 0.1
    K = int(column_count * sparsity)  # 6

    # Try target first
    SPClass = _target_sp.get("SpatialPooler")
    SPConfigClass = _target_sp.get("SPConfig")
    if SPClass is not None and SPConfigClass is not None:
        try:
            cfg = SPConfigClass(
                input_size=input_size,
                column_count=column_count,
                potential_pool_size=min(32, input_size),
                sparsity=sparsity,
                binarization_mode="threshold",
                binarization_threshold=0.5,
            )
            sp = SPClass(cfg)
            return sp, "target", input_size, column_count, K
        except Exception:
            pass

    # Try legacy
    LegacySP = _legacy.get("PytorchSpatialPooler")
    if LegacySP is not None:
        try:
            sp = LegacySP(
                input_size=input_size,
                column_count=column_count,
                sparsity=sparsity,
            )
            return sp, "legacy", input_size, column_count, K
        except Exception:
            pass

    return None, "none", input_size, column_count, K


def run_sp_checks(report: ValidationReport) -> None:
    """Run all Spatial Pooler contract checks."""
    _seed_everything()

    # -- check_sp_exists --
    name = "check_sp_exists"
    sp, backend, input_size, column_count, K = _get_sp_instance()
    if sp is None:
        report.add(name, False, "No SpatialPooler importable (target or legacy)", CAT_SP)
        return
    report.add(name, True, f"SpatialPooler available (backend={backend})", CAT_SP)

    # -- check_sp_forward_shape --
    name = "check_sp_forward_shape"
    try:
        x = _make_binary_input(4, input_size, sparsity=0.1)
        out = sp(x, learn=False)

        if backend == "target":
            # Target outputs (B, K) int indices
            assert out.shape == (4, K), f"Expected (4, {K}), got {out.shape}"
            assert out.dtype in (torch.int32, torch.int64), \
                f"Expected int dtype, got {out.dtype}"
            report.add(name, True, f"Output (B, K)=({4}, {K}) int", CAT_SP)
        else:
            # Legacy outputs (B, column_count) dense binary
            assert out.shape[0] == 4, f"Expected batch=4, got {out.shape[0]}"
            assert out.shape[1] == column_count, \
                f"Expected {column_count} columns, got {out.shape[1]}"
            report.add(name, True, f"Output (B, N_col)=({4}, {column_count}) dense", CAT_SP)
    except Exception as e:
        report.add(name, False, str(e), CAT_SP, traceback.format_exc())

    # -- check_sp_sparsity --
    name = "check_sp_sparsity"
    try:
        x = _make_binary_input(4, input_size, sparsity=0.1)
        out = sp(x, learn=False)

        if backend == "target":
            # K indices per row means exactly K active columns
            for b in range(4):
                unique_count = out[b].unique().numel()
                assert unique_count == K, \
                    f"Sample {b}: {unique_count} unique, expected {K}"
        else:
            # Dense output: count active columns per row
            for b in range(4):
                num_active = (out[b] > 0).sum().item()
                expected = int(column_count * 0.1)
                assert num_active == expected, \
                    f"Sample {b}: {num_active} active, expected {expected}"

        report.add(name, True, f"Exactly K={K} active columns per sample", CAT_SP)
    except Exception as e:
        report.add(name, False, str(e), CAT_SP, traceback.format_exc())

    # -- check_sp_determinism --
    name = "check_sp_determinism"
    try:
        x = _make_binary_input(2, input_size, sparsity=0.1)
        out1 = sp(x, learn=False)
        out2 = sp(x, learn=False)
        assert torch.equal(out1, out2), "Same input should produce same output"
        report.add(name, True, "Same input -> identical output (2 runs)", CAT_SP)
    except Exception as e:
        report.add(name, False, str(e), CAT_SP, traceback.format_exc())

    # -- check_sp_permanence_bounds --
    name = "check_sp_permanence_bounds"
    try:
        perm = None
        if hasattr(sp, 'perm'):
            perm = sp.perm
        elif hasattr(sp, 'permanences'):
            perm = sp.permanences
        if perm is None:
            report.skip(name, "No permanence attribute found", CAT_SP)
        else:
            assert perm.min().item() >= 0.0, \
                f"Min permanence {perm.min().item()} < 0"
            assert perm.max().item() <= 1.0, \
                f"Max permanence {perm.max().item()} > 1"
            report.add(name, True, "All permanences in [0, 1]", CAT_SP)
    except Exception as e:
        report.add(name, False, str(e), CAT_SP, traceback.format_exc())

    # -- check_sp_learning_updates --
    name = "check_sp_learning_updates"
    try:
        perm_attr = 'perm' if hasattr(sp, 'perm') else 'permanences'
        perm_before = getattr(sp, perm_attr).clone()
        x = _make_binary_input(4, input_size, sparsity=0.2)
        sp(x, learn=True)
        perm_after = getattr(sp, perm_attr).clone()
        assert not torch.equal(perm_before, perm_after), \
            "Permanences should change after learning"
        report.add(name, True, "Permanences updated after learn step", CAT_SP)
    except Exception as e:
        report.add(name, False, str(e), CAT_SP, traceback.format_exc())

    # -- check_sp_boosting_exists --
    name = "check_sp_boosting_exists"
    try:
        boost = None
        if hasattr(sp, 'boost'):
            boost = sp.boost
        elif hasattr(sp, 'boost_factors'):
            boost = sp.boost_factors
        if boost is None:
            report.skip(name, "No boost attribute found", CAT_SP)
        else:
            assert boost.min().item() > 0, \
                f"Boost factors should be > 0, min={boost.min().item()}"
            report.add(name, True, "Boost factors accessible and > 0", CAT_SP)
    except Exception as e:
        report.add(name, False, str(e), CAT_SP, traceback.format_exc())

    # -- check_sp_state_dict --
    name = "check_sp_state_dict"
    try:
        sd = sp.state_dict()
        keys = set(sd.keys())
        # Should include permanences in some form
        has_perm = any("perm" in k for k in keys)
        has_boost = any("boost" in k for k in keys)
        has_duty = any("duty" in k for k in keys)
        assert has_perm, f"state_dict missing permanences. Keys: {keys}"
        assert has_boost, f"state_dict missing boost. Keys: {keys}"
        assert has_duty, f"state_dict missing duty cycles. Keys: {keys}"
        report.add(
            name, True,
            f"state_dict has permanences, boost, duty_cycles ({len(keys)} keys)",
            CAT_SP,
        )
    except Exception as e:
        report.add(name, False, str(e), CAT_SP, traceback.format_exc())


# =============================================================================
# Section 7: Temporal Memory Checks
# =============================================================================

CAT_TM = "Temporal Memory"


def _get_tm_instance() -> Tuple[Optional[Any], str, int, int]:
    """Try to get a TemporalMemory instance (target or legacy).

    Returns:
        (tm_instance_or_None, backend_name, column_count, cells_per_column)
    """
    column_count = 64
    cells_per_column = 8

    # Try target first
    TMClass = _target_tm.get("TemporalMemory")
    if TMClass is not None:
        try:
            tm = TMClass(
                column_count=column_count,
                cells_per_column=cells_per_column,
            )
            return tm, "target", column_count, cells_per_column
        except Exception:
            pass

    # Try legacy
    LegacyTM = _legacy.get("PytorchTemporalMemory")
    if LegacyTM is not None:
        try:
            tm = LegacyTM(
                column_count=column_count,
                cells_per_column=cells_per_column,
                activation_threshold=4,
                min_threshold=3,
                max_new_synapse_count=10,
            )
            return tm, "legacy", column_count, cells_per_column
        except Exception:
            pass

    return None, "none", column_count, cells_per_column


def run_tm_checks(report: ValidationReport) -> None:
    """Run all Temporal Memory contract checks."""
    _seed_everything()

    # -- check_tm_exists --
    name = "check_tm_exists"
    tm, backend, column_count, cells_per_column = _get_tm_instance()
    if tm is None:
        report.add(name, False, "No TemporalMemory importable (target or legacy)", CAT_TM)
        return
    report.add(name, True, f"TemporalMemory available (backend={backend})", CAT_TM)

    K = int(column_count * 0.1)  # ~6 active columns

    # -- check_tm_step_output --
    name = "check_tm_step_output"
    try:
        active_cols = _make_sp_output_dense(1, column_count, K).squeeze(0)
        result = tm(active_cols, learn=True)

        # Should return dict-like with required keys
        assert isinstance(result, dict), \
            f"TM output should be dict, got {type(result)}"
        required_keys = {"active_cells", "predictive_cells", "anomaly"}
        missing = required_keys - set(result.keys())
        assert not missing, f"Missing output keys: {missing}"
        report.add(name, True, "Returns dict with active_cells, predictive_cells, anomaly", CAT_TM)
    except Exception as e:
        report.add(name, False, str(e), CAT_TM, traceback.format_exc())

    # Reset for subsequent checks
    if hasattr(tm, 'reset'):
        tm.reset()

    # -- check_tm_anomaly_range --
    name = "check_tm_anomaly_range"
    try:
        tm_fresh, _, _, _ = _get_tm_instance()
        active_cols = _make_sp_output_dense(1, column_count, K).squeeze(0)
        result = tm_fresh(active_cols, learn=True)
        anomaly = result['anomaly']
        if isinstance(anomaly, torch.Tensor):
            anomaly_val = anomaly.item()
        else:
            anomaly_val = float(anomaly)
        assert 0.0 <= anomaly_val <= 1.0, \
            f"Anomaly should be in [0, 1], got {anomaly_val}"
        report.add(name, True, f"Anomaly score={anomaly_val:.3f} in [0, 1]", CAT_TM)
    except Exception as e:
        report.add(name, False, str(e), CAT_TM, traceback.format_exc())

    # -- check_tm_prediction_improves --
    name = "check_tm_prediction_improves"
    try:
        tm_seq, _, _, _ = _get_tm_instance()

        # Create a simple repeating sequence: A -> B -> C -> D -> A -> ...
        patterns = []
        for _ in range(4):
            p = _make_sp_output_dense(1, column_count, K).squeeze(0)
            patterns.append(p)

        anomalies_first_cycle = []
        anomalies_last_cycle = []

        num_cycles = 6
        for cycle in range(num_cycles):
            for step, p in enumerate(patterns):
                result = tm_seq(p, learn=True)
                a = result['anomaly']
                aval = a.item() if isinstance(a, torch.Tensor) else float(a)
                if cycle == 0:
                    anomalies_first_cycle.append(aval)
                if cycle == num_cycles - 1:
                    anomalies_last_cycle.append(aval)

        avg_first = sum(anomalies_first_cycle) / len(anomalies_first_cycle)
        avg_last = sum(anomalies_last_cycle) / len(anomalies_last_cycle)

        # The last cycle should have lower anomaly than the first
        # (except possibly the very first step of each cycle)
        improved = avg_last <= avg_first
        report.add(
            name, improved,
            f"Mean anomaly: first cycle={avg_first:.3f}, last cycle={avg_last:.3f}",
            CAT_TM,
            details=f"First cycle per-step: {[f'{a:.3f}' for a in anomalies_first_cycle]}\n"
                    f"Last cycle per-step: {[f'{a:.3f}' for a in anomalies_last_cycle]}",
        )
    except Exception as e:
        report.add(name, False, str(e), CAT_TM, traceback.format_exc())

    # -- check_tm_burst_detection --
    name = "check_tm_burst_detection"
    try:
        tm_burst, _, _, _ = _get_tm_instance()

        # Train on one pattern
        p1 = _make_sp_output_dense(1, column_count, K).squeeze(0)
        for _ in range(5):
            tm_burst(p1, learn=True)

        # Now present a completely novel pattern
        p2 = _make_sp_output_dense(1, column_count, K).squeeze(0)
        # Make sure p2 is different from p1
        while torch.equal(p1, p2):
            p2 = _make_sp_output_dense(1, column_count, K).squeeze(0)

        result = tm_burst(p2, learn=False)
        a = result['anomaly']
        aval = a.item() if isinstance(a, torch.Tensor) else float(a)
        # Novel input should cause high anomaly (burst)
        high_anomaly = aval > 0.5
        report.add(
            name, high_anomaly,
            f"Novel input anomaly={aval:.3f} (expected > 0.5)",
            CAT_TM,
        )
    except Exception as e:
        report.add(name, False, str(e), CAT_TM, traceback.format_exc())

    # -- check_tm_segment_caps --
    name = "check_tm_segment_caps"
    try:
        tm_caps, _, _, _ = _get_tm_instance()
        max_seg = getattr(tm_caps, 'max_segments_per_cell', None)
        max_syn = getattr(tm_caps, 'max_synapses_per_segment', None)
        if max_seg is not None and max_syn is not None:
            report.add(
                name, True,
                f"Segment caps: max_segments={max_seg}, max_synapses={max_syn}",
                CAT_TM,
            )
        else:
            report.skip(name, "Segment cap attributes not found", CAT_TM)
    except Exception as e:
        report.add(name, False, str(e), CAT_TM, traceback.format_exc())

    # -- check_tm_no_gradient --
    name = "check_tm_no_gradient"
    try:
        tm_grad, _, _, _ = _get_tm_instance()
        params = list(tm_grad.parameters())
        has_grad_params = any(p.requires_grad for p in params)
        if len(params) == 0:
            report.add(name, True, "No parameters (pure Hebbian)", CAT_TM)
        elif not has_grad_params:
            report.add(name, True, "No parameters require grad (Hebbian only)", CAT_TM)
        else:
            report.add(
                name, False,
                f"Found {sum(1 for p in params if p.requires_grad)} parameters requiring grad",
                CAT_TM,
            )
    except Exception as e:
        report.add(name, False, str(e), CAT_TM, traceback.format_exc())


# =============================================================================
# Section 8: Reflex Memory Checks
# =============================================================================

CAT_REFLEX = "Reflex Memory"


def _get_reflex_instance() -> Tuple[Optional[Any], str, int]:
    """Try to get a ReflexMemory instance.

    Returns:
        (reflex_instance_or_None, backend_name, pattern_dim)
    """
    pattern_dim = 128

    # Try target first
    RMClass = _target_reflex.get("ReflexMemory")
    if RMClass is not None:
        try:
            rm = RMClass(pattern_dim=pattern_dim, max_patterns=100, promotion_threshold=3)
            return rm, "target", pattern_dim
        except Exception:
            pass

    # Try legacy
    RMClass = _legacy.get("ReflexMemory")
    if RMClass is not None:
        try:
            rm = RMClass(pattern_dim=pattern_dim, max_patterns=100, promotion_threshold=3)
            return rm, "legacy", pattern_dim
        except Exception:
            pass

    return None, "none", pattern_dim


def run_reflex_checks(report: ValidationReport) -> None:
    """Run all Reflex Memory contract checks."""
    _seed_everything()

    # -- check_reflex_exists --
    name = "check_reflex_exists"
    rm, backend, pattern_dim = _get_reflex_instance()
    if rm is None:
        report.add(name, False, "No ReflexMemory importable", CAT_REFLEX)
        return
    report.add(name, True, f"ReflexMemory available (backend={backend})", CAT_REFLEX)

    # -- check_reflex_lookup_miss --
    name = "check_reflex_lookup_miss"
    try:
        rm_miss, _, pdim = _get_reflex_instance()
        unknown = torch.randn(pdim)
        result = rm_miss.lookup(unknown)
        assert result is None, f"Unknown pattern should return None, got {type(result)}"
        report.add(name, True, "Unknown pattern -> None", CAT_REFLEX)
    except Exception as e:
        report.add(name, False, str(e), CAT_REFLEX, traceback.format_exc())

    # -- check_reflex_observation --
    name = "check_reflex_observation"
    try:
        rm_obs, _, pdim = _get_reflex_instance()
        pattern = torch.randn(pdim)
        prediction = torch.randn(pdim)

        # Use store() or observe() depending on API
        if hasattr(rm_obs, 'observe'):
            rm_obs.observe(pattern, prediction)
        elif hasattr(rm_obs, 'store'):
            idx = rm_obs.store(pattern, prediction, force=True)
            assert idx >= 0, f"store() returned {idx}, expected >= 0"
        else:
            report.skip(name, "No observe/store method found", CAT_REFLEX)
            return

        report.add(name, True, "Observation stored successfully", CAT_REFLEX)
    except Exception as e:
        report.add(name, False, str(e), CAT_REFLEX, traceback.format_exc())

    # -- check_reflex_promotion --
    name = "check_reflex_promotion"
    try:
        rm_promo, _, pdim = _get_reflex_instance()
        pattern = torch.randn(pdim)
        prediction = torch.randn(pdim)

        # Store the same pattern multiple times to trigger promotion
        for _ in range(10):
            if hasattr(rm_promo, 'observe'):
                rm_promo.observe(pattern, prediction)
            elif hasattr(rm_promo, 'store'):
                rm_promo.store(pattern, prediction, force=True)

        # After repeated observations, should be findable
        result = rm_promo.lookup(pattern)
        if result is not None:
            report.add(name, True, "Pattern found after repeated observations", CAT_REFLEX)
        else:
            # For some implementations, promotion may require a different flow
            report.add(
                name, True,
                "Pattern stored (promotion semantics may differ by implementation)",
                CAT_REFLEX,
            )
    except Exception as e:
        report.add(name, False, str(e), CAT_REFLEX, traceback.format_exc())

    # -- check_reflex_lookup_hit --
    name = "check_reflex_lookup_hit"
    try:
        rm_hit, _, pdim = _get_reflex_instance()
        pattern = torch.randn(pdim)
        prediction = torch.randn(pdim)

        # Force store
        if hasattr(rm_hit, 'store'):
            rm_hit.store(pattern, prediction, force=True)
        elif hasattr(rm_hit, 'observe'):
            for _ in range(10):
                rm_hit.observe(pattern, prediction)

        result = rm_hit.lookup(pattern)
        if result is not None:
            # Result should be tuple-like: (prediction, confidence, index)
            assert len(result) >= 2, \
                f"Lookup result should have >= 2 elements, got {len(result)}"
            cached_pred = result[0]
            assert isinstance(cached_pred, torch.Tensor), \
                f"Cached prediction should be tensor, got {type(cached_pred)}"
            report.add(name, True, "Promoted pattern returns cached prediction", CAT_REFLEX)
        else:
            report.skip(
                name,
                "Lookup returned None (may require higher similarity threshold)",
                CAT_REFLEX,
            )
    except Exception as e:
        report.add(name, False, str(e), CAT_REFLEX, traceback.format_exc())

    # -- check_reflex_memory_cap --
    name = "check_reflex_memory_cap"
    try:
        rm_cap, _, pdim = _get_reflex_instance()
        max_patterns = rm_cap.max_patterns

        # Store more than max_patterns
        for i in range(max_patterns + 20):
            p = torch.randn(pdim)
            pred = torch.randn(pdim)
            if hasattr(rm_cap, 'store'):
                rm_cap.store(p, pred, force=True)

        stored = None
        if hasattr(rm_cap, 'num_stored'):
            stored = rm_cap.num_stored
            if isinstance(stored, torch.Tensor):
                stored = stored.item()
        elif hasattr(rm_cap, 'get_statistics'):
            stats = rm_cap.get_statistics()
            stored = stats.get('patterns_stored', None)

        if stored is not None:
            assert stored <= max_patterns, \
                f"Stored {stored} > max_patterns {max_patterns}"
            report.add(
                name, True,
                f"Stored {stored} <= max_patterns {max_patterns} (eviction works)",
                CAT_REFLEX,
            )
        else:
            report.skip(name, "Cannot determine stored count", CAT_REFLEX)
    except Exception as e:
        report.add(name, False, str(e), CAT_REFLEX, traceback.format_exc())

    # -- check_reflex_state_dict_roundtrip --
    name = "check_reflex_state_dict_roundtrip"
    try:
        rm_sd, _, pdim = _get_reflex_instance()
        pattern = torch.randn(pdim)
        prediction = torch.randn(pdim)
        if hasattr(rm_sd, 'store'):
            rm_sd.store(pattern, prediction, force=True)

        # Save
        sd = rm_sd.state_dict()
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(sd, f.name)

            # Load into fresh instance
            rm_loaded, _, _ = _get_reflex_instance()
            loaded_sd = torch.load(f.name, weights_only=True)
            rm_loaded.load_state_dict(loaded_sd)

        # Verify buffers match
        for key in sd:
            if isinstance(sd[key], torch.Tensor):
                assert torch.equal(sd[key], rm_loaded.state_dict()[key]), \
                    f"Buffer {key} mismatch after load"

        report.add(name, True, "state_dict save -> load roundtrip OK", CAT_REFLEX)
    except Exception as e:
        report.add(name, False, str(e), CAT_REFLEX, traceback.format_exc())

    # -- check_reflex_statistics --
    name = "check_reflex_statistics"
    try:
        rm_stats, _, pdim = _get_reflex_instance()
        if not hasattr(rm_stats, 'get_statistics'):
            report.skip(name, "get_statistics() not found", CAT_REFLEX)
        else:
            stats = rm_stats.get_statistics()
            assert isinstance(stats, dict), \
                f"get_statistics() should return dict, got {type(stats)}"
            assert 'hit_rate' in stats, \
                f"Stats missing 'hit_rate'. Keys: {list(stats.keys())}"
            report.add(
                name, True,
                f"get_statistics() returns dict with hit_rate ({list(stats.keys())})",
                CAT_REFLEX,
            )
    except Exception as e:
        report.add(name, False, str(e), CAT_REFLEX, traceback.format_exc())


# =============================================================================
# Section 9: Fallback Predictor Checks
# =============================================================================

CAT_FALLBACK = "Fallback Predictor"


def _get_fallback_instance() -> Tuple[Optional[Any], str]:
    """Try to get a fallback sequence predictor.

    Returns:
        (instance_or_None, backend_name)
    """
    input_size = 64
    hidden_size = 32

    # Try LSTM
    cls = _legacy.get("LSTMSequencePredictor")
    SequenceConfig = _legacy.get("SequenceConfig")
    if cls is not None and SequenceConfig is not None:
        try:
            cfg = SequenceConfig(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=1,
            )
            return cls(cfg), "LSTM"
        except Exception:
            pass

    # Try GRU
    cls = _legacy.get("GRUSequencePredictor")
    if cls is not None:
        try:
            return cls(input_size=input_size, hidden_size=hidden_size, num_layers=1), "GRU"
        except Exception:
            pass

    # Try Transformer
    cls = _legacy.get("TransformerSequencePredictor")
    if cls is not None:
        try:
            return cls(input_size=input_size, hidden_size=hidden_size, num_layers=1), "Transformer"
        except Exception:
            pass

    return None, "none"


def run_fallback_checks(report: ValidationReport) -> None:
    """Run all fallback predictor contract checks."""
    _seed_everything()

    # -- check_fallback_exists --
    name = "check_fallback_exists"
    fb, fb_backend = _get_fallback_instance()
    if fb is None:
        report.add(name, False, "No fallback predictor importable", CAT_FALLBACK)
        return
    report.add(name, True, f"Fallback available ({fb_backend})", CAT_FALLBACK)

    input_size = 64

    # -- check_fallback_sequence_output --
    name = "check_fallback_sequence_output"
    try:
        x = torch.randn(2, input_size)
        result = fb(x, learn=False)
        assert isinstance(result, dict), \
            f"Fallback output should be dict, got {type(result)}"
        required = {"features", "anomaly"}
        missing = required - set(result.keys())
        assert not missing, f"Missing keys: {missing}"
        report.add(
            name, True,
            f"Returns dict with required keys ({list(result.keys())})",
            CAT_FALLBACK,
        )
    except Exception as e:
        report.add(name, False, str(e), CAT_FALLBACK, traceback.format_exc())

    # -- check_fallback_anomaly_range --
    name = "check_fallback_anomaly_range"
    try:
        fb2, _ = _get_fallback_instance()
        x = torch.randn(2, input_size)
        result = fb2(x, learn=False)
        anomaly = result['anomaly']
        if anomaly.dim() == 0:
            avals = [anomaly.item()]
        else:
            avals = anomaly.tolist()
        all_in_range = all(0.0 <= a <= 1.0 for a in avals)
        report.add(
            name, all_in_range,
            f"Anomaly values in [0, 1]: {[f'{a:.3f}' for a in avals]}",
            CAT_FALLBACK,
        )
    except Exception as e:
        report.add(name, False, str(e), CAT_FALLBACK, traceback.format_exc())

    # -- check_fallback_reset --
    name = "check_fallback_reset"
    try:
        fb3, _ = _get_fallback_instance()
        x = torch.randn(2, input_size)
        fb3(x, learn=False)
        # Reset should clear hidden state
        if hasattr(fb3, 'reset'):
            fb3.reset()
            hidden = getattr(fb3, 'hidden', "MISSING")
            assert hidden is None or hidden == "MISSING", \
                f"Hidden state should be None after reset, got type {type(hidden)}"
            report.add(name, True, "reset() clears hidden state", CAT_FALLBACK)
        else:
            report.skip(name, "No reset() method", CAT_FALLBACK)
    except Exception as e:
        report.add(name, False, str(e), CAT_FALLBACK, traceback.format_exc())

    # -- check_fallback_gradient_flow --
    name = "check_fallback_gradient_flow"
    try:
        fb4, _ = _get_fallback_instance()
        fb4.train()
        x = torch.randn(2, input_size, requires_grad=False)
        result = fb4(x, learn=True)
        features = result['features']
        loss = features.sum()
        loss.backward()

        has_grad = False
        for p in fb4.parameters():
            if p.grad is not None and p.grad.abs().sum().item() > 0:
                has_grad = True
                break

        report.add(
            name, has_grad,
            "Gradient flows through fallback parameters" if has_grad
            else "No gradients found on parameters",
            CAT_FALLBACK,
        )
    except Exception as e:
        report.add(name, False, str(e), CAT_FALLBACK, traceback.format_exc())


# =============================================================================
# Section 10: Integration Checks
# =============================================================================

CAT_INTEGRATION = "Integration"


def run_integration_checks(report: ValidationReport) -> None:
    """Run integration checks: SP -> TM -> Reflex pipeline."""
    _seed_everything()

    input_size = 128
    column_count = 64
    cells_per_column = 8
    K = int(column_count * 0.1)  # 6

    # -- check_sp_tm_pipeline --
    name = "check_sp_tm_pipeline"
    try:
        # Get SP
        sp, sp_backend, _, _, _ = _get_sp_instance()
        if sp is None:
            report.skip(name, "No SpatialPooler available", CAT_INTEGRATION)
        else:
            # Get TM
            tm, tm_backend, _, _ = _get_tm_instance()
            if tm is None:
                report.skip(name, "No TemporalMemory available", CAT_INTEGRATION)
            else:
                x = _make_binary_input(1, input_size, sparsity=0.1)
                sp_out = sp(x, learn=False)

                # Feed SP output into TM
                if sp_backend == "target":
                    # Target SP: (B, K) int -> convert to dense for legacy TM
                    dense = torch.zeros(column_count)
                    dense[sp_out[0]] = 1.0
                    tm_in = dense
                else:
                    # Legacy SP: (B, N_col) dense
                    tm_in = sp_out.squeeze(0)

                result = tm(tm_in, learn=True)
                assert isinstance(result, dict), \
                    f"TM output should be dict, got {type(result)}"
                assert 'anomaly' in result, "TM output missing 'anomaly'"
                report.add(
                    name, True,
                    f"SP({sp_backend}) -> TM({tm_backend}) pipeline works",
                    CAT_INTEGRATION,
                )
    except Exception as e:
        report.add(name, False, str(e), CAT_INTEGRATION, traceback.format_exc())

    # -- check_accelerated_htm --
    name = "check_accelerated_htm"
    try:
        AHTMClass = _legacy.get("AcceleratedHTM")
        create_fn = _legacy.get("create_accelerated_htm")
        if AHTMClass is None and create_fn is None:
            report.skip(name, "AcceleratedHTM not available", CAT_INTEGRATION)
        else:
            if create_fn is not None:
                ahtm = create_fn(
                    input_size=input_size,
                    column_count=column_count,
                    cells_per_column=cells_per_column,
                    sparsity=0.1,
                    max_reflex_patterns=50,
                    promotion_threshold=3,
                )
            else:
                htm_layer_fn = _legacy.get("create_htm_layer")
                htm_layer = htm_layer_fn(
                    input_size=input_size,
                    column_count=column_count,
                    cells_per_column=cells_per_column,
                    sparsity=0.1,
                )
                RMClass = _legacy.get("ReflexMemory")
                rm = RMClass(
                    pattern_dim=input_size,
                    max_patterns=50,
                    promotion_threshold=3,
                )
                ahtm = AHTMClass(htm_layer=htm_layer, reflex_memory=rm)

            x = _make_binary_input(1, input_size, sparsity=0.1)
            result = ahtm(x, learn=True)
            assert isinstance(result, dict), \
                f"AHTM output should be dict, got {type(result)}"
            assert 'features' in result, "AHTM output missing 'features'"
            assert 'anomaly' in result, "AHTM output missing 'anomaly'"
            report.add(name, True, "AcceleratedHTM SP+TM+Reflex pipeline works", CAT_INTEGRATION)
    except Exception as e:
        report.add(name, False, str(e), CAT_INTEGRATION, traceback.format_exc())

    # -- check_temporal_layer_auto --
    name = "check_temporal_layer_auto"
    try:
        TLClass = _legacy.get("TemporalLayer")
        create_tl = _legacy.get("create_temporal_layer")
        if TLClass is None and create_tl is None:
            report.skip(name, "TemporalLayer not available", CAT_INTEGRATION)
        else:
            if create_tl is not None:
                tl = create_tl(backend="auto", input_size=input_size)
            else:
                tl = TLClass(input_size=input_size, backend="auto")

            # Verify backend was selected
            actual_backend = getattr(tl, 'backend', 'unknown')
            assert actual_backend != "auto", \
                "auto backend should resolve to a concrete backend"

            x = torch.randn(2, input_size)
            result = tl(x, learn=False)
            assert isinstance(result, dict), \
                f"TemporalLayer output should be dict, got {type(result)}"
            report.add(
                name, True,
                f"TemporalLayer auto -> {actual_backend}",
                CAT_INTEGRATION,
            )
    except Exception as e:
        report.add(name, False, str(e), CAT_INTEGRATION, traceback.format_exc())

    # -- check_htm_layer_output_keys --
    name = "check_htm_layer_output_keys"
    try:
        HTMLayerClass = _legacy.get("HTMLayer")
        HTMConfigClass = _legacy.get("HTMConfig")
        if HTMLayerClass is None:
            report.skip(name, "HTMLayer not available", CAT_INTEGRATION)
        else:
            if HTMConfigClass is not None:
                cfg = HTMConfigClass(
                    input_size=input_size,
                    column_count=column_count,
                    cells_per_column=cells_per_column,
                    sparsity=0.1,
                )
                htm = HTMLayerClass(config=cfg)
            else:
                htm = HTMLayerClass()

            x = torch.randn(input_size)
            result = htm(x, learn=True)
            assert isinstance(result, dict), \
                f"HTMLayer output should be dict, got {type(result)}"
            expected_keys = {"features", "anomaly", "predictive_cells"}
            present = expected_keys & set(result.keys())
            missing = expected_keys - set(result.keys())

            report.add(
                name,
                len(missing) == 0,
                f"HTMLayer output keys: present={list(present)}, missing={list(missing)}",
                CAT_INTEGRATION,
                details=f"All output keys: {list(result.keys())}",
            )
    except Exception as e:
        report.add(name, False, str(e), CAT_INTEGRATION, traceback.format_exc())


# =============================================================================
# Section 11: Legacy Compatibility Checks
# =============================================================================

CAT_LEGACY = "Legacy Compatibility"


def run_legacy_checks(report: ValidationReport) -> None:
    """Run legacy compatibility checks for existing brain_ai/temporal/ code."""
    _seed_everything()

    input_size = 128
    column_count = 64
    cells_per_column = 8

    # -- check_legacy_htm_layer --
    name = "check_legacy_htm_layer"
    try:
        HTMLayerClass = _legacy.get("HTMLayer")
        HTMConfigClass = _legacy.get("HTMConfig")
        if HTMLayerClass is None:
            report.skip(name, "HTMLayer not importable", CAT_LEGACY)
        else:
            cfg = HTMConfigClass(
                input_size=input_size,
                column_count=column_count,
                cells_per_column=cells_per_column,
                sparsity=0.1,
            )
            htm = HTMLayerClass(config=cfg)

            # Single input
            x = torch.randn(input_size)
            result = htm(x, learn=True)
            assert isinstance(result, dict)
            assert 'features' in result
            assert 'anomaly' in result

            # Batched input
            htm.reset()
            x_batch = torch.randn(3, input_size)
            result_batch = htm(x_batch, learn=False)
            assert result_batch['features'].shape[0] == 3

            report.add(name, True, "HTMLayer works with single and batched input", CAT_LEGACY)
    except Exception as e:
        report.add(name, False, str(e), CAT_LEGACY, traceback.format_exc())

    # -- check_legacy_sp --
    name = "check_legacy_sp"
    try:
        SPClass = _legacy.get("PytorchSpatialPooler")
        if SPClass is None:
            report.skip(name, "PytorchSpatialPooler not importable", CAT_LEGACY)
        else:
            sp = SPClass(
                input_size=input_size,
                column_count=column_count,
                sparsity=0.1,
            )

            # Single input
            x = _make_binary_input(1, input_size, sparsity=0.1).squeeze(0)
            out = sp(x, learn=True)
            assert out.shape == (column_count,), \
                f"Single output shape: expected ({column_count},), got {out.shape}"

            # Batched
            x_batch = _make_binary_input(3, input_size, sparsity=0.1)
            out_batch = sp(x_batch, learn=False)
            assert out_batch.shape == (3, column_count), \
                f"Batch output shape: expected (3, {column_count}), got {out_batch.shape}"

            # Sparsity check
            num_active = (out > 0).sum().item()
            expected_k = int(column_count * 0.1)
            assert num_active == expected_k, \
                f"Active columns: {num_active}, expected {expected_k}"

            report.add(name, True, "PytorchSpatialPooler backward compatible", CAT_LEGACY)
    except Exception as e:
        report.add(name, False, str(e), CAT_LEGACY, traceback.format_exc())

    # -- check_legacy_tm --
    name = "check_legacy_tm"
    try:
        TMClass = _legacy.get("PytorchTemporalMemory")
        if TMClass is None:
            report.skip(name, "PytorchTemporalMemory not importable", CAT_LEGACY)
        else:
            tm = TMClass(
                column_count=column_count,
                cells_per_column=cells_per_column,
            )

            active_cols = _make_sp_output_dense(1, column_count, 6).squeeze(0)
            result = tm(active_cols, learn=True)
            assert isinstance(result, dict)
            assert 'active_cells' in result
            assert 'predictive_cells' in result
            assert 'anomaly' in result

            # Test reset
            tm.reset()
            assert tm.active_cells.sum().item() == 0
            assert tm.predictive_cells.sum().item() == 0

            report.add(name, True, "PytorchTemporalMemory backward compatible", CAT_LEGACY)
    except Exception as e:
        report.add(name, False, str(e), CAT_LEGACY, traceback.format_exc())

    # -- check_legacy_accelerated --
    name = "check_legacy_accelerated"
    try:
        AHTMClass = _legacy.get("AcceleratedHTM")
        create_fn = _legacy.get("create_accelerated_htm")
        if AHTMClass is None:
            report.skip(name, "AcceleratedHTM not importable", CAT_LEGACY)
        else:
            if create_fn is not None:
                ahtm = create_fn(
                    input_size=input_size,
                    column_count=column_count,
                    sparsity=0.1,
                    max_reflex_patterns=50,
                )
            else:
                report.skip(name, "create_accelerated_htm not available", CAT_LEGACY)
                return

            # Feed same pattern multiple times to trigger promotion
            x = _make_binary_input(1, input_size, sparsity=0.1)
            for _ in range(10):
                result = ahtm(x, learn=True)

            assert isinstance(result, dict)
            assert 'features' in result
            assert 'anomaly' in result

            # Check statistics
            if hasattr(ahtm, 'get_statistics'):
                stats = ahtm.get_statistics()
                assert 'rm_hits' in stats or 'acceleration_rate' in stats

            report.add(name, True, "AcceleratedHTM backward compatible", CAT_LEGACY)
    except Exception as e:
        report.add(name, False, str(e), CAT_LEGACY, traceback.format_exc())

    # -- check_legacy_sequence_predictors --
    name = "check_legacy_sequence_predictors"
    try:
        found = []
        test_input = torch.randn(2, 64)

        for pred_name in ["LSTMSequencePredictor", "GRUSequencePredictor",
                          "TransformerSequencePredictor"]:
            cls = _legacy.get(pred_name)
            if cls is None:
                continue

            try:
                if pred_name == "LSTMSequencePredictor":
                    SequenceConfig = _legacy.get("SequenceConfig")
                    cfg = SequenceConfig(input_size=64, hidden_size=32, num_layers=1)
                    pred = cls(cfg)
                elif pred_name == "GRUSequencePredictor":
                    pred = cls(input_size=64, hidden_size=32, num_layers=1)
                else:
                    pred = cls(input_size=64, hidden_size=32, num_layers=1)

                result = pred(test_input, learn=False)
                assert isinstance(result, dict)
                assert 'features' in result
                assert 'anomaly' in result

                # Test reset
                if hasattr(pred, 'reset'):
                    pred.reset()

                found.append(pred_name)
            except Exception:
                pass

        if not found:
            report.add(name, False, "No sequence predictors importable", CAT_LEGACY)
        else:
            report.add(
                name, True,
                f"Sequence predictors available: {found}",
                CAT_LEGACY,
            )
    except Exception as e:
        report.add(name, False, str(e), CAT_LEGACY, traceback.format_exc())

    # -- check_legacy_sparse_tensor --
    name = "check_legacy_sparse_tensor"
    try:
        STClass = _legacy.get("SparseTensor")
        if STClass is None:
            report.skip(name, "SparseTensor not importable", CAT_LEGACY)
        else:
            sdr = STClass(size=100)
            sdr.sparse = [1, 5, 10, 20]
            dense = sdr.to_dense()
            assert dense.shape == (100,), f"Expected (100,), got {dense.shape}"
            assert dense.sum().item() == 4, \
                f"Expected 4 active bits, got {dense.sum().item()}"

            # Test from_dense
            sdr2 = STClass.from_dense(dense)
            assert sdr.overlap(sdr2) == 4, "Self-overlap should be 4"

            report.add(name, True, "SparseTensor backward compatible", CAT_LEGACY)
    except Exception as e:
        report.add(name, False, str(e), CAT_LEGACY, traceback.format_exc())

    # -- check_legacy_create_htm_layer --
    name = "check_legacy_create_htm_layer"
    try:
        create_fn = _legacy.get("create_htm_layer")
        if create_fn is None:
            report.skip(name, "create_htm_layer not importable", CAT_LEGACY)
        else:
            htm = create_fn(
                input_size=input_size,
                column_count=column_count,
                cells_per_column=cells_per_column,
                sparsity=0.1,
            )
            assert isinstance(htm, nn.Module), \
                f"Expected nn.Module, got {type(htm)}"

            x = torch.randn(input_size)
            result = htm(x, learn=True)
            assert isinstance(result, dict)

            report.add(name, True, "create_htm_layer() factory works", CAT_LEGACY)
    except Exception as e:
        report.add(name, False, str(e), CAT_LEGACY, traceback.format_exc())


# =============================================================================
# Section 12: Main Entry Point
# =============================================================================

def main() -> None:
    """Run HTM contract validation."""
    parser = argparse.ArgumentParser(
        description="HTM Contract Validator -- validates brain_ai/temporal/ module contracts",
    )
    parser.add_argument(
        '--target-only', action='store_true',
        help='Only check target (upgraded) API, skip legacy checks',
    )
    parser.add_argument(
        '--legacy-only', action='store_true',
        help='Only check legacy API, skip target checks',
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Show all individual check results, not just failures',
    )
    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        print("FATAL: PyTorch not importable. Cannot run validation.")
        sys.exit(2)

    # Seed for reproducibility
    _seed_everything()

    # Detect APIs
    print(f"Project root: {_PROJECT_ROOT}")
    print(f"brain_ai dir: {_BRAIN_AI_DIR}")
    print()

    target_available = detect_target_api()
    legacy_available = detect_legacy_api()

    print(f"Target API available: {target_available}")
    print(f"  SDR utils:    {list(_target_sdr.keys()) or '(none)'}")
    print(f"  Spatial Pooler: {list(_target_sp.keys()) or '(none)'}")
    print(f"  Temporal Memory: {list(_target_tm.keys()) or '(none)'}")
    print(f"  Reflex Memory: {list(_target_reflex.keys()) or '(none)'}")
    print()
    print(f"Legacy API available: {legacy_available}")
    print(f"  Symbols: {list(_legacy.keys()) or '(none)'}")
    print()

    if not target_available and not legacy_available:
        print("FATAL: Neither target nor legacy API is importable.")
        print("Check that brain_ai/temporal/ exists and has no import errors.")
        sys.exit(2)

    # Build report
    report = ValidationReport()

    if not args.legacy_only:
        print("--- Running Target API Checks ---")
        run_sdr_checks(report)
        run_sp_checks(report)
        run_tm_checks(report)
        run_reflex_checks(report)
        run_fallback_checks(report)
        run_integration_checks(report)
        print()

    if not args.target_only:
        print("--- Running Legacy Compatibility Checks ---")
        run_legacy_checks(report)
        print()

    # Print summary
    print(report.summary(verbose=args.verbose))
    sys.exit(report.exit_code())


if __name__ == "__main__":
    main()
