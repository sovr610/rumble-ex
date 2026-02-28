#!/usr/bin/env python3
"""
Phase Boundary Validation System
=================================
Validates artifacts produced at the boundary between consecutive training phases
in the brain-inspired AI pipeline.  Each of the seven phases (SNN Core, Modality
Encoders, HTM, Global Workspace, Active Inference, Reasoning, Meta-Learning)
produces a ``phase_boundary.pt`` file consumed by the next phase.  This module
prevents silent dimension / flag / schema mismatches that would otherwise lead
to cryptic runtime errors deep inside training loops.

Self-contained: no ``brain_ai`` imports.  All external types are represented as
plain dicts so the template can be used in isolation or vendored into the main
training orchestrator.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Phase names (for human-readable messages)
# ---------------------------------------------------------------------------
PHASE_NAMES: Dict[int, str] = {
    1: "SNN Core",
    2: "Modality Encoders",
    3: "HTM (Hierarchical Temporal Memory)",
    4: "Global Workspace",
    5: "Active Inference",
    6: "Reasoning (Dual-Process)",
    7: "Meta-Learning",
}

SCHEMA_VERSION = "1.0"

# ===================================================================
# BoundaryValidationResult
# ===================================================================

@dataclass
class BoundaryValidationResult:
    """Outcome of validating a single phase-boundary transition."""

    passed: bool
    hard_errors: List[str] = field(default_factory=list)
    soft_warnings: List[str] = field(default_factory=list)
    phase: int = 0
    boundary_path: Optional[str] = None
    validation_mode: str = "normal"
    details: Dict[str, Any] = field(default_factory=dict)

    # -- helpers ----------------------------------------------------------
    def add_error(self, msg: str) -> None:
        self.hard_errors.append(msg)
        self.passed = False

    def add_warning(self, msg: str) -> None:
        self.soft_warnings.append(msg)

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        phase_name = PHASE_NAMES.get(self.phase, f"Phase {self.phase}")
        lines = [
            f"=== Boundary Validation: {phase_name} (Phase {self.phase}) ===",
            f"Status       : {status}",
            f"Mode         : {self.validation_mode}",
            f"Boundary file: {self.boundary_path or 'N/A'}",
        ]
        if self.hard_errors:
            lines.append(f"Hard errors  : {len(self.hard_errors)}")
            for i, e in enumerate(self.hard_errors, 1):
                lines.append(f"  [{i}] {e}")
        if self.soft_warnings:
            lines.append(f"Soft warnings: {len(self.soft_warnings)}")
            for i, w in enumerate(self.soft_warnings, 1):
                lines.append(f"  [{i}] {w}")
        if self.details:
            lines.append("Details:")
            for k, v in self.details.items():
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


# ===================================================================
# PhaseContractSpec + PHASE_CONTRACTS
# ===================================================================

@dataclass
class PhaseContractSpec:
    """Describes what Phase *N* requires from the boundary artifact of Phase N-1."""

    phase: int
    required_boundary_from: int
    required_state_dict_prefixes: List[str]
    required_compatibility_fields: List[str]
    required_feature_flags: List[str]
    optional_state_dict_prefixes: List[str] = field(default_factory=list)
    dataset_must_match_phase: Optional[int] = None


# Phase 1 is the starting phase -- no boundary contract needed.
PHASE_CONTRACTS: Dict[int, PhaseContractSpec] = {
    2: PhaseContractSpec(
        phase=2,
        required_boundary_from=1,
        required_state_dict_prefixes=["snn."],
        required_compatibility_fields=["workspace_dim", "snn_hidden"],
        required_feature_flags=["use_snn"],
        optional_state_dict_prefixes=["neuromod."],
        dataset_must_match_phase=None,  # Phase 2 may use different data
    ),
    3: PhaseContractSpec(
        phase=3,
        required_boundary_from=2,
        required_state_dict_prefixes=["snn.", "encoder."],
        required_compatibility_fields=["workspace_dim", "snn_hidden", "vocab_size"],
        required_feature_flags=["use_snn"],
        optional_state_dict_prefixes=["neuromod.", "projection."],
        dataset_must_match_phase=None,
    ),
    4: PhaseContractSpec(
        phase=4,
        required_boundary_from=3,
        required_state_dict_prefixes=["snn.", "encoder.", "htm."],
        required_compatibility_fields=["workspace_dim", "snn_hidden", "htm_columns"],
        required_feature_flags=["use_snn", "use_htm"],
        optional_state_dict_prefixes=["neuromod.", "projection."],
        dataset_must_match_phase=None,
    ),
    5: PhaseContractSpec(
        phase=5,
        required_boundary_from=4,
        required_state_dict_prefixes=["snn.", "encoder.", "htm.", "workspace."],
        required_compatibility_fields=["workspace_dim", "snn_hidden", "htm_columns"],
        required_feature_flags=["use_snn", "use_htm", "use_workspace"],
        optional_state_dict_prefixes=["neuromod.", "projection.", "attention."],
        dataset_must_match_phase=4,
    ),
    6: PhaseContractSpec(
        phase=6,
        required_boundary_from=5,
        required_state_dict_prefixes=["snn.", "encoder.", "htm.", "workspace.", "decision."],
        required_compatibility_fields=["workspace_dim", "snn_hidden", "htm_columns"],
        required_feature_flags=["use_snn", "use_htm", "use_workspace"],
        optional_state_dict_prefixes=["neuromod.", "projection.", "attention.", "efe."],
        dataset_must_match_phase=5,
    ),
    7: PhaseContractSpec(
        phase=7,
        required_boundary_from=6,
        required_state_dict_prefixes=[
            "snn.", "encoder.", "htm.", "workspace.",
            "decision.", "reasoning.",
        ],
        required_compatibility_fields=[
            "workspace_dim", "snn_hidden", "htm_columns", "reasoning_dim",
        ],
        required_feature_flags=[
            "use_snn", "use_htm", "use_workspace", "use_symbolic",
        ],
        optional_state_dict_prefixes=[
            "neuromod.", "projection.", "attention.", "efe.", "system1.", "system2.",
        ],
        dataset_must_match_phase=6,
    ),
}


# ===================================================================
# BoundaryCompatibilityMatrix
# ===================================================================

class BoundaryCompatibilityMatrix:
    """Encodes which dimensions/constraints are shared across phase transitions."""

    # Each entry maps (from_phase, to_phase) -> list of compatibility field names.
    COMPATIBILITY_MATRIX: Dict[Tuple[int, int], List[str]] = {
        (1, 2): ["workspace_dim", "snn_hidden"],
        (2, 3): ["workspace_dim", "snn_hidden", "vocab_size"],
        (3, 4): ["workspace_dim", "snn_hidden", "htm_columns"],
        (4, 5): ["workspace_dim", "snn_hidden", "htm_columns"],
        (5, 6): ["workspace_dim", "snn_hidden", "htm_columns"],
        (6, 7): ["workspace_dim", "snn_hidden", "htm_columns", "reasoning_dim"],
    }

    @classmethod
    def get_required_checks(cls, from_phase: int, to_phase: int) -> List[str]:
        """Return the list of compatibility fields that must match between phases."""
        key = (from_phase, to_phase)
        if key not in cls.COMPATIBILITY_MATRIX:
            raise ValueError(
                f"No compatibility matrix entry for transition "
                f"Phase {from_phase} -> Phase {to_phase}"
            )
        return list(cls.COMPATIBILITY_MATRIX[key])


# ===================================================================
# DimensionChecker
# ===================================================================

class DimensionChecker:
    """Checks that critical dimensions recorded in a boundary artifact match
    the current run configuration."""

    @staticmethod
    def check_workspace_dim(
        boundary: Dict[str, Any], config: Dict[str, Any]
    ) -> Tuple[str, Any, Any, bool]:
        expected = config.get("workspace_dim")
        actual = boundary.get("compatibility", {}).get("workspace_dim")
        return ("workspace_dim", expected, actual, expected == actual)

    @staticmethod
    def check_vocab_size(
        boundary: Dict[str, Any], config: Dict[str, Any]
    ) -> Tuple[str, Any, Any, bool]:
        expected = config.get("vocab_size")
        actual = boundary.get("compatibility", {}).get("vocab_size")
        if expected is None and actual is None:
            return ("vocab_size", None, None, True)
        return ("vocab_size", expected, actual, expected == actual)

    @staticmethod
    def check_snn_neurons(
        boundary: Dict[str, Any], config: Dict[str, Any]
    ) -> Tuple[str, Any, Any, bool]:
        expected = config.get("snn_hidden")
        actual = boundary.get("compatibility", {}).get("snn_hidden")
        return ("snn_hidden", expected, actual, expected == actual)

    @staticmethod
    def check_htm_columns(
        boundary: Dict[str, Any], config: Dict[str, Any]
    ) -> Tuple[str, Any, Any, bool]:
        expected = config.get("htm_columns")
        actual = boundary.get("compatibility", {}).get("htm_columns")
        if expected is None and actual is None:
            return ("htm_columns", None, None, True)
        return ("htm_columns", expected, actual, expected == actual)

    @staticmethod
    def check_reasoning_dim(
        boundary: Dict[str, Any], config: Dict[str, Any]
    ) -> Tuple[str, Any, Any, bool]:
        expected = config.get("reasoning_dim")
        actual = boundary.get("compatibility", {}).get("reasoning_dim")
        if expected is None and actual is None:
            return ("reasoning_dim", None, None, True)
        return ("reasoning_dim", expected, actual, expected == actual)

    @classmethod
    def check_all(
        cls,
        boundary: Dict[str, Any],
        config: Dict[str, Any],
        phase: int,
    ) -> List[Tuple[str, Any, Any, bool]]:
        """Run all dimension checks relevant to the given target phase."""
        from_phase = phase - 1
        required_fields = BoundaryCompatibilityMatrix.get_required_checks(
            from_phase, phase
        )
        checker_map: Dict[str, Callable] = {
            "workspace_dim": cls.check_workspace_dim,
            "snn_hidden": cls.check_snn_neurons,
            "vocab_size": cls.check_vocab_size,
            "htm_columns": cls.check_htm_columns,
            "reasoning_dim": cls.check_reasoning_dim,
        }
        results: List[Tuple[str, Any, Any, bool]] = []
        for field_name in required_fields:
            fn = checker_map.get(field_name)
            if fn is not None:
                results.append(fn(boundary, config))
            else:
                results.append((field_name, "unknown", "unknown", True))
        return results


# ===================================================================
# FeatureFlagChecker
# ===================================================================

class FeatureFlagChecker:
    """Validates that feature flags are consistent across phase boundaries.

    Rule: if Phase N-1 was trained with ``use_X = True``, Phase N must also
    have ``use_X = True`` *unless* an explicit ``ablation_override`` is set.
    """

    # Mapping from phase -> which flags must be True (inherited from prior phases).
    _REQUIRED_FLAGS: Dict[int, List[str]] = {
        2: ["use_snn"],
        3: ["use_snn"],
        4: ["use_snn", "use_htm"],
        5: ["use_snn", "use_htm", "use_workspace"],
        6: ["use_snn", "use_htm", "use_workspace"],
        7: ["use_snn", "use_htm", "use_workspace", "use_symbolic"],
    }

    @classmethod
    def get_required_flags(cls, phase: int) -> List[str]:
        """Return which feature flags Phase *phase* requires from prior phases."""
        return list(cls._REQUIRED_FLAGS.get(phase, []))

    @classmethod
    def check_consistency(
        cls,
        boundary_flags: Dict[str, bool],
        current_flags: Dict[str, bool],
        phase: int,
        ablation_override: bool = False,
    ) -> List[Tuple[str, bool, bool, bool]]:
        """Verify flags are compatible.

        Returns list of ``(flag_name, boundary_val, current_val, passed)`` tuples.
        """
        results: List[Tuple[str, bool, bool, bool]] = []
        for flag_name in cls.get_required_flags(phase):
            b_val = boundary_flags.get(flag_name, False)
            c_val = current_flags.get(flag_name, False)
            # If boundary had it True, current must also be True (unless ablation).
            if b_val and not c_val and not ablation_override:
                results.append((flag_name, b_val, c_val, False))
            else:
                results.append((flag_name, b_val, c_val, True))
        return results


# ===================================================================
# BoundaryArtifactLocator
# ===================================================================

class BoundaryArtifactLocator:
    """Locates boundary artifacts and checkpoints on disk."""

    BOUNDARY_FILENAME = "phase_boundary.pt"
    CHECKPOINT_PREFIX = "ckpt_step"

    @classmethod
    def locate_boundary(cls, phase: int, run_dir: str) -> str:
        """Find ``phase_boundary.pt`` produced by Phase *phase - 1*.

        The expected location is ``<run_dir>/phase<N-1>/phase_boundary.pt``.
        """
        source_phase = phase - 1
        path = os.path.join(run_dir, f"phase{source_phase}", cls.BOUNDARY_FILENAME)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Boundary artifact for Phase {source_phase} not found at "
                f"'{path}'.  Did Phase {source_phase} "
                f"({PHASE_NAMES.get(source_phase, '?')}) complete successfully?"
            )
        return path

    @classmethod
    def locate_boundary_from_run(
        cls,
        source_run_id: str,
        source_phase: int,
        base_dir: str = "runs/",
    ) -> str:
        """Locate a boundary artifact from a different run (for ablation forking)."""
        path = os.path.join(
            base_dir, source_run_id, f"phase{source_phase}", cls.BOUNDARY_FILENAME
        )
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Boundary artifact from run '{source_run_id}', "
                f"Phase {source_phase} not found at '{path}'."
            )
        return path

    @classmethod
    def locate_latest_checkpoint(cls, phase: int, run_dir: str) -> str:
        """Find the most recent ``ckpt_step*.pt`` for Phase *phase* (for resume)."""
        phase_dir = os.path.join(run_dir, f"phase{phase}")
        if not os.path.isdir(phase_dir):
            raise FileNotFoundError(
                f"Phase directory '{phase_dir}' does not exist."
            )
        candidates = sorted(
            [
                f
                for f in os.listdir(phase_dir)
                if f.startswith(cls.CHECKPOINT_PREFIX) and f.endswith(".pt")
            ],
            key=lambda n: int("".join(filter(str.isdigit, n)) or "0"),
        )
        if not candidates:
            raise FileNotFoundError(
                f"No checkpoints matching '{cls.CHECKPOINT_PREFIX}*.pt' "
                f"found in '{phase_dir}'."
            )
        return os.path.join(phase_dir, candidates[-1])


# ===================================================================
# PhaseBoundaryForker
# ===================================================================

class PhaseBoundaryForker:
    """Copies or symlinks a boundary artifact into a new run directory for
    ablation forking experiments."""

    @staticmethod
    def fork(
        source_run_dir: str,
        source_phase: int,
        new_run_dir: str,
        symlink: bool = False,
    ) -> str:
        """Copy (or symlink) the boundary artifact into *new_run_dir*.

        Also writes a ``fork_manifest.json`` recording provenance.
        Returns the path to the copied/linked boundary file.
        """
        src_path = BoundaryArtifactLocator.locate_boundary(
            phase=source_phase + 1, run_dir=source_run_dir
        )
        dest_phase_dir = os.path.join(new_run_dir, f"phase{source_phase}")
        os.makedirs(dest_phase_dir, exist_ok=True)
        dest_path = os.path.join(
            dest_phase_dir, BoundaryArtifactLocator.BOUNDARY_FILENAME
        )

        if symlink:
            os.symlink(os.path.abspath(src_path), dest_path)
        else:
            shutil.copy2(src_path, dest_path)

        # Record provenance
        manifest = {
            "forked_from_run": source_run_dir,
            "forked_from_phase": source_phase,
            "source_path": src_path,
            "destination_path": dest_path,
            "method": "symlink" if symlink else "copy",
        }
        manifest_path = os.path.join(new_run_dir, "fork_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        return dest_path

    @staticmethod
    def validate_fork_compatibility(
        source_boundary: Dict[str, Any],
        new_config: Dict[str, Any],
        target_phase: int,
    ) -> BoundaryValidationResult:
        """Check that the forked boundary is dimensionally compatible with
        the new run's configuration."""
        result = BoundaryValidationResult(
            passed=True,
            phase=target_phase,
            validation_mode="fork",
        )
        dim_results = DimensionChecker.check_all(
            source_boundary, new_config, target_phase
        )
        for name, expected, actual, ok in dim_results:
            if not ok:
                result.add_error(
                    f"Fork dimension mismatch: {name} expected={expected}, "
                    f"found={actual} in source boundary."
                )
            else:
                result.details[f"dim_{name}"] = f"{actual} (OK)"
        return result


# ===================================================================
# PhaseBoundaryValidator
# ===================================================================

class PhaseBoundaryValidator:
    """Top-level validator orchestrating all checks for a phase transition."""

    def __init__(self, validation_mode: str = "normal") -> None:
        if validation_mode not in ("strict", "normal", "permissive"):
            raise ValueError(
                f"Unknown validation_mode '{validation_mode}'. "
                f"Must be 'strict', 'normal', or 'permissive'."
            )
        self.validation_mode = validation_mode

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def validate(
        self,
        phase: int,
        run_dir: str,
        config: Dict[str, Any],
    ) -> BoundaryValidationResult:
        """Validate that Phase *phase* can start given the available artifacts.

        Parameters
        ----------
        phase : int
            The target phase (2-7).  Phase 1 requires no validation.
        run_dir : str
            Root run directory containing ``phase<N>/`` sub-directories.
        config : dict
            The current run configuration dict with keys like ``workspace_dim``,
            ``use_snn``, etc.

        Returns
        -------
        BoundaryValidationResult
        """
        result = BoundaryValidationResult(
            passed=True,
            phase=phase,
            validation_mode=self.validation_mode,
        )

        # Phase 1 has no boundary requirement.
        if phase == 1:
            result.details["note"] = "Phase 1 is the starting phase; no boundary needed."
            return result

        if phase not in PHASE_CONTRACTS:
            result.add_error(f"No contract defined for Phase {phase}.")
            return result

        contract = PHASE_CONTRACTS[phase]

        # --- 1. Artifact existence ---
        boundary_path: Optional[str] = None
        try:
            boundary_path = self.validate_artifact_exists(phase, run_dir)
            result.boundary_path = boundary_path
        except FileNotFoundError as exc:
            result.add_error(str(exc))
            return result  # Cannot proceed without the file

        # --- 2. Load boundary dict ---
        try:
            import torch  # type: ignore
            boundary: Dict[str, Any] = torch.load(
                boundary_path, map_location="cpu", weights_only=False
            )
        except Exception as exc:
            result.add_error(f"Failed to load boundary artifact: {exc}")
            return result

        # --- 3. Schema version ---
        schema_errs = self.validate_schema_version(boundary)
        for e in schema_errs:
            if self.validation_mode == "strict":
                result.add_error(e)
            else:
                result.add_warning(e)

        # --- 4. State dict keys ---
        state_dict = boundary.get("state_dict", {})
        key_errs, key_warns = self.validate_state_dict_keys(
            boundary, contract.required_state_dict_prefixes
        )
        for e in key_errs:
            result.add_error(e)
        for w in key_warns:
            if self.validation_mode == "strict":
                result.add_error(w)
            else:
                result.add_warning(w)

        # --- 5. Dimensions ---
        dim_results = self.validate_dimensions(boundary, config, phase)
        for name, expected, actual, ok in dim_results:
            result.details[f"dim_{name}"] = {
                "expected": expected,
                "actual": actual,
                "passed": ok,
            }
            if not ok:
                msg = (
                    f"Dimension mismatch for '{name}': "
                    f"expected {expected}, got {actual}."
                )
                if self.validation_mode == "permissive":
                    result.add_warning(msg)
                else:
                    result.add_error(msg)

        # --- 6. Feature flags ---
        flag_results = self.validate_feature_flags(boundary, config, phase)
        for fname, b_val, c_val, ok in flag_results:
            result.details[f"flag_{fname}"] = {
                "boundary": b_val,
                "current": c_val,
                "passed": ok,
            }
            if not ok:
                msg = (
                    f"Feature flag inconsistency: '{fname}' was {b_val} in "
                    f"boundary but {c_val} in current config."
                )
                if self.validation_mode == "permissive":
                    result.add_warning(msg)
                else:
                    result.add_error(msg)

        # --- 7. Dataset identity (if required) ---
        if contract.dataset_must_match_phase is not None:
            ds_errs = self.validate_dataset_identity(
                boundary, config.get("datasets", {})
            )
            for e in ds_errs:
                if self.validation_mode == "strict":
                    result.add_error(e)
                else:
                    result.add_warning(e)

        # --- 8. Extra keys (strict mode) ---
        if self.validation_mode == "strict":
            known_prefixes = set(
                contract.required_state_dict_prefixes
                + contract.optional_state_dict_prefixes
            )
            for key in state_dict:
                prefix = key.split(".")[0] + "."
                if prefix not in known_prefixes:
                    result.add_error(
                        f"Strict mode: unexpected state_dict key prefix '{prefix}' "
                        f"(key='{key}')."
                    )

        return result

    # ------------------------------------------------------------------
    # Individual validators
    # ------------------------------------------------------------------
    def validate_artifact_exists(self, phase: int, run_dir: str) -> str:
        """Check that ``phase_boundary.pt`` exists for Phase *phase - 1*."""
        return BoundaryArtifactLocator.locate_boundary(phase, run_dir)

    @staticmethod
    def validate_schema_version(
        boundary_dict: Dict[str, Any],
    ) -> List[str]:
        """Check ``schema_version`` matches expected version."""
        errors: List[str] = []
        version = boundary_dict.get("schema_version")
        if version is None:
            errors.append("Boundary artifact missing 'schema_version' field.")
        elif str(version) != SCHEMA_VERSION:
            errors.append(
                f"Schema version mismatch: expected '{SCHEMA_VERSION}', "
                f"got '{version}'."
            )
        return errors

    @staticmethod
    def validate_state_dict_keys(
        boundary_dict: Dict[str, Any],
        expected_prefixes: List[str],
    ) -> Tuple[List[str], List[str]]:
        """Check that required key prefixes exist in the state_dict.

        Returns ``(hard_errors, soft_warnings)``.
        """
        state_dict = boundary_dict.get("state_dict", {})
        errors: List[str] = []
        warnings: List[str] = []

        if not state_dict:
            errors.append("Boundary artifact has empty or missing 'state_dict'.")
            return errors, warnings

        existing_prefixes = {k.split(".")[0] + "." for k in state_dict}

        for prefix in expected_prefixes:
            if prefix not in existing_prefixes:
                errors.append(
                    f"Required state_dict prefix '{prefix}' not found. "
                    f"Available prefixes: {sorted(existing_prefixes)}."
                )

        # Warn about extra prefixes not in expected list
        extra = existing_prefixes - set(expected_prefixes)
        if extra:
            warnings.append(
                f"Extra state_dict prefixes found (not required): {sorted(extra)}."
            )

        return errors, warnings

    def validate_dimensions(
        self,
        boundary_dict: Dict[str, Any],
        config: Dict[str, Any],
        phase: int,
    ) -> List[Tuple[str, Any, Any, bool]]:
        """Check workspace_dim, vocab_size, etc. against config."""
        return DimensionChecker.check_all(boundary_dict, config, phase)

    def validate_feature_flags(
        self,
        boundary_dict: Dict[str, Any],
        config: Dict[str, Any],
        phase: int,
    ) -> List[Tuple[str, bool, bool, bool]]:
        """Check that enabled modules are consistent across boundary."""
        boundary_flags = boundary_dict.get("feature_flags", {})
        current_flags = {
            k: v for k, v in config.items() if k.startswith("use_")
        }
        ablation = config.get("ablation_override", False)
        return FeatureFlagChecker.check_consistency(
            boundary_flags, current_flags, phase, ablation_override=ablation
        )

    @staticmethod
    def validate_dataset_identity(
        boundary_dict: Dict[str, Any],
        current_datasets: Dict[str, Any],
    ) -> List[str]:
        """Cross-phase dataset fingerprint check.

        Compares a simple hash of dataset names/sizes recorded in the boundary
        with the current run's datasets.
        """
        errors: List[str] = []
        boundary_ds = boundary_dict.get("dataset_fingerprint")
        if boundary_ds is None:
            errors.append(
                "Boundary artifact missing 'dataset_fingerprint'; "
                "cannot verify dataset continuity."
            )
            return errors

        current_fp = _compute_dataset_fingerprint(current_datasets)
        if boundary_ds != current_fp:
            errors.append(
                f"Dataset fingerprint mismatch: boundary='{boundary_ds}', "
                f"current='{current_fp}'.  Datasets may have changed between phases."
            )
        return errors


# ===================================================================
# Helpers
# ===================================================================

def _compute_dataset_fingerprint(datasets: Dict[str, Any]) -> str:
    """Deterministic hash of dataset metadata for identity checks."""
    canonical = json.dumps(datasets, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def make_mock_boundary(
    phase: int,
    config: Dict[str, Any],
    state_dict_prefixes: Optional[List[str]] = None,
    schema_version: str = SCHEMA_VERSION,
    include_dataset_fp: bool = True,
    datasets: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a mock boundary artifact dict for testing.

    This helper constructs a well-formed boundary dict that would normally be
    produced by ``torch.save()`` at the end of a training phase.
    """
    if state_dict_prefixes is None:
        # A boundary produced at the end of Phase N should contain all
        # state_dict prefixes that Phase N+1 expects.  We collect required
        # prefixes from all contracts up to and including Phase N+1.
        prefixes: List[str] = []
        for p in range(2, phase + 2):
            if p in PHASE_CONTRACTS:
                for pfx in PHASE_CONTRACTS[p].required_state_dict_prefixes:
                    if pfx not in prefixes:
                        prefixes.append(pfx)
        if not prefixes:
            prefixes = ["snn."]
        state_dict_prefixes = prefixes

    # Build fake state_dict with one key per prefix
    state_dict: Dict[str, str] = {}
    for pfx in state_dict_prefixes:
        state_dict[f"{pfx}weight"] = "tensor_placeholder"
        state_dict[f"{pfx}bias"] = "tensor_placeholder"

    feature_flags = {
        k: v for k, v in config.items() if k.startswith("use_")
    }

    compatibility = {
        k: v
        for k, v in config.items()
        if k in (
            "workspace_dim", "snn_hidden", "vocab_size",
            "htm_columns", "reasoning_dim",
        )
    }

    boundary: Dict[str, Any] = {
        "schema_version": schema_version,
        "phase": phase,
        "state_dict": state_dict,
        "compatibility": compatibility,
        "feature_flags": feature_flags,
    }

    if include_dataset_fp and datasets:
        boundary["dataset_fingerprint"] = _compute_dataset_fingerprint(datasets)

    return boundary


def make_mock_config(
    workspace_dim: int = 4096,
    snn_hidden: int = 2048,
    vocab_size: int = 32000,
    htm_columns: int = 2048,
    reasoning_dim: int = 1024,
    use_snn: bool = True,
    use_htm: bool = True,
    use_workspace: bool = True,
    use_symbolic: bool = True,
    use_meta: bool = True,
    use_engram: bool = True,
    ablation_override: bool = False,
    datasets: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a mock config dict for testing."""
    cfg: Dict[str, Any] = {
        "workspace_dim": workspace_dim,
        "snn_hidden": snn_hidden,
        "vocab_size": vocab_size,
        "htm_columns": htm_columns,
        "reasoning_dim": reasoning_dim,
        "use_snn": use_snn,
        "use_htm": use_htm,
        "use_workspace": use_workspace,
        "use_symbolic": use_symbolic,
        "use_meta": use_meta,
        "use_engram": use_engram,
        "ablation_override": ablation_override,
    }
    if datasets:
        cfg["datasets"] = datasets
    return cfg


# ===================================================================
# Self-Test Block
# ===================================================================

if __name__ == "__main__":
    import sys

    try:
        import torch  # type: ignore
        _has_torch = True
    except ImportError:
        _has_torch = False
        print("WARNING: torch not installed; using JSON-based mock for save/load.")

    passed_count = 0
    failed_count = 0
    test_names: List[str] = []

    def _assert(condition: bool, msg: str) -> None:
        global passed_count, failed_count
        test_name = test_names[-1] if test_names else "?"
        if condition:
            passed_count += 1
            print(f"  PASS: {msg}")
        else:
            failed_count += 1
            print(f"  FAIL: {msg}  [in {test_name}]")

    def begin_test(name: str) -> None:
        test_names.append(name)
        print(f"\n--- {name} ---")

    # Helper to save/load boundary with or without torch
    def _save_boundary(boundary: Dict[str, Any], path: str) -> None:
        if _has_torch:
            torch.save(boundary, path)
        else:
            with open(path, "w") as f:
                json.dump(boundary, f, default=str)

    def _load_boundary(path: str) -> Dict[str, Any]:
        if _has_torch:
            return torch.load(path, map_location="cpu", weights_only=False)
        else:
            with open(path) as f:
                return json.load(f)

    # Monkey-patch torch.load/save if torch is unavailable so validator works
    if not _has_torch:
        import types as _types

        class _MockTorch:
            @staticmethod
            def save(obj: Any, path: str) -> None:
                with open(path, "w") as f:
                    json.dump(obj, f, default=str)

            @staticmethod
            def load(
                path: str, map_location: Any = None, weights_only: bool = False
            ) -> Any:
                with open(path) as f:
                    return json.load(f)

        sys.modules["torch"] = _MockTorch()  # type: ignore
        torch = sys.modules["torch"]

    # ------------------------------------------------------------------
    # Test 1: Validation passes with correct boundary artifact
    # ------------------------------------------------------------------
    begin_test("test_validation_passes_correct_boundary")
    with tempfile.TemporaryDirectory() as tmpdir:
        config = make_mock_config()
        boundary = make_mock_boundary(phase=1, config=config)
        phase1_dir = os.path.join(tmpdir, "phase1")
        os.makedirs(phase1_dir)
        _save_boundary(boundary, os.path.join(phase1_dir, "phase_boundary.pt"))

        validator = PhaseBoundaryValidator(validation_mode="normal")
        result = validator.validate(phase=2, run_dir=tmpdir, config=config)
        _assert(result.passed, "Phase 2 validation should pass with correct boundary")
        _assert(len(result.hard_errors) == 0, "No hard errors expected")

    # ------------------------------------------------------------------
    # Test 2: Validation fails on missing boundary file
    # ------------------------------------------------------------------
    begin_test("test_validation_fails_missing_boundary")
    with tempfile.TemporaryDirectory() as tmpdir:
        config = make_mock_config()
        validator = PhaseBoundaryValidator()
        result = validator.validate(phase=2, run_dir=tmpdir, config=config)
        _assert(not result.passed, "Should fail when boundary file is missing")
        _assert(len(result.hard_errors) > 0, "Should have at least one hard error")
        _assert("not found" in result.hard_errors[0].lower(), "Error should mention file not found")

    # ------------------------------------------------------------------
    # Test 3: Schema version mismatch detection
    # ------------------------------------------------------------------
    begin_test("test_schema_version_mismatch")
    with tempfile.TemporaryDirectory() as tmpdir:
        config = make_mock_config()
        boundary = make_mock_boundary(phase=1, config=config, schema_version="0.1")
        phase1_dir = os.path.join(tmpdir, "phase1")
        os.makedirs(phase1_dir)
        _save_boundary(boundary, os.path.join(phase1_dir, "phase_boundary.pt"))

        # In strict mode, schema mismatch is a hard error
        validator = PhaseBoundaryValidator(validation_mode="strict")
        result = validator.validate(phase=2, run_dir=tmpdir, config=config)
        _assert(not result.passed, "Strict mode should fail on schema mismatch")
        schema_errors = [e for e in result.hard_errors if "schema" in e.lower() or "version" in e.lower()]
        _assert(len(schema_errors) > 0, "Should detect schema version mismatch")

        # In normal mode, schema mismatch is a soft warning
        validator_normal = PhaseBoundaryValidator(validation_mode="normal")
        result_normal = validator_normal.validate(phase=2, run_dir=tmpdir, config=config)
        _assert(result_normal.passed, "Normal mode should pass with schema mismatch (soft warning)")
        _assert(len(result_normal.soft_warnings) > 0, "Should have soft warning for schema mismatch")

    # ------------------------------------------------------------------
    # Test 4: Dimension mismatch detection (wrong workspace_dim)
    # ------------------------------------------------------------------
    begin_test("test_dimension_mismatch_workspace_dim")
    with tempfile.TemporaryDirectory() as tmpdir:
        config_boundary = make_mock_config(workspace_dim=4096)
        boundary = make_mock_boundary(phase=1, config=config_boundary)
        phase1_dir = os.path.join(tmpdir, "phase1")
        os.makedirs(phase1_dir)
        _save_boundary(boundary, os.path.join(phase1_dir, "phase_boundary.pt"))

        config_current = make_mock_config(workspace_dim=2048)  # Different!
        validator = PhaseBoundaryValidator(validation_mode="normal")
        result = validator.validate(phase=2, run_dir=tmpdir, config=config_current)
        _assert(not result.passed, "Should fail on workspace_dim mismatch")
        dim_errors = [e for e in result.hard_errors if "workspace_dim" in e]
        _assert(len(dim_errors) > 0, "Should mention workspace_dim in error")

    # ------------------------------------------------------------------
    # Test 5: Feature flag inconsistency detection
    # ------------------------------------------------------------------
    begin_test("test_feature_flag_inconsistency")
    with tempfile.TemporaryDirectory() as tmpdir:
        config_boundary = make_mock_config(use_snn=True)
        boundary = make_mock_boundary(phase=1, config=config_boundary)
        phase1_dir = os.path.join(tmpdir, "phase1")
        os.makedirs(phase1_dir)
        _save_boundary(boundary, os.path.join(phase1_dir, "phase_boundary.pt"))

        config_current = make_mock_config(use_snn=False)  # Inconsistent!
        validator = PhaseBoundaryValidator(validation_mode="normal")
        result = validator.validate(phase=2, run_dir=tmpdir, config=config_current)
        _assert(not result.passed, "Should fail when use_snn was True but now False")
        flag_errors = [e for e in result.hard_errors if "use_snn" in e]
        _assert(len(flag_errors) > 0, "Should mention use_snn flag in error")

    # ------------------------------------------------------------------
    # Test 6: Feature flag with ablation override
    # ------------------------------------------------------------------
    begin_test("test_feature_flag_ablation_override")
    with tempfile.TemporaryDirectory() as tmpdir:
        config_boundary = make_mock_config(use_snn=True)
        boundary = make_mock_boundary(phase=1, config=config_boundary)
        phase1_dir = os.path.join(tmpdir, "phase1")
        os.makedirs(phase1_dir)
        _save_boundary(boundary, os.path.join(phase1_dir, "phase_boundary.pt"))

        config_current = make_mock_config(use_snn=False, ablation_override=True)
        validator = PhaseBoundaryValidator(validation_mode="normal")
        result = validator.validate(phase=2, run_dir=tmpdir, config=config_current)
        _assert(result.passed, "Should pass when ablation_override=True allows flag change")

    # ------------------------------------------------------------------
    # Test 7: Per-phase contracts (Phase 2 expects SNN keys)
    # ------------------------------------------------------------------
    begin_test("test_phase2_expects_snn_keys")
    with tempfile.TemporaryDirectory() as tmpdir:
        config = make_mock_config()
        # Create boundary WITHOUT snn. prefix
        boundary = make_mock_boundary(
            phase=1, config=config, state_dict_prefixes=["encoder."]
        )
        phase1_dir = os.path.join(tmpdir, "phase1")
        os.makedirs(phase1_dir)
        _save_boundary(boundary, os.path.join(phase1_dir, "phase_boundary.pt"))

        validator = PhaseBoundaryValidator(validation_mode="normal")
        result = validator.validate(phase=2, run_dir=tmpdir, config=config)
        _assert(not result.passed, "Phase 2 should fail without snn. prefix keys")

    # ------------------------------------------------------------------
    # Test 8: Per-phase contracts (Phase 4 expects HTM keys)
    # ------------------------------------------------------------------
    begin_test("test_phase4_expects_htm_keys")
    with tempfile.TemporaryDirectory() as tmpdir:
        config = make_mock_config()
        # Phase 3 boundary should have snn., encoder., htm.
        # Missing htm. prefix
        boundary = make_mock_boundary(
            phase=3, config=config, state_dict_prefixes=["snn.", "encoder."]
        )
        phase3_dir = os.path.join(tmpdir, "phase3")
        os.makedirs(phase3_dir)
        _save_boundary(boundary, os.path.join(phase3_dir, "phase_boundary.pt"))

        validator = PhaseBoundaryValidator(validation_mode="normal")
        result = validator.validate(phase=4, run_dir=tmpdir, config=config)
        _assert(not result.passed, "Phase 4 should fail without htm. prefix keys")
        htm_errors = [e for e in result.hard_errors if "htm." in e]
        _assert(len(htm_errors) > 0, "Should mention htm. prefix in error")

    # ------------------------------------------------------------------
    # Test 9: Strict mode fails on extra keys
    # ------------------------------------------------------------------
    begin_test("test_strict_mode_extra_keys")
    with tempfile.TemporaryDirectory() as tmpdir:
        config = make_mock_config()
        boundary = make_mock_boundary(
            phase=1, config=config, state_dict_prefixes=["snn.", "unknown_module."]
        )
        phase1_dir = os.path.join(tmpdir, "phase1")
        os.makedirs(phase1_dir)
        _save_boundary(boundary, os.path.join(phase1_dir, "phase_boundary.pt"))

        validator = PhaseBoundaryValidator(validation_mode="strict")
        result = validator.validate(phase=2, run_dir=tmpdir, config=config)
        _assert(not result.passed, "Strict mode should fail on unknown prefix")
        extra_errors = [e for e in result.hard_errors if "unknown_module." in e]
        _assert(len(extra_errors) > 0, "Should flag unknown_module. prefix in strict mode")

    # ------------------------------------------------------------------
    # Test 10: Permissive mode warns instead of failing on dimension mismatch
    # ------------------------------------------------------------------
    begin_test("test_permissive_mode_dimension_warning")
    with tempfile.TemporaryDirectory() as tmpdir:
        config_boundary = make_mock_config(workspace_dim=4096)
        boundary = make_mock_boundary(phase=1, config=config_boundary)
        phase1_dir = os.path.join(tmpdir, "phase1")
        os.makedirs(phase1_dir)
        _save_boundary(boundary, os.path.join(phase1_dir, "phase_boundary.pt"))

        config_current = make_mock_config(workspace_dim=2048)
        validator = PhaseBoundaryValidator(validation_mode="permissive")
        result = validator.validate(phase=2, run_dir=tmpdir, config=config_current)
        _assert(result.passed, "Permissive mode should not fail on dimension mismatch")
        dim_warnings = [w for w in result.soft_warnings if "workspace_dim" in w]
        _assert(len(dim_warnings) > 0, "Should have a soft warning for workspace_dim mismatch")

    # ------------------------------------------------------------------
    # Test 11: BoundaryArtifactLocator with temp directories
    # ------------------------------------------------------------------
    begin_test("test_boundary_artifact_locator")
    with tempfile.TemporaryDirectory() as tmpdir:
        phase2_dir = os.path.join(tmpdir, "phase2")
        os.makedirs(phase2_dir)
        boundary_path = os.path.join(phase2_dir, "phase_boundary.pt")
        _save_boundary({"test": True}, boundary_path)

        found = BoundaryArtifactLocator.locate_boundary(phase=3, run_dir=tmpdir)
        _assert(found == boundary_path, "Should find boundary at expected path")

        # Test missing boundary raises
        try:
            BoundaryArtifactLocator.locate_boundary(phase=5, run_dir=tmpdir)
            _assert(False, "Should raise FileNotFoundError for missing boundary")
        except FileNotFoundError:
            _assert(True, "Correctly raises FileNotFoundError for missing boundary")

    # ------------------------------------------------------------------
    # Test 12: BoundaryArtifactLocator - locate latest checkpoint
    # ------------------------------------------------------------------
    begin_test("test_locate_latest_checkpoint")
    with tempfile.TemporaryDirectory() as tmpdir:
        phase3_dir = os.path.join(tmpdir, "phase3")
        os.makedirs(phase3_dir)
        # Create several checkpoints
        for step in [100, 500, 1000, 2000]:
            ckpt_path = os.path.join(phase3_dir, f"ckpt_step{step}.pt")
            _save_boundary({"step": step}, ckpt_path)

        latest = BoundaryArtifactLocator.locate_latest_checkpoint(phase=3, run_dir=tmpdir)
        _assert(latest.endswith("ckpt_step2000.pt"), "Should find the latest checkpoint (step 2000)")

        # No checkpoints
        empty_phase = os.path.join(tmpdir, "phase4")
        os.makedirs(empty_phase)
        try:
            BoundaryArtifactLocator.locate_latest_checkpoint(phase=4, run_dir=tmpdir)
            _assert(False, "Should raise FileNotFoundError when no checkpoints exist")
        except FileNotFoundError:
            _assert(True, "Correctly raises FileNotFoundError for no checkpoints")

    # ------------------------------------------------------------------
    # Test 13: BoundaryArtifactLocator - locate from another run
    # ------------------------------------------------------------------
    begin_test("test_locate_boundary_from_run")
    with tempfile.TemporaryDirectory() as base_dir:
        run_dir = os.path.join(base_dir, "run_abc123", "phase2")
        os.makedirs(run_dir)
        _save_boundary({"test": True}, os.path.join(run_dir, "phase_boundary.pt"))

        found = BoundaryArtifactLocator.locate_boundary_from_run(
            source_run_id="run_abc123",
            source_phase=2,
            base_dir=base_dir,
        )
        _assert(os.path.isfile(found), "Should locate boundary from another run")

        try:
            BoundaryArtifactLocator.locate_boundary_from_run(
                source_run_id="nonexistent_run",
                source_phase=2,
                base_dir=base_dir,
            )
            _assert(False, "Should raise FileNotFoundError for nonexistent run")
        except FileNotFoundError:
            _assert(True, "Correctly raises FileNotFoundError for nonexistent run")

    # ------------------------------------------------------------------
    # Test 14: DimensionChecker with matching configs
    # ------------------------------------------------------------------
    begin_test("test_dimension_checker_matching")
    config = make_mock_config()
    boundary = make_mock_boundary(phase=1, config=config)
    results = DimensionChecker.check_all(boundary, config, phase=2)
    all_passed = all(r[3] for r in results)
    _assert(all_passed, "All dimension checks should pass with matching config")
    _assert(len(results) > 0, "Should have at least one dimension check")

    # ------------------------------------------------------------------
    # Test 15: DimensionChecker with mismatching configs
    # ------------------------------------------------------------------
    begin_test("test_dimension_checker_mismatching")
    config_original = make_mock_config(workspace_dim=4096, snn_hidden=2048)
    boundary = make_mock_boundary(phase=1, config=config_original)
    config_different = make_mock_config(workspace_dim=1024, snn_hidden=512)
    results = DimensionChecker.check_all(boundary, config_different, phase=2)
    any_failed = any(not r[3] for r in results)
    _assert(any_failed, "Should detect dimension mismatches")
    failed_names = [r[0] for r in results if not r[3]]
    _assert("workspace_dim" in failed_names, "Should detect workspace_dim mismatch")
    _assert("snn_hidden" in failed_names, "Should detect snn_hidden mismatch")

    # ------------------------------------------------------------------
    # Test 16: DimensionChecker for Phase 7 (reasoning_dim)
    # ------------------------------------------------------------------
    begin_test("test_dimension_checker_phase7_reasoning_dim")
    config = make_mock_config(reasoning_dim=1024)
    boundary = make_mock_boundary(phase=6, config=config)
    results = DimensionChecker.check_all(boundary, config, phase=7)
    reasoning_results = [r for r in results if r[0] == "reasoning_dim"]
    _assert(len(reasoning_results) == 1, "Phase 7 should check reasoning_dim")
    _assert(reasoning_results[0][3], "reasoning_dim should match")

    config_bad = make_mock_config(reasoning_dim=512)
    results_bad = DimensionChecker.check_all(boundary, config_bad, phase=7)
    reasoning_bad = [r for r in results_bad if r[0] == "reasoning_dim"]
    _assert(not reasoning_bad[0][3], "Should detect reasoning_dim mismatch")

    # ------------------------------------------------------------------
    # Test 17: FeatureFlagChecker with consistent flags
    # ------------------------------------------------------------------
    begin_test("test_feature_flag_checker_consistent")
    boundary_flags = {"use_snn": True, "use_htm": True}
    current_flags = {"use_snn": True, "use_htm": True, "use_workspace": True}
    results = FeatureFlagChecker.check_consistency(
        boundary_flags, current_flags, phase=4
    )
    all_ok = all(r[3] for r in results)
    _assert(all_ok, "All flags should be consistent")

    # ------------------------------------------------------------------
    # Test 18: FeatureFlagChecker with inconsistent flags
    # ------------------------------------------------------------------
    begin_test("test_feature_flag_checker_inconsistent")
    boundary_flags = {"use_snn": True, "use_htm": True}
    current_flags = {"use_snn": True, "use_htm": False}  # HTM turned off!
    results = FeatureFlagChecker.check_consistency(
        boundary_flags, current_flags, phase=4
    )
    failed_flags = [r for r in results if not r[3]]
    _assert(len(failed_flags) > 0, "Should detect use_htm inconsistency")
    _assert(failed_flags[0][0] == "use_htm", "Failed flag should be use_htm")

    # ------------------------------------------------------------------
    # Test 19: FeatureFlagChecker with ablation override
    # ------------------------------------------------------------------
    begin_test("test_feature_flag_checker_ablation")
    boundary_flags = {"use_snn": True, "use_htm": True}
    current_flags = {"use_snn": True, "use_htm": False}
    results = FeatureFlagChecker.check_consistency(
        boundary_flags, current_flags, phase=4, ablation_override=True
    )
    all_ok = all(r[3] for r in results)
    _assert(all_ok, "All flags should pass with ablation_override=True")

    # ------------------------------------------------------------------
    # Test 20: FeatureFlagChecker.get_required_flags
    # ------------------------------------------------------------------
    begin_test("test_get_required_flags")
    flags_p2 = FeatureFlagChecker.get_required_flags(2)
    _assert("use_snn" in flags_p2, "Phase 2 should require use_snn")
    flags_p7 = FeatureFlagChecker.get_required_flags(7)
    _assert("use_symbolic" in flags_p7, "Phase 7 should require use_symbolic")
    _assert(len(flags_p7) == 4, "Phase 7 should require 4 flags")

    # ------------------------------------------------------------------
    # Test 21: PhaseBoundaryForker
    # ------------------------------------------------------------------
    begin_test("test_phase_boundary_forker")
    with tempfile.TemporaryDirectory() as source_dir:
        with tempfile.TemporaryDirectory() as new_dir:
            config = make_mock_config()
            boundary = make_mock_boundary(phase=2, config=config)
            phase2_dir = os.path.join(source_dir, "phase2")
            os.makedirs(phase2_dir)
            _save_boundary(boundary, os.path.join(phase2_dir, "phase_boundary.pt"))

            forked_path = PhaseBoundaryForker.fork(
                source_run_dir=source_dir, source_phase=2, new_run_dir=new_dir
            )
            _assert(os.path.isfile(forked_path), "Forked boundary should exist")

            manifest_path = os.path.join(new_dir, "fork_manifest.json")
            _assert(os.path.isfile(manifest_path), "Fork manifest should be written")
            with open(manifest_path) as f:
                manifest = json.load(f)
            _assert(manifest["forked_from_phase"] == 2, "Manifest should record source phase")

    # ------------------------------------------------------------------
    # Test 22: PhaseBoundaryForker - validate fork compatibility
    # ------------------------------------------------------------------
    begin_test("test_forker_validate_compatibility")
    config_source = make_mock_config(workspace_dim=4096)
    boundary = make_mock_boundary(phase=2, config=config_source)

    # Compatible config
    config_compat = make_mock_config(workspace_dim=4096)
    result = PhaseBoundaryForker.validate_fork_compatibility(
        boundary, config_compat, target_phase=3
    )
    _assert(result.passed, "Fork should be compatible with matching dimensions")

    # Incompatible config
    config_incompat = make_mock_config(workspace_dim=1024)
    result_bad = PhaseBoundaryForker.validate_fork_compatibility(
        boundary, config_incompat, target_phase=3
    )
    _assert(not result_bad.passed, "Fork should fail with mismatching dimensions")

    # ------------------------------------------------------------------
    # Test 23: BoundaryValidationResult summary formatting
    # ------------------------------------------------------------------
    begin_test("test_validation_result_summary")
    result = BoundaryValidationResult(
        passed=False,
        hard_errors=["Missing snn. prefix", "Dimension mismatch"],
        soft_warnings=["Extra prefix found"],
        phase=3,
        boundary_path="/tmp/phase2/phase_boundary.pt",
        validation_mode="strict",
        details={"dim_workspace_dim": {"expected": 4096, "actual": 2048}},
    )
    summary = result.summary()
    _assert("FAILED" in summary, "Summary should show FAILED status")
    _assert("Phase 3" in summary, "Summary should show phase number")
    _assert("strict" in summary, "Summary should show validation mode")
    _assert("Missing snn." in summary, "Summary should include errors")
    _assert("Extra prefix" in summary, "Summary should include warnings")
    _assert("workspace_dim" in summary, "Summary should include details")

    # ------------------------------------------------------------------
    # Test 24: Phase 1 requires no validation
    # ------------------------------------------------------------------
    begin_test("test_phase1_no_validation_needed")
    with tempfile.TemporaryDirectory() as tmpdir:
        config = make_mock_config()
        validator = PhaseBoundaryValidator()
        result = validator.validate(phase=1, run_dir=tmpdir, config=config)
        _assert(result.passed, "Phase 1 should always pass (no boundary needed)")
        _assert("starting phase" in result.details.get("note", "").lower(),
                "Should note that Phase 1 is the starting phase")

    # ------------------------------------------------------------------
    # Tests 25-30: All 6 phase transitions (1->2, 2->3, ..., 6->7)
    # ------------------------------------------------------------------
    for target_phase in range(2, 8):
        source_phase = target_phase - 1
        begin_test(f"test_transition_phase{source_phase}_to_phase{target_phase}")
        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_mock_config()
            boundary = make_mock_boundary(phase=source_phase, config=config)
            phase_dir = os.path.join(tmpdir, f"phase{source_phase}")
            os.makedirs(phase_dir)
            _save_boundary(boundary, os.path.join(phase_dir, "phase_boundary.pt"))

            validator = PhaseBoundaryValidator(validation_mode="normal")
            result = validator.validate(
                phase=target_phase, run_dir=tmpdir, config=config
            )
            _assert(
                result.passed,
                f"Transition {source_phase}->{target_phase} should pass with valid boundary",
            )
            _assert(
                len(result.hard_errors) == 0,
                f"No hard errors for transition {source_phase}->{target_phase}",
            )

    # ------------------------------------------------------------------
    # Test 31: Invalid validation mode
    # ------------------------------------------------------------------
    begin_test("test_invalid_validation_mode")
    try:
        PhaseBoundaryValidator(validation_mode="bogus")
        _assert(False, "Should raise ValueError for invalid mode")
    except ValueError:
        _assert(True, "Correctly raises ValueError for invalid mode")

    # ------------------------------------------------------------------
    # Test 32: Empty state dict detection
    # ------------------------------------------------------------------
    begin_test("test_empty_state_dict")
    with tempfile.TemporaryDirectory() as tmpdir:
        config = make_mock_config()
        boundary = {
            "schema_version": SCHEMA_VERSION,
            "phase": 1,
            "state_dict": {},
            "compatibility": {"workspace_dim": 4096, "snn_hidden": 2048},
            "feature_flags": {"use_snn": True},
        }
        phase1_dir = os.path.join(tmpdir, "phase1")
        os.makedirs(phase1_dir)
        _save_boundary(boundary, os.path.join(phase1_dir, "phase_boundary.pt"))

        validator = PhaseBoundaryValidator()
        result = validator.validate(phase=2, run_dir=tmpdir, config=config)
        _assert(not result.passed, "Should fail with empty state_dict")
        empty_errors = [e for e in result.hard_errors if "empty" in e.lower()]
        _assert(len(empty_errors) > 0, "Should report empty state_dict")

    # ------------------------------------------------------------------
    # Test 33: Missing schema_version field
    # ------------------------------------------------------------------
    begin_test("test_missing_schema_version")
    with tempfile.TemporaryDirectory() as tmpdir:
        config = make_mock_config()
        boundary = make_mock_boundary(phase=1, config=config)
        del boundary["schema_version"]
        phase1_dir = os.path.join(tmpdir, "phase1")
        os.makedirs(phase1_dir)
        _save_boundary(boundary, os.path.join(phase1_dir, "phase_boundary.pt"))

        validator = PhaseBoundaryValidator(validation_mode="strict")
        result = validator.validate(phase=2, run_dir=tmpdir, config=config)
        _assert(not result.passed, "Strict mode should fail on missing schema_version")

        validator_normal = PhaseBoundaryValidator(validation_mode="normal")
        result_normal = validator_normal.validate(phase=2, run_dir=tmpdir, config=config)
        _assert(len(result_normal.soft_warnings) > 0,
                "Normal mode should warn on missing schema_version")

    # ------------------------------------------------------------------
    # Test 34: BoundaryCompatibilityMatrix.get_required_checks
    # ------------------------------------------------------------------
    begin_test("test_compatibility_matrix")
    checks_1_2 = BoundaryCompatibilityMatrix.get_required_checks(1, 2)
    _assert("workspace_dim" in checks_1_2, "Transition 1->2 should require workspace_dim")
    _assert("snn_hidden" in checks_1_2, "Transition 1->2 should require snn_hidden")

    checks_6_7 = BoundaryCompatibilityMatrix.get_required_checks(6, 7)
    _assert("reasoning_dim" in checks_6_7, "Transition 6->7 should require reasoning_dim")
    _assert(len(checks_6_7) == 4, "Transition 6->7 should have 4 checks")

    try:
        BoundaryCompatibilityMatrix.get_required_checks(1, 5)
        _assert(False, "Should raise ValueError for non-adjacent transition")
    except ValueError:
        _assert(True, "Correctly raises ValueError for missing matrix entry")

    # ------------------------------------------------------------------
    # Test 35: Dataset fingerprint mismatch
    # ------------------------------------------------------------------
    begin_test("test_dataset_fingerprint_mismatch")
    datasets_a = {"train": "imagenet21k", "size": 14000000}
    datasets_b = {"train": "cifar10", "size": 50000}
    with tempfile.TemporaryDirectory() as tmpdir:
        config = make_mock_config(datasets=datasets_a)
        boundary = make_mock_boundary(
            phase=4, config=config,
            state_dict_prefixes=["snn.", "encoder.", "htm.", "workspace."],
            include_dataset_fp=True, datasets=datasets_a,
        )
        phase4_dir = os.path.join(tmpdir, "phase4")
        os.makedirs(phase4_dir)
        _save_boundary(boundary, os.path.join(phase4_dir, "phase_boundary.pt"))

        config_new = make_mock_config(datasets=datasets_b)
        validator = PhaseBoundaryValidator(validation_mode="strict")
        result = validator.validate(phase=5, run_dir=tmpdir, config=config_new)
        ds_errors = [e for e in result.hard_errors if "dataset" in e.lower() or "fingerprint" in e.lower()]
        _assert(len(ds_errors) > 0, "Should detect dataset fingerprint mismatch in strict mode")

    # ------------------------------------------------------------------
    # Test 36: Dataset fingerprint match
    # ------------------------------------------------------------------
    begin_test("test_dataset_fingerprint_match")
    datasets = {"train": "imagenet21k", "size": 14000000}
    with tempfile.TemporaryDirectory() as tmpdir:
        config = make_mock_config(datasets=datasets)
        boundary = make_mock_boundary(
            phase=4, config=config,
            state_dict_prefixes=["snn.", "encoder.", "htm.", "workspace."],
            include_dataset_fp=True, datasets=datasets,
        )
        phase4_dir = os.path.join(tmpdir, "phase4")
        os.makedirs(phase4_dir)
        _save_boundary(boundary, os.path.join(phase4_dir, "phase_boundary.pt"))

        config_same = make_mock_config(datasets=datasets)
        validator = PhaseBoundaryValidator(validation_mode="strict")
        result = validator.validate(phase=5, run_dir=tmpdir, config=config_same)
        ds_errors = [e for e in result.hard_errors if "fingerprint" in e.lower()]
        _assert(len(ds_errors) == 0, "Should have no dataset fingerprint errors when datasets match")

    # ------------------------------------------------------------------
    # Test 37: PhaseContractSpec structure
    # ------------------------------------------------------------------
    begin_test("test_phase_contract_structure")
    for phase_num in range(2, 8):
        contract = PHASE_CONTRACTS[phase_num]
        _assert(contract.phase == phase_num, f"Contract phase should be {phase_num}")
        _assert(
            contract.required_boundary_from == phase_num - 1,
            f"Phase {phase_num} should require boundary from Phase {phase_num - 1}",
        )
        _assert(
            len(contract.required_state_dict_prefixes) > 0,
            f"Phase {phase_num} should have required state_dict prefixes",
        )
        _assert(
            len(contract.required_compatibility_fields) > 0,
            f"Phase {phase_num} should have required compatibility fields",
        )
        _assert(
            len(contract.required_feature_flags) > 0,
            f"Phase {phase_num} should have required feature flags",
        )

    # ------------------------------------------------------------------
    # Test 38: Permissive mode passes on flag inconsistency
    # ------------------------------------------------------------------
    begin_test("test_permissive_mode_flag_warning")
    with tempfile.TemporaryDirectory() as tmpdir:
        config_boundary = make_mock_config(use_snn=True)
        boundary = make_mock_boundary(phase=1, config=config_boundary)
        phase1_dir = os.path.join(tmpdir, "phase1")
        os.makedirs(phase1_dir)
        _save_boundary(boundary, os.path.join(phase1_dir, "phase_boundary.pt"))

        config_current = make_mock_config(use_snn=False)
        validator = PhaseBoundaryValidator(validation_mode="permissive")
        result = validator.validate(phase=2, run_dir=tmpdir, config=config_current)
        _assert(result.passed, "Permissive mode should pass despite flag inconsistency")
        flag_warnings = [w for w in result.soft_warnings if "use_snn" in w]
        _assert(len(flag_warnings) > 0, "Should have soft warning for use_snn")

    # ------------------------------------------------------------------
    # Test 39: Multiple dimension mismatches
    # ------------------------------------------------------------------
    begin_test("test_multiple_dimension_mismatches")
    with tempfile.TemporaryDirectory() as tmpdir:
        config_boundary = make_mock_config(workspace_dim=4096, snn_hidden=2048)
        boundary = make_mock_boundary(phase=1, config=config_boundary)
        phase1_dir = os.path.join(tmpdir, "phase1")
        os.makedirs(phase1_dir)
        _save_boundary(boundary, os.path.join(phase1_dir, "phase_boundary.pt"))

        config_current = make_mock_config(workspace_dim=1024, snn_hidden=256)
        validator = PhaseBoundaryValidator(validation_mode="normal")
        result = validator.validate(phase=2, run_dir=tmpdir, config=config_current)
        _assert(not result.passed, "Should fail with multiple mismatches")
        _assert(len(result.hard_errors) >= 2, "Should have at least 2 hard errors")

    # ------------------------------------------------------------------
    # Test 40: Phase transition 3->4 with all keys present
    # ------------------------------------------------------------------
    begin_test("test_phase3_to_4_complete")
    with tempfile.TemporaryDirectory() as tmpdir:
        config = make_mock_config()
        boundary = make_mock_boundary(
            phase=3, config=config,
            state_dict_prefixes=["snn.", "encoder.", "htm."],
        )
        phase3_dir = os.path.join(tmpdir, "phase3")
        os.makedirs(phase3_dir)
        _save_boundary(boundary, os.path.join(phase3_dir, "phase_boundary.pt"))

        validator = PhaseBoundaryValidator(validation_mode="normal")
        result = validator.validate(phase=4, run_dir=tmpdir, config=config)
        _assert(result.passed, "Phase 4 should pass with snn+encoder+htm keys")

    # ------------------------------------------------------------------
    # Test 41: DimensionChecker individual methods
    # ------------------------------------------------------------------
    begin_test("test_dimension_checker_individual_methods")
    config = make_mock_config(workspace_dim=4096, vocab_size=32000, snn_hidden=2048)
    boundary = make_mock_boundary(phase=1, config=config)

    r = DimensionChecker.check_workspace_dim(boundary, config)
    _assert(r[3], "workspace_dim check should pass")
    _assert(r[0] == "workspace_dim", "check name should be workspace_dim")

    r = DimensionChecker.check_vocab_size(boundary, config)
    _assert(r[3], "vocab_size check should pass")

    r = DimensionChecker.check_snn_neurons(boundary, config)
    _assert(r[3], "snn_hidden check should pass")

    # Test with None values (both None => pass)
    r = DimensionChecker.check_htm_columns(
        {"compatibility": {}}, {"htm_columns": None}
    )
    _assert(r[3], "Both-None htm_columns should pass")

    # ------------------------------------------------------------------
    # Test 42: validate_state_dict_keys directly
    # ------------------------------------------------------------------
    begin_test("test_validate_state_dict_keys_direct")
    boundary = {"state_dict": {"snn.weight": 1, "snn.bias": 2, "encoder.weight": 3}}
    errors, warnings = PhaseBoundaryValidator.validate_state_dict_keys(
        boundary, ["snn.", "encoder."]
    )
    _assert(len(errors) == 0, "No errors when all required prefixes present")
    _assert(len(warnings) == 0, "No warnings when no extra prefixes")

    # Missing prefix
    errors2, warnings2 = PhaseBoundaryValidator.validate_state_dict_keys(
        boundary, ["snn.", "encoder.", "htm."]
    )
    _assert(len(errors2) > 0, "Should error on missing htm. prefix")

    # Extra prefix
    boundary_extra = {"state_dict": {"snn.w": 1, "extra.w": 2}}
    errors3, warnings3 = PhaseBoundaryValidator.validate_state_dict_keys(
        boundary_extra, ["snn."]
    )
    _assert(len(warnings3) > 0, "Should warn on extra prefix")

    # ------------------------------------------------------------------
    # Test 43: validate_schema_version directly
    # ------------------------------------------------------------------
    begin_test("test_validate_schema_version_direct")
    errs = PhaseBoundaryValidator.validate_schema_version({"schema_version": SCHEMA_VERSION})
    _assert(len(errs) == 0, "No errors for matching schema version")

    errs2 = PhaseBoundaryValidator.validate_schema_version({"schema_version": "99.99"})
    _assert(len(errs2) > 0, "Should error on mismatched schema version")

    errs3 = PhaseBoundaryValidator.validate_schema_version({})
    _assert(len(errs3) > 0, "Should error on missing schema_version")

    # ------------------------------------------------------------------
    # Test 44: validate_dataset_identity directly
    # ------------------------------------------------------------------
    begin_test("test_validate_dataset_identity_direct")
    ds = {"train": "mnist", "size": 60000}
    fp = _compute_dataset_fingerprint(ds)
    boundary = {"dataset_fingerprint": fp}
    errs = PhaseBoundaryValidator.validate_dataset_identity(boundary, ds)
    _assert(len(errs) == 0, "No errors when datasets match")

    errs2 = PhaseBoundaryValidator.validate_dataset_identity(boundary, {"train": "cifar"})
    _assert(len(errs2) > 0, "Should error when datasets differ")

    errs3 = PhaseBoundaryValidator.validate_dataset_identity({}, ds)
    _assert(len(errs3) > 0, "Should error when fingerprint missing")

    # ------------------------------------------------------------------
    # Test 45: make_mock_boundary helper
    # ------------------------------------------------------------------
    begin_test("test_make_mock_boundary_helper")
    config = make_mock_config()
    b = make_mock_boundary(phase=3, config=config)
    _assert("schema_version" in b, "Boundary should have schema_version")
    _assert("state_dict" in b, "Boundary should have state_dict")
    _assert("compatibility" in b, "Boundary should have compatibility")
    _assert("feature_flags" in b, "Boundary should have feature_flags")
    _assert(b["compatibility"]["workspace_dim"] == 4096, "workspace_dim should be 4096")

    # Auto-collected prefixes up to phase 3
    sd_prefixes = {k.split(".")[0] + "." for k in b["state_dict"]}
    _assert("snn." in sd_prefixes, "Phase 3 boundary should include snn. prefix")
    _assert("encoder." in sd_prefixes, "Phase 3 boundary should include encoder. prefix")

    # ------------------------------------------------------------------
    # Test 46: make_mock_config helper
    # ------------------------------------------------------------------
    begin_test("test_make_mock_config_helper")
    cfg = make_mock_config(workspace_dim=512, use_snn=False)
    _assert(cfg["workspace_dim"] == 512, "workspace_dim should be 512")
    _assert(cfg["use_snn"] is False, "use_snn should be False")
    _assert("ablation_override" in cfg, "Should have ablation_override key")

    # ------------------------------------------------------------------
    # Test 47: PHASE_NAMES coverage
    # ------------------------------------------------------------------
    begin_test("test_phase_names_coverage")
    for p in range(1, 8):
        _assert(p in PHASE_NAMES, f"PHASE_NAMES should contain phase {p}")
    _assert(PHASE_NAMES[1] == "SNN Core", "Phase 1 name should be SNN Core")
    _assert(PHASE_NAMES[7] == "Meta-Learning", "Phase 7 name should be Meta-Learning")

    # ------------------------------------------------------------------
    # Test 48: PhaseBoundaryForker with symlink
    # ------------------------------------------------------------------
    begin_test("test_forker_symlink")
    with tempfile.TemporaryDirectory() as source_dir:
        with tempfile.TemporaryDirectory() as new_dir:
            config = make_mock_config()
            boundary = make_mock_boundary(phase=1, config=config)
            phase1_dir = os.path.join(source_dir, "phase1")
            os.makedirs(phase1_dir)
            _save_boundary(boundary, os.path.join(phase1_dir, "phase_boundary.pt"))

            forked_path = PhaseBoundaryForker.fork(
                source_run_dir=source_dir, source_phase=1, new_run_dir=new_dir,
                symlink=True,
            )
            _assert(os.path.islink(forked_path), "Forked file should be a symlink")
            _assert(os.path.isfile(forked_path), "Symlink should point to existing file")
            with open(os.path.join(new_dir, "fork_manifest.json")) as f:
                manifest = json.load(f)
            _assert(manifest["method"] == "symlink", "Manifest should record symlink method")

    # ------------------------------------------------------------------
    # Test 49: BoundaryValidationResult.add_error and add_warning
    # ------------------------------------------------------------------
    begin_test("test_result_add_error_and_warning")
    result = BoundaryValidationResult(passed=True, phase=2)
    _assert(result.passed, "Should start as passed")
    result.add_warning("test warning")
    _assert(result.passed, "Warning should not change passed status")
    _assert(len(result.soft_warnings) == 1, "Should have 1 warning")
    result.add_error("test error")
    _assert(not result.passed, "Error should set passed=False")
    _assert(len(result.hard_errors) == 1, "Should have 1 error")

    # ------------------------------------------------------------------
    # Test 50: Full end-to-end Phase 6->7 with all checks
    # ------------------------------------------------------------------
    begin_test("test_full_e2e_phase6_to_7")
    with tempfile.TemporaryDirectory() as tmpdir:
        datasets = {"train": "combined_v2", "size": 5000000}
        config = make_mock_config(
            workspace_dim=4096, snn_hidden=2048, htm_columns=2048,
            reasoning_dim=1024, datasets=datasets,
        )
        boundary = make_mock_boundary(
            phase=6, config=config,
            state_dict_prefixes=[
                "snn.", "encoder.", "htm.", "workspace.",
                "decision.", "reasoning.",
            ],
            include_dataset_fp=True, datasets=datasets,
        )
        phase6_dir = os.path.join(tmpdir, "phase6")
        os.makedirs(phase6_dir)
        _save_boundary(boundary, os.path.join(phase6_dir, "phase_boundary.pt"))

        for mode in ("strict", "normal", "permissive"):
            validator = PhaseBoundaryValidator(validation_mode=mode)
            result = validator.validate(phase=7, run_dir=tmpdir, config=config)
            _assert(
                result.passed,
                f"Phase 6->7 should pass in {mode} mode with correct artifacts",
            )

    # ------------------------------------------------------------------
    # Test 51: _compute_dataset_fingerprint determinism
    # ------------------------------------------------------------------
    begin_test("test_dataset_fingerprint_determinism")
    ds = {"train": "imagenet21k", "val": "imagenet1k", "size": 14000000}
    fp1 = _compute_dataset_fingerprint(ds)
    fp2 = _compute_dataset_fingerprint(ds)
    _assert(fp1 == fp2, "Same input should produce same fingerprint")
    _assert(len(fp1) == 16, "Fingerprint should be 16 hex chars")

    ds_different = {"train": "cifar10", "size": 50000}
    fp3 = _compute_dataset_fingerprint(ds_different)
    _assert(fp1 != fp3, "Different datasets should produce different fingerprints")

    # Key order should not matter
    ds_reordered = {"size": 14000000, "val": "imagenet1k", "train": "imagenet21k"}
    fp4 = _compute_dataset_fingerprint(ds_reordered)
    _assert(fp1 == fp4, "Key order should not affect fingerprint (sort_keys)")

    # ------------------------------------------------------------------
    # Test 52: Phase 5 requires dataset match from Phase 4
    # ------------------------------------------------------------------
    begin_test("test_phase5_dataset_match_required")
    contract = PHASE_CONTRACTS[5]
    _assert(
        contract.dataset_must_match_phase == 4,
        "Phase 5 contract should require dataset match from Phase 4",
    )
    contract_2 = PHASE_CONTRACTS[2]
    _assert(
        contract_2.dataset_must_match_phase is None,
        "Phase 2 should not require dataset match",
    )

    # ------------------------------------------------------------------
    # Test 53: Phase 7 contract has all 6 required prefixes
    # ------------------------------------------------------------------
    begin_test("test_phase7_contract_prefixes")
    c7 = PHASE_CONTRACTS[7]
    _assert(len(c7.required_state_dict_prefixes) == 6,
            "Phase 7 should require 6 state_dict prefixes")
    expected_prefixes = {"snn.", "encoder.", "htm.", "workspace.", "decision.", "reasoning."}
    actual_prefixes = set(c7.required_state_dict_prefixes)
    _assert(expected_prefixes == actual_prefixes,
            "Phase 7 prefixes should be snn/encoder/htm/workspace/decision/reasoning")

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 60)
    total = passed_count + failed_count
    print(f"TOTAL: {total} assertions | PASSED: {passed_count} | FAILED: {failed_count}")
    if failed_count == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"{failed_count} TESTS FAILED")
    print("=" * 60)
    sys.exit(0 if failed_count == 0 else 1)
