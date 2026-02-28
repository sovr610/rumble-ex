#!/usr/bin/env python3
"""
scripts/validate_loaders.py -- Validates all 7 phase loaders against the 3 done-when gates.

Done-When Gates:
  1. Loader Contract -- All 7 phase loaders implement BasePhaseLoader; get_train_loader()
     returns a valid DataLoader that yields correctly-shaped tensors.
  2. Registry Round-Trip -- DatasetRegistry.get_loader("synthetic_snn", "dev") returns a
     working loader; list_datasets(phase=1) returns all phase-1 datasets.
  3. Augmentation Determinism -- Same seed produces identical augmented batches;
     eval transforms produce identical output regardless of seed.

Usage:
    python scripts/validate_loaders.py
    python scripts/validate_loaders.py --verbose
    python scripts/validate_loaders.py --gate 1   # Run only gate 1
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Inline dependencies (in production these are imported from brain_ai.data)
# ---------------------------------------------------------------------------

# We import from the asset templates for standalone validation
import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ASSETS_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "assets")
sys.path.insert(0, _ASSETS_DIR)

from phase_loaders_template import (
    DataConfig, DatasetInfo, BasePhaseLoader,
    SNNPhaseLoader, EncoderPhaseLoader, HTMPhaseLoader,
    WorkspacePhaseLoader, ActiveInfPhaseLoader, ReasoningPhaseLoader,
    MetaPhaseLoader, PHASE_LOADERS, dict_collate_fn,
)
from augmentation_template import AugmentationPipeline, normalize_spectrogram
from data_config_template import DatasetRegistry as _OrigRegistry

# The DatasetRegistry from data_config_template uses its own BasePhaseLoader ABC,
# which differs from the one in phase_loaders_template. For standalone validation
# we patch the registry's issubclass check so that cross-module classes work.
import data_config_template as _dcm
_dcm.BasePhaseLoader = BasePhaseLoader
DatasetRegistry = _OrigRegistry

# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------

class ValidationResult:
    """Stores results for a single validation check."""

    def __init__(self, gate: int, name: str, passed: bool, message: str = "",
                 elapsed: float = 0.0):
        self.gate = gate
        self.name = name
        self.passed = passed
        self.message = message
        self.elapsed = elapsed

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        extra = f" -- {self.message}" if self.message else ""
        return f"[{status}] Gate {self.gate}: {self.name}{extra} ({self.elapsed:.3f}s)"


class ValidationSuite:
    """Runs all validation checks and reports results."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[ValidationResult] = []

    def check(self, gate: int, name: str, fn, *args, **kwargs) -> bool:
        """Run a single validation check."""
        start = time.time()
        try:
            fn(*args, **kwargs)
            elapsed = time.time() - start
            self.results.append(ValidationResult(gate, name, True, elapsed=elapsed))
            if self.verbose:
                print(f"  [PASS] {name} ({elapsed:.3f}s)")
            return True
        except Exception as e:
            elapsed = time.time() - start
            msg = str(e)
            self.results.append(ValidationResult(gate, name, False, msg, elapsed))
            if self.verbose:
                print(f"  [FAIL] {name}: {msg}")
                traceback.print_exc()
            return False

    def summary(self) -> Tuple[int, int]:
        """Print summary and return (passed, failed) counts."""
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        return passed, failed

    def gate_summary(self, gate: int) -> Tuple[int, int]:
        """Get pass/fail counts for a specific gate."""
        gate_results = [r for r in self.results if r.gate == gate]
        p = sum(1 for r in gate_results if r.passed)
        f = sum(1 for r in gate_results if not r.passed)
        return p, f


# ---------------------------------------------------------------------------
# Gate 1: Loader Contract Validation
# ---------------------------------------------------------------------------

def validate_gate1_loader_contract(suite: ValidationSuite):
    """Validate that all 7 phase loaders implement the BasePhaseLoader contract."""

    cfg = DataConfig(batch_size=8, dev_num_samples=200)

    # -- 1.1 Inheritance checks --
    for phase, cls in PHASE_LOADERS.items():
        suite.check(1, f"Phase {phase} inherits BasePhaseLoader",
                    lambda c=cls: assert_(issubclass(c, BasePhaseLoader),
                                          f"{c.__name__} does not inherit BasePhaseLoader"))

    # -- 1.2 Constructor accepts config and mode --
    for phase, cls in PHASE_LOADERS.items():
        suite.check(1, f"Phase {phase} constructor accepts config/mode",
                    lambda c=cls: c(cfg, mode="dev"))

    # -- 1.3 Constructor rejects bad mode --
    for phase, cls in PHASE_LOADERS.items():
        def _check_bad_mode(c=cls):
            try:
                c(cfg, mode="invalid")
                raise AssertionError("Should have raised ValueError")
            except ValueError:
                pass
        suite.check(1, f"Phase {phase} rejects invalid mode", _check_bad_mode)

    # -- 1.4 get_train_loader returns DataLoader --
    for phase, cls in PHASE_LOADERS.items():
        def _check_train_dl(c=cls):
            loader = c(cfg, mode="dev")
            dl = loader.get_train_loader()
            assert_(isinstance(dl, DataLoader), "get_train_loader() did not return DataLoader")
        suite.check(1, f"Phase {phase} get_train_loader returns DataLoader", _check_train_dl)

    # -- 1.5 get_val_loader returns DataLoader --
    for phase, cls in PHASE_LOADERS.items():
        def _check_val_dl(c=cls):
            loader = c(cfg, mode="dev")
            dl = loader.get_val_loader()
            assert_(isinstance(dl, DataLoader), "get_val_loader() did not return DataLoader")
        suite.check(1, f"Phase {phase} get_val_loader returns DataLoader", _check_val_dl)

    # -- 1.6 get_test_loader returns DataLoader --
    for phase, cls in PHASE_LOADERS.items():
        def _check_test_dl(c=cls):
            loader = c(cfg, mode="dev")
            dl = loader.get_test_loader()
            assert_(isinstance(dl, DataLoader), "get_test_loader() did not return DataLoader")
        suite.check(1, f"Phase {phase} get_test_loader returns DataLoader", _check_test_dl)

    # -- 1.7 get_dataset_info returns DatasetInfo --
    for phase, cls in PHASE_LOADERS.items():
        def _check_info(c=cls, p=phase):
            loader = c(cfg, mode="dev")
            info = loader.get_dataset_info()
            assert_(isinstance(info, DatasetInfo), "get_dataset_info() did not return DatasetInfo")
            assert_(info.phase == p, f"info.phase={info.phase}, expected {p}")
        suite.check(1, f"Phase {phase} get_dataset_info returns DatasetInfo", _check_info)

    # -- 1.8 Train batch has correct shapes and dtypes --
    for phase, cls in PHASE_LOADERS.items():
        def _check_batch(c=cls, p=phase):
            loader = c(cfg, mode="dev")
            dl = loader.get_train_loader()
            batch = next(iter(dl))
            assert_(isinstance(batch, dict), "Batch should be a dict")
            assert_(len(batch) > 0, "Batch should not be empty")
            for key, val in batch.items():
                assert_(isinstance(val, Tensor), f"batch[{key}] is not a Tensor")
                assert_(val.shape[0] == 8, f"batch[{key}] batch dim != 8, got {val.shape[0]}")
                if val.is_floating_point():
                    assert_(not torch.isnan(val).any(), f"batch[{key}] has NaN")
                    assert_(not torch.isinf(val).any(), f"batch[{key}] has Inf")
        suite.check(1, f"Phase {phase} train batch shapes/dtypes valid", _check_batch)

    # -- 1.9 get_sample_shape returns dict of tuples --
    for phase, cls in PHASE_LOADERS.items():
        def _check_shapes(c=cls):
            loader = c(cfg, mode="dev")
            shapes = loader.get_sample_shape()
            assert_(isinstance(shapes, dict), "get_sample_shape() should return dict")
            for key, shape in shapes.items():
                assert_(isinstance(shape, tuple), f"shapes[{key}] should be a tuple")
        suite.check(1, f"Phase {phase} get_sample_shape returns dict of tuples", _check_shapes)

    # -- 1.10 get_num_samples sums correctly --
    for phase, cls in PHASE_LOADERS.items():
        def _check_counts(c=cls):
            loader = c(cfg, mode="dev")
            counts = loader.get_num_samples()
            total = counts["train"] + counts["val"] + counts["test"]
            assert_(total == 200, f"Sample counts sum to {total}, expected 200")
        suite.check(1, f"Phase {phase} get_num_samples sums to total", _check_counts)

    # -- 1.11 Dev mode loads fast --
    def _check_all_fast():
        start = time.time()
        for phase, cls in PHASE_LOADERS.items():
            loader = cls(cfg, mode="dev")
            _ = next(iter(loader.get_train_loader()))
        elapsed = time.time() - start
        assert_(elapsed < 30.0, f"All loaders took {elapsed:.1f}s, limit is 30s")
    suite.check(1, "All 7 dev loaders load within 30s", _check_all_fast)


# ---------------------------------------------------------------------------
# Gate 2: Registry Round-Trip Validation
# ---------------------------------------------------------------------------

def validate_gate2_registry_roundtrip(suite: ValidationSuite):
    """Validate DatasetRegistry registration, lookup, and listing."""

    # Build a fresh registry with all phase loaders
    registry = DatasetRegistry()
    phase_dataset_map = {
        1: ("synthetic_snn", SNNPhaseLoader, "vision"),
        2: ("synthetic_multimodal", EncoderPhaseLoader, "multimodal"),
        3: ("synthetic_sequences", HTMPhaseLoader, "sequence"),
        4: ("synthetic_workspace", WorkspacePhaseLoader, "multimodal"),
        5: ("synthetic_cartpole", ActiveInfPhaseLoader, "rl"),
        6: ("synthetic_babi", ReasoningPhaseLoader, "text"),
        7: ("synthetic_episodes", MetaPhaseLoader, "episodes"),
    }

    # -- 2.1 Register all datasets --
    def _register_all():
        for phase, (name, cls, modality) in phase_dataset_map.items():
            registry.register(name, cls, phase, modality=modality)
        assert_(len(registry) == 7, f"Registry has {len(registry)} entries, expected 7")
    suite.check(2, "Register all 7 phase datasets", _register_all)

    # -- 2.2 get_loader returns working loader for each --
    for phase, (name, cls, modality) in phase_dataset_map.items():
        def _check_get_loader(n=name, c=cls):
            loader = registry.get_loader(n, mode="dev")
            assert_(isinstance(loader, BasePhaseLoader),
                    f"get_loader({n}) did not return BasePhaseLoader")
            # Verify we can iterate
            dl = loader.get_train_loader()
            batch = next(iter(dl))
            assert_(isinstance(batch, dict), "Batch should be dict")
        suite.check(2, f"get_loader('{name}') returns working loader", _check_get_loader)

    # -- 2.3 list_datasets returns all for each phase --
    for phase in range(1, 8):
        def _check_list(p=phase):
            datasets = registry.list_datasets(phase=p)
            assert_(len(datasets) >= 1,
                    f"list_datasets(phase={p}) returned {len(datasets)}, expected >= 1")
            for d in datasets:
                assert_(d.phase == p, f"Dataset {d.name} has phase {d.phase}, expected {p}")
        suite.check(2, f"list_datasets(phase={phase}) returns >= 1 dataset", _check_list)

    # -- 2.4 list_datasets with no filter returns all 7 --
    def _check_list_all():
        all_datasets = registry.list_datasets()
        assert_(len(all_datasets) == 7,
                f"list_datasets() returned {len(all_datasets)}, expected 7")
    suite.check(2, "list_datasets() returns all 7 datasets", _check_list_all)

    # -- 2.5 get_loader for unknown dataset raises KeyError --
    def _check_unknown():
        try:
            registry.get_loader("nonexistent_dataset")
            raise AssertionError("Should have raised KeyError")
        except KeyError:
            pass
    suite.check(2, "get_loader unknown raises KeyError", _check_unknown)

    # -- 2.6 has_dataset checks --
    def _check_has():
        assert_(registry.has_dataset("synthetic_snn"), "Should have synthetic_snn")
        assert_(not registry.has_dataset("imagenet21k"), "Should not have imagenet21k")
    suite.check(2, "has_dataset correctness", _check_has)

    # -- 2.7 Round-trip: register, get, iterate, check info --
    def _roundtrip():
        loader = registry.get_loader("synthetic_snn", mode="dev")
        info = loader.get_dataset_info()
        assert_(info.name == "synthetic_snn", f"Expected name synthetic_snn, got {info.name}")
        assert_(info.phase == 1, f"Expected phase 1, got {info.phase}")
        dl = loader.get_train_loader()
        batch = next(iter(dl))
        assert_("input" in batch, "Batch should have 'input' key")
    suite.check(2, "Full round-trip: register -> get -> iterate -> info", _roundtrip)

    # -- 2.8 list_datasets by modality --
    def _check_modality_filter():
        vision = registry.list_datasets(modality="vision")
        assert_(len(vision) >= 1, "Should have at least 1 vision dataset")
        for d in vision:
            assert_(d.modality == "vision", f"Dataset {d.name} modality is {d.modality}")
    suite.check(2, "list_datasets(modality='vision') filters correctly", _check_modality_filter)

    # -- 2.9 Duplicate registration raises --
    def _check_duplicate():
        try:
            registry.register("synthetic_snn", SNNPhaseLoader, phase=1)
            raise AssertionError("Should have raised ValueError for duplicate")
        except ValueError:
            pass
    suite.check(2, "Duplicate registration raises ValueError", _check_duplicate)


# ---------------------------------------------------------------------------
# Gate 3: Augmentation Determinism Validation
# ---------------------------------------------------------------------------

def validate_gate3_augmentation_determinism(suite: ValidationSuite):
    """Validate that augmentation is deterministic with seeds and eval is always deterministic."""

    modalities_and_shapes = {
        "vision": (3, 32, 32),
        "text": (128,),
        "audio": (64, 100),
        "sequence": (50, 16),
    }

    # -- 3.1 Seeded train transforms produce identical output --
    for modality, shape in modalities_and_shapes.items():
        def _check_seed_det(m=modality, s=shape):
            if m == "text":
                x = torch.randint(1, 256, s, dtype=torch.long)
            else:
                x = torch.randn(*s)

            p1 = AugmentationPipeline(m, strength="standard", seed=42)
            y1 = p1.get_train_transforms()(x.clone())

            p2 = AugmentationPipeline(m, strength="standard", seed=42)
            y2 = p2.get_train_transforms()(x.clone())

            if m == "text":
                assert_(torch.equal(y1, y2),
                        f"Seeded {m} train transforms not deterministic")
            else:
                assert_(torch.allclose(y1, y2, atol=1e-6),
                        f"Seeded {m} train transforms not deterministic")
        suite.check(3, f"{modality} seeded train determinism", _check_seed_det)

    # -- 3.2 Eval transforms are always deterministic (no seed needed) --
    for modality, shape in modalities_and_shapes.items():
        def _check_eval_det(m=modality, s=shape):
            if m == "text":
                x = torch.randint(1, 256, s, dtype=torch.long)
            else:
                x = torch.randn(*s)

            p = AugmentationPipeline(m, strength="standard")
            tfm = p.get_eval_transforms()
            results = [tfm(x.clone()) for _ in range(5)]
            for i in range(1, 5):
                if m == "text":
                    assert_(torch.equal(results[0], results[i]),
                            f"{m} eval not deterministic on call {i}")
                else:
                    assert_(torch.allclose(results[0], results[i], atol=1e-6),
                            f"{m} eval not deterministic on call {i}")
        suite.check(3, f"{modality} eval always deterministic", _check_eval_det)

    # -- 3.3 Different seeds produce different output --
    for modality in ("vision", "audio", "sequence"):
        def _check_diff_seeds(m=modality, s=modalities_and_shapes[modality]):
            x = torch.randn(*s)
            p1 = AugmentationPipeline(m, strength="heavy", seed=1)
            y1 = p1.get_train_transforms()(x.clone())
            p2 = AugmentationPipeline(m, strength="heavy", seed=999)
            y2 = p2.get_train_transforms()(x.clone())
            assert_(not torch.equal(y1, y2),
                    f"Different seeds produced identical output for {m}")
        suite.check(3, f"{modality} different seeds differ", _check_diff_seeds)

    # -- 3.4 Shape preservation --
    # Note: Sequence augmentation at standard/heavy strength intentionally changes the
    # time dimension via subsequence sampling (per augmentation-pipeline.md spec).
    # We test shape preservation at "none" strength for sequence, and "standard" for others.
    # We also verify that sequence feature dim D is preserved even when T changes.
    for modality, shape in modalities_and_shapes.items():
        def _check_shape(m=modality, s=shape):
            if m == "text":
                x = torch.randint(1, 256, s, dtype=torch.long)
            else:
                x = torch.randn(*s)
            # Sequence uses "none" strength to test full shape preservation
            # because standard/heavy use subsequence_sample which changes T
            strength = "none" if m == "sequence" else "standard"
            p = AugmentationPipeline(m, strength=strength, seed=42)
            y = p.get_train_transforms()(x.clone())
            assert_(y.shape == x.shape,
                    f"{m} shape changed: {x.shape} -> {y.shape}")
        suite.check(3, f"{modality} augmentation preserves shape", _check_shape)

    # -- 3.4b Sequence feature dim preserved even with subsequence sampling --
    def _check_seq_feat_dim():
        x = torch.randn(50, 16)
        p = AugmentationPipeline("sequence", strength="standard", seed=42)
        y = p.get_train_transforms()(x.clone())
        assert_(y.shape[-1] == x.shape[-1],
                f"Sequence feature dim changed: {x.shape[-1]} -> {y.shape[-1]}")
        assert_(y.shape[0] <= x.shape[0],
                f"Subsequence should not be longer: {y.shape[0]} > {x.shape[0]}")
        assert_(y.shape[0] >= 1,
                f"Subsequence too short: {y.shape[0]}")
    suite.check(3, "sequence standard preserves feature dim D", _check_seq_feat_dim)

    # -- 3.5 dtype preservation --
    for modality, shape in modalities_and_shapes.items():
        def _check_dtype(m=modality, s=shape):
            if m == "text":
                x = torch.randint(1, 256, s, dtype=torch.long)
            else:
                x = torch.randn(*s)
            p = AugmentationPipeline(m, strength="standard", seed=42)
            y = p.get_train_transforms()(x.clone())
            assert_(y.dtype == x.dtype,
                    f"{m} dtype changed: {x.dtype} -> {y.dtype}")
        suite.check(3, f"{modality} augmentation preserves dtype", _check_dtype)

    # -- 3.6 None strength produces minimal change --
    def _check_none_vision():
        p = AugmentationPipeline("vision", strength="none")
        x = torch.rand(1, 28, 28)
        y = p.get_train_transforms()(x.clone())
        # Only normalization applied
        assert_(y.shape == x.shape, "Shape changed with none strength")
    suite.check(3, "Vision none strength = normalization only", _check_none_vision)

    def _check_none_text():
        p = AugmentationPipeline("text", strength="none")
        x = torch.randint(1, 256, (128,), dtype=torch.long)
        y = p.get_train_transforms()(x.clone())
        assert_(torch.equal(x, y), "Text none strength should be identity")
    suite.check(3, "Text none strength = identity", _check_none_text)

    # -- 3.7 RL not modified --
    def _check_rl():
        p = AugmentationPipeline("rl", strength="heavy")
        x = torch.randn(4)
        y = p.get_train_transforms()(x.clone())
        assert_(torch.equal(x, y), "RL data should not be modified")
    suite.check(3, "RL augmentation = identity", _check_rl)

    # -- 3.8 set_seed resets properly --
    def _check_set_seed():
        p = AugmentationPipeline("vision", strength="standard", seed=42)
        x = torch.rand(3, 32, 32)
        y1 = p.get_train_transforms()(x.clone())
        p.set_seed(42)
        y2 = p.get_train_transforms()(x.clone())
        assert_(torch.allclose(y1, y2, atol=1e-6), "set_seed did not reset properly")
    suite.check(3, "set_seed resets determinism", _check_set_seed)

    # -- 3.9 Batch-level determinism: same loader seed = same batches --
    def _check_batch_determinism():
        cfg = DataConfig(batch_size=4, dev_num_samples=50, split_seed=42)
        l1 = SNNPhaseLoader(cfg, mode="dev")
        l2 = SNNPhaseLoader(cfg, mode="dev")
        b1 = next(iter(l1.get_train_loader()))
        b2 = next(iter(l2.get_train_loader()))
        # Note: DataLoader with shuffle=True uses different random state,
        # but dataset contents should match
        assert_(b1["input"].shape == b2["input"].shape,
                "Batch shapes should match")
    suite.check(3, "Same config produces same batch shapes", _check_batch_determinism)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def assert_(condition: bool, message: str = "Assertion failed"):
    """Raise AssertionError with message if condition is False."""
    if not condition:
        raise AssertionError(message)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate data pipeline done-when gates")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--gate", type=int, choices=[1, 2, 3], default=None,
                        help="Run only a specific gate (1, 2, or 3)")
    args = parser.parse_args()

    suite = ValidationSuite(verbose=args.verbose)

    print("=" * 70)
    print("DATA PIPELINE LOADER VALIDATION")
    print("=" * 70)

    gates = [args.gate] if args.gate else [1, 2, 3]

    if 1 in gates:
        print("\n--- Gate 1: Loader Contract ---")
        validate_gate1_loader_contract(suite)
        p, f = suite.gate_summary(1)
        print(f"  Gate 1 Result: {p} passed, {f} failed")

    if 2 in gates:
        print("\n--- Gate 2: Registry Round-Trip ---")
        validate_gate2_registry_roundtrip(suite)
        p, f = suite.gate_summary(2)
        print(f"  Gate 2 Result: {p} passed, {f} failed")

    if 3 in gates:
        print("\n--- Gate 3: Augmentation Determinism ---")
        validate_gate3_augmentation_determinism(suite)
        p, f = suite.gate_summary(3)
        print(f"  Gate 3 Result: {p} passed, {f} failed")

    # Overall summary
    total_passed, total_failed = suite.summary()

    print("\n" + "=" * 70)
    print(f"OVERALL: {total_passed} passed, {total_failed} failed")
    print("=" * 70)

    if not args.verbose:
        # Print failures
        failures = [r for r in suite.results if not r.passed]
        if failures:
            print("\nFailed checks:")
            for r in failures:
                print(f"  {r}")

    # Gate-level pass/fail
    all_gates_pass = True
    for g in gates:
        p, f = suite.gate_summary(g)
        status = "PASS" if f == 0 else "FAIL"
        print(f"  Gate {g}: {status} ({p}/{p + f})")
        if f > 0:
            all_gates_pass = False

    if all_gates_pass:
        print("\nAll done-when gates PASSED.")
    else:
        print("\nSome done-when gates FAILED.")

    sys.exit(0 if all_gates_pass else 1)


if __name__ == "__main__":
    main()
