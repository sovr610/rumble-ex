"""
brain_ai/data/split_manager.py -- SplitManager for deterministic train/val/test splitting.

Provides deterministic, reproducible split creation with support for stratified
splitting, index persistence (save/load), and split validation.

Key classes:
    SplitResult    -- Container for split indices and metadata
    SplitManager   -- Main manager for creating and persisting splits

Usage:
    manager = SplitManager(seed=42)
    result = manager.create_splits(dataset, ratios=(0.8, 0.1, 0.1))
    manager.save_split_indices("splits.json")
    loaded = manager.load_split_indices("splits.json")
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SECTION 1: SplitResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class SplitResult:
    """Container for split results with indices and metadata."""
    train_indices: List[int]
    val_indices: List[int]
    test_indices: List[int]
    seed: int
    ratios: Tuple[float, float, float]
    total_samples: int
    stratified: bool = False
    class_distribution: Optional[Dict[str, Dict[int, int]]] = None

    @property
    def num_train(self) -> int:
        return len(self.train_indices)

    @property
    def num_val(self) -> int:
        return len(self.val_indices)

    @property
    def num_test(self) -> int:
        return len(self.test_indices)

    def verify_no_overlap(self) -> bool:
        """Verify that all three splits have no overlapping indices."""
        train_set = set(self.train_indices)
        val_set = set(self.val_indices)
        test_set = set(self.test_indices)
        return (
            len(train_set & val_set) == 0
            and len(train_set & test_set) == 0
            and len(val_set & test_set) == 0
        )

    def verify_coverage(self) -> bool:
        """Verify that all splits together cover the full dataset."""
        all_indices = set(self.train_indices) | set(self.val_indices) | set(self.test_indices)
        return len(all_indices) == self.total_samples

    def get_subsets(self, dataset: Dataset) -> Tuple[Subset, Subset, Subset]:
        """Create Subset objects from the split indices."""
        return (
            Subset(dataset, self.train_indices),
            Subset(dataset, self.val_indices),
            Subset(dataset, self.test_indices),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize split result to a dictionary."""
        result = {
            "train_indices": self.train_indices,
            "val_indices": self.val_indices,
            "test_indices": self.test_indices,
            "seed": self.seed,
            "ratios": list(self.ratios),
            "total_samples": self.total_samples,
            "stratified": self.stratified,
            "num_train": self.num_train,
            "num_val": self.num_val,
            "num_test": self.num_test,
        }
        if self.class_distribution is not None:
            # Convert int keys to strings for JSON serialization
            result["class_distribution"] = {
                split_name: {str(k): v for k, v in dist.items()}
                for split_name, dist in self.class_distribution.items()
            }
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SplitResult:
        """Deserialize split result from a dictionary."""
        class_dist = None
        if "class_distribution" in data and data["class_distribution"] is not None:
            class_dist = {
                split_name: {int(k): v for k, v in dist.items()}
                for split_name, dist in data["class_distribution"].items()
            }
        return cls(
            train_indices=data["train_indices"],
            val_indices=data["val_indices"],
            test_indices=data["test_indices"],
            seed=data["seed"],
            ratios=tuple(data["ratios"]),
            total_samples=data["total_samples"],
            stratified=data.get("stratified", False),
            class_distribution=class_dist,
        )


# ---------------------------------------------------------------------------
# SECTION 2: SplitManager
# ---------------------------------------------------------------------------

class SplitManager:
    """Manager for deterministic train/val/test split creation and persistence.

    Creates reproducible splits using a fixed random seed, supports stratified
    splitting to maintain class balance, and can save/load split indices to
    ensure exact reproducibility across runs.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._last_result: Optional[SplitResult] = None

    def create_splits(
        self,
        dataset: Dataset,
        ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    ) -> SplitResult:
        """Create deterministic train/val/test splits from a dataset.

        Args:
            dataset: PyTorch Dataset to split.
            ratios: (train_ratio, val_ratio, test_ratio), must sum to 1.0.

        Returns:
            SplitResult with train/val/test indices.

        Raises:
            ValueError: If ratios are invalid.
        """
        self._validate_ratios(ratios)
        total = len(dataset)

        gen = torch.Generator()
        gen.manual_seed(self.seed)
        perm = torch.randperm(total, generator=gen).tolist()

        n_test = int(total * ratios[2])
        n_val = int(total * ratios[1])
        n_train = total - n_val - n_test

        train_idx = perm[:n_train]
        val_idx = perm[n_train:n_train + n_val]
        test_idx = perm[n_train + n_val:]

        result = SplitResult(
            train_indices=train_idx,
            val_indices=val_idx,
            test_indices=test_idx,
            seed=self.seed,
            ratios=ratios,
            total_samples=total,
            stratified=False,
        )
        self._last_result = result
        return result

    def stratified_split(
        self,
        dataset: Dataset,
        labels: Tensor,
        ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    ) -> SplitResult:
        """Create stratified splits maintaining class proportions in each split.

        Args:
            dataset: PyTorch Dataset to split.
            labels: Tensor of integer labels, shape [N].
            ratios: (train_ratio, val_ratio, test_ratio).

        Returns:
            SplitResult with stratified train/val/test indices.

        Raises:
            ValueError: If ratios invalid or labels shape mismatch.
        """
        self._validate_ratios(ratios)
        total = len(dataset)

        if labels.shape[0] != total:
            raise ValueError(
                f"Labels length {labels.shape[0]} != dataset length {total}"
            )

        gen = torch.Generator()
        gen.manual_seed(self.seed)

        # Group indices by class
        unique_classes = labels.unique().tolist()
        class_indices: Dict[int, List[int]] = {c: [] for c in unique_classes}
        for idx in range(total):
            class_indices[int(labels[idx].item())].append(idx)

        train_idx: List[int] = []
        val_idx: List[int] = []
        test_idx: List[int] = []

        # Split each class proportionally
        for cls in unique_classes:
            cls_indices = class_indices[cls]
            n_cls = len(cls_indices)

            # Shuffle within class
            perm = torch.randperm(n_cls, generator=gen).tolist()
            shuffled = [cls_indices[p] for p in perm]

            n_test_cls = max(1, int(n_cls * ratios[2]))
            n_val_cls = max(1, int(n_cls * ratios[1]))
            n_train_cls = n_cls - n_val_cls - n_test_cls

            if n_train_cls < 1:
                # Edge case: very few samples per class
                n_train_cls = max(1, n_cls - 2)
                n_val_cls = min(1, n_cls - n_train_cls)
                n_test_cls = n_cls - n_train_cls - n_val_cls

            train_idx.extend(shuffled[:n_train_cls])
            val_idx.extend(shuffled[n_train_cls:n_train_cls + n_val_cls])
            test_idx.extend(shuffled[n_train_cls + n_val_cls:])

        # Compute class distribution
        class_distribution = {
            "train": {},
            "val": {},
            "test": {},
        }
        for idx in train_idx:
            c = int(labels[idx].item())
            class_distribution["train"][c] = class_distribution["train"].get(c, 0) + 1
        for idx in val_idx:
            c = int(labels[idx].item())
            class_distribution["val"][c] = class_distribution["val"].get(c, 0) + 1
        for idx in test_idx:
            c = int(labels[idx].item())
            class_distribution["test"][c] = class_distribution["test"].get(c, 0) + 1

        result = SplitResult(
            train_indices=train_idx,
            val_indices=val_idx,
            test_indices=test_idx,
            seed=self.seed,
            ratios=ratios,
            total_samples=total,
            stratified=True,
            class_distribution=class_distribution,
        )
        self._last_result = result
        return result

    def save_split_indices(self, path: str) -> None:
        """Save the last split result to a JSON file.

        Args:
            path: File path for the JSON output.

        Raises:
            RuntimeError: If no splits have been created yet.
        """
        if self._last_result is None:
            raise RuntimeError("No splits to save. Call create_splits() first.")

        data = self._last_result.to_dict()

        # Add checksum for integrity verification
        content = json.dumps(data, sort_keys=True)
        checksum = hashlib.sha256(content.encode()).hexdigest()[:16]
        data["checksum"] = checksum

        parent = Path(path).parent
        if parent != Path("."):
            parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved split indices to {path}")

    def load_split_indices(self, path: str) -> SplitResult:
        """Load split indices from a JSON file.

        Args:
            path: File path to load from.

        Returns:
            SplitResult reconstructed from saved data.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is corrupted (checksum mismatch).
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Split file not found: {path}")

        with open(path, "r") as f:
            data = json.load(f)

        # Verify checksum
        saved_checksum = data.pop("checksum", None)
        if saved_checksum is not None:
            content = json.dumps(data, sort_keys=True)
            computed = hashlib.sha256(content.encode()).hexdigest()[:16]
            if computed != saved_checksum:
                raise ValueError(
                    f"Checksum mismatch in {path}: expected {saved_checksum}, "
                    f"got {computed}. File may be corrupted."
                )

        result = SplitResult.from_dict(data)
        self._last_result = result
        return result

    def get_last_result(self) -> Optional[SplitResult]:
        """Return the last split result, or None if none created."""
        return self._last_result

    @staticmethod
    def _validate_ratios(ratios: Tuple[float, float, float]) -> None:
        """Validate that split ratios are valid."""
        if len(ratios) != 3:
            raise ValueError(f"ratios must have 3 elements, got {len(ratios)}")
        for r in ratios:
            if r < 0 or r > 1:
                raise ValueError(f"Each ratio must be in [0,1], got {r}")
        total = sum(ratios)
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"ratios must sum to 1.0, got {total}")


# ---------------------------------------------------------------------------
# SECTION 3: Utility functions
# ---------------------------------------------------------------------------

def quick_split(total: int, val_ratio: float = 0.1, test_ratio: float = 0.1,
                seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
    """Quick utility to create split indices without a Dataset object.

    Args:
        total: Total number of samples.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for test.
        seed: Random seed.

    Returns:
        Tuple of (train_indices, val_indices, test_indices).
    """
    gen = torch.Generator()
    gen.manual_seed(seed)
    perm = torch.randperm(total, generator=gen).tolist()
    n_test = int(total * test_ratio)
    n_val = int(total * val_ratio)
    n_train = total - n_val - n_test
    return perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]


def verify_split_quality(result: SplitResult) -> List[str]:
    """Run quality checks on a SplitResult and return warnings.

    Args:
        result: A SplitResult to validate.

    Returns:
        List of warning/error messages.
    """
    issues = []

    if not result.verify_no_overlap():
        issues.append("ERROR: Splits have overlapping indices")

    if not result.verify_coverage():
        issues.append("ERROR: Splits do not cover all samples")

    actual_train = result.num_train / result.total_samples
    actual_val = result.num_val / result.total_samples
    actual_test = result.num_test / result.total_samples

    if abs(actual_train - result.ratios[0]) > 0.05:
        issues.append(
            f"WARNING: Train ratio {actual_train:.3f} deviates from "
            f"requested {result.ratios[0]:.3f}"
        )
    if abs(actual_val - result.ratios[1]) > 0.05:
        issues.append(
            f"WARNING: Val ratio {actual_val:.3f} deviates from "
            f"requested {result.ratios[1]:.3f}"
        )
    if abs(actual_test - result.ratios[2]) > 0.05:
        issues.append(
            f"WARNING: Test ratio {actual_test:.3f} deviates from "
            f"requested {result.ratios[2]:.3f}"
        )

    if result.num_train == 0:
        issues.append("ERROR: Train split is empty")
    if result.num_val == 0:
        issues.append("WARNING: Validation split is empty")
    if result.num_test == 0:
        issues.append("WARNING: Test split is empty")

    return issues


# ============================================================================
# SELF-TESTS
# ============================================================================

if __name__ == "__main__":
    import sys
    import traceback
    import tempfile

    passed = 0
    failed = 0
    test_results = []

    def run_test(name, fn):
        global passed, failed
        try:
            fn()
            passed += 1
            test_results.append(("PASS", name))
        except Exception as e:
            failed += 1
            test_results.append(("FAIL", name, str(e)))
            traceback.print_exc()

    # Simple dataset for testing
    class _SimpleDataset(Dataset):
        def __init__(self, n):
            self.data = torch.randn(n, 4)
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]

    # ---- SplitResult tests ----

    def test_split_result_properties():
        r = SplitResult(
            train_indices=[0, 1, 2], val_indices=[3], test_indices=[4],
            seed=42, ratios=(0.6, 0.2, 0.2), total_samples=5
        )
        assert r.num_train == 3
        assert r.num_val == 1
        assert r.num_test == 1
    run_test("SplitResult properties", test_split_result_properties)

    def test_split_result_no_overlap():
        r = SplitResult(
            train_indices=[0, 1, 2], val_indices=[3, 4], test_indices=[5, 6],
            seed=42, ratios=(0.4, 0.3, 0.3), total_samples=7
        )
        assert r.verify_no_overlap()
    run_test("SplitResult verify_no_overlap true", test_split_result_no_overlap)

    def test_split_result_overlap_detected():
        r = SplitResult(
            train_indices=[0, 1, 2], val_indices=[2, 3], test_indices=[4],
            seed=42, ratios=(0.6, 0.2, 0.2), total_samples=5
        )
        assert not r.verify_no_overlap()
    run_test("SplitResult verify_no_overlap false", test_split_result_overlap_detected)

    def test_split_result_coverage():
        r = SplitResult(
            train_indices=[0, 1, 2], val_indices=[3], test_indices=[4],
            seed=42, ratios=(0.6, 0.2, 0.2), total_samples=5
        )
        assert r.verify_coverage()
    run_test("SplitResult verify_coverage true", test_split_result_coverage)

    def test_split_result_incomplete_coverage():
        r = SplitResult(
            train_indices=[0, 1], val_indices=[3], test_indices=[4],
            seed=42, ratios=(0.6, 0.2, 0.2), total_samples=5
        )
        assert not r.verify_coverage()
    run_test("SplitResult verify_coverage false", test_split_result_incomplete_coverage)

    def test_split_result_serialization():
        r = SplitResult(
            train_indices=[0, 1, 2], val_indices=[3], test_indices=[4],
            seed=42, ratios=(0.6, 0.2, 0.2), total_samples=5
        )
        d = r.to_dict()
        r2 = SplitResult.from_dict(d)
        assert r2.train_indices == r.train_indices
        assert r2.seed == r.seed
        assert r2.ratios == r.ratios
    run_test("SplitResult to_dict/from_dict roundtrip", test_split_result_serialization)

    def test_split_result_get_subsets():
        ds = _SimpleDataset(10)
        r = SplitResult(
            train_indices=[0, 1, 2, 3, 4, 5, 6, 7],
            val_indices=[8], test_indices=[9],
            seed=42, ratios=(0.8, 0.1, 0.1), total_samples=10
        )
        train_sub, val_sub, test_sub = r.get_subsets(ds)
        assert len(train_sub) == 8
        assert len(val_sub) == 1
        assert len(test_sub) == 1
    run_test("SplitResult get_subsets", test_split_result_get_subsets)

    # ---- SplitManager create_splits ----

    def test_create_splits_basic():
        ds = _SimpleDataset(100)
        mgr = SplitManager(seed=42)
        result = mgr.create_splits(ds)
        assert result.num_train + result.num_val + result.num_test == 100
    run_test("create_splits basic", test_create_splits_basic)

    def test_create_splits_no_overlap():
        ds = _SimpleDataset(1000)
        mgr = SplitManager(seed=42)
        result = mgr.create_splits(ds, ratios=(0.8, 0.1, 0.1))
        assert result.verify_no_overlap()
    run_test("create_splits no overlap", test_create_splits_no_overlap)

    def test_create_splits_coverage():
        ds = _SimpleDataset(500)
        mgr = SplitManager(seed=42)
        result = mgr.create_splits(ds)
        assert result.verify_coverage()
    run_test("create_splits full coverage", test_create_splits_coverage)

    def test_create_splits_ratios():
        ds = _SimpleDataset(1000)
        mgr = SplitManager(seed=42)
        result = mgr.create_splits(ds, ratios=(0.8, 0.1, 0.1))
        assert result.num_val == 100
        assert result.num_test == 100
        assert result.num_train == 800
    run_test("create_splits approximate ratios", test_create_splits_ratios)

    def test_create_splits_reproducible():
        ds = _SimpleDataset(200)
        r1 = SplitManager(seed=99).create_splits(ds)
        r2 = SplitManager(seed=99).create_splits(ds)
        assert r1.train_indices == r2.train_indices
        assert r1.val_indices == r2.val_indices
        assert r1.test_indices == r2.test_indices
    run_test("create_splits reproducible with same seed", test_create_splits_reproducible)

    def test_create_splits_different_seeds():
        ds = _SimpleDataset(200)
        r1 = SplitManager(seed=1).create_splits(ds)
        r2 = SplitManager(seed=2).create_splits(ds)
        assert r1.train_indices != r2.train_indices
    run_test("create_splits different with different seeds", test_create_splits_different_seeds)

    def test_create_splits_custom_ratios():
        ds = _SimpleDataset(100)
        mgr = SplitManager(seed=42)
        result = mgr.create_splits(ds, ratios=(0.7, 0.15, 0.15))
        assert result.num_train + result.num_val + result.num_test == 100
    run_test("create_splits custom ratios", test_create_splits_custom_ratios)

    def test_create_splits_invalid_ratios():
        ds = _SimpleDataset(100)
        mgr = SplitManager(seed=42)
        try:
            mgr.create_splits(ds, ratios=(0.5, 0.5, 0.5))
            assert False, "Should raise ValueError"
        except ValueError:
            pass
    run_test("create_splits invalid ratios", test_create_splits_invalid_ratios)

    def test_create_splits_negative_ratio():
        ds = _SimpleDataset(100)
        mgr = SplitManager(seed=42)
        try:
            mgr.create_splits(ds, ratios=(-0.1, 0.5, 0.6))
            assert False, "Should raise ValueError"
        except ValueError:
            pass
    run_test("create_splits negative ratio", test_create_splits_negative_ratio)

    # ---- Stratified split ----

    def test_stratified_split_basic():
        ds = _SimpleDataset(100)
        labels = torch.cat([torch.zeros(50, dtype=torch.long),
                            torch.ones(50, dtype=torch.long)])
        mgr = SplitManager(seed=42)
        result = mgr.stratified_split(ds, labels)
        assert result.stratified is True
        assert result.verify_no_overlap()
    run_test("stratified_split basic", test_stratified_split_basic)

    def test_stratified_split_all_classes():
        ds = _SimpleDataset(100)
        labels = torch.cat([torch.full((20,), i, dtype=torch.long) for i in range(5)])
        mgr = SplitManager(seed=42)
        result = mgr.stratified_split(ds, labels)
        # All classes should appear in all splits
        train_classes = set(labels[result.train_indices].tolist())
        val_classes = set(labels[result.val_indices].tolist())
        test_classes = set(labels[result.test_indices].tolist())
        assert train_classes == {0, 1, 2, 3, 4}
        assert val_classes == {0, 1, 2, 3, 4}
        assert test_classes == {0, 1, 2, 3, 4}
    run_test("stratified_split all classes in all splits", test_stratified_split_all_classes)

    def test_stratified_split_label_mismatch():
        ds = _SimpleDataset(100)
        labels = torch.zeros(50, dtype=torch.long)  # Wrong length
        mgr = SplitManager(seed=42)
        try:
            mgr.stratified_split(ds, labels)
            assert False, "Should raise ValueError"
        except ValueError:
            pass
    run_test("stratified_split label mismatch", test_stratified_split_label_mismatch)

    def test_stratified_split_reproducible():
        ds = _SimpleDataset(200)
        labels = torch.randint(0, 5, (200,))
        r1 = SplitManager(seed=42).stratified_split(ds, labels)
        r2 = SplitManager(seed=42).stratified_split(ds, labels)
        assert r1.train_indices == r2.train_indices
    run_test("stratified_split reproducible", test_stratified_split_reproducible)

    def test_stratified_class_distribution():
        ds = _SimpleDataset(100)
        labels = torch.cat([torch.full((20,), i, dtype=torch.long) for i in range(5)])
        mgr = SplitManager(seed=42)
        result = mgr.stratified_split(ds, labels)
        assert result.class_distribution is not None
        assert "train" in result.class_distribution
        assert "val" in result.class_distribution
        assert "test" in result.class_distribution
    run_test("stratified_split class distribution", test_stratified_class_distribution)

    # ---- Save / Load ----

    def test_save_load_roundtrip():
        ds = _SimpleDataset(100)
        mgr = SplitManager(seed=42)
        original = mgr.create_splits(ds)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name

        try:
            mgr.save_split_indices(path)
            loaded = mgr.load_split_indices(path)
            assert loaded.train_indices == original.train_indices
            assert loaded.val_indices == original.val_indices
            assert loaded.test_indices == original.test_indices
            assert loaded.seed == original.seed
        finally:
            os.unlink(path)
    run_test("save/load roundtrip", test_save_load_roundtrip)

    def test_save_creates_file():
        ds = _SimpleDataset(50)
        mgr = SplitManager(seed=42)
        mgr.create_splits(ds)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name

        try:
            mgr.save_split_indices(path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)
    run_test("save creates file", test_save_creates_file)

    def test_save_no_splits_raises():
        mgr = SplitManager(seed=42)
        try:
            mgr.save_split_indices("/tmp/test_no_splits.json")
            assert False, "Should raise RuntimeError"
        except RuntimeError:
            pass
    run_test("save without splits raises", test_save_no_splits_raises)

    def test_load_nonexistent_raises():
        mgr = SplitManager(seed=42)
        try:
            mgr.load_split_indices("/tmp/nonexistent_file_12345.json")
            assert False, "Should raise FileNotFoundError"
        except FileNotFoundError:
            pass
    run_test("load nonexistent file raises", test_load_nonexistent_raises)

    def test_load_corrupted_raises():
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
            # Write valid JSON with a mismatched checksum
            data = {
                "train_indices": [0, 1], "val_indices": [2], "test_indices": [3],
                "seed": 42, "ratios": [0.5, 0.25, 0.25], "total_samples": 4,
                "num_train": 2, "num_val": 1, "num_test": 1,
                "checksum": "0000000000000000"
            }
            json.dump(data, f)

        try:
            mgr = SplitManager(seed=42)
            mgr.load_split_indices(path)
            assert False, "Should raise ValueError for checksum mismatch"
        except ValueError:
            pass
        finally:
            os.unlink(path)
    run_test("load corrupted file raises", test_load_corrupted_raises)

    def test_save_load_stratified():
        ds = _SimpleDataset(100)
        labels = torch.randint(0, 5, (100,))
        mgr = SplitManager(seed=42)
        original = mgr.stratified_split(ds, labels)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name

        try:
            mgr.save_split_indices(path)
            loaded = mgr.load_split_indices(path)
            assert loaded.stratified == original.stratified
            assert loaded.class_distribution is not None
        finally:
            os.unlink(path)
    run_test("save/load stratified roundtrip", test_save_load_stratified)

    # ---- quick_split ----

    def test_quick_split_basic():
        t, v, s = quick_split(100)
        assert len(t) + len(v) + len(s) == 100
    run_test("quick_split basic", test_quick_split_basic)

    def test_quick_split_no_overlap():
        t, v, s = quick_split(1000)
        assert len(set(t) & set(v)) == 0
        assert len(set(t) & set(s)) == 0
        assert len(set(v) & set(s)) == 0
    run_test("quick_split no overlap", test_quick_split_no_overlap)

    # ---- verify_split_quality ----

    def test_verify_quality_ok():
        r = SplitResult(
            train_indices=list(range(80)), val_indices=list(range(80, 90)),
            test_indices=list(range(90, 100)),
            seed=42, ratios=(0.8, 0.1, 0.1), total_samples=100
        )
        issues = verify_split_quality(r)
        assert len(issues) == 0
    run_test("verify_split_quality no issues", test_verify_quality_ok)

    def test_verify_quality_overlap():
        r = SplitResult(
            train_indices=[0, 1, 2], val_indices=[2, 3], test_indices=[4],
            seed=42, ratios=(0.6, 0.2, 0.2), total_samples=5
        )
        issues = verify_split_quality(r)
        assert any("overlap" in i.lower() for i in issues)
    run_test("verify_split_quality detects overlap", test_verify_quality_overlap)

    def test_verify_quality_coverage():
        r = SplitResult(
            train_indices=[0, 1], val_indices=[3], test_indices=[4],
            seed=42, ratios=(0.6, 0.2, 0.2), total_samples=5
        )
        issues = verify_split_quality(r)
        assert any("cover" in i.lower() for i in issues)
    run_test("verify_split_quality detects incomplete coverage", test_verify_quality_coverage)

    # ---- get_last_result ----

    def test_get_last_result():
        mgr = SplitManager(seed=42)
        assert mgr.get_last_result() is None
        ds = _SimpleDataset(50)
        mgr.create_splits(ds)
        assert mgr.get_last_result() is not None
    run_test("get_last_result", test_get_last_result)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print(f"SPLIT MANAGER SELF-TESTS: {passed} passed, {failed} failed")
    print("=" * 60)
    for result in test_results:
        status = result[0]
        name = result[1]
        extra = f" -- {result[2]}" if len(result) > 2 else ""
        print(f"  [{status}] {name}{extra}")

    sys.exit(0 if failed == 0 else 1)
