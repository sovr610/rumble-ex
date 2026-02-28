"""
brain_ai/data/registry.py -- DataConfig dataclass, DatasetRegistry class, and validation.

This module provides the centralized dataset registry that maps dataset names to their
loader classes and metadata. It enables discovery, instantiation, and management of
datasets across all 7 training phases.

Key classes:
    DataConfig       -- Configuration for data loading (re-exported from base_loader)
    DatasetEntry     -- Internal registry entry with full metadata
    DatasetRegistry  -- Singleton registry for all datasets

Usage:
    registry = DatasetRegistry()
    registry.register("synthetic_snn", SNNPhaseLoader, phase=1)
    loader = registry.get_loader("synthetic_snn", mode="dev")
    datasets = registry.list_datasets(phase=1)
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SECTION 1: DataConfig (standalone definition for this template)
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Centralized configuration for data loading across all phases."""
    root_dir: str = "data/"
    phase: int = 1
    mode: str = "dev"
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = False
    prefetch_factor: int = 2
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    split_seed: int = 42
    augmentation_strength: str = "standard"
    use_streaming: bool = False
    cache_dir: Optional[str] = None
    dev_num_samples: int = 1000
    dev_image_size: int = 28
    dev_image_channels: int = 1
    prod_image_size: int = 224
    prod_image_channels: int = 3
    dev_seq_len: int = 128
    prod_seq_len: int = 512
    dev_vocab_size: int = 256
    prod_vocab_size: int = 128000
    dev_n_mels: int = 64
    dev_audio_T: int = 100
    prod_n_mels: int = 128
    prod_audio_T: int = 1000
    dev_seq_T: int = 50
    dev_seq_D: int = 16
    prod_seq_T: int = 200
    prod_seq_D: int = 64
    dev_state_dim: int = 4
    dev_action_dim: int = 2
    dev_n_way: int = 5
    dev_k_shot: int = 1
    dev_q_queries: int = 15
    dev_num_classes: int = 10

    def validate(self) -> List[str]:
        errors = []
        if self.mode not in ("dev", "production"):
            errors.append(f"mode must be 'dev' or 'production', got '{self.mode}'")
        if self.batch_size < 1:
            errors.append(f"batch_size must be positive, got {self.batch_size}")
        if self.num_workers < 0:
            errors.append(f"num_workers must be non-negative, got {self.num_workers}")
        if not (1 <= self.phase <= 7):
            errors.append(f"phase must be 1-7, got {self.phase}")
        if self.val_ratio < 0 or self.val_ratio > 1:
            errors.append(f"val_ratio must be in [0,1], got {self.val_ratio}")
        if self.test_ratio < 0 or self.test_ratio > 1:
            errors.append(f"test_ratio must be in [0,1], got {self.test_ratio}")
        if self.val_ratio + self.test_ratio >= 1.0:
            errors.append("val_ratio + test_ratio must be < 1.0")
        if self.augmentation_strength not in ("none", "light", "standard", "heavy"):
            errors.append(f"augmentation_strength invalid: '{self.augmentation_strength}'")
        return errors


# ---------------------------------------------------------------------------
# SECTION 2: DatasetInfo (standalone definition for this template)
# ---------------------------------------------------------------------------

@dataclass
class DatasetInfo:
    """Metadata container for a dataset."""
    name: str
    phase: int
    modality: str
    num_classes: Optional[int] = None
    num_train_samples: int = 0
    num_val_samples: int = 0
    num_test_samples: int = 0
    input_shapes: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    target_shape: Optional[Tuple[int, ...]] = None
    dtype: str = "float32"
    target_dtype: str = "int64"
    description: str = ""
    source: str = "synthetic"
    version: str = "1.0.0"


# ---------------------------------------------------------------------------
# SECTION 3: Minimal BasePhaseLoader ABC stub for registration validation
# ---------------------------------------------------------------------------

from abc import ABC, abstractmethod

class BasePhaseLoader(ABC):
    """Minimal ABC stub for type-checking in the registry."""
    @abstractmethod
    def get_train_loader(self) -> DataLoader: ...
    @abstractmethod
    def get_val_loader(self) -> DataLoader: ...
    @abstractmethod
    def get_test_loader(self) -> DataLoader: ...
    @abstractmethod
    def get_dataset_info(self) -> DatasetInfo: ...


# ---------------------------------------------------------------------------
# SECTION 4: DatasetEntry
# ---------------------------------------------------------------------------

@dataclass
class DatasetEntry:
    """Internal registry entry containing full metadata for a dataset."""
    name: str
    loader_cls: Type[BasePhaseLoader]
    phase: int
    modality: str = "vision"
    description: str = ""
    url: Optional[str] = None
    version: str = "1.0.0"
    size_bytes: Optional[int] = None
    checksum: Optional[str] = None
    requires_auth: bool = False
    tags: List[str] = field(default_factory=list)

    def to_info(self) -> DatasetInfo:
        """Convert entry to public DatasetInfo."""
        return DatasetInfo(
            name=self.name,
            phase=self.phase,
            modality=self.modality,
            description=self.description,
            source="synthetic" if self.url is None else "remote",
            version=self.version,
        )


# ---------------------------------------------------------------------------
# SECTION 5: DatasetRegistry
# ---------------------------------------------------------------------------

VALID_MODALITIES = {"vision", "text", "audio", "sequence", "multimodal", "rl", "episodes"}


class DatasetRegistry:
    """Central registry for all datasets across training phases.

    Thread-safe singleton-pattern registry that maps dataset names to their
    loader classes, phase associations, and download metadata.
    """

    _instance: Optional[DatasetRegistry] = None
    _instance_lock = threading.Lock()

    def __init__(self, cache_dir: Optional[str] = None):
        self._lock = threading.Lock()
        self._entries: Dict[str, DatasetEntry] = {}
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/")

    @classmethod
    def get_instance(cls, cache_dir: Optional[str] = None) -> DatasetRegistry:
        """Get or create the singleton registry instance."""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(cache_dir=cache_dir)
            return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset the singleton (for testing)."""
        with cls._instance_lock:
            cls._instance = None

    def register(
        self,
        name: str,
        loader_cls: Type[BasePhaseLoader],
        phase: int,
        modality: str = "vision",
        description: str = "",
        url: Optional[str] = None,
        version: str = "1.0.0",
        size_bytes: Optional[int] = None,
        checksum: Optional[str] = None,
        requires_auth: bool = False,
        tags: Optional[List[str]] = None,
        force: bool = False,
    ) -> None:
        """Register a new dataset with the registry.

        Args:
            name: Unique dataset identifier.
            loader_cls: Class implementing BasePhaseLoader.
            phase: Training phase (1-7).
            modality: Primary modality.
            description: Human-readable description.
            url: Download URL (None for synthetic).
            version: Version string.
            size_bytes: Approximate download size.
            checksum: SHA-256 checksum.
            requires_auth: Whether download needs auth.
            tags: Searchable tags.
            force: Allow overwrite of existing entry.
        """
        if not name:
            raise ValueError("Dataset name must be non-empty")
        if not (1 <= phase <= 7):
            raise ValueError(f"phase must be 1-7, got {phase}")
        if modality not in VALID_MODALITIES:
            raise ValueError(f"modality must be one of {VALID_MODALITIES}, got '{modality}'")
        if not (isinstance(loader_cls, type) and issubclass(loader_cls, BasePhaseLoader)):
            raise TypeError(f"loader_cls must be a subclass of BasePhaseLoader, got {loader_cls}")

        entry = DatasetEntry(
            name=name,
            loader_cls=loader_cls,
            phase=phase,
            modality=modality,
            description=description,
            url=url,
            version=version,
            size_bytes=size_bytes,
            checksum=checksum,
            requires_auth=requires_auth,
            tags=tags or [],
        )

        with self._lock:
            if name in self._entries and not force:
                raise ValueError(f"Dataset '{name}' is already registered. Use force=True to overwrite.")
            self._entries[name] = entry
            logger.info(f"Registered dataset: {name} (phase={phase}, modality={modality})")

    def unregister(self, name: str) -> None:
        """Remove a dataset from the registry."""
        with self._lock:
            if name not in self._entries:
                raise KeyError(f"Dataset '{name}' not found in registry")
            del self._entries[name]

    def get_loader(
        self,
        name: str,
        mode: str = "dev",
        config: Optional[DataConfig] = None,
    ) -> BasePhaseLoader:
        """Retrieve and instantiate a loader for the named dataset."""
        with self._lock:
            if name not in self._entries:
                raise KeyError(f"Dataset '{name}' not found in registry. "
                               f"Available: {list(self._entries.keys())}")
            entry = self._entries[name]

        if config is None:
            config = DataConfig(phase=entry.phase, mode=mode)

        return entry.loader_cls(config, mode)

    def list_datasets(
        self,
        phase: Optional[int] = None,
        modality: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[DatasetInfo]:
        """List registered datasets with optional filtering."""
        with self._lock:
            entries = list(self._entries.values())

        results = []
        for entry in entries:
            if phase is not None and entry.phase != phase:
                continue
            if modality is not None and entry.modality != modality:
                continue
            if tags is not None and not all(t in entry.tags for t in tags):
                continue
            results.append(entry.to_info())

        return results

    def has_dataset(self, name: str) -> bool:
        """Check if a dataset is registered."""
        with self._lock:
            return name in self._entries

    def get_entry(self, name: str) -> DatasetEntry:
        """Get the full DatasetEntry (internal use)."""
        with self._lock:
            if name not in self._entries:
                raise KeyError(f"Dataset '{name}' not found in registry")
            return self._entries[name]

    def download(self, name: str, root: Optional[str] = None) -> Path:
        """Download a dataset to local storage."""
        entry = self.get_entry(name)
        target = Path(root) if root else self.cache_dir / "datasets" / name / f"v{entry.version}"

        if entry.url is None:
            logger.info(f"Dataset '{name}' is synthetic, no download needed")
            return target

        target.mkdir(parents=True, exist_ok=True)

        marker = target / ".download_complete"
        if marker.exists():
            logger.info(f"Dataset '{name}' already downloaded at {target}")
            return target

        logger.info(f"Would download '{name}' from {entry.url} to {target}")
        # Actual download logic would go here with retry/backoff
        # For template purposes, we just create the marker
        marker.touch()
        return target

    def clear_cache(self, name: Optional[str] = None) -> None:
        """Clear cached dataset files."""
        import shutil
        if name:
            path = self.cache_dir / "datasets" / name
            if path.exists():
                shutil.rmtree(path)
                logger.info(f"Cleared cache for '{name}'")
        else:
            path = self.cache_dir / "datasets"
            if path.exists():
                shutil.rmtree(path)
                logger.info("Cleared all dataset caches")

    def get_cache_size(self) -> int:
        """Get total cache size in bytes."""
        total = 0
        cache_path = self.cache_dir / "datasets"
        if cache_path.exists():
            for f in cache_path.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
        return total

    def save(self, path: str) -> None:
        """Save registry metadata to JSON."""
        with self._lock:
            data = {
                "version": "1.0.0",
                "entries": {},
            }
            for name, entry in self._entries.items():
                data["entries"][name] = {
                    "name": entry.name,
                    "phase": entry.phase,
                    "modality": entry.modality,
                    "description": entry.description,
                    "url": entry.url,
                    "version": entry.version,
                    "size_bytes": entry.size_bytes,
                    "checksum": entry.checksum,
                    "requires_auth": entry.requires_auth,
                    "tags": entry.tags,
                    "loader_cls": f"{entry.loader_cls.__module__}.{entry.loader_cls.__name__}",
                }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __contains__(self, name: str) -> bool:
        return self.has_dataset(name)

    def register_dataset(self, name: str, phase: int, modality: str = "vision", **kwargs):
        """Decorator for class-based registration."""
        def decorator(cls):
            self.register(name, cls, phase, modality=modality, **kwargs)
            return cls
        return decorator


# ---------------------------------------------------------------------------
# SECTION 6: Configuration presets
# ---------------------------------------------------------------------------

def dev_config(phase: int = 1, batch_size: int = 32) -> DataConfig:
    """Create a dev-mode DataConfig preset."""
    return DataConfig(
        phase=phase,
        mode="dev",
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        dev_num_samples=500,
        augmentation_strength="light",
    )


def production_config(phase: int = 1, batch_size: int = 64) -> DataConfig:
    """Create a production-mode DataConfig preset."""
    return DataConfig(
        phase=phase,
        mode="production",
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        augmentation_strength="standard",
    )


def minimal_config(phase: int = 1) -> DataConfig:
    """Create a minimal config for unit tests."""
    return DataConfig(
        phase=phase,
        mode="dev",
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        dev_num_samples=100,
        augmentation_strength="none",
    )


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

    # ---- DataConfig validation tests ----

    def test_config_valid():
        cfg = DataConfig()
        assert len(cfg.validate()) == 0
    run_test("DataConfig valid default", test_config_valid)

    def test_config_bad_mode():
        cfg = DataConfig(mode="test")
        assert len(cfg.validate()) > 0
    run_test("DataConfig bad mode", test_config_bad_mode)

    def test_config_bad_phase():
        cfg = DataConfig(phase=8)
        assert len(cfg.validate()) > 0
    run_test("DataConfig bad phase", test_config_bad_phase)

    def test_config_bad_batch():
        cfg = DataConfig(batch_size=-1)
        assert len(cfg.validate()) > 0
    run_test("DataConfig bad batch_size", test_config_bad_batch)

    # ---- DatasetInfo tests ----

    def test_info_creation():
        info = DatasetInfo(name="test", phase=1, modality="vision")
        assert info.name == "test"
        assert info.phase == 1
    run_test("DatasetInfo creation", test_info_creation)

    # ---- DatasetEntry tests ----

    class _DummyLoader(BasePhaseLoader):
        def __init__(self, config=None, mode="dev"):
            self.config = config
            self.mode = mode
        def get_train_loader(self): return None
        def get_val_loader(self): return None
        def get_test_loader(self): return None
        def get_dataset_info(self):
            return DatasetInfo(name="dummy", phase=1, modality="vision")

    def test_entry_to_info():
        entry = DatasetEntry(name="e", loader_cls=_DummyLoader, phase=1)
        info = entry.to_info()
        assert info.name == "e"
        assert info.phase == 1
    run_test("DatasetEntry to_info", test_entry_to_info)

    # ---- DatasetRegistry core tests ----

    def test_registry_register():
        reg = DatasetRegistry()
        reg.register("test1", _DummyLoader, phase=1)
        assert "test1" in reg
    run_test("Registry register", test_registry_register)

    def test_registry_register_duplicate():
        reg = DatasetRegistry()
        reg.register("dup", _DummyLoader, phase=1)
        try:
            reg.register("dup", _DummyLoader, phase=1)
            assert False, "Should raise ValueError"
        except ValueError:
            pass
    run_test("Registry duplicate raises", test_registry_register_duplicate)

    def test_registry_register_force():
        reg = DatasetRegistry()
        reg.register("force_test", _DummyLoader, phase=1)
        reg.register("force_test", _DummyLoader, phase=2, force=True)
        entry = reg.get_entry("force_test")
        assert entry.phase == 2
    run_test("Registry register force", test_registry_register_force)

    def test_registry_invalid_phase():
        reg = DatasetRegistry()
        try:
            reg.register("bad_phase", _DummyLoader, phase=0)
            assert False, "Should raise ValueError"
        except ValueError:
            pass
    run_test("Registry invalid phase", test_registry_invalid_phase)

    def test_registry_invalid_modality():
        reg = DatasetRegistry()
        try:
            reg.register("bad_mod", _DummyLoader, phase=1, modality="smell")
            assert False, "Should raise ValueError"
        except ValueError:
            pass
    run_test("Registry invalid modality", test_registry_invalid_modality)

    def test_registry_non_loader_class():
        reg = DatasetRegistry()
        try:
            reg.register("bad_cls", str, phase=1)  # type: ignore
            assert False, "Should raise TypeError"
        except TypeError:
            pass
    run_test("Registry non-loader class", test_registry_non_loader_class)

    def test_registry_empty_name():
        reg = DatasetRegistry()
        try:
            reg.register("", _DummyLoader, phase=1)
            assert False, "Should raise ValueError"
        except ValueError:
            pass
    run_test("Registry empty name", test_registry_empty_name)

    def test_registry_get_loader():
        reg = DatasetRegistry()
        reg.register("get_test", _DummyLoader, phase=1)
        loader = reg.get_loader("get_test", mode="dev")
        assert isinstance(loader, _DummyLoader)
        assert loader.mode == "dev"
    run_test("Registry get_loader", test_registry_get_loader)

    def test_registry_get_loader_unknown():
        reg = DatasetRegistry()
        try:
            reg.get_loader("nonexistent")
            assert False, "Should raise KeyError"
        except KeyError:
            pass
    run_test("Registry get_loader unknown", test_registry_get_loader_unknown)

    def test_registry_list_all():
        reg = DatasetRegistry()
        reg.register("l1", _DummyLoader, phase=1)
        reg.register("l2", _DummyLoader, phase=2)
        reg.register("l3", _DummyLoader, phase=3)
        result = reg.list_datasets()
        assert len(result) == 3
    run_test("Registry list all", test_registry_list_all)

    def test_registry_list_by_phase():
        reg = DatasetRegistry()
        reg.register("p1a", _DummyLoader, phase=1)
        reg.register("p1b", _DummyLoader, phase=1)
        reg.register("p2a", _DummyLoader, phase=2)
        result = reg.list_datasets(phase=1)
        assert len(result) == 2
        assert all(d.phase == 1 for d in result)
    run_test("Registry list by phase", test_registry_list_by_phase)

    def test_registry_list_by_modality():
        reg = DatasetRegistry()
        reg.register("v1", _DummyLoader, phase=1, modality="vision")
        reg.register("t1", _DummyLoader, phase=1, modality="text")
        result = reg.list_datasets(modality="text")
        assert len(result) == 1
        assert result[0].modality == "text"
    run_test("Registry list by modality", test_registry_list_by_modality)

    def test_registry_list_by_tags():
        reg = DatasetRegistry()
        reg.register("tagged", _DummyLoader, phase=1, tags=["dev", "small"])
        reg.register("untagged", _DummyLoader, phase=1, tags=["prod"])
        result = reg.list_datasets(tags=["dev"])
        assert len(result) == 1
        assert result[0].name == "tagged"
    run_test("Registry list by tags", test_registry_list_by_tags)

    def test_registry_list_empty_phase():
        reg = DatasetRegistry()
        reg.register("x", _DummyLoader, phase=1)
        result = reg.list_datasets(phase=7)
        assert len(result) == 0
    run_test("Registry list empty phase", test_registry_list_empty_phase)

    def test_registry_has_dataset():
        reg = DatasetRegistry()
        reg.register("exists", _DummyLoader, phase=1)
        assert reg.has_dataset("exists")
        assert not reg.has_dataset("nope")
    run_test("Registry has_dataset", test_registry_has_dataset)

    def test_registry_unregister():
        reg = DatasetRegistry()
        reg.register("unreg", _DummyLoader, phase=1)
        reg.unregister("unreg")
        assert not reg.has_dataset("unreg")
    run_test("Registry unregister", test_registry_unregister)

    def test_registry_unregister_unknown():
        reg = DatasetRegistry()
        try:
            reg.unregister("nope")
            assert False, "Should raise KeyError"
        except KeyError:
            pass
    run_test("Registry unregister unknown", test_registry_unregister_unknown)

    def test_registry_len():
        reg = DatasetRegistry()
        assert len(reg) == 0
        reg.register("a", _DummyLoader, phase=1)
        assert len(reg) == 1
    run_test("Registry __len__", test_registry_len)

    def test_registry_contains():
        reg = DatasetRegistry()
        reg.register("c", _DummyLoader, phase=1)
        assert "c" in reg
        assert "d" not in reg
    run_test("Registry __contains__", test_registry_contains)

    def test_registry_save():
        reg = DatasetRegistry()
        reg.register("save_test", _DummyLoader, phase=1, tags=["test"])
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            reg.save(path)
            with open(path) as f:
                data = json.load(f)
            assert "save_test" in data["entries"]
            assert data["entries"]["save_test"]["phase"] == 1
        finally:
            os.unlink(path)
    run_test("Registry save", test_registry_save)

    def test_registry_decorator():
        reg = DatasetRegistry()

        @reg.register_dataset("deco_test", phase=2, modality="text")
        class DecoLoader(_DummyLoader):
            pass

        assert "deco_test" in reg
        entry = reg.get_entry("deco_test")
        assert entry.phase == 2
        assert entry.modality == "text"
    run_test("Registry decorator", test_registry_decorator)

    def test_registry_download_synthetic():
        reg = DatasetRegistry()
        reg.register("synth_dl", _DummyLoader, phase=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = reg.download("synth_dl", root=tmpdir)
            assert isinstance(path, Path)
    run_test("Registry download synthetic", test_registry_download_synthetic)

    # ---- Config presets ----

    def test_dev_config():
        cfg = dev_config(phase=3, batch_size=16)
        assert cfg.mode == "dev"
        assert cfg.phase == 3
        assert cfg.batch_size == 16
        assert len(cfg.validate()) == 0
    run_test("dev_config preset", test_dev_config)

    def test_production_config():
        cfg = production_config(phase=5)
        assert cfg.mode == "production"
        assert cfg.num_workers == 4
        assert cfg.pin_memory is True
    run_test("production_config preset", test_production_config)

    def test_minimal_config():
        cfg = minimal_config()
        assert cfg.batch_size == 4
        assert cfg.augmentation_strength == "none"
    run_test("minimal_config preset", test_minimal_config)

    # ---- Singleton tests ----

    def test_singleton():
        DatasetRegistry.reset_instance()
        r1 = DatasetRegistry.get_instance()
        r2 = DatasetRegistry.get_instance()
        assert r1 is r2
        DatasetRegistry.reset_instance()
    run_test("Singleton pattern", test_singleton)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print(f"DATA CONFIG & REGISTRY SELF-TESTS: {passed} passed, {failed} failed")
    print("=" * 60)
    for result in test_results:
        status = result[0]
        name = result[1]
        extra = f" -- {result[2]}" if len(result) > 2 else ""
        print(f"  [{status}] {name}{extra}")

    sys.exit(0 if failed == 0 else 1)
