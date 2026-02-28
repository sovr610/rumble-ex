"""
Dataset Fingerprinting System
=============================
Tiered dataset identity tracking for reproducible training across the
seven-phase brain_ai cognitive pipeline.

Tiers:
  1 - HuggingFace datasets  (metadata only, near-zero overhead)
  2 - Local files / shards   (fast hash: first 4 MB + last 4 MB SHA-256)
  3 - Subset identity        (parent fingerprint + sampling parameters)

All fingerprints are serialised to ``artifacts/datasets.json`` inside the
run directory.  This module is fully self-contained -- no brain_ai imports.

Usage:
    from dataset_fingerprint_template import DatasetFingerprinter, FastHash

    fp = DatasetFingerprinter(tier="auto")
    result = fp.fingerprint_auto(my_dataset, name="mnist", split="train")
    fp.save([result], "artifacts/datasets.json")
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Union

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "1.0"
FAST_HASH_CHUNK = 4 * 1024 * 1024       # 4 MB
FULL_HASH_CHUNK = 8 * 1024              # 8 KB streaming chunk
CACHE_FILENAME = ".dataset_fingerprint_cache.json"

# ---------------------------------------------------------------------------
# FastHash
# ---------------------------------------------------------------------------


class FastHash:
    """Lightweight file hashing utilities.

    ``hash_file_fast`` reads only the first 4 MB and last 4 MB of a file,
    yielding constant I/O regardless of file size (< 100 ms for typical
    shard files).
    """

    @staticmethod
    def hash_file_fast(
        path: str,
        head_bytes: int = FAST_HASH_CHUNK,
        tail_bytes: int = FAST_HASH_CHUNK,
    ) -> str:
        """SHA-256 of first *head_bytes* + last *tail_bytes*.

        For files smaller than ``head_bytes + tail_bytes`` the entire
        contents are hashed (no duplicate reads for overlapping regions).
        """
        file_size = os.path.getsize(path)
        h = hashlib.sha256()
        with open(path, "rb") as f:
            if file_size <= head_bytes:
                # Entire file fits in head -- read once.
                h.update(f.read())
            elif file_size <= head_bytes + tail_bytes:
                # File fits in head+tail but regions overlap -- read whole.
                h.update(f.read())
            else:
                h.update(f.read(head_bytes))
                f.seek(file_size - tail_bytes)
                h.update(f.read(tail_bytes))
        return h.hexdigest()

    @staticmethod
    def hash_file_full(path: str) -> str:
        """Full SHA-256 of the entire file (streaming, 8 KB chunks)."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(FULL_HASH_CHUNK)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def hash_string(s: str) -> str:
        """SHA-256 hex digest of a UTF-8 encoded string."""
        return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ShardInfo:
    """Identity of a single local data file (Tier 2)."""

    relative_path: str
    byte_size: int
    mtime: str           # ISO 8601
    fast_hash: str
    full_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "relative_path": self.relative_path,
            "byte_size": self.byte_size,
            "mtime": self.mtime,
            "fast_hash": self.fast_hash,
        }
        if self.full_hash is not None:
            d["full_hash"] = self.full_hash
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ShardInfo":
        return cls(
            relative_path=d["relative_path"],
            byte_size=d["byte_size"],
            mtime=d["mtime"],
            fast_hash=d["fast_hash"],
            full_hash=d.get("full_hash"),
        )


@dataclass
class SubsetInfo:
    """Sampling identity for a subset of a parent dataset (Tier 3)."""

    parent_fingerprint: str       # SHA-256 of parent DatasetFingerprint
    indices: Optional[List[int]]  # explicit indices for small subsets
    seed: int
    subset_size: int
    strategy: str                 # "random", "first_n", "stratified"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parent_fingerprint": self.parent_fingerprint,
            "indices": self.indices,
            "seed": self.seed,
            "subset_size": self.subset_size,
            "strategy": self.strategy,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SubsetInfo":
        return cls(
            parent_fingerprint=d["parent_fingerprint"],
            indices=d.get("indices"),
            seed=d["seed"],
            subset_size=d["subset_size"],
            strategy=d["strategy"],
        )


@dataclass
class DatasetFingerprint:
    """Unified fingerprint record for any dataset tier."""

    name: str
    role: str                                   # "primary", "validation", "meta_train"
    tier: str                                   # "tier1", "tier2", "tier3"
    split: str
    config_name: Optional[str] = None
    version: Optional[str] = None
    hf_fingerprint: Optional[str] = None        # Tier 1
    num_samples: int = 0
    features_hash: Optional[str] = None
    transforms_signature: Optional[str] = None
    shards: Optional[List[ShardInfo]] = None    # Tier 2
    subset_info: Optional[SubsetInfo] = None    # Tier 3

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "name": self.name,
            "role": self.role,
            "tier": self.tier,
            "split": self.split,
        }
        if self.config_name is not None:
            d["config_name"] = self.config_name
        if self.version is not None:
            d["version"] = self.version
        if self.hf_fingerprint is not None:
            d["hf_fingerprint"] = self.hf_fingerprint
        d["num_samples"] = self.num_samples
        if self.features_hash is not None:
            d["features_hash"] = self.features_hash
        if self.transforms_signature is not None:
            d["transforms_signature"] = self.transforms_signature
        if self.shards is not None:
            d["shards"] = [s.to_dict() for s in self.shards]
        if self.subset_info is not None:
            d["subset_info"] = self.subset_info.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DatasetFingerprint":
        shards = None
        if "shards" in d and d["shards"] is not None:
            shards = [ShardInfo.from_dict(s) for s in d["shards"]]
        subset_info = None
        if "subset_info" in d and d["subset_info"] is not None:
            subset_info = SubsetInfo.from_dict(d["subset_info"])
        return cls(
            name=d["name"],
            role=d["role"],
            tier=d["tier"],
            split=d["split"],
            config_name=d.get("config_name"),
            version=d.get("version"),
            hf_fingerprint=d.get("hf_fingerprint"),
            num_samples=d.get("num_samples", 0),
            features_hash=d.get("features_hash"),
            transforms_signature=d.get("transforms_signature"),
            shards=shards,
            subset_info=subset_info,
        )

    def content_hash(self) -> str:
        """Deterministic hash of the entire fingerprint for cross-referencing."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return FastHash.hash_string(canonical)


# ---------------------------------------------------------------------------
# TransformSignature
# ---------------------------------------------------------------------------


class TransformSignature:
    """Capture a deterministic string representation of a transform pipeline."""

    @staticmethod
    def capture(transforms: Any) -> str:
        """Return a deterministic string for *transforms*.

        - ``None`` -> ``"identity"``
        - ``torchvision.transforms.Compose`` -> recursive ``repr()``
        - Custom objects -> ``ClassName(sorted __init__ kwargs)``
        """
        if transforms is None:
            return "identity"
        # torchvision Compose: just use repr -- it recurses into children.
        if hasattr(transforms, "transforms"):
            parts = [repr(t) for t in transforms.transforms]
            return "Compose(" + ",".join(parts) + ")"
        # Fallback: repr of the object itself.
        return repr(transforms)

    @staticmethod
    def hash_signature(signature_str: str) -> str:
        """SHA-256 hex digest (first 16 chars) of *signature_str*."""
        full = FastHash.hash_string(signature_str)
        return f"sha256:{full[:16]}"

    @classmethod
    def capture_and_hash(cls, transforms: Any) -> tuple:
        """Return ``(readable_signature, short_hash)``."""
        sig = cls.capture(transforms)
        return sig, cls.hash_signature(sig)


# ---------------------------------------------------------------------------
# FingerprintComparator
# ---------------------------------------------------------------------------


@dataclass
class ComparisonResult:
    level: str       # "exact", "compatible", "incompatible"
    detail: str      # human-readable explanation


class FingerprintComparator:
    """Compare two DatasetFingerprint objects or two datasets.json files."""

    @staticmethod
    def compare(a: DatasetFingerprint, b: DatasetFingerprint) -> ComparisonResult:
        """Compare fingerprints *a* and *b*."""
        if a.name != b.name:
            return ComparisonResult("incompatible", f"name differs: '{a.name}' vs '{b.name}'")
        if a.split != b.split:
            return ComparisonResult("incompatible", f"split differs: '{a.split}' vs '{b.split}'")

        # Check for exact match across all fields.
        if a.to_dict() == b.to_dict():
            return ComparisonResult("exact", "all fields identical")

        # Tier-specific compatible checks.
        if a.tier == "tier1" and b.tier == "tier1":
            if a.hf_fingerprint and a.hf_fingerprint == b.hf_fingerprint:
                return ComparisonResult("exact", "HF fingerprints match")
            if a.version != b.version:
                return ComparisonResult(
                    "compatible",
                    f"version differs: {a.version} vs {b.version}",
                )
            return ComparisonResult("compatible", "same name/split, fingerprints differ")

        if a.tier == "tier2" and b.tier == "tier2":
            if a.shards is not None and b.shards is not None:
                map_a = {s.relative_path: s.fast_hash for s in a.shards}
                map_b = {s.relative_path: s.fast_hash for s in b.shards}
                if map_a == map_b:
                    return ComparisonResult("exact", "all shard hashes match")
                changed = [p for p in map_a if map_a.get(p) != map_b.get(p)]
                return ComparisonResult("compatible", f"{len(changed)} shard(s) differ")

        if a.tier == "tier3" and b.tier == "tier3":
            if (a.subset_info is not None and b.subset_info is not None
                    and a.subset_info.indices == b.subset_info.indices
                    and a.subset_info.seed == b.subset_info.seed):
                return ComparisonResult("exact", "subset indices and seed match")
            return ComparisonResult("compatible", "same parent, different subset params")

        return ComparisonResult("compatible", f"cross-tier ({a.tier} vs {b.tier})")

    @classmethod
    def compare_datasets_json(cls, path_a: str, path_b: str) -> List[ComparisonResult]:
        """Compare two ``datasets.json`` files entry-by-entry."""
        fps_a = DatasetsJSON.load(path_a)
        fps_b = DatasetsJSON.load(path_b)
        results: List[ComparisonResult] = []
        max_len = max(len(fps_a), len(fps_b))
        for i in range(max_len):
            if i >= len(fps_a):
                results.append(ComparisonResult("incompatible", f"dataset[{i}] missing in A"))
            elif i >= len(fps_b):
                results.append(ComparisonResult("incompatible", f"dataset[{i}] missing in B"))
            else:
                results.append(cls.compare(fps_a[i], fps_b[i]))
        return results


# ---------------------------------------------------------------------------
# FingerprintCache
# ---------------------------------------------------------------------------


class FingerprintCache:
    """On-disk cache for file hashes, stored next to the data directory.

    Cache key: ``(file_path, mtime, size)``.  Invalidates automatically
    when ``mtime`` changes.
    """

    def __init__(self, data_dir: str) -> None:
        self.data_dir = os.path.abspath(data_dir)
        self.cache_path = os.path.join(self.data_dir, CACHE_FILENAME)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r") as f:
                self._cache = json.load(f)

    def _save(self) -> None:
        with open(self.cache_path, "w") as f:
            json.dump(self._cache, f, indent=2)

    def get_or_compute(self, path: str, hash_fn: Callable[[str], str]) -> str:
        """Return cached hash or compute via *hash_fn*, cache, and return."""
        abs_path = os.path.abspath(path)
        stat = os.stat(abs_path)
        key = abs_path
        cached = self._cache.get(key)
        if cached is not None:
            if (cached.get("mtime") == stat.st_mtime
                    and cached.get("size") == stat.st_size):
                return cached["hash"]
        # Compute and store.
        h = hash_fn(abs_path)
        self._cache[key] = {
            "mtime": stat.st_mtime,
            "size": stat.st_size,
            "hash": h,
        }
        self._save()
        return h

    def clear(self) -> None:
        """Remove the cache file and reset in-memory state."""
        self._cache = {}
        if os.path.exists(self.cache_path):
            os.remove(self.cache_path)


# ---------------------------------------------------------------------------
# DatasetsJSON schema helpers
# ---------------------------------------------------------------------------


class DatasetsJSON:
    """Read / write / validate the ``datasets.json`` artefact."""

    VALID_TIERS = {"tier1", "tier2", "tier3", "mixed"}

    @staticmethod
    def validate(data: Dict[str, Any]) -> List[str]:
        """Return a list of validation errors (empty == valid)."""
        errors: List[str] = []
        if data.get("schema_version") != SCHEMA_VERSION:
            errors.append(
                f"schema_version must be '{SCHEMA_VERSION}', "
                f"got '{data.get('schema_version')}'"
            )
        tier = data.get("fingerprint_tier", "")
        if tier not in DatasetsJSON.VALID_TIERS:
            errors.append(f"fingerprint_tier '{tier}' not in {DatasetsJSON.VALID_TIERS}")
        if not isinstance(data.get("datasets"), list):
            errors.append("'datasets' must be a list")
        else:
            for i, ds in enumerate(data["datasets"]):
                if "name" not in ds:
                    errors.append(f"datasets[{i}] missing 'name'")
                if "role" not in ds:
                    errors.append(f"datasets[{i}] missing 'role'")
                if "tier" not in ds:
                    errors.append(f"datasets[{i}] missing 'tier'")
        if "timestamp" not in data:
            errors.append("missing 'timestamp'")
        return errors

    @staticmethod
    def save(fingerprints: List[DatasetFingerprint], output_path: str) -> None:
        """Write *fingerprints* to *output_path* as ``datasets.json``."""
        tiers = {fp.tier for fp in fingerprints}
        tier_label = tiers.pop() if len(tiers) == 1 else "mixed"
        payload = {
            "schema_version": SCHEMA_VERSION,
            "fingerprint_tier": tier_label,
            "datasets": [fp.to_dict() for fp in fingerprints],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def load(path: str) -> List[DatasetFingerprint]:
        """Load ``datasets.json`` and return a list of DatasetFingerprint."""
        with open(path, "r") as f:
            data = json.load(f)
        errors = DatasetsJSON.validate(data)
        if errors:
            raise ValueError(f"Invalid datasets.json: {'; '.join(errors)}")
        return [DatasetFingerprint.from_dict(d) for d in data["datasets"]]


# ---------------------------------------------------------------------------
# DatasetFingerprinter
# ---------------------------------------------------------------------------


class DatasetFingerprinter:
    """Main entry point for fingerprinting datasets across all tiers.

    Parameters
    ----------
    tier : str
        ``"auto"`` (default) inspects the input to choose the tier.
        ``"tier1"``, ``"tier2"``, ``"tier3"`` force a specific tier.
    """

    def __init__(self, tier: str = "auto") -> None:
        if tier not in ("auto", "tier1", "tier2", "tier3"):
            raise ValueError(f"Invalid tier: {tier!r}")
        self.tier = tier

    # -- Tier 1: HuggingFace ------------------------------------------------

    def fingerprint_hf_dataset(
        self,
        dataset: Any,
        name: str,
        config_name: Optional[str] = None,
        split: str = "train",
        role: str = "primary",
        transforms: Any = None,
    ) -> DatasetFingerprint:
        """Tier 1 fingerprint from a HuggingFace ``datasets.Dataset``."""
        info = getattr(dataset, "info", None)
        version = None
        features_hash = None
        hf_fingerprint = getattr(dataset, "_fingerprint", None)

        if info is not None:
            version_obj = getattr(info, "version", None)
            version = str(version_obj) if version_obj else None
            if config_name is None:
                config_name = getattr(info, "config_name", None)
            features = getattr(info, "features", None)
            if features is not None:
                features_hash = f"sha256:{FastHash.hash_string(str(features))}"

        num_samples = len(dataset) if hasattr(dataset, "__len__") else 0
        sig, _ = TransformSignature.capture_and_hash(transforms)

        return DatasetFingerprint(
            name=name,
            role=role,
            tier="tier1",
            split=split,
            config_name=config_name,
            version=version,
            hf_fingerprint=hf_fingerprint,
            num_samples=num_samples,
            features_hash=features_hash,
            transforms_signature=sig if sig != "identity" else None,
        )

    # -- Tier 2: Local files ------------------------------------------------

    def fingerprint_local_files(
        self,
        paths: List[str],
        name: str = "local_dataset",
        split: str = "train",
        role: str = "primary",
        root: Optional[str] = None,
        compute_full_hash: bool = False,
        transforms: Any = None,
        cache: Optional[FingerprintCache] = None,
    ) -> DatasetFingerprint:
        """Tier 2 fingerprint from a list of local file paths.

        Parameters
        ----------
        paths : list of str
            Absolute or relative paths to dataset files.
        root : str, optional
            Base directory for computing relative paths.  If ``None``, the
            common prefix of *paths* is used.
        compute_full_hash : bool
            If ``True``, also compute the full SHA-256 for each file.
        cache : FingerprintCache, optional
            Optional cache to avoid rehashing unchanged files.
        """
        if not paths:
            raise ValueError("paths must be a non-empty list")

        abs_paths = [os.path.abspath(p) for p in paths]
        if root is None:
            root = os.path.commonpath(abs_paths) if len(abs_paths) > 1 else os.path.dirname(abs_paths[0])
        root = os.path.abspath(root)

        shards: List[ShardInfo] = []
        for abs_p in sorted(abs_paths):
            rel = os.path.relpath(abs_p, root)
            stat = os.stat(abs_p)
            mtime_iso = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()

            if cache is not None:
                fh = cache.get_or_compute(abs_p, FastHash.hash_file_fast)
            else:
                fh = FastHash.hash_file_fast(abs_p)

            full_h = None
            if compute_full_hash:
                if cache is not None:
                    full_h = cache.get_or_compute(abs_p + ":full", FastHash.hash_file_full)
                else:
                    full_h = FastHash.hash_file_full(abs_p)

            shards.append(ShardInfo(
                relative_path=rel,
                byte_size=stat.st_size,
                mtime=mtime_iso,
                fast_hash=f"sha256:{fh}",
                full_hash=f"sha256:{full_h}" if full_h else None,
            ))

        num_samples = sum(s.byte_size for s in shards)  # byte-count proxy
        sig, _ = TransformSignature.capture_and_hash(transforms)

        return DatasetFingerprint(
            name=name,
            role=role,
            tier="tier2",
            split=split,
            num_samples=num_samples,
            shards=shards,
            transforms_signature=sig if sig != "identity" else None,
        )

    # -- Tier 3: Subset -----------------------------------------------------

    def fingerprint_subset(
        self,
        parent_fingerprint: DatasetFingerprint,
        indices: List[int],
        seed: int,
        strategy: str = "random",
        name: Optional[str] = None,
        role: str = "primary",
        split: Optional[str] = None,
    ) -> DatasetFingerprint:
        """Tier 3 fingerprint referencing a parent dataset + subset params."""
        parent_hash = parent_fingerprint.content_hash()
        store_indices: Optional[List[int]] = None
        if len(indices) <= 10_000:
            store_indices = sorted(indices)

        subset = SubsetInfo(
            parent_fingerprint=parent_hash,
            indices=store_indices,
            seed=seed,
            subset_size=len(indices),
            strategy=strategy,
        )

        return DatasetFingerprint(
            name=name or f"{parent_fingerprint.name}_subset",
            role=role,
            tier="tier3",
            split=split or parent_fingerprint.split,
            num_samples=len(indices),
            subset_info=subset,
            transforms_signature=parent_fingerprint.transforms_signature,
        )

    # -- Auto dispatch ------------------------------------------------------

    def fingerprint_auto(
        self,
        dataset_or_paths: Any,
        **kwargs: Any,
    ) -> DatasetFingerprint:
        """Detect the input type and dispatch to the appropriate tier.

        - If *dataset_or_paths* has an ``info`` attribute and
          ``_fingerprint`` attribute, treat as HF dataset (Tier 1).
        - If *dataset_or_paths* is a list of strings (paths), use Tier 2.
        - If ``parent_fingerprint`` is in *kwargs*, use Tier 3.
        """
        forced_tier = self.tier if self.tier != "auto" else None

        # Tier 3: explicit parent reference.
        if "parent_fingerprint" in kwargs or forced_tier == "tier3":
            parent = kwargs.pop("parent_fingerprint")
            indices = kwargs.pop("indices", [])
            seed = kwargs.pop("seed", 0)
            strategy = kwargs.pop("strategy", "random")
            return self.fingerprint_subset(
                parent_fingerprint=parent,
                indices=indices,
                seed=seed,
                strategy=strategy,
                **kwargs,
            )

        # Tier 1: HuggingFace Dataset.
        if forced_tier == "tier1" or (
            forced_tier is None
            and hasattr(dataset_or_paths, "info")
            and hasattr(dataset_or_paths, "_fingerprint")
        ):
            name = kwargs.pop("name", "hf_dataset")
            split = kwargs.pop("split", "train")
            config_name = kwargs.pop("config_name", None)
            role = kwargs.pop("role", "primary")
            transforms = kwargs.pop("transforms", None)
            return self.fingerprint_hf_dataset(
                dataset_or_paths,
                name=name,
                config_name=config_name,
                split=split,
                role=role,
                transforms=transforms,
            )

        # Tier 2: local file paths.
        if forced_tier == "tier2" or isinstance(dataset_or_paths, list):
            return self.fingerprint_local_files(dataset_or_paths, **kwargs)

        raise TypeError(
            f"Cannot auto-detect tier for {type(dataset_or_paths).__name__}. "
            "Pass tier='tier1'/'tier2'/'tier3' explicitly."
        )

    # -- Persistence shortcuts -----------------------------------------------

    @staticmethod
    def save(fingerprints: List[DatasetFingerprint], output_path: str) -> None:
        """Write fingerprints to *output_path* as ``datasets.json``."""
        DatasetsJSON.save(fingerprints, output_path)

    @staticmethod
    def load(path: str) -> List[DatasetFingerprint]:
        """Load fingerprints from *path*."""
        return DatasetsJSON.load(path)


# =========================================================================
# Self-test
# =========================================================================

if __name__ == "__main__":
    import sys
    import tempfile
    import struct
    import traceback

    _pass = 0
    _fail = 0
    _errors: List[str] = []

    def check(condition: bool, label: str) -> None:
        global _pass, _fail
        if condition:
            _pass += 1
            print(f"  [PASS] {label}")
        else:
            _fail += 1
            _errors.append(label)
            print(f"  [FAIL] {label}")

    def section(title: str) -> None:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")

    # ------------------------------------------------------------------
    # Helper: create temp files of specific sizes
    # ------------------------------------------------------------------

    def make_temp_file(size: int, content_byte: int = 0xAB,
                       tmpdir: Optional[str] = None) -> str:
        """Create a temp file filled with *content_byte* of *size* bytes."""
        fd, path = tempfile.mkstemp(dir=tmpdir, suffix=".bin")
        with os.fdopen(fd, "wb") as f:
            remaining = size
            chunk_size = min(size, 1024 * 1024)
            chunk = bytes([content_byte]) * chunk_size
            while remaining > 0:
                write_size = min(remaining, chunk_size)
                f.write(chunk[:write_size])
                remaining -= write_size
        return path

    # ==================================================================
    # 1. FastHash tests
    # ==================================================================
    section("FastHash")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Small file (< 4 MB)
        small_path = make_temp_file(1000, 0x01, tmpdir)
        fh_small = FastHash.hash_file_fast(small_path)
        full_small = FastHash.hash_file_full(small_path)
        check(len(fh_small) == 64, "fast hash returns 64-char hex (small)")
        check(fh_small == full_small,
              "fast hash == full hash for small file (< 4 MB)")

        # Medium file (between 4 MB and 8 MB -- overlap region)
        medium_size = 6 * 1024 * 1024  # 6 MB
        medium_path = make_temp_file(medium_size, 0x02, tmpdir)
        fh_med = FastHash.hash_file_fast(medium_path)
        full_med = FastHash.hash_file_full(medium_path)
        check(len(fh_med) == 64, "fast hash returns 64-char hex (medium)")
        check(fh_med == full_med,
              "fast hash == full hash for medium file (overlap region)")

        # Large file (> 8 MB)
        large_size = 12 * 1024 * 1024  # 12 MB
        large_path = make_temp_file(large_size, 0x03, tmpdir)
        fh_large = FastHash.hash_file_fast(large_path)
        full_large = FastHash.hash_file_full(large_path)
        check(len(fh_large) == 64, "fast hash returns 64-char hex (large)")
        # Fast hash should differ from full for large files with uniform content
        # only if head+tail is a strict subset.  For uniform content they would
        # still match because all bytes are 0x03.  Use heterogeneous content:
        hetero_path = os.path.join(tmpdir, "hetero.bin")
        with open(hetero_path, "wb") as hf:
            for i in range(large_size):
                hf.write(struct.pack("B", i % 256))
        fh_hetero = FastHash.hash_file_fast(hetero_path)
        full_hetero = FastHash.hash_file_full(hetero_path)
        check(fh_hetero != full_hetero,
              "fast hash != full hash for large heterogeneous file")

        # hash_string
        hs = FastHash.hash_string("hello world")
        check(len(hs) == 64, "hash_string returns 64-char hex")
        check(hs == FastHash.hash_string("hello world"),
              "hash_string is deterministic")
        check(hs != FastHash.hash_string("hello world!"),
              "hash_string differs for different input")

        # Very small file (0 bytes)
        empty_path = make_temp_file(0, 0x00, tmpdir)
        fh_empty = FastHash.hash_file_fast(empty_path)
        full_empty = FastHash.hash_file_full(empty_path)
        check(fh_empty == full_empty,
              "fast hash == full hash for empty file")

        # Exactly 4 MB file
        exact_4mb = make_temp_file(FAST_HASH_CHUNK, 0x04, tmpdir)
        fh_4mb = FastHash.hash_file_fast(exact_4mb)
        full_4mb = FastHash.hash_file_full(exact_4mb)
        check(fh_4mb == full_4mb,
              "fast hash == full hash for exactly 4 MB file")

        # Exactly 8 MB file
        exact_8mb = make_temp_file(2 * FAST_HASH_CHUNK, 0x05, tmpdir)
        fh_8mb = FastHash.hash_file_fast(exact_8mb)
        full_8mb = FastHash.hash_file_full(exact_8mb)
        check(fh_8mb == full_8mb,
              "fast hash == full hash for exactly 8 MB file")

    # ==================================================================
    # 2. ShardInfo dataclass
    # ==================================================================
    section("ShardInfo")

    si = ShardInfo(
        relative_path="shard_000.tar",
        byte_size=1024,
        mtime="2026-01-15T08:30:00+00:00",
        fast_hash="sha256:aabbcc",
        full_hash="sha256:ddeeff",
    )
    si_dict = si.to_dict()
    check(si_dict["relative_path"] == "shard_000.tar", "ShardInfo.to_dict relative_path")
    check(si_dict["full_hash"] == "sha256:ddeeff", "ShardInfo.to_dict full_hash present")
    si2 = ShardInfo.from_dict(si_dict)
    check(si2.relative_path == si.relative_path, "ShardInfo round-trip relative_path")
    check(si2.full_hash == si.full_hash, "ShardInfo round-trip full_hash")

    si_no_full = ShardInfo(
        relative_path="shard.bin", byte_size=10, mtime="2026-01-01T00:00:00+00:00",
        fast_hash="sha256:abc",
    )
    check("full_hash" not in si_no_full.to_dict(),
          "ShardInfo.to_dict omits None full_hash")

    # ==================================================================
    # 3. SubsetInfo dataclass
    # ==================================================================
    section("SubsetInfo")

    subi = SubsetInfo(
        parent_fingerprint="sha256:parent123",
        indices=[0, 5, 10, 15],
        seed=42,
        subset_size=4,
        strategy="random",
    )
    subi_dict = subi.to_dict()
    check(subi_dict["seed"] == 42, "SubsetInfo.to_dict seed")
    subi2 = SubsetInfo.from_dict(subi_dict)
    check(subi2.indices == [0, 5, 10, 15], "SubsetInfo round-trip indices")
    check(subi2.strategy == "random", "SubsetInfo round-trip strategy")

    # ==================================================================
    # 4. DatasetFingerprint dataclass
    # ==================================================================
    section("DatasetFingerprint")

    fp = DatasetFingerprint(
        name="mnist", role="primary", tier="tier1", split="train",
        version="1.0.0", hf_fingerprint="hf_abc123", num_samples=60000,
        features_hash="sha256:feat_hash",
    )
    fp_dict = fp.to_dict()
    check(fp_dict["name"] == "mnist", "DatasetFingerprint.to_dict name")
    check(fp_dict["tier"] == "tier1", "DatasetFingerprint.to_dict tier")
    check("shards" not in fp_dict, "DatasetFingerprint.to_dict omits None shards")
    fp_rt = DatasetFingerprint.from_dict(fp_dict)
    check(fp_rt.name == "mnist", "DatasetFingerprint round-trip name")
    check(fp_rt.hf_fingerprint == "hf_abc123", "DatasetFingerprint round-trip hf_fingerprint")
    check(fp_rt.shards is None, "DatasetFingerprint round-trip None shards")

    # content_hash determinism
    ch1 = fp.content_hash()
    ch2 = fp.content_hash()
    check(ch1 == ch2, "content_hash is deterministic")
    fp_alt = DatasetFingerprint(
        name="cifar10", role="primary", tier="tier1", split="train",
        num_samples=50000,
    )
    check(fp.content_hash() != fp_alt.content_hash(),
          "content_hash differs for different fingerprints")

    # With shards
    fp_t2 = DatasetFingerprint(
        name="local", role="primary", tier="tier2", split="train",
        num_samples=100,
        shards=[si],
    )
    fp_t2_dict = fp_t2.to_dict()
    check(len(fp_t2_dict["shards"]) == 1, "DatasetFingerprint.to_dict includes shards")
    fp_t2_rt = DatasetFingerprint.from_dict(fp_t2_dict)
    check(fp_t2_rt.shards is not None and len(fp_t2_rt.shards) == 1,
          "DatasetFingerprint round-trip shards")

    # With subset_info
    fp_t3 = DatasetFingerprint(
        name="mnist_sub", role="primary", tier="tier3", split="train",
        num_samples=100, subset_info=subi,
    )
    fp_t3_dict = fp_t3.to_dict()
    check("subset_info" in fp_t3_dict, "DatasetFingerprint.to_dict includes subset_info")
    fp_t3_rt = DatasetFingerprint.from_dict(fp_t3_dict)
    check(fp_t3_rt.subset_info is not None, "DatasetFingerprint round-trip subset_info")
    check(fp_t3_rt.subset_info.seed == 42, "DatasetFingerprint round-trip subset_info.seed")

    # ==================================================================
    # 5. TransformSignature
    # ==================================================================
    section("TransformSignature")

    sig_none = TransformSignature.capture(None)
    check(sig_none == "identity", "capture(None) -> 'identity'")

    sig_hash_none = TransformSignature.hash_signature("identity")
    check(sig_hash_none.startswith("sha256:"), "hash_signature starts with sha256:")
    check(len(sig_hash_none) == 7 + 16, "hash_signature is sha256: + 16 hex chars")

    # Mock Compose
    class MockTransform:
        def __init__(self, name: str, val: float):
            self.name = name
            self.val = val
        def __repr__(self) -> str:
            return f"{self.name}({self.val})"

    class MockCompose:
        def __init__(self, transforms: list):
            self.transforms = transforms

    compose = MockCompose([MockTransform("Resize", 224), MockTransform("Normalize", 0.5)])
    sig_compose = TransformSignature.capture(compose)
    check("Resize(224)" in sig_compose, "capture(Compose) includes Resize")
    check("Normalize(0.5)" in sig_compose, "capture(Compose) includes Normalize")
    check(sig_compose.startswith("Compose("), "capture(Compose) starts with Compose(")

    sig1, h1 = TransformSignature.capture_and_hash(compose)
    sig2, h2 = TransformSignature.capture_and_hash(compose)
    check(sig1 == sig2, "capture_and_hash deterministic signature")
    check(h1 == h2, "capture_and_hash deterministic hash")

    # Plain object fallback
    sig_plain = TransformSignature.capture(MockTransform("Custom", 3.14))
    check("Custom(3.14)" in sig_plain, "capture(plain object) uses repr")

    # ==================================================================
    # 6. Tier 1 fingerprinting with mock HF dataset
    # ==================================================================
    section("Tier 1: HuggingFace fingerprinting")

    class MockHFInfo:
        def __init__(self):
            self.dataset_name = "mnist"
            self.config_name = "default"
            self.version = "1.0.0"
            self.features = {"image": "Image", "label": "ClassLabel(num_classes=10)"}

    class MockHFDataset:
        def __init__(self, size: int = 60000):
            self.info = MockHFInfo()
            self._fingerprint = "hf_fp_abc123def456"
            self._size = size
        def __len__(self) -> int:
            return self._size

    mock_ds = MockHFDataset()
    fper = DatasetFingerprinter(tier="auto")

    fp_hf = fper.fingerprint_hf_dataset(mock_ds, name="mnist", split="train")
    check(fp_hf.tier == "tier1", "HF fingerprint tier is tier1")
    check(fp_hf.name == "mnist", "HF fingerprint name")
    check(fp_hf.hf_fingerprint == "hf_fp_abc123def456", "HF fingerprint hf_fingerprint")
    check(fp_hf.num_samples == 60000, "HF fingerprint num_samples")
    check(fp_hf.version == "1.0.0", "HF fingerprint version")
    check(fp_hf.features_hash is not None and fp_hf.features_hash.startswith("sha256:"),
          "HF fingerprint features_hash")
    check(fp_hf.config_name == "default", "HF fingerprint config_name from info")

    # With explicit config_name override
    fp_hf2 = fper.fingerprint_hf_dataset(mock_ds, name="mnist", config_name="custom", split="test")
    check(fp_hf2.config_name == "custom", "HF fingerprint explicit config_name")
    check(fp_hf2.split == "test", "HF fingerprint explicit split")

    # With transforms
    fp_hf3 = fper.fingerprint_hf_dataset(
        mock_ds, name="mnist", split="train",
        transforms=MockCompose([MockTransform("ToTensor", 0)]),
    )
    check(fp_hf3.transforms_signature is not None, "HF fingerprint with transforms")
    check("ToTensor" in fp_hf3.transforms_signature, "HF fingerprint transform content")

    # ==================================================================
    # 7. Tier 2 fingerprinting with temp files
    # ==================================================================
    section("Tier 2: Local file fingerprinting")

    with tempfile.TemporaryDirectory() as tmpdir:
        f1 = make_temp_file(2048, 0x10, tmpdir)
        f2 = make_temp_file(4096, 0x20, tmpdir)
        f3 = make_temp_file(100, 0x30, tmpdir)

        fp_local = fper.fingerprint_local_files(
            [f1, f2, f3], name="my_dataset", split="train", root=tmpdir,
        )
        check(fp_local.tier == "tier2", "local fingerprint tier is tier2")
        check(fp_local.name == "my_dataset", "local fingerprint name")
        check(fp_local.shards is not None, "local fingerprint has shards")
        check(len(fp_local.shards) == 3, "local fingerprint 3 shards")
        check(all(s.fast_hash.startswith("sha256:") for s in fp_local.shards),
              "all shard fast_hashes have sha256 prefix")
        check(all(s.full_hash is None for s in fp_local.shards),
              "no full_hash by default")

        # With full hash
        fp_local_full = fper.fingerprint_local_files(
            [f1], name="single", split="train", root=tmpdir, compute_full_hash=True,
        )
        check(fp_local_full.shards[0].full_hash is not None,
              "full_hash present when compute_full_hash=True")
        check(fp_local_full.shards[0].full_hash.startswith("sha256:"),
              "full_hash has sha256 prefix")

        # Determinism: fingerprint same files again
        fp_local2 = fper.fingerprint_local_files(
            [f1, f2, f3], name="my_dataset", split="train", root=tmpdir,
        )
        for i in range(3):
            check(
                fp_local.shards[i].fast_hash == fp_local2.shards[i].fast_hash,
                f"shard[{i}] fast_hash deterministic",
            )

    # ==================================================================
    # 8. Tier 3 subset fingerprinting
    # ==================================================================
    section("Tier 3: Subset fingerprinting")

    parent_fp = DatasetFingerprint(
        name="mnist", role="primary", tier="tier1", split="train",
        hf_fingerprint="hf_parent", num_samples=60000,
    )
    indices = [0, 10, 20, 30, 40, 50]
    fp_sub = fper.fingerprint_subset(parent_fp, indices, seed=1337, strategy="random")
    check(fp_sub.tier == "tier3", "subset tier is tier3")
    check(fp_sub.name == "mnist_subset", "subset auto-name")
    check(fp_sub.subset_info is not None, "subset has subset_info")
    check(fp_sub.subset_info.seed == 1337, "subset seed")
    check(fp_sub.subset_info.subset_size == 6, "subset size")
    check(fp_sub.subset_info.strategy == "random", "subset strategy")
    check(fp_sub.subset_info.indices == sorted(indices), "subset indices sorted")
    check(fp_sub.subset_info.parent_fingerprint == parent_fp.content_hash(),
          "subset parent_fingerprint matches parent content_hash")
    check(fp_sub.num_samples == 6, "subset num_samples")

    # Large subset: indices not stored
    big_indices = list(range(20000))
    fp_big_sub = fper.fingerprint_subset(parent_fp, big_indices, seed=42, strategy="first_n")
    check(fp_big_sub.subset_info.indices is None,
          "large subset (>10k) does not store indices")
    check(fp_big_sub.subset_info.subset_size == 20000,
          "large subset records subset_size")

    # Custom name and role
    fp_sub_named = fper.fingerprint_subset(
        parent_fp, [1, 2, 3], seed=0, name="custom_sub", role="validation",
    )
    check(fp_sub_named.name == "custom_sub", "subset custom name")
    check(fp_sub_named.role == "validation", "subset custom role")

    # ==================================================================
    # 9. datasets.json save / load round-trip
    # ==================================================================
    section("datasets.json save/load")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "artifacts", "datasets.json")

        fps_to_save = [fp_hf, fp_sub]
        DatasetFingerprinter.save(fps_to_save, out_path)
        check(os.path.exists(out_path), "datasets.json file created")

        with open(out_path, "r") as f:
            raw = json.load(f)
        check(raw["schema_version"] == "1.0", "schema_version is 1.0")
        check("timestamp" in raw, "timestamp present")
        check(raw["fingerprint_tier"] == "mixed",
              "mixed tiers -> fingerprint_tier='mixed'")
        check(len(raw["datasets"]) == 2, "two datasets in file")

        loaded = DatasetFingerprinter.load(out_path)
        check(len(loaded) == 2, "loaded 2 fingerprints")
        check(loaded[0].name == "mnist", "loaded[0] name")
        check(loaded[0].tier == "tier1", "loaded[0] tier")
        check(loaded[1].tier == "tier3", "loaded[1] tier")
        check(loaded[1].subset_info is not None, "loaded[1] has subset_info")

        # Single tier -> tier label
        out_path2 = os.path.join(tmpdir, "datasets2.json")
        DatasetFingerprinter.save([fp_hf], out_path2)
        with open(out_path2, "r") as f:
            raw2 = json.load(f)
        check(raw2["fingerprint_tier"] == "tier1",
              "single tier -> fingerprint_tier matches")

    # ==================================================================
    # 10. FingerprintComparator
    # ==================================================================
    section("FingerprintComparator")

    comp = FingerprintComparator()

    # Exact match (identical objects)
    r = comp.compare(fp_hf, fp_hf)
    check(r.level == "exact", "identical fingerprints -> exact")

    # Same name/split, different version
    fp_v2 = DatasetFingerprint(
        name="mnist", role="primary", tier="tier1", split="train",
        version="2.0.0", hf_fingerprint="hf_different", num_samples=60000,
    )
    r2 = comp.compare(fp_hf, fp_v2)
    check(r2.level == "compatible", "different version -> compatible")
    check("version" in r2.detail, "detail mentions version")

    # Different name -> incompatible
    fp_other = DatasetFingerprint(
        name="cifar10", role="primary", tier="tier1", split="train",
        num_samples=50000,
    )
    r3 = comp.compare(fp_hf, fp_other)
    check(r3.level == "incompatible", "different name -> incompatible")

    # Different split -> incompatible
    fp_test = DatasetFingerprint(
        name="mnist", role="primary", tier="tier1", split="test",
        num_samples=10000,
    )
    r4 = comp.compare(fp_hf, fp_test)
    check(r4.level == "incompatible", "different split -> incompatible")

    # Tier 2 exact (same shards)
    s1 = ShardInfo("a.bin", 100, "2026-01-01T00:00:00+00:00", "sha256:aaa")
    s2 = ShardInfo("b.bin", 200, "2026-01-01T00:00:00+00:00", "sha256:bbb")
    fp_t2a = DatasetFingerprint(
        name="local", role="primary", tier="tier2", split="train",
        num_samples=300, shards=[s1, s2],
    )
    fp_t2b = DatasetFingerprint(
        name="local", role="primary", tier="tier2", split="train",
        num_samples=300, shards=[s1, s2],
    )
    r5 = comp.compare(fp_t2a, fp_t2b)
    check(r5.level == "exact", "tier2 identical shards -> exact")

    # Tier 2 compatible (one shard changed)
    s2_changed = ShardInfo("b.bin", 200, "2026-02-01T00:00:00+00:00", "sha256:ccc")
    fp_t2c = DatasetFingerprint(
        name="local", role="primary", tier="tier2", split="train",
        num_samples=300, shards=[s1, s2_changed],
    )
    r6 = comp.compare(fp_t2a, fp_t2c)
    check(r6.level == "compatible", "tier2 changed shard -> compatible")
    check("1 shard" in r6.detail, "detail mentions changed shards")

    # Tier 3 exact
    sub_info_a = SubsetInfo("parent_hash", [1, 2, 3], seed=42, subset_size=3, strategy="random")
    sub_info_b = SubsetInfo("parent_hash", [1, 2, 3], seed=42, subset_size=3, strategy="random")
    fp_s3a = DatasetFingerprint(
        name="sub", role="primary", tier="tier3", split="train",
        num_samples=3, subset_info=sub_info_a,
    )
    fp_s3b = DatasetFingerprint(
        name="sub", role="primary", tier="tier3", split="train",
        num_samples=3, subset_info=sub_info_b,
    )
    r7 = comp.compare(fp_s3a, fp_s3b)
    check(r7.level == "exact", "tier3 same indices+seed -> exact")

    # Tier 3 compatible (different seed)
    sub_info_c = SubsetInfo("parent_hash", [4, 5, 6], seed=99, subset_size=3, strategy="random")
    fp_s3c = DatasetFingerprint(
        name="sub", role="primary", tier="tier3", split="train",
        num_samples=3, subset_info=sub_info_c,
    )
    r8 = comp.compare(fp_s3a, fp_s3c)
    check(r8.level == "compatible", "tier3 different indices -> compatible")

    # Cross-tier comparison
    r9 = comp.compare(fp_hf, fp_t2a)
    check(r9.level == "incompatible", "cross-tier different name -> incompatible")

    fp_cross = DatasetFingerprint(
        name="mnist", role="primary", tier="tier2", split="train",
        num_samples=60000, shards=[s1],
    )
    r10 = comp.compare(fp_hf, fp_cross)
    check(r10.level == "compatible", "cross-tier same name/split -> compatible")

    # compare_datasets_json
    with tempfile.TemporaryDirectory() as tmpdir:
        path_a = os.path.join(tmpdir, "a.json")
        path_b = os.path.join(tmpdir, "b.json")
        DatasetFingerprinter.save([fp_hf], path_a)
        DatasetFingerprinter.save([fp_hf], path_b)
        results = comp.compare_datasets_json(path_a, path_b)
        check(len(results) == 1, "compare_datasets_json returns 1 result")
        check(results[0].level == "exact", "compare_datasets_json exact match")

        # Different lengths
        DatasetFingerprinter.save([fp_hf, fp_other], path_b)
        results2 = comp.compare_datasets_json(path_a, path_b)
        check(len(results2) == 2, "compare_datasets_json pads to max length")
        check(results2[1].level == "incompatible",
              "compare_datasets_json missing entry -> incompatible")

    # ==================================================================
    # 11. FingerprintCache
    # ==================================================================
    section("FingerprintCache")

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = FingerprintCache(tmpdir)

        data_file = make_temp_file(512, 0x44, tmpdir)

        # Cache miss: computes hash
        h1 = cache.get_or_compute(data_file, FastHash.hash_file_fast)
        check(len(h1) == 64, "cache miss returns valid hash")

        # Cache hit: returns same hash without recomputing
        h2 = cache.get_or_compute(data_file, FastHash.hash_file_fast)
        check(h1 == h2, "cache hit returns same hash")

        # Verify cache file exists
        cache_file = os.path.join(tmpdir, CACHE_FILENAME)
        check(os.path.exists(cache_file), "cache file written to disk")

        # Invalidation on mtime change
        time.sleep(0.05)  # ensure different mtime
        with open(data_file, "ab") as f:
            f.write(b"\xff" * 100)
        # Force re-read of stat
        h3 = cache.get_or_compute(data_file, FastHash.hash_file_fast)
        check(h3 != h1, "cache invalidated on mtime/size change")

        # New cache instance loads from disk
        cache2 = FingerprintCache(tmpdir)
        h4 = cache2.get_or_compute(data_file, FastHash.hash_file_fast)
        check(h4 == h3, "new cache instance loads persisted data")

        # Clear
        cache.clear()
        check(not os.path.exists(cache_file), "clear() removes cache file")

        # After clear, recomputes
        h5 = cache.get_or_compute(data_file, FastHash.hash_file_fast)
        check(h5 == h3, "after clear, recomputed hash matches")

    # ==================================================================
    # 12. Auto-detection (fingerprint_auto)
    # ==================================================================
    section("Auto-detection (fingerprint_auto)")

    auto_fper = DatasetFingerprinter(tier="auto")

    # Auto -> Tier 1 (HF dataset)
    fp_auto_hf = auto_fper.fingerprint_auto(
        MockHFDataset(1000), name="auto_mnist", split="train",
    )
    check(fp_auto_hf.tier == "tier1", "auto-detect HF dataset -> tier1")
    check(fp_auto_hf.name == "auto_mnist", "auto-detect HF name")

    # Auto -> Tier 2 (file list)
    with tempfile.TemporaryDirectory() as tmpdir:
        tf = make_temp_file(256, 0x55, tmpdir)
        fp_auto_local = auto_fper.fingerprint_auto(
            [tf], name="auto_local", split="train",
        )
        check(fp_auto_local.tier == "tier2", "auto-detect file list -> tier2")

    # Auto -> Tier 3 (parent_fingerprint kwarg)
    fp_auto_sub = auto_fper.fingerprint_auto(
        None,  # dataset_or_paths not used for tier3
        parent_fingerprint=parent_fp,
        indices=[1, 2, 3],
        seed=42,
        strategy="first_n",
        name="auto_sub",
    )
    check(fp_auto_sub.tier == "tier3", "auto-detect parent_fingerprint -> tier3")
    check(fp_auto_sub.subset_info.strategy == "first_n", "auto-detect subset strategy")

    # Forced tier overrides auto-detection
    forced = DatasetFingerprinter(tier="tier1")
    fp_forced = forced.fingerprint_auto(
        MockHFDataset(500), name="forced", split="val",
    )
    check(fp_forced.tier == "tier1", "forced tier1 works")

    # Invalid tier
    try:
        DatasetFingerprinter(tier="tier99")
        check(False, "invalid tier raises ValueError")
    except ValueError:
        check(True, "invalid tier raises ValueError")

    # Unrecognised type with auto
    try:
        auto_fper.fingerprint_auto(12345, name="bad")
        check(False, "unrecognised type raises TypeError")
    except TypeError:
        check(True, "unrecognised type raises TypeError")

    # ==================================================================
    # 13. Schema validation
    # ==================================================================
    section("Schema validation")

    valid_data = {
        "schema_version": "1.0",
        "fingerprint_tier": "tier1",
        "datasets": [{"name": "x", "role": "primary", "tier": "tier1"}],
        "timestamp": "2026-01-01T00:00:00Z",
    }
    errors_valid = DatasetsJSON.validate(valid_data)
    check(len(errors_valid) == 0, "valid data passes validation")

    # Wrong schema version
    bad_version = {**valid_data, "schema_version": "2.0"}
    errors_v = DatasetsJSON.validate(bad_version)
    check(any("schema_version" in e for e in errors_v),
          "wrong schema_version flagged")

    # Invalid tier
    bad_tier = {**valid_data, "fingerprint_tier": "tier99"}
    errors_t = DatasetsJSON.validate(bad_tier)
    check(any("fingerprint_tier" in e for e in errors_t),
          "invalid fingerprint_tier flagged")

    # Missing datasets key
    no_ds = {"schema_version": "1.0", "fingerprint_tier": "tier1", "timestamp": "x"}
    errors_d = DatasetsJSON.validate(no_ds)
    check(any("datasets" in e for e in errors_d),
          "missing datasets flagged")

    # datasets not a list
    bad_ds = {**valid_data, "datasets": "not_a_list"}
    errors_dl = DatasetsJSON.validate(bad_ds)
    check(any("list" in e for e in errors_dl),
          "datasets not a list flagged")

    # Missing required fields in dataset entry
    bad_entry = {**valid_data, "datasets": [{"name": "x"}]}
    errors_e = DatasetsJSON.validate(bad_entry)
    check(any("role" in e for e in errors_e),
          "missing role in dataset entry flagged")
    check(any("tier" in e for e in errors_e),
          "missing tier in dataset entry flagged")

    # Missing timestamp
    no_ts = {
        "schema_version": "1.0",
        "fingerprint_tier": "tier1",
        "datasets": [{"name": "x", "role": "primary", "tier": "tier1"}],
    }
    errors_ts = DatasetsJSON.validate(no_ts)
    check(any("timestamp" in e for e in errors_ts),
          "missing timestamp flagged")

    # Load invalid file raises ValueError
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_path = os.path.join(tmpdir, "bad.json")
        with open(bad_path, "w") as f:
            json.dump({"schema_version": "9.9", "datasets": [], "timestamp": "x"}, f)
        try:
            DatasetsJSON.load(bad_path)
            check(False, "loading invalid datasets.json raises ValueError")
        except ValueError:
            check(True, "loading invalid datasets.json raises ValueError")

    # ==================================================================
    # 14. HuggingFace graceful handling when library missing
    # ==================================================================
    section("Graceful HF handling")

    # The fingerprinter should work even when datasets library is not installed.
    # We test with our mock objects -- if it works with duck-typed objects it
    # works without importing the actual library.
    class MinimalHF:
        """Minimal duck-typed HF dataset without .info.features."""
        def __init__(self):
            self.info = type("Info", (), {"config_name": None, "version": None, "features": None})()
            self._fingerprint = "minimal_fp"
        def __len__(self) -> int:
            return 100

    fp_minimal = fper.fingerprint_hf_dataset(MinimalHF(), name="minimal", split="train")
    check(fp_minimal.tier == "tier1", "minimal HF object -> tier1")
    check(fp_minimal.features_hash is None, "None features -> None features_hash")
    check(fp_minimal.version is None, "None version -> None version")
    check(fp_minimal.hf_fingerprint == "minimal_fp", "minimal hf_fingerprint captured")

    # ==================================================================
    # 15. Edge cases and integration
    # ==================================================================
    section("Edge cases and integration")

    # Empty fingerprint list save/load
    with tempfile.TemporaryDirectory() as tmpdir:
        empty_path = os.path.join(tmpdir, "empty.json")
        DatasetFingerprinter.save([], empty_path)
        loaded_empty = DatasetFingerprinter.load(empty_path)
        check(len(loaded_empty) == 0, "empty fingerprint list round-trips")

    # Tier 2 with FingerprintCache
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = FingerprintCache(tmpdir)
        cf = make_temp_file(1024, 0x66, tmpdir)
        fp_cached = fper.fingerprint_local_files(
            [cf], name="cached_ds", split="train", root=tmpdir, cache=cache,
        )
        check(fp_cached.shards[0].fast_hash.startswith("sha256:"),
              "cached fingerprint has proper hash")
        # Second call should hit cache
        fp_cached2 = fper.fingerprint_local_files(
            [cf], name="cached_ds", split="train", root=tmpdir, cache=cache,
        )
        check(fp_cached.shards[0].fast_hash == fp_cached2.shards[0].fast_hash,
              "cached fingerprint consistent on re-read")

    # fingerprint_local_files with empty list
    try:
        fper.fingerprint_local_files([], name="bad")
        check(False, "empty paths raises ValueError")
    except ValueError:
        check(True, "empty paths raises ValueError")

    # DatasetFingerprint with all optional fields None
    fp_bare = DatasetFingerprint(name="bare", role="primary", tier="tier1", split="train")
    d_bare = fp_bare.to_dict()
    check("config_name" not in d_bare, "bare fingerprint omits config_name")
    check("version" not in d_bare, "bare fingerprint omits version")
    check("hf_fingerprint" not in d_bare, "bare fingerprint omits hf_fingerprint")
    check("shards" not in d_bare, "bare fingerprint omits shards")
    check("subset_info" not in d_bare, "bare fingerprint omits subset_info")
    fp_bare_rt = DatasetFingerprint.from_dict(d_bare)
    check(fp_bare_rt.name == "bare", "bare fingerprint round-trips")

    # Transforms on tier2
    with tempfile.TemporaryDirectory() as tmpdir:
        tf = make_temp_file(100, 0x77, tmpdir)
        fp_t2_trans = fper.fingerprint_local_files(
            [tf], name="trans_ds", split="train", root=tmpdir,
            transforms=MockCompose([MockTransform("Flip", 1)]),
        )
        check(fp_t2_trans.transforms_signature is not None,
              "tier2 with transforms records signature")
        check("Flip" in fp_t2_trans.transforms_signature,
              "tier2 transform signature contains transform name")

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {_pass} passed, {_fail} failed")
    print(f"{'=' * 60}")
    if _errors:
        print("\n  Failed tests:")
        for e in _errors:
            print(f"    - {e}")
    print()
    sys.exit(0 if _fail == 0 else 1)
