"""
DataPipelineConfig — Central configuration for the data pipeline.

Includes all fields from SKILL.md, validation, and serialization.

Usage:
    cfg = DataPipelineConfig(format="hf_streaming", pack_enabled=True,
                              pack_mode="sft_boundary_aware")
    cfg.validate()
    d = cfg.to_dict()
    cfg2 = DataPipelineConfig.from_dict(d)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List


# ---------------------------------------------------------------------------
# Valid enumerations
# ---------------------------------------------------------------------------

VALID_FORMATS = {"hf_streaming", "webdataset_tar", "token_memmap"}
VALID_PACK_MODES = {"none", "pretrain_blocks", "sft_boundary_aware"}


# ---------------------------------------------------------------------------
# DataPipelineConfig
# ---------------------------------------------------------------------------

@dataclass
class DataPipelineConfig:
    """Central configuration for the data loading and packing pipeline."""

    # I/O format
    format: str = "hf_streaming"
    streaming: bool = True
    cache_dir: Optional[str] = None
    data_path: Optional[str] = None

    # Shuffle
    shuffle_buffer_size: int = 10000

    # DataLoader
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True
    pin_memory: bool = True
    batch_size: int = 8

    # Packing
    pack_enabled: bool = False
    pack_mode: str = "none"
    pack_target_seq_len: int = 2048
    pack_bucket_boundaries: Tuple[int, ...] = (256, 512, 1024, 2048)
    pack_boundary_aware: bool = True
    pack_pad_token_id: int = 0

    # Sharding
    shard_deterministic: bool = True
    shard_seed: int = 42

    # ---- Validation ----

    def validate(self) -> List[str]:
        """
        Validate the configuration. Returns list of errors (empty if valid).
        Raises ValueError if any errors found.
        """
        errors: List[str] = []

        if self.format not in VALID_FORMATS:
            errors.append(
                f"format must be one of {VALID_FORMATS}, got '{self.format}'"
            )

        if self.pack_mode not in VALID_PACK_MODES:
            errors.append(
                f"pack_mode must be one of {VALID_PACK_MODES}, got '{self.pack_mode}'"
            )

        if self.pack_mode != "none" and not self.pack_enabled:
            errors.append(
                f"pack_mode='{self.pack_mode}' requires pack_enabled=True"
            )

        if self.pack_enabled and self.pack_mode == "none":
            errors.append(
                "pack_enabled=True requires pack_mode to be set "
                "(not 'none')"
            )

        if self.pack_target_seq_len <= 0:
            errors.append(
                f"pack_target_seq_len must be > 0, got {self.pack_target_seq_len}"
            )

        if self.num_workers < 0:
            errors.append(
                f"num_workers must be >= 0, got {self.num_workers}"
            )

        if self.prefetch_factor < 1:
            errors.append(
                f"prefetch_factor must be >= 1, got {self.prefetch_factor}"
            )

        if self.shuffle_buffer_size < 0:
            errors.append(
                f"shuffle_buffer_size must be >= 0, got {self.shuffle_buffer_size}"
            )

        if self.batch_size <= 0:
            errors.append(
                f"batch_size must be > 0, got {self.batch_size}"
            )

        # Bucket boundaries must be sorted ascending
        boundaries = self.pack_bucket_boundaries
        if len(boundaries) > 1:
            for i in range(len(boundaries) - 1):
                if boundaries[i] >= boundaries[i + 1]:
                    errors.append(
                        f"pack_bucket_boundaries must be sorted ascending, "
                        f"got {boundaries}"
                    )
                    break

        if boundaries and any(b <= 0 for b in boundaries):
            errors.append(
                f"pack_bucket_boundaries must all be > 0, got {boundaries}"
            )

        if errors:
            raise ValueError(
                f"DataPipelineConfig validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )
        return errors

    # ---- Serialization ----

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict."""
        d = asdict(self)
        # Tuple -> list for JSON compatibility
        d["pack_bucket_boundaries"] = list(d["pack_bucket_boundaries"])
        return d

    def to_json(self, path: str) -> None:
        """Write config to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DataPipelineConfig:
        """Construct from a plain dict."""
        d = dict(d)  # copy
        if "pack_bucket_boundaries" in d:
            d["pack_bucket_boundaries"] = tuple(d["pack_bucket_boundaries"])
        # Filter to known fields
        import inspect
        valid_fields = {
            f.name for f in cls.__dataclass_fields__.values()
        }
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path: str) -> DataPipelineConfig:
        """Load config from JSON file."""
        with open(path) as f:
            d = json.load(f)
        return cls.from_dict(d)


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _run_self_tests() -> None:
    import tempfile
    import os

    passed = 0
    failed = 0

    def check(name: str, condition: bool, detail: str = ""):
        nonlocal passed, failed
        status = "PASS" if condition else "FAIL"
        if not condition:
            failed += 1
            print(f"  [{status}] {name}: {detail}")
        else:
            passed += 1
            print(f"  [{status}] {name}")

    print("=" * 60)
    print("DataPipelineConfig Self-Tests")
    print("=" * 60)

    # Test 1: Default config passes validation
    cfg = DataPipelineConfig()
    try:
        cfg.validate()
        check("T1: default config validates", True)
    except ValueError as e:
        check("T1: default config validates", False, str(e))

    # Test 2: Valid pack config passes
    cfg2 = DataPipelineConfig(
        format="hf_streaming",
        pack_enabled=True,
        pack_mode="sft_boundary_aware",
        pack_target_seq_len=2048,
    )
    try:
        cfg2.validate()
        check("T2: valid pack config validates", True)
    except ValueError as e:
        check("T2: valid pack config validates", False, str(e))

    # Test 3: Invalid format raises
    cfg3 = DataPipelineConfig(format="invalid_format")
    try:
        cfg3.validate()
        check("T3: invalid format raises", False, "should have raised")
    except ValueError:
        check("T3: invalid format raises", True)

    # Test 4: Invalid pack_mode raises
    cfg4 = DataPipelineConfig(pack_mode="invalid_mode")
    try:
        cfg4.validate()
        check("T4: invalid pack_mode raises", False, "should have raised")
    except ValueError:
        check("T4: invalid pack_mode raises", True)

    # Test 5: pack_mode without pack_enabled raises
    cfg5 = DataPipelineConfig(pack_enabled=False, pack_mode="pretrain_blocks")
    try:
        cfg5.validate()
        check("T5: pack_mode without pack_enabled raises", False)
    except ValueError:
        check("T5: pack_mode without pack_enabled raises", True)

    # Test 6: pack_enabled=True with pack_mode=none raises
    cfg6 = DataPipelineConfig(pack_enabled=True, pack_mode="none")
    try:
        cfg6.validate()
        check("T6: pack_enabled=True with mode=none raises", False)
    except ValueError:
        check("T6: pack_enabled=True with mode=none raises", True)

    # Test 7: Unsorted bucket boundaries raises
    cfg7 = DataPipelineConfig(
        pack_bucket_boundaries=(1024, 512, 256, 2048)
    )
    try:
        cfg7.validate()
        check("T7: unsorted boundaries raises", False)
    except ValueError:
        check("T7: unsorted boundaries raises", True)

    # Test 8: Negative num_workers raises
    cfg8 = DataPipelineConfig(num_workers=-1)
    try:
        cfg8.validate()
        check("T8: negative num_workers raises", False)
    except ValueError:
        check("T8: negative num_workers raises", True)

    # Test 9: to_dict round-trip
    original = DataPipelineConfig(
        format="token_memmap",
        pack_enabled=True,
        pack_mode="pretrain_blocks",
        pack_target_seq_len=4096,
        pack_bucket_boundaries=(128, 256, 512),
        shard_seed=123,
    )
    d = original.to_dict()
    restored = DataPipelineConfig.from_dict(d)
    check("T9: to_dict/from_dict round-trip",
          restored.format == original.format
          and restored.pack_mode == original.pack_mode
          and restored.pack_target_seq_len == original.pack_target_seq_len
          and restored.shard_seed == original.shard_seed
          and restored.pack_bucket_boundaries == original.pack_bucket_boundaries,
          f"original.format={original.format}, restored.format={restored.format}")

    # Test 10: JSON round-trip
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        original.validate()
        original.to_json(tmp_path)
        loaded = DataPipelineConfig.from_json(tmp_path)
        check("T10: JSON round-trip",
              loaded.format == original.format
              and loaded.pack_mode == original.pack_mode
              and loaded.pack_target_seq_len == original.pack_target_seq_len
              and loaded.pack_bucket_boundaries == original.pack_bucket_boundaries)
    finally:
        os.unlink(tmp_path)

    # Test 11: to_dict produces JSON-serializable dict
    d = cfg2.to_dict()
    try:
        json.dumps(d)
        check("T11: to_dict is JSON-serializable", True)
    except (TypeError, ValueError) as e:
        check("T11: to_dict is JSON-serializable", False, str(e))

    # Test 12: Zero bucket boundary raises
    cfg12 = DataPipelineConfig(pack_bucket_boundaries=(0, 256, 512))
    try:
        cfg12.validate()
        check("T12: zero bucket boundary raises", False)
    except ValueError:
        check("T12: zero bucket boundary raises", True)

    # Test 13: All valid formats accepted
    for fmt in VALID_FORMATS:
        cfgf = DataPipelineConfig(format=fmt)
        try:
            cfgf.validate()
            check(f"T13: format '{fmt}' accepted", True)
        except ValueError as e:
            check(f"T13: format '{fmt}' accepted", False, str(e))

    # Test 14: All valid pack modes accepted (when pack_enabled)
    for mode in VALID_PACK_MODES:
        if mode == "none":
            cfgm = DataPipelineConfig(pack_enabled=False, pack_mode="none")
        else:
            cfgm = DataPipelineConfig(pack_enabled=True, pack_mode=mode)
        try:
            cfgm.validate()
            check(f"T14: pack_mode '{mode}' accepted", True)
        except ValueError as e:
            check(f"T14: pack_mode '{mode}' accepted", False, str(e))

    # Test 15: from_dict ignores unknown fields
    d_extra = {"format": "hf_streaming", "unknown_field": 42}
    try:
        cfg15 = DataPipelineConfig.from_dict(d_extra)
        check("T15: from_dict ignores unknown fields", True)
    except TypeError as e:
        check("T15: from_dict ignores unknown fields", False, str(e))

    print("-" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_self_tests()
