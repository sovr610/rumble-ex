"""
compile_config_template.py
--------------------------
Configuration dataclasses for torch.compile integration.

Provides CompileConfig and BucketConfig with field validation,
dict/YAML serialization, and reasonable defaults. Copy this file
into your project and adjust defaults to match your workload.

Usage:
    from compile_config_template import CompileConfig, BucketConfig

    cfg = CompileConfig(enabled=True, mode="reduce-overhead")
    bucket_cfg = BucketConfig(buckets=[256, 512, 1024], pad_token_id=0)
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_MODES = {
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
}

VALID_BACKENDS = {
    "inductor",
    "aot_eager",
    "cudagraphs",
}

VALID_FAIL_POLICIES = {
    "fallback_eager",
    "raise",
}


# ---------------------------------------------------------------------------
# BucketConfig
# ---------------------------------------------------------------------------


@dataclass
class BucketConfig:
    """
    Configuration for sequence-length bucketing to avoid recompile thrashing.

    Attributes
    ----------
    buckets:
        Sorted list of bucket boundary sizes. Sequences are padded to the
        smallest bucket >= their length. Must be non-empty and strictly
        increasing.
    pad_token_id:
        Token ID used for padding. Typically 0 or the model's PAD token id.
        Padded positions should be masked in attention to avoid contaminating
        the computation.
    """

    buckets: List[int] = field(default_factory=lambda: [256, 512, 1024, 2048])
    pad_token_id: int = 0

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if not self.buckets:
            raise ValueError("BucketConfig.buckets must be non-empty.")
        buckets = sorted(self.buckets)
        if buckets != list(self.buckets):
            raise ValueError(
                f"BucketConfig.buckets must be in ascending order. "
                f"Got {self.buckets!r}, expected {buckets!r}."
            )
        for b in self.buckets:
            if not isinstance(b, int) or b <= 0:
                raise ValueError(
                    f"All bucket sizes must be positive integers. Got {b!r}."
                )
        if len(set(self.buckets)) != len(self.buckets):
            raise ValueError("BucketConfig.buckets must not contain duplicates.")
        if self.pad_token_id < 0:
            raise ValueError(
                f"BucketConfig.pad_token_id must be >= 0. Got {self.pad_token_id}."
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BucketConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def __repr__(self) -> str:
        return (
            f"BucketConfig(buckets={self.buckets!r}, pad_token_id={self.pad_token_id})"
        )


# ---------------------------------------------------------------------------
# CompileConfig
# ---------------------------------------------------------------------------


@dataclass
class CompileConfig:
    """
    All torch.compile settings in one place.

    The safe wrapper (maybe_compile) reads this config to determine whether
    and how to compile a model. Disabled by default — opt-in explicitly.

    Attributes
    ----------
    enabled:
        Master switch. If False, maybe_compile returns the model unchanged.
        Default False to avoid surprising compilation in unintended contexts.
    mode:
        Compilation mode passed to torch.compile.
        - "default": basic kernel fusion, fastest compilation.
        - "reduce-overhead": CUDA graph capture, lower per-step overhead.
        - "max-autotune": autotuned kernels + CUDA graphs, slowest compile.
        - "max-autotune-no-cudagraphs": autotuned kernels without CUDA graphs.
    dynamic:
        Dynamic shape mode.
        - None (default): start static, auto-generalize on first recompile.
        - False: fully static, recompile on any shape change.
        - True: force dynamic everywhere (not recommended for production).
    backend:
        Compilation backend.
        - "inductor" (default): C++/Triton code generation.
        - "aot_eager": debugging without codegen.
        - "cudagraphs": standalone CUDA graph capture.
    fullgraph:
        If True, raise on graph breaks instead of silently segmenting.
        Useful for debugging and maximum optimization. Not recommended as default.
    options:
        Dict passed as torch.compile(options=...). Controls Inductor internals.
        Common keys: "epilogue_fusion", "shape_padding", "fallback_random".
    allowlist:
        If non-empty, only compile submodules whose names match any pattern in
        this list. Patterns are substring matches against module names.
        Empty list = compile the whole model.
    blocklist:
        Submodule name patterns to exclude from compilation. Matching modules
        get torch.compiler.disable applied. Blocklist takes precedence over allowlist.
    healthcheck:
        If True and sample_batch is provided to maybe_compile, run a smoketest
        of smoketest_steps training steps after compilation. Falls back to eager
        if smoketest fails.
    fail_policy:
        What to do when compilation or smoketest fails.
        - "fallback_eager": log error, return original eager model.
        - "raise": re-raise the exception (for strict environments).
    smoketest_steps:
        Number of forward-backward-optimizer steps in the smoketest.
        Must be >= 1. Default 3 (enough to catch compile-time + runtime issues).
    """

    enabled: bool = False
    mode: str = "default"
    dynamic: Optional[bool] = None
    backend: str = "inductor"
    fullgraph: bool = False
    options: Optional[Dict[str, Any]] = None
    allowlist: List[str] = field(default_factory=list)
    blocklist: List[str] = field(default_factory=list)
    healthcheck: bool = True
    fail_policy: str = "fallback_eager"
    smoketest_steps: int = 3

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"CompileConfig.mode must be one of {sorted(VALID_MODES)!r}. "
                f"Got {self.mode!r}."
            )
        if self.backend not in VALID_BACKENDS:
            raise ValueError(
                f"CompileConfig.backend must be one of {sorted(VALID_BACKENDS)!r}. "
                f"Got {self.backend!r}."
            )
        if self.fail_policy not in VALID_FAIL_POLICIES:
            raise ValueError(
                f"CompileConfig.fail_policy must be one of "
                f"{sorted(VALID_FAIL_POLICIES)!r}. Got {self.fail_policy!r}."
            )
        if self.smoketest_steps <= 0:
            raise ValueError(
                f"CompileConfig.smoketest_steps must be > 0. "
                f"Got {self.smoketest_steps}."
            )
        if self.dynamic not in (None, True, False):
            raise ValueError(
                f"CompileConfig.dynamic must be None, True, or False. "
                f"Got {self.dynamic!r}."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (JSON-compatible)."""
        d = asdict(self)
        # asdict converts None correctly; options dict may have non-JSON types
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CompileConfig":
        """Deserialize from a dict. Unknown keys are ignored."""
        known = {k for k in cls.__dataclass_fields__}
        filtered = {k: v for k, v in d.items() if k in known}
        # Convert list fields that may have come in as tuples
        for list_field in ("allowlist", "blocklist"):
            if list_field in filtered and filtered[list_field] is not None:
                filtered[list_field] = list(filtered[list_field])
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str) -> "CompileConfig":
        """
        Load CompileConfig from a YAML file.

        The YAML file should contain a mapping with keys matching
        CompileConfig field names. A 'compile' top-level key is also
        supported (e.g., if the YAML file is a larger config file).

        Requires PyYAML. Falls back to JSON if path ends in .json.
        """
        p = Path(path)
        if p.suffix in (".json",):
            with open(p) as f:
                raw = json.load(f)
        else:
            try:
                import yaml
            except ImportError as e:
                raise ImportError(
                    "PyYAML is required for YAML loading. Install with: pip install pyyaml"
                ) from e
            with open(p) as f:
                raw = yaml.safe_load(f)

        # Support both flat YAML and nested under 'compile' key
        if isinstance(raw, dict) and "compile" in raw:
            raw = raw["compile"]

        return cls.from_dict(raw)

    def to_yaml_string(self) -> str:
        """Serialize to YAML string. Requires PyYAML."""
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "PyYAML is required for YAML serialization. Install with: pip install pyyaml"
            ) from e
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=True)

    def copy_with(self, **overrides: Any) -> "CompileConfig":
        """Return a new CompileConfig with specified fields overridden."""
        d = self.to_dict()
        d.update(overrides)
        return CompileConfig.from_dict(d)

    def __repr__(self) -> str:
        active = {
            "enabled": self.enabled,
            "mode": self.mode,
            "backend": self.backend,
            "dynamic": self.dynamic,
            "fullgraph": self.fullgraph,
            "healthcheck": self.healthcheck,
            "fail_policy": self.fail_policy,
            "smoketest_steps": self.smoketest_steps,
        }
        if self.options:
            active["options"] = self.options
        if self.allowlist:
            active["allowlist"] = self.allowlist
        if self.blocklist:
            active["blocklist"] = self.blocklist
        parts = ", ".join(f"{k}={v!r}" for k, v in active.items())
        return f"CompileConfig({parts})"


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def default_config() -> CompileConfig:
    """Standard starting config: disabled, mode=default, fullgraph=False."""
    return CompileConfig(
        enabled=False,
        mode="default",
        dynamic=None,
        backend="inductor",
        fullgraph=False,
        healthcheck=True,
        fail_policy="fallback_eager",
        smoketest_steps=3,
    )


def fast_dev_config() -> CompileConfig:
    """Config for development: enabled, default mode, fallback on failure."""
    return CompileConfig(
        enabled=True,
        mode="default",
        dynamic=None,
        backend="inductor",
        fullgraph=False,
        healthcheck=True,
        fail_policy="fallback_eager",
        smoketest_steps=3,
    )


def production_config(mode: str = "max-autotune") -> CompileConfig:
    """Config for production training: strict fallback, max-autotune."""
    return CompileConfig(
        enabled=True,
        mode=mode,
        dynamic=None,
        backend="inductor",
        fullgraph=False,
        options={"shape_padding": True, "epilogue_fusion": True},
        healthcheck=True,
        fail_policy="fallback_eager",
        smoketest_steps=5,
    )


def inference_config() -> CompileConfig:
    """Config for inference serving: reduce-overhead with static shapes."""
    return CompileConfig(
        enabled=True,
        mode="reduce-overhead",
        dynamic=False,
        backend="inductor",
        fullgraph=False,
        options={"shape_padding": True},
        healthcheck=True,
        fail_policy="fallback_eager",
        smoketest_steps=3,
    )


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    failures: List[str] = []

    def _check(name: str, condition: bool, msg: str = "") -> None:
        if condition:
            print(f"  PASS  {name}")
        else:
            print(f"  FAIL  {name}: {msg}")
            failures.append(name)

    print("=" * 60)
    print("CompileConfig self-tests")
    print("=" * 60)

    # Test 1: Default config is valid
    try:
        cfg = CompileConfig()
        _check("default_config_valid", True)
    except Exception as e:
        _check("default_config_valid", False, str(e))

    # Test 2: default_config() factory
    try:
        cfg = default_config()
        _check("default_config_factory", not cfg.enabled and cfg.mode == "default")
    except Exception as e:
        _check("default_config_factory", False, str(e))

    # Test 3: Invalid mode raises ValueError
    try:
        _ = CompileConfig(mode="turbo-mode")
        _check("invalid_mode_raises", False, "Should have raised ValueError")
    except ValueError as e:
        _check("invalid_mode_raises", "mode" in str(e).lower())
    except Exception as e:
        _check("invalid_mode_raises", False, f"Wrong exception: {e}")

    # Test 4: Invalid backend raises ValueError
    try:
        _ = CompileConfig(backend="magic")
        _check("invalid_backend_raises", False, "Should have raised ValueError")
    except ValueError as e:
        _check("invalid_backend_raises", "backend" in str(e).lower())
    except Exception as e:
        _check("invalid_backend_raises", False, f"Wrong exception: {e}")

    # Test 5: Invalid fail_policy raises ValueError
    try:
        _ = CompileConfig(fail_policy="silently_ignore")
        _check("invalid_fail_policy_raises", False, "Should have raised ValueError")
    except ValueError as e:
        _check("invalid_fail_policy_raises", "fail_policy" in str(e).lower())
    except Exception as e:
        _check("invalid_fail_policy_raises", False, f"Wrong exception: {e}")

    # Test 6: smoketest_steps=0 raises
    try:
        _ = CompileConfig(smoketest_steps=0)
        _check("smoketest_steps_zero_raises", False, "Should have raised ValueError")
    except ValueError as e:
        _check("smoketest_steps_zero_raises", "smoketest_steps" in str(e).lower())
    except Exception as e:
        _check("smoketest_steps_zero_raises", False, f"Wrong exception: {e}")

    # Test 7: to_dict / from_dict roundtrip
    try:
        cfg = CompileConfig(
            enabled=True,
            mode="reduce-overhead",
            dynamic=None,
            backend="inductor",
            fullgraph=True,
            options={"shape_padding": True},
            allowlist=["attention"],
            blocklist=["sampling"],
            healthcheck=False,
            fail_policy="raise",
            smoketest_steps=5,
        )
        restored = CompileConfig.from_dict(cfg.to_dict())
        _check(
            "to_dict_from_dict_roundtrip",
            cfg.to_dict() == restored.to_dict(),
            f"{cfg.to_dict()} != {restored.to_dict()}",
        )
    except Exception as e:
        _check("to_dict_from_dict_roundtrip", False, str(e))

    # Test 8: All valid modes accepted
    for mode in VALID_MODES:
        try:
            _ = CompileConfig(mode=mode)
            _check(f"valid_mode_{mode.replace('-', '_')}", True)
        except Exception as e:
            _check(f"valid_mode_{mode.replace('-', '_')}", False, str(e))

    # Test 9: All valid backends accepted
    for backend in VALID_BACKENDS:
        try:
            _ = CompileConfig(backend=backend)
            _check(f"valid_backend_{backend}", True)
        except Exception as e:
            _check(f"valid_backend_{backend}", False, str(e))

    # Test 10: copy_with works
    try:
        cfg = CompileConfig(enabled=False, mode="default")
        cfg2 = cfg.copy_with(enabled=True, mode="reduce-overhead")
        _check(
            "copy_with",
            cfg2.enabled is True and cfg2.mode == "reduce-overhead" and cfg.enabled is False,
        )
    except Exception as e:
        _check("copy_with", False, str(e))

    # Test 11: BucketConfig defaults valid
    try:
        bc = BucketConfig()
        _check(
            "bucket_config_defaults",
            bc.buckets == [256, 512, 1024, 2048] and bc.pad_token_id == 0,
        )
    except Exception as e:
        _check("bucket_config_defaults", False, str(e))

    # Test 12: BucketConfig rejects unsorted buckets
    try:
        _ = BucketConfig(buckets=[1024, 256, 512])
        _check("bucket_config_unsorted_raises", False, "Should have raised ValueError")
    except ValueError:
        _check("bucket_config_unsorted_raises", True)
    except Exception as e:
        _check("bucket_config_unsorted_raises", False, f"Wrong exception: {e}")

    # Test 13: BucketConfig rejects empty buckets
    try:
        _ = BucketConfig(buckets=[])
        _check("bucket_config_empty_raises", False, "Should have raised ValueError")
    except ValueError:
        _check("bucket_config_empty_raises", True)
    except Exception as e:
        _check("bucket_config_empty_raises", False, f"Wrong exception: {e}")

    # Test 14: BucketConfig to_dict / from_dict roundtrip
    try:
        bc = BucketConfig(buckets=[128, 256, 512], pad_token_id=1)
        bc2 = BucketConfig.from_dict(bc.to_dict())
        _check("bucket_config_roundtrip", bc.to_dict() == bc2.to_dict())
    except Exception as e:
        _check("bucket_config_roundtrip", False, str(e))

    # Test 15: production_config factory
    try:
        pc = production_config()
        _check(
            "production_config_factory",
            pc.enabled and pc.mode == "max-autotune" and pc.options is not None,
        )
    except Exception as e:
        _check("production_config_factory", False, str(e))

    print()
    if failures:
        print(f"FAILED: {len(failures)} tests: {failures}")
        sys.exit(1)
    else:
        print(f"All {15} tests passed.")
