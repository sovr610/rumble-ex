"""
backend_config_template.py
==========================
AttentionConfig dataclass, BackendConfig, CapabilityReport, AttentionMetrics,
and validation/serialization helpers for the fast-attention-sdpa skill.

Self-tests: python backend_config_template.py
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — allowed values for each field
# ---------------------------------------------------------------------------

ALLOWED_IMPL = {"auto", "sdpa"}
ALLOWED_BACKEND = {"auto", "flash", "efficient", "cudnn", "math", "flash_or_efficient"}
ALLOWED_EXTERNAL_FLASH = {"off", "prefer", "require"}
ALLOWED_DROPOUT_POLICY = {"train_only", "always"}
ALLOWED_MASK_KIND = {"none", "causal", "float_additive", "bool_mask", "padding"}

# ---------------------------------------------------------------------------
# AttentionConfig
# ---------------------------------------------------------------------------


@dataclass
class AttentionConfig:
    """
    Top-level configuration for attention behaviour.

    Fields mirror the SKILL.md public contract.
    """

    impl: str = "sdpa"
    # "auto" | "sdpa"

    backend: str = "auto"
    # "auto" | "flash" | "efficient" | "cudnn" | "math" | "flash_or_efficient"

    force: bool = False
    # If True, remove Math fallback — error if fused kernel cannot run.

    log_backend: bool = True
    # Log capability report at startup.

    external_flash_attn: str = "off"
    # "off" | "prefer" | "require"

    dropout_policy: str = "train_only"
    # "train_only" → always pass 0.0 in eval (correct behaviour)
    # "always"     → pass dropout_p even in eval (legacy bug, use with care)

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    def validate(self) -> None:
        """Raise ValueError if any field holds an invalid value."""
        errors: List[str] = []

        if self.impl not in ALLOWED_IMPL:
            errors.append(
                f"impl={self.impl!r} is not valid. Allowed: {sorted(ALLOWED_IMPL)}"
            )
        if self.backend not in ALLOWED_BACKEND:
            errors.append(
                f"backend={self.backend!r} is not valid. "
                f"Allowed: {sorted(ALLOWED_BACKEND)}"
            )
        if self.external_flash_attn not in ALLOWED_EXTERNAL_FLASH:
            errors.append(
                f"external_flash_attn={self.external_flash_attn!r} is not valid. "
                f"Allowed: {sorted(ALLOWED_EXTERNAL_FLASH)}"
            )
        if self.dropout_policy not in ALLOWED_DROPOUT_POLICY:
            errors.append(
                f"dropout_policy={self.dropout_policy!r} is not valid. "
                f"Allowed: {sorted(ALLOWED_DROPOUT_POLICY)}"
            )
        if not isinstance(self.force, bool):
            errors.append(f"force must be bool, got {type(self.force)}")
        if not isinstance(self.log_backend, bool):
            errors.append(f"log_backend must be bool, got {type(self.log_backend)}")

        if errors:
            raise ValueError("AttentionConfig validation failed:\n" + "\n".join(errors))

    # -----------------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dict suitable for JSON serialization."""
        return {
            "impl": self.impl,
            "backend": self.backend,
            "force": self.force,
            "log_backend": self.log_backend,
            "external_flash_attn": self.external_flash_attn,
            "dropout_policy": self.dropout_policy,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AttentionConfig":
        """Reconstruct from a plain dict. Unknown keys are ignored."""
        cfg = cls(
            impl=d.get("impl", "sdpa"),
            backend=d.get("backend", "auto"),
            force=d.get("force", False),
            log_backend=d.get("log_backend", True),
            external_flash_attn=d.get("external_flash_attn", "off"),
            dropout_policy=d.get("dropout_policy", "train_only"),
        )
        cfg.validate()
        return cfg

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "AttentionConfig":
        return cls.from_dict(json.loads(s))


# ---------------------------------------------------------------------------
# BackendConfig — resolved backend list (used by sdpa_attention)
# ---------------------------------------------------------------------------


@dataclass
class BackendConfig:
    """
    Resolved backend configuration passed to sdpa_attention().

    Do not construct this directly; use BackendConfig.from_attention_config().
    """

    backends: List[Any]  # List[SDPBackend] — SDPBackend imported lazily
    force: bool = False
    log: bool = True

    @classmethod
    def from_attention_config(cls, cfg: AttentionConfig) -> "BackendConfig":
        """
        Map AttentionConfig to a list of SDPBackend values.

        SDPBackend is imported lazily so this module stays importable on
        machines without PyTorch (e.g. CI runners for reference-only usage).
        """
        try:
            from torch.nn.attention import SDPBackend
        except ImportError:
            # PyTorch not installed — return placeholder
            return cls(backends=[], force=cfg.force, log=cfg.log_backend)

        FLASH = SDPBackend.FLASH_ATTENTION
        EFF = SDPBackend.EFFICIENT_ATTENTION
        CUDNN = SDPBackend.CUDNN_ATTENTION
        MATH = SDPBackend.MATH

        # Build candidate list based on policy
        policy_map = {
            "auto": [FLASH, EFF, CUDNN, MATH],
            "flash": [FLASH, MATH],
            "efficient": [EFF, MATH],
            "cudnn": [CUDNN, MATH],
            "flash_or_efficient": [FLASH, EFF, CUDNN, MATH],
            "math": [MATH],
        }
        backends = list(policy_map[cfg.backend])

        # Enforce force mode: remove Math so SDPA raises if fused kernel fails
        if cfg.force and cfg.backend != "math":
            backends = [b for b in backends if b != MATH]

        return cls(backends=backends, force=cfg.force, log=cfg.log_backend)


# ---------------------------------------------------------------------------
# CapabilityReport — result of probing SDPAParams capability checks
# ---------------------------------------------------------------------------


@dataclass
class CapabilityReport:
    """
    Results from running can_use_* capability checks against a set of probe
    tensors. Populated by AttentionAuditor.probe_runtime().
    """

    device: str = "cpu"
    device_name: str = "CPU"
    cuda_capability: str = "N/A"

    flash_built_in: bool = False  # PyTorch compiled with Flash support
    can_flash: bool = False
    can_efficient: bool = False
    can_cudnn: bool = False

    # debug_reasons[backend] = list of reason strings from debug=True output
    debug_reasons: Dict[str, List[str]] = field(default_factory=lambda: {
        "flash": [], "efficient": [], "cudnn": []
    })

    q_shape: tuple = ()
    k_shape: tuple = ()
    v_shape: tuple = ()
    dtype: str = "unknown"
    dropout_p: float = 0.0
    is_causal: bool = False

    def recommended_backend(self) -> str:
        """Return the best available backend string."""
        if self.can_flash:
            return "flash"
        if self.can_efficient:
            return "efficient"
        if self.can_cudnn:
            return "cudnn"
        return "math"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "device_name": self.device_name,
            "cuda_capability": self.cuda_capability,
            "flash_built_in": self.flash_built_in,
            "can_flash": self.can_flash,
            "can_efficient": self.can_efficient,
            "can_cudnn": self.can_cudnn,
            "debug_reasons": self.debug_reasons,
            "q_shape": list(self.q_shape),
            "k_shape": list(self.k_shape),
            "v_shape": list(self.v_shape),
            "dtype": self.dtype,
            "dropout_p": self.dropout_p,
            "is_causal": self.is_causal,
            "recommended_backend": self.recommended_backend(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CapabilityReport":
        return cls(
            device=d.get("device", "cpu"),
            device_name=d.get("device_name", "CPU"),
            cuda_capability=d.get("cuda_capability", "N/A"),
            flash_built_in=d.get("flash_built_in", False),
            can_flash=d.get("can_flash", False),
            can_efficient=d.get("can_efficient", False),
            can_cudnn=d.get("can_cudnn", False),
            debug_reasons=d.get("debug_reasons", {"flash": [], "efficient": [], "cudnn": []}),
            q_shape=tuple(d.get("q_shape", ())),
            k_shape=tuple(d.get("k_shape", ())),
            v_shape=tuple(d.get("v_shape", ())),
            dtype=d.get("dtype", "unknown"),
            dropout_p=d.get("dropout_p", 0.0),
            is_causal=d.get("is_causal", False),
        )

    def format_summary(self) -> str:
        lines = [
            f"Capability Report ({self.device} / {self.device_name})",
            f"  CUDA capability : {self.cuda_capability}",
            f"  Flash built-in  : {self.flash_built_in}",
            f"  can_flash       : {self.can_flash}",
            f"  can_efficient   : {self.can_efficient}",
            f"  can_cudnn       : {self.can_cudnn}",
            f"  Recommended     : {self.recommended_backend()}",
        ]
        for backend, reasons in self.debug_reasons.items():
            if reasons:
                lines.append(f"  {backend} debug: " + "; ".join(reasons))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# AttentionMetrics — fields added to benchmark harness metrics.json
# ---------------------------------------------------------------------------


@dataclass
class AttentionMetrics:
    """
    Flat representation of attention fields for metrics.json injection.
    Mirrors the schema defined in references/benchmark-integration.md.
    """

    # Config-derived
    backend_policy: str = "auto"
    external_flash_attn_mode: str = "off"
    flash_attn_package_version: Optional[str] = None

    # Capability-derived
    sdpa_can_flash: bool = False
    sdpa_can_efficient: bool = False
    sdpa_can_cudnn: bool = False

    # Model-derived
    head_dim: int = 0
    num_heads: int = 0
    dropout_p_train: float = 0.0
    is_causal: bool = False
    mask_kind: str = "none"

    # Runtime-detected
    backend_actually_used: Optional[str] = None
    probe_device: str = "cpu"
    probe_dtype: str = "torch.float32"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend_policy": self.backend_policy,
            "sdpa_can_flash": self.sdpa_can_flash,
            "sdpa_can_efficient": self.sdpa_can_efficient,
            "sdpa_can_cudnn": self.sdpa_can_cudnn,
            "head_dim": self.head_dim,
            "num_heads": self.num_heads,
            "dropout_p_train": self.dropout_p_train,
            "is_causal": self.is_causal,
            "mask_kind": self.mask_kind,
            "backend_actually_used": self.backend_actually_used,
            "external_flash_attn_mode": self.external_flash_attn_mode,
            "flash_attn_package_version": self.flash_attn_package_version,
            "probe_device": self.probe_device,
            "probe_dtype": self.probe_dtype,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AttentionMetrics":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_mask_kind(mask_kind: str) -> None:
    if mask_kind not in ALLOWED_MASK_KIND:
        raise ValueError(
            f"mask_kind={mask_kind!r} not in {sorted(ALLOWED_MASK_KIND)}"
        )


def default_attention_config() -> AttentionConfig:
    """Return a validated default AttentionConfig."""
    cfg = AttentionConfig()
    cfg.validate()
    return cfg


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


def _run_self_tests() -> None:
    print("Running backend_config_template self-tests...")
    failures: List[str] = []

    # --- AttentionConfig validation ---
    try:
        cfg = AttentionConfig()
        cfg.validate()
        print("  PASS: default AttentionConfig validates")
    except Exception as e:
        failures.append(f"Default config validation: {e}")

    try:
        cfg = AttentionConfig(impl="invalid")
        cfg.validate()
        failures.append("Bad impl should have raised ValueError")
    except ValueError:
        print("  PASS: invalid impl raises ValueError")

    try:
        cfg = AttentionConfig(backend="nonexistent")
        cfg.validate()
        failures.append("Bad backend should have raised ValueError")
    except ValueError:
        print("  PASS: invalid backend raises ValueError")

    try:
        cfg = AttentionConfig(external_flash_attn="invalid")
        cfg.validate()
        failures.append("Bad external_flash_attn should have raised ValueError")
    except ValueError:
        print("  PASS: invalid external_flash_attn raises ValueError")

    try:
        cfg = AttentionConfig(dropout_policy="invalid")
        cfg.validate()
        failures.append("Bad dropout_policy should have raised ValueError")
    except ValueError:
        print("  PASS: invalid dropout_policy raises ValueError")

    # --- Serialization roundtrip ---
    try:
        original = AttentionConfig(
            impl="sdpa", backend="flash", force=True, log_backend=False,
            external_flash_attn="prefer", dropout_policy="train_only"
        )
        d = original.to_dict()
        restored = AttentionConfig.from_dict(d)
        assert restored.impl == original.impl
        assert restored.backend == original.backend
        assert restored.force == original.force
        assert restored.log_backend == original.log_backend
        assert restored.external_flash_attn == original.external_flash_attn
        print("  PASS: AttentionConfig to_dict/from_dict roundtrip")
    except Exception as e:
        failures.append(f"Roundtrip: {e}")

    try:
        cfg = AttentionConfig(backend="math", force=False)
        j = cfg.to_json()
        restored = AttentionConfig.from_json(j)
        assert restored.backend == "math"
        print("  PASS: AttentionConfig JSON roundtrip")
    except Exception as e:
        failures.append(f"JSON roundtrip: {e}")

    # --- BackendConfig.from_attention_config ---
    try:
        import torch
        from torch.nn.attention import SDPBackend

        bc_auto = BackendConfig.from_attention_config(AttentionConfig(backend="auto"))
        assert SDPBackend.FLASH_ATTENTION in bc_auto.backends
        assert SDPBackend.MATH in bc_auto.backends
        print("  PASS: BackendConfig auto includes Flash and Math")

        bc_math = BackendConfig.from_attention_config(AttentionConfig(backend="math"))
        assert SDPBackend.MATH in bc_math.backends
        assert SDPBackend.FLASH_ATTENTION not in bc_math.backends
        print("  PASS: BackendConfig math only has Math")

        bc_flash_force = BackendConfig.from_attention_config(
            AttentionConfig(backend="flash", force=True)
        )
        assert SDPBackend.MATH not in bc_flash_force.backends
        assert SDPBackend.FLASH_ATTENTION in bc_flash_force.backends
        print("  PASS: BackendConfig force=True removes Math")

    except ImportError:
        print("  SKIP: PyTorch not available — BackendConfig tests skipped")
    except Exception as e:
        failures.append(f"BackendConfig: {e}")

    # --- CapabilityReport ---
    try:
        cr = CapabilityReport(
            device="cpu", device_name="CPU", cuda_capability="N/A",
            can_flash=False, can_efficient=False, can_cudnn=False,
        )
        assert cr.recommended_backend() == "math"
        print("  PASS: CapabilityReport.recommended_backend() returns 'math' on CPU")

        cr_flash = CapabilityReport(can_flash=True, can_efficient=True, can_cudnn=False)
        assert cr_flash.recommended_backend() == "flash"
        print("  PASS: CapabilityReport.recommended_backend() prefers flash")
    except Exception as e:
        failures.append(f"CapabilityReport: {e}")

    try:
        cr = CapabilityReport(
            device="cuda:0", device_name="A100",
            q_shape=(2, 8, 512, 64), k_shape=(2, 8, 512, 64),
            v_shape=(2, 8, 512, 64), dtype="torch.float16",
            can_flash=True, can_efficient=True, can_cudnn=False,
        )
        d = cr.to_dict()
        restored = CapabilityReport.from_dict(d)
        assert restored.device == cr.device
        assert restored.can_flash == cr.can_flash
        assert tuple(d["q_shape"]) == cr.q_shape
        print("  PASS: CapabilityReport to_dict/from_dict roundtrip")
    except Exception as e:
        failures.append(f"CapabilityReport roundtrip: {e}")

    # --- AttentionMetrics ---
    try:
        am = AttentionMetrics(
            backend_policy="flash", head_dim=128, num_heads=16,
            sdpa_can_flash=True, is_causal=True,
        )
        d = am.to_dict()
        assert d["backend_policy"] == "flash"
        assert d["head_dim"] == 128
        assert d["sdpa_can_flash"] is True
        restored = AttentionMetrics.from_dict(d)
        assert restored.head_dim == 128
        print("  PASS: AttentionMetrics to_dict/from_dict roundtrip")
    except Exception as e:
        failures.append(f"AttentionMetrics: {e}")

    # --- validate_mask_kind ---
    try:
        validate_mask_kind("causal")
        print("  PASS: validate_mask_kind accepts 'causal'")
    except Exception as e:
        failures.append(f"validate_mask_kind: {e}")

    try:
        validate_mask_kind("invalid_kind")
        failures.append("validate_mask_kind should raise for invalid kind")
    except ValueError:
        print("  PASS: validate_mask_kind raises for invalid kind")

    # --- Summary ---
    if failures:
        print(f"\nFAILED {len(failures)} tests:")
        for f in failures:
            print(f"  FAIL: {f}")
        raise SystemExit(1)
    else:
        print(f"\nAll backend_config_template self-tests PASSED")


if __name__ == "__main__":
    _run_self_tests()
