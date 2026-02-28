"""
bench_attn_fields_template.py
==============================
Benchmark harness extension for attention fields.

Functions:
    collect_attention_metrics  -- extract attention metrics from model + config
    inject_attention_fields    -- merge AttentionMetrics into a metrics dict
    format_attention_report    -- human-readable summary string

Self-tests: python bench_attn_fields_template.py
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from backend_config_template import (
    AttentionConfig,
    AttentionMetrics,
    CapabilityReport,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# collect_attention_metrics
# ---------------------------------------------------------------------------


def collect_attention_metrics(
    model: nn.Module,
    cfg: AttentionConfig,
    sample_batch: Optional[Dict[str, torch.Tensor]] = None,
    head_dim: Optional[int] = None,
    num_heads: Optional[int] = None,
    is_causal: bool = False,
    mask_kind: str = "none",
) -> AttentionMetrics:
    """
    Extract attention-relevant metrics from a model and config.

    Attempts to introspect the model for head_dim and num_heads via common
    attribute names. Falls back to explicitly provided values if introspection
    fails.

    Args:
        model        : The nn.Module to inspect.
        cfg          : AttentionConfig describing the attention setup.
        sample_batch : Dict of tensors to build a probe from. If None, a
                       synthetic probe is created using discovered head_dim.
        head_dim     : Override head_dim (use when introspection fails).
        num_heads    : Override num_heads.
        is_causal    : Whether the model uses causal attention.
        mask_kind    : One of "none", "causal", "float_additive",
                       "bool_mask", "padding".

    Returns:
        AttentionMetrics ready for inject_attention_fields().
    """
    # ------------------------------------------------------------------
    # 1. Introspect model for head_dim / num_heads
    # ------------------------------------------------------------------
    resolved_head_dim = head_dim or _infer_head_dim(model)
    resolved_num_heads = num_heads or _infer_num_heads(model)

    # ------------------------------------------------------------------
    # 2. Determine probe device and dtype
    # ------------------------------------------------------------------
    probe_device, probe_dtype = _infer_probe_device_dtype(model, sample_batch)

    # ------------------------------------------------------------------
    # 3. Run capability probe
    # ------------------------------------------------------------------
    cap = _run_capability_probe(
        head_dim=resolved_head_dim,
        num_heads=resolved_num_heads,
        device=probe_device,
        dtype=probe_dtype,
    )

    # ------------------------------------------------------------------
    # 4. Check flash-attn package version
    # ------------------------------------------------------------------
    fa_version: Optional[str] = None
    if cfg.external_flash_attn != "off":
        try:
            import flash_attn
            fa_version = getattr(flash_attn, "__version__", "unknown")
        except ImportError:
            fa_version = None

    # ------------------------------------------------------------------
    # 5. Assemble AttentionMetrics
    # ------------------------------------------------------------------
    return AttentionMetrics(
        backend_policy=cfg.backend,
        external_flash_attn_mode=cfg.external_flash_attn,
        flash_attn_package_version=fa_version,
        sdpa_can_flash=cap.can_flash,
        sdpa_can_efficient=cap.can_efficient,
        sdpa_can_cudnn=cap.can_cudnn,
        head_dim=resolved_head_dim,
        num_heads=resolved_num_heads,
        dropout_p_train=_infer_dropout_p(model),
        is_causal=is_causal,
        mask_kind=mask_kind,
        backend_actually_used=None,  # filled in by runtime profiler if used
        probe_device=str(probe_device),
        probe_dtype=str(probe_dtype),
    )


# ---------------------------------------------------------------------------
# inject_attention_fields
# ---------------------------------------------------------------------------


def inject_attention_fields(
    metrics_dict: Dict[str, Any],
    attn_metrics: AttentionMetrics,
) -> Dict[str, Any]:
    """
    Inject AttentionMetrics into an existing metrics dict under the 'attn' key.

    This extends the perf-regression-gate metrics.json schema without touching
    any existing top-level keys.

    Args:
        metrics_dict : Existing metrics dict (modified in place AND returned).
        attn_metrics : Attention metrics to inject.

    Returns:
        The same metrics_dict with an 'attn' key added.
    """
    metrics_dict["attn"] = attn_metrics.to_dict()
    return metrics_dict


# ---------------------------------------------------------------------------
# format_attention_report
# ---------------------------------------------------------------------------


def format_attention_report(attn_metrics: AttentionMetrics) -> str:
    """
    Format AttentionMetrics as a human-readable summary string.

    Intended for console output at benchmark startup or in log files.
    """
    can_flash = attn_metrics.sdpa_can_flash
    can_eff = attn_metrics.sdpa_can_efficient
    can_cudnn = attn_metrics.sdpa_can_cudnn

    # Infer recommended backend from capability
    if can_flash:
        recommended = "flash"
    elif can_eff:
        recommended = "efficient"
    elif can_cudnn:
        recommended = "cudnn"
    else:
        recommended = "math"

    lines = [
        "=" * 60,
        "Attention Configuration Summary",
        "=" * 60,
        f"  Backend policy       : {attn_metrics.backend_policy}",
        f"  External flash-attn  : {attn_metrics.external_flash_attn_mode}",
        f"  Flash-attn version   : {attn_metrics.flash_attn_package_version or 'N/A'}",
        "",
        "  SDPA Capability (built-in):",
        f"    can_flash           : {can_flash}",
        f"    can_efficient       : {can_eff}",
        f"    can_cudnn           : {can_cudnn}",
        f"    recommended         : {recommended}",
        "",
        "  Model Attention Config:",
        f"    head_dim            : {attn_metrics.head_dim}",
        f"    num_heads           : {attn_metrics.num_heads}",
        f"    dropout_p (train)   : {attn_metrics.dropout_p_train}",
        f"    is_causal           : {attn_metrics.is_causal}",
        f"    mask_kind           : {attn_metrics.mask_kind}",
        "",
        f"  Probe device         : {attn_metrics.probe_device}",
        f"  Probe dtype          : {attn_metrics.probe_dtype}",
    ]
    if attn_metrics.backend_actually_used:
        lines.append(f"  Backend used (runtime): {attn_metrics.backend_actually_used}")
    lines.append("=" * 60)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Regression check helper
# ---------------------------------------------------------------------------


def check_backend_regression(
    baseline: Dict[str, Any],
    current: Dict[str, Any],
) -> list:
    """
    Compare attention fields from two metrics dicts and return warning strings.

    Used by the perf-regression-gate to flag backend regressions.

    Returns:
        List of warning strings (empty if no regression detected).
    """
    warnings = []
    b_attn = baseline.get("attn", {})
    c_attn = current.get("attn", {})

    if not b_attn or not c_attn:
        return warnings

    b_backend = b_attn.get("backend_actually_used")
    c_backend = c_attn.get("backend_actually_used")
    if b_backend and c_backend and b_backend != c_backend:
        warnings.append(
            f"ATTENTION BACKEND CHANGED: {b_backend!r} -> {c_backend!r}. "
            "Throughput regression may be caused by backend change."
        )

    if b_attn.get("sdpa_can_flash") and not c_attn.get("sdpa_can_flash"):
        warnings.append(
            "Flash backend was available in baseline but is NOT available now. "
            "Check PyTorch version, CUDA version, or head_dim/dtype changes."
        )

    if (b_attn.get("backend_policy") != c_attn.get("backend_policy")):
        warnings.append(
            f"Backend policy changed: "
            f"{b_attn.get('backend_policy')!r} -> "
            f"{c_attn.get('backend_policy')!r}"
        )

    return warnings


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _infer_head_dim(model: nn.Module) -> int:
    """Try to extract head_dim from common model attributes."""
    # Common attribute names across different architectures
    candidates = [
        "head_dim", "d_head", "attention_head_size", "per_head_dim",
        "hidden_size_per_attention_head",
    ]
    for attr in candidates:
        val = _deep_getattr(model, attr)
        if val is not None and isinstance(val, int) and val > 0:
            return val

    # Fallback: derive from hidden_size / num_heads
    hidden = _deep_getattr(model, "hidden_size") or _deep_getattr(model, "d_model")
    heads = _infer_num_heads(model)
    if hidden and heads and heads > 0:
        return hidden // heads

    logger.debug(
        "Could not infer head_dim from model; defaulting to 64. "
        "Pass head_dim explicitly to collect_attention_metrics()."
    )
    return 64


def _infer_num_heads(model: nn.Module) -> int:
    """Try to extract num_heads from common model attributes."""
    candidates = [
        "num_heads", "num_attention_heads", "n_heads", "nheads",
        "num_kv_heads", "n_head",
    ]
    for attr in candidates:
        val = _deep_getattr(model, attr)
        if val is not None and isinstance(val, int) and val > 0:
            return val
    return 8  # safe default


def _infer_dropout_p(model: nn.Module) -> float:
    """Try to extract attention dropout probability from model."""
    candidates = [
        "attention_dropout", "attn_drop", "attn_dropout", "dropout_p",
        "attention_probs_dropout_prob",
    ]
    for attr in candidates:
        val = _deep_getattr(model, attr)
        if val is not None and isinstance(val, (int, float)):
            return float(val)
    return 0.0


def _deep_getattr(model: nn.Module, attr: str) -> Any:
    """
    Search for `attr` in the model and its config (if present).
    Returns the first non-None value found, or None.
    """
    # Direct attribute
    val = getattr(model, attr, None)
    if val is not None:
        return val

    # Config attribute (HuggingFace-style)
    cfg = getattr(model, "config", None)
    if cfg is not None:
        val = getattr(cfg, attr, None)
        if val is not None:
            return val

    return None


def _infer_probe_device_dtype(
    model: nn.Module,
    sample_batch: Optional[Dict[str, torch.Tensor]],
) -> tuple:
    """Return (device, dtype) for the capability probe."""
    # Try to get device/dtype from sample_batch first
    if sample_batch:
        for v in sample_batch.values():
            if isinstance(v, torch.Tensor):
                return v.device, v.dtype

    # Fall back to model parameters
    try:
        p = next(model.parameters())
        return p.device, p.dtype
    except StopIteration:
        pass

    # Final fallback
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    return device, dtype


def _run_capability_probe(
    head_dim: int,
    num_heads: int,
    device: torch.device,
    dtype: torch.dtype,
    seq_len: int = 64,
    batch_size: int = 1,
) -> CapabilityReport:
    """Run AttentionAuditor.probe_runtime with synthetic tensors."""
    try:
        from attention_auditor_template import AttentionAuditor
        auditor = AttentionAuditor()
        shape = (batch_size, num_heads, seq_len, head_dim)

        # Use float32 for the probe on CPU (flash/efficient require CUDA anyway)
        probe_dtype = dtype
        if device.type == "cpu" and dtype not in (torch.float32,):
            probe_dtype = torch.float32

        q = torch.randn(*shape, dtype=probe_dtype, device=device)
        return auditor.probe_runtime(q, q.clone(), q.clone(), None, 0.0, False)
    except Exception as exc:
        logger.warning("Capability probe failed: %s", exc)
        return CapabilityReport(device=str(device))


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


def _run_self_tests() -> None:
    print("Running bench_attn_fields_template self-tests...")
    failures = []

    # --- Build a simple model for testing ---
    class MockAttentionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_heads = 8
            self.head_dim = 64
            self.attention_dropout = 0.1
            self.linear = nn.Linear(512, 512)

        def forward(self, x):
            return self.linear(x)

    model = MockAttentionModel()
    cfg = AttentionConfig(backend="auto", external_flash_attn="off")

    # --- collect_attention_metrics ---
    try:
        metrics = collect_attention_metrics(model, cfg)
        assert isinstance(metrics, AttentionMetrics)
        assert metrics.head_dim == 64, f"Expected head_dim=64, got {metrics.head_dim}"
        assert metrics.num_heads == 8, f"Expected num_heads=8, got {metrics.num_heads}"
        assert metrics.backend_policy == "auto"
        print("  PASS: collect_attention_metrics extracts correct fields")
    except Exception as e:
        failures.append(f"collect_attention_metrics: {e}")

    try:
        metrics = collect_attention_metrics(model, cfg, is_causal=True, mask_kind="causal")
        assert metrics.is_causal is True
        assert metrics.mask_kind == "causal"
        print("  PASS: collect_attention_metrics is_causal and mask_kind")
    except Exception as e:
        failures.append(f"collect_attention_metrics causal: {e}")

    # --- collect_attention_metrics with explicit overrides ---
    try:
        metrics = collect_attention_metrics(model, cfg, head_dim=128, num_heads=16)
        assert metrics.head_dim == 128
        assert metrics.num_heads == 16
        print("  PASS: collect_attention_metrics respects explicit head_dim/num_heads overrides")
    except Exception as e:
        failures.append(f"collect_attention_metrics overrides: {e}")

    # --- inject_attention_fields ---
    try:
        metrics = collect_attention_metrics(model, cfg)
        base_dict = {"tokens_per_sec": 12345, "loss": 2.3}
        result = inject_attention_fields(base_dict, metrics)
        assert "attn" in result
        assert result["tokens_per_sec"] == 12345  # original key preserved
        attn_section = result["attn"]
        assert "backend_policy" in attn_section
        assert "sdpa_can_flash" in attn_section
        assert "head_dim" in attn_section
        print("  PASS: inject_attention_fields adds 'attn' key, preserves existing keys")
    except Exception as e:
        failures.append(f"inject_attention_fields: {e}")

    # --- format_attention_report ---
    try:
        metrics = collect_attention_metrics(model, cfg, head_dim=64, num_heads=8)
        report_str = format_attention_report(metrics)
        assert isinstance(report_str, str)
        assert len(report_str) > 50
        assert "backend_policy" in report_str or "Backend policy" in report_str
        assert "head_dim" in report_str or "head" in report_str.lower()
        print("  PASS: format_attention_report returns non-empty string with expected content")
    except Exception as e:
        failures.append(f"format_attention_report: {e}")

    # --- check_backend_regression ---
    try:
        baseline = {"attn": {"sdpa_can_flash": True, "backend_actually_used": "flash",
                              "backend_policy": "auto"}}
        current = {"attn": {"sdpa_can_flash": False, "backend_actually_used": "math",
                             "backend_policy": "auto"}}
        warnings = check_backend_regression(baseline, current)
        assert len(warnings) > 0
        assert any("flash" in w.lower() for w in warnings)
        print("  PASS: check_backend_regression detects flash->math regression")
    except Exception as e:
        failures.append(f"check_backend_regression: {e}")

    try:
        baseline = {"attn": {"sdpa_can_flash": True, "backend_actually_used": "flash",
                              "backend_policy": "auto"}}
        current = {"attn": {"sdpa_can_flash": True, "backend_actually_used": "flash",
                             "backend_policy": "auto"}}
        warnings = check_backend_regression(baseline, current)
        assert len(warnings) == 0
        print("  PASS: check_backend_regression returns empty list when no regression")
    except Exception as e:
        failures.append(f"check_backend_regression no-op: {e}")

    # --- _infer_head_dim / _infer_num_heads ---
    try:
        assert _infer_head_dim(model) == 64
        assert _infer_num_heads(model) == 8
        print("  PASS: _infer_head_dim and _infer_num_heads from model attributes")
    except Exception as e:
        failures.append(f"infer helpers: {e}")

    # --- AttentionMetrics roundtrip ---
    try:
        am = AttentionMetrics(
            backend_policy="flash", head_dim=64, num_heads=8,
            sdpa_can_flash=True, is_causal=True, mask_kind="causal",
        )
        d = am.to_dict()
        restored = AttentionMetrics.from_dict(d)
        assert restored.backend_policy == "flash"
        assert restored.head_dim == 64
        assert restored.sdpa_can_flash is True
        print("  PASS: AttentionMetrics to_dict/from_dict roundtrip")
    except Exception as e:
        failures.append(f"AttentionMetrics roundtrip: {e}")

    # --- Summary ---
    if failures:
        print(f"\nFAILED {len(failures)} tests:")
        for f in failures:
            print(f"  FAIL: {f}")
        raise SystemExit(1)
    else:
        print(f"\nAll bench_attn_fields_template self-tests PASSED")


if __name__ == "__main__":
    _run_self_tests()
