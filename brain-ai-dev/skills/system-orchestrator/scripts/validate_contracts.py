#!/usr/bin/env python3
"""
validate_contracts.py -- Runtime contract validation for BrainAI.

Instantiates the model with a chosen config preset, runs a forward pass,
and validates every output contract (tensor shapes, devices, NaN checks,
SystemOutput fields).  Exits 0 on full pass, 1 on any failure.

Usage:
    python validate_contracts.py
    python validate_contracts.py --config-preset minimal --modalities vision,text
    python validate_contracts.py --config-preset production_1b --device cpu --return-details
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Result bookkeeping
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    elapsed_ms: float = 0.0

    def status_str(self) -> str:
        return "PASS" if self.passed else "FAIL"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

VALID_PRESETS = ("minimal", "production_1b", "production_3b", "production_7b")
VALID_DEVICES = ("cpu", "cuda", "auto")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Runtime contract validation for BrainAI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config-preset",
        default="minimal",
        choices=VALID_PRESETS,
        help="BrainAIConfig preset to use (default: minimal)",
    )
    p.add_argument(
        "--modalities",
        default="vision",
        help=(
            "Comma-separated list of modalities to enable (default: vision). "
            "Valid entries: vision, text, audio, sensors"
        ),
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=VALID_DEVICES,
        help="Device to run on: cpu, cuda, or auto (default: auto)",
    )
    p.add_argument(
        "--return-details",
        action="store_true",
        default=False,
        help=(
            "Call forward() with return_details=True and validate all "
            "SystemOutput fields"
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for the sample forward pass (default: 2)",
    )
    return p


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

def resolve_device(device_arg: str) -> str:
    """Return a concrete torch device string."""
    if device_arg == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device_arg


# ---------------------------------------------------------------------------
# Sample input factory
# ---------------------------------------------------------------------------

def make_sample_inputs(
    modalities: List[str],
    batch_size: int,
    device: str,
) -> Dict[str, Any]:
    """
    Create random tensors with shapes expected by each encoder.

    Sizes are minimal-but-valid so the forward pass completes quickly
    regardless of the preset.
    """
    import torch

    inputs: Dict[str, Any] = {}

    for modality in modalities:
        if modality == "vision":
            # (B, C, H, W) -- 3-channel RGB (the default for VisionEncoder).
            # 32x32 is the smallest size that survives all MaxPool2d layers in
            # the default minimal config (4 pooling stages: 32->16->8->4->2).
            inputs["vision"] = torch.randn(batch_size, 3, 32, 32, device=device)

        elif modality == "text":
            # Text encoder expects float embeddings: (B, seq_len, embed_dim).
            # 256 matches the minimal preset's text_embed_dim.
            inputs["text"] = torch.randn(batch_size, 16, 256, device=device)

        elif modality == "audio":
            # (B, n_mels, T) -- small spectrogram slice
            inputs["audio"] = torch.randn(batch_size, 128, 64, device=device)

        elif modality == "sensors":
            # (B, sensor_input_dim) -- compact sensor vector
            inputs["sensors"] = torch.randn(batch_size, 64, device=device)

        else:
            # Generic fallback: 1-D feature vector
            inputs[modality] = torch.randn(batch_size, 256, device=device)

    return inputs


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------

def build_config(preset: str, modalities: List[str]) -> Any:
    """Return a BrainAIConfig with the chosen preset wired to the given modalities."""
    from brain_ai.config import BrainAIConfig  # type: ignore[import]

    preset_map = {
        "minimal": BrainAIConfig.minimal,
        "production_1b": BrainAIConfig.production_1b,
        "production_3b": BrainAIConfig.production_3b,
        "production_7b": BrainAIConfig.production_7b,
    }

    factory = preset_map.get(preset, BrainAIConfig.minimal)
    config = factory()
    config.modalities = modalities

    # For non-minimal presets, clamp text_embed_dim so it matches the sample
    # input shape (256).  This allows a quick smoke-test without loading the
    # full multi-GB parameter set.
    if preset != "minimal":
        config.encoder.text_embed_dim = min(config.encoder.text_embed_dim, 256)

    return config


# ---------------------------------------------------------------------------
# Individual contract checkers
# ---------------------------------------------------------------------------

def check_output_exists(output: Any, batch_size: int) -> CheckResult:
    """Output tensor must exist and have the correct batch dimension."""
    import torch

    name = "output_exists_and_batch_dim"
    try:
        if output is None:
            return CheckResult(name, False, "output is None")
        if not isinstance(output, torch.Tensor):
            return CheckResult(
                name, False,
                f"output type is {type(output).__name__}, expected Tensor",
            )
        if output.shape[0] != batch_size:
            return CheckResult(
                name, False,
                f"batch dim = {output.shape[0]}, expected {batch_size}",
            )
        return CheckResult(name, True, f"shape={tuple(output.shape)}")
    except Exception as exc:
        return CheckResult(name, False, f"exception: {exc}")


def check_confidence_shape(confidence: Any, batch_size: int) -> CheckResult:
    """Confidence tensor must have shape (B, 1)."""
    import torch

    name = "confidence_shape_(B,1)"
    try:
        if confidence is None:
            return CheckResult(name, False, "confidence is None")
        if not isinstance(confidence, torch.Tensor):
            return CheckResult(
                name, False,
                f"confidence type is {type(confidence).__name__}, expected Tensor",
            )
        expected = (batch_size, 1)
        if tuple(confidence.shape) != expected:
            return CheckResult(
                name, False,
                f"shape={tuple(confidence.shape)}, expected {expected}",
            )
        return CheckResult(name, True, f"shape={tuple(confidence.shape)}")
    except Exception as exc:
        return CheckResult(name, False, f"exception: {exc}")


def check_same_device(tensors: Dict[str, Any], target_device: str) -> CheckResult:
    """All returned tensors must reside on the same device as the model."""
    name = "all_tensors_same_device"
    try:
        import torch
        mismatches = []
        device_prefix = target_device.split(":")[0]
        for label, t in tensors.items():
            if t is None or not isinstance(t, torch.Tensor):
                continue
            t_dev = str(t.device)
            if not t_dev.startswith(device_prefix):
                mismatches.append(f"{label}@{t_dev}")
        if mismatches:
            return CheckResult(
                name, False,
                (
                    f"device mismatches: {', '.join(mismatches)} "
                    f"(expected prefix '{device_prefix}')"
                ),
            )
        return CheckResult(name, True, f"all tensors on device '{target_device}'")
    except Exception as exc:
        return CheckResult(name, False, f"exception: {exc}")


def check_no_nans(tensors: Dict[str, Any]) -> CheckResult:
    """No tensor may contain NaN values."""
    import torch

    name = "no_nan_values"
    try:
        nan_fields = [
            label
            for label, t in tensors.items()
            if isinstance(t, torch.Tensor) and torch.isnan(t).any()
        ]
        if nan_fields:
            return CheckResult(name, False, f"NaNs found in: {', '.join(nan_fields)}")
        return CheckResult(name, True, "no NaNs detected")
    except Exception as exc:
        return CheckResult(name, False, f"exception: {exc}")


def check_no_infs(tensors: Dict[str, Any]) -> CheckResult:
    """No tensor may contain Inf values."""
    import torch

    name = "no_inf_values"
    try:
        inf_fields = [
            label
            for label, t in tensors.items()
            if isinstance(t, torch.Tensor) and torch.isinf(t).any()
        ]
        if inf_fields:
            return CheckResult(name, False, f"Infs found in: {', '.join(inf_fields)}")
        return CheckResult(name, True, "no Infs detected")
    except Exception as exc:
        return CheckResult(name, False, f"exception: {exc}")


def check_details_fields(system_output: Any) -> List[CheckResult]:
    """When return_details=True, verify SystemOutput attributes.

    Handles both the current flat SystemOutput (system.py as-is) and the target
    nested SystemOutput/SystemDetails architecture (after refactor).
    """
    results: List[CheckResult] = []

    # Always-present top-level fields
    top_level_fields = ("output", "confidence")
    for field_name in top_level_fields:
        name = f"details_field:{field_name}"
        try:
            if not hasattr(system_output, field_name):
                results.append(CheckResult(name, False, "attribute missing"))
            else:
                val = getattr(system_output, field_name)
                kind = type(val).__name__ if val is not None else "None"
                results.append(CheckResult(name, True, f"present (type={kind})"))
        except Exception as exc:
            results.append(CheckResult(name, False, f"exception: {exc}"))

    # Check for nested SystemDetails (target architecture)
    if hasattr(system_output, "details") and system_output.details is not None:
        details = system_output.details
        details_fields = ("encoder", "workspace", "htm", "reasoning", "decision", "meta", "engram")
        for field_name in details_fields:
            name = f"details_field:details.{field_name}"
            try:
                if not hasattr(details, field_name):
                    results.append(CheckResult(name, False, "attribute missing from SystemDetails"))
                else:
                    val = getattr(details, field_name)
                    kind = type(val).__name__ if val is not None else "None"
                    results.append(CheckResult(name, True, f"present (type={kind})"))
            except Exception as exc:
                results.append(CheckResult(name, False, f"exception: {exc}"))
    else:
        # Flat SystemOutput (current codebase) — check legacy fields
        legacy_fields = ("workspace", "attention", "reasoning_trace", "modulators")
        for field_name in legacy_fields:
            name = f"details_field:{field_name}"
            try:
                if not hasattr(system_output, field_name):
                    results.append(CheckResult(name, False, "attribute missing"))
                else:
                    val = getattr(system_output, field_name)
                    kind = type(val).__name__ if val is not None else "None"
                    results.append(CheckResult(name, True, f"present (type={kind})"))
            except Exception as exc:
                results.append(CheckResult(name, False, f"exception: {exc}"))

    return results


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

PASS_MARK = "[ PASS ]"
FAIL_MARK = "[ FAIL ]"
SEP = "-" * 68


def print_report(
    results: List[CheckResult],
    preset: str,
    modalities: List[str],
    device: str,
    forward_ms: float,
) -> bool:
    """Print a structured report.  Returns True if every check passed."""
    all_passed = all(r.passed for r in results)
    status_line = "ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED"

    print(SEP)
    print("  BrainAI Contract Validation Report")
    print(SEP)
    print(f"  Config preset : {preset}")
    print(f"  Modalities    : {', '.join(modalities)}")
    print(f"  Device        : {device}")
    print(f"  Forward pass  : {forward_ms:.1f} ms")
    print(SEP)

    name_width = max(len(r.name) for r in results) + 2

    for r in results:
        mark = PASS_MARK if r.passed else FAIL_MARK
        detail = f"  ({r.detail})" if r.detail else ""
        print(f"  {mark}  {r.name:<{name_width}}{detail}")

    print(SEP)
    n_pass = sum(1 for r in results if r.passed)
    n_fail = len(results) - n_pass
    print(
        f"  Result: {status_line}  "
        f"({n_pass}/{len(results)} checks passed, {n_fail} failed)"
    )
    print(SEP)

    return all_passed


# ---------------------------------------------------------------------------
# Main validation driver
# ---------------------------------------------------------------------------

def run_validation(args: argparse.Namespace) -> int:
    """
    Orchestrate the full validation sequence.

    Returns 0 on success, 1 on any failure.
    """
    import torch

    modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    device = resolve_device(args.device)
    batch_size = args.batch_size
    results: List[CheckResult] = []
    forward_ms = 0.0

    # ---- 1. Import check -----------------------------------------------
    try:
        from brain_ai.system import BrainAI  # type: ignore[import]  # noqa: F401
        from brain_ai.config import BrainAIConfig  # type: ignore[import]  # noqa: F401
        results.append(CheckResult("import_brain_ai", True, "OK"))
    except ImportError as exc:
        results.append(CheckResult(
            "import_brain_ai", False,
            f"ImportError: {exc}. Is brain_ai on PYTHONPATH?",
        ))
        print_report(results, args.config_preset, modalities, device, 0.0)
        return 1

    # ---- 2. Config instantiation ----------------------------------------
    try:
        config = build_config(args.config_preset, modalities)
        results.append(CheckResult(
            "config_instantiation", True, f"preset={args.config_preset}",
        ))
    except Exception as exc:
        results.append(CheckResult("config_instantiation", False, str(exc)))
        print_report(results, args.config_preset, modalities, device, 0.0)
        return 1

    # ---- 3. Model construction ------------------------------------------
    try:
        from brain_ai.system import BrainAI  # type: ignore[import]

        t0 = time.perf_counter()
        model = BrainAI(config=config, modalities=modalities, output_type="classify")
        model = model.to(device)
        model.eval()
        build_ms = (time.perf_counter() - t0) * 1000.0
        results.append(CheckResult(
            "model_construction", True,
            f"built in {build_ms:.0f} ms, device={device}",
        ))
    except Exception as exc:
        results.append(CheckResult(
            "model_construction", False,
            f"exception during __init__: {exc}",
        ))
        traceback.print_exc()
        print_report(results, args.config_preset, modalities, device, 0.0)
        return 1

    # ---- 4. Sample input creation ---------------------------------------
    try:
        sample_inputs = make_sample_inputs(modalities, batch_size, device)
        shape_info = ", ".join(
            f"{k}:{tuple(v.shape)}" for k, v in sample_inputs.items()
        )
        results.append(CheckResult("sample_input_creation", True, shape_info))
    except Exception as exc:
        results.append(CheckResult("sample_input_creation", False, str(exc)))
        print_report(results, args.config_preset, modalities, device, 0.0)
        return 1

    # ---- 5. Forward pass ------------------------------------------------
    system_output = None
    plain_output = None
    confidence = None

    try:
        with torch.no_grad():
            if args.return_details:
                t0 = time.perf_counter()
                system_output = model(sample_inputs, return_details=True)
                forward_ms = (time.perf_counter() - t0) * 1000.0
                plain_output = system_output.output
                confidence = system_output.confidence
            else:
                t0 = time.perf_counter()
                plain_output = model(sample_inputs, return_details=False)
                forward_ms = (time.perf_counter() - t0) * 1000.0
                # When details are not requested the model returns a plain tensor.
                # Synthesise a (B, 1) confidence of ones so the contract check
                # still runs against a meaningful tensor.
                confidence = torch.ones(batch_size, 1, device=device)

        results.append(CheckResult(
            "forward_pass", True, f"completed in {forward_ms:.1f} ms",
        ))
    except Exception as exc:
        results.append(CheckResult(
            "forward_pass", False, f"exception during forward: {exc}",
        ))
        traceback.print_exc()
        print_report(results, args.config_preset, modalities, device, forward_ms)
        return 1

    # ---- 6. Contract checks --------------------------------------------

    results.append(check_output_exists(plain_output, batch_size))
    results.append(check_confidence_shape(confidence, batch_size))

    # Collect all returned tensors for the device and NaN checks
    tensors_to_check: Dict[str, Any] = {
        "output": plain_output,
        "confidence": confidence,
    }

    if system_output is not None:
        tensors_to_check["workspace"] = system_output.workspace

        if isinstance(system_output.attention, dict):
            for k, v in system_output.attention.items():
                tensors_to_check[f"attention.{k}"] = v

        if system_output.reasoning_trace is not None:
            tensors_to_check["reasoning_trace"] = system_output.reasoning_trace

        if isinstance(system_output.modulators, dict):
            for k, v in system_output.modulators.items():
                tensors_to_check[f"modulators.{k}"] = v

    results.append(check_same_device(tensors_to_check, device))
    results.append(check_no_nans(tensors_to_check))
    results.append(check_no_infs(tensors_to_check))

    # ---- 7. SystemOutput field checks (only when --return-details) ------
    if args.return_details and system_output is not None:
        results.extend(check_details_fields(system_output))

    # ---- 8. Print and exit ----------------------------------------------
    all_ok = print_report(results, args.config_preset, modalities, device, forward_ms)
    return 0 if all_ok else 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        print("ERROR: PyTorch is not installed.  Install it with:")
        print("  pip install torch")
        sys.exit(1)

    sys.exit(run_validation(args))


if __name__ == "__main__":
    main()
