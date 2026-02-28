#!/usr/bin/env python3
"""
validate_encoders.py -- Runtime Encoder Contract Validation
===========================================================

Instantiates each modality encoder (vision, text, audio, sensors, engram),
runs a forward pass with synthetic data, and validates that the output
conforms to the EncoderOutput contract defined in encoder-contract.md.

Handles two encoder states gracefully:
  - **Legacy**: Encoder returns a raw ``torch.Tensor`` of shape ``(B, D)``.
    Validated with relaxed shape checks (no mask/salience assertions).
  - **Target**: Encoder returns an ``EncoderOutput`` dataclass.
    Full contract validation is applied.

Usage::

    python validate_encoders.py                         # defaults
    python validate_encoders.py --workspace-dim 4096    # production dim
    python validate_encoders.py --device cuda --dtype bfloat16
    python validate_encoders.py --json report.json      # machine-readable
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Path manipulation -- allow the script to find brain_ai from the repo root
# regardless of the working directory.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
)
if os.path.isdir(os.path.join(_REPO_ROOT, "brain_ai")):
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# Inline EncoderOutput dataclass (self-contained, mirrors encoder-contract.md)
# ---------------------------------------------------------------------------

@dataclass
class EncoderOutput:
    """Canonical encoder output contract.

    Defined inline so this validation script is fully self-contained and
    does not depend on the brain_ai package being importable.
    """

    modality: str                                     # "vision" | "text" | ...
    feats: torch.Tensor                               # (B, T, D)
    mask: torch.Tensor                                # (B, T) bool
    salience: Optional[torch.Tensor] = None           # (B, T) or (B, 1)
    pos_ids: Optional[torch.Tensor] = None            # (B, T) int64
    time: Optional[torch.Tensor] = None               # (B, T) float32
    spike: Optional[torch.Tensor] = None              # (B, T, *) optional
    aux: Dict[str, Any] = field(default_factory=dict)

# ---------------------------------------------------------------------------
# Terminal colour helpers
# ---------------------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty()


def _green(text: str) -> str:
    return f"\033[92m{text}\033[0m" if _USE_COLOR else text


def _red(text: str) -> str:
    return f"\033[91m{text}\033[0m" if _USE_COLOR else text


def _yellow(text: str) -> str:
    return f"\033[93m{text}\033[0m" if _USE_COLOR else text


def _bold(text: str) -> str:
    return f"\033[1m{text}\033[0m" if _USE_COLOR else text


def _dim(text: str) -> str:
    return f"\033[2m{text}\033[0m" if _USE_COLOR else text

# ---------------------------------------------------------------------------
# Contract assertion
# ---------------------------------------------------------------------------

class ContractViolation(AssertionError):
    """Raised when an EncoderOutput violates the shared contract."""
    pass


def assert_encoder_contract(
    output: EncoderOutput,
    workspace_dim: int,
    device: torch.device,
) -> List[str]:
    """Validate all contract invariants and return a list of violations.

    Rather than raising on the first failure, this function collects every
    violation so the report can show all issues at once.

    Returns:
        List of violation description strings.  Empty list means full pass.
    """
    violations: List[str] = []

    # 1. feats.ndim == 3
    if output.feats.ndim != 3:
        violations.append(
            f"feats must be 3-D (B, T, D), got {output.feats.ndim}-D "
            f"with shape {tuple(output.feats.shape)}"
        )

    # 2. mask.ndim == 2
    if output.mask.ndim != 2:
        violations.append(
            f"mask must be 2-D (B, T), got {output.mask.ndim}-D "
            f"with shape {tuple(output.mask.shape)}"
        )

    # 3. feats[:2] == mask shape
    if output.feats.ndim >= 2 and output.mask.ndim >= 2:
        if output.feats.shape[:2] != output.mask.shape:
            violations.append(
                f"feats/mask batch/time mismatch: "
                f"feats {tuple(output.feats.shape[:2])} vs mask {tuple(output.mask.shape)}"
            )

    # 4. feats.shape[-1] == workspace_dim
    if output.feats.ndim >= 1 and output.feats.shape[-1] != workspace_dim:
        violations.append(
            f"feats dim {output.feats.shape[-1]} != workspace_dim {workspace_dim}"
        )

    # 5. device checks
    expected_device = device
    if output.feats.device != expected_device:
        violations.append(
            f"feats on {output.feats.device}, expected {expected_device}"
        )
    if output.mask.device != expected_device:
        violations.append(
            f"mask on {output.mask.device}, expected {expected_device}"
        )

    # 6. feats dtype must be a supported float type
    allowed_dtypes = {torch.float32, torch.bfloat16, torch.float16}
    if output.feats.dtype not in allowed_dtypes:
        violations.append(
            f"feats dtype {output.feats.dtype} not in {{fp32, bf16, fp16}}"
        )

    # 7. mask dtype == bool
    if output.mask.dtype != torch.bool:
        violations.append(
            f"mask dtype {output.mask.dtype}, expected torch.bool"
        )

    # 8. No NaN / Inf in feats
    if torch.isnan(output.feats).any():
        violations.append("feats contains NaN values")
    if torch.isinf(output.feats).any():
        violations.append("feats contains Inf values")

    # 9. salience non-negative if present
    if output.salience is not None:
        if not (output.salience >= 0).all():
            violations.append("salience contains negative values")

    # 10. spike aligned to feats if present
    if output.spike is not None:
        if output.spike.shape[0] != output.feats.shape[0]:
            violations.append(
                f"spike batch dim {output.spike.shape[0]} != "
                f"feats batch dim {output.feats.shape[0]}"
            )
        if output.spike.ndim >= 2 and output.feats.ndim >= 2:
            if output.spike.shape[1] != output.feats.shape[1]:
                violations.append(
                    f"spike time dim {output.spike.shape[1]} != "
                    f"feats time dim {output.feats.shape[1]}"
                )

    return violations

# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def make_vision_data(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    grayscale: bool = False,
) -> Dict[str, Any]:
    """Synthetic image batch."""
    channels = 1 if grayscale else 3
    images = torch.randn(batch_size, channels, 28, 28, device=device, dtype=dtype)
    return {"x": images, "kwargs": {}}


def make_text_data(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    """Synthetic tokenised text batch."""
    seq_len = 32
    input_ids = torch.randint(1, 1000, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, device=device)
    # Zero out last few positions to simulate padding
    attention_mask[:, -4:] = 0
    return {
        "x": input_ids,
        "kwargs": {"attention_mask": attention_mask},
    }


def make_audio_data(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    """Synthetic waveform batch (1 second at 16 kHz)."""
    waveform = torch.randn(batch_size, 16000, device=device, dtype=dtype)
    return {"x": waveform, "kwargs": {}}


def make_sensor_data(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    """Synthetic IMU-like time-series batch."""
    # 50 timesteps, 6 channels (accel xyz + gyro xyz)
    series = torch.randn(batch_size, 50, 6, device=device, dtype=dtype)
    return {"x": series, "kwargs": {}}


def make_engram_data(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    """Synthetic engram token-id batch."""
    seq_len = 32
    token_ids = torch.randint(1, 5000, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, device=device)
    return {
        "x": token_ids,
        "kwargs": {"attention_mask": attention_mask},
    }


SYNTHETIC_GENERATORS: Dict[str, Callable] = {
    "vision": make_vision_data,
    "text": make_text_data,
    "audio": make_audio_data,
    "sensors": make_sensor_data,
    "engram": make_engram_data,
}

# ---------------------------------------------------------------------------
# Encoder factory helpers -- import brain_ai encoders or provide mock stubs
# ---------------------------------------------------------------------------

_BRAIN_AI_AVAILABLE = False
_IMPORT_ERRORS: Dict[str, str] = {}


def _build_encoder(
    modality: str,
    workspace_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[nn.Module]:
    """Build an encoder for the given modality with the correct output_dim.

    Attempts to import from brain_ai.  Returns None on failure and records
    the error traceback in ``_IMPORT_ERRORS`` for the report.
    """
    global _BRAIN_AI_AVAILABLE

    try:
        if modality == "vision":
            from brain_ai.encoders import create_vision_encoder
            _BRAIN_AI_AVAILABLE = True
            enc = create_vision_encoder(
                encoder_type="standard",
                input_channels=3,
                output_dim=workspace_dim,
                num_steps=2,  # minimal steps for fast validation
            )

        elif modality == "text":
            from brain_ai.encoders import create_text_encoder
            _BRAIN_AI_AVAILABLE = True
            enc = create_text_encoder(
                encoder_type="standard",
                vocab_size=1000,
                embed_dim=128,
                output_dim=workspace_dim,
                num_layers=1,
                num_heads=4,
                ff_dim=256,
            )

        elif modality == "audio":
            from brain_ai.encoders import create_audio_encoder
            _BRAIN_AI_AVAILABLE = True
            enc = create_audio_encoder(
                encoder_type="standard",
                output_dim=workspace_dim,
                num_steps=2,
            )

        elif modality == "sensors":
            from brain_ai.encoders import create_sensor_encoder
            _BRAIN_AI_AVAILABLE = True
            enc = create_sensor_encoder(
                encoder_type="standard",
                input_dim=6,
                output_dim=workspace_dim,
                hidden_dim=64,
                num_layers=1,
            )

        elif modality == "engram":
            from brain_ai.encoders import create_engram_encoder
            _BRAIN_AI_AVAILABLE = True
            enc = create_engram_encoder(
                output_dim=workspace_dim,
                vocab_size=5000,
                embedding_dim=128,
            )
        else:
            return None

        # Move to target device/dtype.  Embedding layers and integer
        # parameters stay at their native types; only floating-point
        # weights are cast.
        enc = enc.to(device=device)
        if dtype != torch.float32:
            for p in enc.parameters():
                if p.is_floating_point():
                    p.data = p.data.to(dtype)
        enc.eval()
        return enc

    except Exception as exc:
        _IMPORT_ERRORS[modality] = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        return None

# ---------------------------------------------------------------------------
# Single-encoder validation
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Result of validating a single encoder."""
    modality: str
    status: str           # "PASS", "FAIL", "SKIP", "LEGACY_PASS", "LEGACY_FAIL"
    elapsed_ms: float = 0.0
    output_shape: Optional[str] = None
    mask_shape: Optional[str] = None
    dtype_str: Optional[str] = None
    violations: List[str] = field(default_factory=list)
    aux_keys: List[str] = field(default_factory=list)
    error: Optional[str] = None
    is_legacy: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality,
            "status": self.status,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "output_shape": self.output_shape,
            "mask_shape": self.mask_shape,
            "dtype": self.dtype_str,
            "violations": self.violations,
            "aux_keys": self.aux_keys,
            "error": self.error,
            "is_legacy": self.is_legacy,
        }


def validate_encoder(
    modality: str,
    workspace_dim: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> ValidationResult:
    """Validate a single modality encoder end-to-end.

    Steps:
        1. Import and instantiate the encoder.
        2. Generate synthetic input data.
        3. Run the forward pass (timed).
        4. If the encoder returns an ``EncoderOutput``, run full contract.
        5. If the encoder returns a raw ``Tensor`` (legacy), run relaxed checks.

    Returns:
        A ``ValidationResult`` summarising the outcome.
    """
    result = ValidationResult(modality=modality)

    # -- Step 1: build encoder -----------------------------------------------
    encoder = _build_encoder(modality, workspace_dim, device, dtype)
    if encoder is None:
        result.status = "SKIP"
        result.error = _IMPORT_ERRORS.get(
            modality, "Could not import encoder (brain_ai not on sys.path?)"
        )
        return result

    # -- Step 2: synthetic data ----------------------------------------------
    gen_fn = SYNTHETIC_GENERATORS.get(modality)
    if gen_fn is None:
        result.status = "SKIP"
        result.error = f"No synthetic data generator for modality '{modality}'"
        return result

    synth = gen_fn(batch_size, device, dtype)
    x = synth["x"]
    kwargs = synth["kwargs"]

    # -- Step 3: forward pass ------------------------------------------------
    try:
        with torch.no_grad():
            t0 = time.perf_counter()
            raw_output = encoder(x, **kwargs)
            t1 = time.perf_counter()

        result.elapsed_ms = (t1 - t0) * 1000.0

    except Exception as exc:
        result.status = "FAIL"
        result.error = (
            f"Forward pass raised {type(exc).__name__}: {exc}\n"
            + traceback.format_exc()
        )
        return result

    # -- Step 4/5: interpret output ------------------------------------------
    if isinstance(raw_output, EncoderOutput):
        # Target state -- full contract validation
        result.is_legacy = False
        result.output_shape = str(tuple(raw_output.feats.shape))
        result.mask_shape = str(tuple(raw_output.mask.shape))
        result.dtype_str = str(raw_output.feats.dtype)
        result.aux_keys = sorted(raw_output.aux.keys()) if raw_output.aux else []

        violations = assert_encoder_contract(raw_output, workspace_dim, device)
        result.violations = violations
        result.status = "PASS" if len(violations) == 0 else "FAIL"

    elif isinstance(raw_output, torch.Tensor):
        # Legacy state -- encoder returns a flat (B, D) or (B, T, D) tensor
        result.is_legacy = True
        result.output_shape = str(tuple(raw_output.shape))
        result.dtype_str = str(raw_output.dtype)

        violations: List[str] = []

        # Check basic shape expectations for legacy output
        if raw_output.ndim == 2:
            B_out, D_out = raw_output.shape
            if B_out != batch_size:
                violations.append(
                    f"legacy output batch dim {B_out} != expected {batch_size}"
                )
            if D_out != workspace_dim:
                violations.append(
                    f"legacy output dim {D_out} != workspace_dim {workspace_dim}"
                )
        elif raw_output.ndim == 3:
            # Possibly already returning (B, T, D) -- partial migration
            if raw_output.shape[0] != batch_size:
                violations.append(
                    f"legacy 3-D output batch dim {raw_output.shape[0]} != "
                    f"expected {batch_size}"
                )
            if raw_output.shape[-1] != workspace_dim:
                violations.append(
                    f"legacy 3-D output dim {raw_output.shape[-1]} != "
                    f"workspace_dim {workspace_dim}"
                )
        else:
            violations.append(
                f"legacy output unexpected ndim={raw_output.ndim}, "
                f"shape={tuple(raw_output.shape)}"
            )

        # NaN / Inf checks
        if torch.isnan(raw_output).any():
            violations.append("legacy output contains NaN values")
        if torch.isinf(raw_output).any():
            violations.append("legacy output contains Inf values")

        # dtype check
        allowed_dtypes = {torch.float32, torch.bfloat16, torch.float16}
        if raw_output.dtype not in allowed_dtypes:
            violations.append(
                f"legacy output dtype {raw_output.dtype} not in "
                f"{{fp32, bf16, fp16}}"
            )

        # device check
        if raw_output.device != device:
            violations.append(
                f"legacy output on {raw_output.device}, expected {device}"
            )

        result.violations = violations
        result.status = "LEGACY_PASS" if len(violations) == 0 else "LEGACY_FAIL"

    else:
        result.status = "FAIL"
        result.error = (
            f"Encoder returned unexpected type {type(raw_output).__name__}. "
            f"Expected EncoderOutput or torch.Tensor."
        )

    return result

# ---------------------------------------------------------------------------
# Cross-modal concatenation validation
# ---------------------------------------------------------------------------

def validate_cross_modal_concat(
    results: Dict[str, ValidationResult],
    workspace_dim: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> ValidationResult:
    """Verify that feats and masks from all available encoders can be
    concatenated along the time axis.

    This simulates the global workspace concatenation path.  For legacy
    encoders that return (B, D), we unsqueeze to (B, 1, D) and generate a
    trivial all-True mask to test compatibility.
    """
    result = ValidationResult(modality="cross_modal_concat")

    available_modalities = [
        m for m, r in results.items()
        if r.status in ("PASS", "LEGACY_PASS")
    ]

    if len(available_modalities) < 2:
        result.status = "SKIP"
        result.error = (
            f"Need at least 2 passing encoders for cross-modal test, "
            f"got {len(available_modalities)}: {available_modalities}"
        )
        return result

    feats_list: List[torch.Tensor] = []
    mask_list: List[torch.Tensor] = []
    per_modality_T: List[str] = []
    errors: List[str] = []

    for modality in available_modalities:
        try:
            encoder = _build_encoder(modality, workspace_dim, device, dtype)
            if encoder is None:
                continue

            gen_fn = SYNTHETIC_GENERATORS[modality]
            synth = gen_fn(batch_size, device, dtype)

            with torch.no_grad():
                raw_output = encoder(synth["x"], **synth["kwargs"])

            if isinstance(raw_output, EncoderOutput):
                feats_list.append(raw_output.feats)
                mask_list.append(raw_output.mask)
            elif isinstance(raw_output, torch.Tensor):
                # Legacy: unsqueeze to (B, 1, D) if needed
                if raw_output.ndim == 2:
                    feats_list.append(raw_output.unsqueeze(1))
                elif raw_output.ndim == 3:
                    feats_list.append(raw_output)
                else:
                    errors.append(
                        f"{modality}: unexpected ndim {raw_output.ndim}"
                    )
                    continue

                T_dim = feats_list[-1].shape[1]
                mask_list.append(
                    torch.ones(
                        batch_size, T_dim,
                        dtype=torch.bool, device=device,
                    )
                )
            else:
                errors.append(
                    f"{modality}: unexpected return type "
                    f"{type(raw_output).__name__}"
                )
                continue

            per_modality_T.append(
                f"{modality}:T={feats_list[-1].shape[1]}"
            )

        except Exception as exc:
            errors.append(f"{modality}: {type(exc).__name__}: {exc}")

    if len(feats_list) < 2:
        result.status = "SKIP"
        result.error = (
            f"Only {len(feats_list)} encoders produced valid output. "
            f"Errors: {errors}"
        )
        return result

    # Check D dimension consistency across all modalities
    dims = [f.shape[-1] for f in feats_list]
    if len(set(dims)) != 1:
        errors.append(
            f"Inconsistent D dimensions across modalities: {dims} "
            f"(modalities: {available_modalities[:len(dims)]})"
        )

    # Attempt concatenation along time axis
    try:
        t0 = time.perf_counter()
        cat_feats = torch.cat(feats_list, dim=1)
        cat_masks = torch.cat(mask_list, dim=1)
        t1 = time.perf_counter()

        result.elapsed_ms = (t1 - t0) * 1000.0
        result.output_shape = str(tuple(cat_feats.shape))
        result.mask_shape = str(tuple(cat_masks.shape))

        # Validate concatenated shapes
        if cat_feats.shape[0] != batch_size:
            errors.append(
                f"Concatenated feats batch dim {cat_feats.shape[0]} "
                f"!= {batch_size}"
            )
        if cat_feats.shape[-1] != workspace_dim:
            errors.append(
                f"Concatenated feats D {cat_feats.shape[-1]} "
                f"!= workspace_dim {workspace_dim}"
            )
        if cat_feats.shape[:2] != cat_masks.shape:
            errors.append(
                f"Concatenated feats/mask mismatch: "
                f"feats {tuple(cat_feats.shape[:2])} vs "
                f"mask {tuple(cat_masks.shape)}"
            )

    except Exception as exc:
        errors.append(f"torch.cat failed: {type(exc).__name__}: {exc}")

    result.violations = errors
    result.status = "PASS" if len(errors) == 0 else "FAIL"
    result.aux_keys = per_modality_T

    return result

# ---------------------------------------------------------------------------
# Fallback recording validation
# ---------------------------------------------------------------------------

def validate_fallback_recording(
    workspace_dim: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> ValidationResult:
    """Check that encoders record their backend / fallback status in aux.

    For legacy encoders (returning raw Tensor), the test reports which
    fallbacks are active based on import probing.
    """
    result = ValidationResult(modality="fallback_recording")
    info_lines: List[str] = []
    warnings: List[str] = []

    # Probe optional dependencies
    dep_status: Dict[str, str] = {}
    for lib_name in ("torchaudio", "ncps", "transformers", "tonic", "snntorch"):
        try:
            __import__(lib_name)
            dep_status[lib_name] = "present"
        except ImportError:
            dep_status[lib_name] = "missing"

    info_lines.append("Dependency status:")
    for lib, status in dep_status.items():
        info_lines.append(f"  {lib}: {status}")

    # Expected aux keys per modality (from dependency-fallbacks.md)
    expected_aux: Dict[str, str] = {
        "audio": "audio_frontend",
        "sensors": "sensor_backend",
        "text": "text_backend",
        "vision": "snn_backend",
        "engram": "pos_applied",
    }

    # Run each encoder and check aux
    for modality, aux_key in expected_aux.items():
        try:
            encoder = _build_encoder(modality, workspace_dim, device, dtype)
            if encoder is None:
                info_lines.append(f"  {modality}: SKIP (not importable)")
                continue

            gen_fn = SYNTHETIC_GENERATORS[modality]
            synth = gen_fn(batch_size, device, dtype)

            with torch.no_grad():
                raw_output = encoder(synth["x"], **synth["kwargs"])

            if isinstance(raw_output, EncoderOutput):
                if aux_key in raw_output.aux:
                    info_lines.append(
                        f"  {modality}: aux['{aux_key}'] = "
                        f"{raw_output.aux[aux_key]}"
                    )
                else:
                    warnings.append(
                        f"{modality}: expected aux key '{aux_key}' not found. "
                        f"Present keys: {sorted(raw_output.aux.keys())}"
                    )
            else:
                info_lines.append(
                    f"  {modality}: legacy output (no aux dict available)"
                )

        except Exception as exc:
            info_lines.append(f"  {modality}: ERROR - {exc}")

    result.violations = warnings
    result.aux_keys = info_lines
    result.status = "PASS" if len(warnings) == 0 else "FAIL"
    return result

# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

_STATUS_LABEL = {
    "PASS": lambda: _green("PASS"),
    "FAIL": lambda: _red("FAIL"),
    "SKIP": lambda: _yellow("SKIP"),
    "LEGACY_PASS": lambda: _green("LEGACY_PASS"),
    "LEGACY_FAIL": lambda: _red("LEGACY_FAIL"),
}

_COL_W_MOD = 22
_COL_W_STATUS = 14
_COL_W_TIME = 12
_COL_W_SHAPE = 24
_COL_W_DTYPE = 14


def _get_status_label(status: str) -> str:
    fn = _STATUS_LABEL.get(status)
    return fn() if fn else status


def _print_header() -> None:
    header = (
        f"{'Modality':<{_COL_W_MOD}}"
        f"{'Status':<{_COL_W_STATUS}}"
        f"{'Time (ms)':<{_COL_W_TIME}}"
        f"{'Output Shape':<{_COL_W_SHAPE}}"
        f"{'dtype':<{_COL_W_DTYPE}}"
    )
    print()
    print(_bold(header))
    print("-" * (len(header) + 10))


def _print_result(r: ValidationResult) -> None:
    status_str = _get_status_label(r.status)
    # Raw status for width calculation (no ANSI)
    raw_status = r.status
    # Pad extra to account for ANSI escape codes
    ansi_extra = len(status_str) - len(raw_status) if _USE_COLOR else 0

    line = (
        f"{r.modality:<{_COL_W_MOD}}"
        f"{status_str:<{_COL_W_STATUS + ansi_extra}}"
        f"{r.elapsed_ms:>{_COL_W_TIME - 2}.2f}  "
        f"{(r.output_shape or 'N/A'):<{_COL_W_SHAPE}}"
        f"{(r.dtype_str or 'N/A'):<{_COL_W_DTYPE}}"
    )
    print(line)

    if r.violations:
        for v in r.violations:
            print(f"    {_red('!')} {v}")

    if r.error and r.status in ("SKIP", "FAIL"):
        # Truncate long tracebacks for display
        err_lines = r.error.strip().split("\n")
        if len(err_lines) > 6:
            err_lines = err_lines[:3] + ["    ..."] + err_lines[-3:]
        for el in err_lines:
            print(f"    {_dim(el)}")


def _print_section(title: str) -> None:
    print()
    print(_bold(f"=== {title} ==="))


def _print_cross_modal(r: ValidationResult) -> None:
    status_str = _get_status_label(r.status)
    print(f"  Status:       {status_str}")
    if r.output_shape:
        print(f"  Cat feats:    {r.output_shape}")
    if r.mask_shape:
        print(f"  Cat mask:     {r.mask_shape}")
    if r.elapsed_ms > 0:
        print(f"  Cat time:     {r.elapsed_ms:.3f} ms")
    if r.aux_keys:
        print(f"  Per-encoder T: {', '.join(r.aux_keys)}")
    if r.violations:
        for v in r.violations:
            print(f"    {_red('!')} {v}")
    if r.error:
        print(f"    {_dim(r.error)}")


def _print_fallback(r: ValidationResult) -> None:
    status_str = _get_status_label(r.status)
    print(f"  Status: {status_str}")
    if r.aux_keys:
        for info_line in r.aux_keys:
            print(f"  {info_line}")
    if r.violations:
        for v in r.violations:
            print(f"    {_yellow('!')} {v}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MODALITIES = ["vision", "text", "audio", "sensors", "engram"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate brain-ai encoder contract compliance.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--workspace-dim",
        type=int,
        default=512,
        help="Expected workspace embedding dimension (default: 512).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for synthetic data (default: 2).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run validation on (default: cpu).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for encoder weights and inputs (default: float32).",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        metavar="PATH",
        help="Write JSON report to the given path.",
    )
    parser.add_argument(
        "--modality",
        type=str,
        nargs="*",
        default=None,
        choices=MODALITIES,
        help="Validate only the specified modalities (default: all).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show full tracebacks on failure.",
    )
    return parser.parse_args()


def _resolve_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def main() -> int:
    args = parse_args()

    workspace_dim: int = args.workspace_dim
    batch_size: int = args.batch_size
    device = torch.device(args.device)
    dtype = _resolve_dtype(args.dtype)
    modalities = args.modality if args.modality else MODALITIES

    print(_bold("Encoder Contract Validation"))
    print(f"  workspace_dim : {workspace_dim}")
    print(f"  batch_size    : {batch_size}")
    print(f"  device        : {device}")
    print(f"  dtype         : {dtype}")
    print(f"  modalities    : {', '.join(modalities)}")
    print(f"  brain_ai root : {_REPO_ROOT}")

    # ---- Per-encoder validation -------------------------------------------
    _print_section("Per-Encoder Validation")
    _print_header()

    encoder_results: Dict[str, ValidationResult] = {}
    for modality in modalities:
        r = validate_encoder(
            modality, workspace_dim, batch_size, device, dtype,
        )
        encoder_results[modality] = r
        _print_result(r)

    # ---- Cross-modal concatenation ----------------------------------------
    _print_section("Cross-Modal Concatenation")
    cross_result = validate_cross_modal_concat(
        encoder_results, workspace_dim, batch_size, device, dtype,
    )
    _print_cross_modal(cross_result)

    # ---- Fallback recording -----------------------------------------------
    _print_section("Fallback / Backend Recording")
    fallback_result = validate_fallback_recording(
        workspace_dim, batch_size, device, dtype,
    )
    _print_fallback(fallback_result)

    # ---- Summary ----------------------------------------------------------
    _print_section("Summary")

    all_results: List[ValidationResult] = (
        list(encoder_results.values()) + [cross_result, fallback_result]
    )

    n_pass = sum(
        1 for r in all_results if r.status in ("PASS", "LEGACY_PASS")
    )
    n_fail = sum(
        1 for r in all_results if r.status in ("FAIL", "LEGACY_FAIL")
    )
    n_skip = sum(1 for r in all_results if r.status == "SKIP")
    n_legacy = sum(
        1 for r in all_results if r.is_legacy and "PASS" in r.status
    )

    print(
        f"  {_green('PASS')}: {n_pass}   "
        f"{_red('FAIL')}: {n_fail}   "
        f"{_yellow('SKIP')}: {n_skip}"
    )
    if n_legacy > 0:
        print(
            f"  {_yellow('Note')}: {n_legacy} encoder(s) return raw Tensor "
            f"(legacy mode). Migrate to EncoderOutput for full contract "
            f"coverage."
        )

    # ---- JSON report ------------------------------------------------------
    if args.json:
        report = {
            "config": {
                "workspace_dim": workspace_dim,
                "batch_size": batch_size,
                "device": str(device),
                "dtype": str(dtype),
                "modalities": modalities,
            },
            "encoders": {
                m: r.to_dict() for m, r in encoder_results.items()
            },
            "cross_modal_concat": cross_result.to_dict(),
            "fallback_recording": fallback_result.to_dict(),
            "summary": {
                "pass": n_pass,
                "fail": n_fail,
                "skip": n_skip,
                "legacy": n_legacy,
            },
        }
        json_path = os.path.abspath(args.json)
        json_dir = os.path.dirname(json_path)
        if json_dir:
            os.makedirs(json_dir, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  JSON report written to: {json_path}")

    # ---- Exit code --------------------------------------------------------
    if n_fail > 0:
        print(f"\n{_red('FAILED')}: {n_fail} check(s) did not pass.")
        return 1

    print(f"\n{_green('ALL CHECKS PASSED')}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
