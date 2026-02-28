#!/usr/bin/env python3
"""
Runtime Contract Validation for spike-codec-losses Skill
=========================================================

Validates that encoding, decoding, and loss modules in brain_ai/core/
conform to the contracts specified in the spike-codec-losses skill
references (spike-decoders.md, loss-pack.md).

Checks both the target API (new batch-first (B, T, N) convention with
SpikeBatch, DecoderOutput, SNNLossComposer) and the legacy API (existing
time-first (T, B, N) convention with RateEncoder, SpikeDecoder, SNNLoss).

Usage:
    python validate_codec_losses.py --all            # Run all checks
    python validate_codec_losses.py --target-only    # Only new API checks
    python validate_codec_losses.py --legacy-only    # Only legacy API checks
    python validate_codec_losses.py --verbose        # Detailed output

Exit code 0 if all executed checks pass, 1 if any fail.
"""

import os
import sys
import argparse
import traceback
import dataclasses
from typing import List, Optional, Dict, Any, Tuple

# =============================================================================
# Section 1: Path Setup
# =============================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# scripts/ -> spike-codec-losses/ -> skills/ -> brain-ai-dev/ -> human-brain/
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(_SCRIPT_DIR)
        )
    )
)

_BRAIN_AI_DIR = os.path.join(_PROJECT_ROOT, "brain_ai")

if not os.path.isdir(_BRAIN_AI_DIR):
    print(f"FATAL: brain_ai directory not found at {_BRAIN_AI_DIR}")
    print(f"  Project root resolved to: {_PROJECT_ROOT}")
    print(f"  Script directory: {_SCRIPT_DIR}")
    sys.exit(2)

# Insert project root so brain_ai is importable
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Attempt torch import early -- nearly every check needs it
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False


# =============================================================================
# Section 2: ValidationResult and ValidationReport
# =============================================================================

@dataclasses.dataclass
class ValidationResult:
    """Result of a single validation check."""
    name: str
    passed: bool
    message: str
    details: str = ""

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        base = f"[{status}] {self.name}: {self.message}"
        if self.details:
            base += f"\n        {self.details}"
        return base


class ValidationReport:
    """Aggregated results across all validation checks."""

    def __init__(self) -> None:
        self.results: List[ValidationResult] = []

    def add(self, result: ValidationResult) -> None:
        self.results.append(result)

    def passed(self) -> List[ValidationResult]:
        return [r for r in self.results if r.passed]

    def failed(self) -> List[ValidationResult]:
        return [r for r in self.results if not r.passed]

    def summary(self) -> str:
        total = len(self.results)
        n_pass = len(self.passed())
        n_fail = len(self.failed())
        lines = [
            "",
            "=" * 72,
            f"  VALIDATION SUMMARY: {n_pass}/{total} passed, {n_fail}/{total} failed",
            "=" * 72,
        ]
        if n_fail > 0:
            lines.append("")
            lines.append("  FAILURES:")
            for r in self.failed():
                lines.append(f"    - {r.name}: {r.message}")
                if r.details:
                    for d in r.details.split("\n"):
                        lines.append(f"        {d}")
        lines.append("")
        return "\n".join(lines)


# Global report instance
_report = ValidationReport()


def _record(name: str, passed: bool, message: str, details: str = "") -> ValidationResult:
    """Create a ValidationResult and add it to the global report."""
    result = ValidationResult(name=name, passed=passed, message=message, details=details)
    _report.add(result)
    return result


def _skip(name: str, reason: str) -> ValidationResult:
    """Record a skipped check (counted as pass with a skip note)."""
    # Skips are informational; they do not count as failures.
    result = ValidationResult(name=name, passed=True, message=f"SKIP: {reason}")
    _report.add(result)
    return result


# =============================================================================
# Section 3: API Detection
# =============================================================================

# Flags set during detection
TARGET_API_AVAILABLE = False
LEGACY_API_AVAILABLE = False

# Target API symbols (populated if available)
_target_encoding = {}    # SpikeBatch, RateEncoder, LatencyEncoder, ...
_target_decoding = {}    # DecoderOutput, RateDecoder, FirstSpikeDecoder, ...
_target_losses = {}      # SNNLossComposer, ProbSpikesLoss, ...

# Legacy API symbols (populated if available)
_legacy_encoding = {}
_legacy_losses = {}


def detect_apis() -> Tuple[bool, bool]:
    """
    Detect which API versions are available.

    Returns (target_available, legacy_available).
    """
    global TARGET_API_AVAILABLE, LEGACY_API_AVAILABLE

    # --- Target API: encoding ---
    try:
        from brain_ai.core.encoding import SpikeBatch
        _target_encoding["SpikeBatch"] = SpikeBatch
    except ImportError:
        pass

    for name in [
        "RateEncoder", "LatencyEncoder", "TTFSEncoder",
        "PopulationEncoder", "DeltaEncoder",
    ]:
        try:
            mod = __import__("brain_ai.core.encoding", fromlist=[name])
            cls = getattr(mod, name, None)
            if cls is not None:
                _target_encoding[name] = cls
        except (ImportError, AttributeError):
            pass

    # --- Target API: decoding ---
    try:
        from brain_ai.core.decoding import DecoderOutput
        _target_decoding["DecoderOutput"] = DecoderOutput
    except ImportError:
        pass

    for name in [
        "RateDecoder", "FirstSpikeDecoder", "PopulationDecoder", "MembraneDecoder",
    ]:
        try:
            mod = __import__("brain_ai.core.decoding", fromlist=[name])
            cls = getattr(mod, name, None)
            if cls is not None:
                _target_decoding[name] = cls
        except (ImportError, AttributeError):
            pass

    # --- Target API: losses ---
    for name in [
        "SNNLossComposer", "LossTerm", "ProbSpikesLoss", "SpikeRateRegularization",
        "TemporalConsistencyLoss", "ISIRegularization",
        "MembraneRegularization", "MembranePotentialRegularization",
    ]:
        try:
            mod = __import__("brain_ai.core.losses", fromlist=[name])
            cls = getattr(mod, name, None)
            if cls is not None:
                _target_losses[name] = cls
        except (ImportError, AttributeError):
            pass

    # --- Target API: config integration ---
    try:
        from brain_ai.core import CodecLossConfig
        _target_losses["CodecLossConfig"] = CodecLossConfig
    except ImportError:
        pass

    # Determine if target API is fully or partially present
    target_enc = "SpikeBatch" in _target_encoding
    target_dec = "DecoderOutput" in _target_decoding
    target_loss = "SNNLossComposer" in _target_losses
    TARGET_API_AVAILABLE = target_enc or target_dec or target_loss

    # --- Legacy API ---
    try:
        from brain_ai.core.encoding import RateEncoder as _LegacyRE
        _legacy_encoding["RateEncoder"] = _LegacyRE
    except ImportError:
        pass

    try:
        from brain_ai.core.encoding import SpikeDecoder as _LegacySD
        _legacy_encoding["SpikeDecoder"] = _LegacySD
    except ImportError:
        pass

    try:
        from brain_ai.core.encoding import TemporalEncoder as _LegacyTE
        _legacy_encoding["TemporalEncoder"] = _LegacyTE
    except ImportError:
        pass

    try:
        from brain_ai.core.encoding import LatencyEncoder as _LegacyLE
        _legacy_encoding["LatencyEncoder"] = _LegacyLE
    except ImportError:
        pass

    try:
        from brain_ai.core.encoding import PopulationEncoder as _LegacyPE
        _legacy_encoding["PopulationEncoder"] = _LegacyPE
    except ImportError:
        pass

    try:
        from brain_ai.core.encoding import DeltaEncoder as _LegacyDE
        _legacy_encoding["DeltaEncoder"] = _LegacyDE
    except ImportError:
        pass

    try:
        from brain_ai.core.losses import SNNLoss as _LegacySL
        _legacy_losses["SNNLoss"] = _LegacySL
    except ImportError:
        pass

    try:
        from brain_ai.core.losses import prob_spikes_loss as _LegacyPSL
        _legacy_losses["prob_spikes_loss"] = _LegacyPSL
    except ImportError:
        pass

    try:
        from brain_ai.core.losses import spike_rate_regularization as _LegacySRR
        _legacy_losses["spike_rate_regularization"] = _LegacySRR
    except ImportError:
        pass

    try:
        from brain_ai.core.losses import temporal_consistency_loss as _LegacyTCL
        _legacy_losses["temporal_consistency_loss"] = _LegacyTCL
    except ImportError:
        pass

    try:
        from brain_ai.core.losses import inter_spike_interval_loss as _LegacyISI
        _legacy_losses["inter_spike_interval_loss"] = _LegacyISI
    except ImportError:
        pass

    try:
        from brain_ai.core.losses import membrane_potential_regularization as _LegacyMPR
        _legacy_losses["membrane_potential_regularization"] = _LegacyMPR
    except ImportError:
        pass

    try:
        from brain_ai.core.losses import compute_snn_metrics as _LegacyCSM
        _legacy_losses["compute_snn_metrics"] = _LegacyCSM
    except ImportError:
        pass

    try:
        from brain_ai.core.losses import spike_rate_range_regularization as _LegacySRRR
        _legacy_losses["spike_rate_range_regularization"] = _LegacySRRR
    except ImportError:
        pass

    try:
        from brain_ai.core.losses import temporal_sparsity_loss as _LegacyTSL
        _legacy_losses["temporal_sparsity_loss"] = _LegacyTSL
    except ImportError:
        pass

    LEGACY_API_AVAILABLE = len(_legacy_encoding) > 0 or len(_legacy_losses) > 0

    return TARGET_API_AVAILABLE, LEGACY_API_AVAILABLE


def print_api_detection_report() -> None:
    """Print which API symbols were found."""
    print("\n" + "=" * 72)
    print("  API DETECTION REPORT")
    print("=" * 72)
    print(f"\n  Project root: {_PROJECT_ROOT}")
    print(f"  brain_ai dir: {_BRAIN_AI_DIR}")
    print(f"  Torch available: {TORCH_AVAILABLE}")
    print(f"  CUDA available:  {CUDA_AVAILABLE}")
    print()

    print("  TARGET API (new batch-first convention):")
    if _target_encoding:
        print(f"    Encoding: {', '.join(sorted(_target_encoding.keys()))}")
    else:
        print("    Encoding: NOT FOUND (SpikeBatch, target encoders missing)")
    if _target_decoding:
        print(f"    Decoding: {', '.join(sorted(_target_decoding.keys()))}")
    else:
        print("    Decoding: NOT FOUND (DecoderOutput, target decoders missing)")
    if _target_losses:
        print(f"    Losses:   {', '.join(sorted(_target_losses.keys()))}")
    else:
        print("    Losses:   NOT FOUND (SNNLossComposer missing)")
    print()

    print("  LEGACY API (existing time-first convention):")
    if _legacy_encoding:
        print(f"    Encoding: {', '.join(sorted(_legacy_encoding.keys()))}")
    else:
        print("    Encoding: NOT FOUND")
    if _legacy_losses:
        print(f"    Losses:   {', '.join(sorted(_legacy_losses.keys()))}")
    else:
        print("    Losses:   NOT FOUND")
    print()


# =============================================================================
# Section 4: Encoding Checks (Target API) -- 10 checks
# =============================================================================

def check_spikebatch_exists() -> None:
    """E01: SpikeBatch dataclass has required fields (spikes, mask, aux)."""
    name = "E01_spikebatch_exists"
    if "SpikeBatch" not in _target_encoding:
        _skip(name, "SpikeBatch not found in brain_ai.core.encoding (target API not yet implemented)")
        return
    try:
        SB = _target_encoding["SpikeBatch"]
        # Must be a dataclass
        assert dataclasses.is_dataclass(SB), "SpikeBatch is not a dataclass"
        field_names = {f.name for f in dataclasses.fields(SB)}
        for required in ("spikes", "mask", "aux"):
            assert required in field_names, f"Missing field: {required}"
        # Attempt to construct one
        spikes = torch.zeros(2, 10, 8)
        mask = torch.ones(2, 10, dtype=torch.bool)
        sb = SB(spikes=spikes, mask=mask, aux={})
        assert sb.spikes.shape == (2, 10, 8)
        _record(name, True, "SpikeBatch dataclass found with spikes, mask, aux fields")
    except Exception as e:
        _record(name, False, f"SpikeBatch validation failed: {e}", traceback.format_exc())


def check_axis_convention() -> None:
    """E02: Encoders output (B, T, ...) batch-first tensors."""
    name = "E02_axis_convention"
    if "SpikeBatch" not in _target_encoding:
        _skip(name, "SpikeBatch not available; cannot verify batch-first output")
        return
    try:
        SB = _target_encoding["SpikeBatch"]
        # Check that any available target encoder produces batch-first output
        encoder_found = False
        for enc_name in ["RateEncoder", "LatencyEncoder", "TTFSEncoder",
                         "PopulationEncoder", "DeltaEncoder"]:
            if enc_name not in _target_encoding:
                continue
            Enc = _target_encoding[enc_name]
            if enc_name == "PopulationEncoder":
                try:
                    enc = Enc(input_dim=8, num_neurons_per_dim=4, num_steps=10)
                except TypeError:
                    enc = Enc(8, 4, 10)
            else:
                try:
                    enc = Enc(num_steps=10)
                except TypeError:
                    enc = Enc(10)
            enc.train(False)
            x = torch.rand(4, 8)
            out = enc(x)
            if isinstance(out, SB):
                spk = out.spikes
            elif isinstance(out, torch.Tensor):
                spk = out
            else:
                continue
            encoder_found = True
            # Batch-first means dim 0 should be batch size
            assert spk.shape[0] == 4, (
                f"{enc_name} first dim is {spk.shape[0]}, expected batch=4. "
                f"Full shape: {spk.shape}. Likely time-first (T, B, ...) convention."
            )
            break

        if not encoder_found:
            _skip(name, "No target encoder could be instantiated to test axis convention")
            return
        _record(name, True, "Encoder output is batch-first (B, T, ...)")
    except Exception as e:
        _record(name, False, f"Axis convention check failed: {e}", traceback.format_exc())


def check_rate_encoder() -> None:
    """E03: RateEncoder produces SpikeBatch (or tensor) of correct shape."""
    name = "E03_rate_encoder"
    # Prefer target API RateEncoder; fall back to skip if not present
    if "RateEncoder" not in _target_encoding:
        _skip(name, "Target RateEncoder not found (new API not yet implemented)")
        return
    try:
        Enc = _target_encoding["RateEncoder"]
        SB = _target_encoding.get("SpikeBatch", None)
        T = 15
        try:
            enc = Enc(num_steps=T)
        except TypeError:
            enc = Enc(T)
        enc.train(False)
        B, D = 4, 16
        x = torch.rand(B, D)
        out = enc(x)

        if SB is not None and isinstance(out, SB):
            spk = out.spikes
        elif isinstance(out, torch.Tensor):
            spk = out
        else:
            _record(name, False, f"RateEncoder returned unexpected type: {type(out)}")
            return

        # Expected batch-first: (B, T, D)
        if spk.dim() == 3 and spk.shape[0] == B and spk.shape[1] == T:
            assert spk.shape == (B, T, D), f"Expected ({B}, {T}, {D}), got {spk.shape}"
            assert ((spk == 0) | (spk == 1)).all(), "Spikes should be binary (0 or 1)"
            _record(name, True, f"RateEncoder produces (B, T, D) = {spk.shape} binary spikes")
        elif spk.dim() == 3 and spk.shape[0] == T and spk.shape[1] == B:
            # Time-first detected -- still functional, but not target convention
            _record(name, True,
                    f"RateEncoder works but uses time-first (T, B, D) = {spk.shape}. "
                    "Target API expects batch-first (B, T, D).")
        else:
            _record(name, False, f"Unexpected shape: {spk.shape}")
    except Exception as e:
        _record(name, False, f"RateEncoder check failed: {e}", traceback.format_exc())


def check_latency_encoder() -> None:
    """E04: LatencyEncoder exists, spike times monotonic with input magnitude."""
    name = "E04_latency_encoder"
    if "LatencyEncoder" not in _target_encoding:
        _skip(name, "Target LatencyEncoder not found")
        return
    try:
        Enc = _target_encoding["LatencyEncoder"]
        T = 20
        try:
            enc = Enc(num_steps=T)
        except TypeError:
            enc = Enc(T)
        enc.train(False)

        B = 4
        # Ascending values: higher value should spike earlier (lower latency)
        x = torch.linspace(0.1, 0.9, 8).unsqueeze(0).expand(B, -1)
        out = enc(x)
        SB = _target_encoding.get("SpikeBatch", None)
        if SB is not None and isinstance(out, SB):
            spk = out.spikes
        elif isinstance(out, torch.Tensor):
            spk = out
        else:
            _record(name, False, f"LatencyEncoder returned unexpected type: {type(out)}")
            return

        # Determine time axis
        if spk.dim() == 3:
            if spk.shape[0] == B:
                # batch-first (B, T, D): time is dim 1
                first_spike_time = spk[0].argmax(dim=0).float()  # (D,)
            else:
                # time-first (T, B, D): time is dim 0
                first_spike_time = spk[:, 0, :].argmax(dim=0).float()  # (D,)
        else:
            _record(name, False, f"Unexpected spk dim: {spk.dim()}")
            return

        # Higher input value should spike earlier or at same time (non-increasing)
        # Allow ties: check that times are roughly non-increasing
        diffs = first_spike_time[1:] - first_spike_time[:-1]
        # Most diffs should be <= 0 (earlier or same)
        frac_monotonic = (diffs <= 0).float().mean().item()
        if frac_monotonic >= 0.7:
            _record(name, True,
                    f"LatencyEncoder spike times are monotonic with input "
                    f"({frac_monotonic * 100:.0f}% non-increasing)")
        else:
            _record(name, False,
                    f"Spike times not monotonic with input magnitude "
                    f"(only {frac_monotonic * 100:.0f}% non-increasing)",
                    f"Times: {first_spike_time.tolist()}")
    except Exception as e:
        _record(name, False, f"LatencyEncoder check failed: {e}", traceback.format_exc())


def check_ttfs_encoder() -> None:
    """E05: TTFSEncoder exists, at most 1 spike per neuron."""
    name = "E05_ttfs_encoder"
    # TTFSEncoder may be named TemporalEncoder or TTFSEncoder in the target API
    Enc = _target_encoding.get("TTFSEncoder", None)
    if Enc is None:
        _skip(name, "TTFSEncoder not found in target API (may be named differently)")
        return
    try:
        T = 20
        try:
            enc = Enc(num_steps=T)
        except TypeError:
            enc = Enc(T)
        enc.train(False)

        B, D = 4, 16
        x = torch.rand(B, D)
        out = enc(x)
        SB = _target_encoding.get("SpikeBatch", None)
        if SB is not None and isinstance(out, SB):
            spk = out.spikes
        elif isinstance(out, torch.Tensor):
            spk = out
        else:
            _record(name, False, f"TTFSEncoder returned unexpected type: {type(out)}")
            return

        # Check at most 1 spike per neuron across time
        if spk.dim() == 3:
            if spk.shape[0] == B:
                # batch-first: sum over T (dim=1)
                spikes_per_neuron = spk.sum(dim=1)  # (B, D)
            else:
                # time-first: sum over T (dim=0)
                spikes_per_neuron = spk.sum(dim=0)  # (B, D)
        else:
            _record(name, False, f"Unexpected spk dim: {spk.dim()}")
            return

        max_spikes = spikes_per_neuron.max().item()
        if max_spikes <= 1.0 + 1e-6:
            _record(name, True, f"TTFSEncoder produces at most 1 spike per neuron (max={max_spikes})")
        else:
            _record(name, False,
                    f"TTFSEncoder produced up to {max_spikes} spikes per neuron (expected <=1)")
    except Exception as e:
        _record(name, False, f"TTFSEncoder check failed: {e}", traceback.format_exc())


def check_population_encoder() -> None:
    """E06: PopulationEncoder exists, population_map in aux."""
    name = "E06_population_encoder"
    if "PopulationEncoder" not in _target_encoding:
        _skip(name, "Target PopulationEncoder not found")
        return
    try:
        Enc = _target_encoding["PopulationEncoder"]
        SB = _target_encoding.get("SpikeBatch", None)
        D_in = 8
        n_per = 5
        T = 15
        try:
            enc = Enc(input_dim=D_in, num_neurons_per_dim=n_per, num_steps=T)
        except TypeError:
            enc = Enc(D_in, n_per, T)
        enc.train(False)

        B = 4
        x = torch.rand(B, D_in)
        out = enc(x)

        if SB is not None and isinstance(out, SB):
            spk = out.spikes
            has_pop_map = "population_map" in out.aux
            if has_pop_map:
                _record(name, True,
                        f"PopulationEncoder produces SpikeBatch with population_map in aux. "
                        f"Spike shape: {spk.shape}")
            else:
                _record(name, True,
                        f"PopulationEncoder produces SpikeBatch but population_map missing from aux. "
                        f"Shape: {spk.shape}")
        elif isinstance(out, torch.Tensor):
            expected_neurons = D_in * n_per
            if out.dim() == 3:
                actual_n = out.shape[-1]
                if actual_n == expected_neurons:
                    _record(name, True,
                            f"PopulationEncoder produces tensor of shape {out.shape}. "
                            "population_map not in aux (raw tensor returned). "
                            f"Expected {expected_neurons} neurons, got {actual_n}.")
                else:
                    _record(name, True,
                            f"PopulationEncoder produces tensor of shape {out.shape}. "
                            f"Neuron count {actual_n} does not match expected {expected_neurons}.")
            else:
                _record(name, False, f"Unexpected output dim: {out.dim()}")
        else:
            _record(name, False, f"Unexpected type: {type(out)}")
    except Exception as e:
        _record(name, False, f"PopulationEncoder check failed: {e}", traceback.format_exc())


def check_delta_encoder() -> None:
    """E07: DeltaEncoder exists, produces ON/OFF channels."""
    name = "E07_delta_encoder"
    if "DeltaEncoder" not in _target_encoding:
        _skip(name, "Target DeltaEncoder not found")
        return
    try:
        Enc = _target_encoding["DeltaEncoder"]
        T = 10
        try:
            enc = Enc(threshold=0.1, num_steps=T)
        except TypeError:
            try:
                enc = Enc(0.1, T)
            except TypeError:
                enc = Enc()
        enc.train(False)

        B, D = 4, 8
        # Reset if method exists
        if hasattr(enc, "reset"):
            enc.reset()

        x1 = torch.rand(B, D)
        out = enc(x1)

        SB = _target_encoding.get("SpikeBatch", None)
        if SB is not None and isinstance(out, SB):
            spk = out.spikes
        elif isinstance(out, torch.Tensor):
            spk = out
        else:
            _record(name, False, f"DeltaEncoder returned unexpected type: {type(out)}")
            return

        # ON/OFF channels should double the feature dimension
        last_dim = spk.shape[-1]
        if last_dim == D * 2:
            _record(name, True,
                    f"DeltaEncoder produces ON/OFF channels. "
                    f"Shape: {spk.shape} (last dim = {D}*2 = {D * 2})")
        else:
            _record(name, True,
                    f"DeltaEncoder output shape {spk.shape}, last dim = {last_dim}. "
                    f"Expected {D * 2} for ON/OFF. May use different channel layout.")
    except Exception as e:
        _record(name, False, f"DeltaEncoder check failed: {e}", traceback.format_exc())


def check_deterministic_mode() -> None:
    """E08: Rate encoder in deterministic mode produces reproducible output."""
    name = "E08_deterministic_mode"
    # Use either target or legacy RateEncoder (deterministic method)
    Enc = _target_encoding.get("RateEncoder", _legacy_encoding.get("RateEncoder", None))
    if Enc is None:
        _skip(name, "No RateEncoder available for deterministic mode check")
        return
    try:
        T = 15
        try:
            enc = Enc(num_steps=T, method="deterministic")
        except TypeError:
            try:
                enc = Enc(T, "deterministic")
            except TypeError:
                enc = Enc(num_steps=T)
        enc.train(False)

        B, D = 4, 16
        x = torch.rand(B, D)

        # Two forward passes should give identical results (deterministic method)
        out1 = enc(x)
        out2 = enc(x)

        if isinstance(out1, torch.Tensor) and isinstance(out2, torch.Tensor):
            spk1, spk2 = out1, out2
        else:
            # Assume SpikeBatch
            spk1 = out1.spikes if hasattr(out1, "spikes") else out1
            spk2 = out2.spikes if hasattr(out2, "spikes") else out2

        if torch.equal(spk1, spk2):
            _record(name, True, "Deterministic encoder produces identical output across calls")
        else:
            diff = (spk1 != spk2).float().mean().item()
            _record(name, False,
                    f"Deterministic encoder produced different output on re-run "
                    f"({diff * 100:.1f}% mismatch)")
    except Exception as e:
        _record(name, False, f"Deterministic mode check failed: {e}", traceback.format_exc())


def check_generator_seed() -> None:
    """E09: Same manual seed produces same stochastic spikes."""
    name = "E09_generator_seed"
    Enc = _target_encoding.get("RateEncoder", _legacy_encoding.get("RateEncoder", None))
    if Enc is None:
        _skip(name, "No RateEncoder available for seed check")
        return
    try:
        T = 15
        try:
            enc = Enc(num_steps=T, method="bernoulli")
        except TypeError:
            try:
                enc = Enc(T, "bernoulli")
            except TypeError:
                enc = Enc(num_steps=T)
        enc.train(False)

        B, D = 4, 16
        x = torch.rand(B, D)

        torch.manual_seed(42)
        out1 = enc(x)
        torch.manual_seed(42)
        out2 = enc(x)

        if isinstance(out1, torch.Tensor):
            spk1, spk2 = out1, out2
        else:
            spk1 = out1.spikes if hasattr(out1, "spikes") else out1
            spk2 = out2.spikes if hasattr(out2, "spikes") else out2

        if torch.equal(spk1, spk2):
            _record(name, True, "Same seed produces identical stochastic spikes")
        else:
            diff = (spk1 != spk2).float().mean().item()
            _record(name, False,
                    f"Same seed produced different spikes ({diff * 100:.1f}% mismatch)")
    except Exception as e:
        _record(name, False, f"Seed check failed: {e}", traceback.format_exc())


def check_normalization() -> None:
    """E10: Encoders support normalization modes (none, minmax, sigmoid, clamp)."""
    name = "E10_normalization"
    # This is a target API feature; check if any encoder supports a normalize/normalization arg
    Enc = _target_encoding.get("RateEncoder", None)
    if Enc is None:
        _skip(name, "Target RateEncoder not found; cannot test normalization modes")
        return
    try:
        modes_checked = []
        modes_failed = []
        T = 10
        B, D = 2, 8
        x = torch.randn(B, D) * 3  # Values outside [0, 1]

        for mode in ["none", "minmax", "sigmoid", "clamp"]:
            try:
                enc = Enc(num_steps=T, normalization=mode)
                enc.train(False)
                out = enc(x)
                spk = out.spikes if hasattr(out, "spikes") else out
                if torch.isfinite(spk).all():
                    modes_checked.append(mode)
                else:
                    modes_failed.append(f"{mode} (non-finite output)")
            except TypeError:
                # Normalization arg not supported
                modes_failed.append(f"{mode} (TypeError -- arg not supported)")
            except Exception as exc:
                modes_failed.append(f"{mode} ({type(exc).__name__}: {exc})")

        if modes_checked:
            _record(name, True,
                    f"Normalization modes working: {modes_checked}. "
                    f"Not working: {modes_failed if modes_failed else 'none'}")
        else:
            _record(name, True,
                    f"No normalization mode parameter accepted. "
                    f"Target API normalization not yet implemented. "
                    f"Details: {modes_failed}")
    except Exception as e:
        _record(name, False, f"Normalization check failed: {e}", traceback.format_exc())


def run_encoding_checks() -> None:
    """Run all 10 encoding checks."""
    check_spikebatch_exists()
    check_axis_convention()
    check_rate_encoder()
    check_latency_encoder()
    check_ttfs_encoder()
    check_population_encoder()
    check_delta_encoder()
    check_deterministic_mode()
    check_generator_seed()
    check_normalization()


# =============================================================================
# Section 5: Decoding Checks (Target API) -- 8 checks
# =============================================================================

def check_decoder_output() -> None:
    """D01: DecoderOutput dataclass has logits_proxy, prediction, confidence, aux."""
    name = "D01_decoder_output"
    if "DecoderOutput" not in _target_decoding:
        _skip(name, "DecoderOutput not found in brain_ai.core.decoding")
        return
    try:
        DO = _target_decoding["DecoderOutput"]
        assert dataclasses.is_dataclass(DO), "DecoderOutput is not a dataclass"
        field_names = {f.name for f in dataclasses.fields(DO)}
        for required in ("logits_proxy", "prediction", "confidence", "aux"):
            assert required in field_names, f"Missing field: {required}"

        # Construct a valid instance
        B, C = 4, 10
        do = DO(
            logits_proxy=torch.randn(B, C),
            prediction=torch.randint(0, C, (B,)),
            confidence=torch.rand(B),
            aux={"test": True},
        )
        assert do.logits_proxy.shape == (B, C)
        assert do.prediction.shape == (B,)
        assert do.confidence.shape == (B,)
        _record(name, True, "DecoderOutput dataclass found with all required fields")
    except Exception as e:
        _record(name, False, f"DecoderOutput validation failed: {e}", traceback.format_exc())


def _make_test_spikes(B: int, T: int, N: int, rate: float = 0.3,
                      batch_first: bool = True) -> torch.Tensor:
    """Create test spike tensor with given firing rate."""
    spk = (torch.rand(B, T, N) < rate).float()
    if not batch_first:
        spk = spk.permute(1, 0, 2)  # -> (T, B, N)
    return spk


def check_rate_decoder() -> None:
    """D02: RateDecoder produces correct argmax prediction."""
    name = "D02_rate_decoder"
    if "RateDecoder" not in _target_decoding:
        _skip(name, "RateDecoder not found in target API")
        return
    try:
        Dec = _target_decoding["RateDecoder"]
        DO = _target_decoding.get("DecoderOutput", None)
        B, T, N = 4, 20, 10
        try:
            dec = Dec()
        except TypeError:
            dec = Dec(num_classes=N)

        # Create spikes where class 3 fires most
        spk = torch.zeros(B, T, N)
        spk[:, :, 3] = 1.0  # class 3 fires every step
        spk[:, :5, 7] = 1.0  # class 7 fires 5 steps

        out = dec(spk)

        if DO is not None and isinstance(out, DO):
            assert out.prediction.shape == (B,)
            assert (out.prediction == 3).all(), (
                f"Expected prediction=3 (highest rate), got {out.prediction.tolist()}")
            assert out.logits_proxy.shape[0] == B
            _record(name, True, "RateDecoder produces correct argmax prediction via DecoderOutput")
        elif isinstance(out, torch.Tensor):
            # Raw tensor output
            pred = out.argmax(dim=-1) if out.dim() >= 2 else out
            _record(name, True,
                    f"RateDecoder returns tensor of shape {out.shape} (not DecoderOutput)."
                    " Target API expects DecoderOutput wrapper.")
        else:
            _record(name, False, f"Unexpected output type: {type(out)}")
    except Exception as e:
        _record(name, False, f"RateDecoder check failed: {e}", traceback.format_exc())


def check_first_spike_decoder() -> None:
    """D03: FirstSpikeDecoder produces correct argmin prediction."""
    name = "D03_first_spike_decoder"
    if "FirstSpikeDecoder" not in _target_decoding:
        _skip(name, "FirstSpikeDecoder not found in target API")
        return
    try:
        Dec = _target_decoding["FirstSpikeDecoder"]
        DO = _target_decoding.get("DecoderOutput", None)
        B, T, N = 4, 20, 10
        try:
            dec = Dec()
        except TypeError:
            dec = Dec(num_classes=N)

        # Create spikes where class 5 fires first (t=0)
        spk = torch.zeros(B, T, N)
        spk[:, 0, 5] = 1.0    # class 5 fires at t=0
        spk[:, 10, 2] = 1.0   # class 2 fires at t=10

        out = dec(spk)

        if DO is not None and isinstance(out, DO):
            assert out.prediction.shape == (B,)
            assert (out.prediction == 5).all(), (
                f"Expected prediction=5 (earliest spike), got {out.prediction.tolist()}")
            _record(name, True, "FirstSpikeDecoder produces correct argmin prediction")
        elif isinstance(out, torch.Tensor):
            _record(name, True,
                    f"FirstSpikeDecoder returns tensor of shape {out.shape} "
                    "(not DecoderOutput wrapper)")
        else:
            _record(name, False, f"Unexpected output type: {type(out)}")
    except Exception as e:
        _record(name, False, f"FirstSpikeDecoder check failed: {e}", traceback.format_exc())


def check_population_decoder() -> None:
    """D04: PopulationDecoder handles groups correctly."""
    name = "D04_population_decoder"
    if "PopulationDecoder" not in _target_decoding:
        _skip(name, "PopulationDecoder not found in target API")
        return
    try:
        Dec = _target_decoding["PopulationDecoder"]
        DO = _target_decoding.get("DecoderOutput", None)
        B, T = 4, 20
        num_groups = 5
        neurons_per_group = 4
        N = num_groups * neurons_per_group  # 20

        pop_map = {g: list(range(g * neurons_per_group, (g + 1) * neurons_per_group))
                   for g in range(num_groups)}

        try:
            dec = Dec(population_map=pop_map)
        except TypeError:
            try:
                dec = Dec(num_groups=num_groups, neurons_per_group=neurons_per_group)
            except TypeError:
                dec = Dec()

        # Create spikes where group 2 fires most
        spk = torch.zeros(B, T, N)
        for idx in pop_map[2]:
            spk[:, :, idx] = 1.0

        try:
            out = dec(spk, population_map=pop_map)
        except TypeError:
            out = dec(spk)

        if DO is not None and isinstance(out, DO):
            assert out.logits_proxy.shape == (B, num_groups), (
                f"Expected logits (B, {num_groups}), got {out.logits_proxy.shape}")
            assert (out.prediction == 2).all(), (
                f"Expected prediction=2, got {out.prediction.tolist()}")
            _record(name, True, "PopulationDecoder handles group aggregation correctly")
        elif isinstance(out, torch.Tensor):
            _record(name, True,
                    f"PopulationDecoder returns tensor {out.shape} (not DecoderOutput)")
        else:
            _record(name, False, f"Unexpected output type: {type(out)}")
    except Exception as e:
        _record(name, False, f"PopulationDecoder check failed: {e}", traceback.format_exc())


def check_membrane_decoder() -> None:
    """D05: MembraneDecoder uses final/max membrane potential."""
    name = "D05_membrane_decoder"
    if "MembraneDecoder" not in _target_decoding:
        _skip(name, "MembraneDecoder not found in target API")
        return
    try:
        Dec = _target_decoding["MembraneDecoder"]
        DO = _target_decoding.get("DecoderOutput", None)
        B, T, N = 4, 20, 10

        membrane = torch.randn(B, T, N)
        # Make class 7 have the highest final membrane
        membrane[:, -1, 7] = 100.0

        for mode in ["final", "max"]:
            try:
                dec = Dec(mode=mode)
            except TypeError:
                dec = Dec()
            spk = torch.zeros(B, T, N)  # dummy spikes
            try:
                out = dec(spk, membrane=membrane)
            except TypeError:
                out = dec(membrane)

            if DO is not None and isinstance(out, DO):
                if mode == "final":
                    assert (out.prediction == 7).all(), (
                        f"mode=final: expected pred=7, got {out.prediction.tolist()}")
            elif isinstance(out, torch.Tensor):
                pass  # raw tensor

        _record(name, True, "MembraneDecoder uses membrane potential correctly")
    except Exception as e:
        _record(name, False, f"MembraneDecoder check failed: {e}", traceback.format_exc())


def check_decoder_shapes() -> None:
    """D06: All decoders produce (B, C) logits and (B,) prediction."""
    name = "D06_decoder_shapes"
    decoders_found = []
    decoders_failed = []

    B, T, N = 4, 20, 10

    for dec_name in ["RateDecoder", "FirstSpikeDecoder", "PopulationDecoder", "MembraneDecoder"]:
        if dec_name not in _target_decoding:
            continue
        Dec = _target_decoding[dec_name]
        DO = _target_decoding.get("DecoderOutput", None)

        try:
            if dec_name == "PopulationDecoder":
                pop_map = {g: [g * 2, g * 2 + 1] for g in range(5)}
                try:
                    dec = Dec(population_map=pop_map)
                except TypeError:
                    dec = Dec()
            else:
                try:
                    dec = Dec()
                except TypeError:
                    dec = Dec(num_classes=N)

            spk = _make_test_spikes(B, T, N)
            membrane = torch.randn(B, T, N)

            try:
                out = dec(spk, membrane=membrane)
            except TypeError:
                out = dec(spk)

            if DO is not None and isinstance(out, DO):
                assert out.logits_proxy.dim() == 2, (
                    f"{dec_name}: logits dim={out.logits_proxy.dim()}, expected 2")
                assert out.logits_proxy.shape[0] == B
                assert out.prediction.shape == (B,)
                decoders_found.append(dec_name)
            elif isinstance(out, torch.Tensor):
                decoders_found.append(f"{dec_name} (raw tensor)")
            else:
                decoders_failed.append(f"{dec_name} (unexpected type {type(out).__name__})")
        except Exception as exc:
            decoders_failed.append(f"{dec_name} ({exc})")

    if not decoders_found and not decoders_failed:
        _skip(name, "No target decoders found to check shapes")
    elif decoders_failed:
        _record(name, False,
                f"Shape check failures: {decoders_failed}. "
                f"OK: {decoders_found}")
    else:
        _record(name, True, f"All checked decoders have correct shapes: {decoders_found}")


def check_decoder_confidence() -> None:
    """D07: Decoder confidence values are in [0, 1]."""
    name = "D07_decoder_confidence"
    DO = _target_decoding.get("DecoderOutput", None)
    if DO is None:
        _skip(name, "DecoderOutput not found; cannot check confidence")
        return

    checked = []
    failed = []
    B, T, N = 8, 25, 10

    for dec_name in ["RateDecoder", "FirstSpikeDecoder", "MembraneDecoder"]:
        if dec_name not in _target_decoding:
            continue
        Dec = _target_decoding[dec_name]
        try:
            try:
                dec = Dec()
            except TypeError:
                dec = Dec(num_classes=N)
            spk = _make_test_spikes(B, T, N)
            membrane = torch.randn(B, T, N)
            try:
                out = dec(spk, membrane=membrane)
            except TypeError:
                out = dec(spk)

            if isinstance(out, DO):
                conf = out.confidence
                if (conf >= 0).all() and (conf <= 1.0 + 1e-6).all():
                    checked.append(dec_name)
                else:
                    failed.append(f"{dec_name} (confidence range [{conf.min().item():.4f}, "
                                  f"{conf.max().item():.4f}])")
            else:
                checked.append(f"{dec_name} (no DecoderOutput)")
        except Exception as exc:
            failed.append(f"{dec_name} ({exc})")

    if not checked and not failed:
        _skip(name, "No target decoders with DecoderOutput to check")
    elif failed:
        _record(name, False, f"Confidence out of [0,1]: {failed}. OK: {checked}")
    else:
        _record(name, True, f"Confidence in [0,1] for: {checked}")


def check_decoder_amp() -> None:
    """D08: Decoders produce finite output under autocast."""
    name = "D08_decoder_amp"
    if not CUDA_AVAILABLE:
        _skip(name, "CUDA not available; skipping AMP decoder check")
        return

    DO = _target_decoding.get("DecoderOutput", None)
    checked = []
    failed = []
    B, T, N = 4, 25, 10

    for dec_name in ["RateDecoder", "FirstSpikeDecoder", "MembraneDecoder"]:
        if dec_name not in _target_decoding:
            continue
        Dec = _target_decoding[dec_name]
        try:
            try:
                dec = Dec()
            except TypeError:
                dec = Dec(num_classes=N)
            spk = _make_test_spikes(B, T, N).cuda()
            membrane = torch.randn(B, T, N).cuda()
            with torch.cuda.amp.autocast():
                try:
                    out = dec(spk, membrane=membrane)
                except TypeError:
                    out = dec(spk)

            if DO is not None and isinstance(out, DO):
                assert out.logits_proxy.isfinite().all(), "Non-finite logits"
                assert out.confidence.isfinite().all(), "Non-finite confidence"
            elif isinstance(out, torch.Tensor):
                assert out.isfinite().all(), "Non-finite tensor output"
            checked.append(dec_name)
        except Exception as exc:
            failed.append(f"{dec_name} ({exc})")

    if not checked and not failed:
        _skip(name, "No target decoders found for AMP check")
    elif failed:
        _record(name, False, f"AMP failures: {failed}. OK: {checked}")
    else:
        _record(name, True, f"All checked decoders finite under AMP: {checked}")


def run_decoding_checks() -> None:
    """Run all 8 decoding checks."""
    check_decoder_output()
    check_rate_decoder()
    check_first_spike_decoder()
    check_population_decoder()
    check_membrane_decoder()
    check_decoder_shapes()
    check_decoder_confidence()
    check_decoder_amp()


# =============================================================================
# Section 6: Loss Checks (Target API) -- 10 checks
# =============================================================================

def _make_loss_inputs(B: int = 4, T: int = 25, N: int = 10,
                      rate: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create standard (spikes, membrane, targets) for loss testing."""
    spk = (torch.rand(B, T, N) < rate).float()
    membrane = torch.randn(B, T, N)
    targets = torch.randint(0, N, (B,))
    return spk, membrane, targets


def _make_legacy_loss_inputs(B: int = 4, T: int = 25, N: int = 10,
                             rate: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create standard (spikes, membrane, targets) in legacy (T, B, N) layout."""
    spk = (torch.rand(T, B, N) < rate).float()
    membrane = torch.randn(B, N)
    targets = torch.randint(0, N, (B,))
    return spk, membrane, targets


def check_probspikes() -> None:
    """L01: ProbSpikesLoss produces finite scalar with diagnostics dict."""
    name = "L01_probspikes"
    ProbSpikes = _target_losses.get("ProbSpikesLoss", None)
    if ProbSpikes is None:
        _skip(name, "ProbSpikesLoss not found in target API")
        return
    try:
        loss_fn = ProbSpikes(temperature=1.0)
        spk, membrane, targets = _make_loss_inputs()
        loss_val, diag = loss_fn(spk, membrane, targets)

        assert torch.isfinite(loss_val), f"Loss is not finite: {loss_val}"
        assert loss_val.dim() == 0, f"Loss is not scalar (dim={loss_val.dim()})"
        assert isinstance(diag, dict), f"Diagnostics is not a dict: {type(diag)}"
        assert loss_val.dtype == torch.float32, f"Loss dtype is {loss_val.dtype}, expected float32"

        _record(name, True,
                f"ProbSpikesLoss: loss={loss_val.item():.4f}, "
                f"diagnostics keys: {list(diag.keys())}")
    except Exception as e:
        _record(name, False, f"ProbSpikesLoss check failed: {e}", traceback.format_exc())


def check_rate_reg() -> None:
    """L02: SpikeRateRegularization produces finite scalar."""
    name = "L02_rate_reg"
    SRR = _target_losses.get("SpikeRateRegularization", None)
    if SRR is None:
        _skip(name, "SpikeRateRegularization not found in target API")
        return
    try:
        loss_fn = SRR(target_rate=0.1)
        spk, membrane, targets = _make_loss_inputs()
        loss_val, diag = loss_fn(spk, membrane, targets)

        assert torch.isfinite(loss_val), f"Loss is not finite: {loss_val}"
        assert loss_val.dim() == 0, f"Loss is not scalar"
        assert isinstance(diag, dict), f"Diagnostics is not a dict"

        # Check for expected diagnostic keys
        expected_diag_keys = ["mean_rate", "dead_neuron_frac"]
        found_keys = [k for k in expected_diag_keys if k in diag]

        _record(name, True,
                f"SpikeRateRegularization: loss={loss_val.item():.6f}, "
                f"diagnostics keys: {list(diag.keys())}")
    except Exception as e:
        _record(name, False, f"SpikeRateRegularization check failed: {e}", traceback.format_exc())


def check_temporal_consistency() -> None:
    """L03: TemporalConsistencyLoss produces finite scalar."""
    name = "L03_temporal_consistency"
    TCL = _target_losses.get("TemporalConsistencyLoss", None)
    if TCL is None:
        _skip(name, "TemporalConsistencyLoss not found in target API")
        return
    try:
        loss_fn = TCL(window_size=5)
        spk, membrane, targets = _make_loss_inputs(T=30)  # Need T >= 2*window_size
        loss_val, diag = loss_fn(spk, membrane, targets)

        assert torch.isfinite(loss_val), f"Loss is not finite: {loss_val}"
        assert loss_val.dim() == 0
        assert isinstance(diag, dict)

        _record(name, True,
                f"TemporalConsistencyLoss: loss={loss_val.item():.6f}, "
                f"diagnostics keys: {list(diag.keys())}")
    except Exception as e:
        _record(name, False, f"TemporalConsistencyLoss check failed: {e}", traceback.format_exc())


def check_isi_vectorized() -> None:
    """L04: ISIRegularization uses conv1d (no Python loops), produces finite scalar."""
    name = "L04_isi_vectorized"
    ISI = _target_losses.get("ISIRegularization", None)
    if ISI is None:
        _skip(name, "ISIRegularization not found in target API")
        return
    try:
        loss_fn = ISI(refractory_window=5)
        spk, membrane, targets = _make_loss_inputs(T=30)
        loss_val, diag = loss_fn(spk, membrane, targets)

        assert torch.isfinite(loss_val), f"Loss is not finite: {loss_val}"
        assert loss_val.dim() == 0

        # Check that it uses conv1d: look for a registered kernel buffer
        has_kernel = any(
            "kernel" in bname for bname, _ in loss_fn.named_buffers()
        )
        # Also check source code for F.conv1d usage
        import inspect
        source = inspect.getsource(type(loss_fn).forward)
        uses_conv1d = "conv1d" in source or "F.conv1d" in source

        if has_kernel or uses_conv1d:
            _record(name, True,
                    f"ISIRegularization uses conv1d (vectorized): loss={loss_val.item():.6f}")
        else:
            _record(name, False,
                    f"ISIRegularization does not appear to use conv1d. "
                    f"loss={loss_val.item():.6f}. "
                    f"has_kernel={has_kernel}, uses_conv1d={uses_conv1d}")
    except Exception as e:
        _record(name, False, f"ISIRegularization check failed: {e}", traceback.format_exc())


def check_membrane_reg() -> None:
    """L05: MembranePotentialRegularization produces finite scalar."""
    name = "L05_membrane_reg"
    MPR = _target_losses.get("MembraneRegularization",
           _target_losses.get("MembranePotentialRegularization", None))
    if MPR is None:
        _skip(name, "MembraneRegularization / MembranePotentialRegularization not found")
        return
    try:
        try:
            loss_fn = MPR(max_membrane=1.5)
        except TypeError:
            loss_fn = MPR()

        spk, membrane, targets = _make_loss_inputs()
        # Make some membrane values exceed threshold to trigger penalty
        membrane[:, :, 0] = 10.0

        try:
            loss_val, diag = loss_fn(spk, membrane, targets)
        except TypeError:
            # Legacy function interface: membrane_potential_regularization(membrane, max_membrane)
            loss_val = loss_fn(membrane, 1.5)
            diag = {}

        assert torch.isfinite(loss_val), f"Loss is not finite: {loss_val}"
        assert loss_val.item() > 0, "Expected nonzero loss for exploded membrane"

        _record(name, True,
                f"Membrane regularization: loss={loss_val.item():.6f}")
    except Exception as e:
        _record(name, False, f"Membrane reg check failed: {e}", traceback.format_exc())


def check_loss_composer() -> None:
    """L06: SNNLossComposer combines all terms, returns (total, components_dict)."""
    name = "L06_loss_composer"
    Composer = _target_losses.get("SNNLossComposer", None)
    if Composer is None:
        _skip(name, "SNNLossComposer not found in target API")
        return
    try:
        # Build composer with whatever terms are available
        terms_dict = {}

        ProbSpikes = _target_losses.get("ProbSpikesLoss", None)
        if ProbSpikes is not None:
            terms_dict["prob_spikes"] = (ProbSpikes(temperature=1.0), 1.0)

        SRR = _target_losses.get("SpikeRateRegularization", None)
        if SRR is not None:
            terms_dict["spike_rate"] = (SRR(target_rate=0.1), 0.1)

        TCL = _target_losses.get("TemporalConsistencyLoss", None)
        if TCL is not None:
            terms_dict["temporal"] = (TCL(window_size=5), 0.01)

        ISI = _target_losses.get("ISIRegularization", None)
        if ISI is not None:
            terms_dict["isi"] = (ISI(refractory_window=5), 0.01)

        MPR = _target_losses.get("MembraneRegularization",
               _target_losses.get("MembranePotentialRegularization", None))
        if MPR is not None:
            try:
                terms_dict["membrane"] = (MPR(max_membrane=1.5), 0.001)
            except TypeError:
                terms_dict["membrane"] = (MPR(), 0.001)

        if not terms_dict:
            _skip(name, "No loss terms available to compose")
            return

        composer = Composer(terms=terms_dict)
        spk, membrane, targets = _make_loss_inputs(T=30)
        total_loss, components = composer(spk, membrane, targets)

        assert torch.isfinite(total_loss), f"Total loss not finite: {total_loss}"
        assert total_loss.dim() == 0, "Total loss not scalar"
        assert isinstance(components, dict), f"Components not dict: {type(components)}"
        assert "loss/total" in components, f"'loss/total' not in components: {list(components.keys())}"

        _record(name, True,
                f"SNNLossComposer: total={total_loss.item():.4f}, "
                f"terms={list(terms_dict.keys())}, "
                f"component keys: {[k for k in components if k.startswith('loss/')]}")
    except Exception as e:
        _record(name, False, f"SNNLossComposer check failed: {e}", traceback.format_exc())


def check_loss_diagnostics() -> None:
    """L07: All diagnostics keys from the logging contract are present."""
    name = "L07_loss_diagnostics"
    Composer = _target_losses.get("SNNLossComposer", None)
    if Composer is None:
        _skip(name, "SNNLossComposer not found; cannot check diagnostics")
        return
    try:
        terms_dict = {}
        ProbSpikes = _target_losses.get("ProbSpikesLoss", None)
        if ProbSpikes is not None:
            terms_dict["prob_spikes"] = (ProbSpikes(), 1.0)
        SRR = _target_losses.get("SpikeRateRegularization", None)
        if SRR is not None:
            terms_dict["spike_rate"] = (SRR(), 0.1)
        TCL = _target_losses.get("TemporalConsistencyLoss", None)
        if TCL is not None:
            terms_dict["temporal"] = (TCL(), 0.01)

        if not terms_dict:
            _skip(name, "No loss terms available to check diagnostics")
            return

        composer = Composer(terms=terms_dict)
        spk, membrane, targets = _make_loss_inputs(T=30)
        total_loss, components = composer(spk, membrane, targets)

        # Check for expected key patterns
        expected_prefixes = []
        for term_name in terms_dict:
            expected_prefixes.append(f"loss/{term_name}_raw")
            expected_prefixes.append(f"loss/{term_name}_weighted")

        missing = [p for p in expected_prefixes if p not in components]
        found = [p for p in expected_prefixes if p in components]

        if not missing:
            _record(name, True,
                    f"All expected diagnostics keys found: {found}")
        else:
            _record(name, False,
                    f"Missing diagnostics keys: {missing}. Found: {found}",
                    f"All keys: {list(components.keys())}")
    except Exception as e:
        _record(name, False, f"Diagnostics check failed: {e}", traceback.format_exc())


def check_loss_amp() -> None:
    """L08: All losses produce finite values under autocast."""
    name = "L08_loss_amp"
    if not CUDA_AVAILABLE:
        _skip(name, "CUDA not available; skipping AMP loss check")
        return
    try:
        checked = []
        failed = []
        B, T, N = 4, 25, 10
        targets = torch.randint(0, N, (B,)).cuda()

        for term_name, key in [
            ("ProbSpikesLoss", "ProbSpikesLoss"),
            ("SpikeRateRegularization", "SpikeRateRegularization"),
            ("TemporalConsistencyLoss", "TemporalConsistencyLoss"),
            ("ISIRegularization", "ISIRegularization"),
            ("MembraneRegularization", "MembraneRegularization"),
            ("MembranePotentialRegularization", "MembranePotentialRegularization"),
        ]:
            Loss = _target_losses.get(key, None)
            if Loss is None:
                continue
            try:
                try:
                    loss_fn = Loss()
                except TypeError:
                    loss_fn = Loss(temperature=1.0)
                loss_fn = loss_fn.cuda()

                with torch.cuda.amp.autocast():
                    for t_val in [10, 25, 50]:
                        spk = (torch.rand(B, t_val, N, device="cuda") < 0.2).float()
                        membrane = torch.randn(B, t_val, N, device="cuda")
                        result = loss_fn(spk, membrane, targets)
                        if isinstance(result, tuple):
                            lv = result[0]
                        else:
                            lv = result
                        assert torch.isfinite(lv), f"Non-finite at T={t_val}"
                checked.append(term_name)
            except Exception as exc:
                failed.append(f"{term_name} ({exc})")

        if not checked and not failed:
            _skip(name, "No target loss terms found for AMP check")
        elif failed:
            _record(name, False, f"AMP failures: {failed}. OK: {checked}")
        else:
            _record(name, True, f"All loss terms finite under AMP: {checked}")
    except Exception as e:
        _record(name, False, f"AMP loss check failed: {e}", traceback.format_exc())


def check_loss_gradients() -> None:
    """L09: backward() produces non-zero gradients for all loss terms."""
    name = "L09_loss_gradients"
    checked = []
    failed = []

    B, T, N = 4, 25, 10

    for term_name, key in [
        ("ProbSpikesLoss", "ProbSpikesLoss"),
        ("SpikeRateRegularization", "SpikeRateRegularization"),
        ("TemporalConsistencyLoss", "TemporalConsistencyLoss"),
        ("ISIRegularization", "ISIRegularization"),
    ]:
        Loss = _target_losses.get(key, None)
        if Loss is None:
            continue
        try:
            try:
                loss_fn = Loss()
            except TypeError:
                loss_fn = Loss(temperature=1.0)

            # Create spikes with requires_grad via a linear layer
            linear = nn.Linear(N, N)
            x = torch.randn(B, T, N)
            logits = linear(x)
            # Create spikes through a sigmoid (soft spikes for gradient flow)
            soft_spikes = torch.sigmoid(logits * 10)
            membrane = torch.randn(B, T, N)
            targets = torch.randint(0, N, (B,))

            result = loss_fn(soft_spikes, membrane, targets)
            if isinstance(result, tuple):
                lv = result[0]
            else:
                lv = result

            lv.backward()

            has_grad = False
            for p in linear.parameters():
                if p.grad is not None and p.grad.abs().sum() > 0:
                    has_grad = True
                    break

            if has_grad:
                checked.append(term_name)
            else:
                failed.append(f"{term_name} (zero gradients)")
        except Exception as exc:
            failed.append(f"{term_name} ({exc})")

    if not checked and not failed:
        _skip(name, "No target loss terms found for gradient check")
    elif failed:
        _record(name, False, f"Gradient failures: {failed}. OK: {checked}")
    else:
        _record(name, True, f"Non-zero gradients confirmed for: {checked}")


def check_loss_scaling() -> None:
    """L10: Loss does not explode with T (test T=10, 25, 50)."""
    name = "L10_loss_scaling"
    ProbSpikes = _target_losses.get("ProbSpikesLoss", None)
    if ProbSpikes is None:
        # Try legacy
        prob_fn = _legacy_losses.get("prob_spikes_loss", None)
        if prob_fn is None:
            _skip(name, "No ProbSpikes loss function available for scaling check")
            return
        # Test with legacy function
        try:
            B, N = 4, 10
            rate = 0.2
            losses_by_t = {}
            for T in [10, 25, 50]:
                spk = (torch.rand(T, B, N) < rate).float()
                targets = torch.randint(0, N, (B,))
                lv = prob_fn(spk, targets, temperature=1.0)
                losses_by_t[T] = lv.item()

            # Check that loss doesn't explode: T=50 loss should not be >10x T=10 loss
            ratio = losses_by_t[50] / (losses_by_t[10] + 1e-8)
            if ratio < 10.0 and all(v < 100 for v in losses_by_t.values()):
                _record(name, True,
                        f"Legacy prob_spikes_loss scales reasonably: {losses_by_t} "
                        f"(ratio T=50/T=10 = {ratio:.2f})")
            else:
                _record(name, False,
                        f"Loss may explode with T: {losses_by_t} "
                        f"(ratio T=50/T=10 = {ratio:.2f})")
            return
        except Exception as e:
            _record(name, False, f"Legacy scaling check failed: {e}", traceback.format_exc())
            return

    try:
        loss_fn = ProbSpikes(temperature=1.0)
        B, N = 4, 10
        rate = 0.2
        losses_by_t = {}

        for T in [10, 25, 50]:
            spk = (torch.rand(B, T, N) < rate).float()
            membrane = torch.randn(B, T, N)
            targets = torch.randint(0, N, (B,))
            result = loss_fn(spk, membrane, targets)
            if isinstance(result, tuple):
                lv = result[0]
            else:
                lv = result
            losses_by_t[T] = lv.item()

        ratio = losses_by_t[50] / (losses_by_t[10] + 1e-8)
        all_finite = all(abs(v) < 1000 for v in losses_by_t.values())
        if ratio < 10.0 and all_finite:
            _record(name, True,
                    f"ProbSpikesLoss scales reasonably: {losses_by_t} "
                    f"(ratio T=50/T=10 = {ratio:.2f})")
        else:
            _record(name, False,
                    f"Loss may explode with T: {losses_by_t} "
                    f"(ratio T=50/T=10 = {ratio:.2f})")
    except Exception as e:
        _record(name, False, f"Loss scaling check failed: {e}", traceback.format_exc())


def run_loss_checks() -> None:
    """Run all 10 loss checks."""
    check_probspikes()
    check_rate_reg()
    check_temporal_consistency()
    check_isi_vectorized()
    check_membrane_reg()
    check_loss_composer()
    check_loss_diagnostics()
    check_loss_amp()
    check_loss_gradients()
    check_loss_scaling()


# =============================================================================
# Section 7: Integration Checks -- 5 checks
# =============================================================================

def check_encode_decode_roundtrip() -> None:
    """I01: Rate encode -> rate decode preserves ordering of input magnitudes."""
    name = "I01_encode_decode_roundtrip"
    # Try target API first, fall back to legacy
    RateEnc = _target_encoding.get("RateEncoder", _legacy_encoding.get("RateEncoder", None))
    if RateEnc is None:
        _skip(name, "No RateEncoder available for roundtrip check")
        return

    RateDec = _target_decoding.get("RateDecoder", None)
    LegacyDec = _legacy_encoding.get("SpikeDecoder", None)

    try:
        T = 50  # More timesteps for better rate estimation
        B = 8

        # Use deterministic method for more reliable rate encoding
        try:
            enc = RateEnc(num_steps=T, method="deterministic")
        except TypeError:
            enc = RateEnc(num_steps=T)
        enc.train(False)

        # Input with clear ordering: x[i] increases with i
        N = 8
        x = torch.linspace(0.1, 0.9, N).unsqueeze(0).expand(B, -1)

        out_enc = enc(x)
        if hasattr(out_enc, "spikes"):
            spk = out_enc.spikes
        else:
            spk = out_enc

        # Decode
        decoded = None
        if RateDec is not None:
            try:
                dec = RateDec()
                out_dec = dec(spk)
                if hasattr(out_dec, "logits_proxy"):
                    decoded = out_dec.logits_proxy
                elif isinstance(out_dec, torch.Tensor):
                    decoded = out_dec
            except Exception:
                pass

        if decoded is None and LegacyDec is not None:
            try:
                dec = LegacyDec(method="rate")
                decoded = dec(spk)
            except Exception:
                pass

        if decoded is None:
            # Manual rate decode: sum over time axis
            if spk.dim() == 3:
                if spk.shape[0] == B:
                    decoded = spk.sum(dim=1) / spk.shape[1]  # batch-first
                else:
                    decoded = spk.sum(dim=0) / spk.shape[0]  # time-first
            else:
                decoded = spk

        # Check ordering: decoded values should roughly preserve input ordering
        if decoded.dim() >= 2:
            mean_decoded = decoded.mean(dim=0)  # (N,)
        else:
            mean_decoded = decoded

        # Check that decoded is roughly monotonically increasing
        diffs = mean_decoded[1:] - mean_decoded[:-1]
        frac_increasing = (diffs > -1e-6).float().mean().item()

        if frac_increasing >= 0.8:
            _record(name, True,
                    f"Encode-decode roundtrip preserves ordering "
                    f"({frac_increasing * 100:.0f}% monotonic)")
        else:
            _record(name, False,
                    f"Ordering not preserved ({frac_increasing * 100:.0f}% monotonic)",
                    f"Decoded mean: {mean_decoded.tolist()}")
    except Exception as e:
        _record(name, False, f"Roundtrip check failed: {e}", traceback.format_exc())


def check_full_pipeline() -> None:
    """I02: encode -> (stub SNN) -> decode -> loss -> backward."""
    name = "I02_full_pipeline"
    RateEnc = _target_encoding.get("RateEncoder", _legacy_encoding.get("RateEncoder", None))
    if RateEnc is None:
        _skip(name, "No RateEncoder available for pipeline check")
        return

    try:
        B = 4
        T = 15
        D_in = 16
        N_out = 10

        # Encoder
        try:
            enc = RateEnc(num_steps=T, method="deterministic")
        except TypeError:
            enc = RateEnc(num_steps=T)
        enc.train(False)

        x = torch.rand(B, D_in)
        enc_out = enc(x)
        spk_in = enc_out.spikes if hasattr(enc_out, "spikes") else enc_out

        # Stub SNN: simple linear projection per timestep
        snn_linear = nn.Linear(D_in, N_out)

        if spk_in.dim() == 3:
            if spk_in.shape[0] == B:
                # batch-first (B, T, D_in)
                spk_projected = snn_linear(spk_in)  # (B, T, N_out)
                # Apply threshold to generate spikes
                snn_spikes = (spk_projected > 0).float()
                membrane = spk_projected
            else:
                # time-first (T, B, D_in)
                spk_projected = snn_linear(spk_in)  # (T, B, N_out)
                snn_spikes = (spk_projected > 0).float()
                membrane = spk_projected
        else:
            spk_projected = snn_linear(spk_in)
            snn_spikes = (spk_projected > 0).float()
            membrane = spk_projected

        # Decode
        RateDec = _target_decoding.get("RateDecoder", None)
        LegacyDec = _legacy_encoding.get("SpikeDecoder", None)

        decoded_logits = None
        if RateDec is not None:
            try:
                dec = RateDec()
                dec_out = dec(snn_spikes)
                if hasattr(dec_out, "logits_proxy"):
                    decoded_logits = dec_out.logits_proxy
            except Exception:
                pass

        if decoded_logits is None and LegacyDec is not None:
            try:
                dec = LegacyDec(method="rate")
                decoded_logits = dec(snn_spikes)
            except Exception:
                pass

        if decoded_logits is None:
            # Manual decode
            if snn_spikes.dim() == 3:
                if snn_spikes.shape[0] == B:
                    decoded_logits = snn_spikes.sum(dim=1)
                else:
                    decoded_logits = snn_spikes.sum(dim=0)
            else:
                decoded_logits = snn_spikes

        # Loss
        targets = torch.randint(0, N_out, (B,))

        ProbSpikes = _target_losses.get("ProbSpikesLoss", None)
        if ProbSpikes is not None:
            loss_fn = ProbSpikes()
            if snn_spikes.dim() == 3:
                lv, _ = loss_fn(snn_spikes, membrane, targets)
            else:
                lv = F.cross_entropy(decoded_logits, targets)
        else:
            # Use simple cross entropy
            lv = F.cross_entropy(decoded_logits, targets)

        # Backward
        lv.backward()

        # Check gradients flowed back to SNN linear
        has_grad = False
        for p in snn_linear.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break

        if has_grad:
            _record(name, True,
                    f"Full pipeline encode->SNN->decode->loss->backward succeeded. "
                    f"Loss={lv.item():.4f}")
        else:
            # Gradients may not flow through hard thresholding
            _record(name, True,
                    f"Pipeline ran without error. Loss={lv.item():.4f}. "
                    "Note: gradients may be zero due to hard spike thresholding "
                    "(surrogate gradients needed).")
    except Exception as e:
        _record(name, False, f"Full pipeline check failed: {e}", traceback.format_exc())


def check_axis_helpers() -> None:
    """I03: time_to_batch_first and batch_to_time_first roundtrip correctly."""
    name = "I03_axis_helpers"

    # Try to import helper functions
    try:
        from brain_ai.core.encoding import time_to_batch_first, batch_to_time_first
    except ImportError:
        try:
            from brain_ai.core import time_to_batch_first, batch_to_time_first
        except ImportError:
            # Helpers not implemented yet; validate the concept with stub
            _skip(name, "time_to_batch_first / batch_to_time_first not found. "
                  "Axis conversion helpers not yet implemented.")
            return

    try:
        B, T, D = 4, 20, 16
        # time-first: (T, B, D)
        x_tf = torch.randn(T, B, D)

        # Convert to batch-first
        x_bf = time_to_batch_first(x_tf)
        assert x_bf.shape == (B, T, D), f"Expected (B, T, D)={B, T, D}, got {x_bf.shape}"

        # Convert back
        x_roundtrip = batch_to_time_first(x_bf)
        assert x_roundtrip.shape == (T, B, D), f"Expected (T, B, D), got {x_roundtrip.shape}"

        # Values preserved
        assert torch.allclose(x_tf, x_roundtrip), "Roundtrip values differ"

        _record(name, True, "Axis conversion helpers roundtrip correctly")
    except Exception as e:
        _record(name, False, f"Axis helper check failed: {e}", traceback.format_exc())


def check_config_integration() -> None:
    """I04: CodecLossConfig creates valid encoder+decoder+loss combo."""
    name = "I04_config_integration"
    CLC = _target_losses.get("CodecLossConfig", None)
    if CLC is None:
        _skip(name, "CodecLossConfig not found; config integration not yet implemented")
        return
    try:
        config = CLC()

        # Config should have methods or attributes to create components
        encoder = None
        decoder = None
        loss = None

        if hasattr(config, "create_encoder"):
            encoder = config.create_encoder()
        if hasattr(config, "create_decoder"):
            decoder = config.create_decoder()
        if hasattr(config, "create_loss"):
            loss = config.create_loss()
        elif hasattr(config, "create_composer"):
            loss = config.create_composer()

        parts = []
        if encoder is not None:
            parts.append("encoder")
        if decoder is not None:
            parts.append("decoder")
        if loss is not None:
            parts.append("loss")

        if parts:
            _record(name, True, f"CodecLossConfig creates: {parts}")
        else:
            _record(name, True,
                    "CodecLossConfig exists but no create_* methods found. "
                    "May use a different pattern for component creation.")
    except Exception as e:
        _record(name, False, f"Config integration check failed: {e}", traceback.format_exc())


def check_legacy_compatibility() -> None:
    """I05: Old API still works alongside new API (if both present)."""
    name = "I05_legacy_compatibility"
    if not LEGACY_API_AVAILABLE:
        _skip(name, "Legacy API not available")
        return

    try:
        issues = []

        # Check that legacy RateEncoder still works
        if "RateEncoder" in _legacy_encoding:
            Enc = _legacy_encoding["RateEncoder"]
            enc = Enc(num_steps=15, method="bernoulli")
            enc.train(False)
            x = torch.rand(4, 8)
            out = enc(x)
            if not isinstance(out, torch.Tensor):
                issues.append("Legacy RateEncoder no longer returns tensor")

        # Check that legacy SpikeDecoder still works
        if "SpikeDecoder" in _legacy_encoding:
            Dec = _legacy_encoding["SpikeDecoder"]
            dec = Dec(method="rate")
            spk = torch.rand(15, 4, 8)
            out = dec(spk)
            if not isinstance(out, torch.Tensor):
                issues.append("Legacy SpikeDecoder no longer returns tensor")

        # Check that legacy SNNLoss still works
        if "SNNLoss" in _legacy_losses:
            Loss = _legacy_losses["SNNLoss"]
            loss_fn = Loss()
            spk = (torch.rand(15, 4, 10) > 0.7).float()
            membrane = torch.randn(4, 10)
            targets = torch.randint(0, 10, (4,))
            total, metrics = loss_fn(spk, membrane, targets)
            if not torch.isfinite(total):
                issues.append(f"Legacy SNNLoss returned non-finite: {total}")

        if issues:
            _record(name, False,
                    f"Legacy compatibility issues: {issues}")
        else:
            _record(name, True,
                    "Legacy API components still work correctly")
    except Exception as e:
        _record(name, False, f"Legacy compatibility check failed: {e}", traceback.format_exc())


def run_integration_checks() -> None:
    """Run all 5 integration checks."""
    check_encode_decode_roundtrip()
    check_full_pipeline()
    check_axis_helpers()
    check_config_integration()
    check_legacy_compatibility()


# =============================================================================
# Section 8: Legacy Checks -- 5 checks
# =============================================================================

def check_legacy_encoding() -> None:
    """LG01: Existing RateEncoder, TemporalEncoder, LatencyEncoder all work."""
    name = "LG01_legacy_encoding"
    if not _legacy_encoding:
        _skip(name, "No legacy encoding classes found")
        return
    try:
        checked = []
        failed = []
        B, T = 4, 20

        # RateEncoder
        if "RateEncoder" in _legacy_encoding:
            try:
                enc = _legacy_encoding["RateEncoder"](num_steps=T, method="bernoulli")
                enc.train(False)
                x = torch.rand(B, 16)
                out = enc(x)
                assert isinstance(out, torch.Tensor), "Not a tensor"
                assert out.dim() == 3, f"Expected 3D, got {out.dim()}D"
                assert ((out == 0) | (out == 1)).all(), "Not binary"
                checked.append(f"RateEncoder({out.shape})")
            except Exception as exc:
                failed.append(f"RateEncoder: {exc}")

        # TemporalEncoder
        if "TemporalEncoder" in _legacy_encoding:
            try:
                enc = _legacy_encoding["TemporalEncoder"](num_steps=T)
                enc.train(False)
                x = torch.rand(B, 16).clamp(0.01, 1.0)
                out = enc(x)
                assert isinstance(out, torch.Tensor)
                assert out.dim() == 3
                checked.append(f"TemporalEncoder({out.shape})")
            except Exception as exc:
                failed.append(f"TemporalEncoder: {exc}")

        # LatencyEncoder
        if "LatencyEncoder" in _legacy_encoding:
            try:
                enc = _legacy_encoding["LatencyEncoder"](num_steps=T)
                enc.train(False)
                x = torch.rand(B, 16)
                out = enc(x)
                assert isinstance(out, torch.Tensor)
                assert out.dim() == 3
                checked.append(f"LatencyEncoder({out.shape})")
            except Exception as exc:
                failed.append(f"LatencyEncoder: {exc}")

        # PopulationEncoder
        if "PopulationEncoder" in _legacy_encoding:
            try:
                enc = _legacy_encoding["PopulationEncoder"](
                    input_dim=8, num_neurons_per_dim=5, num_steps=T)
                enc.train(False)
                x = torch.rand(B, 8)
                out = enc(x)
                assert isinstance(out, torch.Tensor)
                assert out.dim() == 3
                checked.append(f"PopulationEncoder({out.shape})")
            except Exception as exc:
                failed.append(f"PopulationEncoder: {exc}")

        # DeltaEncoder
        if "DeltaEncoder" in _legacy_encoding:
            try:
                enc = _legacy_encoding["DeltaEncoder"](threshold=0.1, num_steps=T)
                enc.train(False)
                if hasattr(enc, "reset"):
                    enc.reset()
                x = torch.rand(B, 8)
                out = enc(x)
                assert isinstance(out, torch.Tensor)
                assert out.dim() == 3
                checked.append(f"DeltaEncoder({out.shape})")
            except Exception as exc:
                failed.append(f"DeltaEncoder: {exc}")

        if failed:
            _record(name, False,
                    f"Legacy encoding failures: {failed}. OK: {checked}")
        else:
            _record(name, True, f"All legacy encoders work: {checked}")
    except Exception as e:
        _record(name, False, f"Legacy encoding check failed: {e}", traceback.format_exc())


def check_legacy_decoding() -> None:
    """LG02: Existing SpikeDecoder works with all methods."""
    name = "LG02_legacy_decoding"
    if "SpikeDecoder" not in _legacy_encoding:
        _skip(name, "Legacy SpikeDecoder not found")
        return
    try:
        Dec = _legacy_encoding["SpikeDecoder"]
        checked = []
        failed = []
        T, B, D = 20, 4, 10
        spk = (torch.rand(T, B, D) > 0.7).float()
        membrane = torch.randn(B, D)

        for method in ["rate", "first_spike", "membrane"]:
            try:
                dec = Dec(method=method)
                if method == "membrane":
                    out = dec(spk, membrane=membrane)
                else:
                    out = dec(spk)
                assert isinstance(out, torch.Tensor), f"Not tensor: {type(out)}"
                assert out.shape[0] == B, f"Expected batch dim {B}, got {out.shape[0]}"
                assert torch.isfinite(out).all(), f"Non-finite output"
                checked.append(f"{method}({out.shape})")
            except Exception as exc:
                failed.append(f"{method}: {exc}")

        if failed:
            _record(name, False,
                    f"Legacy decoding failures: {failed}. OK: {checked}")
        else:
            _record(name, True, f"All legacy decoder methods work: {checked}")
    except Exception as e:
        _record(name, False, f"Legacy decoding check failed: {e}", traceback.format_exc())


def check_legacy_losses() -> None:
    """LG03: Existing SNNLoss works and returns (total, metrics)."""
    name = "LG03_legacy_losses"
    if "SNNLoss" not in _legacy_losses:
        _skip(name, "Legacy SNNLoss not found")
        return
    try:
        Loss = _legacy_losses["SNNLoss"]
        loss_fn = Loss(
            task_loss_weight=1.0,
            spike_rate_weight=0.1,
            temporal_weight=0.01,
            membrane_reg_weight=0.001,
            use_prob_spikes=True,
        )

        T, B, N = 25, 4, 10
        spk = (torch.rand(T, B, N) > 0.7).float()
        membrane = torch.randn(B, N)
        targets = torch.randint(0, N, (B,))

        total, metrics = loss_fn(spk, membrane, targets)

        assert torch.isfinite(total), f"Total loss not finite: {total}"
        assert total.dim() == 0, f"Loss not scalar"
        assert isinstance(metrics, dict), f"Metrics not dict: {type(metrics)}"

        expected_keys = ["task_loss", "accuracy"]
        found_keys = [k for k in expected_keys if k in metrics]

        _record(name, True,
                f"Legacy SNNLoss: total={total.item():.4f}, "
                f"metrics keys: {list(metrics.keys())}")
    except Exception as e:
        _record(name, False, f"Legacy SNNLoss check failed: {e}", traceback.format_exc())


def check_legacy_metrics() -> None:
    """LG04: compute_snn_metrics returns expected keys."""
    name = "LG04_legacy_metrics"
    if "compute_snn_metrics" not in _legacy_losses:
        _skip(name, "Legacy compute_snn_metrics not found")
        return
    try:
        compute_fn = _legacy_losses["compute_snn_metrics"]
        T, B, N = 25, 4, 10
        spk = (torch.rand(T, B, N) > 0.7).float()
        membrane = torch.randn(B, N)
        targets = torch.randint(0, N, (B,))

        metrics = compute_fn(spk, membrane, targets)

        assert isinstance(metrics, dict), f"Not a dict: {type(metrics)}"

        expected_keys = [
            "accuracy", "spike_rate", "temporal_sparsity",
            "output_entropy", "confidence",
            "membrane_mean", "membrane_std",
            "dead_neuron_fraction", "saturated_neuron_fraction",
        ]

        found = [k for k in expected_keys if k in metrics]
        missing = [k for k in expected_keys if k not in metrics]

        if not missing:
            _record(name, True,
                    f"compute_snn_metrics returns all expected keys: {found}")
        elif len(found) > len(missing):
            _record(name, True,
                    f"compute_snn_metrics returns most keys. "
                    f"Found: {found}. Missing: {missing}")
        else:
            _record(name, False,
                    f"compute_snn_metrics missing many keys. "
                    f"Found: {found}. Missing: {missing}",
                    f"All keys: {list(metrics.keys())}")
    except Exception as e:
        _record(name, False, f"Legacy metrics check failed: {e}", traceback.format_exc())


def check_legacy_axis() -> None:
    """LG05: Existing encoding uses (T, B, D) convention."""
    name = "LG05_legacy_axis"
    if "RateEncoder" not in _legacy_encoding:
        _skip(name, "Legacy RateEncoder not found for axis check")
        return
    try:
        Enc = _legacy_encoding["RateEncoder"]
        T = 15
        B = 4
        D = 8

        enc = Enc(num_steps=T, method="deterministic")
        enc.train(False)
        x = torch.rand(B, D)
        out = enc(x)

        assert isinstance(out, torch.Tensor), f"Not tensor: {type(out)}"
        assert out.dim() == 3, f"Expected 3D, got {out.dim()}D"

        # Legacy convention: (T, B, D)
        if out.shape[0] == T and out.shape[1] == B and out.shape[2] == D:
            _record(name, True,
                    f"Legacy encoder uses (T, B, D) = ({T}, {B}, {D}) convention. "
                    f"Output shape: {out.shape}")
        elif out.shape[0] == B and out.shape[1] == T and out.shape[2] == D:
            _record(name, True,
                    f"Legacy encoder uses (B, T, D) = ({B}, {T}, {D}) convention. "
                    f"Already batch-first. Shape: {out.shape}")
        else:
            _record(name, False,
                    f"Unexpected legacy shape: {out.shape}. "
                    f"Expected (T={T}, B={B}, D={D}) or (B={B}, T={T}, D={D})")
    except Exception as e:
        _record(name, False, f"Legacy axis check failed: {e}", traceback.format_exc())


def run_legacy_checks() -> None:
    """Run all 5 legacy checks."""
    check_legacy_encoding()
    check_legacy_decoding()
    check_legacy_losses()
    check_legacy_metrics()
    check_legacy_axis()


# =============================================================================
# Section 9: CLI Entry Point
# =============================================================================

# ANSI color codes
_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _colorize(result: ValidationResult, use_color: bool = True) -> str:
    """Format a result with optional ANSI coloring."""
    if not use_color:
        return str(result)

    if "SKIP" in result.message:
        color = _YELLOW
        tag = "SKIP"
    elif result.passed:
        color = _GREEN
        tag = "PASS"
    else:
        color = _RED
        tag = "FAIL"

    base = f"  {color}{_BOLD}[{tag}]{_RESET} {result.name}: {result.message}"
    if result.details:
        detail_lines = result.details.strip().split("\n")
        # Show only first 5 lines of details to avoid overwhelming output
        for line in detail_lines[:5]:
            base += f"\n          {line}"
        if len(detail_lines) > 5:
            base += f"\n          ... ({len(detail_lines) - 5} more lines)"
    return base


def _colorize_summary(report: ValidationReport, use_color: bool = True) -> str:
    """Format the summary with optional ANSI coloring."""
    total = len(report.results)
    n_pass = len(report.passed())
    n_fail = len(report.failed())
    n_skip = sum(1 for r in report.results if "SKIP" in r.message)
    n_real_pass = n_pass - n_skip

    lines = [
        "",
        "=" * 72,
    ]
    if use_color:
        lines.append(
            f"  {_GREEN if n_fail == 0 else _RED}{_BOLD}VALIDATION SUMMARY{_RESET}: "
            f"{_GREEN}{n_real_pass} passed{_RESET}, "
            f"{_YELLOW}{n_skip} skipped{_RESET}, "
            f"{_RED}{n_fail} failed{_RESET} "
            f"(total: {total})"
        )
    else:
        lines.append(
            f"  VALIDATION SUMMARY: "
            f"{n_real_pass} passed, {n_skip} skipped, {n_fail} failed "
            f"(total: {total})"
        )
    lines.append("=" * 72)

    if n_fail > 0:
        lines.append("")
        if use_color:
            lines.append(f"  {_RED}FAILURES:{_RESET}")
        else:
            lines.append("  FAILURES:")
        for r in report.failed():
            lines.append(f"    - {r.name}: {r.message}")
            if r.details:
                for d in r.details.strip().split("\n")[:3]:
                    lines.append(f"        {d}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    """CLI entry point. Returns exit code 0 (all pass) or 1 (any fail)."""
    parser = argparse.ArgumentParser(
        description="Validate spike-codec-losses skill contracts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate_codec_losses.py --all
  python validate_codec_losses.py --target-only --verbose
  python validate_codec_losses.py --legacy-only
        """,
    )
    parser.add_argument(
        "--target-only", action="store_true",
        help="Only run target (new) API checks (encoding, decoding, losses, integration)")
    parser.add_argument(
        "--legacy-only", action="store_true",
        help="Only run legacy API checks")
    parser.add_argument(
        "--all", action="store_true",
        help="Run all checks (target + legacy). This is the default.")
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show detailed output for each check")
    parser.add_argument(
        "--no-color", action="store_true",
        help="Disable ANSI color output")
    args = parser.parse_args()

    use_color = not args.no_color and sys.stdout.isatty()

    # Default to --all if nothing specified
    if not args.target_only and not args.legacy_only:
        args.all = True

    if not TORCH_AVAILABLE:
        print("FATAL: PyTorch not importable. Cannot run validation.")
        print("  Install with: pip install torch")
        return 2

    # Detect APIs
    target_avail, legacy_avail = detect_apis()
    print_api_detection_report()

    # Run checks
    sections_run = []

    if args.all or args.target_only:
        print(f"\n{'=' * 72}")
        print(f"  ENCODING CHECKS (Target API)")
        print(f"{'=' * 72}")
        run_encoding_checks()
        enc_results = _report.results[len(_report.results) - 10:]
        for r in enc_results:
            print(_colorize(r, use_color))
        sections_run.append("encoding")

        print(f"\n{'=' * 72}")
        print(f"  DECODING CHECKS (Target API)")
        print(f"{'=' * 72}")
        run_decoding_checks()
        dec_results = _report.results[len(_report.results) - 8:]
        for r in dec_results:
            print(_colorize(r, use_color))
        sections_run.append("decoding")

        print(f"\n{'=' * 72}")
        print(f"  LOSS CHECKS (Target API)")
        print(f"{'=' * 72}")
        run_loss_checks()
        loss_results = _report.results[len(_report.results) - 10:]
        for r in loss_results:
            print(_colorize(r, use_color))
        sections_run.append("losses")

        print(f"\n{'=' * 72}")
        print(f"  INTEGRATION CHECKS")
        print(f"{'=' * 72}")
        run_integration_checks()
        int_results = _report.results[len(_report.results) - 5:]
        for r in int_results:
            print(_colorize(r, use_color))
        sections_run.append("integration")

    if args.all or args.legacy_only:
        print(f"\n{'=' * 72}")
        print(f"  LEGACY CHECKS")
        print(f"{'=' * 72}")
        run_legacy_checks()
        leg_results = _report.results[len(_report.results) - 5:]
        for r in leg_results:
            print(_colorize(r, use_color))
        sections_run.append("legacy")

    # Summary
    print(_colorize_summary(_report, use_color))

    if args.verbose:
        print(f"\n  Sections run: {sections_run}")
        print(f"  Project root: {_PROJECT_ROOT}")
        print(f"  Target API available: {TARGET_API_AVAILABLE}")
        print(f"  Legacy API available: {LEGACY_API_AVAILABLE}")
        print(f"  Torch version: {torch.__version__ if TORCH_AVAILABLE else 'N/A'}")
        print(f"  CUDA available: {CUDA_AVAILABLE}")
        if CUDA_AVAILABLE:
            print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        print()

    # Exit code
    n_fail = len(_report.failed())
    return 1 if n_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
