#!/usr/bin/env python3
"""
encoder_deps_report.py -- Audit encoder-specific optional dependencies.

Checks every optional package used by the brain_ai encoder suite, reports
which are installed (with version), which are missing, and documents the
active fallback path for each.  Optionally tries to instantiate each
encoder class to verify that the fallback path actually works.

Usage:
    python encoder_deps_report.py
    python encoder_deps_report.py --json
    python encoder_deps_report.py --check-instantiate
    python encoder_deps_report.py --verbose --check-instantiate
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import traceback
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version as pkg_version
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# sys.path manipulation so brain_ai can be imported when running from the
# encoder-suite/scripts/ directory.  Mirrors the pattern used by the
# system-orchestrator scripts.
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
        )
    )
)
if os.path.isdir(os.path.join(_REPO_ROOT, "brain_ai")):
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Dependency registry
# ---------------------------------------------------------------------------

@dataclass
class EncoderDep:
    """Descriptor for a single encoder-related optional dependency."""

    name: str               # pip / display name
    import_name: str        # Python module to import
    used_by: str            # which encoder(s) consume it
    fallback: str           # human-readable fallback description
    aux_key: str            # the EncoderOutput.aux dict key that records the backend
    min_version: Optional[str] = None  # optional minimum version constraint


ENCODER_DEPS: List[EncoderDep] = [
    EncoderDep(
        name="torchaudio",
        import_name="torchaudio",
        used_by="AudioEncoder mel frontend",
        fallback="torch.stft + manual mel filterbank",
        aux_key="audio_frontend",
    ),
    EncoderDep(
        name="ncps",
        import_name="ncps",
        used_by="SensorEncoder CfC/LTC",
        fallback="Built-in CfC/LTC classes (sensors.py)",
        aux_key="sensor_backend",
    ),
    EncoderDep(
        name="transformers",
        import_name="transformers",
        used_by="TextEncoder (optional pretrained)",
        fallback="Built-in nn.TransformerEncoder",
        aux_key="text_backend",
    ),
    EncoderDep(
        name="tonic",
        import_name="tonic",
        used_by="EventVisionEncoder",
        fallback="Simple event binning or disabled",
        aux_key="event_frontend",
    ),
    EncoderDep(
        name="snntorch",
        import_name="snntorch",
        used_by="Spiking layers (optional)",
        fallback="Built-in LIFNeuron (core/neurons.py)",
        aux_key="snn_backend",
    ),
    EncoderDep(
        name="torchvision",
        import_name="torchvision",
        used_by="VisionEncoder (optional pretrained backbones)",
        fallback="Built-in spiking CNN",
        aux_key="vision_backbone",
    ),
    EncoderDep(
        name="librosa",
        import_name="librosa",
        used_by="AudioEncoder (optional MFCC features)",
        fallback="torchaudio or manual mel",
        aux_key="audio_features",
    ),
]


# ---------------------------------------------------------------------------
# Single-dependency checker
# ---------------------------------------------------------------------------

@dataclass
class DepStatus:
    """Result of checking one dependency."""

    name: str
    status: str             # "available", "missing", "outdated"
    version: Optional[str]
    fallback: str
    used_by: str
    aux_key: str
    import_path: Optional[str] = None  # __file__ of the imported module
    error: Optional[str] = None
    min_version: Optional[str] = None


def _parse_version_tuple(ver: str):
    """Convert a dotted version string to a comparable tuple of ints."""
    parts: List[int] = []
    for segment in ver.split("."):
        # Strip non-numeric suffixes (e.g. "2.1.0a3" -> 2,1,0)
        numeric = ""
        for ch in segment:
            if ch.isdigit():
                numeric += ch
            else:
                break
        parts.append(int(numeric) if numeric else 0)
    return tuple(parts)


def check_dependency(dep: EncoderDep) -> DepStatus:
    """Try to import *dep* and return its status."""
    try:
        mod = __import__(dep.import_name)
    except ImportError as exc:
        return DepStatus(
            name=dep.name,
            status="missing",
            version=None,
            fallback=dep.fallback,
            used_by=dep.used_by,
            aux_key=dep.aux_key,
            error=str(exc),
            min_version=dep.min_version,
        )

    # Determine installed version
    version: Optional[str] = None
    try:
        version = pkg_version(dep.name)
    except PackageNotFoundError:
        version = getattr(mod, "__version__", "unknown")

    # Determine import file path (useful for --verbose)
    import_path = getattr(mod, "__file__", None)

    # Check minimum version if specified
    status = "available"
    if dep.min_version and version and version != "unknown":
        try:
            if _parse_version_tuple(version) < _parse_version_tuple(dep.min_version):
                status = "outdated"
        except (ValueError, TypeError):
            pass  # version parsing failed — treat as available

    return DepStatus(
        name=dep.name,
        status=status,
        version=version,
        fallback=dep.fallback,
        used_by=dep.used_by,
        aux_key=dep.aux_key,
        import_path=import_path,
        min_version=dep.min_version,
    )


# ---------------------------------------------------------------------------
# Batch checker
# ---------------------------------------------------------------------------

def check_all_dependencies() -> List[DepStatus]:
    """Check every entry in ENCODER_DEPS and return a list of status dicts."""
    return [check_dependency(dep) for dep in ENCODER_DEPS]


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------

@dataclass
class SystemInfo:
    python_version: str
    platform_str: str
    torch_version: Optional[str] = None
    cuda_available: bool = False
    cuda_version: Optional[str] = None
    device_count: int = 0
    gpu_names: List[str] = field(default_factory=list)


def gather_system_info() -> SystemInfo:
    info = SystemInfo(
        python_version=platform.python_version(),
        platform_str=platform.platform(),
    )
    try:
        import torch
        info.torch_version = torch.__version__
        info.cuda_available = torch.cuda.is_available()
        if info.cuda_available:
            info.cuda_version = torch.version.cuda
            info.device_count = torch.cuda.device_count()
            info.gpu_names = [
                torch.cuda.get_device_name(i)
                for i in range(info.device_count)
            ]
    except ImportError:
        pass
    return info


# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------

_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _supports_color() -> bool:
    """Heuristic: emit ANSI only when stdout looks like a terminal."""
    if os.environ.get("NO_COLOR"):
        return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


_USE_COLOR = _supports_color()


def _c(code: str, text: str) -> str:
    if _USE_COLOR:
        return f"{code}{text}{_RESET}"
    return text


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

SEP = "-" * 88


def _print_system_info(info: SystemInfo) -> None:
    print(SEP)
    print(f"  {_c(_BOLD, 'System Environment')}")
    print(SEP)
    rows = [
        ("Python version", info.python_version),
        ("Platform", info.platform_str),
        ("PyTorch version", info.torch_version or "NOT INSTALLED"),
        ("CUDA available", "yes" if info.cuda_available else "no"),
        ("CUDA version", info.cuda_version or "N/A"),
        ("Device count", str(info.device_count)),
    ]
    for i, name in enumerate(info.gpu_names):
        rows.append((f"  GPU {i}", name))

    label_w = max(len(r[0]) for r in rows) + 2
    for label, val in rows:
        print(f"  {label:<{label_w}} {val}")
    print(SEP)


def print_report(results: List[DepStatus], verbose: bool = False) -> None:
    """Print a human-readable table with ANSI colour coding."""
    print()
    print(SEP)
    print(f"  {_c(_BOLD, 'Encoder Dependency Report')}")
    print(SEP)

    # Column widths
    name_w = max(len(r.name) for r in results) + 2
    status_w = 12
    ver_w = 14
    used_w = max(len(r.used_by) for r in results) + 2

    header = (
        f"  {'Package':<{name_w}}"
        f"{'Status':<{status_w}}"
        f"{'Version':<{ver_w}}"
        f"{'Used By':<{used_w}}"
        f"Fallback"
    )
    print(header)
    print(SEP)

    for r in results:
        # Colour the status cell
        if r.status == "available":
            status_str = _c(_GREEN, "available")
        elif r.status == "outdated":
            status_str = _c(_RED, "outdated")
        else:
            # missing — yellow if a fallback exists, red otherwise
            if r.fallback:
                status_str = _c(_YELLOW, "missing")
            else:
                status_str = _c(_RED, "missing")

        ver_str = r.version or "---"
        if r.status == "outdated" and r.min_version:
            ver_str = f"{ver_str} (need >={r.min_version})"

        # Compensate for invisible ANSI chars when padding
        ansi_pad = len(status_str) - len(status_str.replace("\033[", "").replace("m", "").replace("[0", ""))
        # Simpler: just measure visible length
        visible_status = r.status if r.status != "available" else "available"
        extra = len(status_str) - len(visible_status)

        print(
            f"  {r.name:<{name_w}}"
            f"{status_str}{' ' * (status_w - len(visible_status))}"
            f"{ver_str:<{ver_w}}"
            f"{r.used_by:<{used_w}}"
            f"{r.fallback}"
        )

        if verbose and r.import_path:
            print(f"  {'':>{name_w}}  path: {r.import_path}")
        if verbose and r.error:
            print(f"  {'':>{name_w}}  error: {r.error}")

    print(SEP)

    n_available = sum(1 for r in results if r.status == "available")
    total = len(results)
    colour = _GREEN if n_available == total else _YELLOW
    print(f"  {_c(colour, f'{n_available}/{total}')} encoder packages available")
    print(SEP)


# ---------------------------------------------------------------------------
# Instantiation checks
# ---------------------------------------------------------------------------

@dataclass
class InstantiationResult:
    encoder_name: str
    status: str          # "optimal", "fallback", "failed"
    detail: str
    error: Optional[str] = None


def _try_instantiate_audio() -> InstantiationResult:
    """Try to create an AudioEncoder and detect which mel frontend is active."""
    try:
        from brain_ai.encoders.audio import AudioEncoder, MelSpectrogramFrontend  # type: ignore
        frontend = MelSpectrogramFrontend()
        if getattr(frontend, "use_torchaudio", False):
            return InstantiationResult(
                "AudioEncoder",
                "optimal",
                "mel via torchaudio (optimal)",
            )
        else:
            return InstantiationResult(
                "AudioEncoder",
                "fallback",
                "mel via torch.stft fallback (functional)",
            )
    except Exception as exc:
        return InstantiationResult("AudioEncoder", "failed", str(exc), error=traceback.format_exc())


def _try_instantiate_sensor() -> InstantiationResult:
    """Try to create a SensorEncoder and detect CfC/LTC backend."""
    try:
        from brain_ai.encoders.sensors import SensorEncoder, NCPS_AVAILABLE  # type: ignore
        encoder = SensorEncoder(input_dim=64, output_dim=128, hidden_dim=64, num_layers=1)
        if NCPS_AVAILABLE:
            return InstantiationResult(
                "SensorEncoder",
                "optimal",
                "ncps CfC/LTC available; built-in CfC also works",
            )
        else:
            return InstantiationResult(
                "SensorEncoder",
                "fallback",
                "built-in CfC/LTC (ncps not installed)",
            )
    except ImportError:
        # NCPS_AVAILABLE might not be exported — handle gracefully
        try:
            from brain_ai.encoders.sensors import SensorEncoder  # type: ignore
            SensorEncoder(input_dim=64, output_dim=128, hidden_dim=64, num_layers=1)
            return InstantiationResult(
                "SensorEncoder",
                "fallback",
                "built-in CfC/LTC (ncps status unknown)",
            )
        except Exception as exc:
            return InstantiationResult("SensorEncoder", "failed", str(exc), error=traceback.format_exc())
    except Exception as exc:
        return InstantiationResult("SensorEncoder", "failed", str(exc), error=traceback.format_exc())


def _try_instantiate_text() -> InstantiationResult:
    """Try to create a TextEncoder."""
    try:
        from brain_ai.encoders.text import TextEncoder  # type: ignore
        TextEncoder(vocab_size=1000, embed_dim=64, output_dim=128, num_layers=1, num_heads=2, ff_dim=128)
        try:
            import transformers  # noqa: F401
            return InstantiationResult(
                "TextEncoder",
                "optimal",
                "built-in nn.TransformerEncoder + HF transformers available",
            )
        except ImportError:
            return InstantiationResult(
                "TextEncoder",
                "fallback",
                "built-in nn.TransformerEncoder (HF transformers not available)",
            )
    except Exception as exc:
        return InstantiationResult("TextEncoder", "failed", str(exc), error=traceback.format_exc())


def _try_instantiate_vision() -> InstantiationResult:
    """Try to create a VisionEncoder."""
    try:
        from brain_ai.encoders.vision import VisionEncoder  # type: ignore
        VisionEncoder(input_channels=3, output_dim=128, channels=[16, 32], num_steps=2, input_size=(32, 32))
        try:
            import torchvision  # noqa: F401
            return InstantiationResult(
                "VisionEncoder",
                "optimal",
                "built-in spiking CNN + torchvision backbones available",
            )
        except ImportError:
            return InstantiationResult(
                "VisionEncoder",
                "fallback",
                "built-in spiking CNN (torchvision not available)",
            )
    except Exception as exc:
        return InstantiationResult("VisionEncoder", "failed", str(exc), error=traceback.format_exc())


def _try_instantiate_event_vision() -> InstantiationResult:
    """Try to create an EventVisionEncoder."""
    try:
        from brain_ai.encoders.vision import EventVisionEncoder  # type: ignore
        EventVisionEncoder(output_dim=128, height=32, width=32, num_bins=4, channels=[16, 32])
        try:
            import tonic  # noqa: F401
            return InstantiationResult(
                "EventVisionEncoder",
                "optimal",
                "voxel grid encoder + tonic event transforms available",
            )
        except ImportError:
            return InstantiationResult(
                "EventVisionEncoder",
                "fallback",
                "voxel grid encoder only (tonic not available)",
            )
    except Exception as exc:
        return InstantiationResult("EventVisionEncoder", "failed", str(exc), error=traceback.format_exc())


def run_instantiation_checks() -> List[InstantiationResult]:
    """Run all encoder instantiation checks and return results."""
    checks = [
        _try_instantiate_audio,
        _try_instantiate_sensor,
        _try_instantiate_text,
        _try_instantiate_vision,
        _try_instantiate_event_vision,
    ]
    return [fn() for fn in checks]


def print_instantiation_report(results: List[InstantiationResult], verbose: bool = False) -> None:
    """Print instantiation check results."""
    print()
    print(SEP)
    print(f"  {_c(_BOLD, 'Encoder Instantiation Checks')}")
    print(SEP)

    name_w = max(len(r.encoder_name) for r in results) + 2

    for r in results:
        if r.status == "optimal":
            tag = _c(_GREEN, "[OPTIMAL] ")
        elif r.status == "fallback":
            tag = _c(_YELLOW, "[FALLBACK]")
        else:
            tag = _c(_RED, "[FAILED]  ")

        print(f"  {tag}  {r.encoder_name:<{name_w}} {r.detail}")
        if verbose and r.error:
            for line in r.error.strip().splitlines():
                print(f"           {'':>{name_w}} {line}")

    print(SEP)

    n_ok = sum(1 for r in results if r.status in ("optimal", "fallback"))
    n_total = len(results)
    colour = _GREEN if n_ok == n_total else _RED
    print(f"  {_c(colour, f'{n_ok}/{n_total}')} encoders instantiated successfully")

    n_optimal = sum(1 for r in results if r.status == "optimal")
    n_fallback = sum(1 for r in results if r.status == "fallback")
    n_failed = sum(1 for r in results if r.status == "failed")
    print(f"  ({n_optimal} optimal, {n_fallback} fallback, {n_failed} failed)")
    print(SEP)


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def build_json_output(
    sys_info: SystemInfo,
    dep_results: List[DepStatus],
    inst_results: Optional[List[InstantiationResult]] = None,
) -> str:
    """Serialise everything as pretty-printed JSON."""
    system_dict = {
        "python_version": sys_info.python_version,
        "platform": sys_info.platform_str,
        "torch_version": sys_info.torch_version,
        "cuda_available": sys_info.cuda_available,
        "cuda_version": sys_info.cuda_version,
        "device_count": sys_info.device_count,
        "gpu_names": sys_info.gpu_names,
    }

    deps_list = []
    for r in dep_results:
        entry: Dict[str, Any] = {
            "name": r.name,
            "status": r.status,
            "version": r.version,
            "used_by": r.used_by,
            "aux_key": r.aux_key,
            "fallback": r.fallback if r.status != "available" else None,
            "import_path": r.import_path,
            "import_error": r.error,
        }
        if r.min_version:
            entry["min_version"] = r.min_version
        deps_list.append(entry)

    output: Dict[str, Any] = {
        "system": system_dict,
        "encoder_dependencies": deps_list,
        "summary": {
            "total": len(dep_results),
            "available": sum(1 for r in dep_results if r.status == "available"),
            "missing": sum(1 for r in dep_results if r.status == "missing"),
            "outdated": sum(1 for r in dep_results if r.status == "outdated"),
        },
    }

    if inst_results is not None:
        output["instantiation_checks"] = [
            {
                "encoder": r.encoder_name,
                "status": r.status,
                "detail": r.detail,
                "error": r.error,
            }
            for r in inst_results
        ]
        output["instantiation_summary"] = {
            "total": len(inst_results),
            "optimal": sum(1 for r in inst_results if r.status == "optimal"),
            "fallback": sum(1 for r in inst_results if r.status == "fallback"),
            "failed": sum(1 for r in inst_results if r.status == "failed"),
        }

    return json.dumps(output, indent=2)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Audit encoder-specific optional dependencies for brain_ai. "
            "Reports installed packages, active fallback paths, and "
            "optionally verifies encoder instantiation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python encoder_deps_report.py                  # table output\n"
            "  python encoder_deps_report.py --json           # JSON output\n"
            "  python encoder_deps_report.py --check-instantiate --verbose\n"
        ),
    )
    p.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output structured JSON with all dependency and system info",
    )
    p.add_argument(
        "--check-instantiate",
        action="store_true",
        default=False,
        help=(
            "Try to instantiate each encoder type and report which "
            "variants are available vs. degraded (requires brain_ai importable)"
        ),
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Show detailed version info, file paths, and error tracebacks",
    )
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # 1. Gather system info
    sys_info = gather_system_info()

    # 2. Check all encoder dependencies
    dep_results = check_all_dependencies()

    # 3. Optional instantiation checks
    inst_results: Optional[List[InstantiationResult]] = None
    if args.check_instantiate:
        try:
            import torch  # noqa: F401
        except ImportError:
            print("ERROR: --check-instantiate requires PyTorch. Install with:")
            print("  pip install torch")
            sys.exit(1)

        # Verify brain_ai is importable
        try:
            import brain_ai  # noqa: F401
        except ImportError:
            print(
                "WARNING: brain_ai is not importable. Skipping instantiation checks.\n"
                f"  Looked for brain_ai under: {_REPO_ROOT}\n"
                "  Ensure the package is installed or PYTHONPATH is set."
            )
            inst_results = None
        else:
            inst_results = run_instantiation_checks()

    # 4. Output
    if args.json:
        print(build_json_output(sys_info, dep_results, inst_results))
    else:
        _print_system_info(sys_info)
        print_report(dep_results, verbose=args.verbose)
        if inst_results is not None:
            print_instantiation_report(inst_results, verbose=args.verbose)
        print()


if __name__ == "__main__":
    main()
