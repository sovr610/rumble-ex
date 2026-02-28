#!/usr/bin/env python3
"""
deps_report.py -- Check optional BrainAI dependencies and report fallback selections.

Inspects the Python environment for every optional package used by brain_ai,
prints a formatted table showing version or "missing -> <fallback>", and
optionally emits machine-readable JSON.

Usage:
    python deps_report.py
    python deps_report.py --json
    python deps_report.py --json > deps.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version as pkg_version
from typing import List, Optional


# ---------------------------------------------------------------------------
# Dependency descriptor
# ---------------------------------------------------------------------------

@dataclass
class DepInfo:
    """All information about one optional dependency."""
    package: str          # pip / importlib name
    import_name: str      # actual Python module name (may differ)
    role: str             # one-line description of what it provides in brain_ai
    fallback: str         # what brain_ai falls back to if missing
    installed: bool = False
    installed_version: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Dependency catalogue
# ---------------------------------------------------------------------------

# Each entry describes one optional package checked by brain_ai.
# The import_name is tried with importlib; the package name is used for
# importlib.metadata version lookup.

OPTIONAL_DEPS: List[DepInfo] = [
    DepInfo(
        package="ncps",
        import_name="ncps",
        role="Closed-form Continuous-time (CfC) / Liquid Time-Constant (LTC) "
             "working-memory cells in the Global Workspace",
        fallback="GRU (torch.nn.GRU)",
    ),
    DepInfo(
        package="htm.core",
        import_name="htm",
        role="Native C++ Hierarchical Temporal Memory (HTM/SP/TM algorithms)",
        fallback="LSTM fallback (HTMLayer in temporal/htm.py)",
    ),
    DepInfo(
        package="pymdp",
        import_name="pymdp",
        role="Active inference / Expected Free Energy (EFE) planning",
        fallback="Custom EFE implementation in decision/active_inference.py",
    ),
    DepInfo(
        package="learn2learn",
        import_name="learn2learn",
        role="MAML / MAML++ meta-learning (learn2learn.algorithms.MAML)",
        fallback="Custom first-order MAML in meta/neuromodulation.py",
    ),
    DepInfo(
        package="higher",
        import_name="higher",
        role="Higher-order gradient / functional optimizers for meta-learning",
        fallback="Manual gradient unrolling in custom MAML fallback",
    ),
    DepInfo(
        package="snntorch",
        import_name="snntorch",
        role="Pre-built LIF neuron cells and surrogate-gradient functions",
        fallback="Custom LIF / surrogate-gradient implementation in core/snn.py",
    ),
    DepInfo(
        package="ltn",
        import_name="ltn",
        role="Logic Tensor Networks for differentiable neuro-symbolic reasoning",
        fallback="Product t-norm fuzzy logic in reasoning/system2.py",
    ),
]


# ---------------------------------------------------------------------------
# Probing helpers
# ---------------------------------------------------------------------------

def _probe_import(dep: DepInfo) -> None:
    """Try to import dep.import_name and record the result in-place."""
    try:
        mod = __import__(dep.import_name)
        dep.installed = True

        # Try importlib.metadata first (most reliable)
        try:
            dep.installed_version = pkg_version(dep.package)
        except PackageNotFoundError:
            # Fall back to __version__ attribute on the module itself
            dep.installed_version = getattr(mod, "__version__", "unknown")

    except ImportError as exc:
        dep.installed = False
        dep.error = str(exc)


def probe_all(deps: List[DepInfo]) -> None:
    """Probe all dependencies in-place."""
    for dep in deps:
        _probe_import(dep)


# ---------------------------------------------------------------------------
# System-level info
# ---------------------------------------------------------------------------

@dataclass
class SystemInfo:
    python_version: str
    python_impl: str
    platform_str: str
    torch_version: Optional[str]
    cuda_available: bool
    cuda_version: Optional[str]
    num_gpus: int
    gpu_names: List[str]


def gather_system_info() -> SystemInfo:
    py_ver = platform.python_version()
    py_impl = platform.python_implementation()
    platform_str = platform.platform()

    torch_ver: Optional[str] = None
    cuda_avail = False
    cuda_ver: Optional[str] = None
    num_gpus = 0
    gpu_names: List[str] = []

    try:
        import torch
        torch_ver = torch.__version__
        cuda_avail = torch.cuda.is_available()
        if cuda_avail:
            cuda_ver = torch.version.cuda
            num_gpus = torch.cuda.device_count()
            gpu_names = [
                torch.cuda.get_device_name(i) for i in range(num_gpus)
            ]
    except ImportError:
        pass

    return SystemInfo(
        python_version=py_ver,
        python_impl=py_impl,
        platform_str=platform_str,
        torch_version=torch_ver,
        cuda_available=cuda_avail,
        cuda_version=cuda_ver,
        num_gpus=num_gpus,
        gpu_names=gpu_names,
    )


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

def _print_system_table(info: SystemInfo) -> None:
    SEP = "-" * 72
    print(SEP)
    print("  System Environment")
    print(SEP)
    rows = [
        ("Python version", f"{info.python_version} ({info.python_impl})"),
        ("Platform", info.platform_str),
        ("PyTorch version", info.torch_version or "NOT INSTALLED"),
        ("CUDA available", "yes" if info.cuda_available else "no"),
        ("CUDA version", info.cuda_version or "N/A"),
        ("GPU count", str(info.num_gpus)),
    ]
    for i, name in enumerate(info.gpu_names):
        rows.append((f"  GPU {i}", name))

    label_w = max(len(r[0]) for r in rows) + 2
    for label, val in rows:
        print(f"  {label:<{label_w}} {val}")
    print(SEP)


def _print_deps_table(deps: List[DepInfo]) -> None:
    SEP = "-" * 72
    print()
    print(SEP)
    print("  Optional Dependencies")
    print(SEP)

    pkg_w = max(len(d.package) for d in deps) + 2
    status_w = 20  # enough for "2.1.0" or "missing -> GRU"

    header = f"  {'Package':<{pkg_w}} {'Status':<{status_w}} Role"
    print(header)
    print(SEP)

    for dep in deps:
        if dep.installed:
            status = dep.installed_version or "installed"
        else:
            # Truncate the fallback to keep the table readable
            fb = dep.fallback
            if len(fb) > 40:
                fb = fb[:37] + "..."
            status = f"missing -> {fb}"

        # Wrap long role strings
        role_indent = " " * (2 + pkg_w + status_w + 1)
        role_lines = _wrap_text(dep.role, width=40)
        first_line = role_lines[0]
        extra_lines = role_lines[1:]

        print(f"  {dep.package:<{pkg_w}} {status:<{status_w}} {first_line}")
        for line in extra_lines:
            print(f"{role_indent}{line}")

    print(SEP)
    installed_count = sum(1 for d in deps if d.installed)
    print(
        f"  {installed_count}/{len(deps)} optional dependencies installed"
    )
    print(SEP)


def _wrap_text(text: str, width: int) -> List[str]:
    """Simple word-wrapping -- avoids importing textwrap just for this."""
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = (current + " " + word).strip()
    if current:
        lines.append(current)
    return lines if lines else [""]


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def _build_json_output(
    info: SystemInfo,
    deps: List[DepInfo],
) -> str:
    """Serialise environment + deps info as pretty-printed JSON."""
    system_dict = {
        "python_version": info.python_version,
        "python_impl": info.python_impl,
        "platform": info.platform_str,
        "torch_version": info.torch_version,
        "cuda_available": info.cuda_available,
        "cuda_version": info.cuda_version,
        "num_gpus": info.num_gpus,
        "gpu_names": info.gpu_names,
    }

    deps_list = []
    for dep in deps:
        deps_list.append({
            "package": dep.package,
            "import_name": dep.import_name,
            "installed": dep.installed,
            "version": dep.installed_version,
            "fallback": dep.fallback if not dep.installed else None,
            "role": dep.role,
            "import_error": dep.error if not dep.installed else None,
        })

    output = {
        "system": system_dict,
        "optional_dependencies": deps_list,
        "summary": {
            "total": len(deps),
            "installed": sum(1 for d in deps if d.installed),
            "missing": sum(1 for d in deps if not d.installed),
        },
    }

    return json.dumps(output, indent=2)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Check optional BrainAI dependencies and report fallback selections"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Emit machine-readable JSON instead of (or in addition to) the table",
    )
    p.add_argument(
        "--json-only",
        action="store_true",
        default=False,
        help="Emit ONLY JSON (no human-readable table).  Implies --json.",
    )
    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Probe everything
    probe_all(OPTIONAL_DEPS)
    sys_info = gather_system_info()

    emit_table = not args.json_only
    emit_json = args.json or args.json_only

    if emit_table:
        _print_system_table(sys_info)
        _print_deps_table(OPTIONAL_DEPS)

    if emit_json:
        json_str = _build_json_output(sys_info, OPTIONAL_DEPS)
        if emit_table:
            # Separate the JSON from the table with a blank line
            print()
        print(json_str)


if __name__ == "__main__":
    main()
