#!/usr/bin/env python3
"""
validate_scaffold.py
====================
Validate a generated world model scaffold against the three done-when gates.

Gate 1 — Directory Structure: all required directories, __init__.py, and
          base.py files must exist and be non-empty.

Gate 2 — ABC Contracts: base classes must define abstract methods, and
          attempting to instantiate an incomplete subclass must raise TypeError.

Gate 3 — Hot-Swap: two different encoder implementations must be injectable
          into the same BaseWorldModel without modifying dynamics or decoder.

Usage:
    python validate_scaffold.py --output_dir /path/to/generated/project
    python validate_scaffold.py --output_dir ./my_world_model --verbose
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

CheckResult = Tuple[str, bool, str]   # (label, passed, detail)


def _pass(label: str, detail: str = "") -> CheckResult:
    return label, True, detail


def _fail(label: str, detail: str = "") -> CheckResult:
    return label, False, detail


# ---------------------------------------------------------------------------
# Gate 1: Directory Structure
# ---------------------------------------------------------------------------

REQUIRED_FILES = [
    "world_model/__init__.py",
    "world_model/encoders/__init__.py",
    "world_model/encoders/base.py",
    "world_model/dynamics/__init__.py",
    "world_model/dynamics/base.py",
    "world_model/memory/__init__.py",
    "world_model/memory/base.py",
    "world_model/planning/__init__.py",
    "world_model/planning/base.py",
    "world_model/decoders/__init__.py",
    "world_model/decoders/base.py",
    "world_model/training/__init__.py",
    "world_model/training/base.py",
    "world_model/evaluation/__init__.py",
    "world_model/evaluation/base.py",
    "world_model/deployment/__init__.py",
    "world_model/deployment/base.py",
    "world_model/core/__init__.py",
    "world_model/core/world_model.py",
    "world_model/configs/model.yaml",
    "world_model/configs/training.yaml",
    "world_model/configs/dataset.yaml",
    "world_model/configs/hardware.yaml",
    "pyproject.toml",
    "README.md",
]


def gate1_structure(output_dir: str) -> List[CheckResult]:
    """Verify all required directories and files exist and are non-empty."""
    results: List[CheckResult] = []
    root = Path(output_dir)

    for rel in REQUIRED_FILES:
        path = root / rel
        if not path.exists():
            results.append(_fail(f"exists: {rel}", "file not found"))
        elif path.stat().st_size == 0:
            results.append(_fail(f"non-empty: {rel}", "file is empty (0 bytes)"))
        else:
            results.append(_pass(f"exists+non-empty: {rel}", f"{path.stat().st_size} bytes"))

    # Verify Python syntax on all .py files
    pkg_dir = root / "world_model"
    for py_path in sorted(pkg_dir.rglob("*.py")):
        rel = py_path.relative_to(root)
        try:
            src = py_path.read_text(encoding="utf-8")
            ast.parse(src)
            results.append(_pass(f"valid-python: {rel}"))
        except SyntaxError as exc:
            results.append(_fail(f"valid-python: {rel}", f"SyntaxError: {exc}"))
        except Exception as exc:
            results.append(_fail(f"valid-python: {rel}", str(exc)))

    return results


# ---------------------------------------------------------------------------
# Gate 2: ABC Contracts
# ---------------------------------------------------------------------------


def _load_module_from_path(module_name: str, path: Path) -> types.ModuleType:
    """Load a Python module from a file path without installing it."""
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _make_minimal_subclass(base_cls, methods_to_implement: List[str]) -> type:
    """Create a subclass of base_cls implementing only the named methods."""
    namespace = {}
    for method_name in methods_to_implement:
        namespace[method_name] = lambda self, *args, **kwargs: None
    return type(f"_Concrete{base_cls.__name__}", (base_cls,), namespace)


def _get_abstract_methods(cls) -> List[str]:
    """Return the list of abstract method names for an ABC class."""
    return sorted(getattr(cls, "__abstractmethods__", frozenset()))


def gate2_abc_contracts(output_dir: str) -> List[CheckResult]:
    """Verify ABC enforcement for each base class."""
    results: List[CheckResult] = []
    root = Path(output_dir)
    pkg = root / "world_model"

    # Map: (subpackage, class_name, base_file)
    classes_to_check = [
        ("encoders", "BaseEncoder", "base.py"),
        ("dynamics", "BaseDynamics", "base.py"),
        ("memory", "BaseMemory", "base.py"),
        ("planning", "BasePlanner", "base.py"),
        ("decoders", "BaseDecoder", "base.py"),
        ("training", "BaseTrainer", "base.py"),
        ("evaluation", "BaseEvaluator", "base.py"),
        ("deployment", "BaseExporter", "base.py"),
    ]

    # Also add dynamic base paths to sys.path so cross-imports resolve
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    loaded_classes = {}

    for subpkg, class_name, filename in classes_to_check:
        path = pkg / subpkg / filename
        mod_name = f"world_model.{subpkg}.{filename[:-3]}"

        if not path.exists():
            results.append(_fail(f"load: {class_name}", f"file not found: {path}"))
            continue

        try:
            mod = _load_module_from_path(mod_name, path)
        except Exception as exc:
            results.append(_fail(f"load: {class_name}", f"import error: {exc}"))
            continue

        if not hasattr(mod, class_name):
            results.append(
                _fail(f"defined: {class_name}", f"not found in {path}")
            )
            continue

        cls = getattr(mod, class_name)
        loaded_classes[class_name] = cls
        results.append(_pass(f"defined: {class_name}"))

        # Check abstract methods are declared
        abstract_methods = _get_abstract_methods(cls)
        if len(abstract_methods) == 0:
            results.append(
                _fail(f"has-abstract-methods: {class_name}", "no @abstractmethod found")
            )
        else:
            results.append(
                _pass(
                    f"has-abstract-methods: {class_name}",
                    f"methods: {abstract_methods}",
                )
            )

        # Direct instantiation must raise TypeError
        try:
            cls()
            results.append(
                _fail(
                    f"direct-instantiation-raises: {class_name}",
                    "expected TypeError, got no error",
                )
            )
        except TypeError:
            results.append(_pass(f"direct-instantiation-raises: {class_name}"))
        except Exception as exc:
            results.append(
                _fail(
                    f"direct-instantiation-raises: {class_name}",
                    f"expected TypeError, got {type(exc).__name__}: {exc}",
                )
            )

        # Omitting one method at a time must also raise TypeError
        for omit_method in abstract_methods:
            remaining = [m for m in abstract_methods if m != omit_method]
            incomplete_cls = _make_minimal_subclass(cls, remaining)
            try:
                incomplete_cls()
                results.append(
                    _fail(
                        f"missing-method-raises: {class_name}.{omit_method}",
                        "expected TypeError, got no error",
                    )
                )
            except TypeError:
                results.append(
                    _pass(f"missing-method-raises: {class_name}.{omit_method}")
                )
            except Exception as exc:
                results.append(
                    _fail(
                        f"missing-method-raises: {class_name}.{omit_method}",
                        f"expected TypeError, got {type(exc).__name__}: {exc}",
                    )
                )

        # Complete subclass must instantiate
        full_cls = _make_minimal_subclass(cls, abstract_methods)
        try:
            full_cls()
            results.append(_pass(f"complete-subclass-instantiates: {class_name}"))
        except Exception as exc:
            results.append(
                _fail(
                    f"complete-subclass-instantiates: {class_name}",
                    f"{type(exc).__name__}: {exc}",
                )
            )

    return results


# ---------------------------------------------------------------------------
# Gate 3: Hot-Swap
# ---------------------------------------------------------------------------


def gate3_hot_swap(output_dir: str) -> List[CheckResult]:
    """Verify two different encoders can be injected into BaseWorldModel."""
    results: List[CheckResult] = []
    root = Path(output_dir)
    pkg = root / "world_model"

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Load component ABCs first so BaseWorldModel's import resolves to these objects
    try:
        enc_mod = _load_module_from_path(
            "world_model.encoders.base", pkg / "encoders" / "base.py"
        )
        dyn_mod = _load_module_from_path(
            "world_model.dynamics.base", pkg / "dynamics" / "base.py"
        )
        dec_mod = _load_module_from_path(
            "world_model.decoders.base", pkg / "decoders" / "base.py"
        )
        mem_mod = _load_module_from_path(
            "world_model.memory.base", pkg / "memory" / "base.py"
        )
        plan_mod = _load_module_from_path(
            "world_model.planning.base", pkg / "planning" / "base.py"
        )
        BaseEncoder = enc_mod.BaseEncoder
        BaseDynamics = dyn_mod.BaseDynamics
        BaseDecoder = dec_mod.BaseDecoder
        results.append(_pass("load: component ABCs"))
    except Exception as exc:
        return results + [_fail("load: component ABCs", f"{type(exc).__name__}: {exc}")]

    # Load BaseWorldModel — it will re-use the already-registered modules above
    wm_path = pkg / "core" / "world_model.py"
    if not wm_path.exists():
        return [_fail("load: BaseWorldModel", f"file not found: {wm_path}")]

    try:
        wm_mod = _load_module_from_path("world_model.core.world_model", wm_path)
        BaseWorldModel = wm_mod.BaseWorldModel
        results.append(_pass("load: BaseWorldModel"))
    except Exception as exc:
        return results + [_fail("load: BaseWorldModel", f"{type(exc).__name__}: {exc}")]

    # Extract the BaseEncoder that BaseWorldModel actually uses for isinstance checks
    # (it may differ from the one we loaded above if the module was already cached)
    BaseEncoder = getattr(wm_mod, "BaseEncoder", BaseEncoder)
    BaseDynamics = getattr(wm_mod, "BaseDynamics", BaseDynamics)
    BaseDecoder = getattr(wm_mod, "BaseDecoder", BaseDecoder)

    # Try to import torch for mock implementations
    try:
        import torch
        import torch.nn as nn
        from torch import Tensor
        HAS_TORCH = True
    except ImportError:
        HAS_TORCH = False

    if not HAS_TORCH:
        results.append(_fail("hot-swap", "torch not available; cannot create mock components"))
        return results

    # Create two distinct encoder implementations
    class _EncoderA(nn.Module, BaseEncoder):
        def forward(self, x: Tensor) -> Tensor:
            return torch.zeros(x.shape[0], 128)
        def get_embed_dim(self) -> int:
            return 128
        def get_output_shape(self):
            return (128,)

    class _EncoderB(nn.Module, BaseEncoder):
        """Second encoder — simulates a different architecture (e.g. Mamba)."""
        def forward(self, x: Tensor) -> Tensor:
            return torch.zeros(x.shape[0], 128)
        def get_embed_dim(self) -> int:
            return 128
        def get_output_shape(self):
            return (128,)

    class _MockDynamics(nn.Module, BaseDynamics):
        def step(self, state: Tensor, action: Tensor) -> Tensor:
            return torch.zeros_like(state)
        def imagine(self, state: Tensor, policy, horizon: int) -> Tensor:
            return torch.zeros(state.shape[0], horizon, 128)
        def get_state_dim(self) -> int:
            return 128

    class _MockDecoder(nn.Module, BaseDecoder):
        def forward(self, latent: Tensor) -> Tensor:
            return torch.zeros(latent.shape[0], 3, 16, 16)
        def get_output_shape(self):
            return (3, 16, 16)
        def get_input_dim(self) -> int:
            return 128

    shared_dynamics = _MockDynamics()
    shared_decoder = _MockDecoder()

    # Inject encoder A
    try:
        model_a = BaseWorldModel(
            encoder=_EncoderA(),
            dynamics=shared_dynamics,
            decoder=shared_decoder,
        )
        results.append(_pass("hot-swap: EncoderA constructs"))
    except Exception as exc:
        results.append(_fail("hot-swap: EncoderA constructs", f"{type(exc).__name__}: {exc}"))
        return results

    # Inject encoder B (same dynamics and decoder)
    try:
        model_b = BaseWorldModel(
            encoder=_EncoderB(),
            dynamics=shared_dynamics,
            decoder=shared_decoder,
        )
        results.append(_pass("hot-swap: EncoderB constructs (same dynamics+decoder)"))
    except Exception as exc:
        results.append(
            _fail(
                "hot-swap: EncoderB constructs",
                f"{type(exc).__name__}: {exc}",
            )
        )
        return results

    # Both models produce valid output
    obs = torch.randn(2, 32)
    action = torch.randn(2, 4)
    for label, model in [("EncoderA", model_a), ("EncoderB", model_b)]:
        try:
            out = model.forward(obs, action)
            expected_shape = (2, 3, 16, 16)
            if out.shape == expected_shape:
                results.append(_pass(f"hot-swap: {label}.forward() shape={tuple(out.shape)}"))
            else:
                results.append(
                    _fail(
                        f"hot-swap: {label}.forward() shape",
                        f"expected {expected_shape}, got {tuple(out.shape)}",
                    )
                )
        except Exception as exc:
            results.append(
                _fail(f"hot-swap: {label}.forward()", f"{type(exc).__name__}: {exc}")
            )

    # Dimensional mismatch must raise ValueError
    class _EncoderWrongDim(nn.Module, BaseEncoder):
        def forward(self, x: Tensor) -> Tensor:
            return torch.zeros(x.shape[0], 999)
        def get_embed_dim(self) -> int:
            return 999
        def get_output_shape(self):
            return (999,)

    try:
        BaseWorldModel(
            encoder=_EncoderWrongDim(),
            dynamics=shared_dynamics,
            decoder=shared_decoder,
        )
        results.append(
            _fail(
                "hot-swap: dim-mismatch raises ValueError",
                "expected ValueError, got no error",
            )
        )
    except ValueError as exc:
        results.append(_pass("hot-swap: dim-mismatch raises ValueError", str(exc)))
    except Exception as exc:
        results.append(
            _fail(
                "hot-swap: dim-mismatch raises ValueError",
                f"expected ValueError, got {type(exc).__name__}: {exc}",
            )
        )

    return results


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------


def _print_gate(
    gate_name: str,
    results: List[CheckResult],
    verbose: bool,
) -> int:
    """Print gate results and return number of failures."""
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    failures = total - passed

    status = "PASS" if failures == 0 else "FAIL"
    print(f"\n{'='*60}")
    print(f"Gate: {gate_name}  [{status}]  {passed}/{total} checks passed")
    print(f"{'='*60}")

    for label, ok, detail in results:
        icon = "PASS" if ok else "FAIL"
        if ok and not verbose:
            continue
        line = f"  [{icon}] {label}"
        if detail:
            line += f"  —  {detail}"
        print(line)

    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a generated world model scaffold."
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Path to the generated project root.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print all checks, not just failures.",
    )
    parser.add_argument(
        "--skip_gate",
        nargs="*",
        choices=["1", "2", "3"],
        default=[],
        help="Gate numbers to skip (e.g. --skip_gate 2 3).",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    if not os.path.isdir(output_dir):
        print(f"ERROR: output_dir does not exist: {output_dir}")
        sys.exit(1)

    print(f"Validating scaffold at: {output_dir}")

    total_failures = 0

    if "1" not in args.skip_gate:
        r1 = gate1_structure(output_dir)
        total_failures += _print_gate("Gate 1: Directory Structure", r1, args.verbose)
    else:
        print("\nSkipping Gate 1.")

    if "2" not in args.skip_gate:
        r2 = gate2_abc_contracts(output_dir)
        total_failures += _print_gate("Gate 2: ABC Contracts", r2, args.verbose)
    else:
        print("\nSkipping Gate 2.")

    if "3" not in args.skip_gate:
        r3 = gate3_hot_swap(output_dir)
        total_failures += _print_gate("Gate 3: Hot-Swap", r3, args.verbose)
    else:
        print("\nSkipping Gate 3.")

    print(f"\n{'='*60}")
    if total_failures == 0:
        print("OVERALL: PASS — all gates passed.")
    else:
        print(f"OVERALL: FAIL — {total_failures} check(s) failed.")
    print(f"{'='*60}\n")

    sys.exit(0 if total_failures == 0 else 1)


if __name__ == "__main__":
    main()
