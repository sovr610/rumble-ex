#!/usr/bin/env python3
"""
gen_flag_tests.py -- Generate parameterized pytest tests for BrainAI feature flag
combinations.

Generates a self-contained pytest file that instantiates BrainAI under every
selected flag combination and asserts no crash and no NaN output.

Coverage modes
--------------
  pairwise  -- ~20 combinations covering every flag pair at least once
               (AllPairs / orthogonal-array reduction)
  full      -- all 64 combinations (2^6 flags)

Usage:
    python gen_flag_tests.py
    python gen_flag_tests.py --output tests/test_flag_matrix.py
    python gen_flag_tests.py --coverage full --output tests/test_flag_matrix_full.py
"""

from __future__ import annotations

import argparse
import ast
import itertools
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Feature flags managed by this tool
# ---------------------------------------------------------------------------

FLAGS: List[str] = [
    "use_snn",
    "use_htm",
    "use_workspace",
    "use_symbolic",
    "use_meta",
    "use_engram",
]

FlagCombo = Dict[str, bool]


# ---------------------------------------------------------------------------
# AllPairs / pairwise algorithm
# ---------------------------------------------------------------------------

def _all_pairs(flags: List[str]) -> List[FlagCombo]:
    """
    Generate a minimal set of flag combinations that covers every pair of
    (flag_i=v_i, flag_j=v_j) at least once.

    Uses the in-parameter-order (IPO) greedy algorithm for pairwise coverage,
    requiring no external dependency.

    For 6 binary parameters the result is typically 12-20 test cases, well
    under the 64 that full coverage requires.
    """
    n = len(flags)
    values = [False, True]  # binary domain

    # Track which pairs are still uncovered.
    # A pair is (flag_i, val_i, flag_j, val_j) with i < j.
    uncovered: set[Tuple[int, bool, int, bool]] = set()
    for i in range(n):
        for j in range(i + 1, n):
            for vi in values:
                for vj in values:
                    uncovered.add((i, vi, j, vj))

    combos: List[FlagCombo] = []

    def pairs_covered_by(combo: List[bool]) -> set[Tuple[int, bool, int, bool]]:
        covered: set[Tuple[int, bool, int, bool]] = set()
        for i in range(n):
            for j in range(i + 1, n):
                pair = (i, combo[i], j, combo[j])
                if pair in uncovered:
                    covered.add(pair)
        return covered

    while uncovered:
        # Greedily pick the next combo that covers the most uncovered pairs.
        best_combo: List[bool] = []
        best_count = -1

        for candidate in itertools.product(values, repeat=n):
            candidate_list = list(candidate)
            count = len(pairs_covered_by(candidate_list))
            if count > best_count:
                best_count = count
                best_combo = candidate_list

        combos.append(dict(zip(flags, best_combo)))
        uncovered -= pairs_covered_by(best_combo)

    return combos


def _full_matrix(flags: List[str]) -> List[FlagCombo]:
    """Return all 2^n flag combinations."""
    combos: List[FlagCombo] = []
    for vals in itertools.product([False, True], repeat=len(flags)):
        combos.append(dict(zip(flags, vals)))
    return combos


# ---------------------------------------------------------------------------
# Test-file code generation
# ---------------------------------------------------------------------------

# Mapping from flag to short letter for compact test IDs
_FLAG_LETTER = {
    "use_snn": "S",
    "use_htm": "H",
    "use_workspace": "W",
    "use_symbolic": "Y",
    "use_meta": "M",
    "use_engram": "E",
}


def _combo_id(combo: FlagCombo) -> str:
    """Return a compact human-readable ID like 'S1H0W1Y0M1E0'."""
    parts = [f"{_FLAG_LETTER.get(k, k[0])}{int(v)}" for k, v in combo.items()]
    return "".join(parts)


def _combo_repr(combo: FlagCombo) -> str:
    """Return a Python dict literal for a flag combination."""
    items = ", ".join(f'"{k}": {v}' for k, v in combo.items())
    return "{" + items + "}"


def _render_param_list(combos: List[FlagCombo], indent: str = "    ") -> str:
    """
    Render a pytest.param(...) list suitable for embedding inside
    @pytest.mark.parametrize.
    """
    lines: List[str] = []
    for i, combo in enumerate(combos):
        suffix = "," if i < len(combos) - 1 else ""
        lines.append(
            f'{indent}pytest.param({_combo_repr(combo)}, id="{_combo_id(combo)}"){suffix}'
        )
    return "\n".join(lines)


def _generate_test_source(combos: List[FlagCombo], coverage: str) -> str:
    """
    Produce the complete pytest source as a plain string.

    The template is defined at the top level of this function using a raw
    string with explicit PARAM_LIST and other tokens, then substituted via
    str.replace() so that indentation in the generated file is not affected
    by the indentation of this source file.
    """
    param_list = _render_param_list(combos)
    num_combos = len(combos)
    num_flags = len(FLAGS)

    # Build the generated file line-by-line to avoid indentation surprises
    # from any triple-quoted string that lives inside an indented function.
    lines: List[str] = []

    def L(text: str = "") -> None:  # noqa: N802 -- short alias for append
        lines.append(text)

    L('"""')
    L("Auto-generated by gen_flag_tests.py")
    L(f"Coverage mode : {coverage}")
    L(f"Flag count    : {num_flags}")
    L(f"Combinations  : {num_combos}")
    L()
    L("DO NOT EDIT BY HAND -- regenerate with:")
    L(f"    python gen_flag_tests.py --coverage {coverage}")
    L('"""')
    L()
    L("from __future__ import annotations")
    L()
    L("import pytest")
    L("import torch")
    L()
    L("# " + "-" * 75)
    L("# Guard: skip the entire module if brain_ai is not importable.")
    L("# " + "-" * 75)
    L()
    L("try:")
    L("    from brain_ai.system import BrainAI")
    L("    from brain_ai.config import BrainAIConfig")
    L("    BRAIN_AI_AVAILABLE = True")
    L("except ImportError:")
    L("    BRAIN_AI_AVAILABLE = False")
    L()
    L()
    L("pytestmark = pytest.mark.skipif(")
    L("    not BRAIN_AI_AVAILABLE,")
    L('    reason="brain_ai package not found on PYTHONPATH",')
    L(")")
    L()
    L()
    L("# " + "-" * 75)
    L("# Fixtures")
    L("# " + "-" * 75)
    L()
    L()
    L('@pytest.fixture(scope="session")')
    L("def device() -> str:")
    L('    return "cuda" if torch.cuda.is_available() else "cpu"')
    L()
    L()
    L('@pytest.fixture(scope="session")')
    L("def batch_size() -> int:")
    L("    return 2")
    L()
    L()
    L('@pytest.fixture(scope="session")')
    L("def vision_input(device: str, batch_size: int) -> torch.Tensor:")
    L('    """3-channel RGB tensor; 32x32 survives all MaxPool2d stages."""')
    L("    return torch.randn(batch_size, 3, 32, 32, device=device)")
    L()
    L()
    L('@pytest.fixture(scope="session")')
    L("def sample_inputs(vision_input: torch.Tensor) -> dict:")
    L('    """Default multi-modal input dict used by all flag-matrix tests."""')
    L('    return {"vision": vision_input}')
    L()
    L()
    L("# " + "-" * 75)
    L("# Helpers")
    L("# " + "-" * 75)
    L()
    L()
    L("def _build_model(flags: dict, device: str) -> BrainAI:")
    L("    '''")
    L("    Construct a BrainAI model with the minimal config and the given feature")
    L("    flags applied.  Returned model is in eval mode on the target device.")
    L("    '''")
    L("    config = BrainAIConfig.minimal()")
    L('    config.modalities = ["vision"]')
    L()
    L("    for flag_name, flag_value in flags.items():")
    L("        setattr(config, flag_name, flag_value)")
    L()
    L("    model = BrainAI(")
    L("        config=config,")
    L('        modalities=["vision"],')
    L('        output_type="classify",')
    L("    )")
    L("    model.to(device)")
    L("    model.eval()")
    L("    return model")
    L()
    L()
    L("def _has_nan(tensor: torch.Tensor) -> bool:")
    L("    return bool(torch.isnan(tensor).any())")
    L()
    L()
    L("# " + "-" * 75)
    L("# Parameterized test matrix")
    L("# " + "-" * 75)
    L()
    L()
    # test_flag_combination
    L("@pytest.mark.parametrize(")
    L('    "flags",')
    L("    [")
    L(param_list)
    L("    ],")
    L(")")
    L("def test_flag_combination(")
    L("    flags: dict,")
    L("    device: str,")
    L("    batch_size: int,")
    L("    sample_inputs: dict,")
    L(") -> None:")
    L("    '''")
    L("    Instantiate BrainAI with the given flag combination, run one forward")
    L("    pass, and assert:")
    L("      1. No exception is raised during construction or forward.")
    L("      2. The output tensor has the correct batch dimension.")
    L("      3. No NaN values appear in the output.")
    L("    '''")
    L("    model = _build_model(flags, device)")
    L()
    L("    with torch.no_grad():")
    L("        output = model(sample_inputs, return_details=False)")
    L()
    L("    assert isinstance(output, torch.Tensor), (")
    L('        f"Expected Tensor output, got {type(output).__name__}"')
    L("    )")
    L("    assert output.shape[0] == batch_size, (")
    L('        f"Batch dim mismatch: got {output.shape[0]}, expected {batch_size}"')
    L("    )")
    L('    assert not _has_nan(output), "NaN detected in model output"')
    L()
    L()
    # test_flag_combination_with_details
    L("@pytest.mark.parametrize(")
    L('    "flags",')
    L("    [")
    L(param_list)
    L("    ],")
    L(")")
    L("def test_flag_combination_with_details(")
    L("    flags: dict,")
    L("    device: str,")
    L("    batch_size: int,")
    L("    sample_inputs: dict,")
    L(") -> None:")
    L("    '''")
    L("    Same as test_flag_combination but calls forward() with return_details=True")
    L("    and additionally verifies workspace and confidence fields.")
    L("    '''")
    L("    model = _build_model(flags, device)")
    L()
    L("    with torch.no_grad():")
    L("        result = model(sample_inputs, return_details=True)")
    L()
    L("    assert hasattr(result, 'output'), \"SystemOutput missing 'output'\"")
    L("    assert hasattr(result, 'workspace'), \"SystemOutput missing 'workspace'\"")
    L("    assert hasattr(result, 'confidence'), \"SystemOutput missing 'confidence'\"")
    L()
    L("    assert isinstance(result.output, torch.Tensor), (")
    L('        f"Expected Tensor for output, got {type(result.output).__name__}"')
    L("    )")
    L("    assert result.output.shape[0] == batch_size")
    L()
    L("    assert isinstance(result.workspace, torch.Tensor), (")
    L('        f"Expected Tensor for workspace, got {type(result.workspace).__name__}"')
    L("    )")
    L()
    L("    assert isinstance(result.confidence, torch.Tensor), (")
    L('        f"Expected Tensor for confidence, got {type(result.confidence).__name__}"')
    L("    )")
    L("    assert tuple(result.confidence.shape) == (batch_size, 1), (")
    L("        f\"Confidence shape {tuple(result.confidence.shape)} != {(batch_size, 1)}\"")
    L("    )")
    L()
    L('    assert not _has_nan(result.output), "NaN in output"')
    L('    assert not _has_nan(result.workspace), "NaN in workspace"')
    L('    assert not _has_nan(result.confidence), "NaN in confidence"')
    L()
    L()
    # test_flag_combination_cuda
    L("@pytest.mark.skipif(")
    L("    not torch.cuda.is_available(),")
    L('    reason="CUDA not available",')
    L(")")
    L("@pytest.mark.parametrize(")
    L('    "flags",')
    L("    [")
    L(param_list)
    L("    ],")
    L(")")
    L("def test_flag_combination_cuda(")
    L("    flags: dict,")
    L("    batch_size: int,")
    L("    sample_inputs: dict,")
    L(") -> None:")
    L("    '''")
    L("    Run each flag combination on CUDA and verify that all output tensors")
    L("    remain on the GPU.  Skipped automatically when CUDA is unavailable.")
    L("    '''")
    L('    cuda_device = "cuda"')
    L("    cuda_inputs = {")
    L("        k: v.to(cuda_device) for k, v in sample_inputs.items()")
    L("    }")
    L()
    L("    model = _build_model(flags, cuda_device)")
    L()
    L("    with torch.no_grad():")
    L("        output = model(cuda_inputs, return_details=False)")
    L()
    L('    assert output.device.type == "cuda", (')
    L('        f"Expected CUDA tensor, got {output.device}"')
    L("    )")
    L('    assert not _has_nan(output), "NaN detected in CUDA output"')
    L()  # trailing newline

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate parameterized pytest tests for BrainAI feature flags",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--output",
        default="tests/test_flag_matrix.py",
        help="Path for the generated test file (default: tests/test_flag_matrix.py)",
    )
    p.add_argument(
        "--coverage",
        default="pairwise",
        choices=("pairwise", "full"),
        help=(
            "Coverage strategy: 'pairwise' (~20 combos) or "
            "'full' (all 64 combos).  Default: pairwise"
        ),
    )
    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Select combinations
    if args.coverage == "full":
        combos = _full_matrix(FLAGS)
    else:
        combos = _all_pairs(FLAGS)

    print(f"Coverage mode : {args.coverage}")
    print(f"Flags         : {', '.join(FLAGS)}")
    print(f"Combinations  : {len(combos)}")

    # Build and write test source
    source = _generate_test_source(combos, args.coverage)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(source, encoding="utf-8")

    print(f"Generated     : {output_path.resolve()}")

    # Syntax validation via ast.parse (safe -- no code execution)
    try:
        ast.parse(source)
        print("Syntax check  : OK")
    except SyntaxError as exc:
        print(f"Syntax check  : FAILED -- {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
