#!/usr/bin/env python3
"""
generate_project.py
===================
CLI entrypoint for the world model scaffold generator.

Creates a complete, production-grade world model project structure including:
- All 8 ABC base classes (encoders, dynamics, memory, planning, decoders,
  training, evaluation, deployment)
- BaseWorldModel composition root with dependency injection
- YAML configuration files (model, training, dataset, hardware)
- pyproject.toml with dependency declarations
- README.md documenting the modular design

Usage:
    python generate_project.py --output_dir ./my_world_model
    python generate_project.py --output_dir ./my_project --project_name my_model
    python generate_project.py --output_dir ./my_project --skip_configs
    python generate_project.py --output_dir ./my_project --skip_readme --validate
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Locate assets directory
# ---------------------------------------------------------------------------


def _find_assets_dir() -> Path:
    """Locate the assets/ directory relative to this script."""
    this_file = Path(__file__).resolve()

    # scripts/ -> skill root -> assets/
    skill_root = this_file.parent.parent
    assets = skill_root / "assets"
    if assets.is_dir():
        return assets

    # Fallback: look in parent directories
    for parent in this_file.parents:
        candidate = parent / "assets"
        if candidate.is_dir() and (candidate / "scaffold_generator.py").exists():
            return candidate

    raise FileNotFoundError(
        f"Could not locate assets/scaffold_generator.py relative to {this_file}. "
        f"Run this script from the skill directory or ensure assets/ is present."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="generate_project.py",
        description="Generate a world model project scaffold.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap_dedent("""\
            Examples:
              # Generate with default settings
              python generate_project.py --output_dir ./my_world_model

              # Custom project name
              python generate_project.py --output_dir ./my_project --project_name my_model

              # Skip configs and validate the output
              python generate_project.py --output_dir ./my_project --skip_configs --validate

              # Minimal output (no configs, no README)
              python generate_project.py --output_dir ./my_project --skip_configs --skip_readme
        """),
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        metavar="PATH",
        help="Root directory for the generated project. Created if it does not exist.",
    )
    parser.add_argument(
        "--project_name",
        default="world_model",
        metavar="NAME",
        help="PyPI / module name for the project. Default: world_model",
    )
    parser.add_argument(
        "--skip_configs",
        action="store_true",
        default=False,
        help="Do not write YAML configuration files.",
    )
    parser.add_argument(
        "--skip_readme",
        action="store_true",
        default=False,
        help="Do not write README.md.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=False,
        help="Validate the generated scaffold after creation.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress per-file output.",
    )

    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)

    # -----------------------------------------------------------------------
    # Locate and import ScaffoldGenerator
    # -----------------------------------------------------------------------
    try:
        assets_dir = _find_assets_dir()
    except FileNotFoundError as exc:
        _error(str(exc))

    if str(assets_dir) not in sys.path:
        sys.path.insert(0, str(assets_dir))

    try:
        from scaffold_generator import ScaffoldGenerator
    except ImportError as exc:
        _error(
            f"Cannot import ScaffoldGenerator from {assets_dir}: {exc}\n"
            f"Ensure scaffold_generator.py exists in {assets_dir}."
        )

    # -----------------------------------------------------------------------
    # Print header
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("World Model Scaffold Generator")
    print("=" * 60)
    print(f"  Output directory : {output_dir}")
    print(f"  Project name     : {args.project_name}")
    print(f"  Skip configs     : {args.skip_configs}")
    print(f"  Skip README      : {args.skip_readme}")
    print(f"  Validate after   : {args.validate}")
    print("=" * 60)
    print()

    # -----------------------------------------------------------------------
    # Generate scaffold
    # -----------------------------------------------------------------------
    start_time = time.monotonic()

    gen = ScaffoldGenerator(verbose=not args.quiet)

    try:
        gen.generate(
            output_dir=output_dir,
            project_name=args.project_name,
            skip_configs=args.skip_configs,
            skip_readme=args.skip_readme,
        )
    except PermissionError as exc:
        _error(f"Permission denied while writing scaffold: {exc}")
    except OSError as exc:
        _error(f"OS error while writing scaffold: {exc}")
    except Exception as exc:
        _error(f"Unexpected error during scaffold generation: {type(exc).__name__}: {exc}")

    elapsed = time.monotonic() - start_time
    created_count = len(gen._created)

    print()
    print(f"Done in {elapsed:.2f}s — {created_count} file(s) written.")

    # -----------------------------------------------------------------------
    # Print summary of created files
    # -----------------------------------------------------------------------
    if not args.quiet and created_count > 0:
        print()
        print("Files created:")
        for path in sorted(gen._created):
            rel = os.path.relpath(path, output_dir)
            size = os.path.getsize(path)
            print(f"  {rel:55s} {size:>6} bytes")

    # -----------------------------------------------------------------------
    # Validate (optional)
    # -----------------------------------------------------------------------
    if args.validate:
        print()
        print("=" * 60)
        print("Validating scaffold...")
        print("=" * 60)

        valid = gen.validate_scaffold(output_dir)

        if valid:
            print("Validation: PASS")
        else:
            print("Validation: FAIL — some required files are missing or empty.")
            print(
                f"Run 'python scripts/validate_scaffold.py --output_dir {output_dir}' "
                f"for detailed gate reports."
            )
            sys.exit(1)

    # -----------------------------------------------------------------------
    # Next steps
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Next steps:")
    print("=" * 60)
    print(f"  cd {output_dir}")
    print(f"  pip install -e '.[dev]'")
    print(f"  pytest tests/ -v")
    print()
    print("To generate tests:")
    print(
        f"  python {os.path.relpath(str(assets_dir.parent / 'scripts' / 'gen_scaffold_tests.py'))}"
        f" --output tests/test_scaffold.py"
    )
    print()
    print("To validate the scaffold:")
    print(
        f"  python {os.path.relpath(str(assets_dir.parent / 'scripts' / 'validate_scaffold.py'))}"
        f" --output_dir {output_dir}"
    )
    print()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def textwrap_dedent(text: str) -> str:
    """Dedent a multi-line string (avoid importing textwrap at module level)."""
    import textwrap
    return textwrap.dedent(text)


def _error(message: str) -> None:
    """Print an error message and exit."""
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
