"""
pyproject_template.py
=====================
Generator for the pyproject.toml file used by the world model scaffold.

The generated TOML uses setuptools as the build backend and includes:
- Core dependencies: torch>=2.2, einops, decord, wandb, pyyaml, numpy
- Optional jax extras: jax[tpu], flax, optax
- Optional dev extras: pytest, ruff, mypy, pytest-cov
- Package discovery pointing to world_model/
- An entry point for the CLI generator

Usage (self-test):
    python pyproject_template.py
"""

from __future__ import annotations

import re
from typing import List, Optional


def generate_pyproject_toml(
    project_name: str = "world-model",
    version: str = "0.1.0",
    description: str = "Modular world model framework with hot-swappable components",
    author_name: str = "World Model Contributors",
    author_email: str = "noreply@example.com",
    python_requires: str = ">=3.10",
    package_dir: str = "world_model",
    include_cli: bool = True,
) -> str:
    """Generate a complete pyproject.toml string.

    Parameters
    ----------
    project_name : str
        PyPI package name (may contain hyphens).
    version : str
        Package version in semver format.
    description : str
        Short one-line package description.
    author_name : str
        Author display name.
    author_email : str
        Author contact email.
    python_requires : str
        Python version constraint, e.g. ">=3.10".
    package_dir : str
        Python package directory to discover (no trailing slash).
    include_cli : bool
        Whether to include the [project.scripts] entry point.

    Returns
    -------
    str
        Valid TOML string suitable for writing to pyproject.toml.
    """
    # Normalize project name for use as a Python identifier in entry points
    module_name = project_name.replace("-", "_")

    scripts_section = ""
    if include_cli:
        scripts_section = f"""
[project.scripts]
{module_name}-generate = "{module_name}.cli.generate:main"
"""

    return f"""[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "{project_name}"
version = "{version}"
description = "{description}"
readme = "README.md"
requires-python = "{python_requires}"
license = {{text = "Apache-2.0"}}
authors = [
    {{name = "{author_name}", email = "{author_email}"}}
]
keywords = [
    "world model",
    "reinforcement learning",
    "deep learning",
    "model-based",
    "representation learning",
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

dependencies = [
    "torch>=2.2",
    "einops>=0.7",
    "decord>=0.6",
    "wandb>=0.16",
    "pyyaml>=6.0",
    "numpy>=1.24",
]

[project.optional-dependencies]
jax = [
    "jax[tpu]>=0.4.20",
    "flax>=0.8",
    "optax>=0.2",
]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "ruff>=0.2",
    "mypy>=1.8",
    "types-PyYAML",
]
all = [
    "{project_name}[jax,dev]",
]
{scripts_section}
[tool.setuptools.packages.find]
where = ["."]
include = ["{package_dir}*"]

[tool.setuptools.package-data]
{package_dir} = ["py.typed", "configs/*.yaml"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--tb=short -q"
markers = [
    "slow: marks tests as slow (deselect with -m 'not slow')",
    "gpu: marks tests requiring CUDA GPU",
]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
ignore = ["E501"]

[tool.ruff.lint.isort]
known-first-party = ["{package_dir}"]

[tool.mypy]
python_version = "3.10"
strict = false
ignore_missing_imports = true
disallow_untyped_defs = true
warn_return_any = true

[tool.coverage.run]
source = ["{package_dir}"]
omit = ["tests/*", "*/conftest.py"]

[tool.coverage.report]
show_missing = true
skip_covered = false
"""


# ---------------------------------------------------------------------------
# Convenience: write pyproject.toml to a directory
# ---------------------------------------------------------------------------


def write_pyproject_toml(
    output_dir: str,
    project_name: str = "world-model",
    version: str = "0.1.0",
    description: str = "Modular world model framework with hot-swappable components",
) -> str:
    """Write pyproject.toml to output_dir and return the file path.

    Parameters
    ----------
    output_dir : str
        Directory in which to write pyproject.toml.
    project_name : str
        PyPI package name.
    version : str
        Package version.
    description : str
        Package description.

    Returns
    -------
    str
        Absolute path to the written file.
    """
    import os
    from pathlib import Path

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    content = generate_pyproject_toml(
        project_name=project_name,
        version=version,
        description=description,
    )
    path = os.path.join(output_dir, "pyproject.toml")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path


# ---------------------------------------------------------------------------
# Self-test block
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys
    import tempfile
    import os

    # Attempt to import a TOML parser (Python 3.11+ has tomllib built-in)
    try:
        import tomllib  # Python 3.11+
        _TOML_LOAD = lambda s: tomllib.loads(s)  # noqa: E731
        _TOML_AVAILABLE = True
        print("Using built-in tomllib (Python 3.11+)")
    except ImportError:
        try:
            import tomli as _tomli  # type: ignore[import]
            _TOML_LOAD = lambda s: _tomli.loads(s)  # noqa: E731
            _TOML_AVAILABLE = True
            print("Using tomli")
        except ImportError:
            _TOML_AVAILABLE = False
            print("No TOML parser available (tomllib/tomli). Skipping parse checks.")
            _TOML_LOAD = None  # type: ignore[assignment]

    errors: list[str] = []

    def fail(msg: str) -> None:
        errors.append(f"FAIL: {msg}")

    def ok(msg: str) -> None:
        print(f"  PASS: {msg}")

    # -----------------------------------------------------------------------
    # Test 1: Generate default TOML
    # -----------------------------------------------------------------------
    print("\n=== Test 1: Default generation ===")
    try:
        content = generate_pyproject_toml()
        assert isinstance(content, str), "Must return str"
        assert len(content) > 0, "Must be non-empty"
        ok(f"Generated {len(content)} characters")
    except Exception as exc:
        fail(f"Default generation failed: {exc}")

    # -----------------------------------------------------------------------
    # Test 2: Custom project name and version
    # -----------------------------------------------------------------------
    print("\n=== Test 2: Custom project name ===")
    try:
        content = generate_pyproject_toml(
            project_name="my-world-model",
            version="1.2.3",
            description="Custom test",
        )
        assert "my-world-model" in content, "project name must appear in TOML"
        assert "1.2.3" in content, "version must appear in TOML"
        assert "Custom test" in content, "description must appear in TOML"
        ok("Custom project_name, version, description embedded correctly")
    except Exception as exc:
        fail(f"Custom name test failed: {exc}")

    # -----------------------------------------------------------------------
    # Test 3: TOML parsing and required sections
    # -----------------------------------------------------------------------
    print("\n=== Test 3: TOML parsing ===")
    if _TOML_AVAILABLE and _TOML_LOAD is not None:
        try:
            content = generate_pyproject_toml()
            parsed = _TOML_LOAD(content)

            assert "project" in parsed, "[project] section missing"
            ok("[project] section present")

            assert "build-system" in parsed, "[build-system] section missing"
            ok("[build-system] section present")

            project = parsed["project"]
            assert "name" in project, "[project].name missing"
            assert isinstance(project["name"], str) and len(project["name"]) > 0
            ok(f"[project].name = '{project['name']}'")

            assert "version" in project, "[project].version missing"
            version_pattern = r"^\d+\.\d+\.\d+"
            assert re.match(version_pattern, project["version"]), "version not semver"
            ok(f"[project].version = '{project['version']}'")

            assert "requires-python" in project, "[project].requires-python missing"
            ok(f"[project].requires-python = '{project['requires-python']}'")

            assert "dependencies" in project, "[project].dependencies missing"
            deps = project["dependencies"]
            assert isinstance(deps, list) and len(deps) > 0, "dependencies must be list"
            ok(f"[project].dependencies has {len(deps)} entries")

            # Check specific dependencies
            dep_strings = " ".join(deps)
            for dep_check in ["torch", "einops", "pyyaml", "numpy", "wandb", "decord"]:
                found = any(dep_check.lower() in d.lower() for d in deps)
                if found:
                    ok(f"Dependency '{dep_check}' present")
                else:
                    fail(f"Dependency '{dep_check}' not found in: {deps}")

            # Check torch version specifier
            torch_dep = next(
                (d for d in deps if d.lower().startswith("torch")), None
            )
            assert torch_dep is not None, "torch not in dependencies"
            assert ">=" in torch_dep, f"torch dep must have >= specifier: {torch_dep}"
            ok(f"torch dep has >= specifier: {torch_dep}")

            # Check optional deps
            assert "optional-dependencies" in project, "optional-dependencies missing"
            opt = project["optional-dependencies"]

            assert "jax" in opt, "jax extra missing"
            jax_deps = opt["jax"]
            jax_str = " ".join(jax_deps)
            assert "jax" in jax_str.lower(), "jax extra must include jax"
            ok("jax extra present with jax dependency")

            assert "dev" in opt, "dev extra missing"
            dev_deps = opt["dev"]
            dev_str = " ".join(dev_deps)
            for dev_check in ["pytest", "ruff"]:
                assert dev_check.lower() in dev_str.lower(), f"{dev_check} not in dev extra"
                ok(f"dev extra includes '{dev_check}'")

            # Check build-system
            bs = parsed["build-system"]
            assert "requires" in bs and isinstance(bs["requires"], list)
            assert len(bs["requires"]) > 0
            ok(f"[build-system].requires has {len(bs['requires'])} entries")

        except Exception as exc:
            fail(f"TOML parsing test failed: {exc}")
    else:
        print("  SKIP: no TOML parser available")

    # -----------------------------------------------------------------------
    # Test 4: write_pyproject_toml writes file
    # -----------------------------------------------------------------------
    print("\n=== Test 4: write_pyproject_toml ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            path = write_pyproject_toml(tmpdir, project_name="test-pkg")
            assert os.path.isfile(path), f"File not created: {path}"
            assert os.path.getsize(path) > 0, "File is empty"
            ok(f"File written: {path} ({os.path.getsize(path)} bytes)")

            with open(path, encoding="utf-8") as fh:
                raw = fh.read()
            assert "test-pkg" in raw, "project name not in written file"
            ok("project name present in written file")
        except Exception as exc:
            fail(f"write_pyproject_toml failed: {exc}")

    # -----------------------------------------------------------------------
    # Test 5: include_cli = False omits scripts section
    # -----------------------------------------------------------------------
    print("\n=== Test 5: include_cli=False ===")
    try:
        content_no_cli = generate_pyproject_toml(include_cli=False)
        assert "[project.scripts]" not in content_no_cli, (
            "[project.scripts] must be absent when include_cli=False"
        )
        ok("[project.scripts] absent when include_cli=False")

        content_with_cli = generate_pyproject_toml(include_cli=True)
        assert "[project.scripts]" in content_with_cli, (
            "[project.scripts] must be present when include_cli=True"
        )
        ok("[project.scripts] present when include_cli=True")
    except Exception as exc:
        fail(f"CLI entry point test failed: {exc}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n=== Summary ===")
    if errors:
        for e in errors:
            print(f"  {e}")
        print(f"\n{len(errors)} test(s) FAILED.")
        sys.exit(1)
    else:
        print("All pyproject_template tests PASSED.")
        sys.exit(0)
