# Packaging and Dependencies — World Model Scaffold

This document covers the `pyproject.toml` structure, dependency strategy, optional extras
for JAX-based components, development tooling, and editable install workflow for the world
model scaffold.

---

## pyproject.toml Structure

The scaffold uses the `setuptools` backend with `pyproject.toml` as the single configuration
file. The `hatchling` backend is an acceptable alternative with identical semantics.

### Minimal Skeleton

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "world-model"
version = "0.1.0"
description = "Modular world model framework with hot-swappable components"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "Apache-2.0"}
authors = [
    {name = "Your Name", email = "you@example.com"}
]
keywords = ["world model", "reinforcement learning", "deep learning"]
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
    "world-model[jax,dev]",
]

[project.scripts]
wm-generate = "world_model.cli.generate:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["world_model*"]

[tool.setuptools.package-data]
world_model = ["configs/*.yaml", "py.typed"]
```

---

## Core Dependencies

### torch>=2.2

PyTorch 2.2 introduces `torch.compile` stability improvements, `torch.nn.functional.scaled_dot_product_attention` (SDPA) as a stable API, and improved FSDP2. Do not pin to an exact version — use minimum version bounds so downstream users can upgrade PyTorch without reinstalling.

**Why 2.2 specifically:** Earlier 2.x versions have known regressions in `torch.compile` graph
capture for custom autograd functions and in FSDP checkpoint sharding. 2.2 resolves these.

### einops>=0.7

`einops` provides `rearrange`, `reduce`, and `repeat` with PyTorch backend support. Used
extensively in encoder and decoder implementations for readability. Version 0.7 adds
`einops.layers.torch` compatibility with `torch.compile`.

### decord>=0.6

Fast video decoding library. Used by the dataset pipeline to load video clips without
decoding the full file. The `VideoReader` class provides random-access frame retrieval.
Optional at import time — guard with `try/except ImportError` in dataset code if video
loading is not always required.

### wandb>=0.16

Weights & Biases experiment tracking. Called only from `BaseTrainer` implementations; the
base class itself does not depend on wandb directly. Version 0.16 introduced the improved
`wandb.init` API and better offline mode support.

### pyyaml>=6.0

YAML 1.2 parsing. Required for config loading. Use `yaml.safe_load()` exclusively — never
use `yaml.load()` without an explicit Loader argument.

### numpy>=1.24

NumPy 1.24 deprecates several legacy dtypes and array creation APIs. Minimum bound ensures
compatibility with PyTorch 2.x NumPy interop (`torch.from_numpy`, `tensor.numpy()`).

---

## Optional JAX Extras

The `jax` extra group enables JAX-based component implementations alongside PyTorch ones.
Both frameworks can coexist in the same Python environment if installed carefully.

```toml
[project.optional-dependencies]
jax = [
    "jax[tpu]>=0.4.20",
    "flax>=0.8",
    "optax>=0.2",
]
```

### JAX/PyTorch Coexistence

The two frameworks can conflict around XLA and CUDA initialization. Follow these rules:

1. **Import guards:** Every file that uses JAX must guard the import:
   ```python
   try:
       import jax
       import jax.numpy as jnp
       HAS_JAX = True
   except ImportError:
       HAS_JAX = False
   ```

2. **Separate extra groups:** Never add JAX to the `dependencies` list. Always keep it in
   the `jax` optional extra. This ensures PyTorch-only installations never pull in JAX.

3. **CUDA device ordering:** JAX by default claims all GPUs. If using both frameworks on
   GPU, set `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `XLA_PYTHON_CLIENT_ALLOCATOR=platform`
   before importing JAX to prevent JAX from pre-allocating GPU memory.

4. **Separate modules:** JAX-based implementations live in `world_model/*/jax_impl.py`
   files that are only imported when `HAS_JAX` is `True`.

### Installing JAX Extra

```bash
# CPU only
pip install -e ".[jax]"

# GPU with CUDA 12
pip install -e ".[jax]"
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

---

## Development Extras

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "ruff>=0.2",
    "mypy>=1.8",
    "types-PyYAML",
]
```

### pytest>=7.4

The test suite uses pytest fixtures, parametrize, and temporary directory helpers. Version
7.4 introduces stable `pytest.ini`-style `pyproject.toml` configuration.

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--tb=short -q"
markers = [
    "slow: marks tests as slow (run with -m slow)",
    "gpu: marks tests requiring GPU",
]
```

### pytest-cov>=4.1

Coverage reporting integrated with pytest. Run with:

```bash
pytest --cov=world_model --cov-report=html tests/
```

### ruff>=0.2

Replaces flake8 + isort + pyupgrade with a single fast linter. Configuration:

```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
ignore = ["E501"]

[tool.ruff.lint.isort]
known-first-party = ["world_model"]
```

### mypy>=1.8

Type checking. Configuration:

```toml
[tool.mypy]
python_version = "3.10"
strict = false
ignore_missing_imports = true
disallow_untyped_defs = true
warn_return_any = true
```

---

## Editable Install

Install the package in editable mode so changes to source files are immediately reflected
without reinstalling:

```bash
# Base install
pip install -e .

# With dev tools
pip install -e ".[dev]"

# With JAX support and dev tools
pip install -e ".[jax,dev]"

# Everything
pip install -e ".[all]"
```

The `-e` flag creates a `.pth` file in `site-packages` pointing to the source directory.
Changes to `.py` files in `world_model/` take effect immediately in the current interpreter.

---

## Version Pinning Strategy

Use **minimum version bounds**, not exact pins, in `pyproject.toml`. Exact pins (`torch==2.2.0`)
cause dependency resolution failures for downstream users who already have a newer version
installed.

Recommended pattern:

```toml
dependencies = [
    "torch>=2.2",        # Minimum for compile stability
    "einops>=0.7",       # Minimum for torch.compile compat
    "numpy>=1.24",       # Minimum to avoid deprecated dtype APIs
]
```

For reproducible training environments, generate a pinned lockfile separately:

```bash
pip freeze > requirements-lock.txt
```

Check `requirements-lock.txt` into a `lockfiles/` directory for reproducibility, but do not
use it as `install_requires` in `pyproject.toml`.

---

## Package Discovery

The `setuptools` package finder is configured to find all packages under `world_model/`:

```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["world_model*"]
```

This discovers: `world_model`, `world_model.encoders`, `world_model.dynamics`,
`world_model.memory`, `world_model.planning`, `world_model.decoders`,
`world_model.training`, `world_model.evaluation`, `world_model.deployment`,
`world_model.core`.

---

## Entry Points

The scaffold generates a CLI entry point for the project generator:

```toml
[project.scripts]
wm-generate = "world_model.cli.generate:main"
```

After installation, users can run:

```bash
wm-generate --output_dir ./my_project --project_name my_world_model
```

---

## py.typed Marker

Include a `py.typed` marker file to indicate the package ships type annotations:

```bash
touch world_model/py.typed
```

Declare it in `package-data`:

```toml
[tool.setuptools.package-data]
world_model = ["py.typed", "configs/*.yaml"]
```

This enables mypy and pyright to use the package's inline type annotations when type-checking
downstream code.
