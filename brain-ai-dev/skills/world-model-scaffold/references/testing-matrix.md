# Testing Matrix — World Model Scaffold

This document specifies the complete test matrix for validating the world model scaffold.
Tests are organized into six phases, each corresponding to a done-when gate. Each phase lists
individual test cases with their expected outcome.

---

## Phase 1: Directory Structure

Verify that the scaffold generator creates all required directories and files.

### Test Cases

| ID | Test | Expected |
|----|------|----------|
| S01 | `world_model/` directory exists | Pass |
| S02 | `world_model/encoders/` exists | Pass |
| S03 | `world_model/dynamics/` exists | Pass |
| S04 | `world_model/memory/` exists | Pass |
| S05 | `world_model/planning/` exists | Pass |
| S06 | `world_model/decoders/` exists | Pass |
| S07 | `world_model/training/` exists | Pass |
| S08 | `world_model/evaluation/` exists | Pass |
| S09 | `world_model/deployment/` exists | Pass |
| S10 | `world_model/core/` exists | Pass |
| S11 | `world_model/configs/` exists | Pass |
| S12 | `world_model/__init__.py` exists and is non-empty | Pass |
| S13 | `world_model/encoders/__init__.py` exists | Pass |
| S14 | `world_model/dynamics/__init__.py` exists | Pass |
| S15 | `world_model/memory/__init__.py` exists | Pass |
| S16 | `world_model/planning/__init__.py` exists | Pass |
| S17 | `world_model/decoders/__init__.py` exists | Pass |
| S18 | `world_model/training/__init__.py` exists | Pass |
| S19 | `world_model/evaluation/__init__.py` exists | Pass |
| S20 | `world_model/deployment/__init__.py` exists | Pass |
| S21 | `world_model/core/__init__.py` exists | Pass |
| S22 | `world_model/encoders/base.py` exists and is non-empty | Pass |
| S23 | `world_model/dynamics/base.py` exists and is non-empty | Pass |
| S24 | `world_model/memory/base.py` exists and is non-empty | Pass |
| S25 | `world_model/planning/base.py` exists and is non-empty | Pass |
| S26 | `world_model/decoders/base.py` exists and is non-empty | Pass |
| S27 | `world_model/training/base.py` exists and is non-empty | Pass |
| S28 | `world_model/evaluation/base.py` exists and is non-empty | Pass |
| S29 | `world_model/deployment/base.py` exists and is non-empty | Pass |
| S30 | `world_model/core/world_model.py` exists and is non-empty | Pass |
| S31 | `world_model/configs/model.yaml` exists and is non-empty | Pass |
| S32 | `world_model/configs/training.yaml` exists and is non-empty | Pass |
| S33 | `world_model/configs/dataset.yaml` exists and is non-empty | Pass |
| S34 | `world_model/configs/hardware.yaml` exists and is non-empty | Pass |
| S35 | `pyproject.toml` exists at project root | Pass |
| S36 | `README.md` exists at project root | Pass |

### Implementation

```python
import os

def check_structure(output_dir: str) -> list[tuple[str, bool]]:
    results = []
    required = [
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
    for rel in required:
        path = os.path.join(output_dir, rel)
        exists = os.path.isfile(path) and os.path.getsize(path) > 0
        results.append((rel, exists))
    return results
```

---

## Phase 2: ABC Contracts

Verify that each ABC class correctly enforces its abstract method contract.

### Test Cases

| ID | Test | Expected |
|----|------|----------|
| A01 | Instantiate `BaseEncoder` directly | `TypeError` |
| A02 | Subclass `BaseEncoder`, omit `forward` | `TypeError` |
| A03 | Subclass `BaseEncoder`, omit `get_embed_dim` | `TypeError` |
| A04 | Subclass `BaseEncoder`, omit `get_output_shape` | `TypeError` |
| A05 | Full `BaseEncoder` subclass instantiates successfully | No error |
| A06 | Full `BaseEncoder` — `get_embed_dim()` returns `int` | `isinstance(result, int)` |
| A07 | Full `BaseEncoder` — `get_output_shape()` returns `tuple` | `isinstance(result, tuple)` |
| A08 | Instantiate `BaseDynamics` directly | `TypeError` |
| A09 | Subclass `BaseDynamics`, omit `step` | `TypeError` |
| A10 | Subclass `BaseDynamics`, omit `imagine` | `TypeError` |
| A11 | Subclass `BaseDynamics`, omit `get_state_dim` | `TypeError` |
| A12 | Full `BaseDynamics` subclass instantiates successfully | No error |
| A13 | Instantiate `BaseMemory` directly | `TypeError` |
| A14 | Subclass `BaseMemory`, omit `read` | `TypeError` |
| A15 | Subclass `BaseMemory`, omit `write` | `TypeError` |
| A16 | Subclass `BaseMemory`, omit `reset` | `TypeError` |
| A17 | Subclass `BaseMemory`, omit `get_capacity` | `TypeError` |
| A18 | Full `BaseMemory` subclass instantiates successfully | No error |
| A19 | Instantiate `BasePlanner` directly | `TypeError` |
| A20 | Subclass `BasePlanner`, omit `plan` | `TypeError` |
| A21 | Full `BasePlanner` subclass instantiates successfully | No error |
| A22 | Instantiate `BaseDecoder` directly | `TypeError` |
| A23 | Subclass `BaseDecoder`, omit `forward` | `TypeError` |
| A24 | Subclass `BaseDecoder`, omit `get_output_shape` | `TypeError` |
| A25 | Subclass `BaseDecoder`, omit `get_input_dim` | `TypeError` |
| A26 | Full `BaseDecoder` subclass instantiates successfully | No error |
| A27 | Instantiate `BaseTrainer` directly | `TypeError` |
| A28 | Subclass `BaseTrainer`, omit `train_step` | `TypeError` |
| A29 | Subclass `BaseTrainer`, omit `validate` | `TypeError` |
| A30 | Subclass `BaseTrainer`, omit `save_checkpoint` | `TypeError` |
| A31 | Subclass `BaseTrainer`, omit `load_checkpoint` | `TypeError` |
| A32 | Full `BaseTrainer` subclass instantiates successfully | No error |
| A33 | Instantiate `BaseEvaluator` directly | `TypeError` |
| A34 | Subclass `BaseEvaluator`, omit `evaluate` | `TypeError` |
| A35 | Subclass `BaseEvaluator`, omit `compute_metrics` | `TypeError` |
| A36 | Full `BaseEvaluator` subclass instantiates successfully | No error |
| A37 | Instantiate `BaseExporter` directly | `TypeError` |
| A38 | Subclass `BaseExporter`, omit `export` | `TypeError` |
| A39 | Subclass `BaseExporter`, omit `validate_export` | `TypeError` |
| A40 | Full `BaseExporter` subclass instantiates successfully | No error |

---

## Phase 3: Composition

Verify that `BaseWorldModel` correctly composes components and validates dimensions.

### Test Cases

| ID | Test | Expected |
|----|------|----------|
| C01 | Construct `BaseWorldModel` with matching dimensions | No error |
| C02 | `encoder.embed_dim != dynamics.state_dim` | `ValueError` |
| C03 | `encoder.embed_dim != decoder.input_dim` | `ValueError` |
| C04 | `memory=None` — `step()` calls `dynamics.step()` directly | No error |
| C05 | `planner=None` — `has_planner` property is `False` | `False` |
| C06 | `memory` provided — `has_memory` property is `True` | `True` |
| C07 | `planner` provided — `has_planner` property is `True` | `True` |
| C08 | `encode(obs)` — returns tensor of shape `(B, embed_dim)` | Shape matches |
| C09 | `decode(latent)` — returns tensor of `output_shape` | Shape matches |
| C10 | `step(state, action)` — returns tensor of shape `(B, state_dim)` | Shape matches |
| C11 | `imagine(state, policy, horizon)` — shape `(B, H, state_dim)` | Shape matches |
| C12 | `forward(obs, action)` — runs full pipeline without error | No error |
| C13 | Hot-swap encoder A for encoder B — both construct without error | No error |
| C14 | Hot-swap dynamics A for dynamics B — both construct without error | No error |
| C15 | `model.parameters()` includes all component parameters | Non-empty iterator |
| C16 | `model.to("cpu")` moves all component tensors | All on CPU |
| C17 | Pass `BaseEncoder` non-instance as encoder | `TypeError` |
| C18 | Pass `BaseDynamics` non-instance as dynamics | `TypeError` |
| C19 | Pass `BaseDecoder` non-instance as decoder | `TypeError` |

### Mock Implementations for Testing

```python
import torch
from torch import Tensor

class MockEncoder(BaseEncoder):
    def __init__(self, embed_dim: int = 64):
        self._embed_dim = embed_dim

    def forward(self, x: Tensor) -> Tensor:
        return torch.zeros(x.shape[0], self._embed_dim)

    def get_embed_dim(self) -> int:
        return self._embed_dim

    def get_output_shape(self):
        return (self._embed_dim,)


class MockDynamics(BaseDynamics):
    def __init__(self, state_dim: int = 64, action_dim: int = 4):
        self._state_dim = state_dim
        self._action_dim = action_dim

    def step(self, state: Tensor, action: Tensor) -> Tensor:
        return torch.zeros_like(state)

    def imagine(self, state: Tensor, policy, horizon: int) -> Tensor:
        B = state.shape[0]
        return torch.zeros(B, horizon, self._state_dim)

    def get_state_dim(self) -> int:
        return self._state_dim


class MockDecoder(BaseDecoder):
    def __init__(self, input_dim: int = 64):
        self._input_dim = input_dim

    def forward(self, latent: Tensor) -> Tensor:
        return torch.zeros(latent.shape[0], 3, 16, 16)

    def get_output_shape(self):
        return (3, 16, 16)

    def get_input_dim(self) -> int:
        return self._input_dim
```

---

## Phase 4: Config Generation

Verify that generated YAML configs are valid and contain all required fields.

### Test Cases

| ID | Test | Expected |
|----|------|----------|
| Y01 | `model.yaml` parses with `yaml.safe_load` without error | No error |
| Y02 | `model.yaml` has key `encoder` | Present |
| Y03 | `model.yaml["encoder"]` has key `type` | Present |
| Y04 | `model.yaml` has key `dynamics` | Present |
| Y05 | `model.yaml` has key `decoder` | Present |
| Y06 | `training.yaml` parses with `yaml.safe_load` | No error |
| Y07 | `training.yaml` has key `optimizer` | Present |
| Y08 | `training.yaml["optimizer"]` has key `lr` | Present |
| Y09 | `training.yaml` has key `training` | Present |
| Y10 | `training.yaml["training"]` has key `epochs` | Present |
| Y11 | `training.yaml["training"]` has key `batch_size` | Present |
| Y12 | `dataset.yaml` parses with `yaml.safe_load` | No error |
| Y13 | `dataset.yaml` has key `data` | Present |
| Y14 | `dataset.yaml` has key `splits` | Present |
| Y15 | `dataset.yaml` has key `sequence` | Present |
| Y16 | `hardware.yaml` parses with `yaml.safe_load` | No error |
| Y17 | `hardware.yaml` has key `device` | Present |
| Y18 | `hardware.yaml` has key `precision` | Present |
| Y19 | All numeric config values are valid Python numbers | All pass |
| Y20 | `encoder.params.embed_dim == dynamics.params.state_dim` | Equal |

---

## Phase 5: Packaging

Verify that the generated `pyproject.toml` is valid and contains correct metadata.

### Test Cases

| ID | Test | Expected |
|----|------|----------|
| P01 | `pyproject.toml` parses with `tomllib` (Python 3.11+) or `tomli` | No error |
| P02 | `[project]` section present | Present |
| P03 | `[project].name` is a non-empty string | Non-empty |
| P04 | `[project].version` matches semver pattern | Valid |
| P05 | `[project].requires-python` is `">=3.10"` | Matches |
| P06 | `[project].dependencies` contains `"torch>=2.2"` (or with specifier) | Present |
| P07 | `[project].dependencies` contains `"einops"` | Present |
| P08 | `[project].dependencies` contains `"pyyaml"` or `"PyYAML"` | Present |
| P09 | `[project.optional-dependencies].jax` contains `"jax"` | Present |
| P10 | `[project.optional-dependencies].dev` contains `"pytest"` | Present |
| P11 | `[project.optional-dependencies].dev` contains `"ruff"` | Present |
| P12 | `[build-system]` section present | Present |
| P13 | `[build-system].requires` is a non-empty list | Non-empty |
| P14 | `[tool.setuptools.packages.find]` or `[tool.hatch.build]` present | Present |

---

## Phase 6: Scaffold Generator

Verify that the `ScaffoldGenerator` class produces a complete, valid scaffold.

### Test Cases

| ID | Test | Expected |
|----|------|----------|
| G01 | `ScaffoldGenerator().generate(tmpdir, "test_project")` completes without error | No error |
| G02 | All Phase 1 structure checks pass on generated output | All True |
| G03 | All Phase 4 config checks pass on generated configs | All True |
| G04 | `validate_scaffold(tmpdir)` returns `True` | True |
| G05 | Every `.py` file in `world_model/` is non-empty | All pass |
| G06 | Every `__init__.py` can be parsed as valid Python | All pass |
| G07 | Every `base.py` can be parsed as valid Python | All pass |
| G08 | `world_model/core/world_model.py` can be parsed as valid Python | Pass |
| G09 | `pyproject.toml` exists and is non-empty | Pass |
| G10 | `README.md` exists and is non-empty | Pass |
| G11 | Second call to `generate()` on same directory overwrites cleanly | No error |
| G12 | `generate()` with `skip_configs=True` — no YAML files written | Config files absent |
| G13 | `generate()` with `skip_readme=True` — no `README.md` written | README absent |

---

## Edge Cases

| ID | Scenario | Expected |
|----|----------|----------|
| E01 | `BaseWorldModel` with `memory=None` and `planner=None` | Constructs fine |
| E02 | Empty YAML string passed to config loader | `yaml.YAMLError` or empty dict |
| E03 | `imagine(state, policy, horizon=0)` | `ValueError` |
| E04 | `plan(state, dynamics, horizon=0)` | `ValueError` |
| E05 | `encoder.embed_dim=0` (zero-dimension) | `ValueError` at construction |
| E06 | Encoder and dynamics both `embed_dim=1` (minimal) | Constructs fine |
| E07 | `memory.read()` on freshly reset memory | Returns zero tensor |
| E08 | `validate_export()` on non-existent path | Returns `False`, no raise |
| E09 | `load_checkpoint()` on non-existent path | `FileNotFoundError` |
| E10 | `ScaffoldGenerator.generate()` with non-existent output_dir | Creates the dir |
| E11 | `write_configs()` with read-only directory | `PermissionError` or `OSError` |
| E12 | `compute_metrics()` with mismatched prediction/target shapes | `ValueError` |

---

## Test Execution

Run all tests with pytest:

```bash
# Run generated tests
pytest tests/test_scaffold.py -v

# Run only structure tests
pytest tests/test_scaffold.py -v -k "structure"

# Run only ABC tests
pytest tests/test_scaffold.py -v -k "abc"

# Run with coverage
pytest tests/test_scaffold.py --cov=world_model --cov-report=html

# Run validation script directly
python scripts/validate_scaffold.py --output_dir ./my_project
```
