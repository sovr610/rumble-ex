---
name: World Model Architecture Scaffold
description: >
  This skill should be used when the user asks to "scaffold a world model",
  "create world model project", "generate world model directory structure",
  "set up world model codebase", "create base classes for world model",
  "world model ABC hierarchy", "hot-swappable encoder/dynamics/decoder",
  "world model dependency injection", "generate world model configs",
  "Hydra config for world model", "world model pyproject.toml",
  "modular world model architecture", "replace encoder without changing dynamics",
  "world model base class contracts", "BaseEncoder interface",
  "BaseDynamics interface", "BaseWorldModel composition",
  "world model deployment structure", "world model evaluation harness scaffold",
  or needs guidance on structuring a production-grade world model codebase
  with swappable components, config management, and packaging.
version: 0.1.0
---

# World Model Architecture Scaffold

## Overview

Generate a complete, production-grade project structure for a world model codebase. The scaffold establishes ABC-enforced interface contracts for all major components (encoders, dynamics, memory, planning, decoders), composes them via dependency injection in a `BaseWorldModel`, and provides config management, packaging, and documentation. The design principle: **any encoder, dynamics model, or decoder is hot-swappable without touching other components.**

## Directory Tree

```
world_model/
├── encoders/
│   ├── __init__.py
│   └── base.py              # BaseEncoder ABC
├── dynamics/
│   ├── __init__.py
│   └── base.py              # BaseDynamics ABC
├── memory/
│   ├── __init__.py
│   └── base.py              # BaseMemory ABC
├── planning/
│   ├── __init__.py
│   └── base.py              # BasePlanner ABC
├── decoders/
│   ├── __init__.py
│   └── base.py              # BaseDecoder ABC
├── training/
│   ├── __init__.py
│   └── base.py              # BaseTrainer ABC
├── evaluation/
│   ├── __init__.py
│   └── base.py              # BaseEvaluator ABC
├── deployment/
│   ├── __init__.py
│   └── base.py              # BaseExporter ABC
├── core/
│   ├── __init__.py
│   └── world_model.py       # BaseWorldModel (composition root)
├── configs/
│   ├── model.yaml            # Architecture config
│   ├── training.yaml         # Training hyperparameters
│   ├── dataset.yaml          # Dataset paths and settings
│   └── hardware.yaml         # Device/distributed settings
├── pyproject.toml
└── README.md
```

## Public Contract

### BaseEncoder

```python
class BaseEncoder(ABC):
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor: ...
    @abstractmethod
    def get_embed_dim(self) -> int: ...
    @abstractmethod
    def get_output_shape(self) -> Tuple[int, ...]: ...
```

### BaseDynamics

```python
class BaseDynamics(ABC):
    @abstractmethod
    def step(self, state: Tensor, action: Tensor) -> Tensor: ...
    @abstractmethod
    def imagine(self, state: Tensor, policy: Callable,
                horizon: int) -> Tensor: ...
```

### BaseMemory

```python
class BaseMemory(ABC):
    @abstractmethod
    def read(self, query: Tensor) -> Tensor: ...
    @abstractmethod
    def write(self, key: Tensor, value: Tensor) -> None: ...
    @abstractmethod
    def reset(self) -> None: ...
```

### BasePlanner

```python
class BasePlanner(ABC):
    @abstractmethod
    def plan(self, state: Tensor, dynamics: "BaseDynamics",
             horizon: int) -> Tensor: ...
```

### BaseDecoder

```python
class BaseDecoder(ABC):
    @abstractmethod
    def forward(self, latent: Tensor) -> Tensor: ...
    @abstractmethod
    def get_output_shape(self) -> Tuple[int, ...]: ...
```

### BaseWorldModel (Composition Root)

```python
class BaseWorldModel(nn.Module):
    def __init__(self, encoder: BaseEncoder, dynamics: BaseDynamics,
                 decoder: BaseDecoder,
                 memory: Optional[BaseMemory] = None,
                 planner: Optional[BasePlanner] = None): ...
    def encode(self, obs: Tensor) -> Tensor: ...
    def step(self, state: Tensor, action: Tensor) -> Tensor: ...
    def imagine(self, state: Tensor, policy: Callable, horizon: int) -> Tensor: ...
    def decode(self, latent: Tensor) -> Tensor: ...
```

Components are injected at construction. Replacing a ViT encoder with a Mamba encoder requires zero changes to dynamics, decoder, or the world model itself — just pass a different encoder instance.

## Key Concepts

### Dependency Injection Pattern

The `BaseWorldModel` does **not** inherit from encoders or dynamics. It receives them as constructor arguments. This is the strategy pattern: the world model delegates to its components without knowing their concrete types.

```python
# Hot-swap example: ViT → Mamba
model_a = BaseWorldModel(encoder=ViTEncoder(...), dynamics=RSSM(...), decoder=ConvDecoder(...))
model_b = BaseWorldModel(encoder=MambaEncoder(...), dynamics=RSSM(...), decoder=ConvDecoder(...))
```

### Interface Contracts via ABCs

Every base class uses `abc.ABC` + `@abstractmethod`. Subclasses that forget to implement a method raise `TypeError` at instantiation, not at runtime during training. This catches wiring errors early.

### Dimensional Consistency

Encoders declare `get_embed_dim()` and `get_output_shape()`. Dynamics and decoders consume these dimensions. The `BaseWorldModel.__init__` should validate dimensional compatibility at construction time — fail fast if encoder output dim does not match dynamics input dim.

### Config Management

Use flat YAML files (Hydra-compatible but not Hydra-required):

| File | Contents |
|------|----------|
| `model.yaml` | encoder type/params, dynamics type/params, decoder type/params, memory, planner |
| `training.yaml` | lr, batch_size, epochs, optimizer, scheduler, loss weights, grad_clip |
| `dataset.yaml` | data_root, splits, preprocessing, sequence_length, frame_skip |
| `hardware.yaml` | device, dtype, distributed strategy, num_workers, compile |

### Packaging

`pyproject.toml` with pinned dependencies:
- `torch>=2.2`
- `jax[tpu]`, `flax`, `optax` (for JAX-based components)
- `einops`, `decord` (video), `wandb` (logging)
- Development extras: `pytest`, `ruff`, `mypy`

### Training / Evaluation / Deployment ABCs

These follow the same pattern — abstract interfaces that concrete implementations fill:
- `BaseTrainer`: `train_step()`, `validate()`, `save_checkpoint()`
- `BaseEvaluator`: `evaluate()`, `compute_metrics()`
- `BaseExporter`: `export()`, `validate_export()`

## Configuration Surface

The scaffold generates config files, not a config dataclass. The YAML files serve as the single source of truth, loaded by whatever config system the project adopts.

## Done-When Gates

1. **Directory Structure Complete** — All directories exist with `__init__.py` and `base.py` files. `configs/`, `pyproject.toml`, and `README.md` are present.
2. **ABC Contracts Enforced** — Instantiating a subclass that omits any abstract method raises `TypeError`. Dimensional mismatch at `BaseWorldModel` construction raises `ValueError`.
3. **Hot-Swap Works** — Two different encoder implementations can be injected into the same `BaseWorldModel` without code changes to dynamics or decoder.

## Resources

### Reference Files
- **`references/abc-contracts.md`** — Detailed ABC interface specs for all 8 base classes, method signatures, return types, dimensional rules, error handling
- **`references/composition-pattern.md`** — Dependency injection in BaseWorldModel, dimensional validation, factory functions, registry pattern for config-driven instantiation
- **`references/config-management.md`** — YAML structure, Hydra integration notes, config merging, environment overrides, config validation
- **`references/packaging-deps.md`** — pyproject.toml structure, dependency pinning strategy, optional extras, JAX/PyTorch coexistence, editable install
- **`references/testing-matrix.md`** — Test scenarios for scaffold validation

### Asset Files
- **`assets/base_classes_template.py`** — All 8 ABC base classes with full method signatures, docstrings, and type hints
- **`assets/world_model_template.py`** — BaseWorldModel composition root with dependency injection, dimensional validation, forward methods
- **`assets/config_templates.py`** — YAML content generators for model/training/dataset/hardware configs
- **`assets/pyproject_template.py`** — pyproject.toml generator with all dependencies
- **`assets/scaffold_generator.py`** — Main scaffold generator: creates full directory tree, writes all files, validates structure

### Scripts
- **`scripts/validate_scaffold.py`** — Validates done-when gates
- **`scripts/gen_scaffold_tests.py`** — Generates pytest test cases for ABC enforcement, composition, and dimensional checks
- **`scripts/generate_project.py`** — CLI entrypoint: `python generate_project.py --output_dir ./my_world_model`
