# Composition Pattern — World Model Scaffold

This document describes how the `BaseWorldModel` composes the eight abstract components into
a unified forward pass, how dependency injection is used instead of inheritance, and how
factory functions and a registry enable config-driven instantiation.

---

## Core Philosophy: Composition Over Inheritance

A world model is not an encoder, nor a dynamics model, nor a decoder. It *has* each of these.
The `BaseWorldModel` class owns component instances and delegates to them. This is the
**strategy pattern** applied to deep learning modules.

Do not inherit from `BaseEncoder` in `BaseWorldModel`. Do not call encoder code directly
inside the dynamics model. Each component is decoupled — it communicates through the
dimension-declared tensor contracts specified in `abc-contracts.md`.

The benefit: replacing ViT with Mamba as the encoder requires changing exactly one line —
the constructor call. No dynamics code changes. No decoder code changes. No config schema
changes beyond the encoder-specific hyperparameters.

---

## Dependency Injection in BaseWorldModel

### Constructor Signature

```python
class BaseWorldModel(nn.Module):
    def __init__(
        self,
        encoder: BaseEncoder,
        dynamics: BaseDynamics,
        decoder: BaseDecoder,
        memory: Optional[BaseMemory] = None,
        planner: Optional[BasePlanner] = None,
    ) -> None:
```

The constructor takes **instances**, not classes. The caller constructs each component with
its own hyperparameters, then passes the instantiated objects into `BaseWorldModel`. This
separation keeps `BaseWorldModel` ignorant of each component's internal configuration.

### What the Constructor Does

1. **Type-check each required argument.** Raise `TypeError` if `encoder` is not a
   `BaseEncoder`, `dynamics` is not a `BaseDynamics`, or `decoder` is not a `BaseDecoder`.

2. **Dimensional validation.** After type checking, query the dimension-reporting methods:
   ```python
   if encoder.get_embed_dim() != dynamics.get_state_dim():
       raise ValueError(
           f"Encoder embed_dim={encoder.get_embed_dim()} does not match "
           f"dynamics state_dim={dynamics.get_state_dim()}"
       )
   if encoder.get_embed_dim() != decoder.get_input_dim():
       raise ValueError(
           f"Encoder embed_dim={encoder.get_embed_dim()} does not match "
           f"decoder input_dim={decoder.get_input_dim()}"
       )
   ```
   This is the **fail-fast** contract: dimension mismatches are caught at object creation
   time, not buried in a training loop stack trace.

3. **Register components as `nn.Module` children** (for state dict, `.parameters()`, and
   device movement):
   ```python
   self.encoder = encoder
   self.dynamics = dynamics
   self.decoder = decoder
   if memory is not None:
       self.memory = memory
   if planner is not None:
       self.planner = planner
   ```

4. **Store optional flags** for use in the forward path:
   ```python
   self._has_memory = memory is not None
   self._has_planner = planner is not None
   ```

---

## Forward Path

The full forward pass follows this pipeline:

```
obs (raw)
  |
  v
encode(obs)         --> latent state  [encoder]
  |
  +--[if memory]---> memory.read(latent)  --> context
  |                  then write(latent, latent)
  |
  v
dynamics.step(state, action)  --> next_state
  |
  +--[if planner]---> plan(next_state, dynamics, horizon)  --> action_sequence
  |
  v
decode(next_state)  --> reconstructed obs  [decoder]
```

Each stage is a delegate call. The world model adds no transformation logic — only routing.

### Method Breakdown

```python
def encode(self, obs: Tensor) -> Tensor:
    return self.encoder.forward(obs)

def step(self, state: Tensor, action: Tensor) -> Tensor:
    if self._has_memory:
        context = self.memory.read(state)
        augmented = state + context  # or concatenate; implementation decides
        next_state = self.dynamics.step(augmented, action)
        self.memory.write(state, next_state)
    else:
        next_state = self.dynamics.step(state, action)
    return next_state

def imagine(self, state: Tensor, policy: Callable, horizon: int) -> Tensor:
    return self.dynamics.imagine(state, policy, horizon)

def decode(self, latent: Tensor) -> Tensor:
    return self.decoder.forward(latent)

def forward(self, obs: Tensor, action: Tensor) -> Tensor:
    latent = self.encode(obs)
    next_state = self.step(latent, action)
    return self.decode(next_state)
```

The `memory.read` + augmentation step is the one place `BaseWorldModel` applies a non-trivial
operation. Even this can be overridden in a subclass that knows more about its memory fusion
strategy.

---

## Optional Components

Memory and planner are `Optional`. The model operates correctly without them.

```python
# Minimal configuration — no memory, no planner
model = BaseWorldModel(
    encoder=MyEncoder(embed_dim=512),
    dynamics=MyDynamics(state_dim=512),
    decoder=MyDecoder(input_dim=512),
)

# Full configuration
model = BaseWorldModel(
    encoder=MyEncoder(embed_dim=512),
    dynamics=MyDynamics(state_dim=512),
    decoder=MyDecoder(input_dim=512),
    memory=EpisodicMemory(capacity=1000),
    planner=CEMPlanner(num_samples=1000),
)
```

When `memory` is `None`, the `step()` method bypasses read/write and calls `dynamics.step()`
directly. When `planner` is `None`, the model has no planning interface at the world-model
level (the planner can still be used externally by passing the dynamics model directly).

---

## Hot-Swap Pattern

### Example: ViT Encoder to Mamba Encoder

```python
# Original model with ViT encoder
model_vit = BaseWorldModel(
    encoder=ViTEncoder(embed_dim=512, image_size=64, patch_size=8),
    dynamics=RSSMDynamics(state_dim=512, action_dim=6),
    decoder=ConvDecoder(input_dim=512, output_shape=(3, 64, 64)),
)

# Swap ViT for Mamba — zero changes to dynamics or decoder
model_mamba = BaseWorldModel(
    encoder=MambaEncoder(embed_dim=512, d_state=16),
    dynamics=RSSMDynamics(state_dim=512, action_dim=6),
    decoder=ConvDecoder(input_dim=512, output_shape=(3, 64, 64)),
)
```

Both models have the same dynamics and decoder instances (or equivalent configs). The only
change is the encoder argument. Dimensional compatibility is validated automatically.

### Example: RSSM Dynamics to Transformer Dynamics

```python
# Original
model_rssm = BaseWorldModel(
    encoder=ViTEncoder(embed_dim=256),
    dynamics=RSSMDynamics(state_dim=256, action_dim=4),
    decoder=ConvDecoder(input_dim=256),
)

# Swap dynamics model
model_transformer = BaseWorldModel(
    encoder=ViTEncoder(embed_dim=256),
    dynamics=TransformerDynamics(state_dim=256, action_dim=4, n_heads=8),
    decoder=ConvDecoder(input_dim=256),
)
```

The encoder and decoder are unchanged. Only the dynamics implementation changes.

---

## Registry Pattern

The registry maps string names to classes, enabling config-driven instantiation without
importing concrete implementations at the top of config files.

### Registry Implementation

```python
_ENCODER_REGISTRY: Dict[str, Type[BaseEncoder]] = {}
_DYNAMICS_REGISTRY: Dict[str, Type[BaseDynamics]] = {}
_DECODER_REGISTRY: Dict[str, Type[BaseDecoder]] = {}
_MEMORY_REGISTRY: Dict[str, Type[BaseMemory]] = {}
_PLANNER_REGISTRY: Dict[str, Type[BasePlanner]] = {}

def register_encoder(name: str):
    def decorator(cls: Type[BaseEncoder]) -> Type[BaseEncoder]:
        _ENCODER_REGISTRY[name] = cls
        return cls
    return decorator
```

Concrete implementations register themselves:

```python
@register_encoder("vit")
class ViTEncoder(BaseEncoder):
    ...

@register_encoder("mamba")
class MambaEncoder(BaseEncoder):
    ...
```

### Factory Function

```python
def create_world_model(config: Dict[str, Any]) -> BaseWorldModel:
    encoder_cls = _ENCODER_REGISTRY[config["model"]["encoder"]["type"]]
    encoder = encoder_cls(**config["model"]["encoder"]["params"])

    dynamics_cls = _DYNAMICS_REGISTRY[config["model"]["dynamics"]["type"]]
    dynamics = dynamics_cls(**config["model"]["dynamics"]["params"])

    decoder_cls = _DECODER_REGISTRY[config["model"]["decoder"]["type"]]
    decoder = decoder_cls(**config["model"]["decoder"]["params"])

    memory = None
    if "memory" in config["model"]:
        memory_cls = _MEMORY_REGISTRY[config["model"]["memory"]["type"]]
        memory = memory_cls(**config["model"]["memory"]["params"])

    planner = None
    if "planner" in config["model"]:
        planner_cls = _PLANNER_REGISTRY[config["model"]["planner"]["type"]]
        planner = planner_cls(**config["model"]["planner"]["params"])

    return BaseWorldModel(
        encoder=encoder,
        dynamics=dynamics,
        decoder=decoder,
        memory=memory,
        planner=planner,
    )
```

### Usage

```python
import yaml

with open("configs/model.yaml") as f:
    config = yaml.safe_load(f)

model = create_world_model(config)
```

Swapping the encoder in config:

```yaml
# Before
model:
  encoder:
    type: "vit"
    params: {embed_dim: 512, image_size: 64, patch_size: 8}

# After
model:
  encoder:
    type: "mamba"
    params: {embed_dim: 512, d_state: 16}
```

The factory handles instantiation; `BaseWorldModel` handles dimensional validation.

---

## Hydra Integration

When using Hydra, the factory pattern maps naturally to structured configs:

```python
from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass
class EncoderConfig:
    type: str = "vit"
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
```

Hydra CLI overrides work directly:

```bash
python train.py model.encoder.type=mamba model.encoder.params.d_state=32
```

---

## Property Accessors

`BaseWorldModel` exposes read-only properties for inspection:

```python
@property
def embed_dim(self) -> int:
    return self.encoder.get_embed_dim()

@property
def state_dim(self) -> int:
    return self.dynamics.get_state_dim()

@property
def has_memory(self) -> bool:
    return self._has_memory

@property
def has_planner(self) -> bool:
    return self._has_planner
```

These allow external code to query the model's configuration without inspecting internals.

---

## Summary

| Concept | Implementation |
|---------|---------------|
| Composition root | `BaseWorldModel(nn.Module)` |
| Injection point | Constructor arguments (instances) |
| Dimensional check | `encoder.get_embed_dim() == dynamics.get_state_dim()` at `__init__` |
| Optional components | `memory=None`, `planner=None` |
| Hot-swap | Pass different instance to same constructor |
| Config-driven | Registry + factory function `create_world_model(config)` |
| Hydra support | Structured config dataclasses mapping to registry keys |
