# ABC Interface Contracts — World Model Scaffold

This document specifies the complete interface contracts for all eight abstract base classes
in the world model scaffold. Every concrete component must satisfy its ABC contract in full.
Failure to implement any abstract method raises `TypeError` at instantiation time, not during
training — this is the deliberate design choice that catches wiring errors as early as possible.

---

## Design Principles

Abstract base classes in the scaffold follow three invariants:

1. **Fail at instantiation, not at training.** Python's `abc.ABC` + `@abstractmethod` ensures
   that a concrete subclass missing any required method cannot be instantiated. The error surfaces
   at `MyEncoder()` time, not 10 hours into a training run.

2. **Declare dimensions explicitly.** Every class that produces or consumes tensors of a specific
   shape declares a method that returns that shape at runtime. Shapes are not encoded only in
   docstrings — they are queryable from live objects.

3. **Stateful vs stateless contract.** Stateless classes (encoders, decoders) carry no mutable
   runtime state beyond learnable parameters. Stateful classes (memory) expose a `reset()` method
   so callers can restore the class to a clean initial state without reinstantiating it.

---

## 1. BaseEncoder

**Module:** `world_model/encoders/base.py`

**Role:** Maps raw observations (images, point clouds, sensor readings) to a fixed-dimensional
latent vector. The encoder is stateless — the same call with the same input always produces the
same output (given the same parameters). It does not maintain sequence history.

### Class Signature

```python
from abc import ABC, abstractmethod
from typing import Tuple
import torch
from torch import Tensor

class BaseEncoder(ABC):
    ...
```

### Abstract Methods

#### `forward(x: Tensor) -> Tensor`

Transform a batch of raw observations into a batch of latent embeddings.

- **Parameters:**
  - `x: Tensor` — shape `(B, *)` where `B` is batch size and `*` is the observation shape
    (e.g., `(B, C, H, W)` for images, `(B, N, 3)` for point clouds).
- **Returns:**
  - `Tensor` — shape `(B, embed_dim)`. The last dimension must equal `self.get_embed_dim()`.
    A `RuntimeError` may be raised if the input shape does not match the encoder's expected
    input format.
- **Contract:** The output dimension is deterministic and declared by `get_embed_dim()`.
  Callers may rely on this without inspecting the output tensor.

#### `get_embed_dim() -> int`

Return the integer dimension of the embedding this encoder produces.

- **Returns:** `int` — strictly positive, e.g., 256, 512, 1024.
- **Contract:** This value must be consistent with the last dimension of tensors returned by
  `forward()`. It must be a compile-time constant — i.e., it must not change between calls.

#### `get_output_shape() -> Tuple[int, ...]`

Return the full shape of one output embedding (excluding batch dimension).

- **Returns:** `Tuple[int, ...]` — e.g., `(512,)` for a flat embedding, `(16, 16, 256)` for a
  spatial feature map. For flat encoders this is simply `(embed_dim,)`.
- **Contract:** `get_output_shape()[-1]` must equal `get_embed_dim()` for flat encoders.
  For spatial encoders the caller is responsible for interpreting the spatial structure.

### Error Handling

- Raise `ValueError` in `forward()` if the input tensor has the wrong number of dimensions.
- Do not silently reshape or broadcast inputs — let the caller fix the shape mismatch.

### Thread Safety

BaseEncoder subclasses are stateless with respect to input processing. They are safe for
concurrent inference in multiple threads provided PyTorch's GIL semantics are respected.
Learnable parameters are shared across threads; gradient accumulation is the caller's
responsibility.

---

## 2. BaseDynamics

**Module:** `world_model/dynamics/base.py`

**Role:** Model state transitions. Given a latent state and an action, produce the next latent
state. Optionally roll out imagined trajectories using a policy function.

### Class Signature

```python
from abc import ABC, abstractmethod
from typing import Callable, Tuple
import torch
from torch import Tensor

class BaseDynamics(ABC):
    ...
```

### Abstract Methods

#### `step(state: Tensor, action: Tensor) -> Tensor`

Predict the next latent state given the current state and action.

- **Parameters:**
  - `state: Tensor` — shape `(B, state_dim)`. The current latent state.
  - `action: Tensor` — shape `(B, action_dim)`. The action taken in the current state.
- **Returns:**
  - `Tensor` — shape `(B, state_dim)`. The predicted next latent state.
- **Contract:** The output state dimension must equal the input state dimension.
  `step()` must be differentiable with respect to both `state` and `action`.

#### `imagine(state: Tensor, policy: Callable[[Tensor], Tensor], horizon: int) -> Tensor`

Roll out an imagined trajectory by repeatedly applying `step()`.

- **Parameters:**
  - `state: Tensor` — shape `(B, state_dim)`. Initial latent state.
  - `policy: Callable[[Tensor], Tensor]` — maps `(B, state_dim)` to `(B, action_dim)`.
    The policy is called once per step and must be differentiable if gradient-based planning
    is required.
  - `horizon: int` — number of steps to imagine. Must be >= 1.
- **Returns:**
  - `Tensor` — shape `(B, horizon, state_dim)`. The sequence of imagined states,
    excluding the initial state.
- **Contract:** The returned tensor at index `t` equals `step(states[t-1], policy(states[t-1]))`.
  The first imagined state is `step(state, policy(state))`.

#### `get_state_dim() -> int`

Return the integer dimension of the latent state this dynamics model operates on.

- **Returns:** `int` — strictly positive.
- **Contract:** Must equal `get_embed_dim()` of the encoder paired with this dynamics model
  in a `BaseWorldModel`. The `BaseWorldModel.__init__` validates this at construction time.

### Error Handling

- Raise `ValueError` in `step()` if `state` last dimension does not equal `get_state_dim()`.
- Raise `ValueError` in `imagine()` if `horizon < 1`.
- Raise `TypeError` if `policy` is not callable.

### Thread Safety

BaseDynamics subclasses may maintain recurrent hidden state (e.g., LSTM/GRU cells in RSSM).
If hidden state is stored as instance attributes, concurrent use from multiple threads requires
external synchronization. Document whether a concrete implementation is thread-safe.

---

## 3. BaseMemory

**Module:** `world_model/memory/base.py`

**Role:** Episodic or working memory that augments the dynamics model with information from
past steps. Memory is explicitly stateful: it accumulates writes and must be reset between
episodes. This is the only component in the scaffold that is stateful outside of learnable
parameters.

### Class Signature

```python
from abc import ABC, abstractmethod
from typing import Optional
import torch
from torch import Tensor

class BaseMemory(ABC):
    ...
```

### Abstract Methods

#### `read(query: Tensor) -> Tensor`

Retrieve memory contents relevant to a query vector.

- **Parameters:**
  - `query: Tensor` — shape `(B, query_dim)`. The query used to address memory.
- **Returns:**
  - `Tensor` — shape `(B, value_dim)`. Retrieved memory content. `value_dim` may equal
    `query_dim` or differ depending on the memory architecture.
- **Contract:** If memory is empty (freshly reset or never written), `read()` must return a
  zero tensor of shape `(B, value_dim)` rather than raising an error.

#### `write(key: Tensor, value: Tensor) -> None`

Store a key-value pair in memory.

- **Parameters:**
  - `key: Tensor` — shape `(B, key_dim)`. The addressing key.
  - `value: Tensor` — shape `(B, value_dim)`. The content to store.
- **Returns:** `None`
- **Contract:** After `write(key, value)`, a subsequent `read(key)` should return a value
  close to `value` (the exact similarity depends on the memory's addressing mechanism).

#### `reset() -> None`

Clear all stored memory, restoring the module to its initial empty state.

- **Returns:** `None`
- **Contract:** After `reset()`, `read(query)` must return zeros. This must not reinitialize
  learnable parameters — only runtime state (slot tensors, pointer counters, attention masks).

#### `get_capacity() -> int`

Return the maximum number of key-value pairs the memory can hold.

- **Returns:** `int` — strictly positive or `-1` to indicate unlimited capacity (e.g., a
  cache with dynamic growth).

### Error Handling

- Raise `ValueError` in `write()` if `key` and `value` have inconsistent batch sizes.
- Do not raise when reading from empty memory — return zeros instead.

### Thread Safety

BaseMemory is stateful. Concurrent read/write access from multiple threads requires external
locking. Each episode worker should hold its own memory instance, not share one.

---

## 4. BasePlanner

**Module:** `world_model/planning/base.py`

**Role:** Given the current latent state and a dynamics model, compute an action sequence or
policy that optimizes a reward signal over a planning horizon. Planners use the dynamics model
to simulate future states without interacting with the real environment.

### Class Signature

```python
from abc import ABC, abstractmethod
import torch
from torch import Tensor

class BasePlanner(ABC):
    ...
```

### Abstract Methods

#### `plan(state: Tensor, dynamics: "BaseDynamics", horizon: int) -> Tensor`

Compute an optimized action sequence given the current state.

- **Parameters:**
  - `state: Tensor` — shape `(B, state_dim)`. Current latent state.
  - `dynamics: BaseDynamics` — the dynamics model to simulate with. Must satisfy the
    `BaseDynamics` interface. The planner uses `dynamics.step()` and/or `dynamics.imagine()`
    internally.
  - `horizon: int` — planning horizon in steps. Must be >= 1.
- **Returns:**
  - `Tensor` — shape `(B, horizon, action_dim)`. The planned action sequence.
    The first action `[:, 0, :]` is the action to take immediately.
- **Contract:** The returned actions are computed purely in latent space — no real environment
  interaction occurs inside `plan()`. The planner must not modify the dynamics model's
  learnable parameters.

### Error Handling

- Raise `ValueError` if `horizon < 1`.
- Raise `TypeError` if `dynamics` is not an instance of `BaseDynamics`.

### Thread Safety

Planners are typically stateless between calls (CEM, MPPI recompute from scratch each step).
If a planner caches proposals or gradients across calls, document this explicitly.

---

## 5. BaseDecoder

**Module:** `world_model/decoders/base.py`

**Role:** Map a latent vector back to observation space. The decoder is the inverse of the
encoder. Like the encoder, it is stateless.

### Class Signature

```python
from abc import ABC, abstractmethod
from typing import Tuple
import torch
from torch import Tensor

class BaseDecoder(ABC):
    ...
```

### Abstract Methods

#### `forward(latent: Tensor) -> Tensor`

Decode a batch of latent vectors to reconstructed observations.

- **Parameters:**
  - `latent: Tensor` — shape `(B, latent_dim)`. Must equal the value returned by
    `get_input_dim()`.
- **Returns:**
  - `Tensor` — shape `(B, *output_shape)` where `output_shape = get_output_shape()`.
- **Contract:** The output shape is deterministic and declared by `get_output_shape()`.

#### `get_output_shape() -> Tuple[int, ...]`

Return the shape of one decoded observation, excluding batch dimension.

- **Returns:** `Tuple[int, ...]` — e.g., `(3, 64, 64)` for an RGB image.

#### `get_input_dim() -> int`

Return the expected latent dimension this decoder accepts.

- **Returns:** `int` — must equal `encoder.get_embed_dim()` for the paired encoder.
  The `BaseWorldModel` validates this at construction.

### Error Handling

- Raise `ValueError` in `forward()` if the input tensor last dimension does not equal
  `get_input_dim()`.

### Thread Safety

Stateless — same thread-safety properties as BaseEncoder.

---

## 6. BaseTrainer

**Module:** `world_model/training/base.py`

**Role:** Encapsulates one full training run's logic — a single gradient update step, a
validation loop, and checkpoint management. Concrete trainers implement loss functions,
optimizer schedules, and mixed-precision policies.

### Class Signature

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch
from torch import Tensor
from torch.utils.data import DataLoader

class BaseTrainer(ABC):
    ...
```

### Abstract Methods

#### `train_step(batch: Any) -> Dict[str, float]`

Perform one forward pass, compute loss, run backward pass, and update parameters.

- **Parameters:**
  - `batch: Any` — a single batch from a DataLoader. Typically a dict or tuple of tensors.
- **Returns:**
  - `Dict[str, float]` — scalar metric values for logging. Must include at minimum the key
    `"loss"`. Additional keys (e.g., `"recon_loss"`, `"kl_loss"`) are encouraged.
- **Contract:** This method mutates model parameters. It must be called only within a
  training context (gradients enabled). The caller is responsible for calling
  `optimizer.zero_grad()` if not handled internally.

#### `validate(dataloader: DataLoader) -> Dict[str, float]`

Compute validation metrics over a full dataset split without updating parameters.

- **Parameters:**
  - `dataloader: DataLoader` — yields batches in the same format as `train_step`.
- **Returns:**
  - `Dict[str, float]` — aggregated metrics over the full validation set.
- **Contract:** This method must not update learnable parameters. It must disable gradient
  computation internally (use `torch.no_grad()` or `torch.inference_mode()`).
  Set inference mode by calling `module.train(False)` rather than `.eval()`.

#### `save_checkpoint(path: str) -> None`

Persist the model and optimizer state to disk.

- **Parameters:**
  - `path: str` — file path to write the checkpoint. The directory must exist.
- **Returns:** `None`
- **Contract:** The checkpoint must be loadable by `load_checkpoint()`. The format should be
  a `torch.save()` dict containing at least `model_state_dict` and `optimizer_state_dict`.

#### `load_checkpoint(path: str) -> None`

Load model and optimizer state from a checkpoint file.

- **Parameters:**
  - `path: str` — path to a checkpoint written by `save_checkpoint()`.
- **Returns:** `None`
- **Contract:** After this call, the model and optimizer states match those at the time of
  the checkpoint. Raises `FileNotFoundError` if `path` does not exist.

### Error Handling

- `train_step`: raise `RuntimeError` if called in `torch.no_grad()` context.
- `validate`: raise `FileNotFoundError` if DataLoader's dataset source is unreachable.
- `save_checkpoint`: raise `OSError` if the directory does not exist.
- `load_checkpoint`: raise `FileNotFoundError` if the path does not exist.

---

## 7. BaseEvaluator

**Module:** `world_model/evaluation/base.py`

**Role:** Structured evaluation harness. Separates evaluation logic from training. Given a
trained model and a test dataloader, produce a dictionary of evaluation metrics.

### Class Signature

```python
from abc import ABC, abstractmethod
from typing import Any, Dict
import torch
from torch import Tensor
from torch.utils.data import DataLoader

class BaseEvaluator(ABC):
    ...
```

### Abstract Methods

#### `evaluate(model: Any, dataloader: DataLoader) -> Dict[str, float]`

Run a full evaluation pass of `model` over `dataloader`.

- **Parameters:**
  - `model: Any` — a trained world model (typically a `BaseWorldModel` subclass or `nn.Module`).
  - `dataloader: DataLoader` — test or validation dataloader.
- **Returns:**
  - `Dict[str, float]` — evaluation metrics, e.g., `{"fvd": 42.3, "psnr": 28.1, "ssim": 0.92}`.
- **Contract:** This method must not update model parameters. Use `torch.no_grad()` internally.
  Set inference mode by calling `module.train(False)` on the model.

#### `compute_metrics(predictions: Tensor, targets: Tensor) -> Dict[str, float]`

Compute metric values from a batch of predictions and targets.

- **Parameters:**
  - `predictions: Tensor` — predicted outputs, shape `(B, *)`.
  - `targets: Tensor` — ground truth outputs, shape `(B, *)` matching predictions.
- **Returns:**
  - `Dict[str, float]` — per-metric scalar values.
- **Contract:** This is a pure function — no side effects, no parameter updates.
  Raises `ValueError` if `predictions.shape != targets.shape`.

### Error Handling

- Raise `ValueError` in `compute_metrics()` if prediction and target shapes do not match.
- `evaluate()` should propagate all DataLoader errors without catching them silently.

---

## 8. BaseExporter

**Module:** `world_model/deployment/base.py`

**Role:** Export a trained model to a deployment format (TorchScript, ONNX, TensorRT, etc.)
and validate the exported artifact.

### Class Signature

```python
from abc import ABC, abstractmethod
import torch
from torch import Tensor

class BaseExporter(ABC):
    ...
```

### Abstract Methods

#### `export(model: Any, path: str, format: str) -> None`

Serialize the model to a deployment artifact.

- **Parameters:**
  - `model: Any` — trained model to export.
  - `path: str` — output file path. Parent directory must exist.
  - `format: str` — export format identifier. Supported values are implementation-defined;
    typical values are `"torchscript"`, `"onnx"`, `"tensorrt"`.
- **Returns:** `None`
- **Contract:** After this call, a file exists at `path`. The file is loadable by the
  corresponding loader for the specified format. Raises `ValueError` for unsupported formats.

#### `validate_export(path: str) -> bool`

Verify that the exported artifact at `path` is loadable and produces correct outputs.

- **Parameters:**
  - `path: str` — path to an artifact created by `export()`.
- **Returns:**
  - `bool` — `True` if the artifact loads and passes a basic forward-pass sanity check,
    `False` otherwise. Never raises — validation failures are reported as `False`.
- **Contract:** This method must not modify the file at `path`. Validation is read-only.

### Error Handling

- `export()`: raise `ValueError` for unsupported format strings.
- `export()`: raise `OSError` if the output directory does not exist.
- `validate_export()`: return `False` (never raise) for corrupt or incompatible artifacts.

---

## Dimensional Consistency Rules

The following rules govern dimension compatibility across components. The `BaseWorldModel`
enforces these rules at construction time by calling dimension-query methods.

| Rule | Expression | Error if violated |
|------|-----------|-------------------|
| Encoder to Dynamics | `encoder.get_embed_dim() == dynamics.get_state_dim()` | `ValueError` |
| Encoder to Decoder | `encoder.get_embed_dim() == decoder.get_input_dim()` | `ValueError` |
| Dynamics output | `dynamics.step()` output dim == `dynamics.get_state_dim()` | `RuntimeError` at call time |
| Memory query | `memory.read(query)` accepts any `query_dim`; implementation defines mapping | — |

Any violation of the first two rules is detected at `BaseWorldModel.__init__()` time, before
any data flows through the model.

---

## Summary Table

| Class | Stateful | Key Methods | Dimension Methods |
|-------|----------|-------------|-------------------|
| BaseEncoder | No | `forward`, `get_embed_dim`, `get_output_shape` | `get_embed_dim` |
| BaseDynamics | Maybe | `step`, `imagine`, `get_state_dim` | `get_state_dim` |
| BaseMemory | Yes | `read`, `write`, `reset`, `get_capacity` | `get_capacity` |
| BasePlanner | No | `plan` | — |
| BaseDecoder | No | `forward`, `get_output_shape`, `get_input_dim` | `get_input_dim` |
| BaseTrainer | Yes | `train_step`, `validate`, `save_checkpoint`, `load_checkpoint` | — |
| BaseEvaluator | No | `evaluate`, `compute_metrics` | — |
| BaseExporter | No | `export`, `validate_export` | — |
