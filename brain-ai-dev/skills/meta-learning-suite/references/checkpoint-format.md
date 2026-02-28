# Meta-Learning Checkpoint Format

## Overview

A meta-learning checkpoint must contain more than model weights. In standard supervised learning, saving `model.state_dict()` is sufficient to resume training or run inference -- the model's behavior depends only on its parameters and the input. Meta-learning breaks this assumption. The learned initialization theta is the product of a specific inner-loop optimization procedure: a particular algorithm (MAML, FOMAML, Reptile), a particular number of inner steps, particular per-step learning rates, particular batch normalization handling, and particular loss weighting. Change any of these and the initialization that was learned under the original procedure becomes mismatched -- it was optimized to work well after *those specific* inner-loop dynamics, not some other set.

Consider a concrete example. Train MAML with 5 inner steps at lr=0.01. The learned theta settles into a region of parameter space where 5 steps of SGD at lr=0.01 reliably reaches good task-specific solutions. Now load that theta and run 3 inner steps at lr=0.05. The initialization was never optimized for that trajectory. Performance degrades, and the cause is invisible unless the checkpoint records what inner-loop configuration produced it.

The checkpoint format defined here solves this by serializing the complete meta-learning state: model parameters, inner-loop hyperparameters, MAML++ learned modules, outer optimizer state, episode sampler configuration, RNG state for reproducibility, and training progress metadata. Every checkpoint is self-describing -- load it, and the loader can reconstruct or validate the exact conditions under which the initialization was trained.

---

## Checkpoint Schema

The canonical checkpoint is a Python dictionary serialized with `torch.save`. Every key at the top level is documented here. Optional keys are marked; all unmarked keys are required.

```python
meta_checkpoint = {
    # ── Version ──────────────────────────────────────────────
    "format_version": "1.0",
    "created_at": "2026-02-19T14:30:00Z",  # ISO-8601 UTC timestamp

    # ── Model State ──────────────────────────────────────────
    "model_state_dict": model.state_dict(),
    "model_class": "brain_ai.meta.Conv4Backbone",  # Fully qualified class name
    "model_config": {                               # Optional: constructor args
        "in_channels": 1,
        "hidden_channels": 64,
        "num_classes": 5,
    },

    # ── Meta-Algorithm Config ────────────────────────────────
    "algo": "maml",             # "maml", "fomaml", "reptile"
    "second_order": True,       # True for MAML, False for FOMAML/Reptile
    "inner_steps": 5,           # Number of inner-loop gradient steps
    "inner_lr": 0.01,           # Base inner-loop learning rate (scalar)
    "inner_clip": 10.0,         # Inner-loop gradient clipping norm
    "backend": "torch_func",    # "torch_func", "higher", "custom"

    # ── MAML++ Parameters (optional, present when enhancements active) ──
    "maml_plus": {
        "use_lslr": True,
        "lslr_state_dict": lslr_module.state_dict(),  # Per-layer per-step LRs
        "use_msl": True,
        "msl_weights": [0.1, 0.2, 0.3, 0.4, 1.0],    # Per-step loss weights
        "msl_mode": "learned",                          # "uniform", "linear_increase", "learned"
        "use_annealing": True,
        "annealing_start_epoch": 20,
        "annealing_current_order": "second",            # "first" or "second"
        "bn_mode": "per_step",                          # "transductive", "per_step", "frozen"
        "bn_step_stats": {                              # BN running stats per inner step
            0: {"running_mean": tensor, "running_var": tensor},
            1: {"running_mean": tensor, "running_var": tensor},
            # ... one entry per inner step
        },
    },

    # ── Outer Optimizer ──────────────────────────────────────
    "outer_optimizer_state_dict": optimizer.state_dict(),
    "outer_optimizer_class": "torch.optim.AdamW",
    "outer_lr": 0.001,
    "outer_weight_decay": 0.01,
    "outer_scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    "outer_scheduler_class": "torch.optim.lr_scheduler.CosineAnnealingLR" or None,

    # ── Episode Sampler Config ───────────────────────────────
    "sampler_config": {
        "dataset": "omniglot",
        "n_way": 5,
        "k_shot": 1,
        "q_query": 15,
        "episodes_per_epoch": 600,
        "augmentation": {"rotations": True, "resize": 28, "flip": False},
        "class_split": "standard",      # "standard" or explicit class lists
        "train_classes": None,           # Optional: list of class indices
        "val_classes": None,
        "test_classes": None,
    },

    # ── Training State ───────────────────────────────────────
    "epoch": 42,
    "global_step": 25200,
    "best_val_accuracy": 0.89,
    "best_epoch": 38,
    "early_stop_patience_remaining": 7,

    # ── RNG State for Reproducibility ────────────────────────
    "rng_state": {
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": numpy.random.get_state(),
        "python": random.getstate(),
        "episode_seed": 42,
    },

    # ── Metrics History (last N epochs) ──────────────────────
    "metrics_history": [
        {
            "epoch": 41,
            "train_loss": 0.50,
            "train_acc": 0.82,
            "val_loss": 0.55,
            "val_acc": 0.87,
            "auac": 0.74,
            "grad_norm_mean": 1.2,
            "inner_lr_mean": 0.0098,
        },
        {
            "epoch": 42,
            "train_loss": 0.45,
            "train_acc": 0.85,
            "val_loss": 0.50,
            "val_acc": 0.89,
            "auac": 0.77,
            "grad_norm_mean": 1.1,
            "inner_lr_mean": 0.0095,
        },
    ],
}
```

### Field-by-Field Reference

| Field | Type | Required | Purpose |
|---|---|---|---|
| `format_version` | `str` | Yes | Semantic version of the checkpoint format. Loaders check this for compatibility. |
| `created_at` | `str` | Yes | ISO-8601 UTC timestamp. Disambiguates checkpoints with same epoch number from different runs. |
| `model_state_dict` | `Dict[str, Tensor]` | Yes | Output of `model.state_dict()`. Contains all learnable parameters and buffers. |
| `model_class` | `str` | Yes | Fully qualified Python class name. Enables the loader to verify or reconstruct the model architecture. |
| `model_config` | `Dict` | No | Constructor arguments for `model_class`. When present, the loader can instantiate the model without external config files. |
| `algo` | `str` | Yes | Meta-algorithm identifier. One of `"maml"`, `"fomaml"`, `"reptile"`. |
| `second_order` | `bool` | Yes | Whether second-order gradients were used. Always `True` for MAML, `False` for FOMAML and Reptile. |
| `inner_steps` | `int` | Yes | Number of inner-loop SGD steps during training. |
| `inner_lr` | `float` | Yes | Base scalar inner-loop learning rate. When LSLR is active, this is the initialization value; actual per-layer per-step LRs are in `maml_plus.lslr_state_dict`. |
| `inner_clip` | `float` | Yes | Max gradient norm for inner-loop clipping. Set to `0.0` or `float('inf')` if clipping was disabled. |
| `backend` | `str` | Yes | Inner-loop engine implementation used during training. |
| `maml_plus` | `Dict` | No | Present only when MAML++ enhancements were active. Contains LSLR state, MSL weights, annealing state, and BN configuration. |
| `outer_optimizer_state_dict` | `Dict` | Yes | Output of `optimizer.state_dict()`. Contains momentum buffers, adaptive learning rate state, etc. |
| `outer_lr` | `float` | Yes | Outer learning rate at checkpoint time. |
| `outer_scheduler_state_dict` | `Dict or None` | No | Learning rate scheduler state. `None` when no scheduler was used. |
| `sampler_config` | `Dict` | Yes | Complete episode sampler configuration. Enables reproducing the exact task distribution. |
| `epoch` | `int` | Yes | Epoch number at checkpoint time. |
| `global_step` | `int` | Yes | Total number of outer-loop optimization steps completed. |
| `best_val_accuracy` | `float` | Yes | Best validation accuracy observed so far. Used for best-model selection. |
| `rng_state` | `Dict` | No | Full RNG state for all sources of randomness. Required for bit-exact training resumption. |
| `metrics_history` | `List[Dict]` | No | Recent epoch metrics. Enables plotting training curves without external log files. |

---

## Save Function

Implement `save_meta_checkpoint` as the single serialization entry point. Consolidate all state collection here so callers cannot accidentally omit fields.

```python
import torch
import json
import numpy
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def save_meta_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    algo: str,
    inner_steps: int,
    inner_lr: float,
    inner_clip: float,
    backend: str,
    sampler_config: Dict[str, Any],
    epoch: int,
    global_step: int,
    best_val_accuracy: float,
    metrics_history: List[Dict[str, float]],
    model_class: Optional[str] = None,
    model_config: Optional[Dict] = None,
    scheduler: Optional[Any] = None,
    maml_plus: Optional[Dict[str, Any]] = None,
    save_rng: bool = True,
    format_version: str = "1.0",
) -> str:
    """
    Save a complete meta-learning checkpoint.

    Return the resolved path string for logging.
    """
    second_order = (algo == "maml")

    checkpoint = {
        "format_version": format_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_state_dict": model.state_dict(),
        "model_class": model_class or f"{model.__class__.__module__}.{model.__class__.__qualname__}",
        "algo": algo,
        "second_order": second_order,
        "inner_steps": inner_steps,
        "inner_lr": inner_lr,
        "inner_clip": inner_clip,
        "backend": backend,
        "outer_optimizer_state_dict": optimizer.state_dict(),
        "outer_optimizer_class": f"{optimizer.__class__.__module__}.{optimizer.__class__.__qualname__}",
        "outer_lr": optimizer.param_groups[0]["lr"],
        "sampler_config": sampler_config,
        "epoch": epoch,
        "global_step": global_step,
        "best_val_accuracy": best_val_accuracy,
        "metrics_history": metrics_history[-50:],  # Keep last 50 epochs
    }

    if model_config is not None:
        checkpoint["model_config"] = model_config

    if scheduler is not None:
        checkpoint["outer_scheduler_state_dict"] = scheduler.state_dict()
        checkpoint["outer_scheduler_class"] = (
            f"{scheduler.__class__.__module__}.{scheduler.__class__.__qualname__}"
        )

    if maml_plus is not None:
        checkpoint["maml_plus"] = maml_plus

    if save_rng:
        checkpoint["rng_state"] = {
            "torch": torch.random.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
            "numpy": numpy.random.get_state(),
            "python": random.getstate(),
        }

    resolved = Path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(resolved))
    return str(resolved)
```

### Building the MAML++ Sub-Dictionary

When MAML++ enhancements are active, construct the `maml_plus` argument from live module state before calling `save_meta_checkpoint`.

```python
def build_maml_plus_dict(
    lslr_module: Optional[torch.nn.Module],
    msl_weights: Optional[Any],
    msl_mode: str,
    use_annealing: bool,
    annealing_start_epoch: int,
    current_order: str,
    bn_mode: str,
    bn_step_stats: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Assemble the maml_plus checkpoint sub-dictionary.
    """
    result = {}

    # LSLR
    if lslr_module is not None:
        result["use_lslr"] = True
        result["lslr_state_dict"] = lslr_module.state_dict()
    else:
        result["use_lslr"] = False

    # MSL
    if msl_weights is not None:
        result["use_msl"] = True
        if isinstance(msl_weights, torch.nn.Module):
            result["msl_weights"] = msl_weights.state_dict()
        elif isinstance(msl_weights, (list, tuple)):
            result["msl_weights"] = list(msl_weights)
        else:
            result["msl_weights"] = msl_weights
        result["msl_mode"] = msl_mode
    else:
        result["use_msl"] = False

    # Annealing
    result["use_annealing"] = use_annealing
    if use_annealing:
        result["annealing_start_epoch"] = annealing_start_epoch
        result["annealing_current_order"] = current_order

    # Batch Normalization
    result["bn_mode"] = bn_mode
    if bn_mode == "per_step" and bn_step_stats is not None:
        result["bn_step_stats"] = bn_step_stats

    return result
```

---

## Load Function

Implement `load_meta_checkpoint` as the single deserialization entry point. Perform version validation, state restoration, and device mapping in one place.

```python
def load_meta_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    lslr_module: Optional[torch.nn.Module] = None,
    device: str = "cpu",
    strict: bool = True,
    restore_rng: bool = True,
) -> Dict[str, Any]:
    """
    Load a meta-learning checkpoint and restore all state.

    Always load to CPU first (map_location='cpu'), then move the model
    to the target device after state_dict loading. This avoids GPU OOM
    on machines with less VRAM than the saving machine.

    Return the raw checkpoint dict so callers can inspect metadata.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # ── Version gate ──
    validate_checkpoint_version(ckpt)

    # ── Model weights ──
    missing, unexpected = model.load_state_dict(
        ckpt["model_state_dict"], strict=strict
    )
    if missing:
        print(f"[checkpoint] Missing keys (strict={strict}): {missing}")
    if unexpected:
        print(f"[checkpoint] Unexpected keys (strict={strict}): {unexpected}")

    # ── Outer optimizer ──
    if optimizer is not None and "outer_optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["outer_optimizer_state_dict"])

    # ── Scheduler ──
    if scheduler is not None and "outer_scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["outer_scheduler_state_dict"])

    # ── MAML++ LSLR ──
    maml_plus = ckpt.get("maml_plus")
    if maml_plus and maml_plus.get("use_lslr") and lslr_module is not None:
        lslr_module.load_state_dict(maml_plus["lslr_state_dict"])

    # ── RNG state ──
    if restore_rng and "rng_state" in ckpt:
        rng = ckpt["rng_state"]
        torch.random.set_rng_state(rng["torch"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        if rng.get("numpy") is not None:
            numpy.random.set_state(rng["numpy"])
        if rng.get("python") is not None:
            random.setstate(rng["python"])

    # ── Move model to target device ──
    model.to(device)

    return ckpt
```

### Version Validation

```python
SUPPORTED_VERSIONS = {"1.0"}


def validate_checkpoint_version(ckpt: Dict[str, Any]) -> None:
    """
    Raise on incompatible format versions.

    Forward compatibility: unknown keys in the checkpoint are ignored.
    Backward compatibility: missing optional keys get defaults in the caller.
    Breaking changes: increment the major version number.
    """
    version = ckpt.get("format_version")
    if version is None:
        raise ValueError(
            "Checkpoint has no format_version field. "
            "This checkpoint predates the versioned format and cannot "
            "be loaded safely. Re-export from the original training run."
        )
    major = version.split(".")[0]
    supported_majors = {v.split(".")[0] for v in SUPPORTED_VERSIONS}
    if major not in supported_majors:
        raise ValueError(
            f"Checkpoint format_version={version} (major={major}) "
            f"is not supported. Supported: {SUPPORTED_VERSIONS}"
        )
```

---

## Validation on Load

Beyond version gating, validate that the checkpoint's inner-loop configuration is compatible with the current training configuration. Mismatches are not necessarily fatal -- sometimes intentional (e.g., increasing inner steps after initial convergence) -- but they must be surfaced as explicit warnings rather than silently altering behavior.

### Mandatory Checks

Perform these checks on every load. Raise `ValueError` if the check fails and the caller did not explicitly opt out.

1. **`format_version` is compatible.** Handled by `validate_checkpoint_version` above.

2. **`model_state_dict` keys match.** Handled by `model.load_state_dict(strict=True)`. When `strict=False`, log all missing and unexpected keys.

3. **LSLR parameter shapes match model architecture.** If the checkpoint contains `maml_plus.lslr_state_dict` and LSLR is active in the current config, verify that the LSLR parameter tensor has shape `(inner_steps, num_layers)`. A shape mismatch means the model architecture or number of inner steps changed, and the learned per-layer per-step rates are incompatible.

```python
def validate_lslr_shapes(ckpt: Dict, model: torch.nn.Module, inner_steps: int) -> None:
    maml_plus = ckpt.get("maml_plus", {})
    if not maml_plus.get("use_lslr"):
        return
    lslr_sd = maml_plus["lslr_state_dict"]
    expected_layers = len(list(model.parameters()))
    for key, tensor in lslr_sd.items():
        if "log_lrs" in key:
            if tensor.shape != (inner_steps, expected_layers):
                raise ValueError(
                    f"LSLR shape mismatch: checkpoint has {tensor.shape}, "
                    f"expected ({inner_steps}, {expected_layers}). "
                    f"Model architecture or inner_steps changed."
                )
```

4. **MSL weights length matches `inner_steps`.** If `maml_plus.msl_weights` is a list, its length must equal `inner_steps`. A length mismatch means inner steps changed.

```python
def validate_msl_weights(ckpt: Dict, inner_steps: int) -> None:
    maml_plus = ckpt.get("maml_plus", {})
    if not maml_plus.get("use_msl"):
        return
    weights = maml_plus.get("msl_weights")
    if isinstance(weights, list) and len(weights) != inner_steps:
        raise ValueError(
            f"MSL weights length {len(weights)} != inner_steps {inner_steps}. "
            f"Inner step count changed since checkpoint was saved."
        )
```

### Advisory Warnings

Log warnings but do not raise for these mismatches. The caller may be intentionally changing configuration.

- **`algo` mismatch.** Loading a MAML checkpoint to continue training with FOMAML is a valid workflow (e.g., switching to first-order for speed after initial convergence). Warn but proceed.
- **`inner_steps` mismatch.** The initialization was optimized for the checkpointed step count. Different step counts during evaluation or continued training will work but may not match reported accuracy.
- **`inner_lr` mismatch.** Similar reasoning to inner steps. The initialization was tuned for the saved learning rate.
- **`backend` mismatch.** All three backends (torch.func, higher, custom) should produce identical forward behavior. A backend change is safe but worth noting.
- **`second_order` mismatch.** Switching from MAML (second-order) to FOMAML (first-order) is common. Warn because the initialization quality may differ under first-order adaptation.

---

## Hyperparameter Diff Utility

When resuming training from a checkpoint with a different configuration, surface all differences explicitly. This prevents the subtle failure mode where training resumes with mismatched hyperparameters and produces inexplicably different results.

```python
from typing import Any, Dict, Tuple


# Fields to compare between checkpoint and current config
DIFF_FIELDS = [
    ("algo", "algo"),
    ("inner_steps", "inner_steps"),
    ("inner_lr", "inner_lr"),
    ("inner_clip", "inner_clip"),
    ("second_order", "second_order"),
    ("backend", "backend"),
]


def diff_checkpoint_config(
    ckpt: Dict[str, Any],
    current_config: Any,
) -> Dict[str, Tuple[Any, Any]]:
    """
    Compare checkpoint hyperparameters against the current training config.

    Return a dict of {field_name: (checkpoint_value, current_value)} for
    every field that differs. An empty dict means the configs match.
    """
    diffs = {}
    for ckpt_key, config_attr in DIFF_FIELDS:
        ckpt_val = ckpt.get(ckpt_key)
        current_val = getattr(current_config, config_attr, None)
        if ckpt_val != current_val:
            diffs[ckpt_key] = (ckpt_val, current_val)

    # MAML++ sub-fields
    ckpt_mp = ckpt.get("maml_plus", {})
    if hasattr(current_config, "maml_plus"):
        mp_config = current_config.maml_plus
        for field in ["use_lslr", "use_msl", "bn_mode", "use_annealing"]:
            ckpt_val = ckpt_mp.get(field)
            current_val = getattr(mp_config, field, None)
            if ckpt_val != current_val:
                diffs[f"maml_plus.{field}"] = (ckpt_val, current_val)

    # Sampler config
    ckpt_sampler = ckpt.get("sampler_config", {})
    if hasattr(current_config, "sampler_config"):
        sc = current_config.sampler_config
        for field in ["dataset", "n_way", "k_shot", "q_query", "episodes_per_epoch"]:
            ckpt_val = ckpt_sampler.get(field)
            current_val = getattr(sc, field, None)
            if ckpt_val != current_val:
                diffs[f"sampler.{field}"] = (ckpt_val, current_val)

    return diffs


def log_checkpoint_diffs(diffs: Dict[str, Tuple[Any, Any]]) -> None:
    """
    Log all checkpoint-vs-config differences as warnings.
    """
    if not diffs:
        return
    print("[checkpoint] WARNING: Config differs from checkpoint:")
    for key, (ckpt_val, current_val) in sorted(diffs.items()):
        print(f"  {key}: checkpoint={ckpt_val} -> current={current_val}")
    print(
        "[checkpoint] The saved initialization was optimized for the "
        "checkpoint config. Proceeding with the current config may "
        "affect adaptation quality."
    )
```

### Integration Pattern

Call the diff utility immediately after loading, before training resumes.

```python
ckpt = load_meta_checkpoint("checkpoints/meta_maml_omniglot_best.pt", model, optimizer)
diffs = diff_checkpoint_config(ckpt, current_config)
log_checkpoint_diffs(diffs)
# At this point the operator sees exactly what changed and can decide
# whether to proceed or adjust the config.
```

---

## Checkpoint Compatibility

### Forward Compatibility

New fields added in future format versions are ignored by older loaders. The loader accesses only the keys it knows about via `ckpt.get(key, default)`. Unknown keys do not cause errors.

### Backward Compatibility

Missing optional fields receive sensible defaults in the loader. For example, if `rng_state` is absent, the loader skips RNG restoration. If `maml_plus` is absent, the loader assumes no MAML++ enhancements were active.

```python
# Pattern: always use .get() with a default for optional fields
maml_plus = ckpt.get("maml_plus", None)
metrics_history = ckpt.get("metrics_history", [])
rng_state = ckpt.get("rng_state", None)
model_config = ckpt.get("model_config", None)
```

### Breaking Changes

Increment the major version number (`format_version` from `"1.x"` to `"2.0"`) when:

- A required field is renamed or removed.
- The structure of `model_state_dict` changes due to module refactoring (key names change).
- The `maml_plus` sub-dictionary schema changes in a way that old loaders would misinterpret.

When a major version change occurs, provide a migration script:

```python
def migrate_v1_to_v2(ckpt_v1: Dict) -> Dict:
    """
    Migrate a v1.x checkpoint to v2.0 format.
    """
    ckpt_v2 = dict(ckpt_v1)
    ckpt_v2["format_version"] = "2.0"
    # Example: rename 'inner_lr' to 'base_inner_lr'
    ckpt_v2["base_inner_lr"] = ckpt_v2.pop("inner_lr", 0.01)
    # Example: restructure maml_plus
    # ...
    return ckpt_v2
```

### Cross-Device Compatibility

Always save with default device mapping. Always load with `map_location='cpu'`:

```python
# Save: no special handling needed -- torch.save serializes tensors
# with their device info, but the loader controls where they land.
torch.save(checkpoint, path)

# Load: force CPU, then move to target device after loading.
ckpt = torch.load(path, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.to(target_device)
```

This handles all cross-device scenarios: saving on GPU loading on CPU, saving on multi-GPU loading on single GPU, saving on CUDA loading on MPS.

---

## Lightweight Checkpoint (Eval Only)

For deployment and evaluation, save only the fields required to run the inner loop. Omit optimizer state, RNG state, metrics history, and training bookkeeping. This reduces checkpoint size significantly (often 40-60% smaller) and removes training-specific state that is irrelevant at inference time.

```python
def save_eval_checkpoint(
    path: str,
    model: torch.nn.Module,
    algo: str,
    inner_steps: int,
    inner_lr: float,
    inner_clip: float,
    backend: str,
    lslr_module: Optional[torch.nn.Module] = None,
    bn_mode: str = "transductive",
    bn_step_stats: Optional[Dict] = None,
    model_class: Optional[str] = None,
) -> str:
    """
    Save a lightweight checkpoint for evaluation/deployment.
    """
    checkpoint = {
        "format_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_type": "eval",       # Distinguishes from full checkpoint
        "model_state_dict": model.state_dict(),
        "model_class": model_class or f"{model.__class__.__module__}.{model.__class__.__qualname__}",
        "algo": algo,
        "second_order": (algo == "maml"),
        "inner_steps": inner_steps,
        "inner_lr": inner_lr,
        "inner_clip": inner_clip,
        "backend": backend,
    }

    # LSLR is needed at eval time -- the learned LRs are part of the
    # adaptation procedure.
    if lslr_module is not None:
        checkpoint["maml_plus"] = {
            "use_lslr": True,
            "lslr_state_dict": lslr_module.state_dict(),
            "bn_mode": bn_mode,
        }
        if bn_mode == "per_step" and bn_step_stats is not None:
            checkpoint["maml_plus"]["bn_step_stats"] = bn_step_stats

    resolved = Path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(resolved))
    return str(resolved)
```

Note that LSLR parameters are included in eval checkpoints. The learned per-layer per-step learning rates are part of the adaptation procedure itself -- without them, the inner loop runs with the base scalar `inner_lr` and adaptation quality degrades. Similarly, `bn_mode` and `bn_step_stats` are included because batch normalization behavior during the inner loop directly affects adapted model outputs.

---

## Checkpoint Naming Convention

### File Name Pattern

```
meta_{algo}_{dataset}_{n_way}w{k_shot}s_epoch{epoch:04d}.pt
```

Examples:
- `meta_maml_omniglot_5w1s_epoch0042.pt`
- `meta_fomaml_miniimagenet_5w5s_epoch0120.pt`
- `meta_reptile_omniglot_20w1s_epoch0300.pt`

### Best Checkpoint

Maintain a symlink to the current best checkpoint:

```
meta_{algo}_{dataset}_best.pt -> meta_{algo}_{dataset}_{n_way}w{k_shot}s_epoch0038.pt
```

Implementation:

```python
import os

def update_best_symlink(checkpoint_dir: str, algo: str, dataset: str, checkpoint_name: str) -> None:
    """
    Update the 'best' symlink to point to the current best checkpoint.
    """
    link_name = os.path.join(checkpoint_dir, f"meta_{algo}_{dataset}_best.pt")
    target = checkpoint_name  # Just the filename, not the full path
    if os.path.islink(link_name):
        os.unlink(link_name)
    os.symlink(target, link_name)
```

### Eval Checkpoint Naming

Append `_eval` before the extension:

```
meta_maml_omniglot_5w1s_epoch0042_eval.pt
```

### Directory Layout

```
checkpoints/
  meta_maml_omniglot_5w1s_epoch0040.pt
  meta_maml_omniglot_5w1s_epoch0041.pt
  meta_maml_omniglot_5w1s_epoch0042.pt
  meta_maml_omniglot_best.pt -> meta_maml_omniglot_5w1s_epoch0042.pt
  meta_maml_omniglot_5w1s_epoch0042_eval.pt
```

---

## Testing Patterns

### 1. Round-Trip Test

Save a checkpoint, load it into a fresh model, save again, load again. Verify that all state dicts are identical across the two loads.

```python
def test_round_trip(tmp_path):
    model = build_test_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    sampler_config = {"dataset": "omniglot", "n_way": 5, "k_shot": 1, "q_query": 15}

    # Save
    path1 = str(tmp_path / "ckpt1.pt")
    save_meta_checkpoint(
        path1, model, optimizer, algo="maml", inner_steps=5,
        inner_lr=0.01, inner_clip=10.0, backend="torch_func",
        sampler_config=sampler_config, epoch=10, global_step=6000,
        best_val_accuracy=0.85, metrics_history=[],
    )

    # Load into fresh model
    model2 = build_test_model()
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
    ckpt1 = load_meta_checkpoint(path1, model2, optimizer2)

    # Save again
    path2 = str(tmp_path / "ckpt2.pt")
    save_meta_checkpoint(
        path2, model2, optimizer2, algo="maml", inner_steps=5,
        inner_lr=0.01, inner_clip=10.0, backend="torch_func",
        sampler_config=sampler_config, epoch=10, global_step=6000,
        best_val_accuracy=0.85, metrics_history=[],
    )

    # Load second checkpoint
    model3 = build_test_model()
    optimizer3 = torch.optim.Adam(model3.parameters(), lr=0.001)
    ckpt2 = load_meta_checkpoint(path2, model3, optimizer3)

    # Verify state_dicts match
    for (k1, v1), (k2, v2) in zip(
        model2.state_dict().items(), model3.state_dict().items()
    ):
        assert k1 == k2
        assert torch.equal(v1, v2), f"Model param {k1} differs after round-trip"
```

### 2. Hyperparameter Preservation Test

Save a checkpoint with LSLR active. Load it. Verify that the learned per-layer per-step learning rates are restored exactly.

```python
def test_lslr_preservation(tmp_path):
    model = build_test_model()
    lslr = PerLayerPerStepLR(
        param_shapes=[p.shape for p in model.parameters()],
        num_steps=5, init_lr=0.01,
    )
    # Mutate LSLR to non-initial values
    with torch.no_grad():
        lslr.log_lrs.uniform_(-3.0, -1.0)
    original_lrs = lslr.log_lrs.clone()

    maml_plus_dict = build_maml_plus_dict(
        lslr_module=lslr, msl_weights=None, msl_mode="uniform",
        use_annealing=False, annealing_start_epoch=0,
        current_order="first", bn_mode="frozen",
    )

    path = str(tmp_path / "ckpt_lslr.pt")
    optimizer = torch.optim.Adam(model.parameters())
    save_meta_checkpoint(
        path, model, optimizer, algo="maml", inner_steps=5,
        inner_lr=0.01, inner_clip=10.0, backend="torch_func",
        sampler_config={}, epoch=1, global_step=600,
        best_val_accuracy=0.5, metrics_history=[],
        maml_plus=maml_plus_dict,
    )

    # Load into fresh LSLR
    model2 = build_test_model()
    lslr2 = PerLayerPerStepLR(
        param_shapes=[p.shape for p in model2.parameters()],
        num_steps=5, init_lr=0.01,
    )
    load_meta_checkpoint(path, model2, lslr_module=lslr2)

    assert torch.equal(lslr2.log_lrs, original_lrs), "LSLR not restored"
```

### 3. Cross-Device Test

Save on CUDA, load on CPU. Verify model produces identical outputs.

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cross_device(tmp_path):
    model_gpu = build_test_model().cuda()
    x = torch.randn(2, 1, 28, 28, device="cuda")
    out_gpu = model_gpu(x).detach().cpu()

    optimizer = torch.optim.Adam(model_gpu.parameters())
    path = str(tmp_path / "ckpt_gpu.pt")
    save_meta_checkpoint(
        path, model_gpu, optimizer, algo="maml", inner_steps=5,
        inner_lr=0.01, inner_clip=10.0, backend="torch_func",
        sampler_config={}, epoch=1, global_step=600,
        best_val_accuracy=0.5, metrics_history=[],
    )

    # Load on CPU
    model_cpu = build_test_model()
    optimizer_cpu = torch.optim.Adam(model_cpu.parameters())
    load_meta_checkpoint(path, model_cpu, optimizer_cpu, device="cpu")

    x_cpu = x.cpu()
    out_cpu = model_cpu(x_cpu).detach()
    assert torch.allclose(out_gpu, out_cpu, atol=1e-6), "Cross-device output mismatch"
```

### 4. Version Compatibility Test

Verify that a checkpoint without optional fields (simulating an older version) loads correctly with defaults.

```python
def test_backward_compat(tmp_path):
    # Simulate a v1.0 checkpoint missing optional fields
    minimal_ckpt = {
        "format_version": "1.0",
        "created_at": "2026-01-01T00:00:00Z",
        "model_state_dict": build_test_model().state_dict(),
        "model_class": "test.TestModel",
        "algo": "fomaml",
        "second_order": False,
        "inner_steps": 3,
        "inner_lr": 0.05,
        "inner_clip": 5.0,
        "backend": "custom",
        "outer_optimizer_state_dict": torch.optim.SGD(
            build_test_model().parameters(), lr=0.1
        ).state_dict(),
        "outer_lr": 0.1,
        "sampler_config": {"dataset": "omniglot", "n_way": 5, "k_shot": 1},
        "epoch": 5,
        "global_step": 3000,
        "best_val_accuracy": 0.7,
        # Intentionally missing: rng_state, metrics_history, maml_plus
    }
    path = str(tmp_path / "ckpt_minimal.pt")
    torch.save(minimal_ckpt, path)

    model = build_test_model()
    ckpt = load_meta_checkpoint(path, model, restore_rng=True)

    # Should load without error; missing fields get defaults
    assert ckpt["algo"] == "fomaml"
    assert ckpt.get("rng_state") is None
    assert ckpt.get("metrics_history") is None
    assert ckpt.get("maml_plus") is None
```

### 5. Diff Utility Test

Verify that the hyperparameter diff function detects known mismatches.

```python
def test_diff_detection():
    ckpt = {
        "algo": "maml",
        "inner_steps": 5,
        "inner_lr": 0.01,
        "inner_clip": 10.0,
        "second_order": True,
        "backend": "torch_func",
        "sampler_config": {"dataset": "omniglot", "n_way": 5, "k_shot": 1},
    }

    class MockConfig:
        algo = "fomaml"        # Changed
        inner_steps = 5        # Same
        inner_lr = 0.05        # Changed
        inner_clip = 10.0      # Same
        second_order = False   # Changed
        backend = "torch_func" # Same

    diffs = diff_checkpoint_config(ckpt, MockConfig())
    assert "algo" in diffs
    assert diffs["algo"] == ("maml", "fomaml")
    assert "inner_lr" in diffs
    assert diffs["inner_lr"] == (0.01, 0.05)
    assert "second_order" in diffs
    assert "inner_steps" not in diffs
    assert "backend" not in diffs
```

---

## Integration with Phase 7 Runner

The Phase 7 training runner (`scripts/train_phase7_meta.py`) uses the checkpoint format for three workflows.

### Resume Interrupted Training

Save a full checkpoint at the end of every epoch and after every validation improvement. On resume, restore the complete state including RNG for bit-exact continuation.

```python
# In the Phase 7 training loop:
for epoch in range(start_epoch, max_epochs):
    train_one_epoch(model, optimizer, sampler, epoch)
    val_acc = validate(model, val_sampler, epoch)

    is_best = val_acc > best_val_accuracy
    if is_best:
        best_val_accuracy = val_acc

    # Save checkpoint every epoch
    ckpt_name = f"meta_{algo}_{dataset}_{n_way}w{k_shot}s_epoch{epoch:04d}.pt"
    save_meta_checkpoint(
        os.path.join(ckpt_dir, ckpt_name),
        model, optimizer, algo=algo, inner_steps=inner_steps,
        inner_lr=inner_lr, inner_clip=inner_clip, backend=backend,
        sampler_config=sampler.config_dict(), epoch=epoch,
        global_step=global_step, best_val_accuracy=best_val_accuracy,
        metrics_history=metrics_history, scheduler=scheduler,
        maml_plus=build_maml_plus_dict(...) if use_maml_plus else None,
    )

    if is_best:
        update_best_symlink(ckpt_dir, algo, dataset, ckpt_name)

# Resume path:
if args.resume:
    ckpt = load_meta_checkpoint(args.resume, model, optimizer, scheduler, lslr)
    start_epoch = ckpt["epoch"] + 1
    global_step = ckpt["global_step"]
    best_val_accuracy = ckpt["best_val_accuracy"]
    metrics_history = ckpt.get("metrics_history", [])
    diffs = diff_checkpoint_config(ckpt, config)
    log_checkpoint_diffs(diffs)
```

### Best Model Selection

Track `best_val_accuracy` across epochs. When validation accuracy improves, update the best symlink. At the end of training, the best checkpoint is always available at the symlink path.

For evaluation, load the best checkpoint (or an eval-only export of it) and run the standard N-way K-shot evaluation protocol:

```python
# Load best model for evaluation
ckpt = load_meta_checkpoint(
    f"checkpoints/meta_{algo}_{dataset}_best.pt",
    model,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
# Use ckpt["inner_steps"] and ckpt["inner_lr"] for evaluation
# to match the conditions under which this initialization was trained.
eval_results = evaluate_few_shot(
    model, test_sampler,
    inner_steps=ckpt["inner_steps"],
    inner_lr=ckpt["inner_lr"],
    num_episodes=600,
)
```

### Algorithm Comparison

Load the same base model checkpoint and train with different meta-algorithms. The checkpoint provides a controlled starting point so differences in final accuracy are attributable to the algorithm, not the initialization.

```python
for algo in ["maml", "fomaml", "reptile"]:
    model = build_model()
    optimizer = build_optimizer(model)

    # Load shared initialization (pre-meta-training backbone)
    base_ckpt = load_meta_checkpoint(
        "checkpoints/pretrained_backbone.pt",
        model,
        strict=False,  # Base checkpoint may not have meta-specific keys
    )

    # Train with this algorithm
    train_meta(model, optimizer, algo=algo, ...)

    # Save algorithm-specific checkpoint
    save_meta_checkpoint(
        f"checkpoints/meta_{algo}_{dataset}_final.pt",
        model, optimizer, algo=algo, ...
    )
```

---

## Appendix A: Why Inner-Loop Hyperparameters Belong in the Checkpoint

The learned initialization theta in MAML is not a general-purpose weight vector. It is the solution to a bilevel optimization problem:

```
theta* = argmin_theta  E_task [ L_query( theta - alpha * grad L_support(theta) ) ]
```

The outer objective evaluates theta *after* the inner-loop transformation `theta - alpha * grad L_support(theta)`. Change alpha (the inner learning rate), change the number of gradient steps, change the loss function -- and the optimal theta changes. The checkpoint must record these parameters because they define the manifold on which theta was optimized.

Concretely, loading a MAML checkpoint and running FOMAML evaluation (first-order, no `create_graph`) will produce different adaptation trajectories than the MAML evaluation the checkpoint was trained for. The difference may be small (FOMAML is often a good approximation), but it is real and must be documented in the checkpoint rather than left as an implicit assumption.

## Appendix B: RNG State Scope

The RNG state section captures four independent random number generators:

| Generator | What It Controls | Consequence of Not Restoring |
|---|---|---|
| `torch` CPU RNG | Weight initialization, dropout masks, data augmentation on CPU | Different dropout patterns, non-reproducible training |
| `torch.cuda` RNG | Same as CPU but for CUDA operations | GPU-specific non-determinism |
| `numpy` RNG | Dataset loading, class splitting, some augmentation libraries | Different episode compositions |
| `python` `random` RNG | Shuffling, sampling in pure-Python code paths | Different task orderings |

For bit-exact resumption, all four generators must be restored. For approximate resumption (same trajectory, not bit-exact), restoring only `torch` and `torch.cuda` is usually sufficient.

The `episode_seed` field in `sampler_config` is separate from the RNG state. It is the base seed from which per-episode seeds are derived deterministically. Even without restoring the full RNG state, the episode seed ensures that the same episodes can be regenerated.

## Appendix C: Checkpoint Size Estimates

| Component | Typical Size (Conv4, Omniglot) | Typical Size (ResNet-12, mini-ImageNet) |
|---|---|---|
| `model_state_dict` | ~400 KB | ~50 MB |
| `outer_optimizer_state_dict` (Adam) | ~800 KB | ~100 MB |
| `lslr_state_dict` | ~1 KB | ~1 KB |
| `rng_state` | ~10 KB | ~10 KB |
| `metrics_history` (50 epochs) | ~5 KB | ~5 KB |
| **Full checkpoint** | **~1.2 MB** | **~150 MB** |
| **Eval checkpoint** | **~400 KB** | **~50 MB** |

The dominant cost is always the model parameters and the Adam optimizer state (which stores two momentum tensors per parameter, tripling the parameter storage). Eval checkpoints that drop the optimizer state are roughly one-third the size of full checkpoints.
