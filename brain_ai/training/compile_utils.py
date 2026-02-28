"""
Compile and checkpointing utilities for Brain-AI training.

Provides high-level helpers that read from ``TrainingConfig`` and apply
torch.compile / gradient-checkpointing to a :class:`BrainAI` model, plus
lightweight memory-profiling helpers.

Usage::

    from brain_ai.config import BrainAIConfig
    from brain_ai.system import create_brain_ai
    from brain_ai.training.compile_utils import (
        setup_torch_compile,
        setup_gradient_checkpointing,
        profile_memory,
        get_compile_stats,
    )

    config = BrainAIConfig()
    config.training.use_torch_compile = True
    config.training.use_gradient_checkpointing = True

    model = create_brain_ai(modalities=["vision"], num_classes=10)
    setup_gradient_checkpointing(model, config)
    setup_torch_compile(model, config)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# torch.compile
# ---------------------------------------------------------------------------

def setup_torch_compile(
    model: nn.Module,
    config: Any,
    *,
    extra_kwargs: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """Apply ``torch.compile`` to *model* based on *config*.

    If ``config.training.use_torch_compile`` is ``False`` or ``torch.compile``
    is not available, the model is returned unchanged.

    The function delegates to ``model.compile_model(...)`` when it exists
    (i.e. the model is a :class:`BrainAI` instance), which selectively
    compiles the heaviest submodules.  For arbitrary ``nn.Module`` instances
    the whole model is compiled.

    Args:
        model: The model (or :class:`BrainAI` instance) to compile.
        config: A :class:`BrainAIConfig` with a ``training`` attribute.
        extra_kwargs: Extra keyword arguments forwarded to ``torch.compile``.

    Returns:
        The (possibly compiled) model.
    """
    training_cfg = getattr(config, "training", None)
    if training_cfg is None:
        logger.warning("Config has no 'training' attribute -- skipping torch.compile")
        return model

    if not getattr(training_cfg, "use_torch_compile", False):
        logger.info("torch.compile disabled in config -- skipping")
        return model

    if not hasattr(torch, "compile"):
        logger.warning(
            "torch.compile is not available (requires PyTorch >= 2.0). Skipping."
        )
        return model

    backend = getattr(training_cfg, "torch_compile_backend", "inductor")
    kwargs: Dict[str, Any] = extra_kwargs or {}

    # Prefer the selective compile_model() method on BrainAI
    if hasattr(model, "compile_model"):
        logger.info("Compiling BrainAI submodules with backend=%s", backend)
        model.compile_model(backend=backend, **kwargs)
        return model

    # Fallback: compile the entire module
    logger.info("Compiling full model with backend=%s", backend)
    model = torch.compile(model, backend=backend, **kwargs)
    return model


# ---------------------------------------------------------------------------
# Gradient checkpointing
# ---------------------------------------------------------------------------

def setup_gradient_checkpointing(
    model: nn.Module,
    config: Any,
) -> None:
    """Enable gradient checkpointing on *model* based on *config*.

    If ``config.training.use_gradient_checkpointing`` is ``False`` the call
    is a no-op.  The function delegates to ``model.enable_gradient_checkpointing()``
    when it exists (:class:`BrainAI`), otherwise it applies a generic wrapper
    to every submodule that has more than 1M parameters.

    Args:
        model: The model to modify **in-place**.
        config: A :class:`BrainAIConfig` with a ``training`` attribute.
    """
    training_cfg = getattr(config, "training", None)
    if training_cfg is None:
        logger.warning(
            "Config has no 'training' attribute -- skipping gradient checkpointing"
        )
        return

    if not getattr(training_cfg, "use_gradient_checkpointing", False):
        logger.info("Gradient checkpointing disabled in config -- skipping")
        return

    if hasattr(model, "enable_gradient_checkpointing"):
        logger.info("Enabling gradient checkpointing via BrainAI helper")
        model.enable_gradient_checkpointing()
        return

    # Generic fallback: wrap submodules with >1M parameters
    from torch.utils.checkpoint import checkpoint as _checkpoint
    import functools

    _PARAM_THRESHOLD = 1_000_000

    wrapped = 0
    for name, submodule in model.named_children():
        n_params = sum(p.numel() for p in submodule.parameters())
        if n_params < _PARAM_THRESHOLD:
            continue
        if getattr(submodule, "_gradient_checkpointing", False):
            continue

        original_forward = submodule.forward

        @functools.wraps(original_forward)
        def _make_wrapper(fwd):
            def _checkpointed_forward(*args, **kwargs):
                def _run(*a):
                    return fwd(*a, **kwargs)
                return _checkpoint(_run, *args, use_reentrant=False)
            return _checkpointed_forward

        submodule.forward = _make_wrapper(original_forward)
        submodule._gradient_checkpointing = True
        wrapped += 1

    logger.info(
        "Generic gradient checkpointing: wrapped %d submodule(s) with >%d params",
        wrapped,
        _PARAM_THRESHOLD,
    )


# ---------------------------------------------------------------------------
# Memory profiling
# ---------------------------------------------------------------------------

def profile_memory(
    model: nn.Module,
    sample_input: Union[torch.Tensor, Dict[str, torch.Tensor]],
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Profile peak GPU memory for a forward + backward pass.

    Runs a single forward/backward with the given *sample_input* and reports
    peak allocated memory **before** and **after** the run in MiB.

    .. note::

       This resets the CUDA peak-memory stats, so it should only be called
       during an explicit profiling phase -- not mid-training.

    Args:
        model: The model to profile.
        sample_input: A tensor or dict of tensors suitable for ``model.forward``.
        device: Device override.  Detected from *model* parameters by default.

    Returns:
        Dict with keys ``"peak_memory_mib"``, ``"allocated_before_mib"``,
        ``"allocated_after_mib"``.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    result: Dict[str, float] = {}

    if device.type != "cuda":
        logger.info("Memory profiling is only meaningful on CUDA -- returning zeros")
        return {"peak_memory_mib": 0.0, "allocated_before_mib": 0.0, "allocated_after_mib": 0.0}

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    allocated_before = torch.cuda.memory_allocated(device) / (1024 ** 2)

    # Forward
    if isinstance(sample_input, dict):
        out = model(sample_input)
    else:
        out = model(sample_input)

    # Backward (create a scalar loss to differentiate)
    if isinstance(out, torch.Tensor):
        loss = out.sum()
    elif hasattr(out, "output"):
        loss = out.output.sum()
    else:
        # Best-effort: pick first tensor returned
        loss = out[0].sum() if isinstance(out, (tuple, list)) else out.sum()

    loss.backward()

    torch.cuda.synchronize(device)
    allocated_after = torch.cuda.memory_allocated(device) / (1024 ** 2)
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    result["allocated_before_mib"] = round(allocated_before, 2)
    result["allocated_after_mib"] = round(allocated_after, 2)
    result["peak_memory_mib"] = round(peak, 2)

    logger.info(
        "Memory profile: before=%.1f MiB, after=%.1f MiB, peak=%.1f MiB",
        allocated_before,
        allocated_after,
        peak,
    )

    # Clean up gradients
    model.zero_grad(set_to_none=True)

    return result


# ---------------------------------------------------------------------------
# Compile introspection
# ---------------------------------------------------------------------------

def get_compile_stats(model: nn.Module) -> Dict[str, Any]:
    """Return a dict describing which submodules have been compiled.

    For each immediate child and for the model itself, the returned dict
    indicates whether the module is a ``torch._dynamo`` compiled object or
    has the ``_gradient_checkpointing`` flag set.

    Args:
        model: The model to inspect.

    Returns:
        Dict mapping submodule names to their compile/checkpoint status.
    """
    stats: Dict[str, Any] = {
        "torch_compile_available": hasattr(torch, "compile"),
        "model_type": type(model).__name__,
        "submodules": {},
    }

    def _is_compiled(m: nn.Module) -> bool:
        """Heuristic check for whether a module has been torch.compiled."""
        # torch.compile wraps modules in OptimizedModule
        type_name = type(m).__name__
        if "OptimizedModule" in type_name or "Compiled" in type_name:
            return True
        if hasattr(m, "_orig_mod"):
            return True
        return False

    # Check top-level
    stats["model_compiled"] = _is_compiled(model)

    # Check immediate children
    for name, child in model.named_children():
        child_info: Dict[str, Any] = {
            "compiled": _is_compiled(child),
            "gradient_checkpointing": getattr(child, "_gradient_checkpointing", False),
            "num_parameters": sum(p.numel() for p in child.parameters()),
        }

        # For ModuleDict (like encoders), drill one level deeper
        if isinstance(child, nn.ModuleDict):
            child_info["children"] = {}
            for sub_name, sub_child in child.items():
                child_info["children"][sub_name] = {
                    "compiled": _is_compiled(sub_child),
                    "gradient_checkpointing": getattr(
                        sub_child, "_gradient_checkpointing", False
                    ),
                    "num_parameters": sum(p.numel() for p in sub_child.parameters()),
                }

        stats["submodules"][name] = child_info

    return stats
