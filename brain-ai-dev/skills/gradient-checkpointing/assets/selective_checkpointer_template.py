"""
selective_checkpointer_template.py
-----------------------------------
SelectiveCheckpointer that wraps only the most memory-expensive layers
in CheckpointWrapper, based on profiling data or heuristics.

Supports four strategies:
  - "none":       no checkpointing
  - "full":       checkpoint every child module
  - "selective":  checkpoint only layers above memory threshold
  - "sequential": use checkpoint_sequential for nn.Sequential models

Usage:
    from selective_checkpointer_template import SelectiveCheckpointer
    from checkpoint_config_template import CheckpointConfig
    from memory_profiler_template import MemoryProfiler

    config = CheckpointConfig(enabled=True, strategy="selective", memory_threshold_mb=50.0)
    profiler = MemoryProfiler(model, device)
    report = profiler.profile(sample_input)

    checkpointer = SelectiveCheckpointer(config, profiler=profiler)
    model = checkpointer.apply(model, profile_report=report)
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Optional, Set

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import sibling templates
# ---------------------------------------------------------------------------

try:
    from checkpoint_wrapper_template import CheckpointWrapper
except ImportError:
    # Fallback: define a minimal wrapper for standalone testing
    from torch.utils.checkpoint import checkpoint as _checkpoint

    class CheckpointWrapper(nn.Module):  # type: ignore[no-redef]
        def __init__(self, module, use_reentrant=False, preserve_rng_state=True):
            super().__init__()
            self.module = module
            self.use_reentrant = use_reentrant
            self.preserve_rng_state = preserve_rng_state

        def forward(self, *args, **kwargs):
            if not torch.is_grad_enabled():
                return self.module(*args, **kwargs)
            if self.use_reentrant:
                def run_fn(*a):
                    return self.module(*a, **kwargs)
                return _checkpoint(run_fn, *args, use_reentrant=True,
                                   preserve_rng_state=self.preserve_rng_state)
            return _checkpoint(self.module, *args, use_reentrant=False,
                               preserve_rng_state=self.preserve_rng_state, **kwargs)

try:
    from checkpoint_config_template import CheckpointConfig
except ImportError:
    pass

try:
    from memory_profiler_template import MemoryProfiler, ProfileReport
except ImportError:
    ProfileReport = None  # type: ignore[assignment, misc]


# ---------------------------------------------------------------------------
# SelectiveCheckpointer
# ---------------------------------------------------------------------------


class SelectiveCheckpointer:
    """
    Apply gradient checkpointing to a model based on a configurable strategy.

    Parameters
    ----------
    config : CheckpointConfig
        Configuration controlling strategy, threshold, patterns, etc.
    profiler : MemoryProfiler, optional
        If provided and config.profile_before_apply is True, the profiler
        is used to measure per-layer memory before deciding which layers
        to checkpoint.
    """

    def __init__(
        self,
        config: Any,
        profiler: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.profiler = profiler
        self._checkpointed_layers: List[str] = []

    def apply(
        self,
        model: nn.Module,
        profile_report: Optional[Any] = None,
        sample_input: Optional[Dict[str, torch.Tensor]] = None,
    ) -> nn.Module:
        """
        Apply checkpointing to the model based on the configured strategy.

        Parameters
        ----------
        model : nn.Module
            The model to modify (in-place).
        profile_report : ProfileReport, optional
            Pre-computed profiling report. If None and strategy="selective"
            with profile_before_apply=True, the profiler will be invoked.
        sample_input : dict, optional
            Sample input for on-the-fly profiling. Required if profile_report
            is None and profiling is needed.

        Returns
        -------
        nn.Module
            The model with checkpointing applied.
        """
        self._checkpointed_layers = []

        if not self.config.enabled or self.config.strategy == "none":
            logger.info("Checkpointing disabled (enabled=%s, strategy=%s)",
                        self.config.enabled, self.config.strategy)
            return model

        strategy = self.config.strategy

        if strategy == "full":
            return self._apply_full(model)
        elif strategy == "selective":
            return self._apply_selective(model, profile_report, sample_input)
        elif strategy == "sequential":
            return self._apply_sequential(model)
        else:
            raise ValueError(
                f"Unknown checkpointing strategy: {strategy!r}. "
                f"Expected one of: none, full, selective, sequential."
            )

    def get_checkpointed_layers(self) -> List[str]:
        """Return names of layers that were wrapped with CheckpointWrapper."""
        return list(self._checkpointed_layers)

    # -------------------------------------------------------------------
    # Strategy implementations
    # -------------------------------------------------------------------

    def _apply_full(self, model: nn.Module) -> nn.Module:
        """Wrap every direct child module in CheckpointWrapper."""
        count = 0
        for name, child in list(model.named_children()):
            if isinstance(child, CheckpointWrapper):
                logger.debug("Skipping %s — already wrapped", name)
                continue
            if self._is_excluded(name):
                logger.debug("Skipping %s — matches exclude pattern", name)
                continue
            setattr(
                model,
                name,
                CheckpointWrapper(
                    child,
                    use_reentrant=self.config.use_reentrant,
                    preserve_rng_state=self.config.preserve_rng_state,
                ),
            )
            self._checkpointed_layers.append(name)
            count += 1
        logger.info("Full checkpointing: wrapped %d layers", count)
        return model

    def _apply_selective(
        self,
        model: nn.Module,
        profile_report: Optional[Any],
        sample_input: Optional[Dict[str, torch.Tensor]],
    ) -> nn.Module:
        """Wrap only layers above memory threshold."""
        # Determine which layers to checkpoint
        layers_to_checkpoint: Set[str] = set()

        # Always-include patterns
        include_patterns = list(getattr(self.config, "include_patterns", []) or [])
        exclude_patterns = list(getattr(self.config, "exclude_patterns", []) or [])

        # Add always-included layers
        for name, _ in model.named_children():
            if any(re.search(pat, name) for pat in include_patterns):
                layers_to_checkpoint.add(name)

        # Profile-based selection
        if profile_report is not None:
            # Use provided report
            for lp in profile_report.layers:
                if lp.activation_memory_mb > self.config.memory_threshold_mb:
                    layers_to_checkpoint.add(lp.name)
        elif (
            self.profiler is not None
            and getattr(self.config, "profile_before_apply", True)
            and sample_input is not None
        ):
            # Profile on the fly
            report = self.profiler.profile(
                sample_input,
                num_runs=getattr(self.config, "profile_num_runs", 3),
            )
            for lp in report.layers:
                if lp.activation_memory_mb > self.config.memory_threshold_mb:
                    layers_to_checkpoint.add(lp.name)
        else:
            # Fallback: use parameter count as heuristic
            logger.warning(
                "No profile report or profiler available. "
                "Using parameter count heuristic for selective checkpointing."
            )
            param_counts = {}
            for name, child in model.named_children():
                param_counts[name] = sum(p.numel() for p in child.parameters())
            if param_counts:
                avg_params = sum(param_counts.values()) / len(param_counts)
                for name, count in param_counts.items():
                    if count > avg_params:
                        layers_to_checkpoint.add(name)

        # Remove excluded layers
        for name in list(layers_to_checkpoint):
            if any(re.search(pat, name) for pat in exclude_patterns):
                layers_to_checkpoint.discard(name)
                logger.debug("Excluding %s from checkpointing (matches exclude pattern)", name)

        # Apply wrapping
        count = 0
        for name, child in list(model.named_children()):
            if name not in layers_to_checkpoint:
                continue
            if isinstance(child, CheckpointWrapper):
                continue
            setattr(
                model,
                name,
                CheckpointWrapper(
                    child,
                    use_reentrant=self.config.use_reentrant,
                    preserve_rng_state=self.config.preserve_rng_state,
                ),
            )
            self._checkpointed_layers.append(name)
            count += 1

        logger.info(
            "Selective checkpointing: wrapped %d of %d layers (threshold=%.1f MB)",
            count,
            len(list(model.named_children())),
            self.config.memory_threshold_mb,
        )
        return model

    def _apply_sequential(self, model: nn.Module) -> nn.Module:
        """
        Apply checkpoint_sequential to nn.Sequential sub-modules.

        For non-Sequential models, falls back to selective strategy.
        """
        # Find Sequential sub-modules
        sequential_found = False
        for name, child in model.named_children():
            if isinstance(child, nn.Sequential):
                segments = getattr(self.config, "sequential_segments", None)
                if segments is None:
                    segments = max(1, int(math.sqrt(len(child))))
                logger.info(
                    "checkpoint_sequential on %s: %d layers, %d segments",
                    name, len(child), segments,
                )
                # Replace the forward method to use checkpoint_sequential
                original_seq = child
                seq_segments = segments

                class SequentialCheckpointed(nn.Module):
                    def __init__(self, seq, segs):
                        super().__init__()
                        self.seq = seq
                        self.segments = segs

                    def forward(self, x):
                        if torch.is_grad_enabled():
                            return checkpoint_sequential(
                                self.seq, self.segments, x
                            )
                        return self.seq(x)

                setattr(model, name, SequentialCheckpointed(original_seq, seq_segments))
                self._checkpointed_layers.append(name)
                sequential_found = True

        if not sequential_found:
            logger.warning(
                "No nn.Sequential children found. Falling back to selective strategy."
            )
            return self._apply_selective(model, None, None)

        return model

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _is_excluded(self, name: str) -> bool:
        """Check if a layer name matches any exclude pattern."""
        exclude_patterns = list(getattr(self.config, "exclude_patterns", []) or [])
        return any(re.search(pat, name) for pat in exclude_patterns)


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from dataclasses import dataclass, field as dc_field
    from typing import List as ListType

    failures: ListType[str] = []

    def _check(test_name: str, condition: bool, msg: str = "") -> None:
        if condition:
            print(f"  PASS  {test_name}")
        else:
            print(f"  FAIL  {test_name}: {msg}")
            failures.append(test_name)

    print("=" * 60)
    print("SelectiveCheckpointer self-tests")
    print("=" * 60)

    torch.manual_seed(42)

    # --- Minimal config for testing ---
    @dataclass
    class TestConfig:
        enabled: bool = True
        strategy: str = "selective"
        memory_threshold_mb: float = 0.01
        use_reentrant: bool = False
        preserve_rng_state: bool = True
        profile_before_apply: bool = False
        exclude_patterns: ListType[str] = dc_field(default_factory=list)
        include_patterns: ListType[str] = dc_field(default_factory=list)
        sequential_segments: Optional[int] = None

    # --- Test model ---
    class MultiLayerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.small = nn.Linear(8, 8)       # small layer
            self.big1 = nn.Linear(8, 256)      # big layer
            self.big2 = nn.Linear(256, 256)    # big layer
            self.norm = nn.LayerNorm(256)
            self.head = nn.Linear(256, 4)

        def forward(self, x):
            x = torch.relu(self.small(x))
            x = torch.relu(self.big1(x))
            x = torch.relu(self.big2(x))
            x = self.norm(x)
            return self.head(x)

    # --- Test 1: strategy="none" wraps nothing ---
    try:
        cfg = TestConfig(strategy="none")
        sc = SelectiveCheckpointer(cfg)
        model = MultiLayerModel()
        sc.apply(model)
        _check(
            "strategy_none",
            len(sc.get_checkpointed_layers()) == 0,
            f"got {sc.get_checkpointed_layers()}",
        )
    except Exception as e:
        _check("strategy_none", False, str(e))

    # --- Test 2: strategy="full" wraps all children ---
    try:
        cfg = TestConfig(strategy="full")
        sc = SelectiveCheckpointer(cfg)
        model = MultiLayerModel()
        sc.apply(model)
        children_count = len(list(model.named_children()))
        _check(
            "strategy_full",
            len(sc.get_checkpointed_layers()) == children_count,
            f"wrapped {len(sc.get_checkpointed_layers())} of {children_count}",
        )
    except Exception as e:
        _check("strategy_full", False, str(e))

    # --- Test 3: enabled=False wraps nothing ---
    try:
        cfg = TestConfig(enabled=False, strategy="full")
        sc = SelectiveCheckpointer(cfg)
        model = MultiLayerModel()
        sc.apply(model)
        _check(
            "disabled_wraps_nothing",
            len(sc.get_checkpointed_layers()) == 0,
        )
    except Exception as e:
        _check("disabled_wraps_nothing", False, str(e))

    # --- Test 4: Selective with heuristic (no profiler) ---
    try:
        cfg = TestConfig(strategy="selective", memory_threshold_mb=0.01)
        sc = SelectiveCheckpointer(cfg)
        model = MultiLayerModel()
        sc.apply(model)
        # Should checkpoint at least the big layers (above-average params)
        ckpt = sc.get_checkpointed_layers()
        _check(
            "selective_heuristic",
            len(ckpt) > 0 and len(ckpt) < 5,
            f"checkpointed: {ckpt}",
        )
    except Exception as e:
        _check("selective_heuristic", False, str(e))

    # --- Test 5: exclude_patterns removes layers ---
    try:
        cfg = TestConfig(strategy="full", exclude_patterns=["norm", "head"])
        sc = SelectiveCheckpointer(cfg)
        model = MultiLayerModel()
        sc.apply(model)
        ckpt = sc.get_checkpointed_layers()
        _check(
            "exclude_patterns",
            "norm" not in ckpt and "head" not in ckpt and len(ckpt) > 0,
            f"checkpointed: {ckpt}",
        )
    except Exception as e:
        _check("exclude_patterns", False, str(e))

    # --- Test 6: Full checkpointing produces correct gradients ---
    try:
        import copy

        model_orig = MultiLayerModel()
        model_ckpt = copy.deepcopy(model_orig)

        cfg = TestConfig(strategy="full")
        sc = SelectiveCheckpointer(cfg)
        sc.apply(model_ckpt)

        x = torch.randn(4, 8, requires_grad=True)
        x_ckpt = x.detach().clone().requires_grad_(True)

        out_orig = model_orig(x)
        out_orig.sum().backward()

        out_ckpt = model_ckpt(x_ckpt)
        out_ckpt.sum().backward()

        grads_ok = True
        for (n1, p1), (n2, p2) in zip(
            model_orig.named_parameters(), model_ckpt.named_parameters()
        ):
            if p1.grad is None or p2.grad is None:
                grads_ok = False
                break
            if not torch.allclose(p1.grad, p2.grad, atol=1e-5):
                grads_ok = False
                break

        _check("full_gradient_correctness", grads_ok)
    except Exception as e:
        _check("full_gradient_correctness", False, str(e))

    # --- Test 7: Sequential strategy ---
    try:
        class SeqModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(16, 32),
                    nn.ReLU(),
                    nn.Linear(32, 64),
                    nn.ReLU(),
                    nn.Linear(64, 16),
                )

            def forward(self, x):
                return self.layers(x)

        cfg = TestConfig(strategy="sequential", sequential_segments=2)
        sc = SelectiveCheckpointer(cfg)
        seq_model = SeqModel()
        sc.apply(seq_model)
        _check(
            "sequential_strategy",
            len(sc.get_checkpointed_layers()) > 0,
            f"checkpointed: {sc.get_checkpointed_layers()}",
        )

        # Verify forward still works
        x_seq = torch.randn(2, 16, requires_grad=True)
        out_seq = seq_model(x_seq)
        out_seq.sum().backward()
        _check("sequential_forward_backward", x_seq.grad is not None)
    except Exception as e:
        _check("sequential_strategy", False, str(e))

    # --- Test 8: get_checkpointed_layers returns correct names ---
    try:
        cfg = TestConfig(strategy="full")
        sc = SelectiveCheckpointer(cfg)
        model = MultiLayerModel()
        sc.apply(model)
        ckpt = sc.get_checkpointed_layers()
        # All children should be listed
        children = [name for name, _ in model.named_children()]
        _check(
            "get_checkpointed_layers",
            set(ckpt) == set(children),
            f"got {ckpt}, expected {children}",
        )
    except Exception as e:
        _check("get_checkpointed_layers", False, str(e))

    # --- Test 9: include_patterns forces checkpointing ---
    try:
        cfg = TestConfig(
            strategy="selective",
            memory_threshold_mb=1e6,  # very high threshold
            include_patterns=["small"],
        )
        sc = SelectiveCheckpointer(cfg)
        model = MultiLayerModel()
        sc.apply(model)
        ckpt = sc.get_checkpointed_layers()
        _check(
            "include_patterns",
            "small" in ckpt,
            f"checkpointed: {ckpt}",
        )
    except Exception as e:
        _check("include_patterns", False, str(e))

    # --- Test 10: Empty model does not crash ---
    try:
        class EmptyModel(nn.Module):
            def forward(self, x):
                return x

        cfg = TestConfig(strategy="full")
        sc = SelectiveCheckpointer(cfg)
        em = EmptyModel()
        sc.apply(em)
        _check("empty_model", len(sc.get_checkpointed_layers()) == 0)
    except Exception as e:
        _check("empty_model", False, str(e))

    print()
    if failures:
        print(f"FAILED: {len(failures)} tests: {failures}")
        sys.exit(1)
    else:
        print("All 10 tests passed.")
