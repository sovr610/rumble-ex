"""
compile_wrap_template.py
------------------------
Safe wrapper around torch.compile with logging, timing, allowlist/blocklist
filtering, smoketest health check, and configurable fallback policy.

The central function is maybe_compile(). It is the single entry point for
all compilation in the training stack. It never crashes by default — any
compilation or smoketest failure results in a logged warning and the
original eager model being returned (when fail_policy="fallback_eager").

Usage:
    from compile_config_template import CompileConfig
    from compile_wrap_template import maybe_compile
    import logging

    cfg = CompileConfig(enabled=True, mode="default", healthcheck=True)
    model = maybe_compile(model, cfg, logger=logging.getLogger(__name__),
                          sample_batch={"input_ids": ..., "labels": ...})
"""

from __future__ import annotations

import fnmatch
import logging
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

# Import config from companion file (adjust import path as needed)
try:
    from compile_config_template import CompileConfig
except ImportError:
    # Fallback: define a minimal CompileConfig stub for standalone use
    from dataclasses import dataclass, field
    from typing import List, Optional, Dict, Any

    @dataclass
    class CompileConfig:  # type: ignore[no-redef]
        enabled: bool = False
        mode: str = "default"
        dynamic: Optional[bool] = None
        backend: str = "inductor"
        fullgraph: bool = False
        options: Optional[Dict[str, Any]] = None
        allowlist: List[str] = field(default_factory=list)
        blocklist: List[str] = field(default_factory=list)
        healthcheck: bool = True
        fail_policy: str = "fallback_eager"
        smoketest_steps: int = 3


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _matches_any(name: str, patterns: List[str]) -> bool:
    """Return True if name matches any of the glob/substring patterns."""
    for pattern in patterns:
        if fnmatch.fnmatch(name, pattern) or pattern in name:
            return True
    return False


def _apply_blocklist(
    model: nn.Module,
    blocklist: List[str],
    logger: logging.Logger,
) -> nn.Module:
    """
    Apply torch.compiler.disable to all submodules whose names match
    any pattern in blocklist. This prevents compilation of those modules
    while allowing the parent module to be compiled.

    The disable is applied recursively (recursive=True) meaning neither
    the module nor any of its callees will be compiled.
    """
    if not blocklist:
        return model

    disabled_count = 0
    for name, module in model.named_modules():
        if not name:
            continue  # skip the root module itself
        if _matches_any(name, blocklist):
            try:
                # Apply disable in-place by replacing the module's forward
                original_forward = module.forward

                @torch.compiler.disable
                def _disabled_forward(*args, _fwd=original_forward, **kwargs):
                    return _fwd(*args, **kwargs)

                module.forward = _disabled_forward
                disabled_count += 1
                logger.debug("compile: blocklist disabled module '%s'", name)
            except Exception as e:
                logger.warning(
                    "compile: failed to apply blocklist to module '%s': %s", name, e
                )

    if disabled_count > 0:
        logger.info("compile: blocklist disabled %d submodule(s)", disabled_count)

    return model


def _apply_allowlist_compile(
    model: nn.Module,
    allowlist: List[str],
    compile_kwargs: Dict[str, Any],
    logger: logging.Logger,
) -> nn.Module:
    """
    When allowlist is non-empty, compile only the matching submodules.
    All other submodules are left in eager mode.

    Returns the model with matching submodules replaced by their compiled versions.
    """
    compiled_count = 0

    # Collect top-level matching submodules to avoid double-compiling
    # (don't compile parent and child separately — compile the parent)
    compiled_names: set = set()

    for name, module in model.named_modules():
        if not name:
            continue
        if _matches_any(name, allowlist):
            # Don't compile a child if an ancestor is already being compiled
            is_child_of_compiled = any(
                name.startswith(cn + ".") for cn in compiled_names
            )
            if is_child_of_compiled:
                continue

            # Navigate to parent module and replace attribute
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            attr_name = parts[-1]

            try:
                compiled_module = torch.compile(module, **compile_kwargs)
                setattr(parent, attr_name, compiled_module)
                compiled_names.add(name)
                compiled_count += 1
                logger.debug("compile: compiled submodule '%s'", name)
            except Exception as e:
                logger.warning(
                    "compile: failed to compile allowlisted module '%s': %s", name, e
                )

    if compiled_count > 0:
        logger.info(
            "compile: allowlist compiled %d submodule(s): %s",
            compiled_count,
            sorted(compiled_names),
        )
    else:
        logger.warning(
            "compile: allowlist %r matched no modules in model. "
            "Available modules: %s",
            allowlist,
            [n for n, _ in model.named_modules() if n],
        )

    return model


def _run_smoketest(
    model: nn.Module,
    sample_batch: Dict[str, Any],
    steps: int,
    logger: logging.Logger,
) -> bool:
    """
    Run a minimal smoketest: steps forward + backward passes.
    Returns True if all steps succeed, False otherwise.

    Creates a simple SGD optimizer internally. For a more thorough test
    that uses the real optimizer and loss function, use CompileSmoketest
    directly.
    """
    import torch.optim as optim

    optimizer = optim.SGD(model.parameters(), lr=1e-4)

    for step in range(steps):
        try:
            optimizer.zero_grad()

            # Move batch to model's device
            device = next(model.parameters()).device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in sample_batch.items()
            }

            # Forward pass — try various calling conventions
            try:
                output = model(**batch)
            except TypeError:
                # Try positional if kwargs fail
                vals = [v for v in batch.values() if isinstance(v, torch.Tensor)]
                output = model(*vals)

            # Compute a scalar loss
            if isinstance(output, torch.Tensor):
                loss = output.float().mean()
            elif hasattr(output, "loss"):
                loss = output.loss
            elif isinstance(output, (tuple, list)):
                loss = output[0].float().mean()
            else:
                raise RuntimeError(
                    f"Cannot compute loss from output type {type(output)}"
                )

            # Check for NaN loss
            if torch.isnan(loss):
                logger.error(
                    "compile: smoketest step %d produced NaN loss", step + 1
                )
                return False

            loss.backward()
            optimizer.step()

        except Exception as e:
            logger.error(
                "compile: smoketest failed at step %d: %s\n%s",
                step + 1,
                e,
                traceback.format_exc(),
            )
            return False

    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def maybe_compile(
    model: nn.Module,
    cfg: "CompileConfig",
    logger: logging.Logger,
    sample_batch: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """
    Safe entry point for torch.compile integration.

    Behavior
    --------
    1. If cfg.enabled is False, return model unchanged (no-op).
    2. Apply blocklist: decorate matching submodules with torch.compiler.disable.
    3. If cfg.allowlist is non-empty, compile only matching submodules.
       Otherwise, torch.compile the full model.
    4. On any compile exception:
       - Log the exception with full stack trace.
       - If cfg.fail_policy == "fallback_eager": return original model.
       - If cfg.fail_policy == "raise": re-raise.
    5. If cfg.healthcheck is True and sample_batch is provided:
       - Run a smoketest of cfg.smoketest_steps forward-backward steps.
       - If smoketest fails: log error, apply fail_policy.
    6. Log: enabled, backend, mode, dynamic, fullgraph, success/failure,
       compile_wall_time_s, exception (if any).
    7. Return compiled model on success.

    Parameters
    ----------
    model:
        The nn.Module to compile. Not modified in place if compilation fails.
    cfg:
        CompileConfig instance controlling all compilation behavior.
    logger:
        Logger instance. All compilation events are logged here.
        Use logging.getLogger(__name__) or pass a named logger.
    sample_batch:
        Optional dict of sample inputs (e.g., {"input_ids": ..., "labels": ...}).
        Required for smoketest when cfg.healthcheck=True.
        If None and healthcheck=True, smoketest is skipped with a warning.

    Returns
    -------
    nn.Module
        Compiled model on success, original eager model on failure (if
        fail_policy="fallback_eager").
    """
    if not cfg.enabled:
        logger.debug("compile: disabled (cfg.enabled=False), returning model unchanged")
        return model

    logger.info(
        "compile: starting compilation — backend=%s mode=%s dynamic=%s "
        "fullgraph=%s healthcheck=%s fail_policy=%s",
        cfg.backend,
        cfg.mode,
        cfg.dynamic,
        cfg.fullgraph,
        cfg.healthcheck,
        cfg.fail_policy,
    )

    # Build kwargs for torch.compile
    compile_kwargs: Dict[str, Any] = {
        "backend": cfg.backend,
        "mode": cfg.mode,
        "fullgraph": cfg.fullgraph,
    }
    if cfg.dynamic is not None:
        compile_kwargs["dynamic"] = cfg.dynamic
    if cfg.options:
        compile_kwargs["options"] = cfg.options

    # Keep reference to original model for fallback
    original_model = model

    try:
        # Step 1: Apply blocklist (must happen before compile call)
        if cfg.blocklist:
            model = _apply_blocklist(model, cfg.blocklist, logger)

        # Step 2: Compile
        if cfg.allowlist:
            # Selective compilation: compile only matching submodules
            logger.info(
                "compile: selective compilation — allowlist=%r", cfg.allowlist
            )
            model = _apply_allowlist_compile(
                model, cfg.allowlist, compile_kwargs, logger
            )
            compile_time_s = 0.0  # Time measured on first forward call
        else:
            # Full model compilation
            t0 = time.perf_counter()
            model = torch.compile(model, **compile_kwargs)
            compile_time_s = time.perf_counter() - t0
            logger.info(
                "compile: torch.compile() call returned in %.3fs "
                "(actual JIT compilation deferred to first forward pass)",
                compile_time_s,
            )

    except Exception as exc:
        logger.error(
            "compile: torch.compile() raised an exception:\n%s",
            traceback.format_exc(),
        )
        return _apply_fail_policy(
            exc=exc,
            cfg=cfg,
            original_model=original_model,
            logger=logger,
            stage="compile",
        )

    # Step 3: Smoketest / health check
    if cfg.healthcheck:
        if sample_batch is None:
            logger.warning(
                "compile: healthcheck=True but no sample_batch provided — "
                "skipping smoketest. Pass sample_batch to maybe_compile for "
                "full health verification."
            )
        else:
            logger.info(
                "compile: running smoketest (%d steps)...", cfg.smoketest_steps
            )
            t_smoke_start = time.perf_counter()
            success = _run_smoketest(
                model=model,
                sample_batch=sample_batch,
                steps=cfg.smoketest_steps,
                logger=logger,
            )
            smoke_time_s = time.perf_counter() - t_smoke_start

            if not success:
                logger.error(
                    "compile: smoketest FAILED after %.3fs — applying fail_policy=%s",
                    smoke_time_s,
                    cfg.fail_policy,
                )
                return _apply_fail_policy(
                    exc=RuntimeError("CompileSmoketest failed"),
                    cfg=cfg,
                    original_model=original_model,
                    logger=logger,
                    stage="smoketest",
                )
            else:
                logger.info(
                    "compile: smoketest PASSED in %.3fs (includes first-step JIT compilation)",
                    smoke_time_s,
                )

    logger.info(
        "compile: compilation SUCCESSFUL — backend=%s mode=%s dynamic=%s fullgraph=%s",
        cfg.backend,
        cfg.mode,
        cfg.dynamic,
        cfg.fullgraph,
    )
    return model


def _apply_fail_policy(
    exc: Exception,
    cfg: "CompileConfig",
    original_model: nn.Module,
    logger: logging.Logger,
    stage: str,
) -> nn.Module:
    """Apply fail_policy after a compilation or smoketest failure."""
    if cfg.fail_policy == "raise":
        logger.error(
            "compile: fail_policy=raise — propagating %s exception from stage '%s'",
            type(exc).__name__,
            stage,
        )
        raise exc
    else:  # fallback_eager
        logger.warning(
            "compile: fail_policy=fallback_eager — returning original eager model "
            "after %s failure in stage '%s'",
            type(exc).__name__,
            stage,
        )
        return original_model


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    test_logger = logging.getLogger("compile_wrap_test")

    failures: list = []

    def _check(name: str, condition: bool, msg: str = "") -> None:
        if condition:
            print(f"  PASS  {name}")
        else:
            print(f"  FAIL  {name}: {msg}")
            failures.append(name)

    print("=" * 60)
    print("maybe_compile self-tests")
    print("=" * 60)

    # Simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(16, 8)
            self.out = nn.Linear(8, 4)

        def forward(self, x):
            return self.out(torch.relu(self.linear(x)))

    # Test 1: Disabled config returns same model
    model = SimpleModel()
    cfg = CompileConfig(enabled=False)
    result = maybe_compile(model, cfg, test_logger)
    _check(
        "disabled_returns_same_model",
        result is model,
        f"Expected same model object, got {type(result)}",
    )

    # Test 2: Enabled config compiles simple model
    try:
        model = SimpleModel()
        cfg = CompileConfig(enabled=True, mode="default", healthcheck=False)
        result = maybe_compile(model, cfg, test_logger)
        # torch.compile returns a wrapped model (not the same object)
        # Run a forward pass to verify it works
        x = torch.randn(4, 16)
        out = result(x)
        _check(
            "enabled_compiles_simple_model",
            out.shape == (4, 4),
            f"Output shape {out.shape} != (4, 4)",
        )
    except Exception as e:
        _check("enabled_compiles_simple_model", False, str(e))

    # Test 3: Broken model falls back gracefully (fail_policy=fallback_eager)
    try:
        model = SimpleModel()
        cfg_fallback = CompileConfig(
            enabled=True,
            mode="default",
            healthcheck=False,
            fail_policy="fallback_eager",
        )

        # Monkey-patch torch.compile to simulate failure
        original_compile = torch.compile

        def _mock_compile_fail(m, **kwargs):
            raise RuntimeError("Simulated compilation failure")

        torch.compile = _mock_compile_fail
        try:
            result = maybe_compile(model, cfg_fallback, test_logger)
            _check(
                "broken_model_fallback",
                result is model,
                f"Expected original model, got {type(result)}",
            )
        finally:
            torch.compile = original_compile
    except Exception as e:
        _check("broken_model_fallback", False, str(e))

    # Test 4: fail_policy=raise propagates exception
    try:
        model = SimpleModel()
        cfg_raise = CompileConfig(
            enabled=True,
            mode="default",
            healthcheck=False,
            fail_policy="raise",
        )
        original_compile = torch.compile

        def _mock_compile_fail2(m, **kwargs):
            raise RuntimeError("Simulated failure for raise test")

        torch.compile = _mock_compile_fail2
        raised = False
        try:
            maybe_compile(model, cfg_raise, test_logger)
        except RuntimeError:
            raised = True
        finally:
            torch.compile = original_compile
        _check("fail_policy_raise_propagates", raised)
    except Exception as e:
        _check("fail_policy_raise_propagates", False, str(e))

    # Test 5: Smoketest with good model passes
    try:
        model = SimpleModel()
        cfg_smoke = CompileConfig(
            enabled=True,
            mode="default",
            healthcheck=True,
            fail_policy="fallback_eager",
            smoketest_steps=2,
        )
        sample = {"x": torch.randn(4, 16)}
        result = maybe_compile(model, cfg_smoke, test_logger, sample_batch=sample)
        x = torch.randn(4, 16)
        out = result(x)
        _check(
            "smoketest_good_model_passes",
            out.shape == (4, 4),
            f"Output shape {out.shape} != (4, 4)",
        )
    except Exception as e:
        _check("smoketest_good_model_passes", False, str(e))

    # Test 6: Blocklist prevents compilation of matching modules
    try:
        class ModelWithSubmodules(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(16, 8)
                self.sampling = nn.Linear(8, 4)  # should be blocked

            def forward(self, x):
                return self.sampling(torch.relu(self.encoder(x)))

        model = ModelWithSubmodules()
        # Check that blocklist processing doesn't crash
        cfg_block = CompileConfig(
            enabled=True,
            mode="default",
            healthcheck=False,
            blocklist=["sampling"],
        )
        result = maybe_compile(model, cfg_block, test_logger)
        x = torch.randn(4, 16)
        out = result(x)
        _check(
            "blocklist_does_not_crash",
            out.shape == (4, 4),
            f"Output shape {out.shape} != (4, 4)",
        )
    except Exception as e:
        _check("blocklist_does_not_crash", False, str(e))

    # Test 7: Healthcheck=True with no sample_batch logs warning but doesn't crash
    try:
        model = SimpleModel()
        cfg_no_batch = CompileConfig(
            enabled=True,
            mode="default",
            healthcheck=True,
            fail_policy="fallback_eager",
        )
        result = maybe_compile(model, cfg_no_batch, test_logger, sample_batch=None)
        _check(
            "healthcheck_no_sample_batch_no_crash",
            result is not None,
        )
    except Exception as e:
        _check("healthcheck_no_sample_batch_no_crash", False, str(e))

    print()
    if failures:
        print(f"FAILED: {len(failures)} tests: {failures}")
        sys.exit(1)
    else:
        print(f"All {7} tests passed.")
