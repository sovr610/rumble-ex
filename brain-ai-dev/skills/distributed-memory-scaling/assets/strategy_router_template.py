#!/usr/bin/env python3
"""
strategy_router_template.py
-----------------------------
StrategyRouter: single entrypoint enforcing wrapping order invariants for
DDP, FSDP, and DeepSpeed ZeRO-2/3 distributed training.

Wrapping Order Invariants (enforced by state machine):
    1. setup_distributed()       — init process group, set device
    2. wrap_model()              — dispatch to DDP / FSDP / DeepSpeed
    3. build_optimizer()         — over wrapped model's parameters
    4. load_checkpoint() / save_checkpoint() — strategy-aware checkpoint I/O

Violating this order raises RuntimeError with a descriptive message.

Usage
-----
    router = StrategyRouter(DistributedConfig(strategy="fsdp"))
    router.setup_distributed()
    wrapped_model, ctx = router.wrap_model(model)
    optimizer = router.build_optimizer(wrapped_model, optimizer_cfg)
    step = router.load_checkpoint(wrapped_model, optimizer, "/path/ckpt")
    ...training loop...
    router.save_checkpoint(wrapped_model, optimizer, "/path/ckpt", step=100)
    router.export_portable_weights(wrapped_model, "/path/weights.pt")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.distributed as dist
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State machine states
# ---------------------------------------------------------------------------


class RouterState(Enum):
    INITIAL = auto()
    DISTRIBUTED_READY = auto()
    MODEL_WRAPPED = auto()
    OPTIMIZER_BUILT = auto()


# ---------------------------------------------------------------------------
# StrategyContext
# ---------------------------------------------------------------------------


@dataclass
class StrategyContext:
    """Runtime context established after distributed setup and wrapping."""

    strategy: str = "ddp"
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    device: str = "cuda:0"
    is_distributed: bool = False

    def __post_init__(self) -> None:
        self.is_distributed = self.world_size > 1


# ---------------------------------------------------------------------------
# StrategyRouter
# ---------------------------------------------------------------------------


class StrategyRouter:
    """Single entrypoint for distributed training strategy management.

    Enforces the correct wrapping sequence and dispatches to the
    appropriate backend (DDP, FSDP, DeepSpeed) based on config.

    Parameters
    ----------
    cfg:
        DistributedConfig instance (or compatible dataclass with .strategy,
        .world_size, .backend fields).
    fsdp_cfg:
        FSDPConfig instance. Required when cfg.strategy == 'fsdp'.
    ds_cfg:
        DeepSpeedConfig instance. Required when cfg.strategy starts with
        'deepspeed'.
    """

    def __init__(
        self,
        cfg: Any,
        fsdp_cfg: Optional[Any] = None,
        ds_cfg: Optional[Any] = None,
    ) -> None:
        self.cfg = cfg
        self.fsdp_cfg = fsdp_cfg
        self.ds_cfg = ds_cfg
        self._state: RouterState = RouterState.INITIAL
        self._context: Optional[StrategyContext] = None
        self._ds_engine: Optional[Any] = None

        # Validate config combination up front
        self._validate_configs()

    def _validate_configs(self) -> None:
        strategy = self.cfg.strategy
        if strategy == "fsdp" and self.fsdp_cfg is None:
            raise ValueError(
                "FSDPConfig must be provided when strategy='fsdp'."
            )
        if strategy in ("deepspeed_zero2", "deepspeed_zero3") and self.ds_cfg is None:
            raise ValueError(
                f"DeepSpeedConfig must be provided when strategy='{strategy}'."
            )

    # ------------------------------------------------------------------
    # setup_distributed
    # ------------------------------------------------------------------

    def setup_distributed(self) -> None:
        """Initialize the process group and set the current CUDA device.

        Reads environment variables set by the launcher:
          - RANK, WORLD_SIZE, LOCAL_RANK (torchrun / mpirun)

        If not running under a launcher, treats the process as rank 0 /
        world_size 1 (single-GPU mode).

        Raises
        ------
        RuntimeError
            If already called (idempotent guard).
        """
        if self._state != RouterState.INITIAL:
            raise RuntimeError(
                "setup_distributed() has already been called. "
                "Create a new StrategyRouter to re-initialize."
            )

        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        # Override with config world_size if explicitly set
        if self.cfg.world_size != -1:
            world_size = self.cfg.world_size

        # Initialize process group
        if not dist.is_initialized():
            backend = self.cfg.backend
            if not torch.cuda.is_available() and backend == "nccl":
                logger.warning(
                    "CUDA not available, falling back to 'gloo' backend."
                )
                backend = "gloo"
            dist.init_process_group(backend=backend)

        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = f"cuda:{local_rank}"
        else:
            device = "cpu"

        self._context = StrategyContext(
            strategy=self.cfg.strategy,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            device=device,
            is_distributed=world_size > 1,
        )

        self._state = RouterState.DISTRIBUTED_READY
        logger.info(
            "Distributed setup complete: rank=%d/%d, device=%s, strategy=%s",
            rank,
            world_size,
            device,
            self.cfg.strategy,
        )

    # ------------------------------------------------------------------
    # wrap_model
    # ------------------------------------------------------------------

    def wrap_model(
        self,
        model: nn.Module,
    ) -> Tuple[nn.Module, StrategyContext]:
        """Wrap model with the configured distributed strategy.

        Must be called after setup_distributed(). The optimizer must NOT
        be created before this call.

        Parameters
        ----------
        model:
            Unwrapped nn.Module, already placed on the correct device.

        Returns
        -------
        tuple of (wrapped_model, StrategyContext)

        Raises
        ------
        RuntimeError
            If setup_distributed() has not been called first.
        """
        self._assert_state(
            RouterState.DISTRIBUTED_READY,
            "wrap_model() requires setup_distributed() to be called first.",
        )

        strategy = self.cfg.strategy

        if strategy == "ddp":
            wrapped = self._wrap_ddp(model)
        elif strategy == "fsdp":
            wrapped = self._wrap_fsdp(model)
        elif strategy in ("deepspeed_zero2", "deepspeed_zero3"):
            # DeepSpeed wrapping happens in build_optimizer (after param groups
            # are known). Here we just mark the model as ready to wrap.
            # The engine is created in _wrap_deepspeed, called from build_optimizer.
            wrapped = model  # placeholder; real wrap in build_optimizer
        else:
            raise ValueError(f"Unknown strategy '{strategy}'.")

        self._state = RouterState.MODEL_WRAPPED
        assert self._context is not None
        return wrapped, self._context

    def _wrap_ddp(self, model: nn.Module) -> nn.Module:
        """Wrap model with DistributedDataParallel."""
        ctx = self._context
        assert ctx is not None
        if ctx.is_distributed:
            device_ids = None
            if torch.cuda.is_available():
                device_ids = [ctx.local_rank]
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=device_ids,
            )
            logger.info("Model wrapped with DDP (device_ids=%s)", device_ids)
        else:
            logger.info("Single-process mode: DDP wrapping skipped.")
        return model

    def _wrap_fsdp(self, model: nn.Module) -> nn.Module:
        """Wrap model with FSDP using FSDPWrapper."""
        from fsdp_wrapper_template import FSDPWrapper

        wrapper = FSDPWrapper()
        wrapped = wrapper.wrap(model, self.fsdp_cfg)
        self._fsdp_wrapper = wrapper
        return wrapped

    # ------------------------------------------------------------------
    # build_optimizer
    # ------------------------------------------------------------------

    def build_optimizer(
        self,
        model: nn.Module,
        opt_cfg: Any,
    ) -> torch.optim.Optimizer:
        """Create optimizer over wrapped model parameters.

        Must be called after wrap_model(). Creating the optimizer before
        wrap_model() is a critical bug: FSDP changes parameter views and
        the optimizer would reference stale tensors.

        Parameters
        ----------
        model:
            The wrapped model returned by wrap_model().
        opt_cfg:
            OptimizerConfig instance.

        Returns
        -------
        torch.optim.Optimizer

        Raises
        ------
        RuntimeError
            If wrap_model() has not been called first.
        """
        self._assert_state(
            RouterState.MODEL_WRAPPED,
            "build_optimizer() requires wrap_model() to be called first.",
        )

        strategy = self.cfg.strategy

        if strategy in ("deepspeed_zero2", "deepspeed_zero3"):
            # For DeepSpeed, we create a base optimizer first, then pass it
            # to deepspeed.initialize() via _wrap_deepspeed.
            optimizer = self._build_base_optimizer(model, opt_cfg)
            model, optimizer = self._wrap_deepspeed(model, optimizer)
            # Replace the model reference in the router's context
            self._ds_engine = model  # model is now the DS engine
            self._state = RouterState.OPTIMIZER_BUILT
            return optimizer

        optimizer = self._build_base_optimizer(model, opt_cfg)
        self._state = RouterState.OPTIMIZER_BUILT
        return optimizer

    def _build_base_optimizer(
        self,
        model: nn.Module,
        opt_cfg: Any,
    ) -> torch.optim.Optimizer:
        """Build a standard PyTorch optimizer."""
        optimizer_type = opt_cfg.optimizer_type
        params = model.parameters()

        if optimizer_type == "AdamW":
            return torch.optim.AdamW(
                params,
                lr=opt_cfg.lr,
                weight_decay=opt_cfg.weight_decay,
                betas=tuple(opt_cfg.betas),
                eps=opt_cfg.eps,
            )
        elif optimizer_type == "Adam":
            return torch.optim.Adam(
                params,
                lr=opt_cfg.lr,
                betas=tuple(opt_cfg.betas),
                eps=opt_cfg.eps,
            )
        elif optimizer_type == "SGD":
            return torch.optim.SGD(
                params,
                lr=opt_cfg.lr,
                momentum=opt_cfg.momentum,
                weight_decay=opt_cfg.weight_decay,
            )
        else:
            raise ValueError(
                f"Unknown optimizer_type '{optimizer_type}'. "
                "Must be 'AdamW', 'Adam', or 'SGD'."
            )

    def _wrap_deepspeed(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> Tuple[Any, Any]:
        """Initialize DeepSpeed engine."""
        from deepspeed_wrapper_template import DeepSpeedWrapper

        ds_wrapper = DeepSpeedWrapper()
        engine = ds_wrapper.wrap(model, self.ds_cfg, optimizer=optimizer)
        self._ds_wrapper = ds_wrapper
        return engine, optimizer

    # ------------------------------------------------------------------
    # save_checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        path: str,
        step: int = 0,
    ) -> None:
        """Save a checkpoint using the strategy-appropriate method.

        Parameters
        ----------
        model:
            Wrapped model (or DS engine for DeepSpeed).
        optimizer:
            Optimizer (or DS engine handles this internally).
        path:
            Path or directory for the checkpoint.
        step:
            Training step for metadata.

        Raises
        ------
        RuntimeError
            If build_optimizer() has not been called (training not started).
        """
        self._assert_state(
            RouterState.OPTIMIZER_BUILT,
            "save_checkpoint() requires build_optimizer() to be called first.",
        )

        strategy = self.cfg.strategy

        if strategy == "ddp":
            self._save_ddp_checkpoint(model, optimizer, path, step)
        elif strategy == "fsdp":
            self._save_fsdp_checkpoint(model, optimizer, path, step)
        elif strategy in ("deepspeed_zero2", "deepspeed_zero3"):
            self._save_deepspeed_checkpoint(model, path, step)
        else:
            raise ValueError(f"Unknown strategy '{strategy}'.")

    def _save_ddp_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        path: str,
        step: int,
    ) -> None:
        """Save DDP checkpoint on rank 0 only."""
        ctx = self._context
        assert ctx is not None
        if ctx.rank == 0:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            # Unwrap DDP module for clean state dict
            module = (
                model.module
                if isinstance(model, nn.parallel.DistributedDataParallel)
                else model
            )
            torch.save(
                {
                    "model": module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                },
                path,
            )
            logger.info("DDP checkpoint saved to %s (step=%d)", path, step)
        if dist.is_initialized():
            dist.barrier()

    def _save_fsdp_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        path: str,
        step: int,
    ) -> None:
        """Save FSDP checkpoint (full or sharded based on config)."""
        assert self.fsdp_cfg is not None
        wrapper = getattr(self, "_fsdp_wrapper", None)
        if wrapper is None:
            from fsdp_wrapper_template import FSDPWrapper
            wrapper = FSDPWrapper()

        if self.fsdp_cfg.state_dict_type == "full":
            wrapper.save_full_state_dict(model, path)
        else:
            wrapper.save_sharded_checkpoint(model, optimizer, path, step=step)

    def _save_deepspeed_checkpoint(
        self,
        engine: Any,
        path: str,
        step: int,
    ) -> None:
        """Save DeepSpeed checkpoint."""
        tag = f"step_{step}"
        wrapper = getattr(self, "_ds_wrapper", None)
        if wrapper is None:
            from deepspeed_wrapper_template import DeepSpeedWrapper
            wrapper = DeepSpeedWrapper()
        wrapper.save_checkpoint(engine, path, tag=tag, client_state={"step": step})

    # ------------------------------------------------------------------
    # load_checkpoint
    # ------------------------------------------------------------------

    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        path: str,
    ) -> Optional[int]:
        """Load a checkpoint using the strategy-appropriate method.

        Parameters
        ----------
        model:
            Wrapped model (or DS engine for DeepSpeed).
        optimizer:
            Optimizer to restore. May be None for weights-only resume.
        path:
            Path or directory for the checkpoint.

        Returns
        -------
        int or None
            Training step from the checkpoint, or None if unavailable.

        Raises
        ------
        RuntimeError
            If wrap_model() has not been called first.
        """
        # Allow loading after wrap_model OR after build_optimizer
        if self._state not in (RouterState.MODEL_WRAPPED, RouterState.OPTIMIZER_BUILT):
            raise RuntimeError(
                "load_checkpoint() requires wrap_model() to be called first."
            )

        strategy = self.cfg.strategy

        if strategy == "ddp":
            return self._load_ddp_checkpoint(model, optimizer, path)
        elif strategy == "fsdp":
            return self._load_fsdp_checkpoint(model, optimizer, path)
        elif strategy in ("deepspeed_zero2", "deepspeed_zero3"):
            return self._load_deepspeed_checkpoint(model, path)
        else:
            raise ValueError(f"Unknown strategy '{strategy}'.")

    def _load_ddp_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        path: str,
    ) -> Optional[int]:
        """Load DDP checkpoint."""
        ctx = self._context
        assert ctx is not None
        if not os.path.exists(path):
            logger.info("No checkpoint found at %s, starting fresh.", path)
            return None

        map_location = {"cuda:0": f"cuda:{ctx.local_rank}"}
        ckpt = torch.load(path, map_location=map_location)

        module = (
            model.module
            if isinstance(model, nn.parallel.DistributedDataParallel)
            else model
        )
        module.load_state_dict(ckpt["model"])

        if optimizer is not None and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])

        step = ckpt.get("step")
        logger.info("DDP checkpoint loaded from %s (step=%s)", path, step)
        return step

    def _load_fsdp_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        path: str,
    ) -> Optional[int]:
        """Load FSDP checkpoint."""
        assert self.fsdp_cfg is not None
        wrapper = getattr(self, "_fsdp_wrapper", None)
        if wrapper is None:
            from fsdp_wrapper_template import FSDPWrapper
            wrapper = FSDPWrapper()

        if self.fsdp_cfg.state_dict_type == "full":
            if not os.path.exists(path):
                logger.info(
                    "No FSDP full checkpoint found at %s, starting fresh.", path
                )
                return None
            wrapper.load_full_state_dict(model, path)
            return None  # Full state dict does not store step
        else:
            if not os.path.isdir(path):
                logger.info(
                    "No FSDP sharded checkpoint at %s, starting fresh.", path
                )
                return None
            assert optimizer is not None, (
                "optimizer must be provided for sharded checkpoint load."
            )
            return wrapper.load_sharded_checkpoint(model, optimizer, path)

    def _load_deepspeed_checkpoint(
        self,
        engine: Any,
        path: str,
    ) -> Optional[int]:
        """Load DeepSpeed checkpoint."""
        if not os.path.isdir(path):
            logger.info(
                "No DeepSpeed checkpoint at %s, starting fresh.", path
            )
            return None
        wrapper = getattr(self, "_ds_wrapper", None)
        if wrapper is None:
            from deepspeed_wrapper_template import DeepSpeedWrapper
            wrapper = DeepSpeedWrapper()
        step, client_state = wrapper.load_checkpoint(engine, path, tag=None)
        return step

    # ------------------------------------------------------------------
    # export_portable_weights
    # ------------------------------------------------------------------

    def export_portable_weights(
        self,
        model: nn.Module,
        path: str,
    ) -> None:
        """Export weights in a format loadable by any strategy.

        - DDP: saves model.state_dict() directly (rank 0 only)
        - FSDP: saves FULL_STATE_DICT with offload_to_cpu + rank0_only
        - DeepSpeed: calls zero_to_fp32 consolidation

        Parameters
        ----------
        model:
            Wrapped model or DS engine.
        path:
            Output file or directory path.
        """
        strategy = self.cfg.strategy

        if strategy == "ddp":
            ctx = self._context
            assert ctx is not None
            if ctx.rank == 0:
                module = (
                    model.module
                    if isinstance(model, nn.parallel.DistributedDataParallel)
                    else model
                )
                torch.save(module.state_dict(), path)
                logger.info("DDP portable weights saved to %s", path)
            if dist.is_initialized():
                dist.barrier()

        elif strategy == "fsdp":
            wrapper = getattr(self, "_fsdp_wrapper", None)
            if wrapper is None:
                from fsdp_wrapper_template import FSDPWrapper
                wrapper = FSDPWrapper()
            wrapper.save_full_state_dict(model, path)

        elif strategy in ("deepspeed_zero2", "deepspeed_zero3"):
            # path must be a directory (checkpoint tag) for the source
            # and an output file path for the fp32 weights
            wrapper = getattr(self, "_ds_wrapper", None)
            if wrapper is None:
                from deepspeed_wrapper_template import DeepSpeedWrapper
                wrapper = DeepSpeedWrapper()
            # Expect path to be "<checkpoint_dir>:<output_fp32_path>"
            # or, if path is a single string, treat it as output_fp32_path
            # and use self._last_checkpoint_dir.
            if ":" in path and not path.startswith("/"):
                ckpt_dir, out_path = path.split(":", 1)
            else:
                ckpt_dir = getattr(self, "_last_checkpoint_dir", path)
                out_path = path
            wrapper.export_fp32_weights(ckpt_dir, out_path)

        else:
            raise ValueError(f"Unknown strategy '{strategy}'.")

    # ------------------------------------------------------------------
    # Internal state machine helpers
    # ------------------------------------------------------------------

    def _assert_state(self, required: RouterState, message: str) -> None:
        """Raise RuntimeError if not in the required state."""
        if self._state != required:
            raise RuntimeError(
                f"{message} "
                f"(current state: {self._state.name}, "
                f"required: {required.name})"
            )

    @property
    def context(self) -> Optional[StrategyContext]:
        """Return the current StrategyContext, or None if not yet set up."""
        return self._context

    @property
    def state(self) -> RouterState:
        """Return the current router state."""
        return self._state


# ---------------------------------------------------------------------------
# Self-Tests
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys
    from dataclasses import dataclass
    from unittest.mock import MagicMock, patch

    failures: List[str] = []

    def check(name: str, condition: bool) -> None:
        if condition:
            print(f"  PASS  {name}")
        else:
            print(f"  FAIL  {name}")
            failures.append(name)

    def expect_raises(name: str, exc_type: type, fn) -> None:
        try:
            fn()
            print(f"  FAIL  {name} (no exception raised)")
            failures.append(name)
        except exc_type:
            print(f"  PASS  {name}")
        except Exception as e:
            print(f"  FAIL  {name} (wrong exception {type(e).__name__}: {e})")
            failures.append(name)

    # Minimal config stubs
    @dataclass
    class MockDistCfg:
        strategy: str = "ddp"
        world_size: int = 1
        backend: str = "gloo"
        grad_accum: int = 1

    @dataclass
    class MockFSDPCfg:
        sharding_strategy: str = "FULL_SHARD"
        wrap_policy: str = "size_based"
        wrap_module_classes: list = None
        mixed_precision: str = "none"
        activation_checkpointing: str = "off"
        state_dict_type: str = "full"
        sync_module_states: bool = False
        cpu_offload: bool = False
        min_num_params: int = 1

        def __post_init__(self):
            if self.wrap_module_classes is None:
                self.wrap_module_classes = []

    @dataclass
    class MockDSCfg:
        zero_stage: int = 3
        offload_optimizer: str = "none"
        offload_param: str = "none"
        reduce_bucket_size: int = 500_000_000
        allgather_bucket_size: int = 500_000_000
        stage3_prefetch_bucket_size: int = 50_000_000
        stage3_param_persistence_threshold: int = 100_000
        stage3_max_live_parameters: int = 1_000_000_000
        stage3_max_reuse_distance: int = 1_000_000_000
        overlap_comm: bool = True
        contiguous_gradients: bool = True
        nvme_path: Any = None
        fp16: bool = False
        bf16: bool = False
        gradient_clipping: float = 1.0

        def validate(self):
            pass

    @dataclass
    class MockOptCfg:
        optimizer_type: str = "AdamW"
        lr: float = 1e-4
        weight_decay: float = 0.01
        betas: tuple = (0.9, 0.95)
        eps: float = 1e-8
        momentum: float = 0.9
        grad_clip: float = 1.0

    # ------------------------------------------------------------------
    # Test: Wrapping order enforcement
    # ------------------------------------------------------------------
    print("=== Wrapping Order Enforcement tests ===")

    # wrap_model before setup_distributed raises
    router_bad = StrategyRouter(MockDistCfg(strategy="ddp"))
    model_stub = nn.Linear(4, 4)
    expect_raises(
        "wrap_before_setup_raises",
        RuntimeError,
        lambda: router_bad.wrap_model(model_stub),
    )

    # build_optimizer before wrap_model raises
    router_bad2 = StrategyRouter(MockDistCfg(strategy="ddp"))
    # Manually force state to DISTRIBUTED_READY to simulate partial init
    router_bad2._state = RouterState.DISTRIBUTED_READY
    router_bad2._context = StrategyContext(strategy="ddp", world_size=1)
    expect_raises(
        "build_optimizer_before_wrap_raises",
        RuntimeError,
        lambda: router_bad2.build_optimizer(model_stub, MockOptCfg()),
    )

    # load_checkpoint before wrap_model raises
    router_bad3 = StrategyRouter(MockDistCfg(strategy="ddp"))
    router_bad3._state = RouterState.DISTRIBUTED_READY
    router_bad3._context = StrategyContext(strategy="ddp", world_size=1)
    expect_raises(
        "load_before_wrap_raises",
        RuntimeError,
        lambda: router_bad3.load_checkpoint(model_stub, None, "/fake/path"),
    )

    # ------------------------------------------------------------------
    # Test: DDP path (mocked dist)
    # ------------------------------------------------------------------
    print("\n=== DDP Path tests ===")

    with patch("torch.distributed.init_process_group"), \
         patch("torch.distributed.is_initialized", return_value=True), \
         patch("torch.distributed.get_rank", return_value=0), \
         patch("torch.distributed.barrier"):

        router_ddp = StrategyRouter(MockDistCfg(strategy="ddp"))
        router_ddp._state = RouterState.DISTRIBUTED_READY
        router_ddp._context = StrategyContext(
            strategy="ddp", rank=0, world_size=1, local_rank=0,
            device="cpu", is_distributed=False
        )

        model_ddp = nn.Linear(8, 8)
        wrapped, ctx = router_ddp.wrap_model(model_ddp)

        check("ddp_context_strategy", ctx.strategy == "ddp")
        check("ddp_wrapped_is_module", isinstance(wrapped, nn.Module))
        check("ddp_state_is_model_wrapped", router_ddp.state == RouterState.MODEL_WRAPPED)

        optimizer = router_ddp.build_optimizer(wrapped, MockOptCfg())
        check("ddp_optimizer_built", optimizer is not None)
        check("ddp_state_is_optimizer_built", router_ddp.state == RouterState.OPTIMIZER_BUILT)
        check("ddp_optimizer_type", isinstance(optimizer, torch.optim.AdamW))

    # ------------------------------------------------------------------
    # Test: FSDP dispatch (mocked FSDPWrapper)
    # ------------------------------------------------------------------
    print("\n=== FSDP Dispatch tests ===")

    fsdp_cfg = MockFSDPCfg()

    # Use patch.object on the class directly to handle both __main__ and
    # module-import scenarios
    router_fsdp = StrategyRouter(
        MockDistCfg(strategy="fsdp"),
        fsdp_cfg=fsdp_cfg,
    )
    router_fsdp._state = RouterState.DISTRIBUTED_READY
    router_fsdp._context = StrategyContext(
        strategy="fsdp", rank=0, world_size=4, local_rank=0,
        device="cpu", is_distributed=True
    )

    with patch.object(router_fsdp, "_wrap_fsdp", return_value=nn.Linear(4, 4)) as mock_fsdp_wrap:
        wrapped_fsdp, ctx_fsdp = router_fsdp.wrap_model(nn.Linear(4, 4))
        check("fsdp_dispatch_called", mock_fsdp_wrap.called)
        check("fsdp_context_strategy", ctx_fsdp.strategy == "fsdp")

    # ------------------------------------------------------------------
    # Test: DeepSpeed dispatch (mocked)
    # ------------------------------------------------------------------
    print("\n=== DeepSpeed Dispatch tests ===")

    ds_cfg = MockDSCfg(zero_stage=3)

    router_ds = StrategyRouter(
        MockDistCfg(strategy="deepspeed_zero3"),
        ds_cfg=ds_cfg,
    )
    router_ds._state = RouterState.DISTRIBUTED_READY
    router_ds._context = StrategyContext(
        strategy="deepspeed_zero3", rank=0, world_size=4, local_rank=0,
        device="cpu", is_distributed=True
    )

    mock_engine = MagicMock()
    mock_optimizer_obj = MagicMock()

    # wrap_model transitions state (DS doesn't call deepspeed.initialize here)
    model_ds = nn.Linear(4, 4)
    wrapped_ds, ctx_ds = router_ds.wrap_model(model_ds)
    check("ds_context_strategy", ctx_ds.strategy == "deepspeed_zero3")
    check("ds_state_model_wrapped", router_ds.state == RouterState.MODEL_WRAPPED)

    # build_optimizer triggers deepspeed wrapping — patch it on the instance
    with patch.object(
        router_ds, "_wrap_deepspeed", return_value=(mock_engine, mock_optimizer_obj)
    ) as mock_ds_wrap:
        ds_optimizer = router_ds.build_optimizer(wrapped_ds, MockOptCfg())
        check("ds_wrap_called", mock_ds_wrap.called)
        check("ds_state_optimizer_built", router_ds.state == RouterState.OPTIMIZER_BUILT)

    # ------------------------------------------------------------------
    # Test: Config validation at construction
    # ------------------------------------------------------------------
    print("\n=== Config Validation at Construction tests ===")

    expect_raises(
        "fsdp_without_fsdp_cfg_raises",
        ValueError,
        lambda: StrategyRouter(MockDistCfg(strategy="fsdp")),
    )
    expect_raises(
        "deepspeed_without_ds_cfg_raises",
        ValueError,
        lambda: StrategyRouter(MockDistCfg(strategy="deepspeed_zero3")),
    )

    # Valid constructions don't raise
    r = StrategyRouter(MockDistCfg(strategy="ddp"))
    check("ddp_no_fsdp_cfg_ok", r is not None)

    r2 = StrategyRouter(
        MockDistCfg(strategy="fsdp"),
        fsdp_cfg=MockFSDPCfg(),
    )
    check("fsdp_with_fsdp_cfg_ok", r2 is not None)

    r3 = StrategyRouter(
        MockDistCfg(strategy="deepspeed_zero3"),
        ds_cfg=MockDSCfg(),
    )
    check("ds_with_ds_cfg_ok", r3 is not None)

    # ------------------------------------------------------------------
    # Test: Optimizer type selection
    # ------------------------------------------------------------------
    print("\n=== Optimizer Type Selection tests ===")

    router_opt = StrategyRouter(MockDistCfg(strategy="ddp"))
    router_opt._state = RouterState.MODEL_WRAPPED
    router_opt._context = StrategyContext(strategy="ddp", world_size=1)

    model_opt = nn.Linear(4, 4)

    cfg_adam = MockOptCfg(optimizer_type="Adam")
    opt_adam = router_opt._build_base_optimizer(model_opt, cfg_adam)
    check("builds_adam_optimizer", isinstance(opt_adam, torch.optim.Adam))

    cfg_sgd = MockOptCfg(optimizer_type="SGD")
    opt_sgd = router_opt._build_base_optimizer(model_opt, cfg_sgd)
    check("builds_sgd_optimizer", isinstance(opt_sgd, torch.optim.SGD))

    cfg_invalid = MockOptCfg(optimizer_type="RMSprop")
    expect_raises(
        "invalid_optimizer_type_raises",
        ValueError,
        lambda: router_opt._build_base_optimizer(model_opt, cfg_invalid),
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*50}")
    if failures:
        print(f"FAIL: {len(failures)} test(s) failed: {failures}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
        sys.exit(0)
