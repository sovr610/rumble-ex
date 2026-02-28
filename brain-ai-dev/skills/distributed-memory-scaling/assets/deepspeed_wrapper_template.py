#!/usr/bin/env python3
"""
deepspeed_wrapper_template.py
------------------------------
DeepSpeedWrapper: generates deterministic ZeRO config JSON, wraps a model
via deepspeed.initialize(), saves/loads checkpoints, and exports fp32 weights.

Key rules:
  - Config JSON is always generated from the DeepSpeedConfig dataclass.
    Never hand-edit the JSON.
  - offload_param is only valid for ZeRO stage 3. The wrapper raises
    ValueError at config generation time if this constraint is violated.
  - Bucket size defaults follow published best-practice values.
  - Engine lifecycle:
      engine = wrapper.wrap(model, ds_cfg, optimizer)
      engine.backward(loss)
      engine.step()
      wrapper.save_checkpoint(engine, save_dir, tag)
      wrapper.load_checkpoint(engine, load_dir, tag)
      wrapper.export_fp32_weights(checkpoint_dir, output_path)

Usage
-----
    from deepspeed_wrapper_template import DeepSpeedWrapper, DeepSpeedConfig

    ds_cfg = DeepSpeedConfig(zero_stage=3, bf16=True)
    wrapper = DeepSpeedWrapper()
    engine = wrapper.wrap(model, ds_cfg, optimizer=optimizer)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy deepspeed import
# ---------------------------------------------------------------------------


def _import_deepspeed() -> Any:
    """Import deepspeed, raising ImportError with a useful message if absent."""
    try:
        import deepspeed
        return deepspeed
    except ImportError as e:
        raise ImportError(
            "DeepSpeed is not installed. Install with: pip install deepspeed. "
            f"Error: {e}"
        ) from e


# ---------------------------------------------------------------------------
# DeepSpeedWrapper
# ---------------------------------------------------------------------------


class DeepSpeedWrapper:
    """Encapsulates DeepSpeed engine creation, config generation, and checkpointing."""

    def __init__(self) -> None:
        self._engine: Optional[Any] = None

    # ------------------------------------------------------------------
    # Config JSON Generation
    # ------------------------------------------------------------------

    def generate_config_json(self, cfg: Any) -> Dict[str, Any]:
        """Generate a deterministic DeepSpeed config dict from cfg.

        Parameters
        ----------
        cfg:
            DeepSpeedConfig instance.

        Returns
        -------
        dict
            DeepSpeed config dict, suitable for passing to deepspeed.initialize().

        Raises
        ------
        ValueError
            If cfg contains invalid field combinations (e.g., offload_param
            on stage 2).
        """
        # Validate before generating
        cfg.validate()

        config: Dict[str, Any] = {}

        # ---- Zero optimization ----
        zero_opt: Dict[str, Any] = {
            "stage": cfg.zero_stage,
            "contiguous_gradients": cfg.contiguous_gradients,
            "overlap_comm": cfg.overlap_comm,
            "reduce_scatter": True,
            "reduce_bucket_size": cfg.reduce_bucket_size,
            "allgather_bucket_size": cfg.allgather_bucket_size,
        }

        # Stage-3-only fields
        if cfg.zero_stage == 3:
            zero_opt.update(
                {
                    "stage3_prefetch_bucket_size": cfg.stage3_prefetch_bucket_size,
                    "stage3_param_persistence_threshold": cfg.stage3_param_persistence_threshold,
                    "stage3_max_live_parameters": cfg.stage3_max_live_parameters,
                    "stage3_max_reuse_distance": cfg.stage3_max_reuse_distance,
                    "stage3_gather_16bit_weights_on_model_save": True,
                }
            )

        # Offload optimizer
        if cfg.offload_optimizer != "none":
            zero_opt["offload_optimizer"] = self._build_offload_dict(
                cfg.offload_optimizer, cfg.nvme_path
            )

        # Offload param (stage 3 only — already validated)
        if cfg.offload_param != "none":
            zero_opt["offload_param"] = self._build_offload_dict(
                cfg.offload_param, cfg.nvme_path
            )

        config["zero_optimization"] = zero_opt

        # ---- Mixed precision ----
        if cfg.fp16:
            config["fp16"] = {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1,
            }
        elif cfg.bf16:
            config["bf16"] = {"enabled": True}

        # ---- Gradient clipping ----
        config["gradient_clipping"] = cfg.gradient_clipping

        # ---- Misc ----
        config["steps_per_print"] = 100
        config["wall_clock_breakdown"] = False

        return config

    @staticmethod
    def _build_offload_dict(device: str, nvme_path: Optional[str]) -> Dict[str, Any]:
        """Build offload configuration sub-dict."""
        d: Dict[str, Any] = {"device": device, "pin_memory": device == "cpu"}
        if device == "nvme":
            if not nvme_path:
                raise ValueError("nvme_path required for NVMe offload.")
            d["nvme_path"] = nvme_path
            d["pin_memory"] = False
            d["buffer_count"] = 5
            d["buffer_size"] = 100_000_000
        return d

    def config_to_json_string(self, cfg: Any) -> str:
        """Return the config dict as a formatted JSON string."""
        return json.dumps(self.generate_config_json(cfg), indent=2)

    # ------------------------------------------------------------------
    # Wrap (Engine Creation)
    # ------------------------------------------------------------------

    def wrap(
        self,
        model: nn.Module,
        cfg: Any,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        model_parameters: Optional[Any] = None,
    ) -> Any:
        """Wrap model with DeepSpeed engine.

        Parameters
        ----------
        model:
            Unwrapped nn.Module.
        cfg:
            DeepSpeedConfig instance.
        optimizer:
            Pre-built optimizer. If None, DeepSpeed will try to create one
            from the config (requires an 'optimizer' section in config).
        lr_scheduler:
            Optional learning rate scheduler.
        model_parameters:
            Optional parameter groups list. Use instead of optimizer for
            per-parameter-group configurations.

        Returns
        -------
        DeepSpeedEngine
            Wrapped engine. Use engine.backward(loss) and engine.step()
            instead of loss.backward() and optimizer.step().
        """
        deepspeed = _import_deepspeed()
        config_dict = self.generate_config_json(cfg)

        kwargs: Dict[str, Any] = {
            "model": model,
            "config": config_dict,
        }
        if optimizer is not None:
            kwargs["optimizer"] = optimizer
        if lr_scheduler is not None:
            kwargs["lr_scheduler"] = lr_scheduler
        if model_parameters is not None:
            kwargs["model_parameters"] = model_parameters

        engine, ds_optimizer, _, _ = deepspeed.initialize(**kwargs)

        self._engine = engine
        logger.info(
            "DeepSpeed engine initialized: ZeRO stage=%d", cfg.zero_stage
        )
        return engine

    # ------------------------------------------------------------------
    # Save Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        engine: Any,
        save_dir: str,
        tag: str,
        client_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save DeepSpeed sharded checkpoint.

        Parameters
        ----------
        engine:
            DeepSpeedEngine returned by wrap().
        save_dir:
            Root directory for checkpoints. A subdirectory named `tag`
            will be created.
        tag:
            Checkpoint tag (e.g. 'step_1000'). Used as subdirectory name
            and for the 'latest' pointer file.
        client_state:
            Optional dict of extra metadata to save alongside the checkpoint
            (e.g. {'step': 1000, 'epoch': 2}).
        """
        os.makedirs(save_dir, exist_ok=True)
        engine.save_checkpoint(
            save_dir=save_dir,
            tag=tag,
            client_state=client_state or {},
            save_latest=True,
        )
        logger.info(
            "DeepSpeed checkpoint saved to %s/%s", save_dir, tag
        )

    # ------------------------------------------------------------------
    # Load Checkpoint
    # ------------------------------------------------------------------

    def load_checkpoint(
        self,
        engine: Any,
        load_dir: str,
        tag: Optional[str] = None,
        load_optimizer_states: bool = True,
        load_lr_scheduler_states: bool = True,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Load DeepSpeed checkpoint.

        Parameters
        ----------
        engine:
            DeepSpeedEngine to restore into.
        load_dir:
            Root checkpoint directory (same as save_dir in save_checkpoint).
        tag:
            Checkpoint tag. If None, loads the checkpoint pointed to by
            the 'latest' file in load_dir.
        load_optimizer_states:
            Whether to restore optimizer state.
        load_lr_scheduler_states:
            Whether to restore LR scheduler state.

        Returns
        -------
        tuple of (step_or_None, client_state_dict)
        """
        _, client_state = engine.load_checkpoint(
            load_dir=load_dir,
            tag=tag,
            load_optimizer_states=load_optimizer_states,
            load_lr_scheduler_states=load_lr_scheduler_states,
        )
        step = client_state.get("step") if client_state else None
        logger.info(
            "DeepSpeed checkpoint loaded from %s (tag=%s, step=%s)",
            load_dir, tag, step,
        )
        return step, (client_state or {})

    # ------------------------------------------------------------------
    # Export FP32 Weights
    # ------------------------------------------------------------------

    def export_fp32_weights(
        self,
        checkpoint_dir: str,
        output_path: str,
        zero_to_fp32_script: Optional[str] = None,
    ) -> None:
        """Consolidate ZeRO-2/3 sharded checkpoint into a single fp32 state dict.

        Locates the zero_to_fp32.py script from the DeepSpeed installation
        (or uses the path provided) and runs it as a subprocess.

        Parameters
        ----------
        checkpoint_dir:
            Path to the checkpoint tag directory (e.g.
            '/checkpoints/step_1000').
        output_path:
            Destination file path for the consolidated fp32 state dict
            (e.g. '/weights/model_fp32.pt').
        zero_to_fp32_script:
            Explicit path to zero_to_fp32.py. If None, the script is
            located in the DeepSpeed package.

        Raises
        ------
        FileNotFoundError
            If zero_to_fp32.py cannot be located.
        subprocess.CalledProcessError
            If the consolidation script exits with a non-zero code.
        """
        script_path = zero_to_fp32_script or self._find_zero_to_fp32_script()

        if not os.path.isfile(script_path):
            raise FileNotFoundError(
                f"zero_to_fp32.py not found at '{script_path}'. "
                "Install DeepSpeed or provide the explicit script path."
            )

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        cmd = [
            sys.executable,
            script_path,
            checkpoint_dir,
            output_path,
        ]
        logger.info(
            "Running zero_to_fp32.py: %s", " ".join(cmd)
        )
        subprocess.run(cmd, check=True)
        logger.info("FP32 weights exported to %s", output_path)

    @staticmethod
    def _find_zero_to_fp32_script() -> str:
        """Locate zero_to_fp32.py from the installed DeepSpeed package."""
        try:
            import deepspeed as _ds
            pkg_dir = os.path.dirname(os.path.abspath(_ds.__file__))
            candidates = [
                os.path.join(pkg_dir, "utils", "zero_to_fp32.py"),
                os.path.join(pkg_dir, "zero_to_fp32.py"),
            ]
            for path in candidates:
                if os.path.isfile(path):
                    return path
            # Fallback: search in sys.prefix
            prefix_path = os.path.join(sys.prefix, "bin", "zero_to_fp32.py")
            if os.path.isfile(prefix_path):
                return prefix_path
            return candidates[0]  # Return first candidate; caller will check existence
        except ImportError:
            return "zero_to_fp32.py"

    def export_fp32_weights_in_process(
        self,
        model: nn.Module,
        output_path: str,
    ) -> None:
        """Export fp32 weights directly using GatheredParameters (ZeRO-3).

        Use this when you need to export weights during training without
        running a subprocess. Requires DeepSpeed and an active ZeRO-3 engine.

        Parameters
        ----------
        model:
            The ZeRO-3-wrapped model (inside the engine).
        output_path:
            Destination file path for the fp32 state dict.
        """
        deepspeed = _import_deepspeed()
        import torch.distributed as dist

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        with deepspeed.zero.GatheredParameters(
            list(model.parameters()), modifier_rank=0
        ):
            if not dist.is_initialized() or dist.get_rank() == 0:
                state_dict = {
                    k: v.data.float().clone()
                    for k, v in model.named_parameters()
                }
                torch.save(state_dict, output_path)
                logger.info(
                    "In-process FP32 weights exported to %s", output_path
                )

        if dist.is_initialized():
            dist.barrier()

    # ------------------------------------------------------------------
    # Validate Stage-Specific Fields (public helper)
    # ------------------------------------------------------------------

    @staticmethod
    def validate_stage_fields(cfg: Any) -> None:
        """Validate stage-specific field constraints.

        Parameters
        ----------
        cfg:
            DeepSpeedConfig instance.

        Raises
        ------
        ValueError
            For any constraint violation.
        """
        if cfg.offload_param != "none" and cfg.zero_stage != 3:
            raise ValueError(
                f"offload_param='{cfg.offload_param}' is only valid for "
                f"zero_stage=3, got zero_stage={cfg.zero_stage}."
            )
        if cfg.reduce_bucket_size <= 0:
            raise ValueError(
                f"reduce_bucket_size must be > 0, got {cfg.reduce_bucket_size}"
            )
        if cfg.allgather_bucket_size <= 0:
            raise ValueError(
                f"allgather_bucket_size must be > 0, got {cfg.allgather_bucket_size}"
            )
        if cfg.zero_stage == 3:
            if cfg.stage3_prefetch_bucket_size <= 0:
                raise ValueError(
                    f"stage3_prefetch_bucket_size must be > 0, "
                    f"got {cfg.stage3_prefetch_bucket_size}"
                )
            if cfg.stage3_param_persistence_threshold <= 0:
                raise ValueError(
                    f"stage3_param_persistence_threshold must be > 0, "
                    f"got {cfg.stage3_param_persistence_threshold}"
                )


# ---------------------------------------------------------------------------
# Self-Tests
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys
    from dataclasses import dataclass, field
    from typing import List

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

    # Minimal config dataclass for self-tests (standalone, no dependency on
    # distributed_config_template to keep this file self-contained)
    @dataclass
    class MockDeepSpeedConfig:
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
        nvme_path: Optional[str] = None
        fp16: bool = False
        bf16: bool = False
        gradient_clipping: float = 1.0

        def validate(self) -> None:
            valid_stages = {2, 3}
            if self.zero_stage not in valid_stages:
                raise ValueError(f"Invalid zero_stage {self.zero_stage}")
            if self.offload_param != "none" and self.zero_stage != 3:
                raise ValueError(
                    f"offload_param requires zero_stage=3, got {self.zero_stage}"
                )
            if self.fp16 and self.bf16:
                raise ValueError("Cannot enable both fp16 and bf16.")
            if self.reduce_bucket_size <= 0:
                raise ValueError("reduce_bucket_size must be > 0")
            if self.offload_param == "nvme" and not self.nvme_path:
                raise ValueError("nvme_path required for NVMe offload")

    wrapper = DeepSpeedWrapper()

    # ------------------------------------------------------------------
    # Stage 2 JSON generation
    # ------------------------------------------------------------------
    print("=== Stage 2 JSON Generation tests ===")

    cfg2 = MockDeepSpeedConfig(zero_stage=2)
    config2 = wrapper.generate_config_json(cfg2)

    check("stage2_stage_field", config2["zero_optimization"]["stage"] == 2)
    check("stage2_overlap_comm", config2["zero_optimization"]["overlap_comm"] is True)
    check(
        "stage2_contiguous_gradients",
        config2["zero_optimization"]["contiguous_gradients"] is True,
    )
    check(
        "stage2_reduce_bucket_size",
        config2["zero_optimization"]["reduce_bucket_size"] == 500_000_000,
    )
    check(
        "stage2_no_stage3_fields",
        "stage3_prefetch_bucket_size" not in config2["zero_optimization"],
    )
    check("stage2_no_offload_param", "offload_param" not in config2["zero_optimization"])
    check("stage2_gradient_clipping", config2["gradient_clipping"] == 1.0)

    # ------------------------------------------------------------------
    # Stage 3 JSON generation
    # ------------------------------------------------------------------
    print("\n=== Stage 3 JSON Generation tests ===")

    cfg3 = MockDeepSpeedConfig(zero_stage=3, bf16=True)
    config3 = wrapper.generate_config_json(cfg3)

    check("stage3_stage_field", config3["zero_optimization"]["stage"] == 3)
    check(
        "stage3_has_prefetch_bucket",
        "stage3_prefetch_bucket_size" in config3["zero_optimization"],
    )
    check(
        "stage3_has_persistence_threshold",
        "stage3_param_persistence_threshold" in config3["zero_optimization"],
    )
    check(
        "stage3_prefetch_bucket_value",
        config3["zero_optimization"]["stage3_prefetch_bucket_size"] == 50_000_000,
    )
    check("stage3_bf16_enabled", config3.get("bf16", {}).get("enabled") is True)
    check("stage3_no_fp16", "fp16" not in config3)

    # ------------------------------------------------------------------
    # Stage 3 with offload
    # ------------------------------------------------------------------
    print("\n=== Stage 3 with Offload tests ===")

    cfg3_offload = MockDeepSpeedConfig(
        zero_stage=3,
        offload_optimizer="cpu",
        offload_param="cpu",
    )
    config3_off = wrapper.generate_config_json(cfg3_offload)

    check(
        "stage3_offload_optimizer_device",
        config3_off["zero_optimization"]["offload_optimizer"]["device"] == "cpu",
    )
    check(
        "stage3_offload_param_device",
        config3_off["zero_optimization"]["offload_param"]["device"] == "cpu",
    )
    check(
        "stage3_offload_optimizer_pin_memory",
        config3_off["zero_optimization"]["offload_optimizer"]["pin_memory"] is True,
    )

    # ------------------------------------------------------------------
    # Stage 2 with offload_optimizer
    # ------------------------------------------------------------------
    print("\n=== Stage 2 Optimizer Offload tests ===")

    cfg2_offopt = MockDeepSpeedConfig(zero_stage=2, offload_optimizer="cpu")
    config2_offopt = wrapper.generate_config_json(cfg2_offopt)
    check(
        "stage2_offload_optimizer_ok",
        config2_offopt["zero_optimization"]["offload_optimizer"]["device"] == "cpu",
    )

    # ------------------------------------------------------------------
    # Field validation: offload_param on stage 2 raises
    # ------------------------------------------------------------------
    print("\n=== Stage-Specific Field Validation tests ===")

    expect_raises(
        "offload_param_stage2_raises",
        ValueError,
        lambda: wrapper.generate_config_json(
            MockDeepSpeedConfig(zero_stage=2, offload_param="cpu")
        ),
    )

    # Invalid bucket sizes raise
    expect_raises(
        "negative_reduce_bucket_raises",
        ValueError,
        lambda: wrapper.validate_stage_fields(
            MockDeepSpeedConfig(zero_stage=2, reduce_bucket_size=-1)
        ),
    )

    # fp16 + bf16 raises
    expect_raises(
        "fp16_bf16_conflict_raises",
        ValueError,
        lambda: wrapper.generate_config_json(
            MockDeepSpeedConfig(fp16=True, bf16=True)
        ),
    )

    # ------------------------------------------------------------------
    # JSON is parseable string
    # ------------------------------------------------------------------
    print("\n=== JSON Serialization tests ===")

    cfg3_json_str = wrapper.config_to_json_string(cfg3)
    try:
        reparsed = json.loads(cfg3_json_str)
        check("config_json_parseable", reparsed["zero_optimization"]["stage"] == 3)
    except Exception as e:
        print(f"  FAIL  config_json_parseable ({e})")
        failures.append("config_json_parseable")

    # NVMe offload config
    cfg_nvme = MockDeepSpeedConfig(
        zero_stage=3,
        offload_param="nvme",
        nvme_path="/local_nvme",
    )
    config_nvme = wrapper.generate_config_json(cfg_nvme)
    check(
        "nvme_offload_device",
        config_nvme["zero_optimization"]["offload_param"]["device"] == "nvme",
    )
    check(
        "nvme_offload_path",
        config_nvme["zero_optimization"]["offload_param"]["nvme_path"] == "/local_nvme",
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
