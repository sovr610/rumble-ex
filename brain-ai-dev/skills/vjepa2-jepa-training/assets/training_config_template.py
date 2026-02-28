"""
JEPATrainingConfig — Configuration dataclass for V-JEPA 2 self-supervised training.

Covers all fields from SKILL.md, provides three preset factory methods, and
includes field-level validation. The validate() method returns a list of error
strings (empty means valid) so callers can surface multiple problems at once.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class JEPATrainingConfig:
    """
    Complete configuration for one stage of V-JEPA 2 training.

    Fields are grouped by concern: architecture, optimization, EMA, loss,
    annealing/progressive pipeline, and autoregressive DROID mode.
    """

    # ------------------------------------------------------------------
    # Architecture
    # ------------------------------------------------------------------

    #: Internal dimension of the predictor transformer.
    predictor_embed_dim: int = 384

    #: Number of transformer blocks in the predictor.
    predictor_depth: int = 12

    #: Number of attention heads in the predictor.
    predictor_num_heads: int = 12

    #: Number of learnable mask tokens (one slot per target position).
    num_mask_tokens: int = 10

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    #: Peak / reference learning rate (after warmup).
    lr: float = 1e-3

    #: Minimum learning rate at the end of cosine decay.
    final_lr: float = 1e-6

    #: Number of warmup epochs (linear ramp from 0 to lr).
    warmup_epochs: int = 40

    #: Total training epochs for this stage.
    epochs: int = 300

    #: Initial weight decay (applied to 2D+ params only).
    weight_decay: float = 0.04

    #: Final weight decay at the end of the cosine WD schedule.
    final_weight_decay: float = 0.4

    #: Per-device batch size (for memory planning).
    batch_size: int = 64

    #: Use BFloat16 automatic mixed precision.
    use_bfloat16: bool = True

    #: AdamW beta1 coefficient.
    beta1: float = 0.9

    #: AdamW beta2 coefficient.
    beta2: float = 0.95

    #: Gradient clipping max norm (0.0 = disabled).
    clip_grad: float = 1.0

    #: Number of steps to accumulate gradients before optimizer step.
    gradient_accumulation_steps: int = 1

    # ------------------------------------------------------------------
    # EMA (Exponential Moving Average Target Encoder)
    # ------------------------------------------------------------------

    #: EMA momentum at the beginning of training.
    ema_start: float = 0.99925

    #: EMA momentum at the end of training.
    ema_end: float = 0.99925

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    #: Exponent for smooth L1 loss (1.0 = standard Huber, >1.0 sharpens gradients).
    loss_exp: float = 1.0

    #: Apply layer norm to both pred and target before computing loss (DROID mode).
    normalize_reps: bool = False

    #: Beta parameter for smooth L1 (threshold between L1 and L2 regime).
    loss_beta: float = 1.0

    # ------------------------------------------------------------------
    # Progressive Training / Annealing
    # ------------------------------------------------------------------

    #: Set True to use annealing LR schedule (warmup + stable + linear decay).
    is_anneal: bool = False

    #: Path to checkpoint to load at the start of an annealing stage.
    anneal_ckpt: Optional[str] = None

    # ------------------------------------------------------------------
    # Autoregressive DROID Fine-Tuning
    # ------------------------------------------------------------------

    #: Number of autoregressive prediction steps (0 = single-step standard JEPA).
    auto_steps: int = 0

    #: LR scale for encoder during DROID fine-tuning (0.0 = fully frozen).
    encoder_lr_scale: float = 1.0

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    #: Log every N steps.
    log_every: int = 50

    #: Save checkpoint every N epochs.
    save_every: int = 10

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """
        Validate configuration fields.

        Returns:
            List of human-readable error strings. Empty means config is valid.
        """
        errors: List[str] = []

        # Learning rate checks
        if self.lr <= 0.0:
            errors.append(
                f"lr must be > 0, got {self.lr}"
            )
        if self.final_lr < 0.0:
            errors.append(
                f"final_lr must be >= 0, got {self.final_lr}"
            )
        if self.final_lr > self.lr:
            errors.append(
                f"final_lr ({self.final_lr}) must be <= lr ({self.lr})"
            )

        # Epoch / step checks
        if self.epochs <= 0:
            errors.append(
                f"epochs must be > 0, got {self.epochs}"
            )
        if self.warmup_epochs < 0:
            errors.append(
                f"warmup_epochs must be >= 0, got {self.warmup_epochs}"
            )
        if self.warmup_epochs > self.epochs:
            errors.append(
                f"warmup_epochs ({self.warmup_epochs}) must be <= epochs ({self.epochs})"
            )

        # Weight decay checks
        if self.weight_decay < 0.0:
            errors.append(
                f"weight_decay must be >= 0, got {self.weight_decay}"
            )
        if self.final_weight_decay < 0.0:
            errors.append(
                f"final_weight_decay must be >= 0, got {self.final_weight_decay}"
            )

        # Batch / optimizer checks
        if self.batch_size <= 0:
            errors.append(
                f"batch_size must be > 0, got {self.batch_size}"
            )
        if not (0.0 <= self.beta1 < 1.0):
            errors.append(
                f"beta1 must be in [0, 1), got {self.beta1}"
            )
        if not (0.0 <= self.beta2 < 1.0):
            errors.append(
                f"beta2 must be in [0, 1), got {self.beta2}"
            )
        if self.clip_grad < 0.0:
            errors.append(
                f"clip_grad must be >= 0 (0=disabled), got {self.clip_grad}"
            )
        if self.gradient_accumulation_steps < 1:
            errors.append(
                f"gradient_accumulation_steps must be >= 1, got {self.gradient_accumulation_steps}"
            )

        # EMA checks
        if not (0.0 <= self.ema_start <= 1.0):
            errors.append(
                f"ema_start must be in [0, 1], got {self.ema_start}"
            )
        if not (0.0 <= self.ema_end <= 1.0):
            errors.append(
                f"ema_end must be in [0, 1], got {self.ema_end}"
            )

        # Loss checks
        if self.loss_exp <= 0.0:
            errors.append(
                f"loss_exp must be > 0, got {self.loss_exp}"
            )
        if self.loss_beta <= 0.0:
            errors.append(
                f"loss_beta must be > 0, got {self.loss_beta}"
            )

        # Architecture checks
        if self.predictor_embed_dim <= 0:
            errors.append(
                f"predictor_embed_dim must be > 0, got {self.predictor_embed_dim}"
            )
        if self.predictor_depth <= 0:
            errors.append(
                f"predictor_depth must be > 0, got {self.predictor_depth}"
            )
        if self.predictor_num_heads <= 0:
            errors.append(
                f"predictor_num_heads must be > 0, got {self.predictor_num_heads}"
            )
        if self.predictor_embed_dim % self.predictor_num_heads != 0:
            errors.append(
                f"predictor_embed_dim ({self.predictor_embed_dim}) must be divisible "
                f"by predictor_num_heads ({self.predictor_num_heads})"
            )
        if self.num_mask_tokens <= 0:
            errors.append(
                f"num_mask_tokens must be > 0, got {self.num_mask_tokens}"
            )

        # Autoregressive checks
        if self.auto_steps < 0:
            errors.append(
                f"auto_steps must be >= 0, got {self.auto_steps}"
            )
        if not (0.0 <= self.encoder_lr_scale <= 1.0):
            errors.append(
                f"encoder_lr_scale must be in [0, 1], got {self.encoder_lr_scale}"
            )

        # Annealing consistency
        if self.is_anneal and self.anneal_ckpt is None:
            errors.append(
                "is_anneal=True requires anneal_ckpt to be set"
            )

        return errors

    def assert_valid(self) -> None:
        """Raise ValueError if validate() returns any errors."""
        errors = self.validate()
        if errors:
            raise ValueError(
                "JEPATrainingConfig has invalid fields:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    # ------------------------------------------------------------------
    # Preset Factory Methods
    # ------------------------------------------------------------------

    @classmethod
    def pretrain_256(cls) -> "JEPATrainingConfig":
        """
        Stage 1: Standard pretraining at 256px / 16 frames.

        Full warmup + cosine LR schedule over 300 epochs.
        EMA momentum fixed at 0.99925. AdamW with cosine WD schedule.
        """
        return cls(
            # Architecture
            predictor_embed_dim=384,
            predictor_depth=12,
            predictor_num_heads=12,
            num_mask_tokens=10,
            # Optimization
            lr=1e-3,
            final_lr=1e-6,
            warmup_epochs=40,
            epochs=300,
            weight_decay=0.04,
            final_weight_decay=0.4,
            batch_size=64,
            use_bfloat16=True,
            beta1=0.9,
            beta2=0.95,
            clip_grad=1.0,
            # EMA
            ema_start=0.99925,
            ema_end=0.99925,
            # Loss
            loss_exp=1.0,
            normalize_reps=False,
            loss_beta=1.0,
            # Not annealing
            is_anneal=False,
            anneal_ckpt=None,
            # No autoregressive
            auto_steps=0,
            encoder_lr_scale=1.0,
        )

    @classmethod
    def cooldown_384(cls) -> "JEPATrainingConfig":
        """
        Stage 2: Cooldown / annealing at 384px / 64 frames.

        Load from pretrain checkpoint and linearly decay LR to near-zero.
        No warmup; WD fixed at final value. Smaller batch due to memory.
        """
        return cls(
            # Architecture (same as pretrain)
            predictor_embed_dim=384,
            predictor_depth=12,
            predictor_num_heads=12,
            num_mask_tokens=10,
            # Optimization — annealing
            lr=1e-4,            # Start near pretrain final LR
            final_lr=1e-7,      # Decay to near-zero
            warmup_epochs=0,    # No warmup for cooldown
            epochs=30,
            weight_decay=0.4,   # Already at final value
            final_weight_decay=0.4,
            batch_size=16,      # Smaller batch at higher resolution
            use_bfloat16=True,
            beta1=0.9,
            beta2=0.95,
            clip_grad=1.0,
            # EMA
            ema_start=0.99925,
            ema_end=0.99925,
            # Loss
            loss_exp=1.0,
            normalize_reps=False,
            loss_beta=1.0,
            # Annealing mode
            is_anneal=True,
            anneal_ckpt="pretrain_epoch300.pth",  # Override in practice
            # No autoregressive
            auto_steps=0,
            encoder_lr_scale=1.0,
        )

    @classmethod
    def droid_finetune(cls) -> "JEPATrainingConfig":
        """
        Stage 3: DROID robotics post-training at 256px / 8 frames.

        Frozen encoder, deep frame-causal predictor, normalized loss,
        autoregressive 2-step prediction. Differential LR via encoder_lr_scale=0.0.
        """
        return cls(
            # Architecture — deeper predictor for DROID
            predictor_embed_dim=384,
            predictor_depth=12,
            predictor_num_heads=12,
            num_mask_tokens=10,
            # Optimization
            lr=5e-4,
            final_lr=5e-6,
            warmup_epochs=5,
            epochs=100,
            weight_decay=0.05,
            final_weight_decay=0.2,
            batch_size=32,
            use_bfloat16=True,
            beta1=0.9,
            beta2=0.999,        # Standard AdamW for fine-tuning
            clip_grad=1.0,
            # EMA (unused in DROID but kept for interface consistency)
            ema_start=0.99925,
            ema_end=0.99925,
            # Loss — normalized for DROID
            loss_exp=1.0,
            normalize_reps=True,  # Normalize before loss in DROID
            loss_beta=1.0,
            # Load from cooldown checkpoint
            is_anneal=False,
            anneal_ckpt="cooldown_epoch30.pth",  # Override in practice
            # Autoregressive 2-step prediction
            auto_steps=2,
            encoder_lr_scale=0.0,   # Frozen encoder (no gradient)
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def total_steps(self, steps_per_epoch: int) -> int:
        """Compute total training steps for this stage."""
        return self.epochs * steps_per_epoch

    def warmup_steps(self, steps_per_epoch: int) -> int:
        """Compute warmup steps in absolute step count."""
        return self.warmup_epochs * steps_per_epoch

    def effective_batch_size(self, num_gpus: int = 1) -> int:
        """Compute effective batch size across gradient accumulation and GPUs."""
        return self.batch_size * self.gradient_accumulation_steps * num_gpus

    def __repr__(self) -> str:
        cls = type(self).__name__
        lines = [f"{cls}("]
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if v != f.default:
                lines.append(f"    {f.name}={v!r},")
        lines.append(")")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("JEPATrainingConfig self-tests")
    print("=" * 60)

    # --- Test 1: pretrain_256 preset is valid ---
    cfg1 = JEPATrainingConfig.pretrain_256()
    errors1 = cfg1.validate()
    assert len(errors1) == 0, f"pretrain_256 preset has errors: {errors1}"
    print("[PASS] pretrain_256 preset is valid")

    # --- Test 2: cooldown_384 preset is valid ---
    cfg2 = JEPATrainingConfig.cooldown_384()
    errors2 = cfg2.validate()
    assert len(errors2) == 0, f"cooldown_384 preset has errors: {errors2}"
    print("[PASS] cooldown_384 preset is valid")

    # --- Test 3: droid_finetune preset is valid ---
    cfg3 = JEPATrainingConfig.droid_finetune()
    errors3 = cfg3.validate()
    assert len(errors3) == 0, f"droid_finetune preset has errors: {errors3}"
    print("[PASS] droid_finetune preset is valid")

    # --- Test 4: Negative LR is caught ---
    bad_cfg = JEPATrainingConfig(lr=-0.001)
    errors4 = bad_cfg.validate()
    assert any("lr" in e.lower() for e in errors4), (
        f"Should catch lr <= 0, got: {errors4}"
    )
    print("[PASS] Negative lr caught by validate()")

    # --- Test 5: EMA out of range is caught ---
    bad_cfg2 = JEPATrainingConfig(ema_start=1.5)
    errors5 = bad_cfg2.validate()
    assert any("ema" in e.lower() for e in errors5), (
        f"Should catch ema_start > 1.0, got: {errors5}"
    )
    print("[PASS] ema_start > 1.0 caught by validate()")

    # --- Test 6: final_lr > lr is caught ---
    bad_cfg3 = JEPATrainingConfig(lr=1e-4, final_lr=1e-3)
    errors6 = bad_cfg3.validate()
    assert any("final_lr" in e.lower() for e in errors6), (
        f"Should catch final_lr > lr, got: {errors6}"
    )
    print("[PASS] final_lr > lr caught by validate()")

    # --- Test 7: warmup_epochs > epochs is caught ---
    bad_cfg4 = JEPATrainingConfig(warmup_epochs=400, epochs=300)
    errors7 = bad_cfg4.validate()
    assert any("warmup" in e.lower() for e in errors7), (
        f"Should catch warmup_epochs > epochs, got: {errors7}"
    )
    print("[PASS] warmup_epochs > epochs caught by validate()")

    # --- Test 8: mismatched predictor_embed_dim / num_heads is caught ---
    bad_cfg5 = JEPATrainingConfig(predictor_embed_dim=384, predictor_num_heads=7)
    errors8 = bad_cfg5.validate()
    assert any("divisible" in e.lower() for e in errors8), (
        f"Should catch 384 not divisible by 7, got: {errors8}"
    )
    print("[PASS] predictor_embed_dim not divisible by num_heads caught")

    # --- Test 9: is_anneal=True without anneal_ckpt is caught ---
    bad_cfg6 = JEPATrainingConfig(is_anneal=True, anneal_ckpt=None)
    errors9 = bad_cfg6.validate()
    assert any("anneal" in e.lower() for e in errors9), (
        f"Should catch is_anneal=True without anneal_ckpt, got: {errors9}"
    )
    print("[PASS] is_anneal=True without anneal_ckpt caught")

    # --- Test 10: total_steps and warmup_steps utility ---
    cfg = JEPATrainingConfig.pretrain_256()
    steps_per_epoch = 1000
    assert cfg.total_steps(steps_per_epoch) == 300_000
    assert cfg.warmup_steps(steps_per_epoch) == 40_000
    print("[PASS] total_steps and warmup_steps utilities correct")

    # --- Test 11: assert_valid raises ValueError for bad config ---
    bad_cfg7 = JEPATrainingConfig(lr=-1.0)
    raised = False
    try:
        bad_cfg7.assert_valid()
    except ValueError:
        raised = True
    assert raised, "assert_valid should raise ValueError for invalid config"
    print("[PASS] assert_valid raises ValueError for bad config")

    # --- Test 12: DROID preset has auto_steps > 0 and normalize_reps=True ---
    droid = JEPATrainingConfig.droid_finetune()
    assert droid.auto_steps > 0, "DROID preset should have auto_steps > 0"
    assert droid.normalize_reps, "DROID preset should have normalize_reps=True"
    assert droid.encoder_lr_scale == 0.0, "DROID encoder should be frozen (lr_scale=0)"
    print("[PASS] DROID preset properties correct")

    print()
    print("All 12 self-tests passed.")
