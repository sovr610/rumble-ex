"""
SelfSupervisedTrainer: Main training loop pulling together all components.

Implements the 10-step ordering from SKILL.md:
    1. optimizer.zero_grad(set_to_none=True)
    2. with autocast: loss = forward(online, predictor, target, views)
    3. scaler.scale(loss).backward()
    4. scaler.unscale_(optimizer)          -- MUST precede clip
    5. grad_norm = clip_grad_norm_(params) -- at true scale
    6. scaler.step(optimizer)             -- skips if inf/nan
    7. scaler.update()
    8. scheduler.step()
    9. ema_update(online, target, step)   -- AFTER optimizer step
    10. log metrics

Components:
    - AMPContext: autocast + GradScaler (amp_gradient_template.py)
    - EMAUpdater: cosine-annealed EMA (ema_template.py)
    - CheckpointManager: six-state save/load (checkpoint_template.py)
    - WandbLogger: rank-0 metric logging (wandb_logger_template.py)
    - TrainingConfig: all config (training_config_template.py)

CRITICAL: module.train(False) for inference mode — never use the blocked method.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Optional, List, Iterator
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Import co-located templates
_ASSETS_DIR = Path(__file__).parent
sys.path.insert(0, str(_ASSETS_DIR))

from amp_gradient_template import AMPContext
from ema_template import EMAUpdater
from checkpoint_template import CheckpointManager
from wandb_logger_template import WandbLogger
from training_config_template import TrainingConfig


# ---------------------------------------------------------------------------
# Cosine Warmup Scheduler
# ---------------------------------------------------------------------------

class CosineWarmupScheduler:
    """
    Linear warmup followed by cosine decay learning rate scheduler.

    Warmup phase (0 <= step < warmup_steps):
        lr = base_lr * step / warmup_steps

    Cosine decay phase (warmup_steps <= step <= total_steps):
        lr = lr_min + 0.5 * (base_lr - lr_min) * (1 + cos(pi * t / T))
        where t = step - warmup_steps, T = total_steps - warmup_steps

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Number of linear warmup steps.
        total_steps: Total training steps.
        lr_min: Minimum lr at end of cosine decay.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        lr_min: float = 1e-6,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.lr_min = lr_min
        self._step = 0

        # Store base learning rates from optimizer
        self._base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self._last_lrs = list(self._base_lrs)

    def step(self) -> None:
        """Advance the scheduler by one step and update optimizer lr."""
        lrs = self._compute_lrs(self._step)
        for pg, lr in zip(self.optimizer.param_groups, lrs):
            pg['lr'] = lr
        self._last_lrs = lrs
        self._step += 1

    def _compute_lrs(self, step: int) -> List[float]:
        """Compute learning rates at the given step."""
        return [self._compute_lr(step, base_lr) for base_lr in self._base_lrs]

    def _compute_lr(self, step: int, base_lr: float) -> float:
        """Compute lr for one parameter group at the given step."""
        if step < self.warmup_steps:
            # Linear warmup: 0 -> base_lr over warmup_steps
            return base_lr * step / max(1, self.warmup_steps)
        else:
            # Cosine decay: base_lr -> lr_min
            t = step - self.warmup_steps
            T = max(1, self.total_steps - self.warmup_steps)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * t / T))
            return self.lr_min + (base_lr - self.lr_min) * cosine_factor

    def get_last_lr(self) -> List[float]:
        """Return the last computed learning rates (one per param group)."""
        return self._last_lrs

    def state_dict(self) -> dict:
        """Return scheduler state for checkpointing."""
        return {
            '_step': self._step,
            '_base_lrs': self._base_lrs,
            '_last_lrs': self._last_lrs,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'lr_min': self.lr_min,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore scheduler state from checkpoint."""
        self._step = state['_step']
        self._base_lrs = state['_base_lrs']
        self._last_lrs = state['_last_lrs']
        # Restore optimizer lr to match saved state
        for pg, lr in zip(self.optimizer.param_groups, self._last_lrs):
            pg['lr'] = lr


# ---------------------------------------------------------------------------
# Mock SSL Forward (for template / testing)
# ---------------------------------------------------------------------------

def mock_ssl_forward(
    online_encoder: nn.Module,
    predictor: nn.Module,
    target_encoder: nn.Module,
    view1: Tensor,
    view2: Tensor,
) -> Tensor:
    """
    Minimal SSL forward pass for template purposes.

    In production, replace with your actual SSL objective (BYOL, SimCLR, DINO, etc.).

    Structure:
        online_proj = predictor(online_encoder(view1))
        with torch.no_grad():
            target_proj = target_encoder(view2)
        loss = -cosine_similarity(online_proj, target_proj).mean()

    The target encoder forward is wrapped in no_grad because:
        1. Target has requires_grad=False on all params
        2. Gradients must not flow through target path
    """
    # Online path: gradients flow through here
    online_out = online_encoder(view1)
    online_pred = predictor(online_out)

    # Target path: no gradients
    with torch.no_grad():
        target_out = target_encoder(view2)

    # Normalize for cosine similarity (MSE on normalized = BYOL-style loss)
    online_norm = F.normalize(online_pred, dim=-1)
    target_norm = F.normalize(target_out, dim=-1)

    # MSE loss on unit vectors: equivalent to 2 * (1 - cosine_similarity)
    loss = F.mse_loss(online_norm, target_norm.detach())
    return loss


# ---------------------------------------------------------------------------
# SelfSupervisedTrainer
# ---------------------------------------------------------------------------

class SelfSupervisedTrainer:
    """
    Production self-supervised training loop with correct AMP/EMA/DDP ordering.

    The 10-step sequence is enforced in train() and must not be reordered.
    See SKILL.md 'Critical Operation Ordering' for justification.

    Args:
        cfg: TrainingConfig with all hyperparameters.
        rank: Global process rank (0 for single-GPU).
        world_size: Total number of processes (1 for single-GPU).
    """

    def __init__(
        self,
        cfg: TrainingConfig,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'

        # Validate config before any setup
        cfg.validate()

        # Component placeholders (set during setup())
        self.online_encoder: Optional[nn.Module] = None
        self.predictor: Optional[nn.Module] = None
        self.target_encoder: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.AdamW] = None
        self.scheduler: Optional[CosineWarmupScheduler] = None
        self.amp_ctx: Optional[AMPContext] = None
        self.ema_updater: Optional[EMAUpdater] = None
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self.logger: Optional[WandbLogger] = None
        self.start_step: int = 0

    def setup(self) -> None:
        """
        Initialize all components in the correct order.

        Order:
        1. Create online encoder + predictor (randomly initialized)
        2. Create target encoder (copy of online, frozen)
        3. Optionally convert SyncBatchNorm and wrap online with DDP
        4. Create optimizer over online + predictor parameters
        5. Create CosineWarmupScheduler
        6. Create AMPContext
        7. Create EMAUpdater
        8. Create CheckpointManager
        9. Create WandbLogger
        10. Auto-resume if checkpoint exists
        """
        cfg = self.cfg

        # Step 1+2: Create model pair
        # In production, replace with your actual encoder architecture
        embed_dim = 64
        self.online_encoder = nn.Sequential(
            nn.Linear(32, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        ).to(self.device)

        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        ).to(self.device)

        # Target encoder: same architecture, frozen
        import copy
        self.target_encoder = copy.deepcopy(self.online_encoder).to(self.device)
        for param in self.target_encoder.parameters():
            param.requires_grad_(False)

        # Step 3: DDP setup (single-GPU path: skip wrapping)
        # For multi-GPU, use ddp_setup_template.py:
        #     online = wrap_online_model(online, rank, sync_batchnorm=cfg.sync_batchnorm)
        # Target encoder is NEVER wrapped with DDP

        # Step 4: Optimizer (AdamW with ViT hyperparameters)
        # Only optimize online encoder + predictor (not target — it's frozen)
        online_params = list(self.online_encoder.parameters()) + \
                        list(self.predictor.parameters())
        self.optimizer = torch.optim.AdamW(
            online_params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
        )

        # Step 5: Cosine warmup scheduler
        self.scheduler = CosineWarmupScheduler(
            optimizer=self.optimizer,
            warmup_steps=cfg.warmup_steps,
            total_steps=cfg.total_steps,
            lr_min=cfg.lr_min,
        )

        # Step 6: AMP context
        self.amp_ctx = AMPContext(
            dtype=cfg.torch_dtype,
            enabled=cfg.amp_enabled,
            scaler_enabled=cfg.grad_scaler_enabled,
        )

        # Step 7: EMA updater
        self.ema_updater = EMAUpdater(
            tau_base=cfg.ema_tau_base,
            tau_final=cfg.ema_tau_final,
            total_steps=cfg.total_steps,
        )

        # Step 8: Initial synchronization of target with online
        self.ema_updater.initial_sync(self.online_encoder, self.target_encoder)

        # Step 9: Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=cfg.checkpoint_dir,
            keep_last=cfg.keep_checkpoints,
        )

        # Step 10: W&B logger
        self.logger = WandbLogger(cfg=cfg, rank=self.rank)

        # Step 11: Auto-resume
        if cfg.auto_resume:
            resumed = self.checkpoint_manager.resume(
                self.online_encoder,
                self.target_encoder,
                self.optimizer,
                self.amp_ctx,
                self.scheduler,
            )
            if resumed is not None:
                self.start_step = resumed + 1
                if self.rank == 0:
                    print(f"[Trainer] Resumed from step {resumed}, starting at {self.start_step}")

    def train(
        self,
        data_iter: Optional[Iterator] = None,
        num_steps: Optional[int] = None,
    ) -> list:
        """
        Main training loop enforcing the 10-step ordering.

        The ordering is non-negotiable:
            Step 4 (unscale) BEFORE Step 5 (clip)
            Step 5 (clip) BEFORE Step 6 (step)
            Step 9 (EMA) AFTER Step 6 (optimizer step)

        Args:
            data_iter: Iterator yielding batches. If None, uses synthetic data.
            num_steps: Override total training steps (useful for testing).

        Returns:
            List of (step, loss, grad_norm, tau) dicts for each step.
        """
        assert self.online_encoder is not None, "Call setup() before train()"

        cfg = self.cfg
        total = num_steps if num_steps is not None else cfg.total_steps

        history = []

        for step in range(self.start_step, total):

            # ---- Get batch (synthetic for template) ----
            if data_iter is not None:
                try:
                    batch = next(data_iter)
                    view1, view2 = batch[0].to(self.device), batch[1].to(self.device)
                except StopIteration:
                    break
            else:
                # Synthetic views: (B, input_dim)
                view1 = torch.randn(8, 32, device=self.device)
                view2 = torch.randn(8, 32, device=self.device)

            # ================================================================
            # 10-STEP ORDERING — DO NOT REORDER
            # ================================================================

            # Step 1: Zero gradients (set_to_none saves memory vs zeroing)
            self.optimizer.zero_grad(set_to_none=True)

            # Step 2: Forward pass under autocast
            with self.amp_ctx.autocast():
                loss = mock_ssl_forward(
                    self.online_encoder,
                    self.predictor,
                    self.target_encoder,
                    view1,
                    view2,
                )

            # Step 3: Scaled backward (gradients are at scale_factor * true_grad)
            self.amp_ctx.backward(loss)

            # Step 4+5: Unscale THEN clip (MUST be in this order)
            #           unscale_and_clip returns pre-clip norm at true scale
            online_params = list(self.online_encoder.parameters()) + \
                            list(self.predictor.parameters())
            grad_norm = self.amp_ctx.unscale_and_clip(
                self.optimizer,
                online_params,
                max_norm=cfg.max_grad_norm,
            )

            # Step 6+7: Optimizer step (skips if inf/nan) + scale update
            self.amp_ctx.step_and_update(self.optimizer)

            # Step 8: Scheduler step
            self.scheduler.step()

            # Step 9: EMA update — MUST be after optimizer.step()
            #         Target sees the UPDATED online weights
            tau = self.ema_updater.update(self.online_encoder, self.target_encoder, step)

            # Step 10: Log metrics
            lr = self.scheduler.get_last_lr()[0]
            gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

            if step % cfg.log_every == 0:
                loss_val = loss.item()
                self.logger.log_step(
                    step=step,
                    loss=loss_val,
                    grad_norm=grad_norm,
                    tau=tau,
                    lr=lr,
                    gpu_mem_gb=gpu_mem,
                )
                history.append({
                    'step': step,
                    'loss': loss_val,
                    'grad_norm': grad_norm,
                    'tau': tau,
                    'lr': lr,
                })

            # ================================================================
            # Periodic checkpoint save
            # ================================================================
            if step > 0 and step % cfg.checkpoint_every == 0 and self.rank == 0:
                self.checkpoint_manager.save(
                    self.online_encoder,
                    self.target_encoder,
                    self.optimizer,
                    self.amp_ctx,
                    self.scheduler,
                    step,
                )
                self.checkpoint_manager.prune()

        return history

    def save_checkpoint(self, step: int) -> None:
        """Manually save checkpoint at the given step (rank 0 only)."""
        if self.rank == 0 and self.checkpoint_manager is not None:
            self.checkpoint_manager.save(
                self.online_encoder,
                self.target_encoder,
                self.optimizer,
                self.amp_ctx,
                self.scheduler,
                step,
            )

    def load_checkpoint(self) -> Optional[int]:
        """Load latest checkpoint and return the step number, or None."""
        if self.checkpoint_manager is None:
            return None
        return self.checkpoint_manager.resume(
            self.online_encoder,
            self.target_encoder,
            self.optimizer,
            self.amp_ctx,
            self.scheduler,
        )

    def cleanup(self) -> None:
        """
        Clean up resources: finish W&B logger, destroy DDP process group.

        Call in a finally block:
            try:
                trainer.train()
            finally:
                trainer.cleanup()
        """
        if self.logger is not None:
            self.logger.finish()

        # In multi-GPU setup, destroy DDP process group:
        # from ddp_setup_template import cleanup_distributed
        # cleanup_distributed()


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    import shutil
    import copy

    print("=" * 60)
    print("SelfSupervisedTrainer Self-Tests")
    print("=" * 60)

    # Use a temp directory for checkpoints
    test_dir = tempfile.mkdtemp(prefix='trainer_test_')

    try:
        # ----------------------------------------------------------------
        # Test 1: 10 steps produce finite loss and positive grad_norm
        # ----------------------------------------------------------------
        print("\nTest 1: 10 training steps produce finite loss...")

        cfg = TrainingConfig(
            checkpoint_dir=test_dir,
            total_steps=10,
            warmup_steps=2,
            checkpoint_every=100,  # Don't checkpoint during test
            auto_resume=False,
            amp_enabled=False,      # CPU-safe: disable AMP
            grad_scaler_enabled=False,
        )

        trainer = SelfSupervisedTrainer(cfg, rank=0, world_size=1)
        trainer.setup()
        history = trainer.train(num_steps=10)

        assert len(history) == 10, f"Expected 10 history entries, got {len(history)}"

        for entry in history:
            assert math.isfinite(entry['loss']), (
                f"Loss is not finite at step {entry['step']}: {entry['loss']}"
            )
            assert entry['grad_norm'] > 0, (
                f"grad_norm should be positive at step {entry['step']}: {entry['grad_norm']}"
            )
            assert math.isfinite(entry['grad_norm']), (
                f"grad_norm is not finite at step {entry['step']}: {entry['grad_norm']}"
            )

        print(f"  PASS: 10 steps completed. "
              f"Final loss={history[-1]['loss']:.4f}, "
              f"grad_norm={history[-1]['grad_norm']:.4f}")

        # ----------------------------------------------------------------
        # Test 2: EMA tau increases over steps
        # ----------------------------------------------------------------
        print("\nTest 2: EMA tau increases monotonically...")

        cfg2 = TrainingConfig(
            checkpoint_dir=test_dir,
            total_steps=20,
            warmup_steps=2,
            checkpoint_every=100,
            auto_resume=False,
            amp_enabled=False,
            grad_scaler_enabled=False,
        )
        trainer2 = SelfSupervisedTrainer(cfg2, rank=0, world_size=1)
        trainer2.setup()
        history2 = trainer2.train(num_steps=20)

        taus = [h['tau'] for h in history2]
        assert taus[0] <= taus[-1], (
            f"tau should increase over training. tau[0]={taus[0]:.6f}, tau[-1]={taus[-1]:.6f}"
        )
        print(f"  PASS: tau increased from {taus[0]:.6f} to {taus[-1]:.6f}")

        # ----------------------------------------------------------------
        # Test 3: Checkpoint save+load round-trip resumes correctly
        # ----------------------------------------------------------------
        print("\nTest 3: Checkpoint save+load round-trip...")

        ckpt_dir = os.path.join(test_dir, 'ckpt_test')
        cfg3 = TrainingConfig(
            checkpoint_dir=ckpt_dir,
            total_steps=10,
            warmup_steps=2,
            checkpoint_every=100,
            auto_resume=False,
            amp_enabled=False,
            grad_scaler_enabled=False,
        )

        trainer3 = SelfSupervisedTrainer(cfg3, rank=0, world_size=1)
        trainer3.setup()
        history3 = trainer3.train(num_steps=5)

        # Save checkpoint at step 4
        trainer3.save_checkpoint(step=4)

        # Capture weights before save for comparison
        online_weight_before = trainer3.online_encoder[0].weight.data.clone()

        # Create fresh trainer with auto_resume=True
        cfg3b = TrainingConfig(
            checkpoint_dir=ckpt_dir,
            total_steps=10,
            warmup_steps=2,
            checkpoint_every=100,
            auto_resume=True,
            amp_enabled=False,
            grad_scaler_enabled=False,
        )
        trainer3b = SelfSupervisedTrainer(cfg3b, rank=0, world_size=1)
        trainer3b.setup()

        # Verify resume happened
        assert trainer3b.start_step == 5, (
            f"Expected to resume from step 5, got {trainer3b.start_step}"
        )

        # Verify weights match
        online_weight_after = trainer3b.online_encoder[0].weight.data
        weight_diff = (online_weight_before - online_weight_after).abs().max().item()
        assert weight_diff < 1e-5, (
            f"Online encoder weights should match after resume. Max diff: {weight_diff}"
        )
        print(f"  PASS: Resumed from step={trainer3b.start_step}, weight_diff={weight_diff:.2e}")

        # ----------------------------------------------------------------
        # Test 4: EMA diverges from online after updates
        # ----------------------------------------------------------------
        print("\nTest 4: Target encoder diverges from online after training...")

        cfg4 = TrainingConfig(
            checkpoint_dir=test_dir,
            total_steps=50,
            warmup_steps=5,
            checkpoint_every=100,
            auto_resume=False,
            amp_enabled=False,
            grad_scaler_enabled=False,
        )
        trainer4 = SelfSupervisedTrainer(cfg4, rank=0, world_size=1)
        trainer4.setup()

        # Confirm they start identical (initial_sync)
        dist_before = trainer4.ema_updater.compute_distance(
            trainer4.online_encoder, trainer4.target_encoder
        )
        assert dist_before < 1e-8, (
            f"Initial distance should be ~0 after initial_sync, got {dist_before}"
        )

        trainer4.train(num_steps=50)

        dist_after = trainer4.ema_updater.compute_distance(
            trainer4.online_encoder, trainer4.target_encoder
        )
        assert dist_after > 0.0, "Target should diverge from online after training"
        assert math.isfinite(dist_after), f"Distance should be finite, got {dist_after}"
        print(f"  PASS: Distance increased from {dist_before:.2e} to {dist_after:.4f}")

        # ----------------------------------------------------------------
        # Test 5: All components initialized correctly
        # ----------------------------------------------------------------
        print("\nTest 5: All components initialized after setup()...")

        cfg5 = TrainingConfig(
            checkpoint_dir=test_dir,
            total_steps=10,
            warmup_steps=2,
            auto_resume=False,
            amp_enabled=False,
            grad_scaler_enabled=False,
        )
        trainer5 = SelfSupervisedTrainer(cfg5, rank=0, world_size=1)
        trainer5.setup()

        assert trainer5.online_encoder is not None, "online_encoder not initialized"
        assert trainer5.predictor is not None, "predictor not initialized"
        assert trainer5.target_encoder is not None, "target_encoder not initialized"
        assert trainer5.optimizer is not None, "optimizer not initialized"
        assert trainer5.scheduler is not None, "scheduler not initialized"
        assert trainer5.amp_ctx is not None, "amp_ctx not initialized"
        assert trainer5.ema_updater is not None, "ema_updater not initialized"
        assert trainer5.checkpoint_manager is not None, "checkpoint_manager not initialized"
        assert trainer5.logger is not None, "logger not initialized"

        # Verify target encoder is frozen
        assert all(not p.requires_grad for p in trainer5.target_encoder.parameters()), (
            "All target encoder params should be frozen"
        )
        print("  PASS: All 9 components initialized, target encoder frozen")

        # ----------------------------------------------------------------
        # Test 6: Learning rate warmup is linear
        # ----------------------------------------------------------------
        print("\nTest 6: LR warmup is linear...")

        cfg6 = TrainingConfig(
            total_steps=20,
            warmup_steps=10,
            lr=1e-3,
            lr_min=1e-6,
            checkpoint_dir=test_dir,
            auto_resume=False,
            amp_enabled=False,
            grad_scaler_enabled=False,
        )

        sched6 = CosineWarmupScheduler(
            optimizer=torch.optim.AdamW([torch.zeros(1)], lr=1e-3),
            warmup_steps=10,
            total_steps=20,
            lr_min=1e-6,
        )

        warmup_lrs = []
        for i in range(10):
            warmup_lrs.append(sched6._compute_lr(i, 1e-3))

        # At step 5 (midpoint of warmup), lr should be ~0.5 * base_lr
        lr_at_midpoint = warmup_lrs[5]
        expected_midpoint = 1e-3 * 5 / 10  # linear: 0.5 * base_lr
        assert abs(lr_at_midpoint - expected_midpoint) < 1e-8, (
            f"LR at midpoint of warmup should be {expected_midpoint:.6f}, got {lr_at_midpoint:.6f}"
        )
        print(f"  PASS: LR at warmup midpoint={lr_at_midpoint:.6f} (expected {expected_midpoint:.6f})")

    finally:
        shutil.rmtree(test_dir, ignore_errors=True)
        print(f"\nCleaned up temp directory: {test_dir}")

    print("\n" + "=" * 60)
    print("All SelfSupervisedTrainer self-tests PASSED")
    print("=" * 60)
    sys.exit(0)
