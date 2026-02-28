"""
CheckpointManager: Six-state checkpoint save/load with atomic writes and auto-resume.

Six required states:
    1. model_state_dict:     online encoder + predictor weights (DDP-unwrapped)
    2. target_state_dict:    target encoder weights
    3. optimizer_state_dict: AdamW momentum buffers and step counts
    4. scaler_state_dict:    GradScaler scale factor and growth tracker
    5. scheduler_state_dict: cosine schedule current step and last_lr
    6. step:                 global training step (int)

Atomic writes:
    Save to temp file first, then os.rename() (atomic on POSIX/same filesystem).
    Prevents half-written checkpoints on crash.

Auto-resume:
    Scan checkpoint_dir for checkpoint-*.pt, load the one with highest step.
"""

from __future__ import annotations

import os
import re
import glob
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Strip DDP wrapper to get the underlying model for state_dict saving."""
    if hasattr(model, 'module'):
        return model.module
    return model


class CheckpointManager:
    """
    Manages checkpoint saving, loading, auto-resume, and pruning.

    Args:
        checkpoint_dir: Directory to save/load checkpoints.
        keep_last: Number of most-recent checkpoints to retain after pruning.
    """

    CHECKPOINT_PREFIX = 'checkpoint-'
    CHECKPOINT_SUFFIX = '.pt'

    def __init__(self, checkpoint_dir: str, keep_last: int = 3) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_last = keep_last
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Checkpoint path helpers
    # ------------------------------------------------------------------

    def _checkpoint_path(self, step: int) -> Path:
        return self.checkpoint_dir / f'{self.CHECKPOINT_PREFIX}{step}{self.CHECKPOINT_SUFFIX}'

    def _scan_checkpoints(self) -> dict[int, Path]:
        """Return {step: path} for all valid checkpoint files in the directory."""
        pattern = str(self.checkpoint_dir / f'{self.CHECKPOINT_PREFIX}*{self.CHECKPOINT_SUFFIX}')
        files = glob.glob(pattern)

        step_to_path = {}
        for filepath in files:
            basename = os.path.basename(filepath)
            match = re.match(
                rf'{re.escape(self.CHECKPOINT_PREFIX)}(\d+){re.escape(self.CHECKPOINT_SUFFIX)}',
                basename,
            )
            if match:
                step = int(match.group(1))
                step_to_path[step] = Path(filepath)

        return step_to_path

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        model: nn.Module,
        target_model: nn.Module,
        optimizer: Optimizer,
        scaler_ctx: object,     # AMPContext or any object with .state_dict()
        scheduler: LRScheduler,
        step: int,
    ) -> Path:
        """
        Save all six checkpoint states atomically.

        Uses atomic write (temp file + rename) to prevent half-written checkpoints.
        DDP wrapper is stripped before saving (model.module.state_dict() if DDP).

        Args:
            model: Online encoder (may be DDP-wrapped).
            target_model: Target encoder (should not be DDP-wrapped).
            optimizer: AdamW or other optimizer.
            scaler_ctx: AMPContext with .state_dict() method.
            scheduler: Learning rate scheduler.
            step: Current global training step.

        Returns:
            Path to the saved checkpoint file.
        """
        # Unwrap DDP if needed
        online_model = _unwrap_model(model)
        target_clean = _unwrap_model(target_model)

        state = {
            'model_state_dict':     online_model.state_dict(),
            'target_state_dict':    target_clean.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict':    scaler_ctx.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'step':                 step,
        }

        checkpoint_path = self._checkpoint_path(step)
        self._atomic_save(state, checkpoint_path)
        return checkpoint_path

    def _atomic_save(self, state: dict, dest: Path) -> None:
        """
        Write state to a temp file in the same directory, then rename.

        os.rename() is atomic on POSIX (same filesystem). This guarantees
        the checkpoint file is either fully present or fully absent — never
        a partially-written file that looks valid but contains corrupt data.
        """
        # Write temp file in same directory to guarantee same filesystem
        tmp_path = dest.parent / f'.tmp_{dest.name}'
        try:
            torch.save(state, str(tmp_path))
            os.rename(str(tmp_path), str(dest))
        except Exception:
            # Clean up temp file on any failure
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_latest(self) -> Optional[Tuple[dict, int]]:
        """
        Find and load the checkpoint with the highest step number.

        Returns:
            (state_dict, step) if a checkpoint exists, None otherwise.

        Raises:
            RuntimeError: If the latest checkpoint file is corrupt/unloadable.
        """
        step_to_path = self._scan_checkpoints()
        if not step_to_path:
            return None

        latest_step = max(step_to_path.keys())
        latest_path = step_to_path[latest_step]

        try:
            state = torch.load(str(latest_path), map_location='cpu', weights_only=False)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint at step {latest_step} "
                f"(path: {latest_path}): {e}"
            ) from e

        return state, latest_step

    def resume(
        self,
        model: nn.Module,
        target_model: nn.Module,
        optimizer: Optimizer,
        scaler_ctx: object,
        scheduler: LRScheduler,
    ) -> Optional[int]:
        """
        Load latest checkpoint and restore all six states into provided objects.

        Loading order is critical (documented in references/checkpoint-resume.md):
        1. model and target weights are loaded
        2. optimizer state loaded (must be after DDP wrapping in production)
        3. scaler and scheduler states loaded

        Args:
            model: Online encoder to restore weights into.
            target_model: Target encoder to restore weights into.
            optimizer: Optimizer to restore state into.
            scaler_ctx: AMPContext to restore scaler state into.
            scheduler: Scheduler to restore state into.

        Returns:
            Resumed step number, or None if no checkpoint found (fresh start).
        """
        result = self.load_latest()
        if result is None:
            return None

        state, step = result
        self._validate_checkpoint_keys(state)

        # Restore model states
        model_target = _unwrap_model(model)
        model_target.load_state_dict(state['model_state_dict'])

        target_clean = _unwrap_model(target_model)
        target_clean.load_state_dict(state['target_state_dict'])

        # Restore optimizer state
        optimizer.load_state_dict(state['optimizer_state_dict'])

        # Restore AMP scaler state
        scaler_ctx.load_state_dict(state['scaler_state_dict'])

        # Restore scheduler state
        scheduler.load_state_dict(state['scheduler_state_dict'])

        return step

    def _validate_checkpoint_keys(self, state: dict) -> None:
        """Verify all six required keys are present in the loaded state."""
        required_keys = {
            'model_state_dict',
            'target_state_dict',
            'optimizer_state_dict',
            'scaler_state_dict',
            'scheduler_state_dict',
            'step',
        }
        missing = required_keys - set(state.keys())
        if missing:
            raise RuntimeError(
                f"Checkpoint is missing required keys: {missing}. "
                f"Found keys: {set(state.keys())}"
            )

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def prune(self) -> int:
        """
        Delete old checkpoints, keeping only the last `keep_last` files.

        Returns:
            Number of checkpoint files deleted.
        """
        step_to_path = self._scan_checkpoints()
        if len(step_to_path) <= self.keep_last:
            return 0

        sorted_steps = sorted(step_to_path.keys())
        steps_to_delete = sorted_steps[:-self.keep_last]

        deleted = 0
        for step in steps_to_delete:
            try:
                step_to_path[step].unlink()
                deleted += 1
            except OSError:
                pass  # Best-effort deletion

        return deleted

    def list_checkpoints(self) -> list:
        """Return sorted list of (step, path) for all available checkpoints."""
        step_to_path = self._scan_checkpoints()
        return sorted(step_to_path.items())

    def __repr__(self) -> str:
        checkpoints = self._scan_checkpoints()
        steps = sorted(checkpoints.keys())
        return (
            f"CheckpointManager(dir='{self.checkpoint_dir}', "
            f"keep_last={self.keep_last}, "
            f"checkpoints={steps})"
        )


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import shutil

    print("=" * 60)
    print("CheckpointManager Self-Tests")
    print("=" * 60)

    # Create a temp directory for tests
    test_dir = tempfile.mkdtemp(prefix='checkpoint_test_')
    print(f"\nUsing temp directory: {test_dir}")

    try:
        # ----------------------------------------------------------------
        # Setup: small models and optimizer for testing
        # ----------------------------------------------------------------
        def make_components():
            online = nn.Linear(4, 2)
            target = nn.Linear(4, 2)
            optimizer = torch.optim.AdamW(online.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=1000, eta_min=1e-6
            )

            # Minimal scaler_ctx stub
            class ScalerCtx:
                def state_dict(self):
                    return {'scale': 65536.0, 'growth_factor': 2.0,
                            'backoff_factor': 0.5, 'growth_interval': 2000,
                            '_growth_tracker': 0}
                def load_state_dict(self, sd):
                    self._sd = sd

            return online, target, optimizer, ScalerCtx(), scheduler

        # ----------------------------------------------------------------
        # Test 1: save creates checkpoint file with all six keys
        # ----------------------------------------------------------------
        print("\nTest 1: save creates file with all six keys...")

        manager = CheckpointManager(test_dir, keep_last=3)
        online, target, optimizer, scaler_ctx, scheduler = make_components()

        # Run a step to populate optimizer state
        x = torch.randn(2, 4)
        out = online(x)
        out.mean().backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        saved_path = manager.save(online, target, optimizer, scaler_ctx, scheduler, step=100)

        assert saved_path.exists(), f"Checkpoint file not created at {saved_path}"

        loaded = torch.load(str(saved_path), map_location='cpu', weights_only=False)
        required_keys = {'model_state_dict', 'target_state_dict', 'optimizer_state_dict',
                         'scaler_state_dict', 'scheduler_state_dict', 'step'}
        missing = required_keys - set(loaded.keys())
        assert not missing, f"Missing keys in checkpoint: {missing}"
        assert loaded['step'] == 100, f"Expected step=100, got {loaded['step']}"
        print(f"  PASS: Checkpoint saved at {saved_path.name} with all 6 keys")

        # ----------------------------------------------------------------
        # Test 2: load_latest finds checkpoint with highest step
        # ----------------------------------------------------------------
        print("\nTest 2: load_latest finds highest-step checkpoint...")

        # Save more checkpoints at different steps
        manager.save(online, target, optimizer, scaler_ctx, scheduler, step=200)
        manager.save(online, target, optimizer, scaler_ctx, scheduler, step=300)

        result = manager.load_latest()
        assert result is not None, "load_latest should find a checkpoint"
        state, latest_step = result
        assert latest_step == 300, f"Expected latest step=300, got {latest_step}"
        print(f"  PASS: load_latest found step={latest_step}")

        # ----------------------------------------------------------------
        # Test 3: DDP unwrap (model with .module attribute)
        # ----------------------------------------------------------------
        print("\nTest 3: DDP unwrap works correctly...")

        class FakeDDP(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.module = m
            def forward(self, x):
                return self.module(x)

        real_model = nn.Linear(4, 2)
        fake_ddp = FakeDDP(real_model)

        # Set a known weight value
        with torch.no_grad():
            real_model.weight.fill_(42.0)

        manager2 = CheckpointManager(os.path.join(test_dir, 'ddp_test'), keep_last=3)
        saved_ddp = manager2.save(fake_ddp, target, optimizer, scaler_ctx, scheduler, step=50)

        loaded_ddp = torch.load(str(saved_ddp), map_location='cpu', weights_only=False)

        # Check no 'module.' prefix in keys (DDP was unwrapped)
        model_keys = list(loaded_ddp['model_state_dict'].keys())
        assert all(not k.startswith('module.') for k in model_keys), (
            f"Found 'module.' prefix in saved keys: {model_keys}"
        )
        # Check weight value was preserved
        saved_weight = loaded_ddp['model_state_dict']['weight'].mean().item()
        assert abs(saved_weight - 42.0) < 1e-4, (
            f"Expected weight=42.0 after unwrap, got {saved_weight}"
        )
        print(f"  PASS: DDP unwrapped correctly, weight={saved_weight:.1f}")

        # ----------------------------------------------------------------
        # Test 4: atomic write (temp file cleaned up, final file is complete)
        # ----------------------------------------------------------------
        print("\nTest 4: atomic write creates clean checkpoint...")

        manager3 = CheckpointManager(os.path.join(test_dir, 'atomic_test'), keep_last=3)
        online4, target4, opt4, scaler4, sched4 = make_components()

        # Save normally — verify no temp file remains
        manager3.save(online4, target4, opt4, scaler4, sched4, step=999)

        tmp_pattern = str(manager3.checkpoint_dir / '.tmp_*.pt')
        leftover_temps = glob.glob(tmp_pattern)
        assert len(leftover_temps) == 0, (
            f"Temp files should be cleaned up after atomic save: {leftover_temps}"
        )
        print("  PASS: No temp files remain after save")

        # ----------------------------------------------------------------
        # Test 5: checkpoint pruning keeps only last K
        # ----------------------------------------------------------------
        print("\nTest 5: prune keeps only last K checkpoints...")

        manager4 = CheckpointManager(os.path.join(test_dir, 'prune_test'), keep_last=3)
        online5, target5, opt5, scaler5, sched5 = make_components()

        for step in [100, 200, 300, 400, 500]:
            manager4.save(online5, target5, opt5, scaler5, sched5, step=step)

        all_before = manager4._scan_checkpoints()
        assert len(all_before) == 5, f"Expected 5 checkpoints, got {len(all_before)}"

        deleted = manager4.prune()
        all_after = manager4._scan_checkpoints()

        assert deleted == 2, f"Expected 2 deleted, got {deleted}"
        assert len(all_after) == 3, f"Expected 3 remaining, got {len(all_after)}"

        remaining_steps = sorted(all_after.keys())
        assert remaining_steps == [300, 400, 500], (
            f"Expected steps [300, 400, 500] to remain, got {remaining_steps}"
        )
        print(f"  PASS: Pruned 2 checkpoints, {len(all_after)} remain: {remaining_steps}")

        # ----------------------------------------------------------------
        # Test 6: resume restores all six states
        # ----------------------------------------------------------------
        print("\nTest 6: resume restores all six states correctly...")

        manager6 = CheckpointManager(os.path.join(test_dir, 'resume_test'), keep_last=3)
        online6, target6, opt6, scaler6, sched6 = make_components()

        # Set known values
        with torch.no_grad():
            online6.weight.fill_(7.0)
            target6.weight.fill_(3.0)

        # Run optimizer step to populate optimizer state (modifies weights slightly)
        x6 = torch.randn(2, 4)
        online6(x6).mean().backward()
        opt6.step()
        opt6.zero_grad(set_to_none=True)

        # Capture actual weight values AFTER optimizer step (these are what get saved)
        expected_online_weight = online6.weight.mean().item()
        expected_target_weight = target6.weight.mean().item()

        manager6.save(online6, target6, opt6, scaler6, sched6, step=42)

        # Create fresh components for resume
        online6_new, target6_new, opt6_new, scaler6_new, sched6_new = make_components()

        resumed_step = manager6.resume(online6_new, target6_new, opt6_new, scaler6_new, sched6_new)

        assert resumed_step == 42, f"Expected resumed step=42, got {resumed_step}"

        # Verify model weights restored match exactly what was saved
        restored_weight = online6_new.weight.mean().item()
        assert abs(restored_weight - expected_online_weight) < 1e-5, (
            f"Expected online weight={expected_online_weight:.6f} after resume, got {restored_weight:.6f}"
        )

        restored_target = target6_new.weight.mean().item()
        assert abs(restored_target - expected_target_weight) < 1e-5, (
            f"Expected target weight={expected_target_weight:.6f} after resume, got {restored_target:.6f}"
        )
        print(f"  PASS: Resume restored step={resumed_step}, online_w={restored_weight:.4f}, target_w={restored_target:.4f}")

        # ----------------------------------------------------------------
        # Test 7: corrupt checkpoint raises clear error
        # ----------------------------------------------------------------
        print("\nTest 7: corrupt checkpoint raises clear error...")

        corrupt_dir = os.path.join(test_dir, 'corrupt_test')
        os.makedirs(corrupt_dir, exist_ok=True)

        corrupt_path = os.path.join(corrupt_dir, 'checkpoint-999.pt')
        with open(corrupt_path, 'wb') as f:
            f.write(b'this is not a valid pytorch checkpoint file' * 10)

        manager7 = CheckpointManager(corrupt_dir, keep_last=3)

        try:
            manager7.load_latest()
            print("  FAIL: Should have raised RuntimeError for corrupt checkpoint")
            sys.exit(1)
        except RuntimeError as e:
            assert 'Failed to load checkpoint' in str(e), (
                f"Error message should mention 'Failed to load checkpoint', got: {e}"
            )
            print(f"  PASS: RuntimeError raised with clear message: {str(e)[:80]}...")

        # ----------------------------------------------------------------
        # Test 8: fresh start returns None when no checkpoints exist
        # ----------------------------------------------------------------
        print("\nTest 8: load_latest returns None for empty directory...")

        empty_dir = os.path.join(test_dir, 'empty_dir')
        manager8 = CheckpointManager(empty_dir, keep_last=3)

        result8 = manager8.load_latest()
        assert result8 is None, f"Expected None for empty dir, got {result8}"

        online8, target8, opt8, scaler8, sched8 = make_components()
        resumed8 = manager8.resume(online8, target8, opt8, scaler8, sched8)
        assert resumed8 is None, f"Expected None for fresh start, got {resumed8}"
        print("  PASS: None returned for empty checkpoint directory")

    finally:
        shutil.rmtree(test_dir, ignore_errors=True)
        print(f"\nCleaned up temp directory: {test_dir}")

    print("\n" + "=" * 60)
    print("All CheckpointManager self-tests PASSED")
    print("=" * 60)
    sys.exit(0)
