"""
checkpoint_manager_template.py
================================
CheckpointManager: robust checkpoint save/load with retry logic,
prefix stripping, and pretrained weight loading for V-JEPA 2.

Key features:
  - Atomic writes (temp file + rename) to prevent partial checkpoints
  - Exponential backoff retry (2^n + jitter) for transient NFS failures
  - Automatic stripping of "module." and "backbone." prefixes
  - strict=False loading for RoPE models (no pos_embed in checkpoint)
  - Checkpoint rotation to manage disk usage

Usage:
    manager = CheckpointManager(save_dir="/checkpoints/vjepa2/run1")

    # Save
    path = manager.save({"epoch": 5, "encoder": enc.state_dict(), ...}, epoch=5)

    # Load with retry
    ckpt = manager.load(path)

    # Load pretrained (auto-strips prefixes, strict=False)
    missing, unexpected = manager.load_pretrained(encoder, "pretrained.pth")
"""

from __future__ import annotations

import glob
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """
    Manages saving and loading of V-JEPA 2 training checkpoints.

    Checkpoint format:
        {
            "epoch":          int,
            "encoder":        state_dict,
            "predictor":      state_dict,
            "target_encoder": state_dict,
            "opt":            optimizer state_dict,
            "scaler":         grad_scaler state_dict or None,
        }

    Args:
        save_dir: Directory to save checkpoints in. Created if absent.
        max_retries: Number of load retry attempts on failure (default 5).
        keep_last: Number of most-recent checkpoints to retain (0 = keep all).
    """

    def __init__(
        self,
        save_dir: str,
        max_retries: int = 5,
        keep_last: int = 0,
    ) -> None:
        self.save_dir = save_dir
        self.max_retries = max_retries
        self.keep_last = keep_last
        os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        state: Dict[str, Any],
        epoch: int,
        tag: str = "",
    ) -> str:
        """
        Save a checkpoint dictionary to disk atomically.

        Uses write-to-tmp then rename to avoid partial writes that could
        leave a corrupt checkpoint on the filesystem.

        Args:
            state: Checkpoint dictionary (must contain "epoch" key).
            epoch: Epoch number used in the filename (zero-padded to 4 digits).
            tag: Optional suffix for the filename (e.g. "best").

        Returns:
            Absolute path of the saved checkpoint file.
        """
        suffix = f"_{tag}" if tag else ""
        filename = f"checkpoint_{epoch:04d}{suffix}.pth"
        path = os.path.join(self.save_dir, filename)
        tmp_path = path + ".tmp"

        try:
            torch.save(state, tmp_path)
            # Atomic rename: on POSIX filesystems this is atomic
            os.replace(tmp_path, path)
            logger.info("Checkpoint saved: %s", path)
        except Exception as e:
            # Clean up tmp file if rename failed
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise RuntimeError(f"Failed to save checkpoint to {path}: {e}") from e

        # Optionally rotate old checkpoints
        if self.keep_last > 0:
            self._rotate(exclude_tags=["best", "final"])

        return path

    # ------------------------------------------------------------------
    # Load with retry
    # ------------------------------------------------------------------

    def load(self, path: str) -> Dict[str, Any]:
        """
        Load a checkpoint with exponential backoff retry.

        Handles transient NFS/Lustre failures where a file exists but
        temporarily returns an I/O error.

        Backoff schedule: wait = 2^attempt + uniform(0, 1) seconds
          Attempt 0: ~1s,  Attempt 1: ~2s,  Attempt 2: ~4s,
          Attempt 3: ~8s,  Attempt 4: ~16s

        Args:
            path: Absolute or relative path to the checkpoint file.

        Returns:
            Checkpoint dictionary.

        Raises:
            RuntimeError: If all retry attempts fail.
        """
        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                # weights_only=False required for full state dicts
                # (optimizer states, scalers contain Python objects)
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                if attempt > 0:
                    logger.info("Checkpoint loaded after %d retries: %s", attempt, path)
                else:
                    logger.debug("Checkpoint loaded: %s", path)
                return ckpt
            except Exception as exc:
                last_exc = exc
                if attempt == self.max_retries - 1:
                    break

                wait = (2 ** attempt) + random.random()
                logger.warning(
                    "Checkpoint load attempt %d/%d failed (%s). Retrying in %.1fs...",
                    attempt + 1, self.max_retries, exc, wait,
                )
                time.sleep(wait)

        raise RuntimeError(
            f"Failed to load checkpoint after {self.max_retries} attempts: "
            f"{path}\nLast error: {last_exc}"
        ) from last_exc

    # ------------------------------------------------------------------
    # Load pretrained weights
    # ------------------------------------------------------------------

    def load_pretrained(
        self,
        model: nn.Module,
        path: str,
        key: str = "encoder",
        strict: bool = False,
    ) -> Tuple[List[str], List[str]]:
        """
        Load pretrained weights into a model with auto prefix-stripping.

        Handles checkpoints saved from DDP (module. prefix) or with
        backbone wrapper (backbone. prefix). Uses strict=False by default
        to allow loading into RoPE models that omit pos_embed.

        Args:
            model: nn.Module to load weights into.
            path: Path to the checkpoint file.
            key: Key inside the checkpoint dict to use as the state_dict.
                 If the key is absent, treats the whole checkpoint as a state_dict.
            strict: Whether to require exact key match (default False).

        Returns:
            Tuple of (missing_keys, unexpected_keys).
        """
        checkpoint = self.load(path)

        # Extract sub-dict if key exists
        if isinstance(checkpoint, dict) and key in checkpoint:
            state_dict = checkpoint[key]
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint  # Assume checkpoint is the state_dict

        # Strip common DDP / backbone prefixes
        for prefix in ("module.", "backbone.", "encoder."):
            n_with = sum(1 for k in state_dict if k.startswith(prefix))
            if n_with > len(state_dict) // 2:
                state_dict = self.strip_prefix(state_dict, prefix)
                logger.debug("Stripped prefix %r from state_dict (%d keys)", prefix, n_with)

        result = model.load_state_dict(state_dict, strict=strict)

        n_missing = len(result.missing_keys)
        n_unexpected = len(result.unexpected_keys)

        if n_missing > 0:
            logger.info(
                "Pretrained load: %d missing keys (expected for RoPE / head mismatch): %s%s",
                n_missing,
                result.missing_keys[:3],
                " ..." if n_missing > 3 else "",
            )
        if n_unexpected > 0:
            logger.info(
                "Pretrained load: %d unexpected keys: %s%s",
                n_unexpected,
                result.unexpected_keys[:3],
                " ..." if n_unexpected > 3 else "",
            )

        logger.info(
            "Pretrained weights loaded from %s (strict=%s, missing=%d, unexpected=%d)",
            path, strict, n_missing, n_unexpected,
        )

        return result.missing_keys, result.unexpected_keys

    # ------------------------------------------------------------------
    # Prefix stripping
    # ------------------------------------------------------------------

    @staticmethod
    def strip_prefix(
        state_dict: Dict[str, Any],
        prefix: str = "module.",
    ) -> Dict[str, Any]:
        """
        Remove a prefix from all keys in a state_dict.

        Keys that do not start with the prefix are preserved unchanged.

        Args:
            state_dict: OrderedDict from model.state_dict().
            prefix: Prefix string to remove (default "module.").

        Returns:
            New dict with prefix removed from matching keys.
        """
        new_state_dict: Dict[str, Any] = {}
        prefix_len = len(prefix)
        for key, value in state_dict.items():
            new_key = key[prefix_len:] if key.startswith(prefix) else key
            new_state_dict[new_key] = value
        return new_state_dict

    # ------------------------------------------------------------------
    # Checkpoint discovery
    # ------------------------------------------------------------------

    def latest_checkpoint(self, tag: str = "") -> Optional[str]:
        """
        Return the path to the most recent checkpoint, or None.

        Args:
            tag: Optional tag filter (e.g. "best"). Empty = untagged checkpoints.
        """
        if tag:
            pattern = os.path.join(self.save_dir, f"checkpoint_*_{tag}.pth")
        else:
            # Match regular checkpoints (no tag)
            pattern = os.path.join(self.save_dir, "checkpoint_????.pth")

        files = sorted(glob.glob(pattern))
        return files[-1] if files else None

    def list_checkpoints(self) -> List[str]:
        """Return sorted list of all checkpoint paths in save_dir."""
        return sorted(glob.glob(os.path.join(self.save_dir, "checkpoint_*.pth")))

    # ------------------------------------------------------------------
    # Resume helper
    # ------------------------------------------------------------------

    def resume_epoch(
        self,
        encoder: Optional[nn.Module] = None,
        predictor: Optional[nn.Module] = None,
        target_encoder: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler: Optional[Any] = None,
    ) -> int:
        """
        Load the latest checkpoint and restore model/optimizer state.

        Returns the epoch to start training from (last_epoch + 1).
        Returns 0 if no checkpoint is found.
        """
        path = self.latest_checkpoint()
        if path is None:
            logger.info("No checkpoint found in %s — starting from scratch", self.save_dir)
            return 0

        logger.info("Resuming from checkpoint: %s", path)
        ckpt = self.load(path)

        def _load_state(module: Optional[nn.Module], key: str) -> None:
            if module is None or key not in ckpt:
                return
            sd = self.strip_prefix(ckpt[key], "module.")
            module.load_state_dict(sd)

        _load_state(encoder, "encoder")
        _load_state(predictor, "predictor")
        _load_state(target_encoder, "target_encoder")

        if optimizer is not None and "opt" in ckpt:
            optimizer.load_state_dict(ckpt["opt"])

        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])

        start_epoch = ckpt.get("epoch", -1) + 1
        logger.info("Resumed at epoch %d", start_epoch)
        return start_epoch

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _rotate(self, exclude_tags: Optional[List[str]] = None) -> None:
        """Remove oldest checkpoints, keeping only keep_last regular ones."""
        if self.keep_last <= 0:
            return

        all_ckpts = self.list_checkpoints()
        exclude_tags = exclude_tags or []

        regular = [
            f for f in all_ckpts
            if not any(tag in os.path.basename(f) for tag in exclude_tags)
        ]

        for old in regular[: -self.keep_last]:
            try:
                os.remove(old)
                logger.debug("Rotated old checkpoint: %s", old)
            except OSError as e:
                logger.warning("Could not remove checkpoint %s: %s", old, e)

    def __repr__(self) -> str:
        return (
            f"CheckpointManager("
            f"save_dir={self.save_dir!r}, "
            f"max_retries={self.max_retries}, "
            f"keep_last={self.keep_last})"
        )


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import tempfile
    from unittest.mock import patch, MagicMock

    print("=" * 60)
    print("CheckpointManager self-tests")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Test 1: Save / load roundtrip
    # ------------------------------------------------------------------
    print("\n[Test 1] Save/load roundtrip")

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir, max_retries=5)

        enc_weight = torch.randn(16, 8)
        state = {
            "epoch": 5,
            "encoder": {"weight": enc_weight, "bias": torch.zeros(16)},
            "predictor": {"fc.weight": torch.randn(8, 16)},
            "target_encoder": {"weight": enc_weight.clone()},
            "opt": {"state": {}, "param_groups": [{"lr": 1e-3}]},
            "scaler": None,
        }

        path = manager.save(state, epoch=5)
        assert os.path.exists(path), "Checkpoint file should exist"
        assert path.endswith("checkpoint_0005.pth"), f"Unexpected filename: {path}"

        loaded = manager.load(path)
        assert loaded["epoch"] == 5
        assert torch.allclose(loaded["encoder"]["weight"], enc_weight)
        assert loaded["scaler"] is None
        assert loaded["opt"]["param_groups"][0]["lr"] == 1e-3
        print(f"  Saved to {os.path.basename(path)}, all values match  PASS")

    # ------------------------------------------------------------------
    # Test 2: Atomic write (tmp file cleaned up)
    # ------------------------------------------------------------------
    print("\n[Test 2] Atomic write (no leftover .tmp)")

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)
        manager.save({"epoch": 0}, epoch=0)
        tmp_files = glob.glob(os.path.join(tmpdir, "*.tmp"))
        assert len(tmp_files) == 0, f"Leftover .tmp files: {tmp_files}"
        print("  No .tmp files after save  PASS")

    # ------------------------------------------------------------------
    # Test 3: Retry logic — succeed after 2 failures
    # ------------------------------------------------------------------
    print("\n[Test 3] Retry on transient failure (succeed after 2 fails)")

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir, max_retries=5)

        # Save a real checkpoint
        state = {"epoch": 99}
        path = manager.save(state, epoch=99)

        call_count = [0]
        original_load = torch.load

        def flaky_load(p, *args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 3:  # Fail first 2 attempts
                raise IOError("Simulated transient NFS error")
            return original_load(p, *args, **kwargs)

        with patch("torch.load", side_effect=flaky_load):
            with patch("time.sleep"):  # Don't actually sleep in tests
                loaded = manager.load(path)

        assert loaded["epoch"] == 99, f"Expected epoch=99, got {loaded['epoch']}"
        assert call_count[0] == 3, f"Expected 3 calls, got {call_count[0]}"
        print("  Loaded after 2 failures (3 attempts total)  PASS")

    # ------------------------------------------------------------------
    # Test 4: Retry exhausted — raises RuntimeError
    # ------------------------------------------------------------------
    print("\n[Test 4] All retries exhausted -> RuntimeError")

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir, max_retries=3)
        fake_path = os.path.join(tmpdir, "nonexistent.pth")

        try:
            manager.load(fake_path)
            print("  FAIL: should have raised RuntimeError")
            sys.exit(1)
        except RuntimeError as e:
            assert "3 attempts" in str(e) or "Failed" in str(e)
            print(f"  RuntimeError raised after 3 attempts  PASS")

    # ------------------------------------------------------------------
    # Test 5: strip_prefix correctness
    # ------------------------------------------------------------------
    print("\n[Test 5] strip_prefix")

    cases = [
        ({"module.layer.weight": 1.0, "module.layer.bias": 2.0},
         "module.",
         {"layer.weight": 1.0, "layer.bias": 2.0}),
        ({"backbone.weight": 3.0, "other.key": 4.0},
         "backbone.",
         {"weight": 3.0, "other.key": 4.0}),
        ({},
         "module.",
         {}),
        ({"no_prefix.key": 5.0},
         "module.",
         {"no_prefix.key": 5.0}),
    ]

    for original, prefix, expected in cases:
        result = CheckpointManager.strip_prefix(original, prefix)
        assert result == expected, f"strip_prefix({list(original.keys())!r}, {prefix!r}) = {result!r}, expected {expected!r}"

    print("  All strip_prefix cases correct  PASS")

    # ------------------------------------------------------------------
    # Test 6: load_pretrained with strict=False
    # ------------------------------------------------------------------
    print("\n[Test 6] load_pretrained with strict=False (missing keys tolerated)")

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)

        # Build a model and capture its actual state_dict keys
        ref_model = nn.Linear(4, 4)
        ref_sd = ref_model.state_dict()  # keys: "weight", "bias"

        # Simulate: checkpoint has model weights + extra RoPE-incompatible key
        ckpt_state = {
            "encoder": {
                "weight": ref_sd["weight"].clone(),
                "bias": ref_sd["bias"].clone(),
                "pos_embed": torch.randn(1, 10, 4),  # RoPE models don't have this
            }
        }
        ckpt_path = manager.save(ckpt_state, epoch=0)

        # Fresh model (same architecture, different random weights)
        model = nn.Linear(4, 4)
        model_state_keys = set(model.state_dict().keys())
        assert "pos_embed" not in model_state_keys

        missing, unexpected = manager.load_pretrained(
            model, ckpt_path, key="encoder", strict=False
        )

        assert "pos_embed" in unexpected, f"pos_embed should be unexpected: {unexpected}"
        # weight and bias should load correctly
        assert torch.allclose(
            model.state_dict()["weight"],
            ref_sd["weight"]
        ), "Weight values should match after load_pretrained"
        print(f"  missing={missing}, unexpected={unexpected[:2]}...  PASS")

    # ------------------------------------------------------------------
    # Test 7: Checkpoint rotation
    # ------------------------------------------------------------------
    print("\n[Test 7] Checkpoint rotation (keep_last=2)")

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir, keep_last=2)

        for epoch in range(5):
            manager.save({"epoch": epoch}, epoch=epoch)

        remaining = manager.list_checkpoints()
        assert len(remaining) == 2, f"Expected 2 checkpoints, got {len(remaining)}"
        # Should keep epochs 3 and 4
        assert "0003" in remaining[0]
        assert "0004" in remaining[1]
        print(f"  Kept last 2: {[os.path.basename(p) for p in remaining]}  PASS")

    # ------------------------------------------------------------------
    # Test 8: latest_checkpoint returns newest file
    # ------------------------------------------------------------------
    print("\n[Test 8] latest_checkpoint()")

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)
        assert manager.latest_checkpoint() is None, "Should return None when empty"

        manager.save({"epoch": 3}, epoch=3)
        manager.save({"epoch": 7}, epoch=7)
        latest = manager.latest_checkpoint()
        assert latest is not None
        assert "0007" in latest
        print(f"  Latest: {os.path.basename(latest)}  PASS")

    # ------------------------------------------------------------------
    # Test 9: repr
    # ------------------------------------------------------------------
    print("\n[Test 9] __repr__")

    with tempfile.TemporaryDirectory() as tmpdir:
        m = CheckpointManager(tmpdir, max_retries=3, keep_last=5)
        r = repr(m)
        assert "max_retries=3" in r and "keep_last=5" in r
        print(f"  repr: {r}  PASS")

    print("\n" + "=" * 60)
    print("All CheckpointManager self-tests PASSED")
    print("=" * 60)
