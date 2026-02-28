"""
distributed_setup_template.py
==============================
DistributedSetup: auto-detects SLURM or local multi-GPU (torchrun) environments
and initializes NCCL process groups. Falls back gracefully to single-process mode.

Usage:
    setup = DistributedSetup()
    rank, local_rank, world_size = setup.init()
    # ... training ...
    setup.cleanup()
"""

from __future__ import annotations

import logging
import os
from typing import Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class DistributedSetup:
    """
    Manages distributed training initialization and teardown.

    Supports three modes (in detection order):
      1. SLURM -- detected via SLURM_NTASKS / SLURM_PROCID / SLURM_LOCALID
      2. torchrun (local multi-GPU) -- detected via RANK / LOCAL_RANK / WORLD_SIZE
      3. Single-process fallback -- when no distributed env vars are found

    Attributes:
        rank (int): Global process rank (0-indexed).
        local_rank (int): Local rank within the current node.
        world_size (int): Total number of processes.
        _initialized (bool): Whether dist.init_process_group has been called.
    """

    def __init__(self) -> None:
        self.rank: int = 0
        self.local_rank: int = 0
        self.world_size: int = 1
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def init(self) -> Tuple[int, int, int]:
        """
        Initialize distributed training.

        Returns:
            Tuple of (rank, local_rank, world_size).

        Detection order:
            1. SLURM (SLURM_NTASKS + SLURM_PROCID + SLURM_LOCALID)
            2. torchrun (RANK + LOCAL_RANK + WORLD_SIZE)
            3. Single-process fallback
        """
        if self.is_slurm():
            logger.info("SLURM environment detected — using file-based rendezvous")
            self.rank, self.local_rank, self.world_size = self._init_slurm()
        elif self._is_torchrun():
            logger.info("torchrun environment detected — using env:// rendezvous")
            self.rank, self.local_rank, self.world_size = self._init_torchrun()
        else:
            logger.info("No distributed environment detected — single-process mode")
            self.rank, self.local_rank, self.world_size = 0, 0, 1
            # Set CUDA device to 0 if available
            if torch.cuda.is_available():
                torch.cuda.set_device(0)

        return self.rank, self.local_rank, self.world_size

    def is_slurm(self) -> bool:
        """
        Return True if all required SLURM environment variables are set.

        Required: SLURM_NTASKS, SLURM_PROCID, SLURM_LOCALID
        """
        return all(
            var in os.environ
            for var in ("SLURM_NTASKS", "SLURM_PROCID", "SLURM_LOCALID")
        )

    def cleanup(self) -> None:
        """
        Destroy the process group and release NCCL resources.

        Safe to call multiple times (idempotent).
        """
        if dist.is_initialized():
            dist.destroy_process_group()
            self._initialized = False
            logger.debug("Distributed process group destroyed")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_torchrun(self) -> bool:
        """Return True if torchrun env vars are present."""
        return all(
            var in os.environ
            for var in ("RANK", "LOCAL_RANK", "WORLD_SIZE")
        )

    def _init_slurm(self) -> Tuple[int, int, int]:
        """
        Initialize from SLURM environment variables.

        Uses SLURM_JOB_TMPDIR for rendezvous when available
        (local NVMe avoids NFS contention on large clusters).
        """
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        world_size = int(os.environ["SLURM_NTASKS"])

        # Build init URL: prefer local tmpdir to avoid NFS slowness
        job_tmpdir = os.environ.get("SLURM_JOB_TMPDIR", "/tmp")
        job_id = os.environ.get("SLURM_JOB_ID", "0")
        init_file = os.path.join(job_tmpdir, f"dist_init_{job_id}")
        init_url = f"file://{init_file}"

        logger.debug(
            "SLURM init: rank=%d, local_rank=%d, world_size=%d, url=%s",
            rank, local_rank, world_size, init_url,
        )

        # Assign CUDA device before init_process_group
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method=init_url,
            rank=rank,
            world_size=world_size,
        )
        self._initialized = True
        return rank, local_rank, world_size

    def _init_torchrun(self) -> Tuple[int, int, int]:
        """
        Initialize from torchrun (env://) environment variables.

        torchrun sets RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT.
        """
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        logger.debug(
            "torchrun init: rank=%d, local_rank=%d, world_size=%d",
            rank, local_rank, world_size,
        )

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="env://",
        )
        self._initialized = True
        return rank, local_rank, world_size

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def is_main(self) -> bool:
        """True if this process is the primary process (rank 0)."""
        return self.rank == 0

    @property
    def device(self) -> torch.device:
        """Return the torch device for this process."""
        if torch.cuda.is_available():
            return torch.device(f"cuda:{self.local_rank}")
        return torch.device("cpu")

    def barrier(self) -> None:
        """Synchronize all processes. No-op in single-process mode."""
        if dist.is_initialized():
            dist.barrier()

    def __repr__(self) -> str:
        return (
            f"DistributedSetup("
            f"rank={self.rank}, "
            f"local_rank={self.local_rank}, "
            f"world_size={self.world_size}, "
            f"initialized={self._initialized})"
        )


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("DistributedSetup self-tests")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Test 1: Single-process fallback (no env vars set)
    # ------------------------------------------------------------------
    print("\n[Test 1] Single-process fallback")

    # Temporarily clear all distributed env vars
    slurm_vars = ["SLURM_NTASKS", "SLURM_PROCID", "SLURM_LOCALID",
                  "SLURM_JOB_ID", "SLURM_JOB_TMPDIR"]
    torch_vars = ["RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]
    saved_env = {}
    for var in slurm_vars + torch_vars:
        if var in os.environ:
            saved_env[var] = os.environ.pop(var)

    setup = DistributedSetup()
    rank, local_rank, world_size = setup.init()

    assert rank == 0, f"Expected rank=0, got {rank}"
    assert local_rank == 0, f"Expected local_rank=0, got {local_rank}"
    assert world_size == 1, f"Expected world_size=1, got {world_size}"
    assert not dist.is_initialized(), "dist should NOT be initialized in single-process mode"
    print(f"  rank={rank}, local_rank={local_rank}, world_size={world_size}  PASS")

    # ------------------------------------------------------------------
    # Test 2: is_slurm() with partial vars returns False
    # ------------------------------------------------------------------
    print("\n[Test 2] is_slurm() with partial vars")

    os.environ["SLURM_PROCID"] = "0"  # Only one var set
    assert not setup.is_slurm(), "is_slurm() should return False with only SLURM_PROCID"
    del os.environ["SLURM_PROCID"]
    print("  is_slurm() with only SLURM_PROCID -> False  PASS")

    # ------------------------------------------------------------------
    # Test 3: Cleanup is idempotent (no dist initialized)
    # ------------------------------------------------------------------
    print("\n[Test 3] Cleanup idempotency")

    setup2 = DistributedSetup()
    try:
        setup2.cleanup()  # First call — dist not initialized
        setup2.cleanup()  # Second call — must not raise
        print("  Double cleanup (no dist) -> no exception  PASS")
    except Exception as e:
        print(f"  FAIL: cleanup raised {e}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Test 4: device property
    # ------------------------------------------------------------------
    print("\n[Test 4] Device property")
    setup3 = DistributedSetup()
    setup3.init()
    device = setup3.device
    expected_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    assert device == expected_device, f"Expected {expected_device}, got {device}"
    print(f"  device={device}  PASS")

    # ------------------------------------------------------------------
    # Test 5: is_main property
    # ------------------------------------------------------------------
    print("\n[Test 5] is_main property")
    assert setup3.is_main, "rank=0 should be main process"
    print("  is_main=True for rank=0  PASS")

    # ------------------------------------------------------------------
    # Test 6: repr
    # ------------------------------------------------------------------
    print("\n[Test 6] __repr__")
    r = repr(setup3)
    assert "rank=0" in r and "world_size=1" in r
    print(f"  repr: {r}  PASS")

    # Restore saved env vars
    os.environ.update(saved_env)

    print("\n" + "=" * 60)
    print("All DistributedSetup self-tests PASSED")
    print("=" * 60)
