"""
DistributedLauncher — Unified launcher for torchrun, SLURM, and manual multi-node.

Detects the launch environment automatically, initializes process groups,
and dispatches the user-provided training function across all ranks.

Key classes:
    DistributedLauncher   — Main launcher with detect_environment() and launch()
    LaunchConfig          — Per-environment launch parameters
    EnvironmentDetector   — Introspects environment variables for launch mode

Self-tests in __main__ validate all functionality without actual multi-GPU hardware.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports (graceful fallback)
# ---------------------------------------------------------------------------

_DIST_AVAILABLE = False
_MP_AVAILABLE = False

try:
    import torch.distributed as dist
    _DIST_AVAILABLE = True
except ImportError:
    dist = None

try:
    import torch.multiprocessing as mp
    _MP_AVAILABLE = True
except ImportError:
    mp = None


# ===========================================================================
# SECTION 1: Enums and dataclasses
# ===========================================================================

class LaunchMode(str, Enum):
    """Supported launch modes."""
    TORCHRUN = "torchrun"
    SLURM = "slurm"
    MANUAL = "manual"
    SINGLE = "single"


@dataclass
class LaunchConfig:
    """Configuration for distributed launch.

    Attributes:
        mode: The detected or requested launch mode.
        backend: Communication backend ("nccl" or "gloo").
        num_nodes: Number of nodes in the cluster.
        node_rank: This node's rank (0..num_nodes-1).
        nproc_per_node: Processes (GPUs) per node.
        master_addr: Address of the master node.
        master_port: Port of the master node.
        world_size: Total number of processes (auto-computed).
        log_level: Logging verbosity for distributed.
        timeout_seconds: Timeout for collective operations.
        elastic: Whether to enable elastic (fault-tolerant) launch.
        max_restarts: Maximum restarts for elastic launch.
    """
    mode: str = "single"
    backend: str = "nccl"
    num_nodes: int = 1
    node_rank: int = 0
    nproc_per_node: int = 1
    master_addr: str = "localhost"
    master_port: str = "29500"
    world_size: int = 1
    log_level: str = "WARNING"
    timeout_seconds: int = 1800
    elastic: bool = False
    max_restarts: int = 3

    def __post_init__(self):
        if self.world_size == 1 and self.num_nodes > 1:
            self.world_size = self.num_nodes * self.nproc_per_node

    @classmethod
    def for_single_gpu(cls) -> "LaunchConfig":
        """Single-GPU, no distribution."""
        return cls(mode="single", num_nodes=1, nproc_per_node=1, world_size=1)

    @classmethod
    def for_single_node_multi_gpu(
        cls, nproc: int = 0, backend: str = "nccl"
    ) -> "LaunchConfig":
        """Single node, multiple GPUs."""
        if nproc <= 0:
            nproc = max(torch.cuda.device_count(), 1)
        return cls(
            mode="manual",
            backend=backend,
            num_nodes=1,
            nproc_per_node=nproc,
            world_size=nproc,
        )

    @classmethod
    def for_multi_node(
        cls,
        num_nodes: int,
        node_rank: int,
        nproc_per_node: int,
        master_addr: str,
        master_port: str = "29500",
        backend: str = "nccl",
    ) -> "LaunchConfig":
        """Multi-node cluster launch."""
        return cls(
            mode="manual",
            backend=backend,
            num_nodes=num_nodes,
            node_rank=node_rank,
            nproc_per_node=nproc_per_node,
            master_addr=master_addr,
            master_port=master_port,
            world_size=num_nodes * nproc_per_node,
        )

    @classmethod
    def for_slurm(cls, backend: str = "nccl") -> "LaunchConfig":
        """SLURM-detected launch."""
        return cls(mode="slurm", backend=backend)

    def validate(self) -> List[str]:
        """Validate configuration. Returns list of error messages."""
        errors = []
        if self.backend not in ("nccl", "gloo"):
            errors.append(f"Invalid backend: {self.backend}. Must be 'nccl' or 'gloo'.")
        if self.num_nodes < 1:
            errors.append(f"num_nodes must be >= 1, got {self.num_nodes}")
        if self.nproc_per_node < 1:
            errors.append(f"nproc_per_node must be >= 1, got {self.nproc_per_node}")
        if self.node_rank < 0 or self.node_rank >= self.num_nodes:
            errors.append(
                f"node_rank ({self.node_rank}) must be in [0, {self.num_nodes})"
            )
        if self.timeout_seconds < 1:
            errors.append(f"timeout_seconds must be >= 1, got {self.timeout_seconds}")
        if self.max_restarts < 0:
            errors.append(f"max_restarts must be >= 0, got {self.max_restarts}")
        try:
            port = int(self.master_port)
            if port < 1 or port > 65535:
                errors.append(f"master_port must be 1-65535, got {port}")
        except ValueError:
            errors.append(f"master_port must be numeric, got {self.master_port}")
        return errors


# ===========================================================================
# SECTION 2: EnvironmentDetector
# ===========================================================================

class EnvironmentDetector:
    """Detect the distributed launch environment from environment variables.

    Detection priority:
    1. torchrun — RANK, WORLD_SIZE, LOCAL_RANK all set
    2. SLURM — SLURM_PROCID and SLURM_NTASKS set
    3. manual — User-provided configuration
    4. single — No distributed environment detected
    """

    @staticmethod
    def detect() -> str:
        """Detect the current launch environment.

        Returns:
            One of "torchrun", "slurm", "manual", "single".
        """
        if EnvironmentDetector._is_torchrun():
            return "torchrun"
        if EnvironmentDetector._is_slurm():
            return "slurm"
        if EnvironmentDetector._is_manual():
            return "manual"
        return "single"

    @staticmethod
    def _is_torchrun() -> bool:
        """torchrun sets RANK, WORLD_SIZE, LOCAL_RANK."""
        return all(
            var in os.environ
            for var in ("RANK", "WORLD_SIZE", "LOCAL_RANK")
        )

    @staticmethod
    def _is_slurm() -> bool:
        """SLURM sets SLURM_PROCID, SLURM_NTASKS."""
        return all(
            var in os.environ
            for var in ("SLURM_PROCID", "SLURM_NTASKS")
        )

    @staticmethod
    def _is_manual() -> bool:
        """Check if MASTER_ADDR and MASTER_PORT are set manually."""
        return "MASTER_ADDR" in os.environ and "MASTER_PORT" in os.environ

    @staticmethod
    def get_torchrun_info() -> Dict[str, int]:
        """Extract torchrun environment info."""
        return {
            "rank": int(os.environ.get("RANK", "0")),
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        }

    @staticmethod
    def get_slurm_info() -> Dict[str, Any]:
        """Extract SLURM environment info."""
        rank = int(os.environ.get("SLURM_PROCID", "0"))
        world_size = int(os.environ.get("SLURM_NTASKS", "1"))
        local_rank = rank % max(torch.cuda.device_count(), 1)
        node_list = os.environ.get("SLURM_NODELIST", "localhost")
        job_id = os.environ.get("SLURM_JOB_ID", "0")
        return {
            "rank": rank,
            "world_size": world_size,
            "local_rank": local_rank,
            "node_list": node_list,
            "job_id": job_id,
        }

    @staticmethod
    def find_free_port() -> int:
        """Find a free port on localhost."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]


# ===========================================================================
# SECTION 3: DistributedLauncher
# ===========================================================================

class DistributedLauncher:
    """Unified launcher for distributed training across environments.

    Detects the launch environment (torchrun, SLURM, manual) and
    dispatches the user-provided training function with correct
    rank/world_size initialization.

    Args:
        config: LaunchConfig with distributed settings.
    """

    def __init__(self, config: Optional[LaunchConfig] = None):
        self.config = config or LaunchConfig()
        self._detected_mode: Optional[str] = None

    def detect_environment(self) -> str:
        """Detect the distributed environment.

        Returns:
            One of "torchrun", "slurm", "manual", "single".
        """
        self._detected_mode = EnvironmentDetector.detect()
        logger.info(f"Detected environment: {self._detected_mode}")
        return self._detected_mode

    def launch(
        self,
        train_fn: Callable,
        args: Any = None,
    ) -> None:
        """Launch the training function in the detected/configured environment.

        For torchrun/SLURM environments, the process group is already set up
        by the external launcher; this method initializes the group and calls
        train_fn(rank, world_size, args).

        For manual mode, spawns processes using torch.multiprocessing.spawn.

        For single mode, calls train_fn(0, 1, args) directly.

        Args:
            train_fn: Callable(rank, world_size, args) to execute on each rank.
            args: Additional arguments passed to train_fn.
        """
        mode = self._detected_mode or self.detect_environment()

        # Override with config mode if user explicitly set it
        if self.config.mode != "single" and mode == "single":
            mode = self.config.mode

        logger.info(f"Launching in mode: {mode}")

        if mode == "torchrun":
            self._launch_torchrun(train_fn, args)
        elif mode == "slurm":
            self._launch_slurm(train_fn, args)
        elif mode == "manual":
            self._launch_manual(train_fn, args)
        else:
            self._launch_single(train_fn, args)

    def _launch_torchrun(self, train_fn: Callable, args: Any) -> None:
        """Launch under torchrun — process group managed by torchrun."""
        info = EnvironmentDetector.get_torchrun_info()
        rank = info["rank"]
        world_size = info["world_size"]
        local_rank = info["local_rank"]

        self._init_process_group(rank, world_size, local_rank)
        try:
            train_fn(rank, world_size, args)
        finally:
            self._cleanup()

    def _launch_slurm(self, train_fn: Callable, args: Any) -> None:
        """Launch under SLURM — extract rank/world from SLURM vars."""
        info = EnvironmentDetector.get_slurm_info()
        rank = info["rank"]
        world_size = info["world_size"]
        local_rank = info["local_rank"]

        # Set master address from SLURM_NODELIST if not already set
        if "MASTER_ADDR" not in os.environ:
            node_list = info["node_list"]
            os.environ["MASTER_ADDR"] = self._resolve_slurm_master(node_list)
        os.environ.setdefault("MASTER_PORT", self.config.master_port)

        self._init_process_group(rank, world_size, local_rank)
        try:
            train_fn(rank, world_size, args)
        finally:
            self._cleanup()

    def _launch_manual(self, train_fn: Callable, args: Any) -> None:
        """Launch manually — spawn processes with torch.multiprocessing."""
        if not _MP_AVAILABLE:
            raise RuntimeError(
                "torch.multiprocessing not available for manual launch."
            )

        nproc = self.config.nproc_per_node
        if nproc <= 1:
            # Single process — just call directly
            self._launch_single(train_fn, args)
            return

        os.environ["MASTER_ADDR"] = self.config.master_addr
        os.environ["MASTER_PORT"] = self.config.master_port

        mp.spawn(
            self._worker_fn,
            args=(nproc, self.config, train_fn, args),
            nprocs=nproc,
            join=True,
        )

    @staticmethod
    def _worker_fn(
        rank: int,
        world_size: int,
        config: LaunchConfig,
        train_fn: Callable,
        args: Any,
    ) -> None:
        """Worker function executed by each spawned process.

        Args:
            rank: Local rank (0..world_size-1).
            world_size: Total number of processes.
            config: Launch configuration.
            train_fn: Training function to call.
            args: Additional arguments.
        """
        global_rank = config.node_rank * world_size + rank

        os.environ["MASTER_ADDR"] = config.master_addr
        os.environ["MASTER_PORT"] = config.master_port

        if _DIST_AVAILABLE:
            dist.init_process_group(
                backend=config.backend,
                rank=global_rank,
                world_size=config.num_nodes * world_size,
            )

        if torch.cuda.is_available() and config.backend == "nccl":
            torch.cuda.set_device(rank)

        try:
            total_world = config.num_nodes * world_size
            train_fn(global_rank, total_world, args)
        finally:
            if _DIST_AVAILABLE and dist.is_initialized():
                dist.destroy_process_group()

    def _launch_single(self, train_fn: Callable, args: Any) -> None:
        """Launch in single-process mode (no distribution)."""
        logger.info("Running in single-process mode.")
        train_fn(0, 1, args)

    def _init_process_group(
        self, rank: int, world_size: int, local_rank: int
    ) -> None:
        """Initialize the distributed process group."""
        if not _DIST_AVAILABLE:
            logger.warning("torch.distributed not available.")
            return

        if dist.is_initialized():
            logger.info("Process group already initialized.")
            return

        os.environ.setdefault("MASTER_ADDR", self.config.master_addr)
        os.environ.setdefault("MASTER_PORT", self.config.master_port)

        dist.init_process_group(
            backend=self.config.backend,
            rank=rank,
            world_size=world_size,
        )

        if torch.cuda.is_available() and self.config.backend == "nccl":
            torch.cuda.set_device(local_rank)

        logger.info(
            f"Process group initialized: rank={rank}, "
            f"world_size={world_size}, local_rank={local_rank}"
        )

    def _cleanup(self) -> None:
        """Destroy process group if initialized."""
        if _DIST_AVAILABLE and dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Process group destroyed.")

    @staticmethod
    def _resolve_slurm_master(node_list: str) -> str:
        """Resolve the master node address from SLURM_NODELIST.

        Simple parsing: take the first hostname.
        """
        # Handle formats like "node[001-004]" or "node001,node002"
        if "[" in node_list:
            prefix = node_list.split("[")[0]
            range_part = node_list.split("[")[1].split("]")[0]
            first = range_part.split(",")[0].split("-")[0]
            return f"{prefix}{first}"
        return node_list.split(",")[0]

    def get_launch_summary(self) -> Dict[str, Any]:
        """Return a summary of the launch configuration."""
        mode = self._detected_mode or self.detect_environment()
        return {
            "mode": mode,
            "backend": self.config.backend,
            "num_nodes": self.config.num_nodes,
            "nproc_per_node": self.config.nproc_per_node,
            "world_size": self.config.world_size,
            "master_addr": self.config.master_addr,
            "master_port": self.config.master_port,
            "elastic": self.config.elastic,
            "timeout_seconds": self.config.timeout_seconds,
        }


# ===========================================================================
# SECTION 4: Utility functions
# ===========================================================================

def build_torchrun_command(
    script_path: str,
    nproc_per_node: int = 1,
    nnodes: int = 1,
    node_rank: int = 0,
    master_addr: str = "localhost",
    master_port: str = "29500",
    extra_args: Optional[List[str]] = None,
) -> List[str]:
    """Build a torchrun command line.

    Args:
        script_path: Path to the training script.
        nproc_per_node: Number of processes per node.
        nnodes: Number of nodes.
        node_rank: This node's rank.
        master_addr: Master address.
        master_port: Master port.
        extra_args: Additional command-line arguments for the script.

    Returns:
        List of command-line tokens.
    """
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={nproc_per_node}",
        f"--nnodes={nnodes}",
        f"--node_rank={node_rank}",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
        script_path,
    ]
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def seed_everything(seed: int, rank: int = 0) -> None:
    """Set all random seeds, incorporating rank for uniqueness.

    Args:
        seed: Base seed.
        rank: Process rank (added to seed).
    """
    effective_seed = seed + rank
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)
    import random
    random.seed(effective_seed)
    try:
        import numpy as np
        np.random.seed(effective_seed)
    except ImportError:
        pass


def get_rank_info() -> Dict[str, int]:
    """Get current process rank info from environment or torch.distributed."""
    if _DIST_AVAILABLE and dist.is_initialized():
        return {
            "rank": dist.get_rank(),
            "world_size": dist.get_world_size(),
            "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        }
    return {
        "rank": int(os.environ.get("RANK", "0")),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
    }


# ===========================================================================
# SECTION 5: Self-tests
# ===========================================================================

if __name__ == "__main__":
    import traceback

    passed = 0
    failed = 0
    total = 0

    def run_test(name: str, fn: Callable):
        global passed, failed, total
        total += 1
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            traceback.print_exc()
            failed += 1

    print("=" * 70)
    print("DistributedLauncher Self-Tests (single-process simulation)")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # LaunchConfig tests
    # -----------------------------------------------------------------------

    def test_launch_config_defaults():
        cfg = LaunchConfig()
        assert cfg.mode == "single"
        assert cfg.backend == "nccl"
        assert cfg.num_nodes == 1
        assert cfg.nproc_per_node == 1
        assert cfg.world_size == 1
        assert cfg.master_addr == "localhost"
        assert cfg.master_port == "29500"
        assert cfg.timeout_seconds == 1800
        assert cfg.elastic is False

    run_test("LaunchConfig defaults", test_launch_config_defaults)

    def test_launch_config_single_gpu():
        cfg = LaunchConfig.for_single_gpu()
        assert cfg.mode == "single"
        assert cfg.world_size == 1

    run_test("LaunchConfig.for_single_gpu", test_launch_config_single_gpu)

    def test_launch_config_single_node_multi():
        cfg = LaunchConfig.for_single_node_multi_gpu(nproc=4, backend="gloo")
        assert cfg.mode == "manual"
        assert cfg.nproc_per_node == 4
        assert cfg.world_size == 4
        assert cfg.backend == "gloo"

    run_test("LaunchConfig.for_single_node_multi_gpu", test_launch_config_single_node_multi)

    def test_launch_config_multi_node():
        cfg = LaunchConfig.for_multi_node(
            num_nodes=2, node_rank=1, nproc_per_node=8,
            master_addr="10.0.0.1", master_port="29501",
        )
        assert cfg.num_nodes == 2
        assert cfg.node_rank == 1
        assert cfg.nproc_per_node == 8
        assert cfg.world_size == 16
        assert cfg.master_addr == "10.0.0.1"

    run_test("LaunchConfig.for_multi_node", test_launch_config_multi_node)

    def test_launch_config_slurm():
        cfg = LaunchConfig.for_slurm(backend="gloo")
        assert cfg.mode == "slurm"
        assert cfg.backend == "gloo"

    run_test("LaunchConfig.for_slurm", test_launch_config_slurm)

    def test_launch_config_validate_valid():
        cfg = LaunchConfig.for_single_gpu()
        errors = cfg.validate()
        assert len(errors) == 0, f"Unexpected errors: {errors}"

    run_test("LaunchConfig validate (valid)", test_launch_config_validate_valid)

    def test_launch_config_validate_invalid_backend():
        cfg = LaunchConfig(backend="mpi")
        errors = cfg.validate()
        assert any("backend" in e for e in errors)

    run_test("LaunchConfig validate (invalid backend)", test_launch_config_validate_invalid_backend)

    def test_launch_config_validate_invalid_nodes():
        cfg = LaunchConfig(num_nodes=0)
        errors = cfg.validate()
        assert any("num_nodes" in e for e in errors)

    run_test("LaunchConfig validate (invalid num_nodes)", test_launch_config_validate_invalid_nodes)

    def test_launch_config_validate_bad_port():
        cfg = LaunchConfig(master_port="abc")
        errors = cfg.validate()
        assert any("master_port" in e for e in errors)

    run_test("LaunchConfig validate (bad port)", test_launch_config_validate_bad_port)

    def test_launch_config_validate_node_rank_oob():
        cfg = LaunchConfig(num_nodes=2, node_rank=5)
        errors = cfg.validate()
        assert any("node_rank" in e for e in errors)

    run_test("LaunchConfig validate (node_rank out of bounds)", test_launch_config_validate_node_rank_oob)

    def test_launch_config_auto_world_size():
        cfg = LaunchConfig(num_nodes=4, nproc_per_node=8)
        assert cfg.world_size == 32

    run_test("LaunchConfig auto world_size computation", test_launch_config_auto_world_size)

    # -----------------------------------------------------------------------
    # EnvironmentDetector tests
    # -----------------------------------------------------------------------

    def test_detect_single_no_env():
        # Save and clear env vars
        saved = {}
        for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "SLURM_PROCID",
                   "SLURM_NTASKS", "MASTER_ADDR", "MASTER_PORT"):
            saved[k] = os.environ.pop(k, None)
        try:
            result = EnvironmentDetector.detect()
            assert result == "single", f"Expected 'single', got '{result}'"
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    run_test("EnvironmentDetector detects single", test_detect_single_no_env)

    def test_detect_torchrun():
        saved = {}
        for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
            saved[k] = os.environ.get(k)
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "4"
        os.environ["LOCAL_RANK"] = "0"
        try:
            result = EnvironmentDetector.detect()
            assert result == "torchrun", f"Expected 'torchrun', got '{result}'"
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v
                else:
                    os.environ.pop(k, None)

    run_test("EnvironmentDetector detects torchrun", test_detect_torchrun)

    def test_detect_slurm():
        saved = {}
        for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "SLURM_PROCID", "SLURM_NTASKS"):
            saved[k] = os.environ.get(k)
        # Remove torchrun vars so SLURM is detected
        for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
            os.environ.pop(k, None)
        os.environ["SLURM_PROCID"] = "0"
        os.environ["SLURM_NTASKS"] = "8"
        try:
            result = EnvironmentDetector.detect()
            assert result == "slurm", f"Expected 'slurm', got '{result}'"
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v
                else:
                    os.environ.pop(k, None)

    run_test("EnvironmentDetector detects SLURM", test_detect_slurm)

    def test_get_torchrun_info():
        saved = {}
        for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
            saved[k] = os.environ.get(k)
        os.environ["RANK"] = "3"
        os.environ["WORLD_SIZE"] = "8"
        os.environ["LOCAL_RANK"] = "3"
        try:
            info = EnvironmentDetector.get_torchrun_info()
            assert info["rank"] == 3
            assert info["world_size"] == 8
            assert info["local_rank"] == 3
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v
                else:
                    os.environ.pop(k, None)

    run_test("EnvironmentDetector.get_torchrun_info", test_get_torchrun_info)

    def test_get_slurm_info():
        saved = {}
        for k in ("SLURM_PROCID", "SLURM_NTASKS", "SLURM_NODELIST", "SLURM_JOB_ID"):
            saved[k] = os.environ.get(k)
        os.environ["SLURM_PROCID"] = "5"
        os.environ["SLURM_NTASKS"] = "16"
        os.environ["SLURM_NODELIST"] = "node001"
        os.environ["SLURM_JOB_ID"] = "12345"
        try:
            info = EnvironmentDetector.get_slurm_info()
            assert info["rank"] == 5
            assert info["world_size"] == 16
            assert info["node_list"] == "node001"
            assert info["job_id"] == "12345"
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v
                else:
                    os.environ.pop(k, None)

    run_test("EnvironmentDetector.get_slurm_info", test_get_slurm_info)

    def test_find_free_port():
        port = EnvironmentDetector.find_free_port()
        assert isinstance(port, int)
        assert 1 <= port <= 65535

    run_test("EnvironmentDetector.find_free_port", test_find_free_port)

    # -----------------------------------------------------------------------
    # DistributedLauncher construction tests
    # -----------------------------------------------------------------------

    def test_launcher_construction_default():
        launcher = DistributedLauncher()
        assert launcher.config.mode == "single"
        assert launcher._detected_mode is None

    run_test("DistributedLauncher construction default", test_launcher_construction_default)

    def test_launcher_construction_with_config():
        cfg = LaunchConfig.for_single_node_multi_gpu(nproc=4, backend="gloo")
        launcher = DistributedLauncher(config=cfg)
        assert launcher.config.nproc_per_node == 4

    run_test("DistributedLauncher with config", test_launcher_construction_with_config)

    def test_launcher_detect_environment():
        # Without env vars should detect "single"
        saved = {}
        for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "SLURM_PROCID",
                   "SLURM_NTASKS", "MASTER_ADDR", "MASTER_PORT"):
            saved[k] = os.environ.pop(k, None)
        try:
            launcher = DistributedLauncher()
            mode = launcher.detect_environment()
            assert mode == "single"
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    run_test("DistributedLauncher.detect_environment single", test_launcher_detect_environment)

    # -----------------------------------------------------------------------
    # Launch single-process tests
    # -----------------------------------------------------------------------

    def test_launch_single_process():
        results = {}

        def train_fn(rank, world_size, args):
            results["rank"] = rank
            results["world_size"] = world_size
            results["args"] = args

        launcher = DistributedLauncher(LaunchConfig.for_single_gpu())
        launcher.launch(train_fn, args={"lr": 0.001})
        assert results["rank"] == 0
        assert results["world_size"] == 1
        assert results["args"]["lr"] == 0.001

    run_test("launch single-process calls train_fn", test_launch_single_process)

    def test_launch_single_no_args():
        results = {}

        def train_fn(rank, world_size, args):
            results["rank"] = rank
            results["world_size"] = world_size
            results["args"] = args

        launcher = DistributedLauncher(LaunchConfig.for_single_gpu())
        launcher.launch(train_fn)
        assert results["rank"] == 0
        assert results["args"] is None

    run_test("launch single-process no args", test_launch_single_no_args)

    def test_launch_single_model_training():
        """Simulate a real training call in single mode."""
        model = nn.Linear(10, 2)
        results = {}

        def train_fn(rank, world_size, args):
            x = torch.randn(4, 10)
            out = model(x)
            loss = out.sum()
            loss.backward()
            results["loss"] = loss.item()
            results["has_grad"] = model.weight.grad is not None

        launcher = DistributedLauncher(LaunchConfig.for_single_gpu())
        launcher.launch(train_fn)
        assert results["has_grad"] is True
        assert "loss" in results

    run_test("launch single-process with model training", test_launch_single_model_training)

    # -----------------------------------------------------------------------
    # SLURM master resolution tests
    # -----------------------------------------------------------------------

    def test_resolve_slurm_master_simple():
        result = DistributedLauncher._resolve_slurm_master("node001")
        assert result == "node001"

    run_test("resolve SLURM master: simple hostname", test_resolve_slurm_master_simple)

    def test_resolve_slurm_master_range():
        result = DistributedLauncher._resolve_slurm_master("node[001-004]")
        assert result == "node001"

    run_test("resolve SLURM master: range format", test_resolve_slurm_master_range)

    def test_resolve_slurm_master_comma():
        result = DistributedLauncher._resolve_slurm_master("nodeA,nodeB,nodeC")
        assert result == "nodeA"

    run_test("resolve SLURM master: comma-separated", test_resolve_slurm_master_comma)

    # -----------------------------------------------------------------------
    # build_torchrun_command tests
    # -----------------------------------------------------------------------

    def test_build_torchrun_command_basic():
        cmd = build_torchrun_command("train.py", nproc_per_node=8)
        assert "train.py" in cmd
        assert "--nproc_per_node=8" in cmd
        assert "--nnodes=1" in cmd

    run_test("build_torchrun_command basic", test_build_torchrun_command_basic)

    def test_build_torchrun_command_multi_node():
        cmd = build_torchrun_command(
            "train.py", nproc_per_node=8, nnodes=2,
            node_rank=1, master_addr="10.0.0.1", master_port="29501",
            extra_args=["--lr", "0.001"],
        )
        assert "--nnodes=2" in cmd
        assert "--node_rank=1" in cmd
        assert "--master_addr=10.0.0.1" in cmd
        assert "--master_port=29501" in cmd
        assert "--lr" in cmd
        assert "0.001" in cmd

    run_test("build_torchrun_command multi-node", test_build_torchrun_command_multi_node)

    # -----------------------------------------------------------------------
    # seed_everything tests
    # -----------------------------------------------------------------------

    def test_seed_everything():
        seed_everything(42, rank=0)
        a = torch.randn(3)
        seed_everything(42, rank=0)
        b = torch.randn(3)
        assert torch.equal(a, b)

    run_test("seed_everything deterministic", test_seed_everything)

    def test_seed_different_ranks():
        seed_everything(42, rank=0)
        a = torch.randn(3)
        seed_everything(42, rank=1)
        b = torch.randn(3)
        assert not torch.equal(a, b)

    run_test("seed_everything different ranks", test_seed_different_ranks)

    # -----------------------------------------------------------------------
    # get_rank_info tests
    # -----------------------------------------------------------------------

    def test_get_rank_info_default():
        saved = {}
        for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
            saved[k] = os.environ.pop(k, None)
        try:
            info = get_rank_info()
            assert info["rank"] == 0
            assert info["world_size"] == 1
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    run_test("get_rank_info default", test_get_rank_info_default)

    # -----------------------------------------------------------------------
    # Launch summary tests
    # -----------------------------------------------------------------------

    def test_get_launch_summary():
        cfg = LaunchConfig.for_multi_node(
            num_nodes=2, node_rank=0, nproc_per_node=8,
            master_addr="10.0.0.1",
        )
        launcher = DistributedLauncher(config=cfg)
        summary = launcher.get_launch_summary()
        assert summary["num_nodes"] == 2
        assert summary["nproc_per_node"] == 8
        assert summary["world_size"] == 16
        assert summary["master_addr"] == "10.0.0.1"

    run_test("get_launch_summary", test_get_launch_summary)

    # -----------------------------------------------------------------------
    # LaunchMode enum tests
    # -----------------------------------------------------------------------

    def test_launch_mode_values():
        assert LaunchMode.TORCHRUN.value == "torchrun"
        assert LaunchMode.SLURM.value == "slurm"
        assert LaunchMode.MANUAL.value == "manual"
        assert LaunchMode.SINGLE.value == "single"

    run_test("LaunchMode enum values", test_launch_mode_values)

    # -----------------------------------------------------------------------
    # Edge cases
    # -----------------------------------------------------------------------

    def test_launch_exception_propagates():
        def failing_fn(rank, world_size, args):
            raise ValueError("intentional error")

        launcher = DistributedLauncher(LaunchConfig.for_single_gpu())
        try:
            launcher.launch(failing_fn)
            assert False, "Should have raised"
        except ValueError as e:
            assert "intentional" in str(e)

    run_test("launch propagates exceptions", test_launch_exception_propagates)

    def test_multiple_launches():
        counter = {"n": 0}

        def counting_fn(rank, world_size, args):
            counter["n"] += 1

        launcher = DistributedLauncher(LaunchConfig.for_single_gpu())
        for _ in range(3):
            launcher.launch(counting_fn)
        assert counter["n"] == 3

    run_test("multiple launches", test_multiple_launches)

    def test_launch_config_validate_many_errors():
        cfg = LaunchConfig(
            backend="mpi", num_nodes=0, nproc_per_node=-1,
            node_rank=-1, master_port="abc", timeout_seconds=0,
            max_restarts=-5,
        )
        errors = cfg.validate()
        assert len(errors) >= 5, f"Expected many errors, got {len(errors)}: {errors}"

    run_test("LaunchConfig validate catches multiple errors", test_launch_config_validate_many_errors)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed, {total} total")
    print("=" * 70)

    sys.exit(0 if failed == 0 else 1)
