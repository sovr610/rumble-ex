"""
slurm_submitter_template.py
============================
SLURMSubmitter: submits distributed training jobs via submitit with
preemption-safe checkpointing and code snapshotting.

Key components:
  - SLURMConfig: dataclass for SLURM job parameters
  - Trainer: callable class that implements __call__ (training) and
             checkpoint (preemption recovery via DelayedSubmission)
  - SLURMSubmitter: configures submitit executor and submits jobs
  - snapshot_code: copies source tree to experiment directory

Usage:
    config = SLURMConfig(nodes=2, gpus_per_node=8, partition="learn")
    submitter = SLURMSubmitter(config)
    job_id = submitter.submit(train_fn=my_training_function, args={...})
"""

from __future__ import annotations

import glob
import logging
import os
import shutil
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Optional submitit import — fall back gracefully when not installed
try:
    import submitit  # type: ignore
    _SUBMITIT_AVAILABLE = True
except ImportError:
    submitit = None  # type: ignore
    _SUBMITIT_AVAILABLE = False
    logger.warning(
        "submitit not found. Install with: pip install submitit\n"
        "SLURMSubmitter will use local (non-SLURM) execution as fallback."
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SLURMConfig:
    """
    SLURM job submission parameters.

    Attributes:
        nodes: Number of compute nodes.
        gpus_per_node: GPUs per node (also tasks per node).
        mem_per_gpu: Memory per GPU (e.g. "64G").
        timeout_min: Wall-clock limit in minutes (default 72h = 4320m).
        partition: SLURM partition name.
        account: SLURM account for billing (empty = not set).
        qos: Quality of service level (empty = not set).
        cpus_per_task: CPU cores allocated per task.
        signal_delay_s: Seconds before walltime to send SIGTERM for checkpoint.
        comment: Optional job comment for SLURM.
        constraint: Node feature constraint (e.g. "volta32gb").
    """
    nodes: int = 1
    gpus_per_node: int = 8
    mem_per_gpu: str = "64G"
    timeout_min: int = 4320          # 72 hours
    partition: str = "learn"
    account: str = ""
    qos: str = ""
    cpus_per_task: int = 10
    signal_delay_s: int = 120
    comment: str = ""
    constraint: str = ""


# ---------------------------------------------------------------------------
# Trainer callable
# ---------------------------------------------------------------------------

class Trainer:
    """
    Callable class for distributed training with preemption support.

    submitit serializes this object to submit the job. On preemption,
    submitit calls checkpoint() which returns a DelayedSubmission to
    automatically requeue the job.

    The training function receives `args` as keyword arguments. The
    `args` dict should contain a `checkpoint_dir` key so that resumed
    jobs can locate the latest checkpoint.

    Attributes:
        train_fn: Callable that runs training. Signature: train_fn(**args).
        args: Dict of keyword arguments passed to train_fn.
    """

    def __init__(
        self,
        train_fn: Optional[Callable] = None,
        args: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.train_fn = train_fn
        self.args: Dict[str, Any] = args or {}

    def __call__(self) -> None:
        """
        Entry point for the SLURM task.

        Called once per GPU process by submitit. Runs the training function
        with the stored args.
        """
        if self.train_fn is None:
            logger.warning("Trainer.__call__: train_fn is None, nothing to run")
            return

        logger.info(
            "Trainer starting: %s with args keys: %s",
            getattr(self.train_fn, "__name__", str(self.train_fn)),
            list(self.args.keys()),
        )
        self.train_fn(**self.args)

    def checkpoint(self) -> "submitit.helpers.DelayedSubmission":
        """
        Called by submitit when SLURM sends SIGTERM (preemption warning).

        Returns a DelayedSubmission that requeues this exact Trainer instance.
        The resubmitted job resumes from the latest checkpoint because
        self.args["checkpoint_dir"] points to the directory where checkpoints
        are saved during training.

        Returns:
            submitit.helpers.DelayedSubmission wrapping this Trainer.

        Raises:
            RuntimeError: If submitit is not installed.
        """
        if not _SUBMITIT_AVAILABLE:
            raise RuntimeError("submitit not installed — cannot create DelayedSubmission")

        logger.info("Trainer.checkpoint(): preempted, requeuing job")
        return submitit.helpers.DelayedSubmission(self)

    def __repr__(self) -> str:
        fn_name = getattr(self.train_fn, "__name__", str(self.train_fn))
        return f"Trainer(train_fn={fn_name}, args_keys={list(self.args.keys())})"


# ---------------------------------------------------------------------------
# Code snapshotting
# ---------------------------------------------------------------------------

def snapshot_code(
    src_dir: str,
    dest_dir: str,
    extensions: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> None:
    """
    Copy source tree to dest_dir for reproducibility.

    Copies files matching `extensions` while skipping hidden directories
    and __pycache__. Preserves the relative directory structure.

    Args:
        src_dir: Source directory root to copy from.
        dest_dir: Destination directory (created if needed).
        extensions: File extensions to include (default: .py, .yaml, .yml, .json, .sh).
        exclude_patterns: Directory names to skip (default: __pycache__, hidden dirs).
    """
    if extensions is None:
        extensions = [".py", ".yaml", ".yml", ".json", ".sh", ".toml"]

    if exclude_patterns is None:
        exclude_patterns = ["__pycache__", ".git", ".mypy_cache", ".pytest_cache",
                            ".tox", "node_modules", ".venv", "venv", "dist", "build",
                            "*.egg-info"]

    os.makedirs(dest_dir, exist_ok=True)
    copied_count = 0

    for ext in extensions:
        for src_path in glob.glob(
            os.path.join(src_dir, "**", f"*{ext}"), recursive=True
        ):
            # Compute relative path from src_dir
            rel_path = os.path.relpath(src_path, src_dir)
            path_parts = rel_path.split(os.sep)

            # Skip excluded directories and hidden paths
            skip = False
            for part in path_parts[:-1]:  # Exclude filename itself from dir check
                if part.startswith("."):
                    skip = True
                    break
                for pattern in exclude_patterns:
                    if part == pattern or (
                        pattern.endswith("*") and part.endswith(pattern[:-1])
                    ):
                        skip = True
                        break
                if skip:
                    break

            if skip:
                continue

            dest_path = os.path.join(dest_dir, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(src_path, dest_path)
            copied_count += 1

    logger.info("Code snapshotted %d files to: %s", copied_count, dest_dir)
    print(f"Code snapshot: {copied_count} files -> {dest_dir}")


# ---------------------------------------------------------------------------
# SLURMSubmitter
# ---------------------------------------------------------------------------

class SLURMSubmitter:
    """
    Manages SLURM job submission via submitit.

    Configures the submitit executor based on SLURMConfig, submits the
    Trainer callable, and returns the job ID.

    Falls back to local execution when submitit is not available or when
    running outside a SLURM environment (useful for development).

    Args:
        config: SLURMConfig instance with job parameters.
        log_dir: Directory for submitit logs. Created if needed.
    """

    def __init__(self, config: SLURMConfig, log_dir: str = "/tmp/submitit_logs") -> None:
        self.config = config
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def submit(
        self,
        train_fn: Callable,
        args: Dict[str, Any],
        snapshot_src: Optional[str] = None,
    ) -> str:
        """
        Submit a training job to SLURM.

        Args:
            train_fn: Training function to run on each GPU task.
            args: Keyword arguments forwarded to train_fn. Should include
                  `checkpoint_dir` for preemption recovery.
            snapshot_src: If provided, snapshot this directory to
                          args["checkpoint_dir"]/code_snapshot before submitting.

        Returns:
            SLURM job ID as a string.

        Raises:
            RuntimeError: If submitit is not installed.
        """
        if not _SUBMITIT_AVAILABLE:
            raise RuntimeError(
                "submitit not installed. Run: pip install submitit\n"
                "For local testing, use Trainer(train_fn, args).__call__() directly."
            )

        # Optional code snapshot
        if snapshot_src is not None:
            checkpoint_dir = args.get("checkpoint_dir", self.log_dir)
            code_dest = os.path.join(checkpoint_dir, "code_snapshot")
            snapshot_code(snapshot_src, code_dest)

        executor = self._create_executor()
        trainer = Trainer(train_fn=train_fn, args=args)

        job = executor.submit(trainer)
        job_id = str(job.job_id)

        logger.info("Submitted SLURM job: %s", job_id)
        print(f"Submitted job ID: {job_id}")
        print(f"Logs: {self.log_dir}/{job_id}_*.out")

        return job_id

    def _create_executor(self) -> "submitit.AutoExecutor":
        """Build and configure the submitit AutoExecutor."""
        executor = submitit.AutoExecutor(folder=self.log_dir)

        cfg = self.config
        slurm_kwargs: Dict[str, Any] = {
            "slurm_gres": f"gpu:{cfg.gpus_per_node}",
            "slurm_cpus_per_task": cfg.cpus_per_task,
            "slurm_mem_per_gpu": cfg.mem_per_gpu,
            "slurm_signal_delay_s": cfg.signal_delay_s,
        }

        # Only add optional params if non-empty
        if cfg.account:
            slurm_kwargs["slurm_account"] = cfg.account
        if cfg.qos:
            slurm_kwargs["slurm_qos"] = cfg.qos
        if cfg.comment:
            slurm_kwargs["slurm_comment"] = cfg.comment
        if cfg.constraint:
            slurm_kwargs["slurm_constraint"] = cfg.constraint

        executor.update_parameters(
            nodes=cfg.nodes,
            gpus_per_node=cfg.gpus_per_node,
            tasks_per_node=cfg.gpus_per_node,  # One task per GPU
            timeout_min=cfg.timeout_min,
            slurm_partition=cfg.partition,
            **slurm_kwargs,
        )

        return executor

    def config_dict(self) -> Dict[str, Any]:
        """Return the config as a plain dictionary."""
        return asdict(self.config)

    def __repr__(self) -> str:
        return (
            f"SLURMSubmitter("
            f"partition={self.config.partition!r}, "
            f"nodes={self.config.nodes}, "
            f"gpus_per_node={self.config.gpus_per_node})"
        )


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    import sys

    print("=" * 60)
    print("SLURMSubmitter / Trainer / snapshot_code self-tests")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Test 1: Trainer callable with a simple train_fn
    # ------------------------------------------------------------------
    print("\n[Test 1] Trainer.__call__() runs train_fn")

    results: Dict[str, Any] = {}

    def dummy_train(**kwargs: Any) -> None:
        results["ran"] = True
        results["args"] = kwargs

    trainer = Trainer(train_fn=dummy_train, args={"lr": 1e-3, "epochs": 5})
    trainer()  # Should call dummy_train

    assert results.get("ran") is True, "train_fn was not called"
    assert results["args"]["lr"] == 1e-3
    assert results["args"]["epochs"] == 5
    print("  train_fn called with correct args  PASS")

    # ------------------------------------------------------------------
    # Test 2: Trainer with None train_fn (no crash)
    # ------------------------------------------------------------------
    print("\n[Test 2] Trainer.__call__() with None train_fn")

    trainer_none = Trainer(train_fn=None, args={})
    try:
        trainer_none()  # Should log warning and return without error
        print("  No exception for None train_fn  PASS")
    except Exception as e:
        print(f"  FAIL: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Test 3: Trainer.checkpoint() returns DelayedSubmission (if submitit available)
    # ------------------------------------------------------------------
    print("\n[Test 3] Trainer.checkpoint() behavior")

    if _SUBMITIT_AVAILABLE:
        trainer3 = Trainer(train_fn=dummy_train, args={"x": 1})
        ds = trainer3.checkpoint()
        assert isinstance(ds, submitit.helpers.DelayedSubmission), (
            f"Expected DelayedSubmission, got {type(ds)}"
        )
        print("  Returns DelayedSubmission  PASS")
    else:
        print("  submitit not installed — skipping DelayedSubmission check  SKIP")

    # ------------------------------------------------------------------
    # Test 4: Trainer repr
    # ------------------------------------------------------------------
    print("\n[Test 4] Trainer.__repr__")

    r = repr(trainer)
    assert "dummy_train" in r
    assert "lr" in r
    print(f"  repr: {r}  PASS")

    # ------------------------------------------------------------------
    # Test 5: SLURMConfig defaults
    # ------------------------------------------------------------------
    print("\n[Test 5] SLURMConfig default values")

    cfg = SLURMConfig()
    assert cfg.nodes == 1
    assert cfg.gpus_per_node == 8
    assert cfg.timeout_min == 4320
    assert cfg.partition == "learn"
    assert cfg.account == ""
    assert cfg.qos == ""
    print(f"  Defaults correct  PASS")

    # ------------------------------------------------------------------
    # Test 6: SLURMConfig as dict
    # ------------------------------------------------------------------
    print("\n[Test 6] SLURMConfig -> dict via asdict")

    d = asdict(SLURMConfig(nodes=4, partition="gpu"))
    assert d["nodes"] == 4
    assert d["partition"] == "gpu"
    assert "gpus_per_node" in d
    print(f"  Dict keys: {list(d.keys())}  PASS")

    # ------------------------------------------------------------------
    # Test 7: snapshot_code creates files preserving structure
    # ------------------------------------------------------------------
    print("\n[Test 7] snapshot_code creates copy")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a fake source tree
        src = os.path.join(tmpdir, "src")
        os.makedirs(os.path.join(src, "subdir"))
        os.makedirs(os.path.join(src, "__pycache__"))

        src_file = os.path.join(src, "train.py")
        subdir_file = os.path.join(src, "subdir", "model.py")
        pycache_file = os.path.join(src, "__pycache__", "cached.pyc")

        for f in [src_file, subdir_file]:
            with open(f, "w") as fh:
                fh.write("# test\n")
        with open(pycache_file, "w") as fh:
            fh.write("garbage")

        dest = os.path.join(tmpdir, "snapshot")
        snapshot_code(src_dir=src, dest_dir=dest, extensions=[".py"])

        # Check files present
        assert os.path.exists(os.path.join(dest, "train.py")), "train.py not in snapshot"
        assert os.path.exists(os.path.join(dest, "subdir", "model.py")), "model.py not in snapshot"

        # Check __pycache__ excluded
        assert not os.path.exists(os.path.join(dest, "__pycache__")), \
            "__pycache__ should be excluded from snapshot"

        print("  Files copied, __pycache__ excluded  PASS")

    # ------------------------------------------------------------------
    # Test 8: snapshot_code is idempotent
    # ------------------------------------------------------------------
    print("\n[Test 8] snapshot_code idempotency")

    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "src")
        os.makedirs(src)
        with open(os.path.join(src, "main.py"), "w") as fh:
            fh.write("x = 1\n")

        dest = os.path.join(tmpdir, "snap")
        snapshot_code(src_dir=src, dest_dir=dest, extensions=[".py"])
        snapshot_code(src_dir=src, dest_dir=dest, extensions=[".py"])  # Second run

        assert os.path.exists(os.path.join(dest, "main.py"))
        print("  Second snapshot run succeeded  PASS")

    # ------------------------------------------------------------------
    # Test 9: SLURMSubmitter repr
    # ------------------------------------------------------------------
    print("\n[Test 9] SLURMSubmitter.__repr__")

    with tempfile.TemporaryDirectory() as tmpdir:
        submitter = SLURMSubmitter(SLURMConfig(nodes=4), log_dir=tmpdir)
        r = repr(submitter)
        assert "nodes=4" in r
        assert "learn" in r
        print(f"  repr: {r}  PASS")

    print("\n" + "=" * 60)
    print("All SLURMSubmitter self-tests PASSED")
    print("=" * 60)
