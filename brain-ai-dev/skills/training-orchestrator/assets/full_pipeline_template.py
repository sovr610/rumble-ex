#!/usr/bin/env python3
"""
full_pipeline_template.py -- Multi-Phase Pipeline Orchestrator for Brain-AI Training

Orchestrates the seven-phase brain_ai training pipeline, chaining phases
sequentially with boundary validation, resume support, forking, health checks,
reporting, and CLI integration.

Phases:
    1. SNN Core
    2. Modality Encoders
    3. HTM (Hierarchical Temporal Memory)
    4. Global Workspace
    5. Active Inference
    6. Reasoning (Dual-Process)
    7. Meta-Learning (MAML / FOMAML / Reptile)

Classes:
    PipelineState           -- Tracks execution state across phases.
    PipelineOrchestrator    -- Main driver: runs phases, validates boundaries, handles failures.
    PipelineResumeManager   -- Resume an interrupted pipeline from the last completed phase.
    PipelineForkManager     -- Fork a pipeline at a phase boundary for ablation experiments.
    PipelineReport          -- Generate and compare pipeline run reports.
    PipelineCLI             -- Argparse-based CLI for train_full_pipeline.py.
    PhaseTransitionLogger   -- Structured logging for phase transitions.
    PipelineHealthCheck     -- Pre-phase health checks (disk, GPU, dependencies).

Self-contained: no brain_ai imports required.  Uses mock PhaseRunner for self-test.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import logging
import os
import shutil
import signal
import sys
import tempfile
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHASE_NAMES: Dict[int, str] = {
    1: "SNN Core",
    2: "Modality Encoders",
    3: "HTM",
    4: "Global Workspace",
    5: "Active Inference",
    6: "Reasoning",
    7: "Meta-Learning",
}

MIN_PHASE: int = 1
MAX_PHASE: int = 7

SCHEMA_VERSION: str = "1.0"

# Phase-specific optional dependencies.  Maps phase number to a list of
# (package_name, fallback_description) tuples.
PHASE_DEPENDENCIES: Dict[int, List[Tuple[str, str]]] = {
    1: [],
    2: [("ncps", "GRU fallback for CfC/LTC neurons")],
    3: [("htm.core", "LSTM fallback for HTM temporal memory")],
    4: [],
    5: [("pymdp", "Custom active inference backend fallback")],
    6: [],
    7: [("learn2learn", "Custom MAML implementation fallback")],
}


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _get_logger(name: str = "pipeline") -> logging.Logger:
    """Return a module-level logger with a StreamHandler if none is attached."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = _get_logger()


# ---------------------------------------------------------------------------
# Mock PhaseRunner (for standalone self-test; real implementation lives in
# phase_runner_template.py)
# ---------------------------------------------------------------------------

@dataclass
class PhaseResult:
    """Result of running a single training phase."""
    phase: int
    status: str  # "completed", "failed", "interrupted"
    best_metrics: Dict[str, float] = field(default_factory=dict)
    final_metrics: Dict[str, float] = field(default_factory=dict)
    duration_seconds: float = 0.0
    error: Optional[str] = None
    checkpoints_saved: int = 0


class MockPhaseRunner:
    """Mock phase runner that completes instantly with synthetic metrics.

    Parameters
    ----------
    fail_on_phase : int or None
        If set, raise RuntimeError when this phase is run.
    interrupt_on_phase : int or None
        If set, raise KeyboardInterrupt when this phase is run.
    """

    def __init__(
        self,
        fail_on_phase: Optional[int] = None,
        interrupt_on_phase: Optional[int] = None,
    ) -> None:
        self.fail_on_phase = fail_on_phase
        self.interrupt_on_phase = interrupt_on_phase
        self.phases_run: List[int] = []

    def run(self, phase: int, config: Dict[str, Any], run_dir: str) -> PhaseResult:
        """Simulate running a training phase."""
        self.phases_run.append(phase)

        if self.interrupt_on_phase == phase:
            raise KeyboardInterrupt(f"Simulated interrupt on phase {phase}")

        if self.fail_on_phase == phase:
            raise RuntimeError(f"Simulated failure on phase {phase}")

        # Synthetic metrics: loss decreases with phase, accuracy increases.
        base_loss = 1.0 - (phase * 0.1)
        base_acc = 0.5 + (phase * 0.05)
        return PhaseResult(
            phase=phase,
            status="completed",
            best_metrics={"val_loss": base_loss, "val_acc": base_acc},
            final_metrics={"train_loss": base_loss - 0.02, "val_loss": base_loss},
            duration_seconds=0.01 * phase,
            checkpoints_saved=2,
        )


# ---------------------------------------------------------------------------
# Mock PhaseBoundaryValidator (for standalone self-test; real implementation
# lives in phase_boundary_template.py)
# ---------------------------------------------------------------------------

class MockBoundaryValidator:
    """Mock boundary validator that checks for boundary artifact file existence."""

    def __init__(self, mode: str = "normal") -> None:
        self.mode = mode
        self.validations: List[Tuple[int, int]] = []

    def validate(
        self,
        run_dir: str,
        target_phase: int,
        config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Validate phase transition by checking boundary artifact existence."""
        source_phase = target_phase - 1
        self.validations.append((source_phase, target_phase))

        boundary_path = Path(run_dir) / "checkpoints" / f"phase{source_phase}" / "phase_boundary.pt"
        if not boundary_path.exists():
            logger.error(
                f"Boundary artifact not found: {boundary_path}. "
                f"Phase {source_phase} must complete before phase {target_phase}."
            )
            return False
        return True


def _create_mock_boundary(run_dir: str, source_phase: int) -> None:
    """Create a mock phase_boundary.pt file for testing."""
    ckpt_dir = Path(run_dir) / "checkpoints" / f"phase{source_phase}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    boundary_path = ckpt_dir / "phase_boundary.pt"
    boundary_data = {
        "schema_version": SCHEMA_VERSION,
        "source_phase": source_phase,
        "target_phase": source_phase + 1,
        "model_state_dict": {},
        "config_snapshot": {},
        "compatibility": {"workspace_dim": 4096},
        "metadata": {"timestamp": datetime.datetime.utcnow().isoformat() + "Z"},
    }
    with open(str(boundary_path), "w") as f:
        json.dump(boundary_data, f)


# =========================================================================
# PipelineState
# =========================================================================

class PipelineState:
    """Tracks pipeline execution state across phases.

    Provides save/load for persistence and resume-point detection.

    Attributes
    ----------
    phases_completed : list of int
        Phase numbers that finished successfully.
    phases_failed : dict of int to str
        Phase number to error message for phases that failed.
    current_phase : int or None
        Phase currently running, or None if idle.
    phase_durations : dict of int to float
        Phase number to wall-clock seconds.
    phase_metrics : dict of int to dict of str to float
        Phase number to best metrics dict.
    run_id : str
        The pipeline run identifier.
    start_phase : int
        The first phase in this pipeline run.
    end_phase : int
        The last phase in this pipeline run.
    status : str
        One of "pending", "running", "completed", "failed", "interrupted".
    """

    def __init__(
        self,
        run_id: str = "",
        start_phase: int = MIN_PHASE,
        end_phase: int = MAX_PHASE,
    ) -> None:
        self.run_id: str = run_id
        self.start_phase: int = start_phase
        self.end_phase: int = end_phase
        self.phases_completed: List[int] = []
        self.phases_failed: Dict[int, str] = {}
        self.current_phase: Optional[int] = None
        self.phase_durations: Dict[int, float] = {}
        self.phase_metrics: Dict[int, Dict[str, float]] = {}
        self.status: str = "pending"
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def save(self, run_dir: str) -> str:
        """Persist state to ``pipeline_state.json`` in *run_dir*.

        Returns the path to the saved file.
        """
        state_path = os.path.join(run_dir, "pipeline_state.json")
        data = {
            "run_id": self.run_id,
            "start_phase": self.start_phase,
            "end_phase": self.end_phase,
            "phases_completed": self.phases_completed,
            "phases_failed": {str(k): v for k, v in self.phases_failed.items()},
            "current_phase": self.current_phase,
            "phase_durations": {str(k): v for k, v in self.phase_durations.items()},
            "phase_metrics": {str(k): v for k, v in self.phase_metrics.items()},
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }
        tmp_path = state_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, state_path)
        return state_path

    @classmethod
    def load(cls, run_dir: str) -> "PipelineState":
        """Restore state from ``pipeline_state.json`` in *run_dir*."""
        state_path = os.path.join(run_dir, "pipeline_state.json")
        with open(state_path, "r") as f:
            data = json.load(f)

        state = cls(
            run_id=data["run_id"],
            start_phase=data.get("start_phase", MIN_PHASE),
            end_phase=data.get("end_phase", MAX_PHASE),
        )
        state.phases_completed = data.get("phases_completed", [])
        state.phases_failed = {
            int(k): v for k, v in data.get("phases_failed", {}).items()
        }
        state.current_phase = data.get("current_phase")
        state.phase_durations = {
            int(k): v for k, v in data.get("phase_durations", {}).items()
        }
        state.phase_metrics = {
            int(k): v for k, v in data.get("phase_metrics", {}).items()
        }
        state.status = data.get("status", "pending")
        state.start_time = data.get("start_time")
        state.end_time = data.get("end_time")
        return state

    def is_resumable(self) -> bool:
        """Return True if at least one phase completed and pipeline is not done."""
        if not self.phases_completed:
            return False
        if self.status == "completed":
            return False
        return True

    def next_phase(self) -> Optional[int]:
        """Return the next phase to run after the last completed phase.

        Returns None if all requested phases are complete or if no phases
        have been completed and start_phase should be used.
        """
        if not self.phases_completed:
            return self.start_phase
        last = max(self.phases_completed)
        nxt = last + 1
        if nxt > self.end_phase:
            return None
        return nxt

    def total_duration(self) -> float:
        """Return sum of all phase durations."""
        return sum(self.phase_durations.values())

    def __repr__(self) -> str:
        return (
            f"PipelineState(run_id='{self.run_id}', status='{self.status}', "
            f"completed={self.phases_completed}, "
            f"current={self.current_phase})"
        )


# =========================================================================
# PhaseTransitionLogger
# =========================================================================

class PhaseTransitionLogger:
    """Structured logging for phase transitions.

    Logs banners to the console, structured JSON events to a JSONL file,
    and optionally to TensorBoard (if a writer is provided).
    """

    def __init__(self, jsonl_path: Optional[str] = None) -> None:
        self.jsonl_path = jsonl_path

    def log_transition(
        self,
        from_phase: int,
        to_phase: int,
        metrics: Dict[str, float],
        duration: float,
    ) -> None:
        """Log a phase transition to all backends.

        Parameters
        ----------
        from_phase : int
            Phase that just completed.
        to_phase : int
            Phase about to start.
        metrics : dict
            Best metrics from the completed phase.
        duration : float
            Wall-clock seconds for the completed phase.
        """
        # Console banner
        metric_str = ", ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))
        banner = (
            f"=== Phase {from_phase} ({PHASE_NAMES.get(from_phase, '?')}) -> "
            f"Phase {to_phase} ({PHASE_NAMES.get(to_phase, '?')}) | "
            f"{metric_str} | {duration:.1f}s ==="
        )
        logger.info(banner)

        # JSONL event
        if self.jsonl_path:
            event = {
                "event": "phase_transition",
                "from_phase": from_phase,
                "to_phase": to_phase,
                "metrics": metrics,
                "duration_seconds": duration,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            }
            with open(self.jsonl_path, "a") as f:
                f.write(json.dumps(event) + "\n")

    def log_pipeline_start(self, run_id: str, start_phase: int, end_phase: int) -> None:
        """Log pipeline start event."""
        logger.info(
            f"=== Pipeline Start: {run_id} | "
            f"Phases {start_phase}-{end_phase} ==="
        )
        if self.jsonl_path:
            event = {
                "event": "pipeline_start",
                "run_id": run_id,
                "start_phase": start_phase,
                "end_phase": end_phase,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            }
            with open(self.jsonl_path, "a") as f:
                f.write(json.dumps(event) + "\n")

    def log_pipeline_end(self, run_id: str, status: str, total_duration: float) -> None:
        """Log pipeline completion event."""
        logger.info(
            f"=== Pipeline End: {run_id} | "
            f"Status: {status} | {total_duration:.1f}s ==="
        )
        if self.jsonl_path:
            event = {
                "event": "pipeline_end",
                "run_id": run_id,
                "status": status,
                "total_duration_seconds": total_duration,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            }
            with open(self.jsonl_path, "a") as f:
                f.write(json.dumps(event) + "\n")


# =========================================================================
# PipelineHealthCheck
# =========================================================================

class PipelineHealthCheck:
    """Pre-phase health checks for disk space, GPU memory, and dependencies.

    All checks log warnings but do not abort unless explicitly requested.
    """

    def __init__(self, strict: bool = False) -> None:
        self.strict = strict
        self.warnings: List[str] = []
        self.errors: List[str] = []

    def check_disk_space(self, run_dir: str, required_gb: float = 10.0) -> bool:
        """Warn if available disk space in *run_dir* is below *required_gb*."""
        try:
            stat = shutil.disk_usage(run_dir)
            available_gb = stat.free / (1024 ** 3)
            if available_gb < required_gb:
                msg = (
                    f"Low disk space: {available_gb:.1f} GB available, "
                    f"{required_gb:.1f} GB required in {run_dir}"
                )
                self.warnings.append(msg)
                logger.warning(msg)
                return False
            return True
        except OSError as e:
            msg = f"Could not check disk space for {run_dir}: {e}"
            self.warnings.append(msg)
            logger.warning(msg)
            return False

    def check_gpu_memory(self) -> bool:
        """Warn if GPU memory is fragmented or low.

        Checks are skipped gracefully when CUDA is unavailable.
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return True
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_mb = props.total_mem / (1024 ** 2)
                free_mb = (props.total_mem - torch.cuda.memory_allocated(i)) / (1024 ** 2)
                if free_mb < total_mb * 0.1:
                    msg = (
                        f"GPU {i} ({props.name}): only {free_mb:.0f} MB free "
                        f"of {total_mb:.0f} MB total"
                    )
                    self.warnings.append(msg)
                    logger.warning(msg)
                    return False
            return True
        except Exception as e:
            msg = f"GPU memory check skipped: {e}"
            self.warnings.append(msg)
            logger.debug(msg)
            return True

    def check_dependencies(self, phase: int) -> bool:
        """Verify phase-specific optional dependencies are installed.

        Returns True if all required dependencies are available or have
        known fallbacks.
        """
        deps = PHASE_DEPENDENCIES.get(phase, [])
        all_ok = True
        for pkg_name, fallback_desc in deps:
            try:
                __import__(pkg_name.replace(".", "_") if "." in pkg_name else pkg_name)
            except ImportError:
                msg = (
                    f"Phase {phase}: optional dependency '{pkg_name}' not installed. "
                    f"Fallback: {fallback_desc}"
                )
                self.warnings.append(msg)
                logger.warning(msg)
                # Not a hard failure because fallbacks exist.
        return all_ok

    def run_all(self, phase: int, run_dir: str, required_gb: float = 10.0) -> bool:
        """Execute all health checks before a phase starts.

        Returns True if all checks pass.  In strict mode, warnings become errors.
        """
        self.warnings.clear()
        self.errors.clear()

        disk_ok = self.check_disk_space(run_dir, required_gb)
        gpu_ok = self.check_gpu_memory()
        deps_ok = self.check_dependencies(phase)

        if self.strict and self.warnings:
            self.errors.extend(self.warnings)
            return False
        return disk_ok and gpu_ok and deps_ok


# =========================================================================
# PipelineOrchestrator
# =========================================================================

class PipelineOrchestrator:
    """Main pipeline driver that executes phases sequentially.

    Parameters
    ----------
    config : dict
        Configuration dictionary (mock-compatible; real runs use BrainAIConfig).
    mode : str
        Training mode, ``"dev"`` or ``"production"``.
    run_dir : str
        Base directory for pipeline runs.
    start_phase : int
        First phase to execute (1-7).
    end_phase : int
        Last phase to execute (1-7).
    phase_runner : object or None
        Object with a ``run(phase, config, run_dir)`` method.
        Defaults to ``MockPhaseRunner``.
    boundary_validator : object or None
        Object with a ``validate(run_dir, target_phase, config)`` method.
        Defaults to ``MockBoundaryValidator``.
    health_checker : PipelineHealthCheck or None
        Pre-phase health checker.
    transition_logger : PhaseTransitionLogger or None
        Structured phase transition logger.
    run_id : str or None
        Explicit run ID.  If None, one is generated.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        mode: str = "dev",
        run_dir: str = "runs/",
        start_phase: int = MIN_PHASE,
        end_phase: int = MAX_PHASE,
        phase_runner: Any = None,
        boundary_validator: Any = None,
        health_checker: Optional[PipelineHealthCheck] = None,
        transition_logger: Optional[PhaseTransitionLogger] = None,
        run_id: Optional[str] = None,
    ) -> None:
        if not MIN_PHASE <= start_phase <= MAX_PHASE:
            raise ValueError(f"start_phase must be in [{MIN_PHASE}, {MAX_PHASE}], got {start_phase}")
        if not MIN_PHASE <= end_phase <= MAX_PHASE:
            raise ValueError(f"end_phase must be in [{MIN_PHASE}, {MAX_PHASE}], got {end_phase}")
        if start_phase > end_phase:
            raise ValueError(
                f"start_phase ({start_phase}) must be <= end_phase ({end_phase})"
            )

        self.config = config
        self.mode = mode
        self.base_run_dir = run_dir
        self.start_phase = start_phase
        self.end_phase = end_phase
        self.phase_runner = phase_runner or MockPhaseRunner()
        self.boundary_validator = boundary_validator or MockBoundaryValidator()
        self.health_checker = health_checker or PipelineHealthCheck()
        self.run_id = run_id or self._generate_run_id()
        self.run_dir: str = ""  # Set in setup_run
        self.state: PipelineState = PipelineState(
            run_id=self.run_id,
            start_phase=self.start_phase,
            end_phase=self.end_phase,
        )
        self.transition_logger = transition_logger

    # ----- run ID generation ------------------------------------------------

    @staticmethod
    def _generate_run_id() -> str:
        """Generate a unique run ID based on UTC timestamp and random suffix."""
        ts = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
        suffix = uuid.uuid4().hex[:7]
        return f"{ts}_full_pipeline_{suffix}"

    # ----- setup ------------------------------------------------------------

    def setup_run(self) -> str:
        """Create the run directory tree and initialize state.

        Returns the path to the run directory.
        """
        self.run_dir = os.path.join(self.base_run_dir, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)

        # Create phase checkpoint directories
        for phase in range(self.start_phase, self.end_phase + 1):
            phase_dir = os.path.join(self.run_dir, "checkpoints", f"phase{phase}")
            os.makedirs(phase_dir, exist_ok=True)

        # Create supporting directories
        for subdir in ["logs/tensorboard", "logs/wandb", "artifacts", "reports"]:
            os.makedirs(os.path.join(self.run_dir, subdir), exist_ok=True)

        # Initialize transition logger JSONL path
        if self.transition_logger is None:
            jsonl_path = os.path.join(self.run_dir, "metrics.jsonl")
            self.transition_logger = PhaseTransitionLogger(jsonl_path=jsonl_path)

        # Save initial manifest
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "mode": self.mode,
            "start_phase": self.start_phase,
            "end_phase": self.end_phase,
            "config": self.config,
            "timestamp_start": datetime.datetime.utcnow().isoformat() + "Z",
        }
        manifest_path = os.path.join(self.run_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # Initialize state
        self.state.status = "pending"
        self.state.start_time = time.time()
        self.state.save(self.run_dir)

        logger.info(f"Pipeline run initialized: {self.run_dir}")
        return self.run_dir

    # ----- phase execution --------------------------------------------------

    def run_phase(self, phase: int) -> PhaseResult:
        """Delegate to the phase runner and capture results.

        Parameters
        ----------
        phase : int
            Phase number to execute.

        Returns
        -------
        PhaseResult
            Result from the phase runner.

        Raises
        ------
        RuntimeError
            If the phase runner raises an exception.
        """
        self.state.current_phase = phase
        self.state.save(self.run_dir)

        logger.info(
            f"Starting Phase {phase}: {PHASE_NAMES.get(phase, 'Unknown')}"
        )

        start = time.time()
        result = self.phase_runner.run(phase, self.config, self.run_dir)
        elapsed = time.time() - start

        # Override duration with measured time if phase completed
        if result.duration_seconds == 0.0:
            result.duration_seconds = elapsed

        return result

    # ----- boundary validation ----------------------------------------------

    def validate_phase_transition(self, from_phase: int, to_phase: int) -> bool:
        """Validate the boundary between two phases.

        Parameters
        ----------
        from_phase : int
            Phase that produced the boundary artifact.
        to_phase : int
            Phase about to consume it.

        Returns
        -------
        bool
            True if validation passes.
        """
        logger.info(
            f"Validating boundary: Phase {from_phase} -> Phase {to_phase}"
        )
        return self.boundary_validator.validate(
            self.run_dir, to_phase, self.config
        )

    # ----- failure handling -------------------------------------------------

    def handle_phase_failure(
        self, phase: int, error: Exception
    ) -> bool:
        """Record a phase failure and decide whether to continue.

        Parameters
        ----------
        phase : int
            Phase that failed.
        error : Exception
            The exception that caused the failure.

        Returns
        -------
        bool
            True if the pipeline should attempt to continue (currently always False).
        """
        error_msg = f"{type(error).__name__}: {error}"
        self.state.phases_failed[phase] = error_msg
        self.state.current_phase = None
        self.state.status = "failed"
        self.state.end_time = time.time()
        self.state.save(self.run_dir)

        logger.error(f"Phase {phase} failed: {error_msg}")

        # Policy: abort pipeline on phase failure (no skip-and-continue).
        return False

    # ----- main run loop ----------------------------------------------------

    def run(self) -> PipelineState:
        """Execute phases sequentially from start_phase to end_phase.

        Returns
        -------
        PipelineState
            Final pipeline state.
        """
        if not self.run_dir:
            self.setup_run()

        self.state.status = "running"
        self.state.save(self.run_dir)

        if self.transition_logger:
            self.transition_logger.log_pipeline_start(
                self.run_id, self.start_phase, self.end_phase
            )

        prev_phase: Optional[int] = None

        for phase in range(self.start_phase, self.end_phase + 1):
            # Health check
            self.health_checker.run_all(phase, self.run_dir)

            # Boundary validation (skip for phase 1 or the start phase if
            # boundary was pre-validated during resume/fork)
            if phase > MIN_PHASE and phase > self.start_phase:
                if not self.validate_phase_transition(phase - 1, phase):
                    err = RuntimeError(
                        f"Boundary validation failed: Phase {phase - 1} -> Phase {phase}"
                    )
                    self.handle_phase_failure(phase, err)
                    return self.state
            elif phase > MIN_PHASE and phase == self.start_phase:
                # Validate that the boundary from the prior phase exists when
                # starting mid-pipeline.
                if not self.validate_phase_transition(phase - 1, phase):
                    err = RuntimeError(
                        f"Cannot start from Phase {phase}: boundary artifact "
                        f"from Phase {phase - 1} not found or invalid."
                    )
                    self.handle_phase_failure(phase, err)
                    return self.state

            # Run phase
            try:
                result = self.run_phase(phase)
            except KeyboardInterrupt:
                self.state.status = "interrupted"
                self.state.current_phase = phase
                self.state.end_time = time.time()
                self.state.save(self.run_dir)
                logger.warning(f"Pipeline interrupted during Phase {phase}")
                if self.transition_logger:
                    self.transition_logger.log_pipeline_end(
                        self.run_id, "interrupted", self.state.total_duration()
                    )
                return self.state
            except Exception as e:
                should_continue = self.handle_phase_failure(phase, e)
                if not should_continue:
                    if self.transition_logger:
                        self.transition_logger.log_pipeline_end(
                            self.run_id, "failed", self.state.total_duration()
                        )
                    return self.state

            if result.status != "completed":
                err = RuntimeError(
                    f"Phase {phase} returned status '{result.status}'"
                )
                self.handle_phase_failure(phase, err)
                if self.transition_logger:
                    self.transition_logger.log_pipeline_end(
                        self.run_id, "failed", self.state.total_duration()
                    )
                return self.state

            # Record success
            self.state.phases_completed.append(phase)
            self.state.phase_durations[phase] = result.duration_seconds
            self.state.phase_metrics[phase] = result.best_metrics
            self.state.current_phase = None

            # Save mock boundary artifact for next phase
            _create_mock_boundary(self.run_dir, phase)

            self.state.save(self.run_dir)

            # Log transition
            if self.transition_logger and phase < self.end_phase:
                self.transition_logger.log_transition(
                    phase, phase + 1, result.best_metrics, result.duration_seconds
                )

            prev_phase = phase

        # All phases completed
        self.state.status = "completed"
        self.state.end_time = time.time()
        self.state.save(self.run_dir)

        if self.transition_logger:
            self.transition_logger.log_pipeline_end(
                self.run_id, "completed", self.state.total_duration()
            )

        return self.state

    # ----- resume -----------------------------------------------------------

    def resume_from(self, run_id: str, phase: Optional[int] = None) -> PipelineState:
        """Resume an existing pipeline run from a specific phase.

        Parameters
        ----------
        run_id : str
            Run ID of the pipeline to resume.
        phase : int or None
            Phase to resume from.  If None, auto-detect from state.

        Returns
        -------
        PipelineState
            Final pipeline state after resumed execution.
        """
        resume_dir = os.path.join(self.base_run_dir, run_id)
        if not os.path.exists(resume_dir):
            raise FileNotFoundError(f"Run directory not found: {resume_dir}")

        old_state = PipelineState.load(resume_dir)

        if not old_state.is_resumable():
            raise ValueError(f"Run {run_id} is not resumable (status={old_state.status})")

        resume_phase = phase if phase is not None else old_state.next_phase()
        if resume_phase is None:
            raise ValueError(f"Run {run_id} has no next phase to resume")

        # Reconfigure orchestrator for resume
        self.run_id = run_id
        self.run_dir = resume_dir
        self.start_phase = resume_phase
        self.state = old_state
        self.state.status = "running"
        self.state.start_phase = resume_phase

        logger.info(
            f"Resuming pipeline {run_id} from Phase {resume_phase}"
        )

        return self.run()

    # ----- finalize ---------------------------------------------------------

    def finalize(self) -> str:
        """Write summary report and close logger.

        Returns path to the summary report.
        """
        report = PipelineReport()
        report_path = report.generate(self.run_dir, self.state)
        report.print_summary()
        return report_path


# =========================================================================
# PipelineResumeManager
# =========================================================================

class PipelineResumeManager:
    """Manages resuming interrupted pipeline runs.

    Resume creates new manifest entries (append-only) and does NOT overwrite
    original state.
    """

    def can_resume(self, run_id: str, base_dir: str = "runs/") -> bool:
        """Check if a pipeline state exists and is resumable.

        Parameters
        ----------
        run_id : str
            The run identifier to check.
        base_dir : str
            Parent directory for runs.

        Returns
        -------
        bool
            True if the run exists and can be resumed.
        """
        run_dir = os.path.join(base_dir, run_id)
        state_path = os.path.join(run_dir, "pipeline_state.json")
        if not os.path.exists(state_path):
            return False
        try:
            state = PipelineState.load(run_dir)
            return state.is_resumable()
        except (json.JSONDecodeError, KeyError):
            return False

    def find_resume_point(self, run_dir: str) -> Optional[int]:
        """Determine which phase to resume from.

        Returns the phase number after the last completed phase, or None
        if the run cannot be resumed.
        """
        try:
            state = PipelineState.load(run_dir)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        return state.next_phase()

    def validate_resume(
        self,
        run_dir: str,
        current_config: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """Check config compatibility between the original run and current config.

        Parameters
        ----------
        run_dir : str
            Path to the run directory.
        current_config : dict
            Current configuration to compare against.

        Returns
        -------
        tuple of (bool, list of str)
            (is_compatible, list_of_warnings)
        """
        warnings: List[str] = []
        manifest_path = os.path.join(run_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            return True, ["No manifest found; cannot validate config compatibility"]

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        original_config = manifest.get("config", {})

        # Check critical fields
        critical_keys = ["workspace_dim", "mode"]
        for key in critical_keys:
            orig_val = original_config.get(key)
            curr_val = current_config.get(key)
            if orig_val is not None and curr_val is not None and orig_val != curr_val:
                warnings.append(
                    f"Config mismatch on '{key}': original={orig_val}, current={curr_val}"
                )

        is_compatible = len(warnings) == 0
        return is_compatible, warnings

    def resume(
        self,
        run_id: str,
        base_dir: str = "runs/",
        config: Optional[Dict[str, Any]] = None,
        phase_runner: Any = None,
        boundary_validator: Any = None,
    ) -> PipelineOrchestrator:
        """Load state, validate, and return a PipelineOrchestrator configured for resume.

        Parameters
        ----------
        run_id : str
            The run identifier.
        base_dir : str
            Parent directory for runs.
        config : dict or None
            Override config.  If None, uses the original manifest config.
        phase_runner : object or None
            Phase runner to use.
        boundary_validator : object or None
            Boundary validator to use.

        Returns
        -------
        PipelineOrchestrator
            Configured for resume.
        """
        run_dir = os.path.join(base_dir, run_id)
        if not self.can_resume(run_id, base_dir):
            raise ValueError(f"Run {run_id} cannot be resumed")

        state = PipelineState.load(run_dir)
        resume_phase = state.next_phase()
        if resume_phase is None:
            raise ValueError(f"Run {run_id} has no next phase")

        # Load config from manifest if not provided
        if config is None:
            manifest_path = os.path.join(run_dir, "manifest.json")
            if os.path.exists(manifest_path):
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                config = manifest.get("config", {})
            else:
                config = {}

        # Validate config compatibility
        compatible, warnings = self.validate_resume(run_dir, config)
        for w in warnings:
            logger.warning(f"Resume config warning: {w}")

        orchestrator = PipelineOrchestrator(
            config=config,
            mode=config.get("mode", "dev"),
            run_dir=base_dir,
            start_phase=resume_phase,
            end_phase=state.end_phase,
            phase_runner=phase_runner,
            boundary_validator=boundary_validator,
            run_id=run_id,
        )

        # Transfer existing state
        orchestrator.run_dir = run_dir
        orchestrator.state = state
        orchestrator.state.start_phase = resume_phase
        orchestrator.state.status = "running"

        # Append resume event to manifest
        manifest_path = os.path.join(run_dir, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            resume_events = manifest.get("resume_events", [])
            resume_events.append({
                "resume_phase": resume_phase,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            })
            manifest["resume_events"] = resume_events
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

        return orchestrator


# =========================================================================
# PipelineForkManager
# =========================================================================

class PipelineForkManager:
    """Fork a pipeline at a phase boundary for ablation experiments.

    Creates a new run_id, references (or copies) the boundary artifact,
    and records provenance linking the fork to its source.
    """

    def fork(
        self,
        source_run_id: str,
        fork_phase: int,
        new_config: Dict[str, Any],
        base_dir: str = "runs/",
    ) -> str:
        """Create a new run from an existing phase boundary.

        Parameters
        ----------
        source_run_id : str
            The original run to fork from.
        fork_phase : int
            The phase whose boundary artifact to use as the starting point.
            The new run begins at ``fork_phase + 1``.
        new_config : dict
            Configuration for the forked run.
        base_dir : str
            Parent directory for runs.

        Returns
        -------
        str
            The new run_id.
        """
        source_dir = os.path.join(base_dir, source_run_id)
        boundary_path = os.path.join(
            source_dir, "checkpoints", f"phase{fork_phase}", "phase_boundary.pt"
        )
        if not os.path.exists(boundary_path):
            raise FileNotFoundError(
                f"Boundary artifact not found: {boundary_path}. "
                f"Cannot fork from Phase {fork_phase} of run {source_run_id}."
            )

        # Generate new run_id
        ts = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
        suffix = uuid.uuid4().hex[:7]
        new_run_id = f"{ts}_fork_p{fork_phase}_{suffix}"

        # Create new run directory
        new_run_dir = os.path.join(base_dir, new_run_id)
        os.makedirs(new_run_dir, exist_ok=True)

        # Copy boundary artifact into the new run's checkpoint tree
        new_ckpt_dir = os.path.join(
            new_run_dir, "checkpoints", f"phase{fork_phase}"
        )
        os.makedirs(new_ckpt_dir, exist_ok=True)
        shutil.copy2(boundary_path, os.path.join(new_ckpt_dir, "phase_boundary.pt"))

        # Create supporting directories
        for subdir in [
            "logs/tensorboard", "logs/wandb", "artifacts", "reports",
        ]:
            os.makedirs(os.path.join(new_run_dir, subdir), exist_ok=True)
        for phase in range(fork_phase + 1, MAX_PHASE + 1):
            os.makedirs(
                os.path.join(new_run_dir, "checkpoints", f"phase{phase}"),
                exist_ok=True,
            )

        # Record provenance
        provenance = {
            "schema_version": SCHEMA_VERSION,
            "run_id": new_run_id,
            "forked_from": {
                "source_run_id": source_run_id,
                "fork_phase": fork_phase,
                "source_boundary_path": boundary_path,
            },
            "config": new_config,
            "start_phase": fork_phase + 1,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        }
        manifest_path = os.path.join(new_run_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(provenance, f, indent=2)

        # Initialize pipeline state
        state = PipelineState(
            run_id=new_run_id,
            start_phase=fork_phase + 1,
            end_phase=MAX_PHASE,
        )
        state.save(new_run_dir)

        logger.info(
            f"Forked run {source_run_id} at Phase {fork_phase} -> new run {new_run_id}"
        )
        return new_run_id

    def list_forks(self, source_run_id: str, base_dir: str = "runs/") -> List[str]:
        """Find all runs forked from a given source run.

        Scans manifest files in *base_dir* for provenance links.

        Parameters
        ----------
        source_run_id : str
            The source run to search for forks of.
        base_dir : str
            Parent directory for runs.

        Returns
        -------
        list of str
            Run IDs of forked runs.
        """
        forks: List[str] = []
        if not os.path.exists(base_dir):
            return forks

        for entry in os.listdir(base_dir):
            manifest_path = os.path.join(base_dir, entry, "manifest.json")
            if not os.path.isfile(manifest_path):
                continue
            try:
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                forked_from = manifest.get("forked_from", {})
                if forked_from.get("source_run_id") == source_run_id:
                    forks.append(entry)
            except (json.JSONDecodeError, KeyError):
                continue

        return sorted(forks)


# =========================================================================
# PipelineReport
# =========================================================================

class PipelineReport:
    """Generate and compare pipeline run reports.

    Reports are written to ``reports/summary.json`` within the run directory.
    """

    def __init__(self) -> None:
        self._report: Dict[str, Any] = {}

    def generate(
        self,
        run_dir: str,
        state: Optional[PipelineState] = None,
    ) -> str:
        """Create ``reports/summary.json`` from pipeline state.

        Parameters
        ----------
        run_dir : str
            Path to the run directory.
        state : PipelineState or None
            Pipeline state.  If None, loads from ``pipeline_state.json``.

        Returns
        -------
        str
            Path to the generated summary file.
        """
        if state is None:
            state = PipelineState.load(run_dir)

        # Aggregate best metrics across phases
        best_metrics_all: Dict[str, float] = {}
        for phase_num, metrics in state.phase_metrics.items():
            for k, v in metrics.items():
                if k not in best_metrics_all:
                    best_metrics_all[k] = v
                else:
                    # For loss: keep minimum; for accuracy: keep maximum.
                    if "loss" in k.lower():
                        best_metrics_all[k] = min(best_metrics_all[k], v)
                    else:
                        best_metrics_all[k] = max(best_metrics_all[k], v)

        # Final metrics from the last completed phase
        final_metrics: Dict[str, float] = {}
        if state.phases_completed:
            last_phase = max(state.phases_completed)
            final_metrics = state.phase_metrics.get(last_phase, {})

        # Total duration
        total_duration = state.total_duration()

        # Phase durations keyed by "phaseN"
        phase_duration_named = {
            f"phase{p}": d for p, d in state.phase_durations.items()
        }

        # Checkpoints saved (count boundary files)
        checkpoints_saved = 0
        ckpt_base = os.path.join(run_dir, "checkpoints")
        if os.path.exists(ckpt_base):
            for dirpath, dirnames, filenames in os.walk(ckpt_base):
                checkpoints_saved += sum(1 for f in filenames if f.endswith(".pt"))

        # Error info
        error_info = None
        if state.phases_failed:
            first_failed = min(state.phases_failed.keys())
            error_info = {
                "phase": first_failed,
                "message": state.phases_failed[first_failed],
            }

        self._report = {
            "run_id": state.run_id,
            "status": state.status,
            "phases_completed": sorted(state.phases_completed),
            "total_duration_seconds": round(total_duration, 2),
            "best_metrics": best_metrics_all,
            "final_metrics": final_metrics,
            "error": error_info,
            "phase_durations": phase_duration_named,
            "checkpoints_saved": checkpoints_saved,
        }

        # Write report
        reports_dir = os.path.join(run_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        report_path = os.path.join(reports_dir, "summary.json")
        with open(report_path, "w") as f:
            json.dump(self._report, f, indent=2)

        return report_path

    def print_summary(self) -> None:
        """Print formatted console output showing phase-by-phase results."""
        if not self._report:
            print("No report data available.  Call generate() first.")
            return

        r = self._report
        print()
        print("=" * 70)
        print(f"  Pipeline Report: {r['run_id']}")
        print("=" * 70)
        print(f"  Status         : {r['status']}")
        print(f"  Phases Complete: {r['phases_completed']}")
        print(f"  Total Duration : {r['total_duration_seconds']:.1f}s")
        print()

        if r["phase_durations"]:
            print("  Phase Durations:")
            for phase_name, dur in sorted(r["phase_durations"].items()):
                print(f"    {phase_name}: {dur:.2f}s")
            print()

        if r["best_metrics"]:
            print("  Best Metrics (across all phases):")
            for k, v in sorted(r["best_metrics"].items()):
                print(f"    {k}: {v:.4f}")
            print()

        if r["final_metrics"]:
            print("  Final Metrics (last completed phase):")
            for k, v in sorted(r["final_metrics"].items()):
                print(f"    {k}: {v:.4f}")
            print()

        if r["error"]:
            print(f"  Error (Phase {r['error']['phase']}): {r['error']['message']}")
            print()

        print(f"  Checkpoints Saved: {r['checkpoints_saved']}")
        print("=" * 70)

    @staticmethod
    def compare_reports(run_dirs: List[str]) -> Dict[str, Any]:
        """Side-by-side comparison of multiple pipeline runs.

        Parameters
        ----------
        run_dirs : list of str
            Paths to run directories.

        Returns
        -------
        dict
            Comparison data keyed by run_id.
        """
        comparison: Dict[str, Any] = {}
        for run_dir in run_dirs:
            report_path = os.path.join(run_dir, "reports", "summary.json")
            if not os.path.exists(report_path):
                continue
            with open(report_path, "r") as f:
                report = json.load(f)
            comparison[report.get("run_id", run_dir)] = {
                "status": report.get("status"),
                "phases_completed": report.get("phases_completed", []),
                "total_duration_seconds": report.get("total_duration_seconds", 0),
                "best_metrics": report.get("best_metrics", {}),
                "final_metrics": report.get("final_metrics", {}),
            }
        return comparison


# =========================================================================
# PipelineCLI
# =========================================================================

class PipelineCLI:
    """Argparse-based CLI for ``train_full_pipeline.py``."""

    @staticmethod
    def build_parser() -> argparse.ArgumentParser:
        """Build the argument parser for the full pipeline CLI.

        Returns
        -------
        argparse.ArgumentParser
        """
        parser = argparse.ArgumentParser(
            description="Brain-AI Multi-Phase Training Pipeline",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=(
                "Examples:\n"
                "  python train_full_pipeline.py --mode dev\n"
                "  python train_full_pipeline.py --mode production --use-amp\n"
                "  python train_full_pipeline.py --start-phase 4 --end-phase 7\n"
                "  python train_full_pipeline.py --resume-from 2026-02-20_full_abc1234\n"
            ),
        )
        parser.add_argument(
            "--mode",
            type=str,
            choices=["dev", "production"],
            default="dev",
            help="Training mode (default: dev)",
        )
        parser.add_argument(
            "--start-phase",
            type=int,
            default=1,
            help="First phase to run (1-7, default: 1)",
        )
        parser.add_argument(
            "--end-phase",
            type=int,
            default=7,
            help="Last phase to run (1-7, default: 7)",
        )
        parser.add_argument(
            "--resume-from",
            type=str,
            default=None,
            help="Resume from an existing run_id",
        )
        parser.add_argument(
            "--run-dir",
            type=str,
            default="runs/",
            help="Base directory for run outputs (default: runs/)",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=1337,
            help="Base random seed (default: 1337)",
        )
        parser.add_argument(
            "--use-amp",
            action="store_true",
            help="Enable automatic mixed precision",
        )
        parser.add_argument(
            "--config-overrides",
            type=str,
            default=None,
            help='JSON string of config overrides, e.g. \'{"workspace_dim": 2048}\'',
        )
        return parser

    @staticmethod
    def parse_config_overrides(overrides_str: Optional[str]) -> Dict[str, Any]:
        """Parse a JSON string of config overrides.

        Parameters
        ----------
        overrides_str : str or None
            JSON string.

        Returns
        -------
        dict
        """
        if not overrides_str:
            return {}
        try:
            return json.loads(overrides_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in --config-overrides: {e}")

    @classmethod
    def main(cls, args: Optional[List[str]] = None) -> PipelineState:
        """Parse arguments, create config, run pipeline.

        Parameters
        ----------
        args : list of str or None
            Command-line arguments.  If None, reads from sys.argv.

        Returns
        -------
        PipelineState
            Final pipeline state.
        """
        parser = cls.build_parser()
        parsed = parser.parse_args(args)

        # Build config
        config: Dict[str, Any] = {
            "mode": parsed.mode,
            "seed": parsed.seed,
            "use_amp": parsed.use_amp,
        }
        overrides = cls.parse_config_overrides(parsed.config_overrides)
        config.update(overrides)

        # Resume path
        if parsed.resume_from:
            resume_mgr = PipelineResumeManager()
            orchestrator = resume_mgr.resume(
                run_id=parsed.resume_from,
                base_dir=parsed.run_dir,
                config=config,
            )
        else:
            orchestrator = PipelineOrchestrator(
                config=config,
                mode=parsed.mode,
                run_dir=parsed.run_dir,
                start_phase=parsed.start_phase,
                end_phase=parsed.end_phase,
            )
            orchestrator.setup_run()

        # Handle KeyboardInterrupt gracefully
        try:
            state = orchestrator.run()
        except KeyboardInterrupt:
            orchestrator.state.status = "interrupted"
            orchestrator.state.end_time = time.time()
            orchestrator.state.save(orchestrator.run_dir)
            logger.warning("Pipeline interrupted by user. State saved.")
            state = orchestrator.state

        orchestrator.finalize()
        return state


# =========================================================================
# Self-test suite
# =========================================================================

def _banner(text: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}")


def _pass(name: str) -> None:
    print(f"  [PASS] {name}")


def _fail(name: str, detail: str = "") -> None:
    msg = f"  [FAIL] {name}"
    if detail:
        msg += f" -- {detail}"
    print(msg)


def _run_tests() -> None:
    """Execute the full self-test suite."""

    passed = 0
    failed = 0
    failure_details: List[str] = []

    def check(condition: bool, name: str, detail: str = "") -> None:
        nonlocal passed, failed
        if condition:
            _pass(name)
            passed += 1
        else:
            _fail(name, detail)
            failed += 1
            failure_details.append(name)

    # ==================================================================
    # 1. PipelineOrchestrator: Run phases 1-3 in dev mode (mock)
    # ==================================================================
    _banner("PipelineOrchestrator: phases 1-3 dev mode")

    with tempfile.TemporaryDirectory() as tmp:
        runner = MockPhaseRunner()
        orch = PipelineOrchestrator(
            config={"mode": "dev"},
            mode="dev",
            run_dir=tmp,
            start_phase=1,
            end_phase=3,
            phase_runner=runner,
        )
        orch.setup_run()
        state = orch.run()

        check(
            state.status == "completed",
            "Phases 1-3 complete successfully",
            f"status={state.status}",
        )
        check(
            state.phases_completed == [1, 2, 3],
            "Phases completed list is [1,2,3]",
            f"got {state.phases_completed}",
        )
        check(
            runner.phases_run == [1, 2, 3],
            "MockPhaseRunner ran phases 1,2,3",
            f"got {runner.phases_run}",
        )
        check(
            all(p in state.phase_durations for p in [1, 2, 3]),
            "All phase durations recorded",
        )
        check(
            all(p in state.phase_metrics for p in [1, 2, 3]),
            "All phase metrics recorded",
        )

    # ==================================================================
    # 2. Phase transition validation is called between phases
    # ==================================================================
    _banner("Phase transition validation")

    with tempfile.TemporaryDirectory() as tmp:
        validator = MockBoundaryValidator()
        runner = MockPhaseRunner()
        orch = PipelineOrchestrator(
            config={"mode": "dev"},
            mode="dev",
            run_dir=tmp,
            start_phase=1,
            end_phase=3,
            phase_runner=runner,
            boundary_validator=validator,
        )
        orch.setup_run()
        state = orch.run()

        # Validation should be called for phase 1->2 and 2->3.
        check(
            (1, 2) in validator.validations,
            "Boundary validation called for Phase 1->2",
            f"validations={validator.validations}",
        )
        check(
            (2, 3) in validator.validations,
            "Boundary validation called for Phase 2->3",
            f"validations={validator.validations}",
        )
        check(
            state.status == "completed",
            "Pipeline completed with validation",
        )

    # ==================================================================
    # 3. PipelineState save/load round-trip
    # ==================================================================
    _banner("PipelineState save/load")

    with tempfile.TemporaryDirectory() as tmp:
        state = PipelineState(run_id="test_run_001", start_phase=1, end_phase=7)
        state.phases_completed = [1, 2, 3]
        state.phases_failed = {4: "OOM error"}
        state.phase_durations = {1: 10.5, 2: 20.3, 3: 15.1}
        state.phase_metrics = {
            1: {"val_loss": 0.9, "val_acc": 0.55},
            2: {"val_loss": 0.7, "val_acc": 0.65},
            3: {"val_loss": 0.5, "val_acc": 0.75},
        }
        state.status = "failed"
        state.current_phase = None
        state.start_time = 1000.0
        state.end_time = 1100.0

        state.save(tmp)

        loaded = PipelineState.load(tmp)
        check(
            loaded.run_id == "test_run_001",
            "State round-trip: run_id preserved",
        )
        check(
            loaded.phases_completed == [1, 2, 3],
            "State round-trip: phases_completed preserved",
        )
        check(
            loaded.phases_failed == {4: "OOM error"},
            "State round-trip: phases_failed preserved",
        )
        check(
            loaded.phase_durations == {1: 10.5, 2: 20.3, 3: 15.1},
            "State round-trip: phase_durations preserved",
        )
        check(
            loaded.phase_metrics[2]["val_loss"] == 0.7,
            "State round-trip: phase_metrics preserved",
        )
        check(
            loaded.status == "failed",
            "State round-trip: status preserved",
        )
        check(
            loaded.start_time == 1000.0,
            "State round-trip: start_time preserved",
        )
        check(
            loaded.end_time == 1100.0,
            "State round-trip: end_time preserved",
        )

    # ==================================================================
    # 4. Resume from phase 4 (phases 1-3 already completed)
    # ==================================================================
    _banner("Resume from phase 4")

    with tempfile.TemporaryDirectory() as tmp:
        # Set up a "previous" run that completed phases 1-3 and was interrupted
        run_id = "prev_run_resume_test"
        run_dir = os.path.join(tmp, run_id)
        os.makedirs(run_dir, exist_ok=True)

        prev_state = PipelineState(run_id=run_id, start_phase=1, end_phase=7)
        prev_state.phases_completed = [1, 2, 3]
        prev_state.status = "interrupted"
        prev_state.phase_durations = {1: 5.0, 2: 6.0, 3: 7.0}
        prev_state.phase_metrics = {
            1: {"val_loss": 0.9}, 2: {"val_loss": 0.8}, 3: {"val_loss": 0.7}
        }
        prev_state.save(run_dir)

        # Create boundary artifact for phase 3
        _create_mock_boundary(run_dir, 3)

        # Create manifest
        manifest = {"config": {"mode": "dev"}, "run_id": run_id}
        with open(os.path.join(run_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f)

        # Resume
        resume_mgr = PipelineResumeManager()
        check(
            resume_mgr.can_resume(run_id, tmp),
            "can_resume returns True for interrupted run",
        )
        check(
            resume_mgr.find_resume_point(run_dir) == 4,
            "find_resume_point returns 4",
        )

        orch = resume_mgr.resume(
            run_id=run_id,
            base_dir=tmp,
            phase_runner=MockPhaseRunner(),
            boundary_validator=MockBoundaryValidator(),
        )
        state = orch.run()

        check(
            state.status == "completed",
            "Resumed pipeline completed",
            f"status={state.status}",
        )
        check(
            4 in state.phases_completed and 5 in state.phases_completed,
            "Phases 4+ ran after resume",
            f"completed={state.phases_completed}",
        )

    # ==================================================================
    # 5. Fork from existing run
    # ==================================================================
    _banner("PipelineForkManager")

    with tempfile.TemporaryDirectory() as tmp:
        # Create source run with phase 3 boundary
        source_id = "source_run_fork_test"
        source_dir = os.path.join(tmp, source_id)
        os.makedirs(source_dir, exist_ok=True)
        _create_mock_boundary(source_dir, 3)

        source_manifest = {"run_id": source_id, "config": {"mode": "dev"}}
        with open(os.path.join(source_dir, "manifest.json"), "w") as f:
            json.dump(source_manifest, f)

        fork_mgr = PipelineForkManager()
        new_run_id = fork_mgr.fork(
            source_run_id=source_id,
            fork_phase=3,
            new_config={"mode": "dev", "workspace_dim": 2048},
            base_dir=tmp,
        )

        check(
            new_run_id != source_id,
            "Fork creates new run_id",
        )
        new_run_dir = os.path.join(tmp, new_run_id)
        check(
            os.path.exists(new_run_dir),
            "Fork run directory exists",
        )

        # Boundary artifact copied
        forked_boundary = os.path.join(
            new_run_dir, "checkpoints", "phase3", "phase_boundary.pt"
        )
        check(
            os.path.exists(forked_boundary),
            "Forked boundary artifact exists",
        )

        # Manifest records provenance
        with open(os.path.join(new_run_dir, "manifest.json"), "r") as f:
            fork_manifest = json.load(f)
        check(
            fork_manifest.get("forked_from", {}).get("source_run_id") == source_id,
            "Fork manifest records source_run_id",
        )
        check(
            fork_manifest.get("forked_from", {}).get("fork_phase") == 3,
            "Fork manifest records fork_phase",
        )

    # ==================================================================
    # 6. Failure handling (phase 3 fails -> recorded, pipeline stops)
    # ==================================================================
    _banner("Phase failure handling")

    with tempfile.TemporaryDirectory() as tmp:
        runner = MockPhaseRunner(fail_on_phase=3)
        orch = PipelineOrchestrator(
            config={"mode": "dev"},
            mode="dev",
            run_dir=tmp,
            start_phase=1,
            end_phase=5,
            phase_runner=runner,
        )
        orch.setup_run()
        state = orch.run()

        check(
            state.status == "failed",
            "Pipeline status is 'failed' after phase failure",
        )
        check(
            3 in state.phases_failed,
            "Phase 3 recorded in phases_failed",
        )
        check(
            "Simulated failure" in state.phases_failed.get(3, ""),
            "Error message captured for phase 3",
        )
        check(
            state.phases_completed == [1, 2],
            "Only phases 1-2 completed before failure",
            f"got {state.phases_completed}",
        )
        check(
            4 not in state.phases_completed,
            "Phase 4 was not run after phase 3 failure",
        )

    # ==================================================================
    # 7. PipelineReport generation
    # ==================================================================
    _banner("PipelineReport")

    with tempfile.TemporaryDirectory() as tmp:
        runner = MockPhaseRunner()
        orch = PipelineOrchestrator(
            config={"mode": "dev"},
            mode="dev",
            run_dir=tmp,
            start_phase=1,
            end_phase=3,
            phase_runner=runner,
        )
        orch.setup_run()
        state = orch.run()

        report = PipelineReport()
        report_path = report.generate(orch.run_dir, state)

        check(
            os.path.exists(report_path),
            "Report summary.json created",
        )

        with open(report_path, "r") as f:
            rdata = json.load(f)

        check(
            rdata["status"] == "completed",
            "Report status is 'completed'",
        )
        check(
            rdata["phases_completed"] == [1, 2, 3],
            "Report phases_completed correct",
        )
        check(
            "best_metrics" in rdata and "val_loss" in rdata["best_metrics"],
            "Report contains best_metrics",
        )
        check(
            rdata["total_duration_seconds"] >= 0,
            "Report total_duration is non-negative",
        )
        check(
            "phase1" in rdata["phase_durations"],
            "Report contains phase-specific durations",
        )

        # Test print_summary runs without error
        report.print_summary()
        check(True, "print_summary executed without error")

    # ==================================================================
    # 8. PipelineCLI argument parsing
    # ==================================================================
    _banner("PipelineCLI argument parsing")

    parser = PipelineCLI.build_parser()

    # Default args
    args = parser.parse_args([])
    check(args.mode == "dev", "CLI default mode is 'dev'")
    check(args.start_phase == 1, "CLI default start_phase is 1")
    check(args.end_phase == 7, "CLI default end_phase is 7")
    check(args.seed == 1337, "CLI default seed is 1337")
    check(args.use_amp is False, "CLI default use_amp is False")
    check(args.resume_from is None, "CLI default resume_from is None")

    # Custom args
    args = parser.parse_args([
        "--mode", "production",
        "--start-phase", "4",
        "--end-phase", "6",
        "--seed", "42",
        "--use-amp",
        "--config-overrides", '{"workspace_dim": 2048}',
    ])
    check(args.mode == "production", "CLI parses production mode")
    check(args.start_phase == 4, "CLI parses start_phase=4")
    check(args.end_phase == 6, "CLI parses end_phase=6")
    check(args.seed == 42, "CLI parses seed=42")
    check(args.use_amp is True, "CLI parses use_amp=True")

    overrides = PipelineCLI.parse_config_overrides(args.config_overrides)
    check(
        overrides.get("workspace_dim") == 2048,
        "CLI parses config_overrides JSON",
    )

    # Invalid JSON
    raised = False
    try:
        PipelineCLI.parse_config_overrides("{bad json}")
    except ValueError:
        raised = True
    check(raised, "CLI rejects invalid config_overrides JSON")

    # ==================================================================
    # 9. KeyboardInterrupt handling (state saved)
    # ==================================================================
    _banner("KeyboardInterrupt handling")

    with tempfile.TemporaryDirectory() as tmp:
        runner = MockPhaseRunner(interrupt_on_phase=2)
        orch = PipelineOrchestrator(
            config={"mode": "dev"},
            mode="dev",
            run_dir=tmp,
            start_phase=1,
            end_phase=5,
            phase_runner=runner,
        )
        orch.setup_run()
        state = orch.run()

        check(
            state.status == "interrupted",
            "Pipeline status is 'interrupted' after KeyboardInterrupt",
        )
        check(
            state.phases_completed == [1],
            "Only phase 1 completed before interrupt",
            f"got {state.phases_completed}",
        )

        # State was persisted
        loaded = PipelineState.load(orch.run_dir)
        check(
            loaded.status == "interrupted",
            "Interrupted state was persisted to disk",
        )
        check(
            loaded.phases_completed == [1],
            "Persisted state has correct phases_completed",
        )

    # ==================================================================
    # 10. start_phase > 1 validates boundary exists
    # ==================================================================
    _banner("start_phase > 1 boundary validation")

    with tempfile.TemporaryDirectory() as tmp:
        # No boundary artifact exists for phase 2
        runner = MockPhaseRunner()
        orch = PipelineOrchestrator(
            config={"mode": "dev"},
            mode="dev",
            run_dir=tmp,
            start_phase=3,
            end_phase=5,
            phase_runner=runner,
        )
        orch.setup_run()
        state = orch.run()

        check(
            state.status == "failed",
            "Pipeline fails when start_phase > 1 and no boundary exists",
        )
        check(
            3 in state.phases_failed,
            "Phase 3 recorded as failed due to missing boundary",
        )

    # start_phase > 1 WITH boundary present
    with tempfile.TemporaryDirectory() as tmp:
        runner = MockPhaseRunner()
        orch = PipelineOrchestrator(
            config={"mode": "dev"},
            mode="dev",
            run_dir=tmp,
            start_phase=3,
            end_phase=4,
            phase_runner=runner,
        )
        orch.setup_run()
        # Create the required boundary artifact
        _create_mock_boundary(orch.run_dir, 2)

        state = orch.run()
        check(
            state.status == "completed",
            "Pipeline succeeds when boundary exists for start_phase > 1",
        )
        check(
            state.phases_completed == [3, 4],
            "Phases 3-4 completed when starting from phase 3",
            f"got {state.phases_completed}",
        )

    # ==================================================================
    # 11. Full pipeline mock (phases 1-7, dev mode)
    # ==================================================================
    _banner("Full pipeline mock (phases 1-7)")

    with tempfile.TemporaryDirectory() as tmp:
        runner = MockPhaseRunner()
        orch = PipelineOrchestrator(
            config={"mode": "dev"},
            mode="dev",
            run_dir=tmp,
            start_phase=1,
            end_phase=7,
            phase_runner=runner,
        )
        orch.setup_run()
        state = orch.run()

        check(
            state.status == "completed",
            "Full 7-phase pipeline completed",
        )
        check(
            state.phases_completed == [1, 2, 3, 4, 5, 6, 7],
            "All 7 phases completed",
        )
        check(
            len(state.phase_durations) == 7,
            "All 7 phase durations recorded",
        )
        check(
            len(state.phase_metrics) == 7,
            "All 7 phase metrics recorded",
        )
        check(
            state.total_duration() > 0,
            "Total duration is positive",
        )

        # Verify boundary artifacts were created
        for p in range(1, 8):
            boundary = os.path.join(
                orch.run_dir, "checkpoints", f"phase{p}", "phase_boundary.pt"
            )
            check(
                os.path.exists(boundary),
                f"Phase {p} boundary artifact exists",
            )

    # ==================================================================
    # 12. PipelineHealthCheck: disk space, dependencies
    # ==================================================================
    _banner("PipelineHealthCheck")

    with tempfile.TemporaryDirectory() as tmp:
        hc = PipelineHealthCheck()

        # Disk space check with reasonable threshold
        disk_ok = hc.check_disk_space(tmp, required_gb=0.001)
        check(
            disk_ok,
            "Disk space check passes with tiny threshold",
        )

        # Disk space check with absurd threshold
        disk_low = hc.check_disk_space(tmp, required_gb=999999)
        check(
            not disk_low,
            "Disk space check fails with huge threshold",
        )

        # GPU memory (gracefully skipped if no GPU)
        gpu_ok = hc.check_gpu_memory()
        check(
            isinstance(gpu_ok, bool),
            "GPU memory check returns bool",
        )

        # Dependencies check
        hc2 = PipelineHealthCheck()
        deps_ok = hc2.check_dependencies(1)
        check(
            deps_ok,
            "Phase 1 has no required deps (always ok)",
        )

        # run_all
        hc3 = PipelineHealthCheck()
        all_ok = hc3.run_all(1, tmp, required_gb=0.001)
        check(
            all_ok,
            "run_all passes for phase 1 with small disk threshold",
        )

    # ==================================================================
    # 13. PipelineForkManager: list_forks
    # ==================================================================
    _banner("PipelineForkManager: list_forks")

    with tempfile.TemporaryDirectory() as tmp:
        source_id = "source_run_list_forks"
        source_dir = os.path.join(tmp, source_id)
        os.makedirs(source_dir, exist_ok=True)
        _create_mock_boundary(source_dir, 3)
        with open(os.path.join(source_dir, "manifest.json"), "w") as f:
            json.dump({"run_id": source_id}, f)

        fork_mgr = PipelineForkManager()

        # Create two forks
        fork1_id = fork_mgr.fork(source_id, 3, {"variant": "A"}, base_dir=tmp)
        fork2_id = fork_mgr.fork(source_id, 3, {"variant": "B"}, base_dir=tmp)

        forks = fork_mgr.list_forks(source_id, base_dir=tmp)
        check(
            len(forks) == 2,
            "list_forks finds 2 forked runs",
            f"got {len(forks)}",
        )
        check(
            fork1_id in forks and fork2_id in forks,
            "list_forks returns correct fork IDs",
        )

        # No forks for a different source
        check(
            fork_mgr.list_forks("nonexistent_run", base_dir=tmp) == [],
            "list_forks returns empty for unknown source",
        )

    # ==================================================================
    # 14. PipelineState: is_resumable and next_phase
    # ==================================================================
    _banner("PipelineState: is_resumable / next_phase")

    state = PipelineState(run_id="test", start_phase=1, end_phase=7)
    check(
        not state.is_resumable(),
        "Empty state is not resumable",
    )
    check(
        state.next_phase() == 1,
        "Empty state next_phase is start_phase (1)",
    )

    state.phases_completed = [1, 2, 3]
    state.status = "interrupted"
    check(
        state.is_resumable(),
        "Interrupted state with completed phases is resumable",
    )
    check(
        state.next_phase() == 4,
        "next_phase after [1,2,3] is 4",
    )

    state.phases_completed = [1, 2, 3, 4, 5, 6, 7]
    state.status = "completed"
    check(
        not state.is_resumable(),
        "Completed state is not resumable",
    )
    check(
        state.next_phase() is None,
        "next_phase is None when all phases done",
    )

    # ==================================================================
    # 15. PhaseTransitionLogger
    # ==================================================================
    _banner("PhaseTransitionLogger")

    with tempfile.TemporaryDirectory() as tmp:
        jsonl_path = os.path.join(tmp, "events.jsonl")
        tl = PhaseTransitionLogger(jsonl_path=jsonl_path)

        tl.log_pipeline_start("test_run", 1, 7)
        tl.log_transition(1, 2, {"val_loss": 0.9}, 10.5)
        tl.log_transition(2, 3, {"val_loss": 0.7}, 15.2)
        tl.log_pipeline_end("test_run", "completed", 25.7)

        check(
            os.path.exists(jsonl_path),
            "JSONL event file created",
        )

        with open(jsonl_path, "r") as f:
            lines = f.readlines()
        check(
            len(lines) == 4,
            "JSONL has 4 events (start, 2 transitions, end)",
            f"got {len(lines)}",
        )

        event = json.loads(lines[0])
        check(
            event["event"] == "pipeline_start",
            "First event is pipeline_start",
        )

        event = json.loads(lines[1])
        check(
            event["event"] == "phase_transition" and event["from_phase"] == 1,
            "Second event is phase_transition from 1",
        )

        event = json.loads(lines[3])
        check(
            event["event"] == "pipeline_end" and event["status"] == "completed",
            "Last event is pipeline_end with completed status",
        )

    # ==================================================================
    # 16. PipelineReport: compare_reports
    # ==================================================================
    _banner("PipelineReport: compare_reports")

    with tempfile.TemporaryDirectory() as tmp:
        # Create two runs with reports
        for i, run_name in enumerate(["run_a", "run_b"]):
            rd = os.path.join(tmp, run_name)
            os.makedirs(os.path.join(rd, "reports"), exist_ok=True)
            report = {
                "run_id": run_name,
                "status": "completed",
                "phases_completed": [1, 2, 3],
                "total_duration_seconds": 100.0 + i * 50,
                "best_metrics": {"val_loss": 0.5 - i * 0.1},
                "final_metrics": {"val_loss": 0.6 - i * 0.1},
            }
            with open(os.path.join(rd, "reports", "summary.json"), "w") as f:
                json.dump(report, f)

        comparison = PipelineReport.compare_reports([
            os.path.join(tmp, "run_a"),
            os.path.join(tmp, "run_b"),
        ])

        check(
            "run_a" in comparison and "run_b" in comparison,
            "compare_reports has entries for both runs",
        )
        check(
            comparison["run_a"]["total_duration_seconds"] == 100.0,
            "compare_reports: run_a duration correct",
        )
        check(
            comparison["run_b"]["total_duration_seconds"] == 150.0,
            "compare_reports: run_b duration correct",
        )

    # ==================================================================
    # 17. PipelineCLI main (dev mode, mock)
    # ==================================================================
    _banner("PipelineCLI main (mock run)")

    with tempfile.TemporaryDirectory() as tmp:
        state = PipelineCLI.main([
            "--mode", "dev",
            "--start-phase", "1",
            "--end-phase", "2",
            "--run-dir", tmp,
            "--seed", "42",
        ])
        check(
            state.status == "completed",
            "CLI main runs pipeline to completion",
        )
        check(
            state.phases_completed == [1, 2],
            "CLI main: phases 1-2 completed",
            f"got {state.phases_completed}",
        )

    # ==================================================================
    # 18. Orchestrator invalid arguments
    # ==================================================================
    _banner("Orchestrator argument validation")

    raised = False
    try:
        PipelineOrchestrator(config={}, start_phase=0)
    except ValueError:
        raised = True
    check(raised, "start_phase=0 raises ValueError")

    raised = False
    try:
        PipelineOrchestrator(config={}, start_phase=8)
    except ValueError:
        raised = True
    check(raised, "start_phase=8 raises ValueError")

    raised = False
    try:
        PipelineOrchestrator(config={}, start_phase=5, end_phase=3)
    except ValueError:
        raised = True
    check(raised, "start_phase > end_phase raises ValueError")

    # ==================================================================
    # 19. PipelineResumeManager: validate_resume config check
    # ==================================================================
    _banner("PipelineResumeManager: validate_resume")

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = os.path.join(tmp, "validation_run")
        os.makedirs(run_dir, exist_ok=True)
        manifest = {"config": {"mode": "dev", "workspace_dim": 4096}}
        with open(os.path.join(run_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f)

        mgr = PipelineResumeManager()

        # Compatible config
        compat, warns = mgr.validate_resume(
            run_dir, {"mode": "dev", "workspace_dim": 4096}
        )
        check(
            compat and len(warns) == 0,
            "validate_resume: compatible config produces no warnings",
        )

        # Incompatible config
        compat2, warns2 = mgr.validate_resume(
            run_dir, {"mode": "production", "workspace_dim": 2048}
        )
        check(
            not compat2 and len(warns2) > 0,
            "validate_resume: incompatible config produces warnings",
            f"compat={compat2}, warns={warns2}",
        )

    # ==================================================================
    # 20. PipelineResumeManager: can_resume edge cases
    # ==================================================================
    _banner("PipelineResumeManager: can_resume edge cases")

    with tempfile.TemporaryDirectory() as tmp:
        mgr = PipelineResumeManager()

        # Non-existent run
        check(
            not mgr.can_resume("nonexistent", tmp),
            "can_resume returns False for non-existent run",
        )

        # Run with no state file
        no_state_dir = os.path.join(tmp, "no_state")
        os.makedirs(no_state_dir, exist_ok=True)
        check(
            not mgr.can_resume("no_state", tmp),
            "can_resume returns False when no pipeline_state.json",
        )

        # Run that completed successfully (not resumable)
        completed_id = "completed_run"
        completed_dir = os.path.join(tmp, completed_id)
        os.makedirs(completed_dir, exist_ok=True)
        s = PipelineState(run_id=completed_id, start_phase=1, end_phase=3)
        s.phases_completed = [1, 2, 3]
        s.status = "completed"
        s.save(completed_dir)
        check(
            not mgr.can_resume(completed_id, tmp),
            "can_resume returns False for completed run",
        )

    # ==================================================================
    # 21. Manifest append-only on resume
    # ==================================================================
    _banner("Manifest append-only on resume")

    with tempfile.TemporaryDirectory() as tmp:
        run_id = "append_only_test"
        run_dir = os.path.join(tmp, run_id)
        os.makedirs(run_dir, exist_ok=True)

        # Set up interrupted run
        st = PipelineState(run_id=run_id, start_phase=1, end_phase=5)
        st.phases_completed = [1, 2]
        st.status = "interrupted"
        st.phase_durations = {1: 5.0, 2: 6.0}
        st.phase_metrics = {1: {"val_loss": 0.9}, 2: {"val_loss": 0.8}}
        st.save(run_dir)
        _create_mock_boundary(run_dir, 2)

        manifest_data = {"config": {"mode": "dev"}, "run_id": run_id}
        with open(os.path.join(run_dir, "manifest.json"), "w") as f:
            json.dump(manifest_data, f)

        mgr = PipelineResumeManager()
        orch = mgr.resume(
            run_id, base_dir=tmp,
            phase_runner=MockPhaseRunner(),
            boundary_validator=MockBoundaryValidator(),
        )

        # Check that resume_events were appended
        with open(os.path.join(run_dir, "manifest.json"), "r") as f:
            updated_manifest = json.load(f)
        check(
            "resume_events" in updated_manifest,
            "Resume appends resume_events to manifest",
        )
        check(
            len(updated_manifest["resume_events"]) == 1,
            "One resume event recorded",
        )
        check(
            updated_manifest["resume_events"][0]["resume_phase"] == 3,
            "Resume event records correct phase",
        )

    # ==================================================================
    # 22. Run directory structure verification
    # ==================================================================
    _banner("Run directory structure")

    with tempfile.TemporaryDirectory() as tmp:
        orch = PipelineOrchestrator(
            config={"mode": "dev"},
            run_dir=tmp,
            start_phase=2,
            end_phase=5,
        )
        run_dir = orch.setup_run()

        check(
            os.path.exists(run_dir),
            "Run directory created",
        )
        check(
            os.path.exists(os.path.join(run_dir, "manifest.json")),
            "manifest.json created",
        )
        check(
            os.path.exists(os.path.join(run_dir, "logs", "tensorboard")),
            "logs/tensorboard created",
        )
        check(
            os.path.exists(os.path.join(run_dir, "artifacts")),
            "artifacts/ created",
        )
        check(
            os.path.exists(os.path.join(run_dir, "reports")),
            "reports/ created",
        )
        for p in range(2, 6):
            check(
                os.path.exists(
                    os.path.join(run_dir, "checkpoints", f"phase{p}")
                ),
                f"checkpoints/phase{p}/ created",
            )

    # ==================================================================
    # 23. PipelineState total_duration
    # ==================================================================
    _banner("PipelineState total_duration")

    st = PipelineState(run_id="dur_test")
    st.phase_durations = {1: 10.5, 2: 20.3, 3: 15.1}
    check(
        abs(st.total_duration() - 45.9) < 0.01,
        "total_duration sums correctly",
        f"got {st.total_duration()}",
    )

    # ==================================================================
    # 24. PipelineOrchestrator finalize
    # ==================================================================
    _banner("PipelineOrchestrator finalize")

    with tempfile.TemporaryDirectory() as tmp:
        orch = PipelineOrchestrator(
            config={"mode": "dev"},
            run_dir=tmp,
            start_phase=1,
            end_phase=2,
        )
        orch.setup_run()
        orch.run()
        report_path = orch.finalize()

        check(
            os.path.exists(report_path),
            "finalize creates summary report",
        )
        check(
            report_path.endswith("summary.json"),
            "finalize returns summary.json path",
        )

    # ==================================================================
    # 25. PipelineHealthCheck strict mode
    # ==================================================================
    _banner("PipelineHealthCheck strict mode")

    with tempfile.TemporaryDirectory() as tmp:
        hc_strict = PipelineHealthCheck(strict=True)
        # With absurd disk requirement, strict should fail
        all_ok = hc_strict.run_all(1, tmp, required_gb=999999)
        check(
            not all_ok,
            "Strict health check fails on low disk",
        )
        check(
            len(hc_strict.errors) > 0,
            "Strict mode promotes warnings to errors",
        )

    # ==================================================================
    # 26. Fork then run forked pipeline
    # ==================================================================
    _banner("Fork then run forked pipeline")

    with tempfile.TemporaryDirectory() as tmp:
        # Source run
        source_id = "fork_and_run_source"
        source_dir = os.path.join(tmp, source_id)
        os.makedirs(source_dir, exist_ok=True)
        _create_mock_boundary(source_dir, 2)
        with open(os.path.join(source_dir, "manifest.json"), "w") as f:
            json.dump({"run_id": source_id}, f)

        # Fork at phase 2
        fork_mgr = PipelineForkManager()
        fork_id = fork_mgr.fork(source_id, 2, {"mode": "dev"}, base_dir=tmp)
        fork_dir = os.path.join(tmp, fork_id)

        # Initialize pipeline state for fork
        fork_state = PipelineState(run_id=fork_id, start_phase=3, end_phase=5)
        fork_state.save(fork_dir)

        # Run forked pipeline
        orch = PipelineOrchestrator(
            config={"mode": "dev"},
            mode="dev",
            run_dir=tmp,
            start_phase=3,
            end_phase=5,
            phase_runner=MockPhaseRunner(),
            boundary_validator=MockBoundaryValidator(),
            run_id=fork_id,
        )
        orch.run_dir = fork_dir
        orch.state = fork_state
        state = orch.run()

        check(
            state.status == "completed",
            "Forked pipeline completes successfully",
        )
        check(
            state.phases_completed == [3, 4, 5],
            "Forked pipeline runs phases 3-5",
            f"got {state.phases_completed}",
        )

    # ==================================================================
    # 27. PHASE_NAMES constant completeness
    # ==================================================================
    _banner("Constants")

    check(
        len(PHASE_NAMES) == 7,
        "PHASE_NAMES has 7 entries",
    )
    for p in range(1, 8):
        check(
            p in PHASE_NAMES,
            f"PHASE_NAMES has entry for phase {p}",
        )

    # ==================================================================
    # Summary
    # ==================================================================
    _banner("SUMMARY")

    total = passed + failed
    print(f"\n  Total : {total}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")

    if failure_details:
        print("\n  Failed tests:")
        for name in failure_details:
            print(f"    - {name}")

    if failed > 0:
        print(f"\n  EXIT CODE: 1 ({failed} failure(s))")
        sys.exit(1)
    else:
        print("\n  All tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    _run_tests()
