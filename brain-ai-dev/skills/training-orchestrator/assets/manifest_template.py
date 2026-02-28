#!/usr/bin/env python3
"""
manifest_template.py  --  RunManifest system for brain_ai training provenance.

This module is the single source of truth for capturing, serializing, validating,
comparing, and reproducing training run provenance across the seven-phase brain_ai
cognitive pipeline.

Classes:
    ManifestSchema       -- Schema version and validation rules
    GitProvenanceCapture -- Git commit, branch, dirty state, patch capture
    EnvironmentCapture   -- Python, torch, CUDA, hardware, pip freeze
    RunIDGenerator       -- Deterministic run ID generation
    RunDirectory         -- Run directory tree creation and path properties
    RunManifest          -- Core manifest capture, save, load, resume, finalize
    ManifestComparator   -- Compare manifests for reproduction equivalence

Self-contained: no brain_ai imports required.  Uses inline mocks for self-test.
"""

from __future__ import annotations

import copy
import datetime
import getpass
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# ManifestSchema
# ---------------------------------------------------------------------------

class ManifestSchema:
    """Schema version tracking and structural validation for manifest dicts."""

    SCHEMA_VERSION: str = "1.0"

    # Top-level sections that must exist in every manifest
    REQUIRED_SECTIONS: List[str] = [
        "schema_version",
        "identity",
        "git",
        "config",
        "seeds",
        "env",
        "data",
        "logging",
        "resume",
        "checkpoints",
        "results",
    ]

    # Fields within each section that are required
    REQUIRED_FIELDS: Dict[str, List[str]] = {
        "identity": [
            "run_id", "phase", "mode", "timestamp_start",
            "timestamp_end", "hostname", "user",
        ],
        "git": [
            "commit", "branch", "remote_url", "dirty",
            "patch_path", "entrypoint", "cli_args",
        ],
        "config": ["brain_ai", "overrides", "resolved", "feature_flags"],
        "seeds": [
            "base_seed", "per_phase_offsets", "torch_deterministic",
            "cudnn_benchmark", "cudnn_deterministic",
            "use_deterministic_algorithms",
        ],
        "env": [
            "python_version", "os", "cuda_version",
            "torch_version", "pip_freeze_path", "hardware",
        ],
        "data": ["datasets"],
        "logging": ["tensorboard", "wandb"],
        "resume": ["enabled", "from_run_id", "from_checkpoint"],
        "checkpoints": [
            "save_every_n_steps", "best_metric_key",
            "phase_boundary_produced",
        ],
        "results": ["status", "best_metrics", "final_metrics", "error"],
    }

    OPTIONAL_FIELDS: Dict[str, List[str]] = {
        "identity": ["ablation_id"],
        "resume": ["phase_boundary_artifacts", "resume_events"],
    }

    FEATURE_FLAG_KEYS: List[str] = [
        "use_snn", "use_htm", "use_workspace",
        "use_symbolic", "use_meta", "use_engram",
    ]

    VALID_STATUSES: List[str] = ["running", "completed", "failed", "interrupted"]
    VALID_MODES: List[str] = ["dev", "production"]

    @classmethod
    def validate_schema(cls, manifest: Dict[str, Any]) -> List[str]:
        """Validate a manifest dict against the schema.

        Returns a list of error strings.  Empty list means valid.
        Raises ValueError if there are critical structural issues.
        """
        errors: List[str] = []

        # -- top-level sections --
        for section in cls.REQUIRED_SECTIONS:
            if section not in manifest:
                errors.append(f"Missing required top-level section: '{section}'")

        if errors:
            raise ValueError(
                "Manifest schema validation failed:\n  " + "\n  ".join(errors)
            )

        # -- per-section required fields --
        for section, fields in cls.REQUIRED_FIELDS.items():
            if section not in manifest:
                continue
            sect = manifest[section]
            if not isinstance(sect, dict):
                errors.append(f"Section '{section}' must be a dict, got {type(sect).__name__}")
                continue
            for fld in fields:
                if fld not in sect:
                    errors.append(f"Missing required field '{section}.{fld}'")

        # -- type checks --
        identity = manifest.get("identity", {})
        if "phase" in identity:
            phase = identity["phase"]
            if not isinstance(phase, int) or not (1 <= phase <= 7):
                errors.append(f"identity.phase must be int in [1,7], got {phase!r}")
        if "mode" in identity:
            mode = identity["mode"]
            if mode not in cls.VALID_MODES:
                errors.append(f"identity.mode must be one of {cls.VALID_MODES}, got {mode!r}")

        # -- seeds type checks --
        seeds = manifest.get("seeds", {})
        if "per_phase_offsets" in seeds:
            offsets = seeds["per_phase_offsets"]
            if not isinstance(offsets, list) or len(offsets) != 7:
                errors.append(
                    f"seeds.per_phase_offsets must be a list of 7 ints, got length {len(offsets) if isinstance(offsets, list) else type(offsets).__name__}"
                )

        # -- feature flags --
        ff = manifest.get("config", {}).get("feature_flags", {})
        if isinstance(ff, dict):
            for key in cls.FEATURE_FLAG_KEYS:
                if key not in ff:
                    errors.append(f"Missing feature flag: config.feature_flags.{key}")
                elif not isinstance(ff[key], bool):
                    errors.append(f"config.feature_flags.{key} must be bool")

        # -- results status --
        results = manifest.get("results", {})
        if "status" in results:
            if results["status"] not in cls.VALID_STATUSES:
                errors.append(
                    f"results.status must be one of {cls.VALID_STATUSES}, got {results['status']!r}"
                )

        if errors:
            raise ValueError(
                "Manifest schema validation failed:\n  " + "\n  ".join(errors)
            )

        return errors  # empty


# ---------------------------------------------------------------------------
# GitProvenanceCapture
# ---------------------------------------------------------------------------

class GitProvenanceCapture:
    """Capture git repository state for provenance tracking."""

    @staticmethod
    def _run_git(args: List[str], cwd: str) -> Optional[str]:
        """Run a git command, returning stdout or None on failure."""
        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return result.stdout.strip()
            return None
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return None

    @classmethod
    def capture_git_info(cls, repo_dir: str) -> Dict[str, Any]:
        """Capture current git state.

        Returns dict with keys: commit, branch, remote_url, dirty, short_sha.
        Falls back gracefully if git is unavailable or not a repo.
        """
        info: Dict[str, Any] = {
            "commit": "0" * 40,
            "branch": "unknown",
            "remote_url": "unknown",
            "dirty": False,
            "short_sha": "0000000",
        }

        commit = cls._run_git(["rev-parse", "HEAD"], repo_dir)
        if commit is None:
            # Not a git repo or git not installed
            return info

        info["commit"] = commit

        short_sha = cls._run_git(["rev-parse", "--short=7", "HEAD"], repo_dir)
        if short_sha:
            info["short_sha"] = short_sha

        branch = cls._run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_dir)
        if branch:
            info["branch"] = branch

        remote_url = cls._run_git(["remote", "get-url", "origin"], repo_dir)
        if remote_url:
            info["remote_url"] = remote_url

        # Check dirty state
        status = cls._run_git(["status", "--porcelain"], repo_dir)
        info["dirty"] = bool(status)

        return info

    @classmethod
    def save_git_diff(cls, repo_dir: str, output_path: str) -> bool:
        """Save ``git diff HEAD`` to a patch file.

        Returns True if the diff was saved, False if nothing to save or error.
        """
        diff = cls._run_git(["diff", "HEAD"], repo_dir)
        if diff is None or diff == "":
            return False
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(diff)
            return True
        except OSError:
            return False


# ---------------------------------------------------------------------------
# EnvironmentCapture
# ---------------------------------------------------------------------------

class EnvironmentCapture:
    """Capture software environment and hardware details."""

    @staticmethod
    def capture_environment() -> Dict[str, Any]:
        """Return dict with python version, OS, CUDA version, torch version."""
        env: Dict[str, Any] = {
            "python_version": platform.python_version(),
            "os": f"{platform.system()}-{platform.release()}-{platform.machine()}",
            "cuda_version": None,
            "torch_version": "unknown",
        }

        try:
            import torch  # type: ignore
            env["torch_version"] = torch.__version__
            if torch.cuda.is_available():
                env["cuda_version"] = torch.version.cuda
        except ImportError:
            pass

        return env

    @staticmethod
    def save_pip_freeze(output_path: str) -> bool:
        """Run ``pip freeze`` and save to file.  Returns True on success."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                return False
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result.stdout)
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return False

    @staticmethod
    def capture_hardware() -> Dict[str, Any]:
        """Return dict with GPU names, count, memory, CPU info, RAM."""
        hw: Dict[str, Any] = {
            "gpu_names": [],
            "gpu_count": 0,
            "gpu_memory_mb": [],
            "cpu": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count() or 0,
            "ram_mb": 0,
        }

        # RAM via /proc/meminfo (Linux) or platform fallback
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        parts = line.split()
                        hw["ram_mb"] = int(parts[1]) // 1024
                        break
        except (OSError, ValueError, IndexError):
            pass

        # GPU info via torch
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                count = torch.cuda.device_count()
                hw["gpu_count"] = count
                for i in range(count):
                    hw["gpu_names"].append(torch.cuda.get_device_name(i))
                    mem = torch.cuda.get_device_properties(i).total_memory
                    hw["gpu_memory_mb"].append(mem // (1024 * 1024))
        except ImportError:
            pass

        return hw

    @classmethod
    def save_hardware_json(cls, output_path: str) -> bool:
        """Save hardware info as JSON.  Returns True on success."""
        try:
            hw = cls.capture_hardware()
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(hw, f, indent=2)
            return True
        except OSError:
            return False


# ---------------------------------------------------------------------------
# RunIDGenerator
# ---------------------------------------------------------------------------

class RunIDGenerator:
    """Generate deterministic, unique run identifiers."""

    @staticmethod
    def generate_run_id(
        phase: int,
        mode: str,
        git_short: Optional[str] = None,
        timestamp: Optional[datetime.datetime] = None,
    ) -> str:
        """Generate a run ID.

        Format: ``YYYY-MM-DD_HH-MM-SS_phase{N}_{mode}_{git_short}``
        """
        ts = timestamp or datetime.datetime.now(datetime.timezone.utc)
        ts_str = ts.strftime("%Y-%m-%d_%H-%M-%S")
        short = git_short or "0000000"
        return f"{ts_str}_phase{phase}_{mode}_{short}"

    @staticmethod
    def generate_ablation_run_id(
        ablation_id: str,
        run_index: int,
        overrides: Dict[str, Any],
        seed: int,
    ) -> str:
        """Generate an ablation-derived run ID with a stable hash.

        The ablation_id is a content-addressable hash of the overrides.
        """
        ts_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        # Compute stable hash from overrides + seed
        canonical = json.dumps(
            {"overrides": overrides, "seed": seed, "index": run_index},
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]
        return f"{ts_str}_abl_{ablation_id}_{digest}"

    @staticmethod
    def generate_pipeline_run_id(
        mode: str,
        git_short: Optional[str] = None,
        timestamp: Optional[datetime.datetime] = None,
    ) -> str:
        """Generate a pipeline-level run ID (spans multiple phases)."""
        ts = timestamp or datetime.datetime.now(datetime.timezone.utc)
        ts_str = ts.strftime("%Y-%m-%d_%H-%M-%S")
        short = git_short or "0000000"
        return f"{ts_str}_pipeline_{mode}_{short}"

    @staticmethod
    def derive_ablation_id(overrides: Dict[str, Any]) -> str:
        """Derive a stable 8-char ablation ID from an override dict."""
        canonical = json.dumps(overrides, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]


# ---------------------------------------------------------------------------
# RunDirectory
# ---------------------------------------------------------------------------

class RunDirectory:
    """Manage the standard run directory tree.

    Layout::

        runs/<run_id>/
            manifest.json
            manifest.lock.json
            stdout.log
            stderr.log
            logs/
                tensorboard/
                wandb/
            checkpoints/
                phase{1..7}/
            artifacts/
                datasets.json
                env.txt
                pip_freeze.txt
                git_diff.patch
                hardware.json
            reports/
                summary.json
    """

    def __init__(self, base_dir: str, run_id: str) -> None:
        self._base_dir = Path(base_dir)
        self._run_id = run_id
        self._root = self._base_dir / run_id

    # -- creation -----------------------------------------------------------

    @classmethod
    def create(cls, base_dir: str, run_id: str) -> "RunDirectory":
        """Create the full directory tree and return the RunDirectory."""
        rd = cls(base_dir, run_id)
        dirs = [
            rd._root,
            rd._root / "logs" / "tensorboard",
            rd._root / "logs" / "wandb",
            rd._root / "artifacts",
            rd._root / "reports",
        ]
        for phase in range(1, 8):
            dirs.append(rd._root / "checkpoints" / f"phase{phase}")
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        return rd

    # -- path properties ----------------------------------------------------

    @property
    def root(self) -> Path:
        return self._root

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def manifest_path(self) -> Path:
        return self._root / "manifest.json"

    @property
    def lock_path(self) -> Path:
        return self._root / "manifest.lock.json"

    @property
    def tensorboard_dir(self) -> Path:
        return self._root / "logs" / "tensorboard"

    @property
    def wandb_dir(self) -> Path:
        return self._root / "logs" / "wandb"

    def checkpoints_dir(self, phase: int) -> Path:
        return self._root / "checkpoints" / f"phase{phase}"

    @property
    def artifacts_dir(self) -> Path:
        return self._root / "artifacts"

    @property
    def reports_dir(self) -> Path:
        return self._root / "reports"

    @property
    def metrics_path(self) -> Path:
        return self._root / "reports" / "summary.json"

    @property
    def stdout_path(self) -> Path:
        return self._root / "stdout.log"

    @property
    def stderr_path(self) -> Path:
        return self._root / "stderr.log"

    # -- cleanup ------------------------------------------------------------

    def cleanup(
        self,
        keep_best: bool = True,
        keep_boundary: bool = True,
        keep_last: bool = True,
    ) -> int:
        """Remove intermediate checkpoints to save disk space.

        Keeps:
          - ``ckpt_best_*`` files  (if *keep_best*)
          - ``phase_boundary.pt``  (if *keep_boundary*)
          - The latest ``ckpt_step*.pt`` file  (if *keep_last*)

        Returns the number of files removed.
        """
        removed = 0
        for phase in range(1, 8):
            ckpt_dir = self.checkpoints_dir(phase)
            if not ckpt_dir.exists():
                continue

            step_files: List[Path] = sorted(
                ckpt_dir.glob("ckpt_step*.pt"),
                key=lambda p: p.stat().st_mtime if p.exists() else 0,
            )
            last_step = step_files[-1] if step_files else None

            for f in ckpt_dir.iterdir():
                if not f.is_file():
                    continue
                name = f.name

                # Decide whether to keep this file
                if keep_best and name.startswith("ckpt_best"):
                    continue
                if keep_boundary and name == "phase_boundary.pt":
                    continue
                if keep_last and f == last_step:
                    continue

                # Only remove intermediate step checkpoints
                if name.startswith("ckpt_step") and name.endswith(".pt"):
                    f.unlink()
                    removed += 1

        return removed


# ---------------------------------------------------------------------------
# RunManifest
# ---------------------------------------------------------------------------

class RunManifest:
    """Core manifest: capture, save, load, resume, finalize, validate.

    The manifest is the machine-readable provenance record for every training
    run.  It records identity, code provenance, configuration, seeds,
    environment, data, logging, resume chain, checkpoints, and results.
    """

    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        self._data: Dict[str, Any] = data or {}

    # -- properties ---------------------------------------------------------

    @property
    def data(self) -> Dict[str, Any]:
        """Return the raw manifest dict."""
        return self._data

    @property
    def run_id(self) -> str:
        return self._data.get("identity", {}).get("run_id", "")

    @property
    def phase(self) -> int:
        return self._data.get("identity", {}).get("phase", 0)

    @property
    def mode(self) -> str:
        return self._data.get("identity", {}).get("mode", "")

    @property
    def status(self) -> str:
        return self._data.get("results", {}).get("status", "unknown")

    # -- capture ------------------------------------------------------------

    @classmethod
    def capture(
        cls,
        config: Any,
        phase: int,
        mode: str,
        run_id: Optional[str] = None,
        ablation_id: Optional[str] = None,
        repo_dir: Optional[str] = None,
        entrypoint: str = "",
        cli_args: Optional[List[str]] = None,
        datasets: Optional[List[Dict[str, Any]]] = None,
        seeds: Optional[Dict[str, Any]] = None,
        logging_config: Optional[Dict[str, Any]] = None,
        checkpoint_config: Optional[Dict[str, Any]] = None,
    ) -> "RunManifest":
        """Build a complete manifest dict from live state.

        Parameters
        ----------
        config : object
            A config object (dataclass or dict).  Serialized via
            ``dataclasses.asdict`` if a dataclass, or used directly if dict.
        phase : int
            Training phase (1-7).
        mode : str
            ``"dev"`` or ``"production"``.
        run_id : str, optional
            Override run ID.  Auto-generated if ``None``.
        ablation_id : str, optional
            Non-null only for ablation runs.
        repo_dir : str, optional
            Path to the git repository root.  Defaults to cwd.
        entrypoint : str
            Script path relative to repo root.
        cli_args : list of str, optional
            Original command-line arguments.
        datasets : list of dict, optional
            Dataset provenance entries.
        seeds : dict, optional
            Seed configuration.  Uses defaults if ``None``.
        logging_config : dict, optional
            Logging backend config.  Uses defaults if ``None``.
        checkpoint_config : dict, optional
            Checkpoint policy config.  Uses defaults if ``None``.
        """
        repo = repo_dir or os.getcwd()
        git_info = GitProvenanceCapture.capture_git_info(repo)
        env_info = EnvironmentCapture.capture_environment()
        hw_info = EnvironmentCapture.capture_hardware()

        # Generate or use provided run_id
        if run_id is None:
            run_id = RunIDGenerator.generate_run_id(
                phase=phase, mode=mode, git_short=git_info.get("short_sha")
            )

        # Serialize config
        if hasattr(config, "__dataclass_fields__"):
            config_dict = asdict(config)
        elif isinstance(config, dict):
            config_dict = copy.deepcopy(config)
        else:
            config_dict = {"raw": str(config)}

        # Extract feature flags
        feature_flags: Dict[str, bool] = {}
        for key in ManifestSchema.FEATURE_FLAG_KEYS:
            feature_flags[key] = bool(config_dict.get(key, False))

        # Default seeds
        if seeds is None:
            seeds = {
                "base_seed": 1337,
                "per_phase_offsets": [0, 100, 200, 300, 400, 500, 600],
                "torch_deterministic": True,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "use_deterministic_algorithms": True,
            }

        # Default logging
        if logging_config is None:
            logging_config = {
                "tensorboard": {"enabled": True, "log_dir": "logs/tensorboard/"},
                "wandb": {"enabled": False, "project": None, "run_id": None},
            }

        # Default checkpoint config
        if checkpoint_config is None:
            checkpoint_config = {
                "save_every_n_steps": 1000,
                "best_metric_key": "val/loss",
                "phase_boundary_produced": None,
            }

        # Default datasets
        if datasets is None:
            datasets = []

        now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        manifest_dict: Dict[str, Any] = {
            "schema_version": ManifestSchema.SCHEMA_VERSION,
            "identity": {
                "run_id": run_id,
                "ablation_id": ablation_id,
                "phase": phase,
                "mode": mode,
                "timestamp_start": now,
                "timestamp_end": None,
                "hostname": socket.gethostname(),
                "user": _safe_getuser(),
            },
            "git": {
                "commit": git_info["commit"],
                "branch": git_info["branch"],
                "remote_url": git_info.get("remote_url", "unknown"),
                "dirty": git_info["dirty"],
                "patch_path": "artifacts/git_diff.patch" if git_info["dirty"] else None,
                "entrypoint": entrypoint,
                "cli_args": cli_args or [],
            },
            "config": {
                "brain_ai": config_dict,
                "overrides": {},
                "resolved": copy.deepcopy(config_dict),
                "feature_flags": feature_flags,
            },
            "seeds": seeds,
            "env": {
                "python_version": env_info["python_version"],
                "os": env_info["os"],
                "cuda_version": env_info.get("cuda_version"),
                "torch_version": env_info["torch_version"],
                "pip_freeze_path": "artifacts/pip_freeze.txt",
                "hardware": {
                    "gpu_names": hw_info["gpu_names"],
                    "gpu_count": hw_info["gpu_count"],
                    "gpu_memory_mb": hw_info["gpu_memory_mb"],
                },
            },
            "data": {
                "datasets": datasets,
            },
            "logging": logging_config,
            "resume": {
                "enabled": False,
                "from_run_id": None,
                "from_checkpoint": None,
                "phase_boundary_artifacts": [],
                "resume_events": [],
            },
            "checkpoints": checkpoint_config,
            "results": {
                "status": "running",
                "best_metrics": None,
                "final_metrics": None,
                "error": None,
            },
        }

        return cls(data=manifest_dict)

    # -- save / load --------------------------------------------------------

    def save(self, run_dir: Union[str, Path]) -> Path:
        """Write ``manifest.json`` to *run_dir* with indent=2.

        Returns the path to the written file.
        """
        path = Path(run_dir) / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, default=str)
        return path

    @classmethod
    def load(cls, run_dir: Union[str, Path]) -> "RunManifest":
        """Load and validate ``manifest.json`` from *run_dir*."""
        path = Path(run_dir) / "manifest.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ManifestSchema.validate_schema(data)
        return cls(data=data)

    def save_lock(self, run_dir: Union[str, Path]) -> Path:
        """Write ``manifest.lock.json`` -- fully resolved config only.

        The lock file is a standalone snapshot of ``config.resolved`` so that
        re-running from the lock reproduces identical configuration without
        relying on default detection logic.
        """
        lock: Dict[str, Any] = {
            "schema_version": ManifestSchema.SCHEMA_VERSION,
            "resolved_config": copy.deepcopy(
                self._data.get("config", {}).get("resolved", {})
            ),
        }
        path = Path(run_dir) / "manifest.lock.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(lock, f, indent=2, default=str)
        return path

    @staticmethod
    def load_lock(run_dir: Union[str, Path]) -> Dict[str, Any]:
        """Load ``manifest.lock.json`` and return the resolved_config dict."""
        path = Path(run_dir) / "manifest.lock.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("resolved_config", {})

    # -- resume -------------------------------------------------------------

    def add_resume_event(
        self,
        checkpoint_path: str,
        new_args: Optional[Dict[str, Any]] = None,
        step_resumed_at: int = 0,
        reason: str = "manual",
    ) -> None:
        """Append a resume event.  Never overwrites prior events."""
        resume = self._data.setdefault("resume", {})
        resume["enabled"] = True

        events = resume.setdefault("resume_events", [])
        event: Dict[str, Any] = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "from_run_id": resume.get("from_run_id", self.run_id),
            "from_checkpoint": checkpoint_path,
            "step_resumed_at": step_resumed_at,
            "reason": reason,
        }
        if new_args:
            event["new_args"] = new_args
        events.append(event)

        # Update top-level resume pointers
        resume["from_checkpoint"] = checkpoint_path
        if resume.get("from_run_id") is None:
            resume["from_run_id"] = self.run_id

    # -- finalize -----------------------------------------------------------

    def finalize(
        self,
        status: str,
        best_metrics: Optional[Dict[str, Any]] = None,
        final_metrics: Optional[Dict[str, Any]] = None,
        error: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update the results section at run completion.

        Parameters
        ----------
        status : str
            One of ``"completed"``, ``"failed"``, ``"interrupted"``.
        best_metrics : dict, optional
            Best metric values observed during training.
        final_metrics : dict, optional
            Metric values at the last training step.
        error : dict, optional
            Error details (``type``, ``message``, ``traceback``, ``step``).
        """
        if status not in ("completed", "failed", "interrupted"):
            raise ValueError(
                f"Terminal status must be completed/failed/interrupted, got {status!r}"
            )

        self._data["results"] = {
            "status": status,
            "best_metrics": best_metrics,
            "final_metrics": final_metrics,
            "error": error,
        }
        self._data["identity"]["timestamp_end"] = (
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        )

    # -- validate -----------------------------------------------------------

    def validate(self) -> List[str]:
        """Run cross-field consistency checks.

        Returns a list of error strings.  Empty means valid.
        """
        errors: List[str] = []
        d = self._data

        # -- identity checks --
        identity = d.get("identity", {})
        phase = identity.get("phase")
        mode = identity.get("mode")
        run_id = identity.get("run_id", "")

        if phase is not None and (not isinstance(phase, int) or not 1 <= phase <= 7):
            errors.append(f"identity.phase must be int 1-7, got {phase!r}")

        if mode not in ManifestSchema.VALID_MODES:
            errors.append(f"identity.mode must be dev/production, got {mode!r}")

        if phase is not None and f"phase{phase}" not in run_id:
            errors.append(
                f"identity.run_id must contain 'phase{phase}', got {run_id!r}"
            )

        if mode and mode not in run_id:
            errors.append(
                f"identity.run_id must contain '{mode}', got {run_id!r}"
            )

        # Timestamp ordering
        ts_start = identity.get("timestamp_start")
        ts_end = identity.get("timestamp_end")
        if ts_start and ts_end:
            if ts_end < ts_start:
                errors.append("identity.timestamp_end is before timestamp_start")

        # Ablation ID in run_id
        abl_id = identity.get("ablation_id")
        if abl_id is not None and "_abl_" not in run_id:
            errors.append(
                "identity.ablation_id is set but run_id does not contain '_abl_'"
            )

        # -- git checks --
        git = d.get("git", {})
        commit = git.get("commit", "")
        if len(commit) != 40 or not all(c in "0123456789abcdef" for c in commit):
            errors.append(f"git.commit must be 40 hex chars, got {commit!r}")

        dirty = git.get("dirty", False)
        patch = git.get("patch_path")
        if dirty and not patch:
            errors.append("git.dirty is true but git.patch_path is null")
        if not dirty and patch:
            errors.append("git.dirty is false but git.patch_path is set")

        # -- seeds checks --
        seeds = d.get("seeds", {})
        offsets = seeds.get("per_phase_offsets", [])
        if not isinstance(offsets, list) or len(offsets) != 7:
            errors.append("seeds.per_phase_offsets must be list of 7 ints")

        if seeds.get("use_deterministic_algorithms") and seeds.get("cudnn_benchmark"):
            errors.append(
                "seeds: use_deterministic_algorithms=true requires cudnn_benchmark=false"
            )

        # -- resume checks --
        resume = d.get("resume", {})
        if resume.get("enabled"):
            if not resume.get("from_run_id"):
                errors.append("resume.enabled=true requires from_run_id")
            if not resume.get("from_checkpoint"):
                errors.append("resume.enabled=true requires from_checkpoint")
        else:
            if resume.get("from_run_id") is not None:
                errors.append("resume.enabled=false but from_run_id is set")
            if resume.get("from_checkpoint") is not None:
                errors.append("resume.enabled=false but from_checkpoint is set")

        # -- results checks --
        results = d.get("results", {})
        status = results.get("status")
        if status == "failed" and not results.get("error"):
            errors.append("results.status=failed requires error to be non-null")
        if status == "completed":
            if results.get("error") is not None:
                errors.append("results.status=completed but error is non-null")
            if results.get("final_metrics") is None:
                errors.append("results.status=completed requires final_metrics")
        if status == "running":
            if identity.get("timestamp_end") is not None:
                errors.append(
                    "results.status=running but timestamp_end is set"
                )

        # -- feature flags --
        ff = d.get("config", {}).get("feature_flags", {})
        for key in ManifestSchema.FEATURE_FLAG_KEYS:
            if key not in ff:
                errors.append(f"Missing feature flag: {key}")

        return errors


# ---------------------------------------------------------------------------
# ManifestComparator
# ---------------------------------------------------------------------------

class ManifestComparator:
    """Compare two manifests for reproduction equivalence."""

    # Fields that must be identical for reproduction
    EXACT_MATCH_PATHS: List[str] = [
        "git.commit",
        "config.resolved",
        "seeds",
        "identity.phase",
        "identity.mode",
    ]

    # Fields to warn about but not reject
    WARN_PATHS: List[str] = [
        "env.python_version",
        "env.torch_version",
        "env.cuda_version",
        "env.hardware",
    ]

    # Fields to ignore entirely
    IGNORE_PATHS: List[str] = [
        "identity.run_id",
        "identity.timestamp_start",
        "identity.timestamp_end",
        "identity.hostname",
        "identity.user",
        "identity.ablation_id",
        "logging.wandb.run_id",
        "results",
        "resume",
    ]

    @staticmethod
    def _get_nested(data: Dict[str, Any], path: str) -> Any:
        """Traverse a dotted path into a nested dict."""
        keys = path.split(".")
        current: Any = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return _SENTINEL
        return current

    @classmethod
    def compare_for_reproduction(
        cls,
        original: Dict[str, Any],
        current: Dict[str, Any],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Compare two manifest dicts for reproduction equivalence.

        Returns a dict with keys:
            ``"mismatches"``  -- critical differences that prevent reproduction
            ``"warnings"``    -- non-critical differences worth noting
            ``"match"``       -- True if no critical mismatches
        """
        mismatches: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []

        # Exact match fields
        for path in cls.EXACT_MATCH_PATHS:
            orig_val = cls._get_nested(original, path)
            curr_val = cls._get_nested(current, path)
            if orig_val is _SENTINEL and curr_val is _SENTINEL:
                continue
            if orig_val != curr_val:
                mismatches.append({
                    "field": path,
                    "original": orig_val if orig_val is not _SENTINEL else "<missing>",
                    "current": curr_val if curr_val is not _SENTINEL else "<missing>",
                })

        # Dataset fingerprints
        orig_datasets = original.get("data", {}).get("datasets", [])
        curr_datasets = current.get("data", {}).get("datasets", [])
        orig_by_key = {
            (ds.get("name"), ds.get("split")): ds for ds in orig_datasets
        }
        curr_by_key = {
            (ds.get("name"), ds.get("split")): ds for ds in curr_datasets
        }
        all_keys = set(orig_by_key.keys()) | set(curr_by_key.keys())
        for key in sorted(all_keys, key=str):
            if key not in orig_by_key:
                mismatches.append({
                    "field": f"data.datasets[{key}]",
                    "original": "<missing>",
                    "current": "present",
                })
            elif key not in curr_by_key:
                mismatches.append({
                    "field": f"data.datasets[{key}]",
                    "original": "present",
                    "current": "<missing>",
                })
            else:
                orig_fp = orig_by_key[key].get("fingerprint")
                curr_fp = curr_by_key[key].get("fingerprint")
                if orig_fp != curr_fp:
                    mismatches.append({
                        "field": f"data.datasets[{key}].fingerprint",
                        "original": orig_fp,
                        "current": curr_fp,
                    })
                orig_ts = orig_by_key[key].get("transforms_signature")
                curr_ts = curr_by_key[key].get("transforms_signature")
                if orig_ts != curr_ts:
                    mismatches.append({
                        "field": f"data.datasets[{key}].transforms_signature",
                        "original": orig_ts,
                        "current": curr_ts,
                    })

        # Warning fields
        for path in cls.WARN_PATHS:
            orig_val = cls._get_nested(original, path)
            curr_val = cls._get_nested(current, path)
            if orig_val is _SENTINEL and curr_val is _SENTINEL:
                continue
            if orig_val != curr_val:
                warnings.append({
                    "field": path,
                    "original": orig_val if orig_val is not _SENTINEL else "<missing>",
                    "current": curr_val if curr_val is not _SENTINEL else "<missing>",
                })

        return {
            "mismatches": mismatches,
            "warnings": warnings,
            "match": len(mismatches) == 0,
        }

    @classmethod
    def is_reproducible(
        cls,
        original: Dict[str, Any],
        current: Dict[str, Any],
    ) -> bool:
        """Return True if the two manifests are reproduction-equivalent."""
        result = cls.compare_for_reproduction(original, current)
        return result["match"]


# Sentinel for missing nested values
class _SentinelType:
    def __repr__(self) -> str:
        return "<MISSING>"

_SENTINEL = _SentinelType()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_getuser() -> str:
    """Get the current OS user, or 'unknown' on failure."""
    try:
        return getpass.getuser()
    except Exception:
        return os.environ.get("USER", os.environ.get("USERNAME", "unknown"))


# ===========================================================================
# Self-test block
# ===========================================================================

if __name__ == "__main__":
    import traceback as _tb

    _pass = 0
    _fail = 0
    _errors: List[str] = []

    def _assert(condition: bool, name: str, detail: str = "") -> None:
        global _pass, _fail
        if condition:
            _pass += 1
            print(f"  PASS  {name}")
        else:
            _fail += 1
            msg = f"  FAIL  {name}"
            if detail:
                msg += f"  -- {detail}"
            print(msg)
            _errors.append(msg)

    # -----------------------------------------------------------------------
    # Mock config (inline, no brain_ai import)
    # -----------------------------------------------------------------------
    @dataclass
    class _MockSNNConfig:
        beta: float = 0.95
        num_timesteps: int = 10
        hidden_sizes: list = field(default_factory=lambda: [256, 128])

    @dataclass
    class _MockConfig:
        snn: _MockSNNConfig = field(default_factory=_MockSNNConfig)
        use_snn: bool = True
        use_htm: bool = False
        use_workspace: bool = True
        use_symbolic: bool = False
        use_meta: bool = False
        use_engram: bool = False
        learning_rate: float = 0.001
        batch_size: int = 8

    print("=" * 72)
    print("RunManifest system self-test")
    print("=" * 72)

    # -----------------------------------------------------------------------
    # 1. ManifestSchema validation
    # -----------------------------------------------------------------------
    print("\n--- ManifestSchema ---")

    # 1a. Missing sections
    try:
        ManifestSchema.validate_schema({})
        _assert(False, "schema: rejects empty dict")
    except ValueError:
        _assert(True, "schema: rejects empty dict")

    # 1b. Valid minimal manifest structure
    _minimal_manifest = {
        "schema_version": "1.0",
        "identity": {
            "run_id": "2026-01-01_00-00-00_phase1_dev_abc1234",
            "ablation_id": None,
            "phase": 1,
            "mode": "dev",
            "timestamp_start": "2026-01-01T00:00:00Z",
            "timestamp_end": None,
            "hostname": "test",
            "user": "tester",
        },
        "git": {
            "commit": "a" * 40,
            "branch": "main",
            "remote_url": "https://github.com/test/test.git",
            "dirty": False,
            "patch_path": None,
            "entrypoint": "scripts/test.py",
            "cli_args": [],
        },
        "config": {
            "brain_ai": {},
            "overrides": {},
            "resolved": {},
            "feature_flags": {
                "use_snn": True, "use_htm": False, "use_workspace": True,
                "use_symbolic": False, "use_meta": False, "use_engram": False,
            },
        },
        "seeds": {
            "base_seed": 1337,
            "per_phase_offsets": [0, 100, 200, 300, 400, 500, 600],
            "torch_deterministic": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "use_deterministic_algorithms": True,
        },
        "env": {
            "python_version": "3.11.5",
            "os": "Linux-test",
            "cuda_version": None,
            "torch_version": "2.0.0",
            "pip_freeze_path": "artifacts/pip_freeze.txt",
            "hardware": {"gpu_names": [], "gpu_count": 0, "gpu_memory_mb": []},
        },
        "data": {"datasets": []},
        "logging": {
            "tensorboard": {"enabled": True, "log_dir": "logs/tensorboard/"},
            "wandb": {"enabled": False, "project": None, "run_id": None},
        },
        "resume": {
            "enabled": False,
            "from_run_id": None,
            "from_checkpoint": None,
            "phase_boundary_artifacts": [],
            "resume_events": [],
        },
        "checkpoints": {
            "save_every_n_steps": 1000,
            "best_metric_key": "val/loss",
            "phase_boundary_produced": None,
        },
        "results": {
            "status": "running",
            "best_metrics": None,
            "final_metrics": None,
            "error": None,
        },
    }
    try:
        ManifestSchema.validate_schema(_minimal_manifest)
        _assert(True, "schema: accepts valid manifest")
    except ValueError as e:
        _assert(False, "schema: accepts valid manifest", str(e))

    # 1c. Invalid phase
    _bad = copy.deepcopy(_minimal_manifest)
    _bad["identity"]["phase"] = 99
    try:
        ManifestSchema.validate_schema(_bad)
        _assert(False, "schema: rejects phase=99")
    except ValueError:
        _assert(True, "schema: rejects phase=99")

    # 1d. Invalid mode
    _bad = copy.deepcopy(_minimal_manifest)
    _bad["identity"]["mode"] = "turbo"
    try:
        ManifestSchema.validate_schema(_bad)
        _assert(False, "schema: rejects mode=turbo")
    except ValueError:
        _assert(True, "schema: rejects mode=turbo")

    # 1e. Wrong per_phase_offsets length
    _bad = copy.deepcopy(_minimal_manifest)
    _bad["seeds"]["per_phase_offsets"] = [0, 1, 2]
    try:
        ManifestSchema.validate_schema(_bad)
        _assert(False, "schema: rejects 3-element offsets")
    except ValueError:
        _assert(True, "schema: rejects 3-element offsets")

    # 1f. Missing feature flag
    _bad = copy.deepcopy(_minimal_manifest)
    del _bad["config"]["feature_flags"]["use_engram"]
    try:
        ManifestSchema.validate_schema(_bad)
        _assert(False, "schema: rejects missing feature flag")
    except ValueError:
        _assert(True, "schema: rejects missing feature flag")

    # 1g. Invalid status
    _bad = copy.deepcopy(_minimal_manifest)
    _bad["results"]["status"] = "exploded"
    try:
        ManifestSchema.validate_schema(_bad)
        _assert(False, "schema: rejects invalid status")
    except ValueError:
        _assert(True, "schema: rejects invalid status")

    # -----------------------------------------------------------------------
    # 2. RunManifest.capture with mock config
    # -----------------------------------------------------------------------
    print("\n--- RunManifest.capture ---")

    _cfg = _MockConfig()
    _m = RunManifest.capture(
        config=_cfg,
        phase=4,
        mode="dev",
        run_id="2026-02-20_14-30-00_phase4_dev_abc1234",
        entrypoint="scripts/train_phase4.py",
        cli_args=["--mode", "dev"],
        datasets=[{
            "name": "mnist", "config": None, "split": "train",
            "version": "1.0.0", "fingerprint": "abc123",
            "fingerprint_tier": 1, "num_samples": 60000,
            "transforms_signature": "sha256:deadbeef",
        }],
    )

    _assert(_m.run_id == "2026-02-20_14-30-00_phase4_dev_abc1234",
            "capture: run_id matches")
    _assert(_m.phase == 4, "capture: phase is 4")
    _assert(_m.mode == "dev", "capture: mode is dev")
    _assert(_m.status == "running", "capture: initial status is running")
    _assert(
        _m.data["identity"]["hostname"] == socket.gethostname(),
        "capture: hostname auto-detected",
    )
    _assert(
        _m.data["identity"]["user"] == _safe_getuser(),
        "capture: user auto-detected",
    )
    _assert(
        _m.data["identity"]["timestamp_start"] is not None,
        "capture: timestamp_start set",
    )
    _assert(
        _m.data["identity"]["timestamp_end"] is None,
        "capture: timestamp_end is null (running)",
    )
    _assert(
        _m.data["config"]["feature_flags"]["use_snn"] is True,
        "capture: feature flag use_snn extracted",
    )
    _assert(
        _m.data["config"]["feature_flags"]["use_engram"] is False,
        "capture: feature flag use_engram extracted",
    )
    _assert(
        len(_m.data["seeds"]["per_phase_offsets"]) == 7,
        "capture: default seeds have 7 offsets",
    )
    _assert(
        _m.data["env"]["python_version"] == platform.python_version(),
        "capture: python version matches",
    )
    _assert(
        len(_m.data["data"]["datasets"]) == 1,
        "capture: datasets list has 1 entry",
    )
    _assert(
        _m.data["data"]["datasets"][0]["name"] == "mnist",
        "capture: dataset name is mnist",
    )
    _assert(
        _m.data["git"]["entrypoint"] == "scripts/train_phase4.py",
        "capture: entrypoint recorded",
    )
    _assert(
        _m.data["git"]["cli_args"] == ["--mode", "dev"],
        "capture: cli_args recorded",
    )

    # Capture with dict config
    _m_dict = RunManifest.capture(
        config={"use_snn": True, "use_htm": False, "use_workspace": False,
                "use_symbolic": False, "use_meta": False, "use_engram": False,
                "lr": 0.01},
        phase=1,
        mode="dev",
        run_id="2026-01-01_00-00-00_phase1_dev_0000000",
    )
    _assert(
        _m_dict.data["config"]["brain_ai"]["lr"] == 0.01,
        "capture: dict config serialized",
    )

    # -----------------------------------------------------------------------
    # 3. Save / Load round-trip
    # -----------------------------------------------------------------------
    print("\n--- RunManifest save/load ---")

    _tmpdir = tempfile.mkdtemp(prefix="manifest_test_")
    try:
        _saved_path = _m.save(_tmpdir)
        _assert(
            _saved_path.exists(),
            "save: manifest.json written",
        )

        # Read and verify JSON
        with open(_saved_path, "r") as _f:
            _raw = json.load(_f)
        _assert(
            _raw["identity"]["run_id"] == _m.run_id,
            "save: JSON content matches run_id",
        )

        # Load
        _loaded = RunManifest.load(_tmpdir)
        _assert(
            _loaded.run_id == _m.run_id,
            "load: round-trip run_id",
        )
        _assert(
            _loaded.phase == _m.phase,
            "load: round-trip phase",
        )
        _assert(
            _loaded.data["seeds"] == _m.data["seeds"],
            "load: round-trip seeds",
        )
        _assert(
            _loaded.data["config"]["feature_flags"] == _m.data["config"]["feature_flags"],
            "load: round-trip feature_flags",
        )
    finally:
        shutil.rmtree(_tmpdir, ignore_errors=True)

    # -----------------------------------------------------------------------
    # 4. Lock file generation
    # -----------------------------------------------------------------------
    print("\n--- Lock file ---")

    _tmpdir = tempfile.mkdtemp(prefix="manifest_lock_test_")
    try:
        _lock_path = _m.save_lock(_tmpdir)
        _assert(
            _lock_path.exists(),
            "save_lock: manifest.lock.json written",
        )
        with open(_lock_path, "r") as _f:
            _lock_data = json.load(_f)
        _assert(
            "schema_version" in _lock_data,
            "save_lock: has schema_version",
        )
        _assert(
            "resolved_config" in _lock_data,
            "save_lock: has resolved_config",
        )
        _assert(
            _lock_data["resolved_config"] == _m.data["config"]["resolved"],
            "save_lock: resolved_config matches manifest",
        )

        # Load lock
        _loaded_lock = RunManifest.load_lock(_tmpdir)
        _assert(
            _loaded_lock == _m.data["config"]["resolved"],
            "load_lock: round-trip resolved_config",
        )
    finally:
        shutil.rmtree(_tmpdir, ignore_errors=True)

    # -----------------------------------------------------------------------
    # 5. Resume event appending
    # -----------------------------------------------------------------------
    print("\n--- Resume events ---")

    _m_resume = RunManifest.capture(
        config=_cfg, phase=3, mode="dev",
        run_id="2026-02-20_10-00-00_phase3_dev_abc1234",
    )
    _assert(
        _m_resume.data["resume"]["enabled"] is False,
        "resume: initially disabled",
    )

    _m_resume.add_resume_event(
        checkpoint_path="checkpoints/phase3/ckpt_step00005000.pt",
        step_resumed_at=5000,
        reason="preemption",
    )
    _assert(
        _m_resume.data["resume"]["enabled"] is True,
        "resume: enabled after add_resume_event",
    )
    _assert(
        len(_m_resume.data["resume"]["resume_events"]) == 1,
        "resume: 1 event after first add",
    )
    _assert(
        _m_resume.data["resume"]["resume_events"][0]["reason"] == "preemption",
        "resume: event reason is preemption",
    )
    _assert(
        _m_resume.data["resume"]["resume_events"][0]["step_resumed_at"] == 5000,
        "resume: event step is 5000",
    )

    # Append second event
    _m_resume.add_resume_event(
        checkpoint_path="checkpoints/phase3/ckpt_step00008000.pt",
        step_resumed_at=8000,
        reason="manual",
        new_args={"batch_size": 16},
    )
    _assert(
        len(_m_resume.data["resume"]["resume_events"]) == 2,
        "resume: 2 events after second add (append-only)",
    )
    _assert(
        "new_args" in _m_resume.data["resume"]["resume_events"][1],
        "resume: second event has new_args",
    )
    _assert(
        _m_resume.data["resume"]["from_checkpoint"]
        == "checkpoints/phase3/ckpt_step00008000.pt",
        "resume: from_checkpoint updated to latest",
    )

    # -----------------------------------------------------------------------
    # 6. Finalize
    # -----------------------------------------------------------------------
    print("\n--- Finalize ---")

    # 6a. Finalize with success
    _m_fin = RunManifest.capture(
        config=_cfg, phase=2, mode="dev",
        run_id="2026-02-20_12-00-00_phase2_dev_abc1234",
    )
    _m_fin.finalize(
        status="completed",
        best_metrics={"val/loss": 0.05, "val/accuracy": 0.98, "step": 500},
        final_metrics={"train/loss": 0.03, "val/loss": 0.06, "step": 1000},
    )
    _assert(
        _m_fin.status == "completed",
        "finalize: status is completed",
    )
    _assert(
        _m_fin.data["results"]["best_metrics"]["val/loss"] == 0.05,
        "finalize: best_metrics recorded",
    )
    _assert(
        _m_fin.data["results"]["final_metrics"]["step"] == 1000,
        "finalize: final_metrics recorded",
    )
    _assert(
        _m_fin.data["results"]["error"] is None,
        "finalize: error is None for completed",
    )
    _assert(
        _m_fin.data["identity"]["timestamp_end"] is not None,
        "finalize: timestamp_end set",
    )

    # 6b. Finalize with failure
    _m_fail = RunManifest.capture(
        config=_cfg, phase=5, mode="production",
        run_id="2026-02-20_16-00-00_phase5_production_abc1234",
    )
    _m_fail.finalize(
        status="failed",
        best_metrics={"val/loss": 0.15, "step": 200},
        error={
            "type": "RuntimeError",
            "message": "CUDA OOM",
            "traceback": "Traceback...",
            "step": 250,
        },
    )
    _assert(
        _m_fail.status == "failed",
        "finalize: status is failed",
    )
    _assert(
        _m_fail.data["results"]["error"]["type"] == "RuntimeError",
        "finalize: error type recorded",
    )
    _assert(
        _m_fail.data["results"]["final_metrics"] is None,
        "finalize: final_metrics None on failure",
    )

    # 6c. Invalid terminal status
    try:
        _m_fin.finalize(status="running")
        _assert(False, "finalize: rejects 'running' as terminal status")
    except ValueError:
        _assert(True, "finalize: rejects 'running' as terminal status")

    # -----------------------------------------------------------------------
    # 7. Validate (cross-field consistency)
    # -----------------------------------------------------------------------
    print("\n--- Validate ---")

    # 7a. Valid running manifest
    _v = RunManifest.capture(
        config=_cfg, phase=1, mode="dev",
        run_id="2026-01-01_00-00-00_phase1_dev_abc1234",
    )
    _errs = _v.validate()
    _assert(len(_errs) == 0, "validate: valid running manifest has no errors",
            f"errors: {_errs}")

    # 7b. Phase out of range
    _v_bad = RunManifest(data=copy.deepcopy(_minimal_manifest))
    _v_bad._data["identity"]["phase"] = 0
    _errs = _v_bad.validate()
    _assert(
        any("phase" in e for e in _errs),
        "validate: phase=0 flagged",
    )

    # 7c. Phase mismatch in run_id
    _v_bad2 = RunManifest(data=copy.deepcopy(_minimal_manifest))
    _v_bad2._data["identity"]["phase"] = 3
    _v_bad2._data["identity"]["run_id"] = "2026-01-01_00-00-00_phase1_dev_abc1234"
    _errs = _v_bad2.validate()
    _assert(
        any("phase3" in e for e in _errs),
        "validate: phase3 not in run_id flagged",
    )

    # 7d. Resume without from_run_id
    _v_bad3 = RunManifest(data=copy.deepcopy(_minimal_manifest))
    _v_bad3._data["resume"]["enabled"] = True
    _v_bad3._data["resume"]["from_run_id"] = None
    _v_bad3._data["resume"]["from_checkpoint"] = None
    _errs = _v_bad3.validate()
    _assert(
        any("from_run_id" in e for e in _errs),
        "validate: resume without from_run_id flagged",
    )

    # 7e. Completed without final_metrics
    _v_bad4 = RunManifest(data=copy.deepcopy(_minimal_manifest))
    _v_bad4._data["results"]["status"] = "completed"
    _v_bad4._data["results"]["final_metrics"] = None
    _v_bad4._data["results"]["error"] = None
    _v_bad4._data["identity"]["timestamp_end"] = "2026-01-01T01:00:00Z"
    _errs = _v_bad4.validate()
    _assert(
        any("final_metrics" in e for e in _errs),
        "validate: completed without final_metrics flagged",
    )

    # 7f. Failed without error
    _v_bad5 = RunManifest(data=copy.deepcopy(_minimal_manifest))
    _v_bad5._data["results"]["status"] = "failed"
    _v_bad5._data["results"]["error"] = None
    _v_bad5._data["identity"]["timestamp_end"] = "2026-01-01T01:00:00Z"
    _errs = _v_bad5.validate()
    _assert(
        any("error" in e.lower() for e in _errs),
        "validate: failed without error flagged",
    )

    # 7g. Running with timestamp_end set
    _v_bad6 = RunManifest(data=copy.deepcopy(_minimal_manifest))
    _v_bad6._data["results"]["status"] = "running"
    _v_bad6._data["identity"]["timestamp_end"] = "2026-01-01T01:00:00Z"
    _errs = _v_bad6.validate()
    _assert(
        any("timestamp_end" in e for e in _errs),
        "validate: running with timestamp_end flagged",
    )

    # 7h. Deterministic + benchmark conflict
    _v_bad7 = RunManifest(data=copy.deepcopy(_minimal_manifest))
    _v_bad7._data["seeds"]["use_deterministic_algorithms"] = True
    _v_bad7._data["seeds"]["cudnn_benchmark"] = True
    _errs = _v_bad7.validate()
    _assert(
        any("benchmark" in e for e in _errs),
        "validate: deterministic+benchmark conflict flagged",
    )

    # 7i. Dirty without patch_path
    _v_bad8 = RunManifest(data=copy.deepcopy(_minimal_manifest))
    _v_bad8._data["git"]["dirty"] = True
    _v_bad8._data["git"]["patch_path"] = None
    _errs = _v_bad8.validate()
    _assert(
        any("patch_path" in e for e in _errs),
        "validate: dirty without patch_path flagged",
    )

    # 7j. Clean with patch_path set
    _v_bad9 = RunManifest(data=copy.deepcopy(_minimal_manifest))
    _v_bad9._data["git"]["dirty"] = False
    _v_bad9._data["git"]["patch_path"] = "some/path.patch"
    _errs = _v_bad9.validate()
    _assert(
        any("patch_path" in e for e in _errs),
        "validate: clean with patch_path flagged",
    )

    # -----------------------------------------------------------------------
    # 8. GitProvenanceCapture
    # -----------------------------------------------------------------------
    print("\n--- GitProvenanceCapture ---")

    _git_info = GitProvenanceCapture.capture_git_info(os.getcwd())
    _assert(
        isinstance(_git_info, dict),
        "git: capture_git_info returns dict",
    )
    _assert(
        "commit" in _git_info and "branch" in _git_info and "dirty" in _git_info,
        "git: has commit, branch, dirty keys",
    )
    _assert(
        len(_git_info["commit"]) == 40,
        "git: commit is 40 chars (or fallback zeros)",
    )
    _assert(
        len(_git_info["short_sha"]) == 7,
        "git: short_sha is 7 chars",
    )
    _assert(
        isinstance(_git_info["dirty"], bool),
        "git: dirty is bool",
    )

    # Test save_git_diff (may or may not produce output)
    _tmp_patch = tempfile.mktemp(suffix=".patch")
    _diff_result = GitProvenanceCapture.save_git_diff(os.getcwd(), _tmp_patch)
    _assert(
        isinstance(_diff_result, bool),
        "git: save_git_diff returns bool",
    )
    if _diff_result and os.path.exists(_tmp_patch):
        os.unlink(_tmp_patch)

    # Test with non-git directory
    _non_git = tempfile.mkdtemp(prefix="non_git_")
    try:
        _non_git_info = GitProvenanceCapture.capture_git_info(_non_git)
        _assert(
            _non_git_info["commit"] == "0" * 40,
            "git: non-repo returns zero commit",
        )
        _assert(
            _non_git_info["branch"] == "unknown",
            "git: non-repo returns unknown branch",
        )
    finally:
        shutil.rmtree(_non_git, ignore_errors=True)

    # -----------------------------------------------------------------------
    # 9. EnvironmentCapture
    # -----------------------------------------------------------------------
    print("\n--- EnvironmentCapture ---")

    _env = EnvironmentCapture.capture_environment()
    _assert(isinstance(_env, dict), "env: capture returns dict")
    _assert(
        _env["python_version"] == platform.python_version(),
        "env: python version matches platform",
    )
    _assert("os" in _env, "env: has os field")
    _assert("torch_version" in _env, "env: has torch_version field")
    _assert("cuda_version" in _env, "env: has cuda_version field")

    _hw = EnvironmentCapture.capture_hardware()
    _assert(isinstance(_hw, dict), "env: capture_hardware returns dict")
    _assert("gpu_count" in _hw, "env: hardware has gpu_count")
    _assert("gpu_names" in _hw, "env: hardware has gpu_names")
    _assert("cpu" in _hw, "env: hardware has cpu")
    _assert("ram_mb" in _hw, "env: hardware has ram_mb")
    _assert(
        isinstance(_hw["gpu_count"], int),
        "env: gpu_count is int",
    )
    _assert(
        len(_hw["gpu_names"]) == _hw["gpu_count"],
        "env: gpu_names length matches gpu_count",
    )

    # Save pip_freeze
    _tmp_pip = tempfile.mktemp(suffix=".txt")
    _pip_ok = EnvironmentCapture.save_pip_freeze(_tmp_pip)
    _assert(isinstance(_pip_ok, bool), "env: save_pip_freeze returns bool")
    if _pip_ok:
        _assert(os.path.exists(_tmp_pip), "env: pip_freeze.txt written")
        os.unlink(_tmp_pip)

    # Save hardware JSON
    _tmp_hw = tempfile.mktemp(suffix=".json")
    _hw_ok = EnvironmentCapture.save_hardware_json(_tmp_hw)
    _assert(_hw_ok, "env: save_hardware_json succeeds")
    if _hw_ok:
        with open(_tmp_hw, "r") as _f:
            _hw_loaded = json.load(_f)
        _assert("gpu_count" in _hw_loaded, "env: hardware.json has gpu_count")
        os.unlink(_tmp_hw)

    # -----------------------------------------------------------------------
    # 10. RunIDGenerator
    # -----------------------------------------------------------------------
    print("\n--- RunIDGenerator ---")

    _ts = datetime.datetime(2026, 2, 20, 14, 30, 0)
    _rid = RunIDGenerator.generate_run_id(
        phase=4, mode="dev", git_short="abc1234", timestamp=_ts,
    )
    _assert(
        _rid == "2026-02-20_14-30-00_phase4_dev_abc1234",
        "run_id: format matches expected",
        f"got {_rid!r}",
    )

    _rid2 = RunIDGenerator.generate_run_id(
        phase=7, mode="production", git_short="xyz9999", timestamp=_ts,
    )
    _assert(
        "phase7" in _rid2 and "production" in _rid2,
        "run_id: phase7 production format",
    )

    # Default git_short
    _rid3 = RunIDGenerator.generate_run_id(phase=1, mode="dev")
    _assert(
        "0000000" in _rid3,
        "run_id: default git_short is 0000000",
    )

    # Uniqueness (different timestamps produce different IDs)
    _rid_a = RunIDGenerator.generate_run_id(
        phase=1, mode="dev", git_short="aaa1111",
        timestamp=datetime.datetime(2026, 1, 1, 0, 0, 0),
    )
    _rid_b = RunIDGenerator.generate_run_id(
        phase=1, mode="dev", git_short="aaa1111",
        timestamp=datetime.datetime(2026, 1, 1, 0, 0, 1),
    )
    _assert(_rid_a != _rid_b, "run_id: different timestamps produce different IDs")

    # Ablation run ID
    _abl_rid = RunIDGenerator.generate_ablation_run_id(
        ablation_id="3f7a2b1c",
        run_index=0,
        overrides={"use_engram": True, "use_htm": False},
        seed=42,
    )
    _assert("abl_3f7a2b1c" in _abl_rid, "run_id: ablation ID in generated ID")

    # Pipeline run ID
    _pipe_rid = RunIDGenerator.generate_pipeline_run_id(
        mode="dev", git_short="abc1234", timestamp=_ts,
    )
    _assert(
        _pipe_rid == "2026-02-20_14-30-00_pipeline_dev_abc1234",
        "run_id: pipeline format matches",
        f"got {_pipe_rid!r}",
    )

    # Derive ablation ID
    _abl_id_1 = RunIDGenerator.derive_ablation_id({"a": 1, "b": 2})
    _abl_id_2 = RunIDGenerator.derive_ablation_id({"b": 2, "a": 1})
    _assert(
        _abl_id_1 == _abl_id_2,
        "ablation_id: key order does not matter (sorted)",
    )
    _assert(
        len(_abl_id_1) == 8,
        "ablation_id: is 8 hex chars",
    )
    _abl_id_3 = RunIDGenerator.derive_ablation_id({"a": 1, "b": 3})
    _assert(
        _abl_id_1 != _abl_id_3,
        "ablation_id: different overrides produce different IDs",
    )

    # -----------------------------------------------------------------------
    # 11. ManifestComparator
    # -----------------------------------------------------------------------
    print("\n--- ManifestComparator ---")

    _orig = copy.deepcopy(_minimal_manifest)
    _repro = copy.deepcopy(_minimal_manifest)

    # 11a. Identical manifests
    _cmp = ManifestComparator.compare_for_reproduction(_orig, _repro)
    _assert(
        _cmp["match"] is True,
        "comparator: identical manifests match",
    )
    _assert(
        len(_cmp["mismatches"]) == 0,
        "comparator: no mismatches for identical",
    )
    _assert(
        ManifestComparator.is_reproducible(_orig, _repro),
        "comparator: is_reproducible returns True for identical",
    )

    # 11b. Different git commit
    _repro_diff = copy.deepcopy(_minimal_manifest)
    _repro_diff["git"]["commit"] = "b" * 40
    _cmp = ManifestComparator.compare_for_reproduction(_orig, _repro_diff)
    _assert(
        _cmp["match"] is False,
        "comparator: different git.commit is a mismatch",
    )
    _assert(
        any(m["field"] == "git.commit" for m in _cmp["mismatches"]),
        "comparator: git.commit flagged in mismatches",
    )

    # 11c. Different seeds
    _repro_seed = copy.deepcopy(_minimal_manifest)
    _repro_seed["seeds"]["base_seed"] = 9999
    _cmp = ManifestComparator.compare_for_reproduction(_orig, _repro_seed)
    _assert(
        _cmp["match"] is False,
        "comparator: different seed is a mismatch",
    )

    # 11d. Different config.resolved
    _repro_cfg = copy.deepcopy(_minimal_manifest)
    _repro_cfg["config"]["resolved"]["extra_key"] = True
    _cmp = ManifestComparator.compare_for_reproduction(_orig, _repro_cfg)
    _assert(
        _cmp["match"] is False,
        "comparator: different config.resolved is a mismatch",
    )

    # 11e. Ignorable differences (hostname, user, timestamps)
    _repro_ignore = copy.deepcopy(_minimal_manifest)
    _repro_ignore["identity"]["hostname"] = "different-host"
    _repro_ignore["identity"]["user"] = "different-user"
    _repro_ignore["identity"]["timestamp_start"] = "2099-01-01T00:00:00Z"
    _repro_ignore["identity"]["run_id"] = "different-run-id"
    _repro_ignore["results"]["status"] = "completed"
    _cmp = ManifestComparator.compare_for_reproduction(_orig, _repro_ignore)
    _assert(
        _cmp["match"] is True,
        "comparator: ignorable fields do not cause mismatch",
    )

    # 11f. Warning fields (different torch version)
    _repro_warn = copy.deepcopy(_minimal_manifest)
    _repro_warn["env"]["torch_version"] = "3.0.0"
    _cmp = ManifestComparator.compare_for_reproduction(_orig, _repro_warn)
    _assert(
        _cmp["match"] is True,
        "comparator: torch_version diff is warning, not mismatch",
    )
    _assert(
        len(_cmp["warnings"]) > 0,
        "comparator: torch_version diff produces warning",
    )
    _assert(
        any(w["field"] == "env.torch_version" for w in _cmp["warnings"]),
        "comparator: warning field is env.torch_version",
    )

    # 11g. Dataset fingerprint mismatch
    _orig_ds = copy.deepcopy(_minimal_manifest)
    _orig_ds["data"]["datasets"] = [{
        "name": "mnist", "split": "train", "fingerprint": "aaa",
        "transforms_signature": "sig1",
    }]
    _repro_ds = copy.deepcopy(_minimal_manifest)
    _repro_ds["data"]["datasets"] = [{
        "name": "mnist", "split": "train", "fingerprint": "bbb",
        "transforms_signature": "sig1",
    }]
    _cmp = ManifestComparator.compare_for_reproduction(_orig_ds, _repro_ds)
    _assert(
        _cmp["match"] is False,
        "comparator: different dataset fingerprint is mismatch",
    )

    # 11h. Missing dataset in one manifest
    _repro_ds_missing = copy.deepcopy(_minimal_manifest)
    _repro_ds_missing["data"]["datasets"] = []
    _cmp = ManifestComparator.compare_for_reproduction(_orig_ds, _repro_ds_missing)
    _assert(
        _cmp["match"] is False,
        "comparator: missing dataset is mismatch",
    )

    # 11i. Different phase
    _repro_phase = copy.deepcopy(_minimal_manifest)
    _repro_phase["identity"]["phase"] = 7
    _cmp = ManifestComparator.compare_for_reproduction(_orig, _repro_phase)
    _assert(
        _cmp["match"] is False,
        "comparator: different phase is mismatch",
    )

    # 11j. Different mode
    _repro_mode = copy.deepcopy(_minimal_manifest)
    _repro_mode["identity"]["mode"] = "production"
    _cmp = ManifestComparator.compare_for_reproduction(_orig, _repro_mode)
    _assert(
        _cmp["match"] is False,
        "comparator: different mode is mismatch",
    )

    # -----------------------------------------------------------------------
    # 12. RunDirectory
    # -----------------------------------------------------------------------
    print("\n--- RunDirectory ---")

    _tmp_base = tempfile.mkdtemp(prefix="run_dir_test_")
    try:
        _rd = RunDirectory.create(_tmp_base, "test_run_001")

        _assert(_rd.root.exists(), "run_dir: root exists")
        _assert(_rd.run_id == "test_run_001", "run_dir: run_id matches")
        _assert(
            _rd.manifest_path == _rd.root / "manifest.json",
            "run_dir: manifest_path correct",
        )
        _assert(
            _rd.lock_path == _rd.root / "manifest.lock.json",
            "run_dir: lock_path correct",
        )
        _assert(
            _rd.tensorboard_dir.exists(),
            "run_dir: tensorboard dir created",
        )
        _assert(
            _rd.wandb_dir.exists(),
            "run_dir: wandb dir created",
        )
        _assert(
            _rd.artifacts_dir.exists(),
            "run_dir: artifacts dir created",
        )
        _assert(
            _rd.reports_dir.exists(),
            "run_dir: reports dir created",
        )
        _assert(
            _rd.stdout_path == _rd.root / "stdout.log",
            "run_dir: stdout_path correct",
        )
        _assert(
            _rd.stderr_path == _rd.root / "stderr.log",
            "run_dir: stderr_path correct",
        )
        _assert(
            _rd.metrics_path == _rd.root / "reports" / "summary.json",
            "run_dir: metrics_path correct",
        )

        # Phase checkpoint dirs
        for _p in range(1, 8):
            _assert(
                _rd.checkpoints_dir(_p).exists(),
                f"run_dir: checkpoints/phase{_p} created",
            )

        # Cleanup test -- create fake checkpoints
        _ckpt_dir = _rd.checkpoints_dir(1)
        for _name in [
            "ckpt_step00001000.pt", "ckpt_step00002000.pt",
            "ckpt_step00003000.pt", "ckpt_best_val_loss.pt",
            "phase_boundary.pt",
        ]:
            (_ckpt_dir / _name).touch()

        _removed = _rd.cleanup(keep_best=True, keep_boundary=True, keep_last=True)
        _assert(
            _removed == 2,
            f"run_dir: cleanup removed 2 intermediate files, got {_removed}",
        )
        _assert(
            (_ckpt_dir / "ckpt_best_val_loss.pt").exists(),
            "run_dir: cleanup kept best checkpoint",
        )
        _assert(
            (_ckpt_dir / "phase_boundary.pt").exists(),
            "run_dir: cleanup kept phase_boundary",
        )
        _assert(
            (_ckpt_dir / "ckpt_step00003000.pt").exists(),
            "run_dir: cleanup kept last step checkpoint",
        )
        _assert(
            not (_ckpt_dir / "ckpt_step00001000.pt").exists(),
            "run_dir: cleanup removed step 1000",
        )
        _assert(
            not (_ckpt_dir / "ckpt_step00002000.pt").exists(),
            "run_dir: cleanup removed step 2000",
        )
    finally:
        shutil.rmtree(_tmp_base, ignore_errors=True)

    # -----------------------------------------------------------------------
    # 13. Integration: full save/load/validate cycle
    # -----------------------------------------------------------------------
    print("\n--- Integration ---")

    _tmp_int = tempfile.mkdtemp(prefix="manifest_integ_")
    try:
        _rd_int = RunDirectory.create(_tmp_int, "integ_run")
        _m_int = RunManifest.capture(
            config=_cfg,
            phase=4,
            mode="dev",
            run_id="2026-02-20_14-30-00_phase4_dev_abc1234",
            entrypoint="scripts/train_phase4.py",
            cli_args=["--mode", "dev"],
            datasets=[{
                "name": "mnist", "config": None, "split": "train",
                "version": "1.0.0", "fingerprint": "abc123",
                "fingerprint_tier": 1, "num_samples": 60000,
                "transforms_signature": "sha256:deadbeef",
            }],
        )

        # Save manifest and lock
        _m_int.save(_rd_int.root)
        _m_int.save_lock(_rd_int.root)

        _assert(
            _rd_int.manifest_path.exists(),
            "integration: manifest.json exists",
        )
        _assert(
            _rd_int.lock_path.exists(),
            "integration: manifest.lock.json exists",
        )

        # Load and validate
        _m_loaded = RunManifest.load(_rd_int.root)
        _errs = _m_loaded.validate()
        _assert(
            len(_errs) == 0,
            "integration: loaded manifest validates cleanly",
            f"errors: {_errs}",
        )

        # Finalize and re-save
        _m_loaded.finalize(
            status="completed",
            best_metrics={"val/loss": 0.04, "step": 800},
            final_metrics={"val/loss": 0.05, "step": 1000},
        )
        _m_loaded.save(_rd_int.root)

        # Re-load and verify
        _m_final = RunManifest.load(_rd_int.root)
        _assert(
            _m_final.status == "completed",
            "integration: finalized status persisted",
        )
        _assert(
            _m_final.data["identity"]["timestamp_end"] is not None,
            "integration: finalized timestamp_end persisted",
        )

        # Compare original capture vs loaded (should be reproducible)
        _cmp = ManifestComparator.compare_for_reproduction(
            _m_int.data, _m_loaded.data
        )
        _assert(
            _cmp["match"] is True,
            "integration: original and loaded are reproduction-equivalent",
        )
    finally:
        shutil.rmtree(_tmp_int, ignore_errors=True)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"Results: {_pass} passed, {_fail} failed, {_pass + _fail} total")
    print("=" * 72)
    if _fail > 0:
        print("\nFailed tests:")
        for e in _errors:
            print(e)
        sys.exit(1)
    else:
        print("\nAll tests passed.")
        sys.exit(0)
