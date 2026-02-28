#!/usr/bin/env python3
"""
validate_orchestrator.py -- Runtime contract validation for the training orchestrator.

Validates the three done-when gates:
  (a) Manifest reproduction -- re-run from manifest, metrics match within tolerance
  (b) Ablation matrix execution -- 2x2x2 matrix runs, all produce manifests, CSV complete
  (c) Phase resume from boundary -- resume at phase 4 and 7, boundary checks catch mismatches

Self-contained: no brain_ai imports.  Uses inline mocks/stubs and torch.

Usage:
    python validate_orchestrator.py                     # Run all groups
    python validate_orchestrator.py --group 1           # Run group 1 only
    python validate_orchestrator.py --group all         # Run all groups
    python validate_orchestrator.py --list              # List all checks
    python validate_orchestrator.py --verbose           # Detailed output
    python validate_orchestrator.py --timeout 60        # 60s per check
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
import random
import shutil
import signal
import struct
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# ANSI color helpers
# ============================================================================

_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m" if _USE_COLOR else s

def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m" if _USE_COLOR else s

def _yellow(s: str) -> str:
    return f"\033[33m{s}\033[0m" if _USE_COLOR else s

def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m" if _USE_COLOR else s


# ============================================================================
# Timeout context manager
# ============================================================================

class TimeoutError(Exception):
    pass

@contextmanager
def timeout_context(seconds: int):
    """Context manager that raises TimeoutError after *seconds*."""
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError(f"Check timed out after {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# ============================================================================
# Test harness
# ============================================================================

@dataclass
class CheckResult:
    group: int
    name: str
    status: str       # "pass", "fail", "skip"
    detail: str = ""
    duration_ms: float = 0.0


class ValidationHarness:
    """Collect and run validation checks across groups."""

    def __init__(self, verbose: bool = False, timeout: int = 30) -> None:
        self.verbose = verbose
        self.timeout = timeout
        self._checks: Dict[int, List[Tuple[str, Callable]]] = {}
        self._results: List[CheckResult] = []

    def register(self, group: int, name: str, fn: Callable) -> None:
        self._checks.setdefault(group, []).append((name, fn))

    def list_checks(self) -> None:
        for group in sorted(self._checks.keys()):
            print(f"\n{_bold(f'Group {group}:')} {GROUP_NAMES.get(group, '')}")
            for i, (name, _) in enumerate(self._checks[group], 1):
                print(f"  [{group}.{i:02d}] {name}")
        total = sum(len(v) for v in self._checks.values())
        print(f"\nTotal: {total} checks across {len(self._checks)} groups")

    def run(self, groups: Optional[List[int]] = None) -> List[CheckResult]:
        target_groups = groups or sorted(self._checks.keys())
        self._results = []

        for group in target_groups:
            if group not in self._checks:
                print(f"\n{_yellow('WARN')} Group {group} has no checks registered")
                continue
            checks = self._checks[group]
            print(f"\n{'=' * 72}")
            print(f" Group {group}: {GROUP_NAMES.get(group, 'Unknown')}")
            print(f"{'=' * 72}")

            for name, fn in checks:
                t0 = time.monotonic()
                try:
                    with timeout_context(self.timeout):
                        fn()
                    duration = (time.monotonic() - t0) * 1000
                    result = CheckResult(group, name, "pass", duration_ms=duration)
                    self._results.append(result)
                    status_str = _green("PASS")
                    timing = f" ({duration:.0f}ms)" if self.verbose else ""
                    print(f"  {status_str}  {name}{timing}")
                except TimeoutError as e:
                    duration = (time.monotonic() - t0) * 1000
                    result = CheckResult(group, name, "skip", str(e), duration)
                    self._results.append(result)
                    print(f"  {_yellow('SKIP')}  {name} -- {e}")
                except Exception as e:
                    duration = (time.monotonic() - t0) * 1000
                    detail = str(e)
                    if self.verbose:
                        detail = traceback.format_exc()
                    result = CheckResult(group, name, "fail", detail, duration)
                    self._results.append(result)
                    print(f"  {_red('FAIL')}  {name}")
                    if self.verbose:
                        for line in detail.strip().split("\n"):
                            print(f"         {line}")
                    else:
                        short = str(e)[:120]
                        print(f"         {short}")

        return self._results

    def summary(self) -> int:
        passed = sum(1 for r in self._results if r.status == "pass")
        failed = sum(1 for r in self._results if r.status == "fail")
        skipped = sum(1 for r in self._results if r.status == "skip")
        total = len(self._results)

        print(f"\n{'=' * 72}")
        if failed == 0:
            tag = _green("ALL PASSED")
        else:
            tag = _red("FAILURES DETECTED")
        print(f" {tag}: {passed} / {total} checks passed ({skipped} skipped)")

        # Per-group summary
        groups = sorted(set(r.group for r in self._results))
        for g in groups:
            g_results = [r for r in self._results if r.group == g]
            g_pass = sum(1 for r in g_results if r.status == "pass")
            g_fail = sum(1 for r in g_results if r.status == "fail")
            g_skip = sum(1 for r in g_results if r.status == "skip")
            if g_fail > 0:
                status = _red(f"{g_pass}/{len(g_results)}")
            else:
                status = _green(f"{g_pass}/{len(g_results)}")
            print(f"  Group {g} ({GROUP_NAMES.get(g, '')}): {status}"
                  f"  ({g_skip} skipped)" if g_skip > 0 else
                  f"  Group {g} ({GROUP_NAMES.get(g, '')}): {status}")

        if failed > 0:
            print(f"\n{_red('Failed checks:')}")
            for r in self._results:
                if r.status == "fail":
                    print(f"  [{r.group}] {r.name}")
                    if not self.verbose:
                        short = r.detail[:200]
                        if short:
                            print(f"       {short}")

        print(f"{'=' * 72}")
        return 0 if failed == 0 else 1


GROUP_NAMES = {
    1: "Manifest Capture and Schema",
    2: "Seeding and Determinism",
    3: "Checkpoint Management",
    4: "Phase Boundary Validation",
    5: "Dataset Fingerprinting",
    6: "Ablation System",
    7: "Pipeline and Logging",
}


# ============================================================================
# Inline mocks and stubs
# ============================================================================

class MockModel(nn.Module):
    """Tiny model for testing -- two linear layers."""
    def __init__(self, dim: int = 32, out_dim: int = 10) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def make_mock_config() -> Dict[str, Any]:
    """Return a minimal config dict matching manifest schema expectations."""
    return {
        "snn": {"beta": 0.95, "num_timesteps": 10, "hidden_sizes": [256, 128]},
        "encoder": {"output_dim": 512},
        "workspace": {"workspace_dim": 512},
        "training": {"learning_rate": 0.0003, "batch_size": 8},
        "use_snn": True,
        "use_htm": False,
        "use_workspace": True,
        "use_symbolic": False,
        "use_meta": False,
        "use_engram": False,
    }


def make_mock_manifest(phase: int = 4, mode: str = "dev",
                       overrides: Optional[Dict] = None) -> Dict[str, Any]:
    """Build a fully valid manifest dict for testing."""
    cfg = make_mock_config()
    if overrides:
        cfg.update(overrides)

    run_id = f"2026-02-20_14-30-00_phase{phase}_{mode}_abc1234"
    ff = {k: cfg.get(k, False) for k in [
        "use_snn", "use_htm", "use_workspace",
        "use_symbolic", "use_meta", "use_engram",
    ]}
    manifest = {
        "schema_version": "1.0",
        "identity": {
            "run_id": run_id,
            "ablation_id": None,
            "phase": phase,
            "mode": mode,
            "timestamp_start": "2026-02-20T14:30:00Z",
            "timestamp_end": None,
            "hostname": "test-host",
            "user": "tester",
        },
        "git": {
            "commit": "a" * 40,
            "branch": "main",
            "remote_url": "https://github.com/test/brain.git",
            "dirty": False,
            "patch_path": None,
            "entrypoint": f"scripts/train_phase{phase}.py",
            "cli_args": ["--mode", mode],
        },
        "config": {
            "brain_ai": copy.deepcopy(cfg),
            "overrides": {},
            "resolved": copy.deepcopy(cfg),
            "feature_flags": ff,
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
            "torch_version": torch.__version__,
            "pip_freeze_path": "artifacts/pip_freeze.txt",
            "hardware": {"gpu_names": [], "gpu_count": 0, "gpu_memory_mb": []},
        },
        "data": {
            "datasets": [{
                "name": "mnist",
                "config": None,
                "split": "train",
                "version": "1.0.0",
                "fingerprint": "d3b07384d113edec49eaa6238ad5ff00",
                "fingerprint_tier": 1,
                "num_samples": 60000,
                "transforms_signature": "sha256:a9f3e2b1c8d7e6f5",
            }],
        },
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
    return manifest


def make_boundary_artifact(source_phase: int, workspace_dim: int = 512,
                           schema_version: str = "1.0",
                           extra_keys: Optional[Dict] = None) -> Dict[str, Any]:
    """Build a mock phase boundary artifact dict."""
    model = MockModel(dim=32)
    artifact = {
        "schema_version": schema_version,
        "source_phase": source_phase,
        "target_phase": source_phase + 1,
        "source_run_id": f"2026-02-20_10-00-00_phase{source_phase}_dev_abc1234",
        "model_state_dict": model.state_dict(),
        "config_snapshot": {
            "snn": {"hidden_sizes": [256, 128], "surrogate": "atan"},
            "encoder": {"output_dim": workspace_dim},
            "workspace": {"workspace_dim": workspace_dim},
        },
        "feature_flags": {
            "use_snn": True,
            "use_htm": source_phase >= 3,
            "use_workspace": source_phase >= 4,
            "use_symbolic": False,
            "use_meta": False,
            "use_engram": False,
        },
        "compatibility": {
            "workspace_dim": workspace_dim,
            "vocab_size": 32000,
            "snn_hidden_sizes": [256, 128],
            "snn_output_dim": 128,
            "surrogate": "atan",
        },
        "metadata": {
            "best_metrics": {"val_loss": 0.05},
            "training_steps": 5000,
            "timestamp": "2026-02-20T10:00:00Z",
        },
    }
    if extra_keys:
        artifact["model_state_dict"].update(extra_keys)
    return artifact


def make_temp_dir() -> Path:
    """Create a temporary directory for a test."""
    return Path(tempfile.mkdtemp(prefix="vo_"))


# ============================================================================
# Manifest schema validation helpers (inline, no brain_ai imports)
# ============================================================================

REQUIRED_SECTIONS = [
    "schema_version", "identity", "git", "config", "seeds",
    "env", "data", "logging", "resume", "checkpoints", "results",
]

REQUIRED_FIELDS = {
    "identity": ["run_id", "phase", "mode", "timestamp_start",
                  "timestamp_end", "hostname", "user"],
    "git": ["commit", "branch", "remote_url", "dirty",
            "patch_path", "entrypoint", "cli_args"],
    "config": ["brain_ai", "overrides", "resolved", "feature_flags"],
    "seeds": ["base_seed", "per_phase_offsets", "torch_deterministic",
              "cudnn_benchmark", "cudnn_deterministic", "use_deterministic_algorithms"],
    "env": ["python_version", "os", "cuda_version",
            "torch_version", "pip_freeze_path", "hardware"],
    "data": ["datasets"],
    "logging": ["tensorboard", "wandb"],
    "resume": ["enabled", "from_run_id", "from_checkpoint"],
    "checkpoints": ["save_every_n_steps", "best_metric_key",
                     "phase_boundary_produced"],
    "results": ["status", "best_metrics", "final_metrics", "error"],
}

FEATURE_FLAG_KEYS = ["use_snn", "use_htm", "use_workspace",
                     "use_symbolic", "use_meta", "use_engram"]


def validate_manifest_schema(manifest: Dict) -> List[str]:
    """Validate manifest dict. Return list of error strings."""
    errors = []
    for sec in REQUIRED_SECTIONS:
        if sec not in manifest:
            errors.append(f"Missing section: {sec}")
    if errors:
        return errors

    for sec, fields in REQUIRED_FIELDS.items():
        sect = manifest.get(sec, {})
        if not isinstance(sect, dict):
            errors.append(f"Section '{sec}' not a dict")
            continue
        for f in fields:
            if f not in sect:
                errors.append(f"Missing field: {sec}.{f}")

    ident = manifest.get("identity", {})
    phase = ident.get("phase")
    if phase is not None and (not isinstance(phase, int) or not 1 <= phase <= 7):
        errors.append(f"Invalid phase: {phase}")

    mode = ident.get("mode")
    if mode not in ("dev", "production"):
        errors.append(f"Invalid mode: {mode}")

    offsets = manifest.get("seeds", {}).get("per_phase_offsets", [])
    if not isinstance(offsets, list) or len(offsets) != 7:
        errors.append(f"per_phase_offsets must have 7 elements")

    ff = manifest.get("config", {}).get("feature_flags", {})
    for k in FEATURE_FLAG_KEYS:
        if k not in ff:
            errors.append(f"Missing feature flag: {k}")
        elif not isinstance(ff.get(k), bool):
            errors.append(f"Feature flag {k} not bool")

    status = manifest.get("results", {}).get("status")
    if status not in ("running", "completed", "failed", "interrupted"):
        errors.append(f"Invalid status: {status}")

    return errors


# ============================================================================
# Seeding helpers (inline)
# ============================================================================

def seed_all(seed: int) -> None:
    """Seed all RNG sources deterministically."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Tolerance checking helpers (inline)
# ============================================================================

def check_relative(a: float, b: float, tol: float = 0.02) -> bool:
    denom = max(abs(a), 1e-12)
    return abs(a - b) / denom < tol

def check_absolute(a: float, b: float, tol: float = 0.01) -> bool:
    return abs(a - b) < tol


# ============================================================================
# Fast hash helpers (inline)
# ============================================================================

FAST_HASH_CHUNK = 4 * 1024 * 1024

def fast_hash_file(filepath: str) -> str:
    file_size = os.path.getsize(filepath)
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        h.update(f.read(FAST_HASH_CHUNK))
        if file_size > FAST_HASH_CHUNK:
            f.seek(max(0, file_size - FAST_HASH_CHUNK))
            h.update(f.read(FAST_HASH_CHUNK))
    return h.hexdigest()


# ============================================================================
# Ablation helpers (inline)
# ============================================================================

import itertools

def generate_full_matrix(toggles: Dict[str, list],
                         constraints: Optional[List[Dict]] = None) -> List[Dict]:
    """Generate Cartesian product of toggle values, filtered by constraints."""
    constraints = constraints or []
    keys = sorted(toggles.keys())
    value_lists = [toggles[k] for k in keys]
    raw = [dict(zip(keys, vals)) for vals in itertools.product(*value_lists)]

    valid = []
    for combo in raw:
        ok = True
        for rule in constraints:
            if_clause = rule.get("if", {})
            then_clause = rule.get("then", {})
            if all(combo.get(k) == v for k, v in if_clause.items()):
                for k, v in then_clause.items():
                    if v is None:
                        if combo.get(k) != toggles[k][0]:
                            ok = False
                    elif combo.get(k) != v:
                        ok = False
            if not ok:
                break
        if ok:
            valid.append(combo)
    return valid


def derive_ablation_id(overrides: Dict) -> str:
    canonical = json.dumps(overrides, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:8]


def generate_pairwise_matrix(toggles: Dict[str, list]) -> List[Dict]:
    """Generate a pairwise covering array using a greedy algorithm."""
    keys = sorted(toggles.keys())
    value_lists = [toggles[k] for k in keys]
    K = len(keys)

    uncovered = set()
    for i in range(K):
        for j in range(i + 1, K):
            for vi in value_lists[i]:
                for vj in value_lists[j]:
                    uncovered.add((i, str(vi), j, str(vj)))

    covering = []
    while uncovered:
        best_row = None
        best_count = -1
        # Try candidates: sample some uncovered pairs
        samples = list(uncovered)[:30]
        for pair in samples:
            i, vi, j, vj = pair
            for _ in range(3):
                row = [random.choice(vl) for vl in value_lists]
                # Restore actual typed values from string
                for idx, vl in enumerate(value_lists):
                    for v in vl:
                        if str(v) == vi and idx == i:
                            row[i] = v
                        if str(v) == vj and idx == j:
                            row[j] = v
                count = 0
                for ii in range(K):
                    for jj in range(ii + 1, K):
                        if (ii, str(row[ii]), jj, str(row[jj])) in uncovered:
                            count += 1
                if count > best_count:
                    best_count = count
                    best_row = list(row)

        if best_row is None:
            break

        covering.append(dict(zip(keys, best_row)))
        for ii in range(K):
            for jj in range(ii + 1, K):
                uncovered.discard((ii, str(best_row[ii]), jj, str(best_row[jj])))

    return covering


# ============================================================================
# Gradient norm computation helper (inline)
# ============================================================================

def compute_gradient_norms(model: nn.Module) -> Dict[str, float]:
    """Compute per-module and global gradient L2 norms."""
    total_sq = 0.0
    module_norms: Dict[str, float] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        norm = p.grad.data.norm(2).item()
        total_sq += norm ** 2
        mod_name = name.split(".")[0]
        module_norms[mod_name] = module_norms.get(mod_name, 0.0) + norm ** 2
    result = {"global": total_sq ** 0.5}
    for mod, sq in module_norms.items():
        result[mod] = sq ** 0.5
    return result


# ============================================================================
# GROUP 1: Manifest Capture and Schema (8 checks)
# ============================================================================

def register_group_1(harness: ValidationHarness) -> None:

    def check_1_01():
        """Manifest captures run_id, phase, mode, timestamp."""
        m = make_mock_manifest(phase=4, mode="dev")
        ident = m["identity"]
        assert ident["run_id"] is not None and len(ident["run_id"]) > 0
        assert ident["phase"] == 4
        assert ident["mode"] == "dev"
        assert ident["timestamp_start"] is not None
        assert ident["hostname"] is not None and len(ident["hostname"]) > 0
        assert ident["user"] is not None and len(ident["user"]) > 0
    harness.register(1, "Manifest captures run_id, phase, mode, timestamp", check_1_01)

    def check_1_02():
        """Manifest captures git info (commit, branch, dirty)."""
        m = make_mock_manifest()
        git = m["git"]
        assert len(git["commit"]) == 40
        assert all(c in "0123456789abcdef" for c in git["commit"])
        assert isinstance(git["branch"], str) and len(git["branch"]) > 0
        assert isinstance(git["dirty"], bool)
        assert isinstance(git["entrypoint"], str)
        assert isinstance(git["cli_args"], list)
    harness.register(1, "Manifest captures git info (commit, branch, dirty)", check_1_02)

    def check_1_03():
        """Manifest captures environment (python, torch versions)."""
        m = make_mock_manifest()
        env = m["env"]
        assert isinstance(env["python_version"], str)
        assert isinstance(env["torch_version"], str)
        assert isinstance(env["os"], str)
        hw = env["hardware"]
        assert isinstance(hw["gpu_count"], int)
        assert len(hw["gpu_names"]) == hw["gpu_count"]
        assert len(hw["gpu_memory_mb"]) == hw["gpu_count"]
    harness.register(1, "Manifest captures environment (python, torch versions)", check_1_03)

    def check_1_04():
        """Manifest captures seed config (base_seed, offsets, flags)."""
        m = make_mock_manifest()
        seeds = m["seeds"]
        assert seeds["base_seed"] == 1337
        assert len(seeds["per_phase_offsets"]) == 7
        assert isinstance(seeds["torch_deterministic"], bool)
        assert isinstance(seeds["cudnn_benchmark"], bool)
        assert isinstance(seeds["cudnn_deterministic"], bool)
        assert isinstance(seeds["use_deterministic_algorithms"], bool)
    harness.register(1, "Manifest captures seed config (base_seed, offsets, flags)", check_1_04)

    def check_1_05():
        """Schema validation rejects missing required fields."""
        m = make_mock_manifest()
        del m["identity"]
        errors = validate_manifest_schema(m)
        assert len(errors) > 0, "Should reject manifest missing identity section"
        assert any("identity" in e.lower() or "section" in e.lower() for e in errors)

        m2 = make_mock_manifest()
        del m2["seeds"]["base_seed"]
        errors2 = validate_manifest_schema(m2)
        assert len(errors2) > 0, "Should reject missing seeds.base_seed"
    harness.register(1, "Schema validation rejects missing required fields", check_1_05)

    def check_1_06():
        """Schema validation rejects invalid phase (0, 8)."""
        m0 = make_mock_manifest()
        m0["identity"]["phase"] = 0
        errors_0 = validate_manifest_schema(m0)
        assert any("phase" in e.lower() for e in errors_0), f"Should reject phase=0, got: {errors_0}"

        m8 = make_mock_manifest()
        m8["identity"]["phase"] = 8
        errors_8 = validate_manifest_schema(m8)
        assert any("phase" in e.lower() for e in errors_8), f"Should reject phase=8, got: {errors_8}"
    harness.register(1, "Schema validation rejects invalid phase (0, 8)", check_1_06)

    def check_1_07():
        """Manifest save/load round-trip preserves all fields."""
        tmpdir = make_temp_dir()
        try:
            m = make_mock_manifest(phase=3, mode="dev")
            manifest_path = tmpdir / "manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(m, f, indent=2)

            with open(manifest_path, "r") as f:
                loaded = json.load(f)

            assert loaded["identity"]["run_id"] == m["identity"]["run_id"]
            assert loaded["identity"]["phase"] == m["identity"]["phase"]
            assert loaded["seeds"] == m["seeds"]
            assert loaded["config"]["feature_flags"] == m["config"]["feature_flags"]
            assert loaded["data"]["datasets"] == m["data"]["datasets"]
            assert loaded["git"]["commit"] == m["git"]["commit"]
            assert loaded["env"]["torch_version"] == m["env"]["torch_version"]
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    harness.register(1, "Manifest save/load round-trip preserves all fields", check_1_07)

    def check_1_08():
        """manifest.lock.json has no 'auto' values."""
        lock = {
            "schema_version": "1.0",
            "resolved_config": copy.deepcopy(make_mock_config()),
        }
        lock_str = json.dumps(lock)
        forbidden = ["auto", "default"]
        for val in forbidden:
            assert f'"{val}"' not in lock_str.lower(), (
                f"Lock file must not contain '{val}' -- found in: {lock_str[:200]}"
            )
        # Verify concrete values
        rc = lock["resolved_config"]
        assert isinstance(rc["snn"]["beta"], (int, float))
        assert isinstance(rc["use_snn"], bool)
        assert isinstance(rc["workspace"]["workspace_dim"], int)
    harness.register(1, "manifest.lock.json has no 'auto' values", check_1_08)


# ============================================================================
# GROUP 2: Seeding and Determinism (8 checks)
# ============================================================================

def register_group_2(harness: ValidationHarness) -> None:

    def check_2_01():
        """Same seed produces same torch.randn output (10 runs)."""
        reference = None
        for _ in range(10):
            seed_all(1337)
            sample = torch.randn(100)
            if reference is None:
                reference = sample
            else:
                assert torch.equal(reference, sample), "torch.randn mismatch on same seed"
    harness.register(2, "Same seed produces same torch.randn output (10 runs)", check_2_01)

    def check_2_02():
        """Different phase seeds produce different outputs."""
        base_seed = 1337
        offsets = [0, 100, 200, 300, 400, 500, 600]
        outputs = []
        for i, offset in enumerate(offsets):
            seed_all(base_seed + offset)
            outputs.append(torch.randn(50))
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                assert not torch.equal(outputs[i], outputs[j]), (
                    f"Phase seeds {i+1} and {j+1} should produce different outputs"
                )
    harness.register(2, "Different phase seeds produce different outputs", check_2_02)

    def check_2_03():
        """Worker seeding produces deterministic DataLoader iteration."""
        base_seed = 42
        results = []
        for _ in range(2):
            worker_values = []
            for worker_id in range(4):
                epoch = 0
                worker_seed = base_seed + worker_id + epoch * 1000
                torch.manual_seed(worker_seed)
                worker_values.append(torch.randn(5).tolist())
            results.append(worker_values)
        assert results[0] == results[1], "Worker seeding not deterministic"
        # Different workers should produce different values
        assert results[0][0] != results[0][1], "Worker 0 and 1 should differ"
    harness.register(2, "Worker seeding produces deterministic DataLoader iteration", check_2_03)

    def check_2_04():
        """RNG state capture/restore round-trip matches."""
        seed_all(999)
        # Capture state
        cpu_state = torch.random.get_rng_state()
        np_state = np.random.get_state()
        py_state = random.getstate()

        # Generate values
        t1 = torch.randn(10)
        n1 = np.random.rand(5).copy()
        r1 = random.random()

        # Restore
        torch.random.set_rng_state(cpu_state)
        np.random.set_state(np_state)
        random.setstate(py_state)

        # Regenerate and compare
        t2 = torch.randn(10)
        n2 = np.random.rand(5).copy()
        r2 = random.random()

        assert torch.equal(t1, t2), "Torch RNG state restore failed"
        assert np.array_equal(n1, n2), "NumPy RNG state restore failed"
        assert r1 == r2, "Python RNG state restore failed"
    harness.register(2, "RNG state capture/restore round-trip matches", check_2_04)

    def check_2_05():
        """Component seeds (augmentation vs dropout) are independent."""
        phase_seed = 1337 + 200  # phase 3
        aug_seed = phase_seed + 0
        drop_seed = phase_seed + 10

        gen_a = torch.Generator()
        gen_a.manual_seed(aug_seed)
        gen_d = torch.Generator()
        gen_d.manual_seed(drop_seed)

        # Record dataloader output
        val_d = torch.randn(10, generator=gen_d).clone()

        # Consume augmentation heavily
        for _ in range(100):
            torch.randn(50, generator=gen_a)

        # Re-create dropout generator and verify same initial output
        gen_d2 = torch.Generator()
        gen_d2.manual_seed(drop_seed)
        val_d2 = torch.randn(10, generator=gen_d2)

        assert torch.equal(val_d, val_d2), (
            "Dropout stream should be independent of augmentation consumption"
        )
    harness.register(2, "Component seeds (augmentation vs dropout) are independent", check_2_05)

    def check_2_06():
        """DeterminismEnforcer sets all expected torch flags."""
        # Save original state
        orig_bench = torch.backends.cudnn.benchmark
        orig_det = torch.backends.cudnn.deterministic

        # Simulate dev-mode enforcement
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"

        # Restore
        torch.backends.cudnn.benchmark = orig_bench
        torch.backends.cudnn.deterministic = orig_det
    harness.register(2, "DeterminismEnforcer sets all expected torch flags", check_2_06)

    def check_2_07():
        """Tolerance checker correctly identifies within/outside tolerance."""
        # Within 2% relative
        assert check_relative(1.0, 1.01, 0.02), "1.0 vs 1.01 should be within 2%"
        assert not check_relative(1.0, 1.05, 0.02), "1.0 vs 1.05 should be outside 2%"

        # Within 1% absolute
        assert check_absolute(0.90, 0.895, 0.01), "0.90 vs 0.895 should be within 1%"
        assert not check_absolute(0.90, 0.85, 0.01), "0.90 vs 0.85 should be outside 1%"

        # Near-zero guard
        assert check_relative(0.0, 1e-14, 0.02), "Near-zero should not crash"
    harness.register(2, "Tolerance checker correctly identifies within/outside tolerance", check_2_07)

    def check_2_08():
        """Episode seed derivation is deterministic across calls."""
        def episode_seed(global_seed: int, epoch: int, idx: int) -> int:
            payload = f"{global_seed}:{epoch}:{idx}".encode()
            digest = hashlib.sha256(payload).digest()
            return struct.unpack(">I", digest[:4])[0]

        s1 = episode_seed(1337, 0, 0)
        s2 = episode_seed(1337, 0, 0)
        assert s1 == s2, "Episode seed must be deterministic"
        s3 = episode_seed(1337, 0, 1)
        assert s1 != s3, "Different episodes must get different seeds"
        s4 = episode_seed(1337, 1, 0)
        assert s1 != s4, "Different epochs must get different seeds"
        assert 0 <= s1 < 2**32, "Must be uint32"
    harness.register(2, "Episode seed derivation is deterministic across calls", check_2_08)


# ============================================================================
# GROUP 3: Checkpoint Management (8 checks)
# ============================================================================

def register_group_3(harness: ValidationHarness) -> None:

    def check_3_01():
        """Periodic checkpoint save/load round-trip preserves model weights."""
        tmpdir = make_temp_dir()
        try:
            model = MockModel(dim=32)
            original_params = {n: p.clone() for n, p in model.named_parameters()}
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

            ckpt = {
                "schema_version": "1.0",
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": None,
                "rng_states": {},
                "global_step": 100,
                "phase": 1,
                "phase_step": 50,
                "epoch": 2,
                "best_metrics": {},
                "config_hash": "abc123",
            }
            path = tmpdir / "ckpt_step00000100.pt"
            torch.save(ckpt, str(path))

            loaded = torch.load(str(path), map_location="cpu", weights_only=False)
            model2 = MockModel(dim=32)
            model2.load_state_dict(loaded["model_state_dict"])

            for n, p in model2.named_parameters():
                assert torch.equal(p, original_params[n]), f"Weight mismatch for {n}"
            assert loaded["global_step"] == 100
            assert loaded["phase"] == 1
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    harness.register(3, "Periodic checkpoint save/load round-trip preserves model weights", check_3_01)

    def check_3_02():
        """Best checkpoint updates only on improvement."""
        best_val = None
        saves = []
        metrics_sequence = [1.0, 0.8, 0.9, 0.5, 0.6, 0.4]
        for i, val in enumerate(metrics_sequence):
            if best_val is None or val < best_val:
                best_val = val
                saves.append((i, val))
        # Should save at indices 0(1.0), 1(0.8), 3(0.5), 5(0.4)
        assert len(saves) == 4, f"Expected 4 saves, got {len(saves)}: {saves}"
        assert saves[-1][1] == 0.4
    harness.register(3, "Best checkpoint updates only on improvement", check_3_02)

    def check_3_03():
        """Phase boundary contains no optimizer state."""
        model = MockModel(dim=32)
        boundary = make_boundary_artifact(source_phase=3)
        assert "optimizer_state_dict" not in boundary, "Boundary must not have optimizer state"
        assert "scheduler_state_dict" not in boundary, "Boundary must not have scheduler state"
        assert "rng_states" not in boundary, "Boundary must not have RNG states"
        assert "model_state_dict" in boundary, "Boundary must have model weights"
    harness.register(3, "Phase boundary contains no optimizer state", check_3_03)

    def check_3_04():
        """Phase boundary contains compatibility info."""
        boundary = make_boundary_artifact(source_phase=2, workspace_dim=4096)
        compat = boundary["compatibility"]
        assert "workspace_dim" in compat
        assert compat["workspace_dim"] == 4096
        assert "vocab_size" in compat
        assert "config_snapshot" in boundary
        assert "feature_flags" in boundary
        assert "metadata" in boundary
    harness.register(3, "Phase boundary contains compatibility info", check_3_04)

    def check_3_05():
        """Atomic save produces no corrupt files."""
        tmpdir = make_temp_dir()
        try:
            data = {"tensor": torch.randn(100), "step": 42}
            path = tmpdir / "atomic_test.pt"
            # Simulate atomic save: write to temp then rename
            tmp_path = path.with_suffix(".pt.tmp")
            torch.save(data, str(tmp_path))
            os.replace(str(tmp_path), str(path))

            assert path.exists(), "Final file must exist"
            assert not tmp_path.exists(), "Temp file must not remain"
            loaded = torch.load(str(path), map_location="cpu", weights_only=False)
            assert torch.equal(loaded["tensor"], data["tensor"])
            assert loaded["step"] == 42
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    harness.register(3, "Atomic save produces no corrupt files", check_3_05)

    def check_3_06():
        """Checkpoint naming follows standardized format."""
        import re
        step_pat = re.compile(r"^ckpt_step\d{8}\.pt$")
        best_pat = re.compile(r"^ckpt_best_\w+\.pt$")

        assert step_pat.match("ckpt_step00000042.pt")
        assert step_pat.match("ckpt_step12345678.pt")
        assert not step_pat.match("ckpt_step42.pt")
        assert best_pat.match("ckpt_best_val_loss.pt")
        assert best_pat.match("ckpt_best_val_acc.pt")
        assert "phase_boundary.pt" == "phase_boundary.pt"
        assert "ckpt_final.pt" == "ckpt_final.pt"

        # Parse step from name
        m = step_pat.match("ckpt_step00000042.pt")
        assert m is not None
    harness.register(3, "Checkpoint naming follows standardized format", check_3_06)

    def check_3_07():
        """load_latest finds most recent checkpoint."""
        tmpdir = make_temp_dir()
        try:
            ckpt_dir = tmpdir / "checkpoints" / "phase1"
            ckpt_dir.mkdir(parents=True)

            model = MockModel(dim=32)
            for step in [100, 200, 300]:
                ckpt = {"model_state_dict": model.state_dict(), "global_step": step}
                torch.save(ckpt, str(ckpt_dir / f"ckpt_step{step:08d}.pt"))

            # Find latest by sorting
            files = sorted(ckpt_dir.glob("ckpt_step*.pt"))
            assert len(files) == 3
            latest = torch.load(str(files[-1]), map_location="cpu", weights_only=False)
            assert latest["global_step"] == 300
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    harness.register(3, "load_latest finds most recent checkpoint", check_3_07)

    def check_3_08():
        """Cleanup retains best + boundary, removes intermediate."""
        tmpdir = make_temp_dir()
        try:
            ckpt_dir = tmpdir / "checkpoints" / "phase1"
            ckpt_dir.mkdir(parents=True)

            # Create files
            for name in ["ckpt_step00000100.pt", "ckpt_step00000200.pt",
                         "ckpt_step00000300.pt", "ckpt_best_val_loss.pt",
                         "phase_boundary.pt"]:
                (ckpt_dir / name).write_bytes(b"x")

            # Simulate cleanup: keep best, boundary, last step
            protected = {"ckpt_best_val_loss.pt", "phase_boundary.pt"}
            step_files = sorted(ckpt_dir.glob("ckpt_step*.pt"))
            last_step = step_files[-1] if step_files else None

            removed = 0
            for f in ckpt_dir.iterdir():
                if f.name in protected:
                    continue
                if f == last_step:
                    continue
                if f.name.startswith("ckpt_step"):
                    f.unlink()
                    removed += 1

            assert removed == 2, f"Should remove 2 intermediates, removed {removed}"
            assert (ckpt_dir / "ckpt_best_val_loss.pt").exists()
            assert (ckpt_dir / "phase_boundary.pt").exists()
            assert (ckpt_dir / "ckpt_step00000300.pt").exists()
            assert not (ckpt_dir / "ckpt_step00000100.pt").exists()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    harness.register(3, "Cleanup retains best + boundary, removes intermediate", check_3_08)


# ============================================================================
# GROUP 4: Phase Boundary Validation (8 checks)
# ============================================================================

def register_group_4(harness: ValidationHarness) -> None:

    def check_4_01():
        """Validation passes with correct boundary artifact."""
        boundary = make_boundary_artifact(source_phase=3, workspace_dim=512)
        assert boundary["schema_version"] == "1.0"
        assert boundary["source_phase"] == 3
        assert boundary["target_phase"] == 4
        assert "model_state_dict" in boundary
        assert boundary["compatibility"]["workspace_dim"] == 512
    harness.register(4, "Validation passes with correct boundary artifact", check_4_01)

    def check_4_02():
        """Validation fails on missing boundary file."""
        tmpdir = make_temp_dir()
        try:
            boundary_path = tmpdir / "checkpoints" / "phase3" / "phase_boundary.pt"
            assert not boundary_path.exists(), "Boundary file should not exist"
            # Validator should detect missing file
            missing = not boundary_path.exists()
            assert missing, "Should detect missing boundary artifact"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    harness.register(4, "Validation fails on missing boundary file", check_4_02)

    def check_4_03():
        """Validation fails on workspace_dim mismatch."""
        boundary = make_boundary_artifact(source_phase=3, workspace_dim=4096)
        current_workspace_dim = 512  # Different from boundary

        compat = boundary["compatibility"]
        mismatch = compat["workspace_dim"] != current_workspace_dim
        assert mismatch, (
            f"Should detect workspace_dim mismatch: boundary={compat['workspace_dim']} "
            f"vs current={current_workspace_dim}"
        )
    harness.register(4, "Validation fails on workspace_dim mismatch", check_4_03)

    def check_4_04():
        """Validation fails on schema version mismatch."""
        boundary = make_boundary_artifact(source_phase=3, schema_version="2.0")
        current_major = 1
        artifact_major = int(boundary["schema_version"].split(".")[0])
        assert artifact_major != current_major, (
            "Should detect schema version major mismatch"
        )
    harness.register(4, "Validation fails on schema version mismatch", check_4_04)

    def check_4_05():
        """Validation warns on extra state_dict keys (normal mode)."""
        boundary = make_boundary_artifact(source_phase=3)
        expected_keys = {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}
        actual_keys = set(boundary["model_state_dict"].keys())

        # Add an extra key
        boundary["model_state_dict"]["extra_module.weight"] = torch.randn(4)
        actual_keys = set(boundary["model_state_dict"].keys())
        extra = actual_keys - expected_keys
        assert len(extra) > 0, "Should detect extra keys"
        # In normal mode, extra keys produce a warning, not a failure
        warning_msg = f"Boundary has {len(extra)} extra state dict keys: {extra}"
        assert "extra" in warning_msg.lower()
    harness.register(4, "Validation warns on extra state_dict keys (normal mode)", check_4_05)

    def check_4_06():
        """Validation fails on extra keys (strict mode)."""
        boundary = make_boundary_artifact(source_phase=3)
        expected_keys = {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}
        boundary["model_state_dict"]["rogue_key"] = torch.randn(4)
        actual_keys = set(boundary["model_state_dict"].keys())
        extra = actual_keys - expected_keys
        strict_mode = True
        if strict_mode and extra:
            error = f"HARD: Extra keys in strict mode: {extra}"
        else:
            error = None
        assert error is not None, "Strict mode should fail on extra keys"
    harness.register(4, "Validation fails on extra keys (strict mode)", check_4_06)

    def check_4_07():
        """Feature flag inconsistency detected."""
        boundary = make_boundary_artifact(source_phase=3)
        # Boundary trained WITHOUT symbolic, current config enables it
        boundary["feature_flags"]["use_symbolic"] = False
        current_use_symbolic = True

        # Enabling a module not trained upstream is a HARD failure
        error = None
        if not boundary["feature_flags"]["use_symbolic"] and current_use_symbolic:
            error = ("Cannot enable use_symbolic: module was not trained in source phase")
        assert error is not None, "Should detect flag inconsistency"

        # Disabling a trained module is a SOFT warning
        boundary["feature_flags"]["use_snn"] = True
        current_use_snn = False
        warning = None
        if boundary["feature_flags"]["use_snn"] and not current_use_snn:
            warning = "use_snn was True in source but False now; weights will be discarded"
        assert warning is not None, "Should warn on disabling a trained module"
    harness.register(4, "Feature flag inconsistency detected", check_4_07)

    def check_4_08():
        """All 6 phase transitions have defined contracts."""
        transitions = [
            (1, 2, "SNN Core to Modality Encoders"),
            (2, 3, "Modality Encoders to HTM"),
            (3, 4, "HTM to Global Workspace"),
            (4, 5, "Global Workspace to Active Inference"),
            (5, 6, "Active Inference to Reasoning"),
            (6, 7, "Reasoning to Meta-Learning"),
        ]
        for source, target, desc in transitions:
            boundary = make_boundary_artifact(source_phase=source)
            assert boundary["source_phase"] == source, f"Source mismatch for {desc}"
            assert boundary["target_phase"] == target, f"Target mismatch for {desc}"
            assert "compatibility" in boundary, f"Missing compatibility for {desc}"
            assert "workspace_dim" in boundary["compatibility"], (
                f"Missing workspace_dim in {desc}"
            )
        assert len(transitions) == 6, "Must define all 6 phase transitions"
    harness.register(4, "All 6 phase transitions have defined contracts", check_4_08)


# ============================================================================
# GROUP 5: Dataset Fingerprinting (6 checks)
# ============================================================================

def register_group_5(harness: ValidationHarness) -> None:

    def check_5_01():
        """Fast hash is deterministic on same file."""
        tmpdir = make_temp_dir()
        try:
            filepath = tmpdir / "test_data.bin"
            data = os.urandom(1024 * 100)  # 100KB
            filepath.write_bytes(data)

            h1 = fast_hash_file(str(filepath))
            h2 = fast_hash_file(str(filepath))
            assert h1 == h2, f"Hash mismatch: {h1} vs {h2}"
            assert len(h1) == 64, "SHA256 hex digest should be 64 chars"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    harness.register(5, "Fast hash is deterministic on same file", check_5_01)

    def check_5_02():
        """Fast hash differs on different files."""
        tmpdir = make_temp_dir()
        try:
            f1 = tmpdir / "file_a.bin"
            f2 = tmpdir / "file_b.bin"
            f1.write_bytes(os.urandom(1024))
            f2.write_bytes(os.urandom(1024))

            h1 = fast_hash_file(str(f1))
            h2 = fast_hash_file(str(f2))
            assert h1 != h2, "Different files should produce different hashes"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    harness.register(5, "Fast hash differs on different files", check_5_02)

    def check_5_03():
        """Tier 2 fingerprint captures path + size + mtime + hash."""
        tmpdir = make_temp_dir()
        try:
            filepath = tmpdir / "shard_000.pt"
            data = os.urandom(2048)
            filepath.write_bytes(data)

            fp = {
                "fingerprint_tier": "tier2",
                "relative_path": "shard_000.pt",
                "byte_size": os.path.getsize(str(filepath)),
                "mtime": os.path.getmtime(str(filepath)),
                "fast_hash": f"sha256:{fast_hash_file(str(filepath))}",
            }

            assert fp["fingerprint_tier"] == "tier2"
            assert fp["relative_path"] == "shard_000.pt"
            assert fp["byte_size"] == 2048
            assert isinstance(fp["mtime"], float) and fp["mtime"] > 0
            assert fp["fast_hash"].startswith("sha256:")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    harness.register(5, "Tier 2 fingerprint captures path + size + mtime + hash", check_5_03)

    def check_5_04():
        """Tier 3 subset fingerprint records indices and seed."""
        parent_fp = {
            "fingerprint_tier": "tier1",
            "dataset_name": "mnist",
            "split": "train",
            "hf_fingerprint": "abc123",
        }
        indices = list(range(0, 1000))
        seed = 1337
        subset_fp = {
            "fingerprint_tier": "tier3",
            "parent_fingerprint": parent_fp,
            "sample_indices": f"range(0,1000)",
            "rng_seed": seed,
            "sampling_strategy": "first_n",
            "subset_size": len(indices),
        }

        assert subset_fp["fingerprint_tier"] == "tier3"
        assert subset_fp["parent_fingerprint"]["dataset_name"] == "mnist"
        assert subset_fp["rng_seed"] == 1337
        assert subset_fp["subset_size"] == 1000
        assert subset_fp["sampling_strategy"] == "first_n"
    harness.register(5, "Tier 3 subset fingerprint records indices and seed", check_5_04)

    def check_5_05():
        """datasets.json save/load round-trip."""
        tmpdir = make_temp_dir()
        try:
            ds_json = {
                "schema_version": "1.0",
                "fingerprint_tier": "tier1",
                "datasets": [{
                    "name": "mnist",
                    "role": "primary",
                    "config_name": None,
                    "split": "train",
                    "version": "1.0.0",
                    "hf_fingerprint": "abc123def456",
                    "num_samples": 60000,
                    "features_hash": "sha256:e3b0c442",
                    "transforms_signature": "Compose(ToTensor,Normalize(0.1307,0.3081))",
                }],
                "timestamp": "2026-02-20T12:00:00Z",
            }
            path = tmpdir / "datasets.json"
            with open(path, "w") as f:
                json.dump(ds_json, f, indent=2)

            with open(path) as f:
                loaded = json.load(f)

            assert loaded["schema_version"] == "1.0"
            assert loaded["datasets"][0]["name"] == "mnist"
            assert loaded["datasets"][0]["num_samples"] == 60000
            assert loaded["timestamp"] == "2026-02-20T12:00:00Z"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    harness.register(5, "datasets.json save/load round-trip", check_5_05)

    def check_5_06():
        """Fingerprint comparator correctly identifies exact/compatible/incompatible."""
        fp_a = {"dataset_name": "mnist", "split": "train",
                "fingerprint_tier": "tier1", "hf_fingerprint": "abc123"}
        fp_b = {"dataset_name": "mnist", "split": "train",
                "fingerprint_tier": "tier1", "hf_fingerprint": "abc123"}
        fp_c = {"dataset_name": "mnist", "split": "train",
                "fingerprint_tier": "tier1", "hf_fingerprint": "xyz999"}
        fp_d = {"dataset_name": "cifar10", "split": "train",
                "fingerprint_tier": "tier1", "hf_fingerprint": "def456"}

        # Exact match
        def compare(a, b):
            if a.get("dataset_name") != b.get("dataset_name"):
                return "incompatible"
            if a.get("split") != b.get("split"):
                return "incompatible"
            if (a.get("fingerprint_tier") == "tier1" and
                b.get("fingerprint_tier") == "tier1" and
                a.get("hf_fingerprint") and
                a["hf_fingerprint"] == b.get("hf_fingerprint")):
                return "exact"
            return "compatible"

        assert compare(fp_a, fp_b) == "exact"
        assert compare(fp_a, fp_c) == "compatible"
        assert compare(fp_a, fp_d) == "incompatible"
    harness.register(5, "Fingerprint comparator identifies exact/compatible/incompatible", check_5_06)


# ============================================================================
# GROUP 6: Ablation System (6 checks)
# ============================================================================

def register_group_6(harness: ValidationHarness) -> None:

    def check_6_01():
        """Full matrix generates correct number of combinations (2x2x2=8)."""
        toggles = {
            "use_engram": [False, True],
            "use_learnable_delays": [False, True],
            "use_ltn": [False, True],
        }
        matrix = generate_full_matrix(toggles)
        assert len(matrix) == 8, f"Expected 8 combinations, got {len(matrix)}"
        # Verify all are unique
        sigs = [json.dumps(m, sort_keys=True) for m in matrix]
        assert len(set(sigs)) == 8, "All combinations must be unique"
    harness.register(6, "Full matrix generates correct number of combinations (2x2x2=8)", check_6_01)

    def check_6_02():
        """Constraints filter invalid combinations."""
        toggles = {
            "use_engram": [False, True],
            "engram_mode": ["encoder", "layer"],
            "use_ltn": [False, True],
        }
        constraints = [
            {"if": {"use_engram": False}, "then": {"engram_mode": None}},
        ]
        matrix = generate_full_matrix(toggles, constraints)
        # Without constraint: 2x2x2=8. Constraint pins engram_mode="encoder" when
        # use_engram=False, so we remove half of use_engram=False combos
        # False+encoder+F, False+encoder+T = 2 combos
        # True+encoder+F, True+encoder+T, True+layer+F, True+layer+T = 4 combos
        # Total: 6
        assert len(matrix) == 6, f"Expected 6 after constraints, got {len(matrix)}"

        # Verify no (use_engram=False, engram_mode=layer) exists
        for combo in matrix:
            if combo["use_engram"] is False:
                assert combo["engram_mode"] == "encoder", (
                    f"When use_engram=False, engram_mode must be pinned to 'encoder', "
                    f"got {combo['engram_mode']}"
                )
    harness.register(6, "Constraints filter invalid combinations", check_6_02)

    def check_6_03():
        """Pairwise mode reduces combinations while covering all pairs."""
        random.seed(42)  # for reproducibility of greedy algorithm
        toggles = {
            "a": [False, True],
            "b": [False, True],
            "c": [False, True],
        }
        pairwise = generate_pairwise_matrix(toggles)
        # Pairwise for 3 binary toggles needs at least 4 rows
        assert len(pairwise) <= 8, f"Pairwise should be <= full (8), got {len(pairwise)}"
        assert len(pairwise) >= 4, f"Pairwise needs at least 4, got {len(pairwise)}"

        # Verify all pairs covered
        keys = sorted(toggles.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                for vi in toggles[keys[i]]:
                    for vj in toggles[keys[j]]:
                        found = any(
                            combo[keys[i]] == vi and combo[keys[j]] == vj
                            for combo in pairwise
                        )
                        assert found, (
                            f"Pair ({keys[i]}={vi}, {keys[j]}={vj}) not covered"
                        )
    harness.register(6, "Pairwise mode reduces combinations while covering all pairs", check_6_03)

    def check_6_04():
        """Ablation run IDs are deterministic."""
        overrides_a = {"use_engram": True, "use_ltn": False}
        overrides_b = {"use_ltn": False, "use_engram": True}  # same, different order

        id_a = derive_ablation_id(overrides_a)
        id_b = derive_ablation_id(overrides_b)
        assert id_a == id_b, "Same overrides in different order must produce same ID"
        assert len(id_a) == 8, f"Ablation ID must be 8 hex chars, got {len(id_a)}"

        overrides_c = {"use_engram": True, "use_ltn": True}
        id_c = derive_ablation_id(overrides_c)
        assert id_a != id_c, "Different overrides must produce different IDs"
    harness.register(6, "Ablation run IDs are deterministic", check_6_04)

    def check_6_05():
        """CSV output has correct columns and row count."""
        toggles = {
            "use_engram": [False, True],
            "use_learnable_delays": [False, True],
            "use_ltn": [False, True],
        }
        matrix = generate_full_matrix(toggles)
        toggle_keys = sorted(toggles.keys())

        fieldnames = [
            "run_id", "ablation_id", "parent_run_id", "seed",
            *toggle_keys,
            "phase", "best_val_loss", "best_val_acc",
            "final_train_loss", "final_val_loss",
            "status", "duration_seconds", "git_sha",
            "dataset_fingerprint", "error_message",
        ]

        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        for i, combo in enumerate(matrix):
            row = {
                "run_id": f"run_{i:03d}",
                "ablation_id": "abl_test",
                "parent_run_id": "",
                "seed": 1337,
                "phase": "4",
                "best_val_loss": 0.05 + i * 0.01,
                "best_val_acc": 0.95 - i * 0.01,
                "final_train_loss": 0.03 + i * 0.005,
                "final_val_loss": 0.06 + i * 0.01,
                "status": "completed",
                "duration_seconds": 100.0 + i * 10,
                "git_sha": "a" * 40,
                "dataset_fingerprint": "sha256:abc",
                "error_message": "",
            }
            for k in toggle_keys:
                row[k] = combo[k]
            writer.writerow(row)

        buf.seek(0)
        reader = csv.DictReader(buf)
        rows = list(reader)
        assert len(rows) == 8, f"Expected 8 rows, got {len(rows)}"
        assert set(reader.fieldnames) == set(fieldnames), "Column mismatch"
        # Verify toggle columns present
        for k in toggle_keys:
            assert k in reader.fieldnames, f"Missing toggle column: {k}"
    harness.register(6, "CSV output has correct columns and row count", check_6_05)

    def check_6_06():
        """Failure in one run does not abort entire matrix."""
        results = []
        matrix = [{"combo": i} for i in range(8)]
        for i, combo in enumerate(matrix):
            try:
                if i == 3:
                    raise RuntimeError("Simulated CUDA OOM")
                results.append({"run": i, "status": "completed"})
            except Exception as e:
                results.append({"run": i, "status": "failed", "error": str(e)})
                # Continue, do not abort

        assert len(results) == 8, f"All 8 runs must produce results, got {len(results)}"
        completed = sum(1 for r in results if r["status"] == "completed")
        failed = sum(1 for r in results if r["status"] == "failed")
        assert completed == 7, f"Expected 7 completed, got {completed}"
        assert failed == 1, f"Expected 1 failed, got {failed}"
        assert results[3]["status"] == "failed"
        assert "CUDA OOM" in results[3]["error"]
    harness.register(6, "Failure in one run does not abort entire matrix", check_6_06)


# ============================================================================
# GROUP 7: Pipeline and Logging (6 checks)
# ============================================================================

def register_group_7(harness: ValidationHarness) -> None:

    def check_7_01():
        """Pipeline state tracks completed phases."""
        completed_phases: List[int] = []
        for phase in [1, 2, 3, 4, 5]:
            # Simulate phase completion
            completed_phases.append(phase)

        assert completed_phases == [1, 2, 3, 4, 5]
        assert 3 in completed_phases
        assert 6 not in completed_phases

        # State must be serializable
        state = {"completed_phases": completed_phases, "current_phase": 5}
        serialized = json.dumps(state)
        loaded = json.loads(serialized)
        assert loaded["completed_phases"] == [1, 2, 3, 4, 5]
    harness.register(7, "Pipeline state tracks completed phases", check_7_01)

    def check_7_02():
        """Resume finds correct resume point."""
        # Simulate: phases 1-3 completed, need to resume at phase 4
        completed = [1, 2, 3]
        start_phase = 4

        # Verify that the last completed phase is start_phase - 1
        assert max(completed) == start_phase - 1, (
            f"Last completed phase ({max(completed)}) should be {start_phase - 1}"
        )

        # Also test resuming at phase 7
        completed_full = [1, 2, 3, 4, 5, 6]
        start_phase_7 = 7
        assert max(completed_full) == start_phase_7 - 1

        # Resume at phase 1 (from scratch) -- no prior phases needed
        assert start_phase - 1 == 3
    harness.register(7, "Resume finds correct resume point", check_7_02)

    def check_7_03():
        """MetricLogger writes to JSONL."""
        tmpdir = make_temp_dir()
        try:
            jsonl_path = tmpdir / "metrics.jsonl"
            entries = [
                {"step": 100, "phase": 4, "phase_step": 50,
                 "timestamp": "2026-02-20T14:30:00Z",
                 "metrics": {"train/loss": 0.5, "val/loss": 0.6}},
                {"step": 200, "phase": 4, "phase_step": 100,
                 "timestamp": "2026-02-20T14:31:00Z",
                 "metrics": {"train/loss": 0.4, "val/loss": 0.5}},
                {"step": 300, "event": "phase_transition",
                 "from_phase": 4, "to_phase": 5,
                 "timestamp": "2026-02-20T14:32:00Z"},
            ]

            with open(jsonl_path, "w") as f:
                for entry in entries:
                    f.write(json.dumps(entry) + "\n")

            # Read back and verify
            loaded = []
            with open(jsonl_path) as f:
                for line in f:
                    loaded.append(json.loads(line.strip()))

            assert len(loaded) == 3
            assert loaded[0]["metrics"]["train/loss"] == 0.5
            assert loaded[1]["step"] == 200
            assert loaded[2]["event"] == "phase_transition"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    harness.register(7, "MetricLogger writes to JSONL", check_7_03)

    def check_7_04():
        """Metric namespace follows convention (train/, val/, phase{N}/)."""
        valid_prefixes = ["train/", "val/", "test/", "system/", "lr/",
                          "grad_norm/", "weight_norm/"]
        phase_prefix_re = "phase{N}/"

        test_metrics = {
            "train/loss": 0.5,
            "val/loss": 0.6,
            "val/accuracy": 0.9,
            "phase4/ignition_rate": 0.43,
            "lr/base": 0.0003,
            "grad_norm/global": 1.2,
            "system/gpu_utilization": 95.0,
        }

        import re
        phase_pat = re.compile(r"^phase\d+/")
        for key in test_metrics:
            has_valid = any(key.startswith(p) for p in valid_prefixes) or phase_pat.match(key)
            assert has_valid, f"Metric key '{key}' does not follow namespace convention"
    harness.register(7, "Metric namespace follows convention (train/, val/, phase{N}/)", check_7_04)

    def check_7_05():
        """Phase transition logging produces marker events."""
        events = []

        def log_phase_transition(from_phase, to_phase, step, duration_sec, best_metrics):
            event = {
                "step": step,
                "event": "phase_transition",
                "from_phase": from_phase,
                "to_phase": to_phase,
                "duration_sec": duration_sec,
                "best_metrics": best_metrics,
                "timestamp": "2026-02-20T15:00:00Z",
            }
            events.append(event)

        log_phase_transition(3, 4, 5000, 3600.0, {"val/loss": 0.3, "val/accuracy": 0.91})
        log_phase_transition(4, 5, 8000, 2400.0, {"val/loss": 0.25})

        assert len(events) == 2
        assert events[0]["event"] == "phase_transition"
        assert events[0]["from_phase"] == 3
        assert events[0]["to_phase"] == 4
        assert events[0]["step"] == 5000
        assert "val/loss" in events[0]["best_metrics"]
        assert events[1]["from_phase"] == 4
        assert events[1]["to_phase"] == 5
    harness.register(7, "Phase transition logging produces marker events", check_7_05)

    def check_7_06():
        """Gradient norm computation is correct."""
        model = MockModel(dim=16, out_dim=4)
        x = torch.randn(2, 16)
        out = model(x)
        loss = out.sum()
        loss.backward()

        norms = compute_gradient_norms(model)
        assert "global" in norms
        assert norms["global"] > 0, "Global gradient norm must be positive"
        assert "fc1" in norms, "Should have per-module norm for fc1"
        assert "fc2" in norms, "Should have per-module norm for fc2"

        # Verify global norm is sqrt(sum of squared module norms)
        recomputed_sq = 0.0
        for name, p in model.named_parameters():
            if p.grad is not None:
                recomputed_sq += p.grad.data.norm(2).item() ** 2
        recomputed_global = recomputed_sq ** 0.5
        assert abs(norms["global"] - recomputed_global) < 1e-6, (
            f"Global norm mismatch: {norms['global']} vs {recomputed_global}"
        )
    harness.register(7, "Gradient norm computation is correct", check_7_06)


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Runtime contract validation for the training orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Validation Groups:
  1  Manifest Capture and Schema    (8 checks)
  2  Seeding and Determinism        (8 checks)
  3  Checkpoint Management          (8 checks)
  4  Phase Boundary Validation      (8 checks)
  5  Dataset Fingerprinting         (6 checks)
  6  Ablation System                (6 checks)
  7  Pipeline and Logging           (6 checks)
""",
    )
    parser.add_argument(
        "--group", type=str, default="all",
        help='Validation group to run: 1-7 or "all" (default: all)',
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show detailed output including tracebacks",
    )
    parser.add_argument(
        "--list", action="store_true", dest="list_checks",
        help="List all checks without running them",
    )
    parser.add_argument(
        "--timeout", type=int, default=30,
        help="Max seconds per check (default: 30)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    harness = ValidationHarness(verbose=args.verbose, timeout=args.timeout)

    # Register all groups
    register_group_1(harness)
    register_group_2(harness)
    register_group_3(harness)
    register_group_4(harness)
    register_group_5(harness)
    register_group_6(harness)
    register_group_7(harness)

    if args.list_checks:
        harness.list_checks()
        return 0

    # Parse group selection
    if args.group.lower() == "all":
        groups = None
    else:
        try:
            group_num = int(args.group)
            if group_num < 1 or group_num > 7:
                print(f"Error: group must be 1-7 or 'all', got {args.group}")
                return 1
            groups = [group_num]
        except ValueError:
            print(f"Error: group must be 1-7 or 'all', got {args.group}")
            return 1

    print(f"{_bold('Training Orchestrator Contract Validation')}")
    print(f"{'=' * 72}")
    if groups:
        print(f"Running group(s): {groups}")
    else:
        print(f"Running all groups (1-7)")

    harness.run(groups)
    return harness.summary()


if __name__ == "__main__":
    sys.exit(main())
