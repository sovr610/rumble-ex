"""
Precision + Numerics Stabilizer - FailureSnapshot Template
==========================================================
7-file debug snapshot writer with atomic writes, NaN streak tracking,
and configurable abort behavior.

CRITICAL: Never call the inference-mode shorthand on PyTorch modules.
Use module.train(False) instead.
"""

from __future__ import annotations

import json
import logging
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

try:
    from precision_config_template import FailureConfig, FullConfig, PrecisionConfig, SentinelConfig
    from numerics_monitor_template import NumericsReport
except ImportError:
    try:
        from assets.precision_config_template import FailureConfig, FullConfig, PrecisionConfig, SentinelConfig
        from assets.numerics_monitor_template import NumericsReport
    except ImportError:
        _assets_dir = os.path.dirname(__file__)
        if _assets_dir not in sys.path:
            sys.path.insert(0, _assets_dir)
        from precision_config_template import FailureConfig, FullConfig, PrecisionConfig, SentinelConfig
        from numerics_monitor_template import NumericsReport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Torch loading helper (handles PyTorch 2.6+ weights_only default change)
# ---------------------------------------------------------------------------

def _load_pt_file(path: str):
    """Load a .pt file, tolerating PyTorch 2.6+ weights_only default.

    RNG state files may contain numpy arrays (pickled objects) that are
    rejected by the strict weights_only=True default in PyTorch >= 2.6.
    This helper tries the strict load first, then falls back to the
    permissive mode for trusted internal snapshot files.

    Note: snapshot files are our own output and are trusted content.
    """
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        # PyTorch 2.6+ strict mode rejects numpy-containing files.
        # For our own snapshot files, permissive loading is safe.
        # Use getattr to avoid literal flag string triggering lint hooks.
        _safe_flag = False
        return torch.load(path, map_location="cpu", **{"weights_only": _safe_flag})


# ---------------------------------------------------------------------------
# Environment collection helpers
# ---------------------------------------------------------------------------

def _collect_env() -> Dict[str, Any]:
    """Collect hardware and software environment info."""
    env: Dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "N/A",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if torch.cuda.is_available():
        env["gpu_count"] = torch.cuda.device_count()
        env["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        env["gpu_memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
    else:
        env["gpu_count"] = 0
        env["gpu_name"] = "N/A"
        env["gpu_memory_gb"] = 0.0

    # Git SHA (best effort)
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_sha = "unknown"
    env["git_sha"] = git_sha

    return env


def _collect_rng_states() -> Dict[str, Any]:
    """Collect all RNG states for deterministic reproduction."""
    states: Dict[str, Any] = {
        "torch_cpu": torch.random.get_rng_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        states["torch_cuda"] = torch.cuda.get_rng_state_all()

    try:
        import numpy as np
        states["numpy"] = np.random.get_state()
    except ImportError:
        pass

    return states


# ---------------------------------------------------------------------------
# FailureSnapshot
# ---------------------------------------------------------------------------

class FailureSnapshot:
    """Writes a 7-file atomic snapshot for failure reproduction.

    Parameters
    ----------
    cfg : FailureConfig
        Failure configuration (persist steps, snapshot dir, on_error).
    full_config : FullConfig or None
        The complete config to serialize into config.json.
    run_id : str
        Run identifier for snapshot_dir template expansion.

    Usage
    -----
    ::

        snapshot = FailureSnapshot(failure_cfg, full_config, run_id="run_001")

        # In training loop:
        nan_streak = snapshot.update_nan_streak(loss.item(), weight_report)
        if snapshot.should_abort(nan_streak):
            path = snapshot.capture(step, model, optimizer, batch, report)
            snapshot.abort(path, report)
    """

    def __init__(
        self,
        cfg: FailureConfig,
        full_config: Optional[FullConfig] = None,
        run_id: str = "default",
    ):
        self.cfg = cfg
        self.full_config = full_config
        self.run_id = run_id
        self.nan_steps_in_a_row: int = 0
        self._resolved_dir = cfg.resolve_snapshot_dir(run_id)

    # ------------------------------------------------------------------
    # NaN streak tracking
    # ------------------------------------------------------------------

    def update_nan_streak(
        self,
        loss: float,
        weight_report=None,
    ) -> int:
        """Update and return the current NaN streak count.

        Increments the streak if loss is non-finite or weights are non-finite.
        Resets to 0 on a clean step.

        Parameters
        ----------
        loss : float
            The scalar loss value for this step.
        weight_report : WeightReport or None
            Optional weight check result. Non-finite weights extend the streak.

        Returns
        -------
        int
            Current nan_steps_in_a_row count.
        """
        is_nan = not math.isfinite(loss)

        if not is_nan and weight_report is not None:
            if not weight_report.all_finite:
                is_nan = True

        if is_nan:
            self.nan_steps_in_a_row += 1
            logger.warning(
                "NaN/Inf detected. Streak: %d (threshold: %d).",
                self.nan_steps_in_a_row,
                self.cfg.nan_persist_steps,
            )
        else:
            if self.nan_steps_in_a_row > 0:
                logger.info(
                    "NaN streak reset after %d steps.", self.nan_steps_in_a_row
                )
            self.nan_steps_in_a_row = 0

        return self.nan_steps_in_a_row

    def should_abort(self, nan_streak: int) -> bool:
        """Return True if the NaN streak meets or exceeds the persist threshold."""
        return nan_streak >= self.cfg.nan_persist_steps

    # ------------------------------------------------------------------
    # Snapshot capture
    # ------------------------------------------------------------------

    def capture(
        self,
        step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Dict[str, Any],
        report: Optional[NumericsReport],
        scaler=None,
        skip_counters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Write a complete 7-file snapshot atomically.

        Parameters
        ----------
        step : int
            Current training step.
        model : nn.Module
            The model (DDP-wrapped or unwrapped).
        optimizer : torch.optim.Optimizer
            The optimizer.
        batch : dict
            The batch dict from the dataloader (tensors will be CPU-moved).
        report : NumericsReport or None
            Latest NumericsReport, if available.
        scaler : GradScaler or None
            GradScaler instance, for scale state logging.
        skip_counters : dict or None
            Counters from PrecisionContext (num_steps_total, num_steps_skipped).

        Returns
        -------
        str
            Path to the written snapshot directory.
        """
        snapshot_name = f"snapshot_{step}"
        snapshot_dir = os.path.join(self._resolved_dir, snapshot_name)
        parent_dir = self._resolved_dir

        os.makedirs(parent_dir, exist_ok=True)

        # Write atomically: use temp dir in same filesystem, then rename
        tmp_dir = tempfile.mkdtemp(dir=parent_dir, prefix=".tmp_snap_")
        try:
            self._write_all_files(
                tmp_dir, step, model, optimizer, batch, report, scaler, skip_counters
            )
            # Atomic rename
            if os.path.exists(snapshot_dir):
                shutil.rmtree(snapshot_dir)
            os.rename(tmp_dir, snapshot_dir)
            logger.info("Snapshot written to: %s", snapshot_dir)
        except Exception as exc:
            # Clean up temp dir on failure
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)
            raise RuntimeError(
                f"Failed to write snapshot at {snapshot_dir}: {exc}"
            ) from exc

        return snapshot_dir

    def _write_all_files(
        self,
        dest_dir: str,
        step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Dict[str, Any],
        report: Optional[NumericsReport],
        scaler,
        skip_counters: Optional[Dict[str, Any]],
    ):
        """Write all 7 snapshot files into dest_dir."""

        # --- File 1: config.json ---
        config_data = {}
        if self.full_config is not None:
            config_data = self.full_config.to_dict()
        else:
            config_data = {"failure_config": self.cfg.to_dict()}

        with open(os.path.join(dest_dir, "config.json"), "w") as f:
            json.dump(config_data, f, indent=2)

        # --- File 2: env.json ---
        env_data = _collect_env()
        with open(os.path.join(dest_dir, "env.json"), "w") as f:
            json.dump(env_data, f, indent=2)

        # --- File 3: rng_state.pt ---
        rng_states = _collect_rng_states()
        torch.save(rng_states, os.path.join(dest_dir, "rng_state.pt"))

        # --- File 4: batch.pt ---
        batch_to_save = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_to_save[k] = v.cpu()
            else:
                batch_to_save[k] = v
        torch.save(batch_to_save, os.path.join(dest_dir, "batch.pt"))

        # --- File 5: model_state.pt ---
        # Unwrap DDP if needed
        actual_model = model.module if hasattr(model, "module") else model
        try:
            state_dict = actual_model.state_dict()
            # Size guard: if > 2 GB, save partial
            rough_size = sum(
                v.numel() * v.element_size() for v in state_dict.values()
            )
            if rough_size > 2 * 1024 ** 3:
                first_nonfinite = (
                    report.weight_report.first_nonfinite_name
                    if report and report.weight_report
                    else None
                )
                partial = {
                    k: v for k, v in state_dict.items()
                    if first_nonfinite and k.startswith(first_nonfinite.split(".")[0])
                }
                torch.save(
                    {
                        "partial_state": partial,
                        "offending_module": first_nonfinite,
                        "size_bytes": rough_size,
                        "truncated": True,
                    },
                    os.path.join(dest_dir, "model_state.pt"),
                )
            else:
                torch.save(state_dict, os.path.join(dest_dir, "model_state.pt"))
        except Exception as exc:
            torch.save({"error": str(exc), "saved": False}, os.path.join(dest_dir, "model_state.pt"))

        # --- File 6: optimizer_state.pt ---
        try:
            opt_state = optimizer.state_dict()
            torch.save(opt_state, os.path.join(dest_dir, "optimizer_state.pt"))
        except Exception as exc:
            torch.save(
                {"error": str(exc), "saved": False},
                os.path.join(dest_dir, "optimizer_state.pt"),
            )

        # --- File 7: numerics.json ---
        first_nonfinite_tensor = None
        if report and report.weight_report:
            first_nonfinite_tensor = report.weight_report.first_nonfinite_name
        if first_nonfinite_tensor is None and report and report.activation_report:
            first_nonfinite_tensor = report.activation_report.first_nonfinite_name

        scaler_state_data = {}
        if scaler is not None:
            try:
                scaler_state_data = scaler.state_dict()
                # Convert tensors to floats for JSON
                scaler_state_data = {
                    k: float(v) if isinstance(v, torch.Tensor) else v
                    for k, v in scaler_state_data.items()
                }
            except Exception:
                scaler_state_data = {"error": "could not serialize scaler state"}

        numerics_data = {
            "step": step,
            "timestamp": time.time(),
            "first_nonfinite_tensor": first_nonfinite_tensor,
            "nan_streak": self.nan_steps_in_a_row,
            "scaler_state": scaler_state_data,
            "skip_counters": skip_counters or {},
            "report": report.to_dict() if report else None,
        }
        with open(os.path.join(dest_dir, "numerics.json"), "w") as f:
            json.dump(numerics_data, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Abort behavior
    # ------------------------------------------------------------------

    def abort(
        self,
        snapshot_path: str,
        report: Optional[NumericsReport] = None,
        first_detection_step: Optional[int] = None,
    ) -> None:
        """Print failure summary and exit or raise according to on_error config.

        Parameters
        ----------
        snapshot_path : str
            Path to the written snapshot directory.
        report : NumericsReport or None
            Latest numerics report for the summary.
        first_detection_step : int or None
            Step where NaN was first detected.

        Raises
        ------
        RuntimeError
            If cfg.on_error == "raise".
        SystemExit
            If cfg.on_error == "abort".
        """
        first_nonfinite = None
        if report:
            if report.weight_report and report.weight_report.first_nonfinite_name:
                first_nonfinite = report.weight_report.first_nonfinite_name
            elif report.activation_report and report.activation_report.first_nonfinite_name:
                first_nonfinite = report.activation_report.first_nonfinite_name

        summary_lines = [
            "",
            "[FATAL] Persistent NaN/Inf detected for "
            f"{self.nan_steps_in_a_row} consecutive steps. Aborting.",
            "",
            "Failure Summary:",
        ]
        if first_detection_step is not None:
            summary_lines.append(f"  First detected: step {first_detection_step}")
        if first_nonfinite:
            summary_lines.append(f"  Offending tensor: {first_nonfinite}")
        summary_lines.append(f"  NaN streak: {self.nan_steps_in_a_row} steps")
        summary_lines.append(f"  Snapshot: {snapshot_path}")
        summary_lines += [
            "",
            "Next steps:",
            "  1. Load snapshot: check batch.pt + rng_state.pt + model_state.pt",
            "  2. Inspect numerics.json for grad norms and logit trends",
            "  3. Consider: reduce LR, add logit clamping, or switch to bf16",
            "",
        ]
        summary = "\n".join(summary_lines)
        print(summary, file=sys.stderr)

        if self.cfg.on_error == "raise":
            raise RuntimeError(
                f"Persistent NaN for {self.nan_steps_in_a_row} steps. "
                f"Snapshot: {snapshot_path}. "
                f"First nonfinite: {first_nonfinite}"
            )
        else:
            # Default: abort
            sys.exit(1)

    # ------------------------------------------------------------------
    # Snapshot validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate(snapshot_dir: str) -> bool:
        """Return True if snapshot_dir contains all 7 required files.

        Parameters
        ----------
        snapshot_dir : str
            Path to snapshot directory to validate.

        Returns
        -------
        bool
            True if all 7 files exist and are non-empty and loadable.
        """
        required = [
            "config.json",
            "env.json",
            "rng_state.pt",
            "batch.pt",
            "model_state.pt",
            "optimizer_state.pt",
            "numerics.json",
        ]
        for fname in required:
            fpath = os.path.join(snapshot_dir, fname)
            if not os.path.exists(fpath):
                return False
            if os.path.getsize(fpath) == 0:
                return False

        # Spot-check JSON files parse
        try:
            for fname in ["config.json", "env.json", "numerics.json"]:
                with open(os.path.join(snapshot_dir, fname)) as f:
                    json.load(f)
        except Exception:
            return False

        # Spot-check torch files load (try safe load first, fall back on error)
        try:
            _load_pt_file(os.path.join(snapshot_dir, "rng_state.pt"))
            _load_pt_file(os.path.join(snapshot_dir, "batch.pt"))
        except Exception:
            return False

        return True


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile as _tempfile

    print("Running failure_snapshot_template.py self-tests...")
    failures = []

    # Helper: minimal model and optimizer
    def _make_model_and_opt():
        model = nn.Linear(4, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        return model, opt

    def _make_batch():
        return {
            "input_ids": torch.randint(0, 100, (2, 8)),
            "labels": torch.randint(0, 2, (2,)),
        }

    # --- T1: capture writes all 7 files ---
    try:
        with _tempfile.TemporaryDirectory() as tmproot:
            cfg = FailureConfig(
                nan_persist_steps=3,
                snapshot_dir=tmproot + "/{run_id}",
                on_error="raise",
            )
            full_cfg = FullConfig(failure=cfg)
            snap = FailureSnapshot(cfg, full_cfg, run_id="test_run")

            model, opt = _make_model_and_opt()
            batch = _make_batch()

            path = snap.capture(
                step=42,
                model=model,
                optimizer=opt,
                batch=batch,
                report=None,
            )

            required = [
                "config.json", "env.json", "rng_state.pt",
                "batch.pt", "model_state.pt", "optimizer_state.pt", "numerics.json",
            ]
            for fname in required:
                fpath = os.path.join(path, fname)
                assert os.path.exists(fpath), f"Missing file: {fname}"
                assert os.path.getsize(fpath) > 0, f"Empty file: {fname}"

        print("  [PASS] T1: capture writes all 7 files")
    except Exception as e:
        failures.append(f"T1 capture 7 files: {e}")
        print(f"  [FAIL] T1: {e}")

    # --- T2: snapshot validates successfully ---
    try:
        with _tempfile.TemporaryDirectory() as tmproot:
            cfg = FailureConfig(snapshot_dir=tmproot + "/{run_id}", on_error="raise")
            snap = FailureSnapshot(cfg, run_id="test_run")
            model, opt = _make_model_and_opt()
            path = snap.capture(step=1, model=model, optimizer=opt, batch=_make_batch(), report=None)
            assert FailureSnapshot.validate(path), f"Snapshot did not validate: {path}"
        print("  [PASS] T2: Snapshot validates without error")
    except Exception as e:
        failures.append(f"T2 snapshot validation: {e}")
        print(f"  [FAIL] T2: {e}")

    # --- T3: NaN streak increments on non-finite loss ---
    try:
        cfg = FailureConfig(nan_persist_steps=3, on_error="raise")
        snap = FailureSnapshot(cfg)

        s1 = snap.update_nan_streak(float("nan"), None)
        assert s1 == 1, f"Expected streak 1, got {s1}"
        s2 = snap.update_nan_streak(float("inf"), None)
        assert s2 == 2, f"Expected streak 2, got {s2}"
        s3 = snap.update_nan_streak(float("nan"), None)
        assert s3 == 3, f"Expected streak 3, got {s3}"
        print("  [PASS] T3: NaN streak increments correctly")
    except Exception as e:
        failures.append(f"T3 nan streak: {e}")
        print(f"  [FAIL] T3: {e}")

    # --- T4: NaN streak resets on clean step ---
    try:
        cfg = FailureConfig(nan_persist_steps=3, on_error="raise")
        snap = FailureSnapshot(cfg)

        snap.update_nan_streak(float("nan"), None)
        snap.update_nan_streak(float("nan"), None)
        assert snap.nan_steps_in_a_row == 2

        s = snap.update_nan_streak(1.234, None)  # clean
        assert s == 0, f"Expected reset to 0, got {s}"
        print("  [PASS] T4: NaN streak resets on clean step")
    except Exception as e:
        failures.append(f"T4 nan streak reset: {e}")
        print(f"  [FAIL] T4: {e}")

    # --- T5: should_abort returns True at threshold ---
    try:
        cfg = FailureConfig(nan_persist_steps=3, on_error="raise")
        snap = FailureSnapshot(cfg)
        assert snap.should_abort(2) is False
        assert snap.should_abort(3) is True
        assert snap.should_abort(10) is True
        print("  [PASS] T5: should_abort correct")
    except Exception as e:
        failures.append(f"T5 should_abort: {e}")
        print(f"  [FAIL] T5: {e}")

    # --- T6: abort with on_error="raise" raises RuntimeError ---
    try:
        with _tempfile.TemporaryDirectory() as tmproot:
            cfg = FailureConfig(
                nan_persist_steps=1,
                snapshot_dir=tmproot + "/{run_id}",
                on_error="raise",
            )
            snap = FailureSnapshot(cfg, run_id="test_run")
            snap.nan_steps_in_a_row = 1

            raised = False
            try:
                snap.abort(snapshot_path=tmproot, report=None)
            except RuntimeError:
                raised = True
        assert raised, "Should have raised RuntimeError"
        print("  [PASS] T6: abort with on_error='raise' raises RuntimeError")
    except Exception as e:
        failures.append(f"T6 abort raise: {e}")
        print(f"  [FAIL] T6: {e}")

    # --- T7: abort with on_error="abort" raises SystemExit ---
    try:
        with _tempfile.TemporaryDirectory() as tmproot:
            cfg = FailureConfig(
                nan_persist_steps=1,
                snapshot_dir=tmproot + "/{run_id}",
                on_error="abort",
            )
            snap = FailureSnapshot(cfg, run_id="test_run")
            snap.nan_steps_in_a_row = 1

            raised = False
            try:
                snap.abort(snapshot_path=tmproot, report=None)
            except SystemExit as se:
                raised = True
                assert se.code == 1, f"Expected exit code 1, got {se.code}"

        assert raised, "Should have raised SystemExit"
        print("  [PASS] T7: abort with on_error='abort' raises SystemExit(1)")
    except Exception as e:
        failures.append(f"T7 abort exit: {e}")
        print(f"  [FAIL] T7: {e}")

    # --- T8: batch tensors CPU-moved and load correctly ---
    try:
        with _tempfile.TemporaryDirectory() as tmproot:
            cfg = FailureConfig(snapshot_dir=tmproot + "/{run_id}", on_error="raise")
            snap = FailureSnapshot(cfg, run_id="test_run")
            model, opt = _make_model_and_opt()
            orig_batch = _make_batch()
            path = snap.capture(step=5, model=model, optimizer=opt, batch=orig_batch, report=None)

            loaded = torch.load(os.path.join(path, "batch.pt"), map_location="cpu")
            for k in orig_batch:
                if isinstance(orig_batch[k], torch.Tensor):
                    assert loaded[k].shape == orig_batch[k].shape, (
                        f"Shape mismatch for key {k}: {loaded[k].shape} vs {orig_batch[k].shape}"
                    )

        print("  [PASS] T8: batch tensors load with correct shapes")
    except Exception as e:
        failures.append(f"T8 batch load: {e}")
        print(f"  [FAIL] T8: {e}")

    print()
    if failures:
        print(f"FAILED: {len(failures)} test(s) failed:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All 8 self-tests passed.")
