"""
collect_env_template.py
========================
CollectEnv class for capturing a deterministic environment snapshot
and generating a machine profile string for baseline keying.

Usage:
    from collect_env_template import CollectEnv

    env = CollectEnv()
    info = env.collect()
    profile = env.machine_profile()   # e.g. "H100x8_driver550_cuda12.4_torch2.4_sm90"
    env.save("/path/to/env.json")

Self-test:
    python collect_env_template.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Optional torch import — graceful degradation on CPU-only machines
# ---------------------------------------------------------------------------
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ===========================================================================
# CollectEnv
# ===========================================================================

class CollectEnv:
    """
    Collects a deterministic environment snapshot for reproducible performance
    baseline keying. All methods are idempotent — calling multiple times returns
    identical results unless the environment changes between calls.

    Example:
        env = CollectEnv()
        info = env.collect()
        print(env.machine_profile())
        env.save("artifacts/env.json")
    """

    # Tracked environment variable names
    ENV_VAR_NAMES: List[str] = [
        "CUDA_VISIBLE_DEVICES",
        "NCCL_DEBUG",
        "NCCL_IB_DISABLE",
        "NCCL_P2P_DISABLE",
        "NCCL_SOCKET_IFNAME",
        "NCCL_NET_PLUGIN",
        "NCCL_ALGO",
        "NCCL_PROTO",
        "TORCHINDUCTOR_CACHE_DIR",
        "TORCHINDUCTOR_MAX_AUTOTUNE",
        "TORCH_COMPILE_DEBUG",
        "PYTORCH_CUDA_ALLOC_CONF",
        "TORCH_DISTRIBUTED_DEBUG",
        "MASTER_ADDR",
        "MASTER_PORT",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "OMP_NUM_THREADS",
        "TOKENIZERS_PARALLELISM",
    ]

    def __init__(self) -> None:
        self._cached_info: Optional[Dict[str, Any]] = None

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def collect(self) -> Dict[str, Any]:
        """
        Return a complete environment dictionary.

        Returns:
            dict with keys: git, python_version, torch, cuda, gpus, env_vars,
            schema_version, timestamp.
        """
        if self._cached_info is not None:
            return dict(self._cached_info)

        info: Dict[str, Any] = {
            "schema_version": "1.0",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python_version": self._get_python_version(),
            "git": self._get_git_info(),
            "torch": self._get_torch_info(),
            "cuda": self._get_cuda_driver_info(),
            "gpus": self._get_gpu_info(),
            "env_vars": self._get_env_vars(),
        }

        self._cached_info = info
        return dict(info)

    def machine_profile(self) -> str:
        """
        Generate a deterministic machine profile string suitable for use as a
        filename and dict key for baseline storage.

        Format: <GPUModel>x<count>_driver<N>_cuda<M>_torch<V>_sm<C>

        Examples:
            H100x8_driver550_cuda12.4_torch2.4_sm90
            A100x4_driver525_cuda12.1_torch2.3_sm80
            cpu_torch2.4                              (CPU-only machine)

        Returns:
            Deterministic profile string.
        """
        info = self.collect()
        gpus = info.get("gpus", [])

        if not gpus:
            # CPU-only machine
            torch_ver = info["torch"].get("version", "unknown").split("+")[0]
            torch_short = _shorten_version(torch_ver, parts=2)
            return f"cpu_torch{torch_short}"

        # Take the first GPU as representative (all should be same model in HPC)
        first_gpu = gpus[0]
        gpu_count = len(gpus)

        gpu_name = _sanitize_name(first_gpu.get("name", "UNKNOWN"))
        compute_cap = first_gpu.get("compute_cap", "00").replace(".", "")

        # Driver version: "550.54.14" -> "550"
        driver_ver = info["cuda"].get("driver_version", "000")
        driver_short = driver_ver.split(".")[0]

        # CUDA version from torch: "12.4" -> "12.4"
        cuda_ver = info["torch"].get("cuda_version", "0.0")
        cuda_short = _shorten_version(cuda_ver, parts=2)

        # Torch version: "2.4.0" -> "2.4"
        torch_ver = info["torch"].get("version", "0.0.0").split("+")[0]
        torch_short = _shorten_version(torch_ver, parts=2)

        profile = (
            f"{gpu_name}x{gpu_count}"
            f"_driver{driver_short}"
            f"_cuda{cuda_short}"
            f"_torch{torch_short}"
            f"_sm{compute_cap}"
        )
        return profile

    def save(self, path: str) -> None:
        """
        Write env info to a JSON file atomically (write-to-temp + rename).

        Args:
            path: Destination file path. Parent directories are created.
        """
        info = self.collect()
        info["machine_profile"] = self.machine_profile()

        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: write to temp file then rename
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=dest.parent, suffix=".tmp", prefix=dest.stem
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(info, f, indent=2, default=str)
            os.replace(tmp_path, dest)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # -----------------------------------------------------------------------
    # Internal collectors
    # -----------------------------------------------------------------------

    def _get_python_version(self) -> str:
        """Return Python version string like '3.11.4'."""
        v = sys.version_info
        return f"{v.major}.{v.minor}.{v.micro}"

    def _get_git_info(self) -> Dict[str, Any]:
        """Return git sha and dirty status."""
        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).decode().strip()
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            sha = "unknown"

        try:
            dirty_output = subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).decode().strip()
            dirty = len(dirty_output) > 0
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            dirty = False

        return {"sha": sha, "dirty": dirty}

    def _get_torch_info(self) -> Dict[str, Any]:
        """Return torch version, CUDA version, and build flags."""
        if not _TORCH_AVAILABLE:
            return {
                "version": "not_installed",
                "cuda_version": "N/A",
                "build_flags": [],
                "git_version": "N/A",
            }

        build_flags = []
        # torch.version.cuda can be None on CPU builds
        cuda_version = getattr(torch.version, "cuda", None) or "N/A"

        # Collect build configuration flags
        if hasattr(torch, "backends"):
            if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
                build_flags.append(f"cudnn={torch.backends.cudnn.version()}")
            if hasattr(torch.backends, "mkl") and torch.backends.mkl.is_available():
                build_flags.append("mkl")
            if hasattr(torch.backends, "openmp") and torch.backends.openmp.is_available():
                build_flags.append("openmp")

        return {
            "version": torch.__version__,
            "cuda_version": cuda_version,
            "build_flags": build_flags,
            "git_version": getattr(torch.version, "git_version", "N/A"),
        }

    def _get_cuda_driver_info(self) -> Dict[str, Any]:
        """Return CUDA driver version from nvidia-smi."""
        driver_version = "N/A"
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL,
                timeout=10,
            ).decode().strip()
            # Take first line if multiple GPUs
            driver_version = output.splitlines()[0].strip()
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            # Fallback: try parsing from torch
            if _TORCH_AVAILABLE and torch.cuda.is_available():
                try:
                    driver_version = str(torch.version.cuda)
                except Exception:
                    pass

        return {"driver_version": driver_version}

    def _get_gpu_info(self) -> List[Dict[str, Any]]:
        """Return list of GPU info dicts."""
        if not _TORCH_AVAILABLE or not torch.cuda.is_available():
            return []

        gpus = []
        for i in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(i)
                gpus.append({
                    "index": i,
                    "name": props.name,
                    "mem_total": props.total_memory,
                    "mem_total_gb": round(props.total_memory / (1024 ** 3), 1),
                    "compute_cap": f"{props.major}.{props.minor}",
                    "multi_processor_count": props.multi_processor_count,
                })
            except Exception as exc:
                gpus.append({"index": i, "error": str(exc)})

        return gpus

    def _get_env_vars(self) -> Dict[str, str]:
        """Return dict of tracked environment variables (present vars only)."""
        return {
            name: os.environ[name]
            for name in self.ENV_VAR_NAMES
            if name in os.environ
        }


# ===========================================================================
# Helpers
# ===========================================================================

def _shorten_version(version_str: str, parts: int = 2) -> str:
    """
    Shorten a version string to N dot-separated parts.
    "2.4.0" with parts=2 -> "2.4"
    "12.4.1" with parts=2 -> "12.4"
    """
    components = version_str.split(".")
    return ".".join(components[:parts])


def _sanitize_name(name: str) -> str:
    """
    Convert a GPU display name to a compact identifier safe for filenames.
    "NVIDIA H100 SXM5 80GB HBM3" -> "H100SXM5"
    "NVIDIA A100-SXM4-80GB" -> "A100SXM4"
    """
    # Remove "NVIDIA" prefix and common noise words
    noise = ["NVIDIA", "GeForce", "Quadro", "Tesla", "RTX", "GTX", "GB", "HBM", "SXM"]
    result = name
    for n in noise:
        result = result.replace(n, n if n in ("RTX", "SXM") else "")

    # Remove non-alphanumeric chars except dash
    result = re.sub(r"[^A-Za-z0-9]", "", result)
    return result[:20]  # Cap at 20 chars to avoid excessively long filenames


# ===========================================================================
# Self-Tests
# ===========================================================================

def _run_self_tests() -> None:
    """Run all self-tests. Raises AssertionError or SystemExit on failure."""
    print("Running collect_env_template self-tests...")
    failures: List[str] = []

    def check(name: str, condition: bool, msg: str = "") -> None:
        if not condition:
            failures.append(f"FAIL [{name}]: {msg}")
        else:
            print(f"  PASS  {name}")

    env = CollectEnv()

    # --- collect() returns all required keys ---
    info = env.collect()
    required_keys = ["schema_version", "timestamp", "python_version", "git", "torch", "cuda", "gpus", "env_vars"]
    for key in required_keys:
        check(f"collect.has_{key}", key in info, f"Missing key '{key}'")

    # --- Python version format ---
    pver = info["python_version"]
    check(
        "collect.python_version_format",
        re.match(r"^\d+\.\d+\.\d+$", pver) is not None,
        f"python_version '{pver}' doesn't match X.Y.Z format",
    )

    # --- Git section has sha and dirty ---
    check("collect.git.has_sha", "sha" in info["git"])
    check("collect.git.has_dirty", "dirty" in info["git"])
    check("collect.git.sha_is_string", isinstance(info["git"]["sha"], str))
    check("collect.git.dirty_is_bool", isinstance(info["git"]["dirty"], bool))

    # --- Torch section ---
    torch_info = info["torch"]
    check("collect.torch.has_version", "version" in torch_info)
    check("collect.torch.has_cuda_version", "cuda_version" in torch_info)
    check("collect.torch.build_flags_is_list", isinstance(torch_info.get("build_flags", []), list))

    # --- CUDA section ---
    cuda_info = info["cuda"]
    check("collect.cuda.has_driver_version", "driver_version" in cuda_info)

    # --- GPUs section ---
    gpus = info["gpus"]
    check("collect.gpus_is_list", isinstance(gpus, list))
    if gpus:
        first = gpus[0]
        check("collect.gpu.has_name", "name" in first)
        check("collect.gpu.has_mem_total", "mem_total" in first)
        check("collect.gpu.has_compute_cap", "compute_cap" in first)
        check("collect.gpu.mem_total_positive", first["mem_total"] > 0)

    # --- env_vars is a dict ---
    check("collect.env_vars_is_dict", isinstance(info["env_vars"], dict))

    # --- machine_profile() is deterministic ---
    p1 = env.machine_profile()
    p2 = env.machine_profile()
    check("machine_profile.deterministic", p1 == p2, f"Got '{p1}' then '{p2}'")

    # --- machine_profile() is a non-empty string ---
    check("machine_profile.nonempty", len(p1) > 0)
    check("machine_profile.no_spaces", " " not in p1, f"Profile has spaces: '{p1}'")

    # --- machine_profile() format (CPU or GPU) ---
    if gpus:
        # Should have _driver, _cuda, _torch, _sm components
        check("machine_profile.has_driver", "_driver" in p1, f"Missing _driver: '{p1}'")
        check("machine_profile.has_cuda", "_cuda" in p1, f"Missing _cuda: '{p1}'")
        check("machine_profile.has_torch", "_torch" in p1, f"Missing _torch: '{p1}'")
        check("machine_profile.has_sm", "_sm" in p1, f"Missing _sm: '{p1}'")
    else:
        check("machine_profile.cpu_prefix", "cpu" in p1 or "_torch" in p1, f"CPU profile: '{p1}'")

    # --- save() creates valid JSON ---
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, "test_env.json")
        env.save(save_path)
        check("save.file_exists", os.path.isfile(save_path))

        with open(save_path) as f:
            loaded = json.load(f)

        # Roundtrip: loaded should have all required keys
        for key in required_keys:
            check(f"save.roundtrip.has_{key}", key in loaded, f"Missing key '{key}' in loaded JSON")

        # machine_profile should be in saved file
        check("save.has_machine_profile", "machine_profile" in loaded)
        check("save.machine_profile_matches", loaded["machine_profile"] == p1)

    # --- _shorten_version helper ---
    check("shorten_version.2_4_0", _shorten_version("2.4.0") == "2.4")
    check("shorten_version.12_4_1", _shorten_version("12.4.1") == "12.4")
    check("shorten_version.1_part", _shorten_version("2.4.0", parts=1) == "2")

    # --- _sanitize_name helper ---
    sanitized = _sanitize_name("NVIDIA H100 SXM5 80GB HBM3")
    check(
        "sanitize_name.no_spaces",
        " " not in sanitized,
        f"Sanitized name has spaces: '{sanitized}'",
    )
    check(
        "sanitize_name.reasonable_length",
        0 < len(sanitized) <= 20,
        f"Sanitized length {len(sanitized)} out of range",
    )

    # --- JSON serializability of collect() output ---
    try:
        json_str = json.dumps(info, default=str)
        check("collect.json_serializable", len(json_str) > 0)
    except (TypeError, ValueError) as exc:
        check("collect.json_serializable", False, str(exc))

    # Summary
    print()
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    else:
        print("All collect_env_template self-tests passed.")
        print(f"Machine profile: {env.machine_profile()}")


if __name__ == "__main__":
    _run_self_tests()
