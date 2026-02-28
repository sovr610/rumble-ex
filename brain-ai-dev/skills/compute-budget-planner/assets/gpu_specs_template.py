"""
gpu_specs_template.py
=====================
GPUSpecTable: GPU hardware specification registry with fuzzy lookup.

Maintains a database of peak TFLOPS, memory, and related specs for GPUs
commonly used in LLM training. Always defaults to dense (non-sparse) figures
unless explicitly requested.

Provides:
    - GPU_SPECS dict: internal registry of known GPUs
    - GPUSpecTable class: lookup, fuzzy matching, fallback, listing
    - GPUSpec dataclass: single GPU spec (imported from budget_config_template)

Usage
-----
    from gpu_specs_template import GPUSpecTable

    table = GPUSpecTable()
    spec = table.lookup("H100_SXM", dtype="bf16")
    print(spec.peak_tflops)   # 989.0
    print(spec.mem_gb)        # 80.0

    # Fallback to user-supplied values for unknown GPUs:
    spec = table.validate_or_fallback("UnknownGPU9000",
                                       user_peak_tflops=500.0,
                                       user_mem_gb=40.0)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPUSpec dataclass (standalone to avoid circular imports)
# ---------------------------------------------------------------------------

@dataclass
class GPUSpec:
    """Hardware specification for a single GPU.

    Attributes
    ----------
    name : str
        Canonical GPU name (e.g., "H100 SXM").
    peak_tflops : float
        Peak TFLOPS for the selected dtype (dense, not sparse).
    mem_gb : float
        GPU memory in gigabytes.
    is_sparse : bool
        Whether peak_tflops reflects sparse (2:4) operation. Default False.
    compute_capability : float
        NVIDIA compute capability (e.g., 9.0 for H100).
    peak_tflops_bf16 : float
        BF16 dense TFLOPS.
    peak_tflops_fp16 : float
        FP16 dense TFLOPS.
    peak_tflops_fp32 : float
        FP32 dense TFLOPS.
    peak_tflops_tf32 : float
        TF32 dense TFLOPS.
    is_user_supplied : bool
        True if specs were provided by user rather than from internal table.
    """

    name: str
    peak_tflops: float
    mem_gb: float
    is_sparse: bool = False
    compute_capability: float = 0.0
    peak_tflops_bf16: float = 0.0
    peak_tflops_fp16: float = 0.0
    peak_tflops_fp32: float = 0.0
    peak_tflops_tf32: float = 0.0
    is_user_supplied: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "peak_tflops": self.peak_tflops,
            "mem_gb": self.mem_gb,
            "is_sparse": self.is_sparse,
            "compute_capability": self.compute_capability,
            "peak_tflops_bf16": self.peak_tflops_bf16,
            "peak_tflops_fp16": self.peak_tflops_fp16,
            "peak_tflops_fp32": self.peak_tflops_fp32,
            "peak_tflops_tf32": self.peak_tflops_tf32,
            "is_user_supplied": self.is_user_supplied,
        }

    @classmethod
    def user_supplied(cls, peak_tflops: float, mem_gb: float) -> "GPUSpec":
        """Create a GPUSpec from user-provided values."""
        return cls(
            name="user_supplied",
            peak_tflops=peak_tflops,
            mem_gb=mem_gb,
            peak_tflops_bf16=peak_tflops,
            peak_tflops_fp16=peak_tflops,
            peak_tflops_tf32=peak_tflops / 2.0,
            peak_tflops_fp32=peak_tflops / 20.0,
            is_user_supplied=True,
        )


# ---------------------------------------------------------------------------
# GPU Specifications Registry
# ---------------------------------------------------------------------------
# Format for each entry:
#   {
#       "canonical_name": str,
#       "aliases": [str, ...],          # lowercase patterns for fuzzy match
#       "peak_tflops_bf16": float,      # DENSE bf16 TFLOPS
#       "peak_tflops_fp16": float,      # DENSE fp16 TFLOPS
#       "peak_tflops_fp32": float,      # DENSE fp32 TFLOPS (not TF32)
#       "peak_tflops_tf32": float,      # DENSE TF32 TFLOPS (Ampere+)
#       "mem_gb": float,
#       "compute_capability": float,
#   }
#
# All TFLOPS figures are DENSE (non-sparse) unless noted.
# Sources: NVIDIA official datasheets (see references/gpu-spec-table.md).

_GPU_REGISTRY: List[Dict[str, Any]] = [
    # -----------------------------------------------------------------------
    # H100 SXM
    # -----------------------------------------------------------------------
    {
        "canonical_name": "H100 SXM",
        "aliases": ["h100_sxm", "h100 sxm", "h100sxm", "h100-sxm",
                    "h100_80gb_sxm", "h100 80gb sxm"],
        "peak_tflops_bf16": 989.0,
        "peak_tflops_fp16": 989.0,
        "peak_tflops_tf32": 494.0,
        "peak_tflops_fp32": 33.5,
        "mem_gb": 80.0,
        "compute_capability": 9.0,
    },
    # -----------------------------------------------------------------------
    # H100 PCIe
    # -----------------------------------------------------------------------
    {
        "canonical_name": "H100 PCIe",
        "aliases": ["h100_pcie", "h100 pcie", "h100pcie", "h100-pcie"],
        "peak_tflops_bf16": 756.0,
        "peak_tflops_fp16": 756.0,
        "peak_tflops_tf32": 378.0,
        "peak_tflops_fp32": 26.0,
        "mem_gb": 80.0,
        "compute_capability": 9.0,
    },
    # -----------------------------------------------------------------------
    # H100 generic (fallback if user just says "h100"; resolves to SXM)
    # -----------------------------------------------------------------------
    {
        "canonical_name": "H100 SXM",  # canonical same as SXM
        "aliases": ["h100"],            # catch-all for "h100" alone
        "peak_tflops_bf16": 989.0,
        "peak_tflops_fp16": 989.0,
        "peak_tflops_tf32": 494.0,
        "peak_tflops_fp32": 33.5,
        "mem_gb": 80.0,
        "compute_capability": 9.0,
    },
    # -----------------------------------------------------------------------
    # H200 SXM
    # -----------------------------------------------------------------------
    {
        "canonical_name": "H200 SXM",
        "aliases": ["h200_sxm", "h200 sxm", "h200sxm", "h200-sxm",
                    "h200_141gb", "h200 141gb", "h200"],
        "peak_tflops_bf16": 989.0,     # same compute as H100; bigger VRAM
        "peak_tflops_fp16": 989.0,
        "peak_tflops_tf32": 494.0,
        "peak_tflops_fp32": 33.5,
        "mem_gb": 141.0,               # HBM3e 141GB
        "compute_capability": 9.0,
    },
    # -----------------------------------------------------------------------
    # A100 80GB SXM
    # -----------------------------------------------------------------------
    {
        "canonical_name": "A100 80GB SXM",
        "aliases": ["a100_80gb_sxm", "a100 80gb sxm", "a100_sxm",
                    "a100 sxm", "a100sxm", "a100-sxm", "a100_80",
                    "a100 80gb", "a100_80gb"],
        "peak_tflops_bf16": 312.0,
        "peak_tflops_fp16": 312.0,
        "peak_tflops_tf32": 156.0,
        "peak_tflops_fp32": 9.7,
        "mem_gb": 80.0,
        "compute_capability": 8.0,
    },
    # -----------------------------------------------------------------------
    # A100 80GB PCIe
    # -----------------------------------------------------------------------
    {
        "canonical_name": "A100 80GB PCIe",
        "aliases": ["a100_80gb_pcie", "a100 80gb pcie", "a100_pcie",
                    "a100 pcie", "a100pcie", "a100-pcie"],
        "peak_tflops_bf16": 312.0,
        "peak_tflops_fp16": 312.0,
        "peak_tflops_tf32": 156.0,
        "peak_tflops_fp32": 9.7,
        "mem_gb": 80.0,
        "compute_capability": 8.0,
    },
    # -----------------------------------------------------------------------
    # A100 40GB SXM
    # -----------------------------------------------------------------------
    {
        "canonical_name": "A100 40GB SXM",
        "aliases": ["a100_40gb_sxm", "a100 40gb sxm", "a100_40gb",
                    "a100 40gb", "a100_40"],
        "peak_tflops_bf16": 312.0,
        "peak_tflops_fp16": 312.0,
        "peak_tflops_tf32": 156.0,
        "peak_tflops_fp32": 9.7,
        "mem_gb": 40.0,
        "compute_capability": 8.0,
    },
    # -----------------------------------------------------------------------
    # A100 generic (fallback if user says "a100" alone; resolves to 80GB SXM)
    # -----------------------------------------------------------------------
    {
        "canonical_name": "A100 80GB SXM",
        "aliases": ["a100"],
        "peak_tflops_bf16": 312.0,
        "peak_tflops_fp16": 312.0,
        "peak_tflops_tf32": 156.0,
        "peak_tflops_fp32": 9.7,
        "mem_gb": 80.0,
        "compute_capability": 8.0,
    },
    # -----------------------------------------------------------------------
    # L40S
    # -----------------------------------------------------------------------
    {
        "canonical_name": "L40S",
        "aliases": ["l40s", "l40-s", "l40_s", "rtx6000_ada"],
        "peak_tflops_bf16": 362.0,
        "peak_tflops_fp16": 362.0,
        "peak_tflops_tf32": 183.0,
        "peak_tflops_fp32": 91.6,
        "mem_gb": 48.0,
        "compute_capability": 8.9,
    },
    # -----------------------------------------------------------------------
    # RTX 4090
    # -----------------------------------------------------------------------
    {
        "canonical_name": "RTX 4090",
        "aliases": ["rtx4090", "rtx 4090", "rtx-4090", "4090",
                    "geforce_rtx_4090", "geforce rtx 4090"],
        "peak_tflops_bf16": 330.0,
        "peak_tflops_fp16": 330.0,
        "peak_tflops_tf32": 165.0,
        "peak_tflops_fp32": 82.6,
        "mem_gb": 24.0,
        "compute_capability": 8.9,
    },
    # -----------------------------------------------------------------------
    # V100 32GB SXM2
    # -----------------------------------------------------------------------
    {
        "canonical_name": "V100 32GB",
        "aliases": ["v100_32gb", "v100 32gb", "v100_32", "v100 32",
                    "v100_sxm2", "v100 sxm2", "v100"],
        "peak_tflops_bf16": 0.0,       # V100 has no native BF16
        "peak_tflops_fp16": 125.0,
        "peak_tflops_tf32": 0.0,       # No TF32 on V100
        "peak_tflops_fp32": 15.7,
        "mem_gb": 32.0,
        "compute_capability": 7.0,
    },
]

# Deduplicated public list (for user-facing list_gpus())
_CANONICAL_GPU_NAMES = [
    "H100 SXM",
    "H100 PCIe",
    "H200 SXM",
    "A100 80GB SXM",
    "A100 80GB PCIe",
    "A100 40GB SXM",
    "L40S",
    "RTX 4090",
    "V100 32GB",
]


# ---------------------------------------------------------------------------
# GPUSpecTable
# ---------------------------------------------------------------------------

class GPUSpecTable:
    """Registry and lookup table for GPU hardware specifications.

    Methods
    -------
    lookup(gpu_type, dtype="bf16") -> GPUSpec
        Find a GPU spec by name with fuzzy matching.
    list_gpus() -> List[str]
        Return canonical names of all known GPUs.
    validate_or_fallback(gpu_type, user_peak_tflops, user_mem_gb) -> GPUSpec
        Return spec from table or fall back to user-supplied values.
    """

    def __init__(self) -> None:
        self._registry = _GPU_REGISTRY
        logger.debug("GPUSpecTable initialized with %d entries.", len(self._registry))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(self, gpu_type: str, dtype: str = "bf16") -> Optional[GPUSpec]:
        """Find a GPU specification by type name with fuzzy matching.

        Matching is case-insensitive. Checks canonical name and all aliases
        for substring or exact match. Returns the first match.

        Parameters
        ----------
        gpu_type : str
            GPU model name. Examples: "H100_SXM", "h100", "A100 80GB SXM".
        dtype : str
            Requested dtype for peak_tflops resolution.
            One of: "bf16", "fp16", "fp32", "tf32".

        Returns
        -------
        GPUSpec or None
            GPUSpec with peak_tflops set for the requested dtype.
            Returns None if no match found.
        """
        if not gpu_type:
            return None

        query = gpu_type.lower().strip().replace("-", "_").replace(" ", "_")
        dtype_lower = dtype.lower()

        for entry in self._registry:
            canonical = entry["canonical_name"]
            aliases = entry.get("aliases", [])

            # Normalize aliases for comparison
            normalized_aliases = [
                a.lower().replace("-", "_").replace(" ", "_") for a in aliases
            ]
            canonical_norm = canonical.lower().replace("-", "_").replace(" ", "_")

            # Check exact match first (canonical or alias)
            if query == canonical_norm or query in normalized_aliases:
                return self._make_spec(entry, dtype_lower)

            # Fuzzy substring match: query is substring of alias, or alias is substring of query
            for alias_norm in normalized_aliases:
                if query in alias_norm or alias_norm in query:
                    logger.debug(
                        "Fuzzy match: '%s' matched alias '%s' -> '%s'",
                        gpu_type, alias_norm, canonical
                    )
                    return self._make_spec(entry, dtype_lower)

        logger.warning("GPU '%s' not found in GPUSpecTable.", gpu_type)
        return None

    def list_gpus(self) -> List[str]:
        """Return a deduplicated list of canonical GPU names.

        Returns
        -------
        list of str
        """
        return list(_CANONICAL_GPU_NAMES)

    def validate_or_fallback(
        self,
        gpu_type: Optional[str],
        user_peak_tflops: Optional[float] = None,
        user_mem_gb: Optional[float] = None,
        dtype: str = "bf16",
    ) -> GPUSpec:
        """Return spec from table, or fall back to user-supplied values.

        If the GPU type is found in the table, returns the table spec.
        If not found:
            - If user_peak_tflops and user_mem_gb are both provided, returns
              a user-supplied GPUSpec with a warning logged.
            - Otherwise raises ValueError.

        Parameters
        ----------
        gpu_type : str or None
        user_peak_tflops : float or None
        user_mem_gb : float or None
        dtype : str

        Returns
        -------
        GPUSpec

        Raises
        ------
        ValueError
            If GPU not found and user values are not supplied.
        """
        if gpu_type:
            spec = self.lookup(gpu_type, dtype=dtype)
            if spec is not None:
                return spec

        # Not found — try user-supplied fallback
        if user_peak_tflops is not None and user_mem_gb is not None:
            if not math.isfinite(user_peak_tflops) or user_peak_tflops <= 0:
                raise ValueError(
                    f"user_peak_tflops must be a positive finite number, got {user_peak_tflops!r}"
                )
            if not math.isfinite(user_mem_gb) or user_mem_gb <= 0:
                raise ValueError(
                    f"user_mem_gb must be a positive finite number, got {user_mem_gb!r}"
                )
            logger.warning(
                "GPU '%s' not found in spec table. Using user-supplied values: "
                "peak_tflops=%s, mem_gb=%s. "
                "Ensure these are DENSE (non-sparse) figures for the correct dtype.",
                gpu_type, user_peak_tflops, user_mem_gb
            )
            return GPUSpec.user_supplied(user_peak_tflops, user_mem_gb)

        # No fallback available
        known = ", ".join(self.list_gpus())
        raise ValueError(
            f"GPU '{gpu_type}' not found in spec table and no user-supplied values provided. "
            f"Known GPUs: {known}. "
            f"Provide --peak_tflops and --mem_gb to use custom hardware specs."
        )

    def get_peak_tflops(self, gpu_type: str, dtype: str = "bf16") -> float:
        """Convenience method: return peak TFLOPS for a GPU+dtype combination.

        Parameters
        ----------
        gpu_type : str
        dtype : str

        Returns
        -------
        float

        Raises
        ------
        ValueError if GPU not found.
        """
        spec = self.lookup(gpu_type, dtype=dtype)
        if spec is None:
            raise ValueError(f"GPU '{gpu_type}' not found. Use validate_or_fallback() for fallback support.")
        return spec.peak_tflops

    def print_table(self) -> None:
        """Print a formatted table of all known GPU specs to stdout."""
        header = f"{'GPU':<25} {'BF16 TFLOPS':>12} {'FP16 TFLOPS':>12} {'MEM (GB)':>9} {'CC':>4}"
        print(header)
        print("-" * len(header))
        seen = set()
        for entry in self._registry:
            name = entry["canonical_name"]
            if name in seen:
                continue
            seen.add(name)
            bf16 = entry.get("peak_tflops_bf16", 0)
            fp16 = entry.get("peak_tflops_fp16", 0)
            mem = entry.get("mem_gb", 0)
            cc = entry.get("compute_capability", 0)
            print(f"{name:<25} {bf16:>12.1f} {fp16:>12.1f} {mem:>9.1f} {cc:>4.1f}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_spec(self, entry: Dict[str, Any], dtype: str) -> GPUSpec:
        """Build a GPUSpec from a registry entry with the correct peak_tflops for dtype."""
        dtype_key_map = {
            "bf16": "peak_tflops_bf16",
            "bfloat16": "peak_tflops_bf16",
            "fp16": "peak_tflops_fp16",
            "float16": "peak_tflops_fp16",
            "fp32": "peak_tflops_fp32",
            "float32": "peak_tflops_fp32",
            "tf32": "peak_tflops_tf32",
        }

        tflops_key = dtype_key_map.get(dtype, "peak_tflops_bf16")
        peak = entry.get(tflops_key, 0.0)

        # Special case: V100 has no BF16 — fall back to FP16 with warning
        if peak == 0.0 and dtype in ("bf16", "bfloat16"):
            fp16_peak = entry.get("peak_tflops_fp16", 0.0)
            if fp16_peak > 0.0:
                logger.warning(
                    "GPU '%s' has no native BF16 support. "
                    "Using FP16 peak_tflops=%.1f as approximation. "
                    "Consider using fp16 dtype for this GPU.",
                    entry["canonical_name"], fp16_peak
                )
                peak = fp16_peak

        if peak == 0.0:
            logger.warning(
                "GPU '%s' has peak_tflops=0 for dtype='%s'. "
                "Check spec table or provide values manually.",
                entry["canonical_name"], dtype
            )

        spec = GPUSpec(
            name=entry["canonical_name"],
            peak_tflops=peak,
            mem_gb=entry.get("mem_gb", 0.0),
            is_sparse=False,
            compute_capability=entry.get("compute_capability", 0.0),
            peak_tflops_bf16=entry.get("peak_tflops_bf16", 0.0),
            peak_tflops_fp16=entry.get("peak_tflops_fp16", 0.0),
            peak_tflops_fp32=entry.get("peak_tflops_fp32", 0.0),
            peak_tflops_tf32=entry.get("peak_tflops_tf32", 0.0),
            is_user_supplied=False,
        )

        logger.debug(
            "Resolved GPU spec: %s dtype=%s -> peak_tflops=%.1f, mem_gb=%.1f, is_sparse=%s",
            spec.name, dtype, spec.peak_tflops, spec.mem_gb, spec.is_sparse
        )
        return spec


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _run_self_tests() -> None:
    print("Running gpu_specs_template.py self-tests...")
    errors = []

    table = GPUSpecTable()

    # Test 1: A100 80GB SXM BF16
    try:
        spec = table.lookup("A100_80GB_SXM", dtype="bf16")
        assert spec is not None, "A100_80GB_SXM not found"
        assert spec.peak_tflops == 312.0, f"Expected 312.0, got {spec.peak_tflops}"
        assert spec.mem_gb == 80.0, f"Expected 80.0 GB, got {spec.mem_gb}"
        assert spec.is_sparse is False
        print("  [PASS] A100 80GB SXM BF16 lookup: peak_tflops=312, mem_gb=80")
    except Exception as e:
        errors.append(f"  [FAIL] A100 80GB SXM lookup: {e}")

    # Test 2: H100 SXM BF16
    try:
        spec = table.lookup("H100_SXM", dtype="bf16")
        assert spec is not None, "H100_SXM not found"
        assert spec.peak_tflops == 989.0, f"Expected 989.0, got {spec.peak_tflops}"
        assert spec.mem_gb == 80.0
        assert spec.is_sparse is False
        print("  [PASS] H100 SXM BF16 lookup: peak_tflops=989, mem_gb=80")
    except Exception as e:
        errors.append(f"  [FAIL] H100 SXM lookup: {e}")

    # Test 3: H200 BF16
    try:
        spec = table.lookup("H200", dtype="bf16")
        assert spec is not None, "H200 not found"
        assert spec.peak_tflops == 989.0, f"Expected 989.0, got {spec.peak_tflops}"
        assert spec.mem_gb == 141.0, f"Expected 141.0 GB, got {spec.mem_gb}"
        print("  [PASS] H200 BF16 lookup: peak_tflops=989, mem_gb=141")
    except Exception as e:
        errors.append(f"  [FAIL] H200 lookup: {e}")

    # Test 4: V100 FP16
    try:
        spec = table.lookup("V100", dtype="fp16")
        assert spec is not None, "V100 not found"
        assert spec.peak_tflops == 125.0, f"Expected 125.0, got {spec.peak_tflops}"
        assert spec.mem_gb == 32.0
        print("  [PASS] V100 FP16 lookup: peak_tflops=125, mem_gb=32")
    except Exception as e:
        errors.append(f"  [FAIL] V100 lookup: {e}")

    # Test 5: L40S BF16
    try:
        spec = table.lookup("L40S", dtype="bf16")
        assert spec is not None, "L40S not found"
        assert spec.peak_tflops == 362.0, f"Expected 362.0, got {spec.peak_tflops}"
        assert spec.mem_gb == 48.0
        print("  [PASS] L40S BF16 lookup: peak_tflops=362, mem_gb=48")
    except Exception as e:
        errors.append(f"  [FAIL] L40S lookup: {e}")

    # Test 6: RTX 4090
    try:
        spec = table.lookup("RTX 4090", dtype="bf16")
        assert spec is not None, "RTX 4090 not found"
        assert spec.peak_tflops == 330.0
        assert spec.mem_gb == 24.0
        print("  [PASS] RTX 4090 lookup: peak_tflops=330, mem_gb=24")
    except Exception as e:
        errors.append(f"  [FAIL] RTX 4090 lookup: {e}")

    # Test 7: Unknown GPU returns None
    try:
        spec = table.lookup("NonExistentGPU9000_XYZ")
        assert spec is None, f"Expected None for unknown GPU, got {spec}"
        print("  [PASS] Unknown GPU returns None")
    except Exception as e:
        errors.append(f"  [FAIL] Unknown GPU handling: {e}")

    # Test 8: validate_or_fallback with user-supplied values
    try:
        spec = table.validate_or_fallback(
            "NonExistentGPU9000",
            user_peak_tflops=500.0,
            user_mem_gb=40.0,
        )
        assert spec.peak_tflops == 500.0
        assert spec.mem_gb == 40.0
        assert spec.is_user_supplied is True
        print("  [PASS] validate_or_fallback uses user-supplied values for unknown GPU")
    except Exception as e:
        errors.append(f"  [FAIL] validate_or_fallback user-supplied: {e}")

    # Test 9: validate_or_fallback raises on unknown GPU without user values
    try:
        try:
            table.validate_or_fallback("NonExistentGPU9000")
            errors.append("  [FAIL] Should raise ValueError for unknown GPU without user values")
        except ValueError as e:
            assert "not found" in str(e).lower() or "NonExistentGPU9000" in str(e)
        print("  [PASS] validate_or_fallback raises ValueError for unknown GPU without user values")
    except Exception as e:
        errors.append(f"  [FAIL] validate_or_fallback error handling: {e}")

    # Test 10: Fuzzy matching — lowercase
    try:
        spec_h100 = table.lookup("h100")
        assert spec_h100 is not None, "h100 (lowercase) not found"
        assert spec_h100.peak_tflops == 989.0
        print("  [PASS] Fuzzy match: 'h100' -> H100 SXM")
    except Exception as e:
        errors.append(f"  [FAIL] Fuzzy match lowercase h100: {e}")

    # Test 11: Fuzzy matching — mixed case and separators
    try:
        for query in ["H100 SXM", "H100_SXM", "h100_sxm", "h100sxm"]:
            spec = table.lookup(query)
            assert spec is not None, f"'{query}' not found"
            assert spec.peak_tflops == 989.0, f"Wrong peak for '{query}': {spec.peak_tflops}"
        print("  [PASS] Fuzzy match: H100 SXM in multiple formats")
    except Exception as e:
        errors.append(f"  [FAIL] Fuzzy match H100 formats: {e}")

    # Test 12: Fuzzy matching — 4090 shorthand
    try:
        spec = table.lookup("4090")
        assert spec is not None, "'4090' not found"
        assert spec.peak_tflops == 330.0
        print("  [PASS] Fuzzy match: '4090' -> RTX 4090")
    except Exception as e:
        errors.append(f"  [FAIL] Fuzzy match '4090': {e}")

    # Test 13: dtype selection (bf16 vs fp16 for H100 should be same)
    try:
        spec_bf16 = table.lookup("H100_SXM", dtype="bf16")
        spec_fp16 = table.lookup("H100_SXM", dtype="fp16")
        spec_fp32 = table.lookup("H100_SXM", dtype="fp32")
        assert spec_bf16.peak_tflops == spec_fp16.peak_tflops  # H100: equal
        assert spec_bf16.peak_tflops > spec_fp32.peak_tflops
        print("  [PASS] dtype selection: bf16==fp16 for H100, bf16 >> fp32")
    except Exception as e:
        errors.append(f"  [FAIL] dtype selection: {e}")

    # Test 14: V100 BF16 falls back to FP16 (with warning)
    try:
        spec = table.lookup("V100", dtype="bf16")
        assert spec is not None
        # V100 has no bf16, should fall back to fp16=125
        assert spec.peak_tflops == 125.0, f"V100 bf16 fallback should be 125, got {spec.peak_tflops}"
        print("  [PASS] V100 BF16 gracefully falls back to FP16 peak=125")
    except Exception as e:
        errors.append(f"  [FAIL] V100 BF16 fallback: {e}")

    # Test 15: list_gpus returns canonical names
    try:
        gpus = table.list_gpus()
        assert len(gpus) > 0
        assert "H100 SXM" in gpus
        assert "A100 80GB SXM" in gpus
        print(f"  [PASS] list_gpus returns {len(gpus)} canonical GPU names including H100 SXM and A100 80GB SXM")
    except Exception as e:
        errors.append(f"  [FAIL] list_gpus: {e}")

    # Test 16: H100 PCIe vs SXM distinguished
    try:
        sxm = table.lookup("H100_SXM", dtype="bf16")
        pcie = table.lookup("H100_PCIe", dtype="bf16")
        assert sxm is not None and pcie is not None
        assert sxm.peak_tflops > pcie.peak_tflops, (
            f"H100 SXM ({sxm.peak_tflops}) should have more TFLOPS than PCIe ({pcie.peak_tflops})"
        )
        assert sxm.peak_tflops == 989.0
        assert pcie.peak_tflops == 756.0
        print(f"  [PASS] H100 SXM (989 TFLOPS) > H100 PCIe (756 TFLOPS) correctly distinguished")
    except Exception as e:
        errors.append(f"  [FAIL] H100 SXM vs PCIe: {e}")

    # Test 17: A100 40GB has correct memory
    try:
        spec = table.lookup("a100_40gb")
        assert spec is not None
        assert spec.mem_gb == 40.0, f"A100 40GB: expected mem_gb=40, got {spec.mem_gb}"
        assert spec.peak_tflops == 312.0
        print("  [PASS] A100 40GB SXM: mem_gb=40, peak_tflops=312")
    except Exception as e:
        errors.append(f"  [FAIL] A100 40GB: {e}")

    # Summary
    if errors:
        print("\nFailed tests:")
        for err in errors:
            print(err)
        raise SystemExit(1)
    else:
        print("\nAll self-tests passed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    _run_self_tests()
