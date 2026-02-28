"""
flash_attn_checker_template.py
================================
Runtime capability detection for the external flash-attn package.

Classes:
    CheckResult       -- availability and constraint check result
    FlashAttnChecker  -- main checker class

Self-tests: python flash_attn_checker_template.py

Note: Tests mock flash_attn imports via sys.modules so the external package
      is not required to run the tests.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock

import torch

logger = logging.getLogger(__name__)

# Minimum CUDA SM (compute capability * 10) for flash-attn
_FLASH_ATTN_MIN_SM = 80  # Ampere (A100, RTX 30xx)

# Supported dtypes for flash-attn external package
_FLASH_ATTN_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16}

# Supported head_dim range for FA2
_FLASH_ATTN_MIN_HEAD_DIM = 8
_FLASH_ATTN_MAX_HEAD_DIM = 256

# Install hint shown in "require" error messages
_INSTALL_HINT = (
    "Install: pip install flash-attn --no-build-isolation\n"
    "Requirements: CUDA >= 11.6, Linux, Ampere+ GPU (SM80+), "
    "PyTorch >= 1.12, Python >= 3.7"
)


# ---------------------------------------------------------------------------
# CheckResult
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    """Result of an availability or constraint check."""
    available: bool
    reason: str
    version: Optional[str] = None
    constraints: Optional[dict] = None

    def __str__(self) -> str:
        status = "OK" if self.available else "UNAVAILABLE"
        return f"CheckResult({status}): {self.reason}"


# ---------------------------------------------------------------------------
# FlashAttnChecker
# ---------------------------------------------------------------------------


class FlashAttnChecker:
    """
    Runtime capability checks for the external flash-attn package.

    This class checks:
      1. Whether flash-attn is installed and importable.
      2. Whether the current device supports flash-attn (SM80+).
      3. Whether the requested dtype and head_dim are supported.
      4. Mode resolution: off / prefer / require.

    Example:
        checker = FlashAttnChecker()
        if checker.is_available():
            result = checker.check_constraints(head_dim=128, dtype=torch.float16,
                                                device=torch.device("cuda"))
            if result.available:
                # use flash_attn_func
    """

    def is_available(self) -> bool:
        """
        Return True if the flash-attn package is installed and importable.

        Does not check device capability or constraints -- only import success.
        """
        try:
            import flash_attn  # noqa: F401
            return True
        except (ImportError, ModuleNotFoundError):
            return False

    def get_version(self) -> Optional[str]:
        """
        Return the installed flash-attn version string, or None if not installed.
        """
        try:
            import flash_attn
            return getattr(flash_attn, "__version__", "unknown")
        except (ImportError, ModuleNotFoundError):
            return None

    def check_device(self, device: torch.device) -> CheckResult:
        """
        Check if the device supports flash-attn (must be CUDA SM80+).

        Args:
            device : torch.device to check.

        Returns:
            CheckResult with available=True if requirements met.
        """
        if device.type != "cuda":
            return CheckResult(
                available=False,
                reason=f"flash-attn requires a CUDA device; got device type '{device.type}'",
            )

        try:
            props = torch.cuda.get_device_properties(device)
            sm = props.major * 10 + props.minor
        except Exception as exc:
            return CheckResult(
                available=False,
                reason=f"Could not query CUDA device properties: {exc}",
            )

        if sm < _FLASH_ATTN_MIN_SM:
            return CheckResult(
                available=False,
                reason=(
                    f"flash-attn requires SM{_FLASH_ATTN_MIN_SM}+ (Ampere/Ada/Hopper). "
                    f"Device '{props.name}' is SM{sm}. "
                    "Use SDPBackend.EFFICIENT_ATTENTION instead."
                ),
            )

        return CheckResult(
            available=True,
            reason=f"Device '{props.name}' (SM{sm}) is compatible",
            constraints={"sm": sm, "device_name": props.name},
        )

    def check_dtype(self, dtype: torch.dtype) -> CheckResult:
        """
        Check if the dtype is supported by flash-attn.

        Supports: float16, bfloat16
        Does NOT support: float32, int8, etc.
        """
        if dtype not in _FLASH_ATTN_SUPPORTED_DTYPES:
            supported_names = [str(d) for d in _FLASH_ATTN_SUPPORTED_DTYPES]
            return CheckResult(
                available=False,
                reason=(
                    f"flash-attn does not support dtype={dtype}. "
                    f"Supported dtypes: {supported_names}. "
                    "Cast to float16 or bfloat16, or use SDPBackend.EFFICIENT_ATTENTION."
                ),
            )
        return CheckResult(available=True, reason=f"dtype={dtype} is supported")

    def check_head_dim(self, head_dim: int) -> CheckResult:
        """
        Check if the head_dim is within flash-attn's supported range.

        FA2 supports head_dim in [8, 256].
        """
        if not (_FLASH_ATTN_MIN_HEAD_DIM <= head_dim <= _FLASH_ATTN_MAX_HEAD_DIM):
            return CheckResult(
                available=False,
                reason=(
                    f"flash-attn head_dim={head_dim} is outside supported range "
                    f"[{_FLASH_ATTN_MIN_HEAD_DIM}, {_FLASH_ATTN_MAX_HEAD_DIM}]."
                ),
            )
        return CheckResult(
            available=True,
            reason=f"head_dim={head_dim} is within supported range",
        )

    def check_constraints(
        self,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> CheckResult:
        """
        Check all flash-attn constraints: device capability, dtype, head_dim.

        Returns CheckResult with the first failure encountered.
        If all constraints pass, returns CheckResult(available=True).

        Args:
            head_dim : Attention head dimension.
            dtype    : Tensor dtype (must be float16 or bfloat16).
            device   : CUDA device (must be SM80+).
        """
        # Check in order: device > dtype > head_dim
        device_result = self.check_device(device)
        if not device_result.available:
            return device_result

        dtype_result = self.check_dtype(dtype)
        if not dtype_result.available:
            return dtype_result

        head_dim_result = self.check_head_dim(head_dim)
        if not head_dim_result.available:
            return head_dim_result

        return CheckResult(
            available=True,
            reason=(
                f"All constraints met: device={device}, "
                f"dtype={dtype}, head_dim={head_dim}"
            ),
            version=self.get_version(),
            constraints={
                "head_dim": head_dim,
                "dtype": str(dtype),
                "device": str(device),
            },
        )

    def resolve_mode(
        self,
        mode: str,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> str:
        """
        Resolve which attention implementation to use given the mode.

        Modes:
            "off"     -- Always return "sdpa". Ignore flash-attn entirely.
            "prefer"  -- Return "flash_attn" if available and constraints met,
                         otherwise log a warning and return "sdpa".
            "require" -- Return "flash_attn" if available and constraints met,
                         otherwise raise RuntimeError with install instructions.

        Returns:
            "flash_attn" or "sdpa"

        Raises:
            ValueError    : Unknown mode string.
            RuntimeError  : mode="require" and flash-attn not available/compatible.
        """
        if mode == "off":
            return "sdpa"

        if mode not in ("prefer", "require"):
            raise ValueError(
                f"Unknown flash-attn mode {mode!r}. "
                "Valid modes: 'off', 'prefer', 'require'"
            )

        # Check package availability first
        if not self.is_available():
            msg = (
                f"flash-attn package is not installed "
                f"(mode={mode!r}). {_INSTALL_HINT}"
            )
            if mode == "require":
                raise RuntimeError(msg)
            else:
                logger.warning("%s -- falling back to SDPA", msg)
                return "sdpa"

        # Check constraints
        result = self.check_constraints(head_dim=head_dim, dtype=dtype, device=device)
        if not result.available:
            msg = (
                f"flash-attn constraints not met (mode={mode!r}): {result.reason}"
            )
            if mode == "require":
                raise RuntimeError(msg + f"\n{_INSTALL_HINT}")
            else:
                logger.warning("%s -- falling back to SDPA", msg)
                return "sdpa"

        logger.info("flash-attn resolved (mode=%r): using flash_attn", mode)
        return "flash_attn"


# ---------------------------------------------------------------------------
# Self-tests (using sys.modules mocking -- no real flash_attn needed)
# ---------------------------------------------------------------------------


def _run_self_tests() -> None:
    print("Running flash_attn_checker_template self-tests...")
    failures = []

    checker = FlashAttnChecker()

    # --- is_available: not installed ---
    try:
        # Ensure flash_attn is not in sys.modules
        _saved = sys.modules.pop("flash_attn", None)
        try:
            result = checker.is_available()
            # If it truly isn't installed, this should be False
            # If it IS installed, this is True -- both are acceptable
            print(f"  PASS: is_available() returned {result} (package {'IS' if result else 'NOT'} installed)")
        finally:
            if _saved is not None:
                sys.modules["flash_attn"] = _saved
    except Exception as e:
        failures.append(f"is_available no-install: {e}")

    # --- is_available: mock installed ---
    try:
        mock_fa = MagicMock()
        mock_fa.__version__ = "2.4.2"
        sys.modules["flash_attn"] = mock_fa
        assert checker.is_available() is True
        assert checker.get_version() == "2.4.2"
        print("  PASS: is_available() True when mocked")
        print("  PASS: get_version() returns mock version")
    except Exception as e:
        failures.append(f"is_available mock: {e}")
    finally:
        sys.modules.pop("flash_attn", None)

    # --- is_available: mock not installed ---
    try:
        sys.modules.pop("flash_attn", None)
        # Make the import fail by setting to a failing sentinel
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "flash_attn":
                raise ImportError("No module named 'flash_attn'")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = mock_import
        try:
            assert checker.is_available() is False
            assert checker.get_version() is None
            print("  PASS: is_available() False when package raises ImportError")
            print("  PASS: get_version() None when package raises ImportError")
        finally:
            builtins.__import__ = original_import
    except Exception as e:
        failures.append(f"is_available not-installed: {e}")

    # --- check_dtype: supported ---
    try:
        r = checker.check_dtype(torch.float16)
        assert r.available is True
        print("  PASS: check_dtype float16 returns available=True")
    except Exception as e:
        failures.append(f"check_dtype float16: {e}")

    try:
        r = checker.check_dtype(torch.bfloat16)
        assert r.available is True
        print("  PASS: check_dtype bfloat16 returns available=True")
    except Exception as e:
        failures.append(f"check_dtype bfloat16: {e}")

    # --- check_dtype: not supported ---
    try:
        r = checker.check_dtype(torch.float32)
        assert r.available is False
        assert "float32" in r.reason or "float" in r.reason
        print("  PASS: check_dtype float32 returns available=False with reason")
    except Exception as e:
        failures.append(f"check_dtype float32: {e}")

    # --- check_head_dim: in range ---
    try:
        for hd in [32, 64, 128, 256]:
            r = checker.check_head_dim(hd)
            assert r.available is True, f"head_dim={hd} should be available"
        print("  PASS: check_head_dim in range [32,64,128,256]")
    except Exception as e:
        failures.append(f"check_head_dim in range: {e}")

    # --- check_head_dim: out of range ---
    try:
        r = checker.check_head_dim(512)
        assert r.available is False
        assert "512" in r.reason or "head_dim" in r.reason
        print("  PASS: check_head_dim 512 returns available=False")
    except Exception as e:
        failures.append(f"check_head_dim 512: {e}")

    # --- check_device: CPU ---
    try:
        r = checker.check_device(torch.device("cpu"))
        assert r.available is False
        assert "CUDA" in r.reason or "cuda" in r.reason
        print("  PASS: check_device CPU returns available=False")
    except Exception as e:
        failures.append(f"check_device CPU: {e}")

    # --- check_constraints: CPU (short-circuits on device check) ---
    try:
        r = checker.check_constraints(
            head_dim=128, dtype=torch.float16, device=torch.device("cpu")
        )
        assert r.available is False
        print("  PASS: check_constraints on CPU returns available=False")
    except Exception as e:
        failures.append(f"check_constraints CPU: {e}")

    # --- check_constraints: float32 on CUDA (if available) ---
    try:
        if torch.cuda.is_available():
            r = checker.check_constraints(
                head_dim=128, dtype=torch.float32, device=torch.device("cuda")
            )
            assert r.available is False
            assert "float32" in r.reason or "float" in r.reason.lower()
            print("  PASS: check_constraints float32 on CUDA returns available=False")
        else:
            print("  SKIP: check_constraints float32 on CUDA (no CUDA)")
    except Exception as e:
        failures.append(f"check_constraints float32 CUDA: {e}")

    # --- resolve_mode: off ---
    try:
        # mode=off always returns "sdpa" regardless of everything else
        result = checker.resolve_mode("off", 128, torch.float16, torch.device("cpu"))
        assert result == "sdpa"
        print("  PASS: resolve_mode('off') returns 'sdpa'")
    except Exception as e:
        failures.append(f"resolve_mode off: {e}")

    # --- resolve_mode: prefer + not installed ---
    try:
        import builtins
        original_import = builtins.__import__

        def mock_import_fail(name, *args, **kwargs):
            if name == "flash_attn":
                raise ImportError("No module named 'flash_attn'")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = mock_import_fail
        sys.modules.pop("flash_attn", None)
        try:
            result = checker.resolve_mode("prefer", 128, torch.float16, torch.device("cpu"))
            assert result == "sdpa", f"Expected 'sdpa', got {result!r}"
            print("  PASS: resolve_mode('prefer') falls back to 'sdpa' when not installed")
        finally:
            builtins.__import__ = original_import
    except Exception as e:
        failures.append(f"resolve_mode prefer not-installed: {e}")

    # --- resolve_mode: require + not installed ---
    try:
        import builtins
        original_import = builtins.__import__

        def mock_import_fail2(name, *args, **kwargs):
            if name == "flash_attn":
                raise ImportError("No module named 'flash_attn'")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = mock_import_fail2
        sys.modules.pop("flash_attn", None)
        try:
            checker.resolve_mode("require", 128, torch.float16, torch.device("cpu"))
            failures.append("resolve_mode require should raise RuntimeError when not installed")
        except RuntimeError as err:
            assert "pip install" in str(err) or "Install" in str(err) or "flash-attn" in str(err)
            print("  PASS: resolve_mode('require') raises RuntimeError with install hint")
        finally:
            builtins.__import__ = original_import
    except Exception as e:
        failures.append(f"resolve_mode require not-installed: {e}")

    # --- resolve_mode: invalid mode ---
    try:
        checker.resolve_mode("invalid", 128, torch.float16, torch.device("cpu"))
        failures.append("Invalid mode should raise ValueError")
    except ValueError:
        print("  PASS: resolve_mode raises ValueError for unknown mode")

    # --- prefer with mock installed but constraint fail (cpu device) ---
    try:
        mock_fa = MagicMock()
        mock_fa.__version__ = "2.4.2"
        sys.modules["flash_attn"] = mock_fa
        try:
            result = checker.resolve_mode("prefer", 128, torch.float32, torch.device("cpu"))
            assert result == "sdpa"
            print("  PASS: resolve_mode prefer falls back when constraints fail")
        finally:
            sys.modules.pop("flash_attn", None)
    except Exception as e:
        failures.append(f"resolve_mode prefer constraints fail: {e}")

    # --- require with mock installed but constraint fail ---
    try:
        mock_fa = MagicMock()
        mock_fa.__version__ = "2.4.2"
        sys.modules["flash_attn"] = mock_fa
        try:
            checker.resolve_mode("require", 128, torch.float32, torch.device("cpu"))
            failures.append("require with failing constraints should raise RuntimeError")
        except RuntimeError:
            print("  PASS: resolve_mode require raises when constraints fail (mock installed)")
        finally:
            sys.modules.pop("flash_attn", None)
    except Exception as e:
        failures.append(f"resolve_mode require constraint fail: {e}")

    # --- prefer with mock installed + all constraints met (CUDA if available) ---
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            sm = props.major * 10 + props.minor
            if sm >= 80:
                mock_fa = MagicMock()
                mock_fa.__version__ = "2.4.2"
                sys.modules["flash_attn"] = mock_fa
                try:
                    result = checker.resolve_mode(
                        "prefer", 128, torch.float16, torch.device("cuda:0")
                    )
                    assert result == "flash_attn"
                    print("  PASS: resolve_mode prefer returns 'flash_attn' on Ampere+")
                finally:
                    sys.modules.pop("flash_attn", None)
            else:
                print("  SKIP: SM < 80, cannot test Ampere-specific path")
        else:
            print("  SKIP: No CUDA for resolve_mode prefer with all constraints met")
    except Exception as e:
        failures.append(f"resolve_mode prefer success: {e}")

    # --- CheckResult str representation ---
    try:
        r = CheckResult(available=True, reason="All good")
        s = str(r)
        assert "OK" in s
        r2 = CheckResult(available=False, reason="No CUDA")
        s2 = str(r2)
        assert "UNAVAILABLE" in s2
        print("  PASS: CheckResult __str__ contains status tag")
    except Exception as e:
        failures.append(f"CheckResult str: {e}")

    # --- Summary ---
    if failures:
        print(f"\nFAILED {len(failures)} tests:")
        for f in failures:
            print(f"  FAIL: {f}")
        raise SystemExit(1)
    else:
        print(f"\nAll flash_attn_checker_template self-tests PASSED")


if __name__ == "__main__":
    _run_self_tests()
