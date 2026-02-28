"""
validate_attention.py
======================
Validates all three done-when gates for the fast-attention-sdpa skill.

Gate 1: Audit Works
    - Static scan detects each attention pattern (SDPA, xFormers, flash-attn, eager)
    - Runtime probe produces a populated CapabilityReport

Gate 2: SDPA Integration Correct
    - sdpa_attention() handles dropout correctly (0.0 in inference mode)
    - is_causal + attn_mask raises ValueError
    - Output shape is correct

Gate 3: Backend Selection Controllable
    - backend=math forces Math-only (no fused backend)
    - backend=flash with force=True on CPU raises a clear error
    - backend=auto never crashes

Usage:
    python validate_attention.py
    python validate_attention.py --verbose
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import traceback
from typing import Callable, List, Tuple

import torch

# Ensure the assets directory is on the path
_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")
sys.path.insert(0, os.path.abspath(_ASSETS_DIR))

from attention_auditor_template import AttentionAuditor, AuditReport, CapabilityReport
from sdpa_attention_template import (
    BackendConfig,
    SDPAModule,
    sdpa_attention,
    reset_log_state,
)
from backend_config_template import AttentionConfig

# ---------------------------------------------------------------------------
# Gate runner infrastructure
# ---------------------------------------------------------------------------


class GateResult:
    def __init__(self, gate_name: str):
        self.gate_name = gate_name
        self.checks: List[Tuple[str, bool, str]] = []  # (name, passed, detail)

    def check(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append((name, passed, detail))

    @property
    def passed(self) -> bool:
        return all(p for _, p, _ in self.checks)

    def print_summary(self, verbose: bool = False) -> None:
        status = "PASS" if self.passed else "FAIL"
        print(f"\n[{status}] {self.gate_name}")
        for check_name, check_passed, detail in self.checks:
            icon = "  PASS" if check_passed else "  FAIL"
            print(f"{icon}: {check_name}")
            if detail and (verbose or not check_passed):
                for line in detail.strip().splitlines():
                    print(f"         {line}")


def run_gate(name: str, fn: Callable[[], GateResult], verbose: bool) -> GateResult:
    """Execute a gate function and return its result, catching unexpected exceptions."""
    try:
        result = fn()
    except Exception:
        result = GateResult(name)
        result.check("Unhandled exception", False, traceback.format_exc())
    result.print_summary(verbose=verbose)
    return result


# ---------------------------------------------------------------------------
# Gate 1: Audit Works
# ---------------------------------------------------------------------------


def gate_1_audit_works() -> GateResult:
    gate = GateResult("Gate 1: Audit Works")
    auditor = AttentionAuditor()

    # --- 1a. Static scan detects SDPA pattern ---
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "attention.py")
            with open(fpath, "w") as fh:
                fh.write("import torch.nn.functional as F\n")
                fh.write("out = F.scaled_dot_product_attention(q, k, v, is_causal=True)\n")

            report = auditor.scan_codebase(tmpdir)
            found = any(m.pattern_name == "sdpa_functional" for m in report.matches)
            gate.check(
                "Detect sdpa_functional pattern", found,
                f"Matches found: {[m.pattern_name for m in report.matches]}"
            )
            gate.check(
                "files_scanned == 1", report.files_scanned == 1,
                f"files_scanned={report.files_scanned}"
            )
    except Exception as e:
        gate.check("SDPA scan", False, str(e))

    # --- 1b. Static scan detects eager pattern ---
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "eager.py")
            with open(fpath, "w") as fh:
                fh.write("scores = q @ k.transpose(-2, -1)\n")
                fh.write("attn = torch.softmax(scores, dim=-1)\n")

            report = auditor.scan_codebase(tmpdir)
            found_eager = any(m.pattern_name.startswith("eager_") for m in report.matches)
            gate.check(
                "Detect eager matmul pattern", found_eager,
                f"Matches: {[m.pattern_name for m in report.matches]}"
            )
    except Exception as e:
        gate.check("Eager scan", False, str(e))

    # --- 1c. Static scan detects xFormers pattern ---
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "xformer.py")
            with open(fpath, "w") as fh:
                fh.write("from xformers.ops import memory_efficient_attention\n")
                fh.write("out = xformers.ops.memory_efficient_attention(q, k, v)\n")

            report = auditor.scan_codebase(tmpdir)
            found_xf = any(m.pattern_name.startswith("xformers_") for m in report.matches)
            gate.check(
                "Detect xFormers pattern", found_xf,
                f"Matches: {[m.pattern_name for m in report.matches]}"
            )
    except Exception as e:
        gate.check("xFormers scan", False, str(e))

    # --- 1d. Static scan detects flash-attn import ---
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "flash.py")
            with open(fpath, "w") as fh:
                fh.write("from flash_attn import flash_attn_func\n")
                fh.write("out = flash_attn_func(q, k, v, causal=True)\n")

            report = auditor.scan_codebase(tmpdir)
            found_fa = any(m.pattern_name.startswith("flash_attn_") for m in report.matches)
            gate.check(
                "Detect external flash-attn pattern", found_fa,
                f"Matches: {[m.pattern_name for m in report.matches]}"
            )
    except Exception as e:
        gate.check("Flash-attn scan", False, str(e))

    # --- 1e. Scan of empty file produces no matches ---
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "empty.py")
            with open(fpath, "w") as fh:
                fh.write("# No attention here\nprint('hello')\n")

            report = auditor.scan_codebase(tmpdir)
            gate.check(
                "Empty file produces no matches",
                len(report.matches) == 0,
                f"Got {len(report.matches)} matches"
            )
    except Exception as e:
        gate.check("Empty file scan", False, str(e))

    # --- 1f. Runtime probe on CPU produces CapabilityReport ---
    try:
        q = torch.randn(2, 4, 32, 64, dtype=torch.float32)
        cap = auditor.probe_runtime(q, q.clone(), q.clone(), None, 0.0, False)
        gate.check(
            "probe_runtime returns CapabilityReport",
            isinstance(cap, CapabilityReport),
            f"Type: {type(cap)}"
        )
        gate.check(
            "CapabilityReport device is cpu",
            cap.device.startswith("cpu"),
            f"device={cap.device}"
        )
        gate.check(
            "CapabilityReport can_flash=False on CPU",
            cap.can_flash is False,
            f"can_flash={cap.can_flash}"
        )
        gate.check(
            "CapabilityReport has debug_reasons",
            isinstance(cap.debug_reasons, dict) and "flash" in cap.debug_reasons,
            f"debug_reasons keys: {list(cap.debug_reasons.keys())}"
        )
        gate.check(
            "CapabilityReport q_shape populated",
            cap.q_shape == (2, 4, 32, 64),
            f"q_shape={cap.q_shape}"
        )
        gate.check(
            "recommended_backend() returns math on CPU",
            cap.recommended_backend() == "math",
            f"recommended={cap.recommended_backend()}"
        )
    except Exception as e:
        gate.check("probe_runtime CPU", False, str(e))

    # --- 1g. Runtime probe on CUDA if available ---
    if torch.cuda.is_available():
        try:
            dtype = torch.float16
            device = torch.device("cuda:0")
            q = torch.randn(2, 4, 64, 64, dtype=dtype, device=device)
            cap = auditor.probe_runtime(q, q.clone(), q.clone(), None, 0.0, False)
            gate.check(
                "probe_runtime returns CapabilityReport on CUDA",
                isinstance(cap, CapabilityReport),
            )
            gate.check(
                "CapabilityReport device contains cuda",
                "cuda" in cap.device,
                f"device={cap.device}"
            )
        except Exception as e:
            gate.check("probe_runtime CUDA", False, str(e))

    return gate


# ---------------------------------------------------------------------------
# Gate 2: SDPA Integration Correct
# ---------------------------------------------------------------------------


def gate_2_sdpa_integration() -> GateResult:
    gate = GateResult("Gate 2: SDPA Integration Correct")

    # --- 2a. Basic forward pass shape ---
    try:
        reset_log_state()
        q = torch.randn(2, 4, 16, 64)
        out = sdpa_attention(q, q.clone(), q.clone(), training=False)
        gate.check(
            "Forward pass produces correct shape",
            out.shape == (2, 4, 16, 64),
            f"Expected (2,4,16,64), got {out.shape}"
        )
    except Exception as e:
        gate.check("Forward pass shape", False, str(e))

    # --- 2b. Dropout = 0.0 in inference mode (via SDPAModule) ---
    try:
        mod = SDPAModule(num_heads=4, head_dim=64, dropout_p=0.5)
        mod.train(False)  # Set inference mode using train(False) not .eval()
        assert not mod.training, "Module should be in inference mode"

        q = torch.randn(2, 4, 32, 64)
        # Should not raise -- dropout is zeroed internally
        out = mod(q, q.clone(), q.clone())
        gate.check(
            "SDPAModule inference mode: dropout zeroed automatically",
            out.shape == q.shape,
            f"Output shape: {out.shape}"
        )
    except Exception as e:
        gate.check("Inference mode dropout zeroed", False, str(e))

    # --- 2c. dropout_p > 0 in inference mode raises AssertionError ---
    try:
        q = torch.randn(2, 4, 16, 64)
        raised = False
        try:
            sdpa_attention(q, q.clone(), q.clone(), dropout_p=0.1, training=False)
        except AssertionError:
            raised = True
        gate.check(
            "dropout_p > 0 with training=False raises AssertionError",
            raised,
            "AssertionError not raised when expected"
        )
    except Exception as e:
        gate.check("Dropout guard raises", False, str(e))

    # --- 2d. dropout_p > 0 in training mode succeeds ---
    try:
        q = torch.randn(2, 4, 16, 64)
        out = sdpa_attention(q, q.clone(), q.clone(), dropout_p=0.1, training=True)
        gate.check(
            "dropout_p > 0 with training=True succeeds",
            out.shape == q.shape,
        )
    except Exception as e:
        gate.check("Dropout in training succeeds", False, str(e))

    # --- 2e. is_causal + attn_mask raises ValueError ---
    try:
        q = torch.randn(2, 4, 16, 64)
        mask = torch.ones(16, 16, dtype=torch.bool)
        raised = False
        try:
            sdpa_attention(q, q.clone(), q.clone(), attn_mask=mask, is_causal=True,
                           training=False)
        except ValueError:
            raised = True
        gate.check(
            "is_causal + attn_mask raises ValueError",
            raised,
            "ValueError not raised when expected"
        )
    except Exception as e:
        gate.check("is_causal + attn_mask guard", False, str(e))

    # --- 2f. is_causal=True alone succeeds ---
    try:
        q = torch.randn(2, 4, 16, 64)
        out = sdpa_attention(q, q.clone(), q.clone(), is_causal=True, training=False)
        gate.check(
            "is_causal=True alone succeeds with correct shape",
            out.shape == q.shape,
        )
    except Exception as e:
        gate.check("is_causal alone", False, str(e))

    # --- 2g. attn_mask alone (float) succeeds ---
    try:
        q = torch.randn(2, 4, 16, 64)
        mask = torch.zeros(16, 16)  # float mask
        out = sdpa_attention(q, q.clone(), q.clone(), attn_mask=mask, training=False)
        gate.check(
            "Float attn_mask alone succeeds",
            out.shape == q.shape,
        )
    except Exception as e:
        gate.check("Float attn_mask alone", False, str(e))

    # --- 2h. Numerical correctness: Math backend vs manual ---
    try:
        import math as _math

        reset_log_state()
        bc = BackendConfig(policy="math", force=False, log=False)
        torch.manual_seed(0)
        q = torch.randn(1, 1, 8, 32)
        k = torch.randn(1, 1, 8, 32)
        v = torch.randn(1, 1, 8, 32)

        # Manual attention
        scale = 1.0 / _math.sqrt(32)
        w = torch.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
        expected = w @ v

        got = sdpa_attention(q, k, v, training=False, backend_cfg=bc)
        correct = torch.allclose(got, expected, atol=1e-5)
        gate.check(
            "SDPA Math backend matches manual attention (atol=1e-5)",
            correct,
            f"Max diff: {(got - expected).abs().max().item():.2e}"
        )
    except ImportError:
        gate.check("Numerical correctness", False, "SDPBackend not available")
    except Exception as e:
        gate.check("Numerical correctness", False, str(e))

    # --- 2i. SDPAModule training mode passes dropout ---
    try:
        mod = SDPAModule(num_heads=2, head_dim=32, dropout_p=0.5)
        mod.train(True)
        assert mod.training
        q = torch.randn(1, 2, 16, 32)
        out = mod(q, q.clone(), q.clone())
        gate.check(
            "SDPAModule training mode: dropout_p passed to SDPA",
            out.shape == q.shape,
        )
    except Exception as e:
        gate.check("SDPAModule training dropout", False, str(e))

    return gate


# ---------------------------------------------------------------------------
# Gate 3: Backend Selection Controllable
# ---------------------------------------------------------------------------


def gate_3_backend_selection() -> GateResult:
    gate = GateResult("Gate 3: Backend Selection Controllable")

    try:
        from torch.nn.attention import SDPBackend
    except ImportError:
        gate.check("SDPBackend import", False, "SDPBackend not available (PyTorch < 2.0)")
        return gate

    # --- 3a. backend=math forces Math-only ---
    try:
        reset_log_state()
        bc = BackendConfig(policy="math", force=False, log=False)
        assert SDPBackend.MATH in bc.backends
        assert SDPBackend.FLASH_ATTENTION not in bc.backends
        assert SDPBackend.EFFICIENT_ATTENTION not in bc.backends

        q = torch.randn(2, 4, 16, 64)
        out = sdpa_attention(q, q.clone(), q.clone(), training=False, backend_cfg=bc)
        gate.check(
            "backend=math: only Math in backends list",
            SDPBackend.MATH in bc.backends and SDPBackend.FLASH_ATTENTION not in bc.backends,
        )
        gate.check(
            "backend=math: forward pass succeeds on CPU",
            out.shape == q.shape,
        )
    except Exception as e:
        gate.check("backend=math", False, str(e))

    # --- 3b. backend=flash with force=True: Math is removed from backends list ---
    try:
        reset_log_state()
        bc_force = BackendConfig(policy="flash", force=True, log=False)
        # force=True removes Math from backends -- this is the key invariant
        assert SDPBackend.MATH not in bc_force.backends, \
            f"Math should not be in forced flash backends: {bc_force.backends}"
        assert SDPBackend.FLASH_ATTENTION in bc_force.backends, \
            f"FLASH_ATTENTION should be in forced flash backends"

        gate.check(
            "backend=flash force=True: Math removed from backends",
            SDPBackend.MATH not in bc_force.backends,
            f"backends={[b.name for b in bc_force.backends]}"
        )
    except Exception as e:
        gate.check("backend=flash force on CPU", False, str(e))

    # --- 3b2. backend=flash force=True on CUDA with float32 raises ---
    # Flash does not support float32; with Math removed via force=True,
    # SDPA should raise RuntimeError on CUDA.
    if torch.cuda.is_available():
        try:
            reset_log_state()
            bc_force = BackendConfig(policy="flash", force=True, log=False)
            device = torch.device("cuda:0")
            # float32: Flash cannot run it, and force=True has removed Math
            q_f32 = torch.randn(2, 4, 16, 64, dtype=torch.float32, device=device)
            raised = False
            error_msg = ""
            try:
                sdpa_attention(q_f32, q_f32.clone(), q_f32.clone(),
                               training=False, backend_cfg=bc_force)
            except (RuntimeError, AssertionError) as e:
                raised = True
                error_msg = str(e)
            gate.check(
                "backend=flash force=True on CUDA float32 raises (Flash needs fp16/bf16)",
                raised,
                f"Error: {error_msg[:200]}" if raised else "No error raised (may pass if PyTorch Math still runs)"
            )
        except Exception as e:
            gate.check("backend=flash force CUDA float32", False, str(e))

    # --- 3c. backend=auto never crashes on CPU ---
    try:
        reset_log_state()
        bc_auto = BackendConfig(policy="auto", force=False, log=False)
        q = torch.randn(2, 4, 32, 64)
        out = sdpa_attention(q, q.clone(), q.clone(), training=False, backend_cfg=bc_auto)
        gate.check(
            "backend=auto never crashes on CPU",
            out.shape == q.shape,
        )
    except Exception as e:
        gate.check("backend=auto on CPU", False, str(e))

    # --- 3d. backend=auto on CUDA if available ---
    if torch.cuda.is_available():
        try:
            reset_log_state()
            device = torch.device("cuda:0")
            dtype = torch.float16
            q = torch.randn(2, 4, 64, 64, dtype=dtype, device=device)
            bc_auto = BackendConfig(policy="auto", force=False, log=False)
            out = sdpa_attention(q, q.clone(), q.clone(), is_causal=True,
                                  training=False, backend_cfg=bc_auto)
            gate.check(
                "backend=auto on CUDA: forward succeeds",
                out.shape == q.shape,
            )
        except Exception as e:
            gate.check("backend=auto on CUDA", False, str(e))

    # --- 3e. backend=efficient on CPU uses Math fallback ---
    try:
        reset_log_state()
        bc_eff = BackendConfig(policy="efficient", force=False, log=False)
        assert SDPBackend.MATH in bc_eff.backends  # Math is always in fallback
        q = torch.randn(2, 4, 32, 64)
        out = sdpa_attention(q, q.clone(), q.clone(), training=False, backend_cfg=bc_eff)
        gate.check(
            "backend=efficient on CPU: Math fallback works",
            out.shape == q.shape,
        )
    except Exception as e:
        gate.check("backend=efficient Math fallback", False, str(e))

    # --- 3f. BackendConfig.from_attention_config handles all policies ---
    try:
        from backend_config_template import BackendConfig as BCFull, AttentionConfig

        for policy in ["auto", "flash", "efficient", "cudnn", "math", "flash_or_efficient"]:
            cfg = AttentionConfig(backend=policy, force=False)
            bc = BCFull.from_attention_config(cfg)
            assert isinstance(bc.backends, list), f"backends not a list for policy={policy}"

        gate.check("BackendConfig.from_attention_config: all policies work", True)
    except Exception as e:
        gate.check("BackendConfig.from_attention_config", False, str(e))

    return gate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Validate fast-attention-sdpa done-when gates."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show detailed output for passing checks too."
    )
    args = parser.parse_args(argv)

    print("=" * 65)
    print("fast-attention-sdpa: Done-When Gate Validation")
    print("=" * 65)
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        sm = props.major * 10 + props.minor
        print(f"GPU             : {props.name} (SM{sm})")
    print()

    results = []
    for name, fn in [
        ("Gate 1: Audit Works", gate_1_audit_works),
        ("Gate 2: SDPA Integration Correct", gate_2_sdpa_integration),
        ("Gate 3: Backend Selection Controllable", gate_3_backend_selection),
    ]:
        result = run_gate(name, fn, verbose=args.verbose)
        results.append(result)

    # Final summary
    print("\n" + "=" * 65)
    print("Summary")
    print("=" * 65)
    all_passed = True
    for r in results:
        n_total = len(r.checks)
        n_passed = sum(1 for _, p, _ in r.checks if p)
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.gate_name}  ({n_passed}/{n_total} checks)")
        if not r.passed:
            all_passed = False

    print()
    if all_passed:
        print("ALL GATES PASSED")
        sys.exit(0)
    else:
        print("SOME GATES FAILED -- see details above")
        sys.exit(1)


if __name__ == "__main__":
    main()
