"""
attention_auditor_template.py
==============================
AttentionAuditor: static codebase scan + runtime SDPA capability probe.

Classes:
    AuditMatch        -- single pattern match (file, line, pattern, snippet)
    AuditReport       -- collection of matches from scan_codebase()
    CapabilityReport  -- result of probe_runtime()
    AttentionAuditor  -- main auditor class

CLI usage:
    python attention_auditor_template.py --scan /path/to/codebase
    python attention_auditor_template.py --run
    python attention_auditor_template.py --scan /path/to/codebase --run --save report.json

Self-tests: python attention_auditor_template.py
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attention Pattern Dictionary
# ---------------------------------------------------------------------------

ATTENTION_PATTERNS: Dict[str, re.Pattern] = {
    # -----------------------------------------------------------------------
    # SDPA -- the target state
    # -----------------------------------------------------------------------
    "sdpa_functional": re.compile(
        r"F\.scaled_dot_product_attention\s*\("
    ),
    "sdpa_torch_nn_functional": re.compile(
        r"torch\.nn\.functional\.scaled_dot_product_attention\s*\("
    ),
    "sdpa_aten": re.compile(
        r"torch\._C\._nn\.scaled_dot_product_attention\s*\("
    ),

    # -----------------------------------------------------------------------
    # xFormers
    # -----------------------------------------------------------------------
    "xformers_memory_efficient": re.compile(
        r"xformers\.ops\.memory_efficient_attention\s*\("
    ),
    "xformers_mem_eff_import": re.compile(
        r"from\s+xformers\.ops\s+import\s+memory_efficient_attention"
    ),
    "xformers_ops_import": re.compile(
        r"import\s+xformers\.ops"
    ),

    # -----------------------------------------------------------------------
    # External flash-attn package
    # -----------------------------------------------------------------------
    "flash_attn_func": re.compile(
        r"flash_attn_func\s*\("
    ),
    "flash_attn_qkvpacked": re.compile(
        r"flash_attn_qkvpacked_func\s*\("
    ),
    "flash_attn_kvpacked": re.compile(
        r"flash_attn_kvpacked_func\s*\("
    ),
    "flash_attn_varlen": re.compile(
        r"flash_attn_varlen_func\s*\("
    ),
    "flash_attn_import": re.compile(
        r"from\s+flash_attn\s+import"
    ),
    "flash_attn_module_import": re.compile(
        r"import\s+flash_attn"
    ),

    # -----------------------------------------------------------------------
    # Eager (manual) attention -- migration targets
    # -----------------------------------------------------------------------
    "eager_matmul_transpose": re.compile(
        r"q\s*@\s*k\.transpose\s*\(\s*-2\s*,\s*-1\s*\)"
    ),
    "eager_torch_matmul": re.compile(
        r"torch\.matmul\s*\(\s*\w+\s*,\s*\w+\.transpose\s*\(\s*-2\s*,\s*-1\s*\)\s*\)"
    ),
    "eager_bmm": re.compile(
        r"torch\.bmm\s*\(\s*\w+\s*,\s*\w+\.transpose\s*\("
    ),
    "eager_einsum_bhld_bhsd": re.compile(
        r'einsum\s*\(\s*["\']bhld,bhsd->bhls["\']'
    ),
    "eager_einsum_bqhd_bkhd": re.compile(
        r'einsum\s*\(\s*["\']bqhd,bkhd->bhqk["\']'
    ),
    "eager_softmax_chain": re.compile(
        r"F\.softmax\s*\(.*\*\s*(?:math\.sqrt|self\._scale|scale|1\s*/\s*math\.sqrt)"
    ),
    "eager_softmax_explicit": re.compile(
        r"torch\.softmax\s*\(\s*\w+\s*,\s*dim\s*=\s*-1\s*\)"
    ),
    "eager_attn_weights": re.compile(
        r"(?:attention_weights|attn_weights)\s*=\s*(?:torch\.)?(?:matmul|bmm)"
    ),
    "eager_scores_variable": re.compile(
        r"(?:attention_scores|attn_scores)\s*=\s*\w+\s*@\s*\w+"
    ),
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AuditMatch:
    """A single pattern match found in the codebase."""
    file: str
    line: int
    pattern_name: str
    snippet: str

    def to_dict(self):
        return {
            "file": self.file,
            "line": self.line,
            "pattern_name": self.pattern_name,
            "snippet": self.snippet,
        }


@dataclass
class AuditReport:
    """Results from scanning a codebase directory."""
    root: str
    files_scanned: int = 0
    matches: List[AuditMatch] = field(default_factory=list)

    # Convenience properties
    @property
    def has_sdpa(self) -> bool:
        return any(m.pattern_name.startswith("sdpa_") for m in self.matches)

    @property
    def has_eager(self) -> bool:
        return any(m.pattern_name.startswith("eager_") for m in self.matches)

    @property
    def has_xformers(self) -> bool:
        return any(m.pattern_name.startswith("xformers_") for m in self.matches)

    @property
    def has_external_flash(self) -> bool:
        return any(m.pattern_name.startswith("flash_attn_") for m in self.matches)

    def summary(self) -> str:
        lines = [
            f"Audit of: {self.root}",
            f"Files scanned: {self.files_scanned}",
            f"Total matches: {len(self.matches)}",
            f"  SDPA:          {self.has_sdpa}",
            f"  Eager:         {self.has_eager}",
            f"  xFormers:      {self.has_xformers}",
            f"  External FA:   {self.has_external_flash}",
        ]
        return "\n".join(lines)

    def to_dict(self):
        return {
            "root": self.root,
            "files_scanned": self.files_scanned,
            "summary": {
                "has_sdpa": self.has_sdpa,
                "has_eager": self.has_eager,
                "has_xformers": self.has_xformers,
                "has_external_flash": self.has_external_flash,
            },
            "matches": [m.to_dict() for m in self.matches],
        }


@dataclass
class CapabilityReport:
    """Results from probing SDPA backend capabilities."""
    device: str = "cpu"
    device_name: str = "CPU"
    cuda_capability: str = "N/A"

    flash_built_in: bool = False
    can_flash: bool = False
    can_efficient: bool = False
    can_cudnn: bool = False

    debug_reasons: Dict[str, List[str]] = field(default_factory=lambda: {
        "flash": [], "efficient": [], "cudnn": []
    })

    q_shape: Tuple = ()
    k_shape: Tuple = ()
    v_shape: Tuple = ()
    dtype: str = "unknown"
    dropout_p: float = 0.0
    is_causal: bool = False

    def recommended_backend(self) -> str:
        if self.can_flash:
            return "flash"
        if self.can_efficient:
            return "efficient"
        if self.can_cudnn:
            return "cudnn"
        return "math"

    def to_dict(self):
        return {
            "device": self.device,
            "device_name": self.device_name,
            "cuda_capability": self.cuda_capability,
            "flash_built_in": self.flash_built_in,
            "can_flash": self.can_flash,
            "can_efficient": self.can_efficient,
            "can_cudnn": self.can_cudnn,
            "debug_reasons": self.debug_reasons,
            "probe_inputs": {
                "q_shape": list(self.q_shape),
                "k_shape": list(self.k_shape),
                "v_shape": list(self.v_shape),
                "dtype": self.dtype,
                "dropout_p": self.dropout_p,
                "is_causal": self.is_causal,
            },
            "recommended_backend": self.recommended_backend(),
        }

    def format_summary(self) -> str:
        lines = [
            f"Capability Report ({self.device} / {self.device_name})",
            f"  CUDA capability : {self.cuda_capability}",
            f"  Flash built-in  : {self.flash_built_in}",
            f"  can_flash       : {self.can_flash}",
            f"  can_efficient   : {self.can_efficient}",
            f"  can_cudnn       : {self.can_cudnn}",
            f"  Recommended     : {self.recommended_backend()}",
        ]
        for backend, reasons in self.debug_reasons.items():
            if reasons:
                lines.append(f"  {backend} debug: " + "; ".join(reasons[:2]))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# AttentionAuditor
# ---------------------------------------------------------------------------


class AttentionAuditor:
    """
    Audit attention implementations via static scan and runtime probing.

    Example:
        auditor = AttentionAuditor()
        audit = auditor.scan_codebase("/path/to/myproject")
        print(audit.summary())

        q = torch.randn(2, 8, 512, 64)
        cap = auditor.probe_runtime(q, q, q, None, 0.0, False)
        print(cap.format_summary())

        auditor.save_report((audit, cap), "attn_report.json")
    """

    def scan_codebase(
        self,
        root: str,
        extensions: Optional[List[str]] = None,
    ) -> AuditReport:
        """
        Walk root directory and scan all matching files for attention patterns.

        Args:
            root       : Directory path to scan recursively.
            extensions : File extensions to check. Default: ['.py']

        Returns:
            AuditReport with all detected matches and file count.
        """
        if extensions is None:
            extensions = [".py"]

        report = AuditReport(root=root)

        if not os.path.isdir(root):
            logger.warning("scan_codebase: %r is not a directory", root)
            return report

        for dirpath, _dirs, filenames in os.walk(root):
            for fname in filenames:
                if not any(fname.endswith(ext) for ext in extensions):
                    continue
                fpath = os.path.join(dirpath, fname)
                try:
                    new_matches = self._scan_file(fpath)
                    report.matches.extend(new_matches)
                    report.files_scanned += 1
                except Exception as exc:
                    logger.debug("Failed to scan %s: %s", fpath, exc)
                    # Still count the file as scanned (attempted)
                    report.files_scanned += 1

        return report

    def _scan_file(self, filepath: str) -> List[AuditMatch]:
        """Scan a single file and return all pattern matches."""
        matches: List[AuditMatch] = []
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
        except OSError:
            return matches

        for lineno, line in enumerate(lines, start=1):
            for pattern_name, pattern in ATTENTION_PATTERNS.items():
                if pattern.search(line):
                    matches.append(AuditMatch(
                        file=os.path.abspath(filepath),
                        line=lineno,
                        pattern_name=pattern_name,
                        snippet=line.rstrip(),
                    ))
        return matches

    # -----------------------------------------------------------------------
    # Runtime probe
    # -----------------------------------------------------------------------

    def probe_runtime(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
    ) -> CapabilityReport:
        """
        Run SDPA capability checks for the given probe tensors.

        On CPU, all fused backends return False (they require CUDA).
        On CUDA, each backend is tested via can_use_* with debug=True.

        Args:
            q, k, v    : Representative query/key/value tensors.
                         Shape: (B, H, S, D) -- SDPA convention.
            attn_mask  : Optional mask tensor.
            dropout_p  : Dropout probability for the probe.
            is_causal  : Whether to probe for causal attention.

        Returns:
            CapabilityReport with per-backend can_use results.
        """
        report = CapabilityReport(
            q_shape=tuple(q.shape),
            k_shape=tuple(k.shape),
            v_shape=tuple(v.shape),
            dtype=str(q.dtype),
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

        # Device info
        device = q.device
        report.device = str(device)

        if device.type == "cuda":
            props = torch.cuda.get_device_properties(device)
            report.device_name = props.name
            sm = props.major * 10 + props.minor
            report.cuda_capability = f"{props.major}.{props.minor}"
        else:
            report.device_name = "CPU"
            report.cuda_capability = "N/A"
            # All fused backends require CUDA -- short-circuit
            report.debug_reasons = {
                "flash": ["Flash Attention requires CUDA device"],
                "efficient": ["Efficient Attention requires CUDA device"],
                "cudnn": ["cuDNN Attention requires CUDA device"],
            }
            return report

        # Check if Flash is compiled into this PyTorch build
        report.flash_built_in = self._check_flash_built_in()

        # Build SDPAParams and run capability checks
        params = self._build_sdpa_params(q, k, v, attn_mask, dropout_p, is_causal)
        if params is None:
            report.debug_reasons = {
                "flash": ["SDPAParams construction failed"],
                "efficient": ["SDPAParams construction failed"],
                "cudnn": ["SDPAParams construction failed"],
            }
            return report

        # Flash
        can_flash, flash_reasons = self._check_capability(
            torch.backends.cuda.can_use_flash_attention, params
        )
        report.can_flash = can_flash
        report.debug_reasons["flash"] = flash_reasons

        # Efficient
        can_eff, eff_reasons = self._check_capability(
            torch.backends.cuda.can_use_efficient_attention, params
        )
        report.can_efficient = can_eff
        report.debug_reasons["efficient"] = eff_reasons

        # cuDNN
        can_cudnn, cudnn_reasons = self._check_capability(
            torch.backends.cuda.can_use_cudnn_attention, params
        )
        report.can_cudnn = can_cudnn
        report.debug_reasons["cudnn"] = cudnn_reasons

        return report

    def _check_flash_built_in(self) -> bool:
        """Return True if PyTorch was compiled with the Flash backend."""
        try:
            from torch.backends.cuda import flash_sdp_enabled
            return flash_sdp_enabled()
        except ImportError:
            pass
        # Alternative check: try enabling it and see if it exists
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            return True
        except AttributeError:
            return False

    def _build_sdpa_params(
        self,
        q, k, v, attn_mask, dropout_p, is_causal
    ):
        """Construct SDPAParams, returning None on failure."""
        try:
            return torch.backends.cuda.SDPAParams(q, k, v, attn_mask, dropout_p, is_causal)
        except Exception as exc:
            logger.debug("SDPAParams construction failed: %s", exc)
            return None

    def _check_capability(self, check_fn, params) -> Tuple[bool, List[str]]:
        """
        Run a can_use_* function with debug=True, capturing stderr output.

        Returns:
            (can_use: bool, reasons: list of debug strings)
        """
        buf = io.StringIO()
        result = False
        try:
            with contextlib.redirect_stderr(buf):
                result = check_fn(params, debug=True)
        except Exception as exc:
            return False, [str(exc)]

        stderr_text = buf.getvalue().strip()
        reasons = (
            [line.strip() for line in stderr_text.splitlines() if line.strip()]
            if stderr_text else []
        )
        return result, reasons

    # -----------------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------------

    def save_report(
        self,
        report,
        path: str,
    ) -> None:
        """
        Save audit and/or capability report to a JSON file.

        report can be:
          - AuditReport
          - CapabilityReport
          - tuple of (AuditReport, CapabilityReport)
        """
        output = {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        if isinstance(report, tuple):
            audit_rep, cap_rep = report
            output["audit"] = audit_rep.to_dict()
            output["capability"] = cap_rep.to_dict()
        elif isinstance(report, AuditReport):
            output["audit"] = report.to_dict()
        elif isinstance(report, CapabilityReport):
            output["capability"] = report.to_dict()
        else:
            raise TypeError(f"Unknown report type: {type(report)}")

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2)

        logger.info("Attention report saved to %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AttentionAuditor: scan codebase and/or probe SDPA capabilities."
    )
    p.add_argument(
        "--scan", metavar="ROOT",
        help="Directory to scan for attention patterns.",
    )
    p.add_argument(
        "--run", action="store_true",
        help="Run runtime capability probe on synthetic tensors.",
    )
    p.add_argument(
        "--save", metavar="PATH", default=None,
        help="Save report to this JSON file path.",
    )
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for runtime probe. Default: cuda if available else cpu.",
    )
    p.add_argument(
        "--head-dim", type=int, default=64,
        help="Head dim for runtime probe (default 64).",
    )
    p.add_argument(
        "--num-heads", type=int, default=8,
        help="Num heads for runtime probe (default 8).",
    )
    p.add_argument(
        "--seq-len", type=int, default=512,
        help="Sequence length for runtime probe (default 512).",
    )
    p.add_argument(
        "--dtype", default="float16",
        choices=["float32", "float16", "bfloat16"],
        help="Dtype for runtime probe (default float16).",
    )
    return p


def _run_cli(argv=None) -> None:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    auditor = AttentionAuditor()
    audit_report: Optional[AuditReport] = None
    cap_report: Optional[CapabilityReport] = None

    if args.scan:
        print(f"Scanning: {args.scan}")
        audit_report = auditor.scan_codebase(args.scan)
        print(audit_report.summary())
        print()

    if args.run:
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[args.dtype]
        device = torch.device(args.device)

        print(f"Running capability probe on {device} with dtype={dtype}, "
              f"heads={args.num_heads}, head_dim={args.head_dim}, seq={args.seq_len}")
        shape = (2, args.num_heads, args.seq_len, args.head_dim)
        q = torch.randn(*shape, dtype=dtype, device=device)
        cap_report = auditor.probe_runtime(q, q.clone(), q.clone(), None, 0.0, False)
        print(cap_report.format_summary())

    if args.save:
        if audit_report and cap_report:
            auditor.save_report((audit_report, cap_report), args.save)
        elif audit_report:
            auditor.save_report(audit_report, args.save)
        elif cap_report:
            auditor.save_report(cap_report, args.save)
        else:
            print("Nothing to save. Use --scan and/or --run.")
        if args.save:
            print(f"Report saved to {args.save}")


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


def _run_self_tests() -> None:
    import tempfile

    print("Running attention_auditor_template self-tests...")
    failures: List[str] = []

    auditor = AttentionAuditor()

    # --- Pattern detection ---
    test_cases = [
        ("F.scaled_dot_product_attention(q, k, v)", "sdpa_functional"),
        ("torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)",
         "sdpa_torch_nn_functional"),
        ("xformers.ops.memory_efficient_attention(q, k, v)", "xformers_memory_efficient"),
        ("from xformers.ops import memory_efficient_attention", "xformers_mem_eff_import"),
        ("flash_attn_func(q, k, v, causal=True)", "flash_attn_func"),
        ("flash_attn_qkvpacked_func(qkv)", "flash_attn_qkvpacked"),
        ("from flash_attn import flash_attn_func", "flash_attn_import"),
        ("scores = q @ k.transpose(-2, -1)", "eager_matmul_transpose"),
        ("attn_weights = torch.matmul(q, k.transpose(-2, -1))", "eager_attn_weights"),
    ]

    for source_line, expected_pattern in test_cases:
        matched_patterns = [
            name for name, pat in ATTENTION_PATTERNS.items()
            if pat.search(source_line)
        ]
        if expected_pattern in matched_patterns:
            print(f"  PASS: detect {expected_pattern!r}")
        else:
            failures.append(
                f"Pattern {expected_pattern!r} not detected in: {source_line!r}. "
                f"Got: {matched_patterns}"
            )

    # --- scan_codebase on temp directory ---
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a file with known patterns
            src1 = os.path.join(tmpdir, "model.py")
            with open(src1, "w") as fh:
                fh.write("import torch\n")
                fh.write("out = F.scaled_dot_product_attention(q, k, v)\n")
                fh.write("scores = q @ k.transpose(-2, -1)\n")

            src2 = os.path.join(tmpdir, "utils.py")
            with open(src2, "w") as fh:
                fh.write("# nothing here\nprint('hello')\n")

            report = auditor.scan_codebase(tmpdir)
            assert report.files_scanned == 2, f"Expected 2 files, got {report.files_scanned}"
            assert report.has_sdpa, "Should detect SDPA pattern"
            assert report.has_eager, "Should detect eager pattern"
            assert not report.has_xformers, "Should not detect xformers"
            assert len(report.matches) >= 2
            print("  PASS: scan_codebase detects SDPA and eager patterns")
    except Exception as e:
        failures.append(f"scan_codebase: {e}")

    # --- scan_codebase on nonexistent directory ---
    try:
        r = auditor.scan_codebase("/nonexistent/path/xyz")
        assert r.files_scanned == 0
        assert len(r.matches) == 0
        print("  PASS: scan_codebase on nonexistent dir returns empty report")
    except Exception as e:
        failures.append(f"scan_codebase nonexistent: {e}")

    # --- probe_runtime on CPU ---
    try:
        q = torch.randn(2, 4, 64, 32)  # fp32 CPU
        cap = auditor.probe_runtime(q, q.clone(), q.clone(), None, 0.0, False)
        assert cap.device.startswith("cpu")
        assert cap.can_flash is False
        assert cap.can_efficient is False
        assert cap.can_cudnn is False
        assert isinstance(cap.debug_reasons, dict)
        assert "flash" in cap.debug_reasons
        assert cap.q_shape == (2, 4, 64, 32)
        assert cap.recommended_backend() == "math"
        print("  PASS: probe_runtime on CPU returns correct capability report")
    except Exception as e:
        failures.append(f"probe_runtime CPU: {e}")

    # --- format_summary ---
    try:
        q = torch.randn(2, 4, 64, 32)
        cap = auditor.probe_runtime(q, q.clone(), q.clone(), None, 0.0, False)
        summary = cap.format_summary()
        assert "can_flash" in summary
        assert "math" in summary or "flash" in summary  # recommended_backend
        print("  PASS: CapabilityReport.format_summary() returns useful string")
    except Exception as e:
        failures.append(f"format_summary: {e}")

    # --- save_report ---
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "attn_report.json")
            q = torch.randn(2, 4, 64, 32)
            cap = auditor.probe_runtime(q, q.clone(), q.clone())
            with tempfile.TemporaryDirectory() as tmpdir2:
                fpath = os.path.join(tmpdir2, "subdir", "report.json")
                # Check that save_report creates subdirs
                auditor.save_report(cap, fpath)
                assert os.path.exists(fpath), "Report file not created"
                with open(fpath) as fh:
                    data = json.load(fh)
                assert "capability" in data
                assert "schema_version" in data
            print("  PASS: save_report creates JSON file with correct schema")
    except Exception as e:
        failures.append(f"save_report: {e}")

    # --- AuditReport properties ---
    try:
        matches = [
            AuditMatch("a.py", 1, "sdpa_functional", "F.scaled_dot_product_attention(...)"),
            AuditMatch("b.py", 5, "eager_matmul_transpose", "q @ k.transpose(-2, -1)"),
        ]
        rep = AuditReport(root="/tmp", files_scanned=3, matches=matches)
        assert rep.has_sdpa
        assert rep.has_eager
        assert not rep.has_xformers
        assert not rep.has_external_flash
        d = rep.to_dict()
        assert d["files_scanned"] == 3
        assert len(d["matches"]) == 2
        print("  PASS: AuditReport properties and to_dict")
    except Exception as e:
        failures.append(f"AuditReport: {e}")

    # --- Summary ---
    if failures:
        print(f"\nFAILED {len(failures)} tests:")
        for f in failures:
            print(f"  FAIL: {f}")
        raise SystemExit(1)
    else:
        print(f"\nAll attention_auditor_template self-tests PASSED")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # If called with arguments, run CLI; otherwise run self-tests
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--test"):
        _run_cli()
    else:
        _run_self_tests()
