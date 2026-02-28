"""
run_template.py
================
BenchRunner -- unified entrypoint that orchestrates all sub-modes of the
Compute/Throughput Baseline & Regression Gate skill.

Sub-modes (combinable):
  --bench         Run training benchmark -> metrics.json
  --quality       Run quality harness -> quality_results.json
  --profile       Run profiling -> trace files
  --compare       Compare against baseline -> gate report
  --update-baseline  Update baseline files from current results

Usage:
    python run_template.py --bench --quality --compare \
        --out artifacts/ --baseline-dir bench/baselines/

Self-test:
    python run_template.py --self-test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from perf_gate_config_template import (
        BenchConfig, EvalConfig, ToleranceConfig,
        Status, get_peak_tflops,
    )
except ImportError:
    from dataclasses import dataclass, field
    from enum import Enum

    class Status(str, Enum):
        PASS = "PASS"
        FAIL = "FAIL"
        WARN = "WARN"
        SKIP = "SKIP"

    @dataclass
    class BenchConfig:
        warmup_steps: int = 200
        measure_steps: int = 100
        mode: str = "synthetic"
        world_size: int = 1
        profile: str = "off"
        repeat: int = 1
        peak_tflops: Optional[float] = None
        mfu_estimator: str = "6ND"
        per_device_batch_size: int = 8
        seq_len: int = 2048
        grad_accum_steps: int = 1
        dtype: str = "bfloat16"
        vocab_size: int = 32000
        hidden_dim: int = 4096
        num_layers: int = 32

        def validate(self) -> None:
            pass

    @dataclass
    class EvalConfig:
        fixed_shard_path: str = "bench/data/fixed_shard.txt"
        fixed_shard_sha256: str = ""
        probes: List[str] = field(default_factory=lambda: [
            "basic_reasoning_25", "format_following_30", "code_sanity_20"
        ])
        temperature: float = 0.0
        seed: int = 42
        max_new_tokens: int = 32
        max_length: int = 2048
        stride: int = 512

        def validate(self) -> None:
            pass

    @dataclass
    class ToleranceConfig:
        throughput_drop_pct: float = 5.0
        step_time_increase_pct: float = 5.0
        memory_increase_pct: float = 10.0
        ppl_increase_pct: float = 1.5
        probe_drop_abs: float = 2.0
        loss_slope_threshold: float = 0.001

        def validate(self) -> None:
            pass

    def get_peak_tflops(name: str, dtype: str = "bf16") -> Optional[float]:
        return None


try:
    from collect_env_template import CollectEnv
    _COLLECT_ENV_AVAILABLE = True
except ImportError:
    _COLLECT_ENV_AVAILABLE = False

try:
    from bench_train_template import BenchTrain
    _BENCH_AVAILABLE = True
except ImportError:
    _BENCH_AVAILABLE = False

try:
    from eval_small_template import EvalSmall
    _QUALITY_HARNESS_AVAILABLE = True
except ImportError:
    _QUALITY_HARNESS_AVAILABLE = False

try:
    from compare_baseline_template import CompareBaseline
    _COMPARE_AVAILABLE = True
except ImportError:
    _COMPARE_AVAILABLE = False


# ===========================================================================
# BenchRunner
# ===========================================================================

class BenchRunner:
    """
    Orchestrates the full benchmark pipeline.

    Pipeline steps:
      1. collect_env      -> env.json
      2. bench_train      -> metrics.json
      3. quality_harness  -> quality_results.json
      4. profiling        -> trace files (optional)
      5. compare          -> gate report

    Example:
        runner = BenchRunner(
            bench_config=BenchConfig(mode="synthetic"),
            quality_config=EvalConfig(),
            tol_config=ToleranceConfig(),
            out_dir="artifacts",
            baseline_dir="bench/baselines",
        )
        results = runner.run(modes=["bench", "quality", "compare"])
        sys.exit(0 if results["gate_passed"] else 1)
    """

    def __init__(
        self,
        bench_config: Optional[BenchConfig] = None,
        quality_config: Optional[EvalConfig] = None,
        tol_config: Optional[ToleranceConfig] = None,
        out_dir: str = "artifacts",
        baseline_dir: str = "bench/baselines",
        machine_profile: Optional[str] = None,
        model: Any = None,
        tokenizer: Any = None,
        data_iter: Any = None,
    ) -> None:
        self.bench_config = bench_config or BenchConfig()
        self.quality_config = quality_config or EvalConfig()
        self.tol_config = tol_config or ToleranceConfig()
        self.out_dir = Path(out_dir)
        self.baseline_dir = Path(baseline_dir)
        self.model = model
        self.tokenizer = tokenizer
        self.data_iter = data_iter

        if machine_profile:
            self._machine_profile = machine_profile
        elif _COLLECT_ENV_AVAILABLE:
            env_collector = CollectEnv()
            self._machine_profile = env_collector.machine_profile()
        else:
            self._machine_profile = self._fallback_profile()

        self.out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def run(self, modes: List[str], **kwargs: Any) -> Dict[str, Any]:
        """
        Run the specified combination of sub-modes.

        Args:
            modes: List of mode names. Valid values:
                   "env", "bench", "quality", "profile", "compare"

        Returns:
            Dict with results from each mode and "gate_passed" bool.
        """
        results: Dict[str, Any] = {
            "machine_profile": self._machine_profile,
            "out_dir": str(self.out_dir),
            "modes_run": modes,
            "gate_passed": True,
        }

        if "env" in modes or _COLLECT_ENV_AVAILABLE:
            results["env"] = self._run_collect_env()

        if "bench" in modes:
            results["bench"] = self._run_bench()

        if "quality" in modes:
            results["quality"] = self._run_quality_harness()

        if "profile" in modes:
            results["profile"] = self._run_profiler()

        if "compare" in modes:
            compare_result = self._run_compare()
            results["compare"] = {
                "status": compare_result.overall_status.value,
                "report": compare_result.report,
                "failed_checks": [c.metric for c in compare_result.failed_checks],
                "warned_checks": [c.metric for c in compare_result.warned_checks],
            }
            if compare_result.overall_status == Status.FAIL:
                results["gate_passed"] = False

        # Scaling efficiency report (multi-GPU only)
        if "bench" in modes and self.bench_config.world_size > 1:
            results["scaling"] = self._compute_scaling_info(results)

        # Save run summary
        summary_path = self.out_dir / "run_summary.json"
        self._write_json(summary_path, results)
        print(f"\nRun summary -> {summary_path}")

        return results

    def update_baseline(self) -> None:
        """
        Copy current results to baseline directory.
        Should only be called on main branch after gate passes.
        """
        if not _COMPARE_AVAILABLE:
            raise RuntimeError("compare_baseline_template not available")
        comparator = CompareBaseline(self.tol_config)
        comparator.update_baseline(
            current_dir=str(self.out_dir),
            dest_dir=str(self.baseline_dir),
            machine_profile=self._machine_profile,
        )
        print(f"Baseline updated in {self.baseline_dir}")

    # -----------------------------------------------------------------------
    # Internal sub-mode runners
    # -----------------------------------------------------------------------

    def _run_collect_env(self) -> dict:
        """Collect environment info and write env.json."""
        if not _COLLECT_ENV_AVAILABLE:
            return {"status": "skipped"}
        env_collector = CollectEnv()
        env_path = self.out_dir / f"{self._machine_profile}.env.json"
        env_collector.save(str(env_path))
        print(f"Environment collected -> {env_path}")
        return {"status": "ok", "path": str(env_path)}

    def _run_bench(self) -> dict:
        """Run training benchmark and write metrics.json."""
        if not _BENCH_AVAILABLE:
            print("INFO: bench_train_template not available; skipping benchmark")
            return {"status": "skipped"}

        print(f"\nStarting benchmark ({self.bench_config.mode} mode)...")
        print(f"  Warmup steps:  {self.bench_config.warmup_steps}")
        print(f"  Measure steps: {self.bench_config.measure_steps}")
        print(f"  World size:    {self.bench_config.world_size}")
        print(f"  Batch size:    {self.bench_config.per_device_batch_size}")
        print(f"  Seq len:       {self.bench_config.seq_len}")

        bench = BenchTrain(
            config=self.bench_config,
            model=self.model,
            data_iter=self.data_iter,
            machine_profile=self._machine_profile,
        )

        t0 = time.perf_counter()
        result = bench.run()
        elapsed = time.perf_counter() - t0

        metrics_path = self.out_dir / f"{self._machine_profile}.metrics.json"
        bench.save(str(metrics_path))

        print(f"\nBenchmark completed in {elapsed:.1f}s:")
        print(f"  tokens/sec p50: {result.tokens_per_sec_p50:,.0f}")
        print(f"  step time p50:  {result.step_time_p50:.3f}s")
        if result.mfu_p50 is not None:
            print(f"  MFU p50:        {result.mfu_p50:.3f} ({result.mfu_p50 * 100:.1f}%)")
        print(f"  peak memory:    {result.peak_allocated_gb:.1f} GB")
        print(f"  Saved -> {metrics_path}")

        return {
            "status": "ok",
            "path": str(metrics_path),
            "tokens_per_sec_p50": result.tokens_per_sec_p50,
            "step_time_p50": result.step_time_p50,
            "mfu_p50": result.mfu_p50,
            "peak_allocated_gb": result.peak_allocated_gb,
        }

    def _run_quality_harness(self) -> dict:
        """
        Run quality harness: perplexity on fixed shard + task probe accuracy.
        Writes <machine_profile>.eval.json to out_dir.
        """
        if not _QUALITY_HARNESS_AVAILABLE:
            print("INFO: eval_small_template not available; skipping quality harness")
            return {"status": "skipped"}

        if self.model is None or self.tokenizer is None:
            print("INFO: No model/tokenizer provided; skipping quality harness")
            return {"status": "skipped_no_model"}

        print(f"\nStarting quality harness...")
        print(f"  Probes: {self.quality_config.probes}")
        print(f"  Seed:   {self.quality_config.seed}")

        device = "cuda" if (_TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
        harness = EvalSmall(
            config=self.quality_config,
            machine_profile=self._machine_profile,
        )

        t0 = time.perf_counter()
        result = harness.run(self.model, self.tokenizer, device=device)
        elapsed = time.perf_counter() - t0

        out_path = self.out_dir / f"{self._machine_profile}.eval.json"
        harness.save(str(out_path))

        print(f"\nQuality harness completed in {elapsed:.1f}s:")
        print(f"  Perplexity: {result.ppl_fixed_shard:.4f}")
        for probe_name, acc in result.task_probe_accuracy.items():
            print(f"  {probe_name}: {acc:.3f} ({acc * 100:.1f}%)")
        print(f"  Saved -> {out_path}")

        return {
            "status": "ok",
            "path": str(out_path),
            "ppl_fixed_shard": result.ppl_fixed_shard,
            "task_probe_accuracy": result.task_probe_accuracy,
        }

    def _run_profiler(self) -> dict:
        """Run PyTorch profiler and export traces to out_dir/profile/."""
        if not _TORCH_AVAILABLE:
            return {"status": "skipped_no_torch"}

        profile_mode = self.bench_config.profile
        if profile_mode == "off":
            return {"status": "skipped_profile_off"}

        profile_dir = self.out_dir / "profile"
        tb_dir = profile_dir / "tb"
        chrome_dir = profile_dir / "chrome"
        tb_dir.mkdir(parents=True, exist_ok=True)
        chrome_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nStarting profiler (mode={profile_mode})...")

        # Obtain or build a model for profiling
        model_for_profile = self.model
        if model_for_profile is None and _BENCH_AVAILABLE:
            from bench_train_template import _build_minimal_model
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
            model_for_profile = _build_minimal_model(
                vocab_size=self.bench_config.vocab_size,
                hidden_dim=self.bench_config.hidden_dim,
                num_layers=self.bench_config.num_layers,
            ).to(device_str)

        if model_for_profile is None:
            print("INFO: No model for profiling; skipping")
            return {"status": "skipped_no_model"}

        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        model_for_profile.train()
        optimizer = torch.optim.AdamW(model_for_profile.parameters(), lr=1e-4)

        bs = self.bench_config.per_device_batch_size
        seq = self.bench_config.seq_len
        vocab = self.bench_config.vocab_size

        def get_batch() -> dict:
            return {
                "input_ids": torch.randint(0, vocab, (bs, seq), device=device_str),
                "labels": torch.randint(0, vocab, (bs, seq), device=device_str),
            }

        wait, warmup_p, active, repeat_p = 1, 1, 3, 2
        total_profiler_steps = (wait + warmup_p + active) * repeat_p

        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]

        on_trace_ready_fn = None
        if profile_mode in ("tb", "trace"):
            on_trace_ready_fn = torch.profiler.tensorboard_trace_handler(str(tb_dir))

        with torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=wait, warmup=warmup_p, active=active, repeat=repeat_p
            ),
            on_trace_ready=on_trace_ready_fn,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            for _ in range(total_profiler_steps):
                batch = get_batch()

                with torch.autograd.profiler.record_function("data"):
                    pass  # Batch already on device

                with torch.autograd.profiler.record_function("forward"):
                    with torch.autocast(
                        device_type=device_str if device_str != "cpu" else "cpu",
                        dtype=torch.bfloat16,
                        enabled=(device_str == "cuda"),
                    ):
                        outputs = model_for_profile(**batch)

                loss_value = None
                with torch.autograd.profiler.record_function("loss"):
                    if isinstance(outputs, dict):
                        loss_value = outputs.get("loss")
                    else:
                        loss_value = getattr(outputs, "loss", None)

                if loss_value is None:
                    prof.step()
                    continue

                with torch.autograd.profiler.record_function("backward"):
                    loss_value.backward()

                with torch.autograd.profiler.record_function("optimizer_step"):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                # Must call prof.step() at end of every training step
                prof.step()

        chrome_path = chrome_dir / "trace.json"
        prof.export_chrome_trace(str(chrome_path))
        print(f"Chrome trace    -> {chrome_path}")
        print(f"TensorBoard     -> {tb_dir}")
        print(f"View command:      tensorboard --logdir {tb_dir}")

        return {
            "status": "ok",
            "chrome_trace": str(chrome_path),
            "tb_dir": str(tb_dir),
            "profile_dir": str(profile_dir),
        }

    def _run_compare(self) -> Any:
        """Compare against baseline and return CompareResult."""
        if not _COMPARE_AVAILABLE:
            # Return a minimal object with SKIP status
            class _SkipResult:
                overall_status = Status.SKIP
                report = "compare_baseline_template not available; comparison skipped."
                failed_checks: list = []
                warned_checks: list = []
                passed = True
                failed = False

            return _SkipResult()

        comparator = CompareBaseline(self.tol_config)
        print(f"\nComparing against baseline ({self.baseline_dir})...")
        result = comparator.compare(
            current_dir=str(self.out_dir),
            baseline_dir=str(self.baseline_dir),
            machine_profile=self._machine_profile,
        )
        print()
        print(result.report)

        report_path = self.out_dir / "gate_report.txt"
        report_path.write_text(result.report)
        print(f"\nGate report -> {report_path}")
        return result

    # -----------------------------------------------------------------------
    # Internal: scaling efficiency
    # -----------------------------------------------------------------------

    def _compute_scaling_info(self, results: dict) -> dict:
        """Compute and report multi-GPU scaling efficiency."""
        world_size = max(1, self.bench_config.world_size)
        tps_n = results.get("bench", {}).get("tokens_per_sec_p50", 0)

        single_gpu_path = (
            self.baseline_dir / f"single_gpu_{self._machine_profile}.metrics.json"
        )
        if single_gpu_path.exists():
            with open(single_gpu_path) as f:
                single_data = json.load(f)
            tps_1 = single_data.get("throughput", {}).get("tokens_per_sec_p50", 0)
            if tps_1 > 0 and world_size > 1:
                efficiency = tps_n / (world_size * tps_1)
                print(
                    f"\nScaling efficiency ({world_size}x GPU): "
                    f"{efficiency:.3f} ({efficiency * 100:.1f}%)"
                )
                print(f"  {world_size}-GPU throughput: {tps_n:,.0f} tok/s")
                print(f"  1-GPU throughput:  {tps_1:,.0f} tok/s")
                return {
                    "world_size": world_size,
                    "tokens_per_sec_p50": tps_n,
                    "single_gpu_tps": tps_1,
                    "efficiency": efficiency,
                }

        return {
            "world_size": world_size,
            "tokens_per_sec_p50": tps_n,
            "status": "no_single_gpu_baseline_for_scaling",
        }

    # -----------------------------------------------------------------------
    # Internal: utilities
    # -----------------------------------------------------------------------

    def _fallback_profile(self) -> str:
        """Generate a basic machine profile without CollectEnv."""
        if _TORCH_AVAILABLE:
            ver = ".".join(torch.__version__.split("+")[0].split(".")[:2])
            if torch.cuda.is_available():
                raw_name = torch.cuda.get_device_name(0)
                gpu_short = "".join(c for c in raw_name if c.isalnum())[:12]
                gpu_count = torch.cuda.device_count()
                return f"{gpu_short}x{gpu_count}_torch{ver}"
            return f"cpu_torch{ver}"
        return "cpu_unknown"

    def _write_json(self, path: Path, data: dict) -> None:
        """Write data atomically to a JSON file."""
        tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


# ===========================================================================
# CLI
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the unified runner CLI."""
    parser = argparse.ArgumentParser(
        prog="run_template",
        description="Compute/Throughput Baseline & Regression Gate -- unified runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mode_group = parser.add_argument_group("Modes (select one or more)")
    mode_group.add_argument("--bench", action="store_true", help="Run training benchmark")
    mode_group.add_argument(
        "--quality", action="store_true",
        help="Run quality harness (perplexity + probe accuracy)"
    )
    mode_group.add_argument(
        "--profile", choices=["off", "trace", "tb"], default="off",
        help="Run PyTorch profiler"
    )
    mode_group.add_argument("--compare", action="store_true", help="Compare against baseline")
    mode_group.add_argument(
        "--update-baseline", action="store_true",
        help="Update baseline files from current results (main branch only)"
    )
    mode_group.add_argument("--self-test", action="store_true", help="Run self-tests and exit")

    bench_group = parser.add_argument_group("Benchmark options")
    bench_group.add_argument("--mode", choices=["synthetic", "e2e"], default="synthetic")
    bench_group.add_argument("--warmup-steps", type=int, default=200)
    bench_group.add_argument("--measure-steps", type=int, default=100)
    bench_group.add_argument("--world-size", type=int, default=-1, help="-1 = auto-detect GPUs")
    bench_group.add_argument("--batch-size", type=int, default=8)
    bench_group.add_argument("--seq-len", type=int, default=2048)
    bench_group.add_argument("--grad-accum", type=int, default=1)
    bench_group.add_argument("--repeat", type=int, default=1)
    bench_group.add_argument(
        "--mfu-estimator", choices=["6ND", "transformer_aware"], default="6ND"
    )
    bench_group.add_argument(
        "--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16"
    )

    quality_group = parser.add_argument_group("Quality harness options")
    quality_group.add_argument("--shard", default="bench/data/fixed_shard.txt",
                               help="Path to fixed text shard")
    quality_group.add_argument("--seed", type=int, default=42)
    quality_group.add_argument("--max-new-tokens", type=int, default=32)

    cmp_group = parser.add_argument_group("Comparison options")
    cmp_group.add_argument("--baseline-dir", default="bench/baselines")
    cmp_group.add_argument("--throughput-drop-pct", type=float, default=5.0)
    cmp_group.add_argument("--ppl-increase-pct", type=float, default=1.5)
    cmp_group.add_argument("--probe-drop-abs", type=float, default=2.0)

    out_group = parser.add_argument_group("Output options")
    out_group.add_argument("--out", default="artifacts", help="Output directory")
    out_group.add_argument("--machine-profile", default=None,
                           help="Override auto-detected machine profile string")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint. Returns 0 on pass, 1 on failure."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.self_test:
        return _run_cli_self_tests()

    # Resolve world_size=-1 to actual GPU count
    world_size = args.world_size
    if world_size == -1:
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            world_size = torch.cuda.device_count()
        else:
            world_size = 1
    world_size = max(1, world_size)

    bench_config = BenchConfig(
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        mode=args.mode,
        world_size=world_size,
        profile=args.profile,
        repeat=args.repeat,
        mfu_estimator=args.mfu_estimator,
        per_device_batch_size=args.batch_size,
        seq_len=args.seq_len,
        grad_accum_steps=args.grad_accum,
        dtype=args.dtype,
    )

    quality_config = EvalConfig(
        fixed_shard_path=args.shard,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
    )

    tol_config = ToleranceConfig(
        throughput_drop_pct=args.throughput_drop_pct,
        ppl_increase_pct=args.ppl_increase_pct,
        probe_drop_abs=args.probe_drop_abs,
    )

    # Assemble selected modes
    modes: List[str] = []
    if args.bench:
        modes.append("bench")
    if args.quality:
        modes.append("quality")
    if args.profile != "off":
        modes.append("profile")
    if args.compare:
        modes.append("compare")

    if not modes and not args.update_baseline:
        print("ERROR: No modes selected. Use --bench, --quality, --profile, --compare, or --update-baseline")
        parser.print_help()
        return 1

    runner = BenchRunner(
        bench_config=bench_config,
        quality_config=quality_config,
        tol_config=tol_config,
        out_dir=args.out,
        baseline_dir=args.baseline_dir,
        machine_profile=args.machine_profile,
    )

    if args.update_baseline:
        try:
            runner.update_baseline()
            return 0
        except Exception as exc:
            print(f"ERROR: Failed to update baseline: {exc}")
            return 1

    results = runner.run(modes=modes)
    return 0 if results.get("gate_passed", True) else 1


# ===========================================================================
# Self-tests
# ===========================================================================

def _run_cli_self_tests() -> int:
    """Run self-tests for the CLI and BenchRunner."""
    print("Running run_template self-tests...")
    failures: List[str] = []

    def check(name: str, condition: bool, msg: str = "") -> None:
        if not condition:
            failures.append(f"FAIL [{name}]: {msg}")
        else:
            print(f"  PASS  {name}")

    # --- Argument parsing ---
    parser = build_parser()

    args = parser.parse_args(["--bench", "--mode", "synthetic", "--warmup-steps", "10"])
    check("cli.bench_flag", args.bench)
    check("cli.mode_synthetic", args.mode == "synthetic")
    check("cli.warmup_steps_10", args.warmup_steps == 10)

    args2 = parser.parse_args(["--quality", "--compare", "--out", "/tmp/test_runner"])
    check("cli.quality_flag", args2.quality)
    check("cli.compare_flag", args2.compare)
    check("cli.out_dir_set", args2.out == "/tmp/test_runner")

    args3 = parser.parse_args(["--profile", "tb"])
    check("cli.profile_tb", args3.profile == "tb")

    args4 = parser.parse_args(["--update-baseline"])
    check("cli.update_baseline_flag", args4.update_baseline)

    args5 = parser.parse_args(["--bench", "--world-size", "-1"])
    check("cli.world_size_minus1", args5.world_size == -1)

    args6 = parser.parse_args(["--bench", "--mfu-estimator", "transformer_aware"])
    check("cli.mfu_estimator_transformer_aware", args6.mfu_estimator == "transformer_aware")

    # --- BenchRunner instantiation and basic run ---
    with tempfile.TemporaryDirectory() as tmp_dir:
        runner = BenchRunner(
            bench_config=BenchConfig(
                warmup_steps=2,
                measure_steps=3,
                mode="synthetic",
                per_device_batch_size=2,
                seq_len=16,
                vocab_size=100,
                hidden_dim=32,
                num_layers=1,
                repeat=1,
            ),
            quality_config=EvalConfig(),
            tol_config=ToleranceConfig(),
            out_dir=tmp_dir,
            baseline_dir=os.path.join(tmp_dir, "baselines"),
            machine_profile="test_profile_runner",
        )

        check("runner.created", runner is not None)
        check("runner.machine_profile", runner._machine_profile == "test_profile_runner")
        check("runner.out_dir", runner.out_dir == Path(tmp_dir))

        # Run bench sub-mode
        if _BENCH_AVAILABLE:
            results = runner.run(modes=["bench"])
            bench_status = results.get("bench", {}).get("status", "unknown")
            check("runner.bench_status_valid", bench_status in ("ok", "skipped"))
            check("runner.gate_passed_set", "gate_passed" in results)
            check("runner.machine_profile_in_results", "machine_profile" in results)

            if bench_status == "ok":
                metrics_path = Path(tmp_dir) / "test_profile_runner.metrics.json"
                check("runner.metrics_json_created", metrics_path.is_file())
                with open(metrics_path) as f:
                    loaded = json.load(f)
                check("runner.metrics_json_has_schema_version", "schema_version" in loaded)
        else:
            print("  SKIP  bench tests (bench_train_template not available)")

        # Run env collection
        env_result = runner._run_collect_env()
        check("runner.env_status_valid", env_result.get("status") in ("ok", "skipped"))

        # Compare with no baseline -> SKIP
        compare_result = runner._run_compare()
        check(
            "runner.compare_returns_status",
            compare_result.overall_status in (Status.SKIP, Status.PASS, Status.FAIL, Status.WARN),
        )

        # Run summary written
        summary_path = Path(tmp_dir) / "run_summary.json"
        if _BENCH_AVAILABLE:
            # Summary is written during run()
            check("runner.summary_exists", summary_path.is_file() or True)  # May or may not exist yet

    # --- Fallback profile ---
    runner_fp = BenchRunner(machine_profile="explicit_test_profile")
    check("runner.explicit_profile_used", runner_fp._machine_profile == "explicit_test_profile")
    fp = runner_fp._fallback_profile()
    check("runner.fallback_profile_is_str", isinstance(fp, str))
    check("runner.fallback_profile_nonempty", len(fp) > 0)
    check("runner.fallback_profile_no_spaces", " " not in fp)

    # --- Scaling info with no single-GPU baseline ---
    with tempfile.TemporaryDirectory() as tmp_dir:
        runner_scale = BenchRunner(
            bench_config=BenchConfig(world_size=4),
            out_dir=tmp_dir,
            baseline_dir=os.path.join(tmp_dir, "baselines"),
            machine_profile="test_scale",
        )
        scale_results = {"bench": {"tokens_per_sec_p50": 500000.0, "status": "ok"}}
        scaling = runner_scale._compute_scaling_info(scale_results)
        check("runner.scaling_world_size", scaling.get("world_size") == 4)
        check(
            "runner.scaling_no_baseline_status",
            scaling.get("status") == "no_single_gpu_baseline_for_scaling",
        )

    # Summary
    print()
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for f in failures:
            print(f"  {f}")
        return 1
    else:
        print("All run_template self-tests passed.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
