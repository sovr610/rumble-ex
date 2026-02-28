"""
ci_workflow_template.py — Generate CI artefacts for the performance regression gate.

Outputs:
  - .github/workflows/perf_gate.yml   GitHub Actions workflow
  - docs/perf.md                      User-facing guide (local runs, baselines,
                                      result interpretation)

Usage (standalone):
    python ci_workflow_template.py --out-dir .
    python ci_workflow_template.py --out-dir . --runner "self-hosted,gpu,H100"
    python ci_workflow_template.py --self-test

Usage (imported):
    from ci_workflow_template import CIWorkflowGenerator
    gen = CIWorkflowGenerator(runner_labels=["self-hosted", "gpu", "H100"])
    yaml_text = gen.generate_workflow_yaml()
    docs_text = gen.generate_docs_md()
"""

from __future__ import annotations

import argparse
import os
import pathlib
import textwrap
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class WorkflowConfig:
    """Configuration for workflow and docs generation."""

    # GitHub Actions runner labels (space/comma separated -> list)
    runner_labels: List[str] = field(default_factory=lambda: ["self-hosted", "gpu"])

    # Python version used in CI
    python_version: str = "3.11"

    # pip install extras beyond requirements.txt
    extra_pip: List[str] = field(default_factory=list)

    # Branch whose baselines are fetched for comparison
    baseline_branch: str = "main"

    # Artifact retention days
    artifact_days: int = 30

    # Benchmark flags passed to run_template.py
    bench_flags: str = "--bench --quality --compare"

    # Whether to add --profile step (slower, optional)
    include_profile_step: bool = False

    # Slack / Teams webhook secret name (empty = skip notification)
    notify_secret: str = ""

    # concurrency group to cancel in-flight runs on new push
    cancel_in_progress: bool = True

    def runner_string(self) -> str:
        """Render runner labels as YAML list or single string."""
        if len(self.runner_labels) == 1:
            return self.runner_labels[0]
        # Multi-label: format as YAML inline list
        items = ", ".join(f'"{lbl}"' for lbl in self.runner_labels)
        return f"[{items}]"


# ---------------------------------------------------------------------------
# YAML builder helpers
# ---------------------------------------------------------------------------

def _indent(text: str, spaces: int) -> str:
    """Indent every line of *text* by *spaces* spaces."""
    prefix = " " * spaces
    return "\n".join(prefix + line if line.strip() else line for line in text.splitlines())


def _step(name: str, body: str, indent: int = 6) -> str:
    """Format a single GitHub Actions step block."""
    return _indent(f"- name: {name}\n{body}", indent)


def _run_step(name: str, script: str, indent: int = 6, env: Optional[dict] = None) -> str:
    """Format a 'run:' step with optional env dict."""
    env_block = ""
    if env:
        env_lines = "\n".join(f"    {k}: {v}" for k, v in env.items())
        env_block = f"\n  env:\n{env_lines}"
    body = f"  run: |\n{_indent(textwrap.dedent(script).strip(), 4)}{env_block}"
    return _step(name, body, indent)


def _uses_step(name: str, uses: str, with_block: Optional[dict] = None, indent: int = 6) -> str:
    """Format a 'uses:' step with optional 'with:' dict."""
    with_str = ""
    if with_block:
        with_lines = "\n".join(f"    {k}: {v}" for k, v in with_block.items())
        with_str = f"\n  with:\n{with_lines}"
    body = f"  uses: {uses}{with_str}"
    return _step(name, body, indent)


# ---------------------------------------------------------------------------
# CIWorkflowGenerator
# ---------------------------------------------------------------------------

class CIWorkflowGenerator:
    """Generate GitHub Actions YAML and documentation for the perf regression gate."""

    def __init__(self, config: Optional[WorkflowConfig] = None, **kwargs):
        if config is None:
            config = WorkflowConfig(**{k: v for k, v in kwargs.items()
                                       if k in WorkflowConfig.__dataclass_fields__})
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_workflow_yaml(self) -> str:
        """Return complete .github/workflows/perf_gate.yml content."""
        cfg = self.config
        sections = [
            self._header(),
            self._on_trigger(),
            self._concurrency(cfg),
            "jobs:",
            self._perf_gate_job(cfg),
        ]
        if cfg.notify_secret:
            sections.append(self._notify_job(cfg))
        return "\n".join(sections) + "\n"

    def generate_docs_md(self) -> str:
        """Return complete docs/perf.md content."""
        cfg = self.config
        return textwrap.dedent(f"""\
        # Performance Regression Gate

        This document explains how to run the performance benchmark locally,
        update baselines, and interpret results reported in CI.

        ---

        ## Quick Start

        ```bash
        # Install dependencies
        pip install -r requirements.txt

        # Collect environment info + run benchmark + compare
        python assets/run_template.py --bench --quality --compare

        # See full options
        python assets/run_template.py --help
        ```

        ---

        ## Running Locally

        ### Benchmark only (throughput, tokens/sec, MFU)

        ```bash
        python assets/run_template.py --bench
        ```

        Results written to `bench/runs/<machine_profile>/metrics.json`.

        ### Quality harness only (perplexity, task probes)

        ```bash
        python assets/run_template.py --quality
        ```

        Results written to `bench/runs/<machine_profile>/quality.json`.

        ### Full run with comparison

        ```bash
        python assets/run_template.py --bench --quality --compare
        ```

        Exits with code `0` (PASS) or `1` (FAIL / regression detected).

        ### Profiling (optional, adds ~2 min)

        ```bash
        python assets/run_template.py --bench --profile
        ```

        Traces written to `bench/profiles/<machine_profile>/`.
        View with TensorBoard:

        ```bash
        tensorboard --logdir bench/profiles/<machine_profile>/tb/
        ```

        ---

        ## Updating Baselines

        When you intentionally change model architecture or training recipe and the
        new numbers should become the new reference:

        ```bash
        # 1. Run the full benchmark to generate fresh metrics
        python assets/run_template.py --bench --quality

        # 2. Promote results to baselines
        python assets/run_template.py --update-baseline

        # 3. Commit the updated baseline files
        git add bench/baselines/
        git commit -m "perf: update baselines after <change description>"
        git push origin {cfg.baseline_branch}
        ```

        **Important:** Only update baselines on the `{cfg.baseline_branch}` branch.
        PR branches always compare against `{cfg.baseline_branch}` baselines.

        ---

        ## Machine Profiles

        The machine profile string uniquely identifies a hardware + software stack:

        ```
        <GPU>x<count>_driver<N>_cuda<M>_torch<V>_sm<C>
        ```

        Examples:
        - `H100x8_driver550_cuda12.4_torch2.4_sm90`
        - `A100x4_driver525_cuda12.1_torch2.3_sm80`
        - `RTX4090x1_driver535_cuda12.2_torch2.4_sm89`

        Baselines are stored per machine profile so comparisons are always
        apples-to-apples.

        ---

        ## Interpreting Results

        ### Failure Thresholds

        | Metric               | Direction | Threshold | Severity |
        |----------------------|-----------|-----------|----------|
        | Tokens/sec (p50)     | Drop >5%  | FAIL      | Blocks PR |
        | Step time (p50)      | Rise >5%  | FAIL      | Blocks PR |
        | Peak memory          | Rise >10% | WARN      | Advisory  |
        | Perplexity           | Rise >1.5%| FAIL      | Blocks PR |
        | Probe accuracy       | Drop >2pp | FAIL      | Blocks PR |
        | Loss slope           | Positive  | FAIL      | Blocks PR |

        ### Reading the CI Report

        ```
        ┌─────────────────────┬──────────────┬──────────────┬─────────┬────────┐
        │ Metric              │ Baseline     │ Current      │ Delta   │ Status │
        ├─────────────────────┼──────────────┼──────────────┼─────────┼────────┤
        │ tokens_per_sec_p50  │ 48 200       │ 47 100       │ -2.3%   │ PASS   │
        │ step_time_p50_s     │ 0.421        │ 0.430        │ +2.1%   │ PASS   │
        │ peak_memory_gb      │ 38.2         │ 42.1         │ +10.2%  │ WARN   │
        │ perplexity          │ 14.72        │ 14.85        │ +0.9%   │ PASS   │
        │ basic_reasoning     │ 0.84         │ 0.83         │ -1.2pp  │ PASS   │
        └─────────────────────┴──────────────┴──────────────┴─────────┴────────┘
        Overall: PASS (1 WARN)
        ```

        ### Common Failures and Fixes

        **tokens_per_sec regression (>5%)**
        - Check for new synchronization points in the forward/backward pass.
        - Verify batch size / sequence length were not accidentally reduced.
        - Check if a new dependency introduced a CPU bottleneck.

        **Perplexity regression (>1.5%)**
        - Confirm tokenizer shard SHA256 matches expected.
        - Check for changes to model initialization (weight init, dtype).
        - Verify no data preprocessing changes affected the fixed shard.

        **Positive loss slope (FAIL)**
        - Indicates the model is not converging during the benchmark run.
        - Check learning rate schedule and optimizer state initialization.

        **SKIP (no baseline)**
        - First run on this machine profile — results are saved as the new baseline.
        - Re-run to get a comparison on the next CI invocation.

        ---

        ## CI Integration

        The workflow runs automatically on every pull request targeting `{cfg.baseline_branch}`.
        Artifacts (metrics JSON, profile traces) are uploaded and retained for
        {cfg.artifact_days} days.

        ### Workflow File

        See `.github/workflows/perf_gate.yml`.

        ### Manual Trigger

        ```bash
        gh workflow run perf_gate.yml --ref <branch>
        ```

        ### Skipping the Gate

        Add the label `skip-perf-gate` to a PR to bypass the gate.
        Use sparingly — only for documentation-only or config-only changes.

        ---

        ## Architecture Notes

        ```
        assets/
          run_template.py            Orchestrator (entry point)
          bench_train_template.py    Throughput benchmark (CUDA sync timing)
          eval_small_template.py     Quality harness (PPL + task probes)
          compare_baseline_template.py  Delta computation + pass/fail logic
          collect_env_template.py    Machine profile + env snapshot
          perf_gate_config_template.py  Config dataclasses + GPU registry

        bench/
          baselines/<profile>/       Reference numbers (committed to {cfg.baseline_branch})
          runs/<profile>/            Latest run results (gitignored)
          profiles/<profile>/        PyTorch profiler traces (gitignored)

        scripts/
          validate_perf_gate.py      Done-when gate validator
          gen_perf_tests.py          Pytest test generator
          ci_workflow_template.py    This script
        ```

        ---

        ## FAQ

        **Q: Why not use pytest-benchmark?**
        A: We need CUDA-synchronised wall-clock timing, MFU computation against a
        GPU-specific peak TFLOPS registry, and integration with the quality harness.
        A custom harness gives us full control over all three.

        **Q: Why fixed-shard perplexity instead of a held-out dataset split?**
        A: Reproducibility. The shard SHA256 is verified before scoring, so any
        tokenizer or data-pipeline change that would silently shift perplexity is
        caught immediately.

        **Q: How do I add a new task probe?**
        A: Add an entry to `PROBE_REGISTRY` in `eval_small_template.py`.
        Each probe is a dict with `prompt`, `expected`, and `match` keys.
        Re-generate and commit an updated baseline after adding probes.
        """)

    def write_all(self, out_dir: str) -> dict:
        """
        Write all generated files under *out_dir*.

        Returns a dict mapping relative path -> absolute path for each file written.
        """
        out = pathlib.Path(out_dir)
        written = {}

        # .github/workflows/perf_gate.yml
        wf_path = out / ".github" / "workflows" / "perf_gate.yml"
        wf_path.parent.mkdir(parents=True, exist_ok=True)
        wf_path.write_text(self.generate_workflow_yaml(), encoding="utf-8")
        written[".github/workflows/perf_gate.yml"] = str(wf_path.resolve())

        # docs/perf.md
        docs_path = out / "docs" / "perf.md"
        docs_path.parent.mkdir(parents=True, exist_ok=True)
        docs_path.write_text(self.generate_docs_md(), encoding="utf-8")
        written["docs/perf.md"] = str(docs_path.resolve())

        return written

    # ------------------------------------------------------------------
    # Private: YAML sections
    # ------------------------------------------------------------------

    def _header(self) -> str:
        return textwrap.dedent("""\
        # .github/workflows/perf_gate.yml
        # Auto-generated by scripts/ci_workflow_template.py — do not edit manually.
        # Re-generate with: python scripts/ci_workflow_template.py --out-dir .
        name: Perf Regression Gate
        """)

    def _on_trigger(self) -> str:
        return textwrap.dedent("""\
        on:
          pull_request:
            branches:
              - main
            types: [opened, synchronize, reopened]
          workflow_dispatch:
            inputs:
              update_baseline:
                description: "Promote current results to baselines"
                required: false
                default: "false"
                type: choice
                options: ["true", "false"]
        """)

    def _concurrency(self, cfg: WorkflowConfig) -> str:
        cancel = "true" if cfg.cancel_in_progress else "false"
        return textwrap.dedent(f"""\
        concurrency:
          group: perf-gate-${{{{ github.head_ref || github.ref }}}}
          cancel-in-progress: {cancel}
        """)

    def _perf_gate_job(self, cfg: WorkflowConfig) -> str:
        runner = cfg.runner_string()
        steps = self._build_steps(cfg)
        job_body = textwrap.dedent(f"""\
          perf-gate:
            name: "Compute / Throughput Regression Gate"
            runs-on: {runner}
            if: "!contains(github.event.pull_request.labels.*.name, 'skip-perf-gate')"
            timeout-minutes: 60
            steps:
        """)
        job_body += steps
        return _indent(job_body, 2)

    def _build_steps(self, cfg: WorkflowConfig) -> str:
        """Assemble all steps as an indented YAML block."""
        steps = []

        # 1. Checkout PR branch
        steps.append(_uses_step(
            "Checkout PR branch",
            "actions/checkout@v4",
            with_block={"fetch-depth": "0"},
        ))

        # 2. Set up Python
        steps.append(_uses_step(
            "Set up Python",
            "actions/setup-python@v5",
            with_block={"python-version": cfg.python_version},
        ))

        # 3. Cache pip
        steps.append(_uses_step(
            "Cache pip",
            "actions/cache@v4",
            with_block={
                "path": "~/.cache/pip",
                "key": (
                    "${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}"
                ),
                "restore-keys": "${{ runner.os }}-pip-",
            },
        ))

        # 4. Install dependencies
        extra = ""
        if cfg.extra_pip:
            pkgs = " ".join(cfg.extra_pip)
            extra = f"\npip install {pkgs}"
        steps.append(_run_step(
            "Install dependencies",
            f"pip install --upgrade pip\npip install -r requirements.txt{extra}",
        ))

        # 5. Collect environment snapshot
        steps.append(_run_step(
            "Collect environment snapshot",
            "python assets/collect_env_template.py --out bench/runs/env.json",
        ))

        # 6. Run benchmark + quality harness
        bench_flags = cfg.bench_flags
        steps.append(_run_step(
            "Run benchmark and quality harness",
            f"python assets/run_template.py {bench_flags}",
        ))

        # 7. Optional profiling
        if cfg.include_profile_step:
            steps.append(_run_step(
                "Profile forward pass (optional)",
                "python assets/run_template.py --bench --profile",
            ))

        # 8. Fetch baselines from main branch
        steps.append(_run_step(
            f"Fetch baselines from {cfg.baseline_branch}",
            textwrap.dedent(f"""\
            git fetch origin {cfg.baseline_branch}:refs/remotes/origin/{cfg.baseline_branch}
            git show origin/{cfg.baseline_branch}:bench/baselines > /dev/null 2>&1 || true
            git checkout origin/{cfg.baseline_branch} -- bench/baselines/ || true
            """),
        ))

        # 9. Compare against baselines
        steps.append(_run_step(
            "Compare against baselines",
            "python assets/run_template.py --compare",
            env={"PERF_GATE_STRICT": "1"},
        ))

        # 10. Update baselines (manual dispatch only)
        steps.append(_run_step(
            "Update baselines (dispatch only)",
            textwrap.dedent("""\
            if [ "${{ github.event_name }}" = "workflow_dispatch" ] && \
               [ "${{ github.event.inputs.update_baseline }}" = "true" ]; then
              python assets/run_template.py --update-baseline
              git config user.name "github-actions[bot]"
              git config user.email "github-actions[bot]@users.noreply.github.com"
              git add bench/baselines/
              git diff --cached --quiet || \
                git commit -m "perf: update baselines [skip ci]"
              git push origin HEAD:main
            fi
            """),
        ))

        # 11. Upload artifacts
        profile_path = "bench/profiles/**" if cfg.include_profile_step else ""
        artifact_paths = "bench/runs/**\nbench/baselines/**"
        if profile_path:
            artifact_paths += f"\n{profile_path}"

        steps.append(_uses_step(
            "Upload benchmark artifacts",
            "actions/upload-artifact@v4",
            with_block={
                "name": (
                    "perf-gate-${{ github.run_id }}-${{ github.run_attempt }}"
                ),
                "path": artifact_paths,
                "retention-days": str(cfg.artifact_days),
                "if-no-files-found": "warn",
            },
        ))

        # 12. Post status comment on PR
        steps.append(_run_step(
            "Post gate result comment",
            textwrap.dedent("""\
            if [ -f bench/runs/report.txt ]; then
              BODY=$(cat bench/runs/report.txt)
            else
              BODY="No report generated."
            fi
            gh pr comment ${{ github.event.pull_request.number }} \
              --body "## Perf Gate Report\\n\\n\`\`\`\\n${BODY}\\n\`\`\`" \
              --edit-last || \
            gh pr comment ${{ github.event.pull_request.number }} \
              --body "## Perf Gate Report\\n\\n\`\`\`\\n${BODY}\\n\`\`\`"
            """),
            env={"GH_TOKEN": "${{ secrets.GITHUB_TOKEN }}"},
        ))

        return "\n".join(steps)

    def _notify_job(self, cfg: WorkflowConfig) -> str:
        """Optional Slack/Teams notification job that runs after perf-gate."""
        secret = cfg.notify_secret
        job_body = textwrap.dedent(f"""\
          notify:
            name: "Notify on Regression"
            needs: perf-gate
            if: failure()
            runs-on: ubuntu-latest
            steps:
        """)
        notify_step = _run_step(
            "Send failure notification",
            textwrap.dedent(f"""\
            curl -s -X POST \\
              -H 'Content-type: application/json' \\
              --data '{{"text":"Perf regression gate FAILED on ${{{{ github.repository }}}} \
            PR #${{{{ github.event.pull_request.number }}}}."}}' \\
              "${{{{ secrets.{secret} }}}}"
            """),
        )
        job_body += notify_step
        return _indent(job_body, 2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate .github/workflows/perf_gate.yml and docs/perf.md",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--out-dir",
        default=".",
        help="Root directory to write files into (creates subdirs as needed)",
    )
    p.add_argument(
        "--runner",
        default="self-hosted,gpu",
        help="Comma-separated runner labels (e.g. 'self-hosted,gpu,H100')",
    )
    p.add_argument(
        "--python",
        default="3.11",
        dest="python_version",
        help="Python version for the CI environment",
    )
    p.add_argument(
        "--baseline-branch",
        default="main",
        help="Branch from which baselines are fetched",
    )
    p.add_argument(
        "--artifact-days",
        type=int,
        default=30,
        help="Artifact retention days",
    )
    p.add_argument(
        "--include-profile",
        action="store_true",
        help="Include a profiling step in the workflow",
    )
    p.add_argument(
        "--notify-secret",
        default="",
        help="GitHub secret name holding Slack/Teams webhook URL",
    )
    p.add_argument(
        "--self-test",
        action="store_true",
        help="Run self-tests and exit (requires PyYAML)",
    )
    p.add_argument(
        "--print-yaml",
        action="store_true",
        help="Print generated YAML to stdout instead of writing files",
    )
    p.add_argument(
        "--print-docs",
        action="store_true",
        help="Print generated docs/perf.md to stdout instead of writing files",
    )
    return p


def _run_self_tests() -> None:
    """Self-contained tests — no pytest required."""
    import tempfile

    print("Running self-tests for ci_workflow_template.py ...")

    # Test 1: Default config generation produces non-empty output
    gen = CIWorkflowGenerator()
    yaml_text = gen.generate_workflow_yaml()
    assert len(yaml_text) > 500, "YAML output is unexpectedly short"
    assert "perf-gate:" in yaml_text, "Job 'perf-gate' missing from YAML"
    print("  [PASS] Test 1: Default YAML has 'perf-gate' job")

    # Test 2: Structural YAML validation via PyYAML (optional)
    try:
        import yaml as _yaml
        parsed = _yaml.safe_load(yaml_text)
        assert "jobs" in parsed, "Missing 'jobs' key in parsed YAML"
        assert "perf-gate" in parsed["jobs"], "Missing 'perf-gate' job in parsed YAML"
        steps = parsed["jobs"]["perf-gate"]["steps"]
        assert isinstance(steps, list), "Steps must be a list"
        assert len(steps) >= 8, f"Expected at least 8 steps, got {len(steps)}"
        print(f"  [PASS] Test 2: YAML parses cleanly ({len(steps)} steps found)")
    except ImportError:
        print("  [SKIP] Test 2: PyYAML not installed — skipping structural parse")

    # Test 3: Trigger section contains pull_request
    assert "pull_request:" in yaml_text, "Missing pull_request trigger"
    print("  [PASS] Test 3: pull_request trigger present")

    # Test 4: Runner labels respected
    gen_custom = CIWorkflowGenerator(runner_labels=["self-hosted", "gpu", "H100-SXM"])
    yaml_custom = gen_custom.generate_workflow_yaml()
    assert "H100-SXM" in yaml_custom, "Custom runner label not found in YAML"
    print("  [PASS] Test 4: Custom runner labels are embedded correctly")

    # Test 5: Profiling step conditionally included
    gen_no_prof = CIWorkflowGenerator(include_profile_step=False)
    gen_with_prof = CIWorkflowGenerator(include_profile_step=True)
    yaml_no_prof = gen_no_prof.generate_workflow_yaml()
    yaml_with_prof = gen_with_prof.generate_workflow_yaml()
    assert "--profile" not in yaml_no_prof, "Profile step leaked into no-profile workflow"
    assert "--profile" in yaml_with_prof, "Profile step missing from profile-enabled workflow"
    print("  [PASS] Test 5: Profile step correctly gated by include_profile_step flag")

    # Test 6: Docs generation — key sections present
    docs_text = gen.generate_docs_md()
    required_sections = [
        "## Quick Start",
        "## Running Locally",
        "## Updating Baselines",
        "## Machine Profiles",
        "## Interpreting Results",
        "## CI Integration",
        "## Architecture Notes",
        "## FAQ",
    ]
    for section in required_sections:
        assert section in docs_text, f"Missing docs section: {section}"
    print(f"  [PASS] Test 6: docs/perf.md contains all {len(required_sections)} required sections")

    # Test 7: Failure thresholds table present in docs
    assert ">5%" in docs_text, "Throughput threshold missing from docs"
    assert ">1.5%" in docs_text, "Perplexity threshold missing from docs"
    assert ">2pp" in docs_text, "Probe accuracy threshold missing from docs"
    print("  [PASS] Test 7: Failure threshold table present in docs")

    # Test 8: write_all creates both files in a temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        gen_write = CIWorkflowGenerator()
        written = gen_write.write_all(tmpdir)
        assert len(written) == 2, f"Expected 2 written files, got {len(written)}"
        for rel, abs_path in written.items():
            assert os.path.isfile(abs_path), f"File not created: {abs_path}"
            size = os.path.getsize(abs_path)
            assert size > 200, f"File is suspiciously small ({size} bytes): {rel}"
        print(f"  [PASS] Test 8: write_all created both files successfully")

    # Test 9: Notification job included when secret provided
    gen_notify = CIWorkflowGenerator(notify_secret="SLACK_WEBHOOK")
    yaml_notify = gen_notify.generate_workflow_yaml()
    assert "notify:" in yaml_notify, "Notification job missing when secret is set"
    assert "SLACK_WEBHOOK" in yaml_notify, "Secret name not embedded in notification job"
    print("  [PASS] Test 9: Notification job conditionally included")

    # Test 10: cancel-in-progress respected
    gen_no_cancel = CIWorkflowGenerator(cancel_in_progress=False)
    yaml_no_cancel = gen_no_cancel.generate_workflow_yaml()
    assert "cancel-in-progress: false" in yaml_no_cancel, \
        "cancel-in-progress flag not respected"
    print("  [PASS] Test 10: cancel-in-progress flag respected")

    print("\nAll self-tests PASSED.")


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.self_test:
        _run_self_tests()
        return 0

    runner_labels = [lbl.strip() for lbl in args.runner.split(",") if lbl.strip()]
    cfg = WorkflowConfig(
        runner_labels=runner_labels,
        python_version=args.python_version,
        baseline_branch=args.baseline_branch,
        artifact_days=args.artifact_days,
        include_profile_step=args.include_profile,
        notify_secret=args.notify_secret,
    )
    gen = CIWorkflowGenerator(config=cfg)

    if args.print_yaml:
        print(gen.generate_workflow_yaml())
        return 0

    if args.print_docs:
        print(gen.generate_docs_md())
        return 0

    written = gen.write_all(args.out_dir)
    for rel, abs_path in written.items():
        size = os.path.getsize(abs_path)
        print(f"  Wrote {rel}  ({size} bytes)  ->  {abs_path}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
