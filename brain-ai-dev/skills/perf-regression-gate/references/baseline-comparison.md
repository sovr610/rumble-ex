# Baseline Comparison Reference — Phase 4

## Overview

Phase 4 implements the comparison engine that loads a baseline snapshot (captured on main branch) and compares it to the current run, producing a structured pass/fail/warn report. The comparison is keyed by `machine_profile`—only results from equivalent hardware are compared. This prevents false failures when a PR is evaluated on a different GPU type than the baseline.

---

## Baseline Storage Format

### File Layout

```
bench/
  baselines/
    H100x8_driver550_cuda12.4_torch2.4_sm90.metrics.json
    H100x8_driver550_cuda12.4_torch2.4_sm90.eval.json
    A100x4_driver525_cuda12.1_torch2.3_sm80.metrics.json
    A100x4_driver525_cuda12.1_torch2.3_sm80.eval.json
```

The filename is the `machine_profile` string. Each machine profile has two files:
- `.metrics.json` — throughput, timing, memory, MFU
- `.eval.json` — perplexity, probe accuracies

### Why Per-Profile Baselines

Comparing an H100 run against an A100 baseline would almost always show a "regression" (H100 is faster) or false pass (A100 is slower). Machine-specific baselines eliminate hardware-induced noise. The `machine_profile` string encodes:

- GPU model and count (`H100x8`)
- Driver version (`driver550`)
- CUDA version (`cuda12.4`)
- PyTorch version (`torch2.4`)
- Compute capability (`sm90`)

All these factors affect timing, and any change in them should produce a new profile key and a fresh baseline.

### Baseline File Provenance

Baselines must be committed to the repository from `main` branch only. This guarantees:
1. Baselines represent the known-good state, not an in-progress branch
2. Any baseline regression must be reviewed as a PR
3. History of baseline evolution is tracked in git

In CI: fetch the baseline files from the `main` branch using `git show` or download from a CI artifact cache, then compare against the current run.

---

## Comparison Algorithm

### Loading and Matching

```python
import json
from pathlib import Path

def load_results(run_dir: str, machine_profile: str) -> tuple[dict, dict]:
    """Load metrics.json and eval.json for a given machine profile."""
    metrics_path = Path(run_dir) / f"{machine_profile}.metrics.json"
    eval_path = Path(run_dir) / f"{machine_profile}.eval.json"

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"No metrics file for profile '{machine_profile}' in {run_dir}.\n"
            f"Expected: {metrics_path}"
        )
    if not eval_path.exists():
        raise FileNotFoundError(
            f"No eval file for profile '{machine_profile}' in {run_dir}.\n"
            f"Expected: {eval_path}"
        )

    with open(metrics_path) as f:
        metrics = json.load(f)
    with open(eval_path) as f:
        eval_data = json.load(f)

    return metrics, eval_data
```

### Delta Computation

All percentage deltas use the baseline as the reference:

```python
def compute_delta_pct(current: float, baseline: float) -> float:
    """
    Returns signed percentage change relative to baseline.
    Positive = current is higher than baseline.
    """
    if baseline == 0:
        return float("inf") if current != 0 else 0.0
    return ((current - baseline) / abs(baseline)) * 100.0
```

For metrics where "higher is better" (tokens/sec, probe accuracy):
- Positive delta: improvement
- Negative delta beyond threshold: regression -> FAIL

For metrics where "lower is better" (step time, perplexity, memory):
- Negative delta: improvement
- Positive delta beyond threshold: regression -> FAIL

---

## Tolerance Rules

### Default Thresholds

| Gate | Metric | Direction | Threshold | Action |
|------|--------|-----------|-----------|--------|
| Perf | `tokens_per_sec_p50` | drop (current < baseline) | > 5% | FAIL |
| Perf | `step_time_p50` | increase (current > baseline) | > 5% | FAIL |
| Perf | `peak_allocated_bytes` | increase (current > baseline) | > 10% | WARN |
| Quality | `ppl_fixed_shard` | increase (current > baseline) | > 1.5% relative | FAIL |
| Quality | `probe_accuracy[*]` | drop (current < baseline) | > 2% absolute | FAIL |
| Stability | `loss_slope` | positive | > 0.001 | FAIL |

### `ToleranceConfig` Dataclass

```python
from dataclasses import dataclass

@dataclass
class ToleranceConfig:
    # Performance gate (fail on regression)
    throughput_drop_pct: float = 5.0       # tokens/sec p50 drop above this -> FAIL
    step_time_increase_pct: float = 5.0    # step time p50 increase above this -> FAIL
    # Memory gate (warn only)
    memory_increase_pct: float = 10.0      # peak memory increase above this -> WARN
    # Quality gate (fail on regression)
    ppl_increase_pct: float = 1.5          # relative perplexity increase -> FAIL
    probe_drop_abs: float = 2.0            # absolute probe accuracy drop (pp) -> FAIL
    # Stability gate
    loss_slope_threshold: float = 0.001    # positive slope above this -> FAIL
```

### Check Implementation

```python
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional

class Status(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    SKIP = "SKIP"

@dataclass
class CheckResult:
    metric: str
    baseline_val: float
    current_val: float
    delta_pct: float        # signed, relative to baseline
    threshold: float
    status: Status
    message: str

def check_throughput(current: dict, baseline: dict, tol: ToleranceConfig) -> List[CheckResult]:
    results = []

    # tokens_per_sec_p50: higher is better
    cur_tps = current["throughput"]["tokens_per_sec_p50"]
    bas_tps = baseline["throughput"]["tokens_per_sec_p50"]
    delta = compute_delta_pct(cur_tps, bas_tps)
    # Negative delta means current < baseline (regression for throughput)
    status = Status.FAIL if delta < -tol.throughput_drop_pct else Status.PASS
    results.append(CheckResult(
        metric="tokens_per_sec_p50",
        baseline_val=bas_tps,
        current_val=cur_tps,
        delta_pct=delta,
        threshold=-tol.throughput_drop_pct,
        status=status,
        message=f"tokens/s p50 {'dropped' if status == Status.FAIL else 'OK'}: {delta:+.2f}%",
    ))

    # step_time_p50: lower is better
    cur_st = current["timing"]["step_time_p50_s"]
    bas_st = baseline["timing"]["step_time_p50_s"]
    delta_st = compute_delta_pct(cur_st, bas_st)
    # Positive delta means current > baseline (regression for step time)
    status_st = Status.FAIL if delta_st > tol.step_time_increase_pct else Status.PASS
    results.append(CheckResult(
        metric="step_time_p50_s",
        baseline_val=bas_st,
        current_val=cur_st,
        delta_pct=delta_st,
        threshold=tol.step_time_increase_pct,
        status=status_st,
        message=f"step time p50 {'increased' if status_st == Status.FAIL else 'OK'}: {delta_st:+.2f}%",
    ))

    # peak memory: warn only
    cur_mem = current["memory"]["peak_allocated_bytes"]
    bas_mem = baseline["memory"]["peak_allocated_bytes"]
    delta_mem = compute_delta_pct(cur_mem, bas_mem)
    status_mem = Status.WARN if delta_mem > tol.memory_increase_pct else Status.PASS
    results.append(CheckResult(
        metric="peak_allocated_bytes",
        baseline_val=bas_mem,
        current_val=cur_mem,
        delta_pct=delta_mem,
        threshold=tol.memory_increase_pct,
        status=status_mem,
        message=f"peak memory {'WARN: increased' if status_mem == Status.WARN else 'OK'}: {delta_mem:+.2f}%",
    ))

    return results


def check_quality(current_eval: dict, baseline_eval: dict, tol: ToleranceConfig) -> List[CheckResult]:
    results = []

    # Perplexity: lower is better
    cur_ppl = current_eval["ppl_fixed_shard"]
    bas_ppl = baseline_eval["ppl_fixed_shard"]
    delta_ppl = compute_delta_pct(cur_ppl, bas_ppl)
    status_ppl = Status.FAIL if delta_ppl > tol.ppl_increase_pct else Status.PASS
    results.append(CheckResult(
        metric="ppl_fixed_shard",
        baseline_val=bas_ppl,
        current_val=cur_ppl,
        delta_pct=delta_ppl,
        threshold=tol.ppl_increase_pct,
        status=status_ppl,
        message=f"perplexity {'worsened' if status_ppl == Status.FAIL else 'OK'}: {delta_ppl:+.2f}%",
    ))

    # Probe accuracy: higher is better, absolute threshold
    cur_probes = current_eval["task_probe_accuracy"]
    bas_probes = baseline_eval["task_probe_accuracy"]
    for probe_name in bas_probes:
        if probe_name not in cur_probes:
            results.append(CheckResult(
                metric=f"probe/{probe_name}",
                baseline_val=bas_probes[probe_name],
                current_val=float("nan"),
                delta_pct=float("nan"),
                threshold=tol.probe_drop_abs,
                status=Status.WARN,
                message=f"probe {probe_name} missing from current run",
            ))
            continue
        cur_acc = cur_probes[probe_name]
        bas_acc = bas_probes[probe_name]
        # Absolute drop in percentage points (not relative %)
        abs_drop = (bas_acc - cur_acc) * 100.0
        status_probe = Status.FAIL if abs_drop > tol.probe_drop_abs else Status.PASS
        results.append(CheckResult(
            metric=f"probe/{probe_name}",
            baseline_val=bas_acc,
            current_val=cur_acc,
            delta_pct=compute_delta_pct(cur_acc, bas_acc),
            threshold=tol.probe_drop_abs,
            status=status_probe,
            message=(
                f"probe {probe_name} {'FAIL: dropped' if status_probe == Status.FAIL else 'OK'}: "
                f"{-abs_drop:+.1f}pp"
            ),
        ))

    return results


def check_stability(current: dict, tol: ToleranceConfig) -> List[CheckResult]:
    results = []
    loss_slope = current.get("loss", {}).get("loss_slope", None)
    if loss_slope is None:
        return results
    status = Status.FAIL if loss_slope > tol.loss_slope_threshold else Status.PASS
    results.append(CheckResult(
        metric="loss_slope",
        baseline_val=0.0,
        current_val=loss_slope,
        delta_pct=0.0,
        threshold=tol.loss_slope_threshold,
        status=status,
        message=(
            f"loss slope {'FAIL: positive trend' if status == Status.FAIL else 'OK'}: "
            f"{loss_slope:.6f} (threshold {tol.loss_slope_threshold})"
        ),
    ))
    return results
```

---

## `--update-baseline` Mode

### When to Use

Update the baseline when:
- A new feature intentionally improves performance (higher throughput baseline)
- Architecture changes intentionally use more memory (update memory baseline)
- Probe definitions are updated (new probe names = new baselines)
- After merging a PR that passes all gates

### Restrictions

Baseline updates must only run on `main` branch to prevent accidental baseline poisoning from feature branches.

```python
def update_baseline(current_dir: str, dest_dir: str, machine_profile: str) -> None:
    """
    Copy metrics.json and eval.json from current_dir to dest_dir (baselines dir).
    Should only be called after CI gate passes on main branch.
    """
    import shutil
    from pathlib import Path

    src_metrics = Path(current_dir) / f"{machine_profile}.metrics.json"
    src_eval = Path(current_dir) / f"{machine_profile}.eval.json"
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src_metrics, dest / src_metrics.name)
    shutil.copy2(src_eval, dest / src_eval.name)
    print(f"Baseline updated: {dest / src_metrics.name}")
    print(f"Baseline updated: {dest / src_eval.name}")
```

### CI Gate Pattern for Update

```yaml
# In CI workflow (only runs on main after successful gate):
- name: Update baseline
  if: github.ref == 'refs/heads/main' && success()
  run: |
    python -m tools.bench.run \
      --update-baseline \
      --current-dir artifacts/bench_results \
      --baseline-dir bench/baselines
    git add bench/baselines/
    git commit -m "ci: update perf baseline [skip ci]"
    git push
```

---

## Failure Reporting

### Human-Readable Table

```
=== Performance Gate Report ===
Machine Profile: H100x8_driver550_cuda12.4_torch2.4_sm90

Metric                      Baseline        Current         Delta       Status
------------------------------------------------------------------------------------
tokens_per_sec_p50          126500.0        119000.0        -5.93%      FAIL
step_time_p50_s             1.037           1.099           +5.98%      FAIL
peak_allocated_bytes        68.0 GB         74.8 GB         +10.0%      WARN
ppl_fixed_shard             12.34           12.53           +1.54%      PASS
probe/basic_reasoning_25    0.920           0.880           -4.35%      FAIL
probe/format_following_30   0.867           0.867           +0.00%      PASS
probe/code_sanity_20        0.900           0.900           +0.00%      PASS
loss_slope                  --              0.00023         --          PASS

Overall: FAIL (3 failures, 1 warning)

FAIL details:
  - tokens_per_sec_p50: dropped 5.93% (threshold: 5.0%). Baseline: 126500.0, Current: 119000.0
  - step_time_p50_s: increased 5.98% (threshold: 5.0%). Baseline: 1.037s, Current: 1.099s
  - probe/basic_reasoning_25: dropped 4.0pp (threshold: 2.0pp). Baseline: 0.920, Current: 0.880
```

### Failure Reporting Implementation

```python
def format_report(results: list[CheckResult], machine_profile: str) -> str:
    lines = [
        "=== Performance Gate Report ===",
        f"Machine Profile: {machine_profile}",
        "",
        f"{'Metric':<28} {'Baseline':>12} {'Current':>12} {'Delta':>10} {'Status':>8}",
        "-" * 84,
    ]

    for r in results:
        if r.status == Status.SKIP:
            continue
        bas_str = f"{r.baseline_val:.4g}" if not (r.baseline_val != r.baseline_val) else "--"
        cur_str = f"{r.current_val:.4g}" if not (r.current_val != r.current_val) else "--"
        delta_str = f"{r.delta_pct:+.2f}%" if not (r.delta_pct != r.delta_pct) else "--"
        lines.append(
            f"{r.metric:<28} {bas_str:>12} {cur_str:>12} {delta_str:>10} {r.status.value:>8}"
        )

    fails = [r for r in results if r.status == Status.FAIL]
    warns = [r for r in results if r.status == Status.WARN]
    overall = "FAIL" if fails else ("WARN" if warns else "PASS")

    lines.append("")
    lines.append(f"Overall: {overall} ({len(fails)} failures, {len(warns)} warnings)")

    if fails:
        lines.append("")
        lines.append("FAIL details:")
        for r in fails:
            lines.append(f"  - {r.message}")

    if warns:
        lines.append("")
        lines.append("WARN details:")
        for r in warns:
            lines.append(f"  - {r.message}")

    return "\n".join(lines)
```

---

## Handling Missing Baselines

### First-Run Behavior

When no baseline exists for the current machine profile, the comparison cannot run. The correct behavior depends on context:

**In CI on main branch:** Create the baseline automatically (first run creates it).
**In CI on a PR:** Skip comparison gates, emit a warning, allow the PR to pass.
**Developer local run:** Print a clear message explaining how to create the baseline.

```python
def compare(
    self,
    current_dir: str,
    baseline_dir: str,
    machine_profile: str,
) -> CompareResult:
    try:
        baseline_metrics, baseline_eval = load_results(baseline_dir, machine_profile)
    except FileNotFoundError:
        print(
            f"WARNING: No baseline found for profile '{machine_profile}'.\n"
            f"This appears to be the first run on this hardware.\n"
            f"Run with --update-baseline on main branch to create the baseline."
        )
        return CompareResult(
            overall_status=Status.SKIP,
            checks=[],
            report="No baseline available; skipping comparison.",
            machine_profile=machine_profile,
        )

    current_metrics, current_eval = load_results(current_dir, machine_profile)
    # ... proceed with comparison
```

### Baseline Version Compatibility

If the baseline schema version differs from the current schema version, comparison may fail on missing fields. Handle gracefully:

```python
def is_compatible(baseline: dict, current: dict) -> bool:
    return baseline.get("schema_version") == current.get("schema_version")
```

If incompatible, skip comparison and emit a warning to update the baseline.

---

## Tolerance Tuning

### Making Gates Stricter

Reduce thresholds to catch smaller regressions. Trade-off: more false positives from hardware noise.

```python
strict_tol = ToleranceConfig(
    throughput_drop_pct=2.0,    # Was 5.0
    step_time_increase_pct=2.0, # Was 5.0
    memory_increase_pct=5.0,    # Was 10.0
    ppl_increase_pct=0.5,       # Was 1.5
    probe_drop_abs=1.0,         # Was 2.0
)
```

### Making Gates More Lenient

Increase thresholds to reduce false positives. Trade-off: may miss real regressions.

```python
lenient_tol = ToleranceConfig(
    throughput_drop_pct=10.0,
    step_time_increase_pct=10.0,
    memory_increase_pct=20.0,
    ppl_increase_pct=3.0,
    probe_drop_abs=5.0,
)
```

### Typical Noise Levels

| Metric | Typical run-to-run noise | Recommended gate threshold |
|--------|--------------------------|---------------------------|
| tokens_per_sec_p50 | 1-3% | 5% |
| step_time_p50 | 1-3% | 5% |
| peak memory | < 1% | 10% (generous; memory is stable) |
| perplexity | < 0.1% | 1.5% |
| probe accuracy (25 samples) | 0-4% | 2pp absolute |

---

## CI Workflow Integration

### Fetching Baselines from Main

```bash
# Fetch baselines from main branch without checking out
git fetch origin main
git show origin/main:bench/baselines/${MACHINE_PROFILE}.metrics.json \
    > /tmp/baseline.metrics.json
git show origin/main:bench/baselines/${MACHINE_PROFILE}.eval.json \
    > /tmp/baseline.eval.json
```

Or using `gh` CLI:
```bash
gh api repos/{owner}/{repo}/contents/bench/baselines/${MACHINE_PROFILE}.metrics.json \
    --jq '.content' | base64 -d > /tmp/baseline.metrics.json
```

### Artifact Upload

After comparison, upload both current results and the report as CI artifacts:

```yaml
- name: Upload perf artifacts
  uses: actions/upload-artifact@v4
  with:
    name: perf-gate-results
    path: |
      artifacts/bench_results/*.json
      artifacts/gate_report.txt
    retention-days: 30
```

---

## Implementation Checklist

- [ ] Baseline files named by `machine_profile` (not branch name or timestamp)
- [ ] `load_results()` raises `FileNotFoundError` with clear message for missing baselines
- [ ] Delta computation uses baseline as denominator (not current)
- [ ] Memory gate is WARN, not FAIL (memory increase is less critical than throughput)
- [ ] Probe accuracy uses absolute pp drop, not relative %
- [ ] `loss_slope` computed with linear regression over measure_steps loss values
- [ ] `update_baseline` checks git branch before allowing update
- [ ] Report always prints metric name, baseline value, current value, and delta
- [ ] `CompareResult.overall_status` is `FAIL` if any check is `FAIL`
- [ ] Schema version compatibility check before comparison
