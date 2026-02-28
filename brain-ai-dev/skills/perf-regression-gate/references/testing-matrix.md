# Testing Matrix Reference — Phase 6

## Overview

This document catalogues all test scenarios for the perf-regression-gate skill, organized by phase. Each scenario specifies what is tested, the method, expected outcome, and rationale. Use this as the definitive guide when implementing test suites with `gen_perf_tests.py`.

---

## Phase 1: `collect_env` Tests

### ENV-01: All Required Keys Present

**What:** `CollectEnv.collect()` returns a dict containing all required top-level keys.

**Method:** Instantiate `CollectEnv`, call `collect()`, assert each key exists.

**Required keys:** `git`, `python_version`, `torch`, `cuda`, `gpus`, `env_vars`

**Expected:** All keys present regardless of whether GPU is available.

**Rationale:** Downstream consumers rely on these keys existing. Missing keys cause `KeyError` in comparison logic.

---

### ENV-02: Git Section Schema

**What:** The `git` sub-dict has `sha` (string) and `dirty` (bool) fields.

**Method:** Assert `isinstance(env["git"]["sha"], str)` and `isinstance(env["git"]["dirty"], bool)`.

**Edge case:** Git not installed or not in a git repo — `sha` should be `"unknown"`, not raise.

**Expected:** `sha` is a 40-char hex string or `"unknown"`, `dirty` is True/False.

---

### ENV-03: Torch Section Schema

**What:** `torch` sub-dict has `version` (str), `cuda_version` (str), and `build_flags` (list).

**Method:** Check types and non-empty values.

**Expected:** Version like `"2.4.0"`, cuda_version like `"12.4"`, build_flags is a list (may be empty).

---

### ENV-04: GPUs Section Schema

**What:** `gpus` is a list; each entry has `name` (str), `mem_total` (int, bytes), `compute_cap` (str).

**Method:** If CUDA is not available, `gpus` should be an empty list (`[]`), not raise.

**Expected:** `[{"name": "NVIDIA H100 SXM5", "mem_total": 85899345920, "compute_cap": "9.0"}, ...]`

---

### ENV-05: Machine Profile Determinism

**What:** Calling `machine_profile()` twice returns identical strings.

**Method:**
```python
env = CollectEnv()
p1 = env.machine_profile()
p2 = env.machine_profile()
assert p1 == p2
```

**Expected:** Identical strings. The profile is derived from static hardware info and should not vary.

**Rationale:** Profile is used as a dict key and filename. Non-determinism causes baseline mismatches.

---

### ENV-06: Machine Profile Format

**What:** Profile string matches pattern `<GPU><count>_driver<ver>_cuda<ver>_torch<ver>_sm<cap>`.

**Method:** `re.match(r"^[A-Za-z0-9]+x\d+_driver\d+_cuda[\d.]+_torch[\d.]+_sm\d+$", profile)`

**Expected:** Matches pattern. For CPU-only machines: `cpu_torch<ver>` format.

---

### ENV-07: env.json Roundtrip

**What:** `save(path)` writes valid JSON that can be reloaded and compared equal.

**Method:**
```python
env = CollectEnv()
original = env.collect()
env.save("/tmp/test_env.json")
with open("/tmp/test_env.json") as f:
    loaded = json.load(f)
assert loaded == original
```

**Expected:** JSON roundtrip is lossless for all field types.

---

### ENV-08: Missing GPU Handling

**What:** On a machine without CUDA, `collect()` returns `gpus: []` and profile contains `cpu`.

**Method:** Mock `torch.cuda.is_available()` to return False, call collect.

**Expected:** No exception, `gpus` is `[]`, profile string contains `"cpu"`.

---

### ENV-09: Env Vars Captured

**What:** Known environment variables are captured in `env_vars` section.

**Method:** Set `CUDA_VISIBLE_DEVICES=0,1` in env, call collect, verify it appears in output.

**Expected:** `env["env_vars"]["CUDA_VISIBLE_DEVICES"] == "0,1"`.

---

### ENV-10: env.json Schema Version

**What:** Saved env.json contains `schema_version` field.

**Method:** Load env.json after save, assert `"schema_version"` key exists with string value.

**Expected:** `{"schema_version": "1.0", ...}` or similar.

---

## Phase 2: `bench_train` Tests

### BENCH-01: CUDA Sync Timing Correctness

**What:** Step times measured with CUDA sync are > step times without sync (for GPU workloads) or equal (for CPU).

**Method:** Time a GPU matrix multiply with and without synchronization; verify sync version is larger.

**Expected:** Sync time > async time for any meaningful GPU operation.

**Rationale:** Core correctness test—the entire benchmark purpose depends on this.

---

### BENCH-02: Step Times Are Positive

**What:** All measured step times are positive floats.

**Method:** Run `BenchTrain` with synthetic mode, check all step times > 0.

**Expected:** `all(t > 0 for t in step_times)`.

---

### BENCH-03: Tokens Per Second > 0

**What:** Computed tokens/second is a positive float.

**Method:** Run benchmark, assert `result.tokens_per_sec_p50 > 0`.

**Expected:** Positive value. For any non-trivial model on any hardware.

---

### BENCH-04: MFU in Valid Range

**What:** MFU estimate is in (0, 1].

**Method:** Run benchmark with known GPU (mocked or real), check `0 < result.mfu_p50 <= 1.0`.

**Rationale:** MFU > 1.0 indicates a bug in the FLOP counting or peak TFLOPS lookup.

**Edge case:** MFU = None if GPU name not in registry — assert it is None, not > 1.

---

### BENCH-05: metrics.json Schema Validity

**What:** Written metrics.json contains all required top-level fields with correct types.

**Required fields:** `schema_version`, `machine_profile`, `run_config`, `throughput`, `timing`, `memory`, `mfu`, `env`, `timestamp`

**Method:** Write metrics.json to temp dir, load, check each field exists and has correct type.

**Expected:** All fields present. Numeric fields are float/int, not string.

---

### BENCH-06: Warmup Steps Excluded from Measurement

**What:** The step times list has exactly `measure_steps` entries (not `warmup_steps + measure_steps`).

**Method:** Run with `warmup_steps=10, measure_steps=5`, assert `len(result.step_times) == 5`.

**Expected:** Exactly 5 step time measurements.

---

### BENCH-07: Synthetic Mode Consistent Batch Shape

**What:** Synthetic batches have shape `(batch_size, seq_len)` for input_ids.

**Method:** Inspect batch returned by `_create_synthetic_batch()`.

**Expected:** `batch["input_ids"].shape == (per_device_batch_size, seq_len)`.

---

### BENCH-08: Tokens Per Step Computation

**What:** `tokens_per_step = per_device_batch_size * world_size * seq_len * grad_accum_steps`.

**Method:** Set known values, compute tokens_per_step, verify against formula.

**Example:** `batch=4, world=2, seq=512, accum=2` -> `tokens_per_step = 4 * 2 * 512 * 2 = 8192`.

---

### BENCH-09: 6ND MFU Formula Correctness

**What:** MFU computed by 6ND formula matches manual calculation.

**Method:**
```python
N = 7_000_000_000  # non-embedding params
tokens = 131072
step_time = 1.0    # seconds
gpus = 8
peak_tflops = 989.0  # H100

flops = 6 * N * tokens
achieved = flops / step_time / 1e12  # TFLOPS
expected_mfu = achieved / (gpus * peak_tflops)
```

Assert computed MFU matches expected within 1e-6.

---

### BENCH-10: Aggregation Statistics Correctness

**What:** p50, p90 match numpy computation on known data.

**Method:**
```python
import numpy as np
times = [1.0, 1.1, 1.2, 1.3, 2.0]
expected_p50 = np.percentile(times, 50)  # 1.2
expected_p90 = np.percentile(times, 90)  # 1.64
result = aggregate_stats(times, tokens_per_step=1024)
assert abs(result["step_time"]["p50"] - expected_p50) < 1e-9
```

---

### BENCH-11: Peak Memory Recorded

**What:** metrics.json `memory.peak_allocated_bytes` is non-negative integer.

**Method:** Run benchmark (synthetic ok), check memory field.

**Expected:** `result.peak_allocated_bytes >= 0`.

---

### BENCH-12: Repeat Mode Takes Best

**What:** With `repeat=3`, result has the best (lowest) p50 step time from 3 runs.

**Method:** Mock 3 runs with known p50 values `[1.1, 1.0, 1.05]`, verify result p50 is from run with 1.0.

---

### BENCH-13: Synthetic vs E2E Mode Switch

**What:** `mode="synthetic"` bypasses data iterator; `mode="e2e"` calls data iterator.

**Method:** Mock data iterator with a call counter. Synthetic mode: counter stays 0. E2E mode: counter > 0.

---

### BENCH-14: Loss Values in Output

**What:** metrics.json contains `loss.loss_values` list with `measure_steps` entries.

**Method:** Run benchmark, check `len(result.loss_values) == measure_steps`.

---

### BENCH-15: Atomic JSON Write

**What:** metrics.json is written atomically (write-to-temp + rename), not corrupted if interrupted.

**Method:** Check implementation uses `NamedTemporaryFile` + `os.replace()` or `shutil.move()`.

---

## Phase 3: `eval_small` Tests

### EVAL-01: Deterministic Output

**What:** Running eval twice on same model produces identical outputs.

**Method:**
```python
set_deterministic_seeds(42)
r1 = eval_harness.run(model, tokenizer)
set_deterministic_seeds(42)
r2 = eval_harness.run(model, tokenizer)
assert r1.ppl_fixed_shard == r2.ppl_fixed_shard
assert r1.task_probe_accuracy == r2.task_probe_accuracy
```

**Expected:** Bit-identical results.

---

### EVAL-02: Perplexity Is Positive

**What:** `ppl_fixed_shard` is a positive finite float.

**Method:** Run eval, assert `result.ppl_fixed_shard > 0` and `not math.isinf(result.ppl_fixed_shard)`.

---

### EVAL-03: Perplexity Computation on Known Text

**What:** PPL on a text of all-same token (trivial language model) is computable without error.

**Method:** Use a mock model that always assigns 0 loss. Verify PPL = exp(0) = 1.0.

---

### EVAL-04: Probe Accuracy in [0, 1]

**What:** All probe accuracies are in `[0, 1]`.

**Method:** Run eval, check `0 <= acc <= 1` for all values in `task_probe_accuracy`.

---

### EVAL-05: All Configured Probes Run

**What:** If config.probes = ["basic_reasoning_25", "format_following_30"], both appear in output.

**Method:** Check `set(result.task_probe_accuracy.keys()) == set(config.probes)`.

---

### EVAL-06: eval.json Schema

**What:** Saved eval.json has all required fields: `schema_version`, `machine_profile`, `ppl_fixed_shard`, `task_probe_accuracy`, `eval_config`, `env`, `timestamp`.

**Method:** Write to temp file, load, check all fields.

---

### EVAL-07: Fixed Shard SHA256 Verification

**What:** If shard file is modified, `verify_shard()` raises `ValueError`.

**Method:** Write modified content to temp file, assert `ValueError` is raised.

---

### EVAL-08: Greedy Decode Used

**What:** Generation uses `do_sample=False` and `temperature=0` (no randomness).

**Method:** Mock `model.generate()` and inspect call kwargs.

**Expected:** `do_sample=False` in call kwargs.

---

### EVAL-09: Exact Match Scoring

**What:** `score_exact_match("  Hello  ", "hello")` returns True (strip + lower).

**Method:** Unit test the scoring function directly.

---

### EVAL-10: Regex Scoring

**What:** `score_regex("The answer is 42.", r"\d+")` returns True.

**Method:** Unit test with known patterns and strings.

---

### EVAL-11: Probe Output Truncation

**What:** Probes truncate input to 512 tokens max to prevent OOM on long prompts.

**Method:** Pass a 1000-token prompt, verify tokenizer call uses `max_length=512, truncation=True`.

---

### EVAL-12: No Network Calls During Eval

**What:** `eval_harness.run()` makes no HTTP requests.

**Method:** Mock `urllib.request.urlopen` and `requests.get`, verify they are not called.

---

## Phase 4: `compare_baseline` Tests

### COMP-01: Pass Within Tolerance

**What:** When current metrics are within all tolerance thresholds, result is PASS.

**Method:** Create baseline and current with identical values, compare. Assert `PASS`.

---

### COMP-02: Fail on Throughput Drop > 5%

**What:** If `tokens_per_sec_p50` drops by 6%, status is FAIL.

**Method:**
```python
baseline = make_metrics(tokens_per_sec_p50=100_000)
current = make_metrics(tokens_per_sec_p50=94_000)  # -6%
result = compare_baseline.compare(current, baseline)
assert result.overall_status == Status.FAIL
```

---

### COMP-03: Pass at Exactly Threshold Boundary

**What:** A 5.0% drop with threshold 5.0% should be PASS (not strictly exceeding threshold).

**Method:** Set current tokens/sec to exactly 95% of baseline. Assert PASS.

**Rationale:** Boundary conditions are common sources of off-by-one bugs.

---

### COMP-04: Fail on Step Time Increase > 5%

**What:** If `step_time_p50` increases by 6%, status is FAIL.

**Method:** Similar to COMP-02 but for step time.

---

### COMP-05: Warn on Memory Increase > 10%

**What:** Memory increase of 11% produces WARN, not FAIL.

**Method:** Create metrics with 11% memory increase. Assert `Status.WARN` for memory check, `Status.PASS` for overall if no other failures.

---

### COMP-06: Fail on Perplexity Increase > 1.5%

**What:** PPL increase of 2% triggers FAIL.

**Method:** Baseline PPL=10.0, current PPL=10.21 (+2.1%). Assert FAIL.

---

### COMP-07: Fail on Probe Accuracy Drop > 2pp

**What:** Probe accuracy drop of 3pp (e.g., 0.90 -> 0.87) triggers FAIL.

**Method:** Baseline accuracy=0.90, current=0.87. Assert FAIL for that probe.

---

### COMP-08: Missing Baseline Returns SKIP

**What:** When no baseline file exists, compare returns `Status.SKIP`.

**Method:** Point baseline_dir at empty temp directory. Assert `result.overall_status == Status.SKIP`.

---

### COMP-09: Missing Probe in Current Returns WARN

**What:** If baseline has probe X but current doesn't, report WARN for probe X, not FAIL.

**Method:** Baseline has `basic_reasoning_25`, current doesn't. Assert WARN for that check.

---

### COMP-10: Report Contains All Metric Names

**What:** Formatted report string mentions every metric that was checked.

**Method:** Run compare, get report string, verify all expected metric names appear as substrings.

---

### COMP-11: Update Baseline Copies Files

**What:** `update_baseline()` copies metrics.json and eval.json to dest_dir.

**Method:** Write files to temp src dir, call update_baseline, verify files exist in dest dir.

---

### COMP-12: Delta Percentage Computation

**What:** Delta formula is `(current - baseline) / abs(baseline) * 100`.

**Method:** Unit test `compute_delta_pct(current=95.0, baseline=100.0)` returns -5.0.

---

### COMP-13: Loss Slope Stability Check

**What:** Positive loss slope above threshold triggers FAIL.

**Method:** Set `loss_slope=0.002` with threshold 0.001. Assert FAIL.

---

### COMP-14: Schema Version Mismatch

**What:** If baseline has `schema_version="0.9"` and current has `"1.0"`, returns WARN or SKIP.

**Method:** Write mismatched schema versions, compare, assert no exception.

---

### COMP-15: Report Format Includes Values

**What:** Report shows both baseline and current values for each failing metric.

**Method:** Parse report string for baseline/current values in failing rows.

---

## Phase 5: Profiling Tests

### PROF-01: Trace File Generated

**What:** Running with `--profile trace` creates at least one `.json` trace file.

**Method:** Run profiler with `active=2, repeat=1`, check for `.json` files in output dir.

**Expected:** At least 1 file matching `*_pt_trace.json` or `trace.json`.

---

### PROF-02: Chrome Trace File Generated

**What:** `prof.export_chrome_trace()` creates a file that is valid JSON.

**Method:** Export, load as JSON, verify it is a non-empty dict/list.

---

### PROF-03: record_function Regions in Trace

**What:** Named regions ("forward", "backward", "optimizer_step") appear in the trace.

**Method:** Load Chrome trace JSON, search for events with `name` matching region names.

```python
events = trace_data["traceEvents"]
event_names = {e["name"] for e in events}
assert "forward" in event_names
assert "backward" in event_names
```

---

### PROF-04: Schedule Produces Correct Span Count

**What:** With `active=3, repeat=2`, exactly 6 active-phase steps are profiled.

**Method:** Count distinct profiler step events in trace. Verify count = `active * repeat = 6`.

---

### PROF-05: Profile Directory Structure

**What:** Output directory has expected subdirectory structure after profiling.

**Method:** Check for `profile/tb/` directory with trace files after `--profile tb` run.

---

### PROF-06: Profiling Does Not Affect Benchmark Numbers

**What:** Benchmark step times from `--bench` run are not in the profile output dir.

**Method:** Structural test: verify `--profile` and `--bench` write to different output dirs.

---

### PROF-07: prof.step() Called Each Iteration

**What:** `prof.step()` is called exactly once per training step in the profiling loop.

**Method:** Mock profiler, count `step()` calls, verify equals `total_steps = (wait+warmup+active)*repeat`.

---

### PROF-08: Memory Timeline Export

**What:** `export_memory_timeline()` creates an HTML file when `profile_memory=True`.

**Method:** Run with memory profiling, check for `.html` file in output.

---

## Phase 6: CI Workflow Tests

### CI-01: End-to-End Pass on Stable Code

**What:** Running the full pipeline (collect + bench + eval + compare) on identical code produces PASS.

**Method:** Integration test that runs full pipeline twice and compares against its own first run.

---

### CI-02: End-to-End Fail on Injected Regression

**What:** Artificially degrading throughput by 10% produces FAIL in compare step.

**Method:** Patch step time computation to return 1.1x the actual time, run compare, assert FAIL.

---

### CI-03: Artifact Upload Paths Correct

**What:** CI workflow spec references correct artifact paths.

**Method:** Parse generated `perf_gate.yml`, verify `path:` values match actual output filenames.

---

### CI-04: Workflow Triggers on pull_request

**What:** Generated workflow file has `on: pull_request:` trigger.

**Method:** Parse YAML, assert `on.pull_request` is present.

---

### CI-05: Baseline Fetch Step Present

**What:** Workflow includes a step to fetch baseline files from main branch.

**Method:** Parse YAML, look for step containing `git show origin/main:bench/baselines` or equivalent.

---

### CI-06: Self-Hosted Runner Specified

**What:** Workflow runs on `self-hosted` runner (required for GPU).

**Method:** Parse YAML, check `runs-on` contains `self-hosted`.

---

### CI-07: Validate Script Reports All Gates

**What:** `validate_perf_gate.py` prints PASS/FAIL for all 3 done-when gates.

**Method:** Run validator, capture output, verify "Gate 1:", "Gate 2:", "Gate 3:" all appear.

---

### CI-08: Gate Fail Exits with Non-Zero Code

**What:** When any gate fails, the CI step exits with non-zero return code.

**Method:** Mock a gate failure, verify `sys.exit(1)` or equivalent is called.

---

### CI-09: Docs Generated

**What:** `ci_workflow_template.py` generates `docs/perf.md` alongside the YAML.

**Method:** Run script, check for `docs/perf.md` in output.

---

### CI-10: Generated Workflow YAML is Valid

**What:** Generated YAML file is syntactically valid and can be parsed.

**Method:** `yaml.safe_load(open("perf_gate.yml"))` without exception.

---

## Test Utility Functions

### `make_metrics()`

```python
def make_metrics(
    tokens_per_sec_p50: float = 100_000.0,
    step_time_p50_s: float = 1.0,
    peak_allocated_bytes: int = 50 * 1024**3,
    mfu_p50: float = 0.40,
    loss_slope: float = -0.001,
    machine_profile: str = "test_profile",
) -> dict:
    """Create a minimal valid metrics.json dict for testing."""
    return {
        "schema_version": "1.0",
        "machine_profile": machine_profile,
        "throughput": {"tokens_per_sec_p50": tokens_per_sec_p50},
        "timing": {"step_time_p50_s": step_time_p50_s},
        "memory": {"peak_allocated_bytes": peak_allocated_bytes},
        "mfu": {"mfu_p50": mfu_p50},
        "loss": {"loss_slope": loss_slope},
    }
```

### `make_eval_result()`

```python
def make_eval_result(
    ppl: float = 12.0,
    basic_reasoning: float = 0.92,
    format_following: float = 0.87,
    code_sanity: float = 0.90,
    machine_profile: str = "test_profile",
) -> dict:
    """Create a minimal valid eval.json dict for testing."""
    return {
        "schema_version": "1.0",
        "machine_profile": machine_profile,
        "ppl_fixed_shard": ppl,
        "task_probe_accuracy": {
            "basic_reasoning_25": basic_reasoning,
            "format_following_30": format_following,
            "code_sanity_20": code_sanity,
        },
    }
```

---

## Coverage Requirements

| Phase | Minimum Test Count | Required Branches |
|-------|-------------------|-------------------|
| collect_env | 10 | CPU path, GPU path, git missing |
| bench_train | 15 | synthetic, e2e, repeat, MFU=None |
| eval_small | 12 | perplexity, each probe type, determinism |
| compare_baseline | 15 | all tolerance rules, missing baseline |
| profiling | 8 | trace file, chrome, schedule count |
| CI workflow | 10 | YAML valid, end-to-end pass/fail |

**Total minimum: 70 tests.** Target 100+ for robustness.
