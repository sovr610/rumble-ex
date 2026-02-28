# Testing Matrix: Compute-Optimal Budget Planner

## Overview

This document organizes all test scenarios by component and category. It is used by `scripts/gen_budget_tests.py` to generate the full pytest test suite and by `scripts/validate_budget.py` to check done-when gates.

---

## 1. Chinchilla Sanity Tests

These tests verify the fundamental Chinchilla relationship using the canonical 70B/1.4T data point.

### 1.1 tokens_per_param Sanity

```
Input:  n_params=70e9, total_tokens=1.4e12
Expect: tokens_per_param == 20.0 (within 1e-6 relative tolerance)
Method: ChinchillaSolver.undertraining_ratio(n_params, total_tokens) * tokens_per_param_target
```

### 1.2 FLOPs Sanity

```
Input:  n_params=70e9, total_tokens=1.4e12, k=6
Expect: total_flops == 6 * 70e9 * 1.4e12 = 5.88e23 (within 1e-10 relative tolerance)
Method: BudgetPlanner._compute_flops(70e9, 1.4e12, 6)
```

### 1.3 tokens_per_param Derivation

```
Input:  n_params=140e9, total_tokens=1.4e12
Expect: tokens_per_param == 10.0
Note:   Doubling params while keeping tokens constant halves tokens/param
```

### 1.4 Inverse Relationship

```
Input:  n_params=70e9, total_tokens=2.8e12
Expect: tokens_per_param == 40.0 (doubling tokens doubles ratio)
```

---

## 2. FLOPs Scaling Monotonicity

### 2.1 Token Monotonicity

```
For n=70e9, k=6:
  flops(D=1e12) < flops(D=2e12) < flops(D=4e12)
  ratio: flops(D=2e12) / flops(D=1e12) == 2.0 (exact)
```

### 2.2 Parameter Monotonicity

```
For D=1e12, k=6:
  flops(N=7e9) < flops(N=70e9) < flops(N=700e9)
  ratio: flops(N=70e9) / flops(N=7e9) == 10.0 (exact)
```

### 2.3 k-Coefficient Linearity

```
flops(N=70e9, D=1e12, k=6) == 2 * flops(N=70e9, D=1e12, k=3)
```

### 2.4 Joint Scaling

```
flops(N=2N0, D=2D0) == 4 * flops(N=N0, D=D0)
(FLOPs scale as N*D)
```

---

## 3. Wallclock Scaling Tests

### 3.1 Utilization Inverse

```
wallclock(util=0.35) == wallclock(util=0.70) * 2
(halving utilization doubles wallclock)
```

### 3.2 GPU Count Inverse

```
wallclock(num_gpus=8) == wallclock(num_gpus=16) * 2
(doubling GPUs halves wallclock)
```

### 3.3 TFLOPS Inverse

```
wallclock(peak_tflops=989) ~= wallclock(peak_tflops=312) * (312/989)
```

### 3.4 Token Proportionality

```
wallclock(D=2e12) == 2 * wallclock(D=1e12)  for same N, GPU config
```

### 3.5 Non-Negative

```
wallclock(...) > 0 for all valid inputs
```

---

## 4. N_opt / D_opt Solver Tests

### 4.1 Correctness: k*N*D == C

```
For any valid C:
  N_opt = sqrt(C / (k * a))
  D_opt = a * N_opt
  k * N_opt * D_opt == C  within 1e-9 relative tolerance
```

### 4.2 Proportionality: N_opt ~ sqrt(C)

```
N_opt(C=4e23) == 2 * N_opt(C=1e23)  (within 1e-9 tolerance)
D_opt(C=4e23) == 2 * D_opt(C=1e23)
```

### 4.3 Positive Values

```
N_opt(C) > 0 for all C > 0
D_opt(C) > 0 for all C > 0
```

### 4.4 tokens_per_param Consistency

```
D_opt / N_opt == tokens_per_param_target
```

### 4.5 Chinchilla Reference

```
Input: C = 5.88e23, k=6, a=20
Expect: N_opt ≈ 70e9, D_opt ≈ 1.4e12 (within 1% of reference values)
Note: 6 * 70e9 * 1.4e12 = 5.88e23, so this is the exact Chinchilla point
```

### 4.6 Scale Sensitivity

```
N_opt(C=1e20) > 0 and < 1e15  (reasonable range)
N_opt(C=1e26) > 0 and < 1e16  (reasonable range for very large budget)
```

---

## 5. Undertraining Warning Thresholds

### 5.1 Critical Threshold (ratio < 0.5)

```
planned_tokens_per_param = 8.0   (target=20.0, ratio=0.4)
Expect: warnings contain "CRITICAL" severity message
Expect: warnings NOT empty
```

### 5.2 Warning Threshold (0.5 <= ratio < 0.8)

```
planned_tokens_per_param = 12.0  (target=20.0, ratio=0.6)
Expect: warnings contain "WARNING" severity message
Expect: warnings NOT contain "CRITICAL"
```

### 5.3 OK Range (0.8 <= ratio <= 2.0)

```
planned_tokens_per_param = 20.0  (target=20.0, ratio=1.0)
Expect: warnings is empty (or no severity warnings)
planned_tokens_per_param = 16.0  (target=20.0, ratio=0.8)
Expect: warnings is empty
planned_tokens_per_param = 30.0  (target=20.0, ratio=1.5)
Expect: warnings is empty
```

### 5.4 Info Threshold (ratio > 2.0)

```
planned_tokens_per_param = 142.0  (target=20.0, ratio=7.1)
Expect: warnings contain "INFO" (confirm overtrain intent)
Expect: warnings NOT contain "CRITICAL" or "WARNING"
```

### 5.5 Boundary Exactness

```
ratio = 0.5 exactly: should be WARNING (not CRITICAL)
ratio = 0.8 exactly: should be OK (no warning)
ratio = 2.0 exactly: should be OK (no warning)
ratio = 2.0 + epsilon: should be INFO
```

---

## 6. GPU Spec Lookup Tests

### 6.1 A100 SXM BF16

```
spec = GPUSpecTable().lookup("A100_80GB_SXM", dtype="bf16")
Expect: spec.peak_tflops == 312.0
Expect: spec.mem_gb == 80
Expect: spec.is_sparse == False
```

### 6.2 H100 SXM BF16

```
spec = GPUSpecTable().lookup("H100_SXM", dtype="bf16")
Expect: spec.peak_tflops == 989.0
Expect: spec.mem_gb == 80
Expect: spec.is_sparse == False
```

### 6.3 H200 BF16

```
spec = GPUSpecTable().lookup("H200", dtype="bf16")
Expect: spec.peak_tflops == 989.0
Expect: spec.mem_gb == 141
```

### 6.4 V100 FP16

```
spec = GPUSpecTable().lookup("V100", dtype="fp16")
Expect: spec.peak_tflops == 125.0
Expect: spec.mem_gb == 32
```

### 6.5 Unknown GPU — Requires User Input

```
GPUSpecTable().lookup("NonExistentGPU9000", dtype="bf16")
Expect: raises ValueError or returns None
Expect: validate_or_fallback() uses user_peak_tflops when provided
```

### 6.6 Fuzzy Matching — Lowercase

```
GPUSpecTable().lookup("h100") -> H100 SXM spec
GPUSpecTable().lookup("a100") -> A100 80GB SXM spec
GPUSpecTable().lookup("4090") -> RTX 4090 spec
```

### 6.7 Fuzzy Matching — Mixed Case

```
GPUSpecTable().lookup("H100 SXM") -> H100 SXM spec
GPUSpecTable().lookup("H100_SXM") -> H100 SXM spec
GPUSpecTable().lookup("h100_sxm") -> H100 SXM spec
```

### 6.8 Dtype Selection

```
spec_bf16 = GPUSpecTable().lookup("H100_SXM", dtype="bf16")
spec_fp16 = GPUSpecTable().lookup("H100_SXM", dtype="fp16")
spec_fp32 = GPUSpecTable().lookup("H100_SXM", dtype="fp32")
Expect: spec_bf16.peak_tflops > spec_fp32.peak_tflops
Expect: spec_bf16.peak_tflops == spec_fp16.peak_tflops  (H100 has equal bf16/fp16)
```

### 6.9 L40S

```
spec = GPUSpecTable().lookup("L40S", dtype="bf16")
Expect: spec.peak_tflops == 362.0
Expect: spec.mem_gb == 48
```

---

## 7. Report Generation Tests

### 7.1 JSON Roundtrip

```
result = BudgetPlanner(...).validate_run(run_spec)
path = tmp_path / "budget.json"
ReportGenerator().write_json(result, path)
loaded = json.loads(path.read_text())
Expect: loaded["mode"] == "validate_run"
Expect: loaded["derived"]["total_tokens"] == result.derived["total_tokens"]
Expect: loaded["assumptions"]["k"] == 6.0
Expect: "schema_version" in loaded
```

### 7.2 JSON Schema Completeness

```
json_data = loaded_budget_json
Required top-level keys: mode, inputs, assumptions, derived, warnings, suggestions, schema_version
Required derived keys: total_tokens, tokens_per_param, total_flops, predicted_wallclock_hours, undertraining_ratio
```

### 7.3 TXT Non-Empty and Contains Key Fields

```
result = BudgetPlanner(...).validate_run(run_spec)
path = tmp_path / "budget.txt"
ReportGenerator().write_txt(result, path)
content = path.read_text()
Expect: len(content) > 100
Expect: "COMPUTE BUDGET" in content
Expect: "CHINCHILLA CHECK" in content
Expect: "tokens/param" in content
```

### 7.4 SVG Created (if matplotlib available)

```
result = BudgetPlanner(...).validate_run(run_spec)
path = tmp_path / "budget.svg"
ReportGenerator().write_svg(result, path)
if matplotlib_available:
    Expect: path.exists() == True
    Expect: path.stat().st_size > 0
    Expect: path.read_text().startswith("<?xml") or "<svg" in path.read_text()
else:
    Expect: no exception raised (graceful skip)
```

### 7.5 All Modes Produce Valid JSON

```
For each of [validate_run, compute_required, solve_optimal]:
    result = planner.run_mode(...)
    writer.write_json(result, path)
    Expect: json.loads(path.read_text()) validates successfully
```

---

## 8. CLI Tests

### 8.1 Validate Subcommand Parses Correctly

```
args = parse_args(["validate", "--params", "7e9", "--seq_len", "2048",
                   "--batch", "2048", "--steps", "100000",
                   "--gpus", "8", "--gpu", "H100_SXM"])
Expect: args.mode == "validate"
Expect: args.params == 7e9
Expect: args.steps == 100000
```

### 8.2 Required Subcommand Parses Correctly

```
args = parse_args(["required", "--params", "70e9",
                   "--tokens_per_param", "20",
                   "--gpus", "8", "--gpu", "H100_SXM"])
Expect: args.params == 70e9
Expect: args.tokens_per_param == 20.0
```

### 8.3 Optimal Subcommand with FLOPs

```
args = parse_args(["optimal", "--compute_flops", "5e23"])
Expect: args.compute_flops == 5e23
```

### 8.4 Optimal Subcommand with GPU-Hours

```
args = parse_args(["optimal", "--gpus", "8", "--gpu", "H100_SXM",
                   "--hours", "1000", "--util", "0.35"])
Expect: args.hours == 1000.0
Expect: args.util == 0.35
```

### 8.5 Help Text

```
parse_args(["--help"]) -> does not raise (or raises SystemExit(0))
parse_args(["validate", "--help"]) -> does not raise (or raises SystemExit(0))
```

### 8.6 Missing Required Args Fail Gracefully

```
parse_args(["validate"]) -> raises SystemExit (argparse error)
parse_args(["optimal"]) -> raises SystemExit (no budget specified)
```

### 8.7 Output Directory Flag

```
args = parse_args(["validate", ..., "--out", "/tmp/test_run"])
Expect: args.out == "/tmp/test_run"
```

### 8.8 Strict Budget Flag

```
args = parse_args(["validate", ..., "--strict_budget"])
Expect: args.strict_budget == True
```

---

## 9. Edge Cases

### 9.1 Near-Zero Parameters

```
BudgetPlanner._compute_flops(n_params=1, total_tokens=1e9, k=6)
Expect: returns 6e9 (no division by zero, no exception)
```

### 9.2 Near-Zero Tokens

```
BudgetPlanner._compute_flops(n_params=70e9, total_tokens=1, k=6)
Expect: returns 420e9 (no exception)
```

### 9.3 Very Large Parameters (1T)

```
solver = ChinchillaSolver(k=6, tokens_per_param=20)
n_opt = solver.n_opt(C=1e28)
Expect: n_opt > 0 and not inf and not nan
```

### 9.4 Negative Values — Validation

```
BudgetConfig(k=-1) -> raises ValueError
BudgetConfig(tokens_per_param_target=-5) -> raises ValueError
RunSpec(n_params=-1, ...) -> raises ValueError
```

### 9.5 Zero Utilization

```
BudgetConfig(default_utilization=0.0) -> raises ValueError
```

### 9.6 Utilization > 1.0

```
BudgetConfig(default_utilization=1.5) -> raises ValueError
```

### 9.7 Exact Boundary: utilization = 1.0

```
BudgetConfig(default_utilization=1.0) -> valid (theoretical maximum)
```

### 9.8 Very Large Compute Budget

```
solver.n_opt(C=1e30)
Expect: finite positive float
Expect: no overflow exception
```

### 9.9 Float Precision: tokens_per_param = 20.0 exactly

```
n_params = 70e9
total_tokens = n_params * 20.0
ratio = ChinchillaSolver().undertraining_ratio(n_params, total_tokens)
Expect: abs(ratio - 1.0) < 1e-10
```

---

## 10. Mode A/B/C Correctness Integration Tests

### 10.1 Mode A → B Consistency

```
Given a compute-optimal run (ratio == 1.0) as RunSpec:
  result_A = planner.validate_run(run)
  model = ModelSpec(n_params=run.n_params, ...)
  result_B = planner.compute_required(model)
  Expect: abs(result_A.derived["tokens_per_param"] - 20.0) < 0.01
  Expect: abs(result_B.derived["total_tokens"] - result_A.derived["total_tokens"]) / result_A.derived["total_tokens"] < 0.001
```

### 10.2 Mode C → B Consistency

```
result_C = planner.solve_optimal(budget)
model = ModelSpec(n_params=result_C.derived["n_opt"], ...)
result_B = planner.compute_required(model)
Expect: result_B.derived["tokens_per_param"] ≈ tokens_per_param_target
```

### 10.3 Cost Computation

```
wallclock_h = 100.0
num_gpus = 8
cost_per_gpu_hour = 4.0
Expect: cost = 100.0 * 8 * 4.0 = 3200.0
```

### 10.4 Multi-GPU Wallclock

```
wallclock_1gpu = planner._compute_wallclock(total_flops=1e23, num_gpus=1, peak_tflops=989, util=0.35)
wallclock_8gpu = planner._compute_wallclock(total_flops=1e23, num_gpus=8, peak_tflops=989, util=0.35)
Expect: wallclock_1gpu == 8 * wallclock_8gpu
```

### 10.5 All Three Modes Produce BudgetResult

```
For each mode:
  result = planner.run(...)
  Expect: isinstance(result, BudgetResult)
  Expect: result.mode in ["validate_run", "compute_required", "solve_optimal"]
  Expect: result.warnings is not None  (can be empty list)
  Expect: result.suggestions is not None
```

---

## 11. Dataclass Validation Tests

### 11.1 BudgetConfig Defaults Valid

```
config = BudgetConfig()
Expect: config.k == 6.0
Expect: config.tokens_per_param_target == 20.0
Expect: config.default_utilization == 0.35
Expect: BudgetConfig._validate(config) does not raise
```

### 11.2 Serialization Roundtrip

```
config = BudgetConfig(k=8.0, tokens_per_param_target=30.0)
d = config.to_dict()
config2 = BudgetConfig.from_dict(d)
Expect: config2.k == 8.0
Expect: config2.tokens_per_param_target == 30.0
```

### 11.3 RunSpec Validation

```
RunSpec(n_params=0, seq_len=2048, global_batch=2048, steps=1000) -> raises ValueError
RunSpec(n_params=7e9, seq_len=-1, ...) -> raises ValueError
```

### 11.4 BudgetResult Contains All Required Fields

```
result = BudgetResult(mode="validate_run", inputs={}, assumptions={}, derived={}, warnings=[], suggestions=[])
Expect: result.mode == "validate_run"
Expect: result.warnings == []
Expect: result.to_dict() is a dict
```
