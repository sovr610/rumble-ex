# Planner Modes: Detailed Reference

## Overview

The BudgetPlanner operates in three distinct modes, each answering a different planning question. This document describes each mode in detail, including inputs, computation steps, output format, and worked examples.

---

## Mode A: Validate Run

**Question**: "I have an existing training config. Is it compute-optimal? How long will it take?"

**Method**: `BudgetPlanner.validate_run(run: RunSpec) -> BudgetResult`

### Inputs (RunSpec)

| Field | Type | Description |
|-------|------|-------------|
| `n_params` | float | Non-embedding parameter count |
| `seq_len` | int | Sequence length in tokens |
| `global_batch` | int | Global batch size (all GPUs combined) |
| `steps` | int | Number of optimizer steps |
| `num_gpus` | int | Number of GPUs |
| `gpu_type` | str | GPU model name (e.g., "H100_SXM") |
| `dtype` | str | Training dtype ("bf16", "fp16", "fp32") |

### Computation Steps

```
1. total_tokens = steps * global_batch * seq_len
2. tokens_per_param = total_tokens / n_params
3. total_flops = k * n_params * total_tokens
4. peak_flops_per_s = num_gpus * peak_tflops(dtype) * 1e12
5. achieved_flops_per_s = peak_flops_per_s * utilization
6. predicted_wallclock_s = total_flops / achieved_flops_per_s
7. predicted_wallclock_h = predicted_wallclock_s / 3600
8. undertraining_ratio = tokens_per_param / tokens_per_param_target
9. n_opt = sqrt(total_flops / (k * tokens_per_param_target))
10. d_opt = tokens_per_param_target * n_opt
11. Generate warnings/suggestions based on undertraining_ratio
```

### Worked Example

```
Config: 7B model, 2T tokens, 8x H100 SXM, bf16, 35% MFU

n_params     = 7e9
total_tokens = 2e12
k            = 6

tokens_per_param = 2e12 / 7e9 ≈ 285.7
total_flops = 6 * 7e9 * 2e12 = 8.4e22
peak_per_gpu = 989e12 FLOPs/s (H100 SXM bf16 dense)
achieved = 8 * 989e12 * 0.35 = 2.7692e15 FLOPs/s
wallclock_s = 8.4e22 / 2.7692e15 ≈ 3.033e7 s
wallclock_h ≈ 8,425 hours (cluster)
wallclock_per_gpu_h ≈ 1,053 hours ≈ 44 days

undertraining_ratio = 285.7 / 20 ≈ 14.3
Status: INFO — well above compute-optimal (intentional overtrain regime)
```

### Output (BudgetResult)

```json
{
  "mode": "validate_run",
  "inputs": {
    "n_params": 7e9,
    "seq_len": 2048,
    "global_batch": 2048,
    "steps": 488281,
    "num_gpus": 8,
    "gpu_type": "H100_SXM",
    "dtype": "bf16"
  },
  "assumptions": {
    "k": 6.0,
    "tokens_per_param_target": 20.0,
    "utilization": 0.35,
    "peak_tflops_bf16": 989.0,
    "is_sparse": false
  },
  "derived": {
    "total_tokens": 2e12,
    "tokens_per_param": 285.7,
    "total_flops": 8.4e22,
    "predicted_wallclock_hours": 8425.0,
    "predicted_cost_usd": null,
    "undertraining_ratio": 14.3,
    "n_opt": 2.65e9,
    "d_opt": 5.3e10
  },
  "warnings": [],
  "suggestions": [
    "tokens_per_param=285.7 >> target=20.0 (ratio=14.3x). This is intentional overtrain. Confirm this is for inference efficiency.",
    "Compute-optimal for this FLOP budget: N_opt=2.65B params trained on 53B tokens."
  ]
}
```

---

## Mode B: Compute Required

**Question**: "I have a model of size N. How many tokens, steps, GPU-hours, and dollars do I need to train it compute-optimally?"

**Method**: `BudgetPlanner.compute_required(model: ModelSpec) -> BudgetResult`

### Inputs (ModelSpec)

| Field | Type | Description |
|-------|------|-------------|
| `n_params` | float | Non-embedding parameter count |
| `seq_len` | int | Sequence length (default: 2048) |
| `global_batch` | int | Global batch size (default: 2048) |

Plus from `BudgetConfig`:
- `tokens_per_param_target` (default: 20.0)
- `num_gpus`, `gpu_type`, `utilization`, `cost_per_gpu_hour`

### Computation Steps

```
1. target_tokens = tokens_per_param_target * n_params
2. steps = ceil(target_tokens / (global_batch * seq_len))
3. actual_tokens = steps * global_batch * seq_len   # may differ slightly due to ceil
4. total_flops = k * n_params * actual_tokens
5. wallclock_h = total_flops / (num_gpus * peak_tflops * 1e12 * utilization) / 3600
6. cost_usd = wallclock_h * num_gpus * cost_per_gpu_hour  (if cost_per_gpu_hour set)
```

### Worked Example

```
70B model, target 20 tokens/param, 8x H100 SXM, bf16, 35% MFU, seq=2048, batch=2048, $4/H100-hr

target_tokens = 20 * 70e9 = 1.4e12 tokens
tokens_per_step = 2048 * 2048 = 4,194,304
steps = ceil(1.4e12 / 4,194,304) = ceil(333,862.3) = 333,863

actual_tokens = 333,863 * 4,194,304 ≈ 1.4e12

total_flops = 6 * 70e9 * 1.4e12 = 5.88e23

achieved = 8 * 989e12 * 0.35 = 2.7692e15 FLOPs/s
wallclock_s = 5.88e23 / 2.7692e15 = 2.123e8 s
wallclock_h ≈ 58,972 hours (8-GPU cluster run)
wall_clock_elapsed ≈ 7,372 hours ≈ 307 days (single cluster)

cost = 7,372 * 8 * $4 = $235,904
```

### Output (BudgetResult)

```json
{
  "mode": "compute_required",
  "inputs": {
    "n_params": 70e9,
    "seq_len": 2048,
    "global_batch": 2048,
    "tokens_per_param_target": 20.0
  },
  "assumptions": {
    "k": 6.0,
    "utilization": 0.35,
    "num_gpus": 8,
    "gpu_type": "H100_SXM",
    "peak_tflops_bf16": 989.0
  },
  "derived": {
    "target_tokens": 1.4e12,
    "required_steps": 333863,
    "actual_tokens": 1.40002e12,
    "tokens_per_param": 20.0,
    "total_flops": 5.88e23,
    "predicted_wallclock_hours": 58972,
    "predicted_cost_usd": 235904,
    "undertraining_ratio": 1.0,
    "n_opt": 70e9,
    "d_opt": 1.4e12
  },
  "warnings": [],
  "suggestions": []
}
```

---

## Mode C: Solve Optimal

**Question**: "I have a compute budget (FLOPs, GPU-hours, or dollars). What model size and token count are optimal?"

**Method**: `BudgetPlanner.solve_optimal(budget: ComputeBudget) -> BudgetResult`

### Inputs (ComputeBudget)

| Field | Type | Description |
|-------|------|-------------|
| `mode` | str | "flops", "time", or "money" |
| `total_flops` | float | Total FLOPs budget (if mode="flops") |
| `num_gpus` | int | Number of GPUs (if mode="time" or "money") |
| `gpu_type` | str | GPU model (if mode="time" or "money") |
| `hours` | float | Cluster-hours (if mode="time") |
| `utilization` | float | MFU assumption |
| `budget_dollars` | float | Total dollar budget (if mode="money") |

### Computation Steps

```
# Step 1: Convert budget to total_flops
if mode == "time":
    total_flops = hours * 3600 * num_gpus * peak_tflops * 1e12 * utilization
elif mode == "money":
    hours = budget_dollars / (num_gpus * cost_per_gpu_hour)
    total_flops = hours * 3600 * num_gpus * peak_tflops * 1e12 * utilization
# else: total_flops provided directly

# Step 2: Solve N_opt and D_opt
a = tokens_per_param_target
N_opt = sqrt(total_flops / (k * a))
D_opt = a * N_opt

# Step 3: Compute derived metrics
# (same as Mode B with n_params=N_opt and target_tokens=D_opt)
```

### Worked Example

```
Budget: 5e23 FLOPs, k=6, tokens_per_param_target=20

N_opt = sqrt(5e23 / (6 * 20)) = sqrt(5e23 / 120) = sqrt(4.1667e21)
N_opt ≈ 6.455e10 ≈ 64.6B params

D_opt = 20 * 64.6e9 = 1.292e12 tokens

Verification: 6 * 64.6e9 * 1.292e12 = 5.0035e23 ≈ 5e23 ✓ (rounding)
```

```
Budget: 1000 hours on 8x H100 SXM at 35% MFU

total_flops = 1000 * 3600 * 8 * 989e12 * 0.35
            = 1000 * 3600 * 2.7692e15
            = 9.969e21

N_opt = sqrt(9.969e21 / 120) = sqrt(8.307e19) ≈ 2.88e10 ≈ 28.8B

D_opt = 20 * 28.8e9 = 5.76e11 = 576B tokens
```

### Output (BudgetResult)

```json
{
  "mode": "solve_optimal",
  "inputs": {
    "compute_budget_mode": "flops",
    "total_flops": 5e23,
    "tokens_per_param_target": 20.0,
    "k": 6.0
  },
  "assumptions": {
    "tokens_per_param_target": 20.0,
    "k": 6.0
  },
  "derived": {
    "n_opt": 6.455e10,
    "d_opt": 1.291e12,
    "tokens_per_param": 20.0,
    "total_flops": 5e23,
    "predicted_wallclock_hours": null,
    "undertraining_ratio": 1.0
  },
  "warnings": [
    "N_opt/D_opt are heuristic Chinchilla estimates. Actual optimal may differ by architecture."
  ],
  "suggestions": [
    "Train a ~64.6B parameter model on ~1.29T tokens.",
    "Use compute_required mode to estimate wallclock and cost for this configuration."
  ]
}
```

---

## Actionable Suggestions Format

The planner always emits specific, quantitative corrective actions. Examples:

### Undertraining suggestions (Mode A)
```
"To reach 20 tokens/param for this 7B model:
  - Required tokens: 140B (currently: 20B)
  - At global_batch=2048, seq=2048: 16,276 additional steps (total: 22,888)
  - At 35% MFU on 8x H100: +18.4 GPU-hours additional time"

"If time budget is fixed at 24 hours:
  - Maximum tokens at 35% MFU: ~45B tokens
  - Resulting tokens/param: 6.4 (undertrained; ratio=0.32)
  - To hit 20 tokens/param in 24 hours: reduce params to ~2.25B"
```

### Overtrain info (Mode A)
```
"tokens_per_param=143 >> target=20. This is LLaMA-style overtrain for inference efficiency.
  - Compute-optimal for this budget: 35B params / 700B tokens
  - If serving cost is the concern, current 7B model may be justified
  - Verify this is intentional; no action required if inference-optimal"
```

---

## Launch-Time Integration

### Integration Pattern

```python
# In training entrypoint (e.g., train.py):
from budget_planner import BudgetPlanner, GPUSpecTable, BudgetConfig, RunSpec

def pre_training_budget_check(args, run_dir):
    specs = GPUSpecTable()
    config = BudgetConfig(
        k=6.0,
        tokens_per_param_target=args.tokens_per_param_target or 20.0,
        default_utilization=args.mfu or 0.35,
        cost_per_gpu_hour=args.cost_per_gpu_hour,
    )
    planner = BudgetPlanner(specs, config)

    run = RunSpec(
        n_params=count_non_embedding_params(model),
        seq_len=args.seq_len,
        global_batch=args.global_batch,
        steps=args.max_steps,
        num_gpus=args.num_gpus,
        gpu_type=args.gpu_type or "H100_SXM",
        dtype=args.dtype or "bf16",
    )

    result = planner.validate_run(run)

    # Print summary
    reporter = ReportGenerator()
    reporter.write_txt(result, path=None)   # prints to stdout
    reporter.write_json(result, path=os.path.join(run_dir, "budget.json"))

    # Optionally hard-fail on warnings
    if args.strict_budget and result.warnings:
        raise RuntimeError(f"Budget check failed: {result.warnings}")
```

### Never Hard-Fail by Default

The planner is advisory, not gating. Training runs that are undertrained or overrun budget should proceed with logged warnings. Only `--strict_budget` flag triggers hard failures, and only when the user explicitly opts in.

---

## Output File Formats

### budget.json Schema

```json
{
  "schema_version": "1.0",
  "generated_at": "2025-01-15T14:23:11Z",
  "mode": "validate_run | compute_required | solve_optimal",
  "inputs": { /* RunSpec, ModelSpec, or ComputeBudget fields */ },
  "assumptions": {
    "k": 6.0,
    "tokens_per_param_target": 20.0,
    "utilization": 0.35,
    "gpu_spec_source": "table | user_supplied",
    "peak_tflops": 989.0,
    "is_sparse": false
  },
  "derived": {
    "total_tokens": null,
    "tokens_per_param": null,
    "total_flops": null,
    "predicted_wallclock_hours": null,
    "predicted_cost_usd": null,
    "undertraining_ratio": null,
    "n_opt": null,
    "d_opt": null
  },
  "warnings": [],
  "suggestions": []
}
```

### budget.txt Format

```
=============================================================
  COMPUTE BUDGET PLANNER REPORT
  Mode: Validate Run
  Generated: 2025-01-15 14:23:11
=============================================================

PLAN SUMMARY
  Model parameters:      7.00B (non-embedding)
  Training tokens:       2,000.00B (2.00T)
  Sequence length:       2,048
  Global batch size:     2,048
  Optimizer steps:       488,281

COMPUTE BUDGET
  GPU type:              H100 SXM (bf16 dense)
  Peak TFLOPS:           989.0 TFLOPS
  Number of GPUs:        8
  MFU (utilization):     35.0%
  Total FLOPs:           8.40e+22
  Predicted wallclock:   1,053.1 hours per GPU
                         131.6 hours elapsed (8 GPUs)

CHINCHILLA CHECK
  tokens/param:          285.71 (target: 20.0)
  Undertraining ratio:   14.29x
  Status:                INFO — overtrain regime (likely intentional)

SUGGESTIONS
  [INFO] tokens_per_param=285.7 >> target=20.0. LLaMA-style overtrain detected.
  [INFO] Compute-optimal for this FLOP budget: N_opt=2.65B, D_opt=53.0B tokens.

=============================================================
```

### budget.svg: isoFLOPs Curves

The SVG plot shows:
- **X-axis**: Model parameters (log scale, 1e8 to 1e12)
- **Y-axis**: Training tokens (log scale, 1e9 to 1e14)
- **Gray lines**: isoFLOPs curves for C = 1e19, 1e20, ..., 1e24 (one per decade)
- **Blue line**: Compute-optimal frontier D = a * N
- **Red dot**: Current planned run (n_params, total_tokens)
- **Legend**: Shows which isoFLOPs line the current run is on

Rendered with matplotlib; gracefully omitted if matplotlib unavailable.
