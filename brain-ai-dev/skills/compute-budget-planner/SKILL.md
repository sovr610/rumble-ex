---
name: Compute-Optimal Budget Planner (Chinchilla-style)
description: >
  This skill should be used when the user asks to "plan training budget",
  "estimate training time", "compute optimal model size", "Chinchilla scaling",
  "tokens per parameter ratio", "compute-optimal training", "FLOPs budget",
  "GPU hours estimation", "training cost estimate", "undertraining detection",
  "N_opt D_opt solver", "scaling law planning", "isoFLOPs curve",
  "validate training config budget", "tokens per param check",
  "wallclock estimate for training", "how long will training take",
  "is my model undertrained", "budget.json metadata",
  or needs guidance on compute-optimal sizing, Chinchilla-style scaling laws,
  training budget estimation, or launch-time budget validation.
version: 0.1.0
---

# Compute-Optimal Budget Planner (Chinchilla-style)

## Overview

Guide implementation of a budget-planning layer that turns "we have X GPUs for Y hours" (or "we can spend $Z") into a concrete, checkable training plan: model size, total tokens, steps, expected wallclock, and whether the run is likely undertrained. Based on Chinchilla-style compute-optimal scaling (scale tokens and params roughly proportionally). Standardize planner output into run metadata for apples-to-apples comparison across runs.

**Scope**: Dense decoder-only Transformer pretraining (cross-entropy). The Chinchilla heuristic is less reliable for instruction tuning, MoE, heavy retrieval, or long-context variants. Make compute model configurable and log assumptions explicitly.

## Public Contract

### BudgetPlanner

Core planner with three operational modes.

```python
class BudgetPlanner:
    def __init__(self, gpu_specs: GPUSpecTable, config: BudgetConfig): ...
    def validate_run(self, run: RunSpec) -> BudgetResult: ...         # Mode A
    def compute_required(self, model: ModelSpec) -> BudgetResult: ...  # Mode B
    def solve_optimal(self, budget: ComputeBudget) -> BudgetResult: ... # Mode C
```

### ChinchillaSolver

Compute-optimal frontier solver and undertraining detector.

```python
class ChinchillaSolver:
    def __init__(self, k: float = 6.0, tokens_per_param: float = 20.0): ...
    def n_opt(self, compute_flops: float) -> float: ...
    def d_opt(self, compute_flops: float) -> float: ...
    def undertraining_ratio(self, n_params: float, total_tokens: float) -> float: ...
    def warnings(self, ratio: float) -> List[str]: ...
```

### ReportGenerator

Structured and human-readable output.

```python
class ReportGenerator:
    def write_json(self, result: BudgetResult, path: str) -> None: ...
    def write_txt(self, result: BudgetResult, path: str) -> None: ...
    def write_svg(self, result: BudgetResult, path: str) -> None: ...  # isoFLOPs curve
```

## Key Concepts

### Three Planner Modes

| Mode | Input | Output |
|------|-------|--------|
| A: Validate Run | model size + run config + budget | tokens/param, undertraining ratio, warnings |
| B: Compute Required | model size + target tokens/param | required tokens, steps, wallclock, cost |
| C: Solve Optimal | compute budget (FLOPs or GPU-hours) | N_opt, D_opt, steps, expected wallclock |

### Core Formulas

**Token accounting**:
```
total_tokens = steps * global_batch * seq_len
```
Where `global_batch` is true global batch across all data-parallel ranks.

**Compute estimate** (dense Transformer training):
```
total_FLOPs = k * N_params * total_tokens    (default k=6)
```
The `k=6` approximation accounts for forward (~2ND) + backward (~4ND). Referenced in Chinchilla and standard in the scaling-laws literature.

**Wallclock from hardware**:
```
peak_FLOPs_per_s = num_gpus * peak_tflops(dtype) * 1e12
achieved_FLOPs_per_s = peak_FLOPs_per_s * utilization
wallclock_s = total_FLOPs / achieved_FLOPs_per_s
```

**Cost** (optional):
```
cost = wallclock_hours * num_gpus * cost_per_gpu_hour
```

### Chinchilla Compute-Optimal Target

```
target_tokens = tokens_per_param * N_params     (default tokens_per_param=20)
```
Derived from: 70B params trained on 1.4T tokens = 20 tokens/param. Chinchilla's core finding: under fixed compute, scale tokens and params proportionally.

### Undertraining Detection

```
undertraining_ratio = planned_tokens_per_param / tokens_per_param_target
```

| Ratio | Interpretation | Warning |
|-------|---------------|---------|
| < 0.5 | Severely undertrained | CRITICAL |
| 0.5 - 0.8 | Likely undertrained | WARNING |
| 0.8 - 1.5 | Near compute-optimal | OK |
| > 2.0 | Data-rich regime | INFO (confirm intent) |

### N_opt / D_opt Solver (Mode C)

Given compute budget `C`, coefficient `k`, target ratio `a = tokens_per_param`:
```
C = k * N * D,   D = a * N
N_opt = sqrt(C / (k * a))
D_opt = a * N_opt
```
Log as "heuristic optimum" — not universal truth.

### Kaplan vs Chinchilla Note

OpenAI's earlier scaling work suggested compute-optimal training leans toward larger models with fewer tokens. Chinchilla later found the opposite: tokens should scale proportionally with params. The planner makes this assumption explicit and configurable.

### GPU Spec Table

Maintain `gpu_specs.json` with peak tensor TFLOPS per dtype and memory. Default to **dense (non-sparse)** numbers unless explicitly training sparse models. Always log whether specs are sparse or dense.

### Launch-Time Integration

Call planner from training entrypoint before allocating resources:
1. Extract params, seq_len, batch, steps from config
2. Run planner -> print summary + warnings
3. Write `budget.json` into run directory alongside manifest
4. Never hard-fail by default (unless `--strict_budget` flag)

### Actionable Suggestions

Always emit corrective actions:
- "To reach 20 tokens/param, increase to X tokens -> Y steps -> ~Z hours on this cluster."
- "If time budget is fixed, reduce params to N' or accept tokens/param=T'."

## Configuration Surface

```python
@dataclass
class BudgetConfig:
    k: float = 6.0                         # FLOPs coefficient
    tokens_per_param_target: float = 20.0  # Chinchilla-style target
    default_utilization: float = 0.35      # MFU assumption
    cost_per_gpu_hour: Optional[float] = None

@dataclass
class RunSpec:
    n_params: float                        # Non-embedding params
    seq_len: int
    global_batch: int
    steps: int
    num_gpus: int = 1
    gpu_type: str = "H100_SXM"
    dtype: str = "bf16"

@dataclass
class ModelSpec:
    n_params: float
    seq_len: int = 2048
    global_batch: int = 2048

@dataclass
class ComputeBudget:
    mode: str = "flops"                    # "flops" | "time" | "money"
    total_flops: Optional[float] = None
    num_gpus: Optional[int] = None
    gpu_type: Optional[str] = None
    hours: Optional[float] = None
    utilization: float = 0.35
    budget_dollars: Optional[float] = None
```

## Done-When Gates

1. **Chinchilla Sanity** — `params=70B, tokens=1.4T` yields `tokens_per_param=20.0` within float tolerance; FLOPs = `k * 70e9 * 1.4e12` with `k=6`.
2. **Undertraining Detection** — Planner correctly warns when `tokens_per_param < 0.8 * target`; does not warn when within optimal range.
3. **N_opt/D_opt Solver** — `solve_optimal(C)` returns N_opt, D_opt such that `k * N_opt * D_opt == C` within rounding tolerance.

## Resources

### Reference Files
- **`references/chinchilla-scaling.md`** — Chinchilla paper findings, tokens_per_param derivation, Kaplan vs Chinchilla, compute-optimal frontier theory
- **`references/flops-accounting.md`** — C = kND derivation, forward/backward split, MFU, wallclock estimation formulas, gradient accumulation
- **`references/gpu-spec-table.md`** — A100/H100/H200/L40S specs, sparse vs dense, dtype-specific TFLOPS, memory, vendor sourcing
- **`references/planner-modes.md`** — Three modes detailed with examples, actionable suggestions, launch-time integration, metadata schema
- **`references/testing-matrix.md`** — Test scenarios for all phases

### Asset Files
- **`assets/budget_planner_template.py`** — BudgetPlanner with 3 modes, formula implementations, self-tests
- **`assets/chinchilla_solver_template.py`** — ChinchillaSolver with N_opt/D_opt, undertraining detector, warnings
- **`assets/gpu_specs_template.py`** — GPUSpecTable with JSON registry, lookup, fallback to user-supplied values
- **`assets/report_generator_template.py`** — budget.json, budget.txt, optional budget.svg (isoFLOPs) generation
- **`assets/cli_template.py`** — CLI entrypoint for all 3 modes with argparse
- **`assets/budget_config_template.py`** — All config/result dataclasses, validation, serialization

### Scripts
- **`scripts/validate_budget.py`** — Validates done-when gates
- **`scripts/gen_budget_tests.py`** — Generates 100+ pytest test cases
- **`scripts/chinchilla_table.py`** — Generates reference table of N_opt/D_opt for common compute budgets
