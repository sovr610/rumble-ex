# Intermediate Representation (IR) Entities

## Overview

The IR is the internal typed representation that sits between raw paper extraction
and spec emission. All extractors (LaTeX, PDF, HTML) produce IR entities. All emitters
(YAML, Markdown) consume them. The IR enforces type safety and completeness checking.

## Design Principles

1. **Everything is typed** — Pydantic models enforce field types at construction time
2. **UNRESOLVED is explicit** — Fields use `Optional` with sentinel values, never silent defaults
3. **Provenance is mandatory** — Every extracted value carries a `source` trace
4. **Informed-POMDP is first-class** — Training-only vs execution fields are structurally distinct

## Core Entity Hierarchy

```
SpecIR (root)
├── Meta
├── FrameConventions
├── Spaces
│   ├── ObservationExec: list[SpaceField]
│   ├── InformationTrain: list[InformedField]
│   ├── Action: ActionSpace
│   └── State: list[SpaceField]
├── Timing
├── Reward
│   └── terms: list[RewardTerm]
│       └── expression_ast: ExprNode
├── Termination
│   └── conditions: list[TerminationCondition]
│       └── condition_ast: BoolNode
├── GatesTrack (optional, domain-specific)
├── Dynamics
│   ├── equations: list[Equation]
│   ├── parameters: list[DynamicsParam]
│   └── domain_randomization: list[DomainRandEntry]
├── Perception
│   └── augmentations: list[Augmentation]
├── WorldModel
│   └── components: dict[str, ModelComponent]
├── ActorCritic
│   └── regularizers: list[Regularizer]
├── Training
│   ├── replay: ReplayConfig
│   ├── schedule: TrainingSchedule
│   └── optimizer: OptimizerConfig
├── Evaluation
│   └── success_criteria: list[Criterion]
├── Deployment
│   └── runtime_budgets: list[RuntimeBudget]
└── Imports: list[BaselineImport]
```

## Base Types

### Source Trace

Every extracted value carries provenance:

```python
class SourceTrace(BaseModel):
    """Traces a value back to its paper location."""
    section: str | None = None      # "§3.2", "§4.1"
    table: str | None = None        # "Table 1", "Table 3"
    equation: str | None = None     # "Eq. 4", "Eq. 12"
    figure: str | None = None       # "Fig. 2 caption"
    page: int | None = None         # PDF page number
    tex_file: str | None = None     # LaTeX source file
    line_range: tuple[int, int] | None = None  # Line range in TeX
    raw_text: str | None = None     # Original extracted text
```

### Resolvable Fields

Fields that may be UNRESOLVED use a union type:

```python
T = TypeVar('T')
Resolvable = T | Literal["UNRESOLVED"]
```

A `Resolvable[float]` is either a `float` or the string `"UNRESOLVED"`.

### Expression AST Nodes

For reward expressions and dynamics equations:

```python
class ExprNode(BaseModel):
    """AST node for mathematical expressions."""
    type: Literal["literal", "field", "op", "func"]

    # For literal
    value: float | None = None

    # For field reference
    name: str | None = None
    space: str | None = None    # "state", "observation", "action"

    # For operations
    op: str | None = None       # "add", "multiply", "clamp", "min", "max"
    args: list['ExprNode'] = []
    params: dict[str, float] = {}  # e.g., {"min": -1.0, "max": 1.0}

    # For function calls
    func_name: str | None = None
```

### Boolean AST Nodes

For termination conditions:

```python
class BoolNode(BaseModel):
    """AST node for boolean conditions."""
    type: Literal["compare", "logic", "field_check"]

    # For comparisons
    op: str | None = None       # "lt", "gt", "le", "ge", "eq", "ne"
    left: 'BoolNode | ExprNode | None' = None
    right: 'BoolNode | ExprNode | None' = None

    # For logical connectives
    logic_op: str | None = None  # "and", "or", "not"
    children: list['BoolNode'] = []
```

## Entity Details

### SpaceField

```python
class SpaceField(BaseModel):
    name: str
    dtype: str                    # "float32", "bool", "int64"
    shape: list[int | str]        # int for concrete, str for symbolic
    units: str | None = None
    source: SourceTrace
```

### InformedField (extends SpaceField)

```python
class InformedField(SpaceField):
    """Privileged information available only during training."""
    informed_dreamer_key: str | None = None  # Regex for decoder gating
    training_only: bool = True
```

### RewardTerm

```python
class RewardTerm(BaseModel):
    name: str
    expression_ast: Resolvable[ExprNode]
    weight: Resolvable[float]
    clamp: tuple[float, float] | None = None
    zeroing_window: str | None = None   # Condition string
    source: SourceTrace
```

### DynamicsParam

```python
class DynamicsParam(BaseModel):
    name: str
    symbol: str                   # LaTeX symbol
    default_value: Resolvable[float]
    units: str
    source: SourceTrace
```

### DomainRandEntry

```python
class DomainRandEntry(BaseModel):
    parameter: str                # Reference to DynamicsParam.name
    distribution: str             # "uniform", "normal", "log_uniform"
    range: tuple[float, float]
    resample_frequency: str       # "per_episode", "per_step", "fixed"
    source: SourceTrace
```

## Validation Methods

The root `SpecIR` model provides these validation methods:

### count_unresolved()

Walk all fields recursively. Count any field whose value is `"UNRESOLVED"`.
The count must match `meta.unresolved_count`.

### validate_sources()

Ensure every `SourceTrace` has at least one non-None location field
(section, table, equation, figure, or page).

### validate_informed_split()

Check that `information_train` fields do NOT appear in `observation_exec`.
This enforces the informed-POMDP invariant.

### validate_shapes()

For every space field, verify that symbolic dimensions (strings) are
defined elsewhere in the spec (e.g., as a dynamics parameter count).

### flatten_imports(baseline_specs: dict)

Given resolved baseline specs, merge fields following override rules:
1. This spec's explicit values override baseline values
2. Baseline values fill in for fields not mentioned in this spec
3. Conflicts are flagged as warnings
4. Result is written to `spec.resolved.yaml`

## Normalization

### Canonical Naming

Apply consistent naming conventions:
- Snake_case for all field names
- SI units where applicable (meters, seconds, radians)
- NED convention for coordinate frames unless spec explicitly states otherwise

### Unit Normalization

Convert all physical quantities to base SI units:
- Distances: meters (not cm, mm, inches)
- Angles: radians (not degrees)
- Time: seconds (not milliseconds, unless explicitly a delay spec)
- Frequencies: Hz

### Frame Annotations

Every vector field should annotate which frame it's expressed in:
- `p_w` → position in world frame
- `v_b` → velocity in body frame
- `omega_b` → angular velocity in body frame
