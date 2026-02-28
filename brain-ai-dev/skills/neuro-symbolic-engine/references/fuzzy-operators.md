# Fuzzy Operators Reference

Mathematical foundations and implementation details for the differentiable fuzzy logic
operators used in the Neuro-Symbolic Engine. All operators align with Logic Tensor Networks
(LTN) / Real Logic semantics: symbols are grounded as neural computational graphs,
connectives use configurable t-norm families, and training maximizes satisfiability of a
knowledge base expressed as formulas in many-valued logic over the unit interval [0, 1].

Target implementation file: `brain_ai/reasoning/fuzzy_operators.py`

---

## 1. T-Norm Families (AND Operators)

A **t-norm** T: [0,1] x [0,1] -> [0,1] is a binary operation that generalizes classical
conjunction. It must satisfy commutativity, associativity, monotonicity, and
T(x, 1) = x (identity element 1).

All three families below are continuous t-norms and satisfy these axioms exactly.

### 1.1 Godel / Minimum T-Norm

**Formula:**

```
T_G(x, y) = min(x, y)
```

**Gradient behavior:**

The minimum operation routes gradient to exactly one operand. Taking the subgradient:

```
dT_G/dx = 1  if x < y
dT_G/dx = 0  if x > y
dT_G/dx undefined (convention: 0.5) if x = y
```

This produces **sparse gradients**: only the operand with the smaller value receives a
gradient signal. In multi-argument chains, only the single weakest link trains per step.

**When to use:**

- Crisp-ish logic where the "weakest link" interpretation is desired.
- Evaluation/verification mode where exact classical-like boundary behavior matters.
- NOT recommended for training deep chains due to single-path gradient flow.

**PyTorch implementation:**

```python
def t_norm_godel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Godel (minimum) t-norm. T(x,y) = min(x,y)."""
    return torch.min(x, y)
```

### 1.2 Product T-Norm

**Formula:**

```
T_P(x, y) = x * y
```

**Gradient behavior:**

```
dT_P/dx = y
dT_P/dy = x
```

Both operands receive gradient simultaneously, proportional to the other operand's value.
This is **smooth and dense** but **vanishes when either operand approaches 0**. In long
conjunctive chains, truth values compound multiplicatively toward zero, starving upstream
parameters of gradient signal.

**When to use:**

- Mathematically clean semantics needed (e.g., probabilistic interpretation).
- Short formulas with few conjuncts.
- Must pair with eps-projection (stable_product) when truth values regularly approach 0.

**PyTorch implementation:**

```python
def t_norm_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Product t-norm. T(x,y) = x * y."""
    return x * y
```

### 1.3 Lukasiewicz T-Norm

**Formula:**

```
T_L(x, y) = max(0, x + y - 1)
```

**Gradient behavior:**

```
dT_L/dx = 1  if x + y > 1
dT_L/dx = 0  if x + y < 1
```

This is **piecewise linear**. When the sum exceeds 1, both operands receive full unit
gradient. When below the threshold, both receive zero gradient. The transition is sharp,
producing a **sparse activation pattern** similar to ReLU.

**When to use:**

- Sparse activation behavior is desired (analogous to ReLU gating in neural nets).
- The "strong conjunction" interpretation is appropriate: T_L(0.6, 0.6) = 0.2, much
  lower than T_P(0.6, 0.6) = 0.36.
- Caution: large portions of the input space may produce zero output and zero gradient.

**PyTorch implementation:**

```python
def t_norm_lukasiewicz(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Lukasiewicz t-norm. T(x,y) = max(0, x+y-1)."""
    return torch.clamp(x + y - 1, min=0.0)
```

---

## 2. T-Conorm Families (OR Operators)

A **t-conorm** (s-norm) S: [0,1] x [0,1] -> [0,1] is dual to a t-norm with respect to
the standard negation N(x) = 1 - x. The duality is:

```
S(x, y) = 1 - T(1-x, 1-y)
```

Each t-norm family induces a corresponding t-conorm. The identity element is 0:
S(x, 0) = x.

### 2.1 Maximum (Godel) T-Conorm

**Formula:**

```
S_G(x, y) = max(x, y)
```

Dual to the minimum t-norm. Routes gradient to the single operand with the higher value.

**PyTorch implementation:**

```python
def t_conorm_max(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Maximum (Godel) t-conorm. S(x,y) = max(x,y)."""
    return torch.max(x, y)
```

### 2.2 Probabilistic Sum (Product) T-Conorm

**Formula:**

```
S_P(x, y) = x + y - x * y
```

Dual to the product t-norm. Equivalent to 1 - (1-x)(1-y), the inclusion-exclusion formula
for independent events. Smooth gradients:

```
dS_P/dx = 1 - y
dS_P/dy = 1 - x
```

Gradients vanish as the OTHER operand approaches 1, which is less problematic than the
product t-norm's vanishing near 0 (disjunction of strong claims is already near 1).

**PyTorch implementation:**

```python
def t_conorm_prob_sum(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Probabilistic sum t-conorm. S(x,y) = x + y - x*y."""
    return x + y - x * y
```

### 2.3 Lukasiewicz T-Conorm

**Formula:**

```
S_L(x, y) = min(1, x + y)
```

Dual to the Lukasiewicz t-norm. Piecewise linear with unit gradient to both operands when
x + y < 1, zero gradient when saturated at 1.

**PyTorch implementation:**

```python
def t_conorm_lukasiewicz(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Lukasiewicz t-conorm. S(x,y) = min(1, x+y)."""
    return torch.clamp(x + y, max=1.0)
```

---

## 3. Negation

### Standard (Strong) Negation

**Formula:**

```
N(x) = 1 - x
```

**Properties:**

| Property | Condition | Standard negation |
|---|---|---|
| Boundary conditions | N(0) = 1, N(1) = 0 | Satisfied |
| Strict monotonicity | x < y implies N(x) > N(y) | Satisfied |
| Involution | N(N(x)) = x | Satisfied: 1 - (1 - x) = x |
| Continuity | N is continuous | Satisfied |

The involution property is critical: it ensures double negation cancels, which is required
for the De Morgan duality between t-norms and t-conorms:

```
T(x, y) = N(S(N(x), N(y)))
S(x, y) = N(T(N(x), N(y)))
```

**Gradient:** dN/dx = -1 (constant, full gradient flow everywhere).

**PyTorch implementation:**

```python
def negation_standard(x: torch.Tensor) -> torch.Tensor:
    """Standard (strong) negation. N(x) = 1 - x."""
    return 1.0 - x
```

---

## 4. Implication Operators

Fuzzy implication I: [0,1] x [0,1] -> [0,1] generalizes material implication. Two major
classes exist.

### 4.1 S-Implications (Material Style)

Derived from a t-conorm and negation: **I(x, y) = S(N(x), y)**.

S-implications satisfy I(0, y) = 1 and I(1, y) = y. They do NOT generally satisfy the
**modus ponens** condition T(x, I(x,y)) <= y for all x, y.

#### Reichenbach Implication (from Product / Prob-Sum)

```
I_RC(x, y) = S_P(N(x), y) = (1 - x) + (1 - x)*y...
            = 1 - x + x*y
```

Derivation: S_P(1-x, y) = (1-x) + y - (1-x)*y = 1 - x + y - y + xy = 1 - x + xy.

**Gradient:**

```
dI_RC/dx = -1 + y = -(1 - y)
dI_RC/dy = x
```

Gradient with respect to the antecedent x is always non-positive (increasing x decreases
implication truth), and gradient with respect to the consequent y is proportional to x.
Both are smooth and non-vanishing for typical values.

#### Kleene-Dienes Implication (from Maximum)

```
I_KD(x, y) = S_G(N(x), y) = max(1 - x, y)
```

Sparse gradient (only one of N(x), y receives gradient). Not commonly used in LTN training
due to gradient sparsity.

#### Lukasiewicz S-Implication

```
I_SL(x, y) = S_L(N(x), y) = min(1, (1-x) + y) = min(1, 1 - x + y)
```

This coincides with the Lukasiewicz residuum (see below), making Lukasiewicz the only
family where the S-implication equals the residuated implication.

### 4.2 Residuated Implications (R-Implications)

Derived from a t-norm T as the **residuum**: the largest z such that T(x, z) <= y.

```
I_T(x, y) = sup{ z in [0,1] : T(x, z) <= y }
```

Residuated implications always satisfy modus ponens: T(x, I_T(x,y)) <= y.

#### Godel Residuum

```
I_G(x, y) = 1    if x <= y
           = y    if x > y
```

Discontinuous at x = y. The subgradient:

```
dI_G/dx = 0    (everywhere, subgradient)
dI_G/dy = 0    if x <= y
dI_G/dy = 1    if x > y
```

Very sparse gradients. Only the consequent trains, and only when violated.

**PyTorch implementation:**

```python
def implies_godel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Godel residuated implication. I(x,y) = 1 if x<=y else y."""
    return torch.where(x <= y, torch.ones_like(x), y)
```

#### Goguen (Product) Residuum

```
I_Go(x, y) = 1      if x <= y
            = y / x  if x > y
```

Smooth when x > y, but requires guarding against division by zero.

**Gradient (when x > y):**

```
dI_Go/dx = -y / x^2    (negative, increasing antecedent reduces implication)
dI_Go/dy = 1 / x       (positive, scales inversely with antecedent strength)
```

**PyTorch implementation:**

```python
def implies_goguen(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Goguen (product) residuated implication. I(x,y) = 1 if x<=y else y/x."""
    return torch.where(x <= y, torch.ones_like(x), y / x.clamp(min=eps))
```

#### Lukasiewicz Residuum

```
I_L(x, y) = min(1, 1 - x + y)
```

Identical to the Lukasiewicz S-implication. Piecewise linear with unit gradients in the
active region.

**PyTorch implementation:**

```python
def implies_lukasiewicz(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Lukasiewicz residuated implication. I(x,y) = min(1, 1-x+y)."""
    return torch.clamp(1.0 - x + y, max=1.0)
```

### 4.3 Boundary Conditions

All well-formed fuzzy implications must satisfy:

| Condition | Value | Intuition |
|---|---|---|
| I(0, 0) | 1 | False implies false is true |
| I(0, 1) | 1 | False implies anything is true |
| I(1, 0) | 0 | True implies false is false |
| I(1, 1) | 1 | True implies true is true |
| I(1, y) | y | If antecedent is certain, implication reduces to consequent |
| I(0, y) | 1 | If antecedent is impossible, implication is vacuously true |

### 4.4 Implication Class Selection per LTN

LTN/Real Logic recommends:

- Use **S-implications** (especially Reichenbach) when smooth gradient flow is the priority
  and strict modus ponens is not required.
- Use **residuated implications** when the logical semantics must satisfy the adjunction
  property T(a, I(a,b)) <= b, e.g., for formal verification tasks.
- The **Reichenbach implication** (I(x,y) = 1 - x + xy) is the default in the
  `stable_product` bundle because it provides smooth non-vanishing gradients compatible
  with product semantics.

---

## 5. Quantifiers as Differentiable Aggregators

Quantifiers aggregate truth values over a set of variable groundings. In classical logic,
FORALL is the infimum (min) and EXISTS is the supremum (max) over the domain. In
differentiable fuzzy logic, smooth approximations replace these hard operations.

### 5.1 Universal Quantifier (FORALL)

#### pMeanError Aggregation

The LTN approach uses **pMeanError**: compute the generalized mean of the **errors**
(1 - truth), then invert. This ensures the aggregation tends toward the minimum truth
value as p increases.

**Formula:**

```
pMeanError_p(v_1, ..., v_n) = 1 - ( (1/n) * sum_i (1 - v_i)^p )^(1/p)
```

where v_i are the individual truth values and p >= 1 is the exponent.

**Behavior by p:**

| p value | Behavior | Approximates |
|---|---|---|
| p = 1 | Arithmetic mean of errors (1 - mean(v)) | Arithmetic mean |
| p = 2 | Root-mean-square of errors | Penalizes large violations |
| p -> infinity | Largest error dominates | min(v_i) |

**Gradient of pMeanError with respect to v_i:**

```
d(pMeanError)/dv_i = ( (1-v_i)^(p-1) ) / ( n * M^(p-1) )

where M = ( (1/n) sum_j (1-v_j)^p )^(1/p)
```

As p increases, gradient concentrates on the most-violated instance (smallest v_i),
providing a differentiable approximation to "fix the worst violator first."

**PyTorch implementation:**

```python
def forall_pMeanError(
    truth_values: torch.Tensor,
    p: float = 2.0,
    dim: int = -1,
    eps: float = 1e-8,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Universal quantifier via pMeanError aggregation.

    Args:
        truth_values: Truth values in [0,1], shape (..., N, ...).
        p: Generalized mean exponent (>= 1). Higher p -> harder min.
        dim: Dimension to aggregate over (the variable domain).
        eps: Numerical stability epsilon.
        mask: Boolean mask for variable-length instance sets.
               True = valid instance, False = padded (ignored).

    Returns:
        Aggregated truth value in [0,1].
    """
    errors = 1.0 - truth_values  # in [0,1]
    if mask is not None:
        errors = errors.masked_fill(~mask, 0.0)
        n = mask.float().sum(dim=dim, keepdim=True).clamp(min=1.0)
    else:
        n = truth_values.shape[dim]

    mean_error_p = (errors.pow(p).sum(dim=dim, keepdim=True) / n)
    mean_error = mean_error_p.pow(1.0 / p).squeeze(dim)
    return (1.0 - mean_error).clamp(min=0.0, max=1.0)
```

#### LogSumExp Variant for Numerical Stability

When p is large, computing (1 - v_i)^p can overflow or underflow. Use the log-domain
trick:

```python
def forall_lse(
    truth_values: torch.Tensor,
    p: float = 10.0,
    dim: int = -1,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """FORALL via LogSumExp for large p (numerically stable)."""
    errors = 1.0 - truth_values
    log_errors = errors.clamp(min=1e-8).log() * p
    if mask is not None:
        log_errors = log_errors.masked_fill(~mask, -float('inf'))
        n = mask.float().sum(dim=dim, keepdim=True).clamp(min=1.0)
    else:
        n = truth_values.shape[dim]
    lse = torch.logsumexp(log_errors, dim=dim) - math.log(n)
    mean_error = (lse / p).exp()
    return (1.0 - mean_error).clamp(0.0, 1.0)
```

### 5.2 Existential Quantifier (EXISTS)

EXISTS is the dual of FORALL: EXISTS(v) = N(FORALL(N(v))). Alternatively, apply the
generalized mean directly to truth values with a high exponent to approximate max.

#### Generalized Mean (Direct)

```
pMean_p(v_1, ..., v_n) = ( (1/n) * sum_i v_i^p )^(1/p)
```

For p >> 1, this approaches max(v_i). For p = 1, it is the arithmetic mean.

**PyTorch implementation:**

```python
def exists_pMean(
    truth_values: torch.Tensor,
    p: float = 6.0,
    dim: int = -1,
    eps: float = 1e-8,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Existential quantifier via generalized p-mean.

    Args:
        truth_values: Truth values in [0,1].
        p: Exponent. Higher p -> harder max approximation.
        dim: Dimension to aggregate over.
        eps: Numerical stability epsilon.
        mask: Boolean mask for variable-length sets.

    Returns:
        Aggregated truth value in [0,1].
    """
    vals = truth_values.clamp(min=eps)
    if mask is not None:
        vals = vals.masked_fill(~mask, 0.0)
        n = mask.float().sum(dim=dim, keepdim=True).clamp(min=1.0)
    else:
        n = truth_values.shape[dim]
    mean_p = (vals.pow(p).sum(dim=dim, keepdim=True) / n).pow(1.0 / p)
    return mean_p.squeeze(dim).clamp(0.0, 1.0)
```

#### Softmax Pooling Alternative

For cases where the generalized mean's gradient scaling is too aggressive:

```python
def exists_softmax(
    truth_values: torch.Tensor,
    temperature: float = 0.1,
    dim: int = -1,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """EXISTS via softmax-weighted pooling."""
    weights = truth_values / temperature
    if mask is not None:
        weights = weights.masked_fill(~mask, -float('inf'))
    attention = torch.softmax(weights, dim=dim)
    return (attention * truth_values).sum(dim=dim)
```

### 5.3 Parameters and Tuning

| Parameter | Typical Range | Effect |
|---|---|---|
| `p` (FORALL) | 2 -- 10 | Low p = lenient (mean-like), high p = strict (min-like) |
| `p` (EXISTS) | 4 -- 20 | Low p = lenient (mean-like), high p = strict (max-like) |
| `temperature` | 0.05 -- 1.0 | Lower temperature = sharper softmax pooling |

### 5.4 Stability Considerations

**Gradient behavior as p -> infinity:**

The generalized mean gradient with respect to a single element v_i scales as:

```
dM_p/dv_i ~ (v_i / M_p)^(p-1) / n
```

For the dominant element (v_i ~ M_p), the gradient is approximately 1/n. For non-dominant
elements, the factor (v_i / M_p)^(p-1) decays exponentially. At very large p, this causes
effective gradient sparsity and potential training instability.

**Overflow prevention:**

- For p > 10, use the LogSumExp formulation.
- Clamp truth values away from exact 0 before raising to power p.
- Apply gradient clipping to quantifier outputs when using p > 5.

**Masking for variable-length sets:**

Replace invalid positions with neutral values (0 for errors in FORALL, 0 for values in
EXISTS) and normalize the count n by the number of valid elements.

---

## 6. Operator Bundles

An **operator bundle** is a complete, internally consistent set of fuzzy connectives:
AND, OR, NOT, IMPLIES, FORALL, EXISTS. The implementation selects a bundle by name from
the `OperatorConfig.bundle` field.

### 6.1 Bundle Definitions

#### `godel` Bundle

| Operator | Definition |
|---|---|
| AND | T_G(x, y) = min(x, y) |
| OR | S_G(x, y) = max(x, y) |
| NOT | N(x) = 1 - x |
| IMPLIES | I_G(x, y) = 1 if x <= y else y (Godel residuum) |
| FORALL | pMeanError with p = quantifier_p |
| EXISTS | pMean with p = quantifier_p |

#### `product` Bundle

| Operator | Definition |
|---|---|
| AND | T_P(x, y) = x * y |
| OR | S_P(x, y) = x + y - x * y |
| NOT | N(x) = 1 - x |
| IMPLIES | I_RC(x, y) = 1 - x + x * y (Reichenbach) |
| FORALL | pMeanError with p = quantifier_p |
| EXISTS | pMean with p = quantifier_p |

Note: The `product` bundle uses the Reichenbach S-implication by default. Switch to the
Goguen residuum via `OperatorConfig.implication_type = "goguen"`.

#### `lukasiewicz` Bundle

| Operator | Definition |
|---|---|
| AND | T_L(x, y) = max(0, x + y - 1) |
| OR | S_L(x, y) = min(1, x + y) |
| NOT | N(x) = 1 - x |
| IMPLIES | I_L(x, y) = min(1, 1 - x + y) (Lukasiewicz residuum) |
| FORALL | pMeanError with p = quantifier_p |
| EXISTS | pMean with p = quantifier_p |

#### `stable_product` Bundle

| Operator | Definition |
|---|---|
| AND | pi_0(x) * pi_0(y) |
| OR | 1 - (1 - pi_1(x)) * (1 - pi_1(y)) |
| NOT | 1 - x |
| IMPLIES | 1 - pi_0(x) + pi_0(x) * pi_1(y) (stable Reichenbach) |
| FORALL | pMeanError over pi_0(values) |
| EXISTS | pMean over pi_1(values) |

Where pi_0 and pi_1 are the stability projections defined in Section 7.

### 6.2 Bundle Selection at Runtime

```python
from dataclasses import dataclass
from typing import Callable, NamedTuple

class OperatorBundle(NamedTuple):
    """Complete set of fuzzy operators."""
    AND: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    OR: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    NOT: Callable[[torch.Tensor], torch.Tensor]
    IMPLIES: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    FORALL: Callable[[torch.Tensor], torch.Tensor]
    EXISTS: Callable[[torch.Tensor], torch.Tensor]
    name: str

# Registry
OPERATOR_BUNDLES: Dict[str, OperatorBundle] = {}

def register_bundle(name: str, bundle: OperatorBundle) -> None:
    """Register an operator bundle by name."""
    OPERATOR_BUNDLES[name] = bundle

def get_bundle(name: str) -> OperatorBundle:
    """Retrieve a registered operator bundle."""
    if name not in OPERATOR_BUNDLES:
        raise ValueError(
            f"Unknown operator bundle '{name}'. "
            f"Available: {list(OPERATOR_BUNDLES.keys())}"
        )
    return OPERATOR_BUNDLES[name]
```

---

## 7. Stable Product Semantics (Critical Section)

The `stable_product` bundle is the **default for training**. It modifies the pure product
semantics with epsilon-projections that prevent exact zero or exact one truth values,
guaranteeing non-vanishing gradients throughout the formula evaluation graph.

### 7.1 Projection Functions

**pi_0 projection** (avoid exact zeros):

```
pi_0(x) = clamp(x, eps, 1.0)
```

Applied before any operation that would suffer from zero-multiplication (AND, FORALL).

**pi_1 projection** (avoid exact ones):

```
pi_1(x) = clamp(x, 0.0, 1.0 - eps)
```

Applied before any operation that would suffer from (1 - x) = 0, i.e., OR computation
via De Morgan or EXISTS.

**PyTorch implementation:**

```python
def pi_0(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Project away from zero: clamp(x, eps, 1)."""
    return x.clamp(min=eps, max=1.0)

def pi_1(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Project away from one: clamp(x, 0, 1-eps)."""
    return x.clamp(min=0.0, max=1.0 - eps)
```

### 7.2 Stable Operator Definitions

Apply projections immediately before each operator:

**Stable AND:**

```python
def stable_and(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    return pi_0(x, eps) * pi_0(y, eps)
```

**Stable OR:**

```python
def stable_or(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    nx = 1.0 - pi_1(x, eps)  # guaranteed >= eps
    ny = 1.0 - pi_1(y, eps)  # guaranteed >= eps
    return 1.0 - nx * ny
```

**Stable NOT:**

```
stable_not(x) = 1 - x
```

No projection needed for negation since 1 - x does not suffer from zero/one singularities.

**Stable IMPLIES (Reichenbach):**

```python
def stable_implies(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    x_proj = pi_0(x, eps)
    y_proj = pi_1(y, eps)
    return 1.0 - x_proj + x_proj * y_proj
```

Rationale: the antecedent x enters as a multiplicative factor (needs pi_0 to prevent
zero multiplication). The consequent y enters in a (1 - x + xy) form where large y
could push the result above 1 only in edge cases, but the pi_1 projection ensures the
product term xy does not collapse gradient from the OR-like structure.

**Stable FORALL:**

```python
def stable_forall(
    truth_values: torch.Tensor, p: float = 2.0, dim: int = -1, eps: float = 1e-4
) -> torch.Tensor:
    projected = pi_0(truth_values, eps)
    return forall_pMeanError(projected, p=p, dim=dim, eps=eps)
```

**Stable EXISTS:**

```python
def stable_exists(
    truth_values: torch.Tensor, p: float = 6.0, dim: int = -1, eps: float = 1e-4
) -> torch.Tensor:
    projected = pi_1(truth_values, eps)
    return exists_pMean(projected, p=p, dim=dim, eps=eps)
```

### 7.3 Semantic Caveat

The `stable_product` bundle is **NOT a true t-norm**. The identity axiom T(a, 1) = a is
violated:

```
stable_and(a, 1.0, eps=1e-4)
  = pi_0(a) * pi_0(1.0)
  = clamp(a, 1e-4, 1.0) * 1.0
  = clamp(a, 1e-4, 1.0)
  != a  when a < 1e-4
```

Similarly, the OR identity S(a, 0) = a is approximate:

```
stable_or(a, 0.0, eps=1e-4)
  = 1 - (1 - pi_1(a)) * (1 - pi_1(0))
  = 1 - (1 - clamp(a, 0, 1-eps)) * (1 - 0)
  = clamp(a, 0, 1-eps)
  != a  when a > 1-eps
```

**Impact on testing strategy:**

- Identity tests (AND(x, 1) = x, OR(x, 0) = x) MUST use `atol=eps` tolerance for
  `stable_product`.
- Pure bundle identity tests (`godel`, `product`, `lukasiewicz`) MAY use exact equality.
- Always document which bundle is under test and the expected tolerance.

### 7.4 When to Use

| Scenario | Recommended Bundle |
|---|---|
| Training (any phase) | `stable_product` |
| Evaluation / verification | `product` or `godel` |
| Formal identity checking | Pure bundles only |
| Long conjunctive chains (> 5 terms) | `stable_product` (mandatory) |
| Sparse logic (threshold-like) | `lukasiewicz` |

Switching strategy: train with `stable_product`, then optionally switch to `product` at
evaluation time for exact semantics. The truth values learned under stable semantics
transfer because the eps perturbation is negligible for well-trained predicates (whose
outputs are typically in [0.1, 0.9]).

---

## 8. Gradient Analysis

### 8.1 AND Operator Gradients

| Family | dT/dx | dT/dy | Gradient density | Failure mode |
|---|---|---|---|---|
| Product | y | x | Dense | Vanishes when either -> 0 |
| Godel | 1{x<y} | 1{y<x} | Sparse (one path) | No gradient to non-min operand |
| Lukasiewicz | 1{x+y>1} | 1{x+y>1} | Semi-sparse | Zero when x+y <= 1 |
| Stable product | pi_0(y) >= eps | pi_0(x) >= eps | Dense, bounded | Approximate identity |

### 8.2 OR Operator Gradients

| Family | dS/dx | dS/dy |
|---|---|---|
| Prob. sum | 1 - y | 1 - x |
| Maximum | 1{x>y} | 1{y>x} |
| Lukasiewicz | 1{x+y<1} | 1{x+y<1} |
| Stable prob. sum | 1 - pi_1(y) >= eps | 1 - pi_1(x) >= eps |

### 8.3 Implication Gradients

| Implication | dI/dx | dI/dy |
|---|---|---|
| Reichenbach | -(1 - y) | x |
| Godel residuum | 0 (subgradient) | 1{x > y} |
| Goguen (x > y) | -y/x^2 | 1/x |
| Lukasiewicz | -1{1-x+y < 1} | 1{1-x+y < 1} |

### 8.4 Quantifier Gradient Scaling

For FORALL via pMeanError, the gradient with respect to the i-th instance is:

```
dQ/dv_i = (1 - v_i)^(p-1) / (n * M_err^(p-1))
```

where M_err = pMeanError's internal mean-of-errors. As p increases:

- Gradient concentrates on the most-violated instance (smallest v_i).
- The gradient magnitude for the most-violated instance scales as ~1/n (bounded).
- Other instances receive exponentially decaying gradient.
- At p > 10, effectively only 1-3 instances receive meaningful gradient per step.

**Recommendation:** Use p = 2 for training (balanced gradient distribution). Increase to
p = 4-6 for fine-tuning when most constraints are already approximately satisfied and the
goal is to eliminate remaining violations.

### 8.5 Summary Recommendations

| Scenario | Recommended approach |
|---|---|
| General training | `stable_product` bundle, p = 2 for FORALL, p = 6 for EXISTS |
| Formal verification | Pure `product` or `godel` bundle |
| Long chains (> 5 conjuncts) | `stable_product` mandatory; consider reducing p |
| High violation rate (early training) | Low p (1-2) for broad gradient |
| Low violation rate (fine-tuning) | High p (4-10) to target remaining violators |

---

## 9. AMP Compatibility

Automatic Mixed Precision (AMP) uses fp16 for most computations. Fuzzy logic operators
are **sensitive to fp16 rounding** because:

1. Truth values near 0 or 1 lose precision in fp16 (smallest representable positive fp16
   is ~6e-8, but the precision near 0 is only ~6e-8, meaning eps = 1e-4 has only ~2 bits
   of mantissa).
2. Quantifier aggregations involve powers (v^p) and roots (M^(1/p)) that amplify
   rounding errors.
3. Stable product projections (eps = 1e-4) lose their protective effect if clamped
   values are rounded to 0 in fp16.

### 9.1 Required: fp32 Enforcement

All fuzzy operator functions MUST compute in fp32 even when AMP is active. Use PyTorch's
`custom_fwd` decorator:

```python
from torch.cuda.amp import custom_fwd, custom_bwd

class StableAndFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x, y, eps):
        x_p = x.clamp(min=eps, max=1.0)
        y_p = y.clamp(min=eps, max=1.0)
        result = x_p * y_p
        ctx.save_for_backward(x_p, y_p)
        return result

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        x_p, y_p = ctx.saved_tensors
        return grad_output * y_p, grad_output * x_p, None
```

### 9.2 Simpler Alternative: torch.float32 Context

For operators implemented as plain functions (not autograd.Function), wrap the entire
operator bundle evaluation in a float32 context:

```python
def evaluate_formula(formula_fn, inputs, operator_bundle):
    """Evaluate a formula with fp32 precision for operators."""
    with torch.cuda.amp.autocast(enabled=False):
        # Cast inputs to fp32 explicitly
        fp32_inputs = {k: v.float() for k, v in inputs.items()}
        result = formula_fn(fp32_inputs, operator_bundle)
    return result  # result is fp32; autocast will handle downstream
```

### 9.3 Quantifiers: Especially Sensitive

Quantifier aggregations over many instances (n > 100) with large p exponents compound
rounding errors multiplicatively. Always:

1. Cast truth values to fp32 before quantifier aggregation.
2. Use the LogSumExp variant for p > 5.
3. Verify quantifier outputs are in [0, 1] after computation (add a final clamp).

---

## 10. Code Patterns

### 10.1 Operator Bundle as a Dataclass of Callables

```python
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
import torch

@dataclass
class OperatorBundle:
    """A complete set of fuzzy logic operators."""
    name: str
    AND: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    OR: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    NOT: Callable[[torch.Tensor], torch.Tensor]
    IMPLIES: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    FORALL: Callable[[torch.Tensor, int], torch.Tensor]
    EXISTS: Callable[[torch.Tensor, int], torch.Tensor]

    def __post_init__(self):
        """Validate the bundle on construction."""
        x = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        y = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
        # Verify outputs are in [0,1]
        for op_name, op_fn in [("AND", self.AND), ("OR", self.OR)]:
            result = op_fn(x, y)
            assert (result >= 0).all() and (result <= 1).all(), (
                f"{self.name}.{op_name} produced out-of-range values"
            )
```

### 10.2 Registering Custom Bundles

```python
# Global registry
_BUNDLE_REGISTRY: Dict[str, OperatorBundle] = {}

def register_bundle(bundle: OperatorBundle) -> None:
    """Register an operator bundle in the global registry."""
    _BUNDLE_REGISTRY[bundle.name] = bundle

def get_bundle(name: str) -> OperatorBundle:
    """Retrieve a bundle by name. Raises ValueError if not found."""
    if name not in _BUNDLE_REGISTRY:
        available = ", ".join(sorted(_BUNDLE_REGISTRY.keys()))
        raise ValueError(f"Unknown bundle '{name}'. Available: {available}")
    return _BUNDLE_REGISTRY[name]

# Register built-in bundles at module load time
def _build_product_bundle(eps: float = 0.0) -> OperatorBundle:
    """Build a product bundle, optionally with eps projections."""
    if eps > 0:
        _and = lambda x, y: pi_0(x, eps) * pi_0(y, eps)
        _or  = lambda x, y: 1.0 - (1.0 - pi_1(x, eps)) * (1.0 - pi_1(y, eps))
        _imp = lambda x, y: 1.0 - pi_0(x, eps) + pi_0(x, eps) * pi_1(y, eps)
        name = "stable_product"
    else:
        _and = lambda x, y: x * y
        _or  = lambda x, y: x + y - x * y
        _imp = lambda x, y: 1.0 - x + x * y
        name = "product"

    return OperatorBundle(
        name=name,
        AND=_and,
        OR=_or,
        NOT=lambda x: 1.0 - x,
        IMPLIES=_imp,
        FORALL=lambda v, dim=-1: forall_pMeanError(v, p=2.0, dim=dim),
        EXISTS=lambda v, dim=-1: exists_pMean(v, p=6.0, dim=dim),
    )

register_bundle(_build_product_bundle(eps=0.0))
register_bundle(_build_product_bundle(eps=1e-4))

register_bundle(OperatorBundle(
    name="godel",
    AND=lambda x, y: torch.min(x, y),
    OR=lambda x, y: torch.max(x, y),
    NOT=lambda x: 1.0 - x,
    IMPLIES=lambda x, y: torch.where(x <= y, torch.ones_like(x), y),
    FORALL=lambda v, dim=-1: forall_pMeanError(v, p=2.0, dim=dim),
    EXISTS=lambda v, dim=-1: exists_pMean(v, p=6.0, dim=dim),
))

register_bundle(OperatorBundle(
    name="lukasiewicz",
    AND=lambda x, y: torch.clamp(x + y - 1.0, min=0.0),
    OR=lambda x, y: torch.clamp(x + y, max=1.0),
    NOT=lambda x: 1.0 - x,
    IMPLIES=lambda x, y: torch.clamp(1.0 - x + y, max=1.0),
    FORALL=lambda v, dim=-1: forall_pMeanError(v, p=2.0, dim=dim),
    EXISTS=lambda v, dim=-1: exists_pMean(v, p=6.0, dim=dim),
))
```

### 10.3 Switching Bundles at Runtime via Config

```python
@dataclass
class OperatorConfig:
    bundle: str = "stable_product"
    eps: float = 1e-4
    quantifier_p: float = 2.0
    quantifier_temp: float = 1.0
    implication_type: str = "reichenbach"

class FuzzyOperatorModule(nn.Module):
    """Module that wraps an operator bundle for use in nn.Module graphs."""

    def __init__(self, config: OperatorConfig):
        super().__init__()
        self.config = config
        self._bundle = get_bundle(config.bundle)

    @property
    def bundle(self) -> OperatorBundle:
        return self._bundle

    def switch_bundle(self, name: str) -> None:
        """Switch operator bundle at runtime (e.g., train -> eval)."""
        self._bundle = get_bundle(name)

    def AND(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            return self._bundle.AND(x.float(), y.float())

    def OR(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            return self._bundle.OR(x.float(), y.float())

    def NOT(self, x: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            return self._bundle.NOT(x.float())

    def IMPLIES(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            return self._bundle.IMPLIES(x.float(), y.float())

    def FORALL(self, v: torch.Tensor, dim: int = -1) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            return self._bundle.FORALL(v.float(), dim)

    def EXISTS(self, v: torch.Tensor, dim: int = -1) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            return self._bundle.EXISTS(v.float(), dim)
```

### 10.4 Composing Operators for Complex Formulas

Example: evaluate the formula FORALL x. (Cat(x) IMPLIES EXISTS y. (Mouse(y) AND Chases(x,y)))

```python
def evaluate_cat_chases_mouse(
    ops: FuzzyOperatorModule,
    cat_truth: torch.Tensor,    # (N,) truth of Cat(x) for each entity
    mouse_truth: torch.Tensor,  # (M,) truth of Mouse(y) for each entity
    chases_truth: torch.Tensor, # (N, M) truth of Chases(x,y)
) -> torch.Tensor:
    """
    Evaluate: FORALL x. (Cat(x) -> EXISTS y. (Mouse(y) AND Chases(x,y)))

    Args:
        cat_truth: Unary predicate truth values for Cat, shape (N,).
        mouse_truth: Unary predicate truth values for Mouse, shape (M,).
        chases_truth: Binary predicate truth values for Chases, shape (N, M).

    Returns:
        Scalar truth value in [0,1].
    """
    # For each x: Mouse(y) AND Chases(x,y) for all y -> shape (N, M)
    mouse_and_chases = ops.AND(
        mouse_truth.unsqueeze(0).expand_as(chases_truth),
        chases_truth,
    )

    # For each x: EXISTS y. (Mouse(y) AND Chases(x,y)) -> shape (N,)
    exists_prey = ops.EXISTS(mouse_and_chases, dim=-1)

    # Cat(x) -> EXISTS y. (...) for each x -> shape (N,)
    implication = ops.IMPLIES(cat_truth, exists_prey)

    # FORALL x. (...) -> scalar
    return ops.FORALL(implication, dim=-1)
```

---

## Appendix A: Operator Comparison Table

All families evaluated at representative input pairs.

| x | y | Product AND | Godel AND | Luk. AND | Product OR | Godel OR | Luk. OR |
|---|---|---|---|---|---|---|---|
| 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 |
| 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 0.8 | 0.6 | 0.48 | 0.6 | 0.4 | 0.92 | 0.8 | 1.0 |
| 0.5 | 0.5 | 0.25 | 0.5 | 0.0 | 0.75 | 0.5 | 1.0 |
| 0.3 | 0.2 | 0.06 | 0.2 | 0.0 | 0.44 | 0.3 | 0.5 |
| 0.9 | 0.9 | 0.81 | 0.9 | 0.8 | 0.99 | 0.9 | 1.0 |
| 0.1 | 0.1 | 0.01 | 0.1 | 0.0 | 0.19 | 0.1 | 0.2 |

**Implication comparison:**

| x | y | Reichenbach | Godel | Goguen | Lukasiewicz | Kleene-Dienes |
|---|---|---|---|---|---|---|
| 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 0.8 | 0.6 | 0.68 | 0.6 | 0.75 | 0.8 | 0.6 |
| 0.5 | 0.5 | 0.75 | 1.0 | 1.0 | 1.0 | 0.5 |
| 0.9 | 0.1 | 0.19 | 0.1 | 0.111 | 0.2 | 0.1 |

---

## Appendix B: Known LTN Operator Identities per Bundle

These identities MUST hold for pure (non-stable) bundles. For `stable_product`, they hold
within eps tolerance.

### Universal Identities (all bundles)

| Identity | Formula | Expected |
|---|---|---|
| AND identity | T(x, 1) | x |
| AND annihilator | T(x, 0) | 0 |
| OR identity | S(x, 0) | x |
| OR annihilator | S(x, 1) | 1 |
| NOT involution | N(N(x)) | x |
| NOT boundary 0 | N(0) | 1 |
| NOT boundary 1 | N(1) | 0 |
| IMPLIES boundary | I(1, y) | y |
| IMPLIES vacuous | I(0, y) | 1 |
| De Morgan 1 | N(T(x,y)) | S(N(x), N(y)) |
| De Morgan 2 | N(S(x,y)) | T(N(x), N(y)) |
| Commutativity AND | T(x,y) | T(y,x) |
| Commutativity OR | S(x,y) | S(y,x) |
| Monotonicity AND | x1 <= x2 => T(x1,y) <= T(x2,y) | True |
| Monotonicity OR | x1 <= x2 => S(x1,y) <= S(x2,y) | True |

### Residuation Property (residuated implications only)

For Godel, Goguen, and Lukasiewicz residuated implications:

```
T(a, I(a, b)) <= b    for all a, b in [0,1]
```

This does NOT hold for S-implications (Reichenbach, Kleene-Dienes).

### Lukasiewicz-Specific Identities

| Identity | Formula | Expected |
|---|---|---|
| Complementarity | T_L(x, N(x)) | 0 (for all x) |
| Excluded middle | S_L(x, N(x)) | 1 (for all x) |

These do NOT hold for Product or Godel families.

### stable_product Identity Tolerances

| Identity | Exact result | stable_product result | Max error |
|---|---|---|---|
| T(x, 1) = x | x | clamp(x, eps, 1) | eps (when x < eps) |
| T(x, 0) = 0 | 0 | eps * clamp(x, eps, 1) | eps (approximately) |
| S(x, 0) = x | x | clamp(x, 0, 1-eps) | eps (when x > 1-eps) |
| S(x, 1) = 1 | 1 | 1 - (1 - clamp(x,0,1-eps)) * eps | ~eps^2 (negligible) |

---

## Appendix C: References

1. **Badreddine, S., d'Avila Garcez, A., Serafini, L., Spranger, M.** (2022).
   "Logic Tensor Networks." *Artificial Intelligence*, 303, 103649.
   The foundational LTN paper defining Real Logic grounding, operator semantics, and
   satisfiability-based training.

2. **van Krieken, E., Acar, E., van Harmelen, F.** (2022).
   "Analyzing Differentiable Fuzzy Logic Operators." *Artificial Intelligence*, 302, 103602.
   Systematic analysis of gradient behavior across t-norm families, including vanishing
   gradient conditions and recommendations for training.

3. **Klement, E. P., Mesiar, R., Pap, E.** (2000).
   *Triangular Norms.* Kluwer Academic Publishers.
   The definitive mathematical reference on t-norms, t-conorms, and their properties.

4. **Hajek, P.** (1998).
   *Metamathematics of Fuzzy Logic.* Kluwer Academic Publishers.
   Formal treatment of many-valued logics including Godel, Product, and Lukasiewicz logics
   as the three fundamental continuous t-norm logics (Mostert-Shields theorem).

5. **LTN Framework (PyPI: ltn).**
   Reference implementation of Logic Tensor Networks in PyTorch.
   Source of the pMeanError quantifier aggregation and stable product semantics.

6. **Reichenbach, H.** (1935).
   *Wahrscheinlichkeitslehre.* Leiden.
   Origin of the Reichenbach implication I(x,y) = 1 - x + xy as a probabilistic
   conditional.
