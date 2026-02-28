# Rule Network: AST Representation, Attention-Weighted Application, Violation Metrics, and Interpretable Logs

Reference document for the rule network within the Neuro-Symbolic Engine skill. Covers rule
compilation into Abstract Syntax Trees, bottom-up evaluation over entity embeddings, attention-
weighted rule application, constraint loss computation with violation metrics, joint training
modes, interpretable logging, theory management, and integration code patterns.

---

## Table of Contents

1. [Rule Representation as AST](#1-rule-representation-as-ast)
2. [Rule Evaluation Engine](#2-rule-evaluation-engine)
3. [Attention Over Rules](#3-attention-over-rules)
4. [Constraint Loss Computation](#4-constraint-loss-computation)
5. [Joint Training Modes](#5-joint-training-modes)
6. [Interpretable Logs](#6-interpretable-logs)
7. [Rule Compilation from Text/Dict](#7-rule-compilation-from-textdict)
8. [Theory Management](#8-theory-management)
9. [Code Patterns](#9-code-patterns)

Appendices:
- [A. Rule Evaluation Complexity Analysis](#appendix-a-rule-evaluation-complexity-analysis)
- [B. Example Rules in String/Dict/AST Form](#appendix-b-example-rules-in-stringdictast-form)
- [C. Integration with System 2 Reasoning](#appendix-c-integration-with-system-2-reasoning)

---

## 1. Rule Representation as AST

### Motivation

Rules in the neuro-symbolic engine are first-order logic formulas evaluated over continuous
truth values in [0, 1]. Represent each rule as an Abstract Syntax Tree (AST) so that the
evaluation engine can traverse the structure bottom-up, composing truth values from leaf
literals through connectives and quantifiers. The AST representation decouples rule
specification (what the rule says) from rule evaluation (how truth values are computed).

### AST Node Types

Four node types compose every rule AST:

| Node Type | Role | Children | Stored Fields |
|---|---|---|---|
| `LiteralNode` | Leaf: evaluates a predicate or relation | None | `predicate_name`, `variable_names`, `negated` |
| `ConnectiveNode` | Internal: combines child truth values | 2 (left, right) | `op` in {`AND`, `OR`, `IMPLIES`, `IFF`} |
| `QuantifierNode` | Aggregation: iterates over entity dimension | 1 (body) | `quantifier` in {`FORALL`, `EXISTS`}, `variable` |
| `NegationNode` | Unary: inverts a child truth value | 1 (child) | None (operator is always NOT) |

### Node Field Specification

Every AST node stores the following base fields:

| Field | Type | Description |
|---|---|---|
| `node_type` | `str` | One of `"literal"`, `"connective"`, `"quantifier"`, `"negation"` |
| `children` | `List[ASTNode]` | Child nodes; empty for literals |
| `variable_bindings` | `Dict[str, int]` | Maps variable names to entity dimension indices |
| `weight` | `Optional[float]` | Learnable or fixed soft-constraint weight; `None` for hard rules |

Literal nodes additionally store:

| Field | Type | Description |
|---|---|---|
| `predicate_name` | `str` | Name referencing a registered `PredicateModule` or `RelationModule` |
| `variable_names` | `Tuple[str, ...]` | Ordered variable names, e.g., `("x",)` for unary, `("x", "y")` for binary |
| `negated` | `bool` | If `True`, apply NOT to the predicate output |

Connective nodes additionally store:

| Field | Type | Description |
|---|---|---|
| `op` | `str` | Operator type: `"AND"`, `"OR"`, `"IMPLIES"`, `"IFF"` |

Quantifier nodes additionally store:

| Field | Type | Description |
|---|---|---|
| `quantifier` | `str` | `"FORALL"` or `"EXISTS"` |
| `variable` | `str` | The scoped variable name this quantifier binds |

### Weighted Rules

Each rule may carry an optional weight indicating its importance as a soft constraint:

- **Hard rules** (`weight=None`): violations contribute equally to the constraint loss.
- **Soft rules** (`weight=float`): violations are scaled by the weight before aggregation.
- **Learnable weights**: store the weight as an `nn.Parameter` to let the optimizer adjust
  rule importance during training.
- **Fixed weights**: store as a plain float; useful for domain-expert-specified importance.

When `weight` is set, the violation contribution becomes `weight * violation(rule)` before
attention weighting is applied.

### AST Construction Summary

```
Rule string / dict
        |
        v
   [Parser / Compiler]
        |
        v
   ASTNode tree
        |
        v
   Bottom-up evaluator
        |
        v
   Truth value (B,) or (B, R)
```

---

## 2. Rule Evaluation Engine

### Bottom-Up Evaluation

Evaluate a rule AST by traversing from leaves to root. At each node, compute a truth value
tensor based on the node type and the truth values of its children.

**Evaluation order:**

1. Identify all `LiteralNode` leaves.
2. Evaluate each literal by invoking the corresponding predicate/relation module on the
   bound entity embeddings. Result: truth value tensor with shape depending on free variables.
3. Propagate upward through `NegationNode`, `ConnectiveNode`, and `QuantifierNode` nodes,
   applying the operator bundle's corresponding function at each step.
4. The root node produces the final truth value for the rule.

### Variable Binding

Track which entity indices are bound to each variable during evaluation. A binding context
maps variable names to slices of the entity tensor:

| Variable | Binding | Tensor Slice |
|---|---|---|
| `x` | Iterates over entity dimension 1 | `entities[:, i, :]` for index `i` |
| `y` | Iterates over entity dimension 2 (if nested) | `entities[:, j, :]` for index `j` |

For a unary predicate `P(x)`, evaluate as:

```python
# entities: (B, N, D_ent), predicate_P: PredicateModule
truth_P_x = predicate_P(entities)  # (B, N) -- truth per entity
```

For a binary relation `R(x, y)`, evaluate as:

```python
# Expand entities for pairwise evaluation
x_expanded = entities.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, D)
y_expanded = entities.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, D)
truth_R_xy = relation_R(x_expanded, y_expanded)            # (B, N, N)
```

### Quantifier Scoping

Quantifiers aggregate over the entity dimension corresponding to the bound variable:

- `FORALL x`: aggregate over the `x` dimension using the operator bundle's FORALL aggregator.
- `EXISTS x`: aggregate over the `x` dimension using the operator bundle's EXISTS aggregator.

**Nested quantifiers** are evaluated inside-out:

For `FORALL x: EXISTS y: R(x, y)`:

1. Evaluate `R(x, y)` -> `(B, N_x, N_y)` truth tensor.
2. Apply `EXISTS` over the `y` dimension -> `(B, N_x)`.
3. Apply `FORALL` over the `x` dimension -> `(B,)`.

The order of aggregation matters. Always resolve the innermost quantifier first.

### Batched Evaluation

Evaluate all rules in a theory simultaneously across the batch:

- Input: entity embeddings `(B, N, D_ent)`, predicate registry, list of R rule ASTs.
- Output: truth values `(B, R)` -- one truth value per batch element per rule.

Stack rule results along dimension 1 after independent evaluation. Rules with different
structures (different ASTs) are evaluated independently; there is no cross-rule batching
of the AST traversal itself, only batching across the B dimension.

### Literal Evaluation with Negation

When a `LiteralNode` has `negated=True`, apply the operator bundle's NOT function after
predicate evaluation:

```python
truth = predicate(bound_entities)
if literal_node.negated:
    truth = operator_bundle.NOT(truth)
```

### Connective Evaluation

Connective nodes combine two child truth values:

| Connective | Operator Bundle Call | Semantics |
|---|---|---|
| `AND` | `bundle.AND(left, right)` | Both children must be true |
| `OR` | `bundle.OR(left, right)` | At least one child must be true |
| `IMPLIES` | `bundle.IMPLIES(left, right)` | If left then right |
| `IFF` | `bundle.EQUIV(left, right)` | Left and right have same truth |

All connective operations are element-wise over the batch dimension and any free variable
dimensions. The operator bundle selection (Godel, product, Lukasiewicz, stable_product)
determines the exact numerical implementation. See `references/fuzzy-operators.md`.

---

## 3. Attention Over Rules

### Purpose

Instead of applying all rules equally, compute per-rule relevance weights conditioned on
the current context. This allows the network to focus on the most relevant constraints
for each input, reducing noise from irrelevant rules.

### Attention Mechanism

```
rule_embedding = learned vector per rule OR derived from AST structure
alpha_r = softmax(f([workspace_summary; query_embedding; rule_embedding]))
```

The attention function `f` is a single-layer MLP with tanh activation:

```python
# Inputs concatenated: workspace_summary (D_ws), query_embedding (D_q), rule_embedding (D_r)
# D_ws = D_q = D_r is typical but not required
h = torch.cat([workspace_summary, query_embedding, rule_embedding], dim=-1)
score = self.attn_linear(torch.tanh(self.attn_proj(h)))  # scalar per rule
alpha = F.softmax(scores, dim=-1)  # (B, R) normalized per batch element
```

### Inputs to the Attention Function

| Input | Shape | Source | Fallback |
|---|---|---|---|
| `workspace_summary` | `(B, D_ws)` | Global workspace broadcast vector | Required; no fallback |
| `query_embedding` | `(B, D_q)` | Query formula embedding (if evaluating a specific query) | Zeros if no query provided |
| `rule_embedding` | `(R, D_r)` | Per-rule vector, broadcast across batch | Required; one vector per rule |

When no query is provided, zero out the query component and use only workspace summary
and rule embedding to compute attention.

### Rule Embedding Options

**Option A -- Learned embeddings (default):**

Assign one trainable vector per rule. Simple and effective. Each rule gets a distinct
`nn.Parameter` vector of dimension `D_r`. Initialize with orthogonal or Xavier initialization
to ensure rules start with distinct representations.

```python
self.rule_embeddings = nn.Parameter(torch.randn(max_rules, rule_embed_dim))
nn.init.orthogonal_(self.rule_embeddings)
```

Advantages: simple, no dependency on AST structure, fast to compute.
Disadvantage: does not generalize to unseen rules at test time.

**Option B -- Structural embeddings (TreeLSTM):**

Encode the AST structure via recursive tree processing. At each node, compute a hidden
state from the node type embedding and children's hidden states using a TreeLSTM cell.
The root hidden state is the rule embedding.

```python
# Pseudocode for TreeLSTM rule embedding
def embed_node(node):
    if node.is_leaf():
        return leaf_encoder(node.predicate_embedding)
    child_states = [embed_node(c) for c in node.children]
    return tree_lstm_cell(node.type_embedding, child_states)

rule_embedding = embed_node(rule_ast.root)
```

Advantages: generalizes to new rules, captures structural similarity.
Disadvantage: slower, requires TreeLSTM implementation, harder to batch.

**Recommendation:** Use learned embeddings (Option A) for fixed rule sets. Switch to
structural embeddings (Option B) only when rules change at test time or when transfer
across different theories is required.

### Attention Output

The attention weights `alpha_r` have shape `(B, R)`, normalized along the rule dimension
(dim=-1) per batch element. Each `alpha_r[b, r]` indicates how relevant rule `r` is to
batch element `b`.

---

## 4. Constraint Loss Computation

### Violation Metric

The violation of a rule measures how far the rule's truth value is from its intended
truth polarity:

- **Positive rules** (intended to be true): `violation(r) = 1 - truth(r)`
- **Negative rules** (intended to be false): `violation(r) = truth(r)`

Both produce values in [0, 1], where 0 means no violation and 1 means maximum violation.

### Attention-Weighted Constraint Loss

When attention is enabled (`RuleConfig.use_attention=True`):

```
constraint_loss = sum_r(alpha_r * violation(r))
```

Expanded per batch element:

```python
# truth_values: (B, R), alpha: (B, R)
violations = 1.0 - truth_values  # for positive rules
weighted = alpha * violations     # (B, R)
constraint_loss = weighted.sum(dim=-1).mean()  # scalar
```

### Unweighted Fallback

When attention is disabled (`RuleConfig.use_attention=False`), aggregate violations directly
using one of three strategies:

### Aggregation Strategies

| Strategy | Formula | Behavior | Config Value |
|---|---|---|---|
| `"mean"` | `mean(violation(r) for r in rules)` | Equal treatment of all rules; robust baseline | `violation_agg="mean"` |
| `"soft_min"` | `-tau * log(mean(exp(-violation(r)/tau)))` | Penalizes worst violations more heavily; smooth min approximation | `violation_agg="soft_min"` |
| `"p_mean"` | `(mean(violation(r)^p))^(1/p)` | Configurable harshness via `p`; higher `p` focuses on larger violations | `violation_agg="p_mean"` |

**Mean aggregation** is the default. It treats all rule violations equally and produces
stable gradients. Use it unless there is a specific reason to focus on worst-case violations.

**Soft-min aggregation** approximates the maximum violation using a LogSumExp formulation.
The temperature parameter `tau` controls sharpness: lower `tau` approximates true max more
closely but has sharper gradients. Default `tau=0.1`.

```python
def soft_min_agg(violations, tau=0.1):
    # Smooth approximation of max(violations)
    return tau * torch.logsumexp(violations / tau, dim=-1)
```

**P-mean aggregation** uses the generalized power mean. With `p=1` it reduces to arithmetic
mean. With `p=2` it emphasizes larger violations (like L2 norm). With `p -> inf` it
approaches the max.

```python
def p_mean_agg(violations, p=2.0):
    return (violations.pow(p).mean(dim=-1)).pow(1.0 / p)
```

### Rule Weight Integration

When rules have individual weights (soft constraints), apply them before aggregation:

```python
# rule_weights: (R,) from rule AST weight fields
violations = (1.0 - truth_values) * rule_weights.unsqueeze(0)  # (B, R)
```

This stacks with attention weights when both are enabled:

```python
weighted = alpha * rule_weights.unsqueeze(0) * (1.0 - truth_values)
constraint_loss = weighted.sum(dim=-1).mean()
```

---

## 5. Joint Training Modes

### Mode A: Constraint-Only Regularizer

Use the symbolic engine as a regularization term alongside the primary task loss:

```
L_total = L_task + lambda * L_symbolic
```

**Lambda warmup schedule:** Start `lambda` at 0 and linearly increase to the target value
(`RuleConfig.constraint_weight`) over `RuleConfig.warmup_steps` training steps:

```python
def get_lambda(step, warmup_steps, target_lambda):
    if step >= warmup_steps:
        return target_lambda
    return target_lambda * (step / warmup_steps)
```

**Rationale for warmup:** Early in training, predicate modules produce random truth values.
Applying full constraint pressure before groundings are meaningful forces the optimizer to
satisfy rules by collapsing all truth values toward trivial solutions (all 0 or all 1).
Warmup allows the grounding modules to learn meaningful representations before constraint
pressure increases.

**Gradient flow path:**

```
constraint_loss
    -> rule truth values
    -> operator bundle outputs (AND, OR, IMPLIES, ...)
    -> predicate module outputs
    -> entity embeddings
    -> entity extractor
    -> upstream encoder (vision, text, etc.)
```

Gradients from the constraint loss propagate through every differentiable component in the
chain. This is the mechanism by which symbolic rules shape the upstream neural representations.

### Mode B: Supervised Symbolic Truth Training

For logic datasets (ProofWriter, FOLIO) with labeled query truth values:

```
L = L_query + lambda * L_consistency
```

Where:

- `L_query = BCE(predicted_truth(query), label)` for entailed/contradicted queries
- `L_consistency = constraint_loss` from rule violations (as in Mode A)

**Label encoding:**

| Label | Truth Target | Loss Contribution |
|---|---|---|
| Entailed | 1.0 | BCE with target 1.0 |
| Contradicted | 0.0 | BCE with target 0.0 |
| Unknown | 0.5 or masked | BCE with target 0.5, or exclude from loss |

Masking unknown labels is preferred over the 0.5 target when the dataset contains many
unknowns, because training toward 0.5 provides a weak and potentially misleading signal.

### Mode Selection

| Scenario | Mode | Rationale |
|---|---|---|
| Vision/text classification with domain rules | A (regularizer) | Rules enforce prior knowledge; primary loss drives classification |
| ProofWriter / FOLIO evaluation | B (supervised) | Labels available; measure query-answering accuracy |
| Multi-task with logic and task heads | A + B combined | Both constraint and query losses active |

---

## 6. Interpretable Logs

### Log Activation

Pass `return_logs=True` to the `RuleNetwork.forward()` call. When disabled (default),
no logs are produced and no additional computation is performed.

### Log Contents

| Log Key | Shape / Type | Description |
|---|---|---|
| `"rule_attention_weights"` | `(B, R)` tensor | Attention weights `alpha_r` per rule per batch element |
| `"per_rule_truth_values"` | `(B, R)` tensor | Raw truth value of each rule before violation computation |
| `"top_k_violated_rules"` | `Dict` with `"indices"`: `(B, K)`, `"values"`: `(B, K)` | Indices and violation values of the K most violated rules per batch element |
| `"proof_trace"` | `List[Dict]` per batch element | Compressed trace: for each query, which rules contributed most to the truth value |

### Storage Policy

All log tensors are **detached** (`tensor.detach()`) before storage. No gradient tracking
is retained in logs. This prevents accidental memory leaks from holding references to the
computation graph.

**Default mode (top-K only):** Store only the top-K violated rules (default K=5). This
bounds log memory at `O(B * K)` regardless of the number of rules.

**Full tensor mode (opt-in):** Set `log_full_tensors=True` to store the complete `(B, R)`
attention and truth tensors. Use only for debugging; disable in production training.

### Top-K Extraction

```python
def extract_top_k_violations(truth_values, k=5):
    """Extract indices and values of top-K violated rules per batch element."""
    violations = 1.0 - truth_values  # (B, R)
    top_values, top_indices = torch.topk(violations, k=min(k, violations.shape[-1]), dim=-1)
    return {
        "indices": top_indices.detach(),   # (B, K)
        "values": top_values.detach(),     # (B, K)
    }
```

### Proof-Style Trace

For each evaluated query, produce a compressed trace identifying which rules contributed
most to the final truth value. The trace is a list of entries, one per query:

```python
{
    "query": "can_fly(tweety)",
    "truth": 0.87,
    "contributing_rules": [
        {"rule_idx": 3, "attention": 0.45, "truth": 0.92, "violation": 0.08},
        {"rule_idx": 1, "attention": 0.30, "truth": 0.78, "violation": 0.22},
    ]
}
```

The `contributing_rules` list is sorted by descending attention weight and truncated to the
top-K entries (default K=3). This provides a human-readable explanation of why a particular
truth value was assigned.

### Log Extraction Pattern

```python
output = rule_network(entities, predicates, rules, return_logs=True)

if output.logs is not None:
    attn = output.logs["rule_attention_weights"]  # (B, R)
    top_violated = output.logs["top_k_violated_rules"]
    print(f"Most violated rule indices: {top_violated['indices'][0]}")
    print(f"Violation values: {top_violated['values'][0]}")
```

---

## 7. Rule Compilation from Text/Dict

### String Format

Rules expressed as human-readable strings are parsed into AST objects. The grammar supports
quantifiers, connectives, predicates, and negation:

```
FORALL x: is_bird(x) IMPLIES can_fly(x)
FORALL x: FORALL y: parent(x, y) AND parent(y, z) IMPLIES grandparent(x, z)
EXISTS x: is_penguin(x) AND NOT can_fly(x)
FORALL x: is_mammal(x) IFF (has_fur(x) OR has_milk(x))
```

**Parsing rules:**

1. Tokenize on whitespace and parentheses.
2. `FORALL var:` and `EXISTS var:` introduce quantifier nodes with the named variable.
3. `AND`, `OR`, `IMPLIES`, `IFF` introduce connective nodes.
4. `NOT` introduces a negation node.
5. `predicate_name(var1, var2, ...)` introduces a literal node.
6. Operator precedence: NOT > AND > OR > IMPLIES > IFF (lowest).
7. Parentheses override precedence.

### Dict Format

Rules expressed as nested dictionaries provide unambiguous structure without parsing:

```python
{
    "type": "forall",
    "var": "x",
    "body": {
        "type": "implies",
        "lhs": {
            "type": "literal",
            "predicate": "is_bird",
            "args": ["x"],
            "negated": False
        },
        "rhs": {
            "type": "literal",
            "predicate": "can_fly",
            "args": ["x"],
            "negated": False
        }
    },
    "weight": 1.0
}
```

**Dict node type mapping:**

| `"type"` Value | AST Node | Required Keys |
|---|---|---|
| `"literal"` | `LiteralNode` | `"predicate"`, `"args"`, optional `"negated"` |
| `"and"`, `"or"`, `"implies"`, `"iff"` | `ConnectiveNode` | `"lhs"`, `"rhs"` |
| `"forall"`, `"exists"` | `QuantifierNode` | `"var"`, `"body"` |
| `"not"` | `NegationNode` | `"child"` |

### JSON Serialization

Both string and dict forms are JSON-serializable. Store compiled rules alongside model
checkpoints to ensure reproducibility:

```python
import json

def serialize_theory(rules):
    """Serialize a list of rule dicts to JSON string."""
    return json.dumps(rules, indent=2)

def deserialize_theory(json_str):
    """Deserialize JSON string to list of rule dicts."""
    return json.loads(json_str)
```

Include the serialized theory in the checkpoint under the key `"rule_theory"`.

### Validation

After compilation, validate the AST against the predicate registry:

1. Check that every `predicate_name` referenced in a `LiteralNode` exists in the registry.
2. Check that the arity of each predicate matches the number of variables in `variable_names`.
3. Check that every variable referenced in a literal is bound by an enclosing quantifier
   or is a free variable explicitly declared.
4. Check for duplicate variable names in nested quantifiers of the same scope chain.

**Error messages for malformed rules:**

| Error | Message Template |
|---|---|
| Unknown predicate | `"Rule {idx}: predicate '{name}' not found in registry. Available: {list}"` |
| Arity mismatch | `"Rule {idx}: predicate '{name}' expects {expected} args, got {actual}"` |
| Unbound variable | `"Rule {idx}: variable '{var}' is not bound by any enclosing quantifier"` |
| Duplicate binding | `"Rule {idx}: variable '{var}' is already bound in enclosing scope"` |

---

## 8. Theory Management

### Theory Object

A theory is a named collection of rules forming a knowledge base:

```python
@dataclass
class Theory:
    name: str
    rules: List[ASTNode]              # Compiled rule ASTs
    rule_embeddings: nn.Parameter      # (R, D_r) learned rule vectors
    metadata: Dict[str, Any]           # Source info, creation timestamp, etc.
    polarity: List[bool]               # True = positive rule, False = negative rule
```

### Global vs Per-Example Theories

| Theory Type | Scope | Use Case |
|---|---|---|
| Global | Shared across entire batch | Domain knowledge that applies to all inputs (e.g., "birds can fly") |
| Per-example | Unique per batch element | ProofWriter-style datasets where each example has its own fact/rule set |

Global theories are stored as module attributes and evaluated once per batch with results
broadcast. Per-example theories are passed as input and evaluated individually per batch
element.

### Theory Composition

Merge multiple theories for multi-task or multi-domain learning:

```python
def compose_theories(theory_a, theory_b):
    """Merge two theories into a combined theory."""
    combined_rules = theory_a.rules + theory_b.rules
    combined_embeddings = torch.cat(
        [theory_a.rule_embeddings, theory_b.rule_embeddings], dim=0
    )
    combined_polarity = theory_a.polarity + theory_b.polarity
    return Theory(
        name=f"{theory_a.name}+{theory_b.name}",
        rules=combined_rules,
        rule_embeddings=nn.Parameter(combined_embeddings),
        metadata={"sources": [theory_a.name, theory_b.name]},
        polarity=combined_polarity,
    )
```

Attention weights are recomputed over the combined rule set, so the network naturally
balances relevance across both theories.

### Runtime Rule Modification

Add or remove rules at runtime for incremental knowledge updates:

**Adding a rule:**

1. Parse the new rule string/dict into an AST.
2. Validate against the predicate registry.
3. Append the AST to `theory.rules`.
4. Expand `theory.rule_embeddings` by concatenating a new initialized vector.
5. Append polarity to `theory.polarity`.

**Removing a rule:**

1. Identify the rule index to remove.
2. Delete from `theory.rules`, `theory.polarity`.
3. Remove the corresponding row from `theory.rule_embeddings` (re-allocate parameter).

**Constraint:** Rule addition/removal invalidates any cached attention scores. Clear caches
after modification. Rule indices in logged traces refer to positions in the rule list at the
time of evaluation; track rule identifiers (names or hashes) for stable cross-checkpoint
references.

---

## 9. Code Patterns

### AST Node Classes

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

@dataclass
class ASTNode:
    """Base class for all AST nodes."""
    node_type: str
    children: List['ASTNode'] = field(default_factory=list)
    variable_bindings: Dict[str, int] = field(default_factory=dict)
    weight: Optional[float] = None


@dataclass
class LiteralNode(ASTNode):
    """Leaf node: evaluates a predicate or relation."""
    predicate_name: str = ""
    variable_names: Tuple[str, ...] = ()
    negated: bool = False

    def __post_init__(self):
        self.node_type = "literal"
        self.children = []


@dataclass
class ConnectiveNode(ASTNode):
    """Internal node: combines two child truth values."""
    op: str = "AND"  # "AND", "OR", "IMPLIES", "IFF"

    def __post_init__(self):
        self.node_type = "connective"
        assert len(self.children) == 2, f"Connective requires 2 children, got {len(self.children)}"


@dataclass
class QuantifierNode(ASTNode):
    """Aggregation node: iterates over entity dimension."""
    quantifier: str = "FORALL"  # "FORALL", "EXISTS"
    variable: str = "x"

    def __post_init__(self):
        self.node_type = "quantifier"
        assert len(self.children) == 1, f"Quantifier requires 1 child, got {len(self.children)}"


@dataclass
class NegationNode(ASTNode):
    """Unary node: inverts child truth value."""

    def __post_init__(self):
        self.node_type = "negation"
        assert len(self.children) == 1, f"Negation requires 1 child, got {len(self.children)}"
```

### Bottom-Up Rule Evaluator

```python
import torch
from typing import Dict

class RuleEvaluator:
    """Evaluate a rule AST bottom-up given entities and an operator bundle."""

    def __init__(self, operator_bundle, predicate_registry: Dict[str, 'nn.Module']):
        self.ops = operator_bundle
        self.predicates = predicate_registry

    def evaluate(
        self,
        node: ASTNode,
        entities: torch.Tensor,          # (B, N, D_ent)
        bindings: Dict[str, int] = None, # variable -> entity dim index
    ) -> torch.Tensor:
        """
        Recursively evaluate an AST node.

        Returns a truth tensor whose shape depends on free variables:
        - No free vars: (B,)
        - One free var x over N entities: (B, N)
        - Two free vars x, y: (B, N, N)
        """
        if bindings is None:
            bindings = {}

        if isinstance(node, LiteralNode):
            return self._eval_literal(node, entities, bindings)
        elif isinstance(node, NegationNode):
            child_truth = self.evaluate(node.children[0], entities, bindings)
            return self.ops.NOT(child_truth)
        elif isinstance(node, ConnectiveNode):
            left = self.evaluate(node.children[0], entities, bindings)
            right = self.evaluate(node.children[1], entities, bindings)
            return self._apply_connective(node.op, left, right)
        elif isinstance(node, QuantifierNode):
            return self._eval_quantifier(node, entities, bindings)
        else:
            raise ValueError(f"Unknown node type: {node.node_type}")

    def _eval_literal(self, node: LiteralNode, entities, bindings):
        pred = self.predicates[node.predicate_name]
        if len(node.variable_names) == 1:
            # Unary: P(x) -> evaluate over entity dim
            truth = pred(entities)  # (B, N)
        elif len(node.variable_names) == 2:
            # Binary: R(x, y) -> pairwise evaluation
            B, N, D = entities.shape
            x_exp = entities.unsqueeze(2).expand(B, N, N, D)
            y_exp = entities.unsqueeze(1).expand(B, N, N, D)
            truth = pred(x_exp, y_exp)  # (B, N, N)
        else:
            raise ValueError(f"Arity {len(node.variable_names)} not supported")

        if node.negated:
            truth = self.ops.NOT(truth)
        return truth

    def _apply_connective(self, op, left, right):
        if op == "AND":
            return self.ops.AND(left, right)
        elif op == "OR":
            return self.ops.OR(left, right)
        elif op == "IMPLIES":
            return self.ops.IMPLIES(left, right)
        elif op == "IFF":
            return self.ops.EQUIV(left, right)
        else:
            raise ValueError(f"Unknown connective: {op}")

    def _eval_quantifier(self, node: QuantifierNode, entities, bindings):
        # Evaluate body with the quantified variable free
        body_truth = self.evaluate(node.children[0], entities, bindings)

        # Determine which dimension to aggregate
        # The outermost free variable corresponds to the last entity dimension
        if node.quantifier == "FORALL":
            return self.ops.FORALL(body_truth, dim=-1)
        elif node.quantifier == "EXISTS":
            return self.ops.EXISTS(body_truth, dim=-1)
        else:
            raise ValueError(f"Unknown quantifier: {node.quantifier}")
```

### RuleNetwork with Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, List

@dataclass
class RuleNetworkOutput:
    """Output of the RuleNetwork forward pass."""
    truth_values: torch.Tensor          # (B, R)
    constraint_loss: torch.Tensor       # scalar
    attention_weights: torch.Tensor     # (B, R)
    logs: Optional[Dict] = None

class RuleNetwork(nn.Module):
    """
    Evaluate a set of rules with attention-weighted constraint loss.

    Args:
        config: RuleConfig with max_rules, use_attention, violation_agg, etc.
        workspace_dim: Dimension of workspace summary vector.
        rule_embed_dim: Dimension of per-rule embedding vectors.
    """

    def __init__(self, config, workspace_dim: int, rule_embed_dim: int = 128):
        super().__init__()
        self.config = config
        self.max_rules = config.max_rules
        self.use_attention = config.use_attention

        # Per-rule learned embeddings
        self.rule_embeddings = nn.Parameter(
            torch.randn(config.max_rules, rule_embed_dim)
        )
        nn.init.orthogonal_(self.rule_embeddings)

        # Attention MLP: [workspace_summary; query_embed; rule_embed] -> scalar
        if self.use_attention:
            attn_input_dim = workspace_dim + rule_embed_dim + rule_embed_dim
            self.attn_proj = nn.Linear(attn_input_dim, rule_embed_dim)
            self.attn_score = nn.Linear(rule_embed_dim, 1)

    def compute_attention(
        self,
        workspace_summary: torch.Tensor,   # (B, D_ws)
        query_embedding: Optional[torch.Tensor],  # (B, D_q) or None
        num_rules: int,
    ) -> torch.Tensor:
        """Compute per-rule attention weights. Returns (B, R)."""
        B = workspace_summary.shape[0]
        R = num_rules

        # Default query to zeros if not provided
        if query_embedding is None:
            query_embedding = torch.zeros(
                B, self.rule_embeddings.shape[-1],
                device=workspace_summary.device,
            )

        # Expand rule embeddings across batch: (R, D_r) -> (B, R, D_r)
        r_emb = self.rule_embeddings[:R].unsqueeze(0).expand(B, -1, -1)

        # Expand workspace and query: (B, D) -> (B, R, D)
        ws_exp = workspace_summary.unsqueeze(1).expand(-1, R, -1)
        q_exp = query_embedding.unsqueeze(1).expand(-1, R, -1)

        # Concatenate and compute scores
        h = torch.cat([ws_exp, q_exp, r_emb], dim=-1)  # (B, R, D_in)
        scores = self.attn_score(torch.tanh(self.attn_proj(h))).squeeze(-1)  # (B, R)

        return F.softmax(scores, dim=-1)  # (B, R)

    def compute_constraint_loss(
        self,
        truth_values: torch.Tensor,    # (B, R)
        attention: torch.Tensor,        # (B, R)
        polarity: Optional[List[bool]] = None,
    ) -> torch.Tensor:
        """Compute attention-weighted constraint loss."""
        R = truth_values.shape[-1]

        # Compute violations based on polarity
        if polarity is not None:
            polarity_mask = torch.tensor(
                polarity[:R], dtype=torch.float32, device=truth_values.device
            )
            # positive rules: violation = 1 - truth
            # negative rules: violation = truth
            violations = polarity_mask * (1.0 - truth_values) + (1.0 - polarity_mask) * truth_values
        else:
            # Default: all positive rules
            violations = 1.0 - truth_values  # (B, R)

        if self.use_attention:
            weighted = attention * violations  # (B, R)
            return weighted.sum(dim=-1).mean()  # scalar
        else:
            return self._aggregate_violations(violations)

    def _aggregate_violations(self, violations: torch.Tensor) -> torch.Tensor:
        """Aggregate violations without attention."""
        agg = self.config.violation_agg
        if agg == "mean":
            return violations.mean()
        elif agg == "soft_min":
            tau = getattr(self.config, 'soft_min_tau', 0.1)
            return (tau * torch.logsumexp(violations / tau, dim=-1)).mean()
        elif agg == "p_mean":
            p = getattr(self.config, 'p_mean_p', 2.0)
            return (violations.pow(p).mean(dim=-1)).pow(1.0 / p).mean()
        else:
            raise ValueError(f"Unknown aggregation: {agg}")

    def forward(
        self,
        truth_values: torch.Tensor,                   # (B, R)
        workspace_summary: torch.Tensor,               # (B, D_ws)
        query_embedding: Optional[torch.Tensor] = None,
        polarity: Optional[List[bool]] = None,
        return_logs: bool = False,
    ) -> RuleNetworkOutput:
        """
        Compute attention-weighted constraint loss from pre-evaluated rule truth values.
        """
        B, R = truth_values.shape

        # Compute attention
        if self.use_attention:
            alpha = self.compute_attention(workspace_summary, query_embedding, R)
        else:
            alpha = torch.ones(B, R, device=truth_values.device) / R

        # Compute constraint loss
        constraint_loss = self.compute_constraint_loss(truth_values, alpha, polarity)

        # Build logs
        logs = None
        if return_logs:
            logs = {
                "rule_attention_weights": alpha.detach(),
                "per_rule_truth_values": truth_values.detach(),
                "top_k_violated_rules": self._extract_top_k(truth_values),
            }

        return RuleNetworkOutput(
            truth_values=truth_values,
            constraint_loss=constraint_loss,
            attention_weights=alpha,
            logs=logs,
        )

    def _extract_top_k(self, truth_values, k=5):
        violations = 1.0 - truth_values
        k_actual = min(k, violations.shape[-1])
        top_values, top_indices = torch.topk(violations, k=k_actual, dim=-1)
        return {
            "indices": top_indices.detach(),
            "values": top_values.detach(),
        }
```

### Constraint Loss with Warmup

```python
class ConstraintLossWithWarmup:
    """Manage lambda warmup for constraint loss integration."""

    def __init__(self, target_lambda: float, warmup_steps: int):
        self.target_lambda = target_lambda
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def get_lambda(self) -> float:
        """Return current lambda value based on warmup progress."""
        if self.current_step >= self.warmup_steps:
            return self.target_lambda
        return self.target_lambda * (self.current_step / self.warmup_steps)

    def step(self):
        """Advance the warmup counter by one step."""
        self.current_step += 1

    def compute_total_loss(
        self,
        task_loss: torch.Tensor,
        constraint_loss: torch.Tensor,
    ) -> torch.Tensor:
        """Compute L_total = L_task + lambda * L_symbolic."""
        lam = self.get_lambda()
        total = task_loss + lam * constraint_loss
        self.step()
        return total
```

**Usage in a training loop:**

```python
warmup = ConstraintLossWithWarmup(
    target_lambda=config.constraint_weight,  # e.g., 0.1
    warmup_steps=config.warmup_steps,         # e.g., 1000
)

for batch in dataloader:
    task_loss = compute_task_loss(model, batch)
    symbolic_output = rule_network(truth_values, workspace_summary)
    total_loss = warmup.compute_total_loss(task_loss, symbolic_output.constraint_loss)
    total_loss.backward()
    optimizer.step()
```

### Log Extraction and Compression

```python
def format_proof_trace(
    logs: Dict,
    rule_names: Optional[List[str]] = None,
    top_k: int = 3,
) -> List[Dict]:
    """
    Format logged data into human-readable proof traces.

    Args:
        logs: Output from RuleNetwork with return_logs=True.
        rule_names: Optional list of rule name strings for display.
        top_k: Number of top contributing rules to include per trace.

    Returns:
        List of trace dicts, one per batch element.
    """
    attn = logs["rule_attention_weights"]      # (B, R)
    truths = logs["per_rule_truth_values"]     # (B, R)
    B, R = attn.shape

    traces = []
    for b in range(B):
        # Sort rules by attention weight descending
        sorted_indices = torch.argsort(attn[b], descending=True)[:top_k]
        contributing = []
        for idx in sorted_indices:
            i = idx.item()
            name = rule_names[i] if rule_names and i < len(rule_names) else f"rule_{i}"
            contributing.append({
                "rule_idx": i,
                "rule_name": name,
                "attention": attn[b, i].item(),
                "truth": truths[b, i].item(),
                "violation": 1.0 - truths[b, i].item(),
            })
        traces.append({
            "batch_idx": b,
            "contributing_rules": contributing,
        })

    return traces
```

---

## Appendix A: Rule Evaluation Complexity Analysis

### Time Complexity per Rule

| Component | Complexity | Variables |
|---|---|---|
| Unary literal `P(x)` | `O(B * N * D_ent)` | B=batch, N=entities, D=entity dim |
| Binary literal `R(x,y)` | `O(B * N^2 * D_ent)` | Pairwise evaluation over N entities |
| Negation | `O(B * N^k)` | Element-wise; k = number of free variables |
| AND / OR / IMPLIES / IFF | `O(B * N^k)` | Element-wise over free variable dimensions |
| FORALL / EXISTS | `O(B * N^k)` | Reduction over one dimension |

### Full Rule Evaluation

For a theory with R rules, each containing at most Q quantifiers over N entities:

```
Total = O(R * B * N^Q * D_ent)
```

The dominant cost is binary relations with nested quantifiers: `FORALL x: FORALL y: R(x,y)`
is `O(B * N^2 * D_ent)` per rule. For N=32 entities and R=64 rules, this is manageable.
For N > 100, consider:

- Subsampling entities per quantifier scope.
- Approximating quantifiers with k-nearest-entity evaluation.
- Caching predicate evaluations across rules that share the same literal.

### Memory Complexity

Peak memory is dominated by the largest intermediate truth tensor:

| Expression | Peak Tensor Shape | Memory |
|---|---|---|
| `P(x)` | `(B, N)` | `B * N * 4` bytes |
| `R(x, y)` | `(B, N, N)` | `B * N^2 * 4` bytes |
| `FORALL x: EXISTS y: R(x,y)` | `(B, N, N)` before reduction | `B * N^2 * 4` bytes |

With B=32, N=32: peak = 32 * 32 * 32 * 4 = 128 KB per rule. With B=32, N=128: peak =
32 * 128 * 128 * 4 = 2 MB per rule. Monitor memory when scaling entity count.

---

## Appendix B: Example Rules in String/Dict/AST Form

### Example 1: "All birds can fly"

**String:** `FORALL x: is_bird(x) IMPLIES can_fly(x)`

**Dict:**
```json
{
    "type": "forall",
    "var": "x",
    "body": {
        "type": "implies",
        "lhs": {"type": "literal", "predicate": "is_bird", "args": ["x"]},
        "rhs": {"type": "literal", "predicate": "can_fly", "args": ["x"]}
    },
    "weight": 0.8
}
```

**AST:**
```
QuantifierNode(FORALL, var="x")
  └── ConnectiveNode(IMPLIES)
        ├── LiteralNode("is_bird", ("x",), negated=False)
        └── LiteralNode("can_fly", ("x",), negated=False)
```

### Example 2: "Penguins are birds that cannot fly"

**String:** `FORALL x: is_penguin(x) IMPLIES (is_bird(x) AND NOT can_fly(x))`

**Dict:**
```json
{
    "type": "forall",
    "var": "x",
    "body": {
        "type": "implies",
        "lhs": {"type": "literal", "predicate": "is_penguin", "args": ["x"]},
        "rhs": {
            "type": "and",
            "lhs": {"type": "literal", "predicate": "is_bird", "args": ["x"]},
            "rhs": {"type": "not", "child": {"type": "literal", "predicate": "can_fly", "args": ["x"]}}
        }
    }
}
```

**AST:**
```
QuantifierNode(FORALL, var="x")
  └── ConnectiveNode(IMPLIES)
        ├── LiteralNode("is_penguin", ("x",), negated=False)
        └── ConnectiveNode(AND)
              ├── LiteralNode("is_bird", ("x",), negated=False)
              └── NegationNode
                    └── LiteralNode("can_fly", ("x",), negated=False)
```

### Example 3: "Every person has a parent"

**String:** `FORALL x: is_person(x) IMPLIES EXISTS y: parent_of(y, x)`

**Dict:**
```json
{
    "type": "forall",
    "var": "x",
    "body": {
        "type": "implies",
        "lhs": {"type": "literal", "predicate": "is_person", "args": ["x"]},
        "rhs": {
            "type": "exists",
            "var": "y",
            "body": {"type": "literal", "predicate": "parent_of", "args": ["y", "x"]}
        }
    }
}
```

**AST:**
```
QuantifierNode(FORALL, var="x")
  └── ConnectiveNode(IMPLIES)
        ├── LiteralNode("is_person", ("x",), negated=False)
        └── QuantifierNode(EXISTS, var="y")
              └── LiteralNode("parent_of", ("y", "x"), negated=False)
```

### Example 4: "Symmetry of friendship"

**String:** `FORALL x: FORALL y: friends(x, y) IFF friends(y, x)`

**AST:**
```
QuantifierNode(FORALL, var="x")
  └── QuantifierNode(FORALL, var="y")
        └── ConnectiveNode(IFF)
              ├── LiteralNode("friends", ("x", "y"), negated=False)
              └── LiteralNode("friends", ("y", "x"), negated=False)
```

---

## Appendix C: Integration with System 2 Reasoning

### Symbolic Consistency Score

The constraint loss and per-rule violation statistics produced by the rule network feed
into the dual-process reasoning pipeline as a **symbolic consistency score**. This score
indicates how well the current workspace representation satisfies the domain's logical
constraints.

**Score computation:**

```python
symbolic_consistency = 1.0 - constraint_loss.detach().clamp(0, 1)
```

A high consistency score (near 1.0) means the representation satisfies most rules. A low
score (near 0.0) means many rules are violated.

### Metacognitive Router Integration

The symbolic consistency score is an input signal to the metacognitive router (see
`dual-process-reasoning/references/metacognitive-routing.md`). When consistency is low,
the router may escalate to System 2 deliberative reasoning to resolve logical conflicts.

**Integration point:**

```python
# In the dual-process pipeline
route_inputs = {
    "calibrated_conf": s1_confidence,
    "novelty": novelty_score,
    "anomaly": htm_anomaly,
    "ignition": workspace_ignition,
    "symbolic_consistency": symbolic_consistency,  # from rule network
    "remaining_budget": budget,
}
```

Low symbolic consistency acts similarly to low confidence: it increases the route score,
making System 2 engagement more likely.

### System 2 Reasoning with Symbolic Feedback

When System 2 is engaged, it can use rule violation information to guide its iterative
refinement:

1. **Violation-directed attention:** The top-K violated rules identify which logical
   constraints the current representation fails. System 2 can attend to these specific
   rules to prioritize resolving the most critical inconsistencies.

2. **Iterative constraint tightening:** At each System 2 GRU step, re-evaluate the rule
   network and check if violations have decreased. Terminate early if all violations fall
   below a threshold.

3. **Proof trace as explanation:** When System 2 resolves a conflict, the proof trace
   (Section 6) provides an interpretable record of which rules were relevant and how the
   truth values changed across reasoning steps.

### Data Flow Summary

```
Workspace Representation
        |
        v
  Rule Network (evaluate rules, compute violations)
        |
        ├── constraint_loss ──> total training loss (Mode A)
        ├── symbolic_consistency ──> metacognitive router
        ├── top-K violations ──> System 2 attention guidance
        └── proof trace ──> interpretability logs
```

This closes the loop between symbolic reasoning and the dual-process cognitive pipeline:
rules provide constraint signals that influence both training (via loss) and inference
(via routing and System 2 guidance).
