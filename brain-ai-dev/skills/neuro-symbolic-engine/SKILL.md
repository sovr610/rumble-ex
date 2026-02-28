---
name: Neuro-Symbolic Engine (Fuzzy Logic + LTN/Real Logic Grounding)
description: >-
  This skill should be used when the user asks to "implement neuro-symbolic reasoning",
  "add fuzzy logic operators", "implement LTN grounding", "add Logic Tensor Networks",
  "implement t-norm operators", "add symbolic constraints", "implement Real Logic",
  "add differentiable logic", "implement predicate grounding", "add rule network",
  "implement entity extraction", "add fuzzy quantifiers", "implement stable product semantics",
  "add symbolic constraint loss", "implement ProofWriter adapter", "add FOLIO adapter",
  "implement attention-weighted rules", "add operator bundles", "implement fuzzy implication",
  "add violation metrics", "implement symbolic reasoning trace",
  or mentions neuro-symbolic reasoning, fuzzy logic operators, LTN grounding,
  differentiable constraints, t-norms, predicate modules, rule ASTs,
  or symbolic consistency scoring in the cognitive pipeline.
version: 0.1.0
---

# Neuro-Symbolic Engine (Fuzzy Logic + LTN/Real Logic Grounding)

## Purpose

This skill standardizes the "differentiable logic layer" in the reasoning module: given entity
embeddings and a set of symbolic rules, evaluate formulas with many-valued fuzzy semantics in
[0,1], produce differentiable constraint losses for joint training with neural modules, and
optionally score queries for entailment tasks. Aligned with Logic Tensor Networks (LTN) /
"Real Logic" framing — symbols are grounded as neural computational graphs, connectives use
configurable t-norm families, and training maximizes satisfiability.

## Key Files

| Target Module | Template Asset | Purpose |
|---|---|---|
| `brain_ai/reasoning/fuzzy_operators.py` | `assets/fuzzy_operators_template.py` | T-norm/t-conorm families, negation, implication, quantifier aggregators, stable bundles |
| `brain_ai/reasoning/grounding.py` | `assets/grounding_template.py` | EntityExtractor, PredicateModule, RelationModule, Real Logic grounding |
| `brain_ai/reasoning/rule_engine.py` | `assets/rule_engine_template.py` | Rule AST, RuleNetwork, attention-weighted application, violation metrics |
| `brain_ai/reasoning/symbolic.py` | `assets/symbolic_reasoner_template.py` | SymbolicReasoner: main module wiring operators+grounding+rules |
| `brain_ai/reasoning/dataset_adapters.py` | `assets/dataset_adapters_template.py` | ProofWriter and FOLIO adapters |
| `brain_ai/config.py` (extend) | `assets/symbolic_config_template.py` | SymbolicConfig, OperatorConfig, GroundingConfig, RuleConfig, etc. |

## Public Contract

```python
forward(entities, predicates, rules, *, query=None, return_logs=False) -> SymbolicOutput
```

Input `entities` is `(B, N, D_ent)` entity embeddings from workspace slots or a learned extractor.
`predicates` is a registry of predicate/relation modules. `rules` is a compiled rule graph or AST
list. Optional `query` specifies which formula(s) to evaluate for truth estimation.

## SymbolicOutput Contract

| Field | Shape / Type | Description |
|---|---|---|
| `truth` | `(B, Q)` | Query truth values in [0,1] (None if no query) |
| `constraint_loss` | `scalar` or `(R,)` | Differentiable loss from rule violations |
| `violation_stats` | `Dict[str, Tensor]` | Per-rule/per-clause violation rates |
| `logs` | `Optional[Dict]` | Rule attention weights, top violations, proof trace (when `return_logs=True`) |

**Hard invariants**:
- Usable as **(a) standalone loss module** and **(b) feature module** feeding System 2 without changing semantics.
- All operator outputs strictly in [0,1].
- Gradients flow from `constraint_loss` through predicate groundings into upstream encoders.

## Fuzzy Operator Bundles

Configurable operator families, each a complete set of AND/OR/NOT/IMPLIES/quantifiers:

| Bundle | AND (t-norm) | OR (t-conorm) | Stability | Use Case |
|---|---|---|---|---|
| `godel` | min(x,y) | max(x,y) | Stable but sparse grads | Crisp-ish logic |
| `product` | x*y | x+y-x*y | Vanishing near 0 | Mathematically clean |
| `lukasiewicz` | max(0,x+y-1) | min(1,x+y) | Piecewise linear | Sparse activations |
| `stable_product` | product + eps projections | probabilistic sum + eps | Training-safe | **Default for training** |

Negation: `N(x) = 1 - x` (standard). Implication: S-implication and residuated options.
Quantifiers: generalized-mean aggregators with configurable exponent `p` and temperature.

See `references/fuzzy-operators.md` for operator math, stability analysis, and eps-projection details.

## Grounding Pipeline

Entity/predicate embedding contract with clean separation:

- **EntityExtractor**: workspace slots `(B,K,D_ws)` -> entity set `(B,N,D_ent)` + masks
- **PredicateModule**: unary `P(x)` -> truth in [0,1] via MLP+sigmoid
- **RelationModule**: binary `R(x,y)` -> truth in [0,1] via bilinear/NTN scoring
- **Real Logic mode** (`use_ltn=True`): constants as grounded vectors, functions as neural maps

Grounding layer must be checkpointable and expose `symbol_name -> parameters` mapping for auditing.

See `references/grounding-pipeline.md` for architecture details, typed predicates, and checkpoint format.

## Rule Network

Rules compiled into AST with literals, connectives, and quantifiers. Attention-weighted application:

```
alpha_r = softmax(f([workspace_summary; query_embedding; rule_embedding]))
constraint_loss = sum(alpha_r * violation(r))
```

Violation: `1 - truth(r)` for positive rules, aggregated by mean, soft-min, or p-mean.

See `references/rule-network.md` for AST structure, attention mechanism, and violation aggregation.

## Dataset Adapters

- **ProofWriter**: parse facts/rules into constrained logic form, build per-example theory, generate query AST + labels
- **FOLIO**: load FOL annotations, translate to internal AST, map constants to grounding domain

See `references/dataset-adapters.md` for adapter responsibilities, output schemas, and preprocessing.

## Configuration Surface

### SymbolicConfig

| Field | Default | Purpose |
|---|---|---|
| `entity_dim` | 256 | Entity embedding dimension |
| `num_predicates` | 32 | Maximum predicate slots |
| `num_relations` | 16 | Maximum relation slots |
| `use_ltn` | False | Enable Real Logic grounding mode |
| `hidden_dim` | 512 | MLP hidden width |

### OperatorConfig

| Field | Default | Purpose |
|---|---|---|
| `bundle` | `"stable_product"` | Operator family selection |
| `eps` | 1e-4 | Stability epsilon for projections |
| `quantifier_p` | 2.0 | Generalized mean exponent |
| `quantifier_temp` | 1.0 | Quantifier temperature |
| `implication_type` | `"reichenbach"` | S-implication or residuated |

### RuleConfig

| Field | Default | Purpose |
|---|---|---|
| `max_rules` | 64 | Maximum rules per theory |
| `use_attention` | True | Enable attention-weighted application |
| `violation_agg` | `"mean"` | Aggregation: `"mean"`, `"soft_min"`, `"p_mean"` |
| `constraint_weight` | 0.1 | Lambda for constraint loss in total loss |
| `warmup_steps` | 1000 | Steps before full constraint weight |

### GroundingConfig

| Field | Default | Purpose |
|---|---|---|
| `extractor_type` | `"slot_identity"` | `"slot_identity"`, `"proposal_head"` |
| `max_entities` | 32 | Maximum entities per batch item |
| `predicate_type` | `"mlp"` | `"mlp"`, `"bilinear"`, `"ntn"` |
| `relation_type` | `"bilinear"` | `"mlp"`, `"bilinear"`, `"ntn"` |

Presets: `SymbolicFullConfig.minimal()`, `.dev()`, `.production_1b()`, `.production_3b()`, `.production_7b()`.

## Done-When Gates

| Gate | Test | Threshold |
|---|---|---|
| **(a) Joint training** | Toy model: entity MLP + 2-5 rules; constraint loss decreases, predicate grads non-zero, upstream encoder grads non-zero | Loss decreasing, grads > 0 |
| **(b) Logic identity tests** | Per operator bundle: boundary tests (AND(x,1)=x, OR(x,0)=x, etc.), exact for pure bundles, tolerance for stable_product | Exact or eps-tolerance |
| **(c) Constraint violation reduction** | Synthetic FOL dataset: violation rate decreases with training, query accuracy stable/improving | Violation rate drops |

## Common Failure Modes

| Symptom | Cause | Fix |
|---|---|---|
| Vanishing gradients in product ops | Truth values near 0 | Switch to `stable_product` bundle |
| Identity tests fail for stable_product | Eps projection violates exact t-norm identity | Use tolerance-based tests, not exact |
| Constraint loss dominates task loss | Lambda too high or no warmup | Add warmup schedule, reduce `constraint_weight` |
| All rules have equal attention | Rule embeddings not trained | Initialize distinctly, check gradient flow |
| Quantifier gradients explode | Generalized mean exponent too high | Lower `quantifier_p`, add gradient clipping |
| Grounding not auditable | Symbol-to-parameter mapping missing | Expose `symbol_name -> params` dict |
| ProofWriter parse failures | Unhandled English patterns | Use structured variants, add fallback rules |
| NaN in AMP training | Low-precision fuzzy ops | Use fp32 for operator computations |

## Anti-Patterns

- **Single operator family** -- always provide configurable bundles, not hardcoded product
- **Exact identity tests for stable operators** -- stable_product has approximate identities; use tolerance
- **No warmup for constraint loss** -- symbolic constraints overpower task loss early in training
- **Opaque grounding** -- always expose symbol-to-parameter mapping for debugging
- **Non-pure operator functions** -- operators must be side-effect free for composability
- **fp16 for fuzzy computations** -- truth values need fp32 precision for correct semantics
- **Hardcoded lambda for constraint weight** -- use configurable RuleConfig, not magic numbers
- **Monolithic rule evaluation** -- evaluate rules independently, then aggregate with attention

## Additional Resources

### Reference Files

- **`references/fuzzy-operators.md`** -- T-norm/t-conorm math, operator families, stability analysis, eps projections, quantifier aggregators
- **`references/grounding-pipeline.md`** -- EntityExtractor, predicate/relation modules, Real Logic grounding, checkpoint format
- **`references/rule-network.md`** -- Rule AST structure, attention mechanism, violation metrics, interpretable logs
- **`references/dataset-adapters.md`** -- ProofWriter and FOLIO adapter specs, output schemas, preprocessing
- **`references/testing-matrix.md`** -- All test cases: identity tests, gradient sanity, joint training, synthetic FOL

### Asset Templates

- **`assets/fuzzy_operators_template.py`** -- All operator families, stable bundles, quantifier aggregators, self-test
- **`assets/grounding_template.py`** -- EntityExtractor, PredicateModule, RelationModule, grounding pipeline, self-test
- **`assets/rule_engine_template.py`** -- Rule AST, RuleNetwork, attention, violation metrics, self-test
- **`assets/symbolic_reasoner_template.py`** -- SymbolicReasoner main module, SymbolicOutput, integration, self-test
- **`assets/dataset_adapters_template.py`** -- ProofWriter/FOLIO adapters, parsing, output schema, self-test
- **`assets/symbolic_config_template.py`** -- All configs, presets, serialization, self-test

### Scripts

- **`scripts/validate_symbolic.py`** -- Runtime contract validation (joint training, identity tests, violation reduction)
- **`scripts/gen_symbolic_tests.py`** -- Generates `tests/test_neuro_symbolic.py` (~90+ test cases)
- **`scripts/operator_benchmark.py`** -- Benchmark operator throughput, gradient flow, AMP compatibility
