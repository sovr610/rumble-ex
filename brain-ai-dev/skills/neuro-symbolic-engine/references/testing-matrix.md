# Neuro-Symbolic Engine -- Testing Matrix

This document defines all test cases for the neuro-symbolic reasoning subsystem. Tests cover fuzzy operator identities, gradient sanity, entity extraction, predicate grounding, rule AST assessment, rule network attention, symbolic reasoner integration, joint training, logic identity compliance, and constraint violation reduction. Each test class maps to a specific module or integration concern. Target: ~96 test cases across 12 test classes.

---

## Test File Structure

| File | Module(s) Under Test | Purpose |
|---|---|---|
| `tests/test_neuro_symbolic.py` | `fuzzy_operators.py`, `grounding.py`, `rule_engine.py`, `symbolic.py` | All 12 test classes in a single file |

Generate via:

```bash
python brain-ai-dev/skills/neuro-symbolic-engine/scripts/gen_symbolic_tests.py
```

Run via:

```bash
python -m pytest tests/test_neuro_symbolic.py -v
python -m pytest tests/test_neuro_symbolic.py --cov=brain_ai/reasoning --cov-report=html
```

---

## Fixtures

Define the following fixtures at module scope or as `@pytest.fixture` functions. All fixtures produce CPU tensors unless noted otherwise.

| Fixture Name | Returns | Description |
|---|---|---|
| `synthetic_entities` | `Tensor (B=4, N=8, D_ent=256)` | Random entity embeddings drawn from `torch.randn`, seeded for reproducibility |
| `simple_predicates` | `Dict[str, PredicateModule]` | 2--3 MLP-based unary predicate modules with `D_ent=256`, `hidden_dim=128` |
| `simple_relations` | `Dict[str, RelationModule]` | 1--2 bilinear relation modules with `D_ent=256` |
| `simple_rules` | `List[RuleAST]` | 3--5 rules as compiled AST objects covering conjunction, disjunction, implication, and one quantified formula |
| `operator_bundles` | `List[str]` | `["godel", "product", "lukasiewicz", "stable_product"]` for parametrized tests |
| `toy_symbolic_reasoner` | `SymbolicReasoner` | Minimal reasoner wired with `simple_predicates`, `simple_relations`, `simple_rules`, `OperatorConfig(bundle="stable_product")` |
| `synthetic_fol_dataset` | `Dict` | Small dataset with 50 examples, each containing entities `(N=8, D_ent=256)`, ground-truth predicate labels, ground-truth relation labels, and 5 rules with known satisfiability |

### Fixture Implementation Notes

- Seed all random generation with `torch.manual_seed(42)` for deterministic fixtures.
- `simple_predicates` fixture returns a `nn.ModuleDict` so parameters are trackable.
- `simple_rules` fixture returns AST objects constructed from Python dicts (see `TestRuleAST.test_rule_compilation_from_dict`).
- `synthetic_fol_dataset` includes a `DataLoader`-compatible iterator yielding `(entities, predicate_targets, relation_targets, rules)` tuples.

---

## Skip Conditions

| Condition | Decorator | Affected Tests |
|---|---|---|
| CUDA unavailable | `@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")` | `test_operators_no_nan_under_amp` |
| Minimal config only | `@pytest.mark.skipif(os.environ.get("TEST_MINIMAL") == "1", reason="Skipped in minimal config")` | `test_proposal_head_shape`, `test_proposal_head_confidence`, `test_entity_deduplication` |
| Full test suite only | `@pytest.mark.skipif(os.environ.get("TEST_FULL") != "1", reason="Full suite only")` | `test_ntn_relation_shape` |

---

## 1. TestFuzzyOperatorIdentities (~12 tests)

Verify algebraic identities for each of the four operator bundles. Pure bundles (godel, product, lukasiewicz) use exact assertions. The `stable_product` bundle uses tolerance-based assertions with `eps=1e-3`.

Test at representative points: `x in {0.0, 0.2, 0.5, 0.8, 1.0}`.

| Test Method | Bundle | Logic | Assertion |
|---|---|---|---|
| `test_godel_and_identity` | godel | AND(x, 1) = x; AND(x, 0) = 0 | `torch.allclose(result, expected, atol=0)` |
| `test_godel_or_identity` | godel | OR(x, 0) = x; OR(x, 1) = 1 | `torch.allclose(result, expected, atol=0)` |
| `test_godel_negation_involution` | godel | NOT(NOT(x)) = x | `torch.allclose(result, expected, atol=0)` |
| `test_product_and_identity` | product | AND(x, 1) = x; AND(x, 0) = 0 | `torch.allclose(result, expected, atol=0)` |
| `test_product_or_identity` | product | OR(x, 0) = x; OR(x, 1) = 1 | `torch.allclose(result, expected, atol=0)` |
| `test_product_negation_involution` | product | NOT(NOT(x)) = x | `torch.allclose(result, expected, atol=0)` |
| `test_lukasiewicz_and_identity` | lukasiewicz | AND(x, 1) = x; AND(x, 0) = 0 | `torch.allclose(result, expected, atol=0)` |
| `test_lukasiewicz_or_identity` | lukasiewicz | OR(x, 0) = x; OR(x, 1) = 1 | `torch.allclose(result, expected, atol=0)` |
| `test_lukasiewicz_negation_involution` | lukasiewicz | NOT(NOT(x)) = x | `torch.allclose(result, expected, atol=0)` |
| `test_stable_product_and_identity` | stable_product | AND(x, 1) ~ x; AND(x, 0) ~ 0 | `torch.allclose(result, expected, atol=1e-3)` |
| `test_stable_product_or_identity` | stable_product | OR(x, 0) ~ x; OR(x, 1) ~ 1 | `torch.allclose(result, expected, atol=1e-3)` |
| `test_stable_product_negation_involution` | stable_product | NOT(NOT(x)) ~ x | `torch.allclose(result, expected, atol=1e-3)` |

### Implementation Pattern

```python
@pytest.mark.parametrize("x_val", [0.0, 0.2, 0.5, 0.8, 1.0])
def test_godel_and_identity(self, x_val):
    ops = get_operator_bundle("godel")
    x = torch.tensor(x_val)
    one = torch.tensor(1.0)
    zero = torch.tensor(0.0)
    assert torch.allclose(ops.AND(x, one), x, atol=0)
    assert torch.allclose(ops.AND(x, zero), zero, atol=0)
```

For `stable_product`, replace `atol=0` with `atol=1e-3`.

---

## 2. TestDeMorganDistributivity (~6 tests)

Verify De Morgan's laws where each bundle guarantees them. NOT all bundles satisfy De Morgan exactly; the table below documents which do.

| Bundle | De Morgan AND Guaranteed | De Morgan OR Guaranteed |
|---|---|---|
| godel | No (min/max do not distribute through 1-x) | No |
| product | Yes | Yes |
| lukasiewicz | Yes | Yes |
| stable_product | Approximate (within eps) | Approximate (within eps) |

| Test Method | Bundle | Law | Assertion |
|---|---|---|---|
| `test_product_demorgan_and` | product | NOT(AND(x,y)) = OR(NOT(x), NOT(y)) | `torch.allclose(..., atol=0)` |
| `test_product_demorgan_or` | product | NOT(OR(x,y)) = AND(NOT(x), NOT(y)) | `torch.allclose(..., atol=0)` |
| `test_lukasiewicz_demorgan_and` | lukasiewicz | NOT(AND(x,y)) = OR(NOT(x), NOT(y)) | `torch.allclose(..., atol=0)` |
| `test_lukasiewicz_demorgan_or` | lukasiewicz | NOT(OR(x,y)) = AND(NOT(x), NOT(y)) | `torch.allclose(..., atol=0)` |
| `test_stable_product_demorgan_and` | stable_product | NOT(AND(x,y)) ~ OR(NOT(x), NOT(y)) | `torch.allclose(..., atol=1e-3)` |
| `test_stable_product_demorgan_or` | stable_product | NOT(OR(x,y)) ~ AND(NOT(x), NOT(y)) | `torch.allclose(..., atol=1e-3)` |

### Implementation Pattern

```python
def test_product_demorgan_and(self):
    ops = get_operator_bundle("product")
    x = torch.tensor([0.2, 0.5, 0.8])
    y = torch.tensor([0.3, 0.6, 0.9])
    lhs = ops.NOT(ops.AND(x, y))
    rhs = ops.OR(ops.NOT(x), ops.NOT(y))
    assert torch.allclose(lhs, rhs, atol=0)
```

Note: Do NOT include godel De Morgan tests because min/max with standard negation `1 - x` does not satisfy De Morgan's laws. Document this in a comment within the test file.

---

## 3. TestOperatorGradients (~8 tests)

Verify that all operator outputs produce non-zero gradients in the interior of [0, 1] and do not produce NaN under AMP.

| Test Method | Bundle / Target | What Is Checked | Assertion |
|---|---|---|---|
| `test_godel_and_gradient_nonzero` | godel | `d/dx AND(x, y)` for x in [0.2, 0.8] | `x.grad is not None and x.grad.abs().sum() > 0` -- NOTE: godel (min) has subgradient only; accept zero grad at ties |
| `test_product_and_gradient_nonzero` | product | `d/dx AND(x, y)` for x in [0.2, 0.8] | `x.grad.abs().sum() > 0` |
| `test_lukasiewicz_or_gradient_nonzero` | lukasiewicz | `d/dx OR(x, y)` for x in [0.2, 0.8] | `x.grad.abs().sum() > 0` |
| `test_stable_product_and_gradient_nonzero` | stable_product | `d/dx AND(x, y)` for x in [0.2, 0.8] | `x.grad.abs().sum() > 0` |
| `test_stable_product_or_gradient_nonzero` | stable_product | `d/dx OR(x, y)` for x in [0.2, 0.8] | `x.grad.abs().sum() > 0` |
| `test_stable_product_gradient_at_boundary` | stable_product | Gradients remain non-zero near 0.01 and 0.99 | `x.grad.abs().sum() > 0` (this is the key advantage of stable_product) |
| `test_quantifier_gradient_no_explosion` | all | `d/dx FORALL(x)` with generalized mean; grad magnitude bounded | `x.grad.abs().max() < 100.0` |
| `test_operators_no_nan_under_amp` | all | All operators produce valid outputs under `torch.cuda.amp.autocast` | `not torch.isnan(result).any()` and `not torch.isinf(result).any()` |

### Implementation Pattern

```python
def test_product_and_gradient_nonzero(self):
    ops = get_operator_bundle("product")
    x = torch.tensor([0.2, 0.5, 0.8], requires_grad=True)
    y = torch.tensor([0.3, 0.6, 0.7])
    result = ops.AND(x, y)
    result.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0
```

For the AMP test, mark with `@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")` and wrap the forward call in `torch.cuda.amp.autocast()`.

---

## 4. TestQuantifierAggregators (~8 tests)

Verify the generalized-mean quantifier aggregators for FORALL and EXISTS. These operate on a 1-D truth vector and aggregate along a dimension.

| Test Method | Operation | Input | Assertion |
|---|---|---|---|
| `test_forall_all_true` | FORALL | `torch.ones(10)` | `result > 0.99` |
| `test_forall_one_false` | FORALL | `torch.tensor([1,1,1,0,1,1,1,1,1,1])` | `result < 0.5` |
| `test_exists_one_true` | EXISTS | `torch.tensor([0,0,0,1,0,0,0,0,0,0])` | `result > 0.3` |
| `test_exists_all_false` | EXISTS | `torch.zeros(10)` | `result < 0.01` |
| `test_quantifier_masking` | FORALL with mask | `values=[0.9,0.9,0.1]`, `mask=[1,1,0]` | Result computed only over masked-in elements; `result > 0.8` |
| `test_quantifier_p_sensitivity` | FORALL | Same input, `p=2` vs `p=10` | Higher p produces result closer to `min(values)` |
| `test_quantifier_temperature` | FORALL | Same input, `temp=0.5` vs `temp=2.0` | Lower temperature produces sharper (more extreme) aggregation |
| `test_quantifier_gradient_flow` | FORALL | `values` with `requires_grad=True` | `values.grad is not None and values.grad.abs().sum() > 0` |

### Implementation Pattern

```python
def test_forall_all_true(self):
    ops = get_operator_bundle("stable_product")
    values = torch.ones(10)
    result = ops.FORALL(values, dim=0)
    assert result.item() > 0.99

def test_quantifier_p_sensitivity(self):
    values = torch.tensor([0.9, 0.8, 0.3, 0.7])
    result_p2 = generalized_mean(values, p=2.0, dim=0)
    result_p10 = generalized_mean(values, p=10.0, dim=0)
    true_min = values.min()
    # Higher p should be closer to min
    assert abs(result_p10 - true_min) < abs(result_p2 - true_min)
```

---

## 5. TestEntityExtraction (~8 tests)

Verify the EntityExtractor pipeline that converts workspace slots into entity embeddings.

| Test Method | Extractor Type | What Is Checked | Assertion |
|---|---|---|---|
| `test_slot_identity_shape` | `slot_identity` | Input `(B=4, K=16, D_ws=4096)` produces `(B=4, K=16, D_ent=256)` | `output.shape == (4, 16, 256)` |
| `test_slot_identity_mask` | `slot_identity` | Mask `(B=4, K=16)` propagated unchanged through extraction | `output_mask.shape == input_mask.shape` and values match |
| `test_proposal_head_shape` | `proposal_head` | Input `(B=4, K=16, D_ws=4096)` produces `(B=4, N, D_ent=256)` with `N <= K` | `output.shape[1] <= 16` and `output.shape[2] == 256` |
| `test_proposal_head_confidence` | `proposal_head` | Confidence scores in `[0, 1]` | `0.0 <= confidence.min()` and `confidence.max() <= 1.0` |
| `test_entity_projection_gradient` | `slot_identity` | Gradients flow from output back through projection | `projection.weight.grad is not None` and `grad.abs().sum() > 0` |
| `test_variable_length_entities` | `slot_identity` | Batch with different valid entity counts handled via mask | No error; masked positions produce zero or ignored output |
| `test_entity_extractor_factory` | both | `create_entity_extractor(config)` returns correct subclass | `isinstance(extractor, SlotIdentityExtractor)` or `isinstance(extractor, ProposalHeadExtractor)` depending on config |
| `test_entity_deduplication` | `proposal_head` | Near-duplicate proposals (cosine similarity > 0.95) are merged | `output.shape[1] < input_proposals` when duplicates present |

### Implementation Pattern

```python
def test_slot_identity_shape(self):
    extractor = create_entity_extractor(
        GroundingConfig(extractor_type="slot_identity", max_entities=16)
    )
    ws_slots = torch.randn(4, 16, 4096)
    entities, mask = extractor(ws_slots)
    assert entities.shape == (4, 16, 256)
```

---

## 6. TestPredicateRelation (~10 tests)

Verify predicate and relation modules: output range, batch handling, gradient flow, registry lookup, and grounding audit.

| Test Method | Module | What Is Checked | Assertion |
|---|---|---|---|
| `test_unary_predicate_output_range` | `PredicateModule` | Output of `P(x)` in `[0, 1]` for random entities | `0.0 <= output.min()` and `output.max() <= 1.0` |
| `test_unary_predicate_batch` | `PredicateModule` | Input `(B=4, N=8, D_ent=256)` produces `(B=4, N=8)` or `(B=4, N)` | `output.shape == (4, 8)` |
| `test_binary_relation_output_range` | `RelationModule` | Output of `R(x, y)` in `[0, 1]` | `0.0 <= output.min()` and `output.max() <= 1.0` |
| `test_binary_relation_batch` | `RelationModule` | Input pair `(B=4, N=8, D_ent=256)` produces `(B=4, N=8, N=8)` | `output.shape == (4, 8, 8)` |
| `test_bilinear_relation_symmetry` | `RelationModule(symmetric=True)` | `R(x, y) == R(y, x)` when symmetric option enabled | `torch.allclose(R_xy, R_yx, atol=1e-6)` |
| `test_ntn_relation_shape` | `NTNRelationModule` | NTN scoring with `k` slices produces correct output shape | `output.shape == (B, N, N)` |
| `test_predicate_registry_lookup` | Registry | `registry["IsAnimal"]` returns correct `PredicateModule` | `isinstance(result, PredicateModule)` |
| `test_relation_registry_lookup` | Registry | `registry["PartOf"]` returns correct `RelationModule` | `isinstance(result, RelationModule)` |
| `test_predicate_gradient_flow` | `PredicateModule` | Gradients flow from predicate output through entity embeddings | `entities.grad is not None` and `entities.grad.abs().sum() > 0` |
| `test_grounding_audit` | Grounding pipeline | `symbol_to_params()` exposes correct mapping from symbol names to parameter tensors | `"IsAnimal" in mapping` and `isinstance(mapping["IsAnimal"], list)` and all elements are `nn.Parameter` |

### Implementation Pattern

```python
def test_unary_predicate_output_range(self, simple_predicates, synthetic_entities):
    pred = simple_predicates["P1"]
    entities = synthetic_entities  # (4, 8, 256)
    # Run P1 on each entity
    output = pred(entities)  # (4, 8)
    assert output.min() >= 0.0
    assert output.max() <= 1.0

def test_predicate_gradient_flow(self, simple_predicates):
    pred = simple_predicates["P1"]
    entities = torch.randn(2, 4, 256, requires_grad=True)
    output = pred(entities)
    output.sum().backward()
    assert entities.grad is not None
    assert entities.grad.abs().sum() > 0
```

---

## 7. TestRuleAST (~8 tests)

Verify that rule AST nodes compute correctly against predicate modules and compose through fuzzy connectives.

| Test Method | AST Node | What Is Checked | Assertion |
|---|---|---|---|
| `test_literal_computation` | `Literal("P1", var="x")` | Computes `P1(x)` against predicate module; truth in `[0, 1]` | `0.0 <= truth <= 1.0` |
| `test_negation_computation` | `Negation(Literal("P1", var="x"))` | `NOT(P1(x))` = `1 - P1(x)` | `torch.allclose(result, 1 - literal_result, atol=1e-6)` |
| `test_conjunction_computation` | `Conjunction(lit_P1, lit_P2)` | `P1(x) AND P2(x)` uses the configured AND operator | `0.0 <= result <= 1.0` and matches manual `ops.AND(p1, p2)` |
| `test_disjunction_computation` | `Disjunction(lit_P1, lit_P2)` | `P1(x) OR P2(x)` uses the configured OR operator | `0.0 <= result <= 1.0` and matches manual `ops.OR(p1, p2)` |
| `test_implication_computation` | `Implication(lit_P1, lit_P2)` | `P1(x) IMPLIES P2(x)` uses the configured IMPLIES operator | `0.0 <= result <= 1.0` and matches manual `ops.IMPLIES(p1, p2)` |
| `test_universal_quantifier` | `Forall("x", Literal("P1", var="x"))` | `FORALL_x P1(x)` aggregates over all entities | Result is a scalar in `[0, 1]` |
| `test_nested_quantifier` | `Forall("x", Exists("y", Literal("R1", vars=["x","y"])))` | `FORALL_x EXISTS_y R1(x,y)` correctly scoped: inner quantifier aggregates over y, outer over x | Result is a scalar; intermediate shape `(N,)` after EXISTS |
| `test_rule_compilation_from_dict` | Compiler | Dict `{"type": "forall", "var": "x", "body": {"type": "implies", ...}}` compiles to AST and produces identical results to hand-built AST | `torch.allclose(dict_result, manual_result, atol=1e-6)` |

### Implementation Pattern

```python
def test_literal_computation(self, simple_predicates, synthetic_entities):
    lit = Literal(predicate_name="P1", var="x")
    binding = {"x": synthetic_entities[0]}  # First batch item, (N, D_ent)
    truth = lit.run(predicates=simple_predicates, bindings=binding, ops=get_operator_bundle("stable_product"))
    assert truth.min() >= 0.0
    assert truth.max() <= 1.0

def test_nested_quantifier(self, simple_relations, synthetic_entities):
    # FORALL_x EXISTS_y R1(x, y)
    inner = Exists("y", Literal("R1", vars=["x", "y"]))
    outer = Forall("x", inner)
    entities = synthetic_entities[0]  # (N, D_ent)
    ops = get_operator_bundle("stable_product")
    result = outer.run(
        relations=simple_relations,
        entity_set=entities,
        ops=ops,
    )
    assert result.ndim == 0  # scalar
    assert 0.0 <= result.item() <= 1.0
```

---

## 8. TestRuleNetwork (~8 tests)

Verify the RuleNetwork that applies attention-weighted rules and computes violation losses.

| Test Method | What Is Checked | Assertion |
|---|---|---|
| `test_attention_weights_sum_to_one` | `softmax(attention_logits)` sums to 1 over rules dimension | `torch.allclose(weights.sum(dim=-1), torch.ones(B), atol=1e-5)` |
| `test_attention_with_query` | Query embedding shifts attention distribution compared to no-query | `not torch.allclose(weights_with_query, weights_without_query)` |
| `test_attention_without_query` | Works with workspace summary only; no error | No exception raised; output shape correct |
| `test_violation_computation` | `violation = 1 - truth` for positive rules | `torch.allclose(violation, 1 - truth_values, atol=1e-6)` |
| `test_constraint_loss_aggregation_mean` | Mean aggregation over per-rule violations | `torch.allclose(loss, violations.mean(), atol=1e-6)` |
| `test_constraint_loss_aggregation_soft_min` | Soft-min aggregation: `-log(sum(exp(-violations/temp)))` | Loss < mean aggregation (soft-min focuses on worst violations) |
| `test_constraint_loss_with_attention` | Attention-weighted loss: `sum(alpha * violation)` | Loss is a weighted combination; varies with attention |
| `test_rule_log_extraction` | Top-K violated rules extracted with names and violation magnitudes | `len(logs["top_violated"]) == K` and each entry has `"rule_name"` and `"violation"` keys |

### Implementation Pattern

```python
def test_attention_weights_sum_to_one(self, toy_symbolic_reasoner, synthetic_entities):
    reasoner = toy_symbolic_reasoner
    entities = synthetic_entities
    output = reasoner.forward(entities, predicates=..., rules=..., return_logs=True)
    weights = output.logs["attention_weights"]  # (B, R)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(weights.shape[0]), atol=1e-5)

def test_constraint_loss_aggregation_mean(self):
    violations = torch.tensor([0.3, 0.1, 0.5, 0.2])
    loss = aggregate_violations(violations, method="mean")
    assert torch.allclose(loss, violations.mean(), atol=1e-6)
```

---

## 9. TestSymbolicReasoner (~10 tests)

Verify the top-level `SymbolicReasoner` module that wires operators, grounding, and rules together.

| Test Method | Mode | What Is Checked | Assertion |
|---|---|---|---|
| `test_forward_no_query` | No query | Returns `SymbolicOutput` with `truth=None`, `constraint_loss` is a scalar | `output.truth is None` and `output.constraint_loss.ndim == 0` |
| `test_forward_with_query` | With query | Returns `SymbolicOutput` with `truth` of shape `(B, Q)` | `output.truth.shape == (B, Q)` and all values in `[0, 1]` |
| `test_output_contract` | Any | All `SymbolicOutput` fields present and correct types | `hasattr(output, field)` for each of `truth`, `constraint_loss`, `violation_stats`, `logs` |
| `test_constraint_loss_backprop` | Training | Gradients flow from `constraint_loss` to predicate parameters | `pred_param.grad is not None` and `pred_param.grad.abs().sum() > 0` |
| `test_upstream_gradient_flow` | Training | Gradients flow from `constraint_loss` to entity extractor and upstream encoder | `encoder_param.grad is not None` |
| `test_return_logs_false` | `return_logs=False` | `output.logs is None` | `output.logs is None` |
| `test_return_logs_true` | `return_logs=True` | `output.logs` contains `attention_weights`, `violations`, `top_violated` | All three keys present |
| `test_standalone_loss_mode` | Loss module | Reasoner used as loss-only module (no query, no feature output); loss is differentiable scalar | `output.constraint_loss.requires_grad is True` |
| `test_feature_mode` | Feature module | Reasoner outputs truth values usable as features for System 2 | `output.truth` is a valid tensor that can be concatenated with workspace features |
| `test_deterministic_forward` | Mode: inference | Same input produces same output on two forward passes (no stochastic elements during inference) | `torch.allclose(out1.truth, out2.truth, atol=0)` and `torch.allclose(out1.constraint_loss, out2.constraint_loss, atol=0)` |

### Implementation Pattern

```python
def test_output_contract(self, toy_symbolic_reasoner, synthetic_entities, simple_predicates, simple_rules):
    reasoner = toy_symbolic_reasoner
    output = reasoner(
        entities=synthetic_entities,
        predicates=simple_predicates,
        rules=simple_rules,
        query=None,
        return_logs=True,
    )
    assert hasattr(output, "truth")
    assert hasattr(output, "constraint_loss")
    assert hasattr(output, "violation_stats")
    assert hasattr(output, "logs")
    assert isinstance(output.constraint_loss, torch.Tensor)
    assert isinstance(output.violation_stats, dict)

def test_deterministic_forward(self, toy_symbolic_reasoner, synthetic_entities, simple_predicates, simple_rules):
    reasoner = toy_symbolic_reasoner
    reasoner.eval()
    query = simple_rules[:2]  # Use first 2 rules as queries
    out1 = reasoner(entities=synthetic_entities, predicates=simple_predicates, rules=simple_rules, query=query)
    out2 = reasoner(entities=synthetic_entities, predicates=simple_predicates, rules=simple_rules, query=query)
    assert torch.allclose(out1.truth, out2.truth, atol=0)
    assert torch.allclose(out1.constraint_loss, out2.constraint_loss, atol=0)
```

---

## 10. TestJointTraining (~6 tests) -- Done-When Gate (a)

Verify that the symbolic reasoning module trains jointly with a neural encoder, that constraint loss decreases, and that gradients propagate correctly.

| Test Method | What Is Checked | Assertion |
|---|---|---|
| `test_toy_model_constraint_loss_decreases` | 5-step training with SGD on a toy model (encoder MLP + reasoner + 3 rules); constraint loss at step 5 < step 1 | `losses[-1] < losses[0]` |
| `test_predicate_params_change` | Predicate parameters updated by gradient descent after 5 steps | `not torch.allclose(params_before, params_after)` |
| `test_upstream_encoder_grads_nonzero` | Upstream encoder MLP receives non-zero gradients from `constraint_loss.backward()` | `encoder.weight.grad.abs().sum() > 0` |
| `test_warmup_schedule` | Constraint weight ramps from 0 to `constraint_weight` over `warmup_steps` | At step 0: `lambda_t == 0`; at `warmup_steps`: `lambda_t == constraint_weight` |
| `test_no_nan_during_training` | 10 training steps produce no NaN in any tensor (parameters, gradients, outputs) | `not any(torch.isnan(p).any() for p in model.parameters())` and same for `.grad` |
| `test_joint_loss_composition` | `L_total = L_task + lambda * L_symbolic` matches manual computation | `torch.allclose(total_loss, task_loss + lam * symbolic_loss, atol=1e-6)` |

### Toy Model Architecture

```
encoder_mlp(D_in=128, D_out=256) -> entities (B, N, 256) -> SymbolicReasoner -> constraint_loss
                                                          -> task_head(256, num_classes) -> task_loss
L_total = L_task + lambda * constraint_loss
```

### Implementation Pattern

```python
def test_toy_model_constraint_loss_decreases(self):
    torch.manual_seed(42)
    encoder = nn.Linear(128, 256 * 8)  # Produce 8 entities of dim 256
    reasoner = create_symbolic_reasoner(entity_dim=256, bundle="stable_product")
    predicates = nn.ModuleDict({
        "P1": PredicateModule(256, 128),
        "P2": PredicateModule(256, 128),
    })
    rules = compile_rules([{
        "type": "forall", "var": "x",
        "body": {
            "type": "implies",
            "left": {"type": "literal", "pred": "P1", "var": "x"},
            "right": {"type": "literal", "pred": "P2", "var": "x"},
        }
    }])

    all_params = (
        list(encoder.parameters())
        + list(reasoner.parameters())
        + list(predicates.parameters())
    )
    optimizer = torch.optim.SGD(all_params, lr=0.01)
    losses = []
    for step in range(5):
        x = torch.randn(4, 128)
        entities = encoder(x).view(4, 8, 256)
        output = reasoner(entities=entities, predicates=predicates, rules=rules)
        loss = output.constraint_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    assert losses[-1] < losses[0]
```

---

## 11. TestLogicIdentities (~6 tests) -- Done-When Gate (b)

Comprehensive boundary and identity tests across all operator bundles. This class consolidates and extends the per-bundle tests from TestFuzzyOperatorIdentities into systematic grid-point checking.

| Test Method | What Is Checked | Assertion |
|---|---|---|
| `test_all_bundles_boundary_conditions` | For each bundle: AND(0,0)=0, AND(1,1)=1, OR(0,0)=0, OR(1,1)=1, NOT(0)=1, NOT(1)=0 | Exact for all bundles (boundary behavior is shared) |
| `test_pure_bundles_exact_identity` | For godel, product, lukasiewicz: AND(x,1)=x, OR(x,0)=x at all grid points | `torch.allclose(..., atol=0)` |
| `test_stable_product_approximate_identity` | For stable_product: AND(x,1) ~ x, OR(x,0) ~ x at all grid points | `torch.allclose(..., atol=1e-3)` |
| `test_implication_boundary_per_bundle` | For each bundle: IMPLIES(1, y) = y, IMPLIES(0, y) = 1 at grid points | Exact for pure bundles; `atol=1e-3` for stable_product |
| `test_identity_at_grid_points` | Check identities at the grid `x in {0.0, 0.1, 0.2, ..., 1.0}` (11 points) | All 11 points pass |
| `test_commutativity_where_guaranteed` | AND(x, y) = AND(y, x) and OR(x, y) = OR(y, x) for all bundles | `torch.allclose(op(x,y), op(y,x), atol=1e-6)` for all bundles (commutativity is a t-norm axiom) |

### Grid Points

```python
grid = torch.linspace(0.0, 1.0, 11)  # [0.0, 0.1, 0.2, ..., 1.0]
```

### Implication Boundary Details

| Bundle | IMPLIES(1, y) | IMPLIES(0, y) |
|---|---|---|
| godel | `torch.where(1 <= y, 1, y)` which equals y for y in [0, 1) and 1 for y = 1; simplifies to y for all y in [0, 1] | 1 (since 0 <= y is always true) |
| product (reichenbach) | `1 - 1 + 1 * y` = y | `1 - 0 + 0 * y` = 1 |
| lukasiewicz | `min(1, 1 - 1 + y)` = min(1, y) = y | `min(1, 1 - 0 + y)` = min(1, 1 + y) = 1 |
| stable_product | ~ y (within eps) | ~ 1 (within eps) |

### Implementation Pattern

```python
@pytest.mark.parametrize("bundle", ["godel", "product", "lukasiewicz", "stable_product"])
def test_all_bundles_boundary_conditions(self, bundle):
    ops = get_operator_bundle(bundle)
    zero = torch.tensor(0.0)
    one = torch.tensor(1.0)
    assert torch.allclose(ops.AND(zero, zero), zero, atol=1e-6)
    assert torch.allclose(ops.AND(one, one), one, atol=1e-6)
    assert torch.allclose(ops.OR(zero, zero), zero, atol=1e-6)
    assert torch.allclose(ops.OR(one, one), one, atol=1e-6)
    assert torch.allclose(ops.NOT(zero), one, atol=1e-6)
    assert torch.allclose(ops.NOT(one), zero, atol=1e-6)

@pytest.mark.parametrize("bundle", ["godel", "product", "lukasiewicz"])
def test_pure_bundles_exact_identity(self, bundle):
    ops = get_operator_bundle(bundle)
    grid = torch.linspace(0.0, 1.0, 11)
    one = torch.ones_like(grid)
    zero = torch.zeros_like(grid)
    assert torch.allclose(ops.AND(grid, one), grid, atol=0)
    assert torch.allclose(ops.OR(grid, zero), grid, atol=0)
```

---

## 12. TestConstraintViolationReduction (~6 tests) -- Done-When Gate (c)

Verify that training on a synthetic FOL dataset reduces constraint violation rates and maintains or improves query accuracy.

| Test Method | What Is Checked | Assertion |
|---|---|---|
| `test_synthetic_fol_violation_decreases` | Train for 20 steps on `synthetic_fol_dataset`; mean violation rate at step 20 < step 1 | `violation_rates[-1] < violation_rates[0]` |
| `test_query_accuracy_improves_or_stable` | Query accuracy (truth value vs ground-truth label) does not degrade during constraint training | `accuracy_after >= accuracy_before - 0.05` (allow 5% tolerance) |
| `test_top_violated_rules_logged` | After training, logs show which rules are most violated with names and magnitudes | `len(logs["top_violated"]) > 0` and each entry has `"rule_name"` and `"violation"` |
| `test_before_after_comparison` | Collect violation stats before and after 20-step training; after < before | `stats_after["mean_violation"] < stats_before["mean_violation"]` |
| `test_predicate_grounding_improves` | Predicate outputs become more accurate vs ground-truth labels after training | `accuracy_after > accuracy_before` |
| `test_constraint_and_task_loss_balance` | Both task loss and constraint loss decrease (neither dominates the other) | `task_losses[-1] < task_losses[0]` and `constraint_losses[-1] < constraint_losses[0]` |

### Synthetic FOL Dataset Specification

| Property | Value |
|---|---|
| Number of examples | 50 |
| Entities per example | N = 8, D_ent = 256 |
| Predicates | 3 unary: `HasFeatureA`, `HasFeatureB`, `IsCategory` |
| Relations | 1 binary: `SimilarTo` |
| Rules | 5 rules (see below) |
| Ground truth | Deterministic labels derived from entity features |

### Synthetic Rules

| Rule ID | Formula | Natural Language |
|---|---|---|
| R1 | `FORALL x: HasFeatureA(x) IMPLIES IsCategory(x)` | Everything with feature A is in the category |
| R2 | `FORALL x: NOT(HasFeatureB(x)) OR IsCategory(x)` | Feature B implies category (disjunctive form) |
| R3 | `FORALL x, y: SimilarTo(x, y) IMPLIES (IsCategory(x) EQUIV IsCategory(y))` | Similar entities share category |
| R4 | `EXISTS x: HasFeatureA(x) AND HasFeatureB(x)` | At least one entity has both features |
| R5 | `FORALL x: IsCategory(x) IMPLIES (HasFeatureA(x) OR HasFeatureB(x))` | Category members have at least one feature |

### Implementation Pattern

```python
def test_synthetic_fol_violation_decreases(self, synthetic_fol_dataset, toy_symbolic_reasoner):
    reasoner = toy_symbolic_reasoner
    dataset = synthetic_fol_dataset
    optimizer = torch.optim.Adam(reasoner.parameters(), lr=1e-3)

    violation_rates = []
    for step in range(20):
        batch = dataset[step % len(dataset)]
        output = reasoner(
            entities=batch["entities"],
            predicates=batch["predicates"],
            rules=batch["rules"],
        )
        loss = output.constraint_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        violation_rates.append(output.violation_stats["mean_violation"].item())

    assert violation_rates[-1] < violation_rates[0]
```

---

## Assertion Pattern Summary

| Pattern | Use Case | Example |
|---|---|---|
| Truth value range | Any operator or predicate output | `assert 0.0 <= truth <= 1.0` |
| Exact identity | Pure bundles (godel, product, lukasiewicz) | `assert torch.allclose(result, expected, atol=0)` |
| Approximate identity | stable_product bundle | `assert torch.allclose(result, expected, atol=1e-3)` |
| Gradient non-zero | Any trainable module | `assert param.grad is not None and param.grad.abs().sum() > 0` |
| Gradient bounded | Quantifier aggregators | `assert param.grad.abs().max() < 100.0` |
| No NaN | Any forward/backward pass | `assert not torch.isnan(tensor).any()` |
| No Inf | AMP compatibility | `assert not torch.isinf(tensor).any()` |
| Loss decreasing | Joint training, violation reduction | `assert losses[-1] < losses[0]` |
| Shape correct | Entity extraction, predicate output | `assert output.shape == expected_shape` |
| Dictionary keys present | SymbolicOutput, logs | `assert "key" in output_dict` |

---

## Coverage Targets

| Source File | Line Coverage Target | Branch Coverage Target | Key Areas |
|---|---|---|---|
| `brain_ai/reasoning/fuzzy_operators.py` | >= 90% | >= 85% | All 4 operator bundles, quantifier aggregators, eps projections |
| `brain_ai/reasoning/grounding.py` | >= 90% | >= 85% | EntityExtractor (both types), PredicateModule, RelationModule, NTN, registry, audit |
| `brain_ai/reasoning/rule_engine.py` | >= 90% | >= 85% | All AST node types, RuleNetwork attention, all 3 violation aggregation methods, log extraction |
| `brain_ai/reasoning/symbolic.py` | >= 90% | >= 85% | Forward with/without query, loss mode, feature mode, logs on/off, determinism |

### Coverage Verification

```bash
python -m pytest tests/test_neuro_symbolic.py \
    --cov=brain_ai/reasoning/fuzzy_operators \
    --cov=brain_ai/reasoning/grounding \
    --cov=brain_ai/reasoning/rule_engine \
    --cov=brain_ai/reasoning/symbolic \
    --cov-report=term-missing \
    --cov-fail-under=85
```

---

## Done-When Gate Mapping

Each done-when gate from the SKILL.md has a dedicated test class and clear pass criteria.

| Gate | Test Class | Pass Criteria | Number of Tests |
|---|---|---|---|
| **(a) Joint training** | `TestJointTraining` | Constraint loss decreases over 5 steps; predicate gradients non-zero; upstream encoder gradients non-zero; no NaN in 10 steps; warmup schedule correct; loss composition correct | 6 |
| **(b) Logic identity tests** | `TestLogicIdentities` | All boundary conditions pass for all 4 bundles; exact identity for pure bundles; tolerance-based for stable_product; implication boundaries correct; commutativity verified; 11-point grid passes | 6 |
| **(c) Constraint violation reduction** | `TestConstraintViolationReduction` | Violation rate decreases on synthetic FOL; query accuracy stable; top violated rules logged; before/after comparison shows improvement; predicate grounding improves; task and constraint losses both decrease | 6 |

---

## Test Count Summary

| Test Class | Count | Primary Module |
|---|---|---|
| TestFuzzyOperatorIdentities | 12 | `fuzzy_operators.py` |
| TestDeMorganDistributivity | 6 | `fuzzy_operators.py` |
| TestOperatorGradients | 8 | `fuzzy_operators.py` |
| TestQuantifierAggregators | 8 | `fuzzy_operators.py` |
| TestEntityExtraction | 8 | `grounding.py` |
| TestPredicateRelation | 10 | `grounding.py` |
| TestRuleAST | 8 | `rule_engine.py` |
| TestRuleNetwork | 8 | `rule_engine.py` |
| TestSymbolicReasoner | 10 | `symbolic.py` |
| TestJointTraining | 6 | Integration (gate a) |
| TestLogicIdentities | 6 | Integration (gate b) |
| TestConstraintViolationReduction | 6 | Integration (gate c) |
| **Total** | **96** | |

---

## Parametrization Reference

Many tests benefit from `@pytest.mark.parametrize` to avoid code duplication.

| Parametrize Axis | Values | Used In |
|---|---|---|
| `bundle` | `["godel", "product", "lukasiewicz", "stable_product"]` | TestFuzzyOperatorIdentities, TestOperatorGradients, TestLogicIdentities |
| `x_val` | `[0.0, 0.2, 0.5, 0.8, 1.0]` | TestFuzzyOperatorIdentities |
| `grid` | `torch.linspace(0.0, 1.0, 11)` | TestLogicIdentities |
| `agg_method` | `["mean", "soft_min", "p_mean"]` | TestRuleNetwork |
| `extractor_type` | `["slot_identity", "proposal_head"]` | TestEntityExtraction |
| `relation_type` | `["bilinear", "ntn"]` | TestPredicateRelation |

When using parametrize with bundles, split pure and stable tests to use different tolerance assertions rather than a single parametrized tolerance.

---

## Common Test Utilities

Define the following helper functions in the test file or a `conftest.py`.

| Utility | Signature | Purpose |
|---|---|---|
| `get_operator_bundle` | `(name: str) -> OperatorBundle` | Return configured operator bundle by name |
| `compile_rules` | `(rule_dicts: List[dict]) -> List[RuleAST]` | Compile list of rule dicts into AST objects |
| `create_toy_model` | `() -> Tuple[nn.Module, SymbolicReasoner, nn.ModuleDict, List[RuleAST]]` | Return encoder, reasoner, predicates, rules for integration tests |
| `aggregate_violations` | `(violations: Tensor, method: str) -> Tensor` | Aggregate violations by method name |
| `check_no_nan` | `(model: nn.Module) -> bool` | Check all parameters and gradients for NaN |
| `generalized_mean` | `(x: Tensor, p: float, dim: int) -> Tensor` | Reference implementation of generalized mean for comparison |
