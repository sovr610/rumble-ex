# Compliance Test Generation Patterns

## Overview

Generated tests enforce structural and numerical compliance between implementation
code and the paper spec. Tests are designed to fail loudly when code diverges from
spec — the "paper drift detector."

Tests are structural (shape, dtype, range, logic) not just numerical (exact values),
because exact numerical reproduction requires identical seeds and hardware.

## Test Categories

### 1. Space Shape/Dtype Checks

Verify observation, action, and state tensors match spec dimensions.

```python
# Generated test example
import pytest
import torch

class TestSpaceCompliance:
    """Verify observation/action spaces match spec.yaml."""

    def test_observation_exec_gate_mask(self):
        """spec.yaml: spaces.observation_exec[0] gate_mask bool [64, 64]"""
        obs = env.reset()
        gate_mask = obs["gate_mask"]
        assert gate_mask.shape == (64, 64), (
            f"gate_mask shape {gate_mask.shape} != spec (64, 64)"
        )
        assert gate_mask.dtype == torch.bool, (
            f"gate_mask dtype {gate_mask.dtype} != spec bool"
        )

    def test_observation_exec_imu_body_rates(self):
        """spec.yaml: spaces.observation_exec[1] imu_body_rates float32 [3]"""
        obs = env.reset()
        imu = obs["imu_body_rates"]
        assert imu.shape[-1] == 3
        assert imu.dtype == torch.float32

    def test_action_bounds(self):
        """spec.yaml: spaces.action bounds [0.0, 1.0]"""
        action = policy.sample_action(obs)
        assert action.min() >= 0.0, f"Action min {action.min()} < spec bound 0.0"
        assert action.max() <= 1.0, f"Action max {action.max()} > spec bound 1.0"
        assert action.shape[-1] == 4  # motor_cmd dim
```

**Generation pattern**: For each entry in `spaces.observation_exec`,
`spaces.information_train`, and `spaces.action`, emit a test function that
constructs or samples the tensor and asserts shape, dtype, and bounds.

### 2. Reward Expression Checks

Given synthetic state transitions, verify reward terms match the expression AST.

```python
class TestRewardCompliance:
    """Verify reward computation matches spec.yaml expression trees."""

    def test_progress_reward_basic(self):
        """spec.yaml: reward.terms[0] 'progress'"""
        # Construct synthetic transition
        state = make_synthetic_state(gate_progress=0.3)
        next_state = make_synthetic_state(gate_progress=0.5)

        # Compute reward from code
        code_reward = reward_fn.compute_progress(state, next_state)

        # Compute from spec AST
        spec_reward = eval_spec_ast(
            spec["reward"]["terms"][0]["expression_ast"],
            state=state, next_state=next_state
        )

        assert abs(code_reward - spec_reward) < 1e-6, (
            f"Progress reward {code_reward} != spec {spec_reward}"
        )

    def test_reward_discount(self):
        """spec.yaml: reward.discount = 0.997"""
        assert config.discount == pytest.approx(0.997), (
            f"Discount {config.discount} != spec 0.997"
        )
```

**Generation pattern**: For each reward term, emit:
1. A synthetic state constructor with relevant fields
2. AST evaluation function
3. Comparison test with tolerance

### 3. Termination Condition Checks

Verify boolean logic matches the condition ASTs.

```python
class TestTerminationCompliance:
    """Verify termination conditions match spec.yaml boolean ASTs."""

    def test_ground_collision_positive(self):
        """spec.yaml: termination.conditions[0] ground_collision: z > 0"""
        state = make_synthetic_state(z_position=0.5)  # Above ground
        assert not termination_fn.check_ground_collision(state)

    def test_ground_collision_negative(self):
        state = make_synthetic_state(z_position=-0.1)  # Below ground
        assert termination_fn.check_ground_collision(state)

    def test_gate_collision_threshold(self):
        """spec.yaml: termination.conditions[1] gate_collision"""
        # Test at boundary
        state = make_synthetic_state(
            gate_distance=0.49,  # Just inside threshold
            gate_angle=0.0
        )
        # Should match spec condition direction (lt vs le matters)
        result = termination_fn.check_gate_collision(state)
        assert result == eval_spec_bool_ast(
            spec["termination"]["conditions"][1]["condition_ast"],
            state=state
        )
```

**Generation pattern**: For each termination condition, emit:
1. Positive case (condition should trigger)
2. Negative case (condition should not trigger)
3. Boundary case (test at exact threshold value)

### 4. Domain Randomization Range Checks

Verify randomized parameters stay within spec ranges.

```python
class TestDomainRandomization:
    """Verify domain randomization matches spec.yaml ranges."""

    def test_mass_randomization_range(self):
        """spec.yaml: dynamics.domain_randomization[0] mass uniform [0.6, 0.9]"""
        masses = [env.sample_mass() for _ in range(1000)]
        assert all(0.6 <= m <= 0.9 for m in masses), (
            f"Mass samples outside spec range [0.6, 0.9]"
        )

    def test_mass_resample_frequency(self):
        """spec.yaml: resample_frequency = 'per_episode'"""
        env.reset()
        mass_1 = env.get_param("mass")
        for _ in range(10):
            env.step(zero_action)
        mass_2 = env.get_param("mass")
        assert mass_1 == mass_2, "Mass changed mid-episode (spec: per_episode)"

        env.reset()
        # After reset, mass MAY differ (sampled per episode)
```

**Generation pattern**: For each domain_randomization entry:
1. Sample N times, assert within range
2. If `per_episode`: verify constant within episode, may change across episodes
3. If `per_step`: verify changes at each step

### 5. Timing/Delay Checks

Verify delays are applied exactly once in the correct pipeline position.

```python
class TestTimingCompliance:
    """Verify timing matches spec.yaml."""

    def test_control_frequency(self):
        """spec.yaml: timing.control_frequency_hz"""
        dt = env.get_control_dt()
        expected_dt = 1.0 / spec["timing"]["control_frequency_hz"]
        assert abs(dt - expected_dt) < 1e-6

    def test_sensor_delay_applied(self):
        """spec.yaml: timing.sensor_delay_ms"""
        # Verify observation is delayed by spec amount
        delay_steps = int(
            spec["timing"]["sensor_delay_ms"]
            / (1000.0 / spec["timing"]["control_frequency_hz"])
        )
        assert env.get_observation_delay_steps() == delay_steps

    def test_action_delay_applied(self):
        """spec.yaml: timing.action_delay_ms"""
        delay_steps = int(
            spec["timing"]["action_delay_ms"]
            / (1000.0 / spec["timing"]["control_frequency_hz"])
        )
        assert env.get_action_delay_steps() == delay_steps
```

### 6. Informed-POMDP Key Gating Checks

The most critical test category for Informed Dreamer-style papers.

```python
class TestInformedPOMDP:
    """Verify informed-POMDP split: privileged info excluded at execution."""

    def test_privileged_excluded_at_execution(self):
        """spec.yaml: information_train fields must NOT appear in exec obs."""
        exec_keys = set(policy.get_observation_keys(mode="execution"))
        train_keys = set(spec_get_information_train_names())
        overlap = exec_keys & train_keys
        assert len(overlap) == 0, (
            f"Privileged keys leaked to execution: {overlap}"
        )

    def test_privileged_available_at_training(self):
        """spec.yaml: information_train fields MUST be in training decoder targets."""
        decoder_targets = set(world_model.get_decoder_target_keys())
        train_keys = set(spec_get_information_train_names())
        missing = train_keys - decoder_targets
        assert len(missing) == 0, (
            f"Privileged keys missing from decoder targets: {missing}"
        )

    def test_decoder_gating_regex(self):
        """Verify decoder only decodes info_* keys (Informed Dreamer convention)."""
        for field in spec["spaces"]["information_train"]:
            if field.get("informed_dreamer_key"):
                pattern = field["informed_dreamer_key"]
                # Verify the decoder gating regex matches this key
                assert re.match(pattern, field["name"]), (
                    f"Key {field['name']} doesn't match gating pattern {pattern}"
                )
```

### 7. UNRESOLVED Stub Tests

For any field marked UNRESOLVED, generate a failing stub:

```python
class TestUnresolved:
    """Failing stubs for UNRESOLVED spec fields."""

    @pytest.mark.skip(reason="UNRESOLVED: reward.terms[2].expression_ast "
                             "— manual extraction required [§3.3]")
    def test_UNRESOLVED_reward_term_2_expression(self):
        """Manual extraction needed for reward term 2 expression."""
        raise NotImplementedError("UNRESOLVED field")
```

These stubs:
- Show up in test reports as skipped (visible, not hidden)
- Block CI green status if configured with `--strict-markers`
- Include paper source reference for manual resolution

## Test File Organization

```
tests/spec_compliance/
├── conftest.py           # Shared fixtures (spec loader, synthetic state builders)
├── test_spaces.py        # Space shape/dtype/bounds checks
├── test_reward.py        # Reward expression evaluation
├── test_termination.py   # Termination condition logic
├── test_domain_rand.py   # Domain randomization ranges
├── test_timing.py        # Timing and delay correctness
├── test_informed_pomdp.py # Privileged info gating
└── test_unresolved.py    # UNRESOLVED stubs (auto-generated)
```

## conftest.py Fixtures

The generated `conftest.py` provides:

```python
@pytest.fixture
def spec():
    """Load spec.yaml."""
    return yaml.safe_load(open("spec.yaml"))

@pytest.fixture
def make_synthetic_state():
    """Factory for synthetic state dicts matching spec shapes."""
    def _make(**overrides):
        state = {}
        for field in spec["spaces"]["state"]:
            shape = field["shape"]
            state[field["name"]] = torch.zeros(shape)
        state.update(overrides)
        return state
    return _make
```

## Generation Rules

1. One test file per category
2. One test class per spec section
3. One test method per field or condition
4. Every test docstring includes `spec.yaml:` path for traceability
5. UNRESOLVED fields get skip-marked stubs, not silent omissions
6. Boundary conditions tested for inequalities (< vs <=)
7. Tolerance specified for float comparisons (`pytest.approx`)
