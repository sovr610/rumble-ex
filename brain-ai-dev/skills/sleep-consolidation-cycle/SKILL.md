---
name: Sleep Consolidation Cycle (Offline Replay + Synaptic Homeostasis + Systems Consolidation)
description: >-
  This skill should be used when the user asks to "add sleep cycle",
  "memory consolidation", "offline replay", "synaptic homeostasis",
  "sleep-wake cycle", "memory replay consolidation", "sharp wave ripple replay",
  "slow wave sleep computation", "dream-like generative replay",
  "synaptic downscaling", "interleaved replay", "systems consolidation",
  "hippocampal-cortical transfer", "offline training phase",
  "implement sleep consolidator", "add NREM replay phase", "add REM generative phase",
  "implement experience replay scheduler", "add weight downscaling",
  "implement complementary learning systems", "add fast-to-slow transfer",
  "implement two-phase training", "add sleep phase to training orchestrator",
  "implement priority replay sampling", "add compressed replay",
  "implement dream replay from world model", "add pseudo-rehearsal",
  "implement capacity restoration", "add synaptic renormalization",
  or mentions sleep consolidation, offline consolidation phase, NREM structured replay,
  REM creative recombination, sharp-wave ripple compressed replay, synaptic homeostasis
  hypothesis, Tononi-Cirelli downscaling, complementary learning systems, hippocampal-cortical
  transfer, fast-to-slow knowledge distillation, dream-like generative replay, pseudo-experience
  generation, two-phase wake-sleep training, replay priority sampling, capacity restoration,
  or sleep-wake integration with training orchestrator in the cognitive pipeline.
version: 0.1.0
---

# Sleep Consolidation Cycle (Offline Replay + Synaptic Homeostasis + Systems Consolidation)

## Overview

This skill implements an offline consolidation phase -- analogous to biological sleep -- that
replays stored experiences, prunes redundant synaptic weights, and transfers knowledge from
fast-learning episodic systems to slow-learning semantic systems. The biological brain does
not simply shut down during sleep; it engages in highly structured neural activity that
reorganizes, stabilizes, and generalizes waking experience. This skill brings the same
principle to the brain_ai cognitive architecture.

The consolidation cycle operates in two distinct sub-phases:

1. **NREM-like phase** (structured replay + homeostasis): compressed replay of recent experiences
   at accelerated timescale (sharp-wave ripple analogue), interleaved with global synaptic
   downscaling to restore learning capacity.
2. **REM-like phase** (generative + creative recombination): the world model generates
   pseudo-experiences that blend real memories with novel recombinations, promoting
   generalization and creative transfer.

Together these phases implement a biologically grounded offline training loop that runs between
normal (wake-phase) training iterations. The consolidation cycle plugs into the existing
training-orchestrator as an optional phase that can be triggered at configurable intervals
(e.g., every N wake epochs, or when a replay buffer reaches capacity).

## Public Contract

### SleepConsolidator

Orchestrates the full sleep cycle: NREM replay, synaptic homeostasis, REM generative replay,
and systems consolidation. Entry point for the offline training phase.

```python
class SleepConsolidator(nn.Module):
    def __init__(self, cfg: SleepConfig): ...
    def consolidate(
        self,
        model: nn.Module,
        replay_buffer: ReplayBuffer,
        world_model: Optional[nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
    ) -> ConsolidationResult: ...
    def nrem_phase(self, model, replay_buffer) -> PhaseResult: ...
    def rem_phase(self, model, world_model) -> PhaseResult: ...
```

### ReplayScheduler

Manages experience sampling with priority-weighted compressed replay. Implements sharp-wave
ripple-inspired accelerated replay at configurable compression ratios.

```python
class ReplayScheduler:
    def __init__(self, cfg: SleepConfig): ...
    def sample_replay_batch(
        self, replay_buffer: ReplayBuffer, batch_size: int
    ) -> ReplayBatch: ...
    def update_priorities(self, indices: Tensor, td_errors: Tensor) -> None: ...
    def get_compressed_sequence(
        self, experience: Experience, compression_ratio: float
    ) -> CompressedExperience: ...
```

### SynapticHomeostasis

Implements the Tononi-Cirelli synaptic homeostasis hypothesis: global downscaling of
network weights to restore learning capacity while preserving relative weight ratios.

```python
class SynapticHomeostasis:
    def __init__(self, cfg: SleepConfig): ...
    def downscale(
        self, model: nn.Module, factor: Optional[float] = None
    ) -> HomeostasisResult: ...
    def compute_scaling_factor(self, model: nn.Module) -> float: ...
    def selective_downscale(
        self, model: nn.Module, importance: Dict[str, Tensor]
    ) -> HomeostasisResult: ...
```

### SystemsConsolidation

Transfers knowledge from fast-learning (hippocampus-like) systems to slow-learning
(cortex-like) systems. In brain_ai, this maps HTM (fast, episodic) to workspace and
encoder weights (slow, semantic) via knowledge distillation.

```python
class SystemsConsolidation(nn.Module):
    def __init__(self, cfg: SleepConfig): ...
    def transfer(
        self,
        fast_model: nn.Module,
        slow_model: nn.Module,
        replay_buffer: ReplayBuffer,
    ) -> TransferResult: ...
    def compute_distillation_loss(
        self, fast_output: Tensor, slow_output: Tensor
    ) -> Tensor: ...
```

### SleepConfig

Central configuration dataclass for all consolidation parameters.

```python
@dataclass
class SleepConfig:
    # NREM phase
    nrem_replay_steps: int = 100
    compression_ratio: float = 5.0
    priority_exponent: float = 0.6
    priority_correction: float = 0.4
    # Synaptic homeostasis
    downscale_factor: float = 0.85
    downscale_strategy: str = "global"  # "global", "selective", "layerwise"
    protect_threshold: float = 0.1
    # REM phase
    rem_replay_steps: int = 50
    dream_noise_scale: float = 0.1
    creative_blend_ratio: float = 0.3
    # Systems consolidation
    distillation_temperature: float = 2.0
    transfer_learning_rate: float = 1e-4
    fast_to_slow_ratio: float = 0.5
    # Scheduling
    consolidate_every_n_epochs: int = 5
    min_buffer_size: int = 1000
    sleep_duration_budget: float = 0.2  # fraction of wake training time
```

## Key Concepts

### Sharp-Wave Ripple Replay

In biological brains, the hippocampus replays recent experiences during NREM sleep at a
compressed timescale (roughly 5-20x faster than the original experience). Sharp-wave ripples
coordinate this replay, with high-reward or high-surprise experiences replayed preferentially.

In brain_ai, this maps to priority-weighted experience replay from the replay buffer at an
accelerated training rate. The ReplayScheduler implements:

- **Priority sampling**: experiences with higher TD-error or surprise signal are replayed
  more frequently (proportional to `|delta|^alpha` where alpha is the priority exponent).
- **Compressed sequences**: temporal sequences are compressed by subsampling at the
  configured compression ratio, preserving key transitions while reducing compute.
- **Accelerated learning rate**: replay steps use a higher effective learning rate than
  wake-phase training, reflecting the accelerated timescale of biological ripple replay.

### Synaptic Homeostasis Hypothesis (Tononi and Cirelli)

The SHY proposes that waking experience causes a net increase in synaptic strength across
the brain, which is unsustainable. Sleep serves to globally downscale synaptic weights,
restoring capacity for new learning while preserving the relative pattern of strong vs weak
connections that encodes learned information.

Mathematically, for each weight matrix W:

```
W_sleep = W_wake * s    where 0 < s < 1 (global scaling factor)
```

The downscaling factor s is computed based on the average weight magnitude relative to
initialization scale. Three strategies are supported:

- **Global**: uniform scaling factor across all parameters.
- **Selective**: weight-importance-aware scaling that protects high-importance weights
  (those with large Fisher information or gradient magnitude).
- **Layerwise**: per-layer scaling factors computed from layer-specific weight statistics.

### Systems Consolidation

Complementary Learning Systems (CLS) theory (McClelland, McNaughton, O'Reilly, 1995) posits
that the brain uses two learning systems: a fast hippocampal system for rapid episodic
encoding and a slow neocortical system for gradual extraction of statistical regularities.
Sleep consolidation transfers information from the fast to the slow system via interleaved
replay.

In brain_ai, this maps to:

- **Fast system**: HTM spatial-temporal pooler, engram memory, episodic replay buffer --
  systems that learn quickly from single or few exposures.
- **Slow system**: workspace encoder weights, transformer backbone, semantic representations --
  systems that learn gradually through repeated exposure.
- **Transfer mechanism**: knowledge distillation where the fast system acts as teacher and
  the slow system as student, trained on replay buffer experiences.

### Dream-Like Generative Replay

During REM sleep, the brain generates novel experiences that blend elements of real memories
with creative recombinations. This is hypothesized to promote generalization and prevent
catastrophic forgetting of older memories.

In brain_ai, the world model (DreamerV3 RSSM) generates pseudo-experiences by:

1. Sampling initial states from the replay buffer.
2. Rolling out the world model with stochastic action sampling.
3. Blending generated observations with stored real observations at the configured
   creative_blend_ratio.
4. Training the main model on these blended pseudo-experiences interleaved with real replay.

This interleaving of real and generated experiences is the "pseudo-rehearsal" technique that
prevents catastrophic forgetting without requiring storage of all past experiences.

### Two-Phase Training: Wake and Sleep

The consolidation cycle integrates with the training orchestrator as an alternating two-phase
process:

```
Wake Phase (normal training):
  - Process new experiences from environment/data
  - Store experiences in replay buffer with priority scores
  - Update model weights via standard gradient descent
  - Monitor buffer size and wake epoch count

Sleep Phase (consolidation):
  - NREM sub-phase: structured replay + synaptic homeostasis
  - REM sub-phase: generative replay + creative recombination
  - Systems consolidation: fast-to-slow transfer
  - Emit consolidation metrics and artifacts

Trigger conditions (any of):
  - consolidate_every_n_epochs wake epochs have elapsed
  - Replay buffer has reached min_buffer_size
  - Explicit trigger from training orchestrator
```

### NREM vs REM Sub-Phases

| Property | NREM-Like | REM-Like |
|---|---|---|
| Replay source | Real experiences from buffer | Generated pseudo-experiences from world model |
| Replay speed | Compressed (5-20x) | Normal timescale |
| Weight updates | Conservative (low LR) | Exploratory (creative blending) |
| Homeostasis | Yes (global downscaling) | No |
| Purpose | Stabilize, consolidate, prune | Generalize, recombine, create |
| Ordering | Runs first | Runs second |

### Integration with Training Orchestrator

The sleep consolidation cycle is designed to integrate with the existing seven-phase training
pipeline. It can be activated at any phase that maintains a replay buffer:

- **Phase 4+** (Global Workspace and beyond): full consolidation with HTM-to-workspace
  systems consolidation.
- **Phase 6+** (Reasoning): consolidation with symbolic knowledge transfer.
- **Phase 7** (Meta-Learning): consolidation with task-level replay and meta-knowledge
  distillation.

The training orchestrator calls `SleepConsolidator.consolidate()` at the appropriate
intervals, and the consolidation result is logged to the manifest and TensorBoard.

## Configuration Surface

### SleepConfig

| Field | Default | Purpose |
|---|---|---|
| `nrem_replay_steps` | 100 | Number of replay gradient steps in NREM phase |
| `compression_ratio` | 5.0 | Temporal compression factor for sequence replay |
| `priority_exponent` | 0.6 | Alpha for priority sampling (0=uniform, 1=greedy) |
| `priority_correction` | 0.4 | Importance sampling correction beta |
| `downscale_factor` | 0.85 | Global synaptic downscaling multiplier |
| `downscale_strategy` | `"global"` | `"global"`, `"selective"`, `"layerwise"` |
| `protect_threshold` | 0.1 | Fraction of top-importance weights protected from downscaling |
| `rem_replay_steps` | 50 | Number of generative replay steps in REM phase |
| `dream_noise_scale` | 0.1 | Noise injected into world model rollouts |
| `creative_blend_ratio` | 0.3 | Ratio of generated to real content in REM blending |
| `distillation_temperature` | 2.0 | Softmax temperature for knowledge distillation |
| `transfer_learning_rate` | 1e-4 | Learning rate for systems consolidation transfer |
| `fast_to_slow_ratio` | 0.5 | Weight of distillation loss vs reconstruction loss |
| `consolidate_every_n_epochs` | 5 | Wake epochs between consolidation cycles |
| `min_buffer_size` | 1000 | Minimum replay buffer size before consolidation |
| `sleep_duration_budget` | 0.2 | Max fraction of wake training time spent in sleep |
| `enable_nrem` | True | Enable/disable NREM sub-phase |
| `enable_rem` | True | Enable/disable REM sub-phase |
| `enable_homeostasis` | True | Enable/disable synaptic homeostasis |
| `enable_systems_transfer` | True | Enable/disable systems consolidation |
| `replay_buffer_max_size` | 100000 | Maximum replay buffer capacity |

Presets: `SleepConfig.minimal()`, `.dev()`, `.production()`.

## Done-When Gates

| Gate | Test | Threshold |
|---|---|---|
| **(a) Replay fidelity** | Priority-sampled replay batch distribution matches expected power-law; compressed sequences preserve key transitions; 10 deterministic replay cycles produce identical gradient updates | Priority KS-test p > 0.05; key transition retention > 95%; exact gradient match |
| **(b) Homeostasis correctness** | Global downscaling preserves relative weight ratios; weight norms decrease monotonically; capacity metric (mean weight magnitude) returns to target range after downscale | Relative ratio error < 1e-6; norm decrease verified; capacity within 5% of target |
| **(c) Systems consolidation** | Distillation loss decreases over transfer steps; slow model accuracy on replay data improves; fast-to-slow transfer does not degrade fast model performance | Loss decrease > 10%; accuracy improvement > 2%; fast model accuracy drop < 1% |
| **(d) End-to-end cycle** | Full consolidate() call completes without error; ConsolidationResult contains valid metrics for all sub-phases; total sleep time within budget | No errors; all metrics present and finite; sleep_time / wake_time < budget |

## Common Failure Modes

| Symptom | Cause | Fix |
|---|---|---|
| Replay priorities degenerate to uniform | Priority exponent too low or TD errors not updated | Increase alpha; call update_priorities after each replay batch |
| Catastrophic forgetting after homeostasis | Downscale factor too aggressive | Use factor > 0.8; enable selective strategy with importance protection |
| World model generates garbage in REM | World model not trained or poorly calibrated | Ensure world model trained before enabling REM; increase dream_noise_scale gradually |
| Systems consolidation diverges | Transfer learning rate too high | Reduce transfer_learning_rate; use gradient clipping |
| Sleep phase takes too long | Too many replay steps or no budget enforcement | Reduce nrem/rem replay steps; enforce sleep_duration_budget |
| NaN in downscaled weights | Near-zero weights multiplied by small factor | Add epsilon floor to weight magnitudes before scaling |
| Priority sampling OOM | Priority array not bounded | Cap priority values; use float32 for priority storage |
| Replay buffer stale | Old experiences never evicted | Use FIFO eviction with priority-based retention |

## Anti-Patterns

- **No replay buffer** -- consolidation requires stored experiences; never consolidate with empty buffer
- **Homeostasis every step** -- downscaling must happen periodically (sleep), not continuously
- **REM without world model** -- generative replay requires a trained world model; skip REM if unavailable
- **Uniform replay** -- priority sampling is essential; uniform replay wastes compute on redundant experiences
- **Aggressive downscaling** -- factor < 0.7 risks catastrophic forgetting; start conservative
- **Skipping importance sampling correction** -- uncorrected priority sampling introduces bias; always apply IS weights
- **Consolidating during wake** -- sleep consolidation must be offline; do not interleave with live training
- **Ignoring sleep budget** -- unbounded consolidation steals compute from wake training; enforce budget
- **Hardcoded replay steps** -- use SleepConfig, not magic numbers
- **fp16 priority computation** -- priority exponentiation needs fp32; avoid half-precision for priorities

## Additional Resources

### Reference Files

- **`references/replay-mechanisms.md`** -- Sharp-wave ripple biology, compressed replay implementation, priority sampling mathematics, replay buffer design
- **`references/synaptic-homeostasis.md`** -- Tononi-Cirelli SHY, weight downscaling mathematics, capacity restoration metrics, selective vs global strategies
- **`references/systems-consolidation.md`** -- CLS theory, hippocampal-cortical transfer, knowledge distillation for systems consolidation, brain_ai component mapping
- **`references/generative-replay.md`** -- Dream replay from world model, pseudo-rehearsal theory, creative recombination, REM-like generative training
- **`references/testing-matrix.md`** -- Test categories, pytest parameterizations, done-when checklist, benchmarking methodology

### Asset Templates

- **`assets/sleep_consolidator_template.py`** -- SleepConsolidator orchestrating NREM + REM phases, ConsolidationResult, self-test
- **`assets/replay_scheduler_template.py`** -- ReplayScheduler with priority sampling, compressed replay, replay buffer, self-test
- **`assets/synaptic_homeostasis_template.py`** -- SynapticHomeostasis with global/selective/layerwise downscaling, self-test
- **`assets/systems_consolidation_template.py`** -- SystemsConsolidation with fast-to-slow distillation transfer, self-test
- **`assets/sleep_config_template.py`** -- SleepConfig dataclass with presets, serialization, validation, self-test

### Scripts

- **`scripts/validate_consolidation.py`** -- Runtime contract validation (replay fidelity, homeostasis correctness, systems consolidation, end-to-end cycle)
- **`scripts/gen_consolidation_tests.py`** -- Generates `tests/test_sleep_consolidation.py` (~100+ test cases)
- **`scripts/consolidation_benchmark.py`** -- Benchmark consolidation throughput, memory usage, and time budgets
