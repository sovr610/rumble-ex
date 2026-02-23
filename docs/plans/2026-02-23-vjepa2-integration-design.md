# V-JEPA 2 Integration Design: Perceptual Backbone for BrainAI

**Date:** 2026-02-23
**Status:** Proposed
**Scope:** Integrate V-JEPA 2 as the perceptual and predictive backbone across all 7 cognitive layers

---

## 1. Overview

Integrate Meta's V-JEPA 2 (Video Joint Embedding Predictive Architecture v2) as the primary visual perception and world model for the BrainAI cognitive architecture. V-JEPA 2 serves as a "cortical backbone" — a frozen, pre-trained perceptual substrate that feeds rich representations into every downstream cognitive module.

### Why V-JEPA 2

V-JEPA 2 (June 2025) is a self-supervised video model pre-trained on 1M+ hours of internet video. It achieves SOTA motion understanding (77.3% SSv2), enables zero-shot robot control with only 62 hours of unlabeled robot video, and operates 30x faster than comparable world models (Cosmos). Its latent-space predictions align with the Free Energy Principle — predicting abstract states rather than reconstructing pixels.

### Design Principle

**V-JEPA 2 does not replace the cognitive architecture — it supercharges it.** Each of the 7 layers retains its role but operates on dramatically richer perceptual representations and has access to a physics-understanding world model for prediction and planning.

---

## 2. Architecture

```
Sensory Input (Video / Images)
      │
  ┌───┴───────────────────────────┐
  │  V-JEPA 2 ViT-g (frozen, 1B) │  ← "cortical mantle"
  │  Tubelet tokenizer + 3D-RoPE  │
  │  SwiGLU feed-forward blocks   │
  └───┬───────────────────────────┘
      │
      ├─① EncoderBridge ──────→ [Global Workspace Competition]
      │   (project 1408→4096,      with text, audio, sensor
      │    produce EncoderOutput)    modalities
      │
      ├─② TemporalBridge ─────→ [HTM Sequence Memory]
      │   (per-frame token seq       bootstraps temporal
      │    → HTM input)              learning
      │
      ├─③ WorldModelBridge ───→ [Active Inference Agent]
      │   + V-JEPA 2-AC Predictor    replaces MLP generative
      │   (action-conditioned)       model; CEM → EFE planning
      │
      ├─④ ImaginationEngine ──→ [System 2 Reasoning]
      │   (predictor rollout          mental simulation for
      │    in latent space)           counterfactual evaluation
      │
      └─⑤ AttentionMasking ←── [Neuromodulatory Gate]
          (dynamic masking ratio      ACh/NE modulate what
           controlled by modulators)  V-JEPA 2 attends to
```

### Key Design Decisions

- **Frozen encoder**: V-JEPA 2 ViT-g stays frozen (1B params). Only lightweight adapters/projections are trainable. Follows the JEPA evaluation paradigm.
- **Modular bridges**: Each integration is a separate module with its own feature flag. No all-or-nothing dependency.
- **Contract compatibility**: Every bridge produces outputs matching existing BrainAI contracts.
- **Incremental deployment**: Phases can be implemented independently.

---

## 3. Module ① — Encoder Bridge

### Purpose

Wrap V-JEPA 2's frozen ViT-g encoder as a drop-in replacement for `VisionEncoder`, producing output compatible with the existing workspace pipeline.

### Contract

```python
class VJEPA2VisionEncoder(nn.Module):
    """V-JEPA 2 vision encoder adapter for BrainAI pipeline.

    Wraps frozen V-JEPA 2 ViT-g and produces output matching
    the existing VisionEncoder contract: (B, output_dim).
    """
    def __init__(self, cfg: VJEPA2EncoderConfig):
        # Frozen V-JEPA 2 ViT-g backbone (1B params, not trainable)
        # Attentive pooler (4-layer cross-attention, ~22M trainable)
        # Projection: 1408 → workspace_dim (4096)
        # Optional SNN conversion layer for spike output
        # Salience estimator from attention entropy
        ...

    def forward(
        self, x: torch.Tensor, temporal_input: bool = False
    ) -> torch.Tensor:
        """Match VisionEncoder.forward() signature exactly.

        Args:
            x: (B, C, H, W) for images or (T, B, C, H, W) for video
            temporal_input: whether input is temporal sequence
        Returns:
            features: (B, output_dim)
        """
        ...

    def forward_rich(self, x: torch.Tensor) -> EncoderOutput:
        """Extended output with salience, temporal info, spike encoding."""
        ...
```

### Data Flow

```
Input: (B, 3, 384, 384) image or (B, T, 3, 384, 384) video
  → Tubelet tokenizer: (B, N_patches, 1408)   [N_patches ≈ 1200 for 384px]
  → Frozen ViT-g: (B, N_patches, 1408)         [24 transformer blocks]
  → Attentive pooler: (B, N_queries, 1408)      [N_queries=16 learnable queries]
  → Mean pool: (B, 1408)
  → Linear projection: (B, 4096)                [→ workspace_dim]
  → (optional) SNN layer: spike encoding for biological compatibility
Output: (B, 4096)
```

### Configuration

```python
@dataclass
class VJEPA2EncoderConfig:
    enabled: bool = False              # Feature flag
    model_name: str = "vjepa2_vitg"    # ViT variant
    pretrained: bool = True            # Load pretrained weights
    freeze_encoder: bool = True        # Freeze backbone
    num_query_tokens: int = 16         # Attentive pooler queries
    probe_layers: int = 4              # Attentive probe depth
    probe_heads: int = 16              # Attention heads in probe
    output_dim: int = 4096             # Must match workspace_dim
    spike_output: bool = False         # Enable SNN conversion layer
    video_frames: int = 16             # Frames per clip for video
    tubelet_size: int = 2              # Temporal stride of tubelets
```

### Neuroscience Mapping

V-JEPA 2 ViT-g = Retina → LGN → V1 → V2 → V4 → IT cortex (ventral visual stream). Attentive pooler = IT → PFC projection (task-relevant feature selection). SNN conversion = LGN spike encoding.

---

## 4. Module ② — Temporal Bridge to HTM

### Purpose

Extract per-frame temporal representations from V-JEPA 2 and feed them as a sequence to the HTM layer, bootstrapping temporal learning with pre-learned visual abstractions.

### Contract

```python
class VJEPA2TemporalBridge(nn.Module):
    """Extracts temporal sequences from V-JEPA 2 for HTM consumption.

    V-JEPA 2 tubelets span 2 frames each. This bridge:
    1. Groups encoder output tokens by their temporal position
    2. Pools each temporal group to a single vector
    3. Produces a (B, T_frames, htm_input_dim) sequence for HTM
    """
    def __init__(self, cfg: TemporalBridgeConfig):
        # Temporal group pooler (average or attention per frame)
        # Projection: 1408 → htm_input_dim (workspace_dim)
        ...

    def forward(
        self, vjepa2_tokens: torch.Tensor, temporal_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            vjepa2_tokens: (B, N_patches, 1408) from ViT-g
            temporal_positions: (N_patches,) frame index per token
        Returns:
            temporal_seq: (B, T_frames, htm_input_dim)
        """
        ...
```

### Data Flow

```
V-JEPA 2 output: (B, 1200, 1408) [16 frames × 75 spatial patches]
  → Group by temporal position: [(B, 75, 1408)] × 8 groups [2-frame tubelets]
  → Per-group mean pool: (B, 8, 1408)
  → Project: (B, 8, 4096)
  → Feed to HTMLayer as temporal sequence
HTM output: anomaly scores, sequence predictions
```

### Configuration

```python
@dataclass
class TemporalBridgeConfig:
    enabled: bool = False
    pooling_mode: str = "mean"     # mean | attention | max
    output_dim: int = 4096         # Must match workspace_dim / htm input_size
```

### Neuroscience Mapping

Hippocampal-cortical interaction. The cortex (V-JEPA 2) learns slow, stable representations; the hippocampus (HTM) learns rapid temporal sequences over those representations. Prediction error between V-JEPA 2's prediction and actual input drives HTM anomaly detection, analogous to hippocampal mismatch signals.

---

## 5. Module ③ — World Model for Active Inference

### Purpose

Replace the current MLP-based generative model in `ActiveInferenceAgent` with V-JEPA 2-AC's action-conditioned predictor, enabling physics-aware planning with full EFE computation.

### Contract

```python
class VJEPA2WorldModel(nn.Module):
    """V-JEPA 2-AC as the generative model for Active Inference.

    Maps V-JEPA 2-AC components to Active Inference terms:
      - V-JEPA 2 encoder    → observation model p(o|s)
      - V-JEPA 2-AC predictor → transition model p(s'|s,a)
      - CEM planner          → policy optimization (extended with EFE)
    """

    def encode_observation(self, obs: torch.Tensor) -> Distribution:
        """p(s|o) — encode observations to latent state distribution."""
        ...

    def predict_transition(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Distribution:
        """p(s'|s,a) — predict next state given current state + action."""
        ...

    def compute_efe(
        self, state: torch.Tensor, action_sequences: torch.Tensor,
        preferences: torch.Tensor
    ) -> torch.Tensor:
        """Expected Free Energy over action sequences.

        EFE(pi) = sum_t [ -E[ln P(o_t|pi)]   <- pragmatic (L1 to goal)
                        + H[P(s_t|o_t, pi)]   <- epistemic (uncertainty)
                        - empowerment ]        <- instrumental (MI(a;s'))
        """
        ...

    def plan(
        self, current_state: torch.Tensor, preferences: torch.Tensor,
        horizon: int = 8
    ) -> Tuple[torch.Tensor, Dict]:
        """CEM planner extended with EFE minimization."""
        ...
```

### Data Flow

```
Current observation: workspace representation (B, 4096)
  → V-JEPA 2 encoder (frozen): latent state (B, N, 1408)
  → CEM generates K action sequences, each of length H
  → For each sequence, roll out V-JEPA 2-AC predictor:
      state_t, action_t → predicted_state_{t+1}
  → Compute EFE for each sequence:
      pragmatic = L1(predicted_final, preferred_state)
      epistemic = entropy of state distribution
      empowerment = MI(actions; resulting states)
  → Select action sequence minimizing total EFE
  → Execute first action
Output: selected action + planning trace
```

### Configuration

```python
@dataclass
class VJEPA2WorldModelConfig:
    enabled: bool = False
    predictor_dim: int = 384           # V-JEPA 2-AC predictor dimension
    predictor_depth: int = 12          # Predictor transformer layers
    action_dim: int = 7                # Action space (7 for robot, configurable)
    action_embed_dim: int = 384        # Action embedding dimension
    planning_horizon: int = 8          # CEM planning steps
    cem_population: int = 128          # CEM candidate count
    cem_elite_ratio: float = 0.1       # Top 10% for CEM update
    cem_iterations: int = 5            # CEM refinement iterations
    efe_pragmatic_weight: float = 1.0
    efe_epistemic_weight: float = 1.0
    efe_empowerment_weight: float = 0.1
```

### Neuroscience Mapping

Active inference under the Free Energy Principle. The brain minimizes expected free energy by maintaining a generative model and selecting actions that bring predicted future states closer to preferred states. V-JEPA 2-AC provides a powerful generative model (PFC world model), with basal ganglia (CEM planner) selecting action policies. The epistemic term drives curiosity-driven exploration.

---

## 6. Module ④ — Imagination Engine for System 2

### Purpose

Wrap V-JEPA 2's predictor as a "mental simulation" interface for System 2 reasoning — enabling counterfactual evaluation, hypothesis testing, and scenario planning in latent space.

### Contract

```python
class ImaginationEngine(nn.Module):
    """Mental simulation via V-JEPA 2 predictor rollout.

    When System 1 confidence is low, System 2 engages this engine to:
    1. Simulate hypothetical scenarios (what-if reasoning)
    2. Evaluate counterfactuals (what-would-have-happened)
    3. Plan multi-step strategies (look-ahead search)

    All simulation happens in latent space (fast, no pixel generation).
    """

    def imagine(
        self, current_state: torch.Tensor,
        hypothetical_actions: torch.Tensor,
        steps: int = 4
    ) -> List[torch.Tensor]:
        """Simulate future states given hypothetical actions.
        Returns list of predicted future state representations."""
        ...

    def counterfactual(
        self, observed_state: torch.Tensor,
        alternative_context: torch.Tensor
    ) -> torch.Tensor:
        """What would this scene look like under different conditions?
        Uses predictor to fill in masked regions with alternative context."""
        ...

    def evaluate_scenarios(
        self, current_state: torch.Tensor,
        scenario_actions: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compare multiple scenarios, return quality scores."""
        ...
```

### Configuration

```python
@dataclass
class ImaginationConfig:
    enabled: bool = False
    max_rollout_steps: int = 16        # Maximum imagination horizon
    num_parallel_scenarios: int = 8    # Scenarios evaluated in parallel
    quality_metric: str = "coherence"  # coherence | goal_proximity | novelty
```

### Neuroscience Mapping

Hippocampal-PFC prospection circuit. The hippocampus constructs imagined future scenes (predictor rollout), while PFC evaluates them against goals (System 2 scoring). Mental simulation reuses the same representations as perception — neuroscience shows imagination activates overlapping brain areas as real perception.

---

## 7. Module ⑤ — Neuromodulated Attention Masking

### Purpose

Connect V-JEPA 2's masking mechanism to the neuromodulatory system, making attention a dynamic, brain-controlled decision.

### Contract

```python
class NeuromodulatedMasking(nn.Module):
    """Dynamic masking controlled by neuromodulatory signals.

    Maps neuromodulators to masking behavior:
      ACh (acetylcholine) → masking ratio (high ACh = more masking = sharper focus)
      NE (norepinephrine) → masking randomness (high NE = broader scan)
      DA (dopamine)       → mask reward-relevant regions less
      5-HT (serotonin)    → temporal extent of masks (exploration horizon)
    """

    def generate_mask(
        self, tokens: torch.Tensor,
        modulators: Dict[str, torch.Tensor],
        salience_map: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate attention mask conditioned on neuromodulatory state.
        Returns binary mask (B, N_patches) indicating which tokens to process."""
        ...
```

### Configuration

```python
@dataclass
class NeuromodulatedMaskingConfig:
    enabled: bool = False
    base_mask_ratio: float = 0.75          # Default masking ratio
    ach_sensitivity: float = 0.3           # How much ACh modulates ratio
    ne_sensitivity: float = 0.3            # How much NE modulates randomness
    salience_guided: bool = True           # Use salience map to guide masking
```

### Neuroscience Mapping

Attention-neuromodulation coupling. ACh from basal forebrain modulates cortical processing — high ACh sharpens receptive fields. NE from locus coeruleus controls explore-exploit tradeoff. DA from VTA biases attention toward reward-predictive stimuli. 5-HT from raphe nuclei modulates temporal horizons.

---

## 8. New Skills

| Skill Name | Purpose | Phase |
|-----------|---------|-------|
| `vjepa2-encoder-bridge` | V-JEPA 2 as VisionEncoder adapter with attentive probing | 1 |
| `vjepa2-workspace-integration` | V-JEPA 2 representations in workspace competition | 1 |
| `vjepa2-active-inference-world-model` | V-JEPA 2-AC as Active Inference generative model | 2 |
| `vjepa2-htm-temporal-bridge` | Temporal sequence extraction for HTM | 2 |
| `vjepa2-imagination-engine` | Mental simulation for System 2 reasoning | 3 |
| `vjepa2-attention-masking` | Neuromodulated dynamic masking | 3 |

## 9. Existing Skill Updates

| Skill | Update |
|-------|--------|
| `encoder-suite` | Add VJEPA2VisionEncoder as vision encoder option |
| `active-inference-agent` | Add VJEPA2WorldModel as generative model backend |
| `global-workspace-ignition` | Support V-JEPA 2 as visual modality input |
| `dual-process-reasoning` | Add ImaginationEngine for System 2 |
| `htm-spatial-temporal-reflex` | Accept temporal bridge input |
| `system-orchestrator` | Add V-JEPA 2 pipeline stages |
| `world-model-scaffold` | Concrete VJEPA2 implementations of ABCs |

---

## 10. Implementation Phases

### Phase 1: Foundation (Encoder + Workspace)

- VJEPA2VisionEncoder + attentive pooler + projection
- Config: VJEPA2EncoderConfig with feature flag
- Workspace integration (V-JEPA 2 output as modality)
- Tests: encoder shape, workspace competition, forward pass
- Skills: `vjepa2-encoder-bridge`, `vjepa2-workspace-integration`

### Phase 2: World Model (Active Inference + HTM)

- VJEPA2WorldModel wrapping V-JEPA 2-AC predictor
- CEM planner extended with EFE
- Temporal bridge to HTM
- Tests: planning rollout, EFE computation, temporal seq
- Skills: `vjepa2-active-inference-world-model`, `vjepa2-htm-temporal-bridge`

### Phase 3: Cognition (Imagination + Masking)

- ImaginationEngine for System 2
- NeuromodulatedMasking
- Tests: mental simulation, dynamic masking, scenarios
- Skills: `vjepa2-imagination-engine`, `vjepa2-attention-masking`

### Phase 4: Skills + Plugin

- Write 6 new SKILL.md files with references/assets/scripts
- Update 7 existing skills
- Update plugin.json
- Validation scripts for done-when gates

---

## 11. Done-When Gates

1. **Encoder Bridge Works** — `VJEPA2VisionEncoder` produces `(B, 4096)` output matching `VisionEncoder` contract. Forward pass through full BrainAI pipeline succeeds with V-JEPA 2 vision encoder.
2. **Workspace Integration** — V-JEPA 2 visual representation competes with text/audio/sensor in workspace. Attention weights show meaningful cross-modal competition.
3. **World Model Plans** — `VJEPA2WorldModel.plan()` produces action sequences. CEM planner converges. EFE computation includes pragmatic + epistemic + empowerment terms.
4. **Temporal Bridge** — HTM receives `(B, T, 4096)` temporal sequences from V-JEPA 2. Anomaly detection works on V-JEPA 2 temporal features.
5. **Imagination Works** — `ImaginationEngine.imagine()` produces coherent future state sequences. System 2 uses imagination for scenario comparison.
6. **Masking Responds to Modulators** — Masking ratio changes with ACh level. Masking randomness changes with NE level.

---

## 12. Research Sources

- [V-JEPA 2 Paper (arXiv:2506.09985)](https://arxiv.org/abs/2506.09985)
- [V-JEPA 2 Blog (Meta AI)](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/)
- [V-JEPA 2 GitHub](https://github.com/facebookresearch/vjepa2)
- [A Path Towards Autonomous Machine Intelligence (LeCun 2022)](https://openreview.net/pdf?id=BZ5a1r-kVsf)
- [V-JEPA 2 HuggingFace Docs](https://huggingface.co/docs/transformers/model_doc/vjepa2)
- [VL-JEPA: Vision-Language JEPA](https://arxiv.org/abs/2512.10942)
- [Predictive coding under the free-energy principle (Friston 2009)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2666703/)
- [COGITATE: Adversarial testing of GNW and IIT (Nature 2025)](https://www.nature.com/articles/s41586-025-08888-1)
- [Brain-JEPA: Brain Dynamics Foundation Model](https://openreview.net/forum?id=gtU2eLSAmO)
