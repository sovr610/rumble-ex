# Ignition Dynamics Reference

> Subsystem reference for the Global Workspace with Ignition Dynamics skill.
> Covers ignition score computation, gated broadcast, lock-in prevention,
> configuration surface, telemetry, mixed-precision safety, and migration
> from the existing codebase.

---

## 1. Overview

Ignition in the Global Neuronal Workspace (GNW) theory refers to a non-linear,
self-amplifying activation event that stabilizes a neural representation and
makes it globally accessible to all specialized processors. The concept
originates from Dehaene, Changeux, and colleagues, who demonstrated that
conscious access correlates with a sudden, all-or-nothing "ignition" of
distributed cortical activity.

The engineering equivalent: when competition winners are sufficiently stable
and confident, trigger a **committed broadcast** with high gain. Otherwise,
emit a **weak broadcast** with lower gain and conservative memory updates.
This two-regime behavior prevents noisy or ambiguous content from dominating
the workspace while still allowing tentative representations to propagate
at reduced strength.

### 1.1 Why Ignition Matters

Without ignition gating, every competition round produces a broadcast at full
strength. This leads to three failure modes:

1. **Noisy dominance** -- unstable winners broadcast loudly, confusing
   downstream modules.
2. **Memory pollution** -- working memory absorbs half-formed representations
   that overwrite useful prior context.
3. **Loss of interpretability** -- there is no clear signal indicating whether
   the workspace has "decided" or is still deliberating.

Ignition solves all three by providing a principled threshold mechanism that
distinguishes committed from tentative workspace states.

### 1.2 Scope of This Document

This reference covers:

- Theoretical grounding (Section 2)
- Ignition score computation from interpretable component signals (Section 3)
- The ignition gate that modulates broadcast gain (Section 4)
- Lock-in prevention mechanisms (Section 5)
- Full configuration surface (Section 6)
- Telemetry and observability (Section 7)
- Mixed-precision safety (Section 8)
- Migration path from the existing codebase (Section 9)
- Anti-patterns to avoid (Section 10)

---

## 2. GNW Theory Mapping

### 2.1 Theoretical Background

Dehaene and Changeux (2011) describe global ignition as a threshold
phenomenon in the Global Neuronal Workspace. Key properties:

- **Workspace neurons** possess long-range cortico-cortical connections that
  link distant brain areas (prefrontal, parietal, temporal, cingulate).
- **Local processing** occurs in specialized modules (sensory cortices) without
  requiring workspace access.
- **Ignition** occurs when a coalition of specialists provides sufficiently
  strong and coherent bottom-up input, causing the workspace to undergo a
  non-linear phase transition from subliminal to conscious processing.
- **Broadcast** follows ignition: the ignited representation is sent back to
  all specialist modules, making the content globally available.

### 2.2 Engineering Translation

Map the theoretical constructs to computational components:

| GNW Concept                    | Engineering Equivalent                        |
|-------------------------------|-----------------------------------------------|
| Workspace neurons             | `SelectionBroadcastWorkspace` slots            |
| Long-range connections        | Cross-modal attention + broadcast projections  |
| Bottom-up activation          | Salience scores from `ModalityProjection`      |
| Threshold for ignition        | `ignition_threshold` in `IgnitionConfig`       |
| Non-linear amplification      | Sigmoid gating with temperature                |
| Ignited broadcast             | Full-gain broadcast (gain = 1.0)               |
| Subliminal processing         | Weak broadcast (gain = `weak_gain`)            |
| Sustained ignition            | Working memory persistence weight              |
| Extinction / habituation      | Lock-in prevention (Section 5)                 |

### 2.3 The Ignition Equation (Conceptual)

Map the neuroscience to a scalar:

```
ignition = f(stability, confidence, margin, cross_modal_coherence)
```

Where `f` is either a weighted sum (interpretable) or a learned MLP
(production). The output is clamped to [0, 1]. When `ignition >= threshold`,
the workspace has ignited and commits to a full broadcast.

---

## 3. Ignition Score Computation

### 3.1 Component Signals

Four interpretable components feed into the ignition score. Each captures a
distinct aspect of workspace readiness.

#### 3.1.1 Stability (from Iterative Rounds)

Measure how much the competition outcome changes between successive rounds.
Stable outcomes indicate the workspace has converged.

**Winner set stability** -- compute the Jaccard similarity between the set of
winning slot indices at round `r` and round `r-1`:

```python
def winner_set_stability(
    winners_current: torch.Tensor,  # (B, K) indices
    winners_previous: torch.Tensor, # (B, K) indices
) -> torch.Tensor:
    """
    Compute Jaccard similarity between winner sets across rounds.

    Args:
        winners_current: Indices of winning slots at round r.
        winners_previous: Indices of winning slots at round r-1.

    Returns:
        stability: (B,) float in [0, 1]. 1.0 = identical winner sets.
    """
    batch_size = winners_current.shape[0]
    stability = torch.zeros(batch_size, device=winners_current.device)

    for b in range(batch_size):
        set_curr = set(winners_current[b].tolist())
        set_prev = set(winners_previous[b].tolist())
        intersection = len(set_curr & set_prev)
        union = len(set_curr | set_prev)
        stability[b] = intersection / max(union, 1)

    return stability
```

**Slot embedding stability** -- compute the mean cosine similarity between
slot embeddings at round `r` and round `r-1`:

```python
def slot_embedding_stability(
    slots_current: torch.Tensor,   # (B, num_slots, D)
    slots_previous: torch.Tensor,  # (B, num_slots, D)
) -> torch.Tensor:
    """
    Compute mean cosine similarity between slot embeddings across rounds.

    Args:
        slots_current: Slot embeddings at round r.
        slots_previous: Slot embeddings at round r-1.

    Returns:
        stability: (B,) float in [-1, 1], typically [0, 1].
    """
    # Normalize along embedding dimension
    norm_curr = F.normalize(slots_current, dim=-1)
    norm_prev = F.normalize(slots_previous, dim=-1)

    # Cosine similarity per slot, then mean across slots
    cosine = (norm_curr * norm_prev).sum(dim=-1)  # (B, num_slots)
    stability = cosine.mean(dim=-1)                # (B,)

    return stability
```

**Combined stability** -- take the maximum stability observed over the last
`M` rounds to capture peak convergence:

```python
def compute_stability(
    round_history: list[dict],
    num_lookback: int = 3,
) -> torch.Tensor:
    """
    Compute combined stability from round history.

    Use the maximum stability over the last M rounds. This captures
    the peak convergence point, which may not be the final round
    (e.g., if the final round introduces perturbation).

    Args:
        round_history: List of dicts with 'winners' and 'slots' keys.
        num_lookback: Number of past rounds to examine.

    Returns:
        stability: (B,) float in [0, 1].
    """
    if len(round_history) < 2:
        return torch.zeros(round_history[0]['slots'].shape[0],
                           device=round_history[0]['slots'].device)

    stabilities = []
    start = max(1, len(round_history) - num_lookback)

    for r in range(start, len(round_history)):
        ws = winner_set_stability(
            round_history[r]['winners'],
            round_history[r - 1]['winners'],
        )
        es = slot_embedding_stability(
            round_history[r]['slots'],
            round_history[r - 1]['slots'],
        )
        combined = 0.5 * ws + 0.5 * es
        stabilities.append(combined)

    # Max over rounds
    stacked = torch.stack(stabilities, dim=0)  # (num_rounds, B)
    max_stability, _ = stacked.max(dim=0)      # (B,)

    return max_stability
```

#### 3.1.2 Confidence (from Competition Margin)

Measure the softmax margin between the K-th winner and the (K+1)-th loser.
A larger margin indicates the competition produced a decisive outcome.

```python
def compute_confidence(
    salience_scores: torch.Tensor,  # (B, num_items)
    num_winners: int,
    temperature: float = 0.5,
) -> torch.Tensor:
    """
    Compute competition confidence from the margin between winners and losers.

    Apply softmax to salience scores, then measure the gap between the
    weakest winner and the strongest loser.

    Args:
        salience_scores: Raw salience scores from competition.
        num_winners: Number of slots selected as winners (K).
        temperature: Softmax temperature.

    Returns:
        confidence: (B,) float in [0, 1].
    """
    # Softmax over items
    probs = F.softmax(salience_scores / temperature, dim=-1)  # (B, num_items)

    # Sort descending
    sorted_probs, _ = probs.sort(dim=-1, descending=True)

    # K-th winner (0-indexed: index K-1)
    kth_winner = sorted_probs[:, num_winners - 1]

    # (K+1)-th loser (index K), clamp for safety
    if sorted_probs.shape[1] > num_winners:
        first_loser = sorted_probs[:, num_winners]
    else:
        first_loser = torch.zeros_like(kth_winner)

    # Margin as confidence signal
    margin = kth_winner - first_loser  # (B,)

    # Normalize to [0, 1] via sigmoid scaling
    confidence = torch.sigmoid(margin * 10.0)  # Scale factor for sensitivity

    return confidence
```

#### 3.1.3 Margin (Mean Score Gap)

While confidence measures the gap at the decision boundary (K-th vs (K+1)-th),
margin measures the average gap between all winners and all non-winners. This
captures how decisive the overall competition was.

```python
def compute_margin(
    salience_scores: torch.Tensor,  # (B, num_items)
    num_winners: int,
) -> torch.Tensor:
    """
    Compute mean score gap between winners and non-winners.

    Args:
        salience_scores: Raw salience scores.
        num_winners: Number of winners (K).

    Returns:
        margin: (B,) float in [0, 1].
    """
    sorted_scores, _ = salience_scores.sort(dim=-1, descending=True)

    winner_mean = sorted_scores[:, :num_winners].mean(dim=-1)

    if sorted_scores.shape[1] > num_winners:
        loser_mean = sorted_scores[:, num_winners:].mean(dim=-1)
    else:
        loser_mean = torch.zeros_like(winner_mean)

    raw_margin = winner_mean - loser_mean  # (B,)

    # Normalize via sigmoid
    margin = torch.sigmoid(raw_margin * 5.0)

    return margin
```

#### 3.1.4 Cross-Modal Coherence (Optional)

Measure whether the winning coalition spans multiple modalities and whether
its members are mutually consistent.

Two sub-signals:

1. **Modality diversity** -- do winning slots draw from more than one modality?
2. **Embedding coherence** -- are winning slot embeddings mutually consistent
   (mean pairwise cosine similarity)?

```python
def compute_cross_modal_coherence(
    winning_slots: torch.Tensor,        # (B, K, D)
    winning_modality_ids: torch.Tensor,  # (B, K) int, modality index per slot
    num_modalities: int,
) -> torch.Tensor:
    """
    Compute cross-modal coherence of the winning coalition.

    Combines modality diversity (do winners span modalities?) with
    embedding coherence (are winner embeddings mutually consistent?).

    Args:
        winning_slots: Embeddings of winning slots.
        winning_modality_ids: Integer modality index for each winner.
        num_modalities: Total number of modalities.

    Returns:
        coherence: (B,) float in [0, 1].
    """
    batch_size, K, D = winning_slots.shape

    # --- Modality diversity ---
    # Count unique modalities among winners
    diversity = torch.zeros(batch_size, device=winning_slots.device)
    for b in range(batch_size):
        unique_modalities = winning_modality_ids[b].unique().numel()
        diversity[b] = unique_modalities / max(num_modalities, 1)

    # --- Embedding coherence ---
    # Mean pairwise cosine similarity among winning slots
    norm_slots = F.normalize(winning_slots, dim=-1)  # (B, K, D)

    # Pairwise cosine: (B, K, K)
    pairwise_cos = torch.bmm(norm_slots, norm_slots.transpose(1, 2))

    # Mask diagonal (self-similarity = 1)
    eye_mask = torch.eye(K, device=winning_slots.device).unsqueeze(0)
    pairwise_cos = pairwise_cos * (1 - eye_mask)

    # Mean off-diagonal
    num_pairs = K * (K - 1)
    if num_pairs > 0:
        coherence_embed = pairwise_cos.sum(dim=(1, 2)) / num_pairs
    else:
        coherence_embed = torch.zeros(batch_size, device=winning_slots.device)

    # Clamp to [0, 1]
    coherence_embed = coherence_embed.clamp(0, 1)

    # Combine
    coherence = 0.5 * diversity + 0.5 * coherence_embed  # (B,)

    return coherence
```

### 3.2 Combining into Ignition Scalar

#### 3.2.1 Interpretable Weighted Sum (Default)

Combine the four components with configurable weights:

```python
class InterpretableIgnition(nn.Module):
    """
    Compute ignition score from interpretable component signals.

    Use this version for debugging, analysis, and experiments where
    understanding WHY ignition fires is important.
    """

    def __init__(
        self,
        w_stability: float = 0.3,
        w_confidence: float = 0.3,
        w_margin: float = 0.2,
        w_coherence: float = 0.2,
    ):
        super().__init__()
        self.w_stability = w_stability
        self.w_confidence = w_confidence
        self.w_margin = w_margin
        self.w_coherence = w_coherence

    def forward(
        self,
        stability: torch.Tensor,    # (B,)
        confidence: torch.Tensor,    # (B,)
        margin: torch.Tensor,        # (B,)
        coherence: torch.Tensor,     # (B,)
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute ignition score as weighted sum of components.

        Returns:
            ignition: (B,) float clamped to [0, 1].
            components: Dict of individual component scores for telemetry.
        """
        ignition = (
            self.w_stability * stability
            + self.w_confidence * confidence
            + self.w_margin * margin
            + self.w_coherence * coherence
        )

        ignition = ignition.clamp(0.0, 1.0)

        components = {
            'stability': stability,
            'confidence': confidence,
            'margin': margin,
            'coherence': coherence,
        }

        return ignition, components
```

#### 3.2.2 Learned MLP (Production Option)

For production, use a small MLP that takes the four components and produces
the ignition score. This allows the model to learn non-linear interactions
between components.

```python
class LearnedIgnition(nn.Module):
    """
    Compute ignition score via a learned MLP over component signals.

    Use this version for production when maximizing performance is
    more important than interpretability. The MLP can learn non-linear
    interactions (e.g., stability matters more when confidence is low).

    Always log the raw component signals alongside the MLP output
    for post-hoc analysis.
    """

    def __init__(
        self,
        num_components: int = 4,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_components, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        stability: torch.Tensor,    # (B,)
        confidence: torch.Tensor,    # (B,)
        margin: torch.Tensor,        # (B,)
        coherence: torch.Tensor,     # (B,)
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute ignition score via learned MLP.

        Returns:
            ignition: (B,) float in [0, 1].
            components: Dict of individual component scores for telemetry.
        """
        stacked = torch.stack([stability, confidence, margin, coherence], dim=-1)
        ignition = self.mlp(stacked).squeeze(-1)  # (B,)

        components = {
            'stability': stability,
            'confidence': confidence,
            'margin': margin,
            'coherence': coherence,
        }

        return ignition, components
```

#### 3.2.3 Factory Function

Select between interpretable and learned ignition based on config:

```python
def create_ignition_scorer(config: "IgnitionConfig") -> nn.Module:
    """
    Create the ignition scorer module.

    Args:
        config: IgnitionConfig with weights and use_learned_ignition flag.

    Returns:
        InterpretableIgnition or LearnedIgnition module.
    """
    if config.use_learned_ignition:
        return LearnedIgnition(
            num_components=4,
            hidden_dim=32,
        )
    else:
        return InterpretableIgnition(
            w_stability=config.w_stability,
            w_confidence=config.w_confidence,
            w_margin=config.w_margin,
            w_coherence=config.w_coherence,
        )
```

### 3.3 Ignition Threshold

Apply a threshold to the ignition scalar to determine broadcast mode:

- If `ignition >= ignition_threshold` --> **ignited** (committed broadcast)
- If `ignition < ignition_threshold` --> **not ignited** (weak broadcast)

The threshold is configurable via `IgnitionConfig.ignition_threshold`
(default: 0.3).

**Choosing the threshold:**

- Lower thresholds (0.1 - 0.2): ignition fires easily; more content reaches
  full broadcast. Use when downstream modules need frequent updates.
- Moderate thresholds (0.3 - 0.5): balanced behavior. Default recommendation.
- Higher thresholds (0.6 - 0.8): ignition fires rarely; only very confident,
  stable coalitions trigger full broadcast. Use for safety-critical or
  low-noise applications.

```python
def apply_ignition_threshold(
    ignition_score: torch.Tensor,  # (B,)
    threshold: float,
    hard: bool = True,
) -> torch.Tensor:
    """
    Apply threshold to ignition score.

    Args:
        ignition_score: Computed ignition scalar per batch element.
        threshold: Ignition threshold.
        hard: If True, return binary 0/1. If False, return the score
              (useful for soft gating during training).

    Returns:
        ignited: (B,) float. 1.0 if ignited, 0.0 otherwise (hard mode).
    """
    if hard:
        return (ignition_score >= threshold).float()
    else:
        return ignition_score
```

---

## 4. Ignition Gate

The ignition gate translates the binary ignited/not-ignited decision into a
continuous gain factor that modulates the broadcast strength. This is the
core mechanism that connects ignition detection to downstream behavior.

### 4.1 Committed Broadcast (ignited=True)

When ignition fires:

- **Broadcast gain** = 1.0 (full strength).
- **Working memory update** uses full persistence weight: the winning slots
  are written into working memory with high priority.
- **Telemetry** emits `ignited=True`.
- **Workspace state** is updated confidently: the previous context is fully
  replaced by the new workspace content.

This is the normal operating mode when the workspace has converged on a
clear, stable representation.

### 4.2 Weak Broadcast (ignited=False)

When ignition does not fire:

- **Broadcast gain** = `weak_gain` (default: 0.3). The winning content still
  propagates, but at reduced amplitude.
- **Working memory update** is partial: `wm_weight = weak_gain`. The new
  content is mixed with existing memory rather than replacing it.
- **Workspace state** update is conservative: the output is a weighted blend
  of the new workspace content and the previous context.

```python
# Weak broadcast memory update
new_memory = weak_gain * current_content + (1 - weak_gain) * previous_memory
```

This prevents unstable or ambiguous content from dominating the workspace
while still allowing tentative representations to have some influence.

### 4.3 Smooth Gating (Training)

During training, use a smooth sigmoid gate instead of a hard threshold.
Hard thresholds create zero gradients at the boundary, which blocks learning.
The sigmoid approximation preserves gradient flow.

```
effective_gain = sigmoid((ignition_score - threshold) / temperature)
```

- **temperature** controls the sharpness of the transition.
  - Low temperature (0.01 - 0.05): nearly hard threshold, sharp transition.
  - Moderate temperature (0.1 - 0.2): smooth but still decisive.
  - High temperature (0.5 - 1.0): very gradual transition, useful early in
    training when ignition scores are noisy.

During inference, use the hard threshold for deterministic behavior.

### 4.4 Implementation Pattern

```python
class IgnitionGate(nn.Module):
    """
    Gate that converts ignition score into broadcast gain.

    During training, use smooth sigmoid gating to preserve gradients.
    During inference, use hard threshold for determinism.

    The gate modulates:
    1. Broadcast amplitude (how strongly winners propagate)
    2. Working memory write weight (how persistently winners are stored)
    3. Workspace state update rate (how much previous context is replaced)
    """

    def __init__(
        self,
        threshold: float = 0.3,
        weak_gain: float = 0.3,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.threshold = threshold
        self.weak_gain = weak_gain
        self.temperature = temperature

    def forward(
        self,
        slots: torch.Tensor,          # (B, K, D)
        ignition_score: torch.Tensor,  # (B,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply ignition gate to winning slots.

        Args:
            slots: Winning slot embeddings from competition.
            ignition_score: Computed ignition scalar per batch element.

        Returns:
            gated_slots: Slots modulated by ignition gain.
            effective_gain: (B,) the applied gain per batch element.
        """
        if self.training:
            # Smooth sigmoid for gradient flow
            raw_gain = torch.sigmoid(
                (ignition_score - self.threshold) / self.temperature
            )
        else:
            # Hard threshold for determinism
            raw_gain = (ignition_score >= self.threshold).float()

        # Interpolate between weak_gain and 1.0 based on raw_gain
        # When raw_gain=0 (not ignited): effective_gain = weak_gain
        # When raw_gain=1 (ignited): effective_gain = 1.0
        effective_gain = self.weak_gain + (1.0 - self.weak_gain) * raw_gain

        # Apply gain to slots: (B,) -> (B, 1, 1) for broadcasting
        gated_slots = slots * effective_gain.unsqueeze(-1).unsqueeze(-1)

        return gated_slots, effective_gain
```

### 4.5 Gate Integration with Workspace Forward Pass

Show how the ignition gate integrates into the `SelectionBroadcastWorkspace`
forward pass:

```python
def forward(self, modality_inputs, modality_states=None, return_details=False):
    # ... (steps 1-3: project, compete, aggregate) ...

    # Compute interpretable ignition score
    stability = compute_stability(competition_info['history'])
    confidence = compute_confidence(
        competition_info['saliences'], self.config.capacity_limit
    )
    margin = compute_margin(
        competition_info['saliences'], self.config.capacity_limit
    )
    coherence = compute_cross_modal_coherence(
        winning_slots, modality_ids, num_modalities
    )

    ignition_score, components = self.ignition_scorer(
        stability, confidence, margin, coherence
    )

    # Apply ignition gate
    gated_winners, effective_gain = self.ignition_gate(
        winners, ignition_score
    )

    # Aggregate gated winners
    workspace_content = gated_winners.sum(dim=1)

    # Working memory update with gain-modulated persistence
    memory_result = self.working_memory(
        workspace_content,
        write_weight=effective_gain,  # Higher gain = stronger write
    )

    # ... (broadcast, output) ...

    output['ignition_score'] = ignition_score
    output['ignited'] = (ignition_score >= self.ignition_gate.threshold).float()
    output['effective_gain'] = effective_gain
    output['ignition_components'] = components
```

### 4.6 Memory Write Weight Modulation

The ignition gate also modulates how strongly the workspace content is written
into working memory:

```python
class GatedWorkingMemoryUpdate:
    """
    Modulate working memory writes by ignition gain.

    When ignited: write at full strength, replacing old content.
    When not ignited: write at reduced strength, blending with old content.
    """

    def update(
        self,
        memory_buffer: torch.Tensor,    # (B, buffer_size, D)
        new_content: torch.Tensor,       # (B, D)
        write_weight: torch.Tensor,      # (B,) from ignition gate
        write_index: int,
    ) -> torch.Tensor:
        """
        Write new content into memory buffer with gain modulation.

        Args:
            memory_buffer: Current memory buffer.
            new_content: Content to write.
            write_weight: Ignition-gated write strength.
            write_index: Buffer position to write to.

        Returns:
            Updated memory buffer.
        """
        # Expand write_weight for broadcasting: (B,) -> (B, 1)
        w = write_weight.unsqueeze(-1)

        # Blend new content with existing memory at write position
        old_content = memory_buffer[:, write_index, :]
        blended = w * new_content + (1 - w) * old_content

        # Write blended content
        memory_buffer = memory_buffer.clone()
        memory_buffer[:, write_index, :] = blended

        return memory_buffer
```

---

## 5. Lock-In Prevention

### 5.1 The Problem

Without countermeasures, the same coalition of tokens wins every competition
round, every timestep. The workspace becomes "stuck" on one representation,
ignoring novel or changing inputs. This is analogous to perseveration in
neuroscience -- an inability to disengage from a currently active
representation.

Lock-in manifests in three ways:

1. **Within-timestep lock-in**: the same slots win every iterative round,
   preventing the competition from exploring alternatives.
2. **Across-timestep lock-in**: the same modality or content type dominates
   the workspace across consecutive forward passes.
3. **Training lock-in**: the model learns to always select the same features,
   reducing the effective capacity of the workspace.

### 5.2 Novelty Term

Compute a novelty score for each competing token relative to the current
working memory contents. Tokens similar to what is already in memory receive
a lower novelty score; genuinely new content receives a boost.

```python
def compute_novelty(
    tokens: torch.Tensor,          # (B, num_items, D)
    memory_buffer: torch.Tensor,   # (B, buffer_size, D)
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Compute novelty of each token relative to working memory.

    Novelty = 1 - max_cosine_similarity(token, memory).
    Tokens already represented in memory get low novelty;
    genuinely new content gets high novelty.

    Args:
        tokens: Competing tokens.
        memory_buffer: Current working memory contents.
        epsilon: Numerical stability.

    Returns:
        novelty: (B, num_items) float in [0, 1].
    """
    # Normalize
    tokens_norm = F.normalize(tokens, dim=-1)             # (B, N, D)
    memory_norm = F.normalize(memory_buffer, dim=-1)      # (B, M, D)

    # Cosine similarity: (B, N, M)
    similarity = torch.bmm(tokens_norm, memory_norm.transpose(1, 2))

    # Max similarity to any memory slot
    max_sim, _ = similarity.max(dim=-1)  # (B, N)

    # Novelty = 1 - max_similarity
    novelty = 1.0 - max_sim.clamp(0, 1)

    return novelty
```

Incorporate novelty into competition scoring by adding it as a bias to the
salience scores before competition begins:

```python
# In the competition forward pass:
if self.use_novelty and memory_buffer is not None:
    novelty = compute_novelty(projected_tokens, memory_buffer)
    saliences = saliences + self.novelty_weight * novelty.unsqueeze(-1)
```

### 5.3 Slot Dropout (Training Only)

During training, randomly zero out entire slots with probability
`p_slot_dropout`. This forces the workspace to discover alternative coalitions
and prevents over-reliance on a small number of dominant inputs.

```python
class SlotDropout(nn.Module):
    """
    Randomly drop entire slots during training.

    This forces the workspace to find alternative coalitions and
    prevents over-reliance on a small number of dominant inputs.

    During inference, no dropout is applied -- all slots participate.
    """

    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """
        Apply slot dropout.

        Args:
            slots: (B, num_slots, D) slot embeddings.

        Returns:
            Slots with some randomly zeroed out (training only).
        """
        if not self.training or self.p == 0.0:
            return slots

        batch_size, num_slots, D = slots.shape

        # Generate dropout mask per slot (not per element)
        # Shape: (B, num_slots, 1) -- broadcast over D
        keep_prob = 1.0 - self.p
        mask = torch.bernoulli(
            torch.full(
                (batch_size, num_slots, 1),
                keep_prob,
                device=slots.device,
                dtype=slots.dtype,
            )
        )

        # Scale by 1/keep_prob to maintain expected value
        return slots * mask / keep_prob
```

**Important:** Never apply slot dropout during inference. Dropping slots at
inference time introduces non-determinism and degrades output quality.

### 5.4 Winner Decay

Apply a penalty to tokens that won in the previous timestep. This encourages
turnover in the workspace unless a token is genuinely the best choice and
overcomes the decay penalty.

```python
class WinnerDecay:
    """
    Apply a decay penalty to previous winners.

    Tokens that won in the previous timestep receive a score penalty,
    encouraging the workspace to attend to new content. The penalty
    is configurable: higher decay means more turnover; lower decay
    means winners persist longer.

    This is analogous to inhibition of return in neuroscience.
    """

    def __init__(self, decay_factor: float = 0.1):
        self.decay_factor = decay_factor
        self.previous_winners: torch.Tensor | None = None

    def apply(
        self,
        salience_scores: torch.Tensor,  # (B, num_items)
    ) -> torch.Tensor:
        """
        Apply decay penalty to previous winners.

        Args:
            salience_scores: Current salience scores.

        Returns:
            Adjusted salience scores with decay applied.
        """
        if self.previous_winners is None:
            return salience_scores

        # previous_winners is a binary mask: (B, num_items)
        # where 1.0 = was a winner last timestep
        adjusted = salience_scores - self.decay_factor * self.previous_winners

        return adjusted

    def update(
        self,
        attention_weights: torch.Tensor,  # (B, num_items)
        threshold: float = 0.1,
    ):
        """
        Record current winners for next timestep decay.

        Args:
            attention_weights: Final attention weights from competition.
            threshold: Minimum attention weight to count as a winner.
        """
        self.previous_winners = (attention_weights > threshold).float().detach()

    def reset(self):
        """Reset winner history."""
        self.previous_winners = None
```

### 5.5 Ignition Cooldown

After a committed broadcast (ignition fires), temporarily raise the ignition
threshold. This prevents the workspace from igniting on every single timestep,
which would defeat the purpose of having two broadcast regimes.

```python
class IgnitionCooldown:
    """
    Temporarily raise ignition threshold after ignition fires.

    Prevents continuous ignition by introducing a refractory period.
    After ignition, the effective threshold is raised by cooldown_boost
    for cooldown_steps timesteps.

    This is analogous to the refractory period in biological neurons.
    """

    def __init__(
        self,
        cooldown_steps: int = 0,
        cooldown_boost: float = 0.5,
    ):
        self.cooldown_steps = cooldown_steps
        self.cooldown_boost = cooldown_boost
        self.steps_since_ignition: int = 0
        self.cooldown_active: bool = False

    def get_effective_threshold(self, base_threshold: float) -> float:
        """
        Compute the effective ignition threshold, accounting for cooldown.

        Args:
            base_threshold: The configured ignition threshold.

        Returns:
            Effective threshold (possibly raised if cooldown is active).
        """
        if self.cooldown_active and self.steps_since_ignition < self.cooldown_steps:
            return base_threshold * (1.0 + self.cooldown_boost)
        else:
            self.cooldown_active = False
            return base_threshold

    def record_ignition(self, ignited: bool):
        """
        Record whether ignition fired at the current timestep.

        Args:
            ignited: True if ignition fired.
        """
        if ignited:
            self.steps_since_ignition = 0
            self.cooldown_active = True
        elif self.cooldown_active:
            self.steps_since_ignition += 1
            if self.steps_since_ignition >= self.cooldown_steps:
                self.cooldown_active = False

    def reset(self):
        """Reset cooldown state."""
        self.steps_since_ignition = 0
        self.cooldown_active = False
```

### 5.6 Combining Lock-In Prevention Mechanisms

All four mechanisms work together. Apply them in order:

1. **Novelty term**: compute before competition. Add to salience scores.
2. **Winner decay**: compute before competition. Subtract from salience scores.
3. **Slot dropout**: apply during competition (training only).
4. **Ignition cooldown**: apply after competition, when evaluating threshold.

```python
def apply_lock_in_prevention(
    salience_scores: torch.Tensor,     # (B, num_items)
    projected_tokens: torch.Tensor,    # (B, num_items, D)
    memory_buffer: torch.Tensor | None,
    winner_decay: WinnerDecay,
    novelty_weight: float = 0.1,
    slot_dropout: SlotDropout | None = None,
    training: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply all lock-in prevention mechanisms to salience scores and tokens.

    Args:
        salience_scores: Initial salience scores.
        projected_tokens: Projected token embeddings.
        memory_buffer: Working memory contents (for novelty).
        winner_decay: WinnerDecay instance.
        novelty_weight: Weight for novelty bonus.
        slot_dropout: Optional SlotDropout (training only).
        training: Whether in training mode.

    Returns:
        adjusted_saliences: Salience scores after novelty + decay.
        adjusted_tokens: Tokens after optional slot dropout.
    """
    adjusted = salience_scores

    # 1. Add novelty bonus
    if memory_buffer is not None and novelty_weight > 0:
        novelty = compute_novelty(projected_tokens, memory_buffer)
        adjusted = adjusted + novelty_weight * novelty

    # 2. Apply winner decay
    adjusted = winner_decay.apply(adjusted)

    # 3. Apply slot dropout to tokens (training only)
    adjusted_tokens = projected_tokens
    if slot_dropout is not None and training:
        adjusted_tokens = slot_dropout(adjusted_tokens)

    return adjusted, adjusted_tokens
```

---

## 6. IgnitionConfig Surface

All ignition-related configuration fields. These fields extend
`SelectionBroadcastConfig` or exist as a separate `IgnitionConfig` dataclass.

### 6.1 Configuration Dataclass

```python
@dataclass
class IgnitionConfig:
    """
    Configuration for ignition dynamics.

    Controls ignition score computation, broadcast gating,
    lock-in prevention, and telemetry.
    """

    # --- Ignition threshold ---
    ignition_threshold: float = 0.3
    """
    Threshold for committed broadcast. Ignition scores at or above
    this value trigger full-strength broadcast. Below this value,
    broadcast uses weak_gain.

    Range: [0.0, 1.0]. Lower = ignites more easily.
    Default: 0.3 (moderate, fires on reasonably stable coalitions).
    """

    # --- Component weights (interpretable mode) ---
    w_stability: float = 0.3
    """
    Weight for the stability component in ignition score.
    Measures convergence of winner sets across iterative rounds.
    """

    w_confidence: float = 0.3
    """
    Weight for the confidence component in ignition score.
    Measures the softmax margin at the winner/loser boundary.
    """

    w_margin: float = 0.2
    """
    Weight for the margin component in ignition score.
    Measures mean score gap between all winners and all losers.
    """

    w_coherence: float = 0.2
    """
    Weight for the cross-modal coherence component.
    Measures modality diversity and embedding consistency among winners.
    """

    # --- Broadcast gating ---
    weak_gain: float = 0.3
    """
    Broadcast gain when ignition does NOT fire.
    Full-strength broadcast uses gain=1.0; weak broadcast uses this value.

    Range: [0.0, 1.0]. Lower = weaker broadcast when not ignited.
    Set to 0.0 to completely suppress broadcast when not ignited.
    Set to 1.0 to effectively disable ignition gating.
    """

    gate_temperature: float = 0.1
    """
    Temperature for smooth sigmoid gating during training.
    Lower = sharper transition (closer to hard threshold).
    Higher = smoother transition (more gradient flow).

    Only affects training. Inference always uses hard threshold.
    """

    # --- Ignition mode ---
    use_learned_ignition: bool = False
    """
    If True, use a learned MLP to compute ignition from components.
    If False, use interpretable weighted sum.

    The learned version can capture non-linear interactions but is
    harder to debug. Always log raw component signals regardless.
    """

    # --- Lock-in prevention ---
    slot_dropout: float = 0.1
    """
    Probability of dropping a slot during training.
    Forces the workspace to discover alternative coalitions.

    Range: [0.0, 1.0). Set to 0.0 to disable.
    Only applies during training; inference uses all slots.
    """

    winner_decay: float = 0.1
    """
    Decay penalty applied to previous timestep's winners.
    Encourages turnover in the workspace.

    Range: [0.0, inf). Set to 0.0 to disable.
    Typical range: 0.05 - 0.2.
    """

    cooldown_steps: int = 0
    """
    Number of timesteps to raise the ignition threshold after ignition.
    Prevents continuous ignition on every timestep.

    Set to 0 to disable cooldown.
    Typical range: 1 - 5 timesteps.
    """

    cooldown_boost: float = 0.5
    """
    Multiplicative boost to ignition threshold during cooldown.
    Effective threshold = threshold * (1 + cooldown_boost).

    Only applies when cooldown_steps > 0 and cooldown is active.
    """
```

### 6.2 Configuration Field Reference Table

| Field                 | Type    | Default | Range           | Description                                        |
|-----------------------|---------|---------|-----------------|---------------------------------------------------|
| `ignition_threshold`  | `float` | 0.3     | [0.0, 1.0]      | Threshold for committed broadcast                  |
| `w_stability`         | `float` | 0.3     | [0.0, 1.0]      | Weight for stability component                     |
| `w_confidence`        | `float` | 0.3     | [0.0, 1.0]      | Weight for confidence component                    |
| `w_margin`            | `float` | 0.2     | [0.0, 1.0]      | Weight for margin component                        |
| `w_coherence`         | `float` | 0.2     | [0.0, 1.0]      | Weight for cross-modal coherence component         |
| `weak_gain`           | `float` | 0.3     | [0.0, 1.0]      | Broadcast gain when not ignited                    |
| `gate_temperature`    | `float` | 0.1     | (0.0, inf)       | Sigmoid temperature for smooth gating              |
| `use_learned_ignition`| `bool`  | False   | True/False       | Use learned MLP vs interpretable weighted sum      |
| `slot_dropout`        | `float` | 0.1     | [0.0, 1.0)      | Slot dropout probability (training only)           |
| `winner_decay`        | `float` | 0.1     | [0.0, inf)       | Decay penalty for previous winners                 |
| `cooldown_steps`      | `int`   | 0       | [0, inf)         | Refractory steps after ignition                    |
| `cooldown_boost`      | `float` | 0.5     | [0.0, inf)       | Threshold boost during cooldown                    |

### 6.3 Preset Configurations

```python
@classmethod
def default(cls) -> "IgnitionConfig":
    """Standard ignition configuration."""
    return cls()

@classmethod
def aggressive(cls) -> "IgnitionConfig":
    """
    Low threshold, minimal lock-in prevention.
    Use when the workspace needs to ignite frequently.
    """
    return cls(
        ignition_threshold=0.15,
        weak_gain=0.5,
        slot_dropout=0.05,
        winner_decay=0.05,
        cooldown_steps=0,
    )

@classmethod
def conservative(cls) -> "IgnitionConfig":
    """
    High threshold, strong lock-in prevention.
    Use when only very confident representations should broadcast.
    """
    return cls(
        ignition_threshold=0.6,
        weak_gain=0.1,
        slot_dropout=0.15,
        winner_decay=0.2,
        cooldown_steps=3,
        cooldown_boost=0.8,
    )

@classmethod
def debug(cls) -> "IgnitionConfig":
    """
    Interpretable mode with all prevention mechanisms active.
    Use for diagnosing ignition behavior.
    """
    return cls(
        use_learned_ignition=False,
        slot_dropout=0.1,
        winner_decay=0.1,
        cooldown_steps=2,
        cooldown_boost=0.5,
    )
```

### 6.4 Weight Normalization Constraint

The component weights `w_stability`, `w_confidence`, `w_margin`, and
`w_coherence` do not need to sum to 1.0 because the output is clamped to
[0, 1]. However, for interpretability, keep them in a range where the
weighted sum naturally falls in [0, 1] when all components are in [0, 1].

A good rule of thumb: ensure the sum of weights is between 0.8 and 1.2.

```python
def validate_ignition_config(config: IgnitionConfig) -> list[str]:
    """
    Validate ignition configuration and return warnings.

    Returns:
        List of warning messages. Empty if no issues.
    """
    warnings = []

    weight_sum = (
        config.w_stability
        + config.w_confidence
        + config.w_margin
        + config.w_coherence
    )

    if weight_sum < 0.5:
        warnings.append(
            f"Component weight sum ({weight_sum:.2f}) is very low. "
            f"Ignition score will rarely exceed threshold."
        )
    if weight_sum > 1.5:
        warnings.append(
            f"Component weight sum ({weight_sum:.2f}) is very high. "
            f"Ignition score will saturate at 1.0 frequently."
        )

    if config.ignition_threshold <= 0.0:
        warnings.append("ignition_threshold <= 0: ignition always fires.")

    if config.ignition_threshold >= 1.0:
        warnings.append("ignition_threshold >= 1: ignition never fires.")

    if config.weak_gain >= 1.0:
        warnings.append("weak_gain >= 1.0: ignition gate has no effect.")

    if config.gate_temperature <= 0.0:
        warnings.append("gate_temperature <= 0: division by zero in sigmoid.")

    if config.slot_dropout >= 1.0:
        warnings.append("slot_dropout >= 1.0: all slots dropped in training.")

    return warnings
```

---

## 7. Telemetry

Ignition exposes detailed diagnostic information via `return_details=True`
in the workspace forward pass.

### 7.1 Telemetry Fields

| Field                    | Shape       | Type    | Description                                            |
|--------------------------|-------------|---------|--------------------------------------------------------|
| `ignition_score`         | `(B,)`      | `float` | Raw ignition scalar in [0, 1]                          |
| `ignited`                | `(B,)`      | `bool`  | Whether ignition fired (score >= threshold)            |
| `effective_gain`         | `(B,)`      | `float` | Applied broadcast gain (weak_gain to 1.0)              |
| `component_scores`       | `dict`      | `dict`  | Individual component values (see below)                |
| `winner_decay_applied`   | scalar      | `bool`  | Whether winner decay was applied this timestep         |
| `cooldown_active`        | scalar      | `bool`  | Whether cooldown is currently raising the threshold    |
| `effective_threshold`    | scalar      | `float` | Actual threshold used (may differ from config if cooldown) |

### 7.2 Component Scores Dict

The `component_scores` dict contains:

| Key           | Shape  | Description                                      |
|---------------|--------|--------------------------------------------------|
| `stability`   | `(B,)` | Winner set + embedding stability across rounds   |
| `confidence`  | `(B,)` | Softmax margin at winner/loser boundary          |
| `margin`      | `(B,)` | Mean score gap between winners and losers        |
| `coherence`   | `(B,)` | Cross-modal diversity + embedding coherence      |

### 7.3 Telemetry Integration Pattern

```python
def build_ignition_telemetry(
    ignition_score: torch.Tensor,
    effective_gain: torch.Tensor,
    components: dict[str, torch.Tensor],
    threshold: float,
    cooldown: IgnitionCooldown,
    winner_decay: WinnerDecay,
) -> dict[str, object]:
    """
    Build the ignition telemetry dict for return_details=True.

    Args:
        ignition_score: Computed ignition scalar.
        effective_gain: Applied broadcast gain.
        components: Dict of component scores.
        threshold: Base ignition threshold.
        cooldown: IgnitionCooldown instance.
        winner_decay: WinnerDecay instance.

    Returns:
        Telemetry dict suitable for logging, visualization, and debugging.
    """
    ignited = (ignition_score >= cooldown.get_effective_threshold(threshold))

    return {
        'ignition_score': ignition_score.detach(),
        'ignited': ignited.detach(),
        'effective_gain': effective_gain.detach(),
        'component_scores': {
            k: v.detach() for k, v in components.items()
        },
        'winner_decay_applied': winner_decay.previous_winners is not None,
        'cooldown_active': cooldown.cooldown_active,
        'effective_threshold': cooldown.get_effective_threshold(threshold),
    }
```

### 7.4 Logging Recommendations

Log the following at each training step (aggregated over the batch):

```python
def log_ignition_metrics(telemetry: dict, step: int, logger):
    """
    Log ignition telemetry to the training logger.

    Aggregate over the batch and log mean values.

    Args:
        telemetry: Ignition telemetry dict.
        step: Current training step.
        logger: Logger instance (TensorBoard, WandB, etc.).
    """
    # Ignition rate: what fraction of the batch ignited?
    ignition_rate = telemetry['ignited'].float().mean().item()
    logger.log_scalar('workspace/ignition_rate', ignition_rate, step)

    # Mean ignition score
    mean_score = telemetry['ignition_score'].mean().item()
    logger.log_scalar('workspace/ignition_score_mean', mean_score, step)

    # Effective gain distribution
    mean_gain = telemetry['effective_gain'].mean().item()
    logger.log_scalar('workspace/effective_gain_mean', mean_gain, step)

    # Component breakdown
    for name, values in telemetry['component_scores'].items():
        logger.log_scalar(
            f'workspace/ignition_{name}_mean',
            values.mean().item(),
            step,
        )

    # Cooldown and decay state
    logger.log_scalar(
        'workspace/cooldown_active',
        float(telemetry['cooldown_active']),
        step,
    )
    logger.log_scalar(
        'workspace/effective_threshold',
        telemetry['effective_threshold'],
        step,
    )
```

### 7.5 Diagnostic Use Cases

**Ignition never fires:**
- Check `component_scores`: which component is consistently low?
- Check `effective_threshold`: is cooldown raising it too high?
- Lower `ignition_threshold` or adjust component weights.

**Ignition fires every timestep:**
- Check `ignition_rate`: if consistently 1.0, threshold is too low.
- Enable cooldown (`cooldown_steps > 0`).
- Raise `ignition_threshold`.

**Workspace stuck on same content:**
- Check `winner_decay_applied`: is it active?
- Increase `winner_decay` factor.
- Check novelty term: is `novelty_weight` > 0?
- Increase `slot_dropout` during training.

**Ignition score oscillates wildly:**
- Competition may not be converging. Increase `selection_rounds`.
- Check `stability` component: if consistently low, the iterative
  competition is not settling.
- Increase `competition_temperature` to soften competition.

---

## 8. Mixed-Precision Safety

Ignition involves threshold comparisons and gain multiplications that are
sensitive to floating-point precision. Follow these rules to avoid numerical
issues under AMP (automatic mixed precision).

### 8.1 Precision Requirements by Operation

| Operation                          | Required Precision | Reason                                   |
|------------------------------------|--------------------|------------------------------------------|
| Component signal computation       | fp32               | Cosine similarity, Jaccard index         |
| Ignition score combination         | fp32               | Weighted sum, clamping                   |
| Threshold comparison               | fp32               | fp16 has poor resolution near threshold  |
| Sigmoid gating                     | fp32               | Sigmoid saturation in fp16               |
| Gain multiplication                | fp32 then cast     | Multiply in fp32, cast result to slot dtype |
| Broadcast after gating             | slot dtype (bf16)  | Normal forward pass precision            |

### 8.2 Implementation Pattern

```python
def compute_ignition_safe(
    components: dict[str, torch.Tensor],
    weights: dict[str, float],
    threshold: float,
    temperature: float,
    training: bool,
    slot_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute ignition score and gain with mixed-precision safety.

    All intermediate computations use fp32. The final gain is cast
    back to the slot dtype for downstream use.

    Args:
        components: Dict of component signals (may be in any dtype).
        weights: Dict of component weights.
        threshold: Ignition threshold.
        temperature: Sigmoid temperature.
        training: Whether in training mode.
        slot_dtype: dtype of slot embeddings (e.g., torch.bfloat16).

    Returns:
        ignition_score: (B,) in fp32.
        gain: (B,) in slot_dtype.
    """
    # Cast all components to fp32
    fp32_components = {
        k: v.float() for k, v in components.items()
    }

    # Compute ignition score in fp32
    ignition_score = sum(
        weights[k] * fp32_components[k] for k in fp32_components
    ).clamp(0.0, 1.0)

    # Compute gain in fp32
    if training:
        gain = torch.sigmoid(
            (ignition_score - threshold) / max(temperature, 1e-8)
        )
    else:
        gain = (ignition_score >= threshold).float()

    # Cast gain to slot dtype for downstream multiplication
    gain_typed = gain.to(slot_dtype)

    return ignition_score, gain_typed
```

### 8.3 AMP Context Manager Integration

When using `torch.cuda.amp.autocast`, the ignition computation should be
explicitly run in fp32:

```python
with torch.cuda.amp.autocast(enabled=False):
    # Force fp32 for ignition
    ignition_score, gain = compute_ignition_safe(
        components=component_signals,
        weights=ignition_weights,
        threshold=self.config.ignition_threshold,
        temperature=self.config.gate_temperature,
        training=self.training,
        slot_dtype=slots.dtype,
    )
```

### 8.4 Common fp16 Pitfalls

- **Threshold comparison in fp16**: fp16 has only 3-4 decimal digits of
  precision. A threshold of 0.3 and a score of 0.301 may compare as equal
  in fp16. Always compare in fp32.
- **Sigmoid saturation**: `sigmoid(x)` in fp16 saturates to exactly 0.0 or
  1.0 for moderate `|x|` values, losing gradient flow. Use fp32.
- **Cosine similarity**: small vectors in fp16 can produce NaN from division
  by near-zero norms. Use `F.normalize` with `eps=1e-6` in fp32.

---

## 9. Migration from Existing Code

### 9.1 From Current IgnitionDetector

**Current implementation** (in `IterativeCompetition`, lines 607-612 of
`brain_ai/workspace/global_workspace.py`):

```python
# Current: learned MLP ignition detector
self.ignition_detector = nn.Sequential(
    nn.Linear(workspace_dim, workspace_dim // 4),
    nn.ReLU(),
    nn.Linear(workspace_dim // 4, 1),
    nn.Sigmoid(),
)

# Used as:
ignition = self.ignition_detector(current_features.mean(dim=1))
```

**Target implementation**: interpretable 4-component ignition score with an
optional learned mode.

**Migration steps:**

1. Add `IgnitionConfig` dataclass alongside `SelectionBroadcastConfig`.
2. Create `InterpretableIgnition` and `LearnedIgnition` modules.
3. Add component signal extraction to `IterativeCompetition.forward`:
   - Track winner indices per round for stability computation.
   - Extract salience scores for confidence and margin computation.
   - Track modality IDs for coherence computation.
4. Replace `self.ignition_detector` with `self.ignition_scorer` from
   `create_ignition_scorer(config)`.
5. Keep `use_learned_ignition=True` as a backward-compatible option that
   preserves the old MLP behavior (with component signals as inputs instead
   of raw features).

**Backward compatibility**: set `use_learned_ignition=True` to get behavior
closest to the current MLP detector. The learned version takes structured
component signals rather than raw feature means, so it is not identical but
has equivalent capacity.

### 9.2 From Current Confidence Gating

**Current implementation** (in `SelectionBroadcastWorkspace`, lines 897-903):

```python
# Current: separate confidence estimator
if self.config.use_confidence_gating:
    self.confidence_estimator = nn.Sequential(
        nn.Linear(self.config.workspace_dim, self.config.workspace_dim // 4),
        nn.ReLU(),
        nn.Linear(self.config.workspace_dim // 4, 1),
        nn.Sigmoid(),
    )

# Used as:
confidence = self.confidence_estimator(refined_content)
output_workspace = refined_content * confidence
```

**Target**: integrate confidence into the ignition score and use the ignition
gate for broadcast gain modulation.

**Migration steps:**

1. Remove `self.confidence_estimator` from `SelectionBroadcastWorkspace`.
2. The confidence signal is now computed from competition margin (Section 3.1.2)
   and fed into the ignition score.
3. Replace `refined_content * confidence` with `gated_slots` from the
   ignition gate (Section 4.4).
4. The `confidence` field in the output dict is replaced by `ignition_score`
   and `effective_gain`.

**Backward compatibility**: if `use_confidence_gating=True` is set but
ignition is not configured, fall back to the old confidence estimator.
Log a deprecation warning.

```python
def _maybe_apply_legacy_confidence(self, refined_content, output):
    """
    Fall back to legacy confidence gating if ignition is not configured.

    Emit a deprecation warning directing users to migrate to ignition.
    """
    if self.config.use_confidence_gating and not hasattr(self, 'ignition_gate'):
        import warnings
        warnings.warn(
            "use_confidence_gating without IgnitionConfig is deprecated. "
            "Migrate to ignition dynamics for gated broadcast. "
            "See references/ignition-dynamics.md for migration guide.",
            DeprecationWarning,
            stacklevel=2,
        )
        confidence = self.confidence_estimator(refined_content)
        output['workspace'] = refined_content * confidence
        output['confidence'] = confidence
    return output
```

### 9.3 From Current Early Stopping

**Current implementation** (in `IterativeCompetition.forward`, lines 682-684):

```python
# Early stopping on strong ignition
if ignition.mean() > self.ignition_threshold * 1.5:
    break
```

**Target**: replace with structured ignition score. Early stopping should
check the interpretable ignition score, not the raw MLP output.

```python
# Target: early stopping based on structured ignition
ignition_score, _ = self.ignition_scorer(stability, confidence, margin, coherence)
if ignition_score.mean() > self.config.ignition_threshold * 1.5:
    break
```

### 9.4 Migration Checklist

Use this checklist to verify complete migration:

- [ ] Add `IgnitionConfig` to `brain_ai/config.py`
- [ ] Add `IgnitionConfig` field to `WorkspaceConfig` or `SelectionBroadcastConfig`
- [ ] Implement `InterpretableIgnition` module
- [ ] Implement `LearnedIgnition` module
- [ ] Implement `IgnitionGate` module
- [ ] Implement `SlotDropout` module
- [ ] Implement `WinnerDecay` class
- [ ] Implement `IgnitionCooldown` class
- [ ] Implement `compute_novelty` function
- [ ] Implement component signal extraction in `IterativeCompetition.forward`
- [ ] Replace `self.ignition_detector` with `self.ignition_scorer`
- [ ] Replace `self.confidence_estimator` with ignition gate
- [ ] Update `SelectionBroadcastWorkspace.forward` to use ignition gate
- [ ] Update return dict to include ignition telemetry
- [ ] Add fp32 safety to ignition computation
- [ ] Add unit tests for each component signal
- [ ] Add unit tests for ignition gate (training vs inference behavior)
- [ ] Add unit tests for lock-in prevention mechanisms
- [ ] Update `return_details=True` output format
- [ ] Verify backward compatibility with `use_learned_ignition=True`

---

## 10. Anti-Patterns

### 10.1 Using Only a Learned MLP for Ignition

**Problem:** A learned MLP that takes raw features and outputs a scalar
provides no insight into why ignition fires or fails. When the workspace
misbehaves, there is no way to diagnose the cause.

**Solution:** Always compute the four interpretable component signals
(stability, confidence, margin, coherence). Use `InterpretableIgnition` as
default. If using `LearnedIgnition`, still log the raw component signals.

```python
# BAD: opaque MLP
ignition = self.ignition_detector(features.mean(dim=1))

# GOOD: interpretable components
ignition, components = self.ignition_scorer(
    stability, confidence, margin, coherence
)
# components are logged regardless of scorer type
```

### 10.2 Hardcoding Ignition Threshold

**Problem:** Embedding the threshold value directly in the forward pass
prevents configuration and tuning.

**Solution:** Always read from `IgnitionConfig.ignition_threshold`.

```python
# BAD
if ignition > 0.3:
    ...

# GOOD
if ignition >= self.config.ignition_threshold:
    ...
```

### 10.3 No Lock-In Prevention

**Problem:** Without novelty, winner decay, or slot dropout, the workspace
converges to the same coalition permanently. Training degenerates because
the workspace stops exploring.

**Solution:** Enable at least one lock-in prevention mechanism. Recommended
defaults: `winner_decay=0.1` and `slot_dropout=0.1`.

```python
# BAD: no lock-in prevention
winners = self.competition(features, saliences)

# GOOD: apply prevention before competition
adjusted_saliences = self.winner_decay.apply(saliences)
if self.training:
    features = self.slot_dropout(features)
winners = self.competition(features, adjusted_saliences)
```

### 10.4 Igniting on Every Timestep

**Problem:** If the threshold is too low or cooldown is disabled, the
workspace ignites on every timestep. This eliminates the distinction between
committed and weak broadcasts, defeating the purpose of ignition.

**Solution:** Monitor `ignition_rate` in telemetry. If consistently > 0.9,
raise the threshold or enable cooldown.

```python
# BAD: threshold so low that everything ignites
config = IgnitionConfig(ignition_threshold=0.01, cooldown_steps=0)

# GOOD: reasonable threshold with cooldown
config = IgnitionConfig(ignition_threshold=0.3, cooldown_steps=2)
```

### 10.5 Using fp16 for Threshold Comparison

**Problem:** fp16 has insufficient precision for threshold comparisons near
typical ignition values (0.2 - 0.5). Scores that should be below threshold
may compare as equal or above.

**Solution:** Always perform ignition computation in fp32. See Section 8.

```python
# BAD: ignition comparison in fp16 under autocast
with torch.cuda.amp.autocast():
    ignited = (ignition_score >= threshold)  # fp16 comparison

# GOOD: force fp32 for ignition
with torch.cuda.amp.autocast(enabled=False):
    ignited = (ignition_score.float() >= threshold)
```

### 10.6 Not Logging Ignition Components

**Problem:** Without logging component scores, there is no way to diagnose
why ignition fires or fails. When the workspace misbehaves, debugging
becomes guesswork.

**Solution:** Always include component scores in telemetry (Section 7).
Log them at every training step.

```python
# BAD: only return ignition boolean
output['ignited'] = (ignition_score >= threshold)

# GOOD: return full telemetry
output['ignition_score'] = ignition_score
output['ignited'] = (ignition_score >= threshold)
output['effective_gain'] = effective_gain
output['ignition_components'] = components
```

### 10.7 Slot Dropout During Inference

**Problem:** Applying slot dropout during inference introduces
non-determinism and reduces output quality. Dropping slots at inference time
is never correct.

**Solution:** Guard slot dropout with `self.training`.

```python
# BAD: dropout applied unconditionally
slots = self.slot_dropout(slots)

# GOOD: dropout only during training
if self.training:
    slots = self.slot_dropout(slots)

# BEST: use nn.Module (SlotDropout) which checks self.training internally
slots = self.slot_dropout(slots)  # SlotDropout.forward checks self.training
```

### 10.8 Treating Ignition as Optional

**Problem:** Ignition is not a nice-to-have feature. It is the core
mechanism that distinguishes committed from tentative workspace states.
Without it, the workspace broadcasts everything at full strength, which
defeats the purpose of the competition.

**Solution:** Always configure ignition when using `SelectionBroadcastWorkspace`.
The only acceptable reason to disable ignition gating is during initial
debugging of the competition mechanism itself.

```python
# BAD: workspace without ignition
workspace = SelectionBroadcastWorkspace(config=config)
# config has no ignition settings -> broadcasts at full strength always

# GOOD: workspace with ignition configured
config.ignition = IgnitionConfig(
    ignition_threshold=0.3,
    weak_gain=0.3,
)
workspace = SelectionBroadcastWorkspace(config=config)
```

### 10.9 Ignoring Cooldown Interaction with Batch Statistics

**Problem:** `IgnitionCooldown` tracks a scalar `steps_since_ignition`.
When processing batches where some elements ignite and others do not, the
scalar cooldown state is ambiguous.

**Solution:** Track cooldown per batch element, or use the batch-mean ignition
rate to decide cooldown. For simplicity, trigger cooldown if any element in
the batch ignites.

```python
# BAD: cooldown uses single scalar for whole batch
self.cooldown.record_ignition(ignited.any().item())

# BETTER: track per-element cooldown (more complex)
# Or: use mean ignition rate as threshold
mean_ignited = ignited.float().mean().item()
self.cooldown.record_ignition(mean_ignited > 0.5)
```

### 10.10 Forgetting to Reset State Between Sequences

**Problem:** `WinnerDecay` and `IgnitionCooldown` carry state across
timesteps. If not reset between sequences (e.g., different episodes, new
documents), the state from one sequence contaminates the next.

**Solution:** Call `reset()` on all stateful lock-in prevention components
when starting a new sequence.

```python
def reset_state(self):
    """Reset all workspace state for a new sequence."""
    self.working_memory.reset_state()
    self.prev_context = None
    self.winner_decay.reset()
    self.cooldown.reset()
```

---

## Appendix A: Full IgnitionGate Module with Lock-In Prevention

This appendix provides a complete, integrated implementation combining all
the mechanisms described in Sections 3-5.

```python
class IntegratedIgnitionSystem(nn.Module):
    """
    Complete ignition system combining score computation, gating,
    and lock-in prevention.

    Integrates:
    - Interpretable or learned ignition scoring (Section 3)
    - Smooth/hard ignition gate (Section 4)
    - Novelty term, slot dropout, winner decay, cooldown (Section 5)

    Usage:
        system = IntegratedIgnitionSystem(config)
        gated_slots, telemetry = system(
            slots=winners,
            saliences=competition_saliences,
            round_history=competition_history,
            memory_buffer=working_memory_buffer,
            modality_ids=slot_modality_ids,
            num_modalities=len(active_modalities),
        )
    """

    def __init__(self, config: IgnitionConfig, num_winners: int = 7):
        super().__init__()
        self.config = config
        self.num_winners = num_winners

        # Ignition scorer
        self.scorer = create_ignition_scorer(config)

        # Ignition gate
        self.gate = IgnitionGate(
            threshold=config.ignition_threshold,
            weak_gain=config.weak_gain,
            temperature=config.gate_temperature,
        )

        # Lock-in prevention
        self.slot_dropout = SlotDropout(p=config.slot_dropout)
        self.winner_decay = WinnerDecay(decay_factor=config.winner_decay)
        self.cooldown = IgnitionCooldown(
            cooldown_steps=config.cooldown_steps,
            cooldown_boost=config.cooldown_boost,
        )

    def forward(
        self,
        slots: torch.Tensor,                      # (B, K, D)
        saliences: torch.Tensor,                   # (B, num_items)
        round_history: list[dict],
        memory_buffer: torch.Tensor | None = None,
        modality_ids: torch.Tensor | None = None,
        num_modalities: int = 1,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute ignition, apply gate, return telemetry.

        Returns:
            gated_slots: Slots modulated by ignition gain.
            telemetry: Full ignition telemetry dict.
        """
        # Apply slot dropout (training only)
        slots = self.slot_dropout(slots)

        # Compute component signals
        stability = compute_stability(round_history)
        confidence = compute_confidence(saliences, self.num_winners)
        margin = compute_margin(saliences, self.num_winners)

        if modality_ids is not None:
            coherence = compute_cross_modal_coherence(
                slots, modality_ids, num_modalities
            )
        else:
            coherence = torch.zeros(
                slots.shape[0], device=slots.device
            )

        # Compute ignition score
        with torch.cuda.amp.autocast(enabled=False):
            ignition_score, components = self.scorer(
                stability.float(),
                confidence.float(),
                margin.float(),
                coherence.float(),
            )

            # Apply cooldown to threshold
            effective_threshold = self.cooldown.get_effective_threshold(
                self.config.ignition_threshold
            )

            # Temporarily override gate threshold
            self.gate.threshold = effective_threshold

            # Apply gate
            gated_slots, effective_gain = self.gate(
                slots.float(), ignition_score
            )

        # Cast back to slot dtype
        gated_slots = gated_slots.to(slots.dtype)

        # Update stateful components
        ignited = (ignition_score >= effective_threshold).any().item()
        self.cooldown.record_ignition(ignited)

        attention_proxy = saliences.softmax(dim=-1)
        self.winner_decay.update(attention_proxy)

        # Build telemetry
        telemetry = build_ignition_telemetry(
            ignition_score=ignition_score,
            effective_gain=effective_gain,
            components=components,
            threshold=self.config.ignition_threshold,
            cooldown=self.cooldown,
            winner_decay=self.winner_decay,
        )

        return gated_slots, telemetry

    def reset_state(self):
        """Reset all stateful components."""
        self.winner_decay.reset()
        self.cooldown.reset()
```

---

## Appendix B: Relationship to Other Reference Documents

This document focuses on ignition dynamics. Related subsystems are covered in
other reference documents within this skill:

- **competition-broadcast.md** -- covers the iterative competition mechanism,
  salience scoring, winner-take-most dynamics, and broadcast refinement.
  Ignition score computation (this document) depends on signals produced
  during competition (that document).

- **working-memory.md** -- covers the working memory buffer, CfC/LTC/GRU
  backends, and temporal context maintenance. Ignition gate modulates the
  write weight into working memory (this document, Section 4.6).

- **competition-broadcast.md** -- covers multi-head attention, modality projection,
  and the cross-modal coherence signal that feeds into ignition (see Section 3.1.4
  in this document for how coherence is computed).

---

## Appendix C: Key Source Files

| File                                         | Relevant Content                                    |
|----------------------------------------------|-----------------------------------------------------|
| `brain_ai/workspace/global_workspace.py`     | `IterativeCompetition`, `SelectionBroadcastWorkspace` |
| `brain_ai/workspace/working_memory.py`       | `WorkingMemory`, `create_working_memory`             |
| `brain_ai/config.py`                         | `WorkspaceConfig`, `SelectionBroadcastConfig`         |
| `brain_ai/system.py`                         | `BrainAI` orchestrator, workspace integration        |
